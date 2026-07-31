# kv-manager

A paged KV cache manager for LLM inference engines, following the
PagedAttention design (vLLM).

## The problem: fragmentation

Every generated token appends one K and one V vector per layer and KV head to
its sequence's cache. The naive layout gives each sequence one contiguous
region sized for its maximum possible length. That fails in two ways:

- **Internal waste** — output lengths are unpredictable, so regions are
  sized for the worst case and most of the reservation is never used.
- **External fragmentation** — sequences finish at different times, leaving
  variable-sized holes between live regions. A new request can be refused
  even though the total free memory is more than enough, because no single
  hole is large enough.

## The design: paging

`kv-manager` treats the cache region like an OS treats physical RAM:

```
                 logical view (per sequence)          physical pool
                                                     ┌───────────┐
  seq 42: [tok 0..15][tok 16..31][tok 32..            │ block 0   │◄─┐
              │           │          │                │ block 1   │  │
              ▼           ▼          ▼                │ block 2   │◄─┼── seq 42's
        block table: [ 7,          2,        9 ]      │   ...     │  │   block table
                                                     │ block 9   │◄─┘
                                                     └───────────┘
```

- The region is split into fixed-size **blocks**, each holding the K/V data
  of `block_size` tokens for **all** layers and heads.
- A sequence owns a **block table**: an ordered list of block ids. Logically
  contiguous, physically scattered.
- Blocks are allocated on demand as a sequence grows and all returned when
  it finishes.

Because all blocks are interchangeable, any freed block can back any future
request — external fragmentation is impossible by construction. The only
waste left is the unused tail of each sequence's *last* block, bounded by
`block_size - 1` tokens per live sequence.

## Memory layout

One block holds, per token slot, K and V vectors for every layer:

```
token_bytes = 2 (K and V) * num_layers * num_kv_heads * head_dim * dtype_size
block_bytes = block_size * token_bytes
```

Each block is an independent device allocation (a *page*), so blocks have no
byte offsets relative to each other — a block id maps to a page via
`KvCacheManager::block_memory`. Token position `p` of a sequence lives in
slot `p % block_size` of block `block_table[p / block_size]`
(`KvCacheManager::slot_for_token`). How K/V/layer/head data is arranged
*inside* a block is the attention kernel's contract, not this crate's.

## The pool is real, paged GPU memory

The manager owns the physical cache as **one `cudarc::driver::CudaSlice<u8>`
per block**, each a separate, zero-initialized `block_bytes` allocation made
at construction. It is deliberately *not* one big slab:

- A single multi-GiB contiguous `cudaMalloc` can fail on a device whose
  address space is fragmented by earlier allocations (weights, workspaces),
  even when enough total memory is free. Fixed-size pages are always
  satisfiable.
- Every device allocation the crate ever makes has the same size, and pages
  are recycled across sequences — never returned to the driver until the
  manager is dropped — so the cache itself causes no allocator churn or
  further fragmentation.

Constructors take the `CudaStream` the pages should live on:

- `KvCacheManager::new(&stream, config, num_blocks)` — explicit page count.
- `KvCacheManager::with_memory_budget(&stream, config, bytes)` — as many
  pages as fit in a byte budget.
- `KvCacheManager::with_gpu_memory(&stream, config, headroom_bytes)` — asks
  the driver for the device's *actual* free memory (`cuMemGetInfo`) and
  claims everything above the headroom. Call after weights are loaded; the
  headroom covers activations and workspace buffers.

Kernels reach the memory through `block_memory(id)` /
`block_memory_mut(id)` — the `CudaSlice<u8>` page of one block, e.g. for
staging K/V data with `memcpy_htod` or for building the per-sequence pointer
arrays a paged-attention kernel consumes.

Construction fails with `KvCacheError::Driver` if the device cannot provide
the memory. `free(seq)` only returns block ids to the free list — no GPU
work, no zeroing; the next owner overwrites the data.

## Interfaces

| Interface | Role |
|---|---|
| `CacheConfig` | Model shape: `block_size`, `num_layers`, `num_kv_heads`, `head_dim`, `dtype_size`; derives `token_bytes`/`block_bytes` |
| `KvCacheManager::with_gpu_memory(&stream, config, headroom)` | Sizes the pool from the device's real free memory and allocates the pages |
| `KvCacheManager::with_memory_budget(&stream, config, bytes)` | Allocates as many pages as fit in an explicit byte budget |
| `KvCacheManager::new(&stream, config, num_blocks)` | Allocates an explicit number of pages |
| `block_memory(id)` / `block_memory_mut(id)` | The device page (`CudaSlice<u8>`) backing one block |
| `allocate(seq, num_tokens)` | Prefill: reserve `ceil(tokens / block_size)` blocks for a new query |
| `append(seq, n)` | Decode: extend a sequence; consumes last-block slack before allocating |
| `free(seq)` | Return all of a finished sequence's blocks to the pool |
| `can_allocate(tokens)` / `free_blocks()` | Admission control for the scheduler |
| `block_table(seq)` | The block list the paged-attention kernel consumes |
| `slot_for_token(seq, pos)` | Logical→physical translation: token position → (block, slot) |

Errors are all-or-nothing: a failed `allocate`/`append` changes nothing, so
the scheduler can queue the request and retry after other sequences finish.
`OutOfBlocks` is the only error expected during normal operation; the rest
signal engine bugs.

A typical engine loop:

```rust
use cudarc::driver::CudaContext;
use kv_manager::{CacheConfig, KvCacheManager};

let stream = CudaContext::new(0)?.default_stream();
let config = CacheConfig {
    block_size: 16,
    num_layers: 32,
    num_kv_heads: 8,
    head_dim: 128,
    dtype_size: 2,
};
// Claim all free GPU memory except 1 GiB of headroom for activations.
let mut cache = KvCacheManager::with_gpu_memory(&stream, config, 1 << 30)?;

// Admission: accept the request only if its prompt fits.
if cache.can_allocate(prompt_len) {
    cache.allocate(seq_id, prompt_len)?;      // prefill
}
loop {
    let table = cache.append(seq_id, 1)?;     // one decode step
    // pass table.blocks() to the paged-attention kernel ...
}
cache.free(seq_id)?;                          // sequence finished
```

## Internals

- `BlockAllocator` — a LIFO free-list over the fixed pool. `allocate` pops
  ids, `release` pushes them back; O(1) per block, plus an `in_use` bitmap
  that panics on double-free (a bug, not a runtime condition).
- `BlockTable` — `Vec<BlockId>` plus the token count; the gap between
  `num_tokens` and `blocks.len() * block_size` is the slack `append` fills
  before allocating.
- `KvCacheManager` — owns the device pages (`Vec<CudaSlice<u8>>`, one per
  block), the allocator, and the `SequenceId → BlockTable` map, and enforces
  the invariant that every block is either free or owned by exactly one
  sequence.

## Testing

`cargo test -p kv-manager` runs the unit tests (`src/tests.rs`). They
**require a CUDA GPU**, since every manager owns a real device allocation.
Coverage: block-size math, budget sizing, rounding, slack consumption in
`append`, all-or-nothing failure behavior, the fragmentation-reuse scenario,
slot lookups, and a full query lifecycle — plus GPU-specific tests that the
pool allocation actually reduces the device's reported free memory, that the
pool is zero-initialized, and that two sequences' blocks are physically
disjoint device memory (verified with byte-pattern roundtrips). The
crate-level doc example is also checked as a doctest.

## Out of scope (future work)

- **Prefix sharing / copy-on-write** — ref-counted blocks so sequences with
  a common prompt share cache until they diverge.
- **Swapping / eviction** — moving a preempted sequence's blocks to host
  memory and back.
- **Kernel-ready block tables** — uploading each sequence's block list as a
  device tensor for the paged-attention kernel.

//! # kv-manager — a paged KV cache for inference engines
//!
//! During autoregressive inference every generated token appends a key and a
//! value vector to the KV cache of its sequence, for every layer and head.
//! Reserving one contiguous region per sequence fragments GPU memory badly:
//! sequence lengths are unpredictable, so regions must be over-reserved, and
//! when sequences finish they leave variable-sized holes that later requests
//! cannot use.
//!
//! This crate manages the cache the way an OS manages virtual memory
//! (the PagedAttention design):
//!
//! * The GPU cache region is split into fixed-size **blocks**, each holding
//!   the K/V vectors of `block_size` tokens for all layers and heads.
//! * A sequence's cache is a **block table** — a list of block ids that is
//!   logically contiguous but physically scattered. Blocks are allocated on
//!   demand as the sequence grows.
//! * Because every block has the same size, any freed block can back any
//!   future allocation: external fragmentation cannot occur, and the only
//!   waste is the unused tail of each sequence's last block.
//!
//! The pool is *real, paged* GPU memory: at construction the manager
//! allocates one independent [`cudarc::driver::CudaSlice<u8>`] of
//! `block_bytes` per block — deliberately *not* one big slab, since a huge
//! contiguous `cudaMalloc` can fail on a fragmented device even when enough
//! total memory is free, while fixed-size pages are always satisfiable.
//! Pages are recycled across sequences and never returned to the driver
//! until the manager is dropped, so the cache causes no allocator churn.
//! ([`KvCacheManager::with_gpu_memory`] first queries the device's free
//! memory with `cuMemGetInfo` and claims everything above a headroom.)
//! Kernels get a block's device memory via [`KvCacheManager::block_memory`]
//! together with the per-sequence block tables.
//!
//! ## Example
//!
//! ```
//! use cudarc::driver::CudaContext;
//! use kv_manager::{CacheConfig, KvCacheManager};
//!
//! let ctx = CudaContext::new(0).unwrap();
//! let stream = ctx.default_stream();
//!
//! // A small model: 4 layers, 8 KV heads of dim 64, f16, 16 tokens/block.
//! let config = CacheConfig {
//!     block_size: 16,
//!     num_layers: 4,
//!     num_kv_heads: 8,
//!     head_dim: 64,
//!     dtype_size: 2,
//! };
//!
//! // Allocates 64 MiB of device memory as fixed-size pages, one per block.
//! let mut cache =
//!     KvCacheManager::with_memory_budget(&stream, config, 64 * 1024 * 1024).unwrap();
//!
//! // Prefill: a query arrives with a 100-token prompt.
//! let table = cache.allocate(42, 100).unwrap();
//! assert_eq!(table.blocks().len(), 7); // ceil(100 / 16)
//!
//! // Decode: one token per step. Block 7 has 12 free slots, so no new
//! // block is needed until those fill up.
//! let table = cache.append(42, 1).unwrap();
//! assert_eq!(table.blocks().len(), 7);
//! let last_block = table.blocks()[6];
//!
//! // Which block/slot does token 100 live in? (for the attention kernel)
//! assert_eq!(cache.slot_for_token(42, 100), Some((last_block, 4)));
//!
//! // Each block is a real, independently allocated page of device memory.
//! assert_eq!(cache.block_memory(last_block).len(), config.block_bytes());
//!
//! // The sequence finished: all of its blocks return to the pool.
//! cache.free(42).unwrap();
//! ```
//!
//! See `readme.md` for the full design document.

mod allocator;
mod config;
mod manager;

pub use config::CacheConfig;
pub use manager::{BlockId, BlockTable, KvCacheError, KvCacheManager, SequenceId};

#[cfg(test)]
mod tests;

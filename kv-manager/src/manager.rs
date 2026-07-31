use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DriverError};

use crate::allocator::BlockAllocator;
use crate::config::CacheConfig;

/// Identifies a running query/sequence. Assigned by the engine.
pub type SequenceId = u64;

/// Index of a physical cache block in the pool (0..num_blocks).
pub type BlockId = usize;

/// Errors reported by the cache manager. Out-of-blocks is the one callers
/// are expected to handle at runtime (by queuing or preempting requests);
/// the others indicate misuse. `Driver` surfaces CUDA failures (e.g. the
/// pool allocation itself running out of device memory).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheError {
    InvalidConfig(&'static str),
    /// Not enough free blocks. The operation had no effect.
    OutOfBlocks {
        requested: usize,
        available: usize,
    },
    SequenceExists(SequenceId),
    SequenceNotFound(SequenceId),
    /// A CUDA driver call failed.
    Driver(DriverError),
}

impl From<DriverError> for KvCacheError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid cache config: {}", msg),
            Self::OutOfBlocks {
                requested,
                available,
            } => write!(
                f,
                "out of cache blocks: requested {}, only {} available",
                requested, available
            ),
            Self::SequenceExists(id) => write!(f, "sequence {} already has an allocation", id),
            Self::SequenceNotFound(id) => write!(f, "sequence {} has no allocation", id),
            Self::Driver(err) => write!(f, "CUDA driver error: {}", err),
        }
    }
}

impl std::error::Error for KvCacheError {}

/// The per-sequence page table: which physical blocks hold the sequence's
/// tokens, in logical order.
///
/// Token position `p` lives in slot `p % block_size` of block
/// `blocks()[p / block_size]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    num_tokens: usize,
}

impl BlockTable {
    /// Physical block ids backing the sequence, in logical order.
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Tokens currently cached for the sequence.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Tokens that fit in the allocated blocks; the difference to
    /// `num_tokens` is the slack in the last block.
    #[cfg(test)]
    fn capacity(&self, block_size: usize) -> usize {
        self.blocks.len() * block_size
    }
}

/// The paged KV cache manager. See the crate docs for the design.
///
/// Owns the physical block pool as *paged* device memory: one independent,
/// fixed-size [`CudaSlice<u8>`] per block, allocated at construction. There
/// is deliberately no single large allocation — a multi-GiB contiguous
/// `cudaMalloc` can fail on a fragmented device even when enough total
/// memory is free, while `block_bytes`-sized pages are always satisfiable.
///
/// Also owns one [`BlockTable`] per live sequence. All bookkeeping
/// operations are all-or-nothing: on error, no state changed.
#[derive(Debug)]
pub struct KvCacheManager {
    config: CacheConfig,
    /// The stream the pages were allocated on; kept so the memory is
    /// released on the right device when the manager is dropped.
    stream: Arc<CudaStream>,
    /// The physical pages. `pages[b]` is the device memory of [`BlockId`]
    /// `b`: a separate `block_bytes`-sized allocation.
    pages: Vec<CudaSlice<u8>>,
    allocator: BlockAllocator,
    tables: HashMap<SequenceId, BlockTable>,
}

impl KvCacheManager {
    /// Allocates `num_blocks` pages on the GPU (each a separate,
    /// zero-initialized `block_bytes` allocation) and creates a manager
    /// over them.
    pub fn new(
        stream: &Arc<CudaStream>,
        config: CacheConfig,
        num_blocks: usize,
    ) -> Result<Self, KvCacheError> {
        config.validate()?;
        if num_blocks == 0 {
            return Err(KvCacheError::InvalidConfig("num_blocks must be non-zero"));
        }
        // The real GPU allocations, one per block. Fails with a Driver
        // error once the device runs out of memory; pages allocated up to
        // that point are freed again when the Vec is dropped.
        let pages = (0..num_blocks)
            .map(|_| stream.alloc_zeros::<u8>(config.block_bytes()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            config,
            stream: stream.clone(),
            pages,
            allocator: BlockAllocator::new(num_blocks),
            tables: HashMap::new(),
        })
    }

    /// Allocates as many blocks as fit in `budget_bytes` of GPU memory.
    pub fn with_memory_budget(
        stream: &Arc<CudaStream>,
        config: CacheConfig,
        budget_bytes: usize,
    ) -> Result<Self, KvCacheError> {
        config.validate()?;
        let num_blocks = budget_bytes / config.block_bytes();
        if num_blocks == 0 {
            return Err(KvCacheError::InvalidConfig(
                "memory budget smaller than one block",
            ));
        }
        Self::new(stream, config, num_blocks)
    }

    /// Queries the device's *actual* free memory (`cuMemGetInfo`) and claims
    /// everything above `headroom_bytes` for the cache.
    ///
    /// Call this after model weights are loaded; keep enough headroom for
    /// activations and workspace buffers.
    pub fn with_gpu_memory(
        stream: &Arc<CudaStream>,
        config: CacheConfig,
        headroom_bytes: usize,
    ) -> Result<Self, KvCacheError> {
        config.validate()?;
        let (free_bytes, _total_bytes) = stream.context().mem_get_info()?;
        let budget = free_bytes.saturating_sub(headroom_bytes);
        Self::with_memory_budget(stream, config, budget)
    }

    // ---- Capacity queries -------------------------------------------------

    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// The stream the pool lives on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn total_blocks(&self) -> usize {
        self.allocator.total_blocks()
    }

    pub fn free_blocks(&self) -> usize {
        self.allocator.free_blocks()
    }

    pub fn used_blocks(&self) -> usize {
        self.total_blocks() - self.free_blocks()
    }

    /// Blocks needed to cache `num_tokens` tokens (`ceil(tokens / block_size)`).
    pub fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.config.block_size)
    }

    /// Whether a *new* sequence of `num_tokens` tokens would fit right now.
    /// Admission control uses this before accepting a request.
    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        self.blocks_for_tokens(num_tokens) <= self.free_blocks()
    }

    // ---- Allocation lifecycle ---------------------------------------------

    /// Allocates the cache for a new sequence's prompt (prefill).
    ///
    /// Reserves `ceil(num_tokens / block_size)` blocks and returns the block
    /// table. A `num_tokens` of zero is valid and reserves nothing.
    pub fn allocate(
        &mut self,
        seq: SequenceId,
        num_tokens: usize,
    ) -> Result<&BlockTable, KvCacheError> {
        if self.tables.contains_key(&seq) {
            return Err(KvCacheError::SequenceExists(seq));
        }
        let needed = self.blocks_for_tokens(num_tokens);
        let blocks = self
            .allocator
            .allocate(needed)
            .ok_or(KvCacheError::OutOfBlocks {
                requested: needed,
                available: self.allocator.free_blocks(),
            })?;
        self.tables.insert(seq, BlockTable { blocks, num_tokens });
        Ok(&self.tables[&seq])
    }

    /// Extends a sequence by `num_new_tokens` (decode steps).
    ///
    /// Fills the slack in the last block first and only allocates blocks for
    /// what does not fit, so appending a single token allocates a block just
    /// once every `block_size` steps.
    pub fn append(
        &mut self,
        seq: SequenceId,
        num_new_tokens: usize,
    ) -> Result<&BlockTable, KvCacheError> {
        let block_size = self.config.block_size;
        let table = self
            .tables
            .get(&seq)
            .ok_or(KvCacheError::SequenceNotFound(seq))?;

        let new_total = table.num_tokens + num_new_tokens;
        let needed = new_total
            .div_ceil(block_size)
            .saturating_sub(table.blocks.len());

        let new_blocks = self
            .allocator
            .allocate(needed)
            .ok_or(KvCacheError::OutOfBlocks {
                requested: needed,
                available: self.allocator.free_blocks(),
            })?;

        let table = self.tables.get_mut(&seq).unwrap();
        table.blocks.extend(new_blocks);
        table.num_tokens = new_total;
        Ok(&self.tables[&seq])
    }

    /// Releases every block of a finished sequence back to the pool and
    /// returns how many were freed.
    ///
    /// The pages themselves stay allocated (they are recycled for future
    /// sequences, never handed back to the driver) and are not zeroed — the
    /// next owner overwrites them — so freeing costs no GPU work and causes
    /// no allocator churn.
    pub fn free(&mut self, seq: SequenceId) -> Result<usize, KvCacheError> {
        let table = self
            .tables
            .remove(&seq)
            .ok_or(KvCacheError::SequenceNotFound(seq))?;
        self.allocator.release(&table.blocks);
        Ok(table.blocks.len())
    }

    // ---- Device memory access ----------------------------------------------

    /// The device page of one block. Each block is its own fixed-size
    /// allocation — there is no larger buffer it is a view into. Panics if
    /// `block >= total_blocks`.
    pub fn block_memory(&self, block: BlockId) -> &CudaSlice<u8> {
        &self.pages[block]
    }

    /// Mutable variant of [`Self::block_memory`], for writing K/V data into
    /// a block (e.g. with `CudaStream::memcpy_htod`).
    pub fn block_memory_mut(&mut self, block: BlockId) -> &mut CudaSlice<u8> {
        &mut self.pages[block]
    }

    // ---- Lookups for the attention kernel ---------------------------------

    pub fn block_table(&self, seq: SequenceId) -> Option<&BlockTable> {
        self.tables.get(&seq)
    }

    /// Physical location of a token: `(block id, slot within the block)`.
    /// Returns `None` for an unknown sequence or an uncached position.
    pub fn slot_for_token(&self, seq: SequenceId, position: usize) -> Option<(BlockId, usize)> {
        let table = self.tables.get(&seq)?;
        if position >= table.num_tokens {
            return None;
        }
        let block_size = self.config.block_size;
        Some((table.blocks[position / block_size], position % block_size))
    }

    /// Sanity invariant: blocks are either free or owned by exactly one
    /// sequence (block sums must match), and every block is backed by a
    /// page of exactly `block_bytes` device memory. Used by tests.
    #[cfg(test)]
    pub(crate) fn check_invariants(&self) {
        let owned: usize = self.tables.values().map(|t| t.blocks.len()).sum();
        assert_eq!(owned, self.used_blocks());
        for table in self.tables.values() {
            assert!(table.num_tokens <= table.capacity(self.config.block_size));
        }
        assert_eq!(self.pages.len(), self.total_blocks());
        for page in &self.pages {
            assert_eq!(page.len(), self.config.block_bytes());
        }
    }
}

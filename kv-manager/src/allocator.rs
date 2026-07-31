use crate::manager::BlockId;

/// Free-list allocator over a fixed pool of interchangeable blocks.
///
/// Every block has the same size, so allocation is popping an id off a stack
/// and freeing is pushing it back — O(1), and immune to external
/// fragmentation by construction: any freed block can satisfy any future
/// request.
#[derive(Debug)]
pub(crate) struct BlockAllocator {
    /// Ids currently available, used as a LIFO stack. LIFO keeps recently
    /// freed (likely still cache-warm) blocks in circulation.
    free: Vec<BlockId>,
    /// `in_use[id]` guards against double-free bugs in the layer above.
    in_use: Vec<bool>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            // Reversed so block 0 is handed out first; not required, just
            // makes behavior predictable in tests.
            free: (0..num_blocks).rev().collect(),
            in_use: vec![false; num_blocks],
        }
    }

    pub fn total_blocks(&self) -> usize {
        self.in_use.len()
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Takes `count` blocks from the pool, all-or-nothing.
    pub fn allocate(&mut self, count: usize) -> Option<Vec<BlockId>> {
        if count > self.free.len() {
            return None;
        }
        let ids: Vec<BlockId> = (0..count).map(|_| self.free.pop().unwrap()).collect();
        for &id in &ids {
            self.in_use[id] = true;
        }
        Some(ids)
    }

    /// Returns blocks to the pool.
    ///
    /// # Panics
    ///
    /// Panics on double-free or an out-of-range id — both indicate a bug in
    /// the manager, not a recoverable runtime condition.
    pub fn release(&mut self, ids: &[BlockId]) {
        for &id in ids {
            assert!(self.in_use[id], "double free of block {}", id);
            self.in_use[id] = false;
            self.free.push(id);
        }
    }
}

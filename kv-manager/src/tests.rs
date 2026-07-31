use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use crate::{CacheConfig, KvCacheError, KvCacheManager};

/// 2 layers * 2 KV heads * dim 4 * f16 * (K+V) = 64 bytes/token,
/// 4 tokens/block = 256 bytes/block. Small numbers, easy to reason about.
fn test_config() -> CacheConfig {
    CacheConfig {
        block_size: 4,
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 4,
        dtype_size: 2,
    }
}

fn stream() -> Arc<CudaStream> {
    CudaContext::new(0)
        .expect("these tests require a CUDA GPU")
        .default_stream()
}

fn manager(num_blocks: usize) -> KvCacheManager {
    KvCacheManager::new(&stream(), test_config(), num_blocks).unwrap()
}

#[test]
fn block_size_math() {
    let config = test_config();
    assert_eq!(config.token_bytes(), 64);
    assert_eq!(config.block_bytes(), 256);
}

#[test]
fn memory_budget_determines_block_count() {
    let stream = stream();

    // 10.5 blocks worth of memory -> 10 blocks; the remainder is unusable.
    let cache = KvCacheManager::with_memory_budget(&stream, test_config(), 256 * 10 + 128).unwrap();
    assert_eq!(cache.total_blocks(), 10);
    assert_eq!(cache.free_blocks(), 10);
    // Every block is backed by its own block_bytes-sized device page.
    for block in 0..cache.total_blocks() {
        assert_eq!(cache.block_memory(block).len(), 256);
    }

    // Budget below one block is a config error.
    assert!(matches!(
        KvCacheManager::with_memory_budget(&stream, test_config(), 255),
        Err(KvCacheError::InvalidConfig(_))
    ));
}

#[test]
fn invalid_config_is_rejected() {
    let mut config = test_config();
    config.block_size = 0;
    assert!(matches!(
        KvCacheManager::new(&stream(), config, 8),
        Err(KvCacheError::InvalidConfig(_))
    ));
}

#[test]
fn pool_allocation_consumes_real_gpu_memory() {
    let stream = stream();
    let ctx = stream.context().clone();
    let (free_before, _) = ctx.mem_get_info().unwrap();

    // A model-sized config (512 KiB pages) so 64 MiB is 128 pages and the
    // drop in the device's reported free memory is unambiguous.
    let config = CacheConfig {
        block_size: 16,
        num_layers: 8,
        num_kv_heads: 8,
        head_dim: 128,
        dtype_size: 2,
    };
    let budget = 64 * 1024 * 1024;
    let cache = KvCacheManager::with_memory_budget(&stream, config, budget).unwrap();
    assert_eq!(cache.total_blocks(), 128);

    let (free_after, _) = ctx.mem_get_info().unwrap();
    // Other tests allocate/free small pools concurrently, hence the slack.
    assert!(
        free_before.saturating_sub(free_after) >= budget / 2,
        "free memory only dropped {} bytes for a {} byte pool",
        free_before.saturating_sub(free_after),
        budget
    );
}

#[test]
fn with_gpu_memory_claims_free_memory_minus_headroom() {
    let stream = stream();
    let (free, _) = stream.context().mem_get_info().unwrap();

    // Leave all but ~1 MiB as headroom -> a pool of roughly 1 MiB.
    let target = 1024 * 1024;
    let cache = KvCacheManager::with_gpu_memory(&stream, test_config(), free - target).unwrap();
    assert!(cache.total_blocks() >= 1);
    assert!(cache.total_blocks() * cache.config().block_bytes() <= 2 * target);

    // Headroom larger than free memory leaves nothing for the cache.
    assert!(matches!(
        KvCacheManager::with_gpu_memory(&stream, test_config(), usize::MAX),
        Err(KvCacheError::InvalidConfig(_))
    ));
}

#[test]
fn pages_are_zero_initialized() {
    let stream = stream();
    let cache = KvCacheManager::new(&stream, test_config(), 4).unwrap();
    for block in 0..cache.total_blocks() {
        let host = stream.clone_dtoh(cache.block_memory(block)).unwrap();
        assert_eq!(host.len(), 256);
        assert!(host.iter().all(|&b| b == 0));
    }
}

#[test]
fn blocks_are_disjoint_device_memory() {
    let stream = stream();
    let mut cache = KvCacheManager::new(&stream, test_config(), 8).unwrap();
    let block_bytes = cache.config().block_bytes();

    cache.allocate(1, 4).unwrap();
    cache.allocate(2, 4).unwrap();
    let a = cache.block_table(1).unwrap().blocks()[0];
    let b = cache.block_table(2).unwrap().blocks()[0];

    // Write a distinct byte pattern into each sequence's block on the GPU.
    let pattern_a = vec![0xAAu8; block_bytes];
    let pattern_b = vec![0x55u8; block_bytes];
    stream
        .memcpy_htod(&pattern_a, cache.block_memory_mut(a))
        .unwrap();
    stream
        .memcpy_htod(&pattern_b, cache.block_memory_mut(b))
        .unwrap();

    // Both read back intact: the blocks are real, physically disjoint pages.
    assert_eq!(stream.clone_dtoh(cache.block_memory(a)).unwrap(), pattern_a);
    assert_eq!(stream.clone_dtoh(cache.block_memory(b)).unwrap(), pattern_b);
    cache.check_invariants();
}

#[test]
fn allocate_rounds_up_to_blocks() {
    let mut cache = manager(8);

    // 9 tokens at 4 tokens/block -> 3 blocks.
    let table = cache.allocate(1, 9).unwrap();
    assert_eq!(table.blocks().len(), 3);
    assert_eq!(table.num_tokens(), 9);
    assert_eq!(cache.free_blocks(), 5);

    // An exact multiple has no slack block.
    let table = cache.allocate(2, 8).unwrap();
    assert_eq!(table.blocks().len(), 2);
    cache.check_invariants();
}

#[test]
fn allocate_zero_tokens_reserves_nothing() {
    let mut cache = manager(4);
    let table = cache.allocate(1, 0).unwrap();
    assert!(table.blocks().is_empty());
    assert_eq!(cache.free_blocks(), 4);
}

#[test]
fn sequences_get_disjoint_blocks() {
    let mut cache = manager(8);
    cache.allocate(1, 8).unwrap();
    cache.allocate(2, 8).unwrap();

    let a = cache.block_table(1).unwrap().blocks();
    let b = cache.block_table(2).unwrap().blocks();
    assert!(a.iter().all(|id| !b.contains(id)));
    cache.check_invariants();
}

#[test]
fn allocation_failure_leaves_state_unchanged() {
    let mut cache = manager(4);
    cache.allocate(1, 8).unwrap(); // 2 blocks used

    // 12 tokens need 3 blocks but only 2 are free.
    let err = cache.allocate(2, 12).unwrap_err();
    assert_eq!(
        err,
        KvCacheError::OutOfBlocks {
            requested: 3,
            available: 2
        }
    );

    // The failed call reserved nothing and registered no sequence.
    assert_eq!(cache.free_blocks(), 2);
    assert!(cache.block_table(2).is_none());
    cache.check_invariants();
}

#[test]
fn duplicate_sequence_is_rejected() {
    let mut cache = manager(4);
    cache.allocate(1, 4).unwrap();
    assert_eq!(
        cache.allocate(1, 4).unwrap_err(),
        KvCacheError::SequenceExists(1)
    );
}

#[test]
fn append_uses_last_block_slack_first() {
    let mut cache = manager(8);
    cache.allocate(1, 5).unwrap(); // 2 blocks, 3 slots of slack
    assert_eq!(cache.free_blocks(), 6);

    // Three decode steps fit into the slack: no new block.
    for expected_tokens in 6..=8 {
        let table = cache.append(1, 1).unwrap();
        assert_eq!(table.num_tokens(), expected_tokens);
        assert_eq!(table.blocks().len(), 2);
    }

    // The 9th token crosses the block boundary: exactly one new block.
    let table = cache.append(1, 1).unwrap();
    assert_eq!(table.blocks().len(), 3);
    assert_eq!(cache.free_blocks(), 5);
    cache.check_invariants();
}

#[test]
fn append_many_tokens_at_once() {
    let mut cache = manager(8);
    cache.allocate(1, 2).unwrap(); // 1 block, 2 slack
    // 10 more tokens -> total 12 -> 3 blocks -> 2 new ones.
    let table = cache.append(1, 10).unwrap();
    assert_eq!(table.blocks().len(), 3);
    assert_eq!(table.num_tokens(), 12);
    cache.check_invariants();
}

#[test]
fn append_failure_leaves_state_unchanged() {
    let mut cache = manager(2);
    cache.allocate(1, 4).unwrap(); // 1 block used, 1 free

    // The last block is full, so 9 more tokens need ceil(9/4) = 3 blocks.
    let err = cache.append(1, 9).unwrap_err();
    assert_eq!(
        err,
        KvCacheError::OutOfBlocks {
            requested: 3,
            available: 1
        }
    );

    let table = cache.block_table(1).unwrap();
    assert_eq!(table.num_tokens(), 4);
    assert_eq!(table.blocks().len(), 1);
    cache.check_invariants();
}

#[test]
fn append_to_unknown_sequence_fails() {
    let mut cache = manager(2);
    assert_eq!(
        cache.append(7, 1).unwrap_err(),
        KvCacheError::SequenceNotFound(7)
    );
    assert_eq!(
        cache.free(7).unwrap_err(),
        KvCacheError::SequenceNotFound(7)
    );
}

#[test]
fn freed_blocks_are_reusable_no_fragmentation() {
    // The scenario that fragments a contiguous allocator: A and B interleave
    // in memory, A is freed, and a new request wants A's total size. With
    // fixed-size blocks the freed blocks satisfy any new request.
    let mut cache = manager(6);
    cache.allocate(1, 12).unwrap(); // A: 3 blocks
    cache.allocate(2, 12).unwrap(); // B: 3 blocks
    assert_eq!(cache.free_blocks(), 0);

    assert_eq!(cache.free(1).unwrap(), 3);
    assert_eq!(cache.free_blocks(), 3);

    // C fits exactly into the blocks A returned.
    let table = cache.allocate(3, 12).unwrap();
    assert_eq!(table.blocks().len(), 3);
    cache.check_invariants();
}

#[test]
fn admission_control_with_can_allocate() {
    let mut cache = manager(4);
    cache.allocate(1, 12).unwrap(); // 3 of 4 blocks
    assert!(cache.can_allocate(4)); // 1 block fits
    assert!(!cache.can_allocate(5)); // 2 blocks do not
}

#[test]
fn slot_lookup_matches_paging_rule() {
    let mut cache = manager(4);
    let blocks: Vec<_> = cache.allocate(1, 10).unwrap().blocks().to_vec();

    // position p -> (blocks[p / 4], p % 4)
    assert_eq!(cache.slot_for_token(1, 0), Some((blocks[0], 0)));
    assert_eq!(cache.slot_for_token(1, 5), Some((blocks[1], 1)));
    assert_eq!(cache.slot_for_token(1, 9), Some((blocks[2], 1)));

    // Uncached position and unknown sequence.
    assert_eq!(cache.slot_for_token(1, 10), None);
    assert_eq!(cache.slot_for_token(9, 0), None);
}

#[test]
fn every_block_has_its_own_page() {
    let cache = manager(4);
    for block in 0..4 {
        assert_eq!(cache.block_memory(block).len(), 256);
    }
}

#[test]
fn full_query_lifecycle() {
    // Prefill, decode to completion, free — with a second query admitted
    // as soon as memory allows.
    let mut cache = manager(4);

    cache.allocate(1, 13).unwrap(); // 4 blocks: pool exhausted
    assert!(!cache.can_allocate(1));

    // Query 2 must wait; query 1 finishes and frees its memory.
    assert!(matches!(
        cache.allocate(2, 4),
        Err(KvCacheError::OutOfBlocks { .. })
    ));
    cache.free(1).unwrap();

    // Now query 2 runs a full prefill + decode.
    cache.allocate(2, 4).unwrap();
    for _ in 0..8 {
        cache.append(2, 1).unwrap();
    }
    assert_eq!(cache.block_table(2).unwrap().num_tokens(), 12);
    assert_eq!(cache.used_blocks(), 3);
    cache.free(2).unwrap();
    assert_eq!(cache.free_blocks(), 4);
    cache.check_invariants();
}

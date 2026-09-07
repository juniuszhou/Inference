use crate::with_test_stack;
use cutile::half::f16;
use cutile::prelude::*;
use std::sync::Arc;
use tile::kernels;

/// Helper: copy f32 host values to device as f16 logits.
fn make_logits_f16(host_f32: &[f32]) -> Arc<Tensor<f16>> {
    let host_f16: Vec<f16> = host_f32.iter().map(|&v| f16::from_f32(v)).collect();
    api::copy_host_vec_to_device(&Arc::new(host_f16))
        .sync()
        .expect("logits h2d")
        .into()
}

/// Run argmax_blocks_f16 and return (block_max, block_idx) host vecs.
///
/// `backing` is the physical device length (must be num_blocks * block_size).
/// `len` is the logical length passed to the kernel (masking applies for idx >= len).
fn run_argmax(block_size: usize, backing: &[f32], len: usize) -> (Vec<f32>, Vec<u32>) {
    assert_eq!(
        backing.len() % block_size,
        0,
        "backing must be padded to BLOCK_SIZE"
    );
    let num_blocks = backing.len() / block_size;
    let logits = make_logits_f16(backing);
    let block_max = api::zeros::<f32>(&[num_blocks]).partition([1]);
    let block_idx = api::zeros::<u32>(&[num_blocks]).partition([1]);

    let (_, max_out, idx_out, _) =
        kernels::argmax_blocks_f16(logits, block_max, block_idx, len as i32)
            .generics(vec![block_size.to_string()])
            .sync()
            .expect("argmax_blocks_f16");

    let max_host = max_out
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("max to_host");
    let idx_host = idx_out
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("idx to_host");
    (max_host, idx_host)
}

#[test]
fn test_argmax_single_block() {
    with_test_stack(|| {
        // BLOCK_SIZE=32, len=32, single block. Peak at index 10.
        let block_size = 32usize;
        let mut backing = vec![-1.0f32; 32];
        backing[10] = 5.0;
        backing[20] = 2.0;

        let (max_host, idx_host) = run_argmax(block_size, &backing, 32);
        assert_eq!(max_host.len(), 1);
        assert_eq!(idx_host.len(), 1);
        assert!(
            (max_host[0] - 5.0).abs() < 1e-3,
            "expected max 5.0, got {}",
            max_host[0]
        );
        assert_eq!(idx_host[0], 10, "expected idx 10, got {}", idx_host[0]);
    });
}

#[test]
fn test_argmax_multi_block_monotonic() {
    with_test_stack(|| {
        // BLOCK_SIZE=32, len=128, 4 blocks, logits[i] = i.
        // Block b covers [b*32, b*32+32), max is the last element of each block.
        let block_size = 32usize;
        let backing: Vec<f32> = (0..128).map(|i| i as f32).collect();

        let (max_host, idx_host) = run_argmax(block_size, &backing, 128);
        assert_eq!(max_host.len(), 4);
        assert_eq!(idx_host.len(), 4);
        for b in 0..4 {
            let expected_max = (b * 32 + 31) as f32;
            let expected_idx = (b * 32 + 31) as u32;
            assert!(
                (max_host[b] - expected_max).abs() < 1e-3,
                "block {}: expected max {}, got {}",
                b,
                expected_max,
                max_host[b]
            );
            assert_eq!(
                idx_host[b], expected_idx,
                "block {}: expected idx {}, got {}",
                b, expected_idx, idx_host[b]
            );
        }
    });
}

#[test]
fn test_argmax_multi_block_arbitrary_peaks() {
    with_test_stack(|| {
        // Each block gets its own peak at a known offset.
        let block_size = 32usize;
        let mut backing = vec![0.0f32; 128];
        // block 0: peak 9.0 at global idx 5
        backing[5] = 9.0;
        // block 1: peak -1.0 at global idx 40 (all others -100)
        for i in 32..64 {
            backing[i] = -100.0;
        }
        backing[40] = -1.0;
        // block 2: peak 3.5 at global idx 95 (end of block)
        backing[95] = 3.5;
        // block 3: peak 7.25 at global idx 96 (start of block)
        backing[96] = 7.25;

        let (max_host, idx_host) = run_argmax(block_size, &backing, 128);
        assert!(
            (max_host[0] - 9.0).abs() < 1e-3,
            "block0 got {}",
            max_host[0]
        );
        assert_eq!(idx_host[0], 5);
        assert!(
            (max_host[1] - -1.0).abs() < 1e-3,
            "block1 got {}",
            max_host[1]
        );
        assert_eq!(idx_host[1], 40);
        assert!(
            (max_host[2] - 3.5).abs() < 1e-3,
            "block2 got {}",
            max_host[2]
        );
        assert_eq!(idx_host[2], 95);
        assert!(
            (max_host[3] - 7.25).abs() < 1e-3,
            "block3 got {}",
            max_host[3]
        );
        assert_eq!(idx_host[3], 96);
    });
}

#[test]
fn test_argmax_tail_masking_ignores_padding() {
    with_test_stack(|| {
        // BLOCK_SIZE=32, len=100 -> 4 blocks, backing padded to 128.
        // Fill padding (100..128) with huge values; they must be masked out.
        let block_size = 32usize;
        let len = 100usize;
        let mut backing = vec![0.0f32; 128];
        for i in 0..100 {
            backing[i] = i as f32 * 0.1;
        }
        // Peak within valid range of last block.
        backing[99] = 42.0;
        // Poison the padding: would win if masking were broken.
        for i in 100..128 {
            backing[i] = 9999.0;
        }

        let (max_host, idx_host) = run_argmax(block_size, &backing, len);
        assert_eq!(max_host.len(), 4);
        // Blocks 0-2 fully valid: last element of each block wins.
        // block2 covers 64..96: backing[95] = 9.5 is max there.
        assert!(
            (max_host[0] - 3.1).abs() < 1e-2,
            "block0 got {}",
            max_host[0]
        );
        assert_eq!(idx_host[0], 31);
        assert!(
            (max_host[1] - 6.3).abs() < 1e-2,
            "block1 got {}",
            max_host[1]
        );
        assert_eq!(idx_host[1], 63);
        assert!(
            (max_host[2] - 9.5).abs() < 1e-2,
            "block2 got {}",
            max_host[2]
        );
        assert_eq!(idx_host[2], 95);
        // Last block: only 96..100 valid, peak at 99.
        assert!(
            (max_host[3] - 42.0).abs() < 1e-3,
            "block3 expected 42.0, got {}",
            max_host[3]
        );
        assert_eq!(
            idx_host[3], 99,
            "padding must be masked, got {}",
            idx_host[3]
        );
    });
}

#[test]
fn test_argmax_tie_breaks_to_first_index() {
    with_test_stack(|| {
        // Duplicate maxima within a block -> smallest index wins (reduce_min).
        let block_size = 32usize;
        let mut backing = vec![-5.0f32; 64];
        backing[3] = 4.0;
        backing[7] = 4.0;
        backing[30] = 4.0;
        // Second block: all equal -> idx 32 (first of block).
        for i in 32..64 {
            backing[i] = 1.0;
        }

        let (max_host, idx_host) = run_argmax(block_size, &backing, 64);
        assert!(
            (max_host[0] - 4.0).abs() < 1e-3,
            "block0 got {}",
            max_host[0]
        );
        assert_eq!(
            idx_host[0], 3,
            "ties must pick first index, got {}",
            idx_host[0]
        );
        assert!(
            (max_host[1] - 1.0).abs() < 1e-3,
            "block1 got {}",
            max_host[1]
        );
        assert_eq!(idx_host[1], 32);
    });
}

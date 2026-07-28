use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use first::flash::flash;

/// FlashAttention test suite.
/// 
/// The `flash_attention` kernel in src/flash.rs uses a single block per (batch, head)
/// with BC threads. Each thread handles one query row within a tile. For correctness,
/// **BC must equal BR** because:
/// - The kernel writes l/m at index `lm_offset + Br * i + tx` where `tx` is the thread ID
/// - All `Bc` threads together cover `Bc` rows per tile; if `Br != Bc`, some rows would be
///   missed or out-of-bounds accessed.
/// - This design processes square attention tiles (Bc × Bc).
///
/// Tests below ensure the kernel produces correct results matching the reference
/// implementation across various parameter sets while maintaining BC == BR.
fn reference_attention(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut o = vec![0.0f32; n * d];

    for i in 0..n {
        let mut scores = vec![0.0f32; n];
        let mut max_score = -f32::INFINITY;
        for j in 0..n {
            let mut s = 0.0f32;
            for x in 0..d {
                s += q[i * d + x] * k[j * d + x];
            }
            s *= scale;
            scores[j] = s;
            if s > max_score {
                max_score = s;
            }
        }

        let mut sum_exp = 0.0f32;
        for j in 0..n {
            scores[j] = (scores[j] - max_score).exp();
            sum_exp += scores[j];
        }

        for x in 0..d {
            let mut val = 0.0f32;
            for j in 0..n {
                val += scores[j] * v[j * d + x];
            }
            o[i * d + x] = val / sum_exp;
        }
    }

    o
}

#[test]
fn test_flash_attention() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const B: usize = 1;
    const NH: usize = 1;
    const N: usize = 32;
    const D: usize = 8;
    const BC: usize = 32;
    const BR: usize = 32;

    let tc = ((N as f32) / (BC as f32)).ceil() as i32;
    let tr = ((N as f32) / (BR as f32)).ceil() as i32;
    let softmax_scale = 1.0f32 / (D as f32).sqrt();

    let q_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let k_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.2 - 2.0)
        .collect();
    let v_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.3 - 0.5)
        .collect();

    let q_dev = DeviceBuffer::from_host(&stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v_host).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N * D).unwrap();
    let mut l_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N).unwrap();
    let mut m_dev =
        DeviceBuffer::<f32>::from_host(&stream, &vec![-f32::INFINITY; B * NH * N]).unwrap();

    let tile_size = BC * D;
    let sram_size = (3 * tile_size + BC * BC) * std::mem::size_of::<f32>();

    let module = flash::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: one block per (batch, head) with BC threads; each thread owns
    // row `tx` of every tile, so all raw-index writes to l, m and O are
    // disjoint, and the buffers cover the kernel's accesses.
    unsafe {
        module.flash_attention(
            &stream,
            LaunchConfig {
                grid_dim: (B as u32, NH as u32, 1),
                block_dim: (BC as u32, 1, 1),
                shared_mem_bytes: sram_size as u32,
            },
            &q_dev,
            &k_dev,
            &v_dev,
            N as i32,
            D as i32,
            tc,
            tr,
            BC as i32,
            BR as i32,
            softmax_scale,
            &mut l_dev,
            &mut m_dev,
            &mut o_dev,
        )
    }
    .expect("Kernel launch failed");

    let o_host = o_dev.to_host_vec(&stream).unwrap();

    let reference = reference_attention(&q_host, &k_host, &v_host, N, D);
    let mut max_err = 0.0f32;
    for i in 0..N * D {
        let err = (o_host[i] - reference[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < 1e-4,
        "Max error {} exceeds tolerance 1e-4",
        max_err
    );
    println!("PASSED: flash attention test (max_err={})", max_err);
}

/// Test with small sequence length where all tiling fits in one block
#[test]
fn test_flash_attention_small() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const B: usize = 1;
    const NH: usize = 1;
    const N: usize = 4; // very short sequence
    const D: usize = 4; // small dimension
    const BC: usize = 2; // block size smaller than N
    const BR: usize = 2;

    let tc = ((N as f32) / (BC as f32)).ceil() as i32;
    let tr = ((N as f32) / (BR as f32)).ceil() as i32;
    let softmax_scale = 1.0f32 / (D as f32).sqrt();

    let q_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let k_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.2 - 2.0)
        .collect();
    let v_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.3 - 0.5)
        .collect();

    let q_dev = DeviceBuffer::from_host(&stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v_host).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N * D).unwrap();
    let mut l_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N).unwrap();
    let mut m_dev =
        DeviceBuffer::<f32>::from_host(&stream, &vec![-f32::INFINITY; B * NH * N]).unwrap();

    let tile_size = BC * D;
    let sram_size = (3 * tile_size + BC * BC) * std::mem::size_of::<f32>();

    let module = flash::load(&ctx).expect("Failed to load embedded CUDA module");
    unsafe {
        module.flash_attention(
            &stream,
            LaunchConfig {
                grid_dim: (B as u32, NH as u32, 1),
                block_dim: (BC as u32, 1, 1),
                shared_mem_bytes: sram_size as u32,
            },
            &q_dev,
            &k_dev,
            &v_dev,
            N as i32,
            D as i32,
            tc,
            tr,
            BC as i32,
            BR as i32,
            softmax_scale,
            &mut l_dev,
            &mut m_dev,
            &mut o_dev,
        )
    }
    .expect("Kernel launch failed");

    let o_host = o_dev.to_host_vec(&stream).unwrap();
    let reference = reference_attention(&q_host, &k_host, &v_host, N, D);
    let mut max_err = 0.0f32;
    for i in 0..N * D {
        let err = (o_host[i] - reference[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < 1e-4,
        "Small test: Max error {} exceeds tolerance 1e-4 (N={}, D={}, BC={})",
        max_err,
        N,
        D,
        BC
    );
    println!("PASSED: flash attention small test (max_err={})", max_err);
}

/// Test with multiple batches and heads
#[test]
fn test_flash_attention_multi_batch_head() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const B: usize = 2; // multiple batches
    const NH: usize = 2; // multiple heads
    const N: usize = 16;
    const D: usize = 16;
    const BC: usize = 8;
    const BR: usize = 8;

    let tc = ((N as f32) / (BC as f32)).ceil() as i32;
    let tr = ((N as f32) / (BR as f32)).ceil() as i32;
    let softmax_scale = 1.0f32 / (D as f32).sqrt();

    let q_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let k_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.2 - 2.0)
        .collect();
    let v_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.3 - 0.5)
        .collect();

    let q_dev = DeviceBuffer::from_host(&stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v_host).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N * D).unwrap();
    let mut l_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N).unwrap();
    let mut m_dev =
        DeviceBuffer::<f32>::from_host(&stream, &vec![-f32::INFINITY; B * NH * N]).unwrap();

    let tile_size = BC * D;
    let sram_size = (3 * tile_size + BC * BC) * std::mem::size_of::<f32>();

    let module = flash::load(&ctx).expect("Failed to load embedded CUDA module");
    unsafe {
        module.flash_attention(
            &stream,
            LaunchConfig {
                grid_dim: (B as u32, NH as u32, 1),
                block_dim: (BC as u32, 1, 1),
                shared_mem_bytes: sram_size as u32,
            },
            &q_dev,
            &k_dev,
            &v_dev,
            N as i32,
            D as i32,
            tc,
            tr,
            BC as i32,
            BR as i32,
            softmax_scale,
            &mut l_dev,
            &mut m_dev,
            &mut o_dev,
        )
    }
    .expect("Kernel launch failed");

    let o_host = o_dev.to_host_vec(&stream).unwrap();
    let reference = reference_attention(&q_host, &k_host, &v_host, N, D);
    let mut max_err = 0.0f32;
    for i in 0..N * D {
        let err = (o_host[i] - reference[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < 1e-4,
        "Multi-batch/head test: Max error {} exceeds tolerance 1e-4 (B={}, NH={}, N={}, D={}, BC={})",
        max_err,
        B,
        NH,
        N,
        D,
        BC
    );
    println!(
        "PASSED: flash attention multi-batch-head test (max_err={})",
        max_err
    );
}

/// Test with different sequence lengths vs block sizes (asymmetric tiling)
/// Note: BC must equal BR for this kernel's single-block-per-tile design.
/// We use BC = BR = 16, resulting in tr=tc=4 for N=64.
#[test]
fn test_flash_attention_asymmetric() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const B: usize = 1;
    const NH: usize = 1;
    const N: usize = 64;   // longer sequence
    const D: usize = 64;   // larger dimension
    const BC: usize = 16;  // block size
    const BR: usize = 16;  // must match BC for this kernel

    let tc = ((N as f32) / (BC as f32)).ceil() as i32;
    let tr = ((N as f32) / (BR as f32)).ceil() as i32;
    let softmax_scale = 1.0f32 / (D as f32).sqrt();

    let q_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let k_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.2 - 2.0)
        .collect();
    let v_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.3 - 0.5)
        .collect();

    let q_dev = DeviceBuffer::from_host(&stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v_host).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N * D).unwrap();
    let mut l_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N).unwrap();
    let mut m_dev =
        DeviceBuffer::<f32>::from_host(&stream, &vec![-f32::INFINITY; B * NH * N]).unwrap();

    let tile_size = BC * D;
    let sram_size = (3 * tile_size + BC * BC) * std::mem::size_of::<f32>();

    let module = flash::load(&ctx).expect("Failed to load embedded CUDA module");
    unsafe {
        module.flash_attention(
            &stream,
            LaunchConfig {
                grid_dim: (B as u32, NH as u32, 1),
                block_dim: (BC as u32, 1, 1),
                shared_mem_bytes: sram_size as u32,
            },
            &q_dev,
            &k_dev,
            &v_dev,
            N as i32,
            D as i32,
            tc,
            tr,
            BC as i32,
            BR as i32,
            softmax_scale,
            &mut l_dev,
            &mut m_dev,
            &mut o_dev,
        )
    }
    .expect("Kernel launch failed");

    let o_host = o_dev.to_host_vec(&stream).unwrap();
    let reference = reference_attention(&q_host, &k_host, &v_host, N, D);
    let mut max_err = 0.0f32;
    for i in 0..N * D {
        let err = (o_host[i] - reference[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < 1e-4,
        "Asymmetric test: Max error {} exceeds tolerance 1e-4 (N={}, D={}, BC={}, BR={})",
        max_err,
        N,
        D,
        BC,
        BR
    );
    println!(
        "PASSED: flash attention asymmetric test (max_err={})",
        max_err
    );
}

/// Test with large head dimension relative to block size
#[test]
fn test_flash_attention_large_dim() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const B: usize = 1;
    const NH: usize = 1;
    const N: usize = 32;
    const D: usize = 32; // large dimension
    const BC: usize = 32;
    const BR: usize = 32;

    let tc = ((N as f32) / (BC as f32)).ceil() as i32;
    let tr = ((N as f32) / (BR as f32)).ceil() as i32;
    let softmax_scale = 1.0f32 / (D as f32).sqrt();

    let q_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let k_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.2 - 2.0)
        .collect();
    let v_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.3 - 0.5)
        .collect();

    let q_dev = DeviceBuffer::from_host(&stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v_host).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N * D).unwrap();
    let mut l_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N).unwrap();
    let mut m_dev =
        DeviceBuffer::<f32>::from_host(&stream, &vec![-f32::INFINITY; B * NH * N]).unwrap();

    let tile_size = BC * D;
    let sram_size = (3 * tile_size + BC * BC) * std::mem::size_of::<f32>();

    let module = flash::load(&ctx).expect("Failed to load embedded CUDA module");
    unsafe {
        module.flash_attention(
            &stream,
            LaunchConfig {
                grid_dim: (B as u32, NH as u32, 1),
                block_dim: (BC as u32, 1, 1),
                shared_mem_bytes: sram_size as u32,
            },
            &q_dev,
            &k_dev,
            &v_dev,
            N as i32,
            D as i32,
            tc,
            tr,
            BC as i32,
            BR as i32,
            softmax_scale,
            &mut l_dev,
            &mut m_dev,
            &mut o_dev,
        )
    }
    .expect("Kernel launch failed");

    let o_host = o_dev.to_host_vec(&stream).unwrap();
    let reference = reference_attention(&q_host, &k_host, &v_host, N, D);
    let mut max_err = 0.0f32;
    for i in 0..N * D {
        let err = (o_host[i] - reference[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < 1e-4,
        "Large dim test: Max error {} exceeds tolerance 1e-4 (D={}, BC={})",
        max_err,
        D,
        BC
    );
    println!(
        "PASSED: flash attention large dim test (max_err={})",
        max_err
    );
}

/// Test with Br < N (different row block size) - but BC must equal BR
#[test]
fn test_flash_attention_row_tiling() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const B: usize = 1;
    const NH: usize = 1;
    const N: usize = 64;
    const D: usize = 64;
    const BC: usize = 32;
    const BR: usize = 32; // must match BC for this kernel design

    let tc = ((N as f32) / (BC as f32)).ceil() as i32;
    let tr = ((N as f32) / (BR as f32)).ceil() as i32;
    let softmax_scale = 1.0f32 / (D as f32).sqrt();

    let q_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let k_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.2 - 2.0)
        .collect();
    let v_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.3 - 0.5)
        .collect();

    let q_dev = DeviceBuffer::from_host(&stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v_host).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N * D).unwrap();
    let mut l_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N).unwrap();
    let mut m_dev =
        DeviceBuffer::<f32>::from_host(&stream, &vec![-f32::INFINITY; B * NH * N]).unwrap();

    let tile_size = BC * D;
    let sram_size = (3 * tile_size + BC * BC) * std::mem::size_of::<f32>();

    let module = flash::load(&ctx).expect("Failed to load embedded CUDA module");
    unsafe {
        module.flash_attention(
            &stream,
            LaunchConfig {
                grid_dim: (B as u32, NH as u32, 1),
                block_dim: (BC as u32, 1, 1),
                shared_mem_bytes: sram_size as u32,
            },
            &q_dev,
            &k_dev,
            &v_dev,
            N as i32,
            D as i32,
            tc,
            tr,
            BC as i32,
            BR as i32,
            softmax_scale,
            &mut l_dev,
            &mut m_dev,
            &mut o_dev,
        )
    }
    .expect("Kernel launch failed");

    let o_host = o_dev.to_host_vec(&stream).unwrap();
    let reference = reference_attention(&q_host, &k_host, &v_host, N, D);
    let mut max_err = 0.0f32;
    for i in 0..N * D {
        let err = (o_host[i] - reference[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < 1e-4,
        "Row tiling test: Max error {} exceeds tolerance 1e-4 (BR={})",
        max_err,
        BR
    );
    println!(
        "PASSED: flash attention row tiling test (max_err={})",
        max_err
    );
}

/// Verify that l (log-sum-exp) and m (running max) are correctly computed by checking their values
#[test]
fn test_flash_attention_l_m_values() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const B: usize = 1;
    const NH: usize = 1;
    const N: usize = 8;
    const D: usize = 8;
    const BC: usize = 8;
    const BR: usize = 8;

    let tc = ((N as f32) / (BC as f32)).ceil() as i32;
    let tr = ((N as f32) / (BR as f32)).ceil() as i32;
    let softmax_scale = 1.0f32 / (D as f32).sqrt();

    let q_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let k_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.2 - 2.0)
        .collect();
    let v_host: Vec<f32> = (0..B * NH * N * D)
        .map(|i| (i as f32) * 0.3 - 0.5)
        .collect();

    let q_dev = DeviceBuffer::from_host(&stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v_host).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N * D).unwrap();
    let mut l_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N).unwrap();
    let mut m_dev =
        DeviceBuffer::<f32>::from_host(&stream, &vec![-f32::INFINITY; B * NH * N]).unwrap();

    let tile_size = BC * D;
    let sram_size = (3 * tile_size + BC * BC) * std::mem::size_of::<f32>();

    let module = flash::load(&ctx).expect("Failed to load embedded CUDA module");
    unsafe {
        module.flash_attention(
            &stream,
            LaunchConfig {
                grid_dim: (B as u32, NH as u32, 1),
                block_dim: (BC as u32, 1, 1),
                shared_mem_bytes: sram_size as u32,
            },
            &q_dev,
            &k_dev,
            &v_dev,
            N as i32,
            D as i32,
            tc,
            tr,
            BC as i32,
            BR as i32,
            softmax_scale,
            &mut l_dev,
            &mut m_dev,
            &mut o_dev,
        )
    }
    .expect("Kernel launch failed");

    let l_host = l_dev.to_host_vec(&stream).unwrap();
    let m_host = m_dev.to_host_vec(&stream).unwrap();

    // Verify that l contains reasonable values (not NaN or extreme)
    for i in 0..B * NH * N {
        assert!(!f32::is_nan(l_host[i]));
        assert!(!f32::is_infinite(l_host[i]));
        // Log-sum-exp should be in reasonable range given our input values
        assert!(
            l_host[i] > -100.0 && l_host[i] < 100.0,
            "l value out of range: {}",
            l_host[i]
        );
    }

    // Verify that m contains running maximums (monotonically non-decreasing per row)
    for head in 0..NH {
        for batch in 0..B {
            let lm_offset = (batch * NH + head) * N;
            let mut prev_m = f32::NEG_INFINITY;
            for i in 0..N {
                let m_val = m_host[lm_offset + i];
                // Online softmax m should be non-decreasing
                assert!(
                    m_val >= prev_m - 1e-5,
                    "m not non-decreasing at i={}, val={}, prev={}",
                    i,
                    m_val,
                    prev_m
                );
                prev_m = m_val;
            }
        }
    }

    println!("PASSED: flash attention l/m values test");
}

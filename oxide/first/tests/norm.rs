use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use first::norm::{BLOCK_SIZE, norm};

/// Row-wise RMSNorm reference: y = x / sqrt(mean(x^2) + epsilon)
fn reference_rmsnorm(x: &[f32], n_rows: usize, d: usize, epsilon: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; n_rows * d];
    for row in 0..n_rows {
        let base = row * d;
        let mean_sq = x[base..base + d].iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv_rms = 1.0f32 / (mean_sq + epsilon).sqrt();
        for col in 0..d {
            y[base + col] = x[base + col] * inv_rms;
        }
    }
    y
}

/// Row-wise LayerNorm reference: y = (x - mean(x)) / sqrt(var(x) + epsilon)
fn reference_layernorm(x: &[f32], n_rows: usize, d: usize, epsilon: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; n_rows * d];
    for row in 0..n_rows {
        let base = row * d;
        let row_x = &x[base..base + d];
        let mu = row_x.iter().sum::<f32>() / d as f32;
        let variance = row_x.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / d as f32;
        let inv_std = 1.0f32 / (variance + epsilon).sqrt();
        for col in 0..d {
            y[base + col] = (x[base + col] - mu) * inv_std;
        }
    }
    y
}

#[test]
fn test_rmsnorm() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N_ROWS: usize = 4;
    const D: usize = 1024; // larger than BLOCK_SIZE to exercise the strided loops
    const EPSILON: f32 = 1e-5;

    let x_host: Vec<f32> = (0..N_ROWS * D)
        .map(|i| (i as f32 * 0.37).sin() * 2.0 + 0.1)
        .collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, N_ROWS * D).unwrap();

    let module = norm::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: one block per row with BLOCK_SIZE threads; each thread writes a
    // disjoint strided set of columns within its row, and the buffers cover
    // the kernel's accesses. No dynamic shared memory is used.
    unsafe {
        module.rmsnorm(
            &stream,
            LaunchConfig {
                grid_dim: (N_ROWS as u32, 1, 1),
                block_dim: (BLOCK_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &x_dev,
            D as i32,
            EPSILON,
            &mut y_dev,
        )
    }
    .expect("Kernel launch failed");

    let y_host = y_dev.to_host_vec(&stream).unwrap();

    let reference = reference_rmsnorm(&x_host, N_ROWS, D, EPSILON);
    let mut max_err = 0.0f32;
    for i in 0..N_ROWS * D {
        let err = (y_host[i] - reference[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < 1e-4,
        "Max error {} exceeds tolerance 1e-4",
        max_err
    );
    println!("PASSED: rmsnorm test (max_err={})", max_err);
}

#[test]
fn test_layernorm() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N_ROWS: usize = 4;
    const D: usize = 1024; // larger than BLOCK_SIZE to exercise the strided loops
    const EPSILON: f32 = 1e-5;

    // Non-zero mean per row so the mean subtraction actually matters
    let x_host: Vec<f32> = (0..N_ROWS * D)
        .map(|i| (i as f32 * 0.53).cos() * 1.5 + 0.7)
        .collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, N_ROWS * D).unwrap();

    let module = norm::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: one block per row with BLOCK_SIZE threads; each thread writes a
    // disjoint strided set of columns within its row, and the buffers cover
    // the kernel's accesses. No dynamic shared memory is used.
    unsafe {
        module.layernorm(
            &stream,
            LaunchConfig {
                grid_dim: (N_ROWS as u32, 1, 1),
                block_dim: (BLOCK_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &x_dev,
            D as i32,
            EPSILON,
            &mut y_dev,
        )
    }
    .expect("Kernel launch failed");

    let y_host = y_dev.to_host_vec(&stream).unwrap();

    let reference = reference_layernorm(&x_host, N_ROWS, D, EPSILON);
    let mut max_err = 0.0f32;
    for i in 0..N_ROWS * D {
        let err = (y_host[i] - reference[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    // Each normalized row should also have mean ~0 and variance ~1
    for row in 0..N_ROWS {
        let row_y = &y_host[row * D..(row + 1) * D];
        let mean = row_y.iter().sum::<f32>() / D as f32;
        let var = row_y.iter().map(|v| v * v).sum::<f32>() / D as f32;
        assert!(mean.abs() < 1e-3, "Row {} mean {} not ~0", row, mean);
        assert!(
            (var - 1.0).abs() < 1e-2,
            "Row {} variance {} not ~1",
            row,
            var
        );
    }

    assert!(
        max_err < 1e-4,
        "Max error {} exceeds tolerance 1e-4",
        max_err
    );
    println!("PASSED: layernorm test (max_err={})", max_err);
}

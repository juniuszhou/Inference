use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, DynamicSharedArray, SharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Vector addition that stages `a` through static shared memory and `b`
    /// through dynamic shared memory before writing the sum.
    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

        let idx = thread::index_1d();
        let i = idx.get();
        let tid = thread::threadIdx_x() as usize;

        let smem = DynamicSharedArray::<f32>::get();
        if i < a.len() {
            unsafe {
                TILE[tid] = a[i];
                *smem.add(tid) = b[i];
            }
        }

        thread::sync_threads();

        if let Some(c_elem) = c.get_mut(idx) {
            unsafe {
                *c_elem = TILE[tid] + *smem.add(tid);
            }
        }
    }
}

#[test]
fn test_sram() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    const BLOCK_SIZE: u32 = 256;

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: this is a 1D launch and `vecadd` guards its index against the
    // output length before writing; the dynamic shared memory covers one f32
    // per thread in the block.
    unsafe {
        module.vecadd(
            &stream,
            LaunchConfig {
                grid_dim: ((N as u32).div_ceil(BLOCK_SIZE), 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
            },
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }
    .expect("Kernel launch failed");

    let c_host = c_dev.to_host_vec(&stream).unwrap();

    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }
}

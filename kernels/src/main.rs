use cuda_core::{CudaContext, DeviceBuffer, launch_kernel_on_stream};
use std::ffi::c_void;

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let ptx = include_bytes!(concat!(env!("OUT_DIR"), "/vecadd.ptx"));
    let module = ctx
        .load_module_from_image(ptx)
        .expect("Failed to load PTX module");

    let func = module
        .load_function("vecadd")
        .expect("Failed to find vecadd kernel");

    let d_a = a_dev.cu_deviceptr();
    let d_b = b_dev.cu_deviceptr();
    let d_c = c_dev.cu_deviceptr();
    let n: i32 = N as i32;

    let mut args: Vec<*mut c_void> = vec![
        &d_a as *const _ as *mut c_void,
        &d_b as *const _ as *mut c_void,
        &d_c as *const _ as *mut c_void,
        &n as *const _ as *mut c_void,
    ];

    let grid_dim = ((N as u32 + 255) / 256, 1u32, 1u32);
    let block_dim = (256u32, 1u32, 1u32);

    unsafe {
        launch_kernel_on_stream(&func, grid_dim, block_dim, 0, &stream, &mut args)
            .expect("Kernel launch failed");
    }

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

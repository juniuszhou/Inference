use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use std::ffi::CString;

#[test]
fn test_cudarc_mem() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let _inp = stream.clone_htod(&[1.0f32; 100])?;
    let _out = stream.alloc_zeros::<f32>(100)?;

    Ok(())
}

#[test]
fn test_cudarc_call() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Load the pre-compiled PTX from the build script (OUT_DIR)
    let ptx_bytes = include_bytes!("../../vecadd.ptx");
    let ptx_src = CString::new(ptx_bytes as &[u8])?;
    let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src.into_string()?);

    let module = ctx.load_module(ptx)?;
    let func = module.load_function("vecadd")?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = stream.clone_htod(&a_host)?;
    let b_dev = stream.clone_htod(&b_host)?;
    let mut c_dev = stream.alloc_zeros::<f32>(N)?;

    let n: i32 = N as i32;
    let cfg = LaunchConfig::for_num_elems(N as u32);
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&n)
            .launch(cfg)?;
    }

    let mut c_host = vec![0.0f32; N];
    stream.memcpy_dtoh(&c_dev, &mut c_host)?;

    let errors = c_host
        .iter()
        .zip(a_host.iter().zip(b_host.iter()))
        .filter(|&(c, (a, b))| (c - (a + b)).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        panic!("vecadd kernel produced incorrect results");
    }

    Ok(())
}

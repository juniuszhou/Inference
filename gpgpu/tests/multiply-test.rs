use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

#[test]
fn test_multiply() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let m = 4i32;
    let k = 4i32;
    let n = 4i32;

    let a_host = (0..(m * k)).map(|x| x as f32).collect::<Vec<_>>();
    let b_host = (0..(k * n)).map(|x| x as f32).collect::<Vec<_>>();

    let a_dev = stream.clone_htod(&a_host)?;
    let b_dev = stream.clone_htod(&b_host)?;
    let c_dev = stream.alloc_zeros::<f32>((m * n) as usize)?;

    let ptx_src = include_bytes!("../multiply.ptx");
    let ptx = cudarc::nvrtc::Ptx::from_src(std::str::from_utf8(ptx_src).unwrap());
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("matmul")?;

    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&c_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (4, 4, 1),
                shared_mem_bytes: 0,
            })
    }?;

    // block until the kernel is finished
    stream.synchronize()?;
    let c_host = stream.clone_dtoh(&c_dev)?;

    println!("a = {:?}", a_host);
    println!("b = {:?}", b_host);
    println!("c = {:?}", c_host);

    Ok(())
}

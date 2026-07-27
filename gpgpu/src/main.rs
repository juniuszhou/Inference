use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

fn main() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let a_host = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b_host = [6.0f32, 7.0, 8.0, 9.0, 10.0];
    let n = a_host.len();

    let a_dev = stream.clone_htod(&a_host)?;
    let b_dev = stream.clone_htod(&b_host)?;
    let c_dev = stream.alloc_zeros::<f32>(n)?;

    let ptx_src = include_bytes!(concat!(env!("OUT_DIR"), "/add.ptx"));
    let ptx = cudarc::nvrtc::Ptx::from_src(std::str::from_utf8(ptx_src).unwrap());
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("add")?;

    let block_size = 256u32;
    let grid_size = ((n as u32 + block_size - 1) / block_size, 1, 1);

    println!("grid_size = {:?}", grid_size);

    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&c_dev)
            .arg(&(n as i32))
            .launch(LaunchConfig {
                grid_dim: grid_size,
                block_dim: (block_size, 1, 1),
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

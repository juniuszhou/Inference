use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

use anyhow::Result;

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Load pre-compiled PTX from file (compiled with nvcc -arch=sm_75)
    let ptx_src = std::fs::read_to_string("/tmp/sin_kernel.ptx")?;
    let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);

    let module = ctx.load_module(ptx)?;
    let sin_kernel = module.load_function("sin_kernel")?;

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let numel = input_data.len();

    let a_dev = stream.clone_htod(&input_data)?;
    let mut b_dev = a_dev.try_clone()?;

    unsafe {
        stream
            .launch_builder(&sin_kernel)
            .arg(&mut b_dev)
            .arg(&a_dev)
            .arg(&numel)
            .launch(LaunchConfig::for_num_elems(numel as u32))
    }?;

    stream.synchronize()?;

    let output_data = stream.clone_dtoh(&b_dev)?;

    println!("Input:  {:?}", input_data);
    println!("Output: {:?}", output_data);

    Ok(())
}

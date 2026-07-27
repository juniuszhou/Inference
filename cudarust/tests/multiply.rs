use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};

const PTX_SRC: &str = "
extern \"C\" __global__ void matmul(float* A, float* B, float* C, int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}
";

fn compile_and_load(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
) -> Result<cudarc::driver::CudaFunction, DriverError> {
    use std::io::Write;
    use std::process::Command;

    let ptx_path = "/tmp/mul.ptx";

    if !std::path::Path::new(ptx_path).exists() {
        let cu_path = "/tmp/mul.cu";
        {
            let mut f = std::fs::File::create(cu_path).unwrap();
            f.write_all(PTX_SRC.as_bytes()).unwrap();
        }
        let status = Command::new("nvcc")
            .args(&["-arch=sm_75", "--ptx", cu_path, "-o", ptx_path])
            .status()
            .expect("failed to execute nvcc");
        assert!(status.success(), "nvcc compilation failed");
    }

    let ptx_src = std::fs::read_to_string(ptx_path).unwrap();
    let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
    let module = ctx.load_module(ptx)?;
    let f = module.load_function("matmul")?;
    Ok(f)
}

#[test]
fn test_multiply() -> Result<(), DriverError> {
    let start = std::time::Instant::now();

    let ctx = CudaContext::new(0)?;
    println!("Built in {:?}", start.elapsed());

    let f = compile_and_load(&ctx)?;
    println!("Loaded in {:?}", start.elapsed());

    let stream = ctx.default_stream();

    let a_host = [1.0f32, 2.0, 3.0, 4.0];
    let b_host = [1.0f32, 2.0, 3.0, 4.0];
    let mut c_host = [0.0f32; 4];

    let a_dev = stream.clone_htod(&a_host)?;
    let b_dev = stream.clone_htod(&b_host)?;
    let mut c_dev = stream.clone_htod(&c_host)?;

    println!("Copied in {:?}", start.elapsed());

    let mut builder = stream.launch_builder(&f);
    builder.arg(&a_dev);
    builder.arg(&b_dev);
    builder.arg(&mut c_dev);
    builder.arg(&2i32);
    let cfg = LaunchConfig {
        block_dim: (2, 2, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { builder.launch(cfg) }?;

    stream.memcpy_dtoh(&c_dev, &mut c_host)?;
    println!("Found {:?} in {:?}", c_host, start.elapsed());
    Ok(())
}

/*
CUDA graph demo against the cudarc crate (workspace pins `cudarc` 0.19.8).

Builds a two-node computation graph on a stream by *capturing* two
dependent kernel launches, then replays the captured graph via the
cudarc interface:

    node 1 (mul_const):   tmp[i] = a[i] * SCALE
    node 2 (accumulate):  out[i] += tmp[i]

The captures become graph nodes, so launching the graph once is
`out += a * SCALE`. Launching it k times yields `out[i] = k * (a[i] * SCALE)`.

Two cudarc-specific requirements for graph capture:
  - capture needs a real stream created with `CudaContext::new_stream()`;
    the NULL/legacy `default_stream()` does not support capture.
  - the per-slice `CudaEvent` tracking that cudarc inserts around kernel
    launches would create dependencies on uncaptured work, so it is
    disabled for the capture region.

Underlying driver API (wrapped by cudarc):
  cuStreamBeginCapture       -> CudaStream::begin_capture()
  cuStreamEndCapture         -> CudaStream::end_capture()      -> CudaGraph
  cuGraphInstantiate         -> (inside CudaStream::end_capture)
  cuGraphUpload              -> CudaGraph::upload()
  cuGraphLaunch              -> CudaGraph::launch()
*/

use std::sync::Arc;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

const PTX_SRC: &str = "
extern \"C\" __global__ void mul_const(float* out, const float* in, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * s;
    }
}

extern \"C\" __global__ void accumulate(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] += in[i];
    }
}
";

fn compile_and_load(ctx: &Arc<CudaContext>) -> Result<(CudaFunction, CudaFunction), DriverError> {
    use std::io::Write;
    use std::process::Command;

    let ptx_path = "/tmp/cudarc_graph_demo.ptx";

    if !std::path::Path::new(ptx_path).exists() {
        let cu_path = "/tmp/cudarc_graph_demo.cu";
        {
            let mut f = std::fs::File::create(cu_path).unwrap();
            f.write_all(PTX_SRC.as_bytes()).unwrap();
        }
        let status = Command::new("nvcc")
            .args(["-arch=sm_75", "--ptx", cu_path, "-o", ptx_path])
            .status()
            .expect("failed to execute nvcc");
        assert!(status.success(), "nvcc compilation failed");
    }

    let ptx = Ptx::from_src(std::fs::read_to_string(ptx_path).unwrap());
    let module = ctx.load_module(ptx)?;
    let mul_fn = module.load_function("mul_const")?;
    let add_fn = module.load_function("accumulate")?;
    Ok((mul_fn, add_fn))
}

#[test]
fn graph_capture_and_replay() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;

    // `default_stream()` is the legacy NULL stream, which cannot be captured,
    // and the automatic per-slice event tracking would add dependencies on
    // uncaptured work inside the capture, so disable it and use a real stream.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;

    let (mul_fn, add_fn) = compile_and_load(&ctx)?;

    const N: usize = 4;
    const SCALE: f32 = 2.0;
    const REPLAYS: u32 = 3;

    let a_host = [1.0f32, 2.0, 3.0, 4.0];
    let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host)?;
    let mut tmp_dev: CudaSlice<f32> = stream.clone_htod(&[0.0f32; N])?;
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&[0.0f32; N])?;

    // ---- build the graph: start capturing, launch into a graph ----
    stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;

    let numel: i32 = N as i32;
    let cfg = LaunchConfig::for_num_elems(N as u32);
    unsafe {
        stream
            .launch_builder(&mul_fn)
            .arg(&mut tmp_dev)
            .arg(&a_dev)
            .arg(&SCALE)
            .arg(&numel)
            .launch(cfg)
    }?;

    let cfg = LaunchConfig::for_num_elems(N as u32);
    unsafe {
        stream
            .launch_builder(&add_fn)
            .arg(&mut out_dev)
            .arg(&tmp_dev)
            .arg(&numel)
            .launch(cfg)
    }?;

    let graph: CudaGraph = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
        .expect("capture produced no graph");

    // ---- replay the graph through the cudarc interface ----
    graph.upload()?;
    println!("Built 2-node graph; value after launch k: out[i] = k * (a[i] * {SCALE})");
    for k in 1..=REPLAYS {
        graph.launch()?;
        stream.synchronize()?;
        let out_host = stream.clone_dtoh(&out_dev)?;
        println!("after launch {k}: {:?}", out_host);
        for (i, &v) in out_host.iter().enumerate() {
            let expected = k as f32 * (a_host[i] * SCALE);
            assert!(
                (v - expected).abs() < 1e-5,
                "element {i}: got {v}, expected {expected}"
            );
        }
    }

    println!("done");
    Ok(())
}

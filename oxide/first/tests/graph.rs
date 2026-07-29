use std::ffi::c_void;

use cuda_core::{CudaContext, DeviceBuffer};
use first::graph::{CudaGraph, GraphKernels};

const BLOCK: u32 = 256;

fn grid_for(n: usize) -> (u32, u32, u32) {
    ((n as u32).div_ceil(BLOCK), 1, 1)
}

const BLOCK_DIM: (u32, u32, u32) = (BLOCK, 1, 1);

/// Shorthand for CUDA's kernel-argument convention: a `void*` pointing at
/// the argument value on the host.
macro_rules! arg {
    ($value:expr) => {
        &mut $value as *mut _ as *mut c_void
    };
}

#[test]
fn test_single_kernel_node() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let kernels = GraphKernels::load(&ctx).expect("Failed to load kernels");

    const N: usize = 1000; // not a multiple of BLOCK, exercises the bounds check

    let a_host: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let b_host: Vec<f32> = (0..N).map(|i| 100.0 - i as f32).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    // cuGraphCreate -> cuGraphAddKernelNode -> cuGraphInstantiate -> cuGraphLaunch
    let mut graph = CudaGraph::new(&ctx).expect("cuGraphCreate failed");

    let mut a_ptr = a_dev.cu_deviceptr();
    let mut b_ptr = b_dev.cu_deviceptr();
    let mut out_ptr = out_dev.cu_deviceptr();
    let mut n = N as i32;
    // SAFETY: args match vec_add(a, b, out, n); buffers outlive the launch.
    unsafe {
        graph.add_kernel_node(
            &kernels.vec_add,
            grid_for(N),
            BLOCK_DIM,
            0,
            &mut [arg!(a_ptr), arg!(b_ptr), arg!(out_ptr), arg!(n)],
            &[], // no dependencies: root node
        )
    }
    .expect("cuGraphAddKernelNode failed");

    let exec = graph.instantiate().expect("cuGraphInstantiate failed");
    // SAFETY: recorded device pointers are still alive.
    unsafe { exec.launch(&stream) }.expect("cuGraphLaunch failed");
    stream.synchronize().unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();
    for i in 0..N {
        assert!(
            (out_host[i] - (a_host[i] + b_host[i])).abs() < 1e-6,
            "mismatch at {}",
            i
        );
    }
    println!("PASSED: single kernel node graph");
}

#[test]
fn test_kernel_node_chain() {
    // Three dependent nodes: out = relu((a + b) * -1.5)
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let kernels = GraphKernels::load(&ctx).expect("Failed to load kernels");

    const N: usize = 512;
    const FACTOR: f32 = -1.5;

    let a_host: Vec<f32> = (0..N).map(|i| (i as f32 * 0.11).sin()).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i as f32 * 0.07).cos()).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let sum_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let scaled_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let mut graph = CudaGraph::new(&ctx).expect("cuGraphCreate failed");

    let mut a_ptr = a_dev.cu_deviceptr();
    let mut b_ptr = b_dev.cu_deviceptr();
    let mut sum_ptr = sum_dev.cu_deviceptr();
    let mut scaled_ptr = scaled_dev.cu_deviceptr();
    let mut out_ptr = out_dev.cu_deviceptr();
    let mut factor = FACTOR;
    let mut n = N as i32;

    // SAFETY: args match each kernel's signature; buffers outlive execution;
    // the dependency edges order the nodes so each reads completed data.
    unsafe {
        let add = graph
            .add_kernel_node(
                &kernels.vec_add,
                grid_for(N),
                BLOCK_DIM,
                0,
                &mut [arg!(a_ptr), arg!(b_ptr), arg!(sum_ptr), arg!(n)],
                &[],
            )
            .expect("add node failed");

        let scale = graph
            .add_kernel_node(
                &kernels.vec_scale,
                grid_for(N),
                BLOCK_DIM,
                0,
                &mut [arg!(sum_ptr), arg!(scaled_ptr), arg!(factor), arg!(n)],
                &[add], // runs after the add node
            )
            .expect("scale node failed");

        graph
            .add_kernel_node(
                &kernels.vec_relu,
                grid_for(N),
                BLOCK_DIM,
                0,
                &mut [arg!(scaled_ptr), arg!(out_ptr), arg!(n)],
                &[scale], // runs after the scale node
            )
            .expect("relu node failed");
    }

    let exec = graph.instantiate().expect("cuGraphInstantiate failed");
    // SAFETY: recorded device pointers are still alive.
    unsafe { exec.launch(&stream) }.expect("cuGraphLaunch failed");

    // wait for the stream to finish.
    stream.synchronize().unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let expected = ((a_host[i] + b_host[i]) * FACTOR).max(0.0);
        assert!(
            (out_host[i] - expected).abs() < 1e-5,
            "mismatch at {}: got {}, expected {}",
            i,
            out_host[i],
            expected
        );
    }
    println!("PASSED: three-node dependency chain");
}

#[test]
fn test_diamond_dependencies() {
    // Two independent root nodes feed a join node:
    //   left  = a * 2       (root)
    //   right = a * 3       (root)
    //   out   = left + right  == a * 5
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let kernels = GraphKernels::load(&ctx).expect("Failed to load kernels");

    const N: usize = 300;

    let a_host: Vec<f32> = (0..N).map(|i| (i as f32 * 0.31).sin() * 3.0).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let left_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let right_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let mut graph = CudaGraph::new(&ctx).expect("cuGraphCreate failed");

    let mut a_ptr = a_dev.cu_deviceptr();
    let mut left_ptr = left_dev.cu_deviceptr();
    let mut right_ptr = right_dev.cu_deviceptr();
    let mut out_ptr = out_dev.cu_deviceptr();
    let mut two = 2.0f32;
    let mut three = 3.0f32;
    let mut n = N as i32;

    // SAFETY: args match each kernel's signature; buffers outlive execution.
    unsafe {
        let left = graph
            .add_kernel_node(
                &kernels.vec_scale,
                grid_for(N),
                BLOCK_DIM,
                0,
                &mut [arg!(a_ptr), arg!(left_ptr), arg!(two), arg!(n)],
                &[],
            )
            .expect("left node failed");

        let right = graph
            .add_kernel_node(
                &kernels.vec_scale,
                grid_for(N),
                BLOCK_DIM,
                0,
                &mut [arg!(a_ptr), arg!(right_ptr), arg!(three), arg!(n)],
                &[],
            )
            .expect("right node failed");

        graph
            .add_kernel_node(
                &kernels.vec_add,
                grid_for(N),
                BLOCK_DIM,
                0,
                &mut [arg!(left_ptr), arg!(right_ptr), arg!(out_ptr), arg!(n)],
                &[left, right], // join: waits on both branches
            )
            .expect("join node failed");
    }

    let exec = graph.instantiate().expect("cuGraphInstantiate failed");
    // SAFETY: recorded device pointers are still alive.
    unsafe { exec.launch(&stream) }.expect("cuGraphLaunch failed");
    stream.synchronize().unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let expected = a_host[i] * 5.0;
        assert!(
            (out_host[i] - expected).abs() < 1e-4,
            "mismatch at {}: got {}, expected {}",
            i,
            out_host[i],
            expected
        );
    }
    println!("PASSED: diamond dependencies");
}

#[test]
fn test_relaunch_instantiated_graph() {
    // The point of instantiating: build once, launch many times. One node
    // doubles the buffer in place; three launches multiply it by 8.
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let kernels = GraphKernels::load(&ctx).expect("Failed to load kernels");

    const N: usize = 256;

    let x_host: Vec<f32> = (0..N).map(|i| i as f32 + 1.0).collect();
    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();

    let mut graph = CudaGraph::new(&ctx).expect("cuGraphCreate failed");

    let mut x_ptr = x_dev.cu_deviceptr();
    let mut out_ptr = x_dev.cu_deviceptr(); // in place: element i reads and writes only x[i]
    let mut factor = 2.0f32;
    let mut n = N as i32;
    // SAFETY: args match vec_scale(x, out, factor, n); in-place is safe
    // because each thread touches exactly its own element.
    unsafe {
        graph.add_kernel_node(
            &kernels.vec_scale,
            grid_for(N),
            BLOCK_DIM,
            0,
            &mut [arg!(x_ptr), arg!(out_ptr), arg!(factor), arg!(n)],
            &[],
        )
    }
    .expect("scale node failed");

    let exec = graph.instantiate().expect("cuGraphInstantiate failed");
    drop(graph); // the executable graph is independent of the template

    // SAFETY: recorded device pointers are still alive. Launches on the same
    // stream are ordered, so each doubling sees the previous result.
    for _ in 0..3 {
        unsafe { exec.launch(&stream) }.expect("cuGraphLaunch failed");
    }
    stream.synchronize().unwrap();

    let out_host = x_dev.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let expected = x_host[i] * 8.0;
        assert!(
            (out_host[i] - expected).abs() < 1e-4,
            "mismatch at {}: got {}, expected {}",
            i,
            out_host[i],
            expected
        );
    }
    println!("PASSED: relaunching an instantiated graph");
}

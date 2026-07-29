/*
CUDA Graph interop: build a graph of kernel launches with the CUDA Graph
driver API and launch it as a single unit.

This is the mechanism LLM runtimes use to remove per-launch CPU overhead:
describe the kernels and their dependencies once, instantiate an executable
graph, then re-launch that graph every step.

Runtime-API name        Driver-API call used here
----------------        -------------------------
cudaGraphCreate         cuGraphCreate
cudaGraphAddKernelNode  cuGraphAddKernelNode
cudaGraphInstantiate    cuGraphInstantiateWithFlags
cudaGraphLaunch         cuGraphLaunch  (launches the instantiated graph)
cudaGraphExecDestroy    cuGraphExecDestroy
cudaGraphDestroy        cuGraphDestroy

The file has two halves:
  1. `graph` — a #[cuda_module] with small elementwise kernels used as
     graph nodes. They take raw device pointers so their launch ABI is a
     plain pointer per argument, which keeps the node-parameter marshalling
     easy to follow.
  2. `CudaGraph` / `CudaGraphExec` — host-side wrappers over the raw
     driver handles (`CUgraph`, `CUgraphNode`, `CUgraphExec`).
*/

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaFunction, CudaStream, DriverError, IntoResult, sys};
use cuda_device::{kernel, thread};
use cuda_host::{CudaKernel, EmbeddedModuleError, cuda_module};

#[cuda_module]
pub mod graph {
    use super::*;

    /// out[i] = a[i] + b[i]
    #[kernel]
    pub fn vec_add(a: *const f32, b: *const f32, out: *mut f32, n: i32) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        if i < n {
            unsafe {
                *out.add(i as usize) = *a.add(i as usize) + *b.add(i as usize);
            }
        }
    }

    /// out[i] = x[i] * factor
    #[kernel]
    pub fn vec_scale(x: *const f32, out: *mut f32, factor: f32, n: i32) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        if i < n {
            unsafe {
                *out.add(i as usize) = *x.add(i as usize) * factor;
            }
        }
    }

    /// out[i] = max(x[i], 0)
    #[kernel]
    pub fn vec_relu(x: *const f32, out: *mut f32, n: i32) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        if i < n {
            unsafe {
                let v = *x.add(i as usize);
                *out.add(i as usize) = if v > 0.0 { v } else { 0.0 };
            }
        }
    }
}

/// Raw `CUfunction` handles for the kernels above, loaded from this crate's
/// embedded PTX bundle. Graph nodes need raw function handles, which the
/// `#[cuda_module]`-generated `LoadedModule` does not expose.
pub struct GraphKernels {
    pub vec_add: CudaFunction,
    pub vec_scale: CudaFunction,
    pub vec_relu: CudaFunction,
}

impl GraphKernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, EmbeddedModuleError> {
        let module = cuda_host::load_embedded_module(ctx, env!("CARGO_PKG_NAME"))?;
        // The PTX entry point of a #[kernel] is its original function name;
        // the generated `__<name>_CudaKernel` markers carry it.
        Ok(Self {
            vec_add: module.load_function(<graph::__vec_add_CudaKernel as CudaKernel>::PTX_NAME)?,
            vec_scale: module
                .load_function(<graph::__vec_scale_CudaKernel as CudaKernel>::PTX_NAME)?,
            vec_relu: module
                .load_function(<graph::__vec_relu_CudaKernel as CudaKernel>::PTX_NAME)?,
        })
    }
}

/// A node handle returned by [`CudaGraph::add_kernel_node`], used to declare
/// dependencies of later nodes.
#[derive(Clone, Copy)]
pub struct GraphNode(sys::CUgraphNode);

/// A CUDA graph under construction. Wraps the `CUgraph` handle from
/// `cuGraphCreate`.
pub struct CudaGraph {
    ctx: Arc<CudaContext>,
    graph: sys::CUgraph,
    /// Keeps the kernels' parent module loaded for as long as the graph
    /// (and any executable instantiated from it) references them.
    functions: Vec<CudaFunction>,
}

impl CudaGraph {
    /// Creates an empty graph (`cuGraphCreate`).
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;

        // init a CUgraph object, but no neeed to initialize it as zero.
        // used in FFI or other interactions with the driver API.
        // It could be any type and bind until it is used.
        let mut handle: MaybeUninit<sys::CUgraph> = MaybeUninit::uninit();

        // IntoResult is used to convert the tuple of (CUresult, MaybeUninit<T>) into a Result<T, DriverError>.
        let graph = unsafe { (sys::cuGraphCreate(handle.as_mut_ptr(), 0), handle) }.result()?;
        Ok(Self {
            ctx: ctx.clone(),
            graph,
            functions: Vec::new(),
        })
    }

    /// Adds a kernel launch as a graph node (`cuGraphAddKernelNode`).
    ///
    /// `kernel_params` is CUDA's `void**` convention: one pointer per kernel
    /// argument, each pointing at the argument's value on the host. The
    /// driver copies the values during this call, so they only need to live
    /// until it returns. `dependencies` lists nodes that must complete
    /// before this one runs; an empty list makes it a root node.
    ///
    /// # Safety
    ///
    /// `kernel_params` must match the kernel's real signature (count, types,
    /// sizes), and any device pointers passed must stay valid until the
    /// graph has finished executing. The launch geometry must satisfy the
    /// kernel's own indexing assumptions.
    pub unsafe fn add_kernel_node(
        &mut self,
        function: &CudaFunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: &mut [*mut c_void],
        dependencies: &[GraphNode],
    ) -> Result<GraphNode, DriverError> {
        self.ctx.bind_to_thread()?;

        // Zero-init and fill only the fields we use; the zeroed `kern`/`ctx`
        // tail fields tell the driver to take the context from `func`.
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        params.func = unsafe { function.cu_function() };
        params.gridDimX = grid_dim.0;
        params.gridDimY = grid_dim.1;
        params.gridDimZ = grid_dim.2;
        params.blockDimX = block_dim.0;
        params.blockDimY = block_dim.1;
        params.blockDimZ = block_dim.2;
        params.sharedMemBytes = shared_mem_bytes;
        params.kernelParams = kernel_params.as_mut_ptr();

        // convert the dependencies to a pointer to a array of CUgraphNode.
        let deps: Vec<sys::CUgraphNode> = dependencies.iter().map(|d| d.0).collect();
        let deps_ptr = if deps.is_empty() {
            std::ptr::null()
        } else {
            deps.as_ptr()
        };

        let mut node = MaybeUninit::uninit();
        let node = unsafe {
            (
                // add the kernel node to the graph.
                sys::cuGraphAddKernelNode_v2(
                    node.as_mut_ptr(),
                    self.graph,
                    deps_ptr,
                    deps.len(),
                    &params,
                ),
                node,
            )
        }
        .result()?;

        // add the function to the graph.
        self.functions.push(function.clone());
        Ok(GraphNode(node))
    }

    /// Validates the graph and builds an executable instance of it
    /// (`cuGraphInstantiate`). The `CudaGraph` can be dropped afterwards;
    /// the executable graph is independent of it.
    pub fn instantiate(&self) -> Result<CudaGraphExec, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut handle = MaybeUninit::uninit();

        // instantiate the graph. and use the same IntoResult trait to convert tuple to Result.
        let exec = unsafe {
            (
                sys::cuGraphInstantiateWithFlags(handle.as_mut_ptr(), self.graph, 0),
                handle,
            )
        }
        .result()?;
        Ok(CudaGraphExec {
            ctx: self.ctx.clone(),
            exec,
            _functions: self.functions.clone(),
        })
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        // Best-effort cleanup; the handle is invalid afterwards either way.
        if self.ctx.bind_to_thread().is_ok() {
            unsafe {
                sys::cuGraphDestroy(self.graph);
            }
        }
    }
}

/// An instantiated, launchable graph. Wraps the `CUgraphExec` handle and can
/// be launched any number of times.
pub struct CudaGraphExec {
    ctx: Arc<CudaContext>,
    exec: sys::CUgraphExec,
    _functions: Vec<CudaFunction>,
}

impl CudaGraphExec {
    /// Submits the whole graph to `stream` (`cuGraphLaunch`, the driver-API
    /// name of `cudaGraphExecLaunch`). All nodes run on the GPU in
    /// dependency order with a single host-side call.
    ///
    /// # Safety
    ///
    /// Device memory referenced by the recorded kernel parameters must still
    /// be valid, and the recorded launches' data races (if any) are the
    /// caller's responsibility, exactly as with a raw kernel launch.
    pub unsafe fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { sys::cuGraphLaunch(self.exec, stream.cu_stream()) }.result()
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            unsafe {
                sys::cuGraphExecDestroy(self.exec);
            }
        }
    }
}

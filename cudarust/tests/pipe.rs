use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys::{cublasSaxpy_v2, cublasStatus_t};
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::nccl::result::{group_end, group_start};
use cudarc::nccl::{Comm, Id};

/*
NCCL collective + P2P tests on a single physical GPU: WORLD_SIZE ranks, each
bound to its own stream on device 0, mimicking a multi-GPU clique on one card.

Two NCCL group requirements are respected throughout:

1. Comm initialization: ncclCommInitRank is a collective that blocks until every
   rank of the clique has called it. Issued sequentially it would deadlock on the
   first call, so all comms are created inside one group_start()/group_end().

2. Collectives/P2P across the clique's comms: when several ranks share one
   physical GPU (NCCL_MULTI_RANK_GPU_ENABLE), NCCL requires the operations from
   the different comms to be enqueued atomically; issuing them without a group
   can deadlock (observed: all ranks reach the collective, one never returns).
   P2P send/recv pairs would deadlock even on distinct devices without a group.
*/

const WORLD_SIZE: usize = 4;
const NUMEL: usize = 4;

// cargo runs tests in a binary in parallel; these tests create 4 NCCL comms on
// the same physical GPU each, so running them concurrently exhausts NCCL and
// makes comm init fail. Serialize with a global lock.
static NCCL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_nccl() -> std::sync::MutexGuard<'static, ()> {
    NCCL_TEST_LOCK.lock().unwrap()
}

// NCCL rejects several ranks on one GPU unless this is opted in
// (supported since NCCL 2.28). Must be set before the first NCCL call.
fn enable_multi_rank_on_single_gpu() {
    // SAFETY: set before any NCCL call in this test; every test sets the same
    // value, so even if tests race to set it the result is identical.
    unsafe { std::env::set_var("NCCL_MULTI_RANK_GPU_ENABLE", "1") };
}

/// Creates WORLD_SIZE streams and initializes one communicator per rank in a
/// single group, returning `(streams, comms)` where comm[i] is bound to
/// stream[i]. The unique id is what rank 0 would broadcast to other processes;
/// here all comms are created in-process.
fn setup_clique(
    ctx: &std::sync::Arc<CudaContext>,
    id: Id,
) -> (Vec<std::sync::Arc<cudarc::driver::CudaStream>>, Vec<Comm>) {
    let streams: Vec<_> = (0..WORLD_SIZE).map(|_| ctx.new_stream().unwrap()).collect();
    group_start().unwrap();
    let comms: Vec<_> = (0..WORLD_SIZE)
        .map(|rank| Comm::from_rank(streams[rank].clone(), rank, WORLD_SIZE, id).unwrap())
        .collect();
    group_end().unwrap();

    (streams, comms)
}

/*
kernel_add_one as a Rust function (with `threadIdx.x`) is not valid Rust: cudarc
cannot compile it. Instead the kernel is CUDA C compiled to PTX with nvcc (same
pattern as tests/multiply.rs and tests/graph.rs) and loaded from the module.

for_num_elems(NUMEL) launches one 1024-thread block, so the kernel needs a
bounds check (`tid < n`) or threads NUMEL..1024 would read out of bounds.
*/
// const PTX_SRC: &str = "
// extern \"C\" __global__ void add_one(float* out, const float* in, int n) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < n) {
//         out[tid] = in[tid] + 1.0f;
//     }
// }
// ";

// fn compile_and_load(ctx: &std::sync::Arc<CudaContext>) -> Result<CudaFunction, DriverError> {
//     use std::io::Write;
//     use std::process::Command;

//     let ptx_path = "/tmp/cudarc_pipe_demo.ptx";

//     if !std::path::Path::new(ptx_path).exists() {
//         let cu_path = "/tmp/cudarc_pipe_demo.cu";
//         {
//             let mut f = std::fs::File::create(cu_path).unwrap();
//             f.write_all(PTX_SRC.as_bytes()).unwrap();
//         }
//         let status = Command::new("nvcc")
//             .args(["-arch=sm_75", "--ptx", cu_path, "-o", ptx_path])
//             .status()
//             .expect("failed to execute nvcc");
//         assert!(status.success(), "nvcc compilation failed");
//     }

//     let ptx = cudarc::nvrtc::Ptx::from_src(std::fs::read_to_string(ptx_path).unwrap());
//     let module = ctx.load_module(ptx)?;
//     module.load_function("add_one")
// }

// pipeline parallel test, each rank has its own stream, and the output of one rank is the input of the next rank.
#[test]
fn test_pipeline() {
    let _guard = lock_nccl();
    enable_multi_rank_on_single_gpu();

    // 这个 ID 内部其实封装了 Master 节点监听的 IP 地址和 Socket 端口
    // 用于集群启动时的握手与建组，而 rank 是组内的标识
    let id = Id::new().unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let (streams, _comms) = setup_clique(&ctx, id);

    // 每个 rank 在自己的 stream 上跑同一份 saxpy：c := alpha*a + c。
    // 输出 buffer 用全 1 初始化，所以 c = a + 1，等价于原 kernel 的 out = in + 1。
    // rank 0 的输入是 0；之后每个 rank 的输入 = 上一个 rank 的输出，
    // 通过 memcpy_dtod 直接在 device 之间拷贝（跨 stream），不再经过 host。
    let mut host_outputs: Vec<f32> = vec![0.0; NUMEL];
    let mut index = 1.0f32;

    let mut prev_output: Option<CudaSlice<f32>> = None;

    for (i, stream) in streams.iter().enumerate() {
        let blas = CudaBlas::new(stream.clone()).unwrap();

        let mut input: CudaSlice<f32> = stream.alloc_zeros::<f32>(NUMEL).unwrap();
        // 输出 buffer 用全 1 初始化（不是 alloc_zeros），这样 saxpy 的
        // c := alpha*a + c 才是 a + 1，而不是 a + 0 = a。
        let mut output: CudaSlice<f32> = stream.clone_htod(&[1.0f32; NUMEL]).unwrap();

        if let Some(prev) = prev_output.as_ref() {
            // 跨 stream 拷贝：把 rank i-1 的输出（写在 stream[i-1] 上）拷到本 rank
            // 的输入 buffer（属于 stream[i]）。cudarc 的 event tracking 会让
            // stream[i] wait 在 prev 的 write event 上，保证上一个 saxpy 已完成。
            stream.memcpy_dtod(prev, &mut input).unwrap();
        }

        let n: i32 = NUMEL as i32;
        let alpha: f32 = 1.0;
        unsafe {
            let (a_ptr, _sa) = input.device_ptr(stream);
            let (c_ptr, _sc) = output.device_ptr_mut(stream);
            // c := alpha*a + c = a + 1
            let status = cublasSaxpy_v2(
                *blas.handle(),
                n,
                &alpha,
                a_ptr as *const f32,
                1,
                c_ptr as *mut f32,
                1,
            );
            assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }

        // saxpy 入队在 stream[i] 上；同步后保证：
        // 1) 读回 host 时拷贝已完成（clone_dtoh 是异步的）；
        // 2) 下一次 memcpy_dtod 跨 stream 拷贝时该 buffer 已写完。
        stream.synchronize().unwrap();

        prev_output = Some(output);
        host_outputs = stream.clone_dtoh(prev_output.as_ref().unwrap()).unwrap();
        println!("Output: {:?}", host_outputs);
        assert_eq!(host_outputs, vec![index; NUMEL]);

        index += 1.0f32;
    }
}

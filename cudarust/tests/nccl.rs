use cudarc::driver::{CudaContext, CudaSlice};
use cudarc::nccl::result::{group_end, group_start};
use cudarc::nccl::{Comm, Id, ReduceOp};

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

    /*
    因为 ncclCommInitRank（Comm::from_rank 内部调用）本身是一个 collective 操作：它会阻塞住，直到 clique 里所有 rank 都调用了它。
    关键问题在于这里是在单个线程里顺序循环创建 4 个 comm：
    第一次迭代: Comm::from_rank(rank=0)  → 阻塞，等 rank 1/2/3 也来调用
    第二次迭代: 永远到不了，因为线程卡在第一次调用了
    线程在第一个调用上就死锁了——它自己就是唯一会去调用 rank 1/2/3 的人。
    group_start() / group_end() 的作用是把这些调用从"立即执行"改成"先收集，最后一起提交"：
    - group_start()：进入组模式，之后的 NCCL 调用只记录请求，不真正初始化
    - 循环里的 4 次 ncclCommInitRank 被累积进 group，不再阻塞线程
    - group_end()：NCCL 把 4 个 comm 作为一个整体一起初始化，collective 握手在 4 个请求都到齐后自然完成，不死锁
    所以这里的规则是：同一线程内涉及多个 rank 的 NCCL 调用，要么用 group 包裹，要么每个 rank 开一个线程（我最初的版本就是每 rank 一线程，
    每个线程自己阻塞在自己的 init 上，并行完成握手）。后者可行但更脆弱（同一 GPU 上多线程 init 之前观察到会偶发挂死），所以最终统一用单线程 + group 的方案。
     */
    group_start().unwrap();
    let comms: Vec<_> = (0..WORLD_SIZE)
        .map(|rank| Comm::from_rank(streams[rank].clone(), rank, WORLD_SIZE, id).unwrap())
        .collect();
    group_end().unwrap();

    (streams, comms)
}

#[test]
fn test_allreduce() {
    let _guard = lock_nccl();
    enable_multi_rank_on_single_gpu();

    // 这个 ID 内部其实封装了 Master 节点监听的 IP 地址和 Socket 端口
    // 用于集群启动时的握手与建组，而 rank 是组内的标识
    let id = Id::new().unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let (streams, comms) = setup_clique(&ctx, id);

    // Each rank contributes [rank + 1; NUMEL] on its own stream.
    let inputs: Vec<CudaSlice<f32>> = (0..WORLD_SIZE)
        .map(|rank| {
            let data = vec![(rank + 1) as f32; NUMEL];
            streams[rank].clone_htod(&data).unwrap()
        })
        .collect();
    let mut outs: Vec<CudaSlice<f32>> = (0..WORLD_SIZE)
        .map(|i| streams[i].alloc_zeros::<f32>(NUMEL).unwrap())
        .collect();

    group_start().unwrap();
    for (i, comm) in comms.iter().enumerate() {
        comm.all_reduce(&inputs[i], &mut outs[i], &ReduceOp::Sum)
            .unwrap();
    }
    group_end().unwrap();

    // Every rank sees the same reduced result: 1 + 2 + 3 + 4 = 10 per element.
    let expected = vec![10.0f32; NUMEL];
    for (i, out) in outs.iter().enumerate() {
        assert_eq!(streams[i].clone_dtoh(out).unwrap(), expected);
    }
}

#[test]
fn test_allgather() {
    let _guard = lock_nccl();
    enable_multi_rank_on_single_gpu();

    let id = Id::new().unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let (streams, comms) = setup_clique(&ctx, id);

    // Each rank contributes [rank + 1; NUMEL]; after allgather every rank holds
    // the concatenation of all ranks' contributions.
    let inputs: Vec<CudaSlice<f32>> = (0..WORLD_SIZE)
        .map(|rank| {
            let data = vec![(rank + 1) as f32; NUMEL];
            streams[rank].clone_htod(&data).unwrap()
        })
        .collect();
    let mut outs: Vec<CudaSlice<f32>> = (0..WORLD_SIZE)
        .map(|i| streams[i].alloc_zeros::<f32>(WORLD_SIZE * NUMEL).unwrap())
        .collect();

    group_start().unwrap();
    for (i, comm) in comms.iter().enumerate() {
        comm.all_gather(&inputs[i], &mut outs[i]).unwrap();
    }
    group_end().unwrap();

    // Order follows rank order: [1;N] [2;N] [3;N] [4;N].
    let expected: Vec<f32> = (0..WORLD_SIZE)
        .flat_map(|r| vec![(r + 1) as f32; NUMEL])
        .collect();
    for (i, out) in outs.iter().enumerate() {
        assert_eq!(streams[i].clone_dtoh(out).unwrap(), expected);
    }
}

#[test]
fn test_reducescatter() {
    let _guard = lock_nccl();
    enable_multi_rank_on_single_gpu();

    let id = Id::new().unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let (streams, comms) = setup_clique(&ctx, id);

    // sendbuff must be WORLD_SIZE * recvcount long. Each rank contributes
    // [rank + 1; WORLD_SIZE * NUMEL]; the element-wise sum over ranks is
    // 1+2+3+4 = 10, and rank j keeps chunk j of the reduced buffer.
    let inputs: Vec<CudaSlice<f32>> = (0..WORLD_SIZE)
        .map(|rank| {
            let data = vec![(rank + 1) as f32; WORLD_SIZE * NUMEL];
            streams[rank].clone_htod(&data).unwrap()
        })
        .collect();
    let mut outs: Vec<CudaSlice<f32>> = (0..WORLD_SIZE)
        .map(|i| streams[i].alloc_zeros::<f32>(NUMEL).unwrap())
        .collect();

    group_start().unwrap();
    for (i, comm) in comms.iter().enumerate() {
        comm.reduce_scatter(&inputs[i], &mut outs[i], &ReduceOp::Sum)
            .unwrap();
    }
    group_end().unwrap();

    // Every rank ends up with [10; NUMEL]: 1 + 2 + 3 + 4 per element.
    let expected = vec![10.0f32; NUMEL];
    for (i, out) in outs.iter().enumerate() {
        assert_eq!(streams[i].clone_dtoh(out).unwrap(), expected);
    }
}

#[test]
fn test_broadcast_reduce() {
    let _guard = lock_nccl();
    enable_multi_rank_on_single_gpu();

    let id = Id::new().unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let (streams, comms) = setup_clique(&ctx, id);
    const ROOT: usize = 0;

    // broadcast: root seeds [7.0; NUMEL], others zeros; in-place broadcast makes
    // every rank hold [7.0; NUMEL].
    let mut bcast_devs: Vec<CudaSlice<f32>> = (0..WORLD_SIZE)
        .map(|rank| {
            if rank == ROOT {
                streams[rank].clone_htod(&vec![7.0f32; NUMEL]).unwrap()
            } else {
                streams[rank].alloc_zeros::<f32>(NUMEL).unwrap()
            }
        })
        .collect();
    group_start().unwrap();
    for (i, comm) in comms.iter().enumerate() {
        comm.broadcast_in_place(&mut bcast_devs[i], ROOT as i32)
            .unwrap();
    }
    group_end().unwrap();

    // reduce: every rank contributes [rank + 1; NUMEL], root keeps the sum.
    let inputs: Vec<CudaSlice<f32>> = (0..WORLD_SIZE)
        .map(|rank| {
            let data = vec![(rank + 1) as f32; NUMEL];
            streams[rank].clone_htod(&data).unwrap()
        })
        .collect();
    let mut reduce_outs: Vec<Option<CudaSlice<f32>>> = (0..WORLD_SIZE)
        .map(|rank| {
            if rank == ROOT {
                Some(streams[rank].alloc_zeros::<f32>(NUMEL).unwrap())
            } else {
                None
            }
        })
        .collect();
    group_start().unwrap();
    for (i, comm) in comms.iter().enumerate() {
        let recvbuff = reduce_outs[i].as_mut();
        comm.reduce(&inputs[i], recvbuff, &ReduceOp::Sum, ROOT as i32)
            .unwrap();
    }
    group_end().unwrap();

    for (rank, handle) in streams.iter().enumerate() {
        let bcast_out = handle.clone_dtoh(&bcast_devs[rank]).unwrap();
        assert_eq!(bcast_out, vec![7.0f32; NUMEL]);
        if rank == ROOT {
            let reduce_out = handle
                .clone_dtoh(reduce_outs[rank].as_ref().unwrap())
                .unwrap();
            assert_eq!(reduce_out, vec![10.0f32; NUMEL]);
        }
    }
}

#[test]
fn test_alltoall() {
    let _guard = lock_nccl();
    enable_multi_rank_on_single_gpu();

    let id = Id::new().unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let (streams, comms) = setup_clique(&ctx, id);

    // Rank i sends value (i*W + j + 1) to peer j. NCCL has no native alltoall,
    // so it is built from send/recv pairs; all P2P ops are enqueued together in a
    // group so matching pairs can't deadlock.
    let send_devs: Vec<CudaSlice<f32>> = (0..WORLD_SIZE)
        .map(|rank| {
            let data: Vec<f32> = (0..WORLD_SIZE)
                .flat_map(|peer| vec![(rank * WORLD_SIZE + peer + 1) as f32; NUMEL])
                .collect();
            streams[rank].clone_htod(&data).unwrap()
        })
        .collect();
    let mut recv_devs: Vec<CudaSlice<f32>> = (0..WORLD_SIZE)
        .map(|i| streams[i].alloc_zeros::<f32>(WORLD_SIZE * NUMEL).unwrap())
        .collect();

    group_start().unwrap();
    // ncclAlltoAll is available since NCCL 2.28
    /*
    在许多主流深度学习框架（如 PyTorch、Megatron-LM、DeepSpeed）的底层代码中，你可能会发现很多 AllToAll 逻辑依然是写成了一个 for 循环调用 ncclSend 和 ncclRecv 的形式，
    这是因为：如果使用 ncclAllToAll 进行全互连，虽然看起来一次操作就完成了所有通信，但实际上内部会自动拆分成多个 ncclSend 和 ncclRecv 操作，
    每个操作只涉及两个 rank 之间的通信。这样做的好处是：
    - 灵活性：可以按需选择通信的 rank 对，不需要提前知道所有 rank 的连接关系
    - 性能：可以按需控制通信的粒度，比如只和某些特定的 rank 通信
    - 可扩展性：可以方便地扩展到更大的规模，只需要增加 rank 的数量，不需要修改代码
    - 兼容性：可以方便地与其他框架进行交互，比如与其他框架的 AllToAll 操作进行组合
    - 可读性：可以方便地进行调试和优化，比如可以方便地进行通信的 profiler 分析
    - 可移植性：可以方便地进行移植，比如可以方便地移植到其他平台，比如 CPU 平台
     */
    for (i, comm) in comms.iter().enumerate() {
        for peer in 0..WORLD_SIZE {
            let send_view = send_devs[i].slice(peer * NUMEL..(peer + 1) * NUMEL);

            // send to one peer at a time
            comm.send(&send_view, peer as i32).unwrap();

            let mut recv_view = recv_devs[i].slice_mut(peer * NUMEL..(peer + 1) * NUMEL);
            // recv from one peer at a time
            comm.recv(&mut recv_view, peer as i32).unwrap();
        }
    }
    group_end().unwrap();

    // Rank j receives, in sender order, value (i*W + j + 1) from each rank i.
    for (rank, stream) in streams.iter().enumerate() {
        // no need for group ops here, there is no nccl collective operation here
        let out = stream.clone_dtoh(&recv_devs[rank]).unwrap();
        let expected: Vec<f32> = (0..WORLD_SIZE)
            .flat_map(|sender| vec![(sender * WORLD_SIZE + rank + 1) as f32; NUMEL])
            .collect();
        assert_eq!(out, expected);
    }
}

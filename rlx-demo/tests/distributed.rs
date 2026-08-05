//! Multi-GPU / multi-node distributed tensor add.
//!
//! Each rank owns a shard of `x` and `y`, computes the local elementwise add,
//! zero-pads that shard into a full-length buffer, then `all_reduce`s (sum) so
//! every rank ends up with the concatenated result.
//!
//! Communication backends shown here:
//! - single-node multi-GPU → NCCL (`test_distributed_nccl_sharded_add`)
//! - multi-node GPUs → `Node` mesh/star bootstrap, then NCCL id exchange over
//!   the host `ProcessGroup`, then on-device NCCL
//!   (`test_distributed_multinode_nccl_sharded_add`)
//! - NVSHMEM → **not shipped yet**; `test_symmetric_transport_standin_for_nvshmem`
//!   demos the `SymmetricTransport` put/get/barrier surface that a future
//!   `NvshmemTransport` is designed to plug into.

use rlx::distributed::{
    NetTransport, Node, ProcessGroup, Topology, all_reduce, register, register_group,
    unregister_group,
};
use rlx::prelude::*;
// Symmetric-memory collective surface (the seam NVSHMEM will implement).
use rlx_driver::{
    LocalTransport, Rank, ReduceKind as SymReduceKind, SymmetricBuffer, SymmetricTransport,
    all_reduce as symmetric_all_reduce,
};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::thread;

/// Tag used to ship the 128-byte NCCL unique id over the host ProcessGroup.
const TAG_NCCL_ID: u32 = 9_001;

const GROUP_ID: u64 = 42;
const WORLD: usize = 2;
const SHARD: usize = 4;
const FULL: usize = WORLD * SHARD; // 8

/// Global tensors split evenly across ranks:
///   x = [1,2,3,4, 5,6,7,8]
///   y = [10,20,30,40, 50,60,70,80]
/// so the full add is [11,22,33,44, 55,66,77,88].
fn global_x() -> [f32; FULL] {
    [1., 2., 3., 4., 5., 6., 7., 8.]
}
fn global_y() -> [f32; FULL] {
    [10., 20., 30., 40., 50., 60., 70., 80.]
}
fn expected_sum() -> [f32; FULL] {
    [11., 22., 33., 44., 55., 66., 77., 88.]
}

fn shard_of(full: &[f32], rank: usize) -> Vec<f32> {
    let start = rank * SHARD;
    full[start..start + SHARD].to_vec()
}

/// Build `all_reduce(pad(x_shard + y_shard))`. Padding with zeros makes the
/// sum-reduction act like an all-gather of the non-overlapping shards.
fn build_sharded_add_graph(rank: usize, world: usize, group_id: u64) -> Graph {
    let mut g = Graph::new("dist_add");
    let x = g.input("x", Shape::new(&[SHARD], DType::F32));
    let y = g.input("y", Shape::new(&[SHARD], DType::F32));
    let local = g.add(x, y);

    let mut parts = Vec::new();
    if rank > 0 {
        parts.push(g.zeros(&[rank * SHARD], DType::F32));
    }
    parts.push(local);
    let after = (world - rank - 1) * SHARD;
    if after > 0 {
        parts.push(g.zeros(&[after], DType::F32));
    }
    let padded = if parts.len() == 1 {
        parts[0]
    } else {
        g.concat_(parts, 0)
    };

    let out = all_reduce(&mut g, padded, group_id);
    g.set_outputs(vec![out]);
    g
}

/// NCCL worker entry: one OS process per GPU (`CUDA_VISIBLE_DEVICES` pins the
/// ordinal so `Device::Cuda` / `CudaContext::new(0)` hit distinct devices).
fn run_nccl_worker(rank: usize, world: usize, id_file: &std::path::Path) {
    register();

    let ctx = rlx_cuda::device::cuda_context().expect("CUDA context");
    let stream = ctx.default_stream();

    // Rank 0 already wrote the id; every rank reads the same bootstrap bytes.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let id = loop {
        if let Ok(bytes) = std::fs::read(id_file) {
            if bytes.len() == 128 {
                let mut arr = [0u8; 128];
                arr.copy_from_slice(&bytes);
                break rlx_cuda::distributed::id_from_bytes(&arr);
            }
        }
        if std::time::Instant::now() > deadline {
            panic!("timeout waiting for NCCL id at {}", id_file.display());
        }
        thread::sleep(std::time::Duration::from_millis(20));
    };

    rlx_cuda::distributed::init_and_register(GROUP_ID, stream, rank, world, id)
        .expect("NCCL init_and_register");

    let g = build_sharded_add_graph(rank, world, GROUP_ID);
    let mut compiled = Session::new(Device::Cuda).compile(g);
    let x = shard_of(&global_x(), rank);
    let y = shard_of(&global_y(), rank);
    let out = compiled.run(&[("x", x.as_slice()), ("y", y.as_slice())]);

    println!(
        "nccl rank {rank}/{world} gpu={}  shard_x={x:?} shard_y={y:?}  out={:?}",
        std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
        out[0]
    );
    assert_eq!(out[0], expected_sum());

    rlx_cuda::distributed::unregister_nccl_comm(GROUP_ID);
}

/// Parent: write the NCCL unique id, spawn one child process per GPU.
fn run_nccl_parent(world: usize) {
    let id = rlx_cuda::distributed::new_nccl_id().expect("new_nccl_id (is libnccl installed?)");
    let id_file =
        std::env::temp_dir().join(format!("rlx-usage-nccl-id-{}.bin", std::process::id()));
    std::fs::write(&id_file, rlx_cuda::distributed::id_to_bytes(&id)).unwrap();

    let exe = std::env::current_exe().expect("current_exe");
    let mut children = Vec::with_capacity(world);
    for rank in 0..world {
        let child = std::process::Command::new(&exe)
            .args([
                "--exact",
                "test_distributed_nccl_sharded_add",
                "--nocapture",
            ])
            .env("RLX_DIST_WORKER", "1")
            .env("RLX_DIST_RANK", rank.to_string())
            .env("RLX_DIST_WORLD", world.to_string())
            .env("RLX_DIST_ID_FILE", &id_file)
            .env("CUDA_VISIBLE_DEVICES", rank.to_string())
            .spawn()
            .unwrap_or_else(|e| panic!("spawn rank {rank}: {e}"));
        children.push(child);
    }

    for (rank, mut child) in children.into_iter().enumerate() {
        let status = child
            .wait()
            .unwrap_or_else(|e| panic!("wait rank {rank}: {e}"));
        assert!(status.success(), "NCCL worker rank {rank} failed: {status}");
    }
    let _ = std::fs::remove_file(id_file);
}

/// Multi-GPU NCCL path: requires ≥2 NVIDIA GPUs. Skips cleanly otherwise.
#[test]
fn test_distributed_nccl_sharded_add() {
    if let Ok(rank) = std::env::var("RLX_DIST_RANK") {
        let rank: usize = rank.parse().unwrap();
        let world: usize = std::env::var("RLX_DIST_WORLD").unwrap().parse().unwrap();
        let id_file = std::env::var("RLX_DIST_ID_FILE").unwrap();
        assert_eq!(std::env::var("RLX_DIST_WORKER").ok().as_deref(), Some("1"));
        run_nccl_worker(rank, world, std::path::Path::new(&id_file));
        return;
    }

    if !rlx_cuda::is_available() {
        eprintln!("skip test_distributed_nccl_sharded_add: CUDA unavailable");
        return;
    }
    let n_gpu = rlx_cuda::nvml::device_count();
    if n_gpu < WORLD {
        eprintln!(
            "skip test_distributed_nccl_sharded_add: need ≥{WORLD} GPUs for NCCL, found {n_gpu}"
        );
        return;
    }

    run_nccl_parent(WORLD);
}

/// Same sharded-add graph, two ranks in-process over a TCP `ProcessGroup`.
/// Runs on a single GPU (host collective fallback) so the distributed graph
/// is always exercised even without a second device.
#[test]
fn test_distributed_sharded_add_host_group() {
    if !rlx_cuda::is_available() {
        eprintln!("skip test_distributed_sharded_add_host_group: CUDA unavailable");
        return;
    }
    register();

    let listeners: Vec<TcpListener> = (0..WORLD)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let addrs = addrs.clone();
            thread::spawn(move || {
                let t = NetTransport::from_listener(
                    rank as u32,
                    WORLD as u32,
                    listener,
                    addrs,
                    1 << 20,
                )
                .unwrap();
                // Local registry key (per-rank); the ProcessGroup itself knows the peers.
                let gid = 1_000 + rank as u64;
                register_group(gid, Arc::new(ProcessGroup::new(Arc::new(t))));

                let g = build_sharded_add_graph(rank, WORLD, gid);
                let mut compiled = Session::new(Device::Cuda).compile(g);
                let x = shard_of(&global_x(), rank);
                let y = shard_of(&global_y(), rank);
                let out = compiled.run(&[("x", x.as_slice()), ("y", y.as_slice())]);

                println!(
                    "host rank {rank}/{WORLD}  shard_x={x:?} shard_y={y:?}  out={:?}",
                    out[0]
                );
                unregister_group(gid);
                out.into_iter().next().unwrap()
            })
        })
        .collect();

    for (rank, h) in handles.into_iter().enumerate() {
        let y = h.join().unwrap();
        assert_eq!(y, expected_sum(), "rank {rank}");
    }
}

// ── Multi-node NCCL ───────────────────────────────────────────────────────────
//
// Real two-box launch (one GPU per node):
//
//   # node A (10.0.0.1)
//   RANK=0 WORLD=2 PEERS=10.0.0.1:29500,10.0.0.2:29501 TOPOLOGY=mesh \
//     cargo test --test distributed test_distributed_multinode_nccl_sharded_add \
//       -- --exact --nocapture
//
//   # node B (10.0.0.2)
//   RANK=1 WORLD=2 PEERS=10.0.0.1:29500,10.0.0.2:29501 TOPOLOGY=mesh \
//     cargo test --test distributed test_distributed_multinode_nccl_sharded_add \
//       -- --exact --nocapture
//
// Across NAT / no open ports, swap the transport for iroh
// (`TOPOLOGY=iroh RLX_IROH_SEED=…`, needs the `iroh` feature on rlx-driver).
//
// Flow per rank:
//   1. `Node::from_env()?.connect()?`  → host ProcessGroup (TCP / iroh / …)
//   2. Rank 0 creates an NCCL id and `send_bytes`s it to every peer
//   3. `init_and_register` builds the NCCL communicator on this node's GPU
//   4. Same sharded-add graph; `collective.all_reduce` rides NCCL on-device

/// Bootstrap NCCL across nodes: exchange the unique id over the already-
/// connected host [`ProcessGroup`], then register the device communicator.
fn bootstrap_nccl_via_process_group(
    group: &ProcessGroup,
    rank: usize,
    world: usize,
    group_id: u64,
) -> Result<(), String> {
    let ctx = rlx_cuda::device::cuda_context().ok_or("CUDA context unavailable")?;
    let stream = ctx.default_stream();

    let id = if rank == 0 {
        let id = rlx_cuda::distributed::new_nccl_id()?;
        let bytes = rlx_cuda::distributed::id_to_bytes(&id);
        for peer in 1..world as u32 {
            group
                .transport()
                .send_bytes(peer, TAG_NCCL_ID, &bytes)
                .map_err(|e| format!("send NCCL id →{peer}: {e}"))?;
        }
        id
    } else {
        let bytes = group
            .transport()
            .recv_bytes(0, TAG_NCCL_ID)
            .map_err(|e| format!("recv NCCL id ←0: {e}"))?;
        if bytes.len() != 128 {
            return Err(format!("NCCL id length {}, expected 128", bytes.len()));
        }
        let mut arr = [0u8; 128];
        arr.copy_from_slice(&bytes);
        rlx_cuda::distributed::id_from_bytes(&arr)
    };

    rlx_cuda::distributed::init_and_register(group_id, stream, rank, world, id)?;
    Ok(())
}

fn run_multinode_nccl_worker(rank: usize, world: usize) {
    register();

    // Join the multi-node mesh (RANK / WORLD / PEERS / TOPOLOGY from env).
    let group = Node::from_env()
        .unwrap_or_else(|e| panic!("Node::from_env: {e}"))
        .connect()
        .unwrap_or_else(|e| panic!("Node::connect: {e}"));

    bootstrap_nccl_via_process_group(&group, rank, world, GROUP_ID)
        .expect("multi-node NCCL bootstrap");

    // In-graph collective registry key (local to this process).
    let gid = GROUP_ID;
    register_group(gid, group.clone());

    let g = build_sharded_add_graph(rank, world, gid);
    let mut compiled = Session::new(Device::Cuda).compile(g);
    let x = shard_of(&global_x(), rank);
    let y = shard_of(&global_y(), rank);
    let out = compiled.run(&[("x", x.as_slice()), ("y", y.as_slice())]);

    println!(
        "multinode-nccl rank {rank}/{world}  shard_x={x:?} shard_y={y:?}  out={:?}",
        out[0]
    );
    assert_eq!(out[0], expected_sum());

    unregister_group(gid);
    rlx_cuda::distributed::unregister_nccl_comm(GROUP_ID);
}

fn free_tcp_port() -> u16 {
    TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Local rehearsal: spawn WORLD processes on localhost as if they were nodes.
/// Uses a TCP mesh for the host ProcessGroup; enables NCCL only when enough
/// distinct GPUs are visible (`CUDA_VISIBLE_DEVICES` per child).
fn run_multinode_parent_local(n_gpu: usize) {
    let ports: Vec<u16> = (0..WORLD).map(|_| free_tcp_port()).collect();
    let peers: String = ports
        .iter()
        .map(|p| format!("127.0.0.1:{p}"))
        .collect::<Vec<_>>()
        .join(",");

    let exe = std::env::current_exe().expect("current_exe");
    let use_nccl = n_gpu >= WORLD;
    let mut children = Vec::with_capacity(WORLD);
    for rank in 0..WORLD {
        let mut cmd = std::process::Command::new(&exe);
        cmd.args([
            "--exact",
            "test_distributed_multinode_nccl_sharded_add",
            "--nocapture",
        ])
        .env("RLX_MULTINODE_WORKER", "1")
        .env("RANK", rank.to_string())
        .env("WORLD", WORLD.to_string())
        .env("PEERS", &peers)
        .env("TOPOLOGY", "mesh")
        .env("RLX_MULTINODE_USE_NCCL", if use_nccl { "1" } else { "0" });
        if use_nccl {
            cmd.env("CUDA_VISIBLE_DEVICES", rank.to_string());
        }
        children.push(
            cmd.spawn()
                .unwrap_or_else(|e| panic!("spawn multinode rank {rank}: {e}")),
        );
    }

    for (rank, mut child) in children.into_iter().enumerate() {
        let status = child
            .wait()
            .unwrap_or_else(|e| panic!("wait multinode rank {rank}: {e}"));
        assert!(
            status.success(),
            "multinode worker rank {rank} failed: {status}"
        );
    }
}

/// Worker used both for real multi-node launches and the localhost rehearsal.
///
/// When `RLX_MULTINODE_USE_NCCL=0` (single-GPU machine), the same graph runs
/// over the host ProcessGroup only — still validates the multi-node wiring
/// (`Node::from_env` → id exchange shape → sharded add).
fn run_multinode_worker_entry() {
    let rank: usize = std::env::var("RANK").unwrap().parse().unwrap();
    let world: usize = std::env::var("WORLD").unwrap().parse().unwrap();
    let use_nccl = std::env::var("RLX_MULTINODE_USE_NCCL").ok().as_deref() != Some("0");

    register();
    let group = Node::from_env()
        .unwrap_or_else(|e| panic!("Node::from_env: {e}"))
        .connect()
        .unwrap_or_else(|e| panic!("Node::connect: {e}"));

    // Always exercise the NCCL-id exchange path over the host group — even
    // when we won't init NCCL — so the multi-node bootstrap code is covered.
    if use_nccl && rlx_cuda::is_available() {
        bootstrap_nccl_via_process_group(&group, rank, world, GROUP_ID)
            .expect("multi-node NCCL bootstrap");
    } else {
        // Lightweight stand-in: rank 0 broadcasts a dummy 128-byte payload so
        // the send_bytes/recv_bytes wiring is still checked without libnccl.
        let payload = [7u8; 128];
        if rank == 0 {
            for peer in 1..world as u32 {
                group
                    .transport()
                    .send_bytes(peer, TAG_NCCL_ID, &payload)
                    .unwrap();
            }
        } else {
            let got = group.transport().recv_bytes(0, TAG_NCCL_ID).unwrap();
            assert_eq!(got, payload);
        }
        println!("multinode rank {rank}/{world}: host ProcessGroup bootstrap ok (NCCL skipped)");
    }

    // NCCL looks up communicators by the group_id baked into the op attrs, so
    // when NCCL is active every rank must use the same id. Host-only ranks can
    // use a per-process key (the ProcessGroup handle is local anyway).
    let gid = if use_nccl {
        GROUP_ID
    } else {
        2_000 + rank as u64
    };
    register_group(gid, group);

    let g = build_sharded_add_graph(rank, world, gid);
    let device = if rlx_cuda::is_available() {
        Device::Cuda
    } else {
        Device::Cpu
    };
    let mut compiled = Session::new(device).compile(g);
    let x = shard_of(&global_x(), rank);
    let y = shard_of(&global_y(), rank);
    let out = compiled.run(&[("x", x.as_slice()), ("y", y.as_slice())]);

    println!(
        "multinode rank {rank}/{world}  shard_x={x:?} shard_y={y:?}  out={:?}",
        out[0]
    );
    assert_eq!(out[0], expected_sum());

    unregister_group(gid);
    if use_nccl {
        rlx_cuda::distributed::unregister_nccl_comm(GROUP_ID);
    }
}

/// Multi-node NCCL sharded add.
///
/// - With `RLX_MULTINODE_WORKER=1` (+ `RANK`/`WORLD`/`PEERS`): runs one node.
/// - Otherwise: localhost rehearsal spawning 2 processes. Uses NCCL when
///   ≥2 GPUs are present; otherwise host ProcessGroup only.
#[test]
fn test_distributed_multinode_nccl_sharded_add() {
    if std::env::var("RLX_MULTINODE_WORKER").ok().as_deref() == Some("1") {
        run_multinode_worker_entry();
        return;
    }

    // Manual multi-node: user exported RANK/WORLD/PEERS and runs this binary
    // once per box. Detect by RANK being set without our worker flag from a
    // previous convention — prefer explicit worker flag; for convenience also
    // accept a direct Node::from_env launch when RLX_MULTINODE=1.
    if std::env::var("RLX_MULTINODE").ok().as_deref() == Some("1") {
        run_multinode_nccl_worker(
            std::env::var("RANK").unwrap().parse().unwrap(),
            std::env::var("WORLD").unwrap().parse().unwrap(),
        );
        return;
    }

    let n_gpu = if rlx_cuda::is_available() {
        rlx_cuda::nvml::device_count()
    } else {
        0
    };
    if n_gpu < WORLD {
        eprintln!(
            "test_distributed_multinode_nccl_sharded_add: localhost rehearsal with host \
             ProcessGroup (found {n_gpu} GPU(s); NCCL needs ≥{WORLD})"
        );
    }
    run_multinode_parent_local(n_gpu);
}

// ── NVSHMEM stand-in (SymmetricTransport) ─────────────────────────────────────
//
// rlx does **not** ship an `NvshmemTransport` yet (see
// `crates/core/rlx-collectives/DISTRIBUTED_ROADMAP.md` Tier 3). The planned
// drop-in implements the existing [`SymmetricTransport`] trait with
// `nvshmem_putmem` / `nvshmem_getmem` / `nvshmem_barrier` on the device
// symmetric heap — same three methods `LocalTransport` / `NetTransport`
// already satisfy.
//
// This test demos that surface: each "rank" owns a shard of two tensors,
// writes the local sum into its symmetric slot via `put`, then
// `symmetric_all_reduce` gathers every slot with one-sided `get`s.

/// Demo of the one-sided symmetric-memory API that NVSHMEM will implement.
/// Uses in-process [`LocalTransport`] as today's stand-in.
#[test]
fn test_symmetric_transport_standin_for_nvshmem() {
    // When NvshmemTransport lands, swap only the constructor:
    //   let transports = NvshmemTransport::fan_out(WORLD as u32, heap);
    // The put / get / barrier / all_reduce calls below stay identical.
    let transports = LocalTransport::fan_out(WORLD as u32, FULL * 4 /* f32 bytes */);

    let handles: Vec<_> = transports
        .into_iter()
        .enumerate()
        .map(|(rank, t)| {
            thread::spawn(move || {
                // Local shard add — the "tensor computation" on this rank.
                let x = shard_of(&global_x(), rank);
                let y = shard_of(&global_y(), rank);
                let local: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();

                // Zero-pad into a full-length contribution so Sum ≡ all-gather.
                let mut contrib = vec![0.0_f32; FULL];
                let start = rank * SHARD;
                contrib[start..start + SHARD].copy_from_slice(&local);

                let slot = SymmetricBuffer {
                    rank: Rank(rank as u32),
                    offset: 0,
                    len: FULL * 4,
                };
                // One-sided write of our contribution into the symmetric heap.
                let bytes = unsafe {
                    std::slice::from_raw_parts(contrib.as_ptr() as *const u8, slot.len)
                };
                t.put(slot, bytes).unwrap();
                t.barrier().unwrap();

                // Collective over the symmetric heap (put/get/barrier under the hood).
                let buf = SymmetricBuffer {
                    rank: t.this_rank(), // offset/len matter; rank is rewritten inside
                    offset: 0,
                    len: FULL * 4,
                };
                symmetric_all_reduce(&t, buf, &mut contrib, SymReduceKind::Sum).unwrap();

                println!(
                    "symmetric/nvshmem-standin rank {rank}/{WORLD}  local_add={local:?}  reduced={contrib:?}"
                );
                assert_eq!(contrib, expected_sum());
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

/// Documents the multi-node builder API without requiring a second box.
#[test]
fn test_multinode_node_builder_api() {
    // Explicit builder (same thing `Node::from_env` constructs from RANK/WORLD/PEERS):
    let node = Node::new(0, 2)
        .topology(Topology::Mesh)
        .peers(["127.0.0.1:29500", "127.0.0.1:29501"])
        .expect("peers");
    assert_eq!(node.rank(), 0);
    assert_eq!(node.world(), 2);

    // Star topology: only the coordinator address is required; workers dial out
    // (NAT-friendly — no inbound port on the worker).
    let star = Node::new(1, 2)
        .topology(Topology::Star)
        .peers(["10.0.0.1:29500"])
        .expect("coord");
    assert_eq!(star.rank(), 1);
}

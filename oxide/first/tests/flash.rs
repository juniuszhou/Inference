use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use first::flash::flash;

fn reference_attention(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut o = vec![0.0f32; n * d];

    for i in 0..n {
        let mut scores = vec![0.0f32; n];
        let mut max_score = -f32::INFINITY;
        for j in 0..n {
            let mut s = 0.0f32;
            for x in 0..d {
                s += q[i * d + x] * k[j * d + x];
            }
            s *= scale;
            scores[j] = s;
            if s > max_score {
                max_score = s;
            }
        }

        let mut sum_exp = 0.0f32;
        for j in 0..n {
            scores[j] = (scores[j] - max_score).exp();
            sum_exp += scores[j];
        }

        for x in 0..d {
            let mut val = 0.0f32;
            for j in 0..n {
                val += scores[j] * v[j * d + x];
            }
            o[i * d + x] = val / sum_exp;
        }
    }

    o
}

#[test]
fn test_flash_attention() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const B: usize = 1;
    const NH: usize = 1;
    const N: usize = 32;
    const D: usize = 8;
    const BC: usize = 32;
    const BR: usize = 32;

    let tc = ((N as f32) / (BC as f32)).ceil() as i32;
    let tr = ((N as f32) / (BR as f32)).ceil() as i32;
    let softmax_scale = 1.0f32 / (D as f32).sqrt();

    let q_host: Vec<f32> = (0..B * NH * N * D).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let k_host: Vec<f32> = (0..B * NH * N * D).map(|i| (i as f32) * 0.2 - 2.0).collect();
    let v_host: Vec<f32> = (0..B * NH * N * D).map(|i| (i as f32) * 0.3 - 0.5).collect();

    let q_dev = DeviceBuffer::from_host(&stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v_host).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N * D).unwrap();
    let mut l_dev = DeviceBuffer::<f32>::zeroed(&stream, B * NH * N).unwrap();
    let mut m_dev =
        DeviceBuffer::<f32>::from_host(&stream, &vec![-f32::INFINITY; B * NH * N]).unwrap();

    let tile_size = BC * D;
    let sram_size = (3 * tile_size + BC * BC) * std::mem::size_of::<f32>();

    let module = flash::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: one block per (batch, head) with BC threads; each thread owns
    // row `tx` of every tile, so all raw-index writes to l, m and O are
    // disjoint, and the buffers cover the kernel's accesses.
    unsafe {
        module.flash_attention(
            &stream,
            LaunchConfig {
                grid_dim: (B as u32, NH as u32, 1),
                block_dim: (BC as u32, 1, 1),
                shared_mem_bytes: sram_size as u32,
            },
            &q_dev,
            &k_dev,
            &v_dev,
            N as i32,
            D as i32,
            tc,
            tr,
            BC as i32,
            BR as i32,
            softmax_scale,
            &mut l_dev,
            &mut m_dev,
            &mut o_dev,
        )
    }
    .expect("Kernel launch failed");

    let o_host = o_dev.to_host_vec(&stream).unwrap();

    let reference = reference_attention(&q_host, &k_host, &v_host, N, D);
    let mut max_err = 0.0f32;
    for i in 0..N * D {
        let err = (o_host[i] - reference[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < 1e-4,
        "Max error {} exceeds tolerance 1e-4",
        max_err
    );
    println!("PASSED: flash attention test (max_err={})", max_err);
}

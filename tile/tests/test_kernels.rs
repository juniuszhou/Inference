use cutile::half::f16;
use cutile::prelude::*;
use std::sync::Arc;
use tile::kernels;

const TEST_STACK_SIZE: usize = 8_000_000;

fn with_test_stack<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(TEST_STACK_SIZE)
        .spawn(f)
        .expect("Failed to spawn test thread")
        .join()
        .expect("Test thread panicked")
}

#[test]
fn test_gemm_f16() {
    with_test_stack(|| {
        let (m, n, k) = (1024usize, 1024usize, 1024usize);
        // Tile sizes must divide M/N/K. These match cutile's GEMM examples;
        // BM=BN=BK=1024 would be a single oversized MMA tile.
        let (bm, bn, bk) = (16usize, 16usize, 8usize);
        let generics = vec![
            bm.to_string(),
            bn.to_string(),
            bk.to_string(),
            k.to_string(),
        ];

        let mut x_data = vec![f16::ZERO; m * k];
        let mut y_data = vec![f16::ZERO; k * n];
        for i in 0..1024 {
            x_data[i * k + i] = f16::from_f32(1.0);
            y_data[i * n + i] = f16::from_f32(2.0);
        }

        let x: Arc<Tensor<f16>> = api::copy_host_vec_to_device(&Arc::new(x_data))
            .sync()
            .expect("x h2d")
            .reshape(&[m, k])
            .expect("reshape x")
            .into();
        let y: Arc<Tensor<f16>> = api::copy_host_vec_to_device(&Arc::new(y_data))
            .sync()
            .expect("y h2d")
            .reshape(&[k, n])
            .expect("reshape y")
            .into();
        let z = api::copy_host_vec_to_device(&Arc::new(vec![f16::ZERO; m * n]))
            .sync()
            .expect("z h2d")
            .reshape(&[m, n])
            .expect("reshape z")
            .partition([bm, bn]);

        let (z, _, _) = unsafe { kernels::gemm_f16(z, x, y) }
            .generics(generics)
            .sync()
            .expect("gemm_f16");

        let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
        for i in 0..1024 {
            assert_eq!(z_host[i * 1024 + i].to_f32(), 2.0, "diagonal element {}", i);
            for j in 0..1024 {
                if i != j {
                    assert_eq!(
                        z_host[i * 1024 + j].to_f32(),
                        0.0,
                        "off-diagonal element [{}, {}]",
                        i,
                        j
                    );
                }
            }
        }
    });
}

use cutile::prelude::*;
use std::sync::Arc;
use tile::kernels;

#[test]
fn test_sum_1d() {
    let x = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1]).partition([1]);
    let (z, _) = kernels::sum_1d(z, x).sync().expect("sum_1d");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
    assert_eq!(z_host, vec![1024.0]);
}

#[test]
fn test_sum_1d_p() {
    let x = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1]).partition([1]);
    let (z, _) = kernels::sum_1d_p(z, x).sync().expect("sum_1d");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
    assert_eq!(z_host, vec![1024.0]);
}

// 二阶段规约测试：第一阶段每 CTA 求一个 Tile[32] 的和，第二阶段归约 partial -> 标量
#[test]
fn test_sum_two_stage_ones() {
    let x = api::ones::<f32>(&[1024]);
    let n = 1024 / 32; // 32 tiles
    let partial = api::zeros::<f32>(&[n]).partition([1]);
    let z = api::zeros::<f32>(&[1]).partition([1]);
    // 预检查 z 初始为 0（DeviceOp 不可 clone，用独立分配验证）
    {
        let z_check = api::zeros::<f32>(&[1]).partition([1]);
        let z_host = z_check
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("z to_host");
        assert_eq!(z_host, vec![0.0]);
    }

    let (partial, _) = kernels::sum_stage1(partial, x).sync().expect("stage1");
    // stage2 需要 Tensor (非 Partition)，unpartition 后传入
    let partial_t = partial.unpartition();
    let (z, _) = kernels::sum_stage2(z, partial_t).sync().expect("stage2");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
    assert_eq!(z_host, vec![1024.0]);
}

#[test]
fn test_sum_two_stage_seq() {
    let n = 1024usize;
    let x_host = Arc::new((1..=n).map(|v| v as f32).collect::<Vec<f32>>());
    let x = api::copy_host_vec_to_device(&x_host)
        .sync()
        .expect("h2d")
        .reshape(&[n])
        .expect("reshape");
    let p = n / 32;
    let partial = api::zeros::<f32>(&[p]).partition([1]);
    let z = api::zeros::<f32>(&[1]).partition([1]);

    let (partial, _) = kernels::sum_stage1(partial, x).sync().expect("stage1");
    let partial_t = partial.unpartition();
    let (z, _) = kernels::sum_stage2(z, partial_t).sync().expect("stage2");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
    let expected = (n * (n + 1) / 2) as f32; // 524800
    assert!(
        (z_host[0] - expected).abs() < 1e-3,
        "expected {} got {}",
        expected,
        z_host[0]
    );
}

#[test]
fn test_sum_two_stage_not_divisible() {
    // 1000 不是 32 的倍数，验证 padding=0 不影响和
    let n = 1000usize;
    let x = api::ones::<f32>(&[n]);
    let p = (n + 31) / 32; // 32
    let partial = api::zeros::<f32>(&[p]).partition([1]);
    let z = api::zeros::<f32>(&[1]).partition([1]);

    let (partial, _) = kernels::sum_stage1(partial, x).sync().expect("stage1");
    let partial_t = partial.unpartition();
    let (z, _) = kernels::sum_stage2(z, partial_t).sync().expect("stage2");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
    assert!((z_host[0] - 1000.0).abs() < 1e-3, "got {}", z_host[0]);
}

#[test]
fn test_sum_two_stage_vs_single() {
    // 对比单阶段与二阶段结果一致
    let n = 512usize;
    // 单阶段
    let z1 = api::zeros::<f32>(&[1]).partition([1]);
    let (z1, _) = kernels::sum_1d(z1, api::ones::<f32>(&[n]))
        .sync()
        .expect("sum_1d");
    let h1 = z1.unpartition().to_host_vec().sync().expect("h1")[0];
    // 二阶段
    let p = (n + 31) / 32;
    let partial = api::zeros::<f32>(&[p]).partition([1]);
    let z2 = api::zeros::<f32>(&[1]).partition([1]);
    let x2 = api::ones::<f32>(&[n]);
    let (partial, _) = kernels::sum_stage1(partial, x2).sync().expect("s1");
    let partial_t = partial.unpartition();
    let (z2, _) = kernels::sum_stage2(z2, partial_t).sync().expect("s2");
    let h2 = z2.unpartition().to_host_vec().sync().expect("h2")[0];
    assert!((h1 - h2).abs() < 1e-3, "single {} vs two-stage {}", h1, h2);
    assert!((h1 - 512.0).abs() < 1e-3);
}

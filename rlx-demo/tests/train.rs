use rlx::prelude::*;
use rlx_optim::Optimizer;

/// End-to-end training loop on `rlx`: build a forward graph, wrap it in a
/// backward graph (`grad_with_loss`), and drive gradient descent with a
/// host-side `rlx-optim` optimizer.
///
/// Demonstrates, in one pass:
/// - **forward**:  `Graph` primitives (`input` / `param` / `matmul` / `add`)
/// - **loss**:     scalar MSE (built from `sub` / `mul` / `mean`)
/// - **backward**: `grad_with_loss` → backward graph outputs
///   `[loss, dW, dB]`; `d_output = [1.0]` seeds the loss path
/// - **optimizer**: `rlx_optim::Adam` (or `Sgd`) host-side `step`
/// - **train loop** (set params → run → read gradients → step → repeat)
#[test]
fn test_train() {
    if !rlx_cuda::is_available() {
        return;
    }

    // ── Data: learn y = 2·x0 + 3·x1  (deterministic, noise-free). ──────────
    const N: usize = 4;
    let xs: Vec<f32> = vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 5.0];
    let ys: Vec<f32> = vec![8.0, 7.0, 15.0, 23.0];

    // ── Forward graph: pred = x·w + b, loss = mean((pred − y)²) ────────────
    let mut g = Graph::new("linear_reg");
    let x = g.input("x", Shape::new(&[N, 2], DType::F32));
    let y = g.input("y", Shape::new(&[N, 1], DType::F32));
    let w = g.param("w", Shape::new(&[2, 1], DType::F32));
    let b = g.param("b", Shape::new(&[1], DType::F32));

    let mm = g.matmul(x, w, Shape::new(&[N, 1], DType::F32));
    let pred = g.add(mm, b);
    let diff = g.sub(pred, y);
    let sq = g.mul(diff, diff);
    let loss = g.mean(sq, vec![0, 1], false); // scalar
    // here the output just loss
    g.set_outputs(vec![loss]);

    // ── Backward: outputs [loss, dW, dB]; the forward params become run() inputs.
    let bwd = grad_with_loss(&g, &[w, b]);
    let mut compiled = Session::new(Device::Cuda).compile(bwd);

    // ── Optimizer: Adam on the two params (keyed by name, per-tensor state).──
    let mut opt = rlx_optim::Adam::new(0.1);
    let mut w_v = vec![1.0_f32, -1.0];
    let mut b_v = vec![0.0_f32];

    let target_loss = 1e-6_f32;
    let mut final_loss = f32::MAX;
    for step in 1..=2000 {
        compiled.set_param("w", &w_v);
        compiled.set_param("b", &b_v);

        // get the output and gradient in single run
        let out = compiled.run(&[("x", &xs[..]), ("y", &ys[..]), ("d_output", &[1.0_f32])]);
        let loss_v = out[0][0];
        // gradient for w and b
        let dw = &out[1];
        let db = &out[2];

        // update w and b
        opt.step("w", &[2, 1], &mut w_v, dw);
        opt.step("b", &[1], &mut b_v, db);
        opt.end_iteration();

        if step % 200 == 0 || loss_v < target_loss {
            println!("step {step:4}  loss {loss_v:.8}  w {w_v:?}  b {b_v:?}");
        }
        final_loss = loss_v;
        if loss_v < target_loss {
            break;
        }
    }

    println!("final w = {w_v:?}, b = {b_v:?}");
    assert!(
        final_loss < target_loss,
        "training did not converge: final loss {final_loss}"
    );
    // Parity with the no-noise target y = 2·x0 + 3·x1.
    assert!((w_v[0] - 2.0).abs() < 1e-2);
    assert!((w_v[1] - 3.0).abs() < 1e-2);
}

/// Multi-layer (MLP) training loop: shows how the parameters of *different*
/// layers are updated separately.
///
/// A 2-layer ReLU MLP `y ≈ W2·relu(x·W1 + b1) + b2` is trained with MSE. The
/// backward graph returns one gradient per `wrt` parameter
/// (`[loss, dW1, dB1, dW2, dB2]`), and each layer's params are stepped with
/// its own optimizer (per-layer learning rates, e.g. a smaller LR on the
/// input layer) to make the layer-wise update explicit.
#[test]
fn test_train_multiple_modules() {
    if !rlx_cuda::is_available() {
        return;
    }

    const N: usize = 8;
    const H: usize = 4; // hidden width
    let xs: Vec<f32> = vec![
        1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0,
    ];
    let ys: Vec<f32> = vec![5.0, 8.0, 7.0, 10.0, 11.0, 9.0, 13.0, 12.0];

    // ── Forward: layer1 = relu(x·W1 + b1), pred = layer1·W2 + b2 ────────────
    let mut g = Graph::new("mlp");
    let x = g.input("x", Shape::new(&[N, 2], DType::F32));
    let y = g.input("y", Shape::new(&[N, 1], DType::F32));

    let w1 = g.param("w1", Shape::new(&[2, H], DType::F32));
    let b1 = g.param("b1", Shape::new(&[H], DType::F32));
    let w2 = g.param("w2", Shape::new(&[H, 1], DType::F32));
    let b2 = g.param("b2", Shape::new(&[1], DType::F32));

    let mm1 = g.matmul(x, w1, Shape::new(&[N, H], DType::F32));
    let h1 = g.add(mm1, b1);
    let h1 = g.relu(h1);
    let mm2 = g.matmul(h1, w2, Shape::new(&[N, 1], DType::F32));
    let pred = g.add(mm2, b2);
    let diff = g.sub(pred, y);
    let sq = g.mul(diff, diff);
    let loss = g.mean(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);

    // ── Backward: outputs [loss, dW1, dB1, dW2, dB2] ────────────────────────
    let wrt = [w1, b1, w2, b2];
    let bwd = grad_with_loss(&g, &wrt);
    let mut compiled = Session::new(Device::Cuda).compile(bwd);

    // ── Two optimizers, one per layer: a smaller LR on the first layer ─────
    let mut opt_layer1 = rlx_optim::Adam::new(0.05);
    let mut opt_layer2 = rlx_optim::Adam::new(0.1);
    let mut w1_v = vec![0.5_f32, -0.5, 0.3, 0.2, -0.1, 0.4, 0.2, 0.3]; // [2,4]
    let mut b1_v = vec![0.0_f32; H];
    let mut w2_v = vec![0.5_f32, -0.4, 0.3, 0.2]; // [4,1]
    let mut b2_v = vec![0.0_f32];

    let target_loss = 1e-4_f32;
    let mut final_loss = f32::MAX;
    for step in 1..=5000 {
        compiled.set_param("w1", &w1_v);
        compiled.set_param("b1", &b1_v);
        compiled.set_param("w2", &w2_v);
        compiled.set_param("b2", &b2_v);
        let out = compiled.run(&[("x", &xs[..]), ("y", &ys[..]), ("d_output", &[1.0_f32])]);
        let loss_v = out[0][0];

        // Layer 1 update: W1 [2,H], b1 [H].
        opt_layer1.step("w1", &[2, H], &mut w1_v, &out[1]);
        opt_layer1.step("b1", &[H], &mut b1_v, &out[2]);
        // Layer 2 update: W2 [H,1], b2 [1].
        opt_layer2.step("w2", &[H, 1], &mut w2_v, &out[3]);
        opt_layer2.step("b2", &[1], &mut b2_v, &out[4]);
        opt_layer1.end_iteration();
        opt_layer2.end_iteration();

        if step % 500 == 0 || loss_v < target_loss {
            println!(
                "step {step:4}  loss {loss_v:.8}  |W1|={:.3}  |W2|={:.3}  b1={b1_v:?}  b2={b2_v:?}",
                w1_v.iter().map(|v| v * v).sum::<f32>().sqrt(),
                w2_v.iter().map(|v| v * v).sum::<f32>().sqrt(),
            );
        }
        final_loss = loss_v;
        if loss_v < target_loss {
            break;
        }
    }

    println!("final loss {final_loss:.8}");
    println!("layer1 w1 = {w1_v:?}");
    println!("layer2 w2 = {w2_v:?}");
    assert!(
        final_loss < target_loss,
        "MLP did not converge: final loss {final_loss}"
    );
    assert_eq!(w1_v.len(), 2 * H, "W1 must stay a flat [2, H] buffer");
    assert_eq!(w2_v.len(), H, "W2 must stay a flat [H, 1] buffer");
}

use rlx::prelude::*;

/// Autoregressive window size: predict s[t] from the previous 3 values.
const AR: usize = 3;
/// Number of training windows.
const N: usize = 32;

/// True AR(3) recurrence the model must recover:
/// s[t] = 2.0811·s[t-1] − 1.9655·s[t-2] + 0.81225·s[t-3] + 0.05
/// Poles at 0.9 (real) and 0.95·e^{±0.9i}: three distinct decaying modes, so
/// the sliding windows stay linearly independent (well-conditioned least
/// squares) instead of collapsing to a fixed point.
fn make_sequence(len: usize) -> Vec<f32> {
    let mut s = vec![0.8_f32, -0.4, 0.6];
    while s.len() < len {
        let t = s.len();
        let next = 2.0811 * s[t - 1] - 1.9655 * s[t - 2] + 0.81225 * s[t - 3] + 0.05;
        s.push(next);
    }
    s
}

#[test]
fn test_linear() {
    if !rlx_cuda::is_available() {
        return;
    }

    // Sliding windows: x[i] = s[i..i+3], target[i] = s[i+3].
    let seq = make_sequence(N + AR);
    let mut xs = Vec::with_capacity(N * AR);
    let mut targets = Vec::with_capacity(N);
    for i in 0..N {
        xs.extend_from_slice(&seq[i..i + AR]);
        targets.push(seq[i + AR]);
    }

    // Forward graph: linear module pred = x·w + b, RMS loss against target.
    let mut g = Graph::new("linear_ar");
    let x = g.input("x", Shape::new(&[N, AR], DType::F32));
    let target = g.input("target", Shape::new(&[N, 1], DType::F32));
    let w = g.param("w", Shape::new(&[AR, 1], DType::F32));
    let b = g.param("b", Shape::new(&[1], DType::F32));

    let pred = g.matmul(x, w, Shape::new(&[N, 1], DType::F32));
    let pred = g.add(pred, b);
    let diff = g.sub(pred, target);
    let sq = g.mul(diff, diff);
    let mse = g.mean(sq, vec![0, 1], false);
    let loss = g.sqrt(mse); // RMS
    g.set_outputs(vec![loss]);

    // Backward graph: outputs [loss, dW, dB]; params become run() inputs.
    let backward = grad_with_loss(&g, &[w, b]);
    let mut compiled = Session::new(Device::Cuda).compile(backward);

    // Plain gradient descent on (w, b). The RMS loss has a normalized
    // gradient (∇L = Xᵀr / (n·L), constant magnitude near the optimum), so a
    // fixed step size ends in a limit cycle — anneal it geometrically instead.
    let mut w_v = vec![0.1_f32, -0.1, 0.05];
    let mut b_v = vec![0.0_f32];
    let mut lr = 0.5_f32;
    let lr_decay = 0.99_f32;
    let target_loss = 5e-3_f32;
    let mut final_loss = f32::MAX;

    for step in 1..=2000 {
        compiled.set_param("w", &w_v);
        compiled.set_param("b", &b_v);
        let out = compiled.run(&[
            ("x", &xs[..]),
            ("target", &targets[..]),
            ("d_output", &[1.0][..]),
        ]);
        let loss = out[0][0];
        let perplexity = loss.exp();
        println!("round {step:4}  loss {loss:.6}  perplexity {perplexity:.6}");

        final_loss = loss;
        if loss < target_loss {
            println!("converged at round {step}: w = {w_v:?}, b = {b_v:?}");
            break;
        }

        for (p, dp) in w_v.iter_mut().zip(&out[1]) {
            *p -= lr * dp;
        }
        for (p, dp) in b_v.iter_mut().zip(&out[2]) {
            *p -= lr * dp;
        }
        lr *= lr_decay;
    }

    assert!(
        final_loss < target_loss,
        "did not converge: final loss {final_loss}"
    );
}

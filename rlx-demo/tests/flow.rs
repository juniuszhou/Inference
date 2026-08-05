//! Demo: assemble a tiny residual block with `rlx-flow`, compile on CUDA, run.
//!
//! Pipeline (no HIR/Graph in the model code):
//!   residual_save → linear(w = I) → residual_add  ⇒  y = x + x·I = 2x

use rlx::prelude::*;
use rlx_cuda::backend::CudaExecutable;
use rlx_flow::{FlowStage, MapWeights, ModelFlow, blocks::RmsNormStage};

#[test]
fn test_flow() {
    if !rlx_cuda::is_available() {
        return;
    }

    // Weight source consumed by `ModelFlow::build` (key → f32 data + shape).
    let mut weights = MapWeights::default();
    weights.insert("w", vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]);

    // Fluent stage assembly: `.layer` wraps a `LayerStack` as a named scope.
    let flow = ModelFlow::new("residual_linear")
        .input("x", Shape::new(&[1, 2], DType::F32))
        .layer("block", |s| {
            s.residual_save().linear("w", false).residual_add()
        });

    let built = flow.build(&mut weights).expect("flow build");
    let (g, params) = built.into_graph_parts().expect("graph + params");

    let mut exe = CudaExecutable::compile(g);
    for (k, v) in &params {
        exe.set_param(k.as_str(), v.as_slice());
    }

    let out = exe.run(&[("x", &[1.0_f32, 2.0])]);
    println!("{:?}", out);
    // x + x·I = 2x
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], vec![2.0_f32, 4.0]);
}

/// Plain `.linear` stage (no residual): `y = x · Wᵀ` when `transpose = true`.
///
/// Weight is stored HF-style `[out, in]`. Here W = I, so Wᵀ = I and y = x.
#[test]
fn test_flow_linear() {
    if !rlx_cuda::is_available() {
        return;
    }

    let mut weights = MapWeights::default();
    // HF layout `[out, in]` = 2×2 identity; `transpose = true` → matmul uses Wᵀ.
    weights.insert("w", vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]);

    let flow = ModelFlow::new("linear")
        .input("x", Shape::new(&[1, 2], DType::F32))
        .linear("linear_weight", true)
        .rms_norm("norm_weight", 1.0_f32);

    let flow = flow.stage(FlowStage::RmsNorm(RmsNormStage::new("w", 1.0_f32)));

    let built = flow.build(&mut weights).expect("flow build");
    let (g, params) = built.into_graph_parts().expect("graph + params");

    let mut exe = CudaExecutable::compile(g);
    for (k, v) in &params {
        exe.set_param(k.as_str(), v.as_slice());
    }

    let out = exe.run(&[("x", &[1.0_f32, 2.0])]);
    println!("{:?}", out);
    // x · I = x
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], vec![1.0_f32, 2.0]);
}

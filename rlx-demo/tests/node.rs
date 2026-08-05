use rlx::prelude::*;
use rlx_flow::blocks::LinearStage;
use rlx_flow::{FlowStage, MapWeights, ModelFlow};
use rlx_optim::Optimizer;

#[test]
fn test_add_binary_op() {
    if !rlx_cuda::is_available() {
        return;
    }
    let mut g = Graph::new("hello");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.input("y", Shape::new(&[4], DType::F32));
    let z = g.binary(BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![z]);

    let mut exe = rlx_cuda::backend::CudaExecutable::compile(g);
    let out = exe.run(&[
        ("x", &[1.0_f32, 2.0, 3.0, 4.0]),
        ("y", &[10.0_f32, 20.0, 30.0, 40.0]),
    ]);
    println!("{:?}", out);
    assert_eq!(out[0], vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_flow() {
    let mut w = MapWeights::default();
    w.insert("w", vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]);

    let flow = ModelFlow::new("linear")
        .input("x", Shape::new(&[1, 2], DType::F32))
        .stage(FlowStage::Linear(LinearStage::new("w", false)));

    let built = flow.build(&mut w).expect("flow build");
    let (g, params) = built.into_graph_parts().expect("graph + params");

    let mut compiled = Session::new(Device::Cpu).compile(g);
    for (k, v) in &params {
        compiled.set_param(k.as_str(), v.as_slice());
    }
    let out = compiled.run(&[("x", &[1.0_f32, 2.0][..])]);
    println!("{:?}", out);
    assert_eq!(out[0], vec![1.0_f32, 2.0]);
}

#[test]
fn test_flow_linear() {
    if !rlx_cuda::is_available() {
        return;
    }

    let mut weights = MapWeights::default();
    weights.insert("w", vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]);

    // residual_save → linear → residual_add: saves x, computes x·w, then x + (x·w).
    let flow = ModelFlow::new("residual_linear")
        .input("x", Shape::new(&[1, 2], DType::F32))
        .residual_save()
        .linear("w", true)
        .residual_add();

    let built = flow.build(&mut weights).expect("flow build");
    let (g, params) = built.into_graph_parts().expect("graph + params");

    let mut exe = rlx_cuda::backend::CudaExecutable::compile(g);
    for (k, v) in &params {
        exe.set_param(k.as_str(), v.as_slice());
    }

    let out = exe.run(&[("x", &[1.0_f32, 2.0])]);
    println!("{:?}", out);
    // x + x·I = 2x
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], vec![2.0_f32, 4.0]);
}

use rlx::prelude::*;
use rlx_cuda::backend::CudaExecutable;

#[test]
fn test_basic() {
    let mut g = Graph::new("hello");
    let x = g.input("x", Shape::new(&[1, 4], DType::F32));
    let w = g.param("w", Shape::new(&[4, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cuda).compile(g);
    compiled.set_param("w", &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    let out = compiled.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
    println!("{:?}", out);
}

#[test]
fn rlx_dsl_via_umbrella_root() {
    let g = rlx::rlx! {
        graph "umbrella";
        input x: [2, 4];
        param w: [4, 3];
        let y = gelu(x @ w);
        out y;
    };
    assert_eq!(g.name, "umbrella");
    assert_eq!(g.outputs.len(), 1);
}

#[test]
fn rlx_dsl_via_prelude() {
    use rlx::prelude::*;
    let g = rlx! {
        input x: [?, 8];
        param w: [8, 8];
        let y = x @ w;
    };
    assert_eq!(g.outputs.len(), 1);
}

/// The DSL graph doesn't just type-check — it compiles and runs. Inputs and
/// params are fed by their auto-derived names (`x`, `w`, `b`).
#[test]
fn rlx_dsl_compiles_and_runs() {
    use rlx::runtime::{Device, Session};

    let g = rlx::rlx! {
        input x: [1, 2];
        param w: [2, 2];
        param b: [2];
        let y = x @ w + b;
        out y;
    };

    let mut compiled = Session::new(Device::Cuda).compile(g);
    // x = [1, 2]; w = identity; b = [10, 20]  ⇒  x·w + b = [11, 22].
    compiled.set_param("w", &[1.0, 0.0, 0.0, 1.0]);
    compiled.set_param("b", &[10.0, 20.0]);
    let out = compiled.run(&[("x", &[1.0, 2.0][..])]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], vec![11.0, 22.0]);
}

#[test]
fn binary_add_matches_reference() {
    // if !rlx_cuda::is_available() {
    //     return;
    // }
    let mut g = Graph::new("add");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.input("y", Shape::new(&[4], DType::F32));
    let z = g.binary(BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![z]);
    let mut exe = CudaExecutable::compile(g);
    let out = exe.run(&[
        ("x", &[1.0_f32, 2.0, 3.0, 4.0]),
        ("y", &[10.0_f32, 20.0, 30.0, 40.0]),
    ]);
    assert_eq!(out[0], vec![11.0, 22.0, 33.0, 44.0]);
}

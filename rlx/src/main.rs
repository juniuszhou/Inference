use rlx::prelude::*;

fn main() {
    let mut g = Graph::new("hello");
    let x = g.input("x", Shape::new(&[1, 4], DType::F32));
    let w = g.param("w", Shape::new(&[4, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param("w", &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    let out = compiled.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
}

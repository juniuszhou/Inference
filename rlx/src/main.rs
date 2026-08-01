use rlx_tensor::Tensor;

fn main() {
    println!("Hello, world!");
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    println!("{:?}", a);
}

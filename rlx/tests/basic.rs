use rlx_tensor::{Device, Tensor};

#[test]
fn test_basic() {
    println!("Hello, world!");
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let result = a.on(Device::Cuda).to_vec();
    println!("{:?}", a);
    println!("Result: {:?}", result);
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

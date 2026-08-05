#[test]
fn test_basic() {
    println!("Hello, world!");
    let a = rlx_tensor::Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let result = a.to_vec();
    println!("{:?}", a);
    println!("Result: {:?}", result);
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

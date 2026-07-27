use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};

#[test]
fn test_slice() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let numel = input_data.len();
    println!("numel: {}", numel);

    let a_dev: CudaSlice<f32> = stream.clone_htod(&input_data).unwrap();
    let b_dev: CudaSlice<f32> = a_dev.try_clone().unwrap();
    println!("b_dev: {:?}", b_dev);

    // Convert the slice to a view as immutable
    let c_view = b_dev.as_view();
    println!("c_view: {:?}", c_view);
}

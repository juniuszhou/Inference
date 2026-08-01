/*
This test is to test the embedding layer.
It will be used to test the embedding layer in the model.
*/

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};

#[test]
fn test_embedding() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
}

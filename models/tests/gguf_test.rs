use candle_core::{Device, Tensor};
use models::gguf::get_model;

#[test]
fn test_get_model() -> anyhow::Result<()> {
    // let _no_grad = candle_core::NoGradGuard::new();

    let device = Device::cuda_if_available(0)?;
    let (mut model, _device) = get_model()?;
    let ids = Tensor::new(&[1u32, 2], &device)?.unsqueeze(0)?;

    // detach the logits to avoid backpropagation
    let logits = model.forward(&ids, 0)?.detach();
    println!("logits shape: {:?}", logits.shape());
    Ok(())
}

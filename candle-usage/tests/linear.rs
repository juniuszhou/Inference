use candle_core::DType;
use candle_core::{CudaDevice, Device, Tensor};
use candle_nn::{Linear, Module};

#[test]
fn test_linear() -> candle_core::Result<()> {
    let device = Device::Cuda(CudaDevice::new_with_stream(0)?);

    let weight = Tensor::rand(0.0, 1.0, (2, 2), &device)?;
    let weight = weight.to_dtype(DType::F16)?;

    let linear = Linear::new(weight, None);

    let input = Tensor::rand(0.0, 1.0, (2, 2), &device)?.to_dtype(DType::F16)?;
    let output = linear.forward(&input)?;
    println!(
        "output: {:?}",
        output.to_dtype(DType::F32)?.to_vec2::<f32>()?
    );

    Ok(())
}

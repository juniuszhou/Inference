use candle_core::{CudaDevice, Device};
use candle_core::{Result, Tensor};

fn main() -> Result<()> {
    println!("Candle Tensor Demo");

    // Create a tensor from a vector
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(data, (2, 2), &Device::Cuda(CudaDevice::new_with_stream(0)?))?;
    println!("Original tensor:");
    println!("{}", tensor);

    Ok(())
}

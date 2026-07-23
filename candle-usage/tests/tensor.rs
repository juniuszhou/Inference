use candle_core::{CudaDevice, Device, StreamTensor, Tensor};

/*
Cheatsheet:

| PyTorch | Candle |
|---------|--------|
| `torch.Tensor([[1, 2], [3, 4]])` | `Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?` |
| `torch.zeros((2, 2))` | `Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?` |
| `tensor[:, :4]` | `tensor.i((.., ..4))?` |
| `tensor.view((2, 2))` | `tensor.reshape((2, 2))?` |
| `a.matmul(b)` | `a.matmul(&b)?` |
| `a + b` | `&a + &b` |
| `tensor.to(device="cuda")` | `tensor.to_device(&Device::new_cuda(0)?)?` |
| `tensor.to(dtype=torch.float16)` | `tensor.to_dtype(&DType::F16)?` |
| `torch.save({"A": A}, "model.bin")` | `candle::safetensors::save(&HashMap::from([("A", A)]), "model.safetensors")?` |
| `weights = torch.load("model.bin")` | `candle::safetensors::load("model.safetensors", &device)?` |

*/
#[test]
fn test_tensor() -> candle_core::Result<()> {
    let device = Device::Cuda(CudaDevice::new_with_stream(0)?);

    let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    println!("tensor: {:?}", tensor.shape().dims());
    assert_eq!(tensor.shape().dims(), &[2, 2]);

    let tensor = tensor.reshape((4,))?;

    println!("tensor: {:?}", tensor.to_vec1::<f32>()?);
    assert_eq!(tensor.to_vec1::<f32>()?, vec![1.0f32, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn test_tensor_2() -> candle_core::Result<()> {
    let device = Device::Cuda(CudaDevice::new_with_stream(0)?);
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    println!("tensor: {:?}", tensor.dims());
    assert_eq!(tensor.dims(), &[2, 2]);

    Ok(())
}

#[test]
fn test_tensor_3() -> candle_core::Result<()> {
    // let device = Device::Cuda(CudaDevice::new_with_stream(0)?);
    let tensor = StreamTensor::from(None);
    println!("tensor: {:?}", tensor.shape());

    Ok(())
}

#[test]
fn test_tensor_4() -> candle_core::Result<()> {
    let device = Device::Cuda(CudaDevice::new_with_stream(0)?);
    let a = Tensor::new(&[[0f32, 1., 2.], [3., 4., 5.], [6., 7., 8.]], &device)?;

    let b = a.narrow(0, 1, 2)?;
    assert_eq!(b.shape().dims(), &[2, 3]);
    assert_eq!(b.to_vec2::<f32>()?, &[[3., 4., 5.], [6., 7., 8.]]);

    Ok(())
}

use candle_core::DType;
use candle_core::{CudaDevice, Device, Tensor};
use candle_nn::{linear, Linear, Module};
use candle_nn::{VarBuilder, VarMap};

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

#[test]
fn test_linear_2() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let linear = linear(2, 2, vb.pp("linear"))?;

    varmap.data().lock().unwrap().iter().for_each(|(key, val)| {
        println!("key: {:?}", key);
        println!("val: {:?}", val.as_tensor().clone());
    });

    let weight = linear.weight();
    let bias = linear.bias();
    println!("params: {:?} {:?}", weight, bias);
    Ok(())
}

#[test]
fn test_linear_3() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let path = vb.push_prefix("linear");
    println!("path: {:?}", path.get(&[2, 2], "weight"));
    // println!("path: {:?}", path.get("linear.bias"));
    Ok(())
}

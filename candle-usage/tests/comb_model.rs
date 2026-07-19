use candle_core::DType;
use candle_core::{CudaDevice, Device, Tensor};
use candle_nn::{linear, Linear, Module};
use candle_nn::{VarBuilder, VarMap};

struct CombModel {
    layer: Linear,
}

impl CombModel {
    pub fn new(vb: VarBuilder) -> Self {
        Self {
            layer: linear(2, 2, vb.pp("linear")).expect("Failed to create linear layer"),
        }
    }
}

impl Module for CombModel {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.layer.forward(x)
    }
}

#[test]
fn test_comb_model() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let comb_model = CombModel::new(vb);

    let input = Tensor::new(&[[1.0f32, 2.0]], &device)?;
    let output = comb_model.forward(&input)?;
    println!("output: {:?}", output);
    Ok(())
}

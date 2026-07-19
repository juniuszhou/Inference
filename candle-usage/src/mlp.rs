use candle_core::{Result, Tensor};
use candle_nn::linear as linear_fn;
use candle_nn::Linear;
use candle_nn::VarBuilder;
use candle_nn::{Activation, Module};

pub struct Mlp {
    pub up_proj: Linear,
    pub down_proj: Linear,
    pub activation: Activation,
}

impl Mlp {
    pub fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Self {
        Self {
            up_proj: linear_fn(d_model, d_ff, vb.pp("up_proj")).unwrap(),
            down_proj: linear_fn(d_ff, d_model, vb.pp("down_proj")).unwrap(),
            activation: Activation::Swish,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.up_proj.forward(x)?;
        let x = self.activation.forward(&x)?;
        let x = self.down_proj.forward(&x)?;
        Ok(x)
    }
}

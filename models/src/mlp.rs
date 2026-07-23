use candle_nn::Activation;
use candle_nn::Linear;
use candle_nn::VarBuilder;
use candle_nn::linear as linear_fn;

pub struct Mlp {
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl Mlp {
    fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Self {
        Self {
            up_proj: linear_fn(d_model, d_ff, vb.pp("up_proj")).unwrap(),
            down_proj: linear_fn(d_ff, d_model, vb.pp("down_proj")).unwrap(),
            activation: Activation::Gelu,
        }
    }
}

use crate::config::TransformerCasualLLMConfig;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{
    embedding, linear, linear_b, rms_norm, Activation, Embedding, Linear, Module, VarBuilder,
};
use std::vec::Vec;

fn causal_mask(seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::from_vec(mask, (seq_len, seq_len), device)?.to_dtype(dtype)
}

struct MultiheadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiheadAttention {
    fn new(d_model: usize, n_heads: usize, attn_bias: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = d_model / n_heads;
        if !d_model.is_multiple_of(n_heads) {
            return Err(candle_core::Error::Msg(format!(
                "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )));
        }
        let scale = (head_dim as f64).powf(-0.5);
        let q_proj = linear_b(d_model, d_model, attn_bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(d_model, d_model, attn_bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(d_model, d_model, attn_bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(d_model, d_model, attn_bias, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: n_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _d_model) = x.shape().dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * self.scale)?;
        let attn_weights = attn_weights.broadcast_add(mask)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, _d_model))?;

        self.o_proj.forward(&attn_output)
    }
}

struct FeedForward {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let gate = linear(d_model, d_ff, vb.pp("gate"))?;
        let up = linear(d_model, d_ff, vb.pp("up"))?;
        let down = linear(d_ff, d_model, vb.pp("down"))?;
        Ok(Self { gate, up, down })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = Activation::Silu.forward(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&(gate * up)?)
    }
}

pub struct DecoderLayer {
    self_attn: MultiheadAttention,
    feed_forward: FeedForward,
    input_norm: candle_nn::RmsNorm,
    post_attn_norm: candle_nn::RmsNorm,
}

impl DecoderLayer {
    fn new(config: &TransformerCasualLLMConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = MultiheadAttention::new(
            config.d_model,
            config.n_heads,
            config.attn_bias,
            vb.pp("self_attn"),
        )?;
        let feed_forward = FeedForward::new(config.d_model, config.d_ff, vb.pp("feed_forward"))?;
        let input_norm = rms_norm(config.d_model, 1e-5, vb.pp("input_norm"))?;
        let post_attn_norm = rms_norm(config.d_model, 1e-5, vb.pp("post_attn_norm"))?;
        Ok(Self {
            self_attn,
            feed_forward,
            input_norm,
            post_attn_norm,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.input_norm.forward(x)?;
        let x = self.self_attn.forward(&x, mask)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attn_norm.forward(&x)?;
        let x = self.feed_forward.forward(&x)?;
        residual + x
    }
}

pub struct TransformerModel {
    embed: Embedding,
    layers: Vec<DecoderLayer>,
    norm: candle_nn::RmsNorm,
}

impl TransformerModel {
    pub fn new(config: &TransformerCasualLLMConfig, vb: VarBuilder) -> Result<Self> {
        let embed = embedding(config.vocab_size, config.d_model, vb.pp("embed"))?;
        let norm = rms_norm(config.d_model, 1e-5, vb.pp("norm"))?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = DecoderLayer::new(config, vb.pp(format!("layers.{i}")))?;
            layers.push(layer);
        }
        Ok(Self {
            embed,
            layers,
            norm,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.shape().dims2()?;
        let x = self.embed.forward(input_ids)?;
        let device = x.device();
        let dtype = x.dtype();
        let mask = causal_mask(seq_len, dtype, device)?;

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &mask)?;
        }
        self.norm.forward(&x)
    }
}

#[allow(dead_code)]
pub struct TransformerCasualLLM {
    config: TransformerCasualLLMConfig,
    model: TransformerModel,
    header: Linear,
}

impl TransformerCasualLLM {
    pub fn new(config: TransformerCasualLLMConfig, vb: VarBuilder) -> Result<Self> {
        let model = TransformerModel::new(&config, vb.pp("model"))?;
        let header = linear(config.d_model, config.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            config,
            model,
            header,
        })
    }
}

impl Module for TransformerCasualLLM {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.model.forward(x)?;
        self.header.forward(&x)
    }
}

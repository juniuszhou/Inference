use anyhow::Result;
use candle_core::{CudaDevice, DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::llama::{Cache, Config, Llama};

pub const DEFAULT_MAX_SEQ_LEN: usize = 512;

pub fn load_llama(local: bool) -> Result<(Llama, VarMap, Config)> {
    let device = Device::Cuda(CudaDevice::new_with_stream(0)?);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);

    let config = if local {
        get_local_config()
    } else {
        Config::config_7b_v1(true)
    };

    let model = Llama::load(vb, &config)?;
    Ok((model, varmap, config))
}

pub trait LlamaExt {
    fn generate(
        &self,
        input_tok: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor>;
}

impl LlamaExt for Llama {
    fn generate(
        &self,
        input_tok: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        self.forward(input_tok, index_pos, cache)
    }
}

// for local testing with low end GPU
fn get_local_config() -> Config {
    Config {
        hidden_size: 512,
        intermediate_size: 11008,
        vocab_size: 32000,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        use_flash_attn: false,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        bos_token_id: None,
        eos_token_id: None,
        rope_scaling: None,
        max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
        tie_word_embeddings: false,
    }
}

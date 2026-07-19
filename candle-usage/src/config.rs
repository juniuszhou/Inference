#[derive(Clone)]
pub struct TransformerCasualLLMConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub dropout: f32,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    pub attn_bias: bool,
    pub attn_mask: bool,
}

impl TransformerCasualLLMConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        dropout: f32,
        vocab_size: usize,
        max_seq_len: usize,
        pad_token_id: usize,
        eos_token_id: usize,
        bos_token_id: usize,
        attn_bias: bool,
        attn_mask: bool,
    ) -> Self {
        Self {
            d_model,
            n_heads,
            n_layers,
            d_ff,
            dropout,
            vocab_size,
            max_seq_len,
            pad_token_id,
            eos_token_id,
            bos_token_id,
            attn_bias,
            attn_mask,
        }
    }

    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }
}

#[allow(dead_code)]
pub struct TransformerModelConfig {
    pub d_model: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub pad_token_id: usize,
    pub eos_token_id: usize,
    pub bos_token_id: usize,
    pub attn_bias: bool,
    pub attn_mask: bool,
}

impl TransformerModelConfig {
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        vocab_size: usize,
        max_seq_len: usize,
        pad_token_id: usize,
        eos_token_id: usize,
        bos_token_id: usize,
        attn_bias: bool,
        attn_mask: bool,
    ) -> Self {
        Self {
            d_model,
            vocab_size,
            max_seq_len,
            pad_token_id,
            eos_token_id,
            bos_token_id,
            attn_bias,
            attn_mask,
        }
    }
}

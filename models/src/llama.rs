#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) max_position_embeddings: usize,
    pub(crate) eos_token_id: u32,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) stop_token_ids: Vec<u32>,
}

// struct Llama {
//     model: ModelWeights,
//     device: Device,
//     tokenizer: Tokenizer,
//     logits_processor: LogitsProcessor,
// }

// impl Llama {
//     pub fn new(
//         model: ModelWeights,
//         device: Device,
//         tokenizer: Tokenizer,
//         logits_processor: LogitsProcessor,
//     ) -> Self {
//         Self {
//             model,
//             device,
//             tokenizer,
//             logits_processor,
//         }
//     }
// }

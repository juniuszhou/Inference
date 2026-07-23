mod mlp;
pub use mlp::*;

pub mod llama;
pub use llama::*;

pub mod gguf;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::quantized::tokenizer::TokenizerFromGguf;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
pub use gguf::*;
use tokenizers::tokenizer::Tokenizer;
pub struct InferenceEngine {
    pub model: ModelWeights,
    pub device: Device,
    pub tokenizer: Tokenizer,
    pub logits_processor: LogitsProcessor,
}

const CHAT_TEMPLATE: &str = "\
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
{prompt}\
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

impl InferenceEngine {
    pub fn new() -> Result<Self> {
        let (model, device) = get_model()?;

        let model_path = std::path::Path::new(MODEL_PATH);
        let mut file = std::fs::File::open(model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let tokenizer = TokenizerFromGguf::from_gguf(&content)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;

        let logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.9));
        Ok(Self {
            model,
            device,
            tokenizer,
            logits_processor,
        })
    }
}

pub fn serve(engine: &mut InferenceEngine, prompt: &str, max_tokens: usize) -> Result<String> {
    let formatted = CHAT_TEMPLATE.replace("{prompt}", prompt);
    let mut tokens = engine
        .tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::Error::msg(e))?
        .get_ids()
        .to_vec();
    let prompt_len = tokens.len();

    engine.model.clear_kv_cache();

    let input = Tensor::new(tokens.as_slice(), &engine.device)
        .map_err(|e| anyhow::Error::msg(e))?
        .unsqueeze(0)?;
    let logits = engine
        .model
        .forward(&input, 0)
        .map_err(|e| anyhow::Error::msg(e))?;
    let logits = logits.squeeze(0).map_err(|e| anyhow::Error::msg(e))?;
    let mut next_token = engine
        .logits_processor
        .sample(&logits)
        .map_err(|e| anyhow::Error::msg(e))?;
    tokens.push(next_token);

    let eos_id = engine.tokenizer.token_to_id("<|eot_id|>").unwrap_or(2);

    for i in 1..max_tokens {
        let input = Tensor::new(&[next_token], &engine.device)
            .map_err(|e| anyhow::Error::msg(e))?
            .unsqueeze(0)?;
        let logits = engine
            .model
            .forward(&input, prompt_len + i - 1)
            .map_err(|e| anyhow::Error::msg(e))?;
        let logits = logits.squeeze(0).map_err(|e| anyhow::Error::msg(e))?;

        next_token = engine
            .logits_processor
            .sample(&logits)
            .map_err(|e| anyhow::Error::msg(e))?;

        tokens.push(next_token);

        if next_token == eos_id {
            break;
        }
    }

    let output = engine
        .tokenizer
        .decode(&tokens, true)
        .map_err(|e| anyhow::Error::msg(e))?;

    Ok(output)
}

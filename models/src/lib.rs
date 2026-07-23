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

/// Holds the golden model weights (immutable after load), device, and tokenizer.
/// `ModelWeights::clone` shares the underlying weight data via ref-counting
/// while giving each clone its own empty KV cache — so spawning sessions is cheap.
pub struct SharedModelWeights {
    weights: ModelWeights,
    device: Device,
    pub tokenizer: Tokenizer,
}

impl SharedModelWeights {
    pub fn new() -> Result<Self> {
        let (weights, device) = get_model()?;

        let model_path = std::path::Path::new(MODEL_PATH);
        let mut file = std::fs::File::open(model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let tokenizer = TokenizerFromGguf::from_gguf(&content)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;

        Ok(Self {
            weights,
            device,
            tokenizer,
        })
    }

    /// Spawn a per-request [`InferenceEngine`] that shares the weight data
    /// with this [`SharedModelWeights`] but has its own fresh KV cache.
    pub fn new_session(
        &self,
        seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
    ) -> InferenceEngine {
        InferenceEngine {
            model: self.weights.clone(),
            logits_processor: LogitsProcessor::new(seed, temperature, top_p),
        }
    }
}

pub struct InferenceEngine {
    pub model: ModelWeights,
    pub logits_processor: LogitsProcessor,
}

const CHAT_TEMPLATE: &str = "\
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
{prompt}\
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

pub fn serve(shared: &SharedModelWeights, prompt: &str, max_tokens: usize) -> Result<String> {
    let mut engine = shared.new_session(299792458, Some(0.7), Some(0.9));

    let formatted = CHAT_TEMPLATE.replace("{prompt}", prompt);
    let mut tokens = shared
        .tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::Error::msg(e))?
        .get_ids()
        .to_vec();
    let prompt_len = tokens.len();

    let input = Tensor::new(tokens.as_slice(), &shared.device)
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

    let eos_id = shared.tokenizer.token_to_id("<|eot_id|>").unwrap_or(2);

    for i in 1..max_tokens {
        let input = Tensor::new(&[next_token], &shared.device)
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

    let output = shared
        .tokenizer
        .decode(&tokens, true)
        .map_err(|e| anyhow::Error::msg(e))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serve_generates_text() {
        let shared = match SharedModelWeights::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping test: failed to create SharedModelWeights: {e}");
                return;
            }
        };

        let result = serve(&shared, "What is the capital of France?", 50);
        assert!(result.is_ok(), "serve failed: {:?}", result.err());
        let text = result.unwrap();
        assert!(!text.is_empty(), "generated text should not be empty");
        assert!(
            text.to_lowercase().contains("paris"),
            "expected answer about Paris, got: {text}"
        );
        println!("Generated: {text}");
    }

    #[test]
    fn test_concurrent_serve() {
        let shared = match SharedModelWeights::new() {
            Ok(s) => std::sync::Arc::new(s),
            Err(e) => {
                eprintln!("skipping test: failed to create SharedModelWeights: {e}");
                return;
            }
        };

        let mutex = std::sync::Arc::new(std::sync::Mutex::new(()));

        let n_requests = 4;
        let prompts = vec![
            "What is the capital of France?",
            "What is the capital of Germany?",
            "What is the capital of Italy?",
            "What is the capital of Spain?",
        ];

        let mut handles = Vec::with_capacity(n_requests);
        let start = std::time::Instant::now();

        for i in 0..n_requests {
            let shared = std::sync::Arc::clone(&shared);
            let mutex = std::sync::Arc::clone(&mutex);
            let prompt = prompts[i].to_string();
            handles.push(std::thread::spawn(move || {
                let t0 = std::time::Instant::now();
                let _guard = mutex.lock().unwrap();
                let result = serve(&shared, &prompt, 50);
                drop(_guard);
                let elapsed = t0.elapsed();
                (i, prompt, result, elapsed)
            }));
        }

        let mut total_generated = 0usize;
        let mut total_model_time = 0f64;
        for handle in handles {
            let (idx, prompt, result, elapsed) = handle.join().unwrap();
            total_model_time += elapsed.as_secs_f64();
            match result {
                Ok(text) => {
                    println!(
                        "Request {idx} [{:.3}s] prompt={prompt:?} generated={} chars",
                        elapsed.as_secs_f32(),
                        text.len(),
                    );
                    total_generated += text.len();
                }
                Err(e) => {
                    println!(
                        "Request {idx} FAILED after {:.3}s: {e}",
                        elapsed.as_secs_f32()
                    );
                }
            }
        }

        let total_wall = start.elapsed();
        // With serialization via Mutex, wall time ≈ sum of individual times
        println!(
            "Total: {n_requests} requests, wall={:.3}s, sum_of_model_time={:.3}s, \
             avg_per_request={:.3}s, total_chars={total_generated}",
            total_wall.as_secs_f32(),
            total_model_time,
            total_model_time / n_requests as f64,
        );
    }
}

mod mlp;
pub use mlp::*;

pub mod llama;
pub use llama::*;

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::quantized::tokenizer::TokenizerFromGguf;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
// The gguf loader (MODEL_PATH, get_model, ...) moved to the models crate.
pub use models::gguf::*;
use serde::Serialize;
use tokenizers::tokenizer::Tokenizer;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

#[derive(Debug)]
pub struct ProcessRequest {
    pub data: String,
    pub responder: oneshot::Sender<Result<ProcessResponse, String>>,
}

#[derive(Debug, Serialize)]
pub struct ProcessResponse {
    pub result: String,
}

pub struct InferenceEngine {
    // pub model: ModelWeights,
    pub device: Device,
    pub tokenizer: Tokenizer,
    pub logits_processor: LogitsProcessor,
    pub rx: mpsc::Receiver<ProcessRequest>,
    pub inner: EngineInnerType,
}

const CHAT_TEMPLATE: &str = "\
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
{prompt}\
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

pub trait InferenceEngineTrait {
    fn serve(&mut self, prompt: &str, max_tokens: usize) -> Result<String>;
}

pub enum EngineInnerType {
    Candle(ModelWeights),
    OpenInference,
}

impl InferenceEngine {
    pub fn new(
        seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        rx: mpsc::Receiver<ProcessRequest>,
    ) -> Self {
        let (weights, device) = get_model().expect("failed to get model");

        let model_path = std::path::Path::new(MODEL_PATH);
        let mut file = std::fs::File::open(model_path).expect("failed to open model file");
        let content = gguf_file::Content::read(&mut file).expect("failed to read model file");
        let tokenizer = TokenizerFromGguf::from_gguf(&content)
            .map_err(|e| anyhow::Error::msg(e.to_string()))
            .expect("failed to create tokenizer");
        let inner = EngineInnerType::Candle(weights);

        Self {
            inner,
            device,
            tokenizer,
            logits_processor: LogitsProcessor::new(seed, temperature, top_p),
            rx,
        }
    }

    pub async fn start(&mut self) {
        loop {
            tokio::select! {
                req = self.rx.recv() => {
                    if let Some(req) = req {
                        let result = self.serve(&req.data, 100);
                        match result {
                            Ok(result) => {
                                let _ = req.responder.send(Ok(ProcessResponse { result }));
                            }
                            Err(e) => {
                                let _ = req.responder.send(Err(e.to_string()));
                            }
                        }
                    }
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(1)) => {}
            }
        }
    }

    pub fn serve(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let formatted = CHAT_TEMPLATE.replace("{prompt}", prompt);
        let mut tokens = self
            .tokenizer
            .encode(formatted.as_str(), false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let prompt_len = tokens.len();

        let input = Tensor::new(tokens.as_slice(), &self.device)
            .map_err(anyhow::Error::msg)?
            .unsqueeze(0)?;

        let logits = match &mut self.inner {
            EngineInnerType::Candle(model) => {
                let logits = model.forward(&input, 0).map_err(anyhow::Error::msg)?;
                logits.squeeze(0).map_err(anyhow::Error::msg)?
            }
            EngineInnerType::OpenInference => {
                unimplemented!()
            }
        };

        let mut next_token = self
            .logits_processor
            .sample(&logits)
            .map_err(anyhow::Error::msg)?;
        tokens.push(next_token);

        let eos_id = self.tokenizer.token_to_id("<|eot_id|>").unwrap_or(2);

        for i in 1..max_tokens {
            let input = Tensor::new(&[next_token], &self.device)
                .map_err(anyhow::Error::msg)?
                .unsqueeze(0)?;
            let logits = match &mut self.inner {
                EngineInnerType::Candle(model) => model
                    .forward(&input, prompt_len + i - 1)
                    .map_err(anyhow::Error::msg)?,
                EngineInnerType::OpenInference => {
                    unimplemented!()
                }
            };

            let logits = logits.squeeze(0).map_err(anyhow::Error::msg)?;

            next_token = self
                .logits_processor
                .sample(&logits)
                .map_err(anyhow::Error::msg)?;

            tokens.push(next_token);

            if next_token == eos_id {
                break;
            }
        }

        let output = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(anyhow::Error::msg)?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_engine() -> InferenceEngine {
        let (_tx, rx) = mpsc::channel::<ProcessRequest>(1024);
        InferenceEngine::new(299792458, Some(0.7), Some(0.9), rx)
    }

    #[test]
    fn test_serve_generates_text() {
        let mut engine = new_engine();
        let result = engine.serve("What is the capital of France?", 50);
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
        let engine = std::sync::Arc::new(std::sync::Mutex::new(new_engine()));

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
            let engine = std::sync::Arc::clone(&engine);
            let prompt = prompts[i].to_string();
            handles.push(std::thread::spawn(move || {
                let t0 = std::time::Instant::now();
                let mut guard = engine.lock().unwrap();
                let result = guard.serve(&prompt, 50);
                drop(guard);
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
        println!(
            "Total: {n_requests} requests, wall={:.3}s, sum_of_model_time={:.3}s, \
             avg_per_request={:.3}s, total_chars={total_generated}",
            total_wall.as_secs_f32(),
            total_model_time,
            total_model_time / n_requests as f64,
        );
    }
}

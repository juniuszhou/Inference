use arrow::array::{Array, LargeStringArray, StringArray};
use candle_core::{CudaDevice, DType, Device, Tensor};
use candle_nn::{
    loss,
    optim::{AdamW, Optimizer, ParamsAdamW},
    Module, VarBuilder, VarMap,
};
use candle_usage::{get_reader, TransformerCasualLLM, TransformerCasualLLMConfig};
use clap::Parser;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

pub struct Trainer {
    transformer: TransformerCasualLLM,
    varmap: VarMap,
    config: TransformerCasualLLMConfig,
    device: Device,
}

impl Trainer {
    pub fn new(config: TransformerCasualLLMConfig) -> Self {
        let device = Device::Cuda(CudaDevice::new_with_stream(0).unwrap());
        let varmap = VarMap::new();

        let api = Api::new().expect("Failed to create HF API");
        let repo = api.model("gpt2".to_string());
        let tokenizer_path = repo
            .get("tokenizer.json")
            .expect("Failed to download tokenizer");
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");
        let config = config.with_vocab_size(tokenizer.get_vocab_size(true));

        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self {
            transformer: TransformerCasualLLM::new(config.clone(), vb).unwrap(),
            varmap,
            config,
            device,
        }
    }

    pub fn train(&mut self) -> Result<(), candle_core::Error> {
        eprintln!("Training...");

        let tokenizer = Tokenizer::from_pretrained("bert-base-cased".to_string(), None)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {e}")))?;

        let reader = get_reader()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get reader: {e}")))?;

        eprintln!("get reader done ...");

        let seq_len = self.config.max_seq_len;
        let vocab_size = self.config.vocab_size;
        let pad_id = self.config.pad_token_id as u32;

        let stride = seq_len;

        let mut optim = AdamW::new(self.varmap.all_vars(), ParamsAdamW::default())?;

        eprintln!("start steps");

        let mut global_step = 0usize;
        for batch_result in reader {
            let batch =
                batch_result.map_err(|e| candle_core::Error::Msg(format!("Batch error: {e}")))?;
            let col = batch
                .column_by_name("text")
                .ok_or_else(|| candle_core::Error::Msg("No 'text' column".to_string()))?;

            let mut process_row = |row_text: &str| -> Result<(), candle_core::Error> {
                let enc = tokenizer
                    .encode(row_text, true)
                    .map_err(|e| candle_core::Error::Msg(format!("Tokenize error: {e}")))?;
                let ids = enc.get_ids();
                if ids.is_empty() {
                    return Ok(());
                }
                let num_steps = ids.len().saturating_sub(1) / stride;
                for _step in 0..num_steps {
                    let start = _step * stride;
                    let end = (start + seq_len + 1).min(ids.len());
                    let tokens = &ids[start..end];
                    if tokens.len() < 2 {
                        break;
                    }

                    let input_part = &tokens[..tokens.len().min(seq_len)];
                    let target_part = &tokens[1..tokens.len().min(seq_len + 1)];

                    let mut input_ids = input_part.to_vec();
                    let mut target_ids: Vec<u32> = target_part.to_vec();
                    while input_ids.len() < seq_len {
                        input_ids.push(pad_id);
                        target_ids.push(0);
                    }

                    let input_tensor =
                        Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
                    let target_tensor =
                        Tensor::new(target_ids.as_slice(), &self.device)?.unsqueeze(0)?;

                    let logits = self.transformer.forward(&input_tensor)?;

                    let (b, s, _) = logits.shape().dims3()?;
                    let logits_2d = logits.reshape((b * s, vocab_size))?;
                    let target_1d = target_tensor.flatten_all()?;

                    let loss_value = loss::cross_entropy(&logits_2d, &target_1d)?;
                    optim.backward_step(&loss_value)?;

                    let loss_scalar = loss_value.to_scalar::<f32>()?;
                    let ppl = loss_scalar.exp();

                    eprintln!("step {global_step}: loss = {loss_scalar:.4}, perplexity = {ppl:.4}");
                    global_step += 1;
                }
                Ok(())
            };

            if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                for i in 0..arr.len() {
                    if arr.is_null(i) {
                        continue;
                    }
                    process_row(arr.value(i))?;
                }
            } else if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
                for i in 0..arr.len() {
                    if arr.is_null(i) {
                        continue;
                    }
                    process_row(arr.value(i))?;
                }
            } else {
                return Err(candle_core::Error::Msg(
                    "'text' column is not a string array".to_string(),
                ));
            }
        }

        let save_path = "model.safetensors";
        self.varmap
            .save(save_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to save model: {e}")))?;
        eprintln!("Model saved to {save_path}");

        Ok(())
    }
}

fn main() {
    let args = Args::parse();
    let mut trainer = Trainer::new(args.config());
    trainer.train().unwrap();
}

#[derive(Parser)]
#[command(author, version, about = "Train a Transformer Causal LLM")]
struct Args {
    // ── Training hyperparameters ──
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    #[arg(long, default_value_t = 128)]
    seq_len: usize,

    // ── Model architecture (maps to TransformerCasualLLMConfig) ──
    #[arg(long, default_value_t = 128)]
    d_model: usize,
    #[arg(long, default_value_t = 256)]
    d_ff: usize,
    #[arg(long, default_value_t = 8)]
    n_heads: usize,
    #[arg(long, default_value_t = 2)]
    n_layers: usize,
    #[arg(long, default_value_t = 0.1)]
    dropout: f32,
    #[arg(long, default_value_t = 50000)]
    vocab_size: usize,
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,
    #[arg(long, default_value_t = 1)]
    bos_token_id: usize,
    #[arg(long, default_value_t = 2)]
    eos_token_id: usize,
    #[arg(long, default_value_t = 0)]
    pad_token_id: usize,
    #[arg(long, default_value_t = false)]
    attn_bias: bool,
    #[arg(long, default_value_t = true)]
    attn_mask: bool,
}

impl Args {
    fn config(&self) -> TransformerCasualLLMConfig {
        TransformerCasualLLMConfig::new(
            self.d_model,
            self.n_heads,
            self.n_layers,
            self.d_ff,
            self.dropout,
            self.vocab_size,
            self.max_seq_len,
            self.pad_token_id,
            self.eos_token_id,
            self.bos_token_id,
            self.attn_bias,
            self.attn_mask,
        )
    }
}

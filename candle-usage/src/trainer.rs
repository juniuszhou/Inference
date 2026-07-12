use candle_core::{CudaDevice, DType, Device};
use candle_nn::{VarBuilder, VarMap};
use candle_usage::{TransformerCasualLLM, TransformerCasualLLMConfig};
use clap::Parser;

#[allow(dead_code)]
pub struct Trainer {
    transformer: TransformerCasualLLM,
}

impl Trainer {
    pub fn new(config: TransformerCasualLLMConfig) -> Self {
        let device = Device::Cuda(CudaDevice::new_with_stream(0).unwrap());
        // It is very important to use a VarMap, it manages the memory of the model parameters.
        // Unlike pytorch, the model defines the parameters, and optimizer updates them.
        // Rust is memory-safe, so we need to manage the memory of the model parameters ourselves.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self {
            transformer: TransformerCasualLLM::new(config, vb).unwrap(),
        }
    }

    pub fn train(&self) -> Result<(), candle_core::Error> {
        Ok(())
    }
}

fn main() {
    let args = Args::parse();
    let trainer = Trainer::new(args.config());
    trainer.train().unwrap();
}

#[derive(Parser)]
#[command(author, version, about = "Train a Transformer Causal LLM")]
struct Args {
    // ── Data paths ──
    #[arg(short, long)]
    model_path: String,
    #[arg(short, long)]
    vocab_path: String,
    #[arg(short, long)]
    train_path: String,
    #[arg(short)]
    val_path: String,

    // ── Training hyperparameters ──
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    #[arg(long, default_value_t = 128)]
    seq_len: usize,

    // ── Model architecture (maps to TransformerCasualLLMConfig) ──
    #[arg(long, default_value_t = 512)]
    d_model: usize,
    #[arg(long, default_value_t = 2048)]
    d_ff: usize,
    #[arg(long, default_value_t = 8)]
    n_heads: usize,
    #[arg(long, default_value_t = 6)]
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

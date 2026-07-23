use candle_core::Device;
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::api::sync::ApiBuilder;
use std::env;
use std::fs::File;
use std::path::PathBuf;

/*
/home/junius/.cache/huggingface/hub/models--bartowski--Llama-3.2-3B-Instruct-GGUF/
snapshots/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
*/

pub const MODEL_PATH: &str = "/home/junius/.cache/huggingface/hub/models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

pub fn get_gguf() -> anyhow::Result<()> {
    let token = env::var("HF_TOKEN").ok();
    let api = ApiBuilder::new().with_token(token).build()?;

    let model_path = api
        .model("bartowski/Llama-3.2-3B-Instruct-GGUF".to_string())
        .get("Llama-3.2-3B-Instruct-Q4_K_M.gguf")?;

    println!("Model downloaded to: {:?}", model_path);

    Ok(())
}

pub fn load_gguf() -> anyhow::Result<gguf_file::Content> {
    let model_path = PathBuf::from(MODEL_PATH);
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    Ok(content)
}

pub fn get_model() -> anyhow::Result<(ModelWeights, Device)> {
    let device = Device::cuda_if_available(0)?;
    let mut file = File::open(MODEL_PATH)?;
    let content = gguf_file::Content::read(&mut file)?;
    let model = ModelWeights::from_gguf(content, &mut file, &device)?;
    Ok((model, device))
}

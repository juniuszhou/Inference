use anyhow::Result;
use arrow::array::{Array, LargeStringArray, StringArray};
use candle_core::{CudaDevice, DType, Device, IndexOp, Tensor};
use candle_nn::{
    loss,
    optim::{AdamW, Optimizer, ParamsAdamW},
};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama::Cache;
use candle_usage::get_reader;
use models::llama::{LlamaExt, load_llama};
use tokenizers::Tokenizer;

const EOS_TOKEN: &str = "</s>";

#[test]
fn test_llama_training() -> Result<()> {
    let (model, varmap, config) = load_llama(true)?;

    let tokenizer = Tokenizer::from_pretrained("bert-base-cased".to_string(), None)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {e}")))?;

    let reader =
        get_reader().map_err(|e| candle_core::Error::Msg(format!("Failed to get reader: {e}")))?;

    let seq_len = config.max_position_embeddings;

    let stride = seq_len;

    let device = Device::Cuda(CudaDevice::new_with_stream(0)?);
    let mut optim = AdamW::new(varmap.all_vars(), ParamsAdamW::default())?;

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
            eprintln!("ids length: {:?}", ids.len());
            if ids.is_empty() {
                return Ok(());
            }
            let num_steps = ids.len().saturating_sub(1) / stride;
            eprintln!("num_steps: {:?}", num_steps);

            for _step in 0..num_steps {
                let start = _step * stride;
                let end = (start + seq_len + 1).min(ids.len());
                let tokens = &ids[start..end];
                if tokens.len() < 2 {
                    break;
                }
                let actual_len = tokens.len() - 1;

                let mut cache = Cache::new(true, DType::BF16, &config, &device)?;

                let mut all_logits: Vec<Tensor> = Vec::with_capacity(actual_len);

                for i in 0..actual_len {
                    let input_tok = Tensor::new(&[tokens[i]], &device)?.unsqueeze(0)?;
                    let logits = model.forward(&input_tok, i, &mut cache)?;
                    all_logits.push(logits);
                }

                if all_logits.is_empty() {
                    continue;
                }

                let logits_stacked = Tensor::cat(&all_logits, 0)?.squeeze(1)?;
                let mut target_ids: Vec<u32> = tokens[1..=actual_len].to_vec();
                while target_ids.len() < seq_len {
                    target_ids.push(0);
                }
                let target_tensor = Tensor::new(&target_ids[..actual_len], &device)?;

                let loss_value = loss::cross_entropy(&logits_stacked, &target_tensor)?;
                optim.backward_step(&loss_value)?;

                let loss_scalar = loss_value.to_scalar::<f32>()?;
                eprintln!(
                    "step {global_step}: loss = {loss_scalar:.4}, ppl = {}",
                    loss_scalar.exp()
                );
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
            return Err(
                candle_core::Error::Msg("'text' column is not a string array".into()).into(),
            );
        }
    }

    varmap
        .save("model.safetensors")
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save model: {e}")))?;
    eprintln!("Model saved to model.safetensors");

    Ok(())
}

#[test]
fn test_llama_generation() -> Result<()> {
    let (model, _varmap, config) = load_llama(false)?;

    let tokenizer = Tokenizer::from_pretrained("bert-base-cased".to_string(), None)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {e}")))?;

    let device = Device::Cuda(CudaDevice::new_with_stream(0)?);
    let mut cache = Cache::new(true, DType::BF16, &config, &device)?;
    cache.use_kv_cache = false;

    let eos_token_id = tokenizer
        .token_to_id(EOS_TOKEN)
        .map(candle_transformers::models::llama::LlamaEosToks::Single);

    let prompt = "Hello, how are you?";
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| candle_core::Error::Msg(format!("Tokenize error: {e}")))?
        .get_ids()
        .to_vec();

    let mut logits_processor = {
        LogitsProcessor::from_sampling(
            42,
            Sampling::TopK {
                k: 2,
                temperature: 0.9,
            },
        )
    };

    let prompt_len = tokens.len();

    // Prefill: process all prompt tokens at once with index_pos=0
    let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.generate(&input, 0, &mut cache)?;
    let next_token = logits_processor.sample(&logits.squeeze(0)?.i(prompt_len - 1)?)?;
    tokens.push(next_token);

    // Decode: generate 9 more tokens (total 10 new tokens)
    for i in 0..9 {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.generate(&input, prompt_len + 1 + i, &mut cache)?;
        let next_token = logits_processor.sample(&logits.squeeze(0)?.squeeze(0)?)?;
        tokens.push(next_token);

        if let Some(candle_transformers::models::llama::LlamaEosToks::Single(eos)) = eos_token_id {
            if next_token == eos {
                break;
            }
        }
    }

    let output = tokenizer
        .decode(&tokens, true)
        .map_err(|e| candle_core::Error::Msg(format!("Decode error: {e}")))?;
    println!("Generated: {output}");

    Ok(())
}

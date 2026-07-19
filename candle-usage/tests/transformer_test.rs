use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};

use candle_usage::{TransformerCasualLLM, TransformerCasualLLMConfig};

fn make_config() -> TransformerCasualLLMConfig {
    TransformerCasualLLMConfig::new(
        16,    // d_model
        2,     // n_heads
        1,     // n_layers
        32,    // d_ff
        0.0,   // dropout
        20,    // vocab_size
        64,    // max_seq_len
        0,     // pad_token_id
        1,     // eos_token_id
        2,     // bos_token_id
        false, // attn_bias
        true,  // attn_mask
    )
}

#[test]
fn test_transformer_forward() -> candle_core::Result<()> {
    let config = make_config();

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TransformerCasualLLM::new(config, vb)?;

    let input_ids = Tensor::new(&[[2u32, 5]], &device)?;
    let logits = model.forward(&input_ids)?;
    assert_eq!(logits.shape().dims(), &[1, 2, 20]);

    Ok(())
}

#[test]
fn test_var_builder() -> candle_core::Result<()> {
    let device = Device::Cpu;

    // Step 1: create VarMap + VarBuilder and build the model.
    // VarMap lazily creates parameters on first access. VarBuilder is the
    // interface the model uses to request tensors by name.
    let config = make_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TransformerCasualLLM::new(config, vb)?;

    // Step 2: iterate all parameters stored in the VarMap.
    // After model construction, VarMap contains every parameter the model created.
    let params = varmap.data().lock().unwrap();
    let count = params.len();
    eprintln!("model has {} parameters", count);
    for (name, _var) in params.iter() {
        let tensor = _var.as_tensor();
        eprintln!(
            "  {}: shape={:?} dtype={:?}",
            name,
            tensor.shape(),
            tensor.dtype()
        );
    }
    assert!(count > 0, "VarMap should contain model parameters");
    drop(params);

    // Step 3: save weights to safetensors file.
    let checkpoint_path = "/tmp/candle_test_varbuilder.safetensors";
    std::fs::remove_file(checkpoint_path).ok();
    varmap.save(checkpoint_path)?;
    assert!(std::path::Path::new(checkpoint_path).exists());
    eprintln!("saved checkpoint to {}", checkpoint_path);

    // Step 4a: load into a fresh VarMap with the same model architecture.
    // This creates new random parameters then overwrites them with saved weights.
    let mut varmap_loaded = VarMap::new();
    let vb_loaded = VarBuilder::from_varmap(&varmap_loaded, DType::F32, &device);
    let model_loaded = TransformerCasualLLM::new(make_config(), vb_loaded)?;
    varmap_loaded.load(checkpoint_path)?;
    eprintln!("loaded checkpoint into fresh VarMap");

    // Step 4b: verify by running forward pass with both models and comparing outputs.
    let input_ids = Tensor::new(&[[2u32, 5]], &device)?;
    let logits_original = model.forward(&input_ids)?;
    let logits_loaded = model_loaded.forward(&input_ids)?;

    let orig_f32 = logits_original
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let load_f32 = logits_loaded
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(orig_f32.len(), load_f32.len());
    for (i, (a, b)) in orig_f32.iter().zip(load_f32.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "output mismatch at [{}]: {} vs {}",
            i,
            a,
            b
        );
    }
    eprintln!("verified forward outputs match after VarMap save/load");

    // Step 4c: skip VarMap and use VarBuilder directly from file.
    let vb_from_file =
        unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint_path], DType::F32, &device)? };
    let model_from_file = TransformerCasualLLM::new(make_config(), vb_from_file)?;
    let logits_from_file = model_from_file.forward(&input_ids)?;

    let file_f32 = logits_from_file
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(orig_f32.len(), file_f32.len());
    for (i, (a, b)) in orig_f32.iter().zip(file_f32.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "output mismatch at [{}]: {} vs {}",
            i,
            a,
            b
        );
    }
    eprintln!("verified forward outputs match with VarBuilder::from_mmaped_safetensors");

    std::fs::remove_file(checkpoint_path).ok();
    Ok(())
}

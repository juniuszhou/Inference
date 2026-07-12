use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};

use candle_usage::{TransformerCasualLLM, TransformerCasualLLMConfig};

#[test]
fn test_transformer_forward() -> candle_core::Result<()> {
    let config = TransformerCasualLLMConfig::new(
        64,    // d_model
        4,     // n_heads
        2,     // n_layers
        256,   // d_ff
        0.0,   // dropout
        100,   // vocab_size
        512,   // max_seq_len
        0,     // pad_token_id
        1,     // eos_token_id
        2,     // bos_token_id
        false, // attn_bias
        true,  // attn_mask
    );

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TransformerCasualLLM::new(config, vb)?;

    let input_ids = Tensor::new(&[[2u32, 5, 10, 25, 1]], &device)?;
    let logits = model.forward(&input_ids)?;
    assert_eq!(logits.shape().dims(), &[1, 5, 100]);

    let varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);
    let model2 = TransformerCasualLLM::new(
        TransformerCasualLLMConfig::new(64, 4, 2, 256, 0.0, 100, 512, 0, 1, 2, false, true),
        vb2,
    )?;

    let logits2 = model2.forward(&input_ids)?;
    assert_eq!(logits2.shape().dims(), &[1, 5, 100]);

    Ok(())
}

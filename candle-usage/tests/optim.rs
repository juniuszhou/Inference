use candle_core::{DType, Device};
use candle_nn::{
    optim::{AdamW, Optimizer, ParamsAdamW},
    VarBuilder, VarMap,
};
use candle_usage::Mlp;

#[test]
fn test_optim() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F16, &device);
    let _mlp = Mlp::new(32, 64, vb);

    let optim = AdamW::new(varmap.all_vars(), ParamsAdamW::default())?;
    drop(optim);
    Ok(())
}

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    loss,
    optim::{AdamW, Optimizer, ParamsAdamW},
    VarBuilder, VarMap,
};
use candle_usage::Mlp;

#[test]
fn test_optim() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F16, &device);
    let mlp = Mlp::new(32, 64, vb);

    let mut optim = AdamW::new(varmap.all_vars(), ParamsAdamW::default())?;

    for i in 0..10 {
        let input = Tensor::new(&[[1.0f32; 32]], &device)?;
        let output = mlp.forward(&input)?;
        let target = Tensor::zeros((1, 32), DType::F16, &device)?;
        let loss_value = loss::mse(&output, &target)?;

        optim.backward_step(&loss_value)?;
        eprintln!(
            "step {}: loss = {:?}",
            i,
            loss_value
                .to_dtype(DType::F16)?
                .flatten_all()?
                .to_vec1::<f32>()?[0]
        );
    }
    Ok(())
}

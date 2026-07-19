use candle_core::{backend::BackendDevice, CudaDevice, DType, Device};
use candle_nn::{linear, Linear, VarBuilder, VarMap};
use candle_usage::Mlp;

#[allow(dead_code)]
struct MLPTest {
    pub mlp: Mlp,
    pub project: Linear,
}

impl MLPTest {
    fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Self {
        let mlp_vb = vb.pp("mlp");
        let mlp = Mlp::new(32, 64, mlp_vb);
        let project_vb = vb.pp("project");
        let project = linear(d_model, d_ff, project_vb).unwrap();
        Self { mlp, project }
    }
}

#[test]
fn test_mlp() -> candle_core::Result<()> {
    let device = Device::Cuda(CudaDevice::new(0)?);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F16, &device);
    let vb = vb.pp("test");
    let _test = MLPTest::new(32, 64, vb);

    for (name, param) in varmap.data().lock().unwrap().iter() {
        eprintln!(
            "  {}: shape={:?} dtype={:?}",
            name,
            param.as_tensor().shape(),
            param.as_tensor().dtype()
        );
    }
    Ok(())
}

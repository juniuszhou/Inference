use cudaforge::KernelBuilder;
use std::path::Path;

fn main() -> cudaforge::Result<()> {
    let cu_file = Path::new("vecadd.cu");

    let _ptx_output = KernelBuilder::new().source_files([&cu_file]).build_ptx()?;

    Ok(())
}

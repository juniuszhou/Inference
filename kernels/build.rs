use cudaforge::KernelBuilder;
use std::path::Path;

fn main() -> cudaforge::Result<()> {
    println!("cargo:rerun-if-changed=vecadd.cu");
    let cu_file = Path::new("vecadd.cu");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let _ptx_output = KernelBuilder::new()
        .source_files([&cu_file])
        .out_dir(Path::new(&out_dir))
        .build_ptx()?;

    Ok(())
}

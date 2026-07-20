use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let kernel_dir = Path::new("kernels");
    let cu_file = kernel_dir.join("vecadd.cu");
    let ptx_file = Path::new(&out_dir).join("vecadd.ptx");

    println!("cargo:rerun-if-changed={}", cu_file.display());

    let nvcc = std::env::var("CUDA_NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let status = Command::new(&nvcc)
        .args([
            "-ptx",
            "-arch=sm_80",
            cu_file.to_str().unwrap(),
            "-o",
            ptx_file.to_str().unwrap(),
        ])
        .status()
        .expect("failed to execute nvcc");

    if !status.success() {
        panic!("nvcc compilation failed");
    }
}

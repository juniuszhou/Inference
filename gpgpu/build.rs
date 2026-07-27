use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let src_dir = Path::new("src");

    for entry in std::fs::read_dir(src_dir).expect("Failed to read src dir") {
        let entry = entry.expect("Failed to read dir entry");
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "cu") {
            let file_name = path.file_stem().unwrap().to_str().unwrap();
            let ptx_path = Path::new(&out_dir).join(format!("{}.ptx", file_name));

            let output = Command::new("nvcc")
                .args(["--ptx", "-arch=sm_75"])
                .arg(&path)
                .arg("-o")
                .arg(&ptx_path)
                .current_dir(".")
                .output()
                .expect("Failed to compile CUDA");

            if !output.status.success() {
                eprintln!("nvcc stderr: {}", String::from_utf8_lossy(&output.stderr));
                panic!("Failed to compile {}", path.display());
            }

            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

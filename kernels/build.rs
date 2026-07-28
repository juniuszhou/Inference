use std::path::Path;
use std::process::Command;

fn compile_cu_files(dir: &Path, out_dir: &str, object_files: &mut Vec<std::path::PathBuf>) {
    if !dir.exists() {
        return;
    }
    for entry in std::fs::read_dir(dir).expect("Failed to read dir") {
        let entry = entry.expect("Failed to read dir entry");
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "cu") {
            let file_name = path.file_stem().unwrap().to_str().unwrap();
            let ptx_path = Path::new(out_dir).join(format!("{}.ptx", file_name));
            let obj_path = Path::new(out_dir).join(format!("{}.o", file_name));

            // Compile PTX for GPU kernel loading at runtime
            let ptx_output = Command::new("nvcc")
                .args(["--ptx", "-arch=sm_75"])
                .arg(&path)
                .arg("-o")
                .arg(&ptx_path)
                .output()
                .expect("Failed to compile CUDA to PTX");
            if !ptx_output.status.success() {
                eprintln!(
                    "nvcc stderr: {}",
                    String::from_utf8_lossy(&ptx_output.stderr)
                );
                panic!("Failed to compile {} to PTX", path.display());
            }

            // Compile to object file for static linking
            let obj_output = Command::new("nvcc")
                .args(["-c", "-arch=sm_75"])
                .arg(&path)
                .arg("-o")
                .arg(&obj_path)
                .output()
                .expect("Failed to compile CUDA to object file");
            if !obj_output.status.success() {
                eprintln!(
                    "nvcc stderr: {}",
                    String::from_utf8_lossy(&obj_output.stderr)
                );
                panic!("Failed to compile {} to object file", path.display());
            }

            object_files.push(obj_path);
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let mut object_files = Vec::new();

    // Search both root and src/ for .cu files
    compile_cu_files(Path::new("."), &out_dir, &mut object_files);
    compile_cu_files(Path::new("src"), &out_dir, &mut object_files);

    // Create static library from all object files
    if !object_files.is_empty() {
        let lib_path = Path::new(&out_dir).join("libkernels.a");
        let mut cmd = Command::new("ar");
        cmd.arg("crs").arg(&lib_path);
        for obj in &object_files {
            cmd.arg(obj);
        }
        let ar_output = cmd.output().expect("Failed to create static library");
        if !ar_output.status.success() {
            eprintln!("ar stderr: {}", String::from_utf8_lossy(&ar_output.stderr));
            panic!("Failed to create static library");
        }

        println!("cargo:rustc-link-lib=static=kernels");
        println!("cargo:rustc-link-search={}", out_dir);
    }
}

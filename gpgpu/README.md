# GPGPU Workspace

This is a comprehensive workspace for GPU programming and CUDA development.

## Key Features

- MVSHMEM Module: GPU memory sharing functionality similar to NVSHMEM API, implemented in Rust
- Multiple CUDA examples demonstrating GPU programming concepts
- Test suite with comprehensive unit tests

## Projects

1. **mvshmem** - GPU Memory Sharing Module
2. **shmem** - CUDA Program (C++) demonstrating GPU memory sharing
3. **main.rs** - Main binary using cudarc for GPU operations

## Usage

### Initialize MVSHMEM and use it:
```rust
use crate::mvshmem::*;

fn main() -> anyhow::Result<()> {
    init_mvshmem(0)?;
    
    let key = MvshmemKey::new("my_shared_data", 1024);
    let data = vec![1i32, 2i32, 3i32, 4i32, 5i32];
    write_to_mvshmem(&key, &data)?;
    
    let read_data = read_from_mvshmem(&key)?;
    println!("Read data: {:?}", read_data);
    
    Ok(())
}
```

## Build Instructions

```bash
cargo build
cargo run --bin gpgpu
cargo test
```

## Testing

Run the MVSHMEM tests:
```bash
cargo test --test mvshmem-test
```

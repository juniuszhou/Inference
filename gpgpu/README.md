# GPGPU Workspace

This is a comprehensive workspace for GPU programming and CUDA development.

## Key Features

- MVSHMEM Module: GPU memory sharing functionality similar to NVSHMEM API, implemented in Rust
- Multiple CUDA examples demonstrating GPU programming concepts
- CUDA unit tests for all kernel sources
- Makefile for building and testing

## Projects

1. **mvshmem** - GPU Memory Sharing Module (Rust)
2. **shmem** - CUDA Program (C++) demonstrating GPU memory sharing
3. **main.rs** - Main binary using cudarc for GPU operations

## Build Instructions

### CUDA Programs

```bash
make _build    # Compile all CUDA examples
make clean     # Remove build artifacts
```

### CUDA Unit Tests

```bash
make test      # Build and run all CUDA unit tests
```

### Flash Attention Benchmark

```bash
make run-flash # Run flash attention benchmark (requires PyTorch)
```

## Testing

### CUDA Unit Tests

Run all CUDA unit tests:
```bash
make test
```

Tests cover:
- `tests/test_add.cu` — Tests `add` and `addSlice` kernels from `src/add.cu`
- `tests/test_multiply.cu` — Tests `matmul` kernel from `src/multiply.cu`
- `tests/test_shmem.cu` — Tests `cuda_put_kernel` from `src/shmem.cu`
- `tests/test_nccl.cu` — Placeholder for `src/nccl_demo.cu`

### Rust Tests (requires CUDA)

```bash
cargo test
```

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

## Makefile Targets

| Target | Description |
|--------|-------------|
| `_build` | Compile all CUDA examples |
| `test` | Build and run all CUDA unit tests |
| `run` | Build and run standalone CUDA demos |
| `clean` | Remove build artifacts |
| `run-flash` | Run flash attention benchmark (requires PyTorch) |
| `help` | Show this help |

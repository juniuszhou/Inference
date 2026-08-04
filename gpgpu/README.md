# GPGPU Workspace

This is a comprehensive workspace for GPU programming and CUDA development.

## Key Features

- MVSHMEM Module: GPU memory sharing functionality similar to NVSHMEM API, implemented in Rust
- Multiple CUDA examples demonstrating GPU programming concepts
- CUDA unit tests that reuse source kernels via headers
- Makefile for building and testing

## Projects

1. **mvshmem** - GPU Memory Sharing Module (Rust)
2. **shmem** - CUDA Program (C++) demonstrating GPU memory sharing
3. **main.rs** - Main binary using cudarc for GPU operations

## Directory Structure

```
src/
├── add.cuh          # Kernel declarations (include guard)
├── add.cu           # add + addSlice kernels + standalone main
├── multiply.cuh     # Kernel declarations
├── multiply.cu      # matmul kernel + standalone main
├── shmem.cuh        # Kernel declarations
├── shmem.cu         # cuda_put_kernel + standalone main
├── basic.cuh        # Kernel declarations
├── basic.cu         # add1D kernel + standalone main
├── nccl_demo.cuh    # (empty header)
├── nccl_demo.cu     # (empty, stub main)
└── lib.rs / main.rs / mvshmem.rs  # Rust code

tests/
├── test_add.cu      # Tests add.cuh kernels
├── test_multiply.cu # Tests multiply.cuh kernels
├── test_shmem.cu    # Tests shmem.cuh kernels
└── test_nccl.cu     # Placeholder for nccl_demo
```

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

Tests directly `#include` the `.cuh` headers from `src/`, reusing the same kernel implementations:

```bash
make test
```

| Test | Tests | Source |
|------|-------|--------|
| `tests/test_add` | `add`, `addSlice` | `src/add.cuh` |
| `tests/test_multiply` | `matmul` (64x64, 128x32, 32x128) | `src/multiply.cuh` |
| `tests/test_shmem` | `cuda_put_kernel` | `src/shmem.cuh` |
| `tests/test_nccl` | (placeholder) | `src/nccl_demo.cuh` |

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
| `run` | Build and run standalone demos |
| `clean` | Remove build artifacts |
| `run-flash` | Run flash attention benchmark (needs PyTorch) |
| `help` | Show this help |

## Running Individual Tests

```bash
./tests/test_add
./tests/test_multiply
./tests/test_shmem
./tests/test_nccl
```

# AGENTS.md

## Workspace

Rust workspace with 2 members (`candle-usage`, `server`) + 1 unmanaged crate (`oxide/` — no Cargo.toml, not in workspace).

## CI pipeline order

`cargo fmt -- --check` → `cargo clippy -- -D warnings` → `cargo build --release` → `cargo test --release`

## Packages

- **`server/`** — Axum web server (`127.0.0.1:3000`). `GET /health` → `{"status":"ok"}`, `POST /echo` → accepts `{message, value}`, returns `{result, processed_value: value*2}`. Tests co-located in `server/src/main.rs`.
- **`candle-usage/`** — Candle ML framework demo (uses `candle-core` from git, CUDA). Tests co-located in `candle-usage/tests/`.
- **`oxide/`** — Experimental CUDA kernel project (custom `cuda_device`/`cuda_core` crates). Not buildable standalone.

## Commands

| Action | Command |
|--------|---------|
| Build all | `cargo build` |
| Build single pkg | `cargo build -p server` / `cargo build -p candle-usage` |
| Test single pkg | `cargo test -p server` / `cargo test -p candle-usage` |
| Test single test | `cargo test test_health_endpoint` |
| Verbose test | `cargo test test_name -- --nocapture` |
| Lint | `cargo clippy -- -D warnings` |
| Format check | `cargo fmt -- --check` |

## PyTorch → Candle API map

| PyTorch | Candle |
|---------|--------|
| `torch.Tensor([[1, 2], [3, 4]])` | `Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?` |
| `torch.zeros((2, 2))` | `Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?` |
| `tensor[:, :4]` | `tensor.i((.., ..4))?` |
| `tensor.view((2, 2))` | `tensor.reshape((2, 2))?` |
| `a.matmul(b)` | `a.matmul(&b)?` |
| `a + b` | `&a + &b` |
| `tensor.to(device="cuda")` | `tensor.to_device(&Device::new_cuda(0)?)?` |
| `tensor.to(dtype=torch.float16)` | `tensor.to_dtype(&DType::F16)?` |
| `torch.save({"A": A}, "model.bin")` | `candle::safetensors::save(&HashMap::from([("A", A)]), "model.safetensors")?` |
| `weights = torch.load("model.bin")` | `candle::safetensors::load("model.safetensors", &device)?` |

## Quirks

- `CLAUDE.md` is stale (omits `candle-usage` workspace member, claims single-package repo). **Use AGENTS.md as source of truth.**
- `candle-usage` tests require CUDA (uses `Device::Cuda`). Will fail on GPU-less CI.
- `server` depends on workspace-level `axum`, `tokio`, `serde`, `serde_json`. `candle-usage` uses `candle-core`/`candle-nn` from git.

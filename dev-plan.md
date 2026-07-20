# Development Plan: Candle-Oxide Inference Engine

## Overview

Build a learning-focused inference engine similar to vLLM using **Candle** (Rust ML framework) for model inference and **Oxide** (custom CUDA kernels) for performance-critical operations. The engine will support model import, token generation, post-training tuning, KV caching, and multi-GPU parallel inference.

---

## 1. Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| ML Framework | **Candle** (`candle-core`, `candle-nn`, `candle-transformers`) | Pure Rust, CUDA support, clean API, no Python dependency |
| Custom Kernels | **Oxide** (custom CUDA via cudaforge) | Hand-written CUDA kernels for critical paths (attention, KV cache ops) |
| HTTP Server | **Axum** | Already in workspace, async, composable |
| Serialization | **Serde** / **Serde JSON** | Already in workspace |
| Testing | **Built-in `#[test]`** | Rust-native, no extra framework needed |
| Benchmarking | **Criterion** / custom harness | Compare throughput/latency against vLLM & SGLang |

---

## 2. Architecture

```
                    ┌─────────────────────────┐
                    │      HTTP / CLI API       │
                    │   (Axum server or CLI)    │
                    └──────────┬────────────────┘
                               │
                    ┌──────────▼────────────────┐
                    │   Scheduler / Router       │
                    │   (request batching,       │
                    │    scheduling policy)      │
                    └──────────┬────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Model Loader    │ │   KV Cache Mgr   │ │   Tokenizer      │
│ (Candle safetensors│ │ (page-based alloc,│ │ (tokenizers-rs)  │
│  + custom formats)│ │  memory pool)    │ │                  │
└────────┬─────────┘ └────────┬─────────┘ └──────────────────┘
         │                    │
         ▼                    ▼
┌──────────────────────────────────────────────┐
│          Inference Engine Core                │
│  (prefill, decode, attention, MLP, sampler)  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Candle    │ │ Candle   │ │ Oxide CUDA   │ │
│  │ Ops       │ │ Models   │ │ Kernels      │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
└──────────────────────────────────────────────┘
```

### Directory Structure

```
Inference/
├── server/                  # Axum HTTP server (existing)
├── candle-usage/            # Candle demos (existing)
├── kernels/                 # Custom CUDA kernels via cudaforge (existing)
│   ├── vecadd.cu           # Example kernel (existing)
│   ├── build.rs            # cudaforge build script (existing)
│   └── src/
│       ├── main.rs         # Kernel launcher
│       ├── cudarc_call.rs  # Runtime PTX example
│       └── tests/
├── models/                  # Model definitions & loading (new)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── config.rs       # Model configuration (JSON parsing)
│       ├── loader.rs       # Safetensors loading
│       ├── llama.rs        # LLaMA model definition
│       └── quant.rs        # (Optional) Quantization helpers
├── engine/                  # Inference engine (new)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── scheduler.rs    # Request batching & scheduling
│       ├── kv_cache.rs     # Page-based KV cache
│       ├── prefill.rs      # Prefill phase
│       ├── decode.rs       # Autoregressive decode
│       ├── sampler.rs      # Token sampling (top-k, top-p, temp)
│       └── parallel.rs     # Multi-GPU (tensor parallelism)
├── bench/                   # Benchmarking (new)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # Benchmark runner
│       └── compare.rs      # Compare against vLLM/SGLang APIs
├── dev-plan.md              # This file
└── Cargo.toml               # Workspace (existing, updated)
```

---

## 3. Implementation Phases

### Phase 0: Foundation (Weeks 1-2)

**Goals:** Workspace setup, CUDA kernel pipeline, basic model loading

- [x] Workspace with `server`, `candle-usage`, `kernels` crates
- [x] CUDA kernel build pipeline via `cudaforge` (vecadd example working)
- [x] `cudarc`-based kernel launching (test passing)
- [ ] Add `models` crate to workspace
- [ ] Implement `llama.rs` — LLaMA model with Candle (`candle-transformers` patterns)
  - [ ] `LlamaConfig` from JSON (following HF format)
  - [ ] `LlamaModel` struct with `forward()` returning logits
  - [ ] `LlamaForCausalLM` with `generate()` scaffold
- [ ] Implement `kv_cache.rs` — simple contiguous KV cache
- [ ] Implement `engine` crate scaffold with scheduler + sampler
- [ ] Single-request end-to-end: "load → prefill → decode → sample"
- [ ] Unit tests: model forward pass matches HF reference

**Key Deliverables:**
- Models can be loaded from HuggingFace safetensors
- Single prompt → single token generation works
- KV cache populated correctly

### Phase 1: Production Inference Loop (Weeks 3-4)

**Goals:** Correct token generation, batching, sampling

- [ ] **Scheduler**: `Scheduler` struct managing request queue
  - [ ] Add/remove requests dynamically
  - [ ] Continuous batching (add new requests mid-generation)
  - [ ] Max tokens / stop conditions per request
- [ ] **Prefill**: Batched prefill with KV cache population
  - [ ] Support variable-length prompts in a batch
  - [ ] Padded or varlen attention
- [ ] **Decode**: Batched autoregressive generation
  - [ ] Shared KV cache across decode steps
  - [ ] Causal attention mask
- [ ] **Sampler**: Top-k, top-p, temperature
  - [ ] Efficient batched sampling (rejection sampling)
  - [ ] Seeded RNG for reproducibility
- [ ] **Tokenizer**: Integration with `tokenizers` crate
  - [ ] HF tokenizer JSON loading
  - [ ] Encode/decode with special tokens
- [ ] Integration test: `curl` → server → tokens out

**Key Deliverables:**
- Batched inference (multiple requests concurrent)
- Correct sampling (matches HF generation output)
- HTTP endpoint: `POST /generate` with streaming SSE

### Phase 2: Performance & Custom Kernels (Weeks 5-6)

**Goals:** Optimize with Oxide CUDA kernels, page-based KV cache

- [ ] **Oxide Kernels:**
  - [ ] `fused_add_rms_norm` — RMSNorm with residual add
  - [ ] `fused_rotary_embedding` — RoPE with position encoding
  - [ ] `flash_attention_v2` — Tiled online softmax attention
  - [ ] `paged_kv_cache` — Page-based KV store/load
  - [ ] Cross-CUDA kernel validation tests
- [ ] **Paged KV Cache:**
  - [ ] Page table allocator (vLLM-style block manager)
  - [ ] Copy-on-write for shared prefixes
  - [ ] Kernel: page scatter/gather for attention
- [ ] **Memory optimization:**
  - [ ] KV cache memory pooling
  - [ ] Tensor memory reuse across requests
  - [ ] `candle_core::cuda` memory management

**Key Deliverables:**
- Custom CUDA kernels matching or exceeding Candle's built-in ops
- Paged KV cache reduces memory fragmentation
- Benchmarks against baseline Candle implementation

### Phase 3: Multi-GPU (Weeks 7-8)

**Goals:** Tensor parallelism across multiple GPUs

- [ ] **Tensor Parallelism:**
  - [ ] Row/column partitioning for linear layers
  - [ ] All-reduce communication between GPUs
  - [ ] NCCL integration (via `cudarc` NCCL support)
- [ ] **Pipeline Parallelism (optional):**
  - [ ] Layer partitioning across devices
  - [ ] Micro-batching for pipeline fill/drain
- [ ] **Multi-GPU Scheduler:**
  - [ ] Device-aware request routing
  - [ ] Balanced memory allocation across devices
- [ ] Integration tests with 2+ GPU setup

**Key Deliverables:**
- Multi-GPU inference matches single-GPU numerical output
- Performance scales with GPU count (near-linear for fp16)
- CI with multi-GPU smoke test (if hardware available)

### Phase 4: Post-Tuning & Advanced Features (Weeks 9-10)

- [ ] **Post-Tuning:**
  - [ ] LoRA adapter loading (low-rank adaptation)
  - [ ] LoRA fusion at inference time (no separate weights)
  - [ ] Multiple LoRA adapters per model (hot-swap)
  - [ ] Prefix caching (shared prompt prefixes)
- [ ] **Quantization:**
  - [ ] FP16 → FP8 kernel support
  - [ ] Weight-only quantization (INT8, INT4)
  - [ ] KV cache quantization (FP8)
- [ ] **Speculative Decoding (optional):**
  - [ ] Draft model integration
  - [ ] Verification with target model

**Key Deliverables:**
- LoRA adapters loadable without engine restart
- Quantized inference within 1% accuracy loss
- 2x+ throughput with speculative decoding

### Phase 5: Benchmarking & Iteration (Weeks 11-12)

- [ ] **Benchmark Harness:**
  - [ ] Throughput (tokens/sec) at various batch sizes
  - [ ] Time-to-first-token (TTFT) latency
  - [ ] Inter-token latency (ITL) distribution
  - [ ] Memory usage tracking
- [ ] **Comparison Suite:**
  - [ ] vLLM HTTP API comparison (same prompts, same model)
  - [ ] SGLang comparison
  - [ ] Automated nightly benchmark runs
- [ ] **Profiling & Tuning:**
  - [ ] CUDA kernel profiling (Nsight)
  - [ ] Identify bottlenecks → kernel improvements
  - [ ] Iterate on scheduling policy
- [ ] **Documentation:**
  - [ ] API reference (OpenAPI spec)
  - [ ] Architecture docs with diagrams
  - [ ] Performance tuning guide
  - [ ] Contribution guide

**Key Deliverables:**
- Published benchmark results vs vLLM/SGLang
- Profiling report with bottleneck analysis
- Documentation website (or markdown book)

---

## 4. Custom CUDA Kernel Roadmap (Oxide)

| Kernel | Priority | Description | Candle Equivalent |
|--------|----------|-------------|-------------------|
| `fused_add_rms_norm` | P0 | RMS norm + residual in one kernel | `candle_nn::rms_norm` + add |
| `fused_rotary_embedding` | P0 | RoPE with cos/sin precomputation | Manual impl |
| `flash_attention_v1` | P1 | Tiled softmax attention w/o large SRAM | `candle_nn::attn` |
| `paged_kv_cache_copy` | P1 | Scatter/gather for page tables | N/A (custom) |
| `silu_mul` | P0 | SiLU + element-wise multiply (SwiGLU) | `candle_nn::silu` + mul |
| `cross_entropy_fused` | P1 | Softmax + cross-entropy loss | Softmax + log + gather |
| `fp8_quant_dequant` | P2 | FP8 quantization / dequantization | N/A |

P0 = must-have for baseline, P1 = performance critical, P2 = nice to have

---

## 5. Testing Strategy

| Level | Scope | Tools | Frequency |
|-------|-------|-------|-----------|
| Unit | Individual ops, kernels, modules | `#[test]` | Every commit |
| Integration | End-to-end generation matches HF | `#[test]` + golden outputs | Every PR |
| Kernel | CUDA kernel correctness vs reference | `#[test]` with CUDA | Every PR |
| Regression | Fixed bugs don't reappear | Named test cases | Every commit |
| Performance | Throughput, latency, memory | `cargo bench` + custom harness | Nightly |
| Comparison | vs vLLM / SGLang | HTTP API comparison script | Weekly |

### Golden Test Data

Store reference outputs from HuggingFace for:
- LLaMA 7B single forward pass (logits)
- LLaMA 7B 10-token generation (token IDs)
- LLaMA 7B with KV cache (cache state)

---

## 6. Documentation Plan

| Doc | Format | Contents |
|-----|--------|----------|
| `README.md` | Markdown | Overview, quick start, architecture diagram |
| `docs/architecture.md` | Markdown | Detailed architecture, data flow diagrams |
| `docs/api.md` | Markdown | HTTP API reference with examples |
| `docs/kernels.md` | Markdown | Custom CUDA kernel documentation |
| `docs/benchmarks.md` | Markdown | Performance results, comparison charts |
| `docs/contributing.md` | Markdown | How to contribute, coding standards |
| Rustdoc | Doc comments | All public API documented |

---

## 7. Performance Targets

| Metric | Target vs Candle baseline | Target vs vLLM |
|--------|--------------------------|----------------|
| Throughput (tok/s) | 1.5x (via custom kernels) | Within 2x |
| TTFT (ms) | 1.2x (via fused kernels) | Within 1.5x |
| Memory (KV cache) | 2x efficiency (paged) | Comparable |
| Batch scaling | Linear up to batch 64 | Within 2x |

---

## 8. Milestones & Checkpoints

```
M0 (Week 2):  Single-prompt generation working
              ↓
M1 (Week 4):  Batched generation, HTTP endpoint
              ↓
M2 (Week 6):  Custom CUDA kernels, paged KV cache
              ↓
M3 (Week 8):  Multi-GPU tensor parallelism
              ↓
M4 (Week 10): LoRA, quantization, advanced features
              ↓
M5 (Week 12): Benchmarks, docs, vLLM comparison
```

Each milestone includes:
- All preceding items complete
- Tests passing in CI
- Benchmark baseline recorded
- Documentation drafted

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Candle API changes | Blocked model loading | Pin Candle version, read changelog |
| CUDA kernel debugging | Slow development | Start with simple kernels, unit test each |
| Multi-GPU communication | Complex debugging | Use cudarc NCCL, test with 2 GPUs first |
| vLLM features too broad | Scope creep | Focus on core features, defer edge cases |
| LLM model size > GPU memory | Unable to test | Support CPU offloading, start with 7B model |

---

## 10. Getting Started (Quick Start)

```bash
# Clone and build
git clone <repo>
cd Inference

# Build all crates
cargo build --release

# Run tests
cargo test --release

# Launch inference server
cargo run --release -p engine -- --model meta-llama/Llama-2-7b-chat-hf

# Send a request
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 100}'

# Run benchmarks
cargo run --release -p bench -- --model meta-llama/Llama-2-7b-chat-hf
```

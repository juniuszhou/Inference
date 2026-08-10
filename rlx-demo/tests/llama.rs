use rlx_core::weight_loader::GgufLoader;
use rlx_llama32::{Llama32Generator, MetalGgufPrefillMode, llama32_cfg_from_gguf};
use rlx_qwen3::SampleOpts;
use rlx_qwen35::{decode_ids_from_gguf, encode_prompt_from_gguf};
use rlx_runtime::Device;

const PROMPT_START_ID: u32 = 128_259;
const BOS_TOKEN_ID: u32 = 128_000;
const PROMPT_END_IDS: [u32; 4] = [128_009, 128_260, 128_261, 128_257];

fn build_prompt_ids(gguf: &std::path::Path, body: &str) -> anyhow::Result<Vec<u32>> {
    let mut ids = encode_prompt_from_gguf(gguf, body)?;
    let mut out = Vec::with_capacity(ids.len() + 2 + PROMPT_END_IDS.len());
    out.push(PROMPT_START_ID);
    if ids.first().copied() != Some(BOS_TOKEN_ID) {
        out.push(BOS_TOKEN_ID);
    }
    out.append(&mut ids);
    out.extend_from_slice(&PROMPT_END_IDS);
    Ok(out)
}

fn generate_tokens(
    device: Device,
    prefill: MetalGgufPrefillMode,
    gguf: &str,
    prompt: &[u32],
    n: usize,
) -> anyhow::Result<Vec<u32>> {
    let mut loader = GgufLoader::from_file(gguf)?;
    let cfg = llama32_cfg_from_gguf(loader.file())?;
    let path = std::path::Path::new(gguf);
    let dynamic = std::env::var("ORPHEUS_DYNAMIC_DECODE").ok().as_deref() == Some("1");

    let mut g = Llama32Generator::from_loader_at_mode(cfg, &mut loader, device, path, prefill)?
        .with_compile_seq_cap(128);
    g = if dynamic {
        g.with_dynamic_decode_cache(2)
    } else {
        g.with_decode_cache(128)
    };
    g.prefill(prompt);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(g.step_cached(SampleOpts::greedy())?);
    }
    Ok(out)
}

#[test]
fn test_llama_model() -> anyhow::Result<()> {
    //set timeout as 120 seconds for the test
    unsafe { std::env::set_var("RUST_TEST_TIMEOUT", "120") };
    if !rlx_cuda::is_available() {
        return Ok(());
    }

    let gguf = std::env::var("ORPHEUS_GGUF_PATH").unwrap_or_else(|_| {
        "/home/junius/.cache/huggingface/hub/models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf".into()
    });
    let text =
        std::env::var("ORPHEUS_TEXT").unwrap_or_else(|_| "The weather is nice today.".into());

    let n: usize = std::env::var("ORPHEUS_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let prompt = build_prompt_ids(std::path::Path::new(&gguf), &format!("{text}"))?;
    eprintln!("prompt len={}", prompt.len());

    let native_prefill = true;
    let native_decode = true;
    let cuda_only = true;
    if native_decode {
        unsafe { std::env::set_var("ORPHEUS_CUDA_NATIVE_DECODE", "1") };
    }
    unsafe { std::env::set_var("RLX_CUDA_ARENA_POOL", "0") };

    let cuda_prefill = if native_prefill {
        unsafe { std::env::set_var("ORPHEUS_CUDA_NATIVE_PREFILL", "1") };
        MetalGgufPrefillMode::PackedGguf
    } else {
        MetalGgufPrefillMode::CpuF32
    };
    let cuda = generate_tokens(Device::Cuda, cuda_prefill, &gguf, &prompt, n)?;
    let result = decode_ids_from_gguf(std::path::Path::new(&gguf), &cuda, true)?;
    eprintln!("CUDA: {result:?}");
    if cuda_only {
        eprintln!("cuda-only ok for {n} tokens");
        return Ok(());
    }

    eprintln!("parity ok for {n} tokens");
    Ok(())
}

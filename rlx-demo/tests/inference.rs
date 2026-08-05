use rlx::prelude::*;
use rlx_flow::blocks::{
    DecodeRopeParamsStage, LlamaDecodeLayerSpec, LlamaDecoderSpec, RopeTablesStage,
};
use rlx_flow::{FlowStage, MapWeights, ModelFlow, SideOutputs};

// ── Tiny Llama-like config ─────────────────────────────────────────────
const B: usize = 1; // batch
const S: usize = 4; // prompt length
const H: usize = 8; // hidden
const NH: usize = 2; // num attention heads
const HD: usize = 4; // head dim
const NKV: usize = 1; // num kv heads (GQA group = 2)
const INTER: usize = 16; // mlp intermediate
const VOCAB: usize = 10;
const EPS: f32 = 1e-5;
const N_ROT: usize = HD;

/// Deterministic, bounded, non-degenerate fills.
fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.13 + seed).sin() * 0.5)
        .collect()
}

fn weights() -> MapWeights {
    let lp = "model.layers.0";
    let mut w = MapWeights::default();
    w.insert(
        "model.embed_tokens.weight",
        fill(VOCAB * H, 0.1),
        vec![VOCAB, H],
    );
    w.insert(
        format!("{lp}.input_layernorm.weight"),
        vec![1.0; H],
        vec![H],
    );
    w.insert(
        format!("{lp}.self_attn.q_proj.weight"),
        fill(NH * HD * H, 0.2),
        vec![NH * HD, H],
    );
    w.insert(
        format!("{lp}.self_attn.k_proj.weight"),
        fill(NKV * HD * H, 0.3),
        vec![NKV * HD, H],
    );
    w.insert(
        format!("{lp}.self_attn.v_proj.weight"),
        fill(NKV * HD * H, 0.4),
        vec![NKV * HD, H],
    );
    w.insert(
        format!("{lp}.self_attn.o_proj.weight"),
        fill(H * NH * HD, 0.5),
        vec![H, NH * HD],
    );
    w.insert(
        format!("{lp}.post_attention_layernorm.weight"),
        vec![1.0; H],
        vec![H],
    );
    w.insert(
        format!("{lp}.mlp.gate_proj.weight"),
        fill(INTER * H, 0.6),
        vec![INTER, H],
    );
    w.insert(
        format!("{lp}.mlp.up_proj.weight"),
        fill(INTER * H, 0.7),
        vec![INTER, H],
    );
    w.insert(
        format!("{lp}.mlp.down_proj.weight"),
        fill(H * INTER, 0.8),
        vec![H, INTER],
    );
    w.insert("model.norm.weight", vec![1.0; H], vec![H]);
    w
}

fn argmax(v: &[f32]) -> f32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if x > &v[best] {
            best = i;
        }
    }
    best as f32
}

/// End-to-end autoregressive inference on `rlx`:
///
/// 1. Build a **prefill** graph (prompt → hidden → full K/V exported as side
///    outputs) and run it once to get the initial logits + layer-0 KV cache.
/// 2. Build a **decode** graph that *concatenates* past K/V (`past_k_0` /
///    `past_v_0` inputs), attends causally, and re-emits the grown K/V.
/// 3. Loop decode: feed the last token, the KV, read back the next token and
///    the grown KV — the KV cache is passed between runs, never re-processed.
#[test]
fn test_inference() {
    if !rlx_cuda::is_available() {
        return;
    }

    let half_dim = N_ROT / 2;

    // ── 1. Prefill graph ─────────────────────────────────────────────────
    let mut w_pf = weights();
    let kv_sink = SideOutputs::new();
    let flow_pf = ModelFlow::new("prefill")
        .input("ids", Shape::new(&[B, S], DType::F32))
        .rope_tables(RopeTablesStage::param(
            S,
            half_dim,
            fill(S * half_dim, 0.9),
            fill(S * half_dim, 1.0),
        ))
        .zero_beta(H)
        .embed("model.embed_tokens.weight")
        .llama_kv_tap(0, HD, EPS, &kv_sink)
        .llama_prefill_layer(
            0,
            LlamaDecoderSpec {
                num_heads: NH,
                head_dim: HD,
                n_rot: N_ROT,
                num_kv_heads: NKV,
                eps: EPS,
                mask: MaskKind::Causal,
                hidden_shape: Shape::new(&[B, S, H], DType::F32),
                rope_style: rlx::ir::RopeStyle::NeoX,
            },
        )
        .final_norm(EPS)
        // logits for the last token
        .lm_head(VOCAB, H, true); // tied embeddings

    let built = flow_pf.build(&mut w_pf).expect("prefill build");
    let kv_ids = kv_sink.drain(); // [k_0, v_0] node ids → graph outputs
    let built = built.with_extra_hir_outputs(kv_ids);
    let (g_pf, params_pf) = built.into_graph_parts().expect("prefill graph parts");

    let mut pf = rlx_cuda::backend::CudaExecutable::compile(g_pf);
    for (k, v) in &params_pf {
        pf.set_param(k.as_str(), v.as_slice());
    }

    let prompt = [1.0_f32, 3.0, 5.0, 7.0];
    // prefill outputs as
    // 0 is logits for the last token
    // 1 is k cache
    // 2 is v cache
    let pf_out = pf.run(&[("ids", &prompt)]);
    assert_eq!(pf_out.len(), 3, "prefill outputs = [logits, k0, v0]");
    let kv_elems = B * S * NKV * HD;
    assert_eq!(pf_out[1].len(), kv_elems, "k0 = [B, S, NKV, HD]");
    assert_eq!(pf_out[2].len(), kv_elems, "v0 = [B, S, NKV, HD]");
    println!("prefill logits={} kv={} elems", pf_out[0].len(), kv_elems);

    // First decode token: argmax over the last position's logits.
    let last_pos_logits = &pf_out[0][pf_out[0].len() - VOCAB..];
    let mut token = argmax(last_pos_logits);
    let mut past_k = pf_out[1].clone();
    let mut past_v = pf_out[2].clone();

    // ── 2 + 3. Decode graph and autoregressive loop ──────────────────────
    // The decode graph binds explicit `past_k_0`/`past_v_0` inputs whose
    // length is fixed at compile time, so it is rebuilt per step (the KV len
    // grows by one position each step). Cross-run, the grown KV is fed back.
    let new_tokens = 3;
    let mut generated = vec![];
    for step in 0..new_tokens {
        let past_len = S + step; // = prompt len + tokens already generated
        let mut w_dec = weights();
        let kv_out = SideOutputs::new();

        // create a new flow for each step, the past length is growing by one position each step
        let flow_dec = ModelFlow::new(format!("decode_{past_len}"))
            .input("token", Shape::new(&[B, 1], DType::F32))
            .input("past_k_0", Shape::new(&[B, past_len, NKV * HD], DType::F32))
            .input("past_v_0", Shape::new(&[B, past_len, NKV * HD], DType::F32))
            .stage(FlowStage::DecodeRopeParams(DecodeRopeParamsStage::new(
                vec![1.0_f32; half_dim],
                vec![0.0_f32; half_dim],
                half_dim,
            )))
            .zero_beta(H)
            .bind_decode_inputs(1, false, true) // binds past_k_0 / past_v_0 above
            .embed("model.embed_tokens.weight")
            .llama_decode_layer(
                0,
                LlamaDecodeLayerSpec {
                    num_heads: NH,
                    head_dim: HD,
                    n_rot: N_ROT,
                    num_kv_heads: NKV,
                    kv_group_size: NH / NKV,
                    eps: EPS,
                    use_custom_mask: false,
                    hidden_shape: Shape::new(&[B, 1, H], DType::F32),
                    rope_style: rlx::ir::RopeStyle::NeoX,
                },
                kv_out.clone(),
            )
            .final_norm(EPS)
            .lm_head(VOCAB, H, true);

        let built = flow_dec.build(&mut w_dec).expect("decode build");
        let kv_ids = kv_out.drain(); // [new_k_0, new_v_0]
        let built = built.with_extra_hir_outputs(kv_ids);
        let (g_dec, params_dec) = built.into_graph_parts().expect("decode graph parts");
        let mut dec = rlx_cuda::backend::CudaExecutable::compile(g_dec);
        for (k, v) in &params_dec {
            dec.set_param(k.as_str(), v.as_slice());
        }

        let out = dec.run(&[
            ("token", &[token]),
            ("past_k_0", &past_k),
            ("past_v_0", &past_v),
        ]);
        assert_eq!(out.len(), 3, "decode outputs = [logits, new_k0, new_v0]");
        let grown = past_k.len() + B * NKV * HD;
        assert_eq!(out[1].len(), grown, "KV grows by one position per step");
        assert_eq!(out[2].len(), grown, "KV grows by one position per step");

        token = argmax(&out[0]);
        generated.push(token as usize);
        past_k = out[1].clone();
        past_v = out[2].clone();
        println!("step {step}: token {token} (kv {} elems)", past_k.len());
    }

    assert!(!generated.is_empty());
    for t in &generated {
        assert!(*t < VOCAB, "sampled token must be in-vocab");
    }
    println!("generated {generated:?}");
}

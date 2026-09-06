//! Model config parsing: the arch tag + `Cfg` (dense-GQA switches) and the
//! `config.json` → `Cfg` parsers for Gemma / Llama / Qwen. Split out of `lib.rs`
//! (module breakdown). GLM/Nemotron carry their own cfg in `mla.rs`.
use std::path::Path;

use packet::rope::RopeScale;
use serde_json::Value;

/// Which checkpoint architecture we are compiling (tensor naming, norm topology,
/// activation, attention geometry, RoPE differ per arch).
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Arch {
    Gemma4,
    Llama,
    Qwen3,
    /// GPT-OSS (`model_type: "gpt_oss"`): alternating sliding(128)/full GQA attention with
    /// biases, sinks and YaRN RoPE over a top-4-of-32 MXFP4 MoE. Its own [`GptOssCfg`] and
    /// emitter (`gptoss.rs`); never reaches the dense-GQA `Cfg` path.
    #[allow(dead_code)]
    GptOss,
}

/// GPT-OSS geometry, parsed from `config.json` with every field verified present. Weight names
/// are the checkpoint's verbatim (`self_attn.{q,k,v,o}_proj.{weight,bias}`, `self_attn.sinks`,
/// `mlp.router.{weight,bias}`, `mlp.experts.{gate_up,down}_proj_{blocks,scales,bias}`).
pub(crate) struct GptOssCfg {
    pub(crate) hidden: u32,
    /// Per-expert intermediate width (`intermediate_size` IS the expert width; no dense MLP).
    pub(crate) inter: u32,
    pub(crate) layers: u32,
    pub(crate) heads: u32,
    pub(crate) kvh: u32,
    pub(crate) hd: u32,
    pub(crate) window: u32,
    pub(crate) eps: f32,
    pub(crate) vocab: u32,
    /// `layer_types[l] == "full_attention"`.
    pub(crate) is_full: Vec<bool>,
    pub(crate) theta: f64,
    pub(crate) rope_scale: RopeScale,
    pub(crate) attn_scale: f32,
    pub(crate) n_exp: u32,
    pub(crate) top_k: u32,
    /// swiglu_oai immediates: `alpha` is the reference's hard-coded 1.702, `limit` is
    /// `swiglu_limit` (7.0).
    pub(crate) swiglu_alpha: f32,
    pub(crate) swiglu_limit: f32,
    pub(crate) prefix: String,
    /// `tie_word_embeddings`; false ships a separate `lm_head.weight`.
    pub(crate) tied: bool,
}

/// GPT-OSS `config.json` -> [`GptOssCfg`]. Refuses (panics with the field named) anything the
/// emitter does not implement rather than defaulting it: a dequantized (bf16) checkpoint, an
/// explicit YaRN `attention_factor`, a non-yarn rope, a non-silu activation.
pub(crate) fn cfg_gpt_oss(v: &Value) -> GptOssCfg {
    let g = |k: &str| -> u32 {
        v[k].as_u64()
            .unwrap_or_else(|| panic!("gpt_oss config.json missing {k:?}")) as u32
    };
    let layers = g("num_hidden_layers");
    let is_full: Vec<bool> = v["layer_types"]
        .as_array()
        .expect("gpt_oss: layer_types")
        .iter()
        .map(|x| match x.as_str() {
            Some("full_attention") => true,
            Some("sliding_attention") => false,
            other => panic!("gpt_oss: unsupported layer type {other:?}"),
        })
        .collect();
    assert_eq!(
        is_full.len(),
        layers as usize,
        "gpt_oss: layer_types vs num_hidden_layers"
    );
    // The emitter binds the MXFP4 expert tensors (`*_blocks`/`*_scales`) by name. A checkpoint
    // that ships dequantized bf16 experts (`experts.gate_up_proj` [E,2I,H]) has different names
    // and needs a different op, so it is refused here rather than at the coverage gate.
    let qm = v["quantization_config"]["quant_method"].as_str();
    assert_eq!(
        qm,
        Some("mxfp4"),
        "gpt_oss: quantization_config.quant_method must be \"mxfp4\" (got {qm:?}); the emitter \
         binds the checkpoint's MXFP4 expert blocks/scales verbatim and has no bf16-expert arm"
    );
    assert_eq!(
        v["attention_bias"], true,
        "gpt_oss: attention_bias=false is not implemented"
    );
    assert_eq!(v["hidden_act"], "silu", "gpt_oss: swiglu_oai is built on silu");
    let rs = &v["rope_scaling"];
    assert_eq!(
        rs["rope_type"].as_str(),
        Some("yarn"),
        "gpt_oss: rope_scaling.rope_type must be \"yarn\" (got {:?})",
        rs["rope_type"]
    );
    // `RopeScale::Yarn` derives the attention factor from `factor`; an explicit override in the
    // config would be silently ignored, so refuse it.
    for k in ["attention_factor", "mscale", "mscale_all_dim"] {
        assert!(
            rs.get(k).is_none(),
            "gpt_oss: rope_scaling.{k} is not representable in the RoPE table recipe (GenTensor \
             is ABI-locked); remove it or add a field"
        );
    }
    let f = |k: &str| {
        rs[k].as_f64()
            .unwrap_or_else(|| panic!("gpt_oss: rope_scaling.{k}"))
    };
    let rope_scale = RopeScale::Yarn {
        factor: f("factor"),
        beta_fast: rs["beta_fast"].as_f64().unwrap_or(32.0),
        beta_slow: rs["beta_slow"].as_f64().unwrap_or(1.0),
        orig: f("original_max_position_embeddings"),
        // HF's default is true; GPT-OSS ships `truncate: false`.
        truncate: rs["truncate"].as_bool().unwrap_or(true),
    };
    let hd = g("head_dim");
    let top_k = v["num_experts_per_tok"]
        .as_u64()
        .or_else(|| v["experts_per_token"].as_u64())
        .expect("gpt_oss: num_experts_per_tok") as u32;
    let c = GptOssCfg {
        hidden: g("hidden_size"),
        inter: g("intermediate_size"),
        layers,
        heads: g("num_attention_heads"),
        kvh: g("num_key_value_heads"),
        hd,
        window: g("sliding_window"),
        eps: v["rms_norm_eps"].as_f64().expect("gpt_oss: rms_norm_eps") as f32,
        vocab: g("vocab_size"),
        is_full,
        theta: v["rope_theta"].as_f64().expect("gpt_oss: rope_theta"),
        rope_scale,
        attn_scale: 1.0 / (hd as f32).sqrt(),
        n_exp: g("num_local_experts"),
        top_k,
        swiglu_alpha: 1.702,
        swiglu_limit: v["swiglu_limit"].as_f64().unwrap_or(7.0) as f32,
        prefix: "model.".to_string(),
        tied: v["tie_word_embeddings"].as_bool().unwrap_or(false),
    };
    assert!(
        c.heads % c.kvh == 0,
        "gpt_oss: heads {} not a multiple of kv heads {}",
        c.heads,
        c.kvh
    );
    assert!(
        c.hidden % 32 == 0 && c.inter % 32 == 0,
        "gpt_oss: MXFP4 needs K % 32 == 0"
    );
    assert!(
        c.window > 0 && !c.is_full.iter().all(|&x| x),
        "gpt_oss: expected sliding layers"
    );
    crate::require_moe_topk(c.top_k, "gpt_oss");
    c
}

pub(crate) struct Cfg {
    pub(crate) arch: Arch,
    pub(crate) hidden: u32,
    pub(crate) inter: u32,
    pub(crate) layers: u32,
    pub(crate) heads: u32,
    pub(crate) hd_slide: u32,
    pub(crate) hd_full: u32,
    pub(crate) kvh_slide: u32,
    pub(crate) kvh_full: u32,
    pub(crate) window: u32,
    pub(crate) eps: f32,
    pub(crate) vocab: u32,
    pub(crate) softcap: f32,
    pub(crate) is_full: Vec<bool>,
    pub(crate) theta_slide: f64,
    pub(crate) theta_full: f64,
    pub(crate) rope_frac_full: f64,
    pub(crate) rope_scale: RopeScale,
    // Arch switches (Gemma values preserve the old behaviour exactly).
    pub(crate) attn_scale: f32, // Gemma 1.0 (q_norm absorbs it); Llama/Qwen 1/sqrt(head_dim)
    pub(crate) emb_scale: f32,  // Gemma bf16_round(sqrt(hidden)); Llama/Qwen 1.0
    pub(crate) mlp_act: u32,    // 0 = gelu_tanh (Gemma), 1 = silu (Llama/Qwen)
    pub(crate) has_qk_norm: bool, // Gemma & Qwen true; Llama false
    pub(crate) has_v_norm: bool, // Gemma weightless v_norm; Llama/Qwen false
    pub(crate) k_eq_v: bool,    // Gemma full layers share k_proj as V; Llama/Qwen false
    pub(crate) tied: bool, // reuse embed_tokens as lm_head (Gemma, Qwen); Llama has lm_head.weight
    pub(crate) prefix: String, // weight-name prefix: "model.language_model." or "model."
    // Tensor-parallel degree (Megatron sharding). 1 = single-GPU (current path, byte-identical).
    // >1 emits a DECODE-ONLY sharded blob: column-parallel q/k/v/gate/up/
    // lm_head, row-parallel o_proj/down with an XReduce all-reduce after each, attention split by
    // heads. All ranks run the ONE blob; tp-host binds each rank's 1/N weight slice and sets
    // PlowProgram.rank/n_gpu/peer_scratch/xctr. Set from --tp in main() after cfg_from.
    pub(crate) tp: u32,
    // Gemma-4 26B-A4B sparse-MoE (`enable_moe_block`). Every layer is a HYBRID dense+MoE block:
    // the dense MLP (inter) AND the top-`top_k`-of-`n_exp` softmax-routed experts (moe_inter),
    // summed via the h1+h2 sandwich. Decode-only for now.
    pub(crate) moe: bool,
    pub(crate) n_exp: u32,     // 128 routed experts
    pub(crate) top_k: u32,     // 8 experts/token
    pub(crate) moe_inter: u32, // 704 per-expert intermediate
}

pub(crate) fn cfg_from(dir: &Path) -> Cfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .unwrap();
    // Gemma-4 multimodal nests everything under `text_config` (prefix
    // "model.language_model."); the text-only "-it-text" re-export is FLAT with
    // model_type "gemma4_text" (prefix "model."). Same weights, two namings.
    if v.get("text_config").is_some() {
        return cfg_gemma(&v, false);
    }
    let mt = v["model_type"].as_str().unwrap_or("");
    if mt == "gemma4_text" {
        return cfg_gemma(&v, true);
    }
    let arch = match mt {
        "qwen3" => Arch::Qwen3,
        "llama" => Arch::Llama,
        other => panic!("unsupported model_type {other:?}"),
    };
    cfg_llama_qwen(&v, arch)
}

/// The original Gemma-4 config parse, verbatim — do not regress it. `flat` selects
/// the text-only re-export (fields at the root, "model." prefix) vs the multimodal
/// checkpoint (fields under `text_config`, "model.language_model." prefix).
fn cfg_gemma(v: &Value, flat: bool) -> Cfg {
    let t = if flat { v } else { &v["text_config"] };
    let g = |k: &str| t[k].as_u64().unwrap() as u32;
    let lt: Vec<bool> = t["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap() == "full_attention")
        .collect();
    let rp = &t["rope_parameters"];
    let c = Cfg {
        arch: Arch::Gemma4,
        hidden: g("hidden_size"),
        inter: g("intermediate_size"),
        layers: g("num_hidden_layers"),
        heads: g("num_attention_heads"),
        hd_slide: g("head_dim"),
        hd_full: g("global_head_dim"),
        kvh_slide: g("num_key_value_heads"),
        // Nullable: the E-series ships `num_global_key_value_heads: null` and reuses the
        // sliding count. `g()` would panic on the None; fall back the way
        // hf_config::synth_gemma already does, so an unsupported checkpoint reaches the
        // coverage gate and gets a diagnosis instead of an `Option::unwrap` backtrace.
        kvh_full: t["num_global_key_value_heads"]
            .as_u64()
            .map(|x| x as u32)
            .unwrap_or_else(|| g("num_key_value_heads")),
        window: g("sliding_window"),
        eps: t["rms_norm_eps"].as_f64().unwrap() as f32,
        vocab: g("vocab_size"),
        softcap: t["final_logit_softcapping"].as_f64().unwrap() as f32,
        theta_slide: rp["sliding_attention"]["rope_theta"].as_f64().unwrap(),
        theta_full: rp["full_attention"]["rope_theta"].as_f64().unwrap(),
        rope_frac_full: rp["full_attention"]["partial_rotary_factor"]
            .as_f64()
            .unwrap(),
        is_full: lt,
        rope_scale: RopeScale::None,
        attn_scale: 1.0,
        emb_scale: bf16_round((g("hidden_size") as f32).sqrt()),
        mlp_act: 0,
        has_qk_norm: true,
        has_v_norm: true,
        k_eq_v: true,
        tied: true,
        prefix: if flat {
            "model."
        } else {
            "model.language_model."
        }
        .to_string(),
        tp: 1,
        // 26B-A4B: enable_moe_block=true, num_experts=128, top_k_experts=8, moe_inter=704.
        // 12B/31B: field absent -> dense-only (moe=false).
        moe: t["enable_moe_block"].as_bool().unwrap_or(false),
        n_exp: t["num_experts"].as_u64().unwrap_or(0) as u32,
        top_k: t["top_k_experts"].as_u64().unwrap_or(0) as u32,
        moe_inter: t["moe_intermediate_size"].as_u64().unwrap_or(0) as u32,
    };
    if c.moe {
        crate::require_moe_topk(c.top_k, "gemma4 (enable_moe_block)");
    }
    c
}

/// Llama-3.1 / Qwen3: flat config, all-global attention, simple pre-norm, SwiGLU.
fn cfg_llama_qwen(v: &Value, arch: Arch) -> Cfg {
    let g = |k: &str| v[k].as_u64().unwrap() as u32;
    let hidden = g("hidden_size");
    let heads = g("num_attention_heads");
    // Qwen carries head_dim explicitly (and it is NOT hidden/heads: 2560/32 != 128); Llama omits
    // it, so it is hidden/heads = 128.
    let hd = v["head_dim"]
        .as_u64()
        .map(|x| x as u32)
        .unwrap_or(hidden / heads);
    let layers = g("num_hidden_layers");
    let theta = v["rope_theta"].as_f64().unwrap();
    // llama3 rope scaling (Llama-3.1); Qwen has rope_scaling: null.
    let rope_scale = match v.get("rope_scaling").and_then(|r| r.as_object()) {
        Some(r) if r.get("rope_type").and_then(|x| x.as_str()) == Some("llama3") => {
            RopeScale::Llama3 {
                factor: r["factor"].as_f64().unwrap(),
                low: r["low_freq_factor"].as_f64().unwrap(),
                high: r["high_freq_factor"].as_f64().unwrap(),
                orig: r["original_max_position_embeddings"].as_f64().unwrap(),
            }
        }
        // A rope_type we do not implement must be a HARD FAILURE. Falling through
        // to RopeScale::None here compiles silently-wrong rope tables, which
        // produce fluent-but-wrong text with no crash and no numeric gate that
        // catches it. Both gemma-4-12B and gemma-4-31B hit this arm.
        Some(r) => {
            let ty = r
                .get("rope_type")
                .and_then(|x| x.as_str())
                .unwrap_or("<missing>");
            panic!(
                "unsupported rope_type {ty:?} in rope_scaling: this compiler implements \
                 only \"llama3\". Compiling it as unscaled would emit wrong rope tables \
                 and produce fluent-but-wrong output. Add a RopeScale arm for {ty:?}."
            );
        }
        None => RopeScale::None,
    };
    Cfg {
        arch,
        hidden,
        inter: g("intermediate_size"),
        layers,
        heads,
        hd_slide: hd,
        hd_full: hd,
        kvh_slide: g("num_key_value_heads"),
        kvh_full: g("num_key_value_heads"),
        window: 0, // all-global: no sliding window
        eps: v["rms_norm_eps"].as_f64().unwrap() as f32,
        vocab: g("vocab_size"),
        softcap: 0.0, // no final-logit softcapping
        is_full: vec![true; layers as usize],
        theta_slide: theta,
        theta_full: theta,
        rope_frac_full: 1.0, // full rotary
        rope_scale,
        attn_scale: 1.0 / (hd as f32).sqrt(),
        emb_scale: 1.0, // no embedding scaling
        mlp_act: 1,     // SwiGLU (silu)
        has_qk_norm: arch == Arch::Qwen3,
        has_v_norm: false,
        k_eq_v: false,
        tied: v["tie_word_embeddings"].as_bool().unwrap_or(false),
        prefix: "model.".to_string(),
        tp: 1,
        moe: false, // Llama/Qwen3 dense here
        n_exp: 0,
        top_k: 0,
        moe_inter: 0,
    }
}

pub(crate) fn bf16_round(f: f32) -> f32 {
    let u = f.to_bits();
    let r = u.wrapping_add(0x7fff).wrapping_add((u >> 16) & 1);
    f32::from_bits(r & 0xffff_0000)
}

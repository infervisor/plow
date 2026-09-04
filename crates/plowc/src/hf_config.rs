//! Parse a HuggingFace `config.json` into a full-model [`LayerPlan`].
//!
//! Instead of building the full N-layer nn-graph (which OOMs on 48-layer models),
//! we synthesize the model structure from `config.json` dimensions, then build a
//! full-model plan with N copies of the block — each copy's weight tensors are
//! named to match the safetensor keys (e.g.
//! `model.language_model.layers.{L}.self_attn.q_proj.weight`).
//!
//! This is **Option A**: compile-time unroll. The runtime stays unchanged; it sees
//! a flat instruction stream covering all N layers, identical in structure to what
//! the `gemma4` binary produces.
//!
//! The architecture model here mirrors `bin/gemma4.rs::cfg_from` — the verified
//! spec, not memory. Every geometry decision below has a counterpart there:
//!
//! * **Per-layer attention geometry.** Gemma-4 `layer_types` alternates
//!   sliding/full attention; full layers use `global_head_dim` (512) and
//!   `num_global_key_value_heads`, sliding layers `head_dim` (256) and
//!   `num_key_value_heads`. Q/K/O widths, the q/k norm width and the flash shape
//!   all change per layer.
//! * **Full-attention layers have NO `v_proj`** (`attention_k_eq_v`): V is the
//!   raw k_proj output. The checkpoint genuinely lacks those tensors — naming
//!   one anyway would hard-fail the loader (or worse, not fail).
//! * **Sandwich norms + `layer_scalar`.** Four per-layer norms
//!   (input/post_attention/pre_feedforward/post_feedforward), per-head q/k norms
//!   with a real weight, a weightless v_norm, and a learned `[1]` `layer_scalar`
//!   multiplying the hidden state at the end of each layer.
//! * **Tied lm_head** comes from `tie_word_embeddings` (Gemma/Qwen true,
//!   Llama false → a real top-level `lm_head.weight`).
//! * **MoE (26B-A4B, `enable_moe_block`)**: every layer is a HYBRID dense+MoE
//!   block — router (`router.proj.weight`/`router.scale`/`router.per_expert_scale`),
//!   fused expert tensors (`experts.gate_up_proj` [E,2I,H],
//!   `experts.down_proj` [E,H,I]) and three extra sandwich norms
//!   (`pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`,
//!   `post_feedforward_layernorm_2`).
//!
//! Supported checkpoints (same set as `bin/gemma4.rs`): Gemma-4 nested
//! `text_config` (12B/26B-A4B/31B, prefix `model.language_model.`), the flat
//! `gemma4_text` re-export (prefix `model.`), and flat Llama/Qwen3 configs.
//! Anything else is a HARD error — compiling an unknown architecture from
//! defaults produces fluent-but-wrong output with no crash.

use crate::net::{NetConfig, NetOp};
use costmodel::{AttnShape, GemmShape, RowShape};
use rewrite::{LayerPlan, OpKind, OpSpec};
use schedule::ShapeBucket;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;

/// Which checkpoint architecture we are compiling — tensor naming, norm
/// topology and attention geometry all key off this. Mirrors `bin/gemma4.rs`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HfArch {
    Gemma4,
    Llama,
    Qwen3,
    /// Qwen3.5/3.8 dense hybrid: Gated DeltaNet layers interleaved with
    /// output-gated full attention.
    Qwen35,
    /// Kimi K2 (Moonshot): MLA + MoE. Same family as DeepSeek-V2 attention.
    Kimi,
    /// DeepSeek V2/V3: MLA + DeepSeekMoE. Same block structure as Kimi K2.
    DeepSeek,
    /// GLM 5.2 MoE-DSA: MLA + DeepSeekMoE + Dense-Sparse Attention indexer.
    Glm,
    /// Kimi K3 (Moonshot, `kimi_k3` → nested `text_config` `kimi_linear`):
    /// HYBRID attention — 24 MLA layers interleaved with 69 KDA (Kimi Delta
    /// Attention, linear) layers — plus a latent MoE whose 896 routed experts
    /// are mxfp4. Config is accepted here; emission refuses (see
    /// [`build_full_model_plan`] and `devgen::mla::kimi_k3_emit`).
    KimiK3,
}

/// Per-layer attention implementation. This is FIRST-CLASS per-layer data, not
/// a count: Kimi-K3 mixes softmax MLA and linear KDA layers, and which layer is
/// which decides the tensor names, the carried state and the kernel. Collapsing
/// it to "24 full / 69 kda" loses the mapping and silently mis-binds weights.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttnKind {
    /// Full (unwindowed) softmax attention — MHA/GQA or MLA.
    Full,
    /// Sliding-window softmax attention (Gemma-4).
    Sliding,
    /// Kimi Delta Attention: a LINEAR attention with carried recurrent state,
    /// short depthwise convs on q/k/v and a low-rank forget gate. Not softmax
    /// attention and not expressible with any current plow DevOp.
    Kda,
    /// Qwen3.5 Gated DeltaNet. Unlike KDA, Q/K and V have different head
    /// counts and the QKV channels share one projection and one convolution.
    GatedDeltaNet,
}

/// Geometry unique to the Qwen3.5/3.8 hybrid mixer.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Qwen35Geometry {
    pub linear_num_key_heads: i64,
    pub linear_num_value_heads: i64,
    pub linear_key_head_dim: i64,
    pub linear_value_head_dim: i64,
    pub linear_conv_kernel_dim: u32,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub attn_output_gate: bool,
    /// Official fine-grained FP8 checkpoint block geometry. `None` for BF16.
    pub fp8_weight_block_size: Option<[i64; 2]>,
    pub fp8_dynamic_activations: bool,
}

impl Qwen35Geometry {
    pub fn qkv_width(self) -> i64 {
        2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn value_width(self) -> i64 {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
}

/// Resolved model description — every field verified present (or an
/// architecture-defined constant). No silent defaults: a missing required
/// field is a compile error, not a guess.
#[derive(Debug)]
pub struct HfSynthesis {
    pub name: String,
    pub arch: HfArch,
    pub hidden_size: i64,
    pub num_attention_heads: i64,
    /// KV heads on sliding / full-attention layers (equal for Llama/Qwen).
    pub kvh_slide: i64,
    pub kvh_full: i64,
    /// Head dim on sliding / full-attention layers (equal for Llama/Qwen).
    pub hd_slide: i64,
    pub hd_full: i64,
    pub intermediate_size: i64,
    pub num_layers: u32,
    pub vocab_size: i64,
    /// Per-layer attention kind: `true` = full SOFTMAX attention. All-true for
    /// Llama/Qwen (no sliding window). Derived from [`Self::attn_kind`] — a KDA
    /// layer is `false` here because it is not softmax attention at all.
    pub is_full: Vec<bool>,
    /// Per-layer attention implementation, one entry per layer. The single
    /// source of truth; `is_full` is `attn_kind[l] == AttnKind::Full`.
    pub attn_kind: Vec<AttnKind>,
    /// Gemma full layers share k_proj output as V (no v_proj tensor).
    pub k_eq_v: bool,
    /// lm_head tied to embed_tokens (reuse the embedding weight).
    pub tied_embed: bool,
    /// Weight-name prefix: "model.language_model." (nested multimodal) or "model.".
    pub prefix: String,
    /// Gemma & Qwen have weighted q/k RMSNorms; Llama has none.
    pub has_qk_norm: bool,
    /// Gemma applies a WEIGHTLESS RMSNorm to V on every layer.
    pub has_v_norm: bool,
    // Gemma-4 26B-A4B sparse-MoE hybrid block.
    pub moe: bool,
    pub n_exp: i64,
    pub top_k: i64,
    pub moe_inter: i64,
    // --- Kimi K2 MLA (Multi-head Latent Attention) fields ---
    /// Q low-rank projection intermediate dim (0 = not MLA).
    pub q_lora_rank: i64,
    /// KV compressed latent dim.
    pub kv_lora_rank: i64,
    /// RoPE portion of Q/K head dim.
    pub qk_rope_head_dim: i64,
    /// Non-RoPE (nope) portion of Q/K head dim.
    pub qk_nope_head_dim: i64,
    /// V head dim in MLA.
    pub v_head_dim: i64,
    /// First K layers use dense MLP; rest use MoE.
    pub first_k_dense: u32,
    /// Number of shared experts (Kimi MoE; 0 for non-MoE or Gemma).
    pub n_shared_experts: i64,
    /// LATENT-MoE width (`routed_expert_hidden_size`, Kimi-K3 = 3584): the
    /// routed experts do NOT read the hidden state. The block projects
    /// `hidden → moe_latent` once (`routed_expert_down_proj`), runs every routed
    /// expert GEMM at `K = moe_latent`, and projects back with
    /// `routed_expert_up_proj`. 0 = experts read the hidden state directly
    /// (Gemma / DeepSeek / GLM). VERIFIED against the checkpoint: K3's
    /// `experts.{e}.w1.weight_packed` is `[3072, 1792]` = `[moe_inter, 3584/2]`,
    /// so `moe_latent`, NOT `moe_inter`, is the routed-expert GEMM's K.
    pub moe_latent: i64,
    /// Present only for Qwen3.5/3.8 hybrid text towers.
    pub qwen35: Option<Qwen35Geometry>,
    /// Weight dtype for GEMM projections. BF16 by default; F8E4M3 for FP8
    /// quantized checkpoints; F4 for MX microscaling. Norms/embed/activations
    /// always stay BF16 regardless of this setting.
    pub weight_dtype: nn_graph::DType,
    /// Single-block NetConfig (kept for validation + backward compat).
    pub net: NetConfig,
}

/// Synthesize full model metadata from a local HF model directory.
pub fn synthesize_from_hf_dir(dir: &Path) -> Result<HfSynthesis, String> {
    let json = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("cannot read {}/config.json: {e}", dir.display()))?;
    let slug = dir_slug(dir);
    synthesize_full(&json, slug)
}

/// Synthesize full model metadata from a config.json string.
///
/// Architecture dispatch mirrors `bin/gemma4.rs::cfg_from`: a `text_config`
/// object means the nested Gemma-4 multimodal checkpoint; `model_type`
/// "gemma4_text" is the flat text-only re-export (same weights, "model."
/// prefix); "llama"/"qwen3" are the flat dense models. Anything else errors.
pub fn synthesize_full(json: &str, name: String) -> Result<HfSynthesis, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("invalid config.json: {e}"))?;
    // Dispatch on the OUTER model_type BEFORE the `text_config` short-circuit.
    // A nested `text_config` used to mean "Gemma-4 multimodal" and nothing else;
    // Kimi-K3 also nests, so the unconditional short-circuit sent it into
    // synth_gemma and it died on `head_dim` — an error that names the wrong
    // field for the wrong architecture. Every nested arch must be claimed here
    // by name before the Gemma fallback.
    if v["model_type"].as_str() == Some("kimi_k3") {
        return synth_kimi_k3(&v, name);
    }
    if v["model_type"].as_str() == Some("qwen3_5") {
        return synth_qwen35(&v, name);
    }
    if v.get("text_config").is_some() {
        return synth_gemma(&v["text_config"], name, "model.language_model.");
    }
    let mt = v["model_type"].as_str().unwrap_or("<missing>");
    match mt {
        "gemma4_text" => synth_gemma(&v, name, "model."),
        "llama" | "qwen3" => synth_llama_qwen(&v, name, mt),
        "kimi" => synth_mla_moe(&v, name, HfArch::Kimi),
        "deepseek_v2" | "deepseek_v3" => synth_mla_moe(&v, name, HfArch::DeepSeek),
        "glm_moe_dsa" | "glm4" => synth_mla_moe(&v, name, HfArch::Glm),
        other => Err(format!(
            "unsupported model_type {other:?}: --hf-dir implements gemma4 (nested \
             text_config or gemma4_text), llama, qwen3, qwen3_5, kimi, kimi_k3, \
             deepseek_v2/v3, and glm_moe_dsa. Compiling an unknown architecture \
             from defaults would produce a silently-wrong model."
        )),
    }
}

/// Backward-compatible: synthesize just the single-block NetConfig.
pub fn synthesize_from_json(json: &str, name: String) -> Result<NetConfig, String> {
    synthesize_full(json, name).map(|s| s.net)
}

/// Derive a URL-safe slug from a directory path.
pub fn dir_slug(dir: &Path) -> String {
    dir.file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| dir.display().to_string().to_lowercase())
}

/// Fetch a required integer field or error with its name — never default.
fn req_i64(t: &Value, k: &str) -> Result<i64, String> {
    t[k].as_i64()
        .ok_or_else(|| format!("config.json missing required field {k:?}"))
}

/// Parse weight dtype from config.json's `torch_dtype` or `quantization` fields.
/// Returns BF16 by default; F8E4M3 for FP8 checkpoints; F4 for MX microscaling.
fn parse_weight_dtype(v: &Value) -> nn_graph::DType {
    // Check explicit quantization config first (e.g. "quantization": "fp8")
    if let Some(q) = v
        .get("quantization_config")
        .and_then(|q| q.get("quant_method"))
        .and_then(|m| m.as_str())
    {
        match q {
            "fp8" => return nn_graph::DType::F8E4M3,
            // OCP MX microscaling FP4 (e2m1 + one E8M0 scale per 32). Transformers
            // spells the gpt-oss / MX-quantized checkpoints `quant_method: "mxfp4"`.
            "mxfp4" | "fp4" => return nn_graph::DType::F4,
            _ => {}
        }
    }
    // Check torch_dtype string
    if let Some(dt) = v
        .get("torch_dtype")
        .or_else(|| v.get("dtype"))
        .and_then(|d| d.as_str())
    {
        match dt {
            "float8_e4m3fn" | "float8_e4m3" | "fp8" => return nn_graph::DType::F8E4M3,
            "float8_e5m2" => return nn_graph::DType::F8E5M2,
            _ => {}
        }
    }
    nn_graph::DType::BF16
}

fn synth_gemma(t: &Value, name: String, prefix: &str) -> Result<HfSynthesis, String> {
    let hidden = req_i64(t, "hidden_size")?;
    let heads = req_i64(t, "num_attention_heads")?;
    let hd_slide = req_i64(t, "head_dim")?;
    // The three shipping Gemma-4 checkpoints all carry global_head_dim; absent
    // means a genuinely uniform-geometry config, so fall back to head_dim.
    let hd_full = t["global_head_dim"].as_i64().unwrap_or(hd_slide);
    let kvh_slide = req_i64(t, "num_key_value_heads")?;
    let kvh_full = t["num_global_key_value_heads"]
        .as_i64()
        .unwrap_or(kvh_slide);
    let layers = req_i64(t, "num_hidden_layers")?;
    let vocab = req_i64(t, "vocab_size")?;
    let inter = req_i64(t, "intermediate_size")?;
    if t["use_double_wide_mlp"].as_bool() == Some(true) {
        return Err(
            "use_double_wide_mlp=true is not implemented: the gate/up/down \
                    widths would all be wrong. Add the double-wide arm before \
                    compiling this checkpoint."
                .into(),
        );
    }
    // Per-layer attention kind. Required whenever the full/sliding geometry
    // differs — guessing the pattern would compile wrong Q/K widths.
    let is_full: Vec<bool> = match t["layer_types"].as_array() {
        Some(a) => a
            .iter()
            .map(|x| x.as_str() == Some("full_attention"))
            .collect(),
        None if hd_full == hd_slide && kvh_full == kvh_slide => {
            vec![true; layers as usize]
        }
        None => {
            return Err(
                "config.json missing layer_types but global/sliding geometry \
                        differs — cannot infer which layers are full attention"
                    .into(),
            )
        }
    };
    if is_full.len() != layers as usize {
        return Err(format!(
            "layer_types has {} entries but num_hidden_layers is {layers}",
            is_full.len()
        ));
    }
    let moe = t["enable_moe_block"].as_bool().unwrap_or(false);
    let (n_exp, top_k, moe_inter) = if moe {
        (
            req_i64(t, "num_experts")?,
            req_i64(t, "top_k_experts")?,
            req_i64(t, "moe_intermediate_size")?,
        )
    } else {
        (0, 0, 0)
    };
    let mut s = HfSynthesis {
        name,
        arch: HfArch::Gemma4,
        hidden_size: hidden,
        num_attention_heads: heads,
        kvh_slide,
        kvh_full,
        hd_slide,
        hd_full,
        intermediate_size: inter,
        num_layers: layers as u32,
        vocab_size: vocab,
        attn_kind: attn_kind_from_full(&is_full),
        is_full,
        k_eq_v: t["attention_k_eq_v"].as_bool().unwrap_or(true),
        tied_embed: t["tie_word_embeddings"].as_bool().unwrap_or(true),
        prefix: prefix.to_string(),
        has_qk_norm: true,
        has_v_norm: true,
        moe,
        n_exp,
        top_k,
        moe_inter,
        q_lora_rank: 0,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        qk_nope_head_dim: 0,
        v_head_dim: 0,
        first_k_dense: 0,
        n_shared_experts: 0,
        moe_latent: 0,
        qwen35: None,
        weight_dtype: parse_weight_dtype(t),
        net: NetConfig {
            name: String::new(),
            hidden: 0,
            ops: vec![],
        },
    };
    s.net = build_net_config(&s);
    Ok(s)
}

/// Softmax-only archs: `is_full` fully determines the per-layer kind.
fn attn_kind_from_full(is_full: &[bool]) -> Vec<AttnKind> {
    is_full
        .iter()
        .map(|&f| if f { AttnKind::Full } else { AttnKind::Sliding })
        .collect()
}

fn synth_llama_qwen(v: &Value, name: String, mt: &str) -> Result<HfSynthesis, String> {
    let hidden = req_i64(v, "hidden_size")?;
    let heads = req_i64(v, "num_attention_heads")?;
    // Qwen carries head_dim explicitly (NOT hidden/heads); Llama omits it.
    let hd = v["head_dim"].as_i64().unwrap_or(hidden / heads.max(1));
    let kvh = req_i64(v, "num_key_value_heads")?;
    let layers = req_i64(v, "num_hidden_layers")?;
    let arch = if mt == "qwen3" {
        HfArch::Qwen3
    } else {
        HfArch::Llama
    };
    let mut s = HfSynthesis {
        name,
        arch,
        hidden_size: hidden,
        num_attention_heads: heads,
        kvh_slide: kvh,
        kvh_full: kvh,
        hd_slide: hd,
        hd_full: hd,
        intermediate_size: req_i64(v, "intermediate_size")?,
        num_layers: layers as u32,
        vocab_size: req_i64(v, "vocab_size")?,
        is_full: vec![true; layers as usize],
        attn_kind: vec![AttnKind::Full; layers as usize],
        k_eq_v: false,
        tied_embed: v["tie_word_embeddings"].as_bool().unwrap_or(false),
        prefix: "model.".to_string(),
        has_qk_norm: arch == HfArch::Qwen3,
        has_v_norm: false,
        moe: false,
        n_exp: 0,
        top_k: 0,
        moe_inter: 0,
        q_lora_rank: 0,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        qk_nope_head_dim: 0,
        v_head_dim: 0,
        first_k_dense: 0,
        n_shared_experts: 0,
        moe_latent: 0,
        qwen35: None,
        weight_dtype: parse_weight_dtype(v),
        net: NetConfig {
            name: String::new(),
            hidden: 0,
            ops: vec![],
        },
    };
    s.net = build_net_config(&s);
    Ok(s)
}

fn synth_qwen35(v: &Value, name: String) -> Result<HfSynthesis, String> {
    let t = v
        .get("text_config")
        .filter(|c| c.is_object())
        .ok_or("qwen3_5 config.json has no `text_config` object")?;
    let sub = t["model_type"].as_str().unwrap_or("<missing>");
    if sub != "qwen3_5_text" {
        return Err(format!(
            "qwen3_5 text_config.model_type is {sub:?}, expected \"qwen3_5_text\""
        ));
    }

    let hidden = req_i64(t, "hidden_size")?;
    let heads = req_i64(t, "num_attention_heads")?;
    let kvh = req_i64(t, "num_key_value_heads")?;
    let hd = req_i64(t, "head_dim")?;
    let layers = req_i64(t, "num_hidden_layers")?;
    let layer_types = t["layer_types"]
        .as_array()
        .ok_or("qwen3_5 text_config.layer_types is missing or not an array")?;
    if layer_types.len() != layers as usize {
        return Err(format!(
            "qwen3_5 layer_types has {} entries but num_hidden_layers is {layers}",
            layer_types.len()
        ));
    }
    let attn_kind: Vec<AttnKind> = layer_types
        .iter()
        .enumerate()
        .map(|(l, kind)| match kind.as_str() {
            Some("full_attention") => Ok(AttnKind::Full),
            Some("linear_attention") => Ok(AttnKind::GatedDeltaNet),
            other => Err(format!(
                "qwen3_5 layer_types[{l}] is {other:?}; expected \"full_attention\" or \
                 \"linear_attention\""
            )),
        })
        .collect::<Result<_, _>>()?;
    let is_full = attn_kind
        .iter()
        .map(|&kind| kind == AttnKind::Full)
        .collect();

    let conv_kernel = req_i64(t, "linear_conv_kernel_dim")?;
    let linear_conv_kernel_dim = u32::try_from(conv_kernel)
        .ok()
        .filter(|&v| v > 0)
        .ok_or_else(|| {
            format!("linear_conv_kernel_dim must fit u32 and be > 0, got {conv_kernel}")
        })?;
    let rope = t
        .get("rope_parameters")
        .filter(|r| r.is_object())
        .ok_or("qwen3_5 text_config.rope_parameters is missing or not an object")?;
    let partial_rotary_factor = rope["partial_rotary_factor"]
        .as_f64()
        .ok_or("qwen3_5 rope_parameters.partial_rotary_factor is required")?;
    let rope_theta = rope["rope_theta"]
        .as_f64()
        .ok_or("qwen3_5 rope_parameters.rope_theta is required")?;
    let attn_output_gate = t["attn_output_gate"]
        .as_bool()
        .ok_or("qwen3_5 text_config.attn_output_gate is required")?;
    for (field, expected) in [
        ("hidden_act", "silu"),
        ("mamba_ssm_dtype", "float32"),
        ("output_gate_type", "swish"),
    ] {
        let actual = t[field]
            .as_str()
            .ok_or_else(|| format!("qwen3_5 text_config.{field} is required"))?;
        if actual != expected {
            return Err(format!(
                "qwen3_5 text_config.{field} is {actual:?}, expected {expected:?}"
            ));
        }
    }
    if rope["rope_type"].as_str() != Some("default")
        || rope["mrope_interleaved"].as_bool() != Some(true)
    {
        return Err("qwen3_5 requires rope_type=default with mrope_interleaved=true".to_string());
    }
    if !attn_output_gate {
        return Err("qwen3_5 requires attn_output_gate=true".to_string());
    }
    if t["attention_bias"].as_bool() != Some(false) {
        return Err("qwen3_5 requires attention_bias=false".to_string());
    }

    let (fp8_weight_block_size, fp8_dynamic_activations) = match v.get("quantization_config") {
        None => (None, false),
        Some(q) => {
            let block = q["weight_block_size"]
                .as_array()
                .filter(|a| a.len() == 2)
                .and_then(|a| Some([a[0].as_i64()?, a[1].as_i64()?]))
                .ok_or("qwen3_5 FP8 weight_block_size must contain two integers")?;
            if q["quant_method"].as_str() != Some("fp8")
                || q["fmt"].as_str() != Some("e4m3")
                || q["activation_scheme"].as_str() != Some("dynamic")
                || block != [128, 128]
            {
                return Err(format!(
                    "qwen3_5 unsupported quantization_config: expected dynamic fp8/e4m3 \
                         with weight_block_size=[128,128], got method={:?} fmt={:?} \
                         activation={:?} block={block:?}",
                    q["quant_method"], q["fmt"], q["activation_scheme"]
                ));
            }
            (Some(block), true)
        }
    };

    let qwen35 = Qwen35Geometry {
        linear_num_key_heads: req_i64(t, "linear_num_key_heads")?,
        linear_num_value_heads: req_i64(t, "linear_num_value_heads")?,
        linear_key_head_dim: req_i64(t, "linear_key_head_dim")?,
        linear_value_head_dim: req_i64(t, "linear_value_head_dim")?,
        linear_conv_kernel_dim,
        partial_rotary_factor,
        rope_theta,
        attn_output_gate,
        fp8_weight_block_size,
        fp8_dynamic_activations,
    };
    if qwen35.linear_num_key_heads <= 0
        || qwen35.linear_num_value_heads <= 0
        || qwen35.linear_key_head_dim <= 0
        || qwen35.linear_value_head_dim <= 0
    {
        return Err("qwen3_5 linear-attention head counts and dimensions must be > 0".into());
    }
    if qwen35.linear_num_value_heads % qwen35.linear_num_key_heads != 0 {
        return Err(format!(
            "qwen3_5 linear_num_value_heads={} is not divisible by linear_num_key_heads={}",
            qwen35.linear_num_value_heads, qwen35.linear_num_key_heads
        ));
    }

    Ok(HfSynthesis {
        name: name.clone(),
        arch: HfArch::Qwen35,
        hidden_size: hidden,
        num_attention_heads: heads,
        kvh_slide: kvh,
        kvh_full: kvh,
        hd_slide: hd,
        hd_full: hd,
        intermediate_size: req_i64(t, "intermediate_size")?,
        num_layers: layers as u32,
        vocab_size: req_i64(t, "vocab_size")?,
        is_full,
        attn_kind,
        k_eq_v: false,
        tied_embed: t["tie_word_embeddings"].as_bool().unwrap_or(false),
        prefix: "model.language_model.".to_string(),
        has_qk_norm: true,
        has_v_norm: false,
        moe: false,
        n_exp: 0,
        top_k: 0,
        moe_inter: 0,
        q_lora_rank: 0,
        kv_lora_rank: 0,
        qk_rope_head_dim: 0,
        qk_nope_head_dim: 0,
        v_head_dim: 0,
        first_k_dense: 0,
        n_shared_experts: 0,
        moe_latent: 0,
        qwen35: Some(qwen35),
        weight_dtype: parse_weight_dtype(v),
        net: NetConfig {
            name,
            hidden,
            ops: vec![],
        },
    })
}

pub fn ensure_layer_plan_supported(synth: &HfSynthesis) -> Result<(), String> {
    let Some(q) = synth.qwen35 else {
        return Ok(());
    };
    let full_q_width = synth.num_attention_heads * synth.hd_full * 2;
    Err(format!(
        "qwen3_5 frontend parsed the hybrid text tower, but LayerPlan emission is not yet \
         faithful: Gated DeltaNet requires shared qkv+conv width {}, value width {}, and \
         unequal key/value heads ({}/{}); full attention requires packed query+gate width \
         {full_q_width} plus an output gate. The current plan IR has no recurrent-state, \
         causal-depthwise-conv, or attention-output-gate op, so emission is refused.",
        q.qkv_width(),
        q.value_width(),
        q.linear_num_key_heads,
        q.linear_num_value_heads,
    ))
}

/// MLA + MoE architecture (Kimi K2 / DeepSeek V2/V3). Parses DeepSeek-V2-style
/// MLA geometry (q_lora_rank, kv_lora_rank, qk_rope/nope head dims, v_head_dim)
/// and MoE parameters (n_routed_experts, num_experts_per_tok,
/// moe_intermediate_size, first_k_dense_replace, n_shared_experts).
fn synth_mla_moe(v: &Value, name: String, arch: HfArch) -> Result<HfSynthesis, String> {
    let hidden = req_i64(v, "hidden_size")?;
    let heads = req_i64(v, "num_attention_heads")?;
    let layers = req_i64(v, "num_hidden_layers")?;
    let vocab = req_i64(v, "vocab_size")?;
    let inter = req_i64(v, "intermediate_size")?;

    // MLA geometry (all required for Kimi K2).
    let q_lora_rank = req_i64(v, "q_lora_rank")?;
    let kv_lora_rank = req_i64(v, "kv_lora_rank")?;
    let qk_rope_head_dim = req_i64(v, "qk_rope_head_dim")?;
    let qk_nope_head_dim = req_i64(v, "qk_nope_head_dim")?;
    let v_head_dim = req_i64(v, "v_head_dim")?;

    // MoE parameters.
    let n_exp = req_i64(v, "n_routed_experts")?;
    let top_k = req_i64(v, "num_experts_per_tok")?;
    let moe_inter = req_i64(v, "moe_intermediate_size")?;
    let first_k_dense = v["first_k_dense_replace"].as_i64().unwrap_or(0) as u32;
    let n_shared_experts = v["n_shared_experts"].as_i64().unwrap_or(0);

    // The effective "head dim" for flash attention in MLA is qk_nope + qk_rope
    // (the full Q/K head width after the absorbed projection).
    let hd = qk_nope_head_dim + qk_rope_head_dim;

    let mut s = HfSynthesis {
        name,
        arch,
        hidden_size: hidden,
        num_attention_heads: heads,
        kvh_slide: heads, // MLA: all heads attend (no GQA)
        kvh_full: heads,
        hd_slide: hd,
        hd_full: hd,
        intermediate_size: inter,
        num_layers: layers as u32,
        vocab_size: vocab,
        is_full: vec![true; layers as usize],
        attn_kind: vec![AttnKind::Full; layers as usize],
        k_eq_v: false,
        tied_embed: v["tie_word_embeddings"].as_bool().unwrap_or(false),
        prefix: "model.".to_string(),
        has_qk_norm: false,
        has_v_norm: false,
        moe: n_exp > 0,
        n_exp,
        top_k,
        moe_inter,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        v_head_dim,
        first_k_dense,
        n_shared_experts,
        moe_latent: 0,
        qwen35: None,
        weight_dtype: parse_weight_dtype(v),
        net: NetConfig {
            name: String::new(),
            hidden: 0,
            ops: vec![],
        },
    };
    s.net = build_net_config(&s);
    Ok(s)
}

/// Kimi-K3 (`model_type: "kimi_k3"`). A MULTIMODAL wrapper whose `text_config`
/// is a `kimi_linear` model: MLA on 24 layers, KDA (linear attention) on 69, a
/// latent MoE with 896 mxfp4 routed experts + 2 bf16 shared experts.
///
/// TEXT ONLY is in scope. `vision_config` is refused by name rather than
/// ignored — silently dropping a vision tower produces a model that loads, runs
/// and is wrong on every image prompt.
///
/// Key names differ from DeepSeek/GLM and must NOT be reused blindly:
/// `num_experts` (not `n_routed_experts`), `num_experts_per_token` (not
/// `num_experts_per_tok`), `num_shared_experts` (not `n_shared_experts`).
/// `req_i64` on the DeepSeek spelling would hard-error; a `unwrap_or(0)` would
/// silently emit a dense model.
///
/// **`linear_attn_config` layer lists are 1-BASED.** `configuration_kimi_k3.py`
/// `KimiLinearConfig::is_kda_layer` tests `(layer_idx + 1) in kda_layers`, and
/// the checkpoint confirms it: `full_attn_layers` starts at 4 and the MLA
/// tensors (`self_attn.q_a_proj`) live on 0-based layers 3, 7, 11, … So every
/// entry is converted to 0-based HERE, once, and the rest of the compiler sees
/// only 0-based indices.
fn synth_kimi_k3(v: &Value, name: String) -> Result<HfSynthesis, String> {
    if v.get("vision_config")
        .map(|c| !c.is_null())
        .unwrap_or(false)
    {
        return Err(
            "kimi_k3 carries a vision_config (MoonViT tower + mm_projector) and plow \
             implements the TEXT tower only. Compiling the text tower alone would load, \
             run, and be silently wrong on every image prompt, so this is refused rather \
             than ignored. Strip `vision_config` from config.json to compile the text \
             tower explicitly."
                .into(),
        );
    }
    let t = v
        .get("text_config")
        .filter(|c| c.is_object())
        .ok_or("kimi_k3 config.json has no `text_config` object")?;
    let sub = t["model_type"].as_str().unwrap_or("<missing>");
    if sub != "kimi_linear" {
        return Err(format!(
            "kimi_k3 text_config.model_type is {sub:?}, expected \"kimi_linear\" — the \
             geometry below is only known to hold for kimi_linear"
        ));
    }

    let hidden = req_i64(t, "hidden_size")?;
    let heads = req_i64(t, "num_attention_heads")?;
    let layers = req_i64(t, "num_hidden_layers")?;
    let vocab = req_i64(t, "vocab_size")?;
    let inter = req_i64(t, "intermediate_size")?;

    // MLA geometry — same schema as DeepSeek/GLM, same spelling.
    let q_lora_rank = req_i64(t, "q_lora_rank")?;
    let kv_lora_rank = req_i64(t, "kv_lora_rank")?;
    let qk_rope_head_dim = req_i64(t, "qk_rope_head_dim")?;
    let qk_nope_head_dim = req_i64(t, "qk_nope_head_dim")?;
    let v_head_dim = req_i64(t, "v_head_dim")?;

    // MoE — Kimi spellings, all required (see the doc comment).
    let n_exp = req_i64(t, "num_experts")?;
    let top_k = req_i64(t, "num_experts_per_token")?;
    let moe_inter = req_i64(t, "moe_intermediate_size")?;
    let moe_latent = req_i64(t, "routed_expert_hidden_size")?;
    let n_shared_experts = req_i64(t, "num_shared_experts")?;
    let first_k_dense = req_i64(t, "first_k_dense_replace")? as u32;

    let attn_kind = kimi_k3_attn_kinds(t, layers)?;
    let is_full: Vec<bool> = attn_kind.iter().map(|&k| k == AttnKind::Full).collect();
    let hd = qk_nope_head_dim + qk_rope_head_dim;

    let mut s = HfSynthesis {
        name,
        arch: HfArch::KimiK3,
        hidden_size: hidden,
        num_attention_heads: heads,
        kvh_slide: heads, // MLA: every head attends its own latent slice.
        kvh_full: heads,
        hd_slide: hd,
        hd_full: hd,
        intermediate_size: inter,
        num_layers: layers as u32,
        vocab_size: vocab,
        is_full,
        attn_kind,
        k_eq_v: false,
        tied_embed: t["tie_word_embeddings"].as_bool().unwrap_or(false),
        // The multimodal wrapper nests the text tower: the checkpoint spells
        // every text weight `language_model.model.layers.{L}.…`, NOT
        // `model.layers.{L}.…`. Verified against the shard headers.
        prefix: "language_model.model.".to_string(),
        has_qk_norm: false,
        has_v_norm: false,
        moe: true,
        n_exp,
        top_k,
        moe_inter,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        v_head_dim,
        first_k_dense,
        n_shared_experts,
        moe_latent,
        qwen35: None,
        // `quantization_config` lives under text_config for kimi_k3, and its
        // `format` is "mxfp4-pack-quantized" (compressed-tensors), which
        // parse_weight_dtype's `quant_method` probe does not spell.
        weight_dtype: parse_weight_dtype(t),
        net: NetConfig {
            name: String::new(),
            hidden: 0,
            ops: vec![],
        },
    };
    s.net = build_net_config(&s);
    Ok(s)
}

/// Resolve `linear_attn_config.{full_attn_layers,kda_layers}` into one
/// per-layer [`AttnKind`] vector, 0-based.
///
/// Both lists are required and both are checked: the two must PARTITION
/// `0..num_hidden_layers` exactly — no gap, no overlap, nothing out of range.
/// Trusting one list and deriving the other by complement is the §4 bug shape:
/// a typo'd or truncated list would then silently reclassify layers, and a KDA
/// layer compiled as MLA binds tensors that do not exist (or, worse, ones that
/// do and mean something else).
fn kimi_k3_attn_kinds(t: &Value, layers: i64) -> Result<Vec<AttnKind>, String> {
    let lac = t
        .get("linear_attn_config")
        .filter(|c| c.is_object())
        .ok_or(
            "kimi_k3 text_config has no `linear_attn_config` — without it there is no \
             way to know which layers are MLA and which are KDA, and guessing the \
             stride would bind the wrong tensors on 69 of 93 layers",
        )?;
    let list = |k: &str| -> Result<Vec<i64>, String> {
        lac[k]
            .as_array()
            .ok_or_else(|| format!("linear_attn_config.{k} missing or not an array"))?
            .iter()
            .map(|x| {
                x.as_i64()
                    .ok_or_else(|| format!("linear_attn_config.{k} has a non-integer entry"))
            })
            .collect()
    };
    let full = list("full_attn_layers")?;
    let kda = list("kda_layers")?;

    let mut out: Vec<Option<AttnKind>> = vec![None; layers as usize];
    for (src, kind) in [(&full, AttnKind::Full), (&kda, AttnKind::Kda)] {
        for &one_based in src {
            // 1-BASED in config (configuration_kimi_k3.py: `(layer_idx + 1) in
            // kda_layers`). Convert once, here.
            let l = one_based - 1;
            if !(0..layers).contains(&l) {
                return Err(format!(
                    "linear_attn_config lists layer {one_based} (1-based, = {l} 0-based) \
                     but num_hidden_layers is {layers}"
                ));
            }
            if let Some(prev) = out[l as usize] {
                return Err(format!(
                    "linear_attn_config assigns 0-based layer {l} twice ({prev:?} and \
                     {kind:?}); full_attn_layers and kda_layers must be disjoint"
                ));
            }
            out[l as usize] = Some(kind);
        }
    }
    let missing: Vec<usize> = out
        .iter()
        .enumerate()
        .filter(|(_, k)| k.is_none())
        .map(|(i, _)| i)
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "linear_attn_config covers {} of {layers} layers; 0-based layers {:?}… are in \
             neither full_attn_layers nor kda_layers",
            layers as usize - missing.len(),
            &missing[..missing.len().min(8)]
        ));
    }
    Ok(out.into_iter().map(|k| k.unwrap()).collect())
}

/// Build a full-model [`LayerPlan`] with all N layers unrolled, using
/// safetensor-matching weight names. This is the Option A "unroll at emit time"
/// path.
///
/// Per dense block (Gemma; Llama/Qwen drop the ops their arch lacks):
/// input norm → Q/K(/V) projections → q/k norm (+ weightless v norm) → flash →
/// O projection → post-attention norm → attention residual → pre-FF norm →
/// gate/up → activation (GeGLU/SwiGLU, consumes both) → down → post-FF norm →
/// MLP residual → `layer_scalar` scale. MoE layers additionally emit the
/// router and the fused expert GEMMs (see below).
///
/// Plus: embed lookup (first), final norm + lm_head (last, tied or not).
pub fn build_full_model_plan(bucket: &ShapeBucket, synth: &HfSynthesis) -> LayerPlan {
    assert!(
        synth.qwen35.is_none(),
        "qwen3_5 has no faithful LayerPlan; call ensure_layer_plan_supported before planning"
    );
    // Kimi-K3 config is ACCEPTED (see synth_kimi_k3) but not emittable: the
    // block below models a uniform softmax-attention layer with a hidden-space
    // MoE, and K3 is neither on 69 of its 93 layers. Emitting it anyway would
    // produce a plan whose GEMM shapes and weight names are wrong per layer —
    // exactly the "loads and is fluent but wrong" failure this module exists to
    // prevent. `devgen::mla::kimi_k3_emit` holds the full capability list.
    assert_ne!(
        synth.arch,
        HfArch::KimiK3,
        "kimi_k3 has no full-model plan: 69/93 layers are KDA (linear attention) and \
         the MoE is latent + mxfp4. Run `plowc --hf-dir <k3> --emit devblob` for the \
         itemised missing-capability report (devgen::mla::kimi_k3_emit)."
    );
    let rows = bucket.rows();
    let seq = bucket.attn_seq();
    let h = synth.hidden_size;
    let heads = synth.num_attention_heads;
    let inter = synth.intermediate_size;
    let prefix = &synth.prefix;
    // Weight dtype for GEMM projections: FP8 halves weight bandwidth.
    // Norms/embed/activations always stay BF16.
    let wdt = synth.weight_dtype;

    let mut ops: Vec<OpSpec> = Vec::with_capacity(8 + synth.num_layers as usize * 20);

    // Helper macro: push an op with explicit dtype.
    macro_rules! emit {
        ($name:expr, $inputs:expr, $kind:expr) => {{
            let idx = ops.len();
            let output = format!("t{idx}");
            ops.push(OpSpec {
                name: $name.to_string(),
                inputs: $inputs,
                output: output.clone(),
                kind: $kind,
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            });
            output
        }};
    }
    // GEMM with the model's weight dtype (FP8/BF16/MX).
    macro_rules! gemm {
        ($name:expr, $inputs:expr, $shape:expr) => {{
            let idx = ops.len();
            let output = format!("t{idx}");
            ops.push(OpSpec {
                name: $name.to_string(),
                inputs: $inputs,
                output: output.clone(),
                kind: OpKind::Gemm($shape),
                weight_dtype: wdt,
                compute_dtype: nn_graph::DType::BF16,
            });
            output
        }};
    }
    // Row-op shorthands. `norm` = weighted RMSNorm, `ew2` = 2-operand
    // elementwise (residual add / GeGLU combine), `scale1` = multiply by a
    // small per-layer tensor (layer_scalar / per-expert scale).
    macro_rules! norm {
        ($name:expr, $x:expr, $w:expr, $feat:expr) => {
            emit!(
                $name,
                vec![$x, $w],
                OpKind::Row(RowShape {
                    rows,
                    feat: $feat,
                    operands: 2,
                    reduce: true
                })
            )
        };
    }
    macro_rules! ew2 {
        ($name:expr, $a:expr, $b:expr, $feat:expr) => {
            emit!(
                $name,
                vec![$a, $b],
                OpKind::Row(RowShape {
                    rows,
                    feat: $feat,
                    operands: 2,
                    reduce: false
                })
            )
        };
    }

    // --- Embed lookup (modeled as a Row op — the scheduler doesn't have a
    // dedicated embed tile, but the memory/weight accounting needs the tensor) ---
    let embed_name = format!("{prefix}embed_tokens.weight");
    let mut prev = emit!(
        "embed",
        vec!["ids".into(), embed_name.clone()],
        OpKind::Row(RowShape {
            rows,
            feat: h,
            operands: 2,
            reduce: false
        })
    );

    // --- N transformer blocks, each with its OWN geometry ---
    for l in 0..synth.num_layers {
        let full = synth.is_full[l as usize];
        let hd = if full { synth.hd_full } else { synth.hd_slide };
        let kvh = if full {
            synth.kvh_full
        } else {
            synth.kvh_slide
        };
        let qw = heads * hd;
        let kw = kvh * hd;
        // Gemma full layers have NO v_proj: V is the raw k_proj output.
        let keqv = full && synth.k_eq_v;
        let lp = format!("{prefix}layers.{l}");
        let layer_in = prev.clone();

        // ---------- MLA + MoE block (Kimi K2 / DeepSeek V2/V3 / GLM 5.2) ----------
        if matches!(synth.arch, HfArch::Kimi | HfArch::DeepSeek | HfArch::Glm) {
            let qlr = synth.q_lora_rank;
            let kvlr = synth.kv_lora_rank;
            let rope_hd = synth.qk_rope_head_dim;
            let nope_hd = synth.qk_nope_head_dim;
            let vhd = synth.v_head_dim;
            let full_qk_hd = nope_hd + rope_hd;

            // 1. Input layernorm
            prev = norm!(
                format!("norm_in_L{l}"),
                prev.clone(),
                format!("{lp}.input_layernorm.weight"),
                h
            );
            let norm_out = prev.clone();

            // 2. MLA Q path: q_a_proj (down) → q_a_layernorm → q_b_proj (up)
            let qa = gemm!(
                format!("q_a_proj_L{l}"),
                vec![norm_out.clone(), format!("{lp}.self_attn.q_a_proj.weight")],
                GemmShape {
                    m: rows,
                    n: qlr,
                    k: h
                }
            );
            let qa_n = norm!(
                format!("q_a_norm_L{l}"),
                qa,
                format!("{lp}.self_attn.q_a_layernorm.weight"),
                qlr
            );
            let q_out = gemm!(
                format!("q_b_proj_L{l}"),
                vec![qa_n, format!("{lp}.self_attn.q_b_proj.weight")],
                GemmShape {
                    m: rows,
                    n: heads * full_qk_hd,
                    k: qlr
                }
            );

            // 3. MLA KV path: kv_a_proj_with_mqa (down) → kv_a_layernorm → kv_b_proj (up)
            //    kv_a output = kv_lora_rank + qk_rope_head_dim (rope part is passed through)
            let kva_out = kvlr + rope_hd;
            let kva = gemm!(
                format!("kv_a_proj_L{l}"),
                vec![
                    norm_out,
                    format!("{lp}.self_attn.kv_a_proj_with_mqa.weight")
                ],
                GemmShape {
                    m: rows,
                    n: kva_out,
                    k: h
                }
            );
            let kva_n = norm!(
                format!("kv_a_norm_L{l}"),
                kva,
                format!("{lp}.self_attn.kv_a_layernorm.weight"),
                kvlr
            );
            // kv_b_proj expands to heads*(nope_hd + v_head_dim)
            let kvb_out = heads * (nope_hd + vhd);
            let kv_out = gemm!(
                format!("kv_b_proj_L{l}"),
                vec![kva_n, format!("{lp}.self_attn.kv_b_proj.weight")],
                GemmShape {
                    m: rows,
                    n: kvb_out,
                    k: kvlr
                }
            );

            // 4. Flash attention (MLA absorbed: head_dim = full Q/K head width)
            prev = emit!(
                format!("flash_L{l}"),
                vec![q_out, kv_out.clone(), kv_out],
                OpKind::Flash(AttnShape {
                    heads,
                    kv_heads: heads,
                    seq_q: seq,
                    seq_kv: seq,
                    head_dim: full_qk_hd,
                    causal: true,
                    sliding_window: 0,
                })
            );

            // 5. O projection: heads*v_head_dim → H
            prev = gemm!(
                format!("o_proj_L{l}"),
                vec![prev.clone(), format!("{lp}.self_attn.o_proj.weight")],
                GemmShape {
                    m: rows,
                    n: h,
                    k: heads * vhd
                }
            );

            // 6. Residual add (attention output + layer input)
            prev = ew2!(
                format!("resid_attn_L{l}"),
                prev.clone(),
                layer_in.clone(),
                h
            );

            // 7. Post-attention layernorm
            prev = norm!(
                format!("norm_pa_L{l}"),
                prev.clone(),
                format!("{lp}.post_attention_layernorm.weight"),
                h
            );
            let ff_in = prev.clone();

            // 8. MLP — dense for first_k_dense layers, MoE for the rest
            let is_dense = l < synth.first_k_dense;
            if is_dense {
                // Standard SwiGLU MLP
                let gate_out = gemm!(
                    format!("gate_L{l}"),
                    vec![ff_in.clone(), format!("{lp}.mlp.gate_proj.weight")],
                    GemmShape {
                        m: rows,
                        n: inter,
                        k: h
                    }
                );
                let up_out = gemm!(
                    format!("up_L{l}"),
                    vec![ff_in, format!("{lp}.mlp.up_proj.weight")],
                    GemmShape {
                        m: rows,
                        n: inter,
                        k: h
                    }
                );
                prev = ew2!(format!("act_L{l}"), gate_out, up_out, inter);
                prev = gemm!(
                    format!("down_L{l}"),
                    vec![prev.clone(), format!("{lp}.mlp.down_proj.weight")],
                    GemmShape {
                        m: rows,
                        n: h,
                        k: inter
                    }
                );
            } else {
                // MoE: router + shared experts + routed experts
                let (e, tk, mi) = (synth.n_exp, synth.top_k, synth.moe_inter);
                let m_routed = ((rows * tk + e - 1) / e).max(1);

                // Router GEMM: hidden → n_exp scores
                let scores = gemm!(
                    format!("router_L{l}"),
                    vec![ff_in.clone(), format!("{lp}.mlp.gate.weight")],
                    GemmShape {
                        m: rows,
                        n: e,
                        k: h
                    }
                );
                // Router top-k selection (row-wise reduction)
                let _gates = emit!(
                    format!("router_topk_L{l}"),
                    vec![scores, ff_in.clone()],
                    OpKind::Row(RowShape {
                        rows,
                        feat: e,
                        operands: 2,
                        reduce: true
                    })
                );

                // Shared expert(s): standard SwiGLU at moe_intermediate_size
                let sh_inter = synth.n_shared_experts * mi;
                let sh_gate = gemm!(
                    format!("sh_gate_L{l}"),
                    vec![
                        ff_in.clone(),
                        format!("{lp}.mlp.shared_experts.gate_proj.weight")
                    ],
                    GemmShape {
                        m: rows,
                        n: sh_inter,
                        k: h
                    }
                );
                let sh_up = gemm!(
                    format!("sh_up_L{l}"),
                    vec![
                        ff_in.clone(),
                        format!("{lp}.mlp.shared_experts.up_proj.weight")
                    ],
                    GemmShape {
                        m: rows,
                        n: sh_inter,
                        k: h
                    }
                );
                let sh_act = ew2!(format!("sh_act_L{l}"), sh_gate, sh_up, sh_inter);
                let sh_down = gemm!(
                    format!("sh_down_L{l}"),
                    vec![sh_act, format!("{lp}.mlp.shared_experts.down_proj.weight")],
                    GemmShape {
                        m: rows,
                        n: h,
                        k: sh_inter
                    }
                );

                // Routed experts: fused grouped GEMMs (same pattern as Gemma MoE).
                // Weight bytes: gate_up = [E, 2*mi, H], down = [E, H, mi].
                let gu = gemm!(
                    format!("moe_gate_up_L{l}"),
                    vec![ff_in.clone(), format!("{lp}.mlp.experts.gate_up_proj")],
                    GemmShape {
                        m: m_routed,
                        n: e * 2 * mi,
                        k: h
                    }
                );
                let fu = emit!(
                    format!("moe_act_L{l}"),
                    vec![gu, sh_down.clone()],
                    OpKind::Row(RowShape {
                        rows,
                        feat: tk * mi,
                        operands: 2,
                        reduce: false
                    })
                );
                let md = gemm!(
                    format!("moe_down_L{l}"),
                    vec![fu, format!("{lp}.mlp.experts.down_proj")],
                    GemmShape {
                        m: m_routed,
                        n: e * h,
                        k: mi
                    }
                );

                // Combine: shared + routed experts
                prev = ew2!(format!("moe_comb_L{l}"), sh_down, md, h);
            }

            // 9. MLP residual
            prev = ew2!(format!("resid_mlp_L{l}"), prev.clone(), layer_in, h);
            continue;
        }

        // 1. Input layernorm
        prev = norm!(
            format!("norm_in_L{l}"),
            prev.clone(),
            format!("{lp}.input_layernorm.weight"),
            h
        );
        let norm_out = prev.clone();

        // 2. Q/K(/V) projections — all read the norm output.
        let mut q_out = gemm!(
            format!("q_proj_L{l}"),
            vec![norm_out.clone(), format!("{lp}.self_attn.q_proj.weight")],
            GemmShape {
                m: rows,
                n: qw,
                k: h
            }
        );
        let mut k_out = gemm!(
            format!("k_proj_L{l}"),
            vec![norm_out.clone(), format!("{lp}.self_attn.k_proj.weight")],
            GemmShape {
                m: rows,
                n: kw,
                k: h
            }
        );
        let v_src = if keqv {
            k_out.clone()
        } else {
            gemm!(
                format!("v_proj_L{l}"),
                vec![norm_out.clone(), format!("{lp}.self_attn.v_proj.weight")],
                GemmShape {
                    m: rows,
                    n: kw,
                    k: h
                }
            )
        };

        // 3. Per-head q/k RMSNorms (weighted, width = THIS layer's head_dim)
        //    and Gemma's weightless v_norm.
        if synth.has_qk_norm {
            q_out = norm!(
                format!("q_norm_L{l}"),
                q_out,
                format!("{lp}.self_attn.q_norm.weight"),
                qw
            );
            k_out = norm!(
                format!("k_norm_L{l}"),
                k_out,
                format!("{lp}.self_attn.k_norm.weight"),
                kw
            );
        }
        let v_out = if synth.has_v_norm {
            emit!(
                format!("v_norm_L{l}"),
                vec![v_src],
                OpKind::Row(RowShape {
                    rows,
                    feat: kw,
                    operands: 1,
                    reduce: true
                })
            )
        } else {
            v_src
        };

        // 4. Flash attention over this layer's head geometry.
        prev = emit!(
            format!("flash_L{l}"),
            vec![q_out, k_out, v_out],
            OpKind::Flash(AttnShape {
                heads,
                kv_heads: kvh,
                seq_q: seq,
                seq_kv: seq,
                head_dim: hd,
                causal: true,
                // The legacy HfDir synthesis does not retain the window width.
                // Keep zero rather than inventing a value; Source::Model carries
                // the exact config through nn_graph.
                sliding_window: 0,
            })
        );

        // 5. O projection (input width = this layer's q width).
        prev = gemm!(
            format!("o_proj_L{l}"),
            vec![prev.clone(), format!("{lp}.self_attn.o_proj.weight")],
            GemmShape {
                m: rows,
                n: h,
                k: qw
            }
        );

        // 6. Sandwich: post-attention norm, THEN the residual add.
        prev = norm!(
            format!("norm_pa_L{l}"),
            prev.clone(),
            format!("{lp}.post_attention_layernorm.weight"),
            h
        );
        prev = ew2!(format!("resid_attn_L{l}"), prev.clone(), layer_in, h);
        let h1 = prev.clone();

        // 7. Dense MLP: pre-FF norm → gate/up → GeGLU/SwiGLU → down → post-FF norm.
        prev = norm!(
            format!("norm_pf_L{l}"),
            prev.clone(),
            format!("{lp}.pre_feedforward_layernorm.weight"),
            h
        );
        let mlp_in = prev.clone();
        let gate_out = gemm!(
            format!("gate_L{l}"),
            vec![mlp_in.clone(), format!("{lp}.mlp.gate_proj.weight")],
            GemmShape {
                m: rows,
                n: inter,
                k: h
            }
        );
        let up_out = gemm!(
            format!("up_L{l}"),
            vec![mlp_in, format!("{lp}.mlp.up_proj.weight")],
            GemmShape {
                m: rows,
                n: inter,
                k: h
            }
        );
        prev = ew2!(format!("act_L{l}"), gate_out, up_out, inter);
        prev = gemm!(
            format!("down_L{l}"),
            vec![prev.clone(), format!("{lp}.mlp.down_proj.weight")],
            GemmShape {
                m: rows,
                n: h,
                k: inter
            }
        );
        prev = norm!(
            format!("norm_pff_L{l}"),
            prev.clone(),
            format!("{lp}.post_feedforward_layernorm.weight"),
            h
        );

        // 8. MoE branch (26B-A4B hybrid): router + top-k experts from h1, summed
        //    with the dense branch. The fused expert tensors are modeled as ONE
        //    grouped GEMM each with n = E×(out width): the weight BYTES
        //    (n·k = E·2I·H and E·H·I) are exact, and m = ceil(rows·top_k / E)
        //    makes m·n·k ≈ rows·top_k·(routed MACs) — the true routed compute.
        if synth.moe {
            let (e, tk, mi) = (synth.n_exp, synth.top_k, synth.moe_inter);
            let m_routed = ((rows * tk + e - 1) / e).max(1);
            // Extra sandwich norm on the dense branch output (gemma4's g_pf1).
            prev = norm!(
                format!("norm_pff1_L{l}"),
                prev.clone(),
                format!("{lp}.post_feedforward_layernorm_1.weight"),
                h
            );
            let x2 = norm!(
                format!("norm_pf2_L{l}"),
                h1.clone(),
                format!("{lp}.pre_feedforward_layernorm_2.weight"),
                h
            );
            let scores = gemm!(
                format!("router_L{l}"),
                vec![x2.clone(), format!("{lp}.router.proj.weight")],
                GemmShape {
                    m: rows,
                    n: e,
                    k: h
                }
            );
            // Router input scale [H] and per-expert score scale [E]: tiny
            // learned tensors folded into elementwise ops so the loader binds
            // them and the plan covers them.
            let scores = ew2!(
                format!("router_scale_L{l}"),
                scores,
                format!("{lp}.router.scale"),
                e
            );
            let gates = ew2!(
                format!("router_pes_L{l}"),
                scores,
                format!("{lp}.router.per_expert_scale"),
                e
            );
            let gu = gemm!(
                format!("moe_gate_up_L{l}"),
                vec![x2, format!("{lp}.experts.gate_up_proj")],
                GemmShape {
                    m: m_routed,
                    n: e * 2 * mi,
                    k: h
                }
            );
            let fu = emit!(
                format!("moe_act_L{l}"),
                vec![gu, gates],
                OpKind::Row(RowShape {
                    rows,
                    feat: tk * mi,
                    operands: 2,
                    reduce: false
                })
            );
            let md = gemm!(
                format!("moe_down_L{l}"),
                vec![fu, format!("{lp}.experts.down_proj")],
                GemmShape {
                    m: m_routed,
                    n: e * h,
                    k: mi
                }
            );
            let mn = norm!(
                format!("norm_pff2_L{l}"),
                md,
                format!("{lp}.post_feedforward_layernorm_2.weight"),
                h
            );
            // h1 + dense + moe sandwich sum.
            prev = ew2!(format!("moe_comb_L{l}"), prev.clone(), mn, h);
        }

        // 9. MLP residual, then the learned per-layer scale on the whole
        //    hidden state (`layer_scalar`, a real [1] checkpoint tensor).
        prev = ew2!(format!("resid_mlp_L{l}"), prev.clone(), h1, h);
        prev = ew2!(
            format!("scale_L{l}"),
            prev.clone(),
            format!("{lp}.layer_scalar"),
            h
        );
    }

    // --- Final norm ---
    prev = norm!(
        "final_norm",
        prev.clone(),
        format!("{prefix}norm.weight"),
        h
    );

    // --- LM head (logits projection): tied models reuse embed_tokens. ---
    let lm_head_w = if synth.tied_embed {
        embed_name
    } else {
        "lm_head.weight".into()
    };
    gemm!(
        "lm_head",
        vec![prev, lm_head_w],
        GemmShape {
            m: rows,
            n: synth.vocab_size,
            k: h
        }
    );

    LayerPlan { ops }
}

/// Validate a full-model plan against the safetensors actually on disk.
///
/// Both directions, both HARD errors — this is the same philosophy as the
/// `gemma4` binary ("a silently-absent weight is the worst failure mode in
/// this whole stack"):
///
/// 1. Every weight the plan references must exist in the checkpoint, and every
///    GEMM weight's `n·k` must equal the tensor's element count.
/// 2. Every checkpoint tensor under the model prefix must be referenced by the
///    plan — otherwise the compile silently dropped part of the model.
pub fn validate_against_checkpoint(
    dir: &Path,
    plan: &LayerPlan,
    synth: &HfSynthesis,
) -> Result<(), String> {
    let ckpt = safetensor_shapes(dir)?;
    // Weight inputs = plan inputs that are not produced by another op and are
    // not the graph input.
    let produced: BTreeSet<&str> = plan.ops.iter().map(|o| o.output.as_str()).collect();
    let mut referenced: BTreeSet<&str> = BTreeSet::new();
    let mut errs: Vec<String> = Vec::new();
    for op in &plan.ops {
        for (i, inp) in op.inputs.iter().enumerate() {
            if inp == "ids" || produced.contains(inp.as_str()) {
                continue;
            }
            referenced.insert(inp);
            match ckpt.get(inp.as_str()) {
                None => errs.push(format!("op {}: weight {inp:?} not in checkpoint", op.name)),
                Some(shape) => {
                    // GEMM convention: inputs[1] is the [n,k] weight.
                    if i == 1 {
                        if let OpKind::Gemm(g) = &op.kind {
                            let numel: i64 = shape.iter().product();
                            if g.n * g.k != numel {
                                errs.push(format!(
                                    "op {}: weight {inp:?} shape {shape:?} has {numel} \
                                     elements but the plan GEMM is n={} k={}",
                                    op.name, g.n, g.k
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    // Reverse: everything the checkpoint ships under the prefix must be used.
    let uncovered: Vec<&String> = ckpt
        .keys()
        .filter(|k| k.starts_with(&synth.prefix) && !referenced.contains(k.as_str()))
        .collect();
    if !uncovered.is_empty() {
        errs.push(format!(
            "checkpoint tensors NOT covered by the plan ({}): {}",
            uncovered.len(),
            uncovered
                .iter()
                .take(8)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "--hf-dir checkpoint validation failed ({} problems):\n  {}",
            errs.len(),
            errs.join("\n  ")
        ))
    }
}

/// Read the tensor name → shape map from every `*.safetensors` file directly
/// in `dir` (like `bin/gemma4.rs::shard_files`, we enumerate what is actually
/// there rather than trusting an index file).
fn safetensor_shapes(dir: &Path) -> Result<std::collections::HashMap<String, Vec<i64>>, String> {
    let mut out = std::collections::HashMap::new();
    let rd = std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    let mut files: Vec<_> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|x| x.to_str()) == Some("safetensors")
                || p.to_string_lossy().ends_with(".partial.safetensors")
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!(
            "no *.safetensors files in {} — cannot validate the plan against the \
             checkpoint (and the runtime could not bind weights either)",
            dir.display()
        ));
    }
    for p in files {
        let mut f = std::fs::File::open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        use std::io::Read;
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8)
            .map_err(|e| format!("{}: header length: {e}", p.display()))?;
        let hlen = u64::from_le_bytes(len8);
        if hlen > 256 * 1024 * 1024 {
            return Err(format!("{}: implausible header length {hlen}", p.display()));
        }
        let mut hbuf = vec![0u8; hlen as usize];
        f.read_exact(&mut hbuf)
            .map_err(|e| format!("{}: header: {e}", p.display()))?;
        let hdr: Value = serde_json::from_slice(&hbuf)
            .map_err(|e| format!("{}: header json: {e}", p.display()))?;
        let obj = hdr
            .as_object()
            .ok_or_else(|| format!("{}: header is not an object", p.display()))?;
        for (k, v) in obj {
            if k == "__metadata__" {
                continue;
            }
            let shape: Vec<i64> = v["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
                .unwrap_or_default();
            out.insert(k.clone(), shape);
        }
    }
    Ok(out)
}

/// Build the single-block NetConfig from the resolved model (backward compat:
/// this is what `--hf-dir` compiled before the Option A unroll; kept for
/// `NetConfig::validate` and the synthesis tests). Uses the SLIDING-layer
/// geometry — the majority block.
fn build_net_config(s: &HfSynthesis) -> NetConfig {
    let q_width = s.num_attention_heads * s.hd_slide;
    let kv_width = s.kvh_slide * s.hd_slide;
    // Fused QKV width: Q + K + V (K and V distinct on sliding layers).
    let qkv_n = q_width + 2 * kv_width;
    let ops = vec![
        NetOp::Norm {
            feat: Some(s.hidden_size),
        },
        NetOp::Gemm {
            n: qkv_n,
            k: Some(s.hidden_size),
        },
        NetOp::Flash {
            heads: s.num_attention_heads,
            head_dim: s.hd_slide,
        },
        NetOp::Gemm {
            n: s.hidden_size,
            k: Some(q_width),
        },
        NetOp::Norm {
            feat: Some(s.hidden_size),
        },
        NetOp::Gemm {
            n: 2 * s.intermediate_size,
            k: Some(s.hidden_size),
        },
        NetOp::Act,
        NetOp::Gemm {
            n: s.hidden_size,
            k: Some(s.intermediate_size),
        },
    ];
    NetConfig {
        name: s.name.clone(),
        hidden: s.hidden_size,
        ops,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A faithful miniature of the gemma-4-12B nested config (all required
    /// fields, 6-layer 5:1 sliding:full pattern).
    fn gemma_json(extra: &str) -> String {
        format!(
            r#"{{
            "model_type": "gemma4_unified",
            "text_config": {{
                "hidden_size": 3840,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "intermediate_size": 15360,
                "attention_k_eq_v": true,
                "tie_word_embeddings": true,
                "num_hidden_layers": 6,
                "vocab_size": 262144,
                "layer_types": ["sliding_attention","sliding_attention","sliding_attention",
                                "sliding_attention","sliding_attention","full_attention"]
                {extra}
            }}
        }}"#
        )
    }

    #[test]
    fn test_gemma4_12b_synthesis() {
        let s = synthesize_full(&gemma_json(""), "gemma-4-12b-it".into()).unwrap();
        assert_eq!(s.arch, HfArch::Gemma4);
        assert_eq!(s.prefix, "model.language_model.");
        assert_eq!(s.num_layers, 6);
        assert_eq!(s.is_full, vec![false, false, false, false, false, true]);
        assert!(s.tied_embed && s.k_eq_v && s.has_qk_norm && s.has_v_norm);
        assert_eq!((s.hd_slide, s.hd_full), (256, 512));
        assert_eq!((s.kvh_slide, s.kvh_full), (8, 1));
        let nc = &s.net;
        assert_eq!(nc.name, "gemma-4-12b-it");
        assert_eq!(nc.hidden, 3840);
        assert_eq!(nc.ops.len(), 8);
        // Fused QKV on a sliding layer: Q=16*256=4096, K=V=8*256=2048 → 8192.
        match &nc.ops[1] {
            NetOp::Gemm { n, k } => {
                assert_eq!(*n, 8192);
                assert_eq!(*k, Some(3840));
            }
            _ => panic!("expected Gemm"),
        }
    }

    fn qwen35_json() -> &'static str {
        r#"{
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "dtype": "bfloat16",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 4,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attn_output_gate": true,
            "hidden_act": "silu",
            "mamba_ssm_dtype": "float32",
            "output_gate_type": "swish",
            "layer_types": ["linear_attention", "linear_attention",
                            "linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "rope_parameters": {
              "partial_rotary_factor": 0.25,
              "rope_theta": 10000000,
              "rope_type": "default",
              "mrope_interleaved": true
            }
          },
          "vision_config": {"model_type": "qwen3_5"}
        }"#
    }

    #[test]
    fn qwen35_hybrid_geometry_is_parsed_without_dense_fallback() {
        let s = synthesize_full(qwen35_json(), "qwen3.8-27b".into()).unwrap();
        assert_eq!(s.arch, HfArch::Qwen35);
        assert_eq!(s.prefix, "model.language_model.");
        assert_eq!(
            s.attn_kind,
            vec![
                AttnKind::GatedDeltaNet,
                AttnKind::GatedDeltaNet,
                AttnKind::GatedDeltaNet,
                AttnKind::Full
            ]
        );
        assert_eq!(s.is_full, vec![false, false, false, true]);
        assert_eq!((s.num_attention_heads, s.kvh_full, s.hd_full), (24, 4, 256));

        let q = s.qwen35.unwrap();
        assert_eq!(q.qkv_width(), 10_240);
        assert_eq!(q.value_width(), 6_144);
        assert_eq!(q.linear_conv_kernel_dim, 4);
        assert_eq!(q.partial_rotary_factor, 0.25);
        assert_eq!(q.rope_theta, 10_000_000.0);
        assert!(q.attn_output_gate);
        assert_eq!(s.num_attention_heads * s.hd_full * 2, 12_288);

        let err = ensure_layer_plan_supported(&s).unwrap_err();
        assert!(err.contains("qkv+conv width 10240"), "{err}");
        assert!(err.contains("query+gate width 12288"), "{err}");
        assert!(err.contains("refused"), "{err}");
    }

    #[test]
    fn qwen35_fp8_outer_quantization_is_preserved() {
        let json = qwen35_json().replace(
            r#""vision_config""#,
            r#""quantization_config": {
              "activation_scheme": "dynamic",
              "fmt": "e4m3",
              "quant_method": "fp8",
              "modules_to_not_convert": ["lm_head"],
              "weight_block_size": [128, 128]
            },
            "vision_config""#,
        );
        let s = synthesize_full(&json, "qwen38-27b-fp8".into()).unwrap();
        assert_eq!(s.weight_dtype, nn_graph::DType::F8E4M3);
        let q = s.qwen35.unwrap();
        assert_eq!(q.fp8_weight_block_size, Some([128, 128]));
        assert!(q.fp8_dynamic_activations);
    }

    #[test]
    fn qwen35_unknown_fp8_geometry_is_rejected() {
        let json = qwen35_json().replace(
            r#""vision_config""#,
            r#""quantization_config": {
              "activation_scheme": "dynamic",
              "fmt": "e4m3",
              "quant_method": "fp8",
              "modules_to_not_convert": ["lm_head"],
              "weight_block_size": [64, 128]
            },
            "vision_config""#,
        );
        let err = synthesize_full(&json, "qwen38-27b-fp8".into()).unwrap_err();
        assert!(err.contains("weight_block_size=[128,128]"), "{err}");
    }

    #[test]
    fn qwen35_layer_types_are_required_and_checked() {
        let short = qwen35_json().replace(
            "\"linear_attention\", \"linear_attention\",\n                            \"linear_attention\", \"full_attention\"",
            "\"linear_attention\", \"full_attention\"",
        );
        let err = synthesize_full(&short, "qwen".into()).unwrap_err();
        assert!(
            err.contains("2 entries") && err.contains("num_hidden_layers is 4"),
            "{err}"
        );

        let unknown = qwen35_json().replacen("\"linear_attention\"", "\"sliding_attention\"", 1);
        let err = synthesize_full(&unknown, "qwen".into()).unwrap_err();
        assert!(
            err.contains("layer_types[0]") && err.contains("sliding_attention"),
            "{err}"
        );
    }

    #[test]
    fn qwen35_execution_semantics_are_checked() {
        for (from, to, expected) in [
            (
                "\"attention_bias\": false",
                "\"attention_bias\": true",
                "attention_bias=false",
            ),
            (
                "\"attn_output_gate\": true",
                "\"attn_output_gate\": false",
                "attn_output_gate=true",
            ),
            (
                "\"hidden_act\": \"silu\"",
                "\"hidden_act\": \"gelu\"",
                "hidden_act",
            ),
            (
                "\"mamba_ssm_dtype\": \"float32\"",
                "\"mamba_ssm_dtype\": \"bfloat16\"",
                "mamba_ssm_dtype",
            ),
            (
                "\"output_gate_type\": \"swish\"",
                "\"output_gate_type\": \"sigmoid\"",
                "output_gate_type",
            ),
            (
                "\"rope_type\": \"default\"",
                "\"rope_type\": \"linear\"",
                "rope_type=default",
            ),
            (
                "\"mrope_interleaved\": true",
                "\"mrope_interleaved\": false",
                "mrope_interleaved=true",
            ),
        ] {
            let json = qwen35_json().replacen(from, to, 1);
            let err = synthesize_full(&json, "qwen".into()).unwrap_err();
            assert!(err.contains(expected), "{err}");
        }
    }

    #[test]
    fn test_missing_required_field_errors() {
        // Drop vocab_size → hard error, not a 32000 default.
        let json = gemma_json("").replace("\"vocab_size\": 262144,", "");
        let err = synthesize_full(&json, "x".into()).unwrap_err();
        assert!(err.contains("vocab_size"), "{err}");
        // Unknown architecture → hard error.
        let err =
            synthesize_full(r#"{"model_type":"phi3","hidden_size":1}"#, "x".into()).unwrap_err();
        assert!(err.contains("unsupported model_type"), "{err}");
    }

    #[test]
    fn test_flat_llama_style() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "vocab_size": 128256
        }"#;
        let s = synthesize_full(json, "llama-3-8b".into()).unwrap();
        assert_eq!(s.arch, HfArch::Llama);
        assert_eq!(s.prefix, "model.");
        // Llama: untied by default, no qk norm, no k_eq_v, all-full layers.
        assert!(!s.tied_embed && !s.has_qk_norm && !s.has_v_norm && !s.k_eq_v);
        assert!(s.is_full.iter().all(|&f| f));
        // Untied → a real top-level lm_head.weight in the plan.
        let bucket = ShapeBucket {
            batch: 1,
            seq: 512,
            phase: schedule::Phase::Decode,
        };
        let plan = build_full_model_plan(&bucket, &s);
        let lm = plan.ops.last().unwrap();
        assert_eq!(lm.inputs[1], "lm_head.weight");
    }

    #[test]
    fn test_full_model_gemma_geometry() {
        let s = synthesize_full(&gemma_json(""), "gemma-4-12b-it".into()).unwrap();
        let bucket = ShapeBucket {
            batch: 1,
            seq: 512,
            phase: schedule::Phase::Prefill,
        };
        let plan = build_full_model_plan(&bucket, &s);
        let get = |name: &str| {
            plan.ops
                .iter()
                .find(|o| o.name == name)
                .unwrap_or_else(|| panic!("missing op {name}"))
        };
        // Sliding layer 0: q 16×256=4096, k 8×256=2048, real v_proj.
        match &get("q_proj_L0").kind {
            OpKind::Gemm(g) => assert_eq!((g.n, g.k), (4096, 3840)),
            _ => panic!(),
        }
        match &get("k_proj_L0").kind {
            OpKind::Gemm(g) => assert_eq!(g.n, 2048),
            _ => panic!(),
        }
        assert!(plan.ops.iter().any(|o| o.name == "v_proj_L0"));
        // Full layer 5: q 16×512=8192, k 1×512=512, NO v_proj (k_eq_v).
        match &get("q_proj_L5").kind {
            OpKind::Gemm(g) => assert_eq!(g.n, 8192),
            _ => panic!(),
        }
        match &get("k_proj_L5").kind {
            OpKind::Gemm(g) => assert_eq!(g.n, 512),
            _ => panic!(),
        }
        assert!(!plan.ops.iter().any(|o| o.name == "v_proj_L5"));
        match &get("flash_L5").kind {
            OpKind::Flash(a) => assert_eq!(a.head_dim, 512),
            _ => panic!(),
        }
        // Every per-layer checkpoint tensor family is referenced on layer 0.
        let lp = "model.language_model.layers.0.";
        for t in [
            "input_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "layer_scalar",
        ] {
            let full = format!("{lp}{t}");
            assert!(
                plan.ops.iter().any(|o| o.inputs.contains(&full)),
                "plan does not reference {full}"
            );
        }
        // Tied lm_head reuses embed_tokens; GeGLU act consumes gate AND up.
        assert_eq!(
            plan.ops.last().unwrap().inputs[1],
            "model.language_model.embed_tokens.weight"
        );
        let act = get("act_L0");
        assert_eq!(act.inputs.len(), 2);
    }

    #[test]
    fn test_moe_block_coverage() {
        let s = synthesize_full(
            &gemma_json(
                r#", "enable_moe_block": true, "num_experts": 128,
                    "top_k_experts": 8, "moe_intermediate_size": 704"#,
            ),
            "gemma-4-26b-a4b-it".into(),
        )
        .unwrap();
        assert!(s.moe);
        let bucket = ShapeBucket {
            batch: 1,
            seq: 512,
            phase: schedule::Phase::Prefill,
        };
        let plan = build_full_model_plan(&bucket, &s);
        let lp = "model.language_model.layers.0.";
        for t in [
            "router.proj.weight",
            "router.scale",
            "router.per_expert_scale",
            "experts.gate_up_proj",
            "experts.down_proj",
            "pre_feedforward_layernorm_2.weight",
            "post_feedforward_layernorm_1.weight",
            "post_feedforward_layernorm_2.weight",
        ] {
            let full = format!("{lp}{t}");
            assert!(
                plan.ops.iter().any(|o| o.inputs.contains(&full)),
                "MoE plan does not reference {full}"
            );
        }
        // Fused expert GEMM covers the full E×2I×H weight bytes.
        let gu = plan
            .ops
            .iter()
            .find(|o| o.name == "moe_gate_up_L0")
            .unwrap();
        match &gu.kind {
            OpKind::Gemm(g) => assert_eq!(g.n * g.k, 128 * 1408 * 3840),
            _ => panic!(),
        }
    }

    #[test]
    fn test_moe_missing_fields_error() {
        let err = synthesize_full(
            &gemma_json(r#", "enable_moe_block": true"#),
            "bad-moe".into(),
        )
        .unwrap_err();
        assert!(err.contains("num_experts"), "{err}");
    }

    /// Kimi K2 config (scaled down) — same as the one in llama3_pipeline test.
    const KIMI_K2_JSON: &str = r#"{
        "model_type": "kimi",
        "vocab_size": 1000,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "q_lora_rank": 64,
        "kv_lora_rank": 32,
        "qk_rope_head_dim": 16,
        "qk_nope_head_dim": 48,
        "v_head_dim": 64,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "first_k_dense_replace": 2,
        "torch_dtype": "bfloat16"
    }"#;

    #[test]
    fn test_kimi_k2_synthesis() {
        let s = synthesize_full(KIMI_K2_JSON, "kimi-k2".into()).unwrap();
        assert_eq!(s.arch, HfArch::Kimi);
        assert_eq!(s.prefix, "model.");
        assert_eq!(s.num_layers, 4);
        assert_eq!(s.q_lora_rank, 64);
        assert_eq!(s.kv_lora_rank, 32);
        assert_eq!(s.qk_rope_head_dim, 16);
        assert_eq!(s.qk_nope_head_dim, 48);
        assert_eq!(s.v_head_dim, 64);
        assert!(s.moe);
        assert_eq!(s.n_exp, 8);
        assert_eq!(s.top_k, 2);
        assert_eq!(s.moe_inter, 256);
        assert_eq!(s.first_k_dense, 2);
        assert_eq!(s.n_shared_experts, 1);
        assert!(!s.tied_embed);
        // hd_slide/hd_full = qk_nope + qk_rope = 64
        assert_eq!(s.hd_slide, 64);
    }

    #[test]
    fn test_kimi_k2_full_model_plan_geometry() {
        let s = synthesize_full(KIMI_K2_JSON, "kimi-k2".into()).unwrap();
        let bucket = ShapeBucket {
            batch: 1,
            seq: 128,
            phase: schedule::Phase::Prefill,
        };
        let plan = build_full_model_plan(&bucket, &s);
        let get = |name: &str| {
            plan.ops
                .iter()
                .find(|o| o.name == name)
                .unwrap_or_else(|| panic!("missing op {name}"))
        };
        // Dense layer 0: has MLA projections, standard MLP (first_k_dense=2).
        // q_a_proj: h→q_lora_rank = 256→64
        match &get("q_a_proj_L0").kind {
            OpKind::Gemm(g) => assert_eq!((g.n, g.k), (64, 256)),
            _ => panic!("expected Gemm"),
        }
        // q_b_proj: q_lora_rank→heads*(nope+rope) = 64→4*64=256
        match &get("q_b_proj_L0").kind {
            OpKind::Gemm(g) => assert_eq!((g.n, g.k), (256, 64)),
            _ => panic!("expected Gemm"),
        }
        // kv_a_proj: h→kv_lora_rank+rope_hd = 256→48
        match &get("kv_a_proj_L0").kind {
            OpKind::Gemm(g) => assert_eq!((g.n, g.k), (48, 256)),
            _ => panic!("expected Gemm"),
        }
        // kv_b_proj: kv_lora_rank→heads*(nope+v) = 32→4*(48+64)=448
        match &get("kv_b_proj_L0").kind {
            OpKind::Gemm(g) => assert_eq!((g.n, g.k), (448, 32)),
            _ => panic!("expected Gemm"),
        }
        // Dense layer 0 has gate/up/down MLP
        assert!(plan.ops.iter().any(|o| o.name == "gate_L0"));
        assert!(plan.ops.iter().any(|o| o.name == "up_L0"));
        assert!(plan.ops.iter().any(|o| o.name == "down_L0"));
        // MoE layer 2 (first_k_dense=2, so layer 2+ are MoE): has router + experts
        assert!(plan.ops.iter().any(|o| o.name == "router_L2"));
        assert!(plan.ops.iter().any(|o| o.name == "sh_gate_L2"));
        assert!(plan.ops.iter().any(|o| o.name == "moe_gate_up_L2"));
        assert!(plan.ops.iter().any(|o| o.name == "moe_down_L2"));
        // Verify MoE weight names
        let lp = "model.layers.2.";
        for t in [
            "mlp.gate.weight",
            "mlp.shared_experts.gate_proj.weight",
            "mlp.shared_experts.up_proj.weight",
            "mlp.shared_experts.down_proj.weight",
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ] {
            let full = format!("{lp}{t}");
            assert!(
                plan.ops.iter().any(|o| o.inputs.contains(&full)),
                "Kimi plan does not reference {full}"
            );
        }
        // MLA weight names on layer 0
        let lp0 = "model.layers.0.";
        for t in [
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ] {
            let full = format!("{lp0}{t}");
            assert!(
                plan.ops.iter().any(|o| o.inputs.contains(&full)),
                "Kimi plan does not reference {full}"
            );
        }
        // lm_head is untied → separate weight
        assert_eq!(plan.ops.last().unwrap().inputs[1], "lm_head.weight");
    }

    #[test]
    fn test_kimi_k2_missing_mla_field() {
        // Drop q_lora_rank → hard error.
        let json = KIMI_K2_JSON.replace("\"q_lora_rank\": 64,", "");
        let err = synthesize_full(&json, "x".into()).unwrap_err();
        assert!(err.contains("q_lora_rank"), "{err}");
    }

    #[test]
    fn test_deepseek_v3_synthesis() {
        // Same config as Kimi, just different model_type.
        let json = KIMI_K2_JSON.replace(
            "\"model_type\": \"kimi\"",
            "\"model_type\": \"deepseek_v3\"",
        );
        let s = synthesize_full(&json, "deepseek-v3".into()).unwrap();
        assert_eq!(s.arch, HfArch::DeepSeek);
        assert_eq!(s.q_lora_rank, 64);
        assert_eq!(s.kv_lora_rank, 32);
        assert!(s.moe);
        assert_eq!(s.first_k_dense, 2);
        // Block emission should work identically.
        let bucket = ShapeBucket {
            batch: 1,
            seq: 128,
            phase: schedule::Phase::Decode,
        };
        let plan = build_full_model_plan(&bucket, &s);
        assert!(plan.ops.iter().any(|o| o.name == "q_a_proj_L0"));
        assert!(plan.ops.iter().any(|o| o.name == "moe_gate_up_L3"));
    }

    #[test]
    fn test_fp8_weight_dtype_from_config() {
        let json = KIMI_K2_JSON.replace(
            "\"torch_dtype\": \"bfloat16\"",
            "\"torch_dtype\": \"float8_e4m3fn\"",
        );
        let s = synthesize_full(&json, "kimi-fp8".into()).unwrap();
        assert_eq!(s.weight_dtype, nn_graph::DType::F8E4M3);
        // GEMM ops should carry FP8 weight dtype.
        let bucket = ShapeBucket {
            batch: 1,
            seq: 64,
            phase: schedule::Phase::Decode,
        };
        let plan = build_full_model_plan(&bucket, &s);
        let qa = plan.ops.iter().find(|o| o.name == "q_a_proj_L0").unwrap();
        assert_eq!(qa.weight_dtype, nn_graph::DType::F8E4M3);
        // Norms stay BF16.
        let norm = plan.ops.iter().find(|o| o.name == "norm_in_L0").unwrap();
        assert_eq!(norm.weight_dtype, nn_graph::DType::BF16);
    }

    #[test]
    fn test_mxfp4_weight_dtype_from_config() {
        // MX-microscaling FP4 checkpoint: quantization_config.quant_method = "mxfp4".
        let json = KIMI_K2_JSON.replace(
            "\"torch_dtype\": \"bfloat16\"",
            "\"torch_dtype\": \"bfloat16\", \"quantization_config\": {\"quant_method\": \"mxfp4\"}",
        );
        let s = synthesize_full(&json, "kimi-mxfp4".into()).unwrap();
        assert_eq!(s.weight_dtype, nn_graph::DType::F4);
        // GEMM ops carry the F4 weight dtype; norms stay BF16.
        let bucket = ShapeBucket {
            batch: 1,
            seq: 64,
            phase: schedule::Phase::Decode,
        };
        let plan = build_full_model_plan(&bucket, &s);
        let qa = plan.ops.iter().find(|o| o.name == "q_a_proj_L0").unwrap();
        assert_eq!(qa.weight_dtype, nn_graph::DType::F4);
        let norm = plan.ops.iter().find(|o| o.name == "norm_in_L0").unwrap();
        assert_eq!(norm.weight_dtype, nn_graph::DType::BF16);
    }

    /// A faithful miniature of the Kimi-K3 config: multimodal wrapper, nested `kimi_linear`
    /// text tower, Kimi MoE key spellings, 1-BASED hybrid layer lists (6 layers: MLA at 1-based
    /// {3,6}, KDA at {1,2,4,5}).
    fn kimi_k3_json() -> &'static str {
        r#"{
          "model_type": "kimi_k3",
          "text_config": {
            "model_type": "kimi_linear",
            "hidden_size": 256, "num_attention_heads": 8, "num_hidden_layers": 6,
            "vocab_size": 1000, "intermediate_size": 512,
            "q_lora_rank": 64, "kv_lora_rank": 32, "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8, "v_head_dim": 16,
            "num_experts": 32, "num_experts_per_token": 4, "num_shared_experts": 2,
            "moe_intermediate_size": 96, "routed_expert_hidden_size": 128,
            "first_k_dense_replace": 1,
            "linear_attn_config": {
              "full_attn_layers": [3, 6], "kda_layers": [1, 2, 4, 5]
            }
          }
        }"#
    }

    /// `text_config` used to mean "Gemma-4 multimodal" unconditionally, so kimi_k3 was parsed as
    /// Gemma and died on `head_dim` — the wrong field of the wrong architecture. It must now be
    /// claimed by name.
    #[test]
    fn test_kimi_k3_is_not_parsed_as_gemma() {
        let s = synthesize_full(kimi_k3_json(), "kimi-k3".into()).unwrap();
        assert_eq!(s.arch, HfArch::KimiK3);
        assert_eq!(s.prefix, "language_model.model.");
        assert_eq!(s.num_layers, 6);
        assert_eq!((s.q_lora_rank, s.kv_lora_rank), (64, 32));
        // Kimi MoE spellings, not DeepSeek's.
        assert_eq!((s.n_exp, s.top_k, s.n_shared_experts), (32, 4, 2));
        // The routed experts run at the LATENT width, not moe_intermediate_size.
        assert_eq!(s.moe_latent, 128);
        assert_ne!(s.moe_latent, s.moe_inter);
    }

    /// The 1-based → 0-based conversion, and the fact that the map is data rather than a count.
    #[test]
    fn test_kimi_k3_layer_map_is_one_based() {
        let s = synthesize_full(kimi_k3_json(), "kimi-k3".into()).unwrap();
        use AttnKind::*;
        assert_eq!(s.attn_kind, vec![Kda, Kda, Full, Kda, Kda, Full]);
        // `is_full` stays truthful: a KDA layer is not softmax attention.
        assert_eq!(s.is_full, vec![false, false, true, false, false, true]);
    }

    #[test]
    fn test_kimi_k3_layer_list_must_partition() {
        // 1-based layer 5 dropped → 0-based layer 4 unclassified. Deriving the complement would
        // hide it.
        let json = kimi_k3_json().replace("[1, 2, 4, 5]", "[1, 2, 4]");
        let err = synthesize_full(&json, "x".into()).unwrap_err();
        assert!(err.contains("neither"), "{err}");
        // Overlap.
        let json = kimi_k3_json().replace("[1, 2, 4, 5]", "[1, 2, 3, 4, 5]");
        let err = synthesize_full(&json, "x".into()).unwrap_err();
        assert!(err.contains("twice"), "{err}");
    }

    /// Vision is refused BY NAME on this path (the packet path has no vision emit at all and no
    /// report to fall back on).
    #[test]
    fn test_kimi_k3_vision_is_refused() {
        let json = kimi_k3_json().replace(
            r#""model_type": "kimi_k3","#,
            r#""model_type": "kimi_k3", "vision_config": {"vt_hidden_size": 1024},"#,
        );
        let err = synthesize_full(&json, "x".into()).unwrap_err();
        assert!(err.contains("vision_config"), "{err}");
        assert!(err.contains("TEXT tower"), "{err}");
    }

    /// Config acceptance is NOT emission: the plan builder must refuse rather than produce a
    /// uniform-softmax-attention plan for a 69/93-linear-attention model.
    #[test]
    #[should_panic(expected = "kimi_k3 has no full-model plan")]
    fn test_kimi_k3_has_no_full_model_plan() {
        let s = synthesize_full(kimi_k3_json(), "kimi-k3".into()).unwrap();
        let bucket = ShapeBucket {
            batch: 1,
            seq: 128,
            phase: schedule::Phase::Decode,
        };
        let _ = build_full_model_plan(&bucket, &s);
    }

    #[test]
    fn test_glm_moe_dsa_synthesis() {
        let json = r#"{
            "model_type": "glm_moe_dsa",
            "vocab_size": 1000,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "q_lora_rank": 64,
            "kv_lora_rank": 32,
            "qk_rope_head_dim": 16,
            "qk_nope_head_dim": 48,
            "v_head_dim": 64,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 256,
            "first_k_dense_replace": 2
        }"#;
        let s = synthesize_full(json, "glm-5.2".into()).unwrap();
        assert_eq!(s.arch, HfArch::Glm);
        assert_eq!(s.q_lora_rank, 64);
        assert_eq!(s.kv_lora_rank, 32);
        assert!(s.moe);
        assert_eq!(s.first_k_dense, 2);
        // Block emission works (MLA + MoE).
        let bucket = ShapeBucket {
            batch: 1,
            seq: 128,
            phase: schedule::Phase::Decode,
        };
        let plan = build_full_model_plan(&bucket, &s);
        assert!(plan.ops.iter().any(|o| o.name == "q_a_proj_L0"));
        assert!(plan.ops.iter().any(|o| o.name == "moe_gate_up_L3"));
        assert!(plan.ops.iter().any(|o| o.name == "lm_head"));
    }
}

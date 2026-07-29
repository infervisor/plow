//! Kimi K2 (Moonshot) configuration (HuggingFace `config.json`).
//!
//! Kimi K2 uses MLA (multi-head latent attention) + DeepSeek-style MoE,
//! architecturally very close to DeepSeek-V3 but with its own config naming.
//!
//! # Why the geometry fields are required
//!
//! This struct used to carry a blanket `#[serde(default)]` plus an `impl
//! Default` holding the full published K2 geometry (61 layers, hidden 7168, 256
//! experts). Any config.json that spelled a field differently — or omitted it —
//! therefore parsed CLEANLY and produced a real, plausible, wrong graph.
//!
//! That is not hypothetical. Kimi-K3 spells its MoE fields `num_experts` /
//! `num_experts_per_token` / `num_shared_experts`; this struct wants
//! `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts`. Three
//! silent fallbacks, and the result is a 61-layer K2 standing in for a 93-layer
//! K3. plow's own `hf_config.rs` refuses to work this way — every field there
//! goes through a `req_i64` that hard-errors — and the reason is in its module
//! docs: "compiling an unknown architecture from defaults would produce a
//! silently-wrong model". This file now agrees with that.
//!
//! What stays defaulted is only what is genuinely optional in the wild:
//! `rms_norm_eps` and `rope_theta` (stable across the family) and the
//! shared-expert count (absent on non-shared variants). Everything that changes
//! the graph's shape must be stated.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct KimiConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    // --- MLA (multi-head latent attention) ---
    /// Rank of the query down-projection (0 ⇒ no query LoRA).
    pub q_lora_rank: u32,
    /// Rank of the shared key/value down-projection.
    pub kv_lora_rank: u32,
    /// RoPE-carrying portion of the per-head query/key dim.
    pub qk_rope_head_dim: u32,
    /// Non-RoPE (content) portion of the per-head query/key dim.
    pub qk_nope_head_dim: u32,
    /// Per-head value dim.
    pub v_head_dim: u32,

    // --- MoE ---
    pub n_routed_experts: u32,
    /// Absent on variants with no shared expert, so this one may default.
    #[serde(default)]
    pub n_shared_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: i64,
    /// First `first_k_dense_replace` layers use a dense MLP, the rest use MoE.
    pub first_k_dense_replace: u32,

    #[serde(default, alias = "dtype")]
    pub torch_dtype: Option<String>,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10_000.0
}

impl KimiConfig {
    /// Full per-head query/key dim = nope + rope portions.
    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

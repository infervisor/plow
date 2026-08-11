//! Kimi-K3 (Moonshot) configuration (HuggingFace `config.json`).
//!
//! K3 is **not** a bigger K2. Three things differ structurally, and each one
//! silently produces a plausible wrong model if it is assumed away:
//!
//! * **Hybrid attention.** 24 MLA layers interleaved with 69 KDA (Kimi Delta
//!   Attention) layers. Which layer is which decides the tensor names, the
//!   kernel and the carried state — so it is per-layer data read from the
//!   config, never a stride.
//! * **A block residual.** `attn_res_block_size` routes every layer through
//!   `AttnRes`, a softmax mix over a running prefix sum and its snapshots,
//!   applied twice per layer. The plain `residual + attn` wiring is
//!   numerically indistinguishable from it at a non-snapshot layer.
//! * **Latent MoE.** The routed experts do not read the hidden state; the
//!   block projects `hidden → routed_expert_hidden_size` once and every expert
//!   GEMM runs at that width.
//!
//! # No blanket defaults
//!
//! Like [`super::KimiConfig`] since the K2/K3 mix-up, every field that changes
//! the graph is required. K3 spells its MoE fields `num_experts` /
//! `num_experts_per_token` / `num_shared_experts` where K2 spells them
//! `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts`; a struct
//! that defaults those parses one model's config into the other's graph
//! without complaining.

use serde::Deserialize;

use super::ConfigError;

/// The outer `config.json` (`model_type: "kimi_k3"`), which nests the text
/// tower under `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct KimiK3Config {
    pub text_config: KimiK3TextConfig,
    /// Present on the multimodal checkpoint (MoonViT tower + projector).
    /// Carried so the builder can refuse rather than silently compile the text
    /// tower and be wrong on every image prompt.
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
}

/// The text tower (`text_config.model_type: "kimi_linear"`).
#[derive(Debug, Clone, Deserialize)]
pub struct KimiK3TextConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    // --- MLA, on the full-attention layers. Same spelling as DeepSeek/GLM. ---
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_rope_head_dim: u32,
    pub qk_nope_head_dim: u32,
    pub v_head_dim: u32,
    #[serde(default)]
    pub mla_use_output_gate: bool,

    // --- MoE. Kimi-K3 spellings; see the module note. ---
    pub num_experts: u32,
    pub num_experts_per_token: u32,
    pub num_shared_experts: u32,
    pub moe_intermediate_size: i64,
    /// LATENT width. The routed experts run at this K, not at `hidden_size` —
    /// verified against the checkpoint, whose `experts.{e}.w1.weight_packed` is
    /// `[moe_intermediate_size, routed_expert_hidden_size / 2]` (mxfp4 packs
    /// two values per byte).
    pub routed_expert_hidden_size: i64,
    /// First `first_k_dense_replace` layers use a dense MLP.
    pub first_k_dense_replace: u32,
    /// DeepSeek `noaux_tc` group-limited routing. Absent ⇒ flat top-k.
    /// K3's own spelling is `num_expert_group` (default 1 in
    /// `configuration_kimi_k3.py`) — the released checkpoint carries
    /// `num_expert_group: 1, topk_group: 1`; `n_group` is the DeepSeek name.
    #[serde(default, alias = "num_expert_group")]
    pub n_group: Option<u32>,
    #[serde(default)]
    pub topk_group: Option<u32>,

    /// The `situ` activation's two clamps.
    #[serde(default = "default_situ_beta")]
    pub activation_situ_beta: f32,
    #[serde(default = "default_situ_linear_beta")]
    pub activation_situ_linear_beta: f32,

    /// Snapshot period for the block residual. Required: a wrong period puts
    /// the snapshots on the wrong layers, and the error is invisible at every
    /// non-snapshot layer.
    pub attn_res_block_size: u32,

    pub linear_attn_config: LinearAttnConfig,

    #[serde(default, alias = "dtype")]
    pub torch_dtype: Option<String>,
}

/// KDA geometry and the layer partition.
#[derive(Debug, Clone, Deserialize)]
pub struct LinearAttnConfig {
    /// KDA head count. K3 is not GVA — the key heads equal the value heads.
    pub num_heads: u32,
    /// Both the key dim and the value dim, so the recurrent state is square.
    pub head_dim: u32,
    pub short_conv_kernel_size: u32,
    #[serde(default)]
    pub gate_lower_bound: Option<f32>,
    #[serde(default)]
    pub use_full_rank_gate: bool,
    /// **1-BASED** layer indices. Converted once, in [`KimiK3TextConfig::attn_kinds`].
    pub full_attn_layers: Vec<i64>,
    /// **1-BASED**, and disjoint from `full_attn_layers`.
    pub kda_layers: Vec<i64>,
}

/// Which mixer a layer uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum K3Layer {
    /// Multi-head latent attention (softmax).
    Mla,
    /// Kimi Delta Attention — linear, with carried recurrent state.
    Kda,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_situ_beta() -> f32 {
    4.0
}

fn default_situ_linear_beta() -> f32 {
    25.0
}

impl KimiK3TextConfig {
    /// Full per-head query/key dim on the MLA layers.
    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Per-layer mixer, 0-based, one entry per layer.
    ///
    /// The config lists layers **1-based** (`configuration_kimi_k3.py` tests
    /// `(layer_idx + 1) in kda_layers`), so the conversion happens once, here.
    /// Both lists are validated to be disjoint and jointly complete: a layer in
    /// neither would otherwise take whichever branch the builder defaults to
    /// and bind the wrong tensors for the whole layer.
    pub fn attn_kinds(&self) -> Result<Vec<K3Layer>, ConfigError> {
        let n = self.num_hidden_layers as i64;
        let lac = &self.linear_attn_config;
        let mut out: Vec<Option<K3Layer>> = vec![None; n as usize];

        for (src, kind) in [
            (&lac.full_attn_layers, K3Layer::Mla),
            (&lac.kda_layers, K3Layer::Kda),
        ] {
            for &one_based in src {
                let l = one_based - 1;
                if !(0..n).contains(&l) {
                    return Err(ConfigError::Unsupported(format!(
                        "kimi_k3: linear_attn_config lists layer {one_based} (1-based, \
                         = {l} 0-based) but num_hidden_layers is {n}"
                    )));
                }
                if let Some(prev) = out[l as usize] {
                    return Err(ConfigError::Unsupported(format!(
                        "kimi_k3: linear_attn_config assigns 0-based layer {l} twice \
                         ({prev:?} and {kind:?}); full_attn_layers and kda_layers must \
                         be disjoint"
                    )));
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
            return Err(ConfigError::Unsupported(format!(
                "kimi_k3: linear_attn_config covers {} of {n} layers; 0-based layers \
                 {:?} are in neither full_attn_layers nor kda_layers",
                n as usize - missing.len(),
                &missing[..missing.len().min(8)]
            )));
        }
        Ok(out.into_iter().map(|k| k.unwrap()).collect())
    }

    /// Group-limited routing, if the config asks for it.
    ///
    /// Both fields must appear together: `n_group` alone would silently mean
    /// "keep every group" and `topk_group` alone has nothing to index.
    pub fn moe_groups(&self) -> Result<Option<crate::op::MoeGroups>, ConfigError> {
        match (self.n_group, self.topk_group) {
            (None, None) => Ok(None),
            (Some(n_group), Some(topk_group)) => Ok(Some(crate::op::MoeGroups {
                n_group,
                topk_group,
            })),
            (n, t) => Err(ConfigError::Unsupported(format!(
                "kimi_k3: n_group = {n:?} and topk_group = {t:?} must be given together \
                 — group-limited routing selects a different expert set than flat \
                 top-k, so a half-specified pair cannot be resolved by defaulting"
            ))),
        }
    }
}

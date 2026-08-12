//! DeepSeek-V4 (hyper-connections + compressed-KV MQA + FP4 MoE) configuration.
//!
//! This is a different architecture from V2/V3, not a wider one: the residual
//! stream is replaced by `hc_mult` hyper-connected streams, attention is single
//! -KV-head with a sliding window plus a learned KV compressor, and the first
//! `num_hash_layers` route their experts from a frozen token-id table. See
//! [`super::super::deepseek_v4`] for the builder and the reference it follows.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV4Config {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    /// Per-head width of both Q and the single shared KV head (512).
    pub head_dim: u32,
    /// RoPE-carrying lanes at the END of `head_dim`; the rest are content.
    pub qk_rope_head_dim: u32,
    pub q_lora_rank: u32,
    /// Head groups of the block-diagonal output projection.
    pub o_groups: u32,
    pub o_lora_rank: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Local-attention width; also the KV ring floor.
    pub sliding_window: u32,

    // --- compressed KV ---
    /// Per-layer KV compression ratio; `0` means the layer is sliding-window
    /// only. Indexed by layer id, and longer than `num_hidden_layers` because
    /// the checkpoint's DSpark stages are appended to the same list.
    pub compress_ratios: Vec<u32>,
    /// RoPE base for the compressed (YaRN-scaled) positions, which differs from
    /// `rope_theta` — the sliding-window-only layers disable YaRN entirely.
    pub compress_rope_theta: f32,

    // --- sparse index ---
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,

    // --- MoE ---
    pub n_routed_experts: u32,
    pub n_shared_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: i64,
    /// Leading layers whose expert set comes from `gate.tid2eid`, not the scores.
    pub num_hash_layers: u32,
    /// Symmetric clamp applied inside the expert SwiGLU.
    pub swiglu_limit: f32,
    pub routed_scaling_factor: f32,
    pub scoring_func: String,

    // --- hyper-connections ---
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,

    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

impl DeepSeekV4Config {
    /// Content (non-RoPE) lanes of a head.
    pub fn nope_head_dim(&self) -> u32 {
        self.head_dim - self.qk_rope_head_dim
    }

    /// KV compression ratio of `layer`, or `None` for a sliding-window-only layer.
    pub fn compress_ratio(&self, layer: u32) -> Option<u32> {
        match self.compress_ratios.get(layer as usize).copied() {
            Some(0) | None => None,
            Some(r) => Some(r),
        }
    }

    /// The compressor overlaps its windows exactly on the `ratio == 4` layers,
    /// which is also what gives those layers a sparse indexer.
    pub fn overlaps(ratio: u32) -> bool {
        ratio == 4
    }

    /// Per-head width of the block-diagonal `wo_a` input.
    pub fn o_group_width(&self) -> i64 {
        self.num_attention_heads as i64 * self.head_dim as i64 / self.o_groups as i64
    }

    /// Rows of a per-layer hyper-connection projection.
    pub fn hc_rows(&self) -> i64 {
        (2 + self.hc_mult as i64) * self.hc_mult as i64
    }

    pub fn rope_theta_for(&self, layer: u32) -> f32 {
        match self.compress_ratio(layer) {
            Some(_) => self.compress_rope_theta,
            None => self.rope_theta,
        }
    }
}

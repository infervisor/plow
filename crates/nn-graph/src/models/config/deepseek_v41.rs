//! DeepSeek-V4.1-Flash configuration (`model_type: "deepseek_v41"`).
//!
//! V4.1 keeps V4's MQA-at-a-512-latent block, grouped output LoRA, mHC and
//! MXFP4 experts, and replaces the attention schedule around them. Each change
//! parses cleanly into [`super::DeepSeekV4Config`]'s field names with a
//! different meaning, which is why this is a separate front end:
//!
//! * **CED / CSA2 layer modes.** `compress_ratios` is `0`, `1` or `2` (V4: 0,
//!   4, 128) and a ratio-1 layer is a real compressor (one latent per token),
//!   which V4's validate rejects. Compressed KV and index-K caches are OWNED
//!   by `kv_source_layer_ids` / `index_source_layer_ids` and read by every
//!   layer up to the next owner; `candidate_source_layer_id` additionally
//!   selects a candidate block pool that the later Reindex layers score in.
//! * **Block-32 fp8** (`weight_block_size [32,32]`, V4: 128) and an
//!   `expert_dtype` that lives inside `quantization_config`.
//! * **Engram** on `engram_layer_ids`: hashed n-gram tables of
//!   `engram_num_embeddings` rows each, gated into the residual streams.
//! * **Nested** `text_config` / `vision_config`, with `quantization_config`,
//!   `dtype` and `image_token_id` at the root.
//!
//! No blanket defaults: every field that changes the graph is required.

use serde::Deserialize;

use super::deepseek_v4::{DeepSeekV4RopeScaling, MXFP4_GROUP};
use super::ConfigError;

/// The outer `config.json` (`model_type: "deepseek_v41"`).
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV41Config {
    pub text_config: DeepSeekV41TextConfig,
    /// `deepseek_v41_vision` tower (32 blocks) + aligner.
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
    /// Root-level on the HF file, not under `text_config`.
    pub quantization_config: DeepSeekV41QuantizationConfig,
    #[serde(default)]
    pub image_token_id: Option<i64>,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

/// `quantization_config` — block-FP8 ue8m0 projections and MXFP4 routed
/// experts, declared together (V4 put `expert_dtype` at the top level).
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV41QuantizationConfig {
    pub quant_method: String,
    pub activation_scheme: String,
    pub weight_block_size: [i64; 2],
    pub scale_fmt: String,
    pub expert_dtype: String,
}

/// What a layer does with its compressed KV and index.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum V41Attn {
    /// `compress_ratio == 0`: the 128-entry sliding window only.
    Swa,
    /// Owns a compressor (`ratio` tokens per latent), an index-K cache and
    /// its own top-k.
    Full { ratio: u32 },
    /// Reads KV and index-K from the last Full layer; runs its own scorer,
    /// restricted to the candidate pool.
    Reindex,
    /// Reads KV and the top-k of the nearest owner above.
    Reuse,
}

/// The text tower (`text_config.model_type: "deepseek_v41_text"`).
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV41TextConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub moe_intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    /// 1: one KV head serves every query head.
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub q_lora_rank: u32,
    pub o_lora_rank: u32,
    pub o_groups: u32,
    pub hidden_act: String,
    pub swiglu_limit: f32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub max_position_embeddings: i64,
    pub rope_theta: f32,
    pub rope_scaling: DeepSeekV4RopeScaling,
    pub tie_word_embeddings: bool,

    // --- MoE ---
    pub n_routed_experts: u32,
    pub n_shared_experts: u32,
    pub num_experts_per_tok: u32,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,

    // --- CSA2 / CED ---
    pub sliding_window: u32,
    /// One entry per layer, then one per DSpark block; each 0, 1 or 2.
    pub compress_ratios: Vec<u32>,
    pub compress_rope_theta: f32,
    /// Layers that own a compressed-KV cache.
    pub kv_source_layer_ids: Vec<u32>,
    /// Layers that own an index-K cache and a top-k; a superset of the KV sources.
    pub index_source_layer_ids: Vec<u32>,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub candidate_source_layer_id: u32,
    pub candidate_topk_blocks: u32,
    pub candidate_block_size: u32,

    // --- mHC ---
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,

    // --- engram ---
    pub engram_layer_ids: Vec<u32>,
    /// Table rows per engram layer, parallel to `engram_layer_ids`.
    pub engram_num_embeddings: Vec<u64>,
    pub engram_max_ngram_size: u32,
    pub engram_vocab_size: u64,
    pub engram_n_heads: u32,
    pub engram_head_dim: u32,
    pub engram_pad_token_id: i64,
    pub engram_compressed_vocab_size: u64,

    // --- DSpark ---
    pub num_nextn_predict_layers: u32,
    pub dspark_block_size: u32,
    pub dspark_noise_token_id: i64,
    pub dspark_target_layer_ids: Vec<u32>,
    pub dspark_markov_rank: i64,
    pub dspark_n_routed_experts: u32,
    pub dspark_num_experts_per_tok: u32,
}

impl DeepSeekV41TextConfig {
    pub fn qk_nope_head_dim(&self) -> u32 {
        self.head_dim - self.qk_rope_head_dim
    }

    pub fn dspark_blocks(&self) -> usize {
        self.dspark_target_layer_ids.len()
    }

    pub fn layer_has_engram(&self, layer: u32) -> bool {
        self.engram_layer_ids.contains(&layer)
    }

    /// Attention mode of base layer `layer`; `validate()` guarantees the index.
    pub fn attn_mode(&self, layer: u32) -> V41Attn {
        let ratio = self.compress_ratios[layer as usize];
        let kv = self.kv_source_layer_ids.contains(&layer);
        let index = self.index_source_layer_ids.contains(&layer);
        match (ratio, kv, index) {
            (0, _, _) => V41Attn::Swa,
            (ratio, true, true) => V41Attn::Full { ratio },
            (_, false, true) => V41Attn::Reindex,
            (_, _, false) => V41Attn::Reuse,
        }
    }

    /// Per-layer mode, 0-based, one entry per base layer.
    pub fn attn_modes(&self) -> Vec<V41Attn> {
        (0..self.num_hidden_layers)
            .map(|l| self.attn_mode(l))
            .collect()
    }

    /// Nearest KV owner at or above `layer`, if the layer reads compressed KV.
    pub fn kv_source_of(&self, layer: u32) -> Option<u32> {
        if self.compress_ratios[layer as usize] == 0 {
            return None;
        }
        self.kv_source_layer_ids
            .iter()
            .copied()
            .filter(|s| *s <= layer)
            .max()
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |message: String| Err(ConfigError::Unsupported(message));
        let n = self.num_hidden_layers;

        if n == 0
            || self.hidden_size <= 0
            || self.vocab_size <= 0
            || self.moe_intermediate_size <= 0
            || self.head_dim == 0
            || self.num_attention_heads == 0
        {
            return bad("deepseek_v41 dimensions must be positive".into());
        }

        // --- attention geometry ---
        if self.num_key_value_heads != 1 {
            return bad(format!(
                "deepseek_v41 is MQA at one {}-wide latent; num_key_value_heads={} is not the \
                 released geometry",
                self.head_dim, self.num_key_value_heads
            ));
        }
        if self.qk_rope_head_dim == 0 || self.qk_rope_head_dim >= self.head_dim {
            return bad(format!(
                "deepseek_v41 qk_rope_head_dim={} must be a proper prefix of head_dim={}",
                self.qk_rope_head_dim, self.head_dim
            ));
        }
        if self.q_lora_rank == 0 || self.o_lora_rank == 0 {
            return bad("deepseek_v41 q_lora_rank and o_lora_rank must be positive".into());
        }
        if self.o_groups == 0 || self.num_attention_heads % self.o_groups != 0 {
            return bad(format!(
                "deepseek_v41 o_groups={} must divide num_attention_heads={}",
                self.o_groups, self.num_attention_heads
            ));
        }
        if self.attention_bias {
            return bad("deepseek_v41 attention_bias=true is not supported".into());
        }
        if self.sliding_window == 0 {
            return bad("deepseek_v41 sliding_window must be positive".into());
        }

        // --- CSA2 / CED ---
        let expected = n as usize + self.dspark_blocks();
        if self.compress_ratios.len() != expected {
            return bad(format!(
                "deepseek_v41 compress_ratios has {} entries; expected {expected} = \
                 num_hidden_layers ({n}) + dspark_target_layer_ids ({} MTP blocks)",
                self.compress_ratios.len(),
                self.dspark_blocks()
            ));
        }
        if let Some(r) = self.compress_ratios.iter().find(|r| **r > 2) {
            return bad(format!(
                "deepseek_v41 compress_ratios entry {r} is not one of 0 (window only), 1 or 2"
            ));
        }
        for (what, ids) in [
            ("kv_source_layer_ids", &self.kv_source_layer_ids),
            ("index_source_layer_ids", &self.index_source_layer_ids),
        ] {
            if ids.is_empty() {
                return bad(format!("deepseek_v41 {what} is empty"));
            }
            if ids.windows(2).any(|w| w[0] >= w[1]) {
                return bad(format!("deepseek_v41 {what} must be strictly increasing"));
            }
            if let Some(id) = ids.iter().find(|id| **id >= n) {
                return bad(format!(
                    "deepseek_v41 {what} names layer {id} but num_hidden_layers is {n}"
                ));
            }
            if let Some(id) = ids
                .iter()
                .find(|id| self.compress_ratios[**id as usize] == 0)
            {
                return bad(format!(
                    "deepseek_v41 {what} names layer {id}, whose compress_ratio is 0 (window only)"
                ));
            }
        }
        if let Some(id) = self
            .kv_source_layer_ids
            .iter()
            .find(|id| !self.index_source_layer_ids.contains(id))
        {
            return bad(format!(
                "deepseek_v41 kv_source_layer_ids names layer {id}, which is not in \
                 index_source_layer_ids; a KV owner must also own its index"
            ));
        }
        if !self
            .kv_source_layer_ids
            .contains(&self.candidate_source_layer_id)
        {
            return bad(format!(
                "deepseek_v41 candidate_source_layer_id={} is not a kv_source layer {:?}",
                self.candidate_source_layer_id, self.kv_source_layer_ids
            ));
        }
        for layer in 0..n {
            if self.compress_ratios[layer as usize] != 0 && self.kv_source_of(layer).is_none() {
                return bad(format!(
                    "deepseek_v41 layer {layer} has compress_ratio {} but no kv_source layer at \
                     or above it",
                    self.compress_ratios[layer as usize]
                ));
            }
        }
        if self.index_head_dim == 0 || self.index_n_heads == 0 || self.index_topk == 0 {
            return bad("deepseek_v41 indexer dimensions must be positive".into());
        }
        if self.candidate_topk_blocks == 0 || self.candidate_block_size == 0 {
            return bad("deepseek_v41 candidate pool dimensions must be positive".into());
        }
        if self.compress_rope_theta <= 0.0 || self.rope_theta <= 0.0 {
            return bad("deepseek_v41 RoPE bases must be positive".into());
        }

        // --- RoPE ---
        if self.rope_scaling.kind != "yarn" {
            return bad(format!(
                "deepseek_v41 unsupported RoPE scaling {:?}",
                self.rope_scaling.kind
            ));
        }
        if self.rope_scaling.factor <= 0.0
            || self.rope_scaling.original_max_position_embeddings <= 0
            || self.rope_scaling.beta_fast <= self.rope_scaling.beta_slow
        {
            return bad("deepseek_v41 invalid YaRN parameters".into());
        }
        let scaled = (self.rope_scaling.original_max_position_embeddings as f64
            * self.rope_scaling.factor as f64) as i64;
        if scaled != self.max_position_embeddings {
            return bad(format!(
                "deepseek_v41 YaRN factor {} over original_max_position_embeddings {} gives \
                 {scaled}, but max_position_embeddings is {}",
                self.rope_scaling.factor,
                self.rope_scaling.original_max_position_embeddings,
                self.max_position_embeddings
            ));
        }

        // --- MoE ---
        if self.n_routed_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.n_routed_experts
        {
            return bad(format!(
                "deepseek_v41 invalid expert routing: {} of {} routed experts per token",
                self.num_experts_per_tok, self.n_routed_experts
            ));
        }
        if self.hidden_act != "silu"
            || self.scoring_func != "sqrtsoftplus"
            || self.topk_method != "noaux_tc"
        {
            return bad(format!(
                "deepseek_v41 requires silu experts with sqrtsoftplus/noaux_tc routing; got \
                 hidden_act={:?}, scoring_func={:?}, topk_method={:?}",
                self.hidden_act, self.scoring_func, self.topk_method
            ));
        }
        if !self.norm_topk_prob || self.routed_scaling_factor <= 0.0 || self.swiglu_limit <= 0.0 {
            return bad(
                "deepseek_v41 requires normalized, positively scaled top-k weights and a \
                 positive swiglu_limit"
                    .into(),
            );
        }
        for (what, k) in [
            ("hidden_size", self.hidden_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
        ] {
            if k % MXFP4_GROUP != 0 {
                return bad(format!(
                    "deepseek_v41 MXFP4 routed experts need {what}={k} to be a multiple of \
                     the {MXFP4_GROUP}-element microscaling group"
                ));
            }
        }
        if self.tie_word_embeddings {
            return bad(
                "deepseek_v41 ships an untied head.weight; tie_word_embeddings=true would \
                 discard it"
                    .into(),
            );
        }

        // --- mHC ---
        if self.hc_mult == 0 || self.hc_sinkhorn_iters == 0 || self.hc_eps <= 0.0 {
            return bad(format!(
                "deepseek_v41 mHC needs hc_mult>0, hc_sinkhorn_iters>0 and hc_eps>0; got {}, {}, {}",
                self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
            ));
        }

        // --- engram ---
        if self.engram_num_embeddings.len() != self.engram_layer_ids.len() {
            return bad(format!(
                "deepseek_v41 engram_num_embeddings has {} entries for {} engram_layer_ids",
                self.engram_num_embeddings.len(),
                self.engram_layer_ids.len()
            ));
        }
        if let Some(id) = self.engram_layer_ids.iter().find(|id| **id >= n) {
            return bad(format!(
                "deepseek_v41 engram_layer_ids names layer {id} but num_hidden_layers is {n}"
            ));
        }
        if self.engram_num_embeddings.contains(&0)
            || self.engram_max_ngram_size < 2
            || self.engram_vocab_size == 0
            || self.engram_n_heads == 0
            || self.engram_head_dim == 0
            || self.engram_compressed_vocab_size == 0
            || self.engram_compressed_vocab_size > self.vocab_size as u64
        {
            return bad("deepseek_v41 engram dimensions must be positive (compressed vocab within vocab_size)".into());
        }
        if self.engram_pad_token_id < 0 || self.engram_pad_token_id >= self.vocab_size {
            return bad(format!(
                "deepseek_v41 engram_pad_token_id={} is outside vocab_size={}",
                self.engram_pad_token_id, self.vocab_size
            ));
        }

        // --- DSpark ---
        if self.dspark_blocks() == 0 || self.num_nextn_predict_layers == 0 {
            return bad("deepseek_v41 dspark_target_layer_ids and num_nextn_predict_layers must be non-empty".into());
        }
        let mut seen = self.dspark_target_layer_ids.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != self.dspark_target_layer_ids.len() {
            return bad("deepseek_v41 dspark_target_layer_ids must be distinct".into());
        }
        if let Some(id) = seen.last() {
            if *id >= n {
                return bad(format!(
                    "deepseek_v41 dspark_target_layer_ids names layer {id} but num_hidden_layers is {n}"
                ));
            }
        }
        if self.dspark_block_size == 0 || self.dspark_markov_rank <= 0 {
            return bad(
                "deepseek_v41 dspark_block_size and dspark_markov_rank must be positive".into(),
            );
        }
        if self.dspark_noise_token_id < 0 || self.dspark_noise_token_id >= self.vocab_size {
            return bad(format!(
                "deepseek_v41 dspark_noise_token_id={} is outside vocab_size={}",
                self.dspark_noise_token_id, self.vocab_size
            ));
        }
        if self.dspark_n_routed_experts == 0
            || self.dspark_num_experts_per_tok == 0
            || self.dspark_num_experts_per_tok > self.dspark_n_routed_experts
        {
            return bad(format!(
                "deepseek_v41 invalid DSpark routing: {} of {} experts per token",
                self.dspark_num_experts_per_tok, self.dspark_n_routed_experts
            ));
        }
        Ok(())
    }
}

impl DeepSeekV41Config {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.text_config.validate()?;
        let q = &self.quantization_config;
        if q.quant_method != "fp8"
            || q.activation_scheme != "dynamic"
            || q.scale_fmt != "ue8m0"
            || q.weight_block_size != [32, 32]
            || q.expert_dtype != "fp4"
        {
            return Err(ConfigError::Unsupported(format!(
                "deepseek_v41 requires dynamic block-FP8 [32,32] with ue8m0 scales and fp4 \
                 experts; got method={:?}, activation={:?}, scale_fmt={:?}, block={:?}, \
                 expert_dtype={:?}",
                q.quant_method,
                q.activation_scheme,
                q.scale_fmt,
                q.weight_block_size,
                q.expert_dtype
            )));
        }
        if self.torch_dtype.as_deref() != Some("bfloat16") {
            return Err(ConfigError::Unsupported(
                "deepseek_v41 requires dtype=bfloat16".into(),
            ));
        }
        Ok(())
    }

    /// Every piece this compiler cannot lower yet, with the config's numbers,
    /// modeled on [`super::DeepSeekV4Config::unimplemented`]. The config parses
    /// and validates; the emit refuses.
    pub fn unimplemented(&self) -> String {
        let t = &self.text_config;
        let modes = t.attn_modes();
        let count = |pred: fn(&V41Attn) -> bool| modes.iter().filter(|m| pred(m)).count();
        let reindex: Vec<u32> = (0..t.num_hidden_layers)
            .filter(|l| modes[*l as usize] == V41Attn::Reindex)
            .collect();
        let engram_bytes: u64 = t
            .engram_num_embeddings
            .iter()
            .map(|rows| rows * t.engram_head_dim as u64)
            .sum();
        let q = &self.quantization_config;
        format!(
            "deepseek_v41 graph emit is not implemented (config parses and validates; \
             refusing a DeepSeek-V4 or V3 fallback). Unimplemented: \
             (1) CED cross-layer sharing -- {nfull} Full layers {kv:?} own a compressed-KV + \
             index-K cache read by {nreuse} Reuse layers, {nreindex} Reindex layers {reindex:?} \
             re-score inside the candidate pool that layer {cand} selects (top-{ctb} blocks of \
             {cbs}), with the top-{itk} selection forwarded across layers; \
             (2) the ratio-2 (2-slot gated, no ape) and ratio-1 (GEMV + norm) compressors over \
             {layers} layers (compress_rope_theta={crt}, YaRN x{yarn}); \
             (3) the FP4 compressed-KV cache (e2m1 + e4m3 RNE scale per 16, rope included); \
             (4) block-32 fake-fp8 on the {sw}-entry window KV and on the attention output \
             before wo_a; \
             (5) single-pass mHC (hc_mult={mult}, {sink} Sinkhorn iterations, rstd eps {eps:e}, \
             pre computed one sublayer early, exit = last ffn_pre, no hc_head); \
             (6) engram on layers {eng:?} ({ntables} tables, {rows:?} rows x {ehd} fp8 ~ {gb} GB, \
             {enh} heads, n-grams up to {ng}, compressed vocab {cv}); \
             (7) fp8 block-[{b0},{b1}] {sfmt} projections and shared expert; \
             (8) w4a8 {edt} routed experts ({experts} top-{topk} + {shared} shared, I={moe_i}, \
             per-32-K activation scale); \
             (9) the grouped o-LoRA (o_groups={og} x o_lora_rank={olr}) for \
             num_key_value_heads={kvh} at DK={hd}; \
             (10) {scoring}/{method} routing with bias_vl and the clamped SwiGLU (limit {lim}); \
             (11) DSpark ({blocks} MTP blocks over layers {targets:?}, {dexp} experts top-{dtopk}, \
             markov rank {mrank}, block size {bsz}) -- OPTIONAL: base output does not depend on \
             it, so a first bringup is bit-exact without it",
            nfull = count(|m| matches!(m, V41Attn::Full { .. })),
            kv = t.kv_source_layer_ids,
            nreuse = count(|m| *m == V41Attn::Reuse),
            nreindex = reindex.len(),
            cand = t.candidate_source_layer_id,
            ctb = t.candidate_topk_blocks,
            cbs = t.candidate_block_size,
            itk = t.index_topk,
            layers = t.num_hidden_layers,
            crt = t.compress_rope_theta,
            yarn = t.rope_scaling.factor,
            sw = t.sliding_window,
            mult = t.hc_mult,
            sink = t.hc_sinkhorn_iters,
            eps = t.rms_norm_eps,
            eng = t.engram_layer_ids,
            ntables = t.engram_layer_ids.len(),
            rows = t.engram_num_embeddings,
            ehd = t.engram_head_dim,
            gb = engram_bytes / 1_000_000_000,
            enh = t.engram_n_heads,
            ng = t.engram_max_ngram_size - 1,
            cv = t.engram_compressed_vocab_size,
            b0 = q.weight_block_size[0],
            b1 = q.weight_block_size[1],
            sfmt = q.scale_fmt,
            edt = q.expert_dtype,
            experts = t.n_routed_experts,
            topk = t.num_experts_per_tok,
            shared = t.n_shared_experts,
            moe_i = t.moe_intermediate_size,
            og = t.o_groups,
            olr = t.o_lora_rank,
            kvh = t.num_key_value_heads,
            hd = t.head_dim,
            scoring = t.scoring_func,
            method = t.topk_method,
            lim = t.swiglu_limit,
            blocks = t.dspark_blocks(),
            targets = t.dspark_target_layer_ids,
            dexp = t.dspark_n_routed_experts,
            dtopk = t.dspark_num_experts_per_tok,
            mrank = t.dspark_markov_rank,
            bsz = t.dspark_block_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::config::ModelConfig;

    /// deepseek-ai/DeepSeek-V4.1-Flash's `config.json`, VERBATIM.
    pub(crate) const RELEASED: &str = r#"{
  "architectures": ["DeepseekV41ForCausalLM"],
  "model_type": "deepseek_v41",
  "dtype": "bfloat16",
  "transformers_version": "5.6.0",
  "bos_token_id": 0,
  "eos_token_id": 1,
  "pad_token_id": 2,
  "image_token_id": 129264,
  "quantization_config": {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4"
  },
  "text_config": {
    "model_type": "deepseek_v41_text",
    "vocab_size": 129280,
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "rms_norm_eps": 1e-20,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "use_cache": true,
    "tie_word_embeddings": false,
    "max_position_embeddings": 1048576,
    "rope_theta": 10000,
    "rope_scaling": {
      "rope_type": "yarn",
      "factor": 16,
      "beta_fast": 32,
      "beta_slow": 1,
      "original_max_position_embeddings": 65536
    },
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": true,
    "routed_scaling_factor": 1.5,
    "sliding_window": 128,
    "compress_ratios": [0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3
  },
  "vision_config": {
    "model_type": "deepseek_v41_vision",
    "num_hidden_layers": 32,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "intermediate_size": 2816,
    "patch_size": 14,
    "rope_theta": 10000,
    "downsample_ratio": 3,
    "max_image_tokens": 1024,
    "min_pixels": 295936,
    "max_wh_ratio": null
  }
}"#;

    fn released() -> DeepSeekV41Config {
        match ModelConfig::from_json(RELEASED).expect("released config must parse and validate") {
            ModelConfig::DeepSeekV41(cfg) => cfg,
            other => panic!("released DeepSeek-V4.1 config parsed as {other:?}"),
        }
    }

    #[test]
    fn released_config_parses_with_its_released_values() {
        let c = released();
        let t = &c.text_config;
        assert!(c.vision_config.is_some());
        assert_eq!(c.image_token_id, Some(129_264));
        assert_eq!(c.torch_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(c.quantization_config.weight_block_size, [32, 32]);
        assert_eq!(c.quantization_config.expert_dtype, "fp4");
        assert_eq!(t.vocab_size, 129_280);
        assert_eq!(t.hidden_size, 5120);
        assert_eq!(t.num_hidden_layers, 40);
        assert_eq!((t.num_attention_heads, t.num_key_value_heads), (64, 1));
        assert_eq!(
            (t.head_dim, t.qk_rope_head_dim, t.qk_nope_head_dim()),
            (512, 64, 448)
        );
        assert_eq!((t.q_lora_rank, t.o_lora_rank, t.o_groups), (1280, 1024, 8));
        assert_eq!(t.rms_norm_eps, 1e-20);
        assert_eq!(t.rope_scaling.kind, "yarn");
        assert_eq!(
            (
                t.n_routed_experts,
                t.num_experts_per_tok,
                t.moe_intermediate_size
            ),
            (384, 6, 2304)
        );
        assert_eq!(t.compress_ratios.len(), 43);
        assert_eq!(t.kv_source_layer_ids, vec![2, 8, 14, 20]);
        assert_eq!(t.index_source_layer_ids.len(), 8);
        assert_eq!(
            (t.index_n_heads, t.index_head_dim, t.index_topk),
            (32, 128, 512)
        );
        assert_eq!(
            (
                t.candidate_source_layer_id,
                t.candidate_topk_blocks,
                t.candidate_block_size
            ),
            (20, 2048, 8)
        );
        assert_eq!(t.engram_layer_ids, vec![1, 14]);
        assert_eq!(t.engram_num_embeddings, vec![384_006_168, 384_016_682]);
        assert_eq!(t.engram_compressed_vocab_size, 99_092);
        assert_eq!(t.dspark_target_layer_ids, vec![37, 38, 39]);
        assert_eq!(
            (t.dspark_n_routed_experts, t.dspark_num_experts_per_tok),
            (128, 3)
        );
        assert_eq!(t.dspark_blocks(), 3);
    }

    #[test]
    fn layer_modes_match_the_ced_schedule() {
        let t = released().text_config;
        let modes = t.attn_modes();
        assert_eq!(modes.len(), 40);
        assert_eq!(modes[0], V41Attn::Swa);
        assert_eq!(modes[1], V41Attn::Swa);
        for owner in [2, 8, 14] {
            assert_eq!(modes[owner], V41Attn::Full { ratio: 2 }, "layer {owner}");
        }
        assert_eq!(modes[20], V41Attn::Full { ratio: 1 });
        for l in [3, 7, 9, 13, 15, 19, 21, 23, 25, 27, 37, 39] {
            assert_eq!(modes[l], V41Attn::Reuse, "layer {l}");
        }
        for l in [24, 28, 32, 36] {
            assert_eq!(modes[l], V41Attn::Reindex, "layer {l}");
        }
        assert_eq!(t.kv_source_of(0), None);
        assert_eq!(t.kv_source_of(13), Some(8));
        assert_eq!(t.kv_source_of(39), Some(20));
        assert!(t.layer_has_engram(1) && t.layer_has_engram(14) && !t.layer_has_engram(2));
    }

    /// The bare text tower, as `build_text_generation_from_config_json_at`
    /// hands it over (root `dtype` and `quantization_config` lifted in).
    #[test]
    fn bare_text_config_wraps_without_a_vision_tower() {
        let v: serde_json::Value = serde_json::from_str(RELEASED).unwrap();
        let mut text = v["text_config"].clone();
        text["quantization_config"] = v["quantization_config"].clone();
        text["dtype"] = v["dtype"].clone();
        match ModelConfig::from_json(&text.to_string()).expect("deepseek_v41_text parses") {
            ModelConfig::DeepSeekV41(c) => {
                assert!(c.vision_config.is_none());
                assert_eq!(c.image_token_id, None);
                assert_eq!(c.text_config.num_hidden_layers, 40);
            }
            other => panic!("deepseek_v41_text parsed as {other:?}"),
        }
    }

    #[test]
    fn emit_gap_list_names_every_unimplemented_piece() {
        let msg = released().unimplemented();
        for piece in [
            "deepseek_v41 graph emit is not implemented",
            "CED cross-layer sharing -- 4 Full layers [2, 8, 14, 20]",
            "30 Reuse layers",
            "4 Reindex layers [24, 28, 32, 36]",
            "layer 20 selects (top-2048 blocks of 8)",
            "ratio-2 (2-slot gated, no ape) and ratio-1 (GEMV + norm) compressors over 40 layers",
            "FP4 compressed-KV cache (e2m1 + e4m3 RNE scale per 16",
            "block-32 fake-fp8 on the 128-entry window KV and on the attention output",
            "single-pass mHC (hc_mult=4, 20 Sinkhorn iterations, rstd eps 1e-20",
            "engram on layers [1, 14] (2 tables, [384006168, 384016682] rows x 256 fp8 ~ 196 GB",
            "fp8 block-[32,32] ue8m0 projections",
            "w4a8 fp4 routed experts (384 top-6 + 1 shared, I=2304",
            "o_groups=8 x o_lora_rank=1024",
            "sqrtsoftplus/noaux_tc routing with bias_vl and the clamped SwiGLU (limit 10)",
            "DSpark (3 MTP blocks over layers [37, 38, 39], 128 experts top-3",
        ] {
            assert!(msg.contains(piece), "missing {piece:?} in: {msg}");
        }
        let d = msg.find("DSpark").unwrap();
        assert!(msg[d..].contains("OPTIONAL"), "{msg}");
    }

    fn reject(mutate: impl FnOnce(&mut serde_json::Value)) -> String {
        let mut v: serde_json::Value = serde_json::from_str(RELEASED).unwrap();
        mutate(&mut v);
        ModelConfig::from_json(&v.to_string())
            .expect_err("mutated config must be rejected")
            .to_string()
    }

    #[test]
    fn internally_inconsistent_configs_fail_closed() {
        // 42 ratios for 40 layers + 3 DSpark blocks.
        let err = reject(|v| {
            v["text_config"]["compress_ratios"]
                .as_array_mut()
                .unwrap()
                .pop();
        });
        assert!(err.contains("compress_ratios has 42 entries"), "{err}");

        // V4's ratio 4 is not a V4.1 compressor.
        let err = reject(|v| v["text_config"]["compress_ratios"][2] = 4.into());
        assert!(err.contains("entry 4 is not one of 0"), "{err}");

        // A KV owner must own its index too.
        let err =
            reject(|v| v["text_config"]["kv_source_layer_ids"] = serde_json::json!([2, 8, 14, 21]));
        assert!(
            err.contains("names layer 21, which is not in index_source_layer_ids"),
            "{err}"
        );

        // The candidate pool comes from a KV owner.
        let err = reject(|v| v["text_config"]["candidate_source_layer_id"] = 24.into());
        assert!(
            err.contains("candidate_source_layer_id=24 is not a kv_source layer"),
            "{err}"
        );

        // A compressed layer with nothing to read: layer 2 demoted leaves 2..7 ownerless.
        let err =
            reject(|v| v["text_config"]["kv_source_layer_ids"] = serde_json::json!([8, 14, 20]));
        assert!(
            err.contains("layer 2 has compress_ratio 2 but no kv_source layer"),
            "{err}"
        );

        // Two engram layers need two table sizes.
        let err =
            reject(|v| v["text_config"]["engram_num_embeddings"] = serde_json::json!([384006168]));
        assert!(
            err.contains("engram_num_embeddings has 1 entries for 2 engram_layer_ids"),
            "{err}"
        );

        // V4's [128,128] grid binds a quarter of the scale rows a block-32 kernel reads.
        let err = reject(|v| {
            v["quantization_config"]["weight_block_size"] = serde_json::json!([128, 128])
        });
        assert!(err.contains("block-FP8 [32,32]"), "{err}");

        // One KV head only.
        let err = reject(|v| v["text_config"]["num_key_value_heads"] = 8.into());
        assert!(err.contains("num_key_value_heads=8"), "{err}");

        let err = reject(|v| v["text_config"]["scoring_func"] = "sigmoid".into());
        assert!(err.contains("sqrtsoftplus"), "{err}");
    }

    /// `architectures: ["DeepseekV41ForCausalLM"]` alone must not be captured
    /// by the `DeepseekV4` prefix.
    #[test]
    fn architectures_fallback_does_not_reach_deepseek_v4() {
        let mut v: serde_json::Value = serde_json::from_str(RELEASED).unwrap();
        v.as_object_mut().unwrap().remove("model_type");
        match ModelConfig::from_json(&v.to_string()).expect("parses via architectures") {
            ModelConfig::DeepSeekV41(_) => {}
            other => panic!("DeepseekV41ForCausalLM parsed as {other:?}"),
        }
    }
}

//! DeepSeek-V4-Flash configuration (`model_type: "deepseek_v4"`).
//!
//! V4 is **not** a wider V3. It keeps the DeepSeek MoE shape and almost nothing
//! else about the attention block, and each difference silently produces a
//! plausible wrong model if it is defaulted away:
//!
//! * **Compressed sparse attention (CSA).** `compress_ratios` is per-layer data,
//!   not a stride: a `0` layer has no compressor at all, and a nonzero `r`
//!   layer pools `r` tokens into one cache entry through its own gated
//!   projection, its own absolute-position table and its own
//!   `compress_rope_theta`. Which layers are which decides the tensor names,
//!   the KV-cache size and the kernel.
//! * **MQA at a 512-wide latent.** `num_key_value_heads = 1` with
//!   `head_dim = 512`: one KV head serves all 64 query heads, so the KV latent
//!   is REPLICATED under tensor parallelism instead of sharded, and the query
//!   side is the only head-parallel part of the block.
//! * **A grouped output LoRA.** `o_groups` x `o_lora_rank` replaces `o_proj`.
//!   The 64 heads are cut into `o_groups` groups, each group's
//!   `num_attention_heads / o_groups * head_dim` slice gets its own down
//!   projection to `o_lora_rank`, and one shared up projection reads the
//!   concatenation. This is what caps clean tensor parallelism at `o_groups`.
//! * **mHC residuals.** `hc_mult` copies of the residual stream, mixed per
//!   layer by a Sinkhorn-normalized matrix (`hc_sinkhorn_iters`, `hc_eps`).
//!   Plain `residual + attn` is numerically indistinguishable from it only
//!   when `hc_mult == 1`.
//! * **Two weight encodings in one layer.** The routed experts are
//!   `expert_dtype: "fp4"` (MXFP4: nibble-packed, one E8M0 scale per 32 values
//!   along K) while every projection and the shared expert are block-FP8 with
//!   a `ue8m0` scale grid at `weight_block_size`. A single `projection_weight_dtype`
//!   cannot describe the block.
//! * **An attached DSpark module.** `dspark_target_layer_ids` names the main-model
//!   layers whose hidden states feed it; the checkpoint carries one MTP
//!   transformer block per id, not `num_nextn_predict_layers` of them.
//!
//! # No blanket defaults
//!
//! Like [`super::GlmConfig`], every field that changes the graph is required.
//! `#[serde(default)]` here would let a V3 `config.json` deserialize into this
//! struct and build a V4-shaped graph out of V3 weights.
//!
//! # Reconciling `compress_ratios` with the layer count
//!
//! DeepSeek-V4-Flash-0731 has `num_hidden_layers = 43` and 46 `compress_ratios`
//! entries. The extra three are the DSpark blocks: 43 + `dspark_target_layer_ids.len()`
//! = 46, and the checkpoint's `mtp.{0,1,2}.attn` carries no `compressor.*`
//! tensors, matching the three trailing zeros. `num_nextn_predict_layers` (1)
//! counts speculative ITERATIONS and does not reconcile the length.

use serde::Deserialize;

use super::ConfigError;

/// Elements per E8M0 scale in the MXFP4 routed-expert encoding. Fixed by the
/// OCP microscaling format, not by `config.json`; `DevOp::GemvMxfp4` documents
/// the same `[N, K/32]` scale layout the checkpoint ships.
pub const MXFP4_GROUP: i64 = 32;

/// `quantization_config` — block-FP8 with a **ue8m0** (power-of-two) scale
/// grid. The scale format is load-bearing: a ue8m0 scale folds into the
/// dequant exactly, an f32 grid does not, and V3's config carries no
/// `scale_fmt` at all.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV4QuantizationConfig {
    pub activation_scheme: String,
    pub fmt: String,
    pub quant_method: String,
    pub scale_fmt: String,
    pub weight_block_size: [i64; 2],
}

/// `rope_scaling` — YaRN over `original_max_position_embeddings`.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV4RopeScaling {
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub factor: f32,
    pub original_max_position_embeddings: i64,
    #[serde(rename = "type", alias = "rope_type")]
    pub kind: String,
}

/// What a layer's attention does with its KV history.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum V4Attn {
    /// `compress_ratio == 0`: no `attn.compressor.*` tensors; attention reads
    /// the raw per-token KV latent.
    Uncompressed,
    /// A CSA compressor pools `ratio` tokens into one cache entry.
    Compressed { ratio: u32 },
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV4Config {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    /// **1** on the released checkpoint: one shared KV head, so the KV latent
    /// cannot be head-sharded. See [`DeepSeekV4Config::kv_latent_dim`].
    pub num_key_value_heads: u32,
    /// Per-head query dim AND the width of the single KV latent.
    pub head_dim: u32,
    /// RoPE-carrying prefix of `head_dim`; the rest is the content part.
    pub qk_rope_head_dim: u32,
    pub q_lora_rank: u32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub hidden_act: String,
    pub max_position_embeddings: i64,
    pub tie_word_embeddings: bool,

    // --- grouped output LoRA (replaces `o_proj`) ---
    pub o_lora_rank: u32,
    pub o_groups: u32,

    // --- CSA ---
    /// One entry per layer, then one per DSpark block. `0` = no compressor.
    pub compress_ratios: Vec<u32>,
    /// RoPE base for the compressed cache — deliberately different from
    /// `rope_theta`, since compressed entries span `ratio` positions.
    pub compress_rope_theta: f32,
    /// Local span kept uncompressed alongside the compressed summary.
    pub sliding_window: u32,

    // --- lightning indexer ---
    pub index_head_dim: u32,
    pub index_n_heads: u32,
    pub index_topk: u32,

    // --- RoPE ---
    pub rope_theta: f32,
    pub rope_scaling: DeepSeekV4RopeScaling,

    // --- MoE ---
    pub n_routed_experts: u32,
    pub n_shared_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: i64,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    /// Clamp on the SwiGLU gate; 0 would silently mean "no clamp".
    pub swiglu_limit: f32,
    /// Storage encoding of the ROUTED experts only. The shared expert and every
    /// projection stay block-FP8 — see [`DeepSeekV4Config::mxfp4_scale_shape`].
    pub expert_dtype: String,
    /// Leading layers whose router is a token-id lookup (`ffn.gate.tid2eid`)
    /// instead of a learned gate; they carry no `gate.bias`.
    pub num_hash_layers: u32,

    // --- mHC residuals ---
    pub hc_eps: f32,
    /// Residual-stream expansion. `hc_*_fn` reads `hc_mult * hidden_size`.
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,

    // --- DSpark speculative module ---
    pub num_nextn_predict_layers: u32,
    pub dspark_block_size: u32,
    pub dspark_noise_token_id: i64,
    /// Main-model layers whose hidden states feed DSpark. Its LENGTH is the
    /// number of MTP blocks in the checkpoint.
    pub dspark_target_layer_ids: Vec<u32>,
    pub dspark_markov_rank: i64,

    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
    pub quantization_config: DeepSeekV4QuantizationConfig,
}

impl DeepSeekV4Config {
    /// Non-RoPE (content) portion of `head_dim`.
    pub fn qk_nope_head_dim(&self) -> u32 {
        self.head_dim - self.qk_rope_head_dim
    }

    /// Width of the cached KV latent, per token per layer. `num_key_value_heads`
    /// is 1 on the released checkpoint, so this is `head_dim` — and it is the
    /// same on every rank, because a single KV head cannot be split.
    pub fn kv_latent_dim(&self) -> i64 {
        self.num_key_value_heads as i64 * self.head_dim as i64
    }

    /// Query heads in one output-LoRA group.
    pub fn heads_per_o_group(&self) -> u32 {
        self.num_attention_heads / self.o_groups
    }

    /// Input width of one output-LoRA group's down projection: the attention
    /// output slice its heads produce.
    pub fn o_group_in_features(&self) -> i64 {
        self.heads_per_o_group() as i64 * self.head_dim as i64
    }

    /// Number of DSpark MTP transformer blocks — the length of
    /// `dspark_target_layer_ids`, NOT `num_nextn_predict_layers`.
    pub fn dspark_blocks(&self) -> usize {
        self.dspark_target_layer_ids.len()
    }

    /// Attention kind of base layer `layer`. `validate()` guarantees the index
    /// is in range.
    pub fn attn_kind(&self, layer: u32) -> V4Attn {
        Self::attn_kind_of(self.compress_ratios[layer as usize])
    }

    /// Attention kind of DSpark block `block`, whose `compress_ratios` entry
    /// follows the base layers.
    pub fn dspark_attn_kind(&self, block: usize) -> V4Attn {
        Self::attn_kind_of(self.compress_ratios[self.num_hidden_layers as usize + block])
    }

    /// A leading `num_hash_layers` layer routes by token-id lookup.
    pub fn layer_is_hash_routed(&self, layer: u32) -> bool {
        layer < self.num_hash_layers
    }

    fn attn_kind_of(ratio: u32) -> V4Attn {
        match ratio {
            0 => V4Attn::Uncompressed,
            ratio => V4Attn::Compressed { ratio },
        }
    }

    /// `[out_blocks, in_blocks]` of a block-FP8 ue8m0 scale grid.
    pub fn fp8_scale_shape(&self, out_features: i64, in_features: i64) -> [i64; 2] {
        let [out_block, in_block] = self.quantization_config.weight_block_size;
        [
            (out_features + out_block - 1) / out_block,
            (in_features + in_block - 1) / in_block,
        ]
    }

    /// `[out_features, in_features / 32]` — the MXFP4 routed-expert scale grid,
    /// one E8M0 byte per 32 values along K.
    pub fn mxfp4_scale_shape(&self, out_features: i64, in_features: i64) -> [i64; 2] {
        [out_features, (in_features + MXFP4_GROUP - 1) / MXFP4_GROUP]
    }

    /// Every piece this compiler cannot lower yet, listed. The config parses
    /// and validates; the *emit* refuses, so the gap is a checklist rather than
    /// a dead end at `config.json`.
    ///
    /// A line leaves this list only when the KERNEL it names exists AND has a
    /// hardware test. That is a lower bar than "the graph emits", deliberately:
    /// the list is a device-capability inventory, and the emitter's own missing
    /// pieces are named separately in the closing sentence so the two cannot be
    /// confused. `docs/amd/deepseek-v4-bringup-20260908.md` carries the full
    /// state, one section per line here.
    pub fn unimplemented(&self) -> String {
        let mut ratios: Vec<u32> = self
            .compress_ratios
            .iter()
            .copied()
            .filter(|r| *r > 0)
            .collect();
        ratios.sort_unstable();
        ratios.dedup();
        format!(
            "deepseek_v4 graph emit is not implemented (config parses and validates; \
             refusing a DeepSeek-V3 MLA fallback). Unimplemented: \
             (6) mixed {edt} routed experts and block-FP8 {sfmt} projections in one layer -- \
             MoeGluMx/MoeDownMx already read the fp4 expert weights byte-identically, but \
             those arms are w4a16 and the reference is w4a8 (fp8 e4m3 activations, per-128-K \
             power-of-2 scale), which needs an fp8-activation arm on the fp4 expert GEMM; \
             (7) the attached DSpark speculative module ({blocks} MTP blocks, markov rank \
             {mrank}, block size {bsz}) -- OPTIONAL: the shipped generate.py never calls \
             forward_spec, so a first bringup is bit-exact on base output without it. \
             Kernels that now EXIST and are hardware-tested but that no emitter reaches yet: \
             the CSA/HCA compressor (compress_ratios {ratios:?} over {layers} layers, \
             compress_rope_theta={crt}); the lightning indexer at {heads}x{ihd} heads with \
             top-{topk} selection; {scoring} routing and the {hash} hash-routed layers \
             (tid2eid); mHC (hc_mult={mult}, {sink} Sinkhorn iterations) including the \
             learned gated tower exit; the DK={hd} DR=0 MLA prefill for \
             num_key_value_heads={kvh}; and the clamped SwiGLU. Still missing on the EMITTER \
             side for all of those: per-layer rope tables, the 128-entry window ring, the \
             grouped output LoRA (o_groups={og} x o_lora_rank={olr}), and a deepseek_v4 \
             graph builder",
            layers = self.num_hidden_layers,
            crt = self.compress_rope_theta,
            topk = self.index_topk,
            heads = self.index_n_heads,
            ihd = self.index_head_dim,
            scoring = self.scoring_func,
            hash = self.num_hash_layers,
            mult = self.hc_mult,
            sink = self.hc_sinkhorn_iters,
            kvh = self.num_key_value_heads,
            hd = self.head_dim,
            og = self.o_groups,
            olr = self.o_lora_rank,
            edt = self.expert_dtype,
            sfmt = self.quantization_config.scale_fmt,
            blocks = self.dspark_blocks(),
            mrank = self.dspark_markov_rank,
            bsz = self.dspark_block_size,
        )
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |message: String| Err(ConfigError::Unsupported(message));

        if self.num_hidden_layers == 0
            || self.hidden_size <= 0
            || self.vocab_size <= 0
            || self.moe_intermediate_size <= 0
            || self.head_dim == 0
            || self.num_attention_heads == 0
        {
            return bad("deepseek_v4 dimensions must be positive".into());
        }

        // --- attention geometry ---
        if self.qk_rope_head_dim == 0 || self.qk_rope_head_dim >= self.head_dim {
            return bad(format!(
                "deepseek_v4 qk_rope_head_dim={} must be a proper prefix of head_dim={}",
                self.qk_rope_head_dim, self.head_dim
            ));
        }
        if self.num_key_value_heads == 0 || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return bad(format!(
                "deepseek_v4 num_attention_heads={} must be a positive multiple of \
                 num_key_value_heads={}",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if self.q_lora_rank == 0 {
            return bad(
                "deepseek_v4 requires a query LoRA; q_lora_rank=0 has no wq_a/q_norm to bind"
                    .into(),
            );
        }
        // The output LoRA is block-diagonal over `o_groups`, so a group is the
        // indivisible unit of head assignment: a rank that owns a fraction of a
        // group owns a fraction of that group's down projection's INPUT, which
        // no longer matches its rows.
        if self.o_groups == 0 || self.num_attention_heads % self.o_groups != 0 {
            return bad(format!(
                "deepseek_v4 o_groups={} must divide num_attention_heads={}",
                self.o_groups, self.num_attention_heads
            ));
        }
        if self.o_lora_rank == 0 {
            return bad("deepseek_v4 o_lora_rank must be positive".into());
        }
        if self.attention_bias {
            return bad("deepseek_v4 attention_bias=true is not supported".into());
        }

        // --- CSA ---
        let expected = self.num_hidden_layers as usize + self.dspark_blocks();
        if self.compress_ratios.len() != expected {
            return bad(format!(
                "deepseek_v4 compress_ratios has {} entries; expected {expected} = \
                 num_hidden_layers ({}) + dspark_target_layer_ids ({} MTP blocks). \
                 num_nextn_predict_layers ({}) counts speculative iterations and does not \
                 reconcile the length",
                self.compress_ratios.len(),
                self.num_hidden_layers,
                self.dspark_blocks(),
                self.num_nextn_predict_layers
            ));
        }
        if let Some(r) = self.compress_ratios.iter().find(|r| **r == 1) {
            return bad(format!(
                "deepseek_v4 compress_ratios entry {r} pools one token into one entry; use 0 \
                 to mean `no compressor`"
            ));
        }
        if self.sliding_window == 0 {
            return bad("deepseek_v4 sliding_window must be positive".into());
        }
        if self.compress_rope_theta <= 0.0 || self.rope_theta <= 0.0 {
            return bad("deepseek_v4 RoPE bases must be positive".into());
        }

        // --- indexer ---
        if self.index_head_dim == 0 || self.index_n_heads == 0 || self.index_topk == 0 {
            return bad("deepseek_v4 indexer dimensions must be positive".into());
        }
        if (self.index_topk as i64) > self.max_position_embeddings {
            return bad(format!(
                "deepseek_v4 index_topk={} exceeds max_position_embeddings={}",
                self.index_topk, self.max_position_embeddings
            ));
        }

        // --- RoPE ---
        if self.rope_scaling.kind != "yarn" {
            return bad(format!(
                "deepseek_v4 unsupported RoPE scaling {:?}",
                self.rope_scaling.kind
            ));
        }
        if self.rope_scaling.factor <= 0.0
            || self.rope_scaling.original_max_position_embeddings <= 0
            || self.rope_scaling.beta_fast <= self.rope_scaling.beta_slow
        {
            return bad("deepseek_v4 invalid YaRN parameters".into());
        }
        let scaled = (self.rope_scaling.original_max_position_embeddings as f64
            * self.rope_scaling.factor as f64) as i64;
        if scaled != self.max_position_embeddings {
            return bad(format!(
                "deepseek_v4 YaRN factor {} over original_max_position_embeddings {} gives \
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
                "deepseek_v4 invalid expert routing: {} of {} routed experts per token",
                self.num_experts_per_tok, self.n_routed_experts
            ));
        }
        if self.num_hash_layers > self.num_hidden_layers {
            return bad(format!(
                "deepseek_v4 num_hash_layers={} exceeds num_hidden_layers={}",
                self.num_hash_layers, self.num_hidden_layers
            ));
        }
        if self.hidden_act != "silu"
            || self.scoring_func != "sqrtsoftplus"
            || self.topk_method != "noaux_tc"
        {
            return bad(format!(
                "deepseek_v4 requires silu experts with sqrtsoftplus/noaux_tc routing; got \
                 hidden_act={:?}, scoring_func={:?}, topk_method={:?}",
                self.hidden_act, self.scoring_func, self.topk_method
            ));
        }
        if !self.norm_topk_prob || self.routed_scaling_factor <= 0.0 || self.swiglu_limit <= 0.0 {
            return bad(
                "deepseek_v4 requires normalized, positively scaled top-k weights and a \
                 positive swiglu_limit"
                    .into(),
            );
        }

        // --- storage encodings ---
        if self.expert_dtype != "fp4" {
            return bad(format!(
                "deepseek_v4 expert_dtype {:?} is not the released MXFP4 encoding",
                self.expert_dtype
            ));
        }
        // MXFP4 packs two values per byte with one E8M0 scale per 32 along K,
        // so both routed-expert K widths must be whole groups.
        for (what, k) in [
            ("hidden_size", self.hidden_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
        ] {
            if k % MXFP4_GROUP != 0 {
                return bad(format!(
                    "deepseek_v4 MXFP4 routed experts need {what}={k} to be a multiple of \
                     the {MXFP4_GROUP}-element microscaling group"
                ));
            }
        }
        let q = &self.quantization_config;
        if q.quant_method != "fp8"
            || q.fmt != "e4m3"
            || q.activation_scheme != "dynamic"
            || q.scale_fmt != "ue8m0"
            || q.weight_block_size != [128, 128]
        {
            return bad(format!(
                "deepseek_v4 requires dynamic e4m3 block-FP8 [128,128] with ue8m0 scales; got \
                 method={:?}, fmt={:?}, activation={:?}, scale_fmt={:?}, block={:?}",
                q.quant_method, q.fmt, q.activation_scheme, q.scale_fmt, q.weight_block_size
            ));
        }
        if self.torch_dtype.as_deref() != Some("bfloat16") {
            return bad("deepseek_v4 requires dtype=bfloat16".into());
        }
        if self.tie_word_embeddings {
            return bad(
                "deepseek_v4 ships an untied head.weight; tie_word_embeddings=true would \
                 discard it"
                    .into(),
            );
        }

        // --- mHC ---
        if self.hc_mult == 0 || self.hc_sinkhorn_iters == 0 || self.hc_eps <= 0.0 {
            return bad(format!(
                "deepseek_v4 mHC needs hc_mult>0, hc_sinkhorn_iters>0 and hc_eps>0; got {}, {}, {}",
                self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
            ));
        }

        // --- DSpark ---
        if self.dspark_blocks() == 0 {
            return bad(
                "deepseek_v4 dspark_target_layer_ids is empty; the checkpoint's mtp.* blocks \
                 have no source layers"
                    .into(),
            );
        }
        if self.num_nextn_predict_layers == 0 {
            return bad("deepseek_v4 num_nextn_predict_layers must be positive".into());
        }
        let mut seen = self.dspark_target_layer_ids.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != self.dspark_target_layer_ids.len() {
            return bad("deepseek_v4 dspark_target_layer_ids must be distinct".into());
        }
        if let Some(id) = seen.last() {
            if *id >= self.num_hidden_layers {
                return bad(format!(
                    "deepseek_v4 dspark_target_layer_ids names layer {id} but \
                     num_hidden_layers is {}",
                    self.num_hidden_layers
                ));
            }
        }
        if self.dspark_block_size == 0 || self.dspark_markov_rank <= 0 {
            return bad(
                "deepseek_v4 dspark_block_size and dspark_markov_rank must be positive".into(),
            );
        }
        if self.dspark_noise_token_id < 0 || self.dspark_noise_token_id >= self.vocab_size {
            return bad(format!(
                "deepseek_v4 dspark_noise_token_id={} is outside vocab_size={}",
                self.dspark_noise_token_id, self.vocab_size
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::config::ModelConfig;

    /// deepseek-ai/DeepSeek-V4-Flash-0731's `config.json`, VERBATIM.
    const RELEASED: &str = r#"{
  "architectures":["DeepseekV4ForCausalLM"],
  "attention_bias":false,
  "attention_dropout":0.0,
  "bos_token_id":0,
  "eos_token_id":1,
  "expert_dtype":"fp4",
  "hc_eps":1e-06,
  "hc_mult":4,
  "hc_sinkhorn_iters":20,
  "head_dim":512,
  "hidden_act":"silu",
  "hidden_size":4096,
  "index_head_dim":128,
  "index_n_heads":64,
  "index_topk":512,
  "initializer_range":0.02,
  "max_position_embeddings":1048576,
  "model_type":"deepseek_v4",
  "moe_intermediate_size":2048,
  "n_routed_experts":256,
  "n_shared_experts":1,
  "norm_topk_prob":true,
  "num_attention_heads":64,
  "num_experts_per_tok":6,
  "num_hidden_layers":43,
  "num_hash_layers":3,
  "num_key_value_heads":1,
  "num_nextn_predict_layers":1,
  "o_groups":8,
  "o_lora_rank":1024,
  "q_lora_rank":1024,
  "qk_rope_head_dim":64,
  "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","scale_fmt":"ue8m0","weight_block_size":[128,128]},
  "rms_norm_eps":1e-06,
  "rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":65536,"type":"yarn"},
  "rope_theta":10000,
  "routed_scaling_factor":1.5,
  "scoring_func":"sqrtsoftplus",
  "sliding_window":128,
  "swiglu_limit":10.0,
  "tie_word_embeddings":false,
  "topk_method":"noaux_tc",
  "torch_dtype":"bfloat16",
  "transformers_version":"4.57.1",
  "use_cache":true,
  "vocab_size":129280,
  "compress_rope_theta":160000,
  "compress_ratios":[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,
    4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0],
  "dspark_block_size":5,
  "dspark_noise_token_id":128799,
  "dspark_target_layer_ids":[40,41,42],
  "dspark_markov_rank":256
}"#;

    fn released() -> DeepSeekV4Config {
        match ModelConfig::from_json(RELEASED).expect("released config must parse and validate") {
            ModelConfig::DeepSeekV4(cfg) => cfg,
            other => panic!("released DeepSeek-V4 config parsed as {other:?}"),
        }
    }

    #[test]
    fn released_config_parses_with_its_released_values() {
        let c = released();
        assert_eq!(c.vocab_size, 129_280);
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_hidden_layers, 43);
        assert_eq!(c.num_attention_heads, 64);
        assert_eq!(c.num_key_value_heads, 1);
        assert_eq!(c.head_dim, 512);
        assert_eq!(c.qk_rope_head_dim, 64);
        assert_eq!(c.qk_nope_head_dim(), 448);
        assert_eq!(c.q_lora_rank, 1024);
        assert_eq!(c.o_lora_rank, 1024);
        assert_eq!(c.o_groups, 8);
        assert_eq!(c.max_position_embeddings, 1_048_576);
        assert_eq!(c.sliding_window, 128);
        assert_eq!(c.compress_rope_theta, 160_000.0);
        assert_eq!(c.rope_theta, 10_000.0);
        assert_eq!(c.rope_scaling.kind, "yarn");
        assert_eq!(c.rope_scaling.factor, 16.0);
        assert_eq!(c.rope_scaling.original_max_position_embeddings, 65_536);
        assert_eq!(
            (c.index_head_dim, c.index_n_heads, c.index_topk),
            (128, 64, 512)
        );
        assert_eq!(c.n_routed_experts, 256);
        assert_eq!(c.n_shared_experts, 1);
        assert_eq!(c.num_experts_per_tok, 6);
        assert_eq!(c.moe_intermediate_size, 2048);
        assert_eq!(c.scoring_func, "sqrtsoftplus");
        assert_eq!(c.topk_method, "noaux_tc");
        assert_eq!(c.routed_scaling_factor, 1.5);
        assert_eq!(c.swiglu_limit, 10.0);
        assert_eq!(c.expert_dtype, "fp4");
        assert_eq!(c.num_hash_layers, 3);
        assert_eq!((c.hc_mult, c.hc_sinkhorn_iters), (4, 20));
        assert_eq!(c.hc_eps, 1e-6);
        assert_eq!(c.num_nextn_predict_layers, 1);
        assert_eq!(c.dspark_block_size, 5);
        assert_eq!(c.dspark_noise_token_id, 128_799);
        assert_eq!(c.dspark_markov_rank, 256);
        assert_eq!(c.dspark_target_layer_ids, vec![40, 41, 42]);
        assert_eq!(c.quantization_config.scale_fmt, "ue8m0");
        assert_eq!(c.quantization_config.weight_block_size, [128, 128]);
        assert!(!c.tie_word_embeddings);
        assert!(!c.attention_bias);
    }

    /// 46 ratios for 43 layers: the extra three are the DSpark blocks, one per
    /// `dspark_target_layer_ids` entry — NOT `num_nextn_predict_layers`, which
    /// is 1. Every DSpark entry is 0, matching a checkpoint whose `mtp.*.attn`
    /// carries no `compressor.*` tensors.
    #[test]
    fn compress_ratios_reconcile_against_layers_plus_dspark_blocks() {
        let c = released();
        assert_eq!(c.compress_ratios.len(), 46);
        assert_eq!(c.dspark_blocks(), 3);
        assert_eq!(
            c.compress_ratios.len(),
            c.num_hidden_layers as usize + c.dspark_blocks()
        );
        for block in 0..c.dspark_blocks() {
            assert_eq!(c.dspark_attn_kind(block), V4Attn::Uncompressed);
        }
        assert_eq!(c.attn_kind(0), V4Attn::Uncompressed);
        assert_eq!(c.attn_kind(1), V4Attn::Uncompressed);
        // Layers 2..=42 alternate a fine and a coarse compressor.
        for layer in 2..c.num_hidden_layers {
            let ratio = if layer % 2 == 0 { 4 } else { 128 };
            assert_eq!(
                c.attn_kind(layer),
                V4Attn::Compressed { ratio },
                "layer {layer}"
            );
        }
        assert!(c.layer_is_hash_routed(2));
        assert!(!c.layer_is_hash_routed(3));
    }

    /// The o-LoRA is block-diagonal: 64 heads / 8 groups = 8 heads x 512 =
    /// 4096, which is exactly `wo_a.weight`'s stored in-features.
    #[test]
    fn output_lora_group_geometry_matches_the_checkpoint() {
        let c = released();
        assert_eq!(c.heads_per_o_group(), 8);
        assert_eq!(c.o_group_in_features(), 4096);
        assert_eq!(
            c.o_groups as i64 * c.o_lora_rank as i64,
            8192,
            "wo_a out-features / wo_b in-features"
        );
        assert_eq!(c.kv_latent_dim(), 512);
    }

    #[test]
    fn scale_grids_match_the_checkpoint() {
        let c = released();
        // wq_b is [64*512, 1024] block-FP8.
        assert_eq!(c.fp8_scale_shape(32_768, 1024), [256, 8]);
        // wo_a is [8*1024, 4096], wo_b is [4096, 8*1024].
        assert_eq!(c.fp8_scale_shape(8192, 4096), [64, 32]);
        assert_eq!(c.fp8_scale_shape(4096, 8192), [32, 64]);
        // Routed experts are MXFP4: [N, K/32] E8M0 rows.
        assert_eq!(c.mxfp4_scale_shape(2048, 4096), [2048, 128]);
        assert_eq!(c.mxfp4_scale_shape(4096, 2048), [4096, 64]);
    }

    /// The gap list is a DEVICE-CAPABILITY inventory, so this test is in two halves and the
    /// halves must not drift into each other. A piece moves from `STILL_MISSING` to `HAVE_KERNEL`
    /// only when the kernel it names exists AND has a hardware test on gfx942 --
    /// `docs/amd/deepseek-v4-bringup-20260908.md` records which test, on what shapes, and what
    /// negative control proved it bites.
    const STILL_MISSING: [&str; 5] = [
        "graph emit is not implemented",
        "w4a8", // the routed experts' fp8-activation arm
        "ue8m0",
        "DSpark",
        "3 MTP blocks",
    ];
    /// Named in the same string as EXISTING, so a regression that deletes an arm is caught by
    /// the wording changing rather than by the string merely shrinking.
    const HAVE_KERNEL: [&str; 8] = [
        "CSA/HCA compressor",
        "compress_ratios [4, 128]",
        "lightning indexer",
        "sqrtsoftplus",
        "hash-routed",
        "mHC",
        "DR=0 MLA prefill",
        "clamped SwiGLU",
    ];
    /// Present but on the EMITTER's side of the line, not the device's.
    const EMITTER_GAPS: [&str; 4] = [
        "per-layer rope tables",
        "128-entry window ring",
        "o_groups=8",
        "graph builder",
    ];

    #[test]
    fn emit_gap_list_names_every_unimplemented_piece() {
        let msg = released().unimplemented();
        for piece in STILL_MISSING
            .iter()
            .chain(HAVE_KERNEL.iter())
            .chain(EMITTER_GAPS.iter())
        {
            assert!(msg.contains(piece), "missing {piece:?} in: {msg}");
        }
        // The closing sentence is what keeps the two halves apart; without it a reader takes
        // the whole string as "none of this exists", which is what it used to mean.
        assert!(
            msg.contains("Kernels that now EXIST and are hardware-tested"),
            "the gap list must separate device gaps from emitter gaps: {msg}"
        );
        assert!(
            msg.contains("Still missing on the EMITTER side"),
            "the gap list must name the emitter's own gaps: {msg}"
        );
    }

    /// DSpark is optional for a bit-exact base-model bringup (`generate.py` never calls
    /// `forward_spec`), and the refusal must say so rather than reading as a hard blocker.
    #[test]
    fn emit_gap_list_marks_dspark_optional() {
        let msg = released().unimplemented();
        let d = msg.find("DSpark").expect("DSpark named");
        assert!(msg[d..].contains("OPTIONAL"), "DSpark must be marked optional: {msg}");
        assert!(msg[d..].contains("forward_spec"), "and say why: {msg}");
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
        // One ratio too few: 45 entries for 43 layers + 3 DSpark blocks.
        let err = reject(|v| {
            let r = v["compress_ratios"].as_array_mut().unwrap();
            r.pop();
        });
        assert!(err.contains("compress_ratios has 45 entries"), "{err}");
        assert!(err.contains("num_nextn_predict_layers"), "{err}");

        // 8 groups do not divide 60 heads.
        let err = reject(|v| v["num_attention_heads"] = 60.into());
        assert!(
            err.contains("o_groups=8 must divide num_attention_heads=60"),
            "{err}"
        );

        // The YaRN factor must reproduce max_position_embeddings.
        let err = reject(|v| v["rope_scaling"]["factor"] = 8.into());
        assert!(err.contains("max_position_embeddings is 1048576"), "{err}");

        // Routing more experts than exist.
        let err = reject(|v| v["num_experts_per_tok"] = 512.into());
        assert!(err.contains("512 of 256 routed experts"), "{err}");

        // The indexer must be fully specified.
        let err = reject(|v| v["index_topk"] = 0.into());
        assert!(err.contains("indexer dimensions must be positive"), "{err}");

        // f32 scale grids do not fold exactly into the dequant.
        let err = reject(|v| v["quantization_config"]["scale_fmt"] = "float32".into());
        assert!(err.contains("ue8m0"), "{err}");

        // A [128,64] grid would bind half the scale rows a kernel reads.
        let err = reject(|v| {
            v["quantization_config"]["weight_block_size"] = serde_json::json!([128, 64])
        });
        assert!(err.contains("block-FP8 [128,128]"), "{err}");

        // A DSpark source layer outside the model.
        let err = reject(|v| v["dspark_target_layer_ids"] = serde_json::json!([40, 41, 43]));
        assert!(err.contains("names layer 43"), "{err}");

        // V3's sigmoid router is a different normalization.
        let err = reject(|v| v["scoring_func"] = "sigmoid".into());
        assert!(err.contains("sqrtsoftplus"), "{err}");

        // The head is untied in the checkpoint.
        let err = reject(|v| v["tie_word_embeddings"] = true.into());
        assert!(err.contains("untied head.weight"), "{err}");
    }

    /// `architectures: ["DeepseekV4ForCausalLM"]` alone (no `model_type`) must
    /// not fall through to the `DeepseekV3` prefix and build a V3 MLA graph.
    #[test]
    fn architectures_fallback_does_not_reach_deepseek_v3() {
        let mut v: serde_json::Value = serde_json::from_str(RELEASED).unwrap();
        v.as_object_mut().unwrap().remove("model_type");
        match ModelConfig::from_json(&v.to_string()).expect("parses via architectures") {
            ModelConfig::DeepSeekV4(_) => {}
            other => panic!("DeepseekV4ForCausalLM parsed as {other:?}"),
        }
    }
}

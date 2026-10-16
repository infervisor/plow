//! GLM-5.3-Flash configuration (`model_type: "glm5_next"`).
//!
//! glm5_next is **not** a `glm_moe_dsa` with a `text_config` wrapper. The
//! `GlmMoeDsa*` architecture prefix would otherwise capture it, and each
//! difference silently builds a plausible wrong model if it is defaulted away:
//!
//! * **Hybrid attention.** 34 KDA (linear attention, carried recurrent state)
//!   layers and 11 sparse-MLA layers, decided per layer by `layer_types`;
//!   `linear_attn_config` repeats the partition as **0-based** index lists
//!   (Kimi-K3's are 1-based).
//! * **NoPE MLA.** `mla_use_nope` with `qk_rope_head_dim = 0`: no RoPE table
//!   anywhere in the text tower, 256-wide content-only q/k.
//! * **Pooled indexer.** `index_kpool = 4` keys are compressed into one scored
//!   entry; `index_topk` counts tokens, so the selection is `index_topk /
//!   index_kpool` pools plus the tail.
//! * **mHC residuals** (`hc_mult` streams, Sinkhorn-mixed), mean exit.
//! * **Clamped SwiGLU** (`swiglu_limit`) on dense, routed and shared GLUs.
//! * **fp8 with bf16 islands.** `quantization_config` lives at the ROOT of the
//!   file and its `modules_to_not_convert` names every KDA projection, the mHC
//!   tensors, `kv_b_proj` and the indexer as bf16.
//!
//! No blanket defaults: like [`super::DeepSeekV4Config`], every field that
//! changes the graph is required.

use serde::Deserialize;

use super::ConfigError;

/// The outer `config.json` (`model_type: "glm5_next"`).
#[derive(Debug, Clone, Deserialize)]
pub struct Glm5NextConfig {
    pub text_config: Glm5NextTextConfig,
    /// `glm5_next_vision` tower (24 blocks). Carried so a builder refuses
    /// rather than silently compiling the text tower for image prompts.
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
    /// Root-level on the HF file, not under `text_config`.
    pub quantization_config: Glm5NextQuantizationConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Glm5NextQuantizationConfig {
    pub activation_scheme: String,
    pub fmt: String,
    pub quant_method: String,
    pub weight_block_size: [i64; 2],
    /// Module names kept bf16 inside the fp8 checkpoint.
    pub modules_to_not_convert: Vec<String>,
}

/// The text tower (`text_config.model_type: "glm5_next_text"`).
#[derive(Debug, Clone, Deserialize)]
pub struct Glm5NextTextConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub rms_norm_eps: f32,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,

    // --- sparse MLA, on the `deepseek_sparse_attention` layers ---
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    /// 0 on the released checkpoint (`mla_use_nope`).
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub mla_use_nope: bool,

    // --- layer partition ---
    /// One of `linear_attention` / `deepseek_sparse_attention` per layer.
    pub layer_types: Vec<String>,
    pub linear_attn_config: Glm5NextLinearAttnConfig,

    // --- mHC residuals ---
    pub mhc: bool,
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,

    // --- FFN ---
    pub first_k_dense_replace: u32,
    /// One of `dense` / `sparse` per layer; must agree with `first_k_dense_replace`.
    pub mlp_layer_types: Vec<String>,
    pub n_routed_experts: u32,
    pub n_shared_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: i64,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub n_group: u32,
    pub topk_group: u32,
    pub moe_router_dtype: String,
    pub swiglu_limit: f32,

    // --- lightning indexer ---
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    /// In TOKENS; the selection is `index_topk / index_kpool` pools.
    pub index_topk: u32,
    pub index_kpool: u32,
    pub index_kpool_compress: bool,
    pub index_kpool_always_select_tail: bool,
    pub indexer_types: Vec<String>,
    pub index_share_for_mtp_iteration: bool,

    pub num_nextn_predict_layers: u32,

    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

/// KDA geometry and the layer partition.
#[derive(Debug, Clone, Deserialize)]
pub struct Glm5NextLinearAttnConfig {
    pub num_heads: u32,
    pub head_dim: u32,
    pub short_conv_kernel_size: u32,
    pub gate_lower_bound: f32,
    /// **0-BASED** (Kimi-K3's lists are 1-based). Must equal the
    /// `linear_attention` positions of `layer_types`.
    pub kda_layers: Vec<u32>,
    /// **0-BASED**; the `deepseek_sparse_attention` positions of `layer_types`.
    pub full_attn_layers: Vec<u32>,
}

/// Which mixer a layer uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Glm5NextLayer {
    /// Kimi Delta Attention with a low-rank output gate.
    Kda,
    /// NoPE MLA with the pooled lightning indexer.
    SparseMla,
}

const LINEAR: &str = "linear_attention";
const SPARSE: &str = "deepseek_sparse_attention";

fn layer_kind(name: &str) -> Option<Glm5NextLayer> {
    match name {
        LINEAR => Some(Glm5NextLayer::Kda),
        SPARSE => Some(Glm5NextLayer::SparseMla),
        _ => None,
    }
}

impl Glm5NextTextConfig {
    /// Full per-head q/k dim on the sparse-MLA layers.
    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Per-layer mixer, 0-based. `validate()` has already checked that every
    /// `layer_types` entry is one of the two names.
    pub fn attn_kinds(&self) -> Vec<Glm5NextLayer> {
        self.layer_types
            .iter()
            .map(|t| layer_kind(t).expect("validated layer_types"))
            .collect()
    }

    pub fn layer_is_dense(&self, layer: u32) -> bool {
        self.mlp_layer_types[layer as usize] == "dense"
    }

    /// Pools scored by the indexer per query (`index_topk` is in tokens).
    pub fn index_select_pools(&self) -> u32 {
        self.index_topk / self.index_kpool
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |message: String| Err(ConfigError::Unsupported(message));
        let n = self.num_hidden_layers;

        if n == 0
            || self.hidden_size <= 0
            || self.vocab_size <= 0
            || self.intermediate_size <= 0
            || self.moe_intermediate_size <= 0
            || self.num_attention_heads == 0
        {
            return bad("glm5_next dimensions must be positive".into());
        }

        // --- layer partition ---
        if self.layer_types.len() != n as usize {
            return bad(format!(
                "glm5_next layer_types has {} entries for num_hidden_layers={n}",
                self.layer_types.len()
            ));
        }
        let mut sparse = Vec::new();
        let mut linear = Vec::new();
        for (l, t) in self.layer_types.iter().enumerate() {
            match layer_kind(t) {
                Some(Glm5NextLayer::Kda) => linear.push(l as u32),
                Some(Glm5NextLayer::SparseMla) => sparse.push(l as u32),
                None => {
                    return bad(format!(
                        "glm5_next layer_types[{l}] = {t:?} is neither {LINEAR:?} nor {SPARSE:?}"
                    ))
                }
            }
        }
        let lac = &self.linear_attn_config;
        let mut full = lac.full_attn_layers.clone();
        full.sort_unstable();
        if full != sparse {
            return bad(format!(
                "glm5_next linear_attn_config.full_attn_layers (0-based) {:?} does not match \
                 the {SPARSE} positions {sparse:?} of layer_types",
                lac.full_attn_layers
            ));
        }
        let mut kda = lac.kda_layers.clone();
        kda.sort_unstable();
        if kda != linear {
            return bad(format!(
                "glm5_next linear_attn_config.kda_layers (0-based) {:?} does not match the \
                 {LINEAR} positions {linear:?} of layer_types",
                lac.kda_layers
            ));
        }
        if lac.num_heads == 0 || lac.head_dim == 0 || lac.short_conv_kernel_size == 0 {
            return bad("glm5_next linear_attn_config dimensions must be positive".into());
        }

        // --- sparse MLA ---
        if self.mla_use_nope && self.qk_rope_head_dim != 0 {
            return bad(format!(
                "glm5_next mla_use_nope=true requires qk_rope_head_dim=0, got {}",
                self.qk_rope_head_dim
            ));
        }
        if !self.mla_use_nope && self.qk_rope_head_dim == 0 {
            return bad(
                "glm5_next mla_use_nope=false with qk_rope_head_dim=0 has no RoPE part".into(),
            );
        }
        if self.q_lora_rank == 0
            || self.kv_lora_rank == 0
            || self.qk_nope_head_dim == 0
            || self.v_head_dim == 0
        {
            return bad("glm5_next MLA dimensions must be positive".into());
        }
        if self.attention_bias {
            return bad("glm5_next attention_bias=true is not supported".into());
        }

        // --- mHC ---
        if self.mhc && (self.hc_mult == 0 || self.hc_sinkhorn_iters == 0 || self.hc_eps <= 0.0) {
            return bad(format!(
                "glm5_next mhc=true needs hc_mult>=1, hc_sinkhorn_iters>0 and hc_eps>0; got {}, {}, {}",
                self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
            ));
        }

        // --- FFN ---
        if self.mlp_layer_types.len() != n as usize {
            return bad(format!(
                "glm5_next mlp_layer_types has {} entries for num_hidden_layers={n}",
                self.mlp_layer_types.len()
            ));
        }
        if self.first_k_dense_replace > n {
            return bad(format!(
                "glm5_next first_k_dense_replace={} exceeds num_hidden_layers={n}",
                self.first_k_dense_replace
            ));
        }
        for (l, t) in self.mlp_layer_types.iter().enumerate() {
            let expected = if (l as u32) < self.first_k_dense_replace {
                "dense"
            } else {
                "sparse"
            };
            if t != expected {
                return bad(format!(
                    "glm5_next mlp_layer_types[{l}] = {t:?} but first_k_dense_replace={} makes \
                     it {expected:?}",
                    self.first_k_dense_replace
                ));
            }
        }
        if self.n_routed_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.n_routed_experts
        {
            return bad(format!(
                "glm5_next invalid expert routing: {} of {} routed experts per token",
                self.num_experts_per_tok, self.n_routed_experts
            ));
        }
        if self.n_group == 0
            || self.topk_group == 0
            || self.topk_group > self.n_group
            || self.n_routed_experts % self.n_group != 0
        {
            return bad(format!(
                "glm5_next invalid expert groups: topk_group={} of n_group={} over {} experts",
                self.topk_group, self.n_group, self.n_routed_experts
            ));
        }
        if self.hidden_act != "silu"
            || self.scoring_func != "sigmoid"
            || self.topk_method != "noaux_tc"
            || self.moe_router_dtype != "float32"
        {
            return bad(format!(
                "glm5_next requires silu experts with a sigmoid/noaux_tc float32 router; got \
                 hidden_act={:?}, scoring_func={:?}, topk_method={:?}, moe_router_dtype={:?}",
                self.hidden_act, self.scoring_func, self.topk_method, self.moe_router_dtype
            ));
        }
        if !self.norm_topk_prob || self.routed_scaling_factor <= 0.0 || self.swiglu_limit <= 0.0 {
            return bad(
                "glm5_next requires normalized, positively scaled top-k weights and a positive \
                 swiglu_limit"
                    .into(),
            );
        }

        // --- indexer ---
        if self.index_n_heads == 0 || self.index_head_dim == 0 || self.index_topk == 0 {
            return bad("glm5_next indexer dimensions must be positive".into());
        }
        if self.index_kpool == 0 {
            return bad("glm5_next index_kpool must be >= 1".into());
        }
        if self.index_topk % self.index_kpool != 0 {
            return bad(format!(
                "glm5_next index_topk={} must be a multiple of index_kpool={}",
                self.index_topk, self.index_kpool
            ));
        }
        if self.indexer_types.len() != n as usize {
            return bad(format!(
                "glm5_next indexer_types has {} entries for num_hidden_layers={n}",
                self.indexer_types.len()
            ));
        }
        if let Some((l, t)) = self
            .indexer_types
            .iter()
            .enumerate()
            .find(|(_, t)| t.as_str() != "full")
        {
            return bad(format!(
                "glm5_next indexer_types[{l}] = {t:?}; only \"full\" is modeled"
            ));
        }

        // --- MTP ---
        if self.num_nextn_predict_layers > 1 {
            return bad(format!(
                "glm5_next num_nextn_predict_layers={} > 1 is not modeled",
                self.num_nextn_predict_layers
            ));
        }

        if self.torch_dtype.as_deref() != Some("bfloat16") {
            return bad("glm5_next requires dtype=bfloat16".into());
        }
        if self.tie_word_embeddings {
            return bad(
                "glm5_next ships an untied lm_head; tie_word_embeddings=true would discard it"
                    .into(),
            );
        }
        Ok(())
    }
}

impl Glm5NextConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.text_config.validate()?;
        let q = &self.quantization_config;
        if q.quant_method != "fp8"
            || q.fmt != "e4m3"
            || q.activation_scheme != "dynamic"
            || q.weight_block_size != [128, 128]
        {
            return Err(ConfigError::Unsupported(format!(
                "glm5_next requires dynamic e4m3 block-FP8 [128,128]; got method={:?}, fmt={:?}, \
                 activation={:?}, block={:?}",
                q.quant_method, q.fmt, q.activation_scheme, q.weight_block_size
            )));
        }
        if q.modules_to_not_convert.is_empty() {
            return Err(ConfigError::Unsupported(
                "glm5_next quantization_config.modules_to_not_convert is empty; the KDA, mHC, \
                 kv_b_proj and indexer tensors are bf16 in the checkpoint"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The nn-graph refusal. GLM-5.3-Flash is served by the devblob emitter
    /// `mla::glm53_emit_full` (`crates/devgen/src/mla.rs`), which never goes
    /// through this crate; the packet path has no lowering and must not fall
    /// back to the `glm_moe_dsa` MLA builder.
    pub fn unimplemented(&self) -> String {
        let t = &self.text_config;
        let lac = &t.linear_attn_config;
        let kinds = t.attn_kinds();
        let kda = kinds.iter().filter(|k| **k == Glm5NextLayer::Kda).count();
        let sparse: Vec<u32> = (0..t.num_hidden_layers)
            .filter(|l| kinds[*l as usize] == Glm5NextLayer::SparseMla)
            .collect();
        let dense = (0..t.num_hidden_layers)
            .filter(|l| t.layer_is_dense(*l))
            .count();
        let vision = if self.vision_config.is_some() {
            "; and the glm5_next_vision tower"
        } else {
            ""
        };
        format!(
            "glm5_next nn-graph lowering is not implemented (config parses and validates; \
             refusing a glm_moe_dsa / DeepSeek-V3 MLA fallback). GLM-5.3-Flash is served by the \
             devblob emitter `mla::glm53_emit_full` (crates/devgen/src/mla.rs), which does not \
             use nn-graph, so plowc's `--model` packet path fails closed here. Missing in \
             nn-graph: {kda} KDA layers ({kh}x{kd} heads, conv k={conv}, low-rank output gate, \
             gate lower bound {glb}); {ns} NoPE sparse-MLA layers at {sparse:?} (q_lora {ql}, \
             kv_lora {kvl}, {nh}x{nope}+{rope} q/k, v {vd}) with the kpool-{kpool} lightning \
             indexer ({ih}x{ihd} heads, top-{itk} tokens = {pools} pools, tail {tail}); mHC \
             (hc_mult={mult}, {sink} Sinkhorn iterations, eps {eps:e}, mean exit); {dense} dense \
             (I={inter}) + {moe} MoE layers ({experts} routed top-{topk} + {shared} shared, \
             I={moe_i}, {scoring}/{method} {rdtype} router, scale {rsf}, clamped SwiGLU limit \
             {lim}); fp8 {fmt} [{b0},{b1}] projections with {ntc} modules_to_not_convert kept \
             bf16; {mtp} MTP layer (index shared across the MTP iteration = {share}){vision}",
            kh = lac.num_heads,
            kd = lac.head_dim,
            conv = lac.short_conv_kernel_size,
            glb = lac.gate_lower_bound,
            ns = sparse.len(),
            ql = t.q_lora_rank,
            kvl = t.kv_lora_rank,
            nh = t.num_attention_heads,
            nope = t.qk_nope_head_dim,
            rope = t.qk_rope_head_dim,
            vd = t.v_head_dim,
            kpool = t.index_kpool,
            ih = t.index_n_heads,
            ihd = t.index_head_dim,
            itk = t.index_topk,
            pools = t.index_select_pools(),
            tail = t.index_kpool_always_select_tail,
            mult = t.hc_mult,
            sink = t.hc_sinkhorn_iters,
            eps = t.hc_eps,
            inter = t.intermediate_size,
            moe = t.num_hidden_layers as usize - dense,
            experts = t.n_routed_experts,
            topk = t.num_experts_per_tok,
            shared = t.n_shared_experts,
            moe_i = t.moe_intermediate_size,
            scoring = t.scoring_func,
            method = t.topk_method,
            rdtype = t.moe_router_dtype,
            rsf = t.routed_scaling_factor,
            lim = t.swiglu_limit,
            fmt = self.quantization_config.fmt,
            b0 = self.quantization_config.weight_block_size[0],
            b1 = self.quantization_config.weight_block_size[1],
            ntc = self.quantization_config.modules_to_not_convert.len(),
            mtp = t.num_nextn_predict_layers,
            share = t.index_share_for_mtp_iteration,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::config::ModelConfig;

    /// zai-org/GLM-5.3-Flash `config.json`: every scalar verbatim, the 45-entry
    /// per-layer lists intact, `modules_to_not_convert` cut from 1509 entries
    /// to the roots plus layer 0 (KDA) and layer 3 (sparse MLA).
    pub(crate) const RELEASED: &str = r#"{
  "architectures":["Glm5NextForConditionalGeneration"],
  "model_type":"glm5_next",
  "transformers_version":"5.16.0",
  "image_token_id":154854,
  "video_token_id":154855,
  "tie_word_embeddings":false,
  "text_config":{
    "model_type":"glm5_next_text",
    "dtype":"bfloat16",
    "vocab_size":154880,
    "hidden_size":4096,
    "intermediate_size":12288,
    "num_hidden_layers":45,
    "num_attention_heads":64,
    "rms_norm_eps":1e-05,
    "hidden_act":"silu",
    "attention_bias":false,
    "tie_word_embeddings":false,
    "q_lora_rank":1536,
    "kv_lora_rank":512,
    "qk_nope_head_dim":256,
    "qk_rope_head_dim":0,
    "v_head_dim":256,
    "mla_use_nope":true,
    "mhc":true,
    "hc_mult":4,
    "hc_sinkhorn_iters":20,
    "hc_eps":1e-06,
    "first_k_dense_replace":3,
    "n_routed_experts":288,
    "n_shared_experts":1,
    "num_experts_per_tok":8,
    "moe_intermediate_size":2048,
    "scoring_func":"sigmoid",
    "topk_method":"noaux_tc",
    "norm_topk_prob":true,
    "routed_scaling_factor":2.5,
    "n_group":1,
    "topk_group":1,
    "moe_router_dtype":"float32",
    "swiglu_limit":10.0,
    "index_n_heads":32,
    "index_head_dim":128,
    "index_topk":2048,
    "index_kpool":4,
    "index_kpool_compress":true,
    "index_kpool_always_select_tail":true,
    "index_share_for_mtp_iteration":true,
    "num_nextn_predict_layers":1,
    "layer_types":[
      "linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention",
      "linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention",
      "linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention",
      "deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention",
      "linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention",
      "linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention",
      "linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention",
      "deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention",
      "linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention"],
    "linear_attn_config":{"num_heads":64,"head_dim":128,"short_conv_kernel_size":4,"gate_lower_bound":-5.0,
      "full_attn_layers":[3,7,11,15,19,23,27,31,35,39,43],
      "kda_layers":[
        0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,
        22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44]},
    "mlp_layer_types":[
      "dense","dense","dense","sparse","sparse","sparse","sparse","sparse","sparse",
      "sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse",
      "sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse",
      "sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse",
      "sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse"],
    "indexer_types":[
      "full","full","full","full","full","full","full","full","full",
      "full","full","full","full","full","full","full","full","full",
      "full","full","full","full","full","full","full","full","full",
      "full","full","full","full","full","full","full","full","full",
      "full","full","full","full","full","full","full","full","full"]
  },
  "vision_config":{"model_type":"glm5_next_vision","depth":24,"hidden_size":1024,"num_heads":16,
    "intermediate_size":4096,"out_hidden_size":4096,"patch_size":14,"spatial_merge_size":2,
    "temporal_patch_size":2,"image_size":448,"swiglu_limit":10.0},
  "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","weight_block_size":[128,128],
    "modules_to_not_convert":[
      "attn_mha","attn_mqa","dt_bias","hyper_connection","lm_head","mapping_proj","model.embed_tokens",
      "model.layers.0.hc_attn_base","model.layers.0.hc_attn_fn","model.layers.0.hc_attn_scale",
      "model.layers.0.hc_ffn_base","model.layers.0.hc_ffn_fn","model.layers.0.hc_ffn_scale",
      "model.layers.0.input_layernorm","model.layers.0.post_attention_layernorm",
      "model.layers.0.self_attn.A_log","model.layers.0.self_attn.b_proj","model.layers.0.self_attn.dt_bias",
      "model.layers.0.self_attn.f_a_proj","model.layers.0.self_attn.f_b_proj",
      "model.layers.0.self_attn.g_a_proj","model.layers.0.self_attn.g_b_proj",
      "model.layers.0.self_attn.k_conv1d","model.layers.0.self_attn.k_proj",
      "model.layers.0.self_attn.o_norm","model.layers.0.self_attn.o_proj",
      "model.layers.0.self_attn.q_conv1d","model.layers.0.self_attn.q_proj",
      "model.layers.0.self_attn.v_conv1d","model.layers.0.self_attn.v_proj",
      "model.layers.3.hc_attn_base","model.layers.3.hc_attn_fn","model.layers.3.hc_attn_scale",
      "model.layers.3.hc_ffn_base","model.layers.3.hc_ffn_fn","model.layers.3.hc_ffn_scale",
      "model.layers.3.input_layernorm","model.layers.3.post_attention_layernorm",
      "model.layers.3.mlp.gate","model.layers.3.mlp.gate.e_score_correction_bias",
      "model.layers.3.self_attn.indexer.index_kpool_compress_ape",
      "model.layers.3.self_attn.indexer.index_kpool_compress_gate",
      "model.layers.3.self_attn.indexer.k_norm","model.layers.3.self_attn.indexer.k_norm.bias",
      "model.layers.3.self_attn.indexer.weights_proj","model.layers.3.self_attn.indexer.wk",
      "model.layers.3.self_attn.indexer.wq_b","model.layers.3.self_attn.kv_a_layernorm",
      "model.layers.3.self_attn.kv_b_proj","model.layers.3.self_attn.q_a_layernorm"]}
}"#;

    fn released() -> Glm5NextConfig {
        match ModelConfig::from_json(RELEASED).expect("released config must parse and validate") {
            ModelConfig::Glm5Next(cfg) => cfg,
            other => panic!("released GLM-5.3-Flash config parsed as {other:?}"),
        }
    }

    #[test]
    fn released_config_parses_with_its_released_values() {
        let c = released();
        let t = &c.text_config;
        assert!(c.vision_config.is_some());
        assert_eq!(c.quantization_config.modules_to_not_convert.len(), 50);
        assert_eq!(t.vocab_size, 154_880);
        assert_eq!(t.hidden_size, 4096);
        assert_eq!(t.num_hidden_layers, 45);
        assert_eq!(t.num_attention_heads, 64);
        assert_eq!((t.q_lora_rank, t.kv_lora_rank), (1536, 512));
        assert_eq!(
            (t.qk_nope_head_dim, t.qk_rope_head_dim, t.v_head_dim),
            (256, 0, 256)
        );
        assert_eq!(t.qk_head_dim(), 256);
        assert!(t.mla_use_nope);
        assert_eq!(
            (
                t.linear_attn_config.num_heads,
                t.linear_attn_config.head_dim
            ),
            (64, 128)
        );
        assert_eq!(t.linear_attn_config.short_conv_kernel_size, 4);
        assert_eq!(t.linear_attn_config.gate_lower_bound, -5.0);
        assert!(t.mhc);
        assert_eq!((t.hc_mult, t.hc_sinkhorn_iters), (4, 20));
        assert_eq!(t.first_k_dense_replace, 3);
        assert_eq!(
            (
                t.n_routed_experts,
                t.num_experts_per_tok,
                t.n_shared_experts
            ),
            (288, 8, 1)
        );
        assert_eq!(
            (t.moe_intermediate_size, t.intermediate_size),
            (2048, 12288)
        );
        assert_eq!(t.routed_scaling_factor, 2.5);
        assert_eq!(t.swiglu_limit, 10.0);
        assert_eq!(
            (
                t.index_n_heads,
                t.index_head_dim,
                t.index_topk,
                t.index_kpool
            ),
            (32, 128, 2048, 4)
        );
        assert_eq!(t.index_select_pools(), 512);
        assert_eq!(t.num_nextn_predict_layers, 1);
        assert_eq!(t.rms_norm_eps, 1e-5);
    }

    #[test]
    fn layer_partition_matches_the_released_schedule() {
        let t = released().text_config;
        let kinds = t.attn_kinds();
        assert_eq!(kinds.len(), 45);
        for (l, kind) in kinds.iter().enumerate() {
            let expected = if l % 4 == 3 {
                Glm5NextLayer::SparseMla
            } else {
                Glm5NextLayer::Kda
            };
            assert_eq!(*kind, expected, "layer {l}");
        }
        assert_eq!(
            kinds.iter().filter(|k| **k == Glm5NextLayer::Kda).count(),
            34
        );
        assert!(t.layer_is_dense(0) && t.layer_is_dense(2));
        assert!(!t.layer_is_dense(3) && !t.layer_is_dense(44));
    }

    /// The bare text tower, as `build_text_generation_from_config_json_at`
    /// hands it over (root `quantization_config` lifted in, no `vision_config`).
    #[test]
    fn bare_text_config_wraps_without_a_vision_tower() {
        let v: serde_json::Value = serde_json::from_str(RELEASED).unwrap();
        let mut text = v["text_config"].clone();
        text["quantization_config"] = v["quantization_config"].clone();
        match ModelConfig::from_json(&text.to_string()).expect("glm5_next_text parses") {
            ModelConfig::Glm5Next(c) => {
                assert!(c.vision_config.is_none());
                assert_eq!(c.text_config.num_hidden_layers, 45);
            }
            other => panic!("glm5_next_text parsed as {other:?}"),
        }
    }

    #[test]
    fn refusal_names_the_devblob_emitter_and_every_block() {
        let msg = released().unimplemented();
        for piece in [
            "glm5_next nn-graph lowering is not implemented",
            "mla::glm53_emit_full",
            "fails closed",
            "34 KDA layers (64x128 heads, conv k=4",
            "11 NoPE sparse-MLA layers at [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]",
            "kpool-4 lightning indexer (32x128 heads, top-2048 tokens = 512 pools",
            "mHC (hc_mult=4, 20 Sinkhorn iterations",
            "3 dense (I=12288) + 42 MoE layers (288 routed top-8 + 1 shared",
            "sigmoid/noaux_tc float32 router",
            "clamped SwiGLU limit 10",
            "fp8 e4m3 [128,128] projections with 50 modules_to_not_convert",
            "1 MTP layer",
            "glm5_next_vision tower",
        ] {
            assert!(msg.contains(piece), "missing {piece:?} in: {msg}");
        }
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
        // full_attn_layers must repeat layer_types exactly.
        let err = reject(|v| {
            v["text_config"]["linear_attn_config"]["full_attn_layers"]
                .as_array_mut()
                .unwrap()
                .pop();
        });
        assert!(err.contains("full_attn_layers (0-based)"), "{err}");

        // A 1-based list (Kimi-K3 convention) is off by one everywhere.
        let err = reject(|v| {
            v["text_config"]["linear_attn_config"]["kda_layers"] = serde_json::json!([
                1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29,
                30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45
            ]);
        });
        assert!(err.contains("kda_layers (0-based)"), "{err}");

        // NoPE means no RoPE part.
        let err = reject(|v| v["text_config"]["qk_rope_head_dim"] = 64.into());
        assert!(
            err.contains("mla_use_nope=true requires qk_rope_head_dim=0"),
            "{err}"
        );

        // Dense/sparse schedule must agree with first_k_dense_replace.
        let err = reject(|v| v["text_config"]["first_k_dense_replace"] = 2.into());
        assert!(err.contains("mlp_layer_types[2] = \"dense\""), "{err}");

        // Whole pools only.
        let err = reject(|v| v["text_config"]["index_topk"] = 2050.into());
        assert!(
            err.contains("index_topk=2050 must be a multiple of index_kpool=4"),
            "{err}"
        );

        // V3's sqrtsoftplus router is a different normalization.
        let err = reject(|v| v["text_config"]["scoring_func"] = "sqrtsoftplus".into());
        assert!(err.contains("sigmoid/noaux_tc float32 router"), "{err}");

        // The root quantization block decides every projection's dtype.
        let err = reject(|v| {
            v["quantization_config"]["weight_block_size"] = serde_json::json!([128, 64])
        });
        assert!(err.contains("block-FP8 [128,128]"), "{err}");

        let err = reject(|v| v["text_config"]["num_nextn_predict_layers"] = 2.into());
        assert!(err.contains("num_nextn_predict_layers=2 > 1"), "{err}");

        let err = reject(|v| v["text_config"]["hc_mult"] = 0.into());
        assert!(err.contains("mhc=true needs hc_mult>=1"), "{err}");
    }

    /// `architectures: ["Glm5NextForConditionalGeneration"]` alone must not be
    /// captured by the `Glm` prefix and build a glm_moe_dsa graph.
    #[test]
    fn architectures_fallback_does_not_reach_glm_moe_dsa() {
        let mut v: serde_json::Value = serde_json::from_str(RELEASED).unwrap();
        v.as_object_mut().unwrap().remove("model_type");
        match ModelConfig::from_json(&v.to_string()).expect("parses via architectures") {
            ModelConfig::Glm5Next(_) => {}
            other => panic!("Glm5NextForConditionalGeneration parsed as {other:?}"),
        }
    }
}

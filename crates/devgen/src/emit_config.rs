//! Unified emit-time configuration for the plow compiler.
//!
//! Every field has a clap `env` attribute so existing shell scripts that set env vars
//! continue to work. CLI args (via `plowc --flatten`) take precedence over env.
//!
//! # Migration
//!
//! Call sites move from:
//! ```ignore
//! let fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1");
//! ```
//! to:
//! ```ignore
//! let fp8 = cfg.fp8;  // cfg: &EmitConfig
//! ```

use clap::Args;

/// Emit-time configuration — controls what `devgen` and `packet::DevBuild` produce.
///
/// Constructed by `plowc` via `#[command(flatten)]` and threaded through the emit
/// pipeline as `&EmitConfig`. The struct is the single source of truth for every
/// compile-time knob; `std::env::var` calls in `devgen` are being migrated here.
#[derive(Args, Debug, Clone)]
#[command(next_help_heading = "Emit knobs")]
pub struct EmitConfig {
    // ──────────────────────────────────────────────────────────────────────────
    // Precision
    // ──────────────────────────────────────────────────────────────────────────
    /// Enable fp8 weight encoding. On dense families this is w8a16 (sm_120) or
    /// triggers a refusal pointing at --w8a8 (gfx950). On MLA+MoE families it
    /// enables block-fp8 expert arms.
    #[arg(long, env = "PLOW_FP8", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fp8: bool,

    /// fp8 weights + fp8 activations (the w8a8 profile). Mutually exclusive
    /// with --w8a16.
    #[arg(long, env = "PLOW_W8A8", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub w8a8: bool,

    /// fp8 weights, bf16 activations (w8a16 profile). Mutually exclusive with
    /// --w8a8.
    #[arg(long, env = "PLOW_W8A16", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub w8a16: bool,

    /// MXFP4 (A4W4) encoding — both operands are 4-bit with E8M0 microscales.
    #[arg(long, env = "PLOW_MXFP4", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub mxfp4: bool,

    /// e4m3 KV cache (halves KV bytes). Lossy — greedy diverges after ~21
    /// tokens.
    #[arg(long, env = "PLOW_FP8_KV", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fp8_kv: bool,

    /// Mixed fp8 KV: restrict e4m3 cache to full-attention (hd512) layers only.
    /// Requires --fp8-kv.
    #[arg(long, env = "PLOW_FP8_KV_FULL", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fp8_kv_full: bool,

    /// Emit an e4m3 tied embed/lm_head (rtx-19). Requires the fp8 twin to
    /// include the embed/lm_head tensor.
    #[arg(long, env = "PLOW_FP8_HEAD", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fp8_head: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // Scheduling / segmentation
    // ──────────────────────────────────────────────────────────────────────────
    /// Single-segment programs. Required for sm_120 prefill interpreter.
    /// WARNING: do NOT set on gfx950 — silently breaks AMD assets.
    #[arg(long, env = "PLOW_UNISEG", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub uniseg: bool,

    /// Batched decode dispatch width (sequences per launch).
    #[arg(long = "emit-decode-batch", env = "PLOW_DECODE_BATCH", default_value_t = 1)]
    pub decode_batch: u32,

    /// Largest prefill chunk rows (power of two, ≤ 8192). Caps the bucket
    /// ladder and the runtime PLOW_PF_INTERLEAVE ceiling.
    #[arg(long = "emit-max-chunk", env = "PLOW_MAX_CHUNK")]
    pub max_chunk: Option<u32>,

    /// Single block or layer range to emit (e.g. "3" or "0..5"). Also read
    /// from PLOW_BLOCK env.
    #[arg(long = "emit-block", env = "PLOW_BLOCK")]
    pub emit_block: Option<String>,

    /// One segment per op (host-side AQL chaining instead of batched).
    #[arg(long, env = "PLOW_SEG_PER_OP", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub seg_per_op: bool,

    /// Re-slice GEMM segments so both occ-2 blocks/SM get work.
    #[arg(long, env = "PLOW_SEG_CLASS_SLICE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub seg_class_slice: bool,

    /// Emit S·n_cu decode slices for Gemv packets (finer work-stealing).
    #[arg(long, env = "PLOW_GEMV_SPLIT", default_value_t = 1)]
    pub gemv_split: u32,

    /// AMD: emit prefill (tiled) opcodes into the decode bucket.
    #[arg(long, env = "PLOW_DECODE_TILED", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub decode_tiled: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // Fusion (generic, cross-model)
    // ──────────────────────────────────────────────────────────────────────────
    /// Fold greedy argmax into the lm_head GEMV epilogue.
    #[arg(long, env = "PLOW_FUSE_ARGMAX", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fuse_argmax: bool,

    /// Revert fused QKV to split-3 path (A/B control).
    #[arg(long, env = "PLOW_NO_FUSE_QKV", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub no_fuse_qkv: bool,

    /// Fused Q|K|V, per-channel fp8.
    #[arg(long, env = "PLOW_FUSE_QKV_FP8", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fuse_qkv_fp8: bool,

    /// Disable norm+residual+norm fusion.
    #[arg(long, env = "PLOW_NO_FUSE_NRN", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub no_fuse_nrn: bool,

    /// Fuse head-norm + reduce.
    #[arg(long, env = "PLOW_FUSE_HNR", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fuse_hnr: bool,

    /// Fuse merge fold.
    #[arg(long, env = "PLOW_FUSE_MERGE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fuse_merge: bool,

    /// Head-number split (3*nhn <= n_cu).
    #[arg(long, env = "PLOW_HN_SPLIT", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub hn_split: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // Attention / flash-decode geometry
    // ──────────────────────────────────────────────────────────────────────────
    /// AMD flash-decode GQA fusion factor on full-attention layers.
    #[arg(long, env = "PLOW_FA_GF_FULL")]
    pub fa_gf_full: Option<u32>,

    /// Flash merge dsplit (diagnostic, measured dead).
    #[arg(long, env = "PLOW_FLASH_MERGE_DSPLIT", hide = true)]
    pub flash_merge_dsplit: Option<u32>,

    /// Scale the CU-fill target for flash-decode nsplit.
    #[arg(long, env = "PLOW_NS_MUL")]
    pub ns_mul: Option<u32>,

    /// Pin nsplit absolutely.
    #[arg(long, env = "PLOW_NS_ABS")]
    pub ns_abs: Option<u32>,

    /// Pin nsplit for full-attention layers only.
    #[arg(long, env = "PLOW_NS_FULL_ABS")]
    pub ns_full_abs: Option<u32>,

    /// Prefill bucket ladder derivation: "wave" for SM-count-derived rungs.
    /// NVIDIA-only.
    #[arg(long = "pf-ladder", env = "PLOW_PF_LADDER")]
    pub pf_ladder: Option<String>,

    /// Extra prefill ladder rungs, comma-separated (T32: e.g. "640,1152,2176,4224"
    /// swallows the chat template's +14-row overhang in one chunk instead of a
    /// second full-model pass). Rungs above the chunk cap are filtered.
    #[arg(long = "pf-ladder-append", env = "PLOW_PF_LADDER_APPEND")]
    pub pf_ladder_append: Option<String>,

    /// Force prefill lm_head onto M=1 GEMV arm vs tiled. "1"/"0" to force.
    #[arg(long, env = "PLOW_PF_GEMV_HEAD")]
    pub pf_gemv_head: Option<String>,

    // ──────────────────────────────────────────────────────────────────────────
    // Placement / L2 / counters
    // ──────────────────────────────────────────────────────────────────────────
    /// L2-domain packet grouping (compiler half of physical-SM locality).
    #[arg(long, env = "PLOW_L2_PLACE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub l2_place: bool,

    /// Keep fine counter gates instead of collapsing to coarse.
    #[arg(long, env = "PLOW_FINE_FORCE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fine_force: bool,

    /// Cap XReduce participant CUs.
    #[arg(long, env = "PLOW_XR_CUS")]
    pub xr_cus: Option<u32>,

    /// Disable all XReduce collectives (diagnostic — numerically wrong).
    #[arg(long, env = "PLOW_NO_XREDUCE", default_value_t = false, hide = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub no_xreduce: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // MoE (Gemma MoE family)
    // ──────────────────────────────────────────────────────────────────────────
    /// MoE prefill control. "0" to disable, unset = auto (on for MoE bf16).
    #[arg(long, env = "PLOW_MOE_PREFILL")]
    pub moe_prefill: Option<String>,

    /// Disable split router, serialize score GEMV on one CTA.
    #[arg(long, env = "PLOW_GEMMA_MOE_ROUTER_FUSED", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub gemma_moe_router_fused: bool,

    /// CTA count for the split router score GEMV.
    #[arg(long, env = "PLOW_GEMMA_MOE_ROUTER_BLOCKS")]
    pub gemma_moe_router_blocks: Option<u32>,

    /// Exact MoeRouterGemmaScore op instead of ScoreFast.
    #[arg(long, env = "PLOW_GEMMA_MOE_ROUTER_EXACT", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub gemma_moe_router_exact: bool,

    /// Fuse MoE-combine residual/norm tail (B=1 only, reorders summation).
    #[arg(long, env = "PLOW_GEMMA_MOE_TAIL_FUSE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub gemma_moe_tail_fuse: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // K3 model family
    // ──────────────────────────────────────────────────────────────────────────
    /// Emit the full K3 model (all layers). Default is capability-report only.
    #[arg(long, env = "K3_FULL", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_full: bool,

    /// Fuse MLA q/kv/k_rope/gate A-projection into one GemvQkvg (decode-only).
    #[arg(long, env = "PLOW_K3_FUSE_A", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_fuse_a: bool,

    /// Pin FlashMLA-decode nsplit for K3 MLA layers.
    #[arg(long, env = "PLOW_K3_NS")]
    pub k3_ns: Option<u32>,

    /// Layers to emit: "all" (default), a number N (first N layers), or
    /// "single:L" (one specific layer). Replaces K3_FULL + K3_NLAYERS.
    #[arg(long = "k3-layers", env = "PLOW_K3_LAYERS", default_value = "all")]
    pub k3_layers: String,

    /// K3 prefill bucket control. "0" disables prefill.
    #[arg(long, env = "K3_PREFILL")]
    pub k3_prefill: Option<String>,

    // ──────────────────────────────────────────────────────────────────────────
    // GLM model family
    // ──────────────────────────────────────────────────────────────────────────
    /// GLM sparse-attention arm control. "0" forces dense, unset = auto
    /// (on above ctx crossover).
    #[arg(long, env = "PLOW_GLM_DSA")]
    pub glm_dsa: Option<String>,

    /// Pin the MLA head-fusion factor.
    #[arg(long, env = "PLOW_GLM_GF")]
    pub glm_gf: Option<u32>,

    /// Pin MLA flash-decode nsplit.
    #[arg(long, env = "PLOW_GLM_NS")]
    pub glm_ns: Option<u32>,

    /// Vocab-column-parallel lm_head.
    #[arg(long, env = "GLM_SHARD_HEAD", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_shard_head: bool,

    /// Co-resident shared expert mode (0/1/2).
    #[arg(long, env = "GLM_MOE_CORESIDENT")]
    pub glm_moe_coresident: Option<u32>,

    /// CUs for shared expert.
    #[arg(long, env = "GLM_SHARED_CUS")]
    pub glm_shared_cus: Option<u32>,

    /// Spine CU allocation (comma-separated or expression).
    #[arg(long, env = "GLM_SPINE_CUS")]
    pub glm_spine_cus: Option<String>,

    /// fp8 shared-expert linear projections.
    #[arg(long, env = "GLM_LINEAR_FP8", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_linear_fp8: bool,

    /// Split GLU path for fp8 linear.
    #[arg(long, env = "GLM_SHARED_GLU_SPLIT", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_shared_glu_split: bool,

    /// Layers to emit: "all" (default), a number N (first N layers), or
    /// "single:L" (one specific layer). Replaces GLM_FULL/GLM_NLAYERS/GLM_LAYER.
    #[arg(long = "glm-layers", env = "PLOW_GLM_LAYERS", default_value = "all")]
    pub glm_layers: String,

    /// MLA prefill ladder (e.g. "full:512,2048,4096,8192").
    #[arg(long, env = "PLOW_MLA_PREFILL")]
    pub mla_prefill: Option<String>,

    /// GLM expert-parallel mode.
    #[arg(long, env = "GLM_EP", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_ep: bool,

    /// GLM grouped MoE dispatch.
    #[arg(long, env = "GLM_GROUP", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_group: bool,

    /// GLM fuse block-1 residual+norm (opt-in, off by default).
    #[arg(long, env = "PLOW_GLM_FUSE_B1", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_fuse_b1: bool,

    /// GLM router off-shared dispatch (co-resident mode 2 only).
    #[arg(long, env = "GLM_ROUTER_OFF_SHARED", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_router_off_shared: bool,

    /// GLM use legacy (unfused) single-CU router.
    #[arg(long, env = "GLM_ROUTER_OLD", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_router_old: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // K3 model family (additional)
    // ──────────────────────────────────────────────────────────────────────────
    /// K3 fuse norm→GEMV. "1"=force all, "lat"=latency only, "q"=q-side only.
    #[arg(long, env = "PLOW_K3_FUSE_NGEMV")]
    pub k3_fuse_ngemv: Option<String>,

    /// K3 up-projection no-gather mode (diagnostic).
    #[arg(long, env = "PLOW_K3_UP_NOGATHER", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_up_nogather: bool,

    /// K3 up-projection gather-only mode (diagnostic).
    #[arg(long, env = "PLOW_K3_UP_GATHER_ONLY", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_up_gather_only: bool,

    /// K3 vocab-column-parallel lm_head.
    #[arg(long, env = "PLOW_K3_SHARD_HEAD", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_shard_head: bool,

    /// K3 batched decode uses per-sequence GEMV rows.
    #[arg(long, env = "PLOW_K3_SEQ_ROWS", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_seq_rows: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // GEMV geometry
    // ──────────────────────────────────────────────────────────────────────────
    /// AMD compile-time decode row-batch bucket.
    #[arg(long, env = "PLOW_GEMV_MM")]
    pub gemv_mm: Option<u32>,

    /// Wide-arm walk loop for AMD GEMV.
    #[arg(long, env = "PLOW_GEMV_WALK", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub gemv_walk: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // Tuning / diagnostics
    // ──────────────────────────────────────────────────────────────────────────
    /// Tuning database root directory.
    #[arg(long, env = "PLOW_TUNEDB")]
    pub tunedb: Option<String>,

    /// Print TUNEDUMP census line per resolved GEMV shape.
    #[arg(long, env = "PLOW_TUNE_DUMP", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub tune_dump: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // Diagnostic / never-ship (hidden from --help)
    // ──────────────────────────────────────────────────────────────────────────
    /// Splice opcodes out of the chain (garbage tokens — diagnostic only).
    #[arg(long, env = "PLOW_CHAIN_BYPASS", hide = true)]
    pub chain_bypass: Option<String>,

    /// Emit a model known to fail coverage checks (diagnostic only).
    #[arg(long, env = "PLOW_SKIP_COVERAGE", hide = true, default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub skip_coverage: bool,

    /// K3 bisection instrument (diagnostic only).
    #[arg(long, env = "PLOW_K3_ABLATE", hide = true)]
    pub k3_ablate: Option<String>,

    /// UniSeg diagnostic dump.
    #[arg(long, env = "PLOW_SEG_DUMP", hide = true, default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub seg_dump: bool,

    /// Placement report.
    #[arg(long, env = "PLOW_PLACE_REPORT", hide = true, default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub place_report: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // Paths (shared with runtime)
    // ──────────────────────────────────────────────────────────────────────────
    /// Plow repository root (for locating runtime/ sources).
    #[arg(long, env = "PLOW_ROOT")]
    pub plow_root: Option<String>,

    /// Checkpoint directory for weight binding.
    #[arg(long = "emit-checkpoint", env = "PLOW_CHECKPOINT")]
    pub checkpoint: Option<String>,
}

impl EmitConfig {
    /// Construct an EmitConfig by reading environment variables (legacy path).
    /// Mirrors what clap's `env` attribute does: each field reads its `PLOW_*` var.
    pub fn from_env() -> EmitConfig {
        let env_bool = |k: &str| std::env::var(k).ok().as_deref() == Some("1");
        let env_u32 = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<u32>().ok());
        let env_str = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
        EmitConfig {
            fp8: env_bool("PLOW_FP8"),
            w8a8: env_bool("PLOW_W8A8"),
            w8a16: env_bool("PLOW_W8A16"),
            mxfp4: env_bool("PLOW_MXFP4"),
            fp8_kv: env_bool("PLOW_FP8_KV") || env_bool("PLOW_KV_FP8"),
            fp8_kv_full: env_bool("PLOW_FP8_KV_FULL"),
            fp8_head: env_bool("PLOW_FP8_HEAD"),
            uniseg: env_bool("PLOW_UNISEG"),
            decode_batch: env_u32("PLOW_DECODE_BATCH").unwrap_or(1),
            max_chunk: env_u32("PLOW_MAX_CHUNK"),
            emit_block: env_str("PLOW_BLOCK"),
            seg_per_op: env_bool("PLOW_SEG_PER_OP"),
            seg_class_slice: env_bool("PLOW_SEG_CLASS_SLICE"),
            gemv_split: env_u32("PLOW_GEMV_SPLIT").unwrap_or(1),
            decode_tiled: env_bool("PLOW_DECODE_TILED"),
            fuse_argmax: env_bool("PLOW_FUSE_ARGMAX"),
            no_fuse_qkv: env_bool("PLOW_NO_FUSE_QKV"),
            fuse_qkv_fp8: env_bool("PLOW_FUSE_QKV_FP8"),
            no_fuse_nrn: env_bool("PLOW_NO_FUSE_NRN"),
            fuse_hnr: env_bool("PLOW_FUSE_HNR"),
            fuse_merge: env_bool("PLOW_FUSE_MERGE"),
            hn_split: env_bool("PLOW_HN_SPLIT"),
            fa_gf_full: env_u32("PLOW_FA_GF_FULL"),
            flash_merge_dsplit: env_u32("PLOW_FLASH_MERGE_DSPLIT"),
            ns_mul: env_u32("PLOW_NS_MUL"),
            ns_abs: env_u32("PLOW_NS_ABS"),
            ns_full_abs: env_u32("PLOW_NS_FULL_ABS"),
            pf_ladder: env_str("PLOW_PF_LADDER"),
            pf_ladder_append: env_str("PLOW_PF_LADDER_APPEND"),
            pf_gemv_head: env_str("PLOW_PF_GEMV_HEAD"),
            l2_place: env_bool("PLOW_L2_PLACE"),
            fine_force: env_bool("PLOW_FINE_FORCE"),
            xr_cus: env_u32("PLOW_XR_CUS"),
            no_xreduce: env_bool("PLOW_NO_XREDUCE"),
            moe_prefill: env_str("PLOW_MOE_PREFILL"),
            gemma_moe_router_fused: env_bool("PLOW_GEMMA_MOE_ROUTER_FUSED"),
            gemma_moe_router_blocks: env_u32("PLOW_GEMMA_MOE_ROUTER_BLOCKS"),
            gemma_moe_router_exact: env_bool("PLOW_GEMMA_MOE_ROUTER_EXACT"),
            gemma_moe_tail_fuse: env_bool("PLOW_GEMMA_MOE_TAIL_FUSE"),
            k3_full: env_bool("K3_FULL"),
            k3_fuse_a: env_bool("PLOW_K3_FUSE_A"),
            k3_ns: env_u32("PLOW_K3_NS"),
            k3_layers: env_str("PLOW_K3_LAYERS").unwrap_or_else(|| "all".into()),
            k3_prefill: env_str("K3_PREFILL"),
            glm_dsa: env_str("PLOW_GLM_DSA"),
            glm_gf: env_u32("PLOW_GLM_GF"),
            glm_ns: env_u32("PLOW_GLM_NS"),
            glm_shard_head: env_bool("GLM_SHARD_HEAD"),
            glm_moe_coresident: env_u32("GLM_MOE_CORESIDENT"),
            glm_shared_cus: env_u32("GLM_SHARED_CUS"),
            glm_spine_cus: env_str("GLM_SPINE_CUS"),
            glm_linear_fp8: env_bool("GLM_LINEAR_FP8"),
            glm_shared_glu_split: env_bool("GLM_SHARED_GLU_SPLIT"),
            glm_layers: env_str("PLOW_GLM_LAYERS").unwrap_or_else(|| {
                // Legacy synthesis: GLM_FULL=1 → "all" (with GLM_NLAYERS cap),
                // GLM_LAYER=L → "single:L", else "default" (caller decides).
                if std::env::var("GLM_FULL").ok().as_deref() == Some("1") {
                    match std::env::var("GLM_NLAYERS").ok().and_then(|s| s.parse::<u32>().ok()) {
                        Some(n) => n.to_string(),
                        None => "all".into(),
                    }
                } else if let Some(l) = std::env::var("GLM_LAYER").ok().and_then(|s| s.parse::<u32>().ok()) {
                    format!("single:{l}")
                } else {
                    "default".into()
                }
            }),
            mla_prefill: env_str("PLOW_MLA_PREFILL"),
            glm_ep: env_bool("GLM_EP"),
            glm_group: env_bool("GLM_GROUP"),
            glm_fuse_b1: env_bool("PLOW_GLM_FUSE_B1"),
            glm_router_off_shared: env_bool("GLM_ROUTER_OFF_SHARED"),
            glm_router_old: env_bool("GLM_ROUTER_OLD"),
            k3_fuse_ngemv: env_str("PLOW_K3_FUSE_NGEMV"),
            k3_up_nogather: env_bool("PLOW_K3_UP_NOGATHER"),
            k3_up_gather_only: env_bool("PLOW_K3_UP_GATHER_ONLY"),
            k3_shard_head: env_bool("PLOW_K3_SHARD_HEAD"),
            k3_seq_rows: std::env::var_os("PLOW_K3_SEQ_ROWS").is_some(),
            gemv_mm: env_u32("PLOW_GEMV_MM"),
            gemv_walk: env_bool("PLOW_GEMV_WALK"),
            tunedb: std::env::var("PLOW_TUNEDB").ok(), // preserves "" for "disable tuning"
            tune_dump: env_bool("PLOW_TUNE_DUMP"),
            chain_bypass: env_str("PLOW_CHAIN_BYPASS"),
            skip_coverage: env_bool("PLOW_SKIP_COVERAGE"),
            k3_ablate: env_str("PLOW_K3_ABLATE"),
            seg_dump: env_bool("PLOW_SEG_DUMP"),
            place_report: env_bool("PLOW_PLACE_REPORT"),
            plow_root: env_str("PLOW_ROOT"),
            checkpoint: env_str("PLOW_CHECKPOINT"),
        }
    }

    /// Validate cross-field constraints. Panics on incompatible combinations
    /// (same behavior as existing `assert!` calls scattered in devgen).
    pub fn validate(&self) {
        assert!(
            !(self.w8a8 && self.w8a16),
            "PLOW_W8A8=1 and PLOW_W8A16=1 name two activation profiles on one weight axis; pick one"
        );
        assert!(
            !(self.mxfp4 && (self.w8a8 || self.w8a16)),
            "PLOW_MXFP4=1 is A4W4; it is incompatible with PLOW_W8A8/PLOW_W8A16"
        );
        if self.fp8_kv_full && !self.fp8_kv {
            tracing::warn!("--fp8-kv-full has no effect without --fp8-kv");
        }

        // Deprecated alias warnings
        if std::env::var("PLOW_KV_FP8").ok().as_deref() == Some("1") {
            tracing::warn!(
                "PLOW_KV_FP8 is deprecated — use --fp8-kv or PLOW_FP8_KV instead"
            );
        }
        if std::env::var("PLOW_NV_PLACE").ok().as_deref() == Some("1") {
            tracing::warn!(
                "PLOW_NV_PLACE is deprecated — use --l2-place or PLOW_L2_PLACE instead"
            );
        }
    }

    /// Parse a layers spec ("all", "default", "5", "single:3") → (full, n_layers_cap, single_layer).
    ///
    /// - `"all"` → full emit, no cap, no single
    /// - `"default"` → NOT full (single-layer validation gate), no cap, no single
    /// - `"5"` → full emit with cap at 5
    /// - `"single:3"` → emit one specific layer (3)
    pub fn parse_layers(spec: &str) -> (bool, Option<u32>, Option<u32>) {
        match spec {
            "all" => (true, None, None),
            "default" => (false, None, None),
            s if s.starts_with("single:") => {
                let l: u32 = s[7..].parse().expect("--*-layers single:N requires a number");
                (false, None, Some(l))
            }
            s => {
                let n: u32 = s.parse().expect("--*-layers expects 'all', 'default', 'single:N', or a number N");
                (true, Some(n), None)
            }
        }
    }

    /// Resolve GLM layer config from --glm-layers.
    pub fn glm_layer_cfg(&self) -> (bool, Option<u32>, Option<u32>) {
        Self::parse_layers(&self.glm_layers)
    }

    /// Resolve K3 layer config from --k3-layers.
    pub fn k3_layer_cfg(&self) -> (bool, Option<u32>, Option<u32>) {
        Self::parse_layers(&self.k3_layers)
    }

    /// Whether any fp8 weight encoding is active.
    pub fn any_fp8_weights(&self) -> bool {
        self.fp8 || self.w8a8 || self.w8a16
    }

    /// Resolve the tunedb root directory.
    ///
    /// Returns `None` when tuning is explicitly disabled (`PLOW_TUNEDB=""`).
    /// Returns `Some(path)` with the user-specified path or the default tree.
    pub fn tunedb_root(&self) -> Option<String> {
        match &self.tunedb {
            Some(s) if s.is_empty() => None,
            Some(s) => Some(s.clone()),
            None => Some(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tuning")
                    .to_string_lossy()
                    .into_owned(),
            ),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Process-global accessor (same pattern as plowrt::config::RuntimeConfig)
// ──────────────────────────────────────────────────────────────────────────────

use std::sync::OnceLock;

/// Installed config. In production there is exactly one `install()` call at the top of
/// `run_verified`. In test binaries, multiple tests may call `run_verified` with different
/// env-var setups in the same process — they each get a fresh `from_env()` snapshot.
///
/// We leak a `Box<EmitConfig>` on each `install()` so `active()` can return `&'static`.
/// In production that is one allocation; in tests it is one per `run()` call — bounded
/// by the test count and negligible.
static INSTALLED: std::sync::atomic::AtomicPtr<EmitConfig> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Lazy fallback for unit tests that call `active()` without going through `run_verified`.
static FALLBACK: OnceLock<EmitConfig> = OnceLock::new();

/// Install the resolved config for this compile run.
/// Called once at the top of `run_verified`.
pub fn install(cfg: EmitConfig) {
    let ptr = Box::into_raw(Box::new(cfg));
    // In production there is only one call; in tests the last call wins (matches env-var
    // semantics where `set_var` before `run()` is the intent). We intentionally leak the
    // old allocation to keep `&'static` references valid.
    INSTALLED.store(ptr, std::sync::atomic::Ordering::Release);
}

/// Access the active emit config.
///
/// Returns the explicitly [`install`]ed config if one exists. Otherwise, lazily
/// constructs from the process environment — matching the pre-migration semantics
/// where every call site read `std::env::var` directly (needed for unit tests that
/// call emitter helpers without going through `run_verified`).
pub fn active() -> &'static EmitConfig {
    let ptr = INSTALLED.load(std::sync::atomic::Ordering::Acquire);
    if !ptr.is_null() {
        // SAFETY: `install()` wrote a valid Box::into_raw pointer and we never dealloc.
        unsafe { &*ptr }
    } else {
        FALLBACK.get_or_init(|| EmitConfig::from_env())
    }
}

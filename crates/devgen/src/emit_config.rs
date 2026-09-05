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
//!
//! # A field with no reader is worse than no field
//!
//! The migration above is half-done by design, and the half-done state has a trap in it: a knob
//! can be PARSED here and READ nowhere, while the code that actually implements it goes on
//! calling `std::env::var` in another crate. The env var then works and the struct field — and
//! the `--emit-*` CLI flag clap derives from it — silently does nothing.
//!
//! Ten fields were in that state and are deleted rather than wired, because every one of them is
//! genuinely implemented somewhere else off a direct `env::var` read and nothing here ever
//! intended to consume the parsed copy:
//!
//! * `packet::devbuild` reads `PLOW_SEG_PER_OP`, `PLOW_SEG_CLASS_SLICE`, `PLOW_FINE_FORCE`,
//!   `PLOW_CHAIN_BYPASS`, `PLOW_SEG_DUMP`, `PLOW_PLACE_REPORT` directly;
//! * `plowc::main` reads `PLOW_BLOCK` (behind its own `--block`), `PLOW_L2_PLACE` and
//!   `PLOW_ROOT` directly;
//! * `plowrt::config` owns `PLOW_CHECKPOINT` as `--rt-checkpoint`; devgen binds weights from
//!   `--hf-dir` and has no use for it.
//!
//! This is the same duplicated-parse shape that produced the `PLOW_XR_CUS` defect (parsed once,
//! applied to decode only, and found by measurement rather than review). The `--emit-block` and
//! `--emit-checkpoint` CLI flags clap derived from two of them were pure decoration.
//!
//! [`tests::every_field_has_a_reader`] scans the source and fails if a field is added back
//! without one.

use clap::Args;

/// Emit-time configuration — controls what `devgen` and `packet::DevBuild` produce.
///
/// Constructed by `plowc` via `#[command(flatten)]` and threaded through the emit
/// pipeline as `&EmitConfig`. The struct is the single source of truth for every
/// compile-time knob; `std::env::var` calls in `devgen` are being migrated here.
///
/// Knob classes (see `docs/k3-mi355x-20260904/emit-knob-audit.md` for the per-field audit):
/// * generic mechanism switches, kept visible and documented by what they do;
/// * rollbacks of promoted defaults (`=0` restores the pre-promotion packet);
/// * opt-in candidates that still need a network gate;
/// * diagnostics, `hide = true` — they never ship a packet anyone serves.
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

    /// Isolate pure adjacent FlashMlaDecode+MlaMergeFold pairs in their own gfx950 segment.
    /// Default on; `=0` is the rollback to the interpreter-resident pair.
    #[arg(long = "emit-decode-mla-segments", env = "PLOW_SEG_DECODE_MLA", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub decode_mla_segments: bool,

    /// Isolate adjacent grouped MXFP4 GLU+DOWN decode pairs into ordered raw launches.
    /// Unset = decide from qualified per-geometry route measurements
    /// (`moe_decode_measurement.jsonl`, both routes, current digests); missing evidence keeps
    /// the interpreter route. `PLOW_MOE_DECODE_STANDALONE=1` remains the packet-level override.
    #[arg(long = "emit-decode-grouped-moe-segments", env = "PLOW_SEG_DECODE_GROUPED_MOE", value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub decode_grouped_moe_segments: Option<bool>,

    /// Batched decode dispatch width (sequences per launch).
    #[arg(
        long = "emit-decode-batch",
        env = "PLOW_DECODE_BATCH",
        default_value_t = 1
    )]
    pub decode_batch: u32,

    /// DECODE BATCH LADDER: a comma list of decode widths emitted as SEPARATE
    /// programs in ONE blob (e.g. `1,2,4,8,16`), so the runtime picks the smallest
    /// rung that covers the live sequences instead of being committed to one
    /// `PLOW_DECODE_BATCH` at emit.
    ///
    /// Unset (the default) is BYTE-IDENTICAL to today's blob: [`EmitConfig::decode_rungs`]
    /// then returns the single `decode_batch` rung and the emitter takes the exact code
    /// path it always took. Set, the WIDEST rung sizes every per-slot tensor (the KV
    /// cache above all), because a sequence keeps its slot across a rung change and the
    /// per-slot stride must not move with `B`.
    #[arg(long = "emit-decode-batch-ladder", env = "PLOW_DECODE_BATCH_LADDER")]
    pub decode_ladder: Option<String>,

    /// Largest prefill chunk rows (power of two, ≤ 8192). Caps the bucket
    /// ladder and the runtime PLOW_PF_INTERLEAVE ceiling.
    #[arg(long = "emit-max-chunk", env = "PLOW_MAX_CHUNK")]
    pub max_chunk: Option<u32>,

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

    /// Widen the flash-merge dispatch by this factor (diagnostic; measured no effect).
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
    /// Cap XReduce participant CUs.
    #[arg(long, env = "PLOW_XR_CUS")]
    pub xr_cus: Option<u32>,

    /// Use reduce-scatter/all-gather for complete folded-gather collectives. The second
    /// partial is added while the reduced slices are gathered. Default on; `=0` is the
    /// rollback to the one-shot collective.
    #[arg(long, env = "PLOW_XR2_GATHER", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub xr2_gather: bool,

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
    // MLA / hybrid (K3) model family
    // ──────────────────────────────────────────────────────────────────────────
    /// Diagnostic: `K3_FULL=0` prints the legacy K3 capability report instead of emitting.
    #[arg(long, env = "K3_FULL", default_value_t = true, hide = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_full: bool,

    /// Fuse the MLA q/kv/k_rope/gate A-projection GEMVs into one `GemvQkvg` (decode only,
    /// LDS-bounded). Opt-in; not network-gated.
    #[arg(long, env = "PLOW_K3_FUSE_A", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_fuse_a: bool,

    /// Pin the MLA flash-decode `nsplit` (K3 and GLM). Unset = the measured/ctx-adaptive
    /// default; this is the sweep handle for a re-measurement.
    #[arg(long, env = "PLOW_MLA_NS")]
    pub mla_ns: Option<u32>,

    #[arg(long = "k3-ns", env = "PLOW_K3_NS", hide = true)]
    legacy_k3_ns: Option<u32>,

    #[arg(long = "glm-ns", env = "PLOW_GLM_NS", hide = true)]
    legacy_glm_ns: Option<u32>,

    /// Layers to emit (K3 and GLM): "all", a number N (first N layers), or "single:L".
    /// A truncation instrument for block sweeps and TP-equivalence checks; never a served packet.
    #[arg(long, env = "PLOW_LAYERS", hide = true)]
    pub layers: Option<String>,

    #[arg(long = "k3-layers", env = "PLOW_K3_LAYERS", hide = true)]
    legacy_k3_layers: Option<String>,

    #[arg(long = "glm-layers", env = "PLOW_GLM_LAYERS", hide = true)]
    legacy_glm_layers: Option<String>,

    /// K3 prefill bucket control: unset/`full` = the whole ladder, `0` = decode only,
    /// `512,1024` = those rungs.
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

    /// GLM layer-seam fold: the FFN tail's residual and the next layer's input_layernorm as one
    /// AddNorm packet (opt-in, off by default; TP only).
    #[arg(long, env = "PLOW_GLM_FUSE_SEAM", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_fuse_seam: bool,

    /// GLM decode q-rope fold: apply the interleaved q RoPE inside the MLA flash decode's
    /// query staging and drop the `HeadNormRope` packet (opt-in, off by default).
    #[arg(long, env = "PLOW_GLM_FUSE_ROPE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_fuse_rope: bool,

    /// GLM decode q-norm fold: compute `q_a_layernorm` inside fusion G's `GemvQkv` LDS staging
    /// and drop the one-workgroup `RmsNorm` packet (opt-in, off by default).
    #[arg(long, env = "PLOW_GLM_FUSE_QNORM", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_fuse_qnorm: bool,

    /// GLM router off-shared dispatch (co-resident mode 2 only).
    #[arg(long, env = "GLM_ROUTER_OFF_SHARED", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_router_off_shared: bool,

    /// GLM use legacy (unfused) single-CU router.
    #[arg(long, env = "GLM_ROUTER_OLD", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_router_old: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // Norm/GEMV fusion and KDA prefill chain (K3 hybrid family)
    // ──────────────────────────────────────────────────────────────────────────
    /// Fold the decode B1 `RmsNorm -> GEMV` pairs into the GEMV's LDS staging. Default on
    /// (bit-exact); `0` is the unfused rollback, `lat`/`q` keep one site for bisection.
    #[arg(long, env = "PLOW_K3_FUSE_NGEMV")]
    pub k3_fuse_ngemv: Option<String>,

    /// Emit `KdaConvStateStepG` (Conv3 + StateStepG with ping-pong convolution windows) for B1
    /// decode. Opt-in; the decode object must be built `PLOW_K3_KDA_CONV_STEP_DB=1`.
    #[arg(long, env = "PLOW_K3_KDA_CONV_STEP_DB", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_kda_conv_step_db: bool,

    /// Emit the standalone fused KDA decode boundary when its geometry is supported.
    /// Opt-in (benchmark-only so far); unsupported shapes keep the Conv3 -> StateStepG ->
    /// GatedNorm chain.
    #[arg(long = "emit-kda-decode-fused", env = "PLOW_KDA_DECODE_FUSED", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub kda_decode_fused: bool,

    /// Materialize MLA Q/K/V and emit the standalone asymmetric gfx950 prefill boundary.
    /// Opt-in candidate: exact for the first chunk, continuation chunks still diverge.
    #[arg(long, env = "PLOW_MLA_MATERIALIZED_PREFILL", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub mla_materialized_prefill: bool,

    /// Emit the BT64 chunk-KDA prefill pipeline. Default on for gfx950; unsupported shapes keep
    /// the serial recurrence. `=0` forces the serial oracle (rollback).
    #[arg(long, env = "PLOW_KDA_CHUNK", value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub kda_chunk: Option<bool>,

    /// Precompute the V-independent scaled/gated query in chunk W/U. Default on; `=0` rollback.
    #[arg(long, env = "PLOW_KDA_CHUNK_QPRE", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub kda_chunk_qpre: bool,

    /// Isolate exact BT64/D128 chunk-KDA intra packets for the wave-item gfx950 object.
    /// Default on; `=0` is the rollback to the interpreter path.
    #[arg(long, env = "PLOW_KDA_INTRA_WAVE_ITEMS", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub kda_intra_wave_items: bool,

    /// Mark exact qpre BT64/D128 carry segments for the register-resident gfx950 carry object.
    /// Default on; the marked packet requires its paired object at load. `=0` is the rollback
    /// to the interpreter carry.
    #[arg(long, env = "PLOW_KDA_CARRY_REGSTATE", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub kda_carry_regstate: bool,

    /// Mark exact qpre BT64/D128 Wu->carry pairs for the spill-free key-factor gfx950 objects.
    /// Default on at emit; the runtime only takes the route when those objects are built
    /// (`PLOW_HSACO_KDA_KEY_FACTOR`, default OFF: the pair displaces the faster regstate carry).
    #[arg(long, env = "PLOW_KDA_KEY_FACTOR", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub kda_key_factor: bool,

    /// Mark exact qpre BT64/D128 chunk-KDA Wu segments for the lean four-wave gfx950 Wu object.
    /// Default on (TP8 gate 2026-09-04 with the key feed: -52 ms TTFT, bit-exact); the marked
    /// packet requires its paired object. `=0` rolls back to the interpreter Wu.
    #[arg(long, env = "PLOW_KDA_WU_LEAN", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub kda_wu_lean: bool,

    /// Feed the lean Wu's scaled-key hi/lo pair into the register-state carry (implies the lean
    /// Wu; needs `PLOW_KDA_CARRY_REGSTATE`). Default on (same gate); `=0` rolls back.
    #[arg(long, env = "PLOW_KDA_CARRY_KEYFEED", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub kda_carry_keyfeed: bool,

    /// Vocab-column-parallel K3 `lm_head` with an `XArgmaxFin` handoff. Rejected for serving
    /// (TTFT +8 ms for TPOT -0.09 ms); kept for `scripts/k3_tp_equivalence.sh`.
    #[arg(long, env = "PLOW_K3_SHARD_HEAD", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_shard_head: bool,

    /// Diagnostic: force the per-sequence GEMV row carrier at B=1 (bisects the batched-decode
    /// addressing against the known-good B=1 stream).
    #[arg(long, env = "PLOW_K3_SEQ_ROWS", default_value_t = false, hide = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
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

    /// Fold graph-adjacent materialized Residual inputs into AttnRes. Bit-identical and
    /// model-independent. DEFAULT ON; set `PLOW_FUSE_RESIDUAL_INPUT=0` to roll back.
    #[arg(long, env = "PLOW_FUSE_RESIDUAL_INPUT", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fuse_residual_input: bool,

    /// Fuse each AttnRes with its sole following RMSNorm (bit-exact). Default on; `=0` is the
    /// rollback, used to materialize the raw residual seam for a boundary capture.
    #[arg(long, env = "PLOW_K3_FUSE_ARNORM", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub k3_fuse_arnorm: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // GLM-5.2 / gfx942 campaign knobs
    //
    // Every one of these shipped as a direct `std::env::var` read in `mla.rs` or
    // `lib.rs` and was migrated here in one pass. They were never double-parsed —
    // the hazard was the opposite one: a knob that no `--help`, no `--emit-*` flag
    // and no build manifest could see, so a blob's provenance did not record what
    // produced it.
    // ──────────────────────────────────────────────────────────────────────────
    /// Fuse the q/k RMSNorm into the QKV GEMV epilogue.
    #[arg(long, env = "PLOW_QNORM_FUSE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub qnorm_fuse: bool,

    /// Fuse activation quantisation into the producing epilogue. DEFAULT ON for AMD
    /// (opt out with `=0`); the `amd &&` guard stays at the call site.
    #[arg(long, env = "PLOW_FUSE_QUANT", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fuse_quant: bool,

    /// Cap the dispatch width of the fused prefill GEMV.
    #[arg(long, env = "PLOW_GEMV_WG")]
    pub gemv_wg: Option<u32>,

    /// Shape-keyed workgroup caps for blocked decode GEMVs, `NxK=cap[,NxK=cap...]` (for
    /// example `896x7168=224,1536x7168=152`). An A/B override: there is no TuneDB record for
    /// GEMV width, so unset keeps the normal workgroup selection.
    #[arg(long, env = "PLOW_GEMV_WG_TUNING")]
    pub gemv_wg_tuning: Option<String>,

    /// Route GLM's DSA indexer through the prefill chain (requires `has_dsa`).
    #[arg(long, env = "PLOW_GLM_DSA_PF", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_dsa_pf: bool,

    /// Store the MLA latent cache as e4m3 + per-row f32 scale. NOT bit-identical.
    #[arg(long, env = "PLOW_GLM_FP8_KV", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_fp8_kv: bool,

    /// Cap the dispatch width of every blocked GEMV. Unset ⇒ byte-identical.
    #[arg(long, env = "PLOW_GLM_GEMV_WG")]
    pub glm_gemv_wg: Option<u32>,

    /// Fold W_o into the MLA prefill flash epilogue. Reassociated, logit-gate class.
    #[arg(long, env = "PLOW_GLM_OFOLD", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_ofold: bool,

    /// Causal KV-split factor for the V2 MLA prefill flash (2..=8; unset/1 = unsplit).
    #[arg(long, env = "PLOW_GLM_PF_NS")]
    pub glm_pf_ns: Option<u32>,

    /// Widen prefill norm/residual dispatch across CUs. DEFAULT ON (`=0` restores the
    /// single-workgroup emit for A/B). Bit-identical either way.
    #[arg(long, env = "PLOW_GLM_PF_WIDE", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_pf_wide: bool,

    /// Per-XCD CU placement for the GLM prefill chain.
    #[arg(long, env = "PLOW_GLM_PLACE_PF", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_place_pf: bool,

    /// Band count for a prefill TP seam (2..=8; unset/1 = the unbanded emit).
    #[arg(long, env = "PLOW_GLM_XR_BAND")]
    pub glm_xr_band: Option<u32>,

    /// Restrict the banded seam to the first N of the seam's CU list.
    #[arg(long, env = "PLOW_GLM_XR_BAND_CUS")]
    pub glm_xr_band_cus: Option<u32>,

    /// Decode AttnRes on N column-band workgroups with an in-packet tagged rendezvous
    /// (`d_attn_res_mwg`, C3 f32-mix contract). 0/unset = the single-workgroup arm.
    #[arg(long, env = "PLOW_ATTNRES_DECODE_MWG")]
    pub attnres_decode_mwg: Option<u32>,

    /// Diagnostic: restrict banding to one seam (`attn` | `moe`) to bisect a divergence.
    #[arg(long, env = "PLOW_GLM_XR_BAND_SEAM", hide = true)]
    pub glm_xr_band_seam: Option<String>,

    /// Fold the post-collective Residual into the two-shot all-gather. Bit-identical.
    #[arg(long, env = "PLOW_GLM_XR_RES", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_xr_res: bool,

    /// Fuse the seam Residual+Norm into XReduceAddNorm (requires fuse_b1, tp>1).
    #[arg(long, env = "GLM_FUSE_XRN", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_fuse_xrn: bool,

    /// Fold the decode latent `MoeCombine` into the tagged one-shot `XReduce` publish: the
    /// XReduce packet carries `t1 = part`, `i7 = top_k` and no combine packet is emitted.
    /// Needs a `PLOW_XR_COMBINE_FOLD=1` decode object. Default on; `=0` is the rollback.
    #[arg(long, env = "PLOW_XR_COMBINE_FOLD", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub xr_combine_fold: bool,

    /// Fold the K3 decode `f_b` forget-gate GEMV into `KdaStateStepG`'s prologue (L3): the step
    /// packet carries `t4 = f_a`, `j1 = W_fb`, flags bit 2, and the GEMV packet is not emitted.
    /// Needs a `PLOW_KDA_FB_FOLD=1` decode object. Opt-in candidate (default off).
    #[arg(long, env = "PLOW_KDA_FB_FOLD", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub kda_fb_fold: bool,

    /// Run the K3 decode KDA chain (Conv3 -> StateStepG -> GatedNorm) as ONE dataflow-gated
    /// `KdaStateStepG` packet (L7): flags bit 3, `t0 = y`, `t1..t3` raw q/k/v, `t7` = operand
    /// descriptor. Needs a `PLOW_KDA_DECODE_FUSED_ARM=1` decode object. Opt-in candidate.
    #[arg(long, env = "PLOW_KDA_DECODE_FUSED_ARM", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub kda_decode_fused_arm: bool,

    /// Decode objects prefetch a claimed `Gemv` slice's weight rows to L2 before polling its
    /// gate (L8). Packet-inert (loads only); recorded in the manifest so the paired
    /// `plow_config.h` defaults `PLOW_GEMV_PREFETCH` for the decode object. Opt-in candidate.
    #[arg(long, env = "PLOW_GEMV_PREFETCH", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub gemv_prefetch: bool,

    /// Isolate compatible MXFP4 grouped-MoE Down+Combine prefill boundaries for the standalone
    /// deterministic stage-2 object. Default on; `=0` is the rollback to the interpreter route.
    #[arg(long, env = "PLOW_MOE_STAGE2_LEAN", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub moe_stage2_lean: bool,

    /// Isolate compatible MXFP4 grouped-MoE gate/up prefill packets for the standalone stage-1
    /// object. Default on; `=0` is the rollback to the interpreter route.
    #[arg(long, env = "PLOW_MOE_STAGE1_LEAN", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub moe_stage1_lean: bool,

    /// Isolate compatible fixed-order grouped-MoE prefill combines for the standalone combine
    /// object. Default on; `=0` is the rollback to the interpreter route.
    #[arg(long, env = "PLOW_MOE_COMBINE_LEAN", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub moe_combine_lean: bool,

    /// Emit prefill AttnRes packets with the f32-mix contract (separate output-norm epsilon in
    /// `f[1]`) and isolate them for the gfx950 `attn_res_f32mix` object. Default on; tokens
    /// differ from the BF16-seam contract by design. `=0` is the rollback to the interpreter
    /// BF16-seam packet.
    #[arg(long, env = "PLOW_ATTNRES_F32MIX", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub attnres_f32mix: bool,

    /// Split the grouped-MoE prefill align into expert-parallel count/prefix/scatter packets
    /// (T >= 1024). Default on; `=0` is the rollback to the single align packet.
    #[arg(long, env = "PLOW_MOE_ALIGN_PAR", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub moe_align_par: bool,

    /// Sequence-parallel TP seams for prefill: run AttnRes / router / latent xe / top-k on the
    /// reduce-scatter-owned `t/tp` row band and all-gather the results (`XReduceScatter` +
    /// `XAllGather`) instead of replicating the row work on every rank. Default on; the
    /// manifest requires the paired seams arm. `=0` is the rollback to the replicated-row packet.
    #[arg(long, env = "PLOW_SEQ_PAR_SEAMS", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub seq_par_seams: bool,

    /// Whole-expert (expert-parallel) prefill route for graph-proven replicated MoE boundaries.
    /// Opt-in; the emitted EP asset is experiment input for `runtime/bench/amd/moe_ep_boundary`.
    #[arg(long, env = "PLOW_MOE_PREFILL_EP", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub moe_prefill_ep: bool,

    /// Deterministic fused DOWN->combine for the grouped MoE prefill: op 86 accumulates an
    /// integer-valued f64 per token so the k-way sum is exact and order-independent, op 87
    /// reads one contiguous stream. Requires an object built `-DPLOW_MOE_PF_DET=1`
    /// (`plow_moe_pf_det_arm`). Opt-in: gate-passed on GLM-5.2/gfx942 (paired GSM8K 0.9613 vs
    /// 0.9613, TTFT -1.7..-2.9%) and the gfx942 recipe sets it at emit, but `moe_pf_fuse` serves
    /// every MLA+MoE model and a default-on would make Kimi/DeepSeek blobs require the arm on
    /// evidence measured only on GLM. Flip only alongside a Kimi/DeepSeek accuracy run.
    #[arg(long, env = "PLOW_MOE_PF_DET", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub moe_pf_det: bool,

    /// Build the lean gfx950 MoE stage-1 A4-reuse object with the K256 register-B body
    /// (`-DPLOW_MOE1_BODY=1`, bit-identical output). Opt-in; packets are unchanged, the
    /// manifest config header carries the object request.
    #[arg(long, env = "PLOW_MOE_STAGE1_BODY", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub moe_stage1_body: bool,

    /// Build the lean gfx950 MoE stage-2 object with the 64x128 tile body
    /// (`-DPLOW_MOE2_BODY=1`, bit-identical part tensor). Opt-in; same launch contract.
    #[arg(long, env = "PLOW_MOE_STAGE2_BODY", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub moe_stage2_body: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // Pre-campaign knobs swept in by the same migration
    //
    // These predate the GLM work but were direct `env::var` reads in the same two
    // files. Migrated so `tests::no_raw_env_reads` can be a blanket rule rather
    // than a rule with a growing exception list.
    // ──────────────────────────────────────────────────────────────────────────
    /// Opt OUT of the fused GLU GEMM on non-AMD backends. DEFAULT ON (`=1` disables).
    #[arg(long, env = "PLOW_NO_GLU_FUSE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub no_glu_fuse: bool,

    /// Emit TMA descriptors for GEMM operands (sm_90a+).
    #[arg(long, env = "PLOW_TMA_GEMM", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub tma_gemm: bool,

    /// Select the packet-declared native FP8 prefill GEMM role.
    #[arg(long, env = "PLOW_FP8_PF_GEMM_ROLE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fp8_pf_gemm_role: bool,

    /// Isolate FP8 prefill GEMMs while retaining the broad interpreter role.
    #[arg(long, env = "PLOW_QWEN_FP8_PF_ISOLATE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub fp8_pf_isolate: bool,

    #[arg(long, env = "PLOW_QWEN_FP8_M1_TMA", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub qwen_fp8_m1_tma: bool,

    #[arg(long, env = "PLOW_QWEN_W8A8_PREFILL", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub qwen_w8a8_prefill: bool,

    #[arg(long, env = "PLOW_QWEN_DECODE_LT", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub qwen_decode_lt: bool,

    #[arg(long, env = "PLOW_QWEN_FUSE_AB", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub qwen_fuse_ab: bool,

    #[arg(long, env = "PLOW_QWEN_FUSE_MLP", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub qwen_fuse_mlp: bool,

    #[arg(long, env = "PLOW_QWEN_PROJECTION_DAG", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub qwen_projection_dag: bool,

    #[arg(long, env = "PLOW_QWEN_SHARE_QUANT", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub qwen_share_quant: bool,

    #[arg(long, env = "PLOW_QWEN_AB_BLOCKS")]
    pub qwen_ab_blocks: Option<u32>,

    #[arg(long, env = "PLOW_QWEN_PREFILL")]
    pub qwen_prefill: Option<String>,

    /// Fuse the prefill norm pair on Gemma-4 even off the gemv family.
    #[arg(long, env = "PLOW_PF_GFUSE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub pf_gfuse: bool,

    /// Force single-segment emit for buckets at or below this T.
    #[arg(long, env = "PLOW_UNISEG_MAX_T")]
    pub uniseg_max_t: Option<u32>,

    /// Narrow GLM dispatch to the workgroups that own work. DEFAULT ON (`=0` for the
    /// A/B control arm); the emitted arithmetic is unchanged either way.
    #[arg(long, env = "PLOW_GLM_WGFIT", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub glm_wgfit: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // Tuning / diagnostics
    // ──────────────────────────────────────────────────────────────────────────
    /// Tuning database root directory.
    #[arg(long, env = "PLOW_TUNEDB")]
    pub tunedb: Option<String>,

    /// Allow the gfx950 128x384x64 `GemmWide` body on a dense BF16 GEMM. The shape is derived,
    /// not configured: the tile is taken only at the ladder-cap chunk where the exact MxNxK has
    /// a qualified TuneDB measurement naming it the winner and its grid fills every CU. Default
    /// on; `=0` is the rollback to the 128x256x64 body everywhere.
    #[arg(long, env = "PLOW_GEMM_WIDE_C8", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub gemm_wide_c8: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // Diagnostic / never-ship (hidden from --help)
    // ──────────────────────────────────────────────────────────────────────────
    /// Print a TUNEDUMP census line per resolved GEMV shape (tuning-harness diagnostic).
    #[arg(long, env = "PLOW_TUNE_DUMP", hide = true, default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub tune_dump: bool,

    /// Emit a model known to fail coverage checks (diagnostic only).
    #[arg(long, env = "PLOW_SKIP_COVERAGE", hide = true, default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub skip_coverage: bool,

    /// K3 bisection instrument (diagnostic only).
    #[arg(long, env = "PLOW_K3_ABLATE", hide = true)]
    pub k3_ablate: Option<String>,
}

impl EmitConfig {
    /// Return the first valid shape-keyed workgroup cap for `(N,K)`.
    pub fn gemv_wg_for(&self, n: u32, k: u32) -> Option<u32> {
        self.gemv_wg_tuning
            .as_deref()?
            .split(',')
            .find_map(|entry| {
                let (shape, cap) = entry.split_once('=')?;
                let (sn, sk) = shape.split_once('x').or_else(|| shape.split_once('X'))?;
                let sn = sn.trim().parse::<u32>().ok()?;
                let sk = sk.trim().parse::<u32>().ok()?;
                let cap = cap.trim().parse::<u32>().ok()?.max(1);
                (sn == n && sk == k).then_some(cap)
            })
    }

    /// Construct an EmitConfig by reading environment variables (legacy path).
    /// Mirrors what clap's `env` attribute does: each field reads its `PLOW_*` var.
    pub fn from_env() -> EmitConfig {
        let env_bool = |k: &str| std::env::var(k).ok().as_deref() == Some("1");
        let env_bool_opt = |k: &str| std::env::var(k).ok().map(|v| v == "1");
        let env_u32 = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<u32>().ok());
        let env_str = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
        // DEFAULT-ON knobs: the call sites these replaced tested `!= Some("0")`, which is
        // not the negation of `env_bool` — unset enables, and so does any value but "0".
        let env_opt_out = |k: &str| std::env::var(k).ok().as_deref() != Some("0");
        let env_bool_default_true = |k: &str| {
            !matches!(
                std::env::var(k).ok().as_deref(),
                Some("0") | Some("false") | Some("False") | Some("FALSE")
            )
        };
        EmitConfig {
            fp8: env_bool("PLOW_FP8"),
            w8a8: env_bool("PLOW_W8A8"),
            w8a16: env_bool("PLOW_W8A16"),
            mxfp4: env_bool("PLOW_MXFP4"),
            fp8_kv: env_bool("PLOW_FP8_KV") || env_bool("PLOW_KV_FP8"),
            fp8_kv_full: env_bool("PLOW_FP8_KV_FULL"),
            fp8_head: env_bool("PLOW_FP8_HEAD"),
            uniseg: env_bool("PLOW_UNISEG"),
            // The legacy no-config entry remains opt-in. `plowc` supplies the clap default-on
            // value; direct legacy callers must name the feature explicitly.
            decode_mla_segments: env_bool("PLOW_SEG_DECODE_MLA"),
            decode_grouped_moe_segments: env_bool_opt("PLOW_SEG_DECODE_GROUPED_MOE"),
            decode_batch: env_u32("PLOW_DECODE_BATCH").unwrap_or(1),
            decode_ladder: env_str("PLOW_DECODE_BATCH_LADDER"),
            max_chunk: env_u32("PLOW_MAX_CHUNK"),
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
            xr_cus: env_u32("PLOW_XR_CUS"),
            xr2_gather: env_opt_out("PLOW_XR2_GATHER"),
            no_xreduce: env_bool("PLOW_NO_XREDUCE"),
            moe_prefill: env_str("PLOW_MOE_PREFILL"),
            gemma_moe_router_fused: env_bool("PLOW_GEMMA_MOE_ROUTER_FUSED"),
            gemma_moe_router_blocks: env_u32("PLOW_GEMMA_MOE_ROUTER_BLOCKS"),
            gemma_moe_router_exact: env_bool("PLOW_GEMMA_MOE_ROUTER_EXACT"),
            gemma_moe_tail_fuse: env_bool("PLOW_GEMMA_MOE_TAIL_FUSE"),
            k3_full: env_bool_default_true("K3_FULL"),
            k3_fuse_a: env_bool("PLOW_K3_FUSE_A"),
            mla_ns: env_u32("PLOW_MLA_NS"),
            legacy_k3_ns: env_u32("PLOW_K3_NS"),
            legacy_glm_ns: env_u32("PLOW_GLM_NS"),
            k3_prefill: env_str("K3_PREFILL"),
            glm_dsa: env_str("PLOW_GLM_DSA"),
            glm_gf: env_u32("PLOW_GLM_GF"),
            glm_shard_head: env_bool("GLM_SHARD_HEAD"),
            glm_moe_coresident: env_u32("GLM_MOE_CORESIDENT"),
            glm_shared_cus: env_u32("GLM_SHARED_CUS"),
            glm_spine_cus: env_str("GLM_SPINE_CUS"),
            glm_linear_fp8: env_bool("GLM_LINEAR_FP8"),
            glm_shared_glu_split: env_bool("GLM_SHARED_GLU_SPLIT"),
            layers: Some(env_str("PLOW_LAYERS").unwrap_or_else(|| {
                // Legacy synthesis: GLM_FULL=1 → "all" (with GLM_NLAYERS cap),
                // GLM_LAYER=L → "single:L", else "default" (GLM's single-layer validation
                // gate; every other family treats it as "all").
                if std::env::var("GLM_FULL").ok().as_deref() == Some("1") {
                    match std::env::var("GLM_NLAYERS")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        Some(n) => n.to_string(),
                        None => "all".into(),
                    }
                } else if let Some(l) = std::env::var("GLM_LAYER")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    format!("single:{l}")
                } else {
                    "default".into()
                }
            })),
            legacy_k3_layers: env_str("PLOW_K3_LAYERS"),
            legacy_glm_layers: env_str("PLOW_GLM_LAYERS"),
            mla_prefill: env_str("PLOW_MLA_PREFILL"),
            glm_ep: env_bool("GLM_EP"),
            glm_group: env_bool("GLM_GROUP"),
            glm_fuse_b1: env_bool("PLOW_GLM_FUSE_B1"),
            glm_fuse_rope: env_bool("PLOW_GLM_FUSE_ROPE"),
            glm_fuse_seam: env_bool("PLOW_GLM_FUSE_SEAM"),
            glm_fuse_qnorm: env_bool("PLOW_GLM_FUSE_QNORM"),
            glm_router_off_shared: env_bool("GLM_ROUTER_OFF_SHARED"),
            glm_router_old: env_bool("GLM_ROUTER_OLD"),
            k3_fuse_ngemv: env_str("PLOW_K3_FUSE_NGEMV"),
            k3_kda_conv_step_db: env_bool("PLOW_K3_KDA_CONV_STEP_DB"),
            kda_decode_fused: env_bool("PLOW_KDA_DECODE_FUSED"),
            mla_materialized_prefill: env_bool("PLOW_MLA_MATERIALIZED_PREFILL"),
            kda_chunk: env_bool_opt("PLOW_KDA_CHUNK"),
            kda_chunk_qpre: env_opt_out("PLOW_KDA_CHUNK_QPRE"),
            kda_intra_wave_items: env_opt_out("PLOW_KDA_INTRA_WAVE_ITEMS"),
            kda_carry_regstate: env_opt_out("PLOW_KDA_CARRY_REGSTATE"),
            kda_key_factor: env_opt_out("PLOW_KDA_KEY_FACTOR"),
            kda_wu_lean: env_opt_out("PLOW_KDA_WU_LEAN"),
            kda_carry_keyfeed: env_opt_out("PLOW_KDA_CARRY_KEYFEED"),
            k3_shard_head: env_bool("PLOW_K3_SHARD_HEAD"),
            k3_seq_rows: std::env::var_os("PLOW_K3_SEQ_ROWS").is_some(),
            gemv_mm: env_u32("PLOW_GEMV_MM"),
            gemv_walk: env_bool("PLOW_GEMV_WALK"),
            fuse_residual_input: env_opt_out("PLOW_FUSE_RESIDUAL_INPUT"),
            k3_fuse_arnorm: env_opt_out("PLOW_K3_FUSE_ARNORM"),
            // GLM-5.2 / gfx942 campaign knobs. `env_opt_out` is NOT `!env_bool`: the
            // original call sites tested `!= Some("0")`, so any value other than "0"
            // (including an empty string) enables. Preserved verbatim.
            qnorm_fuse: env_bool("PLOW_QNORM_FUSE"),
            fuse_quant: env_opt_out("PLOW_FUSE_QUANT"),
            gemv_wg: env_u32("PLOW_GEMV_WG"),
            gemv_wg_tuning: env_str("PLOW_GEMV_WG_TUNING"),
            glm_dsa_pf: env_bool("PLOW_GLM_DSA_PF"),
            glm_fp8_kv: env_bool("PLOW_GLM_FP8_KV"),
            glm_gemv_wg: env_u32("PLOW_GLM_GEMV_WG"),
            glm_ofold: env_bool("PLOW_GLM_OFOLD"),
            glm_pf_ns: env_u32("PLOW_GLM_PF_NS"),
            glm_pf_wide: env_opt_out("PLOW_GLM_PF_WIDE"),
            glm_place_pf: env_bool("PLOW_GLM_PLACE_PF"),
            glm_xr_band: env_u32("PLOW_GLM_XR_BAND"),
            glm_xr_band_cus: env_u32("PLOW_GLM_XR_BAND_CUS"),
            attnres_decode_mwg: env_u32("PLOW_ATTNRES_DECODE_MWG"),
            glm_xr_band_seam: env_str("PLOW_GLM_XR_BAND_SEAM"),
            glm_xr_res: env_bool("PLOW_GLM_XR_RES"),
            glm_fuse_xrn: env_bool("GLM_FUSE_XRN"),
            xr_combine_fold: env_opt_out("PLOW_XR_COMBINE_FOLD"),
            kda_fb_fold: env_bool("PLOW_KDA_FB_FOLD"),
            kda_decode_fused_arm: env_bool("PLOW_KDA_DECODE_FUSED_ARM"),
            gemv_prefetch: env_bool("PLOW_GEMV_PREFETCH"),
            moe_stage2_lean: env_opt_out("PLOW_MOE_STAGE2_LEAN"),
            moe_align_par: env_opt_out("PLOW_MOE_ALIGN_PAR"),
            seq_par_seams: env_opt_out("PLOW_SEQ_PAR_SEAMS"),
            moe_prefill_ep: env_bool("PLOW_MOE_PREFILL_EP"),
            moe_stage1_lean: env_opt_out("PLOW_MOE_STAGE1_LEAN"),
            moe_combine_lean: env_bool_default_true("PLOW_MOE_COMBINE_LEAN"),
            attnres_f32mix: env_opt_out("PLOW_ATTNRES_F32MIX"),
            moe_pf_det: env_bool("PLOW_MOE_PF_DET"),
            moe_stage1_body: env_bool("PLOW_MOE_STAGE1_BODY"),
            moe_stage2_body: env_bool("PLOW_MOE_STAGE2_BODY"),
            no_glu_fuse: env_bool("PLOW_NO_GLU_FUSE"),
            tma_gemm: env_bool("PLOW_TMA_GEMM"),
            fp8_pf_gemm_role: env_bool("PLOW_FP8_PF_GEMM_ROLE"),
            fp8_pf_isolate: env_bool("PLOW_QWEN_FP8_PF_ISOLATE"),
            qwen_fp8_m1_tma: env_bool("PLOW_QWEN_FP8_M1_TMA"),
            qwen_w8a8_prefill: env_bool("PLOW_QWEN_W8A8_PREFILL"),
            qwen_decode_lt: env_bool("PLOW_QWEN_DECODE_LT"),
            qwen_fuse_ab: env_bool("PLOW_QWEN_FUSE_AB"),
            qwen_fuse_mlp: env_bool("PLOW_QWEN_FUSE_MLP"),
            qwen_projection_dag: env_bool("PLOW_QWEN_PROJECTION_DAG"),
            qwen_share_quant: env_bool("PLOW_QWEN_SHARE_QUANT"),
            qwen_ab_blocks: std::env::var("PLOW_QWEN_AB_BLOCKS")
                .ok()
                .map(|v| v.parse().expect("Qwen a/b block count")),
            qwen_prefill: std::env::var("PLOW_QWEN_PREFILL").ok(),
            pf_gfuse: env_bool("PLOW_PF_GFUSE"),
            uniseg_max_t: env_u32("PLOW_UNISEG_MAX_T"),
            glm_wgfit: env_opt_out("PLOW_GLM_WGFIT"),
            tunedb: std::env::var("PLOW_TUNEDB").ok(), // preserves "" for "disable tuning"
            tune_dump: env_bool("PLOW_TUNE_DUMP"),
            gemm_wide_c8: env_opt_out("PLOW_GEMM_WIDE_C8"),
            skip_coverage: env_bool("PLOW_SKIP_COVERAGE"),
            k3_ablate: env_str("PLOW_K3_ABLATE"),
        }
    }

    /// Validate cross-field constraints. Panics on incompatible combinations
    /// (same behavior as existing `assert!` calls scattered in devgen).
    fn resolve_deprecated_aliases(&mut self) {
        let legacy_ns = match (self.legacy_k3_ns, self.legacy_glm_ns) {
            (Some(k3), Some(glm)) if k3 != glm => {
                panic!("PLOW_K3_NS={k3} conflicts with PLOW_GLM_NS={glm}; use PLOW_MLA_NS")
            }
            (Some(v), _) | (_, Some(v)) => Some(v),
            (None, None) => None,
        };
        if let Some(legacy) = legacy_ns {
            if self.mla_ns.is_some_and(|current| current != legacy) {
                panic!(
                    "PLOW_MLA_NS={} conflicts with deprecated per-model value {legacy}",
                    self.mla_ns.unwrap()
                );
            }
            tracing::warn!("PLOW_K3_NS/PLOW_GLM_NS are deprecated — use PLOW_MLA_NS");
            self.mla_ns = Some(legacy);
        }

        let legacy_layers = match (&self.legacy_k3_layers, &self.legacy_glm_layers) {
            (Some(k3), Some(glm)) if k3 != glm => {
                panic!("PLOW_K3_LAYERS={k3} conflicts with PLOW_GLM_LAYERS={glm}; use PLOW_LAYERS")
            }
            (Some(v), _) | (_, Some(v)) => Some(v.clone()),
            (None, None) => None,
        };
        if let Some(legacy) = legacy_layers {
            if self
                .layers
                .as_deref()
                .is_some_and(|current| current != legacy)
            {
                panic!(
                    "PLOW_LAYERS={} conflicts with deprecated per-model value {legacy}",
                    self.layers.as_deref().unwrap()
                );
            }
            tracing::warn!("PLOW_K3_LAYERS/PLOW_GLM_LAYERS are deprecated — use PLOW_LAYERS");
            self.layers = Some(legacy);
        }
    }

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
            tracing::warn!("PLOW_KV_FP8 is deprecated — use --fp8-kv or PLOW_FP8_KV instead");
        }
        if std::env::var("PLOW_NV_PLACE").ok().as_deref() == Some("1") {
            tracing::warn!("PLOW_NV_PLACE is deprecated — use PLOW_L2_PLACE instead");
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
                let l: u32 = s[7..]
                    .parse()
                    .expect("--*-layers single:N requires a number");
                (false, None, Some(l))
            }
            s => {
                let n: u32 = s
                    .parse()
                    .expect("--*-layers expects 'all', 'default', 'single:N', or a number N");
                (true, Some(n), None)
            }
        }
    }

    /// Resolve the layer truncation from --layers.
    pub fn layer_cfg(&self) -> (bool, Option<u32>, Option<u32>) {
        Self::parse_layers(self.layers.as_deref().unwrap_or("all"))
    }

    /// Whether any fp8 weight encoding is active.
    pub fn any_fp8_weights(&self) -> bool {
        self.fp8 || self.w8a8 || self.w8a16
    }

    /// The decode widths this emit builds programs for, ASCENDING.
    ///
    /// Without `PLOW_DECODE_BATCH_LADDER` this is exactly `[decode_batch]`, which is
    /// what makes an unset ladder byte-identical: the emitter runs its one-decode-program
    /// loop once, at the same `B`, with the same builder settings.
    ///
    /// With it, the list is parsed, clamped to `1..=`[`packet::devbuild::DECODE_RUNG_MAX`],
    /// sorted and deduped. `decode_batch` is IGNORED when a ladder is given — two records
    /// of the same fact is how a B=4 blob once refused itself at load, so there is only one.
    pub fn decode_rungs(&self) -> Vec<u32> {
        let Some(spec) = self.decode_ladder.as_deref() else {
            return vec![self
                .decode_batch
                .clamp(1, packet::devbuild::DECODE_RUNG_MAX)];
        };
        let mut v: Vec<u32> = spec
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .map(|b| b.clamp(1, packet::devbuild::DECODE_RUNG_MAX))
            .collect();
        v.sort_unstable();
        v.dedup();
        assert!(
            !v.is_empty(),
            "PLOW_DECODE_BATCH_LADDER={spec:?} parsed to no rungs — expected a comma list \
             of decode widths, e.g. 1,2,4,8,16"
        );
        v
    }

    /// Is a decode ladder in force? Programs then carry the per-sequence KV addressing
    /// at EVERY rung, including the one-row rung — see `PLOW_DECODE_BATCH_LADDER`.
    pub fn decode_ladder_on(&self) -> bool {
        self.decode_ladder.is_some()
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
pub fn install(mut cfg: EmitConfig) {
    cfg.resolve_deprecated_aliases();
    cfg.validate();
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

#[cfg(test)]
mod tests {
    use super::EmitConfig;
    use clap::Parser;

    #[derive(Parser)]
    struct TestArgs {
        #[command(flatten)]
        emit: EmitConfig,
    }

    #[test]
    fn k3_hybrid_emit_defaults_on_and_keeps_explicit_report_mode() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("K3_FULL", "1")]);
        std::env::remove_var("K3_FULL");
        assert!(EmitConfig::from_env().k3_full);
        assert!(TestArgs::try_parse_from(["test"]).unwrap().emit.k3_full);

        std::env::set_var("K3_FULL", "0");
        assert!(!EmitConfig::from_env().k3_full);
        assert!(
            !TestArgs::try_parse_from(["test", "--k3-full=false"])
                .unwrap()
                .emit
                .k3_full
        );
    }

    #[test]
    fn lean_moe_stage1_default_allows_env_opt_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_MOE_STAGE1_LEAN", "0")]);
        assert!(!super::active().moe_stage1_lean);
    }

    #[test]
    fn lean_moe_stage2_default_allows_env_opt_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_MOE_STAGE2_LEAN", "0")]);
        assert!(!super::active().moe_stage2_lean);
    }

    #[test]
    fn lean_moe_combine_defaults_on_and_allows_env_opt_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_MOE_COMBINE_LEAN", "1")]);
        std::env::remove_var("PLOW_MOE_COMBINE_LEAN");
        assert!(EmitConfig::from_env().moe_combine_lean);

        std::env::set_var("PLOW_MOE_COMBINE_LEAN", "0");
        assert!(!EmitConfig::from_env().moe_combine_lean);

        std::env::set_var("PLOW_MOE_COMBINE_LEAN", "false");
        assert!(!EmitConfig::from_env().moe_combine_lean);
    }

    #[test]
    fn kda_chunk_qpre_defaults_on_and_allows_env_opt_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_KDA_CHUNK_QPRE", "1")]);
        std::env::remove_var("PLOW_KDA_CHUNK_QPRE");
        assert!(EmitConfig::from_env().kda_chunk_qpre);

        std::env::set_var("PLOW_KDA_CHUNK_QPRE", "0");
        assert!(!EmitConfig::from_env().kda_chunk_qpre);
    }

    #[test]
    fn kda_key_factor_defaults_on_and_allows_env_opt_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_KDA_KEY_FACTOR", "1")]);
        std::env::remove_var("PLOW_KDA_KEY_FACTOR");
        assert!(EmitConfig::from_env().kda_key_factor);

        std::env::set_var("PLOW_KDA_KEY_FACTOR", "0");
        assert!(!EmitConfig::from_env().kda_key_factor);

        std::env::remove_var("PLOW_KDA_KEY_FACTOR");
        assert!(
            TestArgs::try_parse_from(["test"])
                .unwrap()
                .emit
                .kda_key_factor
        );
        assert!(
            !TestArgs::try_parse_from(["test", "--kda-key-factor=0"])
                .unwrap()
                .emit
                .kda_key_factor
        );
    }

    #[test]
    fn attnres_f32mix_defaults_on_and_env_opts_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_ATTNRES_F32MIX", "0")]);
        std::env::remove_var("PLOW_ATTNRES_F32MIX");
        assert!(EmitConfig::from_env().attnres_f32mix);
        assert!(
            TestArgs::try_parse_from(["test"])
                .unwrap()
                .emit
                .attnres_f32mix
        );
        std::env::set_var("PLOW_ATTNRES_F32MIX", "0");
        assert!(!EmitConfig::from_env().attnres_f32mix);
        assert!(
            !TestArgs::try_parse_from(["test", "--attnres-f32mix=false"])
                .unwrap()
                .emit
                .attnres_f32mix
        );
    }

    #[test]
    fn kda_carry_regstate_defaults_on_and_allows_env_opt_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_KDA_CARRY_REGSTATE", "1")]);
        std::env::remove_var("PLOW_KDA_CARRY_REGSTATE");
        assert!(EmitConfig::from_env().kda_carry_regstate);
        std::env::set_var("PLOW_KDA_CARRY_REGSTATE", "0");
        assert!(!EmitConfig::from_env().kda_carry_regstate);
    }

    #[test]
    fn kda_intra_wave_items_defaults_on_and_allows_env_opt_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_KDA_INTRA_WAVE_ITEMS", "1")]);
        std::env::remove_var("PLOW_KDA_INTRA_WAVE_ITEMS");
        assert!(EmitConfig::from_env().kda_intra_wave_items);
        assert!(
            TestArgs::try_parse_from(["test"])
                .unwrap()
                .emit
                .kda_intra_wave_items
        );

        std::env::set_var("PLOW_KDA_INTRA_WAVE_ITEMS", "0");
        assert!(!EmitConfig::from_env().kda_intra_wave_items);
        assert!(
            !TestArgs::try_parse_from(["test", "--kda-intra-wave-items=0"])
                .unwrap()
                .emit
                .kda_intra_wave_items
        );
    }

    #[test]
    fn decode_grouped_moe_segments_is_unset_by_default_and_explicit_when_given() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_SEG_DECODE_GROUPED_MOE", "0")]);
        std::env::remove_var("PLOW_SEG_DECODE_GROUPED_MOE");
        assert_eq!(EmitConfig::from_env().decode_grouped_moe_segments, None);
        std::env::set_var("PLOW_SEG_DECODE_GROUPED_MOE", "1");
        assert_eq!(
            EmitConfig::from_env().decode_grouped_moe_segments,
            Some(true)
        );
        std::env::remove_var("PLOW_SEG_DECODE_GROUPED_MOE");

        assert_eq!(
            TestArgs::try_parse_from(["test"])
                .unwrap()
                .emit
                .decode_grouped_moe_segments,
            None
        );
        assert_eq!(
            TestArgs::try_parse_from(["test", "--emit-decode-grouped-moe-segments=false"])
                .unwrap()
                .emit
                .decode_grouped_moe_segments,
            Some(false)
        );
        assert_eq!(
            TestArgs::try_parse_from(["test", "--emit-decode-grouped-moe-segments"])
                .unwrap()
                .emit
                .decode_grouped_moe_segments,
            Some(true)
        );
    }

    #[test]
    fn decode_mla_segments_default_on_for_plowc_but_legacy_is_opt_in() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_SEG_DECODE_MLA", "0")]);
        std::env::remove_var("PLOW_SEG_DECODE_MLA");
        assert!(!EmitConfig::from_env().decode_mla_segments);

        std::env::set_var("PLOW_SEG_DECODE_MLA", "1");
        assert!(EmitConfig::from_env().decode_mla_segments);
        std::env::remove_var("PLOW_SEG_DECODE_MLA");

        assert!(
            TestArgs::try_parse_from(["test"])
                .unwrap()
                .emit
                .decode_mla_segments
        );
        assert!(
            !TestArgs::try_parse_from(["test", "--emit-decode-mla-segments=false"])
                .unwrap()
                .emit
                .decode_mla_segments
        );
    }

    #[test]
    fn deprecated_model_specific_pins_match_unified_pins() {
        let mut old = TestArgs::try_parse_from(["test", "--k3-ns", "4", "--k3-layers", "single:2"])
            .unwrap()
            .emit;
        old.resolve_deprecated_aliases();

        let new = TestArgs::try_parse_from(["test", "--mla-ns", "4", "--layers", "single:2"])
            .unwrap()
            .emit;
        assert_eq!(old.mla_ns, new.mla_ns);
        assert_eq!(old.layers, new.layers);
    }

    #[test]
    fn deprecated_model_specific_pins_reject_conflicts() {
        let mut cfg = TestArgs::try_parse_from(["test", "--mla-ns", "4", "--glm-ns", "8"])
            .unwrap()
            .emit;
        assert!(std::panic::catch_unwind(move || cfg.resolve_deprecated_aliases()).is_err());

        let mut cfg =
            TestArgs::try_parse_from(["test", "--layers", "3", "--glm-layers", "single:1"])
                .unwrap()
                .emit;
        assert!(std::panic::catch_unwind(move || cfg.resolve_deprecated_aliases()).is_err());

        let mut cfg =
            TestArgs::try_parse_from(["test", "--layers", "all", "--glm-layers", "single:1"])
                .unwrap()
                .emit;
        assert!(std::panic::catch_unwind(move || cfg.resolve_deprecated_aliases()).is_err());
    }

    #[test]
    fn materialized_residual_fusion_defaults_on_and_allows_env_opt_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_FUSE_RESIDUAL_INPUT", "1")]);
        std::env::remove_var("PLOW_FUSE_RESIDUAL_INPUT");
        assert!(EmitConfig::from_env().fuse_residual_input);

        std::env::set_var("PLOW_FUSE_RESIDUAL_INPUT", "0");
        assert!(!EmitConfig::from_env().fuse_residual_input);
    }

    #[test]
    fn attnres_norm_fusion_defaults_on_and_allows_capture_opt_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_K3_FUSE_ARNORM", "1")]);
        std::env::remove_var("PLOW_K3_FUSE_ARNORM");
        assert!(EmitConfig::from_env().k3_fuse_arnorm);
        std::env::set_var("PLOW_K3_FUSE_ARNORM", "0");
        assert!(!EmitConfig::from_env().k3_fuse_arnorm);
    }

    #[test]
    fn gemm_wide_c8_defaults_on_and_allows_env_opt_out() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_GEMM_WIDE_C8", "1")]);
        std::env::remove_var("PLOW_GEMM_WIDE_C8");
        assert!(EmitConfig::from_env().gemm_wide_c8);
        assert!(
            TestArgs::try_parse_from(["test"])
                .unwrap()
                .emit
                .gemm_wide_c8
        );
        std::env::set_var("PLOW_GEMM_WIDE_C8", "0");
        assert!(!EmitConfig::from_env().gemm_wide_c8);
        assert!(
            !TestArgs::try_parse_from(["test", "--gemm-wide-c8=0"])
                .unwrap()
                .emit
                .gemm_wide_c8
        );
    }

    #[test]
    fn layers_is_one_knob_for_every_family() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_LAYERS", "2"), ("GLM_FULL", "1")]);
        assert_eq!(EmitConfig::from_env().layer_cfg(), (true, Some(2), None));
        std::env::remove_var("PLOW_LAYERS");
        assert_eq!(EmitConfig::from_env().layer_cfg(), (true, None, None));
        std::env::remove_var("GLM_FULL");
        assert_eq!(EmitConfig::from_env().layer_cfg(), (false, None, None));
        assert_eq!(
            TestArgs::try_parse_from(["test", "--layers", "single:3"])
                .unwrap()
                .emit
                .layer_cfg(),
            (false, None, Some(3))
        );
    }

    #[test]
    fn shape_keyed_gemv_cap_parses_exact_shapes() {
        let mut cfg = EmitConfig::from_env();
        cfg.gemv_wg_tuning = Some("896x7168=224,1536X7168=152".into());
        assert_eq!(cfg.gemv_wg_for(896, 7168), Some(224));
        assert_eq!(cfg.gemv_wg_for(1536, 7168), Some(152));
        assert_eq!(cfg.gemv_wg_for(896, 3584), None);
    }

    #[test]
    fn shape_keyed_gemv_cap_ignores_bad_entries() {
        let mut cfg = EmitConfig::from_env();
        cfg.gemv_wg_tuning = Some("bad,896x=4,896x7168=no,896x7168=0,896x7168=224".into());
        assert_eq!(cfg.gemv_wg_for(896, 7168), Some(1));
    }

    #[test]
    fn unset_shape_keyed_tuning_is_a_noop() {
        let mut cfg = EmitConfig::from_env();
        cfg.gemv_wg_tuning = None;
        assert_eq!(cfg.gemv_wg_for(896, 7168), None);
    }

    /// EVERY `EmitConfig` FIELD MUST BE READ SOMEWHERE, and this is a source grep because the
    /// compiler cannot see the difference.
    ///
    /// A `pub` field on a `pub` struct is never dead code to rustc, so a knob that is parsed
    /// here and consumed nowhere compiles clean, ships, and does nothing — while its env var
    /// keeps working via a direct `env::var` read in another crate, which is what makes the hole
    /// invisible in testing. That is the shape of the `PLOW_XR_CUS` defect (a knob that reached
    /// decode and not prefill, found by measurement, not by review) and of the nine fields
    /// deleted in the same commit as this test.
    ///
    /// The check is deliberately coarse — "does any devgen source outside this file mention
    /// `.field`" — for the reason `every_dispatched_arm_has_an_emit_site` is coarse: a
    /// reachability analysis needs the flag cross-product, and a wrong one fails working builds.
    /// Naming is the cheap 90%, and the failure it catches is exactly "nobody named it at all".
    ///
    /// A field read only through an `impl EmitConfig` method (`tunedb`, `glm_layers`, …) counts:
    /// the method is the reader, and its own callers are what the grep finds for it.
    #[test]
    fn every_field_has_a_reader() {
        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let me = std::fs::read_to_string(src_dir.join("emit_config.rs")).expect("own source");

        // Field names taken from the struct body rather than a hand-kept list — a hand-kept list
        // is the thing that goes stale when someone adds a field.
        let body = {
            let start = me.find("pub struct EmitConfig {").expect("struct");
            let end = me.find("\nimpl EmitConfig").expect("impl");
            &me[start..end]
        };
        let fields: Vec<&str> = body
            .lines()
            .filter_map(|l| l.trim().strip_prefix("pub "))
            .filter_map(|l| l.split_once(':'))
            .map(|(name, _)| name)
            .collect();
        assert!(
            fields.len() > 50,
            "parsed {} fields; the parser is wrong, not the struct",
            fields.len()
        );

        // Every OTHER devgen source, concatenated. `emit_config.rs` is excluded on purpose:
        // `from_env` names every field, so including it would make the test vacuous.
        fn rust_sources(dir: &std::path::Path, out: &mut String) {
            for entry in std::fs::read_dir(dir).expect("source directory") {
                let path = entry.expect("source entry").path();
                if path.is_dir() {
                    rust_sources(&path, out);
                } else if path.extension().is_some_and(|x| x == "rs")
                    && path.file_name().is_some_and(|n| n != "emit_config.rs")
                {
                    out.push_str(&std::fs::read_to_string(path).expect("Rust source"));
                }
            }
        }
        let mut others = String::new();
        rust_sources(&src_dir, &mut others);
        assert!(!others.is_empty(), "no sibling sources readable");

        // The accessors are the indirect readers: a field is live if either it, or the method
        // that resolves it, is named outside this file.
        let via_method: &[(&str, &str)] = &[
            ("tunedb", "tunedb_root()"),
            ("layers", "layer_cfg()"),
            ("fp8", "any_fp8_weights()"),
            ("w8a8", "any_fp8_weights()"),
            ("w8a16", "any_fp8_weights()"),
            ("gemv_wg_tuning", "gemv_wg_for("),
        ];

        let dead: Vec<&str> = fields
            .iter()
            .copied()
            .filter(|f| !others.contains(&format!(".{f}")))
            .filter(|f| {
                !via_method
                    .iter()
                    .any(|(field, m)| field == f && others.contains(m))
            })
            .collect();
        assert!(
            dead.is_empty(),
            "EmitConfig fields parsed but never read: {dead:?}. A knob that silently does \
             nothing is worse than no knob — either wire it to the code that was supposed to \
             consume it, or delete the field and leave the env var to whoever already reads it \
             (that is what happened to PLOW_BLOCK, PLOW_SEG_PER_OP, PLOW_SEG_CLASS_SLICE, \
             PLOW_L2_PLACE, PLOW_FINE_FORCE, PLOW_CHAIN_BYPASS, PLOW_SEG_DUMP, \
             PLOW_PLACE_REPORT, PLOW_ROOT and PLOW_CHECKPOINT). If the reader is a new accessor \
             on EmitConfig, add it to `via_method` above."
        );
    }

    /// THE INVERSE OF [`every_field_has_a_reader`], and the hole it left open.
    ///
    /// That test asks "is every parsed field consumed?". It cannot see the other direction: a
    /// knob implemented as a bare `std::env::var` in `mla.rs` or `lib.rs`, never declared here
    /// at all. Such a knob works — which is exactly why nothing catches it — but it has no
    /// `--emit-*` flag, does not appear in `--help`, and is invisible to anything that records
    /// what produced a blob. The GLM-5.2 / gfx942 campaign added NINETEEN of them before this
    /// test existed.
    ///
    /// Migrating one is mechanical: add the field with its `env =` attribute, add the
    /// `from_env` line, and replace the call site with `emit_config::active().<field>`. Watch
    /// the polarity — several campaign knobs are opt-OUT (`!= Some("0")`), which is NOT the
    /// negation of the `env_bool` helper, since unset must enable.
    #[test]
    fn no_raw_env_reads() {
        // Knobs that legitimately stay raw. Both are deliberate and documented at their site;
        // this list is not a parking spot for new work.
        const ALLOWED: &[(&str, &str)] = &[
            // Owned by `plowc` behind its own `--block`; deliberately NOT an EmitConfig field
            // (see the module header — it was deleted rather than wired).
            ("PLOW_BLOCK", "owned by plowc --block"),
            // Deliberate dual read: env first, `.or(emit_config::active().glm_gf)` second, so an
            // A/B script can repin it mid-process. The config field IS consumed.
            ("PLOW_GLM_GF", "dual read, config field consumed via .or()"),
        ];

        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let re_start = "std::env::var(\"";
        let mut offenders: Vec<String> = Vec::new();

        for entry in std::fs::read_dir(&src_dir).expect("src dir") {
            let path = entry.expect("entry").path();
            if path.extension().is_none_or(|x| x != "rs") {
                continue;
            }
            if path.file_name().is_some_and(|n| n == "emit_config.rs") {
                continue; // from_env legitimately names every var
            }
            let text = std::fs::read_to_string(&path).expect("source");
            for (lineno, line) in text.lines().enumerate() {
                let Some(pos) = line.find(re_start) else {
                    continue;
                };
                let rest = &line[pos + re_start.len()..];
                let Some(end) = rest.find('"') else { continue };
                let var = &rest[..end];
                if !(var.starts_with("PLOW_") || var.starts_with("GLM_")) {
                    continue;
                }
                if ALLOWED.iter().any(|(a, _)| *a == var) {
                    continue;
                }
                let file = path.file_name().unwrap().to_string_lossy().to_string();
                offenders.push(format!("{file}:{} {var}", lineno + 1));
            }
        }

        assert!(
            offenders.is_empty(),
            "devgen reads these knobs straight from the environment, bypassing EmitConfig: \
             {offenders:?}. A knob that only exists as an env var has no --emit-* flag, is \
             absent from --help, and no build record can show it was set. Declare it in \
             EmitConfig (field + `env =` attribute + a `from_env` line) and read it via \
             `emit_config::active()`. If it genuinely belongs to another crate, add it to \
             ALLOWED with the reason — but prefer migrating."
        );
    }
}

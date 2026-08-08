//! Unified runtime configuration for the plow serving engine.
//!
//! Every field has a clap `env` attribute so existing shell scripts / systemd envfiles
//! continue to work. CLI args take precedence over env.
//!
//! # Hot-path access
//!
//! After CLI parse, call [`RuntimeConfig::init`] to store the config globally. All
//! modules then read it through [`RuntimeConfig::get`] — a single atomic load,
//! identical cost to the old `env_flag!` macro.
//!
//! Read with `get()`, not [`RuntimeConfig::global`]: `global()` panics when the
//! config was never installed, which is the normal state for every library
//! embedder (GPU tests, examples, benches — none of them run `main()`'s CLI
//! parse). `get()` falls back to an env-only parse there, so the `PLOW_*`
//! contract holds identically whoever is driving the engine.

use clap::Args;
use std::sync::OnceLock;

/// Runtime configuration for the plow serving engine.
///
/// Stored in a global `OnceLock` after CLI parse for hot-path access (single
/// atomic load). Replaces the scattered `env_flag!` / `env_usize!` macros.
#[derive(Args, Debug, Clone)]
#[command(next_help_heading = "Runtime knobs")]
pub struct RuntimeConfig {
    // ──────────────────────────────────────────────────────────────────────────
    // Shared (both NVIDIA + AMD backends)
    // ──────────────────────────────────────────────────────────────────────────
    /// Checkpoint directory for weight binding. Overrides <assets>/checkpoint.
    #[arg(long = "rt-checkpoint", env = "PLOW_CHECKPOINT", global = true)]
    pub checkpoint: Option<String>,

    /// Checkpoint prefetch depth in tensors.
    #[arg(
        long = "rt-prefetch",
        env = "PLOW_PREFETCH",
        default_value_t = 256,
        global = true
    )]
    pub prefetch: usize,

    /// Prefetch threads per rank. 0 disables prefetch.
    #[arg(
        long = "rt-prefetch-threads",
        env = "PLOW_PREFETCH_THREADS",
        default_value_t = 16,
        global = true
    )]
    pub prefetch_threads: usize,

    /// Single-allocation weight slab (both backends). --no-weight-slab to disable.
    #[arg(long = "rt-weight-slab", env = "PLOW_WEIGHT_SLAB", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub weight_slab: bool,

    /// Keep freed slabs in the pool for reuse.
    #[arg(long = "rt-slab-keep", env = "PLOW_SLAB_KEEP", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub slab_keep: bool,

    /// Per-decode-step timing interval (N = every Nth step). 0 = off.
    #[arg(long = "rt-dstep-every", env = "PLOW_DSTEP_EVERY", global = true)]
    pub dstep_every: Option<u32>,

    // ──────────────────────────────────────────────────────────────────────────
    // Multi-model (S1 switching; see perf-data/multi-model-review-gh200.md)
    // ──────────────────────────────────────────────────────────────────────────
    /// S1 switch drain deadline (ms): past it the victim's live generations are
    /// preempted (`Preempted` finish, queued jobs 429). 0 = preempt immediately;
    /// unset = unbounded drain.
    #[arg(
        long = "drain-timeout-ms",
        env = "PLOW_DRAIN_TIMEOUT_MS",
        global = true
    )]
    pub drain_timeout_ms: Option<u64>,

    /// Speculative next-model preload after an S1 switch. --no-preload disables.
    #[arg(long = "preload", env = "PLOW_PRELOAD", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub preload: bool,

    /// Per-engine KV physical-block reuse pool cap (MiB). 0 disables pooling.
    #[arg(
        long = "kv-pool-mib",
        env = "PLOW_KV_POOL_MIB",
        default_value_t = 512,
        global = true
    )]
    pub kv_pool_mib: u64,

    // ──────────────────────────────────────────────────────────────────────────
    // Diagnostic / observability (shared, off by default)
    // ──────────────────────────────────────────────────────────────────────────
    /// TTFT timeline breakdown (`PLOW_TTFT_LOG=1`).
    #[arg(long = "ttft-log", env = "PLOW_TTFT_LOG", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub ttft_log: bool,

    /// Prefix-cache timing (`PLOW_PFX_LOG=1`).
    #[arg(long = "pfx-log", env = "PLOW_PFX_LOG", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pfx_log: bool,

    /// Decode-step host-phase breakdown (`PLOW_DSTEP_LOG=1`).
    #[arg(long = "dstep-log", env = "PLOW_DSTEP_LOG", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub dstep_log: bool,

    /// Prefill pack-log: one line per batched-prefill launch showing R, rows,
    /// bucket (`PLOW_PF_PACKLOG=1`).
    #[arg(long = "pf-packlog", env = "PLOW_PF_PACKLOG", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_packlog: bool,

    /// Model-load timeline profiling (`PLOW_LOAD_PROFILE=1`).
    #[arg(long = "load-profile", env = "PLOW_LOAD_PROFILE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub load_profile: bool,

    // ──────────────────────────────────────────────────────────────────────────
    // NVIDIA-specific (feature = "cuda")
    // ──────────────────────────────────────────────────────────────────────────
    #[command(flatten)]
    pub nv: NvidiaRuntimeConfig,

    // ──────────────────────────────────────────────────────────────────────────
    // AMD-specific (feature = "hsa")
    // ──────────────────────────────────────────────────────────────────────────
    #[command(flatten)]
    pub amd: AmdRuntimeConfig,
}

/// NVIDIA / sm_120 runtime knobs.
#[derive(Args, Debug, Clone)]
#[command(next_help_heading = "NVIDIA runtime")]
pub struct NvidiaRuntimeConfig {
    /// Bounded device multi-step decode (steps per launch, 2..64). 0/1 = single-step.
    #[arg(
        long = "multistep",
        env = "PLOW_MULTISTEP",
        default_value_t = 8,
        global = true
    )]
    pub multistep: u32,

    /// VMM-backed KV prefix cache. Warm TTFT 3.6×(4k)→23.8×(128k).
    #[arg(long = "vmm-prefix", env = "PLOW_VMM_PREFIX", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub vmm_prefix: bool,

    /// VMM sharing block size (MiB). 2 MiB ≈ 4096 tokens at hd256 bf16.
    #[arg(
        long = "vmm-block-mib",
        env = "PLOW_VMM_BLOCK_MIB",
        default_value_t = 2,
        global = true
    )]
    pub vmm_block_mib: u32,

    /// Cap on retained (unreferenced) VMM blocks. 0 = no cache.
    #[arg(
        long = "vmm-cache-mib",
        env = "PLOW_VMM_CACHE_MIB",
        default_value_t = 0,
        global = true
    )]
    pub vmm_cache_mib: u32,

    /// VMM lazy-commit weight slab (CUDA default ON). --no-nv-weight-vmm to disable.
    #[arg(long = "nv-weight-vmm", env = "PLOW_WEIGHT_VMM", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub weight_vmm: bool,

    /// Direct upload path (CUDA). --no-nv-upload-direct to disable.
    #[arg(long = "nv-upload-direct", env = "PLOW_UPLOAD_DIRECT", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub upload_direct: bool,

    /// Cross-request batched prefill (packs waiting requests' chunks into one launch).
    #[arg(long = "pf-batch", env = "PLOW_PF_BATCH", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_batch: bool,

    /// Chunked prefill quantum — rows admitted per tick before decode runs.
    /// 0 = uncapped.
    #[arg(
        long = "pf-interleave",
        env = "PLOW_PF_INTERLEAVE",
        default_value_t = 2048,
        global = true
    )]
    pub pf_interleave: u32,

    /// Per-request prefill chunk-row cap. 0 = off.
    #[arg(
        long = "pf-chunk",
        env = "PLOW_PF_CHUNK",
        default_value_t = 0,
        global = true
    )]
    pub pf_chunk: u32,

    /// Disable chunked prefill (whole-prompt-per-tick).
    #[arg(long = "pf-no-chunk", env = "PLOW_PF_NO_CHUNK", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_no_chunk: bool,

    /// Disable prefill/decode interleave (prefill-only tick).
    #[arg(long = "pf-no-interleave", env = "PLOW_PF_NO_INTERLEAVE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_no_interleave: bool,

    /// Throughput mode: run prefill chains to completion, skip decode until all
    /// prompts are resident. Trades streaming latency for aggregate tok/s.
    #[arg(long = "pf-defer-decode", env = "PLOW_PF_DEFER_DECODE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_defer_decode: bool,

    /// TP-only prefix cache.
    #[arg(long = "prefix-cache", env = "PLOW_PREFIX_CACHE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub prefix_cache: bool,

    /// Force decode cubin path (bypass discovery).
    #[arg(long = "nv-cubin", env = "PLOW_NV_CUBIN", global = true)]
    pub cubin: Option<String>,

    /// Force prefill cubin path.
    #[arg(long = "nv-cubin-pf", env = "PLOW_NV_CUBIN_PF", global = true)]
    pub cubin_pf: Option<String>,

    /// Force decode kernel entry-point symbol.
    #[arg(long = "nv-kernel", env = "PLOW_NV_KERNEL", global = true)]
    pub kernel: Option<String>,

    /// Force prefill kernel entry-point symbol.
    #[arg(long = "nv-kernel-pf", env = "PLOW_NV_KERNEL_PF", global = true)]
    pub kernel_pf: Option<String>,

    /// Override decode dynamic-smem arena bytes.
    #[arg(long = "nv-smem", env = "PLOW_NV_SMEM", global = true)]
    pub smem: Option<u32>,

    /// Override prefill dynamic-smem arena bytes.
    #[arg(long = "nv-smem-pf", env = "PLOW_NV_SMEM_PF", global = true)]
    pub smem_pf: Option<u32>,

    /// Device sampler enable ("0" to force off, "1" to enable).
    #[arg(long = "dev-sample", env = "PLOW_DEV_SAMPLE", global = true)]
    pub dev_sample: Option<String>,

    /// Sample cubin path override.
    #[arg(long = "nv-cubin-sample", env = "PLOW_NV_CUBIN_SAMPLE", global = true)]
    pub cubin_sample: Option<String>,

    /// Sample kernel symbol override.
    #[arg(
        long = "nv-kernel-sample",
        env = "PLOW_NV_KERNEL_SAMPLE",
        global = true
    )]
    pub kernel_sample: Option<String>,

    /// libcuda.so path override.
    #[arg(long = "libcuda", env = "PLOW_LIBCUDA", global = true)]
    pub libcuda: Option<String>,

    /// Cap the ModelManager VRAM budget (MiB).
    #[arg(long = "vram-budget-mib", env = "PLOW_VRAM_BUDGET_MIB", global = true)]
    pub vram_budget_mib: Option<u64>,

    /// Per-decode-step host-op timing.
    #[arg(long = "step-time", env = "PLOW_STEP_TIME", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub step_time: bool,

    /// L2-domain placement dispatch (accepts old PLOW_NV_PLACE_DISPATCH too).
    #[arg(long = "l2-place-dispatch", env = "PLOW_L2_PLACE_DISPATCH", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub l2_place_dispatch: bool,

    /// Restore covering bucket-pick policy for prefill chunking.
    #[arg(long = "pf-cover", env = "PLOW_PF_COVER", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_cover: bool,

    /// Fixed cost of ONE prefill launch, in padded-row equivalents. 0 = old
    /// pure-minimum-padding policy. Default 512 (measured on sm_120 / gemma-4-12B).
    #[arg(
        long = "pf-chunk-cost",
        env = "PLOW_PF_CHUNK_COST",
        default_value_t = 512,
        global = true
    )]
    pub pf_chunk_cost: usize,

    // ──────────────────────────────────────────────────────────────────────────
    // Segmented prefill (sm_90a T9c..T35 campaign; see
    // perf-data/gemma12b-gh200-prefill-campaign.md)
    // ──────────────────────────────────────────────────────────────────────────
    /// Segmented-prefill object dir (interp_sm90a_pfseg/_pfgemm[/_pffa].cubin).
    /// Unset = single-object prefill.
    #[arg(long = "pf-seg-dir", env = "PLOW_PF_SEG_DIR", global = true)]
    pub pf_seg_dir: Option<String>,

    /// Serve-side segment classing, mirroring the emit-side PLOW_SEG_PURE_GEMM:
    /// "1" = every plain tiled GEMM is GEMM-class, "fp8" = only TMA-mapped fp8
    /// GEMMs (the ws-entry object's sole arm). Must match the blob's emit classing.
    #[arg(long = "pf-seg-pure", env = "PLOW_PF_SEG_PURE", global = true)]
    pub pf_seg_pure: Option<String>,

    /// hd512 flash segments on the dedicated *_pffa object: "1" = hd512 only,
    /// "all" = both head dims (needs an object built PLOW_BUILD_FA_HD256=1 —
    /// the loader refuses a mismatch).
    #[arg(long = "pf-seg-fa512", env = "PLOW_PF_SEG_FA512", global = true)]
    pub pf_seg_fa512: Option<String>,

    /// T35: submit each prefill chunk's segment chain as ONE CUDA graph.
    #[arg(long = "pf-seg-graph", env = "PLOW_PF_SEG_GRAPH", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_seg_graph: bool,

    /// Segment-classing v2 ("1") / q8 variant ("q8").
    #[arg(long = "pf-seg-v2", env = "PLOW_PF_SEG_V2", global = true)]
    pub pf_seg_v2: Option<String>,

    /// Diagnostic: per-class wall attribution via one event pair per segment.
    #[arg(long = "pf-seg-time", env = "PLOW_PF_SEG_TIME", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_seg_time: bool,

    /// Diagnostic: every segment on the fat object (isolates launch serialization).
    #[arg(long = "pf-seg-fatonly", env = "PLOW_PF_SEG_FATONLY", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_seg_fatonly: bool,

    /// Diagnostic: plain (non-cooperative) launch per segment.
    #[arg(long = "pf-seg-noncoop", env = "PLOW_PF_SEG_NONCOOP", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_seg_noncoop: bool,

    /// Per-op prefill attribution: dump block 0's gate/body/signal per opcode
    /// after each chunk (needs a `-DPLOW_NV_TRACE=1` prefill cubin).
    #[arg(long = "pf-trace-log", env = "PLOW_PF_TRACE_LOG", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_trace_log: bool,

    /// Equalize the seg pair's dynamic smem (occ-1 fat object A/B).
    #[arg(long = "pf-seg-eqsmem", env = "PLOW_PF_SEG_EQSMEM", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_seg_eqsmem: bool,
}

impl NvidiaRuntimeConfig {
    /// Prefill interleave rows with "zero = unbounded" semantics.
    /// 0 → `usize::MAX` (no bound), else the configured value.
    pub fn pf_interleave_rows(&self) -> usize {
        if self.pf_interleave == 0 {
            usize::MAX
        } else {
            self.pf_interleave as usize
        }
    }

    /// Per-request prefill chunk-row cap with "zero = unbounded" semantics.
    /// 0 → `usize::MAX`, else the configured value.
    pub fn pf_chunk_rows(&self) -> usize {
        if self.pf_chunk == 0 {
            usize::MAX
        } else {
            self.pf_chunk as usize
        }
    }
}

/// AMD / gfx950 runtime knobs.
#[derive(Args, Debug, Clone)]
#[command(next_help_heading = "AMD runtime")]
pub struct AmdRuntimeConfig {
    /// Counter double-buffering (default ON). --no-amd-ctr-dbuf to disable.
    #[arg(long = "amd-ctr-dbuf", env = "PLOW_CTR_DBUF", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub ctr_dbuf: bool,

    /// Global-queue scheduler. "0" = static per-block-stream, "1" = GQ.
    #[arg(long = "amd-global-queue", env = "PLOW_GLOBAL_QUEUE", global = true)]
    pub global_queue: Option<String>,

    /// Force static scheduler for both phases.
    #[arg(long = "amd-static", env = "PLOW_STATIC", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub static_both: bool,

    /// Force static scheduler for decode only.
    #[arg(long = "amd-static-decode", env = "PLOW_STATIC_DECODE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub static_decode: bool,

    /// Force static scheduler for prefill only.
    #[arg(long = "amd-static-prefill", env = "PLOW_STATIC_PREFILL", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub static_prefill: bool,

    /// Segment enqueue/drain windowing. --no-amd-seg-window to disable.
    #[arg(long = "amd-seg-window", env = "PLOW_SEG_WINDOW", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub seg_window: bool,

    /// VMM-backed KV on ROCr (opt-in, requires hsa_amd_vmem_*).
    #[arg(long = "amd-vmm-kv", env = "PLOW_VMM_KV", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub vmm_kv: bool,

    /// VMM block size for AMD KV (MiB).
    // `id` disambiguates from the NVIDIA twin: clap derive uses the FIELD name
    // as the arg id, and two flattened structs with the same field name break
    // every full parse ("required argument was not provided"). Sharing the env
    // var across backends is intended; sharing the id is not.
    #[arg(
        id = "amd_vmm_block_mib",
        long = "amd-vmm-block-mib",
        env = "PLOW_VMM_BLOCK_MIB",
        default_value_t = 2,
        global = true
    )]
    pub vmm_block_mib: u32,

    /// VMM weight slab on AMD (opt-in, unmeasured).
    #[arg(id = "amd_weight_vmm", long = "amd-weight-vmm", env = "PLOW_WEIGHT_VMM", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub weight_vmm: bool,

    /// Upload-ring pipeline depth.
    #[arg(
        long = "amd-upload-slots",
        env = "PLOW_UPLOAD_SLOTS",
        default_value_t = 4,
        global = true
    )]
    pub upload_slots: u32,

    /// Accept oversubscribed grid (blob.n_cu > device CUs).
    #[arg(long = "amd-oversub", env = "PLOW_OVERSUB", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub oversub: bool,

    /// Shared (vs per-rank) checkpoint mapping across TP ranks.
    #[arg(long = "amd-share-ckpt", env = "PLOW_SHARE_CKPT", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub share_ckpt: bool,

    /// One-at-a-time per-rank load.
    #[arg(long = "amd-tp-serial-load", env = "PLOW_TP_SERIAL_LOAD", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub tp_serial_load: bool,

    /// Cross-rank agreement interval (every Nth step).
    #[arg(
        long = "amd-tp-agree-every",
        env = "PLOW_TP_AGREE_EVERY",
        default_value_t = 1,
        global = true
    )]
    pub tp_agree_every: u32,

    /// Disable redundant-rank audit (for timing runs).
    #[arg(long = "amd-tp-no-audit", env = "PLOW_TP_NO_AUDIT", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub tp_no_audit: bool,

    /// Override prefill pad/launch-rows tradeoff.
    #[arg(long = "amd-launch-rows", env = "PLOW_LAUNCH_ROWS", global = true)]
    pub launch_rows: Option<u32>,

    /// hsaco directory override. Default: <assets>/hsaco.
    #[arg(long = "hsaco", env = "PLOW_HSACO", global = true)]
    pub hsaco: Option<String>,

    /// fp8 checkpoint directory.
    #[arg(long = "fp8-dir", env = "PLOW_FP8_DIR", global = true)]
    pub fp8_dir: Option<String>,

    /// Raw trace output path (per-packet timeline).
    #[arg(long = "trace-raw", env = "PLOW_TRACE_RAW", global = true)]
    pub trace_raw: Option<String>,

    /// LM head row0 debug mode.
    #[arg(long = "amd-lm-row0", env = "PLOW_LM_ROW0", default_value_t = false, hide = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub lm_row0: bool,
}

/// Global runtime config, initialized once at startup from CLI parse.
static RUNTIME_CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

impl RuntimeConfig {
    /// Store the parsed config globally. Call once from `main()` after CLI parse.
    ///
    /// # Panics
    /// Panics if called more than once.
    pub fn init(cfg: RuntimeConfig) {
        RUNTIME_CONFIG
            .set(cfg)
            .expect("RuntimeConfig::init called more than once");
    }

    /// Access the global runtime config.
    ///
    /// # Panics
    /// Panics if [`Self::init`] was not called (programming error — should be
    /// unreachable after main() sets it up).
    pub fn global() -> &'static RuntimeConfig {
        RUNTIME_CONFIG
            .get()
            .expect("RuntimeConfig not initialized — call RuntimeConfig::init() from main")
    }

    /// Whether the global config has been initialized (for tests that don't go through main).
    pub fn is_initialized() -> bool {
        RUNTIME_CONFIG.get().is_some()
    }

    /// A knob whose env var must be re-read on every call, falling back to the
    /// parsed config — for the handful of knobs a test or bench flips
    /// mid-process, after the config snapshot is cached.
    ///
    /// Exists so those sites stop hand-rolling the parse. They had drifted into
    /// four different answers for the same input: one site read `v == "1"`, so
    /// `PLOW_VMM_PREFIX=true` *disabled* VMM prefix while the config path —
    /// clap's `BoolishValueParser` — reads `true` as enabled. Same variable,
    /// opposite meaning depending on which line got there first. This uses
    /// clap's own boolish set, so env and CLI agree by construction.
    pub fn env_bool_or(var: &str, cfg: bool) -> bool {
        match std::env::var(var) {
            Ok(v) => matches!(
                v.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "y" | "on"
            ),
            Err(_) => cfg,
        }
    }

    /// Tri-state form of [`Self::env_bool_or`]: `None` when the var is unset,
    /// for knobs whose unset case is not simply "use the config" (the weight
    /// slab also consults a process-wide default the manager sets).
    pub fn env_bool(var: &str) -> Option<bool> {
        let v = std::env::var(var).ok()?;
        Some(matches!(
            v.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "y" | "on"
        ))
    }

    /// Env-first parse for a value knob, falling back to the parsed config.
    /// A malformed value falls through to the config rather than silently
    /// meaning zero — the old hand-rolled sites disagreed on this.
    pub fn env_parse_or<T: std::str::FromStr>(var: &str, cfg: T) -> T {
        std::env::var(var)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(cfg)
    }

    /// Env-first string knob, falling back to the parsed config.
    pub fn env_str_or(var: &str, cfg: Option<String>) -> Option<String> {
        std::env::var(var).ok().or(cfg)
    }

    /// The initialized global when present, else a cached env-only snapshot.
    ///
    /// Library embedders (GPU tests, examples, benches) construct engines
    /// without running `main()`'s CLI parse; parsing an empty argv lets clap's
    /// `env` attributes do the reading, so the `PLOW_*` contract holds for
    /// every entry point. Cold-path accessor — one atomic load once cached.
    pub fn get() -> &'static RuntimeConfig {
        if let Some(c) = RUNTIME_CONFIG.get() {
            return c;
        }
        static FALLBACK: OnceLock<RuntimeConfig> = OnceLock::new();
        FALLBACK.get_or_init(|| {
            use clap::Parser;
            #[derive(Parser)]
            struct EnvOnly {
                #[command(flatten)]
                cfg: RuntimeConfig,
            }
            EnvOnly::parse_from(["plowrt"]).cfg
        })
    }
}

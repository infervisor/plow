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
    ///
    /// Explicit `id` for the reason spelled out on `hsaco` below: the long was
    /// already `--rt-checkpoint`, but the clap id defaulted to the FIELD name and
    /// collided with `amd-bench --checkpoint`.
    #[arg(
        id = "rt_checkpoint",
        long = "rt-checkpoint",
        env = "PLOW_CHECKPOINT",
        global = true
    )]
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
#[command(next_help_heading = "Scheduling and NVIDIA runtime")]
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

    /// Cross-request prefill scheduling. CUDA packs chunks into one launch. AMD TP packs only
    /// exact-capability programs; unsupported programs retain fair isolated scheduling.
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

    /// Clear per-slot recurrent state with one device kernel per rank.
    #[arg(long = "amd-state-clear-device", env = "PLOW_STATE_CLEAR_DEVICE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub state_clear_device: bool,

    /// Load exact-capability packed-prefill operator-family objects. Mux co-packing additionally
    /// requires --pf-batch, a TP engine, and a compatibly segmented packet.
    #[arg(long = "amd-packed-prefill-route", env = "PLOW_PACKED_PREFILL_ROUTE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub packed_prefill_route: bool,

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

    /// Upload-ring pipeline depth. Values above one are experimental on ROCm:
    /// concurrent copies into one large allocation fault on current gfx950 drivers.
    #[arg(
        long = "amd-upload-slots",
        env = "PLOW_UPLOAD_SLOTS",
        default_value_t = 1,
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

    /// Read the TP counter audit through host-mapped large BAR memory.
    #[arg(long = "amd-tp-audit-direct", env = "PLOW_TP_AUDIT_DIRECT", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub tp_audit_direct: bool,

    /// Compact the exact TP counter audit on device, then read one status word per rank.
    #[arg(long = "amd-tp-audit-compact", env = "PLOW_TP_AUDIT_COMPACT", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub tp_audit_compact: bool,

    /// Override prefill pad/launch-rows tradeoff.
    #[arg(long = "amd-launch-rows", env = "PLOW_LAUNCH_ROWS", global = true)]
    pub launch_rows: Option<u32>,

    /// RAGGED-M prefill: cover a prompt in the FEWEST launches and run the last
    /// chunk at its real row count instead of its padded bucket width.
    ///
    /// **Default ON since the facts gate.** `PLOW_RAGGED_CHUNK=0` restores the
    /// padding-vs-launch DP byte-identically and is the control arm for any A/B
    /// in this area — an arm that merely omits the flag is no longer a control.
    ///
    /// What landing it accepts, stated plainly: **57.8% of prompt LENGTHS produce
    /// different long-form wording than they did**, diverging ~11% into the
    /// answer. That is large, and the reasons it is acceptable are measured, not
    /// assumed: an identical plan gives byte-identical text (62/62); the
    /// determinant is the LAST chunk's executed row count, so this DELETES the
    /// narrow-tail numeric regime rather than adding one, moving prompts into the
    /// wide-chunk regime every on-rung prompt already used; and the quality gate
    /// that had been missing now exists, was proven able to fail, and passed —
    /// `perf-data/probes/facts_gate.py`,
    /// `perf-data/plow-gfx942/glm52-facts-gate.md`.
    ///
    /// ON, the last chunk's bucket is the smallest one that COVERS the remainder
    /// (rather than the cheapest padded cover of it) and every prefill row-count
    /// operand is rewritten from the bucket width to the chunk's real length, so
    /// the padding costs nothing. That is what removes the tail launch a prompt
    /// one token past a bucket used to pay -- measured -239 ms at 4097 tokens.
    /// See `exec::amd::rebase_chunk_rows` and
    /// `perf-data/plow-gfx942/glm52-ragged-tail-chunk.md`.
    ///
    /// The engine REFUSES to serve a packet whose prefill collectives are
    /// row-banded (`PLOW_GLM_XR_BAND`) under this flag rather than half-applying
    /// the shrink.
    #[arg(long = "amd-ragged-chunk", env = "PLOW_RAGGED_CHUNK", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub ragged_chunk: bool,

    /// hsaco directory override. Default: <assets>/hsaco.
    ///
    /// The clap **id** and **long** are both `rt-hsaco`, not `hsaco`. A
    /// `global = true` arg is propagated into every subcommand, so sharing an id
    /// with `amd-bench`'s own `--hsaco` (a required `PathBuf` where this is an
    /// `Option<String>`) made clap hold two definitions under one id and PANIC on
    /// the type downcast — "Mismatch between definition and access of hsaco" —
    /// on EVERY `amd-bench` invocation in the tree (every script under scripts/
    /// that benches a blob). `checkpoint` above had the same collision. The
    /// `rt-` prefix is the convention the other globals here already use.
    #[arg(id = "rt_hsaco", long = "rt-hsaco", env = "PLOW_HSACO", global = true)]
    pub hsaco: Option<String>,

    /// fp8 checkpoint directory.
    #[arg(long = "fp8-dir", env = "PLOW_FP8_DIR", global = true)]
    pub fp8_dir: Option<String>,

    /// Raw trace output path (per-packet timeline).
    #[arg(long = "trace-raw", env = "PLOW_TRACE_RAW", global = true)]
    pub trace_raw: Option<String>,

    /// Directory for per-tick rank-0 counter snapshots (diagnostic).
    #[arg(
        long = "amd-ctr-snap",
        env = "PLOW_CTR_SNAP",
        hide = true,
        global = true
    )]
    pub ctr_snap: Option<String>,

    /// Directory for per-tick tensor snapshots (diagnostic).
    #[arg(
        long = "amd-tens-snap",
        env = "PLOW_TENS_SNAP",
        hide = true,
        global = true
    )]
    pub tens_snap: Option<String>,

    /// Comma-separated named tensors captured by `--amd-tens-snap`.
    #[arg(
        long = "amd-snap-tensors",
        env = "PLOW_SNAP_TENSORS",
        hide = true,
        global = true
    )]
    pub snap_tensors: Option<String>,

    /// Sequence slot captured by `--amd-tens-snap`.
    #[arg(
        long = "amd-snap-slot",
        env = "PLOW_SNAP_SLOT",
        default_value_t = 5,
        hide = true,
        global = true
    )]
    pub snap_slot: usize,

    /// Additional decode-object tiers: `dir:max[,dir:max]`, or one legacy dir.
    #[arg(long = "amd-hsaco-lowrung", env = "PLOW_HSACO_LOWRUNG", global = true)]
    pub hsaco_lowrung: Option<String>,

    /// Widest rung served when `PLOW_HSACO_LOWRUNG` names one legacy directory.
    #[arg(
        long = "amd-lowrung-max",
        env = "PLOW_LOWRUNG_MAX",
        default_value_t = 2,
        global = true
    )]
    pub lowrung_max: u32,

    /// LM head row0 debug mode.
    #[arg(long = "amd-lm-row0", env = "PLOW_LM_ROW0", default_value_t = false, hide = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub lm_row0: bool,

    /// Route `FlashMlaPrefill` segments onto a 4-wave V2 object. On gfx950 a pure dense
    /// bf16 segment prefers the dedicated scratch-free V2+SV object; the general flash
    /// object is the capability-checked fallback. Set false only to run legacy blobs.
    ///
    /// SERVE-TIME AND LOAD-BEARING, not a tuning knob: a `PLOW_GLM_OFOLD=1` blob
    /// is REFUSED without it, because on the 8-wave kernel that blob leaves
    /// unnormalized f32 partials for the fused GEMM to read as bf16 — finite,
    /// fluent, and wrong. Enabled by default for production AMD packets.
    #[arg(long = "amd-mla-pf-v2", env = "PLOW_MLA_PF_V2", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub mla_pf_v2: bool,

    /// Download rank 0's copy of act tensors after the prefill/step:
    /// `name:path[,name:path...]`. A measurement instrument, not a serving path.
    #[arg(long = "amd-dump-act", env = "PLOW_DUMP_ACT", global = true)]
    pub dump_act: Option<String>,
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

#[cfg(test)]
mod tests {
    /// A `RuntimeConfig` field that nothing reads is a CLI flag that silently does nothing.
    ///
    /// This is not hypothetical and it is why the test exists: `amd.trace_raw` was parsed here,
    /// carried a `--trace-raw` flag, and had NO reader anywhere — while `main.rs` and
    /// `exec/amd.rs` read `PLOW_TRACE_RAW` through `env::var_os` in four places. The env var
    /// worked, so nothing looked broken; the flag was decoration. Same duplicated-parse shape as
    /// the `PLOW_XR_CUS` defect, and `devgen::emit_config` already carries the twin of this test.
    ///
    /// Coarse on purpose — "does any plowrt source outside this file mention `.field`". A
    /// reachability analysis needs the feature cross-product and a wrong one fails working
    /// builds; naming is the cheap 90%, and what it catches is "nobody named it at all".
    #[test]
    fn every_field_has_a_reader() {
        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let me = std::fs::read_to_string(src_dir.join("config.rs")).expect("own source");

        // Field names from the struct bodies, not a hand-kept list — a hand-kept list is the
        // thing that goes stale when someone adds a field.
        let mut fields: Vec<String> = Vec::new();
        for marker in [
            "pub struct RuntimeConfig {",
            "pub struct NvidiaRuntimeConfig {",
            "pub struct AmdRuntimeConfig {",
        ] {
            let start = me
                .find(marker)
                .unwrap_or_else(|| panic!("{marker} not found"));
            let body = &me[start..];
            let end = body.find("\n}").expect("struct end");
            for line in body[..end].lines() {
                if let Some(rest) = line.trim().strip_prefix("pub ") {
                    if let Some((name, _)) = rest.split_once(':') {
                        fields.push(name.to_string());
                    }
                }
            }
        }
        assert!(
            fields.len() > 30,
            "parsed {} fields; the parser is wrong",
            fields.len()
        );

        let others: String = walk(&src_dir);
        assert!(!others.is_empty(), "no sibling sources readable");

        let dead: Vec<&String> = fields
            .iter()
            // `amd`/`nvidia` are the sub-struct handles; reads go through them as `.amd.x`.
            .filter(|f| !others.contains(&format!(".{f}")))
            .collect();
        assert!(
            dead.is_empty(),
            "RuntimeConfig fields parsed but never read: {dead:?}. Each is a --flag that does \
             nothing while its env var keeps working through a direct read elsewhere, which is \
             exactly how `trace_raw` went unnoticed. Wire it, or delete the field and leave the \
             env var to whoever already reads it."
        );
    }

    /// The inverse: a knob read straight from the environment, bypassing `RuntimeConfig`.
    /// Such a knob has no `--flag`, is absent from `--help`, and cannot be set any other way.
    #[test]
    fn no_raw_env_reads() {
        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders: Vec<String> = Vec::new();
        for (path, text) in files(&src_dir) {
            if path.file_name().is_some_and(|n| n == "config.rs") {
                continue;
            }
            for (i, line) in text.lines().enumerate() {
                for pat in ["std::env::var(\"", "std::env::var_os(\""] {
                    let Some(pos) = line.find(pat) else { continue };
                    let rest = &line[pos + pat.len()..];
                    let Some(end) = rest.find('"') else { continue };
                    let var = &rest[..end];
                    if var.starts_with("PLOW_") || var.starts_with("GLM_") {
                        let f = path.file_name().unwrap().to_string_lossy().to_string();
                        offenders.push(format!("{f}:{} {var}", i + 1));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "plowrt reads these knobs straight from the environment, bypassing RuntimeConfig: \
             {offenders:?}. Declare the field (with its `env =` attribute) and read it via \
             `RuntimeConfig::get()`, so the knob also has a CLI flag and shows up in --help."
        );
    }

    fn files(dir: &std::path::Path) -> Vec<(std::path::PathBuf, String)> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(dir) else {
            return out;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                out.extend(files(&p));
            } else if p.extension().is_some_and(|x| x == "rs") {
                if let Ok(t) = std::fs::read_to_string(&p) {
                    out.push((p, t));
                }
            }
        }
        out
    }

    fn walk(dir: &std::path::Path) -> String {
        files(dir)
            .into_iter()
            .filter(|(p, _)| p.file_name().is_some_and(|n| n != "config.rs"))
            .map(|(_, t)| t)
            .collect()
    }
}

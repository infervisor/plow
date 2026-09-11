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

    /// Root of the local asset store (`blobs/`, `refs/`, `bundles/`).
    /// Defaults to `$HOME/.plow`.
    #[arg(long = "plow-home", env = "PLOW_HOME", global = true)]
    pub plow_home: Option<String>,

    /// Asset registry a bare model reference resolves against. A `file://` URL
    /// or an absolute path selects a local mirror and needs no HTTP client.
    #[arg(
        long = "registry",
        env = "PLOW_REGISTRY",
        default_value = crate::dist::reference::DEFAULT_REGISTRY,
        global = true
    )]
    pub registry: String,

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

    /// Pack prefill and decode into a shared GPU launch when supported.
    #[arg(long = "fusion", env = "PLOW_FUSION", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub fusion: bool,

    /// Select unified token batching when the backend, model and object support it.
    /// Unsupported configurations use ordinary execution; --fusion takes precedence.
    /// Disable with --token-batch=false or PLOW_TOKEN_BATCH=0.
    #[arg(long = "token-batch", env = "PLOW_TOKEN_BATCH", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub token_batch: bool,

    /// Prefix reuse on compatible AMD and NVIDIA assets.
    #[arg(long = "prefix-cache", env = "PLOW_PREFIX_CACHE", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub prefix_cache: bool,

    /// Soft cap on prefix blocks and boundary snapshots as a fraction of the device's
    /// memory, 0..=1 — the unit vLLM's `--gpu-memory-utilization` uses, scoped here to
    /// the prefix cache. A fixed byte count is the wrong unit: 4 GiB is 5% of an H100
    /// and 2% of an MI300X, and the model decides what is left over. 0 = OOM-driven
    /// eviction only. `--vmm-cache-mib` overrides it with an explicit size.
    #[arg(
        long = "vmm-cache-memory-utilization",
        env = "PLOW_VMM_CACHE_MEMORY_UTILIZATION",
        default_value_t = 0.05,
        value_parser = clap::value_parser!(f64),
        global = true
    )]
    pub vmm_cache_memory_utilization: f64,

    /// Explicit soft cap on prefix blocks and boundary snapshots in MiB; overrides
    /// `--vmm-cache-memory-utilization`. 0 = OOM-driven eviction only.
    #[arg(long = "vmm-cache-mib", env = "PLOW_VMM_CACHE_MIB", global = true)]
    pub vmm_cache_mib: Option<u32>,

    /// VMM sharing block size (MiB) for the prefix pools on either vendor. 2 MiB ≈ 4096 tokens
    /// at hd256 bf16; raise (e.g. 64) for 128k-dedup work.
    #[arg(
        long = "vmm-block-mib",
        env = "PLOW_VMM_BLOCK_MIB",
        default_value_t = 2,
        global = true
    )]
    pub vmm_block_mib: u32,

    /// VMM lazy-commit weight slab. Unset = the vendor default: on for CUDA (the slab reserves
    /// VA in µs and commits pages overlapped with the upload — measured), off for AMD (the flat
    /// slab already saved 7–8.5 s/rank there; the residual win is unmeasured). `=0`/`=1`
    /// overrides on either vendor.
    #[arg(long = "weight-vmm", env = "PLOW_WEIGHT_VMM", value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub weight_vmm: Option<bool>,

    /// Cross-request prefill scheduling. CUDA packs chunks into one launch (unset = off). AMD
    /// co-packs compatible mid-prefill spans into one compiled rung (unset = on; programs the
    /// packed route refuses stay isolated). An explicit `=1` also rotates isolated admission
    /// across slots instead of oldest-first.
    #[arg(long = "pf-batch", env = "PLOW_PF_BATCH", value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_batch: Option<bool>,

    /// Step token budget — prefill rows admitted per tick before decode runs.
    /// Unset: CUDA 2048; AMD the widest compiled prefill rung. 0 = uncapped.
    #[arg(long = "pf-interleave", env = "PLOW_PF_INTERLEAVE", global = true)]
    pub pf_interleave: Option<u32>,

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

    /// Override whether freed slabs remain in the process reuse pool.
    #[arg(long = "rt-slab-keep", env = "PLOW_SLAB_KEEP", value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub slab_keep: Option<bool>,

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

    /// Device ordinals to serve on, e.g. `--devices 0,1,2,3`. Unset = every
    /// visible GPU.
    ///
    /// These index the VISIBLE set, not the physical one: `CUDA_VISIBLE_DEVICES`
    /// and `ROCR_VISIBLE_DEVICES` are applied by the vendor runtime before
    /// plowrt sees a device, so with `ROCR_VISIBLE_DEVICES=4,5` the two visible
    /// GPUs are `--devices 0,1`. Startup logs the mask in force and the visible
    /// count, and an out-of-range ordinal is refused by name.
    #[arg(
        long = "devices",
        env = "PLOW_DEVICES",
        value_delimiter = ',',
        global = true
    )]
    pub devices: Vec<u32>,

    /// How models are laid out over the visible devices: `spread` (one model
    /// per GPU where possible), `pack` (fill a GPU while models fit), or
    /// `explicit` (every model must name its device).
    #[arg(
        long = "place",
        env = "PLOW_PLACE",
        default_value = "spread",
        global = true
    )]
    pub place: crate::serve::placement::Place,

    /// Pin a model to a device: `--pin slug@2`, repeatable. The ordinal is the
    /// first device of the group the model must occupy. Required for every
    /// model under `--place explicit`; an override elsewhere.
    #[arg(long = "pin", env = "PLOW_PIN", value_delimiter = ',', global = true)]
    pub pin: Vec<String>,

    /// How co-resident models take a shared GPU: `free` (private streams, the
    /// driver admits whoever is ready — fastest, and the default) or `rr`
    /// (round-robin turns, which bounds starvation and makes the interleaving
    /// reproducible at the cost of overlap).
    #[arg(
        long = "co-sched",
        env = "PLOW_CO_SCHED",
        default_value = "free",
        global = true
    )]
    pub co_sched: crate::serve::cosched::CoSched,

    /// Consecutive ticks one model keeps the device under `--co-sched rr`.
    /// Not 1 by default: models with different dynamic shared-memory requests
    /// force an SM carveout reconfiguration on every alternation (~150-300us).
    #[arg(
        long = "co-sched-quantum",
        env = "PLOW_CO_SCHED_QUANTUM",
        default_value_t = 4,
        global = true
    )]
    pub co_sched_quantum: u32,

    /// Directories under which `POST /v1/models/load` may take an assets dir.
    /// Repeatable; `PLOW_MODELS_ROOT` takes a `:`-separated list.
    ///
    /// This is a security boundary, not ergonomics: loading a bundle loads and
    /// EXECUTES its cubins/hsaco, so an unconstrained path in a request body is
    /// arbitrary code execution. Unset = only the assets dirs the process was
    /// started with (their parents) are reachable.
    #[arg(
        long = "models-root",
        env = "PLOW_MODELS_ROOT",
        value_delimiter = ':',
        global = true
    )]
    pub models_root: Vec<String>,

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

    /// Per-tick AMD serve breakdown: prefill launches, decode dispatch, host remainder
    /// (`PLOW_TICK_LOG=1`).
    #[arg(long = "tick-log", env = "PLOW_TICK_LOG", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub tick_log: bool,

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

    // ──────────────────────────────────────────────────────────────────────────
    // CPU engine (feature = "cpu")
    // ──────────────────────────────────────────────────────────────────────────
    #[command(flatten)]
    pub cpu: CpuRuntimeConfig,

    #[command(flatten)]
    pub apple: AppleRuntimeConfig,
}

#[derive(Args, Debug, Clone)]
#[command(next_help_heading = "Apple runtime (experimental)")]
pub struct AppleRuntimeConfig {
    /// Select CPU instead of Metal on a Metal-enabled serving build.
    #[arg(long = "apple-backend", env = "PLOW_BACKEND", global = true)]
    pub backend: Option<String>,
    /// Dispatch Metal instructions individually.
    #[arg(long = "metal-serial", env = "PLOW_METAL_SERIAL", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub serial: bool,
    /// Iterations a threadgroup spins on an unmet dependency before it faults the dispatch.
    /// This bounds how long a wrong packet can hold the GPU — and on Apple Silicon that GPU
    /// also draws the display, so the budget is a responsiveness knob, not just a timeout.
    /// Measured at 120 ns/iteration on an M4 Pro (16 threadgroups), so the default is ~0.5 s;
    /// raise it only if a legitimately slow producer starts faulting.
    #[arg(long = "metal-spin-max", env = "PLOW_METAL_SPIN_MAX", global = true)]
    pub spin_max: Option<u32>,
    /// CPU decode column share: percent[:instruction count].
    #[arg(long = "apple-cpu-share", env = "PLOW_CPU_SHARE", global = true)]
    pub cpu_share: Option<String>,
    /// Legacy per-instruction ANE selection: count, all, or program:instruction.
    #[arg(long = "apple-ane", env = "PLOW_ANE", global = true)]
    pub ane: Option<String>,
    /// Enable explicit v3 channel-MLP execution. Not a calibrated policy.
    #[arg(long = "ane-mlp", env = "PLOW_ANE_MLP", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub ane_mlp: bool,
    /// Offload only the first N planned layers; unset selects all planned layers.
    #[arg(long = "ane-mlp-layers", env = "PLOW_ANE_MLP_LAYERS", global = true)]
    pub ane_mlp_layers: Option<usize>,
    /// Built ane_placement probe required for channel graph admission.
    #[arg(
        long = "ane-mlp-placement",
        env = "PLOW_ANE_MLP_PLACEMENT",
        global = true
    )]
    pub ane_mlp_placement: Option<std::path::PathBuf>,
    /// Channel compiled-model cache directory.
    #[arg(long = "ane-mlp-cache", env = "PLOW_ANE_MLP_CACHE", global = true)]
    pub ane_mlp_cache: Option<std::path::PathBuf>,
    /// Inject before_submit, after_submit, or after_join channel failure.
    #[arg(
        long = "ane-mlp-fail",
        env = "PLOW_ANE_MLP_FAIL",
        hide = true,
        global = true
    )]
    pub ane_mlp_fail: Option<String>,
    /// Quantize legacy row-ANE weights to eight bits.
    #[arg(long = "ane-w8", env = "PLOW_ANE_W8", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub ane_w8: bool,
    /// Legacy row-ANE compute units override.
    #[arg(long = "ane-units", env = "PLOW_ANE_UNITS", global = true)]
    pub ane_units: Option<String>,
    /// Legacy row-ANE residual mode; fused keeps the addition in the graph.
    #[arg(long = "ane-resid", env = "PLOW_ANE_RESID", global = true)]
    pub ane_resid: Option<String>,
    /// Enable legacy CoreML output-range diagnostics when present.
    #[arg(long = "ane-out-range", env = "PLOW_OUT_RANGE", global = true)]
    pub out_range: Option<String>,
}

/// Kernel-tier ceiling for the CPU engine (`--cpu-isa`).
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[value(rename_all = "lowercase")]
pub enum CpuIsa {
    /// Highest tier cpuid reports.
    Auto,
    Amx,
    Avx512,
    Scalar,
}

/// CPU engine knobs. See `plans/cpu-backend.md` §2.1.
#[derive(Args, Debug, Clone)]
#[command(next_help_heading = "CPU runtime")]
pub struct CpuRuntimeConfig {
    /// Persistent workers. 0 = model-selected physical-core or logical-CPU width.
    #[arg(
        long = "cpu-threads",
        env = "PLOW_CPU_THREADS",
        default_value_t = 0,
        global = true
    )]
    pub threads: u32,

    /// NUMA: auto interleaves large tensors across allowed nodes (best effort);
    /// off keeps OS memory policy; a node list (0,1) requires successful placement.
    #[arg(
        long = "cpu-numa",
        env = "PLOW_CPU_NUMA",
        default_value = "auto",
        global = true
    )]
    pub numa: crate::exec::cpu::topology::NumaMode,

    /// Override transparent huge-page advice. By default, use ordinary pages
    /// for interleaved tensors and huge-page advice for single-node/OS placement.
    #[arg(long = "cpu-huge-pages", env = "PLOW_CPU_HUGE_PAGES", value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub huge_pages: Option<bool>,

    /// Kernel tier ceiling (for A/B and hosts without AMX).
    #[arg(
        long = "cpu-isa",
        env = "PLOW_CPU_ISA",
        default_value = "auto",
        value_enum,
        global = true
    )]
    pub isa: CpuIsa,

    /// Spin budget (µs) before a blocked worker yields/parks. Decode packets are
    /// 100-500 µs apart; parking on every gap measured +17% TPOT at 50 µs vs 1000.
    #[arg(
        long = "cpu-spin-us",
        env = "PLOW_CPU_SPIN_US",
        default_value_t = 2000,
        global = true
    )]
    pub spin_us: u32,

    /// Largest prefill chunk (rows) one tick may run while other slots decode. 0 = whole prompt.
    /// Measured NEGATIVE at concurrency >= 4 (chunks prefill slower than whole prompts and the
    /// threads are throughput-bound, not stall-bound), so it stays off by default.
    #[arg(
        long = "cpu-prefill-chunk",
        env = "PLOW_CPU_PF_CHUNK",
        default_value_t = 0,
        global = true
    )]
    pub prefill_chunk: u32,

    /// Directory holding the MXFP4 weight twin (`mxfp4/<name>` + `_scale`, quantize_mxfp4.py).
    #[arg(long = "cpu-mxfp4-dir", env = "PLOW_MXFP4_DIR", global = true)]
    pub mxfp4_dir: Option<String>,

    /// Opt in to the global work queue instead of static per-CU streams. Measured 2x slower on
    /// this box; kept for A/B on hosts where the static partition is a poor fit. Named `gq_opt_in`
    /// rather than `global_queue` because `AmdRuntimeConfig` already claims that clap id globally
    /// with a different type.
    #[arg(long = "cpu-global-queue", env = "PLOW_CPU_GQ", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub gq_opt_in: bool,

    /// Place executors by the packet's L2 locality domains instead of `cu % nodes`.
    ///
    /// OFF, because it measured 1.5x SLOWER on the one blob shape testable here (an H100-mapped
    /// Gemma-4-31B, blocked domain map) and never faster on any node count tried. A domain is a
    /// GPU L2 partition: it says which slices share a GPU cache, not which weights they touch, and
    /// CPU model tensors are interleaved across nodes anyway — so grouping by it buys no locality
    /// while costing the round-robin's balance. Kept for A/B on a host whose blob has balanced
    /// domains. Inert on a blob carrying none, and `node_plan` declines a plan that would leave
    /// any node busier than the round-robin even when this is on.
    #[arg(long = "cpu-l2-place", env = "PLOW_CPU_L2_PLACE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub l2_place: bool,
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

    /// VMM prefix reuse. Automatically enabled for eligible Hopper hybrid BF16-KV packets.
    #[arg(long = "vmm-prefix", env = "PLOW_VMM_PREFIX", value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub vmm_prefix: Option<bool>,

    /// Grow packet-described full KV backing with the live frontier, without prefix reuse.
    #[arg(long = "vmm-live", env = "PLOW_VMM_LIVE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub vmm_live: bool,

    /// Retain whole sliding-ring slots on first use; requires live KV without prefix reuse.
    #[arg(long = "vmm-live-rings", env = "PLOW_VMM_LIVE_RINGS", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub vmm_live_rings: bool,

    /// Direct upload path (CUDA). --no-nv-upload-direct to disable.
    #[arg(long = "nv-upload-direct", env = "PLOW_UPLOAD_DIRECT", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub upload_direct: bool,

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

    /// Accept an L2-placed blob on a backend that cannot VERIFY the interpreter honours the
    /// dispatch axis. NVIDIA only, and deliberately still `false`.
    ///
    /// AMD does not read this: `AmdEngine::load` parses with `l2_dispatch_ok = true` and then
    /// checks each code object for `plow_l2_place_dispatch_1`, which is strictly stronger than
    /// an operator assertion — a placed blob against an unplaced object is refused by
    /// inspection. Defaulting this to `true` would therefore buy AMD nothing and would remove
    /// the ONLY guard on the CUDA path, where nothing inspects the cubin. `plowc` places
    /// gfx942/gfx950 blobs by default and no other arch, so an NVIDIA blob is placed only when
    /// someone asked for it, and this flag is how they say the cubin can take it.
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

    /// Experimental BF16 TMA object: ABI1 M<=128; ABI2 Gemma12 M128 o/down; ABI3 Gemma31.
    #[arg(
        long = "pf-seg-gemm-small",
        env = "PLOW_PF_SEG_GEMM_SMALL",
        global = true
    )]
    pub pf_seg_gemm_small: Option<String>,

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

    /// Route exact packed Gemma HD256/GQA2 BF16 attention to its isolated object.
    #[arg(long = "pf-seg-fa256-gqa2", env = "PLOW_PF_SEG_FA256_GQA2", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub pf_seg_fa256_gqa2: bool,

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

impl RuntimeConfig {
    /// CUDA prefill interleave rows with "zero = unbounded" semantics.
    /// Unset → 2048; 0 → `usize::MAX` (no bound), else the configured value.
    pub fn pf_interleave_rows(&self) -> usize {
        match self.pf_interleave.unwrap_or(2048) {
            0 => usize::MAX,
            rows => rows as usize,
        }
    }

    /// AMD per-tick prefill row cap. Unset → 0, which `serve::mux::amd_prefill_tick_cap`
    /// reads as uncapped: one tick may admit up to the widest compiled prefill rung.
    pub fn pf_interleave_amd(&self) -> u32 {
        self.pf_interleave.unwrap_or(0)
    }

    /// Cross-request prefill packing on CUDA: off unless asked.
    pub fn pf_batch_cuda(&self) -> bool {
        self.pf_batch.unwrap_or(false)
    }

    /// Cross-request prefill packing on AMD: on unless `PLOW_PF_BATCH=0`.
    pub fn pf_batch_amd(&self) -> bool {
        self.pf_batch.unwrap_or(true)
    }

    /// Rotate AMD isolated admission across slots instead of serving the oldest request
    /// first. Only an explicit `PLOW_PF_BATCH=1` asks for it; the default is FCFS.
    pub fn pf_rotate(&self) -> bool {
        self.pf_batch == Some(true)
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

    /// Clear per-slot recurrent state with one device kernel per rank (default ON).
    #[arg(long = "amd-state-clear-device", env = "PLOW_STATE_CLEAR_DEVICE", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub state_clear_device: bool,

    /// Load the exact-capability MLA and KDA packed-prefill operator-family objects.
    ///
    /// Unset = automatic: on when the packet carries packed-prefill sibling or token-batch
    /// body programs, off otherwise. When on, a missing family object is a load error naming
    /// the file (never a silent fallback); `=0` is the rollback, `=1` forces the load on a
    /// packet without siblings.
    ///
    /// This flag does NOT gate dense/GQA co-packing, which needs no family object: the dense
    /// consumers are compiled into the ordinary prefill and flash objects
    /// (`PLOW_PACKED_PREFILL_DENSE_CONSUMERS=1` in `scripts/build_gfx942.sh`) and route through
    /// the same interpreter. Dense co-packing needs only `--pf-batch`, two concurrent prefills,
    /// and — the binding constraint in practice — a prefill chunk small enough that at least two
    /// of them fit in one compiled prefill rung. It works at TP1 and under TP alike; the TP
    /// engine has its own all-rank `prefill_packed_chunk`. See
    /// `docs/amd/gemma4-31b-mi300x.md`, "Dense packed prefill is unreachable at chunk 8192".
    #[arg(long = "amd-packed-prefill-route", env = "PLOW_PACKED_PREFILL_ROUTE", value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub packed_prefill_route: Option<bool>,

    /// Spill-isolated KDA-family object for ordinary prefill segments. Set false to disable.
    #[arg(long = "amd-kda-family-route", env = "PLOW_KDA_FAMILY_ROUTE", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub kda_family_route: bool,

    /// Global-queue scheduler. "0" = static per-block-stream, "1" = GQ.
    #[arg(long = "amd-global-queue", env = "PLOW_GLOBAL_QUEUE", global = true)]
    pub global_queue: Option<String>,

    /// Force the static scheduler: `both` (also `1`/`true`), `decode` or `prefill`. Unset
    /// keeps the global queue wherever the blob carries its appendix.
    #[arg(long = "amd-static", env = "PLOW_STATIC", value_parser = clap::builder::PossibleValuesParser::new(["both", "1", "true", "decode", "prefill"]), require_equals = true, num_args = 0..=1, default_missing_value = "both", global = true)]
    pub static_sched: Option<String>,

    /// Segment enqueue/drain windowing. --no-amd-seg-window to disable.
    #[arg(long = "amd-seg-window", env = "PLOW_SEG_WINDOW", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub seg_window: bool,

    /// Enqueue prefill segments in segment-major rank order and drain once per chunk.
    #[arg(long = "amd-tp-prefill-segment-major", env = "PLOW_TP_PREFILL_SEGMENT_MAJOR", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub tp_prefill_segment_major: bool,

    /// Per-segment all-rank barrier timing for prefill (diagnostic; disables segment-major).
    #[arg(long = "amd-prefill-seg-timing", env = "PLOW_PREFILL_SEG_TIMING", hide = true, default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub prefill_seg_timing: bool,

    /// Drain after every launch of the native sparse prefill routes (AITER sparse MLA, TP
    /// indexer) and print the per-kernel host time (diagnostic; serialises those segments).
    #[arg(long = "amd-native-launch-timing", env = "PLOW_NATIVE_LAUNCH_TIMING", hide = true, default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub native_launch_timing: bool,

    /// Directory to write token-batch device-state dumps into (body step and the ordinary
    /// prefill chunk, for row-by-row comparison). Diagnostic; unset = no dumps.
    #[arg(long = "amd-tb-dump", env = "PLOW_TB_DUMP", hide = true, global = true)]
    pub tb_dump: Option<std::path::PathBuf>,

    /// Write `--trace-raw` output for every TP rank (`<path>.rk<N>`), not only rank 0.
    #[arg(long = "amd-trace-allranks", env = "PLOW_TRACE_ALLRANKS", hide = true, default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub trace_allranks: bool,

    /// Persistent workgroup count for the f32-mix AttnRes object (sweep override).
    #[arg(
        long = "amd-attnres-f32mix-grid",
        env = "PLOW_ATTNRES_F32MIX_GRID",
        hide = true,
        global = true
    )]
    pub attnres_f32mix_grid: Option<u32>,

    /// Audited extra resident bytes per rank a replicated-input MoE EP packet may claim.
    #[arg(
        long = "amd-moe-prefill-ep-max-extra-bytes",
        env = "PLOW_MOE_PREFILL_EP_MAX_EXTRA_BYTES",
        hide = true,
        global = true
    )]
    pub moe_prefill_ep_max_extra_bytes: Option<u64>,

    /// Graph-derived spill-isolated prefill phase objects with one prebuilt AQL replay per rank.
    /// Default off until an exact full-network gate demonstrates a device-time win.
    #[arg(long = "amd-phase-objects", env = "PLOW_PHASE_OBJECTS", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub phase_objects: bool,

    /// VMM-backed KV on ROCr (opt-in, requires hsa_amd_vmem_*).
    #[arg(long = "amd-vmm-kv", env = "PLOW_VMM_KV", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub vmm_kv: bool,

    /// Map the KV block after a prefill chunk's last row while the chunk drains, so the
    /// decode that follows never maps on the engine thread (`PLOW_KV_MAP_AHEAD=0` disables).
    #[arg(long = "amd-kv-map-ahead", env = "PLOW_KV_MAP_AHEAD", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub kv_map_ahead: bool,

    /// Share completed MLA prefixes through ROCr VMM (auto on supported gfx942 packets).
    #[arg(long = "amd-shared-prefix", env = "PLOW_AMD_SHARED_PREFIX", value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub shared_prefix: Option<bool>,

    /// VMM block size for AMD KV (MiB).
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
    #[arg(long = "amd-tp-audit-compact", env = "PLOW_TP_AUDIT_COMPACT", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub tp_audit_compact: bool,

    /// Override prefill pad/launch-rows tradeoff.
    #[arg(long = "amd-launch-rows", env = "PLOW_LAUNCH_ROWS", global = true)]
    pub launch_rows: Option<u32>,

    /// Unified token batch: restrict bucket selection to one compiled row count.
    ///
    /// An ATTRIBUTION knob, not a tuning one. The route is confined to buckets with
    /// `nsplit == 1`, so for a given prompt it may run a different rung from the ordinary
    /// route — which means a differing greedy token has two candidate causes at once, the
    /// packing and the reduction order. Pinning the rung holds the packing fixed and moves
    /// only the second.
    #[arg(
        long = "amd-token-batch-rows",
        env = "PLOW_TOKEN_BATCH_ROWS",
        global = true
    )]
    pub token_batch_rows: Option<u32>,

    /// Unified token batch: admit a step with only ONE participant.
    ///
    /// Off by default. A prompt admitted alone runs a `nsplit == 1` rung far wider than the one
    /// the ordinary route would pick for it, and there is nothing to pack it with — measured on
    /// Gemma-4 31B at concurrency 1, -3.5% throughput and -35% TTFT at 512 input, -10.5% and
    /// -48.8% at 2048. The ordinary route already samples a completing prompt's last row in its
    /// own prefill program with no extra pass, so there is nothing to win there either. Kept as
    /// a flag so the policy stays falsifiable.
    #[arg(long = "amd-token-batch-solo", env = "PLOW_TOKEN_BATCH_SOLO", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub token_batch_solo: bool,

    /// Prior-context floor (rows) above which a request's DENSE final chunk is planned into
    /// the sparse (DSA) prefill bucket instead of the smallest bucket that holds it. Unset =
    /// off (the tail stays in the smallest bucket). See `serve::engine::retarget_dense_tail`.
    #[arg(long = "amd-tail-sparse-ctx", env = "PLOW_AMD_TAIL_SPARSE_CTX", global = true)]
    pub tail_sparse_ctx: Option<u32>,

    /// Unified token batch: keep the WIDE dense-GEMM rungs plowc chose per shape.
    ///
    /// The token-batch object is the mixed object's shape, and at four waves the fused-GLU
    /// epilogue's `SN == 2` pins `GM_BN` to 128 — so its plain `Gemm` body is one tile for every
    /// projection. The synthesizer collapses `GemmWide` (128x256) and `GemmC5` (192x256) onto
    /// `Gemm`, which throws away the per-shape choice plowc measured. With this on the two wide
    /// opcodes survive synthesis; they are separate instantiations already compiled into the
    /// object, and only the token-batch route may ask for them (the mixed route splits every
    /// projection into a decode GEMV band and a prefill GEMM band, which the wide arms have no
    /// form of).
    #[arg(long = "amd-token-batch-wide-tiles", env = "PLOW_TOKEN_BATCH_WIDE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub token_batch_wide_tiles: bool,

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

    /// Track the MLA decode's KV-split count from the LIVE `kv_len` instead of
    /// the `max_ctx` the emitter baked it from.
    ///
    /// `devgen::mla::glm_nsplit` sizes `nsplit` for the top of the range a blob
    /// serves, so one blob runs ONE split count at every context — a 32768-max-ctx
    /// server blob runs `ns=64` at live 1024 where the measured chain optimum is
    /// 16. The kernel takes `nsplit` as a runtime argument, so this patches the
    /// flash's and the merge's `i[4]` per step (in practice a handful of times per
    /// generation — the live value is a step function of `kv_len`) and changes no
    /// dispatch, buffer or counter. See `exec::kvrow::mla_live_nsplit`.
    ///
    /// Opt-in: a different partition reassociates the online-softmax merge, so
    /// this is a numerics-visible policy change and not a free win. It is inert on
    /// any packet that is not a plain dense MLA decode (the DSA gather arm splits
    /// over selected rows, not the KV window).
    #[arg(long = "mla-ns-live", env = "PLOW_MLA_NS_LIVE", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub mla_ns_live: bool,

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

    /// One-shot rank-0 capture: `T[@C0]:SEG:tensor=path[,tensor=path...]`.
    #[arg(
        long = "amd-pf-capture",
        env = "PLOW_PF_CAPTURE",
        hide = true,
        global = true
    )]
    pub pf_capture: Option<String>,

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

    /// Use the qualified gfx942 sparse MLA assembly object at isolated prefill boundaries.
    #[arg(long = "amd-mla-pf-aiter", env = "PLOW_MLA_PF_AITER", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub mla_pf_aiter: bool,

    /// Run native GLM MoE prefill rows >= 1024 on AITER's 64-row persistent tile
    /// (`..._psx_64x256.co`), the object its GLM-5 gfx942 tuning selects there.
    #[arg(long = "amd-moe-aiter-tile64", env = "PLOW_MOE_AITER_TILE64", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub moe_aiter_tile64: bool,

    /// Prequantize sorted MXFP4 MoE stage-1 activations once and reuse them across N tiles.
    #[arg(long = "amd-moe-stage1-a4-reuse", env = "PLOW_MOE_STAGE1_A4_REUSE", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set, require_equals = true, num_args = 0..=1, default_missing_value = "true", global = true)]
    pub moe_stage1_a4_reuse: bool,

    /// Download rank 0's copy of act tensors after the prefill/step:
    /// `name:path[,name:path...]`. A measurement instrument, not a serving path.
    #[arg(long = "amd-dump-act", env = "PLOW_DUMP_ACT", global = true)]
    pub dump_act: Option<String>,
}

/// Global runtime config, initialized once at startup from CLI parse.
static RUNTIME_CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

fn select_compat<T>(parsed: T, environment: Option<T>, allow_environment: bool) -> T {
    if allow_environment {
        environment.unwrap_or(parsed)
    } else {
        parsed
    }
}

impl RuntimeConfig {
    fn env_bool(var: &str) -> Option<bool> {
        let value = std::env::var(var).ok()?;
        Some(matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "y" | "on"
        ))
    }

    fn env_parse<T: std::str::FromStr>(var: &str) -> Option<T> {
        std::env::var(var).ok()?.parse().ok()
    }

    fn env_nonempty(var: &str) -> Option<String> {
        std::env::var(var).ok().filter(|value| !value.is_empty())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn nv_vmm_live(&self) -> bool {
        select_compat(
            self.nv.vmm_live,
            Self::env_bool("PLOW_VMM_LIVE"),
            !Self::is_initialized(),
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn nv_live_kv_enabled(
        &self,
        packed_prefill: bool,
        full_cache: bool,
        prefix: bool,
    ) -> bool {
        self.nv_vmm_live() || (packed_prefill && full_cache && !prefix)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn nv_vmm_live_rings(&self) -> bool {
        select_compat(
            self.nv.vmm_live_rings,
            Self::env_bool("PLOW_VMM_LIVE_RINGS"),
            !Self::is_initialized(),
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn nv_vmm_prefix(&self) -> Option<bool> {
        if !self.prefix_cache {
            return Some(false);
        }
        if Self::is_initialized() {
            self.nv.vmm_prefix
        } else {
            Self::env_bool("PLOW_VMM_PREFIX").or(self.nv.vmm_prefix)
        }
    }

    #[cfg(any(feature = "cuda", feature = "hsa"))]
    pub(crate) fn vmm_block_mib(&self) -> u32 {
        select_compat(
            self.vmm_block_mib,
            Self::env_parse("PLOW_VMM_BLOCK_MIB"),
            !Self::is_initialized(),
        )
    }

    /// Prefix-cache soft cap in bytes for a device with `device_bytes` of memory.
    /// An explicit `--vmm-cache-mib` wins; otherwise `--vmm-cache-memory-utilization`
    /// of the device. `device_bytes == 0` means the backend could not say, and the cap
    /// falls back to the 4 GiB the H100 campaign qualified rather than to "unbounded".
    pub(crate) fn prefix_cache_cap_bytes(&self, device_bytes: u64) -> u64 {
        let allow_env = !Self::is_initialized();
        let explicit: Option<u32> = if allow_env {
            Self::env_parse("PLOW_VMM_CACHE_MIB").or(self.vmm_cache_mib)
        } else {
            self.vmm_cache_mib
        };
        if let Some(mib) = explicit {
            return (mib as u64) << 20;
        }
        let fraction = select_compat(
            self.vmm_cache_memory_utilization,
            Self::env_parse("PLOW_VMM_CACHE_MEMORY_UTILIZATION"),
            allow_env,
        )
        .clamp(0.0, 1.0);
        if fraction == 0.0 {
            return 0;
        }
        if device_bytes == 0 {
            return 4096u64 << 20;
        }
        // Whole MiB, so the figure in logs and metrics reads like the knob.
        ((device_bytes as f64 * fraction) as u64) >> 20 << 20
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn nv_multistep(&self) -> u32 {
        select_compat(
            self.nv.multistep,
            Self::env_parse("PLOW_MULTISTEP"),
            !Self::is_initialized(),
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn nv_dev_sample(&self) -> Option<String> {
        select_compat(
            self.nv.dev_sample.clone(),
            Self::env_nonempty("PLOW_DEV_SAMPLE").map(Some),
            !Self::is_initialized(),
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn nv_cubin_sample(&self) -> Option<String> {
        select_compat(
            self.nv.cubin_sample.clone(),
            Self::env_nonempty("PLOW_NV_CUBIN_SAMPLE").map(Some),
            !Self::is_initialized(),
        )
    }

    pub(crate) fn slab_keep_override(&self) -> Option<bool> {
        select_compat(
            self.slab_keep,
            Self::env_bool("PLOW_SLAB_KEEP").map(Some),
            !Self::is_initialized(),
        )
    }

    pub(crate) fn kv_pool_mib(&self) -> u64 {
        select_compat(
            self.kv_pool_mib,
            Self::env_parse("PLOW_KV_POOL_MIB"),
            !Self::is_initialized(),
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn drain_timeout_ms(&self) -> Option<u64> {
        let environment = Self::env_parse("PLOW_DRAIN_TIMEOUT_MS").map(Some);
        select_compat(self.drain_timeout_ms, environment, !Self::is_initialized())
    }

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

/// Explicit runtime settings from the resolved CLI, in replayable environment spelling.
pub fn serve_replay(m: &clap::ArgMatches) -> std::collections::BTreeMap<String, String> {
    use clap::Args;
    let cmd = RuntimeConfig::augment_args(clap::Command::new("plowrt"));
    let mut out = std::collections::BTreeMap::new();
    for arg in cmd.get_arguments() {
        let id = arg.get_id().as_str();
        if !matches!(
            m.value_source(id),
            Some(clap::parser::ValueSource::EnvVariable | clap::parser::ValueSource::CommandLine)
        ) {
            continue;
        }
        let Some(env) = arg.get_env().map(|e| e.to_string_lossy().into_owned()) else {
            continue;
        };
        let val = m
            .get_raw(id)
            .map(|v| {
                v.map(|s| s.to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        out.insert(env, val);
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn token_batch_defaults_on_with_explicit_rollback() {
        use clap::{Args, FromArgMatches};
        let command = super::RuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "token_batch")
            .unwrap();
        assert_eq!(arg.get_default_values(), ["true"]);
        for (flag, enabled) in [("--token-batch", true), ("--token-batch=false", false)] {
            let matches = command
                .clone()
                .try_get_matches_from(["test", flag])
                .unwrap();
            assert_eq!(
                super::RuntimeConfig::from_arg_matches(&matches)
                    .unwrap()
                    .token_batch,
                enabled
            );
        }
    }

    #[test]
    fn fusion_is_an_opt_in_runtime_flag() {
        use clap::{Args, FromArgMatches};
        let command = super::RuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "fusion")
            .unwrap();
        assert_eq!(arg.get_default_values(), ["false"]);
        for (flag, enabled) in [("--fusion", true), ("--fusion=false", false)] {
            let matches = command
                .clone()
                .try_get_matches_from(["test", flag])
                .unwrap();
            assert_eq!(
                super::RuntimeConfig::from_arg_matches(&matches)
                    .unwrap()
                    .fusion,
                enabled
            );
        }
    }

    #[test]
    fn compatibility_overrides_apply_only_without_initialized_cli() {
        assert_eq!(super::select_compat(true, Some(false), false), true);
        assert_eq!(super::select_compat(true, Some(false), true), false);
        assert_eq!(
            super::select_compat(Some(7_u64), Some(None), false),
            Some(7)
        );
        assert_eq!(super::select_compat(Some(7_u64), None, true), Some(7));
    }

    #[test]
    fn shared_prefix_and_token_batch_defaults_allow_independent_rollback() {
        use clap::{Args, FromArgMatches};
        let command = super::RuntimeConfig::augment_args(clap::Command::new("test"));
        for field in ["prefix_cache", "token_batch"] {
            assert_eq!(
                command
                    .get_arguments()
                    .find(|arg| arg.get_id() == field)
                    .unwrap()
                    .get_default_values(),
                ["true"]
            );
        }
        let matches = command
            .try_get_matches_from([
                "test",
                "--prefix-cache=false",
                "--vmm-prefix=true",
                "--vmm-cache-mib=512",
            ])
            .unwrap();
        let config = super::RuntimeConfig::from_arg_matches(&matches).unwrap();
        assert!(!config.prefix_cache);
        assert!(config.token_batch);
        assert_eq!(config.vmm_cache_mib, Some(512));
        // Explicit MiB wins over the percentage, whatever the device size.
        assert_eq!(config.prefix_cache_cap_bytes(192 << 30), 512 << 20);
        #[cfg(feature = "cuda")]
        assert_eq!(config.nv_vmm_prefix(), Some(false));
    }

    #[test]
    fn prefix_cache_defaults_to_auto_with_a_bounded_budget_and_explicit_overrides() {
        use clap::{Args, FromArgMatches};
        let command = super::RuntimeConfig::augment_args(clap::Command::new("test"));
        let prefix = command
            .get_arguments()
            .find(|arg| arg.get_id() == "vmm_prefix")
            .unwrap();
        assert!(prefix.get_default_values().is_empty());
        let budget = command
            .get_arguments()
            .find(|arg| arg.get_id() == "vmm_cache_mib")
            .unwrap();
        assert!(budget.get_default_values().is_empty());
        let fraction = command
            .get_arguments()
            .find(|arg| arg.get_id() == "vmm_cache_memory_utilization")
            .unwrap();
        assert_eq!(fraction.get_default_values(), ["0.05"]);
        // The default scales with the device: 5% of an 80 GiB H100 is the 4 GiB the
        // campaign qualified; an MI300X gets 9.6 GiB; an unknown size keeps 4 GiB.
        let matches = command.clone().try_get_matches_from(["test"]).unwrap();
        let config = super::RuntimeConfig::from_arg_matches(&matches).unwrap();
        assert_eq!(config.prefix_cache_cap_bytes(80 << 30), 4096 << 20);
        assert_eq!(config.prefix_cache_cap_bytes(192 << 30), 9830 << 20);
        assert_eq!(config.prefix_cache_cap_bytes(0), 4096 << 20);
        let matches = command
            .clone()
            .try_get_matches_from(["test", "--vmm-cache-memory-utilization=0"])
            .unwrap();
        let config = super::RuntimeConfig::from_arg_matches(&matches).unwrap();
        assert_eq!(config.prefix_cache_cap_bytes(80 << 30), 0);
        for (flag, expected) in [("--vmm-prefix", true), ("--vmm-prefix=false", false)] {
            let matches = command
                .clone()
                .try_get_matches_from(["test", flag])
                .unwrap();
            assert_eq!(
                super::RuntimeConfig::from_arg_matches(&matches)
                    .unwrap()
                    .nv
                    .vmm_prefix,
                Some(expected)
            );
        }
    }

    #[test]
    fn amd_shared_prefix_defaults_to_auto_and_accepts_explicit_overrides() {
        use clap::{Args, FromArgMatches};
        let command = super::RuntimeConfig::augment_args(clap::Command::new("test"));
        assert!(command.get_arguments().find(|arg| arg.get_id() == "shared_prefix")
            .unwrap().get_default_values().is_empty());
        for (flag, expected) in [("--amd-shared-prefix", true), ("--amd-shared-prefix=false", false)] {
            let matches = command.clone().try_get_matches_from(["test", flag]).unwrap();
            assert_eq!(super::RuntimeConfig::from_arg_matches(&matches).unwrap().amd.shared_prefix,
                Some(expected));
        }
    }

    #[test]
    fn apple_channel_defaults_off_and_has_explicit_cli_overrides() {
        use clap::{Args, FromArgMatches};
        let command = super::AppleRuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = command
            .get_arguments()
            .find(|a| a.get_id() == "ane_mlp")
            .unwrap();
        assert_eq!(arg.get_default_values(), ["false"]);
        for (flag, enabled) in [("--ane-mlp", true), ("--ane-mlp=false", false)] {
            let matches = command
                .clone()
                .try_get_matches_from([
                    "test",
                    flag,
                    "--ane-mlp-layers",
                    "2",
                    "--ane-mlp-placement",
                    "/tmp/probe",
                ])
                .unwrap();
            let config = super::AppleRuntimeConfig::from_arg_matches(&matches).unwrap();
            assert_eq!(config.ane_mlp, enabled);
            assert_eq!(config.ane_mlp_layers, Some(2));
            assert_eq!(
                config.ane_mlp_placement.as_deref(),
                Some(std::path::Path::new("/tmp/probe"))
            );
        }
    }

    #[test]
    fn live_kv_defaults_off_and_can_be_enabled_without_prefix_reuse() {
        use clap::{Args, FromArgMatches};
        let command = super::NvidiaRuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "vmm_live")
            .unwrap();
        assert_eq!(arg.get_default_values(), ["false"]);
        let matches = command
            .try_get_matches_from(["test", "--vmm-live=true", "--vmm-prefix=false"])
            .unwrap();
        let config = super::NvidiaRuntimeConfig::from_arg_matches(&matches).unwrap();
        assert!(config.vmm_live);
        assert_eq!(config.vmm_prefix, Some(false));
        assert!(!config.vmm_live_rings);
        let command = super::NvidiaRuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "vmm_live_rings")
            .unwrap();
        assert_eq!(arg.get_default_values(), ["false"]);
        let matches = command
            .try_get_matches_from(["test", "--vmm-live=true", "--vmm-live-rings=true"])
            .unwrap();
        assert!(
            super::NvidiaRuntimeConfig::from_arg_matches(&matches)
                .unwrap()
                .vmm_live_rings
        );
    }

    #[test]
    fn device_state_clear_defaults_on_and_has_a_false_rollback() {
        use clap::{Args, FromArgMatches};

        let command = super::AmdRuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "state_clear_device")
            .expect("state-clear argument");
        assert_eq!(arg.get_default_values(), ["true"]);

        let matches = command
            .try_get_matches_from(["test", "--amd-state-clear-device=false"])
            .expect("explicit state-clear rollback");
        let config = super::AmdRuntimeConfig::from_arg_matches(&matches).expect("AMD config");
        assert!(!config.state_clear_device);
    }

    #[test]
    fn tp_prefill_segment_major_defaults_on_and_has_a_false_rollback() {
        use clap::{Args, FromArgMatches};

        let command = super::AmdRuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "tp_prefill_segment_major")
            .expect("TP prefill segment-major argument");
        assert_eq!(arg.get_default_values(), ["true"]);

        let matches = command
            .try_get_matches_from(["test", "--amd-tp-prefill-segment-major=false"])
            .expect("explicit TP prefill segment-major rollback");
        let config = super::AmdRuntimeConfig::from_arg_matches(&matches).expect("AMD config");
        assert!(!config.tp_prefill_segment_major);
    }

    #[test]
    fn packed_prefill_route_and_pf_batch_default_to_auto_with_explicit_rollback() {
        use clap::{Args, FromArgMatches};

        // UNSET IS AUTOMATIC, NOT OFF. The route follows the packet: `exec/amd.rs` arms it when
        // the blob carries packed-prefill siblings or token-batch bodies and refuses by object
        // name when a family object is missing. `=false` is the rollback; `=true` forces the
        // load on a packet without siblings (the legacy opt-in).
        let command = super::AmdRuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "packed_prefill_route")
            .expect("packed-prefill route argument");
        assert!(arg.get_default_values().is_empty());
        for (flag, want) in [
            ("--amd-packed-prefill-route=true", Some(true)),
            ("--amd-packed-prefill-route=false", Some(false)),
        ] {
            let matches = command
                .clone()
                .try_get_matches_from(["test", flag])
                .expect("explicit packed-prefill route");
            assert_eq!(
                super::AmdRuntimeConfig::from_arg_matches(&matches)
                    .expect("AMD config")
                    .packed_prefill_route,
                want
            );
        }
        let matches = command
            .clone()
            .try_get_matches_from(["test"])
            .expect("unset packed-prefill route");
        assert_eq!(
            super::AmdRuntimeConfig::from_arg_matches(&matches)
                .expect("AMD config")
                .packed_prefill_route,
            None
        );

        // The KDA family object is a SEPARATE axis and defaults ON: it doubles as the
        // spill-isolation object for ordinary KDA prefill segments, which needs no packing.
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "kda_family_route")
            .expect("KDA family route argument");
        assert_eq!(arg.get_default_values(), ["true"]);

        // `--pf-batch` is the shared mux half: unset resolves per vendor (AMD on, CUDA off),
        // an explicit `=1` additionally rotates isolated admission, `=0` is the rollback.
        let shared = super::RuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = shared
            .get_arguments()
            .find(|arg| arg.get_id() == "pf_batch")
            .expect("pf-batch argument");
        assert!(arg.get_default_values().is_empty());
        let cfg = |args: &[&str]| {
            let matches = shared
                .clone()
                .try_get_matches_from(args)
                .expect("pf-batch args");
            super::RuntimeConfig::from_arg_matches(&matches).expect("runtime config")
        };
        let unset = cfg(&["test"]);
        assert!(unset.pf_batch_amd() && !unset.pf_batch_cuda() && !unset.pf_rotate());
        let on = cfg(&["test", "--pf-batch=true"]);
        assert!(on.pf_batch_amd() && on.pf_batch_cuda() && on.pf_rotate());
        let off = cfg(&["test", "--pf-batch=false"]);
        assert!(!off.pf_batch_amd() && !off.pf_batch_cuda() && !off.pf_rotate());

        // The step budget: unset is the vendor default (CUDA 2048 rows, AMD uncapped = the
        // widest compiled rung); an explicit value clamps both.
        assert_eq!(unset.pf_interleave_rows(), 2048);
        assert_eq!(unset.pf_interleave_amd(), 0);
        let capped = cfg(&["test", "--pf-interleave=512"]);
        assert_eq!(capped.pf_interleave_rows(), 512);
        assert_eq!(capped.pf_interleave_amd(), 512);
        assert_eq!(cfg(&["test", "--pf-interleave=0"]).pf_interleave_rows(), usize::MAX);
    }

    #[test]
    fn phase_objects_default_off_and_have_an_explicit_opt_in() {
        use clap::{Args, FromArgMatches};

        let command = super::AmdRuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "phase_objects")
            .expect("phase-object argument");
        assert_eq!(arg.get_default_values(), ["false"]);

        let matches = command
            .try_get_matches_from(["test", "--amd-phase-objects=true"])
            .expect("explicit phase-object opt-in");
        let config = super::AmdRuntimeConfig::from_arg_matches(&matches).expect("AMD config");
        assert!(config.phase_objects);
    }

    #[test]
    fn moe_stage1_a4_reuse_defaults_on_and_has_a_false_rollback() {
        use clap::{Args, FromArgMatches};

        let command = super::AmdRuntimeConfig::augment_args(clap::Command::new("test"));
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "moe_stage1_a4_reuse")
            .expect("MoE stage-1 A4 reuse argument");
        assert_eq!(arg.get_default_values(), ["true"]);

        let matches = command
            .try_get_matches_from(["test", "--amd-moe-stage1-a4-reuse=false"])
            .expect("explicit MoE stage-1 A4 reuse rollback");
        let config = super::AmdRuntimeConfig::from_arg_matches(&matches).expect("AMD config");
        assert!(!config.moe_stage1_a4_reuse);
    }

    /// A `RuntimeConfig` field that nothing reads is a CLI flag that silently does nothing.
    ///
    /// This is not hypothetical and it is why the test exists: `amd.trace_raw` was parsed here,
    /// carried a `--trace-raw` flag, and had NO reader anywhere — while `main.rs` and
    /// `exec/amd.rs` read `PLOW_TRACE_RAW` through `env::var_os` in four places. The env var
    /// worked, so nothing looked broken; the flag was decoration. Same duplicated-parse shape as
    /// the `PLOW_XR_CUS` defect, and `devgen::emit_config` already carries the twin of this test.
    ///
    /// Coarse on purpose — "does any plowrt source mention `.field`". A
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
            "pub struct AppleRuntimeConfig {",
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
        assert!(!others.is_empty(), "no runtime sources readable");
        let readers = format!("{me}\n{others}");

        let dead: Vec<&String> = fields
            .iter()
            // `amd`/`nvidia` are the sub-struct handles; reads go through them as `.amd.x`.
            .filter(|f| !readers.contains(&format!(".{f}")))
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
            // Whole-file test modules (`exec/amd/tests.rs`, `*_tests.rs`) are `#[cfg(test)]`
            // at their `mod` declaration, so the inline marker below never appears in them;
            // their fixture reads (`PLOW_TEST_*`) are exempt for the same reason as below.
            if path.file_name().is_some_and(|n| {
                let n = n.to_string_lossy();
                n == "tests.rs" || n.ends_with("_tests.rs")
            }) {
                continue;
            }
            // A `#[cfg(test)] mod tests` reads the environment to locate FIXTURES, not to
            // configure the runtime: `PLOW_DSA_VERIFY_CKPT` points an `#[ignore]`d test at a
            // 200 GiB checkpoint and has no business in `--help`. The rule this guard enforces
            // is about knobs a SERVE honours, so it stops at the test module.
            //
            // Both lines are required, and the attribute alone is not enough: a bare
            // `#[cfg(test)]` on a single item would otherwise blind the scan for the rest of
            // the file. Test modules are last and are spelled this way throughout the crate.
            let lines: Vec<&str> = text.lines().collect();
            let cut = lines
                .windows(2)
                .position(|w| {
                    w[0].trim() == "#[cfg(test)]" && w[1].trim_start().starts_with("mod tests")
                })
                .unwrap_or(lines.len());
            for (i, line) in lines[..cut].iter().enumerate() {
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
        files(dir).into_iter().map(|(_, t)| t).collect()
    }
}

//! `plowc` — the plow compiler driver.
//!
//! Resolves a network (a HuggingFace model id or a [`net::NetConfig`]) and a
//! hardware spec into per-shape-bucket runtime packet streams, dumps the shared
//! weight/KV layout, and reports compiler-pass statistics + runtime estimates.
//! The whole pipeline already lives in the workspace crates; this is the thin
//! driver that wires them and writes the artifacts:
//!
//! ```text
//! build graph ─▶ plan_from_all_blocks ─┐
//!   net config ─────────────────────────┤
//!                                       ▼
//!     assemble(soc) ─▶ compile_buckets ─▶ emit_program ─▶ *.pkt + weights.json
//!
//!            └▶ rewrite (fuse) ─▶ RewriteStats     [ANALYSIS ONLY — dropped]
//! ```
//!
//! **`rewrite` is not in the emit path.** `plan_from_all_blocks` lowers the
//! RAW `nn_graph::Graph`; the `FusedGraph` is computed for its statistics and
//! discarded (see the call site below). `plan_from_fused` — the only bridge
//! that consumes a fused term — has no production caller, and the devblob
//! emitter (`devgen`) has no dependency on `rewrite` at all, so **no egglog
//! rewrite reaches a GPU**. Measured coverage on Gemma-4-12B: 0 of 1156 ops
//! and 0 of 24,226 GFLOP. See `perf-data/px18-egglog-wholemodel.md`.
//!
//! HuggingFace models bake **every** transformer block into one plan so
//! `assemble` can chain fine-grained tile-dependencies across block boundaries
//! (`plan_from_all_blocks`). This unlocks cross-block tile pipelining and
//! transparently supports heterogeneous block types (Gemma 4 sliding/full mix,
//! DeepSeek dense-then-MoE, DiT stages). Implicit barriers only remain between
//! distinct compiled programs (voice pipeline, DiT denoising steps).
//!
//! ## Scope (intentional extension points)
//! - **Parallelism** ([`Parallel`]): tensor-parallel (N-split across GPUs) is
//!   wired; data / pipeline / expert parallel are recognized and rejected with a
//!   clear message — the hooks are here, the planning is future work.
//! - **Multi-network workflows** (voice flow: ASR → LLM → TTS): [`compile`]
//!   produces one [`Report`] per network; a workflow is a `Vec<Report>` chained
//!   under a latency budget. Not built yet — see the module docs.
//! - **Lean verification** ([`Options::lean_verify`]): when set, every bucket
//!   is submitted to `plow_verify` (the Lean 4 CLI) for all six checkpoints
//!   A–F. Every dispatcher is backed by a proven universal theorem; a
//!   rejection fails the compile with `PlowcError::LeanVerify`. Requires
//!   `--features lean-verify`. See the design notes.

pub mod hf_config;
pub mod net;
pub mod parallel;
#[cfg(feature = "tuner")]
pub mod tune;
#[cfg(feature = "tuner")]
pub mod tuned;

use std::path::PathBuf;

use clap::ValueEnum;
use costmodel::{hwspec, Soc};
use rewrite::{plan_from_all_blocks, LayerPlan, RewriteStats};
use schedule::{
    compile_buckets_tuned, emit_program, Compiled, Config, KvLayout, Phase, ShapeBucket,
    WeightLayout,
};
use serde::Serialize;
use tracing::{debug, info, trace, warn};

// Re-export shared schema types from plow-asset. These are the single source
// of truth for the compiler→runtime JSON boundary. plowc constructs them;
// plowrt deserializes them.
use plow_asset::{
    BlockRange, Blocks, BufClass, DecodeFlashOp, DecodeKvSchema, ExpertLayer, Experts, Fusion,
    KvLayerPaging, KvPaging, KvSummary, MemEntry, MemoryMap, RequestIo, RequestIoField,
    SharedExpert, StaticTensorEntry, WeightTiling,
};

/// How the work is spread across `--num-gpus`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Parallel {
    /// Tensor-parallel: split each GEMM along N across the GPUs (wired today).
    Tp,
    /// Data-parallel: replicate the model, shard the batch (not yet implemented).
    Dp,
    /// Pipeline-parallel: split layers across GPUs (not yet implemented).
    Pp,
    /// Expert-parallel: shard MoE experts across GPUs (not yet implemented).
    Ep,
}

/// Where the network definition comes from.
pub enum Source {
    /// A plow-native network JSON (offline).
    Net(net::NetConfig),
    /// A HuggingFace model id, resolved over the network.
    Model(String),
    /// A local HuggingFace model directory (config.json → single-block synthesis).
    HfDir(PathBuf),
}

impl Source {
    pub fn name(&self) -> String {
        match self {
            Source::Net(n) => n.name.clone(),
            Source::Model(id) if std::path::Path::new(id).is_dir() => {
                hf_config::dir_slug(std::path::Path::new(id))
            }
            Source::Model(id) => id.clone(),
            Source::HfDir(p) => hf_config::dir_slug(p),
        }
    }
}

/// Everything the driver needs besides the network itself.
pub struct Options {
    pub gpu: String,
    /// Skip the tuning system entirely and rank tiles with the analytical cost
    /// model alone.
    ///
    /// The escape hatch for when a probe is wrong, a toolchain is missing, or a
    /// build must be reproduced exactly as it was before the tuner existed.
    /// `--no-tuning` sets this.
    pub no_tuning: bool,
    /// Tuning database root. `None` disables measurement lookup while leaving
    /// the capability filter active.
    pub tuning_db: Option<std::path::PathBuf>,
    pub num_gpus: usize,
    pub parallel: Parallel,
    pub batches: Vec<i64>,
    pub seqs: Vec<i64>,
    pub phases: Vec<Phase>,
    /// SRAM page size in KiB.
    pub page_kib: u64,
    pub out: PathBuf,
    /// If set, run each bucket's `(schedule, address_map)` through the Lean
    /// verifier CLI (`plow_verify`, see `crates/lean_verify`). Rejection fails
    /// the compile with [`PlowcError::LeanVerify`]. The binary is located via
    /// `PLOW_VERIFY_BIN` or `lean-plow/.lake/build/bin/plow_verify`.
    pub lean_verify: bool,
    /// Run the §8.1 counter-elimination pass on every bucket. Drops counters
    /// that are already covered by resource-order — provably safe by the
    /// `resourceOrdered ⊆ happensBefore` theorem in `Plow.Protocol`. When
    /// combined with `lean_verify`, the reduced schedule is also verified.
    pub counter_elim: bool,
    /// Run the §8.2 scope-narrowing pass. Downgrades `Scope::IntraGpu`
    /// counters whose actual runtime placement is entirely on one SM to
    /// `Scope::IntraSm` — swaps L2 atomic for shared-mem barrier at
    /// runtime (~40 cycles / counter check).
    pub scope_narrow: bool,
    /// Run the §8.3 prefetch-hoisting pass. Reorders each resource stream so
    /// every `TaskKind::DmaIn` sits right after its last stream-local
    /// predecessor — closes the makespan/ideal_makespan gap on memory-bound
    /// workloads.
    pub prefetch: bool,
    /// Run the §8.5 SRAM temporal-fit pass. Phase 1 identifies hand-offs
    /// relax demoted to HBM/DSM that could have stayed same-SM under temporal
    /// disjointness; Phase 2 **mutates the compiled buckets** — it promotes
    /// accepted hand-offs back to SramSameSm and reschedules, so output with
    /// this flag differs from output without it
    /// (see the design notes).
    pub sram_fit: bool,
    /// Enable the Lean performance oracle. When `true`, the scheduler queries
    /// the `plow_verify` binary for provably-optimal decisions (counter
    /// granularity, prefetch depth, lower-bound certificates). Falls back to
    /// Rust heuristics if the binary is unavailable.
    pub lean_oracle: bool,
    /// §P Emit a host-executor `SAMPLE` packet at the tail of every **decode**
    /// bucket: a `Body::Token` gated on the output-stage counter that turns the
    /// logits into a token id on the host (or an SM). Decode-phase only —
    /// prefill produces no per-token draw. Off by default.
    pub emit_sample: bool,
    /// §P Emit a host `TOKENIZE` packet at the head of the graph (text/ids →
    /// the `tokens` buffer), gating the first op. Off by default.
    pub emit_tokenize: bool,
    /// Emit a Chrome Trace Event Format JSON (`{stem}.trace.json`) per
    /// bucket — timeline view of every scheduled task on its resource lane.
    /// Loadable in `chrome://tracing` or `ui.perfetto.dev`.
    pub emit_trace: bool,
    /// KV cache paging config. Tokens per KV block — the compiled address map
    /// reserves an initial block count and reports the layout; the runtime
    /// grows the KV region by allocating additional blocks as sequences
    /// extend past the pre-reserved range. See `KvConfig`.
    pub kv: KvConfig,
    /// Override weight dtype for GEMM projections in the HfDir path.
    /// `None` = auto-detect from config.json's `torch_dtype` / quantization_config.
    /// `Some(dt)` forces that dtype on all GEMM weight operands (norms stay BF16).
    pub weight_dtype_override: Option<nn_graph::DType>,
}

/// KV cache configuration. Layered on top of the schedule crate's `KvLayout`
/// (which is derived from the model's Flash attention shape); this struct
/// carries the runtime-facing paging policy.
#[derive(Clone, Copy, Debug)]
pub struct KvConfig {
    /// Tokens per KV block (a.k.a. page size in vLLM terminology). Powers of
    /// two are typical; the compiler picks 256 by default.
    pub block_tokens: i64,
    /// Initial number of blocks reserved in the arena for the KV cache. The
    /// runtime allocates further blocks past the reserved region as sequences
    /// grow. Sized to cover the largest bucket's prefill by default.
    pub initial_blocks: i64,
}

impl Default for KvConfig {
    fn default() -> Self {
        KvConfig {
            block_tokens: 256,
            initial_blocks: 0,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum PlowcError {
    #[error("unknown GPU spec `{0}` (see `hwspec::registry::ALL`)")]
    UnknownGpu(String),
    #[error("{0}")]
    Parallelism(String),
    #[error("no shape buckets to compile (need at least one batch, seq, and phase)")]
    NoBuckets,
    #[error("invalid dimension: {0}")]
    InvalidDim(String),
    #[error(transparent)]
    Hub(#[from] nn_graph::hub::HubError),
    #[error(transparent)]
    Rewrite(#[from] rewrite::RewriteError),
    #[error(transparent)]
    Bridge(#[from] rewrite::BridgeError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("lean verifier rejected bucket `{bucket}`: {reason}")]
    LeanVerify { bucket: String, reason: String },
    /// Wraps subprocess/marshaling errors from `lean_verify::VerifyError` as
    /// a string so the enum stays feature-independent (`source` as a field
    /// name would trigger thiserror's automatic source-error handling and
    /// require a real `Error`-implementing type).
    #[error("lean verifier subprocess failed for bucket `{bucket}`: {detail}")]
    LeanVerifySpawn { bucket: String, detail: String },
    #[error("`lean_verify: true` was set, but plowc was built without --features lean-verify")]
    LeanVerifyDisabled,
    /// The `; rule: <name>` annotations in `rules.egg` are malformed — a
    /// rewrite without a name (or a dangling annotation) cannot be submitted
    /// to checkpoint A, so the compile fails instead of verifying a stale
    /// hardcoded catalog.
    #[error("rewrite rule catalog error: {0}")]
    RuleCatalog(String),
    /// The compiled address map's device-0 (or per-device) segment exceeds
    /// the GPU's HBM capacity. Emitted per bucket + per device the moment
    /// the plan is built, so the compile fails loudly instead of producing
    /// artifacts that would OOM at load time.
    #[error(
        "bucket `{bucket}` needs {needed} bytes on device {device} but the GPU has only {capacity} bytes of HBM ({overshoot} bytes short)"
    )]
    HbmOverflow {
        bucket: String,
        device: u8,
        needed: u64,
        capacity: u64,
        overshoot: u64,
    },
}

// --- Report (also the `weights.json` manifest) ---

/// Type aliases for plow-asset types used in the Report.
pub type LayoutReport = WeightLayout;
pub type KvReport = KvSummary;
pub type FusionReport = Fusion;

/// Type alias: plowc constructs `BucketStat` directly as `BucketReport`.
pub type BucketReport = plow_asset::BucketStat;

/// Type alias: plowc constructs `Experts` directly as `ExpertsSchema`.
pub type ExpertsSchema = Experts;

/// Type alias: plowc constructs `Blocks` directly as `BlocksSchema`.
pub type BlocksSchema = Blocks;

/// Type alias: plowc constructs `RequestIo` directly as `RequestIoSchema`.
pub type RequestIoSchema = RequestIo;

/// Type alias: plowc constructs `KvPaging` directly as `KvPagingReport`.
pub type KvPagingReport = KvPaging;

/// Type alias: the compiled address map serialized to `*.map.json`.
pub type MemoryReport = MemoryMap;

/// Roll-up of "what does this model+setting need at load and run time?"
/// Emitted as `assets.json` alongside `weights.json`.
///
/// `settings` mirrors the CLI opts so an artifact dir is self-describing;
/// `regions` reports the *peak* HBM demand across every compiled bucket
/// (the runtime allocates the union, not the sum, since a request picks one
/// bucket); `on_disk` reports every artifact plowc wrote so you can plan
/// downstream storage / distribution.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Assets {
    pub network: String,
    pub settings: SettingsSummary,
    pub regions: MemoryRegions,
    pub on_disk: OnDiskAssets,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct SettingsSummary {
    pub gpu: String,
    pub num_gpus: usize,
    pub parallel: String,
    pub page_kib: u64,
    pub batches: Vec<i64>,
    pub seqs: Vec<i64>,
    pub phases: Vec<String>,
    pub prefetch: bool,
    pub counter_elim: bool,
    pub scope_narrow: bool,
    pub sram_fit: bool,
}

/// Peak HBM usage per region across the compiled bucket ladder. Weights
/// are shared across buckets (one copy), so `weights` is the same for
/// every bucket. KV / scratch / arena are max-across-buckets (a live run
/// picks one bucket at a time).
#[derive(Clone, Debug, serde::Serialize)]
pub struct MemoryRegions {
    pub weights: u64,
    pub kv_cache_peak: u64,
    pub scratch_peak: u64,
    pub request_io_peak: u64,
    pub static_: u64,
    pub persistent: u64,
    pub arena_peak: u64,
    pub total_hbm_peak: u64,
    pub hbm_capacity: u64,
    /// `hbm_capacity - total_hbm_peak`. Negative would have failed the
    /// OOM guard already, so this is always ≥ 0 here.
    pub hbm_headroom: u64,
    /// Bucket that drives the peak — the one you should size against.
    pub peak_bucket: PeakBucket,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PeakBucket {
    pub phase: String,
    pub batch: i64,
    pub seq: i64,
}

/// Sizes of every file plowc wrote into `--out`, grouped by kind.
#[derive(Clone, Debug, serde::Serialize)]
pub struct OnDiskAssets {
    pub packets_total: u64,
    pub map_json_total: u64,
    pub blocks_json_total: u64,
    pub experts_json_total: u64,
    pub decode_kv_json_total: u64,
    pub request_io_json_total: u64,
    pub trace_json_total: u64,
    pub weights_json: u64,
    pub footprint_json: u64,
    pub footprint_csv: u64,
    pub static_tensors_bin: u64,
    /// FP8 weight-to-scale bindings emitted for quantized `--model` inputs.
    pub fp8_weights_json: u64,
    /// Output-local HF config/index/tokenizer/chat files plus
    /// `hf_metadata.json`; zero for non-metadata compiles.
    pub hf_metadata_total: u64,
    pub grand_total: u64,
}

/// Per-bucket memory-footprint breakdown, emitted alongside the packet
/// streams as `footprint.json` (+ a `footprint.csv` for spreadsheets).
///
/// Every field is bytes on the target device. The classes are disjoint,
/// so `weight + kv_cache + scratch + request_io + static_ + persistent`
/// approximates HBM usage (packets stream from host and don't add to HBM
/// footprint, but knowing their size matters for compile+load).
#[derive(Clone, Debug, serde::Serialize)]
pub struct Footprint {
    pub phase: String,
    pub batch: i64,
    pub seq: i64,
    /// Sum of GEMM weight tensors (N·K·elem_bytes). Runtime loads these
    /// from safetensors — not part of the address-map arena, but they
    /// live in HBM.
    pub weight_bytes: u64,
    /// KV cache reserved bytes (BufClass::Growable). Initial reservation
    /// covers this bucket's prefill; runtime extends past it as sequences
    /// grow.
    pub kv_cache_bytes: u64,
    /// Intermediate activation storage (BufClass::Scratch) — reused via
    /// liveness intervals, so the reported bytes are the packed peak, not
    /// the sum of individual tiles.
    pub scratch_bytes: u64,
    /// Per-request input/output buffers the host marshals each iteration
    /// (BufClass::RequestIo).
    pub request_io_bytes: u64,
    /// Compile-time static tensors (BufClass::Static): RoPE freq tables,
    /// masks. Same lifetime as weights but sourced from
    /// `static_tensors.bin`.
    pub static_bytes: u64,
    /// Persistent buffers placed directly in the arena (BufClass::Persistent).
    /// Currently 0 — weights live outside the arena.
    pub persistent_bytes: u64,
    /// Compiled packet stream size (host-side .pkt file). Not in HBM.
    pub packet_bytes: u64,
    /// Total arena footprint (kv + scratch + request_io + static + persistent).
    pub arena_bytes: u64,
    /// Full HBM occupancy for this bucket = arena + weights. Matches the
    /// value the OOM guard checks against `spec.mem.capacity`.
    pub total_hbm_bytes: u64,
}

/// Narrow the `kind`/`access` of `RequestIo`-class map entries from the
/// `RequestIo` sidecar's per-field `direction`: an `"output"` field (logits,
/// sampled tokens) is `BufKind::Output` / `Access::Write` (kernel writes, host
/// reads); everything else stays the class default `Input` / `ReadWrite`. This
/// is the direction-aware refinement of the class-derived tags set in
/// `build_memory_report`.
fn refine_request_io_kinds(mem: &mut MemoryReport, request_io: &RequestIo) {
    use plow_asset::{Access, BufClass, BufKind};
    for f in &request_io.fields {
        if f.direction != "output" {
            continue;
        }
        if let Some(e) = mem.entries.iter_mut().find(|e| e.name == f.name) {
            if e.class == BufClass::RequestIo {
                e.kind = Some(BufKind::Output);
                e.access = Some(Access::Write);
            }
        }
    }
}

fn build_memory_report(
    m: &schedule::AddressMap,
    kv_paging: Option<KvPagingReport>,
    weight_shapes: &std::collections::HashMap<String, (i64, i64, nn_graph::DType)>,
) -> MemoryReport {
    MemoryReport {
        arena_bytes: m.arena_bytes,
        growable_base: m.growable_base,
        segments: m.segments.clone(),
        entries: m
            .entries
            .iter()
            .map(|e| {
                let (logical_shape, dtype) = if e.class == BufClass::Persistent {
                    weight_shapes
                        .get(&e.name)
                        .map(|&(n, k, dtype)| (Some(vec![n, k]), Some(dtype.to_string())))
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };
                MemEntry {
                    slot: e.slot,
                    name: e.name.clone(),
                    class: e.class,
                    offset: e.offset,
                    reserved: e.reserved,
                    growable: e.growable,
                    device: e.device,
                    logical_shape,
                    dtype,
                    // Data-type / access tags derived from the lifetime class.
                    // (RequestIo input-vs-output refinement via the sidecar
                    // direction is a follow-up; the class default suffices here.)
                    kind: Some(e.class.default_kind()),
                    access: Some(e.class.default_access()),
                }
            })
            .collect(),
        kv_paging,
    }
}

/// The compile result for one network — also serialized to `weights.json`.
#[derive(Serialize, Debug)]
pub struct Report {
    pub network: String,
    pub gpu: String,
    /// Calibration tier the tile selection actually rested on, and why.
    ///
    /// Recorded so a bundle can say what evidence it was built from rather than
    /// leaving a reader to assume it was measured.
    pub tuning_tier: String,
    pub tuning_provenance: String,
    /// Did the Lean ordering certificate actually cover EVERY bucket here?
    ///
    /// Same job as `tuning_tier` beside it, for the same reason. The gate is on
    /// by default and degrades when there is no `plow_verify`, so "skipped" is
    /// a normal outcome — and a skipped gate is indistinguishable from a passed
    /// one unless the artifact records which. `false` never means "rejected": a
    /// rejection fails the compile, so no bundle describing one exists.
    #[serde(default)]
    pub lean_verified: bool,
    /// Why `lean_verified` reads the way it does, in words.
    #[serde(default)]
    pub lean_provenance: String,
    pub num_gpus: usize,
    pub parallel: Parallel,
    /// `true` iff one `(BN, BK)` is legal across every bucket — a flip moves no
    /// weight bytes.
    pub weight_shared: bool,
    pub weight: Option<LayoutReport>,
    pub kv: Option<KvReport>,
    /// Operator-fusion stats (model front-end only).
    pub fusion: Option<FusionReport>,
    pub buckets: Vec<BucketReport>,
    /// Static tensors emitted alongside the packet streams. Empty until
    /// Phase 5 of the design notes materializes RoPE freq
    /// tables + attention masks.
    pub static_tensors: Vec<StaticTensorEntry>,
    /// Whether `static_tensors.bin` was written next to `weights.json`.
    /// False when `static_tensors` is empty (no file needed).
    pub static_tensors_file_emitted: bool,
    /// Weight-tiling byte-layout spec — present iff `weight_shared`.
    /// The runtime uses this to arrange safetensor bytes into every
    /// `Persistent`-class GEMM-weight entry in the address map.
    /// See the design notes.
    pub weight_tiling: Option<WeightTiling>,
    /// Per-model+setting sizing summary. Persisted separately as
    /// `assets.json` (see `build_assets`) and skipped from `weights.json`
    /// so the manifest keeps its historical shape.
    #[serde(skip)]
    pub assets: Option<Assets>,
}

/// Whether the Lean ordering certificate covered this whole compile, and why.
/// Carried out of `emit_streams` (which is where the per-bucket verification
/// happens) into [`Report`].
struct LeanStatus {
    verified: bool,
    provenance: String,
}

fn phase_name(p: Phase) -> &'static str {
    match p {
        Phase::Prefill => "prefill",
        Phase::Decode => "decode",
    }
}

fn fp8_scale_storage_bytes(graph: &nn_graph::Graph) -> u64 {
    let scales = graph
        .checkpoint_manifest()
        .into_iter()
        .map(|weight| (weight.name, weight))
        .collect::<std::collections::HashMap<_, _>>();
    graph
        .fp8_scale_bindings
        .iter()
        .filter_map(|binding| scales.get(binding.scale.as_str()))
        .filter_map(|scale| {
            let dims = scale
                .shape?
                .dims()
                .iter()
                .map(|dim| dim.as_static())
                .collect::<Option<Vec<_>>>()?;
            let elements = dims
                .iter()
                .fold(1u64, |count, &dim| count.saturating_mul(dim.max(0) as u64));
            Some(scale.dtype.tile_bytes(elements))
        })
        .sum()
}

/// Compile `src` for `opts`, writing one `.pkt` stream per bucket plus a
/// `weights.json` manifest into `opts.out`, and return the [`Report`].
pub fn compile(src: &Source, opts: &Options) -> Result<Report, PlowcError> {
    let started = std::time::Instant::now();
    info!(
        network = %src.name(), target = %opts.gpu, output = %opts.out.display(),
        "compilation started"
    );
    debug!(stage = "frontend", "resolving and validating source");
    // Resolve Hub metadata once. This downloads only the compiler allowlist
    // (config, safetensors index, tokenizer/chat assets), never weight shards.
    // Every bucket builds from this same local snapshot.
    let model_metadata = match src {
        Source::Model(id) => {
            info!(
                stage = "model-resolution", model = %id,
                "resolving Hugging Face compiler metadata"
            );
            Some(nn_graph::hub::resolve_model_metadata(id)?)
        }
        // An indexed local directory contains the same compiler metadata as a
        // Hub snapshot. Lower it through the same frontend so source access
        // (`--model` vs `--hf-dir`) cannot change the emitted program.
        Source::HfDir(dir) if dir.join("model.safetensors.index.json").is_file() => {
            info!(
                stage = "model-resolution", path = %dir.display(),
                "resolving indexed Hugging Face compiler metadata"
            );
            Some(nn_graph::hub::resolve_model_metadata(
                dir.to_string_lossy().as_ref(),
            )?)
        }
        Source::Net(_) | Source::HfDir(_) => None,
    };
    if let Some(metadata) = &model_metadata {
        ensure_model_packet_path_supported(metadata)?;
    }
    let model_checkpoint_graph = model_metadata
        .as_ref()
        .map(|metadata| {
            nn_graph::hub::build_from_metadata(metadata, &nn_graph::models::ShapeBucket::default())
        })
        .transpose()?;
    if let (Some(metadata), Some(graph)) = (&model_metadata, &model_checkpoint_graph) {
        metadata.validate_checkpoint_manifest(graph)?;
        info!(
            stage = "checkpoint-validation",
            tensors = graph.checkpoint_manifest().len(),
            "safetensors index matches the compiled text graph"
        );
    }
    let fp8_scale_bytes = model_checkpoint_graph
        .as_ref()
        .map(fp8_scale_storage_bytes)
        .unwrap_or(0);
    // For HfDir sources, synthesize the full model metadata once (cheap JSON
    // parse) and cache it for all downstream functions. No resolution to
    // Source::Net — we use `build_full_model_plan()` directly.
    let hf_synth: Option<hf_config::HfSynthesis> = match src {
        Source::HfDir(p) if model_metadata.is_none() => {
            info!(
                stage = "model-resolution", path = %p.display(),
                "resolving model from HuggingFace directory"
            );
            let mut synth = hf_config::synthesize_from_hf_dir(p).map_err(PlowcError::InvalidDim)?;
            // CLI --weight-dtype override takes precedence over config.json auto-detection.
            if let Some(dt) = opts.weight_dtype_override {
                synth.weight_dtype = dt;
            }
            Some(synth)
        }
        Source::Net(_) | Source::Model(_) | Source::HfDir(_) => None,
    };

    let spec = hwspec::registry::lookup(&opts.gpu)
        .ok_or_else(|| PlowcError::UnknownGpu(opts.gpu.clone()))?;
    let soc = build_soc(spec, opts)?;
    let buckets = make_buckets(opts);
    if buckets.is_empty() {
        return Err(PlowcError::NoBuckets);
    }
    // Reject degenerate shapes up front — a clean error beats a panic deep in
    // `assemble` / the cost model when a dim is 0 or negative.
    if opts.page_kib == 0 {
        return Err(PlowcError::InvalidDim("page_kib must be > 0".into()));
    }
    for b in &buckets {
        if b.batch <= 0 || b.seq <= 0 {
            return Err(PlowcError::InvalidDim(format!(
                "bucket batch={} seq={}: both must be > 0",
                b.batch, b.seq
            )));
        }
    }
    if let Source::Net(n) = src {
        n.validate().map_err(PlowcError::InvalidDim)?;
    }
    // Validate the synthesized net for HfDir too
    if let Some(synth) = &hf_synth {
        hf_config::ensure_layer_plan_supported(synth).map_err(PlowcError::InvalidDim)?;
        synth.net.validate().map_err(PlowcError::InvalidDim)?;
    }
    info!(
        stage = "frontend", buckets = buckets.len(), gpus = opts.num_gpus,
        parallel = ?opts.parallel, "source and target validated"
    );

    // Pre-build every bucket's plan (fallible: model resolution / lowering may
    // fail). `compile_buckets` then pulls the prebuilt plan, infallibly.
    info!(
        stage = "planning",
        buckets = buckets.len(),
        "lowering bucket plans"
    );
    let mut plans: Vec<(ShapeBucket, LayerPlan)> = Vec::with_capacity(buckets.len());
    let mut fusion: Option<FusionReport> = None;
    for b in &buckets {
        debug!(
            stage = "planning",
            phase = phase_name(b.phase),
            batch = b.batch,
            seq = b.seq,
            "lowering bucket"
        );
        // Egg exploration only for the first bucket: the report keeps only the
        // first bucket's stats (below), and the saturation result is otherwise
        // unused — re-running it per bucket multiplied peak memory for nothing.
        let (plan, st) = build_plan(
            src,
            b,
            hf_synth.as_ref(),
            model_metadata.as_ref(),
            fusion.is_none(),
        )?;
        trace!(
            stage = "planning",
            ops = plan.ops.len(),
            "bucket plan lowered"
        );
        if let (None, Some(s)) = (&fusion, st) {
            fusion = Some(FusionReport {
                ops_before: s.ops_before,
                ops_after: s.ops_after,
                fused: s.fused,
            });
        }
        plans.push((*b, plan));
    }
    info!(
        stage = "planning",
        plans = plans.len(),
        "bucket plans lowered"
    );
    // HfDir: hard-verify the (bucket-invariant) weight coverage of the plan
    // against the safetensors actually in the directory — both directions.
    // A silently-absent (or silently-uncovered) weight is the worst failure
    // mode in this stack; see hf_config::validate_against_checkpoint.
    if let (Source::HfDir(dir), Some(synth), Some((_, plan))) =
        (src, hf_synth.as_ref(), plans.first())
    {
        // Config-only checkpoint: with no safetensors there is nothing to
        // cross-check the plan against (and the runtime cannot bind weights),
        // but a structural compile — same use the devblob path already allows —
        // is legitimate for comparing packets/counters. Skip the gate with a
        // warning rather than erroring, matching `devgen::layer_scalars`.
        let has_shards = std::fs::read_dir(dir).is_ok_and(|rd| {
            rd.filter_map(Result::ok).any(|e| {
                let p = e.path();
                p.extension().and_then(|x| x.to_str()) == Some("safetensors")
                    || p.to_string_lossy().ends_with(".partial.safetensors")
            })
        });
        if has_shards {
            hf_config::validate_against_checkpoint(dir, plan, synth)
                .map_err(PlowcError::InvalidDim)?;
            info!(
                stage = "checkpoint-validation", prefix = %synth.prefix,
                "checkpoint weights match the lowered plan"
            );
        } else {
            warn!(
                stage = "checkpoint-validation", prefix = %synth.prefix,
                "no .safetensors in {} — skipping weight-coverage gate (structural \
                 compile; weights are unbound and numerics are not representative)",
                dir.display()
            );
        }
    }
    let lookup_plan = |b: &ShapeBucket| {
        plans
            .iter()
            .find(|(bb, _)| bb == b)
            .map(|(_, p)| p.clone())
            .expect("every compiled bucket was prebuilt")
    };

    let cfg = Config {
        lean_oracle: opts.lean_oracle,
        ..Config::default()
    };

    // The kernel oracle: capability filter from the probed interpreter, plus
    // qualified measurements where they match this build. `--no-tuning` selects
    // the analytical-only path, which is byte-identical to the compiler before
    // the tuner existed.
    #[cfg(feature = "tuner")]
    let oracle: Box<dyn rewrite::oracle::KernelOracle> = if opts.no_tuning {
        Box::new(tuned::CompilerOracle::disabled())
    } else {
        let hw = hwspec::registry::lookup(&opts.gpu)
            .and_then(kernelcaps::HardwareFingerprint::from_spec);
        match hw {
            Some(hw) => Box::new(tuned::CompilerOracle::new(
                std::path::Path::new("."),
                &hw,
                opts.tuning_db.as_ref(),
            )),
            None => Box::new(tuned::CompilerOracle::disabled()),
        }
    };
    #[cfg(not(feature = "tuner"))]
    let oracle: Box<dyn rewrite::oracle::KernelOracle> = {
        if !opts.no_tuning {
            info!(
                stage = "scheduling",
                "built without the `tuner` feature: analytical cost model only"
            );
        }
        Box::new(rewrite::oracle::NoOracle)
    };
    info!(
        stage = "scheduling",
        buckets = buckets.len(),
        "scheduling buckets"
    );
    let mut compiled = compile_buckets_tuned(&soc, &cfg, &buckets, lookup_plan, oracle.as_ref());
    info!(
        stage = "scheduling",
        streams = compiled.streams.len(),
        "buckets scheduled"
    );

    // Read provenance AFTER compiling: the oracle records tile mismatches while
    // candidates are being priced, so sampling it earlier reports a clean build
    // that was not clean.
    let tuning_tier = oracle.tier().to_string();
    let tuning_provenance = oracle.provenance();

    // §8.5 Phase 2 — promote temporally-fit hand-offs back to SramSameSm and
    // reschedule affected buckets. Provably safe by
    // `Plow.Sram.occupancy_le_of_temporal_fit`; the reschedule then makes
    // the SRAM residency real (packet emitter sees resident flags,
    // list_schedule sees the new colocation groups).
    if opts.sram_fit {
        info!(
            stage = "optimization",
            pass = "sram-fit",
            "running post-schedule pass"
        );
        apply_sram_fit_phase2(&mut compiled, &soc, &cfg);
    }

    info!(stage = "emission", output = %opts.out.display(), "writing compiled artifacts");
    std::fs::create_dir_all(&opts.out)?;
    if let Some(metadata) = &model_metadata {
        metadata.copy_to(&opts.out)?;
        let graph = model_checkpoint_graph
            .as_ref()
            .expect("resolved model metadata has a checkpoint graph");
        if !graph.fp8_scale_bindings.is_empty() {
            let bindings = graph
                .fp8_scale_bindings
                .iter()
                .map(|binding| {
                    serde_json::json!({
                        "weight": binding.weight,
                        "scale": binding.scale,
                        "block_shape": binding.block_shape,
                    })
                })
                .collect::<Vec<_>>();
            std::fs::write(
                opts.out.join("fp8_weights.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "format": "e4m3",
                    "activation_scheme": "dynamic",
                    "scale_dtype": "f32",
                    "total_scale_bytes": fp8_scale_bytes,
                    "bindings": bindings,
                }))?,
            )?;
        } else {
            let stale = opts.out.join("fp8_weights.json");
            if stale.is_file() {
                std::fs::remove_file(stale)?;
            }
        }
        info!(
            stage = "model-metadata",
            files = metadata.filenames().count(),
            "copied Hugging Face metadata beside packet output"
        );
    }
    let (buckets, footprints, lean) =
        emit_streams(&compiled, &soc, &cfg, opts, src, &plans, fp8_scale_bytes)?;

    // Static tensors: compile-time constants (RoPE freq tables, static
    // masks). Phase 4 emits the plumbing with an empty manifest — Phase 5
    // materializes the actual bytes. Written only when non-empty so runtime
    // can rely on `static_tensors_file_emitted` as a presence flag.
    let (static_tensors, static_bin) = collect_static_tensors();
    let static_tensors_file_emitted = !static_bin.is_empty();
    if static_tensors_file_emitted {
        std::fs::write(opts.out.join("static_tensors.bin"), &static_bin)?;
    }

    let gemm_weight_dtypes: std::collections::HashSet<_> = plans
        .iter()
        .flat_map(|(_, plan)| &plan.ops)
        .filter(|op| matches!(op.kind, rewrite::OpKind::Gemm(_)))
        .map(|op| op.weight_dtype)
        .collect();
    let homogeneous_weight_dtype = if gemm_weight_dtypes.len() == 1 {
        gemm_weight_dtypes.iter().next().copied()
    } else {
        None
    };
    let weight_tiling = compiled
        .weight
        .filter(|_| compiled.weight_shared)
        .zip(homogeneous_weight_dtype)
        .and_then(|(w, dtype)| {
            dtype.byte_size().map(|elem_bytes| WeightTiling {
                bn: w.bn,
                bk: w.bk,
                element_dtype: dtype.to_string(),
                elem_bytes,
                block_iteration: "n_major_k_inner".into(),
                within_block_layout: "n_outer_k_inner".into(),
                padding_policy: "zero_extend".into(),
            })
        });

    let mut report = Report {
        tuning_tier,
        tuning_provenance,
        lean_verified: lean.verified,
        lean_provenance: lean.provenance,
        network: src.name(),
        gpu: opts.gpu.clone(),
        num_gpus: opts.num_gpus,
        parallel: opts.parallel,
        weight_shared: compiled.weight_shared,
        weight: compiled
            .weight
            .map(|w: WeightLayout| LayoutReport { bn: w.bn, bk: w.bk }),
        kv: compiled.kv.map(|k: KvLayout| KvReport {
            block_seq: k.block_seq,
            kv_heads: k.kv_heads,
            head_dim: k.head_dim,
        }),
        fusion,
        buckets,
        static_tensors,
        static_tensors_file_emitted,
        weight_tiling,
        assets: None,
    };
    std::fs::write(
        opts.out.join("weights.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    trace!(
        stage = "manifest",
        file = "weights.json",
        "artifact written"
    );

    // Model+setting summary: peak HBM per region + on-disk asset sizes.
    // Written last so `on_disk.weights_json` reflects the file just above.
    let assets = build_assets(&report, opts, spec.mem.capacity.0, &footprints);
    std::fs::write(
        opts.out.join("assets.json"),
        serde_json::to_string_pretty(&assets)?,
    )?;
    trace!(stage = "manifest", file = "assets.json", "artifact written");
    report.assets = Some(assets);
    info!(
        stage = "complete", buckets = report.buckets.len(),
        elapsed_ms = started.elapsed().as_millis(), output = %opts.out.display(),
        "compilation completed"
    );
    Ok(report)
}

fn ensure_model_packet_path_supported(
    metadata: &nn_graph::hub::ModelMetadata,
) -> Result<(), PlowcError> {
    let config: serde_json::Value = serde_json::from_str(metadata.config_json())?;
    let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    let routed_experts = config
        .get("n_routed_experts")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    match model_type {
        Some("kimi_k3" | "kimi_linear") => Err(PlowcError::InvalidDim(
            "Kimi-K3 is supported by the dedicated MI355X devblob emitter, but the \
             metadata-only `--model` scheduled-packet path does not yet lower its AttnRes \
             block-residual state; use the existing Kimi HF-directory/devblob path"
                .into(),
        )),
        Some("deepseek" | "deepseek_v2" | "deepseek_v3") if routed_experts > 1 => {
            Err(PlowcError::InvalidDim(
                "DeepSeek MoE scheduled packets currently model only one representative routed \
                 expert and do not bind the complete expert/FP8-scale manifest; refusing an \
                 incomplete packet"
                    .into(),
            ))
        }
        Some("kimi" | "kimi_k2" | "moonshot") if routed_experts > 1 => Err(PlowcError::InvalidDim(
            "Kimi-K2 MoE scheduled packets currently model only one representative routed \
                 expert and do not bind the complete expert/FP8-scale manifest; refusing an \
                 incomplete packet"
                .into(),
        )),
        Some("glm_moe_dsa") if routed_experts > 1 => Err(PlowcError::InvalidDim(
            "GLM-5.3 scheduled packets do not yet bind the official DSA indexer, every routed \
             expert, FP8 scales, and next-token layer; refusing an incomplete packet"
                .into(),
        )),
        _ => Ok(()),
    }
}

fn build_assets(
    report: &Report,
    opts: &Options,
    hbm_capacity: u64,
    footprints: &[Footprint],
) -> Assets {
    let weights = footprints.first().map(|f| f.weight_bytes).unwrap_or(0);
    let mut peak_idx = 0usize;
    let mut peak_total = 0u64;
    let mut kv_peak = 0u64;
    let mut scratch_peak = 0u64;
    let mut req_peak = 0u64;
    let mut arena_peak = 0u64;
    let mut static_ = 0u64;
    let mut persistent = 0u64;
    for (i, f) in footprints.iter().enumerate() {
        if f.total_hbm_bytes > peak_total {
            peak_total = f.total_hbm_bytes;
            peak_idx = i;
        }
        kv_peak = kv_peak.max(f.kv_cache_bytes);
        scratch_peak = scratch_peak.max(f.scratch_bytes);
        req_peak = req_peak.max(f.request_io_bytes);
        arena_peak = arena_peak.max(f.arena_bytes);
        // Static/persistent are load-time-fixed; same across buckets.
        static_ = static_.max(f.static_bytes);
        persistent = persistent.max(f.persistent_bytes);
    }
    let peak_bucket = footprints
        .get(peak_idx)
        .map(|f| PeakBucket {
            phase: f.phase.clone(),
            batch: f.batch,
            seq: f.seq,
        })
        .unwrap_or(PeakBucket {
            phase: String::new(),
            batch: 0,
            seq: 0,
        });
    let hbm_headroom = hbm_capacity.saturating_sub(peak_total);

    let file_size = |name: &str| -> u64 {
        std::fs::metadata(opts.out.join(name))
            .map(|m| m.len())
            .unwrap_or(0)
    };
    // Sum only files this compile emitted (derived from the report's bucket
    // stems) — a directory scan would also count stale artifacts left in
    // `--out` by prior runs with different ladders.
    let stems: Vec<String> = report
        .buckets
        .iter()
        .map(|b| b.packet_file.trim_end_matches(".pkt").to_string())
        .collect();
    let sum_size = |suffix: &str| -> u64 {
        stems
            .iter()
            .map(|s| file_size(&format!("{s}{suffix}")))
            .sum()
    };

    let packets_total = sum_size(".pkt");
    let map_json_total = sum_size(".map.json");
    let blocks_json_total = sum_size(".blocks.json");
    let experts_json_total = sum_size(".experts.json");
    let decode_kv_json_total = sum_size(".decode_kv.json");
    let request_io_json_total = sum_size(".request_io.json");
    let trace_json_total = sum_size(".trace.json");
    let weights_json = file_size("weights.json");
    let footprint_json = file_size("footprint.json");
    let footprint_csv = file_size("footprint.csv");
    let static_tensors_bin = file_size("static_tensors.bin");
    let fp8_weights_json = file_size("fp8_weights.json");
    let chat_templates_total = std::fs::read_dir(opts.out.join("chat_templates"))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| entry.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .map(|metadata| metadata.len())
        .sum::<u64>();
    let hf_metadata_total = nn_graph::hub::METADATA_FILES
        .iter()
        .map(|name| file_size(name))
        .sum::<u64>()
        .saturating_add(chat_templates_total)
        .saturating_add(file_size("hf_metadata.json"));
    // Excludes assets.json itself: this total is serialized into that file,
    // so its own size isn't known yet.
    let grand_total = packets_total
        + map_json_total
        + blocks_json_total
        + experts_json_total
        + decode_kv_json_total
        + request_io_json_total
        + trace_json_total
        + weights_json
        + footprint_json
        + footprint_csv
        + static_tensors_bin
        + fp8_weights_json
        + hf_metadata_total;

    let phase_str = |p: Phase| match p {
        Phase::Prefill => "prefill".to_string(),
        Phase::Decode => "decode".to_string(),
    };
    Assets {
        network: report.network.clone(),
        settings: SettingsSummary {
            gpu: opts.gpu.clone(),
            num_gpus: opts.num_gpus,
            parallel: format!("{:?}", opts.parallel).to_lowercase(),
            page_kib: opts.page_kib,
            batches: opts.batches.clone(),
            seqs: opts.seqs.clone(),
            phases: opts.phases.iter().copied().map(phase_str).collect(),
            prefetch: opts.prefetch,
            counter_elim: opts.counter_elim,
            scope_narrow: opts.scope_narrow,
            sram_fit: opts.sram_fit,
        },
        regions: MemoryRegions {
            weights,
            kv_cache_peak: kv_peak,
            scratch_peak,
            request_io_peak: req_peak,
            static_,
            persistent,
            arena_peak,
            total_hbm_peak: peak_total,
            hbm_capacity,
            hbm_headroom,
            peak_bucket,
        },
        on_disk: OnDiskAssets {
            packets_total,
            map_json_total,
            blocks_json_total,
            experts_json_total,
            decode_kv_json_total,
            request_io_json_total,
            trace_json_total,
            weights_json,
            footprint_json,
            footprint_csv,
            static_tensors_bin,
            fp8_weights_json,
            hf_metadata_total,
            grand_total,
        },
    }
}

/// Phase 4 stub: no static tensors materialized yet. Phase 5
/// computes RoPE freq tables + causal
/// masks and returns them here for emission.
fn collect_static_tensors() -> (Vec<StaticTensorEntry>, Vec<u8>) {
    (Vec::new(), Vec::new())
}

/// Sentinel `expert_id` value the SM checks against to detect "this slot is
/// unused" and skip the compute (only firing the completion counter). Baked
/// into every routed layer's runtime contract via `experts.json`.
pub const EXPERT_UNUSED_SENTINEL: u32 = u32::MAX;

/// Unique Flash operations in graph order, with their source-layer id.
/// Classification comes from the lowered semantic kind, never display names.
fn flash_ops(tasks: &schedule::TaskGraph, cons: &rewrite::ConstraintSet) -> Vec<(String, u32)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for task in &tasks.tasks {
        if task.kind != schedule::TaskKind::Compute
            || !matches!(
                cons.op_io.get(&task.node).map(|d| d.kind),
                Some(rewrite::OpKind::Flash(_))
            )
            || !seen.insert(task.op.clone())
        {
            continue;
        }
        let layer = task
            .op
            .rsplit_once("_L")
            .and_then(|(_, suffix)| suffix.parse::<u32>().ok())
            .unwrap_or(out.len() as u32);
        out.push((task.op.clone(), layer));
    }
    out
}

/// Build the decode-phase KV read address sidecar. Returns `None` for
/// non-decode buckets. See the design notes.
fn build_decode_kv_schema(
    bucket: &ShapeBucket,
    tasks: &schedule::TaskGraph,
    cons: &rewrite::ConstraintSet,
    kv_paging: Option<&KvPagingReport>,
) -> Option<DecodeKvSchema> {
    use schedule::Phase;
    if bucket.phase != Phase::Decode {
        return None;
    }
    let kv = kv_paging?;
    // Attribute each Flash op to a layer index by scan-order occurrence.
    // Real HF models with `plan_from_all_blocks` carry a `_L{i}` suffix on
    // the op name; use that when available, else fall back to scan-order.
    // Dedupe by op name — a single Flash op tiles into many tasks; we emit
    // one sidecar entry per op, not per tile.
    let mut flash_ops = Vec::new();
    for (op_name, layer_idx) in self::flash_ops(tasks, cons) {
        let buffer_name = kv
            .per_layer
            .iter()
            .find(|p| p.layer_idx == layer_idx)
            .map(|p| p.buffer_name.clone())
            .unwrap_or_else(|| format!("kv_cache_L{layer_idx}"));
        flash_ops.push(DecodeFlashOp {
            op_name,
            layer_idx,
            kv_buffer_name: buffer_name,
            // `FlashBody { coord0: u32, coord1: u32, seq_q: u32, seq_kv: u32, ... }`
            // — `seq_kv` sits at byte offset 12.
            seq_kv_field_offset: 12,
        });
    }
    Some(DecodeKvSchema {
        flash_ops,
        past_len_buffer: "past_len".into(),
        block_tokens: kv.block_tokens,
    })
}

/// Detect MoE layers + shared experts in a compiled bucket. Returns an empty
/// schema on `--net` sources (which model MoE as sequential GEMMs and don't
/// emit any `MoeRouter` op). See the design notes for the full
/// design; this is Phase 1 (detection + sidecar only).
fn build_experts_schema(tasks: &schedule::TaskGraph, _src: &Source) -> ExpertsSchema {
    let mut layers = Vec::new();
    let mut shared = Vec::new();
    for task in &tasks.tasks {
        // Extract `_L{i}` block suffix once for both branches.
        let block = task
            .op
            .rsplit("_L")
            .next()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        if task.op.starts_with("moe_router") {
            // Routed MoE layer. Phase 2 will plumb num_experts / top_k from
            // OpKind::MoeRouter into the Task struct; for now the router op
            // name lets the runtime derive both from the router weight shape.
            layers.push(ExpertLayer {
                block,
                layer_label: format!("layers.{block}.block_sparse_moe"),
                num_experts: 0,
                top_k: 0,
                router_op_name: task.op.clone(),
                routing_table_slot: String::new(),
                expert_weight_table_slot: String::new(),
                // Phase 2 will plumb the per-expert weight names here (from the graph's
                // routed-expert leaves), which resolve_expert_tables turns into the flat
                // expert_weight_table. Empty until then — backward-compatible sidecar.
                routed_experts: Vec::new(),
            });
        } else if task.op.contains("shared_expert") {
            // Shared expert — a dense FFN every token traverses. Naming
            // convention scan; more robust detection lands with Phase 3.
            shared.push(SharedExpert {
                block,
                layer_label: format!("layers.{block}.shared_expert"),
                gate_up_weight: None,
                down_weight: None,
                replicated_across_gpus: false,
            });
        }
    }
    ExpertsSchema {
        layers,
        shared,
        expert_unused_sentinel: EXPERT_UNUSED_SENTINEL,
        // Detection is exhaustive — we scanned every task. `complete` is
        // always true; the empty `layers` case on `--net` sources means
        // "no MoE layers exist", not "we couldn't tell".
        complete: true,
    }
}

/// Build the per-block task-range schema by scanning each task's op-name
/// suffix `_L{block}`. Returns a single-block schema when no `_L` suffix is
/// present (the `--net` linear-chain case).
fn build_blocks_schema(tasks: &schedule::TaskGraph) -> BlocksSchema {
    use std::collections::BTreeMap;
    // Map block-index → (first_task, last_task, count).
    let mut by_block: BTreeMap<u32, (usize, usize, usize)> = BTreeMap::new();
    let mut untagged = 0usize;
    for (tid, task) in tasks.tasks.iter().enumerate() {
        if let Some(idx_str) = task.op.rsplit("_L").next() {
            if idx_str != task.op {
                if let Ok(idx) = idx_str.parse::<u32>() {
                    let e = by_block.entry(idx).or_insert((tid, tid, 0));
                    e.0 = e.0.min(tid);
                    e.1 = e.1.max(tid);
                    e.2 += 1;
                    continue;
                }
            }
        }
        untagged += 1;
    }
    if by_block.is_empty() {
        // No `_L{i}` suffixes — treat the whole plan as one implicit block.
        return BlocksSchema {
            blocks: if tasks.tasks.is_empty() {
                vec![]
            } else {
                vec![BlockRange {
                    index: 0,
                    label: "implicit".into(),
                    first_task: 0,
                    last_task: tasks.tasks.len() - 1,
                    task_count: tasks.tasks.len(),
                }]
            },
            complete: false,
        };
    }
    let blocks = by_block
        .into_iter()
        .map(|(idx, (first, last, count))| BlockRange {
            index: idx,
            label: format!("layers.{idx}"),
            first_task: first,
            last_task: last,
            task_count: count,
        })
        .collect();
    BlocksSchema {
        blocks,
        complete: untagged == 0,
    }
}

/// Build the RequestIo schema for a bucket from the source network. For
/// `--net` sources the input `x` and the final op's output are the two
/// canonical fields (semantic tags `tokens` / `logits`). For HF-model
/// sources the schema is emitted with `complete: false` and a single
/// opaque field per graph input/output — full semantic tagging (image vs
/// text vs mask) awaits Graph.inputs/outputs plumbing.
fn build_request_io_schema(
    src: &Source,
    bucket: &ShapeBucket,
    plan: &LayerPlan,
) -> RequestIoSchema {
    const ELEM_BYTES_ACT: u32 = 2; // bf16 / fp16 activations
    const ELEM_BYTES_I32: u32 = 4; // i32 for indices / positions

    // Every bucket, both phases, has these two runtime-owned control
    // buffers. See the design notes.
    let rows = bucket.rows();
    let mut common = vec![
        RequestIoField {
            name: "position_ids".into(),
            direction: "input".into(),
            semantic: "position_ids".into(),
            shape: vec![rows],
            elem_bytes: ELEM_BYTES_I32,
        },
        RequestIoField {
            name: "past_len".into(),
            direction: "input".into(),
            semantic: "past_len".into(),
            shape: vec![bucket.batch],
            elem_bytes: ELEM_BYTES_I32,
        },
    ];

    match src {
        Source::Net(n) => {
            let hidden = n.hidden;
            let final_output = plan
                .ops
                .last()
                .map(|op| op.output.clone())
                .unwrap_or_default();
            let mut feat = hidden;
            for op in &n.ops {
                match op {
                    net::NetOp::Gemm { n: nn, .. } => feat = *nn,
                    net::NetOp::Norm { feat: f } => {
                        if let Some(f) = f {
                            feat = *f
                        }
                    }
                    _ => {}
                }
            }
            let mut fields = vec![RequestIoField {
                name: "x".into(),
                direction: "input".into(),
                semantic: "tokens".into(),
                shape: vec![rows, hidden],
                elem_bytes: ELEM_BYTES_ACT,
            }];
            fields.append(&mut common);
            fields.push(RequestIoField {
                name: final_output,
                direction: "output".into(),
                semantic: "logits".into(),
                shape: vec![rows, feat],
                elem_bytes: ELEM_BYTES_ACT,
            });
            RequestIoSchema {
                fields,
                complete: true,
            }
        }
        Source::Model(_) | Source::HfDir(_) => {
            // Plumbing gap: HF-model semantic tagging requires walking
            // Graph.inputs/outputs and mapping tensor names → semantic tags
            // (input_ids, attention_mask, pixel_values, logits). Not wired
            // yet — emit the common control buffers + `complete: false` so
            // the runtime falls back to address-map-only lookups for the
            // token/logit paths.
            RequestIoSchema {
                fields: common,
                complete: false,
            }
        }
    }
}

/// §P Append a host `SAMPLE` packet to a decode bucket's program.
///
/// Every terminal instruction (no successors — the logits/output stores)
/// increments a fresh counter the sample packet waits on, so sampling runs the
/// instant the whole compute (incl. logits) is done — unblocked from the tile
/// packets, on the host executor. `vocab` is the logit width; the runtime
/// resolves the logits/tokens buffers by RequestIo semantic, so the slots are
/// left as sentinels here.
fn inject_sample_packet(prog: &mut packet::Program, vocab: u32) {
    use packet::{Body, Counter, Inst, Opcode, ResourceKind, SLOT_NONE};
    let new_id = prog.counters.iter().map(|c| c.id + 1).max().unwrap_or(0);
    let mut threshold = 0u32;
    for inst in prog.insts.iter_mut() {
        if inst.succ.is_empty() && !matches!(inst.body, Body::Host) {
            inst.succ.push(new_id);
            threshold += 1;
        }
    }
    if threshold == 0 {
        return; // empty program — nothing to gate on
    }
    prog.counters.push(Counter {
        id: new_id,
        threshold,
        scope: 2, // CrossUnit: a host reader gates on a device-produced counter
        _pad: [0; 3],
    });
    prog.insts.push(Inst {
        resource: ResourceKind::Host,
        unit: 0,
        index: 0,
        body: Body::Token {
            in_slot: SLOT_NONE,
            out_slot: SLOT_NONE,
            kind: Opcode::TOKEN_SAMPLE_GREEDY,
            vocab,
            arg: 0,
        },
        wait: vec![new_id],
        succ: vec![],
    });
}

/// §P Prepend a host `TOKENIZE` packet at the graph head. Root compute
/// instructions (no waits, on an SM) gate on it, so the first matmul can't start
/// until the `tokens` buffer exists — without serializing independent weight
/// DMAs. Symmetric to [`inject_sample_packet`].
fn inject_tokenize_packet(prog: &mut packet::Program) {
    use packet::{Body, Counter, Inst, Opcode, ResourceKind, SLOT_NONE};
    let new_id = prog.counters.iter().map(|c| c.id + 1).max().unwrap_or(0);
    let mut consumers = 0u32;
    for inst in prog.insts.iter_mut() {
        if inst.wait.is_empty() && matches!(inst.resource, ResourceKind::Sm) {
            inst.wait.push(new_id);
            consumers += 1;
        }
    }
    if consumers == 0 {
        // Every head-of-graph SM inst already waits on something (typically
        // weight-DMA counters), so there is nothing to gate — the flag would
        // otherwise silently emit no TOKENIZE packet.
        warn!(
            "[emit-tokenize] warning: no wait-free SM instruction to gate; \
             TOKENIZE packet not emitted"
        );
        return;
    }
    prog.counters.push(Counter {
        id: new_id,
        threshold: 1,
        scope: 2,
        _pad: [0; 3],
    });
    prog.insts.insert(
        0,
        Inst {
            resource: ResourceKind::Host,
            unit: 0,
            index: 0,
            body: Body::Token {
                in_slot: SLOT_NONE,
                out_slot: SLOT_NONE,
                kind: Opcode::TOKEN_TOKENIZE,
                vocab: 0,
                arg: 0,
            },
            wait: vec![],
            succ: vec![new_id],
        },
    );
}

/// Logit width (vocab / final feature count) for a source, for the SAMPLE body.
/// `--net`: the running feature width after the last width-changing op. HF
/// HF models: the final lm_head width.
fn logits_width(src: &Source, plans: &[(ShapeBucket, LayerPlan)]) -> i64 {
    match src {
        Source::Net(n) => {
            let mut feat = n.hidden;
            for op in &n.ops {
                match op {
                    net::NetOp::Gemm { n: nn, .. } => feat = *nn,
                    net::NetOp::Norm { feat: Some(f) } => feat = *f,
                    _ => {}
                }
            }
            feat
        }
        // HfDir: the plan's last op is the lm_head GEMM, whose N is
        // vocab_size. A 0 here would emit a SAMPLE packet with vocab=0 —
        // an argmax over nothing.
        Source::Model(_) | Source::HfDir(_) => plans
            .first()
            .and_then(|(_, p)| p.ops.last())
            .and_then(|op| match &op.kind {
                rewrite::OpKind::Gemm(g) => Some(g.n),
                _ => None,
            })
            .unwrap_or(0),
    }
}

/// Emit each bucket's packet stream to a `.pkt` file and gather its stats.
fn emit_streams(
    compiled: &Compiled,
    soc: &Soc<'_>,
    _cfg: &Config,
    opts: &Options,
    src: &Source,
    plans: &[(ShapeBucket, LayerPlan)],
    fp8_scale_bytes: u64,
) -> Result<(Vec<BucketReport>, Vec<Footprint>, LeanStatus), PlowcError> {
    let hbm_capacity = soc.unit(0).cm.spec.mem.capacity.0;
    let mut out = Vec::with_capacity(compiled.streams.len());
    let mut footprints: Vec<Footprint> = Vec::with_capacity(compiled.streams.len());
    let kv_layout = compiled.kv;
    let mut lean = if opts.lean_verify {
        LeanStatus {
            verified: true,
            provenance: "certified: every bucket".into(),
        }
    } else {
        LeanStatus {
            verified: false,
            provenance: "not requested (Options::lean_verify = false)".into(),
        }
    };
    for bs in &compiled.streams {
        let phase = phase_name(bs.bucket.phase);
        let stem = format!("{phase}_b{}_s{}", bs.bucket.batch, bs.bucket.seq);
        debug!(
            stage = "emission", bucket = %stem, tasks = bs.sched.tasks.tasks.len(),
            "emitting bucket artifacts"
        );

        // Optional post-schedule passes, applied in order:
        //  1. §8.1 counter elimination — drop counters covered by resource-order
        //  2. §8.2 scope narrowing   — downgrade IntraGpu → IntraSm when the
        //     actual placement is same-SM
        // Both preserve the DAG happens-before semantics (Plow.Protocol);
        // scope only affects the runtime memory barrier, not ordering.
        let mut current: std::borrow::Cow<'_, schedule::Schedule> =
            std::borrow::Cow::Borrowed(&bs.sched.schedule);
        if opts.counter_elim {
            let (reduced, rep) = schedule::counter_elim::eliminate_redundant_counters(&current);
            info!(
                "[counter-elim] {stem}: {} → {} counters ({:.1}% dropped)",
                rep.before,
                rep.kept,
                rep.savings_pct(),
            );
            current = std::borrow::Cow::Owned(reduced);
        }
        if opts.scope_narrow {
            let (narrowed, rep) = schedule::scope_narrow::narrow_scopes(&current);
            info!(
                "[scope-narrow] {stem}: {} IntraGpu counters, {} narrowed to IntraSm ({:.1}%)",
                rep.before_intra_gpu,
                rep.narrowed.len(),
                rep.narrowed_pct(),
            );
            current = std::borrow::Cow::Owned(narrowed);
        }
        if opts.prefetch {
            let old_makespan = current.makespan;
            let (hoisted_sched, rep) =
                schedule::prefetch::hoist_prefetches(&bs.sched.tasks, &current);
            let new_makespan = hoisted_sched.makespan;
            info!(
                "[prefetch] {stem}: {} DMA-in hoisted (avg advance {:.1} slots), \
                 makespan {} → {}",
                rep.hoisted.len(),
                rep.avg_slot_advance(),
                old_makespan,
                new_makespan,
            );
            current = std::borrow::Cow::Owned(hoisted_sched);
        }
        // §8.5 Phase 2 (promotion + reschedule) runs before emit_streams —
        // by this point `bs.sched` already reflects any SRAM promotions.
        let effective_sched = current;

        // HBM bandwidth audit: the list scheduler reserves HBM only for separate
        // DMA tasks, so transfer bytes folded onto compute tasks (dma-fold /
        // collapsed-mode) escape its capacity check. Surface an oversubscribed
        // schedule here rather than emit it silently. Diagnostic only — makespan
        // is unchanged (a bandwidth-accurate makespan needs the bandwidth-aware
        // simulator).
        let hbm_audit =
            schedule::hbm_bandwidth_audit(&bs.sched.tasks, &effective_sched, &bs.sched.machine);
        for (unit, peak, cap) in hbm_audit.oversubscribed() {
            warn!(
                "[hbm-audit] {stem}: unit {unit} peak HBM demand {peak:.2} B/cycle \
                 exceeds capacity {cap:.2} B/cycle (folded compute-task loads \
                 included); makespan may be understated",
            );
        }

        // Surface the Lean oracle's decisions (under --lean-oracle). Previously
        // the whole OracleReport was computed and then dropped; at minimum the
        // lower-bound gap, recovered bubble cycles, and per-unit prefetch-depth
        // recommendations should be visible.
        if let Some(rep) = &bs.sched.oracle_report {
            info!(
                "[oracle] {stem}: makespan {} vs lower bound {} ({:.1}% gap, binding {:?}{}); \
                 bubble-fill recovered {} cycles across {} streams; prefetch depth/unit {:?}",
                rep.achieved_makespan,
                rep.lower_bound.bound,
                rep.optimality_gap * 100.0,
                rep.lower_bound.binding,
                if rep.any_certified {
                    ", lean-certified"
                } else {
                    ""
                },
                rep.bubble_fill.cycles_recovered,
                rep.bubble_fill.streams_modified,
                rep.prefetch_depths
                    .iter()
                    .map(|p| p.depth)
                    .collect::<Vec<_>>(),
            );
        }

        if opts.emit_trace {
            let trace_json = schedule::trace::to_chrome_json(
                &bs.sched.tasks,
                &effective_sched,
                &bs.sched.machine,
            );
            let trace_file = format!("{stem}.trace.json");
            std::fs::write(opts.out.join(&trace_file), trace_json)?;
        }

        let mut prog = emit_program(&bs.graph, &bs.cons, &bs.sched.tasks, &effective_sched);
        // §P Emit a host SAMPLE packet at the decode tail, gated on the output
        // stage — compiler-flag opt-in, decode-phase only.
        if opts.emit_sample && bs.bucket.phase == Phase::Decode {
            inject_sample_packet(&mut prog, logits_width(src, plans) as u32);
        }
        // §P Tokenize is a head-of-graph host op — prefill ingests text/ids.
        if opts.emit_tokenize && bs.bucket.phase == Phase::Prefill {
            inject_tokenize_packet(&mut prog);
        }
        let bytes = prog.to_bytes();
        let packet_file = format!("{stem}.pkt");
        std::fs::write(opts.out.join(&packet_file), &bytes)?;
        trace!(stage = "emission", bucket = %stem, file = %packet_file, bytes = bytes.len(), "packet stream written");

        // Address map: where every buffer lives in the bucket's HBM arena. The
        // task-set map is only consumed by the Lean verifier below (§6.2.2).
        let (mut amap, mut task_sets) = schedule::memory::plan_from_schedule_with_task_sets(
            &bs.sched.tasks,
            &effective_sched,
            &bs.cons,
        );
        // Inject a Growable KV cache entry so the runtime has an explicit
        // (offset, reserved) region to grow into. The compiler picks an
        // initial block count sized to cover this bucket's prefill; the
        // runtime allocates further blocks past `offset + reserved` as
        // sequences extend.
        let kv_paging = inject_kv_growable_entry(
            &mut amap,
            kv_layout,
            &bs.bucket,
            &opts.kv,
            &bs.sched.tasks,
            &bs.cons,
        );
        // Attribute flash tasks to the injected KV entries so the Lean
        // checkpoints (D/F) verify them against real writer/reader sets —
        // a Growable entry with empty sets passes reclamation vacuously.
        // Layer ids mirror `inject_kv_growable_entry` and preserve sparse
        // hybrid-model ids (Qwen full attention at layers 3, 7, ...).
        if kv_paging.is_some() {
            for (op, layer_idx) in flash_ops(&bs.sched.tasks, &bs.cons) {
                let ids: Vec<schedule::TaskId> = bs
                    .sched
                    .tasks
                    .tasks
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.op == op)
                    .map(|(tid, _)| tid)
                    .collect();
                // Flash both appends to and reads the layer's KV region.
                task_sets.insert(format!("kv_cache_L{layer_idx}"), (ids.clone(), ids));
            }
        }
        // Collect weight name → (N, K) shape for every GEMM in this bucket.
        // Runtime pairs this with `weight_tiling` in `weights.json` to
        // arrange safetensor bytes into the tiled layout. See
        // the design notes.
        let mut weight_shapes: std::collections::HashMap<String, (i64, i64, nn_graph::DType)> =
            std::collections::HashMap::new();
        for (_, desc) in bs.cons.op_io.iter() {
            if let rewrite::OpKind::Gemm(g) = &desc.kind {
                // Convention: `inputs[0]` is the activation, `inputs[1]` the weight.
                if let Some(weight_name) = desc.inputs.get(1) {
                    weight_shapes.insert(weight_name.clone(), (g.n, g.k, desc.weight_dtype));
                }
            }
        }
        // OOM guard: refuse to emit a bucket whose HBM footprint (activations
        // + KV growth + weights) exceeds the target GPU's capacity. `amap`
        // segments track activations + KV; weights live in a distinct region
        // the runtime loads from safetensors, so add their bytes here.
        let total_weight_bytes: u64 = weight_shapes
            .values()
            .map(|(n, k, dtype)| {
                let elements = ((*n).max(0) as u64).saturating_mul((*k).max(0) as u64);
                dtype.tile_bytes(elements)
            })
            .sum::<u64>()
            .saturating_add(fp8_scale_bytes);
        // Under tensor parallelism (the only multi-GPU strategy) GEMM weights
        // are sharded along N across units, so each device holds ~1/Nth of the
        // total — charging the full model to every segment would over-count
        // ~num_gpus× and reject buckets that fit.
        let n_devices = amap.segments.len().max(1) as u64;
        let per_device_weight_bytes = total_weight_bytes.div_ceil(n_devices);
        for seg in &amap.segments {
            let needed = seg.size.saturating_add(per_device_weight_bytes);
            if needed > hbm_capacity {
                return Err(PlowcError::HbmOverflow {
                    bucket: stem.clone(),
                    device: seg.device,
                    needed,
                    capacity: hbm_capacity,
                    overshoot: needed - hbm_capacity,
                });
            }
        }

        let mut mem = build_memory_report(&amap, kv_paging.clone(), &weight_shapes);

        // RequestIo sidecar — describes what each per-request buffer is and
        // how the runtime should marshal it. Built before the map is written so
        // its per-field `direction` can narrow the map entries' kind/access.
        // Reuse the plan prebuilt in `compile()` — rebuilding here would re-run
        // model resolution (a network fetch for `Source::Model`) once per
        // bucket, and swallowing its errors would silently emit an empty schema.
        let plan = plans
            .iter()
            .find(|(bb, _)| *bb == bs.bucket)
            .map(|(_, p)| p)
            .expect("every compiled bucket has a prebuilt plan");
        let mut request_io = build_request_io_schema(src, &bs.bucket, plan);
        // §P The SAMPLE packet writes a per-request token-id buffer; advertise it
        // so the runtime marshals it (semantic `tokens`, one id per row, i32).
        if opts.emit_sample && bs.bucket.phase == Phase::Decode {
            request_io.fields.push(RequestIoField {
                name: "sampled_tokens".into(),
                direction: "output".into(),
                semantic: "tokens".into(),
                shape: vec![bs.bucket.rows()],
                elem_bytes: 4,
            });
        }
        refine_request_io_kinds(&mut mem, &request_io);

        let memory_file = format!("{stem}.map.json");
        std::fs::write(
            opts.out.join(&memory_file),
            serde_json::to_string_pretty(&mem)?,
        )?;

        // Decode KV sidecar — only emitted for `phase == Decode` buckets.
        // Runtime patches `seq_kv` + KV base per step. See
        // the design notes.
        if let Some(dk) =
            build_decode_kv_schema(&bs.bucket, &bs.sched.tasks, &bs.cons, kv_paging.as_ref())
        {
            let dk_file = format!("{stem}.decode_kv.json");
            std::fs::write(opts.out.join(&dk_file), serde_json::to_string_pretty(&dk)?)?;
        }

        // Blocks sidecar — per-transformer-block task ranges. Pipeline-
        // parallel runtimes shard whole blocks across stages.
        let blocks = build_blocks_schema(&bs.sched.tasks);
        let blocks_file = format!("{stem}.blocks.json");
        std::fs::write(
            opts.out.join(&blocks_file),
            serde_json::to_string_pretty(&blocks)?,
        )?;

        // Experts sidecar — MoE layers for expert-parallel dispatch.
        // Phase 1 detection only; see the design notes.
        let experts = build_experts_schema(&bs.sched.tasks, src);
        let experts_file = format!("{stem}.experts.json");
        std::fs::write(
            opts.out.join(&experts_file),
            serde_json::to_string_pretty(&experts)?,
        )?;

        // Emit the RequestIo sidecar (built above, before the map write).
        let request_io_file = format!("{stem}.request_io.json");
        std::fs::write(
            opts.out.join(&request_io_file),
            serde_json::to_string_pretty(&request_io)?,
        )?;

        // Optional per-bucket verification via the Lean CLI. The whole
        // marshaling + subprocess lifecycle stays here in Rust so the boundary
        // is one function; Lean receives a JSON payload and returns a
        // certificate. See the design notes.
        if opts.lean_verify {
            let certified = run_lean_verify(
                &stem,
                &bs.sched.tasks,
                &effective_sched,
                &amap,
                &task_sets,
                &bs.graph,
                &bs.cons,
                &bs.sched,
                opts,
                &bytes,
            )?;
            // ONE bucket left unverified makes the WHOLE bundle unverified. The
            // report must not be able to say "verified" about a set of packets
            // where any member was skipped.
            if !certified {
                lean.verified = false;
                lean.provenance = format!(
                    "requested, but skipped from bucket `{stem}` on: no runnable plow_verify"
                );
            }
        }

        // Simulate the *effective* schedule (post counter-elim / scope-narrow /
        // prefetch) so the reported makespan matches the emitted .pkt, not the
        // pre-pass schedule.
        let sim = schedule::simulate(&bs.sched.tasks, &effective_sched);

        // Per-bucket memory-footprint breakdown. Classes are already
        // computed for OOM + address-map emission — just re-bucket them.
        let mut kv_cache_bytes = 0u64;
        let mut scratch_bytes = 0u64;
        let mut request_io_bytes = 0u64;
        let mut static_bytes = 0u64;
        let mut persistent_bytes = 0u64;
        for e in &amap.entries {
            match e.class {
                BufClass::Growable => kv_cache_bytes += e.reserved,
                BufClass::Scratch => scratch_bytes += e.reserved,
                BufClass::RequestIo => request_io_bytes += e.reserved,
                BufClass::Static => static_bytes += e.reserved,
                BufClass::Persistent => persistent_bytes += e.reserved,
            }
        }
        footprints.push(Footprint {
            phase: phase.to_string(),
            batch: bs.bucket.batch,
            seq: bs.bucket.seq,
            weight_bytes: total_weight_bytes,
            kv_cache_bytes,
            scratch_bytes,
            request_io_bytes,
            static_bytes,
            persistent_bytes,
            packet_bytes: bytes.len() as u64,
            arena_bytes: amap.arena_bytes,
            total_hbm_bytes: amap.arena_bytes + total_weight_bytes,
        });

        out.push(BucketReport {
            phase: phase.to_string(),
            batch: bs.bucket.batch,
            seq: bs.bucket.seq,
            packet_file,
            packet_bytes: bytes.len(),
            instructions: prog.insts.len(),
            tile_nodes: bs.graph.nodes.len(),
            tasks: bs.sched.tasks.tasks.len(),
            makespan: sim.makespan,
            ideal_makespan: sim.ideal_makespan,
            arena_bytes: amap.arena_bytes,
            memory_file,
        });
        debug!(
            stage = "emission", bucket = %stem, packet_bytes = bytes.len(),
            arena_bytes = amap.arena_bytes, "bucket artifacts emitted"
        );
    }

    // Sweep-across-buckets footprint sidecars. Handy for capacity planning
    // over a ladder of (batch, seq, phase) inputs — one row per bucket.
    if !footprints.is_empty() {
        std::fs::write(
            opts.out.join("footprint.json"),
            serde_json::to_string_pretty(&footprints)?,
        )?;
        std::fs::write(opts.out.join("footprint.csv"), footprint_csv(&footprints))?;
    }

    Ok((out, footprints, lean))
}

fn footprint_csv(rows: &[Footprint]) -> String {
    let mut s = String::new();
    s.push_str(
        "phase,batch,seq,weight_bytes,kv_cache_bytes,scratch_bytes,request_io_bytes,\
         static_bytes,persistent_bytes,packet_bytes,arena_bytes,total_hbm_bytes\n",
    );
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.phase,
            r.batch,
            r.seq,
            r.weight_bytes,
            r.kv_cache_bytes,
            r.scratch_bytes,
            r.request_io_bytes,
            r.static_bytes,
            r.persistent_bytes,
            r.packet_bytes,
            r.arena_bytes,
            r.total_hbm_bytes,
        ));
    }
    s
}

/// Build the SoC for the requested GPU count and parallel strategy.
/// Hand one bucket over to the Lean verifier: build the request, spawn
/// `plow_verify` for each of the six checkpoints, log per-checkpoint outcomes,
/// and turn any rejection into a compile error.
///
/// Orchestration stays in Rust — every Lean call is a pure decidable check
/// backed by a proven universal lemma (see the design notes). Logs go to
/// stderr so they don't perturb `weights.json`
/// on stdout. Compiled only with `--features lean-verify`.
///
/// Order: A (rewrite rules), B (tile partition), D (schedule + reclamation),
/// E (wire round-trip), F (allocation safety). Any single rejection
/// short-circuits, with a distinct log line per stage.
///
/// **Checkpoint C is intentionally *not* wired into the per-bucket path.**
/// The Rust filter in [`schedule::sram_fit::analyze_temporal_fit`] applies
/// the same two conditions the Lean `checkSramFit` re-checks
/// (`producer_release ≤ consumer_acquire` ∧ `max(producer_pages,
/// consumer_pages) ≤ budget`). Once `Plow.Sram.occupancy_le_of_temporal_fit`
/// establishes those two conditions suffice, the Rust filter is
/// known-correct by construction — running the Lean check per bucket adds
/// no information, only IPC latency. `check_sram_fit` remains available as
/// an opt-in dispatcher (used by `crates/plowc/tests/lean_verify_sram_fit.rs`)
/// for anyone who wants to spot-check the two-condition equivalence.
#[cfg(feature = "lean-verify")]
#[allow(clippy::too_many_arguments)]
fn run_lean_verify(
    bucket: &str,
    tasks: &schedule::TaskGraph,
    sched: &schedule::Schedule,
    amap: &schedule::AddressMap,
    task_sets: &schedule::memory::TensorTaskSets,
    graph: &rewrite::TileGraph,
    cons: &rewrite::ConstraintSet,
    _scheduled: &schedule::Scheduled,
    _opts: &Options,
    _packet_bytes: &[u8],
) -> Result<bool, PlowcError> {
    // `false` from any checkpoint means "no runnable verifier", and it ends the
    // attempt for this bucket: the remaining four would each pay another failed
    // spawn and log another identical warning. `?` still propagates a REJECTION.
    Ok(verify_A(bucket)?
        && verify_B(bucket, graph, cons, sched)?
        && verify_D(bucket, tasks, sched, amap, task_sets)?
        && verify_E(bucket)?
        && verify_F(bucket, tasks, sched, amap, task_sets)?)
}

#[cfg(feature = "lean-verify")]
fn dispatch_cert(
    bucket: &str,
    checkpoint: &'static str,
    cert: Result<lean_verify::Certificate, lean_verify::VerifyError>,
    started: std::time::Instant,
) -> Result<bool, PlowcError> {
    let elapsed_ms = started.elapsed().as_millis();
    let cert = match cert {
        Ok(c) => c,
        // DEGRADE ON AN UNUSABLE VERIFIER — this is the safety mechanism, and it
        // has to live here because this is the first code that has actually
        // tried to run the thing. A caller-side "does the binary exist?" probe
        // cannot cover a binary that exists and does not run (wrong arch, no
        // +x, or lean-plow's /nix/store ELF interpreter outside `nix develop`,
        // which exits 127). Without this, such a binary turns EVERY compile on
        // the packet path into a hard failure of a gate the user never asked
        // for — the gates are on by default now.
        //
        // `Ok(false)` — skipped — is deliberately distinct from `Ok(true)`, and
        // the caller propagates that into the report. A skipped gate that reads
        // like a passed one is this codebase's signature bug.
        Err(source) if source.is_binary_unusable() => {
            tracing::warn!(
                "[lean-verify:{checkpoint}] {bucket}: SKIPPED, no runnable plow_verify: {source}"
            );
            return Ok(false);
        }
        Err(source) => {
            tracing::error!("[lean-verify:{checkpoint}] {bucket}: spawn/marshal failure: {source}");
            return Err(PlowcError::LeanVerifySpawn {
                bucket: bucket.to_string(),
                detail: source.to_string(),
            });
        }
    };
    if cert.ok {
        let notes = cert.notes.as_deref().unwrap_or("(no notes)");
        debug!("[lean-verify:{checkpoint}] {bucket}: accepted in {elapsed_ms}ms — {notes}");
        Ok(true)
    } else {
        let reason = cert
            .reason
            .clone()
            .unwrap_or_else(|| "verifier returned ok=false with no reason".into());
        tracing::error!(
            "[lean-verify:{checkpoint}] {bucket}: REJECTED in {elapsed_ms}ms — {reason}"
        );
        Err(PlowcError::LeanVerify {
            bucket: bucket.to_string(),
            reason,
        })
    }
}

/// Checkpoint A — rewrite rule soundness. Parses the `; rule: <name>`
/// annotations out of the exact `rules.egg` source the engine runs
/// (`rewrite::rules_source()`) and submits that live catalog; the verifier
/// checks every entry against `Plow.Rewrite.soundRules`. A rewrite without an
/// annotation is a hard [`PlowcError::RuleCatalog`] error, and an annotated
/// rule whose name has no Lean `rule_*` theorem fails checkpoint A — so a
/// rule added to `rules.egg` alone cannot slip through. (Rule *bodies* are
/// not structurally checked; editing a rule's RHS under an existing proven
/// name is outside this checkpoint's scope.)
#[cfg(feature = "lean-verify")]
#[allow(non_snake_case)]
fn verify_A(bucket: &str) -> Result<bool, PlowcError> {
    use lean_verify::checkpoints::rewrite::{check_rewrite_rules, RewriteRulesRequest};
    let req = RewriteRulesRequest {
        rules: parse_rule_catalog(rewrite::rules_source())?,
    };
    let started = std::time::Instant::now();
    dispatch_cert(bucket, "A", check_rewrite_rules(&req), started)
}

/// Extract the `; rule: <name>` annotation preceding every `(rewrite ...)`
/// in an egglog source, in order. Errors when a rewrite has no annotation or
/// an annotation dangles without a rewrite, so the catalog submitted to
/// checkpoint A is exactly the set of rules the engine may fire.
#[cfg(feature = "lean-verify")]
fn parse_rule_catalog(src: &str) -> Result<Vec<String>, PlowcError> {
    let mut names: Vec<String> = Vec::new();
    let mut pending: Option<String> = None;
    for (i, line) in src.lines().enumerate() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("; rule:") {
            let name = name.trim();
            if name.is_empty() {
                return Err(PlowcError::RuleCatalog(format!(
                    "rules.egg:{}: empty `; rule:` annotation",
                    i + 1
                )));
            }
            if let Some(prev) = pending.replace(name.to_string()) {
                return Err(PlowcError::RuleCatalog(format!(
                    "rules.egg:{}: annotation `{prev}` not followed by a (rewrite ...)",
                    i + 1
                )));
            }
        } else if t.starts_with("(rewrite") {
            match pending.take() {
                Some(name) => names.push(name),
                None => {
                    return Err(PlowcError::RuleCatalog(format!(
                        "rules.egg:{}: (rewrite ...) without a `; rule: <name>` annotation",
                        i + 1
                    )))
                }
            }
        }
    }
    if let Some(prev) = pending {
        return Err(PlowcError::RuleCatalog(format!(
            "rules.egg: trailing annotation `{prev}` with no rule"
        )));
    }
    Ok(names)
}

/// Checkpoint B — tile partition + cost bounds. For every GEMM node in the
/// tile graph, extract its (m, n, k) shape and the (bm, bn, bk) tile the
/// scheduler picked, and submit them with a generous cost bound. The verifier
/// re-checks partition validity + tile-work ≤ cost_bound.
#[cfg(feature = "lean-verify")]
#[allow(non_snake_case)]
fn verify_B(
    bucket: &str,
    graph: &rewrite::TileGraph,
    cons: &rewrite::ConstraintSet,
    sched: &schedule::Schedule,
) -> Result<bool, PlowcError> {
    use lean_verify::checkpoints::tile_partition::{
        check_tile_partition, GemmShapeJ, TileCandidate, TilePartitionRequest, TileShapeJ,
    };
    let mut candidates: Vec<TileCandidate> = Vec::new();
    for (nid, desc) in cons.op_io.iter() {
        let rewrite::OpKind::Gemm(g) = &desc.kind else {
            continue;
        };
        // Recover the chosen tile shape from the scheduled compute node.
        let (bm, bn, bk) = match graph.nodes.get(*nid) {
            Some(rewrite::GraphNode::Compute {
                kind: rewrite::Compute::Gemm(t),
                ..
            }) => (t.bm, t.bn, t.bk),
            _ => continue,
        };
        // Cost bound independent of the tile-work formula the Lean side
        // re-derives (`tileCount · bm · bn · bk`): per dimension,
        // `ceil(d/b)·b ≤ d + b − 1`, so the padded volume
        // `(m+bm−1)(n+bn−1)(k+bk−1)` bounds the tile-work sum without ever
        // computing a tile count. The previous bound was the identical
        // tileCount expression ×2, so the check could never fail.
        let m = g.m as u64;
        let n = g.n as u64;
        let k = g.k as u64;
        let bmu = bm as u64;
        let bnu = bn as u64;
        let bku = bk as u64;
        let cost_bound = m
            .saturating_add(bmu.saturating_sub(1))
            .saturating_mul(n.saturating_add(bnu.saturating_sub(1)))
            .saturating_mul(k.saturating_add(bku.saturating_sub(1)))
            .max(1);
        candidates.push(TileCandidate {
            gemm: GemmShapeJ { m, n, k },
            tile: TileShapeJ {
                bm: bmu,
                bn: bnu,
                bk: bku,
            },
            cost_bound,
        });
    }
    let _ = sched; // Silence unused warning: bm comes from graph, not sched.
    let started = std::time::Instant::now();
    dispatch_cert(
        bucket,
        "B",
        check_tile_partition(&TilePartitionRequest { candidates }),
        started,
    )
}

/// Checkpoint D — schedule + reclamation. Backed by
/// `Plow.Verify.verifyAddressMap` + `Plow.Protocol.protocol_covers_deps`.
#[cfg(feature = "lean-verify")]
#[allow(non_snake_case)]
fn verify_D(
    bucket: &str,
    tasks: &schedule::TaskGraph,
    sched: &schedule::Schedule,
    amap: &schedule::AddressMap,
    task_sets: &schedule::memory::TensorTaskSets,
) -> Result<bool, PlowcError> {
    let request = schedule::lean_verify::build_schedule_request(tasks, sched, amap, task_sets);
    debug!(
        "[lean-verify:D] {bucket}: submitting — {n_tasks} tasks, {n_counters} counters, {n_entries} entries",
        n_tasks = request.task_graph.n,
        n_counters = request.protocol.threshold.len(),
        n_entries = request.address_map.len(),
    );
    let started = std::time::Instant::now();
    dispatch_cert(
        bucket,
        "D",
        lean_verify::checkpoints::check_schedule(&request),
        started,
    )
}

/// Checkpoint E — wire round-trip. The Lean `Plow.Wire` module abstracts
/// packet framing to a `List Frame` with u16 opcode + u16 payload_len. Rather
/// than reproduce the full `packet::Program::to_bytes` binary layout here
/// (see plan doc §5.10-E for the abstraction gap), we submit a canonical
/// 2-frame sample and confirm it round-trips — a smoke test that the Lean
/// verifier reaches the endpoint. Full parity between the abstract framing
/// and the concrete packet crate is a separate exercise.
#[cfg(feature = "lean-verify")]
#[allow(non_snake_case)]
fn verify_E(bucket: &str) -> Result<bool, PlowcError> {
    use lean_verify::checkpoints::wire::{
        check_wire_roundtrip, encode_program, WireFrame, WireRequest,
    };
    // Two synthetic frames spanning the byte range [0, 255] to exercise
    // the JSON u8-serialization boundaries.
    let frames = vec![
        WireFrame {
            opcode: 0x1234,
            payload: vec![0, 1, 2, 253, 254, 255],
        },
        WireFrame {
            opcode: 0xABCD,
            payload: (0..16).collect(),
        },
    ];
    let raw = encode_program(&frames);
    let started = std::time::Instant::now();
    dispatch_cert(
        bucket,
        "E",
        check_wire_roundtrip(&WireRequest { raw, frames }),
        started,
    )
}

/// Checkpoint F — allocation safety. Shares the D payload but hits the /F/
/// endpoint (same verifier, different semantic framing: pre-emit vs post-emit).
#[cfg(feature = "lean-verify")]
#[allow(non_snake_case)]
fn verify_F(
    bucket: &str,
    tasks: &schedule::TaskGraph,
    sched: &schedule::Schedule,
    amap: &schedule::AddressMap,
    task_sets: &schedule::memory::TensorTaskSets,
) -> Result<bool, PlowcError> {
    let request = schedule::lean_verify::build_schedule_request(tasks, sched, amap, task_sets);
    let started = std::time::Instant::now();
    dispatch_cert(
        bucket,
        "F",
        lean_verify::checkpoints::check_address_map(&request),
        started,
    )
}

/// Two-pass §8.5 driver: analyze the pessimistic schedule for temporal-fit
/// candidates, greedily promote a subset that improves makespan, reschedule.
///
/// # Phase 3 algorithm (2026-07-03)
///
/// Phase 2 showed "promote everything at once" regresses makespan 100% of the
/// time on plowc examples — forcing full colocation trades away too much
/// compute parallelism. Phase 3 is **greedy per-candidate**:
///
/// 1. Sort candidates by `cycles_saved` descending (biggest wins first).
/// 2. Start from the accepted-promotions set = ∅.
/// 3. For each candidate, tentatively add it, reschedule, compare makespan.
///    Keep it iff the makespan improves; drop otherwise.
/// 4. Apply the final accepted set to the bucket.
///
/// Cost: up to O(candidates) reschedules per bucket. In practice most
/// candidates get rejected fast on the second reschedule, so it's typically
/// a small constant. Buckets with zero candidates skip the loop entirely.
fn apply_sram_fit_phase2(
    compiled: &mut schedule::Compiled,
    soc: &costmodel::Soc<'_>,
    cfg: &schedule::Config,
) {
    for bs in compiled.streams.iter_mut() {
        let phase = phase_name(bs.bucket.phase);
        let stem = format!("{phase}_b{}_s{}", bs.bucket.batch, bs.bucket.seq);

        let rep = schedule::sram_fit::analyze_temporal_fit(&bs.sched, &bs.cons, &bs.sched.machine);
        if rep.candidates.is_empty() {
            debug!(
                "[sram-fit] {stem}: 0 promotable of {} demoted — no reschedule",
                rep.demoted
            );
            continue;
        }

        // Sort by biggest savings first (candidates own their cycles_saved).
        let mut candidates = rep.candidates.clone();
        candidates.sort_by(|a, b| b.cycles_saved.cmp(&a.cycles_saved));

        let baseline_makespan = bs.sched.schedule.makespan;
        // Accumulator state: (g, cons, sched, makespan) that reflects the
        // currently-accepted subset of promotions.
        let mut cur_g = bs.graph.clone();
        let mut cur_cons = bs.cons.clone();
        let mut cur_makespan = baseline_makespan;
        let mut accepted: Vec<schedule::sram_fit::PromotionCandidate> = Vec::new();
        let mut rejected = 0usize;
        let mut reschedules = 0usize;

        for cand in candidates {
            // Try promoting {accepted ∪ {cand}} in isolation from cur state
            // (need to redo from scratch each time — promotion is not
            // incremental over its own output).
            let mut trial = accepted.clone();
            trial.push(cand.clone());
            let (trial_g, trial_cons) =
                schedule::sram_fit::promote_temporal_fits(&bs.graph, &bs.cons, &trial);
            reschedules += 1;
            let trial_sched = schedule::schedule(soc, &trial_g, &trial_cons, cfg);
            let trial_makespan = trial_sched.schedule.makespan;

            if trial_makespan < cur_makespan {
                // Improvement — accept.
                accepted = trial;
                cur_g = trial_g;
                cur_cons = trial_cons;
                cur_makespan = trial_makespan;
            } else {
                rejected += 1;
            }
        }

        if accepted.is_empty() {
            debug!(
                "[sram-fit] {stem}: 0/{} handoffs accepted after {} reschedule attempts \
                 (baseline makespan {} kept; all trials regressed)",
                rep.candidates.len(),
                reschedules,
                baseline_makespan,
            );
            continue;
        }

        // Final reschedule to install the accepted set. `cur_g` / `cur_cons`
        // and `cur_makespan` already reflect it; one last `schedule::schedule`
        // gives us a fresh Scheduled to install.
        let final_sched = schedule::schedule(soc, &cur_g, &cur_cons, cfg);
        info!(
            "[sram-fit] {stem}: {}/{} handoffs accepted ({} rejected, {} reschedules), \
             makespan {} → {} ({}% better)",
            accepted.len(),
            rep.candidates.len(),
            rejected,
            reschedules,
            baseline_makespan,
            final_sched.schedule.makespan,
            (100 * baseline_makespan.saturating_sub(final_sched.schedule.makespan)
                / baseline_makespan.max(1)),
        );
        bs.graph = cur_g;
        bs.cons = cur_cons;
        bs.sched = final_sched;
    }
}

/// Append one Growable KV cache entry per attention layer to the bucket's
/// `AddressMap` and return the paging report. Returns `None` when the
/// bucket has no attention op — nothing to page. See `KvPagingReport` for
/// the runtime contract and the design notes for the design.
///
/// **Runtime seam.** The per-layer `AddrEntry.offset` this function writes
/// is the compiler's declaration of where layer `i`'s KV bytes live in the
/// arena. At runtime the mux builds an `AddressSpace` from the same map
/// and calls `AddressSpace::kv_layer_bases(paging)`, which walks each
/// `KvLayerPaging::buffer_name`, matches it to the `MemEntry` written here,
/// and resolves `phys_addr`. Those physical bases feed `KvArena::new`, so
/// the indirection table the mux writes per tick carries **real** device
/// addresses. Renaming or reordering `kv_cache_L{i}` here without updating
/// `KvLayerPaging.buffer_name` will silently produce zero bases (the
/// runtime warns on `AddressSpace::allocate` failure and falls back to
/// zero, but a wrong-name mapping stays undetected until KV writes land).
fn inject_kv_growable_entry(
    amap: &mut schedule::AddressMap,
    kv_layout: Option<schedule::KvLayout>,
    bucket: &ShapeBucket,
    kv_cfg: &KvConfig,
    tasks: &schedule::TaskGraph,
    cons: &rewrite::ConstraintSet,
) -> Option<KvPagingReport> {
    use schedule::{memory::MEM_ALIGN, AddrEntry, BufClass};
    let layout = kv_layout?;
    // Effective block size — config override, or layout default.
    let block_tokens = if kv_cfg.block_tokens > 0 {
        kv_cfg.block_tokens
    } else {
        layout.block_seq
    };
    let block_bytes: u64 = (block_tokens.max(0) as u64)
        .saturating_mul(layout.kv_heads.max(0) as u64)
        .saturating_mul(layout.head_dim.max(0) as u64)
        .saturating_mul(4); // 2 (K+V) × 2 bytes (bf16/fp16)
    if block_bytes == 0 {
        return None;
    }
    let layers = flash_ops(tasks, cons);
    // Per-layer initial block count: caller override, else enough to cover
    // this bucket's prefill seq. Both prefill and decode use the same reserve.
    let per_layer_blocks = if kv_cfg.initial_blocks > 0 {
        kv_cfg.initial_blocks
    } else {
        (bucket.seq.max(0) + block_tokens.max(1) - 1) / block_tokens.max(1)
    };
    // Per-head pool geometry (mirrors `GrowablePool`/`KvPool.lean`). Each layer's
    // buffer is a pool of `kv_factor × kv_heads × max_seqs` contiguous head-slots
    // of `head_slot_bytes` each, addressed by `headSlotOffset`. `elem = 2` (bf16)
    // matches the `× 4` (2·K,V × 2·bf16) that formed `block_bytes` above.
    //
    // `max_seqs = bucket.batch`: the pool reserves one head-slot grid per
    // concurrent sequence the bucket serves (the mux hands each live request a
    // stable seq-slot). `head_slot_bytes` covers a full sequence's positions
    // (`block_tokens × per_layer_blocks`). This makes the per-layer reserve
    // `batch ×` the single-sequence size — the deliberate memory cost of the
    // contiguous per-head layout (vs. the old shared paged pool).
    const KV_ELEM_BYTES: u64 = 2; // bf16
    let kv_factor: i64 = 2; // separate K and V
    let max_seqs: i64 = bucket.batch.max(1);
    let max_seq_positions = block_tokens.max(0) as u64 * per_layer_blocks.max(0) as u64;
    let head_slot_bytes = max_seq_positions * layout.head_dim.max(0) as u64 * KV_ELEM_BYTES;
    let per_layer_reserved = (kv_factor.max(0) as u64)
        .saturating_mul(layout.kv_heads.max(0) as u64)
        .saturating_mul(max_seqs.max(0) as u64)
        .saturating_mul(head_slot_bytes);

    let mut per_layer = Vec::with_capacity(layers.len());
    let mut slot = amap.entries.iter().map(|e| e.slot).max().unwrap_or(0);
    let mut cursor = amap.arena_bytes;
    for (_, layer_idx) in layers {
        cursor = ((cursor + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
        slot += 1;
        let name = format!("kv_cache_L{layer_idx}");
        amap.entries.push(AddrEntry {
            slot,
            name: name.clone(),
            class: BufClass::Growable,
            offset: cursor,
            reserved: per_layer_reserved,
            growable: true,
            device: 0,
        });
        per_layer.push(KvLayerPaging {
            layer_idx,
            buffer_name: name,
            initial_blocks: per_layer_blocks,
        });
        cursor += per_layer_reserved;
    }
    amap.arena_bytes = cursor;
    if let Some(seg0) = amap.segments.iter_mut().find(|s| s.device == 0) {
        seg0.size = seg0.size.max(cursor - seg0.global_base);
    }
    Some(KvPagingReport {
        block_tokens,
        block_bytes,
        kv_heads: layout.kv_heads,
        head_dim: layout.head_dim,
        kv_factor,
        max_seqs,
        head_slot_bytes,
        per_layer,
    })
}

/// Stub used when plowc is built without `--features lean-verify`. Setting
/// `Options.lean_verify = true` in this build hits this and fails the compile
/// with `PlowcError::LeanVerifyDisabled`, so the caller sees a clear signal
/// rather than a silently-skipped verification.
#[cfg(not(feature = "lean-verify"))]
#[allow(clippy::too_many_arguments)]
fn run_lean_verify(
    _bucket: &str,
    _tasks: &schedule::TaskGraph,
    _sched: &schedule::Schedule,
    _amap: &schedule::AddressMap,
    _task_sets: &schedule::memory::TensorTaskSets,
    _graph: &rewrite::TileGraph,
    _cons: &rewrite::ConstraintSet,
    _scheduled: &schedule::Scheduled,
    _opts: &Options,
    _packet_bytes: &[u8],
) -> Result<bool, PlowcError> {
    Err(PlowcError::LeanVerifyDisabled)
}

fn build_soc(spec: &'static hwspec::GpuSpec, opts: &Options) -> Result<Soc<'static>, PlowcError> {
    let page = opts.page_kib * 1024;
    if opts.num_gpus <= 1 {
        return Ok(Soc::single(spec, page));
    }
    match opts.parallel {
        Parallel::Tp => Ok(Soc::homogeneous(spec, opts.num_gpus, page)),
        other => Err(PlowcError::Parallelism(format!(
            "{other:?} across {} GPUs is not yet implemented (only tensor-parallel)",
            opts.num_gpus
        ))),
    }
}

/// The bucket ladder: the cartesian product of phases × batches × seqs.
fn make_buckets(opts: &Options) -> Vec<ShapeBucket> {
    let mut out = Vec::new();
    for &phase in &opts.phases {
        for &batch in &opts.batches {
            for &seq in &opts.seqs {
                out.push(ShapeBucket { batch, seq, phase });
            }
        }
    }
    out
}

/// Lower one bucket to a plan; the model path also returns fusion stats.
///
/// Indexed HF directories and Hub models share the nn-graph frontend. An
/// unindexed `Source::HfDir` retains the legacy full-model synthesis path.
fn build_plan(
    src: &Source,
    b: &ShapeBucket,
    hf_synth: Option<&hf_config::HfSynthesis>,
    model_metadata: Option<&nn_graph::hub::ModelMetadata>,
    explore: bool,
) -> Result<(LayerPlan, Option<RewriteStats>), PlowcError> {
    match src {
        Source::Net(n) => Ok((n.build_plan(b), None)),
        Source::HfDir(_) if model_metadata.is_none() => {
            // Option A: build the full N-layer plan with per-layer weight names.
            let synth = hf_synth.expect("HfDir source requires pre-synthesized HfSynthesis");
            let plan = hf_config::build_full_model_plan(b, synth);
            debug!(
                "[hf-dir] full-model plan: {} layers unrolled, {} ops total",
                synth.num_layers,
                plan.ops.len(),
            );
            Ok((plan, None))
        }
        Source::Model(_) | Source::HfDir(_) => {
            let model = src.name();
            info!(
                stage = "nn-graph", model = %model, batch = b.batch, seq = b.seq,
                "building nn_graph from pretrained model"
            );
            let metadata = model_metadata.expect("Model source requires resolved metadata");
            let mut g = nn_graph::hub::build_from_metadata(
                metadata,
                &nn_graph::models::ShapeBucket::default(),
            )?;
            g.bind(&nn_graph::Bindings::new().set("B", b.batch).set("S", b.seq));
            // The fused graph is not consumed (the plan below is built from the
            // source graph) and the caller keeps only the FIRST bucket's stats
            // for the fusion report — so saturating the full unrolled model
            // once per bucket was pure discarded work, and its e-graph is the
            // compile's peak-memory driver. Explore only when asked (the first
            // bucket); every later bucket skips straight to the plan.
            let stats = if explore {
                info!(
                    stage = "egg-exploration", model = %model, nodes = g.nodes.len(),
                    "running egg rewrite exploration on nn_graph"
                );
                let (_fused, stats) = rewrite::rewrite_graph(&g)?;
                Some(stats)
            } else {
                None
            };
            // Bake every block into one plan so tile-dep chaining spans block
            // boundaries — cross-block consumer tiles unblock per-tile instead
            // of waiting for the whole producer-block output tensor. Supports
            // heterogeneous block types (Gemma 4 mixed attention, DeepSeek
            // dense-then-MoE) transparently — each block's ops keep their
            // own `block` index in the source graph.
            let plan = plan_from_all_blocks(&g)?;
            Ok((plan, stats))
        }
    }
}

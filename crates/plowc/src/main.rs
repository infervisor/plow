//! `plowc` — compile a model or network into runtime packet streams for a GPU.
//!
//! The egglog rewriting stage does not lower packets. On the devblob path its
//! extracted graph now supplies a fail-closed semantic-coverage gate over the
//! finished GPU program: only rewrite/opcode mappings proved exact are counted,
//! and emission is rejected if devgen no longer carries their required ops.
//! `rewrite_applied` remains zero; existing packet fusions are hand-written.
//!
//! This is stated here, at the driver's front door, because the log line alone
//! ("egglog fusion analysis … fusions_found=662") was read as a compiler pass
//! reporting its work, and quoted as such.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
use devgen::emit_config::EmitConfig;
#[cfg(feature = "tuner")]
use plowc::tune::{self, TuneAction, TuneOptions};
use plowc::{compile, net::NetConfig, Options, Parallel, Report, Source};
use schedule::Phase;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

mod fusion_coverage;

#[derive(Parser, Debug)]
#[command(
    name = "plowc",
    about = "plow compiler: model/network → packet streams for a hardware spec"
)]
struct Cli {
    /// Optional subcommand. With none, `plowc` compiles, exactly as before —
    /// every existing invocation keeps working unchanged.
    #[command(subcommand)]
    cmd: Option<Cmd>,

    /// HuggingFace model id (resolved over the network), e.g. `Qwen/Qwen3-4B`.
    #[arg(long, conflicts_with_all = ["net", "hf_dir"])]
    model: Option<String>,

    /// Path to a plow-native network JSON (offline).
    #[arg(long, conflicts_with_all = ["model", "hf_dir"])]
    net: Option<PathBuf>,

    /// Path to a local HuggingFace model directory containing config.json and
    /// safetensors. Compiles the full N-layer model (Option A unroll) with
    /// checkpoint-matching weight names, validated against the safetensors.
    #[arg(long, conflicts_with_all = ["model", "net"])]
    hf_dir: Option<PathBuf>,

    /// Bucket preset: quick (2×2), default (3×3), serve (5×5 crossed), longctx (3×4).
    /// Overrides --batch and --seq when provided.
    #[arg(long, value_enum)]
    preset: Option<Preset>,

    /// GPU spec name or short alias (e.g. `rtx6000pro`, `h100`, `mi350`).
    /// Run with `--list-gpus` to see all recognized names.
    #[arg(long, default_value = "H100 SXM5")]
    gpu: String,

    /// Print the full list of recognized GPU names and aliases, then exit.
    #[arg(long, default_value_t = false)]
    list_gpus: bool,

    /// Number of GPUs.
    #[arg(long, default_value_t = 1)]
    num_gpus: usize,

    /// Parallel strategy across the GPUs.
    #[arg(long, value_enum, default_value_t = Parallel::Tp)]
    parallel: Parallel,

    /// Batch sizes to compile a bucket for (comma-separated).
    #[arg(long, value_delimiter = ',', default_value = "1,4,8")]
    batch: Vec<i64>,

    /// Sequence lengths to compile a bucket for (comma-separated).
    #[arg(long, value_delimiter = ',', default_value = "512,2048,8192")]
    seq: Vec<i64>,

    /// Which phase(s) to compile.
    #[arg(long, value_enum, default_value_t = PhaseArg::Both)]
    phase: PhaseArg,

    /// SRAM page size in KiB.
    #[arg(long, default_value_t = 16)]
    page_kib: u64,

    /// Output directory for the `.pkt` streams and `weights.json`.
    /// Defaults to `plow-out/<model-slug>/` when using --hf-dir.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Artifact form to emit:
    ///   * `devblob` (default with --hf-dir) — a single PLOWDEV `model.pkt`
    ///     the GPU runtime executes, plus a servable `weights.json`. Replaces
    ///     the deprecated `gemma4` binary; the `PLOW_*` emit knobs (FP8,
    ///     PLOW_BLOCK, PLOW_UNISEG, …) are honored exactly as before;
    ///   * `packets` (default with --net/--model) — scheduled `.pkt` bucket
    ///     streams + manifest, run on the CPU reference interpreter /
    ///     simulator;
    ///   * `devblob+cubin` — as `devblob`, then BUILD the interpreter object
    ///     from the manifest it just wrote. Opt-in, and only this form needs a
    ///     CUDA toolkit: `devblob` alone still requires none, so the shipped
    ///     binaries keep working against prebuilt assets.
    #[arg(long, value_enum)]
    emit: Option<EmitKind>,

    /// devblob only: target ISA recorded in `build.json` (`sm_120a`, `sm_90a`,
    /// `gfx950`, …), and the arch `--emit devblob+cubin` builds for.
    ///
    /// METADATA for the packet itself — it does not change a single emitted byte.
    /// The manifest names opcodes, shapes and rules; mapping those to a
    /// toolchain's flags is the backend's job (nvcc → .cubin, hipcc → .hsaco).
    #[arg(long, default_value = "sm_120a")]
    arch: String,

    /// devblob only: max context tokens the program is compiled for.
    #[arg(long, default_value_t = 131072)]
    max_ctx: u32,

    /// devblob only: target executor (SM/CU) count. 0 = use the `--gpu`
    /// spec's `sm_count`.
    #[arg(long, default_value_t = 0)]
    n_cu: u32,

    /// devblob only: `--block l` or `l..r` — emit a single block (env
    /// `PLOW_BLOCK` is the fallback, unchanged).
    #[arg(long)]
    block: Option<String>,

    /// devblob+cubin: also build segmented prefill cubins (_pfseg, _pfgemm).
    /// Implies NOT setting PLOW_UNISEG=1, so the emitted programs carry
    /// wave-class segments the SegPf runtime dispatches per-class.
    #[arg(long)]
    segmented: bool,

    /// devblob only: expand the RoPE tables into the blob's init section instead
    /// of carrying them as recipes the runtime materialises at load.
    ///
    /// The default (recipes, a v7 blob) is the difference between a ~430 MB and a
    /// ~25 MB Gemma-4 `model.pkt` at the default `--max-ctx`, and costs nothing at
    /// run time — `plowrt` regenerates the identical bytes host-side while the
    /// weights upload.
    ///
    /// Pass this for the C host harnesses under `runtime/tests/` (the gfx950 /
    /// sm_120 drivers): they read the init section directly and reject a v7 magic.
    #[arg(long)]
    no_rope_gen: bool,

    /// devblob only: embed an interpreter cubin as a blob section (the former
    /// `gemma4 --embed-cubin`). The runtime loads it from the blob itself.
    #[arg(long)]
    embed_cubin: Option<String>,

    /// devblob only: embed an interpreter hsaco as a blob section (the former
    /// `gemma4 --embed-hsaco`).
    #[arg(long)]
    embed_hsaco: Option<String>,

    /// Disable the Lean ORDERING CERTIFICATE (on by default). Affects only
    /// verification — `--lean-oracle` is a separate switch and keeps running.
    ///
    /// On, each bucket's `(schedule, address_map)` goes to the `plow_verify`
    /// CLI; a REJECTION fails the compile. No usable verifier on this machine
    /// is a warning and a skip, recorded in `build.json` as
    /// `lean.verified: false` with a reason. Binary located via
    /// `PLOW_VERIFY_BIN` or `lean-plow/.lake/build/bin/plow_verify`.
    ///
    /// ONE SWITCH PER SUBSYSTEM, deliberately: an earlier design had both
    /// `--lean-verify` and `--no-lean-verify` bound to two fields, which made
    /// the positive flag a no-op and let the negative one silently disable the
    /// oracle as well.
    /// DEFAULT-ON IS BACKED OUT PENDING AN OPEN REJECTION. Turning it on made
    /// `plowc` refuse to compile Gemma-4-12B at all:
    ///
    ///   gemma4-12b: compile failed: lean verifier rejected bucket `decode_b1_s1`:
    ///   reader/writer sets overlap — strict AddressMapSound not derivable
    ///
    /// That is the gate WORKING — a `plow_verify` binary is present on the
    /// gfx950 box, so the degrade path never ran and the verifier delivered a
    /// real verdict. But it is unresolved: either Gemma-4-12B's decode program
    /// genuinely aliases a reader and a writer (this codebase's signature defect
    /// class — 14 found, every one fluent-but-wrong rather than crashing), or
    /// `AddressMapSound` is too strict for a legitimate in-place op such as a
    /// residual accumulate or KV-ring reuse.
    ///
    /// Until that is settled the flag ships OFF, because the alternative —
    /// downgrading a genuine rejection to a warning — is exactly the
    /// silently-skipped-gate failure the manifest `lean` block was added to
    /// prevent. Opt in with `--lean-verify`; a rejection is still fatal there,
    /// and everything else from that work (the manifest record, the degrade on
    /// a missing binary, and the fix for the GLM/Kimi/DeepSeek early-returns
    /// that were dropping the hook entirely) is unconditional and stays on.
    ///
    /// SCOPE OF THE BACK-OUT NARROWED 2026-08-10: the paragraph above is about
    /// the SCHEDULE (per-bucket) path, which is where AddressMapSound runs and
    /// where Gemma-4-12B rejects. The DEVBLOB path has its own defaulted
    /// switch below (`--lean-verify-devblob`). This flag still force-enables
    /// both paths.
    #[arg(long = "lean-verify", action = clap::ArgAction::SetTrue)]
    lean_verify: bool,

    /// Lean ordering certificate on the DEVBLOB path — ON by default
    /// (`--lean-verify-devblob=false` to disable). Default-on is safe here and
    /// not on the schedule path: this path's checkpoint-D form (GQ order
    /// topological over counter edges) passes every GLM-5.2 program, a missing
    /// or unrunnable `plow_verify` degrades to a recorded skip, and a
    /// REJECTION aborts the emit — a bug caught, never downgraded.
    #[arg(long = "lean-verify-devblob", default_value_t = true,
          action = clap::ArgAction::Set, num_args = 0..=1,
          require_equals = true, default_missing_value = "true")]
    lean_verify_devblob: bool,

    /// Lean performance oracle on the DEVBLOB path — ON by default
    /// (`--lean-oracle-devblob=false` to disable). Read-only: a log line and a
    /// manifest record; the decode lower bound stays labeled certified=false
    /// (analytical, no proof object attached on this path).
    #[arg(long = "lean-oracle-devblob", default_value_t = true,
          action = clap::ArgAction::Set, num_args = 0..=1,
          require_equals = true, default_missing_value = "true")]
    lean_oracle_devblob: bool,

    /// Drop counters already covered by resource-order (§8.1 counter
    /// elimination). Provably safe by the DAG-side theorem; combined with
    /// `--lean-verify`, the reduced schedule is cross-checked per bucket.
    #[arg(long, default_value_t = false)]
    counter_elim: bool,

    /// Narrow `IntraGpu` counter scopes to `IntraSm` when the actual runtime
    /// placement puts every producer and consumer on the same SM (§8.2).
    /// Safe by construction: `Scope` is a runtime memory-visibility flag,
    /// not part of the ordering semantics.
    #[arg(long, default_value_t = false)]
    scope_narrow: bool,

    /// Hoist DMA-in tasks past unrelated compute in each resource stream
    /// (§8.3). Improves compute/DMA overlap on memory-bound workloads.
    #[arg(long, default_value_t = false)]
    prefetch: bool,

    /// Run the §8.5 SRAM temporal-fit pass: promotes temporally-disjoint
    /// handoffs back to same-SM SRAM and reschedules (changes the emitted
    /// schedule; logs accepted/rejected candidates).
    #[arg(long, default_value_t = false)]
    sram_fit: bool,

    /// Disable the Lean PERFORMANCE ORACLE (on by default). Affects only the
    /// oracle — the ordering certificate keeps running unless you also pass
    /// `--no-lean-verify`.
    ///
    /// On, `plow_verify` is queried for provably-optimal counter granularity,
    /// prefetch depth, and lower bounds; Rust heuristics take over when the
    /// binary is unavailable. On the devblob path the decode lower bound comes
    /// back WITHOUT a certificate (the log line reports `certified=false`) —
    /// treat it as analytical.
    /// OFF by default for the same reason as `--lean-verify` above: both share
    /// the `plow_verify` binary, and shipping the oracle on while verification
    /// is backed out would give a compile that consults the verifier but
    /// ignores its verdict. Opt in with `--lean-oracle`.
    #[arg(long = "lean-oracle", action = clap::ArgAction::SetTrue)]
    lean_oracle: bool,

    /// §P Emit a host-executor SAMPLE packet at the tail of every decode bucket
    /// (logits → token id, gated on the output-stage counter). Decode-only.
    #[arg(long, default_value_t = false)]
    emit_sample: bool,

    /// §P Emit a host TOKENIZE packet at the graph head (text/ids → tokens).
    #[arg(long, default_value_t = false)]
    emit_tokenize: bool,

    /// Experiment: fuse structurally matched same-input linear pairs into one
    /// multi-output GPU packet. Full-graph analysis must establish the match.
    #[arg(long, default_value_t = false)]
    experiment_parallel_linear2: bool,

    /// Emit a Chrome Trace Event Format JSON per bucket
    /// (`{stem}.trace.json`) showing every scheduled task as a duration
    /// event on its resource lane (SM / DMA / DPU / Host). Load in
    /// `chrome://tracing` or `ui.perfetto.dev`.
    #[arg(long, default_value_t = false)]
    emit_trace: bool,

    /// Tokens per KV cache block (paging size, à la vLLM). The compiler
    /// reports this in `weights.json` + reserves an initial block count in
    /// the address map; the runtime grows the KV region by allocating
    /// further blocks past the reserved range as sequences extend.
    #[arg(long, default_value_t = 256)]
    kv_block_tokens: i64,

    /// Initial number of KV blocks reserved in the compiled address map. 0
    /// means "auto-size from the largest bucket's prefill length".
    #[arg(long, default_value_t = 0)]
    kv_initial_blocks: i64,

    /// Rank tiles with the analytical cost model alone, ignoring the probed
    /// kernel inventory and any measurements.
    ///
    /// The escape hatch: use it when a probe is wrong, the vendor toolchain is
    /// missing, or a build must be reproduced exactly as it was before the
    /// tuner existed. The compiled output is identical to the pre-tuner
    /// compiler.
    #[arg(long, default_value_t = false)]
    no_tuning: bool,

    /// Tuning database root to read qualified measurements from. Omit to use
    /// the capability filter without measurements.
    #[arg(long)]
    tuning_db: Option<PathBuf>,

    /// Override weight dtype for GEMM projections. Accepts "bf16", "fp8", or
    /// "auto" (infer from config.json's torch_dtype / quantization_config).
    /// Default: "auto". Norms/embed/activations always remain BF16.
    #[arg(long, default_value = "auto")]
    weight_dtype: String,

    /// Emit-time configuration knobs (precision, fusion, placement, etc.).
    /// Each field also reads its `PLOW_*` env var as a fallback.
    #[command(flatten)]
    emit_cfg: EmitConfig,
}

impl Cli {
    /// `--emit` with its context-dependent default: `devblob` when compiling a
    /// checkpoint (`--hf-dir`), `packets` for the `--net`/`--model` simulator
    /// flows (devblob requires a checkpoint).
    fn emit(&self) -> EmitKind {
        self.emit.unwrap_or(if self.hf_dir.is_some() {
            EmitKind::Devblob
        } else {
            EmitKind::Packets
        })
    }
}

/// Bucket presets control the batch×seq grid.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Preset {
    /// Minimal: 2×2×2 = 8 buckets (fast iteration).
    Quick,
    /// Conservative: 3×3×2 = 18 buckets.
    Default,
    /// Full crossed ladder: 5×5×2 = 50 buckets (production serving).
    Serve,
    /// Long-context focus: 3×4×2 = 24 buckets (short+long seq, crossed batch).
    Longctx,
}

impl Preset {
    fn batches(self) -> Vec<i64> {
        match self {
            Preset::Quick => vec![1, 8],
            Preset::Default => vec![1, 4, 8],
            Preset::Serve => vec![1, 2, 4, 8, 16],
            Preset::Longctx => vec![1, 4, 8],
        }
    }
    fn seqs(self) -> Vec<i64> {
        match self {
            Preset::Quick => vec![512, 2048],
            Preset::Default => vec![512, 2048, 8192],
            Preset::Serve => vec![128, 512, 2048, 8192, 32768],
            Preset::Longctx => vec![512, 2048, 8192, 32768],
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PhaseArg {
    Prefill,
    Decode,
    Both,
}

/// What kind of artifact `plowc` writes. See `Cli::emit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EmitKind {
    /// Scheduled `.pkt` bucket streams + manifest (the `--net`/`--model`
    /// default).
    Packets,
    /// A single PLOWDEV device blob the GPU runtime executes.
    Devblob,
    /// [`EmitKind::Devblob`] plus the interpreter object built from the
    /// manifest's own `requires` + `recommends`.
    ///
    /// Deliberately opt-in. The README advertises that "no CUDA toolkit is
    /// required at compile time … prebuilt binaries ship with assets", and that
    /// stays true: `devblob` behaves exactly as before and never looks for a
    /// toolchain. Only this variant does, and it says so clearly when one is
    /// absent rather than failing somewhere inside cmake.
    #[value(name = "devblob+cubin")]
    DevblobCubin,
}

impl PhaseArg {
    fn phases(self) -> Vec<Phase> {
        match self {
            PhaseArg::Prefill => vec![Phase::Prefill],
            PhaseArg::Decode => vec![Phase::Decode],
            PhaseArg::Both => vec![Phase::Prefill, Phase::Decode],
        }
    }
}

/// Subcommands. `compile` stays the default with no subcommand at all, so this
/// is additive to the existing CLI rather than a break.
#[derive(Subcommand, Debug)]
enum Cmd {
    /// Inspect and calibrate kernel selection for a hardware target.
    ///
    /// Deliberately separate from compiling: `compile` may read qualified
    /// tuning records but must never write them, or a build could calibrate
    /// itself against its own output.
    Tune(TuneCli),

    /// Serve an interactive dataflow visualization of the nn-graph in the browser.
    ///
    /// Builds the symbolic operator graph from a HuggingFace `config.json`
    /// (via `--hf-dir`) or from a plow-native network JSON (via `--net`),
    /// then serves a self-contained HTML viewer on a local port. The viewer
    /// shows every tensor, shape, and op in a navigable DAG layout.
    ///
    /// ```text
    /// plowc --hf-dir /path/to/kimi-k3 viz --port 8384
    /// ```
    Viz(VizCli),
}

/// `plowc tune <action>`.
///
/// `--shape` / `--status` are the pre-subcommand spellings and still work, so nothing in flight
/// breaks. New work should use the action word.
///
/// The actions that DERIVE a shape list (`shapes`, `status` with coverage, `gemm --shapes auto`)
/// run a real emit, so they take the compile's own flags from the TOP-LEVEL command — before the
/// word `tune`:
///
/// ```text
/// plowc --hf-dir <ckpt> --max-ctx 4096 --n-cu 256 --num-gpus 4 tune gemm --obj <objdir>
/// ```
///
/// That is deliberate. A shape list is only correct for one configuration — the prefill bucket
/// ladder is part of the demand — so the configuration is stated in the same words the build
/// states it, instead of being re-declared here where it could drift.
#[derive(Args, Debug)]
struct TuneCli {
    /// inventory | select | status | shapes | gemm | ingest | best | regress
    #[arg(value_name = "ACTION")]
    action: Option<String>,

    /// GPU spec name or short alias (e.g. `rtx6000pro`, `h100`, `mi350`).
    #[arg(long, default_value = "H100 SXM5")]
    gpu: String,

    /// Interpreter profile to inspect.
    #[arg(long, default_value = "prefill_dense")]
    profile: String,

    /// Tuning database root.
    #[arg(long, default_value = "tuning")]
    db: PathBuf,

    /// Repository root, used to locate the interpreter sources to probe.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// Resolve one `M,N,K` shape and explain the choice.
    #[arg(long, value_name = "M,N,K")]
    shape: Option<String>,

    /// Deprecated spelling of `tune status`.
    #[arg(long)]
    status: bool,

    /// `gemm`: object directory holding a freshly built `test_kernels.elf`.
    #[arg(long, value_name = "DIR")]
    obj: Option<PathBuf>,

    /// `gemm`/`ingest`: the raw-sample JSONL the C harness writes.
    #[arg(long, value_name = "FILE")]
    samples: Option<PathBuf>,

    /// `gemm`: shape list. `auto` (the default) DERIVES it from the compiler's demand; a path
    /// replays an `M N K [label [quant]]` list — the form that can drift, kept for a
    /// byte-comparable A/B against the bash campaign, and for the models `auto` cannot yet
    /// reach (Kimi-K3 has no full-model emit, so its demand cannot be observed). `quant` is
    /// `None`|`W8A8`|`Mxfp4` and defaults to `None`.
    #[arg(long, default_value = "auto", value_name = "auto|FILE")]
    shapes: String,

    /// `gemm`: wrap every GPU invocation in `perf-data/tools/gpulease -n 1`. Off by default,
    /// matching the bash script, whose caller holds the lease.
    #[arg(long)]
    lease: bool,

    /// `gemm`: measure but do NOT publish. Prints the command that would publish, and says so
    /// loudly — an unpublished campaign changes nothing while looking like a success.
    #[arg(long)]
    no_ingest: bool,

    /// `gemm`: derive the shapes and stop. Measures nothing, touches no GPU.
    #[arg(long)]
    dry_run: bool,

    /// `gemm`/`ingest`: campaign label recorded on every published record.
    #[arg(long, default_value = "gemm-tile-inventory")]
    campaign: String,

    /// `gemm`/`ingest`: store as screening-only, NOT selectable.
    #[arg(long)]
    provisional: bool,

    /// `best`: which weight encoding to report.
    #[arg(long, default_value = "None", value_name = "None|W8A8|Mxfp4")]
    quant: String,

    /// `regress`: report op cases whose median moved by at least this fraction across build
    /// digests. `gemm_c4` regressed ~0.30 and nothing reported it.
    #[arg(long, default_value_t = 0.10, value_name = "FRACTION")]
    threshold: f64,
}

/// `plowc viz` — serve or dump the nn-graph visualization.
#[derive(Args, Debug)]
struct VizCli {
    /// Port to serve the viewer on (0 = write to file instead of serving).
    #[arg(long, default_value_t = 8384)]
    port: u16,

    /// Output file path. When `--port 0`, write the self-contained HTML here.
    #[arg(long, default_value = "graph.html")]
    out: PathBuf,

    /// Open the browser automatically after starting the server.
    #[arg(long, default_value_t = true)]
    open: bool,
}

fn main() -> ExitCode {
    init_logging();
    let mut cli = Cli::parse();

    // DEFAULT ON FOR sm_120. The persistent sm_120 interpreter runs every op in one cooperative
    // launch and implements the coarse single-segment path only, so a segmented blob is not
    // something that target can express. Without this, a plain `--hf-dir --arch sm_120a` compile
    // SUCCEEDS and the asset then fails at serve time, in a different binary, against
    // `plowrt`'s `check_coarse_single_segment` gate — with a message that never names the flag
    // that was missing. Defaulting it here moves the decision to the only place that knows the
    // arch. `deny_uniseg` still wins downstream for targets that must read `seg` (AMD).
    // Opt out with PLOW_UNISEG=0.
    if cli.arch.starts_with("sm_120") && std::env::var_os("PLOW_UNISEG").is_none() {
        // The real gate is `packet::devbuild::Builder`'s own `std::env::var("PLOW_UNISEG")` read,
        // not this struct field — set both so the emitted manifest and the diagnostic agree.
        std::env::set_var("PLOW_UNISEG", "1");
        cli.emit_cfg.uniseg = true;
    }
    let cli = cli;

    // Log the parsed CLI arguments so every invocation is self-describing in logs.
    let source_desc = if let Some(ref m) = cli.model {
        format!("--model {m}")
    } else if let Some(ref n) = cli.net {
        format!("--net {}", n.display())
    } else if let Some(ref d) = cli.hf_dir {
        format!("--hf-dir {}", d.display())
    } else {
        "(no source)".to_string()
    };
    info!(
        source = %source_desc,
        gpu = %cli.gpu,
        num_gpus = cli.num_gpus,
        batch = ?cli.batch,
        seq = ?cli.seq,
        phase = ?cli.phase,
        emit = ?cli.emit(),
        preset = ?cli.preset,
        parallel = ?cli.parallel,
        page_kib = cli.page_kib,
        weight_dtype = %cli.weight_dtype,
        "plowc invoked"
    );

    if cli.list_gpus {
        print_gpu_list();
        return ExitCode::SUCCESS;
    }

    if let Some(Cmd::Tune(t)) = &cli.cmd {
        info!(gpu = %t.gpu, profile = %t.profile, "tuning command started");
        return match run_tune(t, &cli) {
            Ok(()) => {
                info!("tuning command completed");
                ExitCode::SUCCESS
            }
            Err(e) => {
                error!(error = %e, "tuning command failed");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(Cmd::Viz(v)) = &cli.cmd {
        return match run_viz(v, &cli) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                error!(error = %e, "viz failed");
                ExitCode::FAILURE
            }
        };
    }

    if matches!(cli.emit(), EmitKind::Devblob | EmitKind::DevblobCubin) {
        info!(gpu = %cli.gpu, "devblob emit started");
        return match run_devblob(&cli) {
            Ok(out) => {
                info!(out = %out.display(), "devblob emit completed");
                ExitCode::SUCCESS
            }
            Err(e) => {
                error!(error = %e, "devblob emit failed");
                ExitCode::FAILURE
            }
        };
    }

    match run(cli) {
        Ok(report) => {
            print_report(&report);
            ExitCode::SUCCESS
        }
        Err(e) => {
            error!(error = %e, "compilation failed");
            ExitCode::FAILURE
        }
    }
}

fn fusion_coverage_hook(
    coverage: fusion_coverage::FusionCoverage,
    inner: devgen::VerifyHook,
) -> devgen::VerifyHook {
    Box::new(move |model| {
        let report = coverage.validate(model)?;
        let verified = inner(model)?;
        info!(
            graph_ops = coverage.graph_ops,
            extracted = coverage.extracted,
            rewrite_applied = 0,
            gpu_equivalent_covered = report.gpu_equivalent_covered,
            not_opcode_equivalent = report.not_opcode_equivalent,
            missing = 0,
            reaches_gpu_semantics = report.gpu_equivalent_covered > 0,
            same_input_narrow_pairs_missing_rule = coverage.same_input_narrow_pairs,
            parallel_linear2 = ?coverage.parallel_linear2(),
            by_op = ?coverage.by_op,
            "whole-graph fusion coverage: exact mappings gate the emitted decode program; \
             rewrite still lowers no packet. Same-input narrow linear pairs need a generic \
             multi-output rule; GemvQkv with Nv=0 already supplies the GPU opcode."
        );
        Ok(verified)
    })
}

fn whole_graph_fusion_decisions(
    coverage: Option<&fusion_coverage::FusionCoverage>,
    tp: u32,
    experiment_parallel_linear2: bool,
) -> devgen::WholeGraphFusionDecisions {
    coverage
        .map(|coverage| coverage.decisions(tp, experiment_parallel_linear2))
        .unwrap_or_default()
}

/// The devblob Lean gate: certify, per emitted program, that the
/// global-queue stream order is TOPOLOGICAL over the counter edges — the
/// invariant the persistent interpreter's deadlock-freedom argument rests
/// on (and that `plowrt` re-checks structurally at load time).
///
/// Encoding into `plow_verify` checkpoint D: one task per GQ stream entry,
/// waits/succs from the entry's counter gate lists, every task on ONE
/// resource with `stream_idx` = GQ position. The resource-order clause then
/// contributes the total issue order, so the ordering graph has a cycle iff
/// some counter edge points BACKWARD in GQ order — exactly non-topology.
/// The address map is empty: device-arena aliasing is NOT modeled here.
/// Why the Lean gates cannot even be ATTEMPTED here, or `None` if they can.
///
/// Purely an optimisation gate — see the contract on
/// [`lean_verify::binary_available`]. It exists so a default-on compile skips
/// the whole `ScheduleRequest` marshal when there is
/// obviously no verifier, NOT so anyone can assume a `None` here means the
/// verifier will work. The hook below still downgrades a spawn failure.
fn lean_unavailable_reason() -> Option<&'static str> {
    #[cfg(not(feature = "lean-verify"))]
    {
        return Some("plowc was built without the `lean-verify` cargo feature");
    }
    #[cfg(feature = "lean-verify")]
    {
        if lean_verify::binary_available() {
            None
        } else {
            Some("no `plow_verify` binary (set PLOW_VERIFY_BIN, or `lake build` in lean-plow/)")
        }
    }
}

#[cfg(feature = "lean-verify")]
fn devblob_verify_hook(
    do_verify: bool,
    do_oracle: bool,
    bw_bytes_per_cycle: u64,
    clock_hz: u64,
) -> Result<devgen::VerifyHook, Box<dyn std::error::Error>> {
    use lean_verify::checkpoints::schedule as lv;
    Ok(Box::new(move |m: &packet::devbuild::Model| {
        let mut rep = devgen::LeanReport::default();
        if do_oracle {
            match devblob_oracle(m, bw_bytes_per_cycle, clock_hz) {
                Ok(()) => rep.oracle = true,
                // THIS DOWNGRADE IS THE SAFETY MECHANISM, not the `binary_available`
                // probe above. The probe passes for a binary that is present but
                // unrunnable — wrong arch, no +x, or lean-plow's /nix/store ELF
                // interpreter outside `nix develop` (exit 127). Delete this and such
                // a binary turns every compile into a panic.
                Err(e) if e.is_binary_unusable() => {
                    warn!(error = %e, "lean oracle skipped: verifier not runnable");
                    rep.reason = Some(format!("oracle skipped: {e}"));
                }
                Err(e) => return Err(format!("lean oracle failed: {e}")),
            }
        }
        if !do_verify {
            rep.reason.get_or_insert_with(|| {
                "ordering certificate disabled on the command line (--no-lean-verify)".into()
            });
            return Ok(rep);
        }
        for (pi, p) in m.progs.iter().enumerate() {
            let n = p.gq_stream.len();
            let mut waits: Vec<Vec<u64>> = Vec::with_capacity(n);
            let mut succs: Vec<Vec<u64>> = Vec::with_capacity(n);
            let mut threshold = std::collections::BTreeMap::new();
            for e in &p.gq_stream {
                let (wo, wl) = (e.wait_ofs as usize, e.wait_len as usize);
                waits.push(
                    p.waits[wo..wo + wl]
                        .iter()
                        .map(|w| {
                            threshold.insert(w.id.to_string(), w.threshold as u64);
                            w.id as u64
                        })
                        .collect(),
                );
                let (so, sl) = (e.succ_ofs as usize, e.succ_len as usize);
                succs.push(p.succs[so..so + sl].iter().map(|&c| c as u64).collect());
            }
            // ONE CURSOR, OR ONE PER (ORDERED SEGMENT, DOMAIN) — the verifier has to be told
            // which. These are device-side queues; the host only launches the ordered segment.
            //
            // Unplaced, the GQ is a single global cursor, so entry `i`'s issue predecessor is
            // entry `i-1` and `resource = 0 / stream_idx = i` is exactly right.
            //
            // Under `PLOW_L2_PLACE` it is NOT. `Builder::finish` stable-sorts `gq_stream` by
            // `seg` into `l2_domains` windows, and the runtime gives each domain its OWN cursor
            // over its OWN window (see the design notes). Entry `i`'s issue
            // predecessor is then the previous entry OF ITS DOMAIN. Handing the placed stream to
            // the one-cursor model makes every counter edge that crosses from a later window
            // back to an earlier one look like a backward edge, and checkpoint D reports the
            // whole decode program cyclic — a FALSE deadlock. (Measured: Gemma-4-31B decode,
            // "ordering graph has a cycle (160499 nodes unsorted)", where placement OFF on the
            // identical program certifies clean.)
            //
            // With `resource = segment*domains+domain` the check gets sharper, not weaker: each
            // per-XCD queue is internally op-major, while independent XCDs make progress
            // concurrently and may have counter edges in either flattened-array direction.
            let placed = p.l2_domains > 0;
            let (resource, stream_idx): (Vec<u64>, Vec<u64>) = if placed {
                let mut res = Vec::with_capacity(n);
                let mut idx = Vec::with_capacity(n);
                let windows = p.gq_seg_ofs.len().saturating_sub(1).max(1);
                let mut next = vec![0u64; windows];
                for e in &p.gq_stream {
                    let d = ((e.flags & packet::dev::SE_DOMAIN_MASK)
                        >> packet::dev::SE_DOMAIN_SHIFT) as usize;
                    let window = e.seg as usize * p.l2_domains as usize + d;
                    if window >= windows {
                        return Err(format!(
                            "program {pi} (T={}): stream entry selects XCD queue {window} of {windows}",
                            m.prog_t.get(pi).copied().unwrap_or(0)
                        ));
                    }
                    res.push(window as u64);
                    idx.push(next[window]);
                    next[window] += 1;
                }
                (res, idx)
            } else {
                (vec![0; n], (0..n as u64).collect())
            };
            let req = lv::ScheduleRequest {
                task_graph: lv::TaskGraphView {
                    n,
                    edges: Vec::new(),
                },
                protocol: lv::ProtocolView {
                    waits,
                    succs,
                    threshold,
                    resource,
                    stream_idx,
                },
                schedule_order: (0..n as u64).collect(),
                address_map: Vec::new(),
            };
            let t = m.prog_t.get(pi).copied().unwrap_or(0);
            let cert = match lv::check_schedule(&req) {
                Ok(c) => c,
                // Same downgrade as the oracle, and the reason it has to be HERE
                // rather than at the probe: this is the first point that has
                // actually tried to run the thing. Bail out of the whole loop —
                // the remaining programs would each pay another failed spawn and
                // print another identical warning.
                Err(e) if e.is_binary_unusable() => {
                    warn!(error = %e, "lean ordering certificate skipped: verifier not runnable");
                    rep.verified = false;
                    rep.reason = Some(format!("verifier not runnable: {e}"));
                    return Ok(rep);
                }
                Err(e) => return Err(format!("program {pi} (T={t}): lean verifier failed: {e}")),
            };
            // A REJECTION IS A BUG CAUGHT. Never downgraded: `Err` from this hook
            // aborts emission before any bytes are written.
            if !cert.ok {
                return Err(format!(
                    "program {pi} (T={t}): GQ order not topological over counter edges: {}",
                    cert.reason.unwrap_or_default()
                ));
            }
            info!(
                program = pi,
                t,
                entries = n,
                "lean ordering certificate: GQ order topological over counter edges"
            );
            // LdsFitSound (checkpoint G): every always-staged GEMV instance in this
            // program must fit the decode-object LDS arena — the task-9 bug class,
            // rejected at emit. The staged set mirrors op_gemm.h's "x is ALWAYS
            // staged in LDS here" family; demand is the kernel's rows*K (+ the
            // q-norm fold scratch when t[7] rides GemvQkv). Plain Gemv is excluded
            // by design: its body carries a per-op fit fallback.
            {
                use packet::dev::{DevOp, TENSOR_NONE};
                const GV_NORM_SCRATCH: u64 = 16;
                let staged: Vec<serde_json::Value> = p
                    .insts
                    .iter()
                    .enumerate()
                    .filter_map(|(ix, inst)| {
                        let (name, scratch) = if inst.op == DevOp::GemvQkv as u16 {
                            (
                                "GemvQkv",
                                if inst.t[7] != TENSOR_NONE {
                                    GV_NORM_SCRATCH
                                } else {
                                    0
                                },
                            )
                        } else if inst.op == DevOp::GemvQkvg as u16 {
                            ("GemvQkvg", 0)
                        } else if inst.op == DevOp::GemvQkvMxfp4 as u16 {
                            ("GemvQkvMxfp4", 0)
                        } else if inst.op == DevOp::GemvGlu as u16 {
                            ("GemvGlu", 0)
                        } else if inst.op == DevOp::GemvGluSz as u16 {
                            ("GemvGluSz", 0)
                        } else if inst.op == DevOp::GemvGluMxfp4 as u16 {
                            ("GemvGluMxfp4", 0)
                        } else {
                            return None;
                        };
                        Some(serde_json::json!({
                            "op": name,
                            "idx": ix,
                            "rows": inst.i[0],
                            "k": inst.i[2],
                            "scratch": scratch,
                        }))
                    })
                    .collect();
                let n_staged = staged.len();
                let arena = devgen::decode_arena_halves();
                let cert = match lean_verify::call(
                    "G",
                    serde_json::json!({ "arena": arena, "ops": staged }),
                ) {
                    Ok(c) => c,
                    Err(e) if e.is_binary_unusable() => {
                        warn!(error = %e, "LdsFitSound skipped: verifier not runnable");
                        rep.verified = false;
                        rep.reason = Some(format!("verifier not runnable: {e}"));
                        return Ok(rep);
                    }
                    Err(e) => {
                        return Err(format!(
                            "program {pi} (T={t}): LdsFitSound call failed: {e}"
                        ))
                    }
                };
                if !cert.ok {
                    return Err(format!(
                        "program {pi} (T={t}): LdsFitSound REJECTED: {}",
                        cert.reason.unwrap_or_default()
                    ));
                }
                info!(
                    program = pi,
                    t,
                    staged = n_staged,
                    arena,
                    "lean LdsFitSound: staged-LDS demand within arena"
                );
            }
        }
        // Every program certified. `verified` claims exactly that and nothing more.
        rep.verified = true;
        Ok(rep)
    }))
}

/// `--lean-oracle` on the devblob path: a lower bound for one DECODE step of
/// the emitted program, computed by `plow_verify`'s `lower_bound` query.
///
/// NOT CERTIFIED on this path, and the log line says so. `LowerBoundResult`
/// carries `certificate: Option<String>` with `#[serde(default)]`, and the
/// Lean answer for this query returns no `certificate` field — so
/// `certified = false` on every run today. The arithmetic is the oracle's; the
/// proof object is not attached. Read the number as an analytical bound, and do
/// not describe `--lean-oracle` output as "proven" until this reads true.
///
/// The oracle's `lower_bound` query gets
/// the decode program's inst-level counter-edge graph (unit durations →
/// critical path = program depth) and the bytes the decode program actually
/// touches (every tensor referenced by a decode inst, `kv.*` excluded — the
/// KV stream scales with context, the weight stream is the fixed floor).
/// The binding constraint at batch 1 is HBM bandwidth: cycles ≥ bytes / bw.
#[cfg(feature = "lean-verify")]
fn devblob_oracle(
    m: &packet::devbuild::Model,
    bw_bytes_per_cycle: u64,
    clock_hz: u64,
) -> Result<(), lean_verify::VerifyError> {
    use lean_verify::queries::lower_bound as lb;
    let Some(p) = m.progs.last() else {
        return Ok(());
    };
    let pi = m.progs.len() - 1;
    // Inst-level counter edges: producer succ counter ∈ consumer wait list.
    let mut producers: std::collections::HashMap<u32, Vec<usize>> = Default::default();
    for (i, inst) in p.insts.iter().enumerate() {
        let (so, sl) = (inst.succ_ofs as usize, inst.succ_len as usize);
        for &c in &p.succs[so..so + sl] {
            producers.entry(c).or_default().push(i);
        }
    }
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut touched: std::collections::BTreeSet<u32> = Default::default();
    for (i, inst) in p.insts.iter().enumerate() {
        let (wo, wl) = (inst.wait_ofs as usize, inst.wait_len as usize);
        for w in &p.waits[wo..wo + wl] {
            if let Some(ps) = producers.get(&w.id) {
                for &a in ps {
                    edges.push((a, i));
                }
            }
        }
        for &t in &inst.t {
            if t != packet::dev::TENSOR_NONE {
                touched.insert(t);
            }
        }
    }
    let bytes: u64 = touched
        .iter()
        .filter_map(|&t| m.tensors.get(t as usize))
        .filter(|td| !td.name.starts_with("kv."))
        .map(|td| td.bytes)
        .sum();
    let req = lb::LowerBoundRequest {
        edges,
        durations: vec![1; p.insts.len()],
        total_hbm_bytes: bytes,
        peak_bw_bytes_per_cycle: bw_bytes_per_cycle.max(1),
        total_flops: 0,
        peak_flops_per_cycle: 1,
    };
    // Error passed through UNWRAPPED so the caller can tell "no runnable
    // verifier" (downgrade to a warning) from "the query itself failed"
    // (a hard error). Stringifying here would erase that distinction.
    let res = lb::query_lower_bound(&req)?;
    let us = res.lower_bound as f64 / (clock_hz as f64 / 1e6);
    info!(
        program = pi,
        insts = p.insts.len(),
        touched_bytes = bytes,
        critical_path_depth = res.critical_path,
        bw_bound_cycles = res.bw_bound,
        lower_bound_cycles = res.lower_bound,
        lower_bound_us = format!("{us:.1}"),
        binding = ?res.binding_constraint,
        certified = res.certificate.is_some(),
        "[oracle] devblob decode-step lower bound (lean)"
    );
    Ok(())
}

/// Unreachable in practice — `lean_unavailable_reason()` short-circuits to the
/// skip hook in this build — but it must compile, and it must NOT be an error.
/// The gates are on by default now: a plowc built without the feature has to
/// emit the same blob it always did and SAY that nothing was checked, not
/// refuse to compile the model.
#[cfg(not(feature = "lean-verify"))]
fn devblob_verify_hook(
    _do_verify: bool,
    _do_oracle: bool,
    _bw: u64,
    _clock: u64,
) -> Result<devgen::VerifyHook, Box<dyn std::error::Error>> {
    Ok(devgen::skip_hook(
        "plowc was built without the `lean-verify` cargo feature",
    ))
}

/// `--emit devblob`: compile the checkpoint into a single PLOWDEV `model.pkt`
/// via the `devgen` emitter and write a servable `weights.json` next to it, so
/// the output directory is a complete `plowrt serve --assets <dir>` bundle.
/// This is the plowc-native replacement for the standalone `gemma4` binary.
fn run_devblob(cli: &Cli) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir = cli
        .hf_dir
        .clone()
        .ok_or("--emit devblob requires --hf-dir <checkpoint>")?;

    let config_json = std::fs::read_to_string(dir.join("config.json"))?;
    ensure_devblob_arch_supported(&config_json)?;

    let slug = plowc::hf_config::dir_slug(&dir);

    // Two output shapes:
    //   * `--out foo.pkt` → write just the device blob to that exact path (the
    //     `gemma4` binary's behaviour, for the block/trace/C-harness scripts);
    //   * `--out <dir>` (or default) → a full servable bundle: `<dir>/model.pkt`
    //     plus `weights.json`.
    let out_is_pkt = cli
        .out
        .as_ref()
        .map(|p| p.extension().is_some_and(|e| e == "pkt"))
        .unwrap_or(false);
    let (out_dir, pkt): (PathBuf, PathBuf) = if out_is_pkt {
        let pkt = cli.out.clone().unwrap();
        let parent = pkt.parent().map(PathBuf::from).unwrap_or_default();
        (parent, pkt)
    } else {
        let d = cli
            .out
            .clone()
            .unwrap_or_else(|| PathBuf::from("plow-out").join(&slug));
        let pkt = d.join("model.pkt");
        (d, pkt)
    };
    if !out_dir.as_os_str().is_empty() {
        std::fs::create_dir_all(&out_dir)?;
    }

    // Executor count: explicit --n-cu, else the target GPU spec's sm_count.
    let n_cu = if cli.n_cu > 0 {
        cli.n_cu
    } else {
        hwspec::registry::lookup(&cli.gpu)
            .map(|s| s.sm_count)
            .ok_or_else(|| format!("unknown GPU {:?}; pass --n-cu explicitly", cli.gpu))?
    };

    // Only tensor-parallel is wired, exactly as in the packet path
    // (`Soc::homogeneous` there). Reject DP/PP/EP rather than silently emitting
    // a single-GPU program that the manifest would mislabel as multi-GPU.
    if cli.parallel != Parallel::Tp {
        return Err(format!(
            "{:?} across {} GPUs is not yet implemented (only tensor-parallel)",
            cli.parallel, cli.num_gpus
        )
        .into());
    }
    // Tensor-parallel degree = --num-gpus.
    let tp = cli.num_gpus.max(1) as u32;

    // `--block` with the same `PLOW_BLOCK` env fallback the legacy CLI used.
    let block_spec = cli
        .block
        .clone()
        .or_else(|| std::env::var("PLOW_BLOCK").ok().filter(|s| !s.is_empty()));

    // Full-model extraction is now a semantic coverage obligation, not a
    // fusion counter. Partial `--block` emits intentionally contain fewer
    // instances and remain a debugging path, so only a full model is gated.
    let fusion_coverage = if block_spec.is_none() {
        match fusion_coverage::FusionCoverage::analyze(&dir) {
            fusion_coverage::Analysis::Covered(coverage) => Some(coverage),
            fusion_coverage::Analysis::Ineligible => None,
            fusion_coverage::Analysis::AdvisoryFailure(e) => {
                warn!(error = %e, "whole-graph fusion coverage unavailable; no structurally eligible KDA graph was established");
                None
            }
            fusion_coverage::Analysis::RequiredFailure(e) => {
                return Err(format!("whole-graph fusion coverage failed closed: {e}").into())
            }
        }
    } else {
        None
    };
    // Candidate eligibility comes only from full-graph analysis. Qualification
    // remains an explicit experiment until the network gate promotes it.
    let whole_graph_fusions = whole_graph_fusion_decisions(
        fusion_coverage.as_ref(),
        tp,
        cli.experiment_parallel_linear2,
    );

    // The Lean gates on the devblob path. BOTH ARE ON BY DEFAULT (disable with
    // `--no-lean-verify` / `--no-lean-oracle`, one switch each, no coupling).
    //
    // Still strictly additive to the OUTPUT: the hook takes `&Model`, so
    // whatever happens here the emitted blob is byte-identical. What it can do
    // is refuse — a rejection aborts before a single byte is written.
    //
    // Three outcomes, and `build.json` distinguishes all three:
    //   * verified — a certificate for every program;
    //   * skipped — no runnable `plow_verify`; warn, emit, record the reason;
    //   * rejected — panic, nothing written.
    //  * egglog fusion analysis of the same checkpoint's graph, reported;
    //  * `--lean-verify`: a Lean ordering certificate for the EMITTED
    //    programs — every program's global-queue order is checked
    //    topological over its counter edges (plow_verify checkpoint D);
    //  * `--lean-oracle`: a Lean-certified decode lower bound — critical
    //    path + the weight-streaming bandwidth floor of the decode program.
    // DEVBLOB PATH: certificate and oracle are DEFAULT-ON as of 2026-08-10 via
    // their own clap switches (`--lean-verify-devblob` / `--lean-oracle-devblob`,
    // both defaulted true). The Gemma-4-12B AddressMapSound back-out is a
    // SCHEDULE-path verdict and that path keeps its opt-in `--lean-verify`,
    // which still force-enables here too.
    let lean_verify_on = cli.lean_verify || cli.lean_verify_devblob;
    let lean_oracle_on = cli.lean_oracle || cli.lean_oracle_devblob;
    let verify = if !lean_verify_on && !lean_oracle_on {
        Some(devgen::skip_hook(
            "both gates disabled (--lean-verify-devblob=false --lean-oracle-devblob=false)",
        ))
    } else if let Some(why) = lean_unavailable_reason() {
        // DEGRADE, DO NOT FAIL. Absent a verifier the compile proceeds and the
        // blob is byte-identical — the hook is read-only, so it never had any
        // say in the bytes. The warning is for the person watching; the
        // `build.json` record below is for everyone who asks later.
        warn!(
            reason = why,
            "lean gates skipped — this blob is NOT verified (build.json: lean.verified=false)"
        );
        Some(devgen::skip_hook(why))
    } else {
        let spec = hwspec::registry::lookup(&cli.gpu)
            .ok_or_else(|| format!("unknown GPU {:?}", cli.gpu))?;
        // MEASURED bandwidth, not the datasheet peak — `bandwidth_for_bound()`, not
        // `mem.bandwidth`. On MI350X/MI355X those are 6200 GB/s (measured whole-GPU
        // streaming read, runtime/amd/op_gemm.h:38) and 8000 GB/s (datasheet). This
        // divides the weight stream, so the datasheet number made the bound 8000/6200
        // = 1.29x too SMALL: Gemma-4-31B decode read 7719.3 µs where the measured
        // denominator gives 9.96 ms. A lower bound that is 22.5% optimistic is worse
        // than no lower bound — it reports headroom that does not exist, and the
        // measured GEMV already runs at 95–103% of the 6200 ceiling.
        let bw_bytes_per_cycle =
            (spec.mem.bandwidth_for_bound().0 * 1e9 / spec.clock_boost.0 as f64) as u64;
        Some(devblob_verify_hook(
            lean_verify_on,
            lean_oracle_on,
            bw_bytes_per_cycle,
            spec.clock_boost.0,
        )?)
    };
    let verify = match (fusion_coverage, verify) {
        (Some(coverage), Some(inner)) => Some(fusion_coverage_hook(coverage, inner)),
        (None, hook) => hook,
        (Some(_), None) => unreachable!("devblob emission always constructs a verification hook"),
    };
    // `PLOW_L2_PLACE=1`: L2-domain-aware placement. The whole layout — SMs per L2 partition and
    // the partition count — is resolved from `hwspec` (XCD on MI300/MI350, GPC on H100/B200),
    // not from an env-supplied constant; the flag only says whether to use it. `None` (flag
    // unset, or an unpartitioned GPU such as consumer Blackwell) ⇒ byte-identical blob. The
    // physical-SM dispatch that consumes this is a runtime/interp feature —
    // see the design notes.
    //
    // WAS `PLOW_NV_PLACE`, still accepted. The concept is vendor-neutral — an L2 domain is a GPC
    // on NVIDIA and an XCD on AMD — and the NVIDIA-specific name was reading as "this is an
    // NVIDIA feature" for something `hwspec` describes on both vendors. The old spelling stays
    // live so in-flight scripts and recipes do not break.
    let l2_on = |k: &str| std::env::var(k).ok().as_deref() == Some("1");
    let l2_off = |k: &str| std::env::var(k).ok().as_deref() == Some("0");
    // DEFAULT ON FOR gfx942/gfx950. Per-XCD queues are the shipped CDNA decode/prefill path,
    // knob: windowing the global queue by L2 domain drains EIGHT queues concurrently instead of
    // one, and it is what makes the two-level gate (PLOW_GATE_HIER) expressible. MEASURED on
    // MI300X, Gemma-4-12B fp8 decode: placement -1.5%, hierarchy -16.0% on top -- the largest
    // win on that arch by a wide margin. Opt out with PLOW_L2_PLACE=0.
    //
    // Both shipping AMD object recipes carry the checked dispatch marker. Other architectures
    // remain explicit opt-in until their physical-domain mapping is measured.
    let l2_default = matches!(cli.arch.as_str(), "gfx942" | "gfx950") && !l2_off("PLOW_L2_PLACE");
    let l2_layout = if l2_default || l2_on("PLOW_L2_PLACE") || l2_on("PLOW_NV_PLACE") {
        hwspec::registry::lookup(&cli.gpu)
            .and_then(|s| s.l2_partitioning.as_ref())
            // WHICH WORKGROUP LANDS ON WHICH DOMAIN IS VENDOR-SPECIFIC, and getting it from the
            // vendor rather than assuming it is the point of this whole type. NVIDIA fills a GPC
            // with consecutive blocks; AMD's dispatcher assigns workgroups to XCDs round-robin,
            // MEASURED at 100.0% on MI355X (runtime/tests/xcd_map_gfx950_test.hip). Using the
            // block formula on AMD would place packets on workgroups the hardware has scattered
            // across all eight XCDs — correct tokens, inverted locality, invisible to any test.
            .map(|p| packet::devbuild::L2Layout {
                sms: p.sms_per_partition,
                domains: p.partition_count,
                map: match hwspec::registry::lookup(&cli.gpu).map(|s| s.vendor) {
                    Some(hwspec::spec::Vendor::Amd) => packet::devbuild::L2Map::RoundRobin,
                    _ => packet::devbuild::L2Map::Block,
                },
            })
    } else {
        None
    };
    // `--no-tuning` / `--tuning-db` must reach the AMD emitters too, and they did not.
    //
    // Both flags fed only `plowc`'s generic `CompilerOracle` path, while `devgen::pick_tile` —
    // the selector every gfx950 dense model actually goes through — read the store on its own.
    // So `--no-tuning`, whose documented contract is "the compiled output is identical to the
    // pre-tuner compiler", did not disable the one tuner that was choosing tiles. An escape
    // hatch nobody can reach is worse than none: it is a documented promise that silently does
    // not hold. Carried through the environment because that is how every other compiler-side
    // knob in `devgen` is read (`PLOW_BLOCK`, `PLOW_FA_GF_FULL`, `PLOW_MXFP4`, …); an empty
    // value means "no store", which is what `--no-tuning` asks for.
    std::env::set_var(
        "PLOW_TUNEDB",
        if cli.no_tuning {
            String::new()
        } else {
            cli.tuning_db
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| "tuning".to_string())
        },
    );
    devgen::run_verified(
        devgen::EmitArgs {
            dir: dir.clone(),
            ctx: cli.max_ctx,
            out: pkt.to_str().ok_or("non-UTF8 output path")?.to_string(),
            n_cu,
            tp,
            block_spec,
            embed_cubin: cli.embed_cubin.clone(),
            embed_hsaco: cli.embed_hsaco.clone(),
            rope_gen: !cli.no_rope_gen,
            l2_layout,
            gpu: cli.gpu.clone(),
            arch: cli.arch.clone(),
            emit_cfg: Some({
                let mut cfg = cli.emit_cfg.clone();
                if cli.segmented {
                    cfg.uniseg = false;
                }
                cfg
            }),
            whole_graph_fusions,
        },
        verify,
    );

    // `--emit devblob+cubin`: build the object the packet we just wrote needs.
    // Runs AFTER emission, from the manifest that emission produced — never from
    // the CLI's idea of what was emitted, which is the drift this whole change
    // exists to remove.
    if cli.emit() == EmitKind::DevblobCubin {
        build_cubin_from_manifest(&pkt, &cli.arch, cli.segmented)?;
    }

    // Bare-blob mode (`--out foo.pkt`) stops here: no manifest, exactly the
    // legacy `gemma4` output. Bundle mode also writes a servable manifest.
    if !out_is_pkt {
        // The GPU program lives in the device blob, so the bucket list is empty
        // — the runtime's model manager keys off the PLOWDEV blob, and the
        // manifest supplies the network slug + target for the registry. Mirrors
        // the hand-written stub the build scripts used to emit.
        let manifest = plow_asset::Manifest {
            network: slug.clone(),
            gpu: cli.gpu.clone(),
            num_gpus: cli.num_gpus,
            parallel: format!("{:?}", cli.parallel).to_lowercase(),
            weight_shared: false,
            weight: None,
            kv: None,
            fusion: None,
            buckets: Vec::new(),
            static_tensors: Vec::new(),
            static_tensors_file_emitted: false,
            weight_tiling: None,
        };
        std::fs::write(
            out_dir.join("weights.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;

        // A bundle is only servable if the runtime can find the checkpoint and a
        // real tokenizer. `plowrt serve` looks for `<assets>/checkpoint` and
        // `<assets>/tokenizer.json`, so symlink both directly at the checkpoint
        // — no copy, no manual completion step.
        let ckpt = std::fs::canonicalize(&dir)?;
        symlink_force(&ckpt, &out_dir.join("checkpoint"))?;
        let tok = ckpt.join("tokenizer.json");
        if tok.exists() {
            symlink_force(&tok, &out_dir.join("tokenizer.json"))?;
        } else {
            warn!(
                checkpoint = %ckpt.display(),
                "no tokenizer.json in the checkpoint; the bundle will fall back \
                 to the byte tokenizer until one is provided"
            );
        }
    }

    info!(
        slug = %slug, n_cu, tp, bundle = !out_is_pkt, out = %pkt.display(),
        "devblob written"
    );
    Ok(pkt)
}

fn ensure_devblob_arch_supported(config_json: &str) -> Result<(), String> {
    let v: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| format!("invalid config.json: {e}"))?;
    if v["model_type"].as_str() == Some("qwen3_5")
        && (v.get("quantization_config").is_some()
            || v["text_config"].get("quantization_config").is_some())
    {
        return Err("qwen3_5 native checkpoint FP8 lowering is not implemented yet".into());
    }
    Ok(())
}

/// `--emit devblob+cubin`: drive the Phase-A CMake target with the defines the
/// manifest asked for.
///
/// The manifest is the ONLY input. `requires` is the correctness half (a missing
/// arm is `default: __trap()` at first launch, which reads as a driver bug) and
/// `recommends` is the performance half (`GV_MM_MAX`, `PLOW_NV_FA_GF_FULL` — both
/// derived by rule from the packet's own shapes, both worth double-digit
/// percentages when wrong).
///
/// The toolkit check is up front and by name. `plowc --emit devblob` must keep
/// working on a machine with no CUDA at all — the README promises exactly that —
/// so the failure here has to be a clear sentence, not a cmake backtrace.
fn build_cubin_from_manifest(
    pkt: &std::path::Path,
    arch: &str,
    segmented: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mpath = pkt.with_file_name("build.json");
    let man: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&mpath)
            .map_err(|e| format!("--emit devblob+cubin: cannot read {}: {e}", mpath.display()))?,
    )?;

    let nvcc = std::path::Path::new("/usr/local/cuda/bin/nvcc");
    if !nvcc.exists() {
        return Err(format!(
            "--emit devblob+cubin needs a CUDA toolkit: {} not found. The packet and \
             {} were written successfully — build the object separately (see \
             runtime/CMakeLists.txt, -DPLOW_SM120_CUBIN=ON) or use --emit devblob, \
             which needs no toolkit.",
            nvcc.display(),
            mpath.display()
        )
        .into());
    }
    if which_cmake().is_none() {
        return Err(format!(
            "--emit devblob+cubin needs `cmake` on PATH (the packet and {} were \
             written successfully). --emit devblob needs no toolchain at all.",
            mpath.display()
        )
        .into());
    }

    // The manifest is arch-agnostic on purpose; picking the backend is this
    // function's job. Only nvcc/sm_1xx is wired — hipcc → .hsaco (runtime/amd/)
    // is the same shape and is deliberately left for a follow-up.
    let flags = man
        .get("backends")
        .and_then(|b| b.get("nvcc"))
        .ok_or_else(|| format!("{}: no nvcc backend section", mpath.display()))?;
    if !arch.starts_with("sm_") {
        return Err(format!(
            "--emit devblob+cubin: only the nvcc backend is wired; --arch {arch} would \
             need the hipcc/.hsaco backend (runtime/amd/), which is not implemented."
        )
        .into());
    }
    let list = |k: &str| -> Vec<String> {
        flags
            .get(k)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let (req, rec) = (list("requires"), list("recommends"));
    info!(requires = ?req, recommends = ?rec, "building interpreter object from manifest");

    // `requires` maps onto the CMake options; `recommends` are raw defines the
    // table appends verbatim. PLOW_NV_FA_GF_FULL has its own cache variable (it
    // must reach every decode-family object from one place — that was bug #2),
    // so route it there rather than into the raw-append bucket.
    let mut args: Vec<String> = vec!["-DPLOW_SM120_CUBIN=ON".into()];
    if !req.iter().any(|d| d.starts_with("PLOW_NV_GEMMA")) {
        args.push("-DPLOW_CUBIN_GEMMA=OFF".into());
    }
    if req.iter().any(|d| d.starts_with("PLOW_NV_W8A8")) {
        args.push("-DPLOW_NV_W8A8=ON".into());
    }
    if req.iter().any(|d| d.starts_with("PLOW_FP8_KV")) {
        args.push("-DPLOW_FP8_KV=ON".into());
        args.push("-DPLOW_SM120_CUBIN_FP8KV=ON".into());
    }
    if segmented {
        args.push("-DPLOW_SM120_CUBIN_SEG=ON".into());
    }
    let mut extra = Vec::new();
    for d in &rec {
        match d.split_once('=') {
            Some(("PLOW_NV_FA_GF_FULL", v)) => args.push(format!("-DPLOW_NV_FA_GF_FULL={v}")),
            _ => extra.push(format!("-D{d}")),
        }
    }
    if !extra.is_empty() {
        args.push(format!("-DPLOW_EXTRA_DEFINES={}", extra.join(" ")));
    }
    args.push(format!("-DPLOW_CUBIN_ARCH={arch}"));

    let out_dir = pkt.parent().map(PathBuf::from).unwrap_or_default();
    let build_dir = out_dir.join(".cubin-build");
    let runtime_dir = repo_runtime_dir()?;
    args.push(format!("-DPLOW_CUBIN_DIR={}", out_dir.display()));

    let status = std::process::Command::new("cmake")
        .arg("-S")
        .arg(&runtime_dir)
        .arg("-B")
        .arg(&build_dir)
        .args(&args)
        .status()?;
    if !status.success() {
        return Err(format!("cmake configure failed ({status})").into());
    }
    let status = std::process::Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("sm120_cubins")
        .status()?;
    if !status.success() {
        return Err(format!("cmake build failed ({status})").into());
    }
    info!(out = %out_dir.display(), "interpreter object built");
    Ok(())
}

fn which_cmake() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|p| {
        std::env::split_paths(&p)
            .map(|d| d.join("cmake"))
            .find(|c| c.is_file())
    })
}

/// Locate `runtime/` — `PLOW_ROOT` if set (a worktree builds its OWN source),
/// else walk up from the executable, else the current directory.
fn repo_runtime_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(r) = std::env::var("PLOW_ROOT") {
        let d = PathBuf::from(r).join("runtime");
        if d.join("CMakeLists.txt").exists() {
            return Ok(d);
        }
    }
    let mut cur = std::env::current_dir()?;
    loop {
        let d = cur.join("runtime");
        if d.join("CMakeLists.txt").exists() {
            return Ok(d);
        }
        if !cur.pop() {
            break;
        }
    }
    Err(
        "--emit devblob+cubin: cannot find runtime/CMakeLists.txt — set PLOW_ROOT \
         to the repository root"
            .into(),
    )
}

/// Create `link` → `target`, replacing any existing entry. Unix-only; the
/// runtime this serves is Linux/CUDA/ROCm.
#[cfg(unix)]
fn symlink_force(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    let _ = std::fs::remove_file(link);
    std::os::unix::fs::symlink(target, link)
}
#[cfg(not(unix))]
fn symlink_force(_target: &std::path::Path, _link: &std::path::Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "devblob bundle symlinks require a Unix host",
    ))
}

fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("plowc=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}

// Same pattern as `--lean-verify` without `lean-verify`: the CLI surface
// stays stable, the un-featured build answers with an error at runtime.
#[cfg(not(feature = "tuner"))]
fn run_tune(_t: &TuneCli, _cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    Err("this plowc was built without the `tuner` feature; \
         rebuild with `--features tuner` to use `plowc tune`"
        .into())
}

#[cfg(feature = "tuner")]
fn run_tune(t: &TuneCli, cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    // The compile whose demand is being observed, from the TOP-LEVEL flags — the same words the
    // build states it in. `None` when no checkpoint was given, which is what makes the
    // derive-based actions refuse rather than silently derive some default model's demand.
    let emit_spec = |req: &str| -> Result<tune::demand::EmitSpec, Box<dyn std::error::Error>> {
        let dir = cli.hf_dir.clone().ok_or_else(|| {
            format!(
                "{req} derives the compiler's demand by running a real emit, so it needs a \
                 checkpoint: `plowc --hf-dir <ckpt> --max-ctx <c> --n-cu <n> --num-gpus <g> tune \
                 …`. The flags go BEFORE the `tune` word because they are the compile's flags."
            )
        })?;
        Ok(tune::demand::EmitSpec {
            hf_dir: dir,
            ctx: cli.max_ctx,
            n_cu: if cli.n_cu > 0 {
                cli.n_cu
            } else {
                hwspec::registry::lookup(&cli.gpu)
                    .map(|s| s.sm_count)
                    .ok_or_else(|| format!("unknown GPU {:?}; pass --n-cu explicitly", cli.gpu))?
            },
            tp: cli.num_gpus.max(1) as u32,
            gpu: cli.gpu.clone(),
            arch: cli.arch.clone(),
            db: t.db.clone(),
        })
    };

    // `--status` / `--shape` are the pre-subcommand spellings, kept live.
    let word = t.action.clone().unwrap_or_else(|| {
        if t.status {
            "status".into()
        } else if t.shape.is_some() {
            "select".into()
        } else {
            "inventory".into()
        }
    });

    let action = match word.as_str() {
        "inventory" => TuneAction::Inventory,
        "select" => {
            let s = t
                .shape
                .as_ref()
                .ok_or("`tune select` needs --shape M,N,K")?;
            let (m, n, k) = tune::parse_shape(s)?;
            TuneAction::Select { m, n, k }
        }
        // Coverage is optional here on purpose: the digest census and the total-staleness alarm
        // are the part that outranks coverage, and they need no checkpoint at all.
        "status" => TuneAction::Status {
            coverage_from: cli
                .hf_dir
                .as_ref()
                .map(|_| emit_spec("coverage"))
                .transpose()?,
        },
        "regress" => TuneAction::Regress {
            threshold: t.threshold,
        },
        "shapes" => TuneAction::Shapes(emit_spec("`tune shapes`")?),
        "best" => TuneAction::Best {
            quant: t.quant.clone(),
        },
        "ingest" => TuneAction::Ingest {
            samples: t
                .samples
                .clone()
                .ok_or("`tune ingest` needs --samples <sweep.jsonl>")?,
            campaign: t.campaign.clone(),
            provisional: t.provisional,
        },
        "gemm" => TuneAction::Gemm(Box::new(tune::gemm::Campaign {
            obj: t
                .obj
                .clone()
                .ok_or("`tune gemm` needs --obj <dir with test_kernels.elf>")?,
            samples: t
                .samples
                .clone()
                .ok_or("`tune gemm` needs --samples <out.jsonl> for the raw sample rows")?,
            shapes: if t.shapes == "auto" {
                tune::gemm::ShapeSource::Auto(emit_spec("`--shapes auto`")?)
            } else {
                tune::gemm::ShapeSource::File(PathBuf::from(&t.shapes))
            },
            lease: t.lease,
            no_ingest: t.no_ingest,
            db: t.db.clone(),
            campaign: t.campaign.clone(),
            provisional: t.provisional,
            dry_run: t.dry_run,
        })),
        other => {
            return Err(format!(
                "unknown tune action {other:?}; expected one of: \
                 inventory, select, status, regress, shapes, gemm, ingest, best"
            )
            .into())
        }
    };

    tune::run(&TuneOptions {
        root: t.root.clone(),
        gpu: t.gpu.clone(),
        profile: tune::parse_profile(&t.profile)?,
        db: t.db.clone(),
        action,
    })
}

fn run(cli: Cli) -> Result<Report, Box<dyn std::error::Error>> {
    let source = match (&cli.model, &cli.net, &cli.hf_dir) {
        (Some(id), _, _) => Source::Model(id.clone()),
        (_, Some(path), _) => {
            let json = std::fs::read_to_string(path)?;
            Source::Net(serde_json::from_str::<NetConfig>(&json)?)
        }
        (_, _, Some(dir)) => Source::HfDir(dir.clone()),
        (None, None, None) => {
            return Err("one of --model, --net, or --hf-dir is required".into());
        }
    };

    // Resolve batch/seq from preset if given, else use CLI defaults.
    let (batches, seqs) = if let Some(preset) = cli.preset {
        (preset.batches(), preset.seqs())
    } else {
        (cli.batch, cli.seq)
    };

    info!(
        network = %source.name(),
        batches = ?batches,
        ctx_lengths = ?seqs,
        phases = ?cli.phase.phases(),
        "bucket ladder resolved"
    );

    // Derive output directory: plow-out/<model-slug>/ when --out not explicit.
    let out = cli.out.unwrap_or_else(|| {
        let slug = source.name();
        PathBuf::from("plow-out").join(slug)
    });

    let opts = Options {
        no_tuning: cli.no_tuning,
        tuning_db: cli.tuning_db,
        gpu: cli.gpu,
        num_gpus: cli.num_gpus,
        parallel: cli.parallel,
        batches,
        seqs,
        phases: cli.phase.phases(),
        page_kib: cli.page_kib,
        out,
        // PACKET PATH. `Options::lean_verify` is a LIBRARY opt-in and keeps its
        // hard-fail contract (`PlowcError::LeanVerifyDisabled` / `LeanVerifySpawn`)
        // — `tests/lean_verify_disabled.rs` pins that. The default-on + degrade
        // policy therefore lives HERE, at the CLI, which is the layer that has a
        // default to speak of.
        //
        // Note this path runs the verifier ONCE PER BUCKET (a `run_lean_verify`
        // call per `{phase}_b{batch}_s{seq}` stem, checkpoints A/B/D/E/F each
        // time), so a per-bucket spawn against a missing binary is exactly what
        // resolving the flag once, up front, avoids.
        lean_verify: cli.lean_verify && lean_unavailable_reason().is_none(),
        counter_elim: cli.counter_elim,
        scope_narrow: cli.scope_narrow,
        prefetch: cli.prefetch,
        sram_fit: cli.sram_fit,
        lean_oracle: cli.lean_oracle && lean_unavailable_reason().is_none(),
        emit_sample: cli.emit_sample,
        emit_tokenize: cli.emit_tokenize,
        emit_trace: cli.emit_trace,
        kv: plowc::KvConfig {
            block_tokens: cli.kv_block_tokens,
            initial_blocks: cli.kv_initial_blocks,
        },
        weight_dtype_override: match cli.weight_dtype.as_str() {
            "auto" => None,
            "bf16" => Some(nn_graph::DType::BF16),
            "fp8" => Some(nn_graph::DType::F8E4M3),
            "f4" | "mx" => Some(nn_graph::DType::F4),
            other => {
                return Err(format!(
                    "unknown --weight-dtype {other:?}: expected auto, bf16, fp8, or f4"
                )
                .into());
            }
        },
    };
    Ok(compile(&source, &opts)?)
}

/// Print the compiler-pass statistics + runtime estimates as a table.
fn print_report(r: &Report) {
    println!("network   {}", r.network);
    println!("tuning    {} — {}", r.tuning_tier, r.tuning_provenance);
    println!("target    {} × {}  ({:?})", r.num_gpus, r.gpu, r.parallel);
    if let Some(f) = &r.fusion {
        println!(
            "fusion    {} ops → {} ops ({} fused)",
            f.ops_before, f.ops_after, f.fused
        );
    }
    match &r.weight {
        Some(w) => println!(
            "weights   shared={}  layout=(bn={}, bk={})",
            r.weight_shared, w.bn, w.bk
        ),
        None => println!("weights   (no GEMM weights to lay out)"),
    }
    if let Some(k) = &r.kv {
        println!(
            "kv-cache  block_seq={} kv_heads={} head_dim={}",
            k.block_seq, k.kv_heads, k.head_dim
        );
    }
    println!();
    println!(
        "{:<8} {:>6} {:>6} {:>7} {:>6} {:>6} {:>10} {:>12} {:>20}",
        "phase",
        "batch",
        "seq",
        "tiles",
        "tasks",
        "insts",
        "pkt-bytes",
        "makespan",
        "ideal (lost%)"
    );
    for b in &r.buckets {
        let lost = if b.makespan > 0 {
            100.0 * (b.makespan.saturating_sub(b.ideal_makespan)) as f64 / b.makespan as f64
        } else {
            0.0
        };
        println!(
            "{:<8} {:>6} {:>6} {:>7} {:>6} {:>6} {:>10} {:>12} {:>12} ({:>4.1}%)",
            b.phase,
            b.batch,
            b.seq,
            b.tile_nodes,
            b.tasks,
            b.instructions,
            b.packet_bytes,
            b.makespan,
            b.ideal_makespan,
            lost
        );
    }
    println!("\nwrote {} packet streams + weights.json", r.buckets.len());

    if let Some(a) = &r.assets {
        println!();
        println!("hbm memory regions (peak across compiled buckets):");
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        println!("  weights          {:>10.2} MB", mb(a.regions.weights));
        println!(
            "  kv-cache (peak)  {:>10.2} MB",
            mb(a.regions.kv_cache_peak)
        );
        println!("  scratch  (peak)  {:>10.2} MB", mb(a.regions.scratch_peak));
        println!(
            "  request-io (peak){:>10.2} MB",
            mb(a.regions.request_io_peak)
        );
        if a.regions.static_ > 0 {
            println!("  static           {:>10.2} MB", mb(a.regions.static_));
        }
        if a.regions.persistent > 0 {
            println!("  persistent       {:>10.2} MB", mb(a.regions.persistent));
        }
        println!(
            "  total (peak)     {:>10.2} MB / {:>7.2} GB HBM  (headroom {:.2} GB)",
            mb(a.regions.total_hbm_peak),
            a.regions.hbm_capacity as f64 / 1e9,
            a.regions.hbm_headroom as f64 / 1e9,
        );
        println!(
            "  peak bucket:     phase={} batch={} seq={}",
            a.regions.peak_bucket.phase, a.regions.peak_bucket.batch, a.regions.peak_bucket.seq
        );

        println!();
        println!("on-disk artifacts:");
        let kb = |b: u64| b as f64 / 1024.0;
        println!(
            "  packets total    {:>10.1} KB",
            kb(a.on_disk.packets_total)
        );
        println!(
            "  map.json total   {:>10.1} KB",
            kb(a.on_disk.map_json_total)
        );
        if a.on_disk.trace_json_total > 0 {
            println!(
                "  trace.json total {:>10.1} KB",
                kb(a.on_disk.trace_json_total)
            );
        }
        println!("  weights.json     {:>10.1} KB", kb(a.on_disk.weights_json));
        println!(
            "  footprint j+csv  {:>10.1} KB",
            kb(a.on_disk.footprint_json + a.on_disk.footprint_csv)
        );
        println!("  grand total      {:>10.1} KB", kb(a.on_disk.grand_total));
    }
}

/// Print every recognized GPU name and its short aliases.
fn print_gpu_list() {
    println!("Recognized GPU specs (--gpu accepts any name or alias, case-insensitive):\n");
    for spec in hwspec::registry::ALL {
        let aliases: Vec<&str> = hwspec::registry::ALIASES
            .iter()
            .filter(|(_, canon)| canon.eq_ignore_ascii_case(spec.name))
            .map(|(alias, _)| *alias)
            .collect();
        if aliases.is_empty() {
            println!("  {}", spec.name);
        } else {
            println!("  {:30} aliases: {}", spec.name, aliases.join(", "));
        }
    }
}

/// The kimi_k3 builder refuses a `vision_config` because a text-only *compile*
/// would load, run, and be silently wrong on image prompts. A viz is a picture,
/// not a runnable artifact, so that failure mode does not exist here: strip the
/// key and draw the text tower, loudly. Scoped to kimi_k3 — gemma4 multimodal
/// *dispatches* on `vision_config` and must keep seeing it.
fn viz_strip_vision(json: &str) -> String {
    let Ok(mut v) = serde_json::from_str::<serde_json::Value>(json) else {
        return json.to_string();
    };
    if v.get("model_type").and_then(|m| m.as_str()) == Some("kimi_k3")
        && v.get("vision_config")
            .map(|c| !c.is_null())
            .unwrap_or(false)
    {
        warn!("kimi_k3 vision_config ignored for viz: drawing the TEXT tower only");
        v.as_object_mut().map(|o| o.remove("vision_config"));
        return v.to_string();
    }
    json.to_string()
}

/// Build a graph for viz from a single source string: a checkpoint directory
/// (containing `config.json`), a config/net JSON file, or an HF model id.
/// `batch`/`seq` bind the symbolic B/S so every inferred shape comes out
/// concrete; absent, shapes stay symbolic.
fn viz_build(
    src: &str,
    batch: Option<i64>,
    seq: Option<i64>,
) -> Result<(nn_graph::Graph, String), String> {
    let p = std::path::Path::new(src);
    let (json, title) = if p.is_dir() {
        let json = std::fs::read_to_string(p.join("config.json"))
            .map_err(|e| format!("cannot read config.json in {src:?}: {e}"))?;
        (json, plowc::hf_config::dir_slug(p))
    } else if p.is_file() {
        let json = std::fs::read_to_string(p).map_err(|e| format!("cannot read {src:?}: {e}"))?;
        let title = p
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "network".into());
        (json, title)
    } else {
        info!(model = %src, "resolving model config from HuggingFace hub");
        let json = nn_graph::hub::fetch_config(src)
            .map_err(|e| format!("hub fetch failed for {src:?}: {e}"))?;
        (json, src.rsplit('/').next().unwrap_or(src).to_string())
    };

    let json = viz_strip_vision(&json);
    let mut g = nn_graph::models::build_from_config_json_at(
        &json,
        &nn_graph::models::ShapeBucket::default(),
    )
    .map_err(|e| format!("graph build failed for {src:?}: {e}"))?;

    let mut b = nn_graph::Bindings::new();
    if let Some(v) = batch {
        b.insert("B", v);
    }
    if let Some(v) = seq {
        b.insert("S", v);
    }
    g.bind(&b);
    Ok((g, title))
}

/// Percent-decode a URL query value (enough for model ids and paths).
fn urldecode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' => {
                let hex = |c: u8| (c as char).to_digit(16);
                if let (Some(h), Some(l)) = (
                    b.get(i + 1).copied().and_then(hex),
                    b.get(i + 2).copied().and_then(hex),
                ) {
                    out.push((h * 16 + l) as u8);
                    i += 3;
                    continue;
                }
                out.push(b'%');
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `plowc viz`: build the nn-graph from the given source and either serve
/// or dump a self-contained HTML visualization.
///
/// Sources, in priority order: `--hf-dir`, `--model` (HF hub), `--net`.
/// In serve mode the page's model form hits `GET /graph?model=…&batch=…&seq=…`
/// and the server rebuilds — any model, any B/S binding, without restarting.
fn run_viz(v: &VizCli, cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let source = if let Some(ref dir) = cli.hf_dir {
        dir.to_string_lossy().into_owned()
    } else if let Some(ref model_id) = cli.model {
        model_id.clone()
    } else if let Some(ref net_path) = cli.net {
        net_path.to_string_lossy().into_owned()
    } else {
        return Err(
            "viz requires a model source: --hf-dir <path>, --model <hf-id>, or --net <file.json>"
                .into(),
        );
    };

    let (graph, title) = viz_build(&source, None, None)?;
    info!(
        title = %title,
        tensors = graph.tensors.len(),
        nodes = graph.nodes.len(),
        blocks = graph.blocks.len(),
        "graph built for visualization"
    );

    let html = nn_graph::viz::graph_to_html(&graph, &title, &source);

    if v.port == 0 {
        // File-dump mode
        std::fs::write(&v.out, &html)?;
        info!(path = %v.out.display(), "wrote graph visualization HTML");
        return Ok(());
    }

    // Serve mode: tiny stdlib HTTP server
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let addr = format!("127.0.0.1:{}", v.port);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("cannot bind {addr}: {e}"))?;
    let url = format!("http://{addr}");
    info!(url = %url, "serving nn-graph visualizer (Ctrl-C to stop)");
    eprintln!("\n  🌐 Graph viewer: {url}\n");

    if v.open {
        // Best-effort open in browser
        #[cfg(target_os = "macos")]
        {
            let _ = std::process::Command::new("open").arg(&url).spawn();
        }
        #[cfg(target_os = "linux")]
        {
            let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
        }
    }

    let html_bytes = html.into_bytes();
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).unwrap_or(0);
        let req = String::from_utf8_lossy(&buf[..n]);
        let path = req
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or("/");

        let (content_type, body): (&str, Vec<u8>) = if let Some(query) = path
            .strip_prefix("/graph")
            .and_then(|r| r.strip_prefix('?'))
        {
            // /graph?model=…&batch=…&seq=… → rebuild and return JSON.
            let mut model = String::new();
            let (mut batch, mut seq) = (None, None);
            for kv in query.split('&') {
                let mut it = kv.splitn(2, '=');
                match (it.next(), it.next()) {
                    (Some("model"), Some(val)) => model = urldecode(val),
                    (Some("batch"), Some(val)) => batch = urldecode(val).parse().ok(),
                    (Some("seq"), Some(val)) => seq = urldecode(val).parse().ok(),
                    _ => {}
                }
            }
            let src = if model.is_empty() {
                source.clone()
            } else {
                model
            };
            let json = match viz_build(&src, batch, seq) {
                Ok((g, t)) => {
                    info!(source = %src, ?batch, ?seq, "rebuilt graph for viz request");
                    serde_json::json!({ "title": t, "graph": nn_graph::viz::graph_to_value(&g) })
                }
                Err(e) => {
                    error!(source = %src, error = %e, "viz rebuild failed");
                    serde_json::json!({ "error": e })
                }
            };
            ("application/json", json.to_string().into_bytes())
        } else {
            ("text/html; charset=utf-8", html_bytes.clone())
        };

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(&body);
        let _ = stream.flush();
    }
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    use clap::Parser;

    fn parse(extra: &[&str]) -> Cli {
        let mut argv = vec!["plowc", "--hf-dir", "/tmp/x", "--emit", "devblob"];
        argv.extend_from_slice(extra);
        Cli::try_parse_from(argv).expect("parse")
    }

    #[test]
    fn qwen35_devblob_routes_bf16_and_rejects_unimplemented_fp8() {
        ensure_devblob_arch_supported(r#"{"model_type":"qwen3_5"}"#).unwrap();
        ensure_devblob_arch_supported(r#"{"model_type":"qwen3"}"#).unwrap();
        let err = ensure_devblob_arch_supported(
            r#"{"model_type":"qwen3_5","quantization_config":{"quant_method":"fp8"}}"#,
        )
        .unwrap_err();
        assert!(err.contains("FP8 lowering"));
    }

    /// CORRECTION 1, HALF ONE. Both gates are ON with no flags. They used to be
    /// `default_value_t = false` behind a non-default cargo feature, so they
    /// shipped in nothing.
    #[test]
    fn both_lean_gates_default_off_pending_the_gemma_rejection() {
        // Was `both_lean_gates_default_on`. Default-on was BACKED OUT because it
        // made plowc refuse to compile Gemma-4-12B outright:
        //   "lean verifier rejected bucket `decode_b1_s1`: reader/writer sets
        //    overlap — strict AddressMapSound not derivable"
        // That is the gate working (a plow_verify binary IS present on gfx950,
        // so the degrade path never ran and a real verdict came back), but the
        // verdict is unresolved. Shipping it on would block every Gemma compile;
        // downgrading the rejection to a warning would be the silently-skipped
        // gate this whole change exists to prevent. So: OFF, opt in explicitly,
        // and a rejection stays fatal for whoever opts in.
        let c = parse(&[]);
        assert!(!c.lean_verify);
        assert!(!c.lean_oracle);
        // The opt-in still works and is still independent per subsystem.
        assert!(parse(&["--lean-verify"]).lean_verify);
        assert!(parse(&["--lean-oracle"]).lean_oracle);
    }

    #[test]
    fn parallel_linear2_experiment_defaults_off_and_is_explicit() {
        assert!(!parse(&[]).experiment_parallel_linear2);
        assert!(parse(&["--experiment-parallel-linear2"]).experiment_parallel_linear2);
    }

    #[test]
    fn parallel_linear2_experiment_without_full_graph_candidate_is_inert() {
        assert_eq!(
            whole_graph_fusion_decisions(None, 8, true),
            devgen::WholeGraphFusionDecisions::default()
        );
    }

    /// CORRECTION 1, HALF TWO — THE TRAP THIS PINS SHUT.
    ///
    /// The supplied design paired `lean_verify: bool` (`default_value_t = true`)
    /// with a SEPARATE `no_lean_verify` field, then computed
    /// `do_oracle = lean_oracle && !no_lean_verify`. That made `--lean-verify`
    /// incapable of changing anything and — undocumented — made
    /// `--no-lean-verify` silently switch the ORACLE off too.
    ///
    /// One switch per subsystem, no coupling in either direction.
    #[test]
    fn disabling_one_gate_never_disables_the_other() {
        // Same invariant, now expressed on the opt-IN spellings: enabling one
        // subsystem must never enable or disable the other. The trap being
        // pinned shut is unchanged — it is the COUPLING that must not exist,
        // not the polarity.
        let v = parse(&["--lean-verify"]);
        assert!(v.lean_verify);
        assert!(!v.lean_oracle, "--lean-verify must NOT enable the oracle");

        let o = parse(&["--lean-oracle"]);
        assert!(!o.lean_verify, "--lean-oracle must NOT enable verification");
        assert!(o.lean_oracle);

        let both = parse(&["--lean-verify", "--lean-oracle"]);
        assert!(both.lean_verify);
        assert!(both.lean_oracle);
    }

    /// Exactly ONE spelling per subsystem. A stale `--lean-verify` in a script
    /// must fail loudly rather than parse into a field it cannot act on — that
    /// ambiguity is precisely what Correction 1 removed.
    #[test]
    fn the_negative_spellings_are_gone() {
        // Polarity flipped with the default, but the rule is the same one:
        // exactly ONE spelling per subsystem, so a stale flag fails loudly
        // instead of parsing into a field it cannot act on.
        assert!(Cli::try_parse_from(["plowc", "--hf-dir", "/tmp/x", "--no-lean-verify"]).is_err());
        assert!(Cli::try_parse_from(["plowc", "--hf-dir", "/tmp/x", "--no-lean-oracle"]).is_err());
    }

    /// A skip must always carry a reason. `lean.verified: false` with a null
    /// reason is the `tuning.tier == "portable"` ambiguity all over again.
    #[test]
    fn an_absent_verifier_yields_a_skip_hook_that_states_why() {
        std::env::set_var("PLOW_VERIFY_BIN", "/nonexistent/plow_verify");
        let why = lean_unavailable_reason().expect("no verifier ⇒ a reason");
        let empty = packet::devbuild::Model {
            n_cu: 1,
            target: 0,
            tensors: vec![],
            progs: vec![],
            kv_row_insts: vec![],
            prog_t: vec![],
            gen: vec![],
        };
        let rep =
            devgen::skip_hook(why)(&empty).expect("a skip is Ok — degrade, never fail the compile");
        assert!(!rep.verified);
        assert!(!rep.oracle);
        assert!(rep.reason.is_some_and(|r| !r.is_empty()));
        std::env::remove_var("PLOW_VERIFY_BIN");
    }
}

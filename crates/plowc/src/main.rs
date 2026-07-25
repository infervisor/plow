//! `plowc` — compile a model or network into runtime packet streams for a GPU.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(feature = "tuner")]
use plowc::tune::{self, TuneAction, TuneOptions};
use plowc::{compile, net::NetConfig, Options, Parallel, Report, Source};
use schedule::Phase;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

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

    /// Artifact form to emit (requires --hf-dir):
    ///   * `packets` (default) — scheduled `.pkt` bucket streams + manifest,
    ///     run on the CPU reference interpreter / simulator;
    ///   * `devblob` — a single PLOWDEV `model.pkt` the GPU runtime executes,
    ///     plus a servable `weights.json`. Replaces the deprecated `gemma4`
    ///     binary; the `PLOW_*` emit knobs (FP8, PLOW_BLOCK, PLOW_UNISEG, …)
    ///     are honored exactly as before.
    #[arg(long, value_enum, default_value_t = EmitKind::Packets)]
    emit: EmitKind,

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

    /// Send each bucket's `(schedule, address_map)` to the Lean verifier
    /// (`plow_verify` CLI). Rejection fails the compile. The binary is located
    /// via `PLOW_VERIFY_BIN` or `lean-plow/.lake/build/bin/plow_verify`.
    #[arg(long, default_value_t = false)]
    lean_verify: bool,

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

    /// Enable the Lean performance oracle. Queries `plow_verify` for
    /// provably-optimal counter granularity, prefetch depth, and lower-bound
    /// certificates. Falls back to Rust heuristics if the binary is unavailable.
    #[arg(long, default_value_t = false)]
    lean_oracle: bool,

    /// §P Emit a host-executor SAMPLE packet at the tail of every decode bucket
    /// (logits → token id, gated on the output-stage counter). Decode-only.
    #[arg(long, default_value_t = false)]
    emit_sample: bool,

    /// §P Emit a host TOKENIZE packet at the graph head (text/ids → tokens).
    #[arg(long, default_value_t = false)]
    emit_tokenize: bool,

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
    /// Scheduled `.pkt` bucket streams + manifest (the default pipeline).
    Packets,
    /// A single PLOWDEV device blob the GPU runtime executes.
    Devblob,
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
}

#[derive(Args, Debug)]
struct TuneCli {
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

    /// Report what the tuning database holds for this target.
    #[arg(long)]
    status: bool,
}

fn main() -> ExitCode {
    init_logging();
    let cli = Cli::parse();

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
        emit = ?cli.emit,
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
        return match run_tune(t) {
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

    if cli.emit == EmitKind::Devblob {
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

/// `--emit devblob`: compile the checkpoint into a single PLOWDEV `model.pkt`
/// via the `devgen` emitter and write a servable `weights.json` next to it, so
/// the output directory is a complete `plowrt serve --assets <dir>` bundle.
/// This is the plowc-native replacement for the standalone `gemma4` binary.
fn run_devblob(cli: &Cli) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir = cli
        .hf_dir
        .clone()
        .ok_or("--emit devblob requires --hf-dir <checkpoint>")?;

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

    devgen::run(devgen::EmitArgs {
        dir: dir.clone(),
        ctx: cli.max_ctx,
        out: pkt.to_str().ok_or("non-UTF8 output path")?.to_string(),
        n_cu,
        tp,
        block_spec,
        embed_cubin: cli.embed_cubin.clone(),
        embed_hsaco: cli.embed_hsaco.clone(),
        rope_gen: !cli.no_rope_gen,
    });

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
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("plowc=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}

// Same pattern as `--lean-verify` without `lean-verify`: the CLI surface
// stays stable, the un-featured build answers with an error at runtime.
#[cfg(not(feature = "tuner"))]
fn run_tune(_t: &TuneCli) -> Result<(), Box<dyn std::error::Error>> {
    Err("this plowc was built without the `tuner` feature; \
         rebuild with `--features tuner` to use `plowc tune`"
        .into())
}

#[cfg(feature = "tuner")]
fn run_tune(t: &TuneCli) -> Result<(), Box<dyn std::error::Error>> {
    let action = if t.status {
        TuneAction::Status
    } else if let Some(s) = &t.shape {
        let (m, n, k) = tune::parse_shape(s)?;
        TuneAction::Select { m, n, k }
    } else {
        TuneAction::Inventory
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
        lean_verify: cli.lean_verify,
        counter_elim: cli.counter_elim,
        scope_narrow: cli.scope_narrow,
        prefetch: cli.prefetch,
        sram_fit: cli.sram_fit,
        lean_oracle: cli.lean_oracle,
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
                ).into());
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
        println!("  kv-cache (peak)  {:>10.2} MB", mb(a.regions.kv_cache_peak));
        println!("  scratch  (peak)  {:>10.2} MB", mb(a.regions.scratch_peak));
        println!("  request-io (peak){:>10.2} MB", mb(a.regions.request_io_peak));
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
        println!("  packets total    {:>10.1} KB", kb(a.on_disk.packets_total));
        println!("  map.json total   {:>10.1} KB", kb(a.on_disk.map_json_total));
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

//! `plowrt` — the plow host runtime CLI.
//!
//! ```text
//! plowrt serve --assets <dir> [--assets <dir> ...] --port 8080
//! ```
//!
//! Each `--assets <dir>` is one compiled model (a directory of `.pkt` +
//! `weights.json` + sidecars). Models are registered by their manifest network
//! name (the API slug). The default build uses the CPU reference backend; the
//! `cuda` / `hsa` features select a GPU backend.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};

use plowrt::device::{self, Backend};
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::mux::{self, MuxConfig};
use plowrt::serve::{app, AppState};

#[derive(Parser)]
#[command(name = "plowrt", about = "plow host runtime")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Load compiled assets and serve the OpenAI-compatible API.
    Serve {
        /// One or more compiled-model directories.
        #[arg(long = "assets", required = true)]
        assets: Vec<PathBuf>,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Optional Unix domain socket to also listen on (opt-in). Serves the
        /// same OpenAI-compatible router as `--port`; both listeners run in
        /// parallel.
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Number of CPU executor threads (reference backend).
        #[arg(long, default_value_t = 8)]
        executors: u32,
        /// Record a per-packet timeline, dumpable at `GET /trace` (§O).
        #[arg(long, default_value_t = false)]
        trace: bool,
        /// Muxer: upper bound on the arrival-rate batch-formation hold (ms).
        #[arg(long, default_value_t = 8.0)]
        max_hold_ms: f64,
        /// Muxer: admission SLO (ms) — predicted wait above this sheds requests.
        #[arg(long, default_value_t = 250.0)]
        slo_ms: f64,
    },

    /// Dry-run the compiled packets (no device): walk each packet honoring
    /// counters, log what it would do, and report timing + a Chrome trace.
    Simulate {
        /// A single compiled-model directory.
        #[arg(long)]
        assets: PathBuf,
        /// Restrict to one bucket, `<phase>:<batch>:<seq>` (e.g. `decode:1:128`).
        #[arg(long)]
        bucket: Option<String>,
        /// Simulate every bucket in the bundle.
        #[arg(long, default_value_t = false)]
        all_buckets: bool,
        /// `dry` (no math) or `golden` (run reference numerics).
        #[arg(long, default_value = "dry")]
        math: String,
        /// Write the per-packet log to this file (default: stdout).
        #[arg(long)]
        log: Option<PathBuf>,
        /// Write the Chrome trace JSON to this file.
        #[arg(long)]
        chrome: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info".into());
    let filter_str = format!("{filter}");
    tracing_subscriber::fmt().with_env_filter(filter).init();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        cuda = cfg!(feature = "cuda"),
        hsa = cfg!(feature = "hsa"),
        hf_tokenizer = cfg!(feature = "hf-tokenizer"),
        log_filter = %filter_str,
        "plowrt starting"
    );

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve {
            assets,
            port,
            socket,
            executors,
            trace,
            max_hold_ms,
            slo_ms,
        } => {
            serve(
                assets,
                port,
                socket,
                executors,
                trace,
                MuxConfig {
                    max_hold_ms,
                    slo_ms,
                    ..MuxConfig::default()
                },
            )
            .await
        }
        Cmd::Simulate {
            assets,
            bucket,
            all_buckets,
            math,
            log,
            chrome,
        } => simulate(assets, bucket, all_buckets, math, log, chrome),
    }
}

fn simulate(
    assets: PathBuf,
    bucket: Option<String>,
    all_buckets: bool,
    math: String,
    log: Option<PathBuf>,
    chrome: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asset::{BucketKey, ModelBundle};
    use plowrt::obs::trace::Timeline;
    use plowrt::sim::{MathMode, Simulator};

    let math = match math.as_str() {
        "golden" => MathMode::Golden,
        "dry" | _ => MathMode::DryRun,
    };
    let bundle = ModelBundle::load(&assets)?;

    // Which buckets to simulate.
    let keys: Vec<BucketKey> = if let Some(spec) = bucket {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 3 {
            return Err(format!("--bucket must be <phase>:<batch>:<seq>, got '{spec}'").into());
        }
        let k = BucketKey::new(parts[0], parts[1].parse()?, parts[2].parse()?);
        vec![k]
    } else if all_buckets {
        bundle.bucket_keys().collect()
    } else {
        // Default: the first bucket.
        bundle.bucket_keys().take(1).collect()
    };
    if keys.is_empty() {
        return Err("no buckets to simulate".into());
    }

    // Per-packet log destination.
    let mut log_out: Box<dyn std::io::Write> = match &log {
        Some(p) => Box::new(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => Box::new(std::io::stdout()),
    };

    let sim = Simulator::new(math);
    let mut combined = Timeline::new();
    let mut any_incomplete = false;

    for key in keys {
        let b = bundle
            .bucket(key)
            .ok_or_else(|| format!("bucket {key:?} not found"))?;
        let mut report = sim.run(&b.program);
        report.compiler_makespan = Some(b.makespan);
        report.compiler_ideal = Some(b.ideal_makespan);

        writeln!(
            log_out,
            "=== bucket {:?} b{} s{} ({} packets) ===",
            key.phase, key.batch, key.seq, report.stats.total
        )?;
        for e in &report.events {
            writeln!(log_out, "{}", e.log_line())?;
        }
        writeln!(log_out, "{}", report.summary())?;

        if chrome.is_some() {
            for span in report.timeline().spans() {
                combined.push(*span);
            }
        }
        any_incomplete |= !report.stats.completed;
    }
    log_out.flush()?;

    if let Some(path) = chrome {
        std::fs::write(&path, combined.to_chrome_json())?;
        eprintln!(
            "wrote Chrome trace ({} spans) to {}",
            combined.len(),
            path.display()
        );
    }

    if any_incomplete {
        return Err("one or more buckets did not complete (deadlock) — see report".into());
    }
    Ok(())
}

async fn serve(
    assets: Vec<PathBuf>,
    port: u16,
    socket: Option<PathBuf>,
    executors: u32,
    trace: bool,
    mux_cfg: MuxConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // One binary, CPU or GPU: the vendor drivers are `dlopen`ed, so this probes
    // CUDA then HSA (AMD) and falls back to the CPU reference backend when neither
    // loads. Assets stay servable either way — every one of them is compiled for
    // a GPU spec, and the CPU backend interprets that same program.
    //
    // The CUDA probe keeps a TYPED handle: the sm_120 engine needs the
    // backend's cooperative-launch surface, which `dyn Backend` erases.
    #[cfg(feature = "cuda")]
    let cuda: Option<Arc<device::cuda::CudaBackend>> = match device::cuda::CudaBackend::new(0) {
        Ok(b) => Some(Arc::new(b)),
        Err(e) => {
            tracing::warn!(%e, "no CUDA backend");
            None
        }
    };
    #[cfg(feature = "cuda")]
    let backend: Arc<dyn Backend> = match &cuda {
        Some(c) => Arc::clone(c) as Arc<dyn Backend>,
        None => device::select(executors),
    };
    #[cfg(not(feature = "cuda"))]
    let backend: Arc<dyn Backend> = device::select(executors);
    let vendor = backend.vendor();
    if vendor.is_some() {
        tracing::info!(class = ?backend.class(), vendor = ?vendor, "backend ready — GPU accelerated");
    } else {
        tracing::warn!("╔══════════════════════════════════════════════════════════════════╗");
        tracing::warn!("║  WARNING: No GPU backend available — falling back to CPU!       ║");
        tracing::warn!("║  Inference will be orders of magnitude slower than GPU.         ║");
        tracing::warn!("║  To use CUDA: build with --features cuda and ensure libcuda.so  ║");
        tracing::warn!("║  is reachable (NVIDIA driver installed), or set PLOW_LIBCUDA.   ║");
        tracing::warn!("╚══════════════════════════════════════════════════════════════════╝");
        tracing::info!(class = ?backend.class(), executors, "CPU reference backend active");
    }
    let execset = Arc::new(ExecutorSet::bringup(backend)?);

    let mut registry = Registry::new();
    for dir in &assets {
        let slug = registry.load(dir, None)?;
        let target = registry.get(&slug)?.manifest.gpu.clone();
        let target_vendor = hwspec::registry::lookup(&target).map(|s| s.vendor);
        if target_vendor.is_some() && target_vendor == vendor {
            tracing::info!(dir = %dir.display(), %target, "loaded model bundle");
        } else {
            tracing::warn!(
                dir = %dir.display(), %target,
                "loaded model bundle — no matching GPU driver; \
                 running on the CPU reference interpreter (unaccelerated)"
            );
        }
    }
    tracing::info!(models = registry.len(), trace, "registry ready");

    let state = Arc::new(AppState::with_trace(registry, execset, trace));

    // GPU-managed models: any bundle whose assets dir carries a PLOWDEV
    // device blob goes under the S1 model manager — it plans each model's
    // VRAM footprint from the blob header, loads the registration-order
    // subset that fits (co-residency), and switches the rest on demand
    // (evict-LRU + load) from the request path. Checkpoint dir is
    // `<assets>/checkpoint` (PLOW_CHECKPOINT overrides); the initial loads
    // are the slow part of startup (a 12B checkpoint is ~22 GiB of H2D),
    // done before the listeners open. `PLOW_VRAM_BUDGET_MIB` caps the
    // planner's view of the card (A/B, tests).
    #[cfg(feature = "cuda")]
    let mut managed_slugs: std::collections::HashSet<String> = std::collections::HashSet::new();
    #[cfg(feature = "cuda")]
    if let Some(cuda) = &cuda {
        let mut models: Vec<(String, PathBuf, PathBuf)> = Vec::new();
        let slugs: Vec<String> = state.registry.slugs().map(str::to_string).collect();
        for slug in slugs {
            let bundle = state.registry.get(&slug)?;
            if plowrt::asset::devblob::DevBlob::find_in_dir(&bundle.dir)?.is_none() {
                continue;
            }
            let ckpt = std::env::var("PLOW_CHECKPOINT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| bundle.dir.join("checkpoint"));
            managed_slugs.insert(slug.clone());
            models.push((slug, bundle.dir.clone(), ckpt));
        }
        // Keep CLI registration order (registry iteration is hash-order).
        models.sort_by_key(|(_, dir, _)| assets.iter().position(|a| a == dir));
        if !models.is_empty() {
            let budget = std::env::var("PLOW_VRAM_BUDGET_MIB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map(|mib| mib << 20);
            let mgr = Arc::new(plowrt::serve::manager::ModelManager::new(
                Arc::clone(cuda),
                &state,
                mux_cfg,
                models,
                budget,
            )?);
            state.install_manager(Arc::clone(&mgr));
            mgr.load_initial().await?;
        }
    }
    #[cfg(not(feature = "cuda"))]
    let managed_slugs: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Spawn a per-model dispatcher: bucket-mux + arrival-rate batch formation.
    // Each dispatcher owns a Sender clone via AppState::mux(slug). Managed
    // (GPU) models are skipped — their dispatcher lifecycle belongs to the
    // manager (spawned on load, drained+removed on evict).
    let slugs: Vec<String> = state.registry.slugs().map(str::to_string).collect();
    for slug in slugs {
        if managed_slugs.contains(&slug) {
            continue;
        }
        let bundle = state.registry.get(&slug)?;
        let m = mux::spawn(slug.clone(), bundle, Arc::clone(&state), mux_cfg);
        state.install_mux(slug, m);
    }

    let router = app(state);

    // TCP listener: unchanged, always on.
    let tcp_addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let tcp_listener = tokio::net::TcpListener::bind(tcp_addr).await?;
    tracing::info!(%tcp_addr, "plowrt serving OpenAI API over TCP");
    let tcp_router = router.clone();
    let tcp_task = tokio::spawn(async move {
        if let Err(e) = axum::serve(tcp_listener, tcp_router).await {
            tracing::error!(error = %e, "TCP listener error");
        }
    });

    // Optional UDS listener: bridged through hyper directly (axum 0.7's
    // `serve` accepts only TcpListener). Same router as the TCP path.
    let uds_task = if let Some(path) = socket {
        // Clear a stale socket (previous crashed instance left it behind).
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
        let uds_listener = tokio::net::UnixListener::bind(&path)?;
        // Only the owner should be able to talk to the socket by default.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perm = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(&path, perm);
        }
        tracing::info!(socket = %path.display(), "plowrt serving OpenAI API over UDS");
        let uds_router = router.clone();
        Some(tokio::spawn(async move {
            let svc = hyper_util::service::TowerToHyperService::new(uds_router);
            loop {
                let (stream, _addr) = match uds_listener.accept().await {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(error = %e, "UDS accept failed");
                        continue;
                    }
                };
                let svc = svc.clone();
                tokio::spawn(async move {
                    let io = hyper_util::rt::TokioIo::new(stream);
                    if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                        hyper_util::rt::TokioExecutor::new(),
                    )
                    .serve_connection(io, svc)
                    .await
                    {
                        tracing::debug!(error = %e, "UDS connection ended");
                    }
                });
            }
        }))
    } else {
        None
    };

    // Wait until any listener task exits. In practice they run until the
    // process is signaled; the join here just keeps `main` alive.
    match uds_task {
        Some(uds) => {
            tokio::select! {
                r = tcp_task => { if let Err(e) = r { tracing::error!(error = %e, "TCP task join"); } }
                r = uds => { if let Err(e) = r { tracing::error!(error = %e, "UDS task join"); } }
            }
        }
        None => {
            if let Err(e) = tcp_task.await {
                tracing::error!(error = %e, "TCP task join");
            }
        }
    }
    Ok(())
}

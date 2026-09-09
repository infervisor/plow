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

use plowrt::config::RuntimeConfig;
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

    /// Runtime configuration knobs (memory, scheduling, backend-specific).
    /// Each field also reads its `PLOW_*` env var as a fallback.
    #[command(flatten)]
    rt_cfg: RuntimeConfig,
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
        /// Requests allowed to wait outside engine slots. `0` = four batches.
        #[arg(long, default_value_t = 0)]
        max_queued_requests: usize,
    },

    /// Benchmark one model through the production serving scheduler without HTTP.
    Bench {
        /// Compiled-model directory.
        #[arg(long)]
        assets: PathBuf,
        /// Comma-separated token ids. Omit for deterministic random ids.
        #[arg(long, conflicts_with_all = ["random_input_len", "prompt_rows"])]
        prompt_ids: Option<String>,
        /// File containing one comma-separated token-id row per request.
        /// Row count must equal `--warmup-requests + --requests`.
        #[arg(
            long,
            conflicts_with_all = [
                "prompt_ids",
                "random_input_len",
                "prefill_sweep",
                "prefill_lengths",
                "parity_report"
            ]
        )]
        prompt_rows: Option<PathBuf>,
        /// Number of deterministic random prompt tokens.
        #[arg(long, conflicts_with = "prompt_rows")]
        random_input_len: Option<usize>,
        /// Run a one-load, cold-prefill TTFT sweep through the production mux.
        ///
        /// Use `--prompt-ids` for one exact row, or `--prefill-lengths` for
        /// deterministic random rows.
        #[arg(
            long,
            default_value_t = false,
            conflicts_with_all = ["concurrency", "requests", "warmup_requests", "output_len"]
        )]
        prefill_sweep: bool,
        /// Comma-separated deterministic random prompt lengths for
        /// `--prefill-sweep`.
        #[arg(
            long,
            requires = "prefill_sweep",
            conflicts_with_all = ["prompt_ids", "random_input_len"]
        )]
        prefill_lengths: Option<String>,
        /// Timed requests per prefill length.
        #[arg(long, requires = "prefill_sweep")]
        prefill_reps: Option<usize>,
        /// Warm-up requests per prefill length.
        #[arg(long, requires = "prefill_sweep")]
        prefill_warmups: Option<usize>,
        /// Seed for deterministic random token-id prompts.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Maximum simultaneous requests maintained against the mux.
        #[arg(long, default_value_t = 1)]
        concurrency: usize,
        /// Measured requests. New work is submitted as each request completes.
        #[arg(long, default_value_t = 10)]
        requests: usize,
        /// Requests run and validated before timing.
        #[arg(long, default_value_t = 1)]
        warmup_requests: usize,
        /// Exact generated-token count per request.
        #[arg(long, default_value_t = 128)]
        output_len: usize,
        /// Number of CPU executor threads when no matching GPU backend is available.
        #[arg(long, default_value_t = 8)]
        executors: u32,
        /// Mux arrival-rate batch-formation hold ceiling in milliseconds.
        #[arg(long, default_value_t = 8.0)]
        max_hold_ms: f64,
        /// Mux admission SLO in milliseconds.
        #[arg(long, default_value_t = 250.0)]
        slo_ms: f64,
        /// Requests allowed outside engine slots. Zero derives four engine batches.
        #[arg(long, default_value_t = 0)]
        max_queued_requests: usize,
        /// Record bounded production-engine bucket/chunk selections and TP audit policy.
        #[arg(long, default_value_t = false)]
        engine_diagnostics: bool,
        /// Include exact measured prompt/output token IDs for bench/serve parity checks.
        #[arg(long, default_value_t = false, conflicts_with = "prefill_sweep")]
        parity_report: bool,
        /// Include bounded per-request token IDs for production-path correctness audits.
        #[arg(
            long,
            default_value_t = false,
            conflicts_with_all = ["prefill_sweep", "parity_report"]
        )]
        token_audit: bool,
    },

    /// Enumerate every visible device and, with `--tp`, bring up the
    /// tensor-parallel group: peer-mapped reduction regions, the per-rank
    /// cross-GPU counter tables, and the all-pairs peer-visibility check.
    ///
    /// This is the multi-GPU bring-up path on its own, without a model — the
    /// AMD interpreter engine does not exist yet, so an end-to-end sharded
    /// serve cannot be run, but everything the host owes the device (§6a's
    /// `xctr`/`rank`/`n_gpu`/`peer_scratch`, §6d's zero-before-launch) is
    /// exercised and reported here.
    Devices {
        /// TP degree. Omit to only enumerate; `N` brings up an N-rank group
        /// over the first N visible devices.
        #[arg(long)]
        tp: Option<u32>,
        /// Model hidden size, which sizes the all-reduce message (`t·H·2` B).
        #[arg(long, default_value_t = 3840)]
        hidden: u32,
        /// Tokens per dispatch: 1 for decode, the prefill CHUNK for prefill.
        /// The peer region scales linearly with this.
        #[arg(long, default_value_t = 1)]
        max_tokens: u32,
        /// Decoder layers. Sizes the cross-GPU counter region via
        /// `PeerLayout::counters_for`.
        #[arg(long, default_value_t = 48)]
        layers: u32,
        /// Size the counters for the PREFILL program (two-shot all-reduce, two
        /// xctr gates per collective) rather than decode (one-shot, one gate).
        #[arg(long, default_value_t = false)]
        prefill: bool,
    },

    /// Bring the AMD/gfx950 engine up on a compiled blob and time decode steps.
    ///
    /// Runs the schedule directly as a diagnostic. Production performance must
    /// be measured through `plowrt bench` or `plowrt serve`.
    AmdBench {
        /// Compiled device blob (`model.pkt`).
        #[arg(long)]
        blob: PathBuf,
        /// Directory of gfx950 code objects (`interp_*.elf`).
        #[arg(long, id = "amd_bench_hsaco")]
        hsaco: PathBuf,
        /// Safetensors checkpoint. Required; zero-weight execution moved to
        /// the distinct `amd-probe` command.
        ///
        /// The clap ID is EXPLICIT because the GLOBAL `--rt-checkpoint`
        /// (config.rs) already claims the derived id `checkpoint`, and clap
        /// resolves ids, not long names: two args with the same id and
        /// different value types PANIC at access time
        /// ("Mismatch between definition and access of `checkpoint`"), which is
        /// what `amd-bench --checkpoint <dir>` did on EVERY invocation.
        #[arg(long, id = "amd_bench_checkpoint")]
        checkpoint: Option<PathBuf>,
        /// Removed compatibility flag. Use the distinct `amd-probe` command.
        #[arg(long, default_value_t = false, hide = true)]
        synthetic_probe: bool,
        /// Prompt token ids to decode from, comma-separated. Needs
        /// `--checkpoint` to mean anything.
        ///
        /// Under `--batched`, `;` separates one prompt PER SEQUENCE SLOT and
        /// each is prefilled into its own slot. Fewer prompts than slots cycles
        /// them. THIS IS THE CORRECTNESS GATE: with `B` copies of one prompt
        /// every slot must produce the same stream, and with `B` different
        /// prompts each slot must produce what it produces alone. Prompts of
        /// DIFFERENT LENGTHS also make the positions ragged, which is the case
        /// a lockstep batch cannot reach.
        #[arg(long)]
        prompt: Option<String>,
        /// Decode steps to time.
        #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..))]
        steps: u32,
        /// Context position for synthetic no-prompt timing. With `--prompt`, the
        /// actual context is the prompt length and this option is rejected.
        #[arg(long, default_value_t = 1024, conflicts_with = "prompt")]
        ctx: u32,
        /// Drive all `batch` sequences per dispatch (needs a blob compiled with
        /// PLOW_DECODE_BATCH > 1). Reports tpot AND aggregate throughput, which
        /// are the two axes a concurrency sweep compares.
        #[arg(long, default_value_t = false)]
        batched: bool,
        /// Tensor-parallel degree. Needs a blob compiled `--num-gpus N`, and
        /// runs one rank per device over the first N visible GPUs.
        ///
        /// Every rank must emit the SAME token stream: they hold the full
        /// replicated residual and a full-vocab lm_head, so identical ids is
        /// what proves the two all-reduces per layer actually ran. A rank that
        /// skipped its collective still produces fluent-looking ids from its own
        /// shard, so this is checked every step, not sampled.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        tp: u32,
        /// Write rank 0's raw `act.logits` row (bf16, `vocab` wide) after the
        /// prefill and after every decode step, as `<dir>/logits_{prefill,NNN}.bin`.
        ///
        /// The device samples into `in.ids` itself, so a run otherwise reports
        /// only the ARGMAX — and an argmax cannot tell a near-tie apart from a
        /// fault. Two runs that differ in one arm (prefill program vs the
        /// decode-only walk) are compared as VECTORS through these files;
        /// `scripts/glm52_logit_cmp.py` is the reader.
        #[arg(long)]
        dump_logits: Option<PathBuf>,
        /// TIER-2 PREFILL SWEEP: comma-separated prompt lengths, each timed
        /// `--prefill-reps` times on ONE loaded engine, reported as a median.
        ///
        /// It exists because the `--prompt` path prefills exactly ONCE and
        /// prints one decimal — 0.1 ms of quantisation on a 12 ms single-layer
        /// block is 0.8%, which is the size of the effects a tier-2 harness is
        /// built to rank. Amortising the load over every context in the sweep
        /// is the other half: on a truncated GLM blob the load is ~8-25 s and a
        /// prefill is ~12-110 ms, so one invocation per LENGTH would be a
        /// harness that spends 99% of its time loading.
        ///
        /// `prefill` is position-stateless (every chunk's rows come from its
        /// `ChunkStep`), so repeating it simply rewrites the same KV rows.
        #[arg(long)]
        prefill_sweep: Option<String>,
        /// Timed repetitions per `--prefill-sweep` length (one warm-up pass is
        /// always run first and discarded). Zero would leave the sample vector
        /// empty and the median index out of bounds, so the range starts at 1.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
        prefill_reps: u32,
    },

    /// Execute an unbound AMD packet with zero-filled weights.
    ///
    /// This is a synthetic schedule/kernel diagnostic, never a model-quality
    /// or performance result. Checkpoint-bound work belongs to `bench`,
    /// `serve`, or the remaining `amd-bench` diagnostics.
    AmdProbe {
        /// Compiled device blob (`model.pkt`).
        #[arg(long)]
        blob: PathBuf,
        /// Directory of AMD code objects (`interp_*.elf`).
        #[arg(long, id = "amd_probe_hsaco")]
        hsaco: PathBuf,
        /// Synthetic decode steps. Zero is valid for a prefill-only sweep.
        #[arg(long, default_value_t = 32)]
        steps: u32,
        /// Synthetic context position.
        #[arg(long, default_value_t = 1024)]
        ctx: u32,
        /// Drive all compiled sequence slots per decode dispatch.
        #[arg(long, default_value_t = false)]
        batched: bool,
        /// Tensor-parallel degree.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        tp: u32,
        /// Comma-separated synthetic prefill lengths, timed after one warm-up.
        #[arg(long)]
        prefill_sweep: Option<String>,
        /// Timed repetitions per prefill length.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
        prefill_reps: u32,
    },

    /// Run a BLOCK asset (act.x in, act.x out) through the AMD engine.
    ///
    /// The A/B vehicle for numerics: two blocks that differ only in precision,
    /// same weights, same input, and the outputs compared. It exists separately
    /// from `amd-bench` because a block is not a model — no embed, no lm_head,
    /// no argmax — so none of the token-level entry points apply.
    AmdBlock {
        /// Compiled block blob (`model.pkt`).
        #[arg(long)]
        blob: PathBuf,
        #[arg(long, id = "amd_block_hsaco")]
        hsaco: PathBuf,
        #[arg(long, id = "amd_block_checkpoint")]
        checkpoint: Option<PathBuf>,
        /// Prompt token ids (comma-separated) for a model-shaped block — one
        /// that carries `in.ids`/`embed_tokens` and can be driven end to end.
        #[arg(long)]
        prompt: Option<String>,
        /// Tensors to report zero/non-zero statistics for after the run.
        /// Comma-separated; this is the "zero vs merely wrong" instrument.
        #[arg(long, default_value = "act.x,act.hn,act.logits")]
        inspect: String,
        /// List the blob's tensors and exit.
        #[arg(long, default_value_t = false)]
        list_tensors: bool,
        /// Write the output `act.x` bytes here, for a bit-exact diff between
        /// two precisions.
        #[arg(long)]
        dump: Option<PathBuf>,
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

    /// Disassemble a compiled device blob: named operands, kernargs, counters.
    ///
    /// Static and offline — the blob is a file, and reading it needs no GPU, no
    /// driver and no features. Until now the only true device-instruction dump
    /// was the `.insts.txt` sidecar the C harness writes under
    /// `PLOW_TRACE_RAW`, which needs hardware to produce.
    ///
    /// Operand names come from `packet::slots`; the raw slots are printed
    /// alongside them, always, because a name is an interpretation and the bytes
    /// are not.
    Disasm {
        /// `model.pkt`, or an asset directory containing one.
        blob: PathBuf,
        /// Restrict to one program: a prefill bucket `T`, or `1` for decode.
        #[arg(long)]
        program: Option<u32>,
        /// Instruction window, `lo..hi`.
        #[arg(long)]
        range: Option<String>,
        /// `text` (default), `json`, or `jsonl` (one program per line).
        #[arg(long, default_value = "text")]
        format: String,
        /// Kernarg block and the dispatch configuration it implies.
        #[arg(long, default_value_t = false)]
        kernargs: bool,
        /// Tensor table, classified WEIGHT / TABLE / RUNTIME.
        #[arg(long, default_value_t = false)]
        tensors: bool,
        /// Counter analysis: `graphstat`'s aggregates plus per-counter detail.
        #[arg(long, default_value_t = false)]
        counters: bool,
        /// Per-CU stream entries. LARGE — a GLM-5.2 prefill program has 377k of
        /// them, ~45 MB of JSON. Off by default for that reason.
        #[arg(long, default_value_t = false)]
        stream: bool,
        /// Structure only: skip every derived metric.
        #[arg(long, default_value_t = false)]
        no_analysis: bool,
    },

    /// Classify every opcode a compiled blob uses by how it learns a row's
    /// request identity, and report whether the program can be executed as one
    /// packed token batch.
    ///
    /// Static and offline, like `disasm`: the blob is a file. The four classes
    /// are §3 of `plans/unified-token-batch.md`, reproduced in
    /// `docs/arch/17-unified-token-batch.md (Part III)`.
    ///
    /// An opcode with no classification is class C and is REFUSED, because on
    /// AMD the interpreter's dispatch `default:` writes nothing and does not
    /// trap — an unclassified operator is a silent-wrong-answer hazard, not an
    /// inconvenience.
    OpAudit {
        /// `model.pkt`, or an asset directory containing one. Omit with
        /// `--table` to print the whole-ISA table instead.
        blob: Option<PathBuf>,
        /// Print every opcode in the ISA with its static class, with no blob.
        #[arg(long, default_value_t = false)]
        table: bool,
        /// Restrict to one program: a prefill bucket `T`, or `1` for decode.
        #[arg(long)]
        program: Option<u32>,
        /// `text` (default) or `json`.
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    let filter_str = format!("{filter}");
    // `op-audit --format json` writes a document to stdout; the startup banner
    // would land inside it. Same reason `bench` logs to stderr.
    if matches!(&cli.cmd, Cmd::Bench { .. } | Cmd::OpAudit { .. }) {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        cuda = cfg!(feature = "cuda"),
        hsa = cfg!(feature = "hsa"),
        hf_tokenizer = cfg!(feature = "hf-tokenizer"),
        log_filter = %filter_str,
        "plowrt starting"
    );

    RuntimeConfig::init(cli.rt_cfg);
    match cli.cmd {
        Cmd::Serve {
            assets,
            port,
            socket,
            executors,
            trace,
            max_hold_ms,
            slo_ms,
            max_queued_requests,
        } => {
            tracing::info!(
                assets = ?assets,
                port,
                socket = ?socket,
                executors,
                trace,
                max_hold_ms,
                slo_ms,
                max_queued_requests,
                runtime = ?RuntimeConfig::global(),
                environment = ?runtime_environment(),
                "resolved serve configuration"
            );
            // The REPLAY line, separate from the Debug dump above on purpose: that dump is every
            // knob at its resolved value plus every ambient PLOW_* var (PLOW_HIPCC, PLOW_NVCC,
            // toolchain paths), which records the machine rather than the decision. This one is
            // only what this serve chose away from the tree's defaults, in the spelling that sets
            // it again — greppable out of a log a campaign already keeps.
            tracing::info!(
                replay = ?plowrt::config::serve_replay(),
                "serve replay — the runtime half of build.json's emit_config.replay"
            );
            serve(
                assets,
                port,
                socket,
                executors,
                trace,
                MuxConfig {
                    max_hold_ms,
                    slo_ms,
                    max_queued_requests,
                    ..MuxConfig::default()
                },
            )
            .await
        }
        Cmd::Bench {
            assets,
            prompt_ids,
            prompt_rows,
            random_input_len,
            prefill_sweep,
            prefill_lengths,
            prefill_reps,
            prefill_warmups,
            seed,
            concurrency,
            requests,
            warmup_requests,
            output_len,
            executors,
            max_hold_ms,
            slo_ms,
            max_queued_requests,
            engine_diagnostics,
            parity_report,
            token_audit,
        } => {
            bench(
                assets,
                prompt_ids,
                prompt_rows,
                random_input_len.unwrap_or(32),
                prefill_sweep,
                prefill_lengths,
                prefill_reps.unwrap_or(3),
                prefill_warmups.unwrap_or(1),
                seed,
                concurrency,
                requests,
                warmup_requests,
                output_len,
                executors,
                MuxConfig {
                    max_hold_ms,
                    slo_ms,
                    max_queued_requests,
                    ..MuxConfig::default()
                },
                engine_diagnostics,
                parity_report,
                token_audit,
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
        Cmd::Disasm {
            blob,
            program,
            range,
            format,
            kernargs,
            tensors,
            counters,
            stream,
            no_analysis,
        } => disasm_cmd(
            blob,
            program,
            range,
            format,
            plowrt::disasm::Sections {
                kernargs,
                tensors,
                counters,
                stream,
                no_analysis,
            },
        ),
        Cmd::OpAudit {
            blob,
            table,
            program,
            format,
        } => op_audit_cmd(blob, table, program, format),
        Cmd::Devices {
            tp,
            hidden,
            max_tokens,
            layers,
            prefill,
        } => devices(tp, hidden, max_tokens, layers, prefill),
        #[cfg(feature = "hsa")]
        Cmd::AmdBench {
            blob,
            hsaco,
            checkpoint,
            synthetic_probe,
            prompt,
            steps,
            ctx,
            batched,
            tp,
            dump_logits,
            prefill_sweep,
            prefill_reps,
        } => {
            let synthetic_probe = require_synthetic_probe(checkpoint.is_some(), synthetic_probe)?;
            tracing::warn!(
                "amd-bench is diagnostic-only and deprecated as a performance authority; use `plowrt bench` for production-path measurements"
            );
            if tp > 1 {
                amd_bench_tp(
                    blob,
                    hsaco,
                    checkpoint,
                    synthetic_probe,
                    prompt,
                    steps,
                    ctx,
                    tp,
                    batched,
                    dump_logits,
                    prefill_sweep,
                    prefill_reps,
                )
            } else if prefill_sweep.is_some() {
                Err("--prefill-sweep is implemented on the TP path only (--tp N, N>1)".into())
            } else {
                amd_bench(
                    blob,
                    hsaco,
                    checkpoint,
                    synthetic_probe,
                    prompt,
                    steps,
                    ctx,
                    batched,
                    dump_logits,
                )
            }
        }
        #[cfg(feature = "hsa")]
        Cmd::AmdProbe {
            blob,
            hsaco,
            steps,
            ctx,
            batched,
            tp,
            prefill_sweep,
            prefill_reps,
        } => {
            validate_amd_probe_steps(steps, prefill_sweep.is_some())?;
            tracing::warn!(
                "amd-probe executes an unbound zero-weight packet; results are synthetic diagnostics and must not be reported as model performance"
            );
            if tp > 1 {
                amd_bench_tp(
                    blob,
                    hsaco,
                    None,
                    true,
                    None,
                    steps,
                    ctx,
                    tp,
                    batched,
                    None,
                    prefill_sweep,
                    prefill_reps,
                )
            } else if prefill_sweep.is_some() {
                Err("--prefill-sweep is implemented on the TP path only (--tp N, N>1)".into())
            } else {
                amd_bench(blob, hsaco, None, true, None, steps, ctx, batched, None)
            }
        }
        #[cfg(not(feature = "hsa"))]
        Cmd::AmdBench { .. } => Err("plowrt was built without --features hsa".into()),
        #[cfg(not(feature = "hsa"))]
        Cmd::AmdProbe { .. } => Err("plowrt was built without --features hsa".into()),
        #[cfg(feature = "hsa")]
        Cmd::AmdBlock {
            blob,
            hsaco,
            checkpoint,
            prompt,
            inspect,
            list_tensors,
            dump,
        } => amd_block(blob, hsaco, checkpoint, prompt, inspect, list_tensors, dump),
        #[cfg(not(feature = "hsa"))]
        Cmd::AmdBlock { .. } => Err("plowrt was built without --features hsa".into()),
    }
}

#[cfg_attr(not(feature = "hsa"), allow(dead_code))]
fn require_synthetic_probe(
    has_checkpoint: bool,
    synthetic_probe: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    if synthetic_probe {
        return Err("--synthetic-probe moved to the distinct `plowrt amd-probe` command".into());
    }
    if !has_checkpoint {
        return Err(
            "amd-bench requires --checkpoint; use `plowrt amd-probe` for unbound zero-weight \
             packet execution or `plowrt bench` for performance"
                .into(),
        );
    }
    Ok(false)
}

#[cfg_attr(not(feature = "hsa"), allow(dead_code))]
fn validate_amd_probe_steps(
    steps: u32,
    has_prefill_sweep: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if steps == 0 && !has_prefill_sweep {
        return Err("amd-probe --steps 0 requires --prefill-sweep".into());
    }
    Ok(())
}

#[cfg_attr(not(feature = "hsa"), allow(dead_code))]
fn synthetic_timing_prefix(synthetic_probe: bool) -> &'static str {
    if synthetic_probe {
        "SYNTHETIC DIAGNOSTIC: "
    } else {
        ""
    }
}

#[cfg(test)]
mod amd_bench_cli_tests {
    use super::{
        parse_prefill_lengths, require_synthetic_probe, synthetic_timing_prefix,
        validate_amd_probe_steps, validate_parity_report_options, validate_token_audit_options,
        Cli,
    };
    use clap::Parser;
    use plowrt::serve::bench::Input;

    #[test]
    fn unbound_run_requires_explicit_synthetic_probe() {
        assert!(require_synthetic_probe(false, false).is_err());
        assert!(require_synthetic_probe(false, true).is_err());
    }

    #[test]
    fn synthetic_execution_has_a_distinct_probe_command() {
        assert!(Cli::try_parse_from([
            "plowrt",
            "amd-probe",
            "--blob",
            "model.pkt",
            "--hsaco",
            "hsaco",
            "--tp",
            "8",
            "--steps",
            "0",
            "--prefill-sweep",
            "512,1024",
        ])
        .is_ok());
        assert!(Cli::try_parse_from([
            "plowrt",
            "amd-probe",
            "--blob",
            "model.pkt",
            "--hsaco",
            "hsaco",
            "--checkpoint",
            "weights",
        ])
        .is_err());
    }

    #[test]
    fn probe_zero_steps_requires_prefill_sweep() {
        assert!(validate_amd_probe_steps(0, false).is_err());
        assert!(validate_amd_probe_steps(0, true).is_ok());
        assert!(validate_amd_probe_steps(1, false).is_ok());
    }

    #[test]
    fn amd_direct_runners_reject_zero_tp() {
        for command in ["amd-bench", "amd-probe"] {
            assert!(Cli::try_parse_from([
                "plowrt",
                command,
                "--blob",
                "model.pkt",
                "--hsaco",
                "hsaco",
                "--tp",
                "0",
            ])
            .is_err());
        }
    }

    #[test]
    fn checkpoint_bound_run_is_not_labeled_synthetic() {
        assert!(!require_synthetic_probe(true, false).unwrap());
        assert!(require_synthetic_probe(true, true).is_err());
        assert_eq!(synthetic_timing_prefix(false), "");
    }

    #[test]
    fn synthetic_timing_label_rejects_performance_interpretation() {
        let label = synthetic_timing_prefix(true);
        assert!(label.contains("SYNTHETIC DIAGNOSTIC"));
        assert!(!label.to_ascii_lowercase().contains("performance"));
    }

    #[test]
    fn prefill_lengths_are_positive_and_ordered() {
        assert_eq!(
            parse_prefill_lengths("512, 1024,2048").unwrap(),
            [512, 1024, 2048]
        );
        assert!(parse_prefill_lengths("").is_err());
        assert!(parse_prefill_lengths("512,0").is_err());
    }

    #[test]
    fn prefill_sweep_accepts_random_lengths_or_exact_ids() {
        assert!(Cli::try_parse_from([
            "plowrt",
            "bench",
            "--assets",
            "model",
            "--prefill-sweep",
            "--prefill-lengths",
            "512,1024",
        ])
        .is_ok());
        assert!(Cli::try_parse_from([
            "plowrt",
            "bench",
            "--assets",
            "model",
            "--prefill-sweep",
            "--prompt-ids",
            "1,2,3",
        ])
        .is_ok());
    }

    #[test]
    fn prefill_sweep_rejects_normal_throughput_controls() {
        assert!(Cli::try_parse_from([
            "plowrt",
            "bench",
            "--assets",
            "model",
            "--prefill-sweep",
            "--concurrency",
            "2",
        ])
        .is_err());
    }

    #[test]
    fn production_engine_diagnostics_are_explicit() {
        assert!(Cli::try_parse_from([
            "plowrt",
            "bench",
            "--assets",
            "model",
            "--engine-diagnostics",
        ])
        .is_ok());
    }

    #[test]
    fn kda_family_route_defaults_on_and_allows_explicit_rollback() {
        let default = Cli::try_parse_from(["plowrt", "bench", "--assets", "model"]).unwrap();
        assert!(default.rt_cfg.amd.kda_family_route);

        let rollback = Cli::try_parse_from([
            "plowrt",
            "--amd-kda-family-route=false",
            "bench",
            "--assets",
            "model",
        ])
        .unwrap();
        assert!(!rollback.rt_cfg.amd.kda_family_route);
    }

    #[test]
    fn production_snapshot_tensor_selection_is_configurable() {
        let cli = Cli::try_parse_from([
            "plowrt",
            "--amd-tens-snap",
            "snapshots",
            "--amd-snap-tensors",
            "act.logits,act.x",
            "--amd-snap-slot",
            "0",
            "bench",
            "--assets",
            "model",
        ])
        .unwrap();
        assert_eq!(cli.rt_cfg.amd.tens_snap.as_deref(), Some("snapshots"));
        assert_eq!(
            cli.rt_cfg.amd.snap_tensors.as_deref(),
            Some("act.logits,act.x")
        );
        assert_eq!(cli.rt_cfg.amd.snap_slot, 0);
    }

    #[test]
    fn parity_report_requires_one_exact_measured_request() {
        assert!(validate_parity_report_options(true, true, 1, 1, 0).is_ok());
        assert!(Cli::try_parse_from([
            "plowrt",
            "--rt-checkpoint",
            "weights",
            "bench",
            "--assets",
            "model",
            "--prompt-ids",
            "1,2,3",
            "--concurrency",
            "1",
            "--requests",
            "1",
            "--warmup-requests",
            "0",
            "--parity-report",
        ])
        .is_ok());
        assert!(Cli::try_parse_from([
            "plowrt",
            "bench",
            "--assets",
            "model",
            "--random-input-len",
            "8192",
            "--concurrency",
            "1",
            "--requests",
            "1",
            "--warmup-requests",
            "0",
            "--parity-report",
        ])
        .is_ok());
        for invalid in [
            (false, 1, 1, 0),
            (true, 2, 1, 0),
            (true, 1, 2, 0),
            (true, 1, 1, 1),
        ] {
            assert!(validate_parity_report_options(
                true, invalid.0, invalid.1, invalid.2, invalid.3
            )
            .is_err());
        }
        assert!(Cli::try_parse_from([
            "plowrt",
            "bench",
            "--assets",
            "model",
            "--parity-report",
            "--prefill-sweep",
        ])
        .is_err());
    }

    #[test]
    fn token_audit_requires_exact_bounded_rows() {
        assert!(
            validate_token_audit_options(true, &Input::TokenIds(vec![1, 2, 3]), 0, 4, 8).is_ok()
        );
        assert!(
            validate_token_audit_options(true, &Input::Random { len: 3, seed: 0 }, 0, 4, 8)
                .is_err()
        );
        assert!(validate_token_audit_options(true, &Input::TokenIds(vec![1]), 0, 65, 8).is_err());
        assert!(
            validate_token_audit_options(true, &Input::TokenIds(vec![1]), 0, 64, 1024).is_err()
        );
        assert!(validate_token_audit_options(
            true,
            &Input::TokenRows(vec![vec![9], vec![1, 2, 3]]),
            1,
            1,
            8
        )
        .is_ok());
        assert!(Cli::try_parse_from([
            "plowrt",
            "bench",
            "--assets",
            "model",
            "--token-audit",
            "--parity-report",
        ])
        .is_err());
    }

    #[test]
    fn prompt_rows_cli_is_exact_and_conflicts_with_other_input_modes() {
        assert!(Cli::try_parse_from([
            "plowrt",
            "bench",
            "--assets",
            "model",
            "--prompt-rows",
            "rows.csv",
            "--token-audit",
        ])
        .is_ok());
        for conflicting in ["--prompt-ids", "--random-input-len"] {
            let value = if conflicting == "--prompt-ids" {
                "1,2"
            } else {
                "2"
            };
            assert!(Cli::try_parse_from([
                "plowrt",
                "bench",
                "--assets",
                "model",
                "--prompt-rows",
                "rows.csv",
                conflicting,
                value,
            ])
            .is_err());
        }
        for conflicting in ["--prefill-sweep", "--parity-report"] {
            assert!(Cli::try_parse_from([
                "plowrt",
                "bench",
                "--assets",
                "model",
                "--prompt-rows",
                "rows.csv",
                conflicting,
            ])
            .is_err());
        }
    }
}

/// Names the visible-device mask in force, for the tail of a "not enough
/// devices" error. Empty when none is set — in which case the device count is
/// the machine's and there is nothing to point at.
fn visible_mask_hint() -> String {
    let set = plowrt::device::visibility::describe_env();
    if set.is_empty() {
        return String::new();
    }
    let list = set
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ");
    format!(" (a visible-device mask is in force: {list})")
}

fn runtime_environment() -> Vec<(String, String)> {
    let mut vars: Vec<_> = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("PLOW_")
                || matches!(
                    key.as_str(),
                    "RUST_LOG"
                        | "ROCR_VISIBLE_DEVICES"
                        | "HIP_VISIBLE_DEVICES"
                        | "CUDA_VISIBLE_DEVICES"
                        | "HSA_XNACK"
                        | "HSA_OVERRIDE_GFX_VERSION"
                        | "OMP_NUM_THREADS"
                )
        })
        .map(|(key, value)| {
            let upper = key.to_ascii_uppercase();
            let secret = ["TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY"]
                .iter()
                .any(|needle| upper.contains(needle));
            (key, if secret { "<redacted>".into() } else { value })
        })
        .collect();
    vars.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    vars
}

/// Bring the AMD engine up and time decode steps.
#[cfg(feature = "hsa")]
fn amd_bench(
    blob: PathBuf,
    hsaco: PathBuf,
    checkpoint: Option<PathBuf>,
    synthetic_probe: bool,
    prompt: Option<String>,
    steps: u32,
    ctx: u32,
    batched: bool,
    dump_logits: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::amd::AmdEngine;

    if synthetic_probe {
        eprintln!("SYNTHETIC DIAGNOSTIC: weights are unbound; output is not a performance result");
    }
    let timing = synthetic_timing_prefix(synthetic_probe);
    let be = Arc::new(plowrt::device::hsa::HsaBackend::new(0)?);
    let t0 = std::time::Instant::now();
    let mut eng = AmdEngine::load(Arc::clone(&be), &blob, &hsaco, checkpoint.as_deref())?;
    println!(
        "{timing}loaded in {:.1} s: arch={} programs={} max_ctx={} schedulers={:?}",
        t0.elapsed().as_secs_f64(),
        eng.arch(),
        eng.n_programs(),
        eng.max_ctx(),
        eng.schedulers(),
    );
    for p in 0..eng.n_programs() {
        println!(
            "  program {p}: T={} segments={}",
            eng.prog_t(p),
            eng.prog_segments(p)
        );
    }

    if batched {
        let b = eng.batch();
        println!("\nbatched decode: {b} sequences per dispatch, ctx={ctx}");
        // A prompt makes this the REAL gate. `prefill_slot` rebases the KV
        // pointers onto slot s and runs the single-sequence prefill program
        // there, so every slot's cache is genuinely populated and the decode
        // that follows reads rows this run wrote — as opposed to `--ctx N`
        // below, which decodes from position N over KV nobody ever prefilled.
        // That is why the old `identically-seeded sequences agree` line proved
        // nothing: it was reading VRAM history, in both directions
        // (perf-data/batched-decode-amd-status.md).
        let mut pos: Vec<u32> = vec![ctx; b];
        let mut chains: Vec<Vec<u32>> = vec![Vec::new(); b];
        let prompts: Vec<Vec<u32>> = match &prompt {
            None => Vec::new(),
            Some(p) => p
                .split(';')
                .map(|one| {
                    one.split(',')
                        .map(|s| s.trim().parse::<u32>())
                        .collect::<std::result::Result<Vec<u32>, _>>()
                })
                .collect::<std::result::Result<_, _>>()?,
        };
        if !prompts.is_empty() {
            for s in 0..b {
                let ids = &prompts[s % prompts.len()];
                let tok = eng.prefill_slot(s, ids)?;
                println!("  slot {s}: prefill {} tokens -> sampled {tok}", ids.len());
                pos[s] = ids.len() as u32;
                chains[s].push(tok);
            }
        } else {
            eng.seed_ids(&vec![0u32; b])?;
        }

        for i in 0..4u32 {
            let p: Vec<u32> = if prompts.is_empty() {
                vec![ctx + i; b]
            } else {
                pos.clone()
            };
            let k: Vec<u32> = p.iter().map(|x| x + 1).collect();
            if !prompts.is_empty() {
                // SEED EVERY ROW EXPLICITLY. The device does leave its own
                // per-sequence argmax in `in.ids` (commit c50472f), but PREFILL
                // is single-sequence and writes `in.ids[0]` ONLY — so after the
                // per-slot prefills, rows 1.. still hold whatever the last
                // prefill or the previous run left there. Relying on the device
                // here made every slot but one decode from a stale id and looked
                // exactly like "per-sequence KV rows are wrong". `AmdServe`
                // seeds all B rows for the same reason.
                let feed: Vec<u32> = (0..b).map(|s| *chains[s].last().expect("seeded")).collect();
                eng.seed_ids(&feed)?;
            }
            let out = eng.decode_step_batched(&p, &k)?;
            for s in 0..b {
                chains[s].push(out[s]);
            }
            if prompts.is_empty() {
                println!("  step {i}: {out:?}");
            } else {
                // Positions advance per slot, which is what makes this a RAGGED
                // batch when the prompts differ in length.
                for s in 0..b {
                    pos[s] += 1;
                }
            }
        }
        if prompts.is_empty() {
            println!("  seq0 chain: {:?}", chains[0]);
            if b > 1 {
                // Seeded identically from the same position, so identical
                // forward passes — but over KV NOBODY WROTE. Agreement here is
                // a statement about VRAM history, NOT about batching. Kept only
                // because a disagreement is still a real signal.
                let agree = (0..b).all(|s| chains[s] == chains[0]);
                println!(
                    "  identically-seeded sequences agree: {} (NOT a correctness gate \
                     — no prefill ran; pass --prompt for the real one)",
                    if agree { "YES" } else { "NO" }
                );
            }
        } else {
            for s in 0..b {
                println!(
                    "  slot {s} chain ({} prompt tokens): {:?}",
                    prompts[s % prompts.len()].len(),
                    chains[s]
                );
            }
            // Slots fed the SAME prompt must produce the SAME stream. This is
            // the check the old `--prompt` path could not make, because prefill
            // populated slot 0 only and slots 1.. read uninitialised VRAM.
            //
            // Compare each slot against the FIRST slot carrying ITS prompt, not
            // against slot 0. Comparing only against slot 0 checks one prompt
            // class out of `prompts.len()` and reports a green while the other
            // classes diverge — which is exactly what it did at B=16, where
            // slots 13/14/15 were wrong and every slot holding prompt 0 was
            // right.
            let mut verdict = true;
            for s in 1..b {
                let first =
                    (0..s).find(|&r| prompts[r % prompts.len()] == prompts[s % prompts.len()]);
                if let Some(r) = first {
                    if chains[s] != chains[r] {
                        verdict = false;
                        println!("  slot {s} and slot {r} share a prompt and DIFFER");
                    }
                }
            }
            println!(
                "  same-prompt slots agree: {}",
                if verdict {
                    "YES"
                } else {
                    "NO  <-- per-sequence KV rows are wrong"
                }
            );
            println!("  cross-check each DIFFERENT prompt against a batch-1 run of the same ids");
        }

        let t0 = std::time::Instant::now();
        for i in 0..steps {
            let p: Vec<u32> = if prompts.is_empty() {
                vec![ctx + 4 + i; b]
            } else {
                pos.clone()
            };
            let k: Vec<u32> = p.iter().map(|x| x + 1).collect();
            eng.decode_step_batched(&p, &k)?;
            for s in 0..b {
                pos[s] += 1;
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
        println!(
            "\n{timing}{steps} dispatches x batch {b} at ctx={ctx}:\n  \
             {timing}tpot {ms:.3} ms  |  aggregate {:.1} tok/s  |  per-dispatch {ms:.3} ms",
            b as f64 * 1e3 / ms
        );
        if !eng.weights_bound() {
            println!("  (synthetic diagnostic only — not a performance result)");
        }
        // THE FOURTH EXIT. `trace_dump_1`'s own comment names `--batched` as a path that
        // returned before a dump; it stayed that way, so `PLOW_TRACE_RAW` was silent for the
        // batched decode step — the one shape a concurrency-4 residue attribution needs. The
        // buffer holds the LAST launch, which here is a steady-state timed step.
        trace_dump_1(&eng, "")?;
        return Ok(());
    }

    // A prompt makes this a real greedy decode from position 0: the first step
    // writes KV row 0 and attends over exactly [0,1), so nothing is read that
    // was not written. WITHOUT one, decode starts mid-context over KV rows
    // nobody wrote — which samples the same id every step and looks like a
    // working decoder. That is why the two modes are distinguished loudly.
    if let Some(p) = &prompt {
        let ids: Vec<u32> = p
            .split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<std::result::Result<_, _>>()?;
        if ids.is_empty() {
            return Err("--prompt is empty".into());
        }
        // The FULL vocab row, the same surface `amd_bench_tp` dumps and for the same
        // reason: a greedy id alone cannot tell a 1e-3 wobble on a near-tie from a
        // real arithmetic difference. Single-GPU had no dump at all, so a tp=1 run
        // could not be compared to a tp=8 one as a VECTOR — which is the only way to
        // ask whether TP's own geometry (K3's 12 local KDA heads at BV=8, against
        // the 96-head BV=16 shape every block gate validates) changed the answer.
        let dump = |e: &AmdEngine, tag: &str| -> Result<(), Box<dyn std::error::Error>> {
            // PLOW_DUMP_ACT, same contract as the TP closure below. It used to exist only
            // there, so every single-GPU model on this box — Gemma-4 among them — silently
            // dumped nothing and the caller saw an empty range report rather than an error.
            if let Some(spec) = plowrt::config::RuntimeConfig::get().amd.dump_act.as_ref() {
                for one in spec.split(',').filter(|s| !s.is_empty()) {
                    if let Some((name, path)) = one.split_once(':') {
                        let n = e
                            .tensor_bytes(name)
                            .ok_or_else(|| format!("PLOW_DUMP_ACT: no tensor {name}"))?
                            as usize;
                        let mut buf = vec![0u8; n];
                        e.read_tensor(name, &mut buf)?;
                        std::fs::write(format!("{path}.{tag}.bin"), &buf)?;
                    }
                }
            }
            let Some(dir) = &dump_logits else {
                return Ok(());
            };
            std::fs::create_dir_all(dir)?;
            let n = e.tensor_bytes("act.logits").ok_or("no act.logits")? as usize;
            let mut buf = vec![0u8; n];
            e.read_tensor("act.logits", &mut buf)?;
            std::fs::write(dir.join(format!("logits_{tag}.bin")), &buf)?;
            Ok(())
        };

        // A multi-token prompt goes through PREFILL, which populates the KV for
        // [0, n) and leaves the first sampled token in in.ids. A single token
        // needs none: decode at position 0 writes KV row 0 and attends over
        // exactly [0,1), so nothing is read that was not written.
        let (first, mut pos) = if ids.len() > 1 {
            let t0 = std::time::Instant::now();
            // A DECODE-ONLY packet has no bucket ladder to chunk a prompt over, and
            // `plan_chunks` refuses one rather than guessing. `amd_bench_tp` has
            // walked the prompt through the decode program a token at a time since
            // GLM-5.2 (whose `glm_emit_full` emits exactly one program); the
            // single-GPU path did not, so `K3_PREFILL=0` — the arm K3 did its whole
            // bring-up on — simply could not run on one GPU. It is a real forward
            // pass: step `p` writes KV row `p` and attends over `[0, p+1)`.
            let tok = if !eng.has_prefill() {
                let mut last = 0;
                for (p, id) in ids.iter().enumerate() {
                    eng.seed_ids(&[*id])?;
                    last = eng.decode_step(p as u32, p as u32 + 1)?;
                }
                last
            } else {
                eng.prefill(&ids)?
            };
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            dump(&eng, "prefill")?;
            println!(
                "\n{timing}prefill: {} tokens in {ms:.1} ms ({:.0} tok/s) -> {tok}",
                ids.len(),
                ids.len() as f64 / (ms / 1e3)
            );
            // Before any decode step runs. The trace buffer is indexed per
            // (workgroup, packet), so every dispatch overwrites the same slots and only
            // the LAST program to run survives -- take the prefill picture here or it is
            // overwritten by a 542-packet GEMV decode and reads as if prefill never ran.
            trace_dump_1(&eng, ".prefill")?;
            (tok, ids.len() as u32)
        } else {
            eng.seed_ids(&ids)?;
            (u32::MAX, 0)
        };

        println!("greedy decode:");
        let mut out = Vec::new();
        if first != u32::MAX {
            out.push(first);
        }
        let mut timed = std::time::Duration::ZERO;
        for s in 0..steps {
            // in.ids is NOT re-seeded: the device wrote the previous step's
            // sampled token there itself, which is what this step embeds.
            let t0 = std::time::Instant::now();
            out.push(eng.decode_step(pos, pos + 1)?);
            if s > 0 {
                timed += t0.elapsed();
            }
            dump(&eng, &format!("{s:03}"))?;
            pos += 1;
        }
        println!("  {out:?}");
        if steps > 1 {
            let timed_steps = steps - 1;
            let ms = timed.as_secs_f64() * 1e3 / timed_steps as f64;
            println!(
                "  {timing}{timed_steps} timed decode steps (+1 discarded warmup): {ms:.3} ms/token ({:.1} tok/s)",
                1e3 / ms
            );
        } else {
            println!("  1 diagnostic decode step; no timing reported (warmup only)");
        }
        if !eng.weights_bound() {
            println!("  (weights unbound — these ids are noise)");
        }
        trace_dump_1(&eng, "")?;
        return Ok(());
    }

    // One untimed step first. The first dispatch of a code object pays its
    // instruction-cache cold miss and any lazy driver work, and folding that
    // into a mean over 32 steps would move it by a whole millisecond.
    let tok = eng.decode_step(ctx, ctx + 1)?;
    println!("\nwarmup step ok (device sampled id {tok}) — the schedule runs");

    let t0 = std::time::Instant::now();
    let mut last = tok;
    for i in 0..steps {
        // §DSTEP: the single-GPU decode path had no `token()` call at all, so
        // `PLOW_DSTEP_LOG=1` produced NOTHING for the one arm that runs on this
        // box. Same shape as the TP mux tick: stamp the whole step, and let the
        // phases inside `decode_step` account for it.
        let t = plowrt::obs::dstep::on().then(std::time::Instant::now);
        last = eng.decode_step(ctx + 1 + i, ctx + 2 + i)?;
        if !synthetic_probe {
            if let Some(t) = t {
                plowrt::obs::dstep::token(t.elapsed().as_nanos() as u64);
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
    println!(
        "\n{timing}{steps} decode steps at ctx={ctx}: {ms:.3} ms/token ({:.1} tok/s), last id {last}",
        1e3 / ms
    );
    println!(
        "  {timing}dispatch accounting: {} launches, enqueue {:.1} us, drain {:.1} us",
        eng.seg_launches, eng.seg_enq_us, eng.seg_drain_us
    );
    if eng.weights_bound() {
        println!(
            "\nWeights ARE bound, but this ran from ctx={ctx} with no prefill, so the \n\
             KV it attended over was never written. The TIMING is representative; \n\
             the ids are not. Pass --prompt for a real greedy decode from 0."
        );
    } else {
        println!(
            "\nWEIGHTS ARE NOT BOUND — the token ids are meaningless. This is a \n\
             SYNTHETIC DIAGNOSTIC, not a performance result."
        );
    }
    // The THIRD exit of this function, and the one a decode-only run takes. Patching only the
    // two `--prompt` exits left `PLOW_TRACE_RAW` silent for exactly the run that produces a
    // CLEAN decode trace -- a `--prompt` run overwrites only the first 542 of the buffer's 718
    // packet slots, so the leftover prefill records poison the decode report (they read as a
    // negative span and 200,000 us stragglers).
    trace_dump_1(&eng, "")?;
    Ok(())
}

/// Bring up a TP group and time decode steps across every rank.
///
/// The oracle is `runtime/tests/tp_decode.c --tp N`: every rank must emit an
/// IDENTICAL token stream. That is not a sanity check bolted on the side, it is
/// the acceptance test — a rank whose collective silently timed out still
/// samples fluent-looking ids from its own shard, so agreement is the only thing
/// that distinguishes a working all-reduce from a plausible wrong one. It is
/// therefore asserted on every step rather than at the end.
/// Write the packet trace, if `PLOW_TRACE_RAW` asked for one.
///
/// A FUNCTION rather than three copies of the same `if let`, because every copy so far has been on
/// a path some other exit bypassed. `--batched` returned before one; `--prompt` returned before the
/// other. So the instrument was unavailable in exactly the two places the interesting questions
/// live — batch > 1, and PREFILL — and both times it failed by producing NOTHING rather than by
/// complaining. Call this before every exit of a bench that ran packets.
/// The single-GPU twin of [`trace_dump`].
///
/// `amd-bench` routes `--tp 1` to `amd_bench`, which had NO trace call on ANY exit -- so
/// `PLOW_TRACE_RAW` silently produced nothing for every single-GPU run, which is most of them.
/// That is precisely the failure mode [`trace_dump`]'s own comment was written about: the
/// instrument was unavailable exactly where the questions are, and it failed by writing no file
/// rather than by complaining.
#[cfg(feature = "hsa")]
fn trace_dump_1(
    eng: &plowrt::exec::amd::AmdEngine,
    suffix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(p) = plowrt::config::RuntimeConfig::get().amd.trace_raw.as_ref() {
        let mut p = PathBuf::from(p).into_os_string();
        p.push(suffix);
        let p = PathBuf::from(p);
        eng.trace_write(&p)?;
        println!("packet trace -> {}", p.display());
    }
    Ok(())
}

#[cfg(feature = "hsa")]
fn trace_dump(g: &plowrt::exec::amd_tp::AmdTpGroup) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(p) = plowrt::config::RuntimeConfig::get().amd.trace_raw.as_ref() {
        let all = plowrt::config::RuntimeConfig::get().amd.trace_allranks;
        for rank in 0..if all { g.n_gpu() } else { 1 } {
            let mut out = PathBuf::from(p).into_os_string();
            if all {
                out.push(format!(".rk{rank}"));
            }
            let out = PathBuf::from(out);
            g.rank(rank).trace_write(&out)?;
            println!("packet trace -> {}", out.display());
        }
    }
    Ok(())
}

#[cfg(feature = "hsa")]
fn amd_bench_tp(
    blob: PathBuf,
    hsaco: PathBuf,
    checkpoint: Option<PathBuf>,
    synthetic_probe: bool,
    prompt: Option<String>,
    steps: u32,
    ctx: u32,
    tp: u32,
    batched: bool,
    dump_logits: Option<PathBuf>,
    prefill_sweep: Option<String>,
    prefill_reps: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::amd_tp::AmdTpGroup;

    if synthetic_probe {
        eprintln!("SYNTHETIC DIAGNOSTIC: weights are unbound; output is not a performance result");
    }
    let timing = synthetic_timing_prefix(synthetic_probe);
    let mut backends = Vec::with_capacity(tp as usize);
    for d in 0..tp {
        backends.push(Arc::new(plowrt::device::hsa::HsaBackend::new(d as u8)?));
    }
    let t0 = std::time::Instant::now();
    let mut g = AmdTpGroup::load(backends, &blob, &hsaco, checkpoint.as_deref())?;
    // This binary is the TP CORRECTNESS ORACLE: its claim is that every rank
    // emitted an IDENTICAL stream, and a sampled check cannot support that
    // sentence. Serving samples (`DEFAULT_AGREE_EVERY`); the oracle never does.
    g.audit_cadence(1);
    println!(
        "{timing}loaded in {:.1} s: TP={} ranks, max_ctx={}",
        t0.elapsed().as_secs_f64(),
        g.n_gpu(),
        g.max_ctx()
    );

    // `act.logits` is a real device tensor — the lm_head GEMM writes it and the
    // device argmax reads it — so it survives the launch and rank 0 holds the
    // FULL vocab row (lm_head is replicated, every rank argmaxes the same id).
    // Dumping it is what makes prefill-vs-decode comparable as a VECTOR: the
    // greedy id alone cannot distinguish a 1e-3 wobble on a near-tie from a
    // real arithmetic difference, and the whole prefill question is which of
    // those a token flip is.
    let dump = |g: &AmdTpGroup, tag: &str| -> Result<(), Box<dyn std::error::Error>> {
        // PLOW_DUMP_ACT="name:path[,name:path...]" — download rank 0's copy of any act
        // tensor after the prefill/step this closure runs for. A measurement instrument
        // (e.g. the DSA union-coverage analysis reads `act.iumask`), not a serving path;
        // dumped once per tag, later tags overwrite-with-suffix like the logits do.
        if let Some(spec) = plowrt::config::RuntimeConfig::get().amd.dump_act.as_ref() {
            for one in spec.split(',').filter(|s| !s.is_empty()) {
                if let Some((name, path)) = one.split_once(':') {
                    let n = g
                        .rank(0)
                        .tensor_bytes(name)
                        .ok_or_else(|| format!("PLOW_DUMP_ACT: no tensor {name}"))?
                        as usize;
                    let mut buf = vec![0u8; n];
                    g.rank(0).read_tensor(name, &mut buf)?;
                    std::fs::write(format!("{path}.{tag}.bin"), &buf)?;
                }
            }
        }
        let Some(dir) = &dump_logits else {
            return Ok(());
        };
        std::fs::create_dir_all(dir)?;
        let n = g
            .rank(0)
            .tensor_bytes("act.logits")
            .ok_or("no act.logits")? as usize;
        let mut buf = vec![0u8; n];
        g.rank(0).read_tensor("act.logits", &mut buf)?;
        std::fs::write(dir.join(format!("logits_{tag}.bin")), &buf)?;
        Ok(())
    };

    // TIER-2 PREFILL SWEEP. One load, every context, median of N — see the flag's own
    // documentation for why the `--prompt` path cannot serve this role. Terminal: it
    // returns rather than falling through to the decode ladder, because a sweep run has
    // no single position to decode from.
    if let Some(spec) = prefill_sweep.as_deref() {
        let lens: Vec<u32> = spec
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<u32>())
            .collect::<std::result::Result<_, _>>()?;
        if lens.is_empty() {
            return Err("--prefill-sweep parsed to no lengths".into());
        }
        if lens.contains(&0) {
            return Err("--prefill-sweep contains a zero-length prompt".into());
        }
        println!(
            "\n{timing}prefill sweep: {} rep(s) + 1 discarded warm-up per length, max_ctx={}",
            prefill_reps,
            g.max_ctx()
        );
        for t in lens {
            if t as usize > g.max_ctx() {
                println!("  T={t:<6} SKIP (> max_ctx {})", g.max_ctx());
                continue;
            }
            // The ids are meaningless by construction (weights are unbound unless a
            // checkpoint was passed); only the ROW COUNT reaches the schedule.
            let ids: Vec<u32> = (0..t).map(|i| 100 + (i % 1000)).collect();
            let mut ms: Vec<f64> = Vec::with_capacity(prefill_reps as usize);
            for r in 0..=prefill_reps {
                let t0 = std::time::Instant::now();
                let _ = AmdTpGroup::agree(&g.prefill(&ids)?)?;
                if r > 0 {
                    ms.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ms[ms.len() / 2];
            let spread = if med > 0.0 {
                100.0 * (ms[ms.len() - 1] - ms[0]) / med
            } else {
                0.0
            };
            let reps: Vec<String> = ms.iter().map(|v| format!("{v:.3}")).collect();
            println!(
                "  {timing}PFSWEEP T={t} median_ms={med:.3} spread_pct={spread:.2} reps=[{}]",
                reps.join(",")
            );
        }
        trace_dump(&g)?;
        return Ok(());
    }

    // BATCHED TP — the arm `scripts/k3_batch_gate.sh` drives. K3 is TP8-only, so without this
    // the gate has nothing to run against and the two refusals could never be retired on
    // evidence.
    //
    // Its output format is load-bearing: one `  [id, id, ...]` line PER SLOT, which is exactly
    // the shape the scalar arm below prints for its single stream. The gate compares slot `s` of
    // a batched run against a solo B=1 run of the same prompt, so the two must be printed
    // identically or the comparison is between a chain and a formatting difference.
    if batched {
        let b = g.rank(0).batch();
        let prompts: Vec<Vec<u32>> = match &prompt {
            None => Vec::new(),
            Some(p) => p
                .split(';')
                .map(|one| {
                    one.split(',')
                        .map(|s| s.trim().parse::<u32>())
                        .collect::<std::result::Result<Vec<u32>, _>>()
                })
                .collect::<std::result::Result<_, _>>()?,
        };
        if prompts.is_empty() {
            return Err(
                "--batched on TP needs --prompt: without one every slot decodes over KV \
                        nobody wrote, and agreement between slots is then a statement about VRAM \
                        history rather than about batching"
                    .into(),
            );
        }
        println!(
            "\nbatched TP decode: {b} sequences per dispatch, {} ranks",
            g.n_gpu()
        );

        let mut pos_v: Vec<u32> = vec![0; b];
        let mut feed: Vec<u32> = vec![0; b];
        for s in 0..b {
            let ids = &prompts[s % prompts.len()];
            // Prefill is single-sequence on every rank; `prefill_slot` rebases the whole group's
            // KV pointer tables onto slot `s` for the duration and restores them after, so each
            // slot's cache is genuinely populated by this run.
            let tok = AmdTpGroup::agree(&g.prefill_slot(s, ids)?)?;
            println!("  slot {s}: prefill {} tokens -> sampled {tok}", ids.len());
            pos_v[s] = ids.len() as u32;
            feed[s] = tok;
        }

        let mut chains: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut timed = std::time::Duration::ZERO;
        for step in 0..steps {
            // SEED EVERY ROW EXPLICITLY. Prefill is single-sequence and writes `in.ids[0]` only,
            // so rows 1.. still hold whatever the last prefill left there; and after a step the
            // device has written all B rows itself, but re-seeding from the host chain keeps the
            // fed id and the recorded chain provably the same value.
            g.seed_ids(&feed)?;
            let kv: Vec<u32> = pos_v.iter().map(|x| x + 1).collect();
            let t = std::time::Instant::now();
            let out = g.decode_step_batched(&pos_v, &kv)?;
            if step > 0 {
                timed += t.elapsed();
            }
            dump(&g, &format!("b{:03}", chains[0].len()))?;
            for s in 0..b {
                chains[s].push(out[s]);
                feed[s] = out[s];
                pos_v[s] += 1;
            }
        }
        for c in &chains {
            println!("  {c:?}");
        }
        if steps > 1 {
            let timed_steps = steps - 1;
            let ms = timed.as_secs_f64() * 1e3 / timed_steps as f64;
            println!(
                "  {timing}{timed_steps} timed batched steps (+1 discarded warmup): {ms:.3} ms/step, {:.1} tok/s AGGREGATE over {b} \
                 sequences ({:.1} tok/s per stream), all {} ranks token-identical",
                b as f64 * 1e3 / ms,
                1e3 / ms,
                g.n_gpu()
            );
        } else {
            println!("  1 diagnostic batched step; no timing reported (warmup only)");
        }
        trace_dump(&g)?;
        return Ok(());
    }

    // A prompt makes this a real greedy decode from position 0. Without one,
    // decode starts mid-context over KV rows nobody wrote — the timing is
    // representative and the ids are not.
    let mut pos = ctx;
    if let Some(p) = &prompt {
        let ids: Vec<u32> = p
            .split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<std::result::Result<_, _>>()?;
        if ids.is_empty() {
            return Err("--prompt is empty".into());
        }
        if ids.len() > 1 {
            let t = std::time::Instant::now();
            // A DECODE-ONLY packet has no bucket ladder to chunk a prompt over —
            // GLM-5.2's `glm_emit_full` emits exactly one program, because the
            // grouped block-fp8 MoE prefill kernels the emitter would need do not
            // exist (`crates/devgen/src/mla.rs`). Walking the prompt through the
            // decode program one token at a time is what
            // `runtime/tests/glm52_decode.c` does, and it is a real forward pass:
            // step `p` writes KV row `p` and attends over `[0, p+1)`, so nothing
            // is read that was not written. It is O(prompt) dispatches, hence a
            // fallback and not the default.
            let tok = if !g.rank(0).has_prefill() {
                let mut last = 0;
                for (p, id) in ids.iter().enumerate() {
                    g.seed_ids(&[*id])?;
                    last = AmdTpGroup::agree(&g.decode_step(p as u32, p as u32 + 1)?)?;
                }
                last
            } else {
                AmdTpGroup::agree(&g.prefill(&ids)?)?
            };
            let ms = t.elapsed().as_secs_f64() * 1e3;
            dump(&g, "prefill")?;
            println!(
                "\n{timing}prefill: {} tokens in {ms:.1} ms ({:.0} tok/s) -> {tok} \
                 (all {} ranks agree)",
                ids.len(),
                ids.len() as f64 / (ms / 1e3),
                g.n_gpu()
            );
            pos = ids.len() as u32;
            // PREFILL TRACE, dumped HERE and not at the end.
            //
            // The trace buffer is indexed per (workgroup, packet), so every dispatch overwrites
            // the same slots and only the LAST program run survives to the file. A run that
            // prefills and then decodes therefore writes a DECODE trace no matter how big the
            // prompt was -- which is why the first prefill trace ever taken turned out to be
            // 2459 packets of GEMV, i.e. the decode program. Prefill has been untraceable.
            //
            // Written to `<PLOW_TRACE_RAW>.prefill` so both survive one run.
            if let Some(t) = plowrt::config::RuntimeConfig::get().amd.trace_raw.as_ref() {
                let all = plowrt::config::RuntimeConfig::get().amd.trace_allranks;
                for rank in 0..if all { g.n_gpu() } else { 1 } {
                    let mut pf = PathBuf::from(t).into_os_string();
                    if all {
                        pf.push(format!(".rk{rank}"));
                    }
                    pf.push(".prefill");
                    let pf = PathBuf::from(pf);
                    g.rank(rank).trace_write(&pf)?;
                    println!("prefill packet trace -> {}", pf.display());
                }
            }
        } else {
            g.seed_ids(&ids)?;
            pos = 0;
        }

        println!("greedy decode:");
        let mut out = Vec::new();
        let mut timed = std::time::Duration::ZERO;
        for s in 0..steps {
            let t_tok = plowrt::obs::dstep::on().then(std::time::Instant::now);
            let t = std::time::Instant::now();
            let ids = g.decode_step(pos, pos + 1)?;
            out.push(plowrt::obs::dstep::timed(
                &plowrt::obs::dstep::AGREE,
                || AmdTpGroup::agree(&ids),
            )?);
            if s > 0 {
                timed += t.elapsed();
            }
            if !synthetic_probe {
                if let Some(t) = t_tok {
                    plowrt::obs::dstep::token(t.elapsed().as_nanos() as u64);
                }
            }
            dump(&g, &format!("{s:03}"))?;
            pos += 1;
        }
        println!("  {out:?}");
        if steps > 1 {
            let timed_steps = steps - 1;
            let ms = timed.as_secs_f64() * 1e3 / timed_steps as f64;
            println!(
                "  {timing}{timed_steps} timed decode steps (+1 discarded warmup): {ms:.3} ms/token ({:.1} tok/s), all {} ranks \
                 token-identical",
                1e3 / ms,
                g.n_gpu()
            );
        } else {
            println!("  1 diagnostic decode step; no timing reported (warmup only)");
        }
        if !g.weights_bound() {
            println!("  (weights unbound — these ids are noise)");
        }
        // This path had NO trace write at all, so `PLOW_TRACE_RAW` produced nothing for any
        // run carrying a `--prompt` — i.e. every run that PREFILLS. Prefill was untraceable.
        trace_dump(&g)?;
        return Ok(());
    }

    // Untimed warmup: the first dispatch of a code object pays its i-cache cold
    // miss, and folding that into a mean over `steps` moves it by a millisecond.
    let warm = g.decode_step(pos, pos + 1)?;
    println!(
        "\nwarmup ok (ranks sampled {warm:?}) — {}",
        match AmdTpGroup::agree(&warm) {
            Ok(t) => format!("all ranks agree on {t}"),
            Err(e) => format!("DISAGREE: {e}"),
        }
    );

    let t = std::time::Instant::now();
    let mut disagreements = 0u32;
    for i in 0..steps {
        // §DSTEP owns the whole step so `TOKEN` is a real total, not a sum of
        // parts. `PLOW_DSTEP_LOG=1` prints the host/GPU split every
        // `PLOW_DSTEP_EVERY` tokens; off, this is one `OnceLock` load.
        let t_tok = plowrt::obs::dstep::on().then(std::time::Instant::now);
        let ids = g.decode_step(ctx + 1 + i, ctx + 2 + i)?;
        if plowrt::obs::dstep::timed(&plowrt::obs::dstep::AGREE, || AmdTpGroup::agree(&ids))
            .is_err()
        {
            disagreements += 1;
        }
        if !synthetic_probe {
            if let Some(t) = t_tok {
                plowrt::obs::dstep::token(t.elapsed().as_nanos() as u64);
            }
        }
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
    println!(
        "\n{timing}{steps} decode steps at ctx={ctx}, TP={}: {ms:.3} ms/token ({:.1} tok/s)",
        g.n_gpu(),
        1e3 / ms
    );
    if disagreements == 0 {
        println!("  every step: all {} ranks token-identical", g.n_gpu());
    } else {
        println!(
            "  *** {disagreements}/{steps} steps had ranks DISAGREE — a collective did \
             not run ***"
        );
    }
    if !g.weights_bound() {
        println!(
            "\nWEIGHTS ARE NOT BOUND — the ids are meaningless, so rank agreement is \n\
             NOT evidence the collectives ran (every rank computes the same nothing). \n\
             This is a SYNTHETIC DIAGNOSTIC, not a performance result. Pass \
             --checkpoint for the real token-identity check."
        );
    }
    // PACKET TRACE. Dumped AFTER the timed loop, so the records are a
    // steady-state step's and not the warmup's. Every rank traces (the buffer
    // is allocated per engine off the same env var); rank 0's is the one that
    // gets written, which is the same rank the token-agreement check reads.
    trace_dump(&g)?;
    Ok(())
}

/// Drive a block asset: fill `act.x`, run its prefill program, read `act.x`.
#[cfg(feature = "hsa")]
fn amd_block(
    blob: PathBuf,
    hsaco: PathBuf,
    checkpoint: Option<PathBuf>,
    prompt: Option<String>,
    inspect: String,
    list_tensors: bool,
    dump: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::amd::AmdEngine;

    let be = Arc::new(plowrt::device::hsa::HsaBackend::new(0)?);
    let mut eng = AmdEngine::load(Arc::clone(&be), &blob, &hsaco, checkpoint.as_deref())?;
    if list_tensors {
        for n in eng.tensor_names() {
            println!("{n}\t{}", eng.tensor_bytes(n).unwrap_or(0));
        }
        return Ok(());
    }

    // The lm_head's operands BEFORE anything runs: if its weight is a tensor
    // nothing filled, it is zero on device and the logits are zero regardless
    // of how healthy the activation is.
    for p in 0..eng.n_programs() {
        if let Some((idx, op, ops)) = eng.lm_head_operands(p) {
            println!("program {p}: lm_head is inst {idx}, op {op}");
            for (slot, name) in &ops {
                let bytes = eng.tensor_bytes(name).unwrap_or(0);
                let n = (bytes as usize).min(1 << 22);
                let mut buf = vec![0u8; n];
                let nz = match eng.read_tensor(name, &mut buf) {
                    Ok(()) => buf.iter().filter(|&&b| b != 0).count(),
                    Err(_) => 0,
                };
                println!(
                    "  t[{slot}] = {name:<52} {bytes:>12} B  {}",
                    if nz == 0 {
                        "ALL ZERO <-- nothing filled this".to_string()
                    } else {
                        format!("{:.1}% non-zero bytes", 100.0 * nz as f64 / n as f64)
                    }
                );
            }
            if let Some((blocks, i, n_ent, segs)) = eng.lm_head_detail() {
                println!("  blocks={blocks} i={i:?}");
                println!(
                    "  scheduled by {n_ent} stream entries, segments {segs:?}{}",
                    if n_ent == 0 {
                        "  <-- NEVER SCHEDULED: emitted but no stream entry runs it"
                    } else {
                        ""
                    }
                );
            }
            break;
        }
    }

    let ids: Vec<u32> = prompt
        .as_deref()
        .unwrap_or("2,1000,2000,3000")
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<std::result::Result<_, _>>()?;
    let tok = eng.prefill(&ids)?;
    println!("prefill {} tokens -> sampled id {tok}", ids.len());

    // ZERO vs WRONG is the whole question and the two are different hunts. A
    // scale-convention or arithmetic bug gives wrong-but-VARIED values; an
    // all-zero tensor is a store that never landed, or a counter gate that let
    // the consumer run before the producer wrote. Walking the chain says which
    // link went quiet.
    println!(
        "\n{:<16} {:>12} {:>10} {:>14}",
        "tensor", "non-zero", "%", "sum|x| (bf16)"
    );
    for name in inspect.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let Some(bytes) = eng.tensor_bytes(name) else {
            println!("{name:<16} {:>12}", "(absent)");
            continue;
        };
        let n = (bytes as usize / 2).min(1 << 22);
        let mut buf = vec![0u8; n * 2];
        eng.read_tensor(name, &mut buf)?;
        let nz = buf.chunks_exact(2).filter(|c| c != &[0u8, 0u8]).count();
        let sum: f64 = buf
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).abs() as f64)
            .sum();
        println!(
            "{name:<16} {nz:>12} {:>9.1}% {sum:>14.4}",
            100.0 * nz as f64 / n as f64
        );
        if let Some(dir) = &dump {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join(format!("{name}.bin")), &buf)?;
        }
    }
    if let Some(dir) = &dump {
        println!(
            "\nwrote raw tensors to {} — diff two precisions byte-wise",
            dir.display()
        );
    }
    println!(
        "\nALL-ZERO on a tensor means a store that never landed or a gate that let\n\
         the consumer read before the producer wrote — NOT arithmetic, which gives\n\
         wrong-but-varied values. The first zero tensor walking the chain is the\n\
         link to investigate."
    );
    Ok(())
}

/// Enumerate devices and optionally bring up the TP group.
///
/// The first real caller of [`device::select_all`], which had none: plowrt
/// bound device 0 and nothing else. The enumeration is carved into the node's
/// TP replicas, and a backend's position within its replica is its rank.
fn devices(
    tp: Option<u32>,
    hidden: u32,
    max_tokens: u32,
    layers: u32,
    prefill: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::tp::{PeerLayout, TpGroup};

    let all = device::select_all(1);
    println!("visible devices: {}", all.len());
    for (i, be) in all.iter().enumerate() {
        let peer = be.peer();
        println!(
            "  [{i}] class={:?} vendor={:?} executors={} peer={}",
            be.class(),
            be.vendor(),
            be.enumerate().len(),
            match peer {
                Some(p) => format!("yes, maps {} agents", p.peer_agent_count()),
                None => "no".into(),
            }
        );
    }

    let Some(n) = tp else {
        return Ok(());
    };
    if n as usize > all.len() {
        return Err(format!(
            "--tp {n} but only {} devices are visible{}",
            all.len(),
            visible_mask_hint()
        )
        .into());
    }

    // Every whole replica the node can hold, not just the first: 2 × TP4 on an
    // 8-GPU node is the deployment shape, and the point of bringing both up
    // here is to prove they are independent.
    let n_xctr = PeerLayout::counters_for(layers, prefill);
    let layout = PeerLayout::new(hidden, max_tokens, n_xctr).ok_or_else(|| {
        format!("hidden={hidden} x max_tokens={max_tokens} x 2 B is not 128 B-aligned")
    })?;
    let groups = TpGroup::split_replicas(all, n, layout)?;
    println!(
        "\n{} replica(s) of TP={n}, hidden={hidden}, tokens/dispatch={max_tokens}, \
         {} ({n_xctr} xctr gates over {layers} layers), \
         peer footprint={} B/rank (partials {} B, xctr {} B at +{})",
        groups.len(),
        if prefill {
            "PREFILL two-shot"
        } else {
            "DECODE one-shot"
        },
        layout.bytes(),
        layout.xctr_off(),
        layout.xctr_bytes(),
        layout.xctr_off(),
    );
    for (i, group) in groups.iter().enumerate() {
        println!("  replica {i}:");
        for r in group.ranks() {
            println!(
                "    rank {} dev {} grid={} peer_scratch={:#x} xctr={:#x} table={:#x}",
                r.rank(),
                r.ordinal(),
                r.executors(),
                r.scratch_base(),
                r.xctr(),
                r.peer_scratch_table(),
            );
        }
        group.verify_peer_visibility()?;
        group.zero_xctr()?;
        println!(
            "    {} directed peer pairs byte-exact; all ranks' xctr zeroed \
             (the §6d pre-launch obligation)",
            n * (n - 1)
        );

        // The one piece of per-token HOST work the design has: counter reset.
        // plow's claim is a token costs one dispatch per GPU and nothing else,
        // so this number is the size of the gap between that claim and today's
        // `XctrReset::Host`. Measured, not assumed — 96 all-reduces per token
        // is not a budget that survives an unmeasured host pass.
        const ITERS: u32 = 200;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            group.zero_xctr()?;
        }
        let sdma = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            group.zero_xctr_direct()?;
        }
        let direct = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        println!(
            "    xctr reset ({} B x {n} ranks): Host(copy engine) {sdma:.1} us/token, \
             HostDirect(BAR stores) {direct:.2} us/token, Program 0 \
             (needs monotonic device counters). For scale, one inline one-shot \
             all-reduce costs 0.302 us (measured), so a 96-collective decode \
             token spends ~29 us in collectives.",
            layout.xctr_bytes()
        );
    }
    println!(
        "\nNOT exercised here: no persistent dispatch — this brings up the peer \n\
         buffers and counters WITHOUT a model, so nothing runs the collective \n\
         packets. To actually run them: plowrt amd-bench --tp N --blob <a packet \n\
         compiled with plowc --num-gpus N>."
    );
    Ok(())
}

/// `plowrt op-audit` — classify a blob's opcodes by row identity.
///
/// The exit status is the verdict: `0` when every program can be executed as one
/// packed token batch, `1` when any opcode is refused. That makes the audit
/// usable as a gate in the enablement of a (family, backend) pair, which is what
/// the plan asks of it — not just as something to read.
fn op_audit_cmd(
    blob_path: Option<PathBuf>,
    table: bool,
    program: Option<u32>,
    format: String,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asset::devblob::DevBlob;

    if table {
        let rows = plowrt::opaudit::table();
        match format.as_str() {
            "json" => println!("{}", serde_json::to_string_pretty(&rows)?),
            "text" => print!("{}", plowrt::opaudit::table_text(&rows)),
            other => return Err(format!("--format wants text|json, got `{other}`").into()),
        }
        return Ok(());
    }

    let blob_path = blob_path.ok_or("op-audit wants a blob path, or --table")?;
    // Same convention as `disasm`: a directory means the `model.pkt` in it.
    let p = blob_path.as_path();
    let file = if p.is_dir() {
        p.join("model.pkt")
    } else {
        p.to_path_buf()
    };
    let buf = std::fs::read(&file).map_err(|e| format!("read {}: {e}", file.display()))?;
    let magic: Option<&[u8; 8]> = buf.get(..8).and_then(|s| s.try_into().ok());
    if !magic.is_some_and(packet::devbuild::is_blob_magic) {
        return Err(format!("{}: not a PLOWDEV blob", file.display()).into());
    }
    // Static inspection, so an L2-placed blob must be readable — see `disasm_cmd`.
    let blob = DevBlob::parse_l2(&buf, true)?;
    let rep = plowrt::opaudit::audit(&blob, &file.display().to_string(), program);

    match format.as_str() {
        "json" => println!("{}", serde_json::to_string_pretty(&rep)?),
        "text" => print!("{}", plowrt::opaudit::text(&rep)),
        other => return Err(format!("--format wants text|json, got `{other}`").into()),
    }
    if !rep.packable {
        std::process::exit(1);
    }
    Ok(())
}

/// `plowrt disasm` — read a device blob and render it.
///
/// No device, no features, no driver. The heavy lifting is in
/// `plowrt::disasm`; this is argument handling and output.
fn disasm_cmd(
    blob_path: PathBuf,
    program: Option<u32>,
    range: Option<String>,
    format: String,
    sections: plowrt::disasm::Sections,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asset::devblob::DevBlob;

    // Same convention as `graphstat`: a directory means the `model.pkt` in it.
    let p = blob_path.as_path();
    let file = if p.is_dir() {
        p.join("model.pkt")
    } else {
        p.to_path_buf()
    };
    let buf = std::fs::read(&file).map_err(|e| format!("read {}: {e}", file.display()))?;

    // Both compiler outputs are called `.pkt`, so the format is decided by the
    // magic rather than by the extension or a flag. A scheduled stream
    // (`--emit packets`) has no programs, tensors or kernargs, so it takes a
    // different renderer entirely — see `plowrt::disasm::sched`.
    let magic: Option<&[u8; 8]> = buf.get(..8).and_then(|s| s.try_into().ok());
    if !magic.is_some_and(packet::devbuild::is_blob_magic) {
        let prog = packet::Program::decode(&buf).map_err(|e| {
            format!(
                "{}: not a PLOWDEV blob, and not a scheduled .pkt either: {e}",
                file.display()
            )
        })?;
        let rep = plowrt::disasm::sched::report(&prog, &file.display().to_string());
        match format.as_str() {
            "json" | "jsonl" => println!("{}", serde_json::to_string_pretty(&rep)?),
            "text" => print!("{}", plowrt::disasm::sched::text(&rep)),
            other => return Err(format!("--format wants text|json|jsonl, got `{other}`").into()),
        }
        return Ok(());
    }

    // `disasm` is STATIC INSPECTION -- no device, no dispatch -- so it must not refuse an
    // L2-placed blob. It did, which is why reading one needed PLOW_L2_PLACE_DISPATCH=1 set.
    let blob = DevBlob::parse_l2(&buf, true)?;

    let range = match range.as_deref() {
        None => None,
        Some(s) => {
            let (lo, hi) = s
                .split_once("..")
                .ok_or_else(|| format!("--range wants `lo..hi`, got `{s}`"))?;
            Some((lo.trim().parse::<usize>()?, hi.trim().parse::<usize>()?))
        }
    };

    let rep = plowrt::disasm::report(&blob, &file.display().to_string(), sections, program, range);

    match format.as_str() {
        "json" => println!("{}", serde_json::to_string_pretty(&rep)?),
        // One PROGRAM per line, not one instruction: a record has to be
        // self-describing to be worth streaming, and a bare instruction is not —
        // it carries no `T`. A GLM prefill program is ~2k instructions, which is
        // a comfortable line for `jq`.
        "jsonl" => {
            for prog in &rep.programs {
                println!("{}", serde_json::to_string(prog)?);
            }
        }
        "text" => print_disasm_text(&blob, &rep, program, range),
        other => return Err(format!("--format wants text|json|jsonl, got `{other}`").into()),
    }
    Ok(())
}

fn print_disasm_text(
    blob: &plowrt::asset::devblob::DevBlob,
    rep: &plowrt::disasm::BlobReport<'_>,
    program: Option<u32>,
    range: Option<(usize, usize)>,
) {
    println!("blob      {}", rep.blob);
    println!(
        "n_cu      {}  target=0x{:08x}  flags=0x{:x}",
        rep.n_cu, rep.target, rep.flags
    );
    println!(
        "programs  {:?}",
        blob.progs.iter().map(|p| p.t).collect::<Vec<_>>()
    );
    if let Some(tp) = &rep.tp {
        println!(
            "tp        n_gpu={} hidden={} slot_bytes={}",
            tp.n_gpu, tp.hidden, tp.slot_bytes
        );
    }

    if let Some(ts) = &rep.tensors {
        println!("\ntensors ({})", ts.len());
        for t in ts {
            println!(
                "  {:>5}  {:<9} {:>14}  {}{}",
                t.handle,
                t.class,
                t.bytes,
                t.name,
                if t.init { "  (init)" } else { "" }
            );
        }
    }

    let names: Vec<&str> = blob.tensors.iter().map(|t| t.name.as_str()).collect();
    for (pr, prog) in rep.programs.iter().zip(
        blob.progs
            .iter()
            .filter(|p| program.is_none_or(|t| p.t == t)),
    ) {
        println!(
            "\n===== program T={}  {} insts, {} counters",
            pr.t, pr.n_inst, pr.n_counter
        );

        if let Some(k) = &pr.kernargs {
            println!("  kernargs ({} B)", k.size_bytes);
            for (name, state) in &k.pointers {
                println!("    {name:<14} {state}");
            }
            for (name, v) in &k.scalars {
                println!("    {name:<14} {v}");
            }
            let d = &k.derived;
            println!(
                "    -> scheduler={} segmented={} n_seg={} l2_placed={} tp={} n_gpu={}",
                d.scheduler, d.segmented, d.n_seg, d.l2_placed, d.tensor_parallel, d.n_gpu
            );
        }

        if let Some(c) = &pr.counters {
            let a = &c.aggregate;
            println!(
                "  counters: {} ({} dead)  edges {} -> {} tr ({} redundant)  \
                 polls {} -> {} ({} removable)  bumps {}/{} live  critical path {}",
                a.counters,
                a.dead,
                a.edges,
                a.edges_tr,
                a.redundant,
                a.polls,
                a.polls_tr,
                a.polls_removable,
                a.bumps_live,
                a.bumps,
                a.critical_path
            );
            println!(
                "  liveness: peak {} concurrent at #{}  p50={} p99={}",
                c.liveness.max_concurrent, c.liveness.at_inst, c.liveness.p50, c.liveness.p99
            );
            for d in c.dead.iter().take(10) {
                println!(
                    "    DEAD counter {} produced by #{} {} — {} wasted bumps",
                    d.id,
                    d.producer
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "?".into()),
                    d.producer_op.unwrap_or("?"),
                    d.bump_cost
                );
            }
            for e in c.redundant_edges.iter().take(10) {
                println!(
                    "    redundant {} -> {} via {:?}  saves {} polls{}",
                    e.from,
                    e.to,
                    e.via,
                    e.polls_saved,
                    if e.co_placed {
                        "  (co-placed: poll only, no overlap)"
                    } else {
                        ""
                    }
                );
            }
            if c.redundant_edges.len() > 10 {
                println!(
                    "    ... {} more redundant edges",
                    c.redundant_edges.len() - 10
                );
            }
        }

        let (lo, hi) = range.unwrap_or((0, prog.insts.len()));
        let (lo, hi) = (lo.min(prog.insts.len()), hi.min(prog.insts.len()));
        for (k, w) in prog.insts[lo..hi].iter().enumerate() {
            println!(
                "{}",
                plowrt::disasm::text_inst(&packet::disasm::disasm(lo + k, w, &names))
            );
        }
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

async fn bringup_runtime(
    assets: Vec<PathBuf>,
    executors: u32,
    trace: bool,
    mux_cfg: MuxConfig,
) -> Result<Arc<AppState>, Box<dyn std::error::Error>> {
    // One binary, CPU or GPU: the vendor drivers are `dlopen`ed, so this probes
    // CUDA then HSA (AMD) and falls back to the CPU reference backend when neither
    // loads. Assets stay servable either way — every one of them is compiled for
    // a GPU spec, and the CPU backend interprets that same program.
    //
    // The CUDA probe keeps a TYPED handle: the sm_120 engine needs the
    // backend's cooperative-launch surface, which `dyn Backend` erases.
    // Probe the FIRST REQUESTED device, not device 0. Opening 0 unconditionally
    // retained a primary context on a GPU `--devices 1` explicitly excluded —
    // and, worse, left it in the backend list so placement handed it the first
    // model. The device this opens is a device we intend to use.
    #[cfg(feature = "cuda")]
    let cuda_probe: u8 = RuntimeConfig::get()
        .devices
        .first()
        .copied()
        .unwrap_or(0)
        .min(u32::from(u8::MAX)) as u8;
    #[cfg(feature = "cuda")]
    let cuda: Option<Arc<device::cuda::CudaBackend>> =
        match device::cuda::CudaBackend::new(cuda_probe) {
            Ok(b) => Some(Arc::new(b)),
            // A device the operator NAMED must not degrade to a CPU fallback:
            // that answers requests with reference-interpreter output at
            // fictional speed, which is the failure mode this file already
            // refuses elsewhere.
            Err(e) if !RuntimeConfig::get().devices.is_empty() => {
                return Err(format!(
                    "--devices names device {cuda_probe}, which will not open: {e}"
                )
                .into())
            }
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
    let cfg = RuntimeConfig::get();
    if vendor != Some(hwspec::Vendor::Nvidia)
        && (!cfg.devices.is_empty()
            || !cfg.pin.is_empty()
            || cfg.place != plowrt::serve::placement::Place::Spread
            || cfg.nv.vram_budget_mib.is_some())
    {
        return Err("model placement and VRAM residency management currently require CUDA; use ROCR_VISIBLE_DEVICES to restrict AMD devices".into());
    }
    if vendor.is_some() {
        tracing::info!(class = ?backend.class(), vendor = ?vendor, "backend ready — GPU accelerated");
    } else if cfg!(feature = "cpu") {
        tracing::info!("CPU execution selected; native engine enabled for device blobs");
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

    let registry = Registry::new();
    for dir in &assets {
        let slug = registry.load(dir, None)?;
        let target = registry.get(&slug)?.manifest.gpu.clone();
        let target_vendor = hwspec::registry::lookup(&target).map(|s| s.vendor);
        if target_vendor.is_some() && target_vendor == vendor {
            tracing::info!(dir = %dir.display(), %target, "loaded model bundle");
        } else if cfg!(feature = "cpu")
            && vendor.is_none()
            && plowrt::asset::devblob::DevBlob::find_in_dir(dir)?.is_some()
        {
            tracing::info!(dir = %dir.display(), %target, "loaded model bundle for CPU engine");
        } else {
            // A DEVICE BLOB WITH NO DEVICE IS A REFUSAL, NOT A WARNING.
            //
            // This warned and carried on, and the CPU reference interpreter
            // then served the bundle: its logits are a stand-in, the chat path
            // falls back to a bare `role:\ncontent` flatten, and stops are
            // matched on a newline byte. The result is fluent, fast, wrong, and
            // indistinguishable from a working server unless someone reads the
            // log — which is exactly how a GLM-5.3 serve here came up on the
            // CPU backend and answered correctly at fictional speed.
            //
            // A bundle with NO blob is a genuine CPU-reference asset and still
            // warns, because for that one the CPU path is the intended path.
            let has_blob = plowrt::asset::devblob::DevBlob::find_in_dir(dir)
                .ok()
                .flatten()
                .is_some();
            if has_blob {
                return Err(format!(
                    "{}: this bundle carries a compiled device blob for {target}, but no \
                     matching GPU driver was found. Serving it would fall back to the CPU \
                     reference interpreter, which produces fluent WRONG output at fictional \
                     speed. Refusing to start.",
                    dir.display()
                )
                .into());
            }
            tracing::warn!(
                dir = %dir.display(), %target,
                "loaded model bundle — no matching GPU driver; \
                 running on the CPU reference interpreter (unaccelerated)"
            );
        }
    }
    tracing::info!(models = registry.len(), trace, "registry ready");

    let state = Arc::new(AppState::with_trace(registry, execset, trace));

    // Where a control-plane load may take an assets dir from. Explicit
    // `--models-root` entries, plus the parents of the dirs this process was
    // started with — so the common case (serve two of the bundles that already
    // live side by side) needs no extra flag, while an arbitrary path in a
    // request body stays refused. `serve::admin` canonicalizes both sides
    // before the prefix test.
    let mut roots: Vec<PathBuf> = RuntimeConfig::get()
        .models_root
        .iter()
        .map(PathBuf::from)
        .collect();
    roots.extend(
        assets
            .iter()
            .filter_map(|a| a.parent().map(std::path::Path::to_path_buf)),
    );
    state.install_models_roots(roots);

    // GPU-managed models: any bundle whose assets dir carries a PLOWDEV device
    // blob goes under an S1 residency manager, which plans its VRAM footprint
    // from the blob header, loads the subset that fits (co-residency) and
    // switches the rest on demand. Checkpoint dir is `<assets>/checkpoint`
    // (`--rt-checkpoint` / PLOW_CHECKPOINT overrides); the initial loads are
    // the slow part of startup (a 12B checkpoint is ~22 GiB of H2D) and happen
    // before the listeners open. `--vram-budget-mib` caps the planner's view of
    // a card (A/B, tests).
    //
    // PLACEMENT decides which device each model lands on, and it runs entirely
    // on blob headers — nothing is loaded to find out where it goes.
    // `--devices` picks the visible ordinals (default: probe upward until one
    // fails to open), `--place` picks the policy, and one manager is built per
    // group that actually receives a model.
    #[cfg(feature = "cuda")]
    let mut managed_slugs: std::collections::HashSet<String> = std::collections::HashSet::new();
    #[cfg(feature = "cuda")]
    if let Some(cuda) = &cuda {
        use plowrt::memory::vmm::VmmOps as _;
        use plowrt::serve::placement::{self, ModelSpec, Place};

        let mut models: Vec<(String, PathBuf, PathBuf)> = Vec::new();
        let slugs: Vec<String> = state.registry.slugs();
        for slug in slugs {
            let bundle = state.registry.get(&slug)?;
            if plowrt::asset::devblob::DevBlob::find_in_dir(&bundle.dir)?.is_none() {
                continue;
            }
            let ckpt = RuntimeConfig::get()
                .checkpoint
                .clone()
                .map(PathBuf::from)
                .unwrap_or_else(|| bundle.dir.join("checkpoint"));
            managed_slugs.insert(slug.clone());
            models.push((slug, bundle.dir.clone(), ckpt));
        }
        // Keep CLI registration order (registry iteration is sorted by slug).
        models.sort_by_key(|(_, dir, _)| assets.iter().position(|a| a == dir));

        if !models.is_empty() {
            let budget = RuntimeConfig::get().nv.vram_budget_mib.map(|mib| mib << 20);
            let policy: Place = RuntimeConfig::get().place;

            // Per-model footprint and TP degree, both from the blob header.
            let granularity = cuda.granularity()?;
            let mut specs: Vec<ModelSpec> = Vec::with_capacity(models.len());
            for (slug, dir, _) in &models {
                let plan = plowrt::serve::manager::BlobPlan::from_dir_with_granularity(
                    dir,
                    Some(granularity),
                )?;
                specs.push(ModelSpec {
                    slug: slug.clone(),
                    tp: plowrt::serve::manager::tp_degree(dir)?,
                    required: plan.tensor_total()
                        + plowrt::serve::manager::DEFAULT_OVERHEAD
                        + plowrt::serve::manager::RESERVE,
                    device: None,
                });
            }

            // `--pin slug@ordinal`. A pin naming a model this server does not
            // serve is an error: it is almost always a typo, and honouring the
            // rest of the pins while dropping that one places a model somewhere
            // the operator did not ask for.
            for (slug, ordinal) in placement::parse_pins(&RuntimeConfig::get().pin)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?
            {
                match specs.iter_mut().find(|s| s.slug == slug) {
                    Some(spec) => spec.device = Some(ordinal),
                    None => {
                        return Err(format!(
                            "--pin {slug}@{ordinal} names a model this server does not serve"
                        )
                        .into())
                    }
                }
            }

            // Visible ordinals.
            //
            // `CUDA_VISIBLE_DEVICES` is applied by libcuda at `cuInit`, so the
            // count below and every ordinal plowrt uses are ALREADY indices
            // into the masked set — `--devices 1` means the second visible GPU,
            // not physical GPU 1. That is the vendor's numbering and matching
            // it is the point; the startup log prints both so the mapping is
            // never something an operator has to infer.
            let visible_count = cuda.device_count()?;
            let mask = device::visibility::describe_env();
            if !mask.is_empty() {
                tracing::info!(
                    visible_devices = ?mask,
                    visible_count,
                    "visible-device mask in force; every ordinal below indexes the MASKED set"
                );
            }

            let configured = RuntimeConfig::get().devices.clone();
            // Validate against the visible set BEFORE opening anything, so an
            // out-of-range ordinal reads as what it is rather than as a device
            // that would not initialise.
            if let Some(&bad) = configured.iter().find(|&&d| d >= visible_count) {
                let hint = if mask.is_empty() {
                    String::new()
                } else {
                    format!(" (a visible-device mask is in force: {mask:?})")
                };
                return Err(format!(
                    "--devices names device {bad}, but only {visible_count} device(s) are \
                     visible to this process{hint}"
                )
                .into());
            }

            // STRICTLY the requested set: `--devices 1` must not serve on GPU
            // 0. The already-open probe backend is reused only if its ordinal
            // is actually in that set.
            let probe_ordinal = u32::from(cuda.device_ordinal);
            let wanted = placement::requested_ordinals(&configured, visible_count);
            let mut backends: Vec<(u32, Arc<device::cuda::CudaBackend>)> =
                Vec::with_capacity(wanted.len());
            for d in wanted {
                if d == probe_ordinal {
                    backends.push((d, Arc::clone(cuda)));
                    continue;
                }
                match device::cuda::CudaBackend::new(d as u8) {
                    Ok(b) => backends.push((d, Arc::new(b))),
                    // Enumerated but not usable. Not a "that was the last one"
                    // break any more: the count came from the driver, so this
                    // is a real failure on a device we were told exists.
                    Err(e) => {
                        return Err(format!(
                            "device {d} of {visible_count} visible will not open: {e}"
                        )
                        .into())
                    }
                }
            }
            let visible: Vec<u32> = backends.iter().map(|(d, _)| *d).collect();

            let width = placement::grouping_width(&specs, visible.len())
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
            // Capacity is filled in per group below — a node's cards are not
            // necessarily the same size, and using device 0's total for all of
            // them either rejects a model that fits or places one on a card too
            // small for it and fails at load.
            let mut groups = placement::plan_groups(&visible, width, 0)
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
            for group in &mut groups {
                let mut total_bytes = u64::MAX;
                for ord in &group.ordinals {
                    let be = backends
                        .iter()
                        .find(|(d, _)| d == ord)
                        .map(|(_, b)| b)
                        .expect("group ordinal came from the opened set");
                    // A TP group is only as large as its smallest member.
                    total_bytes = total_bytes.min(be.mem_info()?.1);
                }
                group.capacity = budget.map(|b| b.min(total_bytes)).unwrap_or(total_bytes);
            }
            let groups = groups;
            let layout = placement::assign(&specs, &groups, policy)
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;

            // One manager per group that received a model. A group's manager
            // owns that group's backend, free-VRAM view, slab pool, LRU order
            // and switch lock, so a load on one GPU never serializes behind a
            // switch on another.
            let mut managers = Vec::with_capacity(layout.groups.len());
            for (g, group) in layout.groups.iter().enumerate() {
                let members = layout.members(&specs, g);
                let mine: Vec<(String, PathBuf, PathBuf)> = models
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| layout.assignment[*i] == g)
                    .map(|(_, m)| m.clone())
                    .collect();
                // A group with no initial model still gets a manager. Skipping
                // it left an idle GPU with nothing to address, so a later
                // `POST /v1/models/load {"device": 1}` answered "no device group
                // starts at ordinal 1" on a node with a perfectly free card.
                tracing::info!(
                    group = g,
                    devices = ?group.ordinals,
                    models = ?members,
                    capacity_gib = group.capacity as f64 / (1u64 << 30) as f64,
                    ?policy,
                    "placement: group ready"
                );
                let be = backends
                    .iter()
                    .find(|(d, _)| *d == group.first())
                    .map(|(_, b)| Arc::clone(b))
                    .expect("group ordinal came from the opened set");
                let mgr = Arc::new(plowrt::serve::manager::ModelManager::new(
                    be, &state, mux_cfg, mine, budget,
                )?);
                let index = managers.len();
                for slug in members {
                    state.set_slug_group(slug, index);
                }
                managers.push(mgr);
            }
            state.install_managers(managers.clone());
            // One co-tenant turn per group, installed before `load_initial`
            // spawns the first dispatcher — a mux resolves its turn at spawn.
            state.install_device_turns(layout.groups.len());
            // Load each group's initial residents. Sequential: the loads are
            // H2D-bound and share host bandwidth, so overlapping them buys
            // little while making a failure harder to attribute.
            for mgr in &managers {
                mgr.load_initial().await?;
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    let managed_slugs: std::collections::HashSet<String> = std::collections::HashSet::new();

    // AMD/gfx950 engines. Deliberately NOT under the S1 `ModelManager`: that is
    // the multi-model residency planner (VRAM planning, co-residency, evict-LRU)
    // and it is CUDA-only. Every bundle is loaded once and stays up for the life
    // of the process, so the install is a straight loop here.
    //
    // Note that this loop DOES install an engine per bundle, and each opens
    // ordinal 0 — so two AMD models genuinely do share one agent, with no
    // planner accounting for it. `--co-sched rr` is the only ordering available
    // to them until the manager grows an AMD backend.
    //
    // A bundle qualifies exactly as on the CUDA side: its assets dir carries a
    // PLOWDEV blob. It additionally needs the gfx950 code objects, whose dir is
    // `--rt-hsaco` / `PLOW_HSACO` or `<assets>/hsaco`; the TP degree is read off
    // the packet.
    #[cfg(feature = "hsa")]
    if vendor == Some(hwspec::Vendor::Amd) {
        let slugs: Vec<String> = state.registry.slugs();
        // Count the bundles that will actually reach the AMD engine. The loop
        // below skips any without a PLOWDEV blob, so counting every registered
        // slug refused a serve that pairs one AMD bundle with a CPU-reference
        // one — two models registered, but only ever one on the agent.
        let mut co_resident = 0usize;
        for slug in &slugs {
            let bundle = state.registry.get(slug)?;
            if plowrt::asset::devblob::DevBlob::find_in_dir(&bundle.dir)?.is_some() {
                co_resident += 1;
            }
        }
        if co_resident > 1 && cfg.co_sched != plowrt::serve::cosched::CoSched::Rr {
            return Err("AMD co-resident models require --co-sched rr; separate HSA queues do not guarantee whole-grid residency".into());
        }
        for slug in slugs {
            let bundle = state.registry.get(&slug)?;
            let Some(blob) = plowrt::asset::devblob::DevBlob::find_in_dir(&bundle.dir)? else {
                continue;
            };
            // Same refusal the CUDA path makes (`serve::manager::load_model`):
            // a real model driven through the byte-fallback tokenizer produces
            // fluent-looking GARBAGE, not an error, because the ids bear no
            // relation to the checkpoint's vocab. Refuse loudly instead.
            if bundle.tokenizer().is_byte_fallback() {
                return Err(format!(
                    "{slug}: the AMD engine requires a real tokenizer.json in {}",
                    bundle.dir.display()
                )
                .into());
            }
            let hsaco = RuntimeConfig::get()
                .amd
                .hsaco
                .clone()
                .map(PathBuf::from)
                .unwrap_or_else(|| bundle.dir.join("hsaco"));
            let ckpt = RuntimeConfig::get()
                .checkpoint
                .clone()
                .map(PathBuf::from)
                .unwrap_or_else(|| bundle.dir.join("checkpoint"));
            tracing::info!(
                %slug, blob = %blob.display(), hsaco = %hsaco.display(),
                checkpoint = %ckpt.display(), "loading AMD engine"
            );
            let t0 = std::time::Instant::now();
            let eng = plowrt::serve::engine::AmdServe::load(&blob, &hsaco, Some(&ckpt))?;
            tracing::info!(
                %slug, secs = t0.elapsed().as_secs_f64(), max_ctx = eng.max_ctx(),
                "AMD engine loaded"
            );
            state.install_gpu_engine(slug, plowrt::serve::engine::ServeEngine::Amd(eng));
        }
    }

    // No GPU driver: bundles that ship a device blob are served by the CPU
    // engine (persistent pinned workers + C kernels), not the reference
    // interpreter. Same tokenizer refusal as the GPU paths.
    #[cfg(feature = "cpu")]
    if vendor.is_none() {
        let slugs: Vec<String> = state.registry.slugs();
        for slug in slugs {
            let bundle = state.registry.get(&slug)?;
            let Some(blob) = plowrt::asset::devblob::DevBlob::find_in_dir(&bundle.dir)? else {
                continue;
            };
            if bundle.tokenizer().is_byte_fallback() {
                return Err(format!(
                    "{slug}: the CPU engine requires a real tokenizer.json in {}",
                    bundle.dir.display()
                )
                .into());
            }
            let ckpt = RuntimeConfig::get()
                .checkpoint
                .clone()
                .map(PathBuf::from)
                .unwrap_or_else(|| bundle.dir.join("checkpoint"));
            let cpu = &RuntimeConfig::get().cpu;
            let opts = plowrt::exec::cpu::engine::CpuEngineOpts {
                threads: cpu.threads as usize,
                numa: cpu.numa.clone(),
                isa: match cpu.isa {
                    plowrt::config::CpuIsa::Scalar => plowrt::exec::cpu::ffi::Isa::Scalar,
                    plowrt::config::CpuIsa::Avx512 => plowrt::exec::cpu::ffi::Isa::Avx512,
                    plowrt::config::CpuIsa::Amx | plowrt::config::CpuIsa::Auto => {
                        plowrt::exec::cpu::ffi::Isa::Amx
                    }
                },
                spin_us: cpu.spin_us,
            };
            tracing::info!(
                %slug, blob = %blob.display(), checkpoint = %ckpt.display(), ?opts,
                "loading CPU engine"
            );
            let t0 = std::time::Instant::now();
            let eng = plowrt::serve::engine::CpuServe::load(&blob, &ckpt, &opts)?;
            tracing::info!(
                %slug, secs = t0.elapsed().as_secs_f64(), max_ctx = eng.max_ctx(),
                "CPU engine loaded"
            );
            state.install_gpu_engine(slug, plowrt::serve::engine::ServeEngine::Cpu(eng));
        }
    }

    // Backends other than the CUDA placement path serve ONE device set, so one
    // turn covers it. A no-op when that path already installed the real group
    // count. This is not CUDA-only for a reason: the AMD loop above installs an
    // engine per bundle and each opens the same ROCr agent, so two AMD models
    // on one card is a shape that already exists — and HSA has no
    // cooperative-launch refusal to turn the resulting CU oversubscription into
    // an error rather than a hang.
    state.install_device_turns(1);

    // Spawn a per-model dispatcher: bucket-mux + arrival-rate batch formation.
    // Each dispatcher owns a Sender clone via AppState::mux(slug). Managed
    // (GPU) models are skipped — their dispatcher lifecycle belongs to the
    // manager (spawned on load, drained+removed on evict).
    let slugs: Vec<String> = state.registry.slugs();
    for slug in slugs {
        if managed_slugs.contains(&slug) {
            continue;
        }
        let bundle = state.registry.get(&slug)?;
        let m = mux::spawn(slug.clone(), bundle, Arc::clone(&state), mux_cfg);
        state.install_mux(slug, m);
    }

    Ok(state)
}

async fn bench(
    assets: PathBuf,
    prompt_ids: Option<String>,
    prompt_rows: Option<PathBuf>,
    random_input_len: usize,
    prefill_sweep: bool,
    prefill_lengths: Option<String>,
    prefill_reps: usize,
    prefill_warmups: usize,
    seed: u64,
    concurrency: usize,
    requests: usize,
    warmup_requests: usize,
    output_len: usize,
    executors: u32,
    mux_cfg: MuxConfig,
    engine_diagnostics: bool,
    parity_report: bool,
    token_audit: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = match (prompt_ids.as_deref(), prompt_rows.as_deref()) {
        (Some(raw), None) => plowrt::serve::bench::Input::TokenIds(parse_token_ids(raw)?),
        (None, Some(path)) => {
            plowrt::serve::bench::Input::TokenRows(plowrt::serve::bench::read_prompt_rows(path)?)
        }
        (None, None) => plowrt::serve::bench::Input::Random {
            len: random_input_len,
            seed,
        },
        (Some(_), Some(_)) => return Err("--prompt-ids conflicts with --prompt-rows".into()),
    };
    validate_parity_report_options(
        parity_report,
        matches!(
            &input,
            plowrt::serve::bench::Input::TokenIds(_) | plowrt::serve::bench::Input::Random { .. }
        ),
        concurrency,
        requests,
        warmup_requests,
    )?;
    plowrt::serve::bench::validate_request_layout(&input, warmup_requests, requests)?;
    validate_token_audit_options(token_audit, &input, warmup_requests, requests, output_len)?;
    let state = bringup_runtime(vec![assets], executors, false, mux_cfg).await?;
    let models = state.registry.slugs();
    let [model] = models.as_slice() else {
        return Err(format!("bench requires exactly one model, loaded {}", models.len()).into());
    };
    let runtime = serde_json::json!({
        "config": format!("{:#?}", RuntimeConfig::global()),
        "environment": runtime_environment(),
        "features": {
            "cuda": cfg!(feature = "cuda"),
            "hsa": cfg!(feature = "hsa"),
            "hf_tokenizer": cfg!(feature = "hf-tokenizer"),
        },
    });
    if engine_diagnostics {
        plowrt::serve::bench::begin_engine_diagnostics(&state, model)?;
    }
    let result = if prefill_sweep {
        let inputs = match prefill_lengths {
            Some(raw) => parse_prefill_lengths(&raw)?
                .into_iter()
                .enumerate()
                .map(|(i, len)| plowrt::serve::bench::Input::Random {
                    len,
                    seed: seed.wrapping_add(i as u64),
                })
                .collect(),
            None => vec![input],
        };
        plowrt::serve::bench::run_prefill_sweep(
            &state,
            plowrt::serve::bench::PrefillSweepConfig {
                model: model.clone(),
                inputs,
                warmup_requests: prefill_warmups,
                repetitions: prefill_reps,
                runtime,
            },
        )
        .await
        .and_then(|report| {
            serde_json::to_value(report)
                .map_err(|e| plowrt::RuntimeError::Msg(format!("serialize prefill sweep: {e}")))
        })
    } else {
        plowrt::serve::bench::run(
            &state,
            plowrt::serve::bench::Config {
                model: model.clone(),
                input,
                concurrency,
                warmup_requests,
                requests,
                output_tokens: output_len,
                runtime,
                parity_report,
                token_audit,
            },
        )
        .await
        .and_then(|report| {
            serde_json::to_value(report)
                .map_err(|e| plowrt::RuntimeError::Msg(format!("serialize bench report: {e}")))
        })
    };
    if let Some(mux) = state.mux(model) {
        mux.drain().await;
    }
    let diagnostics =
        engine_diagnostics.then(|| plowrt::serve::bench::finish_engine_diagnostics(&state, model));
    #[cfg(feature = "hsa")]
    let trace_result = RuntimeConfig::get().amd.trace_raw.as_ref().map(|path| {
        let path = PathBuf::from(path);
        plowrt::serve::bench::write_amd_packet_trace(&state, model, &path).map(|()| {
            tracing::info!(path = %path.display(), "raw AMD packet trace written");
        })
    });
    let mut report = result?;
    if let Some(diagnostics) = diagnostics {
        let diagnostics = diagnostics?;
        let report_object = report.as_object_mut().ok_or_else(|| {
            plowrt::RuntimeError::Msg("bench report serialization did not produce an object".into())
        })?;
        report_object.insert("diagnostics".into(), serde_json::to_value(diagnostics)?);
    }
    #[cfg(feature = "hsa")]
    if let Some(trace_result) = trace_result {
        trace_result?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

const MAX_TOKEN_AUDIT_REQUESTS: usize = 64;
const MAX_TOKEN_AUDIT_IDS: usize = 65_536;

fn validate_token_audit_options(
    token_audit: bool,
    input: &plowrt::serve::bench::Input,
    warmup_requests: usize,
    requests: usize,
    output_tokens: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if !token_audit {
        return Ok(());
    }
    let prompt_ids = match input {
        plowrt::serve::bench::Input::TokenIds(ids) => ids.len().checked_mul(requests),
        plowrt::serve::bench::Input::TokenRows(rows) => rows
            .get(warmup_requests..)
            .ok_or("--token-audit prompt row layout is incomplete")?
            .iter()
            .try_fold(0usize, |total, row| total.checked_add(row.len())),
        plowrt::serve::bench::Input::Random { .. } => {
            return Err("--token-audit requires --prompt-ids or --prompt-rows".into())
        }
    };
    let total_ids = output_tokens
        .checked_mul(requests)
        .and_then(|outputs| prompt_ids.and_then(|prompts| prompts.checked_add(outputs)))
        .ok_or("--token-audit token count overflow")?;
    if requests == 0 || requests > MAX_TOKEN_AUDIT_REQUESTS || total_ids > MAX_TOKEN_AUDIT_IDS {
        return Err(format!(
            "--token-audit is bounded to {MAX_TOKEN_AUDIT_REQUESTS} requests and \
             {MAX_TOKEN_AUDIT_IDS} total prompt/output token IDs"
        )
        .into());
    }
    Ok(())
}

fn validate_parity_report_options(
    parity_report: bool,
    has_reproducible_input: bool,
    concurrency: usize,
    requests: usize,
    warmup_requests: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if parity_report
        && (!has_reproducible_input || concurrency != 1 || requests != 1 || warmup_requests != 0)
    {
        return Err(
            "--parity-report requires --prompt-ids or --random-input-len, --concurrency 1, \
             --requests 1, and --warmup-requests 0"
                .into(),
        );
    }
    Ok(())
}

fn parse_token_ids(raw: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let ids = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        return Err("--prompt-ids must contain at least one token id".into());
    }
    Ok(ids)
}

fn parse_prefill_lengths(raw: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let lengths = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if lengths.is_empty() || lengths.contains(&0) {
        return Err("--prefill-lengths must contain positive lengths".into());
    }
    Ok(lengths)
}

async fn serve(
    assets: Vec<PathBuf>,
    port: u16,
    socket: Option<PathBuf>,
    executors: u32,
    trace: bool,
    mux_cfg: MuxConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = bringup_runtime(assets, executors, trace, mux_cfg).await?;

    let router = app(Arc::clone(&state));

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
    // `serve` accepts only TcpListener). Also exposes privileged model control.
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
            std::fs::set_permissions(&path, perm)?;
        }
        tracing::info!(socket = %path.display(), "plowrt serving OpenAI API over UDS");
        let uds_router = router.clone().merge(plowrt::serve::admin_app(state));
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

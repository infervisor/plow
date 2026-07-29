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
    /// Runs the REAL schedule through the real production code objects. Weights
    /// are allocated but NOT bound, so the tokens are meaningless and the
    /// timing is not — every instruction, counter gate, and memory access the
    /// decode program performs happens at full size. That makes this a latency
    /// and bring-up instrument, and explicitly not a correctness one.
    AmdBench {
        /// Compiled device blob (`model.pkt`).
        #[arg(long)]
        blob: PathBuf,
        /// Directory of gfx950 code objects (`interp_*.elf`).
        #[arg(long)]
        hsaco: PathBuf,
        /// Safetensors checkpoint. Omit to run with UNBOUND weights: the
        /// schedule and the timing are real, the tokens are not.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
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
        #[arg(long, default_value_t = 32)]
        steps: u32,
        /// Context position the decode steps run at — the KV the attention
        /// actually reads, so it dominates the number.
        #[arg(long, default_value_t = 1024)]
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
        #[arg(long, default_value_t = 1)]
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
        #[arg(long)]
        hsaco: PathBuf,
        #[arg(long)]
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
            prompt,
            steps,
            ctx,
            batched,
            tp,
            dump_logits,
        } => {
            if tp > 1 {
                amd_bench_tp(blob, hsaco, checkpoint, prompt, steps, ctx, tp, dump_logits)
            } else {
                amd_bench(blob, hsaco, checkpoint, prompt, steps, ctx, batched)
            }
        }
        #[cfg(not(feature = "hsa"))]
        Cmd::AmdBench { .. } => Err("plowrt was built without --features hsa".into()),
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

/// Bring the AMD engine up and time decode steps.
#[cfg(feature = "hsa")]
fn amd_bench(
    blob: PathBuf,
    hsaco: PathBuf,
    checkpoint: Option<PathBuf>,
    prompt: Option<String>,
    steps: u32,
    ctx: u32,
    batched: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::amd::AmdEngine;

    let be = Arc::new(plowrt::device::hsa::HsaBackend::new(0)?);
    let t0 = std::time::Instant::now();
    let mut eng = AmdEngine::load(Arc::clone(&be), &blob, &hsaco, checkpoint.as_deref())?;
    println!(
        "loaded in {:.1} s: arch={} programs={} max_ctx={} schedulers={:?}",
        t0.elapsed().as_secs_f64(),
        eng.arch(),
        eng.n_programs(),
        eng.max_ctx(),
        eng.schedulers(),
    );
    for p in 0..eng.n_programs() {
        println!("  program {p}: T={} segments={}", eng.prog_t(p), eng.prog_segments(p));
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
                println!("  slot {s} chain ({} prompt tokens): {:?}",
                    prompts[s % prompts.len()].len(), chains[s]);
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
                let first = (0..s).find(|&r| prompts[r % prompts.len()] == prompts[s % prompts.len()]);
                if let Some(r) = first {
                    if chains[s] != chains[r] {
                        verdict = false;
                        println!("  slot {s} and slot {r} share a prompt and DIFFER");
                    }
                }
            }
            println!(
                "  same-prompt slots agree: {}",
                if verdict { "YES" } else { "NO  <-- per-sequence KV rows are wrong" }
            );
            println!(
                "  cross-check each DIFFERENT prompt against a batch-1 run of the same ids"
            );
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
            "\n{steps} dispatches x batch {b} at ctx={ctx}:\n  \
             tpot {ms:.3} ms  |  aggregate {:.1} tok/s  |  per-dispatch {ms:.3} ms",
            b as f64 * 1e3 / ms
        );
        if !eng.weights_bound() {
            println!("  (weights unbound — timing real, ids are not)");
        }
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
        // A multi-token prompt goes through PREFILL, which populates the KV for
        // [0, n) and leaves the first sampled token in in.ids. A single token
        // needs none: decode at position 0 writes KV row 0 and attends over
        // exactly [0,1), so nothing is read that was not written.
        let (first, mut pos) = if ids.len() > 1 {
            let t0 = std::time::Instant::now();
            let tok = eng.prefill(&ids)?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "\nprefill: {} tokens in {ms:.1} ms ({:.0} tok/s)",
                ids.len(),
                ids.len() as f64 / (ms / 1e3)
            );
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
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            // in.ids is NOT re-seeded: the device wrote the previous step's
            // sampled token there itself, which is what this step embeds.
            out.push(eng.decode_step(pos, pos + 1)?);
            pos += 1;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
        println!("  {out:?}");
        println!("  {steps} decode steps: {ms:.3} ms/token ({:.1} tok/s)", 1e3 / ms);
        if !eng.weights_bound() {
            println!("  (weights unbound — these ids are noise)");
        }
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
        last = eng.decode_step(ctx + 1 + i, ctx + 2 + i)?;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
    println!(
        "\n{steps} decode steps at ctx={ctx}: {ms:.3} ms/token ({:.1} tok/s), last id {last}",
        1e3 / ms
    );
    println!(
        "  dispatch accounting: {} launches, enqueue {:.1} us, drain {:.1} us",
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
            "\nWEIGHTS ARE NOT BOUND — the token ids are meaningless. The timing is \n\
             not: every instruction, counter gate and memory access of the real \n\
             decode program ran at full size."
        );
    }
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
#[cfg(feature = "hsa")]
fn amd_bench_tp(
    blob: PathBuf,
    hsaco: PathBuf,
    checkpoint: Option<PathBuf>,
    prompt: Option<String>,
    steps: u32,
    ctx: u32,
    tp: u32,
    dump_logits: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::amd_tp::AmdTpGroup;

    let mut backends = Vec::with_capacity(tp as usize);
    for d in 0..tp {
        backends.push(Arc::new(plowrt::device::hsa::HsaBackend::new(d as u8)?));
    }
    let t0 = std::time::Instant::now();
    let mut g = AmdTpGroup::load(backends, &blob, &hsaco, checkpoint.as_deref())?;
    println!(
        "loaded in {:.1} s: TP={} ranks, max_ctx={}",
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
        let Some(dir) = &dump_logits else { return Ok(()) };
        std::fs::create_dir_all(dir)?;
        let n = g.rank(0).tensor_bytes("act.logits").ok_or("no act.logits")? as usize;
        let mut buf = vec![0u8; n];
        g.rank(0).read_tensor("act.logits", &mut buf)?;
        std::fs::write(dir.join(format!("logits_{tag}.bin")), &buf)?;
        Ok(())
    };

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
            let tok = if g.rank(0).n_programs() == 1 {
                let mut last = 0;
                for (p, id) in ids.iter().enumerate() {
                    g.seed_ids(&[*id])?;
                    last = AmdTpGroup::agree(&g.decode_step(p as u32, p as u32 + 1)?)?;
                }
                last
            } else {
                AmdTpGroup::agree(&g.prefill(&ids)?)?
            };
            dump(&g, "prefill")?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "\nprefill: {} tokens in {ms:.1} ms ({:.0} tok/s) -> {tok} \
                 (all {} ranks agree)",
                ids.len(),
                ids.len() as f64 / (ms / 1e3),
                g.n_gpu()
            );
            pos = ids.len() as u32;
        } else {
            g.seed_ids(&ids)?;
            pos = 0;
        }

        println!("greedy decode:");
        let mut out = Vec::new();
        let t = std::time::Instant::now();
        for s in 0..steps {
            let ids = g.decode_step(pos, pos + 1)?;
            out.push(AmdTpGroup::agree(&ids)?);
            dump(&g, &format!("{s:03}"))?;
            pos += 1;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
        println!("  {out:?}");
        println!(
            "  {steps} decode steps: {ms:.3} ms/token ({:.1} tok/s), all {} ranks \
             token-identical",
            1e3 / ms,
            g.n_gpu()
        );
        if !g.weights_bound() {
            println!("  (weights unbound — these ids are noise)");
        }
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
        let ids = g.decode_step(ctx + 1 + i, ctx + 2 + i)?;
        if AmdTpGroup::agree(&ids).is_err() {
            disagreements += 1;
        }
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
    println!(
        "\n{steps} decode steps at ctx={ctx}, TP={}: {ms:.3} ms/token ({:.1} tok/s)",
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
             Pass --checkpoint for the real token-identity check."
        );
    }
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
    println!("\n{:<16} {:>12} {:>10} {:>14}", "tensor", "non-zero", "%", "sum|x| (bf16)");
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
            .map(|c| {
                f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).abs() as f64
            })
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
        println!("\nwrote raw tensors to {} — diff two precisions byte-wise", dir.display());
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
        return Err(format!("--tp {n} but only {} devices are visible \
             (AMD: the visible set is ROCR_VISIBLE_DEVICES, not HIP_VISIBLE_DEVICES)",
            all.len())
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
        if prefill { "PREFILL two-shot" } else { "DECODE one-shot" },
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

    // AMD/gfx950 engines. Deliberately NOT under the S1 `ModelManager`: that is
    // the multi-model residency planner (VRAM planning, co-residency, evict-LRU)
    // and it is CUDA-only. An AMD serve is one model, loaded once, up for the
    // life of the process — so the install is a straight loop here.
    //
    // A bundle qualifies exactly as on the CUDA side: its assets dir carries a
    // PLOWDEV blob. It additionally needs the gfx950 code objects, whose dir is
    // `PLOW_HSACO` or `<assets>/hsaco`; the TP degree is read off the packet.
    #[cfg(feature = "hsa")]
    if vendor == Some(hwspec::Vendor::Amd) {
        let slugs: Vec<String> = state.registry.slugs().map(str::to_string).collect();
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
            let hsaco = std::env::var("PLOW_HSACO")
                .map(PathBuf::from)
                .unwrap_or_else(|_| bundle.dir.join("hsaco"));
            let ckpt = std::env::var("PLOW_CHECKPOINT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| bundle.dir.join("checkpoint"));
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

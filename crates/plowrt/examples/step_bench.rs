//! Kernel-only decode-step control for the serving-overhead audit
//! (campaign S5-serve-tuning): drives `GpuEngine` directly — no HTTP, no mux,
//! no spawn_blocking, no SSE — so `served TPOT − this` isolates the serving
//! layer from the kernel at any batch. Same engine code the server runs
//! (`prefill_slot` to build ctx, then one `step_slots` per token).
//!
//! Usage:
//!   step_bench <assets_dir> [slots] [ctx] [steps] [--same] [--warmup N] [--multistep]
//!              [--dump-tensors name,name --dump-dir dir]
//! `--same` feeds every slot the SAME prompt and reports how many slots' greedy
//! streams agree with slot 0 (a within-batch consistency check). `--dump-tensors`
//! writes the named tensors raw after the last step (block_run's format), which
//! with `PLOW_DEBUG_MAX_INST` truncation gives per-layer activations. `--multistep` times the
//! engine's device multi-step quanta (`PLOW_MULTISTEP=K`) instead of single steps; the digest
//! covers the same tokens in the same order, so it compares directly with a single-step run.
//! Env: PLOW_CHECKPOINT (default <assets>/checkpoint), PLOW_STEP_TIME=1 for
//! the engine's host-op breakdown.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("step_bench requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::time::Instant;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let assets = std::path::PathBuf::from(
        args.next()
            .ok_or("usage: step_bench <assets> [slots] [ctx] [steps] [--same] [--warmup N] [--multistep] [--dump-tensors a,b --dump-dir d]")?,
    );
    let want_slots: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(1);
    let ctx: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(4137);
    let steps: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(128);
    let mut same = false;
    let mut multistep = false;
    let mut warmup = 16usize;
    let (mut dump_names, mut dump_dir) = (None::<String>, None::<String>);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--same" => same = true,
            "--multistep" => multistep = true,
            "--warmup" => warmup = args.next().ok_or("--warmup N")?.parse()?,
            "--dump-tensors" => dump_names = Some(args.next().ok_or("--dump-tensors a,b")?),
            "--dump-dir" => dump_dir = Some(args.next().ok_or("--dump-dir d")?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let dumps = match (dump_names, dump_dir) {
        (None, None) => None,
        (Some(n), Some(d)) => Some((n, std::path::PathBuf::from(d))),
        _ => return Err("--dump-tensors and --dump-dir must be provided together".into()),
    };

    let ckpt = std::env::var("PLOW_CHECKPOINT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| assets.join("checkpoint"));
    let be = Arc::new(plowrt::device::cuda::CudaBackend::new(0)?);
    let mut e = plowrt::exec::gpu::GpuEngine::load(be, &assets, &ckpt)?;
    let slots = want_slots.min(e.batch());
    println!(
        "engine batch={} vocab={} max_ctx={} prefill={} -> slots={slots} ctx={ctx} steps={steps}",
        e.batch(),
        e.vocab(),
        e.max_ctx(),
        e.has_prefill()
    );

    // Synthetic prompts, ids in-vocab. One PER SLOT: identical rows route to the same top-k
    // experts, so a MoE model's B>1 step touched 8 experts where serving touches ~60 (slot 0
    // keeps the historical prompt, so B=1 numbers are unchanged). `--same` gives every slot
    // slot 0's prompt on purpose.
    let mut last = vec![0u32; slots];
    for b in 0..slots {
        let bb = if same { 0 } else { b as u32 };
        let prompt: Vec<u32> = (0..ctx as u32)
            .map(|i| 100 + ((i + 131 * bb) % 1000))
            .collect();
        e.begin_slot(b, ctx + steps + 1)?;
        let t0 = Instant::now();
        last[b] = if e.has_prefill() {
            e.prefill_slot(b, &prompt)?
        } else {
            let mut toks = Vec::new();
            e.consume_prompt(b, &prompt, &mut toks)?
        };
        println!(
            "slot {b}: prompt consumed in {:.3} s",
            t0.elapsed().as_secs_f64()
        );
    }

    // Warmup (repo convention: discard 16), then timed steps.
    let feeds_of = |last: &[u32]| -> Vec<(usize, u32)> {
        last.iter().enumerate().map(|(b, &t)| (b, t)).collect()
    };
    let mut toks = Vec::new();
    for _ in 0..warmup {
        e.step_slots(&feeds_of(&last), &mut toks)?;
        last.copy_from_slice(&toks);
    }
    // Drop prefill + warmup from the trace so the profile is timed-decode only.
    e.trace_reset()?;
    let mut ms: Vec<f64> = Vec::with_capacity(steps);
    // Greedy stream digest (FNV-1a over every slot's tokens): two decode objects on one packet
    // can be compared for token agreement without a server.
    let mut digest: u64 = 0xcbf29ce484222325;
    let mut streams: Vec<Vec<u32>> = vec![Vec::with_capacity(steps); slots];
    if multistep && e.multistep_quantum().is_none() {
        return Err("--multistep needs an engine with PLOW_MULTISTEP >= 2".into());
    }
    let mut done = 0;
    while done < steps {
        let t0 = Instant::now();
        if multistep {
            // Fed row r is slot r; `toks` is row-major, K tokens per row.
            let k = e.multi_step_at_most(&feeds_of(&last), steps - done, &mut toks)?;
            let per = t0.elapsed().as_secs_f64() * 1e3 / k as f64;
            for step in 0..k {
                ms.push(per);
                for (b, stream) in streams.iter_mut().enumerate() {
                    let t = toks[b * k + step];
                    digest = (digest ^ t as u64).wrapping_mul(0x100000001b3);
                    stream.push(t);
                }
            }
            for (b, l) in last.iter_mut().enumerate() {
                *l = toks[b * k + k - 1];
            }
            done += k;
            continue;
        }
        e.step_slots(&feeds_of(&last), &mut toks)?;
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
        last.copy_from_slice(&toks);
        for (b, &t) in toks.iter().enumerate() {
            digest = (digest ^ t as u64).wrapping_mul(0x100000001b3);
            streams[b].push(t);
        }
        done += 1;
    }
    println!("TOK_STREAM slots={slots} ctx={ctx} fnv={digest:016x} slot0={:?}", streams[0]);
    if same {
        let agree = streams.iter().filter(|s| **s == streams[0]).count();
        println!("SLOTS_AGREE {agree}/{slots} (identical prompts on every slot)");
        for (b, s) in streams.iter().enumerate().skip(1) {
            if *s != streams[0] {
                println!("  slot{b}={s:?}");
            }
        }
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    let median = ms[ms.len() / 2];
    let sd = if ms.len() > 1 {
        (ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (ms.len() - 1) as f64).sqrt()
    } else {
        0.0
    };
    println!(
        "RAW_STEP slots={slots} ctx={ctx} n={} mean_ms={mean:.3} median_ms={median:.3} \
         sd_ms={sd:.3} min_ms={:.3} max_ms={:.3} per_user_tok_s={:.1} aggregate_tok_s={:.1}",
        ms.len(),
        ms[0],
        ms[ms.len() - 1],
        1000.0 / mean,
        1000.0 / mean * slots as f64,
    );

    if let Some((names, dir)) = dumps {
        std::fs::create_dir_all(&dir)?;
        let mut rows = Vec::new();
        for (index, name) in names.split(',').map(str::trim).enumerate() {
            let bytes = e
                .tensor_bytes(name)
                .ok_or_else(|| format!("unknown dump tensor {name:?}"))?;
            let mut raw = vec![0u8; usize::try_from(bytes)?];
            e.read_tensor(name, &mut raw)?;
            let file = format!("tensor-{index:03}.bin");
            std::fs::write(dir.join(&file), raw)?;
            rows.push(serde_json::json!({"name": name, "bytes": bytes, "file": file}));
        }
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "scope": "raw complete allocations after the last decode step",
                "slots": slots, "ctx": ctx, "steps": steps, "tensors": rows,
            }))?,
        )?;
        println!("  wrote raw tensor dumps to {}", dir.display());
    }

    // Stage-7 profile: with a -DPLOW_NV_TRACE=1 decode cubin, dump block 0's
    // per-opcode gate/body/signal cycle attribution (None on a normal cubin).
    if let Some(profile) = e.trace_summary()? {
        println!("{profile}");
    }
    Ok(())
}

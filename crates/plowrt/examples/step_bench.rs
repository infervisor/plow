//! Kernel-only decode-step control for the serving-overhead audit
//! (campaign S5-serve-tuning): drives `GpuEngine` directly — no HTTP, no mux,
//! no spawn_blocking, no SSE — so `served TPOT − this` isolates the serving
//! layer from the kernel at any batch. Same engine code the server runs
//! (`prefill_slot` to build ctx, then one `step_slots` per token).
//!
//! Usage:
//!   step_bench <assets_dir> [slots] [ctx] [steps]
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
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let assets = std::path::PathBuf::from(args.next().ok_or("usage: step_bench <assets> [slots] [ctx] [steps]")?);
    let want_slots: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(1);
    let ctx: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(4137);
    let steps: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(128);

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

    // Synthetic prompt (numerics are irrelevant to step time; ids in-vocab).
    let prompt: Vec<u32> = (0..ctx as u32).map(|i| 100 + (i % 1000)).collect();
    let mut last = vec![0u32; slots];
    for b in 0..slots {
        e.begin_slot(b, ctx + steps + 1)?;
        let t0 = Instant::now();
        last[b] = if e.has_prefill() {
            e.prefill_slot(b, &prompt)?
        } else {
            let mut tok = 0u32;
            let mut toks = Vec::new();
            for &t in &prompt {
                e.step_slots(&[(b, t)], &mut toks)?;
                tok = toks[0];
            }
            tok
        };
        println!("slot {b}: prompt consumed in {:.3} s", t0.elapsed().as_secs_f64());
    }

    // Warmup (repo convention: discard 16), then timed steps.
    let feeds_of = |last: &[u32]| -> Vec<(usize, u32)> {
        last.iter().enumerate().map(|(b, &t)| (b, t)).collect()
    };
    let mut toks = Vec::new();
    for _ in 0..16 {
        e.step_slots(&feeds_of(&last), &mut toks)?;
        last.copy_from_slice(&toks);
    }
    // Drop prefill + warmup from the trace so the profile is timed-decode only.
    e.trace_reset()?;
    let mut ms: Vec<f64> = Vec::with_capacity(steps);
    for _ in 0..steps {
        let t0 = Instant::now();
        e.step_slots(&feeds_of(&last), &mut toks)?;
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
        last.copy_from_slice(&toks);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    let median = ms[ms.len() / 2];
    let sd = (ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (ms.len() - 1) as f64).sqrt();
    println!(
        "RAW_STEP slots={slots} ctx={ctx} n={} mean_ms={mean:.3} median_ms={median:.3} \
         sd_ms={sd:.3} min_ms={:.3} max_ms={:.3} per_user_tok_s={:.1} aggregate_tok_s={:.1}",
        ms.len(),
        ms[0],
        ms[ms.len() - 1],
        1000.0 / mean,
        1000.0 / mean * slots as f64,
    );

    // Stage-7 profile: with a -DPLOW_NV_TRACE=1 decode cubin, dump block 0's
    // per-opcode gate/body/signal cycle attribution (None on a normal cubin).
    if let Some(profile) = e.trace_summary()? {
        println!("{profile}");
    }
    Ok(())
}

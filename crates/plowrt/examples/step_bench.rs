//! Kernel-only decode-step control for the serving-overhead audit
//! (campaign S5-serve-tuning): drives `GpuEngine` directly — no HTTP, no mux,
//! no spawn_blocking, no SSE — so `served TPOT − this` isolates the serving
//! layer from the kernel at any batch. Same engine code the server runs
//! (`prefill_slot` to build ctx, then one `step_slots` per token).
//!
//! Usage:
//!   step_bench <assets_dir> [slots] [ctx] [steps] [--same | --spread] [--warmup N] [--multistep] [--packed-prefill]
//!              [--dump-tensors name,name --dump-dir dir | --dump-prefill-logits dir]
//!              [--max-inst N | --max-segments N]
//! `--same` feeds every slot the SAME prompt and reports how many slots' greedy
//! streams agree with slot 0 (a within-batch consistency check). `--spread` gives each slot a
//! pseudo-random prompt over 64K ids, so a MoE model routes a decode batch the way served
//! traffic does (the default prompts share one 1000-id cycle). `--packed-prefill`
//! initializes prompts through packed request chunks and the compact terminal, printing each
//! launch's wall ms; `--pf-chunk N` caps the per-request slice below the packet's request chunk and
//! `--pf-reps N` repeats the whole packed prefill (the first pass pays graph capture).
//! `--dump-tensors`
//! writes the named tensors raw after the last step (block_run's format), which
//! with `--max-inst N`, one step and zero warmup gives partial decode activations;
//! with 0 steps and zero warmup it gives the activations right after prefill (the last layer's).
//! Instruction caps require native decode routes. `--multistep` times the
//! engine's device multi-step quanta (`PLOW_MULTISTEP=K`) instead of single steps; the digest
//! covers the same tokens in the same order, so it compares directly with a single-step run.
//! Env: PLOW_CHECKPOINT (default <assets>/checkpoint), PLOW_STEP_TIME=1 for
//! the engine's host-op breakdown.

#[cfg(any(feature = "cuda", test))]
fn record_inputs(histories: &mut [Vec<u32>], last: &[u32], outputs: &[u32], steps: usize) {
    for (row, history) in histories.iter_mut().enumerate() {
        history.push(last[row]);
        history.extend_from_slice(&outputs[row * steps..(row + 1) * steps - 1]);
    }
}

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
    let mut spread = false;
    let mut multistep = false;
    let mut packed_prefill = false;
    let mut pf_chunk = usize::MAX;
    let mut pf_reps = 1usize;
    let mut warmup = 16usize;
    let mut sweep: Option<(u32, u32)> = None;
    let mut max_inst: Option<u32> = None;
    let mut max_segments: Option<usize> = None;
    let mut seg_sweep: Option<(usize, usize, usize)> = None;
    let mut prefill_logits_dir = None::<std::path::PathBuf>;
    let mut ride: Option<(usize, Vec<usize>)> = None;
    let mut ride_dump = None::<std::path::PathBuf>;
    let (mut dump_names, mut dump_dir) = (None::<String>, None::<String>);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--max-segments" => max_segments = Some(args.next().ok_or("--max-segments N")?.parse()?),
            "--max-inst" => max_inst = Some(args.next().ok_or("--max-inst N")?.parse()?),
            "--same" => same = true,
            "--spread" => spread = true,
            "--packed-prefill" => packed_prefill = true,
            "--pf-chunk" => pf_chunk = args.next().ok_or("--pf-chunk N")?.parse()?,
            "--pf-reps" => pf_reps = args.next().ok_or("--pf-reps N")?.parse()?,
            "--multistep" => multistep = true,
            "--warmup" => warmup = args.next().ok_or("--warmup N")?.parse()?,
            // `--sweep LO..HI`: time decode steps with instruction caps LO..=HI in this process.
            "--sweep" => {
                let r = args.next().ok_or("--sweep LO..HI")?;
                let (lo, hi) = r.split_once("..").ok_or("--sweep LO..HI")?;
                sweep = Some((lo.parse()?, hi.parse()?));
            }
            // `--seg-sweep LO..HI[:STEP]`: time the widest library-routed decode rung captured up to
            // segment n for n in LO..=HI (the per-segment profile instruction caps cannot take).
            "--seg-sweep" => {
                let r = args.next().ok_or("--seg-sweep LO..HI[:STEP]")?;
                let (r, step) = r.split_once(':').unwrap_or((r.as_str(), "1"));
                let (lo, hi) = r.split_once("..").ok_or("--seg-sweep LO..HI[:STEP]")?;
                seg_sweep = Some((lo.parse()?, hi.parse()?, step.parse()?));
            }
            "--dump-tensors" => dump_names = Some(args.next().ok_or("--dump-tensors a,b")?),
            "--dump-dir" => dump_dir = Some(args.next().ok_or("--dump-dir d")?),
            "--dump-prefill-logits" => {
                prefill_logits_dir = Some(args.next().ok_or("--dump-prefill-logits dir")?.into())
            }
            "--ride" => {
                let rows = args.next().ok_or("--ride ROWS W1,W2,..")?.parse()?;
                let widths = args
                    .next()
                    .ok_or("--ride ROWS W1,W2,..")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<usize>, _>>()?;
                ride = Some((rows, widths));
            }
            "--ride-dump" => ride_dump = Some(args.next().ok_or("--ride-dump dir")?.into()),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let dumps = match (dump_names, dump_dir) {
        (None, None) => None,
        (Some(n), Some(d)) => Some((n, std::path::PathBuf::from(d))),
        _ => return Err("--dump-tensors and --dump-dir must be provided together".into()),
    };

    if max_inst.is_some() && max_segments.is_some() {
        return Err("choose --max-inst or --max-segments".into());
    }
    let partial = max_inst.is_some() || max_segments.is_some();
    if prefill_logits_dir.is_some() && (partial || dumps.is_some() || sweep.is_some() || multistep) {
        return Err("--dump-prefill-logits captures prefill only".into());
    }
    if partial
        && (steps != 1 || warmup != 0 || multistep || sweep.is_some() || dumps.is_none())
    {
        return Err("partial capture requires one step, --warmup 0, tensor dumps, and no sweep/multistep".into());
    }

    let ckpt = std::env::var("PLOW_CHECKPOINT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| assets.join("checkpoint"));
    let be = Arc::new(plowrt::device::cuda::CudaBackend::new(0)?);
    let mut e = plowrt::exec::gpu::GpuEngine::load(be, &assets, &ckpt)?;
    let slots = want_slots.min(e.batch());
    if prefill_logits_dir.is_some() && (packed_prefill || !e.has_prefill()) {
        return Err("prefill logit capture requires ordinary GPU prefill".into());
    }
    if (max_segments.is_some() || seg_sweep.is_some()) && slots != e.batch() {
        return Err("--max-segments requires the widest decode rung".into());
    }
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
    let mut histories = dumps.as_ref().map(|_| vec![Vec::new(); slots]);
    let mut prefill_cases = Vec::new();
    let mut prefill_raw = prefill_logits_dir
        .as_ref()
        .map(|dir| {
            std::fs::create_dir(dir)?;
            Ok::<_, Box<dyn std::error::Error>>(vec![0u8; e.vocab() * 2])
        })
        .transpose()?;
    let prompt_for = |b: usize| -> Vec<u32> {
        let bb = if same { 0 } else { b as u32 };
        if spread {
            let mut x = 0x9e37_79b9u32 ^ (bb + 1).wrapping_mul(0x85eb_ca6b);
            return (0..ctx)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    1000 + x % 65536
                })
                .collect();
        }
        (0..ctx as u32).map(|i| 100 + ((i + 131 * bb) % 1000)).collect()
    };
    if packed_prefill {
        use plowrt::exec::gpu::PfBatchReq;
        if slots == 0 || ctx == 0 || !e.pf_batch_enabled() || !e.has_packed_terminal() {
            return Err("--packed-prefill requires nonempty prompts, packed prefill and compact terminal".into());
        }
        let prompts: Vec<_> = (0..slots).map(prompt_for).collect();
        let chunk = e.pf_request_max_rows().min(e.pf_max_rows()).min(pf_chunk);
        if chunk == 0 {
            return Err("packed prefill has no chunk capacity".into());
        }
        let width = e.pf_max_rows() / chunk;
        let mut completed = Vec::new();
        for rep in 0..pf_reps.max(1) {
            for b in 0..slots {
                if rep > 0 {
                    e.retire_slot(b, false);
                }
                e.begin_slot(b, ctx + steps + 1)?;
                if let Some(histories) = &mut histories {
                    histories[b] = prompts[b].clone();
                }
            }
            let t0 = Instant::now();
            for c0 in (0..ctx).step_by(chunk) {
                let len = chunk.min(ctx - c0);
                for first in (0..slots).step_by(width) {
                    let end = (first + width).min(slots);
                    let requests: Vec<_> = (first..end).map(|slot| PfBatchReq {
                        slot, prompt: &prompts[slot], c0, len,
                    }).collect();
                    let t = Instant::now();
                    e.prefill_batched_complete(&requests, &mut completed)?;
                    println!("packed prefill: rep={rep} prefix={c0} chunk_rows={len} requests={} rows={} ms={:.3}",
                             requests.len(), requests.len() * len, t.elapsed().as_secs_f64() * 1e3);
                    if c0 + len == ctx && completed.len() != requests.len() {
                        return Err("packed prefill did not complete every request".into());
                    }
                    for &(slot, token) in &completed {
                        last[slot] = token;
                    }
                }
            }
            println!("packed prompts consumed in {:.4} s (rep {rep}, first tokens {:?})",
                     t0.elapsed().as_secs_f64(), &last[..slots.min(8)]);
        }
    } else {
        for b in 0..slots {
            let prompt = prompt_for(b);
            if let Some(histories) = &mut histories {
                histories[b] = prompt.clone();
            }
            e.begin_slot(b, ctx + steps + 1)?;
            let t0 = Instant::now();
            last[b] = if e.has_prefill() {
                e.prefill_slot(b, &prompt)?
            } else {
                let mut toks = Vec::new();
                e.consume_prompt(b, &prompt, &mut toks)?
            };
            if let (Some(dir), Some(raw)) = (&prefill_logits_dir, &mut prefill_raw) {
                e.read_tensor_range("act.logits", 0, raw)?;
                let file = format!("prefill-s{b:03}.bin");
                std::fs::write(dir.join(&file), raw)?;
                let prompt_bytes: Vec<u8> = prompt.iter().flat_map(|id| id.to_le_bytes()).collect();
                prefill_cases.push(serde_json::json!({
                    "id": format!("prefill-s{b:03}"), "file": file, "dtype": "bf16",
                    "prompt_token_ids": prompt, "prompt_len": ctx,
                    "prompt_sha256_u32le": plow_asset::knob::sha256_hex(&prompt_bytes),
                    "sampled_token_id": last[b], "generation_step": 0,
                    "execution_phase": "prefill_output",
                }));
            }
            println!(
                "slot {b}: prompt consumed in {:.4} s",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    if let Some(dir) = &prefill_logits_dir {
        std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "schema": 1, "producer": "plow-step-bench-prefill", "name": "step-bench-prefill",
            "vocab_size": e.vocab(), "cases": prefill_cases,
        }))?)?;
        return Ok(());
    }

    if let Some((rows, widths)) = &ride {
        return ride_bench(&mut e, &last, ctx, steps, *rows, widths, ride_dump.as_deref());
    }

    if let Some(segments) = max_segments {
        e.capture_debug_decode_prefix(segments)?;
    }
    if let Some(cap) = max_inst {
        if !e.set_debug_max_inst(cap, slots)? {
            return Err("decode object has no instruction-cap global".into());
        }
    }

    // Warmup (repo convention: discard 16), then timed steps.
    let feeds_of = |last: &[u32]| -> Vec<(usize, u32)> {
        last.iter().enumerate().map(|(b, &t)| (b, t)).collect()
    };
    let mut toks = Vec::new();
    for _ in 0..warmup {
        e.step_slots(&feeds_of(&last), &mut toks)?;
        if let Some(histories) = &mut histories {
            record_inputs(histories, &last, &toks, 1);
        }
        last.copy_from_slice(&toks);
    }
    if let Some((lo, hi, step)) = seg_sweep {
        let base_pos: Vec<usize> = (0..slots).map(|_| ctx + warmup).collect();
        let base_last = last.clone();
        for n in (lo..=hi).step_by(step.max(1)) {
            e.capture_debug_decode_prefix(n)?;
            for (b, &p) in base_pos.iter().enumerate() {
                e.rewind_slot(b, p)?;
            }
            last.copy_from_slice(&base_last);
            for _ in 0..2 {
                e.step_slots(&feeds_of(&last), &mut toks)?;
            }
            let mut v: Vec<f64> = (0..steps)
                .map(|_| {
                    let t0 = Instant::now();
                    e.step_slots(&feeds_of(&last), &mut toks).map(|_| t0.elapsed().as_secs_f64() * 1e3)
                })
                .collect::<Result<_, _>>()?;
            v.sort_by(f64::total_cmp);
            println!("{{\"seg\":{n},\"ms\":{:.4}}}", v[v.len() / 2]);
        }
        return Ok(());
    }
    if let Some((lo, hi)) = sweep {
        let base_pos: Vec<usize> = (0..slots).map(|_| ctx + warmup).collect();
        let base_last = last.clone();
        for cap in (lo..=hi).chain([u32::MAX]) {
            e.set_debug_max_inst(cap, slots)?;
            // Same kv length at every cap: a drifting position biases each delta by the
            // attention's per-token cost.
            for (b, &p) in base_pos.iter().enumerate() {
                e.rewind_slot(b, p)?;
            }
            last.copy_from_slice(&base_last);
            for _ in 0..4 {
                e.step_slots(&feeds_of(&last), &mut toks)?;
            }
            let mut v: Vec<f64> = (0..steps)
                .map(|_| {
                    let t0 = Instant::now();
                    e.step_slots(&feeds_of(&last), &mut toks).map(|_| t0.elapsed().as_secs_f64() * 1e3)
                })
                .collect::<Result<_, _>>()?;
            v.sort_by(f64::total_cmp);
            println!("{{\"cap\":{},\"ms\":{:.4}}}", if cap == u32::MAX { -1 } else { cap as i64 }, v[v.len() / 2]);
        }
        return Ok(());
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
            if let Some(histories) = &mut histories {
                record_inputs(histories, &last, &toks, k);
            }
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
        if let Some(histories) = &mut histories {
            record_inputs(histories, &last, &toks, 1);
        }
        last.copy_from_slice(&toks);
        for (b, &t) in toks.iter().enumerate() {
            digest = (digest ^ t as u64).wrapping_mul(0x100000001b3);
            streams[b].push(t);
        }
        done += 1;
    }
    if !partial {
        println!("TOK_STREAM slots={slots} ctx={ctx} fnv={digest:016x} slot0={:?}", streams[0]);
    }
    if same && !partial {
        let agree = streams.iter().filter(|s| **s == streams[0]).count();
        println!("SLOTS_AGREE {agree}/{slots} (identical prompts on every slot)");
        for (b, s) in streams.iter().enumerate().skip(1) {
            if *s != streams[0] {
                println!("  slot{b}={s:?}");
            }
        }
    }
    // 0 steps: no timing; a dump then holds the allocations right after prefill.
    if !ms.is_empty() {
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let median = ms[ms.len() / 2];
        let sd = if ms.len() > 1 {
            (ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (ms.len() - 1) as f64).sqrt()
        } else {
            0.0
        };
        let measurement = if partial { "PARTIAL_STEP" } else { "RAW_STEP" };
        println!(
            "{measurement} slots={slots} ctx={ctx} n={} mean_ms={mean:.3} median_ms={median:.3} \
             sd_ms={sd:.3} min_ms={:.3} max_ms={:.3} per_user_tok_s={:.1} aggregate_tok_s={:.1}",
            ms.len(),
            ms[0],
            ms[ms.len() - 1],
            1000.0 / mean,
            1000.0 / mean * slots as f64,
        );
    }

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
                "scope": if partial { "raw allocations after partial decode; outputs may be stale" }
                    else if steps == 0 { "raw complete allocations after prefill" }
                    else { "raw complete allocations after the last decode step" },
                "max_inst": max_inst, "max_segments": max_segments,
                "slots": slots, "ctx": ctx, "steps": steps, "warmup": warmup,
                "packed_prefill": packed_prefill,
                "token_histories": histories,
                "sampled_token_ids": if !partial { Some(last) } else { None },
                "tensors": rows,
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

/// `--ride ROWS W1,W2,..`: the unified token batch's per-rider cost. Slots `0..W` (at `ctx`)
/// ride one packed launch of ROWS fresh prompt rows (intermediate chunks on the slots after the
/// prefilled ones, nothing sampled) against the same launch alone and a standalone decode step of
/// `W` rows. Every arm rewinds the slots first, so context and inputs are fixed; arms interleave
/// per iteration. The riders' logits are compared in process with the standalone step's;
/// `--ride-dump DIR` also writes both (bf16, `W x vocab`) for cross-run comparison.
#[cfg(feature = "cuda")]
fn ride_bench(
    e: &mut plowrt::exec::gpu::GpuEngine,
    last: &[u32],
    ctx: usize,
    iters: usize,
    rows: usize,
    widths: &[usize],
    dump: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use plow_asset::token_batch::{Phase, Request, Selection};
    use std::time::Instant;
    let first = last.len();
    let wmax = *widths.iter().max().ok_or("--ride needs a width")?;
    let chunk = e.pf_request_max_rows().min(e.pf_max_rows()).min(rows).max(1);
    let fills: Vec<usize> = (0..rows.div_ceil(chunk)).map(|i| chunk.min(rows - i * chunk)).collect();
    if !e.token_batch_enabled() || wmax > first || first + fills.len() > e.batch() {
        return Err("--ride needs the token batch, widths <= slots, slots + prefill requests <= batch".into());
    }
    let prompt: Vec<u32> = (0..2 * chunk as u32).map(|i| 1000 + (i * 7919) % 50000).collect();
    for i in 0..fills.len() {
        e.begin_slot(first + i, prompt.len() + 1)?;
    }
    if let Some(dir) = dump {
        std::fs::create_dir_all(dir)?;
    }
    let reset = |e: &mut plowrt::exec::gpu::GpuEngine| -> Result<(), Box<dyn std::error::Error>> {
        for b in 0..wmax {
            e.rewind_slot(b, ctx)?;
        }
        for i in 0..fills.len() {
            e.rewind_slot(first + i, 0)?;
        }
        Ok(())
    };
    let generations: Vec<u32> =
        (0..first + fills.len()).map(|s| e.slot_generation(s).expect("slot")).collect();
    let requests = |w: usize| {
        let request = |slot: usize, phase, tokens| Request {
            id: slot as u32,
            slot: slot as u32,
            state_slot: slot as u32,
            generation: generations[slot],
            phase,
            tokens,
            prompt_len: match phase {
                Phase::Decode => ctx as u32,
                Phase::Prefill => prompt.len() as u32,
            },
            selection: Selection::default(),
        };
        (0..w)
            .map(|b| request(b, Phase::Decode, std::slice::from_ref(&last[b])))
            .chain(fills.iter().enumerate().map(|(i, &n)| request(first + i, Phase::Prefill, &prompt[..n])))
            .collect::<Vec<Request<'_>>>()
    };
    let feeds: Vec<(usize, u32)> = last.iter().copied().enumerate().collect();
    let (mut output, mut toks, mut logits) = (Vec::new(), Vec::new(), Vec::new());
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let bf16 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect() };
    let (mut pure, warm) = (Vec::new(), 4);
    for &w in widths {
        let (mut ride, mut step) = (Vec::new(), Vec::new());
        for it in 0..warm + iters {
            reset(e)?;
            let t = Instant::now();
            e.token_batch_step(&requests(0), &mut output)?;
            if it >= warm {
                pure.push(t.elapsed().as_secs_f64() * 1e3);
            }
            reset(e)?;
            let reqs = requests(w);
            let t = Instant::now();
            e.token_batch_step(&reqs, &mut output)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let ride_ids: Vec<u32> = output.iter().map(|&(_, t)| t).collect();
            if output.iter().map(|&(id, _)| id as usize).ne(0..w) {
                return Err("rider outputs out of order".into());
            }
            let mut ride_logits = Vec::with_capacity(if it == warm { w } else { 0 });
            if it == warm {
                for r in 0..w {
                    e.logits_row(r, &mut logits)?;
                    ride_logits.push(logits.clone());
                }
            }
            reset(e)?;
            let t = Instant::now();
            e.step_slots(&feeds[..w], &mut toks)?;
            let step_ms = t.elapsed().as_secs_f64() * 1e3;
            if it >= warm {
                ride.push(ms);
                step.push(step_ms);
            }
            if it == warm {
                let (mut max_abs, mut kl_max, mut agree) = (0f32, 0f64, 0);
                let mut raw_ride = Vec::new();
                let mut raw_step = Vec::new();
                for (r, rl) in ride_logits.iter().enumerate() {
                    e.logits_row(r, &mut logits)?;
                    agree += usize::from(ride_ids[r] == toks[r]);
                    let lse = |v: &[f32]| {
                        let m = v.iter().copied().fold(f32::MIN, f32::max) as f64;
                        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
                    };
                    let (lp, lq) = (lse(&logits), lse(rl));
                    let kl: f64 = logits
                        .iter()
                        .zip(rl)
                        .map(|(&p, &q)| {
                            let lp_i = p as f64 - lp;
                            lp_i.exp() * (lp_i - (q as f64 - lq))
                        })
                        .sum();
                    kl_max = kl_max.max(kl);
                    max_abs = rl.iter().zip(&logits).map(|(a, b)| (a - b).abs()).fold(max_abs, f32::max);
                    if dump.is_some() {
                        raw_ride.extend(bf16(rl));
                        raw_step.extend(bf16(&logits));
                    }
                }
                println!(
                    "RIDE_VS_STEP w={w} ctx={ctx} top1_agree={agree}/{w} max_kl={kl_max:.3e} max_abs={max_abs:.4} ride_ids={ride_ids:?}"
                );
                if let Some(dir) = dump {
                    std::fs::write(dir.join(format!("ride-w{w}.bf16")), raw_ride)?;
                    std::fs::write(dir.join(format!("step-w{w}.bf16")), raw_step)?;
                }
            }
        }
        let (p, r, s) = (median(&mut pure), median(&mut ride), median(&mut step));
        println!(
            "RIDE rows={rows} w={w} ctx={ctx} n={iters} pure_ms={p:.3} ride_ms={r:.3} step_ms={s:.3} per_rider_ms={:.4} saved_ms={:.3}",
            (r - p) / w as f64,
            p + s - r
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::record_inputs;

    #[test]
    fn histories_include_inputs_but_exclude_the_final_sample() {
        let prompts = vec![vec![10, 11], vec![20, 21]];
        let mut single = prompts.clone();
        record_inputs(&mut single, &[12, 22], &[13, 23], 1);
        record_inputs(&mut single, &[13, 23], &[14, 24], 1);
        record_inputs(&mut single, &[14, 24], &[15, 25], 1);
        let mut multi = prompts;
        record_inputs(&mut multi, &[12, 22], &[13, 14, 15, 23, 24, 25], 3);
        assert_eq!(single, multi);
        assert_eq!(
            single,
            vec![vec![10, 11, 12, 13, 14], vec![20, 21, 22, 23, 24]]
        );
    }
}

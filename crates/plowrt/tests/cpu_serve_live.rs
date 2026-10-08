//! Live CPU-serve regression: the mux's slot lifecycle against a real ladder blob —
//! admit, step, release in the middle, re-admit, and a gap in the live set (slots
//! {0,2} live, 1 idle) that selects a wider rung than the live count. Needs a model:
//! `PLOW_LADDER_BLOB=<model.pkt> PLOW_CKPT=<hf snapshot> cargo test --release
//! --features cpu --test cpu_serve_live -- --ignored --nocapture`.
#![cfg(feature = "cpu")]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use plowrt::exec::cpu::engine::CpuEngineOpts;
use plowrt::serve::cpu_serve::CpuServe;
use plowrt::serve::engine::SeqEngine;

fn env_paths() -> Option<(PathBuf, PathBuf)> {
    let blob = std::env::var_os("PLOW_LADDER_BLOB")?;
    let ckpt = std::env::var_os("PLOW_CKPT")?;
    Some((blob.into(), ckpt.into()))
}

/// Abort (with a stack dump if `eu-stack` is available) if a step takes longer than `secs`.
fn watchdog(secs: u64, what: &'static str) -> std::sync::mpsc::Sender<()> {
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        if rx.recv_timeout(Duration::from_secs(secs)).is_err() {
            let pid = std::process::id();
            let out = std::process::Command::new("eu-stack")
                .args(["-p", &pid.to_string()])
                .output();
            if let Ok(o) = out {
                eprintln!("{}", String::from_utf8_lossy(&o.stdout));
            }
            eprintln!("WATCHDOG: {what} exceeded {secs}s — aborting");
            std::process::abort();
        }
    });
    tx
}

fn step(e: &mut CpuServe, feeds: &[(usize, u32)], what: &'static str) -> Vec<(usize, u32)> {
    let wd = watchdog(240, what);
    let t = Instant::now();
    let out = SeqEngine::step_batch(e, feeds).unwrap_or_else(|err| panic!("{what}: {err}"));
    let _ = wd.send(());
    eprintln!(
        "{what}: feeds={feeds:?} -> {out:?} in {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(out.len(), feeds.len(), "{what}: one output per feed");
    for (k, &(s, _)) in feeds.iter().enumerate() {
        assert_eq!(out[k].0, s, "{what}: outputs follow feed order");
    }
    out
}

#[test]
#[ignore]
fn live_slot_lifecycle_with_gaps() {
    let Some((blob, ckpt)) = env_paths() else {
        eprintln!("PLOW_LADDER_BLOB / PLOW_CKPT unset — skipping");
        return;
    };
    let tok = plowrt::text::tokenizer::load_tokenizer(&ckpt);
    let prompt = |q: &str| {
        tok.encode_with_special_tokens(
            &format!("<bos><|turn>user\n{q}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
            true,
        )
    };
    let a = prompt("What is the capital of France? Answer in one word.");
    let b = prompt("What is the capital of Germany? Answer in one word.");
    let c = prompt("What is the capital of Italy? Answer in one word.");
    let d = prompt("What is the capital of Japan? Answer in one word.");

    let mut opts = CpuEngineOpts::default();
    opts.threads = std::env::var("PLOW_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let mut e = CpuServe::load(&blob, &ckpt, &opts).expect("load");
    assert!(e.batch() >= 4, "ladder blob must serve >= 4 slots");

    // 3 admits, one step each as the mux does (rung 1 -> 2 -> 4).
    let t0 = e.prefill(0, &a).expect("prefill 0");
    let o = step(&mut e, &[(0, t0)], "step rung1");
    let t1 = e.prefill(1, &b).expect("prefill 1");
    let o2 = step(&mut e, &[(0, o[0].1), (1, t1)], "step rung2");
    let t2 = e.prefill(2, &c).expect("prefill 2");
    let o3 = step(
        &mut e,
        &[(0, o2[0].1), (1, o2[1].1), (2, t2)],
        "step rung4 contiguous",
    );

    // Release the MIDDLE slot: live {0,2}, idle 1 inside the rung -> the log's `rung=4 occupied=3`.
    SeqEngine::release(&mut e, 1);
    let o4 = step(&mut e, &[(0, o3[0].1), (2, o3[2].1)], "step rung4 with gap");

    // Re-admit into the gap while the others decode.
    let t3 = e.prefill(1, &d).expect("prefill 1 again");
    let _o5 = step(
        &mut e,
        &[(0, o4[0].1), (1, t3), (2, o4[1].1)],
        "step rung4 refilled",
    );

    // Release all but the highest slot: live {2} -> rows 3 -> still rung 4 with two idle rows.
    SeqEngine::release(&mut e, 0);
    SeqEngine::release(&mut e, 1);
    let _o6 = step(&mut e, &[(2, _o5[2].1)], "step rung4 single high slot");

    // Decode a few greedy tokens on slot 2 and check the answer is sane (Rome, possibly
    // wrapped in Gemma-4 thinking tokens).
    let mut ids = vec![t2, o3[2].1, o4[1].1, _o5[2].1];
    let mut last = _o6[0].1;
    for _ in 0..8 {
        ids.push(last);
        last = step(&mut e, &[(2, last)], "decode slot 2")[0].1;
    }
    let text = tok.decode(&ids);
    eprintln!("slot 2 text: {text:?}");
    // A block asset (plowc --block) has no embed/lm_head, so its tokens are meaningless;
    // set PLOW_EXPECT_TEXT=0 to exercise only the slot lifecycle on it.
    if std::env::var("PLOW_EXPECT_TEXT").map_or(true, |v| v != "0") {
        assert!(
            text.contains("Rome"),
            "slot 2 should answer Rome, got {text:?}"
        );
    }
}

/// Unified token batch: decode feeds ride a packed prefill launch beside a partial chunk and a
/// completing prompt. Completed prompts must sample exactly what a whole-prompt prefill does
/// (packed prefill is chunking-invariant); decode rows must stay on the answer.
#[test]
#[ignore]
fn live_token_batch_matches_isolated_paths() {
    let Some((blob, ckpt)) = env_paths() else {
        eprintln!("PLOW_LADDER_BLOB / PLOW_CKPT unset — skipping");
        return;
    };
    let tok = plowrt::text::tokenizer::load_tokenizer(&ckpt);
    let prompt = |q: &str| {
        tok.encode_with_special_tokens(
            &format!("<bos><|turn>user\n{q}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
            true,
        )
    };
    let a = prompt("What is the capital of France? Answer in one word.");
    let b = prompt("What is the capital of Germany? Answer in one word.");
    let c = prompt("What is the capital of Italy? Answer in one word.");
    let mut opts = CpuEngineOpts::default();
    opts.threads = std::env::var("PLOW_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let mut e = CpuServe::load(&blob, &ckpt, &opts).expect("load");
    assert!(e.batch() >= 3, "blob must serve >= 3 slots");
    assert!(e.token_batch_rows(1, 0, 1).is_some(), "packet has no packed prefill bucket");

    let ref_b = e.prefill(1, &b).expect("prefill b");
    SeqEngine::release(&mut e, 1);
    let ref_c = e.prefill(2, &c).expect("prefill c");
    SeqEngine::release(&mut e, 2);
    let mut ref_a = vec![e.prefill(0, &a).expect("prefill a")];
    for _ in 0..10 {
        let last = *ref_a.last().unwrap();
        ref_a.push(step(&mut e, &[(0, last)], "reference decode")[0].1);
    }
    SeqEngine::release(&mut e, 0);

    let t0 = e.prefill(0, &a).expect("prefill a");
    let mut out = Vec::new();
    for (s, p) in [(1, &b), (2, &c)] {
        e.prepare_packed_prefill_slot(s, p, u32::MAX).expect("prepare");
    }
    let cut = (b.len() / 2) as u32;
    e.token_batch_step(0, &[(0, t0)], &[(1, &b, cut), (2, &c, c.len() as u32)], &mut out)
        .expect("token batch 1");
    eprintln!("token batch 1 -> {out:?}");
    assert_eq!(out.len(), 2, "feed + completed prompt");
    assert_eq!(out[0].0, 0);
    assert_eq!(out[1], (2, ref_c), "completed prompt samples its whole-prefill token");
    assert_eq!(e.prefill_frontier(1), Some(cut as usize));
    let (o0, o2) = (out[0].1, out[1].1);
    e.token_batch_step(0, &[(0, o0), (2, o2)], &[(1, &b, b.len() as u32 - cut)], &mut out)
        .expect("token batch 2");
    eprintln!("token batch 2 -> {out:?}");
    assert_eq!(out.len(), 3);
    assert_eq!(out[2], (1, ref_b), "a prompt split across launches samples its whole-prefill token");

    let mut ids = vec![t0, o0, out[0].1];
    let mut feeds: Vec<(usize, u32)> = out.iter().map(|&(s, t)| (s as usize, t)).collect();
    for _ in 0..8 {
        let o = step(&mut e, &feeds, "decode after token batch");
        ids.push(o[0].1);
        feeds = o;
    }
    eprintln!("slot 0 text: {:?}", tok.decode(&ids));
    assert_eq!(ids, ref_a, "decode rows inside a token batch keep the isolated greedy sequence");
}

/// Token-batch cost on a real packet: a decode step of every slot, a prefill chunk alone, the
/// chunk with the decodes riding it, and the multi-sample head. `PLOW_TB_ROWS` sets the chunk.
#[test]
#[ignore]
fn live_token_batch_cost() {
    let Some((blob, ckpt)) = env_paths() else {
        eprintln!("PLOW_LADDER_BLOB / PLOW_CKPT unset — skipping");
        return;
    };
    let rows: u32 = std::env::var("PLOW_TB_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let mut opts = CpuEngineOpts::default();
    opts.threads = std::env::var("PLOW_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(96);
    let mut e = CpuServe::load(&blob, &ckpt, &opts).expect("load");
    let b = e.batch();
    let live = b - 2;
    let ctx: Vec<u32> = (0..2048u32).map(|i| 1000 + i % 5000).collect();
    let mut feeds = Vec::new();
    for s in 0..live {
        feeds.push((s, e.prefill(s, &ctx[..64 + s]).expect("prefill")));
    }
    let time = |what: &str, f: &mut dyn FnMut()| {
        f();
        let t = Instant::now();
        for _ in 0..3 {
            f();
        }
        eprintln!("{what}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3 / 3.0);
    };
    time(&format!("decode step, {live} rows"), &mut || {
        let o = SeqEngine::step_batch(&mut e, &feeds).expect("step");
        feeds = o;
    });
    let long: Vec<u32> = (0..8192u32).map(|i| 2000 + i % 7000).collect();
    let (pa, pb) = (b - 2, b - 1);
    let mut out = Vec::new();
    let mut reset = |e: &mut CpuServe| {
        SeqEngine::release(e, pa);
        SeqEngine::release(e, pb);
        e.prepare_packed_prefill_slot(pa, &long, u32::MAX).expect("prep");
        e.prepare_packed_prefill_slot(pb, &long[..rows as usize / 2], u32::MAX).expect("prep");
    };
    time(&format!("chunk alone, {rows} rows, no sample"), &mut || {
        reset(&mut e);
        e.token_batch_step(0, &[], &[(pa, &long, rows)], &mut out).expect("chunk");
    });
    time(&format!("chunk + {live} decode feeds"), &mut || {
        reset(&mut e);
        e.token_batch_step(0, &feeds, &[(pa, &long, rows - live as u32)], &mut out).expect("tb");
        feeds = out.iter().map(|&(s, t)| (s as usize, t)).collect();
    });
    time(&format!("chunk + {live} feeds + completing prompt (head on {} rows)", live + 1), &mut || {
        reset(&mut e);
        let half = rows / 2;
        e.token_batch_step(0, &feeds, &[(pa, &long, half - live as u32), (pb, &long[..half as usize], half)], &mut out)
            .expect("tb2");
        feeds = out.iter().filter(|o| (o.0 as usize) < live).map(|&(s, t)| (s as usize, t)).collect();
    });
}

/// Whole-prompt prefill time of one `PLOW_PF_TOKENS`-token prompt (default 12288), for profiling
/// long-context prefill.
#[test]
#[ignore]
fn live_long_prefill_time() {
    let Some((blob, ckpt)) = env_paths() else {
        eprintln!("PLOW_LADDER_BLOB / PLOW_CKPT unset — skipping");
        return;
    };
    let n: usize = std::env::var("PLOW_PF_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(12288);
    let mut opts = CpuEngineOpts::default();
    opts.threads = std::env::var("PLOW_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(96);
    let mut e = CpuServe::load(&blob, &ckpt, &opts).expect("load");
    let prompt: Vec<u32> = (0..n as u32).map(|i| 2000 + (i * 7919) % 30000).collect();
    for run in 0..2 {
        let t = Instant::now();
        e.prefill(0, &prompt).expect("prefill");
        eprintln!("prefill {n} tokens (run {run}): {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        SeqEngine::release(&mut e, 0);
    }
}

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
    eprintln!("{what}: feeds={feeds:?} -> {out:?} in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
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
            &format!("<bos><start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"),
            true,
        )
    };
    let a = prompt("What is the capital of France? Answer in one word.");
    let b = prompt("What is the capital of Germany? Answer in one word.");
    let c = prompt("What is the capital of Italy? Answer in one word.");
    let d = prompt("What is the capital of Japan? Answer in one word.");

    let mut opts = CpuEngineOpts::default();
    opts.threads = std::env::var("PLOW_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let mut e = CpuServe::load(&blob, &ckpt, &opts).expect("load");
    assert!(e.batch() >= 4, "ladder blob must serve >= 4 slots");

    // 3 admits, one step each as the mux does (rung 1 -> 2 -> 4).
    let t0 = e.prefill(0, &a).expect("prefill 0");
    let o = step(&mut e, &[(0, t0)], "step rung1");
    let t1 = e.prefill(1, &b).expect("prefill 1");
    let o2 = step(&mut e, &[(0, o[0].1), (1, t1)], "step rung2");
    let t2 = e.prefill(2, &c).expect("prefill 2");
    let o3 = step(&mut e, &[(0, o2[0].1), (1, o2[1].1), (2, t2)], "step rung4 contiguous");

    // Release the MIDDLE slot: live {0,2}, idle 1 inside the rung -> the log's `rung=4 occupied=3`.
    SeqEngine::release(&mut e, 1);
    let o4 = step(&mut e, &[(0, o3[0].1), (2, o3[2].1)], "step rung4 with gap");

    // Re-admit into the gap while the others decode.
    let t3 = e.prefill(1, &d).expect("prefill 1 again");
    let _o5 = step(&mut e, &[(0, o4[0].1), (1, t3), (2, o4[1].1)], "step rung4 refilled");

    // Release all but the highest slot: live {2} -> rows 3 -> still rung 4 with two idle rows.
    SeqEngine::release(&mut e, 0);
    SeqEngine::release(&mut e, 1);
    let _o6 = step(&mut e, &[(2, _o5[2].1)], "step rung4 single high slot");

    // Decode a few greedy tokens on slot 2 and check the answer is sane (Rome, possibly
    // wrapped in Gemma-4 thinking tokens).
    let mut ids = vec![t2];
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
        assert!(text.contains("Rome"), "slot 2 should answer Rome, got {text:?}");
    }
}

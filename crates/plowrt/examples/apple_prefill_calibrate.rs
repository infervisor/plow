//! Prefill row-share calibrator (plans/apple-heterogeneous-emit.md §6): emit the model once per
//! cell of an (ANE %, CPU %) grid through `plowc --row-split`, time a full-bucket prefill on
//! each, and write the winning split to `tuning/apple-<gpu>-prefill.json`, which `plowc` reads
//! for an Apple target when no `--row-split`/`--unit-shares` is given. Rule: a split is chosen
//! only if it beats GPU-only by > 5% AND produces the same first token.
//!
//! `cargo run --release --features ane --example apple_prefill_calibrate -- <hf-dir> <fp8-dir>
//!     [--gpu m4pro] [--plowc target/release/plowc] [--ane 0,25,40,50] [--cpu 0,10] [--reps 5]
//!     [--work ~/plow-assets/calib] [--keep]`
//! Extra emit knobs come from the environment (e.g. `PLOW_FA_GF_FULL=1` for Llama).

#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    use plowrt::exec::apple::MetalEngine;
    use plowrt::text::tokenizer::load_tokenizer;
    use std::path::PathBuf;
    use std::time::Instant;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let ckpt: PathBuf = args.next().expect("usage: <hf-dir> <fp8-dir>").into();
    let fp8: PathBuf = args.next().expect("usage: <hf-dir> <fp8-dir>").into();
    let mut gpu = String::from("m4pro");
    let mut plowc = PathBuf::from("target/release/plowc");
    let mut ane_grid: Vec<u32> = vec![0, 25, 40, 50];
    let mut cpu_grid: Vec<u32> = vec![0, 10];
    let mut reps = 5usize;
    let mut work = PathBuf::from(std::env::var("HOME").unwrap()).join("plow-assets/calib");
    let mut keep = false;
    let list =
        |s: String| -> Vec<u32> { s.split(',').map(|v| v.trim().parse().unwrap()).collect() };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gpu" => gpu = args.next().unwrap(),
            "--plowc" => plowc = args.next().unwrap().into(),
            "--ane" => ane_grid = list(args.next().unwrap()),
            "--cpu" => cpu_grid = list(args.next().unwrap()),
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--work" => work = args.next().unwrap().into(),
            "--keep" => keep = true,
            other => panic!("unknown arg {other}"),
        }
    }
    std::env::set_var("PLOW_FP8_DIR", &fp8);
    let tok = load_tokenizer(&ckpt);
    // A prompt that fills the smallest bucket (128 rows): every lane has real rows.
    let base = "The history of computation is a history of abstraction: from gears and relays to \
                transistors, from transistors to logic gates, from gates to processors, and from \
                processors to the programs that give them purpose. Each layer hides the one beneath it. ";
    let mut prompt = String::new();
    while tok.encode_with_special_tokens(&prompt, true).len() < 124 {
        prompt.push_str(base);
    }
    let ids = tok.encode_with_special_tokens(&prompt, true);
    let ids = &ids[..ids.len().min(128)];
    println!("prompt: {} tokens", ids.len());

    // The ANE program cache is keyed by rows, so every cell shares one directory.
    let ane_cache = work.join("ane");
    std::fs::create_dir_all(&ane_cache).expect("work dir");
    let mut cells = Vec::new();
    let mut gpu_only: Option<(f64, u32)> = None;
    for &cpu in &cpu_grid {
        for &ane in &ane_grid {
            if ane + cpu >= 100 {
                continue;
            }
            let dir = work.join(format!("ane{ane}-cpu{cpu}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let mut cmd = std::process::Command::new(&plowc);
            cmd.args(["--hf-dir"])
                .arg(&ckpt)
                .args(["--gpu", &gpu, "--max-ctx", "4096", "--w8a16", "--out"])
                .arg(&dir);
            // Named explicitly even for the GPU-only cell: with no split named, plowc would read the
            // previous calibration record and the baseline cell would not be GPU-only.
            cmd.env(
                "PLOW_ROW_SPLIT",
                if ane + cpu > 0 {
                    format!("ane={ane},cpu={cpu}")
                } else {
                    "gpu=100".to_string()
                },
            );
            let t = Instant::now();
            let st = cmd
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .expect("run plowc");
            assert!(st.success(), "plowc failed for ane={ane} cpu={cpu}");
            let emit_s = t.elapsed().as_secs_f64();
            if ane > 0 {
                let link = dir.join("ane");
                let _ = std::os::unix::fs::symlink(&ane_cache, &link);
            }
            let blob = dir.join("model.pkt");
            let mut eng = MetalEngine::load(&blob, &ckpt).expect("metal load");
            let first = eng
                .prefill(ids)
                .expect("prefill (compiles ANE programs on first use)");
            let mut ms = Vec::with_capacity(reps);
            for _ in 0..reps {
                if let Some(h) = eng.hetero.as_mut() {
                    h.reset_stats();
                }
                let t = Instant::now();
                let f = eng.prefill(ids).expect("prefill");
                ms.push(t.elapsed().as_secs_f64() * 1e3);
                assert_eq!(f, first);
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ms[ms.len() / 2];
            let (ane_ms, cpu_ms, wait_ms) = eng
                .hetero
                .as_ref()
                .map(|h| (h.stats.ane_ms, h.stats.cpu_ms, h.stats.gpu_wait_ms))
                .unwrap_or((0.0, 0.0, 0.0));
            if ane + cpu == 0 {
                gpu_only = Some((med, first));
            }
            let same = gpu_only.is_none_or(|(_, f)| f == first);
            println!(
                "ane {ane:>2}% cpu {cpu:>2}%: prefill {med:7.1} ms (min {:.1})  ane {ane_ms:.1} ms  cpu {cpu_ms:.1} ms  gpu-wait {wait_ms:.1} ms  first {first} {:?}{}  [emit {emit_s:.0}s]",
                ms[0],
                tok.decode(&[first]),
                if same { "" } else { " (DIFFERS from gpu-only)" }
            );
            cells.push(serde_json::json!({
                "ane_pct": ane, "cpu_pct": cpu, "prefill_ms_median": med, "prefill_ms_min": ms[0],
                "ane_ms": ane_ms, "cpu_ms": cpu_ms, "gpu_wait_ms": wait_ms, "first_token": first,
                "same_first_token": same,
            }));
            let gpu_name = eng.gpu_name.clone();
            drop(eng);
            if !keep {
                let _ = std::fs::remove_dir_all(&dir);
            }
            let _ = gpu_name;
        }
    }
    let (base_ms, _) = gpu_only.expect("the grid must include ane=0,cpu=0");
    let best = cells
        .iter()
        .filter(|c| c["same_first_token"].as_bool() == Some(true))
        .filter(|c| c["prefill_ms_median"].as_f64().unwrap() < base_ms * 0.95)
        .min_by(|a, b| {
            a["prefill_ms_median"]
                .as_f64()
                .unwrap()
                .partial_cmp(&b["prefill_ms_median"].as_f64().unwrap())
                .unwrap()
        });
    let chosen = best
        .map(|c| {
            (
                c["ane_pct"].as_u64().unwrap(),
                c["cpu_pct"].as_u64().unwrap(),
            )
        })
        .unwrap_or((0, 0));
    println!(
        "gpu-only {base_ms:.1} ms -> chosen prefill split ane {}% cpu {}%",
        chosen.0, chosen.1
    );
    let spec = hwspec::registry::lookup(&gpu).expect("known --gpu");
    let slug = spec.name.to_lowercase().replace(' ', "-");
    let doc = serde_json::json!({
        "schema": "apple-prefill-split-v1",
        "gpu": spec.name,
        "model": ckpt.file_name().map(|s| s.to_string_lossy().to_string()),
        "prompt_tokens": ids.len(),
        "rule": "whole-prefill median; split = argmin if it beats gpu-only by > 5% with the same first token, else 0",
        "gpu_only_ms": base_ms,
        "cells": cells,
        "chosen": {"ane_pct": chosen.0, "cpu_pct": chosen.1},
    });
    let model_slug = ckpt
        .file_name()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "model".into());
    let path = PathBuf::from(format!("tuning/apple-{slug}-{model_slug}-prefill.json"));
    std::fs::create_dir_all(path.parent().unwrap()).ok();
    std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).expect("write");
    println!("wrote {}", path.display());
    if !keep {
        let _ = std::fs::remove_dir_all(&ane_cache);
    }
}

#[cfg(not(all(feature = "ane", target_os = "macos")))]
fn main() {
    eprintln!("build with --features ane on macOS");
}

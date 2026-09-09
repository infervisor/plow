//! Rung 3 calibrator (plans/apple-silicon-backend.md §2.1, §5.4): measure the decode step of a
//! loaded model with the GPU alone and with the CPU taking a column share of the GEMV-family
//! ops INSIDE the persistent walk (`PLOW_CPU_SHARE`, see `exec::apple`), and write the unit
//! shares. Whole steps, not per-op rooflines: a lone GEMV dispatch is dominated by the command
//! buffer floor, and the bus is shared, so only the combined step time says what a share buys.
//!
//! `cargo run --release --features metal --example apple_calibrate -- <model.pkt> <ckpt> [--out shares.json] [--steps N]`

#[cfg(all(feature = "metal", target_os = "macos"))]
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
    let blob: PathBuf = args
        .next()
        .expect("usage: apple_calibrate <model.pkt> <ckpt>")
        .into();
    let ckpt: PathBuf = args
        .next()
        .expect("usage: apple_calibrate <model.pkt> <ckpt>")
        .into();
    let mut out: Option<PathBuf> = None;
    let mut steps = 12usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = Some(args.next().unwrap().into()),
            "--steps" => steps = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    let tok = load_tokenizer(&ckpt);
    let ids = tok.encode_with_special_tokens("The capital of France is", true);

    // (cpu share %, ops) cells: the GPU-only baseline, then a few ops and every GEMV op at two shares.
    let cells: Vec<(u32, Option<usize>)> = vec![
        (0, None),
        (15, Some(8)),
        (30, Some(8)),
        (15, None),
        (30, None),
    ];
    let mut rows = Vec::new();
    let mut gpu_name = String::new();
    let mut baseline = f64::INFINITY;
    for (pct, ops) in &cells {
        let spec = match (pct, ops) {
            (0, _) => String::new(),
            (p, Some(n)) => format!("{p}:{n}"),
            (p, None) => p.to_string(),
        };
        std::env::set_var("PLOW_CPU_SHARE", &spec);
        let mut eng = MetalEngine::load(&blob, &ckpt).expect("metal load");
        gpu_name = eng.gpu_name.clone();
        let first = eng.prefill(&ids).expect("prefill");
        let mut pos = ids.len() as u32;
        let mut ms = Vec::with_capacity(steps);
        let mut last = first;
        let mut text = vec![first];
        for _ in 0..steps {
            let t = Instant::now();
            last = eng.decode_step(pos, pos + 1).expect("decode");
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            text.push(last);
            pos += 1;
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ms[ms.len() / 2];
        let split_ops = eng.last_cpu.map(|(n, _)| n).unwrap_or(0);
        let cpu_ms = eng.last_cpu.map(|(_, m)| m).unwrap_or(0.0);
        if *pct == 0 {
            baseline = med;
        }
        let label = match ops {
            Some(n) => format!("cpu {pct}% on {n} ops"),
            None => format!("cpu {pct}% on all GEMV ops"),
        };
        println!(
            "{label:<28}: decode {med:7.2} ms/tok (split ops {split_ops}, cpu-side {cpu_ms:.1} ms)  {:?}",
            tok.decode(&text)
        );
        rows.push(serde_json::json!({
            "cpu_share_pct": pct, "ops": ops, "decode_ms_median": med, "split_ops": split_ops,
            "cpu_side_ms": cpu_ms, "text": tok.decode(&text)
        }));
    }
    std::env::remove_var("PLOW_CPU_SHARE");
    // §2.1: a share must beat GPU-only by more than the noise band (5%), else 0.
    let best = rows
        .iter()
        .filter(|r| r["decode_ms_median"].as_f64().unwrap() < baseline * 0.95)
        .min_by(|a, b| {
            a["decode_ms_median"]
                .as_f64()
                .unwrap()
                .partial_cmp(&b["decode_ms_median"].as_f64().unwrap())
                .unwrap()
        });
    let chosen = best
        .map(|r| r["cpu_share_pct"].as_u64().unwrap())
        .unwrap_or(0);
    println!("gpu-only {baseline:.2} ms/tok -> chosen decode cpu share {chosen}%");
    let doc = serde_json::json!({
        "schema": "apple-unit-shares-v2",
        "gpu": gpu_name,
        "model": blob.parent().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().to_string()),
        "phase": "decode",
        "rule": "whole-step median; share = argmin if it beats gpu-only by > 5%, else 0",
        "gpu_only_ms": baseline,
        "cells": rows,
        "chosen": {"gpu": 1.0 - chosen as f64 / 100.0, "cpu": chosen as f64 / 100.0, "ane": 0.0},
    });
    let path = out.unwrap_or_else(|| {
        PathBuf::from(format!(
            "tuning/apple-{}-shares.json",
            gpu_name.to_lowercase().replace(' ', "-")
        ))
    });
    std::fs::create_dir_all(path.parent().unwrap()).ok();
    std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).expect("write shares");
    println!("wrote {}", path.display());
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("build with --features metal on macOS");
}

//! End-to-end Metal run: prefill a prompt on the Apple GPU engine, decode greedily, print the
//! text and per-step timings — the GPU twin of `cpu_chat`, same arguments.
//!
//! `cargo run --release --features metal --example apple_chat -- <model.pkt> <checkpoint-dir> [--tokens N] [--prompt "..."] [--chat "..."]`

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use plowrt::exec::apple::MetalEngine;
    use plowrt::text::tokenizer::load_tokenizer;
    use std::path::PathBuf;
    use std::time::Instant;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args
        .next()
        .expect("usage: apple_chat <model.pkt> <checkpoint-dir>")
        .into();
    let ckpt: PathBuf = args
        .next()
        .expect("usage: apple_chat <model.pkt> <checkpoint-dir>")
        .into();
    let mut n_tokens = 16usize;
    let mut pf_reps = 0usize;
    let mut prompt = String::from("The capital of France is");
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tokens" => n_tokens = args.next().unwrap().parse().unwrap(),
            "--pf-reps" => pf_reps = args.next().unwrap().parse().unwrap(),
            "--prompt" => prompt = args.next().unwrap(),
            "--chat" => {
                let q = args.next().unwrap();
                prompt = format!(
                    "<bos><|turn>user\n{q}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
                );
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let tok = load_tokenizer(&ckpt);
    assert!(
        !tok.is_byte_fallback(),
        "no tokenizer.json in {}",
        ckpt.display()
    );
    let ids = tok.encode_with_special_tokens(&prompt, true);
    println!("prompt: {prompt:?} -> {} tokens {:?}", ids.len(), ids);

    let t0 = Instant::now();
    let mut eng = MetalEngine::load(&blob, &ckpt).expect("engine load");
    println!(
        "engine: gpu={} max_ctx={} load={:.1}s",
        eng.gpu_name,
        eng.max_ctx(),
        t0.elapsed().as_secs_f64()
    );

    let t1 = Instant::now();
    let first = eng.prefill(&ids).expect("prefill");
    let ttft = t1.elapsed();
    println!(
        "prefill: {} tokens in {:.1} ms ({:.1} tok/s) -> first token {} {:?}",
        ids.len(),
        ttft.as_secs_f64() * 1e3,
        ids.len() as f64 / ttft.as_secs_f64(),
        first,
        tok.decode(&[first])
    );
    if let Some((ops, ms)) = eng.last_ane {
        println!("ane: {ops} prefill GEMMs ran on the Neural Engine, {ms:.1} ms of ANE time");
    }
    let print_hetero = |eng: &MetalEngine| {
        if let Some(h) = &eng.hetero {
            let s = h.stats;
            println!(
                "hetero: {} segments; ane {} programs {:.1} ms; cpu {} ops {:.1} ms; gpu-wait after lanes {:.1} ms (plan ane {}% cpu {}%)",
                s.segs, s.ane_runs, s.ane_ms, s.cpu_ops, s.cpu_ms, s.gpu_wait_ms, h.plan.ane_pct, h.plan.cpu_pct
            );
        }
    };
    print_hetero(&eng);
    if pf_reps > 0 {
        let mut ms = Vec::with_capacity(pf_reps);
        for _ in 0..pf_reps {
            if let Some(h) = eng.hetero.as_mut() {
                h.reset_stats();
            }
            let t = Instant::now();
            let f = eng.prefill(&ids).expect("prefill");
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(f, first, "prefill is not deterministic");
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "prefill x{pf_reps}: min {:.1} ms, median {:.1} ms, max {:.1} ms",
            ms[0],
            ms[ms.len() / 2],
            ms[ms.len() - 1]
        );
        print_hetero(&eng);
    }
    let mut out = vec![first];
    let mut pos = ids.len() as u32;
    let mut step_ms = Vec::with_capacity(n_tokens);
    for _ in 1..n_tokens {
        let t = Instant::now();
        let next = eng.decode_step(pos, pos + 1).expect("decode");
        step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        if step_ms.len() % 100 == 0 {
            let w = &step_ms[step_ms.len() - 100..];
            println!(
                "  steps {}..{}: mean {:.1} ms/tok, min {:.1}, max {:.1}",
                step_ms.len() - 100,
                step_ms.len(),
                w.iter().sum::<f64>() / 100.0,
                w.iter().cloned().fold(f64::INFINITY, f64::min),
                w.iter().cloned().fold(0.0, f64::max)
            );
        }
        out.push(next);
        pos += 1;
    }
    println!("output: {:?}", tok.decode(&out));
    println!("tokens: {out:?}");
    if let Some((ops, ms)) = eng.last_cpu {
        println!("cpu share: {ops} decode ops split with the CPU, {ms:.2} ms of CPU-side time in the last step");
    }
    if !step_ms.is_empty() {
        let mean = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
        let min = step_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "decode: {} steps, mean {mean:.1} ms/tok ({:.2} tok/s), min {min:.1} ms",
            step_ms.len(),
            1e3 / mean
        );
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("build with --features metal on macOS");
}

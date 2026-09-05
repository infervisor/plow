//! End-to-end CPU run: prefill a prompt on the CPU engine, decode greedily,
//! print the text and per-step timings.
//!
//! `cargo run --release --features cpu --example cpu_chat -- <model.pkt> <checkpoint-dir> [--tokens N] [--threads T] [--isa amx|avx512|scalar] [--prompt "..."]`

#[cfg(feature = "cpu")]
fn main() {
    use plowrt::exec::cpu::engine::{CpuEngine, CpuEngineOpts};
    use plowrt::exec::cpu::ffi::Isa;
    use plowrt::text::tokenizer::load_tokenizer;
    use std::path::PathBuf;
    use std::time::Instant;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage: cpu_chat <model.pkt> <checkpoint-dir>").into();
    let ckpt: PathBuf = args.next().expect("usage: cpu_chat <model.pkt> <checkpoint-dir>").into();
    let mut n_tokens = 16usize;
    let mut opts = CpuEngineOpts::default();
    let mut prompt = String::from("The capital of France is");
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tokens" => n_tokens = args.next().unwrap().parse().unwrap(),
            "--threads" => opts.threads = args.next().unwrap().parse().unwrap(),
            "--spin-us" => opts.spin_us = args.next().unwrap().parse().unwrap(),
            "--isa" => {
                opts.isa = match args.next().unwrap().as_str() {
                    "scalar" => Isa::Scalar,
                    "avx512" => Isa::Avx512,
                    _ => Isa::Amx,
                }
            }
            "--prompt" => prompt = args.next().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }

    let tok = load_tokenizer(&ckpt);
    assert!(!tok.is_byte_fallback(), "no tokenizer.json in {}", ckpt.display());
    let ids = tok.encode_with_special_tokens(&prompt, true);
    println!("prompt: {prompt:?} -> {} tokens {:?}", ids.len(), ids);

    let t0 = Instant::now();
    let mut eng = CpuEngine::load(&blob, &ckpt, &opts).expect("engine load");
    println!(
        "engine: isa={:?} threads={} max_ctx={} load={:.1}s",
        eng.isa,
        eng.threads,
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

    let mut out = vec![first];
    let mut pos = ids.len() as u32;
    let mut step_ms = Vec::with_capacity(n_tokens);
    for _ in 1..n_tokens {
        let t = Instant::now();
        let next = eng.decode_step(pos, pos + 1).expect("decode");
        step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        out.push(next);
        pos += 1;
    }
    println!("output: {:?}", tok.decode(&out));
    println!("tokens: {out:?}");
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

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}

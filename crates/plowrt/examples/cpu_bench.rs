//! CPU latency benchmark: TTFT (prefill) and TPOT (decode) at several prompt
//! lengths, batch 1 — the same quantities vLLM's `benchmark_latency` reports.
//!
//! `cargo run --release --features cpu --example cpu_bench -- <model.pkt> <ckpt> [--prompt-lens 32,128,512] [--decode 32] [--threads T] [--isa amx|avx512|scalar] [--json out.json]`

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
                .unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage: cpu_bench <model.pkt> <ckpt>").into();
    let ckpt: PathBuf = args.next().expect("usage: cpu_bench <model.pkt> <ckpt>").into();
    let mut lens: Vec<usize> = vec![32, 128, 512];
    let mut n_dec = 32usize;
    let mut json: Option<PathBuf> = None;
    let mut opts = CpuEngineOpts::default();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--prompt-lens" => {
                lens = args.next().unwrap().split(',').map(|s| s.parse().unwrap()).collect()
            }
            "--decode" => n_dec = args.next().unwrap().parse().unwrap(),
            "--threads" => opts.threads = args.next().unwrap().parse().unwrap(),
            "--spin-us" => opts.spin_us = args.next().unwrap().parse().unwrap(),
            "--isa" => {
                opts.isa = match args.next().unwrap().as_str() {
                    "scalar" => Isa::Scalar,
                    "avx512" => Isa::Avx512,
                    _ => Isa::Amx,
                }
            }
            "--json" => json = Some(args.next().unwrap().into()),
            other => panic!("unknown arg {other}"),
        }
    }

    let tok = load_tokenizer(&ckpt);
    let mut eng = CpuEngine::load(&blob, &ckpt, &opts).expect("engine load");
    let weight_gib = eng.model().weight_bytes as f64 / (1u64 << 30) as f64;
    println!(
        "engine: isa={:?} threads={} n_cu={} weights={weight_gib:.2} GiB max_ctx={}",
        eng.isa,
        eng.threads,
        eng.model().blob.n_cu,
        eng.max_ctx()
    );
    // A natural-text prompt, repeated to length, so attention sees real structure.
    let base = tok.encode_with_special_tokens(
        "The quick brown fox jumps over the lazy dog while the river flows quietly past the old mill. ",
        false,
    );
    let mut rows = Vec::new();
    for &n in &lens {
        let mut ids = vec![2u32]; // <bos>
        while ids.len() < n {
            ids.extend_from_slice(&base);
        }
        ids.truncate(n);
        // Warm-up (page faults, tile config), then timed.
        let _ = eng.prefill(&ids).expect("prefill warm");
        let t0 = Instant::now();
        let first = eng.prefill(&ids).expect("prefill");
        let ttft_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut pos = n as u32;
        let mut steps = Vec::with_capacity(n_dec);
        let mut out = vec![first];
        for _ in 0..n_dec {
            let t = Instant::now();
            let id = eng.decode_step(pos, pos + 1).expect("decode");
            steps.push(t.elapsed().as_secs_f64() * 1e3);
            out.push(id);
            pos += 1;
        }
        let tpot = steps.iter().sum::<f64>() / steps.len().max(1) as f64;
        let p50 = {
            let mut s = steps.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s.get(s.len() / 2).copied().unwrap_or(0.0)
        };
        let bw = weight_gib * 1.0737 / (tpot / 1e3); // GB/s of weight traffic at 1 pass/token
        println!(
            "prompt={n:>5}  TTFT {ttft_ms:>9.1} ms ({:>7.1} tok/s)  TPOT mean {tpot:>7.1} ms p50 {p50:>7.1} ms ({:>5.2} tok/s, {bw:>5.1} GB/s weights)  text={:?}",
            n as f64 / (ttft_ms / 1e3),
            1e3 / tpot,
            tok.decode(&out[..out.len().min(12)])
        );
        rows.push((n, ttft_ms, tpot, p50));
    }
    if let Some(p) = json {
        let mut s = String::from("[");
        for (i, (n, ttft, tpot, p50)) in rows.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"prompt_len\":{n},\"decode\":{n_dec},\"threads\":{},\"isa\":\"{:?}\",\"ttft_ms\":{ttft:.2},\"tpot_ms\":{tpot:.3},\"tpot_p50_ms\":{p50:.3}}}",
                eng.threads, eng.isa
            ));
        }
        s.push(']');
        std::fs::write(&p, s).expect("write json");
        println!("wrote {}", p.display());
    }
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}

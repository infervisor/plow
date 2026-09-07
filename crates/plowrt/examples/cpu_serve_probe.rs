//! Drive the serve-side slot engine (`CpuServe`: prefill_slot + batched decode) directly,
//! greedy, and print the text — the same loop `cpu_chat` runs through `CpuEngine`, so a
//! divergence between the two isolates the slot path from the kernels.
//!
//! `cargo run --release --features cpu --example cpu_serve_probe -- <model.pkt> <ckpt> [--tokens N] [--threads T] [--prompt "..."]`

#[cfg(feature = "cpu")]
fn main() {
    use plowrt::exec::cpu::engine::CpuEngineOpts;
    use plowrt::serve::engine::CpuServe;
    use plowrt::text::tokenizer::load_tokenizer;
    use std::path::PathBuf;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args
        .next()
        .expect("usage: cpu_serve_probe <model.pkt> <ckpt>")
        .into();
    let ckpt: PathBuf = args
        .next()
        .expect("usage: cpu_serve_probe <model.pkt> <ckpt>")
        .into();
    let mut n_tokens = 16usize;
    let mut opts = CpuEngineOpts::default();
    let mut prompt = String::from("The capital of France is");
    let mut chunk = 0u32;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tokens" => n_tokens = args.next().unwrap().parse().unwrap(),
            "--chunk" => chunk = args.next().unwrap().parse().unwrap(),
            "--threads" => opts.threads = args.next().unwrap().parse().unwrap(),
            "--prompt" => prompt = args.next().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    let tok = load_tokenizer(&ckpt);
    let ids = tok.encode_with_special_tokens(&prompt, true);
    println!("prompt: {prompt:?} -> {} tokens {:?}", ids.len(), ids);
    let mut srv = CpuServe::load(&blob, &ckpt, &opts).expect("load");
    let first = if chunk == 0 {
        srv.prefill(0, &ids).expect("prefill")
    } else {
        loop {
            if let Some(t) = srv.prefill_chunk(0, &ids, chunk).expect("prefill_chunk") {
                break t;
            }
        }
    };
    let mut out = vec![first];
    let mut last = first;
    for _ in 1..n_tokens {
        last = srv.step(0, last).expect("step");
        out.push(last);
    }
    println!("output: {:?}", tok.decode(&out));
    println!("tokens: {out:?}");
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}

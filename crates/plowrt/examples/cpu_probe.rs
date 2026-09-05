//! Numerics probe: prefill a short prompt on the CPU engine, then dump stats of
//! every runtime tensor (NaN/inf counts, min/max as bf16 and f32) and re-derive
//! the argmax on the host. Localizes the first op that produces garbage.
//!
//! `cargo run --release --features cpu --example cpu_probe -- <model.pkt> <ckpt> [--prompt ".."] [--filter act.]`

#[cfg(feature = "cpu")]
fn main() {
    use plowrt::exec::cpu::engine::{CpuEngine, CpuEngineOpts};
    use plowrt::text::tokenizer::load_tokenizer;
    use std::path::PathBuf;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage").into();
    let ckpt: PathBuf = args.next().expect("usage").into();
    let mut prompt = String::from("The capital of France is");
    let mut filter = String::from("act.");
    let mut decode_steps = 0usize;
    let mut prompt_tokens = 0usize;
    let mut dump_logits: Option<PathBuf> = None;
    let mut opts = CpuEngineOpts::default();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--prompt" => prompt = args.next().unwrap(),
            "--filter" => filter = args.next().unwrap(),
            "--decode" => decode_steps = args.next().unwrap().parse().unwrap(),
            // Same synthetic prompt as cpu_bench (<bos> + repeated sentence), for tier A/B.
            "--prompt-tokens" => prompt_tokens = args.next().unwrap().parse().unwrap(),
            "--dump-logits" => dump_logits = Some(args.next().unwrap().into()),
            "--threads" => opts.threads = args.next().unwrap().parse().unwrap(),
            "--isa" => {
                opts.isa = match args.next().unwrap().as_str() {
                    "scalar" => plowrt::exec::cpu::ffi::Isa::Scalar,
                    "avx512" => plowrt::exec::cpu::ffi::Isa::Avx512,
                    _ => plowrt::exec::cpu::ffi::Isa::Amx,
                }
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let tok = load_tokenizer(&ckpt);
    let ids = if prompt_tokens > 0 {
        let base = tok.encode_with_special_tokens(
            "The quick brown fox jumps over the lazy dog while the river flows quietly past the old mill. ",
            false,
        );
        let mut ids = vec![2u32];
        while ids.len() < prompt_tokens {
            ids.extend_from_slice(&base);
        }
        ids.truncate(prompt_tokens);
        ids
    } else {
        tok.encode_with_special_tokens(&prompt, true)
    };
    println!("prompt ids ({}): {:?}", ids.len(), &ids[..ids.len().min(16)]);
    let mut eng = CpuEngine::load(&blob, &ckpt, &opts).expect("load");
    println!("isa={:?} threads={}", eng.isa, eng.threads);
    let first = eng.prefill(&ids).expect("prefill");
    println!("prefill -> token {first} {:?}", tok.decode(&[first]));
    if let Some(p) = &dump_logits {
        let h = eng.model().wk.logits.expect("act.logits");
        // SAFETY: quiescent.
        let bytes = unsafe { eng.model().tensor(h).as_slice() };
        std::fs::write(p, bytes).expect("write logits");
        println!("dumped {} bytes of act.logits to {}", bytes.len(), p.display());
    }
    dump(eng.model(), &filter);
    let mut pos = ids.len() as u32;
    for s in 0..decode_steps {
        let t = eng.decode_step(pos, pos + 1).expect("decode");
        println!("decode step {s} -> token {t} {:?}", tok.decode(&[t]));
        pos += 1;
    }
    if decode_steps > 0 {
        dump(eng.model(), &filter);
    }
}

#[cfg(feature = "cpu")]
fn dump(m: &plowrt::exec::cpu::engine::CpuModel, filter: &str) {
    println!("{:<40} {:>10} | {:>7} {:>7} {:>11} {:>11} | {:>7} {:>7} {:>11} {:>11}",
        "tensor", "bytes", "bf16nan", "bf16inf", "bf16min", "bf16max", "f32nan", "f32inf", "f32min", "f32max");
    for (h, name) in m.names.iter().enumerate() {
        if !name.starts_with(filter) && !name.starts_with("in.") {
            continue;
        }
        let t = m.tensor(h);
        // SAFETY: quiescent (no run in flight).
        let s = unsafe { t.as_slice() };
        let (mut bn, mut bi, mut bmin, mut bmax) = (0usize, 0usize, f32::INFINITY, f32::NEG_INFINITY);
        for c in s.chunks_exact(2) {
            let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            if v.is_nan() { bn += 1 } else if v.is_infinite() { bi += 1 } else { bmin = bmin.min(v); bmax = bmax.max(v) }
        }
        let (mut fn_, mut fi, mut fmin, mut fmax) = (0usize, 0usize, f32::INFINITY, f32::NEG_INFINITY);
        for c in s.chunks_exact(4) {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if v.is_nan() { fn_ += 1 } else if v.is_infinite() { fi += 1 } else { fmin = fmin.min(v); fmax = fmax.max(v) }
        }
        println!("{:<40} {:>10} | {:>7} {:>7} {:>11.4} {:>11.4} | {:>7} {:>7} {:>11.4} {:>11.4}",
            name, t.bytes, bn, bi, bmin, bmax, fn_, fi, fmin, fmax);
        if name == "in.ids" || name == "in.pos" || name == "in.kvlen" {
            let v: Vec<u32> = s.chunks_exact(4).take(8).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            println!("    first u32s: {v:?}");
        }
        if name == "act.logits" {
            // Host argmax over the first row, as f32 and as bf16.
            let f: Vec<f32> = s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let (mut bi_, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in f.iter().enumerate() { if v > bv { bv = v; bi_ = i } }
            println!("    host argmax(f32 view): id {bi_} value {bv}");
            let b: Vec<f32> = s.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
            let (mut bi2, mut bv2) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in b.iter().enumerate() { if v > bv2 { bv2 = v; bi2 = i } }
            println!("    host argmax(bf16 view): id {bi2} value {bv2}");
        }
    }
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}

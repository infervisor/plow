//! Parity gate for a heterogeneous blob against its GPU-only twin: prefill the same prompt on
//! both, compare the last row's logits (max abs diff, top-k agreement) and the greedy token.
//! Text can legitimately diverge after a near-tie argmax; the logits say how far apart the
//! two computations really are.
//!
//! `cargo run --release --features ane --example apple_hetero_check -- <hetero.pkt> <gpu-only.pkt> <ckpt> [--prompt-file f]`

#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    use plowrt::exec::apple::MetalEngine;
    use plowrt::text::tokenizer::load_tokenizer;
    use std::path::PathBuf;

    let mut args = std::env::args().skip(1);
    let a: PathBuf = args
        .next()
        .expect("usage: <hetero.pkt> <gpu-only.pkt> <ckpt>")
        .into();
    let b: PathBuf = args.next().expect("usage").into();
    let ckpt: PathBuf = args.next().expect("usage").into();
    let mut cpu_ref = false;
    let mut prompt = String::from(
        "In the early history of computing, machines were built from vacuum tubes and relays, and \
         programs were entered by rewiring plugboards. The invention of the transistor changed \
         everything: computers became smaller, faster and more reliable, and by the nineteen \
         seventies integrated circuits had placed an entire processor on a single chip. Today a \
         phone carries more computing power than the room-sized systems of that era. The next \
         chapter of this story is likely to be written by",
    );
    while let Some(x) = args.next() {
        match x.as_str() {
            "--prompt-file" => prompt = std::fs::read_to_string(args.next().unwrap()).unwrap(),
            "--cpu-ref" => cpu_ref = true,
            other => panic!("unknown arg {other}"),
        }
    }
    let tok = load_tokenizer(&ckpt);
    let ids = tok.encode_with_special_tokens(&prompt, true);
    println!("prompt: {} tokens", ids.len());
    let logits_of = |blob: &PathBuf| -> (u32, Vec<f32>) {
        let mut eng = MetalEngine::load(blob, &ckpt).expect("load");
        let first = eng.prefill(&ids).expect("prefill");
        let h = eng.model.wk.logits.expect("act.logits");
        let bytes = eng.tensor_bytes(h);
        let bytes = &bytes[..bytes.len() / eng.model.batch];
        // bf16 unless the tensor is exactly vocab*4 wide — read as bf16 (the dense path's lm_head).
        let v: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        (first, v)
    };
    let (fa, la) = logits_of(&a);
    let (fb, lb) = if cpu_ref {
        use plowrt::exec::cpu::engine::{CpuEngine, CpuEngineOpts};
        let mut opts = CpuEngineOpts::default();
        opts.threads = 8;
        let mut cpu = CpuEngine::load(&b, &ckpt, &opts).expect("cpu load");
        let first = cpu.prefill(&ids).expect("cpu prefill");
        let h = cpu.model().wk.logits.expect("act.logits");
        let bytes = unsafe { cpu.model().tensor(h).as_slice() };
        let bytes = &bytes[..bytes.len() / cpu.model().batch];
        let v: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        (first, v)
    } else {
        logits_of(&b)
    };
    let n = la.len().min(lb.len());
    let (la, lb) = (&la[..n], &lb[..n]);
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    let mut norm_b = 0f64;
    for i in 0..n {
        let d = (la[i] - lb[i]).abs();
        max_abs = max_abs.max(d);
        sum_sq += (d as f64) * (d as f64);
        norm_b += (lb[i] as f64) * (lb[i] as f64);
    }
    let topk = |v: &[f32], k: usize| -> Vec<usize> {
        let mut ix: Vec<usize> = (0..v.len()).collect();
        ix.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap());
        ix.truncate(k);
        ix
    };
    let (ta, tb) = (topk(la, 10), topk(lb, 10));
    let overlap = ta.iter().filter(|t| tb.contains(t)).count();
    println!(
        "first token: hetero {fa} {:?} vs gpu-only {fb} {:?}{}",
        tok.decode(&[fa]),
        tok.decode(&[fb]),
        if fa == fb { " (same)" } else { " (DIFFERENT)" }
    );
    println!(
        "logits ({n}): max |diff| {max_abs:.4}, rel L2 {:.5}, top-10 overlap {overlap}/10, top-1 margin gpu-only {:.3}",
        (sum_sq / norm_b.max(1e-30)).sqrt(),
        lb[tb[0]] - lb[tb[1]]
    );
    println!(
        "top-5 hetero  : {:?}",
        ta[..5].iter().map(|&i| (i, la[i])).collect::<Vec<_>>()
    );
    println!(
        "top-5 gpu-only: {:?}",
        tb[..5].iter().map(|&i| (i, lb[i])).collect::<Vec<_>>()
    );
}

#[cfg(not(all(feature = "ane", target_os = "macos")))]
fn main() {
    eprintln!("build with --features ane on macOS");
}

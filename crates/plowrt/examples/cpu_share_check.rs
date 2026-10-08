//! Prefix-share consistency on a `CpuServe`: a prompt prefilled over KV rows copied from another
//! slot vs the same prompt prefilled from scratch. P = prompt rows only (prefill-written); D =
//! the prompt plus `k` generated tokens, whose rows the donor wrote while decoding.
//!
//! `cargo run --release --no-default-features --features cpu --example cpu_share_check -- \
//!     <model.pkt> <ckpt> --gate prompts.json <id> [--k 4]`

#[cfg(feature = "cpu")]
fn main() {
    use plowrt::exec::cpu::engine::CpuEngineOpts;
    use plowrt::serve::cpu_serve::CpuServe;
    use std::path::PathBuf;

    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage").into();
    let ckpt: PathBuf = args.next().expect("usage").into();
    let mut k = 4usize;
    let mut gate: Option<(PathBuf, String)> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--k" => k = args.next().unwrap().parse().unwrap(),
            "--gate" => gate = Some((args.next().unwrap().into(), args.next().unwrap())),
            other => panic!("unknown arg {other}"),
        }
    }
    let (f, id) = gate.expect("--gate prompts.json id");
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(f).unwrap()).unwrap();
    let case = j["cases"].as_array().unwrap().iter().find(|c| c["id"] == id.as_str()).expect("case");
    let prompt: Vec<u32> =
        case["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();

    let mut serve = CpuServe::load(&blob, &ckpt, &CpuEngineOpts::default()).expect("load");
    let logits = |serve: &CpuServe| {
        let mut v = Vec::new();
        assert!(serve.engine().logits_row(0, &mut v));
        v
    };
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let shared = |serve: &mut CpuServe, slot: usize, p: &[u32]| loop {
        if let Some(t) = serve.prefill_chunk(slot, p, u32::MAX).expect("prefill") {
            break t;
        }
    };

    // Donor: slot 0 prefills the prompt and decodes k tokens.
    let mut tok = shared(&mut serve, 0, &prompt);
    let mut gen = Vec::new();
    for _ in 0..k {
        gen.push(tok);
        tok = serve.step(0, tok).expect("step");
    }
    serve.release(0);

    let kl = |a: &[f32], b: &[f32]| {
        let lse = |v: &[f32]| {
            let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
        };
        let (la, lb) = (lse(a), lse(b));
        a.iter().zip(b).map(|(&x, &y)| (x as f64 - la).exp() * ((x as f64 - la) - (y as f64 - lb))).sum::<f64>()
    };
    let mut check = |serve: &mut CpuServe, name: &str, q: &[u32], slot: usize| {
        let ts = shared(serve, slot, q);
        let ls = logits(serve);
        let tf = serve.prefill(slot + 1, q).expect("fresh prefill");
        let lf = logits(serve);
        println!(
            "{name}: {} rows: first token shared/fresh {ts}/{tf}  max|dlogit| {:.4}  KL(fresh||shared) {:.2e}",
            q.len(),
            diff(&ls, &lf),
            kl(&lf, &ls)
        );
        serve.release(slot + 1);
        ts
    };
    // P: only the donor's prompt rows match. D: its prompt and k decoded rows. C: a chain, the
    // way teacher-forced requests reuse each other: D's slot decodes k more tokens, and a third
    // request extends that history by k / 2 of them.
    check(&mut serve, "P", &[&prompt[..], &[gen[0] ^ 1]].concat(), 1);
    serve.release(1);
    let qd = [&prompt[..], &gen[..], &[tok ^ 1]].concat();
    let mut t = check(&mut serve, "D", &qd, 1);
    let mut gen1 = Vec::new();
    for _ in 0..k {
        gen1.push(t);
        t = serve.step(1, t).expect("step");
    }
    serve.release(1);
    check(&mut serve, "C", &[&qd[..], &gen1[..k / 2], &[gen1[k / 2] ^ 1]].concat(), 3);
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}

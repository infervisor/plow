//! Decode-rung consistency: the same sequences decoded on the narrowest rung (slots `0..n`), on
//! the widest rung (slots `0..n`), and on the widest rung from the top slots, teacher-forced with
//! the narrow run's tokens. Prints each run's worst logit distance from the narrow one per step.
//!
//! `cargo run --release --no-default-features --features cpu --example cpu_rung_check -- \
//!     <model.pkt> <ckpt> --gate prompts.json id,id,.. [--steps 16]`

#[cfg(feature = "cpu")]
fn main() {
    use plowrt::exec::cpu::engine::{CpuEngine, CpuEngineOpts};
    use std::path::PathBuf;

    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage").into();
    let ckpt: PathBuf = args.next().expect("usage").into();
    let mut steps = 16usize;
    let mut gate: Option<(PathBuf, Vec<String>)> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--steps" => steps = args.next().unwrap().parse().unwrap(),
            "--gate" => {
                let f = args.next().unwrap().into();
                gate = Some((f, args.next().unwrap().split(',').map(String::from).collect()))
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let (f, ids) = gate.expect("--gate prompts.json id,..");
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(f).unwrap()).unwrap();
    let cases = j["cases"].as_array().unwrap();
    let prompts: Vec<Vec<u32>> = ids
        .iter()
        .map(|id| {
            let c = cases.iter().find(|c| c["id"] == id.as_str()).expect("gate case id");
            c["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
        })
        .collect();

    let mut eng = CpuEngine::load(&blob, &ckpt, &CpuEngineOpts::default()).expect("load");
    let batch = eng.model().batch;
    let n = prompts.len();
    assert!(2 * n <= batch, "{n} prompts need {} slots, packet has {batch}", 2 * n);
    let narrow = eng.model().decode_prog_for(n);
    let wide = eng.model().decode_prog_for(batch);
    let rung = |eng: &CpuEngine, p: usize| eng.model().blob.progs[p].t;
    println!("rungs: narrow {} wide {}", rung(&eng, narrow), rung(&eng, wide));

    // Decode `prompts` from slots `slots`, on program `dp`; `force` = the tokens to feed (None =
    // greedy). Returns per step the tokens and each sequence's logits.
    let run = |eng: &mut CpuEngine, slots: &[usize], dp: usize, force: Option<&Vec<Vec<u32>>>| {
        let mut first = Vec::new();
        for (&s, p) in slots.iter().zip(&prompts) {
            first.push(eng.prefill_slot(s, p).expect("prefill"));
        }
        let mut pos = vec![0u32; batch];
        let mut ids = vec![0u32; batch];
        for (i, &s) in slots.iter().enumerate() {
            pos[s] = prompts[i].len() as u32;
            ids[s] = first[i];
        }
        let mut toks = vec![first.clone()];
        let mut lgs = Vec::new();
        for step in 0..steps {
            if let Some(f) = force {
                for (i, &s) in slots.iter().enumerate() {
                    ids[s] = f[step][i];
                }
            }
            let kv: Vec<u32> = pos.iter().map(|p| p + 1).collect();
            let next = eng.decode_step_batched_at(&pos, &kv, &ids, dp).expect("decode");
            let mut row_lg = Vec::new();
            for &s in slots {
                let mut v = Vec::new();
                assert!(eng.logits_row(s, &mut v));
                row_lg.push(v);
            }
            lgs.push(row_lg);
            toks.push(slots.iter().map(|&s| next[s]).collect());
            for &s in slots {
                ids[s] = next[s];
                pos[s] += 1;
            }
        }
        (toks, lgs)
    };
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);

    let low: Vec<usize> = (0..n).collect();
    let high: Vec<usize> = (batch - n..batch).collect();
    let (tl, ll) = run(&mut eng, &low, narrow, None);
    let (_, lm) = run(&mut eng, &low, wide, Some(&tl));
    let (_, lh) = run(&mut eng, &high, wide, Some(&tl));
    let mut worst = (0.0f32, 0.0f32);
    for step in 0..steps {
        let dm = (0..n).map(|i| diff(&ll[step][i], &lm[step][i])).fold(0.0f32, f32::max);
        let dh = (0..n).map(|i| diff(&ll[step][i], &lh[step][i])).fold(0.0f32, f32::max);
        worst = (worst.0.max(dm), worst.1.max(dh));
        println!("step {step:>2}: wide rung, low slots max|dlogit| {dm:.4}   wide rung, top slots {dh:.4}");
    }
    println!("worst: low slots {:.4} top slots {:.4}", worst.0, worst.1);
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}

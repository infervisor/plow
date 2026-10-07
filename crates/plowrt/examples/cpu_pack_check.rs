//! Packed (cross-request) CPU prefill vs one prefill per slot.
//!
//! `cargo run --release --no-default-features --features cpu --example cpu_pack_check -- \
//!     <model.pkt> <ckpt> [--lens 300,500,77,1000 | --gate prompts.json id,id,..] [--steps 16]`
//!
//! A = one `prefill_slot` per prompt; B = every prompt in one `prefill_packed`; C = two packs,
//! the first carrying an intermediate chunk of prompt 0 and the second its tail at `c0 > 0`;
//! D = each prompt alone in a pack; A' = each prompt alone, unpacked, on B's bucket. Passes iff
//! D == A and B == A' exactly (logits, first token, greedy decode): packing changes nothing but
//! the bucket, whose own effect (A' vs A) is printed as the noise floor.

#[cfg(feature = "cpu")]
fn main() {
    use plowrt::exec::cpu::engine::{CpuEngine, CpuEngineOpts, PackMember};
    use plowrt::text::tokenizer::load_tokenizer;
    use std::path::PathBuf;
    use std::time::Instant;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage").into();
    let ckpt: PathBuf = args.next().expect("usage").into();
    let mut lens = vec![300usize, 500, 77, 1000];
    let mut steps = 16usize;
    let mut gate: Option<(PathBuf, Vec<String>)> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--lens" => {
                lens = args.next().unwrap().split(',').map(|x| x.parse().unwrap()).collect()
            }
            "--steps" => steps = args.next().unwrap().parse().unwrap(),
            // `--gate <prompts.json> <id,id,..>`: the FP32-gate cases' token ids instead of `--lens`.
            "--gate" => {
                let f = args.next().unwrap().into();
                gate = Some((f, args.next().unwrap().split(',').map(String::from).collect()))
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let tok = load_tokenizer(&ckpt);
    let text = "The mill's ledgers record more than flour: the weather on every delivery day, the price \
        of candles, and the names of children hired to pick stones. In the dry year the river fell so \
        low that the wheel stood still for nine weeks, and the miller wrote only the word 'waiting'. ";
    let base = tok.encode_with_special_tokens(text, false);
    let prompts: Vec<Vec<u32>> = if let Some((f, ids)) = &gate {
        let j: serde_json::Value = serde_json::from_slice(&std::fs::read(f).unwrap()).unwrap();
        let cases = j["cases"].as_array().unwrap();
        ids.iter()
            .map(|id| {
                let c = cases.iter().find(|c| c["id"] == id.as_str()).expect("gate case id");
                c["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
            })
            .collect()
    } else {
        lens
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let mut ids = vec![2u32];
            let mut k = i * 13;
            while ids.len() < n {
                ids.push(base[k % base.len()]);
                k += 1 + i;
            }
            ids
        })
        .collect()
    };
    lens = prompts.iter().map(Vec::len).collect();

    let mut eng = CpuEngine::load(&blob, &ckpt, &CpuEngineOpts::default()).expect("load");
    assert!(eng.packs(), "this packet has no packed prefill route (see the load log)");
    let batch = eng.model().batch;
    let n = prompts.len();
    assert!(n <= batch, "{n} prompts but {batch} slots");

    let logits = |eng: &CpuEngine, row: usize| {
        let mut v = Vec::new();
        assert!(eng.logits_row(row, &mut v));
        v
    };
    let decode = |eng: &mut CpuEngine, first: &[u32]| {
        let mut pos = vec![0u32; batch];
        let mut ids = vec![0u32; batch];
        for i in 0..n {
            pos[i] = prompts[i].len() as u32;
            ids[i] = first[i];
        }
        let dp = eng.model().decode_prog_for(n);
        let mut out = vec![Vec::new(); n];
        for _ in 0..steps {
            let kv: Vec<u32> = pos.iter().map(|p| p + 1).collect();
            let next = eng.decode_step_batched_at(&pos, &kv, &ids, dp).expect("decode");
            for i in 0..n {
                out[i].push(next[i]);
                ids[i] = next[i];
                pos[i] += 1;
            }
        }
        out
    };
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let kl = |a: &[f32], b: &[f32]| {
        let lse = |v: &[f32]| {
            let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
        };
        let (la, lb) = (lse(a), lse(b));
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let lpa = x as f64 - la;
                lpa.exp() * (lpa - (y as f64 - lb))
            })
            .sum::<f64>()
    };

    // A: one prefill per slot.
    let t = Instant::now();
    let mut tok_a = Vec::new();
    let mut lg_a = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        tok_a.push(eng.prefill_slot(i, p).expect("prefill"));
        lg_a.push(logits(&eng, 0));
    }
    let ms_a = t.elapsed().as_secs_f64() * 1e3;
    let dec_a = decode(&mut eng, &tok_a);

    // B: every prompt in one launch.
    let members: Vec<PackMember> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| PackMember { slot: i, c0: 0, rows: p, sample: true })
        .collect();
    let t = Instant::now();
    let tok_b = eng.prefill_packed(&members).expect("packed prefill");
    let ms_b = t.elapsed().as_secs_f64() * 1e3;
    let lg_b: Vec<Vec<f32>> = (0..n).map(|i| logits(&eng, i)).collect();
    let dec_b = decode(&mut eng, &tok_b);

    // C: prompt 0 split across two packs.
    let cut = prompts[0].len() / 2;
    let mut first = vec![PackMember { slot: 0, c0: 0, rows: &prompts[0][..cut], sample: false }];
    let mut second =
        vec![PackMember { slot: 0, c0: cut as u32, rows: &prompts[0][cut..], sample: true }];
    for (i, p) in prompts.iter().enumerate().skip(1) {
        let m = PackMember { slot: i, c0: 0, rows: p, sample: true };
        if i % 2 == 1 { first.push(m) } else { second.push(m) }
    }
    let mut tok_c = vec![0u32; n];
    let t1 = eng.prefill_packed(&first).expect("pack 1");
    for (m, t) in first.iter().filter(|m| m.sample).zip(t1) {
        tok_c[m.slot] = t;
    }
    let t2 = eng.prefill_packed(&second).expect("pack 2");
    for (m, t) in second.iter().filter(|m| m.sample).zip(t2) {
        tok_c[m.slot] = t;
    }
    let dec_c = decode(&mut eng, &tok_c);

    // D: each prompt alone in a pack (same bucket and M as A): must be bit-exact with A.
    // A': each prompt alone, unpacked, through the bucket pack B ran: B must equal it exactly.
    // A' vs A is the bucket-choice noise that exists without packing.
    let total: u32 = lens.iter().map(|&l| l as u32).sum();
    let widest = eng
        .prefill_buckets()
        .into_iter()
        .filter(|&(_, t)| t >= total)
        .min_by_key(|&(_, t)| t)
        .expect("a bucket holds pack B");
    let mut lg_d = Vec::new();
    let mut tok_d = Vec::new();
    let mut lg_w = Vec::new();
    let mut tok_w = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let m = [PackMember { slot: i, c0: 0, rows: p, sample: true }];
        tok_d.push(eng.prefill_packed(&m).expect("single pack")[0]);
        lg_d.push(logits(&eng, 0));
        let ch = plowrt::exec::cpu::engine::Chunk { prog: widest.0, c0: 0, clen: p.len() as u32 };
        eng.prefill_slot_chunk(i, p, ch).expect("wide prefill");
        tok_w.push(eng.last_token().unwrap());
        lg_w.push(logits(&eng, 0));
    }
    let dec_w = decode(&mut eng, &tok_w);
    for i in 0..n {
        println!(
            "prompt {i}: single-pack D vs A max|dlogit| {:.4} tok {}  | wide-bucket A' vs A max|dlogit| {:.4} KL {:.2e} tok {} decode {}",
            diff(&lg_a[i], &lg_d[i]),
            if tok_a[i] == tok_d[i] { "same" } else { "DIFFERS" },
            diff(&lg_a[i], &lg_w[i]),
            kl(&lg_a[i], &lg_w[i]),
            if tok_a[i] == tok_w[i] { "same" } else { "DIFFERS" },
            if dec_a[i] == dec_w[i] { "same" } else { "DIFFERS" },
        );
    }

    let rows: usize = lens.iter().sum();
    println!(
        "prefill {rows} rows: sequential {ms_a:.1} ms, packed {ms_b:.1} ms ({:.2}x)",
        ms_a / ms_b
    );
    let mut ok = true;
    for i in 0..n {
        ok &= tok_d[i] == tok_a[i] && diff(&lg_d[i], &lg_a[i]) == 0.0;
        ok &= tok_b[i] == tok_w[i] && diff(&lg_b[i], &lg_w[i]) == 0.0 && dec_b[i] == dec_w[i];
        println!(
            "prompt {i} len {:>5}: first A/B/C {}/{}/{}  B vs A max|dlogit| {:.4} KL {:.2e}  B vs A' {:.4}  decode B {} C {} (vs A)",
            prompts[i].len(),
            tok_a[i],
            tok_b[i],
            tok_c[i],
            diff(&lg_a[i], &lg_b[i]),
            kl(&lg_a[i], &lg_b[i]),
            diff(&lg_b[i], &lg_w[i]),
            if dec_a[i] == dec_b[i] { "same" } else { "DIFFERS" },
            if dec_a[i] == dec_c[i] { "same" } else { "DIFFERS" },
        );
    }
    println!("{}", if ok { "PACK_CHECK OK" } else { "PACK_CHECK MISMATCH" });
    std::process::exit(if ok { 0 } else { 1 });
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("cpu_pack_check needs --features cpu");
}

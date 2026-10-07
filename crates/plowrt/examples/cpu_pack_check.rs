//! Packed (cross-request) CPU prefill vs one prefill per slot.
//!
//! `cargo run --release --no-default-features --features cpu --example cpu_pack_check -- \
//!     <model.pkt> <ckpt> [--lens 300,500,77,1000 | --gate prompts.json id,id,..] [--steps 16]`
//!
//! A = one `prefill_slot` per prompt; B = the prompts packed into as few launches as fit; C = two
//! packs, the first carrying an intermediate chunk of prompt 0 and the second its tail at `c0 > 0`;
//! D = each prompt alone in a pack. Passes iff B, C and D are bit-identical to the same chunks
//! prefilled alone (logits, first token, B's greedy decode): packing is batch-invariant. The
//! distance of an unpacked run on the widest packed bucket from A is printed for scale.

#[cfg(feature = "cpu")]
fn main() {
    use plowrt::exec::cpu::engine::{Chunk, CpuEngine, CpuEngineOpts, PackMember};
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
    assert!(eng.pack_rows().is_some(), "this packet has no packed prefill route (see the load log)");
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

    // B: the prompts packed greedily into launches of at most `pack_rows` rows.
    let cap = eng.pack_rows().unwrap() as usize;
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        match groups.last_mut() {
            Some(g) if g.iter().map(|&j| prompts[j].len()).sum::<usize>() + p.len() <= cap => g.push(i),
            _ => groups.push(vec![i]),
        }
    }
    let mut tok_b = vec![0u32; n];
    let mut lg_b = vec![Vec::new(); n];
    let t = Instant::now();
    for g in &groups {
        let m: Vec<PackMember> =
            g.iter().map(|&i| PackMember { slot: i, c0: 0, rows: &prompts[i], sample: true }).collect();
        for (j, (&i, tok)) in g.iter().zip(eng.prefill_packed(&m).expect("packed prefill")).enumerate() {
            tok_b[i] = tok;
            lg_b[i] = logits(&eng, j);
        }
    }
    let ms_b = t.elapsed().as_secs_f64() * 1e3;
    let dec_b = decode(&mut eng, &tok_b);

    // C: prompt 0 as two chunks, each riding a pack (an intermediate chunk, then its tail at
    // c0 > 0), against the same two chunks prefilled alone.
    let solo = |eng: &CpuEngine, rows: usize| {
        eng.prefill_buckets()
            .into_iter()
            .filter(|&(_, t)| t as usize >= rows)
            .min_by_key(|&(_, t)| t)
            .unwrap()
            .0
    };
    let (p0, cut) = (&prompts[0], prompts[0].len() / 2);
    for (c0, clen) in [(0, cut), (cut, p0.len() - cut)] {
        let ch = Chunk { prog: solo(&eng, clen), c0: c0 as u32, clen: clen as u32 };
        eng.prefill_slot_chunk(0, p0, ch).expect("solo chunk");
    }
    let (tok_c0, lg_c0) = (eng.last_token().unwrap(), logits(&eng, 0));
    fn fill<'a>(head: PackMember<'a>, parity: usize, prompts: &'a [Vec<u32>], cap: usize) -> Vec<PackMember<'a>> {
        let mut v = vec![head];
        let mut rows = head.rows.len();
        for (i, p) in prompts.iter().enumerate().skip(1).filter(|(i, _)| i % 2 == parity) {
            if rows + p.len() <= cap {
                rows += p.len();
                v.push(PackMember { slot: i, c0: 0, rows: p.as_slice(), sample: true });
            }
        }
        v
    }
    let first = fill(PackMember { slot: 0, c0: 0, rows: &p0[..cut], sample: false }, 1, &prompts, cap);
    let second = fill(PackMember { slot: 0, c0: cut as u32, rows: &p0[cut..], sample: true }, 0, &prompts, cap);
    let mut c_ok = true;
    for pack in [&first, &second] {
        let toks = eng.prefill_packed(pack).expect("pack C");
        for (j, (m, tok)) in pack.iter().filter(|m| m.sample).zip(toks).enumerate() {
            let lg = logits(&eng, j);
            let (rt, rl) = if m.slot == 0 { (tok_c0, &lg_c0) } else { (tok_a[m.slot], &lg_a[m.slot]) };
            c_ok &= tok == rt && diff(&lg, rl) == 0.0;
        }
    }

    // D: each prompt alone in a pack. W: each prompt alone, unpacked, on the widest packed bucket
    // (what packing without batch invariance would give; its distance from A is printed).
    let wide = solo(&eng, cap);
    let mut lg_d = Vec::new();
    let mut tok_d = Vec::new();
    let mut lg_w = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let m = [PackMember { slot: i, c0: 0, rows: p, sample: true }];
        tok_d.push(eng.prefill_packed(&m).expect("single pack")[0]);
        lg_d.push(logits(&eng, 0));
        let ch = Chunk { prog: wide, c0: 0, clen: p.len() as u32 };
        eng.prefill_slot_chunk(i, p, ch).expect("wide prefill");
        lg_w.push(logits(&eng, 0));
    }

    let rows: usize = lens.iter().sum();
    println!(
        "prefill {rows} rows in {} pack(s) of <= {cap}: sequential {ms_a:.1} ms, packed {ms_b:.1} ms ({:.2}x)",
        groups.len(),
        ms_a / ms_b
    );
    let mut ok = c_ok;
    for i in 0..n {
        let exact = tok_b[i] == tok_a[i] && diff(&lg_b[i], &lg_a[i]) == 0.0 && dec_b[i] == dec_a[i];
        ok &= exact && tok_d[i] == tok_a[i] && diff(&lg_d[i], &lg_a[i]) == 0.0;
        println!(
            "prompt {i} len {:>5}: first A/B {}/{}  B vs A max|dlogit| {:.4} decode {}  D vs A {:.4}  | unpacked wide bucket vs A {:.4} KL {:.2e}",
            prompts[i].len(),
            tok_a[i],
            tok_b[i],
            diff(&lg_a[i], &lg_b[i]),
            if dec_a[i] == dec_b[i] { "same" } else { "DIFFERS" },
            diff(&lg_a[i], &lg_d[i]),
            diff(&lg_a[i], &lg_w[i]),
            kl(&lg_a[i], &lg_w[i]),
        );
    }
    println!("split-chunk packs (C): {}", if c_ok { "exact" } else { "DIFFER" });
    println!("{}", if ok { "PACK_CHECK OK" } else { "PACK_CHECK MISMATCH" });
    std::process::exit(if ok { 0 } else { 1 });
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("cpu_pack_check needs --features cpu");
}

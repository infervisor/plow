//! Chatterbox T3 numerics gate and step timing against `scripts/tts/t3_ref.py` output.
//!
//!   t3_check <t3-assets> <t3_ref.json> [--jobs N --sample]
//!
//! Per prompt: text ids equal the reference tokenizer's; last-prefill logits of both CFG members
//! (rel-L2, top-1 agreement) against the fp32 reference; greedy guided tokens agree for the
//! reference's steps. Then N sampled jobs run concurrently to report ms per decode step.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("t3_check requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::tts::t3::{T3Engine, T3Job};
    let mut args = std::env::args().skip(1);
    let assets = std::path::PathBuf::from(args.next().ok_or("usage: t3_check <assets> <ref.json>")?);
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(args.next().ok_or("ref.json")?)?)?;
    let mut jobs_n = 8usize;
    let mut sample = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--jobs" => jobs_n = args.next().ok_or("--jobs N")?.parse()?,
            "--sample" => sample = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let mut t3 = T3Engine::load(&assets, 0)?;
    println!("T3 engine: pairs={} contract={:?}", t3.pairs(), t3.c);
    let rel = |a: &[f32], b: &[f64]| {
        let (mut num, mut den) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            num += (*x as f64 - y).powi(2);
            den += y * y;
        }
        (num / den).sqrt()
    };
    let argmax = |v: &[f64]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap_or(0);
    let mut ok = true;
    for r in reference.as_array().ok_or("ref is a list")? {
        let text = r["text"].as_str().ok_or("text")?;
        let want_ids: Vec<u32> = serde_json::from_value(r["text_ids"].clone())?;
        let cref: Vec<f64> = serde_json::from_value(r["cond_logits"].clone())?;
        let uref: Vec<f64> = serde_json::from_value(r["uncond_logits"].clone())?;
        let (ids, cond, uncond) = t3.probe_prefill("default", text)?;
        let (rc, ru) = (rel(&cond, &cref), rel(&uncond, &uref));
        let am = |v: &[f32]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap_or(0);
        let top1 = am(&cond) == argmax(&cref) && am(&uncond) == argmax(&uref);
        let want: Vec<u32> = serde_json::from_value(r["greedy_cfg"].clone())?;
        let mut got = Vec::new();
        t3.run(
            &[T3Job { voice: "default".into(), text: text.into(), seed: None, max_tokens: Some(want.len()) }],
            |_, o| got = o.tokens,
        )?;
        let agree = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        let ids_ok = ids == want_ids;
        println!(
            "ids {} | logits rel-L2 cond {rc:.4} uncond {ru:.4} top1 {} | greedy agree {agree}/{} | {:?}",
            if ids_ok { "ok" } else { "MISMATCH" },
            if top1 { "ok" } else { "DIFF" },
            want.len(),
            &text[..text.len().min(40)]
        );
        ok &= ids_ok && rc < 0.05 && ru < 0.05 && top1;
    }
    println!("GATE {}", if ok { "PASS" } else { "FAIL" });
    if sample {
        let texts: Vec<String> = reference
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["text"].as_str().unwrap_or_default().to_string())
            .collect();
        let jobs: Vec<T3Job> = (0..jobs_n)
            .map(|i| T3Job { voice: "default".into(), text: texts[i % texts.len()].clone(), seed: Some(i as u64 + 1), max_tokens: None })
            .collect();
        let t0 = std::time::Instant::now();
        let mut total_steps = 0;
        let mut outs = vec![Vec::new(); jobs.len()];
        t3.run(&jobs, |i, o| {
            total_steps += o.steps;
            println!("job {i}: {} tokens ({:.2} s audio) prefill {:.1} ms decode {:.1} ms", o.steps, o.steps as f64 / 25.0, o.prefill_us as f64 / 1e3, o.decode_us as f64 / 1e3);
            outs[i] = o.tokens;
        })?;
        let wall = t0.elapsed().as_secs_f64();
        println!(
            "{} jobs: {total_steps} tokens in {wall:.2} s = {:.0} tok/s, {:.2} audio s/s",
            jobs.len(),
            total_steps as f64 / wall,
            total_steps as f64 / 25.0 / wall
        );
        std::fs::write("/tmp/t3_tokens.json", serde_json::to_vec(&outs)?)?;
    }
    Ok(())
}

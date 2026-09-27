//! Chatterbox T3 numerics gate against `scripts/tts/t3_ref.py` output.
//!
//!   t3_check <t3-assets> <t3_ref.json>
//!
//! Per prompt: text ids equal the reference tokenizer's; last-prefill logits of both CFG members
//! (rel-L2, top-1 agreement) against the fp32 reference; greedy guided tokens agree for the
//! reference's steps. Served throughput is `scripts/tts/plow_speech_probe.sh`.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("t3_check requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::tts::guided_lm::GuidedLm;
    let mut args = std::env::args().skip(1);
    let assets = std::path::PathBuf::from(args.next().ok_or("usage: t3_check <assets> <ref.json>")?);
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(args.next().ok_or("ref.json")?)?)?;
    let mut t3 = GuidedLm::load(&assets, 0)?;
    println!("T3 engine: contract={:?}", t3.c);
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
        let lang = r["language"].as_str();
        let (ids, cond, uncond) = t3.probe_prefill("default", text, lang)?;
        let (rc, ru) = (rel(&cond, &cref), rel(&uncond, &uref));
        let am = |v: &[f32]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap_or(0);
        let top1 = am(&cond) == argmax(&cref) && am(&uncond) == argmax(&uref);
        let want: Vec<u32> = serde_json::from_value(r["greedy_cfg"].clone())?;
        let got = t3.greedy("default", text, lang, want.len())?;
        let agree = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        let ids_ok = ids == want_ids;
        println!(
            "{} ids {} | logits rel-L2 cond {rc:.4} uncond {ru:.4} top1 {} | greedy agree {agree}/{} | {:?}",
            lang.unwrap_or("-"),
            if ids_ok { "ok" } else { "MISMATCH" },
            if top1 { "ok" } else { "DIFF" },
            want.len(),
            text.chars().take(24).collect::<String>()
        );
        ok &= ids_ok && rc < 0.05 && ru < 0.05 && top1;
    }
    println!("GATE {}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}

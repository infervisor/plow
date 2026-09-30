//! T3 text frontend gate (CPU): the packet's text rules + tokenizer against reference ids.
//!
//!   t3_text_check <t3-assets> <ref.json>
//!
//! `ref.json`: `[{"language", "text", "ids"}]` from `scripts/tts/mtl_text_ref.py`. Prints the
//! exact-match count per language and each mismatch.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("t3_text_check requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::tts::guided_lm::{GuidedLmContract, PromptTables};
    let mut args = std::env::args().skip(1);
    let assets = std::path::PathBuf::from(args.next().ok_or("usage: t3_text_check <assets> <ref.json>")?);
    let rows: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(args.next().ok_or("ref.json")?)?)?;
    let c = GuidedLmContract::load(&assets)?.ok_or("no guided LM pipeline")?;
    let t0 = std::time::Instant::now();
    let tables = PromptTables::load(&assets, c.hidden)?;
    println!("tables loaded in {:.2} s", t0.elapsed().as_secs_f64());
    let mut per: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    let t0 = std::time::Instant::now();
    for r in &rows {
        let lang = r["language"].as_str().ok_or("language")?;
        let text = r["text"].as_str().ok_or("text")?;
        let want: Vec<u32> = serde_json::from_value(r["ids"].clone())?;
        let l = tables.language(Some(lang))?;
        let got = tables.text_ids(text, l.as_deref())?;
        let e = per.entry(lang.to_string()).or_default();
        e.1 += 1;
        if got == want {
            e.0 += 1;
        } else {
            let at = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
            println!("MISMATCH {lang} at {at}: {:?}\n  got  {:?}\n  want {:?}", text, &got[at.saturating_sub(3)..got.len().min(at + 8)], &want[at.saturating_sub(3)..want.len().min(at + 8)]);
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / rows.len().max(1) as f64;
    let (mut ok, mut n) = (0, 0);
    for (l, (a, b)) in &per {
        println!("{l}: {a}/{b}");
        ok += a;
        n += b;
    }
    println!("TOTAL {ok}/{n} exact ({ms:.3} ms/text)\nGATE {}", if ok == n { "PASS" } else { "FAIL" });
    Ok(())
}

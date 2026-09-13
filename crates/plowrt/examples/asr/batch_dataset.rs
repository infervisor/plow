#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::{frontend::decode_wav, qwen::QwenAsr};
    use serde_json::{json, Value};
    use std::{io::Write, path::Path, sync::atomic::AtomicBool, time::Instant};

    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: asr_batch_dataset CHECKPOINT BLOB MANIFEST.jsonl".into());
    }
    let rows: Vec<Value> = std::fs::read_to_string(&args[3])?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let rows: Vec<_> = rows
        .into_iter()
        .filter(|row| row.get("excluded_reason").is_none())
        .collect();
    if rows.is_empty() {
        return Err("manifest has no eligible recordings".into());
    }
    let samples: Vec<_> = rows
        .iter()
        .map(|row| {
            let path = row["audio"].as_str().ok_or("missing audio")?;
            decode_wav(&std::fs::read(path)?).map_err(Box::<dyn std::error::Error>::from)
        })
        .collect::<Result<_, _>>()?;
    let mut engine = QwenAsr::load(Path::new(&args[2]), Path::new(&args[1]))?;
    engine.set_device_handoff(true);
    let batch = engine.batch_capacity();
    if batch < 2 {
        return Err("batch dataset requires a multi-row packet".into());
    }
    let cancel = AtomicBool::new(false);
    let warm: Vec<_> = samples.iter().take(batch).map(Vec::as_slice).collect();
    engine.transcribe_batch(&warm, None, "", &cancel)?;
    for (cohort, indices) in (0..samples.len())
        .collect::<Vec<_>>()
        .chunks(batch)
        .enumerate()
    {
        let audio: Vec<_> = indices
            .iter()
            .map(|&index| samples[index].as_slice())
            .collect();
        let started = Instant::now();
        let results = engine.transcribe_batch(&audio, None, "", &cancel)?;
        let cohort_seconds = started.elapsed().as_secs_f64();
        for (&index, result) in indices.iter().zip(results) {
            println!(
                "{}",
                json!({
                    "kind":"result", "id":rows[index]["id"], "backend":"native-batch",
                    "reference":rows[index]["reference"],
                    "duration_seconds":samples[index].len() as f64 / 16000.0,
                    "seconds":cohort_seconds / indices.len() as f64,
                    "cohort":cohort, "cohort_size":indices.len(), "cohort_seconds":cohort_seconds,
                    "text":result.text, "language":result.language,
                })
            );
            std::io::stdout().flush()?;
        }
    }
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("asr_batch_dataset requires macOS and --features metal");
    std::process::exit(1);
}

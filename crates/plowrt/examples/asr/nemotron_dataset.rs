fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::frontend::decode_wav;
    use plowrt::asr::nemotron::NemotronAsr;
    use plowrt::asr::Transcriber;
    use serde_json::{json, Value};
    use std::{io::Write, path::Path, sync::atomic::AtomicBool, time::Instant};

    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 && args.len() != 6 {
        return Err(
            "usage: asr_nemotron_dataset LIBRARY MODEL MANIFEST.jsonl cpu|GPU_ORDINAL [LANGUAGE]"
                .into(),
        );
    }
    let gpu = match args[4].as_str() {
        "cpu" => None,
        value => Some(value.parse::<u32>()?),
    };
    let language = args.get(5).map_or("en-US", String::as_str);
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
    let mut engine = NemotronAsr::load(Path::new(&args[1]), Path::new(&args[2]), gpu)?;
    let cancel = AtomicBool::new(false);
    let warm = decode_wav(&std::fs::read(
        rows[0]["audio"].as_str().ok_or("missing audio")?,
    )?)?;
    engine.transcribe(&warm, Some(language), "", &cancel)?;
    for row in rows {
        let samples = decode_wav(&std::fs::read(
            row["audio"].as_str().ok_or("missing audio")?,
        )?)?;
        let started = Instant::now();
        let result = engine.transcribe(&samples, Some(language), "", &cancel)?;
        println!(
            "{}",
            json!({
                "kind":"result", "id":row["id"], "backend":"plow-nemo",
                "reference":row["reference"],
                "duration_seconds":samples.len() as f64 / 16000.0,
                "seconds":started.elapsed().as_secs_f64(),
                "text":result.text, "language":result.language,
            })
        );
        std::io::stdout().flush()?;
    }
    Ok(())
}

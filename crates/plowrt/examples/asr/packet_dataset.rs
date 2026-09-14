fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::{frontend::decode_wav, load_packet_transcriber};
    use serde_json::{json, Value};
    use std::{io::Write, path::Path, sync::atomic::AtomicBool, time::Instant};

    let args: Vec<_> = std::env::args().collect();
    if !(4..=6).contains(&args.len()) {
        return Err(
            "usage: asr_packet_dataset PACKET TOKENIZER MANIFEST.jsonl [BACKEND] [LANGUAGE]".into(),
        );
    }
    let backend = args.get(4).map_or("auto", String::as_str);
    let language = args.get(5).map(String::as_str);
    let rows: Vec<Value> = std::fs::read_to_string(&args[3])?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let rows: Vec<_> = rows
        .into_iter()
        .filter(|row| row.get("excluded_reason").is_none())
        .collect();
    let first = rows.first().ok_or("manifest has no eligible recordings")?;
    let mut loaded = load_packet_transcriber(Path::new(&args[1]), Path::new(&args[2]), backend)?;
    let backend = format!("packet-{}-{}", loaded.driver, loaded.backend);
    let cancel = AtomicBool::new(false);
    let warm = decode_wav(&std::fs::read(
        first["audio"].as_str().ok_or("missing audio")?,
    )?)?;
    loaded.engine.transcribe(&warm, language, "", &cancel)?;

    let mut failed = false;
    for row in rows {
        let samples = decode_wav(&std::fs::read(
            row["audio"].as_str().ok_or("missing audio")?,
        )?)?;
        let started = Instant::now();
        let result = loaded.engine.transcribe(&samples, language, "", &cancel);
        let mut output = json!({
            "kind":"result", "id":row["id"], "backend":backend,
            "reference":row["reference"],
            "duration_seconds":samples.len() as f64 / 16000.0,
            "seconds":started.elapsed().as_secs_f64(),
        });
        match result {
            Ok(result) => {
                output["text"] = result.text.into();
                output["language"] = json!(result.language);
            }
            Err(error) => {
                failed = true;
                output["text"] = "".into();
                output["error"] = error.to_string().into();
            }
        }
        println!("{output}");
        std::io::stdout().flush()?;
    }
    if failed {
        return Err("one or more transcriptions failed; retained in results".into());
    }
    Ok(())
}

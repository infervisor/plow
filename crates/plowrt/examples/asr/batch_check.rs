#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::{frontend::decode_wav, qwen::QwenAsr};
    use serde_json::{json, Value};
    use std::{io::Write, path::Path, sync::atomic::AtomicBool, time::Instant};

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: asr_batch_check CHECKPOINT B1_PACKET BATCH_PACKET MANIFEST OUT".into());
    }
    let rows: Vec<Value> = std::fs::read_to_string(&args[4])?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    if rows.len() < 3 {
        return Err("use at least three recordings to test partial cohorts".into());
    }
    let samples: Vec<_> = rows
        .iter()
        .map(|row| {
            decode_wav(&std::fs::read(
                row["audio"].as_str().ok_or("missing audio")?,
            )?)
            .map_err(Box::<dyn std::error::Error>::from)
        })
        .collect::<Result<_, _>>()?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[5])?;
    let cancel = AtomicBool::new(false);
    let mut baseline = QwenAsr::load(Path::new(&args[2]), Path::new(&args[1]))?;
    assert_eq!(
        baseline.batch_capacity(),
        1,
        "expected a B1 reference packet"
    );
    baseline.transcribe(&samples[0], None, "", &cancel)?;
    let mut expected = Vec::new();
    for (i, audio) in samples.iter().enumerate() {
        let start = Instant::now();
        let result = baseline.transcribe(audio, None, "", &cancel)?;
        writeln!(
            file,
            "{}",
            json!({"kind":"baseline","id":rows[i]["id"],"seconds":start.elapsed().as_secs_f64(),"result":result})
        )?;
        expected.push(result);
    }
    drop(baseline);
    let mut engine = QwenAsr::load(Path::new(&args[3]), Path::new(&args[1]))?;
    let batch = engine.batch_capacity();
    if batch < 2 || samples.len() <= batch || samples.len() % batch == 0 {
        return Err(
            "manifest must exercise full and partial cohorts for the packet capacity".into(),
        );
    }
    assert!(engine.transcribe_batch(&[], None, "", &cancel).is_err());
    assert!(engine
        .transcribe_batch(&vec![samples[0].as_slice(); batch + 1], None, "", &cancel)
        .is_err());
    assert!(engine
        .transcribe_batch(&[&samples[0]], None, "", &AtomicBool::new(true))
        .is_err());
    // Fail after slot zero has completed prefill, then verify reuse below.
    assert!(engine
        .transcribe_batch(&[&samples[0], &[]], None, "", &cancel)
        .is_err());
    for device in [false, true] {
        engine.set_device_handoff(device);
        let warmup: Vec<_> = samples[..batch].iter().map(Vec::as_slice).collect();
        engine.transcribe_batch(&warmup, None, "", &cancel)?;
        for reverse in [false, true] {
            let mut order: Vec<_> = (0..samples.len()).collect();
            if reverse {
                order.reverse();
            }
            for indices in order.chunks(batch) {
                let audio: Vec<_> = indices.iter().map(|&i| samples[i].as_slice()).collect();
                let start = Instant::now();
                let results = engine.transcribe_batch(&audio, None, "", &cancel)?;
                let seconds = start.elapsed().as_secs_f64();
                assert_eq!(results.len(), indices.len());
                let matches = results.iter().zip(indices).all(|(r, &i)| {
                    r.text == expected[i].text && r.language == expected[i].language
                });
                writeln!(
                    file,
                    "{}",
                    json!({"kind":"batch","device_handoff":device,"reverse":reverse,"indices":indices,"seconds":seconds,"matches":matches,"results":results})
                )?;
                file.flush()?;
                if !matches {
                    return Err(format!("batch transcript mismatch at {indices:?}").into());
                }
            }
        }
    }
    println!("{} complete-audio batch transcripts match B1; host/device handoff, reversed slot reuse and partial cohorts pass", samples.len() * 4);
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("asr_batch_check requires macOS and the metal feature");
    std::process::exit(1);
}

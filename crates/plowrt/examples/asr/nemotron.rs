fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::frontend::decode_wav;
    use plowrt::asr::nemotron::NemotronAsr;
    use plowrt::asr::Transcriber;
    use std::path::Path;
    use std::sync::atomic::AtomicBool;

    let args: Vec<_> = std::env::args().collect();
    if !(4..=6).contains(&args.len()) {
        return Err("usage: asr_nemotron LIBRARY MODEL AUDIO [cpu|GPU_ORDINAL] [REPEATS]".into());
    }
    let gpu = match args.get(4).map(String::as_str) {
        None | Some("cpu") => None,
        Some(value) => Some(value.parse::<u32>()?),
    };
    let samples = decode_wav(&std::fs::read(&args[3])?)?;
    let mut recognizer = NemotronAsr::load(Path::new(&args[1]), Path::new(&args[2]), gpu)?;
    let repeats = args.get(5).map_or(Ok(1), |value| value.parse::<usize>())?;
    if repeats == 0 {
        return Err("REPEATS must be positive".into());
    }
    let mut transcript = None;
    for repeat in 0..repeats {
        let started = std::time::Instant::now();
        let next = recognizer.transcribe(&samples, Some("en-US"), "", &AtomicBool::new(false))?;
        eprintln!(
            "plow_nemotron repeat={repeat} seconds={:.6}",
            started.elapsed().as_secs_f64()
        );
        if transcript
            .as_ref()
            .is_some_and(|prior: &plowrt::asr::Transcript| {
                prior.text != next.text || prior.language != next.language
            })
        {
            return Err("Nemotron repeat changed transcript".into());
        }
        transcript = Some(next);
    }
    let transcript = transcript.unwrap();
    println!("{}", serde_json::to_string(&transcript)?);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    use plowrt::asset::gguf::GgufFile;

    let args: Vec<_> = std::env::args().collect();
    if !(3..=5).contains(&args.len()) {
        return Err(
            "usage: asr_rnnt_packet PACKET TOKENIZER_GGUF [REPEATS] [INPUT_F32_OR_WAV]".into(),
        );
    }
    let repeats = args.get(3).map_or(Ok(1), |value| value.parse::<usize>())?;
    if repeats == 0 {
        return Err("REPEATS must be positive".into());
    }
    let requested_backend = std::env::var("PLOW_PACKET_BACKEND").unwrap_or_else(|_| "auto".into());
    let mut execution =
        plowrt::asr::rnnt::PacketRnnt::load(Path::new(&args[1]), &requested_backend)?;
    execution.set_profiling(true);
    let (input, valid_frames) = if let Some(path) = args.get(4) {
        let path = Path::new(path);
        if path.extension().and_then(|value| value.to_str()) == Some("wav") {
            let samples = plowrt::asr::frontend::decode_wav(&std::fs::read(path)?)?;
            let features = execution.log_mel_frontend()?.extract(&samples)?;
            let frames = features.frames;
            let mut values = features.values;
            values.resize(execution.input_elements(), 0.0);
            (values, Some(frames))
        } else {
            (read_f32(path)?, None)
        }
    } else {
        (vec![0.0; execution.input_elements()], None)
    };
    let vocabulary = vocabulary(&GgufFile::open(Path::new(&args[2]))?)?;
    let mut timings = Vec::with_capacity(repeats);
    let mut tokens = Vec::new();
    for repeat in 0..repeats {
        let started = std::time::Instant::now();
        let next = if let Some(frames) = valid_frames {
            execution.transcribe_input_frames(&input, frames)?
        } else {
            execution.transcribe_input(&input)?
        };
        timings.push(started.elapsed().as_secs_f64() * 1000.0);
        if repeat == 0 {
            tokens = next;
        } else if next != tokens {
            return Err(format!("RNNT output changed after state reset on repeat {repeat}").into());
        }
    }
    timings.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({
            "pipeline": "transcribe",
            "driver": "rnnt.greedy.v1",
            "backend": execution.backend(),
            "tokens": tokens,
            "text": plowrt::asr::rnnt::detokenize_sentencepiece(&vocabulary, &tokens),
            "total_median_ms": timings[timings.len() / 2],
            "last_profile": execution.last_profile(),
            "repeats": repeats,
        })
    );
    Ok(())
}

fn vocabulary(model: &plowrt::asset::gguf::GgufFile) -> plowrt::Result<Vec<String>> {
    use gguf_rs_lib::format::metadata::MetadataValue;

    let Some(MetadataValue::Array(values)) = model.metadata().data.get("asr.tokenizer.vocab")
    else {
        return Err(plowrt::RuntimeError::Rejected(
            "GGUF ASR vocabulary is missing".into(),
        ));
    };
    values
        .values
        .iter()
        .map(|value| match value {
            MetadataValue::String(value) => Ok(value.clone()),
            _ => Err(plowrt::RuntimeError::Rejected(
                "GGUF ASR vocabulary contains a non-string".into(),
            )),
        })
        .collect()
}

fn read_f32(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("unaligned FP32 file".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect())
}

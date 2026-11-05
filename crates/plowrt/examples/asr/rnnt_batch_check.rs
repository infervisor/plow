//! Batched RNNT decode against batch 1: each recording's tokens from
//! `PacketRnnt::transcribe_batch_input_frames` (recordings in groups of BATCH) must equal its own
//! `transcribe_input_frames`. One JSON line per recording (its batch-1 tokens), then a summary;
//! exits 1 on a mismatch.
//!
//!   asr_rnnt_batch_check PACKET BATCH AUDIO.wav...

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::frontend::decode_wav;
    use plowrt::asr::rnnt::{detokenize, PacketRnnt};
    use serde_json::json;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: asr_rnnt_batch_check PACKET BATCH AUDIO.wav...".into());
    }
    let batch: usize = args[2].parse()?;
    let backend = std::env::var("PLOW_PACKET_BACKEND").unwrap_or_else(|_| "auto".into());
    let mut execution = PacketRnnt::load(Path::new(&args[1]), &backend)?;
    if batch == 0 || batch > execution.batch_capacity() {
        return Err(format!("BATCH must be 1..={}", execution.batch_capacity()).into());
    }
    let frontend = execution.log_mel_frontend()?;
    let mut inputs = Vec::new();
    let mut samples_total = 0;
    for path in &args[3..] {
        let samples = decode_wav(&std::fs::read(path)?)?;
        samples_total += samples.len();
        let features = frontend.extract(&samples)?;
        let mut values = features.values;
        values.resize(execution.input_elements(), 0.0);
        inputs.push((values, features.frames));
    }
    let started = Instant::now();
    let alone = inputs
        .iter()
        .map(|(input, frames)| execution.transcribe_input_frames(input, *frames))
        .collect::<Result<Vec<_>, _>>()?;
    let alone_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut batched = Vec::with_capacity(inputs.len());
    for group in inputs.chunks(batch) {
        let group: Vec<_> = group
            .iter()
            .map(|(input, frames)| (input.as_slice(), *frames))
            .collect();
        batched.extend(execution.transcribe_batch_input_frames(&group)?);
    }
    let batched_s = started.elapsed().as_secs_f64();
    let mut mismatches = 0;
    for ((path, tokens), other) in args[3..].iter().zip(&alone).zip(&batched) {
        let equal = tokens == other;
        mismatches += usize::from(!equal);
        println!(
            "{}",
            json!({
                "audio": path,
                "equal": equal,
                "tokens": tokens,
                "batched": (!equal).then_some(other),
                "text": execution.output().map(|output| detokenize(output, tokens)),
            })
        );
    }
    let audio_s = samples_total as f64 / 16_000.0;
    println!(
        "{}",
        json!({
            "clips": alone.len(),
            "batch": batch,
            "mismatches": mismatches,
            "audio_s": audio_s,
            "alone_s": alone_s,
            "batched_s": batched_s,
            "rtfx_alone": audio_s / alone_s,
            "rtfx_batched": audio_s / batched_s,
        })
    );
    if mismatches != 0 {
        std::process::exit(1);
    }
    Ok(())
}

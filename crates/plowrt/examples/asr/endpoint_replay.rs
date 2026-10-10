//! Replay 16 kHz WAVs through the streaming endpointer, energy and Silero, and report when each
//! closes a turn relative to the end of the speech it contains (the last speech-like frame by
//! energy, the detector-independent reference used here).
//!
//!   asr_endpoint_replay SILERO_VAD_PKT MIN_SILENCE_MS [--dump=OUT.json] WAV...
//!
//! `--dump` writes each detector's turns (`{detector: {wav: [[start, end], ...]}}`, 16 kHz samples).

use std::sync::Arc;

use plowrt::asr::endpoint::{EndpointConfig, Endpointer};
use plowrt::asr::vad::Vad;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: asr_endpoint_replay SILERO_VAD_PKT MIN_SILENCE_MS WAV...".into());
    }
    let vad = Arc::new(Vad::load(std::path::Path::new(&args[1]))?);
    let min_silence_ms: u32 = args[2].parse()?;
    let dump = args[3].strip_prefix("--dump=").map(str::to_owned);
    let wavs = &args[if dump.is_some() { 4 } else { 3 }..];
    let mut segments: [serde_json::Map<String, serde_json::Value>; 2] = Default::default();
    let config = EndpointConfig { min_silence_ms, max_segment_ms: 25_000 };
    let mut delays: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut turns = [0usize; 2];
    for path in wavs {
        let mut reader = hound::WavReader::open(path)?;
        if reader.spec().sample_rate != 16_000 || reader.spec().channels != 1 {
            return Err(format!("{path}: need 16 kHz mono").into());
        }
        let scale = 2f32.powi(reader.spec().bits_per_sample as i32 - 1);
        let samples: Vec<f32> = reader.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?;
        let loud = loud_frames(&samples);
        for (k, endpointer) in [Endpointer::new(config), Endpointer::with_vad(config, vad.clone(), None)].into_iter().enumerate() {
            let mut endpointer = endpointer;
            let mut fed = 0u64;
            let mut found = Vec::new();
            for chunk in samples.chunks(320) {
                fed += chunk.len() as u64;
                for segment in endpointer.push(chunk) {
                    turns[k] += 1;
                    found.push(serde_json::json!([segment.start, segment.end]));
                    // Speech end: the last loud 20 ms frame inside the segment.
                    let last_loud = loud
                        .iter()
                        .rposition(|&(start, l)| l && start >= segment.start && start + 320 <= segment.end)
                        .map(|i| loud[i].0 + 320);
                    if let Some(speech_end) = last_loud {
                        // The turn closes when the endpointer has received `fed` samples.
                        delays[k].push((fed - speech_end) as f64 / 16.0);
                    }
                }
            }
            if let Some(segment) = endpointer.finish() {
                found.push(serde_json::json!([segment.start, segment.end]));
            }
            segments[k].insert(path.clone(), found.into());
        }
    }
    for (k, name) in ["energy", "silero"].iter().enumerate() {
        let d = &mut delays[k];
        d.sort_by(f64::total_cmp);
        let q = |p: f64| d[((d.len() - 1) as f64 * p) as usize];
        println!(
            "{name:7} turns {:4}  speech end -> turn closed: p50 {:.0} ms  p90 {:.0} ms  p95 {:.0} ms  mean {:.0} ms",
            turns[k],
            q(0.5),
            q(0.9),
            q(0.95),
            d.iter().sum::<f64>() / d.len() as f64
        );
    }
    if let Some(out) = dump {
        let [energy, silero] = segments;
        std::fs::write(out, serde_json::to_vec(&serde_json::json!({"energy": energy, "silero": silero}))?)?;
    }
    Ok(())
}

/// (frame start, speech-like) per 20 ms frame: 12 dB over the 3 s noise floor and over -54 dBFS.
fn loud_frames(samples: &[f32]) -> Vec<(u64, bool)> {
    let rms: Vec<f32> = samples.chunks_exact(320).map(|f| (f.iter().map(|x| x * x).sum::<f32>() / 320.0).sqrt()).collect();
    (0..rms.len())
        .map(|i| {
            let floor = rms[i.saturating_sub(149)..=i].iter().copied().fold(f32::INFINITY, f32::min);
            ((i * 320) as u64, rms[i] > (floor * 4.0).max(0.002))
        })
        .collect()
}

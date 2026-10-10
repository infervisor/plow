//! Cache-aware stream vs the offline transcript, per WAV: the stream is fed in `CHUNK_MS` pieces
//! (default 320) and must reproduce the offline words up to its lookahead. Prints one JSON line per
//! file: both texts, the matched word prefix, and push latency.
//!
//!   asr_stream_check PACKET TOKENIZER [BACKEND] [CHUNK_MS] -- AUDIO.wav...
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::{frontend::decode_wav, load_packet_transcriber};
    use std::{path::Path, sync::atomic::AtomicBool, time::Instant};

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<_> = std::env::args().collect();
    let split = args.iter().position(|a| a == "--").ok_or("usage: asr_stream_check PACKET TOKENIZER [BACKEND] [CHUNK_MS] -- AUDIO.wav...")?;
    if split < 3 {
        return Err("usage: asr_stream_check PACKET TOKENIZER [BACKEND] [CHUNK_MS] -- AUDIO.wav...".into());
    }
    let backend = args.get(3).filter(|_| split > 3).map_or("auto", String::as_str);
    let chunk_ms: usize = args.get(4).filter(|_| split > 4).map_or(Ok(320), |v| v.parse())?;
    let mut loaded = load_packet_transcriber(Path::new(&args[1]), Path::new(&args[2]), backend)?;
    let cancel = AtomicBool::new(false);
    let words = |s: &str| -> Vec<String> {
        s.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '\'').filter(|w| !w.is_empty()).map(str::to_owned).collect()
    };
    for path in &args[split + 1..] {
        let samples = decode_wav(&std::fs::read(path)?)?;
        let offline = loaded.engine.transcribe(&samples, None, "", &cancel)?.text;
        let id = loaded.engine.stream_open()?.ok_or("packet has no encoder stream")?;
        let (mut pushes, mut worst_ms, mut total_ms, mut first_text_s) = (0usize, 0f64, 0f64, None);
        let mut text = String::new();
        for (index, piece) in samples.chunks(16 * chunk_ms).enumerate() {
            let started = Instant::now();
            text = loaded.engine.stream_push(id, piece, &mut |_| {})?;
            let ms = started.elapsed().as_secs_f64() * 1e3;
            pushes += 1;
            total_ms += ms;
            worst_ms = worst_ms.max(ms);
            if first_text_s.is_none() && !text.is_empty() {
                first_text_s = Some((index + 1) as f64 * chunk_ms as f64 / 1e3);
            }
        }
        loaded.engine.stream_close(id);
        let (a, b) = (words(&offline), words(&text));
        let prefix = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
        println!(
            "{}",
            serde_json::json!({
                "audio": path, "seconds": samples.len() as f64 / 16000.0,
                "offline": offline, "stream": text,
                "offline_words": a.len(), "stream_words": b.len(), "matched_prefix_words": prefix,
                "stream_is_prefix": prefix == b.len(),
                "pushes": pushes, "push_ms_mean": total_ms / pushes.max(1) as f64, "push_ms_max": worst_ms,
                "first_text_at_audio_s": first_text_s,
            })
        );
    }
    Ok(())
}

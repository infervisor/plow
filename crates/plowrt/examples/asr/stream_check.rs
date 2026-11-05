//! Cache-aware stream vs the offline transcript, per WAV: the stream is fed in `CHUNK_MS` pieces
//! (default 320) and must reproduce the offline words up to its lookahead. Prints one JSON line per
//! file: both texts, the matched word prefix, and push latency. An engine with stream finals also
//! finishes the stream: its final must equal the offline transcript. With `BATCH` > 0, the files
//! then stream again `BATCH` at a time (`stream_push_batch`, one piece of each per call): each
//! final must equal the file's lone final. A last JSON line sums up.
//!
//!   asr_stream_check PACKET TOKENIZER [BACKEND] [CHUNK_MS] [BATCH] -- AUDIO.wav...
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::{frontend::decode_wav, load_packet_transcriber, StreamPush};
    use std::{path::Path, sync::atomic::AtomicBool, time::Instant};

    const USAGE: &str =
        "usage: asr_stream_check PACKET TOKENIZER [BACKEND] [CHUNK_MS] [BATCH] -- AUDIO.wav...";
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<_> = std::env::args().collect();
    let split = args.iter().position(|a| a == "--").ok_or(USAGE)?;
    if split < 3 {
        return Err(USAGE.into());
    }
    let option = |index: usize| args.get(index).filter(|_| split > index);
    let backend = option(3).map_or("auto", String::as_str);
    let chunk_ms: usize = option(4).map_or(Ok(320), |v| v.parse())?;
    let batch: usize = option(5).map_or(Ok(0), |v| v.parse())?;
    let mut loaded = load_packet_transcriber(Path::new(&args[1]), Path::new(&args[2]), backend)?;
    let finals = loaded.engine.stream_finals();
    let cancel = AtomicBool::new(false);
    let words = |s: &str| -> Vec<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .map(str::to_owned)
            .collect()
    };
    let paths = &args[split + 1..];
    let mut clips = Vec::with_capacity(paths.len());
    let mut final_equal = 0;
    for path in paths {
        let samples = decode_wav(&std::fs::read(path)?)?;
        let offline = loaded.engine.transcribe(&samples, None, "", &cancel)?.text;
        let id = loaded
            .engine
            .stream_open()?
            .ok_or("packet has no encoder stream")?;
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
        let (final_text, finish_ms) = if finals {
            let started = Instant::now();
            let push = StreamPush {
                id,
                samples: &[],
                finish: true,
            };
            let text = loaded.engine.stream_push_batch(&[push]).remove(0)?;
            (Some(text), Some(started.elapsed().as_secs_f64() * 1e3))
        } else {
            loaded.engine.stream_close(id);
            (None, None)
        };
        final_equal += usize::from(final_text.as_ref() == Some(&offline));
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
                "final": final_text, "final_equals_offline": final_text.as_ref().map(|t| *t == offline),
                "finish_ms": finish_ms,
            })
        );
        clips.push((path, samples, final_text));
    }
    let mut batched_equal = 0;
    let mut batched_ms = 0f64;
    if finals && batch > 0 {
        for group in clips.chunks(batch) {
            let ids = group
                .iter()
                .map(|_| {
                    loaded
                        .engine
                        .stream_open()?
                        .ok_or("packet has no encoder stream".into())
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            let piece = 16 * chunk_ms;
            let longest = group.iter().map(|c| c.1.len()).max().unwrap_or(0);
            let mut texts: Vec<Option<String>> = vec![None; group.len()];
            for offset in (0..=longest.div_ceil(piece) * piece).step_by(piece) {
                let pushes: Vec<_> = group
                    .iter()
                    .zip(&ids)
                    .zip(&texts)
                    .filter(|(_, text)| text.is_none())
                    .map(|((clip, &id), _)| {
                        let samples =
                            &clip.1[offset.min(clip.1.len())..(offset + piece).min(clip.1.len())];
                        StreamPush {
                            id,
                            samples,
                            finish: offset + piece >= clip.1.len(),
                        }
                    })
                    .collect();
                if pushes.is_empty() {
                    break;
                }
                let started = Instant::now();
                let results = loaded.engine.stream_push_batch(&pushes);
                batched_ms += started.elapsed().as_secs_f64() * 1e3;
                for (push, result) in pushes.iter().zip(results) {
                    if push.finish {
                        let member = ids.iter().position(|&id| id == push.id).expect("pushed");
                        texts[member] = Some(result?);
                    } else {
                        result?;
                    }
                }
            }
            for (clip, text) in group.iter().zip(texts) {
                let equal = text.is_some() && text == clip.2;
                batched_equal += usize::from(equal);
                println!(
                    "{}",
                    serde_json::json!({"audio": clip.0, "batched_final": text, "batched_equals_alone": equal})
                );
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "clips": clips.len(), "stream_finals": finals, "final_equals_offline": final_equal,
            "batch": batch, "batched_equals_alone": batched_equal, "batched_push_ms_total": batched_ms,
        })
    );
    Ok(())
}

use futures::{SinkExt, StreamExt};
use serde_json::json;
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) || args.get(4).is_some_and(|a| a != "realtime") {
        return Err(
            "usage: asr_stream ws://HOST:PORT/v1/audio/transcriptions/stream MODEL AUDIO.wav [realtime]"
                .into(),
        );
    }
    // `realtime`: send audio at 1x speed and request partial transcripts.
    let realtime = args.len() == 5;
    let started = std::time::Instant::now();
    let samples = plowrt::asr::frontend::decode_wav(&std::fs::read(&args[3])?)?;
    let (mut socket, _) = connect_async(&args[1]).await?;
    socket
        .send(Message::Text(
            json!({"type":"start","version":1,"model":args[2],
        "sample_rate":16000,"format":"pcm_s16le","partials":realtime})
            .to_string(),
        ))
        .await?;
    let (mut offset, mut sequence) = (0usize, 0u64);
    let mut finished = false;
    while let Some(message) = socket.next().await {
        let message = message?;
        if !message.is_text() {
            continue;
        }
        let event: serde_json::Value = serde_json::from_str(message.to_text()?)?;
        println!("{:.3} {event}", started.elapsed().as_secs_f64());
        match event["type"].as_str() {
            Some("final") => return Ok(()),
            Some("error") => return Err(event.to_string().into()),
            Some("ready" | "credit") if !finished => {
                let mut credit = event["credit_samples"].as_u64().ok_or("missing credit")? as usize;
                while credit > 0 && offset < samples.len() {
                    let count = credit.min(if realtime { 1600 } else { 16000 }).min(samples.len() - offset);
                    if realtime {
                        let due = std::time::Duration::from_secs_f64((offset + count) as f64 / 16000.0);
                        tokio::time::sleep(due.saturating_sub(started.elapsed())).await;
                    }
                    let mut bytes = Vec::with_capacity(8 + count * 2);
                    bytes.extend(sequence.to_le_bytes());
                    for sample in &samples[offset..offset + count] {
                        bytes.extend(
                            ((sample * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
                                .to_le_bytes(),
                        );
                    }
                    socket.send(Message::Binary(bytes)).await?;
                    offset += count;
                    credit -= count;
                    sequence += 1;
                }
                if offset == samples.len() {
                    socket
                        .send(Message::Text(r#"{"type":"finish"}"#.into()))
                        .await?;
                    finished = true;
                }
            }
            _ => {}
        }
    }
    Err("stream closed without a final transcript".into())
}

//! Decode codes through an asset's `codec.pkt`: `codec_check ASSETS CODES.i32 FRAMES SEED OUT.f32`
//! (ASSETS may name the `.pkt` itself, for an asset emitted without a speech pipeline).
//! (CODES.i32 = [items][frames][frame_codes] little-endian i32; items = file size / frame bytes).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("codec_check requires --features cuda");
}

#[cfg(feature = "cuda")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: codec_check ASSETS CODES.i32 FRAMES SEED OUT.f32".into());
    }
    let assets = std::path::Path::new(&args[1]);
    let codec = std::sync::Arc::new(if assets.extension().is_some_and(|e| e == "pkt") {
        plowrt::tts::codec::Codec::load_packet(assets)?
    } else {
        plowrt::tts::codec::Codec::load(assets)?
    });
    let frames: usize = args[3].parse()?;
    let seed: u64 = args[4].parse()?;
    let raw = std::fs::read(&args[2])?;
    let codes: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let per = frames * codec.frame_codes;
    // All items submitted before any completes, so the worker batches them into one launch.
    let jobs: Vec<_> = codes
        .chunks_exact(per)
        .map(|c| {
            let (codec, c) = (codec.clone(), c.to_vec());
            tokio::spawn(async move {
                codec
                    .decode(c, frames, seed, plowrt::tts::codec::Urgency::Whole)
                    .await
            })
        })
        .collect();
    let mut out = Vec::new();
    for j in jobs {
        out.extend(j.await??);
    }
    // CODEC_STREAM_STEP=N: decode the first item again through the stateful stream steps (one
    // frame, then N-frame steps), written to OUT.f32.stream for comparison with the whole decode.
    if let Some(step) = std::env::var("CODEC_STREAM_STEP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        let fc = codec.frame_codes;
        let mut slot = codec.acquire_slot().ok_or("codec has no stream slot")?;
        let mut pcm = Vec::new();
        let mut f = 0;
        let t = std::time::Instant::now();
        while f < frames {
            let n = if f == 0 { 1 } else { step.min(frames - f) };
            pcm.extend(
                codec
                    .step(
                        &mut slot,
                        codes[f * fc..(f + n) * fc].to_vec(),
                        n,
                        plowrt::tts::codec::Urgency::Stream,
                    )
                    .await?,
            );
            f += n;
        }
        eprintln!(
            "stream decode {frames} frames in {:.3} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        std::fs::write(
            format!("{}.stream", args[5]),
            bytemuck::cast_slice(&pcm),
        )?;
    }
    let t = std::time::Instant::now();
    for _ in 0..20 {
        codec
            .decode(
                codes[..per].to_vec(),
                frames,
                seed,
                plowrt::tts::codec::Urgency::Whole,
            )
            .await?;
    }
    eprintln!(
        "single-item decode {:.3} ms",
        t.elapsed().as_secs_f64() * 1e3 / 20.0
    );
    std::fs::write(&args[5], bytemuck::cast_slice(&out))?;
    Ok(())
}

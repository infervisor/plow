//! Decode codes through an asset's `codec.pkt`: `codec_check ASSETS CODES.i32 FRAMES SEED OUT.f32`
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
    let codec = std::sync::Arc::new(plowrt::tts::codec::Codec::load(std::path::Path::new(&args[1]))?);
    let frames: usize = args[3].parse()?;
    let seed: u64 = args[4].parse()?;
    let raw = std::fs::read(&args[2])?;
    let codes: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let per = frames * codec.frame_codes;
    // All items submitted before any completes, so the worker batches them into one launch.
    let jobs: Vec<_> = codes
        .chunks_exact(per)
        .map(|c| {
            let (codec, c) = (codec.clone(), c.to_vec());
            tokio::spawn(async move { codec.decode(c, frames, seed, plowrt::tts::codec::Urgency::Whole).await })
        })
        .collect();
    let mut out = Vec::new();
    for j in jobs {
        out.extend(j.await??);
    }
    let t = std::time::Instant::now();
    for _ in 0..20 {
        codec.decode(codes[..per].to_vec(), frames, seed, plowrt::tts::codec::Urgency::Whole).await?;
    }
    eprintln!("single-item decode {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);
    std::fs::write(&args[5], bytemuck::cast_slice(&out))?;
    Ok(())
}

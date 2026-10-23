//! Encode reference audio through `ref_encoder.pkt`: `ref_encoder_check PKT PCM.f32 OUT.u32`
//! (PCM.f32 = mono little-endian f32 at the packet's sample rate; OUT = [frames][frame_codes]).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ref_encoder_check requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: ref_encoder_check PKT PCM.f32 OUT.u32".into());
    }
    let enc = plowrt::tts::ref_encoder::RefEncoder::load(std::path::Path::new(&args[1]), 0)?;
    let raw = std::fs::read(&args[2])?;
    let pcm: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let codes = enc.encode(&pcm)?;
    let t = std::time::Instant::now();
    for _ in 0..10 {
        enc.encode(&pcm)?;
    }
    eprintln!(
        "{} samples -> {} frames, {:.3} ms",
        pcm.len(),
        codes.len() / enc.frame_codes,
        t.elapsed().as_secs_f64() * 1e3 / 10.0
    );
    std::fs::write(&args[3], bytemuck::cast_slice(&codes))?;
    Ok(())
}

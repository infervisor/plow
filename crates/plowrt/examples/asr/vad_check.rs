//! Speech probabilities and step cost of a `vad.silero.v1` packet on the host executor
//! (`plowrt::asr::vad`, what `--asr-vad-packet` serves).
//!
//!   asr_vad_check PACKET probs OUT.json AUDIO.f32...   one stream per raw 16 kHz f32le file;
//!       OUT.json = {"<file>": [p, ...]}, one per whole 512-sample frame
//!   asr_vad_check PACKET bench STREAMS FRAMES [THREADS]   STREAMS streams step one frame each per
//!       32 ms tick, on THREADS threads (default 1); prints per-frame and per-tick cost

use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: asr_vad_check PACKET probs OUT.json AUDIO.f32... | PACKET bench STREAMS FRAMES [THREADS]".into());
    }
    let vad = Arc::new(plowrt::asr::vad::Vad::load(std::path::Path::new(&args[1]))?);
    let frame = vad.frame;
    match args[2].as_str() {
        "probs" => {
            let mut out = serde_json::Map::new();
            for path in &args[4..] {
                let bytes = std::fs::read(path)?;
                let audio: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let mut stream = vad.open();
                let probs: Vec<f32> = audio.chunks_exact(frame).map(|f| vad.step(&mut stream, f)).collect();
                out.insert(path.clone(), probs.into());
            }
            std::fs::write(&args[3], serde_json::to_vec(&out)?)?;
        }
        "bench" => {
            let streams: usize = args[3].parse()?;
            let frames: usize = args.get(4).ok_or("bench needs FRAMES")?.parse()?;
            let threads: usize = args.get(5).map_or(Ok(1), |t| t.parse())?;
            let mut x = 0x9e3779b97f4a7c15u64;
            let audio: Vec<f32> = (0..frame * 64)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
                })
                .collect();
            let mut ticks = Vec::with_capacity(frames);
            let mut sets: Vec<Vec<_>> = (0..threads).map(|t| (t..streams).step_by(threads).map(|_| vad.open()).collect()).collect();
            for f in 0..frames {
                let chunk = &audio[(f % 64) * frame..(f % 64 + 1) * frame];
                let t = std::time::Instant::now();
                std::thread::scope(|s| {
                    for set in sets.iter_mut() {
                        let vad = &vad;
                        s.spawn(move || {
                            for stream in set.iter_mut() {
                                vad.step(stream, chunk);
                            }
                        });
                    }
                });
                ticks.push(t.elapsed().as_secs_f64() * 1e6);
            }
            ticks.sort_by(f64::total_cmp);
            let q = |p: f64| ticks[((ticks.len() - 1) as f64 * p) as usize];
            println!(
                "{{\"streams\":{streams},\"threads\":{threads},\"frames\":{frames},\"tick_p50_us\":{:.1},\"tick_p99_us\":{:.1},\"per_stream_frame_us\":{:.2},\"tick_budget_us\":32000}}",
                q(0.5),
                q(0.99),
                q(0.5) * threads as f64 / streams as f64
            );
        }
        other => return Err(format!("unknown mode {other}").into()),
    }
    Ok(())
}

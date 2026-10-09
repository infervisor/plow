//! Speech probabilities and batched step latency of a `vad.v1` packet, through the serving driver
//! (`plowrt::asr::vad`).
//!
//!   asr_vad_check PACKET probs BACKEND OUT.json AUDIO.f32...   one stream per raw 16 kHz f32le
//!       file, all concurrent; OUT.json = {"<file>": [p, ...]} (one per 512-sample frame)
//!   asr_vad_check PACKET bench BACKEND STREAMS FRAMES          STREAMS streams push one frame
//!       each per step (in lockstep, as live 32 ms audio does); prints step latency percentiles

use std::sync::Arc;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: asr_vad_check PACKET probs BACKEND OUT.json AUDIO.f32... | PACKET bench BACKEND STREAMS FRAMES".into());
    }
    let vad = plowrt::asr::vad::Vad::load(std::path::Path::new(&args[1]), &args[3], 0)?;
    eprintln!("backend {} frame {} max_batch {}", vad.backend, vad.frame_samples, vad.max_batch);
    match args[2].as_str() {
        "probs" => {
            let mut tasks = Vec::new();
            for path in &args[5..] {
                let bytes = std::fs::read(path)?;
                let audio: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let mut stream = vad.stream();
                let path = path.clone();
                // Pushes of 100 ms, as a client streams; partial frames carry over.
                tasks.push(tokio::spawn(async move {
                    let mut probs = Vec::new();
                    for chunk in audio.chunks(1_600) {
                        probs.extend(stream.push(chunk).await?.1);
                    }
                    Ok::<_, String>((path, probs))
                }));
            }
            let mut out = serde_json::Map::new();
            for t in tasks {
                let (path, probs) = t.await??;
                out.insert(path, probs.into());
            }
            std::fs::write(&args[4], serde_json::to_vec(&out)?)?;
        }
        "bench" => {
            let streams: usize = args[4].parse()?;
            let frames: usize = args.get(5).ok_or("bench needs FRAMES")?.parse()?;
            let frame = vad.frame_samples;
            let barrier = Arc::new(tokio::sync::Barrier::new(streams));
            let mut tasks = Vec::new();
            for s in 0..streams {
                let mut stream = vad.stream();
                let barrier = barrier.clone();
                tasks.push(tokio::spawn(async move {
                    let mut x = 0x9e3779b97f4a7c15u64 ^ s as u64;
                    let mut lat = Vec::with_capacity(frames);
                    for _ in 0..frames {
                        let audio: Vec<f32> = (0..frame)
                            .map(|_| {
                                x ^= x << 13;
                                x ^= x >> 7;
                                x ^= x << 17;
                                ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
                            })
                            .collect();
                        barrier.wait().await;
                        let t = std::time::Instant::now();
                        stream.push(&audio).await?;
                        lat.push(t.elapsed().as_secs_f64() * 1e6);
                    }
                    Ok::<_, String>(lat)
                }));
            }
            let mut all = Vec::new();
            for t in tasks {
                let lat = t.await??;
                all.extend(lat.into_iter().skip(10));
            }
            all.sort_by(f64::total_cmp);
            let q = |p: f64| all[((all.len() - 1) as f64 * p) as usize];
            let load = |c: &std::sync::atomic::AtomicU64| c.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let (launches, scored, device_ns) = (load(&vad.stats.launches), load(&vad.stats.frames), load(&vad.stats.device_ns));
            // Push latency: one stream's frame, from the step barrier to its probability.
            // Launch: one batched program run (device time on CUDA), rows = frames per launch.
            println!(
                "{{\"backend\":\"{}\",\"streams\":{streams},\"frames\":{frames},\"push_p50_us\":{:.1},\"push_p90_us\":{:.1},\"push_p99_us\":{:.1},\"launches\":{launches},\"rows_per_launch\":{:.1},\"launch_us\":{:.1},\"launch_us_per_frame\":{:.3}}}",
                vad.backend,
                q(0.5),
                q(0.9),
                q(0.99),
                scored / launches,
                device_ns / launches / 1e3,
                device_ns / scored / 1e3,
            );
        }
        other => return Err(format!("unknown mode {other}").into()),
    }
    Ok(())
}

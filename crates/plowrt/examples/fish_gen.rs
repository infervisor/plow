//! Fish S2 Pro frame generation on one slot: prefill a prompt, then per frame run the depth stage
//! (backbone class + codebooks) and feed the frame back. Writes the codes (`[frames][codebooks]`
//! u32, the stop frame excluded) for the codec gate.
//!
//!   fish_gen ASSETS CKPT PROMPT.u32 OUT.u32 [--prompt-codes CODES.u32] [--frames N] [--seed S]
//!            [--temperature T] [--top-p P] [--top-k K]
//!
//! PROMPT.u32 holds token ids; prompt frame rows (reference audio) take their codes from
//! `--prompt-codes` (`[prompt rows][codebooks]`). `--top-k 1` is greedy and matches the goldens.
use std::path::PathBuf;
use std::sync::Arc;

use plowrt::exec::gpu::{FishRow, GpuEngine};

fn read_u32(p: &str) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Fish keeps temperature and top_p as bf16 tensors (0.7 -> 0.69921875).
fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: fish_gen ASSETS CKPT PROMPT.u32 OUT.u32 [--prompt-codes F] [--frames N] [--seed S] [--temperature T] [--top-p P] [--top-k K]");
        std::process::exit(2);
    }
    let opt = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let frames: usize = opt("--frames").map_or(512, |v| v.parse().unwrap());
    let seed: u64 = opt("--seed").map_or(42, |v| v.parse().unwrap());
    let temperature = bf16(opt("--temperature").map_or(0.7, |v| v.parse().unwrap()));
    let top_p = bf16(opt("--top-p").map_or(0.7, |v| v.parse().unwrap()));
    let top_k: u32 = opt("--top-k").map_or(30, |v| v.parse().unwrap());
    let assets = PathBuf::from(&a[1]);
    let be = Arc::new(plowrt::device::cuda::CudaBackend::new(0)?);
    let mut e = GpuEngine::load(be, &assets, &PathBuf::from(&a[2]))?;
    let info = e.fish_depth_enable(&assets)?;
    let prompt = read_u32(&a[3]);
    if let Some(p) = opt("--prompt-codes") {
        e.fish_prompt_codes(&read_u32(&p))?;
    }
    e.begin_slot(0, prompt.len() + frames + 1)?;
    e.fish_reset_slot(0)?;
    let t0 = std::time::Instant::now();
    e.prefill_slot(0, &prompt)?;
    e.fish_logits_to_slot(0)?;
    let ttft = t0.elapsed();
    let mut row = FishRow {
        slot: 0,
        hidden_row: prompt.len() as u32 - 1,
        seed,
        step: 0,
        temperature,
        top_p,
        top_k,
        ras_temperature: bf16(1.0),
        ras_top_p: bf16(0.9),
    };
    let mut out = Vec::new();
    let mut codes = Vec::new();
    let mut toks = Vec::new();
    let t1 = std::time::Instant::now();
    let mut first_frame = None;
    for _ in 0..frames {
        codes.clear();
        e.fish_depth_step(&[row], &mut codes)?;
        first_frame.get_or_insert_with(|| t0.elapsed());
        let class = codes[0];
        if class == info.stop_class {
            break;
        }
        out.extend_from_slice(&codes);
        e.step_slots(&[(0, info.semantic_base + class)], &mut toks)?;
        row.step += 1;
        row.hidden_row = 0;
    }
    let n = out.len() / info.codebooks;
    let gen = t1.elapsed();
    std::fs::write(&a[4], bytemuck::cast_slice(&out))?;
    let audio_s = n as f64 * 2048.0 / 44100.0;
    println!(
        "frames {n} prefill_ms {:.1} first_frame_ms {:.1} frame_ms {:.2} rtf {:.3}",
        ttft.as_secs_f64() * 1e3,
        first_frame.unwrap_or_default().as_secs_f64() * 1e3,
        gen.as_secs_f64() * 1e3 / n.max(1) as f64,
        gen.as_secs_f64() / audio_s.max(1e-9)
    );
    Ok(())
}

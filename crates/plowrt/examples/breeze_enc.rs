//! Breeze TTS 2 text encoder (an encoder-mode packet, `plow_encoder`) on one text segment: one
//! prefill launch over the whole segment, then the projected rows (`act.enc_out`) as f32.
//!
//!   breeze_enc ASSETS CKPT IDS.u32 OUT.f32
use std::path::PathBuf;
use std::sync::Arc;

use plowrt::exec::gpu::{GpuEngine, PrefillStep};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: breeze_enc ASSETS CKPT IDS.u32 OUT.f32");
        std::process::exit(2);
    }
    const WIDTH: usize = 2048;
    let be = Arc::new(plowrt::device::cuda::CudaBackend::new(0)?);
    let mut e = GpuEngine::load(be, &PathBuf::from(&a[1]), &PathBuf::from(&a[2]))?;
    let ids: Vec<u32> = bytemuck::pod_collect_to_vec(&std::fs::read(&a[3])?);
    let n = ids.len();
    e.begin_slot(0, n)?;
    // Bidirectional reach per row: the local layers' window ahead (T5Gemma2: 256), the full
    // layers' whole segment.
    let local: Vec<u32> = (0..n).map(|r| ((n - 1 - r) as u32).min(256)).collect();
    let full: Vec<u32> = (0..n).map(|r| (n - 1 - r) as u32).collect();
    e.write_tensor("in.enc.span_local", 0, bytemuck::cast_slice(&local))?;
    e.write_tensor("in.enc.span_full", 0, bytemuck::cast_slice(&full))?;
    let t0 = std::time::Instant::now();
    match e.prefill_chunk(0, &ids, usize::MAX)? {
        PrefillStep::Done(_) => {}
        PrefillStep::Progress(p) => {
            return Err(format!("segment split at row {p}: no bucket covers {n}").into())
        }
    }
    let mut row = vec![0u8; n * WIDTH * 2];
    e.read_tensor_range("act.enc_out", 0, &mut row)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let out: Vec<f32> = row
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect();
    std::fs::write(&a[4], bytemuck::cast_slice(&out))?;
    println!("rows {n} ms {ms:.2} -> {}", a[4]);
    Ok(())
}

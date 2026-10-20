//! Breeze TTS 2 backbone + depth stage, teacher-forced on one slot: prefill the projected
//! text-encoder rows as overlay rows, then per frame run the depth program with every class forced
//! to the given frame and feed the frame back (embedded as the sum of its codebook rows). Writes the
//! final-norm hidden after the prompt and after each frame (`OUT.f32`, `[frames + 1][hidden]`), the
//! fp32 backbone logits per frame (`OUT.blogits.f32`, `[frames][classes]`) and the fp32 depth logits
//! (`OUT.depth.f32`, `[frames][codebooks - 1][2048]`). FRAMES.u32 `free:N` instead samples `N`
//! frames (top_k from `BREEZE_TOP_K`, default 50; stopping at the stop class) and also writes them
//! to `OUT.frames.u32`.
//!
//!   breeze_gen ASSETS CKPT TEXT_ENC.f32 FRAMES.u32 OUT.f32 [PROMPT_CODES.u32]
//!
//! TEXT_ENC.f32 is `[text rows][hidden]`; FRAMES.u32 is `[frames][codebooks]`. Without
//! PROMPT_CODES every prompt row is a text row. PROMPT_CODES (`[prompt rows][1 + codebooks]`)
//! lays out a mixed prompt: a row whose first word is 0 takes the next text row, 1 is a reference
//! frame with the given codes.
use std::path::PathBuf;
use std::sync::Arc;

use plowrt::exec::gpu::{FishRow, GpuEngine, PrefillStep};

fn read<T: bytemuck::Pod>(p: &str) -> Vec<T> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    bytemuck::pod_collect_to_vec(&b)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if !(6..=7).contains(&a.len()) {
        eprintln!(
            "usage: breeze_gen ASSETS CKPT TEXT_ENC.f32 FRAMES.u32 OUT.f32 [PROMPT_CODES.u32]"
        );
        std::process::exit(2);
    }
    const HIDDEN: usize = 2048;
    const CODEBOOKS: usize = 16;
    let assets = PathBuf::from(&a[1]);
    let be = Arc::new(plowrt::device::cuda::CudaBackend::new(0)?);
    let mut e = GpuEngine::load(be, &assets, &PathBuf::from(&a[2]))?;
    let enc: Vec<f32> = read(&a[3]);
    let free: Option<usize> = a[4].strip_prefix("free:").map(|n| n.parse().unwrap());
    let frames: Vec<u32> = match free {
        Some(n) => vec![u32::MAX; n * CODEBOOKS],
        None => read(&a[4]),
    };
    let text_rows = enc.len() / HIDDEN;
    let layout: Vec<u32> = match a.get(6) {
        Some(f) => read(f),
        None => (0..text_rows).flat_map(|_| [0u32; CODEBOOKS + 1]).collect(),
    };
    // Overlay index per prompt row (text rows in order, frame rows kept) and the prompt codes.
    let (mut index, mut codes, mut next) = (Vec::new(), Vec::new(), 0u32);
    for r in layout.chunks_exact(CODEBOOKS + 1) {
        if r[0] == 0 {
            index.push(next);
            next += 1;
        } else {
            index.push(u32::MAX - 1);
        }
        codes.extend_from_slice(&r[1..]);
    }
    assert_eq!(
        next as usize, text_rows,
        "prompt layout text rows != TEXT_ENC rows"
    );
    let n = index.len();
    let nf = frames.len() / CODEBOOKS;
    let mut out = Vec::with_capacity((nf + 1) * HIDDEN);
    let mut row = vec![0u8; HIDDEN * 2];
    let mut hidden_row =
        |e: &GpuEngine, r: u32, out: &mut Vec<f32>| -> Result<(), Box<dyn std::error::Error>> {
            e.read_tensor_range("act.hn", u64::from(r) * (HIDDEN * 2) as u64, &mut row)?;
            out.extend(
                row.chunks_exact(2)
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)),
            );
            Ok(())
        };

    e.begin_slot(0, n + nf + 1)?;
    e.write_tensor("in.encoder_overlay", 0, bytemuck::cast_slice(&enc))?;
    // A frame row's token is its codebook-0 code; text rows embed from the overlay.
    let ids: Vec<u32> = (0..n).map(|r| codes[r * CODEBOOKS]).collect();
    let mut c0 = 0;
    loop {
        let hi = n.min(c0 + e.pf_max_rows().max(1));
        e.write_tensor(
            "in.encoder_overlay_index",
            0,
            bytemuck::cast_slice(&index[c0..hi]),
        )?;
        e.write_tensor(
            "in.frame_codes",
            0,
            bytemuck::cast_slice(&codes[c0 * CODEBOOKS..hi * CODEBOOKS]),
        )?;
        match e.prefill_chunk(0, &ids, n)? {
            PrefillStep::Done(_) => break,
            PrefillStep::Progress(p) => c0 = p,
        }
    }
    hidden_row(&e, e.prefill_terminal_row(), &mut out)?;
    let info = e.fish_depth_enable(&assets)?;
    e.fish_reset_slot(0)?;
    let classes = info.stop_class as usize + 1;
    let (mut blogits, mut depth) = (Vec::new(), Vec::new());
    let mut read_f32 = |e: &GpuEngine,
                        name: &str,
                        n: usize,
                        out: &mut Vec<f32>|
     -> Result<(), Box<dyn std::error::Error>> {
        let mut b = vec![0u8; n * 4];
        e.read_tensor_range(name, 0, &mut b)?;
        out.extend_from_slice(bytemuck::cast_slice(&b));
        Ok(())
    };
    let mut row = FishRow {
        slot: 0,
        hidden_row: e.prefill_terminal_row(),
        seed: 42,
        step: 0,
        temperature: 0.9,
        top_p: 1.0,
        top_k: std::env::var("BREEZE_TOP_K").map_or(50, |v| v.parse().unwrap()),
        ras_temperature: 1.0,
        ras_top_p: 1.0,
        repetition_penalty: 1.1,
    };
    let (mut toks, mut codes, mut sampled) = (Vec::new(), Vec::new(), Vec::new());
    for f in frames.chunks_exact(CODEBOOKS) {
        e.write_tensor_ordered("in.breeze.force", 0, bytemuck::cast_slice(f))?;
        codes.clear();
        e.fish_depth_step(&[row], &mut codes)?;
        if free.is_none() {
            assert_eq!(&codes[..], f, "forced frame codes");
        }
        sampled.extend_from_slice(&codes);
        read_f32(&e, "act.fish.blogits", classes, &mut blogits)?;
        for j in 1..CODEBOOKS {
            read_f32(&e, &format!("act.fish.dlogits.{j}"), 2048, &mut depth)?;
        }
        if codes[0] == info.stop_class {
            break;
        }
        e.step_slots(&[(0, codes[0])], &mut toks)?;
        hidden_row(&e, 0, &mut out)?;
        row.hidden_row = 0;
        row.step += 1;
    }
    std::fs::write(&a[5], bytemuck::cast_slice(&out))?;
    std::fs::write(
        format!("{}.blogits.f32", a[5]),
        bytemuck::cast_slice(&blogits),
    )?;
    std::fs::write(format!("{}.depth.f32", a[5]), bytemuck::cast_slice(&depth))?;
    std::fs::write(
        format!("{}.frames.u32", a[5]),
        bytemuck::cast_slice(&sampled),
    )?;
    println!("prompt rows {n} frames {nf} -> {}", a[5]);
    Ok(())
}

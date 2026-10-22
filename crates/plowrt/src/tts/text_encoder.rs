//! An encoder-mode packet (`plow_encoder`) beside a speech LM: one bidirectional prefill launch
//! per text segment, whose projected rows (`act.enc_out`, bf16) become the LM prompt's overlay
//! rows. Breeze TTS 2's T5Gemma2 text encoder: local layers reach `LOCAL_REACH` keys ahead, global
//! layers the whole segment.

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::exec::gpu::{GpuEngine, PrefillStep};

/// Keys ahead a local (sliding) layer sees: half of T5Gemma2's 512-key bidirectional window.
const LOCAL_REACH: u32 = 256;

pub struct TextEncoder {
    e: Mutex<GpuEngine>,
    width: usize,
    rows: usize,
    /// The encoder's own tokenizer (the sub-bundle's `tokenizer.json`).
    pub tok: Arc<dyn crate::text::tokenizer::Tokenize>,
}

impl TextEncoder {
    pub fn load(dir: &Path, device: u8) -> Result<Self, String> {
        let be =
            Arc::new(crate::device::cuda::CudaBackend::new(device).map_err(|e| e.to_string())?);
        let e = GpuEngine::load(be, dir, &crate::asset::serve::checkpoint_dir(dir))
            .map_err(|e| format!("text encoder {}: {e}", dir.display()))?;
        let width = e.vocab();
        if e.tensor_bytes("act.enc_out").is_none() {
            return Err(format!(
                "{} is not an encoder-mode packet (no act.enc_out)",
                dir.display()
            ));
        }
        Ok(TextEncoder {
            rows: e.pf_max_rows(),
            e: Mutex::new(e),
            width,
            tok: crate::text::tokenizer::load_tokenizer(dir),
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// Most prompt rows one segment may have.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// `[ids.len()][width]` f32 rows for one segment.
    pub fn encode(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let n = ids.len();
        let t = std::time::Instant::now();
        let mut e = self.e.lock();
        let wait = t.elapsed();
        if n == 0 || n > e.pf_max_rows() {
            return Err(format!(
                "text segment of {n} tokens; the encoder takes 1..={}",
                e.pf_max_rows()
            ));
        }
        let local: Vec<u32> = (0..n)
            .map(|r| ((n - 1 - r) as u32).min(LOCAL_REACH))
            .collect();
        let full: Vec<u32> = (0..n).map(|r| (n - 1 - r) as u32).collect();
        let run = |e: &mut GpuEngine| -> crate::Result<Vec<u8>> {
            e.begin_slot(0, n)?;
            e.write_tensor("in.enc.span_local", 0, bytemuck::cast_slice(&local))?;
            e.write_tensor("in.enc.span_full", 0, bytemuck::cast_slice(&full))?;
            // No row cap: the bucket pick must see every rung to cover the segment in one launch.
            match e.prefill_chunk(0, ids, usize::MAX)? {
                PrefillStep::Done(_) => {}
                PrefillStep::Progress(p) => {
                    return Err(crate::RuntimeError::Device(format!(
                        "encoder segment split at row {p}"
                    )))
                }
            }
            let mut raw = vec![0u8; n * self.width * 2];
            e.read_tensor_range("act.enc_out", 0, &mut raw)?;
            Ok(raw)
        };
        let raw = run(&mut e).map_err(|err| format!("text encoder: {err}"))?;
        tracing::debug!(
            rows = n,
            wait_us = wait.as_micros() as u64,
            us = (t.elapsed() - wait).as_micros() as u64,
            "text encode"
        );
        Ok(raw
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect())
    }
}

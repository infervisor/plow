use std::path::Path;
use std::sync::Arc;

use crate::device::cuda::CudaBackend;
use crate::exec::gpu::{GpuEngine, PrefillStep};
use crate::{Result, RuntimeError};

use super::{PacketAudioEncoder, QwenDecode, QwenExecution, QwenPrefill, QwenPrefilled};

/// Qwen3-ASR on CUDA: the audio encoder is `encoder.pkt` on the CUDA packet runtime, the decoder
/// `model.pkt` on `GpuEngine`, whose prefill splices the encoder rows over the audio
/// placeholders (`EmbedOverlayBf16`, `in.encoder_overlay` / `in.encoder_overlay_index`).
struct CudaQwenExecution {
    encoder: PacketAudioEncoder,
    decoder: GpuEngine,
    overlay_rows: usize,
    feeds: Vec<(usize, u32)>,
    out: Vec<u32>,
}

impl CudaQwenExecution {
    fn load(blob: &Path, checkpoint: &Path, hidden: usize) -> Result<Self> {
        let be = Arc::new(CudaBackend::new(0)?);
        let assets = blob.parent().unwrap_or(Path::new("."));
        let decoder = GpuEngine::load(be, assets, checkpoint)?;
        let encoder = PacketAudioEncoder::load(&blob.with_file_name("encoder.pkt"), "cuda")?;
        if encoder.output_width() != hidden {
            return Err(RuntimeError::Rejected("audio packet output width does not match the decoder".into()));
        }
        let overlay_rows = decoder
            .tensor_bytes("in.encoder_overlay")
            .map(|b| b as usize / (hidden * 4))
            .ok_or_else(|| RuntimeError::Rejected("decoder packet has no in.encoder_overlay".into()))?;
        Ok(Self { encoder, decoder, overlay_rows, feeds: Vec::new(), out: Vec::new() })
    }
}

impl QwenExecution for CudaQwenExecution {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn batch_capacity(&self) -> usize {
        self.decoder.batch()
    }

    fn max_context(&self) -> usize {
        self.decoder.max_ctx()
    }

    fn prefill(&mut self, slot: usize, input: QwenPrefill<'_>) -> Result<QwenPrefilled> {
        let started = std::time::Instant::now();
        let audio = self.encoder.encode(input.features)?;
        let encoder_ms = started.elapsed().as_secs_f64() * 1000.0;
        let rows = audio.len() / input.hidden;
        let n = input.token_ids.len();
        if rows != input.audio_positions.len() || rows > self.overlay_rows {
            return Err(RuntimeError::Rejected(format!(
                "{rows} audio rows for {} placeholders (overlay capacity {})",
                input.audio_positions.len(),
                self.overlay_rows
            )));
        }
        let prefill_started = std::time::Instant::now();
        self.decoder.begin_slot(slot, self.decoder.max_ctx())?;
        self.decoder.write_tensor("in.encoder_overlay", 0, bytemuck::cast_slice(&audio))?;
        let mut index = vec![u32::MAX; self.decoder.max_ctx().max(n)];
        for (row, &pos) in input.audio_positions.iter().enumerate() {
            index[pos] = row as u32;
        }
        let index_rows = self.decoder.tensor_bytes("in.encoder_overlay_index").unwrap_or(0) as usize / 4;
        index.resize(index_rows, u32::MAX);
        self.decoder.write_tensor("in.encoder_overlay_index", 0, bytemuck::cast_slice(&index))?;
        let token = match self.decoder.prefill_chunk(slot, input.token_ids, n)? {
            PrefillStep::Done(token) => token,
            PrefillStep::Progress(_) => {
                return Err(RuntimeError::ContextLength(format!("ASR prompt of {n} rows must fit one prefill chunk")))
            }
        };
        Ok(QwenPrefilled { token, encoder_ms, prefill_ms: prefill_started.elapsed().as_secs_f64() * 1000.0 })
    }

    fn decode(&mut self, positions: &[u32], kv_lengths: &[u32], token_ids: &[u32], occupied_rows: usize) -> Result<QwenDecode> {
        if positions.len() != kv_lengths.len()
            || positions.len() != token_ids.len()
            || occupied_rows == 0
            || occupied_rows > positions.len()
            || positions.len() > self.batch_capacity()
        {
            return Err(RuntimeError::Rejected("invalid ASR decode cohort".into()));
        }
        // Finished rows inside the occupied prefix carry kv_length 1; the engine tracks positions.
        self.feeds.clear();
        self.feeds.extend((0..occupied_rows).filter(|&i| kv_lengths[i] > 1).map(|i| (i, token_ids[i])));
        let mut tokens = vec![0u32; positions.len()];
        if !self.feeds.is_empty() {
            self.decoder.step_slots(&self.feeds, &mut self.out)?;
            for (&(slot, _), &t) in self.feeds.iter().zip(&self.out) {
                tokens[slot] = t;
            }
        }
        Ok(QwenDecode { tokens, launched_rows: self.feeds.len() })
    }
}

pub(super) fn load_execution(packet: &Path, checkpoint: &Path, hidden: usize) -> Result<Box<dyn QwenExecution>> {
    Ok(Box::new(CudaQwenExecution::load(packet, checkpoint, hidden)?))
}

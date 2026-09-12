use std::path::Path;

use crate::exec::apple::{asr::QwenAudioEncoder, DecodeTuning, MetalEngine};
use crate::exec::packet_runtime::{BoundPacketPipeline, PacketAsset};
use crate::{Result, RuntimeError};

use super::{PacketAudioEncoder, QwenAsr, QwenDecode, QwenExecution, QwenPrefill, QwenPrefilled};

struct MetalQwenExecution {
    encoder: Option<QwenAudioEncoder>,
    packet_encoder: Option<PacketAudioEncoder>,
    decoder: MetalEngine,
    causal: BoundPacketPipeline,
    embedding_handle: usize,
    decode_capacity: usize,
    max_context: usize,
    device_handoff: bool,
    positions: Vec<u32>,
    kv_lengths: Vec<u32>,
    token_ids: Vec<u32>,
}

impl MetalQwenExecution {
    fn load(blob: &Path, checkpoint: &Path, hidden: usize) -> Result<Self> {
        let mut decoder = MetalEngine::load_with_decode_tuning(
            blob,
            checkpoint,
            DecodeTuning {
                qkv_dot4: true,
                single_head_work: true,
            },
        )?;
        let causal = PacketAsset::load(blob)?.bind("decode", &decoder)?;
        if causal.driver() != "causal.v1" {
            return Err(RuntimeError::Rejected(format!(
                "unsupported causal packet driver {:?}",
                causal.driver()
            )));
        }
        decoder.set_ordered_dispatch(causal.parameter("ordered_dispatch").unwrap_or(0) != 0)?;
        let embedding_handle = decoder.embedding_table_handle()?;
        let decode_capacity = usize::try_from(causal.parameter("decode_capacity")?)
            .map_err(|_| RuntimeError::Rejected("decode capacity overflows".into()))?;
        let max_context = usize::try_from(causal.parameter("max_context")?)
            .map_err(|_| RuntimeError::Rejected("maximum context overflows".into()))?;
        let packet_hidden = usize::try_from(causal.parameter("hidden")?)
            .map_err(|_| RuntimeError::Rejected("packet hidden width overflows".into()))?;
        if decode_capacity == 0
            || decode_capacity > u32::MAX as usize
            || max_context == 0
            || max_context > u32::MAX as usize
            || packet_hidden != hidden
        {
            return Err(RuntimeError::Rejected(
                "causal packet geometry does not match the checkpoint".into(),
            ));
        }
        let encoder_packet = blob.with_file_name("encoder.pkt");
        let packet_encoder = encoder_packet
            .is_file()
            .then(|| PacketAudioEncoder::load(&encoder_packet, "metal"))
            .transpose()?;
        if packet_encoder
            .as_ref()
            .is_some_and(|encoder| encoder.output_width() != hidden)
        {
            return Err(RuntimeError::Rejected(
                "audio packet output width does not match the decoder".into(),
            ));
        }
        let mut configure_encoder = |mut encoder: QwenAudioEncoder| -> Result<QwenAudioEncoder> {
            encoder.set_tiled_linear(true);
            encoder.set_packed_conv(true);
            encoder.set_wide_linear(encoder.supports_wide_linear())?;
            encoder.set_large_linear(encoder.supports_large_linear())?;
            encoder.set_simd_attention(encoder.supports_simd_attention())?;
            if encoder.validate_direct_epilogue().unwrap_or(false) {
                encoder.set_direct_epilogue(true)?;
            }
            Ok(encoder)
        };
        let (encoder, packet_encoder) = if let Some(packet) = packet_encoder {
            match QwenAudioEncoder::load_from_packet(checkpoint, &packet.packet)
                .and_then(&mut configure_encoder)
            {
                Ok(encoder) => {
                    tracing::info!("using specialized Metal execution for Qwen audio packet");
                    (Some(encoder), None)
                }
                Err(error) => {
                    tracing::warn!(%error, "Qwen audio packet specialization unavailable; using checkpoint weights");
                    (
                        Some(configure_encoder(QwenAudioEncoder::load(checkpoint)?)?),
                        None,
                    )
                }
            }
        } else {
            (
                Some(configure_encoder(QwenAudioEncoder::load(checkpoint)?)?),
                None,
            )
        };
        Ok(Self {
            encoder,
            packet_encoder,
            decoder,
            causal,
            embedding_handle,
            decode_capacity,
            max_context,
            device_handoff: false,
            positions: vec![0; decode_capacity],
            kv_lengths: vec![1; decode_capacity],
            token_ids: vec![0; decode_capacity],
        })
    }
}

impl QwenExecution for MetalQwenExecution {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn batch_capacity(&self) -> usize {
        self.decode_capacity
    }

    fn max_context(&self) -> usize {
        self.max_context
    }

    fn prefill(&mut self, slot: usize, input: QwenPrefill<'_>) -> Result<QwenPrefilled> {
        let encoder_started = std::time::Instant::now();
        let packet_audio = self
            .packet_encoder
            .as_mut()
            .filter(|encoder| encoder.accepts(input.features.frames))
            .map(|encoder| encoder.encode(input.features))
            .transpose()?;
        let audio = if packet_audio.is_none() {
            Some(
                self.encoder
                    .as_ref()
                    .ok_or_else(|| {
                        RuntimeError::Rejected(
                            "audio packet does not cover the requested feature shape".into(),
                        )
                    })?
                    .encode_device(input.features)?,
            )
        } else {
            None
        };
        let encoder_ms = encoder_started.elapsed().as_secs_f64() * 1000.0;
        let audio_shape = if let Some(audio) = &packet_audio {
            [audio.len() / input.hidden, input.hidden]
        } else if let Some(audio) = &audio {
            audio.shape()
        } else {
            return Err(RuntimeError::Device(
                "ASR encoder produced no output".into(),
            ));
        };
        if audio_shape != [input.audio_positions.len(), input.hidden] {
            return Err(RuntimeError::Rejected(
                "ASR projector dimension mismatch".into(),
            ));
        }

        let prefill_started = std::time::Instant::now();
        let token = if let Some(audio) = packet_audio {
            prefill_host_audio(
                &mut self.decoder,
                self.embedding_handle,
                slot,
                &input,
                &audio,
            )?
        } else if self.device_handoff {
            let audio = audio
                .as_ref()
                .ok_or_else(|| RuntimeError::Device("ASR encoder output is missing".into()))?;
            let encoder = self
                .encoder
                .as_ref()
                .ok_or_else(|| RuntimeError::Device("ASR encoder is missing".into()))?;
            let mut bindings: Vec<_> = input
                .token_ids
                .iter()
                .map(|&token| [token, u32::MAX])
                .collect();
            for (row, &position) in input.audio_positions.iter().enumerate() {
                bindings[position][1] = u32::try_from(row)
                    .map_err(|_| RuntimeError::Rejected("ASR audio row overflows".into()))?;
            }
            let embedding_elements = input
                .token_ids
                .len()
                .checked_mul(input.hidden)
                .ok_or_else(|| RuntimeError::Rejected("ASR embedding shape overflows".into()))?;
            self.decoder.prefill_slot_embeddings_staged(
                slot,
                input.token_ids,
                embedding_elements,
                true,
                |command_buffer, output, table, chunk, hidden, vocab| {
                    if hidden != input.hidden {
                        return Err(RuntimeError::Rejected(
                            "ASR embedding width mismatch".into(),
                        ));
                    }
                    let start = usize::try_from(chunk.c0)
                        .map_err(|_| RuntimeError::Device("ASR chunk offset overflows".into()))?;
                    let end = chunk
                        .c0
                        .checked_add(chunk.clen)
                        .and_then(|end| usize::try_from(end).ok())
                        .ok_or_else(|| RuntimeError::Device("ASR chunk range overflows".into()))?;
                    let bindings = bindings.get(start..end).ok_or_else(|| {
                        RuntimeError::Device("ASR chunk range is outside its bindings".into())
                    })?;
                    if let Some(command_buffer) = command_buffer {
                        encoder.encode_splice(command_buffer, audio, table, vocab, output, bindings)
                    } else {
                        encoder.splice_into(audio, table, vocab, output, bindings)
                    }
                },
            )?
        } else {
            let audio = audio
                .ok_or_else(|| RuntimeError::Device("ASR encoder output is missing".into()))?
                .read();
            prefill_host_audio(
                &mut self.decoder,
                self.embedding_handle,
                slot,
                &input,
                &audio,
            )?
        };
        Ok(QwenPrefilled {
            token,
            encoder_ms,
            prefill_ms: prefill_started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn decode(
        &mut self,
        positions: &[u32],
        kv_lengths: &[u32],
        token_ids: &[u32],
        occupied_rows: usize,
    ) -> Result<QwenDecode> {
        let batch = self.batch_capacity();
        if positions.len() != kv_lengths.len()
            || positions.len() != token_ids.len()
            || positions.is_empty()
            || positions.len() > batch
            || occupied_rows == 0
            || occupied_rows > positions.len()
        {
            return Err(RuntimeError::Rejected("invalid ASR decode cohort".into()));
        }
        let occupied_rows = u32::try_from(occupied_rows)
            .map_err(|_| RuntimeError::Rejected("ASR decode cohort is too large".into()))?;
        let (program, launched_rows) = self.causal.program_for_rows("decode", occupied_rows)?;
        let launched_rows = usize::try_from(launched_rows)
            .map_err(|_| RuntimeError::Rejected("ASR decode capacity overflows".into()))?;
        let mut tokens = if positions.len() == batch {
            self.decoder
                .decode_step_batched_at(positions, kv_lengths, token_ids, program)?
        } else {
            self.positions.fill(0);
            self.kv_lengths.fill(1);
            self.token_ids.fill(0);
            self.positions[..positions.len()].copy_from_slice(positions);
            self.kv_lengths[..kv_lengths.len()].copy_from_slice(kv_lengths);
            self.token_ids[..token_ids.len()].copy_from_slice(token_ids);
            self.decoder.decode_step_batched_at(
                &self.positions,
                &self.kv_lengths,
                &self.token_ids,
                program,
            )?
        };
        tokens.truncate(positions.len());
        Ok(QwenDecode {
            tokens,
            launched_rows,
        })
    }
}

fn prefill_host_audio(
    decoder: &mut MetalEngine,
    embedding_handle: usize,
    slot: usize,
    input: &QwenPrefill<'_>,
    audio: &[f32],
) -> Result<u32> {
    let embedding_elements = input
        .token_ids
        .len()
        .checked_mul(input.hidden)
        .ok_or_else(|| RuntimeError::Rejected("ASR embedding shape overflows".into()))?;
    let mut embeddings = Vec::with_capacity(embedding_elements);
    let table = decoder.tensor_bytes(embedding_handle);
    for &token in input.token_ids {
        let row_bytes = input
            .hidden
            .checked_mul(2)
            .ok_or_else(|| RuntimeError::Rejected("ASR embedding row overflows".into()))?;
        let offset = usize::try_from(token)
            .ok()
            .and_then(|token| token.checked_mul(row_bytes))
            .ok_or_else(|| RuntimeError::Rejected("ASR embedding offset overflows".into()))?;
        let end = offset
            .checked_add(row_bytes)
            .ok_or_else(|| RuntimeError::Rejected("ASR embedding range overflows".into()))?;
        let bytes = table
            .get(offset..end)
            .ok_or_else(|| RuntimeError::Rejected("ASR token outside embedding table".into()))?;
        embeddings.extend(
            bytes
                .chunks_exact(2)
                .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])),
        );
    }
    for (row, &position) in input.audio_positions.iter().enumerate() {
        for column in 0..input.hidden {
            let value = audio[row * input.hidden + column].to_bits();
            embeddings[position * input.hidden + column] =
                (value.wrapping_add(0x7fff + ((value >> 16) & 1)) >> 16) as u16;
        }
    }
    decoder.prefill_slot_embeddings(slot, input.token_ids, &embeddings)
}

pub(super) fn load_execution(
    blob: &Path,
    checkpoint: &Path,
    hidden: usize,
) -> Result<Box<dyn QwenExecution>> {
    Ok(Box::new(MetalQwenExecution::load(
        blob, checkpoint, hidden,
    )?))
}

impl QwenAsr {
    fn metal_execution(&mut self) -> &mut MetalQwenExecution {
        self.execution
            .as_any_mut()
            .downcast_mut::<MetalQwenExecution>()
            .expect("QwenAsr::load installed the Metal executor")
    }

    pub fn set_large_linear(&mut self, enabled: bool) -> Result<()> {
        self.metal_execution()
            .encoder
            .as_mut()
            .map_or(Ok(()), |encoder| encoder.set_large_linear(enabled))
    }

    pub fn set_bf16_linear_weights(&mut self, enabled: bool) -> Result<()> {
        self.metal_execution()
            .encoder
            .as_mut()
            .map_or(Ok(()), |encoder| encoder.set_bf16_linear_weights(enabled))
    }

    pub fn set_simd_attention(&mut self, enabled: bool) -> Result<()> {
        self.metal_execution()
            .encoder
            .as_mut()
            .map_or(Ok(()), |encoder| encoder.set_simd_attention(enabled))
    }

    pub fn set_direct_epilogue(&mut self, enabled: bool) -> Result<()> {
        self.metal_execution()
            .encoder
            .as_mut()
            .map_or(Ok(()), |encoder| encoder.set_direct_epilogue(enabled))
    }

    pub fn set_tile64_selective(&mut self, enabled: bool) -> Result<()> {
        self.metal_execution()
            .encoder
            .as_mut()
            .map_or(Ok(()), |encoder| encoder.set_tile64_selective(enabled))
    }

    pub fn set_wide_linear(&mut self, enabled: bool) -> Result<()> {
        self.metal_execution()
            .encoder
            .as_mut()
            .map_or(Ok(()), |encoder| encoder.set_wide_linear(enabled))
    }

    pub fn set_device_handoff(&mut self, enabled: bool) {
        self.metal_execution().device_handoff = enabled;
    }

    pub fn set_packed_conv(&mut self, enabled: bool) {
        if let Some(encoder) = self.metal_execution().encoder.as_mut() {
            encoder.set_packed_conv(enabled);
        }
    }

    pub fn set_tiled_conv(&mut self, enabled: bool) {
        if let Some(encoder) = self.metal_execution().encoder.as_mut() {
            encoder.set_tiled_conv(enabled);
        }
    }

    pub fn set_tiled_linear(&mut self, enabled: bool) {
        if let Some(encoder) = self.metal_execution().encoder.as_mut() {
            encoder.set_tiled_linear(enabled);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;

    #[test]
    #[ignore = "requires PLOW_TEST_BLOB, PLOW_TEST_CHECKPOINT and PLOW_TEST_AUDIO_DIR"]
    fn device_handoff_preserves_transcripts_and_logits() {
        let blob = std::env::var("PLOW_TEST_BLOB").unwrap();
        let checkpoint = std::env::var("PLOW_TEST_CHECKPOINT").unwrap();
        let audio_dir = std::env::var("PLOW_TEST_AUDIO_DIR").unwrap();
        let mut engine = QwenAsr::load(Path::new(&blob), Path::new(&checkpoint)).unwrap();
        let cancel = AtomicBool::new(false);
        for name in ["short", "english16", "chinese16"] {
            let wav = std::fs::read(Path::new(&audio_dir).join(format!("{name}.wav"))).unwrap();
            let samples = crate::asr::frontend::decode_wav(&wav).unwrap();
            engine.set_device_handoff(false);
            let expected = engine.transcribe(&samples, None, "", &cancel).unwrap();
            let logits = engine.metal_execution().decoder.model.wk.logits.unwrap();
            let expected_logits = engine
                .metal_execution()
                .decoder
                .tensor_bytes(logits)
                .to_vec();
            engine.set_device_handoff(true);
            let actual = engine.transcribe(&samples, None, "", &cancel).unwrap();
            assert_eq!(actual.text, expected.text, "{name}");
            assert_eq!(actual.language, expected.language, "{name}");
            assert_eq!(
                engine.metal_execution().decoder.tensor_bytes(logits),
                expected_logits,
                "{name} final-step logits"
            );
        }
    }
}

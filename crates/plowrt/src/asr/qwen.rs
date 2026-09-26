use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::{
    frontend::{MelFeatures, PacketLogMelFrontend},
    Transcript,
};
use crate::exec::packet_runtime::ForwardPacket;
use crate::serve::template::ChatTemplate;
use crate::text::tokenizer::{load_tokenizer, Tokenize};
use crate::{Result, RuntimeError};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;

pub(crate) struct QwenPrefill<'a> {
    pub features: &'a MelFeatures,
    pub token_ids: &'a [u32],
    pub audio_positions: &'a [usize],
    pub hidden: usize,
}

pub(crate) struct QwenDecode {
    pub tokens: Vec<u32>,
    pub launched_rows: usize,
}

pub(crate) struct QwenPrefilled {
    pub token: u32,
    pub encoder_ms: f64,
    pub prefill_ms: f64,
}

pub(crate) struct PacketAudioEncoder {
    pub(crate) packet: ForwardPacket,
    feature_frames: usize,
    feature_bins: usize,
    output_rows: usize,
    output_width: usize,
    chunking: AudioChunking,
}

/// How features map to encoder rows: fixed-size frame chunks, each producing
/// `ceil(len / frame_stride)` rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AudioChunking {
    pub chunk_frames: usize,
    pub frame_stride: usize,
    pub round_bf16: bool,
}

impl AudioChunking {
    fn from_pipeline(parameter: impl Fn(&str) -> Option<u64>) -> Result<Self> {
        let get = |name: &str| {
            parameter(name)
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or_else(|| RuntimeError::Rejected(format!("audio packet parameter {name:?} is missing")))
        };
        Ok(Self {
            chunk_frames: get("input.chunk_frames")?,
            frame_stride: get("encoder.frame_stride")?,
            round_bf16: parameter("input.round_bf16").unwrap_or(0) == 1,
        })
    }

    pub fn rows(&self, frames: usize) -> usize {
        (frames / self.chunk_frames) * self.chunk_frames.div_ceil(self.frame_stride)
            + (frames % self.chunk_frames).div_ceil(self.frame_stride)
    }
}

impl PacketAudioEncoder {
    pub(crate) fn load(path: &std::path::Path, backend: &str) -> Result<Self> {
        let packet = ForwardPacket::load(path, "audio.encode", backend)?;
        let chunking = AudioChunking::from_pipeline(|name| packet.optional_parameter(name))?;
        let usize_param = |name: &str| {
            usize::try_from(packet.parameter(name)?)
                .map_err(|_| RuntimeError::Rejected(format!("audio packet parameter {name:?} overflows")))
        };
        let feature_frames = usize_param("input_frames")?;
        let output_rows = usize_param("output_rows")?;
        let feature_bins = usize_param("feature_bins")?;
        let output_width = usize_param("output_width")?;
        let chunks = feature_frames.div_ceil(chunking.chunk_frames);
        let input_bytes = chunks
            .checked_mul(feature_bins)
            .and_then(|elements| elements.checked_mul(chunking.chunk_frames * std::mem::size_of::<f32>()));
        let output_bytes = output_rows
            .checked_mul(output_width)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()));
        if feature_frames == 0
            || feature_bins == 0
            || output_width == 0
            || output_rows != chunking.rows(feature_frames)
            || input_bytes != Some(packet.input_bytes())
            || output_bytes != Some(packet.output_bytes())
        {
            return Err(RuntimeError::Rejected("audio packet geometry is inconsistent".into()));
        }
        Ok(Self { packet, feature_frames, feature_bins, output_rows, output_width, chunking })
    }

    pub(crate) fn accepts(&self, frames: usize) -> bool {
        (1..=self.feature_frames).contains(&frames)
    }

    pub(crate) fn encode(&mut self, features: &MelFeatures) -> Result<Vec<f32>> {
        if !self.accepts(features.frames)
            || self
                .feature_bins
                .checked_mul(features.frames)
                .is_none_or(|elements| features.values.len() != elements)
        {
            return Err(RuntimeError::Rejected("audio features do not match packet capacity".into()));
        }
        let chunk = self.chunking.chunk_frames;
        let chunks = self.feature_frames.div_ceil(chunk);
        let mut input = vec![0.0f32; chunks * self.feature_bins * chunk];
        for batch in 0..chunks {
            for bin in 0..self.feature_bins {
                for frame in 0..chunk {
                    let source_frame = batch * chunk + frame;
                    if source_frame < features.frames {
                        let v = features.values[bin * features.frames + source_frame];
                        input[(batch * self.feature_bins + bin) * chunk + frame] =
                            if self.chunking.round_bf16 { round_bf16(v) } else { v };
                    }
                }
            }
        }
        let mut output = vec![0.0f32; self.output_rows * self.output_width];
        let valid_rows = self.chunking.rows(features.frames);
        let valid_rows_u32 = u32::try_from(valid_rows)
            .map_err(|_| RuntimeError::Rejected("audio packet row count overflows".into()))?;
        self.packet.write("valid_rows", &valid_rows_u32.to_le_bytes())?;
        self.packet.run_for_capacity(
            features.frames.try_into().map_err(|_| RuntimeError::Rejected("audio feature frame count overflows".into()))?,
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice_mut(&mut output),
        )?;
        output.truncate(valid_rows * self.output_width);
        Ok(output)
    }

    pub(crate) fn output_width(&self) -> usize {
        self.output_width
    }
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

pub(crate) trait QwenExecution: Send {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn batch_capacity(&self) -> usize;
    fn max_context(&self) -> usize;
    fn prefill(&mut self, slot: usize, input: QwenPrefill<'_>) -> Result<QwenPrefilled>;
    fn decode(
        &mut self,
        positions: &[u32],
        kv_lengths: &[u32],
        token_ids: &[u32],
        occupied_rows: usize,
    ) -> Result<QwenDecode>;
}

/// A causal audio LM (encoder rows spliced over a placeholder token, then greedy decoding)
/// driven entirely by its packets: `encoder.pkt` carries the frontend and chunking, the causal
/// pipeline of `model.pkt` the prompt layout, markers, languages and stop ids. The checkpoint
/// directory supplies only the tokenizer and chat template.
pub struct QwenAsr {
    execution: Box<dyn QwenExecution>,
    frontend: PacketLogMelFrontend,
    tokenizer: Arc<dyn Tokenize>,
    template: Arc<ChatTemplate>,
    contract: AudioLmContract,
}

pub type AudioLmAsr = QwenAsr;

struct AudioLmContract {
    hidden: usize,
    placeholder: u32,
    stop: Vec<u32>,
    max_tokens: usize,
    context_max_tokens: usize,
    messages: serde_json::Value,
    marker: String,
    language_suffix: String,
    forbidden: Vec<String>,
    text_marker: String,
    language_prefix: String,
    language_none: String,
    languages: Vec<String>,
    aliases: Vec<(String, String)>,
    chunking: AudioChunking,
}

impl AudioLmContract {
    fn load(packet: &std::path::Path) -> Result<(Self, PacketLogMelFrontend)> {
        let asset = crate::exec::packet_runtime::PacketAsset::load(packet)?;
        let decoder = asset
            .pipelines()
            .iter()
            .find(|p| p.driver == "causal.v1" && p.parameters.get("overlay_rows").is_some_and(|&r| r > 0))
            .ok_or_else(|| RuntimeError::Rejected("packet has no causal audio pipeline".into()))?;
        let param = |name: &str| {
            decoder
                .parameters
                .get(name)
                .copied()
                .ok_or_else(|| RuntimeError::Rejected(format!("audio LM parameter {name:?} is missing")))
        };
        let text = |name: &str| {
            decoder
                .strings
                .get(name)
                .cloned()
                .ok_or_else(|| RuntimeError::Rejected(format!("audio LM string {name:?} is missing")))
        };
        let lines = |name: &str| -> Result<Vec<String>> {
            Ok(text(name)?.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect())
        };
        let to_usize = |v: u64| usize::try_from(v).map_err(|_| RuntimeError::Rejected("audio LM parameter overflows".into()));
        let stop = (0..param("stop.count")?)
            .map(|i| param(&format!("stop.{i}")).and_then(|id| u32::try_from(id).map_err(|_| RuntimeError::Rejected("stop id overflows".into()))))
            .collect::<Result<Vec<_>>>()?;
        let messages: serde_json::Value = serde_json::from_str(&text("prompt.messages")?)
            .map_err(|e| RuntimeError::Rejected(format!("audio LM prompt.messages: {e}")))?;

        let encoder_path = packet.with_file_name("encoder.pkt");
        let encoder = crate::exec::packet_runtime::PacketAsset::load(&encoder_path)?;
        let audio = encoder
            .pipelines()
            .iter()
            .find(|p| p.name == "audio.encode")
            .ok_or_else(|| RuntimeError::Rejected("encoder packet has no audio.encode pipeline".into()))?;
        let chunking = AudioChunking::from_pipeline(|name| audio.parameters.get(name).copied())?;
        let filterbank = audio
            .tensors
            .get("audio.frontend.filterbank")
            .ok_or_else(|| RuntimeError::Rejected("encoder packet has no frontend filterbank".into()))?;
        let raw = std::fs::read(&encoder_path).map_err(|source| RuntimeError::Io { path: encoder_path.clone(), source })?;
        let blob = crate::asset::devblob::DevBlob::parse(&raw)?;
        let bytes = blob
            .tensors
            .iter()
            .find(|t| t.name == filterbank.name)
            .and_then(|t| t.init.clone())
            .map(|range| blob.init[range].to_vec())
            .ok_or_else(|| RuntimeError::Rejected("frontend filterbank has no data".into()))?;
        let frontend = PacketLogMelFrontend::from_parameters(|name| audio.parameters.get(name).copied(), &bytes)?;
        Ok((
            Self {
                hidden: to_usize(param("hidden")?)?,
                placeholder: u32::try_from(param("audio.token_id")?).map_err(|_| RuntimeError::Rejected("audio token overflows".into()))?,
                stop,
                max_tokens: to_usize(param("output.max_tokens")?)?,
                context_max_tokens: to_usize(param("prompt.context_max_tokens")?)?,
                messages,
                marker: text("audio.marker")?,
                language_suffix: text("prompt.language_suffix")?,
                forbidden: lines("prompt.context_forbidden")?,
                text_marker: text("output.text_marker")?,
                language_prefix: text("output.language_prefix")?,
                language_none: text("output.language_none")?,
                languages: lines("languages")?,
                aliases: lines("language.aliases")?
                    .into_iter()
                    .filter_map(|l| l.split_once('=').map(|(a, b)| (a.to_owned(), b.to_owned())))
                    .collect(),
                chunking,
            },
            frontend,
        ))
    }

    /// `[text_marker]`-separated output with an optional `<language_prefix>NAME` header.
    fn parse(&self, output: &str, forced_language: Option<&str>) -> Result<Transcript> {
        let output = output.trim();
        if output.is_empty() {
            return Ok(Transcript { text: String::new(), language: None });
        }
        let (language, text) = if let Some(language) = forced_language {
            (Some(language.to_owned()), output)
        } else if let Some((header, text)) = output.split_once(self.text_marker.as_str()) {
            let language = header
                .strip_prefix(self.language_prefix.as_str())
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| RuntimeError::Rejected("invalid ASR language header".into()))?;
            let language = language.lines().next().unwrap_or_default().trim();
            ((!language.eq_ignore_ascii_case(&self.language_none)).then(|| language.to_owned()), text)
        } else {
            return Err(RuntimeError::Rejected("missing ASR text marker".into()));
        };
        Ok(Transcript { text: text.trim().to_owned(), language })
    }
}

struct PrefilledAudio {
    token: u32,
    prompt_len: usize,
    language: Option<String>,
    frontend_ms: f64,
    encoder_ms: f64,
    prefill_ms: f64,
}

impl super::Transcriber for QwenAsr {
    fn language(&self, requested: Option<&str>) -> Result<Option<String>> {
        QwenAsr::language(self, requested)
    }
    fn batch_capacity(&self) -> usize {
        self.execution.batch_capacity()
    }

    fn finalization_policy(&self) -> super::FinalizationPolicy {
        super::FinalizationPolicy {
            final_padding_samples: super::frontend::SAMPLE_RATE as usize,
            final_padding_amplitude: 100.0 / 32768.0,
        }
    }
    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &AtomicBool,
    ) -> Result<Transcript> {
        QwenAsr::transcribe(self, samples, language, context, cancel)
    }

    fn transcribe_batch(
        &mut self,
        requests: &[super::TranscriptionInput<'_>],
    ) -> Result<Vec<Result<Transcript>>> {
        self.transcribe_requests(requests)
    }
}

impl QwenAsr {
    pub fn load(packet: &std::path::Path, checkpoint: &std::path::Path) -> Result<Self> {
        Self::load_with_backend(packet, checkpoint, "auto").map(|(engine, _)| engine)
    }

    pub fn load_with_backend(
        packet: &std::path::Path,
        checkpoint: &std::path::Path,
        backend: &str,
    ) -> Result<(Self, &'static str)> {
        let (contract, frontend) = AudioLmContract::load(packet)?;
        let tokenizer = load_tokenizer(checkpoint);
        if tokenizer.is_byte_fallback() {
            return Err(RuntimeError::Rejected("ASR requires a real tokenizer".into()));
        }
        if tokenizer.encode(&contract.marker) != [contract.placeholder] {
            return Err(RuntimeError::Rejected("ASR audio marker does not tokenize to the packet's audio token".into()));
        }
        let template = ChatTemplate::load(checkpoint)
            .ok_or_else(|| RuntimeError::Rejected("ASR requires the checkpoint chat template".into()))?;
        let (execution, loaded_backend) = load_execution(packet, checkpoint, backend, contract.hidden)?;
        Ok((Self { execution, frontend, tokenizer, template, contract }, loaded_backend))
    }

    pub fn batch_capacity(&self) -> usize {
        self.execution.batch_capacity()
    }

    pub fn language(&self, requested: Option<&str>) -> Result<Option<String>> {
        let Some(requested) = requested else {
            return Ok(None);
        };
        let requested = self
            .contract
            .aliases
            .iter()
            .find(|(code, _)| code.eq_ignore_ascii_case(requested))
            .map_or(requested, |(_, name)| name.as_str());
        self.contract
            .languages
            .iter()
            .find(|name| name.eq_ignore_ascii_case(requested))
            .cloned()
            .map(Some)
            .ok_or_else(|| RuntimeError::Rejected(format!("unsupported ASR language {requested:?}")))
    }

    fn prefill_audio(
        &mut self,
        slot: usize,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &AtomicBool,
    ) -> Result<PrefilledAudio> {
        let started = std::time::Instant::now();
        let cancelled = || {
            if cancel.load(Ordering::Relaxed) {
                Err(RuntimeError::Rejected("ASR cancelled".into()))
            } else {
                Ok(())
            }
        };
        cancelled()?;
        let language = self.language(language)?;
        let c = &self.contract;
        if self.tokenizer.encode(context).len() > c.context_max_tokens
            || c.forbidden.iter().any(|marker| context.contains(marker.as_str()))
        {
            return Err(RuntimeError::Rejected(format!(
                "ASR prompt exceeds {} tokens or contains control markers",
                c.context_max_tokens
            )));
        }
        let log_mel = self.frontend.extract(samples)?;
        let mut features = MelFeatures { values: vec![0.0; log_mel.values.len()], frames: log_mel.frames };
        for frame in 0..log_mel.frames {
            for bin in 0..log_mel.bins {
                features.values[bin * log_mel.frames + frame] = log_mel.values[frame * log_mel.bins + bin];
            }
        }
        let rows = c.chunking.rows(features.frames);
        let mut messages = c.messages.clone();
        fill_context(&mut messages, context);
        let messages = messages.as_array().cloned().unwrap_or_default();
        let prompt = self.template.render(&messages).map_err(RuntimeError::Rejected)?;
        if prompt.matches(c.marker.as_str()).count() != 1 {
            return Err(RuntimeError::Rejected("ASR template needs exactly one audio marker".into()));
        }
        let mut prompt = prompt.replace(c.marker.as_str(), &c.marker.repeat(rows));
        if let Some(language) = &language {
            prompt.push_str(&c.language_suffix.replace("{language}", language));
        }
        let ids = self.tokenizer.encode(&prompt);
        let required_context = ids
            .len()
            .checked_add(c.max_tokens)
            .ok_or_else(|| RuntimeError::ContextLength("ASR prompt length overflows".into()))?;
        if required_context > self.execution.max_context() {
            return Err(RuntimeError::ContextLength(format!(
                "ASR needs {} prompt + {} output positions; bundle has {}",
                ids.len(),
                c.max_tokens,
                self.execution.max_context()
            )));
        }
        let positions: Vec<_> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == c.placeholder).then_some(i))
            .collect();
        if positions.len() != rows {
            return Err(RuntimeError::Rejected("ASR placeholder count mismatch".into()));
        }
        cancelled()?;
        let frontend_ms = started.elapsed().as_secs_f64() * 1000.0;
        let prepared = self.execution.prefill(
            slot,
            QwenPrefill {
                features: &features,
                token_ids: &ids,
                audio_positions: &positions,
                hidden: self.contract.hidden,
            },
        )?;
        cancelled()?;
        Ok(PrefilledAudio {
            token: prepared.token,
            prompt_len: ids.len(),
            language,
            frontend_ms,
            encoder_ms: prepared.encoder_ms,
            prefill_ms: prepared.prefill_ms,
        })
    }

    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &AtomicBool,
    ) -> Result<Transcript> {
        let started = std::time::Instant::now();
        let PrefilledAudio {
            mut token,
            prompt_len,
            language,
            frontend_ms,
            encoder_ms,
            prefill_ms,
        } = self.prefill_audio(0, samples, language, context, cancel)?;
        let cancelled = || {
            if cancel.load(Ordering::Relaxed) {
                Err(RuntimeError::Rejected("ASR cancelled".into()))
            } else {
                Ok(())
            }
        };
        let decode_started = std::time::Instant::now();
        let mut output = Vec::new();
        let max_tokens = self.contract.max_tokens;
        for step in 0..max_tokens {
            cancelled()?;
            if self.contract.stop.contains(&token) {
                let result =
                    self.contract.parse(&self.tokenizer.decode(&output), language.as_deref())?;
                let elapsed = started.elapsed().as_secs_f64();
                let audio_seconds = samples.len() as f64 / super::frontend::SAMPLE_RATE as f64;
                tracing::info!(
                    audio_seconds,
                    frontend_ms,
                    encoder_ms,
                    prefill_ms,
                    decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0,
                    output_tokens = output.len(),
                    total_ms = elapsed * 1000.0,
                    real_time_factor = elapsed / audio_seconds,
                    "ASR completed"
                );
                return Ok(result);
            }
            output.push(token);
            if step + 1 == max_tokens {
                break;
            }
            let pos = u32::try_from(prompt_len + step)
                .map_err(|_| RuntimeError::ContextLength("ASR position overflows".into()))?;
            token = self
                .execution
                .decode(&[pos], &[pos + 1], &[token], 1)?
                .tokens[0];
        }
        Err(RuntimeError::Rejected(
            "ASR exceeded output token limit".into(),
        ))
    }

    /// Fixed cohort with sequential encoder/prefill and shared decode steps.
    pub fn transcribe_batch(
        &mut self,
        samples: &[&[f32]],
        language: Option<&str>,
        context: &str,
        cancel: &AtomicBool,
    ) -> Result<Vec<Transcript>> {
        let requests: Vec<_> = samples
            .iter()
            .map(|samples| super::TranscriptionInput {
                samples: *samples,
                language,
                context,
                cancel,
            })
            .collect();
        self.transcribe_requests(&requests)?.into_iter().collect()
    }

    fn transcribe_requests(
        &mut self,
        requests: &[super::TranscriptionInput<'_>],
    ) -> Result<Vec<Result<Transcript>>> {
        let started = std::time::Instant::now();
        let batch = self.execution.batch_capacity();
        if requests.is_empty() || requests.len() > batch {
            return Err(RuntimeError::Rejected(format!(
                "ASR batch needs 1..={batch} recordings"
            )));
        }
        let mut results: Vec<Option<Result<Transcript>>> =
            (0..requests.len()).map(|_| None).collect();
        let mut ready = Vec::with_capacity(requests.len());
        let mut request_indices = Vec::with_capacity(requests.len());
        let mut frontend_ms = 0.0;
        let mut encoder_ms = 0.0;
        let mut prefill_ms = 0.0;
        for (request_index, request) in requests.iter().enumerate() {
            match self.prefill_audio(
                ready.len(),
                request.samples,
                request.language,
                request.context,
                request.cancel,
            ) {
                Ok(prefilled) => {
                    frontend_ms += prefilled.frontend_ms;
                    encoder_ms += prefilled.encoder_ms;
                    prefill_ms += prefilled.prefill_ms;
                    ready.push(Some(prefilled));
                    request_indices.push(request_index);
                }
                Err(error) => {
                    if !matches!(
                        error,
                        RuntimeError::Rejected(_) | RuntimeError::ContextLength(_)
                    ) {
                        return Err(error);
                    }
                    results[request_index] = Some(Err(error));
                }
            }
        }
        let mut tokens = vec![0; batch];
        for (slot, request) in ready.iter().enumerate() {
            if let Some(request) = request {
                tokens[slot] = request.token;
            }
        }
        let mut output = vec![Vec::new(); ready.len()];
        let mut pos = vec![0; batch];
        let mut kvlen = vec![1; batch];
        let decode_started = std::time::Instant::now();
        let mut launched_decode_rows = 0usize;
        let max_tokens = self.contract.max_tokens;
        for step in 0..max_tokens {
            for slot in 0..ready.len() {
                let request_index = request_indices[slot];
                if results[request_index].is_some() {
                    continue;
                }
                if requests[request_index].cancel.load(Ordering::Relaxed) {
                    results[request_index] =
                        Some(Err(RuntimeError::Rejected("ASR cancelled".into())));
                    ready[slot] = None;
                    pos[slot] = 0;
                    kvlen[slot] = 1;
                    tokens[slot] = 0;
                    continue;
                }
                let request = ready[slot]
                    .as_ref()
                    .expect("unfinished request is prepared");
                if self.contract.stop.contains(&tokens[slot]) {
                    results[request_index] = Some(self.contract.parse(
                        &self.tokenizer.decode(&output[slot]),
                        request.language.as_deref(),
                    ));
                    ready[slot] = None;
                    pos[slot] = 0;
                    kvlen[slot] = 1;
                    tokens[slot] = 0;
                } else {
                    output[slot].push(tokens[slot]);
                    pos[slot] = u32::try_from(request.prompt_len + step).map_err(|_| {
                        RuntimeError::ContextLength("ASR position overflows".into())
                    })?;
                    kvlen[slot] = pos[slot] + 1;
                }
            }
            let active_prefix =
                crate::sched::rungs::occupied_extent(ready.iter().map(Option::is_some));
            if active_prefix == 0 {
                tracing::info!(
                    recordings = requests.len(),
                    slots = batch,
                    decode_steps = step,
                    active_decode_rows = output.iter().map(Vec::len).sum::<usize>(),
                    launched_decode_rows,
                    output_tokens = ?output.iter().map(Vec::len).collect::<Vec<_>>(),
                    frontend_ms,
                    encoder_ms,
                    prefill_ms,
                    decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0,
                    total_ms = started.elapsed().as_secs_f64() * 1000.0,
                    "ASR batch completed"
                );
                return Ok(results.into_iter().map(Option::unwrap).collect());
            }
            if step + 1 == max_tokens {
                break;
            }
            let decoded = self
                .execution
                .decode(&pos, &kvlen, &tokens, active_prefix)?;
            launched_decode_rows += decoded.launched_rows;
            tokens = decoded.tokens;
        }
        for result in &mut results {
            if result.is_none() {
                *result = Some(Err(RuntimeError::Rejected(
                    "ASR exceeded output token limit".into(),
                )));
            }
        }
        Ok(results.into_iter().map(Option::unwrap).collect())
    }
}

fn load_execution(
    packet: &std::path::Path,
    checkpoint: &std::path::Path,
    backend: &str,
    hidden: usize,
) -> Result<(Box<dyn QwenExecution>, &'static str)> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if matches!(backend, "auto" | "metal") {
        return Ok((metal::load_execution(packet, checkpoint, hidden)?, "metal"));
    }
    #[cfg(feature = "cuda")]
    if matches!(backend, "auto" | "cuda") {
        return Ok((cuda::load_execution(packet, checkpoint, hidden)?, "cuda"));
    }
    let _ = (packet, checkpoint, hidden);
    Err(RuntimeError::Rejected(format!(
        "packet backend {backend:?} is unavailable for causal ASR"
    )))
}

/// Replace every `"{context}"` string in the packet's message layout with the request context.
fn fill_context(value: &mut serde_json::Value, context: &str) {
    match value {
        serde_json::Value::String(s) if s == "{context}" => *s = context.to_owned(),
        serde_json::Value::Array(items) => items.iter_mut().for_each(|v| fill_context(v, context)),
        serde_json::Value::Object(map) => map.values_mut().for_each(|v| fill_context(v, context)),
        _ => {}
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    /// Chunked rows equal three stride-2 convolutions over each 100-frame chunk.
    #[test]
    fn chunked_rows_cover_all_tails() {
        let chunking = AudioChunking { chunk_frames: 100, frame_stride: 8, round_bf16: true };
        for frames in 0usize..=3000 {
            let expected: usize = (0..frames)
                .step_by(100)
                .map(|start| {
                    let mut length = (frames - start).min(100);
                    for _ in 0..3 {
                        length = length.div_ceil(2);
                    }
                    length
                })
                .sum();
            assert_eq!(chunking.rows(frames), expected);
        }
    }

    #[test]
    fn marker_output_parse() {
        let c = AudioLmContract {
            hidden: 1,
            placeholder: 0,
            stop: vec![],
            max_tokens: 1,
            context_max_tokens: 1,
            messages: serde_json::Value::Null,
            marker: String::new(),
            language_suffix: String::new(),
            forbidden: vec![],
            text_marker: "<asr_text>".into(),
            language_prefix: "language ".into(),
            language_none: "none".into(),
            languages: vec![],
            aliases: vec![],
            chunking: AudioChunking { chunk_frames: 1, frame_stride: 1, round_bf16: false },
        };
        let result = c.parse("language English<asr_text>Hello.", None).unwrap();
        assert_eq!(result.language.as_deref(), Some("English"));
        assert_eq!(result.text, "Hello.");
        assert_eq!(c.parse("你好", Some("Chinese")).unwrap().text, "你好");
        assert!(c.parse("Hello.", None).is_err());
        assert!(c.parse("language None<asr_text>", None).unwrap().language.is_none());
        assert_eq!(c.parse("", None).unwrap().text, "");
    }

    #[test]
    fn context_fills_every_placeholder() {
        let mut v: serde_json::Value = serde_json::from_str(r#"[{"role":"system","content":"{context}"},{"role":"user","content":[{"type":"audio"}]}]"#).unwrap();
        fill_context(&mut v, "hi \"there\"");
        assert_eq!(v[0]["content"], "hi \"there\"");
        assert_eq!(v[1]["content"][0]["type"], "audio");
    }
}

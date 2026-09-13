use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::{
    frontend::{MelFeatures, QwenFrontend},
    Transcript,
};
use crate::exec::packet_runtime::ForwardPacket;
use crate::serve::template::ChatTemplate;
use crate::text::tokenizer::{load_tokenizer, Tokenize};
use crate::{Result, RuntimeError};

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
    packet: ForwardPacket,
    feature_frames: usize,
    feature_bins: usize,
    output_rows: usize,
    output_width: usize,
}

impl PacketAudioEncoder {
    pub(crate) fn load(path: &std::path::Path, backend: &str) -> Result<Self> {
        let packet = ForwardPacket::load(path, "audio.encode", backend)?;
        let feature_frames = usize::try_from(packet.parameter("input_frames")?)
            .map_err(|_| RuntimeError::Rejected("audio packet frame capacity overflows".into()))?;
        let output_rows = usize::try_from(packet.parameter("output_rows")?)
            .map_err(|_| RuntimeError::Rejected("audio packet row capacity overflows".into()))?;
        let feature_bins = usize::try_from(packet.parameter("feature_bins")?)
            .map_err(|_| RuntimeError::Rejected("audio packet feature width overflows".into()))?;
        let output_width = usize::try_from(
            packet.optional_parameter("output_width").unwrap_or(2048),
        )
        .map_err(|_| RuntimeError::Rejected("audio packet output width overflows".into()))?;
        let input_bytes = feature_frames
            .div_ceil(100)
            .checked_mul(feature_bins)
            .and_then(|elements| elements.checked_mul(100 * std::mem::size_of::<f32>()));
        let output_bytes = output_rows
            .checked_mul(output_width)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()));
        if feature_frames == 0
            || feature_frames > 3000
            || feature_bins == 0
            || output_width == 0
            || output_rows != super::qwen_audio_rows(feature_frames)
            || input_bytes != Some(packet.input_bytes())
            || output_bytes != Some(packet.output_bytes())
        {
            return Err(RuntimeError::Rejected(
                "audio packet geometry is inconsistent".into(),
            ));
        }
        Ok(Self {
            packet,
            feature_frames,
            feature_bins,
            output_rows,
            output_width,
        })
    }

    pub(crate) fn accepts(&self, frames: usize) -> bool {
        (50..=self.feature_frames).contains(&frames)
    }

    pub(crate) fn encode(&mut self, features: &MelFeatures) -> Result<Vec<f32>> {
        if !self.accepts(features.frames)
            || self
                .feature_bins
                .checked_mul(features.frames)
                .is_none_or(|elements| features.values.len() != elements)
        {
            return Err(RuntimeError::Rejected(
                "audio features do not match packet capacity".into(),
            ));
        }
        let chunks = self.feature_frames.div_ceil(100);
        let mut input = vec![0.0f32; chunks * self.feature_bins * 100];
        for batch in 0..chunks {
            for bin in 0..self.feature_bins {
                for frame in 0..100 {
                    let source_frame = batch * 100 + frame;
                    if source_frame < features.frames {
                        input[(batch * self.feature_bins + bin) * 100 + frame] =
                            round_bf16(features.values[bin * features.frames + source_frame]);
                    }
                }
            }
        }
        let mut output = vec![0.0f32; self.output_rows * self.output_width];
        let valid_rows = super::qwen_audio_rows(features.frames);
        let valid_rows_u32 = u32::try_from(valid_rows)
            .map_err(|_| RuntimeError::Rejected("audio packet row count overflows".into()))?;
        self.packet
            .write("valid_rows", &valid_rows_u32.to_le_bytes())?;
        self.packet.run_for_capacity(
            features.frames.try_into().map_err(|_| {
                RuntimeError::Rejected("audio feature frame count overflows".into())
            })?,
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

pub struct QwenAsr {
    execution: Box<dyn QwenExecution>,
    frontend: QwenFrontend,
    tokenizer: Arc<dyn Tokenize>,
    template: Arc<ChatTemplate>,
    languages: Vec<String>,
    hidden: usize,
    placeholder: u32,
    stop: Vec<u32>,
}

struct QwenCheckpoint {
    tokenizer: Arc<dyn Tokenize>,
    template: Arc<ChatTemplate>,
    languages: Vec<String>,
    hidden: usize,
    placeholder: u32,
    stop: Vec<u32>,
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
        let model = Self::checkpoint(checkpoint)?;
        let (execution, loaded_backend) =
            load_execution(packet, checkpoint, backend, model.hidden)?;
        Ok((Self::from_checkpoint(model, execution), loaded_backend))
    }

    pub fn batch_capacity(&self) -> usize {
        self.execution.batch_capacity()
    }

    fn checkpoint(checkpoint: &std::path::Path) -> Result<QwenCheckpoint> {
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(checkpoint.join("config.json"))
                .map_err(|e| RuntimeError::Rejected(e.to_string()))?,
        )
        .map_err(|e| RuntimeError::Rejected(e.to_string()))?;
        if cfg["model_type"] != "qwen3_asr" {
            return Err(RuntimeError::Rejected("unsupported ASR family".into()));
        }
        let processor: serde_json::Value = serde_json::from_slice(
            &std::fs::read(checkpoint.join("preprocessor_config.json"))
                .map_err(|e| RuntimeError::Rejected(e.to_string()))?,
        )
        .map_err(|e| RuntimeError::Rejected(e.to_string()))?;
        validate_frontend(&processor)?;
        let tokenizer = load_tokenizer(checkpoint);
        if tokenizer.is_byte_fallback() {
            return Err(RuntimeError::Rejected(
                "ASR requires a real tokenizer".into(),
            ));
        }
        let template = ChatTemplate::load(checkpoint)
            .ok_or_else(|| RuntimeError::Rejected("ASR requires the checkpoint template".into()))?;
        let hidden = cfg["thinker_config"]["text_config"]["hidden_size"]
            .as_u64()
            .ok_or_else(|| RuntimeError::Rejected("ASR hidden size missing".into()))?;
        let hidden = usize::try_from(hidden)
            .map_err(|_| RuntimeError::Rejected("ASR hidden size overflows".into()))?;
        if hidden == 0 {
            return Err(RuntimeError::Rejected("ASR hidden size is zero".into()));
        }
        let placeholder = cfg["thinker_config"]["audio_token_id"]
            .as_u64()
            .ok_or_else(|| RuntimeError::Rejected("ASR audio token missing".into()))?;
        let placeholder = u32::try_from(placeholder)
            .map_err(|_| RuntimeError::Rejected("ASR audio token overflows".into()))?;
        for (marker, field) in [
            ("<|audio_pad|>", "audio_token_id"),
            ("<|audio_start|>", "audio_start_token_id"),
            ("<|audio_end|>", "audio_end_token_id"),
        ] {
            let id = cfg["thinker_config"][field]
                .as_u64()
                .and_then(|id| u32::try_from(id).ok());
            if id.is_none_or(|id| tokenizer.encode(marker) != [id]) {
                return Err(RuntimeError::Rejected("ASR tokenizer id mismatch".into()));
            }
        }
        let languages: Vec<String> = serde_json::from_value(cfg["support_languages"].clone())
            .map_err(|e| RuntimeError::Rejected(e.to_string()))?;
        let stop = read_eos_ids(checkpoint)?;
        if stop.is_empty() {
            return Err(RuntimeError::Rejected("ASR EOS ids missing".into()));
        }
        Ok(QwenCheckpoint {
            tokenizer,
            template,
            languages,
            hidden,
            placeholder,
            stop,
        })
    }

    fn from_checkpoint(checkpoint: QwenCheckpoint, execution: Box<dyn QwenExecution>) -> Self {
        Self {
            execution,
            frontend: QwenFrontend::default(),
            tokenizer: checkpoint.tokenizer,
            template: checkpoint.template,
            languages: checkpoint.languages,
            hidden: checkpoint.hidden,
            placeholder: checkpoint.placeholder,
            stop: checkpoint.stop,
        }
    }

    pub fn language(&self, requested: Option<&str>) -> Result<Option<String>> {
        let Some(requested) = requested else {
            return Ok(None);
        };
        let aliases = [
            ("zh", "Chinese"),
            ("en", "English"),
            ("yue", "Cantonese"),
            ("ar", "Arabic"),
            ("de", "German"),
            ("fr", "French"),
            ("es", "Spanish"),
            ("pt", "Portuguese"),
            ("id", "Indonesian"),
            ("it", "Italian"),
            ("ko", "Korean"),
            ("ru", "Russian"),
            ("th", "Thai"),
            ("vi", "Vietnamese"),
            ("ja", "Japanese"),
            ("tr", "Turkish"),
            ("hi", "Hindi"),
            ("ms", "Malay"),
            ("nl", "Dutch"),
            ("sv", "Swedish"),
            ("da", "Danish"),
            ("fi", "Finnish"),
            ("pl", "Polish"),
            ("cs", "Czech"),
            ("fil", "Filipino"),
            ("fa", "Persian"),
            ("el", "Greek"),
            ("hu", "Hungarian"),
            ("mk", "Macedonian"),
            ("ro", "Romanian"),
        ];
        let requested = aliases
            .iter()
            .find(|(code, _)| code.eq_ignore_ascii_case(requested))
            .map(|(_, name)| *name)
            .unwrap_or(requested);
        self.languages
            .iter()
            .find(|name| name.eq_ignore_ascii_case(requested))
            .cloned()
            .map(Some)
            .ok_or_else(|| {
                RuntimeError::Rejected(format!("unsupported ASR language {requested:?}"))
            })
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
        if self.tokenizer.encode(context).len() > 256
            || context.contains("<|")
            || context.contains("<asr_text>")
        {
            return Err(RuntimeError::Rejected(
                "ASR prompt exceeds 256 tokens or contains control markers".into(),
            ));
        }
        let features = self.frontend.extract(samples)?;
        let rows = super::qwen_audio_rows(features.frames);
        let messages = [
            serde_json::json!({"role":"system","content":context}),
            serde_json::json!({"role":"user","content":[{"type":"audio"}]}),
        ];
        let prompt = self
            .template
            .render(&messages)
            .map_err(RuntimeError::Rejected)?;
        if prompt.matches("<|audio_pad|>").count() != 1 {
            return Err(RuntimeError::Rejected(
                "ASR template needs exactly one audio marker".into(),
            ));
        }
        let mut prompt = prompt.replace("<|audio_pad|>", &"<|audio_pad|>".repeat(rows));
        if let Some(language) = &language {
            prompt.push_str(&format!("language {language}<asr_text>"));
        }
        let ids = self.tokenizer.encode(&prompt);
        let required_context = ids
            .len()
            .checked_add(1024)
            .ok_or_else(|| RuntimeError::ContextLength("ASR prompt length overflows".into()))?;
        if required_context > self.execution.max_context() {
            return Err(RuntimeError::ContextLength(format!(
                "ASR needs {} prompt + 1024 output positions; bundle has {}",
                ids.len(),
                self.execution.max_context()
            )));
        }
        let positions: Vec<_> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == self.placeholder).then_some(i))
            .collect();
        if positions.len() != rows {
            return Err(RuntimeError::Rejected(
                "ASR placeholder count mismatch".into(),
            ));
        }
        cancelled()?;
        let frontend_ms = started.elapsed().as_secs_f64() * 1000.0;
        let prepared = self.execution.prefill(
            slot,
            QwenPrefill {
                features: &features,
                token_ids: &ids,
                audio_positions: &positions,
                hidden: self.hidden,
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
        for step in 0..1024 {
            cancelled()?;
            if self.stop.contains(&token) {
                let result =
                    super::parse_qwen_output(&self.tokenizer.decode(&output), language.as_deref())?;
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
            if step == 1023 {
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
        for step in 0..1024 {
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
                if self.stop.contains(&tokens[slot]) {
                    results[request_index] = Some(super::parse_qwen_output(
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
            if step == 1023 {
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
    let _ = (packet, checkpoint, hidden);
    Err(RuntimeError::Rejected(format!(
        "packet backend {backend:?} is unavailable for causal ASR"
    )))
}

fn validate_frontend(v: &serde_json::Value) -> Result<()> {
    for (key, expected) in [
        ("feature_size", 128),
        ("hop_length", 160),
        ("n_fft", 400),
        ("chunk_length", 30),
        ("n_samples", 480000),
        ("nb_max_frames", 3000),
    ] {
        if v[key].as_u64() != Some(expected) {
            return Err(RuntimeError::Rejected(format!(
                "unsupported ASR frontend {key}"
            )));
        }
    }
    if v["feature_extractor_type"] != "WhisperFeatureExtractor"
        || v["padding_side"] != "right"
        || v["padding_value"].as_f64() != Some(0.0)
        || v["dither"].as_f64() != Some(0.0)
        || v.get("sampling_rate")
            .is_some_and(|rate| rate.as_u64() != Some(16000))
    {
        return Err(RuntimeError::Rejected(
            "unsupported ASR frontend normalization or sample rate".into(),
        ));
    }
    Ok(())
}

fn read_eos_ids(checkpoint: &std::path::Path) -> Result<Vec<u32>> {
    for file in ["generation_config.json", "config.json"] {
        let Ok(bytes) = std::fs::read(checkpoint.join(file)) else {
            continue;
        };
        let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        match config.get("eos_token_id") {
            Some(serde_json::Value::Number(id)) => {
                if let Some(id) = id.as_u64() {
                    return u32::try_from(id)
                        .map(|id| vec![id])
                        .map_err(|_| RuntimeError::Rejected("ASR EOS id overflows".into()));
                }
            }
            Some(serde_json::Value::Array(ids)) => {
                let ids: Option<Vec<_>> = ids
                    .iter()
                    .map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                    .collect();
                if let Some(ids) = ids {
                    if !ids.is_empty() {
                        return Ok(ids);
                    }
                } else {
                    return Err(RuntimeError::Rejected("invalid ASR EOS ids".into()));
                }
            }
            _ => {}
        }
    }
    Ok(Vec::new())
}

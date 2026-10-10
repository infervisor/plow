use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::{
    frontend::{MelFeatures, PacketLogMelFrontend},
    Transcript,
};
use crate::exec::packet_runtime::{ForwardPacket, PacketTensor};
use crate::serve::template::ChatTemplate;
use crate::text::tokenizer::{load_tokenizer, Tokenize};
use crate::{Result, RuntimeError};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;

pub(crate) struct AudioLmPrefill<'a> {
    pub features: &'a MelFeatures,
    pub token_ids: &'a [u32],
    pub audio_positions: &'a [usize],
    pub hidden: usize,
}

pub(crate) struct AudioLmDecode {
    pub tokens: Vec<u32>,
    pub launched_rows: usize,
}

pub(crate) struct AudioLmPrefilled {
    pub token: u32,
    pub encoder_ms: f64,
    pub prefill_ms: f64,
}

pub(crate) struct PacketAudioEncoder {
    pub(crate) packet: ForwardPacket,
    /// Largest single-utterance capacity.
    feature_frames: usize,
    feature_bins: usize,
    output_width: usize,
    chunking: AudioChunking,
    /// Single-utterance capacities (frames), ascending.
    single: Vec<(u32, Vec<usize>)>,
    /// Packed capacities (chunks of any number of utterances), ascending.
    packed: Vec<(usize, Vec<usize>)>,
    input: PacketTensor,
    output: PacketTensor,
    valid_rows: PacketTensor,
    groups: Option<PacketTensor>,
    split_rows: Option<PacketTensor>,
    /// Attention window of the packed programs, in rows.
    window_rows: usize,
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
        let input_frames = usize_param("input_frames")?;
        let output_rows = usize_param("output_rows")?;
        let feature_bins = usize_param("feature_bins")?;
        let output_width = usize_param("output_width")?;
        let pipeline = packet.pipeline();
        let single = pipeline.program_capacity_sequences("forward")?;
        let packed: Vec<(usize, Vec<usize>)> = pipeline
            .program_capacity_sequences("packed")?
            .into_iter()
            .map(|(chunks, programs)| (chunks as usize, programs))
            .collect();
        let groups = pipeline.tensor("groups").ok();
        let split_rows = pipeline.tensor("split_rows").ok();
        let window_rows = packet.optional_parameter("attention.window_rows").unwrap_or(0) as usize;
        let feature_frames = single.last().map_or(input_frames, |&(frames, _)| frames as usize);
        let max_chunks = input_frames.div_ceil(chunking.chunk_frames);
        let input_bytes = max_chunks
            .checked_mul(feature_bins)
            .and_then(|elements| elements.checked_mul(chunking.chunk_frames * std::mem::size_of::<f32>()));
        let output_bytes = output_rows
            .checked_mul(output_width)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()));
        if input_frames == 0
            || feature_bins == 0
            || output_width == 0
            || feature_frames > input_frames
            || output_rows != chunking.rows(input_frames)
            || input_bytes != Some(packet.input_bytes())
            || output_bytes != Some(packet.output_bytes())
            || packed.last().is_some_and(|&(chunks, _)| chunks > max_chunks)
            || (!packed.is_empty()
                && (window_rows == 0
                    || groups.is_none_or(|g| g.bytes < 4 * (1 + 2 * packed.last().unwrap().0))
                    || split_rows.is_none_or(|t| t.bytes < 4 * output_rows)))
        {
            return Err(RuntimeError::Rejected("audio packet geometry is inconsistent".into()));
        }
        Ok(Self {
            input: pipeline.tensor("input")?,
            output: pipeline.tensor("output")?,
            valid_rows: pipeline.tensor("valid_rows")?,
            packet,
            feature_frames,
            feature_bins,
            output_width,
            chunking,
            single,
            packed,
            groups,
            split_rows,
            window_rows,
        })
    }

    pub(crate) fn accepts(&self, frames: usize) -> bool {
        (1..=self.feature_frames).contains(&frames)
    }

    /// Largest number of 100-frame chunks one packed launch holds (0: no packed programs).
    pub(crate) fn max_packed_chunks(&self) -> usize {
        self.packed.last().map_or(0, |&(chunks, _)| chunks)
    }

    pub(crate) fn chunks(&self, frames: usize) -> usize {
        frames.div_ceil(self.chunking.chunk_frames)
    }

    fn check(&self, features: &MelFeatures) -> Result<()> {
        if features.frames == 0
            || self
                .feature_bins
                .checked_mul(features.frames)
                .is_none_or(|elements| features.values.len() != elements)
        {
            return Err(RuntimeError::Rejected("audio features do not match packet capacity".into()));
        }
        Ok(())
    }

    /// `[chunk][bin][frame]` encoder input for features starting at chunk `first`, tail zeroed.
    fn stage(&self, features: &MelFeatures, input: &mut [f32], first: usize) {
        let chunk = self.chunking.chunk_frames;
        for batch in 0..self.chunks(features.frames) {
            for bin in 0..self.feature_bins {
                let row = &mut input[((first + batch) * self.feature_bins + bin) * chunk..][..chunk];
                let source = &features.values[bin * features.frames..][..features.frames];
                for (frame, value) in row.iter_mut().enumerate() {
                    *value = match source.get(batch * chunk + frame) {
                        Some(&v) if self.chunking.round_bf16 => round_bf16(v),
                        Some(&v) => v,
                        None => 0.0,
                    };
                }
            }
        }
    }

    fn launch(&mut self, programs: &[usize], writes: &[(PacketTensor, &[u8])], output: &mut [u8]) -> Result<()> {
        let runtime = self.packet.runtime_mut();
        runtime.begin_execution()?;
        let result = (|| {
            for &(tensor, bytes) in writes {
                runtime.write_tensor_at(tensor, 0, bytes)?;
            }
            runtime.run_sequence(programs)?;
            runtime.read_tensor_at(self.output, 0, output)
        })();
        let ended = runtime.end_execution();
        result.and(ended)
    }

    pub(crate) fn encode(&mut self, features: &MelFeatures) -> Result<Vec<f32>> {
        self.check(features)?;
        let requested = u32::try_from(features.frames)
            .map_err(|_| RuntimeError::Rejected("audio feature frame count overflows".into()))?;
        let (capacity, programs) = self
            .single
            .iter()
            .find(|(capacity, _)| *capacity >= requested)
            .map(|(capacity, programs)| (*capacity as usize, programs.clone()))
            .ok_or_else(|| RuntimeError::Rejected("audio features do not match packet capacity".into()))?;
        let chunk = self.chunking.chunk_frames;
        let mut input = vec![0.0f32; self.chunks(capacity) * self.feature_bins * chunk];
        self.stage(features, &mut input, 0);
        let valid_rows = self.chunking.rows(features.frames);
        let valid_rows_u32 = u32::try_from(valid_rows)
            .map_err(|_| RuntimeError::Rejected("audio packet row count overflows".into()))?;
        let mut output = vec![0.0f32; valid_rows * self.output_width];
        let writes = [(self.input, bytemuck::cast_slice(&input)), (self.valid_rows, &valid_rows_u32.to_le_bytes()[..])];
        self.launch(&programs, &writes, bytemuck::cast_slice_mut(&mut output))?;
        Ok(output)
    }

    /// Several utterances in one packed launch (their chunks back to back, each attending only
    /// within its own windows and summing as its single-utterance capacity does): every item's
    /// rows bit for bit as [`Self::encode`] gives them. Together they must fit
    /// [`Self::max_packed_chunks`].
    pub(crate) fn encode_packed(&mut self, items: &[&MelFeatures]) -> Result<Vec<Vec<f32>>> {
        let (Some(groups), Some(split_rows), Some(&(max_chunks, _))) = (self.groups, self.split_rows, self.packed.last())
        else {
            return items.iter().map(|features| self.encode(features)).collect();
        };
        for features in items {
            self.check(features)?;
        }
        let total: usize = items.iter().map(|f| self.chunks(f.frames)).sum();
        if total == 0 || total > max_chunks {
            return Err(RuntimeError::Rejected(format!("{total} audio chunks exceed the packed encoder")));
        }
        let programs = self.packed.iter().find(|&&(chunks, _)| chunks >= total).map(|(_, p)| p.clone()).unwrap();
        let chunk = self.chunking.chunk_frames;
        let chunk_rows = chunk.div_ceil(self.chunking.frame_stride);
        let mut input = vec![0.0f32; total * self.feature_bins * chunk];
        let mut table = vec![0u32];
        let mut reference = vec![0u32; total * chunk_rows];
        let mut spans = Vec::with_capacity(items.len());
        let mut first = 0;
        for features in items {
            let capacity = self
                .single
                .iter()
                .map(|&(frames, _)| frames as usize)
                .find(|&frames| frames >= features.frames)
                .ok_or_else(|| RuntimeError::Rejected("audio exceeds every single-utterance capacity".into()))?;
            reference[first * chunk_rows..(first + self.chunks(features.frames)) * chunk_rows]
                .fill(self.chunking.rows(capacity) as u32);
            self.stage(features, &mut input, first);
            let (row, rows) = (first * chunk_rows, self.chunking.rows(features.frames));
            for window in (0..rows).step_by(self.window_rows) {
                table.extend([(row + window) as u32, (rows - window).min(self.window_rows) as u32]);
            }
            spans.push((row, rows));
            first += self.chunks(features.frames);
        }
        table[0] = ((table.len() - 1) / 2) as u32;
        let mut output = vec![0.0f32; total * chunk_rows * self.output_width];
        let writes = [
            (self.input, bytemuck::cast_slice(&input)),
            (groups, bytemuck::cast_slice(&table)),
            (split_rows, bytemuck::cast_slice(&reference)),
        ];
        self.launch(&programs, &writes, bytemuck::cast_slice_mut(&mut output))?;
        let width = self.output_width;
        Ok(spans.into_iter().map(|(row, rows)| output[row * width..(row + rows) * width].to_vec()).collect())
    }

    /// Run every capacity once (no valid rows): each sequence's CUDA graph is captured now, not
    /// under load, where a capture overlapping another thread's context synchronize fails.
    pub(crate) fn warm(&mut self) -> Result<()> {
        let mut writes = vec![(self.valid_rows, &[0u8; 4][..])];
        if let Some(groups) = self.groups {
            writes.push((groups, &[0u8; 4][..]));
        }
        let sequences: Vec<Vec<usize>> =
            self.single.iter().map(|(_, p)| p.clone()).chain(self.packed.iter().map(|(_, p)| p.clone())).collect();
        for programs in sequences {
            self.launch(&programs, &writes, &mut [])?;
        }
        Ok(())
    }

    /// Encoder rows one attention window spans (0: unwindowed).
    pub(crate) fn window_rows(&self) -> usize {
        self.window_rows
    }

    pub(crate) fn output_width(&self) -> usize {
        self.output_width
    }

    /// GPU time of the last launch.
    pub(crate) fn last_gpu_us(&self) -> f64 {
        self.packet.runtime().last_run_us()
    }
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

pub(crate) trait AudioLmExecution: Send {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn batch_capacity(&self) -> usize;
    fn max_context(&self) -> usize;
    fn prefill(&mut self, slot: usize, input: AudioLmPrefill<'_>) -> Result<AudioLmPrefilled>;
    fn decode(
        &mut self,
        positions: &[u32],
        kv_lengths: &[u32],
        token_ids: &[u32],
        occupied_rows: usize,
    ) -> Result<AudioLmDecode>;
}

/// A causal audio LM (encoder rows spliced over a placeholder token, then greedy decoding)
/// driven entirely by its packets: `encoder.pkt` carries the frontend and chunking, the causal
/// pipeline of `model.pkt` the prompt layout, markers, languages and stop ids. The checkpoint
/// directory supplies only the tokenizer and chat template.
pub struct AudioLmAsr {
    execution: Box<dyn AudioLmExecution>,
    prompt: AudioLmPrompt,
}

/// The host side of an audio LM request: log-mel frontend, prompt layout, output parsing. Shared by
/// the private [`AudioLmAsr`] loop and the serve mux path (`asr::serving`).
pub struct AudioLmPrompt {
    frontend: PacketLogMelFrontend,
    tokenizer: Arc<dyn Tokenize>,
    template: Arc<ChatTemplate>,
    contract: AudioLmContract,
}

/// One request's prompt: encoder features, token ids, and the ids' audio placeholder positions
/// (one per encoder row, in order).
pub struct AudioLmRequest {
    pub features: MelFeatures,
    pub ids: Vec<u32>,
    pub audio_positions: Vec<usize>,
    pub language: Option<String>,
}

struct AudioLmContract {
    hidden: usize,
    placeholder: u32,
    stop: Vec<u32>,
    max_tokens: usize,
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
        // The serving frontend decodes 16 kHz audio into windows of at most 30 s. A packet that
        // declares other audio is refused here instead of being fed the wrong signal.
        for (name, have) in [
            ("audio.sample_rate", u64::from(super::frontend::SAMPLE_RATE)),
            ("audio.max_seconds", (super::frontend::MAX_SAMPLES / super::frontend::SAMPLE_RATE as usize) as u64),
        ] {
            if let Some(&want) = decoder.parameters.get(name) {
                if want != have {
                    return Err(RuntimeError::Rejected(format!(
                        "packet {name} = {want}; this plowrt's audio frontend serves {have}"
                    )));
                }
            }
        }
        let lines = |name: &str| -> Result<Vec<String>> {
            Ok(text(name)?.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect())
        };
        let to_usize = |v: u64| usize::try_from(v).map_err(|_| RuntimeError::Rejected("audio LM parameter overflows".into()));
        let stop = (0..param("stop.count")?)
            .map(|i| param(&format!("stop.{i}")).and_then(|id| u32::try_from(id).map_err(|_| RuntimeError::Rejected("stop id overflows".into()))))
            .collect::<Result<Vec<_>>>()?;
        let messages: serde_json::Value = serde_json::from_str(&text("prompt.messages")?)
            .map_err(|e| RuntimeError::Rejected(format!("audio LM prompt.messages: {e}")))?;

        let encoder_path = crate::exec::packet_runtime::stage_packet(packet, "encoder.packet", "encoder.pkt")?;
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
                // `prompt.context_max_tokens` (older packets) is not read: the context is fitted
                // to each request's window instead.
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

impl super::Transcriber for AudioLmAsr {
    fn language(&self, requested: Option<&str>) -> Result<Option<String>> {
        AudioLmAsr::language(self, requested)
    }
    fn batch_capacity(&self) -> usize {
        self.execution.batch_capacity()
    }

    fn finalization_policy(&self) -> super::FinalizationPolicy {
        self.prompt.finalization_policy()
    }
    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &AtomicBool,
    ) -> Result<Transcript> {
        AudioLmAsr::transcribe(self, samples, language, context, cancel)
    }

    fn transcribe_batch(
        &mut self,
        requests: &[super::TranscriptionInput<'_>],
    ) -> Result<Vec<Result<Transcript>>> {
        self.transcribe_requests(requests)
    }
}

impl AudioLmPrompt {
    pub fn load(packet: &std::path::Path, checkpoint: &std::path::Path) -> Result<Self> {
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
        Ok(Self { frontend, tokenizer, template, contract })
    }

    pub fn hidden(&self) -> usize {
        self.contract.hidden
    }

    pub fn max_tokens(&self) -> usize {
        self.contract.max_tokens
    }

    pub fn stop(&self) -> &[u32] {
        &self.contract.stop
    }

    pub fn finalization_policy(&self) -> super::FinalizationPolicy {
        super::FinalizationPolicy {
            final_padding_samples: super::frontend::SAMPLE_RATE as usize,
            final_padding_amplitude: 100.0 / 32768.0,
        }
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

    /// Features, prompt ids and placeholder positions for one recording; the prompt must leave
    /// `max_context` at least one output position.
    pub fn request(
        &self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        max_context: usize,
    ) -> Result<AudioLmRequest> {
        let language = self.language(language)?;
        let features = self.features(samples)?;
        let rows = self.contract.chunking.rows(features.frames);
        let (ids, audio_positions) = self.prompt(rows, language.as_deref(), context, max_context)?;
        Ok(AudioLmRequest { features, ids, audio_positions, language })
    }

    /// Its length is not checked here: [`Self::prompt`] fits the context to each request's window.
    /// `context` without the template's control markers: a client's text cannot open or close
    /// a turn of the prompt. Removal repeats until none remains (a removal can join a new one).
    fn clean_context<'a>(&self, context: &'a str) -> std::borrow::Cow<'a, str> {
        let mut context = std::borrow::Cow::Borrowed(context);
        while let Some(marker) = self.contract.forbidden.iter().find(|m| !m.is_empty() && context.contains(m.as_str())) {
            context = context.replace(marker.as_str(), " ").into();
        }
        context
    }

    /// Positions kept free for the transcript: one per audio row (about 13 a second, where speech
    /// rarely needs 6 tokens a second) and 64 more.
    fn output_reserve(rows: usize) -> usize {
        rows + 64
    }

    /// `context` cut to what the window leaves after the template, the audio and the output
    /// reserve (and `PLOW_ASR_CONTEXT_MAX_TOKENS` when set), keeping its most recent words.
    fn fit_context<'a>(
        &self,
        rows: usize,
        language: Option<&str>,
        context: &'a str,
        max_context: usize,
    ) -> Result<&'a str> {
        if context.is_empty() {
            return Ok(context);
        }
        let base = self.encode_prompt(rows, language, "")?.len();
        let window = max_context.saturating_sub(base + Self::output_reserve(rows));
        let cap = crate::config::RuntimeConfig::get().asr_context_max_tokens;
        let budget = if cap == 0 { window } else { window.min(cap) };
        let fits = |text: &str| self.tokenizer.encode(text).len() <= budget;
        if fits(context) {
            return Ok(context);
        }
        // Word starts; dropping more leading words never adds tokens, so bisect the first fit.
        let starts: Vec<usize> = context
            .char_indices()
            .filter(|&(i, ch)| !ch.is_whitespace() && (i == 0 || context[..i].ends_with(char::is_whitespace)))
            .map(|(i, _)| i)
            .collect();
        let (mut lo, mut hi) = (0, starts.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if fits(&context[starts[mid]..]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let kept = starts.get(lo).map_or("", |&start| &context[start..]);
        tracing::debug!(budget, dropped_words = lo, kept_words = starts.len() - lo, "ASR context trimmed to its window");
        Ok(kept)
    }

    fn encode_prompt(&self, rows: usize, language: Option<&str>, context: &str) -> Result<Vec<u32>> {
        let c = &self.contract;
        let mut messages = c.messages.clone();
        fill_context(&mut messages, context);
        let messages = messages.as_array().cloned().unwrap_or_default();
        let prompt = self.template.render(&messages).map_err(RuntimeError::Rejected)?;
        if prompt.matches(c.marker.as_str()).count() != 1 {
            return Err(RuntimeError::Rejected("ASR template needs exactly one audio marker".into()));
        }
        let mut prompt = prompt.replace(c.marker.as_str(), &c.marker.repeat(rows));
        if let Some(language) = language {
            prompt.push_str(&c.language_suffix.replace("{language}", language));
        }
        Ok(self.tokenizer.encode(&prompt))
    }

    /// The encoder's input for one recording: `[bin][frame]` log-mel features.
    pub fn features(&self, samples: &[f32]) -> Result<MelFeatures> {
        let log_mel = self.frontend.extract(samples)?;
        let mut features = MelFeatures { values: vec![0.0; log_mel.values.len()], frames: log_mel.frames };
        for frame in 0..log_mel.frames {
            for bin in 0..log_mel.bins {
                features.values[bin * log_mel.frames + frame] = log_mel.values[frame * log_mel.bins + bin];
            }
        }
        Ok(features)
    }

    /// Prompt ids around `rows` audio placeholders, and the placeholders' positions. `language`
    /// is the resolved name ([`Self::language`]); `context` is fitted to `max_context`.
    pub fn prompt(
        &self,
        rows: usize,
        language: Option<&str>,
        context: &str,
        max_context: usize,
    ) -> Result<(Vec<u32>, Vec<usize>)> {
        let c = &self.contract;
        let cleaned = self.clean_context(context);
        let context = self.fit_context(rows, language, &cleaned, max_context)?;
        let ids = self.encode_prompt(rows, language, context)?;
        // Served, the transcript budget (`max_tokens`) is fitted to the context by the mux, so a
        // narrowed bound (`--live-ctx-models`) caps long transcripts instead of refusing audio.
        if ids.len() >= max_context {
            return Err(RuntimeError::ContextLength(format!(
                "ASR needs {} prompt positions plus output; bundle has {}",
                ids.len(),
                max_context
            )));
        }
        let audio_positions: Vec<_> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == c.placeholder).then_some(i))
            .collect();
        if audio_positions.len() != rows {
            return Err(RuntimeError::Rejected("ASR placeholder count mismatch".into()));
        }
        Ok((ids, audio_positions))
    }

    /// Resolve `language` once for a stream of partial prompts.
    pub fn stream_language(&self, language: Option<&str>, _context: &str) -> Result<Option<String>> {
        self.language(language)
    }

    pub(crate) fn chunking(&self) -> AudioChunking {
        self.contract.chunking
    }

    /// The transcript text generated so far (`None` until the text itself begins): what a
    /// streamed transcript has shown when these ids have been decoded.
    pub fn text_so_far(&self, output: &[u32], language: Option<&str>) -> Option<String> {
        let decoded = self.tokenizer.decode(output);
        let text = if language.is_some() {
            decoded.as_str()
        } else {
            decoded.split_once(self.contract.text_marker.as_str())?.1
        };
        Some(text.trim_start().trim_end_matches('\u{fffd}').to_owned())
    }

    /// The transcript of generated ids (stop token excluded).
    pub fn transcript(&self, output: &[u32], language: Option<&str>) -> Result<Transcript> {
        self.contract.parse(&self.tokenizer.decode(output), language)
    }
}

impl AudioLmAsr {
    pub fn load(packet: &std::path::Path, checkpoint: &std::path::Path) -> Result<Self> {
        Self::load_with_backend(packet, checkpoint, "auto").map(|(engine, _)| engine)
    }

    pub fn load_with_backend(
        packet: &std::path::Path,
        checkpoint: &std::path::Path,
        backend: &str,
    ) -> Result<(Self, &'static str)> {
        let prompt = AudioLmPrompt::load(packet, checkpoint)?;
        let (execution, loaded_backend) = load_execution(packet, checkpoint, backend, prompt.contract.hidden)?;
        Ok((Self { execution, prompt }, loaded_backend))
    }

    pub fn batch_capacity(&self) -> usize {
        self.execution.batch_capacity()
    }

    pub fn language(&self, requested: Option<&str>) -> Result<Option<String>> {
        self.prompt.language(requested)
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
        // This loop decodes the whole `max_tokens` budget itself, so it needs the room up front:
        // the prompt (and its context) gets the rest of the window.
        let max_tokens = self.prompt.contract.max_tokens;
        let window = self.execution.max_context().saturating_sub(max_tokens).saturating_add(1);
        let AudioLmRequest { features, ids, audio_positions: positions, language } =
            self.prompt.request(samples, language, context, window)?;
        if ids.len() + max_tokens > self.execution.max_context() {
            return Err(RuntimeError::ContextLength(format!(
                "ASR needs {} prompt + {max_tokens} output positions; bundle has {}",
                ids.len(),
                self.execution.max_context()
            )));
        }
        cancelled()?;
        let frontend_ms = started.elapsed().as_secs_f64() * 1000.0;
        let prepared = self.execution.prefill(
            slot,
            AudioLmPrefill {
                features: &features,
                token_ids: &ids,
                audio_positions: &positions,
                hidden: self.prompt.contract.hidden,
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
        let max_tokens = self.prompt.contract.max_tokens;
        for step in 0..max_tokens {
            cancelled()?;
            if self.prompt.contract.stop.contains(&token) {
                let result =
                    self.prompt.contract.parse(&self.prompt.tokenizer.decode(&output), language.as_deref())?;
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
        let max_tokens = self.prompt.contract.max_tokens;
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
                if self.prompt.contract.stop.contains(&tokens[slot]) {
                    results[request_index] = Some(self.prompt.contract.parse(
                        &self.prompt.tokenizer.decode(&output[slot]),
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
) -> Result<(Box<dyn AudioLmExecution>, &'static str)> {
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

    /// A context longer than the window is cut to its most recent words, not refused; one that
    /// fits is kept whole. Needs a Qwen3-ASR bundle (the L4 deploy's); skipped without one.
    #[test]
    fn context_is_fitted_to_the_window_keeping_its_latest_words() {
        let dir = std::path::Path::new("/opt/plow-asr/models/qwen3-asr");
        let Ok(prompt) = AudioLmPrompt::load(&dir.join("model.pkt"), &dir.join("checkpoint")) else {
            eprintln!("no Qwen3-ASR bundle at {} — skipping", dir.display());
            return;
        };
        let rows = 390; // 30 s of audio
        let short = "Thank you for calling Gorospe Law Group.";
        let (ids, _) = prompt.prompt(rows, None, short, 2048).unwrap();
        let (bare, _) = prompt.prompt(rows, None, "", 2048).unwrap();
        assert!(ids.len() > bare.len(), "a short context is kept whole");
        let long: String = (0..3000).map(|i| format!("word{i} ")).collect();
        let (ids, _) = prompt.prompt(rows, None, &long, 2048).unwrap();
        assert!(ids.len() + AudioLmPrompt::output_reserve(rows) <= 2048);
        let kept = prompt.fit_context(rows, None, &long, 2048).unwrap();
        assert!(kept.starts_with("word") && kept.trim_end().ends_with("word2999"), "the latest words stay");
        assert!(kept.split_whitespace().count() > 100, "far more than the old 256-token cap allowed");
        // A window the audio alone fills leaves no context, which is still a valid prompt.
        assert_eq!(prompt.fit_context(rows, None, &long, bare.len() + 10).unwrap(), "");
    }

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

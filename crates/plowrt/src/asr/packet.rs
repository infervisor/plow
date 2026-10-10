//! `rnnt.greedy.v1` packets as a [`Transcriber`]: frontend, greedy decode, detokenizer and
//! language all from the packet (ASR contract, `plow_asset::speech_contract`).

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use plow_asset::speech_contract::{self, TokenOutput};

use crate::asr::frontend::PacketLogMelFrontend;
use crate::asr::rnnt::{detokenize, PacketRnnt, RnntStream};
use crate::asr::{Transcriber, Transcript};
use crate::{Result, RuntimeError};

pub struct PacketRnntTranscriber {
    frontend: PacketLogMelFrontend,
    execution: PacketRnnt,
    output: TokenOutput,
    streams: std::collections::HashMap<u64, TranscriberStream>,
    next_stream: u64,
}

/// One open encoder stream: the recording so far, the steps taken and the tokens decoded.
struct TranscriberStream {
    rnnt: RnntStream,
    samples: Vec<f32>,
    steps: usize,
    tokens: Vec<u32>,
}

impl PacketRnntTranscriber {
    pub fn load(packet: &Path, backend: &str) -> Result<Self> {
        let execution = PacketRnnt::load(packet, backend)?;
        speech_contract::check_contract(
            execution.parameter(speech_contract::CONTRACT).ok(),
            speech_contract::ASR_CONTRACT,
            "rnnt.greedy.v1",
        )
        .map_err(rejected)?;
        let output = execution.output().cloned().ok_or_else(|| rejected("packet declares no output contract"))?;
        let frontend = execution.log_mel_frontend()?;
        Ok(Self { frontend, execution, output, streams: Default::default(), next_stream: 0 })
    }

    pub fn backend(&self) -> &'static str {
        self.execution.backend()
    }
}

impl Transcriber for PacketRnntTranscriber {
    fn stream_open(&mut self) -> Result<Option<u64>> {
        if self.execution.stream_windows().is_none() {
            return Ok(None);
        }
        let rnnt = self.execution.stream_open()?;
        let id = self.next_stream;
        self.next_stream += 1;
        self.streams.insert(id, TranscriberStream { rnnt, samples: Vec::new(), steps: 0, tokens: Vec::new() });
        Ok(Some(id))
    }

    /// Runs one encoder step per whole window of final mel frames the new audio completes.
    fn stream_push(&mut self, id: u64, samples: &[f32], on_text: &mut dyn FnMut(&str)) -> Result<String> {
        let windows = self.execution.stream_windows().ok_or_else(|| rejected("packet has no encoder stream"))?;
        let stream = self.streams.get_mut(&id).ok_or_else(|| rejected("unknown ASR stream"))?;
        stream.samples.extend_from_slice(samples);
        let new_per_step = windows.step - windows.history;
        let end = |steps: usize| windows.first + steps * new_per_step;
        // Frames from the next window's start: earlier ones were consumed.
        let from = if stream.steps == 0 { 0 } else { end(stream.steps) - windows.step };
        let mut mel = None;
        loop {
            let (features, complete) = match &mel {
                Some(computed) => computed,
                None => mel.insert(self.frontend.extract_partial(&stream.samples, from)?),
            };
            let stop = end(stream.steps);
            if *complete < stop {
                break;
            }
            let start = if stream.steps == 0 { 0 } else { stop - windows.step };
            let window = features
                .values
                .get((start - from) * windows.bins..(stop - from) * windows.bins)
                .ok_or_else(|| rejected("stream window is outside the computed mel frames"))?;
            let (output, before) = (&self.output, &stream.tokens);
            let tokens = self.execution.stream_step(&mut stream.rnnt, window, &mut |emitted| {
                let all: Vec<u32> = before.iter().chain(emitted).copied().collect();
                on_text(&detokenize(output, &all));
            })?;
            stream.tokens.extend(tokens);
            stream.steps += 1;
        }
        Ok(detokenize(&self.output, &stream.tokens))
    }

    fn stream_close(&mut self, id: u64) {
        if let Some(stream) = self.streams.remove(&id) {
            self.execution.stream_close(stream.rnnt);
        }
    }

    /// A packet declaring a `language` answers in it only (or a declared alias of it); one
    /// without detects the language itself, so a requested language is accepted but not applied.
    fn language(&self, requested: Option<&str>) -> Result<Option<String>> {
        let Some(compiled) = &self.output.language else {
            return Ok(None);
        };
        let Some(requested) = requested else {
            return Ok(Some(compiled.clone()));
        };
        let compatible = requested.eq_ignore_ascii_case(compiled)
            || self.output.language_aliases.iter().any(|(alias, l)| alias.eq_ignore_ascii_case(requested) && l == compiled);
        compatible
            .then(|| Some(compiled.clone()))
            .ok_or_else(|| rejected(format!("packet was compiled for {compiled}, not {requested}")))
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &AtomicBool,
    ) -> Result<Transcript> {
        self.transcribe_streaming(samples, language, context, cancel, &mut |_| {})
    }

    /// Text grows as the greedy RNNT loop emits tokens (after the whole encoder pass).
    fn transcribe_streaming(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Transcript> {
        if !context.is_empty() {
            return Err(rejected("speech context is not implemented"));
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(rejected("ASR cancelled"));
        }
        let language = self.language(language)?;
        let features = self.frontend.extract(samples)?;
        let valid_frames = features.frames;
        let expected = self.execution.input_elements();
        if features.values.len() > expected {
            return Err(rejected(format!(
                "audio produces {} feature values; packet capacity is {expected}",
                features.values.len()
            )));
        }
        let mut input = features.values;
        input.resize(expected, 0.0);
        let output = &self.output;
        let tokens = self.execution.transcribe_input_frames_with(&input, valid_frames, &mut |emitted| {
            on_text(&detokenize(output, emitted))
        })?;
        if cancel.load(Ordering::Relaxed) {
            return Err(rejected("ASR cancelled"));
        }
        Ok(Transcript {
            text: detokenize(&self.output, &tokens),
            language,
        })
    }
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("invalid packet RNNT: {}", message.into()))
}

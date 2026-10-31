//! `rnnt.greedy.v1` packets as a [`Transcriber`]: frontend, greedy decode, detokenizer and
//! language all from the packet (ASR contract, `plow_asset::speech_contract`).

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use plow_asset::speech_contract::{self, TokenOutput};

use crate::asr::frontend::PacketLogMelFrontend;
use crate::asr::rnnt::{detokenize, PacketRnnt, RnntStream, StreamWindows};
use crate::asr::{StreamPush, Transcriber, Transcript};
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
        let output = execution
            .output()
            .cloned()
            .ok_or_else(|| rejected("packet declares no output contract"))?;
        let frontend = execution.log_mel_frontend()?;
        Ok(Self {
            frontend,
            execution,
            output,
            streams: Default::default(),
            next_stream: 0,
        })
    }

    pub fn backend(&self) -> &'static str {
        self.execution.backend()
    }

    /// The mel windows `stream`'s audio completes past its steps, each with the encoder rows to
    /// decode. `finish`: the recording ended, so the windows run to the last row its offline
    /// transcription decodes, cut from its offline mel frames (zeros past them); `None` when the
    /// stream's steps cannot reproduce that transcription.
    fn pending_windows(
        &self,
        stream: &TranscriberStream,
        finish: bool,
    ) -> Result<Option<Vec<(Vec<f32>, usize)>>> {
        let w = self
            .execution
            .stream_windows()
            .ok_or_else(|| rejected("packet has no encoder stream"))?;
        if finish {
            let features = self.frontend.extract(&stream.samples)?;
            return Ok(self
                .execution
                .final_rows(features.frames)?
                .map(|rows| final_windows(&features.values, w, stream.steps, rows)));
        }
        let end = |steps: usize| w.first + steps * (w.step - w.history);
        let start = |steps: usize| if steps == 0 { 0 } else { end(steps) - w.step };
        let mut windows = Vec::new();
        let mut steps = stream.steps;
        let from = start(steps);
        let (features, complete) = self.frontend.extract_partial(&stream.samples, from)?;
        while complete >= end(steps) {
            let window = features
                .values
                .get((start(steps) - from) * w.bins..(end(steps) - from) * w.bins)
                .ok_or_else(|| rejected("stream window is outside the computed mel frames"))?;
            windows.push((window.to_vec(), w.rows));
            steps += 1;
        }
        Ok(Some(windows))
    }
}

/// A finished stream's windows past its first `steps`, up to encoder row `rows`: cut from the
/// recording's mel frames `values`, zeros past them.
fn final_windows(
    values: &[f32],
    w: StreamWindows,
    mut steps: usize,
    rows: usize,
) -> Vec<(Vec<f32>, usize)> {
    let end = |steps: usize| w.first + steps * (w.step - w.history);
    let start = |steps: usize| if steps == 0 { 0 } else { end(steps) - w.step };
    let mut windows = Vec::new();
    while steps * w.rows < rows {
        let (from, to) = (start(steps) * w.bins, end(steps) * w.bins);
        let mut window = vec![0.0; to - from];
        if let Some(have) = values.get(from..to.min(values.len())) {
            window[..have.len()].copy_from_slice(have);
        }
        windows.push((window, w.rows.min(rows - steps * w.rows)));
        steps += 1;
    }
    windows
}

impl Transcriber for PacketRnntTranscriber {
    fn stream_open(&mut self) -> Result<Option<u64>> {
        if self.execution.stream_windows().is_none() {
            return Ok(None);
        }
        let rnnt = self.execution.stream_open()?;
        let id = self.next_stream;
        self.next_stream += 1;
        self.streams.insert(
            id,
            TranscriberStream {
                rnnt,
                samples: Vec::new(),
                steps: 0,
                tokens: Vec::new(),
            },
        );
        Ok(Some(id))
    }

    /// Runs one encoder step per whole window of final mel frames the new audio completes.
    fn stream_push(
        &mut self,
        id: u64,
        samples: &[f32],
        on_text: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let stream = self
            .streams
            .get_mut(&id)
            .ok_or_else(|| rejected("unknown ASR stream"))?;
        stream.samples.extend_from_slice(samples);
        let windows = self
            .pending_windows(&self.streams[&id], false)?
            .unwrap_or_default();
        let stream = self.streams.get_mut(&id).expect("present");
        for (window, _) in windows {
            let (output, before) = (&self.output, &stream.tokens);
            let tokens = self
                .execution
                .stream_step(&mut stream.rnnt, &window, &mut |emitted| {
                    let all: Vec<u32> = before.iter().chain(emitted).copied().collect();
                    on_text(&detokenize(output, &all));
                })?;
            stream.tokens.extend(tokens);
            stream.steps += 1;
        }
        Ok(detokenize(&self.output, &stream.tokens))
    }

    fn stream_finals(&self) -> bool {
        self.execution.stream_slots().is_some()
    }

    fn stream_slots(&self) -> Option<usize> {
        self.execution.stream_slots()
    }

    /// Steps the streams in rounds, a round taking the next window of every stream with one left
    /// (up to `PacketRnnt::stream_batch` per launch).
    fn stream_push_batch(&mut self, pushes: &[StreamPush<'_>]) -> Vec<Result<String>> {
        if self.execution.stream_slots().is_none() {
            // `stream.*` packets: one stream per step, and no finals.
            return pushes
                .iter()
                .map(|push| {
                    if push.finish {
                        self.stream_close(push.id);
                        return Err(rejected("packet answers no finals from its stream"));
                    }
                    self.stream_push(push.id, push.samples, &mut |_| {})
                })
                .collect();
        }
        let mut results: Vec<Option<Result<String>>> = pushes.iter().map(|_| None).collect();
        // (push, stream, its windows): out of the map while they step.
        let mut active = Vec::with_capacity(pushes.len());
        for (index, push) in pushes.iter().enumerate() {
            let Some(mut stream) = self.streams.remove(&push.id) else {
                results[index] = Some(Err(rejected("unknown ASR stream")));
                continue;
            };
            stream.samples.extend_from_slice(push.samples);
            match self.pending_windows(&stream, push.finish) {
                Ok(Some(windows)) => active.push((index, stream, windows)),
                Ok(None) => {
                    self.execution.stream_close(stream.rnnt);
                    let cancel = AtomicBool::new(false);
                    results[index] = Some(
                        self.transcribe(&stream.samples, None, "", &cancel)
                            .map(|transcript| transcript.text),
                    );
                }
                Err(error) => {
                    results[index] = Some(Err(error));
                    if push.finish {
                        self.execution.stream_close(stream.rnnt);
                    } else {
                        self.streams.insert(push.id, stream);
                    }
                }
            }
        }
        let batch = self.execution.stream_batch();
        for round in 0.. {
            let members: Vec<usize> = (0..active.len())
                .filter(|&a| results[active[a].0].is_none() && active[a].2.len() > round)
                .collect();
            if members.is_empty() {
                break;
            }
            for launch in members.chunks(batch) {
                let mut steps = Vec::with_capacity(launch.len());
                let mut next = launch.iter().copied().peekable();
                for (a, (_, stream, windows)) in active.iter_mut().enumerate() {
                    if next.next_if_eq(&a).is_some() {
                        let (window, rows) = &windows[round];
                        steps.push((&mut stream.rnnt, window.as_slice(), *rows));
                    }
                }
                let stepped = self.execution.stream_step_batch(&mut steps);
                drop(steps);
                match stepped {
                    Ok(tokens) => {
                        for (&a, tokens) in launch.iter().zip(tokens) {
                            let stream = &mut active[a].1;
                            stream.tokens.extend(tokens);
                            stream.steps += 1;
                        }
                    }
                    Err(error) => {
                        let message = error.to_string();
                        let mut error = Some(error);
                        for &a in launch {
                            let error = error
                                .take()
                                .unwrap_or_else(|| RuntimeError::Msg(message.clone()));
                            results[active[a].0] = Some(Err(error));
                        }
                    }
                }
            }
        }
        for (index, stream, _) in active {
            let result = results[index]
                .take()
                .unwrap_or_else(|| Ok(detokenize(&self.output, &stream.tokens)));
            // A failed launch may have advanced the stream's caches and predictor slot: retire it.
            let failed = result.is_err();
            results[index] = Some(result);
            if pushes[index].finish || failed {
                self.execution.stream_close(stream.rnnt);
            } else {
                self.streams.insert(pushes[index].id, stream);
            }
        }
        results
            .into_iter()
            .map(|result| result.expect("every push is answered"))
            .collect()
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
            || self
                .output
                .language_aliases
                .iter()
                .any(|(alias, l)| alias.eq_ignore_ascii_case(requested) && l == compiled);
        compatible.then(|| Some(compiled.clone())).ok_or_else(|| {
            rejected(format!(
                "packet was compiled for {compiled}, not {requested}"
            ))
        })
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
        let tokens =
            self.execution
                .transcribe_input_frames_with(&input, valid_frames, &mut |emitted| {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Nemotron's stream: 4 rows from 32 new mel frames, 16 repeated.
    const W: StreamWindows = StreamWindows {
        first: 32,
        step: 48,
        history: 16,
        bins: 2,
        rows: 4,
    };

    #[test]
    fn final_windows_past_the_mel_frames_are_zeros() {
        let values: Vec<f32> = (1..=20).map(|v| v as f32).collect();
        let windows = final_windows(&values, W, 0, 10);
        let rows: Vec<_> = windows.iter().map(|w| w.1).collect();
        assert_eq!(rows, [4, 4, 2]);
        assert_eq!(windows[0].0[..20], values[..]);
        assert!(windows[0].0[20..].iter().all(|&v| v == 0.0));
        // The later windows start past the 10 frames.
        for (window, _) in &windows[1..] {
            assert_eq!(window.len(), W.step * W.bins);
            assert!(window.iter().all(|&v| v == 0.0));
        }
        assert_eq!(final_windows(&values, W, 3, 12), []);
    }
}

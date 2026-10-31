//! `rnnt.greedy.v1` packets as a [`Transcriber`]: frontend, greedy decode, detokenizer and
//! language all from the packet (ASR contract, `plow_asset::speech_contract`).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use plow_asset::speech_contract::{self, TokenOutput};

use crate::asr::frontend::{LogMelFeatures, PacketLogMelFrontend};
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

/// One open encoder stream: the recording so far, the steps taken, the tokens decoded and the
/// windows from `steps` on that a push cut for its later rounds (`cut_finish`: through the final).
struct TranscriberStream {
    rnnt: RnntStream,
    samples: Vec<f32>,
    steps: usize,
    tokens: Vec<u32>,
    cut: VecDeque<(Vec<f32>, usize)>,
    cut_finish: bool,
}

impl TranscriberStream {
    /// Whether `push` steps on through the windows already cut: it brings no audio and finishes
    /// exactly when they run through the final (a superseded partial's stop short of it).
    fn continues(&self, push: &StreamPush<'_>) -> bool {
        push.samples.is_empty() && !self.cut.is_empty() && self.cut_finish == push.finish
    }
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

    /// The mel windows `stream`'s audio completes past its steps (see [`Self::windows`]).
    fn pending_windows(
        &self,
        stream: &TranscriberStream,
        finish: bool,
    ) -> Result<Vec<(Vec<f32>, usize)>> {
        let w = self
            .execution
            .stream_windows()
            .ok_or_else(|| rejected("packet has no encoder stream"))?;
        let frames = self
            .frontend
            .extract_partial(&stream.samples, window_start(&w, stream.steps))?;
        self.windows(&w, stream.steps, frames, finish)
    }

    /// The windows past `steps`, each with the encoder rows to decode, cut from `frames`: the mel
    /// frames from window `steps` on and how many leading frames (counted from 0) are final.
    /// `finish`: the recording ended, so the windows run to the last row its offline transcription
    /// decodes, zeros past its mel frames (partial streaming stepped complete windows, so they
    /// start inside the recording; they equal the whole recording's).
    fn windows(
        &self,
        w: &StreamWindows,
        mut steps: usize,
        (features, complete): (LogMelFeatures, usize),
        finish: bool,
    ) -> Result<Vec<(Vec<f32>, usize)>> {
        let first = window_start(w, steps);
        let mut windows = Vec::new();
        if finish {
            let rows = self.execution.final_rows(first + features.frames)?;
            while steps * w.rows < rows {
                let (from, to) = (
                    (window_start(w, steps) - first) * w.bins,
                    (window_end(w, steps) - first) * w.bins,
                );
                let mut window = vec![0.0; to - from];
                let len = features.values.len();
                let values = &features.values[from.min(len)..to.min(len)];
                window[..values.len()].copy_from_slice(values);
                windows.push((window, w.rows.min(rows - steps * w.rows)));
                steps += 1;
            }
            return Ok(windows);
        }
        while complete >= window_end(w, steps) {
            let window = features
                .values
                .get(
                    (window_start(w, steps) - first) * w.bins
                        ..(window_end(w, steps) - first) * w.bins,
                )
                .ok_or_else(|| rejected("stream window is outside the computed mel frames"))?;
            windows.push((window.to_vec(), w.rows));
            steps += 1;
        }
        Ok(windows)
    }

    /// Each stream's mel frames from its next window on (`extract_partial`), the streams spread
    /// over the rayon pool.
    fn next_frames<'a>(
        &self,
        w: &StreamWindows,
        streams: impl Iterator<Item = &'a TranscriberStream>,
    ) -> Vec<Result<(LogMelFeatures, usize)>> {
        let inputs: Vec<(&[f32], usize)> = streams
            .map(|stream| (stream.samples.as_slice(), window_start(w, stream.steps)))
            .collect();
        let frontend = &self.frontend;
        let extract = |&(samples, from): &(&[f32], usize)| frontend.extract_partial(samples, from);
        #[cfg(feature = "hf-tokenizer")]
        {
            use rayon::prelude::*;
            inputs.par_iter().map(extract).collect()
        }
        #[cfg(not(feature = "hf-tokenizer"))]
        {
            inputs.iter().map(extract).collect()
        }
    }
}

/// The mel frames `window_start(steps)..window_end(steps)` a stream's window `steps` reads.
fn window_end(w: &StreamWindows, steps: usize) -> usize {
    w.first + steps * (w.step - w.history)
}

fn window_start(w: &StreamWindows, steps: usize) -> usize {
    if steps == 0 {
        0
    } else {
        window_end(w, steps) - w.step
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
        self.streams.insert(
            id,
            TranscriberStream {
                rnnt,
                samples: Vec::new(),
                steps: 0,
                tokens: Vec::new(),
                cut: VecDeque::new(),
                cut_finish: false,
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
        let windows = self.pending_windows(&self.streams[&id], false)?;
        let stream = self.streams.get_mut(&id).expect("present");
        stream.cut.clear();
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

    /// One window of every pushed stream with one left, up to one launch of
    /// `PacketRnnt::stream_batch` streams. A push cuts its windows the round it joins; its later
    /// rounds step them.
    fn stream_push_round(&mut self, pushes: &[StreamPush<'_>]) -> Vec<Result<Option<String>>> {
        let (Some(w), Some(_)) = (self.execution.stream_windows(), self.stream_slots()) else {
            return crate::asr::push_alone(self, pushes);
        };
        let started = Instant::now();
        let mut results: Vec<Option<Result<Option<String>>>> =
            pushes.iter().map(|_| None).collect();
        // (push, stream): out of the map while they step.
        let mut active = Vec::with_capacity(pushes.len());
        for (index, push) in pushes.iter().enumerate() {
            match self.streams.remove(&push.id) {
                Some(mut stream) => {
                    stream.samples.extend_from_slice(push.samples);
                    active.push((index, stream));
                }
                None => results[index] = Some(Err(rejected("unknown ASR stream"))),
            }
        }
        let recut: Vec<bool> = active
            .iter()
            .map(|(index, stream)| !stream.continues(&pushes[*index]))
            .collect();
        let cutting = active.iter().zip(&recut).filter(|(_, recut)| **recut);
        let mut frames = self
            .next_frames(&w, cutting.map(|((_, stream), _)| stream))
            .into_iter();
        // (push, stream, its next window and rows, whether it is the last)
        let mut stepping = Vec::with_capacity(active.len());
        let mut done = Vec::new();
        for ((index, mut stream), recut) in active.into_iter().zip(recut) {
            if recut {
                let finish = pushes[index].finish;
                let windows = frames
                    .next()
                    .expect("frames per cut")
                    .and_then(|frames| self.windows(&w, stream.steps, frames, finish));
                match windows {
                    Ok(windows) => (stream.cut, stream.cut_finish) = (windows.into(), finish),
                    Err(error) => {
                        stream.cut.clear();
                        results[index] = Some(Err(error));
                        done.push((index, stream));
                        continue;
                    }
                }
            }
            match stream.cut.pop_front() {
                Some(next) => {
                    let last = stream.cut.is_empty();
                    stepping.push((index, stream, next, last));
                }
                None => done.push((index, stream)),
            }
        }
        let mel_us = started.elapsed().as_micros() as u64;
        // One launch a round, finals first: a second launch would hold every final in the round
        // for both. The rest step next round, in push order.
        stepping.sort_by_key(|(index, ..)| !pushes[*index].finish);
        let deferred = stepping.split_off(stepping.len().min(self.execution.stream_batch()));
        let deferred_count = deferred.len();
        for (index, mut stream, window, _) in deferred {
            stream.cut.push_front(window);
            self.streams.insert(pushes[index].id, stream);
            results[index] = Some(Ok(None));
        }
        let launches = usize::from(!stepping.is_empty());
        if launches > 0 {
            let launch = &mut stepping[..];
            let mut steps: Vec<_> = launch
                .iter_mut()
                .map(|(_, stream, (window, rows), _)| (&mut stream.rnnt, window.as_slice(), *rows))
                .collect();
            let stepped = self.execution.stream_step_batch(&mut steps);
            drop(steps);
            match stepped {
                Ok(tokens) => {
                    for ((_, stream, _, _), tokens) in launch.iter_mut().zip(tokens) {
                        stream.tokens.extend(tokens);
                        stream.steps += 1;
                    }
                }
                Err(error) => {
                    let message = error.to_string();
                    let mut error = Some(error);
                    for (index, stream, ..) in launch.iter_mut() {
                        stream.cut.clear();
                        let error = error
                            .take()
                            .unwrap_or_else(|| RuntimeError::Msg(message.clone()));
                        results[*index] = Some(Err(error));
                    }
                }
            }
        }
        tracing::debug!(
            pushes = pushes.len(),
            finishes = pushes.iter().filter(|push| push.finish).count(),
            stepped = stepping.len(),
            deferred = deferred_count,
            launches,
            mel_us,
            total_us = started.elapsed().as_micros() as u64,
            "rnnt stream push round"
        );
        let stepped = stepping
            .into_iter()
            .map(|(index, stream, _, last)| (index, stream, last));
        for (index, stream, last) in stepped.chain(done.into_iter().map(|(i, s)| (i, s, true))) {
            let result = results[index]
                .take()
                .unwrap_or_else(|| Ok(last.then(|| detokenize(&self.output, &stream.tokens))));
            if pushes[index].finish && (last || result.is_err()) {
                self.execution.stream_close(stream.rnnt);
            } else {
                self.streams.insert(pushes[index].id, stream);
            }
            results[index] = Some(result);
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

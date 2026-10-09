//! Endpointing of a 16 kHz stream into utterance segments, for continuous WebSocket and Realtime
//! sessions. Frames are classified by energy ([`Endpointer::new`], [`Endpointer::push`]) or by a
//! VAD's speech probabilities ([`Endpointer::with_vad`], [`Endpointer::push_scored`]). Only the
//! open segment (or a short lead-in) stays buffered.

use std::collections::VecDeque;

/// 20 ms energy analysis frame.
pub const FRAME: usize = 320;
/// Speech must last this many energy frames (100 ms) to open a segment.
const ONSET_FRAMES: usize = 5;
/// Speech must last this long (ms, rounded up to whole frames) to open a VAD segment.
const VAD_ONSET_MS: usize = 64;
/// Context kept before the onset and after the last speech frame (200 ms, energy).
const CONTEXT: usize = 3_200;
/// Samples searched (backwards, 3 s) for the quietest cut point of an overlong segment.
const CUT_SAMPLES: usize = 48_000;
/// The noise floor is the quietest frame over this many frames (3 s): stationary noise sets it,
/// speech pauses are too short to.
const FLOOR_FRAMES: usize = 150;
/// Speech is this far above the floor (12 dB) and above an absolute level (-54 dBFS).
const MARGIN: f32 = 4.0;
const MIN_LEVEL: f32 = 0.002;
/// A VAD frame below `threshold - VAD_HYSTERESIS` is silence (Silero's `neg_threshold`).
const VAD_HYSTERESIS: f32 = 0.15;

#[derive(Clone, Copy, Debug)]
pub struct EndpointConfig {
    pub min_silence_ms: u32,
    pub max_segment_ms: u32,
}

/// Speech-probability endpointing (OpenAI Realtime `server_vad`): a frame is speech at
/// `threshold` or above and silence below `threshold - 0.15` (in between, the current state
/// holds); a turn opens after 64 ms of speech with `prefix_padding_ms` of audio before it, and
/// ends `min_silence_ms` after the speech, that silence included.
#[derive(Clone, Copy, Debug)]
pub struct VadConfig {
    /// 16 kHz samples per scored frame.
    pub frame: usize,
    pub threshold: f32,
    pub prefix_padding_ms: u32,
}

#[derive(Debug)]
pub struct Segment {
    pub index: u64,
    /// Absolute 16 kHz sample offsets of the segment in the stream.
    pub start: u64,
    pub end: u64,
    pub samples: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Speech,
    /// Between the VAD thresholds: continues what the stream is doing.
    Hold,
    Silence,
}

enum Detector {
    Energy { levels: VecDeque<f32> },
    Vad { on: f32, off: f32 },
}

struct Open {
    index: u64,
    start: u64,
    last_speech_end: u64,
    silence: usize,
    /// (frame start, level) of the last `cut_frames` frames.
    recent: VecDeque<(u64, f32)>,
}

pub struct Endpointer {
    frame: usize,
    onset: usize,
    /// Samples kept before the onset.
    before: usize,
    min_silence: usize,
    max_segment: u64,
    cut_frames: usize,
    detector: Detector,
    buf: Vec<f32>,
    /// Stream offset of `buf[0]`.
    base: u64,
    /// Stream offset of the first sample not yet analysed.
    scanned: u64,
    run: usize,
    open: Option<Open>,
    next_index: u64,
}

impl Endpointer {
    /// Energy endpointing (20 ms frames, noise-floor relative); feed with [`Self::push`].
    pub fn new(config: EndpointConfig) -> Self {
        Self::build(config, FRAME, ONSET_FRAMES, CONTEXT, Detector::Energy { levels: VecDeque::with_capacity(FLOOR_FRAMES) })
    }

    /// VAD endpointing; feed with [`Self::push_scored`].
    pub fn with_vad(config: EndpointConfig, vad: VadConfig) -> Self {
        let off = (vad.threshold - VAD_HYSTERESIS).max(0.01).min(vad.threshold);
        let onset = (VAD_ONSET_MS * 16).div_ceil(vad.frame).max(1);
        Self::build(config, vad.frame, onset, vad.prefix_padding_ms as usize * 16, Detector::Vad { on: vad.threshold, off })
    }

    fn build(config: EndpointConfig, frame: usize, onset: usize, before: usize, detector: Detector) -> Self {
        Self {
            frame,
            onset,
            before,
            min_silence: (config.min_silence_ms as usize * 16).div_ceil(frame),
            max_segment: u64::from(config.max_segment_ms) * 16,
            cut_frames: CUT_SAMPLES / frame,
            detector,
            buf: Vec::new(),
            base: 0,
            scanned: 0,
            run: 0,
            open: None,
            next_index: 0,
        }
    }

    /// Feed 16 kHz samples (energy endpointing); returns the segments they close, in order.
    pub fn push(&mut self, samples: &[f32]) -> Vec<Segment> {
        debug_assert!(matches!(self.detector, Detector::Energy { .. }));
        self.buf.extend_from_slice(samples);
        let mut done = Vec::new();
        while self.scanned + self.frame as u64 <= self.end() {
            let at = (self.scanned - self.base) as usize;
            let frame = &self.buf[at..at + self.frame];
            let rms = (frame.iter().map(|x| x * x).sum::<f32>() / self.frame as f32).sqrt();
            let Detector::Energy { levels } = &mut self.detector else { unreachable!() };
            if levels.len() == FLOOR_FRAMES {
                levels.pop_front();
            }
            levels.push_back(rms);
            let floor = levels.iter().copied().fold(f32::INFINITY, f32::min);
            let class = if rms > (floor * MARGIN).max(MIN_LEVEL) { Class::Speech } else { Class::Silence };
            done.extend(self.frame_at(self.scanned, rms, class));
            self.scanned += self.frame as u64;
        }
        self.trim();
        done
    }

    /// Feed whole frames with their speech probabilities (VAD endpointing):
    /// `samples.len() == probs.len() * frame`. Returns the segments they close, in order.
    pub fn push_scored(&mut self, samples: &[f32], probs: &[f32]) -> Vec<Segment> {
        let Detector::Vad { on, off } = self.detector else { panic!("push_scored on an energy endpointer") };
        assert_eq!(samples.len(), probs.len() * self.frame, "push_scored takes whole frames");
        debug_assert_eq!(self.scanned, self.end());
        self.buf.extend_from_slice(samples);
        let mut done = Vec::new();
        for &p in probs {
            let class = if p >= on {
                Class::Speech
            } else if p < off {
                Class::Silence
            } else {
                Class::Hold
            };
            done.extend(self.frame_at(self.scanned, p, class));
            self.scanned += self.frame as u64;
        }
        self.trim();
        done
    }

    /// Append samples that will not be scored (the partial frame at the end of a turn): they
    /// belong to the open segment, but classify nothing.
    pub fn skip(&mut self, samples: &[f32]) {
        self.buf.extend_from_slice(samples);
        self.scanned = self.end();
    }

    /// End of stream: the open segment, through the last sample.
    pub fn finish(&mut self) -> Option<Segment> {
        let open = self.open.take()?;
        let end = self.end();
        Some(self.cut(open.index, open.start, end))
    }

    /// The open segment's index and audio so far.
    pub fn open_audio(&self) -> Option<(u64, &[f32])> {
        self.open.as_ref().map(|o| (o.index, &self.buf[(o.start - self.base) as usize..]))
    }

    fn end(&self) -> u64 {
        self.base + self.buf.len() as u64
    }

    fn trim(&mut self) {
        let keep = match &self.open {
            Some(open) => open.start,
            None => self.scanned.saturating_sub((self.onset * self.frame + self.before + self.frame) as u64),
        };
        if keep > self.base {
            self.buf.drain(..(keep - self.base) as usize);
            self.base = keep;
        }
    }

    fn frame_at(&mut self, at: u64, level: f32, class: Class) -> Option<Segment> {
        let frame_end = at + self.frame as u64;
        let Some(open) = &mut self.open else {
            self.run = if class == Class::Speech { self.run + 1 } else { 0 };
            if self.run >= self.onset {
                let onset = frame_end - (self.run * self.frame) as u64;
                let start = onset.saturating_sub(self.before as u64).max(self.base);
                let mut recent = VecDeque::with_capacity(self.cut_frames);
                recent.push_back((at, level));
                self.open = Some(Open { index: self.next_index, start, last_speech_end: frame_end, silence: 0, recent });
                self.next_index += 1;
                self.run = 0;
            }
            return None;
        };
        if open.recent.len() == self.cut_frames {
            open.recent.pop_front();
        }
        open.recent.push_back((at, level));
        match class {
            Class::Speech => {
                open.silence = 0;
                open.last_speech_end = frame_end;
            }
            Class::Hold if open.silence == 0 => open.last_speech_end = frame_end,
            Class::Hold | Class::Silence => open.silence += 1,
        }
        if open.silence >= self.min_silence {
            let (index, start) = (open.index, open.start);
            let end = match self.detector {
                Detector::Energy { .. } => (open.last_speech_end + CONTEXT as u64).min(frame_end),
                Detector::Vad { .. } => frame_end,
            };
            self.open = None;
            return Some(self.cut(index, start, end));
        }
        if frame_end - open.start >= self.max_segment {
            // The quietest recent frame ends this segment and starts the next.
            let (quiet, _) = open
                .recent
                .iter()
                .copied()
                .filter(|&(t, _)| t > open.start)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((at, level));
            let cut = quiet + self.frame as u64 / 2;
            let (index, start) = (open.index, open.start);
            open.index = self.next_index;
            open.start = cut;
            open.recent.retain(|&(t, _)| t >= cut);
            self.next_index += 1;
            return Some(self.cut(index, start, cut));
        }
        None
    }

    fn cut(&self, index: u64, start: u64, end: u64) -> Segment {
        let samples = self.buf[(start - self.base) as usize..(end - self.base) as usize].to_vec();
        Segment { index, start, end, samples }
    }
}

/// A stream's turn detection: the endpointer, scored by the process's VAD packet when one is
/// loaded (`--vad-packet`), else by energy. Counts are 16 kHz samples handed to the endpointer.
pub struct Turns {
    endpointer: Endpointer,
    vad: Option<crate::asr::vad::VadStream>,
}

impl Turns {
    /// `threshold` and `prefix_padding_ms` apply with a VAD; energy endpointing has its own.
    pub fn new(config: EndpointConfig, threshold: f32, prefix_padding_ms: u32, vad: Option<&std::sync::Arc<crate::asr::vad::Vad>>) -> Self {
        match vad {
            Some(vad) => Self {
                endpointer: Endpointer::with_vad(config, VadConfig { frame: vad.frame_samples, threshold, prefix_padding_ms }),
                vad: Some(vad.stream()),
            },
            None => Self { endpointer: Endpointer::new(config), vad: None },
        }
    }

    /// Feed samples; (samples the endpointer took, segments closed).
    pub async fn push(&mut self, samples: &[f32]) -> Result<(usize, Vec<Segment>), String> {
        match &mut self.vad {
            None => Ok((samples.len(), self.endpointer.push(samples))),
            Some(vad) => {
                let (frames, probs) = vad.push(samples).await?;
                Ok((frames.len(), self.endpointer.push_scored(&frames, &probs)))
            }
        }
    }

    /// Feed the last samples of a turn and close it: the segments closed, the open one last.
    pub async fn close(&mut self, tail: &[f32]) -> Result<(usize, Vec<Segment>), String> {
        let (mut taken, mut closed) = self.push(tail).await?;
        let (rest, last) = self.finish();
        taken += rest;
        closed.extend(last);
        Ok((taken, closed))
    }

    /// The open segment through every sample fed (an unscored partial frame included).
    pub fn finish(&mut self) -> (usize, Option<Segment>) {
        let mut skipped = 0;
        if let Some(vad) = &mut self.vad {
            let rest = vad.take_rest();
            self.endpointer.skip(&rest);
            skipped = rest.len();
        }
        (skipped, self.endpointer.finish())
    }

    pub fn open_audio(&self) -> Option<(u64, &[f32])> {
        self.endpointer.open_audio()
    }
}

/// Test signal at sample `i`: a tone under a 4 Hz syllable envelope, whose dips (unlike a steady
/// tone's) keep the noise floor below it.
#[cfg(test)]
pub(crate) fn speechlike(i: usize) -> f32 {
    let t = i as f32 / 16_000.0;
    0.3 * (i as f32 * 0.07).sin() * (0.55 + 0.45 * (std::f32::consts::TAU * 4.0 * t).sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f32) -> Vec<f32> {
        (0..(seconds * 16_000.0) as usize).map(speechlike).collect()
    }

    fn silence(seconds: f32) -> Vec<f32> {
        vec![0.0; (seconds * 16_000.0) as usize]
    }

    fn feed(e: &mut Endpointer, audio: &[f32], chunk: usize) -> Vec<Segment> {
        let mut out: Vec<_> = audio.chunks(chunk).flat_map(|c| e.push(c)).collect();
        out.extend(e.finish());
        out
    }

    const CONFIG: EndpointConfig = EndpointConfig { min_silence_ms: 600, max_segment_ms: 25_000 };

    #[test]
    fn bursts_split_at_silences_with_context() {
        let mut audio = silence(0.5);
        let mut onsets = Vec::new();
        for burst in [3.0, 1.5, 6.0, 0.4, 2.0] {
            onsets.push(audio.len() as u64);
            audio.extend(tone(burst));
            audio.extend(silence(1.0));
        }
        let segments = feed(&mut Endpointer::new(CONFIG), &audio, 1_234);
        assert_eq!(segments.len(), 5);
        for (i, (s, onset)) in segments.iter().zip(&onsets).enumerate() {
            assert_eq!(s.index, i as u64);
            assert_eq!(s.samples.len() as u64, s.end - s.start);
            assert!(s.start + CONTEXT as u64 <= *onset + FRAME as u64 && *onset < s.start + CONTEXT as u64 + 2 * FRAME as u64);
        }
        assert!(segments.windows(2).all(|w| w[0].end <= w[1].start));
    }

    #[test]
    fn blips_and_silence_open_nothing() {
        let mut audio = silence(5.0);
        audio.extend(tone(0.06));
        audio.extend(silence(5.0));
        assert!(feed(&mut Endpointer::new(CONFIG), &audio, 4_000).is_empty());
    }

    #[test]
    fn long_speech_is_cut_within_the_limit_without_losing_audio() {
        let audio = tone(60.0);
        let segments = feed(&mut Endpointer::new(CONFIG), &audio, 16_000);
        assert!(segments.len() >= 3);
        assert!(segments.iter().all(|s| s.end - s.start <= 25 * 16_000));
        assert!(segments.windows(2).all(|w| w[0].end == w[1].start));
        assert_eq!(segments.last().unwrap().end, audio.len() as u64);
    }

    #[test]
    fn chunking_does_not_change_segments() {
        let mut audio = tone(2.0);
        audio.extend(silence(1.0));
        audio.extend(tone(3.0));
        let a: Vec<_> = feed(&mut Endpointer::new(CONFIG), &audio, 160).into_iter().map(|s| (s.start, s.end)).collect();
        let b: Vec<_> = feed(&mut Endpointer::new(CONFIG), &audio, 48_000).into_iter().map(|s| (s.start, s.end)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn stationary_noise_is_not_speech() {
        let noise: Vec<f32> = (0..16_000 * 10).map(|i| 0.01 * ((i * 7919 % 1000) as f32 / 500.0 - 1.0)).collect();
        let mut audio = noise.clone();
        for (i, x) in tone(2.0).into_iter().enumerate() {
            audio[16_000 * 4 + i] += x;
        }
        let segments = feed(&mut Endpointer::new(CONFIG), &audio, 3_200);
        assert_eq!(segments.len(), 1);
    }

    const VAD_FRAME: usize = 512;

    fn vad(threshold: f32, prefix_padding_ms: u32, min_silence_ms: u32) -> Endpointer {
        Endpointer::with_vad(
            EndpointConfig { min_silence_ms, max_segment_ms: 25_000 },
            VadConfig { frame: VAD_FRAME, threshold, prefix_padding_ms },
        )
    }

    /// Feeds `probs` (one per 32 ms frame) `per_push` frames at a time; the samples are the frame
    /// index, so a segment's bounds can be read back from its audio.
    fn scored(e: &mut Endpointer, probs: &[f32], per_push: usize) -> Vec<Segment> {
        let samples: Vec<f32> = (0..probs.len() * VAD_FRAME).map(|i| (i / VAD_FRAME) as f32).collect();
        let mut out = Vec::new();
        for (p, s) in probs.chunks(per_push).zip(samples.chunks(per_push * VAD_FRAME)) {
            out.extend(e.push_scored(s, p));
        }
        out.extend(e.finish());
        out
    }

    /// `(frames, p)` runs.
    fn stream(runs: &[(usize, f32)]) -> Vec<f32> {
        runs.iter().flat_map(|&(n, p)| std::iter::repeat_n(p, n)).collect()
    }

    const F: u64 = VAD_FRAME as u64;

    #[test]
    fn vad_turn_bounds_follow_threshold_prefix_and_silence() {
        // 20 silent frames, 40 speech, 30 silent (960 ms), 25 speech, 10 silent; 500 ms silence
        // = 16 frames, 300 ms prefix = 4800 samples.
        let probs = stream(&[(20, 0.02), (40, 0.9), (30, 0.05), (25, 0.8), (10, 0.1)]);
        let segments = scored(&mut vad(0.5, 300, 500), &probs, 3);
        assert_eq!(segments.len(), 2);
        let first = &segments[0];
        assert_eq!(first.start, 20 * F - 4_800);
        // Ends after 16 silent frames, the silence included (OpenAI `audio_end_ms`).
        assert_eq!(first.end, (60 + 16) * F);
        assert_eq!(first.samples.len() as u64, first.end - first.start);
        assert_eq!(first.samples[(20 * F - first.start) as usize], 20.0);
        // The second turn is still open at the end of the stream: finish() closes it there.
        assert_eq!(segments[1].start, 90 * F - 4_800);
        assert_eq!(segments[1].end, probs.len() as u64 * F);
    }

    #[test]
    fn vad_threshold_selects_what_is_speech() {
        let probs = stream(&[(10, 0.0), (30, 0.6), (30, 0.0)]);
        assert_eq!(scored(&mut vad(0.5, 0, 200), &probs, 1).len(), 1);
        assert!(scored(&mut vad(0.7, 0, 200), &probs, 1).is_empty());
    }

    #[test]
    fn vad_hysteresis_holds_between_thresholds() {
        // 0.4 is under the 0.5 threshold but over 0.35: it neither opens a turn nor starts silence.
        let probs = stream(&[(10, 0.0), (10, 0.9), (40, 0.4), (10, 0.9), (20, 0.0)]);
        let segments = scored(&mut vad(0.5, 0, 300), &probs, 7);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 10 * F);
        assert_eq!(segments[0].end, (70 + 10) * F);
        assert!(scored(&mut vad(0.5, 0, 300), &stream(&[(10, 0.0), (60, 0.4), (10, 0.0)]), 1).is_empty());
        // Once silence has started, held frames count towards it.
        let probs = stream(&[(10, 0.0), (10, 0.9), (5, 0.1), (20, 0.4), (10, 0.9)]);
        let segments = scored(&mut vad(0.5, 0, 300), &probs, 1);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].end, (20 + 10) * F);
    }

    #[test]
    fn vad_blips_shorter_than_the_onset_open_nothing() {
        let probs = stream(&[(10, 0.0), (1, 0.99), (10, 0.0), (1, 0.99), (10, 0.0)]);
        assert!(scored(&mut vad(0.5, 300, 200), &probs, 4).is_empty());
    }

    #[test]
    fn vad_prefix_is_clamped_to_the_stream_start() {
        let probs = stream(&[(2, 0.0), (20, 0.9), (20, 0.0)]);
        let segments = scored(&mut vad(0.5, 1_000, 200), &probs, 5);
        assert_eq!(segments[0].start, 0);
    }

    #[test]
    fn vad_push_granularity_does_not_change_segments() {
        let probs = stream(&[(13, 0.1), (31, 0.7), (17, 0.2), (9, 0.95), (40, 0.3), (5, 0.6), (30, 0.0)]);
        let bounds = |n| scored(&mut vad(0.5, 300, 400), &probs, n).into_iter().map(|s| (s.start, s.end)).collect::<Vec<_>>();
        assert_eq!(bounds(1), bounds(64));
        assert_eq!(bounds(1), bounds(5));
    }

    #[test]
    fn vad_long_speech_is_cut_at_the_least_likely_frame() {
        let mut probs = stream(&[(5, 0.0)]);
        probs.extend((0..1_000).map(|i| if i == 700 { 0.45 } else { 0.9 }));
        let segments = scored(&mut vad(0.5, 0, 500), &probs, 16);
        assert!(segments.len() >= 2);
        assert!(segments.iter().all(|s| s.end - s.start <= 25 * 16_000));
        assert_eq!(segments[0].end, (5 + 700) * F + F / 2);
        assert!(segments.windows(2).all(|w| w[0].end == w[1].start));
    }

    #[test]
    fn vad_skipped_tail_belongs_to_the_open_turn() {
        let mut e = vad(0.5, 0, 500);
        let probs = stream(&[(4, 0.0), (10, 0.9)]);
        let samples = vec![0.0; probs.len() * VAD_FRAME];
        assert!(e.push_scored(&samples, &probs).is_empty());
        e.skip(&[1.0; 100]);
        let last = e.finish().unwrap();
        assert_eq!(last.end, probs.len() as u64 * F + 100);
        // Scoring continues frame-aligned after a skip.
        assert!(e.push_scored(&samples, &probs).is_empty());
        assert_eq!(e.finish().unwrap().start, probs.len() as u64 * F + 100 + 4 * F);
    }
}

//! Endpointing of a 16 kHz stream into utterance segments, for continuous WebSocket sessions:
//! by energy, or by a Silero VAD packet's speech probability. Only the open segment (or a short
//! lead-in) stays buffered.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::asr::vad::{Vad, VadStream};

/// 20 ms energy analysis frame.
pub const FRAME: usize = 320;
/// Speech must last this long (100 ms) to open a segment.
const ONSET: usize = 1_600;
/// Context kept before the onset and after the last speech frame (200 ms).
const CONTEXT: usize = 3_200;
/// Searched (backwards, 3 s) for the quietest cut point of an overlong segment.
const CUT: usize = 48_000;
/// The noise floor is the quietest frame over this many frames (3 s): stationary noise sets it,
/// speech pauses are too short to.
const FLOOR_FRAMES: usize = 150;
/// Speech is this far above the floor (12 dB) and above an absolute level (-54 dBFS).
const MARGIN: f32 = 4.0;
const MIN_LEVEL: f32 = 0.002;

#[derive(Clone, Copy, Debug)]
pub struct EndpointConfig {
    pub min_silence_ms: u32,
    pub max_segment_ms: u32,
}

#[derive(Debug)]
pub struct Segment {
    pub index: u64,
    /// Absolute 16 kHz sample offsets of the segment in the stream.
    pub start: u64,
    pub end: u64,
    /// Samples the detector judged speech (confident Silero frames, or loud energy frames).
    pub speech: u64,
    pub samples: Vec<f32>,
}

struct Open {
    index: u64,
    start: u64,
    last_speech_end: u64,
    /// A loud slice has timed this turn's speech: from then on only loud slices extend it.
    loud_seen: bool,
    speech: u64,
    /// The speech end [`Endpointer::tentative`] last offered this turn at.
    offered: Option<u64>,
    /// (frame start, rms or speech probability) of the last `cut_frames` frames.
    recent: VecDeque<(u64, f32)>,
}

enum Detector {
    Energy { levels: VecDeque<f32> },
    /// Silero's hysteresis decides speech (onset, and that a turn is still speech): from
    /// `threshold`, until below `threshold - 0.15`. Where speech ends is timed by energy, in
    /// `SLICE` steps: Silero's probability decays some 60 ms after the voice stops.
    Silero { vad: Arc<Vad>, stream: VadStream, threshold: f32, speaking: bool, levels: VecDeque<f32> },
}

/// Energy slice (8 ms) timing where speech ends inside a Silero frame.
const SLICE: usize = 128;

pub struct Endpointer {
    detector: Detector,
    frame: usize,
    onset_frames: usize,
    cut_frames: usize,
    /// Samples of silence after the last speech that end a turn.
    min_silence: u64,
    max_segment: u64,
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
    pub fn new(config: EndpointConfig) -> Self {
        Self::with_detector(config, Detector::Energy { levels: VecDeque::with_capacity(FLOOR_FRAMES) }, FRAME)
    }

    /// Speech is a Silero frame at or above `threshold` (Realtime `server_vad.threshold`).
    pub fn with_vad(config: EndpointConfig, vad: Arc<Vad>, threshold: f32) -> Self {
        let frame = vad.frame;
        let stream = vad.open();
        let levels = VecDeque::with_capacity(FLOOR_FRAMES * FRAME / SLICE);
        Self::with_detector(config, Detector::Silero { vad, stream, threshold, speaking: false, levels }, frame)
    }

    fn with_detector(config: EndpointConfig, detector: Detector, frame: usize) -> Self {
        Self {
            detector,
            frame,
            onset_frames: ONSET.div_ceil(frame),
            cut_frames: CUT / frame,
            min_silence: u64::from(config.min_silence_ms) * 16,
            max_segment: u64::from(config.max_segment_ms) * 16,
            buf: Vec::new(),
            base: 0,
            scanned: 0,
            run: 0,
            open: None,
            next_index: 0,
        }
    }

    /// Feed 16 kHz samples; returns the segments they close, in order.
    pub fn push(&mut self, samples: &[f32]) -> Vec<Segment> {
        self.buf.extend_from_slice(samples);
        let mut done = Vec::new();
        while self.scanned + self.frame as u64 <= self.end() {
            let at = (self.scanned - self.base) as usize;
            let (score, speech, loud_end, confident) = self.classify(at);
            if let Some(segment) = self.frame(self.scanned, score, speech, loud_end, confident) {
                done.push(segment);
            }
            self.scanned += self.frame as u64;
        }
        let keep = match &self.open {
            Some(open) => open.start,
            None => self.scanned.saturating_sub(((self.onset_frames + 1) * self.frame + CONTEXT) as u64),
        };
        if keep > self.base {
            self.buf.drain(..(keep - self.base) as usize);
            self.base = keep;
        }
        done
    }

    /// End of stream: the open segment, through the last sample.
    pub fn finish(&mut self) -> Option<Segment> {
        let open = self.open.take()?;
        let end = self.end();
        Some(self.cut(open.index, open.start, end, open.speech))
    }

    /// The open turn as it would close now, once [`CONTEXT`] of silence has followed its speech:
    /// its audio is then final unless speech resumes, so its transcription can start before the
    /// turn's silence runs out. Offered once per speech end.
    pub fn tentative(&mut self) -> Option<Segment> {
        let open = self.open.as_mut()?;
        let end = open.last_speech_end + CONTEXT as u64;
        if self.scanned < end || open.offered == Some(open.last_speech_end) {
            return None;
        }
        open.offered = Some(open.last_speech_end);
        let (index, start, speech) = (open.index, open.start, open.speech);
        Some(self.cut(index, start, end, speech))
    }

    /// The open segment's index and audio so far.
    pub fn open_audio(&self) -> Option<(u64, &[f32])> {
        self.open.as_ref().map(|o| (o.index, &self.buf[(o.start - self.base) as usize..]))
    }

    fn end(&self) -> u64 {
        self.base + self.buf.len() as u64
    }

    /// The frame at `buf[at..]`: its cut score (lower is quieter), whether it is speech, where in it
    /// (samples from its start) speech last sounds loud, and whether it is confidently speech.
    fn classify(&mut self, at: usize) -> (f32, bool, Option<usize>, bool) {
        let frame = &self.buf[at..at + self.frame];
        let mut loud = |levels: &mut VecDeque<f32>, slice: &[f32], keep: usize| {
            let rms = (slice.iter().map(|x| x * x).sum::<f32>() / slice.len() as f32).sqrt();
            if levels.len() == keep {
                levels.pop_front();
            }
            levels.push_back(rms);
            let floor = levels.iter().copied().fold(f32::INFINITY, f32::min);
            (rms, rms > (floor * MARGIN).max(MIN_LEVEL))
        };
        match &mut self.detector {
            Detector::Energy { levels } => {
                let (rms, speech) = loud(levels, frame, FLOOR_FRAMES);
                (rms, speech, speech.then_some(frame.len()), speech)
            }
            Detector::Silero { vad, stream, threshold, speaking, levels } => {
                let p = vad.step(stream, frame);
                if p >= *threshold {
                    *speaking = true;
                } else if p < (*threshold - 0.15).max(0.01) {
                    *speaking = false;
                }
                let keep = FLOOR_FRAMES * FRAME / SLICE;
                let last_loud = frame
                    .chunks(SLICE)
                    .enumerate()
                    .filter(|(_, slice)| loud(levels, slice, keep).1)
                    .last()
                    .map(|(i, slice)| i * SLICE + slice.len());
                (p, *speaking, last_loud.filter(|_| *speaking), p >= *threshold)
            }
        }
    }

    fn frame(&mut self, at: u64, score: f32, speech: bool, loud_end: Option<usize>, confident: bool) -> Option<Segment> {
        let frame_end = at + self.frame as u64;
        let Some(open) = &mut self.open else {
            self.run = if speech { self.run + 1 } else { 0 };
            if self.run >= self.onset_frames {
                let onset = frame_end - (self.run * self.frame) as u64;
                let start = onset.saturating_sub(CONTEXT as u64).max(self.base);
                let mut recent = VecDeque::with_capacity(self.cut_frames);
                recent.push_back((at, score));
                let speech = (self.run * self.frame) as u64;
                self.open = Some(Open { index: self.next_index, start, last_speech_end: frame_end, loud_seen: false, speech, offered: None, recent });
                self.next_index += 1;
                self.run = 0;
            }
            return None;
        };
        if open.recent.len() == self.cut_frames {
            open.recent.pop_front();
        }
        open.recent.push_back((at, score));
        if confident {
            open.speech += self.frame as u64;
        }
        // Speech ends where it last sounds loud. A turn too quiet for that (a soft talker under the
        // energy margin) is timed by confident frames instead.
        let speech_end = match loud_end {
            Some(end) => {
                open.loud_seen = true;
                Some(end)
            }
            None if confident && !open.loud_seen => Some(self.frame),
            None => None,
        };
        if let Some(end) = speech_end {
            open.last_speech_end = at + end as u64;
        } else if frame_end - open.last_speech_end >= self.min_silence {
            let (index, start, speech) = (open.index, open.start, open.speech);
            let end = (open.last_speech_end + CONTEXT as u64).min(frame_end);
            self.open = None;
            return Some(self.cut(index, start, end, speech));
        }
        if frame_end - open.start >= self.max_segment {
            // The quietest recent frame ends this segment and starts the next.
            let (quiet, _) = open
                .recent
                .iter()
                .copied()
                .filter(|&(t, _)| t > open.start)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((at, score));
            let cut = quiet + self.frame as u64 / 2;
            let (index, start, speech) = (open.index, open.start, open.speech);
            open.index = self.next_index;
            open.start = cut;
            open.speech = 0;
            open.offered = None;
            open.recent.retain(|&(t, _)| t >= cut);
            self.next_index += 1;
            return Some(self.cut(index, start, cut, speech));
        }
        None
    }

    fn cut(&self, index: u64, start: u64, end: u64, speech: u64) -> Segment {
        let samples = self.buf[(start - self.base) as usize..(end - self.base) as usize].to_vec();
        Segment { index, start, end, speech, samples }
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
    fn silero_segments_follow_its_speech_frames() {
        let dir = std::path::Path::new("/home/ssm-user/models/silero-vad-v5");
        let Ok(vad) = Vad::load(&dir.join("silero_vad.pkt")) else { return };
        let vad = Arc::new(vad);
        let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
        let signal: Vec<f32> = reference["signal"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let speech: Vec<usize> = vad.probabilities(&signal).iter().enumerate().filter(|(_, &p)| p >= 0.5).map(|(f, _)| f * 512).collect();
        let segments = feed(&mut Endpointer::with_vad(CONFIG, vad.clone(), 0.5), &signal, 1_000);
        assert_eq!(segments.len(), 1);
        let (first, last) = (speech[0] as u64, *speech.last().unwrap() as u64 + 512);
        assert_eq!(segments[0].start, first - CONTEXT as u64);
        assert!(segments[0].end >= last && segments[0].end <= last + CONTEXT as u64 + 512);
        assert!(feed(&mut Endpointer::with_vad(CONFIG, vad, 0.5), &silence(5.0), 4_000).is_empty());
    }

    #[test]
    fn silero_turns_close_on_the_voice_not_on_its_decaying_probability() {
        let dir = std::path::Path::new("/home/ssm-user/models/silero-vad-v5");
        let Ok(vad) = Vad::load(&dir.join("silero_vad.pkt")) else { return };
        let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
        let audio: Vec<f32> = reference["signal"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let closed = |mut e: Endpointer| {
            let mut fed = 0u64;
            audio.chunks(160).find_map(|c| {
                fed += c.len() as u64;
                e.push(c).into_iter().next().map(|_| fed)
            })
        };
        let energy = closed(Endpointer::new(CONFIG)).expect("energy closes the burst");
        let silero = closed(Endpointer::with_vad(CONFIG, Arc::new(vad), 0.5)).expect("Silero closes the burst");
        // Both time the end by the voice's energy; Silero decides only in 32 ms frames.
        assert!(silero.abs_diff(energy) <= 512 + 160, "energy closed at {energy}, Silero at {silero}");
    }

    #[test]
    fn a_turn_is_offered_before_it_closes_with_its_final_audio() {
        let mut audio = silence(0.5);
        audio.extend(tone(2.0));
        audio.extend(silence(1.5));
        let mut e = Endpointer::new(CONFIG);
        let (mut offered, mut closed) = (Vec::new(), Vec::new());
        for chunk in audio.chunks(320) {
            closed.extend(e.push(chunk).into_iter().map(|s| (s.index, s.start, s.end, s.samples.len(), offered.len())));
            offered.extend(e.tentative().map(|s| (s.index, s.start, s.end, s.samples.len())));
        }
        assert_eq!(closed.len(), 1);
        let (index, start, end, len, offers_before) = closed[0];
        assert!(offers_before >= 1, "offered before closing");
        assert_eq!(offered.last().map(|o| (o.0, o.1, o.2, o.3)), Some((index, start, end, len)));
        assert!(closed.iter().all(|c| c.3 > 0) && e.tentative().is_none());
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
}

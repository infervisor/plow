//! Energy endpointing of a 16 kHz stream into utterance segments, for continuous WebSocket
//! sessions. Only the open segment (or a short lead-in) stays buffered.

use std::collections::VecDeque;

/// 20 ms analysis frame.
pub const FRAME: usize = 320;
/// Speech must last this many frames (100 ms) to open a segment.
const ONSET_FRAMES: usize = 5;
/// Context kept before the onset and after the last speech frame (200 ms).
const CONTEXT: usize = 3_200;
/// Frames searched (backwards, 3 s) for the quietest cut point of an overlong segment.
const CUT_FRAMES: usize = 150;
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
    pub samples: Vec<f32>,
}

struct Open {
    index: u64,
    start: u64,
    last_speech_end: u64,
    silence: usize,
    /// (frame start, rms) of the last `CUT_FRAMES` frames.
    recent: VecDeque<(u64, f32)>,
}

pub struct Endpointer {
    min_silence: usize,
    max_segment: u64,
    buf: Vec<f32>,
    /// Stream offset of `buf[0]`.
    base: u64,
    /// Stream offset of the first sample not yet analysed.
    scanned: u64,
    levels: VecDeque<f32>,
    run: usize,
    open: Option<Open>,
    next_index: u64,
}

impl Endpointer {
    pub fn new(config: EndpointConfig) -> Self {
        Self {
            min_silence: (config.min_silence_ms as usize * 16).div_ceil(FRAME),
            max_segment: u64::from(config.max_segment_ms) * 16,
            buf: Vec::new(),
            base: 0,
            scanned: 0,
            levels: VecDeque::with_capacity(FLOOR_FRAMES),
            run: 0,
            open: None,
            next_index: 0,
        }
    }

    /// Feed 16 kHz samples; returns the segments they close, in order.
    pub fn push(&mut self, samples: &[f32]) -> Vec<Segment> {
        self.buf.extend_from_slice(samples);
        let mut done = Vec::new();
        while self.scanned + FRAME as u64 <= self.end() {
            let at = (self.scanned - self.base) as usize;
            let frame = &self.buf[at..at + FRAME];
            let rms = (frame.iter().map(|x| x * x).sum::<f32>() / FRAME as f32).sqrt();
            if let Some(segment) = self.frame(self.scanned, rms) {
                done.push(segment);
            }
            self.scanned += FRAME as u64;
        }
        let keep = match &self.open {
            Some(open) => open.start,
            None => self.scanned.saturating_sub((ONSET_FRAMES * FRAME + CONTEXT + FRAME) as u64),
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
        Some(self.cut(open.index, open.start, end))
    }

    /// The open segment's index and audio so far.
    pub fn open_audio(&self) -> Option<(u64, &[f32])> {
        self.open.as_ref().map(|o| (o.index, &self.buf[(o.start - self.base) as usize..]))
    }

    fn end(&self) -> u64 {
        self.base + self.buf.len() as u64
    }

    fn frame(&mut self, at: u64, rms: f32) -> Option<Segment> {
        if self.levels.len() == FLOOR_FRAMES {
            self.levels.pop_front();
        }
        self.levels.push_back(rms);
        let floor = self.levels.iter().copied().fold(f32::INFINITY, f32::min);
        let speech = rms > (floor * MARGIN).max(MIN_LEVEL);
        let frame_end = at + FRAME as u64;
        let Some(open) = &mut self.open else {
            self.run = if speech { self.run + 1 } else { 0 };
            if self.run >= ONSET_FRAMES {
                let onset = frame_end - (self.run * FRAME) as u64;
                let start = onset.saturating_sub(CONTEXT as u64).max(self.base);
                let mut recent = VecDeque::with_capacity(CUT_FRAMES);
                recent.push_back((at, rms));
                self.open = Some(Open { index: self.next_index, start, last_speech_end: frame_end, silence: 0, recent });
                self.next_index += 1;
                self.run = 0;
            }
            return None;
        };
        if open.recent.len() == CUT_FRAMES {
            open.recent.pop_front();
        }
        open.recent.push_back((at, rms));
        if speech {
            open.silence = 0;
            open.last_speech_end = frame_end;
        } else {
            open.silence += 1;
        }
        if open.silence >= self.min_silence {
            let (index, start) = (open.index, open.start);
            let end = (open.last_speech_end + CONTEXT as u64).min(frame_end);
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
                .unwrap_or((at, rms));
            let cut = quiet + FRAME as u64 / 2;
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
}

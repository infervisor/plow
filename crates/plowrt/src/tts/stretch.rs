//! Pitch-preserving time stretch for speech output (`speed`).
//!
//! Waveform-similarity overlap-add: output frames leave at a fixed synthesis hop, and each is
//! read from the input near its ideal position, shifted by up to [`Stretch::search`] to the
//! offset whose head best matches the previous frame's natural continuation. Every output sample
//! comes from the input unresampled, so pitch is untouched; only how often a segment repeats
//! (slower) or is skipped (faster) changes. Resampling instead would move the pitch with the
//! rate, which is the chipmunk a naive `speed` gives.

/// Speeds a reply may be stretched to. Past these the overlap search stops finding a similar
/// segment within one frame and the result tears audibly.
pub(crate) const MIN_SPEED: f32 = 0.5;
pub(crate) const MAX_SPEED: f32 = 2.0;

/// Frame and search lengths, in milliseconds of the model's rate: a 30 ms frame is long enough to
/// hold a pitch period of the lowest speech F0 (~80 Hz = 12.5 ms) twice, and the search covers
/// one such period either way.
const FRAME_MS: usize = 30;
const SEARCH_MS: usize = 10;

pub(crate) struct Stretch {
    speed: f64,
    frame: usize,
    /// Synthesis hop, half a frame: a periodic Hann at 50% overlap sums to exactly 1.
    hop: usize,
    search: usize,
    window: Vec<f32>,
    /// Input from absolute sample [`Self::base`].
    buf: Vec<f32>,
    base: u64,
    /// Next output frame's index.
    k: u64,
    /// Where the previous frame was actually read: its natural continuation is the match target.
    prev: u64,
    /// The previous frame's windowed second half, overlap-added into the next frame's head.
    tail: Vec<f32>,
    /// Real input samples received, which bounds the frames: one whose ideal position is past
    /// the input would be made of [`Self::finish`]'s zero padding alone.
    received: u64,
}

impl Stretch {
    pub fn new(rate: u32, speed: f32) -> std::result::Result<Self, String> {
        if !(MIN_SPEED..=MAX_SPEED).contains(&speed) {
            return Err(format!(
                "speed {speed} is outside {MIN_SPEED}..={MAX_SPEED}"
            ));
        }
        let frame = (rate as usize * FRAME_MS / 1000).max(4) & !1;
        let hop = frame / 2;
        let window = (0..frame)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / frame as f64;
                (0.5 - 0.5 * t.cos()) as f32
            })
            .collect();
        Ok(Self {
            speed: f64::from(speed),
            frame,
            hop,
            search: rate as usize * SEARCH_MS / 1000,
            window,
            buf: Vec::new(),
            base: 0,
            k: 0,
            prev: 0,
            tail: Vec::new(),
            received: 0,
        })
    }

    /// Every output frame `samples` completes.
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        self.buf.extend_from_slice(samples);
        self.received += samples.len() as u64;
        self.drain()
    }

    /// The frames covering the end of the input, then the last one's tail. The input is padded
    /// with zeros so those frames can be formed at all (the resampler flushes the same way), so a
    /// stretched reply ends with a half-frame fade. An input shorter than one frame has nothing to
    /// overlap and leaves unstretched rather than silent.
    pub fn finish(&mut self) -> Vec<f32> {
        self.buf
            .resize(self.buf.len() + self.frame + self.search, 0.0);
        let mut out = self.drain();
        if self.k == 0 {
            let n = ((self.received - self.base) as usize).min(self.buf.len());
            out.extend_from_slice(&self.buf[..n]);
        } else {
            out.append(&mut self.tail);
        }
        self.buf.clear();
        out
    }

    fn at(&self, i: u64) -> usize {
        (i - self.base) as usize
    }

    fn drain(&mut self) -> Vec<f32> {
        let mut out = Vec::new();
        loop {
            let ideal = (self.k as f64 * self.hop as f64 * self.speed).round() as u64;
            if ideal >= self.received {
                break;
            }
            let target = self.prev + self.hop as u64;
            // The window every candidate offset and the match target must both lie inside.
            let lo = ideal.saturating_sub(self.search as u64).min(target);
            let hi = (ideal + self.search as u64 + self.frame as u64).max(target + self.hop as u64);
            if lo < self.base || hi > self.base + self.buf.len() as u64 {
                break;
            }
            let actual = if self.k == 0 {
                ideal
            } else {
                self.best(ideal, target)
            };
            let frame = &self.buf[self.at(actual)..self.at(actual) + self.frame];
            if self.k == 0 {
                // No previous tail: the head leaves unwindowed, so a reply never fades in.
                out.extend_from_slice(&frame[..self.hop]);
            } else {
                out.extend(
                    self.tail
                        .iter()
                        .zip(&frame[..self.hop])
                        .zip(&self.window[..self.hop])
                        .map(|((t, x), w)| t + x * w),
                );
            }
            self.tail.clear();
            self.tail.extend(
                frame[self.hop..]
                    .iter()
                    .zip(&self.window[self.hop..])
                    .map(|(x, w)| x * w),
            );
            self.prev = actual;
            self.k += 1;
            // Nothing before the next frame's lowest candidate is read again.
            let next = (self.k as f64 * self.hop as f64 * self.speed).round() as u64;
            let keep = next
                .saturating_sub(self.search as u64)
                .min(self.prev + self.hop as u64);
            if keep > self.base {
                let cut = self.at(keep);
                self.buf.drain(..cut);
                self.base = keep;
            }
        }
        out
    }

    /// The offset within `±search` of `ideal` whose head best matches the segment at `target`,
    /// by cross-correlation normalized on the candidate (the target's norm is the same for all).
    fn best(&self, ideal: u64, target: u64) -> u64 {
        let want = &self.buf[self.at(target)..self.at(target) + self.hop];
        let lo = ideal.saturating_sub(self.search as u64).max(self.base);
        let hi = ideal + self.search as u64;
        let mut at = ideal;
        let mut score = f32::NEG_INFINITY;
        for c in lo..=hi {
            let cand = &self.buf[self.at(c)..self.at(c) + self.hop];
            let (mut dot, mut norm) = (0.0f32, 0.0f32);
            for (w, x) in want.iter().zip(cand) {
                dot += w * x;
                norm += x * x;
            }
            let s = dot / (norm.sqrt() + 1e-9);
            if s > score {
                (score, at) = (s, c);
            }
        }
        at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 24_000;

    fn tone(n: usize, hz: f64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * hz * i as f64 / f64::from(RATE)).sin() as f32 * 0.5
            })
            .collect()
    }

    /// Rising zero crossings per second, one per cycle: a pure tone's is its own frequency, and a
    /// pitch-preserving stretch leaves it alone however much the duration changes. A resampler
    /// would move it with the rate.
    fn zcr(pcm: &[f32]) -> f64 {
        let n = pcm.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        n as f64 * f64::from(RATE) / pcm.len() as f64
    }

    #[test]
    fn a_stretched_tone_keeps_its_pitch_and_takes_the_asked_duration() {
        let input = tone(RATE as usize, 220.0);
        for speed in [0.5f32, 0.8, 1.25, 2.0] {
            let mut s = Stretch::new(RATE, speed).unwrap();
            let mut out = s.push(&input);
            out.extend(s.finish());
            // Every output frame's head plus the last one's tail: the whole input is consumed,
            // so a dropped tail or an unformed final frame shows up here.
            let want =
                input.len() as f64 / f64::from(speed) + (RATE * FRAME_MS as u32 / 2000) as f64;
            let off = (out.len() as f64 - want).abs();
            assert!(
                off <= f64::from(RATE) * FRAME_MS as f64 / 2000.0,
                "speed {speed}: {} samples, want ~{want}",
                out.len()
            );
            assert!(
                (zcr(&out) - 220.0).abs() < 12.0,
                "speed {speed}: pitch moved to {} Hz",
                zcr(&out)
            );
            assert!(out.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
        }
    }

    /// Fed one sample at a time or all at once, the output is the same: the overlap search reads
    /// only buffered input, never a piece boundary.
    #[test]
    fn the_output_does_not_depend_on_the_piece_boundaries() {
        let input = tone(12_000, 180.0);
        let whole = {
            let mut s = Stretch::new(RATE, 1.4).unwrap();
            let mut o = s.push(&input);
            o.extend(s.finish());
            o
        };
        let pieced = {
            let mut s = Stretch::new(RATE, 1.4).unwrap();
            let mut o = Vec::new();
            for c in input.chunks(97) {
                o.extend(s.push(c));
            }
            o.extend(s.finish());
            o
        };
        assert_eq!(whole, pieced);
    }

    #[test]
    fn only_speeds_it_can_hold_together_are_accepted() {
        assert!(Stretch::new(RATE, MIN_SPEED).is_ok() && Stretch::new(RATE, MAX_SPEED).is_ok());
        assert!(Stretch::new(RATE, 0.49).is_err() && Stretch::new(RATE, 2.01).is_err());
    }
}

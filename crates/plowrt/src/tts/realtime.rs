//! Real-time admission for codec-LM speech: every request shares the model's decode steps, so
//! each one admitted slows the others. A request is admitted only while the decode step projected
//! at one more generating request keeps every playing stream ahead of its playback: its banked
//! audio (lead) covers the deficit of its remaining frames at the projected rate. Requests wait in
//! arrival order; one that cannot be admitted within `PLOW_TTS_ADMIT_WAIT_MS` is refused (429 +
//! `Retry-After`). The step is measured from the streams' own frame times against the number of
//! generating requests, so the rule holds on any GPU and model.

use std::collections::VecDeque;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Lead held back from the deficit test: codec windows and emission between generation and client.
const GUARD_S: f64 = 0.25;
/// Lead a stream with segments to go keeps for the next one's prefill and first window.
const JOIN_S: f64 = 0.3;
/// Longest a new stream holds its first audio to bank the lead its own deficit needs.
pub const MAX_HOLD_S: f64 = 1.5;
/// Prior per-request growth of the decode step, relative to the mean step (Llama-3.2-3B BF16 on
/// L40S: 8.96 ms at B=1, 13.81 at B=32), weighted as `PRIOR_WEIGHT` samples one width apart.
const PRIOR_SLOPE_REL: f64 = 0.017;
const PRIOR_WEIGHT: f64 = 200.0;
/// Per-sample decay of the step fit.
const DECAY: f64 = 0.998;
/// Frames between step samples.
const SAMPLE_FRAMES: usize = 4;
/// Finished segments the frames-per-character quantile is taken over, and the fewest it needs;
/// until then a segment is projected at its budget.
const FPC_SAMPLES: usize = 256;
const FPC_MIN_SAMPLES: usize = 16;
/// Quantile a segment's frames are projected at, capped by its budget. A mean (or mean plus two
/// deviations) let streams underrun: Orpheus runs ~15% of segments to its budget (soak: mean
/// 1.23, p90 1.78 frames per character against a 1.8 budget), so its p95 is the budget.
const FPC_QUANTILE: f64 = 0.95;

/// Frames per input character of the last finished segments (the work generated: runs to the
/// budget and mute attempts included), and their projection quantile, kept current on `add`.
#[derive(Clone, Debug, Default)]
struct Fpc {
    samples: VecDeque<f64>,
    high: Option<f64>,
}

impl Fpc {
    fn add(&mut self, x: f64) {
        if self.samples.len() == FPC_SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back(x);
        if self.samples.len() >= FPC_MIN_SAMPLES {
            let mut v: Vec<f64> = self.samples.iter().copied().collect();
            v.sort_by(f64::total_cmp);
            self.high = Some(v[((v.len() - 1) as f64 * FPC_QUANTILE).round() as usize]);
        }
    }

    /// The projection quantile; `None` (project at the budget) before `FPC_MIN_SAMPLES`.
    fn high(&self) -> Option<f64> {
        self.high
    }
}

/// Frames a segment of `chars` characters with a `budget` of frames is projected to generate.
fn projected(chars: usize, budget: usize, fpc: Option<f64>) -> f64 {
    fpc.map_or(budget as f64, |q| (chars as f64 * q).min(budget as f64))
}

/// Samples a width's own step needs before it is trusted, and its EWMA weight.
const WIDTH_MIN_SAMPLES: u32 = 8;
const WIDTH_ALPHA: f64 = 0.05;
/// Generating requests admitted beyond the widest width with a trusted step. Cold, with no
/// samples, that admits `RAMP_WIDTH`; then the width grows as the step is measured, so a burst
/// cannot overshoot before feedback, nor cross a ladder rung on the line's extrapolation.
const RAMP_WIDTH: usize = 8;

/// Decode seconds per token in the generating width: a line fit with exponential forgetting (the
/// slope pulled to the prior while the widths seen do not spread), and each width's own measured
/// step. The step is not linear: the decode ladder runs a width past a rung on the next rung
/// (L40S Orpheus: 13.8 ms at 32 rows, 18.5 ms at 33-64), which the line under-predicts: a width
/// with its own measured step predicts that, and an unmeasured one never below the step measured
/// at any narrower width.
#[derive(Clone, Debug, Default)]
struct StepFit {
    s0: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
    /// EWMA step and sample count per width.
    by_width: Vec<(f64, u32)>,
}

impl StepFit {
    fn add(&mut self, w: f64, y: f64) {
        for s in [&mut self.s0, &mut self.sx, &mut self.sy, &mut self.sxx, &mut self.sxy] {
            *s *= DECAY;
        }
        self.s0 += 1.0;
        self.sx += w;
        self.sy += y;
        self.sxx += w * w;
        self.sxy += w * y;
        let i = w as usize;
        if self.by_width.len() <= i {
            self.by_width.resize(i + 1, (0.0, 0));
        }
        let (m, n) = &mut self.by_width[i];
        *m = if *n == 0 { y } else { (1.0 - WIDTH_ALPHA) * *m + WIDTH_ALPHA * y };
        *n += 1;
    }

    fn predict(&self, w: f64) -> Option<f64> {
        if self.s0 < 1.0 {
            return None;
        }
        let (mx, my) = (self.sx / self.s0, self.sy / self.s0);
        let cxx = (self.sxx - self.s0 * mx * mx).max(0.0);
        let cxy = self.sxy - self.s0 * mx * my;
        let slope = ((cxy + PRIOR_WEIGHT * PRIOR_SLOPE_REL * my) / (cxx + PRIOR_WEIGHT)).max(0.0);
        let line = (my + slope * (w - mx)).max(0.0);
        let trusted = |&(m, n): &(f64, u32)| (n >= WIDTH_MIN_SAMPLES).then_some(m);
        if let Some(own) = self.by_width.get(w as usize).and_then(trusted) {
            return Some(own);
        }
        Some(self.by_width.iter().take(w as usize + 1).filter_map(trusted).fold(line, f64::max))
    }

    /// The widest width whose own step is trusted (0: none).
    fn explored(&self) -> usize {
        self.by_width.iter().rposition(|&(_, n)| n >= WIDTH_MIN_SAMPLES).unwrap_or(0)
    }
}

/// An admitted request's live state. Its LM drain and its stream update it with relaxed atomics,
/// off the admission lock; admission reads it under that lock only to decide.
#[derive(Debug, Default)]
struct Live {
    stream: bool,
    chars: Vec<usize>,
    budget: Vec<usize>,
    seg: AtomicUsize,
    seg_frames: AtomicUsize,
    /// Nanoseconds after `RealTime::epoch` of the first audio, plus one (0: none yet).
    first_audio: AtomicU64,
    /// Audio sent to the client, seconds (f64 bits).
    audio_s: AtomicU64,
    lm_done: AtomicBool,
    /// Step sampling: when the current sample started (as `first_audio`), and frames since.
    sample_at: AtomicU64,
    since_sample: AtomicUsize,
}

impl Live {
    /// A segment past its projection is running on toward its budget.
    fn remaining_frames(&self, fpc: Option<f64>) -> f64 {
        if self.lm_done.load(Relaxed) {
            return 0.0;
        }
        let k = self.seg.load(Relaxed);
        let seg = |i: usize| projected(self.chars[i], self.budget[i], fpc);
        let done = self.seg_frames.load(Relaxed) as f64;
        let cur = match self.chars.get(k) {
            None => 0.0,
            Some(_) if done < seg(k) => seg(k) - done,
            Some(_) => (self.budget[k] as f64 - done).max(1.0),
        };
        cur + (k + 1..self.chars.len()).map(seg).sum::<f64>()
    }

    /// Unplayed audio at the client beyond the guard (and a join's reserve with segments to go);
    /// `None` before the first audio.
    fn lead(&self, epoch: Instant, now: Instant) -> Option<f64> {
        let first = self.first_audio.load(Relaxed).checked_sub(1)?;
        let played = now.saturating_duration_since(epoch).as_secs_f64() - first as f64 * 1e-9;
        let join = if self.seg.load(Relaxed) + 1 < self.chars.len() { JOIN_S } else { 0.0 };
        Some(f64::from_bits(self.audio_s.load(Relaxed)) - played - GUARD_S - join)
    }
}

struct Inner {
    next: u64,
    active: Vec<(u64, Arc<Live>)>,
    queue: VecDeque<u64>,
    fit: StepFit,
    fpc: Fpc,
}

impl Inner {
    /// Seconds per frame at `width` generating requests; `None` before any sample.
    fn frame_s(&self, width: usize, frame_codes: f64) -> Option<f64> {
        self.fit.predict(width as f64).map(|s| s * frame_codes)
    }
}

/// What a request will generate: whether it plays as it generates, and per segment its
/// characters and frame budget.
#[derive(Clone, Debug)]
pub struct Need {
    pub stream: bool,
    pub chars: Vec<usize>,
    pub budget: Vec<usize>,
}

pub struct RealTime {
    inner: Mutex<Inner>,
    /// Admitted requests still generating.
    width: AtomicUsize,
    epoch: Instant,
    changed: tokio::sync::Notify,
    /// Seconds of audio per codec frame.
    audio_per_frame: f64,
    frame_codes: f64,
    enabled: bool,
    wait: Duration,
}

/// The admission rule: whether one more generating request stays within `RAMP_WIDTH` of the
/// widest measured width and keeps every playing stream, and the new request itself after at most
/// [`MAX_HOLD_S`] of held first audio, ahead of playback.
fn fits(inner: &Inner, width: usize, need: &Need, rt: (f64, f64, Instant), now: Instant) -> bool {
    let (audio_per_frame, frame_codes, epoch) = rt;
    if width == 0 {
        return true;
    }
    if width + 1 > inner.fit.explored() + RAMP_WIDTH {
        return false;
    }
    let Some(p) = inner.frame_s(width + 1, frame_codes) else { return true };
    let a = audio_per_frame;
    if p <= a {
        return true;
    }
    let fpc = inner.fpc.high();
    for (_, s) in inner.active.iter().filter(|(_, s)| s.stream && !s.lm_done.load(Relaxed)) {
        let Some(lead) = s.lead(epoch, now) else { continue };
        if s.remaining_frames(fpc) * (p - a) > lead {
            return false;
        }
    }
    let frames: f64 = need.chars.iter().zip(&need.budget).map(|(&c, &b)| projected(c, b, fpc)).sum();
    !need.stream || frames * (p - a) <= MAX_HOLD_S * a / p
}

impl RealTime {
    pub fn new(audio_per_frame: f64, frame_codes: usize) -> Arc<Self> {
        let rt = crate::config::RuntimeConfig::get();
        Self::with(audio_per_frame, frame_codes, rt.tts_realtime, Duration::from_millis(rt.tts_admit_wait_ms))
    }

    fn with(audio_per_frame: f64, frame_codes: usize, enabled: bool, wait: Duration) -> Arc<Self> {
        Arc::new(RealTime {
            inner: Mutex::new(Inner { next: 0, active: Vec::new(), queue: VecDeque::new(), fit: StepFit::default(), fpc: Fpc::default() }),
            width: AtomicUsize::new(0),
            epoch: Instant::now(),
            changed: tokio::sync::Notify::new(),
            audio_per_frame,
            frame_codes: frame_codes as f64,
            enabled,
            wait,
        })
    }

    fn since_epoch(&self, t: Instant) -> u64 {
        t.saturating_duration_since(self.epoch).as_nanos() as u64 + 1
    }

    /// Wait in arrival order for admission; `Err` is the `Retry-After` of a refusal. Disabled,
    /// every request is admitted at once and nothing is tracked.
    pub async fn admit(self: &Arc<Self>, need: Need) -> Result<Ticket, Duration> {
        let live = |need: Need| Arc::new(Live { stream: need.stream, chars: need.chars, budget: need.budget, ..Live::default() });
        if !self.enabled {
            return Ok(Ticket { rt: Arc::clone(self), id: 0, live: live(need) });
        }
        let id = {
            let mut g = self.inner.lock();
            g.next += 1;
            let id = g.next;
            g.queue.push_back(id);
            id
        };
        // A request dropped while it waits (client gone) leaves the queue with it.
        let _queued = Queued { rt: self, id };
        let deadline = Instant::now() + self.wait;
        loop {
            let changed = self.changed.notified();
            let now = Instant::now();
            {
                let mut g = self.inner.lock();
                let head = g.queue.front() == Some(&id);
                let width = self.width.load(Relaxed);
                if head && fits(&g, width, &need, (self.audio_per_frame, self.frame_codes, self.epoch), now) {
                    g.queue.pop_front();
                    let l = live(need);
                    g.active.push((id, Arc::clone(&l)));
                    self.width.fetch_add(1, Relaxed);
                    drop(g);
                    self.changed.notify_waiters();
                    return Ok(Ticket { rt: Arc::clone(self), id, live: l });
                }
                if now >= deadline {
                    let retry = self.retry_after(&g);
                    tracing::debug!(retry_s = retry.as_secs(), "tts: admission wait exceeded; refusing");
                    return Err(retry);
                }
            }
            let poll = deadline.saturating_duration_since(now).min(Duration::from_millis(20));
            let _ = tokio::time::timeout(poll, changed).await;
        }
    }

    /// Until the soonest generating request is projected to finish, 1..=30 s.
    fn retry_after(&self, g: &Inner) -> Duration {
        let p = g.frame_s(self.width.load(Relaxed), self.frame_codes).unwrap_or(self.audio_per_frame);
        let fpc = g.fpc.high();
        let soonest = g.active.iter().map(|(_, s)| s.remaining_frames(fpc) * p).filter(|&t| t > 0.0).fold(f64::INFINITY, f64::min);
        Duration::from_secs(soonest.ceil().clamp(1.0, 30.0) as u64)
    }
}

struct Queued<'a> {
    rt: &'a RealTime,
    id: u64,
}

impl Drop for Queued<'_> {
    fn drop(&mut self) {
        let mut g = self.rt.inner.lock();
        let len = g.queue.len();
        g.queue.retain(|&q| q != self.id);
        if g.queue.len() != len {
            drop(g);
            self.rt.changed.notify_waiters();
        }
    }
}

/// An admitted request's seat; dropping it frees the seat. Every per-frame call is O(1) on the
/// request's own atomics; only a step sample (every `SAMPLE_FRAMES` frames) takes the lock.
pub struct Ticket {
    rt: Arc<RealTime>,
    /// 0: admission disabled, nothing tracked.
    id: u64,
    live: Arc<Live>,
}

impl Ticket {
    fn tracked(&self) -> bool {
        self.id != 0
    }

    /// Segment `k` starts generating (its prefill is not a decode-step sample).
    pub fn segment(&self, k: usize) {
        if !self.tracked() {
            return;
        }
        let l = &self.live;
        l.seg.store(k, Relaxed);
        l.seg_frames.store(0, Relaxed);
        l.sample_at.store(0, Relaxed);
        l.since_sample.store(0, Relaxed);
    }

    /// One more frame generated.
    pub fn frame(&self) {
        if !self.tracked() {
            return;
        }
        let l = &self.live;
        l.seg_frames.fetch_add(1, Relaxed);
        let now = Instant::now();
        let at = l.sample_at.load(Relaxed);
        if at == 0 {
            l.sample_at.store(self.rt.since_epoch(now), Relaxed);
            return;
        }
        if l.since_sample.fetch_add(1, Relaxed) + 1 < SAMPLE_FRAMES {
            return;
        }
        let t = self.rt.since_epoch(now);
        l.sample_at.store(t, Relaxed);
        l.since_sample.store(0, Relaxed);
        let step = t.saturating_sub(at) as f64 * 1e-9 / (SAMPLE_FRAMES as f64 * self.rt.frame_codes);
        let width = self.rt.width.load(Relaxed);
        self.rt.inner.lock().fit.add(width as f64, step);
    }

    /// The current segment ended: it teaches frames per character.
    pub fn segment_done(&self) {
        if !self.tracked() {
            return;
        }
        let l = &self.live;
        let (k, frames) = (l.seg.load(Relaxed), l.seg_frames.load(Relaxed));
        if let Some(&chars) = l.chars.get(k).filter(|&&c| c > 0 && frames > 0) {
            self.rt.inner.lock().fpc.add(frames as f64 / chars as f64);
        }
    }

    /// Every segment has generated: no longer part of the decode width.
    pub fn lm_done(&self) {
        if self.tracked() && !self.live.lm_done.swap(true, Relaxed) {
            self.rt.width.fetch_sub(1, Relaxed);
            self.rt.changed.notify_waiters();
        }
    }

    /// Audio sent to the client so far, seconds.
    pub fn sent(&self, audio_s: f64) {
        if self.tracked() {
            self.live.audio_s.store(audio_s.to_bits(), Relaxed);
        }
    }

    pub fn first_audio(&self) {
        if self.tracked() {
            let t = self.rt.since_epoch(Instant::now());
            let _ = self.live.first_audio.compare_exchange(0, t, Relaxed, Relaxed);
        }
    }

    /// Frames this request is projected still to generate (0 when admission is off).
    pub fn remaining_frames(&self) -> f64 {
        if !self.tracked() {
            return 0.0;
        }
        let fpc = self.rt.inner.lock().fpc.high();
        self.live.remaining_frames(fpc)
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if !self.tracked() {
            return;
        }
        self.lm_done();
        self.rt.inner.lock().active.retain(|(id, _)| *id != self.id);
        self.rt.changed.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: f64 = 2048.0 / 24000.0;

    fn inner(step_at: impl Fn(f64) -> f64) -> Inner {
        let mut fit = StepFit::default();
        for i in 0..2000 {
            let w = (1 + i % 48) as f64;
            fit.add(w, step_at(w));
        }
        let mut fpc = Fpc::default();
        for _ in 0..FPC_MIN_SAMPLES {
            fpc.add(1.0);
        }
        Inner { next: 0, active: Vec::new(), queue: VecDeque::new(), fit, fpc }
    }

    /// A request whose budget (2 frames per character) never binds the 1.0 learnt rate.
    fn need(stream: bool, chars: Vec<usize>) -> Need {
        Need { stream, budget: chars.iter().map(|c| 2 * c).collect(), chars }
    }

    fn live(chars: Vec<usize>, budget: Vec<usize>, frames: usize) -> Live {
        Live { stream: true, chars, budget, seg_frames: AtomicUsize::new(frames), ..Live::default() }
    }

    struct Bench {
        inner: Inner,
        epoch: Instant,
        now: Instant,
    }

    impl Bench {
        fn new(step_at: impl Fn(f64) -> f64) -> Self {
            let epoch = Instant::now();
            Bench { inner: inner(step_at), epoch, now: epoch + Duration::from_secs(100) }
        }

        /// A stream that generated `frames` (all sent) and has played `played_s`.
        fn play(&mut self, chars: usize, frames: usize, played_s: f64) -> Arc<Live> {
            let l = live(vec![chars], vec![2 * chars], frames);
            l.audio_s.store((frames as f64 * A).to_bits(), Relaxed);
            let first = (self.now - Duration::from_secs_f64(played_s)).saturating_duration_since(self.epoch).as_nanos() as u64 + 1;
            l.first_audio.store(first, Relaxed);
            let l = Arc::new(l);
            self.inner.active.push((self.inner.active.len() as u64 + 1, Arc::clone(&l)));
            l
        }

        fn width(&self) -> usize {
            self.inner.active.iter().filter(|(_, l)| !l.lm_done.load(Relaxed)).count()
        }

        fn fits(&self, need: &Need) -> bool {
            fits(&self.inner, self.width(), need, (A, 7.0, self.epoch), self.now)
        }
    }

    /// The L40S Orpheus step: 8.96 ms at B=1, 13.81 at B=32.
    fn l40s(w: f64) -> f64 {
        (8.96 + (w - 1.0) * (13.81 - 8.96) / 31.0) * 1e-3
    }

    #[test]
    fn fit_recovers_a_linear_step_and_falls_back_to_the_prior() {
        let i = inner(l40s);
        for w in [1.0, 16.0, 40.0] {
            assert!((i.fit.predict(w).unwrap() - l40s(w)).abs() < 1e-4, "w={w}");
        }
        let mut one = StepFit::default();
        assert_eq!(one.predict(2.0), None);
        for _ in 0..100 {
            one.add(4.0, 0.010);
        }
        let p = one.predict(5.0).unwrap();
        assert!(p > 0.010 && p < 0.0102, "{p}");
    }

    /// Past a ladder rung the step jumps: once a width there is measured, no wider prediction
    /// falls back to the line below it.
    #[test]
    fn fit_never_predicts_below_a_measured_rung() {
        let rung = |w: f64| if w <= 32.0 { l40s(w) } else { 18.5e-3 };
        let mut f = StepFit::default();
        for i in 0..2000 {
            f.add((1 + i % 32) as f64, rung((1 + i % 32) as f64));
        }
        assert!(f.predict(34.0).unwrap() < 15e-3, "unseen: the line");
        assert_eq!(f.explored(), 32);
        for _ in 0..WIDTH_MIN_SAMPLES {
            f.add(33.0, rung(33.0));
        }
        assert!(f.predict(33.0).unwrap() >= 18.4e-3 && f.predict(40.0).unwrap() >= 18.4e-3);
        assert!(f.predict(16.0).unwrap() < 12e-3, "narrower widths keep their own step");
    }

    #[test]
    fn segments_project_at_the_tail_capped_by_their_budget() {
        let mut f = Fpc::default();
        assert_eq!(projected(100, 180, f.high()), 180.0, "no samples: the budget");
        // Orpheus-like: 85% speak at ~1.1 frames per character, 15% run to the 1.8 budget.
        for i in 0..200 {
            f.add(if i % 20 < 3 { 1.8 } else { 1.1 });
        }
        assert_eq!(f.high(), Some(1.8));
        assert_eq!(projected(100, 180, f.high()), 180.0);
        assert_eq!(projected(100, 150, f.high()), 150.0);
        let mut v = Fpc::default();
        for i in 0..200 {
            v.add(0.7 + 0.001 * (i % 100) as f64);
        }
        assert!((projected(100, 130, v.high()) - 79.5).abs() < 0.6);
        // A segment past its projection runs on toward its budget.
        let a = live(vec![100, 50], vec![180, 90], 120);
        assert_eq!(a.remaining_frames(Some(1.0)), 60.0 + 50.0);
        assert_eq!(a.remaining_frames(None), 60.0 + 90.0);
    }

    #[test]
    fn admits_freely_while_the_step_stays_real_time() {
        let mut b = Bench::new(l40s);
        // 7 x 11.5 ms = 80 ms < 85 ms a frame: real time at width 17 whatever the backlog.
        for _ in 0..16 {
            b.play(2000, 10, 0.5);
        }
        assert!(b.fits(&need(true, vec![2000])));
    }

    #[test]
    fn a_stream_without_lead_blocks_widening_past_real_time() {
        let mut b = Bench::new(l40s);
        let short = need(true, vec![40]);
        for _ in 0..24 {
            // Plenty of lead: generated 100 frames (8.5 s), played 2 s, 60 frames to go.
            b.play(160, 100, 2.0);
        }
        assert!(b.fits(&short), "every stream is far ahead");
        // One long stream barely ahead: 30 frames banked, 2.4 s played, 1500 frames to go.
        let long = b.play(1530, 30, 2.4);
        assert!(!b.fits(&short));
        // A whole (non-playing) request is held back by the same stream.
        assert!(!b.fits(&need(false, vec![40])));
        // Finished generating: no longer constrained, no longer in the width.
        long.lm_done.store(true, Relaxed);
        assert!(b.fits(&short));
    }

    #[test]
    fn a_long_new_stream_waits_for_a_real_time_width() {
        let mut b = Bench::new(l40s);
        for _ in 0..30 {
            b.play(160, 100, 2.0);
        }
        assert!(b.fits(&need(true, vec![40])));
        assert!(!b.fits(&need(true, vec![150; 10])));
        b.inner.active.truncate(8);
        assert!(b.fits(&need(true, vec![150; 10])));
    }

    /// Cold (no step sample yet) a burst is admitted only `RAMP_WIDTH` wide, and later only
    /// `RAMP_WIDTH` past the widest width whose step has been measured.
    #[tokio::test(flavor = "current_thread")]
    async fn a_cold_burst_does_not_overshoot() {
        let rt = RealTime::with(A, 7, true, Duration::from_millis(30));
        let mut held = Vec::new();
        while let Ok(t) = rt.admit(need(false, vec![40])).await {
            held.push(t);
            assert!(held.len() <= 64);
        }
        assert_eq!(held.len(), RAMP_WIDTH);
        // The step measured up to width 8 (real time there): 8 more may join.
        {
            let mut g = rt.inner.lock();
            for w in 1..=RAMP_WIDTH {
                for _ in 0..WIDTH_MIN_SAMPLES {
                    g.fit.add(w as f64, l40s(w as f64));
                }
            }
        }
        while let Ok(t) = rt.admit(need(false, vec![40])).await {
            held.push(t);
            assert!(held.len() <= 64);
        }
        assert_eq!(held.len(), 2 * RAMP_WIDTH);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn waits_in_order_then_refuses_with_retry_after() {
        let rt = RealTime::with(A, 7, true, Duration::from_millis(100));
        rt.inner.lock().fit = inner(l40s).fit;
        let blocker = rt.admit(need(true, vec![1530])).await.unwrap();
        blocker.sent(30.0 * A);
        blocker.first_audio();
        // Whole requests widen the step freely while it stays real time, then until the blocker's
        // lead no longer covers its deficit; the next waits out the bound and is refused.
        let mut held = Vec::new();
        let retry = loop {
            match rt.admit(need(false, vec![40])).await {
                Ok(t) => held.push(t),
                Err(retry) => break retry,
            }
            assert!(held.len() < 64);
        };
        assert!((16..40).contains(&held.len()), "{}", held.len());
        assert!(retry >= Duration::from_secs(1) && retry <= Duration::from_secs(30));
        assert!(rt.inner.lock().queue.is_empty());
        drop(held);
        drop(blocker);
        assert!(rt.inner.lock().active.is_empty() && rt.width.load(Relaxed) == 0);
        let next = rt.admit(need(true, vec![1530])).await;
        assert!(next.is_ok() && rt.inner.lock().active.len() == 1);
    }

    /// A waiter whose request is dropped (client gone) does not hold the queue head.
    #[tokio::test(flavor = "current_thread")]
    async fn a_dropped_waiter_leaves_the_queue() {
        let rt = RealTime::with(A, 7, true, Duration::from_secs(60));
        rt.inner.lock().fit = inner(l40s).fit;
        let blocker = rt.admit(need(true, vec![1530])).await.unwrap();
        blocker.first_audio();
        let mut held = Vec::new();
        for _ in 0..40 {
            let r = tokio::time::timeout(Duration::from_millis(50), rt.admit(need(false, vec![40]))).await;
            match r {
                Ok(t) => held.push(t.unwrap()),
                Err(_) => break,
            }
        }
        assert!(held.len() < 40, "the blocker never filled");
        assert!(rt.inner.lock().queue.is_empty(), "the timed-out waiter left the queue");
        drop(held);
        assert!(rt.admit(need(false, vec![40])).await.is_ok());
    }

    /// Disabled: everything is admitted and nothing is tracked.
    #[tokio::test(flavor = "current_thread")]
    async fn disabled_admits_all_and_tracks_nothing() {
        let rt = RealTime::with(A, 7, false, Duration::from_millis(10));
        let held: Vec<_> = futures::future::join_all((0..100).map(|_| rt.admit(need(true, vec![1000])))).await;
        assert!(held.iter().all(Result::is_ok));
        let t = held[0].as_ref().unwrap();
        t.frame();
        t.sent(1.0);
        assert!(rt.inner.lock().active.is_empty() && rt.width.load(Relaxed) == 0 && t.remaining_frames() == 0.0);
    }

    /// The per-frame path stays O(1): its cost does not grow with the number of streams.
    #[tokio::test(flavor = "current_thread")]
    async fn per_frame_cost_is_flat_in_width() {
        let rt = RealTime::with(A, 7, true, Duration::from_millis(10));
        {
            let mut g = rt.inner.lock();
            for w in 1..=128 {
                for _ in 0..WIDTH_MIN_SAMPLES {
                    g.fit.add(w as f64, 1e-4);
                }
            }
        }
        let mut held = Vec::new();
        for _ in 0..128 {
            held.push(rt.admit(need(false, vec![100])).await.unwrap());
        }
        let per_frame = |n: usize| {
            let t0 = Instant::now();
            for i in 0..20_000 {
                held[i % n].frame();
            }
            t0.elapsed().as_secs_f64() / 20_000.0
        };
        let (narrow, wide) = (per_frame(2), per_frame(128));
        assert!(wide < 4.0 * narrow + 2e-6, "{narrow:e} vs {wide:e} s per frame");
        eprintln!("per-frame bookkeeping: {:.0} ns (2 streams) / {:.0} ns (128 streams)", narrow * 1e9, wide * 1e9);
    }
}

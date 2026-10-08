//! Real-time admission for codec-LM speech: every request shares the model's decode steps, so
//! each one admitted slows the others. A request is admitted only while the decode step projected
//! at one more generating request keeps every playing stream ahead of its playback: its banked
//! audio (lead) covers the deficit of its remaining frames at the projected rate. Requests wait in
//! arrival order; one that cannot be admitted within `PLOW_TTS_ADMIT_WAIT_MS` is refused (429 +
//! `Retry-After`). The step is measured from the streams' own frame times against the number of
//! generating requests, so the rule holds on any GPU and model.

use std::collections::VecDeque;
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
/// Frames per input character before a segment has finished (about 12 chars/s).
const FRAMES_PER_CHAR0: f64 = 1.0;

/// Frames per input character, learnt from every finished segment (the work generated, runs to
/// the budget and mute attempts included): a mean and mean absolute deviation, so a projection
/// can take the slow tail (`high`).
#[derive(Clone, Copy, Debug)]
struct Fpc {
    mean: f64,
    dev: f64,
}

impl Fpc {
    fn add(&mut self, x: f64) {
        self.dev = 0.9 * self.dev + 0.1 * (x - self.mean).abs();
        self.mean = 0.9 * self.mean + 0.1 * x;
    }

    /// Two deviations over the mean: Orpheus runs ~15% of segments to its budget (soak: mean 1.23,
    /// deviation 0.24, p90 1.78 frames per character), and one deviation let those underrun.
    fn high(&self) -> f64 {
        self.mean + 2.0 * self.dev
    }
}

/// Decode seconds per token as a line in the generating width, fit with exponential forgetting;
/// the slope is pulled to the prior while the widths seen do not spread.
#[derive(Clone, Copy, Debug, Default)]
struct StepFit {
    s0: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
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
    }

    fn predict(&self, w: f64) -> Option<f64> {
        if self.s0 < 1.0 {
            return None;
        }
        let (mx, my) = (self.sx / self.s0, self.sy / self.s0);
        let cxx = (self.sxx - self.s0 * mx * mx).max(0.0);
        let cxy = self.sxy - self.s0 * mx * my;
        let slope = ((cxy + PRIOR_WEIGHT * PRIOR_SLOPE_REL * my) / (cxx + PRIOR_WEIGHT)).max(0.0);
        Some((my + slope * (w - mx)).max(0.0))
    }
}

#[derive(Debug)]
struct Active {
    id: u64,
    stream: bool,
    chars: Vec<usize>,
    seg: usize,
    seg_frames: usize,
    first_audio: Option<Instant>,
    /// Audio sent to the client, seconds.
    audio_s: f64,
    lm_done: bool,
    sample_at: Option<Instant>,
    since_sample: usize,
}

impl Active {
    fn remaining_frames(&self, fpc: f64) -> f64 {
        if self.lm_done {
            return 0.0;
        }
        let cur = self.chars.get(self.seg).map_or(0.0, |&c| (c as f64 * fpc - self.seg_frames as f64).max(1.0));
        cur + self.chars.iter().skip(self.seg + 1).sum::<usize>() as f64 * fpc
    }
}

struct Inner {
    next: u64,
    active: Vec<Active>,
    queue: VecDeque<u64>,
    fit: StepFit,
    fpc: Fpc,
}

impl Inner {
    fn width(&self) -> usize {
        self.active.iter().filter(|a| !a.lm_done).count()
    }

    /// Seconds per frame at `width` generating requests; `None` before any sample.
    fn frame_s(&self, width: usize, frame_codes: f64) -> Option<f64> {
        self.fit.predict(width as f64).map(|s| s * frame_codes)
    }
}

/// What a request will generate: whether it plays as it generates, and its segments' characters.
#[derive(Clone, Debug)]
pub struct Need {
    pub stream: bool,
    pub chars: Vec<usize>,
}

pub struct RealTime {
    inner: Mutex<Inner>,
    changed: tokio::sync::Notify,
    /// Seconds of audio per codec frame.
    audio_per_frame: f64,
    frame_codes: f64,
    enabled: bool,
    wait: Duration,
}

/// The admission rule: whether one more generating request keeps every playing stream, and the
/// new request itself after at most [`MAX_HOLD_S`] of held first audio, ahead of playback.
fn fits(inner: &Inner, need: &Need, audio_per_frame: f64, frame_codes: f64, now: Instant) -> bool {
    let width = inner.width();
    if width == 0 {
        return true;
    }
    let Some(p) = inner.frame_s(width + 1, frame_codes) else { return true };
    let a = audio_per_frame;
    if p <= a {
        return true;
    }
    let fpc = inner.fpc.high();
    let playing = inner.active.iter().filter(|s| s.stream && !s.lm_done);
    for s in playing {
        let Some(first) = s.first_audio else { continue };
        let join = if s.seg + 1 < s.chars.len() { JOIN_S } else { 0.0 };
        let lead = s.audio_s - now.saturating_duration_since(first).as_secs_f64() - GUARD_S - join;
        if s.remaining_frames(fpc) * (p - a) > lead {
            return false;
        }
    }
    !need.stream || need.chars.iter().sum::<usize>() as f64 * fpc * (p - a) <= MAX_HOLD_S * a / p
}

impl RealTime {
    pub fn new(audio_per_frame: f64, frame_codes: usize) -> Arc<Self> {
        let rt = crate::config::RuntimeConfig::get();
        Self::with(audio_per_frame, frame_codes, rt.tts_realtime, Duration::from_millis(rt.tts_admit_wait_ms))
    }

    fn with(audio_per_frame: f64, frame_codes: usize, enabled: bool, wait: Duration) -> Arc<Self> {
        Arc::new(RealTime {
            inner: Mutex::new(Inner { next: 0, active: Vec::new(), queue: VecDeque::new(), fit: StepFit::default(), fpc: Fpc { mean: FRAMES_PER_CHAR0, dev: 0.0 } }),
            changed: tokio::sync::Notify::new(),
            audio_per_frame,
            frame_codes: frame_codes as f64,
            enabled,
            wait,
        })
    }

    /// Wait in arrival order for admission; `Err` is the `Retry-After` of a refusal.
    pub async fn admit(self: &Arc<Self>, need: Need) -> Result<Ticket, Duration> {
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
                if head && (!self.enabled || fits(&g, &need, self.audio_per_frame, self.frame_codes, now)) {
                    g.queue.pop_front();
                    g.active.push(Active {
                        id,
                        stream: need.stream,
                        chars: need.chars,
                        seg: 0,
                        seg_frames: 0,
                        first_audio: None,
                        audio_s: 0.0,
                        lm_done: false,
                        sample_at: None,
                        since_sample: 0,
                    });
                    drop(g);
                    self.changed.notify_waiters();
                    return Ok(Ticket { rt: Arc::clone(self), id });
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
        let p = g.frame_s(g.width(), self.frame_codes).unwrap_or(self.audio_per_frame);
        let fpc = g.fpc.high();
        let soonest = g.active.iter().filter(|s| !s.lm_done).map(|s| s.remaining_frames(fpc) * p).fold(f64::INFINITY, f64::min);
        Duration::from_secs(soonest.ceil().clamp(1.0, 30.0) as u64)
    }

    fn with_active<R>(&self, id: u64, f: impl FnOnce(&mut Active, &mut StepFit, &mut Fpc, usize) -> R) -> Option<R> {
        let mut g = self.inner.lock();
        let width = g.width();
        let Inner { active, fit, fpc, .. } = &mut *g;
        active.iter_mut().find(|a| a.id == id).map(|a| f(a, fit, fpc, width))
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

/// An admitted request's seat; dropping it frees the seat.
pub struct Ticket {
    rt: Arc<RealTime>,
    id: u64,
}

impl Ticket {
    /// Segment `k` starts generating (its prefill is not a decode-step sample).
    pub fn segment(&self, k: usize) {
        self.rt.with_active(self.id, |a, _, _, _| {
            a.seg = k;
            a.seg_frames = 0;
            a.sample_at = None;
            a.since_sample = 0;
        });
    }

    /// One more frame generated.
    pub fn frame(&self) {
        let fc = self.rt.frame_codes;
        self.rt.with_active(self.id, |a, fit, _, width| {
            a.seg_frames += 1;
            let now = Instant::now();
            match a.sample_at {
                None => a.sample_at = Some(now),
                Some(t) => {
                    a.since_sample += 1;
                    if a.since_sample == SAMPLE_FRAMES {
                        let step = now.saturating_duration_since(t).as_secs_f64() / (SAMPLE_FRAMES as f64 * fc);
                        fit.add(width as f64, step);
                        a.sample_at = Some(now);
                        a.since_sample = 0;
                    }
                }
            }
        });
    }

    /// The current segment ended: it teaches frames per character.
    pub fn segment_done(&self) {
        self.rt.with_active(self.id, |a, _, fpc, _| {
            if let Some(&chars) = a.chars.get(a.seg).filter(|&&c| c > 0 && a.seg_frames > 0) {
                fpc.add(a.seg_frames as f64 / chars as f64);
            }
        });
    }

    /// Every segment has generated: no longer part of the decode width.
    pub fn lm_done(&self) {
        self.rt.with_active(self.id, |a, _, _, _| a.lm_done = true);
        self.rt.changed.notify_waiters();
    }

    /// Audio sent to the client so far, seconds.
    pub fn sent(&self, audio_s: f64) {
        self.rt.with_active(self.id, |a, _, _, _| a.audio_s = audio_s);
    }

    pub fn first_audio(&self) {
        self.rt.with_active(self.id, |a, _, _, _| {
            a.first_audio.get_or_insert_with(Instant::now);
        });
    }

    /// Frames this request is projected still to generate.
    pub fn remaining_frames(&self) -> f64 {
        let fpc = self.rt.inner.lock().fpc.high();
        self.rt.with_active(self.id, |a, _, _, _| a.remaining_frames(fpc)).unwrap_or(0.0)
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.rt.inner.lock().active.retain(|a| a.id != self.id);
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
            let w = (1 + i % 32) as f64;
            fit.add(w, step_at(w));
        }
        Inner { next: 0, active: Vec::new(), queue: VecDeque::new(), fit, fpc: Fpc { mean: 1.0, dev: 0.0 } }
    }

    fn playing(id: u64, chars: usize, frames: usize, played_s: f64, now: Instant) -> Active {
        Active {
            id,
            stream: true,
            chars: vec![chars],
            seg: 0,
            seg_frames: frames,
            first_audio: Some(now - Duration::from_secs_f64(played_s)),
            audio_s: frames as f64 * A,
            lm_done: false,
            sample_at: None,
            since_sample: 0,
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

    #[test]
    fn frames_per_char_projects_the_slow_tail() {
        let mut f = Fpc { mean: 1.0, dev: 0.0 };
        for i in 0..200 {
            f.add(if i % 2 == 0 { 0.8 } else { 1.6 });
        }
        assert!((f.mean - 1.2).abs() < 0.1 && f.high() > 1.8, "{f:?}");
    }

    #[test]
    fn admits_freely_while_the_step_stays_real_time() {
        let now = Instant::now();
        let mut i = inner(l40s);
        let need = Need { stream: true, chars: vec![2000] };
        // 7 x 11.5 ms = 80 ms < 85 ms a frame: real time at width 17 whatever the backlog.
        for k in 0..16 {
            i.active.push(playing(k, 2000, 10, 0.5, now));
        }
        assert!(fits(&i, &need, A, 7.0, now));
    }

    #[test]
    fn a_stream_without_lead_blocks_widening_past_real_time() {
        let now = Instant::now();
        let mut i = inner(l40s);
        let short = Need { stream: true, chars: vec![40] };
        for k in 0..24 {
            // Plenty of lead: generated 100 frames (8.5 s), played 2 s, 60 frames to go.
            i.active.push(playing(k, 160, 100, 2.0, now));
        }
        assert!(fits(&i, &short, A, 7.0, now), "every stream is far ahead");
        // One long stream barely ahead: 30 frames banked, 2.4 s played, 1500 frames to go.
        i.active.push(playing(99, 1530, 30, 2.4, now));
        assert!(!fits(&i, &short, A, 7.0, now));
        // A whole (non-playing) request is held back by the same stream.
        assert!(!fits(&i, &Need { stream: false, chars: vec![40] }, A, 7.0, now));
        // Finished generating: no longer constrained, no longer in the width.
        i.active.last_mut().unwrap().lm_done = true;
        assert!(fits(&i, &short, A, 7.0, now));
    }

    #[test]
    fn a_long_new_stream_waits_for_a_real_time_width() {
        let now = Instant::now();
        let mut i = inner(l40s);
        for k in 0..30 {
            i.active.push(playing(k, 160, 100, 2.0, now));
        }
        assert!(fits(&i, &Need { stream: true, chars: vec![40] }, A, 7.0, now));
        assert!(!fits(&i, &Need { stream: true, chars: vec![150; 10] }, A, 7.0, now));
        i.active.truncate(8);
        assert!(fits(&i, &Need { stream: true, chars: vec![150; 10] }, A, 7.0, now));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn waits_in_order_then_refuses_with_retry_after() {
        let rt = RealTime::with(A, 7, true, Duration::from_millis(100));
        rt.inner.lock().fit = inner(l40s).fit;
        let blocker = rt.admit(Need { stream: true, chars: vec![1530] }).await.unwrap();
        rt.with_active(blocker.id, |a, _, _, _| {
            a.audio_s = 30.0 * A;
            a.first_audio = Some(Instant::now());
        });
        // Whole requests widen the step freely while it stays real time, then until the blocker's
        // lead no longer covers its deficit; the next waits out the bound and is refused.
        let mut held = Vec::new();
        let retry = loop {
            match rt.admit(Need { stream: false, chars: vec![40] }).await {
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
        assert!(rt.inner.lock().active.is_empty());
        let next = rt.admit(Need { stream: true, chars: vec![1530] }).await;
        assert!(next.is_ok() && rt.inner.lock().active.len() == 1);
    }

    /// A waiter whose request is dropped (client gone) does not hold the queue head.
    #[tokio::test(flavor = "current_thread")]
    async fn a_dropped_waiter_leaves_the_queue() {
        let rt = RealTime::with(A, 7, true, Duration::from_secs(60));
        rt.inner.lock().fit = inner(l40s).fit;
        let blocker = rt.admit(Need { stream: true, chars: vec![1530] }).await.unwrap();
        rt.with_active(blocker.id, |a, _, _, _| a.first_audio = Some(Instant::now()));
        let mut held = Vec::new();
        for _ in 0..40 {
            let r = tokio::time::timeout(Duration::from_millis(50), rt.admit(Need { stream: false, chars: vec![40] })).await;
            match r {
                Ok(t) => held.push(t.unwrap()),
                Err(_) => break,
            }
        }
        assert!(held.len() < 40, "the blocker never filled");
        assert!(rt.inner.lock().queue.is_empty(), "the timed-out waiter left the queue");
        drop(held);
        assert!(rt.admit(Need { stream: false, chars: vec![40] }).await.is_ok());
    }
}

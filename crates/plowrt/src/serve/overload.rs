//! Overload control: degrade by policy when the device misses deadlines, shed new sessions last.
//!
//! The signal is the fraction of stage deadlines missed over a sliding window. The level rises one
//! step at a time when the fraction has stayed over that level's threshold for a dwell (half the
//! window; a burst of misses is not overload) and falls one step when it drops under half the
//! threshold below, at most once per dwell:
//! 1. stretch ASR partial duty;
//! 2. render TTS only to the playback clock (render-ahead slack to its minimum);
//! 3. refuse new sessions with 429 + `Retry-After`, only once level 2 has held a whole window.
//!    Existing sessions keep being served.
//!
//! On by default only under the deadline co-scheduler (multi-model); `PLOW_OVERLOAD=0|1` forces.
//! Knobs: `PLOW_OVERLOAD_WINDOW_MS`, `PLOW_OVERLOAD_MISS` (`RuntimeConfig`).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::serve::turns::Stage;

const BUCKETS: usize = 10;
/// Fewer deadlines in the window than this is too little evidence to raise the level.
const MIN_SAMPLES: u32 = 20;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Config {
    enabled: Option<bool>,
    window: Duration,
    /// Miss fraction raising level `l` to `l + 1`.
    up: [f64; 3],
}

impl Config {
    fn from_runtime() -> Config {
        let rt = crate::config::RuntimeConfig::get();
        let mut up = [0.05, 0.10, 0.20];
        for (slot, x) in up.iter_mut().zip(rt.overload_miss.split(',').filter_map(|x| x.trim().parse::<f64>().ok())) {
            *slot = x;
        }
        let window = Duration::from_millis(rt.overload_window_ms).max(Duration::from_millis(BUCKETS as u64));
        Config { enabled: rt.overload, window, up }
    }

    fn dwell(&self) -> Duration {
        self.window / 2
    }

    fn retry_after(&self) -> Duration {
        Duration::from_secs(self.window.as_secs_f64().ceil().max(1.0) as u64)
    }
}

#[derive(Clone, Copy, Default)]
struct Bucket {
    epoch: u64,
    total: u32,
    missed: u32,
}

struct Window {
    start: Instant,
    buckets: [Bucket; BUCKETS],
    level: u8,
    changed: Instant,
    /// Since when every refresh has seen the miss fraction over the current level's threshold.
    over_since: Option<Instant>,
}

impl Window {
    fn new(now: Instant) -> Window {
        Window { start: now, buckets: [Bucket::default(); BUCKETS], level: 0, changed: now, over_since: None }
    }

    fn epoch(&self, cfg: &Config, now: Instant) -> u64 {
        (now.saturating_duration_since(self.start).as_nanos() / (cfg.window / BUCKETS as u32).as_nanos().max(1)) as u64
    }

    fn observe(&mut self, cfg: &Config, missed: bool, now: Instant) {
        let epoch = self.epoch(cfg, now);
        let b = &mut self.buckets[epoch as usize % BUCKETS];
        if b.epoch != epoch {
            *b = Bucket { epoch, ..Bucket::default() };
        }
        b.total += 1;
        b.missed += u32::from(missed);
    }

    fn counts(&self, cfg: &Config, now: Instant) -> (u32, u32) {
        let epoch = self.epoch(cfg, now);
        self.buckets
            .iter()
            .filter(|b| b.total > 0 && b.epoch + BUCKETS as u64 > epoch)
            .fold((0, 0), |(t, m), b| (t + b.total, m + b.missed))
    }

    /// Step the level per the window; returns the new level when it changed.
    fn refresh(&mut self, cfg: &Config, now: Instant) -> Option<u8> {
        let (total, missed) = self.counts(cfg, now);
        let frac = if total == 0 { 0.0 } else { f64::from(missed) / f64::from(total) };
        let l = self.level as usize;
        let over = l < 3 && total >= MIN_SAMPLES && frac >= cfg.up[l];
        self.over_since = if over { Some(self.over_since.unwrap_or(now)) } else { None };
        let held = now.saturating_duration_since(self.changed);
        if held < cfg.dwell() {
            return None;
        }
        let sustained = self.over_since.is_some_and(|t| now.saturating_duration_since(t) >= cfg.dwell());
        let next = if sustained && (l < 2 || held >= cfg.window) {
            l + 1
        } else if total == 0 {
            0
        } else if l > 0 && frac < cfg.up[l - 1] / 2.0 {
            l - 1
        } else {
            l
        } as u8;
        (next != self.level).then(|| {
            self.level = next;
            self.changed = now;
            self.over_since = None;
            next
        })
    }
}

struct State {
    cfg: Config,
    window: Mutex<Window>,
}

static LEVEL: AtomicU8 = AtomicU8::new(0);

fn state() -> &'static State {
    static S: OnceLock<State> = OnceLock::new();
    S.get_or_init(|| State { cfg: Config::from_runtime(), window: Mutex::new(Window::new(Instant::now())) })
}

fn enabled(cfg: &Config) -> bool {
    cfg.enabled.unwrap_or_else(|| crate::serve::policy::installed_co_sched() == crate::serve::cosched::CoSched::Deadline)
}

fn publish(stage: Option<Stage>, level: Option<u8>) {
    if let Some(level) = level {
        LEVEL.store(level, Ordering::Relaxed);
        tracing::warn!(level, ?stage, "overload level changed");
    }
}

/// Record a completed stage's slack against its deadline (negative: missed).
pub fn observe_deadline(stage: Stage, slack_ns: i64) {
    let s = state();
    if !enabled(&s.cfg) {
        return;
    }
    let now = Instant::now();
    let mut w = s.window.lock();
    w.observe(&s.cfg, slack_ns < 0, now);
    let changed = w.refresh(&s.cfg, now);
    drop(w);
    publish(Some(stage), changed);
}

/// 0 normal, 1 stretch ASR partials, 2 TTS render-ahead to minimum, 3 shed new sessions.
pub fn level() -> u8 {
    LEVEL.load(Ordering::Relaxed)
}

/// Admit a request; at level 3 a new session is refused with the `Retry-After` to send.
pub fn admit(new_session: bool) -> Result<(), Duration> {
    let s = state();
    if !enabled(&s.cfg) {
        return Ok(());
    }
    let changed = s.window.lock().refresh(&s.cfg, Instant::now());
    publish(None, changed);
    shed(level(), new_session, &s.cfg)
}

fn shed(level: u8, new_session: bool, cfg: &Config) -> Result<(), Duration> {
    if new_session && level >= 3 {
        return Err(cfg.retry_after());
    }
    Ok(())
}

/// Request-entry gate: `None` admits, else the 429 to return. A session is new unless the turn
/// table holds a turn of it within the session TTL; a request with no session id is new.
pub fn gate(ids: &crate::serve::session::RequestIds) -> Option<axum::response::Response> {
    let s = state();
    if !enabled(&s.cfg) {
        return None;
    }
    let table = crate::serve::turns::table();
    let new = ids.session.as_deref().is_none_or(|id| !table.knows_session(id));
    let Err(retry) = admit(new) else {
        if let (true, Some(id)) = (new, &ids.session) {
            table.note_session(id);
        }
        return None;
    };
    let mut r = crate::serve::api_error(
        axum::http::StatusCode::TOO_MANY_REQUESTS,
        "server overloaded: not accepting new sessions",
        "rate_limit_error",
        Some("overloaded"),
        None,
    );
    r.headers_mut().insert(axum::http::header::RETRY_AFTER, axum::http::HeaderValue::from(retry.as_secs()));
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config { enabled: Some(true), window: Duration::from_secs(5), up: [0.05, 0.10, 0.20] }
    }

    /// Feed `n` deadlines with `missed` of them missed at `at`, then step the level.
    fn feed(w: &mut Window, c: &Config, n: u32, missed: u32, at: Instant) -> u8 {
        for i in 0..n {
            w.observe(c, i < missed, at);
        }
        w.refresh(c, at);
        w.level
    }

    #[test]
    fn level_steps_up_one_per_dwell_and_down_with_hysteresis() {
        let c = cfg();
        let t0 = Instant::now();
        let mut w = Window::new(t0);
        let dwell = c.dwell();
        // Too few samples: no rise however many miss.
        assert_eq!(feed(&mut w, &c, 10, 10, t0 + dwell), 0);
        // 30% missed: a burst is not overload; misses sustained over a dwell are.
        assert_eq!(feed(&mut w, &c, 100, 30, t0 + dwell), 0, "not sustained yet");
        assert_eq!(feed(&mut w, &c, 0, 0, t0 + dwell + dwell / 2), 0, "within the dwell");
        assert_eq!(feed(&mut w, &c, 100, 30, t0 + 2 * dwell), 1);
        // One step per sustained dwell.
        assert_eq!(feed(&mut w, &c, 100, 30, t0 + 3 * dwell), 1);
        assert_eq!(feed(&mut w, &c, 100, 30, t0 + 4 * dwell), 2);
        // Level 3 (shedding) only once level 2 has held a whole window.
        assert_eq!(feed(&mut w, &c, 100, 30, t0 + 5 * dwell), 2);
        assert_eq!(feed(&mut w, &c, 100, 30, t0 + 6 * dwell), 3);
        assert_eq!(feed(&mut w, &c, 100, 30, t0 + 7 * dwell), 3);
        // Misses that stop before a dwell has passed do not raise the level.
        let mut v = Window::new(t0);
        assert_eq!(feed(&mut v, &c, 100, 30, t0 + dwell), 0);
        assert_eq!(feed(&mut v, &c, 1000, 0, t0 + dwell + dwell / 2), 0);
        assert_eq!(feed(&mut v, &c, 0, 0, t0 + 2 * dwell), 0);
        // Window rolls over to 7% missed: under level 3's entry (20%) but over half of level 2's
        // (10% / 2), so it falls to 2 and holds there.
        let t1 = t0 + 7 * dwell + c.window;
        assert_eq!(feed(&mut w, &c, 100, 7, t1), 2);
        assert_eq!(feed(&mut w, &c, 0, 0, t1 + dwell), 2);
        // 2% is under half of level 1's 5%: down to 1, then 0.
        let t2 = t1 + dwell + c.window;
        assert_eq!(feed(&mut w, &c, 100, 2, t2), 1);
        assert_eq!(feed(&mut w, &c, 0, 0, t2 + dwell), 0);
        // An empty window (idle) drops straight to 0.
        w.level = 3;
        assert_eq!(feed(&mut w, &c, 0, 0, t2 + 3 * c.window), 0);
    }

    #[test]
    fn admit_sheds_only_new_sessions_at_level_three() {
        let c = cfg();
        assert_eq!(shed(2, true, &c), Ok(()));
        assert_eq!(shed(3, false, &c), Ok(()));
        assert_eq!(shed(3, true, &c), Err(Duration::from_secs(5)));
        assert_eq!(Config { window: Duration::from_millis(1500), ..c }.retry_after(), Duration::from_secs(2));
        assert_eq!(Config { window: Duration::from_millis(200), ..c }.retry_after(), Duration::from_secs(1));
    }
}

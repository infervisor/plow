//! §HOSTT — host time on the serving path (`PLOW_HOST_TIMING=1`). Off by default; every call
//! site is behind [`on`].
//!
//! Per request, three lines: the handler's pre-submit time (template, tokenize), submit -> first
//! token emitted on the engine thread, and handler entry -> first SSE frame; they join by order at
//! concurrency 1. Per decode tick: the tick period
//! split into the engine call (launch + device + D2H), the token emit loop (detokenize + channel
//! send), the rest of the tick body, the dispatcher<->engine handoff, and the dispatcher's own
//! work between ticks. Only decode-only ticks enter the window, so a prefill never inflates a
//! per-token figure.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

pub fn on() -> bool {
    crate::config::RuntimeConfig::get().host_timing
}

/// Decode ticks per summary line.
const WINDOW: u64 = 256;

// Written by the tick body, drained by the dispatcher once the tick has returned. Process-wide:
// with two GPU models decoding at once their windows mix.
static DEV_NS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static EMIT_NS: AtomicU64 = AtomicU64::new(0);
static EMIT_TOK: AtomicU64 = AtomicU64::new(0);

/// One decode engine call (`rows` fed rows) took `ns`.
#[inline]
pub fn engine_call(ns: u64, rows: usize) {
    DEV_NS.fetch_add(ns, Relaxed);
    ROWS.fetch_add(rows as u64, Relaxed);
}

/// One emit loop streamed `tokens` tokens in `ns`.
#[inline]
pub fn emit(ns: u64, tokens: usize) {
    EMIT_NS.fetch_add(ns, Relaxed);
    EMIT_TOK.fetch_add(tokens as u64, Relaxed);
}

/// What the dispatcher measured around one tick.
#[derive(Clone, Copy, Debug, Default)]
pub struct TickTimes {
    /// The tick body on the engine thread.
    pub tick_ns: u64,
    /// Dispatcher-side wait for the tick minus the tick body: the two thread hops.
    pub handoff_ns: u64,
    /// Dispatcher loop time from the previous tick's return to this tick's dispatch.
    pub disp_ns: u64,
}

#[derive(Debug, Default, PartialEq)]
pub struct Window {
    ticks: u64,
    rows: u64,
    tokens: u64,
    dev_ns: u64,
    emit_ns: u64,
    tick_ns: u64,
    handoff_ns: u64,
    disp_ns: u64,
}

impl Window {
    /// Fold one finished tick in, draining the engine-side accumulators either way so a prefill
    /// tick's figures never leak into the next decode tick. Returns a summary line when the
    /// window fills.
    pub fn tick(&mut self, t: TickTimes, decode_only: bool) -> Option<String> {
        let dev_ns = DEV_NS.swap(0, Relaxed);
        let rows = ROWS.swap(0, Relaxed);
        let emit_ns = EMIT_NS.swap(0, Relaxed);
        let tokens = EMIT_TOK.swap(0, Relaxed);
        self.add(t, decode_only, dev_ns, rows, emit_ns, tokens)
    }

    fn add(
        &mut self,
        t: TickTimes,
        decode_only: bool,
        dev_ns: u64,
        rows: u64,
        emit_ns: u64,
        tokens: u64,
    ) -> Option<String> {
        if !decode_only || rows == 0 {
            return None;
        }
        self.ticks += 1;
        self.rows += rows;
        self.tokens += tokens;
        self.dev_ns += dev_ns;
        self.emit_ns += emit_ns;
        self.tick_ns += t.tick_ns;
        self.handoff_ns += t.handoff_ns;
        self.disp_ns += t.disp_ns;
        (self.ticks >= WINDOW).then(|| std::mem::take(self).summary())
    }

    fn summary(&self) -> String {
        let n = self.ticks.max(1) as f64;
        let us = |ns: u64| ns as f64 / 1e3 / n;
        let period = self.tick_ns + self.handoff_ns + self.disp_ns;
        let other = self.tick_ns.saturating_sub(self.dev_ns + self.emit_ns);
        format!(
            "HOSTT decode ticks={} rows/tick={:.2} tok/tick={:.2} period_us={:.1} engine_call_us={:.1} \
             emit_us={:.1} emit_us/tok={:.2} tick_other_us={:.1} handoff_us={:.1} dispatcher_us={:.1} \
             host_share={:.2}%",
            self.ticks,
            self.rows as f64 / n,
            self.tokens as f64 / n,
            us(period),
            us(self.dev_ns),
            us(self.emit_ns),
            self.emit_ns as f64 / 1e3 / self.tokens.max(1) as f64,
            us(other),
            us(self.handoff_ns),
            us(self.disp_ns),
            100.0 * period.saturating_sub(self.dev_ns) as f64 / period.max(1) as f64,
        )
    }
}

/// Engine thread: a request's first output token was just handed to its stream.
pub fn first_token(prompt: usize, since_submit: Duration) {
    tracing::info!(
        prompt,
        submit_to_emit_us = since_submit.as_micros() as u64,
        "HOSTT first token emitted"
    );
}

/// Handler: the request is about to be submitted to the mux (`pre_submit` = handler entry to
/// here: validation, template, tokenize).
pub fn submitted(prompt: usize, pre_submit: Duration) {
    tracing::info!(
        prompt,
        pre_submit_us = pre_submit.as_micros() as u64,
        "HOSTT submitted"
    );
}

/// Handler: the first SSE frame of a request is being yielded (`ttft` from handler entry).
pub fn first_frame(prompt: usize, ttft: Duration) {
    tracing::info!(prompt, ttft_us = ttft.as_micros() as u64, "HOSTT first frame");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_decode_ticks_fill_the_window() {
        let mut w = Window::default();
        let t = TickTimes {
            tick_ns: 10_000,
            handoff_ns: 2_000,
            disp_ns: 1_000,
        };
        assert_eq!(w.add(t, false, 9_000, 4, 500, 4), None);
        assert_eq!(w.add(t, true, 9_000, 0, 0, 0), None);
        assert_eq!(w, Window::default());
        for _ in 0..WINDOW - 1 {
            assert_eq!(w.add(t, true, 9_000, 4, 400, 4), None);
        }
        let line = w.add(t, true, 9_000, 4, 400, 4).expect("window full");
        assert_eq!(w, Window::default(), "the window restarts after a summary");
        for field in [
            "ticks=256",
            "rows/tick=4.00",
            "period_us=13.0",
            "engine_call_us=9.0",
            "emit_us/tok=0.10",
            "tick_other_us=0.6",
            "handoff_us=2.0",
            "dispatcher_us=1.0",
            "host_share=30.77%",
        ] {
            assert!(line.contains(field), "{field} missing from {line}");
        }
    }
}

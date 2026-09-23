//! Serving policy: which profile the live workload is, chosen per tick.
//!
//! The campaign's two serving profiles differ in a handful of knobs, and which one wins is a
//! property of the workload, not of the deployment: measured on p12r4, adaptive prefill packing
//! wins 1024/C4 TTFT (101 vs 123 ms) and loses 128/C4 (39.1 vs 34.2), and the rung fast probe
//! wins a cold C16 backlog (P99 TTFT 170 -> 150 ms) while buying nothing at C1. A server that
//! sees both workloads has to pick one profile for the whole run.
//!
//! `PLOW_SERVE_POLICY=auto` picks per tick instead, from signals the dispatcher already has:
//! occupied decode width and queue depth. Only knobs read at a per-tick decision point move —
//! the decode rung ladder and slot capacity are derived once at startup (and feed the KV
//! admission check), so `--decode-max-rung` still pins those; under `auto` the ladder is left
//! unfiltered and [`crate::sched::rungs::RungController`] adapts the width from measured service
//! times, which is the same decision taken on evidence rather than on a flag.
//!
//! Default is `pinned`: every cert and campaign cell keeps the exact configuration it declares.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};

/// Which profile the current window looks like.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    /// Few live rows, empty queue: latency owns the tick.
    Realtime,
    /// A wide decode batch or a standing queue: throughput owns the tick.
    HighConcurrency,
}

/// Width at or above which the window is high-concurrency, and the width it must fall back to
/// before it is realtime again. The gap is the hysteresis band: C4 and C16 are the measured
/// profiles, so the band sits between them and a C8 workload does not oscillate.
const WIDE_ENTER: usize = 8;
const WIDE_LEAVE: usize = 4;
/// Ticks a class must hold before it may change again. A decode tick is ~10-15 ms here, so this
/// is ~2-3 s of dwell: long enough that a burst does not flip the profile twice inside one
/// request's decode, short enough to follow a real load change.
const DWELL_TICKS: u32 = 200;

static CLASS: AtomicU8 = AtomicU8::new(0);
static DWELL: AtomicU32 = AtomicU32::new(0);

/// The whole decision, as a pure function of the state and one tick's signals: the class in
/// force afterwards and the dwell that goes with it.
fn decide(current: Class, dwell: u32, width: usize, queued: usize) -> (Class, u32) {
    let want = if width >= WIDE_ENTER || queued > 0 {
        Class::HighConcurrency
    } else if width <= WIDE_LEAVE {
        Class::Realtime
    } else {
        current
    };
    // The class changes only after the current one has held for the dwell, so a single wide tick
    // in a latency workload — or one idle tick in a busy one — cannot move the profile.
    if want == current || dwell < DWELL_TICKS {
        (current, dwell.saturating_add(1))
    } else {
        (want, 0)
    }
}

/// Whether `PLOW_SERVE_POLICY=auto` is in force. One atomic load per query, as
/// `RuntimeConfig::get()` is.
pub fn auto() -> bool {
    crate::config::RuntimeConfig::get().serve_policy.as_deref() == Some("auto")
}

pub fn class() -> Class {
    match CLASS.load(Ordering::Relaxed) {
        0 => Class::Realtime,
        _ => Class::HighConcurrency,
    }
}

/// One tick's observation: `width` live decode rows, `queued` requests waiting for a slot.
/// Returns the class in force after it. A no-op when the policy is pinned.
pub fn observe(width: usize, queued: usize) -> Class {
    if !auto() {
        return Class::Realtime;
    }
    let current = class();
    let (next, dwell) = decide(current, DWELL.load(Ordering::Relaxed), width, queued);
    DWELL.store(dwell, Ordering::Relaxed);
    if next != current {
        CLASS.store(u8::from(next == Class::HighConcurrency), Ordering::Relaxed);
        tracing::info!(from = ?current, to = ?next, width, queued, "serve policy: profile switched");
    }
    next
}

/// Queue-driven prefill packing (`PLOW_PF_INTERLEAVE_ADAPTIVE`): the oldest prompt runs whole and
/// later ones join only while that is cheaper. It wins when a few prompts arrive together and
/// loses when the queue is deep enough that filling the launch matters more.
pub fn adaptive_packing(configured: bool) -> bool {
    if auto() { class() == Class::Realtime } else { configured }
}

/// The rung fast probe: a cold backlog tries the widest rung after one sample instead of four.
pub fn fast_probe(configured: bool) -> bool {
    if auto() { class() == Class::HighConcurrency } else { configured }
}

/// Queue TTL. The throughput profile serves `0` (nothing ages out); the latency profile keeps the
/// SLO-derived TTL, so a starved request is rejected rather than left to age.
pub fn queue_ttl_ms(configured: Option<f64>) -> Option<f64> {
    match (auto(), class()) {
        (true, Class::HighConcurrency) => Some(0.0),
        _ => configured,
    }
}

/// Whether the decode rung ladder keeps `--decode-max-rung`. Under `auto` the ladder is left
/// whole so the rung controller can widen on evidence; the flag caps capacity at startup, which
/// is also what the KV admission check sizes its reservation from.
pub fn honor_max_rung() -> bool {
    !auto()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_tick_never_moves_the_profile() {
        let (c, d) = decide(Class::Realtime, DWELL_TICKS, 16, 0);
        assert_eq!((c, d), (Class::HighConcurrency, 0), "wide, and the dwell has run");
        let (c, _) = decide(Class::Realtime, DWELL_TICKS - 1, 16, 0);
        assert_eq!(c, Class::Realtime, "wide, but the class has not held long enough");
    }

    #[test]
    fn the_band_holds_a_mid_width_workload_where_it_was() {
        for from in [Class::Realtime, Class::HighConcurrency] {
            let (c, _) = decide(from, DWELL_TICKS, 6, 0);
            assert_eq!(c, from, "6 rows is inside [{WIDE_LEAVE}, {WIDE_ENTER})");
        }
        assert_eq!(decide(Class::HighConcurrency, DWELL_TICKS, 2, 0).0, Class::Realtime);
        assert_eq!(decide(Class::Realtime, DWELL_TICKS, 1, 3).0, Class::HighConcurrency, "queued");
    }

    #[test]
    fn dwell_accumulates_while_the_class_is_stable() {
        let (_, d) = decide(Class::Realtime, 7, 1, 0);
        assert_eq!(d, 8);
    }

    #[test]
    fn pinned_policy_is_the_identity_on_configured_knobs() {
        assert!(!auto(), "the test process has no PLOW_SERVE_POLICY");
        assert!(adaptive_packing(true) && !adaptive_packing(false));
        assert!(fast_probe(true) && !fast_probe(false));
        assert_eq!(queue_ttl_ms(Some(250.0)), Some(250.0));
        assert!(honor_max_rung());
    }
}

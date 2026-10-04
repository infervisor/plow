//! The serving objective, and every scheduling decision derived from it.
//!
//! A deployment states one goal, `--objective latency|throughput|auto` (`PLOW_OBJECTIVE`). The
//! mux, the engines and the co-tenant turns read the mechanisms below instead of one knob each;
//! what a packet or backend cannot run (the decode pipeline on AMD, token batch at hd128,
//! multistep under decode roles or recurrent state) is switched off by capability where the
//! engine is built, not here.
//!
//! `auto` picks the latency or throughput rules per tick, from signals the dispatcher already
//! has: occupied decode width and queue depth. Measured on p12r4, queue-sized prefill packing wins
//! 1024/C4 TTFT (101 vs 123 ms) and loses 128/C4 (39.1 vs 34.2); the rung fast probe wins a cold
//! C16 backlog (P99 TTFT 170 -> 150 ms) and buys nothing at C1. Startup decisions (the decode
//! rung ladder, KV admission) cannot follow a per-tick class; they take the throughput side under
//! `auto`, so capacity is reserved for every slot.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Objective {
    Latency,
    Throughput,
    #[default]
    Auto,
}

impl std::str::FromStr for Objective {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "latency" => Ok(Objective::Latency),
            "throughput" => Ok(Objective::Throughput),
            "auto" => Ok(Objective::Auto),
            other => Err(format!("unknown objective `{other}` (latency | throughput | auto)")),
        }
    }
}

impl std::fmt::Display for Objective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Objective::Latency => "latency",
            Objective::Throughput => "throughput",
            Objective::Auto => "auto",
        })
    }
}

/// Which objective's rules own the current tick.
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
/// How long the window must stay narrow before throughput hands back to latency: long enough
/// that a burst's tail does not flip the class twice inside one request's decode. Entering
/// throughput is immediate — latency rules under load cost E4B c64/c128 15-20% tok/s and up to
/// 15x TTFT (queue-sized packing serialises a burst), while throughput rules at low load cost
/// little. Wall time, not ticks: an idle server runs no ticks, and a tick-counted dwell held a
/// fresh burst on the latency rules for its first 200 ticks.
const CALM_MS: u64 = 2000;
const NOT_CALM: u64 = u64::MAX;

/// Throughput decode quantum: E4B and Veena serve their best c64/c128 at K = 8 with the
/// single-step rule below (`docs/runtime/gemma4-e4b-h100.md`, `docs/runtime/tts.md`).
pub const THROUGHPUT_K: u32 = 8;

/// [`THROUGHPUT_K`] unless `PLOW_THROUGHPUT_K` (or the packet's serve default) says otherwise.
pub fn throughput_k() -> u32 {
    crate::config::RuntimeConfig::throughput_k().unwrap_or(THROUGHPUT_K)
}
/// AMD latency quantum. AMD has no lookahead pipeline, so a lone stream still needs the
/// deferred read to amortise its host turnaround; every AMD latency recipe ran K = 4.
const AMD_LATENCY_K: u32 = 4;

static CLASS: AtomicU8 = AtomicU8::new(0);
/// When the window last became narrow (ms since `EPOCH`), or `NOT_CALM`.
static CALM_SINCE: AtomicU64 = AtomicU64::new(NOT_CALM);
static EPOCH: OnceLock<Instant> = OnceLock::new();

/// The whole decision, as a pure function of the state and one tick's signals at `now` ms: the
/// class in force afterwards and the calm start that goes with it.
fn decide(current: Class, calm_since: u64, now: u64, width: usize, queued: usize) -> (Class, u64) {
    if width >= WIDE_ENTER || queued > 0 {
        (Class::HighConcurrency, NOT_CALM)
    } else if width <= WIDE_LEAVE && current == Class::HighConcurrency {
        let since = if calm_since == NOT_CALM { now } else { calm_since };
        if now.saturating_sub(since) >= CALM_MS {
            (Class::Realtime, NOT_CALM)
        } else {
            (current, since)
        }
    } else {
        (current, NOT_CALM)
    }
}

pub fn objective() -> Objective {
    crate::config::RuntimeConfig::get().objective
}

/// The class in force: fixed by a pinned objective, observed under `auto`.
pub fn class() -> Class {
    class_for(objective(), CLASS.load(Ordering::Relaxed))
}

fn class_for(objective: Objective, observed: u8) -> Class {
    match (objective, observed) {
        (Objective::Latency, _) | (Objective::Auto, 0) => Class::Realtime,
        _ => Class::HighConcurrency,
    }
}

/// One tick's observation: `width` live decode rows, `queued` requests waiting for a slot.
/// A no-op unless the objective is `auto`.
pub fn observe(width: usize, queued: usize) {
    if objective() != Objective::Auto {
        return;
    }
    let current = class_for(Objective::Auto, CLASS.load(Ordering::Relaxed));
    let now = EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64;
    let (next, calm) = decide(current, CALM_SINCE.load(Ordering::Relaxed), now, width, queued);
    CALM_SINCE.store(calm, Ordering::Relaxed);
    if next != current {
        CLASS.store(u8::from(next == Class::HighConcurrency), Ordering::Relaxed);
        tracing::info!(from = ?current, to = ?next, width, queued, "serve objective: class switched");
    }
}

/// Decode steps per host sync for rows the lookahead pipeline does not carry.
///
/// `PLOW_MULTISTEP` pins it. Otherwise latency on CUDA runs one token group per sync: a chat
/// model one token — the 12B and 26B realtime cells measured K > 1 as a pure wave artifact (p99
/// ITL = K x TPOT, TTFT and TPOT level) — and a speech model one codec frame, the unit its client
/// consumes. AMD latency runs `AMD_LATENCY_K`; throughput runs `THROUGHPUT_K`.
pub fn decode_k(amd: bool, token_group: usize) -> u32 {
    let pinned = crate::config::RuntimeConfig::get().nv.multistep;
    pinned.unwrap_or_else(|| decode_k_for(class(), amd, token_group))
}

fn decode_k_for(class: Class, amd: bool, token_group: usize) -> u32 {
    match class {
        Class::Realtime if amd => AMD_LATENCY_K,
        Class::Realtime => token_group.max(1) as u32,
        Class::HighConcurrency => throughput_k(),
    }
}

/// The quantum an engine is built for: the widest any class may ask for.
pub fn decode_k_capacity() -> u32 {
    crate::config::RuntimeConfig::get().nv.multistep.unwrap_or_else(throughput_k)
}

/// Queue-sized prefill packing: the oldest prompt runs whole and later ones join only while that
/// is cheaper. It wins when a few prompts arrive together and loses when the queue is deep enough
/// that filling the launch matters more.
pub fn adaptive_packing() -> bool {
    class() == Class::Realtime
}

/// The rung fast probe: a cold backlog tries the widest rung after one sample instead of four.
pub fn fast_probe() -> bool {
    class() == Class::HighConcurrency
}

/// Queue TTL override: throughput never sheds a waiting request (`0`); latency keeps the
/// SLO-derived TTL, so a starved request is rejected rather than left to age.
pub fn queue_ttl_ms() -> Option<f64> {
    (class() == Class::HighConcurrency).then_some(0.0)
}

/// How co-resident models share a device. `--co-sched` pins it. One model per process has nothing
/// to share; AMD needs round-robin turns for whole-grid residency; on CUDA `deadline` puts a
/// prompt owed its first token, an ASR final or a speech stream ahead of decode (it passed the
/// 10-call voice SLO, `docs/runtime/gemma4-e4b-h100.md`), and throughput takes plain turns.
pub fn co_sched(models: usize, amd: bool) -> crate::serve::cosched::CoSched {
    use crate::serve::cosched::CoSched;
    if let Some(mode) = crate::config::RuntimeConfig::get().co_sched {
        return mode;
    }
    match (models > 1, amd, objective()) {
        (false, _, _) => CoSched::Free,
        (true, true, _) | (true, false, Objective::Throughput) => CoSched::Rr,
        (true, false, _) => CoSched::Deadline,
    }
}

static CO_SCHED: OnceLock<crate::serve::cosched::CoSched> = OnceLock::new();

/// Record the mode the device turns were installed with.
pub fn set_co_sched(mode: crate::serve::cosched::CoSched) {
    let _ = CO_SCHED.set(mode);
}

/// The installed co-tenant mode (`free` before any turn is installed).
pub fn installed_co_sched() -> crate::serve::cosched::CoSched {
    CO_SCHED.get().copied().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_enters_throughput_at_once() {
        for (width, queued) in [(WIDE_ENTER, 0), (1, 3)] {
            let (c, calm) = decide(Class::Realtime, NOT_CALM, 10, width, queued);
            assert_eq!((c, calm), (Class::HighConcurrency, NOT_CALM));
        }
    }

    #[test]
    fn throughput_leaves_only_after_the_window_stays_narrow() {
        let (c, calm) = decide(Class::HighConcurrency, NOT_CALM, 1_000, 2, 0);
        assert_eq!((c, calm), (Class::HighConcurrency, 1_000), "calm starts");
        let (c, _) = decide(Class::HighConcurrency, calm, 1_000 + CALM_MS - 1, 2, 0);
        assert_eq!(c, Class::HighConcurrency, "not calm for long enough");
        let (c, _) = decide(Class::HighConcurrency, calm, 1_000 + CALM_MS, 2, 0);
        assert_eq!(c, Class::Realtime);
        let (c, calm) = decide(Class::HighConcurrency, calm, 1_500, 16, 0);
        assert_eq!((c, calm), (Class::HighConcurrency, NOT_CALM), "a wide tick resets calm");
    }

    #[test]
    fn the_band_holds_a_mid_width_workload_where_it_was() {
        for from in [Class::Realtime, Class::HighConcurrency] {
            let (c, _) = decide(from, 0, 10 * CALM_MS, 6, 0);
            assert_eq!(c, from, "6 rows is inside [{WIDE_LEAVE}, {WIDE_ENTER})");
        }
    }

    #[test]
    fn pinned_objectives_ignore_the_observed_class() {
        for observed in [0, 1] {
            assert_eq!(class_for(Objective::Latency, observed), Class::Realtime);
            assert_eq!(class_for(Objective::Throughput, observed), Class::HighConcurrency);
        }
        assert_eq!(class_for(Objective::Auto, 0), Class::Realtime);
        assert_eq!(class_for(Objective::Auto, 1), Class::HighConcurrency);
    }

    #[test]
    fn decode_quantum_per_class_and_backend() {
        assert_eq!(decode_k_for(Class::Realtime, false, 1), 1);
        assert_eq!(decode_k_for(Class::Realtime, false, 7), 7, "one Veena frame");
        assert_eq!(decode_k_for(Class::Realtime, true, 1), AMD_LATENCY_K);
        assert_eq!(decode_k_for(Class::HighConcurrency, false, 7), THROUGHPUT_K);
        assert_eq!(decode_k_for(Class::HighConcurrency, true, 1), THROUGHPUT_K);
    }

    #[test]
    fn objective_round_trips() {
        for o in [Objective::Latency, Objective::Throughput, Objective::Auto] {
            assert_eq!(o.to_string().parse::<Objective>(), Ok(o));
        }
        assert!("realtime".parse::<Objective>().is_err());
    }
}

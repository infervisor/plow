//! The serving objective, and every scheduling decision derived from it.
//!
//! A deployment states one goal, `--objective latency|throughput|auto` (`PLOW_OBJECTIVE`). The
//! mux, the engines and the co-tenant turns read the mechanisms below instead of one knob each;
//! what a packet or backend cannot run (the decode pipeline on AMD, token batch at hd128,
//! multistep under decode roles or recurrent state) is switched off by capability where the
//! engine is built, not here.
//!
//! `auto` picks the latency or throughput rules per tick, from signals the dispatcher already
//! has: occupied decode width, queue depth and the share of the KV budget live sequences hold. Measured on p12r4, queue-sized prefill packing wins
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
/// Share of the KV budget live sequences reserve at or above which the window is
/// high-concurrency whatever its width (a few long prompts fill the device as surely as many
/// short ones), and the share it must fall back under before it is realtime again.
const KV_ENTER: f64 = 0.9;
const KV_LEAVE: f64 = 0.75;

/// Throughput decode quantum: E4B and Veena serve their best c64/c128 at K = 8 with the
/// single-step rule below (`docs/runtime/gemma4-e4b-h100.md`, `docs/runtime/tts.md`).
pub const THROUGHPUT_K: u32 = 8;

/// [`THROUGHPUT_K`] unless `PLOW_THROUGHPUT_K` (or the packet's serve default) says otherwise.
/// Read once (every decode tick asks): packet serve defaults are in the environment by then.
pub fn throughput_k() -> u32 {
    static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *K.get_or_init(|| crate::config::RuntimeConfig::throughput_k().unwrap_or(THROUGHPUT_K))
}
/// AMD latency quantum. AMD has no lookahead pipeline, so a lone stream still needs the
/// deferred read to amortise its host turnaround; every AMD latency recipe ran K = 4.
const AMD_LATENCY_K: u32 = 4;

static CLASS: AtomicU8 = AtomicU8::new(0);
static SWITCHES: AtomicU64 = AtomicU64::new(0);
/// When the window last became narrow (ms since `EPOCH`), or `NOT_CALM`.
static CALM_SINCE: AtomicU64 = AtomicU64::new(NOT_CALM);
static EPOCH: OnceLock<Instant> = OnceLock::new();

/// One tick's signals.
#[derive(Clone, Copy, Debug, Default)]
pub struct Load {
    /// Live decode rows (the occupied slot extent).
    pub width: usize,
    /// Requests waiting for a slot.
    pub queued: usize,
    /// Share of the KV-capacity budget the live sequences reserve (0 without a budget).
    pub kv_used: f64,
}

/// The whole decision, as a pure function of the state and one tick's signals at `now` ms: the
/// class in force afterwards and the calm start that goes with it.
fn decide(current: Class, calm_since: u64, now: u64, load: Load) -> (Class, u64) {
    let Load { width, queued, kv_used } = load;
    if width >= WIDE_ENTER || queued > 0 || kv_used >= KV_ENTER {
        (Class::HighConcurrency, NOT_CALM)
    } else if width <= WIDE_LEAVE && kv_used < KV_LEAVE && current == Class::HighConcurrency {
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

/// One tick's observation. A no-op unless the objective is `auto`; `true` when the class
/// switched.
pub fn observe(load: Load) -> bool {
    if objective() != Objective::Auto {
        return false;
    }
    let current = class_for(Objective::Auto, CLASS.load(Ordering::Relaxed));
    let now = EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64;
    let (next, calm) = decide(current, CALM_SINCE.load(Ordering::Relaxed), now, load);
    CALM_SINCE.store(calm, Ordering::Relaxed);
    if next == current {
        return false;
    }
    CLASS.store(u8::from(next == Class::HighConcurrency), Ordering::Relaxed);
    let switches = SWITCHES.fetch_add(1, Ordering::Relaxed) + 1;
    tracing::info!(
        from = ?current,
        to = ?next,
        width = load.width,
        queued = load.queued,
        kv_used = load.kv_used,
        switches,
        "serve objective: class switched"
    );
    true
}

/// Prefill launch width. A packed launch wider than the per-request cap only ever packs several
/// prompts, and every prompt in it finishes when the whole launch does: E4B 1000/128 c64/c128 ran
/// TTFT p50 108/117 ms at the 2048-row cap and 248/261 ms on the full ladder, for +1/+2% tok/s
/// and -10% TTFT p99 (the cold burst).
/// The decode width cannot pick it (c64 is throughput by width), and neither can the prompt rows
/// already seated: widening on four top rungs of them let each closed-loop wave of returning
/// requests (36K rows) re-enter wide and stay synchronized, TTFT p50 550/690 ms. The throughput
/// signals are demand the slots cannot seat: a widest launch of prompt rows queued with every
/// slot taken, or KV pressure. Wide enters at once and narrows after `CALM_MS` with no request
/// waiting at all and KV under the band: a dwell of 0.5 s on "every slot taken" alone flipped 12
/// times in 26 s of a 48 req/s overload as single slots freed and refilled.

static PF_WIDE: AtomicU8 = AtomicU8::new(0);
static PF_SWITCHES: AtomicU64 = AtomicU64::new(0);
static PF_CALM_SINCE: AtomicU64 = AtomicU64::new(NOT_CALM);

/// One tick's prefill signals.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefillLoad {
    /// Prompt rows of the requests waiting while every slot is taken.
    pub overflow_rows: usize,
    /// Requests waiting for a slot.
    pub queued: usize,
    /// The widest compiled prefill launch.
    pub top_rows: usize,
    pub kv_used: f64,
}

fn decide_prefill(wide: bool, calm_since: u64, now: u64, load: PrefillLoad) -> (bool, u64) {
    let PrefillLoad { overflow_rows, queued, top_rows, kv_used } = load;
    if overflow_rows >= top_rows || kv_used >= KV_ENTER {
        (true, NOT_CALM)
    } else if wide && queued == 0 && kv_used < KV_LEAVE {
        let since = if calm_since == NOT_CALM { now } else { calm_since };
        if now.saturating_sub(since) >= CALM_MS {
            (false, NOT_CALM)
        } else {
            (true, since)
        }
    } else {
        (wide, NOT_CALM)
    }
}

/// Whether prefill launches may use the full ladder: fixed by a pinned objective, observed under
/// `auto`.
pub fn wide_prefill() -> bool {
    match objective() {
        Objective::Latency => false,
        Objective::Throughput => true,
        Objective::Auto => PF_WIDE.load(Ordering::Relaxed) != 0,
    }
}

/// Prefill rows per launch while requests decode: the full ladder (`top`) when wide, else the
/// per-request cap (`request`), the widest launch one prompt fills alone.
pub fn prefill_launch_rows(request: usize, top: usize) -> usize {
    if wide_prefill() {
        top
    } else {
        request.min(top)
    }
}

/// One tick's prefill observation. A no-op unless the objective is `auto`; `true` when the width
/// switched.
pub fn observe_prefill(load: PrefillLoad) -> bool {
    if objective() != Objective::Auto || load.top_rows == 0 {
        return false;
    }
    let wide = PF_WIDE.load(Ordering::Relaxed) != 0;
    let now = EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64;
    let (next, calm) = decide_prefill(wide, PF_CALM_SINCE.load(Ordering::Relaxed), now, load);
    PF_CALM_SINCE.store(calm, Ordering::Relaxed);
    if next == wide {
        return false;
    }
    PF_WIDE.store(u8::from(next), Ordering::Relaxed);
    let switches = PF_SWITCHES.fetch_add(1, Ordering::Relaxed) + 1;
    tracing::info!(
        wide = next,
        overflow_rows = load.overflow_rows,
        queued = load.queued,
        top_rows = load.top_rows,
        kv_used = load.kv_used,
        switches,
        "serve objective: prefill width switched"
    );
    true
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

/// Cache-aware admission (`serve::mux::cache_first`): with requests waiting for a slot, seat
/// first the one that attaches the most cached prefix rows. Under latency rules there is no
/// standing queue to reorder.
pub fn cache_aware_admission() -> bool {
    class() == Class::HighConcurrency
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

    fn load(width: usize, queued: usize, kv_used: f64) -> Load {
        Load { width, queued, kv_used }
    }

    #[test]
    fn load_enters_throughput_at_once() {
        for l in [load(WIDE_ENTER, 0, 0.0), load(1, 3, 0.0), load(2, 0, KV_ENTER)] {
            let (c, calm) = decide(Class::Realtime, NOT_CALM, 10, l);
            assert_eq!((c, calm), (Class::HighConcurrency, NOT_CALM), "{l:?}");
        }
    }

    #[test]
    fn kv_pressure_holds_throughput_until_it_drains_below_the_band() {
        let (c, _) = decide(Class::HighConcurrency, 0, 10 * CALM_MS, load(2, 0, 0.8));
        assert_eq!(c, Class::HighConcurrency, "0.8 is inside [{KV_LEAVE}, {KV_ENTER})");
        let (c, calm) = decide(Class::HighConcurrency, NOT_CALM, 1_000, load(2, 0, 0.5));
        assert_eq!((c, calm), (Class::HighConcurrency, 1_000), "calm starts below the band");
        let (c, _) = decide(Class::HighConcurrency, calm, 1_000 + CALM_MS, load(2, 0, 0.5));
        assert_eq!(c, Class::Realtime);
        let (c, _) = decide(Class::Realtime, NOT_CALM, 10, load(2, 0, 0.8));
        assert_eq!(c, Class::Realtime, "the band does not enter throughput either");
    }

    #[test]
    fn throughput_leaves_only_after_the_window_stays_narrow() {
        let (c, calm) = decide(Class::HighConcurrency, NOT_CALM, 1_000, load(2, 0, 0.0));
        assert_eq!((c, calm), (Class::HighConcurrency, 1_000), "calm starts");
        let (c, _) = decide(Class::HighConcurrency, calm, 1_000 + CALM_MS - 1, load(2, 0, 0.0));
        assert_eq!(c, Class::HighConcurrency, "not calm for long enough");
        let (c, _) = decide(Class::HighConcurrency, calm, 1_000 + CALM_MS, load(2, 0, 0.0));
        assert_eq!(c, Class::Realtime);
        let (c, calm) = decide(Class::HighConcurrency, calm, 1_500, load(16, 0, 0.0));
        assert_eq!((c, calm), (Class::HighConcurrency, NOT_CALM), "a wide tick resets calm");
    }

    #[test]
    fn the_band_holds_a_mid_width_workload_where_it_was() {
        for from in [Class::Realtime, Class::HighConcurrency] {
            let (c, _) = decide(from, 0, 10 * CALM_MS, load(6, 0, 0.0));
            assert_eq!(c, from, "6 rows is inside [{WIDE_LEAVE}, {WIDE_ENTER})");
        }
    }

    fn pf(overflow_rows: usize, queued: usize, kv_used: f64) -> PrefillLoad {
        PrefillLoad { overflow_rows, queued, top_rows: 8192, kv_used }
    }

    #[test]
    fn overflow_of_one_widest_launch_or_kv_pressure_widens_prefill_at_once() {
        for l in [pf(8192, 8, 0.0), pf(0, 0, KV_ENTER)] {
            assert_eq!(decide_prefill(false, NOT_CALM, 10, l), (true, NOT_CALM), "{l:?}");
        }
        assert_eq!(decide_prefill(false, NOT_CALM, 10, pf(8191, 8, 0.0)), (false, NOT_CALM));
        assert_eq!(decide_prefill(false, NOT_CALM, 10, pf(0, 64, 0.0)).0, false, "a rung hold is not overflow");
    }

    #[test]
    fn wide_prefill_narrows_only_after_no_request_waits() {
        assert_eq!(decide_prefill(true, NOT_CALM, 10 * CALM_MS, pf(0, 1, 0.0)).0, true, "a queue below overflow");
        assert_eq!(decide_prefill(true, NOT_CALM, 10 * CALM_MS, pf(0, 0, 0.8)).0, true, "KV band");
        let (w, calm) = decide_prefill(true, NOT_CALM, 1_000, pf(0, 0, 0.0));
        assert_eq!((w, calm), (true, 1_000), "calm starts");
        assert_eq!(decide_prefill(true, calm, 1_000 + CALM_MS - 1, pf(0, 0, 0.0)).0, true);
        assert_eq!(decide_prefill(true, calm, 1_000 + CALM_MS, pf(0, 0, 0.0)), (false, NOT_CALM));
        assert_eq!(decide_prefill(true, calm, 1_200, pf(0, 1, 0.0)), (true, NOT_CALM), "a waiter resets calm");
        assert_eq!(decide_prefill(false, NOT_CALM, 10, pf(4_000, 4, 0.8)).0, false, "the bands do not widen");
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

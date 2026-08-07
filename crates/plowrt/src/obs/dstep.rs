//! §DSTEP — where a DECODE token's wall clock goes, host phase by host phase.
//!
//! `PLOW_DSTEP_LOG=1`; off by default and every call site is behind [`on`], a
//! cached `OnceLock` load, so a serving build pays nothing.
//!
//! # Why this is separate from [`crate::obs::ttft`]
//!
//! TTFT measures ONE interval once per request and is reset by the arriving
//! request. Decode is the steady state: thousands of identical ticks, and the
//! question is not "where did this one go" but "what fraction of the mean token
//! is host work that a pipelined submit could hide". So the counters accumulate
//! over a WINDOW of tokens and dump a mean, rather than being reset per request.
//!
//! # The question it exists to answer
//!
//! A TP decode tick is
//!
//! ```text
//! seed ids -> prepare every rank -> re-arm every rank -> zero xctr
//!          -> enqueue every rank -> DRAIN every rank
//!          -> audit xctr -> read every rank's id -> agree -> detok + stream
//! ```
//!
//! Only `DRAIN` is GPU time. Everything before it is host work the GPU waits
//! for, and everything after it is host work that waits for the GPU. If the
//! non-`DRAIN` rows sum to a percent of the token, pipelining `submit`/`complete`
//! buys a percent and the honest answer is to change nothing; if they sum to a
//! fifth, the split API in `exec::amd_tp` is worth wiring up. That number is not
//! guessable from the code — `zero_xctr` alone is a copy-engine pass over
//! `n_gpu · n_xctr · 128 B` and was measured at ~32 µs/token at TP4 — so it is
//! measured.
//!
//! # Which side of the drain a phase is on
//!
//! The label prefix says it, because that is what decides whether pipelining can
//! hide it:
//!
//! * `pre `  — before the dispatch. Hideable ONLY if it touches no buffer the
//!   resident tick reads; `xctr` zeroing and the enqueue itself never are.
//! * `GPU `  — the drain. Not host work at all.
//! * `post`  — after the dispatch. Hideable: it feeds the client, not the device.

use std::sync::atomic::{AtomicU64, Ordering};

pub use crate::obs::ttft::Phase;

/// Whether decode-step logging is active (`--dstep-log` / `PLOW_DSTEP_LOG=1`).
/// Reads from [`RuntimeConfig::get`](crate::config::RuntimeConfig::get).
pub fn on() -> bool {
    crate::config::RuntimeConfig::get().dstep_log
}

/// Tokens per dump (`--rt-dstep-every` / `PLOW_DSTEP_EVERY`), default 64.
fn every() -> u64 {
    crate::config::RuntimeConfig::get()
        .dstep_every
        .map(|n| n as u64)
        .filter(|&n| n > 0)
        .unwrap_or(64)
}

// --- before the dispatch -----------------------------------------------------

/// `seed_ids` on every rank — one 4 B H2D each.
pub static SEED: Phase = Phase::new("pre  seed_ids (H2D in.ids, x ranks)");
/// `decode_prepare` on every rank: `patch_kvrow` + the `pos`/`kvlen` scalars.
pub static PREPARE: Phase = Phase::new("pre  decode_prepare (kvrow patch + scalars)");
/// `rearm_prog` on every rank — local counter/cursor zeroing.
pub static REARM: Phase = Phase::new("pre  rearm_prog (local counters)");
/// `zero_xctr` across the whole group. LIVE during the tick; cannot be hoisted.
pub static XCTR: Phase = Phase::new("pre  zero_xctr (cross-GPU gates, all ranks)");
/// The N AQL launches. Must follow the drain; cannot be hoisted.
pub static ENQUEUE: Phase = Phase::new("pre  enqueue (AQL launch x ranks)");

// --- the dispatch ------------------------------------------------------------

/// The all-rank drain. This is the GPU tick.
pub static DRAIN: Phase = Phase::new("GPU  drain (all ranks)");

// --- after the dispatch ------------------------------------------------------

/// `audit_xctr` — a 12 KiB D2H per rank, gated on `AmdTpGroup::audit`.
pub static AUDIT: Phase = Phase::new("post audit_xctr (12 KiB D2H x ranks)");
/// `read_sampled` on every rank — 4 B D2H each. Rank 0's is the token; the rest
/// are the cross-rank audit.
pub static READ: Phase = Phase::new("post read_sampled (4 B D2H x ranks)");
/// `AmdTpGroup::agree` — pure host compare.
pub static AGREE: Phase = Phase::new("post agree (cross-rank compare)");
/// Detokenise + stop check + SSE frame + channel send, per produced token.
pub static STREAM: Phase = Phase::new("post detok + stop + SSE send");

/// The whole tick as the mux sees it, drain included. The denominator.
pub static TOKEN: Phase = Phase::new("TOKEN TOTAL");

const PHASES: &[&Phase] = &[
    &SEED, &PREPARE, &REARM, &XCTR, &ENQUEUE, &DRAIN, &AUDIT, &READ, &AGREE, &STREAM,
];

/// Tokens counted into the current window.
static WINDOW: AtomicU64 = AtomicU64::new(0);

/// Close out one decode token: add its total and dump the window if it is full.
///
/// Called from the ONE place that owns a whole tick, so `TOKEN` is a real total
/// and not a sum of parts that would hide whatever is between them.
#[inline]
pub fn token(total_ns: u64) {
    if !on() {
        return;
    }
    TOKEN.add(total_ns);
    if WINDOW.fetch_add(1, Ordering::Relaxed) + 1 >= every() {
        dump();
    }
}

/// Time `f` into `p`. Returns `f`'s value.
#[inline]
pub fn timed<T>(p: &Phase, f: impl FnOnce() -> T) -> T {
    if !on() {
        return f();
    }
    let t = std::time::Instant::now();
    let out = f();
    p.add(t.elapsed().as_nanos() as u64);
    out
}

/// Emit the window's mean breakdown and start a new window.
fn dump() {
    let n = WINDOW.swap(0, Ordering::Relaxed).max(1);
    let (tot_ns, _) = TOKEN.read();
    let per = |ns: u64| ns as f64 / n as f64 / 1e3; // µs per token
    let tok_us = per(tot_ns);
    let mut out = format!(
        "\nDECODE STEP BREAKDOWN  n={n} tokens  mean={tok_us:.1} µs/token \
         ({:.1} tok/s)\n{:<46} {:>10} {:>8} {:>7}\n",
        1e6 / tok_us.max(1e-9),
        "phase",
        "µs/tok",
        "calls/tok",
        "%",
    );
    let mut host = 0u64;
    for p in PHASES {
        let (ns, calls) = p.read();
        if !p.label.starts_with("GPU") {
            host += ns;
        }
        out.push_str(&format!(
            "{:<46} {:>10.2} {:>8.1} {:>6.1}%\n",
            p.label,
            per(ns),
            calls as f64 / n as f64,
            100.0 * ns as f64 / tot_ns.max(1) as f64,
        ));
    }
    // The line the pipelining decision is made on.
    out.push_str(&format!(
        "{:<46} {:>10.2} {:>8} {:>6.1}%\n",
        "HOST TOTAL (everything but the drain)",
        per(host),
        "",
        100.0 * host as f64 / tot_ns.max(1) as f64,
    ));
    out.push_str(&format!(
        "{:<46} {:>10.2} {:>8} {:>6.1}%\n",
        "UNATTRIBUTED (mux tick, locks, scheduler)",
        per(tot_ns.saturating_sub(host + DRAIN.read().0)),
        "",
        100.0 * tot_ns.saturating_sub(host + DRAIN.read().0) as f64 / tot_ns.max(1) as f64,
    ));
    eprint!("{out}");
    for p in PHASES {
        p.reset();
    }
    TOKEN.reset();
}

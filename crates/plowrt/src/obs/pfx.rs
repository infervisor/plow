//! Prefix-cache timing: what the state snapshot and restore actually cost.
//!
//! The prefix cache removes prefill tokens and adds a device-to-device copy of one slot's carried
//! recurrent state (MEASURED: 56 MiB, 276 tensors, per rank). On the batched benchmark it removed
//! 75% of prefill tokens for +7.1% throughput, and the gap between those two numbers was INFERRED
//! to be this copy — which is exactly the kind of inference `perf-data/k3-hier2-ceiling.md` §4
//! says to replace with a measurement before optimising against it. This is that measurement.
//!
//! `PLOW_PFX_LOG=1` prints the totals at exit.

use std::sync::atomic::Ordering;

pub use super::ttft::Phase;

crate::env_flag!(pub fn on, "PLOW_PFX_LOG");

pub static SNAP: Phase = Phase::new("prefix snapshot (dtod, per rank)");
pub static RESTORE: Phase = Phase::new("prefix restore  (dtod, per rank)");

/// Print the totals. Called from the same place the other observers report.
pub fn report() {
    if !on() {
        return;
    }
    for p in [&SNAP, &RESTORE] {
        let (ns, n) = p.read();
        if n == 0 {
            continue;
        }
        tracing::info!(
            phase = p.label,
            calls = n,
            total_ms = format_args!("{:.2}", ns as f64 / 1e6),
            mean_us = format_args!("{:.1}", ns as f64 / 1e3 / n as f64),
            "PFX"
        );
    }
    let _ = Ordering::Relaxed;
}

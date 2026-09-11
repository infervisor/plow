//! §TICK — one line per AMD serve tick: prefill launches, decode dispatch, and the host
//! remainder, so a per-tick wall time can be split into its GPU-visible parts.
//!
//! `PLOW_TICK_LOG=1`; off by default and every call site is behind [`on`]. Chunk-level detail
//! (`PFCHUNK`, `PFSEG`) prints from the engine as the chunk runs; the `TICK` line prints when
//! the tick body returns, from the guard's drop, so every exit path of the tick is covered.

use std::cell::Cell;
use std::time::Instant;

pub fn on() -> bool {
    crate::config::RuntimeConfig::get().tick_log
}

thread_local! {
    static SEQ: Cell<u64> = const { Cell::new(0) };
    static LAST_END: Cell<Option<Instant>> = const { Cell::new(None) };
    static PF_NS: Cell<u64> = const { Cell::new(0) };
    static PF_LAUNCHES: Cell<u32> = const { Cell::new(0) };
    static PF_ROWS: Cell<u32> = const { Cell::new(0) };
    static DEC_NS: Cell<u64> = const { Cell::new(0) };
    static DEC_ROWS: Cell<u32> = const { Cell::new(0) };
}

pub struct TickGuard {
    started: Instant,
    idle_ns: u64,
}

pub fn begin() -> Option<TickGuard> {
    if !on() {
        return None;
    }
    let started = Instant::now();
    let idle_ns = LAST_END.with(|l| {
        l.get()
            .map(|e| started.duration_since(e).as_nanos() as u64)
            .unwrap_or(0)
    });
    PF_NS.with(|c| c.set(0));
    PF_LAUNCHES.with(|c| c.set(0));
    PF_ROWS.with(|c| c.set(0));
    DEC_NS.with(|c| c.set(0));
    DEC_ROWS.with(|c| c.set(0));
    Some(TickGuard { started, idle_ns })
}

/// One isolated or packed prefill launch of `rows` real rows took `ns`.
#[inline]
pub fn prefill(ns: u64, rows: u32) {
    if !on() {
        return;
    }
    PF_NS.with(|c| c.set(c.get() + ns));
    PF_LAUNCHES.with(|c| c.set(c.get() + 1));
    PF_ROWS.with(|c| c.set(c.get() + rows));
}

/// The decode dispatch of `rows` fed slots took `ns`.
#[inline]
pub fn decode(ns: u64, rows: u32) {
    if !on() {
        return;
    }
    DEC_NS.with(|c| c.set(c.get() + ns));
    DEC_ROWS.with(|c| c.set(rows));
}

impl Drop for TickGuard {
    fn drop(&mut self) {
        let end = Instant::now();
        let total = end.duration_since(self.started).as_nanos() as u64;
        let seq = SEQ.with(|s| {
            let n = s.get() + 1;
            s.set(n);
            n
        });
        let pf_ns = PF_NS.with(Cell::get);
        let dec_ns = DEC_NS.with(Cell::get);
        let ms = |ns: u64| ns as f64 / 1e6;
        eprintln!(
            "TICK n={seq} total={:.3} pf_launches={} pf_rows={} pf={:.3} dec_rows={} dec={:.3} other={:.3} idle_before={:.3}",
            ms(total),
            PF_LAUNCHES.with(Cell::get),
            PF_ROWS.with(Cell::get),
            ms(pf_ns),
            DEC_ROWS.with(Cell::get),
            ms(dec_ns),
            ms(total.saturating_sub(pf_ns + dec_ns)),
            ms(self.idle_ns),
        );
        LAST_END.with(|l| l.set(Some(end)));
    }
}

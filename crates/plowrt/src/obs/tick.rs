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
    static DEC_VMM_NS: Cell<u64> = const { Cell::new(0) };
    static DEC_MAPS: Cell<u64> = const { Cell::new(0) };
    static DEC_SEGS: Cell<u32> = const { Cell::new(0) };
    static DEC_INFLIGHT_ENQ: Cell<i64> = const { Cell::new(-1) };
    static DEC_INFLIGHT_SUB: Cell<i64> = const { Cell::new(-1) };
    static PREP_VMM_NS: Cell<u64> = const { Cell::new(0) };
    static PREP_PATCH_NS: Cell<u64> = const { Cell::new(0) };
    static PUB_FILL_NS: Cell<u64> = const { Cell::new(0) };
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
    DEC_VMM_NS.with(|c| c.set(0));
    DEC_MAPS.with(|c| c.set(0));
    DEC_SEGS.with(|c| c.set(0));
    DEC_INFLIGHT_ENQ.with(|c| c.set(-1));
    DEC_INFLIGHT_SUB.with(|c| c.set(-1));
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

/// The decode's synchronous KV block mapping (`vmm_ensure`, all ranks) took `ns` and made
/// `maps` driver mappings.
#[inline]
pub fn decode_vmm(ns: u64, maps: u64) {
    if !on() {
        return;
    }
    DEC_VMM_NS.with(|c| c.set(c.get() + ns));
    DEC_MAPS.with(|c| c.set(c.get() + maps));
}

/// Rank 0's dispatches still in flight right after the decode's enqueue and after its
/// post-launch re-arm, for a program of `segs` segments. Near `segs` = the GPU is far behind
/// the host, so the enqueue and re-arm overlap it; near 0 = the GPU waited on the host.
#[inline]
pub fn decode_in_flight(after_enqueue: i64, after_rearm: i64, segs: u32) {
    if !on() {
        return;
    }
    DEC_INFLIGHT_ENQ.with(|c| c.set(after_enqueue));
    DEC_INFLIGHT_SUB.with(|c| c.set(after_rearm));
    DEC_SEGS.with(|c| c.set(segs));
}

/// One rank's `prefill_prepare` spent `ns` in `vmm_ensure`.
#[inline]
pub fn prepare_vmm(ns: u64) {
    if on() {
        PREP_VMM_NS.with(|c| c.set(c.get() + ns));
    }
}

/// One rank's `prefill_prepare` spent `ns` patching and uploading its program.
#[inline]
pub fn prepare_patch(ns: u64) {
    if on() {
        PREP_PATCH_NS.with(|c| c.set(c.get() + ns));
    }
}

/// `(vmm_ns, patch_ns)` accumulated since the last call, all ranks; resets both.
pub fn take_prepare() -> (u64, u64) {
    (PREP_VMM_NS.with(|c| c.replace(0)), PREP_PATCH_NS.with(|c| c.replace(0)))
}

/// A prefix publish spent `ns` copying its snapshot (one rank, one cache group).
#[inline]
pub fn publish_fill(ns: u64) {
    if on() {
        PUB_FILL_NS.with(|c| c.set(c.get() + ns));
    }
}

/// Snapshot-copy time accumulated since the last call; resets it.
pub fn take_publish_fill() -> u64 {
    PUB_FILL_NS.with(|c| c.replace(0))
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
            "TICK n={seq} total={:.3} pf_launches={} pf_rows={} pf={:.3} dec_rows={} dec={:.3} dec_vmm={:.3} dec_maps={} dec_segs={} dec_inflight_enq={} dec_inflight_sub={} other={:.3} idle_before={:.3}",
            ms(total),
            PF_LAUNCHES.with(Cell::get),
            PF_ROWS.with(Cell::get),
            ms(pf_ns),
            DEC_ROWS.with(Cell::get),
            ms(dec_ns),
            ms(DEC_VMM_NS.with(Cell::get)),
            DEC_MAPS.with(Cell::get),
            DEC_SEGS.with(Cell::get),
            DEC_INFLIGHT_ENQ.with(Cell::get),
            DEC_INFLIGHT_SUB.with(Cell::get),
            ms(total.saturating_sub(pf_ns + dec_ns)),
            ms(self.idle_ns),
        );
        LAST_END.with(|l| l.set(Some(end)));
    }
}

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
    static PUB_DEFERRED_NS: Cell<u64> = const { Cell::new(0) };
    static DEC_RUNG: Cell<u32> = const { Cell::new(0) };
    static DEC_DX: Cell<u32> = const { Cell::new(0) };
    static DEC_DSA: Cell<u32> = const { Cell::new(0) };
    static DX_TOTAL: Cell<u64> = const { Cell::new(0) };
    static DSA_TOTAL: Cell<u64> = const { Cell::new(0) };
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
    PUB_DEFERRED_NS.with(|c| c.set(0));
    DEC_RUNG.with(|c| c.set(0));
    DEC_DX.with(|c| c.set(0));
    DEC_DSA.with(|c| c.set(0));
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

/// One decode dispatch launched rung `rung` as its dense-exact twin (`PLOW_AMD_DECODE_DENSE_EXACT`)
/// or as the rung program itself.
#[inline]
pub fn decode_prog(rung: u32, dense_exact: bool) {
    if !on() {
        return;
    }
    DEC_RUNG.with(|c| c.set(rung));
    let (tick, total) = if dense_exact {
        (&DEC_DX, &DX_TOTAL)
    } else {
        (&DEC_DSA, &DSA_TOTAL)
    };
    tick.with(|c| c.set(c.get() + 1));
    total.with(|c| c.set(c.get() + 1));
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
    (
        PREP_VMM_NS.with(|c| c.replace(0)),
        PREP_PATCH_NS.with(|c| c.replace(0)),
    )
}

/// A prefix publish spent `ns` copying its snapshot (one rank, one cache group).
#[inline]
pub fn publish_fill(ns: u64) {
    if on() {
        PUB_FILL_NS.with(|c| c.set(c.get() + ns));
    }
}

/// A deferred prefix publish ran for `ns` under a GPU drain window (all ranks).
#[inline]
pub fn deferred_publish(ns: u64) {
    if on() {
        PUB_DEFERRED_NS.with(|c| c.set(c.get() + ns));
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
            "TICK n={seq} total={:.3} pf_launches={} pf_rows={} pf={:.3} dec_rows={} dec={:.3} dec_vmm={:.3} dec_maps={} dec_segs={} dec_inflight_enq={} dec_inflight_sub={} pub_deferred={:.3} other={:.3} idle_before={:.3} dec_rung={} dec_dx={} dec_dsa={}",
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
            ms(PUB_DEFERRED_NS.with(Cell::get)),
            ms(total.saturating_sub(pf_ns + dec_ns)),
            ms(self.idle_ns),
            DEC_RUNG.with(Cell::get),
            DEC_DX.with(Cell::get),
            DEC_DSA.with(Cell::get),
        );
        if seq % 1000 == 0 {
            eprintln!(
                "TICKDX n={seq} dx={} dsa={}",
                DX_TOTAL.with(Cell::get),
                DSA_TOTAL.with(Cell::get)
            );
        }
        LAST_END.with(|l| l.set(Some(end)));
    }
}

// §TICKTRACE — `PLOW_TICK_TRACE=<path>`: one JSON line per AMD serve tick, for
// `scripts/bench/tick_trace_report.py`. Times are µs on one monotonic clock (process epoch).
// Every hook is a no-op unless a traced tick is open on the calling thread.

use std::cell::RefCell;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

struct TraceLaunch {
    kind: &'static str,
    t0: u64,
    t1: u64,
    enq: u64,
    dr0: u64,
    dr1: u64,
    drain: u64,
    skew: u64,
    slow_rank: u32,
    rows: u32,
    rung: u32,
    spans: u32,
    dec: u32,
}

#[derive(Default)]
struct TraceRec {
    open: bool,
    t0: u64,
    ready_dec: u32,
    waiting: u32,
    waiting_rows: u32,
    held: bool,
    hits: u32,
    launches: Vec<TraceLaunch>,
    first: Vec<(i64, u64)>,
    dec: Vec<(i64, u64)>,
}

thread_local! {
    static REC: RefCell<TraceRec> = RefCell::new(TraceRec::default());
    static TRACE_SEQ: Cell<u64> = const { Cell::new(0) };
}

fn trace_epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

type Sink = Mutex<std::io::BufWriter<std::fs::File>>;

fn trace_sink() -> Option<&'static Sink> {
    static SINK: OnceLock<Option<Sink>> = OnceLock::new();
    SINK.get_or_init(|| {
        let path = crate::config::RuntimeConfig::get().tick_trace.as_ref()?;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(f) => Some(Mutex::new(std::io::BufWriter::new(f))),
            Err(err) => {
                tracing::warn!(path = %path.display(), error = %err, "PLOW_TICK_TRACE: cannot open");
                None
            }
        }
    })
    .as_ref()
}

#[inline]
pub fn trace_on() -> bool {
    crate::config::RuntimeConfig::get().tick_trace.is_some()
}

#[inline]
fn now_us() -> u64 {
    trace_epoch().elapsed().as_nanos() as u64 / 1000
}

#[inline]
fn us_of(t: Instant) -> u64 {
    t.saturating_duration_since(trace_epoch()).as_nanos() as u64 / 1000
}

#[inline]
fn with_open(f: impl FnOnce(&mut TraceRec)) {
    REC.with(|r| {
        let mut r = r.borrow_mut();
        if r.open {
            f(&mut r)
        }
    })
}

pub struct TraceGuard(());

/// Open one traced tick: `ready_dec` live decode rows, `waiting` prompts with `waiting_rows`
/// prompt rows not yet prefilled.
pub fn trace_begin(ready_dec: u32, waiting: u32, waiting_rows: u32) -> Option<TraceGuard> {
    if !trace_on() || trace_sink().is_none() {
        return None;
    }
    let t0 = now_us();
    REC.with(|r| {
        let mut r = r.borrow_mut();
        r.open = true;
        r.t0 = t0;
        r.ready_dec = ready_dec;
        r.waiting = waiting;
        r.waiting_rows = waiting_rows;
        r.held = false;
        r.hits = 0;
        r.launches.clear();
        r.first.clear();
        r.dec.clear();
    });
    Some(TraceGuard(()))
}

/// The tick held its fresh prompts back (`PLOW_PF_HOLD`).
pub fn trace_held() {
    with_open(|r| r.held = true);
}

/// `rows` prompt rows came from the prefix cache instead of a launch.
pub fn trace_hit(rows: usize) {
    if rows > 0 && trace_on() {
        with_open(|r| r.hits = r.hits.saturating_add(rows as u32));
    }
}

/// An engine launch of `kind` (`packed`, `mixed`, `lone`, `decode`) starts now.
pub fn trace_launch(kind: &'static str) {
    if !trace_on() {
        return;
    }
    with_open(|r| {
        r.launches.push(TraceLaunch {
            kind,
            t0: now_us(),
            t1: 0,
            enq: 0,
            dr0: 0,
            dr1: 0,
            drain: 0,
            skew: 0,
            slow_rank: 0,
            rows: 0,
            rung: 0,
            spans: 0,
            dec: 0,
        })
    });
}

/// The open launch returned: `rows` real prefill rows in `spans` spans, `dec` decode rows.
/// `kind` overrides the kind given at start (a mixed launch whose riders were refused).
pub fn trace_launch_done(kind: Option<&'static str>, rows: u32, spans: u32, dec: u32) {
    if !trace_on() {
        return;
    }
    with_open(|r| {
        if let Some(l) = r.launches.last_mut() {
            l.t1 = now_us();
            l.rows = rows;
            l.spans = spans;
            l.dec = dec;
            if let Some(kind) = kind {
                l.kind = kind;
            }
        }
    });
}

/// The engine picked a program of `rung` rows for the open launch.
#[inline]
pub fn trace_rung(rung: u32) {
    if trace_on() {
        with_open(|r| {
            if let Some(l) = r.launches.last_mut() {
                l.rung = l.rung.max(rung);
            }
        });
    }
}

/// Every rank's work for the open launch is enqueued (first call wins).
#[inline]
pub fn trace_enqueued() {
    if trace_on() {
        with_open(|r| {
            if let Some(l) = r.launches.last_mut().filter(|l| l.enq == 0) {
                l.enq = now_us();
            }
        });
    }
}

/// A host drain of every rank began at `start`; `rank_done[i]` is when rank `i`'s drain
/// returned. Ranks drain in order, so a late rank shows as the largest step.
pub fn trace_drain(start: Instant, rank_done: &[Instant]) {
    with_open(|r| {
        let Some(l) = r.launches.last_mut() else {
            return;
        };
        let end = rank_done.last().copied().unwrap_or_else(Instant::now);
        if l.dr0 == 0 {
            l.dr0 = us_of(start);
        }
        l.dr1 = us_of(end);
        l.drain += end.saturating_duration_since(start).as_nanos() as u64 / 1000;
        let mut prev = start;
        let (mut worst, mut worst_rank) = (0u64, 0u32);
        for (i, &t) in rank_done.iter().enumerate() {
            let step = t.saturating_duration_since(prev).as_nanos() as u64 / 1000;
            if i > 0 && step > worst {
                worst = step;
                worst_rank = i as u32;
            }
            prev = t;
        }
        if worst > l.skew {
            l.skew = worst;
            l.slow_rank = worst_rank;
        }
    });
}

/// The request that arrived at `arrived` emitted a token now: its first (`first`) or a decode
/// token. The request key is its arrival in ns from the trace epoch (may be negative).
#[inline]
pub fn trace_token(arrived: Instant, first: bool) {
    if trace_on() {
        with_open(|r| {
            let epoch = trace_epoch();
            let key = if arrived >= epoch {
                arrived.duration_since(epoch).as_nanos() as i64
            } else {
                -(epoch.duration_since(arrived).as_nanos() as i64)
            };
            let entry = (key, now_us());
            if first {
                r.first.push(entry)
            } else {
                r.dec.push(entry)
            }
        });
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        use std::fmt::Write as _;
        let t1 = now_us();
        let seq = TRACE_SEQ.with(|s| {
            let n = s.get() + 1;
            s.set(n);
            n
        });
        let line = REC.with(|r| {
            let mut r = r.borrow_mut();
            r.open = false;
            let mut out = String::with_capacity(256 + 200 * r.launches.len() + 32 * r.dec.len());
            let tid = std::thread::current().id();
            let unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0.0, |d| d.as_secs_f64());
            let _ = write!(
                out,
                "{{\"n\":{seq},\"th\":\"{tid:?}\",\"unix\":{unix:.6},\"t0\":{},\"t1\":{t1},\"ready_dec\":{},\"waiting\":{},\"waiting_rows\":{},\"held\":{},\"hits\":{},\"launches\":[",
                r.t0, r.ready_dec, r.waiting, r.waiting_rows, r.held, r.hits
            );
            for (i, l) in r.launches.iter().enumerate() {
                let _ = write!(
                    out,
                    "{}{{\"kind\":\"{}\",\"t0\":{},\"t1\":{},\"enq\":{},\"dr0\":{},\"dr1\":{},\"drain\":{},\"skew\":{},\"slow_rank\":{},\"rows\":{},\"rung\":{},\"spans\":{},\"dec\":{}}}",
                    if i > 0 { "," } else { "" },
                    l.kind,
                    l.t0,
                    l.t1,
                    l.enq,
                    l.dr0,
                    l.dr1,
                    l.drain,
                    l.skew,
                    l.slow_rank,
                    l.rows,
                    l.rung,
                    l.spans,
                    l.dec
                );
            }
            let list = |v: &[(i64, u64)]| {
                v.iter()
                    .map(|(k, t)| format!("[{k},{t}]"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let _ = writeln!(
                out,
                "],\"first\":[{}],\"dec\":[{}]}}",
                list(&r.first),
                list(&r.dec)
            );
            out
        });
        if let Some(sink) = trace_sink() {
            if let Ok(mut w) = sink.lock() {
                let _ = w.write_all(line.as_bytes());
                if seq % 16 == 0 {
                    let _ = w.flush();
                }
            }
        }
    }
}

//! Counter-gated device-ISA interpreter loop — the CPU twin of `interp.hip`'s
//! per-workgroup walk, run by every persistent worker:
//!
//! ```text
//! for each entry I own (static stream) / claim (global queue):
//!     wait until every wait-counter >= its threshold
//!     exec(inst, slice)
//!     bump every successor counter (Release)
//! ```
//!
//! Two modes, both straight from the blob:
//! * **Static**: virtual executor `cu` streams are owned by workers. A worker
//!   sweeps its owned streams; a blocked head just means "try the next stream",
//!   so one thread owning many streams cannot deadlock (the GPU's co-residency
//!   condition becomes: every stream has a live owner).
//! * **Global queue** (`PLOW_BLOB_F_GQ`): op-major `gq_stream` windowed by
//!   `(seg, domain)`; one atomic cursor per window is the NUMA-local packet
//!   queue, other domains' windows are the steal targets. A claimed entry is
//!   always executed (its producers precede it in op-major order and are
//!   claimed by live threads), so waiting on it is deadlock-free.
//!
//! Kernel execution is behind [`Exec`] so the loop is testable with a mock.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crossbeam_utils::CachePadded;
use packet::dev::{DevInst64, StreamEnt, Wait};
use parking_lot::Mutex;

use crate::exec::counters::CounterPool;
use crate::exec::cpu::control::Feedback;

/// Opt-in per-packet trace (profiling). Off: one relaxed load per packet. On: each
/// worker records `(inst, slice, worker, t0, t1)` into a thread-local buffer and
/// flushes it into [`trace_take`]'s sink when its run drains — no lock on the hot path.
pub static TRACE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[derive(Clone, Copy, Debug)]
pub struct TraceEv {
    pub inst: u32,
    pub slice: u32,
    pub worker: u16,
    /// Nanoseconds since [`trace_begin`].
    pub t0_ns: u64,
    pub t1_ns: u64,
}

static TRACE_EPOCH: Mutex<Option<std::time::Instant>> = Mutex::new(None);
static TRACE_SINK: Mutex<Vec<TraceEv>> = Mutex::new(Vec::new());
thread_local! {
    static TRACE_BUF: std::cell::RefCell<Vec<TraceEv>> = const { std::cell::RefCell::new(Vec::new()) };
    static TRACE_T0: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
}

/// Start recording; clears any previous trace.
pub fn trace_begin() {
    *TRACE_EPOCH.lock() = Some(std::time::Instant::now());
    TRACE_SINK.lock().clear();
    TRACE_ON.store(true, Ordering::SeqCst);
}

/// Stop recording and take everything the workers flushed (call after `wait_done`).
pub fn trace_take() -> Vec<TraceEv> {
    TRACE_ON.store(false, Ordering::SeqCst);
    std::mem::take(&mut *TRACE_SINK.lock())
}

#[inline]
fn trace_now_ns() -> u64 {
    let epoch = TRACE_T0.with(|c| {
        if let Some(t) = c.get() {
            t
        } else {
            let t = TRACE_EPOCH.lock().unwrap_or_else(std::time::Instant::now);
            c.set(Some(t));
            t
        }
    });
    epoch.elapsed().as_nanos() as u64
}

/// Worker: hand this thread's events to the sink (end of a traced run).
pub fn trace_flush() {
    TRACE_T0.with(|c| c.set(None));
    let evs: Vec<TraceEv> = TRACE_BUF.with(|b| std::mem::take(&mut *b.borrow_mut()));
    if !evs.is_empty() {
        TRACE_SINK.lock().extend(evs);
    }
}

/// Per-window cursors of the global queue. `[n_seg * domains]`.
#[derive(Clone)]
pub struct GlobalQueue {
    /// Op-major (topological) permutation of the stream entries.
    pub stream: Vec<StreamEnt>,
    /// `[n_seg * domains + 1]`: window `seg * domains + d` is
    /// `stream[seg_ofs[w]..seg_ofs[w+1]]`.
    pub seg_ofs: Vec<u32>,
    pub domains: u32,
}

/// A program as the interpreter sees it: the blob's tables copied into host
/// memory once at load. Pointer-free, so it is `Send + Sync` for free.
#[derive(Clone)]
pub struct LoadedProgram {
    pub insts: Vec<DevInst64>,
    /// Flattened per-cu streams, `stream[stream_ofs[cu] .. +stream_len[cu]]`.
    pub stream: Vec<StreamEnt>,
    pub stream_ofs: Vec<u32>,
    pub stream_len: Vec<u32>,
    pub waits: Vec<Wait>,
    pub succs: Vec<u32>,
    pub n_cu: u32,
    /// Wave-class segments; 0/1 = unsegmented.
    pub n_seg: u32,
    /// Static-mode per-(cu, seg) windows `[n_cu][n_seg + 1]`, relative to
    /// `stream_ofs[cu]`; `None` ⇒ filter on `StreamEnt::seg`.
    pub seg_ofs: Option<Vec<u32>>,
    pub gq: Option<GlobalQueue>,
    /// Cu ids each worker owns FOR THIS PROGRAM, indexed by worker, or `None` to keep the pool's
    /// spawn-time ownership. Prefill and decode want different widths (prefill is compute-bound and
    /// wants one worker per physical core; a dense single-row decode streams weights and wants the
    /// SMT siblings), so the narrowing is per program and the tail workers simply own nothing.
    /// `None` matters: a pool with fewer threads than cus must still cover every stream.
    pub cus_of: Option<Vec<Vec<u32>>>,
}

impl LoadedProgram {
    #[inline]
    pub fn n_seg(&self) -> u32 {
        self.n_seg.max(1)
    }

    /// Windows the global queue has (`n_seg * domains`), 0 without GQ.
    pub fn gq_windows(&self) -> usize {
        self.gq
            .as_ref()
            .map(|g| self.n_seg() as usize * g.domains as usize)
            .unwrap_or(0)
    }

    #[inline]
    fn cu_range(&self, cu: u32, seg: u32) -> (u32, u32) {
        let base = self.stream_ofs[cu as usize];
        match &self.seg_ofs {
            Some(so) => {
                let row = cu as usize * (self.n_seg() as usize + 1) + seg as usize;
                (base + so[row], base + so[row + 1])
            }
            None => (base, base + self.stream_len[cu as usize]),
        }
    }

    #[inline]
    fn waits_of(&self, e: &StreamEnt) -> &[Wait] {
        &self.waits[e.wait_ofs as usize..e.wait_ofs as usize + e.wait_len as usize]
    }

    #[inline]
    fn succs_of(&self, e: &StreamEnt) -> &[u32] {
        &self.succs[e.succ_ofs as usize..e.succ_ofs as usize + e.succ_len as usize]
    }
}

/// Identity of the calling worker, handed to every kernel.
#[derive(Clone, Copy, Debug)]
pub struct WorkerCtx {
    pub worker: u32,
    pub node: u32,
    pub cpu: u32,
    /// GQ locality domain this worker claims from first.
    pub domain: u32,
}

/// Kernel execution. Implemented over the C library by `ffi`; by a recorder in
/// tests. Must not block on other packets.
pub trait Exec: Send + Sync {
    fn exec(&self, inst: &DevInst64, slice: u32, nblk: u32, worker: &WorkerCtx);
}

/// Spin → yield → park, shared by every wait in the pool. Parking is per node
/// so a wake touches only the sleepers that can make progress.
pub struct Parker {
    sleepers: CachePadded<AtomicU32>,
    threads: Mutex<Vec<std::thread::Thread>>,
}

impl Default for Parker {
    fn default() -> Self {
        Parker::new()
    }
}

impl Parker {
    pub const fn new() -> Parker {
        Parker {
            sleepers: CachePadded::new(AtomicU32::new(0)),
            threads: Mutex::new(Vec::new()),
        }
    }

    /// Register the calling thread (once, at worker start).
    pub fn register(&self) {
        self.threads.lock().push(std::thread::current());
    }

    #[inline]
    pub fn has_sleepers(&self) -> bool {
        self.sleepers.load(Ordering::Relaxed) != 0
    }

    /// Wake every parked registrant. Cold: only called when `has_sleepers`.
    pub fn unpark_all(&self) {
        for t in self.threads.lock().iter() {
            t.unpark();
        }
    }
}

/// Bounded parks so a lost wake costs latency, never liveness.
const PARK_TIMEOUT: Duration = Duration::from_micros(200);
const YIELDS_BEFORE_PARK: u32 = 8;

/// Wait until `cond()` (or `abort()`), spinning `spin_us` first. Returns
/// `false` if aborted.
#[inline]
pub fn wait_until(
    parker: &Parker,
    spin_us: u32,
    mut cond: impl FnMut() -> bool,
    mut abort: impl FnMut() -> bool,
) -> bool {
    if cond() {
        return true;
    }
    let spin = Duration::from_micros(spin_us as u64);
    let t0 = Instant::now();
    let mut n = 0u32;
    loop {
        std::hint::spin_loop();
        n = n.wrapping_add(1);
        if n & 63 == 0 {
            if cond() {
                return true;
            }
            if abort() {
                return false;
            }
            if t0.elapsed() >= spin {
                break;
            }
        }
    }
    for _ in 0..YIELDS_BEFORE_PARK {
        std::thread::yield_now();
        if cond() {
            return true;
        }
        if abort() {
            return false;
        }
    }
    loop {
        parker.sleepers.fetch_add(1, Ordering::AcqRel);
        // Re-check after announcing: a bumper that saw `sleepers == 0` just
        // before our increment has already completed its counter store.
        if cond() || abort() {
            parker.sleepers.fetch_sub(1, Ordering::AcqRel);
            return cond();
        }
        std::thread::park_timeout(PARK_TIMEOUT);
        parker.sleepers.fetch_sub(1, Ordering::AcqRel);
        if cond() {
            return true;
        }
        if abort() {
            return false;
        }
    }
}

/// Everything a run shares between workers, besides the program itself.
pub struct RunShared<'a> {
    pub prog: &'a LoadedProgram,
    pub pool: &'a CounterPool,
    pub seg: u32,
    pub gen: u32,
    pub fb: &'a Feedback,
    /// One per node; index = node position in the pool's node list.
    pub parkers: &'a [Parker],
    pub spin_us: u32,
    /// GQ cursors, `[gq_windows]`, zeroed by the host before the run.
    pub cursors: &'a [CachePadded<AtomicU32>],
}

impl<'a> RunShared<'a> {
    #[inline]
    fn cancelled(&self) -> bool {
        self.fb.cancel_gen.load(Ordering::Relaxed) == self.gen
    }

    #[inline]
    fn gates_open(&self, e: &StreamEnt) -> bool {
        self.prog
            .waits_of(e)
            .iter()
            .all(|w| self.pool.load(w.id) >= w.threshold as u64)
    }

    /// Execute one entry and publish its successors.
    #[inline]
    fn fire(&self, e: &StreamEnt, exec: &dyn Exec, me: &WorkerCtx) {
        let inst = &self.prog.insts[e.inst as usize];
        let tracing = TRACE_ON.load(Ordering::Relaxed);
        let t0 = if tracing { trace_now_ns() } else { 0 };
        exec.exec(inst, e.slice, inst.blocks as u32, me);
        if tracing {
            let ev = TraceEv {
                inst: e.inst,
                slice: e.slice,
                worker: me.worker as u16,
                t0_ns: t0,
                t1_ns: trace_now_ns(),
            };
            TRACE_BUF.with(|b| b.borrow_mut().push(ev));
        }
        let succs = self.prog.succs_of(e);
        for &c in succs {
            self.pool.add(c, 1);
        }
        if !succs.is_empty() {
            // One relaxed load per parker on the hot path; the wake itself is cold.
            for p in self.parkers {
                if p.has_sleepers() {
                    p.unpark_all();
                }
            }
        }
    }
}

/// Per-worker static-mode state, preallocated at spawn (no per-run allocation).
pub struct StaticState {
    /// Cu ids owned at spawn — the fallback when a program does not narrow participation.
    spawn_cus: Vec<u32>,
    /// Cu ids owned for the program being run.
    pub cus: Vec<u32>,
    /// `[head, end)` per owned cu for the current segment.
    heads: Vec<(u32, u32)>,
    /// This worker's index, the key into `LoadedProgram::cus_of`.
    idx: usize,
}

impl StaticState {
    pub fn new(idx: usize, cus: Vec<u32>) -> StaticState {
        let heads = vec![(0, 0); cus.len()];
        StaticState { spawn_cus: cus.clone(), cus, heads, idx }
    }

    fn reset(&mut self, prog: &LoadedProgram, seg: u32) {
        let mine: &[u32] = match &prog.cus_of {
            Some(m) => m.get(self.idx).map(Vec::as_slice).unwrap_or(&[]),
            None => &self.spawn_cus,
        };
        if self.cus != mine {
            self.cus.clear();
            self.cus.extend_from_slice(mine);
            self.heads.clear();
            self.heads.resize(self.cus.len(), (0, 0));
        }
        for (k, &cu) in self.cus.iter().enumerate() {
            self.heads[k] = prog.cu_range(cu, seg);
        }
    }
}

/// Static walk over this worker's owned streams. Returns when every owned
/// stream is drained for `seg`, or on cancel. The caller bumps `done`.
pub fn run_static(
    st: &mut StaticState,
    sh: &RunShared<'_>,
    exec: &dyn Exec,
    me: &WorkerCtx,
    parker: &Parker,
) {
    let prog = sh.prog;
    st.reset(prog, sh.seg);
    let filter_seg = prog.seg_ofs.is_none() && prog.n_seg() > 1;
    loop {
        let mut remaining = false;
        let mut progressed = false;
        for k in 0..st.cus.len() {
            let (mut head, end) = st.heads[k];
            while head < end {
                let e = &prog.stream[head as usize];
                if filter_seg && e.seg as u32 != sh.seg {
                    head += 1;
                    continue;
                }
                if !sh.gates_open(e) {
                    break;
                }
                sh.fire(e, exec, me);
                head += 1;
                progressed = true;
                if sh.cancelled() {
                    return;
                }
            }
            st.heads[k].0 = head;
            remaining |= head < end;
        }
        if !remaining {
            return;
        }
        if !progressed {
            // Every owned head is blocked: wait for any successor bump.
            let heads = &st.heads;
            let ok = wait_until(
                parker,
                sh.spin_us,
                || {
                    heads.iter().any(|&(h, e)| {
                        h < e && {
                            let ent = &prog.stream[h as usize];
                            (filter_seg && ent.seg as u32 != sh.seg) || sh.gates_open(ent)
                        }
                    })
                },
                || sh.cancelled(),
            );
            if !ok {
                return;
            }
        }
    }
}

/// Per-worker GQ state, preallocated at spawn.
pub struct GqState {
    /// Claimed entries (indices into `gq.stream`) whose gates were closed.
    pending: Vec<u32>,
}

/// Bound on claimed-but-blocked entries per worker. When full the worker waits
/// on its pending set instead of claiming more.
pub const GQ_PENDING_CAP: usize = 256;

impl GqState {
    pub fn new() -> GqState {
        GqState {
            pending: Vec::with_capacity(GQ_PENDING_CAP),
        }
    }
}

impl Default for GqState {
    fn default() -> Self {
        GqState::new()
    }
}

/// Global-queue walk for `seg`. Claims from my domain's window; a claimed entry
/// whose gates are closed goes to `pending` and the next claim rotates to the
/// next window, so every window is served even when a domain has no worker of
/// its own. Waits only when pending is full or every window is drained.
///
/// Progress argument: windows are op-major, so the globally earliest unclaimed
/// entry has every producer already claimed; the earliest claimed-unfired entry
/// has every producer fired, hence is ready, and its claimant rescans pending
/// on every iteration. Cancel abandons pending (the host re-zeroes counters and
/// cursors before the next run).
pub fn run_gq(
    st: &mut GqState,
    sh: &RunShared<'_>,
    exec: &dyn Exec,
    me: &WorkerCtx,
    parker: &Parker,
) {
    let prog = sh.prog;
    let Some(gq) = prog.gq.as_ref() else {
        return;
    };
    st.pending.clear();
    let domains = gq.domains.max(1);
    let base = sh.seg as usize * domains as usize;
    let mut d = me.domain % domains;
    let mut drained = 0u32; // consecutive windows found exhausted
    loop {
        // 1. Fire whatever became ready.
        let mut k = 0;
        while k < st.pending.len() {
            let e = &gq.stream[st.pending[k] as usize];
            if sh.gates_open(e) {
                sh.fire(e, exec, me);
                st.pending.swap_remove(k);
                if sh.cancelled() {
                    return;
                }
            } else {
                k += 1;
            }
        }
        // 2. Claim, unless pending is full.
        let mut claimed = false;
        if st.pending.len() < GQ_PENDING_CAP && drained < domains {
            let w = base + d as usize;
            let (lo, hi) = (gq.seg_ofs[w], gq.seg_ofs[w + 1]);
            let idx = lo + sh.cursors[w].fetch_add(1, Ordering::AcqRel);
            if idx < hi {
                claimed = true;
                drained = 0;
                let e = &gq.stream[idx as usize];
                if sh.gates_open(e) {
                    sh.fire(e, exec, me);
                    if sh.cancelled() {
                        return;
                    }
                } else {
                    st.pending.push(idx);
                    // Blocked: serve the next window on the next claim.
                    d = (d + 1) % domains;
                }
            } else {
                drained += 1;
                d = (d + 1) % domains;
                continue;
            }
        }
        if claimed {
            continue;
        }
        if st.pending.is_empty() {
            return; // every window drained, nothing outstanding
        }
        // 3. Nothing to claim (or pending full): wait for any pending gate.
        let pending = &st.pending;
        let ok = wait_until(
            parker,
            sh.spin_us,
            || pending.iter().any(|&i| sh.gates_open(&gq.stream[i as usize])),
            || sh.cancelled(),
        );
        if !ok {
            return;
        }
    }
}

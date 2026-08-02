//! Real CPU reference backend.
//!
//! Its `alloc`/`upload`/`download` treat host memory as "device" memory, and
//! [`interpret`] walks a compiled `Program` honoring the counter protocol — the
//! same gate/dispatch/fan-out loop a real SM runs, minus vendor numerics. This
//! is what the hermetic tests execute, and the oracle GPU output is checked
//! against (golden op bodies are a documented seam, below).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use packet::{Body, Inst, Program, ResourceKind};

use crate::device::{
    Backend, Backing, DeviceMem, ExecutorClass, ExecutorTarget, LaunchCfg, Module,
};
use crate::exec::counters::CounterPool;
use crate::{Result, RuntimeError};

/// Host arena standing in for device HBM. Interior mutability with an explicit
/// `unsafe impl Send + Sync`: the schedule guarantees disjoint concurrent access
/// to distinct byte ranges, exactly as on-device, so no lock is taken on the hot
/// path. Off-path `upload`/`download` also go through the raw slice.
pub struct CpuArena {
    buf: UnsafeCell<Box<[u8]>>,
}

// SAFETY: concurrent accesses are to disjoint ranges by construction (the
// compiler's liveness + counter schedule). No aliasing writes overlap.
unsafe impl Send for CpuArena {}
unsafe impl Sync for CpuArena {}

impl CpuArena {
    fn new(bytes: u64) -> Arc<Self> {
        Arc::new(CpuArena {
            buf: UnsafeCell::new(vec![0u8; bytes as usize].into_boxed_slice()),
        })
    }

    /// Base host pointer as a device-address-like integer.
    fn base(&self) -> u64 {
        // `self.buf.get()` is the address of the Box's fat pointer, not the
        // heap buffer; deref to reach the actual data.
        unsafe { (*self.buf.get()).as_ptr() as u64 }
    }

    /// Whole-arena read view.
    ///
    /// # Safety
    /// The returned slice aliases the **entire** arena, so the caller must
    /// guarantee no concurrent `write`/`upload` for the slice's lifetime — call
    /// only at quiescent points (between iterations, teardown, tests).
    pub(crate) unsafe fn as_slice(&self) -> &[u8] {
        let b: &[u8] = &*self.buf.get();
        std::slice::from_raw_parts(b.as_ptr(), b.len())
    }

    /// SAFETY: caller guarantees `[off, off+len)` is in bounds and does not
    /// overlap any concurrent access to the same range.
    unsafe fn write(&self, off: u64, src: &[u8]) {
        // Raw-pointer copy: never materialize a `&mut` over the whole buffer,
        // so disjoint concurrent writes to other ranges stay sound.
        debug_assert!(off as usize + src.len() <= (&*self.buf.get()).len());
        let base = (*self.buf.get()).as_mut_ptr();
        std::ptr::copy_nonoverlapping(src.as_ptr(), base.add(off as usize), src.len());
    }

    /// SAFETY: as [`write`]: in bounds, no concurrent write to this range.
    unsafe fn read(&self, off: u64, dst: &mut [u8]) {
        debug_assert!(off as usize + dst.len() <= (&*self.buf.get()).len());
        let base = (*self.buf.get()).as_ptr();
        std::ptr::copy_nonoverlapping(base.add(off as usize), dst.as_mut_ptr(), dst.len());
    }
}

/// The CPU executor pool backend.
pub struct CpuBackend {
    executors: u32,
    module_ctr: AtomicU64,
}

impl CpuBackend {
    /// A pool of `executors` CPU executor threads' worth of capability. (The
    /// reference interpreter runs them cooperatively on one thread.)
    pub fn new(executors: u32) -> Self {
        CpuBackend {
            executors,
            module_ctr: AtomicU64::new(1),
        }
    }
}

impl Backend for CpuBackend {
    fn class(&self) -> ExecutorClass {
        ExecutorClass::Cpu
    }

    fn enumerate(&self) -> Vec<ExecutorTarget> {
        (0..self.executors)
            .map(|i| ExecutorTarget {
                class: ExecutorClass::Cpu,
                instance_id: i,
                wave_width: 1,
                worker_count: 1,
                shmem_bytes: 0,
                // CPU backend implements the golden variant of every family.
                opcode_mask: u32::MAX,
            })
            .collect()
    }

    fn alloc(&self, _device: u8, bytes: u64) -> Result<DeviceMem> {
        let arena = CpuArena::new(bytes);
        Ok(DeviceMem {
            base: arena.base(),
            len: bytes,
            backing: Backing::Cpu(arena),
        })
    }

    fn upload(&self, dst: &DeviceMem, off: u64, src: &[u8]) -> Result<()> {
        match &dst.backing {
            Backing::Cpu(a) => {
                if off + src.len() as u64 > dst.len {
                    return Err(RuntimeError::Oom(format!(
                        "upload [{off}, {}) exceeds arena {}",
                        off + src.len() as u64,
                        dst.len
                    )));
                }
                // SAFETY: bounds checked; upload is a startup/pressure op with no
                // concurrent writer to this range.
                unsafe { a.write(off, src) };
                Ok(())
            }
            Backing::Owned { .. } | Backing::View => {
                Err(RuntimeError::Device("upload to device-owned mem".into()))
            }
        }
    }

    fn download(&self, src: &DeviceMem, off: u64, dst: &mut [u8]) -> Result<()> {
        match &src.backing {
            Backing::Cpu(a) => {
                // SAFETY: bounds checked below; read view.
                if off + dst.len() as u64 > src.len {
                    return Err(RuntimeError::Oom("download out of range".into()));
                }
                unsafe { a.read(off, dst) };
                Ok(())
            }
            Backing::Owned { .. } | Backing::View => Err(RuntimeError::Device(
                "download from device-owned mem".into(),
            )),
        }
    }

    fn module_load(&self, _image: &[u8]) -> Result<Module> {
        Ok(Module {
            id: self.module_ctr.fetch_add(1, Ordering::Relaxed),
        })
    }

    fn launch_persistent(&self, _module: &Module, _cfg: LaunchCfg) -> Result<()> {
        // On CPU the "persistent kernel" is the cooperative `interpret` walk,
        // driven by the scheduler per iteration; nothing to launch up front.
        Ok(())
    }
}

/// Outcome of a reference interpretation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterpretStats {
    pub executed: usize,
    pub total: usize,
    pub completed: bool,
    /// Modelled makespan in cycles — the max per-executor finish time.
    pub makespan: u64,
}

/// Observer invoked as each packet fires during a stream walk. `interpret` uses a
/// no-op observer; the §O simulator uses a recording one. This is the single
/// seam that turns the reference interpreter into a dry-run tracer without
/// duplicating the gating/ordering logic.
pub trait StepObserver {
    /// Whether to execute the golden op body (numerics) when a packet fires.
    /// `false` = pure dry run (no math).
    fn run_math(&self) -> bool;
    /// Called after a packet fires, with its modelled `[t_start, t_end)` cycles.
    fn on_fire(&mut self, packet_index: usize, inst: &Inst, t_start: u64, t_end: u64);
}

/// No-op observer that runs (or skips) math but records nothing.
pub struct NoopObserver {
    pub math: bool,
}

impl StepObserver for NoopObserver {
    #[inline]
    fn run_math(&self) -> bool {
        self.math
    }
    #[inline]
    fn on_fire(&mut self, _i: usize, _inst: &Inst, _t_start: u64, _t_end: u64) {}
}

/// Fixed per-op issue overhead (mirrors the cost model's `LAUNCH_CYCLES`).
const LAUNCH_CYCLES: u64 = 500;

#[inline]
fn ceil_div(a: u64, b: u64) -> u64 {
    let b = b.max(1);
    (a + b - 1) / b
}

/// Coarse per-op cycle cost derived from the packet's own fields — enough for a
/// plausible timeline. SEAM: swap in `costmodel` for fidelity.
pub fn op_cost(body: &Body) -> u64 {
    let work = match body {
        Body::Gemm {
            m,
            n,
            k,
            bm,
            bn,
            bk,
            ..
        } => {
            ceil_div(*m as u64, *bm as u64)
                * ceil_div(*n as u64, *bn as u64)
                * ceil_div(*k as u64, *bk as u64)
        }
        Body::Flash {
            seq_q,
            seq_kv,
            head_dim,
            bq,
            bkv,
            heads,
            ..
        } => {
            ceil_div(*seq_q as u64, *bq as u64)
                * ceil_div(*seq_kv as u64, *bkv as u64)
                * (*head_dim as u64)
                * (*heads as u64).max(1)
                / 64
        }
        Body::Row { rows, feat, br, .. } => ceil_div(*rows as u64, *br as u64) * (*feat as u64),
        Body::Layout { shape, rank, .. } => {
            let mut p = 1u64;
            for &s in shape.iter().take(*rank as usize) {
                p = p.saturating_mul((s as u64).max(1));
            }
            p
        }
        Body::Dma { bytes, .. } => (*bytes as u64) / 128,
        Body::Rdma { bytes, .. } => (*bytes as u64) / 64,
        // Host token op (sample/tokenize): ~ reading the logits row.
        Body::Token { vocab, .. } => (*vocab as u64) / 256,
        Body::Host => 0,
    };
    LAUNCH_CYCLES + work
}

/// Walk `program` honoring the counter protocol, modelling every executor's
/// per-resource FIFO stream cooperatively on this thread.
///
/// Each executor consumes its instructions in issue order; an instruction fires
/// once all its `wait` counters have reached threshold, then increments its
/// `succ` counters. Round-robin advancing the stream heads mirrors concurrent
/// executors and makes a genuine deadlock observable: a full sweep with no head
/// advancing while work remains.
pub fn interpret(program: &Program, pool: &CounterPool) -> InterpretStats {
    run_streams(program, pool, &mut NoopObserver { math: true })
}

/// Precomputed per-program stream bucketing plus per-run clocks. Build once per
/// bucket and pass to [`run_streams_reuse`] each iteration so the per-token hot
/// path allocates nothing (the map/vecs are rewound, not rebuilt).
pub struct StreamSet {
    /// Per-executor FIFO of instruction indices, its head, and its cycle clock.
    heads: Vec<(Vec<usize>, usize, u64)>,
    /// Modelled cycle at which each counter reached its threshold (0 = not yet /
    /// no wait), so a consumer can't start before its producer finished.
    ready_time: Vec<u64>,
    /// Ready queue: stream indices whose current head has all waits satisfied.
    /// Populated at reset and after each counter fire. Avoids rescanning all
    /// streams on each iteration (O(fired) amortized instead of O(streams²)).
    ready: Vec<usize>,
    /// Per-counter: which stream indices have a head waiting on this counter.
    /// Rebuilt at reset; updated as heads advance. Enables O(1) lookup of
    /// which streams to re-check when a counter fires.
    waiters: Vec<Vec<usize>>,
}

impl StreamSet {
    /// Bucket `program`'s instruction indices into per-executor FIFO streams
    /// (resource, index). `counters` is the counter-pool length.
    pub fn new(program: &Program, counters: usize) -> Self {
        let mut streams: rustc_hash::FxHashMap<(u8, u16), Vec<usize>> = Default::default();
        for (i, inst) in program.insts.iter().enumerate() {
            streams
                .entry((resource_tag(inst.resource), inst.index))
                .or_default()
                .push(i);
        }
        StreamSet {
            heads: streams.into_values().map(|s| (s, 0usize, 0u64)).collect(),
            ready_time: vec![0u64; counters],
            ready: Vec::new(),
            waiters: vec![Vec::new(); counters],
        }
    }

    /// Rewind heads/clocks for another run of the same program.
    fn reset(&mut self) {
        for (_, head, clock) in self.heads.iter_mut() {
            *head = 0;
            *clock = 0;
        }
        self.ready_time.fill(0);
        self.ready.clear();
        for w in self.waiters.iter_mut() {
            w.clear();
        }
    }
}

/// The shared counter-gated per-executor FIFO walk. `interpret` and the §O
/// simulator both call this; the `observer` decides whether math runs and
/// records each fired packet with its modelled timing.
///
/// Timing model: each executor stream carries a cycle clock; a fired packet runs
/// `[max(stream_clock, waited-producer-finish), + op_cost)` so the timeline
/// respects both per-executor ordering and cross-executor dependencies.
pub fn run_streams<O: StepObserver>(
    program: &Program,
    pool: &CounterPool,
    observer: &mut O,
) -> InterpretStats {
    let mut streams = StreamSet::new(program, pool.len());
    run_streams_reuse(program, pool, observer, &mut streams)
}

/// [`run_streams`] over a caller-owned, reusable [`StreamSet`] — the
/// allocation-free per-token path. `streams` must have been built for this
/// `program`/`pool`; it is reset before the walk.
///
/// Uses a ready-queue: instead of scanning all stream heads each iteration,
/// we seed the queue with zero-wait heads and re-check only streams whose
/// waited counter just fired. Amortized O(fired·successors) vs O(P·streams).
pub fn run_streams_reuse<O: StepObserver>(
    program: &Program,
    pool: &CounterPool,
    observer: &mut O,
    streams: &mut StreamSet,
) -> InterpretStats {
    streams.reset();
    let total = program.insts.len();
    let run_math = observer.run_math();
    let mut executed = 0usize;
    let mut makespan = 0u64;

    // Seed: every stream whose head has no waits (or all waits pre-satisfied)
    // is immediately ready. Register waiters for the rest.
    for si in 0..streams.heads.len() {
        let (ref stream, head, _) = streams.heads[si];
        if head >= stream.len() {
            continue;
        }
        let idx = stream[head];
        let inst = &program.insts[idx];
        if inst.wait.is_empty() || inst.wait.iter().all(|&c| pool.satisfied(c)) {
            streams.ready.push(si);
        } else {
            // Register this stream as waiting on each unsatisfied counter.
            for &c in &inst.wait {
                if !pool.satisfied(c) {
                    if let Some(w) = streams.waiters.get_mut(c as usize) {
                        w.push(si);
                    }
                }
            }
        }
    }

    loop {
        if streams.ready.is_empty() {
            // Check if all done or deadlocked.
            let all_done = streams
                .heads
                .iter()
                .all(|(stream, head, _)| *head >= stream.len());
            return InterpretStats {
                executed,
                total,
                completed: all_done,
                makespan,
            };
        }

        // Drain the ready queue. We swap to a local to avoid borrow issues
        // when pushing newly-ready streams during the loop.
        let batch: Vec<usize> = std::mem::take(&mut streams.ready);
        for si in batch {
            let (ref stream, ref mut head, ref mut clock) = streams.heads[si];
            if *head >= stream.len() {
                continue;
            }
            let idx = stream[*head];
            let inst = &program.insts[idx];

            // Double-check readiness (a stream can land in the queue twice
            // if multiple counters fire for it before it drains).
            if !inst.wait.iter().all(|&c| pool.satisfied(c)) {
                // Not actually ready yet — re-register for remaining waits.
                for &c in &inst.wait {
                    if !pool.satisfied(c) {
                        if let Some(w) = streams.waiters.get_mut(c as usize) {
                            if !w.contains(&si) {
                                w.push(si);
                            }
                        }
                    }
                }
                continue;
            }

            let wait_ready = inst
                .wait
                .iter()
                .map(|&c| streams.ready_time[c as usize])
                .max()
                .unwrap_or(0);
            let t_start = (*clock).max(wait_ready);
            let t_end = t_start + op_cost(&inst.body);
            *clock = t_end;
            makespan = makespan.max(t_end);

            if run_math {
                execute(inst);
            }
            observer.on_fire(idx, inst, t_start, t_end);

            // Fire successors: increment counters, update ready_time, and
            // check if any waiting stream is now satisfied.
            for &c in &inst.succ {
                pool.add(c, 1);
                let rt = &mut streams.ready_time[c as usize];
                *rt = (*rt).max(t_end);

                // Drain waiters for this counter — any now-satisfied stream
                // gets pushed to the ready queue.
                if pool.satisfied(c) {
                    if let Some(waiters) = streams.waiters.get_mut(c as usize) {
                        for &wsi in waiters.iter() {
                            streams.ready.push(wsi);
                        }
                        waiters.clear();
                    }
                }
            }

            *head += 1;
            executed += 1;

            // Advance: check if the new head of this stream is ready.
            if *head < stream.len() {
                let next_idx = stream[*head];
                let next_inst = &program.insts[next_idx];
                if next_inst.wait.is_empty() || next_inst.wait.iter().all(|&c| pool.satisfied(c)) {
                    streams.ready.push(si);
                } else {
                    for &c in &next_inst.wait {
                        if !pool.satisfied(c) {
                            if let Some(w) = streams.waiters.get_mut(c as usize) {
                                if !w.contains(&si) {
                                    w.push(si);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Execute one instruction's op body.
///
/// SEAM: the golden (single-thread, correctness-reference) numerics for each
/// family go here — GEMM, Flash, Row, Layout over the arena buffers. The runtime
/// mechanics (gating, fan-out, ordering) are exercised without them; wiring the
/// bodies makes `interpret` a bit-exact oracle for GPU output.
#[inline]
fn execute(inst: &Inst) {
    match &inst.body {
        Body::Gemm { .. }
        | Body::Flash { .. }
        | Body::Row { .. }
        | Body::Layout { .. }
        | Body::Dma { .. }
        | Body::Rdma { .. }
        | Body::Token { .. } // §P host-op work runs in the HostExecutor observer
        | Body::Host => { /* golden numerics TODO — see seam above */ }
    }
}

#[inline]
fn resource_tag(r: ResourceKind) -> u8 {
    r as u8
}

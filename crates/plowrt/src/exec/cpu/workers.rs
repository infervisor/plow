//! Persistent worker pool: one core-pinned OS thread per executor, spawned once
//! and alive until the pool drops. Work arrives as control commands
//! (`control.rs`); a drained program returns the worker to the control loop —
//! spin, yield, park — never to `thread::exit`. This is the CPU's persistent
//! megakernel: no spawn on any request path, completion is a counter.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_utils::CachePadded;
use parking_lot::Mutex;

use crate::exec::counters::CounterPool;
use crate::exec::cpu::control::{
    Cmd, ControlRing, Feedback, CMD_BARRIER, CMD_CANCEL, CMD_RESET_SLOT, CMD_RUN, CMD_STOP,
};
use crate::exec::cpu::interp::{
    run_gq, run_static, wait_until, Exec, GqState, LoadedProgram, Parker, RunShared,
    StaticState, WorkerCtx,
};
use crate::exec::cpu::topology::{NumaMode, Topology};

/// State every worker shares (read-mostly; the hot path touches counters, one
/// `cancel_gen` load, and the parkers' `sleepers`).
struct Shared {
    ring: ControlRing,
    fb: Feedback,
    parkers: Vec<Parker>,
    /// GQ cursors sized for the largest program run so far. Host-resized under
    /// the run lock before a run; workers only see it through `RunShared`.
    cursors: Mutex<Arc<Vec<CachePadded<AtomicU32>>>>,
    spin_us: u32,
    n_workers: u32,
}

struct WorkerInit {
    idx: u32,
    node: u32,
    node_pos: u32,
    cpu: u32,
    cus: Vec<u32>,
}

/// Keeps the program and counters of the in-flight run alive for the workers,
/// which see them only as raw pointers in the `CMD_RUN` record.
struct Inflight {
    gen: u32,
    _prog: Arc<LoadedProgram>,
    _pool: Arc<CounterPool>,
    _cursors: Arc<Vec<CachePadded<AtomicU32>>>,
}

pub struct WorkerPool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    /// Host-side bookkeeping; every method here is a cold path.
    host: Mutex<Host>,
    /// Which node each worker sits on, `[n_workers]` (for placement callers).
    worker_node: Vec<u32>,
    n_cu: u32,
}

struct Host {
    gen: u32,
    barrier_seq: u32,
    inflight: Option<Inflight>,
}

impl WorkerPool {
    /// Spawn `threads` persistent workers (0 = one per physical core on the
    /// selected nodes). Virtual executor `cu` is placed on node `cu % nodes`,
    /// round-robin across that node's workers, so `n_cu` need not equal the
    /// thread count.
    pub fn spawn(
        topo: &Topology,
        threads: usize,
        numa: &NumaMode,
        spin_us: u32,
        n_cu: u32,
        exec: Arc<dyn Exec>,
    ) -> WorkerPool {
        let nodes = topo.select_nodes(numa);
        let mut cores: Vec<(u32, u32)> = Vec::new(); // (cpu, node)
        for &n in &nodes {
            cores.extend(topo.cores_on_node(n).map(|c| (c.cpu, n)));
        }
        if cores.is_empty() {
            cores.push((0, nodes[0]));
        }
        let threads = if threads == 0 { cores.len() } else { threads.max(1) };
        // Round-robin over cores when oversubscribed.
        let placement: Vec<(u32, u32)> = (0..threads).map(|k| cores[k % cores.len()]).collect();
        let node_pos = |n: u32| nodes.iter().position(|&x| x == n).unwrap_or(0) as u32;

        // cu → worker: node = cu % nodes, then round-robin within the node.
        let mut per_node: Vec<Vec<u32>> = vec![Vec::new(); nodes.len()];
        for (w, &(_, n)) in placement.iter().enumerate() {
            per_node[node_pos(n) as usize].push(w as u32);
        }
        let mut cus_of: Vec<Vec<u32>> = vec![Vec::new(); threads];
        for cu in 0..n_cu {
            let np = (cu as usize) % nodes.len();
            let ws = if per_node[np].is_empty() {
                per_node.iter().find(|v| !v.is_empty()).expect("some worker")
            } else {
                &per_node[np]
            };
            let w = ws[(cu as usize / nodes.len()) % ws.len()];
            cus_of[w as usize].push(cu);
        }

        let shared = Arc::new(Shared {
            ring: ControlRing::new(threads),
            fb: Feedback::default(),
            parkers: (0..nodes.len()).map(|_| Parker::default()).collect(),
            cursors: Mutex::new(Arc::new(Vec::new())),
            spin_us,
            n_workers: threads as u32,
        });

        let mut handles = Vec::with_capacity(threads);
        let mut worker_node = Vec::with_capacity(threads);
        for (w, &(cpu, node)) in placement.iter().enumerate() {
            worker_node.push(node);
            let init = WorkerInit {
                idx: w as u32,
                node,
                node_pos: node_pos(node),
                cpu,
                cus: std::mem::take(&mut cus_of[w]),
            };
            let sh = shared.clone();
            let ex = exec.clone();
            let h = std::thread::Builder::new()
                .name(format!("plow-cpu-{node}-{w}"))
                .spawn(move || worker_main(init, sh, ex))
                .expect("spawn cpu worker");
            handles.push(h);
        }
        WorkerPool {
            shared,
            threads: handles,
            host: Mutex::new(Host {
                gen: 0,
                barrier_seq: 0,
                inflight: None,
            }),
            worker_node,
            n_cu,
        }
    }

    pub fn threads(&self) -> usize {
        self.threads.len()
    }

    pub fn n_cu(&self) -> u32 {
        self.n_cu
    }

    pub fn worker_node(&self) -> &[u32] {
        &self.worker_node
    }

    pub fn feedback(&self) -> &Feedback {
        &self.shared.fb
    }

    /// Start running segment `seg` of `prog` against `counters` (which the
    /// caller has zeroed). Returns the run generation for [`wait_done`] /
    /// [`cancel`]. Exactly one run may be in flight.
    ///
    /// [`wait_done`]: WorkerPool::wait_done
    /// [`cancel`]: WorkerPool::cancel
    pub fn run(&self, prog: &Arc<LoadedProgram>, seg: u32, counters: &Arc<CounterPool>) -> u32 {
        let mut host = self.host.lock();
        assert!(
            host.inflight.is_none(),
            "WorkerPool::run while a run is in flight — wait_done first"
        );
        assert!(
            prog.n_cu <= self.n_cu,
            "program has {} cus, pool was built for {}",
            prog.n_cu,
            self.n_cu
        );
        host.gen = host.gen.wrapping_add(1).max(1);
        let gen = host.gen;

        let cursors = {
            let mut cur = self.shared.cursors.lock();
            let need = prog.gq_windows();
            if cur.len() < need {
                let mut v = Vec::with_capacity(need);
                v.resize_with(need, || CachePadded::new(AtomicU32::new(0)));
                *cur = Arc::new(v);
            } else {
                for c in cur.iter().take(need) {
                    c.store(0, Ordering::Relaxed);
                }
            }
            cur.clone()
        };

        let fb = &self.shared.fb;
        fb.done.store(0, Ordering::Release);
        fb.fault.store(0, Ordering::Release);
        self.shared.ring.push(Cmd::run(
            gen,
            Arc::as_ptr(prog) as u64,
            Arc::as_ptr(counters) as u64,
            seg,
        ));
        self.wake_all();
        host.inflight = Some(Inflight {
            gen,
            _prog: prog.clone(),
            _pool: counters.clone(),
            _cursors: cursors,
        });
        gen
    }

    /// Block until every worker has drained (or abandoned) run `gen`.
    /// Returns the first fault recorded, if any.
    pub fn wait_done(&self, gen: u32) -> Option<u64> {
        let n = self.shared.n_workers;
        let fb = &self.shared.fb;
        // Host-side wait: spin briefly, then yield — the host has no parker.
        let spun = wait_until(
            &HOST_PARKER,
            self.shared.spin_us,
            || fb.done.load(Ordering::Acquire) >= n,
            || false,
        );
        debug_assert!(spun);
        let mut host = self.host.lock();
        match host.inflight.take() {
            Some(inf) => debug_assert_eq!(inf.gen, gen),
            None => panic!("wait_done({gen}) with no run in flight"),
        }
        let f = fb.fault.load(Ordering::Acquire);
        (f != 0).then_some(f)
    }

    /// Abandon run `gen` at every worker's next packet boundary. The caller
    /// still calls [`wait_done`](WorkerPool::wait_done) and must then re-zero
    /// the counters before the next run.
    pub fn cancel(&self, gen: u32) {
        self.shared.fb.cancel_gen.store(gen, Ordering::Release);
        self.shared.ring.push(Cmd::cancel(gen));
        self.wake_all();
    }

    /// Every worker acknowledges; establishes ordering between earlier
    /// commands (resets) and later runs.
    pub fn barrier(&self) {
        let seq = {
            let mut host = self.host.lock();
            host.barrier_seq = host.barrier_seq.wrapping_add(1);
            host.barrier_seq
        };
        let fb = &self.shared.fb;
        fb.barrier_ack.store(0, Ordering::Release);
        self.shared.ring.push(Cmd::barrier(seq));
        self.wake_all();
        let n = self.shared.n_workers;
        wait_until(
            &HOST_PARKER,
            self.shared.spin_us,
            || fb.barrier_ack.load(Ordering::Acquire) >= n,
            || false,
        );
    }

    /// Reset a request slot: zero `ranges` (host memory) with every worker
    /// taking an equal share of each range, then barrier. Must not overlap a
    /// run that touches the same bytes.
    ///
    /// # Safety
    /// Each `(ptr, len)` must be valid writable memory for the duration of the call.
    pub unsafe fn reset_slot(&self, slot: u32, ranges: &[(*mut u8, usize)]) {
        for &(p, len) in ranges {
            if len == 0 {
                continue;
            }
            self.shared
                .ring
                .push(Cmd::reset_slot(slot, p as u64, len as u64));
        }
        self.barrier();
    }

    fn wake_all(&self) {
        for p in &self.shared.parkers {
            if p.has_sleepers() {
                p.unpark_all();
            }
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shared.ring.push(Cmd::stop());
        self.wake_all();
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
    }
}

/// The host never parks (nobody would wake it); a parker with no registrants
/// makes `wait_until` degrade to spin + bounded sleeps.
static HOST_PARKER: Parker = Parker::new();

#[cfg(all(feature = "cpu", target_os = "linux"))]
fn pin_to_cpu(cpu: u32) {
    // SAFETY: cpu_set_t is POD; sched_setaffinity on the calling thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu as usize, &mut set);
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(all(feature = "cpu", target_os = "linux")))]
fn pin_to_cpu(_cpu: u32) {}

fn worker_main(init: WorkerInit, sh: Arc<Shared>, exec: Arc<dyn Exec>) {
    pin_to_cpu(init.cpu);
    let parker = &sh.parkers[init.node_pos as usize];
    parker.register();
    let me = WorkerCtx {
        worker: init.idx,
        node: init.node,
        cpu: init.cpu,
        domain: init.node_pos,
    };
    let mut st = StaticState::new(init.cus);
    let mut gq = GqState::new();
    let mut seen = 0u64;
    let exec: &dyn Exec = &*exec;
    loop {
        let cmd = loop {
            if let Some(c) = sh.ring.peek(seen) {
                seen += 1;
                sh.ring.ack(init.idx as usize, seen);
                break c;
            }
            // Idle: no command. Park until the host pushes one.
            wait_until(parker, sh.spin_us, || sh.ring.tail() != seen, || false);
        };
        match cmd.kind {
            CMD_RUN => {
                // SAFETY: the host keeps both Arcs alive in `Inflight` until
                // every worker has bumped `done` for this generation.
                let prog: &LoadedProgram = unsafe { &*(cmd.a as *const LoadedProgram) };
                let pool: &CounterPool = unsafe { &*(cmd.b as *const CounterPool) };
                let cursors = sh.cursors.lock().clone();
                let run = RunShared {
                    prog,
                    pool,
                    seg: cmd.c as u32,
                    gen: cmd.gen,
                    fb: &sh.fb,
                    parkers: &sh.parkers,
                    spin_us: sh.spin_us,
                    cursors: &cursors,
                };
                // A panicking kernel must not kill a persistent worker: record
                // it as a fault and stay in the loop (release builds abort on
                // panic anyway; this matters for the Rust test mocks).
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if prog.gq.is_some() {
                        run_gq(&mut gq, &run, exec, &me, parker);
                    } else {
                        run_static(&mut st, &run, exec, &me, parker);
                    }
                }));
                if r.is_err() {
                    sh.fb.fault(u16::MAX, u32::MAX, init.idx as u16);
                }
                sh.fb.done.fetch_add(1, Ordering::AcqRel);
            }
            // The store on `cancel_gen` did the work; the record only wakes parked workers.
            CMD_CANCEL => {}
            CMD_RESET_SLOT => {
                let (ptr, len) = (cmd.b as *mut u8, cmd.c as usize);
                let n = sh.n_workers as usize;
                let (w, chunk) = (init.idx as usize, len.div_ceil(n));
                let lo = (w * chunk).min(len);
                let hi = ((w + 1) * chunk).min(len);
                if hi > lo {
                    // SAFETY: the host guarantees `[ptr, ptr+len)` is writable
                    // and unshared with any run for the duration of the barrier.
                    unsafe { std::ptr::write_bytes(ptr.add(lo), 0, hi - lo) };
                }
            }
            CMD_BARRIER => {
                sh.fb.barrier_ack.fetch_add(1, Ordering::AcqRel);
            }
            CMD_STOP => return,
            _ => {}
        }
    }
}

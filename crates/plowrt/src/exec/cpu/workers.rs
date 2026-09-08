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
    run_gq, run_static, wait_until, Exec, GqState, LoadedProgram, Parker, RunShared, StaticState,
    WorkerCtx,
};
use crate::exec::cpu::topology::{NumaMode, Topology};

/// State every worker shares (read-mostly; the hot path touches counters, one
/// `cancel_gen` load, and the parkers' `sleepers`).
struct Shared {
    ring: ControlRing,
    /// Serializes the host side of the single-producer control ring.
    producer: Mutex<()>,
    fb: Feedback,
    parkers: Vec<Parker>,
    /// Workers per node, in placement order — the input `cu_map` needs to re-derive ownership for a
    /// program that runs on fewer workers than the pool has.
    pub per_node: Vec<Vec<u32>>,
    /// GQ cursors sized for the largest program run so far. Host-resized under
    /// the run lock before a run; workers only see it through `RunShared`.
    cursors: Mutex<Arc<Vec<CachePadded<AtomicU32>>>>,
    spin_us: u32,
    n_workers: u32,
}

/// cu -> worker ownership, then round-robin within the node, restricted to the first `active`
/// workers. Used both at spawn (for the pool's own bookkeeping) and per program, so prefill and
/// decode can run on different widths without respawning threads.
///
/// `node_of_cu` is the packet's locality plan (`engine::node_plan`) when the blob carries L2
/// domains and they divide over the nodes; without it the node is `cu % nodes`, which spreads
/// evenly but is blind to which slices actually feed each other.
pub fn cu_map(
    n_cu: u32,
    per_node: &[Vec<u32>],
    nodes: usize,
    active: usize,
    node_of_cu: Option<&[u32]>,
) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> =
        vec![Vec::new(); per_node.iter().map(Vec::len).sum::<usize>().max(active)];
    let live: Vec<Vec<u32>> = per_node
        .iter()
        .map(|ws| {
            ws.iter()
                .copied()
                .filter(|&w| (w as usize) < active)
                .collect()
        })
        .collect();
    // Per-node round-robin cursor. Under the fallback `cu % nodes` the k-th cu on a node is
    // `k * nodes + np`, so this counts exactly what `cu / nodes` used to.
    let mut seen = vec![0usize; nodes.max(1)];
    for cu in 0..n_cu {
        let np = match node_of_cu {
            Some(m) => m[cu as usize] as usize % nodes.max(1),
            None => (cu as usize) % nodes.max(1),
        };
        let ws = if live.get(np).is_some_and(|v| !v.is_empty()) {
            &live[np]
        } else {
            match live.iter().find(|v| !v.is_empty()) {
                Some(v) => v,
                None => return out, // no active worker at all; caller rejects
            }
        };
        let w = ws[seen[np] % ws.len()];
        seen[np] += 1;
        out[w as usize].push(cu);
    }
    out
}

struct WorkerInit {
    cus: Vec<u32>,
    idx: u32,
    node: u32,
    node_pos: u32,
    cpu: u32,
    /// GQ window this worker claims from first. The packet's locality domain when the blob carries
    /// one and it maps onto the nodes, else the node position, as before.
    domain: u32,
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
    /// Spawn `threads` persistent workers (0 = one per online logical cpu on the
    /// selected nodes). Virtual executor `cu` is placed on the node `place` names for it, or on
    /// `cu % nodes` without one, round-robin across that node's workers — so `n_cu` need not equal
    /// the thread count, and one blob serves any core count.
    ///
    /// `place` is the packet's locality plan: `(node_of_cu, domains)` from `engine::node_plan`.
    /// It moves same-domain cus onto one node and points each worker's first GQ claim at a domain
    /// that node owns, instead of at its bare node index.
    pub fn spawn(
        topo: &Topology,
        threads: usize,
        numa: &NumaMode,
        spin_us: u32,
        n_cu: u32,
        place: Option<(&[u32], u32)>,
        exec: Arc<dyn Exec>,
    ) -> WorkerPool {
        let nodes = topo.select_nodes(numa);
        let mut cpus = topo.worker_cpus(&nodes);
        if cpus.is_empty() {
            cpus.push((0, nodes[0]));
        }
        let threads = if threads == 0 {
            cpus.len()
        } else {
            threads.max(1)
        };
        // Round-robin over logical cpus when oversubscribed.
        let placement: Vec<(u32, u32)> = (0..threads).map(|k| cpus[k % cpus.len()]).collect();
        let node_pos = |n: u32| nodes.iter().position(|&x| x == n).unwrap_or(0) as u32;

        // cu → worker: the locality plan's node (else cu % nodes), then round-robin within it.
        let mut per_node: Vec<Vec<u32>> = vec![Vec::new(); nodes.len()];
        for (w, &(_, n)) in placement.iter().enumerate() {
            per_node[node_pos(n) as usize].push(w as u32);
        }
        let mut cus_of = cu_map(n_cu, &per_node, nodes.len(), threads, place.map(|(m, _)| m));

        // Each node owns a contiguous run of `per` domains under `node_plan`; spread that node's
        // workers over them, so no window is left to be reached only by stealing. `spawn` resolves
        // the node list itself, so re-derive the split here instead of trusting the caller's.
        let per = place
            .map(|(_, domains)| domains as usize)
            .filter(|d| *d >= nodes.len() && d.is_multiple_of(nodes.len()))
            .map(|d| d / nodes.len());
        let mut worker_domain = vec![0u32; threads];
        for (p, ws) in per_node.iter().enumerate() {
            for (k, &w) in ws.iter().enumerate() {
                worker_domain[w as usize] = match per {
                    Some(per) => (p * per + k % per) as u32,
                    None => p as u32,
                };
            }
        }

        let shared = Arc::new(Shared {
            ring: ControlRing::new(threads),
            producer: Mutex::new(()),
            fb: Feedback::default(),
            parkers: (0..nodes.len()).map(|_| Parker::default()).collect(),
            per_node: per_node.clone(),
            cursors: Mutex::new(Arc::new(Vec::new())),
            spin_us,
            n_workers: threads as u32,
        });

        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::with_capacity(threads);
        let mut worker_node = Vec::with_capacity(threads);
        for (w, &(cpu, node)) in placement.iter().enumerate() {
            worker_node.push(node);
            let init = WorkerInit {
                cus: std::mem::take(&mut cus_of[w]),
                idx: w as u32,
                node,
                node_pos: node_pos(node),
                domain: worker_domain[w],
                cpu,
            };
            let sh = shared.clone();
            let ex = exec.clone();
            let h = match std::thread::Builder::new()
                .name(format!("plow-cpu-{node}-{w}"))
                .spawn(move || worker_main(init, sh, ex))
            {
                Ok(h) => h,
                Err(e) => {
                    shared.ring.push(Cmd::stop());
                    for p in &shared.parkers {
                        p.unpark_all();
                    }
                    for h in handles.drain(..) {
                        let _ = h.join();
                    }
                    panic!("spawn cpu worker: {e}");
                }
            };
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
        fb.cancel_gen.store(0, Ordering::Release);
        host.inflight = Some(Inflight {
            gen,
            _prog: prog.clone(),
            _pool: counters.clone(),
            _cursors: cursors,
        });
        {
            let _producer = self.shared.producer.lock();
            self.shared.ring.push(Cmd::run(
                gen,
                Arc::as_ptr(prog) as u64,
                Arc::as_ptr(counters) as u64,
                seg,
            ));
            self.wake_all();
        }
        gen
    }

    /// Block until every worker has drained (or abandoned) run `gen`.
    /// Returns the first fault recorded, if any.
    /// Workers per node in placement order, so a program can be loaded for a narrower set.
    pub fn per_node(&self) -> &[Vec<u32>] {
        &self.shared.per_node
    }

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
        let _producer = self.shared.producer.lock();
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
        {
            let _producer = self.shared.producer.lock();
            self.shared.ring.push(Cmd::barrier(seq));
            self.wake_all();
        }
        self.wait_barrier();
    }

    fn wait_barrier(&self) {
        let fb = &self.shared.fb;
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
        let seq = {
            let mut host = self.host.lock();
            host.barrier_seq = host.barrier_seq.wrapping_add(1);
            host.barrier_seq
        };
        self.shared.fb.barrier_ack.store(0, Ordering::Release);
        {
            let _producer = self.shared.producer.lock();
            for &(p, len) in ranges {
                if len == 0 {
                    continue;
                }
                self.shared
                    .ring
                    .push(Cmd::reset_slot(slot, p as u64, len as u64));
            }
            self.shared.ring.push(Cmd::barrier(seq));
            self.wake_all();
        }
        self.wait_barrier();
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
        let gen = self.host.get_mut().inflight.as_ref().map(|run| run.gen);
        if let Some(gen) = gen {
            self.shared.fb.cancel_gen.store(gen, Ordering::Release);
        }
        {
            let _producer = self.shared.producer.lock();
            if let Some(gen) = gen {
                self.shared.ring.push(Cmd::cancel(gen));
            }
            self.shared.ring.push(Cmd::stop());
            self.wake_all();
        }
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
        if cpu as usize >= libc::CPU_SETSIZE as usize {
            tracing::warn!(cpu, "CPU id exceeds affinity mask capacity");
            return;
        }
        libc::CPU_SET(cpu as usize, &mut set);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            tracing::warn!(cpu, error = %std::io::Error::last_os_error(), "CPU affinity failed");
        }
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
        domain: init.domain,
    };
    let mut st = StaticState::new(init.idx as usize, init.cus);
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
                if crate::exec::cpu::interp::TRACE_ON.load(Ordering::Relaxed) {
                    crate::exec::cpu::interp::trace_flush();
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

#[cfg(test)]
mod placement_tests {
    use super::cu_map;

    /// Workers laid out over `nodes`, node-major, as `spawn` builds `per_node`.
    fn per_node(nodes: usize, per: usize) -> Vec<Vec<u32>> {
        (0..nodes)
            .map(|n| ((n * per) as u32..((n + 1) * per) as u32).collect())
            .collect()
    }

    /// Without a plan the node is `cu % nodes` and the worker within it is `cu / nodes` — the
    /// mapping this pool has always used. Pinned so the locality plan stays purely additive.
    #[test]
    fn without_a_plan_placement_is_the_old_round_robin() {
        let pn = per_node(2, 3);
        let got = cu_map(24, &pn, 2, 6, None);
        for (w, cus) in got.iter().enumerate() {
            for &cu in cus {
                assert_eq!(cu as usize % 2, w / 3, "cu {cu} on the wrong node");
                assert_eq!((cu as usize / 2) % 3, w % 3, "cu {cu} on the wrong worker");
            }
        }
        assert_eq!(got.iter().map(Vec::len).sum::<usize>(), 24);
        assert!(got.iter().all(|c| c.len() == 4), "even by construction");
    }

    /// With a plan every cu of one domain lands on that domain's node, and the nodes still split
    /// the work evenly — locality must not cost balance.
    #[test]
    fn a_plan_groups_domains_without_costing_balance() {
        let pn = per_node(2, 3);
        let plan: Vec<u32> = (0..24u32).map(|cu| (cu % 8) / 4).collect();
        let got = cu_map(24, &pn, 2, 6, Some(&plan));
        for (w, cus) in got.iter().enumerate() {
            for &cu in cus {
                assert_eq!(plan[cu as usize] as usize, w / 3, "cu {cu} left its node");
            }
        }
        assert_eq!(got.iter().map(Vec::len).sum::<usize>(), 24);
        assert!(
            got.iter().all(|c| c.len() == 4),
            "3 workers x 12 cus per node"
        );
    }

    /// A node with no live worker still has its cus placed somewhere rather than dropped.
    #[test]
    fn a_plan_survives_a_node_with_no_active_worker() {
        let pn = per_node(2, 3);
        let plan: Vec<u32> = (0..12u32).map(|cu| cu % 2).collect();
        let got = cu_map(12, &pn, 2, 3, Some(&plan));
        assert_eq!(got.iter().map(Vec::len).sum::<usize>(), 12);
        assert!(
            got[3..].iter().all(Vec::is_empty),
            "workers past `active` own nothing"
        );
    }
}

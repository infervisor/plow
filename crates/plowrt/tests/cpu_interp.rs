//! P1 checks for the CPU interpreter + persistent worker pool: dependency
//! ordering under every thread count, no deadlock when threads < n_cu, cancel,
//! reset, barrier, and persistence across many back-to-back runs.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use packet::dev::{DevInst64, StreamEnt, Wait};
use plowrt::exec::counters::CounterPool;
use plowrt::exec::cpu::interp::{Exec, GlobalQueue, LoadedProgram, WorkerCtx};
use plowrt::exec::cpu::topology::{NumaMode, Topology};
use plowrt::exec::cpu::workers::WorkerPool;

/// One op: `blocks` slices, depends on `deps` (op indices), tagged `seg`.
struct Op {
    blocks: u16,
    deps: Vec<usize>,
    seg: u16,
}

/// Build a program in topological op order: op `k` signals counter `k`
/// (threshold = its block count); slices are dealt round-robin over `n_cu`.
/// `gq_domains > 0` also emits an op-major global-queue stream split across
/// that many domain windows per segment.
fn build(ops: &[Op], n_cu: u32, n_seg: u32, gq_domains: u32) -> (LoadedProgram, Vec<packet::Counter>) {
    let mut insts = Vec::new();
    let mut waits = Vec::new();
    let mut succs = Vec::new();
    let mut per_cu: Vec<Vec<StreamEnt>> = vec![Vec::new(); n_cu as usize];
    let mut opmajor: Vec<StreamEnt> = Vec::new();
    for (k, op) in ops.iter().enumerate() {
        let mut inst = DevInst64::default();
        inst.op = (k % 7) as u16 + 1;
        inst.blocks = op.blocks;
        insts.push(inst);
        let wait_ofs = waits.len() as u32;
        for &d in &op.deps {
            waits.push(Wait {
                id: d as u32,
                threshold: ops[d].blocks as u32,
            });
        }
        let succ_ofs = succs.len() as u32;
        succs.push(k as u32);
        for s in 0..op.blocks as u32 {
            let e = StreamEnt {
                inst: k as u32,
                slice: s,
                wait_ofs,
                succ_ofs,
                wait_len: op.deps.len() as u16,
                succ_len: 1,
                flags: 0,
                seg: op.seg,
            };
            per_cu[((k as u32 + s) % n_cu) as usize].push(e);
            opmajor.push(e);
        }
    }
    let mut stream = Vec::new();
    let mut stream_ofs = Vec::new();
    let mut stream_len = Vec::new();
    for s in &per_cu {
        stream_ofs.push(stream.len() as u32);
        stream_len.push(s.len() as u32);
        stream.extend_from_slice(s);
    }
    let gq = (gq_domains > 0).then(|| {
        // Window (seg, d): entries of that seg dealt round-robin over domains,
        // op-major order preserved inside each window.
        let mut gstream = Vec::new();
        let mut seg_ofs = vec![0u32];
        for seg in 0..n_seg.max(1) {
            for d in 0..gq_domains {
                let mut i = 0u32;
                for e in &opmajor {
                    if e.seg as u32 == seg {
                        if i % gq_domains == d {
                            gstream.push(*e);
                        }
                        i += 1;
                    }
                }
                seg_ofs.push(gstream.len() as u32);
            }
        }
        GlobalQueue {
            stream: gstream,
            seg_ofs,
            domains: gq_domains,
        }
    });
    let counters = (0..ops.len())
        .map(|k| packet::Counter {
            id: k as u32,
            threshold: ops[k].blocks as u32,
            scope: 1,
            _pad: [0; 3],
        })
        .collect();
    (
        LoadedProgram {
            insts,
            stream,
            stream_ofs,
            stream_len,
            waits,
            succs,
            n_cu,
            n_seg,
            seg_ofs: None,
            gq,
        },
        counters,
    )
}

/// Records every (inst, slice) firing and checks that all producer slices had
/// fired first (reads their markers with Acquire).
struct Recorder {
    deps: Vec<Vec<usize>>,
    blocks: Vec<u16>,
    /// `fired[inst][slice]`
    fired: Vec<Vec<AtomicU32>>,
    violations: AtomicUsize,
    count: AtomicUsize,
    delay: Duration,
}

impl Recorder {
    fn new(ops: &[Op], delay: Duration) -> Recorder {
        Recorder {
            deps: ops.iter().map(|o| o.deps.clone()).collect(),
            blocks: ops.iter().map(|o| o.blocks).collect(),
            fired: ops
                .iter()
                .map(|o| (0..o.blocks).map(|_| AtomicU32::new(0)).collect())
                .collect(),
            violations: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            delay,
        }
    }
    fn reset(&self) {
        for v in &self.fired {
            for a in v {
                a.store(0, Ordering::Relaxed);
            }
        }
        self.count.store(0, Ordering::Relaxed);
    }
}

impl Exec for Recorder {
    fn exec(&self, inst: &DevInst64, slice: u32, nblk: u32, _w: &WorkerCtx) {
        let idx = self.locate(inst);
        for &d in &self.deps[idx] {
            for s in 0..self.blocks[d] as usize {
                if self.fired[d][s].load(Ordering::Acquire) == 0 {
                    self.violations.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        assert_eq!(nblk, inst.blocks as u32);
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        self.fired[idx][slice as usize].store(1, Ordering::Release);
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

impl Recorder {
    /// The interpreter passes `&insts[k]`; recover `k` from the instruction's
    /// address relative to the program's table (set per run via `base`).
    fn locate(&self, inst: &DevInst64) -> usize {
        let base = BASE.load(Ordering::Acquire) as *const DevInst64;
        let p = inst as *const DevInst64;
        // SAFETY: both point into the same `insts` Vec of the running program.
        unsafe { p.offset_from(base) as usize }
    }
}

static BASE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn pool(threads: usize, n_cu: u32, exec: Arc<dyn Exec>) -> WorkerPool {
    let topo = Topology::detect();
    WorkerPool::spawn(&topo, threads, &NumaMode::Off, 20, n_cu, exec)
}

fn run_once(p: &WorkerPool, prog: &Arc<LoadedProgram>, pool: &Arc<CounterPool>, seg: u32) {
    pool.reset_all();
    BASE.store(prog.insts.as_ptr() as usize, Ordering::Release);
    let gen = p.run(prog, seg, pool);
    assert_eq!(p.wait_done(gen), None, "fault");
}

/// fan-out: op0 (1 slice) → op1..op4 (4 slices each) → op5 (1 slice) waits on all.
fn fan_ops() -> Vec<Op> {
    let mut ops = vec![Op { blocks: 1, deps: vec![], seg: 0 }];
    for _ in 0..4 {
        ops.push(Op { blocks: 4, deps: vec![0], seg: 0 });
    }
    ops.push(Op { blocks: 1, deps: vec![1, 2, 3, 4], seg: 0 });
    ops
}

/// diamond chain: a → (b, c) → d, repeated 8 times, each layer sliced 6-way.
fn diamond_ops() -> Vec<Op> {
    let mut ops = Vec::new();
    let mut prev: Option<usize> = None;
    for _ in 0..8 {
        let a = ops.len();
        ops.push(Op { blocks: 6, deps: prev.into_iter().collect(), seg: 0 });
        ops.push(Op { blocks: 6, deps: vec![a], seg: 0 });
        ops.push(Op { blocks: 6, deps: vec![a], seg: 0 });
        ops.push(Op { blocks: 6, deps: vec![a + 1, a + 2], seg: 0 });
        prev = Some(a + 3);
    }
    ops
}

fn total_slices(ops: &[Op]) -> usize {
    ops.iter().map(|o| o.blocks as usize).sum()
}

#[test]
fn static_mode_orders_dependencies_on_every_thread_count() {
    for ops in [fan_ops(), diamond_ops()] {
        let (prog, ctrs) = build(&ops, 16, 1, 0);
        let prog = Arc::new(prog);
        let ctr = Arc::new(CounterPool::from_counters(&ctrs));
        for threads in [1usize, 4, 16] {
            let rec = Arc::new(Recorder::new(&ops, Duration::ZERO));
            let p = pool(threads, 16, rec.clone());
            for _ in 0..20 {
                rec.reset();
                run_once(&p, &prog, &ctr, 0);
                assert_eq!(rec.count.load(Ordering::Relaxed), total_slices(&ops));
            }
            assert_eq!(rec.violations.load(Ordering::Relaxed), 0, "threads={threads}");
        }
    }
}

#[test]
fn threads_fewer_than_cus_does_not_deadlock() {
    // 2 threads own 16 streams each holding heads that depend on other streams.
    let ops = diamond_ops();
    let (prog, ctrs) = build(&ops, 16, 1, 0);
    let prog = Arc::new(prog);
    let ctr = Arc::new(CounterPool::from_counters(&ctrs));
    let rec = Arc::new(Recorder::new(&ops, Duration::ZERO));
    let p = pool(2, 16, rec.clone());
    for _ in 0..50 {
        rec.reset();
        run_once(&p, &prog, &ctr, 0);
        assert_eq!(rec.count.load(Ordering::Relaxed), total_slices(&ops));
    }
    assert_eq!(rec.violations.load(Ordering::Relaxed), 0);
}

#[test]
fn global_queue_two_domains_orders_dependencies() {
    let ops = diamond_ops();
    let (prog, ctrs) = build(&ops, 16, 1, 2);
    assert!(prog.gq.is_some());
    let prog = Arc::new(prog);
    let ctr = Arc::new(CounterPool::from_counters(&ctrs));
    for threads in [1usize, 4, 16] {
        let rec = Arc::new(Recorder::new(&ops, Duration::ZERO));
        let p = pool(threads, 16, rec.clone());
        for _ in 0..20 {
            rec.reset();
            run_once(&p, &prog, &ctr, 0);
            assert_eq!(rec.count.load(Ordering::Relaxed), total_slices(&ops));
        }
        assert_eq!(rec.violations.load(Ordering::Relaxed), 0, "threads={threads}");
    }
}

#[test]
fn segments_run_one_at_a_time_with_seg_filter() {
    // seg 0: op0 → op1 ; seg 1: op2 (depends on op1) → op3.
    let ops = vec![
        Op { blocks: 4, deps: vec![], seg: 0 },
        Op { blocks: 4, deps: vec![0], seg: 0 },
        Op { blocks: 4, deps: vec![1], seg: 1 },
        Op { blocks: 4, deps: vec![2], seg: 1 },
    ];
    let (prog, ctrs) = build(&ops, 8, 2, 0);
    let prog = Arc::new(prog);
    let ctr = Arc::new(CounterPool::from_counters(&ctrs));
    let rec = Arc::new(Recorder::new(&ops, Duration::ZERO));
    let p = pool(4, 8, rec.clone());
    rec.reset();
    ctr.reset_all();
    BASE.store(prog.insts.as_ptr() as usize, Ordering::Release);
    let g = p.run(&prog, 0, &ctr);
    assert_eq!(p.wait_done(g), None);
    assert_eq!(rec.count.load(Ordering::Relaxed), 8, "only seg 0 ran");
    // Counters carry over between segments (not reset), as on the GPU relaunch.
    let g = p.run(&prog, 1, &ctr);
    assert_eq!(p.wait_done(g), None);
    assert_eq!(rec.count.load(Ordering::Relaxed), 16);
    assert_eq!(rec.violations.load(Ordering::Relaxed), 0);
}

#[test]
fn cancel_returns_pool_to_idle_and_next_run_completes() {
    // A long serial chain with a slow mock: cancel early, then run fully.
    let ops: Vec<Op> = (0..64)
        .map(|k| Op { blocks: 1, deps: if k == 0 { vec![] } else { vec![k - 1] }, seg: 0 })
        .collect();
    let (prog, ctrs) = build(&ops, 4, 1, 0);
    let prog = Arc::new(prog);
    let ctr = Arc::new(CounterPool::from_counters(&ctrs));
    let rec = Arc::new(Recorder::new(&ops, Duration::from_millis(1)));
    let p = pool(4, 4, rec.clone());
    rec.reset();
    ctr.reset_all();
    BASE.store(prog.insts.as_ptr() as usize, Ordering::Release);
    let g = p.run(&prog, 0, &ctr);
    std::thread::sleep(Duration::from_millis(5));
    p.cancel(g);
    assert_eq!(p.wait_done(g), None);
    let partial = rec.count.load(Ordering::Relaxed);
    assert!(partial < 64, "cancel should stop the chain early, ran {partial}");
    // Pool is idle and reusable.
    rec.reset();
    run_once(&p, &prog, &ctr, 0);
    assert_eq!(rec.count.load(Ordering::Relaxed), 64);
    assert_eq!(rec.violations.load(Ordering::Relaxed), 0);
}

#[test]
fn reset_slot_zeroes_through_workers() {
    let ops = fan_ops();
    let rec = Arc::new(Recorder::new(&ops, Duration::ZERO));
    let p = pool(4, 4, rec);
    let mut buf = vec![0xFFu8; 1 << 20];
    let mut buf2 = vec![0xAAu8; 12345];
    unsafe {
        p.reset_slot(3, &[(buf.as_mut_ptr(), buf.len()), (buf2.as_mut_ptr(), buf2.len())]);
    }
    assert!(buf.iter().all(|&b| b == 0));
    assert!(buf2.iter().all(|&b| b == 0));
    p.barrier();
}

#[test]
fn pool_persists_across_many_runs() {
    let ops = fan_ops();
    let (prog, ctrs) = build(&ops, 8, 1, 0);
    let prog = Arc::new(prog);
    let ctr = Arc::new(CounterPool::from_counters(&ctrs));
    let rec = Arc::new(Recorder::new(&ops, Duration::ZERO));
    let p = pool(8, 8, rec.clone());
    for i in 0..1000 {
        rec.reset();
        run_once(&p, &prog, &ctr, 0);
        assert_eq!(rec.count.load(Ordering::Relaxed), total_slices(&ops), "run {i}");
        // Every worker reported done ⇒ every thread is still alive.
        assert_eq!(p.feedback().done.load(Ordering::Acquire), 8);
    }
    assert_eq!(rec.violations.load(Ordering::Relaxed), 0);
    assert_eq!(p.threads(), 8);
}

#[test]
fn topology_fixture_and_detect() {
    let t = Topology::from_sysfs_text(
        "0-7",
        &[(0, "0,4"), (1, "1,5"), (2, "2,6"), (3, "3,7"), (4, "0,4"), (5, "1,5"), (6, "2,6"), (7, "3,7")],
        &[(0, "0-1,4-5"), (1, "2-3,6-7")],
    );
    assert_eq!(t.physical_cores(), 4);
    assert_eq!(t.cores_on_node(1).map(|c| c.cpu).collect::<Vec<_>>(), vec![2, 3]);
    let live = Topology::detect();
    assert!(live.physical_cores() >= 1);
    assert!(!live.nodes.is_empty());
}

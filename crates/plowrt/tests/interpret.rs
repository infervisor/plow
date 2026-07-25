//! CPU reference interpreter: counter-gated execution + deadlock detection.

use packet::{Body, Counter, Inst, Program, ResourceKind};
use plowrt::device::cpu::interpret;
use plowrt::exec::counters::CounterPool;
use plowrt::exec::health::CounterMonitor;

/// Two executors, one dependency edge: A (on SM 0) produces counter 0; B (on
/// SM 1) waits for it. B must fire only after A — and both must fire.
fn producer_consumer(threshold: u32) -> Program {
    let a = Inst {
        resource: ResourceKind::Sm,
        unit: 0,
        index: 0,
        body: Body::Host,
        wait: vec![],
        succ: vec![0],
    };
    let b = Inst {
        resource: ResourceKind::Sm,
        unit: 0,
        index: 1,
        body: Body::Host,
        wait: vec![0],
        succ: vec![],
    };
    Program {
        insts: vec![a, b],
        counters: vec![Counter {
            id: 0,
            threshold,
            scope: 1,
            _pad: [0; 3],
        }],
        bucket_id: 0,
        plan_gen: 0,
        flags: 0,
    }
}

#[test]
fn interprets_to_completion() {
    let prog = producer_consumer(1);
    let pool = CounterPool::from_counters(&prog.counters);
    let stats = interpret(&prog, &pool);
    assert!(stats.completed, "schedule should complete");
    assert_eq!(stats.executed, 2);
    assert!(pool.satisfied(0));
}

#[test]
fn dropped_increment_is_deadlock() {
    // Threshold 2 but only one instruction ever increments counter 0 — a dropped
    // increment / mis-scoped atomic. The consumer can never fire.
    let prog = producer_consumer(2);
    let pool = CounterPool::from_counters(&prog.counters);

    let stats = interpret(&prog, &pool);
    assert!(!stats.completed, "should deadlock: consumer never unblocks");
    assert_eq!(stats.executed, 1, "only the producer fires");

    // The counter-space monitor flags it statically, precisely.
    let monitor = CounterMonitor::new(&prog);
    let reports = monitor.unsatisfiable(&pool);
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].counter, 0);
    assert_eq!(reports[0].threshold, 2);
    assert_eq!(reports[0].max_possible, 1);
}

#[test]
fn progress_reaches_one() {
    let prog = producer_consumer(1);
    let pool = CounterPool::from_counters(&prog.counters);
    let mut monitor = CounterMonitor::new(&prog);
    assert!(monitor.progress(&pool) < 1.0);
    let _ = interpret(&prog, &pool);
    assert!((monitor.progress(&pool) - 1.0).abs() < 1e-9);
}

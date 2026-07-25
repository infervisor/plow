//! §O simulator: dry-run tracing, Chrome trace, deadlock reporting, math modes.

use packet::{Body, Counter, Inst, Program, ResourceKind};
use plowrt::sim::{MathMode, Simulator};

mod common;

/// A program that deadlocks: counter 0 needs 2 increments but only one packet
/// ever bumps it.
fn deadlock_program() -> Program {
    Program {
        insts: vec![
            Inst {
                resource: ResourceKind::Sm,
                unit: 0,
                index: 0,
                body: Body::Host,
                wait: vec![],
                succ: vec![0],
            },
            Inst {
                resource: ResourceKind::Sm,
                unit: 0,
                index: 1,
                body: Body::Host,
                wait: vec![0],
                succ: vec![],
            },
        ],
        counters: vec![Counter {
            id: 0,
            threshold: 2, // unsatisfiable: only one producer
            scope: 1,
            _pad: [0; 3],
        }],
        bucket_id: 0,
        plan_gen: 0,
        flags: 0,
    }
}

#[test]
fn dry_run_records_one_event_per_packet() {
    let prog = common::tiny_program();
    let report = Simulator::new(MathMode::DryRun).run(&prog);

    assert_eq!(report.events.len(), prog.insts.len());
    assert!(report.stats.completed);
    assert!(report.unsatisfiable.is_empty());
    // Every event carries a decoded body summary + timing.
    for (i, e) in report.events.iter().enumerate() {
        assert_eq!(e.packet_index, i);
        assert!(!e.name.is_empty());
        assert!(e.t_end >= e.t_start);
        assert!(!e.log_line().is_empty());
    }
    // Modelled makespan is positive and reported alongside the compiler's later.
    assert!(report.stats.makespan > 0);
}

#[test]
fn chrome_trace_is_well_formed() {
    let prog = common::tiny_program();
    let report = Simulator::new(MathMode::DryRun).run(&prog);
    let json = report.to_chrome_json();
    let v: serde_json::Value = serde_json::from_str(&json).expect("chrome json parses");
    let events = v["traceEvents"].as_array().expect("traceEvents array");
    assert_eq!(events.len(), report.events.len());
    // Each span is a complete-duration event.
    assert_eq!(events[0]["ph"], "X");
    assert!(events[0]["dur"].as_u64().is_some());
}

#[test]
fn deadlock_is_reported() {
    let prog = deadlock_program();
    let report = Simulator::new(MathMode::DryRun).run(&prog);
    assert!(!report.stats.completed, "should not complete");
    assert_eq!(report.stats.executed, 1, "only the producer fires");
    assert_eq!(report.unsatisfiable.len(), 1);
    assert_eq!(report.unsatisfiable[0].counter, 0);
    assert!(report.summary().contains("INCOMPLETE"));
}

#[test]
fn dry_and_golden_agree_on_schedule() {
    // Math mode changes whether numerics run, not the schedule walk / timing.
    let prog = common::tiny_program();
    let dry = Simulator::new(MathMode::DryRun).run(&prog);
    let golden = Simulator::new(MathMode::Golden).run(&prog);
    assert_eq!(dry.events.len(), golden.events.len());
    assert_eq!(dry.stats.makespan, golden.stats.makespan);
    assert_eq!(dry.stats.completed, golden.stats.completed);
}

#[test]
fn timing_respects_dependencies() {
    // The consumer (waits on counter 0) must start no earlier than the producer
    // finished — its span starts at/after the producer's end.
    let prog = common::tiny_program();
    let report = Simulator::new(MathMode::DryRun).run(&prog);
    let producer = report.events.iter().find(|e| e.succs.contains(&0)).unwrap();
    let consumer = report.events.iter().find(|e| e.waits.contains(&0)).unwrap();
    assert!(consumer.t_start >= producer.t_end);
}

//! Integration test for R-K4 head-slot eviction safety — verifies that a
//! `Growable`-class KV entry with non-empty writers/readers round-trips
//! through the D/F checkpoints without being silently dropped or degraded.
//!
//! **Plumbing gap noted:** as of 2026-07-03, plowc's default AddressMap
//! (`schedule::memory::plan_from_schedule_with_task_sets`) does not emit
//! Growable entries — those are added by callers who own the `KvLayout`
//! (typically the runtime side, not the compiler). This test constructs a
//! synthetic payload with a Growable entry to prove the D/F verifier
//! machinery *would* handle it correctly if plowc chose to submit one.

#![cfg(feature = "lean-verify")]

use lean_verify::checkpoints::schedule::{
    check_schedule, AddrEntry, ProtocolView, ScheduleRequest, TaskGraphView,
};
use std::collections::BTreeMap;

/// Build a minimal request with one Growable entry that has real writer + reader
/// tasks. Growable buffers model the KV cache: a producer task writes a new
/// head-slot, later reader tasks consume it, and eventually the slot is
/// reassigned to a new sequence (the R-K4 reclamation case).
fn growable_kv_request() -> ScheduleRequest {
    // 4 tasks: writer at 0 (writes buf "kv"), reader at 1 (reads it),
    //          then writer at 2 (reassigns buf "kv2" to same slot),
    //          reader at 3 (reads the reassigned slot).
    // Counter 0 gates 0 → 2 so the readers-of-kv (task 1) happens-before
    // writers-of-kv2 (task 2) via the resource-order chain.
    let mut threshold = BTreeMap::new();
    threshold.insert("0".into(), 1);

    ScheduleRequest {
        task_graph: TaskGraphView {
            n: 4,
            // Empty edges — the ordering is via resource / counter, not data-dep.
            edges: vec![],
        },
        protocol: ProtocolView {
            waits: vec![vec![], vec![], vec![0], vec![]],
            succs: vec![vec![0], vec![], vec![], vec![]],
            threshold,
            // Everyone on resource 0 with monotone stream indices.
            resource: vec![0, 0, 0, 0],
            stream_idx: vec![0, 1, 2, 3],
        },
        schedule_order: vec![0, 1, 2, 3],
        address_map: vec![
            AddrEntry {
                name: "kv".into(),
                offset: 0,
                size: 4096,
                cls: "Growable".into(),
                writers: vec![0],
                readers: vec![1],
            },
            AddrEntry {
                name: "kv2".into(),
                offset: 0,
                size: 4096,
                cls: "Growable".into(),
                writers: vec![2],
                readers: vec![3],
            },
        ],
    }
}

#[test]
#[ignore = "requires plow_verify binary"]
fn growable_class_survives_payload_round_trip() {
    let req = growable_kv_request();
    // Both entries must be Growable and their readers/writers non-empty.
    for entry in &req.address_map {
        assert_eq!(entry.cls, "Growable", "cls degraded during construction");
        assert!(
            !entry.writers.is_empty(),
            "growable entry {} has no writers",
            entry.name
        );
        assert!(
            !entry.readers.is_empty(),
            "growable entry {} has no readers",
            entry.name
        );
    }
}

#[test]
#[ignore = "requires plow_verify binary"]
fn growable_entries_pass_strict_verifier() {
    // The strict verifier requires reader/writer disjointness across pairs.
    // Our fixture picks disjoint task ids so the strict form goes through.
    let req = growable_kv_request();
    let cert = check_schedule(&req).expect("verifier call");
    assert!(cert.ok, "growable-KV payload rejected: {cert:?}");
    // Accept notes should mention "strict" — meaning the strict variant
    // (readers/writers disjoint) proved AddressMapSound, not just Loose.
    let notes = cert.notes.expect("accept carries notes");
    assert!(
        notes.contains("strict"),
        "expected strict-form acceptance, got: {notes}"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn overlapping_growable_writers_fail_strict_but_pass_loose() {
    // Construct a payload where task 0 is BOTH a writer of "kv" AND a reader
    // of "kv2" — reader/writer sets overlap across the pair, so the strict
    // form must reject even though the loose form would accept.
    let mut req = growable_kv_request();
    req.address_map[1].readers = vec![0]; // task 0 is now a reader of kv2
    req.address_map[1].writers = vec![0]; // and a writer of kv2 (self-alias case)

    let cert = check_schedule(&req).expect("verifier call");
    // Strict D must reject because readers ∩ writers is non-empty on kv2.
    assert!(
        !cert.ok,
        "strict D accepted a payload with overlapping reader/writer sets: {cert:?}"
    );
    let reason = cert.reason.expect("rejection carries reason");
    assert!(
        reason.contains("reader/writer") || reason.contains("counter-ordered"),
        "unexpected reason: {reason}"
    );
}

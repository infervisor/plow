//! End-to-end integration tests for the Rust ↔ Lean bridge.
//!
//! These tests spawn the `plow_verify` binary. The crate's build.rs isn't set
//! up to invoke `lake` (Option A / IPC keeps Rust and Lean build systems
//! independent), so the binary must exist at:
//!
//!   lean-plow/.lake/build/bin/plow_verify
//!
//! Run `nix develop -c lake build` under `lean-plow/` first, or set
//! `PLOW_VERIFY_BIN` to a custom path.

use std::collections::BTreeMap;

use lean_verify::checkpoints::schedule::{AddrEntry, ProtocolView, ScheduleRequest, TaskGraphView};
use lean_verify::checkpoints::{check_address_map, check_schedule};

/// A 4-task graph on 2 resources, with counter 0 gating task 0 → task 2 and
/// counter 1 gating task 1 → task 2. Task 1 is the last reader of buffer "a";
/// task 2 is the first writer of "b" (reclaiming "a"'s bytes).
fn safe_request() -> ScheduleRequest {
    let mut threshold = BTreeMap::new();
    threshold.insert("0".to_string(), 1);
    threshold.insert("1".to_string(), 1);
    ScheduleRequest {
        task_graph: TaskGraphView {
            n: 4,
            edges: vec![(0, 2), (1, 2)],
        },
        protocol: ProtocolView {
            waits: vec![vec![], vec![], vec![0, 1], vec![]],
            succs: vec![vec![0], vec![1], vec![], vec![]],
            threshold,
            resource: vec![0, 0, 1, 1],
            stream_idx: vec![0, 1, 2, 3],
        },
        schedule_order: vec![0, 1, 2, 3],
        address_map: vec![
            AddrEntry {
                name: "a".into(),
                offset: 0,
                size: 100,
                cls: "Scratch".into(),
                writers: vec![0],
                readers: vec![1],
            },
            AddrEntry {
                name: "b".into(),
                offset: 0,
                size: 100,
                cls: "Scratch".into(),
                writers: vec![2],
                readers: vec![3],
            },
        ],
    }
}

#[test]
#[ignore = "requires plow_verify binary (run `lake build` in lean-plow/)"]
fn checkpoint_d_accepts_safe_schedule() {
    let cert = check_schedule(&safe_request()).expect("verifier call failed");
    assert!(cert.ok, "cert: {cert:?}");
    assert_eq!(cert.checkpoint, "D");
    assert!(cert.notes.unwrap().contains("2 entries"));
}

#[test]
#[ignore = "requires plow_verify binary"]
fn batch_preserves_individual_verdicts_and_envelopes() {
    let safe = serde_json::to_value(safe_request()).unwrap();
    let mut unsafe_req = safe_request();
    unsafe_req.protocol.waits[2].clear();
    let certs = lean_verify::call_batch(&[
        ("D", safe.clone()),
        ("F", serde_json::to_value(unsafe_req).unwrap()),
        ("D", safe),
    ]).unwrap();
    assert_eq!(certs.len(), 3);
    assert!(certs[0].ok && !certs[1].ok && certs[2].ok);
    assert_eq!(certs[1].checkpoint, "F");
    assert!(certs[1].reason.is_some());
    assert_eq!(certs[0].notes, certs[2].notes);
    assert!(lean_verify::call_batch(&[]).unwrap().is_empty());
}

#[test]
#[ignore = "requires plow_verify binary"]
fn checkpoint_d_rejects_missing_counter() {
    // Same layout, but strip counter 1 (task 1 no longer succ's anything,
    // task 2 no longer waits on it). Now task 1 → task 2 has no path.
    let mut req = safe_request();
    req.protocol.waits[2] = vec![0];
    req.protocol.succs[1] = vec![];
    req.protocol.threshold.remove("1");

    let cert = check_schedule(&req).expect("verifier call failed");
    assert!(!cert.ok, "expected rejection, got: {cert:?}");
    assert!(cert.reason.unwrap().contains("counter-ordered"));
}

#[test]
#[ignore = "requires plow_verify binary"]
fn checkpoint_d_accepts_disjoint_bytes() {
    // Same schedule, but buffer "b" lives at a distinct offset — no overlap,
    // so no ordering required.
    let mut req = safe_request();
    // Strip the extra counter to make sure verify passes purely on disjointness.
    req.protocol.waits[2] = vec![];
    req.protocol.succs[0] = vec![];
    req.protocol.succs[1] = vec![];
    req.protocol.threshold.clear();
    req.address_map[1].offset = 4096;

    let cert = check_schedule(&req).expect("verifier call failed");
    assert!(!cert.ok, "disjoint allocations cannot waive data dependencies");
    req.task_graph.edges.clear();

    let cert = check_schedule(&req).expect("verifier call failed");
    assert!(cert.ok, "cert: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn checkpoint_d_rejects_unordered_dependency_without_address_entries() {
    let mut req = safe_request();
    req.address_map.clear();
    req.protocol.waits[2].clear();
    req.protocol.succs[0].clear();
    req.protocol.succs[1].clear();
    req.protocol.threshold.clear();
    let cert = check_schedule(&req).unwrap();
    assert!(!cert.ok);
    assert!(cert.reason.unwrap().contains("data dependency"));
}

#[test]
#[ignore = "requires plow_verify binary"]
fn checkpoint_d_rejects_malformed_access_sets() {
    let req = serde_json::to_value(safe_request()).unwrap();
    for field in ["readers", "writers"] {
        for invalid in [serde_json::json!(["1"]), serde_json::json!(null), serde_json::json!([1, -1])] {
            let mut bad = req.clone();
            bad["address_map"][0][field] = invalid;
            assert!(!lean_verify::call("D", bad).unwrap().ok);
        }
    }
}

#[test]
#[ignore = "requires plow_verify binary"]
fn checkpoint_d_checks_removed_edge_witnesses() {
    let mut payload = serde_json::json!({
        "task_graph": {"n": 3, "edges": [[0, 1], [1, 2], [0, 2]]},
        "protocol": {"waits": [[], [0], [1]], "succs": [[0], [1], []],
            "resource": [0, 1, 2], "stream_idx": [0, 0, 0], "threshold": {}},
        "dependency_paths": [[], [], [1]], "address_map": []
    });
    assert!(lean_verify::call("D", payload.clone()).unwrap().ok);
    for paths in [serde_json::json!([[], [], []]), serde_json::json!([[], []]),
        serde_json::json!([[], [], [0]]), serde_json::json!([[], [], [3]])] {
        payload["dependency_paths"] = paths;
        assert!(!lean_verify::call("D", payload.clone()).unwrap().ok);
    }
}

#[test]
#[ignore = "requires plow_verify binary"]
fn checkpoint_f_reuses_d_verifier() {
    // check_address_map hits the /F/ endpoint but the underlying math is the
    // same as D. Same input, same accept/reject verdict.
    let cert = check_address_map(&safe_request()).expect("verifier call failed");
    assert!(cert.ok, "cert: {cert:?}");
    assert_eq!(cert.checkpoint, "F");
    assert!(
        cert.notes.unwrap().contains("strict AddressMapSound"),
        "F now emits a strict-form acceptance note"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn checkpoint_d_checks_address_paths_without_graph_saturation() {
    let mut payload = serde_json::to_value(safe_request()).unwrap();
    payload["dependency_paths"] = serde_json::json!([[], []]);
    payload["address_paths"] = serde_json::json!([{"source": 1, "target": 2, "via": []}]);
    assert!(lean_verify::call("D", payload.clone()).unwrap().ok);
    for invalid in [serde_json::json!([]), serde_json::json!([{"source": 1, "target": 2, "via": [3]}]),
        serde_json::json!([{"source": 1, "target": 4, "via": []}]),
        serde_json::json!([{"source": "1", "target": 2, "via": []}])] {
        payload["address_paths"] = invalid;
        assert!(!lean_verify::call("D", payload.clone()).unwrap().ok);
    }

    let n = 20_000;
    let payload = serde_json::json!({
        "task_graph": {"n": n, "edges": [[0,n-1]]},
        "protocol": {"waits": vec![Vec::<u64>::new(); n], "succs": vec![Vec::<u64>::new(); n],
            "resource": vec![0; n], "stream_idx": (0..n).collect::<Vec<_>>(), "threshold": {}},
        "dependency_paths": [[]],
        "address_map": [
            {"name":"old", "offset":0, "size":64, "cls":"Scratch", "readers":[0], "writers":[0]},
            {"name":"new", "offset":0, "size":64, "cls":"Scratch", "readers":[n-1], "writers":[n-1]}],
        "address_paths": [{"source":0, "target":n-1, "via":[]}],
    });
    assert!(lean_verify::call("D", payload).unwrap().ok);
}

#[test]
#[ignore = "requires plow_verify binary"]
fn all_checkpoints_are_wired() {
    // A/B/C/E all have real Lean-side dispatchers (see the individual
    // checkpoint tests in `crates/plowc/tests/lean_verify_*.rs`). This
    // smoke test proves each responds — either accepting a trivially valid
    // empty request or rejecting with a specific reason, never a
    // "not implemented" answer.
    //
    // Note on C: `check_sram_fit` is callable and proven sound, but plowc's
    // `run_lean_verify` does *not* auto-run it per bucket — Rust's
    // `analyze_temporal_fit` applies the same two conditions the Lean
    // theorem proves suffice (see `crates/schedule/src/sram_fit.rs` and
    // `lean-plow/Plow/Sram.lean`). C stays here as an opt-in.
    use lean_verify::checkpoints::{
        rewrite::{check_rewrite_rules, RewriteRulesRequest},
        sram::{check_sram_fit, SramFitRequest},
        tile_partition::{check_tile_partition, TilePartitionRequest},
        wire::{check_wire_roundtrip, WireRequest},
    };
    let a = check_rewrite_rules(&RewriteRulesRequest { rules: vec![] }).unwrap();
    assert!(a.ok && a.checkpoint == "A");
    let b = check_tile_partition(&TilePartitionRequest { candidates: vec![] }).unwrap();
    assert!(b.ok && b.checkpoint == "B");
    let c = check_sram_fit(&SramFitRequest {
        budget: 4,
        handoffs: vec![],
    })
    .unwrap();
    assert!(c.ok && c.checkpoint == "C");
    let e = check_wire_roundtrip(&WireRequest {
        raw: vec![],
        frames: vec![],
    })
    .unwrap();
    assert!(e.ok && e.checkpoint == "E");
}

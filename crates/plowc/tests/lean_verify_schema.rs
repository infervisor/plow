//! Schema-hardening tests: send malformed payloads directly to `plow_verify`
//! and check the rejection reasons name the offending field. Complements
//! `lean_verify_negative.rs` (which corrupts a semantically-well-typed request);
//! this file exercises the parser's strict typing.

#![cfg(feature = "lean-verify")]

use lean_verify::call;
use serde_json::json;

/// Any rejection reason must (a) come from the parser, and (b) mention the
/// offending field name.
fn assert_parse_reject(payload: serde_json::Value, expect_substr: &str) {
    let cert = call("D", payload).expect("verifier call");
    assert!(!cert.ok, "expected rejection but got ok cert: {cert:?}");
    let reason = cert.reason.expect("rejection carries a reason");
    assert!(
        reason.contains("parse error") && reason.contains(expect_substr),
        "reason `{reason}` missing substring `{expect_substr}`"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_edge_out_of_range() {
    let payload = json!({
        "task_graph": { "n": 2, "edges": [[0, 1], [0, 99]] },
        "protocol": {},
        "address_map": [],
        "schedule_order": [0, 1]
    });
    assert_parse_reject(payload, "task_graph.edges[1].1");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_wrong_length_resource_array() {
    let payload = json!({
        "task_graph": { "n": 3, "edges": [] },
        "protocol": {
            "resource": [0, 0]
        },
        "address_map": [],
        "schedule_order": [0, 1, 2]
    });
    assert_parse_reject(payload, "protocol.resource");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_wrong_length_waits_array() {
    let payload = json!({
        "task_graph": { "n": 2, "edges": [] },
        "protocol": {
            "waits": [[], [], []]
        },
        "address_map": [],
        "schedule_order": [0, 1]
    });
    assert_parse_reject(payload, "protocol.waits");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_unknown_cls_value() {
    let payload = json!({
        "task_graph": { "n": 2, "edges": [] },
        "protocol": {},
        "address_map": [
            { "name": "buf0", "offset": 0, "size": 128,
              "cls": "Bogus", "writers": [0], "readers": [1] }
        ],
        "schedule_order": [0, 1]
    });
    assert_parse_reject(payload, "cls");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_out_of_range_reader_index() {
    let payload = json!({
        "task_graph": { "n": 2, "edges": [] },
        "protocol": {},
        "address_map": [
            { "name": "buf0", "offset": 0, "size": 128,
              "cls": "Scratch", "writers": [0], "readers": [99] }
        ],
        "schedule_order": [0, 1]
    });
    assert_parse_reject(payload, "readers");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_malformed_edge_pair() {
    let payload = json!({
        "task_graph": { "n": 2, "edges": [[0]] },
        "protocol": {},
        "address_map": [],
        "schedule_order": [0, 1]
    });
    assert_parse_reject(payload, "task_graph.edges[0]");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_a_minimal_well_formed_payload() {
    // Sanity: a well-typed empty request must still be accepted.
    let payload = json!({
        "task_graph": { "n": 1, "edges": [] },
        "protocol": {
            "waits":      [[]],
            "succs":      [[]],
            "resource":   [0],
            "stream_idx": [0],
            "threshold":  {}
        },
        "address_map": [],
        "schedule_order": [0]
    });
    let cert = call("D", payload).expect("verifier call");
    assert!(cert.ok, "well-formed minimal payload rejected: {cert:?}");
}

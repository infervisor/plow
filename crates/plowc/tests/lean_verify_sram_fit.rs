//! Integration test for checkpoint C — the Lean verifier is asked to re-check
//! that a set of hand-off promotion candidates satisfies the temporal-fit
//! predicate against a shared page budget. Backed by
//! `Plow.Sram.occupancy_le_of_temporal_fit`.

#![cfg(feature = "lean-verify")]

use lean_verify::checkpoints::sram::{check_sram_fit, Handoff, SramFitRequest};

fn safe_handoff() -> Handoff {
    Handoff {
        producer_pages: 4,
        consumer_pages: 4,
        producer_release: 100,
        consumer_acquire: 100,
        consumer_release: 200,
    }
}

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_temporally_disjoint_budget_ok() {
    let req = SramFitRequest {
        budget: 8,
        handoffs: vec![safe_handoff(), safe_handoff()],
    };
    let cert = check_sram_fit(&req).expect("verifier call");
    assert!(cert.ok, "safe handoffs rejected: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_overlapping_windows() {
    let mut req = SramFitRequest {
        budget: 8,
        handoffs: vec![safe_handoff()],
    };
    // Producer's window ends AFTER consumer's window starts.
    req.handoffs[0].producer_release = 150;
    req.handoffs[0].consumer_acquire = 100;
    let cert = check_sram_fit(&req).expect("verifier call");
    assert!(!cert.ok, "overlapping handoff accepted: {cert:?}");
    let reason = cert.reason.expect("rejection reason");
    assert!(
        reason.contains("temporally overlapping"),
        "unexpected reason: {reason}"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_producer_over_budget() {
    let mut req = SramFitRequest {
        budget: 3,
        handoffs: vec![safe_handoff()],
    };
    req.handoffs[0].producer_pages = 5;
    let cert = check_sram_fit(&req).expect("verifier call");
    assert!(!cert.ok, "over-budget producer accepted: {cert:?}");
    let reason = cert.reason.expect("rejection reason");
    assert!(reason.contains("producer_pages"), "unexpected reason: {reason}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_consumer_over_budget() {
    let mut req = SramFitRequest {
        budget: 4,
        handoffs: vec![safe_handoff()],
    };
    // Producer stays within budget (4), only consumer over.
    req.handoffs[0].consumer_pages = 5;
    let cert = check_sram_fit(&req).expect("verifier call");
    assert!(!cert.ok, "over-budget consumer accepted: {cert:?}");
    let reason = cert.reason.expect("rejection reason");
    assert!(reason.contains("consumer_pages"), "unexpected reason: {reason}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_empty_handoff_list() {
    let req = SramFitRequest { budget: 8, handoffs: vec![] };
    let cert = check_sram_fit(&req).expect("verifier call");
    assert!(cert.ok, "empty list rejected: {cert:?}");
}

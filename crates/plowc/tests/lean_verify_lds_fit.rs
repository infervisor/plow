//! Integration test for checkpoint G (LdsFitSound) — the Lean verifier re-checks
//! that every always-staged GEMV instance fits the decode-object LDS arena.
//! Backed by `Plow.LdsFit.fits_of_check_ok`; the rejection case is the exact
//! task-9 shape (rows=8, K=6144 against the gfx942 OCC4 arena).

#![cfg(feature = "lean-verify")]

use serde_json::json;

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_fitting_staged_ops() {
    let cert = lean_verify::call(
        "G",
        json!({"arena": 15360, "ops": [
            {"op": "GemvQkv", "idx": 0, "rows": 1, "k": 6144, "scratch": 16},
            {"op": "GemvGlu", "idx": 3, "rows": 2, "k": 6144, "scratch": 0},
        ]}),
    )
    .expect("verifier call");
    assert!(cert.ok, "fitting ops rejected: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_the_task9_shape() {
    let cert = lean_verify::call(
        "G",
        json!({"arena": 15360, "ops": [
            {"op": "GemvQkv", "idx": 2, "rows": 8, "k": 6144, "scratch": 0},
        ]}),
    )
    .expect("verifier call");
    assert!(!cert.ok, "the task-9 overflow shape was accepted");
    let why = cert.reason.unwrap_or_default();
    assert!(
        why.contains("inst 2") && why.contains("49152"),
        "rejection names the instance: {why}"
    );
}

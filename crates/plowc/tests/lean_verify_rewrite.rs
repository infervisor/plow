//! Integration test for checkpoint A's legacy name-catalog mode.
//!
//! Backed by `Plow.Rewrite.rule_*` (definitional-equality proofs for the
//! abstract rewrite syntax). Actual body binding is tested in rewrite_body.rs;
//! neither mode establishes floating-point kernel implementation correctness.

#![cfg(feature = "lean-verify")]

use lean_verify::checkpoints::rewrite::{check_rewrite_rules, RewriteRulesRequest};

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_known_sound_rules() {
    let req = RewriteRulesRequest {
        rules: vec![
            "rmsnorm-linear-fuse".into(),
            "linear-act-fuse".into(),
            "gated-mlp-fuse".into(),
            "residual-rmsnorm-fuse".into(),
            "materialized-residual-block-fuse".into(),
            "materialized-residual3-block-fuse".into(),
        ],
    };
    let cert = check_rewrite_rules(&req).expect("verifier call");
    assert!(cert.ok, "sound rules rejected: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_empty_rules_list() {
    let req = RewriteRulesRequest { rules: vec![] };
    let cert = check_rewrite_rules(&req).expect("verifier call");
    assert!(cert.ok, "empty rules rejected: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_unknown_rule() {
    let req = RewriteRulesRequest {
        rules: vec!["rmsnorm-linear-fuse".into(), "my-fancy-new-rule".into()],
    };
    let cert = check_rewrite_rules(&req).expect("verifier call");
    assert!(!cert.ok, "unknown rule accepted: {cert:?}");
    let reason = cert.reason.expect("rejection reason");
    assert!(
        reason.contains("my-fancy-new-rule"),
        "unexpected reason: {reason}"
    );
}

//! Checkpoint P against `plow_verify`: the §5.4 history replay and small synthetic flips. Needs
//! the binary, like `end_to_end.rs`.

use lean_verify::checkpoints::perf::check_perf;
use serde_json::{json, Value};

const RUNG: &str = "glm53-tp8/prog3/T8192/rows8192/prior65536";

fn arm(id: &str, samples: &[f64], controls: Option<(&str, &str)>) -> Value {
    let mut e = json!({
        "id": id, "job": "j", "harness": "proto", "metric": "chunk_ms", "better": "lower",
        "hardware": {"box": "8xMI300X", "rocm": "7.14", "driver": null, "firmware": null},
        "rung": {"digest": RUNG, "role": "prefill", "rows": 8192, "prior": 65536, "topology": "ordinary"},
        "samples": samples,
    });
    if let Some((c, c2)) = controls {
        e["control_of"] = json!(c);
        e["repeat_control_of"] = json!(c2);
    }
    e
}

fn flip(treat: &[f64], neutral: bool, numeric: bool, facts: Value) -> Value {
    let mut touched = json!({"rung": RUNG, "treat": "t"});
    if neutral {
        touched["neutral_evidence"] = json!(["unchanged by design"]);
    }
    json!({
        "ledger": [
            arm("c", &[660.0, 662.0, 664.0], None),
            arm("c2", &[661.0, 663.0, 665.0], None),
            arm("t", treat, Some(("c", "c2"))),
        ],
        "touched": [touched],
        "untouched": [{"rung": "packet", "base": "8b15f4a2", "variant": "8b15f4a2"}],
        "tier4": false,
        "numeric": numeric,
        "facts": facts,
    })
}

/// `accept`, `reject` or `insufficient_evidence`, from the certificate.
fn verdict(p: &Value) -> String {
    let c = check_perf(p).expect("plow_verify answered");
    if c.ok {
        return "accept".into();
    }
    let reason = c.reason.unwrap_or_default();
    if reason.starts_with("flip rejected") {
        "reject".into()
    } else {
        assert!(reason.starts_with("insufficient_evidence"), "{reason}");
        "insufficient_evidence".into()
    }
}

#[test]
#[ignore = "requires plow_verify binary"]
fn improvement_beyond_the_floor_is_accepted() {
    // floor = |662 - 663| + 2 * max(2, 2, 2) = 5; 662 - 640 > 5.
    assert_eq!(
        verdict(&flip(&[638.0, 640.0, 642.0], false, false, json!([]))),
        "accept"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn change_inside_the_floor_is_rejected() {
    assert_eq!(
        verdict(&flip(&[657.0, 659.0, 661.0], false, false, json!([]))),
        "reject"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn neutral_rung_needs_evidence_and_no_regression() {
    let same = [660.0, 662.5, 665.0];
    assert_eq!(verdict(&flip(&same, false, false, json!([]))), "reject");
    assert_eq!(verdict(&flip(&same, true, false, json!([]))), "accept");
    assert_eq!(
        verdict(&flip(&[690.0, 692.0, 694.0], true, false, json!([]))),
        "reject"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn floor_that_cannot_be_computed_is_insufficient() {
    let mut p = flip(&[600.0, 601.0], false, false, json!([]));
    assert_eq!(verdict(&p), "insufficient_evidence");
    // Stats-only records without a MAD.
    for e in p["ledger"].as_array_mut().unwrap() {
        e.as_object_mut().unwrap().remove("samples");
        e["stats"] = json!({"n": 8, "median": 600.0});
    }
    p["ledger"][0]["stats"]["median"] = json!(662.0);
    assert_eq!(verdict(&p), "insufficient_evidence");
    p["ledger"][2]
        .as_object_mut()
        .unwrap()
        .remove("repeat_control_of");
    assert_eq!(verdict(&p), "insufficient_evidence");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn numeric_scope_needs_passing_facts() {
    let fast = [638.0, 640.0, 642.0];
    assert_eq!(
        verdict(&flip(&fast, false, true, json!([]))),
        "insufficient_evidence"
    );
    let fail = json!([{"kind": "retrieval", "pass": false, "evidence": "17/18"}]);
    assert_eq!(verdict(&flip(&fast, false, true, fail)), "reject");
    let pass = json!([{"kind": "retrieval", "pass": true, "evidence": "18/18"}]);
    assert_eq!(verdict(&flip(&fast, false, true, pass)), "accept");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn tier4_and_untouched_digests() {
    let fast = [638.0, 640.0, 642.0];
    let mut p = flip(&fast, false, false, json!([]));
    p["tier4"] = json!(true);
    assert_eq!(verdict(&p), "insufficient_evidence");
    p["serving"] = json!(["t"]);
    assert_eq!(verdict(&p), "accept");
    p["untouched"][0]["variant"] = json!("0badc0de");
    assert_eq!(verdict(&p), "reject");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn history_replay() {
    let fx: Value =
        serde_json::from_str(include_str!("fixtures/perf-history.json")).expect("fixture parses");
    for case in fx["cases"].as_array().unwrap() {
        assert_eq!(
            verdict(&case["request"]),
            case["expect"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

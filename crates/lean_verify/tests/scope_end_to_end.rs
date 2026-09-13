//! Checkpoint S against `plow_verify` on small hand-built packets. Needs the binary, like
//! `end_to_end.rs`.

use lean_verify::checkpoints::scope::check_scope;
use serde_json::{json, Value};

fn key(kind: &str, rows: u32, sparse: bool) -> Value {
    json!({"kind": kind, "rows": rows, "topology": "ordinary", "sparse": sparse})
}

fn inst(op: u32, blocks: u32) -> Value {
    json!([
        op,
        blocks,
        [0, 0, 0],
        [1, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        0,
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
}

/// A pair whose instructions are already aligned: `[base, variant]` per slot, `null` for none.
fn pair(a: Value, b: Value, insts: Value, facts: (Value, Value)) -> Value {
    json!({"a": a, "b": b, "body": {"insts": insts, "tensors": [[], []], "facts": [facts.0, facts.1]}})
}

fn request(delta: &[&str], scope: Value, pairs: Value, routes: Value) -> Value {
    json!({
        "model": "glm_moe_dsa",
        "classes": [[8, ["gemm"]], [29, ["collective"]], [87, ["moe"]]],
        "delta": delta,
        "scope": scope,
        "pairs": pairs,
        "routes": routes,
    })
}

/// Small prefill buckets, not their collectives: the scope `PLOW_GLM_PF_SMALL_CUS` declares.
fn small_cus_scope() -> Value {
    json!([{"kinds": ["prefill"], "rows_min": 0, "rows_max": 2047, "ops": {"not_in": ["collective"]},
            "fields": ["cus", "segments"]}])
}

fn check(p: &Value) -> lean_verify::Certificate {
    check_scope(p).expect("plow_verify answered")
}

fn small(insts: Value) -> Value {
    pair(
        key("prefill", 128, false),
        key("prefill", 128, false),
        insts,
        (json!([]), json!([])),
    )
}

#[test]
#[ignore = "requires plow_verify binary"]
fn identical_packets_pass_without_a_delta() {
    let same = json!({"a": key("decode", 8, false), "b": key("decode", 8, false), "body": null});
    let cert = check(&request(&[], json!([]), json!([same]), json!([])));
    assert!(cert.ok, "{cert:?}");
    assert!(cert.notes.unwrap().starts_with("0 differences"));
}

#[test]
#[ignore = "requires plow_verify binary"]
fn any_difference_without_a_delta_is_rejected() {
    let p = small(json!([[inst(8, 64), inst(8, 16)]]));
    let cert = check(&request(&[], small_cus_scope(), json!([p]), json!([])));
    assert!(!cert.ok && cert.reason.unwrap().contains("no knob delta"));
}

#[test]
#[ignore = "requires plow_verify binary"]
fn a_narrowed_collective_is_outside_the_small_cus_scope() {
    let knob = ["emit.glm_pf_small_cus"];
    let compute_only = small(json!([
        [inst(8, 64), inst(8, 16)],
        [inst(29, 64), inst(29, 64)]
    ]));
    let cert = check(&request(
        &knob,
        small_cus_scope(),
        json!([compute_only]),
        json!([]),
    ));
    assert!(cert.ok, "{cert:?}");
    let collective = small(json!([
        [inst(8, 64), inst(8, 16)],
        [inst(29, 64), inst(29, 16)]
    ]));
    let cert = check(&request(
        &knob,
        small_cus_scope(),
        json!([collective]),
        json!([]),
    ));
    assert!(
        !cert.ok && cert.reason.unwrap().contains("ops [29]"),
        "6deb4025 class"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn a_removed_instruction_is_one_difference() {
    let knob = ["emit.glm_moe_shared_seed"];
    let removed = small(json!([
        [inst(8, 64), inst(8, 64)],
        [inst(87, 64), null],
        [inst(29, 64), inst(29, 64)]
    ]));
    let moe = json!([{"kinds": ["prefill"], "ops": {"in": ["moe"]}, "fields": ["op"]}]);
    let cert = check(&request(&knob, moe, json!([removed]), json!([])));
    assert!(cert.ok, "{cert:?}");
    assert!(cert.notes.unwrap().starts_with("1 differences"));
}

#[test]
#[ignore = "requires plow_verify binary"]
fn a_program_no_allowance_selects_must_be_unchanged() {
    let knob = ["emit.glm_pf_small_cus"];
    let wide = pair(
        key("prefill", 8192, true),
        key("prefill", 8192, true),
        json!([[inst(8, 64), inst(8, 16)]]),
        (json!([]), json!([])),
    );
    let cert = check(&request(&knob, small_cus_scope(), json!([wide]), json!([])));
    assert!(!cert.ok && cert.reason.unwrap().contains("prefill/8192"));
}

#[test]
#[ignore = "requires plow_verify binary"]
fn a_removed_program_is_a_program_set_difference() {
    let knob = ["emit.glm_pf_small_cus"];
    let gone = json!({"a": key("prefill", 128, false), "b": null, "body": null});
    let cert = check(&request(
        &knob,
        small_cus_scope(),
        json!([gone.clone()]),
        json!([]),
    ));
    assert!(!cert.ok && cert.reason.unwrap().contains("program_set"));
    let scope = json!([{"kinds": ["prefill"], "fields": ["program_set"]}]);
    assert!(check(&request(&knob, scope, json!([gone]), json!([]))).ok);
}

#[test]
#[ignore = "requires plow_verify binary"]
fn object_facts_are_filtered_by_name() {
    let knob = ["emit.glm_moe_shared_seed"];
    let g = key("global", 0, false);
    let facts = |extra: &str| {
        pair(
            g.clone(),
            g.clone(),
            json!([]),
            (json!(["union:Gemm"]), json!(["union:Gemm", extra])),
        )
    };
    let scope = json!([{"kinds": ["global"], "fields": ["object_facts"], "facts": ["moe_aiter"]}]);
    let adapter = facts("#define plow_moe_aiter_keep_out_abi_1");
    assert!(check(&request(&knob, scope.clone(), json!([adapter]), json!([]))).ok);
    let other = facts("#define PLOW_GLM_FOLD 1");
    let cert = check(&request(&knob, scope, json!([other]), json!([])));
    assert!(!cert.ok && cert.reason.unwrap().contains("PLOW_GLM_FOLD"));
}

#[test]
#[ignore = "requires plow_verify binary"]
fn a_route_move_needs_a_route_allowance_over_both_programs() {
    let knob = ["rt.tail_sparse_ctx"];
    let step = json!([{"label": "r:0", "off": key("prefill", 2048, false),
                       "on": key("prefill", 8192, true), "off_skip": [], "on_skip": []}]);
    let decode_only = json!([{"kinds": ["decode"], "fields": ["route"]}]);
    assert!(!check(&request(&knob, decode_only, json!([]), step.clone())).ok);
    let prefill = json!([{"kinds": ["prefill"], "fields": ["route"]}]);
    assert!(check(&request(&knob, prefill, json!([]), step)).ok);
}

#[test]
#[ignore = "requires plow_verify binary"]
fn skipped_segments_are_a_route_difference() {
    let knob = ["rt.union_skip"];
    let sparse = key("prefill", 8192, true);
    let step =
        json!([{"label": "r:1", "off": sparse, "on": sparse, "off_skip": [], "on_skip": [3, 7]}]);
    let dense_only = json!([{"kinds": ["prefill"], "sparse": false, "fields": ["route"]}]);
    assert!(!check(&request(&knob, dense_only, json!([]), step.clone())).ok);
    let sparse_only = json!([{"kinds": ["prefill"], "sparse": true, "fields": ["route"]}]);
    assert!(check(&request(&knob, sparse_only, json!([]), step)).ok);
}

#[test]
#[ignore = "requires plow_verify binary"]
fn empty_effect_and_scope_slack_are_warned() {
    let same =
        json!({"a": key("prefill", 128, false), "b": key("prefill", 128, false), "body": null});
    let unused = json!([{"kinds": ["decode"], "fields": ["cus"]}]);
    let cert = check(&request(
        &["emit.glm_ordinary_band"],
        unused,
        json!([same]),
        json!([]),
    ));
    assert!(cert.ok, "{cert:?}");
    let notes = cert.notes.unwrap();
    assert!(
        notes.contains("empty_effect") && notes.contains("scope_slack"),
        "{notes}"
    );
}

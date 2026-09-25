#![cfg(feature = "lean-verify")]

#[test]
#[ignore = "requires built plow_verify; CPU-only"]
fn actual_engine_rules_bind_bodies_and_retain_precision_relevant_attributes() {
    let source = rewrite::rules_source();
    let request = plowc::rewrite_body_request(source).unwrap();
    let cert = lean_verify::call("A", request.clone()).unwrap();
    assert!(cert.ok, "{cert:?}");
    assert!(cert.notes.unwrap().contains("actual rewrite bodies"));
    let mutations = [
        (
            "(FusedNormLinear ?x ?w ?wl ?eps ?out)",
            "(FusedNormLinear ?x ?w ?wl ?eps ?wrong_out)",
        ),
        (
            "(FusedNormLinear ?x ?w ?wl ?eps ?out)",
            "(FusedNormLinear ?w ?x ?wl ?eps ?out)",
        ),
        ("(SwiGLU ?k ?g ?u)", "(SwiGLU \"silu\" ?g ?u)"),
        (
            "(FusedNormRopeScale ?x ?w ?eps ?dim ?theta ?factor)",
            "(FusedNormRopeScale ?x ?w ?eps ?dim ?theta 1.0)",
        ),
        (
            "(FusedResidual3Norm ?x ?a ?b ?w ?eps)",
            "(FusedResidual3Norm ?a ?x ?b ?w ?eps)",
        ),
        (
            "(RmsNorm (Ew \"add\" ?x (Ew \"add\" ?a ?b)) ?w ?eps)",
            "(RmsNorm (Ew \"add\" (Ew \"add\" ?x ?a) ?b) ?w ?eps)",
        ),
        (
            "(FusedMaterializedResidual3Block ?pre ?a ?b ?snaps ?nw ?pw ?max)",
            "(FusedMaterializedResidual3Block ?a ?pre ?b ?snaps ?nw ?pw ?max)",
        ),
    ];
    for (from, to) in mutations {
        assert!(source.contains(from));
        let changed = source.replace(from, to);
        let bad = plowc::rewrite_body_request(&changed).unwrap();
        assert_eq!(bad["rules"], request["rules"]);
        assert_ne!(bad["source_sha256"], request["source_sha256"]);
        assert!(!lean_verify::call("A", bad).unwrap().ok, "mutation {to}");
    }
    for mutation in 0..4 {
        let mut bad = request.clone();
        match mutation {
            0 => {
                bad["bodies"].as_array_mut().unwrap().pop();
            }
            1 => bad["bodies"][0]["name"] = serde_json::json!("other"),
            2 => bad["bodies"][0]["rhs"] = serde_json::json!(["string", "?x", [1]]),
            _ => bad["bodies"] = serde_json::Value::Null,
        }
        assert!(
            !lean_verify::call("A", bad).unwrap().ok,
            "envelope mutation {mutation}"
        );
    }
}

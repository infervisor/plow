use serde_json::json;

#[test]
#[ignore = "requires plow_verify binary"]
fn padded_wv_address_domain_preserves_bf16_boundary_without_expanding_coverage() {
    let valid = json!({"rows":16,"heads":8,"n":256,"k":512,"head_stride":1024,
        "capacity_elements":16*8192,"boundary":"bf16_rne","weight_scale":"scalar_after_group_sum",
        "activation_group":128,"inactive_rows":false});
    let mut requests = vec![("L", valid.clone())];
    for (key, value) in [
        ("rows", json!(32)),
        ("rows", json!(0)),
        ("heads", json!(16)),
        ("n", json!(512)),
        ("k", json!(576)),
        ("head_stride", json!(512)),
        ("capacity_elements", json!(16 * 8192 - 1)),
        ("boundary", json!("fp32")),
        ("weight_scale", json!("per_block")),
        ("activation_group", json!(32)),
        ("inactive_rows", json!(true)),
    ] {
        let mut bad = valid.clone();
        bad[key] = value;
        requests.push(("L", bad));
    }
    let certs = lean_verify::call_batch(&requests).unwrap();
    assert!(certs[0].ok);
    assert!(certs[0]
        .notes
        .as_deref()
        .unwrap()
        .contains("not a floating-point kernel"));
    for (index, cert) in certs[1..].iter().enumerate() {
        assert!(!cert.ok, "mutation {index}: {cert:?}");
    }
}

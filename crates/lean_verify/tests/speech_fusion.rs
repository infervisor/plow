//! `speech_fusion.v1` conv row-scale sites: the Veena SNAC site (slab-backed tensors, 9f5babb0
//! audit bundle) is accepted; each single-field mutation that breaks the contract is rejected.

use serde_json::{json, Value};

fn site() -> Value {
    json!({"kind": "conv_row_scale", "batch": 1, "in_rows": 256, "out_channels": 512, "kernel": 1,
        "stride": 1, "dilation": 1, "pad_before": 0, "pad_after": 0, "out_bytes": 536870912u64,
        "residual_bytes": 536870912u64, "row_scale_bytes": 536870912u64,
        "row_scale_is_out": false, "row_scale_is_residual": false})
}

#[test]
#[ignore = "requires built plow_verify; CPU-only"]
fn slab_conv_site_holds_and_each_mutation_rejects() {
    let exact = {
        let mut s = site();
        s["out_bytes"] = json!(524288);
        s["residual_bytes"] = json!(524288);
        s["row_scale_bytes"] = json!(1024);
        s
    };
    let mut cases = vec![("slab", site()), ("exact", exact.clone())];
    let mutations: Vec<(&str, Value)> = vec![
        ("out_bytes", json!(524284)),
        ("residual_bytes", json!(524284)),
        ("row_scale_bytes", json!(1020)),
        ("row_scale_is_out", json!(true)),
        ("row_scale_is_residual", json!(true)),
        ("kernel", json!(0)),
        ("in_rows", json!(257)),
    ];
    for (field, value) in &mutations {
        let mut s = exact.clone();
        s[*field] = value.clone();
        cases.push((field, s));
    }
    let requests: Vec<_> = cases
        .iter()
        .map(|(_, s)| ("speech_fusion.v1", json!({"schema": 1, "sites": [s]})))
        .collect();
    let certs = lean_verify::call_batch(&requests).unwrap();
    for ((what, _), cert) in cases.iter().zip(&certs).take(2) {
        assert!(cert.ok, "{what}: {:?}", cert.reason);
    }
    for ((what, _), cert) in cases.iter().zip(&certs).skip(2) {
        assert!(!cert.ok, "{what} mutation accepted");
    }
}

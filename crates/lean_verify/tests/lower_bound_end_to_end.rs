use lean_verify::queries::lower_bound::{query_lower_bound, LowerBoundRequest};
use lean_verify::query;
use serde_json::json;

fn payload() -> serde_json::Value {
    json!({"edges": [[2, 0], [0, 1]], "durations": [3, 5, 7],
        "total_hbm_bytes": 0, "peak_bw_bytes_per_cycle": 0,
        "total_flops": 0, "peak_flops_per_cycle": 0})
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_malformed_and_cyclic_graphs() {
    for edges in [
        json!([[0, 1, 2]]),
        json!([[0]]),
        json!([[0, 3]]),
        json!([[0, 0]]),
        json!([[0, 1], [1, 0]]),
    ] {
        let mut p = payload();
        p["edges"] = edges;
        assert!(query("lower_bound", p).is_err());
    }
    let mut zero_cycle = payload();
    zero_cycle["edges"] = json!([[0, 1], [1, 0]]);
    zero_cycle["durations"] = json!([0, 0, 0]);
    assert!(query("lower_bound", zero_cycle).is_err());
}

#[test]
#[ignore = "requires plow_verify binary"]
fn validates_rates_and_preserves_certificate_envelope() {
    let req = LowerBoundRequest {
        edges: vec![(2, 0), (0, 1)],
        durations: vec![3, 5, 7],
        total_hbm_bytes: 0,
        peak_bw_bytes_per_cycle: 0,
        total_flops: 0,
        peak_flops_per_cycle: 0,
    };
    let result = query_lower_bound(&req).unwrap();
    assert_eq!(result.critical_path, 15);
    assert!(result
        .certificate
        .unwrap()
        .contains("conditional on supplied"));
    for key in ["total_hbm_bytes", "total_flops"] {
        let mut p = payload();
        p[key] = json!(1);
        assert!(query("lower_bound", p).is_err());
    }
}

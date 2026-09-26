use serde_json::{json, Value};

fn request() -> Value {
    json!({
        "required": ["m16-n256-k512-h8-positive", "m32-n256-k512-h8-inactive"],
        "candidates": [
            {"domain":"m16-n256-k512-h8-positive","key":"baseline","cost":10.25,"qualified":true},
            {"domain":"m16-n256-k512-h8-positive","key":"candidate","cost":8.125,"qualified":true},
            {"domain":"m16-n256-k512-h8-positive","key":"unqualified-fast","cost":1,"qualified":false},
            {"domain":"m32-n256-k512-h8-inactive","key":"baseline","cost":15,"qualified":true}
        ],
        "choices": [
            {"domain":"m16-n256-k512-h8-positive","key":"candidate"},
            {"domain":"m32-n256-k512-h8-inactive","key":"baseline"}
        ]
    })
}

#[test]
#[ignore = "requires plow_verify binary"]
fn measured_policy_proves_only_supplied_domain_coverage_and_minimum() {
    let valid = request();
    let mut cases = vec![("R", valid.clone())];
    for mutation in 0..8 {
        let mut bad = valid.clone();
        match mutation {
            0 => bad["choices"][0]["key"] = json!("baseline"),
            1 => bad["choices"][0]["key"] = json!("unqualified-fast"),
            2 => {
                bad["choices"].as_array_mut().unwrap().pop();
            }
            3 => bad["required"] = json!([]),
            4 => bad["choices"][1] = bad["choices"][0].clone(),
            5 => bad["candidates"][0]["cost"] = json!(-1),
            6 => bad["candidates"][1]["domain"] = json!("different-tail-domain"),
            _ => bad["candidates"][1]["key"] = json!("baseline"),
        }
        cases.push(("R", bad));
    }
    let certs = lean_verify::call_batch(&cases).unwrap();
    assert!(certs[0].ok, "{:?}", certs[0]);
    assert!(certs[0]
        .notes
        .as_deref()
        .unwrap()
        .contains("no hardware optimality"));
    for (index, cert) in certs[1..].iter().enumerate() {
        assert!(!cert.ok, "mutation {index}: {cert:?}");
    }
}

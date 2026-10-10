//! JSON-schema validation of tool-call arguments (`strict: true` tools, and forced calls).
//!
//! The subset OpenAI accepts for strict function schemas: `type` (one or a list), `enum`,
//! `const`, `properties`, `required`, `additionalProperties` (bool or schema), `items`,
//! `prefixItems`, `minItems`/`maxItems`, `minLength`/`maxLength`, `minimum`/`maximum` and their
//! exclusive forms, `multipleOf`, `anyOf`/`oneOf`/`allOf`, and local `$ref`s
//! (`#/$defs/..`, `#/definitions/..`). `pattern` and `format` are not checked.

use serde_json::Value;

/// Validate `v` against `schema`; the error names the JSON path of the first violation.
pub fn validate(schema: &Value, v: &Value) -> Result<(), String> {
    // Duplicated `$ref` / `anyOf` alternatives make the walk exponential in schema depth, and this
    // runs on the response path: bound the total work by the size of the value being checked.
    let mut budget = 4096 + 32 * nodes(v);
    check(schema, schema, v, &mut String::from("$"), 0, &mut budget)
}

fn nodes(v: &Value) -> u64 {
    1 + match v {
        Value::Array(a) => a.iter().map(nodes).sum(),
        Value::Object(o) => o.values().map(nodes).sum(),
        _ => 0,
    }
}

fn type_ok(t: &str, v: &Value) -> bool {
    match t {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "number" => v.is_number(),
        "integer" => v.as_i64().is_some() || v.as_u64().is_some() || v.as_f64().is_some_and(|f| f.fract() == 0.0),
        _ => true,
    }
}

fn resolve<'a>(root: &'a Value, r: &str) -> Option<&'a Value> {
    let path = r.strip_prefix("#")?;
    if path.is_empty() {
        return Some(root);
    }
    root.pointer(path)
}

fn check(root: &Value, s: &Value, v: &Value, path: &mut String, depth: u32, budget: &mut u64) -> Result<(), String> {
    if depth > 64 {
        return Err(format!("{path}: schema nests too deeply"));
    }
    if *budget == 0 {
        return Err(format!("{path}: schema is too expensive to validate"));
    }
    *budget -= 1;
    let Some(s) = s.as_object() else {
        // `true` / `{}` accept anything; `false` nothing.
        return match s {
            Value::Bool(false) => Err(format!("{path}: no value is allowed here")),
            _ => Ok(()),
        };
    };
    if let Some(r) = s.get("$ref").and_then(Value::as_str) {
        let target = resolve(root, r).ok_or_else(|| format!("{path}: unresolvable $ref {r}"))?;
        check(root, target, v, path, depth + 1, budget)?;
    }
    match s.get("type") {
        Some(Value::String(t)) if !type_ok(t, v) => return Err(format!("{path}: expected {t}, got {}", kind(v))),
        Some(Value::Array(ts)) if !ts.iter().filter_map(Value::as_str).any(|t| type_ok(t, v)) => {
            return Err(format!("{path}: expected one of {}, got {}", Value::Array(ts.clone()), kind(v)))
        }
        _ => {}
    }
    if let Some(e) = s.get("enum").and_then(Value::as_array) {
        if !e.contains(v) {
            return Err(format!("{path}: {v} is not one of {}", Value::Array(e.clone())));
        }
    }
    if let Some(c) = s.get("const") {
        if c != v {
            return Err(format!("{path}: must be {c}"));
        }
    }
    for (k, all) in [("allOf", true), ("anyOf", false), ("oneOf", false)] {
        let Some(subs) = s.get(k).and_then(Value::as_array) else { continue };
        let mut sub_ok = |sub: &Value| -> Result<bool, String> {
            match check(root, sub, v, &mut path.clone(), depth + 1, budget) {
                Ok(()) => Ok(true),
                Err(e) if *budget == 0 => Err(e),
                Err(_) => Ok(false),
            }
        };
        let mut matched = 0;
        for sub in subs {
            let ok = sub_ok(sub)?;
            matched += usize::from(ok);
            if (k == "allOf" && !ok) || (k == "anyOf" && ok) || (k == "oneOf" && matched > 1) {
                break;
            }
        }
        let pass = match k {
            "allOf" => matched == subs.len(),
            "oneOf" => matched == 1,
            _ => matched > 0,
        };
        if !pass {
            return Err(format!("{path}: does not match {k}{}", if all { "" } else { " of its alternatives" }));
        }
    }
    match v {
        Value::Object(o) => {
            let props = s.get("properties").and_then(Value::as_object);
            if let Some(req) = s.get("required").and_then(Value::as_array) {
                for r in req.iter().filter_map(Value::as_str) {
                    if !o.contains_key(r) {
                        return Err(format!("{path}: missing required property `{r}`"));
                    }
                }
            }
            for (k, val) in o {
                let n = path.len();
                path.push('.');
                path.push_str(k);
                let r = match props.and_then(|p| p.get(k)) {
                    Some(ps) => check(root, ps, val, path, depth + 1, budget),
                    None => match s.get("additionalProperties") {
                        Some(Value::Bool(false)) => Err(format!("{path}: property is not allowed")),
                        Some(ap @ Value::Object(_)) => check(root, ap, val, path, depth + 1, budget),
                        _ => Ok(()),
                    },
                };
                r?;
                path.truncate(n);
            }
        }
        Value::Array(a) => {
            let n = a.len() as u64;
            if s.get("minItems").and_then(Value::as_u64).is_some_and(|m| n < m) {
                return Err(format!("{path}: fewer than {} items", s["minItems"]));
            }
            if s.get("maxItems").and_then(Value::as_u64).is_some_and(|m| n > m) {
                return Err(format!("{path}: more than {} items", s["maxItems"]));
            }
            let prefix = s.get("prefixItems").and_then(Value::as_array);
            for (i, item) in a.iter().enumerate() {
                let sub = prefix.and_then(|p| p.get(i)).or_else(|| s.get("items"));
                if let Some(sub) = sub {
                    let len = path.len();
                    path.push_str(&format!("[{i}]"));
                    check(root, sub, item, path, depth + 1, budget)?;
                    path.truncate(len);
                }
            }
        }
        Value::String(t) => {
            let n = t.chars().count() as u64;
            if s.get("minLength").and_then(Value::as_u64).is_some_and(|m| n < m) {
                return Err(format!("{path}: shorter than {} characters", s["minLength"]));
            }
            if s.get("maxLength").and_then(Value::as_u64).is_some_and(|m| n > m) {
                return Err(format!("{path}: longer than {} characters", s["maxLength"]));
            }
        }
        Value::Number(x) => {
            let x = x.as_f64().unwrap_or(f64::NAN);
            let f = |k: &str| s.get(k).and_then(Value::as_f64);
            let bad = f("minimum").is_some_and(|m| x < m)
                || f("maximum").is_some_and(|m| x > m)
                || f("exclusiveMinimum").is_some_and(|m| x <= m)
                || f("exclusiveMaximum").is_some_and(|m| x >= m)
                || f("multipleOf").is_some_and(|m| m > 0.0 && ((x / m).round() * m - x).abs() > 1e-9 * x.abs().max(1.0));
            if bad {
                return Err(format!("{path}: {x} is out of the allowed range"));
            }
        }
        _ => {}
    }
    Ok(())
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strict_schemas_accept_and_reject() {
        let s = json!({"type": "object", "additionalProperties": false, "required": ["city", "unit"],
            "properties": {"city": {"type": "string", "minLength": 1},
                           "unit": {"type": ["string", "null"], "enum": ["c", "f", null]},
                           "days": {"type": "integer", "minimum": 1, "maximum": 7},
                           "tags": {"type": "array", "items": {"$ref": "#/$defs/tag"}, "maxItems": 2}},
            "$defs": {"tag": {"type": "string", "maxLength": 3}}});
        assert!(validate(&s, &json!({"city": "Paris", "unit": null})).is_ok());
        assert!(validate(&s, &json!({"city": "Paris", "unit": "c", "days": 3, "tags": ["a", "bb"]})).is_ok());
        let err = |v: Value| validate(&s, &v).unwrap_err();
        assert!(err(json!({"city": "Paris"})).contains("missing required property `unit`"));
        assert!(err(json!({"city": "Paris", "unit": "k"})).contains("$.unit"));
        assert!(err(json!({"city": "Paris", "unit": "c", "x": 1})).contains("$.x: property is not allowed"));
        assert!(err(json!({"city": "Paris", "unit": "c", "days": 9})).contains("$.days"));
        assert!(err(json!({"city": "Paris", "unit": "c", "days": 2.5})).contains("expected integer"));
        assert!(err(json!({"city": "Paris", "unit": "c", "tags": ["toolong"]})).contains("$.tags[0]"));
        assert!(err(json!({"city": 3, "unit": "c"})).contains("$.city: expected string"));
        assert!(err(json!([1])).contains("expected object"));
        let any = json!({"anyOf": [{"type": "string"}, {"type": "integer"}]});
        assert!(validate(&any, &json!(1)).is_ok() && validate(&any, &json!(true)).is_err());
        let one = json!({"oneOf": [{"type": "integer"}, {"type": "number"}, {"type": "string"}]});
        assert!(validate(&one, &json!(1)).is_err() && validate(&one, &json!(1.5)).is_ok());
    }

    #[test]
    fn duplicated_alternatives_are_bounded() {
        // d0 = {anyOf: [d1, d1]}, ..., d29 = {type: string}: 2^30 paths for a failing value.
        let mut defs = serde_json::Map::new();
        for i in 0..29 {
            let r = json!({"$ref": format!("#/$defs/d{}", i + 1)});
            defs.insert(format!("d{i}"), json!({"anyOf": [r.clone(), r]}));
        }
        defs.insert("d29".into(), json!({"type": "string"}));
        let s = json!({"$ref": "#/$defs/d0", "$defs": defs});
        let t = std::time::Instant::now();
        assert!(validate(&s, &json!("ok")).is_ok());
        assert!(validate(&s, &json!(1)).unwrap_err().contains("too expensive"));
        assert!(t.elapsed() < std::time::Duration::from_millis(500), "{:?}", t.elapsed());
    }
}

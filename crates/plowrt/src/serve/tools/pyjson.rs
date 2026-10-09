//! Python's `json.dumps`, byte for byte, for the `tojson` template filter.
//!
//! `transformers` binds `tojson` to `json.dumps(x, ensure_ascii=False, indent, separators,
//! sort_keys)`. Its default separators are `", "` and `": "` (`","` between items once `indent`
//! is set), and floats print as Python's `repr`. A tool schema rendered with `serde_json`'s compact
//! form is a different prompt from the one the model was trained on.

use serde_json::Value;

#[derive(Clone, Debug, Default)]
pub struct Opts {
    pub ensure_ascii: bool,
    pub indent: Option<usize>,
    /// `(item, key)`.
    pub separators: Option<(String, String)>,
    pub sort_keys: bool,
}

pub fn dumps(v: &Value, o: &Opts) -> String {
    let (item, key) = match &o.separators {
        Some((i, k)) => (i.as_str(), k.as_str()),
        None if o.indent.is_some() => (",", ": "),
        None => (", ", ": "),
    };
    let mut out = String::new();
    write(v, o, item, key, 0, &mut out);
    out
}

fn newline(o: &Opts, depth: usize, out: &mut String) {
    if let Some(n) = o.indent {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', n * depth));
    }
}

fn write(v: &Value, o: &Opts, item: &str, key: &str, depth: usize, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => out.push_str(&i.to_string()),
            (_, Some(u), _) => out.push_str(&u.to_string()),
            (_, _, Some(f)) => out.push_str(&float_repr(f)),
            _ => out.push_str(&n.to_string()),
        },
        Value::String(s) => string(s, o.ensure_ascii, out),
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                newline(o, depth + 1, out);
                write(x, o, item, key, depth + 1, out);
            }
            newline(o, depth, out);
            out.push(']');
        }
        Value::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut entries: Vec<(&String, &Value)> = m.iter().collect();
            if o.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (k, x)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                newline(o, depth + 1, out);
                string(k, o.ensure_ascii, out);
                out.push_str(key);
                write(x, o, item, key, depth + 1, out);
            }
            newline(o, depth, out);
            out.push('}');
        }
    }
}

fn string(s: &str, ensure_ascii: bool, out: &mut String) {
    use std::fmt::Write;
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (ensure_ascii && (c as u32) >= 0x7f) => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{u:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest round-trip digits, positional when the decimal exponent
/// is in `-4..16`, otherwise `d.ddde+XX`.
fn float_repr(f: f64) -> String {
    if !f.is_finite() {
        return if f.is_nan() { "NaN" } else if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    // `{:e}` is the shortest round-trip form: `d.ddde<exp>`.
    let sci = format!("{f:e}");
    let (mant, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let (neg, mant) = mant.strip_prefix('-').map_or((false, mant), |m| (true, m));
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    let decpt = exp + 1;
    if (-3..=16).contains(&decpt) {
        if decpt <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-decpt) as usize));
            out.push_str(&digits);
        } else if (decpt as usize) >= digits.len() {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', decpt as usize - digits.len()));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..decpt as usize]);
            out.push('.');
            out.push_str(&digits[decpt as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push_str(&format!("e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Expected strings are CPython 3 `json.dumps` output.
    #[test]
    fn matches_python_json_dumps() {
        let v = json!({"b": [1, 2.5, "x"], "a": {"k": null, "t": true}, "u": "Zürich \"q\"\n"});
        assert_eq!(
            dumps(&v, &Opts::default()),
            r#"{"b": [1, 2.5, "x"], "a": {"k": null, "t": true}, "u": "Zürich \"q\"\n"}"#
        );
        assert_eq!(
            dumps(&v, &Opts { ensure_ascii: true, sort_keys: true, ..Default::default() }),
            "{\"a\": {\"k\": null, \"t\": true}, \"b\": [1, 2.5, \"x\"], \"u\": \"Z\\u00fcrich \\\"q\\\"\\n\"}"
        );
        assert_eq!(
            dumps(&json!({"a": [1, {}], "b": []}), &Opts { indent: Some(2), ..Default::default() }),
            "{\n  \"a\": [\n    1,\n    {}\n  ],\n  \"b\": []\n}"
        );
        let compact = Opts { separators: Some((",".into(), ":".into())), ..Default::default() };
        assert_eq!(dumps(&json!({"a": [1, 2]}), &compact), r#"{"a":[1,2]}"#);
        assert_eq!(dumps(&json!("😀\u{1}"), &Opts { ensure_ascii: true, ..Default::default() }), "\"\\ud83d\\ude00\\u0001\"");
    }

    #[test]
    fn floats_print_as_python_repr() {
        for (f, want) in [
            (18.0, "18.0"),
            (450.5, "450.5"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (-2.25, "-2.25"),
            (123456789.123, "123456789.123"),
        ] {
            assert_eq!(float_repr(f), want, "{f}");
        }
    }
}

//! OpenAI `response_format`: `json_object` and `json_schema`, validated after generation.
//!
//! The generation starts inside the object ([`OPENER`] ends the prompt) and the finished text is
//! parsed, and for `json_schema` checked with [`crate::serve::tools::schema`]. Nothing constrains
//! the sampler: output that fails is an error, never a 200.

use serde_json::Value;

/// Written after the generation prompt; the reply's `content` starts with it.
pub const OPENER: &str = "{";

#[derive(Clone, Debug)]
pub enum JsonFormat {
    Object,
    Schema(Value),
}

/// `None` for an absent format or `{"type": "text"}`.
pub fn parse(format: Option<&Value>) -> Result<Option<JsonFormat>, String> {
    let Some(format) = format.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    match format.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_object") => Ok(Some(JsonFormat::Object)),
        Some("json_schema") => {
            let schema = format
                .get("json_schema")
                .and_then(|j| j.get("schema"))
                .filter(|s| s.is_object())
                .ok_or("`response_format.json_schema.schema` must be a JSON schema object")?;
            let object_root = match schema.get("type") {
                None => true,
                Some(Value::String(t)) => t == "object",
                Some(Value::Array(ts)) => ts.iter().any(|t| t == "object"),
                Some(_) => false,
            };
            if !object_root {
                return Err(
                    "`response_format.json_schema.schema` must have `type: \"object\"` at its root"
                        .into(),
                );
            }
            Ok(Some(JsonFormat::Schema(schema.clone())))
        }
        _ => Err(
            "`response_format.type` must be \"text\", \"json_object\" or \"json_schema\"".into(),
        ),
    }
}

impl JsonFormat {
    /// The reply's `content` from the text generated after [`OPENER`]: the first JSON value, with
    /// whatever the model wrote after it dropped. `truncated`: the generation was cut short, which
    /// the error for an incomplete value says.
    pub fn finish(&self, generated: &str, truncated: bool) -> Result<String, String> {
        let mut text = String::with_capacity(OPENER.len() + generated.len());
        text.push_str(OPENER);
        text.push_str(generated);
        let mut values = serde_json::Deserializer::from_str(&text).into_iter::<Value>();
        let value = match values.next() {
            Some(Ok(v)) => v,
            Some(Err(e)) if truncated && e.is_eof() => {
                return Err(
                    "the model's output was cut off before the JSON was complete; raise \
                     `max_tokens`"
                        .into(),
                )
            }
            Some(Err(e)) => return Err(format!("the model's output is not valid JSON: {e}")),
            None => return Err("the model's output is not valid JSON: it is empty".into()),
        };
        let end = values.byte_offset();
        if let JsonFormat::Schema(schema) = self {
            crate::serve::tools::schema::validate(schema, &value)
                .map_err(|e| format!("the model's output does not match the schema: {e}"))?;
        }
        text.truncate(end);
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn formats_parse_or_are_refused() {
        assert!(parse(None).unwrap().is_none());
        assert!(parse(Some(&json!({"type": "text"}))).unwrap().is_none());
        assert!(matches!(
            parse(Some(&json!({"type": "json_object"}))),
            Ok(Some(JsonFormat::Object))
        ));
        let schema =
            |s: Value| json!({"type": "json_schema", "json_schema": {"name": "x", "schema": s}});
        assert!(matches!(
            parse(Some(&schema(json!({"type": "object"})))),
            Ok(Some(JsonFormat::Schema(_)))
        ));
        assert!(parse(Some(&schema(json!({"$ref": "#/$defs/a"})))).is_ok());
        assert!(parse(Some(&schema(json!({"type": "array"})))).is_err());
        assert!(parse(Some(&json!({"type": "json_schema"}))).is_err());
        assert!(parse(Some(&json!({"type": "xml"}))).is_err());
    }

    #[test]
    fn finish_returns_the_first_value_or_an_error() {
        let object = JsonFormat::Object;
        assert_eq!(
            object.finish("\"a\": 1}\nanything else", false).unwrap(),
            "{\"a\": 1}"
        );
        assert!(object.finish("\"a\": }", false).is_err());
        assert!(object.finish("\"a\": 1", false).is_err());
        assert!(object
            .finish("\"a\": 1", true)
            .unwrap_err()
            .contains("max_tokens"));
        assert!(object.finish("\"a\": }", true).is_err());

        let schema = JsonFormat::Schema(json!({"type": "object", "required": ["intent"],
            "additionalProperties": false,
            "properties": {"intent": {"enum": ["book", "cancel"]}}}));
        assert_eq!(
            schema.finish("\"intent\": \"book\"}", false).unwrap(),
            "{\"intent\": \"book\"}"
        );
        let err = schema.finish("\"intent\": \"sing\"}", false).unwrap_err();
        assert!(err.contains("$.intent"), "{err}");
        assert!(schema.finish("\"intent\": \"book\"}", true).is_ok());
        assert!(schema.finish("\"other\": 1}", true).is_err());
    }
}

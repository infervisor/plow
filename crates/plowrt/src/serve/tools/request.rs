//! Request side: validate `tools` / `tool_choice` / tool-call history and map the conversation
//! into the template context the way `transformers` + vLLM do.
//!
//! Mapping rules (each one is what makes a template render as it does under vLLM):
//! * `tools` is passed to the template unchanged (OpenAI `{"type": "function", "function": ..}`).
//! * An assistant turn's `tool_calls[].function.arguments` JSON string becomes an object — the
//!   `transformers` convention, and what Gemma's template requires. A template that concatenates
//!   the arguments as a string (DeepSeek's) fails on an object and is rendered again with the
//!   original strings ([`template_messages`] with `args_as_objects = false`).
//! * A tool-call turn's `content: null` becomes `""`: gpt-oss's template fails on `None` and
//!   GLM-4.5's prints it as the word `None`; every other surveyed template renders both the same.
//! * `tool_call_id`, `name` and `reasoning_content` pass through.

use std::sync::Arc;

use serde_json::{json, Value};

use super::{ToolFormat, ToolSupport};
use crate::serve::openai::{ChatRequest, Message};

/// A 400 with the parameter it is about.
#[derive(Debug)]
pub struct Refusal {
    pub message: String,
    pub param: &'static str,
    pub code: &'static str,
}

fn invalid(param: &'static str, message: impl Into<String>) -> Refusal {
    Refusal { message: message.into(), param, code: "invalid_value" }
}

fn unsupported(param: &'static str, message: impl Into<String>) -> Refusal {
    Refusal { message: message.into(), param, code: "unsupported_parameter" }
}

/// One declared function.
#[derive(Clone, Debug)]
pub struct ToolDef {
    pub name: String,
    /// The `parameters` JSON schema (`null` when absent).
    pub params: Value,
    /// `function.strict`: the arguments are validated against `params`.
    pub strict: bool,
}

/// What `tool_choice` asks of the turn.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Force {
    #[default]
    Auto,
    /// At least one call, to any declared function.
    Required,
    /// Exactly one call, to this function.
    Named(String),
}

/// What the response parser needs to know about the request.
#[derive(Clone, Debug)]
pub struct ParseSpec {
    pub format: ToolFormat,
    pub tools: Arc<Vec<ToolDef>>,
    /// `parallel_tool_calls: false` keeps only the first call.
    pub parallel: bool,
    pub force: Force,
    /// Calls already in the conversation (Kimi-K2 numbers its ids across the conversation).
    pub history_calls: usize,
}

impl ParseSpec {
    /// A spec with no declared tools, for a format whose parser also splits plain answers.
    pub fn plain(format: ToolFormat) -> Self {
        ParseSpec { format, tools: Arc::new(Vec::new()), parallel: true, force: Force::Auto, history_calls: 0 }
    }

    pub fn tool(&self, name: &str) -> Option<&ToolDef> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// The call opener a forced `tool_choice` appends to the prompt.
    pub fn opener(&self) -> Option<String> {
        match &self.force {
            Force::Auto => None,
            Force::Required => Some(self.format.opener(None, self.history_calls)),
            Force::Named(n) => Some(self.format.opener(Some(n), self.history_calls)),
        }
    }

    /// The JSON-schema `type` of `tool`'s parameter `param`, when declared.
    pub fn param_type(&self, tool: &str, param: &str) -> Option<&str> {
        let params = &self.tool(tool)?.params;
        let ty = params.get("properties")?.get(param)?.get("type")?;
        match ty {
            Value::String(s) => Some(s),
            // `["string", "null"]`: the first non-null type.
            Value::Array(a) => a.iter().filter_map(Value::as_str).find(|t| *t != "null"),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct Plan {
    /// `tools` for the template; `None` when the request has none or `tool_choice` is `none`.
    pub template_tools: Option<Value>,
    /// Set when the generation must be parsed for calls.
    pub parse: Option<ParseSpec>,
}


/// Validate the request's tool fields against what the model's template can do. `support` is
/// `None` when the model is served without a chat template (built-in prompt builders).
pub fn plan(req: &ChatRequest, support: Option<ToolSupport>) -> Result<Plan, Refusal> {
    for (val, field) in [(&req.functions, "functions"), (&req.function_call, "function_call")] {
        if val.as_ref().is_some_and(|v| !v.is_null()) {
            return Err(unsupported(
                field,
                format!("`{field}` is the deprecated OpenAI function-calling API and is not served; send `tools` / `tool_choice`"),
            ));
        }
    }
    let tools = validate_tools(req.tools.as_ref())?;
    // `None`: the turn renders and parses no tools.
    let force = match req.tool_choice.as_ref() {
        None | Some(Value::Null) => Some(Force::Auto),
        Some(Value::String(s)) if s == "auto" => Some(Force::Auto),
        Some(Value::String(s)) if s == "none" => None,
        Some(Value::String(s)) if s == "required" => Some(Force::Required),
        Some(Value::Object(o)) if o.get("type").and_then(Value::as_str) == Some("function") => {
            // Chat Completions nests the name under `function`; the Responses API puts it at the top.
            let name = o.get("function").and_then(|f| f.get("name")).or_else(|| o.get("name")).and_then(Value::as_str);
            let Some(name) = name else {
                return Err(invalid("tool_choice", "`tool_choice.function.name` is required"));
            };
            Some(Force::Named(name.to_string()))
        }
        Some(Value::Object(o)) if o.get("type").and_then(Value::as_str).is_some() => {
            return Err(unsupported("tool_choice", format!("`tool_choice` type {} is not served", o["type"])))
        }
        Some(v) => {
            return Err(invalid(
                "tool_choice",
                format!("invalid `tool_choice` {v}; expected \"auto\", \"none\", \"required\" or a function"),
            ))
        }
    };
    if let Some(Force::Required | Force::Named(_)) = &force {
        let Some(list) = tools.as_ref() else {
            return Err(invalid("tool_choice", "`tool_choice` forces a call but the request declares no `tools`"));
        };
        if let Some(Force::Named(n)) = &force {
            if !list.iter().any(|t| t["function"]["name"].as_str() == Some(n.as_str())) {
                return Err(invalid("tool_choice", format!("`tool_choice` names `{n}`, which is not in `tools`")));
            }
        }
    }
    let history_calls = req.messages.iter().any(|m| m.tool_calls.as_ref().is_some_and(|v| !v.is_null()));
    if support.is_none() && history_calls {
        return Err(unsupported(
            "messages",
            "this model is served without a chat template, so assistant `tool_calls` in the conversation \
             cannot be rendered",
        ));
    }
    let (Some(tools), Some(force)) = (tools, force) else {
        return Ok(Plan::default());
    };
    let format = match support {
        Some(ToolSupport::Format(f)) => f,
        Some(ToolSupport::Unparsed) => {
            return Err(unsupported(
                "tools",
                "this model's chat template renders tools, but its tool-call output syntax is not one this \
                 server parses; refused rather than returning unparsed calls as text",
            ))
        }
        Some(ToolSupport::None) => {
            return Err(unsupported(
                "tools",
                "this model's chat template does not render `tools`; refused rather than answering without them",
            ))
        }
        None => {
            return Err(unsupported(
                "tools",
                "this model is served without a chat template, so `tools` cannot be rendered",
            ))
        }
    };
    let defs = tools
        .iter()
        .map(|t| {
            let f = &t["function"];
            ToolDef {
                name: f["name"].as_str().unwrap_or_default().to_string(),
                params: f.get("parameters").cloned().unwrap_or(Value::Null),
                strict: f.get("strict").and_then(Value::as_bool).unwrap_or(false),
            }
        })
        .collect();
    let history_calls = req
        .messages
        .iter()
        .filter_map(|m| m.tool_calls.as_ref().and_then(Value::as_array))
        .map(Vec::len)
        .sum();
    // A named function is one call.
    let parallel = req.parallel_tool_calls.unwrap_or(true) && !matches!(force, Force::Named(_));
    Ok(Plan {
        template_tools: Some(Value::Array(tools)),
        parse: Some(ParseSpec { format, tools: Arc::new(defs), parallel, force, history_calls }),
    })
}

fn valid_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 64 && n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// `None` for an absent or empty list.
fn validate_tools(tools: Option<&Value>) -> Result<Option<Vec<Value>>, Refusal> {
    let list = match tools {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(a)) if a.is_empty() => return Ok(None),
        Some(Value::Array(a)) => a,
        Some(_) => return Err(invalid("tools", "`tools` must be an array")),
    };
    let mut names = std::collections::HashSet::new();
    for (i, t) in list.iter().enumerate() {
        match t.get("type").and_then(Value::as_str) {
            Some("function") => {}
            Some(other) => return Err(unsupported("tools", format!("tools[{i}]: type `{other}` is not served; only `function`"))),
            None => return Err(invalid("tools", format!("tools[{i}]: missing `type: \"function\"`"))),
        }
        let Some(f) = t.get("function").filter(|f| f.is_object()) else {
            return Err(invalid("tools", format!("tools[{i}]: missing `function` object")));
        };
        let name = f.get("name").and_then(Value::as_str).unwrap_or_default();
        if !valid_name(name) {
            return Err(invalid("tools", format!("tools[{i}]: function name {name:?} must match ^[a-zA-Z0-9_-]{{1,64}}$")));
        }
        if !names.insert(name) {
            return Err(invalid("tools", format!("tools[{i}]: duplicate function name `{name}`")));
        }
        if f.get("parameters").is_some_and(|p| !p.is_object() && !p.is_null()) {
            return Err(invalid("tools", format!("tools[{i}]: `parameters` must be a JSON-schema object")));
        }
        if f.get("description").is_some_and(|d| !d.is_string() && !d.is_null()) {
            return Err(invalid("tools", format!("tools[{i}]: `description` must be a string")));
        }
        if f.get("strict").is_some_and(|d| !d.is_boolean() && !d.is_null()) {
            return Err(invalid("tools", format!("tools[{i}]: `strict` must be a boolean")));
        }
    }
    Ok(Some(list.clone()))
}

/// Why a conversation did not render.
#[derive(Debug)]
pub enum RenderError {
    /// The request itself is malformed.
    Request(Refusal),
    /// The template refused the conversation (`raise_exception` or a render failure).
    Template(String),
}

/// Render `messages` with the model's template: tool-call arguments as objects, and as the
/// client's JSON strings when the template only renders those.
pub fn render(
    t: &crate::serve::template::ChatTemplate,
    messages: &[Message],
    opts: &crate::serve::template::RenderOpts,
) -> Result<String, RenderError> {
    let msgs = template_messages(messages, true).map_err(RenderError::Request)?;
    match t.render_with(&msgs, opts) {
        Ok(p) => Ok(p),
        Err(e) if has_tool_calls(messages) => {
            let msgs = template_messages(messages, false).map_err(RenderError::Request)?;
            t.render_with(&msgs, opts).map_err(|_| RenderError::Template(e))
        }
        Err(e) => Err(RenderError::Template(e)),
    }
}

/// Whether any assistant turn carries calls.
pub fn has_tool_calls(messages: &[Message]) -> bool {
    messages.iter().any(|m| m.tool_calls.as_ref().is_some_and(|v| !v.is_null()))
}

/// The conversation as template messages (see the module doc for the mapping).
pub fn template_messages(messages: &[Message], args_as_objects: bool) -> Result<Vec<Value>, Refusal> {
    messages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mut v = json!({"role": m.role, "content": m.template_content()});
            if let Some(r) = &m.reasoning_content {
                v["reasoning_content"] = json!(r);
            }
            if let Some(n) = &m.name {
                v["name"] = json!(n);
            }
            if m.role == "tool" {
                let Some(id) = &m.tool_call_id else {
                    return Err(invalid("messages", format!("messages[{i}]: a `tool` message needs `tool_call_id`")));
                };
                v["tool_call_id"] = json!(id);
            }
            match m.tool_calls.as_ref().filter(|v| !v.is_null()) {
                None => {}
                Some(_) if m.role != "assistant" => {
                    return Err(invalid("messages", format!("messages[{i}]: only assistant messages carry `tool_calls`")))
                }
                Some(Value::Array(calls)) => {
                    let calls = calls
                        .iter()
                        .enumerate()
                        .map(|(j, c)| call(c, args_as_objects).map_err(|e| invalid("messages", format!("messages[{i}].tool_calls[{j}]: {e}"))))
                        .collect::<Result<Vec<_>, _>>()?;
                    if !calls.is_empty() {
                        v["tool_calls"] = Value::Array(calls);
                    }
                }
                Some(_) => return Err(invalid("messages", format!("messages[{i}]: `tool_calls` must be an array"))),
            }
            Ok(v)
        })
        .collect()
}

fn call(c: &Value, args_as_objects: bool) -> Result<Value, String> {
    if c.get("type").is_some_and(|t| t.as_str() != Some("function")) {
        return Err("`type` must be \"function\"".into());
    }
    let id = c.get("id").and_then(Value::as_str).ok_or("missing `id`")?;
    let f = c.get("function").ok_or("missing `function`")?;
    let name = f.get("name").and_then(Value::as_str).ok_or("missing `function.name`")?;
    let args = match f.get("arguments") {
        None | Some(Value::Null) => json!({}),
        Some(Value::String(s)) if s.trim().is_empty() => json!({}),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(o @ Value::Object(_)) => o,
            _ => return Err("`function.arguments` must be a JSON object string".into()),
        },
        Some(o @ Value::Object(_)) => o.clone(),
        Some(_) => return Err("`function.arguments` must be a JSON object string".into()),
    };
    let args = match (args_as_objects, f.get("arguments")) {
        (true, _) => args,
        // The client's own string, byte for byte, when the template wants a string.
        (false, Some(Value::String(s))) if !s.trim().is_empty() => Value::String(s.clone()),
        (false, _) => Value::String(serde_json::to_string(&args).unwrap_or_default()),
    };
    Ok(json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(body: Value) -> ChatRequest {
        let mut b = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        b.as_object_mut().unwrap().extend(body.as_object().unwrap().clone());
        serde_json::from_value(b).expect("request parses")
    }

    fn tool(name: &str) -> Value {
        json!({"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}})
    }

    const GEMMA: Option<ToolSupport> = Some(ToolSupport::Format(ToolFormat::Gemma4));

    #[test]
    fn auto_tools_are_rendered_and_parsed() {
        let p = plan(&req(json!({"tools": [tool("f")]})), GEMMA).unwrap();
        assert!(p.template_tools.is_some());
        let spec = p.parse.unwrap();
        assert_eq!(spec.format, ToolFormat::Gemma4);
        assert!(spec.parallel);
        assert_eq!(spec.param_type("f", "x"), Some("integer"));
        let p = plan(&req(json!({"tools": [tool("f")], "tool_choice": "auto", "parallel_tool_calls": false})), GEMMA).unwrap();
        assert!(!p.parse.unwrap().parallel);
    }

    #[test]
    fn tool_choice_none_renders_no_tools_and_parses_nothing() {
        let p = plan(&req(json!({"tools": [tool("f")], "tool_choice": "none"})), GEMMA).unwrap();
        assert!(p.template_tools.is_none() && p.parse.is_none());
        // no template support needed when the tools are not used
        assert!(plan(&req(json!({"tools": [tool("f")], "tool_choice": "none"})), Some(ToolSupport::None)).is_ok());
    }

    #[test]
    fn what_cannot_be_honored_is_refused() {
        let code = |body: Value, s: Option<ToolSupport>| plan(&req(body), s).err().map(|r| (r.param, r.code));
        let t = || json!([tool("f")]);
        assert_eq!(code(json!({"tools": t(), "tool_choice": "sometimes"}), GEMMA), Some(("tool_choice", "invalid_value")));
        assert_eq!(code(json!({"tool_choice": "required"}), GEMMA), Some(("tool_choice", "invalid_value")));
        assert_eq!(
            code(json!({"tools": t(), "tool_choice": {"type": "function", "function": {"name": "g"}}}), GEMMA),
            Some(("tool_choice", "invalid_value"))
        );
        assert_eq!(code(json!({"tools": t(), "tool_choice": {"type": "function"}}), GEMMA), Some(("tool_choice", "invalid_value")));
        assert_eq!(code(json!({"tools": t(), "tool_choice": {"type": "allowed_tools"}}), GEMMA), Some(("tool_choice", "unsupported_parameter")));
        assert_eq!(code(json!({"tools": t(), "tool_choice": "required"}), Some(ToolSupport::None)), Some(("tools", "unsupported_parameter")));
        assert_eq!(code(json!({"tools": [{"type": "function", "function": {"name": "f", "strict": "yes"}}]}), GEMMA), Some(("tools", "invalid_value")));
        assert_eq!(code(json!({"functions": [{"name": "f"}]}), GEMMA), Some(("functions", "unsupported_parameter")));
        assert_eq!(code(json!({"function_call": "auto"}), GEMMA), Some(("function_call", "unsupported_parameter")));
        assert_eq!(code(json!({"tools": t()}), Some(ToolSupport::None)), Some(("tools", "unsupported_parameter")));
        assert_eq!(code(json!({"tools": t()}), Some(ToolSupport::Unparsed)), Some(("tools", "unsupported_parameter")));
        assert_eq!(code(json!({"tools": t()}), None), Some(("tools", "unsupported_parameter")));
        assert_eq!(code(json!({"tools": {"a": 1}}), GEMMA), Some(("tools", "invalid_value")));
        assert_eq!(code(json!({"tools": [{"type": "code_interpreter"}]}), GEMMA), Some(("tools", "unsupported_parameter")));
        assert_eq!(code(json!({"tools": [tool("bad name")]}), GEMMA), Some(("tools", "invalid_value")));
        assert_eq!(code(json!({"tools": [tool("f"), tool("f")]}), GEMMA), Some(("tools", "invalid_value")));
        // empty / null lists carry no tools
        assert!(plan(&req(json!({"tools": []})), None).unwrap().parse.is_none());
        assert!(plan(&req(json!({"tools": null, "functions": null})), None).unwrap().parse.is_none());
        // tool-call history needs a template to render it
        let hist = json!({"messages": [{"role": "user", "content": "q"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]});
        assert_eq!(code(hist.clone(), None), Some(("messages", "unsupported_parameter")));
        assert!(plan(&req(hist), Some(ToolSupport::None)).is_ok());
    }

    #[test]
    fn forced_tool_choice_carries_its_opener() {
        let p = plan(&req(json!({"tools": [tool("f"), tool("g")], "tool_choice": "required"})), GEMMA).unwrap();
        let spec = p.parse.unwrap();
        assert_eq!((spec.force.clone(), spec.parallel), (Force::Required, true));
        assert_eq!(spec.opener().as_deref(), Some("<|tool_call>call:"));
        let named = json!({"tools": [tool("f"), tool("g")], "tool_choice": {"type": "function", "function": {"name": "g"}}});
        let spec = plan(&req(named), GEMMA).unwrap().parse.unwrap();
        assert_eq!((spec.force.clone(), spec.parallel), (Force::Named("g".into()), false), "a named function is one call");
        assert_eq!(spec.opener().as_deref(), Some("<|tool_call>call:g{"));
        // the Responses API shape
        let spec = plan(&req(json!({"tools": [tool("f")], "tool_choice": {"type": "function", "name": "f"}})), GEMMA).unwrap().parse.unwrap();
        assert_eq!(spec.force, Force::Named("f".into()));
        assert_eq!(plan(&req(json!({"tools": [tool("f")]})), GEMMA).unwrap().parse.unwrap().opener(), None);
        // Kimi-K2 numbers its ids across the conversation
        let kimi = Some(ToolSupport::Format(ToolFormat::KimiK2));
        let hist = json!({"tools": [tool("f")], "tool_choice": {"type": "function", "function": {"name": "f"}}, "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "functions.f:0", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "functions.f:1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "functions.f:0", "content": "1"},
            {"role": "tool", "tool_call_id": "functions.f:1", "content": "2"}]});
        let spec = plan(&req(hist), kimi).unwrap().parse.unwrap();
        assert!(spec.opener().unwrap().ends_with("functions.f:2<|tool_call_argument_begin|>"));
        let strict = json!({"tools": [{"type": "function", "function": {"name": "f", "strict": true, "parameters": {"type": "object"}}}]});
        assert!(plan(&req(strict), GEMMA).unwrap().parse.unwrap().tools[0].strict);
    }

    #[test]
    fn history_maps_like_vllm() {
        let r = req(json!({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": null, "reasoning_content": "think",
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1, \"s\": \"é\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "f", "content": "42"}
        ]}));
        let m = template_messages(&r.messages, true).unwrap();
        assert_eq!(m[1], json!({"role": "assistant", "content": "", "reasoning_content": "think",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"x": 1, "s": "é"}}}]}));
        assert_eq!(m[2], json!({"role": "tool", "content": "42", "name": "f", "tool_call_id": "c1"}));
        assert!(m[0].get("tool_calls").is_none(), "absent, not null: Llama tests `'tool_calls' in message`");
        let s = template_messages(&r.messages, false).unwrap();
        assert_eq!(s[1]["tool_calls"][0]["function"]["arguments"], json!("{\"x\": 1, \"s\": \"é\"}"));
    }

    #[test]
    fn malformed_history_is_refused() {
        let bad = |msgs: Value| template_messages(&req(json!({"messages": msgs})).messages, true).is_err();
        assert!(bad(json!([{"role": "tool", "content": "x"}])), "tool result without tool_call_id");
        assert!(bad(json!([{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{}"}}]}])), "no id");
        assert!(bad(json!([{"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "not json"}}]}])));
        assert!(bad(json!([{"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "[1]"}}]}])));
        assert!(bad(json!([{"role": "user", "tool_calls": []}])));
        assert!(!bad(json!([{"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": ""}}]}])));
    }
}

//! The checkpoint's OWN chat template, rendered.
//!
//! WHY THIS EXISTS. Every chat prompt used to be built by a hand-written Rust
//! function per model family, selected by probing the tokenizer for a marker
//! string. That has two failure modes and this server hit both:
//!
//!   * a family matching NO probe was served another family's markers, which
//!     its tokenizer spells out as ordinary text — a fluent answer to a prompt
//!     the model never really saw;
//!   * a family that DID match could still be wrong in detail. The GLM builder
//!     closed a thinking block the checkpoint's template leaves open, dropped a
//!     system line the template always emits, wrote assistant history turns
//!     without their think block, and rendered a `tool` message as a user turn.
//!
//! A template is data that ships WITH the weights. Reading it removes the whole
//! class. The hardcoded builders stay as the fallback for checkpoints that ship
//! no template (Kimi-K3 ships none).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use minijinja::{Environment, Value};

use super::{reasoning_markers, ToolSupport};

#[derive(Clone, Debug, Default)]
pub struct RenderOpts {
    /// vLLM `chat_template_kwargs`, bound as top-level template variables —
    /// which is what `transformers` does with `apply_chat_template(**kwargs)`.
    pub kwargs: BTreeMap<String, serde_json::Value>,
    /// OpenAI `reasoning_effort`.
    pub reasoning_effort: Option<String>,
    /// vLLM `continue_final_message`: render the last assistant turn as a
    /// prefix to continue instead of opening a fresh one.
    pub continue_final_message: bool,
    /// vLLM `add_generation_prompt: false`: end the prompt after the last message.
    pub no_generation_prompt: bool,
    /// OpenAI `tools`, bound as the template's `tools` (`None` renders as Python `None`).
    pub tools: Option<serde_json::Value>,
}

/// A checkpoint's chat template, compiled once at load.
pub struct ChatTemplate {
    env: Environment<'static>,
    /// Specials the HF templates reference by name.
    bos_token: Option<String>,
    eos_token: Option<String>,
    /// Where it came from, for the startup log.
    pub source: String,
    /// Whether the template renders `tools`, and the call syntax its model answers in.
    pub tools: ToolSupport,
    /// The reasoning-trace markers its model writes, when the template shows any.
    pub reasoning: Option<(&'static str, &'static str)>,
    /// The kwarg that switches the template's reasoning on, and its value when a request omits
    /// it (read from a probe render at load).
    pub thinking_knob: Option<(&'static str, bool)>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("source", &self.source)
            .finish()
    }
}

impl ChatTemplate {
    /// Compile the template shipped with the assets, if any (`dir` or `dir/checkpoint`).
    /// `Ok(None)`: no template; `Err`: one that does not compile.
    pub fn load(dir: &Path) -> Result<Option<Arc<ChatTemplate>>, String> {
        let Some(t) = crate::serve_manifest::find_chat_template(dir) else {
            return Ok(None);
        };
        Self::compile(t.text, t.source, t.bos_token, t.eos_token).map(Some)
    }

    /// Compile a template and read what its model answers in from the text: the tool-call
    /// syntax (a probe render with one tool), the reasoning markers, the thinking switch.
    pub fn compile(
        text: String,
        source: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<Arc<ChatTemplate>, String> {
        let mut t = Self::compile_bare(&text, source, bos_token, eos_token)?;
        t.reasoning = reasoning_markers(&text);
        t.tools = ToolSupport::probe(&t, &text);
        t.thinking_knob = t.probe_thinking_knob(&text);
        Ok(Arc::new(t))
    }

    /// Compile a template whose packet DECLARES the tool syntax and reasoning markers: nothing
    /// is read from the text but the thinking switch.
    pub fn compile_declared(
        text: String,
        source: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
        tools: ToolSupport,
        reasoning: Option<(&'static str, &'static str)>,
    ) -> Result<Arc<ChatTemplate>, String> {
        let mut t = Self::compile_bare(&text, source, bos_token, eos_token)?;
        t.tools = tools;
        t.reasoning = reasoning;
        t.thinking_knob = t.probe_thinking_knob(&text);
        Ok(Arc::new(t))
    }

    /// Whether a template compiles, with nothing read from it and nothing kept.
    pub fn check(
        text: &str,
        source: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<(), String> {
        Self::compile_bare(text, source, bos_token, eos_token).map(|_| ())
    }

    fn compile_bare(
        text: &str,
        source: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<ChatTemplate, String> {
        let mut env = Environment::new();
        // `transformers` compiles every chat template with these; without them a template that
        // relies on them (`{% ... %}` followed by a newline) renders extra whitespace.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // Python's `iter(None)` raises, so Jinja2's `none is iterable` is false; minijinja iterates
        // none as empty. `tools is iterable and tools | length > 0` (Qwen3-Coder) depends on it.
        env.add_test("iterable", |v: Value| !v.is_none() && v.try_iter().is_ok());
        // HF templates call `raise_exception` to reject a conversation shape.
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<Value, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        // PYTHON STRING METHODS. HF templates are written for Jinja2 running on
        // Python, so they call real `str` methods on content — GLM's calls
        // `.strip()` on an assistant turn. minijinja has filters, not methods,
        // so without this an ordinary multi-turn conversation fails to render
        // with "string has no method named strip".
        env.set_unknown_method_callback(|_state, value, method, args| {
            use minijinja::value::{from_args, ValueKind};
            // PYTHON DICT METHODS, for the same reason as the string methods
            // below: `message.get('tool_calls')` is how a template reads an
            // OPTIONAL key, and Gemma-4's and Kimi-K2.5's templates open with
            // one. Without it `render` fails for EVERY conversation, and
            // because the template compiled fine the built-in builders are
            // never reached — the server answers 400 to every chat request.
            if value.kind() == ValueKind::Map {
                match method {
                    "get" => {
                        let (key, default): (Value, Option<Value>) = from_args(args)?;
                        let got = value.get_item(&key)?;
                        // Python yields None for a missing key, which the
                        // `a.get(x) or a.get(y)` idiom relies on being falsy.
                        return Ok(if got.is_undefined() {
                            default.unwrap_or_else(|| Value::from(()))
                        } else {
                            got
                        });
                    }
                    "items" => {
                        let () = from_args(args)?;
                        let mut out = Vec::new();
                        for k in value.try_iter()? {
                            let v = value.get_item(&k)?;
                            out.push(Value::from(vec![k, v]));
                        }
                        return Ok(Value::from(out));
                    }
                    "keys" => {
                        let () = from_args(args)?;
                        return Ok(Value::from(value.try_iter()?.collect::<Vec<_>>()));
                    }
                    "values" => {
                        let () = from_args(args)?;
                        let mut out = Vec::new();
                        for k in value.try_iter()? {
                            out.push(value.get_item(&k)?);
                        }
                        return Ok(Value::from(out));
                    }
                    _ => {}
                }
            }
            let Some(s) = value.as_str() else {
                return Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod));
            };
            // Python's optional `chars` argument (`content.rstrip('\n')` in Qwen3's template).
            let strip_set = |chars: Option<&str>| -> Box<dyn Fn(char) -> bool> {
                match chars {
                    Some(c) => {
                        let c = c.to_string();
                        Box::new(move |x| c.contains(x))
                    }
                    None => Box::new(char::is_whitespace),
                }
            };
            match method {
                "strip" => {
                    let (c,): (Option<&str>,) = from_args(args)?;
                    let p = strip_set(c);
                    Ok(Value::from(s.trim_matches(|x| p(x))))
                }
                "lstrip" => {
                    let (c,): (Option<&str>,) = from_args(args)?;
                    let p = strip_set(c);
                    Ok(Value::from(s.trim_start_matches(|x| p(x))))
                }
                "rstrip" => {
                    let (c,): (Option<&str>,) = from_args(args)?;
                    let p = strip_set(c);
                    Ok(Value::from(s.trim_end_matches(|x| p(x))))
                }
                "replace" => {
                    let (from, to): (&str, &str) = from_args(args)?;
                    Ok(Value::from(s.replace(from, to)))
                }
                "startswith" => {
                    let (p,): (&str,) = from_args(args)?;
                    Ok(Value::from(s.starts_with(p)))
                }
                "endswith" => {
                    let (p,): (&str,) = from_args(args)?;
                    Ok(Value::from(s.ends_with(p)))
                }
                "lower" => {
                    let () = from_args(args)?;
                    Ok(Value::from(s.to_lowercase()))
                }
                "upper" => {
                    let () = from_args(args)?;
                    Ok(Value::from(s.to_uppercase()))
                }
                "split" => {
                    // Python's `maxsplit` (`content.split('</think>', 1)` in DeepSeek's template).
                    let (sep, max): (Option<&str>, Option<i64>) = from_args(args)?;
                    let n = max.filter(|&m| m >= 0).map(|m| m as usize + 1);
                    let parts: Vec<Value> = match (sep, n) {
                        (Some(sep), Some(n)) => s.splitn(n, sep).map(Value::from).collect(),
                        (Some(sep), None) => s.split(sep).map(Value::from).collect(),
                        (None, _) => s.split_whitespace().map(Value::from).collect(),
                    };
                    Ok(Value::from(parts))
                }
                _ => Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod)),
            }
        });
        // THE CURRENT DATE, which templates stamp into the system prompt.
        // `transformers` exposes it as `datetime.now().strftime(fmt)`, so this
        // is local time, not UTC. Three families here need it and each fails
        // differently without it: gpt-oss calls it UNGUARDED, so the render
        // dies with "undefined is not callable" and the server 400s on every
        // request; Llama-3.2 guards it and falls back to a hardcoded
        // "26 Jul 2024"; Muse-Glimmer guards it and drops its "Current date:"
        // line. The two guarded ones are worse than the loud one — the model is
        // told the wrong day and nothing anywhere says so.
        env.add_function(
            "strftime_now",
            |format: String| -> Result<String, minijinja::Error> {
                // Parsed rather than formatted straight through: chrono's `Display`
                // errors on an unknown specifier, and `to_string()` on that panics
                // — inside a request handler.
                let parsed = chrono::format::StrftimeItems::new(&format)
                    .parse()
                    .map_err(|e| {
                        minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            format!("strftime_now({format:?}): {e}"),
                        )
                    })?;
                Ok(chrono::Local::now()
                    .format_with_items(parsed.iter())
                    .to_string())
            },
        );
        // `tojson` as `transformers` defines it: Python's `json.dumps`, keyword arguments
        // (`ensure_ascii=False` by default, `indent`, `separators`, `sort_keys`) and spacing
        // (`", "` / `": "`) included, so a tool schema renders byte for byte as the reference.
        env.add_filter(
            "tojson",
            |v: Value, kwargs: minijinja::value::Kwargs| -> Result<String, minijinja::Error> {
                let err =
                    |m: String| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m);
                let mut opts = super::pyjson::Opts {
                    ensure_ascii: kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false),
                    indent: kwargs.get::<Option<usize>>("indent")?,
                    sort_keys: kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false),
                    separators: None,
                };
                if let Some(seps) = kwargs.get::<Option<Vec<String>>>("separators")? {
                    let [item, key] = <[String; 2]>::try_from(seps)
                        .map_err(|_| err("tojson: `separators` must be a pair".into()))?;
                    opts.separators = Some((item, key));
                }
                kwargs.assert_all_used()?;
                let json = serde_json::to_value(&v).map_err(|e| err(e.to_string()))?;
                Ok(super::pyjson::dumps(&json, &opts))
            },
        );
        // A template that will not COMPILE is a defect in the assets, and falling back silently
        // is how a wrong prompt ships: the caller decides, with the file named.
        env.add_template_owned("chat", neutral_generation_blocks(text))
            .map_err(|e| format!("{source}: {e}"))?;
        Ok(ChatTemplate {
            env,
            bos_token,
            eos_token,
            source,
            tools: ToolSupport::None,
            reasoning: None,
            thinking_knob: None,
        })
    }
    /// Whether a request's `chat_template_kwargs` leave the template's reasoning switch on.
    pub fn thinking_requested(&self, kwargs: &BTreeMap<String, serde_json::Value>) -> bool {
        self.thinking_knob.is_some_and(|(knob, default)| {
            kwargs.get(knob).map_or(default, |v| match v {
                serde_json::Value::Null => false,
                serde_json::Value::Bool(b) => *b,
                serde_json::Value::Number(n) => n.as_f64() != Some(0.0),
                serde_json::Value::String(s) => !s.is_empty(),
                serde_json::Value::Array(a) => !a.is_empty(),
                serde_json::Value::Object(o) => !o.is_empty(),
            })
        })
    }

    /// The first of the known reasoning switches the template reads, with its default: a render
    /// without the kwarg equal to the one with it `true` means reasoning is on by default.
    fn probe_thinking_knob(&self, text: &str) -> Option<(&'static str, bool)> {
        let knob = ["enable_thinking", "thinking"]
            .into_iter()
            .find(|k| text.contains(k))?;
        let msgs = [serde_json::json!({"role": "user", "content": "hi"})];
        let with = |on: Option<bool>| {
            let mut opts = RenderOpts::default();
            if let Some(on) = on {
                opts.kwargs.insert(knob.into(), serde_json::Value::Bool(on));
            }
            self.render_with(&msgs, &opts).ok()
        };
        let (none, on, off) = (with(None), with(Some(true)), with(Some(false)));
        // A knob that changes nothing is not a switch.
        if none.is_none() || on == off {
            return None;
        }
        Some((knob, none == on))
    }

    /// Render `messages` with `add_generation_prompt=true` and no extra
    /// template variables.
    pub fn render(&self, messages: &[serde_json::Value]) -> Result<String, String> {
        self.render_with(messages, &RenderOpts::default())
    }

    /// Render `messages` under `opts`.
    ///
    /// Returns `Err` with the template's own message when the conversation is
    /// one the template refuses (`raise_exception`), so the caller can answer
    /// 400 instead of serving a malformed prompt.
    pub fn render_with(
        &self,
        messages: &[serde_json::Value],
        opts: &RenderOpts,
    ) -> Result<String, String> {
        let tmpl = self.env.get_template("chat").map_err(|e| e.to_string())?;

        // The base variables `transformers` always binds, then the caller's
        // `chat_template_kwargs` on top. Merged rather than fixed because
        // `enable_thinking` — the flag every Qwen3/GLM client uses to turn
        // reasoning off — is just one of an open set a template may read, and
        // a fixed context silently drops all of them.
        let mut ctx: BTreeMap<String, Value> = BTreeMap::new();
        ctx.insert("messages".into(), Value::from_serialize(messages));
        ctx.insert(
            "add_generation_prompt".into(),
            Value::from(!opts.continue_final_message && !opts.no_generation_prompt),
        );
        ctx.insert(
            "continue_final_message".into(),
            Value::from(opts.continue_final_message),
        );
        ctx.insert("bos_token".into(), Value::from(self.bos_token.clone()));
        ctx.insert("eos_token".into(), Value::from(self.eos_token.clone()));
        ctx.insert(
            "tools".into(),
            opts.tools
                .as_ref()
                .map_or(Value::from(()), Value::from_serialize),
        );
        if let Some(effort) = &opts.reasoning_effort {
            ctx.insert("reasoning_effort".into(), Value::from(effort.clone()));
        }
        for (k, v) in &opts.kwargs {
            ctx.insert(k.clone(), Value::from_serialize(v));
        }

        let rendered = tmpl.render(Value::from_serialize(&ctx)).map_err(|e| {
            // minijinja chains the cause; the innermost is the template's own
            // `raise_exception` message, which is the useful half.
            let mut msg = e.to_string();
            let mut src: &dyn std::error::Error = &e;
            while let Some(next) = src.source() {
                msg = next.to_string();
                src = next;
            }
            msg
        })?;
        if opts.continue_final_message {
            return continue_final(rendered, messages);
        }
        Ok(rendered)
    }
}

/// `transformers`' `{% generation %}...{% endgeneration %}` (the assistant-token mask for
/// training) renders its body unchanged; minijinja has no such statement, so each tag becomes an
/// always-true `if` with the same whitespace control.
fn neutral_generation_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("{%") {
        let Some(j) = rest[i..].find("%}") else { break };
        let tag = &rest[i..i + j + 2];
        let inner = tag[2..tag.len() - 2].trim_matches(|c: char| c == '-' || c == '+');
        out.push_str(&rest[..i]);
        match inner.trim() {
            "generation" => out.push_str(&tag.replacen("generation", "if true", 1)),
            "endgeneration" => out.push_str(&tag.replacen("endgeneration", "endif", 1)),
            _ => out.push_str(tag),
        }
        rest = &rest[i + j + 2..];
    }
    out.push_str(rest);
    out
}

/// `transformers`' `continue_final_message`: cut the render right after the final message's
/// text, so the model continues it instead of seeing its turn closed (most templates, Gemma 4's
/// among them, close every turn whatever `add_generation_prompt` says).
fn continue_final(mut rendered: String, messages: &[serde_json::Value]) -> Result<String, String> {
    let content = messages.last().and_then(|m| m.get("content"));
    let text = match content {
        Some(serde_json::Value::String(s)) => Some(s.as_str()),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .rev()
            .find_map(|p| p.get("text").and_then(|t| t.as_str())),
        _ => None,
    }
    .ok_or("continue_final_message is set but the final message has no text to continue")?;
    let at = rendered.rfind(text.trim()).ok_or(
        "continue_final_message is set but the final message does not appear in the chat after \
         applying the chat template",
    )?;
    let lead = text.trim_start();
    let end = if rendered[at..].starts_with(lead) {
        at + lead.len()
    } else {
        at + text.trim().len()
    };
    rendered.truncate(end);
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    #[test]
    fn generation_blocks_become_always_true_ifs() {
        let t = "a{%- generation %}b{% endgeneration -%}c{% if x %}d{% endif %}";
        assert_eq!(
            super::neutral_generation_blocks(t),
            "a{%- if true %}b{% endif -%}c{% if x %}d{% endif %}"
        );
    }
}

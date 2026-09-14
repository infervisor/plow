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

use std::path::Path;
use std::sync::Arc;

use minijinja::{Environment, Value};

/// A checkpoint's chat template, compiled once at load.
pub struct ChatTemplate {
    env: Environment<'static>,
    /// Specials the HF templates reference by name.
    bos_token: Option<String>,
    eos_token: Option<String>,
    /// Where it came from, for the startup log.
    pub source: String,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("source", &self.source)
            .finish()
    }
}

fn read_to_string(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok().filter(|s| !s.trim().is_empty())
}

/// The template text plus the special tokens, from `dir` or `dir/checkpoint`.
///
/// HF puts the template in one of two places and BOTH are in the wild:
/// a standalone `chat_template.jinja`, or a `chat_template` string inside
/// `tokenizer_config.json`. The standalone file wins when both exist, which is
/// what `transformers` does.
fn find(dir: &Path) -> Option<(String, String, Option<String>, Option<String>)> {
    for base in [dir.to_path_buf(), dir.join("checkpoint")] {
        let jinja = base.join("chat_template.jinja");
        let cfg_path = base.join("tokenizer_config.json");
        let cfg: Option<serde_json::Value> = std::fs::read_to_string(&cfg_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        let tok = |k: &str| -> Option<String> {
            let v = cfg.as_ref()?.get(k)?;
            match v {
                serde_json::Value::String(s) => Some(s.clone()),
                // HF also writes `{"content": "<bos>", ...}` here.
                serde_json::Value::Object(o) => {
                    o.get("content")?.as_str().map(str::to_string)
                }
                _ => None,
            }
        };
        if let Some(text) = read_to_string(&jinja) {
            return Some((text, jinja.display().to_string(), tok("bos_token"), tok("eos_token")));
        }
        let json_path = base.join("chat_template.json");
        if let Some(text) = read_to_string(&json_path)
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("chat_template")?.as_str().map(str::to_owned))
            .filter(|s| !s.trim().is_empty())
        {
            return Some((
                text,
                json_path.display().to_string(),
                tok("bos_token"),
                tok("eos_token"),
            ));
        }
        if let Some(text) = cfg
            .as_ref()
            .and_then(|c| c.get("chat_template"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .filter(|s| !s.trim().is_empty())
        {
            return Some((
                text,
                format!("{}#chat_template", cfg_path.display()),
                tok("bos_token"),
                tok("eos_token"),
            ));
        }
    }
    None
}

impl ChatTemplate {
    /// Compile the template shipped with the assets, if any.
    pub fn load(dir: &Path) -> Option<Arc<ChatTemplate>> {
        let (text, source, bos_token, eos_token) = find(dir)?;
        let mut env = Environment::new();
        // HF templates call `raise_exception` to reject a conversation shape.
        env.add_function("raise_exception", |msg: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });
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
                return Err(minijinja::Error::from(
                    minijinja::ErrorKind::UnknownMethod,
                ));
            };
            match method {
                "strip" => {
                    let () = from_args(args)?;
                    Ok(Value::from(s.trim()))
                }
                "lstrip" => {
                    let () = from_args(args)?;
                    Ok(Value::from(s.trim_start()))
                }
                "rstrip" => {
                    let () = from_args(args)?;
                    Ok(Value::from(s.trim_end()))
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
                    let (sep,): (Option<&str>,) = from_args(args)?;
                    let parts: Vec<Value> = match sep {
                        Some(sep) => s.split(sep).map(Value::from).collect(),
                        None => s.split_whitespace().map(Value::from).collect(),
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
        env.add_function("strftime_now", |format: String| -> Result<String, minijinja::Error> {
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
        });
        // `tojson` under a different spelling, used by tool-calling templates.
        env.add_filter("tojson", |v: Value| -> Result<String, minijinja::Error> {
            serde_json::to_string(&v)
                .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))
        });
        if let Err(e) = env.add_template_owned("chat", text) {
            // A template that will not COMPILE is a defect in the assets, and
            // falling back silently is how a wrong prompt ships. Say which file.
            tracing::error!(%source, error = %e, "chat template failed to compile — falling back to the built-in prompt builders");
            return None;
        }
        Some(Arc::new(ChatTemplate {
            env,
            bos_token,
            eos_token,
            source,
        }))
    }

    /// Render `messages` with `add_generation_prompt=true`.
    ///
    /// Returns `Err` with the template's own message when the conversation is
    /// one the template refuses (`raise_exception`), so the caller can answer
    /// 400 instead of serving a malformed prompt.
    pub fn render(&self, messages: &[serde_json::Value]) -> Result<String, String> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| e.to_string())?;
        tmpl.render(minijinja::context! {
            messages => Value::from_serialize(messages),
            add_generation_prompt => true,
            bos_token => self.bos_token.clone(),
            eos_token => self.eos_token.clone(),
            tools => Value::from(()),
        })
        .map_err(|e| {
            // minijinja chains the cause; the innermost is the template's own
            // `raise_exception` message, which is the useful half.
            let mut msg = e.to_string();
            let mut src: &dyn std::error::Error = &e;
            while let Some(next) = src.source() {
                msg = next.to_string();
                src = next;
            }
            msg
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ChatTemplate;

    /// Render the REAL checkpoint template and compare against the string
    /// `transformers` produces for the same conversation. Skipped when the
    /// weights are not on this host, so it is a gate where it can be one and
    /// silent where it cannot.
    #[test]
    fn the_real_glm53_template_renders_what_transformers_renders() {
        let dir = std::path::Path::new("/workspace/models/GLM-5.3-FP8");
        if !dir.join("chat_template.jinja").exists() {
            eprintln!("skipped: no GLM-5.3 checkpoint on this host");
            return;
        }
        let t = ChatTemplate::load(dir).expect("template compiles");
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "You are helpful."}),
            serde_json::json!({"role": "user", "content": "Hi there"}),
        ];
        assert_eq!(
            t.render(&msgs).expect("renders"),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>You are helpful.\
             <|user|>Hi there<|assistant|><think>"
        );
    }

    /// A MULTI-TURN conversation, which is what exercises the Python string
    /// methods HF templates call (`content.strip()` on an assistant turn).
    /// Without them minijinja fails the whole render with "string has no method
    /// named strip" and every conversation with history 400s.
    #[test]
    fn a_multi_turn_conversation_renders() {
        let dir = std::path::Path::new("/workspace/models/GLM-5.3-FP8");
        if !dir.join("chat_template.jinja").exists() {
            eprintln!("skipped: no GLM-5.3 checkpoint on this host");
            return;
        }
        let t = ChatTemplate::load(dir).expect("template compiles");
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "one"}),
            serde_json::json!({"role": "assistant", "content": "two"}),
            serde_json::json!({"role": "user", "content": "three"}),
        ];
        let out = t.render(&msgs).expect("multi-turn renders");
        assert!(out.contains("<|assistant|>"), "{out}");
        assert!(out.ends_with("<|assistant|><think>"), "{out}");
    }

    /// HF templates read an OPTIONAL message key with `message.get(...)`.
    /// Gemma-4's and Kimi-K2.5's both do it on the first pass over `messages`,
    /// so without dict methods `render` failed for every conversation — and
    /// because the template itself COMPILED, the built-in builders were never
    /// reached and the server answered 400 to every chat request for those
    /// families. Host-independent, so it gates everywhere the string-method
    /// test cannot.
    #[test]
    fn python_dict_methods_render() {
        let d = std::env::temp_dir().join("plow-template-dict-methods-test");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("chat_template.jinja"),
            "{% for m in messages %}{{ m.get('name') or m['role'] }}\
             ={{ m.get('missing', 'dflt') }};{% endfor %}\
             |{% for k, v in messages[0].items() %}[{{ k }}]{% endfor %}",
        )
        .unwrap();
        let t = ChatTemplate::load(&d).expect("template compiles");
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "name": "bot", "content": "yo"}),
        ];
        let out = t.render(&msgs).expect("renders");
        let (turns, items) = out.split_once('|').expect("both halves rendered");
        assert_eq!(turns, "user=dflt;bot=dflt;");
        assert!(items.contains("[role]") && items.contains("[content]"), "{items}");
    }

    /// `strftime_now` stamps the current date into the system prompt. gpt-oss
    /// calls it UNGUARDED (`chat_template.jinja:202`), so without it the render
    /// dies with "undefined is not callable" and the server answers 400 to
    /// every request; Llama-3.2 and Muse-Glimmer guard it with `is defined` and
    /// quietly serve a stale (`26 Jul 2024`) or missing date instead.
    #[test]
    fn strftime_now_stamps_the_current_date() {
        let d = std::env::temp_dir().join("plow-template-strftime-test");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("chat_template.jinja"),
            "{{ strftime_now('%Y-%m-%d') }}|{{ strftime_now('%d %b %Y') }}",
        )
        .unwrap();
        let t = ChatTemplate::load(&d).expect("template compiles");
        let now = chrono::Local::now();
        assert_eq!(
            t.render(&[]).expect("renders"),
            format!("{}|{}", now.format("%Y-%m-%d"), now.format("%d %b %Y"))
        );
    }

    /// An unknown specifier must be a render error, not a panic: chrono's
    /// `Display` reports it by failing to format, and `to_string()` on that
    /// panics — which here would be a panic inside a request handler.
    #[test]
    fn a_bad_strftime_format_is_an_error_not_a_panic() {
        let d = std::env::temp_dir().join("plow-template-strftime-bad-test");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("chat_template.jinja"), "{{ strftime_now('%Q') }}").unwrap();
        let t = ChatTemplate::load(&d).expect("template compiles");
        assert!(t.render(&[]).is_err());
    }

    /// The REAL Gemma-4 template, which is the one `.get()` was blocking.
    #[test]
    fn the_real_gemma4_template_renders_what_transformers_renders() {
        let dir = std::path::Path::new("/workspace/models/gemma-4-12B-it-cpu");
        if !dir.join("chat_template.jinja").exists() {
            eprintln!("skipped: no Gemma-4 checkpoint on this host");
            return;
        }
        let t = ChatTemplate::load(dir).expect("template compiles");
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "You are helpful."}),
            serde_json::json!({"role": "user", "content": "Hi"}),
        ];
        assert_eq!(
            t.render(&msgs).expect("renders"),
            "<bos><|turn>system\nYou are helpful.<turn|>\n\
             <|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    /// A checkpoint with no template must return `None`, not panic — that is
    /// what keeps the built-in builders reachable for Kimi-K3, which ships none.
    #[test]
    fn a_directory_without_a_template_yields_none() {
        let d = std::env::temp_dir().join("plow-template-absent-test");
        std::fs::create_dir_all(&d).unwrap();
        assert!(ChatTemplate::load(&d).is_none());
    }
}

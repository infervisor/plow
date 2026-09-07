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

    /// A checkpoint with no template must return `None`, not panic — that is
    /// what keeps the built-in builders reachable for Kimi-K3, which ships none.
    #[test]
    fn a_directory_without_a_template_yields_none() {
        let d = std::env::temp_dir().join("plow-template-absent-test");
        std::fs::create_dir_all(&d).unwrap();
        assert!(ChatTemplate::load(&d).is_none());
    }
}

//! The checkpoint's OWN chat template ([`plow_asset::chat::template`]), as the server loads it:
//! a template that does not compile is logged and yields `None`, which sends a legacy or v1
//! packet to the built-in prompt builders (a v2 packet refuses to load instead, in
//! `asset::ModelBundle::load`).

use std::path::Path;
use std::sync::Arc;

pub use plow_asset::chat::template::{ChatTemplate, RenderOpts};

/// Compile the template shipped with the assets, if any (`dir` or `dir/checkpoint`).
pub fn load(dir: &Path) -> Option<Arc<ChatTemplate>> {
    match ChatTemplate::load(dir) {
        Ok(Some(t)) => {
            tracing::info!(source = %t.source, tools = ?t.tools, "chat template tool support");
            Some(t)
        }
        Ok(None) => None,
        Err(e) => {
            tracing::error!(error = %e, "chat template failed to compile — falling back to the built-in prompt builders");
            None
        }
    }
}

/// Compile a template carried by the packet, reading its tool syntax and reasoning markers from
/// the text (legacy and version-1 packets).
pub fn compile(
    text: String,
    source: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
) -> Option<Arc<ChatTemplate>> {
    match ChatTemplate::compile(text, source, bos_token, eos_token) {
        Ok(t) => {
            tracing::info!(source = %t.source, tools = ?t.tools, "chat template tool support");
            Some(t)
        }
        Err(e) => {
            tracing::error!(error = %e, "chat template failed to compile — falling back to the built-in prompt builders");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::load;

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
        let t = load(dir).expect("template compiles");
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
        let t = load(dir).expect("template compiles");
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
        let t = load(&d).expect("template compiles");
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "name": "bot", "content": "yo"}),
        ];
        let out = t.render(&msgs).expect("renders");
        let (turns, items) = out.split_once('|').expect("both halves rendered");
        assert_eq!(turns, "user=dflt;bot=dflt;");
        assert!(
            items.contains("[role]") && items.contains("[content]"),
            "{items}"
        );
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
        let t = load(&d).expect("template compiles");
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
        let t = load(&d).expect("template compiles");
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
        let t = load(dir).expect("template compiles");
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
        assert!(load(&d).is_none());
    }
}

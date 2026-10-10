//! OpenAI tool calling (`tools`, `tool_choice`, `tool_calls`, `role: "tool"`) for any model whose
//! chat template renders tools.
//!
//! Two halves, both model-agnostic:
//!
//! * REQUEST. The checkpoint's own template renders the tool schemas and the tool-call history
//!   ([`request`]), with the conversation mapped the way `transformers` + vLLM hand it to
//!   `apply_chat_template`. Whether a template renders `tools` at all is PROBED at load
//!   ([`ToolSupport::probe`]): a template that ignores them (Mixtral, DeepSeek-V3.x) refuses the
//!   request with a 400 instead of silently answering without tools.
//! * RESPONSE. Each model family answers in its own call syntax. [`ToolFormat`] is selected from
//!   the template's own markers, never from the model name, and [`parse`] holds one streaming
//!   parser per syntax. The text is decoded with special tokens KEPT for these requests (most
//!   families make their call markers special tokens), parsed, and only then stripped.

pub mod parse;
pub mod pyjson;
pub mod request;
pub mod schema;
pub mod stream;

#[cfg(test)]
mod http_tests;
#[cfg(test)]
mod parity_tests;

use std::sync::atomic::{AtomicU64, Ordering};

/// A model family's tool-call output syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolFormat {
    /// Gemma 4: `<|tool_call>call:NAME{key:<|"|>str<|"|>,n:1}<tool_call|>`.
    Gemma4,
    /// Hermes / Qwen2.5 / Qwen3: `<tool_call>\n{"name": .., "arguments": {..}}\n</tool_call>`.
    Hermes,
    /// Qwen3-Coder / Qwen3.5: `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`.
    Qwen3Xml,
    /// GLM-4.5 .. 5.x: `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`.
    Glm45,
    /// Llama 3.1 / 3.2 / 3.3: a bare `{"name": .., "parameters": {..}}`, optionally after
    /// `<|python_tag|>`, several joined by `;`.
    Llama3Json,
    /// Mistral: `[TOOL_CALLS] [{"name": .., "arguments": {..}}]` (v3) or
    /// `[TOOL_CALLS]NAME[ARGS]{..}` (v11+). Call ids are 9 alphanumerics.
    Mistral,
    /// Kimi-K2: `<|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{..}<|tool_call_end|>...`.
    KimiK2,
    /// gpt-oss harmony: `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{..}`.
    Harmony,
    /// DeepSeek V3 / R1: `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\n{..}\n```<｜tool▁call▁end｜>`.
    DeepSeekV3,
    /// DeepSeek V3.1: `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{..}<｜tool▁call▁end｜>`.
    DeepSeekV31,
}

impl ToolFormat {
    pub fn name(self) -> &'static str {
        match self {
            ToolFormat::Gemma4 => "gemma4",
            ToolFormat::Hermes => "hermes",
            ToolFormat::Qwen3Xml => "qwen3_xml",
            ToolFormat::Glm45 => "glm45",
            ToolFormat::Llama3Json => "llama3_json",
            ToolFormat::Mistral => "mistral",
            ToolFormat::KimiK2 => "kimi_k2",
            ToolFormat::Harmony => "harmony",
            ToolFormat::DeepSeekV3 => "deepseek_v3",
            ToolFormat::DeepSeekV31 => "deepseek_v31",
        }
    }

    /// The syntax a template teaches its model, read from the template's own call markers. The
    /// order matters: GLM and Qwen3.5 templates also contain `<tool_call>`.
    pub fn detect(template: &str) -> Option<Self> {
        let has = |m: &str| template.contains(m);
        Some(if has("<|tool_call>") && has("call:") {
            ToolFormat::Gemma4
        } else if has("<arg_key>") {
            ToolFormat::Glm45
        } else if has("<tool_call>") && has("<function=") {
            ToolFormat::Qwen3Xml
        } else if has("<tool_call>") {
            ToolFormat::Hermes
        } else if has("<|tool_calls_section_begin|>") {
            ToolFormat::KimiK2
        } else if has("<｜tool▁calls▁begin｜>") && has("```json") {
            ToolFormat::DeepSeekV3
        } else if has("<｜tool▁calls▁begin｜>") {
            ToolFormat::DeepSeekV31
        } else if has("[TOOL_CALLS]") {
            ToolFormat::Mistral
        } else if has("<|channel|>") && has("functions.") {
            ToolFormat::Harmony
        } else if has("<|python_tag|>") || (has("<|start_header_id|>") && has("\"parameters\"")) {
            ToolFormat::Llama3Json
        } else {
            return None;
        })
    }

    /// The text that opens a call in this syntax, up to the function name, and through it when
    /// `name` is given: appended to the prompt it makes the generation a call
    /// (`tool_choice: "required"` / a named function). `call_index` numbers Kimi-K2's ids.
    pub fn opener(self, name: Option<&str>, call_index: usize) -> String {
        let n = name.unwrap_or("");
        match (self, name.is_some()) {
            (ToolFormat::Gemma4, false) => "<|tool_call>call:".into(),
            (ToolFormat::Gemma4, true) => format!("<|tool_call>call:{n}{{"),
            (ToolFormat::Hermes, false) => "<tool_call>\n{\"name\": \"".into(),
            (ToolFormat::Hermes, true) => format!("<tool_call>\n{{\"name\": \"{n}\", \"arguments\": "),
            (ToolFormat::Qwen3Xml, false) => "<tool_call>\n<function=".into(),
            (ToolFormat::Qwen3Xml, true) => format!("<tool_call>\n<function={n}>\n"),
            (ToolFormat::Glm45, _) => format!("<tool_call>{n}"),
            (ToolFormat::Llama3Json, false) => "{\"name\": \"".into(),
            (ToolFormat::Llama3Json, true) => format!("{{\"name\": \"{n}\", \"parameters\": "),
            (ToolFormat::Mistral, false) => "[TOOL_CALLS] [{\"name\": \"".into(),
            (ToolFormat::Mistral, true) => format!("[TOOL_CALLS] [{{\"name\": \"{n}\", \"arguments\": "),
            (ToolFormat::KimiK2, false) => "<|tool_calls_section_begin|><|tool_call_begin|>functions.".into(),
            (ToolFormat::KimiK2, true) => {
                format!("<|tool_calls_section_begin|><|tool_call_begin|>functions.{n}:{call_index}<|tool_call_argument_begin|>")
            }
            (ToolFormat::Harmony, false) => "<|channel|>commentary to=functions.".into(),
            (ToolFormat::Harmony, true) => format!("<|channel|>commentary to=functions.{n} <|constrain|>json<|message|>"),
            (ToolFormat::DeepSeekV3, false) => "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>".into(),
            (ToolFormat::DeepSeekV3, true) => format!("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{n}\n```json\n"),
            (ToolFormat::DeepSeekV31, false) => "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>".into(),
            (ToolFormat::DeepSeekV31, true) => format!("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{n}<｜tool▁sep｜>"),
        }
    }

    /// The `id` a parsed call gets on the wire.
    pub fn call_id(self) -> String {
        match self {
            // Its template refuses any id that is not 9 characters long.
            ToolFormat::Mistral => random_alnum(9),
            _ => format!("call_{}", random_alnum(24)),
        }
    }
}

/// The reasoning-trace markers a template's model writes, read from the template:
/// Gemma 4's thought channel and `<think>`. Harmony's channels are its parser's.
pub fn reasoning_markers(template: &str) -> Option<(&'static str, &'static str)> {
    if template.contains("<|channel>thought") && template.contains("<channel|>") {
        Some(("<|channel>thought", "<channel|>"))
    } else if template.contains("<think>") && template.contains("</think>") {
        Some(("<think>", "</think>"))
    } else {
        None
    }
}

/// What a chat template can do with `tools`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolSupport {
    /// The template never renders `tools`.
    None,
    /// It renders them, but in a call syntax this server does not parse.
    Unparsed,
    Format(ToolFormat),
}

/// A tool name no real conversation uses, for the load-time probe.
const PROBE_NAME: &str = "plow_probe_fn_7f3a";

impl ToolSupport {
    /// Render a one-message conversation with a probe tool: a template that renders `tools` puts
    /// the probe's name in the prompt.
    pub fn probe(t: &crate::serve::template::ChatTemplate, text: &str) -> Self {
        let tools = serde_json::json!([{"type": "function", "function": {
            "name": PROBE_NAME, "description": "probe",
            "parameters": {"type": "object", "properties": {"x": {"type": "string", "description": "x"}}, "required": ["x"]}
        }}]);
        let opts = crate::serve::template::RenderOpts { tools: Some(tools), ..Default::default() };
        let msgs = [serde_json::json!({"role": "user", "content": "hi"})];
        match t.render_with(&msgs, &opts) {
            Ok(p) if p.contains(PROBE_NAME) => match ToolFormat::detect(text) {
                Some(f) => ToolSupport::Format(f),
                None => ToolSupport::Unparsed,
            },
            _ => ToolSupport::None,
        }
    }
}

/// Special-token text, removed from what a client sees after the tool parser has read it.
#[derive(Debug)]
pub struct SpecialText {
    /// Longest first, so a token that prefixes another does not cut it short.
    tokens: Vec<String>,
    first: [bool; 256],
    longest: usize,
}

impl SpecialText {
    pub fn new(mut tokens: Vec<String>) -> Self {
        tokens.retain(|t| !t.is_empty());
        tokens.sort_by_key(|t| std::cmp::Reverse(t.len()));
        tokens.dedup();
        let mut first = [false; 256];
        for t in &tokens {
            first[t.as_bytes()[0] as usize] = true;
        }
        let longest = tokens.first().map_or(0, String::len);
        SpecialText { tokens, first, longest }
    }

    pub fn strip(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        self.strip_into(s, &mut out);
        out
    }

    /// [`Self::strip`], appended to `out`.
    pub fn strip_into(&self, s: &str, out: &mut String) {
        let Some(first) = s.bytes().position(|b| self.first[b as usize]) else {
            out.push_str(s);
            return;
        };
        out.push_str(&s[..first]);
        let mut i = first;
        'outer: while i < s.len() {
            if self.first[s.as_bytes()[i] as usize] {
                for t in &self.tokens {
                    if s[i..].starts_with(t.as_str()) {
                        i += t.len();
                        continue 'outer;
                    }
                }
            }
            // A token's first byte is never a UTF-8 continuation byte, so `next` is a char boundary.
            let next = s.as_bytes()[i + 1..].iter().position(|&b| self.first[b as usize]).map_or(s.len(), |k| i + 1 + k);
            out.push_str(&s[i..next]);
            i = next;
        }
    }

    /// [`Self::strip_into`] over a stream of pieces: `hold` carries a tail that may begin a token
    /// into the next piece (`last` flushes it), so a token split across pieces is still removed.
    pub fn strip_stream(&self, hold: &mut String, s: &str, out: &mut String, last: bool) {
        if hold.is_empty() {
            let keep = if last { 0 } else { self.partial_tail(s) };
            self.strip_into(&s[..s.len() - keep], out);
            hold.push_str(&s[s.len() - keep..]);
            return;
        }
        hold.push_str(s);
        let w = std::mem::take(hold);
        let keep = if last { 0 } else { self.partial_tail(&w) };
        self.strip_into(&w[..w.len() - keep], out);
        hold.push_str(&w[w.len() - keep..]);
    }

    /// Bytes at the end of `s` that are a proper prefix of some token.
    fn partial_tail(&self, s: &str) -> usize {
        let b = s.as_bytes();
        let from = b.len().saturating_sub(self.longest.saturating_sub(1));
        (from..b.len())
            .find(|&p| {
                self.first[b[p] as usize] && {
                    let rest = &s[p..];
                    self.tokens.iter().any(|t| t.len() > rest.len() && t.starts_with(rest))
                }
            })
            .map_or(0, |p| b.len() - p)
    }

    /// Whether `s` contains any special token's text.
    pub fn any_in(&self, s: &str) -> bool {
        self.tokens.iter().any(|t| s.contains(t.as_str()))
    }
}

static ID_SEQ: std::sync::LazyLock<AtomicU64> = std::sync::LazyLock::new(|| {
    AtomicU64::new(rand::random::<u64>())
});

fn random_alnum(n: usize) -> String {
    const A: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut x = ID_SEQ.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed) ^ rand::random::<u64>();
    (0..n)
        .map(|_| {
            // splitmix64 step per character
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            A[((z ^ (z >> 31)) % A.len() as u64) as usize] as char
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_text_strips_only_whole_tokens() {
        let s = SpecialText::new(vec!["<|tool_call>".into(), "<|\"|>".into(), "<|turn>".into()]);
        assert_eq!(s.strip("a<|\"|>b<|tool_call>c<|tool"), "abc<|tool");
        assert_eq!(s.strip("plain"), "plain");
        assert_eq!(s.strip("é<|turn>ü"), "éü");
    }

    #[test]
    fn ids_have_the_shape_their_format_requires() {
        let m = ToolFormat::Mistral.call_id();
        assert_eq!(m.len(), 9);
        assert!(m.chars().all(|c| c.is_ascii_alphanumeric()));
        let h = ToolFormat::Hermes.call_id();
        assert!(h.starts_with("call_") && h.len() == 29);
        assert_ne!(ToolFormat::Hermes.call_id(), h);
    }
}

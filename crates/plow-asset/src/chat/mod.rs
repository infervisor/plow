//! What a chat template teaches its model, read from the template itself and never from a model
//! name: the tool-call syntax ([`ToolFormat`]), whether `tools` render at all ([`ToolSupport`]),
//! the reasoning-trace markers ([`reasoning_markers`]) and, for checkpoints that ship no template,
//! which built-in prompt builder their vocabulary names ([`builtin_for`]). `plowc` records these
//! in the packet's `serve.json`; `plowrt` parses answers with them.

pub mod pyjson;
pub mod template;

pub use template::{ChatTemplate, RenderOpts};

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
            (ToolFormat::Hermes, true) => {
                format!("<tool_call>\n{{\"name\": \"{n}\", \"arguments\": ")
            }
            (ToolFormat::Qwen3Xml, false) => "<tool_call>\n<function=".into(),
            (ToolFormat::Qwen3Xml, true) => format!("<tool_call>\n<function={n}>\n"),
            (ToolFormat::Glm45, _) => format!("<tool_call>{n}"),
            (ToolFormat::Llama3Json, false) => "{\"name\": \"".into(),
            (ToolFormat::Llama3Json, true) => format!("{{\"name\": \"{n}\", \"parameters\": "),
            (ToolFormat::Mistral, false) => "[TOOL_CALLS] [{\"name\": \"".into(),
            (ToolFormat::Mistral, true) => {
                format!("[TOOL_CALLS] [{{\"name\": \"{n}\", \"arguments\": ")
            }
            (ToolFormat::KimiK2, false) => {
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.".into()
            }
            (ToolFormat::KimiK2, true) => {
                format!("<|tool_calls_section_begin|><|tool_call_begin|>functions.{n}:{call_index}<|tool_call_argument_begin|>")
            }
            (ToolFormat::Harmony, false) => "<|channel|>commentary to=functions.".into(),
            (ToolFormat::Harmony, true) => {
                format!("<|channel|>commentary to=functions.{n} <|constrain|>json<|message|>")
            }
            (ToolFormat::DeepSeekV3, false) => {
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>".into()
            }
            (ToolFormat::DeepSeekV3, true) => format!(
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{n}\n```json\n"
            ),
            (ToolFormat::DeepSeekV31, false) => {
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>".into()
            }
            (ToolFormat::DeepSeekV31, true) => {
                format!("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{n}<｜tool▁sep｜>")
            }
        }
    }

    /// The `id` a parsed call gets on the wire.
    /// The format with this [`Self::name`].
    pub fn from_name(name: &str) -> Option<Self> {
        ALL.iter().copied().find(|f| f.name() == name)
    }
}

const ALL: [ToolFormat; 10] = [
    ToolFormat::Gemma4,
    ToolFormat::Hermes,
    ToolFormat::Qwen3Xml,
    ToolFormat::Glm45,
    ToolFormat::Llama3Json,
    ToolFormat::Mistral,
    ToolFormat::KimiK2,
    ToolFormat::Harmony,
    ToolFormat::DeepSeekV3,
    ToolFormat::DeepSeekV31,
];

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
    pub fn probe(t: &ChatTemplate, text: &str) -> Self {
        let tools = serde_json::json!([{"type": "function", "function": {
            "name": PROBE_NAME, "description": "probe",
            "parameters": {"type": "object", "properties": {"x": {"type": "string", "description": "x"}}, "required": ["x"]}
        }}]);
        let opts = RenderOpts {
            tools: Some(tools),
            ..Default::default()
        };
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

/// The built-in prompt builders `plowrt` has for checkpoints that ship no chat template, in
/// probe order.
pub const BUILTIN_IDS: [&str; 5] = ["k3", "harmony", "glm", "llama3", "gemma"];

/// Which built-in builder a vocabulary names: `one(marker)` is whether the marker is ONE token
/// of the model's own tokenizer (it is in that checkpoint's added tokens, and several ordinary
/// pieces in anyone else's). Probed in this order because K3's `<|end_of_msg|>` is also its eos
/// and the least ambiguous thing in the vocab; `None` when no family's markers are present.
pub fn builtin_for(one: impl Fn(&str) -> bool) -> Option<&'static str> {
    Some(if one("<|end_of_msg|>") {
        "k3"
    } else if one("<|start|>") && one("<|channel|>") {
        "harmony"
    } else if one("<|assistant|>") {
        "glm"
    } else if one("<|start_header_id|>") && one("<|eot_id|>") {
        "llama3"
    } else if one("<|turn>") {
        "gemma"
    } else {
        return None;
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_format_round_trips_through_its_name() {
        for f in ALL {
            assert_eq!(ToolFormat::from_name(f.name()), Some(f));
        }
        assert_eq!(ToolFormat::from_name("none"), None);
    }

    #[test]
    fn builtin_probe_order_is_the_runtime_one() {
        let has = |set: &[&str]| {
            let set: Vec<String> = set.iter().map(|s| s.to_string()).collect();
            move |m: &str| set.iter().any(|s| s == m)
        };
        assert_eq!(
            builtin_for(has(&["<|end_of_msg|>", "<|assistant|>"])),
            Some("k3")
        );
        assert_eq!(
            builtin_for(has(&["<|start|>", "<|channel|>"])),
            Some("harmony")
        );
        assert_eq!(builtin_for(has(&["<|start|>"])), None);
        assert_eq!(builtin_for(has(&["<|assistant|>"])), Some("glm"));
        assert_eq!(builtin_for(has(&["<|start_header_id|>"])), None);
        assert_eq!(
            builtin_for(has(&["<|start_header_id|>", "<|eot_id|>"])),
            Some("llama3")
        );
        assert_eq!(builtin_for(has(&["<|turn>"])), Some("gemma"));
        assert_eq!(builtin_for(has(&[])), None);
    }
}

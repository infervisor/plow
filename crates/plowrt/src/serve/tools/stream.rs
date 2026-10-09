//! One tool-calling generation, token by token: reasoning split, then the format's call parser,
//! then special-token stripping. The buffered and streamed responses both drive this type, so
//! they cannot disagree about a generation.

use std::sync::Arc;

use super::parse::{self, Ev, Parser};
use super::request::ParseSpec;
use super::{SpecialText, ToolFormat};
use crate::serve::openai::{FunctionCall, ToolCall};
use crate::serve::reasoning::{ReasoningMode, ReasoningSplit};

/// What one pushed piece of text produced.
#[derive(Debug, Default)]
pub struct Step {
    pub reasoning: Option<String>,
    pub content: Option<String>,
    /// New complete calls with their response index.
    pub calls: Vec<(u32, ToolCall)>,
}

pub struct ToolStream {
    pub split: ReasoningSplit,
    parser: Box<dyn Parser>,
    strip: Arc<SpecialText>,
    format: ToolFormat,
    parallel: bool,
    n_calls: u32,
    /// Whitespace-only answer text, released once real text follows. A turn that is only calls
    /// answers `content: null`, not the newlines around its call markers.
    ws: String,
}

impl ToolStream {
    pub fn new(spec: &ParseSpec, mode: ReasoningMode, prompt_opens: bool, strip: Arc<SpecialText>) -> Self {
        ToolStream {
            split: ReasoningSplit::new(mode, prompt_opens),
            parser: parse::parser(spec),
            strip,
            format: spec.format,
            parallel: spec.parallel,
            n_calls: 0,
            ws: String::new(),
        }
    }

    pub fn calls(&self) -> u32 {
        self.n_calls
    }

    pub fn push(&mut self, text: &str) -> Step {
        let (r, c) = self.split.push(text);
        let mut evs = Vec::new();
        if let Some(r) = r {
            evs.push(Ev::Reasoning(r));
        }
        if let Some(c) = c {
            self.parser.push(&c, &mut evs);
        }
        self.apply(evs)
    }

    pub fn finish(&mut self) -> Step {
        let (r, c) = self.split.finish();
        let mut evs = Vec::new();
        if let Some(r) = r {
            evs.push(Ev::Reasoning(r));
        }
        if let Some(c) = c {
            self.parser.push(&c, &mut evs);
        }
        self.parser.finish(&mut evs);
        let mut step = self.apply(evs);
        if self.n_calls == 0 && !self.ws.is_empty() {
            step.content.get_or_insert_with(String::new).push_str(&std::mem::take(&mut self.ws));
        }
        step
    }

    fn apply(&mut self, evs: Vec<Ev>) -> Step {
        let mut step = Step::default();
        for ev in evs {
            match ev {
                Ev::Reasoning(r) => {
                    let r = self.strip.strip(&r);
                    if !r.is_empty() {
                        step.reasoning.get_or_insert_with(String::new).push_str(&r);
                    }
                }
                Ev::Text(t) => {
                    // Trailing whitespace is held however the text was chunked, so the
                    // buffered and streamed answers agree.
                    let t = self.strip.strip(&t);
                    let body = t.trim_end();
                    if body.is_empty() {
                        self.ws.push_str(&t);
                    } else {
                        let c = step.content.get_or_insert_with(String::new);
                        c.push_str(&std::mem::take(&mut self.ws));
                        c.push_str(body);
                        self.ws.push_str(&t[body.len()..]);
                    }
                }
                Ev::Call(call) => {
                    if !self.parallel && self.n_calls > 0 {
                        continue;
                    }
                    self.ws.clear();
                    let id = call.id.unwrap_or_else(|| self.format.call_id());
                    step.calls.push((
                        self.n_calls,
                        ToolCall {
                            id,
                            kind: "function",
                            function: FunctionCall { name: call.name, arguments: call.args.to_string() },
                        },
                    ));
                    self.n_calls += 1;
                }
            }
        }
        step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn spec(format: ToolFormat) -> ParseSpec {
        let tools = vec![
            ("get_weather".to_string(), json!({"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}})),
            ("search".to_string(), json!({"type": "object", "properties": {"q": {"type": "string"}, "filters": {"type": "object"}, "n": {"type": "number"}}})),
        ];
        ParseSpec { format, tools: Arc::new(tools), parallel: true }
    }

    fn specials() -> Arc<SpecialText> {
        Arc::new(SpecialText::new(
            [
                "<|tool_call>", "<tool_call|>", "<|\"|>", "<|tool_response>", "<|channel>", "<channel|>",
                "<|python_tag|>", "[TOOL_CALLS]", "[ARGS]", "<|tool_calls_section_begin|>", "<|tool_calls_section_end|>",
                "<|tool_call_begin|>", "<|tool_call_argument_begin|>", "<|tool_call_end|>", "<|channel|>", "<|message|>",
                "<|end|>", "<|start|>", "<|constrain|>", "<|observation|>",
            ]
            .map(String::from)
            .to_vec(),
        ))
    }

    #[derive(Debug, PartialEq)]
    struct Got {
        reasoning: String,
        content: Option<String>,
        calls: Vec<(String, Value)>,
    }

    /// Feed `pieces` and collect; also asserts that indices are dense and in order.
    fn run_with(format: ToolFormat, mode: ReasoningMode, opens: bool, pieces: &[&str]) -> Got {
        let mut s = ToolStream::new(&spec(format), mode, opens, specials());
        let (mut r, mut c, mut calls) = (String::new(), None::<String>, Vec::new());
        let mut take = |st: Step| {
            r.push_str(&st.reasoning.unwrap_or_default());
            if let Some(x) = st.content {
                c.get_or_insert_with(String::new).push_str(&x);
            }
            for (i, call) in st.calls {
                assert_eq!(i as usize, calls.len());
                calls.push((call.function.name, serde_json::from_str::<Value>(&call.function.arguments).unwrap()));
            }
        };
        for p in pieces {
            take(s.push(p));
        }
        take(s.finish());
        // Both response paths trim the trace's outer whitespace.
        Got { reasoning: r.trim().to_string(), content: c, calls }
    }

    fn run(format: ToolFormat, text: &str) -> Got {
        run_with(format, ReasoningMode::None, false, &[text])
    }

    /// Whole-string, every 1-byte split and every 3-byte split agree.
    fn check(format: ToolFormat, mode: ReasoningMode, opens: bool, text: &str, want: &Got) {
        assert_eq!(&run_with(format, mode, opens, &[text]), want, "whole {text:?}");
        for k in [1usize, 3, 7] {
            let mut pieces = Vec::new();
            let mut i = 0;
            while i < text.len() {
                let mut j = (i + k).min(text.len());
                while !text.is_char_boundary(j) {
                    j += 1;
                }
                pieces.push(&text[i..j]);
                i = j;
            }
            assert_eq!(&run_with(format, mode, opens, &pieces), want, "chunks of {k}: {text:?}");
        }
    }

    fn got(content: Option<&str>, calls: &[(&str, Value)]) -> Got {
        Got {
            reasoning: String::new(),
            content: content.map(String::from),
            calls: calls.iter().map(|(n, a)| (n.to_string(), a.clone())).collect(),
        }
    }

    #[test]
    fn gemma4_calls_with_quoted_nested_and_parallel_args() {
        let text = "<|tool_call>call:get_weather{city:<|\"|>Paris, \"FR\"<|\"|>,days:3,metric:true}<tool_call|>\
                    <|tool_call>call:search{filters:{airlines:[<|\"|>LX<|\"|>,<|\"|>UA<|\"|>],max:450.5,x:null},q:<|\"|>a{b}:c<|\"|>}<tool_call|>";
        check(ToolFormat::Gemma4, ReasoningMode::None, false, text, &got(None, &[
            ("get_weather", json!({"city": "Paris, \"FR\"", "days": 3, "metric": true})),
            ("search", json!({"filters": {"airlines": ["LX", "UA"], "max": 450.5, "x": null}, "q": "a{b}:c"})),
        ]));
        // text before the call is content; markers never leak
        check(ToolFormat::Gemma4, ReasoningMode::None, false, "Let me check.<|tool_call>call:get_weather{}<tool_call|>",
            &got(Some("Let me check."), &[("get_weather", json!({}))]));
    }

    #[test]
    fn invalid_or_cut_calls_fall_back_to_stripped_text() {
        // cut by max_tokens inside the call
        let g = run(ToolFormat::Gemma4, "ok <|tool_call>call:get_weather{city:<|\"|>Par");
        assert_eq!(g, got(Some("ok call:get_weather{city:Par"), &[]));
        // malformed body
        let g = run(ToolFormat::Hermes, "<tool_call>\n{\"name\": \"f\", \"arguments\": {bad}}\n</tool_call>");
        assert!(g.calls.is_empty() && g.content.unwrap().contains("\"name\": \"f\""));
        // a plain answer passes through untouched
        assert_eq!(run(ToolFormat::Hermes, "The answer is <b>4</b>."), got(Some("The answer is <b>4</b>."), &[]));
        assert_eq!(run(ToolFormat::Gemma4, "a < b <| c"), got(Some("a < b <| c"), &[]));
    }

    #[test]
    fn hermes_and_qwen3_thinking() {
        let text = "<think>\nNeed weather.\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Zürich\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"search\", \"arguments\": \"{\\\"q\\\": \\\"x\\\"}\"}\n</tool_call>";
        let want = Got {
            reasoning: "Need weather.".into(),
            content: None,
            calls: vec![("get_weather".into(), json!({"city": "Zürich"})), ("search".into(), json!({"q": "x"}))],
        };
        check(ToolFormat::Hermes, ReasoningMode::ThinkTag, false, text, &want);
    }

    #[test]
    fn qwen3_xml_types_by_schema() {
        let text = "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n2\n</parameter>\n<parameter=metric>\nfalse\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=search>\n<parameter=q>\n2\n</parameter>\n<parameter=filters>\n{\"a\": [1]}\n</parameter>\n<parameter=extra>\n{x}\n</parameter>\n</function>\n</tool_call>";
        check(ToolFormat::Qwen3Xml, ReasoningMode::ThinkTag, true, &format!("plan</think>\n\n{text}"), &Got {
            reasoning: "plan".into(),
            content: Some("Checking.".into()),
            calls: vec![
                ("get_weather".into(), json!({"city": "New York", "days": 2, "metric": false})),
                ("search".into(), json!({"q": "2", "filters": {"a": [1]}, "extra": "{x}"})),
            ],
        });
    }

    #[test]
    fn glm45_and_glm5_layouts() {
        let want = got(None, &[("get_weather", json!({"city": "Paris", "days": 3})), ("search", json!({"q": "18", "filters": {"k": "v"}, "u": [1, 2]}))]);
        let v5 = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>\
                  <tool_call>search<arg_key>q</arg_key><arg_value>18</arg_value><arg_key>filters</arg_key><arg_value>{\"k\": \"v\"}</arg_value><arg_key>u</arg_key><arg_value>[1, 2]</arg_value></tool_call>";
        check(ToolFormat::Glm45, ReasoningMode::ThinkTag, true, &format!("</think>{v5}"), &want);
        let v45 = "\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>\
                   \n<tool_call>search\n<arg_key>q</arg_key>\n<arg_value>18</arg_value>\n<arg_key>filters</arg_key>\n<arg_value>{\"k\": \"v\"}</arg_value>\n<arg_key>u</arg_key>\n<arg_value>[1, 2]</arg_value>\n</tool_call>";
        check(ToolFormat::Glm45, ReasoningMode::ThinkTag, false, v45, &want);
    }

    #[test]
    fn llama3_json_and_python_tag() {
        let want = got(None, &[("get_weather", json!({"city": "Paris"})), ("search", json!({"q": "x"}))]);
        check(ToolFormat::Llama3Json, ReasoningMode::None, false,
            "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}; {\"name\": \"search\", \"parameters\": {\"q\": \"x\"}}", &want);
        check(ToolFormat::Llama3Json, ReasoningMode::None, false,
            "<|python_tag|>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}; {\"name\": \"search\", \"parameters\": {\"q\": \"x\"}}", &want);
        // JSON that is not a call is an answer
        assert_eq!(run(ToolFormat::Llama3Json, "{\"answer\": 4}"), got(Some("{\"answer\": 4}"), &[]));
        assert_eq!(run(ToolFormat::Llama3Json, "Paris is sunny."), got(Some("Paris is sunny."), &[]));
    }

    #[test]
    fn mistral_v3_and_v11() {
        let want = got(None, &[("get_weather", json!({"city": "Paris"})), ("search", json!({"q": "x"}))]);
        check(ToolFormat::Mistral, ReasoningMode::None, false,
            "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, {\"name\": \"search\", \"arguments\": {\"q\": \"x\"}}]", &want);
        check(ToolFormat::Mistral, ReasoningMode::None, false,
            "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}[TOOL_CALLS]search[ARGS]{\"q\": \"x\"}", &want);
    }

    #[test]
    fn kimi_k2_keeps_its_ids() {
        let text = "Sure.<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_call_begin|>functions.search:1<|tool_call_argument_begin|>{\"q\": \"x\"}<|tool_call_end|><|tool_calls_section_end|>";
        check(ToolFormat::KimiK2, ReasoningMode::None, false, text,
            &got(Some("Sure."), &[("get_weather", json!({"city": "Paris"})), ("search", json!({"q": "x"}))]));
        let mut s = ToolStream::new(&spec(ToolFormat::KimiK2), ReasoningMode::None, false, specials());
        let a = s.push(text);
        let ids: Vec<String> = a.calls.into_iter().chain(s.finish().calls).map(|(_, c)| c.id).collect();
        assert_eq!(ids, ["functions.get_weather:0", "functions.search:1"]);
    }

    #[test]
    fn harmony_channels() {
        let text = "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}";
        check(ToolFormat::Harmony, ReasoningMode::None, false, text, &Got {
            reasoning: "Need the weather.".into(),
            content: None,
            calls: vec![("get_weather".into(), json!({"city": "Paris"}))],
        });
        check(ToolFormat::Harmony, ReasoningMode::None, false,
            "<|channel|>analysis<|message|>Easy.<|end|><|start|>assistant<|channel|>final<|message|>It is 4.",
            &Got { reasoning: "Easy.".into(), content: Some("It is 4.".into()), calls: vec![] });
    }

    #[test]
    fn parallel_tool_calls_false_keeps_the_first() {
        let mut sp = spec(ToolFormat::Hermes);
        sp.parallel = false;
        let mut s = ToolStream::new(&sp, ReasoningMode::None, false, specials());
        let a = s.push("<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call><tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>");
        let b = s.finish();
        let names: Vec<String> = a.calls.into_iter().chain(b.calls).map(|(_, c)| c.function.name).collect();
        assert_eq!(names, ["a"]);
        assert_eq!(s.calls(), 1);
    }

    #[test]
    fn whitespace_only_answers_survive_when_there_is_no_call() {
        assert_eq!(run(ToolFormat::Hermes, "\n\n"), got(Some("\n\n"), &[]));
    }
}

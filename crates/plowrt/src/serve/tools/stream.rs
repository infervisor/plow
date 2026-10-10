//! One chat generation's response text, token by token: reasoning split, then the format's call
//! parser, then special-token stripping. The buffered and streamed responses both drive this
//! type, so they cannot disagree about a generation.
//!
//! Calls stream as OpenAI sends them: the first delta of a call carries `index`, `id`, `type` and
//! the name, later ones `arguments` fragments as the model writes them. A `strict` tool's
//! arguments are withheld from the stream until the call ends and they validate (one delta);
//! there is no constrained decoding to keep them valid while they are written.
//!
//! The steady state allocates nothing: every buffer is reused across tokens, and a [`Step`]
//! borrows them.

use std::ops::Range;
use std::sync::Arc;

use serde::ser::{Serialize, SerializeSeq, Serializer};

use super::parse::{self, Ev, Out, Parser};
use super::request::{Force, ParseSpec};
use super::SpecialText;
use crate::serve::openai::{FunctionCall, ToolCall};
use crate::serve::reasoning::{ReasoningMode, ReasoningSplit};

struct Call {
    id: String,
    name: String,
    args: String,
    /// Validated against the schema at its end.
    strict: bool,
    /// Arguments withheld from the stream until validated.
    hold: bool,
}

struct DeltaRec {
    call: usize,
    head: bool,
    args: Range<usize>,
}

pub struct ToolStream {
    pub split: ReasoningSplit,
    parser: Option<Box<dyn Parser>>,
    spec: Option<ParseSpec>,
    strip: Arc<SpecialText>,
    out: Out,
    rbuf: String,
    cbuf: String,
    sbuf: String,
    /// Tails that may begin a special token, carried into the next piece.
    chold: String,
    rhold: String,
    content: String,
    has_content: bool,
    reasoning: String,
    deltas: Vec<DeltaRec>,
    dargs: String,
    /// Keep the step's output for the next one (a primed opener rides the first token's frame).
    carry: bool,
    calls: Vec<Call>,
    open: Option<usize>,
    /// Whitespace-only answer text, released once real text follows. A turn that is only calls
    /// answers `content: null`, not the newlines around its call markers.
    ws: String,
    error: Option<String>,
    hold_strict: bool,
}

/// What one pushed piece of text produced, borrowed from the stream.
pub struct Step<'a> {
    pub reasoning: Option<&'a str>,
    pub content: Option<&'a str>,
    pub tool_calls: Option<Deltas<'a>>,
}

/// A step's `delta.tool_calls`, serialized in place.
#[derive(Clone, Copy)]
pub struct Deltas<'a> {
    ts: &'a ToolStream,
}

#[derive(serde::Serialize)]
struct ToolDelta<'a> {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    function: FnDelta<'a>,
}

#[derive(serde::Serialize)]
struct FnDelta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
    arguments: &'a str,
}

impl Serialize for Deltas<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let ts = self.ts;
        let mut seq = s.serialize_seq(Some(ts.deltas.len()))?;
        for d in &ts.deltas {
            let c = &ts.calls[d.call];
            seq.serialize_element(&ToolDelta {
                index: d.call,
                id: d.head.then_some(c.id.as_str()),
                kind: d.head.then_some("function"),
                function: FnDelta { name: d.head.then_some(c.name.as_str()), arguments: &ts.dargs[d.args.clone()] },
            })?;
        }
        seq.end()
    }
}

#[derive(serde::Serialize)]
struct ChoiceFrame<'a> {
    index: u32,
    delta: DeltaFrame<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<&'a crate::serve::logprobs::ChatLogprobs>,
    finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    x_plow_finish_reason: Option<&'static str>,
}

#[derive(serde::Serialize)]
struct DeltaFrame<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Deltas<'a>>,
}

impl Step<'_> {
    /// Whether the step carries anything for the client.
    pub fn is_empty(&self, reasoning: bool) -> bool {
        self.content.is_none() && self.tool_calls.is_none() && (!reasoning || self.reasoning.is_none())
    }

    /// The step as one `chat.completion.chunk` SSE event, byte for byte what
    /// [`crate::serve::openai::ChunkChoice`] serializes to. `reasoning: false` leaves the trace out.
    pub fn frame(
        &self,
        head: &crate::serve::stream::FrameHead,
        role: Option<&'static str>,
        logprobs: Option<&crate::serve::logprobs::ChatLogprobs>,
        reasoning: bool,
        finish: Option<(&'static str, Option<&'static str>)>,
    ) -> axum::body::Bytes {
        head.frame(&ChoiceFrame {
            index: 0,
            delta: DeltaFrame {
                role,
                content: self.content,
                reasoning_content: self.reasoning.filter(|_| reasoning),
                tool_calls: self.tool_calls,
            },
            logprobs,
            finish_reason: finish.map(|f| f.0),
            x_plow_finish_reason: finish.and_then(|f| f.1),
        })
    }
}

impl Deltas<'_> {
    /// `(index, id, name, arguments)` per delta, heads with their id and name.
    pub fn iter(&self) -> impl Iterator<Item = (usize, Option<&str>, Option<&str>, &str)> {
        self.ts.deltas.iter().map(|d| {
            let c = &self.ts.calls[d.call];
            (d.call, d.head.then_some(c.id.as_str()), d.head.then_some(c.name.as_str()), &self.ts.dargs[d.args.clone()])
        })
    }
}

impl ToolStream {
    /// `spec`: the request's tools (`None`: no tools, only reasoning split and special-token
    /// stripping). `stream`: the turn is streamed, so `strict` arguments are withheld until
    /// validated.
    pub fn new(spec: Option<&ParseSpec>, mode: ReasoningMode, prompt_opens: bool, strip: Arc<SpecialText>, stream: bool) -> Self {
        ToolStream {
            split: ReasoningSplit::new(mode, prompt_opens),
            parser: spec.map(parse::parser),
            spec: spec.cloned(),
            strip,
            out: Out::default(),
            rbuf: String::new(),
            cbuf: String::new(),
            sbuf: String::new(),
            chold: String::new(),
            rhold: String::new(),
            content: String::new(),
            has_content: false,
            reasoning: String::new(),
            deltas: Vec::new(),
            dargs: String::new(),
            carry: false,
            calls: Vec::new(),
            open: None,
            ws: String::new(),
            error: None,
            hold_strict: stream,
        }
    }

    /// Feed text the prompt already wrote for the model (a forced call opener); what it produces
    /// rides the next step.
    pub fn prime(&mut self, text: &str) {
        self.step(Some(text), false);
        self.carry = true;
    }

    /// Calls in the response so far.
    pub fn calls(&self) -> usize {
        self.calls.len()
    }

    /// Why the turn failed validation (`strict` arguments, a forced call missing or malformed).
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn take_calls(&mut self) -> Vec<ToolCall> {
        self.calls
            .drain(..)
            .map(|c| ToolCall { id: c.id, kind: "function", function: FunctionCall { name: c.name, arguments: c.args } })
            .collect()
    }

    pub fn push(&mut self, text: &str) -> Step<'_> {
        self.step(Some(text), false);
        self.view()
    }

    /// End of the generation. `cut`: it ended on `max_tokens` (or a preemption), so a call left
    /// open is returned as generated and nothing is validated.
    pub fn finish(&mut self, cut: bool) -> Step<'_> {
        self.step(None, cut);
        if let Some(i) = self.open.take() {
            self.end_call(i, cut);
        }
        if let Some(spec) = &self.spec {
            if !cut && spec.force != Force::Auto && self.calls.is_empty() && self.error.is_none() {
                self.error = Some("`tool_choice` requires a call, and the model's output did not parse as one".into());
            }
        }
        if self.calls.is_empty() && !self.ws.is_empty() {
            self.content.push_str(&self.ws);
            self.ws.clear();
            self.has_content = true;
        }
        self.view()
    }

    fn view(&self) -> Step<'_> {
        Step {
            reasoning: (!self.reasoning.is_empty()).then_some(self.reasoning.as_str()),
            content: self.has_content.then_some(self.content.as_str()),
            tool_calls: (!self.deltas.is_empty()).then_some(Deltas { ts: self }),
        }
    }

    /// `text: None` is the end of the generation.
    fn step(&mut self, text: Option<&str>, _cut: bool) {
        if !self.carry {
            self.content.clear();
            self.has_content = false;
            self.reasoning.clear();
            self.deltas.clear();
            self.dargs.clear();
        }
        self.carry = false;
        self.rbuf.clear();
        self.cbuf.clear();
        match text {
            Some(t) => self.split.push_into(t, &mut self.rbuf, &mut self.cbuf),
            None => self.split.finish_into(&mut self.rbuf, &mut self.cbuf),
        };
        let mut out = std::mem::take(&mut self.out);
        out.clear();
        match self.parser.as_mut() {
            Some(p) => {
                if !self.cbuf.is_empty() {
                    p.push(&self.cbuf, &mut out);
                }
                if text.is_none() {
                    p.finish(&mut out);
                }
            }
            None => out.put_text(&self.cbuf),
        }
        let last = text.is_none();
        for r in [&self.rbuf, &out.reasoning] {
            if !r.is_empty() {
                self.strip.strip_stream(&mut self.rhold, r, &mut self.reasoning, false);
            }
        }
        if last && !self.rhold.is_empty() {
            self.strip.strip_stream(&mut self.rhold, "", &mut self.reasoning, true);
        }
        for ev in out.evs.drain(..) {
            match ev {
                Ev::Text(r) => self.text(&out.text[r], false),
                Ev::Start { name, id } => self.start(name, id),
                Ev::Args(r) => {
                    if let Some(i) = self.open {
                        self.args(i, &out.args[r]);
                    }
                }
                Ev::End => {
                    if let Some(i) = self.open.take() {
                        self.end_call(i, false);
                    }
                }
            }
        }
        if last && !self.chold.is_empty() {
            self.text("", true);
        }
        self.out = out;
    }

    /// Answer text. Trailing whitespace is held however the text was chunked, so the buffered
    /// and streamed answers agree.
    fn text(&mut self, t: &str, last: bool) {
        self.sbuf.clear();
        self.strip.strip_stream(&mut self.chold, t, &mut self.sbuf, last);
        let body = self.sbuf.trim_end();
        if body.is_empty() {
            self.ws.push_str(&self.sbuf);
            return;
        }
        self.content.push_str(&self.ws);
        self.ws.clear();
        self.content.push_str(body);
        self.has_content = true;
        let tail = body.len();
        self.ws.push_str(&self.sbuf[tail..]);
    }

    fn start(&mut self, name: String, id: Option<String>) {
        if let Some(i) = self.open.take() {
            self.end_call(i, false);
        }
        let Some(spec) = &self.spec else { return };
        if !spec.parallel && !self.calls.is_empty() {
            return;
        }
        self.ws.clear();
        let strict = spec.tool(&name).is_some_and(|t| t.strict);
        let id = id.unwrap_or_else(|| spec.format.call_id());
        self.calls.push(Call { id, name, args: String::new(), strict, hold: strict && self.hold_strict });
        let i = self.calls.len() - 1;
        self.open = Some(i);
        let at = self.dargs.len();
        self.deltas.push(DeltaRec { call: i, head: true, args: at..at });
    }

    fn args(&mut self, i: usize, a: &str) {
        self.calls[i].args.push_str(a);
        if !self.calls[i].hold {
            self.delta_args(i, a);
        }
    }

    fn delta_args(&mut self, i: usize, a: &str) {
        let at = self.dargs.len();
        self.dargs.push_str(a);
        if let Some(d) = self.deltas.last_mut() {
            if d.call == i && d.args.end == at {
                d.args.end = self.dargs.len();
                return;
            }
        }
        self.deltas.push(DeltaRec { call: i, head: false, args: at..self.dargs.len() });
    }

    fn end_call(&mut self, i: usize, cut: bool) {
        if self.calls[i].args.trim().is_empty() {
            self.calls[i].args.clear();
            self.args(i, "{}");
        }
        if !cut && self.error.is_none() {
            if let Err(e) = self.validate(i) {
                self.error = Some(e);
                return;
            }
        }
        if self.calls[i].hold {
            let args = std::mem::take(&mut self.calls[i].args);
            self.delta_args(i, &args);
            self.calls[i].args = args;
        }
    }

    /// Forced and `strict` calls only: a declared name, a JSON object, and for `strict` the schema.
    fn validate(&self, i: usize) -> Result<(), String> {
        let Some(spec) = &self.spec else { return Ok(()) };
        let c = &self.calls[i];
        if !c.strict && spec.force == Force::Auto {
            return Ok(());
        }
        let Some(tool) = spec.tool(&c.name) else {
            return Err(format!("the model called `{}`, which is not in `tools`", c.name));
        };
        if let Force::Named(n) = &spec.force {
            if *n != c.name {
                return Err(format!("`tool_choice` names `{n}`, but the model called `{}`", c.name));
            }
        }
        let v: serde_json::Value = serde_json::from_str(&c.args)
            .map_err(|e| format!("the arguments of `{}` are not valid JSON ({e})", c.name))?;
        if !v.is_object() {
            return Err(format!("the arguments of `{}` are not a JSON object", c.name));
        }
        if c.strict && !tool.params.is_null() {
            super::schema::validate(&tool.params, &v)
                .map_err(|e| format!("the arguments of `{}` do not match its `parameters` schema: {e}", c.name))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::tools::request::ToolDef;
    use crate::serve::tools::ToolFormat;
    use serde_json::{json, Value};

    fn spec(format: ToolFormat) -> ParseSpec {
        let tool = |name: &str, params: Value| ToolDef { name: name.into(), params, strict: false };
        let tools = vec![
            tool("get_weather", json!({"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}})),
            tool("search", json!({"type": "object", "properties": {"q": {"type": "string"}, "filters": {"type": "object"}, "n": {"type": "number"}}})),
        ];
        ParseSpec { format, tools: Arc::new(tools), parallel: true, force: Force::Auto, history_calls: 0 }
    }

    fn specials() -> Arc<SpecialText> {
        Arc::new(SpecialText::new(
            [
                "<|tool_call>", "<tool_call|>", "<|\"|>", "<|tool_response>", "<|channel>", "<channel|>",
                "<|python_tag|>", "[TOOL_CALLS]", "[ARGS]", "<|tool_calls_section_begin|>", "<|tool_calls_section_end|>",
                "<|tool_call_begin|>", "<|tool_call_argument_begin|>", "<|tool_call_end|>", "<|channel|>", "<|message|>",
                "<|end|>", "<|start|>", "<|constrain|>", "<|observation|>", "<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>",
                "<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>", "<｜tool▁sep｜>",
            ]
            .map(String::from)
            .to_vec(),
        ))
    }

    #[derive(Debug, PartialEq)]
    struct Got {
        reasoning: String,
        content: Option<String>,
        /// Name and arguments, parsed when they are JSON and the raw string otherwise.
        calls: Vec<(String, Value)>,
    }

    fn args_value(a: &str) -> Value {
        serde_json::from_str(a).unwrap_or_else(|_| Value::String(a.to_string()))
    }

    /// Feed `pieces` and collect. Checks the streamed deltas: a head with id and name opens each
    /// call, indices are dense, and the argument fragments add up to the buffered arguments.
    fn run_spec(sp: &ParseSpec, mode: ReasoningMode, opens: bool, pieces: &[&str], prime: Option<&str>) -> (Got, Option<String>) {
        let mut s = ToolStream::new(Some(sp), mode, opens, specials(), false);
        if let Some(p) = prime {
            s.prime(p);
        }
        let (mut r, mut c) = (String::new(), None::<String>);
        let mut streamed: Vec<(String, String, String)> = Vec::new();
        let mut take = |st: Step| {
            r.push_str(st.reasoning.unwrap_or_default());
            if let Some(x) = st.content {
                c.get_or_insert_with(String::new).push_str(x);
            }
            if let Some(d) = st.tool_calls {
                for (i, id, name, args) in d.iter() {
                    if let (Some(id), Some(name)) = (id, name) {
                        assert_eq!(i, streamed.len(), "a head opens each call, in order");
                        streamed.push((id.to_string(), name.to_string(), String::new()));
                    }
                    streamed[i].2.push_str(args);
                }
            }
        };
        for p in pieces {
            take(s.push(p));
        }
        take(s.finish(false));
        let err = s.error().map(String::from);
        let calls = s.take_calls();
        assert_eq!(calls.len(), streamed.len());
        for (c, (id, name, args)) in calls.iter().zip(&streamed) {
            assert_eq!((&c.id, &c.function.name, &c.function.arguments), (id, name, args), "streamed vs buffered");
        }
        let calls = calls.into_iter().map(|c| (c.function.name, args_value(&c.function.arguments))).collect();
        // Both response paths trim the trace's outer whitespace.
        (Got { reasoning: r.trim().to_string(), content: c, calls }, err)
    }

    fn run_with(format: ToolFormat, mode: ReasoningMode, opens: bool, pieces: &[&str]) -> Got {
        run_spec(&spec(format), mode, opens, pieces, None).0
    }

    fn run(format: ToolFormat, text: &str) -> Got {
        run_with(format, ReasoningMode::None, false, &[text])
    }

    fn chunks(text: &str, k: usize) -> Vec<&str> {
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
        pieces
    }

    /// Whole-string, and 1-, 3- and 7-byte splits agree.
    fn check(format: ToolFormat, mode: ReasoningMode, opens: bool, text: &str, want: &Got) {
        assert_eq!(&run_with(format, mode, opens, &[text]), want, "whole {text:?}");
        for k in [1usize, 3, 7] {
            assert_eq!(&run_with(format, mode, opens, &chunks(text, k)), want, "chunks of {k}: {text:?}");
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
                    <|tool_call>call:search{filters:{airlines:[<|\"|>LX<|\"|>,<|\"|>UA<|\"|>],max:450.5,x:null},q:<|\"|>a{b}:c<|\"|>,e:[],o:{}}<tool_call|>";
        check(ToolFormat::Gemma4, ReasoningMode::None, false, text, &got(None, &[
            ("get_weather", json!({"city": "Paris, \"FR\"", "days": 3, "metric": true})),
            ("search", json!({"filters": {"airlines": ["LX", "UA"], "max": 450.5, "x": null}, "q": "a{b}:c", "e": [], "o": {}})),
        ]));
        // text before the call is content; markers never leak
        check(ToolFormat::Gemma4, ReasoningMode::None, false, "Let me check.<|tool_call>call:get_weather{}<tool_call|>",
            &got(Some("Let me check."), &[("get_weather", json!({}))]));
        // bare strings, nested arrays, escapes, a trailing comma
        check(ToolFormat::Gemma4, ReasoningMode::None, false,
            "<|tool_call>call:search{q:hello world,n:-1.5e3,filters:{k:[[1,2],[<|\"|>a\\b\n<|\"|>]],t:True,},}<tool_call|>",
            &got(None, &[("search", json!({"q": "hello world", "n": -1.5e3, "filters": {"k": [[1, 2], ["a\\b\n"]], "t": true}}))]));
    }

    #[test]
    fn arguments_stream_while_they_are_written() {
        let text = "<|tool_call>call:search{q:<|\"|>a long query string<|\"|>,n:5}<tool_call|>";
        let mut s = ToolStream::new(Some(&spec(ToolFormat::Gemma4)), ReasoningMode::None, false, specials(), true);
        let mut frags = Vec::new();
        for p in chunks(text, 4) {
            if let Some(d) = s.push(p).tool_calls {
                for (i, id, name, args) in d.iter() {
                    frags.push((i, id.map(str::len), name.map(String::from), args.to_string()));
                }
            }
        }
        assert!(s.finish(false).tool_calls.is_none());
        assert_eq!(frags[0].2.as_deref(), Some("search"), "the head carries the name");
        assert_eq!(frags[0].1, Some(29), "and the id");
        assert!(frags.len() > 5, "arguments arrive in pieces: {frags:?}");
        let joined: String = frags.iter().map(|f| f.3.as_str()).collect();
        assert_eq!(joined, r#"{"q":"a long query string","n":5}"#);
    }

    #[test]
    fn cut_calls_stay_calls_and_bad_heads_are_text() {
        // cut by max_tokens inside the arguments: the call stands with what was written
        let g = run(ToolFormat::Gemma4, "ok <|tool_call>call:get_weather{city:<|\"|>Par");
        assert_eq!(g, got(Some("ok"), &[("get_weather", json!("{\"city\":\"Par"))]));
        // cut inside the head: text, markers stripped
        assert_eq!(run(ToolFormat::Gemma4, "ok <|tool_call>call:get_wea"), got(Some("ok call:get_wea"), &[]));
        // a head that is not a call
        assert_eq!(run(ToolFormat::Gemma4, "x<|tool_call>nope<tool_call|>y"), got(Some("xnopey"), &[]));
        // hermes object without a name is text; a named one with bad arguments keeps them raw
        let g = run(ToolFormat::Hermes, "<tool_call>\n{\"arguments\": {}}\n</tool_call>");
        assert!(g.calls.is_empty() && g.content.unwrap().contains("arguments"));
        let g = run(ToolFormat::Hermes, "<tool_call>\n{\"name\": \"f\", \"arguments\": {bad}}\n</tool_call>");
        assert_eq!(g.calls, [("f".to_string(), json!("{bad}"))]);
        // a plain answer passes through untouched
        assert_eq!(run(ToolFormat::Hermes, "The answer is <b>4</b>."), got(Some("The answer is <b>4</b>."), &[]));
        assert_eq!(run(ToolFormat::Gemma4, "a < b <| c"), got(Some("a < b <| c"), &[]));
    }

    #[test]
    fn hermes_and_qwen3_thinking() {
        let text = "<think>\nNeed weather.\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Zürich\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"search\", \"arguments\": \"{\\\"q\\\": \\\"x\\\"}\"}\n</tool_call>\n<tool_call>\n{\"arguments\": {\"q\": \"</tool_call>\"}, \"name\": \"search\"}\n</tool_call>";
        let want = Got {
            reasoning: "Need weather.".into(),
            content: None,
            calls: vec![
                ("get_weather".into(), json!({"city": "Zürich"})),
                ("search".into(), json!({"q": "x"})),
                ("search".into(), json!({"q": "</tool_call>"})),
            ],
        };
        check(ToolFormat::Hermes, ReasoningMode::ThinkTag, false, text, &want);
    }

    #[test]
    fn qwen3_xml_types_by_schema() {
        let text = "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n2\n</parameter>\n<parameter=metric>\nfalse\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=search>\n<parameter=q>\n2\n\n</parameter>\n<parameter=filters>\n{\"a\": [1]}\n</parameter>\n<parameter=extra>\n{x}\n</parameter>\n</function>\n</tool_call>";
        check(ToolFormat::Qwen3Xml, ReasoningMode::ThinkTag, true, &format!("plan</think>\n\n{text}"), &Got {
            reasoning: "plan".into(),
            content: Some("Checking.".into()),
            calls: vec![
                ("get_weather".into(), json!({"city": "New York", "days": 2, "metric": false})),
                ("search".into(), json!({"q": "2\n", "filters": {"a": [1]}, "extra": "{x}"})),
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
        check(ToolFormat::Glm45, ReasoningMode::None, false, "<tool_call>get_weather</tool_call>", &got(None, &[("get_weather", json!({}))]));
    }

    #[test]
    fn llama3_json_and_python_tag() {
        let want = got(None, &[("get_weather", json!({"city": "Paris"})), ("search", json!({"q": "x"}))]);
        check(ToolFormat::Llama3Json, ReasoningMode::None, false,
            "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}; {\"name\": \"search\", \"parameters\": {\"q\": \"x\"}}", &want);
        check(ToolFormat::Llama3Json, ReasoningMode::None, false,
            "<|python_tag|>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}; {\"name\": \"search\", \"parameters\": {\"q\": \"x\"}}", &want);
        // JSON that is not a call is an answer
        check(ToolFormat::Llama3Json, ReasoningMode::None, false, "{\"answer\": 4}", &got(Some("{\"answer\": 4}"), &[]));
        assert_eq!(run(ToolFormat::Llama3Json, "Paris is sunny."), got(Some("Paris is sunny."), &[]));
    }

    #[test]
    fn mistral_v3_and_v11() {
        let want = got(None, &[("get_weather", json!({"city": "Paris"})), ("search", json!({"q": "x"}))]);
        check(ToolFormat::Mistral, ReasoningMode::None, false,
            "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, {\"name\": \"search\", \"arguments\": {\"q\": \"x\"}}]", &want);
        check(ToolFormat::Mistral, ReasoningMode::None, false,
            "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}[TOOL_CALLS]search[CALL_ID]a1b2c3d4e[ARGS]{\"q\": \"x\"}", &want);
    }

    #[test]
    fn kimi_k2_keeps_its_ids() {
        let text = "Sure.<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_call_begin|>functions.search:1<|tool_call_argument_begin|> {\"q\": \"x\"} <|tool_call_end|><|tool_calls_section_end|>";
        check(ToolFormat::KimiK2, ReasoningMode::None, false, text,
            &got(Some("Sure."), &[("get_weather", json!({"city": "Paris"})), ("search", json!({"q": "x"}))]));
        let mut s = ToolStream::new(Some(&spec(ToolFormat::KimiK2)), ReasoningMode::None, false, specials(), false);
        s.push(text);
        s.finish(false);
        let ids: Vec<String> = s.take_calls().into_iter().map(|c| c.id).collect();
        assert_eq!(ids, ["functions.get_weather:0", "functions.search:1"]);
    }

    #[test]
    fn deepseek_v3_and_v31() {
        let want = got(Some("Checking."), &[("get_weather", json!({"city": "Paris"})), ("search", json!({"q": "```x```"}))]);
        let v3 = "Checking.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n```json\n{\"q\": \"```x```\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        check(ToolFormat::DeepSeekV3, ReasoningMode::ThinkTag, false, &format!("<think>hm</think>{v3}"), &Got { reasoning: "hm".into(), ..want });
        let v31 = "Checking.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{\"q\": \"```x```\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        check(ToolFormat::DeepSeekV31, ReasoningMode::None, false, v31,
            &got(Some("Checking."), &[("get_weather", json!({"city": "Paris"})), ("search", json!({"q": "```x```"}))]));
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
        // without tools, the parser still splits the channels
        let (g, _) = run_spec(&ParseSpec::plain(ToolFormat::Harmony), ReasoningMode::None, false,
            &["<|channel|>analysis<|message|>Hm.<|end|><|start|>assistant<|channel|>final<|message|>Four."], None);
        assert_eq!(g, Got { reasoning: "Hm.".into(), content: Some("Four.".into()), calls: vec![] });
    }

    #[test]
    fn gemma4_thought_channel_is_reasoning() {
        let mode = ReasoningMode::Tags { open: "<|channel>thought", close: "<channel|>" };
        // after a tool result the model opens its own (here empty) thought channel
        let want = got(Some("It is 18°C in Paris."), &[]);
        check(ToolFormat::Gemma4, mode, false, "<|channel>thought\n<channel|>It is 18°C in Paris.", &want);
        check(ToolFormat::Gemma4, mode, false, "<|channel>thought\nThe tool said 18.<channel|>It is 18°C in Paris.",
            &Got { reasoning: "The tool said 18.".into(), ..want });
        // thinking enabled: the prompt opened the channel
        assert!(mode.prompt_opens("<|turn>model\n<|channel>thought\n"));
        check(ToolFormat::Gemma4, mode, true, "plan<channel|><|tool_call>call:search{q:<|\"|>x<|\"|>}<tool_call|>",
            &Got { reasoning: "plan".into(), content: None, calls: vec![("search".into(), json!({"q": "x"}))] });
        // a request without tools splits the same way
        let mut s = ToolStream::new(None, mode, false, specials(), true);
        let mut c = String::new();
        let mut r = String::new();
        for p in chunks("<|channel>thought\nhm<channel|>Four.", 2) {
            let st = s.push(p);
            c.push_str(st.content.unwrap_or_default());
            r.push_str(st.reasoning.unwrap_or_default());
        }
        let st = s.finish(false);
        c.push_str(st.content.unwrap_or_default());
        assert_eq!((r.trim(), c.as_str()), ("hm", "Four."));
    }

    #[test]
    fn parallel_tool_calls_false_keeps_the_first() {
        let mut sp = spec(ToolFormat::Hermes);
        sp.parallel = false;
        let (g, _) = run_spec(&sp, ReasoningMode::None, false,
            &["<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call><tool_call>{\"name\": \"search\", \"arguments\": {}}</tool_call>"], None);
        assert_eq!(g.calls, [("get_weather".to_string(), json!({}))]);
    }

    #[test]
    fn whitespace_only_answers_survive_when_there_is_no_call() {
        assert_eq!(run(ToolFormat::Hermes, "\n\n"), got(Some("\n\n"), &[]));
    }

    #[test]
    fn forced_calls_are_primed_and_validated() {
        let mut sp = spec(ToolFormat::Gemma4);
        sp.force = Force::Named("get_weather".into());
        let opener = sp.opener().unwrap();
        assert_eq!(opener, "<|tool_call>call:get_weather{");
        let (g, err) = run_spec(&sp, ReasoningMode::None, false, &["city:<|\"|>Paris<|\"|>}", "<tool_call|>"], Some(&opener));
        assert_eq!((g.calls, err), (vec![("get_weather".to_string(), json!({"city": "Paris"}))], None));
        // required: the model picks the name
        sp.force = Force::Required;
        let opener = sp.opener().unwrap();
        let (g, err) = run_spec(&sp, ReasoningMode::None, false, &["search{q:<|\"|>x<|\"|>}<tool_call|>"], Some(&opener));
        assert_eq!((g.calls.len(), err), (1, None));
        // a name that is not declared, or no call at all, fails the turn
        let (_, err) = run_spec(&sp, ReasoningMode::None, false, &["launch{}<tool_call|>"], Some(&opener));
        assert!(err.unwrap().contains("`launch`, which is not in `tools`"));
        let (_, err) = run_spec(&sp, ReasoningMode::None, false, &[" nothing"], Some(&opener));
        assert!(err.unwrap().contains("requires a call"));
        // every format's opener is parsed as the start of a call
        for f in [ToolFormat::Gemma4, ToolFormat::Hermes, ToolFormat::Qwen3Xml, ToolFormat::Glm45, ToolFormat::Llama3Json,
                  ToolFormat::Mistral, ToolFormat::KimiK2, ToolFormat::Harmony, ToolFormat::DeepSeekV3, ToolFormat::DeepSeekV31] {
            let mut sp = spec(f);
            sp.force = Force::Named("search".into());
            let body = match f {
                ToolFormat::Gemma4 => "q:<|\"|>x<|\"|>}<tool_call|>",
                ToolFormat::Hermes => "{\"q\": \"x\"}}\n</tool_call>",
                ToolFormat::Qwen3Xml => "<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>",
                ToolFormat::Glm45 => "<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>",
                ToolFormat::Llama3Json => "{\"q\": \"x\"}}",
                ToolFormat::Mistral => "{\"q\": \"x\"}}]",
                ToolFormat::KimiK2 => "{\"q\": \"x\"}<|tool_call_end|><|tool_calls_section_end|>",
                ToolFormat::Harmony => "{\"q\": \"x\"}",
                ToolFormat::DeepSeekV3 => "{\"q\": \"x\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                ToolFormat::DeepSeekV31 => "{\"q\": \"x\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            };
            let (g, err) = run_spec(&sp, ReasoningMode::None, false, &chunks(body, 3), sp.opener().as_deref());
            assert_eq!((g.calls, err), (vec![("search".to_string(), json!({"q": "x"}))], None), "{f:?}");
        }
    }

    #[test]
    fn strict_arguments_are_held_until_they_validate() {
        let mut sp = spec(ToolFormat::Hermes);
        Arc::make_mut(&mut sp.tools)[0] = ToolDef {
            name: "get_weather".into(),
            params: json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": false}),
            strict: true,
        };
        let mut s = ToolStream::new(Some(&sp), ReasoningMode::None, false, specials(), true);
        let mut args_frames = 0;
        for p in chunks("<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>", 3) {
            if let Some(d) = s.push(p).tool_calls {
                args_frames += d.iter().filter(|(_, _, _, a)| !a.is_empty()).count();
            }
        }
        assert_eq!(args_frames, 1, "one validated delta");
        assert!(s.finish(false).tool_calls.is_none() && s.error().is_none());
        let (_, err) = run_spec(&sp, ReasoningMode::None, false, &["<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"town\": \"Paris\"}}</tool_call>"], None);
        assert!(err.unwrap().contains("missing required property `city`"));
        // a cut turn is returned as generated, not failed
        let mut s = ToolStream::new(Some(&sp), ReasoningMode::None, false, specials(), true);
        s.push("<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"ci");
        let st = s.finish(true);
        let d: Vec<_> = st.tool_calls.unwrap().iter().map(|(_, _, _, a)| a.to_string()).collect();
        assert_eq!(d, ["{\"ci"]);
        assert!(s.error().is_none());
    }

    /// Generations stitched from every format's markers, JSON and multi-byte text: no panic, and
    /// the result does not depend on how the text was split into tokens.
    #[test]
    fn any_split_of_any_text_parses_the_same() {
        const PIECES: &[&str] = &[
            "<|tool_call>", "<tool_call|>", "call:", "get_weather", "search", "{", "}", "[", "]", ":", ",", "<|\"|>", "city",
            "<tool_call>", "</tool_call>", "{\"name\": \"search\", \"arguments\": ", "\"q\": ", "\"x\\\"y\"", "<function=search>",
            "<parameter=q>", "</parameter>", "</function>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
            "[TOOL_CALLS]", "[ARGS]", "<|python_tag|>", "<|tool_calls_section_begin|>", "<|tool_call_begin|>",
            "functions.search:0", "<|tool_call_argument_begin|>", "<|tool_call_end|>", "<|channel|>", "commentary to=functions.search",
            "<|message|>", "<|end|>", "<|call|>", "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>", "function", "<｜tool▁sep｜>",
            "\n```json\n", "```<｜tool▁call▁end｜>", "<think>", "</think>", "<|channel>thought", "<channel|>", "\n", " ", "é日",
            "<", "<|", "1.5", "true", ";",
        ];
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        for f in [ToolFormat::Gemma4, ToolFormat::Hermes, ToolFormat::Qwen3Xml, ToolFormat::Glm45, ToolFormat::Llama3Json,
                  ToolFormat::Mistral, ToolFormat::KimiK2, ToolFormat::Harmony, ToolFormat::DeepSeekV3, ToolFormat::DeepSeekV31] {
            let n: usize = std::env::var("TOOL_FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
            for _ in 0..n {
                let n = 1 + rnd(24);
                let text: String = (0..n).map(|_| PIECES[rnd(PIECES.len())]).collect();
                let whole = run_with(f, ReasoningMode::ThinkTag, false, &[&text]);
                for k in [1, 2, 5] {
                    assert_eq!(run_with(f, ReasoningMode::ThinkTag, false, &chunks(&text, k)), whole, "{f:?} chunks of {k}: {text:?}");
                }
            }
        }
    }

    #[test]
    fn deltas_serialize_as_openai_tool_call_deltas() {
        let mut s = ToolStream::new(Some(&spec(ToolFormat::Hermes)), ReasoningMode::None, false, specials(), true);
        let st = s.push("<tool_call>{\"name\": \"search\", \"arguments\": {\"q\"");
        let v = serde_json::to_value(st.tool_calls.unwrap()).unwrap();
        let id = v[0]["id"].as_str().unwrap().to_string();
        assert_eq!(v, json!([{"index": 0, "id": id, "type": "function", "function": {"name": "search", "arguments": "{\"q\""}}]));
        let st = s.push(": 1}}</tool_call>");
        assert_eq!(serde_json::to_value(st.tool_calls.unwrap()).unwrap(), json!([{"index": 0, "function": {"arguments": ": 1}"}}]));
    }
}

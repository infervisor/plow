//! Streaming tool-call parsers, one per [`ToolFormat`].
//!
//! Each parser is fed the generation's text WITH special tokens and emits [`Ev`]s. A call is
//! emitted only once it is complete and its arguments parse into a JSON object; a call that is
//! malformed or cut off is returned as text instead (it reaches the client stripped of markers,
//! as before this module existed), so a bad generation never fails the request. Text that could
//! still be the beginning of a call marker is held back until the next piece settles it.

use serde_json::{Map, Value};

use super::request::ParseSpec;
use super::ToolFormat;

#[derive(Clone, Debug, PartialEq)]
pub enum Ev {
    Text(String),
    Reasoning(String),
    Call(RawCall),
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawCall {
    pub name: String,
    /// Always a JSON object.
    pub args: Value,
    /// An id the format itself carries (Kimi-K2's `functions.NAME:IDX`).
    pub id: Option<String>,
}

pub trait Parser: Send {
    fn push(&mut self, s: &str, out: &mut Vec<Ev>);
    fn finish(&mut self, out: &mut Vec<Ev>);
}

pub fn parser(spec: &ParseSpec) -> Box<dyn Parser> {
    let d = |start, end, body| -> Box<dyn Parser> {
        Box::new(Delimited { start, end, body, spec: spec.clone(), buf: String::new(), in_call: false })
    };
    match spec.format {
        ToolFormat::Gemma4 => d("<|tool_call>", Some("<tool_call|>"), gemma_body),
        ToolFormat::Hermes => d("<tool_call>", Some("</tool_call>"), hermes_body),
        ToolFormat::Qwen3Xml => d("<tool_call>", Some("</tool_call>"), qwen3_xml_body),
        ToolFormat::Glm45 => d("<tool_call>", Some("</tool_call>"), glm_body),
        ToolFormat::KimiK2 => d("<|tool_calls_section_begin|>", Some("<|tool_calls_section_end|>"), kimi_body),
        // The calls run to the end of the turn.
        ToolFormat::Mistral => d("[TOOL_CALLS]", None, mistral_body),
        ToolFormat::Llama3Json => Box::new(Llama3 { state: LlamaState::Deciding, buf: String::new() }),
        ToolFormat::Harmony => Box::new(Harmony { buf: String::new(), body: None }),
    }
}

/// Bytes at the end of `buf` that could begin one of `markers` and so must not be emitted yet.
fn held(buf: &str, markers: &[&str]) -> usize {
    let longest = markers.iter().map(|m| m.len()).max().unwrap_or(0);
    (1..longest.min(buf.len() + 1))
        .rev()
        .find(|&k| {
            let at = buf.len() - k;
            buf.is_char_boundary(at) && markers.iter().any(|m| m.starts_with(&buf[at..]))
        })
        .unwrap_or(0)
}

type BodyFn = fn(&str, &ParseSpec) -> Option<Vec<RawCall>>;

/// A call framed by a start marker and (optionally) an end marker.
struct Delimited {
    start: &'static str,
    /// `None`: the call runs to the end of the generation.
    end: Option<&'static str>,
    body: BodyFn,
    spec: ParseSpec,
    buf: String,
    in_call: bool,
}

impl Delimited {
    fn emit(&self, body: &str, end: &str, out: &mut Vec<Ev>) {
        match (self.body)(body, &self.spec) {
            Some(calls) if !calls.is_empty() => out.extend(calls.into_iter().map(Ev::Call)),
            _ => out.push(Ev::Text(format!("{}{body}{end}", self.start))),
        }
    }
}

impl Parser for Delimited {
    fn push(&mut self, s: &str, out: &mut Vec<Ev>) {
        self.buf.push_str(s);
        loop {
            if !self.in_call {
                if let Some(i) = self.buf.find(self.start) {
                    if i > 0 {
                        out.push(Ev::Text(self.buf[..i].to_string()));
                    }
                    self.buf.drain(..i + self.start.len());
                    self.in_call = true;
                    continue;
                }
                let n = self.buf.len() - held(&self.buf, &[self.start]);
                if n > 0 {
                    out.push(Ev::Text(self.buf[..n].to_string()));
                    self.buf.drain(..n);
                }
                return;
            }
            let Some(end) = self.end else { return };
            let Some(j) = self.buf.find(end) else { return };
            let body: String = self.buf.drain(..j + end.len()).collect();
            self.in_call = false;
            self.emit(&body[..j], end, out);
        }
    }

    fn finish(&mut self, out: &mut Vec<Ev>) {
        let rest = std::mem::take(&mut self.buf);
        if self.in_call {
            self.in_call = false;
            self.emit(&rest, "", out);
        } else if !rest.is_empty() {
            out.push(Ev::Text(rest));
        }
    }
}

fn object(v: Value) -> Option<Value> {
    match v {
        Value::Object(_) => Some(v),
        Value::String(s) if s.trim().is_empty() => Some(Value::Object(Map::new())),
        Value::String(s) => serde_json::from_str::<Value>(&s).ok().filter(Value::is_object),
        Value::Null => Some(Value::Object(Map::new())),
        _ => None,
    }
}

/// `{"name": .., "arguments"|"parameters": {..}}`.
fn json_call(v: &Value) -> Option<RawCall> {
    let name = v.get("name")?.as_str()?.trim();
    if name.is_empty() {
        return None;
    }
    let args = v.get("arguments").or_else(|| v.get("parameters")).cloned().unwrap_or(Value::Null);
    Some(RawCall { name: name.to_string(), args: object(args)?, id: None })
}

/// One or more call objects (or arrays of them), separated by whitespace, `;` or `,`.
fn json_calls(text: &str) -> Option<Vec<RawCall>> {
    let mut calls = Vec::new();
    let mut rest = text.trim();
    while !rest.is_empty() {
        let mut it = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
        let v = it.next()?.ok()?;
        match &v {
            Value::Array(a) => {
                for c in a {
                    calls.push(json_call(c)?);
                }
            }
            v => calls.push(json_call(v)?),
        }
        rest = rest[it.byte_offset()..].trim_start_matches(|c: char| c.is_whitespace() || c == ';' || c == ',');
    }
    (!calls.is_empty()).then_some(calls)
}

fn hermes_body(body: &str, _: &ParseSpec) -> Option<Vec<RawCall>> {
    json_calls(body)
}

/// A string-carried parameter value, typed by the tool's schema. An undeclared parameter is
/// parsed as JSON when it is valid JSON (GLM renders every non-string with `tojson`) and kept as
/// a string otherwise.
fn typed(spec: &ParseSpec, tool: &str, param: &str, raw: &str, undeclared_json: bool) -> Value {
    let json = || serde_json::from_str::<Value>(raw.trim()).ok();
    match spec.param_type(tool, param) {
        Some("string") => Value::String(raw.to_string()),
        Some("integer" | "number") => json().filter(Value::is_number).unwrap_or_else(|| Value::String(raw.to_string())),
        Some("boolean") => match raw.trim().to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
        Some(_) => json().unwrap_or_else(|| Value::String(raw.to_string())),
        None if undeclared_json => json().unwrap_or_else(|| Value::String(raw.to_string())),
        None => Value::String(raw.to_string()),
    }
}

/// `<function=NAME>\n<parameter=K>\nV\n</parameter>\n...</function>`.
fn qwen3_xml_body(body: &str, spec: &ParseSpec) -> Option<Vec<RawCall>> {
    let mut calls = Vec::new();
    let mut rest = body;
    while let Some(i) = rest.find("<function=") {
        let after = &rest[i + "<function=".len()..];
        let gt = after.find('>')?;
        let name = after[..gt].trim();
        let fend = after.find("</function>")?;
        let mut inner = &after[gt + 1..fend];
        let mut args = Map::new();
        while let Some(p) = inner.find("<parameter=") {
            let a = &inner[p + "<parameter=".len()..];
            let gt = a.find('>')?;
            let key = a[..gt].trim();
            let vend = a.find("</parameter>")?;
            let raw = &a[gt + 1..vend];
            let raw = raw.strip_prefix('\n').unwrap_or(raw);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            args.insert(key.to_string(), typed(spec, name, key, raw, false));
            inner = &a[vend + "</parameter>".len()..];
        }
        if name.is_empty() {
            return None;
        }
        calls.push(RawCall { name: name.to_string(), args: Value::Object(args), id: None });
        rest = &after[fend + "</function>".len()..];
    }
    (!calls.is_empty() && rest.trim().is_empty()).then_some(calls)
}

/// `NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`, with or without newlines between.
fn glm_body(body: &str, spec: &ParseSpec) -> Option<Vec<RawCall>> {
    let (name, mut rest) = match body.find("<arg_key>") {
        Some(i) => (body[..i].trim(), &body[i..]),
        None => (body.trim(), ""),
    };
    if name.is_empty() || name.contains('<') || name.contains(char::is_whitespace) {
        return None;
    }
    let mut args = Map::new();
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let r = rest.strip_prefix("<arg_key>")?;
        let ke = r.find("</arg_key>")?;
        let key = r[..ke].trim();
        let r = r[ke + "</arg_key>".len()..].trim_start().strip_prefix("<arg_value>")?;
        let ve = r.find("</arg_value>")?;
        args.insert(key.to_string(), typed(spec, name, key, &r[..ve], true));
        rest = &r[ve + "</arg_value>".len()..];
    }
    Some(vec![RawCall { name: name.to_string(), args: Value::Object(args), id: None }])
}

const GQ: &str = "<|\"|>";

/// Gemma's value syntax: `<|"|>str<|"|>`, `{key:value,..}` with bare (or quoted) keys,
/// `[v,..]`, and bare `true` / `false` / `null` / numbers.
struct GemmaValue<'a> {
    s: &'a str,
    at: usize,
}

impl<'a> GemmaValue<'a> {
    fn rest(&self) -> &'a str {
        &self.s[self.at..]
    }

    fn ws(&mut self) {
        let r = self.rest();
        self.at += r.len() - r.trim_start().len();
    }

    fn eat(&mut self, t: &str) -> bool {
        self.ws();
        if self.rest().starts_with(t) {
            self.at += t.len();
            true
        } else {
            false
        }
    }

    fn quoted(&mut self) -> Option<String> {
        let r = &self.s[self.at + GQ.len()..];
        let end = r.find(GQ)?;
        let v = r[..end].to_string();
        self.at += GQ.len() + end + GQ.len();
        Some(v)
    }

    fn value(&mut self) -> Option<Value> {
        self.ws();
        let r = self.rest();
        if r.starts_with(GQ) {
            return self.quoted().map(Value::String);
        }
        if self.eat("{") {
            let mut m = Map::new();
            if self.eat("}") {
                return Some(Value::Object(m));
            }
            loop {
                self.ws();
                let key = if self.rest().starts_with(GQ) {
                    self.quoted()?
                } else {
                    let r = self.rest();
                    let end = r.find(':')?;
                    let k = r[..end].trim().to_string();
                    self.at += end;
                    k
                };
                if key.is_empty() || !self.eat(":") {
                    return None;
                }
                m.insert(key, self.value()?);
                if self.eat("}") {
                    return Some(Value::Object(m));
                }
                if !self.eat(",") {
                    return None;
                }
            }
        }
        if self.eat("[") {
            let mut a = Vec::new();
            if self.eat("]") {
                return Some(Value::Array(a));
            }
            loop {
                a.push(self.value()?);
                if self.eat("]") {
                    return Some(Value::Array(a));
                }
                if !self.eat(",") {
                    return None;
                }
            }
        }
        let end = r.find([',', '}', ']']).unwrap_or(r.len());
        let tok = r[..end].trim();
        self.at += end;
        if tok.is_empty() {
            return None;
        }
        Some(match tok {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" | "None" => Value::Null,
            t => serde_json::from_str::<Value>(t).ok().filter(Value::is_number).unwrap_or_else(|| Value::String(t.to_string())),
        })
    }
}

/// `call:NAME{...}`.
fn gemma_body(body: &str, _: &ParseSpec) -> Option<Vec<RawCall>> {
    let b = body.trim().strip_prefix("call:")?;
    let brace = b.find('{')?;
    let name = b[..brace].trim();
    if name.is_empty() || name.contains(char::is_whitespace) {
        return None;
    }
    let mut p = GemmaValue { s: b, at: brace };
    let args = p.value()?;
    p.ws();
    (args.is_object() && p.rest().is_empty())
        .then(|| vec![RawCall { name: name.to_string(), args, id: None }])
}

/// `<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{..}<|tool_call_end|>...`.
fn kimi_body(body: &str, _: &ParseSpec) -> Option<Vec<RawCall>> {
    const B: &str = "<|tool_call_begin|>";
    const A: &str = "<|tool_call_argument_begin|>";
    const E: &str = "<|tool_call_end|>";
    let mut calls = Vec::new();
    let mut rest = body.trim();
    while !rest.is_empty() {
        let r = rest.strip_prefix(B)?;
        let ai = r.find(A)?;
        let id = r[..ai].trim();
        let r = &r[ai + A.len()..];
        let ei = r.find(E)?;
        let args = object(serde_json::from_str::<Value>(r[..ei].trim()).ok()?)?;
        let fq = id.rsplit_once(':').map_or(id, |(n, _)| n);
        let name = fq.split_once('.').map_or(fq, |(_, n)| n);
        if name.is_empty() {
            return None;
        }
        calls.push(RawCall { name: name.to_string(), args, id: Some(id.to_string()) });
        rest = r[ei + E.len()..].trim();
    }
    (!calls.is_empty()).then_some(calls)
}

/// After the first `[TOOL_CALLS]`: a JSON list (v3), or `NAME[ARGS]{..}` segments joined by
/// further `[TOOL_CALLS]` markers (v11+, with an optional `[CALL_ID]id` before `[ARGS]`).
fn mistral_body(body: &str, _: &ParseSpec) -> Option<Vec<RawCall>> {
    let t = body.trim();
    if t.starts_with('[') || t.starts_with('{') {
        return json_calls(t);
    }
    let mut calls = Vec::new();
    for seg in t.split("[TOOL_CALLS]").map(str::trim).filter(|s| !s.is_empty()) {
        let (head, args) = seg.split_once("[ARGS]")?;
        let name = head.split_once("[CALL_ID]").map_or(head, |(n, _)| n).trim();
        if name.is_empty() {
            return None;
        }
        let args = object(serde_json::from_str::<Value>(args.trim()).ok()?)?;
        calls.push(RawCall { name: name.to_string(), args, id: None });
    }
    (!calls.is_empty()).then_some(calls)
}

const PY_TAG: &str = "<|python_tag|>";

enum LlamaState {
    Deciding,
    /// The answer opened with a call: hold it all and parse at the end.
    Capture,
    Pass,
}

/// Llama 3.x has no call marker of its own: an answer that OPENS with `{` (or `<|python_tag|>`)
/// is a candidate call, decided when the turn ends.
struct Llama3 {
    state: LlamaState,
    buf: String,
}

impl Parser for Llama3 {
    fn push(&mut self, s: &str, out: &mut Vec<Ev>) {
        match self.state {
            LlamaState::Pass => out.push(Ev::Text(s.to_string())),
            LlamaState::Capture => self.buf.push_str(s),
            LlamaState::Deciding => {
                self.buf.push_str(s);
                let t = self.buf.trim_start();
                if t.is_empty() || (PY_TAG.starts_with(t) && t != PY_TAG) {
                    return;
                }
                if t.starts_with('{') || t.starts_with(PY_TAG) {
                    self.state = LlamaState::Capture;
                } else {
                    self.state = LlamaState::Pass;
                    out.push(Ev::Text(std::mem::take(&mut self.buf)));
                }
            }
        }
    }

    fn finish(&mut self, out: &mut Vec<Ev>) {
        let buf = std::mem::take(&mut self.buf);
        if let LlamaState::Capture = self.state {
            let t = buf.trim_start();
            if let Some(calls) = json_calls(t.strip_prefix(PY_TAG).unwrap_or(t)) {
                out.extend(calls.into_iter().map(Ev::Call));
                return;
            }
        }
        if !buf.is_empty() {
            out.push(Ev::Text(buf));
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum HarmonyBody {
    Analysis,
    Final,
    Call(String),
}

/// gpt-oss: a sequence of `<|channel|>CH [to=RECIPIENT] [<|constrain|>json]<|message|>BODY<|end|>`
/// messages. `analysis` is the reasoning trace, `final` (and recipient-less `commentary`) is the
/// answer, and a `commentary` message to `functions.NAME` is a call.
struct Harmony {
    buf: String,
    /// `None` while reading a header.
    body: Option<HarmonyBody>,
}

const H_END: [&str; 4] = ["<|end|>", "<|call|>", "<|return|>", "<|start|>"];

impl Harmony {
    fn header(h: &str) -> HarmonyBody {
        let word = |after: &str| -> Option<String> {
            let i = h.find(after)? + after.len();
            let w: String = h[i..].chars().take_while(|c| !c.is_whitespace() && *c != '<').collect();
            Some(w)
        };
        match (word("to="), word("<|channel|>").as_deref()) {
            (Some(r), _) if r.starts_with("functions.") => HarmonyBody::Call(r["functions.".len()..].to_string()),
            (None, Some("analysis")) => HarmonyBody::Analysis,
            _ => HarmonyBody::Final,
        }
    }

    fn close(kind: &HarmonyBody, body: &str, out: &mut Vec<Ev>) {
        match kind {
            HarmonyBody::Analysis if !body.is_empty() => out.push(Ev::Reasoning(body.to_string())),
            HarmonyBody::Final if !body.is_empty() => out.push(Ev::Text(body.to_string())),
            HarmonyBody::Call(name) => match serde_json::from_str::<Value>(body.trim()).ok().and_then(object) {
                Some(args) if !name.is_empty() => out.push(Ev::Call(RawCall { name: name.clone(), args, id: None })),
                _ => out.push(Ev::Text(body.to_string())),
            },
            _ => {}
        }
    }
}

impl Parser for Harmony {
    fn push(&mut self, s: &str, out: &mut Vec<Ev>) {
        self.buf.push_str(s);
        loop {
            match &self.body {
                None => {
                    let t = self.buf.trim_start();
                    if !t.is_empty() && !t.starts_with("<|") && !"<|".starts_with(t) {
                        // No header: the prompt already opened a message.
                        self.body = Some(HarmonyBody::Final);
                        continue;
                    }
                    let Some(i) = self.buf.find("<|message|>") else { return };
                    self.body = Some(Self::header(&self.buf[..i]));
                    self.buf.drain(..i + "<|message|>".len());
                }
                Some(kind) => {
                    let end = H_END.iter().filter_map(|m| self.buf.find(m).map(|i| (i, m.len()))).min();
                    if let Some((i, n)) = end {
                        let kind = kind.clone();
                        let body: String = self.buf.drain(..i + n).collect();
                        Self::close(&kind, &body[..i], out);
                        self.body = None;
                        continue;
                    }
                    if matches!(kind, HarmonyBody::Call(_)) {
                        return;
                    }
                    let n = self.buf.len() - held(&self.buf, &H_END);
                    if n > 0 {
                        let piece: String = self.buf.drain(..n).collect();
                        let kind = kind.clone();
                        Self::close(&kind, &piece, out);
                    }
                    return;
                }
            }
        }
    }

    fn finish(&mut self, out: &mut Vec<Ev>) {
        let rest = std::mem::take(&mut self.buf);
        match self.body.take() {
            Some(kind) => Self::close(&kind, &rest, out),
            None if !rest.trim().is_empty() => out.push(Ev::Text(rest)),
            None => {}
        }
    }
}

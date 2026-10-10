//! Incremental tool-call parsers, one per [`ToolFormat`].
//!
//! Each parser is fed the generation's text WITH special tokens, piece by piece, and writes into
//! an [`Out`]: answer text, reasoning, and call events (a start carrying the name, argument JSON
//! fragments, an end). Every input byte is scanned once. Between pieces a parser keeps only bytes
//! that may begin a marker (bounded by the longest marker) and, until a call's name is read, the
//! call's own bytes (to give them back as text if the head does not parse). Arguments stream as
//! JSON text while the model writes them; formats that do not write JSON (Gemma 4, Qwen3 XML, GLM)
//! are transcoded on the fly, typed by the tool's schema where the format carries strings.
//!
//! A call head that does not parse is returned as text, markers and all (they are stripped
//! later). Once the name is read the call stands: arguments cut off by `max_tokens` are returned as
//! generated, as OpenAI returns them; a transcoded body that breaks the format's syntax is closed
//! into valid JSON at the break.

use std::ops::Range;

use serde_json::Value;

use super::request::ParseSpec;
use super::ToolFormat;

/// One call event, in generation order.
#[derive(Clone, Debug, PartialEq)]
pub enum Ev {
    /// `id`: an id the format itself carries (Kimi-K2's `functions.NAME:IDX`).
    Start { name: String, id: Option<String> },
    /// A fragment of the open call's arguments JSON, as a range of [`Out::args`].
    Args(Range<usize>),
    End,
    /// Answer text, as a range of [`Out::text`].
    Text(Range<usize>),
}

/// What a piece of text produced. Reused across pieces: [`Out::clear`] keeps the capacity.
#[derive(Debug, Default)]
pub struct Out {
    pub text: String,
    pub reasoning: String,
    pub args: String,
    pub evs: Vec<Ev>,
}

impl Out {
    pub fn clear(&mut self) {
        self.text.clear();
        self.reasoning.clear();
        self.args.clear();
        self.evs.clear();
    }

    fn start(&mut self, name: &str, id: Option<String>) {
        self.evs.push(Ev::Start { name: name.to_string(), id });
    }

    /// Answer text, in order with the call events.
    pub fn put_text(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        let a = self.text.len();
        self.text.push_str(s);
        let b = self.text.len();
        if let Some(Ev::Text(r)) = self.evs.last_mut() {
            if r.end == a {
                r.end = b;
                return;
            }
        }
        self.evs.push(Ev::Text(a..b));
    }

    fn end(&mut self) {
        self.evs.push(Ev::End);
    }

    /// Append to the open call's arguments through `f`.
    fn arg_with(&mut self, f: impl FnOnce(&mut String)) {
        let a = self.args.len();
        f(&mut self.args);
        let b = self.args.len();
        if a == b {
            return;
        }
        if let Some(Ev::Args(r)) = self.evs.last_mut() {
            if r.end == a {
                r.end = b;
                return;
            }
        }
        self.evs.push(Ev::Args(a..b));
    }

    fn arg(&mut self, s: &str) {
        self.arg_with(|a| a.push_str(s));
    }

    /// `s` as the inside of a JSON string.
    fn arg_escaped(&mut self, s: &str) {
        self.arg_with(|a| escape_into(a, s));
    }
}

/// JSON string escaping (no quotes), as `serde_json` writes it.
pub fn escape_into(dst: &mut String, s: &str) {
    let b = s.as_bytes();
    let mut run = 0;
    for (i, &c) in b.iter().enumerate() {
        let esc: &str = match c {
            b'"' => "\\\"",
            b'\\' => "\\\\",
            b'\n' => "\\n",
            b'\r' => "\\r",
            b'\t' => "\\t",
            0x08 => "\\b",
            0x0c => "\\f",
            0..=0x1f => "",
            _ => continue,
        };
        dst.push_str(&s[run..i]);
        if esc.is_empty() {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            dst.push_str("\\u00");
            dst.push(HEX[(c >> 4) as usize] as char);
            dst.push(HEX[(c & 15) as usize] as char);
        } else {
            dst.push_str(esc);
        }
        run = i + 1;
    }
    dst.push_str(&s[run..]);
}

pub trait Parser: Send {
    fn push(&mut self, s: &str, out: &mut Out);
    fn finish(&mut self, out: &mut Out);
}

pub fn parser(spec: &ParseSpec) -> Box<dyn Parser> {
    let spec = spec.clone();
    match spec.format {
        ToolFormat::Gemma4 => Framed::boxed("<|tool_call>", Gemma::default(), spec),
        ToolFormat::Hermes => Framed::boxed("<tool_call>", JList::new(Some("</tool_call>")), spec),
        ToolFormat::Qwen3Xml => Framed::boxed("<tool_call>", QwenXml::default(), spec),
        ToolFormat::Glm45 => Framed::boxed("<tool_call>", Glm::default(), spec),
        ToolFormat::KimiK2 => Framed::boxed("<|tool_calls_section_begin|>", Kimi::default(), spec),
        ToolFormat::Mistral => Framed::boxed("[TOOL_CALLS]", Mistral::default(), spec),
        ToolFormat::DeepSeekV3 | ToolFormat::DeepSeekV31 => Framed::boxed(DS_CALLS_BEGIN, DeepSeek::default(), spec),
        ToolFormat::Llama3Json => Box::new(Llama3 { buf: String::new(), raw: String::new(), state: LlamaState::Deciding, list: JList::new(None), spec }),
        ToolFormat::Harmony => Box::new(Harmony { buf: String::new(), head: String::new(), body: None }),
    }
}

/// Where the next marker begins.
enum Hit {
    /// A whole marker at this offset: (offset, marker index).
    At(usize, usize),
    /// No marker in `[..n]`; the rest may begin one.
    Need(usize),
}

fn scan(s: &str, markers: &[&str]) -> Hit {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if markers.iter().any(|m| m.as_bytes()[0] == c) {
            let rest = &s[i..];
            if let Some(k) = markers.iter().position(|m| rest.starts_with(m)) {
                return Hit::At(i, k);
            }
            if markers.iter().any(|m| m.starts_with(rest)) {
                return Hit::Need(i);
            }
        }
        i += 1;
    }
    Hit::Need(b.len())
}

/// `s` starts with `m`: `Some(true)`; could still (more input needed): `None`; cannot: `Some(false)`.
fn starts(s: &str, m: &str, eof: bool) -> Option<bool> {
    if s.starts_with(m) {
        Some(true)
    } else if !eof && m.starts_with(s) {
        None
    } else {
        Some(false)
    }
}

fn ws_len(s: &str) -> usize {
    s.len() - s.trim_start().len()
}

/// Whether a call body made progress, finished, or turned out not to be a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum St {
    Going,
    Done,
    /// The head did not parse; nothing was emitted for it. Bytes not consumed are text.
    Fail,
}

struct Cx<'a> {
    out: &'a mut Out,
    spec: &'a ParseSpec,
    /// Set by the body when it emits `Start`.
    started: bool,
    eof: bool,
}

impl Cx<'_> {
    fn start(&mut self, name: &str, id: Option<String>) {
        self.started = true;
        self.out.start(name, id);
    }
}

/// The body of one call region, after its start marker.
trait Body: Send + Clone {
    /// Consume what it can of `s`; leftover bytes must be a bounded marker prefix.
    fn feed(&mut self, s: &str, cx: &mut Cx) -> (usize, St);
}

/// Text, then call regions opened by `start`.
struct Framed<B: Body> {
    start: &'static str,
    buf: String,
    in_call: bool,
    started: bool,
    /// Region bytes consumed while the head is unresolved, given back as text on failure.
    raw: String,
    body: B,
    /// A fresh body, cloned at each region start.
    proto: B,
    spec: ParseSpec,
}

impl<B: Body + 'static> Framed<B> {
    fn boxed(start: &'static str, body: B, spec: ParseSpec) -> Box<dyn Parser> {
        Box::new(Framed { start, buf: String::new(), in_call: false, started: false, raw: String::new(), body: body.clone(), proto: body, spec })
    }

    fn run(&mut self, s: &str, eof: bool, out: &mut Out) -> usize {
        let mut i = 0;
        loop {
            if !self.in_call {
                match scan(&s[i..], &[self.start]) {
                    Hit::At(j, _) => {
                        out.put_text(&s[i..i + j]);
                        i += j + self.start.len();
                        self.in_call = true;
                        self.started = false;
                        self.raw.clear();
                        self.body = self.proto.clone();
                    }
                    Hit::Need(j) => {
                        let j = if eof { s.len() - i } else { j };
                        out.put_text(&s[i..i + j]);
                        return i + j;
                    }
                }
                continue;
            }
            let mut cx = Cx { out, spec: &self.spec, started: self.started, eof };
            let (n, st) = self.body.feed(&s[i..], &mut cx);
            let now_started = cx.started;
            if !self.started {
                if now_started {
                    self.raw.clear();
                } else {
                    self.raw.push_str(&s[i..i + n]);
                }
            }
            self.started = now_started;
            i += n;
            match st {
                St::Done => self.in_call = false,
                St::Fail => {
                    out.put_text(self.start);
                    out.put_text(&self.raw);
                    self.in_call = false;
                }
                St::Going if !eof => return i,
                St::Going => {
                    // The turn ended inside the region: a started call is closed by the caller.
                    self.in_call = false;
                    if !self.started {
                        out.put_text(self.start);
                        out.put_text(&self.raw);
                        out.put_text(&s[i..]);
                    }
                    return s.len();
                }
            }
        }
    }
}

impl<B: Body + 'static> Parser for Framed<B> {
    fn push(&mut self, s: &str, out: &mut Out) {
        if self.buf.is_empty() {
            let n = self.run(s, false, out);
            self.buf.push_str(&s[n..]);
        } else {
            let mut buf = std::mem::take(&mut self.buf);
            buf.push_str(s);
            let n = self.run(&buf, false, out);
            buf.drain(..n);
            self.buf = buf;
        }
    }

    fn finish(&mut self, out: &mut Out) {
        let buf = std::mem::take(&mut self.buf);
        self.run(&buf, true, out);
    }
}

/// Scans one JSON value from its first byte: an object or array to its matching close, a string
/// to its closing quote, a scalar to the delimiter after it (not included).
#[derive(Clone, Copy, Debug, Default)]
struct JsonScan {
    depth: u32,
    in_str: bool,
    esc: bool,
    began: bool,
    scalar: bool,
}

impl JsonScan {
    /// Bytes of `s` that belong to the value, and whether it ended there.
    fn feed(&mut self, s: &[u8]) -> (usize, bool) {
        for (i, &b) in s.iter().enumerate() {
            if !self.began {
                self.began = true;
                match b {
                    b'{' | b'[' => self.depth = 1,
                    b'"' => self.in_str = true,
                    _ => self.scalar = true,
                }
                continue;
            }
            if self.in_str {
                if self.esc {
                    self.esc = false;
                } else if b == b'\\' {
                    self.esc = true;
                } else if b == b'"' {
                    self.in_str = false;
                    if self.depth == 0 {
                        return (i + 1, true);
                    }
                }
                continue;
            }
            if self.scalar {
                if matches!(b, b',' | b'}' | b']' | b';') || b.is_ascii_whitespace() {
                    return (i, true);
                }
                continue;
            }
            match b {
                b'"' => self.in_str = true,
                b'{' | b'[' => self.depth += 1,
                b'}' | b']' => {
                    self.depth -= 1;
                    if self.depth == 0 {
                        return (i + 1, true);
                    }
                }
                _ => {}
            }
        }
        (s.len(), false)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum J {
    /// Between call objects.
    #[default]
    Between,
    /// Inside an object: a key or `}`.
    Key,
    KeyStr,
    Colon,
    Value,
    NameStr,
    /// The arguments value, streamed (or kept, before the name).
    Args,
    /// A string-encoded arguments value, kept whole and decoded.
    ArgsStr,
    Skip,
    After,
}

/// JSON call objects `{"name": .., "arguments"|"parameters": {..}}`, optionally in `[..]`,
/// separated by whitespace, `,` or `;`, and closed by `end` when the format has one.
#[derive(Clone, Default)]
struct JList {
    end: Option<&'static str>,
    st: J,
    in_array: bool,
    key: String,
    esc: bool,
    name: Option<String>,
    key_is_args: bool,
    /// Arguments kept until the name is known, or a string-encoded value.
    kept: String,
    have_args: bool,
    started: bool,
    scan: JsonScan,
    calls: usize,
}

impl JList {
    fn new(end: Option<&'static str>) -> Self {
        JList { end, ..Default::default() }
    }

    fn open_object(&mut self) {
        self.st = J::Key;
        self.name = None;
        self.have_args = false;
        self.started = false;
        self.kept.clear();
    }

    fn emit_kept_args(&mut self, cx: &mut Cx) {
        if self.kept.starts_with('"') {
            // `"arguments": "{\"q\": 1}"`: the string's own text is the arguments.
            match serde_json::from_str::<String>(&self.kept) {
                Ok(inner) if inner.trim().is_empty() => cx.out.arg("{}"),
                Ok(inner) => cx.out.arg(inner.trim()),
                Err(_) => cx.out.arg(&self.kept),
            }
        } else {
            cx.out.arg(&self.kept);
        }
        self.kept.clear();
    }

    fn close_object(&mut self, cx: &mut Cx) -> St {
        let Some(name) = self.name.take() else {
            return St::Fail;
        };
        if !self.started {
            cx.start(&name, None);
            self.started = true;
            if self.have_args && !self.kept.is_empty() {
                self.emit_kept_args(cx);
            } else if !self.have_args {
                cx.out.arg("{}");
            }
        } else if !self.have_args {
            cx.out.arg("{}");
        }
        cx.out.end();
        self.calls += 1;
        self.st = J::Between;
        St::Going
    }

    /// Fail the region while no call has started; afterwards, stop it at the junk.
    fn junk(&self) -> St {
        if self.calls == 0 && !self.started {
            St::Fail
        } else {
            St::Done
        }
    }

    fn feed(&mut self, s: &str, cx: &mut Cx) -> (usize, St) {
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            match self.st {
                J::Between => {
                    if c.is_ascii_whitespace() || c == b',' || c == b';' {
                        i += 1;
                        continue;
                    }
                    if let Some(end) = self.end {
                        match starts(&s[i..], end, cx.eof) {
                            Some(true) => return (i + end.len(), St::Done),
                            None => return (i, St::Going),
                            Some(false) => {}
                        }
                    }
                    match c {
                        b'{' => {
                            self.open_object();
                            i += 1;
                        }
                        b'[' if !self.in_array => {
                            self.in_array = true;
                            i += 1;
                        }
                        b']' if self.in_array => {
                            self.in_array = false;
                            i += 1;
                        }
                        _ => return (i, self.junk()),
                    }
                }
                J::Key => {
                    if c.is_ascii_whitespace() || c == b',' {
                        i += 1;
                    } else if c == b'"' {
                        self.key.clear();
                        self.esc = false;
                        self.st = J::KeyStr;
                        i += 1;
                    } else if c == b'}' {
                        if self.close_object(cx) == St::Fail {
                            return (i, self.junk());
                        }
                        i += 1;
                    } else {
                        return (i, self.junk());
                    }
                }
                J::KeyStr | J::NameStr => {
                    let mut j = i;
                    while j < b.len() {
                        let d = b[j];
                        if self.esc {
                            self.esc = false;
                        } else if d == b'\\' {
                            self.esc = true;
                        } else if d == b'"' {
                            break;
                        }
                        j += 1;
                    }
                    let target = if self.st == J::KeyStr { &mut self.key } else { self.name.get_or_insert_with(String::new) };
                    target.push_str(&s[i..j]);
                    if j == b.len() {
                        return (j, St::Going);
                    }
                    i = j + 1;
                    if self.st == J::KeyStr {
                        self.st = J::Colon;
                    } else {
                        let n = self.name.as_deref().unwrap_or("").trim();
                        if n.is_empty() || n.contains('\\') {
                            return (i, self.junk());
                        }
                        let n = n.to_string();
                        self.name = Some(n);
                        self.st = J::After;
                    }
                }
                J::Colon => {
                    if c.is_ascii_whitespace() {
                        i += 1;
                    } else if c == b':' {
                        self.st = J::Value;
                        i += 1;
                    } else {
                        return (i, self.junk());
                    }
                }
                J::Value => {
                    if c.is_ascii_whitespace() {
                        i += 1;
                        continue;
                    }
                    match self.key.as_str() {
                        "name" if c == b'"' && self.name.is_none() => {
                            self.name = Some(String::new());
                            self.esc = false;
                            self.st = J::NameStr;
                            i += 1;
                        }
                        "arguments" | "parameters" if !self.have_args => {
                            self.have_args = true;
                            self.scan = JsonScan::default();
                            self.key_is_args = true;
                            match c {
                                b'{' | b'[' => {
                                    if let (Some(n), false) = (self.name.as_deref(), self.started) {
                                        let n = n.to_string();
                                        cx.start(&n, None);
                                        self.started = true;
                                    }
                                    self.st = J::Args;
                                }
                                b'"' => self.st = J::ArgsStr,
                                _ => {
                                    // `null` or another scalar: no arguments.
                                    self.kept.clear();
                                    self.kept.push_str("{}");
                                    self.st = J::Skip;
                                }
                            }
                        }
                        _ => {
                            self.scan = JsonScan::default();
                            self.key_is_args = false;
                            self.st = J::Skip;
                        }
                    }
                }
                J::Args | J::ArgsStr | J::Skip => {
                    let (n, done) = self.scan.feed(&b[i..]);
                    let piece = &s[i..i + n];
                    match self.st {
                        J::Args if self.started => cx.out.arg(piece),
                        J::Args | J::ArgsStr => self.kept.push_str(piece),
                        _ => {}
                    }
                    i += n;
                    if !done {
                        return (i, St::Going);
                    }
                    if self.st == J::ArgsStr && self.name.is_some() && !self.started {
                        let n = self.name.clone().unwrap_or_default();
                        cx.start(&n, None);
                        self.started = true;
                        self.emit_kept_args(cx);
                    } else if self.st == J::Skip && self.key_is_args && self.started {
                        cx.out.arg("{}");
                        self.kept.clear();
                    }
                    self.st = J::After;
                }
                J::After => {
                    if c.is_ascii_whitespace() {
                        i += 1;
                    } else if c == b',' {
                        self.st = J::Key;
                        i += 1;
                    } else if c == b'}' {
                        if self.close_object(cx) == St::Fail {
                            return (i, self.junk());
                        }
                        i += 1;
                    } else {
                        return (i, self.junk());
                    }
                }
            }
        }
        (i, St::Going)
    }
}

impl Body for JList {
    fn feed(&mut self, s: &str, cx: &mut Cx) -> (usize, St) {
        JList::feed(self, s, cx)
    }
}

/// A transcoder's open containers and the comma owed before the next member.
#[derive(Clone, Default)]
struct Nest {
    stack: Vec<u8>,
    comma: bool,
}

impl Nest {
    fn open(&mut self, c: u8, out: &mut Out) {
        self.member(out);
        out.arg(if c == b'{' { "{" } else { "[" });
        self.stack.push(c);
        self.comma = false;
    }

    /// Before a key or array value.
    fn member(&mut self, out: &mut Out) {
        if self.comma {
            out.arg(",");
            self.comma = false;
        }
    }

    fn close(&mut self, c: u8, out: &mut Out) -> bool {
        let want = if c == b'}' { b'{' } else { b'[' };
        if self.stack.last() != Some(&want) {
            return false;
        }
        self.stack.pop();
        out.arg(if c == b'}' { "}" } else { "]" });
        self.comma = true;
        true
    }

    /// Close everything still open (a broken body), then the call.
    fn seal(&mut self, out: &mut Out) {
        while let Some(c) = self.stack.pop() {
            out.arg(if c == b'{' { "}" } else { "]" });
        }
        out.end();
    }
}

/// A bare scalar as JSON: literals and numbers as themselves, anything else as a string.
fn bare_json(t: &str, out: &mut Out) {
    let t = t.trim();
    match t {
        "true" | "false" | "null" => out.arg(t),
        "True" => out.arg("true"),
        "False" => out.arg("false"),
        "None" => out.arg("null"),
        _ if serde_json::from_str::<serde_json::Number>(t).is_ok() => out.arg(t),
        _ => out.arg_with(|a| {
            a.push('"');
            escape_into(a, t);
            a.push('"');
        }),
    }
}

const GQ: &str = "<|\"|>";
const G_END: &str = "<tool_call|>";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum G {
    #[default]
    Call,
    Name,
    Key,
    KeyQuoted,
    Colon,
    Value,
    Str,
    Bare,
    After,
    Close,
    Skip,
}

/// Gemma 4: `call:NAME{key:<|"|>str<|"|>,n:1,o:{..},a:[..]}<tool_call|>`.
#[derive(Clone, Default)]
struct Gemma {
    st: G,
    acc: String,
    nest: Nest,
}

impl Gemma {
    /// A syntax break inside the arguments: close them and skip to the end marker.
    fn broken(&mut self, out: &mut Out) {
        self.nest.seal(out);
        self.st = G::Skip;
    }

    /// After a close: the call ends with its outermost object.
    fn closed(&mut self, out: &mut Out) {
        if self.nest.stack.is_empty() {
            out.end();
            self.st = G::Close;
        } else {
            self.st = G::After;
        }
    }
}

impl Body for Gemma {
    fn feed(&mut self, s: &str, cx: &mut Cx) -> (usize, St) {
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            match self.st {
                G::Call => {
                    i += ws_len(&s[i..]);
                    if i == b.len() {
                        break;
                    }
                    match starts(&s[i..], "call:", cx.eof) {
                        Some(true) => {
                            i += 5;
                            self.st = G::Name;
                        }
                        None => return (i, St::Going),
                        Some(false) => return (i, St::Fail),
                    }
                }
                G::Name => {
                    let j = s[i..].find(|ch: char| ch == '{' || ch == '<' || ch.is_whitespace()).map_or(b.len(), |k| i + k);
                    self.acc.push_str(&s[i..j]);
                    i = j;
                    if i == b.len() {
                        break;
                    }
                    if b[i] != b'{' || self.acc.is_empty() {
                        return (i, St::Fail);
                    }
                    let name = std::mem::take(&mut self.acc);
                    cx.start(&name, None);
                    self.nest.open(b'{', cx.out);
                    self.st = G::Key;
                    i += 1;
                }
                G::Key => {
                    if c.is_ascii_whitespace() {
                        i += 1;
                    } else if c == b'}' {
                        self.nest.close(b'}', cx.out);
                        i += 1;
                        self.closed(cx.out);
                    } else if c == b'<' {
                        match starts(&s[i..], GQ, cx.eof) {
                            Some(true) => {
                                i += GQ.len();
                                self.acc.clear();
                                self.st = G::KeyQuoted;
                            }
                            None => return (i, St::Going),
                            Some(false) => self.broken(cx.out),
                        }
                    } else {
                        self.acc.clear();
                        self.st = G::Colon;
                    }
                }
                G::KeyQuoted => match scan(&s[i..], &[GQ]) {
                    Hit::At(j, _) => {
                        self.acc.push_str(&s[i..i + j]);
                        i += j + GQ.len();
                        self.st = G::Colon;
                    }
                    Hit::Need(j) => {
                        self.acc.push_str(&s[i..i + j]);
                        return (i + j, St::Going);
                    }
                },
                G::Colon => {
                    // A bare key runs to the colon.
                    let j = s[i..].find([':', ',', '}', '<']).map_or(b.len(), |k| i + k);
                    self.acc.push_str(&s[i..j]);
                    i = j;
                    if i == b.len() {
                        break;
                    }
                    let key = self.acc.trim();
                    if b[i] != b':' || key.is_empty() {
                        self.broken(cx.out);
                        continue;
                    }
                    self.nest.member(cx.out);
                    cx.out.arg_with(|a| {
                        a.push('"');
                        escape_into(a, key);
                        a.push_str("\":");
                    });
                    i += 1;
                    self.st = G::Value;
                }
                G::Value => {
                    if c.is_ascii_whitespace() {
                        i += 1;
                        continue;
                    }
                    match c {
                        b'{' => {
                            self.nest.open(b'{', cx.out);
                            self.st = G::Key;
                            i += 1;
                        }
                        b'[' => {
                            self.nest.open(b'[', cx.out);
                            i += 1;
                        }
                        b']' if self.nest.stack.last() == Some(&b'[') => {
                            self.nest.close(b']', cx.out);
                            i += 1;
                            self.closed(cx.out);
                        }
                        b'<' => match starts(&s[i..], GQ, cx.eof) {
                            Some(true) => {
                                if self.nest.stack.last() == Some(&b'[') {
                                    self.nest.member(cx.out);
                                }
                                cx.out.arg("\"");
                                i += GQ.len();
                                self.st = G::Str;
                            }
                            None => return (i, St::Going),
                            Some(false) => self.broken(cx.out),
                        },
                        _ => {
                            if self.nest.stack.last() == Some(&b'[') {
                                self.nest.member(cx.out);
                            }
                            self.acc.clear();
                            self.st = G::Bare;
                        }
                    }
                }
                G::Str => match scan(&s[i..], &[GQ]) {
                    Hit::At(j, _) => {
                        cx.out.arg_escaped(&s[i..i + j]);
                        cx.out.arg("\"");
                        i += j + GQ.len();
                        self.nest.comma = true;
                        self.st = G::After;
                    }
                    Hit::Need(j) => {
                        cx.out.arg_escaped(&s[i..i + j]);
                        return (i + j, St::Going);
                    }
                },
                G::Bare => {
                    let j = s[i..].find([',', '}', ']', '<']).map_or(b.len(), |k| i + k);
                    self.acc.push_str(&s[i..j]);
                    i = j;
                    if i == b.len() {
                        break;
                    }
                    bare_json(&self.acc, cx.out);
                    self.nest.comma = true;
                    self.st = G::After;
                }
                G::After => {
                    if c.is_ascii_whitespace() {
                        i += 1;
                        continue;
                    }
                    match c {
                        b',' => {
                            self.st = if self.nest.stack.last() == Some(&b'{') { G::Key } else { G::Value };
                            i += 1;
                        }
                        b'}' | b']' => {
                            if self.nest.close(c, cx.out) {
                                i += 1;
                                self.closed(cx.out);
                            } else {
                                self.broken(cx.out);
                            }
                        }
                        _ => self.broken(cx.out),
                    }
                }
                G::Close | G::Skip => match scan(&s[i..], &[G_END]) {
                    Hit::At(j, _) => return (i + j + G_END.len(), St::Done),
                    Hit::Need(j) => return (i + j, St::Going),
                },
            }
        }
        (b.len(), St::Going)
    }
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

/// A parameter value framed by `close`: streamed as a JSON string when the schema says string,
/// kept and typed otherwise.
#[derive(Clone, Default)]
struct Param {
    name: String,
    key: String,
    stream: bool,
    /// Drop one newline after the open tag and before the close tag (Qwen3 XML).
    trim_nl: bool,
    lead: bool,
    nl_held: bool,
    raw: String,
}

impl Param {
    fn begin(&mut self, cx: &mut Cx, nest: &mut Nest, undeclared_json: bool) {
        nest.member(cx.out);
        let key = self.key.trim();
        cx.out.arg_with(|a| {
            a.push('"');
            escape_into(a, key);
            a.push_str("\":");
        });
        self.stream = cx.spec.param_type(&self.name, key) == Some("string") || (!undeclared_json && cx.spec.param_type(&self.name, key).is_none());
        self.lead = self.trim_nl;
        self.nl_held = false;
        self.raw.clear();
        if self.stream {
            cx.out.arg("\"");
        }
    }

    fn piece(&mut self, mut p: &str, out: &mut Out) {
        if p.is_empty() {
            return;
        }
        if self.lead {
            self.lead = false;
            p = p.strip_prefix('\n').unwrap_or(p);
        }
        if !self.stream {
            self.raw.push_str(p);
            return;
        }
        if p.is_empty() {
            return;
        }
        if self.nl_held {
            out.arg("\\n");
            self.nl_held = false;
        }
        if self.trim_nl && p.ends_with('\n') {
            p = &p[..p.len() - 1];
            self.nl_held = true;
        }
        out.arg_escaped(p);
    }

    fn end(&mut self, cx: &mut Cx, nest: &mut Nest, undeclared_json: bool) {
        if self.stream {
            cx.out.arg("\"");
        } else {
            let mut raw = self.raw.as_str();
            if self.trim_nl {
                raw = raw.strip_suffix('\n').unwrap_or(raw);
            }
            let v = typed(cx.spec, &self.name, self.key.trim(), raw, undeclared_json);
            cx.out.arg(&v.to_string());
        }
        nest.comma = true;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Q {
    #[default]
    Func,
    FName,
    Params,
    PName,
    PVal,
    AfterFunc,
}

/// Qwen3-Coder / Qwen3.5: `<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`.
#[derive(Clone, Default)]
struct QwenXml {
    st: Q,
    nest: Nest,
    p: Param,
    calls: usize,
}

impl Body for QwenXml {
    fn feed(&mut self, s: &str, cx: &mut Cx) -> (usize, St) {
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            match self.st {
                Q::Func => {
                    i += ws_len(&s[i..]);
                    if i == b.len() {
                        break;
                    }
                    match starts(&s[i..], "<function=", cx.eof) {
                        Some(true) => {
                            i += "<function=".len();
                            self.p.name.clear();
                            self.st = Q::FName;
                        }
                        None => return (i, St::Going),
                        Some(false) => return (i, if self.calls == 0 { St::Fail } else { St::Done }),
                    }
                }
                Q::FName => {
                    let j = s[i..].find(['>', '<', '\n']).map_or(b.len(), |k| i + k);
                    self.p.name.push_str(&s[i..j]);
                    i = j;
                    if i == b.len() {
                        break;
                    }
                    let name = self.p.name.trim().to_string();
                    if b[i] != b'>' || name.is_empty() {
                        return (i, if self.calls == 0 { St::Fail } else { St::Done });
                    }
                    self.p.name = name;
                    cx.start(&self.p.name.clone(), None);
                    self.nest = Nest::default();
                    self.nest.open(b'{', cx.out);
                    self.calls += 1;
                    i += 1;
                    self.st = Q::Params;
                }
                Q::Params => match scan(&s[i..], &["<parameter=", "</function>", "</tool_call>"]) {
                    Hit::At(j, 0) => {
                        i += j + "<parameter=".len();
                        self.p.key.clear();
                        self.st = Q::PName;
                    }
                    Hit::At(j, k) => {
                        self.nest.seal(cx.out);
                        if k == 1 {
                            i += j + "</function>".len();
                            self.st = Q::AfterFunc;
                        } else {
                            return (i + j + "</tool_call>".len(), St::Done);
                        }
                    }
                    Hit::Need(j) => return (i + j, St::Going),
                },
                Q::PName => {
                    let j = s[i..].find(['>', '<', '\n']).map_or(b.len(), |k| i + k);
                    self.p.key.push_str(&s[i..j]);
                    i = j;
                    if i == b.len() {
                        break;
                    }
                    if b[i] != b'>' {
                        self.st = Q::Params;
                        continue;
                    }
                    i += 1;
                    self.p.trim_nl = true;
                    self.p.begin(cx, &mut self.nest, false);
                    self.st = Q::PVal;
                }
                Q::PVal => match scan(&s[i..], &["</parameter>", "</function>"]) {
                    Hit::At(j, k) => {
                        self.p.piece(&s[i..i + j], cx.out);
                        self.p.end(cx, &mut self.nest, false);
                        i += j;
                        if k == 0 {
                            i += "</parameter>".len();
                        }
                        self.st = Q::Params;
                    }
                    Hit::Need(j) => {
                        self.p.piece(&s[i..i + j], cx.out);
                        return (i + j, St::Going);
                    }
                },
                Q::AfterFunc => {
                    i += ws_len(&s[i..]);
                    if i == b.len() {
                        break;
                    }
                    match scan(&s[i..], &["</tool_call>", "<function="]) {
                        Hit::At(0, 0) => return (i + "</tool_call>".len(), St::Done),
                        Hit::At(0, _) => self.st = Q::Func,
                        Hit::Need(0) if !cx.eof => return (i, St::Going),
                        _ => return (i, St::Done),
                    }
                }
            }
        }
        (b.len(), St::Going)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Gl {
    #[default]
    Name,
    Args,
    Key,
    Val0,
    Val,
}

/// GLM-4.5 .. 5.x: `NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`, with or
/// without newlines between.
#[derive(Clone, Default)]
struct Glm {
    st: Gl,
    nest: Nest,
    p: Param,
}

impl Body for Glm {
    fn feed(&mut self, s: &str, cx: &mut Cx) -> (usize, St) {
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            match self.st {
                Gl::Name => {
                    let j = s[i..].find(['<', '\n']).map_or(b.len(), |k| i + k);
                    self.p.name.push_str(&s[i..j]);
                    i = j;
                    if i == b.len() {
                        break;
                    }
                    if b[i] == b'\n' && self.p.name.trim().is_empty() {
                        i += 1;
                        continue;
                    }
                    let name = self.p.name.trim().to_string();
                    if name.is_empty() || name.contains(char::is_whitespace) {
                        return (i, St::Fail);
                    }
                    self.p.name = name;
                    cx.start(&self.p.name.clone(), None);
                    self.nest.open(b'{', cx.out);
                    self.st = Gl::Args;
                }
                Gl::Args => match scan(&s[i..], &["<arg_key>", "</tool_call>"]) {
                    Hit::At(j, 0) => {
                        i += j + "<arg_key>".len();
                        self.p.key.clear();
                        self.st = Gl::Key;
                    }
                    Hit::At(j, _) => {
                        self.nest.seal(cx.out);
                        return (i + j + "</tool_call>".len(), St::Done);
                    }
                    Hit::Need(j) => return (i + j, St::Going),
                },
                Gl::Key => match scan(&s[i..], &["</arg_key>"]) {
                    Hit::At(j, _) => {
                        self.p.key.push_str(&s[i..i + j]);
                        i += j + "</arg_key>".len();
                        self.st = Gl::Val0;
                    }
                    Hit::Need(j) => {
                        self.p.key.push_str(&s[i..i + j]);
                        return (i + j, St::Going);
                    }
                },
                Gl::Val0 => {
                    i += ws_len(&s[i..]);
                    if i == b.len() {
                        break;
                    }
                    match starts(&s[i..], "<arg_value>", cx.eof) {
                        Some(true) => {
                            i += "<arg_value>".len();
                            self.p.trim_nl = false;
                            self.p.begin(cx, &mut self.nest, true);
                            self.st = Gl::Val;
                        }
                        None => return (i, St::Going),
                        Some(false) => self.st = Gl::Args,
                    }
                }
                Gl::Val => match scan(&s[i..], &["</arg_value>", "</tool_call>"]) {
                    Hit::At(j, k) => {
                        self.p.piece(&s[i..i + j], cx.out);
                        self.p.end(cx, &mut self.nest, true);
                        i += j;
                        if k == 0 {
                            i += "</arg_value>".len();
                        }
                        self.st = Gl::Args;
                    }
                    Hit::Need(j) => {
                        self.p.piece(&s[i..i + j], cx.out);
                        return (i + j, St::Going);
                    }
                },
            }
        }
        (b.len(), St::Going)
    }
}

/// Raw JSON arguments up to a closing marker: leading whitespace dropped, trailing whitespace
/// held until something follows it.
#[derive(Clone, Debug, Default, PartialEq)]
struct RawArgs {
    began: bool,
    ws: String,
}

impl RawArgs {
    fn piece(&mut self, p: &str, out: &mut Out) {
        let p = if self.began { p } else { p.trim_start() };
        if p.is_empty() {
            return;
        }
        self.began = true;
        let body = p.trim_end();
        if body.is_empty() {
            self.ws.push_str(p);
            return;
        }
        if !self.ws.is_empty() {
            out.arg(&self.ws);
            self.ws.clear();
        }
        out.arg(body);
        self.ws.push_str(&p[body.len()..]);
    }

    fn end(&mut self, out: &mut Out) {
        if !self.began {
            out.arg("{}");
        }
        self.began = false;
        self.ws.clear();
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum K {
    #[default]
    Between,
    Id,
    Args,
}

/// Kimi-K2: `<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{..}<|tool_call_end|>`
/// repeated, then `<|tool_calls_section_end|>`.
#[derive(Clone, Default)]
struct Kimi {
    st: K,
    id: String,
    args: RawArgs,
    calls: usize,
}

impl Body for Kimi {
    fn feed(&mut self, s: &str, cx: &mut Cx) -> (usize, St) {
        const B: &str = "<|tool_call_begin|>";
        const A: &str = "<|tool_call_argument_begin|>";
        const E: &str = "<|tool_call_end|>";
        const SE: &str = "<|tool_calls_section_end|>";
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            match self.st {
                K::Between => {
                    i += ws_len(&s[i..]);
                    if i == b.len() {
                        break;
                    }
                    match scan(&s[i..], &[B, SE]) {
                        Hit::At(0, 0) => {
                            i += B.len();
                            self.id.clear();
                            self.st = K::Id;
                        }
                        Hit::At(0, _) => return (i + SE.len(), St::Done),
                        Hit::Need(0) if !cx.eof => return (i, St::Going),
                        _ => return (i, if self.calls == 0 && !cx.started { St::Fail } else { St::Done }),
                    }
                }
                K::Id => match scan(&s[i..], &[A]) {
                    Hit::At(j, _) => {
                        self.id.push_str(&s[i..i + j]);
                        i += j + A.len();
                        let id = self.id.trim();
                        let fq = id.rsplit_once(':').map_or(id, |(n, _)| n);
                        let name = fq.split_once('.').map_or(fq, |(_, n)| n);
                        if name.is_empty() {
                            return (i, if self.calls == 0 { St::Fail } else { St::Done });
                        }
                        let (name, id) = (name.to_string(), id.to_string());
                        cx.start(&name, Some(id));
                        self.st = K::Args;
                    }
                    Hit::Need(j) => {
                        self.id.push_str(&s[i..i + j]);
                        return (i + j, St::Going);
                    }
                },
                K::Args => match scan(&s[i..], &[E, SE]) {
                    Hit::At(j, k) => {
                        self.args.piece(&s[i..i + j], cx.out);
                        self.args.end(cx.out);
                        cx.out.end();
                        self.calls += 1;
                        i += j;
                        if k == 1 {
                            return (i + SE.len(), St::Done);
                        }
                        i += E.len();
                        self.st = K::Between;
                    }
                    Hit::Need(j) => {
                        self.args.piece(&s[i..i + j], cx.out);
                        return (i + j, St::Going);
                    }
                },
            }
        }
        (b.len(), St::Going)
    }
}

const DS_CALLS_BEGIN: &str = "<｜tool▁calls▁begin｜>";
const DS_CALLS_END: &str = "<｜tool▁calls▁end｜>";
const DS_CALL_BEGIN: &str = "<｜tool▁call▁begin｜>";
const DS_CALL_END: &str = "<｜tool▁call▁end｜>";
const DS_SEP: &str = "<｜tool▁sep｜>";
const DS_FENCE_END: &str = "```<｜tool▁call▁end｜>";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum D {
    #[default]
    Between,
    Head,
    V3Name,
    V3Fence,
    V3Args,
    V31Args,
}

/// DeepSeek V3 / R1: `<｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\n{..}\n```<｜tool▁call▁end｜>`;
/// V3.1: `<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{..}<｜tool▁call▁end｜>`; either repeated, then
/// `<｜tool▁calls▁end｜>`.
#[derive(Clone, Default)]
struct DeepSeek {
    st: D,
    head: String,
    args: RawArgs,
    calls: usize,
}

impl DeepSeek {
    fn fail_or_done(&self, cx: &Cx) -> St {
        if self.calls == 0 && !cx.started {
            St::Fail
        } else {
            St::Done
        }
    }
}

impl Body for DeepSeek {
    fn feed(&mut self, s: &str, cx: &mut Cx) -> (usize, St) {
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            match self.st {
                D::Between => {
                    i += ws_len(&s[i..]);
                    if i == b.len() {
                        break;
                    }
                    match scan(&s[i..], &[DS_CALL_BEGIN, DS_CALLS_END]) {
                        Hit::At(0, 0) => {
                            i += DS_CALL_BEGIN.len();
                            self.head.clear();
                            self.st = D::Head;
                        }
                        Hit::At(0, _) => return (i + DS_CALLS_END.len(), St::Done),
                        Hit::Need(0) if !cx.eof => return (i, St::Going),
                        _ => return (i, self.fail_or_done(cx)),
                    }
                }
                D::Head => match scan(&s[i..], &[DS_SEP, DS_CALL_END]) {
                    Hit::At(j, 0) => {
                        self.head.push_str(&s[i..i + j]);
                        i += j + DS_SEP.len();
                        if self.head.trim() == "function" {
                            self.head.clear();
                            self.st = D::V3Name;
                        } else {
                            let name = self.head.trim().to_string();
                            if name.is_empty() {
                                return (i, self.fail_or_done(cx));
                            }
                            cx.start(&name, None);
                            self.st = D::V31Args;
                        }
                    }
                    Hit::At(j, _) => return (i + j, self.fail_or_done(cx)),
                    Hit::Need(j) => {
                        self.head.push_str(&s[i..i + j]);
                        return (i + j, St::Going);
                    }
                },
                D::V3Name => {
                    let j = s[i..].find(['\n', '<']).map_or(b.len(), |k| i + k);
                    self.head.push_str(&s[i..j]);
                    i = j;
                    if i == b.len() {
                        break;
                    }
                    let name = self.head.trim().to_string();
                    if b[i] != b'\n' || name.is_empty() {
                        return (i, self.fail_or_done(cx));
                    }
                    cx.start(&name, None);
                    i += 1;
                    self.st = D::V3Fence;
                }
                D::V3Fence => {
                    i += ws_len(&s[i..]);
                    if i == b.len() {
                        break;
                    }
                    match starts(&s[i..], "```json", cx.eof) {
                        Some(true) => {
                            i += "```json".len();
                            self.st = D::V3Args;
                        }
                        None => return (i, St::Going),
                        Some(false) => self.st = D::V3Args,
                    }
                }
                D::V3Args | D::V31Args => {
                    let marks: &[&str] = if self.st == D::V3Args { &[DS_FENCE_END, DS_CALL_END] } else { &[DS_CALL_END] };
                    match scan(&s[i..], marks) {
                        Hit::At(j, k) => {
                            self.args.piece(&s[i..i + j], cx.out);
                            self.args.end(cx.out);
                            cx.out.end();
                            self.calls += 1;
                            i += j + marks[k].len();
                            self.st = D::Between;
                        }
                        Hit::Need(j) => {
                            self.args.piece(&s[i..i + j], cx.out);
                            return (i + j, St::Going);
                        }
                    }
                }
            }
        }
        (b.len(), St::Going)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum M {
    #[default]
    Decide,
    List,
    Name,
    CallId,
    Args0,
    Args,
    After,
}

/// Mistral, after the first `[TOOL_CALLS]`: a JSON list (v3), or `NAME[ARGS]{..}` segments joined
/// by further `[TOOL_CALLS]` (v11+, with an optional `[CALL_ID]id` before `[ARGS]`). The calls run
/// to the end of the turn.
#[derive(Clone, Default)]
struct Mistral {
    st: M,
    list: JList,
    name: String,
    scan: JsonScan,
    calls: usize,
}

impl Body for Mistral {
    fn feed(&mut self, s: &str, cx: &mut Cx) -> (usize, St) {
        const TC: &str = "[TOOL_CALLS]";
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            match self.st {
                M::Decide => {
                    i += ws_len(&s[i..]);
                    if i == b.len() {
                        break;
                    }
                    if b[i] == b'[' || b[i] == b'{' {
                        self.list = JList::new(None);
                        self.st = M::List;
                    } else {
                        self.name.clear();
                        self.st = M::Name;
                    }
                }
                M::List => {
                    let (n, st) = self.list.feed(&s[i..], cx);
                    return (i + n, st);
                }
                M::Name | M::CallId => match scan(&s[i..], &["[ARGS]", "[CALL_ID]", TC]) {
                    Hit::At(j, k) => {
                        if self.st == M::Name {
                            self.name.push_str(&s[i..i + j]);
                        }
                        i += j;
                        match k {
                            0 => {
                                i += "[ARGS]".len();
                                let name = self.name.trim().to_string();
                                if name.is_empty() || name.contains(char::is_whitespace) {
                                    return (i, if self.calls == 0 { St::Fail } else { St::Done });
                                }
                                cx.start(&name, None);
                                self.st = M::Args0;
                            }
                            1 => {
                                i += "[CALL_ID]".len();
                                self.st = M::CallId;
                            }
                            _ => return (i, if self.calls == 0 { St::Fail } else { St::Done }),
                        }
                    }
                    Hit::Need(j) => {
                        if self.st == M::Name {
                            self.name.push_str(&s[i..i + j]);
                        }
                        return (i + j, St::Going);
                    }
                },
                M::Args0 => {
                    i += ws_len(&s[i..]);
                    if i == b.len() {
                        break;
                    }
                    self.scan = JsonScan::default();
                    self.st = M::Args;
                }
                M::Args => {
                    let (n, done) = self.scan.feed(&b[i..]);
                    cx.out.arg(&s[i..i + n]);
                    i += n;
                    if !done {
                        break;
                    }
                    cx.out.end();
                    self.calls += 1;
                    self.st = M::After;
                }
                M::After => {
                    i += ws_len(&s[i..]);
                    if i == b.len() {
                        break;
                    }
                    match starts(&s[i..], TC, cx.eof) {
                        Some(true) => {
                            i += TC.len();
                            self.name.clear();
                            self.st = M::Name;
                        }
                        None => return (i, St::Going),
                        Some(false) => return (i, St::Done),
                    }
                }
            }
        }
        (b.len(), St::Going)
    }
}

const PY_TAG: &str = "<|python_tag|>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LlamaState {
    Deciding,
    List,
    Pass,
}

/// Llama 3.x has no call marker of its own: an answer that OPENS with `{` (or `<|python_tag|>`)
/// is a candidate call list; an object that turns out not to be a call is text.
struct Llama3 {
    buf: String,
    /// Everything since the answer began, until the first call starts.
    raw: String,
    state: LlamaState,
    list: JList,
    spec: ParseSpec,
}

impl Llama3 {
    fn run(&mut self, s: &str, eof: bool, out: &mut Out) -> usize {
        let mut i = 0;
        loop {
            match self.state {
                LlamaState::Pass => {
                    out.put_text(&s[i..]);
                    return s.len();
                }
                LlamaState::Deciding => {
                    let ws = ws_len(&s[i..]);
                    let t = &s[i + ws..];
                    if t.is_empty() {
                        return if eof {
                            out.put_text(&s[i..]);
                            s.len()
                        } else {
                            i
                        };
                    }
                    if let Some(rest) = t.strip_prefix(PY_TAG) {
                        let k = s.len() - rest.len();
                        self.raw.push_str(&s[i..k]);
                        i = k;
                        continue;
                    }
                    if !eof && PY_TAG.starts_with(t) {
                        return i;
                    }
                    if t.starts_with('{') {
                        self.raw.push_str(&s[i..i + ws]);
                        i += ws;
                        self.state = LlamaState::List;
                    } else {
                        out.put_text(&self.raw);
                        self.raw.clear();
                        self.state = LlamaState::Pass;
                    }
                }
                LlamaState::List => {
                    let mut cx = Cx { out, spec: &self.spec, started: self.list.calls > 0 || self.list.started, eof };
                    let before = cx.started;
                    let (n, st) = self.list.feed(&s[i..], &mut cx);
                    let started = cx.started;
                    if !before {
                        if started {
                            self.raw.clear();
                        } else {
                            self.raw.push_str(&s[i..i + n]);
                        }
                    }
                    i += n;
                    match st {
                        St::Going if !eof => return i,
                        St::Going => {
                            if !started {
                                out.put_text(&self.raw);
                                out.put_text(&s[i..]);
                            }
                            self.raw.clear();
                            return s.len();
                        }
                        St::Fail => {
                            out.put_text(&self.raw);
                            self.raw.clear();
                            self.state = LlamaState::Pass;
                        }
                        St::Done => self.state = LlamaState::Pass,
                    }
                }
            }
        }
    }
}

impl Parser for Llama3 {
    fn push(&mut self, s: &str, out: &mut Out) {
        if self.state == LlamaState::Pass {
            out.put_text(s);
            return;
        }
        let mut buf = std::mem::take(&mut self.buf);
        buf.push_str(s);
        let n = self.run(&buf, false, out);
        buf.drain(..n);
        self.buf = buf;
    }

    fn finish(&mut self, out: &mut Out) {
        let buf = std::mem::take(&mut self.buf);
        self.run(&buf, true, out);
    }
}

#[derive(Clone, Debug, PartialEq)]
enum HarmonyBody {
    Analysis,
    Final,
    Call(RawArgs),
}

/// gpt-oss: a sequence of `<|channel|>CH [to=RECIPIENT] [<|constrain|>json]<|message|>BODY<|end|>`
/// messages. `analysis` is the reasoning trace, `final` (and recipient-less `commentary`) is the
/// answer, and a message to `functions.NAME` is a call.
struct Harmony {
    buf: String,
    head: String,
    /// `None` while reading a header.
    body: Option<HarmonyBody>,
}

const H_END: [&str; 4] = ["<|end|>", "<|call|>", "<|return|>", "<|start|>"];
const H_MSG: &str = "<|message|>";

impl Harmony {
    fn header(h: &str) -> (HarmonyBody, Option<&str>) {
        let word = |after: &str| -> Option<&str> {
            let i = h.find(after)? + after.len();
            let r = &h[i..];
            Some(&r[..r.find(|c: char| c.is_whitespace() || c == '<').unwrap_or(r.len())])
        };
        match (word("to="), word("<|channel|>")) {
            (Some(r), _) if r.starts_with("functions.") && r.len() > "functions.".len() => {
                (HarmonyBody::Call(RawArgs::default()), Some(&r["functions.".len()..]))
            }
            (None, Some("analysis")) => (HarmonyBody::Analysis, None),
            _ => (HarmonyBody::Final, None),
        }
    }

    fn body_piece(body: &mut HarmonyBody, p: &str, out: &mut Out) {
        match body {
            HarmonyBody::Analysis => out.reasoning.push_str(p),
            HarmonyBody::Final => out.put_text(p),
            HarmonyBody::Call(a) => a.piece(p, out),
        }
    }

    fn run(&mut self, s: &str, eof: bool, out: &mut Out) -> usize {
        let mut i = 0;
        while i < s.len() {
            match &mut self.body {
                None => {
                    if self.head.trim().is_empty() {
                        let t = s[i..].trim_start();
                        if !t.is_empty() && !t.starts_with("<|") && !"<|".starts_with(t) {
                            // No header: the prompt already opened a message.
                            out.put_text(&self.head);
                            self.head.clear();
                            self.body = Some(HarmonyBody::Final);
                            continue;
                        }
                    }
                    match scan(&s[i..], &[H_MSG]) {
                        Hit::At(j, _) => {
                            self.head.push_str(&s[i..i + j]);
                            i += j + H_MSG.len();
                            let (kind, name) = Self::header(&self.head);
                            if let Some(n) = name {
                                out.start(n, None);
                            }
                            self.body = Some(kind);
                            self.head.clear();
                        }
                        Hit::Need(j) => {
                            let j = if eof { s.len() - i } else { j };
                            self.head.push_str(&s[i..i + j]);
                            i += j;
                            if !eof {
                                return i;
                            }
                        }
                    }
                }
                Some(body) => match scan(&s[i..], &H_END) {
                    Hit::At(j, k) => {
                        Self::body_piece(body, &s[i..i + j], out);
                        if let HarmonyBody::Call(a) = body {
                            a.end(out);
                            out.end();
                        }
                        self.body = None;
                        i += j + H_END[k].len();
                    }
                    Hit::Need(j) => {
                        let j = if eof { s.len() - i } else { j };
                        Self::body_piece(body, &s[i..i + j], out);
                        i += j;
                        if !eof {
                            return i;
                        }
                    }
                },
            }
        }
        if eof {
            if self.body.is_none() && !self.head.trim().is_empty() {
                out.put_text(&self.head);
            }
            self.head.clear();
        }
        i
    }
}

impl Parser for Harmony {
    fn push(&mut self, s: &str, out: &mut Out) {
        let mut buf = std::mem::take(&mut self.buf);
        buf.push_str(s);
        let n = self.run(&buf, false, out);
        buf.drain(..n);
        self.buf = buf;
    }

    fn finish(&mut self, out: &mut Out) {
        let buf = std::mem::take(&mut self.buf);
        self.run(&buf, true, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaping_matches_serde() {
        for s in ["plain", "q\"b\\s", "nl\n\r\t", "\u{1}\u{1f}\u{7f}", "é日本", ""] {
            let mut got = String::new();
            escape_into(&mut got, s);
            assert_eq!(format!("\"{got}\""), serde_json::to_string(s).unwrap());
        }
    }

    #[test]
    fn json_scan_finds_value_ends() {
        let ends = |s: &str| {
            let mut j = JsonScan::default();
            j.feed(s.as_bytes())
        };
        assert_eq!(ends("{\"a\": \"}\\\"\"}, x"), (12, true));
        assert_eq!(ends("[1, [2]]x"), (8, true));
        assert_eq!(ends("\"s\\\"x\" ,"), (6, true));
        assert_eq!(ends("null}"), (4, true));
        assert_eq!(ends("{\"a\": 1"), (7, false));
    }
}

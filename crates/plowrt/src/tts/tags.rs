use std::borrow::Cow;
use std::collections::BTreeSet;

use plow_asset::speech_tags::{Asterisk, Entry, Markers, SpeechTags, Unknown};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Auto,
    Strict,
    Off,
}

pub fn bypass(text: &str) -> bool {
    !text
        .bytes()
        .any(|b| matches!(b, b'[' | b'<' | b'*' | b'\\'))
}

fn canonical(s: &str) -> Option<&'static str> {
    Some(match s {
        "laugh" | "laughs" | "laughing" | "laughter" | "laughs softly" => "laugh",
        "chuckle" | "chuckles" | "chuckling" => "chuckle",
        "giggle" | "giggles" | "giggling" => "giggle",
        "sigh" | "sighs" | "sighing" => "sigh",
        "gasp" | "gasps" => "gasp",
        "cough" | "coughs" => "cough",
        "clear_throat" | "clear throat" | "clears throat" | "clearing throat" => "clear_throat",
        "inhale" | "inhales" => "inhale",
        "exhale" | "exhales" => "exhale",
        "sniff" | "sniffle" => "sniff",
        "yawn" | "yawns" => "yawn",
        "groan" | "groans" => "groan",
        "pause" | "break" | "short pause" => "pause",
        "whisper" | "whispers" | "whispered" | "whispering" => "whisper",
        "excited" => "excited",
        "happy" | "cheerful" => "happy",
        "sad" => "sad",
        "angry" => "angry",
        "calm" => "calm",
        "serious" => "serious",
        "surprised" => "surprised",
        "warm" | "friendly" => "friendly",
        "casual" => "casual",
        "rate" => "rate",
        "pitch" => "pitch",
        "emphasis" => "emphasis",
        _ => return None,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum Span {
    Text(String),
    Tag {
        name: String,
        value: Option<String>,
        raw: String,
    },
}

#[derive(Default)]
pub struct Parser {
    pending: String,
    text: String,
    spans: Vec<Span>,
    comment: bool,
    style: Option<String>,
    comment_tail: u8,
    marker_tail: Option<bool>,
}

impl Parser {
    fn literal(&mut self, s: &str) {
        if let Some(style) = &mut self.style {
            let room = 64usize.saturating_sub(style.chars().count());
            let end = s.char_indices().nth(room).map_or(s.len(), |(i, _)| i);
            style.push_str(&s[..end]);
        } else {
            self.text.push_str(s);
        }
    }
    fn character(&mut self, c: char) {
        let mut bytes = [0; 4];
        self.literal(c.encode_utf8(&mut bytes));
    }
    fn flush(&mut self) {
        if !self.text.is_empty() {
            self.spans.push(Span::Text(std::mem::take(&mut self.text)));
        }
    }
    fn tag(&mut self, name: &str, value: Option<String>, raw: &str) {
        let name = name.trim();
        self.flush();
        self.spans.push(Span::Tag {
            name: canonical(name).unwrap_or(name).into(),
            value,
            raw: raw.into(),
        });
    }
    pub fn push(&mut self, chunk: &str, syntax: &plow_asset::speech_tags::Syntax) {
        let mut remaining = chunk;
        while !remaining.is_empty() {
            if !self.comment && self.marker_tail.is_none() && self.pending.is_empty() {
                let end = remaining
                    .bytes()
                    .position(|b| matches!(b, b'[' | b'<' | b'*' | b'\\'))
                    .unwrap_or(remaining.len());
                if end > 0 {
                    self.literal(&remaining[..end]);
                    remaining = &remaining[end..];
                    continue;
                }
            }
            let c = remaining.chars().next().unwrap();
            remaining = &remaining[c.len_utf8()..];
            if let Some(pipe) = &mut self.marker_tail {
                if *pipe && c == '>' {
                    self.marker_tail = None;
                } else {
                    *pipe = c == '|';
                }
                continue;
            }
            if self.comment {
                self.comment_tail = match (self.comment_tail, c) {
                    (0, '-') => 1,
                    (1 | 2, '-') => 2,
                    (2, '>') => {
                        self.comment = false;
                        0
                    }
                    _ => 0,
                };
                continue;
            }
            if self.pending.is_empty() {
                if c == '\\'
                    || c == '[' && syntax.bracket
                    || c == '<' && (syntax.ssml || syntax.markers == Markers::Strip)
                    || c == '*' && syntax.asterisk == Asterisk::Known
                {
                    self.pending.push(c);
                } else {
                    self.character(c);
                }
                continue;
            }
            if self.pending == "\\" {
                if !matches!(c, '[' | '<' | '\\') {
                    self.character('\\');
                }
                self.character(c);
                self.pending.clear();
                continue;
            }
            self.pending.push(c);
            if self.pending == "<!--" {
                self.pending.clear();
                self.comment = true;
                continue;
            }
            let start = self.pending.as_bytes()[0];
            if start == b'<' && self.pending.starts_with("<|") && syntax.markers == Markers::Strip {
                if self.pending.ends_with("|>") {
                    self.pending.clear();
                } else if self.pending.chars().count() > 64 {
                    self.pending.clear();
                    self.marker_tail = Some(c == '|');
                }
                continue;
            }
            if start == b'<' && self.pending == "<|" {
                self.literal("<|");
                self.pending.clear();
                continue;
            }
            if start == b'<'
                && self.pending.len() == 2
                && (!syntax.ssml || !(c.is_ascii_alphabetic() || matches!(c, '/' | '!')))
            {
                let pending = std::mem::take(&mut self.pending);
                self.literal(&pending);
                continue;
            }
            let complete = match start {
                b'[' => c == ']',
                b'*' => c == '*',
                b'<' => c == '>' && balanced_quotes(&self.pending),
                _ => false,
            };
            if complete {
                let raw = std::mem::take(&mut self.pending);
                let inner = &raw[1..raw.len() - 1];
                match start {
                    b'[' => self.bracket(inner, &raw),
                    b'*' => {
                        let n = inner.trim().to_ascii_lowercase();
                        if canonical(&n).is_some() {
                            self.tag(&n, None, &raw);
                        } else {
                            self.literal(&raw);
                        }
                    }
                    b'<' => self.element(inner, &raw),
                    _ => unreachable!(),
                }
            } else if self.pending.chars().count() > 64 || start == b'[' && c == '[' {
                // Each character enters and leaves the bounded pending buffer once.
                let pending = std::mem::take(&mut self.pending);
                self.literal(&pending);
            }
        }
    }
    fn bracket(&mut self, inner: &str, raw: &str) {
        let lower = inner.trim().to_ascii_lowercase();
        if let Some(ms) = lower
            .strip_prefix("pause ")
            .or_else(|| lower.strip_prefix("break "))
        {
            if duration(ms).is_some() {
                self.tag("pause", Some(ms.into()), raw);
                return;
            }
        }
        if lower.split(',').all(|n| canonical(n.trim()).is_some()) {
            for n in lower.split(',') {
                self.tag(n.trim(), None, raw);
            }
        } else if !lower.is_empty()
            && lower.len() <= 32
            && lower.split_ascii_whitespace().count() <= 3
            && lower
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b == b' ' || b == b'_')
        {
            self.tag(&lower, None, raw);
        } else {
            self.literal(raw);
        }
    }
    fn element(&mut self, inner: &str, raw: &str) {
        let inner = inner.trim();
        let closing = inner.starts_with('/');
        let body = inner.trim_start_matches('/');
        let name = body
            .split(|c: char| c.is_ascii_whitespace() || c == '/')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b':' | b'-' | b'_'))
        {
            self.literal(raw);
            return;
        }
        let attrs = &body[name.len()..];
        if closing {
            if name == "style" {
                if let Some(body) = self.style.take() {
                    let lower = body.trim().to_ascii_lowercase();
                    if !lower.is_empty() {
                        if lower
                            .split(|c: char| c == ',' || c.is_ascii_whitespace())
                            .filter(|s| !s.is_empty())
                            .all(|s| canonical(s).is_some())
                        {
                            for word in lower
                                .split(|c: char| c == ',' || c.is_ascii_whitespace())
                                .filter(|s| !s.is_empty())
                            {
                                self.tag(word, None, raw);
                            }
                        } else {
                            self.tag("style", Some(lower), raw);
                        }
                    }
                }
            }
            return;
        }
        match name.as_str() {
            "break" => self.tag(
                "pause",
                attr(attrs, "time").or_else(|| {
                    attr(attrs, "strength").map(|s| {
                        match s.as_str() {
                            "none" => "0ms",
                            "x-weak" => "100ms",
                            "weak" => "200ms",
                            "strong" => "600ms",
                            "x-strong" => "1000ms",
                            _ => "400ms",
                        }
                        .into()
                    })
                }),
                raw,
            ),
            "prosody" => {
                if let Some(v) = attr(attrs, "rate") {
                    self.tag("rate", Some(v), raw);
                }
                if let Some(v) = attr(attrs, "pitch") {
                    self.tag("pitch", Some(v), raw);
                }
            }
            "emphasis" => self.tag("emphasis", attr(attrs, "level"), raw),
            "style" if attr(attrs, "name").is_none() && attr(attrs, "style").is_none() => {
                self.flush();
                self.style = Some(String::new());
            }
            "express-as" | "emotion" | "style" | "amazon:effect" => {
                if let Some(v) = attr(attrs, "style").or_else(|| attr(attrs, "name")) {
                    self.tag(&v.to_ascii_lowercase(), None, raw);
                }
            }
            "speak" => {}
            _ => {
                if canonical(&name).is_some() {
                    self.tag(&name, None, raw);
                }
            }
        }
    }
    pub fn finish(mut self) -> Vec<Span> {
        if let Some(style) = self.style.take() {
            self.text.push_str(&style);
        }
        self.text.push_str(&self.pending);
        self.flush();
        self.spans
    }
}

fn balanced_quotes(s: &str) -> bool {
    let mut quote = None;
    for c in s.chars() {
        if quote == Some(c) {
            quote = None;
        } else if quote.is_none() && matches!(c, '\'' | '"') {
            quote = Some(c);
        }
    }
    quote.is_none()
}

fn attr(mut s: &str, key: &str) -> Option<String> {
    while !s.is_empty() {
        s = s.trim_start();
        let n = s
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, ':' | '-' | '_')))
            .unwrap_or(s.len());
        if n == 0 {
            return None;
        }
        let name = &s[..n];
        s = s[n..].trim_start();
        s = s.strip_prefix('=')?.trim_start();
        let quote = s.chars().next()?;
        if !matches!(quote, '\'' | '"') {
            return None;
        }
        s = &s[1..];
        let end = s.find(quote)?;
        let value = &s[..end];
        s = &s[end + 1..];
        if name.eq_ignore_ascii_case(key) {
            return Some(value.into());
        }
    }
    None
}

fn duration(s: &str) -> Option<u32> {
    let s = s.trim();
    let (value, mul) = if let Some(v) = s.strip_suffix("ms") {
        (v, 1.0)
    } else if let Some(v) = s.strip_suffix('s') {
        (v, 1000.0)
    } else {
        return None;
    };
    let v: f64 = value.parse().ok()?;
    if !v.is_finite() || v < 0.0 {
        return None;
    }
    Some((v * mul).min(3000.0).round() as u32)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub active: bool,
    pub applied: Vec<String>,
    pub dropped: Vec<String>,
}
impl Report {
    fn add(out: &mut Vec<String>, name: &str) {
        let name = if !name.is_empty()
            && name.len() <= 32
            && name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || matches!(b, b' ' | b'_'))
        {
            name
        } else {
            "unknown"
        };
        if out.len() < 32 && !out.iter().any(|n| n == name) {
            out.push(name.into());
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    Text(String),
    Native { text: String, carry: bool },
    Silence(u32),
}

pub struct Rendered<'a> {
    pub input: Cow<'a, str>,
    pub parts: Vec<Part>,
    pub instructions: Option<String>,
    pub speed: f32,
    pub report: Report,
}

fn resolve<'a>(map: &'a SpeechTags, name: &str, lang: Option<&str>) -> Option<&'a Entry> {
    let mut e = map.tags.get(name)?;
    loop {
        if !e.langs.is_empty() && !lang.is_some_and(|l| e.langs.iter().any(|v| v == l)) {
            return None;
        }
        match e.alias.as_deref() {
            Some(n) => e = map.tags.get(n)?,
            None => return Some(e),
        }
    }
}

pub fn render<'a>(
    map: Option<&SpeechTags>,
    text: &'a str,
    mode: Mode,
    lang: Option<&str>,
    instructions: Option<&str>,
    speed: f32,
) -> Result<Rendered<'a>, String> {
    let mut out = Rendered {
        input: Cow::Borrowed(text),
        parts: Vec::new(),
        instructions: instructions.map(str::to_owned),
        speed,
        report: Report::default(),
    };
    let Some(map) = map.filter(|_| mode != Mode::Off && !bypass(text)) else {
        return Ok(out);
    };
    out.report.active = true;
    let mut parser = Parser {
        text: String::with_capacity(text.len()),
        pending: String::with_capacity(256),
        ..Parser::default()
    };
    parser.push(text, &map.syntax);
    let mut current = String::with_capacity(text.len());
    let mut added = BTreeSet::new();
    for span in parser.finish() {
        match span {
            Span::Text(t) => {
                let mut t = t;
                for (from, to) in &map.literal_rewrite {
                    t = t.replace(from, to);
                }
                current.push_str(&t.replace("<|", "< |"));
            }
            Span::Tag { name, value, raw } => {
                let e = resolve(map, &name, lang);
                let supported = e.is_some_and(|e| {
                    e.text.is_some()
                        || e.instruction.is_some()
                        || e.silence_ms.is_some()
                        || e.speed == Some(true)
                });
                let mut applied = supported;
                if mode == Mode::Strict && !supported {
                    return Err(format!("unsupported tag {name:?}"));
                }
                if let Some(e) = e {
                    if e.reject == Some(true) {
                        return Err(format!("rejected tag {name:?}"));
                    }
                    if let Some(t) = &e.text {
                        if !current.is_empty() {
                            out.parts.push(Part::Text(
                                std::mem::take(&mut current).replace("<|", "< |"),
                            ));
                        }
                        out.parts.push(Part::Native {
                            text: t.clone(),
                            carry: e.carry,
                        });
                    }
                    if let Some(i) = &e.instruction {
                        if added.insert(i.as_str()) {
                            let dst = out.instructions.get_or_insert_with(String::new);
                            if !dst.is_empty() {
                                dst.push(' ');
                            }
                            dst.push_str(i);
                            if dst.chars().count() > super::MAX_INPUT_CHARS {
                                return Err("tag instructions exceed input limit".into());
                            }
                        }
                    }
                    if let Some(s) = &e.silence_ms {
                        let ms = match value.as_deref() {
                            Some(v) => {
                                duration(&v).ok_or_else(|| "invalid break duration".to_string())?
                            }
                            None => s.default,
                        }
                        .min(s.max);
                        if !current.is_empty() {
                            out.parts.push(Part::Text(
                                std::mem::take(&mut current).replace("<|", "< |"),
                            ));
                        }
                        out.parts.push(Part::Silence(ms));
                    }
                    if e.speed == Some(true) {
                        let factor = match value.as_deref() {
                            Some("x-slow") => 0.5,
                            Some("slow") => 0.75,
                            Some("fast") => 1.25,
                            Some("x-fast") => 1.5,
                            Some("medium" | "default") | None => 1.0,
                            Some(v) => v
                                .strip_suffix('%')
                                .and_then(|s| s.parse::<f32>().ok())
                                .map(|n| n / 100.0)
                                .ok_or_else(|| "invalid prosody rate".to_string())?,
                        };
                        let speed = out.speed * factor;
                        if !(super::stretch::MIN_SPEED..=super::stretch::MAX_SPEED).contains(&speed)
                        {
                            return Err("tag rate exceeds speed limits".into());
                        }
                        out.speed = speed;
                    }
                } else {
                    match map.unknown {
                        Unknown::Reject => return Err(format!("unknown tag {name:?}")),
                        Unknown::Literal => {
                            let mut raw = raw;
                            for (f, t) in &map.literal_rewrite {
                                raw = raw.replace(f, t);
                            }
                            current.push_str(&raw);
                        }
                        Unknown::Passthrough => {
                            if !name.is_empty()
                                && name.len() <= 32
                                && name
                                    .bytes()
                                    .all(|b| b.is_ascii_lowercase() || matches!(b, b' ' | b'_'))
                            {
                                if !current.is_empty() {
                                    out.parts.push(Part::Text(
                                        std::mem::take(&mut current).replace("<|", "< |"),
                                    ));
                                }
                                out.parts.push(Part::Native {
                                    text: format!("[{name}]"),
                                    carry: false,
                                });
                                applied = true;
                            }
                        }
                        Unknown::Strip => {}
                    }
                }
                Report::add(
                    if applied {
                        &mut out.report.applied
                    } else {
                        &mut out.report.dropped
                    },
                    &name,
                );
            }
        }
    }
    if !current.is_empty() {
        out.parts.push(Part::Text(current.replace("<|", "< |")));
    }
    out.input = Cow::Owned(
        out.parts
            .iter()
            .filter_map(|p| match p {
                Part::Text(t) | Part::Native { text: t, .. } => Some(t.as_str()),
                _ => None,
            })
            .collect(),
    );
    if out.report.applied.is_empty() && out.report.dropped.is_empty() && out.input == text {
        out.input = Cow::Borrowed(text);
        out.parts.clear();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn map() -> SpeechTags {
        SpeechTags::parse(r#"{"syntax":{"bracket":true,"ssml":true,"asterisk":"known"},"unknown":"strip","tags":{"pause":{"silence_ms":{"default":400,"max":3000}},"laugh":{"drop":true},"calm":{"instruction":"Speak calmly."}}}"#).unwrap()
    }
    #[test]
    fn parser_shapes_and_unicode() {
        let m = map();
        for (input, expected) in [
            ("[laughs] Hello.", " Hello."),
            ("*laughs* नमस्ते।", " नमस्ते।"),
            ("[warm, casual, friendly] Hello", " Hello"),
            ("[conversational] Hello", " Hello"),
            ("<speak><em>Hello</em><!-- note --></speak>", "Hello"),
            ("<express-as style=\"excited\">Hello</express-as>", "Hello"),
            ("<emotion name='sad'>Hello</emotion>", "Hello"),
            (
                "<amazon:effect name='whispered'>Hello</amazon:effect>",
                "Hello",
            ),
            ("<prosody rate='slow'>Hello</prosody>", "Hello"),
            ("<style>casual warm friendly</style>Hello", "Hello"),
            (
                "(laughs) {laughs} *ordinary*",
                "(laughs) {laughs} *ordinary*",
            ),
            ("[1] [कोई] [see p. 4]", "[1] [कोई] [see p. 4]"),
            ("a < b", "a < b"),
            (r"\[laughs] \<laugh>", "[laughs] <laugh>"),
            ("<|style|> Hello", "< |style|> Hello"),
            ("[laughs", "[laughs"),
        ] {
            assert_eq!(
                render(Some(&m), input, Mode::Auto, None, None, 1.0)
                    .unwrap()
                    .input,
                expected,
                "{input}"
            );
        }
    }
    #[test]
    fn every_push_boundary_matches() {
        let m = map();
        let input = "नमस्ते [laughs] <break time='600ms'/> <!-- long comment --> *sighs*";
        let mut p = Parser::default();
        p.push(input, &m.syntax);
        let expected = p.finish();
        for (i, _) in input.char_indices() {
            let mut p = Parser::default();
            p.push(&input[..i], &m.syntax);
            p.push(&input[i..], &m.syntax);
            assert_eq!(p.finish(), expected, "{i}");
        }
    }

    #[test]
    fn fixture_syntax_and_literal_preservation() {
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../../../scripts/tts/tag_cases.json")).unwrap();
        let map = SpeechTags::strip();
        let compact = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(cases.len(), 74);
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let shape = case["id"].as_str().unwrap().split_once('-').unwrap().1;
            let lang = case["language"].as_str();
            let auto = render(Some(&map), text, Mode::Auto, lang, None, 1.0).unwrap();
            if matches!(shape, "paren" | "curly" | "plain_prefix") {
                let off = render(Some(&map), text, Mode::Off, lang, None, 1.0).unwrap();
                assert_eq!(auto.input.as_bytes(), off.input.as_bytes(), "{shape}");
                assert!(matches!(auto.input, Cow::Borrowed(_)), "{shape}");
                for max in [8, 24, 128] {
                    assert_eq!(
                        auto.segments(max, str::to_owned),
                        off.segments(max, str::to_owned),
                        "{shape}"
                    );
                }
            } else {
                assert_eq!(
                    compact(&auto.input),
                    compact(case["reference"].as_str().unwrap()),
                    "{shape}"
                );
            }
        }
        for text in ["*ordinary words* Hello.", "*साधारण* नमस्ते।"] {
            let auto = render(Some(&map), text, Mode::Auto, None, None, 1.0).unwrap();
            assert_eq!(auto.input, text);
            assert!(matches!(auto.input, Cow::Borrowed(_)));
        }
    }

    #[test]
    fn marker_stripping_is_controlled_by_literal_rewrites() {
        let mut map = SpeechTags::strip();
        map.syntax.markers = Markers::Neutralize;
        map.literal_rewrite = vec![
            ("<|style|>".into(), String::new()),
            ("<|/style|>".into(), String::new()),
        ];
        map.validate().unwrap();
        let text = "<|style|>friendly<|/style|> Hello <|other|>";
        let auto = render(Some(&map), text, Mode::Auto, None, None, 1.0).unwrap();
        assert_eq!(auto.input, "friendly Hello < |other|>");
        assert_eq!(
            render(Some(&map), text, Mode::Off, None, None, 1.0)
                .unwrap()
                .input,
            text
        );
    }

    #[test]
    fn strip_markers_preserves_inner_text_at_every_push_boundary() {
        let mut map = SpeechTags::strip();
        for ssml in [false, true] {
            map.syntax.ssml = ssml;
            for text in [
                "<|style|>Hello नमस्ते।<|/style|>",
                "<|language:hi|>Hello नमस्ते।",
            ] {
                for (i, _) in text.char_indices() {
                    let mut parser = Parser::default();
                    parser.push(&text[..i], &map.syntax);
                    parser.push(&text[i..], &map.syntax);
                    assert_eq!(parser.finish(), vec![Span::Text("Hello नमस्ते।".into())]);
                }
                assert_eq!(
                    render(Some(&map), text, Mode::Auto, None, None, 1.0)
                        .unwrap()
                        .input,
                    "Hello नमस्ते।"
                );
            }
        }
        let mut parser = Parser::default();
        parser.push("<|", &map.syntax);
        for _ in 0..10000 {
            parser.push("x", &map.syntax);
            assert!(parser.pending.len() <= 64);
        }
        parser.push("|>Hello", &map.syntax);
        assert_eq!(parser.finish(), vec![Span::Text("Hello".into())]);
    }

    #[cfg(feature = "hf-tokenizer")]
    #[test]
    fn neutralized_markers_cannot_encode_special_tokens() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[UNK]".into(), 0)].into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        let markers = ["<|style|>", "<|/style|>", "<|event|>"];
        tokenizer
            .add_special_tokens(markers.map(|text| tokenizers::AddedToken::from(text, true)))
            .unwrap();
        let map = SpeechTags::strip();
        for text in markers {
            let id = tokenizer.token_to_id(text).unwrap();
            assert_eq!(tokenizer.encode(text, false).unwrap().get_ids(), &[id]);
            let rendered = render(Some(&map), text, Mode::Auto, None, None, 1.0).unwrap();
            assert!(!tokenizer
                .encode(rendered.input.as_ref(), false)
                .unwrap()
                .get_ids()
                .contains(&id));
        }
    }
    #[test]
    fn silence_instructions_strict_and_bypass() {
        let m = map();
        let r = render(
            Some(&m),
            "Hi[pause 600ms]there<pause/>",
            Mode::Auto,
            None,
            None,
            1.0,
        )
        .unwrap();
        assert_eq!(
            r.parts,
            vec![
                Part::Text("Hi".into()),
                Part::Silence(600),
                Part::Text("there".into()),
                Part::Silence(400)
            ]
        );
        let r = render(
            Some(&m),
            "[calm][calm]Hi",
            Mode::Auto,
            None,
            Some("Clear."),
            1.0,
        )
        .unwrap();
        assert_eq!(r.instructions.as_deref(), Some("Clear. Speak calmly."));
        assert!(render(Some(&m), "[laughs]Hi", Mode::Strict, None, None, 1.0).is_err());
        for (text, map, mode) in [
            ("Hello नमस्ते", Some(&m), Mode::Auto),
            ("[laughs]", None, Mode::Auto),
            ("[laughs]", Some(&m), Mode::Off),
        ] {
            let r = render(map, text, mode, None, None, 1.0).unwrap();
            assert!(matches!(r.input, Cow::Borrowed(_)));
            assert_eq!(r.input.as_ptr(), text.as_ptr());
            assert!(r.parts.is_empty());
        }
    }
    #[test]
    fn bypass_preserves_existing_segmentation_even_empty_normalized_text() {
        let m = map();
        for text in [
            "  Hello world.  ",
            "नमस्ते। फिर मिलेंगे।",
            "(laughs) {ordinary}",
            "  *ordinary*  ",
            "a < b",
        ] {
            let r = render(Some(&m), text, Mode::Auto, None, None, 1.0).unwrap();
            assert!(matches!(r.input, Cow::Borrowed(_)), "{text}");
            for max in [1, 5, 100] {
                let (parts, tail) = r.segments(max, str::to_owned);
                assert_eq!(tail, None);
                let expected: Vec<_> = super::super::segments(text, max)
                    .into_iter()
                    .map(str::to_owned)
                    .collect();
                assert_eq!(
                    parts.into_iter().map(|s| s.text).collect::<Vec<_>>(),
                    expected
                );
            }
            let (parts, tail) = r.segments(10, |_| String::new());
            assert_eq!(parts.len(), 1);
            assert_eq!(parts[0].text, "");
            assert_eq!(tail, None);
        }
    }

    #[test]
    fn actions_aliases_language_and_safe_literal_output() {
        let mut m = map();
        m.tags.insert(
            "sigh".into(),
            serde_json::from_str(r#"{"text":"<sigh>","langs":["en"]}"#).unwrap(),
        );
        m.tags.insert(
            "giggle".into(),
            serde_json::from_str(r#"{"alias":"sigh"}"#).unwrap(),
        );
        m.tags.insert(
            "rate".into(),
            serde_json::from_str(r#"{"speed":true}"#).unwrap(),
        );
        m.tags.insert(
            "cough".into(),
            serde_json::from_str(r#"{"reject":true}"#).unwrap(),
        );
        m.validate_binding(true, |_| true).unwrap();
        assert_eq!(
            render(Some(&m), "[giggles]Hi", Mode::Strict, Some("en"), None, 1.0)
                .unwrap()
                .input,
            "<sigh>Hi"
        );
        assert_eq!(
            render(Some(&m), "[giggles]नमस्ते", Mode::Auto, Some("hi"), None, 1.0)
                .unwrap()
                .input,
            "नमस्ते"
        );
        assert!(render(Some(&m), "[cough]Hi", Mode::Auto, None, None, 1.0).is_err());
        assert_eq!(
            render(
                Some(&m),
                "<prosody rate='fast'>Hi</prosody>",
                Mode::Auto,
                None,
                None,
                1.0
            )
            .unwrap()
            .speed,
            1.25
        );
        assert!(render(
            Some(&m),
            "<prosody rate='NaN%'>Hi</prosody>",
            Mode::Auto,
            None,
            None,
            1.0
        )
        .is_err());
        m.literal_rewrite = vec![("[".into(), "(".into()), ("]".into(), ")".into())];
        let r = render(
            Some(&m),
            r"\[hi] <|event|> [1] [कोई]",
            Mode::Auto,
            None,
            None,
            1.0,
        )
        .unwrap();
        assert_eq!(r.input, "(hi) < |event|> (1) (कोई)");
        for unknown in [Unknown::Literal, Unknown::Passthrough] {
            m.unknown = unknown;
            assert!(!render(
                Some(&m),
                "<emotion name='&lt;|x|>'>Hi</emotion>",
                Mode::Auto,
                None,
                None,
                1.0
            )
            .unwrap()
            .input
            .contains("<|"));
        }
        let r = render(
            Some(&m),
            r"\<[laughs]|x|>[pause]Hi",
            Mode::Auto,
            None,
            None,
            1.0,
        )
        .unwrap();
        for p in r.parts {
            if let Part::Text(t) = p {
                assert!(!t.contains("<|"), "{t}");
            }
        }
        m.unknown = Unknown::Passthrough;
        let forwarded =
            render(Some(&m), "[conversational]Hi", Mode::Auto, None, None, 1.0).unwrap();
        assert_eq!(forwarded.input, "[conversational]Hi");
        assert_eq!(forwarded.report.applied, vec!["conversational"]);
        assert!(forwarded.report.dropped.is_empty());
        m.unknown = Unknown::Reject;
        assert!(render(Some(&m), "[unlisted]Hi", Mode::Auto, None, None, 1.0).is_err());
    }

    #[test]
    fn native_tags_remain_atomic_and_carry_across_pauses() {
        let mut m = map();
        m.tags.insert(
            "whisper".into(),
            serde_json::from_str(r#"{"text":"[whispers]","carry":true}"#).unwrap(),
        );
        let r = render(
            Some(&m),
            "[whispers]One two three[pause]Four five six",
            Mode::Auto,
            None,
            None,
            1.0,
        )
        .unwrap();
        let (pieces, tail) = r.segments(12, str::to_owned);
        assert_eq!(tail, None);
        assert!(pieces.len() > 2);
        assert_eq!(
            pieces
                .iter()
                .filter(|s| s.pause_before == Some(400))
                .count(),
            1
        );
        for p in &pieces {
            assert!(p.text.starts_with("[whispers]"), "{p:?}");
        }
        let native = [Native {
            range: 0..10,
            carry: true,
        }];
        assert_eq!(segment("[whispers]Hi", 3, &native)[0].text, "[whispers]");
        assert_eq!(safe_cut("[whispers]Hi", 4, &native), Some(10));
        assert_eq!(safe_cut("[whispers]", 4, &native), None);
        let literal = render(
            Some(&m),
            r"\[whispers]One two three four",
            Mode::Auto,
            None,
            None,
            1.0,
        )
        .unwrap();
        let (parts, _) = literal.segments(12, str::to_owned);
        assert!(parts.iter().all(|p| p.native.is_empty()));
        assert_eq!(
            parts
                .iter()
                .filter(|p| p.text.starts_with("[whispers]"))
                .count(),
            1
        );
    }

    #[test]
    fn fitting_splits_preserve_carry_and_refuse_no_progress() {
        let t = TextSegment {
            text: "[whispers]One two".into(),
            native: vec![Native {
                range: 0..10,
                carry: true,
            }],
            pause_before: None,
        };
        assert!(split_at(&t, 10).is_none());
        let [left, right] = split_at(&t, 13).unwrap();
        assert_eq!(left.text, "[whispers]One");
        assert_eq!(right.text, "[whispers]two");
        assert_eq!(
            right.native,
            vec![Native {
                range: 0..10,
                carry: true
            }]
        );
        let mut m = map();
        m.tags.insert(
            "whisper".into(),
            serde_json::from_str(r#"{"text":"[whispers]","carry":true}"#).unwrap(),
        );
        let r = render(
            Some(&m),
            r"\[whispers]One two three four [whispers]five six seven",
            Mode::Auto,
            None,
            None,
            1.0,
        )
        .unwrap();
        let (parts, _) = r.segments(16, str::to_owned);
        let first_native = parts.iter().position(|p| !p.native.is_empty()).unwrap();
        assert!(first_native > 0);
        assert!(parts[..first_native].iter().all(|p| p.native.is_empty()));
        for p in parts {
            for n in &p.native {
                assert_eq!(&p.text[n.range.clone()], "[whispers]");
            }
        }
    }

    #[test]
    fn normalization_cannot_restore_special_token_openers() {
        let m = map();
        let r = render(Some(&m), "<\u{200b}|event|>", Mode::Auto, None, None, 1.0).unwrap();
        let (parts, _) = r.segments(100, |t| t.replace('\u{200b}', ""));
        assert_eq!(parts[0].text, "< |event|>");
    }

    #[test]
    fn packet_tag_version_is_required_and_fail_closed() {
        let mut p = plow_asset::packet_pipeline::PacketPipeline {
            name: "speech".into(),
            driver: "tts.codec_lm.v1".into(),
            programs: Default::default(),
            tensors: Default::default(),
            parameters: Default::default(),
            strings: Default::default(),
        };
        assert!(load(&p).unwrap().is_none());
        p.strings
            .insert("text.tags".into(), serde_json::to_string(&map()).unwrap());
        assert!(load(&p).is_err());
        p.parameters.insert("text.tags_version".into(), 1);
        assert!(load(&p).unwrap().is_some());
        p.parameters.insert("text.tags_version".into(), 2);
        assert!(load(&p).is_err());
        p.strings.remove("text.tags");
        assert!(load(&p).is_err());
        assert!(serde_json::from_str::<Mode>(r#""anything""#).is_err());
    }

    #[test]
    fn fallback_is_opt_in_and_never_masks_packet_metadata() {
        let mut p = plow_asset::packet_pipeline::PacketPipeline {
            name: "speech".into(),
            driver: "tts.codec_lm.v1".into(),
            programs: Default::default(),
            tensors: Default::default(),
            parameters: Default::default(),
            strings: Default::default(),
        };
        assert!(load_with_default(&p, false).unwrap().is_none());
        let a = load_with_default(&p, true).unwrap().unwrap();
        let b = load_with_default(&p, true).unwrap().unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b));
        assert_eq!(*a, SpeechTags::strip());
        assert!(matches!(
            render(Some(&a), "[laughs]Hello", Mode::Off, None, None, 1.0)
                .unwrap()
                .input,
            Cow::Borrowed(_)
        ));
        let mut explicit = SpeechTags::strip();
        explicit.unknown = Unknown::Reject;
        p.strings.insert(
            "text.tags".into(),
            serde_json::to_string(&explicit).unwrap(),
        );
        assert!(load_with_default(&p, true).is_err());
        p.parameters.insert("text.tags_version".into(), 1);
        assert_eq!(
            load_with_default(&p, true).unwrap().unwrap().unknown,
            Unknown::Reject
        );
        p.parameters.insert("text.tags_version".into(), 2);
        assert!(load_with_default(&p, true).is_err());
    }

    #[test]
    fn malformed_unicode_and_delimiters_are_chunk_independent() {
        let m = map();
        let alphabet = [
            '[', ']', '<', '>', '|', '*', '\\', ' ', '\'', '"', 'a', 'b', '!', '-', 'ह', 'ि', '🙂',
        ];
        let mut state = 19u64;
        for _ in 0..500 {
            let mut input = String::new();
            for _ in 0..100 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                input.push(alphabet[(state >> 32) as usize % alphabet.len()]);
            }
            let mut full = Parser::default();
            full.push(&input, &m.syntax);
            let expected = full.finish();
            let mut split = Parser::default();
            for c in input.chars() {
                let mut bytes = [0; 4];
                split.push(c.encode_utf8(&mut bytes), &m.syntax);
                assert!(split.pending.chars().count() <= 64);
                assert!(split.style.as_ref().is_none_or(|s| s.chars().count() <= 64));
            }
            assert_eq!(split.finish(), expected);
            let r = render(Some(&m), &input, Mode::Auto, None, None, 1.0).unwrap();
            assert!(!r.input.contains("<|"));
        }
    }

    #[test]
    fn bounded_pathological_input() {
        let m = map();
        let mut p = Parser::default();
        for _ in 0..10000 {
            p.push("[", &m.syntax);
            assert!(p.pending.len() <= 256);
        }
        assert!(!p.finish().is_empty());
    }
}

pub fn load(
    p: &plow_asset::packet_pipeline::PacketPipeline,
) -> crate::Result<Option<std::sync::Arc<SpeechTags>>> {
    load_with_default(
        p,
        crate::config::RuntimeConfig::get()
            .tts_tags_default
            .as_deref()
            == Some("strip"),
    )
}

fn load_with_default(
    p: &plow_asset::packet_pipeline::PacketPipeline,
    strip: bool,
) -> crate::Result<Option<std::sync::Arc<SpeechTags>>> {
    use plow_asset::speech_tags::{KEY, VERSION, VERSION_KEY};
    match (p.strings.get(KEY), p.parameters.get(VERSION_KEY)) {
        (None, None) if strip => {
            static MAP: std::sync::OnceLock<std::sync::Arc<SpeechTags>> =
                std::sync::OnceLock::new();
            Ok(Some(std::sync::Arc::clone(
                MAP.get_or_init(|| std::sync::Arc::new(SpeechTags::strip())),
            )))
        }
        (None, None) => Ok(None),
        (Some(json), Some(&VERSION)) => SpeechTags::parse(json)
            .map(|map| Some(std::sync::Arc::new(map)))
            .map_err(crate::RuntimeError::Rejected),
        _ => Err(crate::RuntimeError::Rejected(
            "unsupported or incomplete text.tags version".into(),
        )),
    }
}

pub fn card(map: &SpeechTags) -> serde_json::Value {
    let supported: serde_json::Map<String, serde_json::Value> = map
        .tags
        .keys()
        .map(|name| {
            let mut e = &map.tags[name];
            while let Some(a) = &e.alias {
                e = &map.tags[a];
            }
            let kind = if e.text.is_some() {
                "native"
            } else if e.instruction.is_some() {
                "instruction"
            } else if e.silence_ms.is_some() {
                "silence"
            } else if e.speed == Some(true) {
                "speed"
            } else {
                "dropped"
            };
            (name.clone(), kind.into())
        })
        .collect();
    serde_json::json!({"version":plow_asset::speech_tags::VERSION,"syntax":map.syntax,"unknown":map.unknown,"supported":supported})
}

#[derive(Clone, Debug, PartialEq)]
pub struct Native {
    pub range: std::ops::Range<usize>,
    pub carry: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextSegment {
    pub text: String,
    pub native: Vec<Native>,
    pub pause_before: Option<u32>,
}

pub fn segment(text: &str, max: usize, native: &[Native]) -> Vec<TextSegment> {
    let pieces = super::segments(text, max);
    if native.is_empty() {
        return pieces
            .into_iter()
            .map(|text| TextSegment {
                text: text.into(),
                native: Vec::new(),
                pause_before: None,
            })
            .collect();
    }
    let boundaries: Vec<_> = pieces
        .iter()
        .map(|p| p.as_ptr() as usize - text.as_ptr() as usize + p.len())
        .collect();
    let mut boundary = 0;
    let mut out = Vec::new();
    let mut start = 0;
    let mut cursor = 0;
    let mut active: Vec<&str> = Vec::new();
    while start < text.len() {
        while boundary < boundaries.len() && boundaries[boundary] <= start {
            boundary += 1;
        }
        let mut end = boundaries.get(boundary).copied().unwrap_or(text.len());
        let reserve: usize = active.iter().map(|t| t.chars().count()).sum();
        let budget = max.saturating_sub(reserve).max(1);
        let cap = text[start..]
            .char_indices()
            .nth(budget)
            .map_or(text.len(), |(i, _)| start + i);
        end = end.min(cap);
        let from = cursor;
        while cursor < native.len() && native[cursor].range.start < end {
            let n = &native[cursor];
            if n.range.end > end {
                if n.range.start > start {
                    end = n.range.start;
                    break;
                }
                end = n.range.end;
            }
            cursor += 1;
        }
        let current = &native[from..cursor];
        let range = trim_range(text, start..end, current);
        let begin = range.start;
        let body = &text[range];
        let mut text_out = String::new();
        let mut spans = Vec::new();
        for t in &active {
            if !current
                .iter()
                .any(|n| n.range.start == begin && &text[n.range.clone()] == *t)
            {
                let begin = text_out.len();
                text_out.push_str(t);
                spans.push(Native {
                    range: begin..text_out.len(),
                    carry: true,
                });
            }
        }
        let prefix = text_out.len();
        text_out.push_str(body);
        for n in current {
            spans.push(Native {
                range: prefix + n.range.start - begin..prefix + n.range.end - begin,
                carry: n.carry,
            });
            let t = &text[n.range.clone()];
            if n.carry && !active.contains(&t) {
                active.push(t);
            }
        }
        if !body.is_empty() {
            out.push(TextSegment {
                text: text_out,
                native: spans,
                pause_before: None,
            });
        }
        start = end;
    }
    out
}

fn trim_range(
    text: &str,
    range: std::ops::Range<usize>,
    native: &[Native],
) -> std::ops::Range<usize> {
    let raw = &text[range.clone()];
    let mut begin = range.start + raw.len() - raw.trim_start().len();
    let mut end = range.start + raw.trim_end().len();
    for n in native
        .iter()
        .filter(|n| n.range.start >= range.start && n.range.end <= range.end)
    {
        begin = begin.min(n.range.start);
        end = end.max(n.range.end);
    }
    begin..end.max(begin)
}

fn normalize_piece(
    text: &str,
    preserve: bool,
    normalize: &mut impl FnMut(&str) -> String,
) -> String {
    let normalized = normalize(text);
    if !preserve {
        return normalized;
    }
    if text.trim().is_empty() {
        return text.to_owned();
    }
    let mut out = String::new();
    if !normalized.starts_with(char::is_whitespace) {
        out.push_str(&text[..text.len() - text.trim_start().len()]);
    }
    out.push_str(&normalized);
    if !normalized.ends_with(char::is_whitespace) {
        out.push_str(&text[text.trim_end().len()..]);
    }
    out
}

fn neutralize(text: &mut String, native: &mut [Native]) {
    let positions: Vec<_> = text.match_indices("<|").map(|(at, _)| at + 1).collect();
    if positions.is_empty() {
        return;
    }
    let (mut before_start, mut before_end) = (0, 0);
    for n in native {
        while before_start < positions.len() && positions[before_start] <= n.range.start {
            before_start += 1;
        }
        while before_end < positions.len() && positions[before_end] < n.range.end {
            before_end += 1;
        }
        n.range.start += before_start;
        n.range.end += before_end;
    }
    *text = text.replace("<|", "< |");
}

impl Rendered<'_> {
    pub fn segments(
        &self,
        max: usize,
        normalize: impl FnMut(&str) -> String,
    ) -> (Vec<TextSegment>, Option<u32>) {
        let mut normalize = normalize;
        if self.parts.is_empty() {
            let mut text = normalize(&self.input);
            if self.report.active {
                text = text.replace("<|", "< |");
            }
            if matches!(self.input, Cow::Borrowed(_)) {
                return (segment(&text, max, &[]), None);
            }
            return (Vec::new(), None);
        }
        let mut out = Vec::new();
        let mut pause: Option<u32> = None;
        let mut carried: Vec<String> = Vec::new();
        let mut flush = |parts: &[Part], pause: &mut Option<u32>| {
            if parts.is_empty() {
                return;
            }
            let preserve =
                !carried.is_empty() || parts.iter().any(|p| matches!(p, Part::Native { .. }));
            let mut text = String::new();
            let mut native = Vec::new();
            for t in &carried {
                if !matches!(parts.first(),Some(Part::Native {text,..}) if text==t) {
                    let start = text.len();
                    text.push_str(t);
                    native.push(Native {
                        range: start..text.len(),
                        carry: true,
                    });
                }
            }
            for part in parts {
                match part {
                    Part::Text(t) => text.push_str(&normalize_piece(t, preserve, &mut normalize)),
                    Part::Native { text: t, carry } => {
                        let start = text.len();
                        text.push_str(t);
                        native.push(Native {
                            range: start..text.len(),
                            carry: *carry,
                        });
                        if *carry && !carried.contains(t) {
                            carried.push(t.clone());
                        }
                    }
                    Part::Silence(_) => unreachable!(),
                }
            }
            neutralize(&mut text, &mut native);
            for mut piece in segment(&text, max, &native) {
                if !piece.text.trim().is_empty() {
                    piece.pause_before = pause.take();
                    out.push(piece);
                }
            }
        };
        let mut begin = 0;
        for (i, part) in self.parts.iter().enumerate() {
            if let Part::Silence(ms) = part {
                flush(&self.parts[begin..i], &mut pause);
                pause = Some(pause.unwrap_or(0).saturating_add(*ms));
                begin = i + 1;
            }
        }
        flush(&self.parts[begin..], &mut pause);
        (out, pause)
    }
}

pub fn safe_cut(text: &str, at: usize, native: &[Native]) -> Option<usize> {
    let at = native
        .iter()
        .find(|n| n.range.start < at && at < n.range.end)
        .map_or(at, |n| {
            if n.range.start > 0 {
                n.range.start
            } else {
                n.range.end
            }
        });
    (at > 0 && at < text.len()).then_some(at)
}

pub fn split_at(text: &TextSegment, at: usize) -> Option<[TextSegment; 2]> {
    let make = |range: std::ops::Range<usize>| {
        let range = trim_range(&text.text, range, &text.native);
        let begin = range.start;
        let end = range.end;
        TextSegment {
            text: text.text[range].to_owned(),
            pause_before: None,
            native: text
                .native
                .iter()
                .filter(|n| n.range.start >= begin && n.range.end <= end)
                .map(|n| Native {
                    range: n.range.start - begin..n.range.end - begin,
                    carry: n.carry,
                })
                .collect(),
        }
    };
    let left = make(0..at);
    let mut right = make(at..text.text.len());
    if !right.text.is_empty() {
        let mut prefix = String::new();
        let mut spans = Vec::new();
        let mut carried = Vec::new();
        for n in left.native.iter().filter(|n| n.carry) {
            let t = &left.text[n.range.clone()];
            if !right
                .native
                .iter()
                .any(|n| n.range.start == 0 && &right.text[n.range.clone()] == t)
                && !carried.contains(&t)
            {
                let start = prefix.len();
                prefix.push_str(t);
                spans.push(Native {
                    range: start..prefix.len(),
                    carry: true,
                });
                carried.push(t);
            }
        }
        for n in &mut right.native {
            n.range.start += prefix.len();
            n.range.end += prefix.len();
        }
        spans.append(&mut right.native);
        prefix.push_str(&right.text);
        right.text = prefix;
        right.native = spans;
    }
    (left.text.len() < text.text.len() && right.text.len() < text.text.len())
        .then_some([left, right])
}

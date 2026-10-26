//! Voice reply guard (opt-in per model, `"reply_guard": "voice"` in `--chat-defaults`): moves
//! planning lines a reasoning model writes after its trace (`Wait, the prompt says…`,
//! `*Self-correction:*`, `Who is calling: …`, `Tamil detected.`) from `content` to `reasoning`,
//! and answers a turn left with no spoken text with a fallback line in the caller's script.
//!
//! Line by line, on the answer text both response paths produce. A line is held only while its
//! prefix could still be planning and released the moment it cannot, so a spoken line goes out
//! before its newline. Each decision reads the shortest prefix that settles it, so the result
//! does not depend on how the text was chunked.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GuardMode {
    Voice,
}

/// Fallback lines by kind, then by caller script (`hi`, `bn`, `ta`; `hi` when the script has none).
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardLines {
    #[serde(default)]
    pub wait: BTreeMap<String, String>,
    #[serde(default)]
    pub goodbye: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    Wait,
    Goodbye,
}

impl Fallback {
    pub fn as_str(self) -> &'static str {
        match self {
            Fallback::Wait => "wait",
            Fallback::Goodbye => "goodbye",
        }
    }
}

/// The script of the caller's last message, as a fallback-line key.
pub fn caller_script(text: &str) -> &'static str {
    for c in text.chars() {
        match c as u32 {
            0x0900..=0x097F => return "hi",
            0x0980..=0x09FF => return "bn",
            0x0B80..=0x0BFF => return "ta",
            _ => {}
        }
    }
    "hi"
}

/// Openers of a planning line (lower case, matched at a word boundary).
const OPENERS: &[&str] = &[
    "wait",
    "hmm",
    "alternatively",
    "self-correction",
    "the user",
    "the caller",
    "the prompt",
    "the instruction",
    "the system",
    "the most logical",
    "the best response",
    "the best reply",
    "let me",
    "let's",
    "i need to",
    "i should",
    "my response",
    "my reply",
];

/// A label line (`Who is calling: …`) has at most this many words before its colon.
const LABEL_WORDS: usize = 4;
/// Bytes after which a line with no planning shape is speech.
const DECIDE_BYTES: usize = 32;

/// First words that mark reasoning in an answer, counted whether or not the line moved.
const ANSWER_REASONING: &[&str] = &["wait", "alternatively", "hmm", "actually"];

/// `Some(true)`: planning; `Some(false)`: speech; `None`: not settled by this prefix. `complete`:
/// the line ended.
fn planning(line: &str, complete: bool) -> Option<bool> {
    let t = line.trim_start();
    let Some(first) = t.chars().next() else {
        return complete.then_some(false);
    };
    if matches!(first, '*' | '(' | '[' | '#') {
        return Some(true);
    }
    if !first.is_ascii_alphabetic() {
        return Some(false);
    }
    let boundary = |rest: &str| {
        rest.chars().next().map_or(complete, |c| {
            !c.is_ascii_alphanumeric() && c != '-' && c != '\''
        })
    };
    let mut open = false;
    for o in OPENERS {
        let n = o.len().min(t.len());
        if !t.is_char_boundary(n) || !t[..n].eq_ignore_ascii_case(&o[..n]) {
            continue;
        }
        if t.len() < o.len() {
            open = !complete;
        } else if boundary(&t[o.len()..]) {
            return Some(true);
        } else if t.len() == o.len() {
            open = true;
        }
    }
    if open {
        return None;
    }
    // A short label then a colon, or a short status line (`Tamil detected.`, `None.`).
    let k = t
        .char_indices()
        .find(|&(_, c)| !(c.is_ascii_alphabetic() || c == ' '))
        .map(|(i, _)| i);
    match k {
        Some(k) if k <= DECIDE_BYTES => {
            let label = &t[..k];
            let words = label.split_whitespace().count();
            let end = t[k..].chars().next();
            let last = label.split_whitespace().last().unwrap_or("");
            Some(match end {
                Some(':') => words <= LABEL_WORDS,
                Some('.') => {
                    words <= 2
                        && (last.eq_ignore_ascii_case("detected")
                            || label.trim().eq_ignore_ascii_case("none"))
                }
                _ => false,
            })
        }
        None if t.len() <= DECIDE_BYTES && !complete => None,
        _ => Some(false),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Line {
    /// Not settled: held.
    Open,
    Speak,
    Move,
}

/// One chat response's guard.
#[derive(Debug)]
pub struct ReplyGuard {
    /// The current line's held bytes.
    line: String,
    state: Line,
    /// Non-whitespace speech has gone out.
    spoke: bool,
    first_word_seen: bool,
    /// Reasoning has gone out: the next moved line starts on a new line.
    reasoning_out: bool,
    pub moved: bool,
    pub answer_reasoning: bool,
    moved_text: String,
    wait: Option<String>,
    goodbye: Option<String>,
}

impl ReplyGuard {
    pub fn new(lines: Option<&GuardLines>, script: &str) -> Self {
        let pick = |m: &BTreeMap<String, String>| m.get(script).or_else(|| m.get("hi")).cloned();
        ReplyGuard {
            line: String::new(),
            state: Line::Open,
            spoke: false,
            first_word_seen: false,
            reasoning_out: false,
            moved: false,
            answer_reasoning: false,
            moved_text: String::new(),
            wait: lines.and_then(|l| pick(&l.wait)),
            goodbye: lines.and_then(|l| pick(&l.goodbye)),
        }
    }

    /// The response's trace has gone out; a moved line follows it on a new line.
    pub fn note_reasoning(&mut self) {
        self.reasoning_out = true;
    }

    fn speak(&mut self, s: &str, c: &mut String) {
        if !self.first_word_seen {
            if let Some(w) = s
                .split(|ch: char| !ch.is_alphanumeric())
                .find(|w| !w.is_empty())
            {
                self.first_word_seen = true;
                self.answer_reasoning = ANSWER_REASONING.iter().any(|a| w.eq_ignore_ascii_case(a));
            }
        }
        self.spoke |= !s.trim().is_empty();
        c.push_str(s);
    }

    fn move_text(&mut self, s: &str, r: &mut String) {
        let s = s.trim_end_matches('\n');
        r.push_str(s);
        self.moved_text.push_str(s);
    }

    fn settle(&mut self, plan: bool, r: &mut String, c: &mut String) {
        let line = std::mem::take(&mut self.line);
        if plan {
            if self.reasoning_out {
                r.push('\n');
            }
            self.reasoning_out = true;
            self.moved = true;
            self.state = Line::Move;
            self.move_text(line.trim_start(), r);
        } else {
            self.state = Line::Speak;
            self.speak(&line, c);
        }
        self.line = line;
        self.line.clear();
    }

    /// Answer text in; moved lines appended to `r`, released text to `c`.
    pub fn push(&mut self, mut text: &str, r: &mut String, c: &mut String) {
        while !text.is_empty() {
            let (piece, nl) = match text.find('\n') {
                Some(i) => (&text[..=i], true),
                None => (text, false),
            };
            text = &text[piece.len()..];
            match self.state {
                Line::Speak => self.speak(piece, c),
                Line::Move => self.move_text(piece, r),
                Line::Open => {
                    self.line.push_str(piece);
                    let body = self.line.trim_end_matches('\n');
                    if let Some(plan) = planning(body, nl) {
                        self.settle(plan, r, c);
                    }
                }
            }
            if nl {
                self.state = Line::Open;
            }
        }
    }

    /// End of the answer. `calls`: the turn made tool calls (no fallback then). Returns the
    /// fallback line's kind when the turn had nothing to say and a line was configured.
    pub fn finish(&mut self, calls: bool, r: &mut String, c: &mut String) -> Option<Fallback> {
        if self.state == Line::Open && !self.line.is_empty() {
            let plan = planning(&self.line, true).unwrap_or(false);
            self.settle(plan, r, c);
        }
        if self.spoke || calls {
            return None;
        }
        let bye = [
            "goodbye",
            "bye",
            "end the call",
            "hang up",
            "ending the call",
        ]
        .iter()
        .any(|w| self.moved_text.to_ascii_lowercase().contains(w));
        let (kind, line) = if bye {
            (Fallback::Goodbye, self.goodbye.as_ref())
        } else {
            (Fallback::Wait, self.wait.as_ref())
        };
        let line = line?;
        c.push_str(line);
        self.spoke = true;
        Some(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines() -> GuardLines {
        let m = |hi: &str, bn: &str, ta: &str| {
            [("hi", hi), ("bn", bn), ("ta", ta)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        GuardLines {
            wait: m("एक सेकंड, कृपया।", "এক সেকেন্ড, প্লিজ।", "ஒரு நொடி, தயவுசெய்து."),
            goodbye: m("धन्यवाद, नमस्ते।", "ধন্যবাদ, নমস্কার।", "நன்றி, வணக்கம்."),
        }
    }

    fn run(text: &str, k: usize, script: &str) -> (String, String, Option<Fallback>) {
        let l = lines();
        let mut g = ReplyGuard::new(Some(&l), script);
        let (mut r, mut c) = (String::new(), String::new());
        let mut i = 0;
        while i < text.len() {
            let mut j = (i + k).min(text.len());
            while !text.is_char_boundary(j) {
                j += 1;
            }
            g.push(&text[i..j], &mut r, &mut c);
            i = j;
        }
        let f = g.finish(false, &mut r, &mut c);
        (r, c, f)
    }

    /// Whole and 1/3/7-byte pieces agree.
    fn check(text: &str, moved: &str, content: &str) {
        for k in [text.len().max(1), 1, 3, 7] {
            let (r, c, _) = run(text, k, "hi");
            assert_eq!(
                (r.as_str(), c.as_str()),
                (moved, content),
                "chunks of {k}: {text:?}"
            );
        }
    }

    #[test]
    fn planning_lines_move_to_reasoning() {
        check(
            "Wait, the prompt says one question.\nAap kaun hain?",
            "Wait, the prompt says one question.",
            "Aap kaun hain?",
        );
        check("None.\nAap kaun?", "None.", "Aap kaun?");
        check("Tamil detected.\nநீங்கள் யார்?", "Tamil detected.", "நீங்கள் யார்?");
        check(
            "*Self-correction:* ask once.\nJi, boliye.",
            "*Self-correction:* ask once.",
            "Ji, boliye.",
        );
        check(
            "Theek hai.\nWho is calling: Asked twice already.\n",
            "Who is calling: Asked twice already.",
            "Theek hai.\n",
        );
        check(
            "The most logical interpretation is a delivery.\nAapka parcel aa gaya hai.",
            "The most logical interpretation is a delivery.",
            "Aapka parcel aa gaya hai.",
        );
        check("Hmm\nHaan ji?", "Hmm", "Haan ji?");
    }

    #[test]
    fn speech_passes_through() {
        for text in [
            "ठीक है, धन्यवाद। नमस्ते!",
            "I thought Rahul might call back later.",
            "Thought kiya tha ki aap aayenge.",
            "Okay, main abhi check karta hoon.",
            "Sure, I can help you with that order.",
            "Amazon se bol rahe hain? Haan ji.",
            "Waiting ke liye shukriya.",
            "Theek hai.\n\nAap kaun?",
        ] {
            check(text, "", text);
        }
    }

    /// A spoken line is released before its newline once it cannot be planning.
    #[test]
    fn a_spoken_line_is_released_early() {
        let mut g = ReplyGuard::new(None, "hi");
        let (mut r, mut c) = (String::new(), String::new());
        g.push("Aap kaun", &mut r, &mut c);
        assert_eq!(c, "", "a label is still possible");
        g.push(" hain?", &mut r, &mut c);
        assert_eq!(c, "Aap kaun hain?");
        let mut g = ReplyGuard::new(None, "hi");
        g.push("नमस्ते", &mut r, &mut c);
        assert!(c.ends_with("नमस्ते"));
    }

    #[test]
    fn an_empty_turn_gets_the_callers_fallback() {
        assert_eq!(
            run("", 1, "bn"),
            (
                String::new(),
                "এক সেকেন্ড, প্লিজ।".into(),
                Some(Fallback::Wait)
            )
        );
        let (r, c, f) = run("Wait, I should end the call.\n", 3, "ta");
        assert_eq!(
            (r.as_str(), c.as_str(), f),
            (
                "Wait, I should end the call.",
                "நன்றி, வணக்கம்.",
                Some(Fallback::Goodbye)
            )
        );
        let mut g = ReplyGuard::new(Some(&lines()), "hi");
        let (mut r, mut c) = (String::new(), String::new());
        assert_eq!(g.finish(true, &mut r, &mut c), None, "a call is an answer");
        let mut g = ReplyGuard::new(None, "hi");
        assert_eq!(g.finish(false, &mut r, &mut c), None, "no line configured");
        assert_eq!(caller_script("मैं राहुल"), "hi");
        assert_eq!(caller_script("hello"), "hi");
        assert_eq!(caller_script("আমি"), "bn");
    }

    #[test]
    fn any_chunking_guards_the_same() {
        const PIECES: &[&str] = &[
            "Wait",
            ", ",
            "the prompt",
            "None",
            ".",
            "\n",
            " ",
            "Who is calling",
            ":",
            "Aap",
            " kaun",
            "?",
            "नमस्ते",
            "*",
            "Tamil detected",
            "Hmm",
            "Theek hai",
            "\n\n",
            "é日",
        ];
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        for _ in 0..2000 {
            let n = 1 + rnd(12);
            let text: String = (0..n).map(|_| PIECES[rnd(PIECES.len())]).collect();
            let whole = run(&text, text.len().max(1), "hi");
            for k in [1, 2, 5] {
                assert_eq!(run(&text, k, "hi"), whole, "chunks of {k}: {text:?}");
            }
        }
    }

    #[test]
    fn answer_reasoning_is_counted_without_moving() {
        let mut g = ReplyGuard::new(None, "hi");
        let (mut r, mut c) = (String::new(), String::new());
        g.push("Actually, aapka order kal aayega.", &mut r, &mut c);
        g.finish(false, &mut r, &mut c);
        assert!(g.answer_reasoning && !g.moved);
        assert_eq!(c, "Actually, aapka order kal aayega.");
    }
}

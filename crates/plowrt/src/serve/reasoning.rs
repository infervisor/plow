//! Splitting a reasoning trace out of the answer.
//!
//! A trace is framed by `<think>` … `</think>`, and the OPENING marker turns up
//! in one of two places depending on the checkpoint:
//!
//!   * GLM-5.2/5.3 leave it dangling at the end of the generation prompt, so
//!     generation begins already inside the trace and only the CLOSE is
//!     generated;
//!   * Qwen3 and DeepSeek-R1 end the prompt on an ordinary assistant turn and
//!     the MODEL emits `<think>` as its first output.
//!
//! Handling only the first shape is what let a live Qwen3 return its whole
//! trace — marker and all — inside `content`. `</think>` is not a special token
//! for either family, so `skip_special_tokens` does not remove it.
//!
//! WHY ONE TYPE FOR BOTH RESPONSE PATHS. The buffered and streamed paths used
//! to carry separate implementations of this split, and they disagreed: on a
//! trace that opened and never closed one called the text an answer and the
//! other called it a trace. They are the same generation, so they get the same
//! state machine; the buffered path is just `push` + `finish`.

/// How a model frames a reasoning trace.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReasoningMode {
    /// No separable trace: everything generated is the answer.
    #[default]
    None,
    /// `<think>` … `</think>`, opened by the prompt or by the generation.
    ThinkTag,
    /// Another marker pair, from the packet's `serve.json` (leaked once per loaded model).
    Tags { open: &'static str, close: &'static str },
}

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

impl ReasoningMode {
    /// The mode a packet declares. A legacy packet (no `serve.json`) keeps `<think>` splitting
    /// for every model, as before the section existed.
    pub fn from_serve(serve: &crate::asset::serve::ServeInfo) -> Self {
        if !serve.from_packet {
            return ReasoningMode::ThinkTag;
        }
        match serve.manifest.chat.as_ref().and_then(|c| c.reasoning.as_ref()) {
            None => ReasoningMode::None,
            Some(t) if t.open == OPEN && t.close == CLOSE => ReasoningMode::ThinkTag,
            Some(t) if !t.open.is_empty() && !t.close.is_empty() => ReasoningMode::Tags {
                open: Box::leak(t.open.clone().into_boxed_str()),
                close: Box::leak(t.close.clone().into_boxed_str()),
            },
            Some(_) => ReasoningMode::None,
        }
    }

    pub fn for_bundle(bundle: &crate::asset::ModelBundle) -> Self {
        bundle.reasoning()
    }

    pub fn markers(self) -> Option<(&'static str, &'static str)> {
        match self {
            ReasoningMode::None => None,
            ReasoningMode::ThinkTag => Some((OPEN, CLOSE)),
            ReasoningMode::Tags { open, close } => Some((open, close)),
        }
    }

    /// Whether the rendered prompt leaves the trace OPEN.
    ///
    /// Tested as a SUFFIX, never a search. `add_generation_prompt` puts the
    /// generation prompt last, so only a marker at the very end is the model's
    /// own. The previous probe searched the whole rendered prompt — which
    /// contains USER TEXT — so asking about the `<think>` tag routed the entire
    /// answer into `reasoning_content` and returned an empty `content`.
    ///
    /// A request that turned thinking off renders the pair CLOSED
    /// (`<think></think>`) and so correctly reads as not-open here.
    pub fn prompt_opens(self, prompt: &str) -> bool {
        // Whitespace after the marker is the template's (Qwen3.5 `<think>\n`, Gemma 4
        // `<|channel>thought\n`).
        self.markers().is_some_and(|(open, _)| prompt.trim_end().ends_with(open))
    }

    /// The marker that closes a trace, when the mode splits one.
    pub fn close_marker(self) -> Option<&'static str> {
        self.markers().map(|(_, close)| close)
    }

    /// The packet's declared framing, else the one the chat template's model writes (a packet
    /// that declares none and whose template has a known trace channel, e.g. Gemma 4).
    pub fn resolve(serve: &crate::asset::serve::ServeInfo, template: Option<(&'static str, &'static str)>) -> Self {
        let declared = serve.from_packet && serve.manifest.chat.as_ref().is_some_and(|c| c.reasoning.is_some());
        match template {
            // `<think>` stays the packet's call: its emit reads the tokenizer for it.
            Some((open, close)) if !declared && (open, close) != (OPEN, CLOSE) => ReasoningMode::Tags { open, close },
            _ => Self::from_serve(serve),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Inside the trace; waiting for `</think>`.
    Open,
    /// Not yet known: the model may still open a trace with `<think>` as its
    /// first output. Bytes are withheld only until that is decided.
    Deciding,
    /// Settled — everything from here is the answer.
    Answer,
}

/// Incremental `(reasoning, content)` splitter over a generation.
#[derive(Debug)]
pub struct ReasoningSplit {
    state: State,
    open: &'static str,
    close: &'static str,
    /// Bytes generated but not yet attributed to one side or the other.
    hold: String,
    /// Tokens seen while inside the trace, for `reasoning_tokens`.
    pub trace_tokens: u64,
    /// Whether a non-empty reasoning piece has gone out yet.
    started_reasoning: bool,
    /// Whether this generation ever had a trace, and whether the first piece of
    /// answer after it has gone out.
    had_trace: bool,
    started_answer: bool,
}

impl ReasoningSplit {
    /// `prompt_opens` comes from [`ReasoningMode::prompt_opens`]; `mode` being
    /// [`ReasoningMode::None`] disables splitting entirely.
    pub fn new(mode: ReasoningMode, prompt_opens: bool) -> Self {
        let (open, close) = mode.markers().unwrap_or((OPEN, CLOSE));
        let state = match (mode, prompt_opens) {
            (ReasoningMode::None, _) => State::Answer,
            (_, true) => State::Open,
            (_, false) => State::Deciding,
        };
        ReasoningSplit {
            state,
            open,
            close,
            hold: String::new(),
            trace_tokens: 0,
            started_reasoning: false,
            had_trace: state == State::Open,
            started_answer: false,
        }
    }

    /// Whether the next byte fed belongs to the trace, for token accounting.
    pub fn in_trace(&self) -> bool {
        self.state != State::Answer
    }

    /// Emit a piece of trace into `dst`; whether it carried any.
    ///
    /// Whitespace is trimmed only at the OUTER boundaries of the trace — the
    /// start of the first piece and the end of the last. Trimming each piece
    /// as it went out ate the space BETWEEN two pieces: a trace ending
    /// `"...should be"` + `" four."` came back as `"...should befour."`, a
    /// silent corruption of the model's own words that only shows up when the
    /// close marker lands in its own token.
    fn emit_reasoning(&mut self, piece: &str, dst: &mut String) -> bool {
        let piece = if self.started_reasoning {
            piece
        } else {
            piece.trim_start()
        };
        if piece.is_empty() {
            return false;
        }
        self.started_reasoning = true;
        dst.push_str(piece);
        true
    }

    /// Emit a piece of answer. Only the first piece AFTER a trace is trimmed;
    /// a generation with no trace is passed through byte for byte (an empty
    /// piece still counts as emitted).
    fn emit_answer(&mut self, piece: &str, dst: &mut String) -> bool {
        if !self.had_trace {
            dst.push_str(piece);
            return true;
        }
        let piece = if self.started_answer {
            piece
        } else {
            piece.trim_start()
        };
        if piece.is_empty() {
            return false;
        }
        self.started_answer = true;
        dst.push_str(piece);
        true
    }

    /// Feed one token's text. Returns whatever can be attributed now; bytes
    /// that could still be part of a marker stay held until a later token or
    /// [`Self::finish`] settles them.
    pub fn push(&mut self, text: &str) -> (Option<String>, Option<String>) {
        let (mut r, mut c) = (String::new(), String::new());
        let (hr, hc) = self.push_into(text, &mut r, &mut c);
        (hr.then_some(r), hc.then_some(c))
    }

    /// [`Self::push`], appending to `r` / `c` (no allocation of its own once the hold has
    /// grown); returns whether each side was emitted.
    pub fn push_into(&mut self, text: &str, r: &mut String, c: &mut String) -> (bool, bool) {
        if self.state == State::Answer {
            // An empty delta is still emitted when there was no trace: a
            // partial UTF-8 token still gets a chunk, so the client's token
            // count stays accurate. That is the streamed path's existing contract.
            return (false, self.emit_answer(text, c));
        }
        self.trace_tokens += 1;
        self.hold.push_str(text);

        if self.state == State::Deciding {
            let trimmed = self.hold.trim_start();
            if trimmed.starts_with(self.open) {
                // The model opened a trace. The marker and the whitespace in
                // front of it are framing, not answer.
                let consumed = self.hold.len() - trimmed.len() + self.open.len();
                self.hold.drain(..consumed);
                self.state = State::Open;
                self.had_trace = true;
            } else if trimmed.is_empty() || self.open.starts_with(trimmed) {
                // Nothing but whitespace yet, or still a possible prefix of
                // `<think>` — wait for more.
                //
                // The whitespace case is load-bearing: settling on it would
                // mean a model whose FIRST token is a newline and whose second
                // is `<think>` never opens a trace at all, and the whole thing
                // leaks into `content`. Qwen3 happens to emit `<think>` as one
                // whole first token; nothing guarantees that tokenization.
                return (false, false);
            } else {
                // Settled: this generation has no trace. Everything held is
                // answer, and nothing is withheld from here on.
                self.state = State::Answer;
                self.trace_tokens = 0;
                let out = std::mem::take(&mut self.hold);
                let e = self.emit_answer(&out, c);
                self.hold = out;
                self.hold.clear();
                return (false, e);
            }
        }

        match self.hold.find(self.close) {
            Some(i) => {
                // Last piece of the trace: trim only its END.
                let hold = std::mem::take(&mut self.hold);
                let er = self.emit_reasoning(hold[..i].trim_end(), r);
                self.state = State::Answer;
                let ec = self.emit_answer(&hold[i + self.close.len()..], c);
                self.hold = hold;
                self.hold.clear();
                (er, ec)
            }
            None => {
                // Withhold only what could still begin the close marker.
                let keep = (self.close.len() - 1).min(self.hold.len());
                let cut = (0..=keep)
                    .rev()
                    .map(|k| self.hold.len() - k)
                    .find(|&c| self.hold.is_char_boundary(c))
                    .unwrap_or(self.hold.len());
                let hold = std::mem::take(&mut self.hold);
                let er = self.emit_reasoning(&hold[..cut], r);
                self.hold = hold;
                self.hold.drain(..cut);
                (er, false)
            }
        }
    }

    /// Flush whatever is still held at the end of the generation.
    ///
    /// Nothing may be dropped here. The streamed path used to simply abandon
    /// the hold, so a generation that ended inside its trace lost up to
    /// `len("</think>") - 1` bytes of `reasoning_content` — and disagreed with
    /// the buffered path about the same generation.
    pub fn finish(&mut self) -> (Option<String>, Option<String>) {
        let (mut r, mut c) = (String::new(), String::new());
        let (hr, hc) = self.finish_into(&mut r, &mut c);
        (hr.then_some(r), (hc && !c.is_empty()).then_some(c))
    }

    /// [`Self::finish`], appending to `r` / `c`.
    pub fn finish_into(&mut self, r: &mut String, c: &mut String) -> (bool, bool) {
        let out = std::mem::take(&mut self.hold);
        match self.state {
            // Opened and never closed: the generation ran out inside its own
            // trace, so all of it is trace and there is no answer yet.
            State::Open => (self.emit_reasoning(out.trim_end(), r), false),
            // Never got enough bytes to tell — it was never a trace.
            State::Deciding => {
                self.trace_tokens = 0;
                (false, self.emit_answer(&out, c))
            }
            // Settled. `trace_tokens` is already right: `push` zeroed it if the
            // generation turned out to have no trace, and otherwise it is the
            // real count. Zeroing here too threw away every closed trace's
            // count, which is what `completion_tokens_details` reports.
            State::Answer => (false, self.emit_answer(&out, c)),
        }
    }
}

/// Split a COMPLETE generation in one call.
///
/// For TESTS and any caller that only has the finished text. It deliberately
/// does NOT report `trace_tokens`: fed one string it cannot know how many
/// tokens the trace spanned, and a plausible-looking wrong count is worse than
/// no count. The serving paths drive `push` per token and read
/// [`ReasoningSplit::trace_tokens`].
pub fn split(mode: ReasoningMode, prompt_opens: bool, text: &str) -> (Option<String>, String) {
    let mut s = ReasoningSplit::new(mode, prompt_opens);
    let (r1, c1) = s.push(text);
    let (r2, c2) = s.finish();
    let mut reasoning = r1.unwrap_or_default();
    if let Some(r) = r2 {
        reasoning.push_str(&r);
    }
    let mut content = c1.unwrap_or_default();
    if let Some(c) = c2 {
        content.push_str(&c);
    }
    let reasoning = reasoning.trim().to_string();
    ((!reasoning.is_empty()).then_some(reasoning), content)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffered(prompt_opens: bool, text: &str) -> (Option<String>, String) {
        split(ReasoningMode::ThinkTag, prompt_opens, text)
    }

    /// GLM: the prompt leaves `<think>` open and only the close is generated.
    #[test]
    fn a_prompt_opened_trace_splits_at_the_close() {
        let (r, c) = buffered(true, "weighing it up</think>The answer is 4.");
        assert_eq!(r.as_deref(), Some("weighing it up"));
        assert_eq!(c, "The answer is 4.");
    }

    /// Qwen3 / DeepSeek-R1: the MODEL emits the opening marker. This is the
    /// shape a live Qwen3 proved was reaching `content` verbatim.
    #[test]
    fn a_generation_opened_trace_splits_too() {
        let (r, c) = buffered(false, "<think>\nlet me see\n</think>\n\nfour");
        assert_eq!(r.as_deref(), Some("let me see"));
        assert_eq!(c, "four");
    }

    /// THE REGRESSION. Ordinary prose that merely mentions the marker must come
    /// back untouched, with a non-empty `content`.
    #[test]
    fn prose_mentioning_the_marker_is_all_answer() {
        let (r, c) = buffered(false, "The </think> tag closes a thinking block.");
        assert!(r.is_none());
        assert_eq!(c, "The </think> tag closes a thinking block.");
    }

    #[test]
    fn a_disabled_mode_never_splits() {
        let (r, c) = split(ReasoningMode::None, false, "<think>x</think>y");
        assert!(r.is_none());
        assert_eq!(c, "<think>x</think>y");
    }

    /// `trace_tokens` counts PUSHES inside the trace, so both serving paths
    /// must drive it per token. Feeding the whole generation at once would
    /// report 1, which is why `split` does not expose it.
    #[test]
    fn trace_tokens_counts_tokens_not_calls() {
        let mut s = ReasoningSplit::new(ReasoningMode::ThinkTag, true);
        for t in ["abc", "def", "ghi", "</think>", "answer"] {
            s.push(t);
        }
        s.finish();
        assert_eq!(s.trace_tokens, 4, "three trace tokens plus the closing one");
    }

    /// Opened and never closed — both paths must call it all trace.
    #[test]
    fn an_unclosed_trace_is_all_reasoning() {
        let (r, c) = buffered(true, "still thinking");
        assert_eq!(r.as_deref(), Some("still thinking"));
        assert_eq!(c, "");
    }

    /// Fed one byte at a time, the streamed path must produce exactly what the
    /// buffered path produces. These two disagreeing is the bug this type
    /// exists to make impossible.
    #[test]
    fn streaming_byte_by_byte_agrees_with_the_buffered_split() {
        for (opens, text) in [
            (true, "weighing it up</think>The answer is 4."),
            (false, "<think>\nlet me see\n</think>\n\nfour"),
            (false, "plain answer with no marker"),
            (true, "unclosed trace"),
            (false, "<think>only a trace"),
        ] {
            let (want_r, want_c) = buffered(opens, text);
            let mut s = ReasoningSplit::new(ReasoningMode::ThinkTag, opens);
            let (mut got_r, mut got_c) = (String::new(), String::new());
            for ch in text.chars() {
                let (r, c) = s.push(&ch.to_string());
                got_r.push_str(&r.unwrap_or_default());
                got_c.push_str(&c.unwrap_or_default());
            }
            let (r, c) = s.finish();
            got_r.push_str(&r.unwrap_or_default());
            got_c.push_str(&c.unwrap_or_default());
            assert_eq!(
                got_r.trim(),
                want_r.unwrap_or_default().trim(),
                "reasoning differs for {text:?}"
            );
            assert_eq!(got_c.trim(), want_c.trim(), "content differs for {text:?}");
        }
    }

    /// A trace whose opening marker does not land in the FIRST token. Settling
    /// on the leading whitespace would leak the whole trace into `content`.
    #[test]
    fn a_trace_opened_after_leading_whitespace_is_still_found() {
        let mut s = ReasoningSplit::new(ReasoningMode::ThinkTag, false);
        let (mut r, mut c) = (String::new(), String::new());
        for t in ["\n", "<think>", "abc", "</think>", "ans"] {
            let (rr, cc) = s.push(t);
            r.push_str(&rr.unwrap_or_default());
            c.push_str(&cc.unwrap_or_default());
        }
        let (rr, cc) = s.finish();
        r.push_str(&rr.unwrap_or_default());
        c.push_str(&cc.unwrap_or_default());
        assert_eq!(r.trim(), "abc");
        assert_eq!(c.trim(), "ans");
    }

    /// A generation that is ONLY whitespace must still come back as content.
    #[test]
    fn an_all_whitespace_generation_is_content() {
        let mut s = ReasoningSplit::new(ReasoningMode::ThinkTag, false);
        let (r, c) = s.push("   ");
        assert!(r.is_none() && c.is_none(), "held while undecided");
        let (r, c) = s.finish();
        assert!(r.is_none());
        assert_eq!(c.as_deref(), Some("   "));
    }

    /// THE EATEN SPACE. When the close marker lands in its own token the trace
    /// arrives as two pieces, and trimming each piece as it went out joined
    /// them without the whitespace between: `"should be"` + `" four."` came
    /// back as `"should befour."`.
    #[test]
    fn whitespace_between_two_trace_pieces_survives() {
        let mut s = ReasoningSplit::new(ReasoningMode::ThinkTag, true);
        let (a, _) = s.push("The answer should be");
        let (b, _) = s.push(" four.\n");
        let (c, _) = s.push("</think>");
        let joined = format!(
            "{}{}{}",
            a.unwrap_or_default(),
            b.unwrap_or_default(),
            c.unwrap_or_default()
        );
        assert_eq!(joined, "The answer should be four.");
    }

    /// Nothing may be lost when the stream ends mid-marker.
    #[test]
    fn a_stream_ending_inside_the_close_marker_loses_nothing() {
        let mut s = ReasoningSplit::new(ReasoningMode::ThinkTag, true);
        let (r, _) = s.push("abc</thin");
        let (r2, _) = s.finish();
        let all = format!("{}{}", r.unwrap_or_default(), r2.unwrap_or_default());
        assert_eq!(all.trim(), "abc</thin");
    }

    #[test]
    fn the_prompt_suffix_decides_whether_a_trace_is_open() {
        assert!(ReasoningMode::ThinkTag.prompt_opens("<|user|>hi<|assistant|><think>"));
        // thinking disabled renders the pair closed
        assert!(!ReasoningMode::ThinkTag.prompt_opens("<|assistant|><think></think>"));
        // user text cannot reach the end
        assert!(!ReasoningMode::ThinkTag.prompt_opens("what is <think>?<|assistant|>"));
    }

    #[test]
    fn packet_declared_markers_split_the_trace() {
        let mode = ReasoningMode::Tags { open: "<|think|>", close: "<|/think|>" };
        assert!(mode.prompt_opens("...<|think|>"));
        assert!(!ReasoningMode::None.prompt_opens("...<think>"));
        let (r, c) = split(mode, false, "<|think|>plan<|/think|>answer");
        assert_eq!((r.as_deref(), c.as_str()), (Some("plan"), "answer"));
        let (r, c) = split(mode, false, "<think>x</think>y");
        assert_eq!((r, c.as_str()), (None, "<think>x</think>y"));
    }
}

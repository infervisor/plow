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
}

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

impl ReasoningMode {
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
    pub fn prompt_opens(prompt: &str) -> bool {
        prompt.ends_with(OPEN)
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
        let state = match (mode, prompt_opens) {
            (ReasoningMode::None, _) => State::Answer,
            (ReasoningMode::ThinkTag, true) => State::Open,
            (ReasoningMode::ThinkTag, false) => State::Deciding,
        };
        ReasoningSplit {
            state,
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

    /// Emit a piece of trace.
    ///
    /// Whitespace is trimmed only at the OUTER boundaries of the trace — the
    /// start of the first piece and the end of the last. Trimming each piece
    /// as it went out ate the space BETWEEN two pieces: a trace ending
    /// `"...should be"` + `" four."` came back as `"...should befour."`, a
    /// silent corruption of the model's own words that only shows up when the
    /// close marker lands in its own token.
    fn emit_reasoning(&mut self, piece: &str) -> Option<String> {
        let piece = if self.started_reasoning {
            piece
        } else {
            piece.trim_start()
        };
        if piece.is_empty() {
            return None;
        }
        self.started_reasoning = true;
        Some(piece.to_string())
    }

    /// Emit a piece of answer. Only the first piece AFTER a trace is trimmed;
    /// a generation with no trace is passed through byte for byte.
    fn emit_answer(&mut self, piece: &str) -> Option<String> {
        if !self.had_trace {
            return Some(piece.to_string());
        }
        let piece = if self.started_answer {
            piece
        } else {
            piece.trim_start()
        };
        if piece.is_empty() {
            return None;
        }
        self.started_answer = true;
        Some(piece.to_string())
    }

    /// Feed one token's text. Returns whatever can be attributed now; bytes
    /// that could still be part of a marker stay held until a later token or
    /// [`Self::finish`] settles them.
    pub fn push(&mut self, text: &str) -> (Option<String>, Option<String>) {
        if self.state == State::Answer {
            // Some("") rather than None even for an empty delta when there was
            // no trace: a partial UTF-8 token still gets a chunk, so the
            // client's token count stays accurate. That is the streamed path's
            // existing contract.
            return (None, self.emit_answer(text));
        }
        self.trace_tokens += 1;
        self.hold.push_str(text);

        if self.state == State::Deciding {
            let trimmed = self.hold.trim_start();
            if trimmed.starts_with(OPEN) {
                // The model opened a trace. The marker and the whitespace in
                // front of it are framing, not answer.
                let consumed = self.hold.len() - trimmed.len() + OPEN.len();
                self.hold.drain(..consumed);
                self.state = State::Open;
                self.had_trace = true;
            } else if OPEN.starts_with(trimmed) && !trimmed.is_empty() {
                // Still a possible prefix of `<think>` — wait for more.
                return (None, None);
            } else {
                // Settled: this generation has no trace. Everything held is
                // answer, and nothing is withheld from here on.
                self.state = State::Answer;
                self.trace_tokens = 0;
                let out = std::mem::take(&mut self.hold);
                return (None, self.emit_answer(&out));
            }
        }

        match self.hold.find(CLOSE) {
            Some(i) => {
                // Last piece of the trace: trim only its END.
                let before = self.hold[..i].trim_end().to_string();
                let after = self.hold[i + CLOSE.len()..].to_string();
                self.hold.clear();
                self.state = State::Answer;
                let r = self.emit_reasoning(&before);
                let c = self.emit_answer(&after);
                (r, c)
            }
            None => {
                // Withhold only what could still begin the close marker.
                let keep = (CLOSE.len() - 1).min(self.hold.len());
                let cut = (0..=keep)
                    .rev()
                    .map(|k| self.hold.len() - k)
                    .find(|&c| self.hold.is_char_boundary(c))
                    .unwrap_or(self.hold.len());
                let emit = self.hold[..cut].to_string();
                self.hold.drain(..cut);
                (self.emit_reasoning(&emit), None)
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
        let out = std::mem::take(&mut self.hold);
        match self.state {
            // Opened and never closed: the generation ran out inside its own
            // trace, so all of it is trace and there is no answer yet.
            State::Open => {
                let t = out.trim_end().to_string();
                (self.emit_reasoning(&t), None)
            }
            // Never got enough bytes to tell — it was never a trace.
            State::Deciding => {
                self.trace_tokens = 0;
                (None, self.emit_answer(&out))
            }
            // Settled. `trace_tokens` is already right: `push` zeroed it if the
            // generation turned out to have no trace, and otherwise it is the
            // real count. Zeroing here too threw away every closed trace's
            // count, which is what `completion_tokens_details` reports.
            State::Answer => {
                let c = self.emit_answer(&out);
                (None, c.filter(|s| !s.is_empty()))
            }
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
        assert!(ReasoningMode::prompt_opens("<|user|>hi<|assistant|><think>"));
        // thinking disabled renders the pair closed
        assert!(!ReasoningMode::prompt_opens("<|assistant|><think></think>"));
        // user text cannot reach the end
        assert!(!ReasoningMode::prompt_opens("what is <think>?<|assistant|>"));
    }
}

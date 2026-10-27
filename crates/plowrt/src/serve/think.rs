//! Thinking control for one request, on committed token ids: a budget per trace block that ends
//! the block with forced tokens (a bridge sentence and the close marker), and a cap on the reply
//! after the trace.
//!
//! The mux consults it before each produced token is committed. Forcing replaces the SAMPLED id
//! after sampling, so it needs no logits and runs on every backend. A replacement is exact only
//! where the host feeds the next input; the chained decode paths (pipelined lookahead, multistep
//! quanta, speculative verify) feed the device's own token, so they cap a quantum at
//! [`ThinkCtl::chain_left`]: a replacement then only ever lands on a quantum's last token.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Weak};

/// The longest chain a live-controlled request's decode paths run. A live `reasoning_end` ends
/// the trace this many tokens after it is seen, past the end of any chain sized before it arrived.
pub const LIVE_CHAIN: u32 = 16;

static LIVE: std::sync::LazyLock<
    parking_lot::Mutex<std::collections::HashMap<Arc<str>, Weak<AtomicBool>>>,
> = std::sync::LazyLock::new(Default::default);

/// Register `request` for live control; the flag lives as long as its controller.
pub fn live_register(request: &Arc<str>) -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let mut m = LIVE.lock();
    m.retain(|_, w| w.strong_count() > 0);
    m.insert(Arc::clone(request), Arc::downgrade(&flag));
    flag
}

/// Ask a live request to end its trace now (a caller barged in). False when no such request is
/// in flight under live control.
pub fn live_end(request: &str) -> bool {
    match LIVE.lock().get(request).and_then(Weak::upgrade) {
        Some(f) => {
            f.store(true, Relaxed);
            true
        }
        None => false,
    }
}

/// What a forced end commits before the close marker when nothing names one: Qwen's documented
/// budget-ending sentence. It turns the model from deliberating to answering; a bare close does not.
pub const DEFAULT_BRIDGE: &str =
    "\n\nConsidering the limited time by the user, I have to give the \
     solution based on the thinking directly now.\n";

/// A model's trace markers as token ids. Built once per model.
#[derive(Debug)]
pub struct ThinkMarkers {
    /// The special token that opens a trace block (Gemma 4's `<|channel>`; a label may follow).
    pub open: u32,
    pub close: u32,
    /// Ids whose text ends a line, sorted: forcing waits for one during the newline grace.
    pub newline: Vec<u32>,
}

impl ThinkMarkers {
    /// `open` must be one token or begin with a special token, and `close` must be one token: a
    /// marker spread over ordinary tokens cannot be told from prose by id.
    pub fn tokenize(
        tok: &dyn crate::text::tokenizer::Tokenize,
        specials: &crate::serve::tools::SpecialText,
        (open, close): (&str, &str),
    ) -> Result<Self, String> {
        let one = |s: &str| tok.encode_with_special_tokens(s, false);
        let o = one(open);
        let c = one(close);
        let first = o.first().map(|&id| tok.decode_keep_special(&[id]));
        let (Some(&open_id), Some(text)) = (o.first(), first) else {
            return Err(format!("the open marker {open:?} encodes to no tokens"));
        };
        // One token (Qwen3's `<think>`, an added token), or a special token then a label
        // (Gemma 4's `<|channel>` + `thought`).
        if o.len() != 1 && (!specials.any_in(&text) || !open.starts_with(text.as_str())) {
            return Err(format!(
                "the open marker {open:?} is neither one token nor led by a special token ({o:?})"
            ));
        }
        let [close_id] = c[..] else {
            return Err(format!(
                "the close marker {close:?} is not a single token ({c:?})"
            ));
        };
        let mut newline: Vec<u32> = (0..tok.vocab_size() as u32)
            .filter(|&id| tok.decode_keep_special(&[id]).ends_with('\n'))
            .collect();
        newline.sort_unstable();
        Ok(ThinkMarkers {
            open: open_id,
            close: close_id,
            newline,
        })
    }

    fn ends_line(&self, id: u32) -> bool {
        self.newline.binary_search(&id).is_ok()
    }
}

/// Host-logit steering (CUDA). A step that carries a bias samples on the host, so it leaves the
/// device fast paths for that step; each rule is opt-in per model.
#[derive(Clone, Debug, Default)]
pub struct ThinkBias {
    /// The request's own `logit_bias`, kept under the steering entries.
    pub base: Vec<(u32, f32)>,
    /// Added to these ids for the first two reply tokens (no reply opening with `*`, `Wait`, …).
    pub reply_first: Vec<(u32, f32)>,
    /// −100 on the open marker while replying.
    pub ban_reopen: bool,
    /// `(k, max)`: at line ends over a block's last `k` budget tokens, a bias on the close marker
    /// rising to `max`.
    pub close_ramp: Option<(u32, f32)>,
    /// At line ends inside a trace, close the block when the close marker's logit ranks below this.
    pub early_exit_rank: Option<u32>,
}

impl ThinkBias {
    fn on(&self) -> bool {
        !self.reply_first.is_empty()
            || self.ban_reopen
            || self.close_ramp.is_some()
            || self.early_exit_rank.is_some()
    }
}

/// What a request asks of the controller (request > model default).
#[derive(Clone, Debug, Default)]
pub struct ThinkLimits {
    /// Tokens allowed inside each trace block.
    pub budget: Option<u32>,
    /// Tokens allowed after the close marker.
    pub reply_cap: Option<u32>,
    /// Ids committed in place of the model's once a block's budget is spent; ends with `close`.
    pub forced_end: Arc<[u32]>,
    /// Extra trace tokens the model may write past the budget while waiting for a line end.
    pub grace: u32,
    /// A blank line in the reply ends the request.
    pub stop_blank_line: bool,
    /// Set from outside ([`live_end`]): end the current trace block within [`LIVE_CHAIN`] tokens.
    pub live: Option<Arc<AtomicBool>>,
    pub bias: ThinkBias,
    /// While forcing, a speculative tick verifies the forced ids as the row's drafts.
    pub spec_forced: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// The model may still open a trace as its first token.
    Deciding,
    Trace,
    Reply,
}

/// What a request's controller did, for `x_plow_thinking` and the metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ThinkReport {
    pub blocks: u32,
    pub forced: u32,
    pub grace_tokens: u32,
    /// The voice reply guard moved a planning line (set by the chat handler).
    pub guard_moved: bool,
}

/// One committed token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commit {
    pub token: u32,
    /// `token` differs from the sampled id.
    pub replaced: bool,
    /// The reply cap is reached with this token.
    pub capped: bool,
}

#[derive(Clone, Debug)]
pub struct ThinkCtl {
    m: Arc<ThinkMarkers>,
    lim: ThinkLimits,
    phase: Phase,
    trace_n: u32,
    reply_n: u32,
    /// The block's budget (and grace) is spent: forcing starts once the codepoint is complete.
    due: bool,
    /// Index of the next forced id.
    forcing: Option<usize>,
    /// Trace blocks the model opened or the prompt left open.
    pub blocks: u32,
    /// Blocks ended by forcing.
    pub forced: u32,
    /// Grace tokens spent across blocks.
    pub grace_spent: u32,
    /// The last committed id was a marker or forced: its text is framing, not reply.
    framing: bool,
    reply_spoke: bool,
    /// The reply's text so far ends in a newline (and blanks).
    reply_nl: bool,
    /// The current block ends at this trace count (a live `reasoning_end`).
    cut: Option<u32>,
    /// The last committed trace token ended a line.
    line_end: bool,
}

impl ThinkCtl {
    /// `opened`: the prompt leaves a trace open.
    pub fn new(m: Arc<ThinkMarkers>, lim: ThinkLimits, opened: bool) -> Self {
        ThinkCtl {
            m,
            lim,
            phase: if opened {
                Phase::Trace
            } else {
                Phase::Deciding
            },
            trace_n: 0,
            reply_n: 0,
            due: false,
            forcing: None,
            blocks: u32::from(opened),
            forced: 0,
            grace_spent: 0,
            framing: false,
            reply_spoke: false,
            reply_nl: false,
            cut: None,
            line_end: false,
        }
    }

    fn open_block(&mut self) {
        self.phase = Phase::Trace;
        self.trace_n = 0;
        self.due = false;
        self.cut = None;
        self.blocks += 1;
    }

    /// The id to commit for `sampled`. Called before the token is appended to the output.
    pub fn commit(&mut self, sampled: u32) -> Commit {
        let mut c = Commit {
            token: sampled,
            replaced: false,
            capped: false,
        };
        self.framing = true;
        self.line_end = false;
        if let Some(i) = self.forcing {
            c.token = self.lim.forced_end[i];
            c.replaced = c.token != sampled;
            if i + 1 < self.lim.forced_end.len() {
                self.forcing = Some(i + 1);
            } else {
                self.forcing = None;
                self.phase = Phase::Reply;
            }
            return c;
        }
        match self.phase {
            Phase::Deciding | Phase::Reply if sampled == self.m.open => self.open_block(),
            Phase::Deciding | Phase::Reply if sampled == self.m.close => self.phase = Phase::Reply,
            Phase::Deciding | Phase::Reply => {
                self.phase = Phase::Reply;
                self.framing = false;
                self.reply_n += 1;
                c.capped = self.lim.reply_cap.is_some_and(|cap| self.reply_n >= cap);
            }
            Phase::Trace if sampled == self.m.close => self.phase = Phase::Reply,
            Phase::Trace => {
                self.trace_n += 1;
                self.line_end = self.m.ends_line(sampled);
                if self
                    .lim
                    .live
                    .as_ref()
                    .is_some_and(|f| f.swap(false, Relaxed))
                {
                    let at = self.trace_n.saturating_add(LIVE_CHAIN);
                    self.cut = Some(self.cut.map_or(at, |c| c.min(at)));
                }
                self.due |= self.cut.is_some_and(|c| self.trace_n >= c);
                if let Some(b) = self.lim.budget {
                    if self.trace_n > b {
                        self.grace_spent += 1;
                    }
                    self.due |= self.trace_n >= b
                        && (self.trace_n >= b.saturating_add(self.lim.grace)
                            || self.m.ends_line(sampled));
                }
            }
        }
        c
    }

    /// With `spec_forced`, the forced ids still to commit, next first: a speculative tick drafts
    /// them. Each accepted draft commits as itself and the span's last token is replaced as
    /// usual, so every committed id equals the input the verify consumed.
    pub fn forced_ahead(&self) -> Option<&[u32]> {
        let i = self.forcing.filter(|_| self.lim.spec_forced)?;
        Some(&self.lim.forced_end[i..])
    }

    /// Whether [`Self::next_bias`] has anything to say.
    pub fn steering(&self) -> bool {
        self.lim.bias.on()
    }

    /// The `logit_bias` for the next token: the request's own, then the steering rules that hold.
    pub fn next_bias(&self, out: &mut Vec<(u32, f32)>) {
        let b = &self.lim.bias;
        out.clear();
        out.extend_from_slice(&b.base);
        if self.forcing.is_some() {
            return;
        }
        match self.phase {
            Phase::Reply => {
                if self.reply_n < 2 {
                    out.extend_from_slice(&b.reply_first);
                }
                if b.ban_reopen {
                    out.push((self.m.open, -100.0));
                }
            }
            Phase::Trace if self.line_end => {
                if let (Some((k, max)), Some(budget)) = (b.close_ramp, self.lim.budget) {
                    let start = budget.saturating_sub(k);
                    if self.trace_n >= start && k > 0 {
                        let step = (self.trace_n - start + 1).min(k);
                        out.push((self.m.close, max * step as f32 / k as f32));
                    }
                }
                if b.early_exit_rank.is_some() && !out.iter().any(|&(t, _)| t == self.m.close) {
                    // A zero bias: the step samples on the host, where `probe_close` reads the row.
                    out.push((self.m.close, 0.0));
                }
            }
            _ => {}
        }
    }

    /// `(close id, rank)` when this step should close the block if the close marker's logit ranks
    /// below `rank` (a line end inside a trace, `early_exit_rank` set). The host sampler asks.
    pub fn probe_close(&self) -> Option<(u32, u32)> {
        let r = self.lim.bias.early_exit_rank?;
        (self.phase == Phase::Trace && self.line_end && self.forcing.is_none())
            .then_some((self.m.close, r))
    }

    /// With `stop_blank_line`: the committed reply token's text; the byte offset in it where a
    /// blank line after spoken reply text begins, which ends the request there.
    pub fn blank_line(&mut self, delta: &str) -> Option<usize> {
        if !self.lim.stop_blank_line || self.framing || self.phase != Phase::Reply {
            return None;
        }
        for (i, ch) in delta.char_indices() {
            match ch {
                '\n' if self.reply_nl && self.reply_spoke => return Some(i),
                '\n' => self.reply_nl = true,
                ' ' | '\t' | '\r' => {}
                _ => {
                    self.reply_nl = false;
                    self.reply_spoke = true;
                }
            }
        }
        None
    }

    /// After `commit`, with the committed output so far: arm forcing for the next token when the
    /// block is due and `utf8_pending` (the last token left an incomplete codepoint) is false. A
    /// forced sequence the model already began is resumed, not repeated.
    pub fn arm(&mut self, out_ids: &[u32], utf8_pending: bool) {
        if !self.due || utf8_pending || self.phase != Phase::Trace || self.forcing.is_some() {
            return;
        }
        self.due = false;
        let f = &self.lim.forced_end;
        let k = (1..f.len())
            .rev()
            .find(|&k| out_ids.ends_with(&f[..k]))
            .unwrap_or(0);
        self.forcing = Some(k);
        self.forced += 1;
    }

    /// How many tokens a chained path may produce before the host must see one: a replacement
    /// can only land on the last of them. `usize::MAX` when nothing can be replaced.
    pub fn chain_left(&self) -> usize {
        if self.forcing.is_some() || self.due || self.steer_window() {
            return 1;
        }
        let n = match (self.lim.budget, self.phase) {
            (None, _) if self.cut.is_none() => usize::MAX,
            (b, Phase::Trace) => {
                let end = b.unwrap_or(u32::MAX).min(self.cut.unwrap_or(u32::MAX)) as usize;
                end.saturating_sub(self.trace_n as usize) + 1
            }
            // The open marker, the block's budget, then the first forced id.
            (b, Phase::Deciding | Phase::Reply) => b.map_or(usize::MAX, |b| b as usize + 2),
        };
        if self.lim.live.is_some() {
            n.min(LIVE_CHAIN as usize)
        } else {
            n
        }
    }

    /// A steering bias may apply to the next token, set only once the host has seen this one (a
    /// line end, a close): the device must not have computed that token by its own argmax already.
    fn steer_window(&self) -> bool {
        let b = &self.lim.bias;
        match self.phase {
            // `early_exit_rank` and the first reply tokens follow any line end or close.
            Phase::Deciding | Phase::Trace
                if b.early_exit_rank.is_some() || !b.reply_first.is_empty() =>
            {
                true
            }
            Phase::Trace => b
                .close_ramp
                .zip(self.lim.budget)
                .is_some_and(|((k, _), budget)| self.trace_n + 1 >= budget.saturating_sub(k)),
            Phase::Reply => self.reply_n < 2 && !b.reply_first.is_empty(),
            Phase::Deciding => false,
        }
    }

    pub fn report(&self) -> ThinkReport {
        ThinkReport {
            blocks: self.blocks,
            forced: self.forced,
            grace_tokens: self.grace_spent,
            guard_moved: false,
        }
    }

    /// Whether the generation is past its last trace block (the first reply token was committed).
    pub fn replying(&self) -> bool {
        self.phase == Phase::Reply && self.forcing.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: u32 = 100;
    const CLOSE: u32 = 101;
    const NL: u32 = 10;

    fn ctl(
        budget: Option<u32>,
        cap: Option<u32>,
        forced: &[u32],
        grace: u32,
        opened: bool,
    ) -> ThinkCtl {
        let m = Arc::new(ThinkMarkers {
            open: OPEN,
            close: CLOSE,
            newline: vec![NL],
        });
        ThinkCtl::new(
            m,
            ThinkLimits {
                budget,
                reply_cap: cap,
                forced_end: forced.into(),
                grace,
                stop_blank_line: false,
                live: None,
                bias: ThinkBias::default(),
                spec_forced: false,
            },
            opened,
        )
    }

    /// Drive `sampled` through the controller as the mux does; the committed ids and whether the
    /// request was capped.
    fn run(t: &mut ThinkCtl, sampled: &[u32]) -> (Vec<u32>, bool) {
        let mut out = Vec::new();
        for &s in sampled {
            let c = t.commit(s);
            out.push(c.token);
            if c.capped {
                return (out, true);
            }
            t.arm(&out, false);
        }
        (out, false)
    }

    #[test]
    fn the_budget_forces_the_end_sequence() {
        let mut t = ctl(Some(3), None, &[7, 8, CLOSE], 0, true);
        let (out, _) = run(&mut t, &[1, 2, 3, 4, 5, 6, 9]);
        assert_eq!(out, [1, 2, 3, 7, 8, CLOSE, 9]);
        assert_eq!((t.forced, t.blocks), (1, 1));
    }

    #[test]
    fn a_model_that_closes_itself_is_untouched() {
        let mut t = ctl(Some(10), None, &[7, CLOSE], 0, true);
        let (out, _) = run(&mut t, &[1, CLOSE, 5, 6]);
        assert_eq!(out, [1, CLOSE, 5, 6]);
        assert_eq!(t.forced, 0);
    }

    #[test]
    fn a_model_opened_trace_counts_from_the_token_after_the_marker() {
        let mut t = ctl(Some(2), None, &[CLOSE], 0, false);
        let (out, _) = run(&mut t, &[OPEN, 1, 2, 3, 4]);
        assert_eq!(out, [OPEN, 1, 2, CLOSE, 4]);
    }

    #[test]
    fn the_reply_cap_counts_only_reply_tokens() {
        let mut t = ctl(None, Some(2), &[CLOSE], 0, false);
        assert_eq!(run(&mut t, &[1, 2, 3]), (vec![1, 2], true));
        let mut t = ctl(None, Some(2), &[CLOSE], 0, true);
        assert_eq!(
            run(&mut t, &[1, 2, 3, 4, CLOSE, 5, 6, 7]),
            (vec![1, 2, 3, 4, CLOSE, 5, 6], true)
        );
    }

    #[test]
    fn a_sampled_id_equal_to_the_forced_one_is_not_a_replacement() {
        let mut t = ctl(Some(1), None, &[7, CLOSE], 0, true);
        t.commit(1);
        t.arm(&[1], false);
        assert_eq!(
            t.commit(7),
            Commit {
                token: 7,
                replaced: false,
                capped: false
            }
        );
        assert!(t.commit(9).replaced);
    }

    #[test]
    fn a_second_block_gets_a_fresh_budget() {
        let mut t = ctl(Some(2), None, &[CLOSE], 0, true);
        let (out, _) = run(&mut t, &[1, CLOSE, 5, OPEN, 1, 2, 3, 6]);
        assert_eq!(out, [1, CLOSE, 5, OPEN, 1, 2, CLOSE, 6]);
        assert_eq!((t.blocks, t.forced), (2, 1));
    }

    #[test]
    fn a_partially_written_end_sequence_is_resumed() {
        let mut t = ctl(Some(3), None, &[7, 8, CLOSE], 0, true);
        let (out, _) = run(&mut t, &[1, 2, 7, 4, 5, 6]);
        assert_eq!(out, [1, 2, 7, 8, CLOSE, 6]);
    }

    #[test]
    fn a_blank_line_after_reply_text_ends_the_reply() {
        let mut t = ctl(None, None, &[CLOSE], 0, true);
        t.lim.stop_blank_line = true;
        t.commit(1);
        assert_eq!(t.blank_line("plan\n\n"), None, "trace text");
        t.commit(CLOSE);
        assert_eq!(t.blank_line("\n\n"), None, "the close marker's own text");
        t.commit(2);
        assert_eq!(t.blank_line("\n \n"), None, "before any reply text");
        t.commit(3);
        assert_eq!(t.blank_line("Theek hai.\n"), None);
        t.commit(4);
        assert_eq!(t.blank_line(" \nAap"), Some(1));
    }

    /// A live end request cuts the block within `LIVE_CHAIN` tokens, without a budget, and only
    /// that block.
    #[test]
    fn a_live_reasoning_end_cuts_the_current_block() {
        let request: Arc<str> = "req-live-1".into();
        let mut t = ctl(None, None, &[7, CLOSE], 0, true);
        t.lim.live = Some(live_register(&request));
        assert_eq!(t.chain_left(), LIVE_CHAIN as usize);
        let (out, _) = run(&mut t, &[1, 2]);
        assert_eq!(out, [1, 2]);
        assert!(live_end("req-live-1"));
        assert!(!live_end("req-unknown"));
        let mut sampled = vec![3; LIVE_CHAIN as usize + 2];
        sampled.extend([9, 9]);
        let (out, _) = run(&mut t, &sampled);
        let n = LIVE_CHAIN as usize + 1;
        assert_eq!(out[..n], vec![3; n][..], "the model's tokens up to the cut");
        assert_eq!(out[n..], [7, CLOSE, 9]);
        assert_eq!(t.forced, 1);
        drop(t);
        assert!(!live_end("req-live-1"), "gone with its controller");
    }

    #[test]
    fn steering_biases_follow_the_phase() {
        let mut t = ctl(Some(6), None, &[CLOSE], 0, true);
        t.lim.bias = ThinkBias {
            base: vec![(42, 1.0)],
            reply_first: vec![(50, -10.0)],
            ban_reopen: true,
            close_ramp: Some((2, 6.0)),
            early_exit_rank: Some(3),
        };
        let mut b = Vec::new();
        let step = |t: &mut ThinkCtl, s: u32, b: &mut Vec<(u32, f32)>| {
            t.commit(s);
            t.next_bias(b);
            b.clone()
        };
        assert_eq!(
            step(&mut t, 1, &mut b),
            [(42, 1.0)],
            "mid-line: the request's own"
        );
        assert_eq!(
            step(&mut t, NL, &mut b),
            [(42, 1.0), (CLOSE, 0.0)],
            "a line end probes"
        );
        assert_eq!(t.probe_close(), Some((CLOSE, 3)));
        assert_eq!(step(&mut t, 1, &mut b), [(42, 1.0)]);
        assert_eq!(t.probe_close(), None);
        assert_eq!(
            step(&mut t, NL, &mut b),
            [(42, 1.0), (CLOSE, 3.0)],
            "the ramp starts"
        );
        assert_eq!(step(&mut t, NL, &mut b), [(42, 1.0), (CLOSE, 6.0)]);
        t.commit(CLOSE);
        t.next_bias(&mut b);
        assert_eq!(
            b,
            [(42, 1.0), (50, -10.0), (OPEN, -100.0)],
            "first reply tokens"
        );
        assert_eq!(t.probe_close(), None);
        step(&mut t, 5, &mut b);
        assert_eq!(step(&mut t, 6, &mut b), [(42, 1.0), (OPEN, -100.0)]);
        assert!(t.steering());
        assert!(!ctl(Some(4), None, &[CLOSE], 0, true).steering());
        let mut t = ctl(Some(64), None, &[CLOSE], 0, true);
        t.lim.bias.early_exit_rank = Some(1);
        assert_eq!(
            t.chain_left(),
            1,
            "an early exit can land on any trace token"
        );
        // the ramp window and the first reply tokens are host-fed; outside them chains run
        let mut t = ctl(Some(10), None, &[CLOSE], 0, true);
        t.lim.bias.close_ramp = Some((4, 5.0));
        assert_eq!(t.chain_left(), 11);
        for _ in 0..5 {
            t.commit(1);
        }
        assert_eq!(t.chain_left(), 1, "trace_n 5: the ramp starts at 6");
        let mut t = ctl(Some(10), None, &[CLOSE], 0, true);
        t.lim.bias.reply_first = vec![(50, -5.0)];
        assert_eq!(t.chain_left(), 1, "a close could come next");
        t.commit(CLOSE);
        t.commit(5);
        t.commit(6);
        assert_eq!(t.chain_left(), 12, "past the first two reply tokens");
    }

    /// The verify's span for a forcing row: `[pending, forced ids…]`. Whatever the target accepts
    /// commits as the forced ids; the correction token is replaced by the next forced id.
    #[test]
    fn forced_ids_drafted_through_a_verify_commit_as_forced() {
        let forced = [7, 8, 9, CLOSE];
        let mut t = ctl(Some(1), None, &forced, 0, true);
        t.lim.spec_forced = true;
        t.commit(1);
        t.arm(&[1], false);
        assert_eq!(t.forced_ahead(), Some(&forced[..]));
        // the target accepts 7 and 8, then wants 5 where 9 was drafted
        let emitted = [7, 8, 5];
        let committed: Vec<u32> = emitted.iter().map(|&s| t.commit(s).token).collect();
        assert_eq!(committed, [7, 8, 9]);
        assert_eq!(t.forced_ahead(), Some(&[CLOSE][..]));
        // all accepted, the bonus is the model's first reply token
        assert_eq!(t.commit(CLOSE).token, CLOSE);
        assert_eq!(t.forced_ahead(), None);
        assert_eq!(t.commit(42).token, 42);
    }

    #[test]
    fn grace_waits_for_a_line_end() {
        let mut t = ctl(Some(2), None, &[CLOSE], 4, true);
        let (out, _) = run(&mut t, &[1, 2, 3, 4, NL, 5, 6]);
        assert_eq!(out, [1, 2, 3, 4, NL, CLOSE, 6]);
        assert_eq!(t.grace_spent, 3);
        let mut t = ctl(Some(2), None, &[CLOSE], 4, true);
        let (out, _) = run(&mut t, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(out, [1, 2, 3, 4, 5, 6, CLOSE, 8]);
    }

    #[test]
    fn forcing_waits_for_a_complete_codepoint() {
        let mut t = ctl(Some(1), None, &[CLOSE], 0, true);
        t.commit(1);
        t.arm(&[1], true);
        assert_eq!(
            t.commit(2).token,
            2,
            "the codepoint's next byte is the model's"
        );
        t.arm(&[1, 2], false);
        assert_eq!(t.commit(3).token, CLOSE);
    }

    /// Chained paths fix a quantum of `chain_left()` tokens at its start; every replacement must
    /// land on a quantum's last token.
    #[test]
    fn replacements_land_only_on_a_quantums_last_token() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n
        };
        for _ in 0..2000 {
            let budget = 1 + rnd(6) as u32;
            let grace = rnd(3) as u32;
            let forced: Vec<u32> = (0..rnd(3)).map(|i| 20 + i as u32).chain([CLOSE]).collect();
            let cap = (rnd(2) == 0).then(|| 1 + rnd(8) as u32);
            let budget = (rnd(4) != 0).then_some(budget);
            let mut t = ctl(budget, cap, &forced, grace, rnd(2) == 0);
            let live = Arc::new(AtomicBool::new(false));
            if rnd(2) == 0 {
                t.lim.live = Some(Arc::clone(&live));
            }
            let mut out = Vec::new();
            let mut left = 0usize;
            'gen: for _ in 0..200 {
                if left == 0 {
                    left = t.chain_left().min(1 + rnd(9) as usize);
                }
                let s = match rnd(10) {
                    0 => OPEN,
                    1 => CLOSE,
                    2 => NL,
                    _ => 1 + rnd(5) as u32,
                };
                if rnd(16) == 0 {
                    live.store(true, Relaxed);
                }
                let c = t.commit(s);
                assert!(!c.replaced || left == 1, "replacement inside a quantum");
                left -= 1;
                out.push(c.token);
                if c.capped {
                    break 'gen;
                }
                t.arm(&out, rnd(8) == 0);
            }
        }
    }
}

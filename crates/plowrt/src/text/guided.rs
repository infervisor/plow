//! §L Structured / guided decoding — grammar/JSON/regex FSM logit masking.
//!
//! Compile the request's schema to an FSM; per step, mask disallowed tokens
//! before sampling (`text::sample::sample` takes the mask). Skeleton: a trait +
//! a token-set FSM; production uses a compressed FSM (xgrammar-style) to keep
//! masking cheap over large vocabularies.

/// A stepwise decoding constraint.
pub trait GuidedDecoder: Send {
    /// Boolean mask over the vocabulary: `true` = token allowed at this step.
    fn allowed_mask(&self, vocab: usize) -> Vec<bool>;
    /// Advance the FSM by the chosen token.
    fn advance(&mut self, token: u32);
    /// Whether the constraint has reached an accepting terminal state.
    fn done(&self) -> bool;
}

/// The simplest useful constraint: restrict output to a fixed allowed token set
/// (e.g. an enum of literals). Enough to exercise the mask path.
pub struct AllowedSet {
    allowed: Vec<u32>,
    steps: usize,
    max_steps: usize,
}

impl AllowedSet {
    pub fn new(allowed: Vec<u32>, max_steps: usize) -> Self {
        AllowedSet {
            allowed,
            steps: 0,
            max_steps,
        }
    }
}

impl GuidedDecoder for AllowedSet {
    fn allowed_mask(&self, vocab: usize) -> Vec<bool> {
        let mut m = vec![false; vocab];
        for &t in &self.allowed {
            if (t as usize) < vocab {
                m[t as usize] = true;
            }
        }
        m
    }

    fn advance(&mut self, _token: u32) {
        self.steps += 1;
    }

    fn done(&self) -> bool {
        self.steps >= self.max_steps
    }
}

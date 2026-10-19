//! §L Speculative decoding — a special case of running multiple models.
//!
//! ## Design Philosophy (plow runtime)
//!
//! Speculative decode is **not** a special scheduler mode — it is a design
//! artifact of the multi-model workflow system. Any two models can participate:
//! a draft (small/fast) and a target (large/accurate). The draft proposes `k`
//! tokens; the target verifies them in one batched forward pass. This reuses the
//! same zero-copy Pipeline dataflow as vision→decoder or encoder→decoder.
//!
//! The draft is an ordinary [`ModelBundle`] in the registry (a small model,
//! EAGLE head, ngram predictor); the target is another bundle. The pipeline
//! chains them: the draft's output buffer aliases the target's input slot —
//! same device-packet zero-copy mechanism as all inter-stage transfers. The
//! accepted-prefix length returns over §OOB (`SpecVerdict`); the `CONDITIONAL`
//! packet flag commits the accepted tokens.
//!
//! Different speculative strategies (EAGLE, Medusa, ngram, Lookahead, etc.) all
//! reduce to "which bundle plays the draft role and how many tokens `k` it
//! proposes." The workflow customization is config-only — no code changes.

/// Longest-prefix acceptance: how many of the draft's proposed tokens match the
/// target's argmax verification. Rejected tokens are dropped; the loop repeats
/// from the first divergence.
pub fn accepted_prefix(draft: &[u32], verified: &[u32]) -> usize {
    draft
        .iter()
        .zip(verified.iter())
        .take_while(|(d, v)| d == v)
        .count()
}

/// How many tokens the step commits: the accepted prefix plus the one bonus
/// token the target itself produces past the accepted run.
pub fn committed_tokens(accepted: usize) -> usize {
    accepted + 1
}

/// Most draft tokens one verify span carries (`SpecVerdict::tokens` holds `k + 1`).
pub const SPEC_KMAX: usize = 15;
/// `in.rowslot` entry flag: the row reads its slot's KV but stores none (verify padding).
pub const SPEC_NO_STORE: u32 = 1 << 31;
/// `SpecVerdict::flags`: emission stopped at a stop token.
pub const VERDICT_STOP: u32 = 1;

/// One verify span's outcome, written by `plow_spec_accept` (device) or [`accept_greedy`] (host
/// golden). Layout is shared with `runtime/{nvidia/sample_sm120.cu,amd/spec_accept.hip}`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SpecVerdict {
    /// The slot generation the span was launched for; a stale verdict is dropped.
    pub req_gen: u32,
    /// Draft tokens accepted and emitted.
    pub n_acc: u16,
    /// Tokens emitted = KV rows committed (the span's first `n_emit` inputs).
    pub n_emit: u16,
    /// Verify row whose hidden state produced the last emitted token.
    pub hrow: u32,
    pub flags: u32,
    pub tokens: [u32; SPEC_KMAX + 1],
}

/// Device span record for `plow_spec_accept`: rows `row0..=row0 + k` of the verify launch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SpecSpanDev {
    pub row0: u32,
    pub k: u32,
    pub req_gen: u32,
    pub _pad: u32,
}

/// Greedy acceptance of one span. `inputs` = the span's tokens (pending token, then `k` drafts);
/// `argmax` = the verify rows' argmax, same length. Draft `j` survives while
/// `inputs[j + 1] == argmax[j]`; the emission is the accepted drafts plus `argmax[n]`, cut after
/// the first stop token.
pub fn accept_greedy(inputs: &[u32], argmax: &[u32], row0: u32, req_gen: u32, stop: &[u32]) -> SpecVerdict {
    let k = inputs.len() - 1;
    assert!(k <= SPEC_KMAX && argmax.len() == inputs.len());
    let n = accepted_prefix(&inputs[1..], argmax);
    let mut v = SpecVerdict {
        req_gen,
        ..Default::default()
    };
    for (j, &t) in inputs[1..=n].iter().chain(std::iter::once(&argmax[n])).enumerate() {
        v.tokens[j] = t;
        v.n_emit = j as u16 + 1;
        if stop.contains(&t) {
            v.flags |= VERDICT_STOP;
            break;
        }
    }
    v.n_acc = v.n_emit.min(n as u16);
    v.hrow = row0 + u32::from(v.n_emit) - 1;
    v
}

/// Prompt-lookup drafter: find the most recent earlier occurrence of the history's last `n`
/// tokens (longest `n` in `max_n..=min_n` first) and propose up to `k` tokens that followed it.
pub fn ngram_propose(history: &[u32], k: usize, min_n: usize, max_n: usize, out: &mut Vec<u32>) {
    out.clear();
    if k == 0 {
        return;
    }
    let len = history.len();
    for n in (min_n.max(1)..=max_n).rev() {
        if n >= len {
            continue;
        }
        let tail = &history[len - n..];
        if let Some(start) = (0..len - n).rev().find(|&i| &history[i..i + n] == tail) {
            let from = start + n;
            out.extend_from_slice(&history[from..(from + k).min(len)]);
            return;
        }
    }
}

/// Configuration for a speculative decode workflow (a multi-model pipeline).
#[derive(Clone, Debug)]
pub struct SpecConfig {
    /// Registry slug for the draft model.
    pub draft_slug: String,
    /// Registry slug for the target (verifier) model.
    pub target_slug: String,
    /// Number of draft tokens to propose per speculation step.
    pub k: usize,
    /// Maximum batch size for the target's verification pass.
    pub verify_batch: usize,
}

impl Default for SpecConfig {
    fn default() -> Self {
        SpecConfig {
            draft_slug: String::new(),
            target_slug: String::new(),
            k: 5,
            verify_batch: 8,
        }
    }
}

/// A two-model speculative decoder. This is a concrete workflow built on top of
/// the multi-model pipeline — each model has its own `ModelMux`, and the
/// speculative orchestrator coordinates the draft→verify→accept cycle.
///
/// Per tick:
/// 1. Run draft model for `k` steps (fast, small model)
/// 2. Feed all `k` proposed tokens to the target in one batched pass
/// 3. `accepted_prefix()` determines how many survive
/// 4. Roll back KV for rejected suffix (release pages past accepted_prefix)
/// 5. Commit the accepted tokens + 1 bonus from the target
pub struct SpeculativeWorkflow {
    pub config: SpecConfig,
    /// Draft tokens proposed in the current step.
    draft_tokens: Vec<u32>,
    /// Verified tokens from the target.
    verified_tokens: Vec<u32>,
}

impl SpeculativeWorkflow {
    pub fn new(config: SpecConfig) -> Self {
        SpeculativeWorkflow {
            config,
            draft_tokens: Vec::new(),
            verified_tokens: Vec::new(),
        }
    }

    /// Record draft proposals. Called after the draft model produces `k` tokens.
    pub fn set_draft(&mut self, tokens: &[u32]) {
        self.draft_tokens.clear();
        self.draft_tokens.extend_from_slice(tokens);
    }

    /// Record target verification. Called after the target model verifies.
    pub fn set_verified(&mut self, tokens: &[u32]) {
        self.verified_tokens.clear();
        self.verified_tokens.extend_from_slice(tokens);
    }

    /// Resolve the acceptance: returns (accepted_count, bonus_token).
    /// The caller commits `accepted_count` draft tokens plus the bonus.
    pub fn resolve(&self) -> (usize, Option<u32>) {
        let accepted = accepted_prefix(&self.draft_tokens, &self.verified_tokens);
        let bonus = self.verified_tokens.get(accepted).copied();
        (accepted, bonus)
    }

    /// Number of KV pages to roll back (tokens to undo past the accepted prefix).
    pub fn rollback_count(&self) -> usize {
        let accepted = accepted_prefix(&self.draft_tokens, &self.verified_tokens);
        self.draft_tokens.len().saturating_sub(accepted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_acceptance() {
        let draft = [10, 20, 30, 40, 50];
        let verified = [10, 20, 30, 40, 50];
        assert_eq!(accepted_prefix(&draft, &verified), 5);
        assert_eq!(committed_tokens(5), 6);
    }

    #[test]
    fn partial_acceptance() {
        let draft = [10, 20, 99, 40, 50];
        let verified = [10, 20, 30, 40, 50];
        assert_eq!(accepted_prefix(&draft, &verified), 2);
        assert_eq!(committed_tokens(2), 3);
    }

    #[test]
    fn greedy_accept_cases() {
        // all accepted: k drafts + bonus
        let v = accept_greedy(&[5, 1, 2, 3], &[1, 2, 3, 9], 4, 7, &[]);
        assert_eq!((v.n_acc, v.n_emit, v.hrow, v.req_gen), (3, 4, 7, 7));
        assert_eq!(&v.tokens[..4], &[1, 2, 3, 9]);
        // first draft rejected: the correction only
        let v = accept_greedy(&[5, 1, 2], &[8, 2, 4], 0, 0, &[]);
        assert_eq!((v.n_acc, v.n_emit, v.hrow), (0, 1, 0));
        assert_eq!(v.tokens[0], 8);
        // k = 0 is a plain decode row
        let v = accept_greedy(&[5], &[6], 3, 0, &[]);
        assert_eq!((v.n_acc, v.n_emit, v.hrow, v.tokens[0]), (0, 1, 3, 6));
        // stop inside the accepted drafts cuts emission there
        let v = accept_greedy(&[5, 1, 2, 3], &[1, 2, 3, 9], 0, 0, &[2]);
        assert_eq!((v.n_acc, v.n_emit, v.flags, v.hrow), (2, 2, VERDICT_STOP, 1));
        // stop as the correction token
        let v = accept_greedy(&[5, 1, 2], &[1, 7, 4], 0, 0, &[7]);
        assert_eq!((v.n_acc, v.n_emit, v.flags), (1, 2, VERDICT_STOP));
    }

    #[test]
    fn ngram_lookup() {
        let mut out = Vec::new();
        ngram_propose(&[1, 2, 3, 4, 9, 1, 2, 3], 3, 1, 4, &mut out);
        assert_eq!(out, [4, 9, 1]);
        ngram_propose(&[1, 2, 3], 3, 1, 4, &mut out);
        assert!(out.is_empty());
        ngram_propose(&[7, 8, 7], 5, 1, 4, &mut out);
        assert_eq!(out, [8, 7]);
    }

    #[test]
    fn verdict_layout() {
        assert_eq!(std::mem::size_of::<SpecVerdict>(), 16 + 4 * (SPEC_KMAX + 1));
        assert_eq!(std::mem::size_of::<SpecSpanDev>(), 16);
    }

    #[test]
    fn workflow_resolve() {
        let mut wf = SpeculativeWorkflow::new(SpecConfig {
            k: 4,
            ..Default::default()
        });
        wf.set_draft(&[1, 2, 3, 4]);
        wf.set_verified(&[1, 2, 3, 99]);
        let (accepted, bonus) = wf.resolve();
        assert_eq!(accepted, 3);
        assert_eq!(bonus, Some(99));
        assert_eq!(wf.rollback_count(), 1);
    }
}

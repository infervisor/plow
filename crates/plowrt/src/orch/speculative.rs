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

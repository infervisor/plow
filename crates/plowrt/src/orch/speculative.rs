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
pub fn accept_greedy(
    inputs: &[u32],
    argmax: &[u32],
    row0: u32,
    req_gen: u32,
    stop: &[u32],
) -> SpecVerdict {
    let k = inputs.len() - 1;
    assert!(k <= SPEC_KMAX && argmax.len() == inputs.len());
    let n = accepted_prefix(&inputs[1..], argmax);
    let mut v = SpecVerdict {
        req_gen,
        ..Default::default()
    };
    for (j, &t) in inputs[1..=n]
        .iter()
        .chain(std::iter::once(&argmax[n]))
        .enumerate()
    {
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

/// Sampled acceptance of one span with a deterministic drafter (`temperature > 0`, penalties,
/// bias). `sample(j)` is the token plain decode would sample from verify row `j`: its logits under
/// the row's sampling transform, with history `.. ++ inputs[1..=j]` and the draw of that output
/// position. Draft `j` survives while it equals that sample; the emission is the accepted drafts
/// plus the first differing sample (or the bonus sample after all `k`). With a one-hot draft this
/// is standard rejection sampling (accept `x` w.p. `p(x)`, else draw from `p` without `x`), and
/// with plain decode's own draws it emits exactly plain decode's tokens. Rows past the first
/// mismatch or stop are never sampled.
pub fn accept_sampled<E>(
    inputs: &[u32],
    mut sample: impl FnMut(usize) -> Result<u32, E>,
    row0: u32,
    req_gen: u32,
    stop: &[u32],
) -> Result<SpecVerdict, E> {
    let k = inputs.len() - 1;
    assert!(k <= SPEC_KMAX);
    let mut rows = [0u32; SPEC_KMAX + 1];
    for j in 0..=k {
        rows[j] = sample(j)?;
        if j == k || rows[j] != inputs[j + 1] || stop.contains(&rows[j]) {
            break;
        }
    }
    Ok(accept_greedy(inputs, &rows[..=k], row0, req_gen, stop))
}

/// Plain decode's host draw for verify row `j` (`drafts` = the span's drafts before it): the
/// penalties and bias over `history ++ drafts`, then [`sample`](crate::text::sample::sample).
pub fn host_row_sample(
    logits: &mut [f32],
    params: &crate::text::sample::SamplingParams,
    history: &[u32],
    drafts: &[u32],
    rng01: f32,
    prior: &mut Vec<u32>,
) -> u32 {
    prior.clear();
    if params.repetition_penalty != 1.0
        || params.presence_penalty != 0.0
        || params.frequency_penalty != 0.0
    {
        prior.extend_from_slice(history);
        prior.extend_from_slice(drafts);
    }
    crate::text::sample::apply_penalties(logits, prior, params);
    crate::text::sample::sample(logits, params, None, rng01)
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
        assert_eq!(
            (v.n_acc, v.n_emit, v.flags, v.hrow),
            (2, 2, VERDICT_STOP, 1)
        );
        // stop as the correction token
        let v = accept_greedy(&[5, 1, 2], &[1, 7, 4], 0, 0, &[7]);
        assert_eq!((v.n_acc, v.n_emit, v.flags), (1, 2, VERDICT_STOP));
    }

    use crate::text::sample::SamplingParams;
    use std::collections::HashMap;

    const V: usize = 6;
    const STOP: u32 = 5;

    fn mix(mut h: u64) -> u64 {
        h ^= h >> 31;
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^ (h >> 29)
    }

    /// A history-dependent toy target: logits from the last three tokens and the length.
    fn target(history: &[u32]) -> Vec<f32> {
        let mut h = history.len() as u64;
        for &t in history.iter().rev().take(3) {
            h = mix(h ^ u64::from(t + 1));
        }
        (0..V as u64)
            .map(|i| (mix(h ^ (i * 0x51)) % 1000) as f32 / 250.0 - 2.0)
            .collect()
    }

    /// A deterministic drafter that is right about half the time.
    fn draft(history: &[u32]) -> u32 {
        let mut l = target(history);
        let h = mix(history.iter().fold(7u64, |a, &t| mix(a ^ u64::from(t))));
        for (i, x) in l.iter_mut().enumerate() {
            *x += ((mix(h ^ i as u64) % 100) as f32) / 40.0;
        }
        crate::text::sample::argmax(&l)
    }

    struct Rng(u64);
    impl Rng {
        fn unit(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            (mix(self.0) >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// `n` tokens (or up to STOP) of plain decode; `draw(out)` is the uniform for the next token.
    fn plain(
        prompt: &[u32],
        p: &SamplingParams,
        n: usize,
        mut draw: impl FnMut(&[u32]) -> f32,
    ) -> Vec<u32> {
        let (mut out, mut prior) = (Vec::new(), Vec::new());
        while out.len() < n && out.last() != Some(&STOP) {
            let hist: Vec<u32> = prompt.iter().chain(&out).copied().collect();
            let u = draw(&out);
            let t = host_row_sample(&mut target(&hist), p, &out, &[], u, &mut prior);
            out.push(t);
        }
        out
    }

    /// The same request decoded speculatively with depth `k`; `draw(out, j)` is verify row j's
    /// uniform given the emitted tokens and the span's drafts.
    fn spec(
        prompt: &[u32],
        p: &SamplingParams,
        n: usize,
        k: usize,
        mut draw: impl FnMut(&[u32], &[u32]) -> f32,
    ) -> Vec<u32> {
        let (mut out, mut prior) = (Vec::<u32>::new(), Vec::new());
        while out.len() < n && out.last() != Some(&STOP) {
            let mut inputs = vec![*out.last().unwrap_or(prompt.last().unwrap())];
            let mut hist: Vec<u32> = prompt.iter().chain(&out).copied().collect();
            for _ in 0..k.min(n - out.len() - 1) {
                let d = draft(&hist);
                inputs.push(d);
                hist.push(d);
            }
            let v = accept_sampled::<()>(
                &inputs,
                |j| {
                    let drafts = &inputs[1..=j];
                    let hist: Vec<u32> = prompt.iter().chain(&out).chain(drafts).copied().collect();
                    let u = draw(&out, drafts);
                    Ok(host_row_sample(
                        &mut target(&hist),
                        p,
                        &out,
                        drafts,
                        u,
                        &mut prior,
                    ))
                },
                0,
                0,
                &[STOP],
            )
            .unwrap();
            assert!(v.n_emit >= 1 && usize::from(v.n_emit) <= inputs.len());
            out.extend_from_slice(&v.tokens[..usize::from(v.n_emit)]);
        }
        out
    }

    fn settings() -> Vec<(&'static str, SamplingParams)> {
        let t = |temperature| SamplingParams {
            temperature,
            ..Default::default()
        };
        vec![
            ("t0.7", t(0.7)),
            (
                "t1.0 top_p0.8",
                SamplingParams {
                    top_p: 0.8,
                    ..t(1.0)
                },
            ),
            ("t0.9 top_k3", SamplingParams { top_k: 3, ..t(0.9) }),
            (
                "t1.2 min_p0.15",
                SamplingParams {
                    min_p: 0.15,
                    ..t(1.2)
                },
            ),
            (
                "t0.8 penalties+bias",
                SamplingParams {
                    repetition_penalty: 1.4,
                    presence_penalty: 0.5,
                    frequency_penalty: 0.3,
                    logit_bias: vec![(2, 0.7), (4, -1.0)],
                    ..t(0.8)
                },
            ),
            (
                "greedy penalties",
                SamplingParams {
                    repetition_penalty: 1.3,
                    frequency_penalty: 0.4,
                    ..t(0.0)
                },
            ),
        ]
    }

    /// Two-sample chi-square statistic and its degrees of freedom.
    fn chi2(a: &HashMap<Vec<u32>, u32>, b: &HashMap<Vec<u32>, u32>) -> (f64, usize) {
        let mut cells: Vec<&Vec<u32>> = a.keys().chain(b.keys()).collect();
        cells.sort();
        cells.dedup();
        let stat = cells
            .iter()
            .map(|c| {
                let (x, y) = (
                    f64::from(*a.get(*c).unwrap_or(&0)),
                    f64::from(*b.get(*c).unwrap_or(&0)),
                );
                (x - y).powi(2) / (x + y)
            })
            .sum();
        (stat, cells.len().saturating_sub(1))
    }

    fn histogram(
        n: usize,
        mut f: impl FnMut(&mut Rng) -> Vec<u32>,
        seed: u64,
    ) -> HashMap<Vec<u32>, u32> {
        let mut rng = Rng(seed);
        let mut h = HashMap::new();
        for _ in 0..n {
            *h.entry(f(&mut rng)).or_insert(0) += 1;
        }
        h
    }

    /// Emitted 3-token sequences: speculative decoding with independent per-row draws matches
    /// plain sampling in distribution (chi-square far below the 5-sigma bound, total variation
    /// within sampling noise), at several depths; a target off by 20% in temperature is rejected.
    #[test]
    fn sampled_spec_is_lossless_in_distribution() {
        const N: usize = 60_000;
        let prompt = [1u32, 3, 2];
        for (name, p) in settings() {
            let reference = histogram(N, |r| plain(&prompt, &p, 3, |_| r.unit()), 11);
            for k in [1, 3] {
                let got = histogram(
                    N,
                    |r| spec(&prompt, &p, 3, k, |_, _| r.unit()),
                    29 + k as u64,
                );
                let (stat, df) = chi2(&reference, &got);
                let bound = df as f64 + 5.0 * (2.0 * df as f64).sqrt();
                let tv: f64 = reference
                    .keys()
                    .chain(got.keys())
                    .collect::<std::collections::HashSet<_>>()
                    .iter()
                    .map(|c| {
                        (f64::from(*reference.get(*c).unwrap_or(&0))
                            - f64::from(*got.get(*c).unwrap_or(&0)))
                        .abs()
                    })
                    .sum::<f64>()
                    / (2.0 * N as f64);
                assert!(
                    stat <= bound.max(1.0) && tv < 0.03,
                    "{name} k{k}: chi2 {stat:.1} (df {df}, bound {bound:.1}), TV {tv:.4}"
                );
            }
            if p.temperature > 0.0 && p.top_k == 0 {
                let off = SamplingParams {
                    temperature: p.temperature * 1.2,
                    ..p.clone()
                };
                let got = histogram(N, |r| spec(&prompt, &off, 3, 3, |_, _| r.unit()), 97);
                let (stat, df) = chi2(&reference, &got);
                assert!(
                    stat > df as f64 + 5.0 * (2.0 * df as f64).sqrt(),
                    "{name}: the test cannot see a 20% temperature error (chi2 {stat:.1}, df {df})"
                );
            }
        }
    }

    /// With plain decode's own seeded draws, speculative decoding emits plain decode's tokens
    /// exactly, for every depth, setting and seed (stops included).
    #[test]
    fn sampled_spec_replays_plain_decode() {
        let prompt = [4u32, 0, 2, 2];
        for (name, p) in settings() {
            for seed in 0..200u64 {
                let s = Some(seed);
                let want = plain(&prompt, &p, 12, |out| {
                    crate::serve::seeded_unit_with(&prompt, out, out.len(), s)
                });
                for k in [1, 2, 4, 7] {
                    let got = spec(&prompt, &p, 12, k, |out, drafts| {
                        let mut d = vec![0f32; drafts.len() + 1];
                        crate::serve::seeded_units_ext(&prompt, out, drafts, out.len(), s, &mut d);
                        d[drafts.len()]
                    });
                    assert_eq!(got, want, "{name} seed {seed} k{k}");
                }
            }
        }
    }

    #[test]
    fn sampled_accept_samples_only_what_it_needs() {
        let mut seen = Vec::new();
        // Row 1 rejects draft 2: rows 2.. are never sampled.
        let v = accept_sampled::<()>(
            &[9, 1, 2, 3],
            |j| {
                seen.push(j);
                Ok([1, 4, 3, 0][j])
            },
            5,
            1,
            &[],
        )
        .unwrap();
        assert_eq!(seen, [0, 1]);
        assert_eq!(
            (v.n_acc, v.n_emit, v.hrow, &v.tokens[..2]),
            (1, 2, 6, &[1u32, 4][..])
        );
        // A sampled stop that matches its draft ends the span there: no bonus.
        seen.clear();
        let v = accept_sampled::<()>(
            &[9, 1, 7, 3],
            |j| {
                seen.push(j);
                Ok([1, 7, 3, 8][j])
            },
            0,
            0,
            &[7],
        )
        .unwrap();
        assert_eq!(seen, [0, 1]);
        assert_eq!((v.n_acc, v.n_emit, v.flags), (2, 2, VERDICT_STOP));
        // All accepted: the bonus row is sampled.
        let v = accept_sampled::<()>(&[9, 1, 2], |j| Ok([1, 2, 6][j]), 0, 0, &[]).unwrap();
        assert_eq!(
            (v.n_acc, v.n_emit, &v.tokens[..3]),
            (2, 3, &[1u32, 2, 6][..])
        );
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

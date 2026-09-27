//! §H/§L Sampler — greedy / temperature / top-k / top-p / min-p, with hooks for
//! repetition penalty and logit bias. Reads a logits row; returns a token id.
//!
//! Default decode samples on-device (only the token id comes back). This host
//! sampler is the fallback path used when logprobs / guided / beam need the full
//! logits row (§M per-token output).

use std::cell::RefCell;

use rustc_hash::FxHashMap;

/// Sampling parameters from the request.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// OpenAI `presence_penalty`: a flat subtraction applied once to any token
    /// that has already been produced. DISTINCT from `repetition_penalty`,
    /// which is multiplicative and sign-dependent — a client setting one does
    /// not get the other.
    pub presence_penalty: f32,
    /// OpenAI `frequency_penalty`: subtracted once per prior occurrence, so it
    /// scales with how often the token has been used.
    pub frequency_penalty: f32,
    /// (token, bias) additive logit adjustments.
    pub logit_bias: Vec<(u32, f32)>,
    /// OpenAI `logprobs`: report each generated token's (and its alternatives') log-probability.
    pub logprobs: Option<crate::text::logprobs::LogprobRequest>,
}

impl SamplingParams {
    /// Whether this row must be sampled on the HOST.
    ///
    /// The device sampler covers temperature/top_k/top_p/min_p but owns no
    /// per-row token history, so anything derived from what the row has already
    /// emitted — the three penalties — and any logit bias needs the full logits
    /// row downloaded. One predicate so the eligibility checks in `serve::mux`
    /// cannot drift apart from the set of knobs this struct carries.
    pub fn needs_host_logits(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || !self.logit_bias.is_empty()
            || self.logprobs.is_some()
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: Vec::new(),
            logprobs: None,
        }
    }
}

/// Reusable workspace for stochastic sampling.
///
/// Keeping this buffer across calls avoids allocating and freeing a vocab-sized
/// vector for every generated token. Callers with per-executor state can use
/// sample_with_scratch directly; sample uses one workspace per worker thread.
#[derive(Debug, Default)]
pub struct SamplerScratch {
    probs: Vec<(usize, f32)>,
}

impl SamplerScratch {
    pub fn new(vocab: usize) -> Self {
        Self {
            probs: Vec::with_capacity(vocab),
        }
    }
}

thread_local! {
    static SAMPLER_SCRATCH: RefCell<SamplerScratch> =
        RefCell::new(SamplerScratch::default());
}

/// Greedy argmax — the cheapest path (temperature 0 / top_k 1).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

thread_local! {
    /// Token -> occurrences in `prior`, reused across calls so the presence /
    /// frequency pass allocates nothing after the first row it runs on.
    static PENALTY_COUNTS: RefCell<FxHashMap<u32, u32>> = RefCell::new(FxHashMap::default());
}

/// Apply the three penalties and logit bias in place (pre-softmax).
///
/// `prior` is the row's own generated history, which is what OpenAI's
/// presence/frequency penalties are defined over.
pub fn apply_penalties(logits: &mut [f32], prior: &[u32], params: &SamplingParams) {
    if params.repetition_penalty != 1.0 {
        for &t in prior {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 {
                    *l / params.repetition_penalty
                } else {
                    *l * params.repetition_penalty
                };
            }
        }
    }
    if params.presence_penalty != 0.0 || params.frequency_penalty != 0.0 {
        PENALTY_COUNTS.with(|c| {
            let mut counts = c.borrow_mut();
            counts.clear();
            for &t in prior {
                *counts.entry(t).or_insert(0) += 1;
            }
            for (&t, &n) in counts.iter() {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l -= params.presence_penalty + params.frequency_penalty * n as f32;
                }
            }
        });
    }
    for &(t, b) in &params.logit_bias {
        if let Some(l) = logits.get_mut(t as usize) {
            *l += b;
        }
    }
}

/// Sample a token from `logits` under `params`, using `rng01` in `[0, 1)` for the
/// stochastic draw. Deterministic given `rng01` (tests pass a fixed value).
pub fn sample(logits: &[f32], params: &SamplingParams, mask: Option<&[bool]>, rng01: f32) -> u32 {
    SAMPLER_SCRATCH
        .with(|scratch| sample_with_scratch(logits, params, mask, rng01, &mut scratch.borrow_mut()))
}

/// Sample using caller-owned reusable workspace.
pub fn sample_with_scratch(
    logits: &[f32],
    params: &SamplingParams,
    mask: Option<&[bool]>,
    rng01: f32,
    scratch: &mut SamplerScratch,
) -> u32 {
    // Structured-decoding mask: forbid disallowed tokens (§L guided).
    let allowed = |i: usize| mask.map_or(true, |m| m.get(i).copied().unwrap_or(false));

    if params.temperature <= f32::EPSILON {
        // Greedy over the allowed set.
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if allowed(i) && v > best_v {
                best_v = v;
                best = i;
            }
        }
        return best as u32;
    }

    // Temperature-scaled softmax over the allowed set, then top-k/top-p/min-p
    // truncation, then inverse-CDF draw with `rng01`.
    let inv_t = 1.0 / params.temperature;
    let probs = &mut scratch.probs;
    probs.clear();
    probs.extend(
        logits
            .iter()
            .enumerate()
            .filter(|(i, _)| allowed(*i))
            .map(|(i, &v)| (i, v * inv_t)),
    );
    if probs.is_empty() {
        return 0;
    }
    let maxl = probs
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for (_, v) in probs.iter_mut() {
        *v = (*v - maxl).exp();
        sum += *v;
    }
    for (_, v) in probs.iter_mut() {
        *v /= sum;
    }
    let desc = |a: &(usize, f32), b: &(usize, f32)| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
    // Top-k first selects, then sorts only the k survivors: a full sort of a 262k vocabulary
    // costs milliseconds per row.
    if params.top_k > 0 && probs.len() > params.top_k {
        probs.select_nth_unstable_by(params.top_k - 1, desc);
        probs.truncate(params.top_k);
    }
    probs.sort_unstable_by(desc);
    if params.min_p > 0.0 {
        let thresh = probs[0].1 * params.min_p;
        probs.retain(|(_, p)| *p >= thresh);
    }
    // top-p nucleus
    if params.top_p < 1.0 {
        let mut acc = 0.0;
        let mut cut = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            acc += *p;
            if acc >= params.top_p {
                cut = i + 1;
                break;
            }
        }
        probs.truncate(cut);
    }
    let total: f32 = probs.iter().map(|(_, p)| *p).sum();
    let mut target = rng01 * total;
    for (i, p) in probs.iter() {
        target -= *p;
        if target <= 0.0 {
            return *i as u32;
        }
    }
    probs.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Classifier-free guidance over a (conditional, unconditional) logits pair, then the reference
/// sampling chain: repetition penalty, temperature, min_p, top_p.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CfgParams {
    pub cfg_weight: f32,
    pub temperature: f32,
    pub min_p: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

/// Deterministic per-request draws (splitmix64).
#[derive(Clone, Debug)]
pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        SplitMix(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// cond + w(cond - uncond), the repetition penalty once per distinct `history` token (as HF's
/// `RepetitionPenaltyLogitsProcessor`: gather / scatter), temperature, min_p, top_p, then a draw at
/// `u` in [0,1); `None` = greedy (argmax of the guided logits).
pub fn sample_cfg(
    p: &CfgParams,
    cond: &[f32],
    uncond: &[f32],
    history: impl IntoIterator<Item = u32>,
    u: Option<f32>,
    scratch: &mut Vec<f32>,
) -> u32 {
    scratch.clear();
    scratch.extend(cond.iter().zip(uncond).map(|(a, b)| a + p.cfg_weight * (a - b)));
    sample_guided(p, history, u, scratch)
}

/// [`sample_cfg`] straight from the step's little-endian bf16 logits rows: the conversion fuses
/// into the guidance pass instead of filling two f32 rows first. Same token.
pub fn sample_cfg_bf16(
    p: &CfgParams,
    cond: &[u8],
    uncond: &[u8],
    history: impl IntoIterator<Item = u32>,
    u: Option<f32>,
    scratch: &mut Vec<f32>,
) -> u32 {
    let f = |b: &[u8]| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
    scratch.clear();
    scratch.extend(cond.chunks_exact(2).zip(uncond.chunks_exact(2)).map(|(a, b)| {
        let (a, b) = (f(a), f(b));
        a + p.cfg_weight * (a - b)
    }));
    sample_guided(p, history, u, scratch)
}

/// The draw over guided logits already in `scratch`.
fn sample_guided(
    p: &CfgParams,
    history: impl IntoIterator<Item = u32>,
    u: Option<f32>,
    scratch: &mut Vec<f32>,
) -> u32 {
    let Some(u) = u else {
        return argmax(scratch);
    };
    if p.repetition_penalty != 1.0 {
        thread_local! {
            static SEEN: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        SEEN.with(|seen| {
            let mut seen = seen.borrow_mut();
            seen.clear();
            seen.resize(scratch.len().div_ceil(64), 0);
            for t in history {
                let (w, bit) = (t as usize / 64, 1u64 << (t % 64));
                if let (Some(x), Some(s)) = (scratch.get_mut(t as usize), seen.get_mut(w)) {
                    if *s & bit == 0 {
                        *s |= bit;
                        *x = if *x < 0.0 { *x * p.repetition_penalty } else { *x / p.repetition_penalty };
                    }
                }
            }
        });
    }
    let inv_t = 1.0 / p.temperature;
    let m = scratch.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for x in scratch.iter_mut() {
        *x = ((*x - m) * inv_t).exp();
    }
    // min_p over probabilities == min_p over unnormalised weights (the max weight is 1).
    let mut kept = 0.0f32;
    for x in scratch.iter_mut() {
        if *x < p.min_p {
            *x = 0.0;
        } else {
            kept += *x;
        }
    }
    if p.top_p < 1.0 {
        let mut order: Vec<usize> = (0..scratch.len()).filter(|&i| scratch[i] > 0.0).collect();
        order.sort_by(|&a, &b| scratch[b].total_cmp(&scratch[a]));
        let (mut acc, mut cut) = (0.0f32, order.len());
        for (k, &i) in order.iter().enumerate() {
            acc += scratch[i];
            if acc >= p.top_p * kept {
                cut = k + 1;
                break;
            }
        }
        for &i in &order[cut..] {
            kept -= scratch[i];
            scratch[i] = 0.0;
        }
    }
    let target = u * kept;
    let mut acc = 0.0f32;
    for (i, &x) in scratch.iter().enumerate() {
        acc += x;
        if x > 0.0 && acc > target {
            return i as u32;
        }
    }
    argmax(scratch)
}

#[cfg(test)]
mod tests {

    #[test]
    fn sample_cfg_bf16_matches_f32_rows() {
        let p = CfgParams { cfg_weight: 0.5, temperature: 0.8, min_p: 0.05, top_p: 0.9, repetition_penalty: 1.2 };
        let bf = |x: f32| (x.to_bits() >> 16) as u16;
        let cond: Vec<f32> = (0..97).map(|i| f32::from_bits(u32::from(bf(((i * 37) % 97) as f32 * 0.1 - 3.0)) << 16)).collect();
        let uncond: Vec<f32> = (0..97).map(|i| f32::from_bits(u32::from(bf(((i * 11) % 97) as f32 * 0.07 - 2.0)) << 16)).collect();
        let raw = |v: &[f32]| v.iter().flat_map(|x| bf(*x).to_le_bytes()).collect::<Vec<u8>>();
        let (rc, ru) = (raw(&cond), raw(&uncond));
        let (mut s1, mut s2) = (Vec::new(), Vec::new());
        for i in 0..50 {
            let u = Some(i as f32 / 50.0);
            let a = sample_cfg(&p, &cond, &uncond, [3, 5, 5], u, &mut s1);
            let b = sample_cfg_bf16(&p, &rc, &ru, [3, 5, 5], u, &mut s2);
            assert_eq!(a, b);
        }
    }
    use super::*;

    #[test]
    fn caller_scratch_reuses_capacity_after_warmup() {
        let logits = vec![0.0; 1024];
        let params = SamplingParams::default();
        let mut scratch = SamplerScratch::new(logits.len());

        let _ = sample_with_scratch(&logits, &params, None, 0.5, &mut scratch);
        let capacity = scratch.probs.capacity();
        let allocation = scratch.probs.as_ptr();

        for draw in [0.0, 0.1, 0.5, 0.9] {
            let _ = sample_with_scratch(&logits, &params, None, draw, &mut scratch);
            assert_eq!(scratch.probs.capacity(), capacity);
            assert_eq!(scratch.probs.as_ptr(), allocation);
        }
    }

    #[test]
    fn reusable_scratch_preserves_mask_and_sampling_filters() {
        let logits = [0.0, 4.0, 3.0, 2.0];
        let mask = [true, false, true, true];
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 2,
            top_p: 0.9,
            min_p: 0.1,
            ..SamplingParams::default()
        };
        let mut scratch = SamplerScratch::new(logits.len());

        let token = sample_with_scratch(&logits, &params, Some(&mask), 0.0, &mut scratch);

        assert_eq!(token, 2);
    }
}

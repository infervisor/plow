//! §H/§L Sampler — greedy / temperature / top-k / top-p / min-p, with hooks for
//! repetition penalty and logit bias. Reads a logits row; returns a token id.
//!
//! Default decode samples on-device (only the token id comes back). This host
//! sampler is the fallback path used when logprobs / guided / beam need the full
//! logits row (§M per-token output).

use std::cell::RefCell;

/// Sampling parameters from the request.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// (token, bias) additive logit adjustments.
    pub logit_bias: Vec<(u32, f32)>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            logit_bias: Vec::new(),
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

/// Apply repetition penalty and logit bias in place (pre-softmax).
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
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if params.top_k > 0 && probs.len() > params.top_k {
        probs.truncate(params.top_k);
    }
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

#[cfg(test)]
mod tests {
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

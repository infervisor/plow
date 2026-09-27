//! OpenAI `logprobs` / `top_logprobs` from one logits row.
//!
//! Values are over the RAW model distribution: the row as the packet wrote it (after the model's
//! own final-logit softcap), before temperature, top-k/p, penalties or logit bias. That is
//! vLLM's default `logprobs_mode = "raw_logprobs"`; `raw_logits` returns the logits themselves.

/// Most alternatives a request may ask for (OpenAI's `top_logprobs` cap).
pub const MAX_TOP_LOGPROBS: u8 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogprobRequest {
    /// Alternatives per position, `0..=MAX_TOP_LOGPROBS`.
    pub top: u8,
    /// Report raw logits instead of log-probabilities (vLLM `logprobs_mode = "raw_logits"`).
    pub raw_logits: bool,
}

/// One generated position.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprobs {
    pub logprob: f32,
    /// `(token, value)`, best first; ties keep the lower id first.
    pub top: Vec<(u32, f32)>,
}

/// Row statistics taken before any sampling adjustment rewrites the row.
#[derive(Clone, Debug)]
pub struct RowStats {
    lse: f32,
    raw_logits: bool,
    top: Vec<(u32, f32)>,
}

impl RowStats {
    pub fn of(logits: &[f32], req: LogprobRequest) -> Self {
        let lse = log_sum_exp(logits);
        let k = usize::from(req.top.min(MAX_TOP_LOGPROBS));
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        if k > 0 {
            let mut floor = f32::NEG_INFINITY;
            for (c, chunk) in logits.chunks(LANES).enumerate() {
                // Most chunks hold nothing above the current k-th value: one vector compare.
                if top.len() == k && chunk.iter().fold(floor, |m, &x| if x > m { x } else { m }) <= floor {
                    continue;
                }
                for (j, &x) in chunk.iter().enumerate() {
                    if top.len() == k && x <= floor {
                        continue;
                    }
                    let at = top.partition_point(|&(_, v)| v >= x);
                    top.insert(at, ((c * LANES + j) as u32, x));
                    top.truncate(k);
                    if top.len() == k {
                        floor = top[k - 1].1;
                    }
                }
            }
        }
        RowStats { lse, raw_logits: req.raw_logits, top }
    }

    /// Statistics computed elsewhere (the device kernel): `top` best first.
    pub fn from_parts(lse: f32, raw_logits: bool, top: Vec<(u32, f32)>) -> Self {
        RowStats { lse, raw_logits, top }
    }

    /// The entry for the chosen `token`, whose raw logit is `logit`.
    pub fn finish(self, logit: f32) -> TokenLogprobs {
        let shift = if self.raw_logits { 0.0 } else { self.lse };
        TokenLogprobs {
            logprob: logit - shift,
            top: self.top.into_iter().map(|(t, v)| (t, v - shift)).collect(),
        }
    }
}

const LANES: usize = 16;

/// `ln(sum(exp(x)))` over a vocab row. The row is read once per decode step for every request
/// that asked for logprobs (262144 entries for Gemma), so the exponentials are a branch-free
/// polynomial that vectorizes (libm `exp` cost ~1.4 ms per row, the whole step budget at c32).
/// Each 1024-entry block sums in f32 lanes, the blocks in f64: absolute lse error < 1e-5.
fn log_sum_exp(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU supports the enabled features (checked above).
        return unsafe { log_sum_exp_avx2(row) };
    }
    log_sum_exp_body(row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log_sum_exp_avx2(row: &[f32]) -> f32 {
    log_sum_exp_body(row)
}

#[inline(always)]
fn log_sum_exp_body(row: &[f32]) -> f32 {
    let mut lanes = [f32::NEG_INFINITY; LANES];
    let blocks = row.chunks_exact(LANES);
    let tail = blocks.remainder();
    for b in blocks {
        for (m, &x) in lanes.iter_mut().zip(b) {
            *m = if x > *m { x } else { *m };
        }
    }
    let max = tail.iter().chain(&lanes).copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    let mut sum = 0.0f64;
    for block in row.chunks(1024) {
        let mut acc = [0.0f32; LANES];
        let lanes = block.chunks_exact(LANES);
        let tail = lanes.remainder();
        for w in lanes {
            for (a, &x) in acc.iter_mut().zip(w) {
                *a += exp_nonpos(x - max);
            }
        }
        sum += tail.iter().map(|&x| f64::from(exp_nonpos(x - max))).sum::<f64>();
        sum += acc.iter().map(|&a| f64::from(a)).sum::<f64>();
    }
    max + sum.ln() as f32
}

/// `exp(x)` for `x <= 0` (relative error < 4e-6 down to x = -80, set by rounding x*log2(e); flushes below 2^-126 to ~2^-126, which a row
/// sum dominated by its max term never sees).
#[inline(always)]
fn exp_nonpos(x: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23: adding it rounds to an integer in the low bits
    let t = x * std::f32::consts::LOG2_E;
    let t = if t < -126.0 { -126.0 } else { t };
    let v = t + MAGIC;
    let f = t - (v - MAGIC);
    let n = v.to_bits() as i32 - MAGIC.to_bits() as i32;
    // 2^f on [-0.5, 0.5]: Taylor in f*ln2 through degree 6.
    const C: [f32; 7] = [1.0, 0.693_147_2, 0.240_226_5, 0.055_504_11, 0.009_618_129, 0.001_333_355_8, 0.000_154_035_3];
    let p = C[0] + f * (C[1] + f * (C[2] + f * (C[3] + f * (C[4] + f * (C[5] + f * C[6])))));
    p * f32::from_bits(((n + 127) << 23) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprobs_normalize_and_rank() {
        let logits = [1.0f32, 3.0, 2.0, 3.0, -1.0];
        let s = RowStats::of(&logits, LogprobRequest { top: 3, raw_logits: false });
        let t = s.finish(logits[2]);
        let z: f32 = logits.iter().map(|x| x.exp()).sum();
        assert!((t.logprob - (2.0f32.exp() / z).ln()).abs() < 1e-6);
        let ids: Vec<u32> = t.top.iter().map(|x| x.0).collect();
        assert_eq!(ids, [1, 3, 2]);
        let raw = RowStats::of(&logits, LogprobRequest { top: 1, raw_logits: true }).finish(1.0);
        assert_eq!((raw.logprob, raw.top.clone()), (1.0, vec![(1, 3.0)]));
    }

    #[test]
    fn log_sum_exp_matches_f64() {
        let mut state = 0x9e37_79b9u32;
        let mut row: Vec<f32> = (0..262_147)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 60.0 - 30.0
            })
            .collect();
        row[7] = f32::NEG_INFINITY;
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exact = f64::from(max) + row.iter().map(|&x| (f64::from(x) - f64::from(max)).exp()).sum::<f64>().ln();
        assert!((f64::from(log_sum_exp(&row)) - exact).abs() < 2e-5, "{} vs {exact}", log_sum_exp(&row));
        for x in [0.0f32, -1e-3, -0.5, -1.0, -7.25, -30.0, -80.0] {
            assert!((exp_nonpos(x) / x.exp() - 1.0).abs() < 4e-6, "{x}");
        }
    }
}

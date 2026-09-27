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
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = logits.iter().map(|&x| f64::from((x - max).exp())).sum();
        let lse = max + sum.ln() as f32;
        let k = usize::from(req.top.min(MAX_TOP_LOGPROBS));
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        if k > 0 {
            for (i, &x) in logits.iter().enumerate() {
                if top.len() == k && x <= top[k - 1].1 {
                    continue;
                }
                let at = top.partition_point(|&(_, v)| v >= x);
                top.insert(at, (i as u32, x));
                top.truncate(k);
            }
        }
        RowStats { lse, raw_logits: req.raw_logits, top }
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
}

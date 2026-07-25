//! M/D/1 admission: exact service-time table + analytic batch window.
//!
//! This module replaces the two *estimates* in [`super::admission`] and
//! [`super::batching`] with *computations*, which is only possible because plow
//! is compile-time static:
//!
//! * [`admission::LoadEstimator`](super::admission::LoadEstimator) learns the
//!   service time online with an EWMA. But the compiler already knows the packet
//!   count, the per-op block counts and the byte cost of a decode step before any
//!   request arrives, so service time is a *lookup*, not an observation.
//! * [`batching::formation_window_ms`](super::batching::formation_window_ms)
//!   caps a `1/λ` heuristic with a tuned `max_hold_ms`. With a known service
//!   time the optimal window is the minimiser of a latency function of λ.
//!
//! # Why M/D/1 and not M/M/1
//!
//! MEASURED on Qwen3-4B / RTX 5090 (sm_120a), 143 timed steps at ctx 4096:
//! mean 6.8699 ms, sd 0.0105 ms — **CV = 0.15 %**, min 6.845 / max 6.895. The
//! decode step is deterministic to within a part in 650. The Pollaczek–Khinchine
//! formula for M/G/1 waiting time,
//!
//! ```text
//!   Wq = ρ·E[S]·(1 + CV²) / (2(1 − ρ))
//! ```
//!
//! collapses to the M/D/1 form at CV = 0 and to the M/M/1 form at CV = 1. At
//! CV = 0.0015 the M/D/1 form is exact to 2·10⁻⁶ relative, so **plow queues at
//! half the M/M/1 waiting time for the same utilisation**. See `qsim` for the
//! trace-driven validation and its negative control.
//!
//! # Status of the batch axis
//!
//! [`ServiceTable`] is parameterised by batch, but **batch > 1 is not compilable
//! today** — the `gemma4` emitter hardcodes the decode bucket to one row, the
//! KV/activation arenas carry no batch dimension, and the sm_120 GEMV kernels are
//! instantiated at one row (`gemv_rows<1>`; `d_gemv_qkv`/`d_gemv_glu` ignore `M`
//! outright). Batch > 1 predictions from this table are therefore **projections
//! of the byte model, not measurements**. Anything reading them must say so.

/// Per-step cost of a model, in the quantities the compiler already knows.
///
/// The byte model alone under-predicts: a decode step reads
/// `weight_bytes + batch·kv_bytes_per_token·ctx` and nothing else, but the
/// measured step is longer than that read by a near-constant margin (gate
/// latency across the 401 packets, the cooperative launch, and the 16.7 %
/// occupancy the interpreter runs at). `fixed_ms` is that margin, and it is a
/// *fitted* constant — the only one in this module.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelCost {
    /// Bytes of weights streamed per decode step. Batch-independent — this is
    /// the whole reason batching pays.
    pub weight_bytes: u64,
    /// KV bytes per token **per sequence**: `2 · n_kv_head · head_dim · elem · layers`.
    pub kv_bytes_per_token: u64,
    /// Device packets per token (Qwen3-4B decode: 401). Not used in the cost
    /// model itself; carried so a table entry is self-describing.
    pub packets_per_token: u32,
    /// Effective achieved HBM bandwidth, GB/s.
    pub bandwidth_gbps: f64,
    /// Per-step cost not explained by the byte read, ms.
    pub fixed_ms: f64,
}

impl ModelCost {
    /// Qwen3-4B on RTX 5090: 36 layers, 8 KV heads, head_dim 128, bf16.
    /// `weight_bytes` is the 7.49 GiB checkpoint; `fixed_ms` and
    /// `bandwidth_gbps` are fitted against the measured ctx sweep (see
    /// `ServiceTable::fit`).
    pub fn qwen3_4b(bandwidth_gbps: f64, fixed_ms: f64) -> Self {
        ModelCost {
            weight_bytes: 8_045_000_000,
            // 2 · 8 · 128 · 2 B · 36 layers
            kv_bytes_per_token: 147_456,
            packets_per_token: 401,
            bandwidth_gbps,
            fixed_ms,
        }
    }

    /// Bytes moved by one decode step at `(ctx, batch)`.
    pub fn step_bytes(&self, ctx: u64, batch: u32) -> u64 {
        self.weight_bytes + (batch as u64) * self.kv_bytes_per_token * ctx
    }

    /// Predicted service time of one decode step, ms.
    pub fn step_ms(&self, ctx: u64, batch: u32) -> f64 {
        let gb = self.step_bytes(ctx, batch) as f64 / 1e9;
        self.fixed_ms + 1000.0 * gb / self.bandwidth_gbps
    }

    /// Predicted service time **per token**, ms. This is what batching moves:
    /// the batch-independent part (`fixed_ms` + the weight read) is divided by
    /// the batch, the per-sequence KV read is not.
    pub fn per_token_ms(&self, ctx: u64, batch: u32) -> f64 {
        self.step_ms(ctx, batch) / batch.max(1) as f64
    }

    /// The affine decomposition `S(B) = a + B·c` that the window derivation uses.
    /// `a` is batch-independent (fixed cost + weight read), `c` is per-sequence.
    pub fn affine(&self, ctx: u64) -> (f64, f64) {
        let a = self.fixed_ms + 1000.0 * (self.weight_bytes as f64 / 1e9) / self.bandwidth_gbps;
        let c = 1000.0 * ((self.kv_bytes_per_token * ctx) as f64 / 1e9) / self.bandwidth_gbps;
        (a, c)
    }
}

/// One precomputed table row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ServiceEntry {
    pub ctx: u64,
    pub batch: u32,
    pub step_ms: f64,
    pub per_token_ms: f64,
}

/// Service time per `(ctx-bucket, batch)`, precomputed so that admission is a
/// lookup rather than an online estimate.
#[derive(Clone, Debug)]
pub struct ServiceTable {
    pub cost: ModelCost,
    pub entries: Vec<ServiceEntry>,
}

impl ServiceTable {
    /// Build the cartesian product of the compiled ctx ladder and batch ladder.
    pub fn build(cost: ModelCost, ctxs: &[u64], batches: &[u32]) -> Self {
        let mut entries = Vec::with_capacity(ctxs.len() * batches.len());
        for &ctx in ctxs {
            for &batch in batches {
                entries.push(ServiceEntry {
                    ctx,
                    batch,
                    step_ms: cost.step_ms(ctx, batch),
                    per_token_ms: cost.per_token_ms(ctx, batch),
                });
            }
        }
        ServiceTable { cost, entries }
    }

    /// Exact lookup: the smallest compiled `(ctx, batch)` covering the request.
    pub fn lookup(&self, ctx: u64, batch: u32) -> Option<&ServiceEntry> {
        self.entries
            .iter()
            .filter(|e| e.ctx >= ctx && e.batch >= batch)
            .min_by_key(|e| (e.ctx, e.batch))
    }

    /// Least-squares fit of `(bandwidth_gbps, fixed_ms)` to measured
    /// `(ctx, batch, step_ms)` samples. The byte model is affine in bytes,
    /// `t = fixed + bytes/bw`, so this is an ordinary linear regression of
    /// time on bytes; the slope is `1/bw`.
    ///
    /// Returns `None` if the samples are degenerate (all the same byte count),
    /// which is exactly the batch-1-single-ctx case — a fit needs a ctx sweep.
    pub fn fit(samples: &[(u64, u32, f64)], mut cost: ModelCost) -> Option<ModelCost> {
        let n = samples.len() as f64;
        if samples.len() < 2 {
            return None;
        }
        let xs: Vec<f64> = samples
            .iter()
            .map(|&(ctx, b, _)| cost.step_bytes(ctx, b) as f64 / 1e9)
            .collect();
        let ys: Vec<f64> = samples.iter().map(|&(_, _, ms)| ms).collect();
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
        if sxx <= f64::EPSILON {
            return None;
        }
        let sxy: f64 = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| (x - mx) * (y - my))
            .sum();
        let slope = sxy / sxx; // ms per GB
        if slope <= 0.0 {
            return None;
        }
        cost.bandwidth_gbps = 1000.0 / slope;
        cost.fixed_ms = my - slope * mx;
        Some(cost)
    }
}

/// Waiting-time formulas. `service_ms` is E[S], `rho` = λ·E[S] must be < 1.
pub mod wait {
    /// M/D/1 — deterministic service. `Wq = ρ·E[S] / (2(1−ρ))`.
    pub fn md1(service_ms: f64, rho: f64) -> f64 {
        if rho >= 1.0 {
            return f64::INFINITY;
        }
        rho * service_ms / (2.0 * (1.0 - rho))
    }

    /// M/M/1 — exponential service. `Wq = ρ·E[S] / (1−ρ)`. Exactly twice M/D/1.
    pub fn mm1(service_ms: f64, rho: f64) -> f64 {
        if rho >= 1.0 {
            return f64::INFINITY;
        }
        rho * service_ms / (1.0 - rho)
    }

    /// Pollaczek–Khinchine, the general M/G/1 case that both of the above are
    /// corners of: `Wq = ρ·E[S]·(1 + CV²) / (2(1−ρ))`.
    pub fn mg1(service_ms: f64, rho: f64, cv: f64) -> f64 {
        if rho >= 1.0 {
            return f64::INFINITY;
        }
        rho * service_ms * (1.0 + cv * cv) / (2.0 * (1.0 - rho))
    }

    /// M/M/1 waiting-time quantile, closed form. The M/M/1 waiting time has an
    /// atom at 0 of mass `1−ρ` and an exponential tail:
    /// `P(W > t) = ρ·e^{−(μ−λ)t}`, so for `q > 1−ρ` the quantile is
    /// `ln(ρ/(1−q)) / (μ−λ)`.
    ///
    /// There is deliberately **no** `md1_quantile` counterpart here. The exact
    /// M/D/1 waiting-time CDF (the Crommelin/Erlang series) is an alternating
    /// sum whose terms grow like `e^{λ(t−jD)}`; it loses all significance for
    /// the `t/D` ratios a p99 at high ρ needs, and a quietly wrong quantile is
    /// worse than none. `qsim` takes M/D/1 quantiles from simulation instead.
    pub fn mm1_quantile(service_ms: f64, rho: f64, q: f64) -> f64 {
        if rho >= 1.0 {
            return f64::INFINITY;
        }
        if rho <= 0.0 || q <= 1.0 - rho {
            return 0.0;
        }
        let mu = 1.0 / service_ms;
        let lambda = rho * mu;
        (rho / (1.0 - q)).ln() / (mu - lambda)
    }
}

/// Mean per-token latency at batch window `w_ms`, given arrival rate `lambda`
/// (requests/sec) and the affine step cost `S(B) = a + B·c`.
///
/// Three terms: the batching wait a request pays (mean `w/2`), the queueing
/// wait at the resulting utilisation (M/D/1), and the service itself.
pub fn latency_at_window(cost: &ModelCost, ctx: u64, lambda: f64, w_ms: f64) -> f64 {
    let (a, c) = cost.affine(ctx);
    let lam_ms = lambda / 1000.0; // arrivals per ms
    let batch = 1.0 + lam_ms * w_ms;
    let s_eff = a / batch + c; // per-token service, ms
    let rho = lam_ms * s_eff;
    if rho >= 1.0 {
        return f64::INFINITY;
    }
    0.5 * w_ms + wait::md1(s_eff, rho) + s_eff
}

/// Is *any* batching worth it at this arrival rate? The derivative of
/// [`latency_at_window`] at `w = 0`:
///
/// ```text
///   dT/dw|₀ = 1/2 − λa · [ ρ(2−ρ)/(2(1−ρ)²) + 1 ]
/// ```
///
/// Negative means holding pays. This is the exact replacement for the tuned
/// `util < 0.85` constant in [`super::admission::admit`]: the threshold is a
/// function of λ and the known step cost, not a number someone picked.
pub fn batching_pays(cost: &ModelCost, ctx: u64, lambda: f64) -> bool {
    let (a, c) = cost.affine(ctx);
    let lam_ms = lambda / 1000.0;
    let s0 = a + c;
    let rho = lam_ms * s0;
    if rho >= 1.0 {
        return true; // saturated: batching is the only way out
    }
    let bracket = rho * (2.0 - rho) / (2.0 * (1.0 - rho) * (1.0 - rho)) + 1.0;
    lam_ms * a * bracket > 0.5
}

/// The latency-minimising batch window, ms. Derived from λ and the known step
/// cost by minimising [`latency_at_window`] — not a tuned constant.
///
/// Returns 0 when [`batching_pays`] is false, which is the correct answer at
/// low load: holding an isolated request only adds latency.
pub fn optimal_window_ms(cost: &ModelCost, ctx: u64, lambda: f64, max_ms: f64) -> f64 {
    if !batching_pays(cost, ctx, lambda) {
        return 0.0;
    }
    // Golden-section on [0, max_ms]; T(w) is unimodal in w for fixed λ.
    let (mut lo, mut hi) = (0.0f64, max_ms);
    let phi = 0.5 * (5.0f64.sqrt() - 1.0);
    let (mut x1, mut x2) = (hi - phi * (hi - lo), lo + phi * (hi - lo));
    let (mut f1, mut f2) = (
        latency_at_window(cost, ctx, lambda, x1),
        latency_at_window(cost, ctx, lambda, x2),
    );
    for _ in 0..120 {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - phi * (hi - lo);
            f1 = latency_at_window(cost, ctx, lambda, x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + phi * (hi - lo);
            f2 = latency_at_window(cost, ctx, lambda, x2);
        }
    }
    0.5 * (lo + hi)
}

/// Deadline-aware admission. With deterministic service this is a computation,
/// not a guess: given the queue depth we know exactly when a newly admitted
/// request would finish, so the SLO check is exact rather than probabilistic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Run it: the deadline is met with the queue as it stands.
    Admit,
    /// Hold to batch — [`batching_pays`] says the amortisation beats the wait
    /// and the deadline still clears.
    Hold,
    /// Shed: no admissible schedule meets the deadline.
    Shed,
}

/// `queue_depth` requests are already committed ahead of this one, each needing
/// `tokens_each` decode steps; the arriving request needs `tokens` steps and has
/// `slack_ms` until its deadline.
pub fn admit_deadline(
    table: &ServiceTable,
    ctx: u64,
    batch: u32,
    queue_depth: u32,
    tokens_each: u32,
    tokens: u32,
    slack_ms: f64,
    lambda: f64,
) -> Verdict {
    let Some(e) = table.lookup(ctx, batch) else {
        return Verdict::Shed;
    };
    // Exact head-of-line delay: everything ahead must finish first.
    let ahead_ms = queue_depth as f64 * tokens_each as f64 * e.per_token_ms;
    let own_ms = tokens as f64 * e.per_token_ms;
    if ahead_ms + own_ms > slack_ms {
        return Verdict::Shed;
    }
    if batching_pays(&table.cost, ctx, lambda) {
        let w = optimal_window_ms(&table.cost, ctx, lambda, 4.0);
        if ahead_ms + own_ms + w <= slack_ms {
            return Verdict::Hold;
        }
    }
    Verdict::Admit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost() -> ModelCost {
        ModelCost::qwen3_4b(1673.0, 1.70)
    }

    #[test]
    fn md1_is_exactly_half_mm1() {
        for &rho in &[0.1, 0.5, 0.8, 0.95] {
            let d = wait::md1(6.87, rho);
            let m = wait::mm1(6.87, rho);
            assert!((m / d - 2.0).abs() < 1e-12, "rho={rho}: {m} vs {d}");
        }
    }

    #[test]
    fn pk_interpolates_the_two_corners() {
        let (s, rho) = (6.87, 0.8);
        assert!((wait::mg1(s, rho, 0.0) - wait::md1(s, rho)).abs() < 1e-12);
        assert!((wait::mg1(s, rho, 1.0) - wait::mm1(s, rho)).abs() < 1e-12);
        // At the MEASURED cv the M/D/1 form is exact to within 2e-6 relative.
        let cv = 0.0105 / 6.8699;
        let rel = (wait::mg1(s, rho, cv) - wait::md1(s, rho)).abs() / wait::md1(s, rho);
        assert!(rel < 1e-5, "measured cv {cv} gives rel error {rel}");
    }

    #[test]
    fn no_batching_at_low_load() {
        // One request every 500 ms against a 6.9 ms step: nothing to batch with.
        let c = cost();
        assert!(!batching_pays(&c, 4096, 2.0));
        assert_eq!(optimal_window_ms(&c, 4096, 2.0, 4.0), 0.0);
    }

    #[test]
    fn batching_pays_near_saturation() {
        let c = cost();
        // rho ~ 0.9 against the batch-1 step time.
        let (a, cc) = c.affine(4096);
        let lambda = 0.9 / (a + cc) * 1000.0;
        assert!(batching_pays(&c, 4096, lambda));
        let w = optimal_window_ms(&c, 4096, lambda, 4.0);
        assert!(w > 0.0, "expected a positive window, got {w}");
    }

    #[test]
    fn window_optimum_beats_its_neighbours() {
        let c = cost();
        let (a, cc) = c.affine(4096);
        let lambda = 0.9 / (a + cc) * 1000.0;
        let w = optimal_window_ms(&c, 4096, lambda, 4.0);
        let t = latency_at_window(&c, 4096, lambda, w);
        for d in [-0.5, -0.1, 0.1, 0.5] {
            let wn = w + d;
            if wn < 0.0 || wn > 4.0 {
                continue;
            }
            assert!(
                latency_at_window(&c, 4096, lambda, wn) >= t - 1e-9,
                "w={w} t={t} beaten by w={wn}"
            );
        }
    }

    #[test]
    fn table_lookup_rounds_up() {
        let t = ServiceTable::build(cost(), &[4096, 8192, 16384], &[1, 2, 4, 8]);
        let e = t.lookup(5000, 3).unwrap();
        assert_eq!((e.ctx, e.batch), (8192, 4));
        assert!(t.lookup(99999, 1).is_none());
    }

    #[test]
    fn per_token_falls_with_batch_but_not_linearly() {
        // The byte model's whole claim: the weight read amortises, KV does not.
        let c = cost();
        let p1 = c.per_token_ms(4096, 1);
        let p8 = c.per_token_ms(4096, 8);
        assert!(p8 < p1);
        // Strictly worse than the ideal 8x, because KV scales with the batch.
        assert!(p8 > p1 / 8.0, "p1={p1} p8={p8}");
    }

    #[test]
    fn fit_recovers_known_parameters() {
        let truth = ModelCost::qwen3_4b(1673.0, 1.70);
        let samples: Vec<(u64, u32, f64)> = [4096u64, 8192, 16384, 32768]
            .iter()
            .map(|&ctx| (ctx, 1u32, truth.step_ms(ctx, 1)))
            .collect();
        let got = ServiceTable::fit(&samples, ModelCost::qwen3_4b(1.0, 0.0)).unwrap();
        assert!((got.bandwidth_gbps - 1673.0).abs() < 1e-6, "{got:?}");
        assert!((got.fixed_ms - 1.70).abs() < 1e-9, "{got:?}");
    }

    #[test]
    fn fit_rejects_degenerate_input() {
        // A single ctx point cannot separate `fixed_ms` from bandwidth.
        let c = cost();
        let s = vec![(4096u64, 1u32, 6.87), (4096, 1, 6.88)];
        assert!(ServiceTable::fit(&s, c).is_none());
        assert!(ServiceTable::fit(&[], c).is_none());
    }

    #[test]
    fn deadline_sheds_under_overload() {
        let t = ServiceTable::build(cost(), &[4096], &[1]);
        // 64 requests of 128 tokens ahead, at ~6.9 ms/token, is ~56 s of work.
        let v = admit_deadline(&t, 4096, 1, 64, 128, 128, 1_000.0, 10.0);
        assert_eq!(v, Verdict::Shed);
        // Same request with an empty queue and a generous deadline is admitted.
        let v = admit_deadline(&t, 4096, 1, 0, 128, 128, 5_000.0, 10.0);
        assert_ne!(v, Verdict::Shed);
    }

    #[test]
    fn mm1_quantile_matches_its_own_tail() {
        let s = 6.87;
        for &rho in &[0.3, 0.6, 0.85] {
            let mu = 1.0 / s;
            let lambda = rho * mu;
            for &q in &[0.5, 0.9, 0.99] {
                let t = wait::mm1_quantile(s, rho, q);
                if t == 0.0 {
                    assert!(q <= 1.0 - rho);
                    continue;
                }
                // Invert: P(W > t) must equal 1 - q.
                let tail = rho * (-(mu - lambda) * t).exp();
                assert!((tail - (1.0 - q)).abs() < 1e-12, "rho={rho} q={q} tail={tail}");
            }
        }
        // Below the atom mass the quantile is exactly zero: at rho=0.3, 65% of
        // arrivals find the server idle and wait not at all.
        assert_eq!(wait::mm1_quantile(s, 0.3, 0.5), 0.0);
    }
}

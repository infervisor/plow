//! §I Admission control — queuing-theory arrival/service estimation + shed.

/// Exponentially-weighted moving-average rate estimator (arrival λ / service μ).
#[derive(Clone, Copy, Debug)]
pub struct Ewma {
    value: f64,
    alpha: f64,
}

impl Ewma {
    pub fn new(alpha: f64) -> Self {
        Ewma { value: 0.0, alpha }
    }

    #[inline]
    pub fn update(&mut self, sample: f64) -> f64 {
        self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        self.value
    }

    pub fn get(&self) -> f64 {
        self.value
    }
}

/// Per-slug load estimate: arrival rate λ and per-batch service rate μ(B).
pub struct LoadEstimator {
    pub lambda: Ewma,
    pub service_ms: Ewma,
}

impl Default for LoadEstimator {
    fn default() -> Self {
        LoadEstimator {
            lambda: Ewma::new(0.2),
            service_ms: Ewma::new(0.2),
        }
    }
}

impl LoadEstimator {
    /// Utilization ρ = λ / μ given the current batch's service time.
    pub fn utilization(&self) -> f64 {
        let mu = if self.service_ms.get() > 0.0 {
            1000.0 / self.service_ms.get()
        } else {
            f64::INFINITY
        };
        if mu.is_finite() && mu > 0.0 {
            self.lambda.get() / mu
        } else {
            0.0
        }
    }
}

/// Admission verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    /// Run this iteration now.
    Now,
    /// Hold briefly to form a larger batch (queuing gain outweighs the wait).
    Defer,
    /// Reject — predicted wait blows the SLO or memory can't seat it.
    Shed,
}

/// Decide admission from utilization, predicted wait, and the SLO.
pub fn admit(util: f64, predicted_wait_ms: f64, slo_ms: f64, mem_ok: bool) -> Admit {
    if !mem_ok || predicted_wait_ms > slo_ms {
        return Admit::Shed;
    }
    // Below saturation there's headroom (and no queue to batch with): run
    // immediately — deferring an isolated request only adds latency. Near
    // saturation (ρ ≥ 0.85), hold briefly so the batch fills; the queuing
    // (throughput) gain outweighs the short wait.
    if util < 0.85 {
        Admit::Now
    } else {
        Admit::Defer
    }
}

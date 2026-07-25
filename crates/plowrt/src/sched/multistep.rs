//! §L Multi-step / overlap scheduling.
//!
//! Enqueue `k` decode iterations' worth of packet streams at once so the host
//! isn't in the loop every token, and overlap host sampling/detok of step `t`
//! with device compute of step `t+1` (SGLang overlap-scheduler idea). Reduces
//! the host→device turnaround that otherwise caps decode throughput at small
//! batch.

/// How many decode steps to pipeline per scheduler pass.
#[derive(Clone, Copy, Debug)]
pub struct MultiStep {
    pub steps: u32,
}

impl Default for MultiStep {
    fn default() -> Self {
        MultiStep { steps: 1 }
    }
}

impl MultiStep {
    /// Choose a step count from batch size: small batches (host-turnaround
    /// bound) pipeline more steps; large batches (compute bound) need fewer.
    pub fn for_batch(batch: i64) -> Self {
        let steps = if batch <= 2 {
            4
        } else if batch <= 8 {
            2
        } else {
            1
        };
        MultiStep { steps }
    }
}

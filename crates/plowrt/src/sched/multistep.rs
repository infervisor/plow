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

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DecodeBudgetError {
    #[error("slot {0} out of range")]
    Slot(usize),
    #[error("context exhausted at {position} (compiled max {max_ctx})")]
    Context { position: u32, max_ctx: usize },
}

pub fn decode_quantum(
    slots: impl IntoIterator<Item = usize>,
    positions: &[u32],
    max_ctx: usize,
    requested: usize,
    capacity: usize,
) -> Result<usize, DecodeBudgetError> {
    let mut quantum = requested.min(capacity);
    let mut active = false;
    for slot in slots {
        let &position = positions.get(slot).ok_or(DecodeBudgetError::Slot(slot))?;
        let room = max_ctx.saturating_sub(position as usize);
        if room == 0 {
            return Err(DecodeBudgetError::Context { position, max_ctx });
        }
        active = true;
        quantum = quantum.min(room);
    }
    Ok(if active { quantum } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_decode_budget_obeys_every_active_context() {
        let positions = [12, 100, 98, 20];
        assert_eq!(decode_quantum([0, 3], &positions, 100, 8, 4), Ok(4));
        assert_eq!(decode_quantum([0, 2], &positions, 100, 8, 4), Ok(2));
        assert_eq!(decode_quantum([0, 3], &positions, 100, 1, 4), Ok(1));
        assert_eq!(decode_quantum([], &positions, 100, 8, 4), Ok(0));
        assert_eq!(
            decode_quantum([4], &positions, 100, 8, 4),
            Err(DecodeBudgetError::Slot(4))
        );
        assert_eq!(
            decode_quantum([1], &positions, 100, 8, 4),
            Err(DecodeBudgetError::Context {
                position: 100,
                max_ctx: 100
            })
        );
    }
}

use plow_asset::mixed_step::{self, DecodeRequest, Plan, PrefillRequest};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StageError {
    #[error("{0}")]
    Plan(String),
    #[error("mixed step has no pending device submission")]
    NoPendingPlan,
    #[error("mixed step already has a pending device submission")]
    PendingPlan,
    #[error("mixed step decode output has {actual} rows, expected {expected}")]
    OutputRows { expected: usize, actual: usize },
    #[error("mixed step frontier slot {slot} is outside capacity {capacity}")]
    FrontierCapacity { slot: u32, capacity: usize },
    #[error(
        "mixed step frontier slot {slot} changed before commit: expected {expected}, got {actual}"
    )]
    FrontierChanged {
        slot: u32,
        expected: u32,
        actual: u32,
    },
}

/// Reusable host storage for a mixed decode/prefill device submission.
///
/// Staging reads committed frontiers but does not change them. The caller
/// commits only after the device reports successful completion.
pub struct MixedStepStaging {
    plan: Plan,
    pending: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalDeviceMetadata<'a> {
    pub decode_slots: &'a [i32],
    pub prefill_spans: &'a [packet::dev::PrefillSpan],
    pub parked: &'a [u32],
    pub rows: u32,
    pub decode_rows: u32,
}

impl MixedStepStaging {
    pub fn with_capacity(
        row_capacity: usize,
        prefill_capacity: usize,
        active_capacity: usize,
    ) -> Self {
        Self {
            plan: Plan::with_capacity(row_capacity, prefill_capacity, active_capacity),
            pending: false,
        }
    }

    pub fn stage<'a>(
        &'a mut self,
        decode: &[DecodeRequest],
        prefill: &[PrefillRequest<'_>],
        frontiers: &[u32],
        rows: u32,
        max_ctx: u32,
        auxiliary_program: u32,
    ) -> Result<&'a Plan, StageError> {
        if self.pending {
            return Err(StageError::PendingPlan);
        }
        mixed_step::plan_into(
            decode,
            prefill,
            frontiers,
            rows,
            max_ctx,
            auxiliary_program,
            &mut self.plan,
        )
        .map_err(StageError::Plan)?;
        self.pending = true;
        Ok(&self.plan)
    }

    pub fn pending_plan(&self) -> Option<&Plan> {
        self.pending.then_some(&self.plan)
    }

    /// The three canonical host images an adapter uploads for a selected mixed
    /// program. Prefill consumers read the span/mask pair through `PlowProgram`;
    /// only the compact decode-slot image is a packet tensor.
    pub fn pending_device_metadata(&self) -> Option<CanonicalDeviceMetadata<'_>> {
        self.pending.then(|| CanonicalDeviceMetadata {
            decode_slots: &self.plan.decode_slots,
            prefill_spans: &self.plan.prefill_spans,
            parked: &self.plan.parked,
            rows: self.plan.rows.len() as u32,
            decode_rows: self.plan.decode_rows,
        })
    }

    /// Publish the logical KV progress after the staged device work succeeds.
    /// Every frontier is checked before any is changed.
    pub fn commit_after_device_success(&mut self, frontiers: &mut [u32]) -> Result<(), StageError> {
        if !self.pending {
            return Err(StageError::NoPendingPlan);
        }

        for row in self.plan.rows.iter().take(self.plan.decode_rows as usize) {
            check_frontier(frontiers, row.slot, row.position)?;
        }
        for span in &self.plan.prefill_spans {
            check_frontier(frontiers, span.slot, span.kv_row0)?;
        }

        for row in self.plan.rows.iter().take(self.plan.decode_rows as usize) {
            frontiers[row.slot as usize] = row.kv_len;
        }
        for span in &self.plan.prefill_spans {
            frontiers[span.slot as usize] = span.kv_len;
        }
        self.pending = false;
        Ok(())
    }

    /// Validate and publish a completed submission, then scatter the compact
    /// decode prefix into caller-owned logical-request order.
    pub fn finish_after_device_success(
        &mut self,
        frontiers: &mut [u32],
        device_tokens: &[u32],
        output: &mut [u32],
    ) -> Result<(), StageError> {
        if !self.pending {
            return Err(StageError::NoPendingPlan);
        }
        let expected = self.plan.decode_rows as usize;
        if device_tokens.len() != expected || output.len() != expected {
            return Err(StageError::OutputRows {
                expected,
                actual: if device_tokens.len() != expected {
                    device_tokens.len()
                } else {
                    output.len()
                },
            });
        }
        self.commit_after_device_success(frontiers)?;
        output.copy_from_slice(device_tokens);
        Ok(())
    }

    pub fn discard(&mut self) {
        self.pending = false;
    }
}

fn check_frontier(frontiers: &[u32], slot: u32, expected: u32) -> Result<(), StageError> {
    let Some(&actual) = frontiers.get(slot as usize) else {
        return Err(StageError::FrontierCapacity {
            slot,
            capacity: frontiers.len(),
        });
    };
    if actual != expected {
        return Err(StageError::FrontierChanged {
            slot,
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "mixed_step_staging_tests.rs"]
mod tests;

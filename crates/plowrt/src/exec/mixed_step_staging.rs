use crate::RuntimeError;
use packet::dev::PrefillSpan;
use plow_asset::mixed_step::{self, DecodeRequest, Plan, PrefillRequest};
use plow_asset::token_batch;

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
    #[error(
        "token batch slot {slot} was recycled before commit: request {request} held \
         generation {expected}, the slot now holds {actual}"
    )]
    GenerationChanged {
        request: u32,
        slot: u32,
        expected: u32,
        actual: u32,
    },
    #[error("token batch produced {actual} sampled ids, expected S = {expected}")]
    SampleRows { expected: usize, actual: usize },
}

/// Reusable host storage for a mixed decode/prefill device submission.
///
/// Staging reads committed frontiers but does not change them. The caller
/// commits only after the device reports successful completion.
pub struct MixedStepStaging {
    plan: Plan,
    pending: bool,
    /// Logical request id per leading row, filled only by [`Self::stage_requests`].
    owners: Vec<u32>,
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
            owners: Vec::with_capacity(active_capacity),
        }
    }

    /// Stage a unified token batch from the shared request contract
    /// ([`mixed_step::plan_requests_into`]); deliver it with
    /// [`Self::finish_requests_after_device_success`].
    #[allow(clippy::too_many_arguments)]
    pub fn stage_requests<'a>(
        &'a mut self,
        requests: &[token_batch::Request<'_>],
        frontiers: &[u32],
        generations: &[u32],
        rows: u32,
        max_ctx: u32,
        auxiliary_program: u32,
    ) -> Result<&'a Plan, StageError> {
        if self.pending {
            return Err(StageError::PendingPlan);
        }
        mixed_step::plan_requests_into(
            requests,
            frontiers,
            generations,
            rows,
            max_ctx,
            auxiliary_program,
            &mut self.plan,
            &mut self.owners,
        )
        .map_err(StageError::Plan)?;
        self.pending = true;
        Ok(&self.plan)
    }

    /// [`Self::finish_after_device_success`] delivering by LOGICAL REQUEST: one
    /// `(request, id)` per leading row, in sample order — the same contract the CUDA route's
    /// [`TokenBatchStaging::deliver`] uses, so a mux arm reads both backends' output alike.
    pub fn finish_requests_after_device_success(
        &mut self,
        frontiers: &mut [u32],
        device_tokens: &[u32],
        out: &mut Vec<(u32, u32)>,
    ) -> Result<(), StageError> {
        if !self.pending {
            return Err(StageError::NoPendingPlan);
        }
        let expected = self.plan.decode_rows as usize;
        if device_tokens.len() != expected || self.owners.len() != expected {
            return Err(StageError::OutputRows {
                expected,
                actual: device_tokens.len(),
            });
        }
        self.commit_after_device_success(frontiers)?;
        out.clear();
        out.extend(
            self.owners
                .iter()
                .zip(device_tokens)
                .map(|(&owner, &id)| (owner, id)),
        );
        Ok(())
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
        self.stage_cover(
            decode,
            prefill,
            frontiers,
            rows,
            max_ctx,
            auxiliary_program,
            mixed_step::SpanCover::DecodeBand,
        )
    }

    /// Stage under an explicit span contract. `SpanCover::PrefixFree` is the unified token
    /// batch's: spans cover exactly `[0, M)`.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_cover<'a>(
        &'a mut self,
        decode: &[DecodeRequest],
        prefill: &[PrefillRequest<'_>],
        frontiers: &[u32],
        rows: u32,
        max_ctx: u32,
        auxiliary_program: u32,
        cover: mixed_step::SpanCover,
    ) -> Result<&'a Plan, StageError> {
        if self.pending {
            return Err(StageError::PendingPlan);
        }
        // A positional stage has no owners; a later `finish_requests_*` must refuse, not
        // deliver against a stale table.
        self.owners.clear();
        mixed_step::plan_into_cover(
            decode,
            prefill,
            frontiers,
            rows,
            max_ctx,
            auxiliary_program,
            cover,
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

        // Per REQUEST, not per span: under `SpanCover::PrefixFree` a completing prompt owns
        // two spans on one slot, so a per-span check would compare the terminal span's
        // `kv_row0` against a frontier that belongs to the body span. Every frontier is
        // checked before any is changed, so a refusal leaves host state exactly as it was.
        for commit in &self.plan.commits {
            check_frontier(frontiers, commit.slot, commit.expect)?;
        }
        for commit in &self.plan.commits {
            frontiers[commit.slot as usize] = commit.after;
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

pub(crate) const SPAN_WORDS: usize = std::mem::size_of::<PrefillSpan>() / 4;

#[derive(Clone)]
pub(crate) struct HostLayout {
    pub(crate) rows: usize,
    pub(crate) decode: usize,
    pub(crate) spans: usize,
    pub(crate) ids: std::ops::Range<usize>,
    pub(crate) pos: std::ops::Range<usize>,
    pub(crate) kvlen: std::ops::Range<usize>,
    pub(crate) decode_slot: std::ops::Range<usize>,
    pub(crate) parked: std::ops::Range<usize>,
    pub(crate) prefill_spans: std::ops::Range<usize>,
}

impl HostLayout {
    pub(crate) fn new(rows: usize, decode: usize, spans: usize) -> crate::Result<Self> {
        let mut next = 0usize;
        let mut take = |count: usize| -> crate::Result<std::ops::Range<usize>> {
            let start = next;
            next = next
                .checked_add(count)
                .ok_or_else(|| RuntimeError::Rejected("mixed step staging overflow".into()))?;
            Ok(start..next)
        };
        let ids = take(rows)?;
        let pos = take(rows)?;
        let kvlen = take(rows)?;
        let decode_slot = take(decode)?;
        let parked = take(rows)?;
        let prefill_spans =
            take(spans.checked_mul(SPAN_WORDS).ok_or_else(|| {
                RuntimeError::Rejected("mixed step span staging overflow".into())
            })?)?;
        Ok(Self {
            rows,
            decode,
            spans,
            ids,
            pos,
            kvlen,
            decode_slot,
            parked,
            prefill_spans,
        })
    }

    pub(crate) fn words(&self) -> usize {
        self.prefill_spans.end
    }
}

pub(crate) fn fill_words(
    layout: &HostLayout,
    words: &mut [u32],
    plan: &plow_asset::mixed_step::Plan,
) -> crate::Result<()> {
    if words.len() < layout.words()
        || plan.rows.len() > layout.rows
        || plan.decode_slots.len() > layout.decode
        || plan.prefill_spans.len() > layout.spans
    {
        return Err(RuntimeError::Rejected(
            "mixed step exceeds preallocated staging".into(),
        ));
    }
    for (index, row) in plan.rows.iter().enumerate() {
        words[layout.ids.start + index] = row.token;
        words[layout.pos.start + index] = row.position;
        words[layout.kvlen.start + index] = row.kv_len;
    }
    for (index, &slot) in plan.decode_slots.iter().enumerate() {
        words[layout.decode_slot.start + index] = slot as u32;
    }
    words[layout.parked.start..layout.parked.start + plan.parked.len()]
        .copy_from_slice(&plan.parked);
    for (index, span) in plan.prefill_spans.iter().enumerate() {
        let at = layout.prefill_spans.start + index * SPAN_WORDS;
        words[at..at + SPAN_WORDS].copy_from_slice(&[
            span.row0,
            span.n_rows,
            span.slot,
            span.flags,
            span.kv_row0,
            span.kv_len,
            span.state_slot,
            span.program,
        ]);
    }
    Ok(())
}

// ================================================================================================
// Unified token batch
// ================================================================================================

/// Reusable host storage for one unified token-batch submission.
///
/// The same discipline as [`MixedStepStaging`], with one addition the token batch needs and the
/// mixed step did not: SLOT GENERATIONS. Staging reads the committed frontier and the
/// generation but changes neither; the commit re-checks BOTH for every span before mutating
/// anything, so a slot recycled while the step was in flight is refused rather than handed
/// another request's output.
///
/// If the body succeeds but the tail fails, the caller [`TokenBatchStaging::discard`]s and
/// publishes nothing. Physical KV may already have changed — that is a fault-path problem for
/// the affected slots, not something this type can roll back, and pretending otherwise is how a
/// stale frontier gets retried against a ring cache that has already wrapped.
pub struct TokenBatchStaging {
    plan: token_batch::Plan,
    pending: bool,
}

impl TokenBatchStaging {
    pub fn with_capacity(row_capacity: usize, request_capacity: usize) -> Self {
        TokenBatchStaging {
            plan: token_batch::Plan::with_capacity(row_capacity, request_capacity),
            pending: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stage<'a>(
        &'a mut self,
        requests: &[token_batch::Request<'_>],
        frontiers: &[u32],
        generations: &[u32],
        row_capacity: u32,
        max_ctx: u32,
        program: u32,
    ) -> Result<&'a token_batch::Plan, StageError> {
        if self.pending {
            return Err(StageError::PendingPlan);
        }
        token_batch::plan_into(
            requests,
            frontiers,
            generations,
            row_capacity,
            max_ctx,
            program,
            &mut self.plan,
        )
        .map_err(StageError::Plan)?;
        self.pending = true;
        Ok(&self.plan)
    }

    pub fn pending_plan(&self) -> Option<&token_batch::Plan> {
        self.pending.then_some(&self.plan)
    }

    /// Publish the logical KV progress after the whole chain succeeds. Every expected frontier
    /// and every slot generation is checked before ANY frontier is changed, so a refusal leaves
    /// host state exactly as it was.
    ///
    /// Frontiers advance only for rows this step consumed as INPUT. A sampled token advances
    /// nothing until it is fed back as an input.
    pub fn commit_after_device_success(
        &mut self,
        frontiers: &mut [u32],
        generations: &[u32],
    ) -> Result<(), StageError> {
        if !self.pending {
            return Err(StageError::NoPendingPlan);
        }
        for commit in &self.plan.pending {
            let slot = commit.slot as usize;
            let (Some(&actual), Some(&generation)) = (frontiers.get(slot), generations.get(slot))
            else {
                return Err(StageError::FrontierCapacity {
                    slot: commit.slot,
                    capacity: frontiers.len().min(generations.len()),
                });
            };
            if generation != commit.generation {
                return Err(StageError::GenerationChanged {
                    request: commit.request,
                    slot: commit.slot,
                    expected: commit.generation,
                    actual: generation,
                });
            }
            if actual != commit.expected_frontier {
                return Err(StageError::FrontierChanged {
                    slot: commit.slot,
                    expected: commit.expected_frontier,
                    actual,
                });
            }
        }
        for commit in &self.plan.pending {
            frontiers[commit.slot as usize] = commit.new_frontier;
        }
        self.pending = false;
        Ok(())
    }

    /// Scatter `S` device-produced ids to their logical requests, in sample order. Exactly one
    /// delivery per selected row: the owner comes from [`token_batch::SampleOwner`], not from
    /// the row number, because packed rows, physical slots and compact output rows are three
    /// different numberings.
    pub fn deliver(
        &self,
        sampled_ids: &[u32],
        out: &mut Vec<(u32, u32)>,
    ) -> Result<(), StageError> {
        if !self.pending {
            return Err(StageError::NoPendingPlan);
        }
        let expected = self.plan.sample_rows as usize;
        if sampled_ids.len() != expected {
            return Err(StageError::SampleRows {
                expected,
                actual: sampled_ids.len(),
            });
        }
        out.clear();
        out.extend(
            self.plan
                .sample_owners
                .iter()
                .zip(sampled_ids)
                .map(|(owner, &id)| (owner.request, id)),
        );
        Ok(())
    }

    pub fn discard(&mut self) {
        self.pending = false;
    }
}

/// Word layout of one token-batch metadata upload, laid out once and reused.
///
/// This is the token-batch twin of [`HostLayout`], and it lives HERE, beside it, for the reason
/// the plan gives: exactly one host staging filler shared by every backend adapter. A second
/// copy is how two device layouts drift apart while both look tested. A backend that writes the
/// arrays directly — the CPU engine does — reads the ranges and skips the upload; it is not
/// forced through a fake one.
#[derive(Clone, Debug)]
pub struct TokenBatchLayout {
    pub row_capacity: usize,
    pub spans: usize,
    pub samples: usize,
    pub input_ids: std::ops::Range<usize>,
    pub positions: std::ops::Range<usize>,
    pub active: std::ops::Range<usize>,
    pub span_table: std::ops::Range<usize>,
    pub sample_rows: std::ops::Range<usize>,
}

impl TokenBatchLayout {
    pub fn new(row_capacity: usize, spans: usize, samples: usize) -> crate::Result<Self> {
        let mut next = 0usize;
        let mut take = |count: usize| -> crate::Result<std::ops::Range<usize>> {
            let start = next;
            next = next
                .checked_add(count)
                .ok_or_else(|| RuntimeError::Rejected("token batch staging overflow".into()))?;
            Ok(start..next)
        };
        let input_ids = take(row_capacity)?;
        let positions = take(row_capacity)?;
        let active = take(row_capacity)?;
        let span_table =
            take(spans.checked_mul(SPAN_WORDS).ok_or_else(|| {
                RuntimeError::Rejected("token batch span staging overflow".into())
            })?)?;
        let sample_rows = take(samples)?;
        Ok(TokenBatchLayout {
            row_capacity,
            spans,
            samples,
            input_ids,
            positions,
            active,
            span_table,
            sample_rows,
        })
    }

    pub fn words(&self) -> usize {
        self.sample_rows.end
    }
}

/// Fill one metadata slab from a plan. The arrays are written at their FULL compiled extent —
/// padding included — because the descriptor declares `row_capacity` and a device that reads a
/// row past `real_rows` must find `active == 0` there, not whatever the previous step left.
pub fn fill_token_batch_words(
    layout: &TokenBatchLayout,
    words: &mut [u32],
    plan: &token_batch::Plan,
) -> crate::Result<()> {
    if words.len() < layout.words()
        || plan.row_capacity as usize != layout.row_capacity
        || plan.spans.len() > layout.spans
        || plan.sample_rows as usize > layout.samples
    {
        return Err(RuntimeError::Rejected(
            "token batch exceeds preallocated staging".into(),
        ));
    }
    if plow_asset::token_batch::validate(&plan.batch()).is_err() {
        return Err(RuntimeError::Rejected(
            "token batch plan failed the shared row-resolver invariants".into(),
        ));
    }
    words[layout.input_ids.clone()].copy_from_slice(&plan.input_ids);
    words[layout.positions.clone()].copy_from_slice(&plan.positions);
    words[layout.active.clone()].copy_from_slice(&plan.active);
    for (index, span) in plan.spans.iter().enumerate() {
        let at = layout.span_table.start + index * SPAN_WORDS;
        words[at..at + SPAN_WORDS].copy_from_slice(&[
            span.row0,
            span.n_rows,
            span.slot,
            span.flags,
            span.kv_row0,
            span.kv_len,
            span.state_slot,
            span.program,
        ]);
    }
    let samples = plan.sample_rows as usize;
    words[layout.sample_rows.start..layout.sample_rows.start + samples]
        .copy_from_slice(&plan.sample_input_rows[..samples]);
    Ok(())
}

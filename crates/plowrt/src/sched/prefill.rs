//! Backend-neutral admission for one cross-request prefill launch.

use packet::dev::PrefillSpan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpanPolicy {
    /// A planned span is indivisible. Used by backends whose packet program
    /// fixes each request's next chunk before admission.
    Whole,
    /// Divide the row budget across the rotated candidates. Short requests
    /// return unused rows to other candidates in the same launch.
    FairSplit,
}

const MAX_SLOTS: usize = u128::BITS as usize;

#[derive(Debug, Eq, PartialEq)]
pub struct PrefillPack {
    spans: [PrefillSpan; MAX_SLOTS],
    len: usize,
    pub bucket_rows: u32,
}

impl Default for PrefillPack {
    fn default() -> Self {
        Self {
            spans: [PrefillSpan::default(); MAX_SLOTS],
            len: 0,
            bucket_rows: 0,
        }
    }
}

impl PrefillPack {
    pub fn spans(&self) -> &[PrefillSpan] {
        &self.spans[..self.len]
    }

    pub fn dense_rows(&self) -> u32 {
        self.spans().iter().map(|span| span.n_rows).sum()
    }

    pub fn padding_rows(&self) -> u32 {
        self.bucket_rows.saturating_sub(self.dense_rows())
    }

    pub fn last_slot(&self) -> Option<usize> {
        self.spans().last().map(|span| span.slot as usize)
    }

    /// Keep at most `max` spans (plans/unified-token-batch.md §5.4, §7).
    ///
    /// SELECTION, NOT TRUNCATION, and the difference is where it happens. This runs while the
    /// pack is still a candidate list -- before any cursor moves, before any frontier is
    /// committed -- so a request whose span is dropped is simply not admitted this tick, exactly
    /// as it is not admitted when the row budget runs out. Truncating a plan that has already
    /// been staged is the silently-short answer §9 forbids; that case is refused at
    /// `stage_packed_prefill` instead.
    pub fn limit_spans(&mut self, max: usize) {
        self.len = self.len.min(max);
    }
}

/// Rotate candidates by physical slot, select one compatible packet program,
/// and pack dense rows under that program's bucket and the current tick limit.
pub fn admit(
    candidates: impl IntoIterator<Item = PrefillSpan>,
    row_limit: u32,
    start: usize,
    capacity: usize,
    policy: SpanPolicy,
    mut program_rows: impl FnMut(u32) -> Option<u32>,
) -> PrefillPack {
    if row_limit == 0 || capacity == 0 || capacity > MAX_SLOTS {
        return PrefillPack::default();
    }
    let mut by_slot = [None; MAX_SLOTS];
    let mut occupied = 0u128;
    for span in candidates {
        let Ok(slot) = usize::try_from(span.slot) else {
            continue;
        };
        if slot >= capacity {
            continue;
        }
        let bit = 1u128 << slot;
        let valid = span.n_rows != 0
            && occupied & bit == 0
            && span.state_slot == span.slot
            && span.kv_row0.checked_add(span.n_rows) == Some(span.kv_len)
            && (policy == SpanPolicy::FairSplit || span.n_rows <= row_limit);
        if valid {
            by_slot[slot] = Some(span);
            occupied |= bit;
        }
    }
    let start = start % capacity;
    let slots = (0..capacity).map(|offset| (start + offset) % capacity);
    let Some(program) = slots
        .clone()
        .find_map(|slot| by_slot[slot].map(|span| span.program))
    else {
        return PrefillPack::default();
    };
    let Some(bucket_rows) = program_rows(program).map(|rows| rows.min(row_limit)) else {
        return PrefillPack::default();
    };
    if bucket_rows == 0 {
        return PrefillPack::default();
    }

    let count = slots
        .clone()
        .filter(|&slot| by_slot[slot].is_some_and(|span| span.program == program))
        .count();
    let mut rows = 0u32;
    let mut pack = PrefillPack {
        bucket_rows,
        ..PrefillPack::default()
    };
    for (index, mut span) in slots
        .filter_map(|slot| by_slot[slot])
        .filter(|span| span.program == program)
        .enumerate()
    {
        let remaining = bucket_rows - rows;
        if remaining == 0 {
            break;
        }
        let take = match policy {
            SpanPolicy::Whole if span.n_rows <= remaining => span.n_rows,
            SpanPolicy::Whole => continue,
            SpanPolicy::FairSplit => {
                let candidates_left = u32::try_from(count - index).unwrap_or(u32::MAX);
                span.n_rows.min(remaining.div_ceil(candidates_left))
            }
        };
        span.row0 = rows;
        span.n_rows = take;
        span.kv_len = span.kv_row0 + take;
        rows += take;
        pack.spans[pack.len] = span;
        pack.len += 1;
    }
    if policy == SpanPolicy::FairSplit && rows < bucket_rows {
        // Preserve each initial share; reclaim unused rows in rotation order.
        let mut spare = bucket_rows - rows;
        let mut row0 = 0;
        for span in &mut pack.spans[..pack.len] {
            let offered = by_slot[span.slot as usize]
                .expect("admitted candidate")
                .n_rows;
            let extra = spare.min(offered - span.n_rows);
            spare -= extra;
            span.n_rows += extra;
            span.row0 = row0;
            span.kv_len = span.kv_row0 + span.n_rows;
            row0 += span.n_rows;
        }
    }
    pack
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(slot: u32, position: u32, rows: u32, program: u32) -> PrefillSpan {
        PrefillSpan {
            row0: 99,
            n_rows: rows,
            slot,
            flags: 0,
            kv_row0: position,
            kv_len: position + rows,
            state_slot: slot,
            program,
        }
    }

    #[test]
    fn limit_spans_keeps_the_rotation_prefix_and_is_idempotent() {
        // The D-class cap (plans/unified-token-batch.md §5.4) applied to a legal 3-span pack.
        // The kept spans must be the FIRST ones in rotation order, unchanged, so the requests
        // that lose their span are simply the ones not admitted this tick.
        let mut pack = admit(
            [span(0, 0, 4, 7), span(1, 8, 4, 7), span(2, 16, 4, 7)],
            64,
            0,
            3,
            SpanPolicy::Whole,
            |_| Some(64),
        );
        let all: Vec<_> = pack.spans().to_vec();
        assert_eq!(all.len(), 3);
        pack.limit_spans(u32::MAX as usize);
        assert_eq!(pack.spans(), all.as_slice());
        pack.limit_spans(1);
        assert_eq!(pack.spans(), &all[..1]);
        assert_eq!(pack.dense_rows(), 4);
        pack.limit_spans(1);
        assert_eq!(pack.spans(), &all[..1]);
        pack.limit_spans(0);
        assert!(pack.spans().is_empty());
    }

    #[test]
    fn fair_split_rotates_and_reclaims_short_request_rows() {
        let pack = admit(
            [span(0, 40, 100, 7), span(1, 8, 5, 7), span(2, 70, 100, 7)],
            64,
            1,
            3,
            SpanPolicy::FairSplit,
            |_| Some(64),
        );
        assert_eq!(
            pack.spans()
                .iter()
                .map(|span| (span.slot, span.row0, span.kv_row0, span.n_rows, span.kv_len))
                .collect::<Vec<_>>(),
            [(1, 0, 8, 5, 13), (2, 5, 70, 30, 100), (0, 35, 40, 29, 69)]
        );
        assert_eq!(pack.dense_rows(), 64);
        assert_eq!(pack.padding_rows(), 0);
        assert_eq!(pack.last_slot(), Some(0));
    }

    #[test]
    fn fair_split_reclaims_rows_from_a_short_last_request() {
        let pack = admit(
            [span(0, 40, 100, 7), span(1, 8, 1, 7)],
            64,
            0,
            2,
            SpanPolicy::FairSplit,
            |_| Some(64),
        );
        assert_eq!(
            pack.spans()
                .iter()
                .map(|s| (s.slot, s.row0, s.n_rows, s.kv_row0, s.kv_len))
                .collect::<Vec<_>>(),
            [(0, 0, 63, 40, 103), (1, 63, 1, 8, 9)]
        );
        assert_eq!(pack.padding_rows(), 0);
        assert_eq!(pack.last_slot(), Some(1));
    }

    #[test]
    fn fair_split_uses_available_rows_without_exceeding_request_caps() {
        for shape in 0..625u32 {
            let mut encoded = shape;
            let demands: [u32; 4] = std::array::from_fn(|_| {
                let rows = encoded % 5;
                encoded /= 5;
                rows
            });
            for start in 0..4 {
                for budget in 1..=16 {
                    let pack = admit(
                        demands
                            .iter()
                            .enumerate()
                            .map(|(slot, &rows)| span(slot as u32, 10 + slot as u32, rows, 7)),
                        budget,
                        start,
                        4,
                        SpanPolicy::FairSplit,
                        |_| Some(16),
                    );
                    assert_eq!(pack.dense_rows(), budget.min(demands.iter().sum()));
                    let mut row = 0;
                    let mut seen = 0u8;
                    let mut previous = None;
                    for s in pack.spans() {
                        let slot = s.slot as usize;
                        let order = (slot + 4 - start) % 4;
                        assert!(previous.is_none_or(|p| p < order));
                        previous = Some(order);
                        assert_eq!(seen & (1 << slot), 0);
                        seen |= 1 << slot;
                        assert!(s.n_rows > 0 && s.n_rows <= demands[slot]);
                        assert_eq!(s.row0, row);
                        assert_eq!(s.kv_row0, 10 + s.slot);
                        assert_eq!(s.kv_len, s.kv_row0 + s.n_rows);
                        assert_eq!(s.state_slot, s.slot);
                        assert_eq!(s.program, 7);
                        row += s.n_rows;
                    }
                }
            }
        }
    }

    #[test]
    fn whole_spans_preserve_program_and_leave_bucket_padding() {
        let pack = admit(
            [span(0, 0, 5, 3), span(1, 20, 2, 4), span(2, 9, 2, 3)],
            8,
            2,
            3,
            SpanPolicy::Whole,
            |program| (program == 3).then_some(8),
        );
        assert_eq!(
            pack.spans()
                .iter()
                .map(|span| (span.slot, span.row0, span.n_rows, span.program))
                .collect::<Vec<_>>(),
            [(2, 0, 2, 3), (0, 2, 5, 3)]
        );
        assert_eq!(pack.dense_rows(), 7);
        assert_eq!(pack.padding_rows(), 1);
    }

    #[test]
    fn invalid_or_duplicate_physical_slots_are_excluded() {
        let mut wrong_state = span(1, 0, 2, 0);
        wrong_state.state_slot = 0;
        let mut wrong_frontier = span(2, 4, 2, 0);
        wrong_frontier.kv_len = 9;
        let pack = admit(
            [
                wrong_state,
                wrong_frontier,
                span(3, 0, 2, 0),
                span(0, 1, 3, 0),
                span(0, 1, 3, 0),
            ],
            8,
            0,
            3,
            SpanPolicy::FairSplit,
            |_| Some(8),
        );
        assert_eq!(pack.spans().len(), 1);
        assert_eq!(pack.spans()[0].slot, 0);
        assert!(admit(
            [span(0, 0, 1, 0)],
            1,
            0,
            MAX_SLOTS + 1,
            SpanPolicy::FairSplit,
            |_| Some(1),
        )
        .spans()
        .is_empty());
    }

    #[test]
    fn saturated_launches_rotate_the_first_admitted_request() {
        let candidates = || [span(0, 0, 8, 0), span(1, 0, 8, 0), span(2, 0, 8, 0)];
        let first = admit(candidates(), 2, 0, 3, SpanPolicy::FairSplit, |_| Some(2));
        assert_eq!(
            first
                .spans()
                .iter()
                .map(|span| span.slot)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        let second = admit(
            candidates(),
            2,
            first.last_slot().unwrap() + 1,
            3,
            SpanPolicy::FairSplit,
            |_| Some(2),
        );
        assert_eq!(
            second
                .spans()
                .iter()
                .map(|span| span.slot)
                .collect::<Vec<_>>(),
            [2, 0]
        );
    }

    /// Dense co-packing is arithmetic, not a feature flag: a whole-span pack can only hold two
    /// requests when two chunks fit in one compiled prefill rung. A chunk sized AT the widest
    /// rung therefore admits exactly one span, the mux's `packed.len() >= 2` test fails, and
    /// packing silently never runs — which is what `PLOW_PF_CHUNK=8192` does to Gemma 4 31B on
    /// gfx942, whose ladder tops out at 8192. Halving the chunk is what makes it reachable.
    #[test]
    fn a_chunk_filling_the_widest_rung_admits_one_span_and_half_of_it_admits_two() {
        let rung = 8192;
        let pack_at = |chunk: u32| {
            admit(
                (0..4).map(|slot| span(slot, chunk, chunk, 0)),
                u32::MAX,
                0,
                4,
                SpanPolicy::Whole,
                |_| Some(rung),
            )
        };
        assert_eq!(pack_at(rung).spans().len(), 1);
        assert_eq!(pack_at(rung / 2).spans().len(), 2);
        assert_eq!(pack_at(rung / 4).spans().len(), 4);
    }

    #[test]
    fn fair_split_never_emits_zero_rows_when_candidates_outnumber_budget() {
        let pack = admit(
            (0..8).map(|slot| span(slot, 0, 16, 0)),
            3,
            0,
            8,
            SpanPolicy::FairSplit,
            |_| Some(3),
        );
        assert_eq!(pack.spans().len(), 3);
        assert!(pack.spans().iter().all(|span| span.n_rows == 1));
    }
}

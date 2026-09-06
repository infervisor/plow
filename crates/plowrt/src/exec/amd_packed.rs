use packet::dev::{PrefillSpan, PREFILL_SPAN_RESET_STATE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedRows {
    pub n_spans: u32,
    pub real_rows: u32,
}

/// Validate the row layout shared by AMD packed prefill and a future mixed-step adapter.
/// `row_base` reserves leading active rows, such as compact decode rows; prefill spans cover
/// the dense range immediately after them and the remaining rows are parked padding.
pub(super) fn validate_rows(
    program: u32,
    row_base: u32,
    row_capacity: u32,
    slot_capacity: usize,
    spans: &[PrefillSpan],
    parked: &[u32],
) -> Result<PackedRows, String> {
    if row_base > row_capacity {
        return Err(format!(
            "packed rows begin at {row_base}, past row capacity {row_capacity}"
        ));
    }

    let mut row = row_base;
    for (index, span) in spans.iter().enumerate() {
        if span.row0 != row || span.n_rows == 0 {
            return Err(format!(
                "packed span {index} is not dense: row0={} n_rows={} expected row0={row}",
                span.row0, span.n_rows
            ));
        }
        if span.flags & !PREFILL_SPAN_RESET_STATE != 0 {
            return Err(format!(
                "packed span {index} has unknown flags {:#x}",
                span.flags
            ));
        }
        let reset = span.flags & PREFILL_SPAN_RESET_STATE != 0;
        if reset != (span.kv_row0 == 0) {
            return Err(format!(
                "packed span {index} reset flag disagrees with kv_row0={}",
                span.kv_row0
            ));
        }
        let kv_end = span
            .kv_row0
            .checked_add(span.n_rows)
            .ok_or_else(|| format!("packed span {index} KV range overflows u32"))?;
        if kv_end != span.kv_len {
            return Err(format!(
                "packed span {index} has kv_row0+n_rows={kv_end}, kv_len={}",
                span.kv_len
            ));
        }
        if span.slot as usize >= slot_capacity
            || span.state_slot as usize >= slot_capacity
            || span.slot != span.state_slot
        {
            return Err(format!(
                "packed span {index} has incompatible KV/state slots {}/{} for capacity {slot_capacity}",
                span.slot, span.state_slot
            ));
        }
        if span.program != program {
            return Err(format!(
                "packed span {index} names program {}, staged for {program}",
                span.program
            ));
        }
        if spans[..index].iter().any(|prior| prior.slot == span.slot) {
            return Err(format!(
                "packed slot {} appears in more than one span",
                span.slot
            ));
        }
        row = row
            .checked_add(span.n_rows)
            .ok_or_else(|| "packed row count overflows u32".to_string())?;
    }

    if row > row_capacity {
        return Err(format!(
            "packed rows end at {row}, past row capacity {row_capacity}"
        ));
    }
    if parked.len() != row_capacity as usize
        || parked.iter().any(|&value| value > 1)
        || parked[..row as usize].iter().any(|&value| value != 0)
        || parked[row as usize..].iter().any(|&value| value == 0)
    {
        return Err(format!(
            "packed parked mask must have {row_capacity} binary rows, active [0,{row})=0 and padding [{row},{row_capacity})!=0 (got {})",
            parked.len()
        ));
    }

    Ok(PackedRows {
        n_spans: spans
            .len()
            .try_into()
            .map_err(|_| "packed span count exceeds u32".to_string())?,
        real_rows: row,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use plow_asset::mixed_step::{self, DecodeRequest, PrefillRequest};

    fn span(row0: u32, slot: u32, program: u32) -> PrefillSpan {
        PrefillSpan {
            row0,
            n_rows: 2,
            slot,
            flags: 0,
            kv_row0: 8,
            kv_len: 10,
            state_slot: slot,
            program,
        }
    }

    #[test]
    fn accepts_prefill_after_leading_decode_rows() {
        let rows = validate_rows(
            7,
            2,
            8,
            4,
            &[span(2, 1, 7), span(4, 3, 7)],
            &[0, 0, 0, 0, 0, 0, 1, 1],
        )
        .unwrap();
        assert_eq!(
            rows,
            PackedRows {
                n_spans: 2,
                real_rows: 6
            }
        );
    }

    #[test]
    fn rejects_a_prefill_span_that_overwrites_decode_rows() {
        assert!(validate_rows(7, 2, 8, 4, &[span(0, 1, 7)], &[0, 0, 1, 1, 1, 1, 1, 1],).is_err());
    }

    #[test]
    fn accepts_the_backend_neutral_mixed_plan_directly() {
        let plan = mixed_step::plan(
            &[DecodeRequest {
                slot: 2,
                state_slot: 2,
                token: 7,
            }],
            &[PrefillRequest {
                slot: 1,
                state_slot: 1,
                start: 4,
                tokens: &[8, 9],
                prompt_len: 12,
            }],
            &[0, 4, 8, 12],
            8,
            32,
            7,
        )
        .unwrap();

        let rows = validate_rows(
            7,
            plan.decode_rows,
            plan.rows.len() as u32,
            4,
            &plan.prefill_spans,
            &plan.parked,
        )
        .unwrap();
        assert_eq!(rows.real_rows, plan.real_rows);
        assert_eq!(rows.n_spans, 1);
    }
}

use packet::dev::{DevOp, PrefillSpan, PREFILL_SPAN_RESET_STATE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedRows {
    pub n_spans: u32,
    pub real_rows: u32,
}

/// How many request spans one packed launch of a program may carry, as far as its RECURRENT
/// operators are concerned. `u32::MAX` means "as many as the row budget fits".
///
/// This is §3's D class in `plans/unified-token-batch.md` made executable, restricted to the
/// operator family that actually has the property: a recurrent operator's carried state is an
/// operand whose SHAPE has no request axis (`QwenGdnPrefill`'s `state`/`outstate` `[1,HV,V,K]`,
/// `QwenGdnConvPrefill`'s `history[1,C,W-1]`), so a launch either has a per-span form that walks
/// the span table itself, or it can express exactly one request. §5.4's contract is that the
/// limit is STATED and enforced, not that it is hidden: the dense, MoE, norm and output work of
/// a GDN/KDA model still packs.
///
/// Three dispositions, and the third is the one this file exists for:
///
/// * **Per-span form present.** The AMD arm loops `prog->prefill_spans` itself
///   (`d_kda_chunk_*_packed_bt64`, `d_kda_conv3_packed`, `d_kda_state_step_packed`). No limit.
/// * **Row-agnostic over the packed row axis.** `KdaGate` and `KdaGatedNorm` are elementwise /
///   per-row over `T` with no carried state at all — class A, no limit.
/// * **Single-sequence, no per-span form.** One span, or the program is refused outright when
///   even one span would not be correct.
///
/// An unlisted recurrent opcode is NOT given the benefit of the doubt. §3: "An opcode with no
/// classification is treated as C and refused." That matters more here than anywhere else,
/// because AMD's interpreter dispatch `default:` is a silent NOP outside `PLOW_MIXED_STEP`
/// builds, so an unrouted recurrent op leaves its state exactly as it found it and the run
/// completes, fluently, on the previous request's state.
pub(super) fn recurrent_span_limit(ops: impl IntoIterator<Item = u16>) -> Result<u32, String> {
    let mut limit = u32::MAX;
    for raw in ops {
        let Some(op) = DevOp::from_u16(raw) else {
            continue;
        };
        match op {
            // Per-span arms exist in runtime/amd/op_kda.h; the dispatch in interp.hip selects
            // them under PLOW_PACKED_PREFILL_KDA_CONSUMERS.
            DevOp::KdaChunkPrepare
            | DevOp::KdaChunkIntra
            | DevOp::KdaChunkWu
            | DevOp::KdaChunkCarry
            | DevOp::KdaConv3
            | DevOp::KdaStateStep
            | DevOp::KdaStateStepG => {}
            // Stateless over the packed row axis.
            DevOp::KdaGate | DevOp::KdaGatedNorm => {}
            // Single-sequence with no per-span arm: one request per launch.
            DevOp::KdaDecodeFused => limit = limit.min(1),
            // Single-sequence prefill state with no per-span arm. `KdaConv`/`KdaConvStateStepG`
            // are additionally rejected by `packed_kda_compatible`; naming them here too keeps
            // the refusal legible when a program carries one and no other KDA operator.
            DevOp::KdaConv
            | DevOp::KdaConvStateStepG
            | DevOp::Mamba2Scan
            | DevOp::QwenGdnConvPrefill
            | DevOp::QwenGdnQkvPrep
            | DevOp::QwenGdnGatePrep
            | DevOp::QwenGdnPrefill => {
                return Err(format!(
                    "packed prefill has no per-span arm for {op:?} (op {raw}): its carried state \
                     is an operand with no request axis and no AMD arm reads the span table for \
                     it, so a packed launch would run every span against one request's state. \
                     Serve this program on the isolated prefill route, or land the per-span arm \
                     and its oracle first (plans/unified-token-batch.md §5.4)."
                ));
            }
            // Decode-shaped recurrent ops: their row axis is the SLOT (`active[B]`), not the
            // token. A packed-prefill launch's rows are tokens of a span, so binding one of
            // these to a span table is a category error, not a missing arm.
            DevOp::QwenGdnConv
            | DevOp::QwenGdnStep
            | DevOp::QwenGatedNorm
            | DevOp::QwenQGateSplit
            | DevOp::QwenSigmoidGate
            | DevOp::QwenRmsNorm
            | DevOp::QwenHeadNormRope => {
                return Err(format!(
                    "packed prefill cannot carry {op:?} (op {raw}): it is indexed by decode SLOT \
                     through the ISA's `active[B]` mask, and a packed-prefill row is a token of a \
                     span. This program belongs on the decode path (plans/unified-token-batch.md \
                     §5.4)."
                ));
            }
            _ => {}
        }
    }
    Ok(limit)
}

/// Validate a UNIFIED TOKEN BATCH row layout: the host twin of `plow_tb_view`'s predicates
/// in `runtime/amd/token_batch.h`, run before anything reaches a device.
///
/// It differs from [`validate_rows`] in exactly the two ways the contract does
/// (`plans/unified-token-batch.md` §4.3, §4.4):
///
/// * **There is no decode prefix.** The spans cover exactly `[0, real_rows)` from zero.
/// * **A completing prompt owns two spans on one slot** — its terminal token as a leading
///   length-one span, its body as an ordinary span — so "one slot, one span" is replaced by
///   the stronger statement that the pair must be contiguous in KV: the terminal span starts
///   exactly where the body span ends.
///
/// The device traps on a violation and AMD's dispatch `default:` neither writes nor traps, so
/// the point of doing it here is to refuse by name instead of trapping a wavefront.
pub(super) fn validate_token_batch_rows(
    program: u32,
    row_capacity: u32,
    slot_capacity: usize,
    leading: usize,
    spans: &[PrefillSpan],
    parked: &[u32],
) -> Result<PackedRows, String> {
    if spans.is_empty() {
        return Err("token batch has no spans; the descriptor must cover [0, M)".into());
    }
    // `plow_tb_decode_spans` defines the attention partition — and, through
    // `PLOW_SAMPLE_ROWS`, the selection stage's row count — as the LEADING RUN of length-one
    // spans. The host decides which rows it will deliver; if the device would count a
    // different number the two disagree about who owns a sampled id, so the disagreement is
    // refused here rather than discovered as an off-by-one token.
    //
    // The case that reaches this is a body span of length one directly behind the leading run,
    // which happens when a two-token prompt is completed in one step. Refusing leaves it to
    // the ordinary route, which is a scheduling limit and not a wrong answer.
    let device_leading = spans.iter().take_while(|s| s.n_rows == 1).count();
    if device_leading != leading {
        return Err(format!(
            "token batch would sample {device_leading} leading row(s) but the host planned \
             {leading}: a body span of length one sits directly behind the leading run"
        ));
    }
    let mut row = 0u32;
    for (index, span) in spans.iter().enumerate() {
        if span.row0 != row || span.n_rows == 0 {
            return Err(format!(
                "token-batch span {index} is not dense: row0={} n_rows={} expected row0={row}",
                span.row0, span.n_rows
            ));
        }
        if span.flags & !PREFILL_SPAN_RESET_STATE != 0 {
            return Err(format!(
                "token-batch span {index} has unknown flags {:#x}",
                span.flags
            ));
        }
        if (span.flags & PREFILL_SPAN_RESET_STATE != 0) != (span.kv_row0 == 0) {
            return Err(format!(
                "token-batch span {index} reset flag disagrees with kv_row0={}",
                span.kv_row0
            ));
        }
        let kv_end = span
            .kv_row0
            .checked_add(span.n_rows)
            .ok_or_else(|| format!("token-batch span {index} KV range overflows u32"))?;
        if kv_end != span.kv_len {
            return Err(format!(
                "token-batch span {index} has kv_row0+n_rows={kv_end}, kv_len={}",
                span.kv_len
            ));
        }
        if span.slot as usize >= slot_capacity
            || span.state_slot as usize >= slot_capacity
            || span.slot != span.state_slot
        {
            return Err(format!(
                "token-batch span {index} has incompatible KV/state slots {}/{} for capacity \
                 {slot_capacity}",
                span.slot, span.state_slot
            ));
        }
        if span.program != program {
            return Err(format!(
                "token-batch span {index} names program {}, staged for {program}",
                span.program
            ));
        }
        let siblings: Vec<&PrefillSpan> = spans[..index]
            .iter()
            .filter(|prior| prior.slot == span.slot)
            .collect();
        match siblings.as_slice() {
            [] => {}
            // The terminal/body pair. The terminal span leads, so the earlier one is it, and
            // this one must resume exactly where the terminal token does NOT overlap: the body
            // ends where the terminal begins.
            [terminal] => {
                if terminal.n_rows != 1 || terminal.kv_row0 != kv_end {
                    return Err(format!(
                        "token-batch slot {} owns two spans that are not a terminal/body pair: \
                         terminal kv[{}, {}) body kv[{}, {})",
                        span.slot, terminal.kv_row0, terminal.kv_len, span.kv_row0, kv_end
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "token-batch slot {} appears in more than two spans",
                    span.slot
                ))
            }
        }
        row = row
            .checked_add(span.n_rows)
            .ok_or_else(|| "token-batch row count overflows u32".to_string())?;
    }
    if row > row_capacity {
        return Err(format!(
            "token-batch rows end at {row}, past row capacity {row_capacity}"
        ));
    }
    if parked.len() != row_capacity as usize
        || parked.iter().any(|&value| value > 1)
        || parked[..row as usize].iter().any(|&value| value != 0)
        || parked[row as usize..].iter().any(|&value| value == 0)
    {
        return Err(format!(
            "token-batch parked mask must have {row_capacity} binary rows, active [0,{row})=0 \
             and padding [{row},{row_capacity})!=0 (got {})",
            parked.len()
        ));
    }
    Ok(PackedRows {
        n_spans: spans
            .len()
            .try_into()
            .map_err(|_| "token-batch span count exceeds u32".to_string())?,
        real_rows: row,
    })
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

    fn tb_span(row0: u32, n_rows: u32, slot: u32, kv_row0: u32) -> PrefillSpan {
        PrefillSpan {
            row0,
            n_rows,
            slot,
            flags: u32::from(kv_row0 == 0) * PREFILL_SPAN_RESET_STATE,
            kv_row0,
            kv_len: kv_row0 + n_rows,
            state_slot: slot,
            program: 3,
        }
    }

    fn parked(real: usize, capacity: usize) -> Vec<u32> {
        let mut mask = vec![0u32; capacity];
        mask[real..].fill(1);
        mask
    }

    /// The shape the route actually stages: decode spans of length one, then the terminal row
    /// of a completing prompt, then that prompt's body — two spans on one slot, contiguous in
    /// KV, covering `[0, M)` from zero.
    #[test]
    fn a_terminal_and_body_pair_on_one_slot_is_the_legal_prefix_free_shape() {
        let spans = [
            tb_span(0, 1, 0, 100),
            tb_span(1, 1, 2, 119),
            tb_span(2, 49, 2, 70),
            tb_span(51, 80, 3, 0),
        ];
        let rows = validate_token_batch_rows(3, 256, 4, 2, &spans, &parked(131, 256)).unwrap();
        assert_eq!((rows.n_spans, rows.real_rows), (4, 131));
    }

    /// `validate_rows` — the decode-band validator — must keep REFUSING that shape, or the two
    /// contracts have quietly become one and `plow_mixed_prefill_span` will resolve a row from
    /// a table that has no decode prefix.
    #[test]
    fn the_decode_band_validator_still_refuses_a_prefix_free_table() {
        let spans = [tb_span(0, 1, 0, 100), tb_span(1, 1, 2, 119), tb_span(2, 49, 2, 70)];
        let err = validate_rows(3, 0, 256, 4, &spans, &parked(51, 256)).unwrap_err();
        assert!(err.contains("more than one span"), "{err}");
    }

    /// The device counts the leading run of length-one spans; the host counts the rows it will
    /// deliver. A two-token prompt completed in one step makes those disagree, and a
    /// disagreement about who owns a sampled id is refused rather than delivered.
    #[test]
    fn a_length_one_body_behind_the_leading_run_is_refused_by_name() {
        let spans = [tb_span(0, 1, 1, 1), tb_span(1, 1, 1, 0)];
        let err = validate_token_batch_rows(3, 64, 4, 1, &spans, &parked(2, 64)).unwrap_err();
        assert!(err.contains("leading row"), "{err}");
    }

    #[test]
    fn a_gap_an_overlap_and_a_third_span_on_one_slot_all_refuse() {
        // Gap: the second span does not start where the first ended.
        let gap = [tb_span(0, 1, 0, 5), tb_span(2, 3, 1, 0)];
        assert!(validate_token_batch_rows(3, 64, 4, 1, &gap, &parked(4, 64)).is_err());
        // kv_row0 + n_rows != kv_len.
        let mut bad = tb_span(0, 4, 0, 0);
        bad.kv_len = 9;
        assert!(validate_token_batch_rows(3, 64, 4, 0, &[bad], &parked(4, 64)).is_err());
        // Three spans on one slot is never a terminal/body pair.
        let three = [tb_span(0, 1, 1, 9), tb_span(1, 1, 1, 8), tb_span(2, 1, 1, 7)];
        assert!(validate_token_batch_rows(3, 64, 4, 3, &three, &parked(3, 64)).is_err());
    }

    /// Padding is inert and outside every span; an unparked pad row would be a live row nobody
    /// planned.
    #[test]
    fn the_parked_mask_must_be_exactly_the_padded_suffix() {
        let spans = [tb_span(0, 1, 0, 100), tb_span(1, 3, 1, 0)];
        assert!(validate_token_batch_rows(3, 64, 4, 1, &spans, &parked(4, 64)).is_ok());
        let mut wrong = parked(4, 64);
        wrong[10] = 0;
        assert!(validate_token_batch_rows(3, 64, 4, 1, &spans, &wrong).is_err());
    }

    #[test]
    fn the_packed_kda_operator_group_carries_as_many_spans_as_fit() {
        // Every one of these has a `d_kda_*_packed_bt64` arm that walks `prog->prefill_spans`
        // itself, so the span table is the only thing bounding the launch.
        let ops = [
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkIntra,
            DevOp::KdaChunkWu,
            DevOp::KdaChunkCarry,
            DevOp::KdaConv3,
            DevOp::KdaStateStepG,
            DevOp::KdaGate,
            DevOp::KdaGatedNorm,
            DevOp::Gemm,
            DevOp::FlashPrefill,
        ];
        assert_eq!(
            recurrent_span_limit(ops.iter().map(|&op| op as u16)),
            Ok(u32::MAX)
        );
    }

    #[test]
    fn a_single_sequence_recurrent_arm_is_capped_at_one_span() {
        assert_eq!(
            recurrent_span_limit([DevOp::KdaDecodeFused as u16, DevOp::Gemm as u16]),
            Ok(1)
        );
    }

    #[test]
    fn recurrent_operators_with_no_per_span_arm_are_refused_by_name() {
        // The four that would otherwise fall through `check_packed_prefill_program` to `Ok`:
        // none of them belongs to a family that function recognises, and AMD's dispatch
        // `default:` writes nothing rather than trapping.
        for op in [
            DevOp::QwenGdnPrefill,
            DevOp::QwenGdnConvPrefill,
            DevOp::QwenGdnQkvPrep,
            DevOp::QwenGdnGatePrep,
            DevOp::Mamba2Scan,
            DevOp::KdaConv,
            DevOp::KdaConvStateStepG,
        ] {
            let error = recurrent_span_limit([DevOp::Gemm as u16, op as u16])
                .expect_err("must refuse, not cap");
            assert!(
                error.contains(&format!("{op:?}")) && error.contains("per-span"),
                "refusal must name the operator and the missing capability: {error}"
            );
        }
    }

    #[test]
    fn decode_shaped_recurrent_operators_are_refused_as_a_category_error() {
        for op in [
            DevOp::QwenGdnConv,
            DevOp::QwenGdnStep,
            DevOp::QwenGatedNorm,
            DevOp::QwenQGateSplit,
            DevOp::QwenSigmoidGate,
            DevOp::QwenRmsNorm,
            DevOp::QwenHeadNormRope,
        ] {
            let error = recurrent_span_limit([op as u16]).expect_err("must refuse");
            assert!(
                error.contains(&format!("{op:?}")) && error.contains("active[B]"),
                "refusal must name the operator and the mask it is indexed by: {error}"
            );
        }
    }

    #[test]
    fn a_program_with_no_recurrent_operator_is_unbounded() {
        assert_eq!(
            recurrent_span_limit([
                DevOp::Gemm as u16,
                DevOp::FlashPrefill as u16,
                DevOp::RmsNorm as u16,
                DevOp::SoftCap as u16,
            ]),
            Ok(u32::MAX)
        );
    }

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

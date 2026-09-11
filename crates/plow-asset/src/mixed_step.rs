use crate::aux_program;
use packet::dev::{DevOp, PrefillSpan, PREFILL_SPAN_RESET_STATE, TENSOR_NONE16};
use packet::devbuild::{SECT_CUBIN, SECT_HSACO, SECT_NAME_LEN, SECT_PROGRAMS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, String>;

pub const SECTION: &str = "mixed_step";
pub const VERSION: u32 = 1;
pub const PROGRAM_CAPABILITY: &str = aux_program::CAPABILITY;
pub const OBJECT_CAPABILITY: &str = "plow_mixed_interpreter";
pub const DECODE_SLOT_TENSOR: &str = "in.decode_slot";

#[derive(Clone, Copy)]
pub struct TensorContract<'a> {
    pub name: &'a str,
    pub bytes: u64,
    pub initialized: bool,
}

fn require(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("mixed step: {reason}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RowPhase {
    Decode = 0,
    Prefill = 1,
    Parked = 2,
}

/// One dense activation row. Device adapters may transpose these fields into
/// their existing input tensors, but may not infer a physical slot from `row`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    pub token: u32,
    pub slot: u32,
    pub state_slot: u32,
    pub position: u32,
    pub kv_len: u32,
    pub phase: RowPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeRequest {
    pub slot: u32,
    pub state_slot: u32,
    pub token: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct PrefillRequest<'a> {
    pub slot: u32,
    pub state_slot: u32,
    pub start: u32,
    pub tokens: &'a [u32],
    pub prompt_len: u32,
}

/// How the plan's [`PrefillSpan`] table covers the packed rows.
///
/// The two are different contracts over one span array, and mixing them is the drift
/// `plans/unified-token-batch.md` §4.3 exists to stop: `runtime/common/mixed_step.h` TRAPS
/// unless `spans[0].row0 != 0`, and `runtime/amd/token_batch.h` traps unless the spans cover
/// `[0, M)` from zero. A consumer must be told which one it is looking at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpanCover {
    /// Mixed step v1. Rows `[0, decode_rows)` are a decode BAND that belongs to no span, and
    /// the spans cover `[decode_rows, real_rows)`.
    #[default]
    DecodeBand,
    /// Unified token batch (§4.3: "there is no decode prefix. Every row belongs to a span,
    /// decode spans included"). The spans cover exactly `[0, real_rows)`.
    ///
    /// Two things follow that the decode band cannot express:
    ///
    /// * A decode request is a span of length one. The converse is NOT true — a final prefill
    ///   chunk can also have length one — so length never decides a span's phase here either;
    ///   what the leading run of length-one spans decides is only which attention kernel
    ///   covers a span, which is the one thing §4.1 explicitly permits.
    /// * A prefill span that COMPLETES its prompt is split: its final token leads the batch as
    ///   a length-one span and the rest follows as an ordinary span, both in the same step and
    ///   both on the same physical slot. The whole prompt's KV is written by the step's single
    ///   RoPE/cache stage before any attention reads it, so the terminal row attends over its
    ///   own complete prompt. That is what lets one launch produce the prompt's first generated
    ///   token, and it is why this route does not need AMD's
    ///   `split_terminal_prefill`/`finish_prefill_batch` replay of that token through a second,
    ///   decode-shaped transformer pass (§1, §6.5).
    PrefixFree,
    /// AMD TP lowering for the MLA/MoE families (`plans/unified-token-batch.md`, "AMD TP8
    /// lowering decision"). Rows `[0, band)` are indexed by PHYSICAL KV SLOT — row `t` is slot
    /// `t`, exactly the batched-decode convention every GLM decode-chain op already implements
    /// (`HeadNormRope i6=n_batch_kv`, `FlashMlaDecode*`, `IndexScore`/`IndexSelect` per-row
    /// `kv_len[t]`), so the decode kernels need no per-row slot indirection. The spans cover
    /// `[band, real_rows)` contiguously, as under [`SpanCover::DecodeBand`].
    ///
    /// * A decode request occupies its slot's band row.
    /// * A prompt that COMPLETES this step puts its final token on its slot's band row and the
    ///   rest of its chunk in a body span, so the step samples the prompt's first generated
    ///   token without a replay pass — the same property as [`SpanCover::PrefixFree`], expressed
    ///   in the band instead of a leading length-one span.
    /// * Every other band row is PARKED: `kv_len` 1, no KV write, not sampled. A mid-prefill
    ///   slot's band row is parked while its span is live in the same step; the device's
    ///   parked check is what keeps that row from clobbering the span's frontier KV row.
    ///
    /// The sampled rows are the active band rows in slot order (`Plan::decode_slots` lists
    /// them); `Plan::decode_rows` is the band width, not the sample count. `S = 0` is legal.
    SlotBand {
        /// Band width: the decode ladder's widest rung, compiled into the body program.
        band: u32,
    },
}

/// One request's KV frontier transition, checked and applied as a unit after the device
/// succeeds.
///
/// It is per REQUEST, not per span: under [`SpanCover::PrefixFree`] a completing prompt owns
/// two spans on one slot, and committing per span would compare the terminal span's `kv_row0`
/// against a frontier that belongs to the body span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commit {
    pub slot: u32,
    /// The frontier this plan was built against. Re-checked before anything is mutated.
    pub expect: u32,
    /// The frontier after the step's rows are consumed as INPUT. A sampled token advances
    /// nothing until it is fed back.
    pub after: u32,
}

/// Backend-neutral host plan for one combined dispatch.
///
/// Decode rows occupy `[0, decode_rows)`. The canonical [`PrefillSpan`] values
/// cover `[decode_rows, real_rows)` densely. `parked` is the compiled-row binary
/// mask: zero for `[0, real_rows)`, one for the padded suffix. In this contract `PrefillSpan::program`
/// names the selected auxiliary program, rather than an ordinary prefill rung.
#[derive(Debug, PartialEq, Eq)]
pub struct Plan {
    pub decode_rows: u32,
    pub real_rows: u32,
    pub rows: Vec<Row>,
    /// Physical KV slot for each compact decode row. This is the host image of
    /// the optional `FlashDecode.t6` packet tensor.
    pub decode_slots: Vec<i32>,
    pub prefill_spans: Vec<PrefillSpan>,
    pub parked: Vec<u32>,
    /// Largest mapped KV end per active physical slot, including bounded
    /// parked padding for adapters whose fixed-width kernels still address it.
    pub mapped_ends: Vec<(u32, u32)>,
    /// Which contract [`Plan::prefill_spans`] follows.
    pub cover: SpanCover,
    /// One entry per admitted request, in row order.
    pub commits: Vec<Commit>,
}

impl Plan {
    /// Allocate persistent storage for repeated [`plan_into`] calls.
    pub fn with_capacity(
        row_capacity: usize,
        prefill_capacity: usize,
        active_capacity: usize,
    ) -> Self {
        Self {
            decode_rows: 0,
            real_rows: 0,
            rows: Vec::with_capacity(row_capacity),
            decode_slots: Vec::with_capacity(row_capacity),
            prefill_spans: Vec::with_capacity(prefill_capacity),
            parked: Vec::with_capacity(row_capacity),
            mapped_ends: Vec::with_capacity(active_capacity),
            cover: SpanCover::DecodeBand,
            commits: Vec::with_capacity(active_capacity),
        }
    }

    fn clear(&mut self) {
        self.decode_rows = 0;
        self.real_rows = 0;
        self.rows.clear();
        self.decode_slots.clear();
        self.prefill_spans.clear();
        self.parked.clear();
        self.mapped_ends.clear();
        self.cover = SpanCover::DecodeBand;
        self.commits.clear();
    }
}

/// Build a mixed plan in caller-owned storage. With sufficient capacities, the
/// successful path performs no heap allocation and never grows an output vector.
///
/// This is mixed step v1's shape — a decode band ahead of the spans — and is kept verbatim
/// because it is the only route that serves today. [`plan_into_cover`] is the same planner
/// with the span contract chosen explicitly.
pub fn plan_into(
    decode: &[DecodeRequest],
    prefill: &[PrefillRequest<'_>],
    frontiers: &[u32],
    rows: u32,
    max_ctx: u32,
    auxiliary_program: u32,
    out: &mut Plan,
) -> Result<()> {
    plan_into_cover(
        decode,
        prefill,
        frontiers,
        rows,
        max_ctx,
        auxiliary_program,
        SpanCover::DecodeBand,
        out,
    )
}

/// Build a plan under an explicit [`SpanCover`].
///
/// `SpanCover::PrefixFree` is the unified token batch's contract: spans cover exactly
/// `[0, real_rows)`, decode requests become spans of length one, and a prefill request that
/// completes its prompt contributes its final token as a leading length-one span so the step
/// can sample it without a second pass.
#[allow(clippy::too_many_arguments)]
pub fn plan_into_cover(
    decode: &[DecodeRequest],
    prefill: &[PrefillRequest<'_>],
    frontiers: &[u32],
    rows: u32,
    max_ctx: u32,
    auxiliary_program: u32,
    cover: SpanCover,
    out: &mut Plan,
) -> Result<()> {
    out.clear();
    out.cover = cover;
    let result = plan_into_inner(
        decode,
        prefill,
        frontiers,
        rows,
        max_ctx,
        auxiliary_program,
        cover,
        out,
    );
    if result.is_err() {
        out.clear();
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn plan_into_inner(
    decode: &[DecodeRequest],
    prefill: &[PrefillRequest<'_>],
    frontiers: &[u32],
    rows: u32,
    max_ctx: u32,
    auxiliary_program: u32,
    cover: SpanCover,
    out: &mut Plan,
) -> Result<()> {
    require(
        (!decode.is_empty() || !prefill.is_empty()) && rows > 0 && max_ctx > 0,
        "empty work or capacity",
    )?;
    require(
        decode.len().saturating_add(prefill.len()) <= frontiers.len(),
        "active request capacity",
    )?;
    if let SpanCover::SlotBand { band } = cover {
        return plan_slot_band_inner(
            decode,
            prefill,
            frontiers,
            rows,
            max_ctx,
            auxiliary_program,
            band,
            out,
        );
    }

    let capacity = usize::try_from(rows).map_err(|_| "mixed step: row capacity")?;
    let active = decode
        .len()
        .checked_add(prefill.len())
        .ok_or("mixed step: active request overflow")?;
    // PrefixFree turns every decode request into a span and can split a completing prompt
    // into two, so the span table is bounded by `decode + 2 * prefill`, not by `prefill`.
    let prefix_free = cover == SpanCover::PrefixFree;
    let span_capacity = if prefix_free {
        decode
            .len()
            .checked_add(prefill.len().checked_mul(2).ok_or("mixed step: span count")?)
            .ok_or("mixed step: span count")?
    } else {
        prefill.len()
    };
    let leading_capacity = if prefix_free { active } else { decode.len() };
    require(
        out.rows.capacity() >= capacity
            && out.prefill_spans.capacity() >= span_capacity
            && out.decode_slots.capacity() >= leading_capacity
            && out.parked.capacity() >= capacity
            && out.commits.capacity() >= active
            && out.mapped_ends.capacity() >= active,
        "output buffer capacity",
    )?;

    for (index, request) in decode.iter().enumerate() {
        let slot = request.slot as usize;
        let state_slot = request.state_slot as usize;
        require(
            slot < frontiers.len()
                && state_slot < frontiers.len()
                && !decode[..index]
                    .iter()
                    .any(|prior| prior.slot == request.slot)
                && !decode[..index]
                    .iter()
                    .any(|prior| prior.state_slot == request.state_slot),
            "physical/state slot or duplicate",
        )?;
        let position = frontiers[slot];
        let kv_len = position
            .checked_add(1)
            .ok_or("mixed step: decode position overflow")?;
        require(
            kv_len <= max_ctx && out.rows.len() < capacity,
            "decode extent",
        )?;
        out.rows.push(Row {
            token: request.token,
            slot: request.slot,
            state_slot: request.state_slot,
            position,
            kv_len,
            phase: RowPhase::Decode,
        });
        out.decode_slots
            .push(i32::try_from(request.slot).map_err(|_| "mixed step: physical slot overflow")?);
        if prefix_free {
            out.prefill_spans.push(PrefillSpan {
                row0: u32::try_from(index).map_err(|_| "mixed step: decode row offset")?,
                n_rows: 1,
                slot: request.slot,
                flags: u32::from(position == 0) * PREFILL_SPAN_RESET_STATE,
                kv_row0: position,
                kv_len,
                state_slot: request.state_slot,
                program: auxiliary_program,
            });
        }
        out.commits.push(Commit {
            slot: request.slot,
            expect: position,
            after: kv_len,
        });
        out.mapped_ends.push((request.slot, kv_len));
    }

    // The terminal rows of prompts that COMPLETE in this step. They lead the batch beside the
    // decode rows because the selection stage samples the leading rows, and because a one-row
    // span at frontier p attends over [0, p+1) with its query at p under either attention
    // kernel — so a completing prompt's first generated token is produced by the same launch
    // that consumed the prompt, with no replay through a decode-shaped pass.
    if prefix_free {
        for request in prefill {
            let n_rows =
                u32::try_from(request.tokens.len()).map_err(|_| "mixed step: prefill row count")?;
            let end = request
                .start
                .checked_add(n_rows)
                .ok_or("mixed step: prefill extent overflow")?;
            if n_rows == 0 || end != request.prompt_len {
                continue;
            }
            let position = end - 1;
            require(
                end <= max_ctx && out.rows.len() < capacity,
                "terminal prefill extent",
            )?;
            out.prefill_spans.push(PrefillSpan {
                row0: u32::try_from(out.rows.len())
                    .map_err(|_| "mixed step: terminal row offset")?,
                n_rows: 1,
                slot: request.slot,
                flags: u32::from(position == 0) * PREFILL_SPAN_RESET_STATE,
                kv_row0: position,
                kv_len: end,
                state_slot: request.state_slot,
                program: auxiliary_program,
            });
            out.rows.push(Row {
                token: request.tokens[request.tokens.len() - 1],
                slot: request.slot,
                state_slot: request.state_slot,
                position,
                kv_len: end,
                phase: RowPhase::Decode,
            });
            out.decode_slots.push(
                i32::try_from(request.slot).map_err(|_| "mixed step: physical slot overflow")?,
            );
        }
    }
    let decode_rows = u32::try_from(out.rows.len()).map_err(|_| "mixed step: decode rows")?;

    for (index, request) in prefill.iter().enumerate() {
        let slot = request.slot as usize;
        let state_slot = request.state_slot as usize;
        let duplicate_physical = decode.iter().any(|prior| prior.slot == request.slot)
            || prefill[..index]
                .iter()
                .any(|prior| prior.slot == request.slot);
        let duplicate_state = decode
            .iter()
            .any(|prior| prior.state_slot == request.state_slot)
            || prefill[..index]
                .iter()
                .any(|prior| prior.state_slot == request.state_slot);
        require(
            slot < frontiers.len()
                && state_slot < frontiers.len()
                && !duplicate_physical
                && !duplicate_state,
            "physical/state slot or duplicate",
        )?;
        let n_rows =
            u32::try_from(request.tokens.len()).map_err(|_| "mixed step: prefill row count")?;
        let end = request
            .start
            .checked_add(n_rows)
            .ok_or("mixed step: prefill extent overflow")?;
        require(
            n_rows > 0
                && request.start == frontiers[slot]
                && end <= request.prompt_len
                && end <= max_ctx
                && out.rows.len().saturating_add(request.tokens.len()) <= capacity,
            "prefill frontier or extent",
        )?;
        // Under PrefixFree a completing prompt's last token has already been placed as a
        // leading length-one span, so the body span is the rest — possibly empty, when the
        // whole remaining prompt was one token.
        let completes = prefix_free && end == request.prompt_len;
        let body = if completes {
            &request.tokens[..request.tokens.len() - 1]
        } else {
            request.tokens
        };
        if !body.is_empty() {
            let row0 =
                u32::try_from(out.rows.len()).map_err(|_| "mixed step: prefill row offset")?;
            let body_rows = u32::try_from(body.len()).map_err(|_| "mixed step: body row count")?;
            out.prefill_spans.push(PrefillSpan {
                row0,
                n_rows: body_rows,
                slot: request.slot,
                flags: u32::from(request.start == 0) * PREFILL_SPAN_RESET_STATE,
                kv_row0: request.start,
                kv_len: request.start + body_rows,
                state_slot: request.state_slot,
                program: auxiliary_program,
            });
            for (offset, &token) in body.iter().enumerate() {
                let position = request.start + offset as u32;
                out.rows.push(Row {
                    token,
                    slot: request.slot,
                    state_slot: request.state_slot,
                    position,
                    kv_len: position + 1,
                    phase: RowPhase::Prefill,
                });
            }
        }
        out.commits.push(Commit {
            slot: request.slot,
            expect: request.start,
            after: end,
        });
        out.mapped_ends.push((request.slot, end));
    }

    let real_rows = u32::try_from(out.rows.len()).map_err(|_| "mixed step: real rows")?;
    let owner = out
        .rows
        .last()
        .copied()
        .ok_or("mixed step: padding owner")?;
    let pad = capacity - out.rows.len();
    let padded_end = owner
        .position
        .checked_add(1)
        .and_then(|end| end.checked_add(pad as u32))
        .ok_or("mixed step: padding extent overflow")?;
    require(padded_end <= max_ctx, "padding exceeds physical context")?;
    out.parked.resize(capacity, 0);
    out.parked[real_rows as usize..].fill(1);
    for offset in 0..pad {
        let position = owner.position + 1 + offset as u32;
        out.rows.push(Row {
            token: 0,
            slot: owner.slot,
            state_slot: owner.state_slot,
            position,
            kv_len: position + 1,
            phase: RowPhase::Parked,
        });
    }
    if pad > 0 {
        let end = out
            .mapped_ends
            .iter_mut()
            .find(|(slot, _)| *slot == owner.slot)
            .ok_or("mixed step: padding owner mapping")?;
        end.1 = padded_end;
    }
    out.decode_rows = decode_rows;
    out.real_rows = real_rows;
    Ok(())
}

/// The [`SpanCover::SlotBand`] planner: a slot-indexed band of `band` rows, then the spans.
///
/// Refusals are by name and leave `out` cleared (the caller does that). The band is filled
/// with parked rows first and the requests claim their slots, so a slot with no request this
/// step is parked by construction rather than by a second pass.
#[allow(clippy::too_many_arguments)]
fn plan_slot_band_inner(
    decode: &[DecodeRequest],
    prefill: &[PrefillRequest<'_>],
    frontiers: &[u32],
    rows: u32,
    max_ctx: u32,
    auxiliary_program: u32,
    band: u32,
    out: &mut Plan,
) -> Result<()> {
    require(band > 0 && band <= rows, "slot band width outside the row capacity")?;
    let capacity = usize::try_from(rows).map_err(|_| "mixed step: row capacity")?;
    let band_rows = band as usize;
    let active = decode
        .len()
        .checked_add(prefill.len())
        .ok_or("mixed step: active request overflow")?;
    require(
        out.rows.capacity() >= capacity
            && out.prefill_spans.capacity() >= prefill.len()
            && out.decode_slots.capacity() >= band_rows.min(active)
            && out.parked.capacity() >= capacity
            && out.commits.capacity() >= active
            && out.mapped_ends.capacity() >= active,
        "output buffer capacity",
    )?;

    // Every band row starts parked on its own slot; the requests below claim theirs.
    out.parked.resize(capacity, 1);
    for slot in 0..band {
        out.rows.push(Row {
            token: 0,
            slot,
            state_slot: slot,
            position: 0,
            kv_len: 1,
            phase: RowPhase::Parked,
        });
    }

    for (index, request) in decode.iter().enumerate() {
        let slot = request.slot as usize;
        require(
            slot < frontiers.len()
                && request.slot < band
                && request.state_slot == request.slot
                && !decode[..index]
                    .iter()
                    .any(|prior| prior.slot == request.slot),
            "physical/state slot or duplicate",
        )?;
        let position = frontiers[slot];
        let kv_len = position
            .checked_add(1)
            .ok_or("mixed step: decode position overflow")?;
        require(kv_len <= max_ctx, "decode extent")?;
        out.rows[slot] = Row {
            token: request.token,
            slot: request.slot,
            state_slot: request.state_slot,
            position,
            kv_len,
            phase: RowPhase::Decode,
        };
        out.parked[slot] = 0;
        out.commits.push(Commit {
            slot: request.slot,
            expect: position,
            after: kv_len,
        });
        out.mapped_ends.push((request.slot, kv_len));
    }

    for (index, request) in prefill.iter().enumerate() {
        let slot = request.slot as usize;
        let duplicate = decode.iter().any(|prior| prior.slot == request.slot)
            || prefill[..index]
                .iter()
                .any(|prior| prior.slot == request.slot);
        require(
            slot < frontiers.len() && request.state_slot == request.slot && !duplicate,
            "physical/state slot or duplicate",
        )?;
        let n_rows =
            u32::try_from(request.tokens.len()).map_err(|_| "mixed step: prefill row count")?;
        let end = request
            .start
            .checked_add(n_rows)
            .ok_or("mixed step: prefill extent overflow")?;
        require(
            n_rows > 0
                && request.start == frontiers[slot]
                && end <= request.prompt_len
                && end <= max_ctx,
            "prefill frontier or extent",
        )?;
        let completes = end == request.prompt_len;
        // The completing prompt's final token takes its slot's band row, which needs the slot
        // to HAVE a band row. A slot past the band can still prefill here; it cannot complete.
        require(
            !completes || request.slot < band,
            "completing prompt's slot is outside the band",
        )?;
        let body = if completes {
            &request.tokens[..request.tokens.len() - 1]
        } else {
            request.tokens
        };
        if completes {
            out.rows[slot] = Row {
                token: request.tokens[request.tokens.len() - 1],
                slot: request.slot,
                state_slot: request.state_slot,
                position: end - 1,
                kv_len: end,
                phase: RowPhase::Decode,
            };
            out.parked[slot] = 0;
        }
        if !body.is_empty() {
            require(
                out.rows.len().saturating_add(body.len()) <= capacity,
                "prefill extent",
            )?;
            let row0 =
                u32::try_from(out.rows.len()).map_err(|_| "mixed step: prefill row offset")?;
            let body_rows = u32::try_from(body.len()).map_err(|_| "mixed step: body row count")?;
            out.prefill_spans.push(PrefillSpan {
                row0,
                n_rows: body_rows,
                slot: request.slot,
                flags: u32::from(request.start == 0) * PREFILL_SPAN_RESET_STATE,
                kv_row0: request.start,
                kv_len: request.start + body_rows,
                state_slot: request.state_slot,
                program: auxiliary_program,
            });
            for (offset, &token) in body.iter().enumerate() {
                let position = request.start + offset as u32;
                out.rows.push(Row {
                    token,
                    slot: request.slot,
                    state_slot: request.state_slot,
                    position,
                    kv_len: position + 1,
                    phase: RowPhase::Prefill,
                });
            }
        }
        out.commits.push(Commit {
            slot: request.slot,
            expect: request.start,
            after: end,
        });
        out.mapped_ends.push((request.slot, end));
    }

    // The sampled rows, in row (= slot) order.
    for slot in 0..band_rows {
        if out.parked[slot] == 0 {
            out.decode_slots.push(slot as i32);
        }
    }
    let real_rows = u32::try_from(out.rows.len()).map_err(|_| "mixed step: real rows")?;
    out.parked[band_rows..real_rows as usize].fill(0);

    // Padding: parked, and addressed after the last span row so a fixed-width kernel that still
    // touches it stays inside that slot's mapped KV. A band-only step has no span to extend
    // and parks the suffix on slot 0 at position 0, which nothing maps or reads.
    let pad = capacity - out.rows.len();
    if pad > 0 {
        let owner = out
            .prefill_spans
            .last()
            .map(|span| (span.slot, span.state_slot, span.kv_len - 1));
        if let Some((slot, state_slot, position)) = owner {
            let padded_end = position
                .checked_add(1)
                .and_then(|end| end.checked_add(pad as u32))
                .ok_or("mixed step: padding extent overflow")?;
            require(padded_end <= max_ctx, "padding exceeds physical context")?;
            for offset in 0..pad {
                let position = position + 1 + offset as u32;
                out.rows.push(Row {
                    token: 0,
                    slot,
                    state_slot,
                    position,
                    kv_len: position + 1,
                    phase: RowPhase::Parked,
                });
            }
            let end = out
                .mapped_ends
                .iter_mut()
                .find(|(s, _)| *s == slot)
                .ok_or("mixed step: padding owner mapping")?;
            end.1 = padded_end;
        } else {
            for _ in 0..pad {
                out.rows.push(Row {
                    token: 0,
                    slot: 0,
                    state_slot: 0,
                    position: 0,
                    kv_len: 1,
                    phase: RowPhase::Parked,
                });
            }
        }
    }
    out.decode_rows = band;
    out.real_rows = real_rows;
    Ok(())
}

/// Check a [`SpanCover::SlotBand`] plan against the contract a token-batch body program
/// executes: row `t` of the band is slot `t`; a completing prompt's terminal row sits exactly
/// where its body span ends; spans are dense from the band; padding is parked. Backend-neutral
/// so the AMD and CUDA adapters refuse the same plans by the same names before any upload.
pub fn validate_slot_band(plan: &Plan, slot_capacity: usize, max_ctx: u32) -> Result<()> {
    let SpanCover::SlotBand { band } = plan.cover else {
        return Err("mixed step: plan is not a slot-band plan".into());
    };
    let band_rows = band as usize;
    let capacity = plan.rows.len();
    require(
        band_rows > 0 && band_rows <= slot_capacity && band_rows <= capacity,
        "slot band wider than the slot or row capacity",
    )?;
    require(
        plan.parked.len() == capacity && plan.parked.iter().all(|&v| v <= 1),
        "parked mask is not a binary mask over every row",
    )?;
    let real = plan.real_rows as usize;
    require(
        real >= band_rows && real <= capacity && plan.decode_rows == band,
        "real rows or decode_rows disagree with the band",
    )?;
    let mut sampled = Vec::with_capacity(band_rows);
    for t in 0..band_rows {
        let row = &plan.rows[t];
        require(
            row.slot as usize == t && row.state_slot as usize == t,
            "band row is not on its own slot",
        )?;
        match (plan.parked[t], row.phase) {
            (1, RowPhase::Parked) => {}
            (0, RowPhase::Decode) => {
                require(
                    row.kv_len == row.position.wrapping_add(1) && row.kv_len <= max_ctx,
                    "active band row extent",
                )?;
                sampled.push(t as i32);
            }
            _ => return Err("mixed step: band row phase disagrees with its parked bit".into()),
        }
    }
    require(
        plan.decode_slots == sampled,
        "sampled slots disagree with the active band rows",
    )?;
    let mut row = band;
    for (index, span) in plan.prefill_spans.iter().enumerate() {
        require(span.row0 == row && span.n_rows > 0, "span is not dense after the band")?;
        require(
            span.flags & !PREFILL_SPAN_RESET_STATE == 0
                && (span.flags & PREFILL_SPAN_RESET_STATE != 0) == (span.kv_row0 == 0),
            "span flags",
        )?;
        let kv_end = span
            .kv_row0
            .checked_add(span.n_rows)
            .ok_or("mixed step: span KV range overflow")?;
        require(
            kv_end == span.kv_len && kv_end <= max_ctx,
            "span kv_row0 + n_rows != kv_len or past context",
        )?;
        require(
            (span.slot as usize) < slot_capacity && span.slot == span.state_slot,
            "span slot outside capacity or split from its state slot",
        )?;
        require(
            !plan.prefill_spans[..index]
                .iter()
                .any(|prior| prior.slot == span.slot),
            "slot appears in more than one span",
        )?;
        if (span.slot as usize) < band_rows && plan.parked[span.slot as usize] == 0 {
            // The terminal/body pair: the band row is the token right after the body.
            require(
                plan.rows[span.slot as usize].position == span.kv_len,
                "band row and body span on one slot are not a terminal/body pair",
            )?;
        }
        for local in 0..span.n_rows {
            let r = &plan.rows[(span.row0 + local) as usize];
            require(
                r.slot == span.slot
                    && r.position == span.kv_row0 + local
                    && r.phase == RowPhase::Prefill
                    && plan.parked[(span.row0 + local) as usize] == 0,
                "span row disagrees with its span",
            )?;
        }
        row = row
            .checked_add(span.n_rows)
            .ok_or("mixed step: span rows overflow")?;
    }
    require(row as usize == real, "spans do not end at real_rows")?;
    require(
        plan.parked[real..].iter().all(|&v| v == 1)
            && plan.rows[real..].iter().all(|r| r.phase == RowPhase::Parked),
        "padding is not parked",
    )?;
    Ok(())
}

/// Plan a unified token batch from the backend-neutral request contract.
///
/// [`crate::token_batch::Request`] is what the CUDA route plans from. Taking it here too means
/// both backends admit ONE request shape, apply the same slot-generation and selection checks,
/// and deliver by logical request id — while the row layout stays the one this planner has
/// always produced ([`SpanCover::PrefixFree`]). The AMD object samples the leading band and the
/// CUDA terminal gathers arbitrary rows through a table; that is a device contract each object
/// carries, not a host choice, so the two layouts are not unified here.
///
/// `owners` receives the logical request id of every leading (sampled) row in plan order — the
/// decode requests, then the prompts that complete in this step, each in request order. It is
/// derived independently of the planner and then cross-checked against the plan's
/// `decode_slots`, so the two statements of that order cannot drift apart silently.
///
/// The two partition vectors are the per-step allocations the AMD serving layer made itself
/// before this existed; they are not new cost, and `plan_into_cover` remains allocation-free.
#[allow(clippy::too_many_arguments)]
pub fn plan_requests_into(
    requests: &[crate::token_batch::Request<'_>],
    frontiers: &[u32],
    generations: &[u32],
    rows: u32,
    max_ctx: u32,
    auxiliary_program: u32,
    out: &mut Plan,
    owners: &mut Vec<u32>,
) -> Result<()> {
    plan_requests_into_cover(
        requests,
        frontiers,
        generations,
        rows,
        max_ctx,
        auxiliary_program,
        SpanCover::PrefixFree,
        out,
        owners,
    )
}

/// [`plan_requests_into`] under an explicit physical cover.
///
/// The logical contract is the same for every cover; what changes is the row layout and hence
/// the ORDER of `owners`: [`SpanCover::PrefixFree`] samples decode requests then completing
/// prompts in request order, [`SpanCover::SlotBand`] samples the active band rows in slot
/// order. Either way `owners[i]` owns the plan's `decode_slots[i]`, and that is cross-checked.
/// [`SpanCover::DecodeBand`] is refused: it cannot sample a completing prompt.
#[allow(clippy::too_many_arguments)]
pub fn plan_requests_into_cover(
    requests: &[crate::token_batch::Request<'_>],
    frontiers: &[u32],
    generations: &[u32],
    rows: u32,
    max_ctx: u32,
    auxiliary_program: u32,
    cover: SpanCover,
    out: &mut Plan,
    owners: &mut Vec<u32>,
) -> Result<()> {
    use crate::token_batch::Phase;
    require(
        cover != SpanCover::DecodeBand,
        "the decode band cannot sample a completing prompt",
    )?;
    // A refusal before the planner runs must not leave the previous plan looking stageable.
    out.clear();
    owners.clear();
    require(
        frontiers.len() == generations.len(),
        "frontier and generation tables disagree on slot capacity",
    )?;
    let mut decode = Vec::with_capacity(requests.len());
    let mut prefill = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        request.selection.validate()?;
        let slot = request.slot as usize;
        require(slot < frontiers.len(), "physical slot outside capacity")?;
        require(
            generations[slot] == request.generation,
            "slot generation does not match the admitted request",
        )?;
        require(
            !requests[..index].iter().any(|prior| prior.id == request.id),
            "duplicate request id",
        )?;
        let start = frontiers[slot];
        match request.phase {
            Phase::Decode => {
                require(
                    request.tokens.len() == 1
                        && start >= request.prompt_len
                        && request.prompt_len > 0,
                    "decode span must be one token at or past the prompt end",
                )?;
                decode.push(DecodeRequest {
                    slot: request.slot,
                    state_slot: request.state_slot,
                    token: request.tokens[0],
                });
            }
            Phase::Prefill => {
                let n_rows = u32::try_from(request.tokens.len())
                    .map_err(|_| "mixed step: prefill row count")?;
                let end = start
                    .checked_add(n_rows)
                    .ok_or("mixed step: prefill extent overflow")?;
                require(
                    end <= request.prompt_len,
                    "prefill span overruns the prompt",
                )?;
                prefill.push(PrefillRequest {
                    slot: request.slot,
                    state_slot: request.state_slot,
                    start,
                    tokens: request.tokens,
                    prompt_len: request.prompt_len,
                });
            }
        }
    }
    plan_into_cover(
        &decode,
        &prefill,
        frontiers,
        rows,
        max_ctx,
        auxiliary_program,
        cover,
        out,
    )?;
    let sampled = |r: &&crate::token_batch::Request<'_>| {
        r.phase == Phase::Decode
            || frontiers[r.slot as usize] + r.tokens.len() as u32 == r.prompt_len
    };
    match cover {
        SpanCover::SlotBand { .. } => {
            let mut by_slot: Vec<(u32, u32)> = requests
                .iter()
                .filter(sampled)
                .map(|r| (r.slot, r.id))
                .collect();
            by_slot.sort_unstable();
            owners.extend(by_slot.into_iter().map(|(_, id)| id));
        }
        _ => {
            owners.extend(
                requests
                    .iter()
                    .filter(|r| r.phase == Phase::Decode)
                    .map(|r| r.id),
            );
            owners.extend(
                requests
                    .iter()
                    .filter(|r| r.phase == Phase::Prefill && sampled(r))
                    .map(|r| r.id),
            );
        }
    }
    let consistent = owners.len() == out.decode_slots.len()
        && owners.iter().zip(&out.decode_slots).all(|(id, &slot)| {
            requests
                .iter()
                .any(|r| r.id == *id && i32::try_from(r.slot) == Ok(slot))
        });
    if !consistent {
        out.clear();
        owners.clear();
        return Err("mixed step: leading-row owners disagree with the plan".into());
    }
    Ok(())
}

/// Allocate an owned reference plan and delegate to [`plan_into`].
pub fn plan(
    decode: &[DecodeRequest],
    prefill: &[PrefillRequest<'_>],
    frontiers: &[u32],
    rows: u32,
    max_ctx: u32,
    auxiliary_program: u32,
) -> Result<Plan> {
    let row_capacity = usize::try_from(rows).map_err(|_| "mixed step: row capacity")?;
    let active_capacity = decode
        .len()
        .checked_add(prefill.len())
        .ok_or("mixed step: active request overflow")?;
    let mut out = Plan::with_capacity(row_capacity, prefill.len(), active_capacity);
    plan_into(
        decode,
        prefill,
        frontiers,
        rows,
        max_ctx,
        auxiliary_program,
        &mut out,
    )?;
    Ok(out)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capability {
    pub name: String,
    pub version: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadKind {
    Programs,
    Cubin,
    Hsaco,
}

impl PayloadKind {
    pub fn section_kind(self) -> u32 {
        match self {
            Self::Programs => SECT_PROGRAMS,
            Self::Cubin => SECT_CUBIN,
            Self::Hsaco => SECT_HSACO,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadBinding {
    pub section: String,
    pub kind: PayloadKind,
    pub version: u32,
    pub sha256: String,
    pub capability: Capability,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramBinding {
    pub index: u32,
    pub payload: PayloadBinding,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub rows: u32,
    pub decode_rows: u32,
    pub program: ProgramBinding,
    #[serde(default)]
    pub objects: Vec<PayloadBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub n_cu: u32,
    /// Maximum number of decode rows plus prefill spans in one step.
    pub max_active_requests: u32,
    /// Size of the persistent KV/state slot namespace. Active requests may use
    /// sparse indices anywhere in `[0, physical_slot_capacity)`.
    pub physical_slot_capacity: u32,
    pub variants: Vec<Variant>,
}

#[derive(Clone, Copy)]
pub struct ValidatedManifest<'a> {
    manifest: &'a Manifest,
}

#[derive(Clone, Copy)]
pub struct Payload<'a> {
    pub section: &'a str,
    pub kind: PayloadKind,
    pub version: u32,
    pub n_cu: u32,
    pub bytes: &'a [u8],
}

fn identifier(s: &str, max: usize) -> bool {
    !s.is_empty()
        && s.len() <= max
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}

fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

pub fn payload_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl PayloadBinding {
    fn validate(&self) -> Result<()> {
        require(
            identifier(&self.section, SECT_NAME_LEN - 1)
                && self.section != SECTION
                && self.version > 0
                && digest(&self.sha256)
                && identifier(&self.capability.name, 63)
                && self.capability.version > 0,
            "payload identity",
        )
    }

    fn bind_identity(&self, expected_n_cu: u32, payload: &Payload<'_>) -> Result<()> {
        self.validate()?;
        require(
            payload.section == self.section
                && payload.kind == self.kind
                && payload.version == self.version
                && payload.n_cu == expected_n_cu
                && payload_sha256(payload.bytes) == self.sha256,
            "payload binding",
        )
    }

    fn bind_object_with(
        &self,
        expected_kind: PayloadKind,
        expected_n_cu: u32,
        payload: &Payload<'_>,
        mut read_capability: impl FnMut(&str) -> Option<u32>,
    ) -> Result<()> {
        require(self.kind == expected_kind, "object backend kind")?;
        self.bind_identity(expected_n_cu, payload)?;
        require(
            read_capability(&self.capability.name) == Some(self.capability.version),
            "object capability",
        )
    }
}

impl Manifest {
    /// Validate packet metadata once at load time. Per-step callers retain the
    /// returned wrapper, whose plan checks do not allocate.
    pub fn validate(&self) -> Result<ValidatedManifest<'_>> {
        require(
            self.version == VERSION
                && self.n_cu > 0
                && self.max_active_requests > 0
                && self.max_active_requests <= self.physical_slot_capacity
                && !self.variants.is_empty(),
            "version or capacity",
        )?;
        let mut shapes = std::collections::BTreeSet::new();
        let mut payloads = std::collections::BTreeMap::new();
        for variant in &self.variants {
            require(
                variant.rows > 1
                    && variant.decode_rows > 0
                    && variant.decode_rows < variant.rows
                    && variant.decode_rows <= self.max_active_requests
                    && shapes.insert((variant.rows, variant.decode_rows))
                    && variant.program.payload.kind == PayloadKind::Programs
                    && variant.program.payload.version == VERSION
                    && variant.program.payload.capability.name == PROGRAM_CAPABILITY
                    && variant.program.payload.capability.version == VERSION,
                "variant geometry",
            )?;
            variant.program.payload.validate()?;
            record_payload(&mut payloads, &variant.program.payload)?;
            require(!variant.objects.is_empty(), "variant has no backend object")?;
            let mut objects = std::collections::BTreeSet::new();
            let mut backends = std::collections::BTreeSet::new();
            for object in &variant.objects {
                require(
                    matches!(object.kind, PayloadKind::Cubin | PayloadKind::Hsaco)
                        && object.version == VERSION
                        && object.capability.name == OBJECT_CAPABILITY
                        && object.capability.version == VERSION
                        && backends.insert(object.kind.section_kind())
                        && objects.insert((object.kind.section_kind(), object.section.as_str())),
                    "object kind or duplicate",
                )?;
                object.validate()?;
                record_payload(&mut payloads, object)?;
            }
        }
        Ok(ValidatedManifest { manifest: self })
    }
}

impl ValidatedManifest<'_> {
    pub fn variant(&self, rows: u32, decode_rows: u32) -> Option<&Variant> {
        self.manifest
            .variants
            .iter()
            .find(|v| v.rows == rows && v.decode_rows == decode_rows)
    }

    pub fn validate_plan(&self, plan: &Plan) -> Result<&Variant> {
        let rows = u32::try_from(plan.rows.len()).map_err(|_| "mixed step: plan rows")?;
        let variant = self
            .variant(rows, plan.decode_rows)
            .ok_or("mixed step: missing plan variant")?;
        variant.validate_plan(
            plan,
            self.manifest.max_active_requests,
            self.manifest.physical_slot_capacity,
        )?;
        Ok(variant)
    }
}

impl Variant {
    fn validate_plan(
        &self,
        plan: &Plan,
        max_active_requests: u32,
        physical_slot_capacity: u32,
    ) -> Result<()> {
        require(
            plan.rows.len() == self.rows as usize
                && plan.decode_rows == self.decode_rows
                && plan.decode_rows <= plan.real_rows
                && plan.real_rows <= self.rows
                && !plan.prefill_spans.is_empty()
                && plan
                    .prefill_spans
                    .len()
                    .saturating_add(plan.decode_rows as usize)
                    <= max_active_requests as usize
                && plan
                    .prefill_spans
                    .iter()
                    .all(|span| span.program == self.program.index),
            "plan does not match variant",
        )?;

        let decode_rows = plan.decode_rows as usize;
        let real_rows = plan.real_rows as usize;
        let owner = plan.rows[real_rows - 1];
        let parked_end = (real_rows < plan.rows.len()).then(|| plan.rows.last().unwrap().kv_len);
        require(
            plan.mapped_ends.len() == decode_rows + plan.prefill_spans.len(),
            "mapped slot count",
        )?;
        validate_decode_slots(&plan.decode_slots, plan.decode_rows, physical_slot_capacity)?;
        require(
            plan.rows[..decode_rows].iter().all(|row| {
                row.phase == RowPhase::Decode
                    && row.kv_len == row.position.checked_add(1).unwrap_or(0)
            }),
            "decode row coordinates",
        )?;

        let mut next = decode_rows;
        for (index, row) in plan.rows[..decode_rows].iter().enumerate() {
            let prior = &plan.rows[..index];
            let mapped_end = parked_end
                .filter(|_| row.slot == owner.slot)
                .unwrap_or(row.kv_len);
            require(
                row.slot < physical_slot_capacity
                    && plan.decode_slots[index] as u32 == row.slot
                    && row.state_slot < physical_slot_capacity
                    && !prior.iter().any(|p| p.slot == row.slot)
                    && !prior.iter().any(|p| p.state_slot == row.state_slot)
                    && plan.mapped_ends[index] == (row.slot, mapped_end),
                "duplicate or out-of-range active slot",
            )?;
        }
        for (span_index, span) in plan.prefill_spans.iter().enumerate() {
            let end = span
                .row0
                .checked_add(span.n_rows)
                .ok_or("mixed step: prefill row overflow")?;
            let kv_end = span
                .kv_row0
                .checked_add(span.n_rows)
                .ok_or("mixed step: prefill KV overflow")?;
            let prior_span = &plan.prefill_spans[..span_index];
            let prior_physical = plan.rows[..decode_rows]
                .iter()
                .any(|row| row.slot == span.slot)
                || prior_span.iter().any(|prior| prior.slot == span.slot);
            let prior_state = plan.rows[..decode_rows]
                .iter()
                .any(|row| row.state_slot == span.state_slot)
                || prior_span
                    .iter()
                    .any(|prior| prior.state_slot == span.state_slot);
            let mapped_end = parked_end
                .filter(|_| span.slot == owner.slot)
                .unwrap_or(span.kv_len);
            require(
                span.row0 as usize == next
                    && span.n_rows > 0
                    && end as usize <= real_rows
                    && span.kv_len == kv_end
                    && span.flags == u32::from(span.kv_row0 == 0) * PREFILL_SPAN_RESET_STATE
                    && span.slot < physical_slot_capacity
                    && span.state_slot < physical_slot_capacity
                    && !prior_physical
                    && !prior_state
                    && plan.mapped_ends[decode_rows + span_index] == (span.slot, mapped_end),
                "prefill span coverage",
            )?;
            for (offset, row) in plan.rows[span.row0 as usize..end as usize]
                .iter()
                .enumerate()
            {
                let position = span.kv_row0 + offset as u32;
                require(
                    row.phase == RowPhase::Prefill
                        && row.slot == span.slot
                        && row.state_slot == span.state_slot
                        && row.position == position
                        && row.kv_len == position + 1,
                    "prefill row coordinates",
                )?;
            }
            next = end as usize;
        }
        require(next == real_rows, "real row coverage")?;
        for (offset, row) in plan.rows[real_rows..].iter().enumerate() {
            let position = owner
                .position
                .checked_add(1)
                .and_then(|p| p.checked_add(offset as u32))
                .ok_or("mixed step: parked row overflow")?;
            let kv_len = position
                .checked_add(1)
                .ok_or("mixed step: parked KV overflow")?;
            require(
                row.phase == RowPhase::Parked
                    && row.token == 0
                    && row.slot == owner.slot
                    && row.state_slot == owner.state_slot
                    && row.position == position
                    && row.kv_len == kv_len,
                "parked row coordinates",
            )?;
        }
        require(
            plan.parked.len() == plan.rows.len()
                && plan.parked[..real_rows].iter().all(|&value| value == 0)
                && plan.parked[real_rows..].iter().all(|&value| value == 1),
            "parked suffix or mapping",
        )
    }

    pub fn bind_program(
        &self,
        expected_n_cu: u32,
        tensor_count: usize,
        program: &Payload<'_>,
    ) -> Result<aux_program::Section> {
        self.program.payload.bind_identity(expected_n_cu, program)?;
        require(
            self.program.payload.kind == PayloadKind::Programs
                && self.program.payload.capability.name == aux_program::CAPABILITY
                && self.program.payload.capability.version == aux_program::VERSION,
            "program capability",
        )?;
        let parsed = aux_program::parse(program.bytes, expected_n_cu, tensor_count)?;
        require(
            parsed
                .programs
                .get(self.program.index as usize)
                .is_some_and(|p| p.rows == self.rows),
            "auxiliary program index",
        )?;
        Ok(parsed)
    }

    /// Bind a CUDA object after its adapter has read an actual module or ELF
    /// capability symbol. The callback maps the backend-neutral capability
    /// name to the object's initialized u32 value.
    pub fn bind_cubin_with(
        &self,
        expected_n_cu: u32,
        payload: &Payload<'_>,
        read_capability: impl FnMut(&str) -> Option<u32>,
    ) -> Result<()> {
        self.bind_object_with(PayloadKind::Cubin, expected_n_cu, payload, read_capability)
    }

    /// HSACO twin of [`Self::bind_cubin_with`]. HSA module symbol lookup stays
    /// in the runtime adapter; shared policy only compares the proven value.
    pub fn bind_hsaco_with(
        &self,
        expected_n_cu: u32,
        payload: &Payload<'_>,
        read_capability: impl FnMut(&str) -> Option<u32>,
    ) -> Result<()> {
        self.bind_object_with(PayloadKind::Hsaco, expected_n_cu, payload, read_capability)
    }

    fn bind_object_with(
        &self,
        kind: PayloadKind,
        expected_n_cu: u32,
        payload: &Payload<'_>,
        read_capability: impl FnMut(&str) -> Option<u32>,
    ) -> Result<()> {
        self.objects
            .iter()
            .find(|binding| binding.section == payload.section && binding.kind == kind)
            .ok_or_else(|| "mixed step: undeclared object".to_string())?
            .bind_object_with(kind, expected_n_cu, payload, read_capability)
    }
}

/// Validate the host image uploaded to `FlashDecode.t6`.
pub fn validate_decode_slots(
    slots: &[i32],
    decode_rows: u32,
    physical_slot_capacity: u32,
) -> Result<()> {
    require(
        slots.len() == decode_rows as usize
            && slots
                .iter()
                .all(|&slot| slot >= 0 && (slot as u32) < physical_slot_capacity),
        "decode slot map",
    )
}

/// Validate the physical-slot image against the selected program operand.
pub fn validate_decode_slot_binding(
    slots: &[i32],
    decode_rows: u32,
    physical_slot_capacity: u32,
    operand: Option<u16>,
) -> Result<()> {
    validate_decode_slots(slots, decode_rows, physical_slot_capacity)?;
    require(
        operand.is_some()
            || slots
                .iter()
                .enumerate()
                .all(|(row, &slot)| slot == row as i32),
        "unmapped FlashDecode requires compact slots",
    )
}

pub fn validate_decode_slot_tensor(
    handle: u16,
    decode_rows: u32,
    tensor: Option<TensorContract<'_>>,
) -> Result<Option<u16>> {
    if handle == TENSOR_NONE16 {
        return Ok(None);
    }
    let tensor = tensor.ok_or("mixed step: invalid FlashDecode slot map handle")?;
    require(
        tensor.name == DECODE_SLOT_TENSOR
            && !tensor.initialized
            && tensor.bytes >= u64::from(decode_rows) * 4,
        "FlashDecode slot map tensor",
    )?;
    Ok(Some(handle))
}

/// Validate and return the runtime-filled physical-slot tensor used by every
/// ordinary BF16 FlashDecode in one mixed program.
pub fn flash_decode_slot_operand(
    program: &aux_program::Program,
    decode_rows: u32,
    tensors: &[TensorContract<'_>],
) -> Result<Option<u16>> {
    let mut operand: Option<Option<u16>> = None;
    for inst in &program.insts {
        if inst.op != DevOp::FlashDecode as u16 || inst.i[1] & (1 << 16) != 0 {
            continue;
        }
        require(
            inst.i[0] == decode_rows && inst.t[7] == TENSOR_NONE16,
            "FlashDecode row or operand contract",
        )?;
        let current = validate_decode_slot_tensor(
            inst.t[6],
            decode_rows,
            tensors.get(inst.t[6] as usize).copied(),
        )?;
        if let Some(prior) = operand {
            require(current == prior, "inconsistent FlashDecode slot map")?;
        } else {
            operand = Some(current);
        }
    }
    Ok(operand.flatten())
}

/// Validate the row-sensitive operands required by a BF16 direct-KV mixed
/// program. Canonical prefill spans are supplied through `PlowProgram`; no
/// instruction tensor carries a second request-table encoding.
pub fn dense_consumer_contract(
    program: &aux_program::Program,
    decode_rows: u32,
    tensors: &[TensorContract<'_>],
) -> Result<u16> {
    validate_dense_consumers(program, decode_rows, tensors, false)
}

pub fn dense_amd_consumer_contract(
    program: &aux_program::Program,
    decode_rows: u32,
    tensors: &[TensorContract<'_>],
) -> Result<u16> {
    validate_dense_consumers(program, decode_rows, tensors, true)
}

pub fn dense_amd_capacity_consumer_contract(
    program: &aux_program::Program,
    decode_capacity: u32,
    tensors: &[TensorContract<'_>],
) -> Result<u16> {
    let slot = dense_amd_consumer_contract(program, decode_capacity, tensors)?;
    for inst in &program.insts {
        match DevOp::from_u16(inst.op) {
            Some(
                DevOp::Embed
                | DevOp::RmsNorm
                | DevOp::HeadNormRope
                | DevOp::NormResidual
                | DevOp::GemmGlu,
            ) => {
                require(inst.i[0] == program.rows, "mixed dynamic body row capacity")?;
            }
            Some(DevOp::Gemm) => {
                require(
                    (inst.i[0] == program.rows || inst.i[0] == decode_capacity)
                        && inst.i[4] == 0
                        && inst.i[5] == 0,
                    "mixed dynamic GEMM row capacity or offset",
                )?;
            }
            Some(DevOp::FlashMerge) if inst.i[4] == 0 => {
                require(
                    inst.i[0] == decode_capacity,
                    "mixed dynamic decode merge row capacity",
                )?;
            }
            _ => {}
        }
        if inst.op == DevOp::SoftCap as u16 {
            require(
                inst.i[1] == decode_capacity && inst.i[0] > 0 && inst.i[0] % decode_capacity == 0,
                "mixed dynamic softcap row capacity",
            )?;
            require(
                tensors
                    .get(inst.t[0] as usize)
                    .is_some_and(|t| !t.initialized && t.bytes >= u64::from(inst.i[0]) * 2),
                "mixed dynamic softcap tensor capacity",
            )?;
        } else if matches!(
            DevOp::from_u16(inst.op),
            Some(DevOp::Argmax | DevOp::ArgmaxFin)
        ) {
            require(
                inst.i[1].max(1) == decode_capacity,
                "mixed dynamic argmax row capacity",
            )?;
        }
    }
    Ok(slot)
}

fn validate_dense_consumers(
    program: &aux_program::Program,
    decode_rows: u32,
    tensors: &[TensorContract<'_>],
    split_prefill: bool,
) -> Result<u16> {
    require(
        decode_rows > 0 && decode_rows < program.rows,
        "mixed dense row geometry",
    )?;
    let decode_slot = flash_decode_slot_operand(program, decode_rows, tensors)?
        .ok_or("mixed step: dense FlashDecode requires a slot map")?;
    let mut decode_attention = 0usize;
    let mut prefill_attention = 0usize;
    let mut kv_writers = 0usize;
    for (index, inst) in program.insts.iter().enumerate() {
        let op = DevOp::from_u16(inst.op).ok_or("mixed step: unknown opcode")?;
        match op {
            DevOp::FlashDecode if inst.i[1] & (1 << 16) == 0 => {
                decode_attention += 1;
            }
            DevOp::FlashPrefill => {
                require(
                    inst.i[0] == program.rows
                        && inst.i[1] == program.rows
                        && (if inst.i[7] == 1 {
                            inst.t[5] != TENSOR_NONE16
                        } else {
                            split_prefill && inst.i[7] > 1 && inst.t[5] == TENSOR_NONE16
                        })
                        && inst.t[6] == TENSOR_NONE16,
                    "mixed dense FlashPrefill span contract",
                )?;
                if inst.i[7] > 1 {
                    let merge = program
                        .insts
                        .get(index + 1)
                        .ok_or("mixed step: split prefill missing merge")?;
                    require(
                        merge.op == DevOp::FlashMerge as u16
                            && merge.i[0] == program.rows
                            && merge.i[1] == inst.i[2]
                            && merge.i[2] == inst.i[7]
                            && merge.i[3] == inst.i[6]
                            && merge.i[4] == decode_rows
                            && merge.t[1] == inst.t[0]
                            && merge.t[2] == inst.t[1],
                        "mixed dense split prefill merge pairing",
                    )?;
                    require(
                        inst.t[0] != inst.t[1]
                            && inst.t[0] != merge.t[0]
                            && inst.t[1] != merge.t[0]
                            && program
                                .insts
                                .iter()
                                .filter(|d| d.op == DevOp::FlashDecode as u16)
                                .all(|d| {
                                    ![d.t[0], d.t[1]]
                                        .iter()
                                        .any(|t| *t == inst.t[0] || *t == inst.t[1])
                                }),
                        "mixed dense split prefill scratch isolation",
                    )?;
                    let extent = |factors: &[u32]| -> Result<u64> {
                        factors.iter().try_fold(1u64, |bytes, &factor| {
                            require(factor != 0, "mixed dense split prefill zero extent")?;
                            bytes
                                .checked_mul(factor as u64)
                                .ok_or_else(|| "mixed step: split prefill extent overflow".into())
                        })
                    };
                    let rows = program.rows;
                    let heads = inst.i[2];
                    let hd = inst.i[6];
                    let ns = inst.i[7];
                    for (tensor, bytes) in [
                        (inst.t[0], extent(&[rows, heads, ns, hd, 4])?),
                        (inst.t[1], extent(&[rows, heads, ns, 2, 4])?),
                        (merge.t[0], extent(&[rows, heads, hd, 2])?),
                    ] {
                        require(
                            tensors
                                .get(tensor as usize)
                                .is_some_and(|t| !t.initialized && t.bytes >= bytes),
                            "mixed dense split prefill tensor capacity",
                        )?;
                    }
                }
                prefill_attention += 1;
            }
            DevOp::FlashMerge if inst.i[4] != 0 => {
                require(
                    split_prefill
                        && index > 0
                        && program.insts[index - 1].op == DevOp::FlashPrefill as u16
                        && program.insts[index - 1].i[7] > 1,
                    "mixed dense unexpected prefill merge tag",
                )?;
            }
            DevOp::HeadNormRope if inst.fj[1] != 0 => {
                require(
                    inst.i[0] == program.rows
                        && inst.t[6] == decode_slot
                        && inst.t[7] == TENSOR_NONE16,
                    "mixed dense HeadNormRope row contract",
                )?;
                kv_writers += 1;
            }
            DevOp::FlashDecodeFp8 | DevOp::FlashPrefillFp8 | DevOp::HeadNormRopeFp8 => {
                return Err("mixed step: dense consumer requires BF16 direct KV".into());
            }
            _ => {}
        }
    }
    require(
        decode_attention > 0 && prefill_attention > 0 && kv_writers > 0,
        "mixed dense attention or KV writer coverage",
    )?;
    Ok(decode_slot)
}

fn record_payload(
    payloads: &mut std::collections::BTreeMap<(u32, String), (u32, String, Capability)>,
    binding: &PayloadBinding,
) -> Result<()> {
    let key = (binding.kind.section_kind(), binding.section.clone());
    let identity = (
        binding.version,
        binding.sha256.clone(),
        binding.capability.clone(),
    );
    require(
        payloads
            .insert(key, identity.clone())
            .is_none_or(|existing| existing == identity),
        "conflicting payload identity",
    )
}

#[cfg(test)]
#[path = "mixed_step_tests.rs"]
mod tests;

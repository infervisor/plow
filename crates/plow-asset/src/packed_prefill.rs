use crate::{live_kv, program::Packet};
use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SECTION: &str = "packed_prefill";
pub const CAPABILITY: &str = "plow_pf_request_abi";
pub const CAPABILITY_VALUE: u32 = 2;
pub const MASKED_PADDING_CAPABILITY: &str = "plow_pf_masked_padding_abi";
pub const FP8_MASKED_PADDING_CAPABILITY: &str = "plow_pf_fp8_masked_padding_abi";
pub const FP8_CAPABILITY: &str = "plow_pf_fp8_request_abi";
pub const FP8_CAPABILITY_VALUE: u32 = 1;
// FP8 attention uses t6/t7 for scales; tagged i4 carries the request handle.
pub const FP8_REQUEST_TAG: u32 = 1 << 31;
type Result<T> = std::result::Result<T, String>;
fn need(ok: bool, text: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("packed prefill: {text}"))
    }
}

fn downstream_merge(insts: &[DevInst64], pc: usize) -> Result<&DevInst64> {
    let d = &insts[pc];
    // Layer scratch is reused after its merge has consumed the partials.
    let mut merges = insts[pc + 1..]
        .iter()
        .take_while(|m| {
            !(matches!(
                DevOp::from_u16(m.op),
                Some(DevOp::FlashPrefill | DevOp::FlashPrefillFp8)
            ) && m.t[..2].iter().any(|h| d.t[..2].contains(h)))
        })
        .filter(|m| m.op == DevOp::FlashMerge as u16 && m.t[1] == d.t[0] && m.t[2] == d.t[1]);
    let merge = merges
        .next()
        .ok_or("packed prefill: missing downstream merge")?;
    need(merges.next().is_none(), "one downstream merge required")?;
    Ok(merge)
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Map {
    pub original: u16,
    pub slots: u16,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_request_rows: Option<u32>,
    pub slot: u16,
    pub request: u16,
    pub maps: Vec<Map>,
    pub programs: Vec<String>,
}
impl Manifest {
    pub fn validate_object(
        &self,
        mut read_capability: impl FnMut(&str) -> Option<u32>,
    ) -> Result<()> {
        need(matches!(self.version, 1 | 2), "object manifest version")?;
        if self.max_request_rows.is_some() {
            need(
                read_capability(MASKED_PADDING_CAPABILITY) == Some(1),
                "object requires masked padding ABI1",
            )?;
            if self.version == 2 {
                need(
                    read_capability(FP8_MASKED_PADDING_CAPABILITY) == Some(1),
                    "object requires FP8 masked padding ABI1",
                )?;
            }
        }
        need(
            read_capability(CAPABILITY) == Some(CAPABILITY_VALUE),
            "object requires packed request ABI2",
        )?;
        if self.version == 2 {
            need(
                read_capability(FP8_CAPABILITY) == Some(FP8_CAPABILITY_VALUE),
                "object requires FP8 request ABI1",
            )?;
        }
        Ok(())
    }

    /// Bind a validated prefill instruction, or restore its ordinary request operands.
    pub fn bind_request(&self, d: &mut DevInst64, packed: bool) {
        let slot = if packed { self.slot } else { TENSOR_NONE16 };
        let request = if packed { self.request } else { TENSOR_NONE16 };
        match DevOp::from_u16(d.op) {
            Some(DevOp::HeadNormRope) => d.t[6] = slot,
            Some(DevOp::HeadNormRopeFp8) => d.t[7] = slot,
            Some(DevOp::FlashPrefill) => {
                d.t[6] = request;
                for map in &self.maps {
                    if d.t[7] == if packed { map.original } else { map.slots } {
                        d.t[7] = if packed { map.slots } else { map.original };
                        break;
                    }
                }
            }
            Some(DevOp::FlashPrefillFp8) => {
                d.i[4] = if packed {
                    FP8_REQUEST_TAG | u32::from(self.request)
                } else {
                    0
                };
            }
            Some(DevOp::FlashMerge) => d.t[7] = request,
            _ => {}
        }
    }

    pub fn validate(&self, p: &Packet<'_>, live: &live_kv::Manifest) -> Result<()> {
        live.validate(p)?;
        need(
            self.version == live.version && !p.tp && p.prefill_count > 0,
            "version/topology",
        )?;
        need(
            self.programs.len() == p.prefill_count && live.batch <= i32::MAX as u32 / 4,
            "bucket/table capacity",
        )?;
        let rows = p.programs[..p.prefill_count]
            .iter()
            .map(|x| x.rows)
            .max()
            .unwrap();
        need(rows <= i32::MAX as u32, "row index width")?;
        let write_rows = self.max_request_rows.unwrap_or(rows);
        if self.max_request_rows.is_some() {
            need(
                write_rows > 0
                    && write_rows <= rows
                    && p.programs[..p.prefill_count]
                        .iter()
                        .any(|g| g.rows == write_rows),
                "request limit must match a prefill rung",
            )?;
        }
        for cache in &live.caches {
            need(
                cache.window == 0
                    || u64::from(cache.stride) >= u64::from(live.max_ctx)
                    || u64::from(cache.stride)
                        >= u64::from(cache.window) + u64::from(write_rows) - 1,
                "ring must retain the attention window across all padded KV writes",
            )?;
        }
        let mut handles = BTreeSet::new();
        for (h, name, bytes) in [
            (self.slot, "pf.request.slot", u64::from(rows) * 4),
            (
                self.request,
                "pf.request.table",
                (1 + 4 * u64::from(live.batch)) * 4,
            ),
        ] {
            let t = p.tensors.get(h as usize).ok_or("packed table handle")?;
            need(
                h != TENSOR_NONE16
                    && handles.insert(h)
                    && t.name == name
                    && t.bytes == bytes
                    && !t.initialized,
                "declared table geometry",
            )?;
        }
        let mut originals = BTreeSet::new();
        for m in &self.maps {
            let t = p
                .tensors
                .get(m.slots as usize)
                .ok_or("packed map table handle")?;
            need(
                handles.insert(m.slots)
                    && originals.insert(m.original)
                    && t.name == format!("pf.request.maps.{}", m.original)
                    && t.bytes == 8 * u64::from(live.batch)
                    && !t.initialized
                    && live.maps.iter().any(|x| x.handle == u32::from(m.original)),
                "declared descriptor table",
            )?;
        }
        need(
            originals == live.maps.iter().map(|m| m.handle as u16).collect(),
            "descriptor table coverage",
        )?;
        need(
            !p.programs.iter().flat_map(|g| g.insts).any(|d| {
                matches!(
                    DevOp::from_u16(d.op),
                    Some(
                        DevOp::KdaConv
                            | DevOp::KdaGate
                            | DevOp::KdaStateStep
                            | DevOp::KdaGatedNorm
                            | DevOp::KdaConv3
                            | DevOp::KdaStateStepG
                            | DevOp::KdaConvStateStepG
                            | DevOp::KdaChunkPrepare
                            | DevOp::KdaChunkIntra
                            | DevOp::KdaChunkWu
                            | DevOp::KdaChunkCarry
                            | DevOp::KdaDecodeFused
                            | DevOp::QwenGdnConv
                            | DevOp::QwenGdnStep
                            | DevOp::QwenGdnConvPrefill
                            | DevOp::QwenGdnQkvPrep
                            | DevOp::QwenGdnGatePrep
                            | DevOp::QwenGdnPrefill
                    )
                )
            }),
            "recurrent state is unsupported",
        )?;
        for (pi, g) in p.programs.iter().enumerate() {
            let mut flash_sites = 0usize;
            let mut slot_writers = 0usize;
            if pi < p.prefill_count {
                need(
                    self.programs[pi] == live_kv::program_digest(g),
                    "program identity",
                )?;
                need(
                    g.gq_seg_ofs.len() > 2
                        && g.gq_seg_ofs.first() == Some(&0)
                        && g.gq_seg_ofs.last() == Some(&(g.gq_stream.len() as u32))
                        && g.gq_seg_ofs.windows(2).all(|w| w[0] < w[1]),
                    "complete segmented chain required",
                )?;
            }
            for (pc, d) in g.insts.iter().enumerate() {
                need(
                    !d.t.iter().any(|h| handles.contains(h)),
                    "table already consumed by baseline packet",
                )?;
                if pi >= p.prefill_count {
                    continue;
                }
                let op = DevOp::from_u16(d.op).ok_or("packed opcode")?;
                if matches!(op, DevOp::FlashPrefill | DevOp::FlashPrefillFp8) {
                    flash_sites += 1;
                    need(
                        matches!(d.i[6], 256 | 512)
                            && d.i[7] > 0
                            && (op == DevOp::FlashPrefillFp8 || d.t[6] == TENSOR_NONE16),
                        "attention contract",
                    )?;
                    let product = |xs: &[u32], bytes: u64| -> Result<u64> {
                        xs.iter().try_fold(bytes, |n, &x| {
                            n.checked_mul(u64::from(x))
                                .ok_or("attention extent overflow".into())
                        })
                    };
                    let extent = |h: u16, bytes: u64| -> Result<()> {
                        let t = p.tensors.get(h as usize).ok_or("attention tensor handle")?;
                        need(
                            !t.initialized && t.bytes >= bytes,
                            "attention tensor extent",
                        )
                    };
                    let qbytes = product(&[g.rows, d.i[2], d.i[6]], 2)?;
                    extent(d.t[2], qbytes)?;
                    need(
                        d.t[..5].iter().copied().collect::<BTreeSet<_>>().len() == 5,
                        "attention operand alias",
                    )?;
                    if d.t[5] == TENSOR_NONE16 {
                        extent(d.t[0], product(&[g.rows, d.i[2], d.i[6], d.i[7]], 4)?)?;
                        extent(d.t[1], product(&[g.rows, d.i[2], d.i[7]], 8)?)?;
                        let merge = downstream_merge(g.insts, pc)?;
                        need(
                            merge.i[..4] == [g.rows, d.i[2], d.i[7], d.i[6]],
                            "merge geometry",
                        )?;
                        extent(merge.t[0], qbytes)?;
                        need(!d.t[..5].contains(&merge.t[0]), "merge output alias")?;
                    } else {
                        need(
                            d.i[7] == 1 && !d.t[..5].contains(&d.t[5]),
                            "fused attention extent/alias",
                        )?;
                        extent(d.t[5], qbytes)?;
                    }

                    // Local request tiles change slice ownership; the original fine waits
                    // remain valid only across a completed kernel boundary.
                    let mut covering = 0;
                    for pair in g.gq_seg_ofs.windows(2) {
                        let entries = g
                            .gq_stream
                            .get(pair[0] as usize..pair[1] as usize)
                            .ok_or("packed segment range")?;
                        if entries.iter().any(|e| e.inst as usize == pc) {
                            need(
                                entries.iter().all(|e| e.inst as usize == pc),
                                "attention must occupy its complete segment",
                            )?;
                            need(
                                entries.len() == d.blocks as usize
                                    && entries.iter().map(|e| e.slice).collect::<BTreeSet<_>>()
                                        == (0..u32::from(d.blocks)).collect(),
                                "attention slice coverage",
                            )?;
                            covering += 1;
                        }
                    }
                    need(covering == 1, "attention segment coverage")?;
                }
                if matches!(op, DevOp::HeadNormRope | DevOp::HeadNormRopeFp8) {
                    let slot_operand = if op == DevOp::HeadNormRopeFp8 { 7 } else { 6 };
                    need(d.t[slot_operand] == TENSOR_NONE16, "existing slot map")?;
                    slot_writers += usize::from(d.fj[1] != 0);
                }
                if op == DevOp::FlashMerge {
                    need(
                        d.t[4..].iter().all(|&h| h == TENSOR_NONE16),
                        "merge request operand",
                    )?;
                }
            }
            if pi < p.prefill_count {
                need(
                    flash_sites > 0 && slot_writers > 0,
                    "request-aware attention and KV writer required",
                )?;
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Request {
    pub slot: usize,
    pub start: usize,
    pub len: usize,
    pub prompt: usize,
}
#[derive(Debug, PartialEq, Eq)]
pub struct Plan {
    pub table: Vec<i32>,
    pub slots: Vec<i32>,
    pub positions: Vec<i32>,
    pub mapped_ends: Vec<(usize, u32)>,
}
pub fn plan(
    requests: &[Request],
    frontiers: &[u32],
    bucket: usize,
    max_ctx: usize,
) -> Result<Plan> {
    plan_with_limit(requests, frontiers, bucket, max_ctx, None)
}

pub fn plan_with_limit(
    requests: &[Request],
    frontiers: &[u32],
    bucket: usize,
    max_ctx: usize,
    max_request_rows: Option<u32>,
) -> Result<Plan> {
    need(
        !requests.is_empty()
            && requests.len() <= frontiers.len()
            && bucket > 0
            && bucket <= i32::MAX as usize
            && max_ctx <= i32::MAX as usize,
        "request/table capacity",
    )?;
    let mut total = 0usize;
    let mut seen = BTreeSet::new();
    let mut out = Plan {
        table: vec![requests.len() as i32],
        slots: Vec::with_capacity(bucket),
        positions: Vec::with_capacity(bucket),
        mapped_ends: Vec::with_capacity(requests.len()),
    };
    for r in requests {
        need(
            max_request_rows.is_none_or(|limit| r.len <= limit as usize),
            "request exceeds compiled chunk limit",
        )?;
        need(
            r.slot < frontiers.len() && seen.insert(r.slot) && r.slot <= i32::MAX as usize,
            "physical slot or duplicate",
        )?;
        let end = r
            .start
            .checked_add(r.len)
            .ok_or("packed request overflow")?;
        need(
            r.len > 0 && r.start == frontiers[r.slot] as usize && end <= r.prompt && end <= max_ctx,
            "request frontier/extent",
        )?;
        let next = total.checked_add(r.len).ok_or("packed row overflow")?;
        need(next <= bucket, "packed bucket overflow")?;
        out.table
            .extend_from_slice(&[total as i32, r.len as i32, r.slot as i32, end as i32]);
        out.slots.extend(std::iter::repeat_n(r.slot as i32, r.len));
        out.positions.extend((r.start..end).map(|x| x as i32));
        out.mapped_ends.push((r.slot, end as u32));
        total = next;
    }
    let padding = bucket - total;
    if max_request_rows.is_some() {
        out.slots.resize(bucket, -1);
        out.positions.resize(bucket, 0);
        return Ok(out);
    }
    let (padding_index, padding_request, padded) = requests
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, request)| {
            let end = request.start.checked_add(request.len)?;
            let padded = end.checked_add(padding)?;
            (padded <= max_ctx).then_some((index, request, padded))
        })
        .ok_or("packed prefill: padding exceeds every request's physical context")?;
    let end = padding_request.start + padding_request.len;
    out.slots
        .extend(std::iter::repeat_n(padding_request.slot as i32, padding));
    out.positions.extend((end..padded).map(|x| x as i32));
    out.mapped_ends[padding_index].1 = padded as u32;
    Ok(out)
}

/// The span table for sub-chunk STAGE `stage` of a packed prefill processed in slices of
/// `stage_rows` rows per request.
///
/// WHY. The sliding-window KV ring invariant (`dev_isa.h`, "SLIDING-WINDOW KV RING") is
/// `ring >= window + rows_written_per_request_per_launch - 1`, because a launch writes ALL its
/// K/V rows before any flash reads and a request's rows must not wrap onto the rows its own
/// queries read. Writing a whole 4096-row chunk therefore needs an 8192-row ring — 2.5 GiB per
/// slot on the 12B, so only 16 slots fit at ctx 16k. Writing the same chunk in 1024-row stages
/// needs only 2048 ring rows (640 MiB) and 32 slots fit, WITHOUT dropping the launch width to
/// 1024 (which is what makes the chunk-1024 32-slot packet lose long prompts: it turns a 15k
/// prompt into 15 launches).
///
/// The flash body needs NO change. It already derives `rq0`, `qlen`, `slot` and `kvlen` per
/// request from this table and computes `qp0 = kvlen - qlen` itself (`op_attention_sm90.cuh`),
/// so clipping the table is enough: for stage `i` this yields `qp0 = r.start + i * stage_rows`,
/// the stage's true absolute query position, which is what the causal mask and the
/// sliding-window floor read.
///
/// A request shorter than the stage offset contributes a ZERO-length entry rather than being
/// dropped, so entry `r` keeps addressing request `r` in every stage. The kernel already
/// guards this: its per-request work count is 0 when `qlen <= 0`.
pub fn plan_stage(plan: &Plan, stage: usize, stage_rows: usize) -> Result<Vec<i32>> {
    let nreq = stage_table_reqs(&plan.table, stage_rows)?;
    let skip = stage.checked_mul(stage_rows).ok_or("stage offset overflow")?;
    let mut out = Vec::with_capacity(plan.table.len());
    out.push(plan.table[0]);
    for r in 0..nreq {
        let rq0 = plan.table[1 + 4 * r];
        let qlen = plan.table[2 + 4 * r];
        let slot = plan.table[3 + 4 * r];
        let kvlen = plan.table[4 + 4 * r];
        need(qlen >= 0 && kvlen >= qlen, "span entry")?;
        // `kvlen - qlen` is the request's frontier: the prior context this chunk continues from.
        let start = kvlen - qlen;
        let taken = (qlen as usize).min(skip) as i32;
        let len = ((qlen - taken) as usize).min(stage_rows) as i32;
        out.extend_from_slice(&[rq0 + taken, len, slot, start + taken + len]);
    }
    Ok(out)
}

/// The per-row slot map for sub-chunk STAGE `stage`: every row outside the stage is masked to
/// `-1`, so only this stage's rows write K/V.
///
/// A stage's rows are NOT contiguous in the packed buffer — with two requests packed, stage 0 is
/// rows `[rq0_a, rq0_a+S)` and `[rq0_b, rq0_b+S)` — so a row offset and count cannot express it.
/// A mask can, and needs no kernel change: `d_headnorm_rope` already skips a masked row
/// (`op_norm.cuh:781`, `if (out_stride && pfslot && pfslot[t] < 0) continue;`, and the same at
/// `:985` for the fp8 arm). That is the masked-padding mechanism, reused per stage.
///
/// Requires a plan whose padding is ALREADY masked — `plan_with_limit` with `max_request_rows`,
/// i.e. a `PLOW_MAX_REQUEST_CHUNK` packet. On an unmasked plan the padding rows carry a real
/// slot and exist so every bucket row is owned; masking them here would silently change what the
/// KV write covers, and the object must also advertise `plow_pf_masked_padding_abi`.
pub fn stage_slots(plan: &Plan, stage: usize, stage_rows: usize) -> Result<Vec<i32>> {
    let nreq = stage_table_reqs(&plan.table, stage_rows)?;
    let skip = stage.checked_mul(stage_rows).ok_or("stage offset overflow")?;
    let covered: usize = (0..nreq)
        .map(|r| plan.table[2 + 4 * r].max(0) as usize)
        .sum();
    need(
        plan.slots.len() >= covered && plan.slots[covered..].iter().all(|&s| s < 0),
        "stage slots need a masked-padding plan (plan_with_limit with max_request_rows)",
    )?;
    let mut out = vec![-1i32; plan.slots.len()];
    for r in 0..nreq {
        let rq0 = plan.table[1 + 4 * r] as usize;
        let qlen = plan.table[2 + 4 * r].max(0) as usize;
        let taken = qlen.min(skip);
        let len = (qlen - taken).min(stage_rows);
        let from = rq0 + taken;
        need(from + len <= plan.slots.len(), "stage row extent")?;
        out[from..from + len].copy_from_slice(&plan.slots[from..from + len]);
    }
    Ok(out)
}

/// How many stages `plan` needs at `stage_rows`: the longest request decides, and a plan whose
/// every request already fits one stage needs exactly one — the unstaged launch, unchanged.
pub fn stages_needed(plan: &Plan, stage_rows: usize) -> Result<usize> {
    let nreq = stage_table_reqs(&plan.table, stage_rows)?;
    let longest = (0..nreq)
        .map(|r| plan.table[2 + 4 * r].max(0) as usize)
        .max()
        .unwrap_or(0);
    Ok(longest.div_ceil(stage_rows).max(1))
}

fn stage_table_reqs(table: &[i32], stage_rows: usize) -> Result<usize> {
    need(stage_rows > 0, "stage_rows must be non-zero")?;
    need(!table.is_empty() && table.len() % 4 == 1, "span table shape")?;
    let nreq = table[0].max(0) as usize;
    need(table.len() == 1 + 4 * nreq, "span table request count")?;
    Ok(nreq)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn masked_padding_requires_dtype_specific_object_capabilities() {
        for version in [1, 2] {
            for masked in [None, Some(0), Some(1), Some(2)] {
                for fp8_masked in [None, Some(0), Some(1), Some(2)] {
                    let manifest = Manifest {
                        version,
                        max_request_rows: Some(1024),
                        slot: 0,
                        request: 1,
                        maps: vec![],
                        programs: vec![],
                    };
                    assert_eq!(
                        manifest
                            .validate_object(|name| match name {
                                CAPABILITY => Some(CAPABILITY_VALUE),
                                FP8_CAPABILITY => Some(FP8_CAPABILITY_VALUE),
                                MASKED_PADDING_CAPABILITY => masked,
                                FP8_MASKED_PADDING_CAPABILITY => fp8_masked,
                                _ => None,
                            })
                            .is_ok(),
                        masked == Some(1) && (version == 1 || fp8_masked == Some(1)),
                    );
                }
            }
        }
    }

    #[test]
    fn masked_padding_never_extends_a_real_requests_kv_writes() {
        let request = Request {
            slot: 1,
            start: 15360,
            len: 1024,
            prompt: 16384,
        };
        let p = plan_with_limit(&[request], &[0, 15360], 8192, 16384, Some(1024)).unwrap();
        assert_eq!(p.table, [1, 0, 1024, 1, 16384]);
        assert_eq!(p.mapped_ends, [(1, 16384)]);
        assert!(p.slots[..1024].iter().all(|&slot| slot == 1));
        assert_eq!(p.positions[..1024], (15360..16384).collect::<Vec<_>>());
        assert!(p.slots[1024..].iter().all(|&slot| slot == -1));
        assert!(p.positions[1024..].iter().all(|&pos| pos == 0));
        assert!(plan(&[request], &[0, 15360], 8192, 16384).is_err());
        assert!(
            plan_with_limit(&[request], &[0, 15360], 8192, 16384, Some(512))
                .unwrap_err()
                .contains("compiled chunk limit")
        );
        assert!(
            plan_with_limit(&[request, request], &[0, 15360], 8192, 16384, Some(1024))
                .unwrap_err()
                .contains("duplicate")
        );
    }

    #[test]
    fn legacy_manifest_does_not_emit_a_request_limit() {
        let bytes = br#"{"version":1,"slot":0,"request":1,"maps":[],"programs":[]}"#;
        let manifest: Manifest = serde_json::from_slice(bytes).unwrap();
        assert_eq!(manifest.max_request_rows, None);
        assert_eq!(serde_json::to_vec(&manifest).unwrap(), bytes);
    }

    #[test]
    fn object_contract_refuses_missing_or_incompatible_fp8_bindings() {
        for (version, base, fp8, valid) in [
            (1, Some(2), None, true),
            (1, Some(2), Some(1), true),
            (1, Some(1), None, false),
            (2, Some(2), Some(1), true),
            (2, Some(2), None, false),
            (2, Some(1), Some(1), false),
            (2, None, Some(1), false),
            (2, Some(2), Some(2), false),
            (3, Some(2), Some(1), false),
        ] {
            let m = Manifest {
                max_request_rows: None,
                version,
                slot: 0,
                request: 1,
                maps: Vec::new(),
                programs: Vec::new(),
            };
            assert_eq!(
                m.validate_object(|name| if name == CAPABILITY { base } else { fp8 })
                    .is_ok(),
                valid,
                "version={version} base={base:?} fp8={fp8:?}"
            );
        }
    }

    #[test]
    #[ignore = "CPU cubin inspection; set TEST_PACKED_FP8_CUBINS to colon-separated paths"]
    fn fp8_objects_advertise_both_request_contracts() {
        let paths = std::env::var_os("TEST_PACKED_FP8_CUBINS").unwrap();
        let paths: Vec<_> = std::env::split_paths(&paths).collect();
        assert!(!paths.is_empty());
        for path in paths {
            let image = std::fs::read(&path).unwrap();
            assert_eq!(
                crate::cubin::global_u32(&image, CAPABILITY),
                Some(CAPABILITY_VALUE),
                "{}",
                path.display()
            );
            assert_eq!(
                crate::cubin::global_u32(&image, FP8_CAPABILITY),
                Some(FP8_CAPABILITY_VALUE),
                "{}",
                path.display()
            );
        }
    }

    #[test]
    #[ignore = "CPU cubin inspection; set TEST_PACKED_FP8_MASKED_CUBIN"]
    fn fp8_masked_object_advertises_fixed_writer() {
        let path = std::env::var_os("TEST_PACKED_FP8_MASKED_CUBIN").unwrap();
        let image = std::fs::read(path).unwrap();
        let manifest = Manifest {
            version: 2,
            max_request_rows: Some(1024),
            slot: 0,
            request: 1,
            maps: vec![],
            programs: vec![],
        };
        manifest
            .validate_object(|n| crate::cubin::global_u32(&image, n))
            .unwrap();
        assert!(manifest
            .validate_object(|n| {
                if n == FP8_MASKED_PADDING_CAPABILITY {
                    None
                } else {
                    crate::cubin::global_u32(&image, n)
                }
            })
            .is_err());
    }

    #[test]
    fn layer_scratch_reuse_requires_a_merge_before_overwrite() {
        let mut flash = DevInst64 {
            op: DevOp::FlashPrefill as u16,
            blocks: 1,
            fj: [0; 3],
            t: [TENSOR_NONE16; 8],
            i: [0; 8],
        };
        flash.t[..2].copy_from_slice(&[0, 1]);
        let mut merge = flash;
        merge.op = DevOp::FlashMerge as u16;
        merge.t[..3].copy_from_slice(&[2, 0, 1]);
        let chain = [flash, merge, flash, merge];
        assert!(downstream_merge(&chain, 0).is_ok());
        assert!(downstream_merge(&chain, 2).is_ok());
        assert!(downstream_merge(&[flash, flash, merge], 0).is_err());
        assert!(downstream_merge(&[merge, flash], 1).is_err());
        assert!(downstream_merge(&[flash, merge, merge], 0).is_err());
        let mut fp8 = flash;
        fp8.op = DevOp::FlashPrefillFp8 as u16;
        assert!(downstream_merge(&[fp8, merge, flash, merge], 0).is_ok());
        assert!(downstream_merge(&[fp8, flash, merge], 0).is_err());
        assert!(downstream_merge(&[flash, fp8, merge], 0).is_err());
    }
    #[test]
    fn ragged_physical_slots_and_padding() {
        let mut f = vec![0; 16];
        f[3] = 31;
        f[15] = 63;
        let p = plan(
            &[
                Request {
                    slot: 15,
                    start: 63,
                    len: 2,
                    prompt: 100,
                },
                Request {
                    slot: 3,
                    start: 31,
                    len: 3,
                    prompt: 100,
                },
            ],
            &f,
            8,
            128,
        )
        .unwrap();
        assert_eq!(p.table, vec![2, 0, 2, 15, 65, 2, 3, 3, 34]);
        assert_eq!(p.slots, vec![15, 15, 3, 3, 3, 3, 3, 3]);
        assert_eq!(p.positions, vec![63, 64, 31, 32, 33, 34, 35, 36]);
        assert_eq!(p.mapped_ends, vec![(15, 65), (3, 37)]);
    }
    #[test]
    fn malformed_requests_rejected_before_mapping() {
        let f = [0; 4];
        let r = Request {
            slot: 0,
            start: 0,
            len: 2,
            prompt: 4,
        };
        assert!(plan(&[], &f, 4, 8).is_err());
        assert!(plan(&[r, r], &f, 4, 8).is_err());
        for bad in [
            Request { slot: 4, ..r },
            Request { len: 0, ..r },
            Request { start: 1, ..r },
            Request { len: 5, ..r },
            Request {
                len: usize::MAX,
                ..r
            },
        ] {
            assert!(plan(&[bad], &f, 4, 8).is_err());
        }
        assert!(plan(&[r], &f, 1, 8).is_err());
        assert!(plan(&[r], &f, 8, 4).is_err());
        let tail = Request {
            start: 7,
            len: 1,
            prompt: 8,
            ..r
        };
        assert!(plan(&[tail], &[7], 2, 8).is_err());
    }
    #[test]
    fn padding_uses_a_request_with_remaining_context_capacity() {
        let frontiers = [2, 7];
        let p = plan(
            &[
                Request {
                    slot: 0,
                    start: 2,
                    len: 2,
                    prompt: 8,
                },
                Request {
                    slot: 1,
                    start: 7,
                    len: 1,
                    prompt: 8,
                },
            ],
            &frontiers,
            4,
            8,
        )
        .unwrap();
        assert_eq!(p.table, [2, 0, 2, 0, 4, 2, 1, 1, 8]);
        assert_eq!(p.slots, [0, 0, 1, 0]);
        assert_eq!(p.positions, [2, 3, 7, 4]);
        assert_eq!(p.mapped_ends, [(0, 5), (1, 8)]);
    }
    #[test]
    fn real_partial_rows_and_padding_have_disjoint_ownership() {
        for hd in [256usize, 512] {
            for heads in [4usize, 8, 16] {
                for ns in [1usize, 3, 17, 33] {
                    let p = plan(
                        &[
                            Request {
                                slot: 3,
                                start: 31,
                                len: 33,
                                prompt: 128,
                            },
                            Request {
                                slot: 15,
                                start: 95,
                                len: 31,
                                prompt: 128,
                            },
                        ],
                        &[0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95],
                        128,
                        256,
                    )
                    .unwrap();
                    let mut spans = Vec::new();
                    for r in p.table[1..].chunks_exact(4) {
                        let q0 = r[0] as usize;
                        let rows = r[1] as usize;
                        spans.push((q0 * heads * ns * hd, (q0 + rows) * heads * ns * hd));
                    }
                    assert_eq!(spans[0].0, 0);
                    assert_eq!(spans[0].1, spans[1].0);
                    assert_eq!(spans[1].1, 64 * heads * ns * hd);
                }
            }
        }
    }

    /// Two requests of different lengths sharing one 4096-row launch, staged at 1024.
    fn staged_plan() -> Plan {
        plan_with_limit(
            &[
                // a long request continuing a 5000-token prefix, and a short fresh one
                Request { slot: 0, start: 5000, len: 4000, prompt: 9000 },
                Request { slot: 1, start: 0, len: 96, prompt: 96 },
            ],
            &[5000, 0],
            4096,
            16384,
            Some(4096),
        )
        .expect("plan")
    }

    #[test]
    fn stage_tables_cover_every_row_exactly_once_and_in_order() {
        let plan = staged_plan();
        let s = 1024;
        assert_eq!(stages_needed(&plan, s).unwrap(), 4); // 4000 rows -> 4 stages
        for r in 0..2usize {
            let rq0 = plan.table[1 + 4 * r];
            let qlen = plan.table[2 + 4 * r];
            let mut next = rq0;
            let mut total = 0;
            for stage in 0..stages_needed(&plan, s).unwrap() {
                let t = plan_stage(&plan, stage, s).unwrap();
                assert_eq!(t[0], plan.table[0], "request count is preserved");
                assert_eq!(t[3 + 4 * r], plan.table[3 + 4 * r], "slot is preserved");
                let (srq0, slen) = (t[1 + 4 * r], t[2 + 4 * r]);
                assert!(slen >= 0 && slen <= s as i32);
                assert_eq!(srq0, next, "stages are contiguous in the packed Q buffer");
                next += slen;
                total += slen;
            }
            assert_eq!(total, qlen, "every row of the request is covered exactly once");
        }
    }

    /// The whole point of the clipped table: the flash body computes `qp0 = kvlen - qlen`, and
    /// that must land on the stage's true absolute query position or the causal mask and the
    /// sliding-window floor are wrong.
    #[test]
    fn derived_qp0_is_the_stage_absolute_query_position() {
        let plan = staged_plan();
        let s = 1024;
        for r in 0..2usize {
            let qlen = plan.table[2 + 4 * r];
            let start = plan.table[4 + 4 * r] - qlen;
            for stage in 0..stages_needed(&plan, s).unwrap() {
                let t = plan_stage(&plan, stage, s).unwrap();
                let (slen, skvlen) = (t[2 + 4 * r], t[4 + 4 * r]);
                let taken = qlen.min((stage * s) as i32);
                assert_eq!(skvlen - slen, start + taken, "stage {stage} request {r} qp0");
                // KV visible to this stage is exactly what has been written by the end of it.
                assert_eq!(skvlen, start + taken + slen);
            }
        }
    }

    #[test]
    fn a_request_shorter_than_the_stage_offset_contributes_a_zero_length_entry() {
        let plan = staged_plan();
        // request 1 is 96 rows, so it is exhausted after stage 0.
        for stage in 1..4 {
            let t = plan_stage(&plan, stage, 1024).unwrap();
            assert_eq!(t[2 + 4 * 1], 0, "stage {stage} leaves the short request empty");
            // entry index still addresses request 1, and its slot is still intact
            assert_eq!(t[3 + 4 * 1], plan.table[3 + 4 * 1]);
        }
    }

    #[test]
    fn one_stage_wide_enough_reproduces_the_unstaged_table() {
        let plan = staged_plan();
        assert_eq!(stages_needed(&plan, 4096).unwrap(), 1);
        assert_eq!(plan_stage(&plan, 0, 4096).unwrap(), plan.table);
    }

    /// The ring invariant is about K/V WRITES, so the union of the stages' unmasked rows must be
    /// exactly the rows the unstaged launch would have written — no row written twice (it would
    /// wrap onto itself), none dropped (the KV would have a hole).
    #[test]
    fn stage_slot_masks_partition_the_written_rows() {
        let plan = staged_plan();
        let s = 1024;
        let stages = stages_needed(&plan, s).unwrap();
        let mut writes = vec![0u32; plan.slots.len()];
        for stage in 0..stages {
            let m = stage_slots(&plan, stage, s).unwrap();
            assert_eq!(m.len(), plan.slots.len());
            for (row, &slot) in m.iter().enumerate() {
                if slot >= 0 {
                    assert_eq!(slot, plan.slots[row], "an unmasked row keeps its own slot");
                    writes[row] += 1;
                }
            }
        }
        for (row, &n) in writes.iter().enumerate() {
            let expected = u32::from(plan.slots[row] >= 0);
            assert_eq!(n, expected, "row {row} written {n} times, expected {expected}");
        }
    }

    #[test]
    fn stage_slots_refuse_an_unmasked_padding_plan() {
        // No max_request_rows => padding rows carry a real slot, and masking them would change
        // what the KV write covers.
        let unmasked = plan(
            &[Request { slot: 0, start: 0, len: 100, prompt: 4096 }],
            &[0],
            4096,
            16384,
        )
        .expect("plan");
        assert!(unmasked.slots.iter().all(|&s| s >= 0));
        assert!(stage_slots(&unmasked, 0, 1024).is_err());
    }

    #[test]
    fn stage_rejects_a_malformed_table_and_a_zero_width() {
        let plan = staged_plan();
        assert!(plan_stage(&plan, 0, 0).is_err());
        let bad = Plan { table: vec![2, 0, 1, 0, 1], ..staged_plan() };
        assert!(plan_stage(&bad, 0, 1024).is_err(), "count/length disagree");
        assert!(stages_needed(&bad, 1024).is_err());
    }
}

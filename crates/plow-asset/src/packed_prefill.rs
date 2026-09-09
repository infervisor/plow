use crate::{live_kv, program::Packet};
use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SECTION: &str = "packed_prefill";
pub const CAPABILITY: &str = "plow_pf_request_abi";
pub const CAPABILITY_VALUE: u32 = 2;
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
    pub slot: u16,
    pub request: u16,
    pub maps: Vec<Map>,
    pub programs: Vec<String>,
}
impl Manifest {
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
        for cache in &live.caches {
            need(
                cache.window == 0
                    || u64::from(cache.stride) >= u64::from(cache.window) + u64::from(rows) - 1,
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
#[cfg(test)]
mod tests {
    use super::*;
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
}

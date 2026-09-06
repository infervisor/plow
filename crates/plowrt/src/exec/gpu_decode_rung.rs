use super::*;

pub(super) struct DecodeRung {
    pub(super) rows: usize,
    pub(super) object: Option<Arc<BoundDecodeObject>>,
    pub(super) kernarg: DevProgram,
    pub(super) counters: DeviceMem,
    pub(super) counter_bytes: usize,
    _tables: Vec<DeviceMem>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecodeSelection {
    Base(Option<usize>),
    Context(usize),
}

pub(super) fn decode_selection(base: Option<usize>, context: Option<usize>) -> DecodeSelection {
    context.map_or(DecodeSelection::Base(base), DecodeSelection::Context)
}

pub(super) fn decode_rung_index(
    mut widths: impl Iterator<Item = usize>,
    highest_slot: usize,
) -> Option<usize> {
    widths.position(|rows| rows > highest_slot)
}

pub(super) fn effective_decode_widths(
    narrow: impl Iterator<Item = usize>,
    widest: usize,
    widest_only: bool,
) -> Box<[u32]> {
    if widest_only {
        return vec![widest as u32].into_boxed_slice();
    }
    narrow
        .map(|rows| rows as u32)
        .chain(std::iter::once(widest as u32))
        .collect()
}

pub(super) fn validate_decode_ladder(blob: &DevBlob) -> Result<bool> {
    let splitk = blob
        .with_packet_view(plow_asset::splitk::validate)
        .map_err(RuntimeError::Rejected)?;
    let programs = blob.decode_progs();
    if programs.len() < 2 {
        return Ok(false);
    }
    let reject = |reason: &str| RuntimeError::Rejected(format!("decode ladder: {reason}"));
    if programs[0].t == 0 || programs.windows(2).any(|w| w[0].t >= w[1].t) {
        return Err(reject("widths must increase strictly"));
    }
    for g in programs {
        if g.packed_prefill_only
            || g.insts.is_empty()
            || g.gq_stream.is_empty()
            || g.gq_seg_ofs.len() < 2
            || g.gq_seg_ofs.first() != Some(&0)
            || g.gq_seg_ofs.last().copied() != Some(g.gq_stream.len() as u32)
            || g.gq_seg_ofs.windows(2).any(|w| w[0] >= w[1])
            || g.insts.iter().any(|d| {
                d.t.iter()
                    .any(|&id| id != TENSOR_NONE16 && id as usize >= blob.tensors.len())
            })
            || g.stream_ofs.len() != blob.n_cu as usize
            || g.stream_len.len() != blob.n_cu as usize
            || g.stream_ofs.iter().zip(&g.stream_len).any(|(&ofs, &len)| {
                (ofs as usize)
                    .checked_add(len as usize)
                    .is_none_or(|end| end > g.stream.len())
            })
            || g.stream.iter().chain(&g.gq_stream).any(|e| {
                e.inst as usize >= g.insts.len()
                    || e.wait_ofs as usize + e.wait_len as usize > g.waits.len()
                    || e.succ_ofs as usize + e.succ_len as usize > g.succs.len()
            })
            || g.waits.iter().any(|w| w.id >= g.n_counter)
            || g.succs.iter().any(|&id| id >= g.n_counter)
        {
            return Err(reject("invalid program tables"));
        }
        g.check_gq_topological()?;
        if g.l2_domains == 0 {
            validate_segment_windows(g)?;
        }
    }
    // Optional placed, segmented and opaque programs retain the existing widest path.
    if programs.iter().any(|g| {
        g.l2_domains != 0 || g.gq_seg_ofs.len() != 2 || g.check_coarse_single_segment().is_err()
    }) {
        return Ok(false);
    }
    let widest = programs.last().expect("multiple programs");
    let mut compatible = true;
    let mut same_shape = true;
    let mut normalized = Vec::new();
    for (index, g) in programs.iter().enumerate() {
        let logical = splitk.as_ref().map(|proof| &proof.canonical[index]);
        if let Some(proof) = &splitk {
            if proof.canonical[index].dependencies != proof.canonical[0].dependencies {
                return Err(reject(
                    "canonical projection dependencies differ across rungs",
                ));
            }
        }
        let mut insts = Vec::new();
        for d in logical.map_or(g.insts.as_slice(), |p| p.instructions.as_slice()) {
            let mut d = *d;
            d.blocks = 0;
            match DevOp::from_u16(d.op) {
                Some(DevOp::Nop) => {}
                Some(DevOp::Residual | DevOp::Glu | DevOp::SoftCap) => {
                    if d.i[0] == 0 || d.i[0] % g.t != 0 {
                        return Err(reject("invalid elementwise row extent"));
                    }
                    d.i[0] /= g.t;
                }
                Some(DevOp::Argmax | DevOp::ArgmaxFin) => {
                    if d.i[1].max(1) != g.t {
                        return Err(reject("invalid argmax row extent"));
                    }
                    d.i[1] = 1;
                }
                Some(
                    DevOp::RmsNorm
                    | DevOp::RowRms
                    | DevOp::HeadNormRope
                    | DevOp::Gemm
                    | DevOp::GemmNorm
                    | DevOp::Gemv
                    | DevOp::GemvQkv
                    | DevOp::GemvGlu
                    | DevOp::GemmGlu
                    | DevOp::NormResidual
                    | DevOp::NormResidualNorm
                    | DevOp::AddNorm
                    | DevOp::Embed
                    | DevOp::FlashDecode
                    | DevOp::FlashMerge,
                ) => {
                    if d.i[0] != g.t {
                        return Err(reject("instruction rows disagree with rung width"));
                    }
                    d.i[0] = 1;
                    if d.op == DevOp::HeadNormRope as u16 && d.i[6] != 0 {
                        if d.i[6] != g.t {
                            return Err(reject("KV writer uses a different slot count"));
                        }
                        d.i[6] = 1;
                    }
                    if d.op == DevOp::FlashDecode as u16 {
                        d.i[5] = 0;
                        d.fj[1] = 0;
                    } else if d.op == DevOp::FlashMerge as u16 {
                        d.i[2] = 0;
                    }
                }
                _ => {
                    compatible = false;
                }
            }
            insts.push(d);
        }
        if normalized.is_empty() {
            normalized = insts;
        } else if normalized != insts {
            same_shape = false;
        }
    }
    if !compatible {
        return Ok(false);
    }
    let tensor_id = |name: &str| blob.tensors.iter().position(|t| t.name == name);
    let pos = tensor_id("in.pos").ok_or_else(|| reject("missing position tensor"))?;
    let kvlen = tensor_id("in.kvlen").ok_or_else(|| reject("missing KV length tensor"))?;
    let ids = tensor_id("in.ids").ok_or_else(|| reject("missing token tensor"))?;
    if blob.tensors[kvlen].bytes != u64::from(widest.t) * 4
        || blob.tensors[pos].bytes < u64::from(widest.t) * 4
        || blob.tensors[ids].bytes < u64::from(widest.t) * 4
    {
        return Err(reject("runtime inputs do not cover physical slots"));
    }
    let mut caches = std::collections::BTreeMap::new();
    for d in &widest.insts {
        if d.op != DevOp::FlashDecode as u16 {
            continue;
        }
        if d.t[6] != TENSOR_NONE16 || d.t[7] != TENSOR_NONE16 || !matches!(d.i[6], 256 | 512) {
            return Ok(false);
        }
        let (heads, hd, stride, window, mask) = (d.i[2], d.i[6], d.i[3], d.i[4], d.i[7]);
        if heads == 0
            || stride == 0
            || d.i[1] == 0
            || d.i[1] % heads != 0
            || d.t[3] == d.t[4]
            || (window == 0
                && (mask != u32::MAX || u64::from(stride) * 4 != blob.tensors[pos].bytes))
            || (window != 0 && (!stride.is_power_of_two() || window > stride || mask != stride - 1))
        {
            return Err(reject("invalid KV reader geometry"));
        }
        let bytes = u64::from(widest.t)
            .checked_mul(u64::from(heads))
            .and_then(|n| n.checked_mul(u64::from(stride)))
            .and_then(|n| n.checked_mul(u64::from(hd) * 2))
            .ok_or_else(|| reject("KV extent overflow"))?;
        for &id in &d.t[3..5] {
            if blob
                .tensors
                .get(id as usize)
                .is_none_or(|t| t.bytes != bytes || t.init.is_some())
                || caches.insert(id, (heads, hd, stride, mask)).is_some()
                || [pos, kvlen, ids].contains(&(id as usize))
            {
                return Err(reject("invalid or aliased BF16 KV tensor extent"));
            }
        }
    }
    if caches.is_empty() {
        return Ok(false);
    }
    for g in programs {
        let mut writes = std::collections::BTreeSet::new();
        for (ix, d) in g.insts.iter().enumerate() {
            if d.op == DevOp::FlashDecode as u16 {
                if d.t[5] as usize != kvlen
                    || d.i[5] == 0
                    || d.t[3] == d.t[4]
                    || !caches.contains_key(&d.t[3])
                    || !caches.contains_key(&d.t[4])
                {
                    return Err(reject("invalid KV length handle or split count"));
                }
                let bound = u64::from(g.t)
                    .checked_mul(u64::from(d.i[2]))
                    .and_then(|n| n.checked_mul(u64::from(d.i[3])))
                    .ok_or_else(|| reject("KV bounds overflow"))?;
                let row_heads = u64::from(g.t) * u64::from(d.i[1]);
                let partials = row_heads
                    .checked_mul(u64::from(d.i[5]))
                    .filter(|&n| n > 0 && n <= u64::from(u32::MAX))
                    .ok_or_else(|| reject("invalid attention work extent"))?;
                let extent = |id: u16, elements: u64, bytes: u64| -> Result<()> {
                    let bytes = elements
                        .checked_mul(bytes)
                        .ok_or_else(|| reject("attention extent overflow"))?;
                    if blob
                        .tensors
                        .get(id as usize)
                        .is_none_or(|t| t.bytes < bytes)
                    {
                        return Err(reject("undersized attention tensor"));
                    }
                    Ok(())
                };
                extent(d.t[0], partials, u64::from(d.i[6]) * 4)?;
                extent(d.t[1], partials, 8)?;
                extent(d.t[2], row_heads, u64::from(d.i[6]) * 2)?;
                if d.fj[1] != 0 && u64::from(d.fj[1]) != bound {
                    return Err(reject("KV bounds disagree with physical slot geometry"));
                }
            }
            if d.op == DevOp::FlashMerge as u16 {
                let producer = g.insts[..ix]
                    .iter()
                    .rev()
                    .find(|a| {
                        a.op == DevOp::FlashDecode as u16 && a.t[0] == d.t[1] && a.t[1] == d.t[2]
                    })
                    .ok_or_else(|| reject("merge has no matching attention producer"))?;
                if [d.i[1], d.i[2], d.i[3]] != [producer.i[1], producer.i[5], producer.i[6]] {
                    return Err(reject("merge geometry disagrees with attention"));
                }
                let bytes = u64::from(g.t)
                    .checked_mul(u64::from(d.i[1]))
                    .and_then(|n| n.checked_mul(u64::from(d.i[3]) * 2))
                    .ok_or_else(|| reject("merge output extent overflow"))?;
                if blob
                    .tensors
                    .get(d.t[0] as usize)
                    .is_none_or(|t| t.bytes < bytes)
                {
                    return Err(reject("undersized merge output"));
                }
            }
            for (operand, &id) in d.t.iter().enumerate() {
                let Some(&(heads, hd, stride, mask)) = caches.get(&id) else {
                    continue;
                };
                match DevOp::from_u16(d.op) {
                    Some(DevOp::FlashDecode) if operand == 3 || operand == 4 => {
                        if (d.i[2], d.i[6], d.i[3], d.i[7]) != (heads, hd, stride, mask) {
                            return Err(reject("KV reader addressing changes across rungs"));
                        }
                    }
                    Some(DevOp::HeadNormRope) if operand == 0 => {
                        if d.i[6] != g.t
                            || d.i[3] != 0
                            || d.t[5] as usize != pos
                            || (d.i[1], d.i[2], d.fj[1], d.fj[2]) != (heads, hd, stride, mask)
                        {
                            return Err(reject(
                                "KV writer requires physical-slot position addressing",
                            ));
                        }
                        if d.t[6] != TENSOR_NONE16 || d.t[7] != TENSOR_NONE16 {
                            return Ok(false);
                        }
                        if !writes.insert(id) {
                            return Err(reject("duplicate KV writer"));
                        }
                    }
                    _ => return Ok(false),
                }
            }
        }
        if writes.len() != caches.len() {
            return Err(reject("missing direct KV writer"));
        }
    }
    Ok(same_shape)
}

impl DecodeRung {
    pub(super) fn upload(
        be: &CudaBackend,
        g: &crate::asset::devblob::DevProg,
        base: DevProgram,
    ) -> Result<Self> {
        let upload = |bytes: &[u8]| -> Result<DeviceMem> {
            let mem = be.alloc(0, bytes.len().max(4) as u64)?;
            if !bytes.is_empty() {
                be.upload(&mem, 0, bytes)?;
            }
            Ok(mem)
        };
        let tables = vec![
            upload(pod_bytes(&g.insts))?,
            upload(pod_bytes(&g.stream))?,
            upload(pod_bytes(&g.stream_ofs))?,
            upload(pod_bytes(&g.stream_len))?,
            upload(pod_bytes(&g.waits))?,
            upload(pod_bytes(&g.succs))?,
            upload(pod_bytes(&g.gq_stream))?,
            upload(pod_bytes(&g.gq_seg_ofs))?,
        ];
        let cursor_offset = (g.n_counter as usize * CTR_STRIDE as usize * 4).max(4);
        let counter_bytes = cursor_offset + CTR_STRIDE as usize * 4;
        let counters = be.alloc(0, counter_bytes as u64)?;
        let kernarg = DevProgram {
            insts: tables[0].base,
            stream: tables[1].base,
            stream_ofs: tables[2].base,
            stream_len: tables[3].base,
            waits: tables[4].base,
            succs: tables[5].base,
            gq_stream: tables[6].base,
            gq_seg_ofs: tables[7].base,
            counters: counters.base,
            gq_cursor: counters.base + cursor_offset as u64,
            ..base
        };
        Ok(Self {
            rows: g.t as usize,
            object: None,
            kernarg,
            counters,
            counter_bytes,
            _tables: tables,
        })
    }
}

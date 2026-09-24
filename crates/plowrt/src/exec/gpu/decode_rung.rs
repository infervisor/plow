use super::*;
use packet::dev::ROPE_PAIR_HALF;

pub(super) struct DecodeRung {
    pub(super) host_insts: Vec<DevInst64>,
    pub(super) library: Option<super::cublaslt::CublasLtDecodeGraph>,
    pub(super) rows: usize,
    /// This rung runs the grouped-MoE arm (align rows >= its threshold), so its launch needs the
    /// object's full arena; the other rungs launch with `plow_arena_bytes_narrow`.
    pub(super) group_arena: bool,
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
    validate_decode_ladder_impl(blob, false)
}

pub(super) fn validate_cublaslt_ladder(blob: &DevBlob, metadata: &SegmentRoles) -> Result<bool> {
    if blob.decode_progs().len() < 2 {
        return Ok(false);
    }
    let start = blob.progs.len() - blob.decode_progs().len();
    let mut previous_roles = None;
    for (index, program) in blob.progs.iter().enumerate().skip(start) {
        let roles = &metadata
            .program(index)
            .ok_or_else(|| {
                RuntimeError::Rejected(
                    "cuBLASLt ladder requires roles for every decode width".into(),
                )
            })?
            .roles;
        if !roles
            .iter()
            .copied()
            .any(plow_asset::segment_roles::is_projection)
            || previous_roles.is_some_and(|previous| previous != roles)
        {
            return Err(RuntimeError::Rejected(
                "cuBLASLt ladder projection roles differ".into(),
            ));
        }
        packet_role_segments(program, roles, &blob.tensors)?;
        cublaslt::decode_segments(program, &blob.tensors, roles)?;
        previous_roles = Some(roles);
    }
    blob.with_packet_view(|packet| {
        let mut previous = None;
        for program in &packet.programs[start..] {
            let mut stream = program.stream.to_vec();
            let mut queue = program.gq_stream.to_vec();
            for entry in stream.iter_mut().chain(&mut queue) {
                entry.seg = 0;
            }
            let window = [0, queue.len() as u32];
            let normalized = plow_asset::program::Program {
                stream: &stream,
                gq_stream: &queue,
                gq_seg_ofs: &window,
                ..*program
            };
            let dependencies = plow_asset::splitk::dependencies(&normalized)?;
            if previous.as_ref().is_some_and(|old| old != &dependencies) {
                return Err("cuBLASLt ladder dependencies differ".to_string());
            }
            previous = Some(dependencies);
        }
        Ok(())
    })
    .map_err(RuntimeError::Rejected)?;
    validate_decode_ladder_impl(blob, true)
}

/// A ladder whose grouped-arm rungs declare `MOE_DECODE_CUBLASLT` segments. Every rung is
/// served on its own (routed, or run as one merged window), so unlike the dense library ladder
/// the rungs' dependencies may differ: a rung below the grouped threshold does not wait on the
/// align op.
pub(super) fn validate_moe_lt_ladder(blob: &DevBlob, metadata: &SegmentRoles) -> Result<bool> {
    let start = blob.progs.len() - blob.decode_progs().len();
    for (index, program) in blob.progs.iter().enumerate().skip(start) {
        match metadata.program(index) {
            Some(roles) => {
                moe_lt::decode_segments(program, &blob.tensors, &roles.roles)?;
            }
            None => program.check_coarse_single_segment()?,
        }
    }
    validate_decode_ladder_impl(blob, true)
}

fn validate_decode_ladder_impl(blob: &DevBlob, segmented: bool) -> Result<bool> {
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
    for g in &programs {
        if g.role.is_packed_sibling()
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
        g.l2_domains != 0
            || (!segmented && (g.gq_seg_ofs.len() != 2 || g.check_coarse_single_segment().is_err()))
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
                Some(DevOp::MoeRouterTopkPf) => {
                    if d.i[4] != g.t {
                        return Err(reject("MoE router rows disagree with rung width"));
                    }
                    d.i[4] = 1;
                }
                Some(DevOp::MoeGluMx | DevOp::MoeDownMx) => {
                    if d.i[6] != g.t {
                        return Err(reject("MoE expert rows disagree with rung width"));
                    }
                    d.i[6] = 1;
                }
                Some(DevOp::MoeCombinePf) => {
                    if d.i[2] != g.t {
                        return Err(reject("MoE combine rows disagree with rung width"));
                    }
                    d.i[2] = 1;
                }
                // Grouped-decode align (PLOW_GEMMA_MOE_DEC_GROUP): rows ride i0, like prefill.
                Some(DevOp::MoeAlignGemmaPf) => {
                    if d.i[0] != g.t {
                        return Err(reject("MoE align rows disagree with rung width"));
                    }
                    d.i[0] = 1;
                }
                // Gemma MoE decode carries B in a spare immediate, 0 at B=1 (devgen `nb`).
                Some(
                    op @ (DevOp::MoeRouterGemmaScore
                    | DevOp::MoeRouterGemmaScoreFast
                    | DevOp::MoeRouterGemmaTopk
                    | DevOp::MoeExpertGluNormGemma
                    | DevOp::MoeExpertDownGemma
                    | DevOp::MoeCombineNormGemma
                    | DevOp::MoeCombineResidNormGemma),
                ) => {
                    let field = match op {
                        DevOp::MoeRouterGemmaScore
                        | DevOp::MoeRouterGemmaScoreFast
                        | DevOp::MoeCombineNormGemma
                        // op72, the fused combine+NRN tail: carries B in i[2] exactly as the
                        // unfused op70 it replaces, so it normalizes the same way. Without this
                        // arm the ladder silently falls back to widest-only execution.
                        | DevOp::MoeCombineResidNormGemma => 2,
                        DevOp::MoeRouterGemmaTopk => 3,
                        _ => 5,
                    };
                    if d.i[field] != if g.t > 1 { g.t } else { 0 } {
                        return Err(reject("Gemma MoE rows disagree with rung width"));
                    }
                    d.i[field] = 0;
                }
                Some(
                    DevOp::RmsNorm
                    | DevOp::RowRms
                    | DevOp::HeadNormRope
                    | DevOp::HeadNormRopeFp8
                    | DevOp::Gemm
                    | DevOp::GemmNorm
                    | DevOp::Gemv
                    | DevOp::GemvArgmax
                    | DevOp::GemvFp8
                    | DevOp::GemvQkv
                    | DevOp::GemvGlu
                    | DevOp::GemvGluFp8
                    | DevOp::GemmGlu
                    | DevOp::NormResidual
                    | DevOp::NormResidualNorm
                    | DevOp::AddNorm
                    | DevOp::Embed
                    | DevOp::FlashDecode
                    | DevOp::FlashDecodeFp8
                    | DevOp::FlashMerge,
                ) => {
                    if d.i[0] != g.t {
                        return Err(reject("instruction rows disagree with rung width"));
                    }
                    d.i[0] = 1;
                    if matches!(
                        DevOp::from_u16(d.op),
                        Some(DevOp::HeadNormRope | DevOp::HeadNormRopeFp8)
                    ) && d.i[6] != 0
                    {
                        if d.i[6] != g.t {
                            return Err(reject("KV writer uses a different slot count"));
                        }
                        d.i[6] = 1;
                    }
                    if matches!(
                        DevOp::from_u16(d.op),
                        Some(DevOp::FlashDecode | DevOp::FlashDecodeFp8)
                    ) {
                        d.i[5] = 0;
                        d.fj[1] = 0;
                    } else if d.op == DevOp::FlashMerge as u16 {
                        d.i[2] = 0;
                    }
                }
                _ => {
                    if crate::config::RuntimeConfig::get().nv.ladder_debug {
                        eprintln!(
                            "ladder: rung {index} inst {} op {} has no normalization arm",
                            insts.len(),
                            d.op
                        );
                    }
                    compatible = false;
                }
            }
            insts.push(d);
        }
        if normalized.is_empty() {
            normalized = insts;
        } else if normalized != insts {
            // Falling back here is SILENT (Ok(false) -> the widest rung runs every step), and it
            // reads exactly like a large kernel regression. Name the first difference.
            if crate::config::RuntimeConfig::get().nv.ladder_debug {
                if normalized.len() != insts.len() {
                    eprintln!(
                        "ladder: rung {index} has {} insts, rung 0 has {}",
                        insts.len(),
                        normalized.len()
                    );
                } else if let Some((k, (a, b))) = normalized
                    .iter()
                    .zip(insts.iter())
                    .enumerate()
                    .find(|(_, (a, b))| a != b)
                {
                    eprintln!(
                        "ladder: rung {index} inst {k} differs after normalization\n                           rung0 op={} blocks={} i={:?} t={:?}\n                           rung{index} op={} blocks={} i={:?} t={:?}",
                        a.op, a.blocks, a.i, a.t, b.op, b.blocks, b.i, b.t
                    );
                }
            }
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
    if blob.tensors[kvlen].bytes < u64::from(widest.t) * 4
        || blob.tensors[pos].bytes < u64::from(widest.t) * 4
        || blob.tensors[ids].bytes < u64::from(widest.t) * 4
    {
        return Err(reject("runtime inputs do not cover physical slots"));
    }
    let mut caches = std::collections::BTreeMap::new();
    let mut scales = std::collections::BTreeMap::new();
    for d in &widest.insts {
        if !matches!(
            DevOp::from_u16(d.op),
            Some(DevOp::FlashDecode | DevOp::FlashDecodeFp8)
        ) {
            continue;
        }
        let fp8 = d.op == DevOp::FlashDecodeFp8 as u16;
        if !matches!(d.i[6], 64 | 256 | 512)
            || (fp8 && d.i[6] == 64)
            || (!fp8 && (d.t[6] != TENSOR_NONE16 || d.t[7] != TENSOR_NONE16))
        {
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
            .and_then(|n| n.checked_mul(u64::from(hd) * if fp8 { 1 } else { 2 }))
            .ok_or_else(|| reject("KV extent overflow"))?;
        for (kind, &id) in d.t[3..5].iter().enumerate() {
            let scale = fp8.then_some(d.t[6 + kind]);
            if let Some(scale) = scale {
                let scale_bytes = u64::from(widest.t) * u64::from(heads) * u64::from(stride) * 4;
                if blob
                    .tensors
                    .get(scale as usize)
                    .is_none_or(|t| t.bytes != scale_bytes || t.init.is_some())
                    || [pos, kvlen, ids].contains(&(scale as usize))
                    || scales.insert(scale, id).is_some()
                {
                    return Err(reject("invalid or aliased FP8 KV scale tensor"));
                }
            }
            let pair_mode = if hd == 64 && kind == 0 {
                ROPE_PAIR_HALF
            } else {
                0
            };
            if blob
                .tensors
                .get(id as usize)
                .is_none_or(|t| t.bytes != bytes || t.init.is_some())
                || caches
                    .insert(id, (heads, hd, stride, mask, pair_mode, scale))
                    .is_some()
                || [pos, kvlen, ids].contains(&(id as usize))
            {
                return Err(reject("invalid or aliased KV tensor extent"));
            }
        }
    }
    if scales.keys().any(|id| caches.contains_key(id)) {
        return Err(reject("FP8 KV scale aliases cache data"));
    }
    if caches.is_empty() {
        return Ok(false);
    }
    for g in programs {
        let mut writes = std::collections::BTreeSet::new();
        for (ix, d) in g.insts.iter().enumerate() {
            if matches!(
                DevOp::from_u16(d.op),
                Some(DevOp::FlashDecode | DevOp::FlashDecodeFp8)
            ) {
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
                        matches!(
                            DevOp::from_u16(a.op),
                            Some(DevOp::FlashDecode | DevOp::FlashDecodeFp8)
                        ) && a.t[0] == d.t[1]
                            && a.t[1] == d.t[2]
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
                if let Some(&cache) = scales.get(&id) {
                    let valid = match DevOp::from_u16(d.op) {
                        Some(DevOp::HeadNormRopeFp8) => operand == 6 && d.t[0] == cache,
                        Some(DevOp::FlashDecodeFp8) => {
                            (operand == 6 || operand == 7) && d.t[operand - 3] == cache
                        }
                        _ => false,
                    };
                    if !valid {
                        return Err(reject("FP8 KV scale has an incompatible reader or writer"));
                    }
                }
                let Some(&(heads, hd, stride, mask, pair_mode, scale)) = caches.get(&id) else {
                    continue;
                };
                match DevOp::from_u16(d.op) {
                    Some(DevOp::FlashDecode | DevOp::FlashDecodeFp8)
                        if operand == 3 || operand == 4 =>
                    {
                        if (d.op == DevOp::FlashDecodeFp8 as u16) != scale.is_some()
                            || scale.is_some_and(|scale| d.t[operand + 3] != scale)
                        {
                            return Err(reject("KV reader encoding or scale changes across rungs"));
                        }
                        if (d.i[2], d.i[6], d.i[3], d.i[7]) != (heads, hd, stride, mask) {
                            return Err(reject("KV reader addressing changes across rungs"));
                        }
                    }
                    Some(DevOp::HeadNormRope | DevOp::HeadNormRopeFp8) if operand == 0 => {
                        if (d.op == DevOp::HeadNormRopeFp8 as u16) != scale.is_some()
                            || scale.is_some_and(|scale| d.t[6] != scale || d.t[7] != TENSOR_NONE16)
                        {
                            return Err(reject("KV writer encoding or scale changes across rungs"));
                        }
                        if d.i[6] != g.t
                            || d.i[3] != 0
                            || d.t[5] as usize != pos
                            || d.i[5] != pair_mode
                            || (d.i[1], d.i[2], d.fj[1], d.fj[2]) != (heads, hd, stride, mask)
                        {
                            return Err(reject(
                                "KV writer requires physical-slot position addressing",
                            ));
                        }
                        // FUSED KV PAIR (devgen `fuse_kv`, i[7]=1): ONE HeadNormRope writes
                        // both halves of the pair, so t6/t7 carry v's (cache, source) rather
                        // than being empty. bf16 only -- under fp8 t6 is the scale handle,
                        // which the `scale.is_some()` branch above already pins.
                        if scale.is_none()
                            && d.i[7] != 1
                            && (d.t[6] != TENSOR_NONE16 || d.t[7] != TENSOR_NONE16)
                        {
                            return Ok(false);
                        }
                        if !writes.insert(id) {
                            return Err(reject("duplicate KV writer"));
                        }
                    }
                    // The v half of a fused pair. Its addressing immediates are SHARED with the
                    // k half at operand 0, which was checked above; what has to hold here is
                    // that this cache agrees with them, so one instruction cannot silently
                    // write two caches of different geometry.
                    Some(DevOp::HeadNormRope) if operand == 6 && d.i[7] == 1 => {
                        // hd==64 gives the two halves DIFFERENT pair modes (ROPE_PAIR_HALF on k,
                        // 0 on v) and d.i[5] can only carry one: leave such a packet unqualified
                        // rather than validate it under the k half's mode.
                        if scale.is_some() || pair_mode != 0 {
                            return Ok(false);
                        }
                        if (d.i[1], d.i[2], d.fj[1], d.fj[2]) != (heads, hd, stride, mask) {
                            return Err(reject("fused KV pair writers disagree on addressing"));
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
        Self::upload_with_insts(be, g, base, &g.insts, &g.waits, &g.gq_seg_ofs)
    }

    /// `gq_seg_ofs` is the queue window set this rung launches with: the program's own, or the
    /// merged `[0, len]` window that runs a segmented program as one launch.
    pub(super) fn upload_with_insts(
        be: &CudaBackend,
        g: &crate::asset::devblob::DevProg,
        base: DevProgram,
        insts: &[DevInst64],
        waits: &[packet::dev::Wait],
        gq_seg_ofs: &[u32],
    ) -> Result<Self> {
        let upload = |bytes: &[u8]| -> Result<DeviceMem> {
            let mem = be.alloc(0, bytes.len().max(4) as u64)?;
            if !bytes.is_empty() {
                be.upload(&mem, 0, bytes)?;
            }
            Ok(mem)
        };
        let tables = vec![
            upload(pod_bytes(insts))?,
            upload(pod_bytes(&g.stream))?,
            upload(pod_bytes(&g.stream_ofs))?,
            upload(pod_bytes(&g.stream_len))?,
            upload(pod_bytes(waits))?,
            upload(pod_bytes(&g.succs))?,
            upload(pod_bytes(&g.gq_stream))?,
            upload(pod_bytes(gq_seg_ofs))?,
        ];
        let cursor_offset = (g.n_counter as usize * CTR_STRIDE as usize * 4).max(4);
        let cursor_bytes = gq_seg_ofs.len().saturating_sub(1).max(1) * CTR_STRIDE as usize * 4;
        let counter_bytes = cursor_offset + cursor_bytes;
        let (counters, [counter_view, cursor_view]) =
            slab_carve(be, [cursor_offset, cursor_bytes])?;
        let kernarg = DevProgram {
            insts: tables[0].base,
            stream: tables[1].base,
            stream_ofs: tables[2].base,
            stream_len: tables[3].base,
            waits: tables[4].base,
            succs: tables[5].base,
            gq_stream: tables[6].base,
            gq_seg_ofs: tables[7].base,
            counters: counter_view.base,
            gq_cursor: cursor_view.base,
            ..base
        };
        Ok(Self {
            host_insts: insts.to_vec(),
            library: None,
            rows: g.t as usize,
            group_arena: insts.iter().any(|d| {
                DevOp::from_u16(d.op) == Some(DevOp::MoeAlignGemmaPf) && d.i[3] != 0 && d.i[0] >= d.i[3]
            }),
            object: None,
            kernarg,
            counters,
            counter_bytes,
            _tables: tables,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// CPU-only reproduction of the ladder decision. The real check runs during a GPU load, so
    /// diagnosing a silent widest-fallback otherwise costs a lease. Point `PLOW_RUNG_PKT` at a
    /// model.pkt and run with `PLOW_LADDER_DEBUG=1 --nocapture`; a no-op when unset.
    #[test]
    fn report_ladder_decision() {
        let Some(path) = std::env::var_os("PLOW_RUNG_PKT") else {
            return;
        };
        let buf = std::fs::read(&path).expect("read blob");
        let blob = DevBlob::parse(&buf).expect("parse devblob");
        eprintln!("decode rungs: {:?}", blob.decode_rungs());
        eprintln!("segmented=false -> {:?}", validate_decode_ladder_impl(&blob, false));
        eprintln!("segmented=true  -> {:?}", validate_decode_ladder_impl(&blob, true));
        // Which of the three validators gpu.rs picks is decided here, and picking the
        // non-segmented one makes a perfectly good ladder read as unqualified.
        let prefill = blob.prefill_progs().len();
        match segment_role_metadata(&blob, &buf) {
            Ok(Some(roles)) => {
                for p in &roles.programs {
                    if p.index >= prefill {
                        eprintln!("  prog {} roles {:?}", p.index, p.roles);
                    }
                }
                let moe = roles.programs.iter().any(|p| {
                    p.index >= prefill
                        && p.roles
                            .contains(&plow_asset::segment_roles::MOE_DECODE_CUBLASLT)
                });
                eprintln!("moe_lt_decode_roles = {moe}");
                // The three branches gpu.rs chooses between. Whichever one the runtime takes
                // is what decides the ladder; calling all three here localizes a silent
                // fallback without a GPU lease.
                eprintln!("validate_moe_lt_ladder   -> {:?}", validate_moe_lt_ladder(&blob, &roles));
                eprintln!("validate_cublaslt_ladder -> {:?}", validate_cublaslt_ladder(&blob, &roles));
                eprintln!("validate_decode_ladder   -> {:?}", validate_decode_ladder(&blob));
            }
            Ok(None) => eprintln!("NO segment role metadata -> falls to validate_decode_ladder"),
            Err(e) => eprintln!("segment role metadata error: {e:?}"),
        }
    }
}

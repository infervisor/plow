//! Generated-kernel catalog: role objects AOT-built by `scripts/gen_kernels/build_catalog.py`
//! from its checked-in tuning table (`tuning/nvidia/sm_90a/h100-sxm5/gen_kernels.json`), keyed
//! by op signature, never by model. `PLOW_EMIT_GEN_KERNELS=<entry,...>` binds every packet op
//! matching an entry's signature to that entry's role; without it the packet is unchanged.

use crate::attention_prefill_role::{self, Selection};
use packet::dev::{DevInst, DevOp, TENSOR_NONE};
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use plow_asset::segment_roles::{
    AttentionCapability, GeneratedAbi, ProgramRoles, SegmentObject, SegmentRoles, GENERATED_FIRST,
    GENERATED_FLASH_DECODE_ABI, GENERATED_FLASH_PREFILL_ABI, GENERATED_FLASH_PREFILL_FP8KV_ABI,
    INTERPRETER, SECTION,
};
use std::collections::BTreeSet;
use std::path::Path;

/// KV cache encoding an entry reads; part of the op signature (the packet op differs:
/// `FlashPrefill` vs `FlashPrefillFp8`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvDtype {
    Bf16,
    /// OCP e4m3 with one f32 scale per (position, KV head) row.
    Fp8,
}

impl KvDtype {
    pub(crate) fn op(self) -> DevOp {
        match self {
            KvDtype::Bf16 => DevOp::FlashPrefill,
            KvDtype::Fp8 => DevOp::FlashPrefillFp8,
        }
    }
    fn abi_family(self) -> &'static str {
        match self {
            KvDtype::Bf16 => GENERATED_FLASH_PREFILL_ABI,
            KvDtype::Fp8 => GENERATED_FLASH_PREFILL_FP8KV_ABI,
        }
    }
    /// The object's `plow_gen_flash_prefill_abi`.
    fn object_abi(self) -> u32 {
        match self {
            KvDtype::Bf16 => 1,
            KvDtype::Fp8 => 2,
        }
    }
}

pub(crate) struct Entry {
    pub name: String,
    pub role: u8,
    pub file: String,
    pub kv: KvDtype,
    /// Causal flash prefill over one head width.
    pub head_dim: u32,
    /// Sliding window (`FlashPrefill` i[5]); 0 = global attention, `ANY_SLIDING` = any nonzero
    /// window (a runtime operand of the object).
    pub window: u32,
    /// Accepts a KV ring (masked row index); otherwise the op must index KV linearly.
    pub ring_kv: bool,
    /// The object runs query heads in pairs per KV head: the GQA ratio must be even.
    pub pair_heads: bool,
    /// Smallest prefill rung (rows) bound to the entry; narrower rungs keep their existing route.
    pub min_rows: u32,
    /// Claims exactly the media-span sites (`FlashPrefillFp8` i[7] high half); the object exports
    /// `plow_attention_media_span = 1`.
    pub media_span: bool,
    /// A flash-decode object (`FlashDecode`, one query row per sequence): the runtime launches
    /// it itself, so the packet only carries the object (no segment binds it).
    pub decode: bool,
    /// Decode entries: the GQA ratio range the body serves (compiled in: `gqa[0] == gqa[1]`).
    pub gqa: [u32; 2],
    /// Decode entries: the widest split-KV launch the object merges in one pass.
    pub nsplit_max: u32,
    /// KV row layout the object addresses (0 head-major, 1 head-interleaved): it binds only when
    /// the packet's layout (PLOW_QWEN_KV_INTERLEAVE) is the same.
    pub kv_layout: u32,
}

/// The table build_catalog.py builds from and `tune` writes: the one source of every entry's
/// signature and object name.
const TABLE: &str = include_str!("../../../tuning/nvidia/sm_90a/h100-sxm5/gen_kernels.json");

/// Devgen's binding policy per table entry, which the table does not carry: the role ID (a new
/// entry takes the next generated ID) and the smallest prefill rung it binds.
const POLICY: [(&str, u8, u32); 9] = [
    ("attn_pf_hd512", GENERATED_FIRST, 1024),
    ("attn_pf_hd256_sliding", GENERATED_FIRST + 1, 1024),
    ("attn_pf_hd256_sliding_fp8kv", GENERATED_FIRST + 2, 128),
    ("attn_pf_hd512_fp8kv", GENERATED_FIRST + 3, 128),
    ("attn_pf_hd256_sliding_fp8kv_span", GENERATED_FIRST + 4, 128),
    // Window 0 over a linear KV (Qwen3.5 full-attention layers). Every bucket: the interpreter's
    // hd256 body runs at ~8% of peak in that packet (pf object at the 255-register cap).
    ("attn_pf_hd256_causal", GENERATED_FIRST + 5, 128),
    // hd256 GQA6 decode attention (the same layers): the runtime launches it for riders and
    // routed decode rungs (interpreter FlashDecode 3.8x the HBM floor at B16 x 8K; this 1.2x).
    ("attn_dec_hd256_gqa6", GENERATED_FIRST + 6, 0),
    // The same two objects over head-interleaved KV rows (PLOW_QWEN_KV_INTERLEAVE): a packet binds
    // one layout, so each twin shares its counterpart's role ID.
    ("attn_pf_hd256_causal_ilv", GENERATED_FIRST + 5, 128),
    ("attn_dec_hd256_gqa6_ilv", GENERATED_FIRST + 6, 0),
];

fn table_entry(
    name: &str,
    role: u8,
    min_rows: u32,
    row: &serde_json::Value,
) -> Result<Entry, String> {
    let sig = &row["signature"];
    let text = |key: &str| sig[key].as_str().unwrap_or_default();
    let flag = |key: &str| sig[key].as_bool().unwrap_or(false);
    let number = |key: &str| {
        sig[key]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or(format!("{name}: signature lacks {key}"))
    };
    let decode = text("op") == "flash_decode";
    let media_span = match (decode, text("mask")) {
        (true, "") | (false, "causal") => false,
        (false, "causal_media_span") => true,
        (_, other) => return Err(format!("{name}: unknown mask {other:?}")),
    };
    if !matches!(text("op"), "flash_prefill" | "flash_decode")
        || (text("dtype"), text("arch")) != ("bf16", "sm_90a")
    {
        return Err(format!("{name}: unsupported signature {sig}"));
    }
    let kv = match text("kv_dtype") {
        "bf16" => KvDtype::Bf16,
        "fp8_e4m3_rowscale" => KvDtype::Fp8,
        other => return Err(format!("{name}: unknown kv_dtype {other:?}")),
    };
    let kv_layout = match text("kv_layout") {
        "" | "head_major" => 0,
        "row_interleaved" => 1,
        other => return Err(format!("{name}: unknown kv_layout {other:?}")),
    };
    let gqa = if decode {
        let gqa = sig["gqa"]
            .as_array()
            .filter(|a| a.len() == 2)
            .map(|a| {
                a.iter()
                    .map(|v| v.as_u64().unwrap_or(0) as u32)
                    .collect::<Vec<_>>()
            })
            .ok_or(format!("{name}: signature lacks gqa"))?;
        if gqa[0] == 0 || gqa[0] != gqa[1] || kv != KvDtype::Bf16 {
            return Err(format!("{name}: decode entries compile one bf16 GQA ratio"));
        }
        [gqa[0], gqa[1]]
    } else {
        [0, 0]
    };
    Ok(Entry {
        name: name.into(),
        role,
        file: row["object"]
            .as_str()
            .ok_or(format!("{name}: no object"))?
            .into(),
        kv,
        head_dim: number("head_dim")?,
        window: if flag("window_any") {
            ANY_SLIDING
        } else {
            number("window")?
        },
        ring_kv: flag("ring_kv"),
        pair_heads: flag("gqa_even"),
        min_rows,
        media_span,
        decode,
        gqa,
        nsplit_max: if decode { number("nsplit_max")? } else { 0 },
        kv_layout,
    })
}

fn load_catalog(table: &str) -> Result<Vec<Entry>, String> {
    let table: serde_json::Value =
        serde_json::from_str(table).map_err(|e| format!("gen_kernels.json: {e}"))?;
    let rows = table["entries"]
        .as_object()
        .ok_or("gen_kernels.json: no entries")?;
    let unbound: Vec<_> = rows
        .keys()
        .filter(|n| !POLICY.iter().any(|(p, ..)| p == n))
        .collect();
    if !unbound.is_empty() {
        return Err(format!(
            "gen_kernels.json entries without a devgen role: {unbound:?}"
        ));
    }
    POLICY
        .iter()
        .map(|&(name, role, min_rows)| {
            let row = rows
                .get(name)
                .ok_or(format!("{name}: not in gen_kernels.json"))?;
            table_entry(name, role, min_rows, row)
        })
        .collect()
}

pub(crate) static CATALOG: std::sync::LazyLock<Vec<Entry>> = std::sync::LazyLock::new(|| {
    load_catalog(TABLE).unwrap_or_else(|e| panic!("generated-kernel catalog: {e}"))
});

pub(crate) const ANY_SLIDING: u32 = u32::MAX;

/// Whether an op's window (`FlashPrefill` i[5]) and heads (i[2] / i[3]) fit an entry's.
pub(crate) fn heads_and_window_match(op: &DevInst, window: u32, pair_heads: bool) -> bool {
    (if window == ANY_SLIDING {
        op.i[5] > 0
    } else {
        op.i[5] == window
    }) && (!pair_heads || (op.i[3] > 0 && op.i[2] % (2 * op.i[3]) == 0))
}

/// The packet entry; plowrt also requires the `_direct` entry it launches with packed requests.
pub(crate) const ENTRY_SYMBOL: &str = "plow_gen_flash_prefill";
/// The decode family's one entry (runtime-launched, `gen_flash_decode.cu`).
pub(crate) const DECODE_ENTRY_SYMBOL: &str = "plow_gen_flash_decode";

/// Record a runtime-launched object in the packet's segment-role metadata (no segment binds
/// it). The section must already exist with role programs: a packet gets its first roles from
/// a segment-bound object.
fn record_object(
    sections: &mut [SectionData],
    role: u8,
    object: SegmentObject,
) -> Result<(), String> {
    let index = sections
        .iter()
        .position(|section| section.name == SECTION && section.kind == SECT_METADATA)
        .ok_or("a runtime-launched generated object needs a packet with segment roles")?;
    let mut metadata = SegmentRoles::from_bytes(&sections[index].data)?;
    if metadata.objects.contains_key(&role) {
        return Err("generated decode role already declared".into());
    }
    metadata.objects.insert(role, object);
    metadata.validate_schema()?;
    sections[index].data = serde_json::to_vec(&metadata).map_err(|error| error.to_string())?;
    Ok(())
}

/// The segment holding instruction `ix` alone (every entry of that segment is `ix`), if any.
pub(crate) fn lone_segment(program: &packet::devbuild::Program, ix: usize) -> Option<usize> {
    let seg = program
        .stream
        .iter()
        .chain(&program.gq_stream)
        .find(|e| e.inst as usize == ix)?
        .seg;
    program
        .stream
        .iter()
        .chain(&program.gq_stream)
        .all(|e| (e.inst as usize == ix) == (e.seg == seg))
        .then_some(seg as usize)
}

/// Bind a decode entry: on every decode program whose matching `FlashDecode` and its
/// `FlashMerge` are isolated single-instruction segments, the role is recorded on both
/// segments (the runtime replaces the pair with one launch of the object, or runs the
/// untouched interpreter pair). Programs without such a pair leave the object runtime-launched
/// only (riders).
fn bind_decode(
    model: &Model,
    sections: &mut Vec<SectionData>,
    entry: &Entry,
    object: SegmentObject,
    prefill_count: usize,
) -> Result<(), String> {
    let mut matched = false;
    let mut uses = Vec::new();
    for index in prefill_count..model.progs.len() {
        let program = &model.progs[index];
        let mut roles = vec![INTERPRETER; program.gq_seg_ofs.len().saturating_sub(1)];
        let mut bound = false;
        for (ix, op) in program.insts.iter().enumerate() {
            if !entry.matches(op) {
                continue;
            }
            matched = true;
            // The merge that follows this decode: the layers reuse the partial tensors, so a
            // whole-program search found layer 0's merge for every layer (one merge role, the
            // runtime rejecting the unpaired decodes).
            let merge = program
                .insts
                .iter()
                .enumerate()
                .skip(ix + 1)
                .find(|(_, m)| {
                    m.op == DevOp::FlashMerge as u16 && m.t[1] == op.t[0] && m.t[2] == op.t[1]
                })
                .map(|(m, _)| m);
            let (Some(seg), Some(merge_seg)) = (
                lone_segment(program, ix),
                merge.and_then(|m| lone_segment(program, m)),
            ) else {
                continue;
            };
            roles[seg] = entry.role;
            roles[merge_seg] = entry.role;
            bound = true;
        }
        if bound {
            uses.push(ProgramRoles {
                index,
                roles,
                library: Vec::new(),
            });
        }
    }
    if !matched {
        return Err(format!(
            "packet has no decode op matching generated entry {}",
            entry.name
        ));
    }
    if uses.is_empty() {
        return record_object(sections, entry.role, object);
    }
    let position = sections
        .iter()
        .position(|section| section.name == SECTION && section.kind == SECT_METADATA);
    let mut metadata = match position {
        Some(index) => SegmentRoles::from_bytes(&sections[index].data)?,
        None => SegmentRoles {
            version: 1,
            objects: Default::default(),
            programs: Vec::new(),
        },
    };
    if metadata.objects.contains_key(&entry.role) {
        return Err("generated decode role already declared".into());
    }
    metadata.objects.insert(entry.role, object);
    for record in uses {
        match metadata
            .programs
            .iter_mut()
            .find(|p| p.index == record.index)
        {
            Some(prior) => {
                if prior.roles.len() != record.roles.len() {
                    return Err("generated decode role: segment count differs".into());
                }
                for (have, &want) in prior.roles.iter_mut().zip(&record.roles) {
                    if want == INTERPRETER {
                        continue;
                    }
                    if *have != INTERPRETER {
                        return Err("generated decode role overlaps another role".into());
                    }
                    *have = want;
                }
            }
            None => metadata.programs.push(record),
        }
    }
    metadata.validate_schema()?;
    let data = serde_json::to_vec(&metadata).map_err(|error| error.to_string())?;
    match position {
        Some(index) => sections[index].data = data,
        None => sections.push(SectionData {
            kind: SECT_METADATA,
            name: SECTION.into(),
            data,
        }),
    }
    Ok(())
}

pub(crate) fn parse(list: &str) -> Result<Vec<&'static Entry>, String> {
    let mut out: Vec<&'static Entry> = Vec::new();
    for name in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let entry = CATALOG.iter().find(|e| e.name == name).ok_or_else(|| {
            let known: Vec<_> = CATALOG.iter().map(|e| e.name.as_str()).collect();
            format!(
                "unknown generated-kernel entry {name:?} (catalog: {})",
                known.join(", ")
            )
        })?;
        if !out.iter().any(|e| e.name == name) {
            out.push(entry);
        }
    }
    Ok(out)
}

impl Entry {
    /// The packet op signature: causal one-split fused flash prefill at this head width and
    /// window. Heads, KV heads and the ring size are runtime operands. A decode entry matches
    /// `FlashDecode` at its head width and GQA ratio (window i[4], ring mask i[7] are runtime
    /// operands of the object).
    pub(crate) fn matches(&self, op: &DevInst) -> bool {
        if self.kv_layout != crate::emit_config::active().qwen_kv_interleave as u32 {
            return false;
        }
        if self.decode {
            return op.op == DevOp::FlashDecode as u16
                && op.i[6] == self.head_dim
                && op.i[2] > 0
                && op.i[1] % op.i[2] == 0
                && (self.gqa[0]..=self.gqa[1]).contains(&(op.i[1] / op.i[2]))
                && (self.window == ANY_SLIDING || op.i[4] == self.window || op.i[4] == 0)
                && (self.ring_kv || op.i[7] == u32::MAX);
        }
        op.op == self.kv.op() as u16
            && op.i[6] == self.head_dim
            && heads_and_window_match(op, self.window, self.pair_heads)
            && packet::dev::flash_nsplit(op.i[7]) == 1
            && packet::dev::flash_span(op.i[7]).is_some() == self.media_span
            && (self.ring_kv || op.j[1] == u32::MAX)
            // FP8-KV roles need the per-row K/V scales; gen_flash_prefill.cu reads t[6]/t[7] unchecked.
            && (self.kv == KvDtype::Bf16 || (op.t[6] != TENSOR_NONE && op.t[7] != TENSOR_NONE))
    }

    /// A decode entry's packet object: hash-pinned, with the geometry the runtime checks
    /// (`plow_gen_flash_decode_abi`, GQA ratio, tiles, split width).
    fn decode_object(
        &self,
        directory: &Path,
        profile: &str,
        gpu: &str,
    ) -> Result<SegmentObject, String> {
        let path = directory.join(&self.file);
        let image = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let info = plow_asset::cubin::inspect(&image)
            .ok_or_else(|| format!("{} is not a valid cubin", path.display()))?;
        let global = |name| plow_asset::cubin::global_u32(&image, name);
        let (Some(block), Some(smem), Some(kv_tile), Some(warps)) = (
            global("plow_gen_block"),
            global("plow_gen_arena_bytes"),
            global("plow_attention_kv_tile"),
            global("plow_attention_warps"),
        ) else {
            return Err(format!("{} lacks generated-role geometry", path.display()));
        };
        if profile != "sm90a"
            || info.sm != 90
            || !info
                .entries
                .iter()
                .any(|entry| entry == DECODE_ENTRY_SYMBOL)
            || global("plow_gen_flash_decode_abi") != Some(1)
            || global("plow_attention_head_dim") != Some(self.head_dim)
            || global("plow_gen_decode_gqa") != Some(self.gqa[0])
            || global("plow_attention_query_tile") != Some(self.gqa[0])
            || global("plow_gen_decode_nsplit_max") != Some(self.nsplit_max)
        {
            return Err(format!(
                "{} is not a generated {} flash-decode object",
                path.display(),
                self.name
            ));
        }
        attention_prefill_role::validate_hardware_resources(gpu, profile, block, warps, smem)?;
        let abi = GeneratedAbi {
            family: GENERATED_FLASH_DECODE_ABI.into(),
            entry: self.name.clone(),
            block,
            smem,
        };
        Ok(SegmentObject {
            abi: abi.format(),
            file: self.file.clone(),
            sha256: Some(plow_asset::decode_objects::image_sha256(&image)),
            promote_k512: None,
            attention: Some(AttentionCapability {
                profile: "sm90a".into(),
                dtype: "bf16".into(),
                head_dim: self.head_dim,
                query_tile: self.gqa[0],
                kv_tile,
                warps,
                shape: None,
            }),
            gemm: None,
            decode_plan: None,
        })
    }

    fn object(&self, directory: &Path, profile: &str, gpu: &str) -> Result<Selection, String> {
        let path = directory.join(&self.file);
        let image = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let info = plow_asset::cubin::inspect(&image)
            .ok_or_else(|| format!("{} is not a valid cubin", path.display()))?;
        let global = |name| plow_asset::cubin::global_u32(&image, name);
        let (Some(block), Some(smem), Some(query_tile), Some(kv_tile), Some(warps)) = (
            global("plow_gen_block"),
            global("plow_gen_arena_bytes"),
            global("plow_attention_query_tile"),
            global("plow_attention_kv_tile"),
            global("plow_attention_warps"),
        ) else {
            return Err(format!("{} lacks generated-role geometry", path.display()));
        };
        if profile != "sm90a"
            || info.sm != 90
            || !info.entries.iter().any(|entry| entry == ENTRY_SYMBOL)
            || global("plow_gen_flash_prefill_abi") != Some(self.kv.object_abi())
            || global("plow_attention_head_dim") != Some(self.head_dim)
            || global("plow_pf_request_abi") != Some(2)
            || (global("plow_attention_media_span") == Some(1)) != self.media_span
        {
            return Err(format!(
                "{} is not a generated {} flash-prefill object",
                path.display(),
                self.name
            ));
        }
        attention_prefill_role::validate_hardware_resources(gpu, profile, block, warps, smem)?;
        let abi = GeneratedAbi {
            family: self.kv.abi_family().into(),
            entry: self.name.clone(),
            block,
            smem,
        };
        let attention = AttentionCapability {
            profile: "sm90a".into(),
            dtype: "bf16".into(),
            head_dim: self.head_dim,
            query_tile,
            kv_tile,
            warps,
            shape: None,
        };
        Ok(Selection::generated(
            self.file.clone(),
            &image,
            attention_prefill_role::Generated {
                role: self.role,
                abi: abi.format(),
                attention,
                window: self.window,
                ring_kv: self.ring_kv,
                pair_heads: self.pair_heads,
                kv: self.kv,
                media_span: self.media_span,
            },
        ))
    }
}

/// Bind each listed entry's matching prefill ops (rungs of at least `min_rows`) to its role.
/// Runs before the hand-written attention roles, which then keep the remaining rungs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_output_objects(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    profile: &str,
    output: &Path,
    gpu: &str,
    packed: bool,
    list: &str,
    hd512_px4_role: bool,
) -> Result<(), String> {
    let profile = if profile == "sm_90a" {
        "sm90a"
    } else {
        profile
    };
    let entries = parse(list)?;
    if entries.is_empty() {
        return Ok(());
    }
    // gen_flash_prefill.cu serves an unpacked chunk as one request; only the FP8-KV and
    // media-span objects read the packed request table.
    if !packed && entries.iter().any(|e| e.kv == KvDtype::Fp8 || e.media_span) {
        return Err("FP8-KV / media-span generated roles require packed prefill metadata".into());
    }
    if hd512_px4_role && entries.iter().any(|e| e.head_dim == 512) {
        return Err("PLOW_EMIT_GEN_KERNELS hd512 and the px4 BQ64 role both claim HD512".into());
    }
    let directory = output.parent().unwrap_or_else(|| Path::new("."));
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    for entry in entries {
        if entry.decode {
            let bound = (prefill_count..model.progs.len())
                .any(|index| model.progs[index].insts.iter().any(|op| entry.matches(op)));
            if !bound {
                return Err(format!(
                    "packet has no decode op matching generated entry {}",
                    entry.name
                ));
            }
            let object = entry.decode_object(directory, profile, gpu)?;
            bind_decode(model, sections, entry, object, prefill_count)?;
            continue;
        }
        let selection = entry.object(directory, profile, gpu)?;
        let programs: BTreeSet<usize> = (0..prefill_count)
            .filter(|&index| {
                model.prog_t[index] >= entry.min_rows
                    && model.progs[index].insts.iter().any(|op| entry.matches(op))
            })
            .collect();
        if programs.is_empty() {
            return Err(format!(
                "packet has no prefill op matching generated entry {}",
                entry.name
            ));
        }
        attention_prefill_role::apply(model, sections, &selection, profile, Some(&programs))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_unknown_entries_and_dedups() {
        assert_eq!(parse("attn_pf_hd512, attn_pf_hd512,").unwrap().len(), 1);
        assert!(parse("").unwrap().is_empty());
        assert!(matches!(parse("attn_pf_hd999"), Err(e) if e.contains("attn_pf_hd512")));
    }

    #[test]
    fn sliding_entry_takes_any_window_and_paired_heads() {
        let entry = CATALOG
            .iter()
            .find(|e| e.name == "attn_pf_hd256_sliding")
            .unwrap();
        let mut op = DevInst {
            op: DevOp::FlashPrefill as u16,
            ..Default::default()
        };
        // 12B / 26B sliding layers, then E4B's.
        op.i = [4096, 4096, 16, 8, 0, 1024, 256, 1];
        op.j[1] = 8191;
        assert!(entry.matches(&op));
        op.i[2..6].copy_from_slice(&[8, 2, 0, 512]);
        assert!(entry.matches(&op));
        // Global layers, odd GQA ratios and split launches stay with their own roles.
        op.i[5] = 0;
        assert!(!entry.matches(&op));
        op.i[5] = 512;
        op.i[2] = 6;
        assert!(!entry.matches(&op));
        op.i[2] = 8;
        op.i[7] = 2;
        assert!(!entry.matches(&op));
    }

    #[test]
    fn kv_dtype_is_part_of_the_signature() {
        let entry = |name| CATALOG.iter().find(|e| e.name == name).unwrap();
        let (bf16, fp8) = (
            entry("attn_pf_hd256_sliding"),
            entry("attn_pf_hd256_sliding_fp8kv"),
        );
        let mut op = DevInst {
            op: DevOp::FlashPrefill as u16,
            ..Default::default()
        };
        op.i = [4096, 4096, 16, 8, 0, 1024, 256, 1];
        op.j[1] = 2047;
        assert!(bf16.matches(&op) && !fp8.matches(&op));
        op.op = DevOp::FlashPrefillFp8 as u16;
        assert!(!bf16.matches(&op) && fp8.matches(&op));
        // Media-span sites (i7 high half) belong to the span twin only.
        let span = entry("attn_pf_hd256_sliding_fp8kv_span");
        assert!(!span.matches(&op));
        op.i[7] = 1 | 900 << 16;
        assert!(!fp8.matches(&op) && span.matches(&op));
        op.i[7] = 1;
        // FP8-KV global attention has its own entry; the bf16 one does not claim it.
        op.i[5] = 0;
        op.i[6] = 512;
        op.j[1] = u32::MAX;
        assert!(entry("attn_pf_hd512_fp8kv").matches(&op) && !entry("attn_pf_hd512").matches(&op));
        op.t[7] = TENSOR_NONE;
        assert!(!entry("attn_pf_hd512_fp8kv").matches(&op));
    }

    #[test]
    fn catalog_roles_are_distinct_generated_ids() {
        let mut roles = BTreeSet::new();
        for entry in CATALOG.iter() {
            assert!(plow_asset::segment_roles::is_generated(entry.role));
            assert!(roles.insert((entry.role, entry.kv_layout)));
        }
    }

    #[test]
    fn catalog_signatures_come_from_the_table() {
        let got: Vec<_> = CATALOG
            .iter()
            .map(|e| {
                (
                    e.name.as_str(),
                    e.file.as_str(),
                    e.kv,
                    e.head_dim,
                    e.window,
                    e.ring_kv,
                    e.pair_heads,
                    e.media_span,
                )
            })
            .collect();
        assert_eq!(
            got,
            [
                (
                    "attn_pf_hd512",
                    "gen_sm90a_attn_pf_hd512.cubin",
                    KvDtype::Bf16,
                    512,
                    0,
                    false,
                    false,
                    false
                ),
                (
                    "attn_pf_hd256_sliding",
                    "gen_sm90a_attn_pf_hd256_sliding.cubin",
                    KvDtype::Bf16,
                    256,
                    ANY_SLIDING,
                    true,
                    true,
                    false
                ),
                (
                    "attn_pf_hd256_sliding_fp8kv",
                    "gen_sm90a_attn_pf_hd256_sliding_fp8kv.cubin",
                    KvDtype::Fp8,
                    256,
                    ANY_SLIDING,
                    true,
                    true,
                    false
                ),
                (
                    "attn_pf_hd512_fp8kv",
                    "gen_sm90a_attn_pf_hd512_fp8kv.cubin",
                    KvDtype::Fp8,
                    512,
                    0,
                    false,
                    false,
                    false
                ),
                (
                    "attn_pf_hd256_sliding_fp8kv_span",
                    "gen_sm90a_attn_pf_hd256_sliding_fp8kv_span.cubin",
                    KvDtype::Fp8,
                    256,
                    ANY_SLIDING,
                    true,
                    true,
                    true,
                ),
                (
                    "attn_pf_hd256_causal",
                    "gen_sm90a_attn_pf_hd256_causal.cubin",
                    KvDtype::Bf16,
                    256,
                    0,
                    false,
                    true,
                    false
                ),
                (
                    "attn_dec_hd256_gqa6",
                    "gen_sm90a_attn_dec_hd256_gqa6.cubin",
                    KvDtype::Bf16,
                    256,
                    ANY_SLIDING,
                    true,
                    false,
                    false
                ),
                (
                    "attn_pf_hd256_causal_ilv",
                    "gen_sm90a_attn_pf_hd256_causal_ilv.cubin",
                    KvDtype::Bf16,
                    256,
                    0,
                    false,
                    true,
                    false
                ),
                (
                    "attn_dec_hd256_gqa6_ilv",
                    "gen_sm90a_attn_dec_hd256_gqa6_ilv.cubin",
                    KvDtype::Bf16,
                    256,
                    ANY_SLIDING,
                    true,
                    false,
                    false
                ),
            ]
        );
    }

    #[test]
    fn decode_entry_matches_flash_decode_at_its_gqa_only() {
        let entry = CATALOG
            .iter()
            .find(|e| e.name == "attn_dec_hd256_gqa6")
            .unwrap();
        assert!(entry.decode && entry.gqa == [6, 6] && entry.nsplit_max == 8);
        let mut op = DevInst {
            op: DevOp::FlashDecode as u16,
            ..Default::default()
        };
        // Qwen3.8-27B decode attention: 24 / 4 heads, hd256, linear KV, any split count.
        op.i = [16, 24, 4, 16384, 0, 3, 256, u32::MAX];
        assert!(entry.matches(&op));
        op.i[7] = 16383;
        assert!(entry.matches(&op), "ring KV is a runtime operand");
        op.i[4] = 1024;
        assert!(entry.matches(&op), "window is a runtime operand");
        op.i[2] = 8;
        assert!(!entry.matches(&op), "GQA 3");
        op.i[2] = 4;
        op.i[6] = 128;
        assert!(!entry.matches(&op), "head width");
        op.i[6] = 256;
        op.op = DevOp::FlashPrefill as u16;
        assert!(!entry.matches(&op), "prefill op");
        // Prefill entries never claim decode ops.
        op.op = DevOp::FlashDecode as u16;
        assert!(CATALOG
            .iter()
            .filter(|e| !e.decode)
            .all(|e| !e.matches(&op)));
    }

    #[test]
    fn causal_hd256_entry_takes_global_linear_kv_with_even_gqa() {
        let entry = CATALOG
            .iter()
            .find(|e| e.name == "attn_pf_hd256_causal")
            .unwrap();
        let sliding = CATALOG
            .iter()
            .find(|e| e.name == "attn_pf_hd256_sliding")
            .unwrap();
        let mut op = DevInst {
            op: DevOp::FlashPrefill as u16,
            ..Default::default()
        };
        // Qwen3.8-27B full-attention layer: 24 / 4 heads, hd256, global, linear KV, one split.
        op.i = [1024, 1024, 24, 4, 0, 0, 256, 1];
        op.j[1] = u32::MAX;
        assert!(entry.matches(&op) && !sliding.matches(&op));
        op.j[1] = 16383;
        assert!(!entry.matches(&op), "ring KV");
        op.j[1] = u32::MAX;
        op.i[5] = 1024;
        assert!(!entry.matches(&op), "window");
        op.i[5] = 0;
        op.i[2] = 20;
        assert!(!entry.matches(&op), "odd GQA ratio");
        op.i[2] = 24;
        op.i[7] = 2;
        assert!(!entry.matches(&op), "split launch");
    }

    #[test]
    fn table_and_policy_must_agree() {
        let mut table: serde_json::Value = serde_json::from_str(TABLE).unwrap();
        let rows = table["entries"].as_object_mut().unwrap();
        let row = rows["attn_pf_hd512"].clone();
        rows.insert("attn_pf_new".into(), row);
        assert!(matches!(load_catalog(&table.to_string()), Err(e) if e.contains("attn_pf_new")));
        let rows = table["entries"].as_object_mut().unwrap();
        rows.remove("attn_pf_new");
        rows.remove("attn_pf_hd512_fp8kv");
        assert!(
            matches!(load_catalog(&table.to_string()), Err(e) if e.contains("attn_pf_hd512_fp8kv"))
        );
    }
}

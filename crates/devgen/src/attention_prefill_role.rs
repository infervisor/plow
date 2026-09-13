use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use plow_asset::segment_roles::{
    AttentionCapability, ProgramRoles, SegmentObject, SegmentRoles, INTERPRETER,
    PREFILL_ATTENTION_HD256_BKV32, PREFILL_ATTENTION_HD256_BKV32_ABI,
    PREFILL_ATTENTION_HD256_GQA2_BKV32, PREFILL_ATTENTION_HD256_GQA2_BKV32_ABI,
    PREFILL_ATTENTION_HD512_WG32, PREFILL_ATTENTION_HD512_WG32_ABI, SECTION,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const OBJECT_FILE: &str = "interp_sm90a_pfattn_hd512.cubin";
const OBJECT_ENTRY: &str = "plow_sm90a_pfattn_hd512";
const OBJECT_GLOBALS: [(&str, u32); 7] = [
    ("plow_attention_sm90_hd512_wg32_abi", 1),
    ("plow_attention_head_dim", 512),
    ("plow_attention_query_tile", 32),
    ("plow_attention_kv_tile", 16),
    ("plow_attention_warps", 8),
    ("plow_block_pfattn_hd512", 256),
    ("plow_arena_bytes_pfattn_hd512", 70_672),
];
const HD256_OBJECT_FILE: &str = "interp_sm90a_pfattn_hd256_bkv32.cubin";
const HD256_OBJECT_ENTRY: &str = "plow_sm90a_pfattn_hd256_bkv32";
const HD256_OBJECT_GLOBALS: [(&str, u32); 7] = [
    ("plow_attention_sm90_hd256_bkv32_abi", 1),
    ("plow_attention_head_dim", 256),
    ("plow_attention_query_tile", 64),
    ("plow_attention_kv_tile", 32),
    ("plow_attention_warps", 8),
    ("plow_block_pfattn_hd256_bkv32", 256),
    ("plow_arena_bytes_pfattn_hd256_bkv32", 103_424),
];
const HD256_GQA2_OBJECT_FILE: &str = "interp_sm90a_pfattn_hd256_gqa2_bkv32.cubin";
const HD256_GQA2_OBJECT_ENTRY: &str = "plow_sm90a_pfattn_hd256_gqa2_bkv32";
const HD256_GQA2_OBJECT_GLOBALS: [(&str, u32); 7] = [
    ("plow_attention_sm90_hd256_gqa2_bkv32_abi", 1),
    ("plow_attention_head_dim", 256),
    ("plow_attention_query_tile", 64),
    ("plow_attention_kv_tile", 32),
    ("plow_attention_warps", 8),
    ("plow_block_pfattn_hd256_gqa2_bkv32", 256),
    ("plow_arena_bytes_pfattn_hd256_gqa2_bkv32", 141_312),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Hd256Bkv32,
    Hd256Gqa2Bkv32,
    Hd512,
}

pub struct Selection {
    pub file: String,
    pub sha256: String,
    wgmma: bool,
    query_tile: u32,
    kv_tile: u32,
    kind: Kind,
}

struct Hd256Qualification {
    programs: BTreeSet<usize>,
    file: String,
    sha256: String,
}

fn hd256_implementation() -> String {
    plow_asset::decode_objects::image_sha256(include_bytes!(
        "../../../runtime/nvidia/interp_sm90a_pfattn_hd256_bkv32.cu"
    ))
}

fn live_kv_buckets(ctx: u32) -> Vec<tunedb::KvBucket> {
    [1, 2048, 6144, 12288, 24576, 49152, 98304]
        .into_iter()
        .filter(|&lower| lower <= ctx)
        .map(tunedb::KvBucket::of)
        .collect()
}

fn required_cells(
    ctx: u32,
    packed: bool,
    m_rung: u32,
) -> Vec<(tunedb::KvBucket, tunedb::AttentionTopology)> {
    let mut cells = Vec::new();
    for bucket in live_kv_buckets(ctx) {
        if m_rung <= ctx && tunedb::KvBucket::of(m_rung) <= bucket {
            cells.push((bucket, tunedb::AttentionTopology::Single));
        }
        if packed && m_rung > 1 {
            cells.push((bucket, tunedb::AttentionTopology::PackedHomogeneous));
            cells.push((bucket, tunedb::AttentionTopology::PackedRagged));
        }
    }
    cells
}

fn qualify_hd256_programs(
    model: &Model,
    records: &[tunedb::AttentionRoleMeasurement],
    hardware: &str,
    arch: &str,
    ctx: u32,
    packed: bool,
    toolchain: &str,
) -> Result<Option<Hd256Qualification>, String> {
    if records.is_empty() {
        return Ok(None);
    }
    let digests = plow_asset::program::with_model(model, |packet| {
        Ok::<_, String>(
            packet.programs[..packet.prefill_count]
                .iter()
                .map(plow_asset::live_kv::program_digest)
                .collect::<Vec<_>>(),
        )
    })?;
    if crate::emit_config::active().tune_dump {
        for (index, digest) in digests.iter().enumerate() {
            if model.progs[index]
                .insts
                .iter()
                .any(packet::dev::DevInst::is_hd256_gqa2_sliding_prefill)
            {
                eprintln!(
                    "ATTENTION_ROLE_DIGEST program={index} rows={} sha256={digest}",
                    model.prog_t[index]
                );
            }
        }
    }
    let implementation = hd256_implementation();
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    if crate::emit_config::active().tune_dump {
        eprintln!(
            "ATTENTION_ROLE_CONTEXT hardware={hardware} arch={arch} ctx={ctx} packed={packed} toolchain={toolchain} records={}",
            records.len()
        );
    }
    let mut qualification: Option<Hd256Qualification> = None;
    for index in 0..prefill_count {
        if !model.progs[index]
            .insts
            .iter()
            .any(packet::dev::DevInst::is_hd256_gqa2_sliding_prefill)
        {
            continue;
        }
        let mut selected = Vec::new();
        for (bucket, topology) in required_cells(ctx, packed, model.prog_t[index]) {
            let cell = tunedb::AttentionRoleCell {
                hardware: hardware.into(),
                n_cu: model.n_cu,
                arch: arch.into(),
                dtype: "bf16".into(),
                kv_dtype: "bf16".into(),
                head_dim: 256,
                gqa: 2,
                window: 1024,
                m_rung: model.prog_t[index],
                live_kv_bucket: bucket,
                topology,
            };
            let Some(record) = tunedb::select_attention_role(
                records,
                &cell,
                PREFILL_ATTENTION_HD256_BKV32,
                &digests[index],
                &implementation,
                toolchain,
            ) else {
                if crate::emit_config::active().tune_dump {
                    eprintln!(
                        "ATTENTION_ROLE_MISS program={index} cell={} sha256={}",
                        cell.key(),
                        digests[index]
                    );
                }
                selected.clear();
                break;
            };
            if record.object_file != HD256_OBJECT_FILE
                || record.config.query_tile != 64
                || record.config.kv_tile != 32
                || record.config.warps != 8
                || record.config.stages != 2
                || record.config.nsplit != 1
                || record.config.group_factor != 2
            {
                selected.clear();
                break;
            }
            selected.push(record);
        }
        if selected.is_empty() {
            continue;
        }
        let sha256 = &selected[0].object_sha256;
        if selected
            .iter()
            .any(|record| record.object_sha256 != *sha256)
        {
            return Err("qualified HD256 attention cells require conflicting objects".into());
        }
        match &mut qualification {
            Some(old) if old.sha256 != *sha256 || old.file != HD256_OBJECT_FILE => {
                return Err("qualified HD256 attention rungs require conflicting objects".into())
            }
            Some(old) => {
                old.programs.insert(index);
            }
            None => {
                qualification = Some(Hd256Qualification {
                    programs: BTreeSet::from([index]),
                    file: HD256_OBJECT_FILE.into(),
                    sha256: sha256.clone(),
                });
            }
        }
    }
    Ok(qualification)
}

fn apply_qualified_hd256(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    profile: &str,
    directory: &Path,
    qualification: &Hd256Qualification,
) -> Result<(), String> {
    let path = directory.join(&qualification.file);
    let image = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    if plow_asset::decode_objects::image_sha256(&image) != qualification.sha256 {
        return Err(format!("{} differs from qualified SHA256", path.display()));
    }
    let info = plow_asset::cubin::inspect(&image)
        .ok_or_else(|| format!("{} is not a valid cubin", path.display()))?;
    if profile != "sm90a"
        || info.sm != 90
        || !info.entries.iter().any(|entry| entry == HD256_OBJECT_ENTRY)
        || HD256_OBJECT_GLOBALS
            .iter()
            .any(|&(name, value)| plow_asset::cubin::global_u32(&image, name) != Some(value))
    {
        return Err(format!(
            "{} has incompatible HD256 BKV32 prefill attention capabilities",
            path.display()
        ));
    }
    apply(
        model,
        sections,
        &Selection::hd256_bkv32(qualification.file.clone(), &image),
        profile,
        Some(&qualification.programs),
    )
}

impl Selection {
    pub fn from_image(file: String, image: &[u8], wgmma: bool) -> Self {
        let query_tile = plow_asset::cubin::global_u32(image, "plow_attention_query_tile")
            .unwrap_or(if wgmma { 64 } else { 32 });
        let kv_tile = plow_asset::cubin::global_u32(image, "plow_attention_kv_tile")
            .unwrap_or(if wgmma { 32 } else { 16 });
        Self {
            file,
            sha256: plow_asset::decode_objects::image_sha256(image),
            wgmma: query_tile == 64,
            query_tile,
            kv_tile,
            kind: Kind::Hd512,
        }
    }

    fn hd256_bkv32(file: String, image: &[u8]) -> Self {
        Self {
            file,
            sha256: plow_asset::decode_objects::image_sha256(image),
            wgmma: false,
            query_tile: 64,
            kv_tile: 32,
            kind: Kind::Hd256Bkv32,
        }
    }

    fn hd256_gqa2_bkv32(image: &[u8]) -> Self {
        Self {
            file: HD256_GQA2_OBJECT_FILE.into(),
            sha256: plow_asset::decode_objects::image_sha256(image),
            wgmma: true,
            query_tile: 64,
            kv_tile: 32,
            kind: Kind::Hd256Gqa2Bkv32,
        }
    }

    fn role(&self) -> u8 {
        match self.kind {
            Kind::Hd256Bkv32 => PREFILL_ATTENTION_HD256_BKV32,
            Kind::Hd256Gqa2Bkv32 => PREFILL_ATTENTION_HD256_GQA2_BKV32,
            Kind::Hd512 => PREFILL_ATTENTION_HD512_WG32,
        }
    }
}

fn capability(query_tile: u32, kv_tile: u32) -> AttentionCapability {
    AttentionCapability {
        profile: "sm90a".into(),
        dtype: "bf16".into(),
        head_dim: 512,
        query_tile,
        kv_tile,
        warps: 8,
    }
}

fn eligible(op: &packet::dev::DevInst, n_cu: u16) -> bool {
    match DevOp::from_u16(op.op) {
        Some(DevOp::FlashPrefill) => {
            op.blocks == n_cu
                && op.i[0] > 0
                && op.i[1] > 0
                && op.i[2] > 0
                && op.i[3] > 0
                && op.i[2] % op.i[3] == 0
                && op.i[6] == 512
                && op.i[7] > 0
                && (op.t[5] == TENSOR_NONE || op.i[7] == 1)
                && op.t[6] == TENSOR_NONE
                && op.t[7] != TENSOR_NONE
                && op.f[0].is_finite()
        }
        _ => false,
    }
}

fn eligible_for(op: &packet::dev::DevInst, n_cu: u16, selection: &Selection) -> bool {
    match selection.kind {
        Kind::Hd512 => eligible(op, n_cu),
        Kind::Hd256Bkv32 | Kind::Hd256Gqa2Bkv32 => {
            op.blocks == n_cu
                && op.is_hd256_gqa2_sliding_prefill()
                && op.i[0] > 0
                && op.i[1] > 0
                && op.t[6] == TENSOR_NONE
                && op.t[7] != TENSOR_NONE
                && op.f[0].is_finite()
        }
    }
}

fn is_hd512_attention(op: &packet::dev::DevInst) -> bool {
    (op.op == DevOp::FlashPrefill as u16 && op.i[6] == 512)
        || (op.op == DevOp::FlashMerge as u16 && op.i[3] == 512)
}

fn valid_hd512_merge(op: &packet::dev::DevInst) -> bool {
    op.op == DevOp::FlashMerge as u16 && op.i[0] > 0 && op.i[1] > 0 && op.i[2] > 0 && op.i[3] == 512
}

/// Bind qualified HD256 attention rungs and legacy sibling-selected HD512 attention.
/// Missing HD256 records and missing HD512 objects leave their packet segments untouched.
pub(crate) fn apply_output_object(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    profile: &str,
    output: &Path,
    gpu: &str,
    ctx: u32,
    packed: bool,
    tunedb_root: Option<&str>,
    gqa2_role: bool,
) -> Result<bool, String> {
    let profile = if profile == "sm_90a" {
        "sm90a"
    } else {
        profile
    };
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    let has_hd256 = model.progs[..prefill_count]
        .iter()
        .flat_map(|program| &program.insts)
        .any(packet::dev::DevInst::is_hd256_gqa2_sliding_prefill);
    let has_hd512 = model.progs[..prefill_count]
        .iter()
        .flat_map(|program| &program.insts)
        .any(is_hd512_attention);
    let mut applied = false;
    let directory = output.parent().unwrap_or_else(|| Path::new("."));
    if gqa2_role {
        if !packed {
            return Err("paired HD256/GQA2 role requires packed prefill metadata".into());
        }
        let path = directory.join(HD256_GQA2_OBJECT_FILE);
        let image = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let info = plow_asset::cubin::inspect(&image)
            .ok_or_else(|| format!("{} is not a valid cubin", path.display()))?;
        if profile != "sm90a"
            || info.sm != 90
            || !info
                .entries
                .iter()
                .any(|entry| entry == HD256_GQA2_OBJECT_ENTRY)
            || HD256_GQA2_OBJECT_GLOBALS
                .iter()
                .any(|&(name, value)| plow_asset::cubin::global_u32(&image, name) != Some(value))
        {
            return Err(format!(
                "{} has incompatible paired HD256/GQA2 prefill attention capabilities",
                path.display()
            ));
        }
        let programs = model.progs[..prefill_count]
            .iter()
            .enumerate()
            .filter(|(index, program)| {
                matches!(model.prog_t[*index], 4096 | 8192)
                    && program
                        .insts
                        .iter()
                        .any(packet::dev::DevInst::is_hd256_gqa2_sliding_prefill)
            })
            .map(|(index, _)| index)
            .collect::<BTreeSet<_>>();
        if programs.is_empty() {
            return Err("packet has no M4096/M8192 paired HD256/GQA2 attention rungs".into());
        }
        apply(
            model,
            sections,
            &Selection::hd256_gqa2_bkv32(&image),
            profile,
            Some(&programs),
        )?;
        applied = true;
    }
    if has_hd256 && !gqa2_role {
        let qualification = if let (Some(root), Some(spec)) =
            (tunedb_root, hwspec::registry::lookup(gpu))
        {
            let fingerprint = kernelcaps::HardwareFingerprint::from_spec(spec)
                .ok_or("missing hardware fingerprint")?;
            let hardware = fingerprint.tuning_path();
            let records = tunedb::TuneStore::new(root)
                .load_attention_roles(&hardware)
                .map_err(|error| error.to_string())?;
            let toolchain = kernelcaps::toolchain_label(fingerprint.isa);
            qualify_hd256_programs(model, &records, &hardware, profile, ctx, packed, &toolchain)?
        } else {
            None
        };
        if let Some(qualification) = qualification {
            apply_qualified_hd256(model, sections, profile, directory, &qualification)?;
            applied = true;
        }
    }
    if !has_hd512 {
        return Ok(applied);
    }
    let path = directory.join(OBJECT_FILE);
    let image = match std::fs::read(&path) {
        Ok(image) => image,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(applied),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let info = plow_asset::cubin::inspect(&image)
        .ok_or_else(|| format!("{} is not a valid cubin", path.display()))?;
    let query_tile = plow_asset::cubin::global_u32(&image, "plow_attention_query_tile");
    let kv_tile = plow_asset::cubin::global_u32(&image, "plow_attention_kv_tile");
    let wgmma = query_tile == Some(64);
    let score_partitions =
        plow_asset::cubin::global_u32(&image, "plow_attention_score_partitions").unwrap_or(1);
    let mut expected = OBJECT_GLOBALS;
    match (query_tile, kv_tile) {
        (Some(32), Some(16)) => {}
        (Some(64), Some(16)) => {
            expected[2].1 = 64;
            expected[3].1 = 16;
            expected[6].1 = 134_144;
        }
        (Some(64), Some(32)) => {
            expected[2].1 = 64;
            expected[3].1 = 32;
            expected[6].1 = 201_728;
        }
        (Some(64), Some(64)) => {
            expected[2].1 = 64;
            expected[3].1 = 64;
            expected[6].1 = match score_partitions {
                1 => 205_824,
                2 => 206_848,
                _ => 0,
            };
        }
        _ => expected[2].1 = 0,
    }
    if profile != "sm90a"
        || info.sm != 90
        || !matches!(score_partitions, 1 | 2)
        || (score_partitions == 2 && (query_tile != Some(64) || kv_tile != Some(64)))
        || !info.entries.iter().any(|entry| entry == OBJECT_ENTRY)
        || expected
            .iter()
            .any(|&(name, value)| plow_asset::cubin::global_u32(&image, name) != Some(value))
    {
        return Err(format!(
            "{} has incompatible HD512 prefill attention capabilities",
            path.display()
        ));
    }
    apply(
        model,
        sections,
        &Selection::from_image(OBJECT_FILE.into(), &image, wgmma),
        profile,
        None,
    )?;
    Ok(true)
}

fn apply(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    selection: &Selection,
    profile: &str,
    allowed_programs: Option<&BTreeSet<usize>>,
) -> Result<(), String> {
    if profile != "sm90a" {
        return Err(match selection.kind {
            Kind::Hd256Bkv32 => "HD256 BKV32 prefill attention role requires sm90a".into(),
            Kind::Hd256Gqa2Bkv32 => {
                "paired HD256/GQA2 prefill attention role requires sm90a".into()
            }
            Kind::Hd512 => "HD512 WG32 prefill attention role requires sm90a".into(),
        });
    }
    let attention = match selection.kind {
        Kind::Hd256Bkv32 | Kind::Hd256Gqa2Bkv32 => AttentionCapability {
            profile: "sm90a".into(),
            dtype: "bf16".into(),
            head_dim: 256,
            query_tile: 64,
            kv_tile: 32,
            warps: 8,
        },
        Kind::Hd512 => capability(selection.query_tile, selection.kv_tile),
    };
    let role = selection.role();
    let object = SegmentObject {
        abi: match selection.kind {
            Kind::Hd256Bkv32 => PREFILL_ATTENTION_HD256_BKV32_ABI,
            Kind::Hd256Gqa2Bkv32 => PREFILL_ATTENTION_HD256_GQA2_BKV32_ABI,
            Kind::Hd512 => PREFILL_ATTENTION_HD512_WG32_ABI,
        }
        .into(),
        file: selection.file.clone(),
        sha256: Some(selection.sha256.clone()),
        promote_k512: None,
        attention: Some(attention),
    };
    let mut metadata = SegmentRoles {
        version: 1,
        objects: BTreeMap::new(),
        programs: Vec::new(),
    };
    let matches: Vec<_> = sections
        .iter()
        .enumerate()
        .filter(|(_, section)| section.name == SECTION)
        .map(|(index, _)| index)
        .collect();
    if matches.len() > 1
        || matches
            .first()
            .is_some_and(|&index| sections[index].kind != SECT_METADATA)
    {
        return Err("duplicate segment role metadata".into());
    }
    if let Some(&index) = matches.first() {
        metadata = SegmentRoles::from_bytes(&sections[index].data)?;
    }
    if metadata.objects.contains_key(&role) {
        return Err(match selection.kind {
            Kind::Hd256Bkv32 => "HD256 BKV32 prefill attention role already declared".into(),
            Kind::Hd256Gqa2Bkv32 => {
                "paired HD256/GQA2 prefill attention role already declared".into()
            }
            Kind::Hd512 => "HD512 prefill attention role already declared".into(),
        });
    }

    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    let n_cu = u16::try_from(model.n_cu).map_err(|_| "packet grid exceeds role ABI")?;
    let tensor_bytes: Vec<_> = model.tensors.iter().map(|tensor| tensor.bytes).collect();
    struct Update {
        index: usize,
        roles: Vec<u8>,
        inst_segment: Vec<Option<u16>>,
        bounds: Vec<u32>,
        prior_position: Option<usize>,
    }
    let mut updates = Vec::new();
    let mut selected = 0usize;
    for (index, program) in model.progs[..prefill_count].iter().enumerate() {
        if allowed_programs.is_some_and(|programs| !programs.contains(&index)) {
            continue;
        }
        let eligible: Vec<bool> = program
            .insts
            .iter()
            .map(|op| {
                let output_head_dim = match selection.kind {
                    Kind::Hd256Bkv32 | Kind::Hd256Gqa2Bkv32 => 256,
                    Kind::Hd512 => 512,
                };
                eligible_for(op, n_cu, selection)
                    && (!selection.wgmma || (op.t[5] != TENSOR_NONE && op.i[7] == 1))
                    && tensor_bytes
                        .get(op.t[7] as usize)
                        .is_some_and(|&bytes| bytes == 256)
                    && (op.t[5] == TENSOR_NONE
                        || (op.t[..5].iter().all(|&tensor| tensor != op.t[5])
                            && u64::from(op.i[0])
                                .checked_mul(u64::from(op.i[2]))
                                .and_then(|elements| elements.checked_mul(output_head_dim * 2))
                                .is_some_and(|bytes| {
                                    tensor_bytes
                                        .get(op.t[5] as usize)
                                        .is_some_and(|&extent| extent >= bytes)
                                })))
            })
            .collect();
        let unfused_flash = program
            .insts
            .iter()
            .filter(|op| {
                op.op == DevOp::FlashPrefill as u16 && op.i[6] == 512 && op.t[5] == TENSOR_NONE
            })
            .count();
        let full_merge = program
            .insts
            .iter()
            .filter(|op| op.op == DevOp::FlashMerge as u16 && op.i[3] == 512)
            .count();
        if selection.kind == Kind::Hd512 && unfused_flash != full_merge {
            return Err(format!(
                "incompatible HD512 prefill attention pairing in program {index}"
            ));
        }
        if let Some((pc, (op, _))) =
            program
                .insts
                .iter()
                .zip(&eligible)
                .enumerate()
                .find(|(_, (op, selected))| {
                    (selection.kind != Kind::Hd512 || is_hd512_attention(op))
                        && (!matches!(selection.kind, Kind::Hd256Bkv32 | Kind::Hd256Gqa2Bkv32)
                            || op.is_hd256_gqa2_sliding_prefill())
                        && if op.op == DevOp::FlashPrefill as u16 {
                            !**selected
                        } else {
                            !valid_hd512_merge(op)
                        }
                })
        {
            return Err(format!(
                "incompatible {} prefill attention operand contract at program {index} pc {pc}: blocks={} i={:?} t={:?} map_bytes={:?}",
                match selection.kind {
                    Kind::Hd256Bkv32 => "HD256 BKV32",
                    Kind::Hd256Gqa2Bkv32 => "paired HD256/GQA2 BKV32",
                    Kind::Hd512 => "HD512",
                },
                op.blocks,
                op.i,
                op.t,
                tensor_bytes.get(op.t[7] as usize)
            ));
        }
        if !eligible.iter().any(|&yes| yes) {
            continue;
        }
        if program.l2_domains != 0 || program.hier_base != 0 {
            return Err(match selection.kind {
                Kind::Hd256Bkv32 => "HD256 BKV32 role requires a plain prefill program".into(),
                Kind::Hd256Gqa2Bkv32 => {
                    "paired HD256/GQA2 role requires a plain prefill program".into()
                }
                Kind::Hd512 => "HD512 role requires a plain prefill program".into(),
            });
        }
        let prior_position = metadata
            .programs
            .iter()
            .position(|program| program.index == index);
        let prior_roles = if let Some(position) = prior_position {
            let prior = &metadata.programs[position];
            if prior.roles.len() + 1 != program.gq_seg_ofs.len() {
                return Err("existing prefill role window coverage".into());
            }
            prior.roles.clone()
        } else {
            vec![INTERPRETER; program.gq_seg_ofs.len() - 1]
        };
        let mut instruction_roles = vec![None; program.insts.len()];
        for entry in program.stream.iter().chain(&program.gq_stream) {
            let role = *prior_roles
                .get(entry.seg as usize)
                .ok_or("existing prefill role segment out of bounds")?;
            let slot = instruction_roles
                .get_mut(entry.inst as usize)
                .ok_or("prefill queue instruction out of bounds")?;
            if slot.is_some_and(|prior| prior != role) {
                return Err("existing prefill instruction crosses role segments".into());
            }
            *slot = Some(role);
        }
        if !eligible
            .iter()
            .enumerate()
            .any(|(inst, &yes)| yes && instruction_roles[inst] == Some(INTERPRETER))
        {
            continue;
        }
        let mut roles = Vec::new();
        let mut inst_segment = vec![None; program.insts.len()];
        let mut bounds = vec![0u32];
        let mut last_key = None;
        for (queue_index, entry) in program.gq_stream.iter().enumerate() {
            let inst = entry.inst as usize;
            let prior_role = instruction_roles
                .get(inst)
                .and_then(|&role| role)
                .ok_or("prefill instruction absent from stream")?;
            let selected = *eligible
                .get(inst)
                .ok_or("prefill queue instruction out of bounds")?
                && prior_role == INTERPRETER;
            let role = if selected { role } else { prior_role };
            let key = if selected {
                (u32::MAX, role, inst)
            } else {
                (u32::from(entry.seg), role, usize::MAX)
            };
            let new_segment = last_key != Some(key);
            if new_segment {
                if !roles.is_empty() {
                    bounds.push(queue_index as u32);
                }
                roles.push(role);
            }
            let segment = u16::try_from(roles.len() - 1)
                .map_err(|_| "too many prefill attention role segments")?;
            if let Some(prior) = inst_segment[inst] {
                if prior != segment {
                    return Err(match selection.kind {
                        Kind::Hd256Bkv32 => {
                            "HD256 BKV32 role instruction is not contiguous in the queue".into()
                        }
                        Kind::Hd256Gqa2Bkv32 => {
                            "paired HD256/GQA2 role instruction is not contiguous in the queue"
                                .into()
                        }
                        Kind::Hd512 => {
                            "HD512 role instruction is not contiguous in the queue".into()
                        }
                    });
                }
            } else {
                inst_segment[inst] = Some(segment);
            }
            last_key = Some(key);
        }
        bounds.push(program.gq_stream.len() as u32);
        selected += roles.iter().filter(|&&candidate| candidate == role).count();
        updates.push(Update {
            index,
            roles,
            inst_segment,
            bounds,
            prior_position,
        });
    }
    if selected == 0 {
        return Err(match selection.kind {
            Kind::Hd256Bkv32 => {
                "packet has no compatible HD256 BKV32 prefill attention segments".into()
            }
            Kind::Hd256Gqa2Bkv32 => {
                "packet has no compatible paired HD256/GQA2 prefill attention segments".into()
            }
            Kind::Hd512 => "packet has no compatible HD512 prefill attention segments".into(),
        });
    }
    metadata.objects.insert(role, object);
    for update in &updates {
        let record = ProgramRoles {
            index: update.index,
            roles: update.roles.clone(),
        };
        if let Some(position) = update.prior_position {
            metadata.programs[position] = record;
        } else {
            metadata.programs.push(record);
        }
    }
    metadata.validate_schema()?;
    let section = SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&metadata).map_err(|error| error.to_string())?,
    };
    for update in updates {
        let program = &mut model.progs[update.index];
        for entry in program.stream.iter_mut().chain(&mut program.gq_stream) {
            entry.seg = update
                .inst_segment
                .get(entry.inst as usize)
                .and_then(|&segment| segment)
                .ok_or("prefill instruction absent from global queue")?;
        }
        program.gq_seg_ofs = update.bounds;
    }
    if let Some(&index) = matches.first() {
        sections[index] = section;
    } else {
        sections.push(section);
    }
    Ok(())
}

#[cfg(test)]
#[path = "attention_prefill_role_tests.rs"]
mod tests;

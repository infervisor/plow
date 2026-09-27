//! DeepSeek-V4.1 sm_90a prefill roles: every isolated `GemmFp8Mx` segment runs on
//! `interp_sm90a_pfgemm_fp8mx.cubin`, every isolated `MoeGroupGluPf` / `MoeGroupDownPf` segment on
//! `interp_sm90a_pfmoe_fp4.cubin`. Same shape as `w8a16_prefill_role`: the object must sit next to
//! the output and carry the expected entry and globals, and only one-instruction segments are
//! marked, so an absent object leaves the packet on the interpreter (which traps on these ops).
use packet::dev::DevOp;
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use plow_asset::segment_roles::{
    ProgramRoles, SegmentObject, SegmentRoles, FP4_PREFILL_MOE, FP4_PREFILL_MOE_ABI, FP8MX_PREFILL_GEMM,
    FP8MX_PREFILL_GEMM_ABI, INTERPRETER, PFFLASH_V41, PFFLASH_V41_ABI, SECTION,
};
use std::collections::BTreeMap;
use std::path::Path;

struct Role {
    id: u8,
    abi: &'static str,
    file: &'static str,
    entry: &'static str,
    abi_global: &'static str,
    block_global: &'static str,
    arena_global: &'static str,
    ops: &'static [DevOp],
}

const ROLES: [Role; 3] = [
    Role {
        id: FP8MX_PREFILL_GEMM,
        abi: FP8MX_PREFILL_GEMM_ABI,
        file: "interp_sm90a_pfgemm_fp8mx.cubin",
        entry: "plow_sm90a_pfgemm_fp8mx",
        abi_global: "plow_pfgemm_fp8mx_abi",
        block_global: "plow_block_pfgemm_fp8mx",
        arena_global: "plow_arena_bytes_pfgemm_fp8mx",
        ops: &[DevOp::GemmFp8Mx],
    },
    Role {
        id: FP4_PREFILL_MOE,
        abi: FP4_PREFILL_MOE_ABI,
        file: "interp_sm90a_pfmoe_fp4.cubin",
        entry: "plow_sm90a_pfmoe_fp4",
        abi_global: "plow_pfmoe_fp4_abi",
        block_global: "plow_block_pfmoe_fp4",
        arena_global: "plow_arena_bytes_pfmoe_fp4",
        ops: &[DevOp::MoeGroupGluPf, DevOp::MoeGroupDownPf],
    },
    Role {
        id: PFFLASH_V41,
        abi: PFFLASH_V41_ABI,
        file: "interp_sm90a_pfflash_v41.cubin",
        entry: "plow_sm90a_pfflash_v41",
        abi_global: "plow_pfflash_v41_abi",
        block_global: "plow_block_pfflash_v41",
        arena_global: "plow_arena_bytes_pfflash_v41",
        ops: &[DevOp::FlashMlaPrefill, DevOp::IndexScorePf],
    },
];

/// Marks the roles whose object is present next to `output`. Returns the role ids applied.
pub(crate) fn apply_output_objects(model: &Model, sections: &mut Vec<SectionData>, output: &Path) -> Result<Vec<u8>, String> {
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    let directory = output.parent().unwrap_or_else(|| Path::new("."));
    let positions: Vec<_> = sections.iter().enumerate().filter(|(_, s)| s.name == SECTION).map(|(i, _)| i).collect();
    if positions.len() > 1 || positions.first().is_some_and(|&i| sections[i].kind != SECT_METADATA) {
        return Err("duplicate segment role metadata".into());
    }
    let mut metadata = match positions.first() {
        Some(&i) => SegmentRoles::from_bytes(&sections[i].data)?,
        None => SegmentRoles { version: 1, objects: BTreeMap::new(), programs: Vec::new() },
    };
    let mut applied = Vec::new();
    for role in &ROLES {
        let path = directory.join(role.file);
        let image = match std::fs::read(&path) {
            Ok(image) => image,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let info = plow_asset::cubin::inspect(&image).ok_or_else(|| format!("{} is not a valid cubin", path.display()))?;
        let global = |name| plow_asset::cubin::global_u32(&image, name);
        if info.sm != 90
            || !info.entries.iter().any(|e| e == role.entry)
            || global(role.abi_global) != Some(1)
            || global(role.block_global) != Some(384)
            || global(role.arena_global).is_none_or(|v| v == 0)
        {
            return Err(format!("{} has incompatible role capabilities", path.display()));
        }
        if metadata.objects.contains_key(&role.id) {
            return Err(format!("role {} already declared", role.id));
        }
        let mut selected = 0usize;
        for (index, program) in model.progs[..prefill_count].iter().enumerate() {
            let prior = metadata.programs.iter().position(|r| r.index == index);
            let mut roles = match prior {
                Some(p) => metadata.programs[p].roles.clone(),
                None => vec![INTERPRETER; program.gq_seg_ofs.len() - 1],
            };
            if roles.len() + 1 != program.gq_seg_ofs.len() {
                return Err("existing prefill role window coverage".into());
            }
            for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
                let entries = &program.gq_stream[bounds[0] as usize..bounds[1] as usize];
                let Some(first) = entries.first() else { continue };
                let pc = first.inst as usize;
                if roles[segment] != INTERPRETER
                    || entries.iter().any(|e| e.inst as usize != pc)
                    || !role.ops.iter().any(|&op| program.insts[pc].op == op as u16)
                    || program.gq_stream.iter().any(|e| e.inst as usize == pc && e.seg as usize != segment)
                {
                    continue;
                }
                roles[segment] = role.id;
                selected += 1;
            }
            if roles.contains(&role.id) {
                let record = ProgramRoles { index, roles };
                match prior {
                    Some(p) => metadata.programs[p] = record,
                    None => metadata.programs.push(record),
                }
            }
        }
        if selected == 0 {
            continue;
        }
        metadata.objects.insert(
            role.id,
            SegmentObject {
                abi: role.abi.into(),
                file: role.file.into(),
                sha256: Some(plow_asset::decode_objects::image_sha256(&image)),
                promote_k512: None,
                attention: None,
            },
        );
        applied.push(role.id);
    }
    if applied.is_empty() {
        return Ok(applied);
    }
    metadata.validate_schema()?;
    let section = SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
    };
    match positions.first() {
        Some(&i) => sections[i] = section,
        None => sections.push(section),
    }
    Ok(applied)
}

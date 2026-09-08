use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use plow_asset::segment_roles::{
    AttentionCapability, ProgramRoles, SegmentObject, SegmentRoles, INTERPRETER,
    PREFILL_ATTENTION_HD512_WG32, PREFILL_ATTENTION_HD512_WG32_ABI, SECTION,
};
use std::collections::BTreeMap;
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
    ("plow_arena_bytes_pfattn_hd512", 201_728),
];

pub struct Selection {
    pub file: String,
    pub sha256: String,
}

impl Selection {
    pub fn from_image(file: String, image: &[u8]) -> Self {
        Self {
            file,
            sha256: plow_asset::decode_objects::image_sha256(image),
        }
    }
}

fn capability() -> AttentionCapability {
    AttentionCapability {
        profile: "sm90a".into(),
        dtype: "bf16".into(),
        head_dim: 512,
        query_tile: 32,
        kv_tile: 16,
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

fn is_hd512_attention(op: &packet::dev::DevInst) -> bool {
    (op.op == DevOp::FlashPrefill as u16 && op.i[6] == 512)
        || (op.op == DevOp::FlashMerge as u16 && op.i[3] == 512)
}

fn valid_hd512_merge(op: &packet::dev::DevInst) -> bool {
    op.op == DevOp::FlashMerge as u16 && op.i[0] > 0 && op.i[1] > 0 && op.i[2] > 0 && op.i[3] == 512
}

/// Bind the canonical object beside `output` when the packet contains HD512 prefill attention.
/// Object presence is the selection input; absence leaves the emitted packet untouched.
pub(crate) fn apply_output_object(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    profile: &str,
    output: &Path,
) -> Result<bool, String> {
    let profile = if profile == "sm_90a" {
        "sm90a"
    } else {
        profile
    };
    if !model.progs[..packet::devbuild::decode_rung_lo(&model.prog_t)]
        .iter()
        .flat_map(|program| &program.insts)
        .any(is_hd512_attention)
    {
        return Ok(false);
    }
    let directory = output.parent().unwrap_or_else(|| Path::new("."));
    let path = directory.join(OBJECT_FILE);
    let image = match std::fs::read(&path) {
        Ok(image) => image,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let info = plow_asset::cubin::inspect(&image)
        .ok_or_else(|| format!("{} is not a valid cubin", path.display()))?;
    if profile != "sm90a"
        || info.sm != 90
        || !info.entries.iter().any(|entry| entry == OBJECT_ENTRY)
        || OBJECT_GLOBALS
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
        &Selection::from_image(OBJECT_FILE.into(), &image),
        profile,
    )?;
    Ok(true)
}

fn apply(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    selection: &Selection,
    profile: &str,
) -> Result<(), String> {
    if profile != "sm90a" {
        return Err("HD512 WG32 prefill attention role requires sm90a".into());
    }
    let object = SegmentObject {
        abi: PREFILL_ATTENTION_HD512_WG32_ABI.into(),
        file: selection.file.clone(),
        sha256: Some(selection.sha256.clone()),
        promote_k512: None,
        attention: Some(capability()),
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
    if metadata.objects.contains_key(&PREFILL_ATTENTION_HD512_WG32) {
        return Err("HD512 prefill attention role already declared".into());
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
        let eligible: Vec<bool> = program
            .insts
            .iter()
            .map(|op| {
                eligible(op, n_cu)
                    && tensor_bytes
                        .get(op.t[7] as usize)
                        .is_some_and(|&bytes| bytes == 256)
                    && (op.t[5] == TENSOR_NONE
                        || (op.t[..5].iter().all(|&tensor| tensor != op.t[5])
                            && u64::from(op.i[0])
                                .checked_mul(u64::from(op.i[2]))
                                .and_then(|elements| elements.checked_mul(512 * 2))
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
        if unfused_flash != full_merge {
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
                    is_hd512_attention(op)
                        && if op.op == DevOp::FlashPrefill as u16 {
                            !**selected
                        } else {
                            !valid_hd512_merge(op)
                        }
                })
        {
            return Err(format!(
                "incompatible HD512 prefill attention operand contract at program {index} pc {pc}: blocks={} i={:?} t={:?} map_bytes={:?}",
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
            return Err("HD512 role requires a plain prefill program".into());
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
            let role = if selected {
                PREFILL_ATTENTION_HD512_WG32
            } else {
                prior_role
            };
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
                    return Err("HD512 role instruction is not contiguous in the queue".into());
                }
            } else {
                inst_segment[inst] = Some(segment);
            }
            last_key = Some(key);
        }
        bounds.push(program.gq_stream.len() as u32);
        selected += roles
            .iter()
            .filter(|&&role| role == PREFILL_ATTENTION_HD512_WG32)
            .count();
        updates.push(Update {
            index,
            roles,
            inst_segment,
            bounds,
            prior_position,
        });
    }
    if selected == 0 {
        return Err("packet has no compatible HD512 prefill attention segments".into());
    }
    metadata
        .objects
        .insert(PREFILL_ATTENTION_HD512_WG32, object);
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

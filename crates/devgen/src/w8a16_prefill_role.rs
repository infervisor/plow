use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use plow_asset::segment_roles::{
    ProgramRoles, SegmentObject, SegmentRoles, INTERPRETER, SECTION, W8A16_PREFILL_M1,
    W8A16_PREFILL_M1_ABI,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const OBJECT_FILE: &str = "interp_sm90a_pfgemm_w8a16_m1.cubin";
const OBJECT_ENTRY: &str = "plow_sm90a_pfgemm_w8a16_m1";
const QUALIFIED_SHAPES: [(u32, u32); 8] = [
    (512, 3840),
    (2048, 3840),
    (3840, 4096),
    (3840, 8192),
    (3840, 15360),
    (4096, 3840),
    (8192, 3840),
    (15360, 3840),
];

fn eligible(model: &Model, d: &packet::dev::DevInst) -> bool {
    if d.op != DevOp::GemmFp8 as u16
        || d.i[0] != 1
        || d.i[2] % 16 != 0
        || !QUALIFIED_SHAPES.contains(&(d.i[1], d.i[2]))
        || d.i[3..].iter().any(|&value| value != 0)
        || d.j != [0; 2]
        || d.f.iter().any(|value| value.to_bits() != 0)
        || d.t[0] == TENSOR_NONE
        || d.t[1] == TENSOR_NONE
        || d.t[2] == TENSOR_NONE
        || d.t[3] != TENSOR_NONE
        || d.t[4] == TENSOR_NONE
        || d.t[5..].iter().any(|&tensor| tensor != TENSOR_NONE)
        || [d.t[0], d.t[1], d.t[2], d.t[4]]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .len()
            != 4
    {
        return false;
    }
    let tensors = &model.tensors;
    let Some(output) = tensors.get(d.t[0] as usize) else {
        return false;
    };
    let Some(input) = tensors.get(d.t[1] as usize) else {
        return false;
    };
    let Some(weight) = tensors.get(d.t[2] as usize) else {
        return false;
    };
    let Some(scale) = tensors.get(d.t[4] as usize) else {
        return false;
    };
    output.bytes >= u64::from(d.i[1]) * 2
        && input.bytes >= u64::from(d.i[2]) * 2
        && weight.bytes >= u64::from(d.i[1]) * u64::from(d.i[2])
        && scale.bytes >= u64::from(d.i[1]) * 4
        && weight.name.starts_with("fp8/")
        && scale.name == format!("{}_scale", weight.name)
}

pub(crate) fn apply_output_object(
    model: &Model,
    sections: &mut Vec<SectionData>,
    profile: &str,
    output: &Path,
) -> Result<bool, String> {
    let profile = if profile == "sm_90a" {
        "sm90a"
    } else {
        profile
    };
    if profile != "sm90a" {
        return Ok(false);
    }
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    if !model.progs[..prefill_count]
        .iter()
        .enumerate()
        .any(|(index, program)| {
            model.prog_t[index] == 1 && program.insts.iter().any(|d| eligible(model, d))
        })
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
    let globals = [
        ("plow_w8a16_prefill_m1_abi", 1),
        ("plow_w8a16_prefill_m1_max_rows", 1),
        ("plow_w8a16_prefill_m1_k_multiple", 16),
        ("plow_block_pfgemm_w8a16_m1", 256),
        ("plow_arena_bytes_pfgemm_w8a16_m1", 1),
    ];
    if info.sm != 90
        || !info.entries.iter().any(|entry| entry == OBJECT_ENTRY)
        || globals
            .iter()
            .any(|&(name, value)| plow_asset::cubin::global_u32(&image, name) != Some(value))
    {
        return Err(format!(
            "{} has incompatible native W8A16 M1 capabilities",
            path.display()
        ));
    }

    let positions: Vec<_> = sections
        .iter()
        .enumerate()
        .filter(|(_, section)| section.name == SECTION)
        .map(|(index, _)| index)
        .collect();
    if positions.len() > 1
        || positions
            .first()
            .is_some_and(|&index| sections[index].kind != SECT_METADATA)
    {
        return Err("duplicate segment role metadata".into());
    }
    let mut metadata = if let Some(&index) = positions.first() {
        SegmentRoles::from_bytes(&sections[index].data)?
    } else {
        SegmentRoles {
            version: 1,
            objects: BTreeMap::new(),
            programs: Vec::new(),
        }
    };
    if metadata.objects.contains_key(&W8A16_PREFILL_M1) {
        return Err("native W8A16 M1 role already declared".into());
    }

    let mut selected = 0usize;
    for (index, program) in model.progs[..prefill_count].iter().enumerate() {
        if model.prog_t[index] != 1 {
            continue;
        }
        let prior = metadata
            .programs
            .iter()
            .position(|roles| roles.index == index);
        let mut roles = if let Some(position) = prior {
            metadata.programs[position].roles.clone()
        } else {
            vec![INTERPRETER; program.gq_seg_ofs.len() - 1]
        };
        if roles.len() + 1 != program.gq_seg_ofs.len() {
            return Err("existing prefill role window coverage".into());
        }
        for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
            if roles[segment] != INTERPRETER {
                continue;
            }
            let entries = &program.gq_stream[bounds[0] as usize..bounds[1] as usize];
            let pcs: BTreeSet<_> = entries.iter().map(|entry| entry.inst as usize).collect();
            if !pcs.is_empty()
                && pcs.iter().all(|&pc| eligible(model, &program.insts[pc]))
                && pcs.iter().all(|&pc| {
                    program
                        .gq_stream
                        .iter()
                        .all(|entry| entry.inst as usize != pc || entry.seg as usize == segment)
                        && program
                            .stream
                            .iter()
                            .all(|entry| entry.inst as usize != pc || entry.seg as usize == segment)
                })
            {
                roles[segment] = W8A16_PREFILL_M1;
                selected += 1;
            }
        }
        if roles.iter().any(|&role| role == W8A16_PREFILL_M1) {
            let record = ProgramRoles { index, roles };
            if let Some(position) = prior {
                metadata.programs[position] = record;
            } else {
                metadata.programs.push(record);
            }
        }
    }
    if selected == 0 {
        return Ok(false);
    }
    metadata.objects.insert(
        W8A16_PREFILL_M1,
        SegmentObject {
            abi: W8A16_PREFILL_M1_ABI.into(),
            file: OBJECT_FILE.into(),
            sha256: Some(plow_asset::decode_objects::image_sha256(&image)),
            promote_k512: None,
            attention: None,
        },
    );
    metadata.validate_schema()?;
    let section = SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&metadata).map_err(|error| error.to_string())?,
    };
    if let Some(&index) = positions.first() {
        sections[index] = section;
    } else {
        sections.push(section);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::DevInst;
    use packet::devbuild::TensorDecl;

    fn model() -> Model {
        Model {
            n_cu: 132,
            target: 0,
            tensors: [
                ("out", 3840 * 2),
                ("x", 15360 * 2),
                ("fp8/layer.down.weight", 3840 * 15360),
                ("unused", 1),
                ("fp8/layer.down.weight_scale", 3840 * 4),
            ]
            .into_iter()
            .map(|(name, bytes)| TensorDecl {
                name: name.into(),
                bytes,
                init: None,
            })
            .collect(),
            progs: Vec::new(),
            kv_row_insts: Vec::new(),
            prog_t: Vec::new(),
            gen: Vec::new(),
        }
    }

    fn down() -> DevInst {
        let mut d = DevInst::default();
        d.op = DevOp::GemmFp8 as u16;
        d.blocks = 264;
        d.t = [
            0,
            1,
            2,
            TENSOR_NONE,
            4,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
        ];
        d.i[..3].copy_from_slice(&[1, 3840, 15360]);
        d
    }

    #[test]
    fn selection_is_exact_to_qualified_m1_w8a16_cells() {
        let model = model();
        let d = down();
        assert!(eligible(&model, &d));
        for invalid in [
            {
                let mut x = d.clone();
                x.i[0] = 2;
                x
            },
            {
                let mut x = d.clone();
                x.i[1] = 1024;
                x
            },
            {
                let mut x = d.clone();
                x.t[4] = TENSOR_NONE;
                x
            },
        ] {
            assert!(!eligible(&model, &invalid));
        }
    }
}

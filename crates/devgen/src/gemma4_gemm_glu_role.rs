use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use packet::rope::GEN_TMAP_BF16;
use plow_asset::segment_roles::{
    ProgramRoles, SegmentObject, SegmentRoles, BF16_PREFILL_GEMM_GLU_GEMMA4,
    BF16_PREFILL_GEMM_GLU_GEMMA4_ABI, INTERPRETER, SECTION,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub(crate) const OBJECT_FILE: &str = "interp_sm90a_pfgemm_glu_gemma4.cubin";
const OBJECT_ENTRY: &str = "plow_sm90a_pfgemm_glu_gemma4";
const OBJECT_GLOBALS: [(&str, u32); 11] = [
    ("plow_pfgemm_glu_gemma4_abi", 1),
    ("plow_pfgemm_glu_gemma4_min_rows", 4096),
    ("plow_pfgemm_glu_gemma4_max_rows", 8192),
    ("plow_pfgemm_glu_gemma4_n", 15360),
    ("plow_pfgemm_glu_gemma4_k", 3840),
    ("plow_pfgemm_glu_gemma4_stages", 4),
    ("plow_pfgemm_glu_gemma4_bm", 128),
    ("plow_pfgemm_glu_gemma4_bn", 128),
    ("plow_pfgemm_glu_gemma4_bk", 64),
    ("plow_block_pfgemm_glu_gemma4", 384),
    ("plow_arena_bytes_pfgemm_glu_gemma4", 197696),
];

pub(crate) fn exact_shape(m: u32, n: u32, k: u32) -> bool {
    matches!(m, 4096 | 8192) && n == 15360 && k == 3840
}

pub(crate) fn select_fused(existing: bool, role: bool, m: u32, n: u32, k: u32) -> bool {
    existing || (role && exact_shape(m, n, k))
}

fn tmap_matches(model: &Model, handle: u32, target: u32, rows: u32, k: u32) -> bool {
    model.gen.iter().any(|map| {
        map.tensor == handle
            && map.kind == GEN_TMAP_BF16
            && map.aux == target
            && map.ctx == rows
            && map.hd == k
            && map.scale == 128
    }) && model
        .tensors
        .get(handle as usize)
        .is_some_and(|tensor| tensor.bytes == 128)
}

fn eligible(model: &Model, rows: u32, op: &packet::dev::DevInst) -> bool {
    if op.op != DevOp::GemmGlu as u16
        || u32::from(op.blocks) != model.n_cu
        || op.i[0] != rows
        || !exact_shape(op.i[0], op.i[1], op.i[2])
        || op.i[4] != 0
        || op.i[5] != 0
        || op.j != [0; 2]
        || op.f.iter().any(|value| value.to_bits() != 0)
        || [op.t[0], op.t[1], op.t[2], op.t[5]]
            .into_iter()
            .any(|tensor| tensor == TENSOR_NONE)
        || op.t[3] != TENSOR_NONE
        || op.t[4] != TENSOR_NONE
        || op.t[6] != TENSOR_NONE
        || op.t[7] != TENSOR_NONE
        || [op.t[0], op.t[1], op.t[2], op.t[5]]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .len()
            != 4
        || op.i[3] == 0
        || op.i[6] == 0
        || op.i[7] == 0
    {
        return false;
    }
    let bytes = |a: u32, b: u32| {
        u64::from(a)
            .checked_mul(u64::from(b))
            .and_then(|elements| elements.checked_mul(2))
    };
    [
        (op.t[0], bytes(op.i[0], op.i[1])),
        (op.t[1], bytes(op.i[0], op.i[2])),
        (op.t[2], bytes(op.i[1], op.i[2])),
        (op.t[5], bytes(op.i[1], op.i[2])),
    ]
    .into_iter()
    .all(|(handle, required)| {
        required.is_some_and(|bytes| {
            model
                .tensors
                .get(handle as usize)
                .is_some_and(|tensor| tensor.bytes >= bytes)
        })
    }) && tmap_matches(model, op.i[6], op.t[1], op.i[0], op.i[2])
        && tmap_matches(model, op.i[7], op.t[2], op.i[1], op.i[2])
        && tmap_matches(model, op.i[3], op.t[5], op.i[1], op.i[2])
}

pub(crate) fn apply_output_object(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    profile: &str,
    output: &Path,
) -> Result<usize, String> {
    let profile = if profile == "sm_90a" {
        "sm90a"
    } else {
        profile
    };
    if profile != "sm90a" {
        return Err("Gemma-4 BF16 GemmGlu role requires sm90a".into());
    }
    let path = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(OBJECT_FILE);
    let image = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let info = plow_asset::cubin::inspect(&image)
        .ok_or_else(|| format!("{} is not a valid cubin", path.display()))?;
    if info.sm != 90
        || !info.entries.iter().any(|entry| entry == OBJECT_ENTRY)
        || OBJECT_GLOBALS
            .iter()
            .any(|&(name, value)| plow_asset::cubin::global_u32(&image, name) != Some(value))
    {
        return Err(format!(
            "{} has incompatible Gemma-4 BF16 GemmGlu capabilities",
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
    if metadata.objects.contains_key(&BF16_PREFILL_GEMM_GLU_GEMMA4) {
        return Err("Gemma-4 BF16 GemmGlu role already declared".into());
    }

    struct Update {
        index: usize,
        roles: Vec<u8>,
        inst_segment: Vec<Option<u16>>,
        bounds: Vec<u32>,
        prior_position: Option<usize>,
    }
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    let mut updates = Vec::new();
    let mut selected = 0usize;
    for index in 0..prefill_count {
        let rows = model.prog_t[index];
        if !matches!(rows, 4096 | 8192) {
            continue;
        }
        let program = &model.progs[index];
        let eligible: Vec<_> = program
            .insts
            .iter()
            .map(|op| eligible(model, rows, op))
            .collect();
        if !eligible.iter().any(|&yes| yes) {
            return Err(format!(
                "prefill rung M{rows} has no compatible Gemma-4 BF16 GemmGlu instruction"
            ));
        }
        if program.l2_domains != 0 || program.hier_base != 0 {
            return Err("Gemma-4 BF16 GemmGlu role requires a plain prefill program".into());
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
        if !eligible.iter().enumerate().any(|(inst, &yes)| {
            yes && instruction_roles[inst].is_some_and(|role| role == INTERPRETER)
        }) {
            return Err(format!(
                "prefill rung M{rows} GemmGlu instruction is owned by another object role"
            ));
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
            let route = eligible[inst] && prior_role == INTERPRETER;
            let role = if route {
                BF16_PREFILL_GEMM_GLU_GEMMA4
            } else {
                prior_role
            };
            let key = if route {
                (u32::MAX, role, inst)
            } else {
                (u32::from(entry.seg), role, usize::MAX)
            };
            if last_key != Some(key) {
                if !roles.is_empty() {
                    bounds.push(queue_index as u32);
                }
                roles.push(role);
            }
            let segment = u16::try_from(roles.len() - 1)
                .map_err(|_| "too many Gemma-4 BF16 GemmGlu role segments")?;
            if inst_segment[inst].is_some_and(|prior| prior != segment) {
                return Err("Gemma-4 BF16 GemmGlu instruction is not contiguous".into());
            }
            inst_segment[inst] = Some(segment);
            last_key = Some(key);
        }
        bounds.push(program.gq_stream.len() as u32);
        selected += roles
            .iter()
            .filter(|&&role| role == BF16_PREFILL_GEMM_GLU_GEMMA4)
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
        return Err("packet has no M4096/M8192 Gemma-4 BF16 GemmGlu rungs".into());
    }

    metadata.objects.insert(
        BF16_PREFILL_GEMM_GLU_GEMMA4,
        SegmentObject {
            abi: BF16_PREFILL_GEMM_GLU_GEMMA4_ABI.into(),
            file: OBJECT_FILE.into(),
            sha256: Some(plow_asset::decode_objects::image_sha256(&image)),
            promote_k512: None,
            attention: None,
        },
    );
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
    if let Some(&index) = positions.first() {
        sections[index] = section;
    } else {
        sections.push(section);
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::Builder;
    use packet::rope::GenTensor;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn program(rows: u32, map_padding: usize) -> (packet::devbuild::Program, Vec<GenTensor>) {
        let mut builder = Builder::new(132);
        builder.force_uniseg();
        let out = builder.tensor("act.fu", 8192 * 15360 * 2);
        let input = builder.tensor("act.hn", 8192 * 3840 * 2);
        let gate = builder.tensor("model.layers.0.mlp.gate_proj.weight", 15360 * 3840 * 2);
        let up = builder.tensor("model.layers.0.mlp.up_proj.weight", 15360 * 3840 * 2);
        for index in 0..map_padding {
            let name = format!("tmap.padding.{index}");
            builder.tensor(&name, 128);
        }
        let ma = builder.tensor("tmap.a", 128);
        let mg = builder.tensor("tmap.g", 128);
        let mu = builder.tensor("tmap.u", 128);
        let before = builder.emit(DevOp::Nop, builder.all(), &[], |_| {});
        let glu = builder.emit(DevOp::GemmGlu, builder.all(), &[before], |op| {
            op.t = [
                out,
                input,
                gate,
                TENSOR_NONE,
                TENSOR_NONE,
                up,
                TENSOR_NONE,
                TENSOR_NONE,
            ];
            op.i = [rows, 15360, 3840, mu, 0, 0, ma, mg];
        });
        builder.emit(DevOp::Nop, builder.all(), &[glu], |_| {});
        let mut gen = [
            GenTensor::tmap_bf16(input, rows, 3840, 128),
            GenTensor::tmap_bf16(gate, 15360, 3840, 128),
            GenTensor::tmap_bf16(up, 15360, 3840, 128),
        ];
        for (map, handle) in gen.iter_mut().zip([ma, mg, mu]) {
            map.tensor = handle;
        }
        (builder.finish(), gen.into())
    }

    fn model() -> Model {
        let (p4, mut gen) = program(4096, 0);
        let (p8, gen8) = program(8192, 3);
        gen.extend(gen8);
        let (small, _) = program(2048, 0);
        let mut decode = Builder::new(132);
        decode.adopt_tensors(p8.tensors.clone());
        decode.emit(DevOp::Nop, decode.all(), &[], |_| {});
        Model {
            n_cu: 132,
            target: 0,
            tensors: p8.tensors.clone(),
            progs: vec![small, p4, p8, decode.finish()],
            kv_row_insts: vec![],
            prog_t: vec![2048, 4096, 8192, 1],
            gen,
        }
    }

    fn output_dir(label: &str) -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "plow-gemma4-gemm-glu-role-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn object_image(globals: &[(&str, u32)]) -> Vec<u8> {
        plow_asset::cubin::synthetic_elf(OBJECT_ENTRY, globals, 90)
    }

    #[test]
    fn exact_shape_contract_is_narrow() {
        for rows in [4096, 8192] {
            assert!(exact_shape(rows, 15360, 3840));
        }
        for shape in [
            (2048, 15360, 3840),
            (16384, 15360, 3840),
            (4096, 15359, 3840),
            (4096, 15360, 4096),
        ] {
            assert!(!exact_shape(shape.0, shape.1, shape.2));
        }
        assert!(!select_fused(false, false, 4096, 15360, 3840));
        assert!(select_fused(false, true, 4096, 15360, 3840));
        assert!(!select_fused(false, true, 2048, 15360, 3840));
        assert!(select_fused(true, true, 2048, 15360, 3840));
    }

    #[test]
    fn qualified_object_routes_only_exact_rungs_and_preserves_scheduling() {
        let directory = output_dir("exact");
        let output = directory.join("model.pkt");
        let image = object_image(&OBJECT_GLOBALS);
        std::fs::write(directory.join(OBJECT_FILE), &image).unwrap();
        let mut model = model();
        assert_eq!(model.progs[1].insts[1].blocks, 132);
        let original: Vec<_> = model
            .progs
            .iter()
            .map(|program| {
                (
                    program.insts.clone(),
                    program.waits.clone(),
                    program.succs.clone(),
                    program.n_counter,
                    program.stream_ofs.clone(),
                    program.stream_len.clone(),
                    program.stream.clone(),
                    program.gq_stream.clone(),
                )
            })
            .collect();
        let mut sections = vec![SectionData {
            kind: SECT_METADATA,
            name: "packed_prefill.json".into(),
            data: b"packed scheduling marker".to_vec(),
        }];
        assert_eq!(
            apply_output_object(&mut model, &mut sections, "sm_90a", &output).unwrap(),
            2
        );
        assert_eq!(sections[0].data, b"packed scheduling marker");
        let role_section = sections.iter().find(|s| s.name == SECTION).unwrap();
        let roles = SegmentRoles::from_bytes(&role_section.data).unwrap();
        assert_eq!(
            roles.objects[&BF16_PREFILL_GEMM_GLU_GEMMA4].sha256,
            Some(plow_asset::decode_objects::image_sha256(&image))
        );
        assert!(!roles.programs.iter().any(|record| record.index == 0));
        for (index, old) in original.iter().enumerate() {
            let new = &model.progs[index];
            assert_eq!(new.insts, old.0);
            assert_eq!(new.waits, old.1);
            assert_eq!(new.succs, old.2);
            assert_eq!(new.n_counter, old.3);
            assert_eq!(new.stream_ofs, old.4);
            assert_eq!(new.stream_len, old.5);
            for (a, b) in new
                .stream
                .iter()
                .zip(&old.6)
                .chain(new.gq_stream.iter().zip(&old.7))
            {
                let mut a = *a;
                a.seg = b.seg;
                assert_eq!(&a, b);
            }
        }
        for index in [1, 2] {
            let record = roles
                .programs
                .iter()
                .find(|record| record.index == index)
                .unwrap();
            assert_eq!(record.roles, [0, BF16_PREFILL_GEMM_GLU_GEMMA4, 0]);
            assert_eq!(model.progs[index].gq_seg_ofs.len(), 4);
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn missing_or_incompatible_object_cannot_mutate_the_packet() {
        let directory = output_dir("reject");
        let output = directory.join("model.pkt");
        let mut model = model();
        let before = model.to_blob();
        let mut sections = Vec::new();
        assert!(apply_output_object(&mut model, &mut sections, "sm90a", &output).is_err());
        assert_eq!(model.to_blob(), before);
        assert!(sections.is_empty());

        let mut globals = OBJECT_GLOBALS;
        globals[5].1 = 3;
        std::fs::write(directory.join(OBJECT_FILE), object_image(&globals)).unwrap();
        assert!(apply_output_object(&mut model, &mut sections, "sm90a", &output).is_err());
        assert_eq!(model.to_blob(), before);
        assert!(sections.is_empty());
        std::fs::remove_dir_all(directory).unwrap();
    }
}

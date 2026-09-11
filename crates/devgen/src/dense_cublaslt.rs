use packet::dev::DevOp;
use packet::devbuild::{Builder, Model, SectionData, SECT_METADATA};
use plow_asset::segment_roles::{ProgramRoles, SegmentRoles, CUBLASLT, INTERPRETER, SECTION};

pub(crate) fn apply(model: &mut Model) -> Result<SectionData, String> {
    apply_projections(model, false)
}

pub(crate) fn apply_prefill(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    profile: &str,
) -> Result<usize, String> {
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
            objects: Default::default(),
            programs: Vec::new(),
        }
    };

    struct Update {
        index: usize,
        roles: Vec<u8>,
        inst_segment: Vec<Option<u16>>,
        bounds: Vec<u32>,
        prior_position: Option<usize>,
    }
    let mut updates = Vec::new();
    let mut selected = 0usize;
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    for index in 0..prefill_count {
        let rows = model.prog_t[index];
        let program = &model.progs[index];
        let eligible: Vec<_> = program
            .insts
            .iter()
            .map(|op| prefill_eligible(model, op, rows, profile))
            .collect();
        if !eligible.iter().any(|&yes| yes) {
            continue;
        }
        if program.l2_domains != 0 || program.hier_base != 0 {
            return Err("cuBLASLt prefill requires coarse single-domain packets".into());
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
            let route = eligible[inst] && prior_role == INTERPRETER;
            let role = if route { CUBLASLT } else { prior_role };
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
            let segment =
                u16::try_from(roles.len() - 1).map_err(|_| "too many cuBLASLt prefill segments")?;
            if inst_segment[inst].is_some_and(|prior| prior != segment) {
                return Err("cuBLASLt prefill instruction is not contiguous in the queue".into());
            }
            inst_segment[inst] = Some(segment);
            last_key = Some(key);
        }
        bounds.push(program.gq_stream.len() as u32);
        selected += roles.iter().filter(|&&role| role == CUBLASLT).count();
        updates.push(Update {
            index,
            roles,
            inst_segment,
            bounds,
            prior_position,
        });
    }
    if selected == 0 {
        return Ok(0);
    }
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
    Ok(selected)
}

fn prefill_eligible(model: &Model, op: &packet::dev::DevInst, rows: u32, profile: &str) -> bool {
    if !matches!(
        DevOp::from_u16(op.op),
        Some(DevOp::Gemm | DevOp::GemmMed | DevOp::GemmSmall)
    ) || op.blocks == 0
        || op.i[0] != rows
        || !plow_asset::segment_roles::cublaslt_prefill_bf16(profile, op.i[0], op.i[1], op.i[2])
        || op.i[3..6].iter().any(|&value| value != 0)
        || op.t[3..]
            .iter()
            .any(|&tensor| tensor != packet::dev::TENSOR_NONE)
        || [op.t[0], op.t[1], op.t[2]]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != 3
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
    ]
    .into_iter()
    .all(|(handle, required)| {
        required.is_some_and(|bytes| {
            model
                .tensors
                .get(handle as usize)
                .is_some_and(|tensor| tensor.bytes >= bytes)
        })
    })
}

fn apply_projections(model: &mut Model, head: bool) -> Result<SectionData, String> {
    let mut programs = Vec::new();
    for index in packet::devbuild::decode_rung_lo(&model.prog_t)..model.progs.len() {
        let roles = apply_program(model, index, head)?;
        programs.push(serde_json::json!({"index": index, "roles": roles}));
    }
    Ok(SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&serde_json::json!({
            "version": 1, "objects": {}, "programs": programs
        }))
        .map_err(|e| e.to_string())?,
    })
}

pub(crate) fn apply_native(
    model: &mut Model,
    output: &std::path::Path,
) -> Result<SectionData, String> {
    let file = "gemv_sm90_transposed.cubin";
    if model.prog_t[packet::devbuild::decode_rung_lo(&model.prog_t)..]
        .iter()
        .any(|&m| ![1, 2, 4, 8, 16, 32].contains(&m))
    {
        return Err("native tensor-core decode requires measured B1/B2/B4/B8/B16/B32 rungs".into());
    }
    let path = output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(file);
    let image = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if plow_asset::cubin::global_u32(&image, "plow_gemv_transposed_abi") != Some(1)
        || plow_asset::cubin::global_u32(&image, "plow_gemv_transposed_block") != Some(128)
        || plow_asset::cubin::inspect(&image).is_none_or(|i| i.sm != 90)
    {
        return Err("native decode object requires transposed ABI1".into());
    }
    if model.prog_t[packet::devbuild::decode_rung_lo(&model.prog_t)..]
        .iter()
        .any(|&m| m == 32)
        && plow_asset::cubin::global_u32(&image, "plow_gemv_transposed_max_rows") != Some(32)
    {
        return Err("native decode object has no B32 capability".into());
    }
    let mut section = apply_projections(model, true)?;
    let mut roles = plow_asset::segment_roles::SegmentRoles::from_bytes(&section.data)?;
    for p in &mut roles.programs {
        for role in &mut p.roles {
            if *role == CUBLASLT {
                *role = plow_asset::segment_roles::NATIVE_DECODE_TC;
            }
        }
    }
    roles.objects.insert(
        plow_asset::segment_roles::NATIVE_DECODE_TC,
        plow_asset::segment_roles::SegmentObject {
            abi: "gemv_transposed_sm90_bf16_v1".into(),
            file: file.into(),
            sha256: Some(plow_asset::decode_objects::image_sha256(&image)),
            promote_k512: None,
            attention: None,
        },
    );
    roles.validate_schema()?;
    section.data = serde_json::to_vec(&roles).map_err(|e| e.to_string())?;
    Ok(section)
}

fn apply_program(model: &mut Model, index: usize, head: bool) -> Result<Vec<u8>, String> {
    let dependencies = plow_asset::program::with_model(model, |packet| {
        plow_asset::splitk::dependencies(&packet.programs[index])
    })?;
    let old = &model.progs[index];
    let mut placement: Vec<_> = old
        .insts
        .iter()
        .map(|d| vec![u32::MAX; d.blocks as usize])
        .collect();
    for cu in 0..model.n_cu as usize {
        let offset = old.stream_ofs[cu] as usize;
        for entry in &old.stream[offset..offset + old.stream_len[cu] as usize] {
            placement[entry.inst as usize][entry.slice as usize] = cu as u32;
        }
    }
    let mut builder = Builder::new(model.n_cu);
    builder.force_uniseg();
    builder.adopt_tensors(model.tensors.clone());
    let mut projections = Vec::new();
    for (pc, inst) in old.insts.iter().enumerate() {
        let selected = inst.op == DevOp::Gemv as u16 && {
            let name = &model.tensors[inst.t[2] as usize].name;
            name.contains(".layers.")
                || (head
                    && (name.ends_with("lm_head.weight") || name.ends_with("embed_tokens.weight")))
        };
        if selected
            && (!(1..=32).contains(&inst.i[0])
                || inst.i[0] != model.prog_t[index]
                || inst.i[1] == 0
                || inst.i[2] == 0
                || inst.i[3..].iter().any(|&v| v != 0))
        {
            return Err("cuBLASLt requires ordinary BF16 projection operands".into());
        }
        let deps: Vec<_> = dependencies[pc].iter().map(|&d| d as u32).collect();
        let counter = builder.emit(
            DevOp::from_u16(inst.op).ok_or("unknown dense opcode")?,
            placement[pc].clone(),
            &deps,
            |d| {
                d.t = inst.t;
                d.i = inst.i;
                d.f = inst.f;
                d.j = inst.j;
            },
        );
        assert_eq!(counter as usize, pc);
        if selected {
            builder.isolate(counter);
            projections.push(pc);
        }
    }
    if projections.is_empty() {
        return Err("no eligible dense decode projections".into());
    }
    let program = builder.finish();
    if !old
        .insts
        .iter()
        .zip(&program.insts)
        .all(|(a, b)| a.pack() == b.pack())
    {
        return Err("cuBLASLt segmentation changed instruction operands".into());
    }
    let mut roles = vec![INTERPRETER; program.gq_seg_ofs.len() - 1];
    for pc in projections {
        let segment = program
            .stream
            .iter()
            .find(|e| e.inst as usize == pc)
            .ok_or("missing projection stream")?
            .seg as usize;
        if !program
            .stream
            .iter()
            .all(|e| (e.inst as usize == pc) == (e.seg as usize == segment))
        {
            return Err("projection was not isolated".into());
        }
        roles[segment] = CUBLASLT;
    }
    model.progs[index] = program;
    Ok(roles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Model {
        model_rows(&[128, 8])
    }

    fn model_rows(widths: &[u32]) -> Model {
        let mut programs = Vec::new();
        let mut tensors = Vec::new();
        for &rows in widths {
            let mut b = Builder::new(2);
            b.force_uniseg();
            let input = b.tensor("act.x", 128 * 64 * 2);
            let weight = b.tensor("model.layers.0.mlp.down_proj.weight", 64 * 64 * 2);
            let output = b.tensor("act.y", 128 * 64 * 2);
            let head = b.tensor("lm_head.weight", 64 * 64 * 2);
            let ready = b.emit(DevOp::Nop, vec![0, 1], &[], |_| {});
            let projection = b.emit(DevOp::Gemv, vec![0, 1], &[ready], |d| {
                d.t[..3].copy_from_slice(&[output, input, weight]);
                d.i[..3].copy_from_slice(&[rows, 64, 64]);
            });
            b.emit(DevOp::Gemv, vec![0, 1], &[projection], |d| {
                d.t[..3].copy_from_slice(&[input, output, head]);
                d.i[..3].copy_from_slice(&[rows, 64, 64]);
            });
            tensors = b.tensors();
            programs.push(b.finish());
        }
        Model {
            n_cu: 2,
            target: 0,
            tensors,
            progs: programs,
            prog_t: widths.to_vec(),
            gen: Vec::new(),
            kv_row_insts: vec![1],
        }
    }

    fn prefill_model() -> Model {
        let widths = [1, 64, 128, 256, 512, 1024, 1];
        let mut model = model_rows(&widths);
        for program in &mut model.progs[..6] {
            let rows = program.insts[1].i[0];
            program.insts[1].op = DevOp::Gemm as u16;
            program.insts[1].i[..].copy_from_slice(&[rows, 3840, 15360, 0, 0, 0, 7, 8]);
            program.insts[2].op = DevOp::GemmMed as u16;
            program.insts[2].i[..].copy_from_slice(&[rows, 3840, 8192, 0, 0, 0, 9, 10]);
        }
        let extent = 3840u64 * 15360 * 2;
        for tensor in &mut model.tensors {
            tensor.bytes = extent;
        }
        model
    }

    #[test]
    fn isolates_body_projection_without_changing_math_or_dependencies() {
        let mut m = model();
        let prefill = m.progs[0].to_blob();
        let insts: Vec<_> = m.progs[1].insts.iter().map(|d| d.pack()).collect();
        let deps = plow_asset::program::with_model(&m, |p| {
            plow_asset::splitk::dependencies(&p.programs[1])
        })
        .unwrap();
        let metadata = apply(&mut m).unwrap();
        let roles = plow_asset::segment_roles::SegmentRoles::from_bytes(&metadata.data).unwrap();
        assert_eq!(roles.programs[0].index, 1);
        assert_eq!(
            roles.programs[0].roles,
            [INTERPRETER, CUBLASLT, INTERPRETER]
        );
        assert_eq!(m.progs[0].to_blob(), prefill);
        assert_eq!(
            m.progs[1]
                .insts
                .iter()
                .map(|d| d.pack())
                .collect::<Vec<_>>(),
            insts
        );
        assert_eq!(m.kv_row_insts, [1]);
        // The dependency checker accepts one window; erase only the segment labels.
        let program = &mut m.progs[1];
        program.gq_seg_ofs = vec![0, program.gq_stream.len() as u32];
        for entry in program.stream.iter_mut().chain(&mut program.gq_stream) {
            entry.seg = 0;
        }
        let after = plow_asset::program::with_model(&m, |p| {
            plow_asset::splitk::dependencies(&p.programs[1])
        })
        .unwrap();
        assert_eq!(after, deps);
    }

    #[test]
    fn native_object_is_hash_bound_and_does_not_declare_library_roles() {
        let directory =
            std::env::temp_dir().join(format!("plow-native-role-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let image = plow_asset::cubin::synthetic_elf(
            "plow_gemv_bf16_m8_bk128_s3",
            &[
                ("plow_gemv_transposed_abi", 1),
                ("plow_gemv_transposed_block", 128),
            ],
            90,
        );
        std::fs::write(directory.join("gemv_sm90_transposed.cubin"), &image).unwrap();
        let mut m = model();
        let section = apply_native(&mut m, &directory.join("model.pkt")).unwrap();
        let metadata = plow_asset::segment_roles::SegmentRoles::from_bytes(&section.data).unwrap();
        assert_eq!(
            metadata.programs[0].roles,
            [
                0,
                plow_asset::segment_roles::NATIVE_DECODE_TC,
                plow_asset::segment_roles::NATIVE_DECODE_TC
            ]
        );
        assert_eq!(
            metadata.objects[&plow_asset::segment_roles::NATIVE_DECODE_TC].sha256,
            Some(plow_asset::decode_objects::image_sha256(&image))
        );
        let mut wide = model_rows(&[128, 1, 16, 32]);
        assert!(apply_native(&mut wide, &directory.join("model.pkt"))
            .err()
            .expect("missing B32 capability")
            .contains("B32 capability"));
        let mut prefill_32 = model_rows(&[32, 128, 1, 16]);
        assert!(apply_native(&mut prefill_32, &directory.join("model.pkt")).is_ok());
        let wide_image = plow_asset::cubin::synthetic_elf(
            "plow_gemv_bf16_m32_bk128_s3",
            &[
                ("plow_gemv_transposed_abi", 1),
                ("plow_gemv_transposed_block", 128),
                ("plow_gemv_transposed_max_rows", 32),
            ],
            90,
        );
        std::fs::write(directory.join("gemv_sm90_transposed.cubin"), &wide_image).unwrap();
        assert!(apply_native(&mut wide, &directory.join("model.pkt")).is_ok());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn native_decode_rejects_unmeasured_batch_before_object_loading() {
        let mut m = model_rows(&[128, 64]);
        let error = apply_native(&mut m, std::path::Path::new("missing/model.pkt"))
            .err()
            .expect("unmeasured batch");
        assert!(error.contains("measured B1/B2/B4/B8/B16"));
    }

    #[test]
    fn emits_each_decode_width_and_rejects_nonordinary_operands() {
        let mut ladder = model_rows(&[128, 1, 4, 8]);
        let metadata = apply(&mut ladder).unwrap();
        let metadata = plow_asset::segment_roles::SegmentRoles::from_bytes(&metadata.data).unwrap();
        assert_eq!(metadata.programs.len(), 3);
        for (index, program) in metadata.programs.iter().enumerate() {
            assert_eq!(program.index, index + 1);
            assert_eq!(program.roles, [INTERPRETER, CUBLASLT, INTERPRETER]);
        }
        let mut m = model();
        m.prog_t = vec![1, 8];
        assert!(apply(&mut m).err().unwrap().contains("ordinary BF16"));
        let mut m = model();
        m.progs[1].insts[1].i[4] = 1;
        assert!(apply(&mut m).err().unwrap().contains("ordinary BF16"));
    }

    #[test]
    fn prefill_policy_isolates_two_exact_shapes_for_qualified_rungs() {
        let mut model = prefill_model();
        let decode = model.progs[6].to_blob();
        let insts: Vec<_> = model
            .progs
            .iter()
            .map(|program| program.insts.iter().map(|op| op.pack()).collect::<Vec<_>>())
            .collect();
        let mut sections = Vec::new();
        assert_eq!(
            apply_prefill(&mut model, &mut sections, "sm_90a").unwrap(),
            6
        );
        assert_eq!(model.progs[6].to_blob(), decode);
        assert_eq!(sections.len(), 1);
        let metadata = SegmentRoles::from_bytes(&sections[0].data).unwrap();
        assert_eq!(metadata.programs.len(), 3);
        for roles in &metadata.programs {
            let index = roles.index;
            assert!(matches!(model.prog_t[index], 128 | 256 | 512));
            assert_eq!(
                roles.roles.iter().filter(|&&role| role == CUBLASLT).count(),
                2
            );
            for (pc, op) in model.progs[index].insts.iter().enumerate() {
                assert_eq!(op.pack(), insts[index][pc]);
                let segments: std::collections::BTreeSet<_> = model.progs[index]
                    .gq_stream
                    .iter()
                    .filter(|entry| entry.inst as usize == pc)
                    .map(|entry| entry.seg as usize)
                    .collect();
                assert_eq!(segments.len(), 1);
                let role = roles.roles[*segments.iter().next().unwrap()];
                assert_eq!(role == CUBLASLT, pc == 1 || pc == 2);
            }
        }
    }

    #[test]
    fn prefill_policy_is_opt_in_exact_and_preserves_existing_roles() {
        let mut model = prefill_model();
        model.progs[2].insts[1].op = DevOp::GemmFp8 as u16;
        model.progs[3].insts[1].i[1] = 4096;
        model.progs[4].insts[1].t[7] = 0;
        let prior = ProgramRoles {
            index: 1,
            roles: vec![plow_asset::segment_roles::PREFILL_ATTENTION],
        };
        let mut sections = vec![SectionData {
            kind: SECT_METADATA,
            name: SECTION.into(),
            data: serde_json::to_vec(&SegmentRoles {
                version: 1,
                objects: [(
                    plow_asset::segment_roles::PREFILL_ATTENTION,
                    plow_asset::segment_roles::SegmentObject {
                        abi: "attention_sm90_hd256_v1".into(),
                        file: "attention.cubin".into(),
                        sha256: None,
                        promote_k512: None,
                        attention: None,
                    },
                )]
                .into_iter()
                .collect(),
                programs: vec![prior],
            })
            .unwrap(),
        }];
        let selected = apply_prefill(&mut model, &mut sections, "sm90a").unwrap();
        assert_eq!(selected, 3);
        let metadata = SegmentRoles::from_bytes(&sections[0].data).unwrap();
        assert_eq!(
            metadata
                .programs
                .iter()
                .find(|p| p.index == 1)
                .unwrap()
                .roles,
            [plow_asset::segment_roles::PREFILL_ATTENTION]
        );

        let mut unsupported = prefill_model();
        let mut none = Vec::new();
        assert_eq!(
            apply_prefill(&mut unsupported, &mut none, "sm120").unwrap(),
            0
        );
        assert!(none.is_empty());
    }
}

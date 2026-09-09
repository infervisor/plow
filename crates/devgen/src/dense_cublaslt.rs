use packet::dev::DevOp;
use packet::devbuild::{Builder, Model, SectionData, SECT_METADATA};
use plow_asset::segment_roles::{CUBLASLT, INTERPRETER, SECTION};

pub(crate) fn apply(model: &mut Model) -> Result<SectionData, String> {
    let index = packet::devbuild::decode_rung_lo(&model.prog_t);
    if index + 1 != model.progs.len() {
        return Err("Gemma cuBLASLt currently requires one explicit decode rung".into());
    }
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
        let selected = inst.op == DevOp::Gemv as u16
            && model.tensors[inst.t[2] as usize].name.contains(".layers.");
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
    Ok(SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&serde_json::json!({
            "version": 1, "objects": {}, "programs": [{"index": index, "roles": roles}]
        }))
        .map_err(|e| e.to_string())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Model {
        let mut programs = Vec::new();
        let mut tensors = Vec::new();
        for rows in [128, 8] {
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
            prog_t: vec![128, 8],
            gen: Vec::new(),
            kv_row_insts: vec![1],
        }
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
    fn rejects_unqualified_ladders_and_nonordinary_operands() {
        let mut m = model();
        m.prog_t = vec![1, 8];
        assert!(apply(&mut m)
            .err()
            .unwrap()
            .contains("one explicit decode rung"));
        let mut m = model();
        m.progs[1].insts[1].i[4] = 1;
        assert!(apply(&mut m).err().unwrap().contains("ordinary BF16"));
    }
}

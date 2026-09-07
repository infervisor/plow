use packet::dev::{DevOp, SE_FINE, SE_XCTR};
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use std::path::Path;

const OBJECT_FILE: &str = "interp_sm90a_mxfp4_moe.cubin";
const OBJECT_ENTRY: &str = "plow_sm90a_mxfp4_moe";
const OBJECT_GLOBALS: [(&str, u32); 4] = [
    ("plow_mxfp4_moe_sm90_abi", 1),
    ("plow_block_mxfp4_moe", 256),
    ("plow_arena_bytes_mxfp4_moe", 4),
    ("plow_mxfp4_moe_ctas_per_sm", 4),
];

pub(crate) fn apply_output_object(
    model: &mut Model,
    profile: &str,
    output: &Path,
) -> Result<Option<SectionData>, String> {
    let decode = packet::devbuild::decode_rung_lo(&model.prog_t);
    if profile != "sm_90a" || model.prog_t[decode..] != [1] {
        return Ok(None);
    }
    let path = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(OBJECT_FILE);
    let image = match std::fs::read(&path) {
        Ok(image) => image,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let info = plow_asset::cubin::inspect(&image)
        .ok_or_else(|| format!("{} is not a valid cubin", path.display()))?;
    if info.sm != 90
        || !info.entries.iter().any(|entry| entry == OBJECT_ENTRY)
        || OBJECT_GLOBALS
            .iter()
            .any(|&(name, value)| plow_asset::cubin::global_u32(&image, name) != Some(value))
    {
        return Err(format!(
            "{} has incompatible MXFP4 MoE capabilities",
            path.display()
        ));
    }
    Ok(Some(apply(
        model,
        plow_asset::decode_objects::image_sha256(&image),
    )))
}

fn apply(model: &mut Model, sha256: String) -> SectionData {
    let index = model.progs.len() - 1;
    assert_eq!(model.prog_t[index], 1, "MXFP4 MoE role requires M1 decode");
    assert!(model.prog_t[..index].iter().all(|&t| t > 1));
    let program = &mut model.progs[index];
    assert!(program.l2_domains == 0 && program.hier_base == 0 && program.gq_seg_ofs.len() == 2);
    assert!(program
        .stream
        .iter()
        .all(|entry| entry.seg == 0 && entry.flags & (SE_FINE | SE_XCTR) == 0));
    assert!(program
        .gq_stream
        .windows(2)
        .all(|pair| pair[0].inst <= pair[1].inst));

    let mut roles = Vec::new();
    let mut segments = Vec::new();
    for inst in &program.insts {
        let role = if matches!(
            DevOp::from_u16(inst.op),
            Some(DevOp::MoeGluMx | DevOp::MoeDownMx)
        ) {
            plow_asset::segment_roles::MXFP4_MOE
        } else {
            plow_asset::segment_roles::INTERPRETER
        };
        if roles.is_empty()
            || role != plow_asset::segment_roles::INTERPRETER
            || roles.last() != Some(&plow_asset::segment_roles::INTERPRETER)
        {
            roles.push(role);
        }
        segments.push(u16::try_from(roles.len() - 1).expect("too many decode segments"));
    }
    assert!(
        roles.contains(&plow_asset::segment_roles::MXFP4_MOE),
        "no MXFP4 MoE decode instructions"
    );
    for entry in program.stream.iter_mut().chain(&mut program.gq_stream) {
        entry.seg = segments[entry.inst as usize];
    }
    program.gq_seg_ofs = vec![0];
    for (index, pair) in program.gq_stream.windows(2).enumerate() {
        if pair[0].seg != pair[1].seg {
            program.gq_seg_ofs.push((index + 1) as u32);
        }
    }
    program.gq_seg_ofs.push(program.gq_stream.len() as u32);
    assert_eq!(program.gq_seg_ofs.len(), roles.len() + 1);

    SectionData {
        kind: SECT_METADATA,
        name: plow_asset::segment_roles::SECTION.into(),
        data: serde_json::to_vec(&serde_json::json!({
            "version": 1,
            "objects": {
                plow_asset::segment_roles::MXFP4_MOE.to_string(): {
                    "abi": plow_asset::segment_roles::MXFP4_MOE_ABI,
                    "file": OBJECT_FILE,
                    "sha256": sha256
                }
            },
            "programs": [{"index": index, "roles": roles}]
        }))
        .unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::Builder;

    #[test]
    fn isolates_each_expert_projection_without_changing_work() {
        let mut builder = Builder::new(132);
        builder.force_uniseg();
        let a = builder.emit(DevOp::Nop, builder.all(), &[], |_| {});
        let glu = builder.emit(DevOp::MoeGluMx, builder.all(), &[a], |d| {
            d.i = [4, 2880, 2880, 32, 0, 3, 1, 0];
        });
        let down = builder.emit(DevOp::MoeDownMx, builder.all(), &[glu], |d| {
            d.i = [4, 2880, 2880, 32, 0, 0, 1, 0];
        });
        builder.emit(DevOp::Nop, builder.all(), &[down], |_| {});
        let original = builder.finish();
        let original_insts = original.insts.clone();
        let original_waits = original.waits.clone();
        let original_succs = original.succs.clone();
        let original_n_counter = original.n_counter;
        let original_stream = original.stream.clone();
        let original_gq_stream = original.gq_stream.clone();
        let original_stream_ofs = original.stream_ofs.clone();
        let original_stream_len = original.stream_len.clone();
        let mut model = Model {
            n_cu: 132,
            target: 0,
            tensors: Vec::new(),
            progs: vec![original],
            kv_row_insts: Vec::new(),
            prog_t: vec![1],
            gen: Vec::new(),
        };
        let section = apply(&mut model, "a".repeat(64));
        let program = &model.progs[0];
        assert_eq!(program.insts, original_insts);
        assert_eq!(program.waits, original_waits);
        assert_eq!(program.succs, original_succs);
        assert_eq!(program.n_counter, original_n_counter);
        assert_eq!(program.stream_ofs, original_stream_ofs);
        assert_eq!(program.stream_len, original_stream_len);
        for (actual, expected) in program
            .stream
            .iter()
            .zip(&original_stream)
            .chain(program.gq_stream.iter().zip(&original_gq_stream))
        {
            let mut actual = *actual;
            actual.seg = 0;
            assert_eq!(actual, *expected);
        }
        let metadata = plow_asset::segment_roles::SegmentRoles::from_bytes(&section.data).unwrap();
        assert_eq!(metadata.programs[0].roles, [0, 7, 7, 0]);
        assert_eq!(program.gq_seg_ofs, [0, 132, 264, 396, 528]);
    }

    #[test]
    fn object_presence_selects_and_binds_the_exact_image() {
        let mut builder = Builder::new(132);
        builder.force_uniseg();
        let x = builder.tensor("x", 2880 * 2);
        let idx = builder.tensor("idx", 4 * 8);
        let w = builder.tensor("w", 32 * 2 * 2880 * 2880 / 2);
        let s = builder.tensor("s", 32 * 2 * 2880 * 2880 / 32);
        let out = builder.tensor("out", 4 * 2880 * 2);
        builder.emit(DevOp::MoeGluMx, builder.all(), &[], |d| {
            d.t[..5].copy_from_slice(&[out, x, idx, w, s]);
            d.i = [4, 2880, 2880, 32, 0, 3, 1, 0];
        });
        let program = builder.finish();
        let mut model = Model {
            n_cu: 132,
            target: 0,
            tensors: program.tensors.clone(),
            progs: vec![program],
            kv_row_insts: Vec::new(),
            prog_t: vec![1],
            gen: Vec::new(),
        };
        let directory =
            std::env::temp_dir().join(format!("plow-mxfp4-role-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let output = directory.join("model.pkt");
        let before = model.to_blob();
        assert!(apply_output_object(&mut model, "sm_90a", &output)
            .unwrap()
            .is_none());
        assert_eq!(model.to_blob(), before);

        let image = plow_asset::cubin::synthetic_elf(OBJECT_ENTRY, &OBJECT_GLOBALS, 90);
        std::fs::write(directory.join(OBJECT_FILE), &image).unwrap();
        let section = apply_output_object(&mut model, "sm_90a", &output)
            .unwrap()
            .unwrap();
        let metadata = plow_asset::segment_roles::SegmentRoles::from_bytes(&section.data).unwrap();
        assert_eq!(
            metadata.objects[&plow_asset::segment_roles::MXFP4_MOE]
                .sha256
                .as_deref(),
            Some(plow_asset::decode_objects::image_sha256(&image).as_str())
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}

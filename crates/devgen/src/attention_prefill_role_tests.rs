use super::*;
use packet::devbuild::Builder;
use std::sync::atomic::{AtomicU64, Ordering};

fn fixture(hd: u32, with_map: bool, fused: bool) -> Model {
    let mut builder = Builder::new(132);
    let q = builder.tensor("q", 128 * 32 * u64::from(hd) * 2);
    let k = builder.tensor("k", 4 * 65536 * u64::from(hd) * 2);
    let v = builder.tensor("v", 4 * 65536 * u64::from(hd) * 2);
    let partial = builder.tensor("partial", 128 * 32 * u64::from(hd) * 4);
    let ml = builder.tensor("ml", 128 * 32 * 8);
    let out = builder.tensor("out", 128 * 32 * u64::from(hd) * 2);
    let map = builder.tensor("map", 256);
    let before = builder.emit(DevOp::Nop, builder.all(), &[], |_| {});
    let flash = builder.emit(DevOp::FlashPrefill, builder.all(), &[before], |op| {
        op.t[..5].copy_from_slice(&[partial, ml, q, k, v]);
        op.t[5] = if fused { out } else { TENSOR_NONE };
        op.t[7] = if with_map { map } else { TENSOR_NONE };
        op.i = [128, 128, 32, 4, 0, 0, hd, 1];
        op.j[0] = 65536;
        op.j[1] = u32::MAX;
        op.f[0] = 1.0;
    });
    let after = if fused {
        flash
    } else {
        builder.emit(DevOp::FlashMerge, builder.all(), &[flash], |op| {
            op.t[..3].copy_from_slice(&[out, partial, ml]);
            op.i[..4].copy_from_slice(&[128, 32, 1, hd]);
        })
    };
    builder.emit(DevOp::Nop, builder.all(), &[after], |_| {});
    let prefill = builder.finish();
    let mut decode = Builder::new(132);
    decode.adopt_tensors(prefill.tensors.clone());
    decode.emit(DevOp::Nop, decode.all(), &[], |_| {});
    Model {
        n_cu: 132,
        target: 0,
        tensors: prefill.tensors.clone(),
        progs: vec![prefill, decode.finish()],
        kv_row_insts: vec![],
        prog_t: vec![128, 1],
        gen: vec![],
    }
}

fn selection() -> Selection {
    Selection::from_image("attention.cubin".into(), b"cubin")
}

fn output_dir(label: &str) -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "plow-attention-role-{label}-{}-{}",
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
fn isolates_only_compatible_hd512_instructions_and_binds_hash() {
    let mut model = fixture(512, true, false);
    let original_insts = model.progs[0].insts.clone();
    let mut sections = Vec::new();
    apply(&mut model, &mut sections, &selection(), "sm90a").unwrap();
    assert_eq!(model.progs[0].insts, original_insts);
    let roles = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    let program = &roles.programs[0];
    assert_eq!(program.roles, [0, 6, 0]);
    assert_eq!(model.progs[0].gq_seg_ofs.len(), program.roles.len() + 1);
    let object = &roles.objects[&PREFILL_ATTENTION_HD512_WG32];
    assert_eq!(
        object.sha256.as_deref(),
        Some(plow_asset::decode_objects::image_sha256(b"cubin").as_str())
    );
    assert_eq!(object.attention.as_ref(), Some(&capability()));
}

#[test]
fn stays_inert_without_apply_and_rejects_incompatible_geometry() {
    let model = fixture(512, true, false);
    assert_eq!(model.to_blob(), model.to_blob());
    for mut model in [fixture(256, true, false), fixture(512, false, false)] {
        assert!(apply(&mut model, &mut Vec::new(), &selection(), "sm90a").is_err());
    }
    let mut model = fixture(512, true, false);
    model.tensors.last_mut().unwrap().bytes = 128;
    assert!(apply(&mut model, &mut Vec::new(), &selection(), "sm90a").is_err());
    let mut model = fixture(512, true, false);
    assert!(apply(&mut model, &mut Vec::new(), &selection(), "gfx950").is_err());
}

#[test]
fn output_object_is_explicit_inert_and_validated_before_mutation() {
    let directory = output_dir("selection");
    let output = directory.join("model.pkt");
    let mut model = fixture(512, true, false);
    let before = model.to_blob();
    let mut sections = Vec::new();
    assert!(!apply_output_object(&mut model, &mut sections, "sm90a", &output).unwrap());
    assert_eq!(model.to_blob(), before);
    assert!(sections.is_empty());

    std::fs::write(directory.join(OBJECT_FILE), b"not a cubin").unwrap();
    assert!(apply_output_object(&mut model, &mut sections, "sm90a", &output).is_err());
    assert_eq!(model.to_blob(), before);
    assert!(sections.is_empty());

    let mut stale = OBJECT_GLOBALS;
    stale[3].1 = 32;
    std::fs::write(directory.join(OBJECT_FILE), object_image(&stale)).unwrap();
    assert!(apply_output_object(&mut model, &mut sections, "sm90a", &output).is_err());
    assert_eq!(model.to_blob(), before);
    assert!(sections.is_empty());

    std::fs::write(directory.join(OBJECT_FILE), object_image(&OBJECT_GLOBALS)).unwrap();
    assert!(apply_output_object(&mut model, &mut sections, "sm90a", &output).unwrap());
    assert_eq!(sections.len(), 1);
    let metadata = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    assert_eq!(
        metadata.objects[&PREFILL_ATTENTION_HD512_WG32].file,
        OBJECT_FILE
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn composes_with_existing_role_objects_and_preserves_their_segments() {
    let mut model = fixture(512, true, false);
    let program = &mut model.progs[0];
    let split = program
        .gq_stream
        .iter()
        .position(|entry| entry.inst != 0)
        .unwrap();
    for entry in program.stream.iter_mut().chain(&mut program.gq_stream) {
        entry.seg = u16::from(entry.inst != 0);
    }
    program.gq_seg_ofs = vec![0, split as u32, program.gq_stream.len() as u32];
    let mut sections = vec![SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: br#"{"version":1,"objects":{"1":{"abi":"fp8_gemm_tma128_v1","file":"existing.cubin"}},"programs":[{"index":0,"roles":[1,0]}]}"#.to_vec(),
    }];
    apply(&mut model, &mut sections, &selection(), "sm90a").unwrap();
    let metadata = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    let existing = &metadata.objects[&plow_asset::segment_roles::FP8_PREFILL_GEMM];
    assert_eq!(existing.abi, "fp8_gemm_tma128_v1");
    assert_eq!(existing.file, "existing.cubin");
    assert!(existing.sha256.is_none() && existing.attention.is_none());
    let record = metadata
        .programs
        .iter()
        .find(|program| program.index == 0)
        .unwrap();
    let roles_for_inst: Vec<_> = model.progs[0]
        .insts
        .iter()
        .enumerate()
        .map(|(inst, _)| {
            let segment = model.progs[0]
                .gq_stream
                .iter()
                .find(|entry| entry.inst as usize == inst)
                .unwrap()
                .seg as usize;
            record.roles[segment]
        })
        .collect();
    assert_eq!(roles_for_inst, [1, 6, 0, 0]);
}

#[test]
fn accepts_fused_output_and_rejects_unsafe_fused_contracts() {
    let mut model = fixture(512, true, true);
    let mut sections = Vec::new();
    apply(&mut model, &mut sections, &selection(), "sm90a").unwrap();
    let metadata = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    assert_eq!(metadata.programs[0].roles, [0, 6, 0]);

    let mut short = fixture(512, true, true);
    let output = short.progs[0]
        .insts
        .iter()
        .find(|op| op.op == DevOp::FlashPrefill as u16)
        .unwrap()
        .t[5] as usize;
    short.tensors[output].bytes -= 1;
    assert!(apply(&mut short, &mut Vec::new(), &selection(), "sm90a").is_err());

    let mut split = fixture(512, true, true);
    split.progs[0]
        .insts
        .iter_mut()
        .find(|op| op.op == DevOp::FlashPrefill as u16)
        .unwrap()
        .i[7] = 2;
    assert!(apply(&mut split, &mut Vec::new(), &selection(), "sm90a").is_err());
}

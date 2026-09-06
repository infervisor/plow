use super::*;
use packet::dev::{DevInst64, DevOp};
use packet::devbuild::{Builder, TensorDecl};

fn model() -> Model {
    Model {
        n_cu: 2,
        target: 0,
        tensors: vec![TensorDecl {
            name: "act".into(),
            bytes: 32,
            init: None,
        }],
        progs: vec![],
        kv_row_insts: vec![],
        prog_t: vec![],
        gen: vec![],
    }
}

fn program(n_cu: u32) -> packet::devbuild::Program {
    let mut builder = Builder::new(n_cu);
    builder.force_uniseg();
    builder.emit(DevOp::Nop, builder.all(), &[], |_| {});
    builder.finish()
}

fn inputs<'a>(
    programs: &'a [program::Program<'a>],
    objects: &'a [ObjectSection<'a>],
    variants: &'a [Variant<'a>],
) -> Spec<'a> {
    Spec {
        max_active_requests: 4,
        physical_slot_capacity: 8,
        programs: ProgramSection {
            section: "mixed_programs",
            version: 1,
            n_cu: 2,
            programs,
        },
        objects,
        variants,
    }
}

fn object(
    kind: mixed_step::PayloadKind,
    name: &'static str,
    machine: u16,
) -> ObjectSection<'static> {
    let mut bytes = plow_asset::cubin::synthetic_elf(
        "mixed_entry",
        &[(mixed_step::OBJECT_CAPABILITY, mixed_step::VERSION)],
        u32::from(machine == EM_CUDA) * 90,
    );
    bytes[0x12..0x14].copy_from_slice(&machine.to_le_bytes());
    ObjectSection {
        section: name,
        kind,
        version: mixed_step::VERSION,
        n_cu: 2,
        capability: mixed_step::OBJECT_CAPABILITY,
        capability_version: mixed_step::VERSION,
        bytes: Box::leak(bytes.into_boxed_slice()),
    }
}

#[test]
fn appends_exact_program_manifest_and_backend_objects() {
    let model = model();
    let owned = [program(2), program(2)];
    let packed: Vec<Vec<DevInst64>> = owned
        .iter()
        .map(|program| program.insts.iter().map(|inst| inst.pack()).collect())
        .collect();
    let programs: Vec<_> = owned
        .iter()
        .zip(&packed)
        .enumerate()
        .map(|(index, (program, insts))| program::Program {
            rows: [8, 16][index],
            packed_prefill_only: false,
            n_counter: program.n_counter,
            insts,
            stream: &program.stream,
            stream_ofs: &program.stream_ofs,
            stream_len: &program.stream_len,
            waits: &program.waits,
            succs: &program.succs,
            gq_stream: &program.gq_stream,
            gq_seg_ofs: &program.gq_seg_ofs,
            l2_domains: program.l2_domains,
        })
        .collect();
    let objects = [
        object(mixed_step::PayloadKind::Cubin, "mixed_sm90a", EM_CUDA),
        object(mixed_step::PayloadKind::Hsaco, "mixed_gfx", EM_AMDGPU),
    ];
    let variants = [
        Variant {
            rows: 8,
            decode_rows: 1,
            program_index: 0,
            object_indices: &[0, 1],
        },
        Variant {
            rows: 16,
            decode_rows: 4,
            program_index: 1,
            object_indices: &[0, 1],
        },
    ];
    let mut sections = Vec::new();
    append(
        &model,
        &mut sections,
        &inputs(&programs, &objects, &variants),
    )
    .unwrap();

    assert_eq!(sections.len(), 4);
    assert_eq!(sections[0].kind, SECT_PROGRAMS);
    assert_eq!(sections[1].kind, SECT_METADATA);
    let manifest: mixed_step::Manifest = serde_json::from_slice(&sections[1].data).unwrap();
    manifest.validate().unwrap();
    assert_eq!(manifest.variants[1].program.index, 1);
    assert_eq!(manifest.variants[1].rows, 16);
    assert_eq!(manifest.variants[1].decode_rows, 4);
    assert_eq!(manifest.variants[0].objects.len(), 2);
    assert_eq!(
        manifest.variants[0].program.payload.sha256,
        mixed_step::payload_sha256(&sections[0].data)
    );
    for variant in &manifest.variants {
        variant
            .bind_program(
                model.n_cu,
                model.tensors.len(),
                &mixed_step::Payload {
                    section: &sections[0].name,
                    kind: mixed_step::PayloadKind::Programs,
                    version: 1,
                    n_cu: model.n_cu,
                    bytes: &sections[0].data,
                },
            )
            .unwrap();
        variant
            .bind_cubin_with(
                model.n_cu,
                &mixed_step::Payload {
                    section: &sections[2].name,
                    kind: mixed_step::PayloadKind::Cubin,
                    version: 1,
                    n_cu: model.n_cu,
                    bytes: &sections[2].data,
                },
                |name| plow_asset::cubin::global_u32(&sections[2].data, name),
            )
            .unwrap();
        variant
            .bind_hsaco_with(
                model.n_cu,
                &mixed_step::Payload {
                    section: &sections[3].name,
                    kind: mixed_step::PayloadKind::Hsaco,
                    version: 1,
                    n_cu: model.n_cu,
                    bytes: &sections[3].data,
                },
                |name| plow_asset::cubin::global_u32(&sections[3].data, name),
            )
            .unwrap();
    }
    assert_eq!(sections[2].data, objects[0].bytes);
    assert_eq!(sections[3].data, objects[1].bytes);
}

fn one_program<'a>(
    owned: &'a packet::devbuild::Program,
    insts: &'a [DevInst64],
    rows: u32,
) -> program::Program<'a> {
    program::Program {
        rows,
        packed_prefill_only: false,
        n_counter: owned.n_counter,
        insts,
        stream: &owned.stream,
        stream_ofs: &owned.stream_ofs,
        stream_len: &owned.stream_len,
        waits: &owned.waits,
        succs: &owned.succs,
        gq_stream: &owned.gq_stream,
        gq_seg_ofs: &owned.gq_seg_ofs,
        l2_domains: owned.l2_domains,
    }
}

fn signature(sections: &[SectionData]) -> Vec<(u32, String, Vec<u8>)> {
    sections
        .iter()
        .map(|section| (section.kind, section.name.clone(), section.data.clone()))
        .collect()
}

#[test]
fn every_validation_error_is_atomic() {
    let model = model();
    let owned = program(2);
    let packed: Vec<_> = owned.insts.iter().map(|inst| inst.pack()).collect();
    let program = one_program(&owned, &packed, 8);
    let programs = [program];
    let objects = [object(
        mixed_step::PayloadKind::Cubin,
        "mixed_sm90a",
        EM_CUDA,
    )];
    let good_variant = Variant {
        rows: 8,
        decode_rows: 1,
        program_index: 0,
        object_indices: &[0],
    };
    let mut sections = vec![SectionData {
        kind: SECT_METADATA,
        name: "keep".into(),
        data: vec![7],
    }];
    let before = signature(&sections);
    let model_before = model.to_blob();

    let bad_index = [Variant {
        program_index: 1,
        ..good_variant
    }];
    assert!(append(
        &model,
        &mut sections,
        &inputs(&programs, &objects, &bad_index)
    )
    .is_err());
    assert_eq!(signature(&sections), before);

    let bad_rows = [Variant {
        rows: 16,
        ..good_variant
    }];
    assert!(append(
        &model,
        &mut sections,
        &inputs(&programs, &objects, &bad_rows)
    )
    .is_err());
    assert_eq!(signature(&sections), before);

    let mut bad_object = object(mixed_step::PayloadKind::Cubin, "bad", EM_CUDA);
    bad_object.n_cu = 3;
    assert!(append(
        &model,
        &mut sections,
        &inputs(&programs, &[bad_object], &[good_variant])
    )
    .is_err());
    assert_eq!(signature(&sections), before);
    assert_eq!(model.to_blob(), model_before);
}

#[test]
fn rejects_unsafe_programs_duplicate_sections_and_capability_bytes() {
    let model = model();
    let owned = program(2);
    let mut packed: Vec<_> = owned.insts.iter().map(|inst| inst.pack()).collect();
    packed[0].t[0] = 1;
    let unsafe_program = [one_program(&owned, &packed, 8)];
    let variant = [Variant {
        rows: 8,
        decode_rows: 1,
        program_index: 0,
        object_indices: &[],
    }];
    let mut sections = Vec::new();
    assert!(append(
        &model,
        &mut sections,
        &inputs(&unsafe_program, &[], &variant)
    )
    .is_err());
    assert!(sections.is_empty());

    let packed: Vec<_> = owned.insts.iter().map(|inst| inst.pack()).collect();
    let programs = [one_program(&owned, &packed, 8)];
    assert!(append(&model, &mut sections, &inputs(&programs, &[], &variant)).is_err());
    assert!(sections.is_empty());

    let two_cuda = [
        object(mixed_step::PayloadKind::Cubin, "cuda_a", EM_CUDA),
        object(mixed_step::PayloadKind::Cubin, "cuda_b", EM_CUDA),
    ];
    let two_cuda_variant = [Variant {
        object_indices: &[0, 1],
        ..variant[0]
    }];
    assert!(append(
        &model,
        &mut sections,
        &inputs(&programs, &two_cuda, &two_cuda_variant)
    )
    .is_err());
    assert!(sections.is_empty());

    let placed = [program::Program {
        l2_domains: 2,
        ..programs[0]
    }];
    assert!(append(&model, &mut sections, &inputs(&placed, &[], &variant)).is_err());
    assert!(sections.is_empty());

    let duplicate_objects = [
        object(mixed_step::PayloadKind::Cubin, "same", EM_CUDA),
        object(mixed_step::PayloadKind::Cubin, "same", EM_CUDA),
    ];
    assert!(append(
        &model,
        &mut sections,
        &inputs(&programs, &duplicate_objects, &variant)
    )
    .is_err());
    assert!(sections.is_empty());

    let mut bad_capability = object(mixed_step::PayloadKind::Hsaco, "bad_cap", EM_AMDGPU);
    bad_capability.capability_version = 2;
    assert!(append(
        &model,
        &mut sections,
        &inputs(&programs, &[bad_capability], &variant)
    )
    .is_err());
    assert!(sections.is_empty());

    for objects in [
        [object(
            mixed_step::PayloadKind::Cubin,
            "swapped_cuda",
            EM_AMDGPU,
        )],
        [object(
            mixed_step::PayloadKind::Hsaco,
            "swapped_amd",
            EM_CUDA,
        )],
    ] {
        assert!(append(
            &model,
            &mut sections,
            &inputs(&programs, &objects, &variant)
        )
        .is_err());
        assert!(sections.is_empty());
    }

    sections.push(SectionData {
        kind: SECT_METADATA,
        name: mixed_step::SECTION.into(),
        data: vec![],
    });
    let before = signature(&sections);
    assert!(append(&model, &mut sections, &inputs(&programs, &[], &variant)).is_err());
    assert_eq!(signature(&sections), before);
}

use super::*;
use packet::dev::DevOp;
use packet::devbuild::{Builder, Model, SECT_CUBIN, SECT_HSACO, SECT_PROGRAMS};
use plow_asset::mixed_step::{
    payload_sha256, Capability, Manifest, PayloadBinding, ProgramBinding, Variant,
    OBJECT_CAPABILITY, PROGRAM_CAPABILITY, VERSION,
};

const N_CU: u32 = 2;
const ROWS: u32 = 8;
const DECODE_ROWS: u32 = 1;

fn program_payload() -> Vec<u8> {
    program_payload_rows(&[ROWS])
}

fn program_payload_rows(rows: &[u32]) -> Vec<u8> {
    let progs = rows
        .iter()
        .map(|_| {
            let mut builder = Builder::new(N_CU);
            builder.force_uniseg();
            builder.emit(DevOp::Nop, builder.all(), &[], |_| {});
            builder.finish()
        })
        .collect();
    let model = Model {
        n_cu: N_CU,
        target: 0,
        tensors: vec![],
        progs,
        kv_row_insts: vec![],
        prog_t: rows.to_vec(),
        gen: vec![],
    };
    plow_asset::program::with_model(&model, |packet| {
        plow_asset::aux_program::encode(N_CU, 0, packet.programs).unwrap()
    })
}

#[test]
fn shared_payloads_are_bound_once_across_variants() {
    let programs = program_payload_rows(&[ROWS, 16]);
    let cubin = b"shared cubin";
    let program_binding = binding(
        "mixed_programs",
        mixed_step::PayloadKind::Programs,
        &programs,
    );
    let object_binding = binding("mixed_cuda", mixed_step::PayloadKind::Cubin, cubin);
    let metadata = serde_json::to_vec(&Manifest {
        version: VERSION,
        n_cu: N_CU,
        max_active_requests: 4,
        physical_slot_capacity: 16,
        variants: vec![
            Variant {
                rows: ROWS,
                decode_rows: 1,
                program: ProgramBinding {
                    index: 0,
                    payload: program_binding.clone(),
                },
                objects: vec![object_binding.clone()],
            },
            Variant {
                rows: 16,
                decode_rows: 2,
                program: ProgramBinding {
                    index: 1,
                    payload: program_binding,
                },
                objects: vec![object_binding],
            },
        ],
    })
    .unwrap();
    let sections = [
        section(SECT_METADATA, mixed_step::SECTION, &metadata),
        section(SECT_PROGRAMS, "mixed_programs", &programs),
        section(SECT_CUBIN, "mixed_cuda", cubin),
    ];
    let mut capability_reads = 0;

    let loaded = load(
        &sections,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        |_, _| {
            capability_reads += 1;
            Some(VERSION)
        },
    )
    .unwrap()
    .unwrap();

    assert_eq!(capability_reads, 1);
    assert_eq!(loaded.programs.len(), 1);
    assert_eq!(loaded.objects.len(), 1);
    assert_eq!(loaded.select(ROWS, 1).unwrap().program().rows, ROWS);
    assert_eq!(loaded.select(16, 2).unwrap().program().rows, 16);
}

fn binding(name: &str, kind: mixed_step::PayloadKind, bytes: &[u8]) -> PayloadBinding {
    let capability = match kind {
        mixed_step::PayloadKind::Programs => PROGRAM_CAPABILITY,
        mixed_step::PayloadKind::Cubin | mixed_step::PayloadKind::Hsaco => OBJECT_CAPABILITY,
    };
    PayloadBinding {
        section: name.into(),
        kind,
        version: VERSION,
        sha256: payload_sha256(bytes),
        capability: Capability {
            name: capability.into(),
            version: VERSION,
        },
    }
}

fn manifest(programs: &[u8], objects: Vec<PayloadBinding>) -> Vec<u8> {
    serde_json::to_vec(&Manifest {
        version: VERSION,
        n_cu: N_CU,
        max_active_requests: 4,
        physical_slot_capacity: 16,
        variants: vec![Variant {
            rows: ROWS,
            decode_rows: DECODE_ROWS,
            program: ProgramBinding {
                index: 0,
                payload: binding(
                    "mixed_programs",
                    mixed_step::PayloadKind::Programs,
                    programs,
                ),
            },
            objects,
        }],
    })
    .unwrap()
}

fn section<'a>(kind: u32, name: &'a str, bytes: &'a [u8]) -> PacketSection<'a> {
    PacketSection { kind, name, bytes }
}

#[test]
fn binds_exact_variant_and_selects_without_revalidation() {
    let programs = program_payload();
    let cubin = b"cubin with capability";
    let hsaco = b"hsaco with capability";
    let metadata = manifest(
        &programs,
        vec![
            binding("mixed_cuda", mixed_step::PayloadKind::Cubin, cubin),
            binding("mixed_amd", mixed_step::PayloadKind::Hsaco, hsaco),
        ],
    );
    let sections = [
        section(SECT_METADATA, mixed_step::SECTION, &metadata),
        section(SECT_PROGRAMS, "mixed_programs", &programs),
        section(SECT_CUBIN, "mixed_cuda", cubin),
        section(SECT_HSACO, "mixed_amd", hsaco),
    ];

    let loaded = load(
        &sections,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        |object, symbol| {
            (object.name == "mixed_cuda" && object.bytes == cubin && symbol == OBJECT_CAPABILITY)
                .then_some(VERSION)
        },
    )
    .unwrap()
    .unwrap();

    assert_eq!(loaded.n_cu(), N_CU);
    assert_eq!(loaded.max_active_requests(), 4);
    assert_eq!(loaded.physical_slot_capacity(), 16);
    assert_eq!(loaded.backend(), mixed_step::PayloadKind::Cubin);
    let selected = loaded.select(ROWS, DECODE_ROWS).unwrap();
    assert_eq!(selected.program_index(), 0);
    assert_eq!(selected.program().rows, ROWS);
    assert_eq!(selected.object().name, "mixed_cuda");
    assert!(loaded.select(ROWS, DECODE_ROWS + 1).is_none());

    let loaded = load(
        &sections,
        N_CU,
        0,
        mixed_step::PayloadKind::Hsaco,
        |object, symbol| {
            (object.name == "mixed_amd" && object.bytes == hsaco && symbol == OBJECT_CAPABILITY)
                .then_some(VERSION)
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(loaded.backend(), mixed_step::PayloadKind::Hsaco);
    assert_eq!(
        loaded.select(ROWS, DECODE_ROWS).unwrap().object().name,
        "mixed_amd"
    );
}

#[test]
fn stock_packet_without_metadata_is_ignored() {
    let sections = [section(SECT_CUBIN, "ordinary", b"ordinary object")];
    let loaded = load(
        &sections,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        |_, _| panic!("capability reader must not run"),
    )
    .unwrap();
    assert!(loaded.is_none());
}

#[test]
fn metadata_and_exact_payload_sections_are_unique_and_typed() {
    let programs = program_payload();
    let cubin = b"cubin";
    let metadata = manifest(
        &programs,
        vec![binding("mixed_cuda", mixed_step::PayloadKind::Cubin, cubin)],
    );
    let good = [
        section(SECT_METADATA, mixed_step::SECTION, &metadata),
        section(SECT_PROGRAMS, "mixed_programs", &programs),
        section(SECT_CUBIN, "mixed_cuda", cubin),
    ];
    let capability = |_: ObjectSection<'_>, _: &str| Some(VERSION);

    let duplicate_metadata = [good[0], good[0], good[1], good[2]];
    assert!(load(
        &duplicate_metadata,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        capability
    )
    .is_err());
    let wrong_metadata_kind = [section(SECT_PROGRAMS, mixed_step::SECTION, &metadata)];
    assert!(load(
        &wrong_metadata_kind,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        capability
    )
    .is_err());
    let duplicate_program = [good[0], good[1], good[1], good[2]];
    assert!(load(
        &duplicate_program,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        capability
    )
    .is_err());
    let missing_program = [good[0], good[2]];
    assert!(load(
        &missing_program,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        capability
    )
    .is_err());
    let wrong_program_kind = [
        good[0],
        section(SECT_CUBIN, "mixed_programs", &programs),
        good[2],
    ];
    assert!(load(
        &wrong_program_kind,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        capability
    )
    .is_err());
    let duplicate_object = [good[0], good[1], good[2], good[2]];
    assert!(load(
        &duplicate_object,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        capability
    )
    .is_err());
    let missing_object = [good[0], good[1]];
    assert!(load(
        &missing_object,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        capability
    )
    .is_err());
}

#[test]
fn selected_backend_requires_one_declared_object() {
    let programs = program_payload();
    let cubin_a = b"cubin a";
    let cubin_b = b"cubin b";
    let undeclared_metadata = manifest(
        &programs,
        vec![binding(
            "mixed_amd",
            mixed_step::PayloadKind::Hsaco,
            b"hsaco",
        )],
    );
    let undeclared = [
        section(SECT_METADATA, mixed_step::SECTION, &undeclared_metadata),
        section(SECT_PROGRAMS, "mixed_programs", &programs),
        section(SECT_CUBIN, "mixed_cuda", cubin_a),
    ];
    assert!(load(
        &undeclared,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        |_, _| Some(VERSION)
    )
    .is_err());

    let extra_metadata = manifest(
        &programs,
        vec![
            binding("mixed_cuda_a", mixed_step::PayloadKind::Cubin, cubin_a),
            binding("mixed_cuda_b", mixed_step::PayloadKind::Cubin, cubin_b),
        ],
    );
    let extra = [
        section(SECT_METADATA, mixed_step::SECTION, &extra_metadata),
        section(SECT_PROGRAMS, "mixed_programs", &programs),
        section(SECT_CUBIN, "mixed_cuda_a", cubin_a),
        section(SECT_CUBIN, "mixed_cuda_b", cubin_b),
    ];
    assert!(load(
        &extra,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        |_, _| Some(VERSION)
    )
    .is_err());
}

#[test]
fn digest_capability_and_grid_are_bound_to_actual_payloads() {
    let programs = program_payload();
    let cubin = b"cubin";
    let metadata = manifest(
        &programs,
        vec![binding("mixed_cuda", mixed_step::PayloadKind::Cubin, cubin)],
    );
    let wrong_cubin = b"other";
    let sections = [
        section(SECT_METADATA, mixed_step::SECTION, &metadata),
        section(SECT_PROGRAMS, "mixed_programs", &programs),
        section(SECT_CUBIN, "mixed_cuda", wrong_cubin),
    ];
    assert!(load(
        &sections,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        |_, _| Some(VERSION)
    )
    .is_err());

    let sections = [
        section(SECT_METADATA, mixed_step::SECTION, &metadata),
        section(SECT_PROGRAMS, "mixed_programs", &programs),
        section(SECT_CUBIN, "mixed_cuda", cubin),
    ];
    assert!(load(
        &sections,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        |_, _| None
    )
    .is_err());
    assert!(load(
        &sections,
        N_CU + 1,
        0,
        mixed_step::PayloadKind::Cubin,
        |_, _| Some(VERSION)
    )
    .is_err());
}

#[test]
fn auxiliary_parser_enforces_packet_tensor_count() {
    let mut programs = program_payload();
    // First instruction t0. The normal NOP payload uses TENSOR_NONE16 here;
    // changing it to tensor 0 makes a zero-tensor packet invalid.
    programs[72..74].copy_from_slice(&0u16.to_le_bytes());
    let cubin = b"cubin";
    let metadata = manifest(
        &programs,
        vec![binding("mixed_cuda", mixed_step::PayloadKind::Cubin, cubin)],
    );
    let sections = [
        section(SECT_METADATA, mixed_step::SECTION, &metadata),
        section(SECT_PROGRAMS, "mixed_programs", &programs),
        section(SECT_CUBIN, "mixed_cuda", cubin),
    ];
    assert!(load(
        &sections,
        N_CU,
        0,
        mixed_step::PayloadKind::Cubin,
        |_, _| Some(VERSION)
    )
    .is_err());
}

use super::*;
use packet::dev::DevOp;
use packet::devbuild::{Builder, Model};

fn binding(section: &str, kind: PayloadKind, bytes: &[u8], capability: &str) -> PayloadBinding {
    PayloadBinding {
        section: section.into(),
        kind,
        version: 1,
        sha256: payload_sha256(bytes),
        capability: Capability {
            name: capability.into(),
            version: 1,
        },
    }
}

fn program_payload(n_cu: u32, rows: u32, count: usize) -> Vec<u8> {
    let mut builder = Builder::new(n_cu);
    builder.force_uniseg();
    builder.emit(DevOp::Nop, builder.all(), &[], |_| {});
    let model = Model {
        n_cu,
        target: 0,
        tensors: vec![],
        progs: vec![builder.finish()],
        kv_row_insts: vec![],
        prog_t: vec![rows],
        gen: vec![],
    };
    crate::program::with_model(&model, |packet| {
        let programs = vec![packet.programs[0]; count];
        crate::aux_program::encode(n_cu, 0, &programs).unwrap()
    })
}

#[test]
fn sparse_decode_and_ragged_prefill_preserve_absolute_identity() {
    let mut frontiers = vec![0; 16];
    frontiers[0] = 1024;
    frontiers[7] = 31;
    frontiers[12] = 4096;
    frontiers[15] = 63;
    let decode = [
        DecodeRequest {
            slot: 12,
            state_slot: 2,
            token: 9,
        },
        DecodeRequest {
            slot: 0,
            state_slot: 3,
            token: 5,
        },
    ];
    let prefill = [
        PrefillRequest {
            slot: 15,
            state_slot: 4,
            start: 63,
            tokens: &[101, 102],
            prompt_len: 100,
        },
        PrefillRequest {
            slot: 7,
            state_slot: 5,
            start: 31,
            tokens: &[201, 202, 203],
            prompt_len: 100,
        },
    ];
    let plan = plan(&decode, &prefill, &frontiers, 8, 8192, 6).unwrap();

    let mut reusable = Plan::with_capacity(8, 2, 4);
    plan_into(&decode, &prefill, &frontiers, 8, 8192, 6, &mut reusable).unwrap();
    assert_eq!(reusable, plan);
    let pointers = (
        reusable.rows.as_ptr(),
        reusable.decode_slots.as_ptr(),
        reusable.prefill_spans.as_ptr(),
        reusable.parked.as_ptr(),
        reusable.mapped_ends.as_ptr(),
    );
    let capacities = (
        reusable.rows.capacity(),
        reusable.decode_slots.capacity(),
        reusable.prefill_spans.capacity(),
        reusable.parked.capacity(),
        reusable.mapped_ends.capacity(),
    );
    plan_into(&decode, &prefill, &frontiers, 8, 8192, 6, &mut reusable).unwrap();
    assert_eq!(reusable, plan);
    assert_eq!(
        pointers,
        (
            reusable.rows.as_ptr(),
            reusable.decode_slots.as_ptr(),
            reusable.prefill_spans.as_ptr(),
            reusable.parked.as_ptr(),
            reusable.mapped_ends.as_ptr(),
        )
    );
    assert_eq!(
        capacities,
        (
            reusable.rows.capacity(),
            reusable.decode_slots.capacity(),
            reusable.prefill_spans.capacity(),
            reusable.parked.capacity(),
            reusable.mapped_ends.capacity(),
        )
    );

    assert_eq!(plan.decode_rows, 2);
    assert_eq!(plan.decode_slots, [12, 0]);
    assert_eq!(plan.real_rows, 7);
    assert_eq!(
        plan.rows.iter().map(|r| r.token).collect::<Vec<_>>(),
        [9, 5, 101, 102, 201, 202, 203, 0]
    );
    assert_eq!(
        plan.rows.iter().map(|r| r.position).collect::<Vec<_>>(),
        [4096, 1024, 63, 64, 31, 32, 33, 34]
    );
    assert_eq!(
        plan.rows.iter().map(|r| r.kv_len).collect::<Vec<_>>(),
        [4097, 1025, 64, 65, 32, 33, 34, 35]
    );
    assert_eq!(
        plan.rows.iter().map(|r| r.slot).collect::<Vec<_>>(),
        [12, 0, 15, 15, 7, 7, 7, 7]
    );
    assert_eq!(
        plan.rows.iter().map(|r| r.state_slot).collect::<Vec<_>>(),
        [2, 3, 4, 4, 5, 5, 5, 5]
    );
    assert_eq!(plan.parked, [0, 0, 0, 0, 0, 0, 0, 1]);
    assert_eq!(plan.prefill_spans[0].row0, 2);
    assert_eq!(plan.prefill_spans[0].kv_row0, 63);
    assert_eq!(plan.prefill_spans[0].kv_len, 65);
    assert_eq!(plan.prefill_spans[0].program, 6);
    assert_eq!(plan.prefill_spans[1].row0, 4);
    assert_eq!(plan.mapped_ends, [(12, 4097), (0, 1025), (15, 65), (7, 35)]);
    let variant = Variant {
        rows: 8,
        decode_rows: 2,
        program: ProgramBinding {
            index: 6,
            payload: binding(
                "programs",
                PayloadKind::Programs,
                b"program",
                PROGRAM_CAPABILITY,
            ),
        },
        objects: vec![binding(
            "mixed_cuda",
            PayloadKind::Cubin,
            b"cubin",
            OBJECT_CAPABILITY,
        )],
    };
    let manifest = Manifest {
        version: VERSION,
        n_cu: 132,
        max_active_requests: 4,
        physical_slot_capacity: 16,
        variants: vec![variant],
    };
    let validated = manifest.validate().unwrap();
    validated.validate_plan(&plan).unwrap();

    let mut malformed = plan;
    malformed.rows[4].position += 1;
    assert!(validated.validate_plan(&malformed).is_err());
    malformed.rows[4].position -= 1;
    malformed.mapped_ends[3].1 -= 1;
    assert!(validated.validate_plan(&malformed).is_err());
    malformed.mapped_ends[3].1 += 1;
    malformed.rows[0].slot = 16;
    malformed.mapped_ends[0].0 = 16;
    assert!(validated.validate_plan(&malformed).is_err());
}

#[test]
fn decode_slot_map_has_exact_compact_extent_and_physical_bounds() {
    assert!(validate_decode_slots(&[15, 0], 2, 16).is_ok());
    assert!(validate_decode_slots(&[0], 2, 16).is_err());
    assert!(validate_decode_slots(&[0, -1], 2, 16).is_err());
    assert!(validate_decode_slots(&[0, 16], 2, 16).is_err());

    assert!(validate_decode_slot_binding(&[0, 1], 2, 16, None).is_ok());
    assert!(validate_decode_slot_binding(&[1, 0], 2, 16, None).is_err());
    assert!(validate_decode_slot_binding(&[1, 0], 2, 16, Some(4)).is_ok());
}

#[test]
fn flash_decode_slot_operand_is_optional_consistent_and_runtime_filled() {
    let mut decode = packet::dev::DevInst64 {
        op: DevOp::FlashDecode as u16,
        blocks: 1,
        fj: [0; 3],
        t: [packet::dev::TENSOR_NONE16; 8],
        i: [0; 8],
    };
    decode.i[0] = 2;
    let program = |insts| crate::aux_program::Program {
        rows: 8,
        n_counter: 1,
        insts,
        stream: vec![],
        stream_ofs: vec![],
        stream_len: vec![],
        waits: vec![],
        succs: vec![],
        gq_stream: vec![],
        gq_seg_ofs: vec![],
    };
    assert_eq!(
        flash_decode_slot_operand(&program(vec![decode]), 2, &[]).unwrap(),
        None
    );

    decode.t[6] = 0;
    let tensors = [TensorContract {
        name: DECODE_SLOT_TENSOR,
        bytes: 8,
        initialized: false,
    }];
    assert_eq!(
        flash_decode_slot_operand(&program(vec![decode, decode]), 2, &tensors).unwrap(),
        Some(0)
    );
    let mut absent = decode;
    absent.t[6] = packet::dev::TENSOR_NONE16;
    assert!(flash_decode_slot_operand(&program(vec![decode, absent]), 2, &tensors).is_err());
    assert!(flash_decode_slot_operand(
        &program(vec![decode]),
        2,
        &[TensorContract {
            bytes: 4,
            ..tensors[0]
        }]
    )
    .is_err());
}

#[test]
fn dense_mixed_consumers_share_decode_slots_and_canonical_prefill_spans() {
    let inst = |op| packet::dev::DevInst64 {
        op: op as u16,
        blocks: 1,
        fj: [0; 3],
        t: [packet::dev::TENSOR_NONE16; 8],
        i: [0; 8],
    };
    let mut decode = inst(DevOp::FlashDecode);
    decode.i[0] = 2;
    decode.t[6] = 0;
    let mut writer = inst(DevOp::HeadNormRope);
    writer.i[0] = 8;
    writer.fj[1] = 16;
    writer.t[6] = 0;
    let mut prefill = inst(DevOp::FlashPrefill);
    prefill.i[0] = 8;
    prefill.i[1] = 8;
    prefill.i[7] = 1;
    prefill.t[5] = 1;
    let program = |insts| crate::aux_program::Program {
        rows: 8,
        n_counter: 1,
        insts,
        stream: vec![],
        stream_ofs: vec![],
        stream_len: vec![],
        waits: vec![],
        succs: vec![],
        gq_stream: vec![],
        gq_seg_ofs: vec![],
    };
    let tensors = [
        TensorContract {
            name: DECODE_SLOT_TENSOR,
            bytes: 8,
            initialized: false,
        },
        TensorContract {
            name: "act.attn",
            bytes: 128,
            initialized: false,
        },
    ];

    assert_eq!(
        dense_consumer_contract(&program(vec![decode, writer, prefill]), 2, &tensors).unwrap(),
        0
    );
    let mut legacy_table = prefill;
    legacy_table.t[6] = 0;
    assert!(
        dense_consumer_contract(&program(vec![decode, writer, legacy_table]), 2, &tensors).is_err()
    );
    let mut wrong_writer = writer;
    wrong_writer.t[6] = packet::dev::TENSOR_NONE16;
    assert!(
        dense_consumer_contract(&program(vec![decode, wrong_writer, prefill]), 2, &tensors)
            .is_err()
    );
    let mut split_prefill = prefill;
    split_prefill.i[7] = 2;
    assert!(
        dense_consumer_contract(&program(vec![decode, writer, split_prefill]), 2, &tensors)
            .is_err()
    );
}

fn buffers(rows: usize, spans: usize, parked: usize, mapped: usize) -> Plan {
    Plan {
        decode_rows: 0,
        real_rows: 0,
        rows: Vec::with_capacity(rows),
        decode_slots: Vec::with_capacity(rows),
        prefill_spans: Vec::with_capacity(spans),
        parked: Vec::with_capacity(parked),
        mapped_ends: Vec::with_capacity(mapped),
    }
}

#[test]
fn plan_into_rejects_each_short_buffer_and_clears_partial_output() {
    let decode = [DecodeRequest {
        slot: 0,
        state_slot: 0,
        token: 7,
    }];
    let prefill = [PrefillRequest {
        slot: 1,
        state_slot: 1,
        start: 4,
        tokens: &[8, 9],
        prompt_len: 12,
    }];
    let frontiers = [0, 4];
    for mut out in [
        buffers(0, 1, 4, 2),
        buffers(4, 0, 4, 2),
        buffers(4, 1, 0, 2),
        buffers(4, 1, 4, 0),
    ] {
        let capacities = (
            out.rows.capacity(),
            out.prefill_spans.capacity(),
            out.parked.capacity(),
            out.mapped_ends.capacity(),
        );
        assert!(plan_into(&decode, &prefill, &frontiers, 4, 16, 3, &mut out).is_err());
        assert_eq!(out.decode_rows, 0);
        assert_eq!(out.real_rows, 0);
        assert!(out.rows.is_empty());
        assert!(out.prefill_spans.is_empty());
        assert!(out.parked.is_empty());
        assert!(out.mapped_ends.is_empty());
        assert_eq!(
            capacities,
            (
                out.rows.capacity(),
                out.prefill_spans.capacity(),
                out.parked.capacity(),
                out.mapped_ends.capacity(),
            )
        );
    }

    let mut out = Plan::with_capacity(4, 1, 2);
    plan_into(&decode, &prefill, &frontiers, 4, 16, 3, &mut out).unwrap();
    let duplicate = [PrefillRequest {
        slot: 0,
        state_slot: 1,
        ..prefill[0]
    }];
    assert!(plan_into(&decode, &duplicate, &frontiers, 4, 16, 3, &mut out).is_err());
    assert_eq!(out, buffers(4, 1, 4, 2));
}

#[test]
fn invalid_alias_frontier_bucket_and_padding_are_rejected() {
    let decode = DecodeRequest {
        slot: 1,
        state_slot: 1,
        token: 7,
    };
    let prefill = PrefillRequest {
        slot: 1,
        state_slot: 1,
        start: 4,
        tokens: &[8],
        prompt_len: 8,
    };
    let frontiers = [0, 4, 8, 12];
    assert!(plan(&[decode, decode], &[], &frontiers, 4, 16, 0).is_err());
    assert!(plan(&[decode], &[prefill], &frontiers, 4, 16, 0).is_err());
    assert!(plan(
        &[decode],
        &[PrefillRequest {
            slot: 2,
            state_slot: 1,
            start: 8,
            tokens: &[8],
            prompt_len: 12,
        }],
        &frontiers,
        4,
        16,
        0,
    )
    .is_err());
    assert!(plan(
        &[],
        &[PrefillRequest {
            start: 3,
            ..prefill
        }],
        &frontiers,
        4,
        16,
        0
    )
    .is_err());
    assert!(plan(&[decode], &[], &frontiers, 0, 16, 0).is_err());
    assert!(plan(
        &[DecodeRequest {
            slot: 0,
            state_slot: 0,
            token: 1
        }],
        &[],
        &[7],
        2,
        8,
        0
    )
    .is_err());
}

#[test]
fn variant_binds_exact_program_and_either_backend_object() {
    let program_bytes = program_payload(132, 128, 3);
    let cubin_bytes = b"cuda object";
    let hsaco_bytes = b"amd object";
    let program = binding(
        "mixed_programs",
        PayloadKind::Programs,
        &program_bytes,
        PROGRAM_CAPABILITY,
    );
    let cubin = binding(
        "mixed_sm90a",
        PayloadKind::Cubin,
        cubin_bytes,
        "plow_mixed_interpreter",
    );
    let hsaco = binding(
        "mixed_gfx950",
        PayloadKind::Hsaco,
        hsaco_bytes,
        "plow_mixed_interpreter",
    );
    let manifest = Manifest {
        version: VERSION,
        n_cu: 132,
        max_active_requests: 16,
        physical_slot_capacity: 16,
        variants: vec![Variant {
            rows: 128,
            decode_rows: 4,
            program: ProgramBinding {
                index: 2,
                payload: program.clone(),
            },
            objects: vec![cubin.clone(), hsaco.clone()],
        }],
    };
    manifest.validate().unwrap();
    let program_section = Payload {
        section: "mixed_programs",
        kind: PayloadKind::Programs,
        version: 1,
        n_cu: 132,
        bytes: &program_bytes,
    };
    let amd = Payload {
        section: "mixed_gfx950",
        kind: PayloadKind::Hsaco,
        version: 1,
        n_cu: 132,
        bytes: hsaco_bytes,
    };
    let variant = &manifest.variants[0];
    variant.bind_program(132, 0, &program_section).unwrap();
    variant
        .bind_hsaco_with(132, &amd, |name| (name == OBJECT_CAPABILITY).then_some(1))
        .unwrap();
    let cuda = Payload {
        section: "mixed_sm90a",
        kind: PayloadKind::Cubin,
        version: 1,
        n_cu: 132,
        bytes: cubin_bytes,
    };
    variant
        .bind_cubin_with(132, &cuda, |name| (name == OBJECT_CAPABILITY).then_some(1))
        .unwrap();

    let wrong = Payload {
        bytes: b"other",
        ..amd
    };
    assert!(variant.bind_hsaco_with(132, &wrong, |_| Some(1)).is_err());
    assert!(variant.bind_hsaco_with(132, &amd, |_| None).is_err());

    let short_bytes = program_payload(132, 128, 2);
    let short = Payload {
        bytes: &short_bytes,
        ..program_section
    };
    assert!(variant.bind_program(132, 0, &short).is_err());
}

#[test]
fn manifest_rejects_unknown_fields_and_duplicate_geometry() {
    let raw = r#"{"version":1,"n_cu":1,"max_active_requests":1,"physical_slot_capacity":1,"variants":[],"model":"x"}"#;
    assert!(serde_json::from_str::<Manifest>(raw).is_err());
    let bytes = b"program";
    let variant = Variant {
        rows: 8,
        decode_rows: 1,
        program: ProgramBinding {
            index: 0,
            payload: binding("programs", PayloadKind::Programs, bytes, PROGRAM_CAPABILITY),
        },
        objects: vec![],
    };
    let manifest = Manifest {
        version: VERSION,
        n_cu: 1,
        max_active_requests: 1,
        physical_slot_capacity: 1,
        variants: vec![variant.clone(), variant.clone()],
    };
    assert!(manifest.validate().is_err());
    let bad_capacity = Manifest {
        version: VERSION,
        n_cu: 1,
        max_active_requests: 2,
        physical_slot_capacity: 1,
        variants: vec![variant.clone()],
    };
    assert!(bad_capacity.validate().is_err());

    let mut a = variant.clone();
    let mut b = variant;
    b.rows = 16;
    b.program.payload.sha256 = "a".repeat(64);
    a.program.payload.sha256 = "b".repeat(64);
    let conflicting = Manifest {
        version: VERSION,
        n_cu: 1,
        max_active_requests: 1,
        physical_slot_capacity: 1,
        variants: vec![a, b],
    };
    assert!(conflicting.validate().is_err());
}

#[test]
fn manifest_requires_exactly_one_object_per_declared_backend() {
    let mut variant = Variant {
        rows: 8,
        decode_rows: 1,
        program: ProgramBinding {
            index: 0,
            payload: binding(
                "programs",
                PayloadKind::Programs,
                b"program",
                PROGRAM_CAPABILITY,
            ),
        },
        objects: vec![],
    };
    let manifest = |variant| Manifest {
        version: VERSION,
        n_cu: 1,
        max_active_requests: 1,
        physical_slot_capacity: 1,
        variants: vec![variant],
    };

    assert!(manifest(variant.clone()).validate().is_err());
    variant.objects.push(binding(
        "cuda_a",
        PayloadKind::Cubin,
        b"cuda a",
        OBJECT_CAPABILITY,
    ));
    manifest(variant.clone()).validate().unwrap();

    let mut reserved = variant.clone();
    reserved.objects[0].section = SECTION.into();
    assert!(manifest(reserved).validate().is_err());

    variant.objects.push(binding(
        "cuda_b",
        PayloadKind::Cubin,
        b"cuda b",
        OBJECT_CAPABILITY,
    ));
    assert!(manifest(variant.clone()).validate().is_err());

    variant.objects.pop();
    variant.objects.push(binding(
        "amd",
        PayloadKind::Hsaco,
        b"amd",
        OBJECT_CAPABILITY,
    ));
    manifest(variant).validate().unwrap();
}

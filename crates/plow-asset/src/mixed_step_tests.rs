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

    split_prefill.i[2] = 2;
    split_prefill.i[6] = 8;
    split_prefill.t[0] = 2;
    split_prefill.t[1] = 3;
    split_prefill.t[5] = packet::dev::TENSOR_NONE16;
    let mut merge = inst(DevOp::FlashMerge);
    merge.t[0] = 1;
    merge.t[1] = 2;
    merge.t[2] = 3;
    merge.i[..5].copy_from_slice(&[8, 2, 2, 8, 2]);
    let mut split_tensors = tensors.to_vec();
    split_tensors[1].bytes = 256;
    split_tensors.extend([
        TensorContract {
            name: "act.prefill_opart",
            bytes: 1024,
            initialized: false,
        },
        TensorContract {
            name: "act.prefill_mlpart",
            bytes: 256,
            initialized: false,
        },
    ]);
    let valid = vec![decode, writer, split_prefill, merge];
    assert_eq!(
        dense_amd_consumer_contract(&program(valid.clone()), 2, &split_tensors).unwrap(),
        0
    );
    assert!(dense_consumer_contract(&program(valid.clone()), 2, &split_tensors).is_err());
    assert!(dense_amd_consumer_contract(&program(valid[..3].to_vec()), 2, &split_tensors).is_err());
    for (field, value) in [(0, 7), (1, 3), (2, 3), (3, 16), (4, 1)] {
        let mut bad = valid.clone();
        bad[3].i[field] = value;
        assert!(dense_amd_consumer_contract(&program(bad), 2, &split_tensors).is_err());
    }
    for tensor in [1, 2, 3] {
        let mut bad = split_tensors.clone();
        bad[tensor].bytes -= 1;
        assert!(dense_amd_consumer_contract(&program(valid.clone()), 2, &bad).is_err());
    }
    let mut alias = valid.clone();
    alias[0].t[0] = split_prefill.t[0];
    assert!(dense_amd_consumer_contract(&program(alias), 2, &split_tensors).is_err());
    let mut softcap = inst(DevOp::SoftCap);
    softcap.t[0] = 1;
    softcap.i[0] = 16;
    softcap.i[1] = 2;
    let mut capacity = valid.clone();
    capacity.push(softcap);
    assert!(
        dense_amd_capacity_consumer_contract(&program(capacity.clone()), 2, &split_tensors).is_ok()
    );
    capacity.last_mut().unwrap().i[1] = 0;
    assert!(dense_amd_capacity_consumer_contract(&program(capacity), 2, &split_tensors).is_err());

    for op in [
        DevOp::Embed,
        DevOp::RmsNorm,
        DevOp::HeadNormRope,
        DevOp::NormResidual,
        DevOp::GemmGlu,
    ] {
        let mut body = inst(op);
        body.i[0] = 8;
        let mut candidate = valid.clone();
        candidate.push(body);
        assert!(dense_amd_capacity_consumer_contract(
            &program(candidate.clone()),
            2,
            &split_tensors
        )
        .is_ok());
        for rows in [0, 1, 2, 7, 9] {
            candidate.last_mut().unwrap().i[0] = rows;
            assert!(
                dense_amd_capacity_consumer_contract(
                    &program(candidate.clone()),
                    2,
                    &split_tensors
                )
                .is_err(),
                "{op:?} rows={rows}"
            );
            assert!(
                dense_amd_consumer_contract(&program(candidate.clone()), 2, &split_tensors).is_ok(),
                "legacy {op:?} validation unchanged"
            );
        }
    }
    for rows in [2, 8] {
        let mut gemm = inst(DevOp::Gemm);
        gemm.i[0] = rows;
        let mut candidate = valid.clone();
        candidate.push(gemm);
        assert!(dense_amd_capacity_consumer_contract(
            &program(candidate.clone()),
            2,
            &split_tensors
        )
        .is_ok());
        for field in [4, 5] {
            candidate.last_mut().unwrap().i[field] = 1;
            assert!(dense_amd_capacity_consumer_contract(
                &program(candidate.clone()),
                2,
                &split_tensors
            )
            .is_err());
            candidate.last_mut().unwrap().i[field] = 0;
        }
    }
    for rows in [0, 1, 3, 7, 9] {
        let mut gemm = inst(DevOp::Gemm);
        gemm.i[0] = rows;
        let mut candidate = valid.clone();
        candidate.push(gemm);
        assert!(
            dense_amd_capacity_consumer_contract(&program(candidate), 2, &split_tensors).is_err()
        );
    }
    let mut decode_merge = inst(DevOp::FlashMerge);
    decode_merge.i[0] = 2;
    let mut candidate = valid;
    candidate.push(decode_merge);
    assert!(
        dense_amd_capacity_consumer_contract(&program(candidate.clone()), 2, &split_tensors)
            .is_ok()
    );
    candidate.last_mut().unwrap().i[0] = 8;
    assert!(dense_amd_capacity_consumer_contract(&program(candidate), 2, &split_tensors).is_err());
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
        cover: SpanCover::DecodeBand,
        commits: Vec::with_capacity(mapped),
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

// ================================================================================================
// Unified token batch: SpanCover::PrefixFree
// ================================================================================================

fn tb_plan(
    decode: &[DecodeRequest],
    prefill: &[PrefillRequest<'_>],
    frontiers: &[u32],
    rows: u32,
    max_ctx: u32,
) -> Result<Plan> {
    let active = decode.len() + prefill.len();
    let mut out = Plan::with_capacity(rows as usize, decode.len() + prefill.len() * 2, active);
    // `with_capacity` sizes the span vector from its second argument, and PrefixFree needs the
    // decode-slot vector to hold every LEADING row, not just the decode ones.
    out.decode_slots = Vec::with_capacity(active);
    out.commits = Vec::with_capacity(active);
    plan_into_cover(
        decode,
        prefill,
        frontiers,
        rows,
        max_ctx,
        7,
        SpanCover::PrefixFree,
        &mut out,
    )?;
    Ok(out)
}

/// §4.4: "Spans cover exactly `[0, M)` with no overlaps, gaps or zero-length entries."
/// `runtime/amd/token_batch.h`'s `plow_tb_view` traps on any violation, so the planner is the
/// only place this can be got right.
#[test]
fn prefix_free_spans_tile_the_batch_with_no_decode_prefix() {
    // §6.1's worked example: two ongoing decodes, a prompt that finishes, an intermediate chunk.
    let decode = [
        DecodeRequest { slot: 0, state_slot: 0, token: 11 },
        DecodeRequest { slot: 1, state_slot: 1, token: 22 },
    ];
    let c: Vec<u32> = (0..50).collect();
    let d: Vec<u32> = (0..80).collect();
    let prefill = [
        PrefillRequest { slot: 2, state_slot: 2, start: 70, tokens: &c, prompt_len: 120 },
        PrefillRequest { slot: 3, state_slot: 3, start: 0, tokens: &d, prompt_len: 400 },
    ];
    let frontiers = [100, 900, 70, 0];
    let plan = tb_plan(&decode, &prefill, &frontiers, 256, 4096).unwrap();

    assert_eq!(plan.cover, SpanCover::PrefixFree);
    assert_eq!(plan.real_rows, 132, "M = 2 decodes + 50 + 80");
    // Three sampled rows: the two decodes and the prompt that completes. The intermediate
    // chunk contributes tokens to M and ZERO to S.
    assert_eq!(plan.decode_rows, 3);

    // Dense cover of [0, M) from zero — the property the decode band cannot express.
    let mut expect = 0;
    for span in &plan.prefill_spans {
        assert_eq!(span.row0, expect, "spans must tile from 0");
        assert!(span.n_rows > 0);
        assert_eq!(span.kv_row0 + span.n_rows, span.kv_len);
        expect += span.n_rows;
    }
    assert_eq!(expect, plan.real_rows);
    assert_eq!(plan.prefill_spans[0].row0, 0, "there is no decode prefix");

    // The leading run of length-one spans is exactly S. `plow_tb_decode_spans` reads that run
    // as the attention partition AND, through PLOW_SAMPLE_ROWS, as the selection row count.
    let leading = plan.prefill_spans.iter().take_while(|s| s.n_rows == 1).count();
    assert_eq!(leading as u32, plan.decode_rows);

    // The completing prompt's terminal token leads the batch at its own absolute position, and
    // its body follows: two spans, one slot, contiguous in KV.
    let terminal = plan.prefill_spans[2];
    assert_eq!((terminal.slot, terminal.kv_row0, terminal.kv_len), (2, 119, 120));
    let body = plan
        .prefill_spans
        .iter()
        .find(|s| s.slot == 2 && s.n_rows > 1)
        .expect("body span");
    assert_eq!((body.kv_row0, body.kv_len), (70, 119));
    assert_eq!(plan.rows[terminal.row0 as usize].token, c[49]);
    assert_eq!(plan.rows[terminal.row0 as usize].position, 119);

    // The intermediate chunk samples nothing and its span is not in the leading run.
    let intermediate = plan.prefill_spans.last().unwrap();
    assert_eq!((intermediate.slot, intermediate.n_rows, intermediate.kv_len), (3, 80, 80));

    // Commit is per REQUEST: the completing prompt advances once, to prompt_len.
    assert_eq!(plan.commits.len(), 4);
    assert!(plan.commits.contains(&Commit { slot: 2, expect: 70, after: 120 }));
    assert!(plan.commits.contains(&Commit { slot: 0, expect: 100, after: 101 }));
}

/// The shared request contract plans EXACTLY the plan the AMD request pair does — same rows,
/// same spans, same commits — and names the owner of every leading row. This is the property
/// that lets one mux arm feed both backends' token-batch routes.
#[test]
fn the_request_contract_plans_the_same_leading_band_layout() {
    use crate::token_batch::{Phase, Request, Selection};
    let c: Vec<u32> = (0..50).collect();
    let d: Vec<u32> = (0..80).collect();
    let frontiers = [100, 900, 70, 0];
    let generations = [3, 1, 2, 9];
    let req = |id: u32, slot: u32, phase: Phase, tokens: &'static [u32], prompt_len: u32| Request {
        id,
        slot,
        state_slot: slot,
        generation: generations[slot as usize],
        phase,
        tokens,
        prompt_len,
        selection: Selection::default(),
    };
    let eleven: &'static [u32] = &[11];
    let twenty_two: &'static [u32] = &[22];
    let c_static: &'static [u32] = Box::leak(c.clone().into_boxed_slice());
    let d_static: &'static [u32] = Box::leak(d.clone().into_boxed_slice());
    // Ids are deliberately not slot numbers, and the completing prompt is listed BEFORE a
    // decode: the leading order is decode-then-completing regardless of request order.
    let requests = [
        req(40, 2, Phase::Prefill, c_static, 120),
        req(10, 0, Phase::Decode, eleven, 90),
        req(30, 3, Phase::Prefill, d_static, 400),
        req(20, 1, Phase::Decode, twenty_two, 800),
    ];
    let mut plan = Plan::with_capacity(256, 6, 4);
    plan.decode_slots = Vec::with_capacity(4);
    plan.commits = Vec::with_capacity(4);
    let mut owners = Vec::new();
    plan_requests_into(&requests, &frontiers, &generations, 256, 4096, 7, &mut plan, &mut owners)
        .unwrap();

    let decode = [
        DecodeRequest { slot: 0, state_slot: 0, token: 11 },
        DecodeRequest { slot: 1, state_slot: 1, token: 22 },
    ];
    let prefill = [
        PrefillRequest { slot: 2, state_slot: 2, start: 70, tokens: &c, prompt_len: 120 },
        PrefillRequest { slot: 3, state_slot: 3, start: 0, tokens: &d, prompt_len: 400 },
    ];
    let reference = tb_plan(&decode, &prefill, &frontiers, 256, 4096).unwrap();
    assert_eq!(plan, reference);
    assert_eq!(owners, [10, 20, 40], "decodes in order, then the completing prompt");
    assert_eq!(plan.decode_slots, [0, 1, 2]);

    // The shared checks the AMD pair never had: slot generation, request identity, decode shape.
    let mut stale = generations;
    stale[0] += 1;
    let err = plan_requests_into(&requests, &frontiers, &stale, 256, 4096, 7, &mut plan, &mut owners)
        .unwrap_err();
    assert!(err.contains("generation"), "{err}");
    assert_eq!((plan.real_rows, owners.len()), (0, 0), "a refusal leaves no residue");

    let mut dup = requests;
    dup[3].id = 10;
    let err = plan_requests_into(&dup, &frontiers, &generations, 256, 4096, 7, &mut plan, &mut owners)
        .unwrap_err();
    assert!(err.contains("duplicate request id"), "{err}");

    let mut wide = requests;
    wide[1].tokens = &[1, 2];
    let err = plan_requests_into(&wide, &frontiers, &generations, 256, 4096, 7, &mut plan, &mut owners)
        .unwrap_err();
    assert!(err.contains("decode span"), "{err}");
}

/// A one-token prompt is one span with one selected row, and no empty body span — a zero-length
/// entry would trap in `plow_tb_view`.
#[test]
fn a_one_token_prompt_is_one_length_one_span() {
    let tokens = [5u32];
    let prefill = [PrefillRequest { slot: 0, state_slot: 0, start: 0, tokens: &tokens, prompt_len: 1 }];
    let plan = tb_plan(&[], &prefill, &[0], 8, 64).unwrap();
    assert_eq!(plan.prefill_spans.len(), 1);
    assert_eq!(plan.prefill_spans[0].n_rows, 1);
    assert_eq!(plan.prefill_spans[0].kv_row0, 0);
    assert_ne!(plan.prefill_spans[0].flags & PREFILL_SPAN_RESET_STATE, 0);
    assert_eq!(plan.decode_rows, 1);
    assert_eq!(plan.real_rows, 1);
}

/// An intermediate chunk contributes tokens to M and nothing to S, and the plan is legal with
/// no sampled row at all — `S = 0` means no output segment, and it is the adapter's job to
/// refuse the step rather than let argmax read zero as one.
#[test]
fn an_intermediate_chunk_alone_samples_nothing() {
    let tokens: Vec<u32> = (0..64).collect();
    let prefill = [PrefillRequest { slot: 0, state_slot: 0, start: 0, tokens: &tokens, prompt_len: 512 }];
    let plan = tb_plan(&[], &prefill, &[0], 128, 4096).unwrap();
    assert_eq!(plan.decode_rows, 0);
    assert_eq!(plan.real_rows, 64);
    assert_eq!(plan.prefill_spans.len(), 1);
    assert_eq!(plan.prefill_spans[0].n_rows, 64);
}

/// Mixed step v1's shape is untouched: the same requests under `DecodeBand` still put the
/// decode rows in a band ahead of the spans and hold the prompt's last token back.
#[test]
fn the_decode_band_contract_is_unchanged_by_the_new_mode() {
    let decode = [DecodeRequest { slot: 0, state_slot: 0, token: 11 }];
    let tokens: Vec<u32> = (0..50).collect();
    let prefill = [PrefillRequest { slot: 1, state_slot: 1, start: 70, tokens: &tokens, prompt_len: 120 }];
    let frontiers = [100, 70];
    let band = plan(&decode, &prefill, &frontiers, 128, 4096, 7).unwrap();
    assert_eq!(band.cover, SpanCover::DecodeBand);
    assert_eq!(band.decode_rows, 1);
    assert_eq!(band.prefill_spans.len(), 1);
    assert_eq!(band.prefill_spans[0].row0, 1, "spans start AFTER the decode band");
    assert_eq!(band.prefill_spans[0].n_rows, 50, "no terminal split");
    // The commit list reproduces exactly what the per-span commit used to do.
    assert_eq!(band.commits.len(), 2);
    assert!(band.commits.contains(&Commit { slot: 1, expect: 70, after: 120 }));
}

/// Padding belongs to nobody: it is parked, and a live row is never invented for it.
#[test]
fn prefix_free_padding_is_parked_and_outside_every_span() {
    let decode = [DecodeRequest { slot: 0, state_slot: 0, token: 11 }];
    let frontiers = [10, 0];
    let tokens: Vec<u32> = (0..4).collect();
    let prefill = [PrefillRequest { slot: 1, state_slot: 1, start: 0, tokens: &tokens, prompt_len: 4 }];
    let plan = tb_plan(&decode, &prefill, &frontiers, 32, 4096).unwrap();
    assert_eq!(plan.real_rows, 5);
    assert!(plan.parked[..5].iter().all(|&v| v == 0));
    assert!(plan.parked[5..].iter().all(|&v| v == 1));
    assert_eq!(plan.parked.len(), 32);
    let covered: u32 = plan.prefill_spans.iter().map(|s| s.n_rows).sum();
    assert_eq!(covered, plan.real_rows);
}

use super::*;

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

#[test]
fn sparse_decode_and_ragged_prefill_preserve_absolute_identity() {
    let mut frontiers = vec![0; 16];
    frontiers[0] = 1024;
    frontiers[7] = 31;
    frontiers[12] = 4096;
    frontiers[15] = 63;
    let plan = plan(
        &[
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
        ],
        &[
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
        ],
        &frontiers,
        8,
        8192,
        6,
    )
    .unwrap();

    assert_eq!(plan.decode_rows, 2);
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
        objects: vec![],
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
    let program_bytes = b"shared dev program";
    let cubin_bytes = b"cuda object";
    let hsaco_bytes = b"amd object";
    let program = binding(
        "mixed_programs",
        PayloadKind::Programs,
        program_bytes,
        PROGRAM_CAPABILITY,
    );
    let cubin = binding(
        "mixed_sm90a",
        PayloadKind::Cubin,
        cubin_bytes,
        "plow.mixed.interpreter",
    );
    let hsaco = binding(
        "mixed_gfx950",
        PayloadKind::Hsaco,
        hsaco_bytes,
        "plow.mixed.interpreter",
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
    let caps = [Capability {
        name: PROGRAM_CAPABILITY.into(),
        version: 1,
    }];
    let program_payload = Payload {
        section: "mixed_programs",
        kind: PayloadKind::Programs,
        version: 1,
        n_cu: 132,
        bytes: program_bytes,
        capabilities: &caps,
        program_count: Some(3),
    };
    let object_caps = [Capability {
        name: "plow.mixed.interpreter".into(),
        version: 1,
    }];
    let amd = Payload {
        section: "mixed_gfx950",
        kind: PayloadKind::Hsaco,
        version: 1,
        n_cu: 132,
        bytes: hsaco_bytes,
        capabilities: &object_caps,
        program_count: None,
    };
    manifest.variants[0]
        .bind(132, &program_payload, Some(&amd))
        .unwrap();

    let wrong = Payload {
        bytes: b"other",
        ..amd
    };
    assert!(manifest.variants[0]
        .bind(132, &program_payload, Some(&wrong))
        .is_err());
    let short = Payload {
        program_count: Some(2),
        ..program_payload
    };
    assert!(manifest.variants[0].bind(132, &short, None).is_err());
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

use super::*;
fn model() -> Model {
    let mut programs = Vec::new();
    let mut tensors = Vec::new();
    for rows in [1, 2, 4, 8, 16] {
        let mut b = Builder::new(2);
        b.force_uniseg();
        let a = b.tensor("act.a", 16 * 64 * 2);
        let w = b.tensor("model.layers.0.mlp.down_proj.weight", 128 * 64 * 2);
        let c = b.tensor("act.c", 16 * 128 * 2);
        b.emit(DevOp::Gemv, vec![0, 1], &[], |d| {
            d.t[..3].copy_from_slice(&[c, a, w]);
            d.i[..3].copy_from_slice(&[rows, 128, 64]);
            d.f[0] = 1e-6;
        });
        tensors = b.tensors();
        programs.push(b.finish());
    }
    Model {
        n_cu: 2,
        target: 0,
        tensors,
        progs: programs,
        prog_t: vec![1, 2, 4, 8, 16],
        gen: vec![],
        kv_row_insts: vec![],
    }
}
fn gemma31_model() -> Model {
    let mut programs = Vec::new();
    let mut tensors = Vec::new();
    for rows in [1, 2, 4, 8, 16] {
        let mut b = Builder::new(2);
        b.force_uniseg();
        let hn = b.tensor("act.hn", 16 * 5376 * 2);
        let qw = b.tensor("model.layers.0.self_attn.q_proj.weight", 16384 * 5376 * 2);
        let qg = b.tensor("act.qg", 16 * 16384 * 2);
        let kw = b.tensor("model.layers.0.self_attn.k_proj.weight", 2048 * 5376 * 2);
        let kg = b.tensor("act.kg", 16 * 2048 * 2);
        let at = b.tensor("act.at", 16 * 16384 * 2);
        let ow = b.tensor("model.layers.0.self_attn.o_proj.weight", 5376 * 16384 * 2);
        let og = b.tensor("act.og", 16 * 5376 * 2);
        let fu = b.tensor("act.fu", 16 * 21504 * 2);
        let dw = b.tensor("model.layers.0.mlp.down_proj.weight", 5376 * 21504 * 2);
        let dg = b.tensor("act.dg", 16 * 5376 * 2);
        let q = b.emit(DevOp::Gemv, vec![0, 1], &[], |d| {
            d.t[..3].copy_from_slice(&[qg, hn, qw]);
            d.i[..3].copy_from_slice(&[rows, 16384, 5376]);
        });
        let k = b.emit(DevOp::Gemv, vec![0, 1], &[], |d| {
            d.t[..3].copy_from_slice(&[kg, hn, kw]);
            d.i[..3].copy_from_slice(&[rows, 2048, 5376]);
        });
        let o = b.emit(DevOp::Gemv, vec![0, 1], &[q, k], |d| {
            d.t[..3].copy_from_slice(&[og, at, ow]);
            d.i[..3].copy_from_slice(&[rows, 5376, 16384]);
        });
        b.emit(DevOp::Gemv, vec![0, 1], &[o], |d| {
            d.t[..3].copy_from_slice(&[dg, fu, dw]);
            d.i[..3].copy_from_slice(&[rows, 5376, 21504]);
        });
        tensors = b.tensors();
        programs.push(b.finish());
    }
    Model {
        n_cu: 2,
        target: 0,
        tensors,
        progs: programs,
        prog_t: vec![1, 2, 4, 8, 16],
        gen: vec![],
        kv_row_insts: vec![],
    }
}
fn fixture() -> (DecodeObjects, ProjectionMeasurement) {
    let baseline = DecodeObject {
        file: "old.cubin".into(),
        sha256: "a".repeat(64),
        profile: "sm90a".into(),
        entry: "_Z12interp_sm90a11PlowProgram".into(),
        threads: 256,
        arena_bytes: 16448,
        grid: 2,
    };
    let mut candidate = baseline.clone();
    candidate.file = "s8.cubin".into();
    candidate.sha256 = "b".repeat(64);
    candidate.arena_bytes = 82944;
    let metadata = DecodeObjects {
        version: 1,
        kernarg_bytes: std::mem::size_of::<packet::dev::DevProgram>(),
        objects: BTreeMap::from([(0, baseline.clone())]),
        programs: [1, 2, 4, 8, 16]
            .into_iter()
            .enumerate()
            .map(
                |(index, rows)| plow_asset::decode_objects::DecodeProgramObject {
                    index,
                    rows,
                    object: 0,
                },
            )
            .collect(),
    };
    let stats = tunedb::projection::ProjectionTiming {
        median_ns: 68.,
        p10_ns: 67.,
        p90_ns: 69.,
        samples: 40,
    };
    let mut scalar = stats.clone();
    scalar.median_ns = 392.;
    scalar.p10_ns = 391.;
    scalar.p90_ns = 393.;
    let record = ProjectionMeasurement {
        cell: ProjectionCell {
            hardware: "test".into(),
            n_cu: 2,
            threads: 256,
            rows: 4,
            n: 128,
            k: 64,
        },
        split: 8,
        baseline_object: baseline,
        candidate_object: candidate,
        baseline_registers: 200,
        candidate_registers: 216,
        native_blocks: vec![tunedb::projection::NativeBlockGuard {
            context_tokens: 1024,
            packet_sha256: "c".repeat(64),
            stats: tunedb::projection::NativeBlockTiming {
                median_ns: 800.,
                p95_ns: 810.,
                samples: 40,
            },
            baseline: tunedb::projection::NativeBlockTiming {
                median_ns: 1100.,
                p95_ns: 1110.,
                samples: 40,
            },
        }],
        digests: tunedb::Digests {
            implementation: "body".into(),
            interpreter: "b".repeat(64),
            toolchain: "cuda".into(),
            oracle: PROJECTION_ORACLE.into(),
        },
        stats,
        baseline: scalar,
        correctness: tunedb::Correctness::Pass,
        state: tunedb::RecordState::Qualified,
        campaign: "block qualification".into(),
    };
    (metadata, record)
}
#[test]
fn missing_stale_failed_records_preserve_binding_resources_and_packet_bytes() {
    let (metadata, r) = fixture();
    for case in 0..6 {
        let mut m = model();
        let before = m.to_blob();
        let mut records = vec![r.clone()];
        match case {
            0 => records.clear(),
            1 => records[0].digests.implementation = "stale".into(),
            2 => records[0].correctness = tunedb::Correctness::Unchecked,
            3 => records[0].state = tunedb::RecordState::Provisional,
            4 => records[0].native_blocks.clear(),
            5 => {
                records[0].correctness = tunedb::Correctness::Fail {
                    detail: "numerical mismatch".into(),
                }
            }
            _ => unreachable!(),
        }
        let (selected, bindings) = plan(&m, &metadata, &records, "test", &r.digests, |_| {
            panic!("fallback must not load candidate")
        })
        .unwrap();
        assert_eq!(rewrite(&mut m, &selected).unwrap(), 0);
        assert_eq!(before, m.to_blob());
        assert_eq!(bindings, metadata);
        assert_eq!(
            serde_json::to_vec(&bindings).unwrap(),
            serde_json::to_vec(&metadata).unwrap()
        );
    }
}
#[test]
fn selected_rung_rebinds_only_after_qualification_and_conflicts_reject() {
    let (metadata, r) = fixture();
    let mut m = model();
    let (selected, bindings) = plan(
        &m,
        &metadata,
        std::slice::from_ref(&r),
        "test",
        &r.digests,
        |_| Ok((Some(1), 82944)),
    )
    .unwrap();
    assert_eq!(selected, BTreeMap::from([((2, 0), 8)]));
    assert_eq!(
        bindings.objects[&bindings.programs[0].object],
        r.baseline_object
    );
    assert_eq!(
        bindings.objects[&bindings.programs[2].object],
        r.candidate_object
    );
    assert!(plan(
        &m,
        &metadata,
        std::slice::from_ref(&r),
        "test",
        &r.digests,
        |_| Ok((None, 82944))
    )
    .is_err());
    let mut second = r.clone();
    second.cell.n = 64;
    second.candidate_object.file = "other.cubin".into();
    second.candidate_object.sha256 = "d".repeat(64);
    second.digests.interpreter = "d".repeat(64);
    let mut inst = m.progs[2].insts[0].clone();
    inst.i[1] = 64;
    m.progs[2].insts.push(inst);
    assert!(plan(
        &m,
        &metadata,
        &[r.clone(), second],
        "test",
        &r.digests,
        |_| Ok((Some(1), 82944))
    )
    .unwrap_err()
    .contains("conflicting objects"));
}
#[test]
fn absent_selection_preserves_entire_blob() {
    let mut m = model();
    let before = m.to_blob();
    let config = crate::emit_config::EmitConfig::from_env();
    assert!(!config.decode_projection_tuning);
    assert_eq!(
        apply(
            &mut m,
            &config,
            "unknown",
            "unknown",
            Path::new("/missing/no-output")
        )
        .unwrap(),
        None
    );
    assert_eq!(rewrite(&mut m, &BTreeMap::new()).unwrap(), 0);
    assert_eq!(before, m.to_blob());
}
#[test]
fn measured_rewrite_preserves_b1_b2_and_canonical_graph() {
    let mut m = model();
    let old: Vec<_> = m.progs.iter().map(|p| p.insts[0].pack()).collect();
    assert_eq!(
        rewrite(
            &mut m,
            &BTreeMap::from([((2, 0), 8), ((3, 0), 8), ((4, 0), 8)])
        )
        .unwrap(),
        3
    );
    assert_eq!(m.progs[0].insts[0].pack(), old[0]);
    assert_eq!(m.progs[1].insts[0].pack(), old[1]);
    assert_eq!(m.tensors.len(), 4);
    assert_eq!(m.tensors[3].bytes, 16 * 128 * 4 + 256);
    let proof = plow_asset::program::with_model(&m, plow_asset::splitk::validate)
        .unwrap()
        .unwrap();
    for (canonical, mut expected) in proof.canonical.iter().zip(old) {
        expected.fj[0] = 0;
        assert_eq!(canonical.instructions, [expected]);
    }
}
#[test]
fn gemma31_routes_all_ordinary_projections_at_measured_rungs() {
    let (metadata, record) = fixture();
    let mut records = Vec::new();
    for rows in [4, 8, 16] {
        for (n, k, split) in [
            (16384, 5376, 2),
            (2048, 5376, 8),
            (5376, 16384, 8),
            (5376, 21504, if rows == 4 { 16 } else { 8 }),
        ] {
            let mut measured = record.clone();
            measured.cell.rows = rows;
            measured.cell.n = n;
            measured.cell.k = k;
            measured.split = split;
            records.push(measured);
        }
    }
    let mut m = gemma31_model();
    let untouched: Vec<_> = m.progs[..2].iter().map(|p| p.to_blob()).collect();
    let (selected, bindings) = plan(&m, &metadata, &records, "test", &record.digests, |_| {
        Ok((Some(1), 82944))
    })
    .unwrap();
    assert_eq!(selected.len(), 12);
    assert_eq!(
        selected.values().copied().collect::<Vec<_>>(),
        [2, 8, 8, 16, 2, 8, 8, 8, 2, 8, 8, 8]
    );
    assert_eq!(bindings.programs[0].object, 0);
    assert_eq!(bindings.programs[1].object, 0);
    assert_ne!(bindings.programs[2].object, 0);
    assert_eq!(bindings.programs[2].object, bindings.programs[3].object);
    assert_eq!(bindings.programs[3].object, bindings.programs[4].object);

    let tensor_count = m.tensors.len();
    assert_eq!(rewrite(&mut m, &selected).unwrap(), 12);
    assert_eq!(
        m.progs[..2].iter().map(|p| p.to_blob()).collect::<Vec<_>>(),
        untouched
    );
    assert_eq!(m.tensors.len(), tensor_count + 4);
    assert_eq!(
        m.tensors[tensor_count..]
            .iter()
            .map(|t| t.bytes)
            .collect::<Vec<_>>(),
        [
            16 * 16384 * 4 + 256,
            16 * 2048 * 4 + 256,
            16 * 5376 * 4 + 256,
            16 * 5376 * 4 + 256
        ]
    );
    for program in &m.progs[2..] {
        for op in [DevOp::ZeroF32, DevOp::GemmSplitK, DevOp::CastF32Bf16] {
            assert_eq!(
                program.insts.iter().filter(|d| d.op == op as u16).count(),
                4
            );
        }
    }
}
#[test]
fn unsupported_selection_rejected() {
    for key in [(0, 0), (4, 1), (5, 0)] {
        assert!(rewrite(&mut model(), &BTreeMap::from([(key, 8)])).is_err());
    }
    assert!(rewrite(&mut model(), &BTreeMap::from([((4, 0), 3)])).is_err());
}

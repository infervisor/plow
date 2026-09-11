use super::*;

#[test]
fn sandwich_norm_split_preserves_operands_and_rounding_boundary() {
    for gamma_b in [TENSOR_NONE16, 4] {
        for gamma_n in [TENSOR_NONE16, 5] {
            for residual in [1, 2, 3] {
                let mut inst = DevInst64 {
                    op: DevOp::NormResidualNorm as u16,
                    t: [
                        0,
                        residual,
                        2,
                        3,
                        gamma_b,
                        gamma_n,
                        TENSOR_NONE16,
                        TENSOR_NONE16,
                    ],
                    i: [128, 5376, 0, 0, 0, 0, 0, 0],
                    fj: [1e-6f32.to_bits(), 0.625f32.to_bits(), 0],
                    ..Default::default()
                };
                let [r, n] = split_norm_residual_norm(&inst, 128).unwrap();
                assert_eq!(r.op, DevOp::NormResidual as u16);
                assert_eq!(
                    r.t,
                    [
                        residual,
                        2,
                        3,
                        gamma_b,
                        TENSOR_NONE16,
                        TENSOR_NONE16,
                        TENSOR_NONE16,
                        TENSOR_NONE16
                    ]
                );
                assert_eq!(r.fj, inst.fj);
                assert_eq!(n.op, DevOp::RmsNorm as u16);
                assert_eq!(
                    n.t,
                    [
                        0,
                        residual,
                        gamma_n,
                        TENSOR_NONE16,
                        TENSOR_NONE16,
                        TENSOR_NONE16,
                        TENSOR_NONE16,
                        TENSOR_NONE16
                    ]
                );
                assert_eq!(n.fj, [inst.fj[0], 0, 0]);
                assert_eq!(r.i, inst.i);
                assert_eq!(n.i, inst.i);
                assert!(split_norm_residual_norm(&inst, 512).is_err());
                inst.t[0] = residual;
                assert!(split_norm_residual_norm(&inst, 128).is_err());
            }
        }
    }
    let inst = DevInst64 {
        op: DevOp::NormResidualNorm as u16,
        t: [0, 1, 1, 2, 3, 4, TENSOR_NONE16, TENSOR_NONE16],
        i: [128, 5376, 0, 0, 0, 0, 0, 0],
        ..Default::default()
    };
    for gamma in [4, 5] {
        for output in [0, 1] {
            let mut bad = inst;
            bad.t[gamma] = bad.t[output];
            assert!(split_norm_residual_norm(&bad, 128).is_err());
        }
    }
}

#[test]
#[ignore = "requires paired ordinary PF_GFUSE off/on assets; no GPU"]
fn sandwich_norm_assets_synthesize_identical_mixed_programs() {
    let root = std::path::PathBuf::from(std::env::var_os("TEST_GFUSE_ROOT").unwrap());
    let mut results = Vec::new();
    for mode in ["off", "on"] {
        let raw = std::fs::read(root.join(format!("{mode}-assets/model.pkt"))).unwrap();
        let blob = DevBlob::parse_l2(&raw, true).unwrap();
        let batch = blob.decode_progs().last().unwrap().t as usize;
        results.push(synthesize(&blob, batch, false).unwrap());
    }
    let [off, on] = results.as_slice() else {
        unreachable!()
    };
    let tensors = |s: &SynthesizedMixed| {
        s.tensors
            .iter()
            .map(|t| (t.handle, t.name.clone(), t.bytes))
            .collect::<Vec<_>>()
    };
    assert_eq!(tensors(off), tensors(on));
    assert_eq!(off.programs.len(), on.programs.len());
    let mut evidence = Vec::new();
    for (a, b) in off.programs.iter().zip(&on.programs) {
        assert_eq!(
            (a.decode_rows, a.decode_slot),
            (b.decode_rows, b.decode_slot)
        );
        assert_eq!(a.program.insts.len(), b.program.insts.len());
        for (index, (x, y)) in a.program.insts.iter().zip(&b.program.insts).enumerate() {
            assert_eq!(x, y, "capacity {} instruction {index}", a.program.rows);
        }
        assert_eq!(a.program, b.program, "all queue/dependency tables");
        evidence.push(serde_json::json!({"capacity":a.program.rows,"instructions":a.program.insts.len(),"program_and_dependencies_exact":true}));
    }
    std::fs::write(
        root.join("synthesis-parity.json"),
        serde_json::to_vec_pretty(&evidence).unwrap(),
    )
    .unwrap();
    eprintln!("SYNTHESIS {}", serde_json::to_string(&evidence).unwrap());
}

// TEST_MIXED_ASSETS points at an ordinary compiled model; this test uses no GPU.
#[test]
#[ignore = "requires an ordinary dense BF16 model asset"]
fn ordinary_asset_synthesis() {
    let path = std::path::PathBuf::from(std::env::var_os("TEST_MIXED_ASSETS").unwrap());
    let raw = std::fs::read(path.join("model.pkt")).unwrap();
    let mut blob = DevBlob::parse_l2(&raw, true).unwrap();
    let expected: BTreeSet<_> = blob
        .progs
        .iter()
        .filter(|p| {
            !p.role.is_packed_sibling()
                && p.insts.iter().any(|i| i.op == DevOp::FlashPrefill as u16)
        })
        .map(|p| p.t)
        .collect();
    let kvlen = tensor(&blob, "in.kvlen").unwrap() as usize;
    let batch = (blob.tensors[kvlen].bytes / 4) as usize;
    {
        let mixed = synthesize(&blob, batch, false).unwrap();
        assert_eq!(
            mixed
                .programs
                .iter()
                .map(|p| p.program.rows)
                .collect::<BTreeSet<_>>(),
            expected
        );
        for spec in &mixed.programs {
            assert_eq!(
                spec.decode_rows,
                (batch as u32 - 1).min(spec.program.rows - 1)
            );
            assert_eq!(spec.program.gq_seg_ofs.len(), 2);
            assert!(spec
                .program
                .insts
                .iter()
                .all(|i| i.blocks as u32 == blob.n_cu));
            assert!(spec
                .program
                .insts
                .iter()
                .any(|i| i.op == DevOp::FlashDecode as u16));
            assert!(spec
                .program
                .insts
                .iter()
                .any(|i| i.op == DevOp::GemmGlu as u16));
        }
        for name in ["in.ids", "in.pos", "in.kvlen"] {
            assert!(mixed.tensors.iter().any(|t| t.name == name));
        }
        assert!(mixed
            .tensors
            .iter()
            .all(|t| t.name.starts_with("in.") || t.name.starts_with("act.")));
        eprintln!(
            "runtime synthesis batch={batch}: capacities={expected:?}, private tensors={}",
            mixed.tensors.len()
        );
    }
    assert!(synthesize(&blob, 1, false).is_err());
    assert!(synthesize(&blob, batch + 1, false).is_err());
    if batch > 2 {
        assert!(synthesize(&blob, batch - 1, false).is_err());
    }
    blob.tensors[kvlen].bytes += 4;
    assert!(synthesize(&blob, batch, false).is_err());
    blob.tensors[kvlen].bytes -= 4;
    let k = blob
        .progs
        .iter()
        .flat_map(|p| &p.insts)
        .find(|i| i.op == DevOp::FlashDecode as u16)
        .unwrap()
        .t[3] as usize;
    let size = blob.tensors[k].bytes;
    blob.tensors[k].bytes = 2;
    assert!(synthesize(&blob, batch, false).is_err());
    blob.tensors[k].bytes = size;
    let prefill = blob
        .progs
        .iter_mut()
        .find(|p| p.t > 1 && p.insts.iter().any(|i| i.op == DevOp::FlashPrefill as u16))
        .unwrap();
    let first = prefill.insts[0].op;
    prefill.insts[0].op = u16::MAX;
    assert!(synthesize(&blob, batch, false).is_err());
    let prefill = blob
        .progs
        .iter_mut()
        .find(|p| p.t > 1 && p.insts.iter().any(|i| i.op == DevOp::FlashPrefill as u16))
        .unwrap();
    prefill.insts[0].op = first;
    let flash = prefill
        .insts
        .iter_mut()
        .find(|i| i.op == DevOp::FlashPrefill as u16)
        .unwrap();
    flash.i[6] = 128;
    assert!(synthesize(&blob, batch, false).is_err());
}

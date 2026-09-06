use super::*;
use crate::asset::devblob::{DevProg, DevTensor};
use packet::dev::StreamEnt;

pub(super) fn fixture() -> DevBlob {
    let progs = [1, 2, 4, 8, 16]
        .into_iter()
        .map(|rows| {
            let mut insts = Vec::new();
            for id in [3, 4] {
                let mut d = DevInst64 {
                    op: DevOp::HeadNormRope as u16,
                    blocks: 1,
                    t: [TENSOR_NONE16; 8],
                    ..Default::default()
                };
                d.t[0] = id;
                d.t[5] = 0;
                d.i = [rows, 1, 256, 0, 0, 0, rows, 0];
                d.fj = [0, 1024, u32::MAX];
                insts.push(d);
            }
            let mut flash = DevInst64 {
                op: DevOp::FlashDecode as u16,
                blocks: 1,
                t: [TENSOR_NONE16; 8],
                ..Default::default()
            };
            flash.t[..6].copy_from_slice(&[5, 6, 7, 3, 4, 1]);
            flash.i = [rows, 8, 1, 1024, 0, 2, 256, u32::MAX];
            flash.fj[1] = if rows > 1 { rows * 1024 } else { 0 };
            insts.push(flash);
            let mut merge = DevInst64 {
                op: DevOp::FlashMerge as u16,
                blocks: 1,
                t: [TENSOR_NONE16; 8],
                ..Default::default()
            };
            merge.t[..3].copy_from_slice(&[7, 5, 6]);
            merge.i[..4].copy_from_slice(&[rows, 8, 2, 256]);
            insts.push(merge);
            let stream: Vec<_> = (0..insts.len())
                .map(|ix| StreamEnt {
                    inst: ix as u32,
                    ..Default::default()
                })
                .collect();
            DevProg {
                t: rows,
                packed_prefill_only: false,
                n_counter: 0,
                insts,
                stream: stream.clone(),
                stream_ofs: vec![0],
                stream_len: vec![4],
                waits: vec![],
                succs: vec![],
                gq_stream: stream,
                gq_seg_ofs: vec![0, 4],
                l2_domains: 0,
            }
        })
        .collect();
    let tensors = [
        ("in.pos", 4096),
        ("in.kvlen", 64),
        ("in.ids", 64),
        ("key", 16 * 1024 * 256 * 2),
        ("value", 16 * 1024 * 256 * 2),
        ("partial", 16 * 8 * 2 * 256 * 4),
        ("ml", 16 * 8 * 2 * 8),
        ("query", 16 * 8 * 256 * 2),
    ]
    .into_iter()
    .map(|(name, bytes)| DevTensor {
        name: name.into(),
        bytes,
        init: None,
    })
    .collect();
    DevBlob {
        n_cu: 1,
        flags: 0,
        target: 0,
        tensors,
        init: vec![],
        kvrow: vec![],
        progs,
        sections: vec![],
        gen: vec![],
        tp: None,
    }
}

#[test]
fn selects_by_highest_physical_slot_and_preserves_sparse_slots() {
    for (feeds, expected) in [
        (vec![0], Some(0)),
        (vec![1], Some(1)),
        (vec![0, 3], Some(2)),
        (vec![7, 0], Some(3)),
        (vec![0, 8], None),
        (vec![15], None),
    ] {
        assert_eq!(
            decode_rung_index([1, 2, 4, 8].into_iter(), *feeds.iter().max().unwrap()),
            expected
        );
    }
    assert_eq!(decode_rung_index(std::iter::empty(), 0), None);
}

#[test]
fn effective_widths_include_main_and_preserve_widest_only_fallbacks() {
    assert_eq!(
        effective_decode_widths([1, 2, 4, 8].into_iter(), 16, false).as_ref(),
        [1, 2, 4, 8, 16]
    );
    assert_eq!(
        effective_decode_widths([1, 2, 4, 8].into_iter(), 16, true).as_ref(),
        [16]
    );
    assert_eq!(
        effective_decode_widths(std::iter::empty(), 16, true).as_ref(),
        [16]
    );
}

#[test]
fn validates_direct_kv_ladder_and_rejects_stale_slot_addressing() {
    assert!(validate_decode_ladder(&fixture()).unwrap());
    let mutations: &[fn(&mut DevBlob)] = &[
        |b| b.progs[0].insts[0].i[6] = 0,
        |b| b.progs[1].insts[0].i[6] = 16,
        |b| b.progs[0].insts[0].t[5] = 1,
        |b| b.progs[0].insts[0].fj[1] = 512,
        |b| b.progs[0].insts[2].i[3] = 512,
        |b| b.progs[0].insts[2].t[5] = 0,
        |b| b.progs[0].insts[2].i[0] = 16,
        |b| b.progs[0].insts[3].i[2] = 3,
        |b| b.tensors[3].bytes /= 2,
        |b| b.tensors[5].bytes = 1,
        |b| b.progs[0].insts[2].i[5] = u32::MAX,
        |b| b.progs[0].insts[2].t[3] = 7,
        |b| b.progs[0].gq_seg_ofs[1] = 3,
        |b| b.progs[0].gq_stream[0].slice = 1,
        |b| b.progs[0].stream[0].wait_len = 1,
        |b| b.progs[0].stream_len[0] = 5,
    ];
    for (case, mutate) in mutations.iter().enumerate() {
        let mut blob = fixture();
        mutate(&mut blob);
        assert!(validate_decode_ladder(&blob).is_err(), "case={case}");
    }
}

#[test]
fn decode_ladder_accepts_runtime_input_capacity_beyond_widest_rung() {
    let mut blob = fixture();
    blob.tensors[1].bytes = 128 * 4;
    assert!(validate_decode_ladder(&blob).unwrap());
}

#[test]
fn unsupported_families_and_single_rung_keep_widest_execution() {
    let mut blob = fixture();
    for g in &mut blob.progs {
        g.insts[2].op = DevOp::FlashDecodeFp8 as u16;
    }
    assert!(!validate_decode_ladder(&blob).unwrap());
    let mut blob = fixture();
    for g in &mut blob.progs {
        g.insts[0].t[7] = 7;
    }
    assert!(!validate_decode_ladder(&blob).unwrap());
    let mut blob = fixture();
    blob.progs.drain(..4);
    blob.progs[0].insts[0].i[6] = 0;
    assert!(!validate_decode_ladder(&blob).unwrap());
}

#[test]
#[ignore = "CPU-only actual packet check; set TEST_DECODE_RUNG_PACKET"]
fn actual_packet_decode_ladder() {
    let path = std::env::var("TEST_DECODE_RUNG_PACKET").unwrap();
    let blob = DevBlob::parse(&std::fs::read(&path).unwrap()).unwrap();
    assert!(
        validate_decode_ladder(&blob).unwrap(),
        "ladder did not qualify: {path}"
    );
    eprintln!("{path}: qualified widths {:?}", blob.decode_rungs());
}

#[test]
#[ignore = "GPU full-logit gate; set TEST_DECODE_RUNG_GPU, TEST_DECODE_RUNG_ASSETS, TEST_DECODE_RUNG_BASELINE"]
fn gpu_decode_rungs_match_widest_full_logits() {
    assert_eq!(std::env::var("TEST_DECODE_RUNG_GPU").as_deref(), Ok("1"));
    let ladder = std::path::PathBuf::from(std::env::var("TEST_DECODE_RUNG_ASSETS").unwrap());
    let baseline = std::path::PathBuf::from(std::env::var("TEST_DECODE_RUNG_BASELINE").unwrap());
    let read_blob = |assets: &std::path::Path| {
        let path = DevBlob::find_in_dir(assets).unwrap().unwrap();
        DevBlob::parse(&std::fs::read(path).unwrap()).unwrap()
    };
    let base_blob = read_blob(&baseline);
    let ladder_blob = read_blob(&ladder);
    let batch = base_blob.decode_prog().unwrap().t as usize;
    let widths: Vec<_> = ladder_blob
        .decode_rungs()
        .into_iter()
        .map(|w| w as usize)
        .collect();
    assert!(batch >= 2 && widths.len() > 1);
    assert_eq!(base_blob.decode_rungs(), [batch as u32]);
    assert_eq!(widths.last(), Some(&batch));
    assert!(validate_decode_ladder(&ladder_blob).unwrap());
    assert_eq!(
        base_blob.decode_prog().unwrap().insts,
        ladder_blob.decode_prog().unwrap().insts,
        "baseline widest instructions differ"
    );
    assert!(
        base_blob
            .tensors
            .iter()
            .map(|t| (&t.name, t.bytes))
            .eq(ladder_blob.tensors.iter().map(|t| (&t.name, t.bytes))),
        "baseline tensor geometry differs"
    );
    drop((base_blob, ladder_blob));
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut references: Vec<(String, u32, Vec<u32>)> = Vec::new();
    for (candidate, assets) in [(false, &baseline), (true, &ladder)] {
        let mut e = GpuEngine::load(Arc::clone(&be), assets, &assets.join("checkpoint")).unwrap();
        assert_eq!(e.batch(), batch);
        assert!(
            e.cublaslt_decode.is_empty() && e.multistep.is_none(),
            "disable Lt and multistep for this gate"
        );
        if candidate {
            assert_eq!(
                e.decode_rungs.iter().map(|r| r.rows).collect::<Vec<_>>(),
                widths[..widths.len() - 1]
            );
        } else {
            assert!(e.decode_rungs.is_empty());
        }
        let rows = e
            .prefill
            .iter()
            .map(|p| p.t)
            .min()
            .expect("prefill required") as usize;
        let mut checked = 0;
        let mut compare = |e: &mut GpuEngine, row: usize, token: u32, tag: String| {
            let mut logits = Vec::new();
            e.logits_row(row, &mut logits).unwrap();
            assert!(
                !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
                "{tag}: invalid logits"
            );
            let bits: Vec<_> = logits.iter().map(|v| v.to_bits()).collect();
            if candidate {
                let (expected_tag, expected_token, expected) = &references[checked];
                assert_eq!(&tag, expected_tag);
                assert_eq!(token, *expected_token, "{tag}: greedy token");
                assert_eq!(bits.len(), expected.len());
                if let Some((index, (actual, want))) = bits
                    .iter()
                    .zip(expected)
                    .enumerate()
                    .find(|(_, (a, b))| a != b)
                {
                    panic!("{tag}: logit {index} differs: {actual:#010x} vs {want:#010x}");
                }
            } else {
                references.push((tag, token, bits));
            }
            checked += 1;
        };
        let mut tokens = vec![0u32; batch];
        let mut decoded = Vec::new();
        for slot in 0..batch {
            let prompt: Vec<_> = (0..rows)
                .map(|i| 100 + ((i * (2 * slot + 1) + 173 * slot) % 1000) as u32)
                .collect();
            e.begin_slot(slot, rows + 64).unwrap();
            tokens[slot] = e.prefill_slot(slot, &prompt).unwrap();
            compare(&mut e, 0, tokens[slot], format!("prefill slot={slot}"));
        }
        let schedule: &[&[usize]] = &[
            &[0],
            &[0],
            &[1],
            &[0, 3],
            &[7],
            &[0, 8],
            &[15],
            &[0, 1, 2, 3],
            &[2],
            &[7, 0],
            &[0],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ];
        let mut schedule: Vec<Vec<usize>> = schedule
            .iter()
            .map(|slots| {
                slots
                    .iter()
                    .copied()
                    .filter(|&slot| slot < batch)
                    .collect::<Vec<_>>()
            })
            .filter(|slots| !slots.is_empty())
            .collect();
        schedule.push(vec![batch - 1]);
        schedule.push((0..batch).collect());
        for phase in 0..2 {
            if phase == 1 {
                let prompt: Vec<_> = (0..rows + 1)
                    .map(|i| 100 + ((i * 13 + 79) % 1000) as u32)
                    .collect();
                let reset_slot = 7.min(batch - 1);
                e.begin_slot(reset_slot, prompt.len() + 64).unwrap();
                tokens[reset_slot] = e.prefill_slot(reset_slot, &prompt).unwrap();
                compare(
                    &mut e,
                    0,
                    tokens[reset_slot],
                    format!("reset slot={reset_slot} padded prefill"),
                );
                tokens[1] = e.consume_prompt(1, &[341, 617, 829], &mut decoded).unwrap();
                compare(&mut e, 1, tokens[1], "slot=1 consume continuation".into());
            }
            for (step, slots) in schedule.iter().enumerate() {
                let highest = *slots.iter().max().unwrap();
                let selected = e
                    .decode_rung(highest)
                    .map_or(batch, |ix| e.decode_rungs[ix].rows);
                if candidate {
                    assert_eq!(
                        selected,
                        *widths.iter().find(|&&width| width > highest).unwrap()
                    );
                } else {
                    assert_eq!(selected, batch);
                }
                let feeds: Vec<_> = slots.iter().map(|&slot| (slot, tokens[slot])).collect();
                e.step_slots(&feeds, &mut decoded).unwrap();
                assert_eq!(decoded.len(), slots.len());
                for (i, &slot) in slots.iter().enumerate() {
                    tokens[slot] = decoded[i];
                    compare(
                        &mut e,
                        slot,
                        tokens[slot],
                        format!("phase={phase} step={step} slot={slot}"),
                    );
                }
                eprintln!("candidate={candidate} phase={phase} step={step} slots={slots:?} rung={selected}: full logits exact");
            }
        }
        eprintln!("candidate={candidate}: {checked} full-logit snapshots");
    }
}

pub(super) fn splitk_fixture() -> DevBlob {
    use packet::devbuild::{Builder, Model, TensorDecl};
    let old = fixture();
    let mut tensors: Vec<_> = old
        .tensors
        .iter()
        .map(|t| TensorDecl {
            name: t.name.clone(),
            bytes: t.bytes,
            init: None,
        })
        .collect();
    tensors.extend([
        TensorDecl {
            name: "act.a".into(),
            bytes: 16 * 64 * 2,
            init: None,
        },
        TensorDecl {
            name: "model.layers.0.mlp.down_proj.weight".into(),
            bytes: 128 * 64 * 2,
            init: None,
        },
        TensorDecl {
            name: "act.c".into(),
            bytes: 16 * 128 * 2,
            init: None,
        },
        TensorDecl {
            name: "act.partial".into(),
            bytes: 16 * 128 * 4,
            init: None,
        },
    ]);
    let mut programs = Vec::new();
    for p in old.progs {
        let mut b = Builder::new(1);
        b.force_uniseg();
        b.adopt_tensors(tensors.clone());
        let mut prior = None;
        for d in p.insts {
            prior = Some(b.emit(
                DevOp::from_u16(d.op).unwrap(),
                vec![0],
                &prior.into_iter().collect::<Vec<_>>(),
                |i| {
                    i.t = d.t.map(|h| {
                        if h == TENSOR_NONE16 {
                            packet::dev::TENSOR_NONE
                        } else {
                            u32::from(h)
                        }
                    });
                    i.i = d.i;
                    i.f = [f32::from_bits(d.fj[0]), 0.];
                    i.j = [d.fj[1], d.fj[2]];
                },
            ));
        }
        if p.t < 4 {
            b.emit(
                DevOp::Gemv,
                vec![0],
                &prior.into_iter().collect::<Vec<_>>(),
                |i| {
                    i.t[..3].copy_from_slice(&[10, 8, 9]);
                    i.i[..3].copy_from_slice(&[p.t, 128, 64]);
                    i.f[0] = 1e-6;
                },
            );
        } else {
            let z = b.emit(DevOp::ZeroF32, vec![0], &[], |i| {
                i.t[0] = 11;
                i.i[..2].copy_from_slice(&[p.t, 128]);
            });
            let g = b.emit(DevOp::GemmSplitK, vec![0], &[prior.unwrap(), z], |i| {
                i.t[..3].copy_from_slice(&[11, 8, 9]);
                i.i[..4].copy_from_slice(&[p.t, 128, 64, 8]);
            });
            b.emit(DevOp::CastF32Bf16, vec![0], &[g], |i| {
                i.t[..2].copy_from_slice(&[10, 11]);
                i.i[..2].copy_from_slice(&[p.t, 128]);
            });
        }
        programs.push(b.finish());
    }
    DevBlob::parse(
        &Model {
            n_cu: 1,
            target: 0,
            tensors,
            progs: programs,
            prog_t: vec![1, 2, 4, 8, 16],
            gen: vec![],
            kv_row_insts: vec![],
        }
        .to_blob(),
    )
    .unwrap()
}
#[test]
fn canonical_splitk_ladder_preserves_shapes_and_rejects_changed_dependency() {
    assert!(validate_decode_ladder(&splitk_fixture()).unwrap());
    let mut b = splitk_fixture();
    let g = &mut b.progs[2];
    for e in g
        .stream
        .iter_mut()
        .chain(&mut g.gq_stream)
        .filter(|e| e.inst == 5)
    {
        let zero = g.waits[e.wait_ofs as usize..e.wait_ofs as usize + e.wait_len as usize]
            .iter()
            .position(|w| w.id == 4)
            .unwrap();
        e.wait_ofs += zero as u32;
        e.wait_len = 1;
    }
    assert!(validate_decode_ladder(&b).is_err());
    let mut b = splitk_fixture();
    b.progs[2].insts[5].i[1] = 64;
    b.progs[2].insts[4].i[1] = 64;
    b.progs[2].insts[6].i[1] = 64;
    assert!(validate_decode_ladder(&b).is_err());
}
#[test]
fn bound_projection_requires_capability_only_on_assigned_e3_rungs() {
    let b = splitk_fixture();
    let coverage = plow_asset::decode_coverage::DenseBf16([1, 2, 8, 16448, 16, 82944]);
    b.with_packet_view(|p| {
        assert!(coverage.program(p, 0, None).is_ok());
        assert!(coverage.program(p, 1, None).is_ok());
        assert!(coverage.program(p, 2, None).is_err());
        assert!(coverage.program(p, 2, Some(0)).is_err());
        assert!(coverage.program(p, 2, Some(1)).is_ok());
        assert!(
            plow_asset::decode_coverage::DenseBf16([1, 2, 8, 16448, 16, 16448])
                .program(p, 2, Some(1))
                .is_err()
        );
    });
}

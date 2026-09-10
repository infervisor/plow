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

fn cublaslt_fixture() -> (DevBlob, SegmentRoles) {
    let mut blob = fixture();
    for name in ["projection.out", "projection.in", "model.layers.0.weight"] {
        blob.tensors.push(DevTensor {
            name: name.into(),
            bytes: 16 * 16 * 2,
            init: None,
        });
    }
    for g in &mut blob.progs {
        let mut projection = DevInst64 {
            op: DevOp::Gemv as u16,
            blocks: 1,
            t: [TENSOR_NONE16; 8],
            ..Default::default()
        };
        projection.t[..3].copy_from_slice(&[8, 9, 10]);
        projection.i[..3].copy_from_slice(&[g.t, 16, 16]);
        g.insts.push(projection);
        g.n_counter = 5;
        g.succs = (0..5).collect();
        g.waits = (0..4)
            .map(|id| packet::dev::Wait { id, threshold: 1 })
            .collect();
        g.stream = (0..5)
            .map(|pc| StreamEnt {
                inst: pc,
                seg: u16::from(pc == 4),
                succ_ofs: pc,
                succ_len: 1,
                wait_ofs: pc.saturating_sub(1),
                wait_len: u16::from(pc != 0),
                ..Default::default()
            })
            .collect();
        g.gq_stream = g.stream.clone();
        g.stream_len = vec![5];
        g.gq_seg_ofs = vec![0, 4, 5];
    }
    let metadata = SegmentRoles {
        version: 1,
        objects: Default::default(),
        programs: (0..blob.progs.len())
            .map(|index| plow_asset::segment_roles::ProgramRoles {
                index,
                roles: vec![
                    plow_asset::segment_roles::INTERPRETER,
                    plow_asset::segment_roles::CUBLASLT,
                ],
            })
            .collect(),
    };
    (blob, metadata)
}

#[test]
fn cublaslt_ladder_requires_complete_equivalent_roles_and_dependencies() {
    let (blob, mut metadata) = cublaslt_fixture();
    assert!(validate_cublaslt_ladder(&blob, &metadata).unwrap());
    assert!(!validate_decode_ladder(&blob).unwrap());
    let removed = metadata.programs.remove(1);
    assert!(validate_cublaslt_ladder(&blob, &metadata)
        .unwrap_err()
        .to_string()
        .contains("every decode width"));
    metadata.programs.insert(1, removed);
    metadata.programs[1].roles[1] = plow_asset::segment_roles::INTERPRETER;
    assert!(validate_cublaslt_ladder(&blob, &metadata)
        .unwrap_err()
        .to_string()
        .contains("roles differ"));

    let (mut blob, metadata) = cublaslt_fixture();
    blob.progs[1].waits[3].id = 0;
    assert!(validate_cublaslt_ladder(&blob, &metadata)
        .unwrap_err()
        .to_string()
        .contains("dependencies differ"));
    blob.progs[1].waits[3].id = 3;
    blob.progs[1].waits[3].threshold = 0;
    assert!(validate_cublaslt_ladder(&blob, &metadata).is_err());
}

#[test]
fn cublaslt_ladder_rejects_stale_kv_addressing_and_invalid_projection_storage() {
    let (mut blob, metadata) = cublaslt_fixture();
    blob.progs[1].insts[0].i[6] = 16;
    assert!(validate_cublaslt_ladder(&blob, &metadata).is_err());
    blob.progs[1].insts[0].i[6] = 2;
    blob.progs[1].insts[2].i[3] = 512;
    assert!(validate_cublaslt_ladder(&blob, &metadata).is_err());
    blob.progs[1].insts[2].i[3] = 1024;
    assert!(validate_cublaslt_ladder(&blob, &metadata).unwrap());
    blob.tensors[8].bytes -= 2;
    assert!(validate_cublaslt_ladder(&blob, &metadata).is_err());
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
fn validates_channel_fp8_rows_and_preserves_projection_and_kv_checks() {
    for op in [DevOp::GemvFp8, DevOp::GemvGluFp8] {
        let mut blob = fixture();
        for (name, bytes) in [("x", 256), ("wg", 128), ("sg", 64), ("wu", 128), ("su", 64)] {
            blob.tensors.push(DevTensor {
                name: name.into(),
                bytes,
                init: None,
            });
        }
        for g in &mut blob.progs {
            let mut d = DevInst64 {
                op: op as u16,
                blocks: 1,
                t: [TENSOR_NONE16; 8],
                ..Default::default()
            };
            d.t[..3].copy_from_slice(&[7, 8, 9]);
            if op == DevOp::GemvFp8 {
                d.t[5] = 10;
            } else {
                d.t[3..6].copy_from_slice(&[10, 12, 11]);
            }
            d.i[..3].copy_from_slice(&[g.t, 16, 8]);
            let entry = StreamEnt {
                inst: g.insts.len() as u32,
                ..Default::default()
            };
            g.insts.push(d);
            g.stream.push(entry);
            g.gq_stream.push(entry);
            g.stream_len[0] += 1;
            g.gq_seg_ofs[1] += 1;
        }
        assert!(validate_decode_ladder(&blob).unwrap());
        blob.progs[0].insts[4].i[0] = 2;
        assert!(validate_decode_ladder(&blob).is_err());
        blob.progs[0].insts[4].i[0] = 1;
        blob.progs[0].insts[4].t[5] = 12;
        assert!(!validate_decode_ladder(&blob).unwrap());
        blob.progs[0].insts[4].t[5] = if op == DevOp::GemvFp8 { 10 } else { 11 };
        blob.progs[0].insts[0].i[6] = 0;
        assert!(validate_decode_ladder(&blob).is_err());
    }
}

#[test]
fn validates_hd64_half_split_attention_ladder() {
    let mut blob = fixture();
    for g in &mut blob.progs {
        for d in &mut g.insts[..2] {
            d.i[2] = 64;
        }
        g.insts[0].i[5] = packet::dev::ROPE_PAIR_HALF;
        g.insts[2].i[6] = 64;
        g.insts[3].i[3] = 64;
    }
    for t in &mut blob.tensors[3..5] {
        t.bytes /= 4;
    }
    blob.tensors[5].bytes /= 4;
    blob.tensors[7].bytes /= 4;
    assert!(validate_decode_ladder(&blob).unwrap());
    blob.progs[0].insts[0].i[5] = 0;
    assert!(validate_decode_ladder(&blob).is_err());
}

fn append_flat_mxfp4_moe(blob: &mut DevBlob) {
    for g in &mut blob.progs {
        let rows = g.t;
        let mut router = DevInst64 {
            op: DevOp::MoeRouterTopkPf as u16,
            blocks: 1,
            t: [TENSOR_NONE16; 8],
            ..Default::default()
        };
        router.i = [0, 32, 4, 2, rows, 0, 0, 0];
        let mut glu = DevInst64 {
            op: DevOp::MoeGluMx as u16,
            blocks: 1,
            t: [TENSOR_NONE16; 8],
            ..Default::default()
        };
        glu.i = [4, 2880, 2880, 32, 0, 3, rows, 0];
        let mut down = DevInst64 {
            op: DevOp::MoeDownMx as u16,
            blocks: 1,
            t: [TENSOR_NONE16; 8],
            ..Default::default()
        };
        down.i = [4, 2880, 2880, 32, 0, 0, rows, 0];
        let mut combine = DevInst64 {
            op: DevOp::MoeCombinePf as u16,
            blocks: 1,
            t: [TENSOR_NONE16; 8],
            ..Default::default()
        };
        combine.i = [2880, 4, rows, 0, 0, 0, 0, 0];
        for d in [router, glu, down, combine] {
            let entry = StreamEnt {
                inst: g.insts.len() as u32,
                ..Default::default()
            };
            g.insts.push(d);
            g.stream.push(entry);
            g.gq_stream.push(entry);
        }
        g.stream_len[0] = g.stream.len() as u32;
        g.gq_seg_ofs[1] = g.gq_stream.len() as u32;
    }
}

#[test]
fn validates_flat_mxfp4_moe_decode_ladder_rows() {
    let mut blob = fixture();
    append_flat_mxfp4_moe(&mut blob);
    assert!(validate_decode_ladder(&blob).unwrap());
    for (inst, field) in [(4, 4), (5, 6), (6, 6), (7, 2)] {
        let mut invalid = fixture();
        append_flat_mxfp4_moe(&mut invalid);
        invalid.progs[1].insts[inst].i[field] = 1;
        assert!(validate_decode_ladder(&invalid).is_err());
    }
}

#[test]
fn fine_gated_prefill_selects_segment_pair_from_packet_topology() {
    let mut blob = fixture();
    let mut prefill = blob.progs.remove(0);
    prefill.t = 256;
    prefill.stream[0].flags |= packet::dev::SE_FINE;
    blob.progs.insert(0, prefill);
    assert!(prefill_needs_segment_pair(&blob, None));
    blob.progs[0].stream[0].flags &= !packet::dev::SE_FINE;
    assert!(!prefill_needs_segment_pair(&blob, None));
}

#[test]
#[ignore = "CPU-only actual packet check; set TEST_SEGMENTED_PREFILL_PACKET"]
fn actual_seven_program_packet_selects_segment_pair() {
    let path = std::env::var("TEST_SEGMENTED_PREFILL_PACKET").unwrap();
    let blob = DevBlob::parse(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(blob.progs.len(), 7);
    assert_eq!(blob.prefill_progs().len(), 3);
    assert!(prefill_needs_segment_pair(&blob, None));
}

#[test]
fn decode_ladder_accepts_runtime_input_capacity_beyond_widest_rung() {
    let mut blob = fixture();
    blob.tensors[1].bytes = 128 * 4;
    assert!(validate_decode_ladder(&blob).unwrap());
}

fn fp8_kv_fixture() -> DevBlob {
    let mut blob = fixture();
    for index in [3, 4] {
        blob.tensors[index].bytes /= 2;
    }
    for name in ["key.scale", "value.scale"] {
        blob.tensors.push(DevTensor {
            name: name.into(),
            bytes: 16 * 1024 * 4,
            init: None,
        });
    }
    for g in &mut blob.progs {
        for index in 0..2 {
            g.insts[index].op = DevOp::HeadNormRopeFp8 as u16;
            g.insts[index].t[6] = 8 + index as u16;
        }
        g.insts[2].op = DevOp::FlashDecodeFp8 as u16;
        g.insts[2].t[6..8].copy_from_slice(&[8, 9]);
    }
    blob
}

#[test]
fn fp8_kv_ladder_validates_full_and_sliding_cache_geometry() {
    for hd in [256, 512] {
        for window in [0, 512] {
            let mut blob = fp8_kv_fixture();
            for index in [3, 4, 5, 7] {
                blob.tensors[index].bytes *= u64::from(hd / 256);
            }
            for g in &mut blob.progs {
                for index in 0..2 {
                    g.insts[index].i[2] = hd;
                    g.insts[index].fj[2] = if window == 0 { u32::MAX } else { 1023 };
                }
                g.insts[2].i[4] = window;
                g.insts[2].i[6] = hd;
                g.insts[2].i[7] = if window == 0 { u32::MAX } else { 1023 };
                g.insts[3].i[3] = hd;
            }
            assert!(
                validate_decode_ladder(&blob).unwrap(),
                "hd={hd} window={window}"
            );
        }
    }
}

#[test]
fn fp8_kv_ladder_rejects_invalid_scale_storage_and_aliases() {
    for mutation in 0..6 {
        let mut blob = fp8_kv_fixture();
        match mutation {
            0 => blob.tensors[8].bytes -= 4,
            1 => blob.tensors[9].bytes *= 2,
            2 => blob.tensors[8].init = Some(0..4),
            3 => {
                for g in &mut blob.progs {
                    g.insts[2].t[7] = 8;
                }
            }
            4 => {
                for g in &mut blob.progs {
                    g.insts[2].t[6] = TENSOR_NONE16;
                }
            }
            5 => {
                for g in &mut blob.progs {
                    g.insts[2].t[6] = 3;
                }
            }
            _ => unreachable!(),
        }
        assert!(
            validate_decode_ladder(&blob).is_err(),
            "mutation={mutation}"
        );
    }
}

#[test]
fn fp8_kv_ladder_rejects_changed_reader_writer_and_scale_addressing() {
    for mutation in 0..6 {
        let mut blob = fp8_kv_fixture();
        let g = &mut blob.progs[0];
        match mutation {
            0 => g.insts[0].t[6] = 9,
            1 => g.insts[2].t[6..8].swap(0, 1),
            2 => g.insts[0].i[6] = 0,
            3 => g.insts[0].op = DevOp::HeadNormRope as u16,
            4 => g.insts[2].op = DevOp::FlashDecode as u16,
            5 => g.insts[3].t[0] = 8,
            _ => unreachable!(),
        }
        assert!(
            validate_decode_ladder(&blob).is_err(),
            "mutation={mutation}"
        );
    }
}

#[test]
fn unsupported_families_and_single_rung_keep_widest_execution() {
    let mut blob = fixture();
    for g in &mut blob.progs {
        g.insts[2].op = DevOp::FlashPrefillFp8 as u16;
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
    check_gpu_decode_rungs(false);
}

#[test]
#[ignore = "GPU cuBLASLt gate; set TEST_DECODE_RUNG_GPU, TEST_DECODE_RUNG_ASSETS, TEST_DECODE_RUNG_BASELINE"]
fn gpu_cublaslt_rungs_match_widest_logits() {
    check_gpu_decode_rungs(true);
}

#[test]
#[ignore = "H100 packed-schedule diagnostic; set TEST_DECODE_RUNG_ASSETS and TEST_PACKED_LOGITS_OUT"]
fn gpu_packed_prefill_schedule_logits() {
    let assets = std::path::PathBuf::from(std::env::var("TEST_DECODE_RUNG_ASSETS").unwrap());
    let output = std::env::var("TEST_PACKED_LOGITS_OUT").unwrap();
    let mut e = GpuEngine::load(
        Arc::new(CudaBackend::new(0).unwrap()),
        &assets,
        &assets.join("checkpoint"),
    )
    .unwrap();
    assert!(e.batch() >= 16 && e.pf_max_rows() >= 4096);
    let prompt = vec![29104u32; 16384]; // Gemma 4 tokenizer: repeated " hello".
    let prompts: Vec<_> = [29104, 1902, 1594, 1262]
        .map(|token| vec![token; prompt.len()])
        .into();
    let mut reference: Vec<Vec<f32>> = Vec::new();
    let mut feeds = Vec::new();
    let mut report = Vec::new();
    for (pass, (slots, chunk)) in [
        (vec![0], 4096),
        (vec![0, 4, 8, 12], 1024),
        (vec![0], 1024),
        (vec![12, 8, 4, 0], 1024),
        ((0..16).collect(), 256),
        ((0..16).rev().collect(), 256),
        (vec![0, 4, 8, 12], 1024),
    ]
    .into_iter()
    .enumerate()
    {
        for &slot in &slots {
            e.begin_slot(slot, prompt.len() + 128).unwrap();
        }
        for c0 in (0..prompt.len() - 1).step_by(chunk) {
            let requests: Vec<_> = slots
                .iter()
                .map(|&slot| PfBatchReq {
                    slot,
                    prompt: &prompts[slot % 4],
                    c0,
                    len: chunk.min(prompt.len() - 1 - c0),
                })
                .collect();
            e.prefill_batched(&requests).unwrap();
        }
        let mut next = *prompt.last().unwrap();
        for step in 0..128 {
            let requests: Vec<_> = slots
                .iter()
                .map(|&slot| {
                    (
                        slot,
                        if step == 0 {
                            *prompts[slot % 4].last().unwrap()
                        } else {
                            next
                        },
                    )
                })
                .collect();
            let mut ids = Vec::new();
            let unified = pass == 6;
            if unified {
                use plow_asset::token_batch::{Phase, Request, Selection};
                let rows: Vec<_> = requests
                    .iter()
                    .map(|&(slot, ref token)| Request {
                        id: slot as u32,
                        slot: slot as u32,
                        state_slot: slot as u32,
                        generation: e.slot_generation(slot).unwrap(),
                        phase: if step == 0 {
                            Phase::Prefill
                        } else {
                            Phase::Decode
                        },
                        tokens: std::slice::from_ref(token),
                        prompt_len: prompt.len() as u32,
                        selection: Selection::default(),
                    })
                    .collect();
                let mut selected = Vec::new();
                e.token_batch_step(&rows, &mut selected).unwrap();
                ids = selected.iter().map(|&(_, token)| token).collect();
            } else {
                e.step_slots(&requests, &mut ids).unwrap();
            }
            for (i, &slot) in slots.iter().enumerate() {
                if slot % 4 != 0 {
                    continue;
                }
                let mut logits = Vec::new();
                e.logits_row(if unified { i } else { slot }, &mut logits)
                    .unwrap();
                assert!(logits.len() == e.vocab && logits.iter().all(|v| v.is_finite()));
                if pass == 0 {
                    reference.push(logits.clone());
                    feeds.push(ids[i]);
                }
                let expected = &reference[step];
                let changed = logits
                    .iter()
                    .zip(expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let max_abs = logits
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                report.push(serde_json::json!({
                    "pass": pass, "slots": slots, "chunk": chunk, "step": step, "unified": unified,
                    "slot": slot, "changed_logits": changed, "max_abs": max_abs,
                    "logits_sha256": plow_asset::decode_objects::image_sha256(bytemuck::cast_slice(&logits)),
                    "token": ids[i], "reference_token": feeds[step],
                    "reference_token_logit": logits[feeds[step] as usize],
                    "selected_token_logit": logits[ids[i] as usize],
                }));
            }
            next = feeds[step];
        }
        for &slot in &slots {
            e.retire_slot(slot, false);
        }
        eprintln!(
            "packed schedule pass={pass} slots={slots:?} chunk={chunk}: 128 teacher-forced frames"
        );
    }
    std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}

#[test]
#[ignore = "H100 FP8-KV prefix gate; set TEST_FP8_KV_ASSETS and PLOW_VMM_PREFIX=1, PLOW_MULTISTEP=0"]
fn gpu_fp8_kv_cached_rungs_match_uncached_suffix_logits() {
    let assets = std::path::PathBuf::from(std::env::var("TEST_FP8_KV_ASSETS").unwrap());
    let mut e = GpuEngine::load(
        Arc::new(CudaBackend::new(0).unwrap()),
        &assets,
        &assets.join("checkpoint"),
    )
    .unwrap();
    assert!(e.vmm_prefix_enabled() && e.multistep.is_none());
    assert_eq!(e.batch(), 16);
    assert!(e.prefill.iter().all(|p| p.fp8_kv));
    assert_eq!(
        e.decode_rungs.iter().map(|r| r.rows).collect::<Vec<_>>(),
        [1, 2, 4, 8]
    );
    for length in [127, 129, 513, 1057, 2113, 16417] {
        let prompt: Vec<_> = (0..length)
            .map(|i| 100 + ((i * 13 + length * 17) % 2000) as u32)
            .collect();
        let mut reference = Vec::new();
        for (pass, slot) in [15, 0, 1, 3, 7, 15].into_iter().enumerate() {
            e.begin_slot(slot, length + 8).unwrap();
            assert_eq!(e.vmm.as_ref().unwrap().kv.mapped_rows(slot), 0);
            let cached = e.attach_prompt(slot, &prompt).unwrap();
            assert_eq!(cached, if pass == 0 { 0 } else { (length - 1) / 32 * 32 });
            let mut next = e.prefill_slot(slot, &prompt).unwrap();
            if pass == 0 {
                // Recompute the suffix with the warm path's bucket while keeping
                // the independently computed prefix resident in its original slot.
                let mut cold = Vec::new();
                e.logits_row(0, &mut cold).unwrap();
                e.pos[slot] = ((length - 1) / 32 * 32) as u32;
                next = e.prefill_slot(slot, &prompt).unwrap();
                let mut suffix = Vec::new();
                e.logits_row(0, &mut suffix).unwrap();
                let max_abs = cold
                    .iter()
                    .zip(&suffix)
                    .map(|(a, b)| (*a - *b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("FP8 KV length={length}: cold vs suffix-bucket max_abs={max_abs}");
            }
            for step in 0..8 {
                let mut logits = Vec::new();
                e.logits_row(if step == 0 { 0 } else { slot }, &mut logits)
                    .unwrap();
                assert_eq!(logits.len(), e.vocab);
                assert!(logits.iter().all(|v| v.is_finite()));
                let bits: Vec<_> = logits.iter().map(|v| v.to_bits()).collect();
                if pass == 0 {
                    reference.push(bits);
                } else {
                    let max_abs = logits
                        .iter()
                        .zip(&reference[step])
                        .map(|(a, b)| (*a - f32::from_bits(*b)).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        bits == reference[step],
                        "length={length} slot={slot} step={step} max_abs={max_abs}"
                    );
                }
                if step < 7 {
                    let mut ids = Vec::new();
                    e.step_slots(&[(slot, next)], &mut ids).unwrap();
                    next = ids[0];
                }
            }
            e.retire_slot(slot, false);
            eprintln!("FP8 KV prefix length={length} slot={slot} cached={cached}: 8 full-logit frames exact");
        }
    }
}

fn check_gpu_decode_rungs(library: bool) {
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
    if library {
        let raw = std::fs::read(DevBlob::find_in_dir(&ladder).unwrap().unwrap()).unwrap();
        let roles = segment_role_metadata(&ladder_blob, &raw).unwrap().unwrap();
        assert!(validate_cublaslt_ladder(&ladder_blob, &roles).unwrap());
    } else {
        assert!(validate_decode_ladder(&ladder_blob).unwrap());
    }
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
    let original_waits = library.then(|| ladder_blob.decode_prog().unwrap().waits.clone());
    drop((base_blob, ladder_blob));
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut references: Vec<(String, u32, Vec<u32>)> = Vec::new();
    // Reuse Lt plans: load-time tuning can select different math on separate loads.
    let mut library_engine = library
        .then(|| GpuEngine::load(Arc::clone(&be), &ladder, &ladder.join("checkpoint")).unwrap());
    let mut library_rungs = library_engine
        .as_mut()
        .map(|e| std::mem::take(&mut e.decode_rungs));
    let mut library_control = library_engine.as_mut().map(|e| {
        let bytes = pod_bytes(original_waits.as_ref().unwrap());
        let waits = be.alloc(0, bytes.len().max(4) as u64).unwrap();
        be.upload(&waits, 0, bytes).unwrap();
        let reference = e.capture_library_reference_graph(waits.base).unwrap();
        let elided = e.cublaslt_decode_graph.replace(reference).unwrap();
        (waits, Some(elided))
    });
    for (candidate, assets) in [(false, &baseline), (true, &ladder)] {
        let mut native_engine = (!library)
            .then(|| GpuEngine::load(Arc::clone(&be), assets, &assets.join("checkpoint")).unwrap());
        let mut e = if let Some(e) = library_engine.as_mut() {
            if candidate {
                e.decode_rungs = library_rungs.take().unwrap();
                let elided = library_control.as_mut().unwrap().1.take().unwrap();
                let reference = e.cublaslt_decode_graph.replace(elided).unwrap();
                be.graph_destroy(reference);
            }
            e
        } else {
            native_engine.as_mut().unwrap()
        };
        assert_eq!(e.batch(), batch);
        assert_eq!(!e.cublaslt_decode.is_empty(), library);
        assert!(e.multistep.is_none(), "disable multistep for this gate");
        if candidate {
            assert_eq!(
                e.decode_rungs.iter().map(|r| r.rows).collect::<Vec<_>>(),
                widths[..widths.len() - 1]
            );
            assert!(e
                .decode_rungs
                .iter()
                .all(|r| r.library.is_some() == library));
        } else {
            assert!(e.decode_rungs.is_empty());
        }
        let minimum_rows = e
            .prefill
            .iter()
            .map(|p| p.t)
            .min()
            .expect("prefill required") as usize;
        let rows = std::env::var("TEST_DECODE_RUNG_PROMPT_ROWS")
            .map(|s| s.parse::<usize>().expect("positive prompt row count"))
            .unwrap_or(minimum_rows);
        assert!(rows >= minimum_rows);
        eprintln!("candidate={candidate}: checking {rows}-token prompts");
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
                eprintln!("candidate={candidate} library={library} phase={phase} step={step} slots={slots:?} rung={selected}: full logits exact");
            }
        }
        if candidate {
            for slot in 0..batch - 1 {
                e.retire_slot(slot, true);
                if let Some(vmm) = &e.vmm {
                    assert_eq!(e.pos[slot], 0);
                    assert_eq!(vmm.kv.mapped_rows(slot), 0);
                }
            }
        }
        let slot = batch - 1;
        for step in 0..4 {
            e.step_slots(&[(slot, tokens[slot])], &mut decoded).unwrap();
            tokens[slot] = decoded[0];
            compare(
                &mut e,
                slot,
                tokens[slot],
                format!("retired lower slots step={step}"),
            );
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

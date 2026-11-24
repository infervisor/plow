use super::*;
use crate::asset::devblob::DevTensor;

fn tensor(name: &str, bytes: u64) -> DevTensor {
    DevTensor {
        name: name.into(),
        bytes,
        init: None,
    }
}

#[test]
fn generic_segment_metadata_roundtrips_without_model_ops() {
    use packet::devbuild::{Builder, Model, SectionData, SECT_METADATA};
    let mut b = Builder::new(2);
    b.force_uniseg();
    let a = b.emit(DevOp::Nop, b.all(), &[], |_| {});
    let c = b.emit(DevOp::GemmFp8, b.all(), &[a], |d| {
        d.i[0] = 128;
        d.i[1] = 128;
        d.i[2] = 128;
        d.i[6] = 1;
        d.i[7] = 2;
    });
    b.isolate(c);
    b.emit(DevOp::Nop, b.all(), &[c], |_| {});
    let mut decode = Builder::new(2);
    decode.emit(DevOp::Nop, decode.all(), &[], |_| {});
    let m = Model {
        n_cu: 2,
        target: 0,
        tensors: vec![],
        progs: vec![b.finish(), decode.finish()],
        kv_row_insts: vec![],
        prog_t: vec![128, 1],
        gen: vec![],
    };
    for role in [0, 1] {
        let objects = if role == 1 {
            serde_json::json!({"1":{"abi":"fp8_gemm_tma128_v1","file":"role.cubin"}})
        } else {
            serde_json::json!({})
        };
        let section=SectionData {kind:SECT_METADATA,name:"segment_roles.json".into(),data:serde_json::to_vec(&serde_json::json!({"version":1,"objects":objects,"programs":[{"index":0,"roles":[0,role,0]}]})).unwrap()};
        let raw = m.to_blob_v6(&[section]);
        let blob = DevBlob::parse(&raw).unwrap();
        let metadata = blob
            .section_data_named(&raw, SECT_METADATA, "segment_roles.json")
            .unwrap();
        let roles = SegmentRoles::parse(metadata, &blob).unwrap();
        assert_eq!(roles.program(0).unwrap().roles, [0, role, 0]);
        assert_eq!(
            packet_role_segments(
                &blob.progs[0],
                &roles.program(0).unwrap().roles,
                &blob.tensors
            )
            .unwrap(),
            [0, role, 0]
        );
        assert!(qwen_prefill_segments(&blob.progs[0], &[])
            .unwrap()
            .is_empty());
    }
}

#[test]
fn fp8_role_preserves_distinct_window_conventions() {
    let mut base: DevProgram = unsafe { std::mem::zeroed() };
    base.gq_seg_ofs = 1000;
    base.gq_cursor = 2000;
    let mut arg = base;
    segment_window(&mut arg, &base, 3, false);
    assert_eq!(
        (arg.cur_seg, arg.gq_seg_ofs, arg.gq_cursor),
        (0, 1012, 2000 + 12 * u64::from(CTR_STRIDE))
    );
    segment_window(&mut arg, &base, 3, true);
    assert_eq!(
        (arg.cur_seg, arg.gq_seg_ofs, arg.gq_cursor),
        (3, 1000, 2000)
    );
    assert!(check_fp8_gemm_role(Some(1), Some(256)).is_ok());
    for (cap, block) in [
        (None, Some(256)),
        (Some(0), Some(256)),
        (Some(1), Some(384)),
        (Some(1), None),
    ] {
        assert!(check_fp8_gemm_role(cap, block).is_err());
    }
}

#[test]
fn fp8_role_rejects_mixed_missing_and_duplicate_work() {
    use crate::asset::devblob::DevProg;
    use packet::dev::StreamEnt;
    let mut gemm = DevInst64 {
        op: DevOp::GemmFp8 as u16,
        blocks: 2,
        ..Default::default()
    };
    gemm.i[0] = 1024;
    gemm.i[6] = 1;
    gemm.i[7] = 2;
    let norm = DevInst64 {
        op: DevOp::Nop as u16,
        blocks: 1,
        ..Default::default()
    };
    let stream = vec![
        StreamEnt {
            inst: 0,
            seg: 0,
            ..Default::default()
        },
        StreamEnt {
            inst: 1,
            seg: 1,
            slice: 0,
            ..Default::default()
        },
        StreamEnt {
            inst: 1,
            seg: 1,
            slice: 1,
            ..Default::default()
        },
    ];
    let mut g = DevProg {
        t: 1024,
        role: packet::devbuild::ProgramRole::PrefillBucket { rows: 1024 },
        n_counter: 2,
        insts: vec![norm, gemm],
        stream: stream.clone(),
        stream_ofs: vec![],
        stream_len: vec![],
        waits: vec![],
        succs: vec![],
        gq_stream: stream,
        gq_seg_ofs: vec![0, 1, 3],
        l2_domains: 0,
    };
    assert_eq!(packet_role_segments(&g, &[0, 1], &[]).unwrap(), [0, 1]);
    assert_eq!(packet_role_segments(&g, &[0, 0], &[]).unwrap(), [0, 0]);
    let control =
        serde_json::json!({"version":1,"objects":{},"programs":[{"index":0,"roles":[0,0]}]});
    let candidate = serde_json::json!({"version":1,"objects":{"1":{"abi":"fp8_gemm_tma128_v1","file":"role.cubin"}},"programs":[{"index":0,"roles":[0,1]}]});
    let validate = |v: serde_json::Value| -> bool {
        serde_json::from_value::<SegmentRoles>(v)
            .is_ok_and(|r| r.validate(std::slice::from_ref(&g), &[0], &[]).is_ok())
    };
    assert!(validate(control.clone()));
    assert!(validate(candidate.clone()));
    for path in ["/tmp/role.cubin", "../role.cubin", "a/../role.cubin", ""] {
        let mut bad = candidate.clone();
        bad["objects"]["1"]["file"] = serde_json::json!(path);
        assert!(!validate(bad));
    }
    for roles in [
        serde_json::json!([0]),
        serde_json::json!([0, 2]),
        serde_json::json!([1, 0]),
        serde_json::json!([0, 0]),
    ] {
        let mut bad = candidate.clone();
        bad["programs"][0]["roles"] = roles;
        assert!(!validate(bad));
    }
    let mut bad = candidate.clone();
    bad["programs"] = serde_json::json!([{"index":0,"roles":[0,1]},{"index":0,"roles":[0,1]}]);
    assert!(!validate(bad));
    let mut bad = candidate.clone();
    bad["programs"][0]["index"] = serde_json::json!(1);
    assert!(!validate(bad));
    let mut bad = candidate.clone();
    bad["objects"]["1"]["abi"] = serde_json::json!("unknown");
    assert!(!validate(bad));
    let mut bad = candidate.clone();
    bad["objects"] = serde_json::json!({});
    assert!(!validate(bad));
    let mut bad = control.clone();
    bad["objects"] = serde_json::json!({"2":{"abi":"fp8_gemm_tma128_v1","file":"role.cubin"}});
    assert!(!validate(bad));
    g.gq_stream[2].inst = 0;
    assert!(packet_role_segments(&g, &[0, 1], &[]).is_err());
    g.gq_stream[2].inst = 1;
    g.gq_stream[2].slice = 0;
    assert!(packet_role_segments(&g, &[0, 1], &[]).is_err());
    g.gq_stream[2].slice = 1;
    g.stream.pop();
    assert!(packet_role_segments(&g, &[0, 1], &[]).is_err());
    g.stream = g.gq_stream.clone();
    g.insts[1].i[6] = 0;
    assert!(packet_role_segments(&g, &[0, 1], &[]).is_err());
    g.insts[1].i[6] = 1;
    g.gq_seg_ofs = vec![0, 3];
    assert!(packet_role_segments(&g, &[0, 1], &[]).is_err());
}

#[test]
fn recurrent_state_comes_from_the_section() {
    use plow_asset::recurrent_state::{Buffer, Manifest, VERSION};
    let tensors = [
        tensor("state.qwen.0.conv", 4 * 61440),
        tensor("state.qwen.0.gdn", 4 * 3145728),
        tensor("in.active", 16),
        tensor("act.gdn.initstate", 4 * 3145728),
        tensor("kv.0.k", 4096),
    ];
    let manifest = Manifest {
        version: VERSION,
        slots: 4,
        buffers: vec![
            Buffer {
                tensor: 0,
                slot_bytes: 61440,
            },
            Buffer {
                tensor: 1,
                slot_bytes: 3145728,
            },
        ],
        active: 2,
        prefill_initstate: Some(Buffer {
            tensor: 3,
            slot_bytes: 3145728,
        }),
        prefill_updates_state: true,
        step_seed: false,
        verify: None,
    };
    let state = recurrent_state_from(Some(&manifest), &tensors, 4)
        .unwrap()
        .unwrap();
    assert_eq!(state.active, 2);
    assert_eq!(state.tensors, [(0, 61440), (1, 3145728)]);
    assert_eq!(state.initstate, Some((3, 3145728)));
    assert!(state.prefill_updates_state);
    // Sized for the packet's physical slots, nothing else.
    assert!(recurrent_state_from(Some(&manifest), &tensors, 8).is_err());
    // Every state-class tensor is declared; a partial declaration is refused.
    let mut partial = manifest.clone();
    partial.buffers.remove(0);
    assert!(matches!(
        recurrent_state_from(Some(&partial), &tensors, 4),
        Err(RuntimeError::Rejected(message)) if message.contains("does not declare")
    ));
    // Undeclared state-class tensors or the active mask predate the contract.
    assert!(recurrent_state_from(None, &tensors, 4).is_err());
    assert!(recurrent_state_from(None, &[tensor("in.active", 16)], 4).is_err());
    assert!(recurrent_state_from(None, &[tensor("kv.0.k", 4096)], 4)
        .unwrap()
        .is_none());
    // Verify snapshots are declared state-class tensors; undeclared they are refused.
    let mut snap_tensors = tensors.to_vec();
    snap_tensors.push(tensor("state.qwen.0.conv.snap", 4 * 2 * 61440));
    snap_tensors.push(tensor("state.qwen.0.gdn.snap", 4 * 2 * 3145728));
    assert!(recurrent_state_from(Some(&manifest), &snap_tensors, 4).is_err());
    let mut rollback = manifest.clone();
    rollback.verify = Some(plow_asset::recurrent_state::VerifyRollback {
        depth: 2,
        snapshots: vec![
            plow_asset::recurrent_state::Snapshot {
                buffer: 0,
                tensor: 5,
            },
            plow_asset::recurrent_state::Snapshot {
                buffer: 1,
                tensor: 6,
            },
        ],
        replay: None,
    });
    let state = recurrent_state_from(Some(&rollback), &snap_tensors, 4)
        .unwrap()
        .unwrap();
    let (v, buffers) = state.verify.as_ref().unwrap();
    assert_eq!(v.depth, 2);
    assert_eq!(buffers.len(), 2);
    // The GDN prefill operand must be a declared buffer of the library's stride, section or not.
    let mut scan = DevInst64::default();
    scan.op = DevOp::QwenGdnPrefill as u16;
    scan.t[6] = 1;
    let program = DevProg {
        t: 128,
        role: packet::devbuild::ProgramRole::PrefillBucket { rows: 128 },
        n_counter: 1,
        insts: vec![scan],
        stream: vec![],
        stream_ofs: vec![],
        stream_len: vec![],
        waits: vec![],
        succs: vec![],
        gq_stream: vec![],
        gq_seg_ofs: vec![0, 1],
        l2_domains: 0,
    };
    let declared = recurrent_state_from(Some(&manifest), &tensors, 4)
        .unwrap()
        .unwrap();
    assert!(check_gdn_state_operands(&[&program], Some(&declared)).is_ok());
    assert!(check_gdn_state_operands(&[&program], None).is_err());
    let mut other = program;
    other.insts[0].t[6] = 0;
    assert!(check_gdn_state_operands(&[&other], Some(&declared)).is_err());
    other.insts[0].op = DevOp::Gemm as u16;
    assert!(check_gdn_state_operands(&[&other], None).is_ok());
}

/// The verify-rollback contract end to end on a host model of the span kernels: a span chains
/// its rows through the state, keeps each non-final row's state in the snapshot and the final
/// one in the buffer; after greedy acceptance the commit's copies leave every buffer exactly
/// where plain decode of the committed rows would.
#[test]
fn verify_rollback_commit_matches_plain_decode_of_the_committed_rows() {
    use crate::orch::speculative::accept_greedy;
    use plow_asset::recurrent_state::{Buffer, Snapshot, VerifyRollback};
    const SLOTS: usize = 3;
    const DEPTH: u32 = 3;
    // Two buffers of different strides (a conv history and a recurrent state).
    let buffers = vec![
        Buffer {
            tensor: 0,
            slot_bytes: 6,
        },
        Buffer {
            tensor: 1,
            slot_bytes: 16,
        },
    ];
    let rollback = VerifyRollback {
        depth: DEPTH,
        snapshots: vec![
            Snapshot {
                buffer: 0,
                tensor: 2,
            },
            Snapshot {
                buffer: 1,
                tensor: 3,
            },
        ],
        replay: None,
    };
    let step = |state: &mut [Vec<u8>], token: u32| {
        for (i, s) in state.iter_mut().enumerate() {
            for (j, x) in s.iter_mut().enumerate() {
                *x = x
                    .wrapping_mul(31)
                    .wrapping_add(token as u8)
                    .wrapping_add((i * 7 + j) as u8);
            }
        }
    };
    let argmax = |state: &[Vec<u8>]| u32::from(state[1][0] % 5);
    let slot_state = |mem: &[Vec<u8>], slot: usize| -> Vec<Vec<u8>> {
        buffers
            .iter()
            .map(|b| {
                let n = b.slot_bytes as usize;
                mem[b.tensor as usize][slot * n..(slot + 1) * n].to_vec()
            })
            .collect()
    };
    // (slot, drafts): none = oracle drafts (plain decode's own), else fixed (mostly rejected).
    for (slot, drafts) in [
        (1usize, vec![]),
        (0, vec![9, 9, 9]),
        (2, vec![]),
        (1, vec![4, 4]),
        (2, vec![1]),
    ] {
        // Device memory: buffers 0/1, snapshots 2/3.
        let mut mem: Vec<Vec<u8>> = buffers
            .iter()
            .map(|b| {
                (0..SLOTS * b.slot_bytes as usize)
                    .map(|i| i as u8)
                    .collect()
            })
            .collect();
        for b in &buffers {
            mem.push(vec![0xEE; SLOTS * DEPTH as usize * b.slot_bytes as usize]);
        }
        let before = slot_state(&mem, slot);
        let mut inputs = vec![3u32];
        if drafts.is_empty() {
            let mut s = before.clone();
            for _ in 0..DEPTH {
                step(&mut s, *inputs.last().unwrap());
                inputs.push(argmax(&s));
            }
        } else {
            inputs.extend(&drafts);
        }
        // The span kernels.
        let mut s = before.clone();
        let mut rows = Vec::new();
        for (j, &t) in inputs.iter().enumerate() {
            step(&mut s, t);
            rows.push(argmax(&s));
            for (b, state) in buffers.iter().zip(&s) {
                let n = b.slot_bytes as usize;
                let (tensor, at) = if j + 1 < inputs.len() {
                    let snap = rollback
                        .snapshots
                        .iter()
                        .find(|x| x.buffer == b.tensor)
                        .unwrap()
                        .tensor as usize;
                    (snap, (slot * DEPTH as usize + j) * n)
                } else {
                    (b.tensor as usize, slot * n)
                };
                mem[tensor][at..at + n].copy_from_slice(state);
            }
        }
        let v = accept_greedy(&inputs, &rows, 0, 0, &[]);
        let n = u32::from(v.n_emit);
        for c in rollback
            .commit(&buffers, slot as u32, inputs.len() as u32, n)
            .unwrap()
        {
            let src = mem[c.snapshot as usize][c.src as usize..(c.src + c.bytes) as usize].to_vec();
            mem[c.buffer as usize][c.dst as usize..(c.dst + c.bytes) as usize]
                .copy_from_slice(&src);
        }
        let mut plain = before;
        for &t in &inputs[..n as usize] {
            step(&mut plain, t);
        }
        assert_eq!(
            slot_state(&mem, slot),
            plain,
            "slot {slot} inputs {inputs:?} n {n}"
        );
        if drafts.is_empty() {
            assert_eq!(n as usize, inputs.len(), "oracle drafts are all accepted");
        }
    }
}

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
        packed_prefill_only: false,
        token_batch_body: false,
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
fn qwen_w8a8_decode_and_prefill_require_separate_capabilities() {
    assert!(check_qwen_w8a8_capability(false, 1, Some(1)).is_ok());
    assert!(check_qwen_w8a8_capability(false, 4, Some(1)).is_err());
    for rows in [128, 1024, 4096, 8192] {
        assert!(check_qwen_w8a8_capability(true, rows, Some(1)).is_ok());
        assert!(check_qwen_w8a8_capability(true, rows, None).is_err());
        assert!(check_qwen_w8a8_capability(true, rows, Some(0)).is_err());
    }
    for rows in [1, 256, 16384] {
        assert!(check_qwen_w8a8_capability(true, rows, Some(1)).is_err());
    }
    assert!(check_qwen_w8a8_capability(false, 1, None).is_err());
}

#[test]
fn active_only_block_uses_lifecycle_without_fake_state() {
    let tensors = [tensor("in.active", 4)];
    let block = recurrent_state_layout_with_active(&tensors, 1, true)
        .unwrap()
        .unwrap();
    assert_eq!(block.active, 0);
    assert!(block.tensors.is_empty());
    assert!(recurrent_state_layout(&tensors, 1).unwrap().is_some());
    assert!(recurrent_state_layout_with_active(&[], 1, true).is_err());
    assert!(recurrent_state_layout_with_active(&tensors, 4, true).is_err());
}

#[test]
fn dense_models_need_no_recurrent_mask() {
    assert!(recurrent_state_layout(&[tensor("kv.0.k", 4096)], 1)
        .unwrap()
        .is_none());
}

#[test]
fn state_regions_preserve_physical_slots() {
    let tensors = [
        tensor("state.qwen.0.conv", 4 * 10240 * 3 * 2),
        tensor("state.qwen.0.gdn", 4 * 48 * 128 * 128 * 4),
        tensor("in.active", 16),
    ];
    let state = recurrent_state_layout(&tensors, 4).unwrap().unwrap();
    assert_eq!(state.active, 2);
    assert_eq!(state.tensors, [(0, 10240 * 3 * 2), (1, 48 * 128 * 128 * 4)]);
    for &(index, stride) in &state.tensors {
        for slot in 0..4 {
            assert!((slot + 1) * stride <= tensors[index].bytes);
        }
        assert_eq!(4 * stride, tensors[index].bytes);
    }
}

#[test]
fn invalid_state_or_missing_mask_is_rejected() {
    assert!(recurrent_state_layout(&[tensor("state.qwen.0.gdn", 64)], 4).is_err());
    for (name, bytes) in [
        ("state.other.0.gdn", 64),
        ("state.qwen.0.unknown", 64),
        ("state.qwen.0.gdn", 63),
        ("state.qwen.0.gdn", 0),
    ] {
        assert!(
            recurrent_state_layout(&[tensor(name, bytes), tensor("in.active", 16)], 4).is_err()
        );
    }
    assert!(
        recurrent_state_layout(&[tensor("state.qwen.0.gdn", 64), tensor("in.active", 4)], 4)
            .is_err()
    );
}

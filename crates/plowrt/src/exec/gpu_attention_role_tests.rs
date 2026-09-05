use super::*;
use crate::asset::devblob::{DevProg, DevTensor};
use packet::dev::StreamEnt;

fn fixture(rows: u32, splits: u32) -> (DevProg, Vec<DevTensor>) {
    let work = u64::from(rows) * 24;
    let tensors = [
        work * u64::from(splits) * 256 * 4,
        work * u64::from(splits) * 8,
        work * 256 * 2,
        4 * 65536 * 256 * 2,
        4 * 65536 * 256 * 2,
        work * 256 * 2,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, bytes)| DevTensor {
        name: format!("t{i}"),
        bytes,
        init: None,
    })
    .collect();
    let mut gemm = DevInst64 {
        op: DevOp::GemmFp8 as u16,
        blocks: 2,
        ..Default::default()
    };
    gemm.i[0] = rows;
    gemm.i[6] = 1;
    gemm.i[7] = 2;
    let mut attention = DevInst64 {
        op: DevOp::FlashPrefill as u16,
        blocks: 3,
        t: [TENSOR_NONE16; 8],
        ..Default::default()
    };
    attention.t[..5].copy_from_slice(&[0, 1, 2, 3, 4]);
    attention.i = [rows, rows, 24, 4, 0, 0, 256, splits];
    attention.fj = [0.0625f32.to_bits(), 65536, u32::MAX];
    let mut merge = DevInst64 {
        op: DevOp::FlashMerge as u16,
        blocks: 2,
        t: [TENSOR_NONE16; 8],
        ..Default::default()
    };
    merge.t[..3].copy_from_slice(&[5, 0, 1]);
    merge.i[..4].copy_from_slice(&[rows, 24, splits, 256]);
    let nop = DevInst64 {
        op: DevOp::Nop as u16,
        blocks: 1,
        ..Default::default()
    };
    let insts = vec![nop, gemm, attention, merge, nop];
    let mut stream = Vec::new();
    let mut bounds = vec![0];
    for (ix, inst) in insts.iter().enumerate() {
        stream.extend((0..u32::from(inst.blocks)).map(|slice| StreamEnt {
            inst: ix as u32,
            seg: ix as u16,
            slice,
            ..Default::default()
        }));
        bounds.push(stream.len() as u32);
    }
    (
        DevProg {
            t: rows,
            packed_prefill_only: false,
            n_counter: 0,
            insts,
            stream: stream.clone(),
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            gq_stream: stream,
            gq_seg_ofs: bounds,
            l2_domains: 0,
        },
        tensors,
    )
}

fn metadata() -> serde_json::Value {
    serde_json::json!({"version":1,"objects":{
        "1":{"abi":"fp8_gemm_tma128_v1","file":"gemm.cubin"},
        "2":{"abi":"attention_sm90_hd256_v1","file":"interp_sm90a_pfattn_hd256.cubin"}
    },"programs":[{"index":0,"roles":[0,1,2,2,0]}]})
}

#[test]
fn accepts_combined_roles_and_preserves_packet_split_counts() {
    for (rows, splits) in [(128, 3), (1024, 2), (4096, 1), (8192, 4)] {
        let (g, tensors) = fixture(rows, splits);
        assert_eq!(
            packet_role_segments(&g, &[0, 1, 2, 2, 0], &tensors).unwrap(),
            [0, 1, 2, 2, 0]
        );
        let roles: SegmentRoles = serde_json::from_value(metadata()).unwrap();
        roles
            .validate(std::slice::from_ref(&g), &[0], &tensors)
            .unwrap();
        let mut control = metadata();
        control["objects"] = serde_json::json!({});
        control["programs"][0]["roles"] = serde_json::json!([0, 0, 0, 0, 0]);
        serde_json::from_value::<SegmentRoles>(control)
            .unwrap()
            .validate(std::slice::from_ref(&g), &[0], &tensors)
            .unwrap();
    }
    assert!(check_attention_role("sm90a", Some(1), Some(256)).is_ok());
    for (arch, cap, block) in [
        ("sm120", Some(1), Some(256)),
        ("sm90a", None, Some(256)),
        ("sm90a", Some(0), Some(256)),
        ("sm90a", Some(1), None),
        ("sm90a", Some(1), Some(128)),
    ] {
        assert!(check_attention_role(arch, cap, block).is_err());
    }
}

#[test]
fn rejects_unsupported_attention_operands_and_partial_extents() {
    for case in 0..18 {
        let (mut g, mut tensors) = fixture(1024, 2);
        match case {
            0 => g.insts[2].op = DevOp::FlashPrefillFp8 as u16,
            1 => g.insts[2].i[6] = 512,
            2 => g.insts[2].i[0] = 128,
            3 => g.insts[2].i[2] = 23,
            4 => g.insts[2].i[3] = 0,
            5 => g.insts[2].i[7] = 0,
            6 => g.insts[2].t[5] = 5,
            7 => g.insts[2].t[6] = 5,
            8 => g.insts[2].t[7] = 5,
            9 => g.insts[2].fj[0] = f32::NAN.to_bits(),
            10 => tensors[0].bytes -= 1,
            11 => tensors[1].bytes -= 1,
            12 => g.insts[3].i[3] = 512,
            13 => g.insts[3].i[2] = 0,
            14 => g.insts[3].op = DevOp::QwenGdnPrefill as u16,
            15 => tensors[5].bytes -= 1,
            16 => g.insts[3].t[0] = g.insts[3].t[1],
            17 => g.packed_prefill_only = true,
            _ => unreachable!(),
        }
        assert!(
            packet_role_segments(&g, &[0, 1, 2, 2, 0], &tensors).is_err(),
            "case {case}"
        );
    }
    for index in [2, 3] {
        for slot in 0..if index == 2 { 5 } else { 3 } {
            let (mut g, tensors) = fixture(1024, 2);
            g.insts[index].t[slot] = TENSOR_NONE16;
            assert!(packet_role_segments(&g, &[0, 1, 2, 2, 0], &tensors).is_err());
        }
    }
}

#[test]
fn rejects_mixed_split_missing_and_duplicate_attention_work() {
    for case in 0..6 {
        let (mut g, tensors) = fixture(1024, 2);
        match case {
            0 => g.gq_stream[3].inst = 1,
            1 => {
                g.gq_stream[4].slice = 0;
                g.stream[4].slice = 0;
            }
            2 => {
                g.gq_stream.remove(4);
                g.stream.remove(4);
                for b in &mut g.gq_seg_ofs[3..] {
                    *b -= 1;
                }
            }
            3 => {
                g.gq_stream[5].seg = 3;
                g.stream[5].seg = 3;
                g.gq_seg_ofs[3] -= 1;
            }
            4 => g.stream.pop().map(|_| ()).unwrap(),
            5 => g.l2_domains = 1,
            _ => unreachable!(),
        }
        assert!(
            packet_role_segments(&g, &[0, 1, 2, 2, 0], &tensors).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn rejects_missing_unused_and_mismatched_role_objects() {
    let (g, tensors) = fixture(1024, 2);
    for case in 0..10 {
        let mut value = metadata();
        match case {
            0 => {
                value["objects"].as_object_mut().unwrap().remove("2");
            }
            1 => value["programs"][0]["roles"] = serde_json::json!([0, 1, 0, 0, 0]),
            2 => value["objects"]["2"]["abi"] = "fp8_gemm_tma128_v1".into(),
            3 => value["objects"]["2"]["file"] = "../attention.cubin".into(),
            4 => value["objects"]["2"]["file"] = "/tmp/attention.cubin".into(),
            5 => value["objects"]["3"] = value["objects"]["2"].clone(),
            6 => value["programs"][0]["roles"] = serde_json::json!([0, 1, 3, 3, 0]),
            7 => value["programs"][0]["index"] = 1.into(),
            8 => value["programs"][0]["roles"] = serde_json::json!([0, 1, 2, 2]),
            9 => value["version"] = 2.into(),
            _ => unreachable!(),
        }
        let roles: SegmentRoles = serde_json::from_value(value).unwrap();
        assert!(
            roles
                .validate(std::slice::from_ref(&g), &[0], &tensors)
                .is_err(),
            "case {case}"
        );
    }
}

#[test]
#[ignore = "requires TEST_ATTENTION_ROLE_PACKET pointing to a role2 packet"]
fn actual_packet_attention_roles() {
    let path = std::env::var("TEST_ATTENTION_ROLE_PACKET").expect("packet path");
    let raw = std::fs::read(path).unwrap();
    let blob = DevBlob::parse(&raw).unwrap();
    let metadata = blob
        .section_data_named(&raw, packet::devbuild::SECT_METADATA, "segment_roles.json")
        .expect("packet roles");
    let roles = SegmentRoles::parse(metadata, &blob).unwrap();
    let attention = roles
        .programs
        .iter()
        .flat_map(|p| &p.roles)
        .filter(|&&r| r == 2)
        .count();
    assert!(attention > 0);
    println!(
        "validated {attention} attention segments in {} programs",
        roles.programs.len()
    );
}

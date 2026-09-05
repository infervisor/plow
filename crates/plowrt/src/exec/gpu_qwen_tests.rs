use super::*;
use crate::asset::devblob::{DevProg, DevTensor};
use packet::dev::StreamEnt;

fn fixture() -> (DevProg, Vec<DevTensor>) {
    let rows = 128_u32;
    let mut native = DevInst64::default();
    native.op = DevOp::QwenGdnPrefill as u16;
    native.blocks = 1;
    native.t = [0, 1, 2, 3, 4, 5, 6, 7];
    native.i[..5].copy_from_slice(&[rows, 16, 48, 128, 128]);
    native.fj[0] = (1.0 / 128_f32.sqrt()).to_bits();
    let ordinary = DevInst64 {
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
            ..Default::default()
        },
        StreamEnt {
            inst: 2,
            seg: 2,
            ..Default::default()
        },
    ];
    let sizes = [
        128 * 12288,
        128 * 4096,
        128 * 4096,
        128 * 12288,
        128 * 192,
        128 * 192,
        3145728,
        3145728,
    ];
    let tensors = sizes
        .into_iter()
        .enumerate()
        .map(|(i, bytes)| DevTensor {
            name: format!("operand{i}"),
            bytes,
            init: None,
        })
        .collect();
    (
        DevProg {
            t: rows,
            packed_prefill_only: false,
            n_counter: 0,
            insts: vec![ordinary, native, ordinary],
            stream: stream.clone(),
            stream_ofs: vec![0],
            stream_len: vec![3],
            waits: vec![],
            succs: vec![],
            gq_stream: stream,
            gq_seg_ofs: vec![0, 1, 2, 3],
            l2_domains: 0,
        },
        tensors,
    )
}

#[test]
fn recognizes_one_isolated_external_segment() {
    let (g, t) = fixture();
    let segments = qwen_prefill_segments(&g, &t).unwrap();
    assert_eq!(segments.len(), 3);
    assert!(segments[0].is_none() && segments[2].is_none());
    assert_eq!(segments[1].unwrap().op, DevOp::QwenGdnPrefill as u16);
}

#[test]
fn rejects_mixed_missing_or_split_external_segments() {
    let (mut g, t) = fixture();
    g.stream[0].seg = 1;
    assert!(qwen_prefill_segments(&g, &t).is_err());
    let (mut g, t) = fixture();
    g.stream.retain(|e| e.inst != 1);
    assert!(qwen_prefill_segments(&g, &t).is_err());
    let (mut g, t) = fixture();
    g.stream.push(StreamEnt {
        inst: 1,
        seg: 2,
        ..Default::default()
    });
    assert!(qwen_prefill_segments(&g, &t).is_err());
}

#[test]
fn rejects_incorrect_geometry_and_operand_extent() {
    let (mut g, t) = fixture();
    g.insts[1].i[2] = 16;
    assert!(qwen_prefill_segments(&g, &t).is_err());
    let (mut g, t) = fixture();
    g.insts[1].fj[0] = f32::NAN.to_bits();
    assert!(qwen_prefill_segments(&g, &t).is_err());
    let (g, mut t) = fixture();
    t[6].bytes -= 1;
    assert!(qwen_prefill_segments(&g, &t).is_err());
    let (mut g, t) = fixture();
    g.insts[1].t[7] = u16::MAX;
    assert!(qwen_prefill_segments(&g, &t).is_err());
}

#[test]
fn rejects_unsupported_bucket_topology() {
    let (mut g, t) = fixture();
    g.l2_domains = 2;
    assert!(qwen_prefill_segments(&g, &t).is_err());
    let (mut g, t) = fixture();
    g.t = 8193;
    assert!(qwen_prefill_segments(&g, &t).is_err());
}

#[test]
fn rejects_gq_window_labels_and_external_membership() {
    let (mut g, t) = fixture();
    g.gq_stream.swap(0, 1);
    assert!(qwen_prefill_segments(&g, &t).is_err());
    let (mut g, t) = fixture();
    g.gq_seg_ofs = vec![0, 0, 2, 3];
    g.gq_stream[0].seg = 1;
    assert!(qwen_prefill_segments(&g, &t).is_err());
}

#[test]
fn rejects_invalid_gq_window_bounds() {
    let (mut g, t) = fixture();
    g.gq_seg_ofs = vec![0, 1, 3];
    assert!(qwen_prefill_segments(&g, &t).is_err());
    let (mut g, t) = fixture();
    g.gq_seg_ofs = vec![0, 2, 1, 3];
    assert!(qwen_prefill_segments(&g, &t).is_err());
    let (mut g, t) = fixture();
    g.gq_seg_ofs = vec![0, 1, 2, 4];
    assert!(qwen_prefill_segments(&g, &t).is_err());
}

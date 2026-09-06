use super::*;
use packet::dev::DevOp;
use packet::devbuild::{Builder, Model};

fn fixture() -> Vec<u8> {
    let mut builder = Builder::new(2);
    builder.force_uniseg();
    builder.emit(DevOp::Nop, builder.all(), &[], |_| {});
    let model = Model {
        n_cu: 2,
        target: 0,
        tensors: vec![],
        progs: vec![builder.finish()],
        kv_row_insts: vec![],
        prog_t: vec![8],
        gen: vec![],
    };
    program::with_model(&model, |packet| {
        encode(packet.n_cu, packet.tensors.len(), packet.programs).unwrap()
    })
}

fn set_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn set_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn roundtrip_proves_complete_program_graph() {
    let bytes = fixture();
    let section = parse(&bytes, 2, 0).unwrap();
    assert_eq!(section.n_cu, 2);
    assert_eq!(section.programs.len(), 1);
    let program = &section.programs[0];
    assert_eq!(program.rows, 8);
    assert_eq!(program.insts.len(), 1);
    assert_eq!(program.stream.len(), 2);
    assert_eq!(program.gq_stream, program.stream);
    assert_eq!(program.gq_seg_ofs, [0, 2]);
}

#[test]
fn preflight_rejects_counts_and_truncation_before_decode() {
    let base = fixture();
    for len in 0..base.len() {
        assert!(parse(&base[..len], 2, 0).is_err(), "accepted {len} bytes");
    }
    let mut huge_programs = base.clone();
    set_u32(&mut huge_programs, 12, u32::MAX);
    assert!(parse(&huge_programs, 2, 0).is_err());
    let mut huge_stream = base.clone();
    set_u32(&mut huge_stream, 16 + 5 * 4, u32::MAX);
    assert!(parse(&huge_stream, 2, 0).is_err());
    let mut trailing = base;
    trailing.push(0);
    assert!(parse(&trailing, 2, 0).is_err());
}

#[test]
fn rejects_opcode_tensor_grid_stream_and_counter_mutations() {
    let base = fixture();
    assert!(parse(&base, 3, 0).is_err());

    let inst = SECTION_HEADER_BYTES + PROGRAM_HEADER_BYTES;
    let mut bad_op = base.clone();
    set_u16(&mut bad_op, inst, u16::MAX);
    assert!(parse(&bad_op, 2, 0).is_err());

    let mut bad_tensor = base.clone();
    set_u16(&mut bad_tensor, inst + 16, 0);
    assert!(parse(&bad_tensor, 2, 0).is_err());

    let stream = inst + 64;
    let mut bad_inst_ref = base.clone();
    set_u32(&mut bad_inst_ref, stream, 1);
    assert!(parse(&bad_inst_ref, 2, 0).is_err());

    let mut bad_slice = base.clone();
    set_u32(&mut bad_slice, stream + 4, 2);
    assert!(parse(&bad_slice, 2, 0).is_err());

    let stream_ofs = stream + 2 * 24;
    let mut bad_range = base.clone();
    set_u32(&mut bad_range, stream_ofs + 4, 0);
    assert!(parse(&bad_range, 2, 0).is_err());

    let parsed = parse(&base, 2, 0).unwrap();
    let p = &parsed.programs[0];
    let gq = stream_ofs + 2 * 4 + 2 * 4 + p.waits.len() * 8 + p.succs.len() * 4;
    let mut bad_permutation = base.clone();
    set_u32(&mut bad_permutation, gq + 4, 1);
    assert!(parse(&bad_permutation, 2, 0).is_err());

    let mut bad_counter = base;
    let successors = stream_ofs + 2 * 4 + 2 * 4 + p.waits.len() * 8;
    if !p.succs.is_empty() {
        set_u32(&mut bad_counter, successors, p.n_counter);
        assert!(parse(&bad_counter, 2, 0).is_err());
    }
}

#[test]
fn rejects_queue_window_segment_mismatch() {
    let base = fixture();
    let parsed = parse(&base, 2, 0).unwrap();
    let p = &parsed.programs[0];
    let inst = SECTION_HEADER_BYTES + PROGRAM_HEADER_BYTES;
    let stream = inst + p.insts.len() * 64;
    let stream_ofs = stream + p.stream.len() * 24;
    let gq = stream_ofs + 2 * 4 + 2 * 4 + p.waits.len() * 8 + p.succs.len() * 4;
    let mut bad_segment = base;
    set_u16(&mut bad_segment, gq + 22, 1);
    assert!(parse(&bad_segment, 2, 0).is_err());
}

#[test]
fn work_and_wait_thresholds_may_exceed_resident_grid() {
    let inst = DevInst64 {
        op: DevOp::Nop as u16,
        blocks: 2,
        t: [TENSOR_NONE16; 8],
        ..DevInst64::default()
    };
    let waits = [Wait {
        id: 0,
        threshold: 2,
    }];
    let succs = [0];
    let stream = [
        StreamEnt {
            inst: 0,
            slice: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            wait_len: 1,
            succ_len: 1,
            flags: 0,
            seg: 0,
        },
        StreamEnt {
            slice: 1,
            ..StreamEnt {
                inst: 0,
                slice: 0,
                wait_ofs: 0,
                succ_ofs: 0,
                wait_len: 1,
                succ_len: 1,
                flags: 0,
                seg: 0,
            }
        },
    ];
    let offsets = [0];
    let lengths = [2];
    let windows = [0, 2];
    let input = crate::program::Program {
        rows: 1,
        packed_prefill_only: false,
        n_counter: 1,
        insts: &[inst],
        stream: &stream,
        stream_ofs: &offsets,
        stream_len: &lengths,
        waits: &waits,
        succs: &succs,
        gq_stream: &stream,
        gq_seg_ofs: &windows,
        l2_domains: 0,
    };
    let bytes = encode(1, 0, &[input]).unwrap();
    parse(&bytes, 1, 0).unwrap();
}

#[test]
fn validates_tensor_handles_carried_in_integer_slots() {
    let mut inst = DevInst64 {
        op: DevOp::GemvQkvg as u16,
        blocks: 1,
        t: [0; 8],
        ..DevInst64::default()
    };
    inst.i[6] = 7;
    assert!(integer_tensor_handles_valid(DevOp::GemvQkvg, &inst, 8));
    inst.i[6] = 8;
    assert!(!integer_tensor_handles_valid(DevOp::GemvQkvg, &inst, 8));

    inst.i[4] = 0;
    inst.i[5] = 5;
    inst.i[6] = 6;
    inst.i[7] = TENSOR_NONE_I;
    assert!(integer_tensor_handles_valid(DevOp::GemvQkvFp8, &inst, 8));
    inst.i[4] = 1;
    assert!(!integer_tensor_handles_valid(DevOp::GemvQkvFp8, &inst, 8));
}

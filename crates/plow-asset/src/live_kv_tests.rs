use super::*;
use crate::program::{Packet, Program, Tensor};
use packet::dev::DevInst64;

fn inst(op: DevOp) -> DevInst64 {
    DevInst64 {
        op: op as u16,
        blocks: 1,
        fj: [0; 3],
        t: [TENSOR_NONE16; 8],
        i: [0; 8],
    }
}

fn validate(slot_map: Option<Tensor<'_>>, mutate: impl FnOnce(&mut DevInst64)) -> Result<()> {
    let mut tensors = vec![
        Tensor {
            name: "in.pos",
            bytes: 16,
            initialized: false,
        },
        Tensor {
            name: "in.kvlen",
            bytes: 8,
            initialized: false,
        },
        Tensor {
            name: "kv.k",
            bytes: 4096,
            initialized: false,
        },
        Tensor {
            name: "kv.v",
            bytes: 4096,
            initialized: false,
        },
        Tensor {
            name: "act.q",
            bytes: 1024,
            initialized: false,
        },
        Tensor {
            name: "act.op",
            bytes: 2048,
            initialized: false,
        },
        Tensor {
            name: "act.ml",
            bytes: 16,
            initialized: false,
        },
    ];
    let slot_handle = slot_map.map(|tensor| {
        tensors.push(tensor);
        (tensors.len() - 1) as u16
    });
    let mut decode = inst(DevOp::FlashDecode);
    decode.t[..6].copy_from_slice(&[5, 6, 4, 2, 3, 1]);
    decode.t[6] = slot_handle.unwrap_or(TENSOR_NONE16);
    decode.i = [2, 1, 1, 4, 0, 1, 256, u32::MAX];
    mutate(&mut decode);
    let writer = |cache| {
        let mut d = inst(DevOp::HeadNormRope);
        d.t[0] = cache;
        d.t[5] = 0;
        d.i = [2, 1, 256, 0, 0, 0, 2, 0];
        d.fj[1] = 4;
        d.fj[2] = u32::MAX;
        d
    };
    let insts = [decode, writer(2), writer(3)];
    let program = Program {
        rows: 2,
        packed_prefill_only: false,
        n_counter: 1,
        insts: &insts,
        stream: &[],
        stream_ofs: &[],
        stream_len: &[],
        waits: &[],
        succs: &[],
        gq_stream: &[],
        gq_seg_ofs: &[],
        l2_domains: 0,
    };
    emit(&Packet {
        n_cu: 1,
        tp: false,
        prefill_count: 0,
        tensors: &tensors,
        programs: &[program],
        generated: &[],
        kv_row_insts: &[],
    })
    .map(|_| ())
}

#[test]
fn flash_decode_slot_map_is_optional_and_has_a_runtime_i32_extent() {
    validate(None, |_| {}).unwrap();
    validate(
        Some(Tensor {
            name: "in.decode_slot",
            bytes: 8,
            initialized: false,
        }),
        |_| {},
    )
    .unwrap();
    for bad in [
        Tensor {
            name: "in.decode_slot",
            bytes: 4,
            initialized: false,
        },
        Tensor {
            name: "in.decode_slot",
            bytes: 8,
            initialized: true,
        },
        Tensor {
            name: "act.decode_slot",
            bytes: 8,
            initialized: false,
        },
    ] {
        assert!(validate(Some(bad), |_| {}).is_err());
    }
    assert!(validate(None, |d| d.t[7] = 4).is_err());
}

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

fn fp8_cache(insts: &mut Vec<DevInst64>, tensors: &mut Vec<Tensor<'_>>) {
    for h in [2, 3] {
        tensors[h].bytes /= 2;
    }
    for name in ["kv.k.scale", "kv.v.scale"] {
        tensors.push(Tensor {
            name,
            bytes: 32,
            initialized: false,
        });
    }
    insts[0].op = DevOp::FlashDecodeFp8 as u16;
    insts[0].t[6..8].copy_from_slice(&[7, 8]);
    for (d, scale) in insts[1..].iter_mut().zip([7, 8]) {
        d.op = DevOp::HeadNormRopeFp8 as u16;
        d.t[6] = scale;
    }
}

#[test]
fn fp8_cache_requires_exact_scale_ownership_and_extents() {
    validate_all(None, fp8_cache).unwrap();
    let mutations: &[fn(&mut Vec<DevInst64>, &mut Vec<Tensor<'_>>)] = &[
        |_, ts| ts[7].bytes -= 4,
        |_, ts| ts[8].bytes += 4,
        |_, ts| ts[7].initialized = true,
        |_, ts| ts[2].bytes *= 2,
        |ds, _| ds[0].t[6] = 8,
        |ds, _| ds[0].t[7] = TENSOR_NONE16,
        |ds, _| ds[0].t[6] = 2,
        |ds, _| ds[1].t[6] = 8,
        |ds, _| ds[2].t[6] = 7,
        |ds, _| ds[1].t[6] = TENSOR_NONE16,
        |ds, _| ds[1].op = DevOp::HeadNormRope as u16,
        |ds, _| ds[0].op = DevOp::FlashDecode as u16,
        |ds, _| ds[1].fj[1] *= 2,
        |ds, _| ds[1].t[5] = 7,
        |ds, _| ds[1].t[7] = 8,
        |ds, _| ds[1].t[0] = 7,
    ];
    for (i, mutate) in mutations.iter().enumerate() {
        assert!(
            validate_all(None, |ds, ts| {
                fp8_cache(ds, ts);
                mutate(ds, ts);
            })
            .is_err(),
            "mutation {i}"
        );
    }
    for op in [
        DevOp::Gemm,
        DevOp::GemmMed,
        DevOp::GemmSmall,
        DevOp::GemmWide,
        DevOp::GemmC5,
        DevOp::GemvFp8,
        DevOp::HeadNormRope,
        DevOp::Residual,
    ] {
        for operand in 0..8 {
            assert!(
                validate_all(None, |ds, ts| {
                    fp8_cache(ds, ts);
                    let mut d = inst(op);
                    d.t[operand] = 7;
                    ds.push(d);
                })
                .is_err(),
                "{op:?} operand {operand}"
            );
        }
    }
}

#[test]
fn fp8_direct_operands_cannot_alias_cache() {
    for op in [
        DevOp::GemvFp8,
        DevOp::GemvGluFp8,
        DevOp::QuantFp8,
        DevOp::GemmFp8,
        DevOp::GemmGluFp8,
    ] {
        let mut d = inst(op);
        d.t = [5; 8];
        assert!(validate_all(None, |ops, _| ops.push(d)).is_ok());
        for operand in 0..8 {
            let mut alias = d;
            alias.t[operand] = 2;
            assert!(validate_all(None, |ops, _| ops.push(alias))
                .unwrap_err()
                .contains("unsupported cache operand access"));
            alias.t[operand] = 100;
            assert!(validate_all(None, |ops, _| ops.push(alias))
                .unwrap_err()
                .contains("operand handle out of range"));
        }
    }
}

#[test]
fn fp8_indirect_operands_remain_rejected() {
    for (op, operands) in [
        (DevOp::GemmFp8, &[6, 7][..]),
        (DevOp::GemmGluFp8, &[3, 6, 7][..]),
    ] {
        for &operand in operands {
            for handle in [2, 5, u32::MAX] {
                let mut d = inst(op);
                d.i[operand] = handle;
                assert!(validate_all(None, |ops, _| ops.push(d))
                    .unwrap_err()
                    .contains("tensor-map operands"));
            }
        }
    }
    let mut d = inst(DevOp::GemvGluFp8);
    d.fj[2] = 1;
    d.i[3] = 2;
    assert!(validate_all(None, |ops, _| ops.push(d))
        .unwrap_err()
        .contains("folded operands"));
}

fn validate(slot_map: Option<Tensor<'_>>, mutate: impl FnOnce(&mut DevInst64)) -> Result<()> {
    validate_all(slot_map, |insts, _| mutate(&mut insts[0]))
}

fn validate_all(
    slot_map: Option<Tensor<'_>>,
    mutate: impl FnOnce(&mut Vec<DevInst64>, &mut Vec<Tensor<'_>>),
) -> Result<()> {
    validate_generated(slot_map, |insts, tensors, _| mutate(insts, tensors))
}

fn validate_generated(
    slot_map: Option<Tensor<'_>>,
    mutate: impl FnOnce(&mut Vec<DevInst64>, &mut Vec<Tensor<'_>>, &mut Vec<packet::rope::GenTensor>),
) -> Result<()> {
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
    let writer = |cache| {
        let mut d = inst(DevOp::HeadNormRope);
        d.t[0] = cache;
        d.t[5] = 0;
        d.i = [2, 1, 256, 0, 0, 0, 2, 0];
        d.fj[1] = 4;
        d.fj[2] = u32::MAX;
        d
    };
    let mut insts = vec![decode, writer(2), writer(3)];
    let mut generated = Vec::new();
    mutate(&mut insts, &mut tensors, &mut generated);
    let program = Program {
        rows: 2,
        packed_prefill_only: false,
        token_batch_body: false,
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
        generated: &generated,
        kv_row_insts: &[],
    })
    .map(|_| ())
}

#[test]
fn fp8_gemm_maps_match_sources_and_extents() {
    for op in [DevOp::GemmFp8, DevOp::GemmMedFp8, DevOp::GemmSmallFp8] {
        for mutation in 0..12 {
            let result = validate_generated(None, |insts, tensors, generated| {
                for _ in 0..2 {
                    tensors.push(Tensor {
                        name: "tmap",
                        bytes: 128,
                        initialized: false,
                    });
                }
                for (handle, source, rows) in [(7, 4, 2), (8, 5, 4)] {
                    let mut g = packet::rope::GenTensor::tmap_e4m3(source, rows, 256, 128);
                    g.tensor = handle;
                    generated.push(g);
                }
                let mut d = inst(op);
                d.t[1] = 4;
                d.t[2] = 5;
                d.i = [2, 4, 256, 0, 0, 0, 7, 8];
                match mutation {
                    0 => {}
                    1 => generated[0].aux = 2,
                    2 => generated[0].kind = packet::rope::GEN_TMAP_BF16,
                    3 => generated[0].hd = 128,
                    4 => generated[0].ctx = 1,
                    5 => generated.push(generated[0].clone()),
                    6 => tensors[7].bytes = 64,
                    7 => tensors[4].bytes = 1,
                    8 => d.i[7] = 0,
                    9 => d.i[6] = u32::MAX,
                    10 => d.i[4] = 1,
                    11 => generated[1].aux = 4,
                    _ => unreachable!(),
                }
                insts.push(d);
            });
            assert_eq!(
                result.is_ok(),
                mutation == 0,
                "{op:?} mutation {mutation}: {result:?}"
            );
        }
    }
}

#[test]
fn hd64_half_split_cache_geometry_is_valid() {
    let hd64 = |insts: &mut Vec<DevInst64>, tensors: &mut Vec<Tensor<'_>>| {
        insts[0].i[6] = 64;
        for d in &mut insts[1..] {
            d.i[2] = 64;
        }
        insts[1].i[5] = packet::dev::ROPE_PAIR_HALF;
        tensors[2].bytes /= 4;
        tensors[3].bytes /= 4;
        tensors[4].bytes /= 4;
        tensors[5].bytes /= 4;
    };
    validate_all(None, hd64).unwrap();
    assert!(validate_all(None, |insts, tensors| {
        hd64(insts, tensors);
        insts[1].i[5] = 0;
    })
    .is_err());
}

#[test]
fn flat_mxfp4_moe_operands_are_direct_and_cannot_alias_kv() {
    for op in [
        DevOp::MoeRouterTopkPf,
        DevOp::MoeAlignPf,
        DevOp::MoeCombinePf,
        DevOp::MoeGluMx,
        DevOp::MoeDownMx,
        DevOp::MoeGluMxPf,
        DevOp::MoeDownMxPf,
    ] {
        validate_all(None, |insts, _| insts.push(inst(op))).unwrap();
        assert!(validate_all(None, |insts, _| {
            let mut d = inst(op);
            d.t[0] = 2;
            insts.push(d);
        })
        .is_err());
    }
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

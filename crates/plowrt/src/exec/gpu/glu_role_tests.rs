use super::*;
use crate::asset::devblob::DevTensor;
use plow_asset::segment_roles::{
    legacy_gemm, GemmCapability, BF16_PREFILL_GEMM_GLU_GEMMA4, W8A8_PREFILL_GEMM_GLU_GEMMA4,
};

fn tensors(sizes: &[u64]) -> Vec<DevTensor> {
    sizes
        .iter()
        .map(|&bytes| DevTensor {
            name: "t".into(),
            bytes,
            init: None,
        })
        .collect()
}

fn bf16(rows: u32, n: u32, k: u32) -> (DevInst64, Vec<DevTensor>) {
    let [m, n64, k64] = [rows, n, k].map(u64::from);
    let inst = DevInst64 {
        op: DevOp::GemmGlu as u16,
        i: [rows, n, k, 4, 0, 0, 5, 6],
        t: [0, 1, 2, TENSOR_NONE16, TENSOR_NONE16, 3, TENSOR_NONE16, TENSOR_NONE16],
        ..Default::default()
    };
    let sizes = [m * n64 * 2, m * k64 * 2, n64 * k64 * 2, n64 * k64 * 2, 128, 128, 128];
    (inst, tensors(&sizes))
}

fn w8a8(rows: u32, n: u32, k: u32) -> (DevInst64, Vec<DevTensor>) {
    let [m, n64, k64] = [rows, n, k].map(u64::from);
    let inst = DevInst64 {
        op: DevOp::GemmGluFp8 as u16,
        i: [rows, n, k, 7, 0, 0, 8, 9],
        t: [0, 1, 2, 3, 4, 5, 6, TENSOR_NONE16],
        ..Default::default()
    };
    let sizes = [
        m * n64 * 2,
        m * k64,
        n64 * k64,
        m * 4,
        n64 * 4,
        n64 * k64,
        n64 * 4,
        128,
        128,
        128,
    ];
    (inst, tensors(&sizes))
}

#[test]
fn glu_roles_check_the_descriptor_shape_or_the_legacy_gemma_shape() {
    for role in [BF16_PREFILL_GEMM_GLU_GEMMA4, W8A8_PREFILL_GEMM_GLU_GEMMA4] {
        let legacy = legacy_gemm(role).unwrap();
        let shaped = GemmCapability {
            rows: vec![2048],
            n: 4224,
            k: 2816,
            ..legacy.clone()
        };
        let check = |(inst, tensors): (DevInst64, Vec<DevTensor>), gemm: &GemmCapability| {
            if role == BF16_PREFILL_GEMM_GLU_GEMMA4 {
                validate_gemma4_glu_role_inst(&inst, inst.i[0], &tensors, gemm)
            } else {
                validate_gemma4_w8a8_glu_role_inst(&inst, inst.i[0], &tensors, gemm)
            }
        };
        let build = if role == BF16_PREFILL_GEMM_GLU_GEMMA4 { bf16 } else { w8a8 };
        for rows in [4096, 8192] {
            check(build(rows, 15360, 3840), &legacy).unwrap();
            assert!(check(build(rows, 15360, 3840), &shaped).is_err());
        }
        assert!(check(build(2048, 15360, 3840), &legacy).is_err());
        assert!(check(build(2048, 4224, 2816), &legacy).is_err());
        check(build(2048, 4224, 2816), &shaped).unwrap();
        assert!(check(build(4096, 4224, 2816), &shaped).is_err());
        let (inst, mut short) = build(2048, 4224, 2816);
        short[2].bytes -= 1;
        assert!(check((inst, short), &shaped).is_err());
    }
}

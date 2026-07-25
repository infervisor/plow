//! The runtime packet ABI: each scheduled task lowers to a variable-length
//! `packet::Inst` (kernel opcode + only-needed parameters + counter windows),
//! and the program round-trips through the compact wire format.

use costmodel::{hwspec, DEFAULT_PAGE_BYTES, GemmShape, RowShape, Soc, SramPolicy};
use rewrite::{assemble, LayerPlan, OpKind, OpSpec};
use schedule::packet::{Body, Opcode, Program};
use schedule::{emit_program, schedule, Config};

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}

fn plan() -> LayerPlan {
    LayerPlan {
        ops: vec![
            OpSpec {
                name: "norm".into(),
                inputs: vec!["x".into(), "nw".into()],
                output: "h".into(),
                kind: OpKind::Row(RowShape {
                    rows: 512,
                    feat: 512,
                    operands: 2,
                    reduce: true,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
            OpSpec {
                name: "proj".into(),
                inputs: vec!["h".into(), "w".into()],
                output: "y".into(),
                kind: OpKind::Gemm(GemmShape {
                    m: 512,
                    n: 512,
                    k: 512,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
        ],
    }
}

fn emit() -> Program {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &Config::default());
    emit_program(&g, &cons, &s.tasks, &s.schedule)
}

#[test]
fn lowers_each_op_to_its_body() {
    let prog = emit();

    // The GEMM body carries full problem + tile dims; a DMA only bytes/slot.
    let gemm = prog.insts.iter().find_map(|i| match i.body {
        Body::Gemm {
            m,
            n,
            k,
            bm,
            bn,
            bk,
            ..
        } => Some((m, n, k, bm, bn, bk)),
        _ => None,
    });
    let (m, n, k, bm, bn, bk) = gemm.expect("a GEMM instruction");
    assert!(m > 0 && n > 0 && k > 0 && bm > 0 && bn > 0 && bk > 0);
    assert!(prog
        .insts
        .iter()
        .any(|i| matches!(i.body, Body::Dma { load: true, bytes, .. } if bytes > 0)));
    assert!(prog
        .insts
        .iter()
        .any(|i| matches!(i.body, Body::Row { reduce: true, .. })));

    // Counter windows reference real counters.
    let n_ctr = prog.counters.len() as u32;
    for i in &prog.insts {
        assert!(i.wait.iter().chain(&i.succ).all(|&c| c < n_ctr));
    }
    assert!(prog.counters.iter().all(|c| c.threshold >= 1));
}

#[test]
fn stream_round_trips() {
    let prog = emit();
    let bytes = prog.to_bytes();
    assert_eq!(Program::decode(&bytes).unwrap(), prog);
}

#[test]
fn variable_length_beats_fixed_width() {
    // The compact stream is far smaller than a fixed 96-byte-per-op struct array.
    let prog = emit();
    let bytes = prog.to_bytes().len();
    let fixed = prog.insts.len() * 96;
    assert!(
        bytes < fixed,
        "variable stream {bytes} not smaller than fixed {fixed}"
    );
}

#[test]
fn opcode_is_u16_structured() {
    let prog = emit();
    // All GEMM instructions carry family=GEMM with variant=BF16 (default dtype)
    let expected = Opcode::new(0, Opcode::FAMILY_GEMM, Opcode::VARIANT_BF16);
    for i in &prog.insts {
        if matches!(i.body, Body::Gemm { .. }) {
            assert_eq!(i.body.opcode(), expected);
            assert_eq!(i.body.opcode().family(), Opcode::FAMILY_GEMM);
        }
    }
}

#[test]
fn stream_header_carries_metadata() {
    let prog = emit();
    // Default emit sets bucket_id=0, plan_gen=0
    assert_eq!(prog.bucket_id, 0);
    assert_eq!(prog.plan_gen, 0);

    // Round-trip preserves metadata
    let mut p2 = prog.clone();
    p2.bucket_id = 123;
    p2.plan_gen = 42;
    let decoded = Program::decode(&p2.to_bytes()).unwrap();
    assert_eq!(decoded.bucket_id, 123);
    assert_eq!(decoded.plan_gen, 42);
}

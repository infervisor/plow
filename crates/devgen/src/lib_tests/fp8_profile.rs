//! The `PLOW_FP8=1` profile emits w8a16 — and gfx950 has no w8a16 prefill GEMM, only w8a8.
//! `d_gemm_fp8` there reads t[1] as e4m3 bytes and dereferences `ascale[m]` with no null check,
//! so a w8a16 packet faults on its first prefill GEMM with no diagnostic. These pin the gate
//! that refuses it, and pin that the profile which DOES work is left alone.
use super::*;
use packet::dev::DevInst;
use packet::devbuild::Program;

fn prog(insts: Vec<DevInst>) -> Program {
    Program {
        hier_base: 0,
        n_cu: 4,
        n_counter: 0,
        insts,
        stream: vec![],
        stream_ofs: vec![],
        stream_len: vec![],
        waits: vec![],
        succs: vec![],
        tensors: vec![],
        gq_stream: vec![],
        gq_seg_ofs: vec![],
        l2_sms: 0,
        l2_domains: 0,
    }
}

/// `t[3]` is a_scale. `TENSOR_NONE` there is w8a16; a bound handle is w8a8.
fn model(a_scale: u32) -> Model {
    let mut i = DevInst {
        op: DevOp::GemmFp8 as u16,
        blocks: 1,
        ..Default::default()
    };
    i.t[3] = a_scale;
    Model {
        n_cu: 256,
        target: 0,
        tensors: vec![],
        progs: vec![prog(vec![i])],
        kv_row_insts: vec![],
        prog_t: vec![128],
        gen: vec![],
    }
}

#[test]
#[should_panic(expected = "fp8_w8a16_prefill")]
fn w8a16_fp8_prefill_is_refused_on_gfx950() {
    check_fp8_a_scale_bound(&model(TENSOR_NONE), "gfx950", "");
}

/// w8a8 binds t[3], which is the profile that actually runs on gfx950.
#[test]
fn w8a8_fp8_prefill_passes_on_gfx950() {
    check_fp8_a_scale_bound(&model(7), "gfx950", "");
}

/// sm_120 HAS a w8a16 cubin, so the same packet is valid there and must not be refused —
/// the gate is about one target's kernel, not about w8a16 being wrong.
#[test]
fn w8a16_fp8_prefill_is_fine_on_sm120() {
    check_fp8_a_scale_bound(&model(TENSOR_NONE), "sm_120a", "RTX5090");
}

/// The trap-asset case: `--arch sm_120a` (where w8a16 is legitimate) with an AMD `--gpu`. An
/// arch-only gate would pass this, and `build-amd/g31b-fp8kv` is exactly it — emitted for
/// sm_120a, sized for 256 CUs, run on gfx950, faulted. Either signal saying AMD is enough.
#[test]
#[should_panic(expected = "fp8_w8a16_prefill")]
fn w8a16_is_refused_when_only_the_gpu_says_amd() {
    check_fp8_a_scale_bound(&model(TENSOR_NONE), "sm_120a", "MI350X");
}

/// …and the target predicate behind it agrees with both signals independently.
#[test]
fn target_is_amd_reads_either_signal() {
    assert!(target_is_amd("gfx950", ""));
    assert!(target_is_amd("", "MI350X"));
    assert!(
        target_is_amd("sm_120a", "MI350X"),
        "the gpu is enough on its own"
    );
    assert!(!target_is_amd("sm_120a", "RTX5090"));
    assert!(
        !target_is_amd("", ""),
        "no target => unchanged emission (golden tests)"
    );
}

/// EVERY fp8 rung is gated, not just the three the ladder started with.
///
/// The regression this pins: the tile-inventory campaign grew `GFX950_RUNGS` from 3 rungs to
/// 5, so `pick_tile` could return `GemmWideFp8` (128x256) and `GemmC5Fp8` (192x256) — and the
/// gate's hand-written opcode list still named only `Gemm/GemmMed/GemmSmall`. A w8a16 packet
/// whose shape resolved to either new rung compiled clean and null-dereferenced `ascale[m]` on
/// device, which is precisely what this gate exists to stop.
///
/// Written as a loop over the table rather than five literal cases on purpose: a sixth rung
/// is covered the moment it is added, with no second place to remember to update. That is the
/// same argument `GFX950_RUNGS`'s own doc comment makes for there being one table.
#[test]
fn every_fp8_rung_is_refused_when_a_scale_is_unbound() {
    for (_, fp8, _, bm, bn, _) in GFX950_RUNGS {
        let mut i = DevInst {
            op: fp8 as u16,
            blocks: 1,
            ..Default::default()
        };
        i.t[3] = TENSOR_NONE;
        let m = Model {
            n_cu: 256,
            target: 0,
            tensors: vec![],
            progs: vec![prog(vec![i])],
            kv_row_insts: vec![],
            prog_t: vec![128],
            gen: vec![],
        };
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_fp8_a_scale_bound(&m, "gfx950", "")
        }));
        assert!(
            caught.is_err(),
            "the {bm}x{bn} fp8 rung ({fp8:?}) is emittable by pick_tile but not gated: a \
                 w8a16 packet on that shape would reach d_gemm_fp8's epilogue and dereference a \
                 null ascale on device"
        );
    }
}

/// A bf16 packet has no fp8 GEMM at all and must sail through on every target.
#[test]
fn bf16_packets_are_untouched() {
    let i = DevInst {
        op: DevOp::Gemm as u16,
        blocks: 1,
        ..Default::default()
    };
    let m = Model {
        n_cu: 256,
        target: 0,
        tensors: vec![],
        progs: vec![prog(vec![i])],
        kv_row_insts: vec![],
        prog_t: vec![128],
        gen: vec![],
    };
    check_fp8_a_scale_bound(&m, "gfx950", "");
}

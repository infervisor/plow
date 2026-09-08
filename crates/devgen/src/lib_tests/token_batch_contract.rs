//! The dense-GQA emit preconditions the unified token-batch route depends on.
//!
//! `plans/unified-token-batch.md` §8 Phase 2 / `docs/arch/17-unified-token-batch-dense-gqa.md`.
//! The AMD route schedules `FlashPrefill` as one flat work list over every span's query tiles.
//! That schedule is bit-identical to an isolated per-request run only while a span boundary
//! cannot move the KV partition, which holds exactly at `nsplit == 1`; `runtime/amd/interp.hip`
//! TRAPS a packet with `i7 != 1` rather than silently changing the arithmetic under packing.
//!
//! So the emitter's "packing pins `ns = 1`" is not a nicety, it is the route's precondition, and
//! a change that lost it would turn every packed prefill into a hard stop on a device. These
//! tests are where that gets caught.

use super::*;

/// Packing pins an unsplit flash with a fused epilogue, for every geometry — not just the one
/// the ladder happens to emit today.
#[test]
fn packing_pins_an_unsplit_fused_flash() {
    for &n_cu in &[64u32, 128, 256, 304] {
        for &heads in &[4u32, 8, 16, 32, 64] {
            for &t in &[1u32, 128, 512, 2048, 8192] {
                let (ns, fused) = dense_flash_split(false, true, n_cu, heads, t);
                assert_eq!(ns, 1, "n_cu={n_cu} heads={heads} t={t}");
                assert!(fused, "n_cu={n_cu} heads={heads} t={t}");
            }
        }
    }
}

/// Without packing the split is free to grow, and the fused epilogue disappears with it. This is
/// the negative control: without it the test above would pass on an emitter that had stopped
/// splitting entirely, and would prove nothing about packing.
#[test]
fn without_packing_a_wide_machine_still_splits() {
    let (ns, fused) = dense_flash_split(false, false, 304, 8, 512);
    assert!(ns > 1, "expected a split at n_cu=304 heads=8 t=512, got {ns}");
    assert!(!fused);
}

/// The GEMV (decode) family keeps its own split and never takes the fused epilogue, packed or
/// not — decode partials always go through `FlashMerge`.
#[test]
fn the_gemv_family_is_unaffected_by_packing() {
    for packed in [false, true] {
        let (ns, fused) = dense_flash_split(true, packed, 256, 8, 1);
        assert_eq!(ns, 32);
        assert!(!fused);
    }
}

/// `nsplit` is never zero: it is a divisor of the work list and a `0` would emit a packet no
/// kernel can execute.
#[test]
fn nsplit_is_never_zero() {
    for &gemv in &[false, true] {
        for &packed in &[false, true] {
            for &heads in &[1u32, 1024] {
                for &n_cu in &[1u32, 304] {
                    let (ns, _) = dense_flash_split(gemv, packed, n_cu, heads, 1);
                    assert!(ns >= 1, "gemv={gemv} packed={packed} heads={heads} n_cu={n_cu}");
                }
            }
        }
    }
}

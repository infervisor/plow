//! `GemmGluMxfp4`'s emit gate, against the hardware it is a proxy for.
//!
//! The fused mxfp4 gate|up arm exists at **256x256 only** — the GLU epilogue's wave->column remap
//! needs `SN == 2`, i.e. `BN = 256` — so it can LOSE to an unfused pair at a narrower rung even
//! though it moves 8 fewer bytes per output element. Whether it wins is a TILE question, not a
//! fusion question, and `glu_fusion_wins_mxfp4` is the answer to it.
//!
//! Measured on gfx950 / 256 CU at the Kimi-K3 shared-expert prefill shape (K = hidden = 7168,
//! N = `num_shared_experts * moe_intermediate_size / TP` = 6144/TP, which is 768 at the reference
//! TP8), fused against the BEST of the five unfused fp4 rungs —
//! `runtime/tests/gemm_glu_mxfp4_gfx950_test.hip`:
//!
//! ```text
//!   T     N=6144 (TP1)   N=1536 (TP4)   N=768 (TP8)
//!    128   +5.9% lose    +35.5% lose    +23.2% lose
//!    512   -4.7% win      +9.5% lose    +10.2% lose
//!   2048   +5.3% lose     -4.7% win      +8.9% lose
//!   4096  -20.6% win     -34.7% win      -4.8% win
//!   8192  -32.1% win      +4.9% lose    -35.2% win
//! ```
//!
//! Against the SAME tile unfused the fused arm wins 39-49% at every one of these shapes; the table
//! is the fusion NET of the tile it costs. The +/-5% band is where the two are indistinguishable
//! run to run (one shape in an earlier pass measured 3x its own median), so only the shapes outside
//! it are pinned here.
//!
//! # A residual this file records rather than hides
//!
//! At **N = 768** — K3's own TP8 width — the gate refuses at every T, including T = 8192 where the
//! fused arm measures **-35.2%**. The gate asks the tuning store whether 256x256 is the winning fp4
//! rung at `(T, 2N, K)`, and the store's record for `8192 x 1536 x 7168 Mxfp4` ranks 128x128
//! (0.634 ms) ahead of 256x256 (0.852 ms). The harness above measures the opposite at that shape by
//! a factor of 1.6 (256x256 0.774 ms for the pair against 128x128's 1.260 ms). Two harnesses, one
//! kernel, one shape, opposite answers — that disagreement is about the tile CALIBRATION and not
//! about the fusion, so it is left standing and the gate stays on the store's side of it. The cost
//! is a win not taken; the alternative is a compiler that ignores its own measurements.
//!
//! # Why this is an integration test and not a unit test
//!
//! `devgen`'s `#[cfg(test)]` build swaps `gfx950_gemm_inventory` for a fixed FIXTURE, so a unit
//! test never reaches the tuning store and always answers from the analytical model. Production
//! reads the store. Both tiers must satisfy the property below, and only this file can check the
//! one that ships.

use devgen::glu_fusion_wins_mxfp4;

const K: u32 = 7168; // Kimi-K3 hidden
const NCU: u32 = 256;

/// Shapes where the unfused pair measured clearly (>8%) faster. Fusing here would BUY the round
/// trip and PAY a tile that does not fill the machine.
#[test]
fn the_gate_never_fuses_where_a_narrow_rung_clearly_wins() {
    for (t, n) in [
        (128u32, 1536u32),
        (128, 768),
        (512, 1536),
        (512, 768),
        (2048, 768),
    ] {
        assert!(
            !glu_fusion_wins_mxfp4(t, n, K, NCU),
            "T={t} N={n} K={K}: the gate fused, but the fused 256x256 arm measured slower than the \
             unfused pair at the rung this shape actually wants. The GLU epilogue needs SN == 2 \
             (BN = 256), so it cannot follow the shape down to 128x128 or 64x128."
        );
    }
}

/// ...and the large wins it must not give up.
#[test]
fn the_gate_takes_the_large_wins_at_machine_filling_shapes() {
    for (t, n) in [(4096u32, 6144u32), (8192, 6144), (4096, 1536)] {
        assert!(
            glu_fusion_wins_mxfp4(t, n, K, NCU),
            "T={t} N={n} K={K}: the gate refused, giving up a measured 20-35%. This shape fills the \
             machine at 256x256 and the unfused pair pays two HBM round trips of [T, N] for nothing."
        );
    }
}

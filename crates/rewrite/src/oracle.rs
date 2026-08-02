//! What the compiler consults about kernels it did not enumerate itself.
//!
//! `tilegraph::assemble` enumerates tiles from the analytical cost model and
//! asks egglog for the argmin. Two facts that model cannot know:
//!
//! * **which tiles the target can actually execute.** The cost model
//!   synthesizes legal-looking shapes; the interpreter being compiled for
//!   contains a specific set. On NVIDIA the tile is not even selectable — one
//!   `d_gemm` with a compile-time macro tile serves every GEMM opcode — so the
//!   right question is not "may I use this synthesized tile" but "what tiles
//!   exist". Filtering synthesized candidates and hoping one survives is how a
//!   plan comes to name a tile the runtime does not build.
//! * **what a tile actually costs.** The model is analytical. Where a
//!   measurement exists for this hardware, this build, and this dtype, it wins.
//!
//! Both are answered against a [`GemmQuery`] — shape **and** dtype/quant — so a
//! bf16 answer can never be applied to an fp8 op. Both are supplied by an
//! oracle rather than wired in, so `rewrite` keeps its dependencies.
//! [`NoOracle`] is the analytical-only default and is exactly the compiler
//! before an oracle existed.

use costmodel::tile::{GemmShape, TileShape};

/// A GEMM the compiler wants tiles for: its shape and the operand formats that
/// decide *which kernel variant* is emitted.
///
/// Dtype is part of the key because it is part of the kernel identity: the
/// emitter selects a different variant for bf16, fp8, block-quant, and native
/// fp4 (`schedule::emit::gemm_variant_for`). An oracle that ignored dtype would
/// let a bf16 answer stand in for an fp8 op — admitting a tile that exists only
/// in a different build, or pricing it with a measurement of different math.
#[derive(Clone, Copy, Debug)]
pub struct GemmQuery {
    pub shape: GemmShape,
    /// Bytes per weight element (1 = fp8 / 4-bit amortized, 2 = bf16).
    pub weight_elem: u64,
    /// Bytes per activation element.
    pub activation_elem: u64,
    /// Weight is a block-quantized format (selects the W4A8 dequant kernel).
    pub block_quant: bool,
    /// Weight is native MX fp4 (selects the FP4 tensor-core kernel).
    pub native_fp4: bool,
}

impl GemmQuery {
    /// Plain bf16 — the common case and what tests use.
    pub fn bf16(shape: GemmShape) -> Self {
        GemmQuery {
            shape,
            weight_elem: 2,
            activation_elem: 2,
            block_quant: false,
            native_fp4: false,
        }
    }

    /// A short label for the operand format, for reports and measurement keys.
    /// Mirrors the cases in `gemm_variant_for`.
    pub fn variant(self) -> &'static str {
        if self.block_quant {
            "w4a8"
        } else if self.native_fp4 {
            "fp4"
        } else {
            match (self.weight_elem, self.activation_elem) {
                (2, 2) => "bf16",
                (1, 2) => "fp8",
                (1, 1) => "fp8fp8",
                _ => "bf16",
            }
        }
    }
}

/// Equality over shape dims and dtype, since `GemmShape` itself is not `Eq`.
impl PartialEq for GemmQuery {
    fn eq(&self, o: &Self) -> bool {
        self.shape.m == o.shape.m
            && self.shape.n == o.shape.n
            && self.shape.k == o.shape.k
            && self.weight_elem == o.weight_elem
            && self.activation_elem == o.activation_elem
            && self.block_quant == o.block_quant
            && self.native_fp4 == o.native_fp4
    }
}

/// What the oracle knows about the tiles for one GEMM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileAdvice {
    /// No opinion — no inventory was probed. Use the analytical candidates.
    Analytical,
    /// The exact tiles the target can execute for this op. These are
    /// authoritative: the compiler chooses among *these*, not among synthesized
    /// shapes. An empty vector is not produced here — see [`Self::Unverified`].
    Buildable(Vec<TileShape>),
    /// An inventory was probed, but it does not cover this op's dtype (e.g. a
    /// bf16-prefill object asked about fp8). The compiler falls back to the
    /// analytical model for this op, and the build report records that the
    /// choice was *not* capability-verified — which is the honest state, not a
    /// silent pass.
    Unverified,
}

/// Answers about kernels the analytical model cannot give.
///
/// The default methods make [`NoOracle`] the analytical-only compiler, so the
/// trait is additive: threading an oracle through changes nothing until a real
/// one is supplied.
pub trait KernelOracle {
    /// Which tiles the target can execute for this GEMM.
    fn gemm_tiles(&self, _q: &GemmQuery) -> TileAdvice {
        TileAdvice::Analytical
    }

    /// Measured costs for the whole candidate list, or `None`.
    ///
    /// All-or-nothing by construction. Measured costs are nanoseconds and
    /// analytical costs are cycles; substituting only the candidates that
    /// happen to have records compares two scales and reliably prefers
    /// whichever is numerically smaller, which is not a decision. Returning a
    /// full vector or nothing makes that mistake unrepresentable. The vector
    /// must match `tiles` in length and order.
    fn measured_gemm(&self, _q: &GemmQuery, _tiles: &[TileShape]) -> Option<Vec<u64>> {
        None
    }

    /// Calibration tier the answers rest on, recorded in the build report.
    fn tier(&self) -> &'static str {
        "portable"
    }

    /// Human-readable note about the oracle's provenance, for the report.
    fn provenance(&self) -> String {
        "analytical cost model only".to_string()
    }

    /// Called when the pinned weight layout excluded every buildable tile — a
    /// genuine conflict between the shared `(BN, BK)` and the kernels that
    /// exist. Reported rather than crashed; the caller falls back to analytical
    /// for that op.
    fn note_pin_conflict(&self, _q: &GemmQuery, _buildable: &[TileShape]) {}

    /// Called when an op's dtype is not covered by the probed inventory.
    fn note_unverified(&self, _q: &GemmQuery) {}
}

/// The analytical-only oracle: no inventory, no measurements. Behaviourally
/// identical to the compiler before an oracle existed, which is what makes the
/// plumbing safe to land ahead of the data.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoOracle;

impl KernelOracle for NoOracle {}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(bm: i64, bn: i64, bk: i64) -> TileShape {
        TileShape {
            bm,
            bn,
            bk,
            split_k: 1,
        }
    }
    fn shape() -> GemmShape {
        GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        }
    }

    #[test]
    fn the_default_oracle_has_no_opinion() {
        let o = NoOracle;
        assert_eq!(
            o.gemm_tiles(&GemmQuery::bf16(shape())),
            TileAdvice::Analytical
        );
        assert!(o
            .measured_gemm(&GemmQuery::bf16(shape()), &[tile(128, 128, 32)])
            .is_none());
        assert_eq!(o.tier(), "portable");
    }

    #[test]
    fn variant_labels_track_the_emitter() {
        assert_eq!(GemmQuery::bf16(shape()).variant(), "bf16");
        let fp8 = GemmQuery {
            activation_elem: 2,
            weight_elem: 1,
            ..GemmQuery::bf16(shape())
        };
        assert_eq!(fp8.variant(), "fp8");
        let bq = GemmQuery {
            block_quant: true,
            ..GemmQuery::bf16(shape())
        };
        assert_eq!(bq.variant(), "w4a8");
        let mx = GemmQuery {
            native_fp4: true,
            ..GemmQuery::bf16(shape())
        };
        assert_eq!(mx.variant(), "fp4");
    }

    /// dtype participates in the key, so two GEMMs of the same shape but
    /// different operand formats are different queries.
    #[test]
    fn dtype_distinguishes_queries() {
        let bf16 = GemmQuery::bf16(shape());
        let fp8 = GemmQuery {
            weight_elem: 1,
            ..bf16
        };
        assert_ne!(bf16, fp8);
    }
}

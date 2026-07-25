//! Calibrated measurement store for kernel selection.
//!
//! `perf-data/` already holds ~200 files of measurements, but they answer a
//! different question: `all-perf-data.json` is a *serving-level* index keyed by
//! `(model, engine, precision, phase, tp, ctx)`, with no column for GPU, shape,
//! tile, or kernel, and one hardware string per file. It is the right shape for
//! "is plow faster than vLLM on this model" and the wrong shape for "which
//! kernel should this GEMM use on this card". This crate is the level beneath
//! it, and does not replace it.
//!
//! What the store guarantees:
//!
//! - **No single-sample records.** [`Stats`] requires enough samples to carry
//!   dispersion, and [`Stats::beats`] refuses a win that sits inside the noise.
//! - **Correct before fast.** A measurement cannot be published until its
//!   oracle has passed; there is no path that promotes an unchecked kernel.
//! - **Atomic campaigns.** One unqualifiable record aborts the whole
//!   publication, so an interrupted run cannot leave a selectable half-winner.
//! - **Specific staleness.** Digests are per-input, so a recompiled kernel
//!   invalidates its own records and nothing else.
//! - **Retained negatives.** Rejections keep their reason, so a campaign does
//!   not spend GPU time rediscovering the same dead end.
//!
//! Only an explicit tuning run writes here. A compile may read qualified
//! records; it must never publish, or the thing being measured and the thing
//! doing the measuring stop being separable.

pub mod decode;
pub mod record;
pub mod sample;
pub mod store;

pub use decode::{
    rank_by_cell, CellRanking, CtxBucket, DecodeCell, DecodeKnobs, DecodeMeasurement,
};
pub use record::{
    blockers_for, BlockDefinition, BlockMeasurement, Correctness, Digests, KernelMeasurement,
    OpCase, RecordState, Selection,
};
pub use sample::{SampleError, Stats};
pub use store::{StaleNote, StoreError, TuneStore};

/// Exit code `gpulease` uses for "the run completed but the GPU was contended".
///
/// `perf-data/harness/README.md` is explicit that a contended run silently
/// invalidates timings, so this is treated as a failed measurement rather than
/// a result with a caveat.
pub const GPULEASE_CONTENDED: i32 = 76;

/// Whether a harness exit status means the timings may be kept.
pub fn measurement_is_trustworthy(exit_code: i32) -> bool {
    exit_code == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_contended_run_is_not_a_result() {
        assert!(measurement_is_trustworthy(0));
        assert!(!measurement_is_trustworthy(GPULEASE_CONTENDED));
        assert!(!measurement_is_trustworthy(1));
    }
}

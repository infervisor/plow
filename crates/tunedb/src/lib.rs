//! Calibrated measurement store for kernel selection.
//!
//! `perf-data/` already holds ~200 files of measurements, but they answer a
//! different question: those campaign files are *serving-level*, keyed by
//! `(model, engine, precision, phase, tp, ctx)`, with no column for GPU, shape,
//! tile, or kernel, and one hardware string per file. They are the right shape
//! for "is plow faster than vLLM on this model" and the wrong shape for "which
//! kernel should this GEMM use on this card". This crate is the level beneath
//! them, and does not replace them.
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

pub mod attention;
pub mod decode;
pub mod gemm;
pub mod gemv;
pub mod moe_decode;
pub mod object;
pub mod record;
pub mod sample;
pub mod store;

pub use gemm::{
    amd_tuning_cell, gemm_op_case, gemm_rung_emit_plan, gemm_rung_opcode, GemmEmitPlan,
    GEMM_ORACLE, GEMM_WIDE_C8_MEASUREMENT_ID, GFX950_CELL,
};
pub use gemv::{gemv_case, gemv_op_case, gemv_sample_bucket, gemv_sample_opcode, GEMV_ORACLE};

pub use attention::{
    select_attention, AttentionAlgorithm, AttentionCapabilities, AttentionCell,
    AttentionMeasurement, AttentionSelection, AttentionSource, KvBucket, ATTENTION_ORACLE,
};
pub use decode::{
    rank_by_cell, CellRanking, CtxBucket, DecodeCell, DecodeKnobs, DecodeMeasurement,
};
pub use moe_decode::{
    select_moe_decode_route, MoeDecodeCell, MoeDecodeMeasurement, MoeDecodeRoute,
    MoeDecodeSelection, MoeDecodeSource, GFX950_SEGMENT_HANDOFF_NS, MIN_GAIN_FRACTION,
    MOE_DECODE_ORACLE,
};
pub use object::{
    rank_by_cell as rank_objects_by_cell, ObjectCell, ObjectConfig, ObjectMeasurement,
    ObjectRanking, WindowClass,
};
pub use record::{
    blockers_for, BlockDefinition, BlockMeasurement, Correctness, Digests, KernelMeasurement,
    OpCase, RecordState, Selection,
};
pub use sample::{SampleError, Stats};
pub use store::{StaleNote, StoreError, TuneStore};

/// Exit code `gpulease` uses for "the run completed but the GPU was contended".
///
/// `perf-data/tools/README.md` is explicit that a contended run silently
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

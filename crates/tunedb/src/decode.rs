//! Decode-object knob records.
//!
//! [`KernelMeasurement`](crate::KernelMeasurement) answers "which opcode should
//! this shape dispatch to". That is the wrong question for the knobs the
//! H100/26B campaign moved (`perf-data/gemma26b-h100-gemv-mlp.md`): every one of
//! them — unroll depth, MoE stream count, lane-split width, blocks/SM, flash
//! split — is a **compile-time define on the whole decode object**, or a value
//! baked into the packet at emit time. There is no opcode to choose between;
//! there is one object, and the question is which defines it was built with.
//!
//! Three properties of that search, all measured rather than assumed, shape
//! this record:
//!
//! - **The optimum is not a constant.** Unroll depth *inverts* with occupancy —
//!   deep wins at 1 block/SM and spills at 2 — so a record that does not carry
//!   the occupancy it was measured at is not reusable.
//! - **The optimum moves with context.** Decode TPOT is not flat in ctx (the
//!   campaign measured 6.196 / 6.935 / 9.209 ms at 1k / 8k / 32k) and the knobs
//!   governing the growth are exactly the ctx-dependent ones. So `ctx_bucket` is
//!   part of the key, and the buckets are geometric because 1k→8k is nearly flat
//!   while 8k→32k is not.
//! - **`(FORCE_MINBLK, --n-cu)` is one knob, not two.** The engine refuses a
//!   packet emitted for 132 blocks against an object that reaches 264, so the
//!   pair is carried together and `n_cu` is in the cell key.
//!
//! The value is the whole define-set, not a winner's name, because reproducing
//! the measurement means rebuilding the object: [`DecodeKnobs::defines`] emits
//! the exact `nvcc` flags and [`DecodeKnobs::emit_env`] the exact `plowc`
//! invocation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::record::{blockers_for, Correctness, Digests, RecordState};
use crate::sample::Stats;

/// Geometric context buckets.
///
/// Deliberately not linear. The campaign's ctx curve is flat from 1k to 8k and
/// steep from 8k to 32k, because the growth is the 5 full-attention layers
/// reading the whole context; linear buckets would spend most of their
/// resolution where nothing changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CtxBucket {
    #[serde(rename = "1k")]
    K1,
    #[serde(rename = "8k")]
    K8,
    #[serde(rename = "32k")]
    K32,
    #[serde(rename = "128k")]
    K128,
}

impl CtxBucket {
    /// The bucket a measured context falls in — the nearest bucket at or below
    /// it, so an unmeasured 4k is served by the 1k record rather than by the 8k
    /// one it was never compared against.
    pub fn of(ctx: u32) -> CtxBucket {
        match ctx {
            0..=4095 => CtxBucket::K1,
            4096..=16383 => CtxBucket::K8,
            16384..=65535 => CtxBucket::K32,
            _ => CtxBucket::K128,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            CtxBucket::K1 => "1k",
            CtxBucket::K8 => "8k",
            CtxBucket::K32 => "32k",
            CtxBucket::K128 => "128k",
        }
    }
}

/// The cell a decode record answers for.
///
/// `(gpu, dtype, occupancy, ctx bucket, model shape)` — the design's record key.
/// Two records in different cells are not rivals and must never be ranked
/// against each other.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DecodeCell {
    /// Hardware key, as `HardwareFingerprint::tuning_path` renders it.
    pub hardware: String,
    /// Weight dtype the packet was emitted for: `bf16`, `fp8`.
    pub dtype: String,
    /// Resident grid width — the occupancy half of the `(FORCE_MINBLK, --n-cu)`
    /// pair. 132 = 1 block/SM on this part, 264 = 2.
    pub n_cu: u32,
    pub ctx_bucket: CtxBucket,
    /// Model identity. A knob set tuned on a 26B MoE says nothing about a dense
    /// 12B, so the shape is in the key rather than in a comment.
    pub model: String,
}

impl DecodeCell {
    /// Stable one-line key, used to group records and to name report rows.
    pub fn key(&self) -> String {
        format!(
            "{}|{}|{}|ncu{}|{}",
            self.hardware,
            self.model,
            self.dtype,
            self.n_cu,
            self.ctx_bucket.label()
        )
    }
}

/// One point in the decode knob grid — everything needed to rebuild it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeKnobs {
    /// `PLOW_NV_FORCE_MINBLK` — blocks/SM the object is compiled for.
    pub minblk: u32,
    /// `plowc --n-cu` — must match `minblk` (132 ↔ 1, 264 ↔ 2).
    pub n_cu: u32,
    /// `GV_UNROLL` — dense GEMV weight streams per thread.
    pub gv_unroll: u32,
    /// `GV_UNROLL_GLU`. 0 means "not overridden", i.e. the source default.
    pub gv_unroll_glu: u32,
    /// `GV_MOE_UN` — MoE expert arm streams.
    pub gv_moe_un: u32,
    /// `PLOW_MOE_DOWN_SG` — MoE-down lane-split sub-groups.
    pub moe_down_sg: u32,
    /// `PLOW_NS_ABS` — flash decode split count, baked at packet emit.
    pub ns_abs: u32,
}

impl DecodeKnobs {
    /// The `nvcc` flags that rebuild this object, on top of the shipped recipe
    /// in `scripts/build_sm90a_cubin.sh` (which reads them from
    /// `PLOW_EXTRA_DEFINES`).
    pub fn defines(&self) -> Vec<String> {
        let mut v = vec![
            format!("-DPLOW_NV_FORCE_MINBLK={}", self.minblk),
            format!("-DGV_UNROLL={}", self.gv_unroll),
            format!("-DGV_MOE_UN={}", self.gv_moe_un),
            format!("-DPLOW_MOE_DOWN_SG={}u", self.moe_down_sg),
        ];
        if self.gv_unroll_glu != 0 {
            v.push(format!("-DGV_UNROLL_GLU={}", self.gv_unroll_glu));
        }
        v
    }

    /// The `plowc` invocation that emits the matching packet. Separate from
    /// [`defines`](Self::defines) because these two land in different artifacts
    /// and get out of sync exactly when they are written down as one string.
    pub fn emit_env(&self) -> Vec<String> {
        vec![
            "PLOW_UNISEG=1".into(),
            format!("PLOW_NS_ABS={}", self.ns_abs),
            format!("--n-cu {}", self.n_cu),
        ]
    }

    /// Compact identity, matching the sweep script's `config` field.
    pub fn label(&self) -> String {
        format!(
            "mb{}_ncu{}_un{}_glu{}_mun{}_sg{}_ns{}",
            self.minblk,
            self.n_cu,
            self.gv_unroll,
            self.gv_unroll_glu,
            self.gv_moe_un,
            self.moe_down_sg,
            self.ns_abs
        )
    }

    /// Whether the occupancy pair is self-consistent for a 132-SM part. An
    /// object built for 2 blocks/SM against a 132-block packet is not a slower
    /// configuration, it is a configuration the engine refuses to launch.
    pub fn occupancy_is_consistent(&self, sm_count: u32) -> bool {
        self.n_cu == sm_count * self.minblk
    }
}

/// One end-to-end decode timing at one grid point.
///
/// The score is `step_bench` TPOT and nothing else. The campaign's central
/// methodological finding is that the isolated microbench disagrees with the
/// megakernel — `gemv_lab_h100.cu` says row-blocking wins 1.4x on every decode
/// shape and in context it loses — so a record whose number came from a
/// microbench would be actively misleading. There is deliberately no field for
/// one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecodeMeasurement {
    pub cell: DecodeCell,
    pub knobs: DecodeKnobs,
    /// Exact context measured, not just the bucket.
    pub ctx: u32,
    pub digests: Digests,
    /// TPOT samples, nanoseconds, one per `step_bench` invocation.
    pub stats: Stats,
    /// Registers of the decode megakernel this object compiled to. Recorded
    /// because a knob that buys ILP by raising pressure for every other arm is
    /// not obviously a win — and because the campaign's "255 registers explains
    /// the regression" story was later retracted, so the number is evidence to
    /// be kept, not a conclusion.
    pub registers: Option<u32>,
    pub correctness: Correctness,
    pub state: RecordState,
    pub campaign: String,
}

impl DecodeMeasurement {
    pub fn qualification_blockers(&self) -> Vec<String> {
        blockers_for(&self.correctness, self.stats.samples)
    }

    /// Median TPOT in milliseconds — the unit every card in `perf-data/` reports.
    pub fn median_ms(&self) -> f64 {
        self.stats.median_ns / 1.0e6
    }
}

/// A cell's candidates, best first, with the margin over the runner-up.
#[derive(Clone, Debug, PartialEq)]
pub struct CellRanking {
    pub cell: DecodeCell,
    /// Sorted by median TPOT ascending.
    pub ranked: Vec<DecodeMeasurement>,
}

impl CellRanking {
    pub fn winner(&self) -> Option<&DecodeMeasurement> {
        self.ranked.first()
    }

    /// Milliseconds between the winner and the runner-up. `None` when the cell
    /// holds one candidate — a "winner" of a field of one is not a result, and
    /// the report must be able to say so rather than printing a margin of 0.
    pub fn margin_ms(&self) -> Option<f64> {
        match self.ranked.as_slice() {
            [a, b, ..] => Some(b.median_ms() - a.median_ms()),
            _ => None,
        }
    }

    /// Whether the winner is faster than the runner-up by more than the noise
    /// in either. A grid point that wins inside its own dispersion has not been
    /// shown to win at all, and the campaign found several knobs whose whole
    /// effect was smaller than run-to-run spread.
    pub fn winner_is_decisive(&self) -> bool {
        match self.ranked.as_slice() {
            [a, b, ..] => a.stats.beats(&b.stats),
            _ => false,
        }
    }
}

/// Group measurements by cell and rank each cell by median TPOT.
///
/// Ranking happens *within* a cell only. Comparing a 32k number against a 1k
/// number would make "long context is slower" look like "this knob set is
/// worse", which is precisely the mistake the ctx axis exists to prevent.
pub fn rank_by_cell(records: Vec<DecodeMeasurement>) -> Vec<CellRanking> {
    let mut by_cell: BTreeMap<String, (DecodeCell, Vec<DecodeMeasurement>)> = BTreeMap::new();
    for r in records {
        by_cell
            .entry(r.cell.key())
            .or_insert_with(|| (r.cell.clone(), Vec::new()))
            .1
            .push(r);
    }
    by_cell
        .into_values()
        .map(|(cell, mut ranked)| {
            ranked.sort_by(|a, b| {
                a.stats.median_ns.partial_cmp(&b.stats.median_ns).expect("finite medians")
            });
            CellRanking { cell, ranked }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digests() -> Digests {
        Digests {
            implementation: "impl-a".into(),
            interpreter: "interp-a".into(),
            toolchain: "cuda-13.0".into(),
            oracle: "step_bench-tpot".into(),
        }
    }

    fn knobs(minblk: u32, n_cu: u32, unroll: u32, ns: u32) -> DecodeKnobs {
        DecodeKnobs {
            minblk,
            n_cu,
            gv_unroll: unroll,
            gv_unroll_glu: 0,
            gv_moe_un: 2,
            moe_down_sg: 4,
            ns_abs: ns,
        }
    }

    fn meas(n_cu: u32, ctx: u32, knobs: DecodeKnobs, ms: f64) -> DecodeMeasurement {
        let ns = ms * 1.0e6;
        DecodeMeasurement {
            cell: DecodeCell {
                hardware: "nvidia/sm_90a/h100-nvl".into(),
                dtype: "bf16".into(),
                n_cu,
                ctx_bucket: CtxBucket::of(ctx),
                model: "gemma-4-26B-A4B-it".into(),
            },
            knobs,
            ctx,
            digests: digests(),
            stats: Stats::from_samples(vec![ns - 1000.0, ns, ns, ns, ns + 1000.0]).unwrap(),
            registers: Some(180),
            correctness: Correctness::Pass,
            state: RecordState::Provisional,
            campaign: "c1".into(),
        }
    }

    /// The buckets are geometric because the ctx curve is. 4096 belongs with
    /// 8k, not with 1k: the campaign measured 1k→8k as nearly flat and it is
    /// the 8k→32k leg that moves.
    #[test]
    fn context_buckets_are_geometric() {
        assert_eq!(CtxBucket::of(1024), CtxBucket::K1);
        assert_eq!(CtxBucket::of(4095), CtxBucket::K1);
        assert_eq!(CtxBucket::of(8192), CtxBucket::K8);
        assert_eq!(CtxBucket::of(32768), CtxBucket::K32);
        assert_eq!(CtxBucket::of(131072), CtxBucket::K128);
    }

    /// A record must be able to rebuild its own object. If `defines()` drifts
    /// from what the sweep compiled, every stored number is unreproducible.
    #[test]
    fn knobs_render_the_flags_that_rebuild_them() {
        let k = DecodeKnobs { gv_unroll_glu: 2, ..knobs(2, 264, 4, 32) };
        let d = k.defines();
        assert!(d.contains(&"-DPLOW_NV_FORCE_MINBLK=2".to_string()));
        assert!(d.contains(&"-DGV_UNROLL=4".to_string()));
        assert!(d.contains(&"-DGV_UNROLL_GLU=2".to_string()));
        assert!(d.contains(&"-DPLOW_MOE_DOWN_SG=4u".to_string()));
        assert_eq!(k.label(), "mb2_ncu264_un4_glu2_mun2_sg4_ns32");

        // 0 means "leave the source default alone" — emitting `-DGV_UNROLL_GLU=0`
        // would silently compile a different kernel than the one measured.
        assert!(!knobs(1, 132, 8, 16).defines().iter().any(|s| s.contains("GLU")));
    }

    /// The occupancy pair is one knob. A mismatched pair is not a slow
    /// configuration, it is one the engine refuses to launch.
    #[test]
    fn the_occupancy_pair_must_agree() {
        assert!(knobs(1, 132, 8, 16).occupancy_is_consistent(132));
        assert!(knobs(2, 264, 4, 32).occupancy_is_consistent(132));
        assert!(!knobs(2, 132, 4, 32).occupancy_is_consistent(132));
        assert!(!knobs(1, 264, 8, 16).occupancy_is_consistent(132));
    }

    /// Occupancy and ctx are cell keys, so records from different occupancies
    /// or contexts are never ranked against one another. Without this, "32k is
    /// slower than 1k" would be reported as "these knobs lost".
    #[test]
    fn cells_do_not_mix_occupancy_or_context() {
        let recs = vec![
            meas(132, 1024, knobs(1, 132, 8, 16), 6.04),
            meas(132, 1024, knobs(1, 132, 4, 16), 6.33),
            meas(264, 1024, knobs(2, 264, 4, 32), 5.62),
            meas(132, 32768, knobs(1, 132, 8, 16), 9.21),
        ];
        let ranked = rank_by_cell(recs);
        assert_eq!(ranked.len(), 3, "one cell per (n_cu, ctx bucket)");
        for cell in &ranked {
            let n = cell.ranked.len();
            assert!(n <= 2);
            for r in &cell.ranked {
                assert_eq!(r.cell.n_cu, cell.cell.n_cu);
                assert_eq!(r.cell.ctx_bucket, cell.cell.ctx_bucket);
            }
        }
    }

    #[test]
    fn the_winner_is_the_fastest_in_its_cell_and_carries_a_margin() {
        let ranked = rank_by_cell(vec![
            meas(132, 1024, knobs(1, 132, 4, 16), 6.33),
            meas(132, 1024, knobs(1, 132, 8, 16), 6.04),
        ]);
        let cell = &ranked[0];
        assert_eq!(cell.winner().unwrap().knobs.gv_unroll, 8);
        assert!((cell.margin_ms().unwrap() - 0.29).abs() < 1e-6);
        assert!(cell.winner_is_decisive());
    }

    /// A field of one has no margin. Reporting 0.000 would read as "a tie",
    /// which is a different and much stronger claim than "nothing else ran".
    #[test]
    fn a_single_candidate_has_no_margin_and_is_not_decisive() {
        let ranked = rank_by_cell(vec![meas(132, 1024, knobs(1, 132, 8, 16), 6.04)]);
        assert!(ranked[0].margin_ms().is_none());
        assert!(!ranked[0].winner_is_decisive());
    }

    /// A win smaller than the run-to-run spread is not a win. Several knobs in
    /// the campaign moved TPOT by less than its own jitter.
    #[test]
    fn a_win_inside_the_noise_is_reported_as_indecisive() {
        let mut a = meas(132, 1024, knobs(1, 132, 8, 16), 6.04);
        let mut b = meas(132, 1024, knobs(1, 132, 4, 16), 6.05);
        a.stats = Stats::from_samples(vec![5.9e6, 6.0e6, 6.04e6, 6.2e6, 6.4e6]).unwrap();
        b.stats = Stats::from_samples(vec![5.9e6, 6.0e6, 6.05e6, 6.2e6, 6.4e6]).unwrap();
        let ranked = rank_by_cell(vec![a, b]);
        assert!(ranked[0].margin_ms().is_some());
        assert!(!ranked[0].winner_is_decisive(), "0.01 ms apart with ~0.35 ms jitter");
    }

    /// Same gate as a kernel measurement: fast-but-unchecked never qualifies.
    #[test]
    fn an_unchecked_configuration_cannot_qualify() {
        let mut m = meas(132, 1024, knobs(1, 132, 8, 16), 6.04);
        m.correctness = Correctness::Unchecked;
        assert_eq!(m.qualification_blockers().len(), 1);
        m.correctness = Correctness::Pass;
        assert!(m.qualification_blockers().is_empty());
    }

    #[test]
    fn records_round_trip_through_json() {
        let m = meas(264, 32768, knobs(2, 264, 4, 32), 8.1);
        let text = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<DecodeMeasurement>(&text).unwrap(), m);
        assert!(text.contains("\"32k\""), "buckets serialise by their label");
    }
}

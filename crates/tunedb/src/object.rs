//! Complete-object records — rank by end-to-end performance, not isolated kernel time.
//!
//! [`KernelMeasurement`](crate::KernelMeasurement) answers "which opcode should this
//! shape dispatch to". [`DecodeMeasurement`](crate::DecodeMeasurement) answers "which
//! define-set should the decode object be built with". Neither answers the question
//! `tuning/README.md`'s `nvidia/sm_120a/rtx-5090` cell already had to invent an ad hoc
//! JSON schema for (`perf-data/px13_emit_tuning.py`, `prefill_tile_measurement.jsonl`):
//! on this architecture the real prefill GEMM tuning axis is *which interpreter object
//! is built* — tile, warp split, pipeline depth are all compile-time defines on the
//! whole object, and three dense-GEMM opcodes alias to one body, so a record keyed by
//! `op_case`/`kernel_id` cannot even express `BN=128` vs `BN=64` as two different
//! things. That doc says outright: "not loadable by `TuneStore::load_kernels`...
//! until that entity grows a build-identity column." This module is that column,
//! generalized past the one PX-13 campaign that needed it first.
//!
//! Two rules carried over from that ad hoc precedent, now enforced by the type rather
//! than by convention:
//!
//! - **The ranking key is [`ObjectMeasurement::end_to_end`], always.** `isolated` and
//!   `complete_object` are recorded as evidence — the PX-13 cell's own reason for
//!   existing is that `PGM_GLU_BN=64` measured +6.1% isolated and shipped a 2.3%
//!   end-to-end regression, so a tuner that ranked on the isolated number would have
//!   promoted the regression. [`rank_by_cell`] never reads either evidence field.
//! - **Correctness gates the same way every other record kind does.** The ad hoc
//!   JSONL rows carry `"correctness":"unchecked"` next to `"state":"qualified"` — a
//!   combination [`blockers_for`] would refuse. [`TuneStore::publish_object`] enforces
//!   it, so a migrated or newly-written row cannot silently skip the gate the way the
//!   Python emitter did.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::record::{blockers_for, Correctness, Digests, RecordState};
use crate::sample::Stats;

/// Whether an attention shape is window-limited or reads the whole context.
///
/// A separate axis from `head_dim`, because Gemma-4-12B's own layer mix
/// (`gemma4-12b-longctx-5090.md`) is the reason this exists: hd256 sliding and
/// hd512 full attention are different kernel bodies at different head dims, and a
/// GEMM cell has neither, so this is not folded into `head_dim` as a sentinel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowClass {
    /// Windowed/sliding attention (Gemma-4's hd256 layers, window=1024).
    Sliding,
    /// Full, unwindowed causal attention.
    Full,
    /// Not an attention op — GEMM, GLU, norm, etc.
    NotApplicable,
}

/// The cell a complete-object record answers for.
///
/// Every axis the mission's object-level tuner design names as part of the key:
/// architecture, toolchain, model/precision, concurrency, and problem shape. Two
/// records in different cells are not rivals and must never be ranked together —
/// the same discipline `DecodeCell` already enforces for decode knobs.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ObjectCell {
    /// Hardware key, as `HardwareFingerprint::tuning_path` renders it
    /// (`nvidia/sm_120a/rtx-5090`) — already encodes GPU architecture.
    pub hardware: String,
    /// Resident grid width the object was built/launched for. Named `sm_count`
    /// rather than reusing `DecodeCell::n_cu`'s name because this module has no
    /// `(FORCE_MINBLK, --n-cu)` pair to justify that spelling — it is simply how
    /// many SMs/CUs the measurement's grid covered.
    pub sm_count: u32,
    /// Compiler/toolkit identity string (e.g. `cuda-13.0`). Kept in the cell key,
    /// not just in `Digests`, because — as `Digests::toolchain`'s own doc
    /// comment notes — a toolchain change has moved register counts by tens of
    /// registers on this exact interpreter before, which is cell-defining, not
    /// staleness-only, for an object-level record.
    pub toolchain: String,
    pub model: String,
    /// Weight/activation dtype the object was built and the packet emitted for.
    pub dtype: String,
    /// KV cache dtype. Separate from `dtype`: `docs/flags-reference.md`
    /// documents bf16-weight+fp8-KV as one legitimate, independently-tunable
    /// combination among several, and folding it into `dtype` would silently
    /// merge cells that are not interchangeable.
    pub kv_dtype: String,
    /// Concurrency the end-to-end number was measured at.
    pub batch: u32,
    /// Prefill bucket / M this cell answers for. Named `m_bucket` (not `m`) to
    /// signal it is a bucket boundary, not an exact runtime M — mirrors
    /// `DecodeCell::ctx_bucket` serving an unmeasured value from the nearest
    /// bucket at or below it.
    pub m_bucket: u32,
    pub n: i64,
    pub k: i64,
    /// 0 for a non-attention cell (GEMM/GLU/norm) — `window` carries
    /// `NotApplicable` in that case, so a reader cannot mistake the zero for a
    /// measured hd0 arm.
    pub head_dim: u32,
    pub window: WindowClass,
}

impl ObjectCell {
    /// Stable one-line key, used to group records and to name report rows.
    pub fn key(&self) -> String {
        let window = match self.window {
            WindowClass::Sliding => "sliding",
            WindowClass::Full => "full",
            WindowClass::NotApplicable => "n/a",
        };
        format!(
            "{}|{}|{}|{}|kv{}|sm{}|b{}|m{}|n{}k{}|hd{}-{}",
            self.hardware,
            self.toolchain,
            self.model,
            self.dtype,
            self.kv_dtype,
            self.sm_count,
            self.batch,
            self.m_bucket,
            self.n,
            self.k,
            self.head_dim,
            window,
        )
    }
}

/// One point in the object configuration space — the compile-time/launch
/// choices that make two objects with the same cubin bytes-per-opcode perform
/// differently. Values are free-form strings for the axes that do not reduce
/// to one number (raster order, warp split), following `DecodeKnobs`'
/// `extra_defines`/`extra_emit` precedent for the open-ended remainder: a knob
/// family with no typed field yet rides `extra_defines` rather than blocking on
/// a schema change.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectConfig {
    /// `BM x BN x BK` (or `BK8` on the fp8 path), e.g. `"128x128x64"`.
    pub tile: String,
    /// Producer/consumer warp assignment, e.g. `"4prod_4cons_2x2"` for px22's
    /// arm, or `"uniform_4x2"` for the shipped barrier-synchronized body. Free
    /// text rather than a struct because the space of splits (uniform, N-prod/
    /// M-cons, and the warp grid each side uses) does not reduce to one shape
    /// without losing the distinction that mattered in `px22-warp-specialized-
    /// staging.md` Result 4 (decoupling vs. the forced retiling it costs).
    pub warp_split: String,
    /// Ring-buffer stage count (`PGM_STAGES`-equivalent).
    pub pipeline_depth: u32,
    /// Tile visitation order, e.g. `"row_major"`, `"swizzled"`. Free text for
    /// the same reason as `warp_split` — this axis is not swept anywhere in
    /// this tree yet, and typing it now would be guessing its eventual shape.
    pub raster_order: String,
    /// Split-K factor. 1 = no split-K.
    pub split_k: u32,
    /// Query tile width for flash-attention cells; 0 for a non-attention cell.
    pub bq: u32,
    /// KV tile width for flash-attention cells; 0 for a non-attention cell.
    pub bkv: u32,
    /// Operand double/triple/N-buffering depth, when distinct from
    /// `pipeline_depth` (e.g. a separate Q-buffer depth on the flash path).
    pub buffer_depth: u32,
    /// Extra compile-time knobs not yet promoted to a typed field, `NAME ->
    /// VALUE` in the backend's flag syntax — `DecodeKnobs::extra_defines`'
    /// exact convention, so a new knob family is not a schema break.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_defines: BTreeMap<String, String>,
}

impl ObjectConfig {
    /// Compact identity, stable regardless of `extra_defines`' insertion order.
    pub fn label(&self) -> String {
        let mut out = format!(
            "tile{}_{}_pipe{}_{}_splitk{}_bq{}_bkv{}_buf{}",
            self.tile,
            self.warp_split,
            self.pipeline_depth,
            self.raster_order,
            self.split_k,
            self.bq,
            self.bkv,
            self.buffer_depth,
        );
        for (k, v) in &self.extra_defines {
            out.push('_');
            out.push_str(&k.rsplit('_').next().unwrap_or(k).to_ascii_lowercase());
            out.push_str(v);
        }
        out
    }
}

/// One end-to-end timing of one complete interpreter object at one cell.
///
/// `isolated` and `complete_object` are evidence, never the ranking key — see
/// the module doc's PX-13 example of why a tuner that ranked on either would
/// have shipped a measured regression. `end_to_end` is required (not
/// `Option`): a record with no end-to-end number has nothing this module
/// exists to rank.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectMeasurement {
    pub cell: ObjectCell,
    pub config: ObjectConfig,
    /// Identity of the compiled object this was measured on — the "build-
    /// identity column" `tuning/README.md` says the schema is missing. A
    /// content hash of the `.cubin` (e.g. its md5/sha256), not a path: two
    /// paths can hold byte-identical objects and two identical paths can hold
    /// different bytes after a rebuild.
    pub object_hash: String,
    pub digests: Digests,
    pub registers: Option<u32>,
    pub stack_bytes: Option<u32>,
    /// 0 when confirmed spill-free by `ptxas -v`, `None` when not probed.
    /// Distinguishing "checked, zero" from "not checked" matters here more than
    /// in `KernelMeasurement`: this schema exists partly so a complete-object
    /// win cannot be qualified while silently hiding a spill regression the
    /// isolated kernel didn't have (`scripts/regcheck_sm120.sh`'s whole purpose).
    pub spill_bytes: Option<u32>,
    pub shared_mem_bytes: Option<u32>,
    /// Standalone microbench timing — the isolated-kernel number. Evidence
    /// only.
    pub isolated: Option<Stats>,
    /// The op timed inside the real compiled object (not a standalone
    /// microbench, not full request-level serving). Evidence only.
    pub complete_object: Option<Stats>,
    /// Full request-level serving wall time. THE ranking key.
    pub end_to_end: Stats,
    pub correctness: Correctness,
    pub state: RecordState,
    pub campaign: String,
}

impl ObjectMeasurement {
    pub fn qualification_blockers(&self) -> Vec<String> {
        blockers_for(&self.correctness, self.end_to_end.samples)
    }

    /// Median end-to-end time in milliseconds — the ranking value, in the unit
    /// every card in `perf-data/` reports.
    pub fn median_ms(&self) -> f64 {
        self.end_to_end.median_ns / 1.0e6
    }
}

/// A cell's candidates, best first by end-to-end median, with the margin over
/// the runner-up. Mirrors `decode::CellRanking` exactly.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectRanking {
    pub cell: ObjectCell,
    pub ranked: Vec<ObjectMeasurement>,
}

impl ObjectRanking {
    pub fn winner(&self) -> Option<&ObjectMeasurement> {
        self.ranked.first()
    }

    /// Milliseconds between the winner and the runner-up. `None` for a field of
    /// one — a "winner" with no rival is not a result.
    pub fn margin_ms(&self) -> Option<f64> {
        match self.ranked.as_slice() {
            [a, b, ..] => Some(b.median_ms() - a.median_ms()),
            _ => None,
        }
    }

    /// Whether the winner beats the runner-up by more than either's own noise.
    pub fn winner_is_decisive(&self) -> bool {
        match self.ranked.as_slice() {
            [a, b, ..] => a.end_to_end.beats(&b.end_to_end),
            _ => false,
        }
    }
}

/// Group measurements by cell and rank each cell by median end-to-end time.
///
/// Ranking happens *within* a cell only, and *only* on `end_to_end` — never on
/// `isolated` or `complete_object`. Comparing across cells or ranking on
/// evidence fields is exactly the mistake this module exists to make
/// structurally impossible rather than a matter of remembering to do it right.
pub fn rank_by_cell(records: Vec<ObjectMeasurement>) -> Vec<ObjectRanking> {
    let mut by_cell: BTreeMap<String, (ObjectCell, Vec<ObjectMeasurement>)> = BTreeMap::new();
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
                a.end_to_end
                    .median_ns
                    .partial_cmp(&b.end_to_end.median_ns)
                    .expect("finite medians")
            });
            ObjectRanking { cell, ranked }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digests() -> Digests {
        Digests {
            implementation: "op_gemm.cuh@abc123".into(),
            interpreter: "interp_sm120_pf.cubin@d7b3e784".into(),
            toolchain: "cuda-13.0".into(),
            oracle: "vllm-bench-serve-conc1".into(),
        }
    }

    fn cell(m_bucket: u32) -> ObjectCell {
        ObjectCell {
            hardware: "nvidia/sm_120a/rtx-5090".into(),
            sm_count: 170,
            toolchain: "cuda-13.0".into(),
            model: "gemma-4-12B-it".into(),
            dtype: "fp8_w8a8".into(),
            kv_dtype: "bf16".into(),
            batch: 1,
            m_bucket,
            n: 15360,
            k: 3840,
            head_dim: 0,
            window: WindowClass::NotApplicable,
        }
    }

    fn config(tile: &str, warp_split: &str) -> ObjectConfig {
        ObjectConfig {
            tile: tile.into(),
            warp_split: warp_split.into(),
            pipeline_depth: 3,
            raster_order: "row_major".into(),
            split_k: 1,
            bq: 0,
            bkv: 0,
            buffer_depth: 3,
            extra_defines: Default::default(),
        }
    }

    fn meas(m_bucket: u32, tile: &str, warp_split: &str, ms: f64) -> ObjectMeasurement {
        let ns = ms * 1.0e6;
        ObjectMeasurement {
            cell: cell(m_bucket),
            config: config(tile, warp_split),
            object_hash: "sha256:deadbeef".into(),
            digests: digests(),
            registers: Some(242),
            stack_bytes: Some(1024),
            spill_bytes: Some(0),
            shared_mem_bytes: Some(81664),
            isolated: Stats::from_samples(vec![300.0, 310.0, 305.0, 308.0, 302.0]).ok(),
            complete_object: None,
            end_to_end: Stats::from_samples(vec![ns - 5.0, ns, ns, ns, ns + 5.0]).unwrap(),
            correctness: Correctness::Pass,
            state: RecordState::Provisional,
            campaign: "iter2".into(),
        }
    }

    /// The PX-13 example this module exists to formalize: the isolated number
    /// and the end-to-end number disagree in sign, and ranking must follow
    /// end_to_end only.
    #[test]
    fn ranking_follows_end_to_end_even_when_isolated_disagrees() {
        let mut fast_isolated_slow_e2e = meas(1024, "128x128x64", "uniform_4x2", 959.0);
        fast_isolated_slow_e2e.isolated =
            Stats::from_samples(vec![250.0, 251.0, 252.0, 253.0, 254.0]).ok();
        let mut slow_isolated_fast_e2e = meas(1024, "128x64x64", "uniform_4x2", 940.0);
        slow_isolated_fast_e2e.isolated =
            Stats::from_samples(vec![300.0, 305.0, 310.0, 315.0, 320.0]).ok();

        let ranked = rank_by_cell(vec![fast_isolated_slow_e2e, slow_isolated_fast_e2e]);
        assert_eq!(ranked.len(), 1);
        let winner = ranked[0].winner().unwrap();
        assert_eq!(winner.config.tile, "128x64x64", "the end-to-end winner");
        assert!((winner.median_ms() - 940.0).abs() < 1e-6);
    }

    /// Different M buckets are different cells and must never be ranked
    /// together, the same discipline `DecodeCell::ctx_bucket` enforces.
    #[test]
    fn cells_do_not_mix_m_bucket() {
        let ranked = rank_by_cell(vec![
            meas(1024, "128x128x64", "uniform_4x2", 950.0),
            meas(1024, "128x128x64", "ws4_prod4_cons4_2x2", 900.0),
            meas(8192, "128x128x64", "uniform_4x2", 6500.0),
        ]);
        assert_eq!(ranked.len(), 2, "one cell per m_bucket");
        for r in &ranked {
            for m in &r.ranked {
                assert_eq!(m.cell.m_bucket, r.cell.m_bucket);
            }
        }
    }

    #[test]
    fn an_unchecked_configuration_cannot_qualify() {
        let mut m = meas(1024, "128x128x64", "uniform_4x2", 950.0);
        m.correctness = Correctness::Unchecked;
        assert_eq!(m.qualification_blockers().len(), 1);
        m.correctness = Correctness::Pass;
        assert!(m.qualification_blockers().is_empty());
    }

    /// A field of one has no margin and is never decisive — a "winner" with no
    /// rival has not been shown to be a winner.
    #[test]
    fn a_single_candidate_has_no_margin_and_is_not_decisive() {
        let ranked = rank_by_cell(vec![meas(1024, "128x128x64", "uniform_4x2", 950.0)]);
        assert!(ranked[0].margin_ms().is_none());
        assert!(!ranked[0].winner_is_decisive());
    }

    /// A win inside the run-to-run spread is not a win.
    #[test]
    fn a_win_inside_the_noise_is_indecisive() {
        let mut a = meas(1024, "128x128x64", "ws4_prod4_cons4_2x2", 950.0);
        let mut b = meas(1024, "128x128x64", "uniform_4x2", 950.5);
        a.end_to_end = Stats::from_samples(vec![930.0, 940.0, 950.0, 970.0, 990.0]).unwrap();
        b.end_to_end = Stats::from_samples(vec![930.0, 940.0, 950.5, 970.0, 990.0]).unwrap();
        let ranked = rank_by_cell(vec![a, b]);
        assert!(
            !ranked[0].winner_is_decisive(),
            "0.5 ms apart with ~40 ms jitter"
        );
    }

    #[test]
    fn config_label_is_stable_and_distinguishes_arms() {
        let a = config("128x128x64", "uniform_4x2");
        let b = config("128x128x64", "ws4_prod4_cons4_2x2");
        assert_ne!(a.label(), b.label());

        let mut c = a.clone();
        c.extra_defines.insert("PGM_SW8_V2".into(), "1".into());
        assert_ne!(a.label(), c.label());
    }

    #[test]
    fn records_round_trip_through_json() {
        let m = meas(1024, "128x128x64", "ws4_prod4_cons4_2x2", 940.0);
        let text = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<ObjectMeasurement>(&text).unwrap(), m);
    }
}

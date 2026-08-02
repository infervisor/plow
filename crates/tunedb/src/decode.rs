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
/// `(gpu, dtype, occupancy, ctx bucket, model shape, decode batch)` — the
/// design's record key plus the axis px15 added. Two records in different cells
/// are not rivals and must never be ranked against each other.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DecodeCell {
    /// Hardware key, as `HardwareFingerprint::tuning_path` renders it.
    pub hardware: String,
    /// Weight dtype the packet was emitted for: `bf16`, `fp8`.
    pub dtype: String,
    /// Resident grid width — the occupancy half of the `(FORCE_MINBLK, --n-cu)`
    /// pair. 132 = 1 block/SM on an H100 NVL, 264 = 2; 170/340 on a 5090.
    pub n_cu: u32,
    pub ctx_bucket: CtxBucket,
    /// Model identity. A knob set tuned on a 26B MoE says nothing about a dense
    /// 12B, so the shape is in the key rather than in a comment.
    pub model: String,
    /// Decode slots stepped together.
    ///
    /// NOT a nicety. `GV_MM_MAX` is the widest `gemv_*_rows<MM>` instantiated,
    /// so a batch of B costs `ceil(B/GV_MM_MAX)` weight passes — the knob's
    /// whole effect is a function of B, and `op_gemm.cuh`'s own ladder measures
    /// the inversion: at B=8, `=8` gives 355 tok/s and `=16` gives 294; at
    /// B=16 the order flips to 387 vs 520. A cell that cannot say which batch
    /// it was measured at cannot express either half of that, and a campaign
    /// asset shipped `=16` while serving B=8 for exactly this reason
    /// (`perf-data/px10-batched-decode.md`: −19.4% at 131k, −33.8% at 1k).
    ///
    /// Deliberately has no `serde(default)`. A record with no batch is not a
    /// record measured at batch 1, it is a record whose provenance was lost,
    /// and defaulting would turn the second into the first silently. Rows
    /// written before this field existed are migrated explicitly (see
    /// `tuning/nvidia/sm_90a/h100-nvl/decode_measurement.jsonl`), where the
    /// value is *recoverable*: the sweep script passed a literal `1` for
    /// `step_bench`'s slot count and could not have measured anything else.
    pub batch: u32,
}

impl DecodeCell {
    /// Stable one-line key, used to group records and to name report rows.
    pub fn key(&self) -> String {
        format!(
            "{}|{}|{}|ncu{}|b{}|{}",
            self.hardware,
            self.model,
            self.dtype,
            self.n_cu,
            self.batch,
            self.ctx_bucket.label()
        )
    }
}

/// Which toolchain's flag syntax rebuilds an object for a given hardware key.
///
/// The knob VALUES are portable; the way they are spelled on a command line is
/// not. Deriving this from the hardware key rather than assuming nvcc is what
/// keeps a second backend from silently inheriting `-D` flags it cannot use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// `nvcc -DNAME=VALUE`.
    Nvidia,
    /// AMD/HSA. No decode sweep has been run on it yet; `defines_for` refuses
    /// rather than emitting nvcc syntax that would build the wrong object.
    Hsa,
}

impl Backend {
    /// `nvidia/sm_90a/h100-nvl` -> `Nvidia`. Unknown vendors are an error at the
    /// call site, not a silent default.
    pub fn from_hardware(hardware: &str) -> Option<Self> {
        match hardware.split('/').next()? {
            "nvidia" => Some(Backend::Nvidia),
            "amd" | "hsa" => Some(Backend::Hsa),
            _ => None,
        }
    }
}

/// One point in the decode knob grid — everything needed to rebuild it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    /// `GV_MM_MAX` — widest batched GEMV rung instantiated. `None` = the source
    /// default (8).
    ///
    /// The batch-dependent knob, and the reason `DecodeCell` carries `batch`.
    /// It is `Option` for the same reason the flash knobs are: the value that
    /// means "not overridden" is not 0, it is *absent*, and recording 0 would
    /// describe a `gemv_walk` that instantiates no rung at all.
    #[serde(default)]
    pub gv_mm_max: Option<u32>,
    /// `PLOW_MOE_DOWN_SG` — MoE-down lane-split sub-groups.
    pub moe_down_sg: u32,
    /// `PLOW_NS_ABS` — flash decode split count, baked at packet emit.
    pub ns_abs: u32,
    // ---- flash decode arm -------------------------------------------------
    // `None` means "not overridden", i.e. whatever the build script passes.
    // That is distinct from a value: the shipped recipe sets FA_WPR=1 while the
    // source defaults it to 0, so recording 0 for "unset" would describe an
    // object nobody built.
    /// `PLOW_NV_FA_WPR` — warp-per-row score phase.
    #[serde(default)]
    pub fa_wpr: Option<u32>,
    /// `PLOW_NV_FA_GF` — GQA fusion width on the sliding layers.
    #[serde(default)]
    pub fa_gf: Option<u32>,
    /// `PLOW_NV_FA_GF_FULL` — GQA fusion width on the FULL-attention layers.
    ///
    /// A PAIR, like `(minblk, n_cu)`: the object define tells the kernel how
    /// many query heads one flash work item carries, and `PLOW_FA_GF_FULL`
    /// tells the packet compiler the same number so it can size `nsplit` to
    /// fill the grid (`n_grp = heads / GF_FULL`). Set one without the other and
    /// the run measures the DISAGREEMENT, not the knob — measured on the
    /// Gemma-4-12B full block at ctx 130560 with the packet pinned at 2, the
    /// widest fusion looked worst precisely because it was furthest from the
    /// packet's assumption. So [`defines`](Self::defines) and
    /// [`emit_env`](Self::emit_env) both render it.
    #[serde(default)]
    pub fa_gf_full: Option<u32>,
    /// `PLOW_NV_FA_KUN` — K-stream pre-issue depth.
    #[serde(default)]
    pub fa_kun: Option<u32>,
    /// `PLOW_NS_FULL_ABS` — packet-side nsplit for the full-attention layers
    /// only. 0 = the emitter's own value. Separate from `ns_abs` because those
    /// 5 layers read the whole context while the sliding ones are window-capped,
    /// which makes this the ctx-sensitive half of the split.
    #[serde(default)]
    pub ns_full_abs: u32,
    // ---- open extension point --------------------------------------------
    // The typed fields above are the families that have been swept. A family
    // that has NOT been swept yet rides these maps instead of growing the
    // struct, so adding one is not a schema break and old rows still load.
    /// Extra compile-time knobs, `NAME -> VALUE`, in the backend's flag syntax.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_defines: BTreeMap<String, String>,
    /// Extra packet-emit knobs, `NAME -> VALUE`, passed to `plowc` as env. Kept
    /// separate from `extra_defines` for the same reason `emit_env` is separate
    /// from `defines`: they land in different artifacts and drift apart exactly
    /// when they are written down as one string.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_emit: BTreeMap<String, String>,
}

impl DecodeKnobs {
    /// The `nvcc` flags that rebuild this object, on top of the shipped recipe
    /// in `scripts/build_sm90a_cubin.sh` (which reads them from
    /// `PLOW_EXTRA_DEFINES`).
    pub fn defines(&self) -> Vec<String> {
        self.defines_for(Backend::Nvidia)
            .expect("Nvidia flag syntax is always available")
    }

    /// Same, rendered in `backend`'s flag syntax. Returns `None` for a backend
    /// whose decode sweep has not been built, so a caller cannot quietly rebuild
    /// an AMD object with nvcc spellings.
    pub fn defines_for(&self, backend: Backend) -> Option<Vec<String>> {
        if backend != Backend::Nvidia {
            return None;
        }
        let mut v = vec![
            format!("-DPLOW_NV_FORCE_MINBLK={}", self.minblk),
            format!("-DGV_UNROLL={}", self.gv_unroll),
            format!("-DGV_MOE_UN={}", self.gv_moe_un),
            format!("-DPLOW_MOE_DOWN_SG={}u", self.moe_down_sg),
        ];
        if self.gv_unroll_glu != 0 {
            v.push(format!("-DGV_UNROLL_GLU={}", self.gv_unroll_glu));
        }
        if let Some(x) = self.gv_mm_max {
            v.push(format!("-DGV_MM_MAX={x}"));
        }
        if let Some(x) = self.fa_wpr {
            v.push(format!("-DPLOW_NV_FA_WPR={x}"));
        }
        if let Some(x) = self.fa_gf {
            v.push(format!("-DPLOW_NV_FA_GF={x}"));
        }
        if let Some(x) = self.fa_gf_full {
            v.push(format!("-DPLOW_NV_FA_GF_FULL={x}"));
        }
        if let Some(x) = self.fa_kun {
            v.push(format!("-DPLOW_NV_FA_KUN={x}"));
        }
        for (k, val) in &self.extra_defines {
            v.push(format!("-D{k}={val}"));
        }
        Some(v)
    }

    /// The `plowc` invocation that emits the matching packet. Separate from
    /// [`defines`](Self::defines) because these two land in different artifacts
    /// and get out of sync exactly when they are written down as one string.
    pub fn emit_env(&self) -> Vec<String> {
        let mut v = vec!["PLOW_UNISEG=1".to_string()];
        if self.ns_abs != 0 {
            v.push(format!("PLOW_NS_ABS={}", self.ns_abs));
        }
        if self.ns_full_abs != 0 {
            v.push(format!("PLOW_NS_FULL_ABS={}", self.ns_full_abs));
        }
        // The packet half of the GF_FULL pair. Emitting it here rather than
        // leaving it to the operator is the whole point: the object flag and
        // this one are the same number seen from two sides, and they drift the
        // moment a human has to remember both.
        if let Some(x) = self.fa_gf_full {
            v.push(format!("PLOW_FA_GF_FULL={x}"));
        }
        v.push(format!("--n-cu {}", self.n_cu));
        for (k, val) in &self.extra_emit {
            v.push(format!("{k}={val}"));
        }
        v
    }

    /// Compact identity, matching the sweep script's `config` field.
    pub fn label(&self) -> String {
        let d = |x: Option<u32>| x.map(|v| v.to_string()).unwrap_or_else(|| "d".into());
        let base = format!(
            "mb{}_ncu{}_un{}_glu{}_mun{}_sg{}_mm{}_ns{}_nsf{}_wpr{}_gf{}_gff{}_kun{}",
            self.minblk,
            self.n_cu,
            self.gv_unroll,
            self.gv_unroll_glu,
            self.gv_moe_un,
            self.moe_down_sg,
            d(self.gv_mm_max),
            self.ns_abs,
            self.ns_full_abs,
            d(self.fa_wpr),
            d(self.fa_gf),
            d(self.fa_gf_full),
            d(self.fa_kun),
        );
        /* Extras append in BTreeMap order so a label is stable for a knob set
         * regardless of the order the sweep discovered them. */
        let mut out = base;
        for (k, val) in self.extra_defines.iter().chain(self.extra_emit.iter()) {
            out.push('_');
            out.push_str(&k.rsplit('_').next().unwrap_or(k).to_ascii_lowercase());
            out.push_str(val);
        }
        out
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
                a.stats
                    .median_ns
                    .partial_cmp(&b.stats.median_ns)
                    .expect("finite medians")
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
            extra_defines: Default::default(),
            extra_emit: Default::default(),
            minblk,
            n_cu,
            gv_unroll: unroll,
            gv_unroll_glu: 0,
            gv_moe_un: 2,
            moe_down_sg: 4,
            gv_mm_max: None,
            ns_abs: ns,
            fa_wpr: None,
            fa_gf: None,
            fa_gf_full: None,
            fa_kun: None,
            ns_full_abs: 0,
        }
    }

    fn meas(n_cu: u32, ctx: u32, knobs: DecodeKnobs, ms: f64) -> DecodeMeasurement {
        meas_b(n_cu, 1, ctx, knobs, ms)
    }

    fn meas_b(n_cu: u32, batch: u32, ctx: u32, knobs: DecodeKnobs, ms: f64) -> DecodeMeasurement {
        let ns = ms * 1.0e6;
        DecodeMeasurement {
            cell: DecodeCell {
                hardware: "nvidia/sm_90a/h100-nvl".into(),
                dtype: "bf16".into(),
                n_cu,
                ctx_bucket: CtxBucket::of(ctx),
                model: "gemma-4-26B-A4B-it".into(),
                batch,
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
        let k = DecodeKnobs {
            gv_unroll_glu: 2,
            ..knobs(2, 264, 4, 32)
        };
        let d = k.defines();
        assert!(d.contains(&"-DPLOW_NV_FORCE_MINBLK=2".to_string()));
        assert!(d.contains(&"-DGV_UNROLL=4".to_string()));
        assert!(d.contains(&"-DGV_UNROLL_GLU=2".to_string()));
        assert!(d.contains(&"-DPLOW_MOE_DOWN_SG=4u".to_string()));
        assert_eq!(
            k.label(),
            "mb2_ncu264_un4_glu2_mun2_sg4_mmd_ns32_nsf0_wprd_gfd_gffd_kund"
        );

        // 0 means "leave the source default alone" — emitting `-DGV_UNROLL_GLU=0`
        // would silently compile a different kernel than the one measured.
        assert!(!knobs(1, 132, 8, 16)
            .defines()
            .iter()
            .any(|s| s.contains("GLU")));
    }

    /// A flash-tuned record must rebuild its own object too, or the second knob
    /// family is recorded but not reproducible.
    #[test]
    fn flash_knobs_render_the_flags_and_the_emit_that_rebuild_them() {
        let k = DecodeKnobs {
            fa_wpr: Some(1),
            fa_gf: Some(2),
            fa_gf_full: Some(8),
            fa_kun: Some(4),
            ns_full_abs: 66,
            ..knobs(2, 264, 4, 32)
        };
        let d = k.defines();
        assert!(d.contains(&"-DPLOW_NV_FA_WPR=1".to_string()));
        assert!(d.contains(&"-DPLOW_NV_FA_GF=2".to_string()));
        assert!(d.contains(&"-DPLOW_NV_FA_GF_FULL=8".to_string()));
        assert!(d.contains(&"-DPLOW_NV_FA_KUN=4".to_string()));
        // NS_FULL_ABS is packet-side, not a define — mixing the two is how a
        // record stops being reproducible.
        assert!(!d.iter().any(|s| s.contains("NS_FULL")));
        assert!(k.emit_env().contains(&"PLOW_NS_FULL_ABS=66".to_string()));
        // GF_FULL is the one knob that renders on BOTH sides, because the
        // kernel and the packet compiler each derive something from it and a
        // mismatch measures neither value.
        assert!(k.emit_env().contains(&"PLOW_FA_GF_FULL=8".to_string()));

        // None means "not overridden", which is NOT the same as a value: the
        // shipped recipe sets FA_WPR=1 while the source defaults it to 0, so
        // emitting a 0 here would describe an object nobody built.
        let plain = knobs(1, 132, 8, 16);
        assert!(!plain.defines().iter().any(|s| s.contains("FA_")));
        assert!(!plain.emit_env().iter().any(|s| s.contains("NS_FULL")));

        // Two objects differing only in a flash knob must not share a label.
        assert_ne!(
            k.label(),
            DecodeKnobs {
                fa_gf_full: Some(4),
                ..k
            }
            .label()
        );
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

    /// The px15 axis. `GV_MM_MAX`'s optimum INVERTS with batch — `op_gemm.cuh`
    /// measures 355 tok/s for `=8` and 294 for `=16` at B=8, then 387 vs 520 at
    /// B=16 — so two batches must not be ranked against each other. Without
    /// this the tuner would report one winner for a knob that provably has two,
    /// which is how a campaign asset came to ship `=16` while serving B=8.
    #[test]
    fn cells_do_not_mix_batch() {
        let mm = |x: u32| DecodeKnobs {
            gv_mm_max: Some(x),
            ..knobs(1, 132, 8, 16)
        };
        let ranked = rank_by_cell(vec![
            // B=8: the narrow rung wins (no spill tax, one extra weight pass).
            meas_b(132, 8, 1024, mm(8), 22.53),
            meas_b(132, 8, 1024, mm(16), 27.21),
            // B=16: the order flips — the halved weight traffic now pays.
            meas_b(132, 16, 1024, mm(8), 41.34),
            meas_b(132, 16, 1024, mm(16), 30.80),
        ]);
        assert_eq!(ranked.len(), 2, "one cell per batch");
        let by_batch: BTreeMap<u32, u32> = ranked
            .iter()
            .map(|c| (c.cell.batch, c.winner().unwrap().knobs.gv_mm_max.unwrap()))
            .collect();
        assert_eq!(by_batch[&8], 8);
        assert_eq!(by_batch[&16], 16);
        // Pooled into one cell the 16-winner would be invisible: B=8's absolute
        // times are lower, so a batch-blind ranking answers "8" for every batch.
        let pooled = rank_by_cell(vec![
            meas_b(132, 1, 1024, mm(8), 22.53),
            meas_b(132, 1, 1024, mm(16), 27.21),
            meas_b(132, 1, 1024, mm(8), 41.34),
            meas_b(132, 1, 1024, mm(16), 30.80),
        ]);
        assert_eq!(pooled.len(), 1);
        assert_eq!(pooled[0].winner().unwrap().knobs.gv_mm_max, Some(8));
    }

    /// A stored record with no batch is not a batch-1 record, it is a record
    /// whose provenance was lost. `DecodeCell.batch` therefore has no
    /// `serde(default)`, so an un-migrated row fails loudly instead of being
    /// re-labelled as something nobody measured.
    #[test]
    fn a_cell_without_a_batch_refuses_to_load() {
        let no_batch = r#"{"hardware":"nvidia/sm_90a/h100-nvl","dtype":"fp8","n_cu":132,
                           "ctx_bucket":"1k","model":"gemma-4-26B-A4B-it"}"#;
        let e = serde_json::from_str::<DecodeCell>(no_batch).expect_err("must not load");
        assert!(
            e.to_string().contains("batch"),
            "the error names the missing field: {e}"
        );
        let with_batch = r#"{"hardware":"nvidia/sm_90a/h100-nvl","dtype":"fp8","n_cu":132,
                             "ctx_bucket":"1k","model":"gemma-4-26B-A4B-it","batch":1}"#;
        assert_eq!(
            serde_json::from_str::<DecodeCell>(with_batch)
                .unwrap()
                .batch,
            1
        );
    }

    /// `GV_MM_MAX` must rebuild its own object, and "unset" must emit no flag —
    /// a `-DGV_MM_MAX=0` would instantiate no rung at all.
    #[test]
    fn the_batched_gemv_rung_renders_its_flag_and_absence_renders_nothing() {
        let k = DecodeKnobs {
            gv_mm_max: Some(16),
            ..knobs(1, 132, 8, 16)
        };
        assert!(k.defines().contains(&"-DGV_MM_MAX=16".to_string()));
        assert!(!knobs(1, 132, 8, 16)
            .defines()
            .iter()
            .any(|s| s.contains("GV_MM_MAX")));
        assert_ne!(k.label(), knobs(1, 132, 8, 16).label());
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
        assert!(
            !ranked[0].winner_is_decisive(),
            "0.01 ms apart with ~0.35 ms jitter"
        );
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
    /// A family with no typed field yet must still be recordable and
    /// rebuildable — without growing the struct.
    #[test]
    fn a_new_op_family_rides_the_extra_maps() {
        let mut k = knobs(2, 264, 4, 32);
        k.extra_defines
            .insert("PLOW_NV_FUTURE_OP".into(), "3".into());
        k.extra_emit.insert("PLOW_FUTURE_EMIT".into(), "7".into());
        let d = k.defines();
        assert!(d.contains(&"-DPLOW_NV_FUTURE_OP=3".to_string()));
        // packet-side knobs must NOT leak into the object's defines
        assert!(!d.iter().any(|f| f.contains("FUTURE_EMIT")));
        assert!(k.emit_env().contains(&"PLOW_FUTURE_EMIT=7".to_string()));
        assert_ne!(k.label(), knobs(2, 264, 4, 32).label());
    }

    /// Rows written before a family existed still deserialize.
    #[test]
    fn knobs_without_extras_still_load() {
        let json = r#"{"minblk":2,"n_cu":264,"gv_unroll":4,"gv_unroll_glu":0,
                       "gv_moe_un":2,"moe_down_sg":4,"ns_abs":32}"#;
        let k: DecodeKnobs = serde_json::from_str(json).expect("legacy row loads");
        assert!(k.extra_defines.is_empty() && k.extra_emit.is_empty());
    }

    /// Knob VALUES are portable; their spelling is not. A backend with no sweep
    /// must refuse rather than inherit nvcc syntax and build the wrong object.
    #[test]
    fn a_backend_without_a_sweep_refuses_to_render_flags() {
        let k = knobs(2, 264, 4, 32);
        assert!(k.defines_for(Backend::Nvidia).is_some());
        assert!(k.defines_for(Backend::Hsa).is_none());
        assert_eq!(
            Backend::from_hardware("nvidia/sm_90a/h100-nvl"),
            Some(Backend::Nvidia)
        );
        assert_eq!(
            Backend::from_hardware("amd/gfx950/mi355x"),
            Some(Backend::Hsa)
        );
        assert_eq!(Backend::from_hardware("acme/tpu"), None);
    }
}

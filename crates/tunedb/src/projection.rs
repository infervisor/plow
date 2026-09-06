use crate::{Correctness, Digests, RecordState, Stats};
use serde::{Deserialize, Serialize};
pub const PROJECTION_ORACLE: &str = "bf16-splitk-packet-v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionTiming {
    pub median_ns: f64,
    pub p10_ns: f64,
    pub p90_ns: f64,
    pub samples: usize,
}
impl From<Stats> for ProjectionTiming {
    fn from(s: Stats) -> Self {
        Self {
            median_ns: s.median_ns,
            p10_ns: s.p10_ns,
            p90_ns: s.p90_ns,
            samples: s.samples,
        }
    }
}
impl ProjectionTiming {
    fn beats(&self, baseline: &Self) -> bool {
        // The same dispersion margin as Stats::beats; legacy CSVs omit minima.
        self.median_ns + (self.p90_ns - self.median_ns).max(baseline.p90_ns - baseline.median_ns)
            < baseline.median_ns
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionCell {
    pub hardware: String,
    pub n_cu: u32,
    pub threads: u32,
    pub rows: u32,
    pub n: u32,
    pub k: u32,
}
impl ProjectionCell {
    pub fn key(&self) -> String {
        format!(
            "{}|cu{}|threads{}|bf16|m{}n{}k{}",
            self.hardware, self.n_cu, self.threads, self.rows, self.n, self.k
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeBlockTiming {
    pub median_ns: f64,
    pub p95_ns: f64,
    pub samples: usize,
}
impl NativeBlockTiming {
    fn valid(&self) -> bool {
        self.median_ns.is_finite()
            && self.median_ns > 0.
            && self.p95_ns.is_finite()
            && self.p95_ns >= self.median_ns
            && self.samples >= Stats::MIN_SAMPLES
    }
    fn beats(&self, baseline: &Self) -> bool {
        self.valid()
            && baseline.valid()
            && self.median_ns
                + (self.p95_ns - self.median_ns).max(baseline.p95_ns - baseline.median_ns)
                < baseline.median_ns
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeBlockGuard {
    pub context_tokens: u32,
    pub packet_sha256: String,
    pub stats: NativeBlockTiming,
    pub baseline: NativeBlockTiming,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionMeasurement {
    pub cell: ProjectionCell,
    pub split: u32,
    pub baseline_object: plow_asset::decode_objects::DecodeObject,
    pub candidate_object: plow_asset::decode_objects::DecodeObject,
    pub baseline_registers: u32,
    pub candidate_registers: u32,
    pub native_blocks: Vec<NativeBlockGuard>,
    pub digests: Digests,
    // Full zero + compute + finalizer interval, never the compute kernel alone.
    pub stats: ProjectionTiming,
    pub baseline: ProjectionTiming,
    pub correctness: Correctness,
    pub state: RecordState,
    pub campaign: String,
}
impl ProjectionMeasurement {
    pub fn qualification_blockers(&self) -> Vec<String> {
        let mut blockers = crate::blockers_for(
            &self.correctness,
            self.stats.samples.min(self.baseline.samples),
        );
        if !matches!(self.cell.rows, 4 | 8 | 16)
            || self.cell.n == 0
            || self.cell.k == 0
            || self.cell.k % 8 != 0
            || self.cell.n_cu == 0
            || self.cell.threads != 256
            || !matches!(self.split, 1 | 2 | 4 | 8 | 16)
        {
            blockers.push("unsupported projection geometry".into());
        }
        if self.baseline_object.validate(self.cell.n_cu).is_err()
            || self.candidate_object.validate(self.cell.n_cu).is_err()
            || self.candidate_object.sha256 != self.digests.interpreter
            || self.candidate_object.threads != self.cell.threads
            || self.baseline_object.threads != self.cell.threads
            || self.candidate_object.arena_bytes < 82944
            || self.baseline_registers == 0
            || self.candidate_registers == 0
            || self.native_blocks.is_empty()
        {
            blockers
                .push("missing baseline/candidate resources or native block qualification".into());
        }
        for guard in &self.native_blocks {
            if guard.context_tokens == 0
                || guard.packet_sha256.len() != 64
                || !guard
                    .packet_sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                || !guard.stats.beats(&guard.baseline)
                || guard.stats.samples.min(guard.baseline.samples) < Stats::MIN_SAMPLES
            {
                blockers.push("native block role overhead is not qualified".into());
            }
        }
        for stats in [&self.stats, &self.baseline] {
            if [stats.p10_ns, stats.median_ns, stats.p90_ns]
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.)
                || stats.p10_ns > stats.median_ns
                || stats.median_ns > stats.p90_ns
            {
                blockers.push("invalid timing distribution".into());
            }
        }
        if self.campaign.is_empty()
            || self.digests.implementation.is_empty()
            || self.digests.interpreter.is_empty()
            || self.digests.toolchain.is_empty()
            || self.digests.oracle != PROJECTION_ORACLE
        {
            blockers.push("missing projection qualification identity".into());
        }
        blockers
    }
}
pub fn select_projection<'a>(
    records: &'a [ProjectionMeasurement],
    cell: &ProjectionCell,
    want: &Digests,
    baseline: &plow_asset::decode_objects::DecodeObject,
    splitk_abi: Option<u32>,
    arena: u32,
) -> Option<&'a ProjectionMeasurement> {
    if splitk_abi != Some(1) || arena < 82944 {
        return None;
    }
    records
        .iter()
        .filter(|r| {
            r.cell == *cell
                && r.baseline_object == *baseline
                && r.candidate_object.arena_bytes == arena
                && r.state.is_selectable()
                && r.qualification_blockers().is_empty()
                && r.digests.stale_against(want).is_empty()
                && r.stats.beats(&r.baseline)
        })
        .min_by(|a, b| a.stats.median_ns.total_cmp(&b.stats.median_ns))
}
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) fn record() -> ProjectionMeasurement {
        ProjectionMeasurement {
            cell: ProjectionCell {
                hardware: "nvidia/sm_90a/h100-sxm".into(),
                n_cu: 132,
                threads: 256,
                rows: 16,
                n: 3840,
                k: 15360,
            },
            split: 8,
            baseline_object: plow_asset::decode_objects::DecodeObject {
                file: "old.cubin".into(),
                sha256: "a".repeat(64),
                profile: "sm90a".into(),
                entry: "_Z12interp_sm90a11PlowProgram".into(),
                threads: 256,
                arena_bytes: 16448,
                grid: 132,
            },
            candidate_object: plow_asset::decode_objects::DecodeObject {
                file: "s8.cubin".into(),
                sha256: "b".repeat(64),
                profile: "sm90a".into(),
                entry: "_Z12interp_sm90a11PlowProgram".into(),
                threads: 256,
                arena_bytes: 82944,
                grid: 132,
            },
            baseline_registers: 200,
            candidate_registers: 216,
            native_blocks: vec![NativeBlockGuard {
                context_tokens: 1024,
                packet_sha256: "c".repeat(64),
                stats: NativeBlockTiming {
                    median_ns: 800.,
                    p95_ns: 810.,
                    samples: 5,
                },
                baseline: NativeBlockTiming {
                    median_ns: 1100.,
                    p95_ns: 1110.,
                    samples: 5,
                },
            }],
            digests: Digests {
                implementation: "body".into(),
                interpreter: "b".repeat(64),
                toolchain: "cuda".into(),
                oracle: PROJECTION_ORACLE.into(),
            },
            stats: Stats::from_samples(vec![68.; 5]).unwrap().into(),
            baseline: Stats::from_samples(vec![392.; 5]).unwrap().into(),
            correctness: Correctness::Pass,
            state: RecordState::Qualified,
            campaign: "paired-packet-gates".into(),
        }
    }
    #[test]
    fn exact_qualified_cell_only() {
        let r = record();
        assert_eq!(
            select_projection(
                std::slice::from_ref(&r),
                &r.cell,
                &r.digests,
                &r.baseline_object,
                Some(1),
                82944
            )
            .unwrap()
            .split,
            8
        );
        for case in 0..15 {
            let mut x = r.clone();
            match case {
                0 => x.cell.rows = 8,
                1 => x.cell.n_cu = 528,
                2 => x.state = RecordState::Provisional,
                3 => x.correctness = Correctness::Unchecked,
                4 => x.digests.interpreter = "old".into(),
                5 => x.digests.toolchain = "old".into(),
                6 => x.split = 3,
                7 => x.stats.samples = 1,
                8 => x.stats.median_ns = 500.,
                9 => x.stats.p90_ns = f64::NAN,
                10 => x.native_blocks.clear(),
                11 => x.native_blocks[0].stats.median_ns = 1200.,
                12 => x.baseline_object.arena_bytes = 82944,
                13 => x.candidate_object.sha256 = "d".repeat(64),
                14 => x.native_blocks[0].stats.samples = 1,
                _ => unreachable!(),
            }
            assert!(
                select_projection(
                    &[x],
                    &r.cell,
                    &r.digests,
                    &r.baseline_object,
                    Some(1),
                    82944
                )
                .is_none(),
                "case {case}"
            );
        }
        assert!(select_projection(
            std::slice::from_ref(&r),
            &r.cell,
            &r.digests,
            &r.baseline_object,
            None,
            82944
        )
        .is_none());
        assert!(select_projection(
            std::slice::from_ref(&r),
            &r.cell,
            &r.digests,
            &r.baseline_object,
            Some(1),
            16448
        )
        .is_none());
        assert!(
            select_projection(&[], &r.cell, &r.digests, &r.baseline_object, Some(1), 82944)
                .is_none()
        );
    }
}

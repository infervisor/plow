//! Model-neutral attention runtime-policy records.
//!
//! Attention choices are not GEMM tiles. `nsplit` changes packet scratch and
//! merge operands, while a persistent body must actually be compiled into the
//! enclosing interpreter. Selection checks qualification and packet/object
//! capability before returning a measured choice.

use serde::{Deserialize, Serialize};

use crate::{Correctness, Digests, RecordState, Stats};

pub const ATTENTION_ORACLE: &str = "attention-chain-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvBucket {
    #[serde(rename = "1k")]
    K1,
    #[serde(rename = "4k")]
    K4,
    #[serde(rename = "8k")]
    K8,
    #[serde(rename = "16k")]
    K16,
    #[serde(rename = "32k")]
    K32,
    #[serde(rename = "64k")]
    K64,
    #[serde(rename = "128k")]
    K128,
}

impl KvBucket {
    /// Round down. A record measured at a larger KV length is not evidence for
    /// a smaller request where split/merge overhead has a different balance.
    pub fn of(kv_len: u32) -> Self {
        match kv_len {
            0..=2047 => Self::K1,
            2048..=6143 => Self::K4,
            6144..=12287 => Self::K8,
            12288..=24575 => Self::K16,
            24576..=49151 => Self::K32,
            49152..=98303 => Self::K64,
            _ => Self::K128,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::K1 => "1k",
            Self::K4 => "4k",
            Self::K8 => "8k",
            Self::K16 => "16k",
            Self::K32 => "32k",
            Self::K64 => "64k",
            Self::K128 => "128k",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionAlgorithm {
    SplitReduce,
    Persistent,
}

/// A model-neutral cell. `shape` is operator geometry such as
/// `mla/dk512/dr64/h12/gf4`, never a model identifier.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AttentionCell {
    pub hardware: String,
    pub n_cu: u32,
    pub decode_rung: u32,
    pub kv_bucket: KvBucket,
    pub shape: String,
}

impl AttentionCell {
    pub fn key(&self) -> String {
        format!(
            "{}|ncu{}|b{}|{}|{}",
            self.hardware,
            self.n_cu,
            self.decode_rung,
            self.kv_bucket.label(),
            self.shape
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttentionMeasurement {
    pub cell: AttentionCell,
    pub algorithm: AttentionAlgorithm,
    pub nsplit: u32,
    pub digests: Digests,
    pub stats: Stats,
    pub correctness: Correctness,
    pub state: RecordState,
    pub campaign: String,
}

impl AttentionMeasurement {
    pub fn qualification_blockers(&self) -> Vec<String> {
        crate::blockers_for(&self.correctness, self.stats.samples)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionCapabilities {
    pub max_nsplit: u32,
    pub persistent: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionSource {
    FixedFallback,
    Qualified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionSelection {
    pub algorithm: AttentionAlgorithm,
    pub nsplit: u32,
    pub source: AttentionSource,
}

fn eligible(record: &AttentionMeasurement, cell: &AttentionCell, want: &Digests,
    caps: AttentionCapabilities) -> bool {
    record.cell == *cell && record.state.is_selectable()
        && record.qualification_blockers().is_empty()
        && record.stats.median_ns.is_finite() && record.stats.median_ns > 0.0
        && record.digests.stale_against(want).is_empty()
        && record.nsplit >= 1 && record.nsplit <= caps.max_nsplit.max(1)
        && (record.algorithm != AttentionAlgorithm::Persistent || caps.persistent)
}

/// Select the fastest exact-cell record that is qualified, current, correct,
/// and executable by this packet/object pair. Missing evidence preserves the
/// fixed fallback; there is deliberately no nearest-neighbour interpolation.
pub fn select_attention(
    records: &[AttentionMeasurement],
    cell: &AttentionCell,
    want: &Digests,
    caps: AttentionCapabilities,
    fallback_nsplit: u32,
) -> AttentionSelection {
    if let Some(best) = records
        .iter()
        .filter(|record| eligible(record, cell, want, caps))
        .min_by(|a, b| a.stats.median_ns.total_cmp(&b.stats.median_ns))
    {
        return AttentionSelection {
            algorithm: best.algorithm,
            nsplit: best.nsplit,
            source: AttentionSource::Qualified,
        };
    }
    AttentionSelection {
        algorithm: AttentionAlgorithm::SplitReduce,
        nsplit: fallback_nsplit.clamp(1, caps.max_nsplit.max(1)),
        source: AttentionSource::FixedFallback,
    }
}

/// The compiler can check selection against the exact eligible measured population.
/// This is not a four-arm serving-performance certificate.
pub fn policy_witness(records: &[AttentionMeasurement], cell: &AttentionCell, want: &Digests,
    caps: AttentionCapabilities, selected: AttentionSelection) -> Option<serde_json::Value> {
    if selected.source != AttentionSource::Qualified { return None; }
    let domain = plow_asset::decode_objects::image_sha256(&serde_json::to_vec(
        &(cell, want, caps.max_nsplit, caps.persistent)).ok()?);
    let candidates: std::collections::BTreeMap<_, _> = records.iter()
        .filter(|record| eligible(record, cell, want, caps))
        .map(|record| {
            let key = plow_asset::decode_objects::image_sha256(&serde_json::to_vec(record).unwrap());
            (key, record)
        }).collect();
    let (key, _) = candidates.iter().filter(|(_, record)|
        record.algorithm == selected.algorithm && record.nsplit == selected.nsplit)
        .min_by(|(_, a), (_, b)| a.stats.median_ns.total_cmp(&b.stats.median_ns))?;
    Some(serde_json::json!({
        "required": [domain],
        "candidates": candidates.iter().map(|(key, record)| serde_json::json!({
            "domain": domain, "key": key, "cost": record.stats.median_ns, "qualified": true
        })).collect::<Vec<_>>(),
        "choices": [{"domain": domain, "key": key}],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digests(tag: &str) -> Digests {
        Digests {
            execution: None,
            implementation: tag.into(),
            interpreter: tag.into(),
            toolchain: "rocm-7.2".into(),
            oracle: "attention-chain-v1".into(),
        }
    }

    fn cell(rung: u32, kv: u32) -> AttentionCell {
        AttentionCell {
            hardware: "amd/gfx950/mi350x".into(),
            n_cu: 256,
            decode_rung: rung,
            kv_bucket: KvBucket::of(kv),
            shape: "mla/dk512/dr64/h12/gf4".into(),
        }
    }

    fn rec(cell: AttentionCell, ns: u32, us: f64) -> AttentionMeasurement {
        AttentionMeasurement {
            cell,
            algorithm: AttentionAlgorithm::SplitReduce,
            nsplit: ns,
            digests: digests("build-a"),
            stats: Stats::from_samples(vec![us * 1000.0; 5]).unwrap(),
            correctness: Correctness::Pass,
            state: RecordState::Qualified,
            campaign: "test".into(),
        }
    }

    #[test]
    fn exact_cell_selects_fastest_qualified_record() {
        let c = cell(32, 8192);
        let got = select_attention(
            &[rec(c.clone(), 16, 80.0), rec(c.clone(), 32, 60.0)],
            &c,
            &digests("build-a"),
            AttentionCapabilities {
                max_nsplit: 64,
                persistent: false,
            },
            64,
        );
        assert_eq!(got.nsplit, 32);
        assert_eq!(got.source, AttentionSource::Qualified);
    }

    #[test]
    fn witness_binds_measured_population_and_ignores_invalid_records() {
        let c = cell(16, 8192);
        let want = digests("build-a");
        let caps = AttentionCapabilities { max_nsplit: 64, persistent: false };
        let baseline = rec(c.clone(), 16, 80.0);
        let fast = rec(c.clone(), 32, 60.0);
        let mut invalid = rec(c.clone(), 8, 1.0);
        invalid.stats.samples = 1;
        let mut nan = invalid.clone();
        nan.stats.samples = 5;
        nan.stats.median_ns = f64::NAN;
        let records = vec![baseline, fast.clone(), fast, invalid, nan];
        let selected = select_attention(&records, &c, &want, caps, 64);
        assert_eq!(selected.nsplit, 32);
        let witness = policy_witness(&records, &c, &want, caps, selected).unwrap();
        assert_eq!(witness["candidates"].as_array().unwrap().len(), 2);
        let chosen = witness["choices"][0]["key"].as_str().unwrap();
        let candidate = witness["candidates"].as_array().unwrap().iter()
            .find(|entry| entry["key"] == chosen).unwrap();
        assert_eq!(candidate["cost"], 60000.0);
        let mut changed = want.clone();
        changed.oracle = "new-oracle".into();
        assert!(policy_witness(&records, &c, &changed, caps, selected).is_none());
    }

    #[test]
    fn decode_rungs_select_independently() {
        let b1 = cell(1, 8192);
        let b8 = cell(8, 8192);
        let records = [rec(b1.clone(), 32, 38.0), rec(b8.clone(), 16, 80.0)];
        let caps = AttentionCapabilities {
            max_nsplit: 64,
            persistent: false,
        };
        assert_eq!(
            select_attention(&records, &b1, &digests("build-a"), caps, 64).nsplit,
            32
        );
        assert_eq!(
            select_attention(&records, &b8, &digests("build-a"), caps, 64).nsplit,
            16
        );
    }

    #[test]
    fn stale_or_uncompiled_choices_preserve_fallback() {
        let c = cell(1, 4096);
        let mut stale = rec(c.clone(), 16, 1.0);
        stale.digests = digests("old");
        let mut persistent = rec(c.clone(), 16, 0.5);
        persistent.algorithm = AttentionAlgorithm::Persistent;
        let got = select_attention(
            &[stale, persistent],
            &c,
            &digests("build-a"),
            AttentionCapabilities {
                max_nsplit: 64,
                persistent: false,
            },
            64,
        );
        assert_eq!(got.source, AttentionSource::FixedFallback);
        assert_eq!(got.nsplit, 64);
        assert_eq!(got.algorithm, AttentionAlgorithm::SplitReduce);
    }

    #[test]
    fn selection_never_exceeds_compiled_scratch_ceiling() {
        let c = cell(128, 131072);
        let got = select_attention(
            &[rec(c.clone(), 128, 1.0)],
            &c,
            &digests("build-a"),
            AttentionCapabilities {
                max_nsplit: 64,
                persistent: false,
            },
            128,
        );
        assert_eq!(got.nsplit, 64);
        assert_eq!(got.source, AttentionSource::FixedFallback);
    }
}

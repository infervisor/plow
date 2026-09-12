//! Exact-cell records for packet-declared attention objects.

use serde::{Deserialize, Serialize};

use crate::{Correctness, Digests, KvBucket, RecordState, Stats};

pub const ATTENTION_ROLE_ORACLE: &str = "attention-packed-role-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionTopology {
    Single,
    PackedHomogeneous,
    PackedRagged,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionRoleCell {
    pub hardware: String,
    pub n_cu: u32,
    pub arch: String,
    pub dtype: String,
    pub kv_dtype: String,
    pub head_dim: u32,
    pub gqa: u32,
    pub window: u32,
    pub m_rung: u32,
    pub live_kv_bucket: KvBucket,
    pub topology: AttentionTopology,
}

impl AttentionRoleCell {
    pub fn key(&self) -> String {
        format!(
            "{}|ncu{}|{}|{}/{}|hd{}gqa{}w{}|m{}|kv{}|{:?}",
            self.hardware,
            self.n_cu,
            self.arch,
            self.dtype,
            self.kv_dtype,
            self.head_dim,
            self.gqa,
            self.window,
            self.m_rung,
            self.live_kv_bucket.label(),
            self.topology,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionRoleConfig {
    pub query_tile: u32,
    pub kv_tile: u32,
    pub warps: u32,
    pub stages: u32,
    pub nsplit: u32,
    pub group_factor: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionRoleMeasurement {
    pub cell: AttentionRoleCell,
    pub role: u8,
    pub object_file: String,
    pub object_sha256: String,
    pub program_sha256: String,
    pub config: AttentionRoleConfig,
    /// Loaded packet/counter-path timing. This is the selection score.
    pub stats: Stats,
    pub baseline: Stats,
    pub digests: Digests,
    pub correctness: Correctness,
    pub state: RecordState,
    pub campaign: String,
}

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl AttentionRoleMeasurement {
    pub fn qualification_blockers(&self) -> Vec<String> {
        let mut blockers = crate::blockers_for(
            &self.correctness,
            self.stats.samples.min(self.baseline.samples),
        );
        if self.cell.hardware.is_empty()
            || self.cell.n_cu == 0
            || self.cell.arch.is_empty()
            || self.cell.dtype.is_empty()
            || self.cell.kv_dtype.is_empty()
            || self.cell.head_dim == 0
            || self.cell.gqa == 0
            || self.cell.m_rung == 0
        {
            blockers.push("invalid attention role cell".into());
        }
        if self.role == 0
            || self.object_file.is_empty()
            || std::path::Path::new(&self.object_file)
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
            || !digest(&self.object_sha256)
            || !digest(&self.program_sha256)
            || self.digests.interpreter != self.object_sha256
        {
            blockers.push("invalid attention role object identity".into());
        }
        if self.config.query_tile == 0
            || self.config.kv_tile == 0
            || self.config.warps == 0
            || self.config.stages == 0
            || self.config.nsplit == 0
            || self.config.group_factor == 0
        {
            blockers.push("invalid attention role configuration".into());
        }
        if !self.stats.beats(&self.baseline) {
            blockers.push("attention role does not beat its packet baseline".into());
        }
        if self.campaign.is_empty()
            || self.digests.implementation.is_empty()
            || self.digests.toolchain.is_empty()
            || self.digests.oracle != ATTENTION_ROLE_ORACLE
        {
            blockers.push("missing attention role qualification identity".into());
        }
        blockers
    }
}

/// Select only a qualified, current record for one exact packet cell.
pub fn select_attention_role<'a>(
    records: &'a [AttentionRoleMeasurement],
    cell: &AttentionRoleCell,
    role: u8,
    program_sha256: &str,
    implementation: &str,
    toolchain: &str,
) -> Option<&'a AttentionRoleMeasurement> {
    records
        .iter()
        .filter(|record| {
            record.cell == *cell
                && record.role == role
                && record.program_sha256 == program_sha256
                && record.state.is_selectable()
                && record.qualification_blockers().is_empty()
                && record.digests.implementation == implementation
                && record.digests.toolchain == toolchain
                && record.digests.oracle == ATTENTION_ROLE_ORACLE
        })
        .min_by(|a, b| a.stats.median_ns.total_cmp(&b.stats.median_ns))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell() -> AttentionRoleCell {
        AttentionRoleCell {
            hardware: "nvidia/sm_90a/h100-sxm".into(),
            n_cu: 132,
            arch: "sm90a".into(),
            dtype: "bf16".into(),
            kv_dtype: "bf16".into(),
            head_dim: 256,
            gqa: 2,
            window: 1024,
            m_rung: 512,
            live_kv_bucket: KvBucket::K16,
            topology: AttentionTopology::PackedRagged,
        }
    }

    fn record() -> AttentionRoleMeasurement {
        AttentionRoleMeasurement {
            cell: cell(),
            role: 10,
            object_file: "attention.cubin".into(),
            object_sha256: "a".repeat(64),
            program_sha256: "b".repeat(64),
            config: AttentionRoleConfig {
                query_tile: 64,
                kv_tile: 64,
                warps: 8,
                stages: 2,
                nsplit: 1,
                group_factor: 2,
            },
            stats: Stats::from_samples(vec![50.0; 5]).unwrap(),
            baseline: Stats::from_samples(vec![100.0; 5]).unwrap(),
            digests: Digests {
                implementation: "impl".into(),
                interpreter: "a".repeat(64),
                toolchain: "cuda-13".into(),
                oracle: ATTENTION_ROLE_ORACLE.into(),
            },
            correctness: Correctness::Pass,
            state: RecordState::Qualified,
            campaign: "gemma-attention".into(),
        }
    }

    #[test]
    fn qualified_current_exact_cell_selects() {
        let record = record();
        assert_eq!(
            select_attention_role(&[record], &cell(), 10, &"b".repeat(64), "impl", "cuda-13")
                .map(|record| record.role),
            Some(10)
        );
    }

    #[test]
    fn stale_provisional_losing_and_inexact_records_do_not_select() {
        let check = |record: AttentionRoleMeasurement| {
            select_attention_role(&[record], &cell(), 10, &"b".repeat(64), "impl", "cuda-13")
                .is_none()
        };
        let mut stale = record();
        stale.digests.implementation = "old".into();
        assert!(check(stale));
        let mut provisional = record();
        provisional.state = RecordState::Provisional;
        assert!(check(provisional));
        let mut losing = record();
        losing.stats = Stats::from_samples(vec![101.0; 5]).unwrap();
        assert!(check(losing));
        let mut inexact = record();
        inexact.cell.m_rung = 256;
        assert!(check(inexact));
    }
}

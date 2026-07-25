//! The stored entities and the state a record may be in.

use serde::{Deserialize, Serialize};

use crate::sample::Stats;

/// Everything that, if it changes, invalidates a measurement without saying
/// anything about unrelated ones.
///
/// Staleness is per-digest and not global: recompiling one kernel must not
/// throw away the campaign for every other. Each field is the identity of one
/// input to the number that was measured.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Digests {
    /// Identity of the kernel body. Changes when the code changes.
    pub implementation: String,
    /// Identity of the built interpreter object the kernel ran inside. Two
    /// kernels can be byte-identical and still perform differently in objects
    /// with different register pressure, so this is separate.
    pub interpreter: String,
    /// Compiler/toolkit identity. The sm90a interpreter's register count moved
    /// by 58 between the documented figure and CUDA 13.0, so this is load
    /// bearing rather than provenance decoration.
    pub toolchain: String,
    /// Identity of the correctness oracle the result was checked against. A
    /// weaker oracle does not certify what a stronger one did.
    pub oracle: String,
}

impl Digests {
    /// Which parts of `self` disagree with `want`. Empty means the record is
    /// still current.
    pub fn stale_against(&self, want: &Digests) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.implementation != want.implementation {
            out.push("implementation");
        }
        if self.interpreter != want.interpreter {
            out.push("interpreter");
        }
        if self.toolchain != want.toolchain {
            out.push("toolchain");
        }
        if self.oracle != want.oracle {
            out.push("oracle");
        }
        out
    }
}

/// Where a record sits in its lifecycle.
///
/// A microbenchmark result enters as [`Provisional`](RecordState::Provisional)
/// and becomes selectable only after the oracle, resource, and block gates pass.
/// Rejections are kept, with the reason, so a campaign does not rediscover the
/// same dead end every time it runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum RecordState {
    /// Measured, not yet qualified. Never selectable.
    Provisional,
    /// Passed every gate. The only selectable state.
    Qualified,
    /// Measured and refused. Retained so the negative result is not relearned.
    Rejected { reason: String },
    /// Was qualified; an input digest has since moved.
    Stale { changed: Vec<String> },
}

impl RecordState {
    pub fn is_selectable(&self) -> bool {
        matches!(self, RecordState::Qualified)
    }
}

/// Outcome of checking a measured kernel against its correctness oracle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Correctness {
    /// Matched the oracle within tolerance.
    Pass,
    /// Did not match. A fast wrong kernel is not a candidate.
    Fail { detail: String },
    /// Not yet checked. Blocks qualification.
    Unchecked,
}

/// One timed kernel on one hardware/toolchain/profile combination.
///
/// This is the `kernel_measurement` entity: the finest-grained thing the
/// database stores.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelMeasurement {
    /// Op case this served, by content hash — see [`OpCase`].
    pub op_case: String,
    /// Device opcode, as the live ABI numbers it.
    pub kernel_id: u16,
    /// `dev_isa.h` spelling, so a C-side reader can match without a table.
    pub kernel_name: String,
    /// Interpreter profile the kernel ran in.
    pub profile: String,
    /// Hardware key, as `HardwareFingerprint::tuning_path` renders it.
    pub hardware: String,
    /// Exact SKU, for reports.
    pub sku: String,
    pub digests: Digests,
    pub stats: Stats,
    pub correctness: Correctness,
    /// Registers of the enclosing interpreter object, when probed. Recorded on
    /// the measurement because an isolated kernel win that raises the object's
    /// register pressure is not a win.
    pub registers: Option<u32>,
    pub state: RecordState,
    /// Campaign label, for provenance.
    pub campaign: String,
}

impl KernelMeasurement {
    /// Whether every precondition for promotion holds.
    ///
    /// Kept separate from the act of promoting so the reasons can be reported.
    pub fn qualification_blockers(&self) -> Vec<String> {
        blockers_for(&self.correctness, self.stats.samples)
    }
}

/// The two gates every record kind shares: correct before fast, and never on a
/// sample too small to carry dispersion.
///
/// Shared so a second record type cannot quietly qualify on a weaker rule than
/// the first — which is the only way this invariant ever erodes.
pub fn blockers_for(correctness: &Correctness, samples: usize) -> Vec<String> {
    let mut out = Vec::new();
    match correctness {
        Correctness::Pass => {}
        Correctness::Fail { detail } => out.push(format!("correctness failed: {detail}")),
        Correctness::Unchecked => out.push("correctness not checked".into()),
    }
    if samples < Stats::MIN_SAMPLES {
        out.push(format!(
            "{} samples is below the {} minimum",
            samples,
            Stats::MIN_SAMPLES
        ));
    }
    out
}

/// A deduplicated op instance the tuner was asked to serve. Identified by
/// content hash so two structurally identical layers are measured once.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpCase {
    pub hash: String,
    pub semantic: String,
    pub phase: String,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub quant: String,
}

/// A canonicalized network block, and how much it matters.
///
/// `occurrences` is why this exists: a block repeated 60 times and a block that
/// runs once are not equally worth optimizing, and the extraction objective
/// multiplies by it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockDefinition {
    pub hash: String,
    pub model_family: String,
    pub block_kind: String,
    /// Structurally equivalent layer indices folded into this definition.
    pub layer_ids: Vec<u32>,
    /// Execution weight — normally `layer_ids.len()`.
    pub occurrences: u32,
    /// Op-case hashes this block contains.
    pub op_cases: Vec<String>,
}

/// End-to-end timing of a whole block with a chosen kernel set.
///
/// Required because op-level wins are not additive: fusion, the shared-memory
/// union, register pressure, packet gates, and state traffic all interact. A
/// kernel that wins in isolation and loses here loses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockMeasurement {
    pub block: String,
    pub hardware: String,
    pub profile: String,
    pub digests: Digests,
    pub stats: Stats,
    /// Kernel ids selected for this run.
    pub kernels: Vec<u16>,
    pub peak_bytes: Option<u64>,
    pub state: RecordState,
    pub campaign: String,
}

/// The decision a compile may consume, with the provenance to reproduce it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Selection {
    pub op_case: String,
    pub hardware: String,
    pub profile: String,
    pub objective: String,
    pub kernel_id: u16,
    pub kernel_name: String,
    /// Ordered fallbacks, best first.
    pub fallbacks: Vec<u16>,
    /// Calibration tier this decision actually rests on.
    pub tier: String,
    pub digests: Digests,
    pub campaign: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digests() -> Digests {
        Digests {
            implementation: "impl-a".into(),
            interpreter: "interp-a".into(),
            toolchain: "cuda-13.0".into(),
            oracle: "oracle-v1".into(),
        }
    }

    fn measurement(correctness: Correctness) -> KernelMeasurement {
        KernelMeasurement {
            op_case: "case-1".into(),
            kernel_id: 8,
            kernel_name: "PLOW_DOP_GEMM".into(),
            profile: "prefill_dense".into(),
            hardware: "nvidia/sm_90a/h100-nvl".into(),
            sku: "H100 NVL".into(),
            digests: digests(),
            stats: Stats::from_samples(vec![10.0, 11.0, 12.0, 13.0, 14.0]).unwrap(),
            correctness,
            registers: Some(208),
            state: RecordState::Provisional,
            campaign: "c1".into(),
        }
    }

    #[test]
    fn only_qualified_records_are_selectable() {
        assert!(RecordState::Qualified.is_selectable());
        assert!(!RecordState::Provisional.is_selectable());
        assert!(!RecordState::Rejected { reason: "slow".into() }.is_selectable());
        assert!(!RecordState::Stale { changed: vec!["toolchain".into()] }.is_selectable());
    }

    /// A fast kernel that fails its oracle must never qualify. This is the
    /// check that keeps "fastest" from quietly meaning "wrong".
    #[test]
    fn an_incorrect_kernel_cannot_qualify() {
        let m = measurement(Correctness::Fail { detail: "max abs err 0.3".into() });
        let blockers = m.qualification_blockers();
        assert_eq!(blockers.len(), 1);
        assert!(blockers[0].contains("correctness failed"));
    }

    #[test]
    fn an_unchecked_kernel_cannot_qualify() {
        assert_eq!(measurement(Correctness::Unchecked).qualification_blockers().len(), 1);
    }

    #[test]
    fn a_correct_well_sampled_kernel_has_no_blockers() {
        assert!(measurement(Correctness::Pass).qualification_blockers().is_empty());
    }

    /// Staleness names what moved, so an unrelated campaign is not discarded.
    #[test]
    fn staleness_is_per_digest_and_specific() {
        let have = digests();
        let mut want = digests();
        assert!(have.stale_against(&want).is_empty());

        want.toolchain = "cuda-12.4".into();
        assert_eq!(have.stale_against(&want), vec!["toolchain"]);

        want.implementation = "impl-b".into();
        let changed = have.stale_against(&want);
        assert!(changed.contains(&"toolchain") && changed.contains(&"implementation"));
        assert_eq!(changed.len(), 2, "unrelated digests stay current");
    }

    #[test]
    fn records_round_trip_through_json() {
        let m = measurement(Correctness::Pass);
        let text = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<KernelMeasurement>(&text).unwrap(), m);

        let rejected = RecordState::Rejected { reason: "alias of PLOW_DOP_GEMM".into() };
        let text = serde_json::to_string(&rejected).unwrap();
        assert_eq!(serde_json::from_str::<RecordState>(&text).unwrap(), rejected);
    }
}

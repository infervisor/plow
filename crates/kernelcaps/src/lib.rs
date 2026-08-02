//! Kernel capability registry: what the runtime can *actually execute*.
//!
//! The compiler currently picks kernels in two unrelated ways. The generic path
//! synthesizes a `TileShape` from the analytical cost model and passes
//! `bm/bn/bk` as packet fields. The model emitters call a per-model `pick_tile`
//! that returns a [`DevOp`] whose tile is baked into the opcode. Neither asks
//! whether the interpreter it is compiling for contains that kernel.
//!
//! It usually does, so this mostly works. Where it does not, the failure is
//! quiet: AMD's dispatch default no-ops an unknown opcode
//! (`runtime/amd/interp.hip:785`) where NVIDIA traps
//! (`runtime/nvidia/interp_sm120.cu:1214`), so a kernel the build did not
//! instantiate reads as a slightly wrong answer rather than a crash.
//!
//! This crate is the single place that says which kernels exist, for which
//! hardware, in which interpreter profile, and at what resource cost.
//!
//! # Two things it is built to prevent
//!
//! **Selecting a kernel that is not there.** [`Inventory::candidates`] filters by
//! capability predicate before anything is ranked, so a candidate list can only
//! ever contain kernels the target interpreter dispatches.
//!
//! **Selecting between aliases.** On NVIDIA, `PLOW_DOP_GEMM`, `..._MED` and
//! `..._SMALL` all fall through to one `d_gemm` body
//! (`interp_sm120.cu:524`, comment: "one body, three tile opcodes"); the tile is
//! a compile-time macro. On AMD the same three opcodes reach three genuinely
//! distinct instantiations. A selector that ranks three aliases is performing
//! arithmetic, not tuning, and [`Inventory::alias_groups`] makes that visible
//! instead of leaving it to be rediscovered from a flat benchmark.

pub mod build;
pub mod probe;
pub mod resource;
pub mod select;
pub mod spec;
pub mod sweep;
pub mod targets;

// Re-exported so a consumer of the registry does not need a separate `hwspec`
// dependency just to name the hardware it is selecting for.
pub use build::{BuildId, Provenance};
pub use hwspec::{CalibrationTier, HardwareFingerprint, IsaLevel};
pub use probe::{probe, ProbeError, ProbeTarget, ProbedObject};
pub use resource::{GateVerdict, ResourceEnvelope, ResourceGate};
pub use select::{
    select_kernel, MeasuredCosts, NoMeasurements, Rationale, Realization, SelectError,
};
pub use spec::{
    Determinism, KernelId, KernelSpec, OpSignature, Phase, ProfileId, QuantScheme, SemanticOp,
    ShapeClass, ShapeSignature, TileConfig,
};
pub use sweep::{classify as classify_macro, knobs, Knob, Sweepable};
pub use targets::{dense_gemm_inventory, prefill_recipe, ObjectRecipe};

use std::collections::BTreeMap;

/// The set of kernels one build of the runtime can execute.
///
/// A registry describes *a build*, not the source tree: the same
/// `interp_sm120.cu` yields eight different objects under different `-D` flags
/// (`runtime/CMakeLists.txt:127-320`), and they do not have the same
/// capabilities. Registries are therefore expected to be produced by probing a
/// built artifact, not by reading source.
#[derive(Clone, Debug)]
pub struct Inventory {
    provenance: Provenance,
    specs: Vec<KernelSpec>,
}

impl Inventory {
    /// Build an inventory for an object that has been probed.
    ///
    /// There is no constructor that omits provenance. An inventory whose origin
    /// is unknown is indistinguishable from a hand-written claim, and that is
    /// the failure this crate exists to prevent.
    pub fn probed(build: BuildId, specs: impl IntoIterator<Item = KernelSpec>) -> Self {
        Inventory {
            provenance: Provenance::Probed(build),
            specs: specs.into_iter().collect(),
        }
    }

    /// The build this inventory describes.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    pub fn build(&self) -> &BuildId {
        self.provenance.build()
    }

    /// Whether this inventory still describes `build`, and if not, what moved.
    /// Empty means current.
    pub fn staleness(&self, build: &BuildId) -> Vec<String> {
        self.build().differences(build)
    }

    pub fn insert(&mut self, spec: KernelSpec) {
        self.specs.push(spec);
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &KernelSpec> {
        self.specs.iter()
    }

    /// Every kernel that can legally run `op` on `hw` within `profile`.
    ///
    /// This is a hard filter, not a ranking. Anything it rejects must never
    /// reach a cost model, because pricing a kernel that does not exist is how
    /// a plan ends up naming one.
    pub fn candidates(
        &self,
        op: &OpSignature,
        hw: &HardwareFingerprint,
        profile: ProfileId,
    ) -> Vec<&KernelSpec> {
        self.specs
            .iter()
            .filter(|k| k.profile == profile)
            .filter(|k| k.runs_on(hw))
            .filter(|k| k.accepts(op))
            .collect()
    }

    /// Kernels grouped by implementation hash, keeping only groups with more
    /// than one member — i.e. distinct opcodes that reach the same body.
    ///
    /// The tuning pipeline must refuse to report a winner *between* members of
    /// one group: any difference measured is noise plus dispatch, and promoting
    /// it produces a tuning record that will not reproduce.
    pub fn alias_groups(&self) -> BTreeMap<&str, Vec<&KernelSpec>> {
        let mut by_hash: BTreeMap<&str, Vec<&KernelSpec>> = BTreeMap::new();
        for k in &self.specs {
            by_hash
                .entry(k.implementation_hash.as_str())
                .or_default()
                .push(k);
        }
        by_hash.retain(|_, v| v.len() > 1);
        by_hash
    }

    /// Whether these two kernels are the same code reached by different opcodes.
    pub fn are_aliases(&self, a: KernelId, b: KernelId) -> bool {
        if a == b {
            return false;
        }
        let find = |id: KernelId| self.specs.iter().find(|k| k.id == id);
        match (find(a), find(b)) {
            (Some(x), Some(y)) => x.implementation_hash == y.implementation_hash,
            _ => false,
        }
    }

    /// Declared kernels whose dispatch arm is absent from this build.
    ///
    /// Not an error on its own — an opcode can be reserved ahead of its
    /// implementation, which several are. It is an error to *select* one, so
    /// the tuner reports these rather than silently dropping them.
    pub fn declared_but_absent(&self) -> Vec<&KernelSpec> {
        self.specs.iter().filter(|k| !k.dispatched).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hwspec::registry as hwreg;
    use packet::dev::DevOp;

    fn h100() -> HardwareFingerprint {
        HardwareFingerprint::from_spec(hwreg::lookup("H100 NVL").unwrap()).unwrap()
    }

    fn mi350() -> HardwareFingerprint {
        HardwareFingerprint::from_spec(hwreg::lookup("MI350X").unwrap()).unwrap()
    }

    fn gemm(m: i64, n: i64, k: i64) -> OpSignature {
        OpSignature::gemm(Phase::Prefill, m, n, k)
    }

    /// The NVIDIA reality: three opcodes, one body. A registry built from that
    /// build must report them as aliases, not as three candidates.
    #[test]
    fn nvidia_gemm_tile_opcodes_are_reported_as_aliases() {
        let reg = Inventory::probed(
            test_build(hwspec::IsaLevel::Sm90a),
            [
                KernelSpec::gemm_tile(
                    DevOp::Gemm,
                    hwspec::IsaLevel::Sm90a,
                    128,
                    128,
                    32,
                    "d_gemm@nv",
                ),
                KernelSpec::gemm_tile(
                    DevOp::GemmMed,
                    hwspec::IsaLevel::Sm90a,
                    128,
                    128,
                    32,
                    "d_gemm@nv",
                ),
                KernelSpec::gemm_tile(
                    DevOp::GemmSmall,
                    hwspec::IsaLevel::Sm90a,
                    128,
                    128,
                    32,
                    "d_gemm@nv",
                ),
            ],
        );

        let groups = reg.alias_groups();
        assert_eq!(groups.len(), 1, "one shared body");
        assert_eq!(groups["d_gemm@nv"].len(), 3, "all three opcodes reach it");
        assert!(reg.are_aliases(KernelId(DevOp::Gemm), KernelId(DevOp::GemmMed)));
        assert!(reg.are_aliases(KernelId(DevOp::Gemm), KernelId(DevOp::GemmSmall)));
    }

    /// The AMD reality: the same three opcodes are three real instantiations,
    /// so they must NOT be reported as aliases.
    #[test]
    fn amd_gemm_tile_opcodes_are_distinct_kernels() {
        let reg = Inventory::probed(
            test_build(hwspec::IsaLevel::Sm90a),
            [
                KernelSpec::gemm_tile(
                    DevOp::Gemm,
                    hwspec::IsaLevel::Gfx950,
                    256,
                    256,
                    64,
                    "exec_gemm",
                ),
                KernelSpec::gemm_tile(
                    DevOp::GemmMed,
                    hwspec::IsaLevel::Gfx950,
                    128,
                    128,
                    64,
                    "exec_gemm_med",
                ),
                KernelSpec::gemm_tile(
                    DevOp::GemmSmall,
                    hwspec::IsaLevel::Gfx950,
                    64,
                    128,
                    64,
                    "exec_gemm_small",
                ),
            ],
        );

        assert!(
            reg.alias_groups().is_empty(),
            "three distinct bodies, no aliasing"
        );
        assert!(!reg.are_aliases(KernelId(DevOp::Gemm), KernelId(DevOp::GemmMed)));
    }

    /// A kernel built for gfx950 must never be offered for an H100, however
    /// well its analytical cost scores. This is the filter that makes the
    /// "compiler candidates == executable kernels" property hold.
    #[test]
    fn a_kernel_for_another_isa_is_never_a_candidate() {
        let reg = Inventory::probed(
            test_build(hwspec::IsaLevel::Sm90a),
            [KernelSpec::gemm_tile(
                DevOp::Gemm,
                hwspec::IsaLevel::Gfx950,
                256,
                256,
                64,
                "exec_gemm",
            )],
        );

        assert!(reg
            .candidates(&gemm(4096, 4096, 4096), &h100(), ProfileId::PrefillDense)
            .is_empty());
        assert_eq!(
            reg.candidates(&gemm(4096, 4096, 4096), &mi350(), ProfileId::PrefillDense)
                .len(),
            1
        );
    }

    /// Wrong profile is as disqualifying as wrong hardware: a prefill-only
    /// tiled GEMM is not linked into the decode object, so selecting it there
    /// would emit an opcode the decode interpreter does not dispatch.
    #[test]
    fn profile_is_part_of_the_capability_filter() {
        let reg = Inventory::probed(
            test_build(hwspec::IsaLevel::Sm90a),
            [KernelSpec::gemm_tile(
                DevOp::Gemm,
                hwspec::IsaLevel::Sm90a,
                128,
                128,
                32,
                "d_gemm@nv",
            )],
        );
        assert_eq!(
            reg.candidates(&gemm(4096, 4096, 4096), &h100(), ProfileId::PrefillDense)
                .len(),
            1
        );
        assert!(reg
            .candidates(&gemm(1, 4096, 4096), &h100(), ProfileId::DecodeDense)
            .is_empty());
    }

    /// A declared-but-undispatched opcode is reported, never silently selected.
    /// `FLASH_MLA_PREFILL` and `FLASH_GATHER_PREFILL` are exactly this today.
    #[test]
    fn reserved_opcodes_are_visible_but_not_selectable() {
        let mut absent = KernelSpec::gemm_tile(
            DevOp::FlashMlaPrefill,
            hwspec::IsaLevel::Sm90a,
            128,
            128,
            32,
            "reserved",
        );
        absent.dispatched = false;
        absent.semantic = SemanticOp::LatentAttention;

        let reg = Inventory::probed(test_build(hwspec::IsaLevel::Sm90a), [absent]);
        assert_eq!(reg.declared_but_absent().len(), 1);

        let op = OpSignature {
            semantic: SemanticOp::LatentAttention,
            ..gemm(4096, 4096, 4096)
        };
        assert!(
            reg.candidates(&op, &h100(), ProfileId::PrefillDense)
                .is_empty(),
            "an opcode with no dispatch arm must never be a candidate"
        );
    }

    #[test]
    fn kernel_ids_come_from_the_live_abi() {
        // Not a fresh numbering: the id *is* the device opcode, so it cannot
        // drift from the ABI that `packet::tests::dev_opcodes` locks.
        assert_eq!(KernelId(DevOp::GemmMed).raw(), DevOp::GemmMed as u16);
        assert_eq!(KernelId(DevOp::GemmMed).raw(), 15);
    }
}

/// A build identity for tests. Tests construct inventories from fixtures
/// rather than probing a real object; the fixture is a test *input*, never
/// shipped data, so it does not reintroduce a hand-written authority.
#[cfg(test)]
pub(crate) fn test_build(isa: IsaLevel) -> BuildId {
    BuildId::new(isa, ["PLOW_TEST=1".to_string()], "test", "test")
}

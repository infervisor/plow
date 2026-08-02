//! Kernel selection: registry filter, then rank.
//!
//! This replaces per-model `pick_tile` functions. Two of those exist today
//! (`plowc/src/bin/gemma4.rs:479`, `plowc/src/bin/llama3.rs:78`), they disagree
//! with each other, and the Gemma one resolves `hwspec::registry::lookup(
//! "MI350X")` unconditionally — so compiling Gemma for an H100 ranks tiles using
//! an AMD cost model. It then returns one of three opcodes that, on NVIDIA, all
//! dispatch to the same `d_gemm` body.
//!
//! The order here follows the tuning architecture's selection pipeline:
//!
//! 1. ask the registry what is executable — never rank a kernel that is absent;
//! 2. collapse aliases, so a group of opcodes sharing one body is reported as
//!    one choice rather than ranked against itself;
//! 3. use a measurement when one matches this exact hardware and build;
//! 4. otherwise fall back to the analytical model, and say that is what
//!    happened.
//!
//! The analytical ranker is supplied by the caller rather than wired in. That
//! keeps this crate independent of `costmodel`, and it lets the existing
//! wall-clock tile model be passed in verbatim, so replacing `pick_tile` with
//! `select_kernel` is provably not a behaviour change on the hardware
//! `pick_tile` was written for.

use hwspec::{CalibrationTier, HardwareFingerprint};

use crate::spec::{KernelId, KernelSpec, OpSignature, ProfileId, TileConfig};
use crate::Inventory;

/// Why a kernel was chosen. Carried into the bundle so a selection can be
/// explained without re-running the compiler.
#[derive(Clone, Debug, PartialEq)]
pub enum Rationale {
    /// Chosen on a measurement taken on matching hardware and build.
    Measured { median_ns: f64 },
    /// Chosen by the analytical model; no matching measurement existed.
    Analytical { cost: u64 },
    /// Exactly one kernel was legal. Ranking would have been theatre.
    OnlyCandidate,
    /// Every legal candidate shares one implementation, so the "choice" is
    /// between names for the same code. The canonical opcode is returned and
    /// the group size recorded.
    ///
    /// This is the NVIDIA GEMM triple. Reporting it is the point: a tuning
    /// campaign that ranks these three is measuring dispatch noise, and a
    /// database that stores the winner will not reproduce it.
    AliasCollapsed { members: usize },
}

impl Rationale {
    pub fn tier(&self) -> CalibrationTier {
        match self {
            Rationale::Measured { .. } => CalibrationTier::SkuCalibrated,
            _ => CalibrationTier::Portable,
        }
    }
}

/// A selected kernel and the evidence behind it.
#[derive(Clone, Debug, PartialEq)]
pub struct Realization {
    pub kernel: KernelId,
    pub tile: Option<TileConfig>,
    pub rationale: Rationale,
    /// Legal alternatives, best first, for fallback and for the record.
    pub fallbacks: Vec<KernelId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectError {
    /// Nothing in this build can run the op. A hard error: the alternative is
    /// emitting an opcode the interpreter does not dispatch, which AMD silently
    /// no-ops.
    NoCandidate {
        op: String,
        hardware: String,
        profile: String,
    },
}

impl std::fmt::Display for SelectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectError::NoCandidate {
                op,
                hardware,
                profile,
            } => write!(
                f,
                "no kernel in the {profile} profile on {hardware} can run {op}; \
                 compiling one anyway would emit an opcode with no dispatch arm"
            ),
        }
    }
}

impl std::error::Error for SelectError {}

/// A measured latency for one kernel, keyed by opcode.
pub trait MeasuredCosts {
    /// Median nanoseconds for this kernel on this op, if a current measurement
    /// exists for the hardware and build being compiled for.
    fn median_ns(&self, kernel: KernelId) -> Option<f64>;
}

/// No measurements available — the cold-start case.
pub struct NoMeasurements;

impl MeasuredCosts for NoMeasurements {
    fn median_ns(&self, _: KernelId) -> Option<f64> {
        None
    }
}

/// Select the kernel to emit for `op`.
///
/// `analytical` ranks a candidate in arbitrary units where lower is better; it
/// is only consulted when no measurement applies.
pub fn select_kernel(
    reg: &Inventory,
    op: &OpSignature,
    hw: &HardwareFingerprint,
    profile: ProfileId,
    measured: &dyn MeasuredCosts,
    analytical: impl Fn(&KernelSpec) -> u64,
) -> Result<Realization, SelectError> {
    let candidates = reg.candidates(op, hw, profile);
    if candidates.is_empty() {
        return Err(SelectError::NoCandidate {
            op: format!(
                "{:?} {:?} {}x{}x{}",
                op.semantic, op.phase, op.shape.m, op.shape.n, op.shape.k
            ),
            hardware: hw.tuning_path(),
            profile: profile.label().to_string(),
        });
    }

    // Step 2. If every legal candidate is the same code, there is nothing to
    // rank. Pick the lowest opcode as canonical so the choice is stable, and
    // record that the alternatives were names rather than kernels.
    let first_hash = candidates[0].implementation_hash.as_str();
    if candidates.len() > 1
        && candidates
            .iter()
            .all(|k| k.implementation_hash == first_hash)
    {
        let mut ids: Vec<KernelId> = candidates.iter().map(|k| k.id).collect();
        ids.sort();
        let canonical = ids[0];
        let tile = candidates
            .iter()
            .find(|k| k.id == canonical)
            .and_then(|k| k.tile);
        return Ok(Realization {
            kernel: canonical,
            tile,
            rationale: Rationale::AliasCollapsed {
                members: candidates.len(),
            },
            fallbacks: ids.into_iter().skip(1).collect(),
        });
    }

    if candidates.len() == 1 {
        let k = candidates[0];
        return Ok(Realization {
            kernel: k.id,
            tile: k.tile,
            rationale: Rationale::OnlyCandidate,
            fallbacks: Vec::new(),
        });
    }

    // Step 3. Measurements win where they exist, but only if every candidate has
    // one. Ranking a measured candidate against an estimated one compares two
    // different units and reliably prefers whichever scale happens to be
    // smaller.
    let all_measured: Option<Vec<(KernelId, f64, Option<TileConfig>)>> = candidates
        .iter()
        .map(|k| measured.median_ns(k.id).map(|ns| (k.id, ns, k.tile)))
        .collect();

    if let Some(mut scored) = all_measured {
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite medians"));
        let (kernel, median_ns, tile) = scored[0];
        return Ok(Realization {
            kernel,
            tile,
            rationale: Rationale::Measured { median_ns },
            fallbacks: scored.into_iter().skip(1).map(|(id, _, _)| id).collect(),
        });
    }

    // Step 4. Cold start.
    let mut scored: Vec<(KernelId, u64, Option<TileConfig>)> = candidates
        .iter()
        .map(|k| (k.id, analytical(k), k.tile))
        .collect();
    // Stable on ties: equal cost must not depend on registry insertion order,
    // or the same compile produces different bundles.
    scored.sort_by_key(|(id, cost, _)| (*cost, id.raw()));
    let (kernel, cost, tile) = scored[0];
    Ok(Realization {
        kernel,
        tile,
        rationale: Rationale::Analytical { cost },
        fallbacks: scored.into_iter().skip(1).map(|(id, _, _)| id).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{Phase, SemanticOp};
    use hwspec::registry as hwreg;
    use packet::dev::DevOp;
    use std::collections::BTreeMap;

    struct Table(BTreeMap<u16, f64>);

    impl MeasuredCosts for Table {
        fn median_ns(&self, k: KernelId) -> Option<f64> {
            self.0.get(&k.raw()).copied()
        }
    }

    fn fp(name: &str) -> HardwareFingerprint {
        HardwareFingerprint::from_spec(hwreg::lookup(name).unwrap()).unwrap()
    }

    fn amd_registry() -> Inventory {
        Inventory::probed(
            crate::test_build(hwspec::IsaLevel::Gfx950),
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
        )
    }

    fn nvidia_registry() -> Inventory {
        // All three opcodes, one body -- what interp_sm120.cu actually does.
        Inventory::probed(
            crate::test_build(hwspec::IsaLevel::Gfx950),
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
        )
    }

    fn op() -> OpSignature {
        OpSignature::gemm(Phase::Prefill, 4096, 4096, 4096)
    }

    /// On NVIDIA the three tile opcodes are one kernel, so selection must say so
    /// instead of returning a "winner". A campaign that ranks these is measuring
    /// dispatch noise.
    #[test]
    fn nvidia_tile_opcodes_collapse_instead_of_ranking() {
        let r = select_kernel(
            &nvidia_registry(),
            &op(),
            &fp("H100 NVL"),
            ProfileId::PrefillDense,
            &NoMeasurements,
            |_| 1,
        )
        .unwrap();

        assert_eq!(r.rationale, Rationale::AliasCollapsed { members: 3 });
        assert_eq!(
            r.kernel,
            KernelId(DevOp::Gemm),
            "lowest opcode is canonical"
        );
        assert_eq!(r.fallbacks.len(), 2);
    }

    /// On AMD the same opcodes are distinct kernels, so the analytical model is
    /// consulted and a real choice is made.
    #[test]
    fn amd_tile_opcodes_are_ranked_for_real() {
        let prefer_med = |k: &KernelSpec| {
            if k.id == KernelId(DevOp::GemmMed) {
                1
            } else {
                9
            }
        };
        let r = select_kernel(
            &amd_registry(),
            &op(),
            &fp("MI350X"),
            ProfileId::PrefillDense,
            &NoMeasurements,
            prefer_med,
        )
        .unwrap();

        assert_eq!(r.kernel, KernelId(DevOp::GemmMed));
        assert_eq!(r.rationale, Rationale::Analytical { cost: 1 });
        assert_eq!(r.fallbacks.len(), 2, "the losers are retained as fallbacks");
    }

    /// A measurement overrides the analytical model.
    #[test]
    fn measurements_win_over_estimates() {
        let mut t = BTreeMap::new();
        t.insert(DevOp::Gemm as u16, 300.0);
        t.insert(DevOp::GemmMed as u16, 100.0);
        t.insert(DevOp::GemmSmall as u16, 200.0);

        let r = select_kernel(
            &amd_registry(),
            &op(),
            &fp("MI350X"),
            ProfileId::PrefillDense,
            &Table(t),
            // The analytical model insists on the big tile; the measurement
            // disagrees and must win.
            |k| {
                if k.id == KernelId(DevOp::Gemm) {
                    0
                } else {
                    100
                }
            },
        )
        .unwrap();

        assert_eq!(r.kernel, KernelId(DevOp::GemmMed));
        assert_eq!(r.rationale, Rationale::Measured { median_ns: 100.0 });
        assert_eq!(r.rationale.tier(), CalibrationTier::SkuCalibrated);
    }

    /// Partial measurement coverage must not mix units. If only some candidates
    /// were measured, ranking nanoseconds against model cycles would pick
    /// whichever scale is numerically smaller, which is not a decision.
    #[test]
    fn partial_measurements_fall_back_rather_than_mix_units() {
        let mut t = BTreeMap::new();
        t.insert(DevOp::GemmMed as u16, 1.0); // nanoseconds
        let r = select_kernel(
            &amd_registry(),
            &op(),
            &fp("MI350X"),
            ProfileId::PrefillDense,
            &Table(t),
            |k| {
                if k.id == KernelId(DevOp::Gemm) {
                    5
                } else {
                    900
                }
            },
        )
        .unwrap();

        assert_eq!(r.kernel, KernelId(DevOp::Gemm), "analytical model decided");
        assert!(matches!(r.rationale, Rationale::Analytical { .. }));
        assert_eq!(r.rationale.tier(), CalibrationTier::Portable);
    }

    /// The failure that matters: rather than emitting a kernel the interpreter
    /// cannot dispatch, selection fails loudly.
    #[test]
    fn no_candidate_is_an_error_not_a_guess() {
        let err = select_kernel(
            &amd_registry(),
            &op(),
            &fp("H100 NVL"), // AMD kernels, NVIDIA target
            ProfileId::PrefillDense,
            &NoMeasurements,
            |_| 1,
        )
        .unwrap_err();

        assert!(matches!(err, SelectError::NoCandidate { .. }));
        assert!(err.to_string().contains("no dispatch arm"));
    }

    /// Equal analytical cost must resolve deterministically, or two identical
    /// compiles emit different bundles.
    #[test]
    fn ties_break_deterministically() {
        let pick = || {
            select_kernel(
                &amd_registry(),
                &op(),
                &fp("MI350X"),
                ProfileId::PrefillDense,
                &NoMeasurements,
                |_| 42,
            )
            .unwrap()
            .kernel
        };
        assert_eq!(pick(), pick());
        assert_eq!(pick(), KernelId(DevOp::Gemm), "lowest opcode wins a tie");
    }

    #[test]
    fn a_single_legal_candidate_is_reported_as_such() {
        let reg = Inventory::probed(
            crate::test_build(hwspec::IsaLevel::Gfx950),
            [KernelSpec::gemm_tile(
                DevOp::Gemm,
                hwspec::IsaLevel::Gfx950,
                256,
                256,
                64,
                "exec_gemm",
            )],
        );
        let r = select_kernel(
            &reg,
            &op(),
            &fp("MI350X"),
            ProfileId::PrefillDense,
            &NoMeasurements,
            |_| 1,
        )
        .unwrap();
        assert_eq!(r.rationale, Rationale::OnlyCandidate);
        assert!(r.fallbacks.is_empty());
    }

    #[test]
    fn an_op_the_build_cannot_serve_fails_even_with_kernels_present() {
        let mut latent = op();
        latent.semantic = SemanticOp::LatentAttention;
        assert!(select_kernel(
            &amd_registry(),
            &latent,
            &fp("MI350X"),
            ProfileId::PrefillDense,
            &NoMeasurements,
            |_| 1
        )
        .is_err());
    }
}

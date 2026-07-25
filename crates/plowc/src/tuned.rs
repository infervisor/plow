//! The compiler's kernel oracle: capability filter plus measured costs.
//!
//! This is what connects the tuning system to a real build. Without it,
//! `plowc::compile` prices tiles the analytical model synthesized and never asks
//! whether the target interpreter contains them, nor whether anyone has measured
//! them.
//!
//! Two answers, both refusable:
//!
//! * **capability** — a tile the probed inventory does not carry is dropped
//!   before pricing, so it cannot win an argmin. When no inventory can be probed
//!   (no vendor toolchain), the filter is disabled rather than guessed: rejecting
//!   everything would fail every build, and admitting a guess is what the
//!   inventory exists to prevent. The report says which happened.
//! * **cost** — a qualified measurement matching this hardware *and* this build
//!   replaces the estimate. All-or-nothing per choice point, because ns and
//!   cycles are different scales.
//!
//! Selection stays deterministic and offline. Nothing here reads the GPU.

use std::path::PathBuf;

use costmodel::tile::TileShape;
use kernelcaps::{HardwareFingerprint, Inventory};
use rewrite::oracle::{GemmQuery, KernelOracle, TileAdvice};
use tunedb::{Digests, TuneStore};

/// How much evidence the oracle actually has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleEvidence {
    /// Tuning was switched off; the analytical model decides everything.
    Disabled,
    /// No inventory could be probed and no measurements applied.
    None { why: String },
    /// Capability filtering only — the inventory was probed, nothing measured.
    Capability { build: String, kernels: usize },
    /// Capability filtering plus measurements.
    Measured { build: String, kernels: usize, records: usize },
}

impl OracleEvidence {
    pub fn tier(&self) -> &'static str {
        match self {
            OracleEvidence::Measured { .. } => "sku-calibrated",
            _ => "portable",
        }
    }

    pub fn describe(&self) -> String {
        match self {
            OracleEvidence::Disabled => "tuning disabled (--no-tuning)".into(),
            OracleEvidence::None { why } => format!("analytical only: {why}"),
            OracleEvidence::Capability { build, kernels } => {
                format!("capability filter from probed build {build} ({kernels} kernels)")
            }
            OracleEvidence::Measured { build, kernels, records } => format!(
                "probed build {build} ({kernels} kernels), {records} qualified measurement(s)"
            ),
        }
    }
}

/// Capability + measurement oracle for one target.
pub struct CompilerOracle {
    inventory: Option<Inventory>,
    /// Weight-layout conflicts: the shared (BN, BK) excluded every buildable
    /// tile. Counted, reported, not fatal.
    mismatches: std::cell::RefCell<Vec<String>>,
    /// Operand formats the probed object does not carry, so the choice was
    /// analytical and unverified.
    unverified: std::cell::RefCell<std::collections::BTreeSet<&'static str>>,
    /// Median ns per case key (shape + variant + tile), from qualified records
    /// matching this build.
    measured: std::collections::HashMap<String, u64>,
    evidence: OracleEvidence,
}

impl CompilerOracle {
    /// Build an oracle for `hw`, probing the interpreter under `root` and
    /// reading qualified records from `db`.
    ///
    /// Never fails: a target whose toolchain is absent yields an oracle that
    /// admits everything and says so. That is the honest degradation — the
    /// alternative is failing builds that used to work on machines that cannot
    /// probe.
    pub fn new(root: &std::path::Path, hw: &HardwareFingerprint, db: Option<&PathBuf>) -> Self {
        let inventory = match kernelcaps::dense_gemm_inventory(root, hw.isa) {
            Ok(inv) => Some(inv),
            Err(e) => {
                return CompilerOracle {
                    inventory: None,
                    mismatches: Default::default(),
                    unverified: Default::default(),
                    measured: Default::default(),
                    evidence: OracleEvidence::None { why: e.to_string() },
                }
            }
        };
        let inv = inventory.as_ref().expect("just set");
        let build = inv.build().label();
        let kernels = inv.len();

        // Measurements must match this BUILD, not merely this hardware. The
        // digests carry the identity, and -- critically -- the toolchain comes
        // from the probed build, not from `hw.toolchain` (which `from_spec`
        // leaves `None`). Keying it "unknown" was a silent bug: a campaign files
        // records under the real toolchain, so every one of them read stale.
        let mut measured = std::collections::HashMap::new();
        let mut records = 0usize;
        if let Some(dbp) = db {
            let store = TuneStore::new(dbp.clone());
            let want = build_digests(inv.build());
            if let Ok((best, _stale)) = store.best_for(&hw.tuning_path(), &want) {
                for (case, rec) in best {
                    // The case key already carries shape + variant + tile; store
                    // it verbatim so lookup uses the same string the campaign
                    // wrote. A mismatch here means measurements are silently
                    // never found.
                    measured.insert(case, rec.stats.median_ns.round().max(1.0) as u64);
                    records += 1;
                }
            }
        }

        let evidence = if records > 0 {
            OracleEvidence::Measured { build, kernels, records }
        } else {
            OracleEvidence::Capability { build, kernels }
        };
        CompilerOracle {
            inventory,
            mismatches: Default::default(),
            unverified: Default::default(),
            measured,
            evidence,
        }
    }

    /// An oracle that does nothing, for `--no-tuning`.
    pub fn disabled() -> Self {
        CompilerOracle {
            inventory: None,
            mismatches: Default::default(),
            unverified: Default::default(),
            measured: Default::default(),
            evidence: OracleEvidence::Disabled,
        }
    }

    pub fn evidence(&self) -> &OracleEvidence {
        &self.evidence
    }
}

/// The oracle-identity string both the compiler and a campaign must agree on,
/// so a record's `oracle` digest is not treated as stale on lookup.
pub const GEMM_ORACLE: &str = "gemm-cpu-ref-v1";

/// The digests a GEMM measurement for `build` must carry to be selectable.
///
/// Public so a campaign constructs the *same* digests it will be looked up by.
/// implementation and interpreter are both the build content hash: for a
/// probed interpreter object the kernel body and the object are the same
/// artifact, so one digest identifies both.
pub fn build_digests(build: &kernelcaps::BuildId) -> Digests {
    Digests {
        implementation: build.label(),
        interpreter: build.label(),
        toolchain: build.toolchain.clone(),
        oracle: GEMM_ORACLE.to_string(),
    }
}

/// Render an op case key for storage: `m,n,k,variant,bm,bn,bk`.
///
/// Public so a tuning campaign files records under the same key the compiler
/// looks them up by. The **variant** is in the key because a bf16 and an fp8
/// GEMM of the same dimensions emit different kernels
/// (`schedule::emit::gemm_variant_for`); without it a bf16 measurement would be
/// served for an fp8 op, pricing one kernel with a timing of another.
pub fn case_key(q: &GemmQuery, t: TileShape) -> String {
    let g = q.shape;
    format!("{},{},{},{},{},{},{}", g.m, g.n, g.k, q.variant(), t.bm, t.bn, t.bk)
}

/// Whether a probed kernel serves this query's operand format.
///
/// The inventory produced by `dense_gemm_inventory` today is bf16 only, so an
/// fp8 or block-quant query finds nothing and the op is reported unverified
/// rather than judged against a bf16 kernel.
fn kernel_serves(k: &kernelcaps::KernelSpec, q: &GemmQuery) -> bool {
    use kernelcaps::QuantScheme::*;
    match q.variant() {
        "bf16" => k.quant == None,
        "fp8" | "fp8fp8" => matches!(k.quant, W8A16 | W8A8),
        "w4a8" => matches!(k.quant, W4A8 | BlockFp8),
        "fp4" => matches!(k.quant, Fp4 | Mxfp4),
        _ => false,
    }
}

fn to_tileshape(t: kernelcaps::TileConfig) -> TileShape {
    TileShape { bm: t.bm, bn: t.bn, bk: t.bk, split_k: t.split_k }
}

impl KernelOracle for CompilerOracle {
    fn gemm_tiles(&self, q: &GemmQuery) -> TileAdvice {
        // No inventory (disabled, or the probe could not run): no opinion. The
        // analytical model decides, and the report says the choice is
        // unverified -- never a silent claim of capability.
        let Some(inv) = &self.inventory else {
            return TileAdvice::Analytical;
        };
        // The tiles the probed object builds for THIS operand format. A bf16
        // object asked about fp8 yields nothing here -- reported as unverified,
        // not judged against a bf16 kernel.
        let built: Vec<TileShape> = inv
            .iter()
            .filter(|k| kernel_serves(k, q))
            .filter_map(|k| k.tile.map(to_tileshape))
            .collect();
        if built.is_empty() {
            TileAdvice::Unverified
        } else {
            TileAdvice::Buildable(built)
        }
    }

    fn measured_gemm(&self, q: &GemmQuery, tiles: &[TileShape]) -> Option<Vec<u64>> {
        if self.measured.is_empty() || tiles.is_empty() {
            return None;
        }
        // All or nothing: a partial set would mix nanoseconds with cycles.
        tiles.iter().map(|t| self.measured.get(&case_key(q, *t)).copied()).collect()
    }

    fn note_pin_conflict(&self, q: &GemmQuery, buildable: &[TileShape]) {
        let mut m = self.mismatches.borrow_mut();
        if m.len() < 4 {
            let g = q.shape;
            let built = buildable
                .first()
                .map(|t| format!("{}x{}x{}", t.bm, t.bn, t.bk))
                .unwrap_or_else(|| "?".into());
            m.push(format!(
                "{}x{}x{} {}: shared weight layout excludes the built tile {built}",
                g.m, g.n, g.k, q.variant()
            ));
        }
    }

    fn note_unverified(&self, q: &GemmQuery) {
        let mut u = self.unverified.borrow_mut();
        u.insert(q.variant());
    }

    fn tier(&self) -> &'static str {
        self.evidence.tier()
    }

    fn provenance(&self) -> String {
        let mut out = self.evidence.describe();
        let u = self.unverified.borrow();
        if !u.is_empty() {
            let mut v: Vec<&str> = u.iter().copied().collect();
            v.sort();
            out.push_str(&format!(
                "; {} dtype(s) not covered by the probed object, analytical [{}]",
                v.len(),
                v.join(", ")
            ));
        }
        let m = self.mismatches.borrow();
        if !m.is_empty() {
            // On NVIDIA `d_gemm` takes its tile from compile-time macros and
            // ignores the packet's tile fields, so a plan/object disagreement
            // does not fail at run time and nothing said so. This says so.
            out.push_str(&format!(
                "; {} weight-layout conflict(s) [{}]",
                m.len(),
                m.join("; ")
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use costmodel::tile::GemmShape;
    use kernelcaps::{BuildId, IsaLevel, KernelSpec, QuantScheme};
    use packet::dev::DevOp;
    use rewrite::oracle::TileAdvice;

    fn shape() -> GemmShape {
        GemmShape { m: 4096, n: 4096, k: 4096 }
    }
    fn tile(bm: i64, bn: i64, bk: i64) -> TileShape {
        TileShape { bm, bn, bk, split_k: 1 }
    }
    fn bf16(s: GemmShape) -> GemmQuery {
        GemmQuery::bf16(s)
    }
    fn fp8(s: GemmShape) -> GemmQuery {
        GemmQuery { weight_elem: 1, activation_elem: 2, ..GemmQuery::bf16(s) }
    }

    fn build() -> BuildId {
        BuildId::new(IsaLevel::Sm90a, ["X=1".to_string()], "cuda-13.0", "srcdigest")
    }

    /// An oracle whose probed object carries one bf16 128x128x32 kernel.
    fn with_bf16_inventory() -> CompilerOracle {
        CompilerOracle {
            mismatches: Default::default(),
            unverified: Default::default(),
            inventory: Some(Inventory::probed(
                build(),
                [KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm90a, 128, 128, 32, "b")],
            )),
            measured: Default::default(),
            evidence: OracleEvidence::Capability { build: "b".into(), kernels: 1 },
        }
    }

    /// The whole point: the oracle enumerates the tile the build carries, so the
    /// compiler chooses among buildable tiles instead of filtering synthesized
    /// ones and hoping.
    #[test]
    fn buildable_tiles_are_the_probed_ones() {
        let o = with_bf16_inventory();
        match o.gemm_tiles(&bf16(shape())) {
            TileAdvice::Buildable(t) => assert_eq!(t, vec![tile(128, 128, 32)]),
            other => panic!("expected Buildable, got {other:?}"),
        }
    }

    /// The finding the reviewer named: a bf16 inventory must NOT judge an fp8
    /// op. It reports the op unverified rather than admitting or rejecting a
    /// bf16 tile for it.
    #[test]
    fn a_bf16_inventory_does_not_judge_an_fp8_op() {
        let o = with_bf16_inventory();
        assert_eq!(o.gemm_tiles(&fp8(shape())), TileAdvice::Unverified);
        // ...and that fact reaches the report.
        o.note_unverified(&fp8(shape()));
        assert!(o.provenance().contains("fp8"), "{}", o.provenance());
    }

    /// Without a probe the oracle has no opinion -- analytical, so a machine
    /// with no vendor toolchain still builds.
    #[test]
    fn no_inventory_means_analytical() {
        let o = CompilerOracle::disabled();
        assert_eq!(o.gemm_tiles(&bf16(shape())), TileAdvice::Analytical);
        assert_eq!(o.tier(), "portable");
        assert_eq!(o.evidence(), &OracleEvidence::Disabled);
    }

    /// A measurement is keyed by variant, so a bf16 record is never served for
    /// an fp8 op of the same dimensions -- the reviewer's P1 #3.
    #[test]
    fn measurements_are_keyed_by_variant() {
        let mut o = with_bf16_inventory();
        o.measured.insert(case_key(&bf16(shape()), tile(128, 128, 32)), 500);

        assert_eq!(o.measured_gemm(&bf16(shape()), &[tile(128, 128, 32)]), Some(vec![500]));
        assert_eq!(
            o.measured_gemm(&fp8(shape()), &[tile(128, 128, 32)]),
            None,
            "the bf16 record must not answer an fp8 query"
        );
    }

    /// Partial coverage is refused wholesale: mixing measured ns with analytical
    /// cycles picks whichever scale is smaller.
    #[test]
    fn partial_measurements_are_refused_wholesale() {
        let mut o = with_bf16_inventory();
        o.measured.insert(case_key(&bf16(shape()), tile(128, 128, 32)), 500);
        let tiles = [tile(128, 128, 32), tile(256, 256, 64)];
        assert_eq!(o.measured_gemm(&bf16(shape()), &tiles), None, "only one of two");

        o.measured.insert(case_key(&bf16(shape()), tile(256, 256, 64)), 900);
        assert_eq!(o.measured_gemm(&bf16(shape()), &tiles), Some(vec![500, 900]));
    }

    /// A measurement filed under a different shape must not be served.
    #[test]
    fn measurements_are_keyed_by_shape() {
        let mut o = with_bf16_inventory();
        let other = GemmShape { m: 512, n: 512, k: 512 };
        o.measured.insert(case_key(&bf16(other), tile(128, 128, 32)), 500);
        assert_eq!(o.measured_gemm(&bf16(shape()), &[tile(128, 128, 32)]), None);
    }

    /// The key the compiler looks up must be the key a campaign writes.
    #[test]
    fn case_key_carries_shape_and_variant() {
        assert_eq!(case_key(&bf16(shape()), tile(128, 128, 32)), "4096,4096,4096,bf16,128,128,32");
        assert_eq!(case_key(&fp8(shape()), tile(128, 128, 32)), "4096,4096,4096,fp8,128,128,32");
    }

    /// The digests the compiler queries with must be the digests a campaign
    /// writes -- toolchain included, from the BUILD not from `hw` (which
    /// `from_spec` leaves None). The reviewer's P1 #4.
    #[test]
    fn build_digests_carry_the_real_toolchain() {
        let d = build_digests(&build());
        assert_eq!(d.toolchain, "cuda-13.0", "not 'unknown'");
        assert_eq!(d.oracle, GEMM_ORACLE);
        assert_eq!(d.implementation, build().label());
    }

    #[test]
    fn kernel_variant_matching() {
        let k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm90a, 128, 128, 32, "b");
        assert_eq!(k.quant, QuantScheme::None);
        assert!(kernel_serves(&k, &bf16(shape())));
        assert!(!kernel_serves(&k, &fp8(shape())));
    }

    #[test]
    fn tier_reflects_evidence() {
        assert_eq!(OracleEvidence::Disabled.tier(), "portable");
        assert_eq!(
            OracleEvidence::Capability { build: "b".into(), kernels: 3 }.tier(),
            "portable",
            "a capability filter is not a measurement"
        );
        assert_eq!(
            OracleEvidence::Measured { build: "b".into(), kernels: 3, records: 2 }.tier(),
            "sku-calibrated"
        );
    }
}

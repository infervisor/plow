//! The decode-GEMV tuning cell: what a record is keyed by, and what it can and cannot decide.
//!
//! # Read this before assuming it works like [`crate::gemm`]
//!
//! The prefill-GEMM cell ranks **five tiles against each other at one shape**, and
//! `devgen::pick_tile` picks the winner. The decode-GEMV path has no such axis and it is
//! important to say so rather than build a store that looks like it selects something:
//!
//! * The row bucket `PLOW_GEMV_MM` is a **compile-time macro of the object**
//!   (`runtime/amd/op_gemm.h`), set once per build from `PLOW_DECODE_BATCH`
//!   (`scripts/build_gfx950.sh`). It is not per-op and the packet cannot choose it.
//! * The K-unroll is a **runtime branch on K inside the kernel** (`K == 2560` -> 5,
//!   `K == 4096` -> 8, else 6), not an emitted immediate.
//! * So for a given `(M, N, K, quant)` there is exactly **one** opcode the emitter can
//!   reach, chosen by static branching on the weight encoding. Nothing to rank.
//!
//! **What this cell is for, then.** Two things, and only these:
//!
//! 1. **Coverage.** `PLOW_TUNE_DUMP=1` now dumps every GEMV shape the compiler resolves
//!    ([`packet::devbuild`]); this key is what turns that census into HIT/MISS. The GEMM
//!    campaign's shape list was hand-authored and GLM-5.2 prefill was 100% unmeasured for the
//!    tuner's whole life as a result. A census with no store to check against is a list
//!    nobody can audit.
//! 2. **The M curve.** The one decision measurement can settle on this path is the *object*
//!    one — which `PLOW_GEMV_MM`, and whether the outer walk is on — and that is answered by
//!    the same shape measured at several M. `plans/knob-contract.md` §6g-BATCH prices the
//!    endpoint of exactly that curve (57.9 / 106.5 / 141.7 / 202.3 / 142.4 tok/s at
//!    B=1/2/4/8/16) and §6g-WALK attributes the B=16 loss to spill plus lost fusions. Those
//!    are aggregate token rates; this cell is the per-shape decomposition under them.
//!
//! Because the axis is the OBJECT, the discriminator is `Digests::implementation` — records
//! from an `MM=8` build and an `MM=16` build key to different digests and can never be
//! compared as if they were the same kernel. That is the same limitation
//! `tuning/README.md` records for the `nvidia/sm_120a/rtx-5090` prefill-tile cell, and it is
//! a property of the entity, not an oversight here.

use kernelcaps::QuantScheme;
use packet::dev::DevOp;

/// The op-case key a GEMV measurement is filed and looked up under.
///
/// Re-exported from `packet` rather than re-implemented: the PRODUCER of these keys is
/// `packet::devbuild::Builder::emit_dep`, which is the one place every GEMV emit funnels
/// through, and `packet` has no dependencies by design. Two formatters that agree only until
/// one changes is precisely how a tuner comes to publish records the compiler never finds —
/// silently, because "no record" and "never measured" produce identical output.
pub use packet::devbuild::gemv_op_case;

/// Identity of the correctness oracle a GEMV measurement was checked against: an f64
/// dot-product reference over sampled output elements, computed as
/// `runtime/ubench/gemv_row_sweep.c` computes it.
///
/// Distinct from [`crate::GEMM_ORACLE`] because it checks a different thing: a GEMV's
/// reduction is per output COLUMN over the full K with no tiling, and its M rows are
/// predicated rather than blocked — so "row `m >= M` was left untouched" is a failure this
/// oracle must catch and the GEMM one never had to. That failure is not hypothetical: it is
/// the shape of the `PLOW_GEMV_MM` bug (`scripts/build_gfx950.sh:51`), where every AMD decode
/// object compiled at MM=1, wrote row 0, and left rows 1..B-1 as it found them.
pub const GEMV_ORACLE: &str = "gemv-f64-dot-allrow-v1";

/// The `(M, N, K, quant)` an emitted GEMV op carries, or `None` if `op` is not a GEMV.
///
/// Thin wrapper over [`DevOp::gemv_case`] that lifts the quant spelling back into a
/// [`QuantScheme`], so an ingest can key a record the same way the compiler will look it up.
pub fn gemv_case(op: DevOp, i: &[u32; 8]) -> Option<(&'static str, u32, u32, u32, QuantScheme)> {
    let (fam, m, n, k, q) = op.gemv_case(i)?;
    Some((fam, m, n, k, crate::gemm::parse_quant(q)?))
}

/// The op-case family a sweep symbol belongs to (`gemv_glu_m8` -> `"gemvglu"`).
///
/// Must agree with [`DevOp::gemv_case`]'s family, and `sweep_symbols_map_to_opcode_and_bucket`
/// checks that it does — a writer and a reader that spell the key separately agree only until
/// one of them changes, and then the compiler silently finds no record.
pub fn gemv_sample_family(sym: &str) -> Option<&'static str> {
    let op = gemv_sample_opcode(sym)?;
    Some(op.gemv_case(&[1; 8])?.0)
}

/// The opcode a GEMV sample was taken on, from the symbol the sweep harness timed.
///
/// The map is explicit rather than derived from the symbol name for the same reason
/// [`crate::gemm::gemm_rung_opcode`] is: a harness kernel that no interpreter arm dispatches
/// is a legitimate measurement of a kernel body and NOT a selectable fact, and storing the two
/// alike is how a plan comes to name a kernel that does not exist. `gemv_m2` has no
/// interpreter arm today (`scripts/build_gfx950.sh` rounds the bucket to a power of two, so
/// MM=2 is reachable, but no shipped recipe builds it) — it is measured for the M curve and
/// returns the same opcode, because the OPCODE is not what varies across the rungs; the
/// object is.
pub fn gemv_sample_opcode(sym: &str) -> Option<DevOp> {
    Some(match sym {
        s if s.starts_with("gemv_m") => DevOp::Gemv,
        s if s.starts_with("gemv_glu_m") => DevOp::GemvGlu,
        s if s.starts_with("gemv_qkv_m") => DevOp::GemvQkv,
        _ => return None,
    })
}

/// The compiled row bucket a sweep symbol was built at (`gemv_m8` -> 8).
///
/// Part of the record's identity in the only way `KernelMeasurement` can express it: the M
/// the campaign asked for goes in the op case, and the bucket the object was compiled at is
/// carried in the campaign label. They are different numbers whenever the walk is on — that
/// is the whole point of the walk — and conflating them would make an `MM=8` object serving
/// `M=16` indistinguishable from an `MM=16` object.
pub fn gemv_sample_bucket(sym: &str) -> Option<u32> {
    sym.rsplit_once('m')
        .and_then(|(_, n)| n.parse().ok())
        .filter(|n| *n >= 1 && *n <= 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key must separate the two families, or a prefill GEMM timing could be served for a
    /// decode GEMV at the same `(m,n,k)` — different kernels moving different bytes.
    #[test]
    fn the_key_separates_gemv_from_gemm() {
        assert_eq!(gemv_op_case("gemv", 1, 6144, 512, "None"), "gemv/1x6144x512/None");
        assert_ne!(
            gemv_op_case("gemv", 1, 6144, 512, "None"),
            crate::gemm::gemm_op_case(1, 6144, 512, QuantScheme::None)
        );
    }

    /// The three arms must NOT share a case. `TuneStore::best_for` ranks every kernel filed
    /// under one case and returns the fastest, so a shared case makes it compare a plain GEMV
    /// against a fused Q|K|V as though they were two implementations of one operation — and
    /// report whichever was faster as the winner for a shape only one of them can serve. It
    /// did exactly that before the family entered the key.
    #[test]
    fn the_three_arms_do_not_share_a_case() {
        let mut i = [0u32; 8];
        i[0] = 8;
        i[1] = 5376;
        i[2] = 5376;
        let plain = DevOp::Gemv.gemv_case(&i).unwrap();
        let glu = DevOp::GemvGlu.gemv_case(&i).unwrap();
        let qkv = DevOp::GemvQkv.gemv_case(&i).unwrap();
        let key = |c: (&str, u32, u32, u32, &str)| gemv_op_case(c.0, c.1, c.2, c.3, c.4);
        assert_ne!(key(plain), key(glu));
        assert_ne!(key(glu), key(qkv));
        assert_ne!(key(plain), key(qkv));
    }

    /// `GemvQkv` carries THREE output widths, and the shape that governs its cost is their
    /// sum. Getting this wrong files the fused op under the q-projection's shape, where it
    /// would be compared against a measurement of a third of the work.
    #[test]
    fn qkv_sums_its_three_output_widths() {
        let mut i = [0u32; 8];
        i[0] = 8; // M
        i[1] = 4096; // Nq
        i[2] = 5376; // K
        i[3] = 1024; // Nk
        i[4] = 1024; // Nv
        assert_eq!(
            DevOp::GemvQkv.gemv_case(&i),
            Some(("gemvqkv", 8, 4096 + 1024 + 1024, 5376, "None"))
        );
    }

    /// Every GEMV opcode the ISA declares must produce a case, and no non-GEMV may.
    ///
    /// Written against [`DevOp::ALL`] rather than a hand-listed set, so an opcode added to the
    /// ISA without a shape rule fails here instead of silently vanishing from the census —
    /// which is the exact failure this whole census exists to stop.
    #[test]
    fn every_gemv_opcode_has_a_shape_rule() {
        let i = [1u32, 2, 3, 4, 5, 6, 7, 8];
        for op in DevOp::ALL {
            let is_gemv = format!("{op:?}").starts_with("Gemv");
            assert_eq!(
                op.gemv_case(&i).is_some(),
                is_gemv,
                "{op:?}: gemv_case disagrees with the opcode family"
            );
        }
    }

    #[test]
    fn sweep_symbols_map_to_opcode_and_bucket() {
        assert_eq!(gemv_sample_opcode("gemv_m8"), Some(DevOp::Gemv));
        assert_eq!(gemv_sample_opcode("gemv_glu_m16"), Some(DevOp::GemvGlu));
        assert_eq!(gemv_sample_opcode("gemv_qkv_m1"), Some(DevOp::GemvQkv));
        assert_eq!(gemv_sample_opcode("gemm_c0"), None);
        assert_eq!(gemv_sample_bucket("gemv_m8"), Some(8));
        assert_eq!(gemv_sample_bucket("gemv_glu_m16"), Some(16));
        assert_eq!(gemv_sample_bucket("gemv_m32"), None);
        // The sweep's family and the emitter's family must be the same string.
        assert_eq!(gemv_sample_family("gemv_m8"), Some("gemv"));
        assert_eq!(gemv_sample_family("gemv_glu_m8"), Some("gemvglu"));
        assert_eq!(gemv_sample_family("gemv_qkv_m8"), Some("gemvqkv"));
    }
}

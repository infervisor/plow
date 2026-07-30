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

/// The decode-GEMV sweep symbols, and the opcode each one exercises.
///
/// This is the GEMV twin of [`crate::gemm`]'s `RUNGS`, and it carries the same axis: the GEMM
/// table is a `(bf16, fp8, mxfp4)` triple per tile, and this one is a per-encoding ROW per
/// family. The shape differs because the two paths differ — a GEMM rung is a tile that exists
/// in all three encodings, while a GEMV family may have no quantized twin at all (`gemv_qkv`)
/// or exist ONLY quantized (`gemv_blk`) — and an `Option` grid over a fixed triple would have
/// to spell those holes as `None` anyway. Listing the reachable cells is the same fact stated
/// without the holes.
///
/// The quant an entry belongs to is NOT restated here: [`DevOp::gemv_case`] already carries
/// both the op-case family and the quant spelling for every GEMV opcode, and restating either
/// would be a second speller that agrees only until one of them changes — the exact failure
/// this module's header calls out. [`gemv_rung_opcode`] therefore ASKS `gemv_case` rather than
/// matching on a column.
///
/// The map is explicit rather than derived from the symbol name for the same reason
/// [`crate::gemm::gemm_rung_opcode`] is: a harness kernel that no interpreter arm dispatches
/// is a legitimate measurement of a kernel body and NOT a selectable fact, and storing the two
/// alike is how a plan comes to name a kernel that does not exist.
const SYMBOLS: [(&str, DevOp); 9] = [
    ("gemv", DevOp::Gemv),
    ("gemv_fp8", DevOp::GemvFp8),
    ("gemv_mxfp4", DevOp::GemvMxfp4),
    ("gemv_blk", DevOp::GemvFp8Blk),
    ("gemv_glu", DevOp::GemvGlu),
    ("gemv_glu_fp8", DevOp::GemvGluFp8),
    ("gemv_glu_mxfp4", DevOp::GemvGluMxfp4),
    ("gemv_qkv", DevOp::GemvQkv),
    ("gemv_qkvg", DevOp::GemvQkvg),
];

/// The opcode serving op-case family `family` under `quant`, or `None` when no arm does.
///
/// The GEMV twin of [`crate::gemm::gemm_rung_opcode`], and the reason `tune best --quant
/// Mxfp4` resolves on this path as it does on the GEMM one. `None` is the right answer for
/// `("gemvqkv", Mxfp4)` — the fused q|k|v arm has no quantized twin in the ISA — and reporting
/// it as such is what stops a bf16 q|k|v timing being served for an mxfp4 op that does not
/// exist.
pub fn gemv_rung_opcode(family: &str, quant: QuantScheme) -> Option<DevOp> {
    SYMBOLS.iter().find_map(|&(_, op)| {
        let (fam, _, _, _, q) = op.gemv_case(&[1; 8])?;
        (fam == family && crate::gemm::parse_quant(q) == Some(quant)).then_some(op)
    })
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
/// The stem is matched WHOLE against [`SYMBOLS`] after the `_m<bucket>` suffix is split off,
/// not by prefix. Prefix matching is what the first version did, and it silently mapped
/// `gemv_mxfp4_m1` onto [`DevOp::Gemv`] — the string starts with `gemv_m` — filing every mxfp4
/// measurement under the bf16 op case, where `best_for` would rank a 4-bit timing as if it
/// were the 16-bit kernel's. That is the precise shape of "the tuner publishes a record the
/// compiler finds and must not trust".
///
/// `gemv_m2` has no interpreter arm today (`scripts/build_gfx950.sh` rounds the bucket to a
/// power of two, so MM=2 is reachable, but no shipped recipe builds it) — it is measured for
/// the M curve and returns the same opcode, because the OPCODE is not what varies across the
/// rungs; the object is.
pub fn gemv_sample_opcode(sym: &str) -> Option<DevOp> {
    let (stem, _) = sym.rsplit_once("_m")?;
    SYMBOLS.iter().find(|(s, _)| *s == stem).map(|&(_, op)| op)
}

/// The compiled row bucket a sweep symbol was built at (`gemv_m8` -> 8).
///
/// Part of the record's identity in the only way `KernelMeasurement` can express it: the M
/// the campaign asked for goes in the op case, and the bucket the object was compiled at is
/// carried in the campaign label. They are different numbers whenever the walk is on — that
/// is the whole point of the walk — and conflating them would make an `MM=8` object serving
/// `M=16` indistinguishable from an `MM=16` object.
pub fn gemv_sample_bucket(sym: &str) -> Option<u32> {
    sym.rsplit_once("_m")
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

    /// `GemvQkvg` extends that convention to a FOURTH stream, and must sum all four.
    ///
    /// K3's KDA projects one pre-normed x four ways (q, k, v and the full-rank output gate),
    /// and 69 of its 93 layers now emit this op rather than four separate `Gemv`s. At the real
    /// TP8 geometry every width is 1536, so summing three instead of four would file the op
    /// under 4608 — three quarters of the bytes it actually moves, and a shape a DIFFERENT
    /// packet legitimately occupies.
    #[test]
    fn qkvg_sums_its_four_output_widths() {
        let mut i = [0u32; 8];
        i[0] = 1; // M — decode
        i[1] = 1536; // Nq  (12288 / TP8)
        i[2] = 7168; // K   — K3 hidden
        i[3] = 1536; // Nk
        i[4] = 1536; // Nv
        i[5] = 1536; // Ng  — the fourth stream op 22 does not have
        assert_eq!(
            DevOp::GemvQkvg.gemv_case(&i),
            Some(("gemvqkvg", 1, 4 * 1536, 7168, "None"))
        );
        // The 3-stream and 4-stream fusions must not share a case: op 22 forwards into the
        // same body with Ng = 0, so they are the same KERNEL but not the same WORK.
        let three = DevOp::GemvQkv.gemv_case(&i).unwrap();
        let four = DevOp::GemvQkvg.gemv_case(&i).unwrap();
        assert_ne!(gemv_op_case(three.0, three.1, three.2, three.3, three.4),
                   gemv_op_case(four.0, four.1, four.2, four.3, four.4));
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

    /// The quantized symbols resolve to the QUANTIZED opcodes, and the bf16 stem does not
    /// swallow them.
    ///
    /// `gemv_mxfp4_m1` starts with `gemv_m`. Under the prefix match this map used to do, it
    /// resolved to [`DevOp::Gemv`] and every mxfp4 sample was filed under the bf16 op case —
    /// where `best_for` ranks records against each other, so a 4-bit timing would have been
    /// returned as the winning implementation of the 16-bit op. Nothing else in the pipeline
    /// could have caught it: the record is well-formed, the digests are live, and the gate is
    /// green.
    #[test]
    fn quantized_symbols_do_not_collide_with_the_bf16_stem() {
        assert_eq!(gemv_sample_opcode("gemv_mxfp4_m1"), Some(DevOp::GemvMxfp4));
        assert_eq!(gemv_sample_opcode("gemv_glu_mxfp4_m1"), Some(DevOp::GemvGluMxfp4));
        assert_eq!(gemv_sample_opcode("gemv_fp8_m8"), Some(DevOp::GemvFp8));
        assert_eq!(gemv_sample_opcode("gemv_blk_m1"), Some(DevOp::GemvFp8Blk));
        assert_eq!(gemv_sample_bucket("gemv_mxfp4_m1"), Some(1));
        assert_eq!(gemv_sample_bucket("gemv_glu_mxfp4_m16"), Some(16));
        // A stem that is not in the table is not a GEMV sample, however GEMV-ish it reads.
        assert_eq!(gemv_sample_opcode("gemv_mxfp8_m1"), None);
        assert_eq!(gemv_sample_opcode("gemv_qkv_mxfp4_m1"), None);
    }

    /// An mxfp4 GEMV and its bf16 twin must land in DIFFERENT op cases.
    ///
    /// The GEMV twin of `the_key_separates_precisions`, and the reason the quant axis had to
    /// exist at all: these two kernels move 4 bits and 16 bits per weight for the same
    /// `(M, N, K)`, so one timing served for the other is off by ~4x in the direction that
    /// makes the slow choice look fast.
    #[test]
    fn the_key_separates_gemv_precisions() {
        let k = |q| {
            let op = gemv_rung_opcode("gemv", q).expect("gemv has this encoding");
            let (fam, _, _, _, qs) = op.gemv_case(&[1; 8]).unwrap();
            gemv_op_case(fam, 1, 12288, 7168, qs)
        };
        let bf16 = k(QuantScheme::None);
        let mx = k(QuantScheme::Mxfp4);
        assert_eq!(bf16, "gemv/1x12288x7168/None");
        assert_eq!(mx, "gemv/1x12288x7168/Mxfp4");
        assert_ne!(bf16, mx);
        assert_ne!(bf16, k(QuantScheme::W8A8));
    }

    /// Every symbol resolves to a distinct opcode, and every opcode round-trips back to the
    /// `(family, quant)` it is filed under.
    ///
    /// The GEMV twin of `rungs_map_to_distinct_opcodes`. The round-trip is the load-bearing
    /// half: [`gemv_rung_opcode`] answers by ASKING [`DevOp::gemv_case`], so this pins that the
    /// lookup a campaign writes with and the one the compiler reads with are one function.
    #[test]
    fn symbols_map_to_distinct_opcodes_and_round_trip() {
        let mut seen = std::collections::BTreeSet::new();
        for (sym, op) in SYMBOLS {
            assert!(seen.insert(op as u16), "{op:?} is reachable from two symbols");
            let (fam, _, _, _, q) = op.gemv_case(&[1; 8]).expect("a GEMV opcode has a shape rule");
            let quant = crate::gemm::parse_quant(q).expect("a spelling QuantScheme knows");
            assert_eq!(gemv_rung_opcode(fam, quant), Some(op), "{sym} does not round-trip");
        }
        assert_eq!(seen.len(), SYMBOLS.len());
        // The fused q|k|v arm has NO quantized twin in the ISA. Answering `None` is what stops
        // a bf16 q|k|v timing being served for an mxfp4 op that cannot be emitted.
        assert_eq!(gemv_rung_opcode("gemvqkv", QuantScheme::Mxfp4), None);
        assert_eq!(gemv_rung_opcode("gemvqkv", QuantScheme::W8A8), None);
        // And block-fp8 is its own family, not a column of the plain one.
        assert_eq!(gemv_rung_opcode("gemvblk", QuantScheme::W8A8), Some(DevOp::GemvFp8Blk));
        assert_eq!(gemv_rung_opcode("gemvblk", QuantScheme::None), None);
    }

    /// Every symbol in the table must be a kernel the sweep harness actually launches.
    ///
    /// The GEMV twin of `every_rung_is_compiled_into_the_sweep_harness`, and it reads
    /// `gemv_row_sweep.c` rather than restating it: a symbol added here but never swept is a
    /// case that can be looked up and never measured, which is the state this whole cell
    /// exists to leave behind. `gemv_fp8` and `gemv_blk` are deliberately NOT asserted — they
    /// are schema cells with no sweep arm yet, and the assertion below names them so the
    /// omission is a stated residual rather than a silent hole.
    #[test]
    fn every_swept_symbol_is_driven_by_the_harness() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../runtime/ubench/gemv_row_sweep.c");
        let src = std::fs::read_to_string(&p).expect("gemv_row_sweep.c");
        for stem in ["gemv", "gemv_glu", "gemv_qkv", "gemv_mxfp4", "gemv_glu_mxfp4"] {
            assert!(
                src.contains(&format!("\"{stem}_m\"")),
                "{stem}_m is not a base the sweep drives — the case cannot be measured"
            );
        }
        // Named residuals: the schema knows these, the harness has no arm for them.
        //
        // `gemv_fp8` / `gemv_blk` need a per-channel and a `[N/128][K/128]` f32 scale grid
        // respectively — a different ORACLE, not a different shape. `gemv_qkvg` needs a
        // `GEMV_QKVG_WALK_VARIANT` golden that `test_kernels.hip` does not yet declare; the op
        // itself is covered on hardware by `runtime/tests/gemv_qkvg_gfx950_test.hip`, which is
        // a correctness golden and not a shape sweep. Each reads MISS in the census, which is
        // the correct reading, and is why this asserts their absence rather than omitting them.
        assert!(!src.contains("\"gemv_fp8_m\""), "gemv_fp8 gained an arm — assert it above");
        assert!(!src.contains("\"gemv_blk_m\""), "gemv_blk gained an arm — assert it above");
        assert!(!src.contains("\"gemv_qkvg_m\""), "gemv_qkvg gained an arm — assert it above");
    }
}

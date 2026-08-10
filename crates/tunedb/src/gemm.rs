//! The prefill-GEMM tuning cell: what a record is keyed by, and which opcode carries a tile.
//!
//! This is deliberately in `tunedb` rather than in the compiler. Both the campaign that WRITES
//! a measurement and the compiler that READS one need the op-case key and the tile→opcode map,
//! and if each formats them itself they agree only until one of them changes. The failure mode
//! is silent: the compiler finds no record, falls back to the analytical model, and reports the
//! `portable` tier — which is exactly what it reports when no measurement was ever taken. A
//! campaign can then run, publish, and change nothing, with every gate green.
//!
//! `devgen::pick_tile` calls [`gemm_op_case`]; `plowc tune ingest` calls it too.

use kernelcaps::QuantScheme;
use packet::dev::DevOp;

/// `HardwareFingerprint::tuning_path()` for the MI350X/MI355X cell.
pub const GFX950_CELL: &str = "amd/gfx950/mi350x";

/// The tuning cell an AMD part's records live in — the ONE rule, shared by every writer and
/// every reader.
///
/// It lives here for the reason the op-case key does (module header): a campaign that publishes
/// into one cell and a compiler that looks up another produce no error and no diff. They produce
/// a MISS, and a miss is byte-identical to "never measured". `tunedb-gemv` had `GFX950_CELL`
/// hardcoded while `devgen::amd_tuning_cell` already keyed off `--gpu`, so a gfx942 decode-GEMV
/// campaign would have published into MI350X's cell and then been rejected as wrong-hardware —
/// which is the mechanical reason the decode-GEMV census has zero records on any AMD part.
///
/// gfx950 keeps the ONE checked-in cell regardless of SKU: the repo's CDNA4 records live in
/// `amd/gfx950/mi350x`, and keying MI355X by fingerprint would send it to an empty
/// `amd/gfx950/mi355x` and the analytical fallback — a silent regression for the part the repo's
/// AMD numbers were measured on. Per-SKU cells start with gfx942.
pub fn amd_tuning_cell(spec: &hwspec::GpuSpec) -> String {
    if hwspec::IsaLevel::from_spec(spec) != Some(hwspec::IsaLevel::Gfx942) {
        return GFX950_CELL.to_string();
    }
    hwspec::HardwareFingerprint::from_spec(spec)
        .map(|hw| hw.tuning_path())
        .unwrap_or_else(|| GFX950_CELL.to_string())
}

/// Identity of the correctness oracle a GEMM measurement was checked against: an f64
/// dot-product reference over sampled output elements, as `runtime/ubench/gemm_tile_sweep.c`
/// and `runtime/tests/gemm_gfx950_test.c` both compute it.
///
/// Part of the staleness key, so a record checked by a weaker oracle cannot be served to a
/// caller expecting this one.
pub const GEMM_ORACLE: &str = "gemm-f64-dot-spotcheck-v1";

/// The op-case key a GEMM measurement is filed and looked up under.
///
/// `quant` is in the key because a bf16 timing must never be served for an fp8 op — they are
/// different kernels moving different numbers of bytes. Deliberately NOT keyed by tile: the
/// value stored against one case is a map over tiles, which is what makes ranking possible.
pub fn gemm_op_case(m: i64, n: i64, k: i64, quant: QuantScheme) -> String {
    format!("gemm/{m}x{n}x{k}/{quant:?}")
}

/// The gfx950 prefill GEMM rungs: `BMxBNxBK` as the sweep spells it, and the opcode carrying
/// that tile in each of the three weight encodings.
///
/// Mirrors `devgen::GFX950_RUNGS` and `kernelcaps::targets::GFX950_QUANT_OBJECTS`. The three
/// tables are checked against each other by `rung_tables_agree_across_crates`.
const RUNGS: [(&str, DevOp, DevOp, DevOp); 5] = [
    ("256x256x64", DevOp::Gemm, DevOp::GemmFp8, DevOp::GemmMxfp4),
    (
        "128x128x64",
        DevOp::GemmMed,
        DevOp::GemmMedFp8,
        DevOp::GemmMedMxfp4,
    ),
    (
        "64x128x64",
        DevOp::GemmSmall,
        DevOp::GemmSmallFp8,
        DevOp::GemmSmallMxfp4,
    ),
    (
        "128x256x64",
        DevOp::GemmWide,
        DevOp::GemmWideFp8,
        DevOp::GemmWideMxfp4,
    ),
    (
        "192x256x64",
        DevOp::GemmC5,
        DevOp::GemmC5Fp8,
        DevOp::GemmC5Mxfp4,
    ),
];

/// The opcode that carries `tile` under `quant`, or `None` when no dispatch arm does.
///
/// `None` is the right answer for the calibration-only tiles the sweep also compiles
/// (320x128, 384x128, 128x384, 192x128 — `runtime/amd/test_kernels.hip`). They are legitimate
/// measurements of a kernel body and NOT selectable facts, because the interpreter has no arm
/// for them; storing them as if they were is how a plan comes to name a kernel that does not
/// exist.
pub fn gemm_rung_opcode(tile: &str, quant: QuantScheme) -> Option<DevOp> {
    let (_, bf16, fp8, mx) = RUNGS.iter().find(|r| r.0 == tile)?;
    Some(match quant {
        QuantScheme::None => *bf16,
        QuantScheme::W8A8 => *fp8,
        QuantScheme::Mxfp4 => *mx,
        _ => return None,
    })
}

/// Parse the `quant` field the sweep writes.
pub fn parse_quant(s: &str) -> Option<QuantScheme> {
    Some(match s {
        "None" => QuantScheme::None,
        "W8A8" => QuantScheme::W8A8,
        "Mxfp4" => QuantScheme::Mxfp4,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gfx942 part resolves to a gfx942 cell, NOT to gfx950's.
    ///
    /// This is the assertion `tunedb-gemv` failed: it imported `GFX950_CELL as CELL` and filed
    /// every record under it, so a decode-GEMV campaign on an MI300X published into MI350X's
    /// cell. The failure is silent in both directions — `devgen`'s reader finds nothing and
    /// falls back to the analytical model, which is byte-identical to "never measured", and the
    /// MI350X cell gains MI300X timings that look like real provenance. There is no diff to
    /// notice, which is why the test is the deliverable and not the four-line fix.
    #[test]
    fn a_gfx942_part_resolves_to_a_gfx942_cell() {
        let mi300 = hwspec::registry::lookup("MI300X").expect("MI300X in registry");
        let cell = amd_tuning_cell(mi300);
        assert_eq!(cell, "amd/gfx942/mi300x");
        assert_ne!(
            cell, GFX950_CELL,
            "a gfx942 campaign filed into gfx950's cell is rejected as wrong-hardware, and \
             rejection is indistinguishable from never having measured"
        );

        // The converse half: CDNA4 keeps the ONE checked-in cell whatever the SKU, so the 3080
        // records already in `amd/gfx950/mi350x` stay reachable from an MI355X.
        for sku in ["MI350X", "MI355X"] {
            let spec = hwspec::registry::lookup(sku).expect("in registry");
            assert_eq!(amd_tuning_cell(spec), GFX950_CELL, "{sku}");
        }
    }

    /// The key must round-trip the precision, because that is the field whose omission would
    /// serve a bf16 timing for an fp8 op.
    #[test]
    fn the_key_separates_precisions() {
        let a = gemm_op_case(128, 576, 6144, QuantScheme::None);
        let b = gemm_op_case(128, 576, 6144, QuantScheme::W8A8);
        let c = gemm_op_case(128, 576, 6144, QuantScheme::Mxfp4);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        assert_eq!(a, "gemm/128x576x6144/None");
    }

    /// Every rung resolves to a distinct opcode in every encoding, and a calibration-only tile
    /// resolves to none.
    #[test]
    fn rungs_map_to_distinct_opcodes() {
        let mut seen = std::collections::BTreeSet::new();
        for (tile, ..) in RUNGS {
            for q in [QuantScheme::None, QuantScheme::W8A8, QuantScheme::Mxfp4] {
                let op = gemm_rung_opcode(tile, q).expect("every rung has an opcode");
                assert!(seen.insert(op as u16), "{op:?} is reachable from two rungs");
            }
        }
        assert_eq!(seen.len(), RUNGS.len() * 3);
        assert_eq!(gemm_rung_opcode("320x128x64", QuantScheme::None), None);
        assert_eq!(gemm_rung_opcode("128x384x64", QuantScheme::None), None);
        // An encoding with no prefill GEMM family at all must not silently borrow another's.
        assert_eq!(gemm_rung_opcode("256x256x64", QuantScheme::BlockFp8), None);
    }

    /// The rung table here and the one the sweep harness compiles must list the same tiles.
    ///
    /// Reads `test_kernels.hip` rather than restating it: a rung added to `op_gemm.h` and
    /// `dev_isa.h` but never compiled into the sweep is a rung that can be selected and never
    /// measured, which is the state the whole campaign existed to leave behind.
    #[test]
    fn every_rung_is_compiled_into_the_sweep_harness() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../runtime/amd/test_kernels.hip");
        let src = std::fs::read_to_string(&p).expect("test_kernels.hip");
        for (tile, ..) in RUNGS {
            let dims: Vec<&str> = tile.split('x').collect();
            let want = format!("{}, {}, {})", dims[0], dims[1], dims[2]);
            assert!(
                src.contains(&want),
                "no GEMM_VARIANT(..., {want} in test_kernels.hip — rung {tile} cannot be measured"
            );
        }
    }
}

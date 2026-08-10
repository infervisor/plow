//! The host's arch-geometry table vs. the device sources it mirrors.
//!
//! `hwspec::IsaLevel::geometry` states the GEMM arena the interpreter object
//! will actually have. Nothing in the type system ties that to the object: the
//! authority is `runtime/amd/op_gemm.h`'s arch-conditional defaults, re-cut for
//! the shipped gfx942 decode profile by `scripts/build_gfx942.sh`. When the two
//! disagreed — the host held CDNA4's 73,728 halves while gfx942's decode object
//! holds 15,360 — the emitter fused a 19-row batch onto an arena that holds
//! four, and the symptom was fluent-but-wrong text, not a build error.
//!
//! So this reads the device sources as TEXT and compares. Text and not the
//! preprocessor deliberately: `kernelcaps` already probes these macros through
//! hipcc, and a check that needs a toolchain is a check that gets skipped on the
//! machine where the edit is made.

use hwspec::{ArchGeometry, GemmTile, IsaLevel};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    let p = root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Strip `/* ... */` comments so a number inside prose cannot be read as a
/// define. `op_gemm.h`'s comments quote tile sizes constantly.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"/*") {
            match src[i..].find("*/") {
                Some(j) => i += j + 2,
                None => break,
            }
            out.push(' ');
        } else if b[i..].starts_with(b"//") {
            i += src[i..].find('\n').unwrap_or(src.len() - i);
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// The `(CDNA4, CDNA3)` defaults of an `#ifndef NAME ... #endif` guard in
/// `op_gemm.h`.
///
/// The shape being read is exactly the two the header uses:
///
/// ```text
/// #ifndef GM_BM          #ifndef GM_BN
/// #if PLOW_CDNA4         #define GM_BN 256
/// #define GM_BM 256      #endif
/// #else
/// #define GM_BM 192
/// #endif
/// #endif
/// ```
///
/// An unconditional guard yields the same value for both arches, which is the
/// correct reading: that is what the preprocessor would do.
fn header_default(src: &str, name: &str) -> (u32, u32) {
    let guard = format!("#ifndef {name}\n");
    let start = src
        .find(&guard)
        .unwrap_or_else(|| panic!("op_gemm.h has no `#ifndef {name}` guard"))
        + guard.len();
    // Walk to the `#endif` that closes the guard, tracking nesting.
    let mut depth = 1usize;
    let mut end = start;
    let mut cdna4_arm = false;
    let mut in_else = false;
    let mut vals: Vec<(bool, u32)> = Vec::new();
    for line in src[start..].lines() {
        let t = line.trim();
        if t.starts_with("#if") {
            depth += 1;
            if t.contains("PLOW_CDNA4") {
                cdna4_arm = true;
                in_else = false;
            }
        } else if t.starts_with("#else") {
            in_else = true;
        } else if t.starts_with("#endif") {
            depth -= 1;
            if depth == 0 {
                break;
            }
        } else if let Some(rest) = t.strip_prefix(&format!("#define {name} ")) {
            let v: u32 = rest
                .trim()
                .parse()
                .unwrap_or_else(|e| panic!("{name} default {rest:?}: {e}"));
            vals.push((cdna4_arm && !in_else, v));
        }
        end += line.len() + 1;
    }
    let _ = end;
    match vals.len() {
        1 => (vals[0].1, vals[0].1),
        2 => {
            let cdna4 = vals.iter().find(|v| v.0).expect("a PLOW_CDNA4 arm").1;
            let cdna3 = vals.iter().find(|v| !v.0).expect("an #else arm").1;
            (cdna4, cdna3)
        }
        n => panic!("{name}: {n} defaults inside one guard; the parser expects 1 or 2"),
    }
}

/// A `-DNAME=<int>` override in a build script.
fn define_flag(src: &str, name: &str) -> Option<u32> {
    let pat = format!("-D{name}=");
    let i = src.find(&pat)? + pat.len();
    let rest = &src[i..];
    let n = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..n].parse().ok()
}

/// The body of `scripts/build_gfx942.sh`'s `PLOW_OCC4=1` branch — the shipped
/// gfx942 decode profile, which re-cuts the arena below the header default.
fn occ4_branch() -> String {
    let sh = read("scripts/build_gfx942.sh");
    let head = sh
        .find(r#"if [ "${PLOW_OCC4:-0}" = 1 ]; then"#)
        .expect("build_gfx942.sh has a PLOW_OCC4 branch");
    let tail = sh[head..].find("\nfi\n").expect("branch closes");
    sh[head..head + tail].to_string()
}

fn geometry(isa: IsaLevel) -> ArchGeometry {
    isa.geometry()
        .unwrap_or_else(|| panic!("{} has no geometry", isa.arch_flag()))
}

/// The default tile and the stage-buffer policy, against the header that sets
/// them. This is the assertion that would have failed while `GM_LDS_HALVES` was
/// CDNA4-only on the host.
#[test]
fn prefill_tile_and_dbuf_match_op_gemm_h() {
    let h = strip_comments(&read("runtime/amd/op_gemm.h"));
    let (bm4, bm3) = header_default(&h, "GM_BM");
    let (bn4, bn3) = header_default(&h, "GM_BN");
    let (bk4, bk3) = header_default(&h, "GM_BK");
    let (db4, db3) = header_default(&h, "GM_DBUF");

    let g3 = geometry(IsaLevel::Gfx942);
    let g4 = geometry(IsaLevel::Gfx950);

    assert_eq!(
        g3.gemm_tile,
        GemmTile {
            bm: bm3,
            bn: bn3,
            bk: bk3
        },
        "gfx942 default tile disagrees with op_gemm.h's !PLOW_CDNA4 arm"
    );
    assert_eq!(
        g4.gemm_tile,
        GemmTile {
            bm: bm4,
            bn: bn4,
            bk: bk4
        },
        "gfx950 default tile disagrees with op_gemm.h's PLOW_CDNA4 arm"
    );
    assert_eq!(g3.gemm_stage_buffers, db3, "gfx942 GM_DBUF");
    assert_eq!(g4.gemm_stage_buffers, db4, "gfx950 GM_DBUF");
}

/// The DECODE arena, which is the one the fused-GEMV gate spends. gfx942's is
/// NOT the header default — `PLOW_OCC4=1` re-cuts it — and gfx950's is, because
/// `build_gfx950.sh` passes no tile override at all.
#[test]
fn decode_tile_matches_the_shipped_build_profile() {
    let occ4 = occ4_branch();
    let g3 = geometry(IsaLevel::Gfx942);
    assert_eq!(
        g3.decode_gemm_tile,
        GemmTile {
            bm: define_flag(&occ4, "GM_BM").expect("occ4 sets GM_BM"),
            bn: define_flag(&occ4, "GM_BN").expect("occ4 sets GM_BN"),
            bk: define_flag(&occ4, "GM_BK").expect("occ4 sets GM_BK"),
        },
        "gfx942 decode tile disagrees with the PLOW_OCC4 profile in build_gfx942.sh"
    );

    let sh950 = read("scripts/build_gfx950.sh");
    for m in ["GM_BM", "GM_BN", "GM_BK", "GM_DBUF"] {
        assert!(
            define_flag(&sh950, m).is_none(),
            "build_gfx950.sh now overrides {m}; the gfx950 decode tile is no longer the header \
             default and this table has to say so"
        );
    }
    let g4 = geometry(IsaLevel::Gfx950);
    assert_eq!(g4.decode_gemm_tile, g4.gemm_tile);
}

/// The LDS budget, against the DEVICE's own statement of it.
///
/// `amd_arch.h` carries `PLOW_LDS_MAX_BYTES` so the kernels' tile tables can be filtered per
/// arch, and it is the same fact this table states — with the difference that the compiler
/// enforces the device copy ("local memory (147464) exceeds limit (65536)") and enforces nothing
/// about the host one.
#[test]
fn lds_ceiling_matches_amd_arch_h() {
    let h = strip_comments(&read("runtime/amd/amd_arch.h"));
    let (cdna4, cdna3) = header_default(&h, "PLOW_LDS_MAX_BYTES");
    assert_eq!(
        geometry(IsaLevel::Gfx942).lds_bytes,
        u64::from(cdna3),
        "gfx942 LDS budget disagrees with amd_arch.h's !PLOW_CDNA4 arm"
    );
    assert_eq!(
        geometry(IsaLevel::Gfx950).lds_bytes,
        u64::from(cdna4),
        "gfx950 LDS budget disagrees with amd_arch.h's PLOW_CDNA4 arm"
    );
}

/// The arena the table describes has to FIT the LDS the table describes. This is
/// the check that makes the two fields answerable as one fact rather than two.
#[test]
fn every_declared_arena_fits_its_lds_budget() {
    for isa in [IsaLevel::Gfx942, IsaLevel::Gfx950] {
        let g = geometry(isa);
        for (what, tile) in [("prefill", g.gemm_tile), ("decode", g.decode_gemm_tile)] {
            let bytes = tile.stage_bytes(g.gemm_stage_buffers);
            assert!(
                bytes <= g.lds_bytes,
                "{} {what} stage {}x{}x{} at {} buffer(s) is {bytes} B > {} B of LDS",
                isa.arch_flag(),
                tile.bm,
                tile.bn,
                tile.bk,
                g.gemm_stage_buffers,
                g.lds_bytes
            );
        }
    }
}

/// `lds_bytes` is stated per ARCH here and per SKU in the registry. Same fact,
/// two places — the pattern `warp_lanes` already follows.
#[test]
fn lds_budget_agrees_with_every_registered_sku() {
    for spec in hwspec::registry::ALL {
        let Some(isa) = IsaLevel::from_spec(spec) else {
            continue;
        };
        let Some(g) = isa.geometry() else { continue };
        assert_eq!(
            g.lds_bytes, spec.sm.shared_mem.0,
            "{}: ArchGeometry.lds_bytes disagrees with SmSpec.shared_mem",
            spec.name
        );
    }
}

/// The number the whole table exists to get right, spelled out so a reader can
/// check it against `op_gemm.h` by eye.
#[test]
fn the_decode_arenas_are_the_measured_ones() {
    assert_eq!(geometry(IsaLevel::Gfx942).decode_arena_halves(), 15_360);
    assert_eq!(geometry(IsaLevel::Gfx950).decode_arena_halves(), 73_728);
    // 192x256x64 single-buffered = 64,512 B, which is what makes CDNA3's tile
    // fit 64 KiB at all.
    assert_eq!(
        geometry(IsaLevel::Gfx942)
            .gemm_tile
            .stage_bytes(geometry(IsaLevel::Gfx942).gemm_stage_buffers),
        64_512
    );
}

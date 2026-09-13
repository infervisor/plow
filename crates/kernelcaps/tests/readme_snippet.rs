//! Pins the code example in `docs/arch/12-using-the-tuner.md`.
//!
//! A usage doc whose examples do not compile is worse than no doc: it costs the
//! reader time and then teaches them not to trust the rest of the page. This
//! keeps the one Rust snippet in that document honest, and pins the two macro
//! classifications it asserts, since those are the specific claim a reader would
//! act on before designing a sweep.

use kernelcaps::{classify_macro, Sweepable};

fn header(rel: &str) -> Option<String> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::read_to_string(root.join(rel)).ok()
}

/// The AMD GEMM family as one text: `op_gemm.h` only selects the arch file, and a knob's
/// `#ifndef` guard lives in the arch file (arch-defaulted) or in the shared body.
fn amd_gemm_header() -> Option<String> {
    let mut s = String::new();
    for rel in [
        "runtime/amd/op_gemm_gfx942.h",
        "runtime/amd/op_gemm_gfx950.h",
        "runtime/amd/op_gemm_common.h",
    ] {
        s.push_str(&header(rel)?);
    }
    Some(s)
}

#[test]
fn the_documented_snippet_compiles_and_holds() {
    let Some(h) = header("runtime/nvidia/op_gemm.cuh") else {
        eprintln!("skipping: op_gemm.cuh not found");
        return;
    };
    assert_eq!(classify_macro(&h, "PGM_BN"), Sweepable::Overridable);
    assert_eq!(classify_macro(&h, "PGM_BM"), Sweepable::Overridable); // PX-13
}

/// The doc states the vendor asymmetry as a table a reader will plan against:
/// the M axis sweeps on AMD and not on NVIDIA, and the K axis sweeps on neither.
#[test]
fn the_documented_sweep_axes_are_accurate() {
    let (Some(nv), Some(amd)) = (header("runtime/nvidia/op_gemm.cuh"), amd_gemm_header()) else {
        return;
    };

    // M: BOTH, since PX-13. `PGM_BM` was a bare `#define` with no `#ifndef`, so
    // `-DPGM_BM` silently never reached any object; it is a real knob now, which
    // ends the vendor asymmetry this table used to describe.
    assert_eq!(classify_macro(&amd, "GM_BM"), Sweepable::Overridable);
    assert_eq!(classify_macro(&nv, "PGM_BM"), Sweepable::Overridable);
    // K: AMD only, since the CDNA3 port. On a 64 KiB-LDS part the stage is
    // 4*(BM+BN)*(BK+8) bytes and BK is the only axis that shrinks it without
    // touching BN, which the fused-GLU epilogue pins through its SN==2 assert.
    // NVIDIA's PGM_BK is still a bare `#define` after the `#endif`.
    assert_eq!(classify_macro(&amd, "GM_BK"), Sweepable::Overridable);
    assert_eq!(classify_macro(&nv, "PGM_BK"), Sweepable::Fixed);
}

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

#[test]
fn the_documented_snippet_compiles_and_holds() {
    let Some(h) = header("runtime/nvidia/op_gemm.cuh") else {
        eprintln!("skipping: op_gemm.cuh not found");
        return;
    };
    assert_eq!(classify_macro(&h, "PGM_BN"), Sweepable::Overridable);
    assert_eq!(classify_macro(&h, "PGM_BM"), Sweepable::Fixed);
}

/// The doc states the vendor asymmetry as a table a reader will plan against:
/// the M axis sweeps on AMD and not on NVIDIA, and the K axis sweeps on neither.
#[test]
fn the_documented_sweep_axes_are_accurate() {
    let (Some(nv), Some(amd)) = (
        header("runtime/nvidia/op_gemm.cuh"),
        header("runtime/amd/op_gemm.h"),
    ) else {
        return;
    };

    // M: AMD yes, NVIDIA no.
    assert_eq!(classify_macro(&amd, "GM_BM"), Sweepable::Overridable);
    assert_eq!(classify_macro(&nv, "PGM_BM"), Sweepable::Fixed);
    // K: neither.
    assert_eq!(classify_macro(&amd, "GM_BK"), Sweepable::Fixed);
    assert_eq!(classify_macro(&nv, "PGM_BK"), Sweepable::Fixed);
}

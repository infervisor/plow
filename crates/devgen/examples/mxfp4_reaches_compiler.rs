//! Ad-hoc check: do the published MXFP4 records reach `pick_tile`, and do they change its answer?
//!
//! The bf16 half of this question is a test (`tests/tuned_tile_selection.rs`). This is deliberately
//! an EXAMPLE and not a test: the mxfp4 records cover Kimi-K3's shapes only, and pinning them in
//! the test suite would make `cargo test` fail for anyone who re-runs the bf16 half of the campaign
//! alone (`scripts/rebench_tune_gemm_all.sh --bf16-only`) — a red suite for a reason that is not a
//! regression. Run it explicitly after a campaign:
//!
//!     cargo run -p devgen --example mxfp4_reaches_compiler
use devgen::{gfx950_measured_rungs, gfx950_prefill_tile};
use kernelcaps::QuantScheme::{Mxfp4, None as Bf16};

fn main() {
    // K3 shapes the campaign measured, spanning the narrow-M and saturating regimes.
    for (m, n, k, label) in [
        (128u32, 576u32, 7168u32, "k3 mla-kv-a M=128"),
        (8192, 7168, 3584, "k3 moe-latent-up M=8192"),
        (4096, 1536, 7168, "k3 mla-q-a M=4096"),
    ] {
        let (rb, rx) = (
            gfx950_measured_rungs(m as i64, n as i64, k as i64, Bf16),
            gfx950_measured_rungs(m as i64, n as i64, k as i64, Mxfp4),
        );
        println!(
            "{label:<26} {m}x{n}x{k}  bf16 {rb} rung(s) -> {:?}   mxfp4 {rx} rung(s) -> {:?}",
            gfx950_prefill_tile(m, n, k, 256, Bf16),
            gfx950_prefill_tile(m, n, k, 256, Mxfp4),
        );
        assert!(
            rx > 0,
            "{label}: NO qualified mxfp4 record — the store is stale for this build"
        );
    }
}

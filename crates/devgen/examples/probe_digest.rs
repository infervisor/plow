//! Print the gfx950 GEMM inventory and tuning-family digests this checkout probes.
//!
//! The inventory digest proves which dispatch arms exist. The narrower tuning digest identifies
//! the standalone dense-GEMM family the tile campaign actually measures.
fn main() {
    let root = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."));
    match kernelcaps::dense_gemm_inventory(&root, hwspec::IsaLevel::Gfx950) {
        Ok(inv) => {
            let b = inv.build();
            println!(
                "gfx950 gemm build label (implementation == interpreter) = {}",
                b.label()
            );
            println!("toolchain = {}", b.toolchain);
            println!("defines   = {:?}", b.defines);
            println!("rungs     = {}", inv.iter().count());
            match kernelcaps::dense_gemm_tuning_build(&root, hwspec::IsaLevel::Gfx950) {
                Ok(tuned) => println!("tuning    = {}", tuned.label()),
                Err(e) => println!("tuning fingerprint failed: {e:?}"),
            }
        }
        Err(e) => println!("probe failed: {e:?}"),
    }
}

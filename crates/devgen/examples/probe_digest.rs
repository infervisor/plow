//! Print the gfx950 GEMM-inventory build digest this checkout probes.
//!
//! The digest is what makes a tunedb record selectable or stale (`devgen::gfx950_gemm_measurements`
//! builds `tunedb::Digests` from it), and it moves whenever anything reachable from
//! `runtime/amd/interp.hip` changes — a tile edit in `op_gemm.h` included. There was no way to ASK
//! for it: `tuned_tile_selection` only reports that records went stale, not what to re-key them to.
//! A campaign re-run needs the number, so print it.
fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
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
        }
        Err(e) => println!("probe failed: {e:?}"),
    }
}

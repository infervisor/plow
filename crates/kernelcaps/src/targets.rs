//! Build recipes for the interpreter objects, and the inventory derived from them.
//!
//! # What is declared here, and what is not
//!
//! Declared: **how to build an object** — which translation unit, which `-D`
//! flags, which arch. That mirrors `scripts/build_sm90a_cubin.sh`,
//! `scripts/build_gfx950.sh`, and `runtime/CMakeLists.txt`, and it is the same
//! kind of fact a Makefile holds.
//!
//! Not declared: **which kernels an object contains**. That is read out of the
//! object by [`crate::probe`]. The distinction is the point of this crate: a
//! recipe that drifts from the build produces a probe failure or a mismatched
//! [`BuildId`], whereas a hand-written kernel list produces a confident wrong
//! answer.

use std::path::Path;

use hwspec::IsaLevel;
use packet::dev::DevOp;

use crate::probe::{probe, probe_macros, ProbeError, ProbeTarget};
use crate::spec::{KernelSpec, ProfileId};
use crate::Inventory;

/// The recipe for one interpreter object.
pub struct ObjectRecipe {
    pub isa: IsaLevel,
    pub profile: ProfileId,
    /// Path to the translation unit, relative to the repository root.
    pub source: &'static str,
    /// Include directories, relative to the repository root.
    pub includes: &'static [&'static str],
    /// `-D` flags this object is built with.
    pub defines: &'static [&'static str],
    /// Arch flag passed to the compiler.
    pub arch_flag: &'static str,
    /// Compiler driver.
    pub compiler: &'static str,
    /// Dispatch function whose arms define the object's capability.
    pub dispatch_fn: &'static str,
    /// Header and macro names giving the object's compile-time GEMM tile.
    pub tile_macros: Option<(&'static str, [&'static str; 3])>,
}

impl ObjectRecipe {
    pub fn target(&self, root: &Path) -> ProbeTarget {
        ProbeTarget {
            compiler: self.compiler.to_string(),
            arch_flag: self.arch_flag.to_string(),
            includes: self
                .includes
                .iter()
                .map(|i| root.join(i).to_string_lossy().into_owned())
                .collect(),
            defines: self.defines.iter().map(|d| d.to_string()).collect(),
            source: root.join(self.source).to_string_lossy().into_owned(),
            dispatch_fn: self.dispatch_fn.to_string(),
        }
    }
}

const NV_INCLUDES: &[&str] = &["runtime/common", "runtime/nvidia"];
const AMD_INCLUDES: &[&str] = &["runtime/common", "runtime/amd"];

/// The prefill objects, per ISA. Flags mirror the build scripts.
pub fn prefill_recipe(isa: IsaLevel) -> Option<ObjectRecipe> {
    Some(match isa {
        // scripts/build_sm90a_cubin.sh, prefill variant.
        IsaLevel::Sm90a => ObjectRecipe {
            isa,
            profile: ProfileId::PrefillDense,
            source: "runtime/nvidia/interp_sm90a.cu",
            includes: NV_INCLUDES,
            // interp_sm90a.cu hard-defines PLOW_NV_HOPPER=1 (`:16`), so the
            // object's dense GEMM is the wgmma body `d_gemm_sm90`, not the
            // Ampere `d_gemm`. Declaring it here is a benign identical
            // redefinition for the build, and it is REQUIRED for the tile
            // probe: op_gemm_sm90.cuh #errors without it, so probing PGM90_*
            // needs the flag on the synthetic macro-probe TU too.
            defines: &[
                "PLOW_NV_GEMMA=1",
                "PLOW_NV_FA_GF=2",
                "PLOW_NV_EMBED_SMEM=1",
                "PLOW_NV_PREFILL=1",
                "PLOW_NV_HOPPER=1",
            ],
            arch_flag: "-arch=sm_90a",
            compiler: "/usr/local/cuda/bin/nvcc",
            dispatch_fn: "plow_exec",
            // The tile the object actually executes: the wgmma body's fixed
            // 128x128x64 (bf16) in op_gemm_sm90.cuh, NOT the Ampere PGM_*
            // triple, which under a Hopper build only feeds the dead #else arm.
            tile_macros: Some(("op_gemm_sm90.cuh", ["PGM90_BM", "PGM90_BN", "PGM90_BK"])),
        },
        // scripts/build_sm120_cubin.sh, prefill variant.
        IsaLevel::Sm120a => ObjectRecipe {
            isa,
            profile: ProfileId::PrefillDense,
            source: "runtime/nvidia/interp_sm120.cu",
            includes: NV_INCLUDES,
            defines: &[
                "PLOW_NV_GEMMA=1",
                "PLOW_NV_FA_GF=2",
                "PLOW_NV_EMBED_SMEM=1",
                "PLOW_NV_PREFILL=1",
            ],
            arch_flag: "-arch=sm_120a",
            compiler: "/usr/local/cuda/bin/nvcc",
            dispatch_fn: "plow_exec",
            tile_macros: Some(("op_gemm.cuh", ["PGM_BM", "PGM_BN", "PGM_BK"])),
        },
        // scripts/build_gfx950.sh, interp_prefill.elf.
        IsaLevel::Gfx950 => ObjectRecipe {
            isa,
            profile: ProfileId::PrefillDense,
            source: "runtime/amd/interp.hip",
            includes: AMD_INCLUDES,
            // ONLY the flag the build script passes. `PLOW_BUCKET_PREFILL` is
            // derived in the source (`interp.hip:71`,
            // `#define PLOW_BUCKET_PREFILL (!PLOW_BUCKET_DECODE)`), so passing
            // it here as well would collide with that definition. A recipe that
            // over-specifies is as wrong as one that under-specifies: it stops
            // describing the object the build actually produces.
            defines: &["PLOW_BUCKET_DECODE=0"],
            arch_flag: "--offload-arch=gfx950",
            compiler: "hipcc",
            dispatch_fn: "plow_exec",
            // AMD's tiles are per-opcode compile-time instantiations rather than
            // one object-wide macro triple, so they are read per kernel below.
            tile_macros: None,
        },
        _ => return None,
    })
}

/// AMD's three GEMM tiles are separate instantiations, each with its own macro
/// triple in `runtime/amd/op_gemm.h`.
const GFX950_TILE_MACROS: [(DevOp, [&str; 3]); 3] = [
    (DevOp::Gemm, ["GM_BM", "GM_BN", "GM_BK"]),
    (DevOp::GemmMed, ["GM_MD_BM", "GM_MD_BN", "GM_MD_BK"]),
    (DevOp::GemmSmall, ["GM_SM_BM", "GM_SM_BN", "GM_SM_BK"]),
];

/// Derive the dense-GEMM inventory for one ISA by probing its prefill object.
///
/// Every kernel returned was found in the built object, and the tile attached to
/// it was expanded from that object's macros. Nothing here is asserted.
pub fn dense_gemm_inventory(root: &Path, isa: IsaLevel) -> Result<Inventory, ProbeError> {
    let recipe = prefill_recipe(isa).ok_or_else(|| ProbeError::CompilerMissing {
        program: format!("no interpreter recipe for {}", isa.arch_flag()),
    })?;
    let target = recipe.target(root);
    let obj = probe(&target, isa, toolchain_label(&recipe))?;

    let gemm_ops = [DevOp::Gemm, DevOp::GemmMed, DevOp::GemmSmall];
    let mut specs = Vec::new();

    match recipe.tile_macros {
        // NVIDIA: one body, one object-wide tile. Every dispatched tile opcode
        // reaches it, so they share an implementation hash and the inventory
        // reports them as aliases rather than as three choices.
        Some((header, names)) => {
            let vals = probe_macros(&target, header, &names)?;
            let tile = (vals[0], vals[1], vals[2]);
            // Hopper delegates d_gemm -> d_gemm_sm90 (op_gemm.cuh:512), so the
            // body that runs under sm_90a is the wgmma one; record it as such.
            let body_fn = if isa == IsaLevel::Sm90a { "d_gemm_sm90" } else { "d_gemm" };
            let body = format!("{}:{}@{}", isa.arch_flag(), body_fn, obj.build().label());
            for op in gemm_ops {
                if !obj.dispatches(op) {
                    continue;
                }
                let (Some(bm), Some(bn), Some(bk)) = tile else { continue };
                specs.push(KernelSpec::gemm_tile(op, isa, bm, bn, bk, &body));
            }
        }
        // AMD: three separately compiled instantiations, three distinct bodies.
        None => {
            for (op, names) in GFX950_TILE_MACROS {
                if !obj.dispatches(op) {
                    continue;
                }
                let vals = probe_macros(&target, "op_gemm.h", &names)?;
                let (Some(bm), Some(bn), Some(bk)) = (vals[0], vals[1], vals[2]) else { continue };
                let body = format!("{}:{}@{}", isa.arch_flag(), op.c_name(), obj.build().label());
                specs.push(KernelSpec::gemm_tile(op, isa, bm, bn, bk, &body));
            }
        }
    }

    Ok(Inventory::probed(obj.build().clone(), specs))
}

fn toolchain_label(recipe: &ObjectRecipe) -> &'static str {
    // Recorded rather than detected: a probe that silently accepted whatever
    // compiler happened to be on PATH would key its inventory to the wrong
    // build. Detection belongs in the tuning campaign, which knows the
    // deployment.
    match recipe.isa {
        IsaLevel::Gfx942 | IsaLevel::Gfx950 => "rocm",
        _ => "cuda",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// The recipes must name files that exist, or the first probe on a fresh
    /// machine fails for a reason unrelated to capability.
    #[test]
    fn every_recipe_points_at_a_real_source() {
        for isa in [IsaLevel::Sm90a, IsaLevel::Sm120a, IsaLevel::Gfx950] {
            let r = prefill_recipe(isa).expect("recipe");
            let src = root().join(r.source);
            assert!(src.exists(), "{} missing for {}", r.source, isa.arch_flag());
            for inc in r.includes {
                assert!(root().join(inc).exists(), "include {inc} missing");
            }
        }
    }

    /// Hardware with no recipe returns None rather than borrowing another ISA's.
    #[test]
    fn undeclared_hardware_has_no_recipe() {
        assert!(prefill_recipe(IsaLevel::CpuRef).is_none());
        assert!(prefill_recipe(IsaLevel::Sm89).is_none());
    }

    /// The prefill recipes must actually request prefill, or the probe returns
    /// the decode object's arms and the tiled GEMM silently disappears.
    #[test]
    fn nvidia_prefill_recipes_enable_prefill() {
        for isa in [IsaLevel::Sm90a, IsaLevel::Sm120a] {
            let r = prefill_recipe(isa).unwrap();
            assert!(
                r.defines.iter().any(|d| d.starts_with("PLOW_NV_PREFILL")),
                "{} prefill recipe does not enable PLOW_NV_PREFILL",
                isa.arch_flag()
            );
        }
    }

    /// A recipe is only useful if it matches the build script it mirrors, and
    /// that can be checked **without the vendor toolchain** — which matters,
    /// because the recipe for a GPU nobody has locally is exactly the one most
    /// likely to rot.
    ///
    /// Guards the specific mistake this test was written after making: the AMD
    /// recipe originally also passed `PLOW_BUCKET_PREFILL=1`, which the source
    /// derives from `PLOW_BUCKET_DECODE` (`interp.hip:71`). Over-specifying is
    /// as wrong as under-specifying — either way the recipe stops describing
    /// the object the build produces.
    #[test]
    fn recipe_defines_are_not_derived_in_the_source() {
        for isa in [IsaLevel::Sm90a, IsaLevel::Sm120a, IsaLevel::Gfx950] {
            let r = prefill_recipe(isa).unwrap();
            let src = std::fs::read_to_string(root().join(r.source)).expect("source");
            // Follow a one-level `#include "x"` of a sibling TU, since
            // interp_sm90a.cu is a wrapper around interp_sm120.cu.
            let mut text = src.clone();
            for line in src.lines() {
                if let Some(rest) = line.trim().strip_prefix("#include \"") {
                    if let Some(name) = rest.split('"').next() {
                        let sib = root().join(r.source).parent().unwrap().join(name);
                        if let Ok(more) = std::fs::read_to_string(&sib) {
                            text.push_str(&more);
                        }
                    }
                }
            }
            for d in r.defines {
                let key = d.split('=').next().unwrap();
                let derived = format!("#define {key} (");
                assert!(
                    !text.contains(&derived),
                    "{}: recipe passes -D{key}, but the source derives it ({derived}...). \
                     Passing it collides with that definition.",
                    isa.arch_flag()
                );
            }
        }
    }

    /// The recipe must name the dispatch function the source actually defines,
    /// or the probe reports "no dispatch function" on a machine that has the
    /// toolchain — a confusing failure for a correct setup.
    #[test]
    fn recipe_dispatch_fn_exists_in_the_source() {
        for isa in [IsaLevel::Sm90a, IsaLevel::Sm120a, IsaLevel::Gfx950] {
            let r = prefill_recipe(isa).unwrap();
            let mut text = std::fs::read_to_string(root().join(r.source)).expect("source");
            for line in text.clone().lines() {
                if let Some(rest) = line.trim().strip_prefix("#include \"") {
                    if let Some(name) = rest.split('"').next() {
                        let sib = root().join(r.source).parent().unwrap().join(name);
                        if let Ok(more) = std::fs::read_to_string(&sib) {
                            text.push_str(&more);
                        }
                    }
                }
            }
            assert!(
                text.contains(&format!("void {}(", r.dispatch_fn)),
                "{}: no definition of {} in {}",
                isa.arch_flag(),
                r.dispatch_fn,
                r.source
            );
        }
    }

    /// AMD's per-opcode tile macros must exist, or the probe expands nothing and
    /// the inventory silently loses its tiles. Checked by reading the header,
    /// so it holds on a machine with no ROCm.
    #[test]
    fn amd_tile_macros_exist_in_the_header() {
        let hdr = std::fs::read_to_string(root().join("runtime/amd/op_gemm.h"))
            .expect("runtime/amd/op_gemm.h");
        for (op, names) in GFX950_TILE_MACROS {
            for n in names {
                assert!(
                    hdr.contains(&format!("#define {n} ")),
                    "{:?} names {n}, which op_gemm.h does not define",
                    op
                );
            }
        }
    }

    /// Probing needs the vendor toolchain. Without it the failure must name the
    /// compiler, not look like "this hardware has no kernels".
    #[test]
    fn a_missing_toolchain_is_an_explicit_failure() {
        let err = dense_gemm_inventory(&root(), IsaLevel::Gfx950);
        if let Err(e) = err {
            let msg = e.to_string();
            assert!(
                msg.contains("hipcc") || msg.contains("preprocess") || msg.contains("built object"),
                "unhelpful error: {msg}"
            );
        }
        // If hipcc IS present this returns Ok, which is equally correct.
    }

    /// End to end on whatever NVIDIA toolchain this machine has: the inventory
    /// must be derived, carry provenance, and report the alias collapse.
    #[test]
    fn probed_nvidia_inventory_reports_the_alias_collapse() {
        let Ok(inv) = dense_gemm_inventory(&root(), IsaLevel::Sm90a) else {
            eprintln!("skipping: no CUDA toolchain");
            return;
        };
        assert!(!inv.is_empty(), "the prefill object dispatches tiled GEMM");
        assert_eq!(inv.build().isa, IsaLevel::Sm90a);

        let groups = inv.alias_groups();
        assert_eq!(groups.len(), 1, "one shared d_gemm body");
        assert_eq!(
            groups.values().next().unwrap().len(),
            inv.len(),
            "every dispatched tile opcode reaches it"
        );

        // The tile came from the object's macros, not from a table.
        let t = inv.iter().next().unwrap().tile.expect("tile");
        assert!(t.bm > 0 && t.bn > 0 && t.bk > 0);
    }
}

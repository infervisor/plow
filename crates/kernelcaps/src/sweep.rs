//! Which build macros a tuner may actually sweep.
//!
//! A tuning campaign that varies a macro the source hard-defines does not
//! retune anything: it produces a macro redefinition, and depending on the
//! compiler either a warning that is lost in build noise or an error that looks
//! like the recipe is broken. Either way the campaign burns GPU time and
//! reports numbers for a tile that was never built.
//!
//! The distinction is visible in the header:
//!
//! ```c
//! #define PGM_BM 128                 // fixed: -DPGM_BM=256 collides
//! #ifndef PGM_BN
//! #define PGM_BN 128                 // overridable: -DPGM_BN=64 works
//! #endif
//! static_assert(PGM_BK8 == 64, ...)  // locked: the mainloop depends on it
//! ```
//!
//! so it can be answered without a toolchain and without a GPU — which matters
//! for a target nobody can build locally.

/// Whether a macro can be varied from the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sweepable {
    /// `#ifndef NAME` guards the definition. `-DNAME=…` takes effect.
    Overridable,
    /// A bare `#define NAME`. Passing `-DNAME=…` is a redefinition, not a
    /// retune.
    Fixed,
    /// Overridable in form, but a `static_assert` pins the value. Changing it
    /// fails the build, which is the honest outcome — the body depends on it.
    Asserted,
    /// Not defined in the text given.
    Absent,
}

impl Sweepable {
    pub fn can_sweep(self) -> bool {
        matches!(self, Sweepable::Overridable)
    }
}

/// Classify one macro in a header's text.
pub fn classify(header_text: &str, name: &str) -> Sweepable {
    let define = format!("#define {name} ");
    let define_alt = format!("#define {name}(");
    let has_define = header_text.contains(&define) || header_text.contains(&define_alt);
    if !has_define {
        return Sweepable::Absent;
    }

    // A `static_assert` naming the macro pins it regardless of the guard.
    for line in header_text.lines() {
        let l = line.trim_start();
        if (l.starts_with("static_assert") || l.starts_with("PLOW_SASSERT"))
            && mentions_ident(l, name)
        {
            return Sweepable::Asserted;
        }
    }

    // `#ifndef NAME` anywhere before the definition makes it overridable.
    let guard = format!("#ifndef {name}");
    if let Some(g) = header_text.find(&guard) {
        if let Some(d) = header_text
            .find(&define)
            .or_else(|| header_text.find(&define_alt))
        {
            if g < d {
                return Sweepable::Overridable;
            }
        }
    }
    // `#if !defined(NAME)` is the same guard spelled differently.
    for spelling in [
        format!("#if !defined({name})"),
        format!("#if !defined ({name})"),
    ] {
        if let Some(g) = header_text.find(&spelling) {
            if let Some(d) = header_text.find(&define) {
                if g < d {
                    return Sweepable::Overridable;
                }
            }
        }
    }
    Sweepable::Fixed
}

/// Whether `line` mentions `name` as a whole identifier.
fn mentions_ident(line: &str, name: &str) -> bool {
    let b = line.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = line[from..].find(name) {
        let at = from + rel;
        let before_ok = at == 0 || !(b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_');
        let after = at + name.len();
        let after_ok = after >= b.len() || !(b[after].is_ascii_alphanumeric() || b[after] == b'_');
        if before_ok && after_ok {
            return true;
        }
        from = at + name.len();
    }
    false
}

/// A macro and what a tuner may do with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Knob {
    pub name: String,
    pub sweepable: Sweepable,
    /// Header the classification was read from.
    pub header: String,
}

/// Classify a list of macros against a header's text.
pub fn knobs(header_text: &str, header_name: &str, names: &[&str]) -> Vec<Knob> {
    names
        .iter()
        .map(|n| Knob {
            name: (*n).to_string(),
            sweepable: classify(header_text, n),
            header: header_name.to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn read(p: &str) -> Option<String> {
        std::fs::read_to_string(root().join(p)).ok()
    }

    const SAMPLE: &str = r#"
#define PGM_BM 128
#ifndef PGM_BN
#define PGM_BN 128
#endif
#define PGM_BK 32
#ifndef PGM_STAGES
#define PGM_STAGES 3
#endif
#define PGM_BK8 64
static_assert(PGM_BK8 == 64, "the mainloop reads two k32 subgroups per K-tile");
"#;

    #[test]
    fn distinguishes_guarded_from_hard_defines() {
        assert_eq!(classify(SAMPLE, "PGM_BM"), Sweepable::Fixed);
        assert_eq!(classify(SAMPLE, "PGM_BK"), Sweepable::Fixed);
        assert_eq!(classify(SAMPLE, "PGM_BN"), Sweepable::Overridable);
        assert_eq!(classify(SAMPLE, "PGM_STAGES"), Sweepable::Overridable);
        assert_eq!(classify(SAMPLE, "PGM_NOT_THERE"), Sweepable::Absent);
    }

    /// A `static_assert` outranks the guard: the value is pinned even if the
    /// definition looks overridable.
    #[test]
    fn a_static_assert_pins_a_macro() {
        assert_eq!(classify(SAMPLE, "PGM_BK8"), Sweepable::Asserted);
        assert!(!Sweepable::Asserted.can_sweep());
    }

    #[test]
    fn only_overridable_is_sweepable() {
        assert!(Sweepable::Overridable.can_sweep());
        assert!(!Sweepable::Fixed.can_sweep());
        assert!(!Sweepable::Absent.can_sweep());
    }

    /// A substring must not be mistaken for the macro: `PGM_BK` and `PGM_BK8`
    /// are different knobs and misclassifying one as the other would either
    /// hide a real axis or invent an impossible one.
    #[test]
    fn identifier_matching_is_not_substring_matching() {
        assert!(mentions_ident(
            "static_assert(PGM_BK8 == 64, \"x\");",
            "PGM_BK8"
        ));
        assert!(!mentions_ident(
            "static_assert(PGM_BK8 == 64, \"x\");",
            "PGM_BK"
        ));
    }

    /// The real NVIDIA GEMM header. This is the finding that motivated the
    /// module: the tile a tuner would most want to sweep is the one it cannot.
    #[test]
    fn nvidia_gemm_tile_is_mostly_not_sweepable() {
        let Some(h) = read("runtime/nvidia/op_gemm.cuh") else {
            eprintln!("skipping: op_gemm.cuh not found");
            return;
        };
        let k = knobs(
            &h,
            "op_gemm.cuh",
            &[
                "PGM_BM",
                "PGM_BN",
                "PGM_BK",
                "PGM_BK8",
                "PGM_STAGES",
                "PGM_GLU_STAGES",
            ],
        );
        for knob in &k {
            eprintln!("{:<16} {:?}", knob.name, knob.sweepable);
        }

        let by = |n: &str| k.iter().find(|x| x.name == n).unwrap().sweepable;
        // PX-13 made PGM_BM overridable: it was a bare `#define` with no `#ifndef`,
        // so `-DPGM_BM` had never reached any object. Sweepable now, by design.
        assert_eq!(by("PGM_BM"), Sweepable::Overridable);
        assert_eq!(by("PGM_BK"), Sweepable::Fixed, "PGM_BK is a hard #define");
        assert_eq!(by("PGM_BN"), Sweepable::Overridable);
        assert_eq!(by("PGM_STAGES"), Sweepable::Overridable);
        assert_eq!(by("PGM_GLU_STAGES"), Sweepable::Overridable);
        assert_eq!(
            by("PGM_BK8"),
            Sweepable::Asserted,
            "static_assert pins it to 64"
        );

        let sweepable: Vec<&str> = k
            .iter()
            .filter(|x| x.sweepable.can_sweep())
            .map(|x| x.name.as_str())
            .collect();
        assert_eq!(
            sweepable,
            vec!["PGM_BM", "PGM_BN", "PGM_STAGES", "PGM_GLU_STAGES"]
        );
    }

    /// The GEMV knobs are the opposite case: the header says they exist "for
    /// autotune", and they are all genuinely overridable.
    #[test]
    fn nvidia_gemv_knobs_are_sweepable() {
        let Some(h) = read("runtime/nvidia/op_gemm.cuh") else {
            return;
        };
        for n in [
            "GV_UNROLL",
            "GV_UNROLL_GLU",
            "GV_MM_MAX",
            "GV_UN16",
            "GV_UN32",
        ] {
            assert_eq!(
                classify(&h, n),
                Sweepable::Overridable,
                "{n} should be tunable"
            );
        }
        // GV_STEP is the vector width and is NOT a knob.
        assert_eq!(classify(&h, "GV_STEP"), Sweepable::Fixed);
    }

    /// The Hopper wgmma GEMM body (op_gemm_sm90.cuh, dispatched under sm_90a)
    /// has its own knob set, disjoint from the Ampere PGM_* one. A campaign that
    /// swept PGM_STAGES on a Hopper object would retune the dead #else arm, so
    /// the tuner must know these are the real knobs — and their real limits.
    #[test]
    fn hopper_wgmma_gemm_knobs() {
        let Some(h) = read("runtime/nvidia/op_gemm_sm90.cuh") else {
            return;
        };
        // The pipeline-depth knobs are #ifndef-guarded but bounded by arena
        // static_asserts (`:79-83`): raising them past what the smem arena holds
        // fails the build, so the classifier reports Asserted, not a free knob.
        // A sweep may still lower them — the honest signal is "there is a limit".
        assert_eq!(classify(&h, "PGM90_STAGES"), Sweepable::Asserted);
        assert_eq!(classify(&h, "PGM90_GLU_STAGES"), Sweepable::Asserted);
        // The only free toggle: the two-level fp8 shadow accumulator (0/1), not
        // named by any static_assert.
        assert_eq!(classify(&h, "PGM90_FP8_PROMOTE"), Sweepable::Overridable);
        // Fixed: the tile is pinned to the wgmma m64n128 / 128 B swizzle shape.
        assert_eq!(classify(&h, "PGM90_BM"), Sweepable::Fixed);
        assert_eq!(classify(&h, "PGM90_BN"), Sweepable::Fixed);
        assert_eq!(classify(&h, "PGM90_BK"), Sweepable::Fixed);
        assert_eq!(classify(&h, "PGM90_BK8"), Sweepable::Fixed);
    }

    /// AMD's MoE toggles are overridable.
    #[test]
    fn amd_moe_toggles_are_sweepable() {
        let Some(h) = read("runtime/amd/op_moe.h") else {
            return;
        };
        assert_eq!(classify(&h, "PLOW_MOE_GROUP_FLAT"), Sweepable::Overridable);
        assert_eq!(classify(&h, "PLOW_MOE_MFMA"), Sweepable::Overridable);
    }

    /// Which axes of the NVIDIA dense GEMM tile a sweep can actually move.
    ///
    /// Split out of `amd_gemm_tile_is_sweepable_where_nvidia_is_not` so it stands
    /// alone: these assertions used to sit behind a two-vendor `let-else`, so a
    /// missing `runtime/amd/op_gemm.h` silently skipped NVIDIA coverage as well.
    ///
    /// M became a knob in PX-13. `PGM_BM` used to be a bare `#define`, so
    /// `-DPGM_BM=...` was a macro redefinition rather than a retune and a tuner
    /// that swept it would report results for a tile it never built. It is
    /// `#ifndef`-guarded now, so M and N sweep and only K is pinned.
    #[test]
    fn nvidia_gemm_tile_sweeps_m_and_n_but_not_k() {
        let Some(nv) = read("runtime/nvidia/op_gemm.cuh") else {
            return;
        };
        assert_eq!(classify(&nv, "PGM_BM"), Sweepable::Overridable); // PX-13
        assert_eq!(classify(&nv, "PGM_BN"), Sweepable::Overridable);
        assert_eq!(classify(&nv, "PGM_BK"), Sweepable::Fixed);
    }

    /// The vendors differ on the axis a tuner most wants, and the difference is
    /// not cosmetic.
    ///
    /// AMD guards its GEMM tile, and the tree already exploits that:
    /// `scripts/build_gfx950_qwen.sh:29` ships `-DGM_BM=192`. NVIDIA hard-defined
    /// `PGM_BM` until PX-13 guarded it, so the same sweep there WAS a macro
    /// redefinition rather than a retune. Both M axes sweep now; the K axis still
    /// sweeps on neither.
    ///
    /// This test previously asserted `GM_BM` was Fixed, which was simply wrong.
    #[test]
    fn amd_gemm_tile_is_sweepable_where_nvidia_is_not() {
        let (Some(amd), Some(nv)) = (
            read("runtime/amd/op_gemm.h"),
            read("runtime/nvidia/op_gemm.cuh"),
        ) else {
            return;
        };

        // AMD guards BM and BN; `-DGM_BM=192` is shipped by
        // scripts/build_gfx950_qwen.sh:29.
        assert_eq!(classify(&amd, "GM_BM"), Sweepable::Overridable);
        assert_eq!(classify(&amd, "GM_BN"), Sweepable::Overridable);
        // NVIDIA guards it too since PX-13.
        assert_eq!(classify(&nv, "PGM_BM"), Sweepable::Overridable);
        assert_eq!(classify(&nv, "PGM_BN"), Sweepable::Overridable);

        // And the K axis is fixed on BOTH vendors -- GM_BK sits after the
        // #endif that closes GM_BN's guard, so it is a bare #define. A K sweep
        // is not available anywhere, which is worth knowing before designing
        // one.
        assert_eq!(classify(&amd, "GM_BK"), Sweepable::Fixed);
        assert_eq!(classify(&nv, "PGM_BK"), Sweepable::Fixed);

        assert_eq!(classify(&amd, "GV_UNROLL"), Sweepable::Overridable);
    }
}

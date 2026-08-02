//! Probe the real interpreter objects and check what comes back.
//!
//! Skips when the toolchain is absent, matching `packet/tests/dev_abi.rs` —
//! these tests need the compiler that builds the object, because an inventory
//! is derived from a build and there is no other way to obtain one.
//!
//! What this pins down is the claim the whole crate rests on: **capability is a
//! property of a build, not of the source tree.** The decode and prefill objects
//! come from the same translation unit and differ only by `-DPLOW_NV_PREFILL=1`,
//! and they must not report the same kernels.

use std::path::PathBuf;

use kernelcaps::probe::{dispatched_opcodes, probe_macros, ProbeTarget};
use kernelcaps::{probe, IsaLevel};
use packet::dev::DevOp;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn nvcc() -> Option<String> {
    let p = "/usr/local/cuda/bin/nvcc";
    std::path::Path::new(p).exists().then(|| p.to_string())
}

/// The sm90a interpreter, as `scripts/build_sm90a_cubin.sh` builds it.
fn sm90a_target(extra: &[&str]) -> Option<ProbeTarget> {
    let root = repo_root();
    let source = root.join("runtime/nvidia/interp_sm90a.cu");
    if !source.exists() {
        return None;
    }
    // interp_sm90a.cu hard-defines PLOW_NV_HOPPER=1; declaring it matches the
    // real object and lets the macro probe include op_gemm_sm90.cuh (which
    // #errors without it).
    let mut defines = vec![
        "PLOW_NV_GEMMA=1".to_string(),
        "PLOW_NV_FA_GF=2".to_string(),
        "PLOW_NV_EMBED_SMEM=1".to_string(),
        "PLOW_NV_HOPPER=1".to_string(),
    ];
    defines.extend(extra.iter().map(|s| s.to_string()));

    Some(ProbeTarget {
        compiler: nvcc()?,
        arch_flag: "-arch=sm_90a".into(),
        includes: vec![
            root.join("runtime/common").to_string_lossy().into_owned(),
            root.join("runtime/nvidia").to_string_lossy().into_owned(),
        ],
        defines,
        source: source.to_string_lossy().into_owned(),
        dispatch_fn: "plow_exec".into(),
    })
}

#[test]
fn probes_the_decode_object() {
    let Some(t) = sm90a_target(&[]) else {
        eprintln!("skipping: no CUDA toolchain or interpreter source");
        return;
    };
    let obj = probe(&t, IsaLevel::Sm90a, "cuda-13.0").expect("probe the decode object");

    assert!(
        obj.opcodes.len() > 20,
        "a real interpreter dispatches many opcodes, got {}",
        obj.opcodes.len()
    );

    // Decode-path opcodes that must be present in any build of this object.
    for op in [DevOp::Nop, DevOp::RmsNorm, DevOp::Gemv, DevOp::FlashDecode] {
        assert!(obj.dispatches(op), "{op:?} missing from the decode object");
    }

    // Provenance is not optional: the inventory names the build it came from.
    assert_eq!(obj.build().isa, IsaLevel::Sm90a);
    assert_eq!(obj.build().toolchain, "cuda-13.0");
    assert!(obj.build().defines.contains(&"PLOW_NV_GEMMA".to_string()));
}

/// The load-bearing test. Same source file, one extra `-D`, different kernels.
/// If this ever passes with equal sets, reading capability off the source would
/// have been fine and this crate is unnecessary.
#[test]
fn prefill_and_decode_objects_expose_different_kernels() {
    let (Some(dec), Some(pre)) = (sm90a_target(&[]), sm90a_target(&["PLOW_NV_PREFILL=1"])) else {
        eprintln!("skipping: no CUDA toolchain or interpreter source");
        return;
    };

    let d = probe(&dec, IsaLevel::Sm90a, "cuda-13.0").expect("decode probe");
    let p = probe(&pre, IsaLevel::Sm90a, "cuda-13.0").expect("prefill probe");

    assert_ne!(
        d.opcodes, p.opcodes,
        "the same TU under different -D must not report the same kernels"
    );
    assert_ne!(d.build(), p.build(), "different flags are different builds");

    // The tiled prefill GEMM is gated on PLOW_NV_PREFILL, so it appears in one
    // and not the other. This is the concrete reason a source grep is wrong.
    assert!(
        p.dispatches(DevOp::Gemm) && !d.dispatches(DevOp::Gemm),
        "PLOW_DOP_GEMM should be prefill-only: decode={} prefill={}",
        d.dispatches(DevOp::Gemm),
        p.dispatches(DevOp::Gemm)
    );
}

/// On NVIDIA the three tile opcodes reach one body, so a build that has any of
/// them has all three. Their tile is a macro of the object, probed separately.
#[test]
fn nvidia_tile_opcodes_appear_together() {
    let Some(t) = sm90a_target(&["PLOW_NV_PREFILL=1"]) else {
        eprintln!("skipping: no CUDA toolchain or interpreter source");
        return;
    };
    let obj = probe(&t, IsaLevel::Sm90a, "cuda-13.0").expect("prefill probe");

    let present: Vec<bool> = [DevOp::Gemm, DevOp::GemmMed, DevOp::GemmSmall]
        .iter()
        .map(|op| obj.dispatches(*op))
        .collect();
    assert!(
        present.iter().all(|p| *p) || present.iter().all(|p| !*p),
        "the aliased triple must be all-or-nothing, got {present:?}"
    );
}

/// The tile is a compile-time constant of the object. Expanding it through the
/// same preprocessor run is how the inventory learns the tile it will actually
/// execute, rather than assuming one.
#[test]
fn expands_the_objects_tile_macros() {
    let Some(t) = sm90a_target(&["PLOW_NV_PREFILL=1"]) else {
        eprintln!("skipping: no CUDA toolchain or interpreter source");
        return;
    };
    // The sm_90a object runs the wgmma body d_gemm_sm90, whose tile is the
    // fixed PGM90_* triple in op_gemm_sm90.cuh (128x128x64), not the Ampere
    // PGM_* one. Probe the tile it actually executes.
    let vals = probe_macros(
        &t,
        "op_gemm_sm90.cuh",
        &["PGM90_BM", "PGM90_BN", "PGM90_BK"],
    )
    .expect("expand tile macros");

    for (name, v) in ["PGM90_BM", "PGM90_BN", "PGM90_BK"].iter().zip(&vals) {
        let v = v.unwrap_or_else(|| panic!("{name} did not expand to an integer"));
        assert!(
            v > 0 && v <= 1024,
            "{name} = {v} is not a plausible tile dimension"
        );
        assert!(v % 8 == 0, "{name} = {v} is not MMA-shaped");
    }
    eprintln!("probed tile: {:?}", vals);
}

/// Re-probing the same object must give the same answer, or the inventory is
/// not reproducible and committing it is meaningless.
#[test]
fn probing_is_deterministic() {
    let Some(t) = sm90a_target(&[]) else {
        eprintln!("skipping: no CUDA toolchain or interpreter source");
        return;
    };
    let a = probe(&t, IsaLevel::Sm90a, "cuda-13.0").expect("first probe");
    let b = probe(&t, IsaLevel::Sm90a, "cuda-13.0").expect("second probe");
    assert_eq!(a.opcodes, b.opcodes);
    assert_eq!(a.build(), b.build());
}

/// Every opcode the probe reports must be one the Rust ABI knows, since the
/// compiler can only emit those.
#[test]
fn probed_opcodes_are_all_known_to_the_abi() {
    let Some(t) = sm90a_target(&["PLOW_NV_PREFILL=1"]) else {
        eprintln!("skipping: no CUDA toolchain or interpreter source");
        return;
    };
    let obj = probe(&t, IsaLevel::Sm90a, "cuda-13.0").expect("probe");
    let known: Vec<u16> = DevOp::ALL.iter().map(|o| *o as u16).collect();
    for op in &obj.opcodes {
        assert!(
            known.contains(op),
            "probed opcode {op} has no DevOp variant"
        );
    }
}

/// A source file that does not exist is an error, not an empty inventory.
/// Silently returning "no kernels" would read as "this hardware supports
/// nothing" and send selection down a fallback path.
#[test]
fn a_missing_source_is_an_error() {
    let Some(mut t) = sm90a_target(&[]) else {
        return;
    };
    t.source = "/nonexistent/interp.cu".into();
    assert!(probe(&t, IsaLevel::Sm90a, "cuda-13.0").is_err());
}

#[test]
fn dispatch_fn_must_exist() {
    let Some(t) = sm90a_target(&[]) else { return };
    let text = t.preprocess().expect("preprocess");
    assert!(dispatched_opcodes(&text, "no_such_function_xyz").is_err());
}

/// Alias detection, derived rather than asserted.
///
/// The NVIDIA GEMM triple is the known case: three opcodes falling through to
/// one `d_gemm`. Reading it out of the object means the same parser finds
/// aliasing in every other family too, without anyone writing a table.
#[test]
fn derives_the_gemm_alias_group_from_the_object() {
    let Some(t) = sm90a_target(&["PLOW_NV_PREFILL=1"]) else {
        eprintln!("skipping: no CUDA toolchain");
        return;
    };
    let text = t.preprocess().expect("preprocess");
    let arms = kernelcaps::probe::dispatch_arms(&text, "plow_exec").expect("arms");

    assert!(
        arms.len() > 15,
        "a real interpreter has many arms, got {}",
        arms.len()
    );

    let gemm = arms
        .iter()
        .find(|a| a.opcodes.contains(&(DevOp::Gemm as u16)))
        .expect("an arm dispatching PLOW_DOP_GEMM");

    assert_eq!(gemm.callee, "d_gemm", "the tiled GEMM body");
    let mut got = gemm.opcodes.clone();
    got.sort();
    let mut want = vec![
        DevOp::Gemm as u16,
        DevOp::GemmMed as u16,
        DevOp::GemmSmall as u16,
    ];
    want.sort();
    assert_eq!(got, want, "all three tile opcodes share one body");
}

/// Every arm must name a plausible device function, or the parser is picking up
/// casts and macro noise instead of calls.
#[test]
fn every_derived_arm_names_a_callee() {
    let Some(t) = sm90a_target(&["PLOW_NV_PREFILL=1"]) else {
        return;
    };
    let text = t.preprocess().expect("preprocess");
    let arms = kernelcaps::probe::dispatch_arms(&text, "plow_exec").expect("arms");

    for a in &arms {
        assert!(!a.opcodes.is_empty(), "an arm with no opcodes: {a:?}");
        assert_ne!(
            a.callee, "__trap",
            "opcodes {:?} resolved to __trap; the parser walked past the real body \
             (template calls like d_flash_prefill_mux<256,64,32> are the usual cause)",
            a.opcodes
        );
        // An empty callee is legitimate: PLOW_DOP_NOP is `case ...: break;`,
        // a real arm that does nothing.
        assert!(
            a.callee.is_empty()
                || a.callee.starts_with("d_")
                || a.callee.starts_with("exec_")
                || a.callee.contains("plow"),
            "implausible callee {:?} for opcodes {:?}",
            a.callee,
            a.opcodes
        );
    }
}

/// The interpreter expresses shape specialization as template arguments, so the
/// probe gets attention tiles and head-dim coverage for free -- data a tuner
/// would otherwise have to be told.
#[test]
fn recovers_shape_specializations_from_template_arguments() {
    let Some(t) = sm90a_target(&["PLOW_NV_PREFILL=1"]) else {
        return;
    };
    let text = t.preprocess().expect("preprocess");
    let arms = kernelcaps::probe::dispatch_arms(&text, "plow_exec").expect("arms");

    let flash = arms
        .iter()
        .find(|a| a.opcodes.contains(&(DevOp::FlashPrefill as u16)))
        .expect("a FLASH_PREFILL arm");

    assert!(
        flash.callee.contains("flash_prefill"),
        "callee {:?} is not the flash body",
        flash.callee
    );
    assert!(
        flash.specializations.len() >= 2,
        "flash prefill specializes on head_dim; got {:?}",
        flash.specializations
    );
    for s in &flash.specializations {
        assert_eq!(s.len(), 3, "expected <head_dim, bq, bkv>, got {s:?}");
        for a in s {
            assert!(a.parse::<i64>().is_ok(), "non-numeric tile arg {a:?}");
        }
    }
    eprintln!("flash specializations: {:?}", flash.specializations);

    let total: usize = arms.iter().map(|a| a.specializations.len()).sum();
    eprintln!("arms={} total specializations={}", arms.len(), total);
    assert!(
        total >= 5,
        "the object specializes in several places, found {total}"
    );
}

/// The arms must account for the same opcodes `dispatched_opcodes` reports, or
/// the two views of one object disagree.
#[test]
fn arms_and_opcode_set_agree() {
    let Some(t) = sm90a_target(&["PLOW_NV_PREFILL=1"]) else {
        return;
    };
    let text = t.preprocess().expect("preprocess");
    let arms = kernelcaps::probe::dispatch_arms(&text, "plow_exec").expect("arms");
    let ops = dispatched_opcodes(&text, "plow_exec").expect("opcodes");

    let from_arms: std::collections::BTreeSet<u16> = arms
        .iter()
        .flat_map(|a| a.opcodes.iter().copied())
        .collect();

    // Arms may miss an opcode whose segment has no call (a pure `break;` arm),
    // so arms are a subset; nothing may appear in arms that is not an opcode.
    for op in &from_arms {
        assert!(ops.contains(op), "arm reports {op}, not in the opcode set");
    }
    let missing: Vec<u16> = ops.difference(&from_arms).copied().collect();
    eprintln!(
        "arms={} opcodes={} callee-less={:?}",
        arms.len(),
        ops.len(),
        missing
    );
}

/// The parser must find alias groups nobody told it about.
///
/// The bf16 GEMM triple was known when this was written. The fp8 triple
/// (GEMM_FP8 / GEMM_MED_FP8 / GEMM_SMALL_FP8 -> one d_gemm_w8a8 body) was NOT:
/// it was found later by reading the interpreter. A derived parser should have
/// had it all along, which is the point of deriving rather than tabulating.
#[test]
fn finds_alias_groups_that_were_not_known_in_advance() {
    let Some(t) = sm90a_target(&["PLOW_NV_PREFILL=1", "PLOW_NV_W8A8=1"]) else {
        return;
    };
    let text = t.preprocess().expect("preprocess");
    let arms = kernelcaps::probe::dispatch_arms(&text, "plow_exec").expect("arms");

    let groups: Vec<&kernelcaps::probe::DispatchArm> =
        arms.iter().filter(|a| a.opcodes.len() > 1).collect();
    assert!(
        groups.len() >= 2,
        "expected at least the bf16 and fp8 GEMM triples, found {:?}",
        groups
            .iter()
            .map(|a| (&a.callee, &a.opcodes))
            .collect::<Vec<_>>()
    );

    let fp8 = arms
        .iter()
        .find(|a| a.opcodes.contains(&(DevOp::GemmFp8 as u16)))
        .expect("an arm dispatching PLOW_DOP_GEMM_FP8");
    let mut got = fp8.opcodes.clone();
    got.sort();
    let mut want = vec![
        DevOp::GemmFp8 as u16,
        DevOp::GemmMedFp8 as u16,
        DevOp::GemmSmallFp8 as u16,
    ];
    want.sort();
    assert_eq!(got, want, "the fp8 tile opcodes are one body too");
    assert!(fp8.callee.contains("w8a8"), "callee {:?}", fp8.callee);

    for g in &groups {
        eprintln!("alias group: {} <- {:?}", g.callee, g.opcodes);
    }
}

/// An opcode must appear in at most one arm. AMD gives `GEMV_FP8_BLK` two
/// mutually exclusive `case` labels under `#if PLOW_FP8` / `#else`; only one
/// survives preprocessing, but a parser that double-counted would report a
/// kernel twice and let a tuner rank it against itself.
#[test]
fn no_opcode_appears_in_two_arms() {
    for defs in [
        vec!["PLOW_NV_PREFILL=1"],
        vec!["PLOW_NV_PREFILL=1", "PLOW_NV_W8A8=1"],
        vec![],
    ] {
        let Some(t) = sm90a_target(&defs) else { return };
        let text = t.preprocess().expect("preprocess");
        let arms = kernelcaps::probe::dispatch_arms(&text, "plow_exec").expect("arms");

        let mut seen: std::collections::BTreeMap<u16, &str> = std::collections::BTreeMap::new();
        for a in &arms {
            for op in &a.opcodes {
                if let Some(prev) = seen.insert(*op, &a.callee) {
                    panic!("opcode {op} dispatched twice: {prev} and {}", a.callee);
                }
            }
        }
    }
}

//! The hwspec-driven picker is a STATIC, shape-agnostic choice — so it is testable
//! offline, with no GPU. These lock in the tile chosen for every projection of the three
//! supported architectures at the prefill chunk sizes that matter, proving the picker both
//! fills the CUs on the underutilized shapes AND does not regress the ones that already
//! saturate. `n_cu = 256` (MI350X).
use super::{
    amd_target, amd_tuning_cell, gemm_lds_bytes, gemm_lds_bytes_buffered, glu_era_inventory,
    hwspec, pick_tile, select_gemm_over, set_amd_target, stage_buffers, DevOp, GFX950_RUNGS,
    GFX950_TILES,
};
use costmodel::cost::{dma_cycles, macs_cycles};
use costmodel::MmaDtype;
use kernelcaps::QuantScheme;

const N_CU: u32 = 256;
fn pt(m: u32, n: u32, k: u32) -> DevOp {
    pick_tile(m, n, k, N_CU, QuantScheme::None)
}
/// The picker restricted to the three rungs that existed before the tile-inventory
/// campaign — the set the legacy reference below ranks over.
fn pt_legacy_rungs(m: u32, n: u32, k: u32, n_cu: u32) -> DevOp {
    select_gemm_over(glu_era_inventory(), m, n, k, n_cu, QuantScheme::None).0
}

/// The picker exactly as it was before selection moved behind the capability
/// registry: one loop over a constant table, first-match-wins on ties.
///
/// Kept as the differential reference. The assertions below pin the shapes
/// that were reasoned about by hand; this pins everything else, which is
/// what actually rules out a silent regression on some shape nobody listed.
fn pick_tile_legacy(m: u32, n: u32, k: u32, n_cu: u32) -> DevOp {
    let spec = hwspec::registry::lookup("MI350X").expect("gfx950 spec in registry");
    let lds_budget = spec.sm.shared_mem.0;
    let (m, n, k) = (m as u64, n as u64, k as u64);
    let n_cu = (n_cu as u64).max(1);

    let mut best = (DevOp::Gemm, u64::MAX);
    for (op, bm, bn, bk) in GFX950_TILES {
        if gemm_lds_bytes(bm, bn, bk) > lds_budget {
            continue;
        }
        let tiles = m.div_ceil(bm) * n.div_ceil(bn);
        let rounds = tiles.div_ceil(n_cu);
        let k_iters = k.div_ceil(bk);
        let compute = k_iters * macs_cycles(spec, bm * bn * bk, MmaDtype::Bf16);
        let dma = dma_cycles(spec, (bm * k + k * bn) * 2, false);
        let cost = rounds.saturating_mul(compute.max(dma));
        if cost < best.1 {
            best = (op, cost);
        }
    }
    best.0
}

const MS: [u32; 12] = [1, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
const NS: [u32; 12] = [
    128, 512, 1024, 2048, 2560, 4096, 5376, 8192, 9728, 14336, 16384, 21504,
];
const KS: [u32; 8] = [128, 512, 2560, 4096, 5376, 8192, 14336, 21504];
const CUS: [u32; 5] = [1, 64, 128, 256, 304];

/// Routing selection through the registry must not change a single answer **among the three
/// rungs the old picker had**. Swept rather than sampled: tie-breaking was the real risk,
/// since the old loop preferred the larger tile by table order while opcode order would put
/// `GemmSmall` (14) ahead of `GemmMed` (15).
///
/// Scoped to the legacy rungs deliberately. The campaign ADDED two tiles, so comparing the
/// full picker against a three-tile reference would assert the new rungs are never chosen —
/// the opposite of the intent. What must not drift is the *ranking rule*, and that is what
/// this pins.
#[test]
fn the_original_rungs_still_rank_exactly_as_the_legacy_picker_did() {
    let mut checked = 0usize;
    for &m in &MS {
        for &n in &NS {
            for &k in &KS {
                for &n_cu in &CUS {
                    let want = pick_tile_legacy(m, n, k, n_cu);
                    let got = pt_legacy_rungs(m, n, k, n_cu);
                    assert_eq!(
                        got, want,
                        "diverged at m={m} n={n} k={k} n_cu={n_cu}: \
                             registry chose {got:?}, legacy chose {want:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, MS.len() * NS.len() * KS.len() * CUS.len());
}

/// The added rungs must never be chosen where they cannot help, and the ranking must stay
/// TOTAL — no shape may resolve by opcode number.
///
/// The second half is the real content. `tile_cost`'s tie-break used three hand-written
/// brackets over `BM*BN`, and 192x256 (49152) and 128x256 (32768) both landed in the same
/// bracket as 256x256 (65536), so on any shape where their wall-clock costs tied — which is
/// every shape small enough that all three take one round — the winner would have been
/// decided by `DevOp` number. `GemmC5` is 95 and `Gemm` is 8, so the *old* tile would have
/// won silently and the campaign would have measured nothing.
#[test]
fn every_shape_resolves_on_cost_rather_than_opcode_number() {
    use super::{gfx950_gemm_inventory, tile_cost};
    let spec = hwspec::registry::lookup("MI350X").unwrap();
    for &m in &MS {
        for &n in &NS {
            for &k in &KS {
                for &n_cu in &CUS {
                    let chosen = pick_tile(m, n, k, n_cu, QuantScheme::None);
                    let costs: Vec<(DevOp, u64)> = gfx950_gemm_inventory()
                        .iter()
                        .filter(|s| s.quant == QuantScheme::None)
                        .map(|s| {
                            (
                                s.id.0,
                                tile_cost(spec, s, m as i64, n as i64, k as i64, n_cu),
                            )
                        })
                        .collect();
                    let best = costs.iter().map(|c| c.1).min().unwrap();
                    let ties: Vec<DevOp> =
                        costs.iter().filter(|c| c.1 == best).map(|c| c.0).collect();
                    assert_eq!(
                        ties.len(),
                        1,
                        "m={m} n={n} k={k} n_cu={n_cu}: {ties:?} tie at cost {best}, so the \
                             winner is decided by opcode number"
                    );
                    assert_eq!(chosen, ties[0], "m={m} n={n} k={k} n_cu={n_cu}");
                }
            }
        }
    }
}

/// `tile_cost` COSTS AGAINST ITS `spec` ARGUMENT, not against whatever the ambient
/// `amd_target` thread-local happens to hold.
///
/// The regression this locks in: `plowc tune select` looks its spec up from the hardware
/// fingerprint and never calls `set_amd_target`, so `spec` was MI300X (64 KiB LDS) while the
/// unset thread-local still answered the MI350X default -- stage buffers 2. Every rung except
/// 64x128 then failed the LDS filter and scored `u64::MAX`, and `tune select` reported
/// `PLOW_DOP_GEMM_SMALL` for EVERY gfx942 shape, including ones where the measured ladder puts
/// 64x128 at roughly HALF of 192x256 (8192x32768x2048: 247 vs 481 TF/s, measured on this box).
/// The emitters were never affected -- both entry points call `set_amd_target` first -- but the
/// command whose doc-comment promises "the SAME ranking the compiler uses" disagreed with it,
/// which is exactly the failure that comment exists to forbid.
///
/// Asserted with the thread-local pointed at the OTHER part, because that is the state the bug
/// needed: agreeing values cannot distinguish the two sources.
#[test]
fn tile_cost_costs_against_its_spec_not_the_ambient_target() {
    let mi300 = hwspec::registry::lookup("MI300X").expect("MI300X in registry");
    let c5 = kernelcaps::KernelSpec::gemm_tile(
        DevOp::GemmC5,
        hwspec::IsaLevel::Gfx942,
        192,
        256,
        64,
        "probe",
    );

    // Point the ambient target at the CDNA4 part, then cost a CDNA3 tile against the CDNA3
    // spec. 192x256 is 64,512 B single-buffered (fits gfx942) and 129,024 B double-buffered
    // (does not), so a thread-local read here scores it unavailable.
    set_amd_target("MI350X");
    let cost = crate::tile_cost(mi300, &c5, 8192, 32768, 2048, 304);
    assert_ne!(
        cost,
        u64::MAX,
        "192x256 fits gfx942's 64 KiB single-buffered; costing it as double-buffered because \
             the ambient target says MI350X is the `tune select` reporting bug"
    );

    // And the converse: the same tile against the CDNA4 spec must stay available too, so the
    // fix cannot have simply hardcoded one buffer count.
    let mi350 = hwspec::registry::lookup("MI350X").expect("MI350X in registry");
    set_amd_target("MI300X");
    assert_ne!(
        crate::tile_cost(mi350, &c5, 8192, 32768, 2048, 256),
        u64::MAX
    );
}

/// `--arch gfx942` WITHOUT A USABLE `--gpu` MUST NOT COST AGAINST gfx950.
///
/// `amd_target::set` only ever read `--gpu`, and its failure arms `eprintln!`d and returned,
/// leaving the MI350X/Gfx950 default in force. So an emit with `--arch gfx942` and an absent
/// or misspelled `--gpu` costed CDNA3 tiles against 160 KiB of LDS and double-rate MFMA and
/// resolved its tuning cell to `amd/gfx950/mi350x`. The canonical GLM line passes
/// `--gpu MI300X` and was never affected; nothing enforced that it had to.
///
/// Asserted from the gfx950 default in both directions, because the bug is invisible when
/// the fallback and the answer happen to agree.
#[test]
fn an_absent_or_unknown_gpu_falls_back_to_the_arch_not_to_gfx950() {
    for gpu in ["", "MI300", "H100"] {
        set_amd_target("MI350X");
        amd_target::set_for("gfx942", gpu);
        let (spec, isa) = amd_target::active();
        assert_eq!(
            isa,
            hwspec::IsaLevel::Gfx942,
            "--arch gfx942, --gpu {gpu:?}"
        );
        assert_eq!(spec.sm.shared_mem.0, 64 * 1024, "CDNA3 LDS, --gpu {gpu:?}");
        assert_eq!(amd_tuning_cell(), "amd/gfx942/mi300x", "--gpu {gpu:?}");
    }

    // A resolvable --gpu still wins over --arch: the SKU is finer than the arch, and it is
    // what the tuning cell is keyed by.
    amd_target::set_for("gfx942", "MI350X");
    assert_eq!(amd_target::active().1, hwspec::IsaLevel::Gfx950);

    // And with neither usable, the old MI350X default stands — warned about, not silent.
    set_amd_target("MI300X");
    amd_target::set_for("", "");
    assert_eq!(
        amd_target::active().1,
        hwspec::IsaLevel::Gfx942,
        "no arch and no gpu must not clobber a target already set"
    );
}

/// THE COMPILER READS THE CELL THE CAMPAIGN WRITES.
///
/// `amd_tuning_cell` (reader) and `tunedb-gemv` (writer) each had their own copy of the rule,
/// and the writer's was the constant `GFX950_CELL`. A gfx942 decode-GEMV campaign would have
/// published into `amd/gfx950/mi350x`, which this function never opens on a gfx942 compile —
/// and neither side errors, because a miss and an unmeasured shape produce identical bytes.
/// One rule now, in `tunedb`; this is the reader's half of the pin.
#[test]
fn the_tuning_cell_follows_the_target_and_matches_the_campaign_writer() {
    for (gpu, want) in [
        ("MI300X", "amd/gfx942/mi300x"),
        ("MI350X", tunedb::GFX950_CELL),
    ] {
        set_amd_target(gpu);
        assert_eq!(amd_tuning_cell(), want, "{gpu}");
        assert_eq!(
            amd_tuning_cell(),
            tunedb::amd_tuning_cell(amd_target::active().0),
            "{gpu}: reader and campaign writer must resolve the same cell"
        );
    }
}

/// THE EMITTER FOLLOWS `--gpu`, and the tile it picks is one the part can actually hold.
///
/// Both halves matter and both were broken. `select_gemm_over` opened with a hardcoded
/// `lookup("MI350X")`, so every AMD tile was costed against 160 KiB of LDS and double-rate
/// MFMA whatever the caller asked for; and `gemm_lds_bytes` hardcoded a double buffer, so
/// once CDNA3 went single-buffered the filter rejected the tiles the object can run.
#[test]
fn amd_tile_selection_follows_the_target_hwspec() {
    use kernelcaps::QuantScheme::None as Bf16;

    // Gemma-4 31B gate/up at a real prefill length.
    let (m, n, k, n_cu) = (4096u32, 21504u32, 5376u32, 304u32);

    let dims = |op: DevOp| -> Option<(i64, i64, i64)> {
        GFX950_RUNGS
            .iter()
            .find(|r| r.0 == op || r.1 == op || r.2 == op)
            .map(|r| (r.3, r.4, r.5))
    };

    set_amd_target("MI300X");
    let (spec3, isa3) = amd_target::active();
    assert_eq!(
        isa3,
        hwspec::IsaLevel::Gfx942,
        "MI300X must resolve to CDNA3"
    );
    assert_eq!(spec3.sm.shared_mem.0, 64 * 1024, "CDNA3 workgroup LDS");
    let cdna3 = pick_tile(m, n, k, n_cu, Bf16);
    let (bm, bn, bk) = dims(cdna3).expect("chosen tile is a known rung");

    // The tile the emitter picked must fit the part it picked it for, at the stage the
    // object is actually built with.
    let single = gemm_lds_bytes_buffered(bm as u64, bn as u64, bk as u64, 1);
    assert!(
        single <= spec3.sm.shared_mem.0,
        "picked {bm}x{bn}x{bk} = {single} B single-buffered, over CDNA3's 64 KiB"
    );

    set_amd_target("MI350X");
    let (spec4, isa4) = amd_target::active();
    assert_eq!(isa4, hwspec::IsaLevel::Gfx950);
    assert_eq!(spec4.sm.shared_mem.0, 160 * 1024, "CDNA4 workgroup LDS");
    let cdna4 = pick_tile(m, n, k, n_cu, Bf16);

    // The parts differ in LDS, MFMA rate and CU count, so the same shape need not land on
    // the same rung. What must hold is that each answer is legal on ITS OWN part -- the old
    // code gave CDNA4's answer for both.
    let (bm4, bn4, bk4) = dims(cdna4).expect("chosen tile is a known rung");
    assert!(
        gemm_lds_bytes_buffered(bm4 as u64, bn4 as u64, bk4 as u64, 2) <= spec4.sm.shared_mem.0,
        "CDNA4 tile {bm4}x{bn4}x{bk4} must fit 160 KiB double-buffered"
    );

    // And the buffer count is the thing that makes CDNA3's answer reachable at all: the
    // chosen tile is one a double-buffered filter would have thrown away on this part.
    set_amd_target("MI300X");
    assert_eq!(stage_buffers(hwspec::IsaLevel::Gfx942), 1);
    assert_eq!(stage_buffers(hwspec::IsaLevel::Gfx950), 2);
}

/// Precision changes the ANSWER, not just the opcode name.
///
/// Two things are asserted, and the second is the one that was broken: every encoding must
/// select from its OWN rungs (a bf16 opcode emitted for an mxfp4 op would read packed fp4
/// bytes as bf16), and the fp8/fp4 answer must be free to differ from bf16's — the fp8 body
/// moves half the operand bytes for the same MFMA count, so `max(compute, dma)` tips.
#[test]
fn each_encoding_selects_from_its_own_rungs() {
    for &m in &MS {
        for &n in &NS {
            for &k in &KS {
                for (quant, ok) in [
                    (QuantScheme::None, &super::GFX950_RUNGS.map(|r| r.0)),
                    (QuantScheme::W8A8, &super::GFX950_RUNGS.map(|r| r.1)),
                    (QuantScheme::Mxfp4, &super::GFX950_RUNGS.map(|r| r.2)),
                ] {
                    let got = pick_tile(m, n, k, N_CU, quant);
                    assert!(
                        ok.contains(&got),
                        "{quant:?} at {m}x{n}x{k} selected {got:?}, which is not one of its \
                             own rungs {ok:?}"
                    );
                }
            }
        }
    }
}

/// The mxfp4 prefill GEMM is no longer pinned to 256x256 for every shape.
///
/// This is the T3 regression guard. `mla.rs` used to emit `DevOp::GemmMxfp4` unconditionally,
/// so Kimi's `kv_a_proj` (M=128, N=576) ran as THREE 256x256 tiles on 256 CUs — measured at
/// ≈0.4% of peak, the worst number in the campaign.
/// WHICH SHAPES THE CAMPAIGN CHANGED, and which it deliberately did not.
///
/// `legacy` is the three-rung analytical picker that shipped; `new` is the five-rung one.
/// Measurements do not apply here — the unit-test fixture inventory has build label
/// `test-fixture`, so every record in `tuning/` is correctly stale against it — so this
/// isolates what the RUNGS alone bought. The extra shapes measurement then corrects are
/// pinned in `tests/tuned_tile_selection.rs`.
///
/// Measured TF/s (whole GPU, 1660 sustained bf16 peak, `runtime/ubench/gemm_tile_sweep.c`):
///
/// | shape                     | legacy   | new     | TF/s legacy -> new    |
/// |---------------------------|----------|---------|-----------------------|
/// | g31b q_proj      M=1024   | 256x256  | 128x256 |  521.3 ->  915.0 1.76x|
/// | g31b kv global   M=4096   | 256x256  | 128x256 |  513.7 ->  921.7 1.79x|
/// | llama-8B k/v     M=8192   | 256x256  | 128x256 |  475.4 ->  832.4 1.75x|
/// | g31b o_proj      M=4096   | 256x256  | 192x256 |  792.1 -> 1194.3 1.51x|
/// | qwen o_proj      M=4096   | 256x256  | 192x256 |  583.5 ->  926.5 1.59x|
/// | g31b down_proj   M=2048   | 256x256  | 192x256 |  789.5 -> 1025.8 1.30x|
///
/// And the six M=128 "utilisation disaster" shapes are UNCHANGED, on purpose: 64x128 was
/// already selected and is already the fastest of all twelve tiles compiled into the sweep.
/// Their deficit is CU fill (2-34 tiles on 256 CUs), which no tile can fix — see the report
/// and `plans/` for why split-K is the lever there.
#[test]
fn the_new_rungs_change_the_fill_limited_shapes_and_leave_the_rest() {
    for (m, n, k, legacy, new, label) in [
        // Unchanged: already on the narrowest rung, and it is already optimal.
        (
            128u32,
            128u32,
            2816u32,
            DevOp::GemmSmall,
            DevOp::GemmSmall,
            "gemma26b router",
        ),
        (
            128,
            256,
            6144,
            DevOp::GemmSmall,
            DevOp::GemmSmall,
            "glm52 router",
        ),
        (
            128,
            576,
            6144,
            DevOp::GemmSmall,
            DevOp::GemmSmall,
            "glm52 kv_a_proj",
        ),
        (
            128,
            576,
            7168,
            DevOp::GemmSmall,
            DevOp::GemmSmall,
            "kimi kv_a_proj",
        ),
        (
            128,
            512,
            3840,
            DevOp::GemmSmall,
            DevOp::GemmSmall,
            "g12b k_proj global",
        ),
        (
            128,
            2112,
            2816,
            DevOp::GemmSmall,
            DevOp::GemmSmall,
            "g26b dense gate/up",
        ),
        (
            256,
            8192,
            5376,
            DevOp::GemmSmall,
            DevOp::GemmSmall,
            "g31b q M=256",
        ),
        (
            512,
            8192,
            5376,
            DevOp::GemmMed,
            DevOp::GemmMed,
            "g31b q M=512",
        ),
        // Changed: fill- or quantisation-limited at 256x256.
        (
            1024,
            8192,
            5376,
            DevOp::Gemm,
            DevOp::GemmWide,
            "g31b q M=1024",
        ),
        (
            4096,
            2048,
            5376,
            DevOp::Gemm,
            DevOp::GemmWide,
            "g31b kv global M=4096",
        ),
        (
            8192,
            1024,
            4096,
            DevOp::Gemm,
            DevOp::GemmWide,
            "llama-8B k/v M=8192",
        ),
        (
            4096,
            5376,
            8192,
            DevOp::Gemm,
            DevOp::GemmC5,
            "g31b o M=4096",
        ),
        (
            4096,
            2560,
            4096,
            DevOp::Gemm,
            DevOp::GemmC5,
            "qwen o M=4096",
        ),
        (
            2048,
            5376,
            21504,
            DevOp::Gemm,
            DevOp::GemmC5,
            "g31b down M=2048",
        ),
    ] {
        assert_eq!(pt_legacy_rungs(m, n, k, N_CU), legacy, "legacy: {label}");
        assert_eq!(pt(m, n, k), new, "new: {label}");
    }
}

#[test]
fn mxfp4_prefill_is_tile_selected_not_pinned() {
    assert_eq!(
        pick_tile(128, 576, 7168, N_CU, QuantScheme::Mxfp4),
        DevOp::GemmSmallMxfp4,
        "Kimi kv_a_proj: the narrow-M rung, not the 256x256 default"
    );
    assert_eq!(
        pick_tile(128, 576, 6144, N_CU, QuantScheme::Mxfp4),
        DevOp::GemmSmallMxfp4,
        "GLM-5.2 kv_a_proj"
    );
    // ...and it still picks a large tile where a large tile is right, so this is selection
    // rather than a blanket swap in the other direction.
    assert_ne!(
        pick_tile(8192, 8192, 5376, N_CU, QuantScheme::Mxfp4),
        DevOp::GemmSmallMxfp4,
        "a saturating shape must not get the narrow tile"
    );
}

#[test]
fn llama31_8b_prefill_4k() {
    // hidden 4096, inter 14336, heads 32, kv_heads 8, hd 128.
    // q/o saturate 256 CUs at 256x256 (16x16 = 256 tiles) — keep the big tile.
    assert_eq!(pt(4096, 4096, 4096), DevOp::Gemm, "q_proj");
    assert_eq!(pt(4096, 4096, 4096), DevOp::Gemm, "o_proj");
    // k/v (N=1024) are only 16x4 = 64 tiles at 256x256 — a QUARTER of the machine. The
    // picker drops to 128x128 (16x8 = 256 tiles) to fill all 256 CUs. This is the fix the
    // old heuristic missed (it pinned k/v to 256x256, blind to CU fill).
    assert_eq!(pt(4096, 1024, 4096), DevOp::GemmMed, "k_proj / v_proj");
    // gate/up (fused GemmGlu path keys off Gemm) and down saturate — keep 256x256.
    assert_eq!(pt(4096, 14336, 4096), DevOp::Gemm, "gate/up (fused)");
    assert_eq!(pt(4096, 4096, 14336), DevOp::Gemm, "down_proj");
}

#[test]
fn llama31_8b_prefill_8k_kv_already_half_full() {
    // At M=8192 k/v make 32x4 = 128 tiles at 256x256 — HALF the machine. The old comment
    // here concluded "splitting to 128x128 would need 2 rounds for equal cost, so the
    // higher-intensity 256x256 stays", and that was true of the rungs available: both ways
    // of doubling the tile count halved BOTH dimensions or halved BN, and neither paid.
    //
    // 128x256 is the rung that was missing. It halves BM only, so the tile count doubles to
    // 64x4 = 256 — exactly full — while BN stays 256 and the A-operand reuse is untouched.
    // This is the shape class the campaign was for.
    assert_eq!(pt(8192, 1024, 4096), DevOp::GemmWide, "k/v at 8k");
}

#[test]
fn qwen3_4b_prefill_4k() {
    // hidden 2560, inter 9728, heads 32, kv_heads 8, hd 128.
    assert_eq!(pt(4096, 4096, 2560), DevOp::Gemm, "q_proj");
    assert_eq!(
        pt(4096, 1024, 2560),
        DevOp::GemmMed,
        "k_proj / v_proj (fill)"
    );
    assert_eq!(pt(4096, 9728, 2560), DevOp::Gemm, "gate/up");
    // down_proj is N=2560, which is 10 tile-columns at BN=256 — so at 256x256 it is
    // 16x10 = 160 tiles, 62.5% of the machine, and has been all along. 192x256 gives
    // 22x10 = 220 (86%). MEASURED on the sibling o_proj shape (4096x2560x4096, whole GPU,
    // runtime/ubench/gemm_tile_sweep.c): 256x256 587.7 TF/s vs 192x256 940.6 — **1.60x**,
    // the largest single-shape win in the campaign.
    assert_eq!(
        pt(4096, 2560, 9728),
        DevOp::GemmC5,
        "down_proj (fill: 62.5% -> 86%)"
    );
}

#[test]
fn gemma31b_tiles() {
    // hidden 5376, inter 21504. The projections that genuinely saturate 256 CUs at 256x256
    // keep it — the campaign must not drag them onto a smaller tile.
    assert_eq!(
        pt(4096, 8192, 5376),
        DevOp::Gemm,
        "q sliding (32x32 = 1024 tiles)"
    );
    assert_eq!(pt(4096, 16384, 5376), DevOp::Gemm, "q global");
    assert_eq!(
        pt(4096, 4096, 5376),
        DevOp::Gemm,
        "kv sliding (N=4096, 16x16 = 256)"
    );
    assert_eq!(pt(4096, 21504, 5376), DevOp::Gemm, "gate/up");
    // o_proj and down_proj are both N=5376 = 21 tile-columns at BN=256, so 256x256 gives
    // 16x21 = 336 tiles = 2 rounds at 65.6% efficiency — the tile-count QUANTIZATION case
    // rather than the under-fill case. 192x256 gives 22x21 = 462 = 2 rounds at 90.2%.
    // MEASURED on this N at M=2048 (2048x5376x21504, whole GPU): 256x256 794.4 TF/s vs
    // 192x256 1033.4 — **1.30x**.
    assert_eq!(
        pt(4096, 5376, 8192),
        DevOp::GemmC5,
        "o sliding (quantization: 66% -> 90%)"
    );
    assert_eq!(
        pt(4096, 5376, 21504),
        DevOp::GemmC5,
        "down (same N, same quantization)"
    );
    // kv GLOBAL is N=2048 = 8 tile-columns at BN=256, so 16x8 = 128 tiles — HALF the
    // machine, and the previous version of this test asserted that as "no regression"
    // because there was no rung that could fix it. 128x256 makes it 32x8 = 256, exactly
    // full, at the same BN and so the same A-reuse.
    assert_eq!(
        pt(4096, 2048, 5376),
        DevOp::GemmWide,
        "kv global (fill: 50% -> 100%)"
    );
}

#[test]
fn short_prompt_buckets_use_narrow_tiles() {
    // A 128-row chunk cannot fill 256 CUs with a 256x256 tile (q_proj = 1x16 = 16 tiles),
    // so the picker drops to the narrow-M kernels — matching the measured T=128 optima in
    // op_gemm.h (64x128 fastest for the tall projections at small M).
    assert_eq!(pt(128, 8192, 5376), DevOp::GemmSmall, "T=128 q sliding");
    assert_ne!(
        pt(128, 4096, 4096),
        DevOp::Gemm,
        "T=128 must not pick the big tile"
    );
}

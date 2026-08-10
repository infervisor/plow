//! The measured path, end to end, against the REAL probed inventory and the REAL store.
//!
//! The unit tests in `lib.rs` run against a `#[cfg(test)]` fixture inventory whose build label
//! is `test-fixture`, so every record in `tuning/` is correctly stale for them and they only
//! ever exercise the analytical model. That is the right thing for a machine with no ROCm, and
//! it means those tests cannot see the bug this file exists to catch: a campaign that publishes
//! records the compiler never reads.
//!
//! The failure is silent by construction. If the op-case key, the hardware cell, the oracle
//! string or the build digest disagree between writer and reader, `best_for` returns nothing,
//! `pick_tile` falls back to the analytical model and reports tier `portable` — which is
//! exactly what it reports when no campaign has ever run. Nothing is red. So the check has to
//! be "did the measurement actually change an answer", not "did a lookup succeed".
//!
//! Skips itself, loudly, when the interpreter cannot be probed (no hipcc). A skip is honest;
//! asserting the analytical answer on a machine that cannot probe would pin the fallback as if
//! it were the contract.

use devgen::{gfx950_measured_rungs, gfx950_prefill_tile};
use kernelcaps::QuantScheme::None as Bf16;
use packet::dev::DevOp;

const N_CU: u32 = 256;

fn probe_available() -> bool {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    kernelcaps::dense_gemm_inventory(&root, hwspec::IsaLevel::Gfx950).is_ok()
}

/// Shapes where the ANALYTICAL model and the HARDWARE disagree, and what each names.
///
/// This is the whole case for T2, and it is a specific, explicable bias rather than general
/// inaccuracy. `tile_cost` prices `rounds x max(compute, dma)` for ONE tile, so as soon as
/// every candidate fits in a single round the `rounds` term is 1 for all of them and the
/// ranking is monotone in arithmetic intensity — it names the largest tile, always. That is
/// right when the shape genuinely saturates and wrong exactly when tile-count QUANTISATION is
/// what limits it, which is the regime these four shapes are in.
///
/// Measured on the leased card, whole GPU, 10 samples of 4 launches each after a 50-launch
/// clock warm-up (`runtime/ubench/gemm_tile_sweep.c`; the run in `tuning/`):
///
/// | shape                 | analytical | measured | TF/s analytical -> measured |
/// |-----------------------|------------|----------|-----------------------------|
/// | 2048x8192x5376  (q)   | 256x256    | 128x256  |  878.8 ->  999.8  (1.14x)   |
/// | 8192x8192x5376  (q)   | 256x256    | 192x256  | 1112.2 -> 1218.1  (1.10x)   |
/// | 2048x21504x5376 (g/u) | 256x256    | 192x256  |  972.3 -> 1021.9  (1.05x)   |
/// | 4096x9728x2560  (g/u) | 256x256    | *        |  762.7 ->  945.3  (1.24x)   |
///
/// `*` — on that last shape 128x256 (944.2) and 192x256 (945.3) are **0.12% apart**, which is
/// inside the run-to-run spread, so the expectation is the SET rather than one of them. Pinning
/// either would make this test fail on a re-measure for a reason that is not a regression; that
/// is precisely the mistake `Stats::beats` exists to stop the store from making.
const ANALYTICAL_LOSES: &[(u32, u32, u32, &[DevOp], &str)] = &[
    (
        2048,
        8192,
        5376,
        &[DevOp::GemmWide],
        "Gemma-31B q_proj M=2048",
    ),
    (
        8192,
        8192,
        5376,
        &[DevOp::GemmC5],
        "Gemma-31B q_proj M=8192",
    ),
    (
        2048,
        21504,
        5376,
        &[DevOp::GemmC5],
        "Gemma-31B gate/up M=2048",
    ),
    (
        4096,
        9728,
        2560,
        &[DevOp::GemmWide, DevOp::GemmC5],
        "Qwen gate/up M=4096 (tie)",
    ),
];

#[test]
fn published_measurements_reach_the_compiler_and_change_its_answer() {
    if !probe_available() {
        eprintln!("SKIP: cannot probe the gfx950 interpreter (no hipcc); measured path untested");
        return;
    }
    let mut without_data = Vec::new();
    for &(m, n, k, want, label) in ANALYTICAL_LOSES {
        let rungs = gfx950_measured_rungs(m as i64, n as i64, k as i64, Bf16);
        if rungs == 0 {
            without_data.push(label);
            continue;
        }
        let got = gfx950_prefill_tile(m, n, k, N_CU, Bf16);
        assert!(
            want.contains(&got),
            "{label}: {rungs} measured rung(s) are in the store for this shape, and the \
             compiler chose {got:?} rather than the measured winner {want:?}. The analytical \
             model names Gemm (256x256) here, so this is the case the store exists for."
        );
        assert_ne!(
            got,
            DevOp::Gemm,
            "{label}: the compiler fell back to the analytical answer despite {rungs} \
             measured rung(s) being present"
        );
    }
    assert!(
        without_data.is_empty(),
        "no qualified measurement reached the compiler for {without_data:?}. Either the \
         campaign was never ingested (plowc tune ingest --db tuning --samples ...), or the \
         interpreter was recompiled since and every record is stale against the new build \
         digest — re-run the campaign. This is the failure mode that is INVISIBLE without \
         this test: selection silently reverts to the analytical model and reports tier \
         `portable`, which is what it reports when nothing was ever measured."
    );
}

/// The probe must find all fifteen rungs — five tiles in each of three encodings — in the
/// objects the build script produces.
///
/// This is the reverse-coverage question of §4 asked at the level of the INVENTORY: an arm can
/// exist in `interp.hip`, be register-gated, compile, and still be invisible to the selector
/// because the probe never preprocesses the object that holds it. That is what happened to the
/// fp8 and mxfp4 rungs before `GFX950_QUANT_OBJECTS` existed: `dense_gemm_inventory` probed
/// only the bf16 prefill object, so `pick_tile` had bf16 rungs and nothing else no matter what
/// precision it was asked about.
#[test]
fn the_probe_finds_every_rung_in_every_encoding() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let Ok(inv) = kernelcaps::dense_gemm_inventory(&root, hwspec::IsaLevel::Gfx950) else {
        eprintln!("SKIP: cannot probe the gfx950 interpreter (no hipcc)");
        return;
    };
    use kernelcaps::QuantScheme::{Mxfp4, None as Bf16Q, W8A8};
    for (quant, label) in [(Bf16Q, "bf16"), (W8A8, "w8a8 fp8"), (Mxfp4, "mxfp4")] {
        let mut tiles: Vec<(i64, i64)> = inv
            .iter()
            .filter(|s| s.quant == quant)
            .filter_map(|s| s.tile.map(|t| (t.bm, t.bn)))
            .collect();
        tiles.sort();
        assert_eq!(
            tiles,
            vec![(64, 128), (128, 128), (128, 256), (192, 256), (256, 256)],
            "{label}: the probed inventory does not carry all five rungs"
        );
    }
    assert_eq!(inv.len(), 15, "five rungs x three encodings");
}

/// A shape with no record must still select, from the analytical model, without panicking.
#[test]
fn an_unmeasured_shape_still_selects() {
    if !probe_available() {
        return;
    }
    // Deliberately not in the campaign.
    assert_eq!(gfx950_measured_rungs(333, 777, 1024, Bf16), 0);
    let got = gfx950_prefill_tile(333, 777, 1024, N_CU, Bf16);
    assert!(
        [
            DevOp::Gemm,
            DevOp::GemmMed,
            DevOp::GemmSmall,
            DevOp::GemmWide,
            DevOp::GemmC5
        ]
        .contains(&got),
        "{got:?} is not a bf16 rung"
    );
}

/// The measured M=128 shapes and what the analytical model says about them.
///
/// # THIS TEST USED TO ASSERT AGREEMENT ON THE WHOLE CLASS. MEASUREMENT FALSIFIED IT.
///
/// The original claim was "every M=128 shape selects the narrowest rung, and the *measurement*
/// agrees with the analytical model on this whole class — where the two agree, the campaign
/// confirms rather than corrects." Re-running the campaign on the tree at
/// `gfx950-1802fd083a2269c1` (2026-07-29) broke it on two of the six shapes, and the reason is
/// worth more than the assertion was:
///
/// **`gemm_c4` (64x128, `GemmSmall`) regressed ~30% at some point in this tree's history, and
/// nothing was reading the store's own history closely enough to see it.** At
/// `128x128x2816`, across the eight build digests in `tuning/`:
///
/// | tile | `a4208ab` / `aea715a` / `5562461` / `f9c85e0` / `1481151` | `9b634bb` / `d2bd91a` / `a168b6e` / `1802fd0` |
/// |---|--:|--:|
/// | `gemm_c3` 128x128 (`GemmMed`) | 28.1–28.4 us | 28.5–29.2 us |
/// | `gemm_c4` 64x128 (`GemmSmall`) | **23.4–23.8 us** | **28.7–30.7 us** |
///
/// Every other rung at that shape is flat to within 2%; only the smallest tile moved. That
/// tile is the one EVERY narrow shape selects — the 0.8–3.5%-of-peak routers and
/// `kv_a_proj`s that the design notes §0-EXT-RESULT calls the shape-coverage
/// disasters — so a regression in it is a regression in exactly the class plow is already
/// worst at.
///
/// The measured winners on this build, `c4/c3` ratio in brackets:
///
/// | shape | winner | ratio |
/// |---|---|--:|
/// | Gemma-26B router `128x128x2816` | **`GemmMed`** | 1.070 |
/// | Gemma-12B k_proj global `128x512x3840` | **`GemmMed`** | 1.076 |
/// | GLM-5.2 router `128x256x6144` | `GemmSmall` | 0.846 |
/// | GLM-5.2 `kv_a_proj` `128x576x6144` | `GemmSmall` | 0.857 |
/// | Kimi `kv_a_proj` `128x576x7168` | `GemmSmall` | 0.840 |
/// | Gemma-26B dense gate/up `128x2112x2816` | `GemmSmall` | 0.905 |
///
/// So the test now asserts the two facts that are actually load-bearing, and neither is
/// "the model and the hardware agree":
///
/// 1. Every one of these shapes still selects a NARROW rung (`GemmSmall` or `GemmMed`) —
///    the class-level claim that survives, and the one a wrong answer would break loudly.
/// 2. Each shape selects the rung the CAMPAIGN measured fastest, i.e. the measurement is
///    reaching the compiler. Asserting a hard-coded opcode per shape would just re-encode
///    today's silicon; asserting "the compiler picks what the store measured" is the property
///    the tuner exists to have, and it keeps failing loudly if the store goes stale.
#[test]
fn the_narrow_shapes_select_a_narrow_rung_and_follow_the_measurement() {
    if !probe_available() {
        return;
    }
    for (m, n, k, label) in [
        (128u32, 128u32, 2816u32, "Gemma-26B router"),
        (128, 256, 6144, "GLM-5.2 router"),
        (128, 576, 6144, "GLM-5.2 kv_a_proj"),
        (128, 576, 7168, "Kimi kv_a_proj"),
        (128, 512, 3840, "Gemma-12B k_proj global"),
        (128, 2112, 2816, "Gemma-26B dense gate/up"),
    ] {
        let got = gfx950_prefill_tile(m, n, k, N_CU, Bf16);
        assert!(
            matches!(got, DevOp::GemmSmall | DevOp::GemmMed),
            "{label}: an M=128 shape selected {got:?}, which is not a narrow rung — \
             at 128 rows the wide tiles leave 3/4 of the machine idle"
        );
        // The store must be the thing deciding. `gfx950_measured_rungs` counts qualified,
        // non-stale records for this case; zero means the campaign is stale against this build
        // and the answer above came from the analytical model, which is the silent-degradation
        // failure the whole tunedb exists to make visible.
        assert!(
            gfx950_measured_rungs(m as i64, n as i64, k as i64, Bf16) > 0,
            "{label}: NO qualified measurement for this shape on the probed build — \
             tile selection fell back to the analytical model. Re-run \
             scripts/rebench_tune_gemm.sh and `plowc tune ingest`."
        );
    }
}

/// The gfx942 cell must actually reach the compiler — the guard above only ever asked gfx950.
///
/// This test exists because the file it lives in mentioned `gfx950` eleven times and `gfx942`
/// ZERO times, on a campaign that ships gfx942. `plowc tune status --gpu MI300X` on
/// 2026-08-09 reported all 196 records in `amd/gfx942/mi300x` STALE — selection had been
/// falling back to the analytical model for days, through every published measurement — and
/// nothing was red, because the gfx950 tests kept passing on gfx950's data. A guard is only a
/// guard for the configurations it enumerates.
///
/// The two shapes are GLM-5.2's own prefill projections, and they are the two the campaign
/// actually measured (`lib.rs`: "GLM-5.2 came to have exactly two measured shapes (M=128
/// N=256 K=6144, M=128 N=576 K=6144)"). Asserting on a shape nobody measured would pin a
/// wish rather than a contract.
///
/// It separates the two failures the store can have, because they are different work orders:
/// an EMPTY cell means the campaign never ran, while a cell with records and no usable ones
/// means every record is stale and the campaign must be RE-run against the current object.
#[test]
fn gfx942_measurements_reach_the_compiler() {
    if !probe_available_942() {
        eprintln!(
            "SKIP: cannot probe the gfx942 inventory (no hipcc) — refusing to assert the \
                   analytical fallback as if it were the contract"
        );
        return;
    }
    devgen::set_amd_target_for("gfx942", "MI300X");

    let cell = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tuning/amd/gfx942/mi300x/kernel_measurement.jsonl");
    let on_disk = std::fs::read_to_string(&cell)
        .map(|t| t.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0);

    // Shapes taken from the PUBLISHED CAMPAIGN, not from a code comment. The first version of
    // this test pinned (128, 576, 6144) "GLM-5.2 kv_a_proj" out of a TP4-era note in lib.rs --
    // and that shape DOES NOT EXIST in the shipping compile: TP8 shards the kv_a_proj into
    // 128x512x6144 + 128x64x6144, so the guard was asserting on a shape the demand never asks
    // about and would have reported 0 usable rungs forever, for the wrong reason.
    //
    // 2048x6144x256 is included deliberately: it is the largest disagreement the campaign found
    // between the analytical model and the measurements (GEMM 192x256 at 74729 ns vs MED
    // 128x128 at 55478 ns, 1.35x), so if measured selection ever stops reaching the compiler
    // this is the shape that loses the most.
    let shapes: &[(i64, i64, i64, &str)] = &[
        (128, 256, 6144, "GLM-5.2 router/shared M=128"),
        (128, 512, 6144, "GLM-5.2 kv_a_proj (TP8-sharded) M=128"),
        (2048, 6144, 256, "GLM-5.2 M=2048 — the 1.35x disagreement"),
    ];
    let mut usable = 0usize;
    for &(m, n, k, label) in shapes {
        let r = gfx950_measured_rungs(m, n, k, Bf16);
        eprintln!("  {label}: {r} usable rung(s)");
        usable += r;
    }
    assert!(
        on_disk > 0,
        "tuning/amd/gfx942/mi300x holds NO records at all — the campaign was never ingested \
         for this cell. Run: plowc --hf-dir <ckpt> ... tune gemm --obj <objdir> --samples \
         <out.jsonl>, then plowc tune ingest."
    );
    // THE DECISION, not the lookup. `measured_rungs > 0` is NOT enough and asserting it is how
    // this test first shipped: `select_kernel` uses measurements "only if EVERY candidate has
    // one" (kernelcaps::select.rs:175), so a shape with 4 of 5 candidates measured falls back to
    // the analytical model while every lookup still reports HIT. That is the live state on
    // gfx942 as of 2026-08-09 -- the ingest rung table keys tiles to opcodes by GEOMETRY, and
    // `Gemm` and `GemmC5` share 192x256x64 here, so all 48 timings landed on `GemmC5` and opcode
    // 8 got zero records. A freshly published, wholly CURRENT campaign covering 48/48 demanded
    // shapes emitted a packet BYTE-IDENTICAL to `--no-tuning`.
    let mut decided = 0usize;
    let mut by_measurement = 0usize;
    for &(m, n, k, label) in shapes {
        let tier = devgen::gfx950_prefill_tile_tier(m as u32, n as u32, k as u32, 304, Bf16);
        let measured = tier == kernelcaps::CalibrationTier::SkuCalibrated;
        eprintln!(
            "  {label}: decided by {}",
            if measured {
                "MEASUREMENT"
            } else {
                "the analytical model"
            }
        );
        decided += 1;
        by_measurement += usize::from(measured);
    }

    assert!(
        usable > 0,
        "the gfx942 cell holds {on_disk} record(s) and the compiler can use NONE of them for \
         GLM's own measured shapes: every record is STALE against the current build digest, so \
         tile selection silently fell back to the analytical model and reports tier `portable` \
         -- which is byte-identical to what it reports when nothing was ever measured. The \
         staleness key is defines + toolchain + preprocessed-source digest, so ONE edit under \
         runtime/amd/ re-stales the whole store. Re-run the campaign AGAINST THE SHIPPING \
         OBJECT RECIPE (the defines participate in the digest, so a campaign run against a \
         different -D set is stale on arrival), then plowc tune ingest. \
         `plowc tune status --gpu MI300X` prints the digest census."
    );
    assert!(
        decided > 0 && by_measurement == decided,
        "{by_measurement} of {decided} tile decisions were made BY MEASUREMENT. The store has \
         usable records for these shapes ({usable} rung(s) found by lookup) and selection STILL \
         fell back to the analytical model -- which means the records do not cover every \
         CANDIDATE, not that they are missing. Check the per-opcode census: \
         `plowc tune status --gpu MI300X`, and compare the kernel_name histogram in \
         tuning/amd/gfx942/mi300x/kernel_measurement.jsonl against the probed inventory. A rung \
         with zero records disqualifies every shape it is a candidate for."
    );
}

fn probe_available_942() -> bool {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    kernelcaps::dense_gemm_inventory(&root, hwspec::IsaLevel::Gfx942).is_ok()
}

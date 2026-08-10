//! `plowc tune ingest` / `plowc tune best` — the publishing half of a GEMM campaign.
//!
//! # Why this is a separate subcommand and not a step inside `gemm`
//!
//! It is both. `tune gemm` calls it, AND it stays addressable on its own, because collapsing
//! measurement and publication into one opaque step is how a campaign silently produces nothing.
//!
//! `scripts/rebench_tune_gemm.sh`'s header states the split and the reason: *"the C harness writes
//! SAMPLES (it cannot know the build identity), [the ingest half] attaches the probed digests and
//! applies the store's gates."* On 2026-07-29 a benchmark run died `rc=101` with
//! `wrote 180 rows -> tune_final.jsonl` immediately followed by
//! `1260 record(s) skipped as STALE … NO usable records remain` — because the orchestrating script
//! ran the campaign and **never ingested**. Both halves reported success at their own boundary.
//!
//! So: `gemm` always ingests (that failure cannot recur by omission), and `ingest` is still
//! reachable alone (a campaign whose samples exist but were never published can be repaired
//! without re-measuring). Both print the **published-record count**, which is the only number that
//! says the store changed.
//!
//! # Why the digests cannot come from the sweep
//!
//! A measurement is only valid for the object it ran inside. `kernelcaps` derives that object's
//! identity by preprocessing `interp.hip` and hashing the result, so an edit to `op_gemm.h` — a
//! tile constant, say — changes it and every prior record becomes stale rather than silently
//! authoritative. The C harness cannot compute that, and should not: it would mean two
//! implementations of the identity rule. Digest churn is the dominant operational fact here (23
//! commits touching `runtime/amd/*` in one day produced seven distinct digests), which is why
//! every subcommand prints the digest it probed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kernelcaps::QuantScheme;
use packet::dev::DevOp;
use tunedb::gemm::parse_quant;
use tunedb::{
    gemm_op_case, gemm_rung_opcode, Correctness, Digests, KernelMeasurement, RecordState, Stats,
    TuneStore, GEMM_ORACLE,
};

type Err = Box<dyn std::error::Error>;

/// One row as `runtime/ubench/gemm_tile_sweep.c` writes it.
#[derive(serde::Deserialize)]
struct Row {
    m: i64,
    n: i64,
    k: i64,
    quant: String,
    tile: String,
    sym: String,
    correct: bool,
    samples_ns: Vec<f64>,
}

/// The build identity every record is keyed to.
///
/// Probing is REQUIRED, not best-effort: a record filed under a guessed build would be selectable
/// against an object nobody measured, which is worse than no record at all.
pub fn digests(root: &Path, isa: hwspec::IsaLevel) -> Result<Digests, Err> {
    Ok(probe_inventory(root, isa)?.1)
}

fn probe_inventory(
    root: &Path,
    isa: hwspec::IsaLevel,
) -> Result<(kernelcaps::Inventory, Digests), Err> {
    // `isa` is a PARAMETER because the digest identifies the object the records describe. Probing
    // Gfx950 unconditionally keyed every gfx942 measurement to a gfx950 build, which is exactly
    // the "record filed under a guessed build" this function's own doc comment refuses.
    let inv = kernelcaps::dense_gemm_inventory(root, isa).map_err(|e| {
        let arch = isa.arch_flag();
        format!(
            "cannot probe the {arch} interpreter ({e}); ingest needs it to key records to a build"
        )
    })?;
    let want = Digests {
        implementation: inv.build().label(),
        interpreter: inv.build().label(),
        toolchain: inv.build().toolchain.clone(),
        oracle: GEMM_ORACLE.to_string(),
    };
    Ok((inv, want))
}

/// Publish a sample file into the store. Returns the number of QUALIFIED records published — the
/// count callers print, and the one that distinguishes "measured" from "measured and usable".
pub fn ingest(
    root: &Path,
    db: &PathBuf,
    samples: &Path,
    campaign: &str,
    provisional: bool,
    isa: hwspec::IsaLevel,
    cell: &str,
) -> Result<usize, Err> {
    let text = std::fs::read_to_string(samples)
        .map_err(|e| format!("cannot read samples {}: {e}", samples.display()))?;
    let (inv, want) = probe_inventory(root, isa)?;
    println!("build digest: {}", want.interpreter);
    println!("toolchain   : {}", want.toolchain);

    // Tile → opcodes, derived from the PROBED inventory rather than the static gfx950 RUNGS
    // table. The distinction is load-bearing on gfx942: its object is built GM_BM=192 GM_BN=256,
    // so `Gemm` and `GemmC5` carry the SAME 192x256x64 tile, and a map that files each geometry
    // under one opcode leaves the other with zero records — after which `select_kernel` (which
    // uses measurements only if EVERY candidate has one) discards the whole campaign. A timing is
    // a fact about a tile in this build, so it is filed under every dispatched opcode carrying
    // that tile.
    let tile_ops: Vec<(String, QuantScheme, DevOp)> = inv
        .iter()
        .filter(|s| s.dispatched)
        .filter_map(|s| {
            s.tile
                .map(|t| (format!("{}x{}x{}", t.bm, t.bn, t.bk), s.quant, s.id.0))
        })
        .collect();

    let mut records = Vec::new();
    let mut skipped_no_opcode: BTreeMap<String, usize> = BTreeMap::new();
    let mut failed = Vec::new();

    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let r: Row = serde_json::from_str(line)?;
        let quant = parse_quant(&r.quant).ok_or_else(|| format!("unknown quant {:?}", r.quant))?;
        // A tile with no dispatch arm is a legitimate measurement of a kernel BODY and not a
        // selectable fact — the sweep also compiles calibration-only tiles (320x128, 384x128,
        // 128x384, 192x128). Reported, never stored. The static RUNGS map is kept as a fallback
        // for a probe that found no tiled specs (it is also what pins the sweep↔dispatch
        // agreement in `rungs_map_to_distinct_opcodes`).
        let mut ops: Vec<DevOp> = tile_ops
            .iter()
            .filter(|(t, q, _)| *t == r.tile && *q == quant)
            .map(|(_, _, op)| *op)
            .collect();
        ops.sort_unstable_by_key(|op| *op as u16);
        ops.dedup();
        if ops.is_empty() {
            if let Some(op) = gemm_rung_opcode(&r.tile, quant) {
                ops.push(op);
            } else {
                *skipped_no_opcode.entry(r.tile.clone()).or_default() += 1;
                continue;
            }
        }
        let stats = Stats::from_samples(r.samples_ns.clone())
            .map_err(|e| format!("{} {}x{}x{}: {e}", r.sym, r.m, r.n, r.k))?;
        let op_case = gemm_op_case(r.m, r.n, r.k, quant);
        let correctness = if r.correct {
            Correctness::Pass
        } else {
            Correctness::Fail {
                detail: format!("f64 dot spot-check mismatch on {}", r.sym),
            }
        };
        for op in ops {
            let (kernel_id, kernel_name) = (op as u16, op.c_name().to_string());
            if !r.correct {
                failed.push(format!("{op_case} {kernel_name}"));
            }
            records.push(KernelMeasurement {
                op_case: op_case.clone(),
                kernel_id,
                kernel_name,
                profile: "prefill_dense".into(),
                hardware: cell.to_string(),
                // The cell is `vendor/isa/sku` (store.rs), so the sku IS the cell's last segment.
                // The historical gfx950 cell keeps its recorded label (MI355X silicon measured
                // into the mi350x cell) rather than silently relabeling old provenance.
                sku: if cell == tunedb::GFX950_CELL {
                    "MI355X".into()
                } else {
                    cell.rsplit('/').next().unwrap_or(cell).to_uppercase()
                },
                digests: want.clone(),
                stats: stats.clone(),
                correctness: correctness.clone(),
                // Deliberately None: the register cost of these tiles is a property of the
                // OBJECT, and it is checked by `scripts/build_gfx950.sh`'s cliff gate rather
                // than re-derived here. Recording a number this command did not probe would be
                // a claim.
                registers: None,
                state: RecordState::Provisional,
                campaign: campaign.into(),
            });
        }
    }

    for (tile, n) in &skipped_no_opcode {
        println!("skipped     : {tile} x{n} — calibration-only tile, no dispatch arm");
    }

    let store = TuneStore::new(db.clone());
    // Correct-but-slow is publishable; incorrect is not, and the failure is STORED with its reason
    // so the next campaign does not re-measure the same broken tile.
    let (bad, good): (Vec<_>, Vec<_>) = records
        .into_iter()
        .partition(|r| !matches!(r.correctness, Correctness::Pass));
    if !bad.is_empty() {
        let n = store.record_rejected(cell, bad, "f64 dot spot-check mismatch")?;
        println!(
            "rejected    : {n} record(s) failed the oracle: {}",
            failed.join(", ")
        );
    }
    if provisional {
        println!(
            "provisional : {} record(s) stored, NOT selectable",
            good.len()
        );
        let n = store.record_rejected(cell, good, "--provisional: screening pass only")?;
        println!("stored      : {n}");
        return Ok(0);
    }
    let n = store.publish(cell, good)?;
    println!(
        "published   : {n} qualified record(s) into {}/{cell}",
        db.display()
    );
    Ok(n)
}

/// What the store would serve the compiler right now, for the probed build.
pub fn best(
    root: &Path,
    db: &PathBuf,
    quant: &str,
    isa: hwspec::IsaLevel,
    cell: &str,
) -> Result<(), Err> {
    let store = TuneStore::new(db.clone());
    let want = digests(root, isa)?;
    let (best, stale) = store.best_for(cell, &want)?;
    let q = format!("/{quant}");
    println!("cell        : {cell}");
    println!("build digest: {}", want.interpreter);
    let mut shown = 0;
    for (case, rec) in &best {
        if !case.ends_with(&q) {
            continue;
        }
        println!(
            "  {case:<28} {:<26} median {:>10.1} ns  ({} samples, p90-med {:.1})",
            rec.kernel_name,
            rec.stats.median_ns,
            rec.stats.samples,
            rec.stats.jitter_ns()
        );
        shown += 1;
    }
    if shown == 0 {
        println!("  (no selectable records for quant {quant})");
    }
    if !stale.is_empty() {
        println!(
            "stale       : {} record(s) exist but are for a different build",
            stale.len()
        );
    }
    Ok(())
}

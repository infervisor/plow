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

use tunedb::gemm::parse_quant;
use tunedb::{
    gemm_op_case, gemm_rung_opcode, Correctness, Digests, KernelMeasurement, RecordState, Stats,
    TuneStore, GEMM_ORACLE, GFX950_CELL as CELL,
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
pub fn digests(root: &Path) -> Result<Digests, Err> {
    let inv = kernelcaps::dense_gemm_inventory(root, hwspec::IsaLevel::Gfx950).map_err(|e| {
        format!("cannot probe the gfx950 interpreter ({e}); ingest needs it to key records to a build")
    })?;
    Ok(Digests {
        implementation: inv.build().label(),
        interpreter: inv.build().label(),
        toolchain: inv.build().toolchain.clone(),
        oracle: GEMM_ORACLE.to_string(),
    })
}

/// Publish a sample file into the store. Returns the number of QUALIFIED records published — the
/// count callers print, and the one that distinguishes "measured" from "measured and usable".
pub fn ingest(
    root: &Path,
    db: &PathBuf,
    samples: &Path,
    campaign: &str,
    provisional: bool,
) -> Result<usize, Err> {
    let text = std::fs::read_to_string(samples)
        .map_err(|e| format!("cannot read samples {}: {e}", samples.display()))?;
    let want = digests(root)?;
    println!("build digest: {}", want.interpreter);
    println!("toolchain   : {}", want.toolchain);

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
        // 128x384, 192x128). Reported, never stored.
        let Some(op) = gemm_rung_opcode(&r.tile, quant) else {
            *skipped_no_opcode.entry(r.tile.clone()).or_default() += 1;
            continue;
        };
        let (kernel_id, kernel_name) = (op as u16, op.c_name().to_string());
        let stats = Stats::from_samples(r.samples_ns.clone())
            .map_err(|e| format!("{} {}x{}x{}: {e}", r.sym, r.m, r.n, r.k))?;
        let op_case = gemm_op_case(r.m, r.n, r.k, quant);
        let correctness = if r.correct {
            Correctness::Pass
        } else {
            Correctness::Fail { detail: format!("f64 dot spot-check mismatch on {}", r.sym) }
        };
        if !r.correct {
            failed.push(format!("{op_case} {kernel_name}"));
        }
        records.push(KernelMeasurement {
            op_case,
            kernel_id,
            kernel_name,
            profile: "prefill_dense".into(),
            hardware: CELL.into(),
            sku: "MI355X".into(),
            digests: want.clone(),
            stats,
            correctness,
            // Deliberately None: the register cost of these tiles is a property of the OBJECT,
            // and it is checked by `scripts/build_gfx950.sh`'s cliff gate rather than re-derived
            // here. Recording a number this command did not probe would be a claim.
            registers: None,
            state: RecordState::Provisional,
            campaign: campaign.into(),
        });
    }

    for (tile, n) in &skipped_no_opcode {
        println!("skipped     : {tile} x{n} — calibration-only tile, no dispatch arm");
    }

    let store = TuneStore::new(db.clone());
    // Correct-but-slow is publishable; incorrect is not, and the failure is STORED with its reason
    // so the next campaign does not re-measure the same broken tile.
    let (bad, good): (Vec<_>, Vec<_>) =
        records.into_iter().partition(|r| !matches!(r.correctness, Correctness::Pass));
    if !bad.is_empty() {
        let n = store.record_rejected(CELL, bad, "f64 dot spot-check mismatch")?;
        println!("rejected    : {n} record(s) failed the oracle: {}", failed.join(", "));
    }
    if provisional {
        println!("provisional : {} record(s) stored, NOT selectable", good.len());
        let n = store.record_rejected(CELL, good, "--provisional: screening pass only")?;
        println!("stored      : {n}");
        return Ok(0);
    }
    let n = store.publish(CELL, good)?;
    println!("published   : {n} qualified record(s) into {}/{CELL}", db.display());
    Ok(n)
}

/// What the store would serve the compiler right now, for the probed build.
pub fn best(root: &Path, db: &PathBuf, quant: &str) -> Result<(), Err> {
    let store = TuneStore::new(db.clone());
    let want = digests(root)?;
    let (best, stale) = store.best_for(CELL, &want)?;
    let q = format!("/{quant}");
    println!("cell        : {CELL}");
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
        println!("stale       : {} record(s) exist but are for a different build", stale.len());
    }
    Ok(())
}

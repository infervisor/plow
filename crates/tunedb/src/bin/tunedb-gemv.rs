//! `tunedb-gemv` — move a decode GEMV row sweep into the tuning store.
//!
//! The twin of `tunedb-gemm`, for the half of the network the tuner has never seen.
//! `runtime/ubench/gemv_row_sweep.c` knows how to drive HSA, warm the clock and check that
//! every output row was written; it knows nothing about what a publishable record is. This
//! binary is the other half: it attaches the BUILD IDENTITY (which is what decides staleness),
//! maps each swept symbol onto the opcode it exercises, and applies the store's gates —
//! correct before fast, never on a sample too small to carry dispersion, atomic publication.
//!
//!   tunedb-gemv ingest --db tuning --samples <sweep.jsonl> [--campaign NAME] [--provisional]
//!   tunedb-gemv best   --db tuning
//!
//! # Read `tunedb::gemv` before assuming this selects anything
//!
//! It does not, and that is a property of the path rather than of this binary. `PLOW_GEMV_MM`
//! is a compile-time macro of the OBJECT and the K-unroll is a runtime branch inside the
//! kernel, so for a given `(M, N, K, quant)` the emitter can reach exactly one opcode. There is
//! no per-shape rung to rank. What these records are for is (a) turning the `PLOW_TUNE_DUMP`
//! GEMV census into HIT/MISS, so a campaign can be audited for coverage rather than assumed
//! complete — the exact failure that left GLM-5.2 prefill 100% unmeasured on the GEMM side —
//! and (b) the M curve, which is what prices the object-level decisions: which `PLOW_GEMV_MM`,
//! and whether `PLOW_GEMV_WALK` is on.
//!
//! # Why the digests cannot come from the sweep
//!
//! A measurement is only valid for the object it ran inside. `kernelcaps` derives that
//! object's identity by preprocessing `interp.hip` and hashing the result, so an edit to
//! `op_gemm.h` changes it and every prior record becomes stale rather than silently
//! authoritative. The C harness cannot compute that, and should not: it would mean two
//! implementations of the identity rule. A missing ingest step is not a smaller version of
//! this — it silently leaves the store untouched while every gate stays green.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use tunedb::gemm::parse_quant;
use tunedb::gemv::gemv_sample_family;
use tunedb::{
    gemv_op_case, gemv_sample_bucket, gemv_sample_opcode, Correctness, Digests, KernelMeasurement,
    RecordState, Stats, TuneStore, GEMV_ORACLE, GFX950_CELL as CELL,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tunedb-gemv: {e}");
            ExitCode::FAILURE
        }
    }
}

type Err = Box<dyn std::error::Error>;

/// One row as `gemv_row_sweep.c` writes it.
///
/// `mm` is the compiled row bucket the sample was taken at and is NOT the same number as `m`:
/// whenever the walk is on they differ, and that difference is the whole experiment.
#[derive(serde::Deserialize)]
struct Row {
    m: u32,
    n: u32,
    k: u32,
    quant: String,
    mm: u32,
    sym: String,
    correct: bool,
    samples_ns: Vec<f64>,
}

fn run() -> Result<(), Err> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let opt = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let flag = |name: &str| args.iter().any(|a| a == name);
    let db = PathBuf::from(opt("--db").unwrap_or_else(|| "tuning".into()));

    match cmd {
        "ingest" => {
            let samples = opt("--samples").ok_or("ingest needs --samples <sweep.jsonl>")?;
            ingest(
                &db,
                &PathBuf::from(samples),
                &opt("--campaign").unwrap_or_else(|| "gemv-row-inventory".into()),
                flag("--provisional"),
            )
        }
        "best" => best(&db),
        _ => {
            eprintln!("usage: tunedb-gemv ingest --db <dir> --samples <sweep.jsonl>");
            eprintln!("       tunedb-gemv best   --db <dir>");
            Err("no command".into())
        }
    }
}

/// The build identity every record is keyed to.
///
/// Probing is REQUIRED, not best-effort: a record filed under a guessed build would be
/// selectable against an object nobody measured, which is worse than no record at all.
fn digests(root: &std::path::Path) -> Result<Digests, Err> {
    let inv = kernelcaps::dense_gemm_inventory(root, hwspec::IsaLevel::Gfx950).map_err(|e| {
        format!(
            "cannot probe the gfx950 interpreter ({e}); ingest needs it to key records to a build"
        )
    })?;
    Ok(Digests {
        implementation: inv.build().label(),
        interpreter: inv.build().label(),
        toolchain: inv.build().toolchain.clone(),
        // The GEMV oracle checks something the GEMM one never had to: that no output ROW was
        // left untouched. Keying on it stops a record checked by the weaker oracle being
        // served to a caller expecting this one.
        oracle: GEMV_ORACLE.to_string(),
    })
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn ingest(db: &PathBuf, samples: &PathBuf, campaign: &str, provisional: bool) -> Result<(), Err> {
    let text = std::fs::read_to_string(samples)?;
    let want = digests(&repo_root())?;
    println!("build       : {}", want.interpreter);
    println!("toolchain   : {}", want.toolchain);

    let mut records = Vec::new();
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    let mut failed = Vec::new();

    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let r: Row = serde_json::from_str(line)?;
        // Validated, then discarded: the key carries the quant as the string the sweep wrote,
        // and this is what refuses a spelling the compiler could never look up.
        parse_quant(&r.quant).ok_or_else(|| format!("unknown quant {:?}", r.quant))?;
        let (Some(op), Some(mm), Some(fam)) = (
            gemv_sample_opcode(&r.sym),
            gemv_sample_bucket(&r.sym),
            gemv_sample_family(&r.sym),
        ) else {
            *skipped.entry(r.sym.clone()).or_default() += 1;
            continue;
        };
        if mm != r.mm {
            return Err(format!("{}: symbol says MM={mm}, row says MM={}", r.sym, r.mm).into());
        }
        let stats = Stats::from_samples(r.samples_ns.clone())
            .map_err(|e| format!("{} {}x{}x{}: {e}", r.sym, r.m, r.n, r.k))?;
        let op_case = gemv_op_case(fam, r.m, r.n, r.k, &r.quant);
        let correctness = if r.correct {
            Correctness::Pass
        } else {
            Correctness::Fail {
                detail: format!("row-coverage or f64 dot check failed on {}", r.sym),
            }
        };
        if !r.correct {
            failed.push(format!("{op_case} {}", r.sym));
        }
        records.push(KernelMeasurement {
            op_case,
            kernel_id: op as u16,
            kernel_name: op.c_name().to_string(),
            // `decode_gemv`, not `prefill_dense`: the profile is what separates these rows from
            // the GEMM cell's inside one file, and they are measurements of different phases on
            // different rooflines. A shared profile would let `best_for` rank one against the
            // other at a coincidentally equal shape.
            profile: "decode_gemv".into(),
            hardware: CELL.into(),
            sku: "MI355X".into(),
            digests: want.clone(),
            stats,
            correctness,
            // The row bucket the object was compiled at. `KernelMeasurement` has no build-
            // identity column beyond `digests` (`tuning/README.md` records the same limitation
            // for the sm_120a prefill-tile cell), and here every rung comes from ONE object, so
            // the digest cannot distinguish them. The campaign label is the only field that
            // can, and losing it would make an MM=8 object serving M=16 indistinguishable from
            // an MM=16 object — which is precisely the comparison the walk exists to make.
            campaign: format!("{campaign}/mm{mm}"),
            registers: None,
            state: RecordState::Provisional,
        });
    }

    for (sym, n) in &skipped {
        println!("skipped     : {sym} x{n} — no GEMV opcode maps to this symbol");
    }

    let store = TuneStore::new(db.clone());
    let (bad, good): (Vec<_>, Vec<_>) = records
        .into_iter()
        .partition(|r| !matches!(r.correctness, Correctness::Pass));
    if !bad.is_empty() {
        let n = store.record_rejected(CELL, bad, "gemv row-coverage/dot check failed")?;
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
        let n = store.record_rejected(CELL, good, "--provisional: screening pass only")?;
        println!("stored      : {n}");
        return Ok(());
    }
    let n = store.publish(CELL, good)?;
    println!(
        "published   : {n} qualified record(s) into {}/{CELL}",
        db.display()
    );
    Ok(())
}

/// Print the M curve, which is the shape of this cell's only real answer.
///
/// Grouped by `(N, K)` and laid out across M, because "the per-token cost of a decode GEMV as
/// the batch widens" is one row of a table and not one number — and the point at which it stops
/// falling is the whole result (`plans/knob-contract.md` §6g-BATCH: aggregate tok/s peaks at
/// B=8 and LOSES 30% at B=16).
fn best(db: &PathBuf) -> Result<(), Err> {
    let store = TuneStore::new(db.clone());
    let want = digests(&repo_root())?;
    let (best, stale) = store.best_for(CELL, &want)?;
    println!("cell        : {CELL}");
    println!("build       : {}", want.interpreter);
    let mut rows: BTreeMap<(u32, u32, String), BTreeMap<u32, f64>> = BTreeMap::new();
    for (case, rec) in &best {
        if rec.profile != "decode_gemv" {
            continue;
        }
        let Some((_fam, rest)) = case.split_once('/') else {
            continue;
        };
        let Some((dims, _)) = rest.split_once('/') else {
            continue;
        };
        let d: Vec<u32> = dims.split('x').filter_map(|s| s.parse().ok()).collect();
        if d.len() != 3 {
            continue;
        }
        rows.entry((d[1], d[2], rec.kernel_name.clone()))
            .or_default()
            .insert(d[0], rec.stats.median_ns);
    }
    if rows.is_empty() {
        println!("  (no selectable GEMV records for this build)");
    }
    println!(
        "  {:<10} {:<8} {:<22} {}",
        "N", "K", "op", "median ns / ns-per-row at M=1,2,4,8,16"
    );
    for ((n, k, op), by_m) in &rows {
        print!("  {n:<10} {k:<8} {op:<22}");
        for m in [1u32, 2, 4, 8, 16] {
            match by_m.get(&m) {
                Some(ns) => print!(" {:>9.0}/{:<8.0}", ns, ns / m as f64),
                None => print!(" {:>9}/{:<8}", "-", "-"),
            }
        }
        println!();
    }
    if !stale.is_empty() {
        println!(
            "stale       : {} record(s) exist but are for a different build",
            stale.len()
        );
    }
    Ok(())
}

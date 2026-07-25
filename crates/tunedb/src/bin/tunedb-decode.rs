//! `tunedb-decode` — move a decode knob sweep into the tuning store, and read
//! the winner back out.
//!
//! `scripts/tune_decode_sweep.sh` produces raw rows: it knows how to drive
//! nvcc, plowc, gpulease and step_bench, and nothing about what a publishable
//! record is. This binary is the other half — it applies the store's gates
//! (correct before fast, never on a sample too small to carry dispersion,
//! atomic publication) and ranks within a cell.
//!
//!   tunedb-decode ingest --db tuning --results <sweep.jsonl> [--oracle NAME]
//!                        [--correctness pass|unchecked] [--provisional]
//!   tunedb-decode best   --db tuning --hardware nvidia/sm_90a/h100-nvl
//!                        [--json <out>] [--all]
//!
//! `ingest` refuses to publish a sweep whose reps are below
//! `Stats::MIN_SAMPLES`, and `--provisional` is the honest way to keep such a
//! screening pass: the numbers are stored and reportable but never selectable.
//! There is deliberately no flag that lowers the sample floor.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use tunedb::{
    rank_by_cell, Correctness, CtxBucket, DecodeCell, DecodeKnobs, DecodeMeasurement, Digests,
    RecordState, Stats, TuneStore,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tunedb-decode: {e}");
            ExitCode::FAILURE
        }
    }
}

type Err = Box<dyn std::error::Error>;

fn run() -> Result<(), Err> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let opt = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let flag = |name: &str| args.iter().any(|a| a == name);
    let db = PathBuf::from(opt("--db").unwrap_or_else(|| "tuning".into()));

    match cmd {
        "ingest" => {
            let results = opt("--results").ok_or("ingest needs --results <sweep.jsonl>")?;
            let oracle = opt("--oracle").unwrap_or_else(|| "step_bench-tpot".into());
            // Defaults to Unchecked, NOT Pass. The sweep measures speed; nothing
            // in it checks that a configuration still produces the right tokens.
            // Defaulting to Pass would let the store take the operator's word for
            // its one non-negotiable gate, which is how "correct before fast"
            // quietly becomes "fast". `--correctness pass` is an assertion the
            // caller makes after running the oracle (gpu_lifecycle), and it should
            // read like one at the call site.
            let correctness = match opt("--correctness").as_deref() {
                Some("pass") => Correctness::Pass,
                Some("unchecked") | None => Correctness::Unchecked,
                Some(other) => Correctness::Fail { detail: other.into() },
            };
            ingest(&db, &PathBuf::from(results), &oracle, correctness, flag("--provisional"))
        }
        "best" => {
            let hw = opt("--hardware").ok_or("best needs --hardware <cell>")?;
            best(&db, &hw, opt("--json").map(PathBuf::from), flag("--all"))
        }
        _ => Err("usage: tunedb-decode <ingest|best> [options]".into()),
    }
}

/// One raw row as `tune_decode_sweep.sh` writes it.
#[derive(serde::Deserialize)]
struct Row {
    config: String,
    ctx: u32,
    dtype: String,
    hardware: String,
    model: String,
    minblk: u32,
    n_cu: u32,
    gv_unroll: u32,
    gv_unroll_glu: u32,
    gv_moe_un: u32,
    moe_down_sg: u32,
    ns_abs: u32,
    // Flash family. Absent in rows written before it existed, and absent from a
    // row that did not name the knob — both mean "not overridden".
    #[serde(default)]
    ns_full_abs: u32,
    #[serde(default)]
    extra_defines: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    extra_emit: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    fa_wpr: Option<u32>,
    #[serde(default)]
    fa_gf: Option<u32>,
    #[serde(default)]
    fa_gf_full: Option<u32>,
    #[serde(default)]
    fa_kun: Option<u32>,
    /// Ablated-twin median and the op cost it implies, when the sweep built one.
    #[serde(default)]
    op_cost_ms: Option<f64>,
    /// TPOT of each `step_bench` invocation, milliseconds.
    samples_ms: Vec<f64>,
    registers: Option<u32>,
    toolchain: String,
    implementation: String,
    cubin_sha: String,
    campaign: String,
    /// Whether the sweep could verify the GPU was ours. Absent in older rows,
    /// which predate the check — treated as unverified rather than as a pass.
    #[serde(default)]
    uncontended: bool,
    #[serde(default)]
    vram_before_mib: Option<u64>,
}

fn ingest(
    db: &PathBuf,
    results: &PathBuf,
    oracle: &str,
    correctness: Correctness,
    provisional: bool,
) -> Result<(), Err> {
    let text = std::fs::read_to_string(results)?;
    let mut by_hw: BTreeMap<String, Vec<DecodeMeasurement>> = BTreeMap::new();
    let mut skipped = 0usize;
    let mut contended = 0usize;
    let mut worst_vram = 0u64;

    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let r: Row = serde_json::from_str(line)
            .map_err(|e| format!("{}:{}: {e}", results.display(), n + 1))?;
        if !r.uncontended {
            contended += 1;
            worst_vram = worst_vram.max(r.vram_before_mib.unwrap_or(0));
        }
        // Milliseconds in the harness, nanoseconds in the store: the store's
        // unit is fixed and converting at the boundary is the only place the
        // two can be reconciled once.
        let ns: Vec<f64> = r.samples_ms.iter().map(|ms| ms * 1.0e6).collect();
        let stats = match Stats::from_samples(ns) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  skip {} ctx={}: {e}", r.config, r.ctx);
                skipped += 1;
                continue;
            }
        };
        let m = DecodeMeasurement {
            cell: DecodeCell {
                hardware: r.hardware.clone(),
                dtype: r.dtype,
                n_cu: r.n_cu,
                ctx_bucket: CtxBucket::of(r.ctx),
                model: r.model,
            },
            knobs: DecodeKnobs {
                minblk: r.minblk,
                n_cu: r.n_cu,
                gv_unroll: r.gv_unroll,
                gv_unroll_glu: r.gv_unroll_glu,
                gv_moe_un: r.gv_moe_un,
                moe_down_sg: r.moe_down_sg,
                ns_abs: r.ns_abs,
                fa_wpr: r.fa_wpr,
                fa_gf: r.fa_gf,
                fa_gf_full: r.fa_gf_full,
                fa_kun: r.fa_kun,
                ns_full_abs: r.ns_full_abs,
                // Families with no typed field yet ride these, so a new sweep
                // needs no schema change and old rows still load.
                extra_defines: r.extra_defines.clone(),
                extra_emit: r.extra_emit.clone(),
            },
            ctx: r.ctx,
            digests: Digests {
                implementation: r.implementation,
                // The built object's own identity: two knob sets can share a
                // source tree and still be different objects, which is exactly
                // what this sweep varies.
                interpreter: r.cubin_sha,
                toolchain: r.toolchain,
                oracle: oracle.to_string(),
            },
            stats,
            registers: r.registers,
            correctness: correctness.clone(),
            state: RecordState::Provisional,
            campaign: r.campaign,
        };
        by_hw.entry(r.hardware).or_default().push(m);
    }

    // tuning/README.md is explicit: a contended run is discarded, not stored
    // with a caveat. The sweep's own VRAM check is stricter than gpulease's
    // namespace-blind audit, so honour it here rather than letting a row that
    // could not verify the GPU was ours become selectable.
    if contended > 0 && !provisional {
        return Err(format!(
            "{contended} of these rows could not verify the GPU was uncontended \
             (worst reading {worst_vram} MiB resident before a run).\n  \
             tuning/ does not store a caveated measurement as qualified.\n  \
             Re-run them on an idle card, or ingest with --provisional to keep them \
             unselectable."
        )
        .into());
    }

    let store = TuneStore::new(db.clone());
    for (hw, records) in by_hw {
        let n = records.len();
        if provisional {
            store.record_decode_unqualified(&hw, records, RecordState::Provisional)?;
            println!("{hw}: {n} record(s) stored PROVISIONAL (screening pass, not selectable)");
        } else {
            match store.publish_decode(&hw, records.clone()) {
                Ok(n) => println!("{hw}: {n} record(s) published QUALIFIED"),
                Err(e) => {
                    eprintln!("{hw}: not published — {e}");
                    eprintln!("  storing as provisional instead so the numbers are not lost");
                    store.record_decode_unqualified(&hw, records, RecordState::Provisional)?;
                }
            }
        }
    }
    if skipped > 0 {
        println!("{skipped} row(s) skipped: too few samples to carry dispersion");
    }
    if contended > 0 {
        println!("{contended} row(s) could not verify an uncontended GPU — kept unselectable");
    }
    Ok(())
}

fn best(db: &PathBuf, hw: &str, json: Option<PathBuf>, all: bool) -> Result<(), Err> {
    let store = TuneStore::new(db.clone());
    let records = store.load_decode(hw)?;
    if records.is_empty() {
        println!("no decode records for {hw}");
        return Ok(());
    }
    let usable: Vec<DecodeMeasurement> = if all {
        records
    } else {
        records.into_iter().filter(|r| r.state.is_selectable()).collect()
    };
    if usable.is_empty() {
        println!("{hw}: records exist but none are qualified (run without --provisional, or --all to see them)");
        return Ok(());
    }

    let rankings = rank_by_cell(usable);
    println!("{:<44} {:>9} {:>8} {:>5}  {}", "cell", "TPOT ms", "margin", "REG", "winning knobs");
    let mut rows = Vec::new();
    for c in &rankings {
        let w = c.winner().expect("a ranking has a winner");
        let margin = c
            .margin_ms()
            .map(|m| format!("{m:+.3}"))
            .unwrap_or_else(|| "  n/a".into());
        println!(
            "{:<44} {:>9.3} {:>8} {:>5}  {}{}",
            c.cell.key(),
            w.median_ms(),
            margin,
            w.registers.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
            w.knobs.label(),
            // "only candidate" and "won inside its own noise" are different
            // claims, and printing the same caveat for both would hide which
            // cells were actually contested.
            match (c.ranked.len(), c.winner_is_decisive()) {
                (1, _) => "  (only candidate)",
                (_, false) => "  (inside noise)",
                _ => "",
            }
        );
        rows.push(serde_json::json!({
            "cell": c.cell.key(),
            "hardware": c.cell.hardware,
            "model": c.cell.model,
            "dtype": c.cell.dtype,
            "n_cu": c.cell.n_cu,
            "ctx_bucket": c.cell.ctx_bucket.label(),
            "winner": {
                "knobs": w.knobs,
                "ctx": w.ctx,
                "tpot_ms": w.median_ms(),
                "registers": w.registers,
                "samples": w.stats.samples,
                "defines": w.knobs.defines(),
                "emit": w.knobs.emit_env(),
            },
            "runner_up_margin_ms": c.margin_ms(),
            "decisive": c.winner_is_decisive(),
            "candidates": c.ranked.len(),
        }));
    }
    if let Some(path) = json {
        std::fs::write(&path, serde_json::to_string_pretty(&rows)? + "\n")?;
        println!("\nwrote {}", path.display());
    }
    Ok(())
}

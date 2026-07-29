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
//!   tunedb-decode best   --db tuning --hardware nvidia/sm_120a/rtx-5090
//!                        [--model M] [--dtype fp8] [--n-cu N] [--batch B] [--ctx T]
//!                        [--print defines|emit] [--json <out>] [--all]
//!
//! `best`'s cell filters and `--print` are what `tuning/README-decode-tuner.md`
//! has always documented as the way a build consumes the store:
//!
//!   PLOW_EXTRA_DEFINES="$(tunedb-decode best ... --print defines)" \
//!     scripts/build_sm120_cubin.sh out/interp_sm120.cubin
//!
//! `--print` refuses when the filter leaves more than one cell standing, since
//! a flag string names ONE object and the union of two cells' winners is an
//! object nobody measured.
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

fn one() -> u32 {
    1
}

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
            let num = |name: &str| -> Result<Option<u32>, Err> {
                opt(name).map(|v| v.parse::<u32>()).transpose().map_err(|e| {
                    format!("{name} takes a number: {e}").into()
                })
            };
            let filter = CellFilter {
                model: opt("--model"),
                dtype: opt("--dtype"),
                n_cu: num("--n-cu")?,
                batch: num("--batch")?,
                ctx_bucket: num("--ctx")?.map(CtxBucket::of),
            };
            let print = match opt("--print").as_deref() {
                None => Print::Table,
                Some("defines") => Print::Defines,
                Some("emit") => Print::Emit,
                Some(other) => {
                    return Err(format!("--print takes defines|emit, not {other:?}").into())
                }
            };
            best(&db, &hw, filter, print, opt("--json").map(PathBuf::from), flag("--all"))
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
    /// Decode slots the row was measured at.
    ///
    /// Defaults to 1, and that default is SAFE here in a way it would not be on
    /// a stored record: until px15 the sweep script's only `step_bench`
    /// invocation passed a literal `1` for the slot count, so a raw row without
    /// the field is one the harness could not have measured at any other batch.
    /// The value is recovered, not assumed. `DecodeCell.batch` has no default
    /// for exactly the opposite reason.
    #[serde(default = "one")]
    batch: u32,
    /// `GV_MM_MAX`. Absent means the sweep did not name it, i.e. the source
    /// default (8) — which is NOT the same as the value 0.
    #[serde(default)]
    gv_mm_max: Option<u32>,
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
    ///
    /// Part of the record SCHEMA, not of any decision this binary makes — no
    /// code reads it. Kept (rather than dropped to silence the warning) because
    /// the JSONL is append-only and permanent: a field removed from the struct
    /// is a field silently ignored on every future read of every historical
    /// row, which is how a store quietly loses a column it was recording.
    #[allow(dead_code)]
    #[serde(default)]
    op_cost_ms: Option<f64>,
    /// Whether the sweep's reps agreed. Absent in rows written before the check
    /// existed, which default to unstable rather than to a pass.
    #[serde(default)]
    stable: bool,
    #[serde(default)]
    rel_spread: Option<f64>,
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
    let mut unstable = 0usize;
    let mut worst_spread = 0.0f64;

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
        // Reps that disagree are not a measurement. A configuration cannot make
        // its own timing erratic on a quiet card, so a wide spread means the run
        // was disturbed -- the same class of problem as a contended GPU, and it
        // gets the same treatment rather than a footnote.
        if !r.stable {
            unstable += 1;
            worst_spread = worst_spread.max(r.rel_spread.unwrap_or(f64::INFINITY));
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
                batch: r.batch,
            },
            knobs: DecodeKnobs {
                minblk: r.minblk,
                n_cu: r.n_cu,
                gv_unroll: r.gv_unroll,
                gv_unroll_glu: r.gv_unroll_glu,
                gv_moe_un: r.gv_moe_un,
                moe_down_sg: r.moe_down_sg,
                gv_mm_max: r.gv_mm_max,
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

    if unstable > 0 && !provisional {
        return Err(format!(
            "{unstable} of these rows have reps that disagree (worst relative spread \
             {worst_spread:.3}).\n  \
             A configuration cannot make its own timing erratic on a quiet card; the run \
             was disturbed.\n  \
             Re-measure them, or ingest with --provisional to keep them unselectable."
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
    if unstable > 0 {
        println!("{unstable} row(s) had reps that disagree — kept unselectable");
    }
    Ok(())
}

/// Which cells `best` should answer for. Every field is optional; a `None`
/// leaves that axis unconstrained.
///
/// This exists because `tuning/README-decode-tuner.md` documents
/// `best --model ... --dtype ... --n-cu ... --ctx ...` and the binary accepted
/// none of them — the documented consumption path did not run. `--batch` joins
/// them as the px15 axis.
#[derive(Default)]
struct CellFilter {
    model: Option<String>,
    dtype: Option<String>,
    n_cu: Option<u32>,
    batch: Option<u32>,
    ctx_bucket: Option<CtxBucket>,
}

impl CellFilter {
    fn matches(&self, c: &DecodeCell) -> bool {
        self.model.as_ref().is_none_or(|m| *m == c.model)
            && self.dtype.as_ref().is_none_or(|d| *d == c.dtype)
            && self.n_cu.is_none_or(|n| n == c.n_cu)
            && self.batch.is_none_or(|b| b == c.batch)
            && self.ctx_bucket.is_none_or(|b| b == c.ctx_bucket)
    }

    /// How the filter reads back to a human, for the "nothing matched" message.
    fn describe(&self) -> String {
        let mut p = Vec::new();
        if let Some(v) = &self.model {
            p.push(format!("model={v}"));
        }
        if let Some(v) = &self.dtype {
            p.push(format!("dtype={v}"));
        }
        if let Some(v) = self.n_cu {
            p.push(format!("n_cu={v}"));
        }
        if let Some(v) = self.batch {
            p.push(format!("batch={v}"));
        }
        if let Some(v) = self.ctx_bucket {
            p.push(format!("ctx={}", v.label()));
        }
        if p.is_empty() { "(unfiltered)".into() } else { p.join(" ") }
    }
}

/// What `best` writes to stdout.
enum Print {
    /// The human ranking table.
    Table,
    /// Just the `nvcc` flags, one line, for `PLOW_EXTRA_DEFINES="$(...)"`.
    Defines,
    /// Just the packet-emit knobs. Deliberately a SEPARATE mode rather than
    /// more words on the same line: object flags and packet flags land in
    /// different artifacts, and a cubin built for 2 blocks/SM against a
    /// 132-block packet is not slower, it is a launch the engine refuses.
    Emit,
}

fn best(
    db: &PathBuf,
    hw: &str,
    filter: CellFilter,
    print: Print,
    json: Option<PathBuf>,
    all: bool,
) -> Result<(), Err> {
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
    let usable: Vec<DecodeMeasurement> =
        usable.into_iter().filter(|r| filter.matches(&r.cell)).collect();
    if usable.is_empty() {
        println!("{hw}: no qualified records match {}", filter.describe());
        return Ok(());
    }

    let rankings = rank_by_cell(usable);

    // A flag string is consumed by a shell, so it must name exactly one object.
    // Printing the union of two cells' defines would build a third object that
    // was never measured — the failure mode this whole store exists to prevent.
    if let Print::Defines | Print::Emit = print {
        if rankings.len() != 1 {
            return Err(format!(
                "--print names ONE object but {} cells match {}.\n  \
                 Narrow with --model/--dtype/--n-cu/--batch/--ctx; cells are:\n    {}",
                rankings.len(),
                filter.describe(),
                rankings.iter().map(|c| c.cell.key()).collect::<Vec<_>>().join("\n    ")
            )
            .into());
        }
        let w = rankings[0].winner().expect("a ranking has a winner");
        let out = match print {
            Print::Defines => w.knobs.defines(),
            _ => w.knobs.emit_env(),
        };
        println!("{}", out.join(" "));
        return Ok(());
    }

    println!("{:<52} {:>9} {:>8} {:>5}  {}", "cell", "TPOT ms", "margin", "REG", "winning knobs");
    let mut rows = Vec::new();
    for c in &rankings {
        let w = c.winner().expect("a ranking has a winner");
        let margin = c
            .margin_ms()
            .map(|m| format!("{m:+.3}"))
            .unwrap_or_else(|| "  n/a".into());
        println!(
            "{:<52} {:>9.3} {:>8} {:>5}  {}{}",
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
            "batch": c.cell.batch,
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

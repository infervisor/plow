//! `plowc tune status` — is the store keyed to the object about to ship, and does it cover what
//! the compiler will ask?
//!
//! # Staleness outranks coverage, and this is the command that says so
//!
//! Coverage gaps are cheap and visible: a missing shape is one shape selecting from the analytical
//! model, and a census prints it. **Total silent staleness is neither cheap nor visible.** The
//! tunedb key is the preprocessed dense-GEMM-family digest. Relevant body, expanded macro, or
//! toolchain changes stale the records; unrelated interpreter arms do not. Total staleness still
//! silently reverts the compiler to tier `portable`, so it remains the first check here.
//!
//! So this command does three things, in this order:
//!
//! 1. **Digest census.** Which build identities the store's records are keyed to, how many under
//!    each, and which one is the object being probed right now. A store with two digests and the
//!    current one holding 40 records is a very different situation from one holding 3000, and
//!    "3160 records" alone cannot tell them apart.
//! 2. **The total-staleness alarm.** Records exist and NONE are selectable ⇒ this is an **error**,
//!    not a line of output. That state means the compiler is on the analytical model for
//!    everything while the store looks full.
//! 3. **Coverage**, when a checkpoint is given: run the emit, take the demand census, and report
//!    HIT/MISS. Second, deliberately — against a 100%-stale store every shape reads MISS whatever
//!    the campaign covered, so coverage is only meaningful once (1) and (2) are clean.
//!
//! # `regress`: the store is a regression detector and nobody was reading it that way
//!
//! Records under older digests are not garbage to be swept — they are the same op cases measured
//! against earlier objects, i.e. a time series. `gemm_c4` (64x128) regressed **~30%** across the
//! store's own history (23.4–23.8 µs to 28.7–30.7 µs at 128x128x2816) while every other rung
//! stayed flat within 2%, and that is the tile every narrow `M=128` shape selects. Nothing
//! reported it because nothing compared across digests. [`regress`] does.
//!
//! The store has no timestamps, so the timeline is **file order** — the order `publish` appended
//! records in. Stated rather than hidden: it is a proxy, and a store that was ever hand-edited or
//! re-sorted would break it.

use std::collections::BTreeMap;
use std::path::PathBuf;

use tunedb::{Digests, KernelMeasurement, TuneStore};

use super::demand::{self, EmitSpec};

type Err = Box<dyn std::error::Error>;

/// Report the store's health against `want`, the digests of the object being probed.
pub fn status(
    db: &PathBuf,
    cell: &str,
    want: &Digests,
    coverage_from: Option<&EmitSpec>,
) -> Result<(), Err> {
    let store = TuneStore::new(db.clone());
    let all = store.load_kernels(cell)?;

    println!("database    : {}", db.display());
    println!("cell        : {cell}");
    println!("build digest: {}", want.interpreter);
    println!("toolchain   : {}", want.toolchain);
    if all.is_empty() {
        println!();
        // AN EMPTY CELL IS AMBIGUOUS, AND SAYING "COLD START" RESOLVES THE AMBIGUITY THE WRONG WAY.
        //
        // `--gpu mi355x` is the SKU stamped on every record in this store (`sku: "MI355X"`), but
        // `GFX950_CELL` is `amd/gfx950/mi350x`, so naming the real hardware lands on an empty cell
        // and this printed "COLD-START ... not a fault" over 3160 existing records. That is the
        // silent-emptiness failure this subcommand was written to prevent, reproduced by the
        // subcommand itself. Rather than guess which spelling is canonical — a naming question,
        // not a reporting one — say which cell was read and which siblings are not empty.
        let siblings: Vec<(String, usize)> = store
            .cells()
            .unwrap_or_default()
            .into_iter()
            .filter(|c| c != cell)
            .filter_map(|c| match store.load_kernels(&c) {
                Ok(r) if !r.is_empty() => Some((c, r.len())),
                _ => None,
            })
            .collect();
        if siblings.is_empty() {
            println!("no kernel measurements for `{cell}` — the COLD-START case, not a fault.");
            println!("selection will use the analytical model and report tier `portable`.");
        } else {
            println!("no kernel measurements for `{cell}`, but this database is NOT empty:");
            for (c, n) in &siblings {
                println!("  {c:<28} {n} record(s)");
            }
            println!();
            println!("so this is a CELL MISMATCH, not a cold start. Re-run naming the cell above");
            println!("(`--gpu` picks it), or the compiler will silently fall back to the");
            println!("analytical model while a measured store sits right there.");
        }
        return Ok(());
    }

    let qualified = all.iter().filter(|r| r.state.is_selectable()).count();
    println!("records     : {} ({qualified} qualified)", all.len());
    println!();

    // 1. THE DIGEST CENSUS.
    println!("digest census — which build each record is keyed to:");
    let mut current_selectable = 0usize;
    for (d, rs) in group_by_digest(&all) {
        let changed = d.stale_against(want);
        let sel = rs.iter().filter(|r| r.state.is_selectable()).count();
        // `qualified`, NOT `selectable`. `RecordState::is_selectable` is the record's own
        // qualification — it passed the oracle and carries enough samples — and says nothing
        // about whether this build can use it. A stale row that reads "270 selectable" is
        // precisely the reassuring-but-wrong number this command exists to stop printing.
        if changed.is_empty() {
            current_selectable += sel;
            println!(
                "  {:<18} {:>5} records  {:>5} qualified  CURRENT — the compiler can use these",
                short(&d.interpreter),
                rs.len(),
                sel
            );
        } else {
            println!(
                "  {:<18} {:>5} records  {:>5} qualified  STALE — changed: {}",
                short(&d.interpreter),
                rs.len(),
                sel,
                changed.join(", ")
            );
        }
    }
    println!();

    // 2. THE TOTAL-STALENESS ALARM.
    if current_selectable == 0 {
        println!("*** EVERY RECORD IN THIS CELL IS STALE. ***");
        println!();
        println!(
            "The store holds {} record(s) and the compiler can use NONE of them: selection",
            all.len()
        );
        println!("falls back to the analytical model for every shape and reports tier `portable`,");
        println!("while `tuned_tile_selection` keeps passing on whatever other cell has data.");
        println!();
        println!("The key is the preprocessed dense-GEMM family digest. Re-run the campaign");
        println!("against the current family when its implementation or toolchain changes:");
        println!("  plowc --hf-dir <ckpt> --max-ctx <c> --n-cu <n> --num-gpus <g> \\");
        println!("        tune gemm --obj <objdir> --samples <out.jsonl>");
        return Err("every record in this cell is stale; the compiler has no measurements".into());
    }

    let (best, stale) = store.best_for(cell, want)?;
    println!(
        "selectable  : {} op cases against the probed build",
        best.len()
    );
    if !stale.is_empty() {
        println!(
            "stale       : {} record(s) exist but cannot be used",
            stale.len()
        );
    }

    // 3. COVERAGE, last.
    let Some(spec) = coverage_from else {
        println!();
        println!("coverage    : not checked — pass `--hf-dir <checkpoint>` before `tune status`");
        println!("              to derive the compiler's demand and census HIT/MISS against it.");
        return Ok(());
    };
    println!();
    let shapes = demand::derive(spec)?;
    let (hit, miss) = demand::coverage(&shapes);
    println!(
        "demand      : {} distinct shapes from {}",
        shapes.len(),
        spec.hf_dir.display()
    );
    println!("coverage    : {hit} HIT / {miss} MISS");
    if miss > 0 {
        println!("              the {miss} MISS shapes select from the ANALYTICAL MODEL. Close");
        println!("              them with `tune gemm` (its --shapes auto derives this same list).");
    }
    Ok(())
}

/// Compare each op case's timing across the build digests the store has seen, newest last, and
/// report where it moved by more than `threshold` (a fraction, e.g. `0.10` for 10%).
///
/// This is the read that found `gemm_c4`'s ~30% regression. A tile can get slower without any
/// campaign noticing, because each campaign only ever looks at its own digest.
pub fn regress(db: &PathBuf, cell: &str, threshold: f64) -> Result<(), Err> {
    let store = TuneStore::new(db.clone());
    let all = store.load_kernels(cell)?;
    if all.is_empty() {
        println!("no records for {cell}.");
        return Ok(());
    }

    // Timeline = FILE ORDER. The store has no timestamps; `publish` appends. Said out loud
    // because it is a proxy, and a hand-edited or re-sorted store would silently invert it.
    let mut order: Vec<String> = Vec::new();
    for r in &all {
        if !order.contains(&r.digests.interpreter) {
            order.push(r.digests.interpreter.clone());
        }
    }
    println!("cell        : {cell}");
    println!(
        "timeline    : {} build digest(s), oldest first (file order):",
        order.len()
    );
    for d in &order {
        println!("  {}", short(d));
    }
    if order.len() < 2 {
        println!();
        println!("only one build digest present — nothing to compare against.");
        return Ok(());
    }
    println!();

    // (op_case, kernel) -> digest -> median
    let mut series: BTreeMap<(String, String), BTreeMap<String, f64>> = BTreeMap::new();
    for r in all
        .iter()
        .filter(|r| matches!(r.correctness, tunedb::Correctness::Pass))
    {
        series
            .entry((r.op_case.clone(), r.kernel_name.clone()))
            .or_default()
            // Last write wins within a digest: `publish` is atomic per campaign, so the later
            // row is the later measurement of the same thing.
            .insert(r.digests.interpreter.clone(), r.stats.median_ns);
    }

    // First-to-last drift per (op case, kernel).
    let mut drift: BTreeMap<(String, String), (f64, f64, f64, usize)> = BTreeMap::new();
    for ((case, kernel), by_digest) in &series {
        let seen: Vec<&String> = order
            .iter()
            .filter(|d| by_digest.contains_key(*d))
            .collect();
        if seen.len() < 2 {
            continue;
        }
        let (first, last) = (by_digest[seen[0]], by_digest[seen[seen.len() - 1]]);
        if first <= 0.0 {
            continue;
        }
        drift.insert(
            (case.clone(), kernel.clone()),
            ((last - first) / first, first, last, seen.len()),
        );
    }

    let mut worst: Vec<(f64, String)> = Vec::new();
    for ((case, kernel), (delta, first, last, n)) in &drift {
        if delta.abs() < threshold {
            continue;
        }
        // THE DISCRIMINATOR. A rung that moved while its SIBLING rungs at the same op case
        // stayed flat is a tile regression. A rung that moved along with all of them is a
        // shape-wide or machine-wide shift (a clock, a harness, a driver) and blaming the tile
        // for it is how a real cause goes unlooked-for. `gemm_c4` qualified on exactly this
        // test: ~30% while every other rung held within 2%.
        let siblings: Vec<f64> = drift
            .iter()
            .filter(|((c, kn), _)| c == case && kn != kernel)
            .map(|(_, (d, ..))| *d)
            .collect();
        let verdict = if siblings.is_empty() {
            "only rung measured"
        } else if siblings.iter().all(|d| d.abs() < threshold / 2.0) {
            "TILE-SPECIFIC — siblings flat"
        } else {
            "shape-wide — siblings moved too"
        };
        worst.push((
            *delta,
            format!(
                "  {:+7.1}%  {case:<32} {kernel:<24} {:>9.1} -> {:>9.1} ns  ({n} digests)  {verdict}",
                delta * 100.0,
                first,
                last
            ),
        ));
    }
    if worst.is_empty() {
        println!(
            "no op case moved by {:.0}% or more across digests.",
            threshold * 100.0
        );
        return Ok(());
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("finite"));
    println!(
        "moved by >= {:.0}% between the first and last digest that measured it:",
        threshold * 100.0
    );
    println!("(positive = SLOWER now)");
    for (_, line) in &worst {
        println!("{line}");
    }
    println!();
    println!("A regression here is invisible to every campaign, because a campaign only ever");
    println!("looks at its own digest. Nothing else in the toolchain compares across them.");
    println!();
    println!("Read TILE-SPECIFIC rows first. A `shape-wide` row is more likely the machine or the");
    println!("harness than the kernel, and a 2-3 digest span is a short window to conclude from.");
    Ok(())
}

fn group_by_digest(all: &[KernelMeasurement]) -> Vec<(Digests, Vec<&KernelMeasurement>)> {
    let mut out: Vec<(Digests, Vec<&KernelMeasurement>)> = Vec::new();
    for r in all {
        match out.iter_mut().find(|(d, _)| *d == r.digests) {
            Some((_, v)) => v.push(r),
            None => out.push((r.digests.clone(), vec![r])),
        }
    }
    out
}

/// Build labels are long; the distinguishing part is the hash tail the digest census is read by.
fn short(label: &str) -> String {
    match label.rsplit_once('-') {
        Some((_, tail)) if tail.len() >= 8 => tail.to_string(),
        _ => label.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tunedb::{Correctness, RecordState, Stats};

    fn rec(case: &str, kernel: &str, interp: &str, ns: f64) -> KernelMeasurement {
        KernelMeasurement {
            op_case: case.into(),
            kernel_id: 1,
            kernel_name: kernel.into(),
            profile: "prefill_dense".into(),
            hardware: "amd/gfx950/mi350x".into(),
            sku: "MI355X".into(),
            digests: Digests {
                implementation: interp.into(),
                interpreter: interp.into(),
                toolchain: "rocm-7.2".into(),
                oracle: "gemm-f64-dot-spotcheck-v1".into(),
            },
            stats: Stats::from_samples(vec![ns; 5]).unwrap(),
            correctness: Correctness::Pass,
            registers: None,
            state: RecordState::Qualified,
            campaign: "t".into(),
        }
    }

    #[test]
    fn the_digest_census_separates_records_that_look_identical_in_a_total() {
        let all = vec![
            rec("gemm/a/None", "gemm_c4", "gfx950-aaaaaaaaaaaaaaaa", 10.0),
            rec("gemm/b/None", "gemm_c4", "gfx950-aaaaaaaaaaaaaaaa", 11.0),
            rec("gemm/a/None", "gemm_c4", "gfx950-bbbbbbbbbbbbbbbb", 13.0),
        ];
        let g = group_by_digest(&all);
        assert_eq!(g.len(), 2, "two build identities, not one pile of three");
        assert_eq!(g[0].1.len(), 2);
    }

    /// The label tail is what a digest is read by in practice (`b2f50de835dd9495`).
    #[test]
    fn digest_labels_shorten_to_their_hash() {
        assert_eq!(short("gfx950-b2f50de835dd9495"), "b2f50de835dd9495");
        assert_eq!(short("short"), "short");
    }

    /// A digest whose records are all stale must contribute ZERO selectable, which is what makes
    /// the total-staleness alarm fire instead of a reassuring record count.
    #[test]
    fn a_stale_digest_contributes_no_selectable_records() {
        let all = vec![rec("gemm/a/None", "gemm_c4", "gfx950-old", 10.0)];
        let want = Digests {
            implementation: "gfx950-new".into(),
            interpreter: "gfx950-new".into(),
            toolchain: "rocm-7.2".into(),
            oracle: "gemm-f64-dot-spotcheck-v1".into(),
        };
        let g = group_by_digest(&all);
        assert!(!g[0].0.stale_against(&want).is_empty());
    }
}

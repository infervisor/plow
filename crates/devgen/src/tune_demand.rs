//! What the compiler ACTUALLY ASKS the tuning store about, recorded so a campaign can be
//! derived from it instead of authored by hand.
//!
//! # Why this module exists
//!
//! A shape list that is **authored by hand** drifts from the compiler's demand, and no guard can
//! see the drift: the guard reads the records that exist, never the lookups that missed. So
//! `tuned_tile_selection` stays green as long as *some* qualified record exists for *some* model,
//! and the calibration tier still reads `measured` while a given model's prefill selects entirely
//! from the analytical model.
//!
//! **Where that actually costs something: Gemma-31B.** Its census reads **100 HIT / 1073 MISS**,
//! and GEMM genuinely dominates prefill there (`op_gemm.h:271`'s ~61% figure is a *Gemma* number).
//! That is the case `--shapes auto` is justified on.
//!
//! **How it was DISCOVERED: GLM-5.2** — and the discovery's headline number does not survive
//! measurement, so it is recorded correctly here rather than repeated. `PLOW_TUNE_DUMP=1` on a GLM
//! TP4 emit printed 32 distinct shapes, all 32 MISS, because every `M>=256` record in the store
//! was a Gemma-31B or Qwen shape (`K` in {2560, 4096, 5376, 8192, 21504}) and GLM's `K` is 6144.
//! But the interleaved A/B of tuned vs untuned on GLM-5.2 TP4 measured **|Δ| < 0.3% of TTFT**,
//! with the sign flipping between mean and median at both 4k and 16k and within-arm spread (2.07%)
//! exceeding every between-arm delta. The tuner *works* there — 10 of 32 shapes get a different
//! tile and the packets differ by 771 bytes — it simply governs almost nothing, because GLM-5.2
//! prefill is MLA flash plus a 256-expert MoE and dense GEMM is ~1.3% of 4k TTFT.
//!
//! So: GLM exposed the mechanism, Gemma pays for it. Do not re-quote GLM as the impact.
//!
//! # And the thing that outranks coverage
//!
//! Coverage gaps are cheap and visible. **Total silent staleness is neither**, and it is what has
//! actually bitten: the store read `0 HIT / 2472 MISS` on this tree against a campaign ingested
//! hours earlier, because the key is the PREPROCESSED `interp.hip` build digest and any
//! `runtime/amd/*.hip|h` edit re-stales every record at once. That is why `plowc tune status`
//! reports the digest census, and why it treats "records exist and none are selectable" as a
//! failure rather than a line of output. This module supplies the other half of that report — the
//! demand to check coverage *against*.
//!
//! # Why a recorder and not just the `eprintln!`
//!
//! `PLOW_TUNE_DUMP=1`'s `TUNEDUMP …` lines are kept, byte-identical, because scripts and reports
//! quote them. But stderr scraping is a second parser for a fact the process already has. The
//! recorder is the same fact, in the same process, with a type — so `--shapes auto` cannot drift
//! from `pick_tile`'s real lookups the way a hand list drifted from the compiler.
//!
//! # Coupling to note
//!
//! The demand is a function of the **prefill bucket ladder** (`PLOW_MLA_PREFILL`): each bucket M
//! generates its own `(M, N, K)` lookups. `scripts/rebench_emit_glm.sh`'s header records that
//! coupling and the rule that follows from it — change the ladder, re-derive the shapes. With
//! `--shapes auto` the rule enforces itself, because the ladder is read off the same emit.

#[cfg(not(test))]
use std::sync::Mutex;

use kernelcaps::QuantScheme;

/// One dense-GEMM tuning-store lookup the compiler performed while emitting.
///
/// The DECODE-GEMV twin of this census lives in `packet::devbuild` (`TUNEDUMP_GEMV`, hooked at
/// `Builder::emit_dep`) with its own store family in `tunedb::gemv`. Two censuses because they
/// observe two different choke points — `pick_tile` here, packet emission there — not because
/// either is a copy of the other.
#[derive(Clone, Debug, PartialEq)]
pub struct Demand {
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub quant: QuantScheme,
    /// Whether the store answered. `false` is the interesting case: that shape's tile came from
    /// the analytical model, whatever tier the build reported.
    pub hit: bool,
}

impl Demand {
    /// A stable, human-readable name for this shape, used as the harness's label argument and in
    /// campaign output. Derived, never authored — the whole point.
    pub fn label(&self) -> String {
        format!("auto-{}x{}x{}-{:?}", self.m, self.n, self.k, self.quant)
    }
}

/// `None` = not recording. Recording is opt-in so a normal compile pays nothing but the lock-free
/// `is_none` check, and so a long-running emit cannot accumulate an unbounded log nobody reads.
#[cfg(not(test))]
static SINK: Mutex<Option<Vec<Demand>>> = Mutex::new(None);
#[cfg(test)]
thread_local! {
    static TEST_SINK: std::cell::RefCell<Option<Vec<Demand>>> = const {
        std::cell::RefCell::new(None)
    };
}

/// Begin recording, discarding anything already collected.
///
/// Emission is single-process and the emitters are not concurrent across models, so a global is
/// the honest shape here: there is exactly one compile per process, and `pick_tile` is reached
/// from too many call sites to thread a context through without touching every emitter.
pub fn start_recording() {
    #[cfg(not(test))]
    {
        *SINK.lock().expect("tune-demand sink") = Some(Vec::new());
    }
    #[cfg(test)]
    TEST_SINK.with_borrow_mut(|sink| *sink = Some(Vec::new()));
}

/// Stop recording and return what was collected, in lookup order (duplicates included — the
/// caller decides whether repetition is signal).
pub fn take() -> Vec<Demand> {
    #[cfg(not(test))]
    return SINK
        .lock()
        .expect("tune-demand sink")
        .take()
        .unwrap_or_default();
    #[cfg(test)]
    return TEST_SINK.with_borrow_mut(|sink| sink.take().unwrap_or_default());
}

/// Record one lookup. Called from `GemmMeasurements::for_shape`, the single place the compiler
/// asks the store anything about a dense GEMM.
pub(crate) fn record(m: i64, n: i64, k: i64, quant: QuantScheme, hit: bool) {
    // Unchanged from the original inline `eprintln!`, deliberately: reports and scripts quote
    // these lines verbatim.
    if crate::emit_config::active().tune_dump {
        eprintln!(
            "TUNEDUMP {} {} {} {:?} {}",
            m,
            n,
            k,
            quant,
            if hit { "HIT" } else { "MISS" }
        );
    }
    let demand = Demand {
        m,
        n,
        k,
        quant,
        hit,
    };
    #[cfg(not(test))]
    {
        if let Some(v) = SINK.lock().expect("tune-demand sink").as_mut() {
            v.push(demand);
        }
    }
    #[cfg(test)]
    TEST_SINK.with_borrow_mut(|sink| {
        if let Some(v) = sink.as_mut() {
            v.push(demand);
        }
    });
}

#[cfg(not(test))]
static DECIDED_MEASURED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(not(test))]
static DECISIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
thread_local! {
    static TEST_DECIDED_MEASURED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_DECISIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Record what actually DECIDED one tile: the selector's calibration tier.
///
/// Deliberately NOT the same signal as `record`'s `hit` above, and the difference is a bug this
/// file already shipped once. `hit` means "the store held a record for this op_case"; it does
/// NOT mean a measurement chose the tile. `select_kernel` uses measurements "only if EVERY
/// candidate has one" (`kernelcaps::select.rs:175`), so a shape with 4 of 5 candidates measured
/// falls back to the analytical model while every lookup still reports HIT.
///
/// That is not hypothetical. On 2026-08-09 a freshly published, wholly CURRENT gfx942 campaign
/// covered 48/48 demanded shapes and changed NOTHING: the ingest rung table maps tiles to
/// opcodes by GEOMETRY, and on gfx942 `Gemm` and `GemmC5` share 192x256x64, so every timing
/// landed on `GemmC5` and opcode 8 got zero records. Five independent signals said "measured"
/// -- `tune status`, `tune shapes`, the guard test, and `build.json`'s own `tile_source`, all of
/// which read the `hit` tally -- while the emitted packet was byte-identical to a `--no-tuning`
/// build. Provenance has to come from the DECISION, not from the lookup.
pub fn note_decision(measured: bool) {
    #[cfg(not(test))]
    {
        DECISIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if measured {
            DECIDED_MEASURED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    #[cfg(test)]
    {
        TEST_DECISIONS.set(TEST_DECISIONS.get() + 1);
        if measured {
            TEST_DECIDED_MEASURED.set(TEST_DECIDED_MEASURED.get() + 1);
        }
    }
}

/// `(decided_by_measurement, decisions)` over every dense-GEMM tile this process selected.
pub fn tally() -> (usize, usize) {
    #[cfg(not(test))]
    return (
        DECIDED_MEASURED.load(std::sync::atomic::Ordering::Relaxed),
        DECISIONS.load(std::sync::atomic::Ordering::Relaxed),
    );
    #[cfg(test)]
    return (TEST_DECIDED_MEASURED.get(), TEST_DECISIONS.get());
}

#[cfg(test)]
pub(crate) fn reset_tally() {
    TEST_DECIDED_MEASURED.set(0);
    TEST_DECISIONS.set(0);
}

/// Collapse a lookup log into the distinct shapes a campaign must cover.
///
/// Sorted by `(K, N, M)` so the output is deterministic and shapes that share a weight matrix sit
/// together — a campaign run reads as a walk up each operator's bucket ladder, which is how the
/// hand-authored list was grouped and what makes a diff against it readable.
pub fn distinct(mut log: Vec<Demand>) -> Vec<Demand> {
    log.sort_by_key(|d| (d.k, d.n, d.m, format!("{:?}", d.quant)));
    log.dedup_by(|a, b| (a.m, a.n, a.k, a.quant) == (b.m, b.n, b.k, b.quant));
    log
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Not recording must cost nothing and lose nothing: `take` on a fresh sink is empty, and a
    /// record taken while disabled does not resurrect later.
    #[test]
    fn recording_is_opt_in() {
        let _ = take();
        record(1, 2, 3, QuantScheme::None, false);
        assert!(
            take().is_empty(),
            "a lookup outside a recording window is not kept"
        );
    }

    #[test]
    fn distinct_collapses_repeats_and_orders_deterministically() {
        let mk = |m, n, k| Demand {
            m,
            n,
            k,
            quant: QuantScheme::None,
            hit: false,
        };
        let out = distinct(vec![
            mk(8192, 64, 6144),
            mk(512, 64, 6144),
            mk(8192, 64, 6144),
            mk(512, 6144, 512),
        ]);
        assert_eq!(out.len(), 3, "the repeated lookup collapses");
        assert_eq!(
            (out[0].m, out[0].n, out[0].k),
            (512, 6144, 512),
            "K ascending"
        );
        assert_eq!((out[1].m, out[1].n, out[1].k), (512, 64, 6144));
        assert_eq!((out[2].m, out[2].n, out[2].k), (8192, 64, 6144));
    }

    /// The same shape under two quant schemes is two campaign entries: a bf16 timing must never
    /// be served for an fp8 op (`tunedb::gemm_op_case` keys on quant for exactly this reason).
    #[test]
    fn quant_is_part_of_the_shape_identity() {
        let out = distinct(vec![
            Demand {
                m: 512,
                n: 6144,
                k: 512,
                quant: QuantScheme::None,
                hit: false,
            },
            Demand {
                m: 512,
                n: 6144,
                k: 512,
                quant: QuantScheme::W8A8,
                hit: false,
            },
        ]);
        assert_eq!(out.len(), 2);
    }
}

/// Print the tile-provenance verdict ONCE per process, at emit time.
///
/// The point is that a compile should say what it is when it is being made, not leave the
/// answer to be excavated afterwards. plowc already warns when records are STALE, but that
/// warning cannot fire in the case that actually happened on 2026-08-09: a store that is
/// CURRENT and COMPLETE by every count (48/48 op cases, 192 qualified records at the live
/// digest) and still decides nothing, because `select_kernel` needs EVERY candidate measured
/// and one rung had zero records. Records-present and measurement-used are different facts and
/// only the second one matters.
pub fn report() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Ordering::Relaxed) {
        return;
    }
    let (measured, total) = tally();
    if total == 0 {
        return;
    }
    if measured == total {
        eprintln!("  tunedb: all {total} dense-GEMM tile(s) chosen BY MEASUREMENT");
    } else if measured == 0 {
        eprintln!(
            "  >>> tunedb: ALL {total} dense-GEMM tile(s) chosen by the ANALYTICAL MODEL. This \
             build is UNMEASURED -- `pick_tile` reports tier `portable`, which is what it \
             reports when no campaign has ever run. Diagnose with `plowc tune status --gpu \
             <gpu>`: if records are STALE, re-run the campaign against THIS object recipe \
             (defines are part of the digest); if they are CURRENT, check the per-opcode census \
             -- a rung with zero records disqualifies every shape it is a candidate for, \
             because selection uses measurements only if EVERY candidate has one."
        );
    } else {
        eprintln!(
            "  >>> tunedb: {measured} of {total} dense-GEMM tile(s) chosen by measurement; the \
             remaining {} fell back to the ANALYTICAL MODEL. Mixed provenance -- `plowc tune \
             status --gpu <gpu>` names which shapes are uncovered.",
            total - measured
        );
    }
}

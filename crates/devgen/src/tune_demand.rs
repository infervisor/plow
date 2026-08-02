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
static SINK: Mutex<Option<Vec<Demand>>> = Mutex::new(None);

/// Begin recording, discarding anything already collected.
///
/// Emission is single-process and the emitters are not concurrent across models, so a global is
/// the honest shape here: there is exactly one compile per process, and `pick_tile` is reached
/// from too many call sites to thread a context through without touching every emitter.
pub fn start_recording() {
    *SINK.lock().expect("tune-demand sink") = Some(Vec::new());
}

/// Stop recording and return what was collected, in lookup order (duplicates included — the
/// caller decides whether repetition is signal).
pub fn take() -> Vec<Demand> {
    SINK.lock()
        .expect("tune-demand sink")
        .take()
        .unwrap_or_default()
}

/// Record one lookup. Called from `GemmMeasurements::for_shape`, the single place the compiler
/// asks the store anything about a dense GEMM.
pub(crate) fn record(m: i64, n: i64, k: i64, quant: QuantScheme, hit: bool) {
    // Unchanged from the original inline `eprintln!`, deliberately: reports and scripts quote
    // these lines verbatim.
    if std::env::var("PLOW_TUNE_DUMP").ok().as_deref() == Some("1") {
        eprintln!(
            "TUNEDUMP {} {} {} {:?} {}",
            m,
            n,
            k,
            quant,
            if hit { "HIT" } else { "MISS" }
        );
    }
    if let Some(v) = SINK.lock().expect("tune-demand sink").as_mut() {
        v.push(Demand {
            m,
            n,
            k,
            quant,
            hit,
        });
    }
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

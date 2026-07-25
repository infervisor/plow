//! Hybrid tile *exploration* — egglog/datalog selection over `costmodel` costs.
//!
//! The split is deliberate (the user's "hybrid"):
//!
//! * **Rust ([`costmodel`])** owns what is *legal* and what it *costs* — the full
//!   model (MMA legality, paged SRAM, DMA bytes, streaming passes) lives in Rust,
//!   where it can use real arithmetic.
//! * **egglog/datalog** owns the *selection*: candidate costs are asserted as
//!   facts, an argmin is computed declaratively with a `:merge (min …)` function,
//!   and the winning row is read back.
//!
//! For a single isolated GEMM, [`costmodel::CostModel::best_tile`]'s Rust argmin
//! is equivalent (and the tests assert exactly that). The egglog layer earns its
//! keep when the selection is *joint*: a chosen tile feeds colocation / SRAM
//! hand-off facts (§2.5) that change another op's cost in the same saturation —
//! which a per-op Rust argmin cannot see. This module is that seam.

use costmodel::{CostModel, GemmShape, SramPolicy, TileShape};
use std::collections::HashMap;

#[derive(thiserror::Error, Debug)]
pub enum ExploreError {
    #[error("egglog error: {0}")]
    Egglog(String),
    #[error("no candidate for choice point {0}")]
    NoCandidate(i64),
}

/// One candidate at a choice point: an opaque `tag` (index back into the
/// caller's candidate list) with its integer `cost`.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub tag: i64,
    pub cost: u64,
}

/// A choice point (e.g. one GEMM) with its mutually-exclusive candidates.
#[derive(Clone, Debug)]
pub struct ChoicePoint {
    pub id: i64,
    pub candidates: Vec<Candidate>,
}

/// Select the minimum-cost candidate at each choice point via egglog datalog.
///
/// Returns `id → chosen tag`. egglog computes the argmin declaratively; this is
/// the generic engine every op's tiler funnels through.
pub fn select(points: &[ChoicePoint]) -> Result<HashMap<i64, i64>, ExploreError> {
    let mut prog = String::new();
    prog.push_str(
        "(relation cand (i64 i64 i64))\n\
         (function best (i64) i64 :merge (min old new))\n\
         (relation chosen (i64 i64))\n",
    );
    for p in points {
        for c in &p.candidates {
            prog.push_str(&format!("(cand {} {} {})\n", p.id, c.tag, c.cost));
        }
    }
    // Reduce each choice point's candidates to their minimum cost, then keep the
    // candidate(s) achieving it.
    prog.push_str(
        "(rule ((cand ?id ?t ?c)) ((set (best ?id) ?c)))\n\
         (rule ((cand ?id ?t ?c) (= ?c (best ?id))) ((chosen ?id ?t)))\n\
         (run 100)\n\
         (print-function chosen 1000000)\n",
    );

    let mut egraph = egglog::EGraph::default();
    let msgs = egraph
        .parse_and_run_program(None, &prog)
        .map_err(|e| ExploreError::Egglog(e.to_string()))?;

    let text = msgs
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let mut chosen: HashMap<i64, i64> = HashMap::new();
    for (id, tag) in parse_chosen(&text) {
        // On a cost tie multiple rows fire; keep the first (smallest tag) seen.
        chosen.entry(id).or_insert(tag);
    }

    for p in points {
        if !chosen.contains_key(&p.id) {
            return Err(ExploreError::NoCandidate(p.id));
        }
    }
    Ok(chosen)
}

/// Scan egglog's `print-function` output for `(chosen <id> <tag>)` rows. Robust
/// to whatever line-wrapping egglog applies to a relation dump.
fn parse_chosen(text: &str) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find("(chosen") {
        rest = &rest[pos + "(chosen".len()..];
        let ints: Vec<i64> = rest
            .split(|c: char| !(c.is_ascii_digit() || c == '-'))
            .filter(|s| !s.is_empty())
            .take(2)
            .filter_map(|s| s.parse().ok())
            .collect();
        if ints.len() == 2 {
            out.push((ints[0], ints[1]));
        }
    }
    out
}

/// One GEMM to tile, tagged with a stable `id`.
#[derive(Clone, Copy, Debug)]
pub struct GemmJob {
    pub id: i64,
    pub g: GemmShape,
}

/// The tile chosen for a job, with its cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Choice {
    pub id: i64,
    pub tile: TileShape,
    pub cycles: u64,
}

/// Explore + select GEMM tilings: `costmodel` enumerates legal candidates and
/// costs them; egglog picks the argmin per job.
pub fn explore_tiles(
    cm: &CostModel,
    jobs: &[GemmJob],
    policy: SramPolicy,
) -> Result<Vec<Choice>, ExploreError> {
    // Per job, the legal candidates and their costs (tag = index into this vec).
    //
    // §8.4 dominance pruning is applied before pricing: any tile that is
    // Pareto-dominated in (passes, sram_pages, output_tiles) is provably
    // ≥-cost by `Plow.Cost.dominates_implies_cost_le`, so dropping it cannot
    // change the argmin. This shrinks the fact set egglog reasons about.
    let per_job: Vec<(GemmJob, Vec<(TileShape, u64)>)> = jobs
        .iter()
        .map(|j| {
            let raw = cm.candidates(j.g, policy);
            let (pruned, _rep) = costmodel::dominance::prune_dominated(&raw, j.g, cm);
            let cands = pruned
                .into_iter()
                .map(|t| (t, cm.gemm_cost(j.g, t)))
                .collect();
            (*j, cands)
        })
        .collect();

    let points: Vec<ChoicePoint> = per_job
        .iter()
        .map(|(j, cands)| ChoicePoint {
            id: j.id,
            candidates: cands
                .iter()
                .enumerate()
                .map(|(i, &(_, cost))| Candidate {
                    tag: i as i64,
                    cost,
                })
                .collect(),
        })
        .collect();

    let chosen = select(&points)?;

    let mut out = Vec::new();
    for (j, cands) in &per_job {
        let tag = chosen
            .get(&j.id)
            .copied()
            .ok_or(ExploreError::NoCandidate(j.id))?;
        let (tile, cycles) = cands[tag as usize];
        out.push(Choice {
            id: j.id,
            tile,
            cycles,
        });
    }
    Ok(out)
}

// --- Cost-driven chunk count (CHUNK-2) --------------------------------------

/// Inputs to the chunk-count cost model for one producer→consumer prefill pair.
///
/// All cycle figures are full-op (`k=1`) costs from [`CostModel`]; the model
/// derives the `k`-dependent pipeline time from them.
#[derive(Clone, Copy, Debug)]
pub struct ChunkCostIn {
    /// Producer op full compute cycles.
    pub producer_cycles: u64,
    /// Consumer op full compute cycles.
    pub consumer_cycles: u64,
    /// Per-chunk launch/counter-gate overhead (cycles). In the persistent
    /// double-buffered megakernel this is the counter wait + smem double-buffer
    /// refill latency, *not* a full kernel relaunch.
    pub gate_cycles: u64,
    /// Producer row extent (Gemm `M` / seq len) and its tile row-block `bm`, for
    /// the wave-quantization tail when rows-per-chunk don't tile evenly.
    pub m_rows: i64,
    pub bm: i64,
    /// Total producer output bytes (the tensor the consumer re-reads). Split
    /// across `k` chunks; when a chunk's slice fits `l2_bytes` the consumer reads
    /// it hot from L2 instead of HBM.
    pub out_bytes: u64,
    /// L2 reuse budget in bytes (the GPU's L2 available to keep a chunk output
    /// resident for the consumer).
    pub l2_bytes: u64,
    /// HBM bytes moved per cycle (the re-read cost L2 residency saves).
    pub hbm_bytes_per_cycle: f64,
}

/// Wave-quantization tail: rows-per-chunk rounded up to `bm` wastes producer
/// work when `m_rows/k` is not a multiple of `bm`. Returns the wasted-work cost
/// in cycles (producer time scaled by the wasted-row fraction).
fn wave_tail(k: u32, producer_cycles: u64, m_rows: i64, bm: i64) -> u64 {
    if m_rows <= 0 || bm <= 0 {
        return 0;
    }
    let k = k.max(1) as i64;
    let per_chunk_rows = (m_rows + k - 1) / k; // ceil
    let tiles = (per_chunk_rows + bm - 1) / bm; // ceil
    let covered = tiles * bm * k;
    let wasted = (covered - m_rows).max(0);
    (producer_cycles as u128 * wasted as u128 / m_rows as u128) as u64
}

/// Modeled prefill time (cycles) for running a producer→consumer pair as `k`
/// row-chunks through the double-buffered, statically-colocated pipeline.
///
/// The cost model reflects the physics of the underlying [`crate::CostModel`]:
/// its GEMM cost is *aggregate SM-work*, so chunking cannot reduce the compute
/// floor `P + C` (the chip does the same total work however the rows are grouped,
/// and two compute-bound ops overlapped on the chip still contend for the same
/// SMs). The lever chunking actually pulls is **memory**:
///
/// ```text
/// time(k) = (P + C)                       // compute floor (invariant in k)
///         + hbm_roundtrip                 // IF a chunk's output spills L2 → producer
///                                         //   stores to HBM and consumer re-reads it
///         + k · gate                      // per-chunk counter/sync + double-buffer prime
///         + tail(k)                       // wave-quant tail when rows don't tile to bm
///
/// hbm_roundtrip = 2 · out_bytes / hbm_bytes_per_cycle
/// spills L2  ⇔  out_bytes / k  >  l2_bytes
/// ```
///
/// When a chunk's output slice fits the L2 reuse budget the consumer reads the
/// producer's output *hot* from L2 — the whole store+load HBM round-trip is
/// eliminated (static colocation keeps producer and consumer on the same L2
/// slice). Below that `k` the round-trip is paid; above it every extra chunk only
/// adds gate overhead. The argmin is therefore the **largest chunk that still
/// fits L2** — `k* = ⌈out_bytes / l2_bytes⌉` — the chunk-thesis sweet spot. `k*`
/// rises with context length (bigger `M` ⇒ bigger producer output ⇒ more chunks
/// to stay L2-resident).
pub fn chunk_prefill_cycles(k: u32, i: &ChunkCostIn) -> u64 {
    let k = k.max(1);
    let compute_floor = i.producer_cycles.saturating_add(i.consumer_cycles);
    let overhead = i.gate_cycles.saturating_mul(k as u64);
    let tail = wave_tail(k, i.producer_cycles, i.m_rows, i.bm);
    // Producer-output HBM round-trip (store + re-read), paid only when a chunk's
    // output slice does not fit the L2 reuse budget.
    let per_chunk_out = i.out_bytes / k as u64;
    let hbm_roundtrip = if i.hbm_bytes_per_cycle > 0.0 && per_chunk_out > i.l2_bytes {
        (2.0 * i.out_bytes as f64 / i.hbm_bytes_per_cycle) as u64
    } else {
        0
    };
    compute_floor
        .saturating_add(hbm_roundtrip)
        .saturating_add(overhead)
        .saturating_add(tail)
}

/// Rust argmin over `k ∈ 1..=k_max` of [`chunk_prefill_cycles`]. Returns
/// `(best_k, cycles)`.
pub fn best_chunk_count(k_max: u32, i: &ChunkCostIn) -> (u32, u64) {
    (1..=k_max.max(1))
        .map(|k| (k, chunk_prefill_cycles(k, i)))
        .min_by_key(|&(_, c)| c)
        .unwrap()
}

/// Same argmin, but routed through the egglog [`select`] engine: each `k` is
/// asserted as a candidate fact and egglog computes the argmin declaratively.
///
/// For an *isolated* prefill pair (no cross-op SRAM colocation coupling one op's
/// cost to another op's `k`) the selection is a plain 1-D scan, so this returns
/// exactly what [`best_chunk_count`] does — the tests assert that equality. The
/// e-graph earns its keep only when the choice is *joint* (a chosen `k` changes a
/// sibling op's cost in the same saturation), which the linear prefill chain's
/// single shared `k` does not create. We therefore use the Rust argmin in the
/// hot path and keep egglog as the equivalence oracle.
pub fn best_chunk_count_egglog(k_max: u32, i: &ChunkCostIn) -> Result<u32, ExploreError> {
    let candidates: Vec<Candidate> = (1..=k_max.max(1))
        .map(|k| Candidate {
            tag: k as i64,
            cost: chunk_prefill_cycles(k, i),
        })
        .collect();
    let chosen = select(&[ChoicePoint { id: 0, candidates }])?;
    Ok(chosen[&0] as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use costmodel::DEFAULT_PAGE_BYTES;

    fn h100() -> &'static costmodel::hwspec::GpuSpec {
        costmodel::hwspec::registry::lookup("H100 SXM5").unwrap()
    }

    #[test]
    fn datalog_picks_the_min_cost_candidate() {
        let points = vec![ChoicePoint {
            id: 7,
            candidates: vec![
                Candidate { tag: 0, cost: 900 },
                Candidate { tag: 1, cost: 100 }, // cheapest
                Candidate { tag: 2, cost: 500 },
            ],
        }];
        let chosen = select(&points).unwrap();
        assert_eq!(chosen[&7], 1);
    }

    #[test]
    fn hybrid_matches_rust_argmin() {
        // egglog selection must agree with costmodel's own best_tile.
        let cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };
        let jobs = [GemmJob { id: 0, g }];

        let chosen = explore_tiles(&cm, &jobs, SramPolicy::Stream).unwrap();
        let (best, cost) = cm.best_tile(g, SramPolicy::Stream).unwrap();
        assert_eq!(chosen.len(), 1);
        assert_eq!((chosen[0].tile, chosen[0].cycles), (best, cost));
    }

    #[test]
    fn selects_per_job_independently() {
        let cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        let jobs = [
            GemmJob {
                id: 0,
                g: GemmShape {
                    m: 4096,
                    n: 4096,
                    k: 4096,
                },
            }, // prefill
            GemmJob {
                id: 1,
                g: GemmShape {
                    m: 1,
                    n: 4096,
                    k: 4096,
                },
            }, // decode
        ];
        let chosen = explore_tiles(&cm, &jobs, SramPolicy::Stream).unwrap();
        assert_eq!(chosen.len(), 2);
        // Decode (M=1) must land on a skinny BM tile.
        let decode = chosen.iter().find(|c| c.id == 1).unwrap();
        assert_eq!(decode.tile.bm, 64);
    }

    fn rtx5090() -> &'static costmodel::hwspec::GpuSpec {
        costmodel::hwspec::registry::lookup("RTX 5090").unwrap()
    }

    /// Cost inputs for the gemma-12B FFN prefill pair (up-proj → down-proj) at
    /// M=2048, hidden=3840, inter=15360, on the RTX 5090 (96 MiB L2, 1792 GB/s).
    fn gemma12b_ffn(m_rows: i64) -> ChunkCostIn {
        let cm = CostModel::new(rtx5090(), DEFAULT_PAGE_BYTES);
        let up = GemmShape {
            m: m_rows,
            n: 15360,
            k: 3840,
        }; // hidden -> inter
        let down = GemmShape {
            m: m_rows,
            n: 3840,
            k: 15360,
        }; // inter -> hidden
        let (up_tile, up_cyc) = cm.best_tile(up, SramPolicy::Stream).unwrap();
        let (_dn_tile, dn_cyc) = cm.best_tile(down, SramPolicy::Stream).unwrap();
        // producer output = up-proj activations [M, inter] bf16.
        let out_bytes = (up.m * up.n) as u64 * 2;
        // RTX 5090: L2 96 MiB; use half as the reuse budget (the rest holds the
        // consumer's streamed weight tile + its own inputs).
        let l2_bytes = 96 * 1024 * 1024 / 2;
        // HBM bytes/cycle @ 2407 MHz, 1792 GB/s ≈ 1792e9 / 2.407e9 ≈ 744 B/cyc.
        let hbm_bytes_per_cycle = 1792.0e9 / 2.407e9;
        ChunkCostIn {
            producer_cycles: up_cyc,
            consumer_cycles: dn_cyc,
            gate_cycles: 2000,
            m_rows,
            bm: up_tile.bm,
            out_bytes,
            l2_bytes,
            hbm_bytes_per_cycle,
        }
    }

    #[test]
    fn chunk_cost_picks_l2_fit_point() {
        // The argmin is the largest chunk that still fits the L2 reuse budget:
        // k* = ceil(out_bytes / l2_bytes). Below it the HBM round-trip is paid;
        // above it only gate overhead grows.
        let i = gemma12b_ffn(2048);
        let expect = (i.out_bytes as f64 / i.l2_bytes as f64).ceil() as u32;
        let (best_k, _) = best_chunk_count(16, &i);
        assert_eq!(best_k, expect, "12B M=2048 should pick the L2-fit k");
        assert!(best_k >= 2, "k=1 spills the producer output to HBM");
        // Optimum is a true interior minimum of the modeled cost.
        let c_best = chunk_prefill_cycles(best_k, &i);
        assert!(chunk_prefill_cycles(1, &i) > c_best); // HBM penalty
        assert!(chunk_prefill_cycles(16, &i) > c_best); // gate overhead
    }

    #[test]
    fn chunk_count_egglog_equals_rust() {
        // The honest-egglog check: for an isolated prefill pair the e-graph argmin
        // must equal the Rust argmin (no joint SRAM-colocation constraint here).
        for m in [512, 2048, 8192] {
            let i = gemma12b_ffn(m);
            let (rust_k, _) = best_chunk_count(16, &i);
            let egg_k = best_chunk_count_egglog(16, &i).unwrap();
            assert_eq!(rust_k, egg_k, "egglog vs Rust disagree at M={m}");
        }
    }

    #[test]
    fn longer_context_wants_more_chunks() {
        // As M grows the producer output overflows L2, so more chunks are needed to
        // keep each chunk's output resident for the consumer (monotone non-decreasing).
        let (k_short, _) = best_chunk_count(32, &gemma12b_ffn(2048));
        let (k_long, _) = best_chunk_count(32, &gemma12b_ffn(16384));
        assert!(
            k_long >= k_short,
            "long context k={k_long} should be ≥ short context k={k_short}"
        );
    }

    #[test]
    #[ignore]
    fn print_chunk_numbers() {
        let cm = CostModel::new(rtx5090(), DEFAULT_PAGE_BYTES);
        let hbm = 1792.0e9 / 2.407e9;
        let l2h = 96 * 1024 * 1024 / 2;
        let mk = |name: &str, up: GemmShape, down: GemmShape, m: i64| {
            let (ut, uc) = cm.best_tile(up, SramPolicy::Stream).unwrap();
            let (_dt, dc) = cm.best_tile(down, SramPolicy::Stream).unwrap();
            let out = (up.m * up.n) as u64 * 2;
            let i = ChunkCostIn {
                producer_cycles: uc,
                consumer_cycles: dc,
                gate_cycles: 2000,
                m_rows: m,
                bm: ut.bm,
                out_bytes: out,
                l2_bytes: l2h,
                hbm_bytes_per_cycle: hbm,
            };
            let (k, cyc) = best_chunk_count(32, &i);
            let c1 = chunk_prefill_cycles(1, &i);
            println!(
                "{name} M={m}: up_tile.bm={} P={uc} C={dc} out={:.1}MiB k*={k} time(k*)={cyc} time(1)={c1} speedup={:.3}x",
                ut.bm,
                out as f64 / 1048576.0,
                c1 as f64 / cyc as f64
            );
        };
        for m in [2048, 8192, 16384] {
            mk(
                "12B",
                GemmShape { m, n: 15360, k: 3840 },
                GemmShape { m, n: 3840, k: 15360 },
                m,
            );
        }
        for m in [2048, 8192, 16384] {
            mk(
                "31B",
                GemmShape { m, n: 21504, k: 5376 },
                GemmShape { m, n: 5376, k: 21504 },
                m,
            );
        }
    }

    #[test]
    fn wave_tail_zero_when_even() {
        // M=2048, bm=128 ⇒ 16 M-tiles; k∈{2,4,8} divide evenly ⇒ no tail.
        let i = gemma12b_ffn(2048);
        // bm may be 128 or 256 depending on the chosen tile; assert tail is 0 for
        // a k that divides the M-tile count.
        let mtiles = (i.m_rows + i.bm - 1) / i.bm;
        for k in [2u32, 4] {
            if mtiles % k as i64 == 0 {
                assert_eq!(wave_tail(k, i.producer_cycles, i.m_rows, i.bm), 0);
            }
        }
    }
}

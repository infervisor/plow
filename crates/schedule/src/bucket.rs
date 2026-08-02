//! Shape buckets — one precompiled packet stream per representative
//! `(batch, seq)` shape, with a **single** shared weight + KV layout (design
//! §10). The hard invariant: flipping between streams moves no weight or KV
//! bytes — only the activation tiling / schedule differs.
//!
//! Pipeline per bucket: `build_plan(bucket)` → [`assemble`] (with the *pinned*
//! `(BN, BK)` so the weight layout is identical across buckets) →
//! [`collapse`](rewrite::collapse) (cost-driven hand-off lowering) →
//! [`relax`](crate::relax) (demote infeasible same-SM hand-offs) → [`schedule`].
//! [`choose_buckets`] picks the bucket ladder from a workload with the cost
//! model; [`choose_weight_tiling`] picks the one shared weight layout.

use crate::{relax, schedule, Config, Machine, Scheduled};
use costmodel::{CostModel, GemmShape, Soc, SramPolicy};
use rewrite::tilegraph::assemble_tuned;
use rewrite::{collapse, ConstraintSet, LayerPlan, OpKind, TileGraph};

// Re-export `Phase` and `WeightLayout` from plow-asset — the single source of truth.
pub use plow_asset::Phase;
pub use plow_asset::WeightLayout;

/// Default KV block size (tokens per cache block). A compile-time choice.
const DEFAULT_KV_BLOCK_SEQ: i64 = 256;
const ELEM: u64 = 2; // bf16

/// A representative compiled shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeBucket {
    pub batch: i64,
    pub seq: i64,
    pub phase: Phase,
}

/// A runtime request shape (rounded up to a bucket at dispatch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub batch: i64,
    pub seq: i64,
    pub phase: Phase,
}

impl ShapeBucket {
    /// GEMM `M` (rows): all tokens in prefill, just the batch in decode.
    pub fn rows(&self) -> i64 {
        match self.phase {
            Phase::Prefill => self.batch * self.seq,
            Phase::Decode => self.batch,
        }
    }
    /// Attention query / KV length.
    pub fn attn_seq(&self) -> i64 {
        self.seq
    }
}

/// The single, shared KV-cache layout + growth, determined by compilation
/// (§10.6–10.8). Only block *addresses* are dynamic at runtime; the layout
/// never changes on flip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayout {
    pub block_seq: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
}

impl KvLayout {
    /// KV blocks a prefill of `seq` tokens appends.
    pub fn blocks_for_prefill(&self, seq: i64) -> i64 {
        let b = self.block_seq.max(1);
        (seq.max(0) + b - 1) / b
    }
    /// Whether decode step `step` (1-based) starts a fresh KV block.
    pub fn appends_block_at(&self, step: i64) -> bool {
        step >= 1 && (step - 1) % self.block_seq.max(1) == 0
    }
    /// Bytes of one KV block (K and V), for the cache allocator. Saturating so a
    /// pathological layout reports `u64::MAX` rather than wrapping to a small size.
    pub fn block_bytes(&self) -> u64 {
        let seq = self.block_seq.max(0) as u64;
        let heads = self.kv_heads.max(0) as u64;
        let dim = self.head_dim.max(0) as u64;
        // 2 (K and V) × ELEM bytes × seq × heads × dim.
        seq.saturating_mul(heads)
            .saturating_mul(dim)
            .saturating_mul(2 * ELEM)
    }
}

/// One bucket's compiled artifacts: the tile graph it assembled (tiles + the
/// pinned weight layout live here), its constraints, and its packet stream.
pub struct BucketStream {
    pub bucket: ShapeBucket,
    pub graph: TileGraph,
    pub cons: ConstraintSet,
    pub sched: Scheduled,
}

/// A compiled model: one shared weight + KV layout, plus a packet stream per
/// bucket. `select` is the runtime flip.
pub struct Compiled {
    pub weight: Option<WeightLayout>,
    /// `true` iff every bucket's GEMM tiles share the same `(BN, BK)` — a flip
    /// between streams moves no weight bytes. `false` when no common tiling
    /// exists and each bucket picked independently (flips may require weight
    /// movement).
    pub weight_shared: bool,
    pub kv: Option<KvLayout>,
    pub streams: Vec<BucketStream>,
}

impl Compiled {
    /// Round a request up to the smallest same-phase bucket that covers it
    /// (`batch ≥` and `seq ≥`), padding the rest. `None` ⇒ exceeds every bucket.
    pub fn select(&self, req: &Request) -> Option<&BucketStream> {
        self.streams
            .iter()
            .filter(|s| {
                s.bucket.phase == req.phase
                    && s.bucket.batch >= req.batch
                    && s.bucket.seq >= req.seq
            })
            .min_by_key(|s| (s.bucket.rows(), s.bucket.seq))
    }
}

/// Compile one packet stream per bucket, sharing a single weight layout (pinned
/// `(BN, BK)`) and KV layout.
pub fn compile_buckets(
    soc: &Soc,
    cfg: &Config,
    buckets: &[ShapeBucket],
    build_plan: impl Fn(&ShapeBucket) -> LayerPlan,
) -> Compiled {
    compile_buckets_tuned(soc, cfg, buckets, build_plan, &rewrite::oracle::NoOracle)
}

/// [`compile_buckets`], with an oracle consulted about kernel availability and
/// measured cost. With `NoOracle` the two are identical.
pub fn compile_buckets_tuned(
    soc: &Soc,
    cfg: &Config,
    buckets: &[ShapeBucket],
    build_plan: impl Fn(&ShapeBucket) -> LayerPlan,
    oracle: &dyn rewrite::oracle::KernelOracle,
) -> Compiled {
    let cm = &soc.unit(0).cm;
    // Reference plan from the largest bucket (most tile options); its GEMM (N,K)
    // pairs + Flash attention config define the shared layouts.
    let reference = buckets.iter().max_by_key(|b| b.rows()).copied();
    let (weight, kv) = match reference {
        Some(rb) => {
            let plan = build_plan(&rb);
            let gemms: Vec<(i64, i64)> = plan
                .ops
                .iter()
                .filter_map(|o| match o.kind {
                    OpKind::Gemm(g) => Some((g.n, g.k)),
                    _ => None,
                })
                .collect();
            let load: Vec<i64> = buckets.iter().map(|b| b.rows()).collect();
            let weight = choose_weight_tiling_tuned(cm, &gemms, &load, SramPolicy::Stream, oracle);
            let kv = plan.ops.iter().find_map(|o| match o.kind {
                OpKind::Flash(a) => Some(KvLayout {
                    block_seq: DEFAULT_KV_BLOCK_SEQ,
                    kv_heads: a.heads,
                    head_dim: a.head_dim,
                }),
                _ => None,
            });
            (weight, kv)
        }
        None => (None, None),
    };

    let weight_shared = weight.is_some();
    let pin = weight.map(|w| (w.bn, w.bk));
    let machine = Machine::from_soc(soc, cfg);
    let mut streams = Vec::new();
    for b in buckets {
        let plan = build_plan(b);
        let (graph, cons) =
            assemble_tuned(soc, &plan, SramPolicy::Stream, pin, oracle).expect("assemble bucket");
        // Cost-driven hand-off lowering (HBM / SRAM / DSM defaults + dma-fold),
        // then relax same-SM choices that would over-subscribe an SM's pages.
        let (graph, cons) = collapse(soc, &graph, &cons);
        let (graph, cons) = relax(&machine, &graph, &cons);
        let sched = schedule(soc, &graph, &cons, cfg);
        streams.push(BucketStream {
            bucket: *b,
            graph,
            cons,
            sched,
        });
    }
    Compiled {
        weight,
        weight_shared,
        kv,
        streams,
    }
}

/// The "common across tiles is an optimization" step: pick the single `(BN, BK)`
/// minimizing the workload-weighted GEMM cost across all buckets — `BM` stays
/// free per bucket, but the weight layout is shared. Returns `None` if there is
/// no `(BN, BK)` legal for every (GEMM, bucket-rows) pair.
pub fn choose_weight_tiling(
    cm: &CostModel,
    gemms: &[(i64, i64)],
    rows: &[i64],
    policy: SramPolicy,
) -> Option<WeightLayout> {
    choose_weight_tiling_tuned(cm, gemms, rows, policy, &rewrite::oracle::NoOracle)
}

/// [`choose_weight_tiling`], restricting the shared `(BN, BK)` to layouts a
/// buildable kernel actually uses.
///
/// This is what keeps the tile filter in `gemm_cands` from routinely emptying:
/// if the pinned weight layout is chosen without regard to the target's
/// kernels, it can exclude the only tile that exists (on NVIDIA it pins
/// 256x64 while the sole `d_gemm` is 128x32). Consulting the oracle here pins a
/// layout the kernel can read. With `NoOracle` this is exactly the analytical
/// choice.
pub fn choose_weight_tiling_tuned(
    cm: &CostModel,
    gemms: &[(i64, i64)],
    rows: &[i64],
    policy: SramPolicy,
    oracle: &dyn rewrite::oracle::KernelOracle,
) -> Option<WeightLayout> {
    let (n, k) = *gemms.iter().max_by_key(|(n, k)| n * k)?;
    let big = GemmShape {
        m: *rows.iter().max().unwrap_or(&1),
        n,
        k,
    };
    let mut pairs: Vec<(i64, i64)> = cm
        .candidates(big, policy)
        .iter()
        .map(|t| (t.bn, t.bk))
        .collect();
    pairs.sort_unstable();
    pairs.dedup();

    // Prefer a shared (BN, BK) that a buildable kernel actually uses, when the
    // oracle knows AND such a layout is also analytically legal across every
    // bucket. `Buildable` is the target's real kernel set; `Analytical` and
    // `Unverified` leave the analytical choice untouched.
    //
    // Deliberately a preference, not a force. Forcing a buildable (BN, BK) that
    // the analytical enumerator never produced empties the legality filter below
    // and yields no layout at all; and on backends whose GEMM kernels differ in
    // BN (gfx950: 256 vs 128), no single shared BN matches all of them, so the
    // `weight_shared` model -- BN/BK fixed, BM varying -- cannot express it.
    // Resolving that is weight-layout/kernel co-design beyond this pass, so the
    // residual disagreement is reported per op (`note_pin_conflict`) rather than
    // silently papered over.
    let q = rewrite::oracle::GemmQuery::bf16(big);
    if let rewrite::oracle::TileAdvice::Buildable(built) = oracle.gemm_tiles(&q) {
        let ok: std::collections::BTreeSet<(i64, i64)> =
            built.iter().map(|t| (t.bn, t.bk)).collect();
        let restricted: Vec<(i64, i64)> =
            pairs.iter().copied().filter(|p| ok.contains(p)).collect();
        if !restricted.is_empty() {
            pairs = restricted;
        }
    }

    pairs
        .into_iter()
        // Legal only if every GEMM at every bucket's M has a tile with this (BN, BK).
        .filter(|&(bn, bk)| {
            gemms.iter().all(|&(n, k)| {
                rows.iter().all(|&m| {
                    cm.candidates(GemmShape { m, n, k }, policy)
                        .iter()
                        .any(|t| t.bn == bn && t.bk == bk)
                })
            })
        })
        .min_by_key(|&(bn, bk)| {
            rows.iter()
                .map(|&m| {
                    gemms
                        .iter()
                        .map(|&(n, k)| {
                            let g = GemmShape { m, n, k };
                            cm.candidates(g, policy)
                                .iter()
                                .filter(|t| t.bn == bn && t.bk == bk)
                                .map(|&t| cm.gemm_cost(g, t))
                                .min()
                                .unwrap_or(u64::MAX) as u128
                        })
                        .sum::<u128>()
                })
                .sum::<u128>()
        })
        .map(|(bn, bk)| WeightLayout { bn, bk })
}

/// Choose ≤`max_batch_buckets` × ≤`max_seq_buckets` buckets per phase from a
/// workload, minimizing round-up padding waste (cost-model driven). Batch and
/// seq are bucketed independently (a grid), as production serving engines do.
pub fn choose_buckets(
    workload: &[(Request, u64)],
    max_batch_buckets: usize,
    max_seq_buckets: usize,
    cost: impl Fn(i64) -> u64,
) -> Vec<ShapeBucket> {
    let mut out = Vec::new();
    for phase in [Phase::Prefill, Phase::Decode] {
        let reqs: Vec<&(Request, u64)> =
            workload.iter().filter(|(r, _)| r.phase == phase).collect();
        if reqs.is_empty() {
            continue;
        }
        let batch_buckets = pick_axis(
            &hist(reqs.iter().map(|(r, f)| (r.batch, *f))),
            max_batch_buckets,
            &cost,
        );
        let seq_buckets = pick_axis(
            &hist(reqs.iter().map(|(r, f)| (r.seq, *f))),
            max_seq_buckets,
            &cost,
        );
        for &batch in &batch_buckets {
            for &seq in &seq_buckets {
                out.push(ShapeBucket { batch, seq, phase });
            }
        }
    }
    out
}

/// Collapse `(value, freq)` pairs into a sorted distinct histogram.
fn hist(it: impl Iterator<Item = (i64, u64)>) -> Vec<(i64, u64)> {
    let mut m = std::collections::BTreeMap::new();
    for (v, f) in it {
        *m.entry(v).or_insert(0u64) += f;
    }
    m.into_iter().collect()
}

/// 1-D k-staircase: partition sorted distinct `values` into ≤`k` contiguous
/// groups (each rounds up to its max); return the group-max values (the chosen
/// buckets) minimizing `Σ freq·(cost(max) − cost(value))`. O(n²k).
fn pick_axis(values: &[(i64, u64)], k: usize, cost: &impl Fn(i64) -> u64) -> Vec<i64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.max(1).min(n);
    if n <= k {
        return values.iter().map(|&(v, _)| v).collect();
    }
    // waste(p, i): group values[p..i) rounded up to values[i-1].
    let waste = |p: usize, i: usize| -> u128 {
        let top = cost(values[i - 1].0);
        values[p..i]
            .iter()
            .map(|&(v, f)| f as u128 * (top - cost(v)) as u128)
            .sum()
    };
    // dp[j][i] = min waste covering first i values with j groups; par for reconstruction.
    let inf = u128::MAX;
    let mut dp = vec![vec![inf; n + 1]; k + 1];
    let mut par = vec![vec![0usize; n + 1]; k + 1];
    dp[0][0] = 0;
    for j in 1..=k {
        for i in 1..=n {
            for p in (j - 1)..i {
                if dp[j - 1][p] == inf {
                    continue;
                }
                let c = dp[j - 1][p] + waste(p, i);
                if c < dp[j][i] {
                    dp[j][i] = c;
                    par[j][i] = p;
                }
            }
        }
    }
    // Best j ≤ k covering all n values; reconstruct group maxes.
    let best_j = (1..=k).min_by_key(|&j| dp[j][n]).unwrap();
    let mut buckets = Vec::new();
    let (mut j, mut i) = (best_j, n);
    while j > 0 {
        let p = par[j][i];
        buckets.push(values[i - 1].0);
        i = p;
        j -= 1;
    }
    buckets.reverse();
    buckets
}

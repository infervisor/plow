# CHUNK-2 — cost-driven chunk sizing + PerChunk M-row pipeline

Feeds CHUNK-1's double-buffered prefill kernel a **principled chunk count `k`**
and the **1:1 producer→consumer chunk graph** it runs on. Rust-only, no GPU.

## 1. Roles (egglog / Rust / Lean)

- **Rust (`costmodel`)** owns legality + cost: per-op GEMM cycles, SRAM pages,
  DMA bytes. It also owns the `k` cost model (`rewrite::explore`).
- **egglog** owns *selection by argmin*. `k` candidates are asserted as facts and
  the winner read back — same engine as the tile selection (`explore::select`).
- **Lean** unchanged here (no new certificate).

## 2. Pipeline

```
TileGraph + ConstraintSet   (rewrite::assemble — unchanged)
        │
        ▼   Granularity::PerChunk(k)
expand_prefill_chunks(soc, machine, g, cons, n_cu, k, policy)
        │
        ├─ group_by_row_axis        tiles → k row-chunks   (the MERGE)
        ├─ chunk-range overlap      1:1 producer→consumer edges
        ├─ build_counters(Fine)     per-consumer-chunk counters, threshold 1
        └─ SM-set placement         chunk c → CU set [c·n_cu/k, (c+1)·n_cu/k)
        ▼
PerChunkPlan { tasks, placement, counters, wait_of, succ_of, chunk_edges, k }
```

## 3. The tile→chunk MERGE (`expand::group_by_row_axis`)

The heart of the deliverable. Each op's tile grid is partitioned into `k`
contiguous groups **along its token/row axis**:

| op kind | grid coord | row axis | row extent |
|---------|-----------|----------|-----------|
| Gemm    | `[i, j]`  | axis 0 (`i`, the `M`-block) | `M` |
| Flash   | `[h, q]`  | axis 1 (`q`, the `seq_q`-block) | `seq_q` |
| Row     | `[r]`     | axis 0 (`r`) | `rows` |

`TileDomain::row_axis()` returns `(grid_axis, block)`. With `n =
⌈rows/block⌉` row-tiles, chunk `b` owns the even contiguous split
`[⌊b·n/k⌋, ⌊(b+1)·n/k⌋)` of the row-tile index, i.e. **every tile sharing a
row-block index lands in the same chunk**. For a GEMM this keeps *all* `N`-tiles
of one `M`-block together (the whole activation row-range), which is exactly what
the consumer op reads.

Each chunk also carries its **row-element interval** `[rt0·block, rt1·block)` on
the shared token axis — the key used to wire cross-op edges.

Example — GEMM `M=2048, bm=128` (16 M-tiles), `N=15360, bn=256` (60 N-tiles):

```
k=2 → 2 chunks × (8 M-tiles × 60 N-tiles) = 480 tiles each, ranges [0,1024) [1024,2048)
k=4 → 4 chunks × (4 × 60)  = 240 tiles each,  ranges [0,512) … [1536,2048)
k=8 → 8 chunks × (2 × 60)  = 120 tiles each
```

`k` above the row-tile count is capped (no empty chunks). A chunk task's
duration/bytes = per-tile cost × tiles-in-chunk.

## 4. 1:1 producer→consumer edges (the pipeline)

On a row-coupled boundary (a recorded `TileDep`), producer chunk `p` → consumer
chunk `c` iff their row-element intervals overlap:

```
edge(p, c)  ⇔  p.range ∩ c.range ≠ ∅
```

When both ops chunk the *same* token axis into `k` equal groups this is the
**diagonal**: chunk `c` ← chunk `c`, exactly `k` edges, each consumer chunk with
in-degree 1. `build_counters` in `Fine` mode then emits **one counter per
consumer chunk with threshold = 1** — a consumer chunk fires the instant its one
producer chunk finishes. **This is not a global barrier**; a global barrier
(threshold = k, all-wait-all) would serialize the SM sets and destroy the
overlap. Boundaries with no row coupling (e.g. a K-reduction) correctly fall back
to the coarse all-to-all edge — there the chunks are not independently
pipelinable.

## 5. Cross-SM placement + static L2 affinity

`ChunkPlacementPolicy`:

- **`StaticColocated`** (chunk thesis): chunk `c`'s *entire* producer→consumer
  chain pins to SM set `[c·n_cu/k, (c+1)·n_cu/k)`. The producer writes its chunk
  output into that partition's L2 slice; the consumer, on the *same* SMs, reads
  it **hot from L2** — no HBM round-trip across the op boundary. Chunk `c`'s
  consumer overlaps chunk `c+1`'s producer on a *disjoint* SM set → true
  concurrency, no contention.
- **`GlobalQueue`** (A/B baseline): every chunk may run on any SM and work-steal.
  A scattered consumer loses L2 residency and re-reads the producer output from
  HBM.

The `PerChunkPlan.placement` carries the per-chunk CU range so CHUNK-1's
megakernel can A/B the two policies. The L2-boundary-sharing argument is the
central hypothesis: **static + L2-locality beats global-queue for chunks**,
reversing the decode GQ default.

## 6. Cost model for `k` (`explore::chunk_prefill_cycles`)

The underlying `costmodel` GEMM cost is **aggregate SM-work**, so chunking cannot
shrink the compute floor `P + C` (the chip does the same total work, and two
compute-bound ops overlapped on-chip still contend for the same SMs). The lever
chunking actually pulls is **memory**, dominated by L2 reuse:

```
time(k) = (P + C)                        compute floor, invariant in k
        + hbm_roundtrip                  IF a chunk output spills L2
        + k · gate                       per-chunk counter/double-buffer prime
        + tail(k)                        wave-quant tail when rows don't tile to bm

hbm_roundtrip = 2 · out_bytes / hbm_bytes_per_cycle      (store + re-read)
spills L2      ⇔  out_bytes / k  >  l2_budget
tail(k)        = P · wasted_rows / M ,  wasted from ⌈(M/k)/bm⌉·bm·k − M
```

- **L2 reuse is the driver.** When a chunk's output slice fits `l2_budget` the
  consumer reads it hot → the whole HBM round-trip vanishes. Below that `k` it is
  paid; above it every extra chunk only adds `gate`. So the argmin is the
  **largest chunk that still fits L2**: `k* ≈ ⌈out_bytes / l2_budget⌉`. `k*`
  rises with context length — bigger `M` ⇒ bigger producer output ⇒ more chunks
  to stay resident. This *is* the "chunk small enough to keep its output in L2"
  sweet spot.
- **Wave-tail refines it.** A `k` that fits L2 but leaves `M/k` not a multiple of
  `bm` pays a wave-quant tail, so the model prefers the smallest L2-fitting `k`
  that *also* divides the M-tile count.
- `l2_budget` = ½·L2 (the other half holds the consumer's streamed weight tile +
  its own inputs). RTX 5090: L2 = 96 MiB → budget 48 MiB, HBM 1792 GB/s @
  2.407 GHz ≈ 744 B/cycle.

### egglog vs Rust — verdict: **Rust argmin; egglog as equivalence oracle**

For an *isolated* prefill pair the `k` choice is a plain 1-D scan — there is **no
joint constraint** (no chosen-`k` fact that changes a sibling op's cost in the
same e-graph saturation, unlike tile choice feeding an SRAM-colocation fact). So
the e-graph adds nothing here. We use `best_chunk_count` (Rust argmin) in the hot
path and keep `best_chunk_count_egglog` (routes the same `k` candidates through
`explore::select`) purely as a test oracle. `chunk_count_egglog_equals_rust`
asserts they agree at M ∈ {512, 2048, 8192} — the "egglog == Rust for isolated
ops" pattern, honestly.

## 7. Chosen `k` (RTX 5090, up_proj → down_proj FFN pair, bf16)

`out_bytes` = producer activation `[M, inter]`. Reproduce with the ignored test
`rewrite::explore::tests::print_chunk_numbers`.

| model | dims (hidden / inter) | M | producer out | k* | why |
|-------|----------------------|----|-------------|----|-----|
| 12B   | 3840 / 15360 | 2048  | 60 MiB  | **2**  | 60/48 → 2, divides 16 M-tiles |
| 12B   | 3840 / 15360 | 8192  | 240 MiB | **8**  | fits L2 at k≥5, k=8 first even divisor of 64 |
| 12B   | 3840 / 15360 | 16384 | 480 MiB | **16** | fits at k≥10, k=16 divides 128 M-tiles |
| 31B   | 5376 / 21504 | 2048  | 84 MiB  | **2**  | 84/48 → 2 |
| 31B   | 5376 / 21504 | 8192  | 336 MiB | **8**  | fits at k≥7, k=8 even |
| 31B   | 5376 / 21504 | 16384 | 672 MiB | **16** | fits at k≥14, k=16 even |

`k*` scales with context length exactly as the L2-residency thesis predicts.

**Honesty note.** In this *aggregate-work* cost model the compute floor `P+C`
dwarfs the L2 saving, so the modeled wall-time delta is <0.1% — the model's job
here is **`k`-selection (ranking)**, not absolute-time prediction. The measured
prefill speedup comes from CHUNK-1's kernel overlapping the memory-bound stores/
loads with compute and from L2 residency, which this static work model bounds but
does not fully price. The `k` it picks (largest chunk that stays L2-resident and
tiles evenly) is the right target regardless.

## 8. API handed to CHUNK-1

```rust
use schedule::{expand_prefill_chunks, ChunkPlacementPolicy, PerChunkPlan};

// prefill op chain already assembled → (g, cons); n_cu = SM count; k from
// rewrite::best_chunk_count(k_max, &ChunkCostIn{ .. }).
let plan: PerChunkPlan = expand_prefill_chunks(
    soc, machine, &g, &cons, n_cu, k, ChunkPlacementPolicy::StaticColocated);

plan.tasks        // PerChunk(k) TaskGraph (chunk tasks + chunk edges)
plan.placement    // Vec<ChunkPlacement{ task, node, chunk, cu_range }>  ← SM-set per chunk
plan.chunk_edges  // Vec<(TaskId, TaskId)>  compute→compute 1:1 chunk edges
plan.counters     // Vec<Counter>  fine, threshold-1 per consumer chunk (no global barrier)
plan.wait_of / plan.succ_of   // per-task counter wait / increment lists
plan.k
```

`ChunkPlacementPolicy::GlobalQueue` yields the work-steal baseline (`cu_range =
[0, n_cu)` for every chunk) for the A/B.

Chunk count from the cost model:

```rust
use rewrite::{best_chunk_count, ChunkCostIn};
let (k, _cycles) = best_chunk_count(k_max, &ChunkCostIn {
    producer_cycles, consumer_cycles, gate_cycles,
    m_rows, bm, out_bytes, l2_bytes, hbm_bytes_per_cycle,
});
```

## 9. Global-barrier audit

`build_counters(Fine)` never emits a global barrier on a coupled matched-count
boundary: each consumer chunk gets its own counter, threshold = its producer-chunk
in-degree (1 on the diagonal). The only coarse (all-wait-all) counters are on
genuinely all-to-all boundaries (no row coupling) — there the chunks share every
producer and cannot pipeline anyway, so coarse is correct, not a regression.

`list_schedule`'s placement heuristic (earliest-free SM) does **not** by itself
pin chunks to disjoint SM sets — the SM-set affinity lives in
`PerChunkPlan.placement` and is consumed directly by CHUNK-1's megakernel (which
assigns CU ranges), not re-derived by the generic list scheduler.

## Kimi-K3 on 8x MI355X: 2026-09-04 campaign

`codex/amd-agent-harness` is the single consolidated branch for the Kimi-K3 MI355X work; the earlier side branches (`codex/d1-moe-decode-rule`, `codex/harness-merge-ready`, `codex/seq-parallel-seams`, `codex/decode-l1-l2`) are merged into it. Most of it was fast-forwarded to `main` during the day at the owner's request; this PR carries the remaining commits and serves as the review record for the whole stack.

### Served result (C1, 8192→1024, TP8, pinned vLLM 0.28 image)

| metric | vLLM 0.28 | Plow before (09-04 morning) | Plow published (stack-2) | engine gate now (seams + router) |
|---|---:|---:|---:|---:|
| median TTFT | 566 ms | 1272 ms | 1113 ms | 962 ms |
| median TPOT | 20.9 ms | 28.5 ms | 25.25 ms | 25.1 ms |
| output tok/s | 46.7 | 33.6 | 38.0 | — |

Plow still trails vLLM; the gap closed from 2.24x to ~1.7x TTFT and from 1.37x to 1.20x TPOT.

### Promoted to default (each gated on 3 alternating TP8 folds; exact unless noted)

- HSA queue 4096 and exact per-segment AQL chain reservation.
- Standalone grouped-MoE decode route selected by a measured TuneDB rule; GLU GEMV K=7168 UN=7 rung; ASAP global-queue window order.
- Register-resident KDA carry (−112 ms TTFT) with interpreter fallback below 512 rows and per-slot operand rebase (probed exact at 300/8400/9000 tokens and concurrency 2).
- f32-mix AttnRes with vLLM's separate output-norm epsilon (−48 ms; semantic change, GSM8K 122 vs 124 / 200).
- Tagged one-shot decode XReduce (−2.12 ms/token); split-tile MLA merge-fold (−0.31 ms/token).
- Align-parallel MoE and the GemmWide c8 tile at 8192x1536x7168 (−22.9 ms).
- Sequence-parallel TP seams: post-attention row work on the reduce-scatter-owned band, results all-gathered (−109 ms, no weight replication).
- Wave-parallel exact router top-k select (−0.18 ms/token in-network; 15.85 → 12.07 µs isolated).

### Records and reproduction

- `perf-data/kimi-k3-plowrt-mi355x-baseline.md` / `-c1.json`: published two-fold campaign with full artifact identity.
- `perf-data/kimi-k3-mi355x-campaign-summary-20260904.md`: every experiment, verdict, and closed mechanism (59 verbose reports folded in).
- `docs/k3-mi355x-20260904/`: plan, decode-gap plan, scaling audit, seams feasibility, gate scripts, and the exact compilation recipe.

### Known limitations

- The gains are a concurrency-1 latency stack; the scaling audit lists which routes are batch-1 or shape-keyed and the gates needed for long context and throughput serving.
- `devgen tuned_tile_selection::gfx942_*` fail on `main` and here alike (stale gfx942 tuning cell; needs a MI300X requalification).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01FEN7AT33rmNdwAoePSheAX

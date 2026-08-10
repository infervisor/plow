# B2: head-batched sparse MLA prefill — built, correct, still net-negative at 16k

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **MODEL-PROPERTY + CDNA3 kernel** — the union-dilution ceiling is a property of GLM-5.2's DSA selections and is arch-independent; the kernel measurements are gfx942.

Base worktree-glm52-bringup @ 6ad92f8 (merged). The Phase-B follow-on named in
glm52-dsa-sparse-prefill.md: replace the 64-query UNION (which diluted to 80% of
the causal range) with 8-query PACKS whose 64 MFMA M-rows are (8 queries × 8
heads) — all 8 per-rank heads ride the M dimension, so a pack's union is walked
ONCE per head instead of per-head-per-tile.

## What changed (opt-in, control byte-identical to the shipped V2 blob — cmp verified)

- `d_flash_mla_prefill_v2<GATHER>` GATHER arm rewritten: work item = one 8-query
  pack; row = (query, head) via my_r0; Q-load / softmax / epilogue remapped;
  membership bit is pack-local (0..7).
- op 119 `d_index_union_pf` gains `i4 = tile_p` (queries/tile; 8 for B2, 0 = the
  legacy 64) and a SLICE-indexed umask scratch (bound by block count, not the
  8×-denser pack count — the per-qt umask would have cost ~600 MB).
- emit: `GLM_DSA_PF_PACK = 8`; iuni/iumask sized per pack; union packet on
  n_tok/8 blocks with i4=8; sparse arm gated on `nh_l == 8` (the row map's shape).

## Correctness (amd-bench --tp 8, 16k prompt, PLOW_DSA_PF object)

All 8 ranks agree; sparse output finite; first token 198 (dense) vs 167 (sparse,
expected — the model was trained with DSA, sparse ≠ dense); top-1 logit differs,
max|Δ| 1.42 on amax 5.75. Knobs unset re-emit byte-identical; a sparse blob on a
no-arm object runs dense (inherited safety).

## Measured — NET-NEGATIVE at 16k, and why (traced per-MoE-layer, 16k prompt)

| phase | dense V2 | sparse B2 | Δ |
|---|---|---|---|
| attn-flash | 16557 µs | 14003 µs | **−15%** |
| indexer (score+select+union, "other") | 869 | **4575** | +3706 µs |
| moe / gemm / xreduce | ~14000 | ~14000 | ~0 |
| **layer span** | 30565 | 31964 | **+4.6%** |
| prefill wall | 4070 ms | 4322 ms | +6.2% |

**Two findings:**
1. Head-batching WORKS but under-delivers vs the probe: flash busy 3.78 → 3.24
   CU-s (−14%), not the ~3.8× the union-of-8 mask probe predicted (398 vs 1512
   rows/query). The per-pack `ucount` walked is not shrinking to the probe's
   union size — the compaction is filling closer to the causal range than the
   distinct-position count. This is the open kernel bug: the win is real but a
   fraction of the ceiling until the walk length equals the true union.
2. **Blocker 2 now dominates.** The indexer score materializes a T×ctx f32
   matrix (op 117) — 3.7 ms/layer at 16k, MORE than the flash saving. It was
   left unaddressed here (the directive's second blocker): it needs the vLLM
   chunk-the-query-axis + fp8-key treatment before ANY flash saving nets out.

## Verdict

The B2 decomposition is the right shape and the selector/emit/routing are in
place and safe, but it is NOT a win yet: (a) the union walk must actually shrink
to the probe-measured size, and (b) the indexer score must be chunked+fp8'd.
Both are bounded, named, and independent. Default OFF on both axes.

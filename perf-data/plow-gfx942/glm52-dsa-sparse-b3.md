# B3: DSA sparse prefill — the union-coverage ceiling, measured directly

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **MODEL-PROPERTY** — 89% top-2048 overlap between adjacent queries is a property of the MODEL's selections, not of any GPU. It closes this route on every architecture.

Base worktree-glm52-bringup @ 2a36112. Directive: close B2's two "bounded"
blockers and turn the head-batched sparse MLA prefill into a 16k/32k win.
Result: **the win is not reachable on this path, and the reason is a hard
property of DSA selections — not a fixable kernel bug.** Both B2 blockers were
probed directly (PLOW_DUMP_ACT on `act.iidx_pf` + `act.iuni`, 16k prompt,
DSA-armed objects); the analysis script is `perf-data/.../probes/dsa_overlap.py`.

## Blocker 1 as B2 described it is FALSE

B2 claimed the union walk was longer than the true union ("compaction fills
closer to the causal range than the distinct-position count", probe 398 vs
1512). Measured on the dumped tables:

    union-of-8 TRUE: mean 3092   kernel cnt: mean 3092   causal: mean 10244
    kernel==true fraction: 1.00   max|diff| 0

**The kernel's union is EXACTLY the true union, every pack.** op 119 is correct
and optimal. B2's 398-vs-1512 figure was mismeasured (wrong position / stale
mask read).

## The real ceiling: DSA selections are 89% correlated across adjacent queries

    adjacent-query overlap: mean 1824 / 2048   (random baseline 458)
    p10/p50/p90: 1779 / 1832 / 1858
    sample query: 2048 sel, 21 in first-64 (sink), 497 in last-512 (recency)

Adjacent queries share **89%** of their top-2048 (vs 22% if selections were
random) — DSA overwhelmingly picks the shared recency tail + attention sinks.
So a union-of-8 cannot get small: it is **3092 / 10244 = 0.30 of causal**, and
that ratio is intrinsic, not tunable. Wider packs barely help (the union
saturates — 89% overlap means query 9 adds almost nothing new); narrower packs
lose the head-batching amortization.

## And 0.30 walk does NOT convert to 0.30 flash busy (the true, re-characterized Blocker 1)

Traced per-MoE-layer @16k (both 8192 chunks), sparse-arm vs dense-V2, same objects:

| phase | dense V2 busy | sparse B3 busy | Δ |
|---|---|---|---|
| attn-flash | 10.33 CU-s | 9.22 CU-s | **−11%** |
| indexer (score+select+union, "other") | ~0.0 | **1.19 CU-s** | +1.19 (serial chain) |

A 0.30 walk ratio should give ≈ −68% flash busy; the head-batched kernel
delivers only −11%, because it recomputes every union KV row against all 64
(8 queries × 8 heads) pack rows even though the average union row is live for
only ~5.3 of the 8 queries, and the 1024-pack grid underfills 304 CUs where
dense runs 32k work items. Converting the walk saving to busy needs a
per-query-membership-skipping MFMA — a from-scratch kernel, not a knob.

## Blocker 2 (indexer materialization) is real but secondary

op 117 still writes a T×ctx f32 score matrix; the score+select+union chain costs
~3.7 ms/layer serial. Chunking it (the vLLM design) would cut this — but it only
matters once the flash actually saves its fraction, which (above) it does not.

## Served verdict — sparse LOSES at 16k

    prefill: 16384 tokens  dense-V2 6652 ms  vs  sparse-B3 7300 ms  (+9.7%)

vLLM is 1631 ms @16k. Sparse does not close the gap; it widens it. Correctness
held (first token 198 both; 8 ranks agree; knobs-unset emit byte-identical to
the shipped V2 blob — cmp verified), so nothing here is unsafe — it is simply
not a win.

## Honest conclusion for the goal

Beating vLLM at 16k/32k via this sparse decomposition is **blocked by a data
property** (89% adjacent-query overlap → 0.30 union floor) compounded by a
kernel that can't even bank the 0.30. The three routes that remain are all
large: (a) a membership-skipping sparse flash whose busy tracks the union (weeks,
new kernel); (b) accept dense and push the asm-class MoE rewrite + the dense
flash's remaining ~1 ms; (c) a fundamentally cheaper indexer. None is a
blocker-close. Caveat worth stating: the 89% overlap was measured on the
campaign's random-token bench prompt; natural text may select less
recency-dominated context and overlap somewhat less — but not enough to move a
0.30 floor to a win, and vLLM is measured on the same prompts.

Sparse arm stays **default-OFF on both axes** (PLOW_GLM_DSA_PF / PLOW_DSA_PF);
this branch adds only the probe tooling and this record — no kernel or emit
change beyond what B2 already merged.

# DSA sparse prefill for GLM-5.2 on gfx942: built, correct, and MEASURED NET-NEGATIVE as designed

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **MODEL-PROPERTY + CDNA3 kernel** — the union-per-tile decomposition fails on selection statistics (arch-independent); register and spill figures are gfx942.

2026-08-07, branch `glm52-dsa-sparse-pf` (base worktree-glm52-bringup @ 759a8eb). Phase B of
glm52-experiments.md (consolidated: the OPEN section). **Opt-in and default OFF on both axes** (emit `PLOW_GLM_DSA_PF=1`,
object `PLOW_DSA_PF=1`); knobs unset re-emit **byte-identical** to the shipped V2 blob
(verified by cmp against `/workspace/assets/gfx942/glm52-tp8-v2/model.pkt`).

## What landed

The per-query selector whose absence blocked the gathered prefill (`emit_glm_mla_prefill`'s
own note: "nothing produces `idx[b][t][top_k]`") now exists, plus the tile packaging the V2
flash needs:

| op | kernel | what |
|---|---|---|
| 117 `IndexScorePf` | `d_index_score_pf` | T-row lightning-indexer score: one (query, 32-position) MFMA subtile per WAVE, A = the query's 32 head rows and B = 32 keys read straight from global (both L2-hot across the query axis), w-weighted ReLU epilogue identical to the decode `d_index_score_mfma`. No LDS, no barriers. |
| 118 `IndexSelectPf` | `d_index_select_pf` | per-query-row EXACT top-k: the op-59 radix (same monotone `dsa_pack_key_a`, score desc / lowest-index tie-break) run wholly inside ONE workgroup per row — LDS histograms, `__syncthreads`, no grid barrier. |
| 119 `IndexUnionPf` | `d_index_union_pf` | per-64-query-tile UNION: scatter the tile's rows into a u64 membership word per kv position, compact ascending into `[u32 count[n_qt]][cap i32 pos][cap u32 maskLo][cap u32 maskHi]`. |
| 51 (t7) | `d_flash_mla_prefill_v2<..., GATHER=true>` | the V2 full-column-wave body with the KV walk over the tile's union entries; per-query exactness from the staged mask words (one extra 256 B LDS strip). |

Emit: `emit_glm_dsa_prefill_select` (T-row wq_b/wk/weights_proj projections + the two indexer
RoPEs, writing the shared `kv.{l}.kidx` cache at the chunk base like `krot`), gated by
`glm_dsa_pf_bucket` (t >= 2048 for the V2 BQ=64 fill floor, and t > index_topk — below it every
top-k is the identity and the indexer is pure overhead). The indexer WEIGHTS are now declared
under either gate (`idx_on = dsa || dsa_pf`); previously only the 64k-crossover decode gate
declared them, which is why the first sparse emit silently produced zero sparse packets.

## Correctness

Same first token as dense (198 @8k, 19 @32k), all 8 ranks agree, TPOT unmoved. The **degrade
paths are proven, not asserted**: a sparse blob on an object built without the arm reads no t7
and runs DENSE — same token 198 (measured), and a blob emitted without the knob has no sparse
packets at all.

## Measured — and this is the finding

amd-bench, TP8, traced (`trace_block`-style per-op reduction, last chunk):

| ctx | dense V2 | sparse | Δ |
|---|---|---|---|
| 8k | 1744 ms | 1896 ms | **+8.7%** |
| 32k | 11542 ms | 12121 ms | **+5.0%** |

Per-op at 32k (the widest causal range, where sparse should win biggest):

| op | dense | sparse |
|---|---|---|
| FlashMlaPrefill | 2779 ms span / 669.8 CU-s busy | **2217 / 535.1** (0.80×) |
| IndexScorePf | — | 489 ms |
| IndexSelectPf | — | 129 ms |
| IndexUnionPf | — | ~8 ms |

**Blocker 1 — union dilution (the structural one).** Flash busy sparse/dense = **0.80 at 32k
and 0.95 at 8k**: that ratio IS the fraction of the causal range a 64-query tile's union
covers. Each query selects top_k=2048 of a ~16k causal row at 32k (1/8), but 64 adjacent
queries between them touch 80% of it, so the gathered walk saves 20% instead of 87%. The
union-per-tile decomposition cannot deliver the topk/S ratio at ANY context — it gets worse as
the tile widens, and our tile is 64 rows because the V2 kernel is 4 waves × 16 query rows.
**The fix is the FlashMLA/AITER decomposition, not a tuning knob**: batch the HEADS into the
MFMA M dimension (MQA over the absorbed latent) so a work item is ONE query token and its
gather is exactly its own top_k — no union exists to dilute. That is a different kernel shape
from V2 (which uses query rows as M and heads as the work axis) and is the honest Phase-B2.

**Blocker 2 — indexer cost.** 626 ms of the 32k prefill, dominated by the score (489 ms): it
materialises a T×ctx f32 logits matrix (2.4 GB at the 8192 bucket × 73728 ctx) to HBM for the
selector to re-read. vLLM chunks the query axis and keeps fp8 keys (`sparse_attn_indexer.py`).
Chunking + an fp8 key path should cut this several-fold; it is worth doing only alongside
Blocker 1, since 626 ms would still exceed the 562 ms the current gather saves.

## Cost accounting for the register budget

The gathered body is a second instantiation in a megakernel whose allocation is the worst case
over every inlined arm: the flash object's spill went **204 → 287** with it. Hence the separate
`PLOW_DSA_PF=1` build axis — the shipped object is byte-for-byte the V2 one.

## Verdict

Working, correct, safely gated, and **not shippable as a perf win in this decomposition**. The
selector chain (117/118) is reusable as-is by the MQA rebuild — it is the gather packaging (119
+ the union walk) that the rebuild replaces with per-query gathers.

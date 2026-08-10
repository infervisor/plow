# GLM-5.2 / gfx942 — the experiment ledger

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **MIXED -- the ledger; each row carries its own scope** — rows sourced from `docs/amd/aiter-tensile-notes.md` and `op_gemm.h` ceilings are gfx950/MI350X measurements and are marked where they appear.

Every experiment the campaign ran, its verdict, and why. One entry each; the detailed reports that
survive are named in the last column. Method lessons are in `LESSONS.md`; the shipped state is in
`README.md`; the merge-review summary is in `glm52-experiments.md`.

**Read the CLOSED table before proposing work.** Most of it was closed by measurement, and several
entries were closed *twice* because the first closure was later found to have measured the wrong
thing.

---

## Shipped result

**SUPERSEDED FOR THE vLLM COLUMN — see "Round 2 re-baseline" immediately below.** The vLLM TTFT
figures in this table could not be reproduced in a single session with both engines measured by
one client and prefix caching off on both. The plow figures stand.

| | plow (default) | plow (+16384 rung) | vLLM 0.26 + AITER *(stored, unreproducible)* |
|---|--:|--:|--:|
| TTFT 1k | 370 ms | 370 | 69 |
| TTFT 4k | **752 ms** | 752 | 566 |
| TTFT 8k | ~1662 ms | **1418** | 672 |
| TTFT 16k | ~3573 ms | **3296** | 1631 |
| TPOT | 26.7–29.8 ms | — | ~18 |

Campaign arc at 8k: **8819 → 1418 ms (6.22×)**. Decode: **48.6 → 26.72 ms (−45%)**.
The 16384 rung is an emit opt-in, not a default — see `docs/arch/13-prefill-chunking.md` §5.

---

## Round 2 re-baseline (2026-08-09) — BOTH ENGINES, ONE SESSION, ONE CLIENT

`README.md` calls this "the single most valuable missing measurement". It is now done, and it
**moves the headline**. Harness: `scripts/twoengine/` — the same client drives both engines, so
the ratio cannot partly measure the harnesses. Prefix caching **off on both** (plow has none, so
leaving it on measures a *feature* gap and reports it as a *kernel* gap).

| | plow | vLLM 0.26 + AITER | ratio | stored vLLM |
|---|--:|--:|--:|--:|
| TTFT 1k | 319.2 ms | 211.3 | 1.51× | 69 |
| TTFT 4k | 712.0 ms | 535.3 | 1.33× | 566 |
| TTFT 8k | 1372.2 ms | 1179.5 | **1.16×** | 672 |
| TTFT 16k | 3244.9 ms | 2504.1 | 1.30× | 1631 |
| TPOT | 26.503 ms | 17.919 | 1.48× | 17.95–18.68 |
| **GSM8K 8-shot, n=100** | **0.9700** | 0.1900 † | — | never measured |

**TPOT reproduced the stored number exactly** (17.919 vs 17.95–18.68) — the decode baseline was
sound. **TTFT did not**: same-session vLLM is 1.75× slower at 8k and 3× slower at 1k than
recorded. So **plow's prefill deficit is 1.16× at 8k, not the 2.11× this file used to claim**,
and the honest remaining gap is concentrated in **decode**, not split across both. Most likely
cause of the stored figures: prefix caching left on (the ladder repeats each prompt 3×), and/or
the dense-MHA path below 2048 tokens that this vLLM build can no longer serve. The README already
warned the stored CSV had failed to reproduce once, a 33% move.

Instrument quality: round-to-round spreads **0.04–1.0%**, against the 17.9% DVFS noise the
campaign fought previously. That is the difference between being able to land a 2% win and not.

† **Do not quote the vLLM accuracy as a quality comparison.** vLLM 0.26 on this ROCm stack cannot
serve GLM-5.2 below 2048 tokens at all — the dense-MHA path its own scheduler selects is
unimplemented on `ROCM_AITER_MLA_SPARSE`, and the documented workaround runs the sparse path
outside its design range. It is a **capability** finding, scoped to this build.
Full citations: `vllm026-rocm-short-prompt-gap.md`. Consequence: **no short-prompt vLLM number
from this box is publishable**; all speed benching must stay above 2048 tokens.

### Throughput — measured, and it does not scale

Streaming, 4096-token distinct prompts, no prefix caching, same client:

| conc | out tok/s | TTFT p50 | TPOT p50 |
|--:|--:|--:|--:|
| 1 | 26.94 | 1,052 ms | 29.1 ms |
| 4 | 22.76 | 17,541 ms | 38.2 ms |
| 8 | 24.63 | 36,611 ms | 31.6 ms |
| 16 | 25.38 | 75,900 ms | 30.7 ms |

Aggregate output is **pinned at ~25 tok/s** while TTFT grows **linearly** with concurrency. That
is pure serialization, not batching, against vLLM's recorded 54 → 814 tok/s ladder. Root cause is
stated in plow's own source (`crates/plowrt/src/serve/mux.rs`): the muxer is a real
continuous-batching engine (slot table, admission, EWMA λ, bucket re-pick, streaming) but **"true
batched exec that fires all live slots in one bucket walk"** is *"not here (yet)"* — it admits N
slots and advances them one at a time. Independently, GLM compiles **decode M=1 only**, so the
ladder has no rung above 1 to select. **There is no tuning path to a throughput win.**

---

## LANDED

| lever | effect | note |
|---|---|---|
| Wide prefill packets | **TTFT −66%** | b=1 norms were 73% of the layer span. The single largest win of the campaign. |
| MoE-PF LDS out-of-bounds fix | correctness | An 81,920 B double-buffer against a 64,512 B union. **Every gfx942 MLA prefill before it ran corrupted MoE math** and passed coherence gates because the distortion was shared by both arms of every A/B. |
| V2 MLA flash + causal KV-split `ns2` + SV bank swizzle | flash −6.9%, −15.5%; ns2 −7.7% | ns2's marginal cost is 0.7% of TTFT, not the 120 ms it was once booked at. |
| `PLOW_L2HIER` (per-XCD placement + hierarchical gating) | **TPOT −12%** | The claim path's one win per architecture. |
| `PLOW_MOE_DEC_LG` (lane-group decode DOWN) | **TPOT −7.6%** | GLM routes DOWN at K=256 where the old body left 48/64 lanes dead. Recorded as a null for weeks — see `LESSONS.md` §1. |
| `PLOW_MOE_PF_EPI` (DOWN epilogue hoist) | −18.8% on that kernel | The epilogue five k-loop attacks never examined. |
| `PLOW_MOE_PF_DET` (order-independent MoE combine) | **TTFT −2.07% @8k** | Integer-valued f64 accumulation ⇒ arrival order cannot matter. Run-to-run byte-identical where the atomic twin scored 3/6. Opt-in: not bit-identical to the shipped slot-ordered sum, and provably cannot be. |
| `PLOW_MLA_FOLD_TB` (token-blocked `MlaMergeFold`) | **TTFT −5.49% @8k** | Bit-identical by construction. Op 57 was a batched GEMM run one M-row at a time, streaming a 256 KiB `W_uv` panel per output row. |
| `PLOW_XR_AGG` (device-local gate_ag) | TTFT −1.73% @4k | Remote atomics 2432 → 8 per rank per collective. TPOT unmoved, as predicted — the op is absent from decode. |
| `PLOW_GLM_FUSE_QNORM` | TPOT −1.27% | About a third of the census projection, which had counted a gate round trip no fold reaches. |
| `PLOW_RAGGED_CHUNK` (**default ON**) | **−24% @4k**, −38% @1025 | Output-visible; gated on answer quality. `docs/arch/13`. |
| fp8 `−0` handling + maskless 6-VALU dequant | logits byte-identical | Dtype-driven loader scrub + an encoder that never emits `−0`. |
| GLU-into-quant AMD arm | correctness | The arm was **never written** on AMD while NVIDIA's twin existed. |
| Per-arch geometry table (`hwspec::IsaLevel::geometry`) | correctness | Ends the duplicated-arena class. `docs/arch/14`. |
| Decode batch-size ladder | 1.84× throughput (Gemma) | Narrow ladder beats wide: `B_max=4` wins on throughput *and* latency, because all rungs share one object compiled at one `PLOW_GEMV_MM`. |

---

## What is ON BY DEFAULT

The pattern throughout: the **header default is the SAFE value (0)** and `scripts/build_gfx942.sh`
sets policy — so including a header never changes behaviour, but building via the script does.

| axis | default | effect | evidence |
|---|---|---|---|
| `PLOW_MLA_PF2_DBUF` | **on** (header `1`) | register-prefetch K pipeline in v2 flash | byte-identical, flash −6.9% |
| `PLOW_MLA_PF_SMX` | **on** for CDNA3 (opt out `=0`) | split softmax across column-group waves | bit-identical, flash −15.5% |
| `PLOW_MOE_DEC_LG` | **on** (opt out `=0`) | lane-group decode DOWN; GLM routes DOWN at K=256 where the old body left 48/64 lanes dead | character-identical incl. a 14.7k-token prompt, **−7.6% TPOT** |
| `PLOW_L2HIER` → `PLOW_L2_PLACE_DISPATCH` + `PLOW_GATE_HIER` | **on** for gfx942 decode | per-XCD placement + hierarchical gating | **−12% TPOT** |
| `PLOW_MLA_PF_V2_ARM` | **on** (arm present) | the v2 flash kernel is COMPILED IN; ROUTING stays env-gated by `PLOW_MLA_PF_V2` | unset ⇒ byte-identical emit |
| `PLOW_FUSE_QUANT` | **on** for AMD | the GLU-into-quant fold; its AMD kernel arm was missing until this campaign | character-identical after the fix |
| `PLOW_RAGGED_CHUNK` | **on** | the only OUTPUT-VISIBLE default here — see `docs/arch/13-prefill-chunking.md` | quality-gated, not character-gated |

Everything else this campaign added is **opt-in and default-off**, including `PLOW_MLA_PF_SV`,
`PLOW_MOE_PF_EPI`, `PLOW_MOE_PF_DET`, `PLOW_MLA_FOLD_TB`, `PLOW_XR_AGG`, `PLOW_GLM_FUSE_QNORM`,
`PLOW_GLM_OFOLD`, `PLOW_GLM_FP8_KV` and the `*_ABL` ablation instruments. Note that two arms in the
shipped object recipe (`PLOW_MLA_PF_SV=1`, `PLOW_MOE_PF_EPI=1`) are default-OFF in the tree and
passed explicitly — flipping them on for gfx942 is a policy call, not a correctness one.

**Every knob goes through `EmitConfig` / `RuntimeConfig`**, each keeping its `env =` attribute so
existing scripts are unaffected, and guarded by `every_field_has_a_reader` + `no_raw_env_reads` in
both crates. Two documented exceptions stay raw: `PLOW_BLOCK` (owned by `plowc --block`) and
`PLOW_GLM_GF` (a deliberate dual read whose config field is consumed via `.or()`).

---

## ROUND 2 RESULTS (2026-08-09) — one session, one client, instrument resolves ~1%

| lever | verdict | number |
|---|---|---|
| **`PLOW_GEMV_LG` → default-ON for gfx942** | **ADOPTED** (`ff7b486`) | **TPOT 26.503 → 26.077 ms, −1.61%**, 5× the 0.04–0.31% spread. TTFT flat (correct negative control for a decode-only flag). |
| `PLOW_GLM_XR_RES` | **REJECTED** | −2.0% at 1k but **+0.9% at 4k/8k/16k** against 0.1–0.3% spreads. Recorded as −0.7/−1.7%; does not reproduce. Byte-identical, so this is a perf call not a safety one. |
| per-XCD **prefill** placement | **NO-GO** | −9.1% at 1k, **+27.9/+57.3/+97.6% at 4k/8k/16k**. See CLOSED table. |
| `PLOW_DECODE_BATCH` on GLM | **silently ignored → now warns** (`d6af634`) | `PLOW_DECODE_BATCH=4` emitted `decode M=1` with no diagnostic. |

**Both adopted/rejected flags were in the same state: landed, measured, character/byte-gated —
and never passed by any recipe.** `PLOW_GEMV_LG` had been sitting unused since `52d6dd5`. Two
others are still only alive because `build_gfx942.sh` passes them explicitly (`PLOW_MLA_PF_SV`,
`PLOW_MOE_PF_EPI`). **Grepping the tree for opt-in flags carrying a measured win that no recipe
passes is worth more than most new experiments** — it has now paid twice.

**Run single-variable arms.** LG (objects) and XR (blob) were measured separately; each moved
only its own half. Combined, the net would have read as a modest win with a prefill regression
hidden inside it.

---

## CLOSED BY MEASUREMENT — do not reopen without new evidence

| route | why it is closed |
|---|---|
| **DSA sparse attention** | Closed at audit grade, twice. The union route dies on 89% top-2048 overlap between adjacent queries. The per-query route dies on arithmetic: **selection is intrinsically 47% of the arithmetic of the attention it avoids** (8,192 vs 17,408 FLOP/pair) and fp8 is bf16-rate on CDNA3, so a *perfect* indexer still lands at +14.2% at 16k. Rebuilding the indexer 2.9–3.6× did not change the verdict. **The last surviving lever is now SCOPED and also closed** (`glm52-dsa-tp-shard.md`, 2026-08-09): TP-sharding the replicated indexer's query axis is worth **+409 ms at 16k** and takes sparse to 16.2–18.1% of TTFT — it does clear plow's own 15% bar — but it closes only **37.6% of the vLLM deficit** (3245 → 2638 against 1631) and every figure assumes the *ideal* sparse flash, which does not exist. Two surprises worth carrying: the **double buffer is not the win, u16 index positions are** (38 → 19 ms; a whole overlap pipeline adds only 19 ms more, because the fabric probe shows the all-gather rate is dead linear in workgroup count so an overlapped gather steals CUs), and **gfx950 does not change the decision** (CDNA4's 2× fp8 MFMA halves a term sharding already cut to 59 ms of a 666 ms gross). Do not build; measure a 32k TTFT first if ever revived. |
| **TP4 × PP2** | Memory-feasible (85.8/87.8 GiB/rank vs TP8's 86.8 — the "tp4 infeasible" note was about *pure* TP4). Rejected on cost: the seam saving is a seam *cost* (+5.8% TTFT, 3 XGMI links instead of 7), the stage handoff it pays for is free (<0.15%), and α = 1.17/1.26 caps it at 1.71×. |
| **Packet-boundary protocol** | Both commissioned arms null on two independent instruments; one arm a reproducible +6.8% regression with a mechanism. The boundary *front* is off the exposed path — deleting the header prefetch entirely and fixing its redundancy both move nothing. |
| **Claim-path / scheduling rebuild** | Pays exactly once per architecture (the L2-placement + `GATE_HIER` pair) and every subsequent attack measured **zero**: bands, `XR_CUS`, `XR_MLP`, protocol folds, `GLM_GROUP`, and NVIDIA's analogues, across three architectures. |
| **Prefill attention body** | Online softmax already optimal in both kernels; wave-specialisation NO-GO on CDNA3; QK1 dedup −1.4% at 32k only. The attention *body* is not the prefill bottleneck. |
| **Collectives** | Correctly tuned: full mesh, pull beats push, fabric limited by cross-thread request concurrency. Overlap is structurally capped — on a saturated grid it is a static partition of the same CUs. |
| **MoE k-loop** | Six arms, all null. The inner loop is ~0% of the gap; the cost was decomposition and epilogue. |
| **Prefill L2 placement — now measured DIRECTLY on GLM, and it is not null, it is a large REGRESSION** | 2026-08-09. `PLOW_GLM_PLACE_PF=1` blob + `PLOW_L2HIER_PF=1` objects. Emits fine (8/8 programs placed, 0.1% skew) and the served coherence gate **passes** — so the documented "whole prefill on the flash object, zero logits" collapse is a dense-GQA hazard GLM does not hit, and **the seg × domain composition is unnecessary**. But TTFT: **−9.1% @1k, +27.9% @4k, +57.3% @8k, +97.6% @16k** against 0.0–0.8% spreads. Monotone in context. Mechanism: placement swaps the global queue's DYNAMIC balancing for a STATIC per-XCD partition — at 1k the working set is L2-local and locality wins; at 4k+ the per-domain set is ~201 MB/layer against a 4 MiB L2 so locality buys nothing and each XCD can no longer steal work. Same family as "banding closed negative". Keep prefill UNPLACED; decode stays placed (that is where `PLOW_L2HIER`'s −12% TPOT lives). Narrow exception: the −9.1% at 1k is real, so a short-prompt-only deployment could opt in below ~2k — bisect the crossover first. |
| **`ds_read_u16_d16_hi` in the flash PV staging** | 2026-08-09, refuted **in hardware**. On GFX940+ a D16 instruction writes the FULL 32-bit VGPR: `probes/d16sem.hip` seeds a destination and issues one `ds_read_u16_d16_hi`, which returns `0xab040000` where preservation would give `0xab045a5a`. There is no merge to exploit, which is why the backend never selects the pattern — the absence was *correct*, not a missed optimisation, and inline asm cannot rescue it. `probes/d16probe.hip` gates 8 arms of the real PV shape bit-for-bit: all three d16 arms mismatch in all 8192 words **including the one with a full `lgkmcnt(0)` drain between the low and high loads** (which rules out an in-flight hazard), both source-idiom rewrites compile byte-identical to shipped, and the two correct asm arms buy nothing (one trades 29 partial waits for 33 full drains, the other hits vgpr 260 in a kernel already at occ 1). Also corrects the lever's own numbers: the perms are **23% of the PV body, not 8%**, and the shipped double buffer already schedules partial waits. |
| **EP, band overlap, split-K, `GLM_GROUP`, `ns≠2`** | All measured null. |
| **GEMM tile / double-buffer re-cut on CDNA3** | Closed twice. `BM=192, BK=32` is not merely LDS-limited, it is *structurally impossible*: `static_assert(APT % 8 == 0)` with `APT = BM*BK/THREADS = 12`. At BK=32 the assert demands `BM % 128 == 0`, so 192 and 64 both fail (`op_gemm.h:1033`). The XOR swizzle that would "save the padding" **already exists** (`GM_XORSWZ`, all 8 sites, `STRIDE = BK`); `GM_STRIDE` survives only in arena sizing. And the full ladder re-cut *was already built* — MD 128×128, WD 128×256, C5 256×128, default 128×256, all BK=32 DBUF=2 — fits at 61,448 B, passes the golden GEMM oracle, and measured **575.2 → 642.6 ms (+11.7%)** on a 4096-token Gemma-4-12B prefill. "The ping-pong does not pay for the BK it costs." |
| **Mega-mode decode** | Placed STATIC 14.61 vs placed GQ 10.63 — the deficit is the fixed assignment, not missing placement. |
| **Gemma 4k chaining** | Census found **0 chainable 1:1 edges** on the real decode stream. Ceiling ~0.01 ms against a 0.6 ms gate. |
| **Finer prefill rungs (32/64)** | Reduce padding, which is already small, while leaving per-pass fixed cost untouched — and give the DP more ways to add chunks that each cost a full pass. |
| **`LAUNCH_ROWS` reprice** | Real (the constant is ~4× low) but dominated by ragged-M at every length where either changes the plan, with a strictly smaller blast radius. Under ragged-M the constant is never read. It is a partial ragged, not an alternative. |
| **`W_ofold`** | Aimed at a fold that is now 2.6× cheaper, and pays a doubled `o_proj` plus the loss of ns2 to do it. |

---

## OPEN — the next levers, in the order I would take them

1. **The exposed decode boundary is concentrated, not uniform.** `GemvQkv` alone owns 38%, and four
   edges own 77%. Unserialising the `GemvQkv → b=1 norms → flash` chain is worth up to 12.4 µs of
   32.5 on its own. **This is a schedule/emit problem, not a protocol one** — which is why the
   protocol round came back null.
2. **Batched GLM decode — RE-SCOPED AGAIN 2026-08-09, and the previous rescope named the wrong
   blocker.** Full working: `glm52-batched-decode-scope.md`. Measured symptom unchanged: aggregate
   output is **pinned at 22.8–27.0 tok/s from concurrency 1 to 32** while TTFT grows linearly
   (1.05 s → 148.2 s) — queueing, not batching.
   **This entry's own title was false.** The muxer does *not* advance slots one at a time:
   `mux.rs:1682` hands every live slot to `step_batch`, `engine.rs:586` forwards to `dispatch_all`,
   and `amd.rs:5748` fires **one** dispatch for all rows. What caps concurrency at 1 is
   `check_slot` rejecting `slot >= self.batch`, and `self.batch` is the **blob's compiled
   `PLOW_DECODE_BATCH`** — GLM emits at 1 (`mla.rs:5456`). So this is an emitter problem end to
   end; no runtime work is needed. (A serve banner confirms it directly: `batch=1`.)
   The MLA chain is **already** batch-aware — `FlashMlaDecode`, `MlaMergeFold`, `OUvFold` all take
   `i[0] = n_batch`, `GemvQkv` takes `i[0] = M`, `HeadNormRope` takes `i[6] = n_batch_kv`,
   `SAMPLE_BATCH` is wired at cap 16, and activations are already sized for the widest prefill
   bucket so a B ≤ 16 decode is free.
   **The real blocker, which nobody had named: the MoE DECODE ops carry no token dimension**
   (`MoeRouterTopk`, `MoeGroupGluFp8Blk`, `MoeGroupDownFp8Blk`, `MoeCombine` — every prefill
   counterpart carries `T`). The Gemma ladder that proves batched decode at 1.84× does not exercise
   a block-fp8 grouped MoE, so this is *unproven*, not merely unwritten — which is why "days, not
   weeks" should not have been trusted.
   **Route:** at `rows > 1` emit the MoE seam with the **prefill** op family at `T = rows` rather
   than extending four decode kernels. Precedent in the same file: `emit_glm_dense_block_prefill`
   (`mla.rs:4179`) already reuses `MoeGroupGluPf`/`DownPf` for the *dense* FFN at `n_exp=1,
   top_k=1`. It is also the right shape on the merits — at B=16 the grouped form reads each touched
   expert's weights once instead of once per routing token.
   **Gate:** `rows = 1` must re-emit **byte-identical** to the shipped blob. Wants its own branch.
3. **`gemv_rows<MM>` weight-stream amortisation.** 1.39× across 16 rows against vLLM's 10.08×. This
   is the *entire* throughput gap — not the rung count.
4. **GEMM rate**, 0.51× hipBLASLt. Real and identical at TP1 and TP8, but only ~6.6% of TTFT for a
   multi-week occ-1 deep-pipeline rewrite. Poor ms-per-week; listed for completeness.
5. **Residual correctness debt**: the RmsNorm `t3/t4` arm has no marker of its own, so an object
   predating it would ignore both and leave `xq` stale — the same silent class as the GLU arm, one
   op over.

---

## Tooling worth knowing about

| tool | what it is for |
|---|---|
| `perf-data/probes/facts_gate.py` | The standing answer-QUALITY gate (`run` / `verdict` / `selftest`). Off-rung cells, pairwise McNemar, exits 2 rather than PASS when powerless. |
| `scripts/gemma_xgate.sh` | The cross-model gate that RE-EMITS. Replaces the stored-asset procedure that could not catch an emitter regression. |
| `scripts/asm_audit.py` + `asm_expect_gfx{942,950}.json` | Instruction-selection contracts. The two are inverses; the audit reads each object's arch from its own ELF header. |
| `crates/hwspec/tests/device_header_agreement.rs` | Host/device geometry drift guard. |
| `scripts/glm52_layer_census.py` | The trace reducer behind the cost decomposition (group by `inst`, not `pc`). |
| `perf-data/plow-gfx942/probes/dsa_indexer_net.py` | Re-derives the sparse-attention net from a measured indexer cost. **Use this before reopening sparsity** — it is the arithmetic that closes the route. |
| `perf-data/plow-gfx942/probes/dsa_tp_shard_net.py` | The same arithmetic with the TP-shard, u16 and overlap terms, plus the distance to vLLM. The companion to the row above. |
| `perf-data/plow-gfx942/probes/d16sem.hip` | Twenty lines that settle whether a D16 instruction preserves the other half of its VGPR on this target. The template for *test the hardware before writing asm to force an instruction*. |
| `perf-data/probes/chunk_flip_ladder.sh` | Produces the shipped TTFT ladder; the runner behind the restated headline. |

---

## Surviving detailed reports

Consolidated here; these keep their own files because they are cited by the architecture docs or
carry an argument that does not compress:

| file | what it holds |
|---|---|
| `glm52-current-cost-decomposition.md` | The authoritative per-block attribution. Supersedes every earlier trace-based per-layer number. |
| `gemv-mlp-and-tensile.md` | The Gemma-4 decode campaign end to end, and the aiter/Tensile ceilings. |
| `glm52-chunk-policy.md`, `glm52-facts-gate.md` | The chunk-policy acceptance class and the quality gate. Cited by `docs/arch/13`. |
| `glm52-decode-batch-ladder.md`, `glm52-decode-ladder-vs-vllm026.md` | The ladder's design and its external reference. |
| `glm52-tp4-pp2-evaluation.md` | The PP verdict, and the first throughput baseline. |
| `glm52-packet-protocol-xcd.md`, `glm52-packet-boundary-roundtrips.md` | The protocol analysis and the nulls that close it. |
| `glm52-moe-deterministic-writer.md`, `glm52-moe-fusion.md` | The MoE combine arms and why bit-identity is unattainable there. |
| `glm52-dsa-indexer-rebuild.md` | The indexer rebuild and the arithmetic that closes sparsity. |
| `glm52-dsa-tp-shard.md` | The last sparse lever, priced and declined. |
| `glm52-batched-decode-scope.md` | Where the throughput blocker actually is, with three corrections. |
| `arch-hygiene.md` | The audit behind `docs/arch/14`. |

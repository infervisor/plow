# GLM-5.2-FP8 TP8 on gfx942 — the ragged tail chunk, priced and removed

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **PLOW-ARCHITECTURAL** — a chunk-planning defect, not a kernel one -- one token past a bucket paid a whole forward pass. The fix is arch-independent; 205/231 ms is gfx942.

**One sentence: a prompt one token past a compiled prefill bucket paid a whole
extra 78-layer forward pass — 205 ms at 1k, 231 ms at 4k — and `PLOW_RAGGED_CHUNK`
removes it wherever the ladder can absorb the remainder into an existing launch.**

| | |
|---|---|
| branch / sha | `ragged-chunk` @ `5209ac0` (base `worktree-glm52-bringup` @ `968fc3a`) |
| blob | `/workspace/assets/gfx942/glm52-tp8-final2/model.pkt` (UNCHANGED — no re-emit) |
| objects | `/root/.claude/jobs/b09a4bcc/tmp/hsaco_glm18` (UNCHANGED — no kernel change) |
| env | `PLOW_MLA_PF_V2=1`; test arm adds `PLOW_RAGGED_CHUNK=1` |
| loaded objects, BOTH arms | `interp_prefill_mla_moe_gq.elf`, `interp_flash_gq.elf`, `interp_decode_gq.elf` (variant `Bf16` — the block-scaled-fp8 `Variant::detect` outcome the campaign records; identical across arms by construction, the change is host-side only) |
| instrument | served TTFT (first SSE content delta) at EXACT prompt token counts, plus `amd-bench --prompt <ids> --dump-logits` |
| session | 2026-08-08 13:27–14:17 UTC, GPU lock held continuously and exclusively |

---

## 1. Reproduction, before anything was changed

Prompts built to land on an EXACT token count (calibrated against the server's own
`usage.prompt_tokens`: the GLM chat template is 6 tokens and the filler ` the` is
1 token, so a target is hit exactly and verified per cell before timing).

```
served TTFT, shipped config, 3 reps, interleaved

  1024 tokens -> 336.2 ms   (spread 0.18%)      4096 tokens -> 720.2 ms   (0.03%)
  1025 tokens -> 540.7 ms   (spread 0.08%)      4097 tokens -> 951.2 ms   (0.26%)
      ONE MORE TOKEN = +204.5 ms                    ONE MORE TOKEN = +231.0 ms
```

**Confirmed, at 800x and 900x the control's own round-to-round spread.** The
device-wall figures in `glm52-current-cost-decomposition.md` §1.4 (343.5/546.5 and
731.2/964.5, i.e. +203.0 and +233.3) reproduce on the serve path to within 1%.

It is NOT flat in T: +204.5 at 1k against +231.0 at 4k. The tail chunk's flash
attends over `c0 + 128` KV rows, so the extra launch gets more expensive the
further into the prompt it sits. The decomposition's "flat" label is right to
within 15% and the direction it is wrong in is the one that matters at long
context.

## 2. Where the cost comes from — `plan_chunks` and a baked row count

`plan_chunks` (`crates/plowrt/src/exec/amd.rs:1347`) covers a prompt from the
compiled ladder `[128, 512, 1024, 2048, 4096, 8192]` with a DP minimising
`sum(bucket_i + LAUNCH_ROWS)`. The ladder rungs are separately EMITTED PROGRAMS —
2021 instructions each, one per bucket, with the row count `T` baked into every
instruction's immediates.

That baking is the whole defect. `4097` cannot be expressed, so the cover is
`[4096, 128]` and the 128-row chunk runs a complete 78-layer pass.

**And the DP is not making a mistake.** Its two options at 4097 are 4224 padded
rows in two launches or 8192 padded rows in one; under a regime where a padded row
costs full compute, two launches really is cheaper. Repricing `LAUNCH_ROWS` cannot
fix this (§6) — the LADDER CANNOT EXPRESS "4097 rows in one pass", and that is a
different problem from a mis-tuned cost model.

### 2.1 The fixed-vs-marginal anatomy of a pass, measured rather than fitted

Every number here is a difference of two served TTFTs in the same session.

| quantity | how it was obtained | value |
|---|---|---|
| a 128-row tail launch at `c0=4096` | ctrl(4097) − ragged(4097), same 4097 real rows | **239.2 ms** |
| a 128-row tail launch at `c0=8192` | ctrl(8193) − ctrl(8192) | **270.7 ms** |
| a **ONE-ROW** tail launch at `c0=8192` | ragged(8193) − ctrl(8192) | **230.9 ms** |
| marginal value of the tail's other 127 rows | 270.7 − 230.9 | 39.8 ms (0.31 ms/row at ctx 8k) |
| marginal cost per row, single launch, 1k–4k | slope of ragged(1025..4224) | **0.130 ms/row** |
| fixed cost per launch, T→0 intercept, low ctx | intercept of the same fit | **189 ms** |

**A ONE-ROW 78-layer GLM prefill pass costs 231 ms.** That is the number the whole
report turns on: 85% of a 128-row tail chunk is invariant in its row count, and no
amount of shrinking the tail can recover it. The only lever is not to launch it.

For scale, against the same session's single-launch costs: the fixed term is 69%
of a 1024-row chunk (336 ms), 32% of a 4096-row chunk (724 ms) and 17% of an
8192-row chunk (1399 ms).

Why a 1-row pass costs 231 ms, mechanically — the trace-level apportionment is
`glm52-current-cost-decomposition.md` §1.5 and is not re-derived here, but the
three terms it names are all row-count-invariant by construction:

1. **Narrow-M GEMM at one row-tile.** `d_gemm_t` runs `tm = ceil(M/BM)` row-tiles;
   at M=1 and M=128 alike that is ONE, so `tn` workgroups of 304 CUs stream the
   entire weight column block either way. §1.5 fits 1588 us/layer of intercept to
   this family — 119 ms over 75 MoE layers, the largest single fixed category.
2. **157 segment launches per chunk** (`plowrt disasm` + `derive_segments` on this
   blob), each a host-barriered drain across all 8 ranks, independent of T.
3. **Per-layer norms, router and collectives**, whose intercepts §1.5 fits at
   456 / 109 / 275 us/layer.

## 3. The fix — `PLOW_RAGGED_CHUNK`, opt-in, no re-emit and no kernel change

Two changes, both host-side, both in `crates/plowrt`:

1. **`rebase_chunk_rows`** rewrites every prefill row-count immediate from the
   bucket width to the chunk's REAL length, from a static opcode->field table
   derived from this blob's own op census. `in.kvlen` moves with the flash's
   `n_tok` (the flash derives `qpos = kv_len - n_tok + t`, so the two are one
   decision and are read from one place, `ragged_bucket`).
2. **`plan_chunks`** then covers the prompt in the FEWEST launches —
   `ceil(n / 8192)` — because the padding no longer costs anything.

### 3.1 Why this needed no kernel work, and the one question that decided it

The kernels already take the row count as a runtime operand and already bound
themselves by it: `d_gemm_t` computes `tm = ceil(M/BM)` and guards every A fetch
with `r < M`; `d_flash_mla_prefill` sizes its work list
`n_work = n_batch * n_tok * n_grp * nsplit`; the grouped MoE pair (ops 85/86)
carries no `T` at all and works off the sorted-row `meta` table op 84 builds.

**The feasibility question was the GATE PROTOCOL, not the arithmetic**: a workgroup
whose tile range is now empty must still satisfy its successor counters or the pass
deadlocks. It does, by construction — `runtime/amd/interp.hip` calls `plow_exec`
(the op body) and then publishes successor counters UNCONDITIONALLY in the stream
loop, outside the opcode switch. A body that iterates zero tiles still signals.
Nothing about the counter DAG changes.

Also worth stating because it is a second-order win: the shortened chunk touches
FEWER EXPERTS. At 128 rows x top-8 = 1024 slots essentially all 256 routed experts
get a tile and their weights are streamed; at 1 row only 8 do. That is part of the
39.8 ms the 8193 cell recovers.

### 3.2 The guard, and the one configuration that is refused

A field that holds the row count in one packet holds something else in another —
the lm_head `Gemv`'s `M = 1`, a `PLOW_GLM_XR_BAND` row-band `Gemm`'s `M = T/kb`.
So the shrink applies only where the field EXACTLY equals the bucket width (or is
an exact multiple, for element-count operands). Anything else is left alone, which
is always safe: computing padded rows is what the engine did before.

Row-banded prefill is the one case where "left alone" is not enough (the band GEMM
would be skipped by the guard while its `XReduce` partner was rescaled), so
`refuse_unraggable` REFUSES the flag on such a packet by its signature — a
non-lm_head matmul with `a_row0 != 0`, or a `MoeCombinePf` with `t_row0 != 0`. The
shipped blob is unbanded (the axis is emit-time, default OFF, measured
net-negative), so this is a guard rather than a limitation.

## 4. Measured: A/B at round AND ragged lengths

Served TTFT, **3 interleaved passes**, arms alternated (ctrl, rag, ctrl, rag, ...),
one server per arm, same binary, same asset, same prompts, GPU lock held for the
whole battery.

| prompt tok | plan (ctrl) | plan (ragged) | ctrl mean | spread | ragged mean | spread | delta | |
|--:|---|---|--:|--:|--:|--:|--:|--:|
| 1024 | `[1024]` | `[1024]` | 339.4 | 0.91% | 337.8 | 0.15% | −1.5 | −0.4% |
| **1025** | `[1024,128]` | `[2048]` | 546.9 | 1.17% | **321.9** | 0.28% | **−225.0** | **−41.1%** |
| **1152** | `[1024,128]` | `[2048]` | 546.9 | 1.21% | **340.1** | 0.15% | **−206.8** | **−37.8%** |
| 2048 | `[2048]` | `[2048]` | 449.2 | 0.80% | 447.0 | 0.20% | −2.3 | −0.5% |
| 4096 | `[4096]` | `[4096]` | 726.0 | 0.41% | 724.1 | 0.44% | −2.0 | −0.3% |
| **4097** | `[4096,128]` | `[8192]` | 960.3 | 0.60% | **721.1** | 0.22% | **−239.2** | **−24.9%** |
| **4224** | `[4096,128]` | `[8192]` | 959.8 | 0.77% | **736.2** | 0.18% | **−223.6** | **−23.3%** |
| 8192 | `[8192]` | `[8192]` | 1399.2 | 0.28% | 1398.9 | 0.02% | −0.3 | −0.0% |
| 8193 | `[8192,128]` | `[8192,128]` | 1669.6 | 0.59% | 1629.8 | 0.07% | −39.8 | −2.4% |
| **12345** | `[8192,4096,128]` | `[8192,8192]` | 2679.9 | 0.69% | **2369.5** | 0.16% | **−310.4** | **−11.6%** |

Read it as three regimes:

* **Exactly-covered lengths are a NULL** (−0.0% to −0.5%, inside the control's own
  0.3–1.2% spread). The plan is identical and the shrink is a no-op, which is the
  intended byte-identical behaviour and is asserted by a unit test.
* **Lengths where the remainder fits in an existing launch win 205–310 ms.** At
  1025 the ragged arm (321.9 ms) is FASTER THAN THE 1024 CELL (337.8): the
  2048-bucket program carrying 1025 real rows beats the 1024-bucket program
  carrying 1024, because the rungs carry different GEMM tiles.
* **Past 8192 the extra launch is STRUCTURAL** — no bucket is wider than
  `MAX_CHUNK = 8192`, so `ceil(8193/8192) = 2` launches is already the minimum and
  only the tail's rows shrink (128 → 1). 8193 recovers 39.8 ms of its 270.7 ms
  tail; the other 231 ms is the T-invariant pass and is NOT addressable by this
  axis. **This corrects the decomposition's gap table, which credits "ragged tail
  chunk 203 ms" as removable at 8k and 16k as well as 4k.**

## 5. Correctness

The risk is an off-by-one at the END of the prompt, which is exactly where the
answer is generated. Three instruments, weakest to strongest.

### 5.1 Token identity of the prefill's own output — PASS

`amd-bench --tp 8 --prompt <exact ids> --steps 8`, ctrl vs ragged, same binary:

| prompt ids | ctrl sampled + 8 greedy | ragged sampled + 8 greedy | |
|--:|---|---|---|
| 1025 | 4401, `[315,3460,4128,4119,702,5497,1246,3162]` | 4401, `[315,3460,4128,4119,702,5497,1246,3162]` | **identical** |
| 4097 | 4128, `[4119,702,5497,1246,3162,374,5326,13]` | 4128, `[4119,702,5497,1246,3162,374,5326,13]` | **identical** |
| 8193 | 1246, `[3162,374,5326,13,48391,1431,7512,7385]` | 1246, `[3162,374,5326,13,48391,1431,7512,7385]` | **identical** |

All 8 ranks agree in every run. **A dropped or duplicated prompt token would
change the very first sampled id; none does, at three ragged lengths.**

### 5.2 Prefill logits — top-1 identical, NOT bit-identical

`--dump-logits`, rank 0's full vocab row after prefill:

| prompt ids | top-1 ctrl / ragged | top-1 logit | top-2 gap | max abs delta | delta / logit range |
|--:|---|--:|--:|--:|--:|
| 1025 | 4401 / 4401 | 28.75 | 12.13 | 1.56 | 4.6% |
| 4097 | 4128 / 4128 | 29.88 | 14.31 | 0.82 | 2.2% |
| 8193 | 1246 / 1246 | 30.38 | 14.63 | 1.48 | 4.3% |

The winner is never close to contested (a 12–15 logit margin against a ≤1.6
perturbation). The perturbation is expected and its source is named: a ragged
chunk runs a DIFFERENT bucket program with different GEMM tiles, and the grouped
MoE sorts a different number of rows into different tiles, so the f32 accumulation
ORDER changes. This is the same acceptance class the campaign recorded for
`PLOW_MLA_PF_V2` (top-1 match, max|Δ| 0.445).

### 5.3 Character identity of served answers — PASS on short answers, FAILS on long free-form

Fixed questions, greedy, at round AND deliberately ragged lengths, 35 answers per
arm (5 questions x {no padding, 1024, 1025, 4096, 4097, 8193, 12345}):

* **22 of 35 character-identical**, including EVERY short answer (Paris at all
  seven lengths, `391 = 17 x 23` at five of seven) and EVERY answer at a round
  length (1024, 4096).
* **13 of 35 differ**, and every one of them is (a) a long free-form generation of
  96–200 tokens and (b) at a length where the ragged plan differs — never at 1024
  or 4096. The texts are fluent, on-topic paraphrases, not truncations.

That is a failure against the stated bar, so it was chased rather than excused.

**The decisive control: `PLOW_LAUNCH_ROWS=4000` on the SHIPPED code path.** At that
price the padding-vs-launch DP prefers one padded `[8192]` chunk to `[4096, 128]`
for a 4097-token prompt. No row shrink, no new code path, only a different rung —
so if the shrink were introducing the divergence, this arm would agree with the
control. Three arms, one session, same binary:

| cell | ctrl == lr4000 | ctrl == ragged | **lr4000 == ragged** |
|---|:--:|:--:|:--:|
| essay@4097 | no (diverges at char 139) | no (diverges at char 139) | **YES — byte-identical** |
| gold@4097 | no (char 122) | no (char 122) | **YES** |
| sky@4097 | no (char 185) | no (char 185) | **YES** |
| essay@8193 | YES (same plan) | no (char 86) | no |
| gold@8193 | YES | no (char 150) | no |
| sky@8193 | YES | no (char 282) | no |

**At 4097 the ragged arm's text is BYTE-IDENTICAL to the shipped engine's own text
when the shipped engine is merely priced into choosing the same rung, and both
diverge from the two-launch cover at exactly the same character.** The divergence
belongs to WHICH BUCKET PROGRAM RUNS THE PROMPT — a property the shipped planner
already has — and not to the row shrink. At 8193 the plan is the same in ctrl and
lr4000 (hence they agree) and only the ragged arm's TAIL differs (128 rows vs 1),
which reorders the tail chunk's MoE tiles: the same class, one level down.

Every arm is internally deterministic (6/6 within-arm repeats byte-identical), so
none of this is run-to-run noise.

**The strongest single result, and the one that retires the off-by-one fear:**
`amd-bench --steps 64` on the same 4097-id prompt gives **the same 64 greedy token
ids in all three arms**, character for character, all 8 ranks agreeing:

```
4119, 702, 5497, 1246, 3162, 374, 5326, 13, 48391, 1431, 7512, 7385, 323, 1077,
264, 1614, 8192, 264, 1156, 9955, 11, 1221, 3395, 432, 13, 576, 11043, 4401, ...
```

### 5.4 Verdict

**No off-by-one. No dropped or duplicated prompt token.** Three independent
instruments say the last real row of the prompt is the row the lm_head reads and
the row the KV ends at: identical first sampled token at 1025/4097/8193, identical
64-token greedy continuation at 4097, identical top-1 logit with a 12–15 margin.

**Character identity across a PLAN CHANGE is not achievable on this engine and
never was.** The shipped planner already produces different long-form text for one
prompt depending on which rung it picks (`lr4000` proves it on unmodified code),
because the rungs carry different GEMM tiles and group the MoE rows differently, so
the f32 accumulation order differs. `PLOW_RAGGED_CHUNK` lands INSIDE that
pre-existing equivalence class rather than widening it — which is the strongest
statement the evidence supports, and it is stronger than "4/5 answers matched".

Anyone shipping this should know that is the acceptance class, and it is the same
one `PLOW_MLA_PF_V2` and `PLOW_GLM_FUSE_SEAM` were accepted under.

## 6. `LAUNCH_ROWS` repricing, measured — a partial win at 1k and a NULL at 4k+

`LAUNCH_ROWS = 416` charges 416 rows per launch. The measured launch is ~231 ms
(§2.1), i.e. ~1780 rows at the 0.130 ms/row marginal rate — the constant
understates a GLM launch by **~4.3x** (the decomposition's estimate was 3.4x).
So the constant IS wrong. It is also nearly powerless here.

Plans, computed from the shipped ladder (`cargo test --test glm_pf_shape`):

| tokens | LR=416 (shipped) | LR=1400 | LR=4000 | ragged |
|--:|---|---|---|---|
| 1025 | `[1024,128]` | `[2048]` | `[2048]` | `[2048]` |
| 1152 | `[1024,128]` | `[2048]` | `[2048]` | `[2048]` |
| 4097 | `[4096,128]` | `[4096,128]` | `[8192]` | `[8192]` |
| 8193 | `[8192,128]` | `[8192,128]` | `[8192,128]` | `[8192,128]` |
| 12345 | `[8192,4096,128]` | `[8192,4096,128]` | `[8192,8192]` | `[8192,8192]` |

Served TTFT, same session:

| tokens | LR=416 | **LR=1400** | ragged |
|--:|--:|--:|--:|
| 1024 | 339.4 | 337.2 | 337.8 |
| 1025 | 546.9 | **447.7** | **321.9** |
| 1152 | 546.9 | **447.7** | **340.1** |
| 4097 | 960.3 | 953.1 | **721.1** |
| 8193 | 1669.6 | 1662.0 | **1629.8** |
| 12345 | 2679.9 | 2697.9 | **2369.5** |

**Repricing captures 44% of the win at 1025/1152 and 0% of it at 4097, 8193 and
12345.** At 1025 it trades the second launch for 1023 padded rows — a real gain,
and less than half of what the shrink gets, because the shrink pays for neither.
At 4097 it changes nothing until LR=4000, and at LR=4000 the padded `[8192]` cover
is a large REGRESSION, measured directly on the shipped code path:

```
amd-bench, 4097 real tokens, device prefill wall
  [4096, 128]  two launches, 4224 padded rows   973.9 ms   (shipped)
  [8192]       one launch,   8192 padded rows  1401.4 ms   (LR=4000, +44%)
  [8192]       one launch,   4097 REAL rows     742.4 ms   (ragged, -24%)
```

**That is the whole argument in three lines.** Padding up is worse than a second
launch; a second launch is worse than not padding. The cost model is mis-tuned and
fixing it is worth doing for its own sake, but it is not the lever — the ladder's
inability to express 4097 rows is.

## 7. Alternatives, and why they were rejected

**(b) A rung ladder with no remainder — rejected on arithmetic.** The ladder's
quantum is 128, so any `T` not a multiple of 128 still leaves a remainder, and
covering an arbitrary `T` in ONE launch needs a rung at every `T` — up to
`max_ctx/128 = 576` separately emitted 2021-instruction programs. Finer rungs also
do not touch the term that costs: a 40-row remainder in a 64-row chunk instead of a
128-row one saves 64 rows x 0.130 = 8 ms and still pays the 231 ms pass. And
because the DP undervalues launches 4.3x, extra small rungs give it more ways to
add chunks that look cheap and are not.

**(c) Repricing `LAUNCH_ROWS` — measured, kept, but not the fix.** §6. It is a real
correction (the constant is 4.3x low) and it is worth landing on its own merits at
the 1k band, but it captures none of the 4k+ waste, and pushed far enough it
actively regresses (+44% at 4097).

**Raising `MAX_CHUNK` above 8192 — not attempted, and it is the only route to the
8k/16k residue.** It would let 8193 and 16386 run in one and two launches. It is
also the expensive option: `act.part` is `[T*top_k][hidden]` f32, already 1.61 GB
per rank at T=8192, so a 16384 rung doubles it to 3.2 GB per rank on top of a
model that already sits near the card. `MAX_CHUNK_MAX = 8192` is asserted in
`devgen` and tied to the KV-ring invariant. Out of scope here; recorded as the
named next lever for the 8k/16k residue.

## 8. What the campaign's published TTFT numbers become

**Every published TTFT at >= 4k contains a tail chunk, and this is the correction.**
`scripts/bench_speed.sh` builds its prompt as `ntok//9` repeats of a 9-word
sentence; tokenised with the GLM tokenizer that is 1017 / 4095 / 8190 / 16380
content tokens, and the chat template adds 6:

| bench_speed `IN_LENS` | real prompt tokens | shipped plan | ragged plan |
|--:|--:|---|---|
| 1024 | 1023 | `[1024]` | `[1024]` — unchanged |
| 4096 | **4101** | `[4096, 128]` | `[8192]` at 4101 real rows |
| 8192 | **8196** | `[8192, 128]` | `[8192, 128]` at 4 real tail rows |
| 16384 | **16386** | `[8192, 8192, 128]` | `[8192, 8192, 128]` at 2 real tail rows |

Applying the measured deltas for those exact covers (the 4097/4224 cells bracket
4101; the 8193 cell is the 8196 case; 16386 is the same 2-launch-plus-tail shape):

| context | published (round-3 gate) | with `PLOW_RAGGED_CHUNK` | change |
|--:|--:|--:|---|
| 1k | 343.3 | **343.3** | none — the 1k prompt never had a tail |
| 4k | 973.4 | **~735** | **−238 ms, −24%** |
| 8k | 1677.0 | **~1637** | −40 ms, −2.4% |
| 16k | 3627.4 | **~3587** | ~−40 ms, −1.1% |

vs vLLM 0.26 AITER on this box (69 / 566 / 672 / 1631): the ratio at 4k moves from
**1.72x to 1.30x**; 8k from 2.50x to 2.44x; 16k from 2.22x to 2.20x.

**The correction the decomposition needs.** Its gap-attribution table (§3.2) lists
`ragged tail chunk` as a flat **203 ms at 4k, 8k AND 16k**. The 203 ms is real at
all three — but it is only ADDRESSABLE at 4k. At 8k and 16k the prompt is past
`MAX_CHUNK = 8192`, so `ceil(n/8192)` launches is already the minimum and the tail
launch is structural; only its rows shrink, worth ~40 ms. **The addressable pool at
8k is therefore 631 − 163 = 468 ms of the 1005 ms gap, not 631; and at 16k
1325 − 163 = 1162 ms of 1996, not 1325.** Item 2 of that report's ranked list
(`ragged tail chunk`, 203/203/203) should read **203 / 40 / 40**, with the residue
reassigned to a `MAX_CHUNK` raise that has not been costed.

## 9. Reproducing this

```
git checkout ragged-chunk          # 5209ac0
cargo build --release -p plowrt --features hsa
cargo test  --release -p plowrt --features hsa --lib     # 198 pass, 3 new
PLOW_MLA_PF_V2=1 PLOW_L2_PLACE_DISPATCH=1 \
  PLOW_PF_PKT=/workspace/assets/gfx942/glm52-tp8-final2/model.pkt \
  cargo test --release -p plowrt --features hsa --test glm_pf_shape -- --nocapture
                                                          # the cover sweep of §6

# served A/B: one server per arm, arms alternated, EXACT token counts
PLOW_MLA_PF_V2=1 [PLOW_RAGGED_CHUNK=1] \
  target/release/plowrt serve --assets /workspace/assets/gfx942/glm52-tp8-final2 --port 8196

# token-identity gate
PLOW_MLA_PF_V2=1 [PLOW_RAGGED_CHUNK=1] target/release/plowrt amd-bench --tp 8 --steps 64 \
  --blob .../model.pkt --hsaco .../hsaco --checkpoint .../checkpoint \
  --prompt "<4097 comma-separated ids>" --dump-logits <dir>
```

Two traps this run hit, recorded so the next one does not:

1. `glm_pf_shape` needs **both** `PLOW_L2_PLACE_DISPATCH=1` (the blob is L2-placed)
   and `PLOW_MLA_PF_V2=1` (the `ns=2` causal split is refused otherwise). Without
   them it fails at `DevBlob::parse` and at `derive_segments` respectively, in ways
   that look like packet corruption and are not.
2. `cargo fmt -p plowrt` reformats FOUR unrelated files and rewrites every clap
   `#[arg(...)]` in `config.rs` onto multiple lines. The tree is not fmt-clean;
   format your own hunks by hand or the diff triples.

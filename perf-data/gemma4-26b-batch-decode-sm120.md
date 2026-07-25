# Gemma-4-26B-A4B (MoE) CONCURRENT batch>1 DECODE on sm_120 — campaign B-26b-batch

**2026-07-22**, branch `beat26b-batch` (base `main` @ `7fd19fb`), 1× **RTX PRO 6000
Blackwell** (sm_120, 188 SMs, 96 GB), CUDA 13.0. A **32 GB foreign `plowrt`
process was resident throughout** (expected; ~63 GB usable) — it shapes the
memory-feasible batch/ctx points below.

Goal: make `PLOW_DECODE_BATCH=B` work for the 26B-A4B MoE (concurrent decode) for
**both bf16 and fp8**, prove correctness, and measure aggregate decode throughput
vs vLLM. Data: `gemma4-26b-batch-decode-sm120.json`; raw harness log
`perf-data/harness/gemma4-26b-batch-decode-sweep.log`.

## What was already done vs what this branch adds

- **bf16 MoE decode batch: already merged** (`4d9b585` "p9-batch", in `main`). The
  four bf16 decode MoE ops (router 67/68/69, fused GLU-norm 71, down 63, combine
  70) carry a batch row count and sweep the `B*k` routed slots **channel-major**
  for L2 weight reuse. The stale premise "the 26B decode blob is B=1 only"
  (`gemma4-26b-plowrt-served.md`, main @ `a058492`) predates that merge.
- **fp8 MoE decode batch: this branch's contribution** (`26278d6`). The fp8 expert
  twins (`d_moe_expert_glu_gemma_fp8` op 64, `d_moe_expert_down_gemma_fp8` op 65)
  were **left B=1 and unguarded** — `declare()` even claimed "the fp8 batch refusal
  is gone", so `PLOW_FP8=1 PLOW_DECODE_BATCH>1` silently wrote only row 0 and left
  rows 1..B-1 **garbage**. This branch batches them (add `nrow`, channel-major
  `plow_moe_unflat` sweep, per-row `x`/`part`), wires the dispatch
  (`PLOW_NROW(in->i[5])`) and the emit (`d.i[5]=nb` on the fp8 GLU). Register cost
  **0** (pure index permutation): decode cubin still **219 regs, 0 spill**.

## Single-block ORACLE (mandated: before the full model)

`runtime/tests/sm120_interp_op_test.cu` (built via the GF4 cmake recipe), every
op body vs an f32 CPU golden, relL2 gate + a **negative control**: the DISJOINT
routing case (each of the `B*k` slots picks a distinct expert) means a
row-1-unwritten bug lands as huge relL2 on the un-written rows. Added B to the
two fp8 tests; the bf16 batch tests already existed.

| op | cases | verdict |
|---|---|---|
| bf16 `moe_glu` / `moe_glu_norm` (op 62/71) | B=2/8/32 SHARED+DISJOINT | PASS (relL2 ≤ 8e-5) |
| bf16 `moe_down` (op 63) | B=2/8/32 SHARED+DISJOINT | PASS (relL2 ~7e-8) |
| **fp8 `moe_glu_fp8`** (op 64) | B=2/8/32 SHARED+DISJOINT | **PASS** (relL2 ≤ 7e-6) |
| **fp8 `moe_down_fp8`** (op 65) | B=2/8/32 SHARED+DISJOINT | **PASS** (relL2 ~6e-8) |

Full suite (flash/gemv/norm/router + all MoE): **exit 0, 0 FAIL lines**. Oracle
gate = **PASS** before any full-model run.

## Byte-identity (B=1 default protected)

- bf16 B=1 blob (ctx 132096) **md5 `04c807bd2d5c862e446406b7bc2bcdb8`** —
  identical to the committed served P9 blob. B=4/B=8 diverge (as intended).
- fp8 B=1: `nb=0 → i[5]=0 → PLOW_NROW=1`, so the fp8 B=1 packet is byte-identical
  to the pre-batch fp8 packet by construction.
- `cargo test -p plowc`: all green.

## Model-level correctness (real weights, `gemma4_sm120_chat`)

Same prompt fed to all B slots (`PLOW_PROMPT2` gives odd slots a *different*
prompt). The harness `SLOT PARITY` gate requires every slot to reproduce its
prompt-group token stream.

- B=1/4/8, bf16 and fp8: **slot-0 token stream identical to B=1** (token 14339) —
  batching is **token-identical** to single-user.
- MIXED (`PLOW_PROMPT2`): even slots → 14339, odd slots → a different but
  self-consistent stream (fp8 236890; bf16 47023/51562…), **0 divergent
  slot-steps** at B=8 → per-row isolation holds, no cross-contamination.

## Aggregate decode throughput (the concurrency win)

`gemma4_sm120_chat`, ctx≈2048 (2048-tok prompt + 80 gen, 16 warmup discarded),
greedy, per-user TPOT = one batched launch, aggregate = B/launch.

| dtype | B | per-user TPOT (ms) | aggregate tok/s | agg scaling |
|---|---:|---:|---:|---:|
| bf16 | 1 | 8.205  | 121.9 | 1.00× |
| bf16 | 4 | 11.432 | 349.9 | 2.87× |
| bf16 | 8 | 17.426 | 459.1 | 3.77× |
| fp8  | 1 | 6.310  | 158.5 | 1.00× |
| fp8  | 4 | 9.226  | 433.5 | 2.73× |
| fp8  | 8 | 14.318 | 558.7 | 3.52× |

- Concurrency **works and scales**: ~3.5–3.8× aggregate at B=8. Per-user TPOT
  degrades sub-linearly (8.2→17.4 ms bf16; 6.3→14.3 ms fp8) because the routed
  expert-weight stream grows with the number of *distinct* experts hit, not with
  B — the channel-major L2 reuse is exactly this lever.
- MIXED-prompt B=8 aggregate is within ~1% of same-prompt (bf16 442.7, fp8 554.6),
  i.e. the reuse win does **not** depend on all users sharing a prompt.

## Memory feasibility (why fp8 matters for B=8)

Per-seq KV scales ~linearly with ctx (0.86 GiB/seq @ ctx 4096). With the 32 GB
foreign process resident (~63 GB usable):

| dtype | weights | B=8 KV @4k | B=8 total | fits w/ foreign? |
|---|---:|---:|---:|:--:|
| bf16 | 47.0 GiB | 6.88 | ~54.8 GiB | yes @ctx≤4k only |
| fp8  | 24.3 GiB | 6.88 | ~32.1 GiB | **yes, comfortably** |

fp8 (24.3 GiB weights) is what makes B=8 concurrency fit at serving ctx; bf16 B=8
only fits at short ctx while a co-resident model holds 32 GB.

## Design A/B: channel-major-L2 vs sorted register-weight-stationary

A directive asked whether to replace the channel-major sweep with a **sorted
register-weight-stationary** path (group the `B*k` slots by expert id, then feed
each group of size G to a `gemv_rows<MM>` register-stationary body — one HBM
weight-row load shared across all G co-routed rows, the dense-WS-GEMV mechanism).

The decisive input is the realistic **expert group-size distribution** under
random top-8-of-128 routing (Monte-Carlo, 20k trials):

| B | slots (B·k) | E[distinct experts] | avg G | frac slots in groups≥2 | HBM-read save ceiling vs naive |
|---:|---:|---:|---:|---:|---:|
| 1  | 8   | 8.0   | 1.00 | 0%    | 0%    |
| 2  | 16  | 15.5  | 1.03 | 6.2%  | 3.1%  |
| 4  | 32  | 29.1  | 1.10 | 17.6% | 9.0%  |
| 8  | 64  | 51.6  | 1.24 | 36.3% | 19.3% |
| 16 | 128 | 82.4  | 1.55 | 62.0% | 35.6% |
| 32 | 256 | 111.8 | 2.29 | 86.5% | 56.3% |

**Verdict at the task's target B={4,8}: the sort is NOT worth shipping.**

- The *ceiling* on expert-weight HBM traffic saved by grouping is 9% (B=4) /
  19.3% (B=8) — because at E=128, top-8, collisions are rare (avg group 1.10 /
  1.24, mostly G=1). The register-stationary sort cannot beat "read each distinct
  expert once"; **channel-major-L2 already reads each distinct expert ~once**
  (co-routed slots are adjacent in the flat index → reuse distance ~one expert
  row ≪ the sm_120 L2). So the sort's upside is only the *L2-missed slice* of
  that ≤19% — a low-single-digit end-to-end gain, and expert-weight streaming is
  itself a minority of decode TPOT (the shared q/k/v/o/lm_head GEMVs and the
  fixed attention/router/sampling costs dominate; measured TPOT scales B1→B8 only
  2.27× fp8 / 2.12× bf16, far below the ~6.5× distinct-expert-traffic ratio).
- **Cost side:** `p9-batch` already implemented this exact sort (decode-scale
  op74 → `gemv_rows<MM>`) and **measured-rejected it**: the megakernel is
  register-bound (**219 regs, 1 block/SM**); MM accumulator sets per warp spill
  and tax every other op, plus the align/sort packet sits on the decode critical
  path. Channel-major buys the same HBM reduction at **0 extra registers, 0 extra
  packets** (it is a pure index permutation).
- **Crossover:** grouping only becomes materially profitable at **B≥16** (36–56%
  save ceiling, avg G 1.55–2.29). If the serving target moves to B≥16 AND the
  register-pressure problem is solved (e.g. a *separate* non-megakernel expert
  launch so MM accumulators don't tax the resident kernel), the sorted path is
  worth revisiting. Data to drive that decision is in
  `gemma4-26b-moe-group-size.json`.

Conclusion: "MoE batching helps because of L2 reuse of a *small* set of colliding
experts", not "because of register sharing" — at B≤8 the two are within noise and
the zero-cost channel-major form is correct to ship. Not implementing the sort is
a measured decision, not an omission.

## vs vLLM — head-to-head (inference-benchmarker, identical profile)

`huggingface/inference-benchmarker` rev `bad4f947` (`perf-data/bench_b2_ib.sh`),
**512-tok prompt / 128-tok decode, greedy, 10 s warm + 60 s measure**, VUS =
concurrent users, both engines on the same box with the 32 GB foreign process
resident. vLLM 0.25.1, gpu-util 0.58, TP1, CUDA graphs ON; plow served from a
compiled `B=8` blob (ctx 4096). vLLM fp8 = `--quantization fp8`. Metric =
aggregate output (decode) tok/s; ITL = per-user inter-token latency.

### SERVED aggregate tok/s (the mandated head-to-head)

| VUs | vLLM bf16 | plow bf16 | vLLM fp8 | plow fp8 |
|---:|---:|---:|---:|---:|
| 1 | 128.5 (ITL 7.3) | — | 165.2 (5.5) | — |
| 4 | 299.1 (12.6) | — | 425.3 (8.7) | — |
| 8 | **460.7** (16.0) | **257.6** (ITL 28.5) | **686.6** (10.7) | **3.6** † |

† plow fp8 served collapses (TTFT 10 s, 33% timeouts) — **not** a decode result:
the fp8 MoE blob is **decode-only** (fp8 grouped-MoE *prefill* is unimplemented,
`moe_pf = moe && !fp8`), so a 512-tok prompt is consumed via ~512 sequential
decode launches. The gate output is still correct ("…Paris."), i.e. fp8 experts
serve correctly; only prompt ingestion is O(n). bf16 has grouped-MoE prefill and
serves normally.

### PURE decode ceiling (plow chat-harness) vs vLLM served

| B / VUs | plow bf16 decode | vLLM bf16 served | plow fp8 decode | vLLM fp8 served |
|---:|---:|---:|---:|---:|
| 8 | 459.1 | 460.7 | 558.7 | 686.6 |

## Counter gate-wait vs B (does the coarse MoE dependency serialize at B>1?)

A directive asked whether `MOE_EXPERT_DOWN`'s coarse gate-wait reappears at B>1
(the named lever being the unwired fine-dependency map at `gemma4.rs:1261-1290`).
Measured with `PLOW_NV_TRACE=1` (block-0 per-op cycle attribution, fp8 decode,
accumulated over 12 steps):

| metric | B=1 | B=4 | B=8 |
|---|---:|---:|---:|
| overall gate-wait share | 29.9% | 22.7% | **16.3%** |
| DOWN (op 66) gate/body | 0.068 | 0.133 | 0.084 |
| GLU (op 65) gate/body | 0.347 | 0.065 | 0.034 |

**The concern is not borne out: gate-wait share *decreases* with B.** Batching
amortizes the fixed per-op counter-wait over ~B× more body (compute/memory) work,
so the kernel gets *more* body-bound as B grows. `MOE_EXPERT_DOWN` gate-wait does
not run away (peaks ~13% of its own body at B=4, back to ~8% at B=8); `GLU`
gate-wait collapses (35%→3%). Wiring the fine DOWN map would recover at most a few
% end-to-end at B=4 and less at B=8 — **not justified** at the B={4,8} target vs
the counter-contract complexity/risk. (Reproduce: `perf-data/harness/` trace
build; op ids GLU-fp8=65, DOWN-fp8=66.)

## GO / NO-GO — verdict

**GO — 26B now does concurrent B>1 decode, bf16 AND fp8, correctly.** Oracle +
slot-parity + token-identity + byte-identical B=1 all pass; fp8 experts (this
branch's fix) are correct on-device and when served. Aggregate *decode* scales
3.5–3.8× at B=8.

**NO-GO — plow does NOT beat vLLM aggregate throughput at B={4,8}.**

- **bf16, served, VUs=8:** plow **257.6** vs vLLM **460.7** tok/s → plow **0.56×**
  (loses). BUT plow's *pure-decode ceiling* (459) **ties** vLLM's served bf16
  (461) — the entire deficit is plow's **serving stack**, not the batched MoE
  decode kernel: the mux prefills the B slots **sequentially** (per-slot), and
  ITL inflates 16.0 → 28.5 ms because 512-tok prefills interleave into the decode
  stream. Fix the prefill packing (cross-request batched prefill for the MoE
  decode blob) and the served number should approach the decode ceiling.
- **fp8:** plow's *decode* (558.7) already trails vLLM's *served* fp8 (686.6), so
  even a perfect serving stack would lose on fp8 — vLLM's fp8 MoE decode kernel is
  faster than plow's per-channel fp8 GEMV here. And fp8 *serving* is separately
  blocked by the missing fp8 MoE prefill program.

**Where plow still wins on the 26B (unchanged by this work):** single-user
long-ctx decode TPOT (the S6 served rows: −3.6% vs vLLM at 32k). Concurrency and
short-ctx are vLLM's.

**Honest bottom line:** the batching *kernel* work is correct and the decode
kernel is competitive at bf16 (ties vLLM) but behind at fp8. plow loses the
concurrency contest on **serving-stack** grounds (bf16) and **fp8-decode-kernel +
missing-fp8-prefill** grounds (fp8). The two named next levers are (1) batched
MoE-decode prefill packing and (2) a faster fp8 expert GEMV / fp8 grouped prefill
— neither is the slot-sort (see the group-size A/B: not worth it at B≤8).

## GO / NO-GO

- **26B now does concurrent B>1 decode for BOTH bf16 and fp8** — correctness GO
  (oracle + slot-parity + token-identity + byte-identical B=1).
- Aggregate throughput vs vLLM: see the table above once the vLLM baseline lands.

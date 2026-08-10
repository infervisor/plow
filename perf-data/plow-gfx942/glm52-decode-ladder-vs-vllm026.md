# The decode batch ladder against vLLM 0.26 + AITER, same box, same model, same client

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4), vLLM 0.26 + AITER same box · **CDNA3-SPECIFIC** — an external reference measured here. Both engines move on CDNA4.

`glm52-decode-batch-ladder.md` measured a decode batch-size program ladder on Gemma-4-12B fp8
and left one thing open: **"vLLM at matched concurrency — not measured."** This is that
measurement. Nothing here changes the ladder's design or its plow-internal A/Bs; it supplies
the external reference they were missing and prices the result against it.

Measured 2026-08-08 20:34–21:33 UTC, one GPU-lock hold, gfx942 (MI300X, 1 GPU of 8, 304 CU).

---

## 0. THE HEADLINE, BOTH AXES

Gemma-4-12B fp8, in 1024 / out 128, 16 requests per cell, real KV, prefix caching OFF on both
engines. plow columns are `glm52-decode-batch-ladder.md` §7 and §11 verbatim; the vLLM column
is new.

### Aggregate output throughput, tok/s (higher better)

| conc | plow B=1 (shipped) | plow LADDER 1,2,4,8,16 | **plow LADDER 1,2,4 (best)** | plow B=16 | **vLLM 0.26 fp8** | vLLM / plow-best |
|--:|--:|--:|--:|--:|--:|--:|
| 1  | 80.8 | 14.4 | 46.5 | 9.1 | **116.3** | 1.44× |
| 2  | 69.2 | 23.0 | 67.1 | 16.4 | **198.5** | 2.87× |
| 4  | 74.3 | 45.1 | 118.5 | 33.7 | **340.9** | 2.88× |
| 8  | 73.8 | 72.4 | 134.7 | 62.7 | **586.8** | 4.36× |
| 16 | 73.3 | 106.4 | 134.8 | 106.1 | **918.0** | 6.81× |

### TPOT, ms/token (lower better) — the per-user axis

| conc | plow B=1 (shipped) | plow LADDER 1,2,4,8,16 | plow LADDER 1,2,4 | plow B=16 | **vLLM 0.26 fp8** |
|--:|--:|--:|--:|--:|--:|
| 1  | 10.92 | 68.49 | 20.12 | 109.74 | **8.13** |
| 2  | 13.01 | 84.45 | 27.49 | 119.60 | **9.30** |
| 4  | 12.01 | 85.36 | 30.68 | 115.18 | **10.27** |
| 8  | 12.11 | 103.53 | 27.66 | 120.03 | **10.97** |
| 16 | 12.20 | 126.29 | 27.65 | 125.63 | **12.46** |

### Mean TTFT, ms — the axis that says whether concurrency batched or queued

| conc | plow B=1 | plow LADDER 1,2,4,8,16 | plow B=16 | **vLLM 0.26 fp8** |
|--:|--:|--:|--:|--:|
| 1  | 196.8 | 197.7 | 198.3 | **68.1** |
| 2  | 1947.7 | 377.9 | 414.5 | **77.7** |
| 4  | 4713.5 | 472.9 | 510.5 | **196.9** |
| 8  | 9309.8 | 818.7 | 880.4 | **350.6** |
| 16 | 13318.0 | 2356.5 | 2512.0 | **643.6** |

(The narrow `1,2,4` ladder's TTFT was not published in §11, so it cannot be tabled.)

**There is no crossover.** vLLM is ahead on both axes at every concurrency measured, and the
throughput gap *widens* monotonically with load: 1.44× at concurrency 1 → 6.81× at 16.

**What the ladder did buy, priced against the external reference:** it halves the deficit at
the top of the range. The shipped B=1 blob is **12.5×** behind vLLM on aggregate throughput at
concurrency 16 (73.3 vs 918.0); the best ladder is **6.8×** behind (134.8). That is a real
1.84× improvement and it is exactly the improvement §11 claimed — it just lands in the middle
of a gap that is an order of magnitude wide, not at its edge.

**Read the plow B=1 row's flat TPOT correctly.** Arm a's 12.20 ms at concurrency 16 is *lower*
than vLLM's 12.46, and that is not a win: it is a serialized stream. Its TTFT at that point is
13,318 ms against vLLM's 644 — the requests are queueing one at a time, so its "per-user
latency" is measured on a user who has already waited 13 seconds. §7 of the ladder report says
this in its own terms; it needs saying again the moment a vLLM column sits next to it.

---

## 1. Why the gap is what it is: plow's decode barely amortises the weight stream across rows

One number explains the whole table. Take TPOT at concurrency 1 and at concurrency 16 and
divide by the rows in flight — that is milliseconds per *token emitted by the engine*:

| engine / blob | ms/token at 1 row | ms/token at 16 rows | amortisation 1→16 |
|---|--:|--:|--:|
| plow `PLOW_DECODE_BATCH=1` | 10.92 | 12.20 (1 row; the rest queue) | **1.00×** |
| plow `PLOW_DECODE_BATCH=16` | 109.74 | 125.63 / 16 = **7.85** | **1.39×** |
| plow LADDER `1,2,4` (B_max 4) | 20.12 | 27.65 / 4 = **6.91** | **2.91×** |
| **vLLM 0.26 fp8** | 8.15 | 12.94 / 16 = **0.81** | **10.08×** |

A decode step is a weight stream. The whole point of batching it is that 16 rows re-use one
pass over the weights, so the marginal cost of row 16 should be near zero and the amortisation
should approach 16×. vLLM gets 10.1× of that. plow's widest batched blob gets **1.39×** — it
advances 16 rows for 11.5× the time of advancing one.

That is not a scheduling defect and no ladder can fix it: the ladder chooses *which* decode
program runs, and every program available to it has this property. §11 already isolated the
mechanism from the inside — the `PLOW_GEMV_MM` object tax, 1.83× at MM=4 and **6.19× at
MM=16** on an identical one-row blob — and concluded "ship at B_max = 4". The vLLM column says
what that conclusion costs: capping B_max at 4 caps aggregate throughput at
`B_max / TPOT(B_max)` ≈ 4 / 27.65 ms ≈ **145 tok/s**, which is 134.8 measured, against an
engine with no such cap. **The ladder is a correct scheduler sitting on top of a decode kernel
that does not batch.** Fixing `gemv_rows<MM>` is worth more than any rung count.

---

## 2. THE RUNG-QUANTISATION QUESTION, AND A CORRECTION TO ITS PREMISE

The commission's caution was that vLLM's scheduler is token-budget based
(`num_new_tokens = min(num_new_tokens, token_budget)`), so its batching is continuous and a
padded plow rung must not be compared against an exact vLLM batch. **That is half right, and
the other half changes the question.**

**Scheduling is continuous — confirmed directly.** The server's own engine stats over the
sweep, at a 1 s logging interval, show running batches that are not powers of two:

    Running: 1, 2, 3, 4, 6, 8, 9, 12, 15, 16 reqs

Every requested concurrency appears as its own running-batch size. Nothing is rounded at admission.

**Execution is NOT continuous — it is quantised, to the same rungs.** vLLM 0.26 captures full
CUDA graphs and pads the decode batch up to the next captured size. This server's captured
sizes, from its own config dump:

    cudagraph_capture_sizes: [1, 2, 4, 8, 16, 24, 32, 40, ... 512]

Below 16 that is **[1, 2, 4, 8, 16] — the ladder's rung set, exactly.** A running batch of 3
replays the size-4 graph; 6 replays 8; 12 replays 16. So the two engines make the same
discretisation choice at the same points, and the comparison is not "quantised vs continuous".

**So how much does the quantisation cost?** Measured directly, by sweeping the off-rung
concurrencies 3, 6 and 12 alongside the rungs, with the per-cell request count scaled to
`4 × conc` so every cell runs exactly four waves (see §5 for why that control is required):

| conc | 1 | 2 | **3** | 4 | **6** | 8 | **12** | 16 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TPOT ms | 8.15 | 9.31 | **10.37** | 10.30 | **10.51** | 11.03 | **12.01** | 12.94 |
| tok/s | 116.0 | 197.4 | **261.1** | 340.2 | **476.8** | 584.3 | **766.4** | 912.7 |

The worst off-rung point is concurrency 3, which pads to the size-4 graph and pays **+0.7% TPOT
against concurrency 4 itself** (10.37 vs 10.30, against a 0.4–0.6% rep spread — real, but
0.7%). Concurrency 6 pads to 8 and is *faster* than 8. Throughput rises monotonically through
every off-rung point.

**Conclusion: rung quantisation costs under 1% on an engine whose per-row decode cost is
small, and it is therefore not what the ladder should worry about.** §9 of the ladder report
argued from the plow side that finer rungs (3/6/12) are not worth buying until the object tax
is gone; this is independent confirmation from the other engine, and it strengthens the
argument — the granularity axis is worth ~1%, the object axis is worth 6.19×.

**What is NOT measured, and must not be inferred:** plow's own off-rung cost. The ladder's §7
and §11 sweeps ran concurrency 1/2/4/8/16 only — *exactly* its rungs — so the ladder has never
been run at an occupancy its rung set does not cover, and the report says so ("Rungs finer than
powers of two... emit and the blob is correct; no timing was taken"). The plow ladder's
quantisation tax remains an unmeasured quantity. It cannot be assumed to be vLLM's 0.7%,
because plow's TPOT spread across rungs is much wider (20.12 → 27.65 over 4× rows on the narrow
ladder, 68.49 → 126.29 over 16× on the wide one), which is precisely the spread that bounds the
wasted-row cost.

---

## 3. WHAT WAS RUN — versions, flags, and the client

**vLLM.** `0.26.0+rocm723`, native from the `/workspace/vllm26` venv (no docker daemon on this
box; this is the same install the campaign's other 0.26 baselines used —
`fusion-review-and-crossover-sweep.md` §15/§17 — so the methodology matches the recorded one
rather than the container script's 0.23.0).

    vllm       0.26.0+rocm723
    amd-aiter  0.1.16.post3
    torch      2.11.0+gitd0c8b1f
    triton     3.6.0
    ROCm       7.2.4 (/opt/rocm-7.2.4)

Serve flags, as the server itself logged them:

    vllm serve /workspace/models/gemma-4-12B-it \
      --served-model-name gemma-4-12B-it --tensor-parallel-size 1 \
      --max-model-len 4096 --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.85 --quantization fp8 \
      --no-enable-prefix-caching --port 8477
    env: HIP_VISIBLE_DEVICES=0  VLLM_ROCM_USE_AITER=1  HF_HUB_OFFLINE=1

    Model loading took 12.49 GiB          (bf16 on this box loads 22.73 GiB — fp8 weights confirmed)
    GPU KV cache size: 463,488 tokens
    Graph capturing finished in 17 s, 5.38 GiB   (PIECEWISE + FULL, 51 sizes)

**AITER is engaged, but not everywhere — say it plainly.** The server logs
`IrOpPriorityConfig(rms_norm=['aiter','native'], fused_add_rms_norm=['aiter','native'])` and
imports `module_aiter_core`, `module_rmsnorm`, `module_quant`; attention selects
**`TRITON_ATTN`**, not an AITER attention backend. So "vLLM 0.26 + AITER" here means AITER
norms and quantisation with Triton attention. This is the stack's own default choice for
Gemma-4 on gfx942 (sliding-window attention), not a configuration decision made here, but a
reader comparing against the GLM baselines — which do run AITER sparse-MLA — should not assume
the same code paths.

**plow.** No plow blob was rebuilt or re-emitted. The plow rows are
`glm52-decode-batch-ladder.md` §7/§11 as published, from branch `decode-ladder`
(`PLOW_FP8=1 PLOW_W8A8=1`, `--seq 128,512,1024 --max-ctx 4096`, fused activation quant off).

**The client is the same code on both engines.** `scripts/bench_speed.sh` carries a warning
that its numbers must not be tabled next to a vLLM number, *because it is a different client*.
The fix for that is not to avoid the comparison; it is to point one client at both servers. The
metric block of `bench_speed.sh` is now extracted to **`scripts/sweep_client.py`** — same TTFT
(first SSE content delta), same TPOT (`(last−first)/(n−1)`), same ITL p99, same
`deltas / wall` throughput, same filler prompt, same 16-requests-per-cell queue-and-drain. Both
engines were driven by it. The driver is **`scripts/bench_vllm26_native.sh`**.

**Prompt length, on each engine's own tokenizer:** the `in_len 1024` cell is
**1030 prompt tokens** as counted and reported by both servers' `usage.prompt_tokens`
(the filler is 1017 raw tokens plus the chat template). Identical text to both.

---

## 4. THE GATES

* **Coherence, both engines, every server instance, before every sweep.** vLLM:
  `'The capital of France is Paris.'` plow: `'The capital of France is **Paris**.'` No arm
  below was measured on an ungated server.
* **Real KV, and the server proves it.** Prefix caching was disabled on the primary arm and the
  server logged **`Prefix cache hit rate: 0.0%`** on five consecutive samples. This is stated
  because a prior campaign baseline was retracted when an "8k win" turned out to be a garbage-KV
  bench artifact; the point here is a *cache* rather than a garbage cache, but the discipline is
  the same — the engine's own report, not an assumption. §6 shows what the number becomes when
  the cache is left on, which is why it was turned off.
* **3 interleaved reps** (rep loop outermost: rep1 sweeps all concurrencies, then rep2, then
  rep3), all inside one lock hold, one server instance per arm.
* **Control spread, measured rather than assumed** — and this is the surprise:

  | arm | worst rep-to-rep spread, TPOT | worst spread, tok/s |
  |---|--:|--:|
  | vLLM fp8, fixed 16 req/cell | **0.8%** | 0.9% |
  | vLLM fp8, balanced load | **1.0%** | 0.8% |
  | plow, shipped B=1 asset (§5) | **17.9%** | 16.4% |

  The box's ±20% DVFS noise is real **on plow and not on vLLM**. Every vLLM number in this file
  is reproducible to ~1%; the plow instrument check moved 15.10 → 18.05 → 16.39 ms across three
  back-to-back reps at concurrency 1. Consequence for reading the tables: **differences inside
  ~18% in a plow column are not separable**, while the plow-vs-vLLM gaps (1.4× to 6.8×) are far
  outside it and are not at risk from this.

---

## 5. THE CLIENT WAS CALIBRATED, NOT TRUSTED — and one plow instrument check

**Against the reference client.** The same vLLM server, same points, was also driven by
`vllm bench serve --backend openai-chat --dataset-name random --ignore-eos`, the client the
campaign's stored baselines used:

| conc | sweep_client TPOT | `vllm bench serve` TPOT | Δ | sweep_client tok/s | `vllm bench serve` tok/s |
|--:|--:|--:|--:|--:|--:|
| 1  | 8.13 | 8.40 | +3.3% | 116.3 | 110.8 |
| 2  | 9.30 | 9.34 | +0.4% | 198.5 | 185.3 |
| 4  | 10.27 | 10.59 | +3.1% | 340.9 | 331.2 |
| 8  | 10.97 | 11.04 | +0.6% | 586.8 | 351.0 † |
| 16 | 12.46 | 12.56 | +0.8% | 918.0 | 908.6 |

TPOT agrees to **≤3.3%** at every point, throughput to ≤7% except one outlier. † the
concurrency-8 reference throughput is discarded as a single-invocation hiccup: its mean TTFT is
1514.9 ms where its neighbours are 200 and 656, i.e. that one bench process stalled at start;
its TPOT (11.04, the steady-state metric) is in line. **The extracted client is the same
instrument as the reference client.** That is what licenses putting a plow column and a vLLM
column in the same table.

**Against plow.** `sweep_client.py` was also run against a live plow server in the same lock
hold — the **shipped `/workspace/assets/gfx942/g12b-fp8` asset with its own bundled objects**,
concurrency 1/2/4/8/16, 3 reps:

| conc | TPOT ms | tok/s | TTFT ms |
|--:|--:|--:|--:|
| 1  | 16.51 | 56.2 | 192.8 |
| 2  | 18.34 | 50.8 | 2562.7 |
| 4  | 17.36 | 53.5 | 6429.7 |
| 8  | 16.52 | 55.9 | 12266.6 |
| 16 | 16.28 | 56.7 | 17133.1 |

This reproduces the *shape* of ladder arm a exactly — flat TPOT, throughput flat near its
one-row rate, TTFT climbing 89× across the range, i.e. pure queueing at `batch=1` — and its
TTFT at concurrency 1 (192.8) matches arm a's 196.8 to 2%. It does **not** reproduce arm a's
10.92 ms TPOT, and the reason is known and is not the client: this stored asset carries the
2026-08-04 object generation (the README's "plow fp8+occ4 = 15.27 ms" line), while arm a was
built on `decode-ladder` HEAD with `PLOW_OCC4=1 PLOW_GEMV_MM=1` objects on the ROCm 7.2.4
runtime, which the campaign has separately priced at −0.5 ms/token. **So this is an instrument
check, not a plow arm, and it is labelled as one.**

**Load-balance control.** `bench_speed.sh` sends a fixed 16 requests per cell, so a cell runs
`ceil(16/conc)` waves and the last one is short whenever `conc` does not divide 16. At
concurrency 12 that is one full wave of 12 plus a wave of 4, and it depressed measured
throughput to 587.4 tok/s — *identical to concurrency 8* — which would have read as a
scheduler cliff and is nothing of the kind. Re-running with the request count scaled to
`4 × conc` (four full waves at every point, `NPROMPT_SCALE=4`) gives **766.4 tok/s** at
concurrency 12 and a monotone curve. The ladder's own sweeps only ever used divisor
concurrencies, so they never met this; any off-rung sweep must control for it. Both arms are
tabled: §0 uses the fixed-16 protocol because that is what the plow rows used; §2 uses the
balanced protocol because that is what an off-rung question requires.

---

## 6. What vLLM's default prefix caching would have done — and why it is off

`sweep_client.py` sends the *same* prompt to all 16 requests of a cell (`bench_speed.sh` always
did). plow has no prefix cache, so for plow that is 16 independent prefills. vLLM's default APC
would serve 15 of the 16 out of cache. Measured, same server, `PREFIX=on`, 2 reps:

| conc | TPOT ms | tok/s | TTFT ms | prefix hit rate |
|--:|--:|--:|--:|--:|
| 1  | 8.07 | 121.3 | **31.0** | — |
| 4  | 10.12 | 382.7 | **51.2** | — |
| 16 | 11.61 | **1319.8** | **73.8** | **97.5–98.4%** |

TTFT collapses 644 → 74 ms and throughput rises 918 → 1320 tok/s at concurrency 16. **None of
that is decode work** — it is 98% of the prefills not happening. Tabling it against plow would
have credited vLLM with a feature plow does not have, on a workload the client makes
artificially cacheable. It is recorded here and used nowhere above. (The campaign's stored
baselines are unaffected by this: they used `vllm bench serve --dataset-name random`, whose
prompts share no prefix, so their default-on cache had nothing to hit.)

---

## 7. WHERE THESE NUMBERS ARE NOT COMMENSURABLE — the full list

1. **Not same-session.** The plow rows were taken in an earlier lock hold on 2026-08-08; the
   vLLM rows between 20:34 and 21:33 UTC the same day. The campaign's own gold standard
   (`fusion-review-and-crossover-sweep.md` §15) is strict alternation of both engines within one
   window, and **this is not that.** What was done instead: the same client was pointed at a
   live plow server inside the vLLM lock hold (§5), which fixes the *instrument* but not the
   *session*. Given the 17.9% rep spread measured on plow and the ≥1.44× gaps in every cell,
   this does not threaten any conclusion here, but a future ±20% claim about plow would need the
   alternating protocol.
2. **Quantisation is analogous, not identical.** plow is fp8 weights + w8a8 through its own
   encoder; vLLM is `--quantization fp8`, dynamic per-tensor weight fp8 with per-token
   activation fp8, applied at load to a bf16 checkpoint (12.49 GiB resident vs bf16's 22.73).
   Both are "fp8 W8A8" and neither is the other's kernel. A bf16 vLLM arm was not run.
3. **AITER covers norms and quant, not attention** (§3). Attention is `TRITON_ATTN`.
4. **The engines' capacity limits differ and it was left that way.** plow's slot table is
   `B_max` (4 or 16); vLLM ran its default `max_num_seqs` with a 463,488-token KV cache. Neither
   binds below concurrency 16, so no cell in these tables is affected, but the two are not
   configured to the same capacity and a sweep past 16 would diverge for that reason alone.
5. **Throughput is counted in stream deltas, not tokenizer tokens.** That is `bench_speed.sh`'s
   definition and it is applied identically to both engines, but they are not the same quantity:
   at concurrency 2 and 3 some vLLM requests hit EOS before 128 tokens (median
   `completion_tokens` 114 at concurrency 3), so those cells emit fewer than `16 × 128` deltas.
   The reference-client cross-check in §5 ran `--ignore-eos` and agrees on TPOT regardless.
6. **`max_model_len 4096` on vLLM matches plow's `--max-ctx 4096` blob**, so the KV geometries
   are comparable — but vLLM's 463k-token cache is far larger than plow's 16 slots × 4096, and a
   larger cache is not free of paging effects at higher load than measured here.
7. **The narrow ladder's TTFT was never published**, so the TTFT table has no `1,2,4` column.

---

## 8. VERDICT, AND WHAT IT CHANGES

**Does the ladder's throughput result survive contact with an external reference?** Yes as a
plow-internal result, and it is smaller than it looks in absolute terms.

* The ladder's own claim — 1.84× aggregate throughput over the shipped B=1 blob at concurrency
  16, at 3.4× better concurrency-1 TPOT than the wide ladder — **is not challenged by anything
  here.** It is a claim about plow arms and it stands.
* Against vLLM 0.26 + AITER on the same box, same model, same client: **plow is behind on
  aggregate throughput at every concurrency**, by 1.44× (concurrency 1) to 6.81× (concurrency
  16). On TPOT it is behind by 1.34× at concurrency 1 and 2.21× at concurrency 16 against every
  arm that actually batches. The **one** cell where a plow number is lower is arm a's 12.20 ms
  at concurrency 16 against vLLM's 12.46 — and that arm's TTFT in the same cell is 13,318 ms
  against 644, so it is not serving 16 users at 12.20 ms, it is serving one at a time (§0). The
  ladder moves the concurrency-16 throughput deficit from 12.5× to 6.8×.
* **The crossover does not exist in this range** and the trend is the wrong way: the gap widens
  with load, because plow's aggregate is capped at `B_max / TPOT(B_max)` while vLLM's is not.

**The one thing this measurement changes about what to build next.** §11 recommended shipping
the ladder at `B_max = 4` because raising it is negative on both axes. That recommendation is
correct and this confirms its premise from outside — but it should be read as a *containment*,
not a destination: `B_max = 4` fixes the aggregate ceiling at ~145 tok/s. The ranked follow-up
from the outside view is therefore:

1. **`gemv_rows<MM>` is the whole gap.** 6.19× object tax at MM=16, 1.83× at MM=4, on an
   identical one-row blob; and the resulting 1.39× weight amortisation across 16 rows against
   vLLM's 10.08× (§1). Everything else in the ladder's follow-up list is worth less than this.
2. **Per-rung object selection** (ladder report §10) is the structural version of the same
   thing and is what would let `B_max` rise again.
3. **Finer rungs are worth ~1%** and should stay closed — now measured on the other engine
   (§2), not just argued.

### Reproducing this

    # vLLM arm (server + both clients + scheduler trace), one command:
    CONCS="1 2 4 8 16" NPROMPT=16 OUTLEN=128 IN_LENS=1024 REPS=3 \
      QUANT=fp8 PREFIX=off CALIB=1 PORT=8477 \
      bash scripts/bench_vllm26_native.sh fp8-nopc

    # off-rung / balanced-load arm:
    CONCS="1 2 3 4 6 8 12 16" NPROMPT_SCALE=4 REPS=3 QUANT=fp8 PREFIX=off CALIB=0 \
      bash scripts/bench_vllm26_native.sh fp8-bal

    # the same client against any plow server:
    BASE_URL=http://127.0.0.1:8196 MODEL=auto TAG=plow \
      IN_LENS=1024 CONCS="1 2 4 8 16" NPROMPT=16 OUTLEN=128 REPS=3 \
      python3 scripts/sweep_client.py

Raw per-rep CSVs, server logs and provenance dumps for every arm are under
`/tmp/vllm26_fp8-nopc/`, `/tmp/vllm26_fp8-bal/`, `/tmp/vllm26_fp8-pc/` and `/tmp/plow_ctl/`;
the per-rep rows are reproduced in `glm52-decode-ladder-vs-vllm026.csv` next to this file.

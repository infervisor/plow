# plow end-to-end serving — capacity report **v2** (Gemma-4, sm_120)

Date: 2026-07-21. Branch: `main` @ `b953a7b` (all serving features merged:
prefill-in-serve, batch>1, chunk-interleaved admission with the prefill-aware
shed fix `2ba3fbc`, multi-model manager; VMM prefix flag-gated **off** —
measured with defaults). Box: 1× RTX PRO 6000 Blackwell 96 GB (sm_120, 188
SMs), driver 580.82.07, CUDA 13.0. Engine: `plowrt serve` (OpenAI-compatible,
`cuda,hf-tokenizer`), persistent sm_120 interpreter, global-queue mux, chunk-
interleaved prefill admission, **default `--slo-ms 250`**. Comparator: **vLLM
0.25.1**, same checkpoints/revisions, same GPU, same harness.

The multi-user numbers come from `huggingface/inference-benchmarker` rev
`bad4f947` (pinned), **identical tool / tokenizer / prompt profile / durations
for both engines** — directly comparable. TTFT **includes server-side
queueing** (capacity convention). Data: `b2-concurrency-12b.json`,
`b2-concurrency-family.md`; single-user latency: `gemma4-{12b,31b,26b}-plowrt-
served*.md`.

> **Honesty banner.** This v2 pass measured the **12B concurrency head-to-head
> in full** and the 12B **16k** plow points. The run was **stopped early by
> operator request** before the 31B/26B concurrency sweeps and the 12B+26B
> co-resident scenario. Those rows are **single-user only** here (committed S6
> data) and are labeled as such. Nothing below is projected — every cell is a
> measured row with a source file.

> **ADDENDUM — campaign `B2-bfix` (2026-07-21, mux fix `498a0f4`).** v2 blamed
> the B=16/B=24 truncation on a "638ce37 B>8 **emitter bug**." **That diagnosis
> was wrong.** The kernel is correct: with the admission shedder disabled a
> B=16 blob serves 8 **and 16** concurrent distinct-prompt streams
> **byte-identical** to their B=1 runs (compute-sanitizer `0 errors` on every
> batched decode kernel at M≤32). The real cause was the **mux admission model**
> — `predicted_wait = live × service_ms` (serial M/M/1) sheds every live slot
> when `live × step_ms > --slo-ms`; a B=16 blob steps at ~40 ms/token, so 8
> users = 320 ms > 250 ms and all streams were 429'd. Fixed
> (`predicted_wait = ceil(live / batch) × service_ms`, B=1 byte-identical).
> **After the fix, B=16 serves up to 16 streams with 0 dropped requests.**
> **But it does NOT raise SLO-bounded capacity** (§1, §4): at `GV_MM_MAX=8` a
> B=16 decode step re-reads all 22 GiB of weights **twice**, so doubling the
> batch doubles step time — **same aggregate throughput, 2× the per-token
> latency**. 12B decode is **HBM-bandwidth-bound**; batching alone buys nothing
> under the ITL SLO. **B=8 stays the best 12B serving config; plow still trails
> vLLM.** The genuine levers are wide single-pass GEMV (`GV_MM_MAX≥16`) and
> faster prefill — not the batch cap. New rows: tag `plow-b16-bfix` (valid).

---

## 1. Headline — max users supported, throughput, latency (per model, vs vLLM)

**Convention.** "Max users" = highest fixed-VU point sustained with **0 failed
requests** under the stated SLO. Two SLOs: **TPOT/ITL p99 ≤ 50 ms** and **TTFT
p99 ≤ 5 s**. "Peak tok/s" = unconstrained max-throughput point.

| model | prec | profile | engine | max users (both SLOs) | peak tok/s (VU) | single-user TTFT / TPOT | source |
|---|---|---|---|---|---|---|---|
| **12B** | bf16 | 4k/128 | **plow** B=8 | **2** (54 tok/s) | 97 (VU8) | 0.80 s / 21.8 ms | B2-ib |
| **12B** | bf16 | 4k/128 | **plow** B=16-bfix (mm8, 2-pass) | **1** (22 tok/s) | 93 (VU16) | 0.82 s / 39.7 ms | B2-bfix |
| **12B** | bf16 | 4k/128 | **plow** B=16-mm16 (1-pass) | **2** (46 tok/s) | 102 (VU16, prefill-bound) | 0.81 s / 28.3 ms | WS-batched-gemv |
| **12B** | bf16 | 4k/128 | **vLLM** | **8** (177 tok/s) | **239 (VU32)** | 0.38 s / 20.0 ms | B2-ib |
| **12B** | bf16 | 16k/128 | **plow** | **1** (23 tok/s) | 31 (VU4) | 2.49 s / 22.8 ms | B2-ib |
| **12B** | bf16 | 16k/128 | vLLM | *not run this pass* | — | — | — |
| **31B** | bf16 | 4k/128 | plow | *single-user only* | — | **1.65 s / 46.4 ms** | S6-31b |
| **31B** | bf16 | 4k/128 | vLLM | *single-user only* | — | 0.71 s / 45.2 ms | S6-31b |
| **26B-A4B** | bf16 | 4k/128 | plow | *single-user only* | — | **0.30 s / 8.48 ms** | S6-26b |
| **26B-A4B** | bf16 | 4k/128 | vLLM | *single-user only* | — | 0.17 s / 7.90 ms | S6-26b |

**The capacity answer, stated plainly:**

- **12B: vLLM wins the multi-user contest decisively — ~4-8× the SLO-bounded
  users (8 vs 1-2) and 2.5× peak throughput (239 vs 97 tok/s).** Two structural
  causes in plow today, both real and both **bandwidth**, not a batch cap:
  (1) **decode is HBM-bandwidth-bound.** The `B2-bfix` fix (§4a) removed the
  admission shed that made B=16/B=24 blobs *appear* broken, but re-measuring the
  fixed B=16 blob shows **no throughput gain** over B=8 (peak 93 vs 97 tok/s):
  at `GV_MM_MAX=8` a B=16 step re-reads all weights twice, so 2× the batch =
  2× the step time = same tokens/s and **2× the per-token latency** — B=16
  actually *loses* an SLO-user vs B=8 (ITL p99 51 ms already at VU2). Batching
  helps only once the GEMV does B rows in **one** weight pass (`GV_MM_MAX≥16`).
  (2) prefill is **~2× slower** (TTFT 0.80 s vs 0.38 s), documented HBM-bound;
  TTFT p99 blows the 5 s SLO by VU8 regardless of batch. vLLM's continuous
  batching + faster paged/fused attention interleaves prefill+decode and keeps
  both ITL and TTFT under SLO out to VU8.
- **31B / 26B: single-user latency is at parity or better for plow, but
  multi-user capacity was not measured this pass.** 31B TPOT 46.4 ms (+2.6% at
  4k, **wins −0.7% at 32k**); 26B TPOT 8.48 ms (+7.4% at 4k, **wins −3.6% at
  32k**). Both are **B=1** blobs in production; 26B has a **gate-passing B=8
  blob** built this pass (§4) but was not concurrency-benched before the stop.

## 2. 12B concurrency detail (the measured head-to-head)

Fixed-VU, 120 s, 0 failures throughout. Full table + percentiles:
`b2-concurrency-family.md`.

| VU | plow B=8 tok/s | plow B=8 ITL p99 | plow B=16-bfix tok/s | plow B=16-bfix ITL p99 | plow B=16-bfix TTFT p99 | vLLM tok/s | vLLM ITL p99 | vLLM TTFT p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 35.8 | 21.9 | 21.9 | 39.7 | 0.82 s | 44.0 | 20.1 | 0.39 s |
| 2  | 54.4 | 43.4 | 35.7 | 51.4 | 1.68 s | 83.5 | 26.9 | 0.72 s |
| 4  | 75.5 | 59.8 | 54.9 | 68.8 | 3.59 s | 126.1 | 30.4 | 1.35 s |
| 8  | 97.2 | 81.6 | 75.6 | 115.1 | 7.64 s | 177.3 | 41.8 | 2.68 s |
| 16 | 97.1 | 82.4 (TTFT 16.5 s) | 92.9 | 163.6 | **16.3 s** | 220.7 | 68.5 | 5.32 s |
| 32 | — (B=8 caps at 8) | — | — (B=16 caps at 16) | — | — | 238.8 | 122.7 | 10.75 s |

**Read this carefully.** The `B2-bfix` fix means B=16 now serves **all 16
slots with 0 dropped requests** (the pre-fix "truncation" is gone). But B=16 is
**not faster** than B=8: its per-token ITL is *higher* at every VU (39.7 vs 21.9
at VU1, 115 vs 81.6 at VU8) because each decode step re-reads all weights twice,
and its peak throughput (92.9) ties B=8 (97). B=16 crosses the 50 ms ITL SLO at
**VU2**; B=8 crosses at VU4. Neither keeps TTFT p99 ≤ 5 s past VU4. **Batching
the decode did not move plow's SLO-bounded capacity — the bound is bandwidth,
not the batch size.**

At VU16 plow's decode ITL stays flat (82 ms) while TTFT explodes to 16.5 s —
**textbook mux queueing above B, not a decode regression**. That IS the
capacity answer: plow serves ≤B concurrent streams at full decode rate, then
queues. vLLM degrades gracefully via continuous batching but its ITL p99
crosses 50 ms by VU16 and TTFT crosses 5 s by VU32 — its own capacity wall,
just much further out.

## 3. Single-user latency — all three models (served, HEAD, vs vLLM)

TTFT (OpenAI SSE, includes tokenize + prefill + first token) and TPOT (per
decode token), single stream, greedy. Sources: `gemma4-{12b,31b,26b}-plowrt-
served*.md` (all pass Paris + token-parity gates).

| model | ctx | plow TTFT | vLLM TTFT | plow TPOT | vLLM TPOT | verdict |
|---|---:|---:|---:|---:|---:|---|
| 12B | 4k   | 0.70 s | 0.32 s | 18.47 ms | ~19.8 ms | plow wins TPOT; vLLM wins TTFT |
| 12B | 128k | 37.79 s | 16.3 s | 18.5→24.3 ms scaling | — | vLLM wins TTFT ~2.3× |
| 31B | 4k   | 1.65 s | 0.71 s | 46.36 ms | 45.20 ms | +2.6% TPOT |
| 31B | 32k  | 14.12 s | 10.73 s | **48.78 ms** | 49.14 ms | **plow wins TPOT −0.7%** |
| 26B | 4k   | 0.30 s | 0.17 s | 8.48 ms | 7.90 ms | +7.4% TPOT |
| 26B | 32k  | 2.72 s | 1.54 s | **9.22 ms** | 9.57 ms | **plow wins TPOT −3.6%** |

**Pattern:** plow's decode (TPOT) is at parity or ahead, and the gap turns into
a **win at long context** on every model (the flash-decode ladder). plow's
**prefill (TTFT) is the consistent structural loss** — ~2× behind at all
contexts. Serving overhead itself is negligible: +0.88% over the raw kernel on
12B, unmeasurable on the 46 ms/9 ms 31B/26B steps.

## 4. What holds each model back (honest, per model)

**12B — the limiter is decode HBM bandwidth + prefill, NOT a batch cap.**

**4a. The "emitter bug" was a misdiagnosis — it was the admission shed.**
v1/v2 recorded the B=16/B=24 blobs as "emitter-broken: pass 4-way, fail 8-way,
truncate to ~12 tok/req." Reproduced, traced, and **disproven**:
- With `--slo-ms 100000` (shedder off) the **same** B=16 blob serves **8 and 16**
  concurrent distinct-prompt streams **byte-identical** to their B=1 runs
  (64/64 tokens each; `bfix_verify_out.txt`, `bfix_x_out.txt`). The B=32 blob
  loads and decodes correctly too. compute-sanitizer memcheck on the batched
  decode kernels (bf16/fp8/w8a8 GEMV+GLU, flash, argmax) at M up to 32:
  **`ERROR SUMMARY: 0 errors`**. The 638ce37 kernel is correct.
- The truncation was `crates/plowrt/src/serve/mux.rs`: the admission gate
  computed `predicted_wait = live × service_ms` (serial M/M/1) and **429'd every
  live slot** once it crossed `--slo-ms` (default 250). A B=16 blob steps at
  ~40 ms/token, so 8 users = 8×40 = 320 > 250 → shed. B=8 survived only because
  8×20 = 160 < 250 — luck of the threshold, not correctness. Fixed
  (`498a0f4`): `predicted_wait = ceil(live / batch) × service_ms`, which is the
  real cost of a data-parallel launch and stays byte-identical for B=1
  (`capacity == 1` ⇒ the old formula). After the fix the B=16 sweep runs
  VU1..16 with **0 sheds, 0 failed requests** (tag `plow-b16-bfix`, valid).

**4b. But batching does not move 12B SLO-capacity — decode is bandwidth-bound.**
Re-measured with the fix, B=16 peak = **92.9 tok/s** (VU16) vs B=8's **97**
(VU8): a **tie**. At `GV_MM_MAX=8` a B=16 GEMV does `ceil(16/8)=2` weight passes,
re-reading all 22 GiB of weights twice, so 2× the batch buys 2× the step time —
identical tokens/s, **2× the per-token latency**. B=16 crosses the 50 ms ITL SLO
at VU2 (B=8 at VU4), so under the dual SLO B=16 max-users = **1** (B=8 = 2). The
real throughput lever is a **single-pass wide GEMV** (`GV_MM_MAX≥16`, the
"~1.3× aggregate" knob flagged in `op_gemm.cuh`) — a cubin/register-pressure
tradeoff, not shipped here. (The ctx-8k B=32 blob's 84 GiB KV still overflows
the planner; a **ctx-4096 B=32** blob (42 GiB KV) fits and was built/gated for
**correctness only** — 32-way token identity + compute-sanitizer. Its 4k/128
sweep is a ctx/profile mismatch (4128 > 4096 ⇒ every request context-rejected),
so tag `plow-b32-bfix` is flagged `"valid": false` and used for **no** capacity
claim; a valid B=32 sweep needs a ctx≥~4300 blob.)
- Prefill ~2× slower than vLLM (HBM-bound flash, no TMA/FA4) → TTFT p99 blows
  the 5 s SLO by VU8 on **any** batch, independent of the decode fix.

**4c. WS-batched-gemv (2026-07-21) — the single-pass wide GEMV, now measured.**
The §4b/§7.1 lever (`GV_MM_MAX≥16`) is built and gated
(`perf-data/ws-batched-gemv.md`; committed `op_gemm.cuh` per-MM shallow unroll).
ptxas: MM16 = **234 regs, 0 spill** (the max rung at 0 spill; occ-1 cooperative
launch makes the 212→234 reg rise free — B=1 unchanged at 18.6 ms); MM32 = 255
regs, small spill (0.7% B=1 tax). Gates all pass: oracle gemv/qkv/glu M≤32 vs
f32 (relL2 ≤1.7e-3), compute-sanitizer 0 errors on the MM16/MM32 rungs, B=16/32
isolation 0 cross-bleed. **Decode microbench (the point):** a B=16 step in **one**
weight pass hits **475 tok/s** (vs 353 for the shipped 2-pass MM8, **+34%**) at a
33.7 ms per-user TPOT — batching finally scales past the 8-wide 325 tok/s
ceiling. **Served (B=16-mm16 blob):** MM16 beats the 2-pass B=16-bfix blob **17–34%
tok/s at every VU** and restores B=16 SLO-capacity from **1 → 2 users**. **But
max-users does not pass 2** (= B=8): ITL p99 crosses 50 ms at VU4 and TTFT blows
5 s by VU8 — both **prefill-driven** (new 4k prefills interleave with decode; the
16-wide kernel runs even at partial occupancy). Saturated peak (VU16, shedder
relaxed, clean 127 tok/req) = **101.8 tok/s — barely above B=8's 97**: at
saturation the decode kernel shares the GPU with 16 concurrent 4k prefills, so
serving ITL (139 ms) runs **4.1× the pure-decode TPOT (33.7 ms)** and the 475
tok/s the kernel can do does not survive. **The serving peak is prefill-bound,
not weight-pass-bound.** **Net: the WS GEMV closes the decode-throughput half of
the gap to vLLM (and is a real win at low concurrency / for the kernel), but the
12B serving plateau is set by prefill — that is now the sole lever left.** Full
curve + peak: `ws-batched-gemv.md` §4.

**31B — KV wall + no batched blob measured.**
- 132k B=1 blob is **22.6 GiB KV/seq beside 57.2 GiB weights ≈ 82.5 GiB of 96
  GiB** — exactly **one** 132k sequence fits (vLLM has the same wall: its 31B
  bf16 KV pool holds ~1.45× a 132k request). A mid-ctx B>1 31B blob was **not
  built/gated this pass**.
- Prefill 2.13× behind at 1k, narrowing to **1.04× (near parity) at 128k**.

**26B-A4B — batched blob exists, not yet concurrency-benched.**
- A **B=8 ctx-8k MoE blob was built and passed the 8-way token-identity gate
  this pass** (`/root/final-blobs/26b-ctx8192-b8`, 268 tok/s aggregate on the
  8-way isolation smoke). The B2 concurrency sweep for it was **cut by the
  operator stop** — the one clearly-actionable "finish next lease" item.
- Production blob is still B=1; MoE prefill 1.48–1.73× behind vLLM's autotuned
  CUTLASS MoE (router-split + packet fusion on the roadmap).

## 5. Multi-model co-residency (the wave-4 promise) — staged, not measured

- **12B + 26B co-resident fits** (measured earlier at 132k; both 132k B=1 blobs
  ≈ weights 22.2 + 47.0 GiB + KV, well under 96 GiB). The concurrent-load MM1
  benchmark (`perf-data/bench_mm_ib.sh`, two model slugs, one `plowrt serve`
  process) is **written and staged** but was **cut by the operator stop before
  it ran**. No measured aggregate-tok/s / per-model-latency row exists yet;
  claiming one would be fabrication.
- **VMM prefix sharing** (rtx-09 V1, flag-gated **off** by default): probe- and
  campaign-validated to recover ~**10 GiB/sharer** of KV dedup at 31B/128k for
  common-prefix workloads (`vmm-prefix-v1.json`, attach-vs-copy table, TPOT-
  neutral, leak-clean). Not on the default serving path — measured here with
  defaults, so it does not enter the capacity rows above; it is the lever for
  the 31B/26B KV wall when enabled.

## 6. Reliability (unchanged from v1, re-verified)

- Zero-leak verdict (memcheck 0 errors; RSS/VRAM bit-flat over 228-request soak
  incl. cancels), panic-path KV release fixed, pinned-staging leak fixed.
- Tokenizer byte-fallback class made unbuildable + refused at startup.
- 3 latent OOB classes found and fixed (argmax ×2, flash sliding-window skip);
  device bounds-trap available (`PLOW_NV_KVBOUNDS`).
- Every perf row behind a correctness gate: Paris greedy + token-parity per
  model, 4-way (and 8-way where the blob is valid) concurrent zero-bleed
  isolation.

## 7. What's still open (staged, one lease each)

1. **Single-pass wide GEMV (`GV_MM_MAX≥16`)** — **DONE and measured**
   (campaign `WS-batched-gemv`, §4c, `ws-batched-gemv.md`). MM16 = 234 regs /
   0 spill, free at B=1; B=16 decode in one pass = **475 tok/s (+34% vs the
   2-pass MM8)**, served throughput up 17–34% at every VU, B=16 SLO-capacity
   1→2 users. Confirms the "~1.3× aggregate" `op_gemm.cuh` prediction. Ship
   `GV_MM_MAX=16` for B≥16 deployments. **This closes the decode-throughput
   half of the gap; it does not move max-users past 2** — prefill (item 2) is
   now the sole binding constraint on 12B SLO-capacity.
2. **Faster prefill** (TMA/FA4 flash) — TTFT is plow's other structural loss and
   blows the 5 s SLO by VU8 on any batch; independent of the decode path.
3. **26B B=8 concurrency sweep** — blob already gated; just needs the lease.
4. **31B / 26B concurrency head-to-head** + **12B 16k vLLM comparator**.
5. **MM1 co-resident load** — script staged (`bench_mm_ib.sh`).

All drivers are parameterized (`bench_b2_ib.sh`: `CAMPAIGN`/`PROMPT_TOKS`/`VUS`/
`DO_SWEEP`/`MODEL_NAME`/`TOKENIZER`/`ASSETS`) and the consolidator loader
(`consolidate_perf.py::do_b2_concurrency`) already ingests all three
`b2-concurrency-{12b,31b,26b}.json` files — dropping the missing JSONs in
lights up the remaining rows with no code change.

# Chunked-prefill FIXED overhead — measured decomposition + verdict (sm_120)

RTX PRO 6000 Blackwell (sm_120, 188 SMs, CUDA 13.0), 2026-07-23. Branch `beat-chunk-overhead`
(from `beat-vllm-consolidated`, post fp8-mma merge). Campaign: cut the per-chunk FIXED overhead
of chunked prefill — the "~44.5 ms constant + per-launch costs × 16 chunks @128k" hypothesis
behind the remaining 1.22×/1.36× TTFT gap to vLLM fp8kv.

**Headline: the hypothesis is dead on arrival — MEASURED. At 128k the ENTIRE host + launch +
sync + inter-chunk-gap budget is 2.8 ms of 6271 ms (0.04%). The 44.5 ms constant no longer
exists on this tree (it was fit on the pre-fp8mma ladder); the current 26B ladder is
TTFT ≈ 2 + 49.0·C ms (C in k-tok), constant ≈ 2 ms. The remaining vLLM gap is pure LINEAR
slope (49 vs ~39 ms/k-tok) — per-token compute inside the prefill kernels, not launch
boundaries.**

## 0. gpu-exec-stage1 audit (user-directed, done before building)

What the merge (de15f67, commits 4e191a6..cc86c3e) landed, and whether it touches per-chunk
prefill overhead:

| stage | what | prefill effect |
|-------|------|----------------|
| 1 (4e191a6) | async submission: one non-blocking engine stream, pinned StepStage, decode tick = enqueue-all + ONE stream sync, zero steady-state ctx syncs | none for prefill: `run_one_prefill_chunk` (exec/gpu.rs) still does per chunk sync `memcpy_htod` (inst window) + 3 sync uploads + sync memset + launch + `stream_synchronize` |
| 2/3 (f14f3d5) | dynamic B=1 KV row + per-slot tensor tables | removed per-chunk shared-table restore (already in) |
| 4 (ec4b555, 8d62ab3) | device sampler in serve path | decode only |
| 5 (4480ee7, 0cdcbf0) | bounded multi-step decode: K tokens per ONE host sync via `plow_advance` (device-owned pos/kvlen), `PLOW_MULTISTEP=K`, greedy, default off | decode only |
| 9 (325cd15, cc86c3e) | double-buffered pinned H2D checkpoint load; skip KV zeroing | load time only, not TTFT |

- Does anything already reduce per-chunk PREFILL overhead? **No.** And the published TTFT
  ladders come from the CUDA harness (`gemma4_sm120_chat`), which none of this touched — its
  chunk loop is heavier still (per-chunk malloc + full-instruction-stream sync `cudaMemcpy` +
  `cudaDeviceSynchronize`).
- Is multi-step decode reusable as the multi-chunk-per-launch template? **Yes, directly** —
  and prefill is the EASIER case: all chunks' args (ids/pos/kvlen/patch values) are known up
  front, so no device-side advance kernel is needed. Pinned staging + stream-ordered async
  H2D (stream order makes the in-place `d_inst` re-patch WAR-safe) + back-to-back cooperative
  launches + one sync. Implemented as `PLOW_CHUNK_ASYNC=1` (below). The measurement then says
  the prize was ~2 ms.

## 1. Measured decomposition — 26B fp8 (w8a8) mixed fp8-KV, 128k = 16×T=8192 chunks

`PLOW_CHUNK_PROF=1` (new, env-gated): per-chunk host timestamps + CUDA events. Packet
`PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 PLOW_MOE_PREFILL=1 PLOW_FP8=1 PLOW_W8A8=1 PLOW_FP8_KV=1
PLOW_FP8_KV_FULL=1`, ctx 132096; build `-DPLOW_NV_W8A8=ON -DPLOW_FP8_KV=ON
-DPLOW_FP8_KV_FASTPF=ON -DCMAKE_CUDA_FLAGS=-DPLOW_NV_FA_GF_FULL=4` (the fp8mma recipe;
reproduces the published ladder to ≤0.3 ms at every ctx).

Totals over all 16 chunks at 128k (sync baseline, wall 6271.4 ms):

| component | total | per chunk |
|-----------|------:|----------:|
| host patch (malloc+scan+patch, ~all sites) | 0.23 ms | 14 µs |
| inst stream H2D (full stream, sync) | 0.65 ms | 41 µs |
| ids/pos/kvlen H2D | 0.84 ms | 52 µs |
| counter + GQ-cursor memsets | 0.54 ms | 33 µs |
| launch enqueue | 0.43 ms | 27 µs |
| device idle gap between kernels (events) | 2.56 ms | 160 µs |
| first-token readback | 0.05 ms | — |
| **kernel wall** | **6268.6 ms** | **391.8 ms** |

(The host phases overlap the device gap — total non-kernel wall = 2.8 ms, 0.04%.)

Per-chunk kernel walls are FLAT: 390.5 (chunk 0, zero history) → 392.0 ms (chunk 15, 122k
history). The causal-flash history growth is invisible at 128k — the fp8mma campaign really
did delete the quadratic term.

## 2. The "44.5 ms fixed term" is gone

Single-chunk runs (one launch, prof on): 2k = 117.0 ms, 4k = 198.0, 8k = 390.3. Ladder fit on
this tree: slope 49.0 ms/k-tok everywhere (16→32: 49.0, 32→64: 49.0, 64→128: 49.0), constant
= 198.0 − 4×49.0 ≈ **2 ms**. The 44.5 ms a-term was fit on the PRE-fp8mma ladder
(gemma4-26b-fp8-prefill-sm120.json) and was absorbed/deleted by the mma+locality win. Of the
~2 ms, host+launch is 0.13 ms; the rest is small-bucket per-row inefficiency, not overhead.

## 3. Kernel-INTERNAL per-chunk fixed cost (the only real per-chunk multiplier left)

F = 2·kern(4096) − kern(8192) = 2×197.9 − 390.2 = **5.6 ms/chunk** (lm_head GEMV runs every
chunk ≈ 0.9 ms; GQ ramp/drain; per-op weight-stream cold starts). Caveat: per-row efficiency
is T-dependent (kern(2048) = 116.8 vs 101.8 predicted, +15 ms), so F is an upper bound of the
mergeable part; 8192→16384 per-row gain extrapolates to ≈ −0.5%/chunk. Perfect chunk-merge
ceiling at 128k ≈ 8–16 × 5.6 ≈ **45–90 ms (0.7–1.4%)**; 0 at ≤8k (single chunk).

## 4. Lever A/B — `PLOW_CHUNK_ASYNC=1` (shipped, default off)

The multi-step-decode analogue for prefill in the harness: pre-patch ALL chunks' instruction
streams + ids/pos/kvlen into one pinned slab, enqueue [inst H2D → ids/pos/kvlen H2D → memsets
→ cooperative launch] × N on stream 0, ONE `cudaDeviceSynchronize`. Device inter-chunk gaps
collapse 2.56 → 0.27 ms at 128k (17 µs/gap — pure back-to-back launch latency).

TTFT ladder, 26B fp8 mixed-KV (ms; single runs, reproduce published best-of-2 ≤0.3 ms):

| ctx | sync (default) | **PLOW_CHUNK_ASYNC=1** | Δ | published fp8mma | vLLM fp8kv | gap |
|-----|-------:|-------:|------:|-------:|-------:|-----:|
| 4k  | 198.0  | 198.1  | +0.1  | 196.9  | 134  | 1.48× |
| 16k | 782.0  | 781.9  | −0.1  | 782.5  | 526  | 1.49× |
| 32k | 1567.1 | 1566.5 | −0.6  | 1566.1 | 939  | 1.67× |
| 64k | 3135.4 | 3133.8 | −1.6  | 3135.3 | 2646 | 1.18× |
| 128k| 6271.4 | 6269.9 | −1.5  | 6271.1 | 5133 | 1.22× |

Exactly the measured ceiling: −0.02% at 128k, nothing at short ctx. **The gap to vLLM does
not move.**

## 5. Gates

- **Token identity (the required gate for host/launch changes): PASS** — first generated
  token AND the full 24-token greedy stream identical sync vs async at 128k
  (51406, 93914×23); first tokens + streams identical at 4k/16k/32k/64k (88124/238506/
  128499/176900 + continuations).
- **Default byte-identical:** both additions are env-gated (`PLOW_CHUNK_PROF`,
  `PLOW_CHUNK_ASYNC`, default 0); the default sync loop logic is untouched (only inert
  timestamp reads added). Unset-env runs reproduce the committed ladder: 6271.8 vs published
  6271.1.
- **Oracle:** not required — no kernel/emitter change anywhere in this campaign.

## 6. Honest NO-GOs (all measured, none built beyond the A/B)

- **Back-to-back / multi-chunk-per-launch in plowrt** (`run_one_prefill_chunk` async port):
  ceiling is the same 0.17 ms/chunk ⇒ ~2.8 ms at 128k, 0.13 ms at 4k. Not worth runtime
  churn; the serve path also deliberately returns per chunk to interleave decode ticks.
- **T=16384 bucket:** upside = F×8 ≈ 45–90 ms at 128k (0.7–1.4%), 0 at ≤16k. Cost: MAX_CHUNK
  and KV_RING are MIRRORED compile-time constants (dev_isa.h ↔ gemma4.rs) with the ring
  invariant `KV_RING ≥ window + MAX_CHUNK − 1` ⇒ ring doubles, activation arena doubles
  (2.04 → 4.1 GiB), packet grows, full oracle+parity re-run. Poor cost/benefit.
- **Patch-site reduction / per-chunk pre-baked programs:** total patch cost is 231 µs per
  128k PREFILL (not per chunk). Nothing to win.
- **lm_head skip on non-final chunks:** the biggest single identified item inside F
  (~0.9 ms/chunk ⇒ ~13.5 ms at 128k, 0.2%), but the wg-packets are pre-expanded in the
  stream — a host patch cannot cheaply no-op them; needs a kernel-side skip flag + oracle.
  Documented, not built at this payoff.
- **CHUNK-1 (cross-SM overlap):** not re-litigated per prior NO-GO.

## 7. What the remaining 1.22× actually is

49.0 vs ~39 ms/k-tok LINEAR slope = per-token compute in the chunk kernels (grouped-MoE GEMM
+ dense GEMM efficiency; flat 391.8 ms per 8192-token chunk, history-independent). The next
campaign must attack kernel throughput (MoE grouped-GEMM overlap/efficiency, sliding flash,
GEMM tile efficiency at T=8192), not launch structure. At 4k the story is identical: 198 ms =
one 4096-row chunk kernel at 49 ms/k-tok, host overhead 0.13 ms; vLLM's 134 ms is a faster
kernel, not a cheaper launch.

## Method / repro

- Harness: `runtime/tests/gemma4_sm120_chat.cu` + `PLOW_CHUNK_PROF=1` (per-chunk table),
  `PLOW_CHUNK_ASYNC=1` (one-sync arm). Build dir `build-chunk` (flags above).
- Packet: `target/release/gemma4` from this tree (env above), 26B ctx 132096, fp8 twins
  `PLOW_FP8_DIR=<ckpt>/fp8-full-plow`.
- Prompts: random-token ids, seed 42 (4k/16k/32k/64k/128k), 43 (8k), 44 (2k).
- Raw numbers: `perf-data/gemma4-chunk-overhead-sm120.json`.

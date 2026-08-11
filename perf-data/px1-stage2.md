# PX-1 stage 2 — block-diagonal varlen flash for batched prefill, Gemma-4-12B, sm_120

Campaign **PX1-s2**, 2026-07-21/22, branch `px1-varlen-flash` (based on
`px1-gemm-batching` @ 62082c1). Box: 1× RTX PRO 6000 Blackwell 96 GB (sm_120,
188 SMs). Per the design notes, PX-1, sequencing step 2 —
replace stage 1's per-request-SERIAL attention with a true varlen/cu_seqlens
block-diagonal flash: **all packed requests' prefill attention in ONE
persistent-grid kernel pass**.

Source of every perf number: `perf-data/px1-stage2.json` (transcribed verbatim
from the tool's reports in `perf-data/tools/b2-ib/px1s2b8-{off,s1,varlen}/`).
Harness: `huggingface/inference-benchmarker` rev `bad4f947` via
`perf-data/bench_b2_ib.sh`, 4000-token github_code prompts / 128 out, greedy,
streaming, 15 s warm + 120 s measure. TTFT includes server-side queueing.

## What was built

`runtime/nvidia/op_attention.cuh`: the PIPE=1 (shipped) `d_flash_prefill` takes
an optional `req` table (`[R, {q0, qlen, slot, kvlen}]`, host-patched t6 — the
cu_seqlens layout). In varlen mode the persistent work loop enumerates the
UNION of every request's (q_tile, head) items — request-major, head-fastest —
instead of stage 1's R sequential per-request passes. Query tiles are
PER-REQUEST (a tile never spans a request boundary; tail rows past `qlen`
stage zero and are dropped by the epilogue bound), and each item's K/V base is
slot-offset into the batch-major ring cache — so cross-request attention is
impossible **by construction** (block-diagonal causal on the 8 full hd512
layers; independent per-request windows on the 40 sliding hd256 layers), and
the per-item math is IDENTICAL to the stage-1 serial call. `req == nullptr`
keeps every legacy packet byte-identical; the PIPE=0 A/B control object keeps
the stage-1 serial loop. Host side (gpu.rs / mux.rs) is UNCHANGED from stage 1
— same ABI, same `PLOW_PF_BATCH=1` opt-in. Prefill object: **240 regs, 0
spill** (identical to stage 1).

## Correctness gates (before any perf number)

- **Kernel oracle (new, `runtime/tests/sm120_interp_op_test.cu`): PASS.**
  6 varlen packs — mid-tile request boundaries (qlen 33/47/100/17/9/…),
  shuffled KV slots, chunked `q_pos0`, sliding window — each **BIT-EXACT vs
  the stage-1 per-request-serial launches** and within flash tolerance of the
  T4-fixed per-request f32 reference (relL2 ≤ 2.0e-3). Full 134-test suite
  PASS.
- **Gate A — per-request token identity: PASS.** 5-prompt set (chunk-boundary
  crossers) concurrent-burst vs solo **byte-identical per request** on the
  varlen build; multi-request packed launches confirmed in the server log.
- **Gate B — cross-request bleed: PASS.** Poison/victim, both submission
  orders: victim byte-identical to solo (`4`); the concat sensitivity control
  flips to `PINEAPPLE`, so the test detects cross-request attention.
- Cross-check: serialized-solo vs batched-solo byte-identical on all 8 gate
  prompts. Harness: `perf-data/px1_gates.py` + `px1_run_gates.sh`, PORT 8093.

## VRAM deviation (forced): B=8 blob, not stage 1's B=16

A foreign long-running `plowrt serve` (pid 850160, port 8091, 12B ctx132k)
held ~33 GB for the whole campaign window and could not be touched. The B=16
ctx8k blob plans 66.6 GiB (42 GiB KV) and does not co-fit on 96 GB, so the
three arms ran the **12B ctx8k B=8 blob** (44.8 GiB measured) instead —
`gemma4-12b-ctx8k-b8.pkt`, decode cubin `gpu-assets-b4/interp_sm120.cubin`,
prefill cubin per arm rebuilt from source. The three arms are same-binary,
same-assets-shape, mutually comparable; comparisons against stage 1's
committed B=16 rows or vLLM's B=16-class numbers carry this caveat. A
contention sampler (`nvidia-smi pmon`, 1 s) ran through the campaign: only the
three arm servers burned SM — the windows are clean. (Earlier attempts were
poisoned — VU1 ITL flapped 22 ms → exactly 100.0 ms while the foreign server
served long-ctx traffic and another run's microbenches ran; those runs
were discarded and rerun.)

## Three-arm sweep (B=8 blob, 4k/128, ConstantVUs)

off = serialized prefill (`PLOW_PF_BATCH` unset); s1 = stage-1 batched GEMM +
serial attention; varlen = stage-2 block-diagonal varlen flash.

| VU | tok/s off/s1/varlen | ITL p99 off/s1/varlen (ms) | TTFT p99 off/s1/varlen (s) |
|---:|---:|---:|---:|
| 1  | 17.0 / 35.2 / **35.8** | 55.4 / 36.3 / **21.9** | 1.08 / 0.83 / **0.82** |
| 4  | 76.0 / 83.5 / **83.8** | 59.8 / 53.4 / 54.4 | 3.08 / 2.23 / **2.25** |
| 8  | 97.0 / 110.4 / **111.7** | 90.0 / 69.4 / **67.6** | 6.63 / 4.46 / **4.43** |
| 16 | 97.9 / **115.7** / 113.5 | 81.6 / 67.7 / **66.0** | 16.4 / 12.6 / 12.7 |
| 32 | 96.6 / 111.5 / **115.4** | 87.4 / **68.2** / 90.8 | 36.0 / 29.7 / **29.3** |

Zero failed requests everywhere. **SLO capacity (ITL p99 ≤ 50 ms AND TTFT p99
≤ 5 s): off = 0 VUs, s1 = 1 VU, varlen = 1 VU.** (The off arm's VU1 ITL p99
55.4 ms reflects a legacy-path decode-bucket drift — per-request medians
quantize at 22/34/56 ms; both batched arms hold a flat 21.9 ms. The stage-1
B=16 campaign showed the same off-path pattern, milder.)

## Verdict: end-to-end WASH vs stage 1 — and why (the honest negative)

**s1 → varlen is a wash at this profile**: +0.3..+3.5% tok/s at VU 1/4/8/32,
−1.9% at VU16 — all within run-to-run noise; ITL p99 −1.4..−14.4 ms at
VU 1/8/16, +22.6 ms at VU32 (one straggler window). The varlen kernel is
correct and never worse than serial where it matters, but it did NOT move SLO
capacity (1 VU either way) or saturated throughput.

Two measured reasons:

1. **The mux barely packs multi-request attention at this profile.** Pack
   stats from the campaign servers: s1 = 2132 packs of R=1 vs 227 of R=2;
   varlen = 2068 vs 240. ~90% of batched prefill launches carry ONE request
   (4k prompts fill the 2048-row interleave quantum alone), so the serial
   loop stage 2 removes almost never runs more than once per launch.
2. **Kernel-level, the win is real but layer-dependent**
   (`runtime/nvidia/experiments/fa_varlen_bench.cu`, 188×256, model shapes,
   2048-row pack, kvlen 4096): sliding hd256 varlen/serial speedup = 0.98×
   (R=1), **1.30× (R=2/4), 2.57× (R=8)** — the per-request partial-wave tails
   the plan predicted. The HBM-bound full hd512 flash however is 0.93–1.26×:
   interleaving requests' work items ACROSS the grid doubles the L2-resident
   KV working set, costing locality that the serial order got for free; at
   the 8192-row cold shape it is a 0.93–1.02× wash. The ~1–7% R=1 overhead in
   the standalone bench does not reproduce end-to-end (varlen ≥ s1 at 4/5 VU
   points).

**What this buys going forward:** the varlen kernel makes the pack-R axis
FREE for the scheduler on sliding layers (2.57× flash at R=8), so the next
lever is mux packing policy — admit more, smaller per-request chunks per tick
(e.g. 4×512 instead of 1×2048) so concurrent prefills actually share
launches; stage 2's kernel is the prerequisite that makes that policy change
attention-cost-neutral. A full-layer item ORDERING that keeps one request's
tiles L2-resident per grid wave (chunked rather than strided work
distribution) is the other follow-up.

## vLLM gap

The prize (vLLM 12B, same harness, B=16-class): 8 users / 239 tok/s. This
campaign's best: **115.7 tok/s** (s1, VU16) / **115.4** (varlen, VU32) on the
B=8 blob — not directly comparable to stage 1's 129–130 tok/s (B=16 blob,
16 KV slots) or to vLLM. On the same-B=8 arms the gap is ~2.1× at saturation;
stage 1's B=16 measurement (1.7–1.8× behind) remains the reference until the
B=16 rerun below.

## Honest negatives / limitations

- **Varlen did not improve SLO capacity or saturated throughput end-to-end**
  at the campaign profile; the bound is the shared prefill stall itself (a
  2048-row forward on every tick) plus the mux's R≈1 packing, not the serial
  attention loop stage 2 removed.
- **Full-layer (hd512) varlen is locality-negative** at large kvlen — a
  measured 0.93× at R≤2; block-diagonal fusion helps only the window-capped
  sliding layers at this context.
- **B=16 arms are missing**: blocked by a foreign 33 GB server for the whole
  window (the B=16 blob needs 66.6 GiB). The three-arm B=16 campaign
  (`perf-data/px1s2_campaign.sh`, assets `/root/gpu-assets-px1{,s2}`) should
  be run as-is when the card frees; expected to be the config where R>1 packs
  (up to 16 waiting slots) actually exercise the varlen path.
- The off arm's VU1/VU4 ITL is inflated by a legacy-path decode-bucket drift
  (22/34/56 ms per-request quantization) that both batched arms avoid; off's
  SLO capacity 0 should be read with that in mind (stage 1's B=16 off arm
  qualified at VU1).

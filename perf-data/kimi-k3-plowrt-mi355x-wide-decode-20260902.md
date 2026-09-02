# MI355X wide-decode GEMV sweep — 2026-09-02

This is a Plow-internal production-path A/B, not the matched vLLM baseline.
Kimi-K3 is the validation graph; the selected object policy depends only on
gfx950 decode width. MTP, speculative decoding, prefix caching, and FP8 KV were
disabled.

## Contract

- 8×MI355X, TP8, BF16 KV, MXFP4 weights.
- `plowrt bench` → `ModelMux` → `AmdServe`, 128 concurrent requests.
- Deterministic 2-token prompts, exact 16-token output, 128 requests, no warmup.
- Throughput-defer scheduling, GQ decode, L2 placement, MLA prefill V2.
- Same packet, checkpoint, scheduler, and row-parallel XArgmaxFin implementation.
- MM16 and MM8 ran twice in interleaved order. MM4 was a screening arm.
- Every run completed 128/128 with zero failures and checksum
  `fnv1a64:a4c7e3990608d2c6`.

## Results

| GEMV MM | rounds | duration (s) | output tok/s | median ITL (ms) | median TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 39.440, 39.374 | 51.927, 52.014 | 771.164, 771.408 | 1679.200, 1679.262 |
| 8 | 2 | 35.912, 35.763 | 57.028, 57.265 | 519.671, 520.770 | 1445.034, 1434.290 |
| 4 | 1 | 36.636 | 55.901 | 580.231 | 1494.936 |

MM8 vs MM16 means:

- output throughput: 51.970 → 57.147 tok/s, **+10.0%**;
- duration: 39.407 → 35.838 s, **−9.1%**;
- median ITL: 771.286 → 520.221 ms, **−32.6%**;
- median TPOT: 1679.231 → 1439.662 ms, **−14.3%**.

The object metadata explains why MM4 was worth screening but not selecting.
The K3 GQ object reports 32 spilled VGPRs at MM16/MM8 and 6 at MM4, but MM4's
extra weight walks cost more than the reduced spill. MM8 is the measured balance.
The CMake default therefore uses MM8 only when the compiled gfx950 decode
capacity exceeds B32. Narrow objects keep their prior derived widths. An explicit
`PLOW_GEMV_MM` CMake cache value remains available for controlled tuning.

A final run through the asset's default HSACO path, after rebuilding it with
the selected policy, completed 128/128 at 57.471 output tok/s, 519.260 ms median
ITL, and the same checksum. This verifies that the default path matches the
explicit MM8 arm rather than only proving an override directory.

The row-parallel XArgmaxFin change is correctness-covered here but not separately
attributed: all three arms include it. It replaces the B×nparts lane-0 fold with
independent row workers, retains the serial B1 path, and performs an acquire on
each worker before peer-value loads. A long all-rank agreement campaign remains
required before claiming its isolated speedup or closure of the historical rare
peer-coherence failure.

## Relation to vLLM

No current Plow-vLLM pair is workload-matched. The pinned vLLM reference is
8192→1024: C1 TPOT 20.768 ms and C128 output throughput 1133.93 tok/s. This sweep
uses 2→16 to isolate wide decode and must not be divided into those numbers.
The current 64→2 Plow results are also correctness smoke tests, not a substitute
for the pending 8192→1024 C1/C32/C64/C128 production gate.

## Remaining performance levers

1. Cross-request packed prefill. AMD currently runs one request per prefill
   packet; the packet ABI lacks row owner, KV base, position, length, and
   recurrence-boundary metadata.
2. Wide grouped MoE. Route/align/GLU/down/combine remains less fused than the
   selected vLLM path and historically dominated wide-decode stragglers.
3. Prefill KDA scan. Serial-token recurrence was 43.1% of the historical T8192
   prefill; a parallel scan needs a numerical and full-network gate.
4. Regenerate the stale gfx950 tuning database and rerun fused-kernel → block →
   full-network gates.
5. Separate prefill/decode scratch before attempting actual phase overlap.
6. Run the exact 8192→1024 comparison. Host submission and TP segment-drain
   removal are deprioritized: prior attribution puts decode host work near 3%,
   and prefill segment barriers are required for collective progress.

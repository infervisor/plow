# 26B-A4B router-hoist expert PREFETCH (decode) — L2-prefetch microbench — **NO-GO**

RTX PRO 6000 Blackwell (sm_120, 188 SMs, 128 MiB L2), 2026-07-23. Branch `beat-moe-prefetch`.
Mandated methodology step: single-kernel GO/NO-GO microbench BEFORE any opcode/emitter work.
Verdict is **NO-GO — the prefetch packet was not built**. This file documents the negative.

## The hypothesis

The P9 router hoist emits the MoE router (score+topk) *before* the dense-MLP packets, so the
8 routed experts' IDs are known one dense-MLP-duration (~50–80 µs) before the expert-GLU GEMV
reads their gate_up weights. A `prefetch.global.L2` leaf packet fired in that window would let
the GLU hit L2 instead of HBM — a window vLLM structurally lacks. Honest physics risk, stated
up front in the plan: decode is globally HBM-BW-bound, so prefetch may only *move* the bytes,
stealing window bandwidth 1:1 for zero (or negative) net.

## Harness

`runtime/tests/moe_prefetch_bw_sm120.cu`. Real decode geometry H2816 / E128 / k8 / I_moe704,
nrow=1, exact megakernel op bodies (`d_moe_expert_glu_gemma[_fp8]`, `d_moe_expert_down_gemma[_fp8]`),
megakernel launch shape (grid 188, 256 thr). cudaEvent, 30 timed iters after 5 warm; numbers
reproduced across runs to ±0.5 µs.

- Prefetch = `prefetch.global.L2` (and `::evict_last` variant), 128B lines, block-strided,
  **gate_up rows of the 8 routed experts only** (60.5 MiB bf16 / 30.2 MiB fp8 vs 128 MiB L2).
- Window model = warp-per-row bf16 dense GEMV streaming ~100 MiB (bf16 cfg) / ~50 MiB (fp8 cfg,
  fp8 dense weights halve the real window) at 1.07–1.23 TB/s.
- Fused variant: same grid, blocks `[0,pfb)` issue their prefetch slice first then join the
  dense sweep — dense work total unchanged, models the hoisted leaf packet running co-resident.
- L2 hygiene (this bench's hard-won part): 384 MiB memset + 384 MiB read-stream flush per iter,
  **and the routed expert set rotates through 8 disjoint sets per iteration** (as in real decode,
  where routing changes every token). Without the rotation, Blackwell's L2 streaming-insertion
  heuristics let the previous iteration's expert lines survive a read-only flush and silently
  pre-warm the "cold" baseline (first bench version showed glu_after_dense ≈ glu_warm).

## Results (ms, mean of 30; run 2 of 2 — run 1 within ±0.0005)

Baselines:

| cfg | glu_cold | glu_warm | max win | down_cold | moe step (glu+down) | dense_base | dense GB/s | glu_after_dense (op baseline) |
|-----|---------:|---------:|--------:|----------:|--------------------:|-----------:|-----------:|------------------------------:|
| bf16 | 0.0525 | 0.0164 | 0.0362 | 0.0273 | 0.0798 | 0.0800 | 1221 | 0.0473 |
| fp8  | 0.0343 | 0.0160 | 0.0184 | 0.0221 | 0.0565 | 0.0457 | 1068 | 0.0285 |

Idle-window control (prefetch alone, nothing competing, then GLU) — the *ceiling*:

| cfg | mode | pf_ms | glu_after | recovered | % of max win |
|-----|------|------:|----------:|----------:|-------------:|
| bf16 | plain      | 0.0204 | 0.0345 | 0.0181 | 50% |
| bf16 | evict_last | 0.0207 | 0.0344 | 0.0181 | 50% |
| fp8  | plain      | 0.0134 | 0.0239 | 0.0104 | 57% |
| fp8  | evict_last | 0.0140 | 0.0233 | 0.0111 | 60% |

Fused dense‖prefetch (the actual proposal), net = glu_gain − dense_slowdown:

| cfg | variant | dense_ms | d_slow | glu_ms | glu_gain | net_ms | net/moe-step |
|-----|---------|---------:|-------:|-------:|---------:|-------:|-------------:|
| bf16 | plain pfb=4    | 0.1562 | 0.0762 | 0.0207 | 0.0266 | −0.0496 | −62.2% |
| bf16 | plain pfb=16   | 0.1123 | 0.0323 | 0.0336 | 0.0137 | −0.0186 | −23.3% |
| bf16 | plain pfb=64   | 0.0973 | 0.0173 | 0.0405 | 0.0068 | −0.0105 | −13.1% |
| bf16 | plain pfb=188  | 0.1000 | 0.0200 | 0.0426 | 0.0047 | −0.0153 | −19.2% |
| bf16 | evlast pfb=4   | 0.1559 | 0.0759 | 0.0201 | 0.0273 | −0.0486 | −60.9% |
| bf16 | evlast pfb=16  | 0.1120 | 0.0320 | 0.0303 | 0.0170 | −0.0150 | −18.8% |
| bf16 | evlast pfb=64  | 0.0973 | 0.0173 | 0.0332 | 0.0141 | −0.0032 | −4.1% |
| bf16 | evlast pfb=188 | 0.0998 | 0.0199 | 0.0339 | 0.0134 | −0.0064 | −8.0% |
| fp8  | plain pfb=4    | 0.0865 | 0.0408 | 0.0160 | 0.0125 | −0.0282 | −50.0% |
| fp8  | plain pfb=16   | 0.0636 | 0.0179 | 0.0175 | 0.0109 | −0.0069 | −12.3% |
| fp8  | plain pfb=64   | 0.0572 | 0.0115 | 0.0180 | 0.0105 | −0.0010 | −1.7% |
| fp8  | plain pfb=188  | 0.0576 | 0.0119 | 0.0197 | 0.0088 | −0.0031 | −5.5% |
| fp8  | evlast pfb=4   | 0.0868 | 0.0411 | 0.0160 | 0.0125 | −0.0286 | −50.6% |
| fp8  | evlast pfb=16  | 0.0637 | 0.0179 | 0.0175 | 0.0109 | −0.0070 | −12.4% |
| fp8  | evlast pfb=64  | 0.0571 | 0.0114 | 0.0180 | 0.0105 | **−0.0009** | **−1.7%** |
| fp8  | evlast pfb=188 | 0.0577 | 0.0119 | 0.0198 | 0.0087 | −0.0033 | −5.8% |

(`glu_gain` is vs `glu_after_dense`, the in-sequence cold baseline; `glu_cold` right after the
flush is ~5 µs inflated by dirty-L2 writeback drain and is not used for gain accounting.)

## Verdict: NO-GO — net negative in all 16 configurations, on both physics counts

GO required net > ~1% of the MoE step AND dense slowdown ≈ noise. Measured: **best case is
−1.7% of the MoE step (fp8, pfb=64), worst −62%**, and the dense slowdown (11–76 µs) is far
above noise (±0.5 µs) in every configuration. Two independent losses stack:

1. **The window has no idle bandwidth.** The dense GEMV streams at 1.1–1.2 TB/s; the prefetch's
   30–60 MiB must come out of the same HBM pipe, so the window pays ≥ the bytes it prefetches
   (d_slow ≥ glu_gain in all 16 rows). This is exactly the "prefetch only moves the bytes"
   failure mode the plan flagged.
2. **The prefetch decays before it is consumed.** Even with a *completely idle* window the GLU
   recovers only 50–60% of the warm-vs-cold headroom (fire-and-forget requests are dropped under
   MSHR pressure; the following dense stream evicts more — `evict_last` recovers some of that for
   bf16, 14.1 µs vs 6.8 µs at pfb=64, but never enough to go positive).

Upper bound on the prize, even granting a free window: ~11–18 µs per MoE layer vs a measured
window cost of the same order or larger. There is no tuning direction left inside this design —
fewer prefetch blocks lower the window tax but starve the prefetch (pfb=4 lands most, costs
most); more blocks (188) delay dense start for less landed data. The knob sweep brackets the
optimum and it is negative everywhere.

**Consequences:** no `MoeExpertPrefetch` opcode, no emitter packet, no `PLOW_MOE_PREFETCH` flag;
shipped kernels/blob untouched (this campaign adds only the test + this doc). Revisit only if
decode gains a genuinely BW-idle window ≥ the expert working set in duration (e.g. a future
compute-bound dense phase), or if experts move to a slower tier (NVMe/host-paged experts —
there prefetch is not an L2 trick but an actual transfer overlap, different mechanism).

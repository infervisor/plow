# fp8 mma for the hd256 SLIDING-window prefill flash — beat-sliding-fp8mma (sm_120)

RTX PRO 6000 Blackwell (sm_120, 188 SMs, CUDA 13.0), 2026-07-23. Branch `beat-sliding-fp8mma`
(from `beat-vllm-consolidated`). Goal: extend the shipped fp8 QK^T mma prefill win
(`gemma4-fp8-mma-prefill-sm120.md`, hd512 FULL arm, 2× compute) to the hd256 sliding-window
prefill flash — 25/30 sliding layers on 26B, 40/48 on 12B.

## VERDICT: NO-GO by ceiling (for the fp8-mma lever). Stopped before kernel work.

The shipped fp8-mma win rests on the hd512 FULL arm being **COMPUTE-bound** (px4 budget: QK 35%,
softmax 35%, P.V 18% of a compute side that binds the wall). The hd256 sliding arm at window 1024
is the opposite: **MEMORY/latency-bound**. fp8 QK mma speeds only the QK compute, which is HIDDEN
here → it buys ≈0. Measured, not assumed (the exact risk the task flagged).

## 1. Measured sliding-arm budget (`runtime/tests/flashpre_slide_bw_sm120.cu`)

Current bf16 sliding kernel `d_flash_prefill<256,64,32>` (PIPE=1 generic body), Gemma-4 sliding
geometry (16 q-heads / 8 kv-heads, hd256, window 1024), one 8192-query chunk at the tail (full
1024 windows), persistent 188-block grid, best of 20:

| ctx | ms/chunk-layer | effBW GB/s | bf16 compute floor (QK+PV) | bound |
|-----|---------------:|-----------:|---------------------------:|-------|
| 8k  | 1.284 | 1778 | 0.296 ms | MEMORY |
| 32k | 1.343 | 1699 | 0.296 ms | MEMORY |
| 128k| 1.342 | 1701 | 0.296 ms | MEMORY |

- ctx-FLAT (window-bounded, as designed) → total sliding cost is LINEAR in ctx.
- Wall 1.34 ms is **4.5× the full bf16 QK+PV compute floor (0.296 ms)**. Even crediting softmax
  (px4: ~equal to QK+PV, floor not counted) puts full compute ~0.56 ms — still <45% of the wall.
- effBW ~1700 GB/s (HBM-class, well UNDER the ~5 TB/s L2 roofline the window reuse should hit) →
  the arm is **latency/occupancy-bound**, not even bandwidth-saturated. Few tiles/query (window
  1024 / BKV 32 ≈ 34), low ILP: adding faster mma changes nothing on the wall.

## 2. Share of prefill TTFT (identical hd256/window geometry on 12B and 26B)

Both models: 16 q / 8 kv heads, hd256, window 1024 → 1.34 ms/chunk-layer applies to both. Total
sliding-flash = 1.34 ms × (ctx / 8192) × n_sliding_layers. TTFT anchors from the shipped
fp8-mma ladder (mixed-KV).

| model | n_slide | ctx | sliding-flash total | plow TTFT | **share** |
|-------|--------:|-----|--------------------:|----------:|----------:|
| 26B | 25 | 32k  | 134 ms | 1566 ms | 8.6% |
| 26B | 25 | 128k | 537 ms | 6271 ms | 8.6% |
| 12B | 40 | 32k  | 214 ms | 2627 ms | 8.2% |
| 12B | 40 | 128k | 858 ms | 10990 ms | 7.8% |

Sliding flash is a consistent **~8-9% of prefill TTFT** — a real slice, but it is memory-bound.

## 3. Ceiling math (stated as the gate requires)

- **If sliding flash HALVED** (ideal, unreachable by any single lever): 26B@128k −268 ms = **−4.3%
  TTFT**; 12B@128k −429 ms = **−3.9%**. Just at the 3-4% NO-GO floor even in the impossible best case.
- **fp8 QK mma (the campaign's lever)**: touches only QK compute (~0.10-0.20 ms of the 1.34 ms
  wall), which is hidden under the memory/latency side. Best case if fully exposed: 40 ms @26B /
  64 ms @12B = **≈0.6% TTFT**. Realistically **≈0**. → **NO-GO.**
- **fp8 KV cache (e4m3, half the K+V bytes)** is the only lever on the binding side. Perfectly
  bandwidth-bound it would give ~1.67× on the arm → −3.5% (26B) / −3.2% (12B). But the arm runs at
  only 1700 GB/s (latency-bound, not bandwidth-saturated), so byte-halving yields LESS than that.
  AND it forces the DECODE sliding arm onto e4m3 (the ring is read at decode too) for ~0 decode
  gain (window-1024 ring is tiny). Marginal, at/below the go-bar, and a DIFFERENT campaign than
  the fp8-mma one asked for.

This confirms the shipped doc's own note (§3): "Sliding-layer ring caches are window-bounded
(tiny) — fp8 buys them nothing."

## 4. What would actually move the linear front (out of scope here)

The remaining 128k gap to vLLM fp8kv (26B 1.22×, 12B 1.36×) is the LINEAR front. Sliding flash is
~8% of it and memory-bound. The larger, compute/overlap-shaped levers the shipped doc names —
MoE grouped-GEMM overlap, per-chunk launch/counter overhead — are where the TTFT is.

## Reproduce

```
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
  -I runtime/common -I runtime/nvidia -include cstdint \
  runtime/tests/flashpre_slide_bw_sm120.cu -o slide_bw
GPU_LEASE_TIMEOUT=7200 GPU_LEASE_IDLE_MIB=35000 gpulease <tag> ./slide_bw 20 8192
```

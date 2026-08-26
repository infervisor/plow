# Kernel-by-kernel prefill sweep — Gemma-4-12B / RTX 5090 (sm_120a), bf16

Follow-on to `perf-data/prefill-bf16-gap-attribution-2026-08-26.md`, same session. That report
found plow's bf16 GEMM at 66-71% of cuBLASLt's throughput on identical shapes. Per user direction,
this report completes the sweep across every op body in the bf16 prefill path and scopes what's
actually worth optimizing.

## The sweep is smaller than it sounds

Walking `runtime/nvidia/interp_sm120.cu`'s dispatch table for a bf16, dense (non-MoE/MLA/DSA)
Gemma-4-12B build, the prefill path touches: `d_gemm`/`d_gemm_glu` (GEMM), `d_flash_prefill`
(attention), and five cheap ops — `d_rmsnorm`, `d_norm_residual`, `d_headnorm_rope`, `d_softcap`,
`d_embed`, `d_residual`, `d_argmax`. This repo's own docs (w8a8/fp8 analogs on other hardware — no
bf16 split exists) put GEMM+flash-attention at roughly 74-85%+ of prefill wall-clock combined, and
separately document the norm-family ops directly as sub-1%-each ("fusing the argmax cannot move
the intercept," `perf-data/rtx19-e5-lmhead.md:26-30`). **Only two ops are worth benchmarking: GEMM
(done) and flash-attention (this report).**

## Flash-attention-prefill: isolated bf16 throughput, no ncu needed

Standalone microbench (same technique as the GEMM one — a `.cu` file reusing
`runtime/nvidia/op_attention.cuh`'s real `d_flash_prefill<HD,BQ,BKV>` body directly, timed with
CUDA events), at the exact shipped tiling and the real Gemma-4-12B shapes (16 heads; hd256 sliding
layers, `kvh=8`, window=1024, from the checkpoint's own `config.json`; hd512 full-attention
layers, `kvh=1`, causal only), seq=8192, full-grid (170 SMs).

| arm | layers using it | ms | TFLOP/s (masked-aware) | % of bf16 mma ceiling (259.2 TFLOP/s, PX-9) |
|---|---|---|---|---|
| hd256 sliding (BQ64/BKV32, shipped) | 40 of 48 | 1.4587 | 88.3 | **34.1%** |
| hd512 full (BQ32/BKV16, shipped) | 8 of 48 | 9.6732 | 113.7 | **43.9%** |

TFLOP/s is computed against the *actual* causal(+sliding-window) work done — the causal triangle
for hd512, the causal-and-in-window band for hd256 — not naive dense `seq²`, since the kernel
correctly skips fully-masked KV tiles and a dense denominator would understate its efficiency.

**No live cross-engine comparison was possible in this sandbox**: `flash-attn` (FlashAttention-2)
has no prebuilt wheel for this venv's torch 2.13.0+cu130, and a from-source build (typically
20-45+ minutes, uncertain sm_120a support since FA2 targets Ampere/Hopper `mma.sync`/`wgmma`
generations and sm_120a's compute-capability listing isn't a standard FA2 build target) wasn't
attempted given the time cost — reported honestly rather than substituting a guess or a number
from a different GPU.

**Reading**: 34-44% of the raw hardware mma ceiling is a bigger relative gap than GEMM's 66-71%
of cuBLASLt (a different reference point — library-vs-library for GEMM, kernel-vs-raw-hardware for
attention, not directly ratio-comparable — but both readings point the same direction: real,
substantial headroom). Flash-attention's inherently harder dependency chain (running softmax
statistics, row-max/row-sum bookkeeping between KV tiles, P.V using the just-computed
probabilities) makes it plausible this is harder to pipeline well than a "pure" GEMM, consistent
with it looking further from its ceiling.

## Verdict — both GEMM and flash-attention have real, confirmed headroom

The "copy code from cuBLAS/vLLM directly" framing was reconsidered this session (see
`/root/.claude/plans/zazzy-skipping-hellman.md` for the full research and the user's explicit
confirmation): plow has zero cuBLAS/CUTLASS/FlashAttention linkage in production, and this repo's
own prior experiment with many separate per-op kernel launches (`SegPf` segmented mode) lost ~60%
of one op-class's time to pure per-launch floor overhead — a documented, repo-native precedent
against literally linking external libraries as separate host-launched kernels for a per-layer
granularity. **Confirmed path forward: port cuBLAS/CUTLASS's and FlashAttention-2's techniques
(TMA operand staging, software pipelining) into plow's own inlined `d_gemm`/`d_gemm_glu`/
`d_flash_prefill` bodies — same file, same persistent kernel, same fusion — not external library
calls.**

## What's next — scoped, not started (per explicit user direction this session)

1. **GEMM TMA mainloop port** — headroom proven (66-71% of cuBLASLt), technique confirmed
   reachable on sm_120a (`perf-data/px9-gemm-body.md` §Result 7: single-CTA TMA exists here even
   without `wgmma`). Target: `op_gemm.cuh`'s `d_gemm`/`d_gemm_glu`.
2. **Flash-attention pipelining/staging port** — headroom now also confirmed (34-44% of ceiling).
   FlashAttention-2's Ampere-class tiling/pipelining ideas (not its Hopper-only
   warp-specialization/TMA tricks — sm_120a lacks TMEM) are the relevant reference to study.
   Target: `op_attention.cuh`'s `d_flash_prefill`.
3. Both are genuine new kernel-body work, multi-session scope, no existing sm_120a precedent for
   either technique in this tree, and carry real correctness risk (this repo has a documented
   history of "fluent but wrong" regressions from similarly-scoped changes). Neither was started
   this session. If greenlit, follow the full gate sequence already used throughout this campaign:
   numeric oracle → register/spill diff on the real production object → exact-output-match smoke
   test → GSM8K at N=200 minimum → live TTFT bench against the established vLLM baseline — one op
   at a time, correctness-gated before the other starts.

Microbench sources promoted into the tree: `runtime/bench/nvidia/bf16_gemm_vs_cublas_bench.cu`
(GEMM vs cuBLASLt) and `runtime/bench/nvidia/fa_prefill_bench.cu` (isolated flash-attention
throughput). Neither is wired into `CMakeLists.txt` yet (built by hand this session, direct
`nvcc` invocations in each file's header comment) — add build targets if this line of work
continues.

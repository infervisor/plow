# 26B-A4B grouped-MoE PREFILL — single-block GEMM microbench (ops 75/76) — beat26b-prefill

RTX PRO 6000 Blackwell (sm_120, 188 SMs), 2026-07-22. Branch `beat26b-prefill`.
Mandated methodology step: measure the grouped GLU (op 75) / DOWN (op 76) GEMMs **in isolation**
BEFORE any full-model sweep, pick the tile/scheduler winner, then integrate + run the TTFT ladder.

Harness: `runtime/tests/moe_group_bench.cu`. Real geometry H2816 / E128 / k8 / I_moe704.
Launch = the megakernel config (grid 188, 256 thr, block-stride flat (expert,m-tile)×n-tile work
list). cudaEvent, 50 timed iters after 10 warm. fp8 = native w8a8 (both operands e4m3,
`mma.sync.m16n8k32`, BK8=64, `pgm_sw8` swizzle — reuses the 08a2bdd dense mainloop helpers).
fp8 relL2 is vs the bf16 baseline output (the inherent w8a8 weight+activation quant error).

## Results — DEFAULT tile (BM128 / BN128 / BK32 / GLU_STAGES2 / STAGES3), ms per op

| T (chunk) | GLU bf16 | GLU fp8 | GLU × | DOWN bf16 | DOWN fp8 | DOWN × | GLU relL2 | DOWN relL2 |
|-----------|---------:|--------:|------:|----------:|---------:|-------:|----------:|-----------:|
| 512  | 1.251 | 0.619 | **2.02×** | 0.570 | 0.374 | **1.52×** | 5.2e-2 | 3.6e-2 |
| 2048 | 1.374 | 0.684 | **2.01×** | 0.720 | 0.516 | **1.39×** | 5.2e-2 | 3.6e-2 |
| 8192 | 2.699 | 1.612 | **1.67×** | 1.908 | 1.512 | **1.26×** | 5.2e-2 | 3.6e-2 |

GLU peaks ~322 TFLOP/s fp8 (T=8192) vs ~193 bf16. DOWN is smaller-K (K=I_moe=704) so it tops out
lower and the fp8 win compresses at large T (the DOWN A-tile — contiguous `fu` — re-reads across
H/BN=22 n-tiles, so DOWN becomes A-bandwidth-bound before it saturates the fp8 tensor cores).

## Tile / scheduler A/B (the mandated sweep)

| variant | change | verdict |
|---------|--------|---------|
| **default** | BN128, GLU2/ST3 | **WINNER** — best bf16 and fp8 across all T |
| bn64 | BN=64 | **LOSS** — DOWN 35% slower @T8192 (2.59 vs 1.91 ms): 2× the n-tiles ⇒ 2× the contiguous-A re-read; GLU ~flat |
| glu3 | GLU_STAGES=3 | noise-identical to default, +50% GLU arena; not worth |

**bf16 verdict: the grouped GEMM is already at its tile-param optimum.** No `-D` tile/scheduler
knob beats the shipped design. "Activation-stationary reuse" is not expressible at this register
budget (keeping A resident across all n-tiles needs tiles_n×MFRAG×NFRAG×4 accumulators — infeasible),
and larger N-tiles (BN64→ more tiles) make the A re-read *worse*, not better. So the residual bf16
TTFT gap vs vLLM is NOT a tile-tuning problem — it is the untuned-grouped-vs-autotuned-CUTLASS-MoE
gap plus the FLASH_PREFILL quadratic (per the P9 report the ratio worsens with ctx, tracking flash,
not the MoE GEMM).

## fp8 verdict: implement it (the real lever, the softest target)

w8a8 grouped MoE gives **1.67–2.02× on GLU** and **1.26–1.52× on DOWN** at the kernel level, and
vLLM's fp8 MoE prefill is untuned Triton (equal-or-worse than its own bf16). fp8 grouped MoE
prefill is currently NOT IMPLEMENTED in plow → integrating the validated w8a8 grouped kernels is
the path to the fp8 TTFT column. (End-to-end fp8 win is diluted by the dtype-invariant
flash/router/norm/combine surface, largest at short-mid ctx where the FFN GEMMs dominate.)

## Correctness note
The fp8 relL2 (5.2e-2 GLU / 3.6e-2 DOWN) is the per-GEMM w8a8 quant error, same magnitude class as
the shipped dense 31B w8a8 prefill. End-to-end greedy coherence + the oracle grouped-op gate are
verified in the integration step (not this isolated GEMM bench).

# GEMM MFMA-shape / accumulator-chain-count probe — VERDICT: **NO-GO**

Branch `gemm-mfma-shape`, MI350X / gfx950, ROCm 7.0.2. Follows `GEMM_OCC1_POC_VERDICT.md`.
Tests the one hypothesis the occ-1 PoC surfaced: plow's prefill GEMM is capped by the number of
**independent MFMA accumulator chains in flight per SIMD** (the MFMA→MFMA RAW chain), and
hipBLASLt reaches ~2× util because its smaller **16×16** MFMA has a 4-VGPR accumulator (vs plow's
32×32 = 16-VGPR), fitting ~4× more independent accumulator tiles in the same register budget.

**Conclusion: matching the library's MFMA shape (16×16×32) AND its accumulator-chain count
(24 / 32 / 40 chains/wave) in a plow-style HIP kernel does NOT lift MfmaUtil.** It lands at
18–30% — identical to plow's existing 32×32 kernels, ~2× below hipBLASLt. MFMA shape and chain
count are **not** the lever. The library's 2× advantage is its Tensile hand-scheduled *assembly
software pipeline*, which the HIP/LLVM compiler does not generate from C++ source regardless of
MFMA shape. plow's prefill GEMM is at the achievable ceiling for a HIP-source megakernel.

## Step 1 — hipBLASLt IS util-bound, not clock-bound (measured)

`scratchpad_bench/bench_lt_prof` (screens all algos, runs only the winner N× so the rocprof CSV is
one clean kernel). M=4096, Qwen3-4B bf16 TN shapes, whole GPU. Un-profiled TF/s + rocprofv3
`--pmc MfmaUtil MemUnitStalled` (values under rocprof are depressed but internally consistent — every
row below shares identical profiler conditions):

| shape | hipBLASLt kernel (winner) | TF/s (unprof) | **MfmaUtil** | MemStall | VGPR | AGPR | WG |
|---|---|---|---|---|---|---|---|
| q_proj  | `Custom …MT256x256x64_MI16x16x1…`      | 1619 (98% peak) | **57.6%** | 0.0 | 252 | 0 | 256 |
| o_proj  | `…MT256x160x64_MI16x16x1…MIWT8_5`      | 1425 | **51.7%** | 0.0 | 208 | 0 | 256 |
| gate/up | `Custom …MT256x256x64_MI16x16x1…`      | 718  | **55.3%** | 0.1 | 252 | 0 | 256 |
| down    | `…MT256x160x64_MI16x16x1…MIWT8_5`      | 766  | **62.5%** | 0.0 | 208 | 0 | 256 |

Win is **utilization, not clock**: MfmaUtil differs 2.4× (57.6 vs 24.0 for q) and TF/s differs ~2.9×,
both in the same direction. If it were a clock win, MfmaUtil would match and only TF/s would move.
Peak used = 1660 TF/s SUSTAINED (already the power-limited whole-GPU clock), so both library and plow
run the same clock regime. bf16 sustained peak ≈ 1450 = ~64% of the 2265 theoretical.

## Step 2 — the concrete structural difference (disassembly)

Unbundled `libhipblaslt`'s bf16 TN code object
(`TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_…Alik_Bljk…gfx950.co`) via
`clang-offload-bundler -unbundle` and disassembled with `llvm-objdump -d`. Opcode census of the
whole library: **26170× `v_mfma_f32_16x16x32_bf16`**, 15× `32x32x16` (one edge kernel). It uses the
16×16 MFMA essentially everywhere. (Raw evidence: `mfma_shape_results/library_disasm.txt`.)

|  | plow `gemm_c5` (prod) | hipBLASLt (o_proj/down) |
|---|---|---|
| MFMA instruction | `v_mfma_f32_32x32x16_bf16` | `v_mfma_f32_16x16x32_bf16` |
| output tile | 32×32 | 16×16 |
| accumulator VGPRs / tile | 16 (`f32x16`) | **4 (`f32x4`)** |
| independent accumulator tiles / wave | 12 | **40** (`MIWT8_5`) |
| independent MFMA→MFMA chains in flight | 12 | **40** |
| accumulator register budget | 192 | 160 |

The library issues a long run of MFMAs each writing a **distinct** 4-register accumulator
(`a[0:3], a[4:7], a[8:11] … a[60:63] …` back-to-back, no MFMA consuming the previous one's result)
→ ~40 independent RAW chains resident to hide the fixed MFMA latency, in the **same ~160-VGPR
accumulator budget** plow spends on 12. This is exactly the hypothesised mechanism.

## Step 3 — the microbench refutes the portable version of the hypothesis

`gemm_mfma16_poc.hip` — a plow-style double-buffered GEMM rewritten on `v_mfma_f32_16x16x32_bf16`
with `f32x4` accumulators, sweeping the independent-chain count (SM·SN) at occ-2 (the production
256-VGPR / 8-wave budget) and at the library's 4-wave/40-chain layout. Correctness spot-checked vs
CPU dot (rel ≤ 0.004 all configs). MfmaUtil via the same rocprof pass, **same session** as the two
plow production kernels — a clean apples-to-apples 3-way:

| kernel | MFMA | chains/wave | MfmaUtil q · gate · down |
|---|---|---|---|
| **hipBLASLt** | 16×16×32 | ~40 | **57.6 · 55.3 · 62.5** |
| plow `gemm_c5` 192×256 (prod, tuned ping-pong) | 32×32×16 | 12 | 24.0 · 32.2 · 36.7 |
| plow `gemm_c0` 256×256 (prod) | 32×32×16 | 16 | 22.8 · 21.5 · 22.7 |
| m16 128×256 o2 (8w) | 16×16×32 | 16 | 23.5 · 26.1 · 19.7 |
| m16 192×256 o2 (8w) | 16×16×32 | 24 | 19.7 · 26.7 · 29.6 |
| m16 256×256 o2 (8w) | 16×16×32 | 32 | 25.3 · 26.4 · 23.0 |
| m16 256×160 w4 (library tile/40-chain) | 16×16×32 | 40 | 17.8 · 22.9 · 26.1 |
| m16 192×256 w4 | 16×16×32 | 48 | 16.8 · 21.9 · 25.1 |
| m16 256×256 w4 | 16×16×32 | 64 | 22.8 · 22.4 · 20.3 |

**The sweep is flat.** Going 16 → 24 → 32 → 40 → 48 → 64 independent 16×16 chains does not move
MfmaUtil off ~26%. Matching the library's exact MFMA shape *and* tile *and* chain count (256×160,
40 chains) gives 18–26% — no better than, and often below, plow's own 32×32 kernels. MemUnitStalled
≈ 0 and LDS conflicts are not the issue. The 16×16 microbench does not even beat plow's tuned 32×32
`gemm_c5` (which has the s_setprio ping-pong the naive microbench lacks).

### Why the occ-1 doc's "accumulator width is the only lever" does not extrapolate
Within a single plow kernel family, widening the accumulator moved util 21%→27% — but it **saturates
~30%**. Above that the binding constraint is not chain count; it is instruction scheduling — keeping
the matrix unit continuously fed with interleaved `ds_read` / global-prefetch / MFMA issue. Tensile
emits this by hand in assembly (`PGR2` double global prefetch, `PLR1` local prefetch, scheduled
operand reads, `s_setprio`, exact issue order). HIP/LLVM does not generate it from C++ for either
plow's 32×32 or this 16×16 kernel, so both plateau at ~2× below the library.

## Productionization cost (moot, since NO-GO)
Even if it had worked: at 4-wave WG the HIP compiler put the 40 accumulators in **AGPRs** (210 VGPR +
160 AGPR → occ-1), not the arch-VGPR occ-2 Tensile achieves at 208 VGPR / 0 AGPR. Reproducing the
library's register placement needs assembly-level control plow's persistent megakernel does not have.

## Bottom line
- MFMA shape (16×16 vs 32×32) and independent-chain count are **NOT** the lever for plow's ~30%
  prefill-GEMM MfmaUtil. Adopting the 16×16 MFMA into `op_gemm.h` `d_gemm_t` would **not** raise
  utilization and is not worth the megakernel register/scheduling churn.
- The library's 2× is its hand-scheduled Tensile assembly pipeline, unreachable from HIP source.
- plow's prefill GEMM is at the achievable ceiling for a HIP-source megakernel. Combined with the
  megakernel's zero launch overhead (already beats vLLM prefill 1.41×), **the GEMM investigation
  closes cleanly here.** fp8's wide-K `mfma_*_f8f6f4` (2× MACs/instr) remains the only lever that
  changes the arithmetic itself, not the scheduling — but that rides the same capped pipeline and
  fp8-gemm already measured K64 *dropping* MfmaUtil.

Artifacts: `gemm_mfma16_poc.hip`, `mfma_shape_results/summary.csv`,
`mfma_shape_results/library_disasm.txt`, `scratchpad_bench/bench_lt_prof.cpp`.

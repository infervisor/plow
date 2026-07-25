# PX-2 — native full-tile w8a8 fp8 GEMM mainloop (Gemma-4-12B, sm_120)

Campaign **PX-2-fp8-mainloop**, 2026-07-21, branch `px2-fp8-mainloop` @ `288a714`
(kernel). Box: 1× RTX PRO 6000 Blackwell 96 GB (sm_120, 188 SMs), CUDA 13.0.

## Thesis

T8's w8a8 GEMM used the **bf16 K-step (BK=32)**, so the fp8 `mma.sync.m16n8k32` ran
at **bf16 cadence**: one k32 mma per K-tile vs bf16's two k16, i.e. HALF the compute
per `cp.async` stage + `__syncthreads`, doubling the pipeline/sync overhead per K
element. In the occ-1 prefill megakernel (240 regs, 81 KiB smem, 1 block/SM, no
second block to hide sync stalls) that cadence binds — the campaign's **−28% GEMM
body cyc** (not the −50% the 2× fp8 cores allow).

**PX-2 fix** (`d_gemm_w8a8` / `d_gemm_glu_w8a8`, behind `PLOW_NV_W8A8`):
1. **Deeper K-tile BK 32→64.** fp8 packs 2× per 128-bit `cp.async` line, so a 64-deep
   fp8 tile is the **same smem bytes** as bf16's 32-deep tile. Two k32 mmas per K-tile
   → **half the `__syncthreads`/`cp.async` commit/wait per K element** (native fp8 cadence).
2. **fp8-native swizzle** `pgm_sw8` = CuTe `Swizzle<3,4,3>` over the `[row][k]` fp8 tile
   (MBase=4 keeps the 16-elem cp.async line contiguous). Applied identically on the
   `cp.async` store and the plain-uint32 frag read — a **pure address bijection**, so the
   mma sees the same e4m3 bytes and the oracle is **bit-unchanged**. Breaks the 4-way bank
   conflict of the unswizzled plain reads (fp8 has no `ldmatrix`).
3. T8's two-scale epilogue (`a_scale[m]·w_scale[n]`) and the T3 cp.async ring — **unchanged**.

## Gates (BEFORE any perf claim)

- **Oracle** (`sm120_interp_op_test` w8a8 arm, dequant-f32 e4m3 ref, T8 convention):
  PX-2 is **BIT-IDENTICAL to T8** — quant_fp8 relL2=0 (×2); gemm_w8a8 relL2 0…5.24e-5
  (×4, incl lm_head `a_row0`); gemm_glu_w8a8 5.28e-5…9.61e-5 (×2). ALL PASS (gate 5e-3/8e-3).
  w8a8-vs-full-precision relL2 3.6% (inherent e4m3, unchanged). Whole suite `ok`.
- **Prefill parity**: first greedy token **BIT-IDENTICAL** across bf16 / T8-w8a8 / PX-2-w8a8
  at 4k (236770) and 8k (236770) — within the w8a8-vs-bf16 divergence budget.
- **ptxas** (`interp_sm120_pf`, `PLOW_NV_W8A8=1`): **240 regs** (T8 238, +2), **0 spill**,
  static smem 2320 B, dynamic smem 81664 B — **occupancy unchanged** (the deeper-K arena
  still fits under the bf16/flash smem union). The deeper-K register risk did not materialize.

## GEMM body cycles (block-0, first 4k prefill chunk, `PLOW_NV_TRACE=1`)

clock64 serializes on the recording thread — read the **shape/ratios**, not the absolute
total (per `trace_reduce.py`). Same w8a8 pkt under the T8 vs PX-2 cubin; bf16 = same harness,
bf16 pkt. Family = GEMM(+FP8) + GEMM_GLU(+FP8).

| op (Mcyc)            | bf16   | T8 w8a8 | PX-2 w8a8 | PX-2 vs bf16 | PX-2 vs T8 |
|----------------------|--------|---------|-----------|--------------|------------|
| GEMM / GEMM_FP8      | 467.1  | 351.8   | 259.7     | **−44.4%**   | −26.2%     |
| GEMM_GLU / _FP8      | 504.1  | 315.1   | 241.9     | **−52.0%**   | −23.2%     |
| **GEMM-family**      | 971.2  | 667.0   | 501.7     | **−48.3%**   | **−24.8%** |
| QUANT_FP8            |  —     |  58.7   |  58.4     | (unchanged)  |  ~0        |
| FLASH_PREFILL        | 120.6  | 120.5   | 121.9     | (untouched)  |  ~0        |
| total body           | 1127.1 | 880.0   | 716.0     | −36.5%       | −18.6%     |

**PX-2 hits the target**: GEMM-family body **−48.3% vs bf16** (up from T8's −31.3%),
approaching the −50% the fp8 tensor cores allow, and **−24.8% vs T8**.

## End-to-end prefill wall time (`gemma4_sm120_chat`, chunked prefill, batch 1)

| ctx  | bf16 (ms) | T8 w8a8 | PX-2 w8a8 | PX-2 vs bf16 | PX-2 vs T8 | T8 vs bf16 |
|------|-----------|---------|-----------|--------------|------------|------------|
| 4096 | 512.09    | 432.20  | **339.51**| **−33.7%**   | **−21.4%** | −15.6%     |
| 8000 | 989.93    | 868.02  | **689.63**| **−30.3%**   | **−20.6%** | −12.3%     |

first token 236770 at both ctx for all three. (pkt compiled T=8192; longer-ctx points
need a larger-CTX emit — the 8 full hd512 flash layers grow the O(ctx²) tail, so the
GEMM-share win shrinks with ctx exactly as in the T8 campaign, but the GEMM-body delta is
ctx-independent.)

## Honest negative — the isolated GEMM bench

`experiments/px2_gemm_bench.cu` times the GEMM kernel **standalone** (188 blocks, real
gemma-12B projection shapes). There PX-2 is **3–10% SLOWER** than T8 (w8a8/bf16 ratio
0.50–0.54 vs T8's 0.46–0.52); **both already ~2× bf16**. Reason: the standalone GEMM has a
small footprint and runs at **higher occupancy**, so BK=32's extra syncs are hidden and
BK=64's larger `cp.async` stages overlap compute *worse*. The PX-2 win appears **only** in the
occ-1 megakernel where the sync cadence binds — precisely the "bf16 cadence" the plan named.
This is why the campaign measures in-context: the standalone number would have (wrongly)
killed the experiment. (Swizzle A/B in the same standalone bench: swizzle-on beats swizzle-off
0.50–0.54 vs 0.54–0.58, confirming the fp8 `Swizzle<3,4,3>` reduces the plain-read bank conflict.)

## Reproduce

```
# oracle
nvcc -arch=sm_120a -O2 -I runtime/common -I runtime/nvidia \
  -DPLOW_NV_GEMMA=1 -DPLOW_NV_PREFILL=1 -DPLOW_NV_FA_GF=2 \
  runtime/tests/sm120_interp_op_test.cu -o /tmp/oracle && /tmp/oracle | grep w8a8
# w8a8 pkt (shared by T8/PX-2 cubins)
env PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 PLOW_FP8=1 PLOW_W8A8=1 \
  target/release/gemma4 /workspace/models/gemma-4-12B-it 8192 model.pkt 188
# W8A8 + trace harness
cmake -S runtime -B build -DPLOW_CUDA=ON -DPLOW_NV_W8A8=ON -DPLOW_NV_TRACE_PF=ON
cmake --build build --target gemma4_sm120_chat
PLOW_PREFILL=1 PLOW_UNISEG=1 PLOW_FP8_DIR=<model>/fp8 \
  build/gemma4_sm120_chat model.pkt /workspace/models/gemma-4-12B-it fill4096.ids 2 \
  | python3 scripts/trace_reduce.py
```

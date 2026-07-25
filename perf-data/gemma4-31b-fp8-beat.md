# Gemma-4-31B fp8 (w8a8) vs vLLM — single-user, sm_120

**Campaign fp8-31b-beat, 2026-07-22, branch `fp8-31b`.** Box: 1× RTX PRO 6000
Blackwell 96 GB (sm_120, 188 SMs), CUDA 13.0. Single sequence, batch 1.

Brings the **PX-2 native w8a8 fp8 GEMM mainloop** (`mma.sync.m16n8k32.e4m3`, BK8=64,
`Swizzle<3,4,3>`, two-scale epilogue — validated on 12B in `px2-fp8-mainloop.md`)
to **31B prefill**, on top of the already-shipped weight-only fp8 decode
(`gemma4-31b-plow-sm120.md`). Goal: attack the measured `<64k` prefill/TTFT gap
(2.1× behind vLLM in bf16) with fp8 tensor cores, and close the +3–6% fp8 decode gap.

## What was built

- **fp8 weight twin**: `quantize_fp8.py` → `/workspace/models/gemma-4-31B-it/fp8/`,
  **29.30 GB**, 410 projections, per-row e4m3 (`amax/448`). Dequant vs bf16 relL2 **2.65%**
  (q_proj[0]) — the expected e4m3 band. Disk managed: built once, final free ~22 GB.
- **w8a8 packet**: `PLOW_UNISEG=1 PLOW_FP8=1 PLOW_W8A8=1 gemma4 <dir> 132096 w8a8.pkt 188`
  (weights **29.9 GiB** fp8; the extra `QUANT_FP8` ops raise the per-program packet count
  vs the bf16 pkt — the activation-quant half of w8a8).
- **harness / cubin**: `cmake -DPLOW_CUDA=ON -DPLOW_NV_W8A8=ON` → `gemma4_sm120_chat`; the
  standalone `_pf` cubin (`-DPLOW_NV_W8A8=1`) is **REG 242, 0 spill** (px2 12B was 240; +2
  for the 31B hd512 full-layer flash smem union). System toolchain (`env -i`), not nix.

## Correctness gates (all PASS — before any perf claim)

**1. w8a8 oracle vs dequant-f32 on 31B shapes** (`sm120_interp_op_test`, added H=5376/I=21504 cases):

| op (31B shape)                         | relL2      | tol   | result |
|----------------------------------------|-----------:|------:|:------:|
| gemm_w8a8 m64 n5376 k5376 (o_proj)     | 4.223e-05  | 5e-3  | PASS   |
| gemm_w8a8 m200 n5376 k21504 (down)     | 7.314e-05  | 5e-3  | PASS   |
| gemm_w8a8 m1 n300 k5376 (lm_head)      | 0.0        | 5e-3  | PASS   |
| gemm_glu_w8a8 m64 n21504 k5376 (gate/up)| 3.171e-05 | 8e-3  | PASS   |
| quant_fp8 m64 k5376 / m200 k21504      | 0.0        | 1e-3  | PASS   |

Same band as the validated 12B arm (5e-5). w8a8-vs-full-precision relL2 **0.033–0.036**
(inherent e4m3, identical to px2's 0.036).

**2. w8a8 path is taken** — traced block-0 per-op dump executes the fp8 opcodes:
`op=32 QUANT_FP8 ×240`, `op=33 GEMM_FP8 ×180`, `op=36 GEMM_GLU_FP8 ×60`. `QUANT_FP8`
exists **only** under `PLOW_NV_W8A8`, so its execution proves the true `mma.sync.m16n8k32`
mainloop ran (not the w8a16 dequant fallback).

**3. prefill+decode parity vs bf16** — coherent prompt (ctx 1024, greedy):
bf16 and w8a8 produce a **bit-identical greedy stream** (first token **506** both; all 32
steps match; device==host argmax AGREE on both). *(A uniform-random prompt instead gives an
all-near-tie degenerate logit row — top-5 within 1.1 logits — where bf16=240017/w8a8=537
differ; that is exactly the documented "fp8 disagrees only at bf16 near-ties" budget, not a
w8a8 defect, so parity is judged on the coherent prompt.)*

## Prefill TTFT (ms, lower is better)

| ctx  | plow-fp8 (w8a8) | plow-bf16 (prior) | vLLM bf16 | vLLM fp8 | fp8 vs plow-bf16 | fp8 vs **vLLM bf16** | fp8 vs **vLLM fp8** |
|-----:|----------------:|------------------:|----------:|---------:|:----------------:|:--------------------:|:-------------------:|
| 1024 |   **278.4**     | 473.4             | 222.4     | 159.5    | 0.588× (−41%)    | 1.25× slower         | 1.75× slower        |
| 4096 |   **896.5**     | 1543.3            | 705.7     | 510.2    | 0.581× (−42%)    | 1.27× slower         | 1.76× slower        |
| 16384|   **3851.4**    | 6169.8            | 3450.7    | 2709.4   | 0.624× (−38%)    | 1.12× slower         | 1.42× slower        |
| 32768|   **9030.7**    | 14126.9           | 10730.2   | 5490.8   | 0.639× (−36%)    | **0.84× — FASTER**   | 1.64× slower        |

- **w8a8 lands on 31B prefill**: −36…−42% vs plow bf16 across the ladder (the PX-2 GEMM win,
  end-to-end; px2 measured −30…−34% on 12B, 31B's larger GEMM share gives a bit more).
- **vs vLLM bf16**: the 2.1× bf16 gap is cut to **1.12–1.27×** at 1k–16k and **reversed at 32k**
  (0.84×, 16% faster). plow-fp8 **beats vLLM bf16 TTFT at 32k**, closes hard below.
- **vs vLLM fp8**: **1.42–1.76× behind** — plow-fp8 does **not** match vLLM's fp8 prefill
  (their fp8 prefill kernel + cudagraphs are faster; the plow harness pays per-chunk relaunch
  and no graph capture).

## Decode TPOT (ms/token, lower is better) — sd ≤ 0.11%

| ctx  | plow-fp8 (w8a8) | vLLM bf16 | vLLM fp8 | vs **vLLM bf16** | vs **vLLM fp8** |
|-----:|----------------:|----------:|---------:|:----------------:|:---------------:|
| 1024 |   **26.354**    | 44.670    | 25.620   | −41.0% (faster)  | +2.9%           |
| 4096 |   **26.474**    | 45.200    | 26.160   | −41.4%           | +1.2%           |
| 16384|   **27.421**    | 46.930    | 27.800   | −41.6%           | **−1.4% faster**|
| 32768|   **28.966**    | 49.140    | 29.860   | −41.1%           | **−3.0% faster**|

- **Beats vLLM bf16 by ~41%** at every ctx (fp8 weights halve the decode weight-read).
- **Matches-or-beats vLLM fp8**: +2.9%/+1.2% at 1k/4k, **−1.4%/−3.0% (faster)** at 16k/32k.
  The prior fp8 decode ladder trailed vLLM fp8 by +3–6%; **that gap is closed and reversed**
  at long ctx. (Same weight-only fp8 decode kernel; the w8a8 build's decode object is
  numerically identical — GEMV_FP8 is untouched by the W8A8 flag.)

## Verdict — does plow-fp8 BEAT vLLM on 31B `<64k`?

- **Decode: YES.** Beats vLLM bf16 ~41% everywhere; matches-or-beats vLLM fp8 (fp8 gap closed).
- **Prefill: PARTIAL.** Beats **vLLM bf16** at **32k** (0.84×) and closes the old 2.1× gap to
  1.1–1.3× at 1k–16k — a large win over the bf16 baseline. But **loses to vLLM fp8 prefill**
  (1.4–1.8×): matching vLLM's fp8 TTFT needs more than the GEMM mainloop — cudagraph capture
  of the chunk loop (PX-5) and/or the flash-prefill lever (PX-4), which are out of scope here.

**Net:** fp8 (w8a8 prefill + fp8 decode) makes 31B single-user **decode a clear win vs vLLM**
and **prefill a large improvement** (beats vLLM bf16 at 32k, near-parity below), but does **not**
categorically beat vLLM's own fp8 on prefill `<64k`. Honest negative recorded.

## Limits / caveats

- **VRAM**: a foreign process held **33 GiB** throughout (shared box). bf16 31B (57 GiB wts +
  22.6 GiB KV at max_ctx 132096 ≈ 82 GiB) did **not** fit alongside it, so the bf16 parity ran
  on a T=2048 packet (small KV). fp8 (~55 GiB at full KV) fit with ~8 GiB headroom → the full
  fp8 ladder is real; the bf16 comparison rows above 4k are the prior-campaign measurement.
- plow prefill_ms = chunked prefill, batch 1, **no cudagraph**; vLLM TTFT includes the 1st
  decode token and runs cudagraphs on — the task's specified served baseline (same as the
  prior 31B campaign), not a kernel-isolated comparison.
- GPU shared under a per-run `flock`; foreign plowrt (pid 850160) left untouched.

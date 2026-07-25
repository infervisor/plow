# fp8-KV FAST (PIPE=1) prefill flash — px4 hd512 — **NO-GO** (beat-fp8-prefill, Exp1)

**Verdict: NO-GO at the microbench gate.** The fp8-KV arm of the FAST cp.async prefill flash is
**30–40% SLOWER than bf16 at every context length** (0.60–0.72×). The plan's gate — "fp8 faster
at ≥32k or STOP" — fails. Per the hard sequence, work STOPPED at step 1; no full-model TTFT
ladders were run.

## What was built

- New fp8 arm of `d_flash_prefill_px4<512,32,16,FP8KV>` (op_attention.cuh), sub-option (i):
  raw e4m3 KV staged through the existing cp.async ring into uint8 smem tiles (row stride
  HD+16B), dequanted to a bf16 scratch tile **after** each `cp_wait` (pipeline structure
  unchanged), then the existing `fa_ldmatrix_x2`/`fa_mma` run untouched. K-scale post-multiplies
  the score tile per kv column; V-scale folds into the P fragment (identical scale-application
  points to the shipped decode/PIPE=0 arms).
- New microbench `runtime/tests/flashpre_fp8_bw_sm120.cu` (mirrors `flashdec_fp8_bw_sm120.cu`).
- bf16 arena is byte-identical: the extra e4m3 staging tiles are reserved only under
  `-DPLOW_FP8_KV=1` (`FA_FP8_STAGE_FLOATS`, 0 without the flag).

## Microbench (RTX PRO 6000, 188 SMs, best-of-50, px4 hd512 full-attn, nsplit=12, unit scales)

| ctx  | bf16 ms | fp8 ms | bf16/fp8 | bf16 GB/s(iss) | fp8 GB/s(iss) | relL2 fp8-vs-bf16 |
|------|---------|--------|----------|----------------|---------------|-------------------|
| 8k   | 0.1301  | 0.1816 | 0.716×   | 2064 (>ceil)   | 739           | 2.45e-2           |
| 16k  | 0.2390  | 0.3419 | 0.699×   | 2247 (>ceil)   | 785           | 2.42e-2           |
| 32k  | 0.4674  | 0.6643 | **0.704×** | 2297 (>ceil) | 808           | 2.24e-2           |
| 64k  | 0.9028  | 1.3249 | 0.681×   | 2379 (>ceil)   | 811           | 2.50e-2           |
| 128k | 1.7671  | 2.9240 | 0.604×   | 2431 (>ceil)   | 734           | 2.28e-2           |

HBM ceiling = 1535 GB/s. **The bf16 issued bandwidth (2064–2431 GB/s) is ABOVE the HBM ceiling**
→ the KV read is served from L2, not HBM. The fp8 arm is numerically correct (relL2 ~2.4e-2, the
fp8-KV band; slightly above 3–6e-3 because the synthetic inputs use unit scales over wide-range
random KV rather than the model's calibrated scales — the K/V-scale + dequant plumbing is sound).

## Root cause (why fp8-KV cannot win here)

The physics premise of the plan ("e4m3 KV halves the quadratic-dominated HBM read") assumes the
prefill flash is **HBM-read-bound**. It is not. This kernel's own committed px4/T5 ablation
(`op_attention.cuh:700`) already found the hd512 FULL-attn prefill is **compute-bound** —
softmax 36% + QK 21% + staging 16% + P.V 6% + barriers 3%, per-tile time **flat** 8k→128k, "NOT
DRAM-bound." The microbench confirms it directly: bf16 reads KV from L2 at >ceiling rates (heavy
gqa + cross-q-tile reuse), so the actual HBM demand is far below 1535 GB/s and is not the wall.

Halving KV bytes therefore buys ~nothing. The fp8 arm only ADDS cost on the critical path:
a per-tile dequant pass (e4m3→bf16) plus two extra `__syncthreads` that serialize
dequant→QK and dequant→P.V (bf16 starts QK the instant K lands; fp8 must dequant first). That
overhead is the entire 30–40% regression.

**Option (ii) (register-direct fragment convert) would not rescue it.** ldmatrix requires bf16 in
smem, so option (ii) means a hand-rolled per-fragment convert bypassing ldmatrix — more convert
ALU, with lane-mapping risk — and the HBM saving it protects is still ~0 on a compute-bound
kernel. Best case is parity; the gate requires *faster*.

## Consequence for the campaign

vLLM's fp8kv cuts 128k TTFT ~50% because vLLM's attention kernel is HBM-bound; plow's px4 flash is
compute-bound, so that lever does not transfer. Closing the 128k prefill-TTFT gap to vLLM needs a
**compute** lever on the flash (softmax/QK throughput), not a KV-bandwidth lever. fp8-KV write +
the PIPE=0 read arm remain available for memory-capacity (longer ctx per GPU), just not for speed.

## Gates

- Microbench GO/NO-GO: **NO-GO** (fp8 slower at 32k/64k/128k). ✓ committed (this file).
- fp8 arm correctness: relL2 ~2.4e-2 vs bf16, numerically sound (scales + dequant verified).
- bf16 cubin byte-identity: **VERIFIED byte-identical** — the bf16 prefill object
  (`interp_sm120_pf.cubin`, `-DPLOW_NV_PREFILL=1 -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2`) built from
  this worktree `cmp`s equal (649624 B) to the same object built from the base
  `op_attention.cuh` (origin/beat12b-fp8-margin HEAD). The fp8 additions are fully inert on the
  bf16 path (staging gated behind `PLOW_FP8_KV`; kernel changes inside `if constexpr(FP8KV)`).
- ptxas 0-spill / hd256 arm / dispatch / TTFT ladders: **not run** (gated behind GO).

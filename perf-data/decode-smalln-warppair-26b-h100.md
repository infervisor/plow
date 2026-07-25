# Intra-block warp-pairing on the small-N decode shapes — the closing "beat vLLM" probe (26B, H100, sm_90a)

GPU: **H100 NVL** (sm_90a, 132 SMs, 63 MB L2, HBM3 3350 GB/s spec), bf16, M=1. Date 2026-07-24.
All runs under `gpulease` (rc=0, clean).
**Probe:** `runtime/nvidia/experiments/decode_smalln_warppair.cu` — forks `decode_seg_cpasync.cu`. Tests the ONE untested lever behind the 138 us floor: **INTRA-block warp-pairing** — `Wpr` warps cooperate on ONE output row (each dotting a strided K-slice via cp.async), partials reduced **within the block** by warp-shuffle + a 32-byte smem combine, **NO global atomicAdd**. Only the two starved shapes: o_proj (N2816,K4096) and down_proj (N2816,K2112), weights COLD (192 MB L2 flush/rep), median 120 reps, in-context back-to-back.

## VERDICT (one line)

**NO — plow does NOT beat vLLM decode. Intra-block warp-pairing does not lift the two small-N shapes: at EVERY occupancy and both Wpr∈{2,4} it is EQUAL-OR-WORSE than the 1-warp/row baseline (o_proj stays 33 %, down_proj stays 25 %). The best combined small-N is still the baseline 34.6 us, so the GEMV-family aggregate stays 138 us = 1.07× vLLM, 12 us above the 126 us beat-line.** The idle-warps-per-block premise was a red herring: the GPU already has thousands of resident warps, so pairing adds no net memory-level parallelism — it only shortens each warp's cp.async pipeline and adds sync/reduce overhead. The 25–33 % floor is a **cold-ramp / working-set-too-small** limit (11.9 / 23 MB finish in 14–21 us before HBM reaches steady-state), not a scheduling defect. The defect is real but **not fixable by any warp schedule** — a rigorous negative.

## 1. Per-shape sweep — us / GB/s / %HBM at blocks/SM {2,3,4,5,6} (cp.async D=6, W cold)

**o_proj (N2816,K4096) 23.1 MB**

| schedule | [2] | [3] | [4] | [5] | [6] |
|---|---|---|---|---|---|
| baseline 1w/row | 21.8 / 1057 / 32% | 21.0 / 1101 / 33% | **20.7 / 1116 / 33%** | 21.2 / 1089 / 33% | 20.8 / 1107 / 33% |
| Wpr=2 | 23.2 / 996 / 30% | 22.2 / 1039 / 31% | 22.1 / 1042 / 31% | 22.5 / 1024 / 31% | 21.8 / 1060 / 32% |
| Wpr=4 | 24.3 / 949 / 28% | 22.7 / 1015 / 30% | 21.7 / 1065 / 32% | 22.2 / 1039 / 31% | 21.3 / 1081 / 32% |

**down_proj (N2816,K2112) 11.9 MB**

| schedule | [2] | [3] | [4] | [5] | [6] |
|---|---|---|---|---|---|
| baseline 1w/row | 14.8 / 801 / 24% | **14.0 / 849 / 25%** | 14.2 / 839 / 25% | 14.2 / 837 / 25% | 14.2 / 839 / 25% |
| Wpr=2 | 15.8 / 754 / 23% | 14.8 / 803 / 24% | 14.9 / 798 / 24% | 15.5 / 768 / 23% | 14.4 / 828 / 25% |
| Wpr=4 | 18.5 / 644 / 19% | 16.2 / 733 / 22% | 15.5 / 768 / 23% | 15.6 / 763 / 23% | 14.8 / 803 / 24% |

**Answer to Q1: NO.** Warp-pairing never lifts either shape toward 50 %. Its best-ever cell only *reconverges* to the baseline floor (o_proj Wpr4 occ-6 = 21.3 us/32 %, still < baseline 20.7/33 %; down_proj Wpr2 occ-6 = 14.4/25 %, matching baseline 14.0/25 %). At low occupancy pairing is markedly **worse** (Wpr4 occ-2 drops down_proj to 19 %) because splitting K starves each warp's cp.async pipeline. Occupancy ceiling (dynamic smem, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`): baseline 6–7, Wpr2/Wpr4 = 6 blk/SM — the sweep to occ-6 is real for all three.

## 2. Aggregate plug-in — does GEMV-family drop below 126 us?

Best small-N per shape (min over baseline + Wpr2 + Wpr4, all occupancies):
- o_proj = **20.7 us** (baseline occ-4) — warp-pairing gave nothing.
- down_proj = **14.0 us** (baseline occ-3) — warp-pairing gave nothing.
- combined small-N = **34.7 us** ≈ the prior 34.6 us. No change.

GEMV-family aggregate = 138.0 − 34.6 + 34.7 = **138.1 us** (43 % HBM). Beat-vLLM target < **126 us** → **NOT met, gap = 12 us.** No Wpr/occupancy combination moves it.

## 3. ptxas (sm_90a, -O3) — spill-free, occupancy preserved

| kernel | registers | spill (st/ld) | barriers |
|---|---:|---|---:|
| gemv_cp<6> (baseline) | 32 | 0 | 1 |
| **gemv_cp_wp<6,2>** | **40** | **0** | 1 |
| **gemv_cp_wp<6,4>** | **40** | **0** | 1 |
| gemv_rows (production ref) | 26 | 0 | 0 |

Warp-pair kernels are 40 reg / **0 spill** — occupancy is NOT the reason they fail (they hit 6 blk/SM). The extra 8 reg vs baseline is the reduce/index arithmetic; harmless (smem is the occupancy limiter, not registers).

## 4. Correctness — bit-exact despite the reassociated K-split reduce

vs production `gemv_rows`, occ-4, both shapes:

| schedule | o_proj | down_proj |
|---|---|---|
| baseline 1w/row | 0/2816 mismatches, BIT-EXACT | 0/2816, BIT-EXACT |
| Wpr=2 | **0/2816, max\|abs\|=0, BIT-EXACT** | **0/2816, BIT-EXACT** |
| Wpr=4 | **0/2816, max\|abs\|=0, BIT-EXACT** | **0/2816, BIT-EXACT** |

The Wpr-way split reassociates the fp32 accumulation, yet the stored **bf16** result is byte-identical. Reason: fp32 reassociation error over K=2112–4096 terms is ~1e-5 relative, ~3 orders of magnitude below the bf16 quantization step (2⁻⁸≈4e-3), so every result rounds to the same bf16. Bit-exactness holds by a wide margin — the reduce order is a non-issue.

## Why warp-pairing fails (the mechanism)

The occ-sweep doc read the 33 %/25 % floor as "2–4 of 8 warps idle per block." True per-block, but **irrelevant at the GPU level**: at occ-4 there are 132 SMs × 4 blocks × 6 active warps ≈ 3168 warps resident, each with a depth-6 cp.async pipeline → tens of thousands of loads already in flight. The pipe is already fed. Pairing therefore:

1. **Adds no net MLP** — same bytes, same warps issuing, just a different warp owns each chunk.
2. **Shortens each warp's cp.async pipeline.** down_proj has only 9 K-chunks; Wpr=4 gives ~2–3 chunks/warp — below depth D=6, so the pipeline never fills → *less* latency hiding per warp. This is why low-occupancy Wpr=4 collapses to 19 %.
3. **Adds a per-row cost:** 2× `__syncthreads` + the smem combine.

Net: strictly ≤ baseline. The genuine limiter is that these matrices are **too small to reach steady-state HBM cold** — 11.9 MB / 23 MB stream in 14–21 us, which is the ramp region of the HBM curve (the 1476 MB lm_head, running 150–200× longer, reaches 75 %; the 200 MB aggregate reaches 43 %). No warp schedule shortens a cold-ramp floor set by transfer size.

## THE RESIDUAL

Small-N floor warp-pairing actually reaches: **o_proj 32–33 %, down_proj 24–25 %** — identical to baseline, not the 50 % that would have closed the gap. True residual to the beat-line = **12 us** (138 vs 126), entirely the two small shapes' cold-ramp %HBM, which is **not a scheduling defect and not fixable by intra-block cooperation, cp.async depth, or occupancy** — all three are already saturated. Beating vLLM decode on this path is not reachable by reorganizing the GEMV; it would need a different regime (e.g. keeping these weights warm in L2 across the token so they are not cold, or a fused schedule that overlaps them with the flash/other ops to hide the ramp) — out of scope for a GEMV-schedule probe.

## End-to-end (unchanged from the cp.async result)

| path | GEMV-family | +flash | +relaunch | per-layer | TPOT | vs vLLM |
|---|---:|---:|---:|---:|---:|---:|
| cp.async+occ-4 (best measured) | **138** | 23 | 12 | **~173 us** | **~5.19 ms** | **~1.07×** |
| + warp-pairing (this probe) | **138** | 23 | 12 | ~173 us | ~5.19 ms | ~1.07× |
| vLLM 0.25.1 | — | — | — | 161 us | 4.83 ms | 1.0× |

**Warp-pairing changes nothing. plow decode remains 1.07× vLLM — a MEASURED non-beat. The campaign's residual 12 us is a cold-ramp floor on two <24 MB matrices, not a warp-scheduling defect.**

## Repro

```
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -arch=sm_90a -O3 -Xptxas -v \
    -I runtime/common -I runtime/nvidia -o /tmp/dsw runtime/nvidia/experiments/decode_smalln_warppair.cu
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease dsw /tmp/dsw
```

No interpreter/emitter source modified — standalone probe + this markdown only.

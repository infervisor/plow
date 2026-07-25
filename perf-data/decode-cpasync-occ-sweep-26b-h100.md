# cp.async GEMV × high-occupancy sweep — the decisive H100 decode experiment (Gemma-4-26B-A4B, sm_90a)

GPU: **H100 NVL** (sm_90a, 132 SMs, 63 MB L2, 228 KB smem/SM, HBM3 3350 GB/s spec), bf16, M=1.
All runs under `gpulease` (rc=0, clean). Date 2026-07-24.
**Probe:** `runtime/nvidia/experiments/decode_seg_cpasync.cu` — cp.async row-staged GEMV (depth-6 winner ported from `gemv_cpasync_h100.cu`) run on the REAL 26B decode shapes + flat grouped MoE, weights COLD (192 MB L2 flush/rep), median 120 reps, N-split column ownership (`per=ceil(N/nblk)`, one warp/row), at blocks/SM = grid/132 ∈ {1..6}.

## VERDICT (one line)

**NO — the cp.async GEMV-family aggregate does NOT dip below the ~126 us needed to beat vLLM's 161 us per-layer floor. It floors at 138.0 us at 4 blocks/SM (43 % HBM), spill-free.** BUT it is a genuinely strong result: at 138 us it **beats vLLM's raw GEMV-op sum (147 us)** and **beats the production segmented-GEMV at every matched occupancy** (272/178/149 → 172/146/140 at occ 1/2/3), and it does so **0-spill through occ-6** where the megakernel spilled 8.5 KB at occ-3. The residual +12 us to the beat-vLLM target is **small-N one-warp-per-row starvation** (o_proj 33 %, down_proj 25 %), which high occupancy + smem-staging + cp.async cannot fix because `per=ceil(N/nblk)` collapses to a few rows/block and idles warps.

## Combining the two levers

| lever | source | what it gives |
|---|---|---|
| (1) high occupancy | `decode_seg_gemv.cu` / segmented-decode doc | 272→149 us occ-1→occ-3, **but occ-3 blocked by SPILL** (80-reg cap → 8.5 KB spill/block) |
| (2) cp.async row-staging | `gemv_cpasync_h100.cu` / gemv-cpasync doc | **30–40 reg / 0 spill**, smem-stages weights+operand; only +5–9 % alone at occ-1 |

Hypothesis: (2)'s low register count removes the spill that eroded (1)'s occ-3, and its smem-staging lifts in-context %HBM toward the isolated 46–58 %. **Confirmed on both counts — the kernel is 32 reg / 0 spill and reaches 43–50 % HBM in context — yet the aggregate still floors above the beat-vLLM line.**

## 1. Per-shape sweep — us / %HBM at 1..6 blocks/SM (cp.async D=6, W cold)

| shape (M=1) | weightMB | [1] | [2] | [3] | [4] | [5] | [6] |
|---|---:|---|---|---|---|---|---|
| qkv_proj (N8192,K2816) | 46.1 | 33.9us 41% | 28.1 49% | 27.7 50% | **27.5 50%** | 27.7 50% | 27.8 50% |
| o_proj (N2816,K4096) | 23.1 | 23.0us 30% | 21.8 32% | 20.7 33% | **20.6 33%** | 20.9 33% | 20.8 33% |
| gate_up (N4224,K2816) | 23.8 | 21.6us 33% | 19.5 36% | 19.7 36% | **19.2 37%** | 19.2 37% | 19.6 36% |
| down_proj (N2816,K2112) | 11.9 | 15.6us 23% | 14.7 24% | **14.0 25%** | 14.0 25% | 14.1 25% | 14.0 25% |
| **moe_experts (8 exp)** | 95.2 | 78.0us 36% | 61.8 46% | 58.3 49% | **56.6 50%** | 57.5 49% | 56.9 50% |
| lm_head (N262144) once/tok | 1476 | 692us 64% | 596 74% | 587 75% | 592 74% | 591 75% | **587 75%** |

Occupancy ceiling (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, includes dynamic smem): gemv_cp<6> = **7 blk/SM** at K=2816/2112 (29.5/28.1 KB), **6 blk/SM** at K=4096 (32.0 KB). The sweep to occ-6 is real, not wave-serialized.

## 2. GEMV-family aggregate (per-layer shapes qkv+o+gate_up+down+moe = 200 MB)

| blocks/SM | aggregate us | agg GB/s | agg %HBM |
|---:|---:|---:|---:|
| 1 | 172.1 | 1163 | 35% |
| 2 | 145.9 | 1372 | 41% |
| 3 | 140.4 | 1425 | 43% |
| **4** | **138.0** | **1450** | **43%** |
| 5 | 139.5 | 1434 | 43% |
| 6 | 139.0 | 1439 | 43% |

**Aggregate MIN = 138.0 us at 4 blocks/SM.** Saturates at occ-4; occ-5/6 add nothing (small-N shapes already starved). lm_head (separate, once/token) best = **586.7 us at 6 blk/SM (75 % HBM)**.

- vLLM GEMV-family per-op sum = **147 us** → **cp.async 138 us BEATS it by 6 %.**
- Beat-vLLM-**per-layer** target = **< ~126 us** (so that +flash 23 +relaunch 12 < 161) → **NOT met** (138 > 126, gap = 12 us).

## 3. Per-shape optimum occupancy — small-N starvation vs big-N scaling

| shape | peak us | peak %HBM | at blk/SM | behavior |
|---|---:|---:|---:|---|
| qkv_proj (N8192) | 27.5 | 50% | 4 | big-N, scales cleanly to 50%, flat 4–6 |
| gate_up (N4224) | 19.2 | 37% | 4 | mid-N |
| moe_experts (8×N1408) | 56.6 | 50% | 4 | flat schedule = E·N rows, scales like big-N |
| lm_head (N262144) | 586.7 | 75% | 6 | huge-N, keeps scaling to occ-6, 75% |
| **o_proj (N2816)** | 20.6 | **33%** | 4 | **small-N STARVED** — plateaus at 33% |
| **down_proj (N2816)** | 14.0 | **25%** | 3 | **small-N STARVED** — peaks at occ-3, then idles |

**What caps the aggregate:** the two small-N shapes. At nblk=132·4=528, `per=ceil(2816/528)=6` rows/block over 8 warps → **2 of 8 warps idle**; at occ-5/6 it collapses to 5→4 rows/block (3–4 warps idle). o_proj (33 %) + down_proj (25 %) contribute 34.6 us of the 138 us aggregate at ~29 % HBM while qkv/moe/gate_up already run 50/50/37 %. If those two reached 50 % like the big shapes, aggregate → ~124 us (< 126) — the entire residual gap IS the small-N starvation. This is exactly c1r's "the win erodes beyond ~5/SM" for small-N.

## 4. cp.async+smem-staging vs production segmented-GEMV — matched occupancy

| blocks/SM | production gemv_rows (segmented doc) | cp.async D6 (this) | speedup | spill? |
|---:|---:|---:|---:|---|
| 1 | 272 us (22%) | **172 us (35%)** | **1.58×** | cp: 0 |
| 2 | 178 us (33%) | **146 us (41%)** | **1.22×** | cp: 0 |
| 3 | 149 us (40%)¹ | **140 us (43%)** | **1.06×** | prod spilled 8.5 KB; cp: 0 |
| 4 | — (megakernel could not) | **138 us (43%)** | — | cp: 0 |
| 5–6 | — | 139 us (43%) | — | cp: 0 |

¹ The production occ-3 (149 us) was a **clean-kernel upper bound**; the real megakernel occ-3 needed an 80-reg cap → 8.5 KB spill-loads/block, so its true occ-3 lands between 149 and 178. cp.async **removes that spill entirely** (32 reg) and is faster at every matched occupancy. **The biggest win is at low occupancy** (occ-1: 1.58×) where cp.async's deep pipeline + smem-staged operand hide the HBM latency that the production 8-warp/global-x loop cannot — the two levers compound.

## 5. ptxas (sm_90a, -O3) — spill-free where the megakernel spilled

| kernel | registers | spill (st/ld) | note |
|---|---:|---|---|
| gemv_cp<4> | 34 | 0 | |
| **gemv_cp<6>** | **32** | **0** | the swept kernel |
| gemv_cp<8> | 40 | 0 | |
| moe_gateup_cp<6> | 38 | 0 | |
| moe_down_cp<6> | 36 | 0 | |
| gemv_rows (production ref) | 26 | 0 | |
| _megakernel lean occ-3 (ref)_ | _80_ | _8.5 KB ld_ | _the spill this probe avoids_ |

All ≤ 40 reg, **0 spill at every depth/occupancy** — the hypothesis's premise holds: cp.async runs occ-4/5/6 with none of the spill that eroded the megakernel's occ-3.

## Depth sensitivity (D=4,6,8 on qkv, occ-4)

D4 27.2 us (51%) · D6 27.4 us (50%) · D8 27.8 us (49%) — within 1 %. **At high occupancy, pipeline depth is second-order; occupancy is the dominant lever.** D=6 is fine (D=4 marginally better + smaller smem → more occupancy headroom).

## Bit-exact

gemv_cp<6> vs production `gemv_rows` (qkv N8192 K2816, occ-3): **mismatches = 0 / 8192 — BIT-EXACT.** cp.async changes only where the weight bytes come from (smem ring vs global); the per-lane 256-chunk dot8 order and warp_sum32 epilogue are identical, so the bf16 result is byte-identical by construction.

## End-to-end projection

| path | GEMV-family | +flash | +relaunch | per-layer | TPOT | vs vLLM |
|---|---:|---:|---:|---:|---:|---:|
| current megakernel (occ-1) | 272 | 23 | 0 | ~311 us | 9.34 ms | 1.93× |
| segmented lean occ-2 (prod, solid) | 178 | 23 | 12 | ~213 us | 6.4 ms | 1.32× |
| **cp.async+occ-4 (this, MEASURED, spill-free)** | **138** | 23 | 12 | **~173 us** | **~5.19 ms** | **~1.07×** |
| vLLM 0.25.1 | — | — | — | 161 us | 4.83 ms | 1.0× |

**cp.async × high occupancy is the best measured plow decode path — 311 → 173 us/layer (1.8× plow self-speedup), spill-free, bit-exact — and it narrows the vLLM gap to 1.07×, closer than any prior result. But it does NOT overtake vLLM per-layer.** The true residual to the 126 us beat-line (12 us) is **small-N (o_proj, down_proj) one-warp-per-row starvation** — the N-split arm leaves warps idle at high blocks/SM, holding those two shapes at 25–33 % HBM while everything else reaches 50–75 %. Beating vLLM would require a **column-split / multi-warp-per-row** schedule for the small-N shapes (so all 8 warps stay busy at high occupancy), not more pipeline depth and not more blocks/SM — those are already saturated.

## Repro

```
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -arch=sm_90a -O3 -Xptxas -v \
    -I runtime/common -I runtime/nvidia -o /tmp/dscp runtime/nvidia/experiments/decode_seg_cpasync.cu
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease dscp /tmp/dscp
```

No interpreter/emitter source modified — standalone probe + this markdown only.

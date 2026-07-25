# fp8 (e4m3 w8a16) single-block DECODE sweep — Gemma-4-26B-A4B on H100 NVL (sm_90a)

GPU: **H100 NVL** (sm_90a, 132 SMs, 63 MB L2, 228 KB smem/SM, HBM3 3350 GB/s spec), M=1. Date 2026-07-24.
All runs under `gpulease` (rc=0, clean). fp8 weight = e4m3, 1 byte/elt; x = bf16 (w8a16, dequant-on-load).
This is the fp8 sibling of the bf16 campaign (`decode-cpasync-occ-sweep-26b-h100.md`); shapes IDENTICAL so
the fp8-vs-bf16 comparison is matched. First H100 fp8 decode data (all prior fp8 perf-data is sm120/RTX).

## VERDICT (one line)

**PARITY, not a beat. plow fp8 decode floors at a GEMV-family aggregate of 112.6 us at 4 blocks/SM
(spill-free, bit-exact), projecting to ~147.6 us/layer and TPOT ~4.43 ms vs vLLM fp8's 147 us/layer /
4.417 ms → 1.004x — a statistical tie, 0.6 us over the 112.0 us beat-line.** This is the CLOSEST any plow
decode path has reached (bf16 was 1.07x) and it BEATS vLLM's fp8 per-op GEMV sum (139.6 us with router;
plow fuses the router) and BEATS vLLM's fp8 MoE (47.4 vs 58.4 us). But it does NOT overtake per-layer.
**Why it stops at parity, not the expected 2x:** fp8 halves the weight bytes (200→100 MB) yet the aggregate
only drops 138→112.6 us (−18%, not −50%) because achieved **%HBM collapses ~0.65x on every shape** — halving
each working set pushes it deeper into the HBM **cold-ramp** region (the exact limiter that held bf16 at
1.07x, now WORSE). The byte-halving and the ramp-erosion nearly cancel; net −18% GEMV, and since vLLM fp8
only gains ~9% per-layer (its M=1 MoE/small-proj get NO fp8 speedup), the two lines meet at ~1.00x.

---

## 1. vLLM fp8 per-op DECODE baseline (the target to beat)

`block_op_bench.py --quant fp8` (new flag): vLLM 0.25.1 `Gemma4DecoderLayer` with `Fp8Config`
(online per-tensor weight-quant + dynamic per-tensor activation → w8a8 scaled_mm/cutlass), CUDA-graph-per-op,
L2-flushed cold weights, M=1, ctx=1024. GB/s recomputed with the **real fp8 read bytes = N·K·1** (harness
prints with bf16 accounting; the `us` is the ground truth). Decode weight-ops are ctx-independent (attn is
the only ctx-varying op, and it is unquantized). Venv `/workspace/venvs/vllm-blk`.

| op | fp8 us | fp8 GB/s | % HBM | bf16 us | fp8/bf16 | note |
|---|---:|---:|---:|---:|---:|---|
| **moe_experts** (top-8) | **58.40** | 815 | 24 % | 56.67 | **1.03×** | fp8 gives NO decode speedup at M=1 |
| **qkv_proj** | **16.15** | 1428 | 43 % | 21.79 | 0.74× | best fp8 win (big-N dense) |
| o_proj | 12.48 | 924 | 28 % | 15.70 | 0.80× | |
| mlp_gate_up | 12.29 | 968 | 29 % | 13.80 | 0.89× | |
| mlp_down | 10.72 | 555 | 17 % | 9.97 | **1.08×** | small-K, fp8 SLOWER than bf16 |
| moe_router | 29.55 | — | — | 28.77 | 1.03× | triton routing, latency-bound (plow fuses) |
| attn (FlashDecode) | ~23.0 | — | — | 22.97 | 1.00× | unquantized (KV bf16) → == bf16 |
| **GEMV-family sum (w/ router)** | **139.6** | | | 146.7 | | the per-op microbench target |
| GEMV-family sum (no router) | 110.0 | | | 117.9 | | plow fuses router → excl. |

**Reading it.** vLLM's fp8 win is UNEVEN: qkv/o/gate_up get 0.74–0.89× (cutlass scaled_mm exploits the byte
halving on the bigger dense shapes), but **moe_experts (1.03×) and mlp_down (1.08×) get NOTHING** — at M=1
the fp8 grouped-GEMM / small-K scaled_mm overhead eats the byte savings. Net per-layer (full-model campaign):
vLLM bf16 161 us → **fp8 147 us, only 8.7 % faster**. That soft vLLM-fp8 gain is what lets plow reach parity.

---

## 2. plow fp8 `gemv_cp_fp8` per-op + aggregate at each occupancy

Probe `runtime/nvidia/experiments/decode_fp8_occ.cu` (fork of `decode_seg_cpasync.cu`): production
`dot8_fp8` dequant-on-load (`op_gemm.cuh` gemv_rows_fp8), **same N-split column ownership** (per=ceil(N/nblk),
one warp/row), same 256-K warp-chunk (8 fp8/lane = uint2), x staged in smem, e4m3 weights **COLD** (192 MB L2
flush/rep), median 120 reps, cp.async depth-6 ring (8-byte `cp.async.ca`), blocks/SM ∈ {1..6}.
**GB/s = N·K·1 byte / time.** Occupancy ceiling (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
incl. dynamic smem): gemv_cp_fp8<6> = **8 blk/SM** at all three K (16–20 KB smem) — sweep to occ-6 is real.

### Per-shape sweep — us / %HBM at 1..6 blocks/SM

| shape (M=1) | fp8 MB | [1] | [2] | [3] | [4] | [5] | [6] |
|---|---:|---|---|---|---|---|---|
| qkv_proj (N8192,K2816) | 23.1 | 28.8us 24% | 22.3 31% | 21.1 33% | **20.4 34%** | 21.5 32% | 22.1 31% |
| o_proj (N2816,K4096) | 11.5 | 20.3us 17% | 18.7 18% | **17.1 20%** | 17.1 20% | 17.7 19% | 17.2 20% |
| gate_up (N4224,K2816) | 11.9 | 19.1us 19% | 15.9 22% | 16.4 22% | **15.3 23%** | 15.9 22% | 16.2 22% |
| down_proj (N2816,K2112) | 5.9 | 14.7us 12% | 13.4 13% | **12.3 14%** | 12.4 14% | 12.8 14% | 12.4 14% |
| **moe_experts (8 exp)** | 47.6 | 80.2us 18% | 56.4 25% | 49.8 29% | **47.4 30%** | 48.7 29% | 47.5 30% |
| lm_head (N262144) once/tok | 738 | 575.8us 38% | 465.9 47% | 457.0 48% | 449.2 49% | **432.1 51%** | 434.3 51% |

### GEMV-family aggregate (qkv+o+gate_up+down+moe = 100 MB fp8)

| blocks/SM | aggregate us | agg GB/s | agg %HBM |
|---:|---:|---:|---:|
| 1 | 163.1 | 613 | 18% |
| 2 | 126.8 | 789 | 24% |
| 3 | 116.7 | 857 | 26% |
| **4** | **112.6** | **888** | **27%** |
| 5 | 116.6 | 858 | 26% |
| 6 | 115.4 | 867 | 26% |

**Aggregate MIN = 112.6 us at 4 blocks/SM** (saturates at occ-4, same as bf16). lm_head (separate,
once/token) best = **432.1 us at 5 blk/SM (51 % HBM)**. Per-shape optima: qkv 20.4us(34%)@4, o_proj
17.1us(20%)@3, gate_up 15.3us(23%)@4, down 12.3us(14%)@3, moe 47.4us(30%)@4.

- vLLM fp8 GEMV-op sum (w/ router) = **139.6 us** → plow **112.6 us BEATS it by 19 %.**
- vLLM fp8 GEMV-op sum (no router) = **110.0 us** → plow 112.6 is **+2.4 % (a tie).**
- Beat-vLLM-**per-layer** target = GEMV-family **< 112.0 us** (so +flash 23 +relaunch 12 < 147) →
  **NOT met by 0.6 us.**

---

## 3. fp8 vs bf16 at matched occupancy — the cold-ramp erosion, quantified

**Aggregate @ occ-4:** bf16 138.0 us / 43 % HBM / 200 MB  →  fp8 **112.6 us / 27 % HBM / 100 MB**.
Bytes HALVED, time only **−18 % (1.23×), NOT −50 %.** The missing 2x is the **%HBM collapse**:

| shape | bf16 peak us / %HBM | fp8 peak us / %HBM | fp8/bf16 time | fp8/bf16 %HBM |
|---|---:|---:|---:|---:|
| qkv_proj (N8192) | 27.5 / 50% | 20.4 / 34% | 0.74× | 0.68× |
| o_proj (N2816,K4096) | 20.6 / 33% | 17.1 / 20% | 0.83× | 0.61× |
| gate_up (N4224) | 19.2 / 37% | 15.3 / 23% | 0.80× | 0.62× |
| down_proj (N2816,K2112) | 14.0 / 25% | 12.3 / 14% | 0.88× | 0.56× |
| moe_experts (8×) | 56.6 / 50% | 47.4 / 30% | 0.84× | 0.60× |
| lm_head (N262144, huge) | 586.7 / 75% | 432.1 / 51% | 0.74× | 0.68× |

**Every shape loses ~0.60–0.68× of its %HBM going bf16→fp8** — exactly the hypothesis's prediction (a).
Halving the bytes halves each working set (qkv 46→23 MB, down 12→6 MB, moe 95→48 MB), and a smaller cold
transfer sits further down the HBM ramp, so achieved BW drops. The erosion is worst where it already hurt:
**down_proj falls to 14 % HBM (5.9 MB streams in 12 us) and o_proj to 20 %** — the same two small-N shapes
that capped the bf16 aggregate, now even more ramp-starved. Even the 738 MB lm_head drops 75→51 %. The
byte-halving still wins net (because time = bytes/BW and bytes fell faster than BW), but only by ~1.23×,
so fp8 does NOT deliver the naive 2×. **This is a fundamental cold-ramp limit of small COLD decode weights,
not a kernel defect** — confirmed identical-mechanism to the bf16 warp-pair negative.

---

## 4. THE VERDICT — projected per-layer & TPOT vs vLLM fp8

`+flash 23 us` (unquantized FlashDecode, == bf16) and `+relaunch 12 us` (segmented multi-launch overhead)
carry over from the bf16 campaign unchanged.

| path | GEMV-family | +flash | +relaunch | per-layer | TPOT (×30) | vs vLLM fp8 |
|---|---:|---:|---:|---:|---:|---:|
| current fp8 megakernel (occ-1) | ~ | ~ | 0 | ~246 us | **7.395 ms** | 1.67× |
| **fp8 cp.async + occ-4 (this, MEASURED)** | **112.6** | 23 | 12 | **~147.6 us** | **~4.428 ms** | **~1.004×** |
| vLLM fp8 (0.25.1, campaign) | — | — | — | 147 us | 4.417 ms | 1.0× |
| _(bf16 cp.async+occ-4, for reference)_ | _138.0_ | _23_ | _12_ | _~173 us_ | _~5.19 ms_ | _1.07× (vs vLLM bf16)_ |

**Does plow fp8 decode BEAT vLLM fp8? NO — it reaches PARITY (~1.004×), a hair short of a beat.**
- Exact numbers: plow fp8 GEMV-family **112.6 us @ occ-4**, per-layer **147.6 us**, TPOT **4.428 ms**;
  vLLM fp8 147 us / 4.417 ms. Margin = **+0.6 us/layer / +11 us TPOT = 1.004×.**
- fp8 is a genuine improvement over bf16 (1.07× → 1.00×) and a 1.67× plow self-speedup over the current
  fp8 megakernel (7.395 → 4.43 ms), spill-free and bit-exact. plow **beats vLLM's fp8 MoE** (47.4 vs 58.4)
  and its **per-op GEMV sum-with-router** (112.6 vs 139.6).
- **The floor and why:** the residual is NOT dequant register pressure (gemv_cp_fp8<6> = **30 reg / 0 spill**,
  LOWER than bf16's 32; the megakernel's ~680-spill concern is a PREFILL-wgmma-GLU issue, absent here) and
  NOT latency-bound-at-half-bytes at the aggregate. It is **cold-ramp %HBM erosion**: fp8 runs the whole
  GEMV-family at only 27 % HBM (bf16 43 %), because halving the weights halves every working set into the
  HBM ramp. plow loses the four dense projections to vLLM's cutlass scaled_mm (qkv 34 vs 43 %, o_proj 20 vs
  28 %, gate_up 23 vs 29 %, down 14 vs 17 %) and wins them back only on the big MoE (30 vs 24 %) — netting a
  tie. **True residual ratio: 1.004× (≈ parity).** Beating vLLM fp8 would need the same regime-change the
  bf16 doc named — keeping these <24 MB weights WARM across the token, or fusing/overlapping to hide the
  cold ramp — not any GEMV-schedule change (occupancy, depth, N-split are all saturated).

---

## ptxas (sm_90a, -O3) — dequant does NOT raise decode-GEMV registers

| kernel | registers | spill (st/ld) | barriers |
|---|---:|---|---:|
| **gemv_cp_fp8<6>** (the swept kernel) | **30** | **0** | 1 |
| gemv_cp_fp8<4> | 30 | 0 | 1 |
| gemv_cp_fp8<8> | 38 | 0 | 1 |
| moe_gateup_cp_fp8<6> | 36 | 0 | 1 |
| moe_down_cp_fp8<6> | 38 | 0 | 1 |
| gemv_rows_fp8_ref (production body) | 25 | 0 | 0 |
| _bf16 gemv_cp<6> (reference)_ | _32_ | _0_ | _1_ |

fp8 dequant-on-load is **30 reg / 0 spill — LOWER than the bf16 kernel (32)** because the e4m3 ring stores
half the bytes and the cvt is register-cheap. The op_gemm.cuh:519 "~680 spill" warning is about the wgmma
PREFILL GLU fork in the megakernel — the standalone decode probe never touches it. **Occupancy is
smem-limited (8 blk/SM ceiling), not register-limited.**

## Depth sensitivity (D=4,6,8 on qkv, occ-4)

D4 19.7 us (35%) · D6 20.4 us (34%) · D8 21.1 us (33%) — D=4 marginally best. As in bf16, at high occupancy
**pipeline depth is second-order; occupancy is the dominant lever.**

## Bit-exact

gemv_cp_fp8<6> vs production gemv_rows_fp8 (qkv N8192 K2816, occ-3): **mismatches = 0 / 8192,
max|abs Δ| = 0.000 — BIT-EXACT.** cp.async changes only where the fp8 bytes come from (smem ring vs global);
the per-lane 8-fp8 dot ordering, the `__nv_cvt_fp8x2_to_halfraw2` dequant, the fmaf order, the warp_sum32
epilogue, and the `scale[n]` factor are byte-identical to the production body → the bf16 result is identical
by construction.

## Repro

```
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -arch=sm_90a -O3 -Xptxas -v \
    -I runtime/common -I runtime/nvidia -o /tmp/dfp8 runtime/nvidia/experiments/decode_fp8_occ.cu
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease dfp8 /tmp/dfp8

# vLLM fp8 per-op baseline (new --quant fp8 flag on block_op_bench.py):
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  PLOW_PY=/workspace/venvs/vllm-blk/bin/python gpulease blkopf \
  /workspace/venvs/vllm-blk/bin/python scripts/block_op_bench.py \
  <block-configs>/gemma4-26b-a4b-moe.json --phases decode --ctx 1024 --quant fp8 \
  --out /dev/shm/block-op/26b-op-fp8.json
```

No production interpreter/emitter source modified — standalone probe + measurement-harness `--quant fp8`
extension (defaults unchanged for bf16) + this markdown only.

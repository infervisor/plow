# Segmented DECODE for Gemma-4-26B-A4B on H100 NVL — the lean high-occupancy GEMV object

GPU: **H100 NVL** (sm_90a, 132 SMs, 63 MB L2, HBM3 3350 GB/s spec), bf16, M=1.
All GPU runs under `gpulease` (uncontended). Date 2026-07-24.

## The thesis (mission)

plow's decode megakernel is `__launch_bounds__(256,1)` → **1 block/SM = 12.5 % occupancy**;
its GEMV runs at **~21 % of HBM peak** (816 GB/s), the exact 1.9× gap vs vLLM. The
proven fix (AMD `interp.hip:430`, NVIDIA prefill `_pfgemm`): run memory-bound GEMV as a
**LEAN high-occupancy segment object** (register-hungry flash arms compiled OUT → more
blocks/SM), route flash to a separate occ-1 object. Isolated `ksplit_gemv.cu` hit
**46–58 % HBM at 3 blocks/SM**. Does that transfer to the real 26B decode op sequence
in context — and does it beat vLLM's decode block?

## What was built

1. **`runtime/nvidia/experiments/decode_seg_gemv.cu`** — production-faithful N-split GEMV
   (`op_gemm.cuh gemv_rows`: one warp/row, block owns a column slice `per=ceil(N/nblk)`,
   x read from **global**) across the REAL 26B decode GEMV shapes + a **flat grouped MoE**
   (op_moe.cuh schedule), run back-to-back with **weights cold via L2 flush** (block_op_bench
   methodology), at occupancy = grid/nSM ∈ {1,2,3}. Plus the per-segment relaunch cost.
2. **`interp_sm120.cu` — `PLOW_NV_LEAN_DECODE` gate** (default 0, byte-identical when off;
   verified: default decode object still 177 regs / symbol present). When 1, the
   `d_flash_decode<512>` / MLA / DSA / flash-merge arms compile OUT and the flash arena
   collapses to the 2·WARPS floor — the DECODE analog of `PLOW_NV_SEG_GEMM`. Used here to
   measure the real lean object's register/occupancy ceiling.

## Result 1 — occupancy DOES lift in-context GEMV BW (validated against the megakernel)

`decode_seg_gemv.cu`, W cold, us / GB/s / %HBM at 1,2,3 blocks/SM:

| op (M=1) | weightMB | 1 blk/SM | 2 blk/SM | 3 blk/SM |
|---|---:|---|---|---|
| qkv_proj (N8192,K2816) | 46.1 | 60.7us 760 **23%** | 37.7us 1225 37% | 33.5us 1379 **41%** |
| o_proj (N2816,K4096) | 23.1 | 34.0us 679 20% | 25.8us 895 27% | 18.5us 1247 **37%** |
| gate_up (N4224,K2816) | 23.8 | 32.5us 732 22% | 21.2us 1121 33% | 21.6us 1103 33% |
| down_proj (N2816,K2112) | 11.9 | 21.6us 551 16% | 17.1us 697 21% | 12.7us 937 **28%** |
| **moe_experts (8 exp)** | 95.2 | 123.4us 771 **23%** | 76.8us 1240 37% | 62.8us 1515 **45%** |
| **GEMV-family SUM** | 200 | **272 us (22%)** | **178 us (33%)** | **149 us (40%)** |

**vLLM GEMV-family per-op sum (measured) = 147 us; whole decode block = 356 us.**

- **The occ-1 column reproduces the megakernel.** 272 us / 22 % HBM ≈ the campaign's measured
  **21 %-of-peak / ~272 us GEMV / 311 us-per-layer** decode body. The standalone is faithful.
- **moe_experts — the prime target (56 us, biggest single op) — IS fixed by occupancy:**
  23 % → **45 %** HBM (123 → 63 us at occ-3). Higher occupancy is the lever, confirmed.
- **At occ-3 the aggregate GEMV time (149 us) matches vLLM's GEMV sum (147 us).** The
  per-op %HBM still trails vLLM per-op (vLLM qkv 63 %, moe 50 % — dedicated cuBLAS/MoE
  kernels beat the interpreter's one-warp-per-row arm at matched occupancy), but the
  aggregate washes out.
- **In-context plateaus below the isolated probe.** Aggregate tops at 33 % (occ-2) / 40 %
  (occ-3), NOT the isolated ksplit 46–58 % — because production reads x from global (not
  smem-staged) and the small shapes (down 28 %, o 37 %) stay bandwidth-modest cold.

## Result 2 — the real lean object reaches occ-2 cleanly, occ-3 only by spilling

`interp_sm90a.cu` built `-DPLOW_NV_LEAN_DECODE=1` (Gemma serving flags), `-Xptxas -v`:

| build | regs | spill (st/ld) | blocks/SM | note |
|---|---:|---|---:|---|
| baseline decode (flash IN) | 177 | 0 | **1** | shipped object, the ceiling |
| lean (flash OUT), natural | **163** | 0 | 1 | flash sheds only 14 regs — qkv/headnorm/argmax now own 163 |
| lean + FORCE_MINBLK=2 | **128** | 500/1044 B | **2** | light spill — occ-2 clean |
| lean + FORCE_MINBLK=3 | **80** | 4466/8492 B | **3** | **heavy spill (8.5 KB ld)** — competes with HBM |

- Dropping flash-decode alone is **not enough** (177 → 163, still 1 block/SM): the decode
  megakernel is one switch-function whose register max is set by several arms, not flash
  alone. Reaching occ-2/3 needs the `__launch_bounds__` register cap → spills.
- **occ-2 (128 regs, negligible spill) is the SOLID reachable point.** occ-3 needs an
  80-reg cap with **8.5 KB spill-loads/block** — on a memory-bound kernel that spill DRAM
  traffic will erode the occupancy gain (the exact A/B the source comment warns of). So the
  probe's clean-kernel occ-3 (40 % HBM, 149 us) is an **upper bound**; a real occ-3 lands
  between occ-2 and occ-3.

## Result 3 — per-segment relaunch overhead is NOT the killer

Measured on H100: empty-kernel **launch = 2.11 us** (async, amortized), **launch+sync =
4.16 us**. Decode has ~2–3 GEMV↔flash segment boundaries per layer → ~**12 us/layer** of
relaunch tax (~4 % of a layer). Contrary to the Step-3 worry, relaunch does **not** negate
the occupancy gain for decode. (T9c measured the same: 3.08 us/launch enqueue, negligible.)

## End-to-end projection & VERDICT

Per-layer decode (the transferable floor; the 356 us whole-block over-states vLLM's fused-away
norm launches — real per-layer floor is vLLM **161 us** / plow **311 us**, the 1.93× gap):

| path | GEMV-family | +flash | +relaunch | **per-layer** | TPOT | vs vLLM |
|---|---:|---:|---:|---:|---:|---:|
| current megakernel (occ-1) | 272 | 23 | 0 | **~311 us** (measured) | 9.34 ms | 1.93× |
| **segmented lean occ-2 (SOLID)** | 178 | 23 | ~12 | **~213 us** | ~6.4 ms | **~1.32×** |
| segmented lean occ-3 (upper bound, spill-eroded) | 149 | 23 | ~12 | **~184 us** | ~5.5 ms | ~1.14× |
| vLLM 0.25.1 | — | — | — | **161 us** | 4.83 ms | 1.0× |

**Segmented decode CLOSES most of the gap but does NOT beat vLLM.** It takes plow from
**1.93× → ~1.3×** (occ-2, solid) and fixes the single biggest op (moe_experts 23 % → 45 %
HBM, 123 → 63 us). occ-3 could reach ~1.14× **if** the 80-reg spill traffic doesn't erode
it — it will, partly. The residual loss is NOT relaunch overhead (~4 %); it is **(a)** the
in-context lean-object BW plateau (33–40 % HBM, short of vLLM's 50–63 % per-op — the
interpreter's one-warp-per-row GEMV arm trails vLLM's dedicated cuBLAS/MoE kernels even at
matched occupancy) and **(b)** occ-3's spill cost. This matches the prefill T10 finding
(occ-2 needs a register/pipeline sacrifice that partly cancels the occupancy headroom), but
DECODE is the better case: memory-bound, so the occupancy headroom is real (+46 % on the
GEMV family occ-1→occ-3) where prefill's GEMM was already latency-hidden and regressed.

**Recommendation.** The lean occ-2 decode segment object is a real, correctness-gateable
**~1.4× speedup for plow's own decode** (311 → 213 us/layer) — worth shipping to narrow the
gap and as the substrate for concurrent/batched decode — but on 26B/H100 short-ctx B=1 it
does not overtake vLLM. To actually beat 161 us/layer would additionally require GEMV arms
that match vLLM's per-op efficiency at occupancy (the 33–40 % vs 50–63 % gap), not just the
occupancy fix.

## Next step for a full Step-2 (not built here — measurement gates it)

Two cubins (`interp_sm90a` lean occ-2 GEMV + the existing occ-1 flash object) + a decode
packet split into GEMV/flash wave-class segments + `exec/gpu.rs` per-segment dispatch
(mirror the prefill `_pfseg`/`_pfgemm` path). The `PLOW_NV_LEAN_DECODE` gate is the object
half; the emit + dispatch half is the remaining plumbing. Build it only if ~1.3× (a plow
self-speedup, not a vLLM win) is worth the launch-orchestration complexity — the numbers
above say it narrows but does not close.

## Repro

```
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -arch=sm_90a -O3 -Xptxas -v \
    -o /tmp/dsg runtime/nvidia/experiments/decode_seg_gemv.cu
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease dsg-step1 /tmp/dsg
# lean-object register ceiling:
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -arch=sm_90a -O3 -cubin \
  -I runtime/common -I runtime/nvidia -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 \
  -DPLOW_NV_FA_GF_FULL=4 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_NV_MLA=0 -DPLOW_NV_MAMBA=0 \
  -DPLOW_NV_DSA=0 -DPLOW_NV_LEAN_DECODE=1 -DPLOW_NV_FORCE_MINBLK=2 \
  -o /tmp/dec_lean_mb2.cubin runtime/nvidia/interp_sm90a.cu
```

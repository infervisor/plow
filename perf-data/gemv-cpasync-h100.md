# cp.async weight-staging vs production decode GEMV — H100 NVL (sm_90a)

**Probe:** `runtime/nvidia/experiments/gemv_cpasync_h100.cu`
**Date:** 2026-07-24 · **Device:** NVIDIA H100 NVL, 132 SM, 63 MB L2, HBM3 (3350 GB/s achievable)
**Geometry:** GRID=132 (1 block/SM = megakernel), blockDim=256 (8 warps), M=1 bf16, `slice=blockIdx.x`, `nblk=gridDim.x`, x smem-staged, weights HBM-resident.
**Method:** weights read COLD (192 MB L2-flush memset between every timed rep), median of 120 reps. GB/s = N·K·2 / t.
Two independent runs agreed to <1% on every cell.

## Verdict (one line)

cp.async row-staging is **bit-exact** and gives a **modest 1.05–1.09× at pipeline depth 6–8** on the mid-size decode bodies (MoE expert gate/up + down, dense gate/up + down — the bulk of 26B decode), **neutral on o_proj (1.01×), and a slight LOSS on lm_head (0.97×)**. It does **not** reproduce the RTX c1r 1.25–1.30× win and does **not** close the H100 decode gap: the mid-size bodies stay at 12–42 % HBM even with cp.async, because at 1 block/SM = 8 warps the small-N shapes never launch enough warps to fill H100's bandwidth-latency product — a per-warp prefetch change cannot fix an occupancy-parallelism deficit. **Not the decode-gap lever.** Worth wiring into `d_gemv` only as a small, safe, zero-register-cost sweetener (it costs *fewer* regs than baseline), and only in combination with the real fix (more blocks/SM, the segmented-decode / lean-object direction), not on its own.

## A/B table (GB/s, %HBM, speedup vs baseline; mm = bit-exact mismatches)

| shape (26B decode op) | N | K | baseline GB/s (%HBM) | D2 | D3 | D4 | D6 | D8 | best |
|---|---|---|---|---|---|---|---|---|---|
| moe_gate_up  op71 (~37% of decode, per-expert) | 1408 | 2816 | 512 (15%) | 457 ·0.89× | 503 ·0.98× | 527 ·1.03× | **545 ·1.06×** | 538 ·1.05× | D6 1.06× |
| moe_down     op63 | 2816 | 704  | 385 (11%) | 368 ·0.96× | 400 ·1.04× | **405 ·1.05×** | 395 ·1.03× | 390 ·1.01× | D4 1.05× |
| qkv          op22 | 8192 | 2816 | 1275 (38%) | 1028 ·0.81× | 1191 ·0.93× | 1291 ·1.01× | 1380 ·1.08× | **1394 ·1.09×** | D8 1.09× |
| o_proj       op10 | 2816 | 4096 | 1003 (30%) | 801 ·0.80× | 900 ·0.90× | 964 ·0.96× | 1003 ·1.00× | **1017 ·1.01×** | D8 1.01× |
| dense_gate_up op19 (GLU) | 4224 | 2816 | 1035 (31%) | 879 ·0.85× | 991 ·0.96× | 1065 ·1.03× | **1111 ·1.07×** | 1106 ·1.07× | D6 1.07× |
| dense_down   | 2816 | 2112 | 692 (21%) | 635 ·0.92× | 708 ·1.02× | 746 ·1.08× | **757 ·1.09×** | 751 ·1.08× | D6 1.09× |
| lm_head (control, big N) | 262144 | 2816 | 2205 (66%) | 1675 ·0.76× | 1980 ·0.90× | 2127 ·0.96× | 2140 ·0.97× | 2096 ·0.95× | **loses** |

**Bit-exactness: mm = 0 on every shape at every depth** (memcmp of the full N-vector of bf16 outputs against the baseline). cp.async changes only *where* the weight bytes come from (smem vs global); the per-lane 256-chunk dot8 order and warp_sum32 epilogue are identical, so the bf16 result is byte-identical by construction.

## ptxas (sm_90a, `-O3 --use_fast_math`)

| kernel | registers | spill | notes |
|---|---|---|---|
| `gemv_base` (production d_gemv inner loop) | **56** | 0 | holds `wv[GV_UNROLL=8]` bf16v8 in registers |
| `gemv_cpasync<2>` | 30 | 0 | |
| `gemv_cpasync<3>` | 32 | 0 | |
| `gemv_cpasync<4>` | 34 | 0 | |
| `gemv_cpasync<6>` | 32 | 0 | |
| `gemv_cpasync<8>` | 40 | 0 | |

cp.async uses **fewer registers than baseline** (weights live in the smem ring, not registers) — matching the c1r observation (64 vs 80 on RTX). All variants ≤ 40 regs, well under the 128 target; wiring it in would *lower* the megakernel register ceiling, not raise it. Smem at D=8: xs(K·2 ≤ 8 KB) + ring(8·8·256·2 = 32 KB) = 40 KB < 48 KB, so no `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` needed.

## Why it barely wins (the physics the data shows)

The production baseline is **not** a naive blocking loader — its `GV_UNROLL=8` manual prefetch already issues **8 outstanding 16 B `ld_glob8` per lane** before consuming any, i.e. it already carries deep memory-level parallelism. That is exactly why:
- **D2/D3 lose** (0.76–0.98×): a depth-2/3 cp.async pipeline keeps *fewer* loads in flight than the baseline's 8-deep prefetch.
- cp.async only reaches **parity at D≈4 and pulls ahead ~5–9 % at D=6–8**, once its in-flight depth matches/exceeds the baseline's, and even then the per-256-chunk `commit_group`/`wait_group` overhead partially cancels the latency it hides.
- **lm_head loses** at every depth: at N=262144 the baseline already launches enough warps to reach 66 % HBM, so there is no latency left to hide and cp.async's commit/wait is pure overhead.

The mid-size bodies stay at **12–42 % HBM even with cp.async**. The ceiling is not per-warp prefetch depth — it is that 8 warps/SM at 1 block/SM cannot generate enough concurrent misses to saturate H100's HBM3 on small-N shapes. Raising **blocks/SM** (the lean high-occupancy object in `decode_seg_gemv.cu`, which hit 46–58 % at 3 blocks/SM) is the real lever; cp.async row-staging is a second-order refinement on top of it.

## The 3 defects in the prior probe `gemv_transport.cu` and how (B) avoids them

1. **Not bit-exact vs production.** transport's `gemv_bf16_cpasync` accumulates x as **fp32** (`float* xs`), stages a shared `ROWS×SK` block across a warp, and walks columns `k=lane*8; k<SK; k+=WARP*8` — a **different summation order** than `gemv_rows`' per-lane contiguous dot8. Result is only relL2-close, never byte-identical, so it could never drop into `d_gemv`.
   → (B) keeps the **exact lane ownership** (lane l owns `[l*8, l*8+8)` of every 256-chunk) and the **exact dot8/warp_sum32 order**; verified 0-mismatch memcmp against baseline on all 7 shapes.
2. **Hard-wired K = 2560, `nst = K/SK` with no tail predicate.** A K not divisible by SK (26B has K=704, 2112) would silently drop the last partial chunk or over-read the row.
   → (B) carries the production `k < K` predicate on **both** the cp.async issue (`src_bytes = k<K ? 16 : 0`; hardware zero-fills and reads no gmem past the row) **and** the consume (skip the dot, exactly like baseline's `if(kk>=K) continue`). Works on the non-256-multiple K's (704, 2112) with 0 mismatches.
3. **One `cp.async.commit_group` + `wait_group 0` per stage = fully blocking, no overlap** — it measured a transport, not a pipeline.
   → (B) keeps a **constant depth of D commit-groups in flight** (`wait_group<D-1>`), issuing chunk `c+D` while consuming chunk `c`, so the next HBM chunk lands while the current one is being FMA'd. Depth is the swept knob.

   (transport also measured **L2, not HBM**, for buffers < L2 — this probe L2-flushes a 192 MB region between every rep so weights read cold from HBM3.)

## Build / run

```
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  nvcc -arch=sm_90a -O3 --use_fast_math -Xptxas -v \
  -I runtime/nvidia -I runtime/common -o /tmp/gcah \
  runtime/nvidia/experiments/gemv_cpasync_h100.cu
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH \
  PLOW_LIBCUDA=/usr/local/cuda-13.0/compat/libcuda.so.1 \
  gpulease gcah /tmp/gcah
```

No interpreter/emitter source was modified — standalone probe + this markdown only.

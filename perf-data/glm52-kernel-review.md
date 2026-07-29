# GLM-5.2 TP4 block-fp8 decode — review of the EXISTING kernels

Scope: the user asked to *review the existing kernels for high performance first*, and only add new
ones if the inventory is not enough. This is that audit. **No kernel was written or modified.**

Measured 2026-07-28 on one idle MI355X (gfx950) under `gpulease -n 1`, no "CONTENDED AT START"
warning on any run. Every kernel benched is the **production `__device__` body included verbatim**
from `runtime/amd/op_{moe,gemm,attention,norm,elementwise}.h` — no copies, the `runtime/ubench`
pattern. Launch geometry mirrors the persistent interpreter: `blockDim = PLOW_THREADS = 512`
(8 waves = 2 waves/SIMD), `gridDim` = the workgroup count the emitter actually gives that packet.
Denominator is **6200 GB/s** (contract §5), never the 8 TB/s datasheet number.

Bench sources checked in next to this file:
`perf-data/glm52_kbench_dev.hip` (device wrappers over the production ops),
`perf-data/glm52_kbench_moe.cpp` (§4 MoE slice-map A/B),
`perf-data/glm52_kbench_ops.cpp` (§3 remaining ops),
`perf-data/glm52_kern_probe.hip` (compile-only register/occupancy probe, no GPU needed).

Reproduce (ROCm tooling must run OUTSIDE `nix develop` — §0a):
```
hipcc --offload-arch=gfx950 -O3 -w -DPLOW_BUCKET_DECODE=1 -std=c++17 --genco \
      perf-data/glm52_kbench_dev.hip -o /tmp/glm_kdev.co -Iruntime/amd -Iruntime/common
g++ -O2 -std=c++17 perf-data/glm52_kbench_moe.cpp -o /tmp/kb_moe \
      -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ -L/opt/rocm/lib -lamdhip64
perf-data/harness/gpulease -n 1 kbench sg render -c \
      'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; /tmp/kb_moe /tmp/glm_kdev.co slot-outer'
```
`unset HIP_VISIBLE_DEVICES` is load-bearing: `gpulease` exports BOTH `ROCR_VISIBLE_DEVICES` and
`HIP_VISIBLE_DEVICES` to the same absolute id, and they compose — ROCR exposes the leased card as
agent 0, then HIP re-filters for index 5 and finds nothing. `hipInit` returns
"no ROCm-capable device is detected" on a correctly leased GPU. Fourth instance of the
lease/visibility class of bug in §0a.

---

## 0. TL;DR

1. **The kernels are not where the GLM token goes.** Summing every measured kernel body over one
   token gives **~10.2 ms of a 31.1 ms token**. The other ~21 ms is the packet protocol (gates,
   dispatch, the serial 2756-packet chain) plus the 3.84 ms of collectives §6e-0 already priced.
2. **The three GEMV families are already excellent and must not be touched**: `Gemv` bf16 at
   **83–106 % of the 6200 GB/s ceiling**, `GemvQkv` (fusion A/G) at **71–83 %**. lm_head measures
   **106 %** of ceiling.
3. **The three kernels worth optimising are `MlaMergeFold`, `MoeExpertDownFp8Blk` and
   `FlashMlaDecode`**, and all three fail for the *same* reason and it is **not** pipelining,
   register pressure, LDS, or scale-grid indexing. It is **how many waves the slice map gives them**.
   In every case `achieved % of ceiling ≈ active-wave fraction`, to within a factor of ~1.2.
4. Top-3 recoverable ≈ **3.3–3.9 ms/token**. That takes 31.1 → ~27.5 ms. vLLM is 22.5 (chat
   backend) / 24.9 (completions). **Kernel work alone cannot close the GLM gap.**
5. Validation that this bench measures the right thing: the kernel-level A/B reproduces the
   full-model `GLM_GROUP=1` knob result (§6g-KNOBS, +2.88 ms) at **+2.95 ms**, within 3 %.

---

## 1. Shapes GLM actually emits (TP4, M=1, ctx 1024, per rank)

`hidden 6144, q_lora 2048, kv_lora 512, qk_rope 64, v_head 256, heads 64 → nh_l 16,
experts 256 top_k 8, moe_inter 2048 → imoe_l 512, dense_inter 12288 → di_l 3072,
78 layers = 3 dense + 75 sparse, vocab 154880.`

| what the kernel gets | shape | fp8? |
|---|---|---|
| routed expert gate/up (op 45) | N=512 (**skinny**), K=6144, ×2 proj ×8 experts | yes, [128,128] |
| routed expert down (op 46) | N=6144, **K=512** — 1 chunk, half the wave dead | yes |
| dense FFN gate/up (op 47) | N=3072, K=6144, layers 0–2 only | yes |
| dense FFN down / DSA indexer (op 44) | N=6144 K=3072 / N=4096 K=2048 | yes |
| q_a+kv_a+k_rope (GemvQkv fusion A) | Nq2048 Nk512 Nv64, K=6144 | **no, bf16** |
| q_absorb+q_rope (GemvQkv fusion G) | Nq8192 Nk1024, K=2048 | **no, bf16** |
| o_proj (Gemv) | N=6144, K=4096 | **no, bf16** |
| shared expert (GemvGlu/Gemv) | N=512 K=6144 / N=6144 K=512 | **no, bf16** |
| lm_head (Gemv) | N=154880, K=6144, **NOT TP-sharded** | no (bf16 on disk) |
| FlashMlaDecode | DK512 DR64, GF=2, nh_l 16, nsplit 16 | n/a |
| MlaMergeFold | nh_l 16, DK 512, V 256, VT **256**, ns 16 | n/a |
| HeadNormRope INTERLEAVE | hd=64, nhead=**16** (q) and **1** (k_rope) | n/a |
| RmsNorm / Residual | feat 6144 / 2048 / 512, on **ONE workgroup** | n/a |

### 1a. The weight stream is 19.8 GB/rank/token, not the 9.9 GB §6g assumed

`scripts/glm52_prep.py` **dequantises every attention weight and the shared expert to bf16**
(lines 172–214: `p_bf16(dequant_blockfp8(... q_a_proj ...))`, same for `o_proj`, all three
`shared_experts.*`; `q_absorb`/`v_absorb`/`kv_a_latent`/`k_rope` are *derived* products so they are
bf16 by construction). Only the routed experts, the dense-layer FFN and the DSA indexer stay fp8.

| per sparse layer, per rank | MB |
|---|--:|
| attention (all bf16) | 124.5 |
| router (bf16) + shared expert (bf16) | 22.0 |
| routed 8 experts (fp8) | 75.5 |
| **layer total** | **222.0** |

`75×222.0 + 3×181.1 + lm_head 1903 + KV latent 736` ⇒ **19.84 GB/rank/token → 3.20 ms floor**,
not the 1.6 ms in §6g. An all-fp8 GLM with a TP-sharded lm_head would stream 12.85 GB → **2.07 ms**.
So the missing block-fp8 routing on the attention path is worth **1.13 ms of floor** — real, but
3.6 % of today's token. **Do not sell it as the lever; sell it as the thing that has to be true
before the floor matters.**

---

## 2. Static evidence — occupancy, registers, LDS, load schedule

`hipcc -Rpass-analysis=kernel-resource-usage`, the exact flags `scripts/build_glm52_decode.sh:45`
uses (`-DPLOW_BUCKET_DECODE=1`).

**The GLM decode megakernel:** `VGPR 248, AGPR 0, occupancy 2 waves/SIMD, VGPR spill 0,
SGPR spill 76, scratch 136 B/lane, LDS 147 464 B/block.`

Standalone, each op compiled as its own `__global__` (`/tmp/glm_kern_probe.hip`):

| kernel | VGPR | own occ | loads in flight per wave (disasm) |
|---|--:|--:|--:|
| `d_moe_expert_glu_fp8_blk` (45) | 42 | 8 | **3** (1 weight + 2 x) |
| `d_moe_expert_down_fp8_blk` (46) | 36 | 8 | **3** |
| `d_moe_group_glu_fp8_blk` (48) | 51 | 8 | 3 |
| `d_dense_glu_fp8_blk` (47) | 41 | 8 | 4 |
| `d_gemv_fp8_blk` (44) | 142 | 3 | **18** |
| `d_gemv` bf16 (10) | 72 | 7 | **16** |
| `d_flash_mla_decode` GF=4 / GF=2 | 138 / 128 | 3 / 4 | 4 / 8 |
| `d_mla_merge_fold` | 21 | 8 | 4 |
| `d_headnorm_rope<64,true>` | 22 | 8 | 2 |

Three things follow, and they are the frame for everything below.

**(a) Decode runs at 2 waves/SIMD and that is a *launch-geometry* fact, not a register fact.**
grid == n_cu × 512 threads ⇒ one workgroup of 8 waves per CU ⇒ 2 waves/SIMD, whatever the
register count. Getting to 4 waves/SIMD needs either 1024 threads/WG or 2 WGs/CU, and **both need
VGPR ≤ 128** (512 VGPRs per SIMD lane-slot / 4 waves). Today the megakernel is at **248**. So the
register cliff is real, but it is the barrier to *doubling* occupancy, not the cause of today's
numbers. The 2-WG/CU route additionally needs LDS ≤ 80 KB.

**(b) The decode object carries 118 KB of dead LDS.** `PLOW_SMEM_HALVES` is
`max(PLOW_GM_ARENA, FA_LDS_HALVES(512))` and `PLOW_GM_ARENA = GM_LDS_HALVES_T(256,256,64)` =
147 464 B — the **prefill** GEMM tile. In the decode bucket `PLOW_BUCKET_PREFILL == 0`, so the GEMM
arms are `#if`'d out and never dispatched; the largest real decode consumer is
`FA_MLA_DEC_LDS_FLOATS(512,64,4)` = **29 248 B** (measured on the standalone MLA kernels). Harmless
today (grid == n_cu means 1 WG/CU regardless) but it is the *second* of the two locks on a 2-WG/CU
experiment, and it is free to remove.

**(c) The MoE expert kernels have 1/6 the memory-level parallelism of the GEMV that does the same
arithmetic.** `wave_dot_fp8_blk` issues one `buffer_load_dwordx4` and waits; `gemv_rows_fp8_blk`
keeps 2 output channels × UN∈{2,3,6,8} = 12–18 loads outstanding. That is the whole structural
difference between op 45/46 and op 44.

### 2a. What "loads in flight" is worth — measured

Pure coalesced fp8 stream, interpreter geometry, 604 MB arena (past the 256 MB LLC):

| loads in flight per wave | grid 28 | grid 256 |
|---|--:|--:|
| 1 | 482 GB/s | **4006 GB/s (64.6 %)** |
| 3 | 1144 | **6420 GB/s (103.5 %)** |
| 8 | 1644 | 5786 (93.3 %) |

**One load in flight caps the machine at 65 % of HBM peak; three saturate it.** This is the
reference every "% of roofline" below is read against.

---

## 3. Measured: per-kernel verdict table

`us` = kernel body only, on an idle GPU, launch overhead removed (empty-kernel floor of this
harness is 2.4–2.5 µs; the sub-µs ops were loop-amortised inside one launch). Roofline = bytes the
kernel must move / 6200 GB/s. `ms/token` = us × (packets of this kind per token).

| # | kernel (opcode) | shape | grid | µs | roofline µs | % of roofline | ms/token | binding constraint | headroom |
|---|---|---|--:|--:|--:|--:|--:|---|---|
| 1 | **MlaMergeFold** (57) | nh16 DK512 V256 VT256 ns16 | 16 of 256 | **23.10** | 0.68 | **2.9 %** | **1.80** | 16 work items × 256 of 512 threads; one thread does a 512-long scalar dot | **YES, large** |
| 2 | **MoeExpertDownFp8Blk** (46) | N6144 **K512**, 8 experts | 8×28 | **25.94** | 4.06 | **15.6 %** | **1.95** | K=512 ⇒ 1 chunk, lanes 32–63 dead, 1 load in flight | **YES** |
| 3 | **FlashMlaDecode** (50) | GF2 nh16 ns16 ctx1024 | 128 of 256 | **15.83** | 1.52 | **9.6 %** | **1.24** | 64 KV rows into a 512-row tile; 128 of 256 wgs | **YES (short ctx only)** |
| 4 | **MoeExpertGluFp8Blk** (45) | N512 K6144, 8 experts | 8×28 | **24.72** | 8.12 | **32.8 %** | 1.85 | 8×28=224 of 256 wgs; 2.29 channels/wave imbalance; 1 load in flight | YES, moderate |
| 5 | Gemv bf16 o_proj (10) | N6144 K4096 | 256 | 9.73 | 8.12 | **83.4 %** | 0.76 | — at the ceiling | **no** |
| 6 | GemvQkv fusion A (22) | Nq2048 Nk512 Nv64 K6144 | 256 | 6.27 | 5.20 | **83.0 %** | 0.49 | — | **no** |
| 7 | GemvQkv fusion G (22) | Nq8192 Nk1024 K2048 | 256 | 8.54 | 6.09 | **71.3 %** | 0.67 | — | little |
| 8 | Gemv bf16 lm_head (10) | N154880 K6144 | 256 | 288.2 | 306.9 | **106.5 %** | 0.29 | — at/over the ceiling | **no** (shard it, see §5) |
| 9 | DenseGluFp8Blk (47) | N3072 K6144, 3 layers | 256 | 16.83 | 6.09 | 36.2 % | 0.05 | 3072 channels = 3072 waves > 2048, fine; 1 load in flight | not worth it |
| 10 | RmsNorm h=6144 (1) | 1 workgroup | 1 | **1.29** | 0.004 | 0.3 % | ~0.30 | nothing — body is 1.3 µs, the packet costs ~5.9 µs (§6a) | **no — gate, not kernel** |
| 11 | Residual n=6144 (4) | 1 workgroup | 1 | 1.02 | 0.006 | 0.6 % | ~0.15 | same | **no** |
| 12 | HeadNormRope hd64 nh16 / nh1 (3) | 256 wgs, 16 / **1** wave of work | 256 | **0.32 / 0.29** | ~0 | 0.2 % | ~0.05 | same — body is 300 ns | **no** |

Notes on rows 1–4, with the arithmetic that proves the diagnosis:

* **Row 1.** `n_work = n_batch × nh_l × ceil(V/VT) = 1 × 16 × 1 = 16`. Inside a workgroup the fold
  loop is `for v = v0+tid; v < v1; v += 512` with `v1−v0 = 256`, so **256 of 512 threads** run, each
  executing a serial 512-iteration dot. Active waves = 16 × 4 = **64 of 2048 = 3.1 %**; measured
  **2.9 % of ceiling**. The model closes exactly. Confirmed grid-invariant: **23.15 / 23.08 / 23.10 µs
  at grid 16 / 64 / 256**, and **VT=64 gives 23.31 µs** — i.e. VT is the wrong axis, which is why
  the header's "swept empirically" sweep found nothing. This kernel is 34× above its roofline and
  costs 1.80 ms/token, **independent of context length**.
* **Row 2.** `step = 64 lanes × 16 fp8 = 1024`, but K = `imoe/tp` = **512**, so `nchunk = 1` and
  lanes 32–63 read past `num_records` and contribute zero — **half the wave is structurally dead**,
  and each wave has exactly one 512-B useful load outstanding. Layout barely matters:
  serial-all-256 23.86 µs, coresident 8×28 25.94, coresident 8×32 22.72. The comment at
  `op_moe.h:246` reasons about this op as "K=I_moe=2048 → nchunk=2"; **at TP4 it is 512 and nchunk=1**,
  which is why the UN-unroll experiment recorded there could not help — UN has no chunks to unroll.
* **Row 3.** `n_work = (nh_l/GF) × nsplit = 8 × 16 = 128` of 256 workgroups, and each split covers
  `ctx/nsplit = 64` KV rows into a `FA_DEC_TILE = 512`-row tile ⇒ 64 of 512 threads score-active.
  Active fraction ≈ 6.25 %, measured 9.6 %. **At ctx 32768 the same kernel measures 121.8 µs =
  40.0 % of ceiling** (ns=64 ⇒ exactly 512 rows/split, a full tile). The kernel is fine; `glm_nsplit`'s
  short-ctx floor is what starves it.
* **Row 4.** Under the shipping `GLM_MOE_CORESIDENT=2`, `expert_parts = tk+1 = 9` ⇒ 28 CUs/expert ⇒
  224 of 256 wgs, and 512 channels over 224×8 = 1792 waves gives 2.29 channels/wave (some waves do
  3, some 2 — a 1.31× straggler). At **8×32 CU** (i.e. `expert_parts = tk`) it measures **19.58 µs,
  41.5 %** — 512 channels over exactly 2048 waves, 2 each, no imbalance.

---

## 4. The MoE slice-map A/B — the number that validates this whole bench

One GLM sparse layer, routed experts only, top_k 8, TP4, 4.83 GB arena:

| layout | grid | µs/layer | GB/s | % ceil |
|---|--:|--:|--:|--:|
| GLU coresident 8 experts × 28 CU (**shipping, CORESIDENT=2**) | 224 | 24.72 | 2036 | 32.8 % |
| GLU coresident 8 experts × 32 CU (`expert_parts = tk`) | 256 | **19.58** | 2571 | 41.5 % |
| GLU serial, 8 × 256 CU (`CORESIDENT=0`) | 256 | 65.93 | 763 | 12.3 % |
| GLU grouped op48, **slot-outer (the shipped body)** | 256 | **66.00** | 763 | 12.3 % |
| GLU grouped op48, `PLOW_MOE_GROUP_FLAT=1` | 256 | **20.54** | 2450 | 39.5 % |
| DOWN coresident 8 × 28 CU | 224 | 25.94 | 970 | 15.6 % |
| DOWN coresident 8 × 32 CU | 256 | **22.72** | 1108 | 17.9 % |
| DOWN serial 8 × 256 CU | 256 | 23.86 | 1055 | 17.0 % |
| DOWN grouped op49, slot-outer | 256 | 24.03 | 1047 | 16.9 % |
| DOWN grouped op49, `PLOW_MOE_GROUP_FLAT=1` | 256 | **49.26** | 511 | 8.2 % |

Rooflines for this layer: GLU **8.12 µs**, DOWN **4.06 µs**.

Three results:

1. **`GLM_GROUP=1` is mechanically identical to the serial layout** — 66.00 vs 65.93 µs, 0.1 %
   apart. The grouped body (`op_moe.h:392`) is literally `for slot in 0..k { per_slot_kernel }` on
   all 256 CUs. Cost vs coresident = (66.00−24.72)+(24.03−25.94) = 39.4 µs/layer × 75 =
   **+2.95 ms**. §6g-KNOBS measured **+2.88 ms** on the full model. **The kernel bench reproduces
   the full-model knob to 2 %.** Treat the rest of this table as trustworthy.
2. **`PLOW_MOE_GROUP_FLAT=1` has never been measured and it is half-right**: it fixes the grouped
   GLU (66.00 → 20.54, matching coresident) and **breaks the grouped DOWN** (24.03 → 49.26, 2×
   worse). The DOWN regression is the flat sweep re-resolving `wtab[eid*3+2]`, `stab[...]` and
   `moe_slot_gate` per *output* — 3 dependent scalar loads against a kernel that only has **1**
   weight load per output at K=512. Do not enable it as a pair; if op-count ever matters, use FLAT
   for the GLU and slot-outer for the DOWN.
3. **The shipping 9-way split costs 8.36 µs/layer = 0.63 ms/token** relative to an 8-way split
   (GLU 24.72→19.58, DOWN 25.94→22.72). `GLM_MOE_CORESIDENT=2` buys the co-resident shared expert
   but pays for it by shrinking every routed expert's slice from 32 CUs to 28. The knob table
   records the net as −3.00 ms, so the overlap is worth more than 0.63 ms — but **0.63 ms is on the
   table if the shared expert can be given CUs without taking them from the routed ones.**

---

## 5. Top 3 worth optimising, RANKED BY MS RECOVERABLE

Ranked by ms, not by distance from roofline. (Row 9 is at 36 % of roofline and costs 0.05 ms —
not worth touching. Rows 10–12 are at 0.3 % of roofline and are *correct as they are*.)

### #1 `MlaMergeFold` — 1.80 ms today, ~0.15–0.30 ms achievable, **≈ 1.5 ms recoverable**

*Evidence it is not at its limit:* 2.9 % of ceiling, 34× above roofline, and **grid-invariant**
(23.15 / 23.08 / 23.10 µs at 16 / 64 / 256 workgroups) — the machine is 97 % idle while it runs, and
adding CUs changes nothing. Active waves 64 of 2048.

*Root cause, precisely:* the W_uv fold gives **one thread** the whole `DK = 512` reduction
(`op_attention.h:2256`: `for l in 0..DK: acc += olds[l] * bf2f(wv[l*V+v])`). Output space is
`nh_l × V = 16 × 256 = 4096` values; one thread each caps the kernel at 4096 active lanes on a
131 072-lane machine. **VT cannot fix this** — VT only redistributes the same 4096 threads across
workgroups, which is exactly what the grid-invariance shows.

*The fix is already in this codebase, one file away.* `wave_dot_fp8_blk` (`op_moe.h:237`) records
that making the K-reduction wave-cooperative instead of one-thread-per-output was "the ~1000× lever"
for the expert body. The same transform here gives 4096 outputs × 64 lanes = 262 144 lane-slots ≈ 2×
the machine, 8 elements per lane, one `wave_sum`. W_uv reads stay coalesced (consecutive `v` are
contiguous). Alternatively this is a tiny batched matvec (16 × [512×256]) and is MFMA-shaped.

*Risks:* none of the §6b-i "widening" hazard applies — this does **not** widen a producer whose
consumer has a dense edge; it changes the *internal* thread mapping of one op. Workgroup count can
stay 16 if desired.

### #2 `MoeExpertDownFp8Blk` — 1.95 ms today (1.70 at an 8-way split), **≈ 1.1–1.3 ms recoverable**

*Evidence it is not at its limit:* 15.6 % of ceiling, 6.4× above roofline, and essentially
layout-invariant (23.86 serial / 25.94 at 28 CU / 22.72 at 32 CU) — the loss is inside the wave,
not in the slice map.

*Root cause, precisely:* at TP4 `K = imoe/tp = 512`, but `wave_dot_fp8_blk`'s pass is
`64 lanes × 16 fp8 = 1024`. So **`nchunk = 1`, lanes 32–63 read past `num_records` and add zero**,
and each wave has exactly one useful 512-byte load outstanding. Against the §2a curve (1 load in
flight → 4006 GB/s at full occupancy) with 2048 waves × 512 useful bytes in flight, the predicted
ceiling for this form is ~2100 GB/s; measured 1108. The extra ~2× is the per-chunk `srow[kb]`
dependent scalar load plus the DPP `wave_sum` chain, both of which are amortised over one chunk
instead of six.

*Two independent fixes, both local to the existing kernel:*
(a) make the per-lane chunk width adaptive — 8 fp8/lane when `K < 1024` — so all 64 lanes are live;
(b) adopt the `has2` idiom from `gemv_rows_fp8_blk` (`op_gemm.h:1648`): **two output rows per wave**,
which doubles bytes in flight without depending on K at all. `H = 6144` outputs over 2048 waves is
3 rows/wave, so there is plenty of row parallelism to pair up.
This is the case the recorded negative result at `op_moe.h:243-249` did *not* test — it tested UN
(chunk unroll), and at nchunk=1 there is nothing to unroll.

### #3 `FlashMlaDecode` at short context — 1.24 ms today, **≈ 0.8–1.0 ms recoverable**

*Evidence it is not at its limit — and that the kernel is fine:* the **same kernel at ctx 32768
measures 40.0 % of ceiling** (121.8 µs, ns=64, one full 512-row tile per split). At ctx 1024 it is
9.6 %, because `nsplit = 16` divides 1024 rows into 64-row splits against `FA_DEC_TILE = 512`.
`n_work = (nh_l/GF) × nsplit = 8 × 16 = 128` of 256 workgroups.

*Root cause:* `glm_nsplit`'s `NS_FLOOR = 16` (`mla.rs:212`). Its cost model
(`mla.rs:190-201`) balances the decode saving against **"plow's FlashMerge is a SEPARATE O(nsplit)
pass"** whose growth eats it. **That premise no longer holds on this path**: GLM uses the *fused*
`MlaMergeFold`, and this bench measures it at 23.1 µs **independent of grid and of VT**. Re-derive
`glm_nsplit` against the fused merge (sweep ns ∈ {2,4,8,16} at ctx ≤ 4k), and expect the short-ctx
floor to come down. This is a **tuning** change in the emitter, not a kernel change — cheapest of
the three.

*Caveat:* `nsplit` also sets `Opart`/`mlpart` size and the merge's inner `for s in 0..nsplit` loops;
verify the merge does not regress in the same sweep.

**Total for the top three: ≈ 3.3–3.9 ms.** 31.1 → ~27.5 ms. vLLM chat-backend TPOT is 22.48.
This is worth doing and it does not win the model.

---

## 6. What is ALREADY GOOD — do not spend time here

| kernel | number that proves it |
|---|---|
| **`Gemv` bf16, wide N** | lm_head N=154880 K=6144: **288.2 µs = 6604 GB/s = 106.5 % of the 6200 ceiling.** o_proj N=6144 K=4096: **83.4 %.** 16 loads in flight per wave. Nothing to win. |
| **`GemvQkv` fusions A and G** | 83.0 % and 71.3 % of ceiling. The design is *right*: fusion A concatenates the N=64 `k_rope` and N=512 `kv_a` columns into one N=2624 GEMV, which is exactly the fix for the CU starvation those skinny projections would otherwise cause. Do not un-fuse; do not "improve" the skinny arms separately. |
| **`gemv_rows_fp8_blk`'s K-adaptive unroll** (`op_gemm.h:2310-2314`) | UN ∈ {2,3,6,8} chosen so it divides `nchunk`. 18 loads in flight measured in the disassembly. This is the reference implementation of MLP that op 45/46 should be judged against. |
| **`HeadNormRope<64, INTERLEAVE=true>`** | body **0.29–0.32 µs**. It is 20× below a packet's gate cost. INTERLEAVE is confirmed correct (§6g) and the `__shfl_xor(.,1,64)` partner fetch is the only cross-lane traffic RoPE needs. The §6c/L4 "kill the dependent `pos` load" lever is worth **at most 0.3 µs/packet**. **Leave it alone.** |
| **`RmsNorm` / `Residual` on 1 workgroup** | bodies **1.29 / 1.02 µs**; §6a prices the same class at ~5.9 µs/packet inside the interpreter, so **~80 % of a narrow packet is gate + dispatch, not kernel**. Measured directly: RmsNorm h=6144 is **1.288 µs at grid 1 and 1.259 µs at grid 256** — widening buys 2 %, and §6b-i showed widening actively *costs* when the consumer's edge is dense. Emitting them on 1 workgroup is correct. |
| **`GLM_MOE_CORESIDENT` ≥ 1** | serial 65.93 → coresident 24.72 µs/layer on the GLU. Concurrency is worth 2.7× here. §6g-KNOBS' "op count is not the objective function" is confirmed at the kernel level. |
| **The block-fp8 scale handling** | the per-128-block f32 multiply is one FMA per chunk with no cross-lane reshuffle (a lane's 16 fp8 lie inside one 128-K block). Scale-grid indexing is **not** a measurable cost anywhere in this table — the DOWN op's 1-chunk cost is the *dependent scalar load*, not the indexing arithmetic. Also correct: the scale is deliberately **not** folded into `v_cvt_scalef32_*` (E8M0-only, ~22 % error — `amd_common.h`). |
| **`MoeExpertGluFp8Blk`'s two-call gate/up form** | the header records a hand-fused gate+up loop miscompiling in the megakernel; the x re-read is an L2 hit while the weights (the bandwidth term) are read once either way. Confirmed harmless: at the same shape and grid, op 44 with LDS x-staging and 18 loads in flight gets **742 GB/s** vs op 45's **810 GB/s** — the MoE kernel is already the *faster* of the two here. |

---

## 7. Not-a-kernel findings that the audit turned up

Recorded because they change what "optimising the kernels" is worth, not because they are kernel work.

1. **~21 of the 31.1 ms is not kernel body.** Sum of §3: routed GLU 1.85 + routed DOWN 1.95 +
   MlaMergeFold 1.80 + FlashMlaDecode 1.24 + QkvA 0.49 + QkvG 0.67 + o_proj 0.76 + dense GLU 0.05 +
   lm_head 0.29 + narrow ops ~0.53 + shared/router ~0.53 (estimated, not measured) ≈ **10.2 ms**.
   Collectives are 3.84 ms (§6e-0, measured). The residual ~17 ms is the 2756-packet gate/dispatch
   chain. *Caveat: these bodies are isolated-GPU timings with no counter gates and no cross-rank
   sync, so they are a lower bound on in-interpreter cost — which makes the residual an upper bound,
   not an exact figure. The sibling's `PLOW_TRACE_RAW` attribution is the instrument that settles it.*
2. **lm_head is not TP-sharded.** `mla.rs:380` binds `lm_head.weight` at full `vocab × hidden` and
   `mla.rs:1536` emits `d.i[1] = c.vocab` with no `/tp`. Every rank streams the same 1.903 GB every
   token, measured at **288 µs**. Column-sharding it (38 720 logits/rank + a 4-way (value,index)
   reduce, which is smaller than any collective already in the program) recovers **~0.22 ms**. The
   kernel is at 106 % of ceiling — this is purely a sharding gap.
3. **Block-fp8 still has no prefill arm — VERIFIED, unchanged.** The six `*Fp8Blk` opcodes (44–49)
   are all M=1 GEMV/expert shapes; there is **no `GemmFp8Blk`**. The only fp8 GEMM,
   `d_gemm_fp8_t` (`op_gemm.h:992`), takes a per-output-channel `wscale`, which structurally cannot
   express `weight_block_size: [128,128]`. §6g's served TTFT is **37.9 s vs vLLM's 1.9 s** because
   1024 prompt tokens run as 1024 decode dispatches. **This is a 20× deficit against a 1.4× decode
   deficit** — it dwarfs every kernel item in this document, and it is the one place where "the
   kernel inventory is not enough" is unambiguously true.
4. **The GLM build scripts skip the register-cliff gate.** `build_glm52_decode.sh:45` and
   `build_glm52_trace.sh` call `hipcc --genco` directly; only `build_gfx950.sh` has `check()`.
   Nobody was measuring the GLM decode object's VGPR/occupancy. It is **248 / occ 2 / 0 VGPR spill**
   (§2) — inside the cliff, but with 8 registers of headroom, which is worth knowing before anyone
   adds an arm.
5. **118 KB of the decode object's 144 KB LDS is the prefill GEMM arena** and the GEMM arms are
   `#if`'d out of that bucket (§2b).

---

## 8. Method, and what would falsify this

* Every kernel timing is the production `__device__` body, included from the real headers, at
  `blockDim = 512` and the emitter's real workgroup count. Weight arenas are 0.6–4.8 GB so the
  256 MB LLC cannot hold them; expert sweeps walk 64–512 distinct experts.
* Sub-µs ops were loop-amortised inside one launch; the empty-kernel floor of this harness is
  **2.4–2.5 µs** and is not in any reported body number.
* **Known limitation:** these kernels are compiled standalone (VGPR 21–142) while in the
  interpreter they are inlined into a 248-VGPR function. Register allocation could change the load
  schedule in either direction. It cannot change the *slice-map* arithmetic — "16 of 256 workgroups",
  "half the lanes read past K", "64 rows into a 512-row tile" are properties of the emitted shape,
  and those are what every #1–#4 diagnosis rests on. The proof that the standalone numbers track
  reality is §4: the kernel-level `GLM_GROUP` A/B reproduces the full-model knob to 2 %.
* **What would falsify #1–#3:** the sibling's `PLOW_TRACE_RAW` per-op attribution showing
  `MLA_MERGE_FOLD`, `MOE_EXPERT_DOWN_FP8_BLK` or `FLASH_MLA_DECODE` at materially less than
  1.8 / 1.9 / 1.2 ms per token. If it does, the slice-map arithmetic above is still right about
  *why* they are slow — it would only mean they overlap with something else and the recoverable ms
  is smaller.

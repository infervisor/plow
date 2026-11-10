# sm_90a — NVIDIA Hopper (H100 SXM; H200 shares the ISA)

| | |
|---|---|
| `IsaLevel` | `IsaLevel::Sm90a` (`crates/hwspec/src/isa.rs`), arch flag `sm_90a`, `compute_cap (9, 0)`; caps: `mma_sync`, `wgmma`, `tma`, `dsm_cluster`; no `tcgen05`/TMEM/block-scale MMA/E8M0 convert; dtypes fp16/bf16/fp8/int8 |
| `GpuSpec` | `H100_SXM5` (shipped), `H200_SXM`, `H100_PCIE`, `H100_NVL` in `crates/hwspec/src/nvidia/h100.rs`; script mirror `scripts/campaign/roofline.py` `HARDWARE_REGISTRY["h100 sxm5"]` |
| SKUs plow ships | H100 SXM 80 GB only: every `recipes/infervisor/*/sm90a-h100-tp1*.toml` (Gemma-4 12B/26B-A4B/31B/E4B, Qwen3.8-27B, Qwen3-ASR, Nemotron 3.5 ASR, Silero VAD, Orpheus, Veena, Chatterbox(+MTL), Fish S2 Pro, Breeze TTS 2, Maya2). H200 has no recipe; hwspec, `costmodel` and `roofline.py` model it and it reuses the sm_90a objects. |
| Kernels | `runtime/nvidia/`: `interp_sm90a*.cu` (interpreter + segment objects; `interp_sm90a.cu` is an alias wrapper around `interp_sm120.cu` with `PLOW_NV_HOPPER=1`), `sm90_wgmma.cuh` (primitives), `op_gemm_sm90.cuh`, `op_attention_sm90*.cuh`, `gen_attn_pf_hd256_*`, `op_moe_sm90.cuh`, `op_gemv_*_sm90.cuh`/`op_gemv_fp8_wgmma.cuh`, `gemm_fp8_block_sm90.cu`, `register_hopper.cu` (golden dispatch table only); probes in `runtime/nvidia/experiments/` (not built) |
| Tuning records | `tuning/nvidia/sm_90a/h100-sxm5/` (`cublaslt_algos.jsonl`, `attention_role_measurement.jsonl`, `gen_kernels.json`), `tuning/nvidia/sm_90a/h100-nvl/` |

Verified against: PTX ISA 9.4 (docs.nvidia.com, last updated 2026-10-12), Hopper Tuning Guide v13.4,
CUDA Driver API 13.4 (Tensor Map Object Management), NVIDIA H100 and H200 product pages; fetched 2026-11-03.

Code paths are on `prod-next`. `experiments/README.md` Hopper rows: H100 NVL capped at 310 W, use
as ratios only; its `t3_pipe_probe.cu` and `gemv_transport.cu` rows are sm_120a.

## 1. Execution model

| quantity | H100 SXM | source |
|---|---|---|
| warp / warpgroup | 32 lanes / 4 contiguous warps = 128 threads (first warp index % 4 == 0) | [vendor documented: PTX ISA §9.7.17 "warpgroup"] |
| SMs | 132 in 8 GPCs (16.5/GPC average, so some GPCs have < 18; hwspec models a full 18-SM GPC). Query `cudaOccupancyMaxActiveClusters` before relying on cluster size 16 | `h100.rs`; [measured in plow: docs/runtime/attention-roofline.md "All 132 blocks"] |
| tensor cores | 4 per SM, 4th gen | `HOPPER_SM.tensor_cores` |
| register file | 64 K x 32-bit per SM; max 255 regs/thread | [vendor documented: Hopper Tuning Guide §1.4.1.1] |
| reg alloc granularity | 8 regs/thread (256/warp) | `cuda_occupancy.h` `cudaOccRegAllocationGranularity` |
| threads / warps / blocks per SM | 2048 / 64 / 32 | [vendor documented: CUDA PG compute-capability table; Tuning Guide §1.4.1.1 (warps, blocks)] |
| shared memory | 228 KB/SM; 227 KB max per block (CUDA reserves 1 KB per block); carve-outs 0/8/16/32/64/100/132/164/196/228 KB; static smem is capped at 48 KB, more needs dynamic opt-in | [vendor documented: Tuning Guide §1.4.1.1, §1.4.2.4] |
| L1 + smem | 256 KB unified per SM | [vendor documented: Tuning Guide §1.4.2.4] |
| smem banks | 32 x 4 B | [vendor documented: CUDA C++ Programming Guide, shared memory] |
| smem bandwidth | ~128 B/clk/SM (32 banks x 4 B/clk), as the wgmma GEMM design assumes (UNVERIFIED for CC 9.0) | `op_gemm_sm90.cuh` T15 note |
| L2 | 50 MB (hwspec: 8 x 6.25 MiB partitions, ~1.5 TB/s each, modelled) | [vendor documented: Tuning Guide §1.4.2.2]; partitioning (UNVERIFIED) |
| HBM | 80 GB HBM3, 5120-bit, 3.35 TB/s | [vendor documented: H100 product page] |
| clocks | max SM 1980 MHz; 1830 MHz read once before the calibration runs, not a sustained GEMM clock (not measured) | [measured in plow: `calibrate_ceilings.py:67` output, 2026-10-29] |
| clusters | 8 CTAs portable, 16 non-portable (`cudaFuncAttributeNonPortableClusterSizeAllowed`) | [vendor documented: Tuning Guide §1.4.1.3] |
| H200 | same SM; 141 GB HBM3e, 4.8 TB/s; L2 not in the vendor spec (hwspec 50 MiB, UNVERIFIED: query `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`) | [vendor documented: H200 product page] |

Occupancy facts plow lives with:
* The interpreter objects are persistent kernels at **256 threads = 2 warpgroups** by default
  (`PLOW_NV_THREADS`). Exceptions: the WS384 segment objects launch 384 threads (exported as
  `plow_block`, `interp_sm120.cu`) and the GEMV512 role launches 512 (`interp_sm90a_gemv512.cu`). Its register count is the maximum over **all inlined arms**
  (`crates/kernelcaps/src/resource.rs`). CUDA 13.0: decode 208 regs before dead-arm gating, 177
  after; prefill 255 regs, 180 B spill stores, 644 B spill loads; 1 block/SM either way. Re-read
  `cuobjdump --dump-resource-usage` before relying on these
  [measured in plow: resource.rs header; build_sm90a_cubin.sh dead-arm comment].
* Two blocks/SM means at most ~113 KiB dynamic smem per block (228 KiB/SM, 1 KiB reserved per
  block); the per-block cap is 232448 B (227 KiB) [vendor documented: Tuning Guide §1.4.1.1]. The lean SEG_GEMM object uses `__maxnreg__(128)` to get
  2 blocks/SM.
* A larger dynamic-smem claim steps the L1 carve-out down. Once L1 drops to 28 KiB, flash-decode
  slows ~1.7x [measured in plow: runtime/nvidia/op_gemv_wide_sm90.cuh header].

## 2. Matrix / tensor instructions

### wgmma.mma_async (sm_90a only; the only path to the Hopper tensor-core rate)

| A/B type | shape | N | D types | A from regs | transpose |
|---|---|---|---|---|---|
| `.f16` / `.bf16` | `m64nNk16` | 8..256 step 8 | f32 (f16 also for `.f16`) | yes (4 x b32/thread) | `imm-trans-b` in SS and RS; `imm-trans-a` SS only (register A cannot transpose) |
| `.tf32` | `m64nNk8` | 8..256 step 8 | f32 | yes | no (K-major) |
| `.e4m3` / `.e5m2` (mixable) | `m64nNk32` | 8..256 step 8 | f32, f16 | yes | **no (K-major only)** |
| `.s8` / `.u8` | `m64nNk32` | 8,16,24,32,48,64,80,...,256 (step 16 above 32) | s32 (`.satfinite` optional) | yes | no |
| `.b1` (`.and.popc`) | `m64nNk256` | 8,16,24,32,48,64,...,256 (step 16 above 32) | s32 | yes | no |

[vendor documented: PTX ISA 9.4 §9.7.17.5.2]

* Syntax (SS): `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 {d..}, a-desc, b-desc, p-scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;`
  fp8 and tf32 drop the trans immediates but keep `imm-scale-a/b`. int8 and b1 have neither: the
  operand list ends at `scale-d` (negation is floating-point only). RS form: `{a0..a3}` replaces
  `a-desc`, and for f16/bf16 only `imm-trans-b` is left. `scale-d` false means D = A·B. Plow uses it to seed accumulators without a
  zeroing pass (`sm90_wgmma.cuh`). imm-scale = -1 negates the operand.
* Accumulator: N/2 f32 per thread (m64n128: 64, m64n256: 128). Plow-verified layout, the same for
  bf16 k16 and e4m3 k32 [measured in plow: sm90_wgmma.cuh `sm90_acc_row/col`]:
  warp w of the warpgroup owns rows `[16w,16w+16)`. For n-block g: reg `4g+2hi+lo`,
  row `16w + lane/4 + 8hi`, col `8g + 2(lane%4) + lo`. Per 8 columns this is the same pattern as an
  `mma.m16n8k16` C fragment, so mma.sync epilogues port unchanged.
* After softmax, pack each 16-column f32 S slice into 4 bf16x2 registers (`fa3_pack_bf16`): that is
  the RS A fragment, so P·V runs in RS form with no smem round trip (`op_attention_sm90_v3.cuh`,
  `gen_attn_pf_hd256_wgmma.inc:hd256_pv_bn32`).
* FP8 precision: with f32 D, accumulation is "higher than half, lower than single precision"
  [vendor documented: PTX §9.7.17.5.2 wgmma.mma_async Note]. Plow measured relL2 3.9e-5 at K=32 and
  1.14e-3 at K=3840. Promoting into an f32 shadow every 128 K cuts this ~10.9x for 8-16%
  throughput [measured in plow: sm90_wgmma.cuh header, `PGM90_FP8_PROMOTE`].
* Throughput: dense fp8/int8 = 2x bf16 [vendor documented: H100 page, "with sparsity" figures / 2].
  Per-shape issue latency (UNVERIFIED; not in PTX ISA).

### Shared-memory matrix descriptor (64-bit, uniform across the warpgroup)

| bits | field |
|---|---|
| 13:0 | start address `(addr & 0x3FFFF) >> 4` |
| 29:16 | LBO >> 4 |
| 45:32 | SBO >> 4 |
| 51:49 | base offset: 0 if the swizzle pattern starts on its natural boundary (1024/512/256 B for 128/64/32 B swizzle), else `(pattern_start >> 7) & 7` |
| 63:62 | swizzle: 0 none, 1 = 128 B, 2 = 64 B, 3 = 32 B |

[vendor documented: PTX ISA §9.7.17.5.1.2.2 "Matrix Descriptor Format"]. smem operand start must be
16 B aligned. For K-major swizzled layouts LBO is unused (assumed 1). SBO = stride between 8-row
groups. For MN-major swizzled layouts the fields mean something else, not a plain swap, and rows and
columns count 128-bit-normalized elements: LBO = offset between groups of (swizzle bytes / 16)
rows, SBO = offset between 8-column groups (PTX §9.7.17.5.1.2.1.1-2).

**Plow's 128 B recipe** (`sm90_wgmma.cuh`; other encoders: `fa90_desc_ns` swizzle 0, `snac.cu:wg_desc` tf32):
* 128 B swizzle, K-major (`trans_a = trans_b = 0`), LBO = 16 B, SBO = 1024 B, swizzle = 1,
  base offset 0.
* One logical row = 128 B = 64 bf16 or 128 e4m3. That fixes BK = 64 (bf16) / 128 (fp8).
* Store-side XOR: `row*BK + ((chunk ^ (row & 7)) * elems_per_16B)` (`sm90_swz_off`).
* A k16/k32 substep advances the **start address by +32 B only**.
* Tile base **1024 B aligned** (`sm90_align1024`; `extern __shared__` alone does not guarantee
  it).
* MN-major V (P·V, `trans-b = 1`) keeps LBO = 16, SBO = 1024 under 128 B swizzle
  [measured in plow: op_attention_sm90.cuh; the swapped guess gave relL2 1.0].
* Register-written tiles with rows < 128 B (Ps, BKV = 32) use swizzle mode 0 core-matrix packing
  (`op_attention_sm90.cuh`).

### Plow use of wgmma (cite file:symbol)

| shape | where |
|---|---|
| m64n128k16 bf16 SS | `sm90_wgmma.cuh:wgmma_m64n128k16`, `op_gemm_sm90.cuh:d_gemm_sm90`/`_tma` (128x128 tile, 2 consumer WGs) |
| m64n256k16 bf16 | `wgmma_m64n256k16` (T20b uni256), `gen_attn_pf_hd256_wgmma.inc:hd256_pv_bn32/bn64` (RS, trans-b 1) |
| m64n32/n64k16 bf16 | `gen_attn_pf_hd256_wgmma.inc:hd256_qk_n32/n64`, `op_attention_sm90.cuh`, `op_attention_sm90_v3.cuh`, `op_gemv_wide_sm90.cuh` (swap-AB m64n64, wide-batch decode) |
| m64n128k32 / m64n256k32 e4m3 | `wgmma_m64n128k32`, `wgmma_m64n256k32`; `d_gemm_w8a8_sm90*`, ws384 role (`interp_sm90a_pfgemm_glu_w8a8_gemma4.cu`) |
| m64n8/16/32k32 e4m3 (swap-AB decode) | `op_gemm_fp8_block_sm90.cuh:fp8b_mma` (`gemm_fp8_block_sm90.cu`), `op_gemv_fp8_wgmma.cuh:gemv_fp8_wgmma_mma`, `op_gemm.cuh` |
| m64n64/128k8 tf32, m64n64/128k16 f16 | `snac/snac.cu`, `s3gen/s3gen.cu` (speech) |

### mma.sync fallback

* `mma.sync.m16n8k16` bf16/f16 → `HMMA.16816` on sm_90a, below the wgmma rate (no vendor figure;
  measure before sizing a compute-bound kernel on it). Used by decode GEMV walks
  (`op_gemv_mma.cuh`, `op_gemv_fp8_tc.cuh`, `op_gemv_k8_sm90.cuh`) and the shared `op_gemm.cuh`
  bodies.
* **There is no native fp8 mma.sync on Hopper.** `mma.sync.m16n8k32.e4m3` lowers to
  12x `F2FP.F16.E4M3.UNPACK_B` + 2x `HMMA.16816`. Use wgmma (QGMMA) for fp8, or convert
  e4m3 → bf16 exactly in registers and use bf16 mma (`op_gemv_fp8_tc.cuh`)
  [measured in plow: experiments/README.md "Hopper has no native fp8 mma.sync"].

## 3. Memory movement

| mechanism | form plow uses | rules |
|---|---|---|
| vector LD/ST | 16 B (`uint4`/`float4`, LDG.128 / LDS.128) | natural alignment |
| `cp.async` | `cp.async.cg.shared.global [s],[g],16,src_size` (`sm90_cp16`), `commit_group` / `wait_group N` | sizes 4/8/16; `.cg` = 16 only; `src_size < 16` zero-fills (plow's M/K tail) [vendor documented: PTX cp.async]. cp.async is a weak operation **in the generic proxy** [vendor documented: PTX cp.async, Memory Consistency Model], so a wgmma read of a cp.async stage needs `fence.proxy.async.shared::cta` after the wait. Several plow bodies omit it (see pitfalls) |
| TMA tile | `cp.async.bulk.tensor.{2d,3d}.shared::cluster.global.mbarrier::complete_tx::bytes [dst],[map,{c0,c1(,c2)}],[mbar]` (`sm90_tma2d/3d`) | one elected thread issues; c0 = innermost coordinate; OOB elements zero-filled (TMA zero-fills ragged M and the K tail, relL2 unchanged [measured in plow: op_gemm_sm90.cuh TMA note]) |
| TMA multicast | `.multicast::cluster` + 16-bit ctaMask | works on H100 (SASS `UTMALDG.2D.MULTICAST`); no production win (see pitfalls) |
| bulk store | `cp.async.bulk` smem → global (speech wgmma epilogue, `op_speech_f32.cuh`) | 16 B multiple |
| `ldmatrix` | `.m8n8.x1/.x2/.x4{.trans}.b16` in mma.sync bodies (`op_gemm.cuh`, `op_attention.cuh`, `op_gemv_*`) | B operand can be read from a natural `[n][k]` tile with non-trans `.x2`, bit-identical on sm_120a [measured in plow: experiments `t3_pipe_probe.cu`] |
| `stmatrix` | sm_90+: `.m8n8.x1/x2/x4{.trans}.b16` [vendor documented: PTX stmatrix] | **unused in plow** (epilogues store from fragments or stage through smem) |

Tensor maps (`cuTensorMapEncodeTiled`, host side `crates/plowrt/src/device/cuda.rs`,
`exec/gpu/fp8_block.rs`, `exec/gpu/fa4_sm90.rs`) [vendor documented: CUDA Driver API 13.4]:
* map = 128 B opaque blob, 64 B aligned (plow allocates 128-aligned); rank 1..5; globalAddress 16 B
  aligned; globalStrides multiples of 16 B and < 2^40; boxDim[i] <= 256; `boxDim[0]*elem` a multiple
  of 16; with swizzle, the inner box bytes must be <= the swizzle span (128 B → 64 bf16 / 128 e4m3);
  elementStrides <= 8.
* oobFill `NONE` = zeros; `NAN_REQUEST_ZERO_FMA` is floats only. l2Promotion NONE/64B/128B/256B.
* Plow interpreter maps: bf16 (9) or uint8 (0) for e4m3, rank 2, `SWIZZLE_128B` (3),
  `L2_PROMOTION_128B`, `OOB_FILL_NONE`, box `{64 | 128, rows}` (`cuda.rs` GEN_TMAP). The fp8 block
  decode object uses `L2_PROMOTION_256B`.
* Two ways to deliver a map: (a) a pointer to a map in global memory, acquired per CTA with
  `fence.proxy.tensormap::generic.acquire` + `prefetch.tensormap` (`sm90_tmap_acquire`, interpreter
  path). The CUDA PG uses `.sys` scope for a map written by `cudaMemcpy`; plow uploads the map
  (`exec/gpu.rs` GEN_TMAP) and uses `.gpu`. New code: `.sys`. (b) a `const __grid_constant__`
  kernel parameter (`gemm_fp8_block_sm90.cu`).
* `CU_TENSOR_MAP_SWIZZLE_128B` with a 128 B inner box writes exactly the layout the plow descriptor
  describes, so moving staging from cp.async to TMA leaves the wgmma side unchanged
  (`sm90_wgmma.cuh` TMA note).

Bank conflicts: with no swizzle, a K-major core-matrix layout has a 1024 B row stride. That puts every
row on one bank: 8-way conflicts on both the cp.async store and the wgmma read, and **2.2x slower**
(84 vs 177 TF/s) at an identical relL2 [measured in plow: sm90_wgmma.cuh].

## 4. Synchronization and ordering

* **wgmma.fence.sync.aligned** must come before the first wgmma of a warpgroup. It also must come
  between any register access and a wgmma that reads or writes the same accumulator or A-fragment
  registers. Exception: back-to-back wgmmas of the same shape on the same accumulator are ordered
  implicitly [vendor documented: PTX §9.7.17.7.1].
* **commit_group / wait_group N** are warpgroup-scoped. Reading D or A-fragment registers before a
  `wait_group` that covers their wgmma is UB [vendor documented: PTX §9.7.17.7.3]. With 2 consumer
  warpgroups sharing a smem ring, one warpgroup's `wait_group` does not cover the other, so plow puts
  a `__syncthreads()` between the wait and the refill [measured in plow: op_gemm_sm90.cuh
  "cross-warpgroup smem hazard"].
* `.sync.aligned`: every warp of the warpgroup must execute the same wgmma/fence/commit/wait, so no
  divergent guards. `sm90_bad_k` returns true block-uniformly (thread 0 traps), so callers bail.
* **Async proxy:** wgmma operands and TMA run in the async proxy. Generic `st.shared` (register →
  smem: Ps tiles, dequantized fp8 → bf16 stages, `op_gemv_fp8_wgmma.cuh` staging) needs
  `fence.proxy.async.shared::cta` and then a barrier before wgmma reads it
  [vendor documented: PTX §9.7.17.4]. The same holds for stages filled by plain `cp.async`, which is
  generic proxy; only TMA (`cp.async.bulk*`) writes land in the async proxy. wgmma completion
  carries an implicit async → generic fence.
* **mbarrier** (`sm90_mbar_*`): `mbarrier.init.shared::cta.b64 [b], count`.
  * Producer: `mbarrier.arrive.expect_tx.shared::cta.b64 _, [b], bytes`, then the TMA completes
    `bytes`.
  * Consumer: spin on `mbarrier.try_wait.parity.shared::cta.b64 p, [b], phase` and flip the parity
    each ring wrap.
  * tx-count range is ±(2^20−1); arrival count is 1..2^20−1 [vendor documented: PTX §9.7.15.16].
  * `mbarrier.inval` before reusing the arena as data. PTX requires it; plow does it per op.
  * Ring pattern: full + empty barrier per stage. The last warpgroup to release a stage refills it
    (`op_attention_sm90_v3.cuh`).
* **setmaxnreg.{inc,dec}.sync.aligned.u32 N**: N must be an immediate in 24..256, a multiple of 8,
  and the same for all 4 warps of the warpgroup. `.inc` blocks until the CTA-local pool can supply
  the registers; new registers are undefined. It needs the kernel compiled with an explicit max-reg
  bound, otherwise it may do nothing [vendor documented: PTX setmaxnreg].
  * Plow uses `__maxnreg__(160)` on ws384, with producer dec → 32 and consumers inc → 224:
    128x32 + 256x224 = 61440 <= 64 K (`op_gemm_sm90.cuh` T31).
  * Use immediates only (`sm90_reg_dec/inc` switch).
  * The pool is per-CTA: `dec` does **not** raise blocks/SM.
* **Clusters / DSMEM:** `barrier.cluster.arrive/wait` (cluster.sync → `UCGABAR_*` + `MEMBAR.ALL.GPU`),
  `mapa` + `ld.shared::cluster` [measured in plow: experiments `cluster_gq_probe.cu`]. Not used in
  production. A cluster barrier across divergent persistent streams deadlocks (pitfalls).
* **Named barriers:** `bar.sync id, nthreads` with wg-scoped ids 1/2 in the ws384 smem epilogue
  (`op_gemm_sm90.cuh` T38). id 0 = `__syncthreads`.
* **PDL:** `griddepcontrol.wait` / `launch_dependents` (sm_90+). The device side is
  `plow_pdl_wait()` (`sm120_common.cuh`) and the host side `CudaBackend::launch_cooperative_pdl`. It
  is not wired into serving (pitfalls).

## 5. Toolchain as plow uses it

* **Compiler:** nvcc from the flake's `cudaPackages`, exported as `PLOW_NVCC` by the dev shell
  (`flake.nix`) and used as the `PLOW_CUBIN_NVCC` default in `runtime/CMakeLists.txt`; a host
  `/usr/local/cuda/bin/nvcc` is only the fallback (`PLOW_CAMPAIGN_NO_NIX=1` boxes). The cubin must not be newer than the
  driver accepts, or the load fails with `CUDA_ERROR_INVALID_IMAGE` (`scripts/build_sm90a_cubin.sh`).
* **Interpreter objects:** built through `runtime/CMakeLists.txt` (`PLOW_SM90A_CUBIN=ON`, which
  plowc passes via `cubin_arch_option("sm_90a")`, `crates/plowc/src/main.rs`) and
  `runtime/cmake/nvcc_cubin.sh` under a clean `env -i` environment.
  * Base flags: `-arch=sm_90a -O3 -cubin -I runtime/common -I runtime/nvidia
    -DPLOW_NV_EMBED_SMEM=1 [-DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2]`.
  * Per-arch `_nv_var_sm90a`: `-DPLOW_NV_MLA=0 -DPLOW_NV_MAMBA=0 -DPLOW_NV_DSA=0 -DPLOW_NV_GEMV_RB=1
    -DPLOW_NV_RB_GEMV=1 -DPLOW_NV_RB_QKV=1 -DPLOW_NV_RB_LMHEAD=1 -DPLOW_NV_GEMV_XREG=1
    -DPLOW_NV_GEMV_KPANEL=1 -DPLOW_MOE_DOWN_LANESPLIT=1 -DPLOW_NV_FA_WPR=1 -DPLOW_NV_FP8_RB=4`.
  * Axes: decode `-DPLOW_BUCKET_DECODE=1 -DPLOW_NV_FA_GF_FULL=4`; prefill `-DPLOW_BUCKET_DECODE=0
    -DPLOW_NV_PREFILL=1`; segments add `-DPLOW_NV_SEGMENTS=1 [-DPLOW_NV_SEG_GEMM=1] [-DPLOW_NV_PACKED_REQUEST=1]`.
  * Symbols: `_Z12interp_sm90a11PlowProgram` / `_Z15interp_sm90a_pf11PlowProgram`.
  * Standalone path: `scripts/build_sm90a_cubin.sh`. kernelcaps mirror:
    `crates/kernelcaps/src/targets.rs` (`IsaLevel::Sm90a`).
* **Role objects:** `scripts/build_sm90a_fp8_block.py` (`gemm_fp8_block_sm90.cu`),
  `scripts/build_sm90a_gemma4_segments.sh`, `scripts/build_sm90a_qwen_gdn.py` +
  `scripts/export_gdn_prefill_sm90.py`, and `scripts/export_fa4_hd{256,512}_sm90.py`
  (CuTe-DSL AOT → packet cubin + host stub, never a `.so`).
* **Executables (probes):** use `-gencode arch=compute_90a,code=sm_90a`. `-arch=sm_90a -o exe` and
  `-arch=native` reject wgmma ("not supported on .target sm_90") [measured in plow:
  experiments/README.md "nvcc: how to actually get wgmma"]. A `.cubin` with `-arch=sm_90a` is fine.
* Inline-asm style (matches `sm90_wgmma.cuh` / `op_gemm_fp8_block_sm90.cuh`; predicate `scale-d`,
  `"memory"` clobber on fence/commit/wait):

```cuda
sm90_wg_fence();                                   // wgmma.fence.sync.aligned
asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, p, 1, 1; }"
    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]) : "l"(desc_a), "l"(desc_b), "r"(accumulate));
sm90_wg_commit(); sm90_wg_wait<0>();
```

* **Inspect:**
  * `cuobjdump --dump-resource-usage <cubin>`: REG/STACK/SHARED/LOCAL per object.
  * `cuobjdump -sass` / `nvdisasm`: look for `HGMMA`/`QGMMA`, `UTMALDG`, `HMMA.16816`, `F2FP`,
    `STL/LDL` (spills).
  * `-Xptxas -v` output is parsed and gated by `crates/kernelcaps/src/resource.rs`
    (`parse_ptxas`).
  * `cuobjdump -symbols` checks the entry symbol.
  * `scripts/plow_isa.py` / `scripts/asm_audit.py` are AMD-only.
* Every `#if PLOW_*` token in `runtime/` must be registered in `crates/devgen/src/knob_spec.rs`.
  Derived internal constants use a non-`PLOW_` prefix (`PGM90_*`, `FA3_*`).

## 6. Pitfalls

**wgmma / smem layout**
* **cp.async → wgmma needs `fence.proxy.async.shared::cta`; without it the read is undefined.**
  cp.async writes are generic-proxy, wgmma reads async-proxy [vendor documented: PTX §9.7.10.28.3,
  §9.7.17.4]. `d_gemm_sm90` (cp.async path), the `op_moe_sm90.cuh` rings and `op_gemv_wide_sm90.cuh`
  omit the fence and pass oracles today, which proves nothing; the `op_attention_sm90.cuh:fa90_async_proxy_fence`
  comment claiming otherwise is wrong. New code: wait → fence → barrier → wgmma, or stage with TMA
  [measured in plow: code read; not a measured failure].
* LBO/SBO are already encoded `>>4` inside `sm90_make_desc`. Shifting them again gives silent
  garbage [measured in plow: runtime/nvidia/sm90_wgmma.cuh].
* A 128 B-swizzle tile base that is not 1024 B aligned gives silently **wrong** results, not a fault
  [measured in plow: sm90_wgmma.cuh]. Vendor rule: base offset 0 only on the pattern boundary
  [vendor documented: PTX §9.7.17.5.1.2.2].
* Mixing the no-swizzle recipe (+256 B K-core stride per k16) with the 128 B recipe (+32 B start
  advance) is wrong [measured in plow: sm90_wgmma.cuh].
* Swizzle-free layouts pass the oracle but run 2.2x slower (8-way bank conflicts)
  [measured in plow: sm90_wgmma.cuh, experiments `wgmma_bf16_probe.cu`].
* The fp8 wgmma has no transpose: operands must be K-major (plow's TN contract)
  [vendor documented: PTX wgmma.mma_async "transpose only for .f16/.bf16"].
* int8 wgmma N above 32 must be a multiple of 16 (no 40, 56, ..., 248) [ptxas 12.9 probe].
* The 128 B-swizzle stagers need K to be a multiple of 8 (bf16) / 16 (e4m3). Otherwise the cp.async
  source misaligns (UB). `sm90_bad_k` traps [measured in plow: sm90_wgmma.cuh]. TMA maps also
  need `K*elem % 16 == 0` (`cuda.rs` GEN_TMAP assert) [vendor documented: Driver API globalStrides].
* Flash attention:
  * S through smem ran at 0.785x the mma.sync baseline; keep softmax in-fragment.
  * Transposing V in smem took 46% of the probe's runtime; consume V MN-major with trans-b = 1.
  * [measured in plow: op_attention_sm90.cuh header]
* A V tail tile past `hi` must be zeroed in smem: P = 0 there, but 0 * NaN = NaN
  [measured in plow: op_attention_sm90_v3.cuh].
* fp8 wgmma without promotion: relL2 1.14e-3 at K=3840. It passes the 1.6e-3 oracle bar, but the
  margin is thin; promote every 128 K when register budget allows
  [measured in plow: op_gemm_sm90.cuh `PGM90_FP8_PROMOTE`].

**ptxas, inlining, registers, stack**
* wgmma in a called (`__noinline__`/ABI) function is serialized (ptxas **C7510**). Inside the
  megakernel that cost a uniform ~1.8x on every prefill packet. Keep wgmma bodies inlined, or put
  them in a separate role object [measured in plow: op_gemm_sm90.cuh T-ws note;
  docs/runtime/kernel-generators.md].
* ptxas time grows superlinearly with the arms inlined into the interpreter. The speech cubin with
  every arm took 4-7 h; one packet's op set takes 1.5-8 min. Compile only the ops the packet emits
  (`PLOW_SPEECH_OPS`, dead-arm gating) [measured in plow: docs/runtime/tts.md].
* One arm sets registers, stack and occupancy for the whole object.
  * #115 inlined the Maya2 GEMV flag modes into Gemma/Qwen entries: stack 400 → 592 B
    (26B decode), 352 → 736 B (Qwen) and 12B FP8 decode +4-5%, with token digests unchanged.
  * Fixed by 10003a277 (speech-only compile) [measured in plow: 10003a277, b8d5a0f13].
  * Check `cuobjdump --dump-resource-usage` and step_bench time on every RC, not just digests.
* Dead-arm gating (`PLOW_NV_MLA/MAMBA/DSA=0`) shrank the decode cubin 43% (REG 208 → 177,
  stack 1024 → 0) [measured in plow: scripts/build_sm90a_cubin.sh comment].
* An m64n128 accumulator inlined into the decode entry cost other arms more (flash-decode +40% at
  B = 64) than it saved; the wide GEMV holds 32 accumulators [measured in plow:
  op_gemv_wide_sm90.cuh].
* `setmaxnreg`:
  * A split that reconverges into a shared loop is **dropped** by ptxas (C7507). So is a split
    where any reachable arm lifts the producer floor past 32.
  * Cycling dec/inc per op deadlocked in-model.
  * Split once per launch and run per-role loops with aligned `__syncthreads` counts.
  * [measured in plow: interp_sm120.cu `plow_ws_role_loop`, op_gemm_sm90.cuh T11]
* `setmaxnreg` needs an entry register bound: `__maxnreg__(N)` (plow) or `__launch_bounds__(T, minBlocks)`.
  `__launch_bounds__(T)` alone or no bound drops it with ptxas C7508 [nvcc 12.9 sm_90a probe;
  vendor documented: PTX §9.7.21.5].
* A cp.async producer warpgroup has little to donate: donation works, but below ~88 producer regs
  throughput drops 20-45% with no spills. Deep donation (dec to 24-40) needs a TMA producer
  [measured in plow: experiments `hopper_warpspec_prefill.cu`].
* A 256-thread megakernel with a dedicated producer warpgroup halves the math warpgroups: 1368 vs
  750 ms at T = 4096. In-megakernel TMA is therefore "uniform" (an elected thread issues while both
  warpgroups compute). Producer/consumer (ws384, `__maxnreg__(160)`) lives only in lean role
  objects [measured in plow: op_gemm_sm90.cuh].
* An oversized smem arena claim is not a compile error: plowrt rejects static + dynamic > opt-in at
  serve time (`device/cuda.rs` `set_max_dynamic_smem`). Keep `static_assert`s on arena sizes.
* An arena claim leaking into other packets halved occupancy: 2 x 164864 B > 232448 B silently
  dropped the 12B to 1 block/SM: 2 x (164864 + 1024 reserved) B > 233472 B (228 KiB/SM)
  [measured in plow: 6284c800e].

**Builds and reproducibility**
* `interp_sm90a.cubin` is not byte/SASS-reproducible for some configs. Prove "no-op for other
  packets" with resource usage, a re-gate and step_bench digests [measured in plow: docs/bringup/agent-tools.md "Standing rules"].
* `PLOW_EXTRA_DEFINES` in the env is ignored by emits; variants must be manifest tuning keys.
  `campaign.py build` never rebuilds plowc. A killed build leaves `make`/`ptxas` writing
  stale-hash cubins, so use a fresh out dir [measured in plow: docs/bringup/agent-tools.md "Standing rules"].
* One `interp_sm90a` nvcc takes 22-38 GB RAM, so never compile beside a serving grid
  [measured in plow: docs/bringup/agent-tools.md "Standing rules"].
* Build, emit and test inside `nix develop`; `nvcc_cubin.sh` runs nvcc under `env -i` itself. Only
  standalone CUDA-runtime probes are compiled in the plain shell: built in nix they abort at runtime
  init ("stack smashing detected") [measured in plow: docs/bringup/agent-tools.md "Standing rules"].
* Role objects are hash-pinned to the packet, and the live copy is in `assets/`. A cubin swap in an
  objects dir is silent and inert. A/B a role by re-emitting, with a positive control
  [measured in plow: 9d89adfad].

**Memory / KV / TMA**
* Under live KV only rows `[0, kv_len)` are mapped (4096-row commits). Never load past `kv_len`,
  masked or not. Run `scripts/bench/step_boundary.sh` (ctx 4094/4095/4096/8191/8192) on any new
  KV-reading arm [measured in plow: docs/bringup/agent-tools.md "Standing rules"].
* The FA4 SM90 producer reads up to `n + splits - 2` page-table entries (and entry -1). Pad the
  table and verify with a poison table [measured in plow: docs/bringup/agent-tools.md "Standing rules"].
* TMA for 1-D weight streaming was the slowest of 37 configs; `cp.async.cg` won (measured on
  sm_120a) [measured in plow: experiments/README.md `gemv_transport.cu`]. This covers plain GEMV
  streams only; the swap-AB fp8 block decode object uses TMA + a producer warp.
* Clusters / DSMEM / multicast work on H100; no route qualified.
  * Multicast 0.78-0.98x off-L2 (duplicate reads are L2 hits); compute-bound BM64 +1-4%, BM128 0.97-0.99x.
  * Cluster-cooperative claiming costs 4.6x and faults under sustained load.
  * DSM split-K/flash merges are 1.15-3.35x slower.
  * `cuLaunchCooperativeKernel` cannot carry a cluster dim.
  * [measured in plow: experiments/README.md `hopper_cluster_multicast.cu`, `cluster_gq_probe.cu`]
* Flash-decode split count is a grid-alignment cliff: nsplit = 132/gcd(n_grp,132) = 33.
  34 costs +67% at 128k [measured in plow: experiments `fa_gf_full_h100_ab.cu`,
  build_sm90a_cubin.sh].
* SplitZip (exponent compression) loses 0.72-0.85x on H100: at HBM rate the H100 NVL has ~0.53
  warp-instruction issue slots per bf16 weight element (RTX PRO 6000 ~2.4), so CUDA-core decode
  cannot hide behind the load [measured in plow: experiments `splitzip_h100_ab.cu`].
* GEMM epilogues: ncu estimates ~27% loss from uncoalesced C stores; stage through a dead ring slot
  [measured in plow: op_gemm_sm90.cuh T38]. L2 tile rasterization (bands) lifts a 450 TF/s
  B-reload ceiling [measured in plow: op_gemm_sm90.cuh T13].

**Vendor library routes (cuBLASLt, FA4, DeepGEMM)**
* cuBLASLt algorithm choice is timing-based and re-rolls per server start, so c1 numerics change
  between starts. Pin with `<assets>/cublaslt_algos.jsonl` (`campaign.py probe`) and diff the
  `algorithm selected` lines before blaming a build [measured in plow: docs/bringup/agent-tools.md "Standing rules"].
* G26 / grouped-Lt packets require cuBLASLt >= 13.4. The lab loader's `.so.13` is 13.1, so ship
  `lib/libcublasLt.so.13` [measured in plow: docs/bringup/agent-tools.md "Standing rules"].
* FP8 Lt fast accumulation (`PLOW_LT_FP8_FAST_ACCUM_MAX_K`): 12B pf4k 118.5 → 109.4 ms
  [measured in plow: 5b8f11da8].
* The Qwen3.8 FP8 canonical recipe (status "emits", **not production**: it loses the prod mix to
  vLLM) runs its projections as block-scaled cuBLASLt (BLK128x128 weight scales x VEC128 activation
  scales). It also enables the native block-scaled FP8 decode GEMM object for planned decode
  segments (`PLOW_EMIT_DECODE_FP8_BLOCK`, rows <= 32)
  [measured in plow: recipes/infervisor/qwen3.8-27b/sm90a-h100-tp1-fp8.toml].
* DeepGEMM is 2-21% slower than cuBLASLt for the Gemma MoE rungs (no bf16 swap-AB on SM90); no
  route built [measured in plow: docs/arch/11-tuning-coverage.md].
* cuBLASLt fp8 runs 1324-1468 TF/s at 12B shapes, vs 950-1170 for plow's 256-thread uniform body.
  The gap is the dedicated producer warpgroup [measured in plow: op_gemm_sm90.cuh T31].
* PDL:
  * Routed rungs were unchanged.
  * In prefill it saved ~0.04 ms but shifted E4B parity 0.985 → 0.974 deterministically.
  * nvjet does not trigger dependents early.
  * MoE kernels with 1 CTA/SM at full smem cannot co-reside.
  * It stays parked on branch `sm90-pdl`.
  * [measured in plow: docs/runtime/gemma4-e4b-h100.md; docs/arch/11-tuning-coverage.md]

**Measurement**
* The H100 NVL probe card (310 W cap) dropped from 1785 to ~700 MHz under sustained wgmma.
  Memory-bound GEMV sat at 600-615 MHz. Use short bursts, 25 ms gaps, rotated arms, min-of-N >= 12,
  NVML clock sampling and L2-defeating buffers (`tma_ws_moe_group.cu` harness)
  [measured in plow: experiments/README.md "MANDATORY benchmarking harness"].
* A wgmma opcode swap alone is not enough: cp.async-staged tiles stay staging-bound (~1536
  cp.async vs 4 HGMMA per BK = 64 stage). The win needs TMA plus bigger tiles
  [measured in plow: experiments/README.md].

## 7. Roofline numbers

| quantity | datasheet (H100 SXM) | plow-calibrated | where it lives |
|---|---|---|---|
| HBM read | 3352 GB/s (3.35 TB/s) | **3094 GB/s** (92%); copy 3043 GB/s | datasheet: `h100.rs` `bandwidth: GBps(3352.0)` (`bandwidth_measured: None`), `roofline.py`; calibrated: `ceilings.json` 2026-10-29 (scratch, `block-harness/validate/`) cited in `recipes/infervisor/qwen3.8-27b/sm90a-h100-tp1-fp8.toml` header |
| BF16 dense | 989 TF (1979 sparse / 2) | **771.3 TF** (78%; 8192³ torch.matmul) | `crates/costmodel/src/roofline.rs:peak_compute_tflops`, `roofline.py`; calibrated: `ceilings.json` |
| FP8 e4m3 dense | 1979 TF (3958 sparse / 2) | **1407 TF** (71%; `_scaled_mm`) | same; recipe comment "ceilings 3094 GB/s / 1407 FP8 TFLOP/s" |
| INT8 dense | 1979 TOPS | not calibrated | vendor page |
| practical stream (doc folklore) | — | "~2.9 TB/s practical" | `docs/runtime/attention-roofline.md` |
| H200 | 4800 GB/s; same TF | not measured | `h100.rs` `H200_SXM` |

Calibration conditions: `NVIDIA H100 80GB HBM3`, torch 2.13+cu130, driver 595.91.07, 700 W,
SM 1830 (read before the runs) / max 1980 MHz, medians of 30; bf16 spread 7.2%.

Re-measure with `scripts/bench/calibrate_ceilings.py --out ceilings.json` (one short lease via
`gpuq.py submit`). Feed it to `scripts/campaign/block_roofline.py --ceilings` or `op_roof.py`.
Without `--ceilings` the output says "NOT calibrated" and uses registry peaks. No sm90a recipe
carries a `[roofline]` table today (AMD recipes do).

Repo-vs-vendor notes, flagged and not edited:
* `h100.rs` `mma.fp16 = 256` is a relative anchor. The costmodel's analytical fallback
  (`tensor_cores * macs * 2 * clock`) would give ~535 TF for H100, half the datasheet. It is bypassed
  only because `peak_compute_tflops` hardcodes names containing "H100"/"H200". That also gives
  **H100 PCIe** 989 TF, although PCIe is a lower-clocked 114-SM part (its datasheet dense bf16 is
  lower; exact value UNVERIFIED here).
* H200 boost clock: `h100.rs` 1830 MHz vs `roofline.py` 1980 MHz (UNVERIFIED: settle with
  `nvidia-smi --query-gpu=clocks.max.sm` on an H200).
* L2: vendor 50 MB; experiments README "L2 measures 60 MB" on the H100 NVL (UNVERIFIED: settle with
  `cudaDevAttrL2CacheSize` on that card). hwspec uses 50 MiB for all SKUs.
* `H100_NVL` is `MemKind::Hbm3e` in hwspec, but NVIDIA's H100 NVL product brief (PB-11773-001
  v01) says **HBM3**, 94 GB, 3938 GB/s. The bandwidth matches hwspec; the memory kind does not.

## 8. Decision checklist

* **About to write a GEMM/attention MMA for M >= 64:** use `wgmma` through `sm90_wgmma.cuh`
  descriptors (128 B swizzle, 1024 B aligned, +32 B k-substeps). Never fp8 `mma.sync`, which is
  emulated.
* **Decode GEMV (M = 1..8):** stay bandwidth-first: FFMA/`cp.async.cg` or the mma.sync / swap-AB
  wgmma walks (`op_gemv_*`). TMA and warp specialization lost on plain 1-D streams (sm_120a).
* **Narrow M with block-scaled fp8:** swap-AB (weights = A m64, tokens = n8/16/32), as in
  `op_gemm_fp8_block_sm90.cuh`. cuBLASLt block-scaled is the baseline route. The native object is
  declared per planned decode segment (<= 32 rows) and must beat Lt per rung.
* **Choosing staging:** inside the 256-thread megakernel, use uniform TMA (one elected issuer,
  mbarrier ring NS = 3-4). Producer/consumer + `setmaxnreg` only in lean objects (256-thread entry
  split at occ 2, or WS384), split once per launch.
* **Tile and smem:** budget the staged TMA boxes, not BM x BK: stage = A box + B box (+ padding,
  barriers, alignment). The fp8 BM64/BN256 path stages a full 128-row A box: 16 + 32 = 48 KiB
  (`op_gemm_sm90.cuh:2145`). Keep the arena <= ~113 KiB if you
  need 2 blocks/SM. Keep the decode object's dynamic smem low enough that L1 stays above 28 KiB.
  `static_assert` the claim.
* **Accumulators:** m64n256 = 128 f32/thread; plow's WS384 consumer needs >= 154 regs. A uniform
  128-reg cap cannot hold it; a 256-thread occ-2 object can, by `setmaxnreg` donation (producer
  dec 32, consumer inc 224; `op_gemm_sm90.cuh:1984-2020`). Decide occupancy first, then N.
* **fp8 accuracy:** promote into an f32 shadow every 128 K if regs allow. Otherwise check the
  op-test relL2 at the largest K.
* **Writing smem from registers or with plain cp.async, then wgmma:**
  `fence.proxy.async.shared::cta` + barrier.
  `wgmma.fence` before the first MMA and after touching accumulators or A fragments.
* **Two warpgroups sharing a ring:** `__syncthreads` (or mbarrier release) between `wait_group`
  and the refill.
* **New inlined arm in `interp_sm90a`:** record REG/STACK/spills before and after
  (`cuobjdump --dump-resource-usage`), time step_bench on all packets, and gate it out of objects
  that never emit it.
* **New KV-reading arm:** clamp every row to `< kv_len`, then run `step_boundary.sh`.
* **Vendor routes:** pin cuBLASLt algorithms (`cublaslt_algos.jsonl`), ship `libcublasLt.so` >= 13.4
  beside plowrt; ship FA4 as packet cubin + host stub.
* **Flash-decode split count:** align work items to 132 SMs (nsplit 33 at 4 groups); scale by
  1/n_batch.
* **Probe executables:** `-gencode arch=compute_90a,code=sm_90a`, compiled in the plain shell
  (CUDA-runtime probes only), run via `gpuq.py`, with the DVFS-safe harness.
* **Before claiming a perf number:** compare against calibrated ceilings (3094 GB/s, 771 / 1407 TF),
  not datasheet peaks.

## 9. Sources

| topic | document | section | version/date |
|---|---|---|---|
| wgmma shapes, descriptor, swizzle, fragments, fence/commit/wait, async proxy | PTX ISA — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html | §9.7.17 (Warpgroup MMA), §9.7.17.4, §9.7.17.5.1.2.2, §9.7.17.5.2, §9.7.17.7 | 9.4, updated 2026-10-12 |
| cp.async, bulk, TMA, ldmatrix/stmatrix, mbarrier, griddepcontrol, fences, setmaxnreg | same | cp.async §9.7.10.28.3.1, bulk §9.7.10.28.4.1, tensor §9.7.10.28.5.3; ldmatrix/stmatrix §9.7.16.5.15-16; mbarrier §9.7.15.16; griddepcontrol §9.7.15.14; fence §9.7.15.4; setmaxnreg §9.7.21.5 | 9.4 |
| SM limits, smem carve-outs, L1, L2, clusters, TMA overview | Hopper Tuning Guide — https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html | §1.4.1.1-1.4.1.3, §1.4.2.2, §1.4.2.4 | v13.4 |
| Tensor maps | CUDA Driver API — https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html | `cuTensorMapEncodeTiled` | 13.4 |
| smem banking, cluster launch, `__grid_constant__` | CUDA Programming Guide — https://docs.nvidia.com/cuda/cuda-programming-guide/ | shared memory, thread block clusters | v13.4.2 |
| H100 SXM/NVL peaks, memory, TDP | https://www.nvidia.com/en-us/data-center/h100/ | Specifications (tensor figures are "with sparsity") | fetched 2026-11-03 |
| H200 peaks, memory | https://www.nvidia.com/en-us/data-center/h200/ | Specifications | fetched 2026-11-03 |
| Hopper architecture (SM/GPC layout, clocks) | NVIDIA H100 Tensor Core GPU Architecture whitepaper (GTC22) | SM and GPC sections | V1.0x; not re-fetched (clock-derived numbers UNVERIFIED) |

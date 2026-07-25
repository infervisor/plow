// tma_ws_gemm_fp8.cu -- Hopper (sm_90a) FP8 e4m3 w8a8 PREFILL GEMM: full pipeline.
//
//   TMA producer (cuTensorMapEncodeTiled, HW 128B swizzle, mbarrier expect_tx, one elected
//   thread) + producer/consumer WARP SPECIALIZATION (setmaxnreg dec/inc) + multi-stage smem
//   ring (NS = 3..6) + LARGE tiles (BM x BN in {128x128, 128x256, 64x256}) + fp8 WGMMA
//   (wgmma.mma_async.m64nNk32.f32.e4m3.e4m3) + DeepGEMM-style two-level accumulation.
//
// CONTRACT (plow w8a8 prefill linear, TN):
//   C[m,n] = a_scale[m] * w_scale[n] * sum_k A[m,k]*B[n,k]
//   A e4m3 [M,K] k-contiguous, B e4m3 [N,K] k-contiguous, C f32 [M,N].
//   Per-ROW a_scale and per-COL w_scale are applied in the epilogue from the (row,col) map
//   at ~zero cost (BN/8 * 2 scalar loads per tile, amortised over K).
//
// WHAT THIS FILE BUILDS ON (validated in-tree, not re-derived here):
//   * runtime/nvidia/experiments/wgmma_fp8_probe.cu   -- QGMMA.64x128x32.F32.E4M3.E4M3 works;
//     accumulator map row = 16*warp + lane/4 + 8*hi, col = 8*g + 2*(lane%4) + lo, reg = 4g+2hi+lo,
//     N/2 f32 regs/thread. fp8 wgmma is K-MAJOR ONLY (no trans immediates) == plow's TN layout.
//   * runtime/nvidia/experiments/wgmma_moe_group_probe.cu -- 128B swizzle is worth 2.2-2.35x on
//     the fp8 wgmma path. A BK=128 e4m3 row is exactly 128 B = one swizzle atom row, so the
//     descriptor is LBO=16 B, SBO=1024 B, swizzle-mode=1, and a k32 substep advances only the
//     START ADDRESS by +32 B. 1024 B tile alignment is MANDATORY.
//   * runtime/nvidia/experiments/hopper_warpspec_prefill.cu -- warp spec +11..23%; setmaxnreg is
//     real but a cp.async producer cannot be squeezed below ~88 regs. TMA is the enabler.
//   * Hopper has NO native fp8 mma.sync (m16n8k32.e4m3 lowers to F2FP + HMMA.16816 on sm_90a),
//     so wgmma is the ONLY route to the fp8 tensor core on H100.
//   * Native fp8 WGMMA does NOT accumulate in true f32 (error grows with K), hence the
//     PROMOTE_K two-level accumulation option below.
//
// KEY DIFFERENCE vs the cp.async probes: with CU_TENSOR_MAP_SWIZZLE_128B declared in the tensor
// map, the HARDWARE performs the swizzle. Do NOT also XOR store-side. The cp.async variant in
// this file (PRODM=0) keeps the store-side XOR and is the A/B control.
//
// ===========================================================================================
// RESULTS (H100 NVL, 132 SMs, CUDA 13.0, driver 570; GPU SHARED with another tenant and power
// capped at 310 W -- see the MEASUREMENT CAVEAT next to bench())
// ===========================================================================================
// SASS PROOF (cuobjdump -sass; census over all 28 instantiated kernels)
//   every PRODM=1 kernel:  14 x UTMALDG.2D              0 x LDGSTS
//   every PRODM=0 kernel:   0 x UTMALDG.2D             15 x LDGSTS.E.BYPASS.128.ZFILL
//   mainloop math:  QGMMA.64x128x32.F32.E4M3.E4M3 / QGMMA.64x256x32.F32.E4M3.E4M3
//   setmaxnreg lowers to USETMAXREG.DEALLOC.CTAPOOL (dec) + USETMAXREG.TRY_ALLOC.CTAPOOL (inc),
//   present in exactly the MODE=2 kernels, absent in free/clamp. mbarriers are
//   SYNCS.ARRIVE.TRANS64 / SYNCS.PHASECHK.TRANS64.TRYWAIT; wgmma groups are WARPGROUP.ARRIVE /
//   WARPGROUP.DEPBAR.LE. So the pipeline really is TMA + mbarrier + QGMMA, not a lowering.
//
// CORRECTNESS (f32 CPU oracle over the same e4m3 bytes, per-row x per-col scales, gate 6e-3):
//   shape (M,N,K)      no-promote   promote 64   promote 128  promote 256
//   (64,128,32)        3.990e-05    3.990e-05    3.990e-05    3.990e-05     (K < 128, one stage)
//   (128,256,64)       6.108e-05    6.108e-05    6.108e-05    6.108e-05     (K < 128, one stage)
//   (512,4096,3840)    1.1387e-03   6.185e-05    1.0381e-04   1.7433e-04
//   (512,15360,3840)   1.1382e-03   6.209e-05    1.0426e-04   1.7478e-04
//   (200,4096,3840)    1.1414e-03   6.207e-05    1.0427e-04   1.7486e-04
//   -> ALL PASS. Two-level accumulation buys 18.4x (promote 64) / 10.9x (128) / 6.5x (256) at
//   K=3840, reproducing the DeepGEMM result exactly. plow's fp8 oracles land ~1.6e-3, so the
//   un-promoted 1.14e-3 eats ~70% of the budget before any quantisation error; promote-128 at
//   1.04e-4 leaves the budget essentially free. Promotion is a no-op below K=128 by construction.
//
// TENSOR-CORE CEILING ON THIS PART (k_qgmma_rate: back-to-back wgmma out of resident smem, no
// global traffic): 826 / 878 / 892 TF/s over three runs = 42-45% of the 1979 TF/s 700 W-SXM
// datasheet number. The 310 W cap, not the kernel, is what makes the datasheet peak unreachable
// here, so the sweep reports % of this measured ceiling; divide by ~2.2 for % of datasheet.
//
// THROUGHPUT (TF/s; 30 interleaved shuffled passes, per-cell max; representative run)
//                            (512,4096,3840)  (512,15360,3840)  (200,4096,3840)
//   tma 128x128 NS=3..6          452-504            213-260          178-192
//   tma 128x256 NS=3/4           198-215            290-299           77-100
//   tma 64x128  NS=4/6/8         346-377            155-168          213-232
//   tma 64x256  NS=4/5           411-435            173-176          163-166
//   cp.async 128x128 (fair A/B)  185-233            145-157           71-74
//   cp.async 128x128 dec=40      115-119 (SPILLS)   111-117           49-50
//   in-tree wgmma cp.async ref   172.9              169.2             --
//   in-tree mma.sync e4m3 ref    116.5              120.9             --
//
//   BEST per shape: 128x128 NS=5 (498-504 TF/s, 56-57% of the measured ceiling, 25% of the
//   datasheet peak) at (512,4096,3840); 128x256 NS=4 (299) at (512,15360,3840); 64x128 NS=8
//   (232) at (200,4096,3840).
//   vs the in-tree wgmma cp.async probe: 2.9x / 1.8x. vs emulated mma.sync: 4.3x / 2.5x.
//
// FINDINGS
//   1. TMA IS THE WHOLE STORY, and it beats cp.async decisively: 480-504 vs 192-233 TF/s
//      (2.1-2.6x across runs) on an otherwise IDENTICAL kernel -- same tiles, same 128 B swizzle, same warp specialisation,
//      same ring depth, same wgmma, only the transport differs. At BM=BN=128, BK=128 a stage is
//      32 KB; a 128-thread cp.async producer needs 16 LDGSTS per thread per stage plus all the
//      address arithmetic, and it simply cannot keep 2 consumer warpgroups fed. TMA needs ONE
//      instruction pair issued by ONE thread. This is also why the earlier in-tree probes had to
//      stay at small tiles: cp.async caps the usable tile size.
//   2. setmaxnreg DOES NOT PAY once the producer is TMA. At 128x128 NS=5 the whole family --
//      free 469/496, clamp 421/405, dec=24 462/446, dec=40 478/504, dec=64 446/446, dec=88
//      499/462 (two independent runs) -- is one noise band, with no ordering that survives a rerun.
//      The reason is structural: with TMA the producer warpgroup has nothing to spend registers
//      on (127 of its 128 threads retire immediately after the dec), so there is nothing to
//      donate. It is not harmful either -- unlike the cp.async case, dec=24 costs nothing. The
//      register knob mattered in hopper_warpspec_prefill.cu precisely BECAUSE the producer was
//      cp.async; removing that producer removes the knob. KEEP the warp specialisation, DROP
//      setmaxnreg from the tuning space for TMA kernels.
//   3. The reverse still holds for cp.async: dec=40 on the cp.async producer is the ONLY
//      non-promotion config in the sweep that spills (10 STL / 10 LDL, 40 B local) and it costs
//      38% (119 vs 192 TF/s). This independently re-confirms the ~88-register floor.
//   4. RING DEPTH IS NOT THE KNOB. NS=3 (97 KB) through NS=6 (193 KB) are within noise at
//      128x128. With TMA the producer is so cheap that 3 stages already cover the latency; the
//      extra smem buys nothing and costs occupancy headroom. Pick the SHALLOWEST ring that
//      works. (Occupancy is 1 CTA/SM for every variant except the deliberately reg-starved
//      64x128 occ2 point, which was SLOWER -- 257 vs 346 -- so 2 CTAs/SM is not worth chasing.)
//   5. TILE SIZE IS SHAPE-DEPENDENT, and the mechanism is wave quantisation, not arithmetic
//      intensity. At (512,4096,3840): 128x256 -> only 64 tiles for 132 SMs (half the GPU idle)
//      -> 215 TF/s, while 128x128 -> 128 tiles -> 478. At (512,15360,3840) it inverts: 128x256
//      gives 240 tiles = 1.82 waves and wins with 299, 128x128 gives 480 tiles = 3.64 waves and
//      loses with 247. At (200,4096,3840), BM=128 wastes 56 of 256 rows AND yields 64 tiles, so
//      BM=64 (128 tiles) wins. => the AOT tuner MUST select (BM,BN) from (M,N,SM count), and
//      the tile-count/SM-count ratio is the feature that matters, not flops/byte.
//   6. TWO-LEVEL ACCUMULATION IS CHEAP ENOUGH TO ALWAYS ENABLE at BN=128: promote-256 costs 8%
//      (441 vs 478), promote-128 costs 12% (421), promote-64 costs 16% (400), for 6.5x/10.9x/
//      18.4x less error. promote-128 is the knee. At BN=256 it is NOT viable: 128 accumulators
//      + a 128-register f32 shadow exceeds the 256-register file and ptxas spills hard (138 STL
//      / 142 LDL, 248 B local, 135 vs 215 TF/s = -37%). So the accuracy/tile-size choice is
//      coupled: BN=256 buys throughput at wide N but forfeits two-level accumulation.
//   7. Per-row a_scale[m] x per-col w_scale[n] really is free: the (row,col) map already gives
//      both indices, so the epilogue costs BN/8 * 2 scalar loads per tile, amortised over K.
//
// ===========================================================================================
// WHAT plow's PACKET / ABI MUST CARRY FOR TMA (see runtime/common/dev_isa.h)
// ===========================================================================================
// A CUtensorMap is 128 B, 64-B aligned, and can ONLY be built on the host by the driver call
// cuTensorMapEncodeTiled. It embeds the RAW GLOBAL ADDRESS plus globalDim/globalStride/boxDim/
// swizzle. plow resolves operands at runtime as prog.tensors[handle] inside ONE persistent
// megakernel launch, so the CUTLASS-style `__grid_constant__ CUtensorMap` kernel parameter used
// in this file is NOT available to it. Consequences:
//
//   1. Descriptors must live in a DEVICE-GLOBAL table parallel to prog.tensors:
//        const CUtensorMap* prog.tmaps;   // n_tmap x 128 B, host-filled, 64-B aligned
//      cp.async.bulk.tensor accepts a generic/global address for the map, so this is legal with
//      no fence as long as the host writes it before launch and never mutates it afterwards.
//      Device-side mutation (e.g. rebinding a MoE expert base) requires tensormap.replace +
//      tensormap.cp_fenceproxy.global.shared::cta.release.cluster.sync.aligned and a re-acquire;
//      that is expensive and should be treated as a non-goal.
//   2. A descriptor is keyed by (tensor handle, box rows), NOT by tensor handle alone, because
//      boxDim[1] is BM for the A operand and BN for the B operand. BK is pinned to 128 by the
//      128 B swizzle rule (boxDim[0] * 1 B <= 128 B), so K never enters the key.
//   3. The packet does NOT need to grow. PlowDevInst is 64 B with t[8] u16 handles and i[8] u32
//      integers; the fp8 prefill GEMM ops (PLOW_DOP_GEMM_MED_FP8=34 / _SMALL_FP8=35 / _GLU_FP8)
//      use only i0=M i1=N i2=K i4=a_row0, leaving i[5]/i[6]/i[7] free. Put the A and B descriptor
//      indices there: &prog.tmaps[in->i[5]] and &prog.tmaps[in->i[6]]. That is one extra u32 load
//      each, already in the same 64-B packet line the interpreter fetches.
//   4. The blob needs a descriptor SECTION so the runtime can encode after it resolves pointers:
//        { u16 tensor_handle; u16 box_rows; u8 dtype; u8 rank; u32 dim0 /*K*/, dim1 /*rows*/; }
//      ~12 B per entry, encoded once at load. Re-encode is mandatory whenever a base pointer
//      moves (VMM growth, expert rebinding) -- the address is baked into the map.
//   5. Hard preconditions the compiler must assert: row stride (K bytes for e4m3) is a multiple
//      of 16; the base pointer is 16-B aligned; the destination smem tile is >=128-B aligned
//      (this file uses 1024 B, which the wgmma 128 B swizzle descriptor requires anyway).
//      OOB rows and OOB K are ZERO-FILLED by the TMA engine, which is exactly the ragged-M /
//      ragged-K behaviour plow needs, and it replaces the cp.async src-size zero-fill trick.
//
// HOW MANY DESCRIPTORS DOES A GEMMA-4 12B fp8 PREFILL NEED?
//   Gemma-4 ~12B: 48 layers, hidden 3840, inter 15360, 16 q / 8 kv heads, head_dim 256.
//   WEIGHTS (B operand, one descriptor per weight per BN class):
//     per layer: q[4096,3840] k[2048,3840] v[2048,3840] o[3840,4096] Wgate[15360,3840]
//                Wup[15360,3840] Wdown[3840,15360]  = 7
//     48 layers x 7 = 336 descriptors for a single BN. Finding 5 says the tuner wants BN=128
//     for the N=4096 GEMMs and BN=256 for the N=15360 ones -- that is still ONE box class per
//     weight, because each weight is consumed by exactly one op. So 336, and only if a weight
//     is shared across two differently-tiled ops does it need a second: <= 672 worst case.
//   ACTIVATIONS (A operand, keyed by (tensor, BM, M-bucket)): plow emits DOP_QUANT_FP8 once per
//     activation and reuses it, so ~4 distinct xq tensors per layer (pre-qkv, pre-o, pre-gate/up,
//     pre-down) = 192, times the number of BM classes actually used (1-2) and once per prefill
//     bucket. With 2 BM classes: <= 384.
//   TOTAL: ~530 typical, <= ~1050 worst case = 66-134 KB of descriptor table. Negligible next to
//   a 12 GB fp8 checkpoint, and it is read-only, host-built, and L2-resident.
//
// BUILD (executables MUST use -gencode; -arch=sm_90a -o exe is rejected, -arch=native -> sm_90):
//   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 \
//     -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common -I runtime/nvidia \
//     -include cstdint runtime/nvidia/experiments/tma_ws_gemm_fp8.cu -lcuda -o <bin>
// RUN (GPU serialised):  flock /tmp/plow_gpu.lock <bin>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA ERR %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_)); exit(1);} }while(0)
#define CKD(x) do{ CUresult r_=(x); if(r_!=CUDA_SUCCESS){ const char* s_="?"; \
  cuGetErrorString(r_,&s_); printf("CU ERR %s:%d %s -> %s\n",__FILE__,__LINE__,#x,s_); exit(1);} }while(0)

typedef __nv_fp8_e4m3 fp8;

// ---------------------------------------------------------------- fixed geometry
// BK is pinned to 128 e4m3 = 128 B = exactly one 128B-swizzle atom row. This is simultaneously
// (a) the maximum TMA box inner dimension allowed with CU_TENSOR_MAP_SWIZZLE_128B and
// (b) the layout the validated wgmma descriptor (LBO=16, SBO=1024, swz=1) describes.
static constexpr int BK    = 128;
static constexpr int KSUB  = BK / 32;   // wgmma k32 substeps per staged tile
static constexpr int CHUNK = BK / 16;   // 16 B chunks per smem row
static constexpr uint64_t LBO = 16, SBO = 1024;
static constexpr int SWZ = 1;           // descriptor swizzle mode 1 = 128 B

// ---------------------------------------------------------------- device primitives
__device__ __forceinline__ uint32_t smem_u32(const void* p){
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint64_t desc_enc(uint64_t x){ return (x & 0x3FFFFull) >> 4; }
__device__ __forceinline__ uint64_t make_desc(const void* p){
  uint64_t d = desc_enc((uint64_t)__cvta_generic_to_shared(p));
  d |= desc_enc(LBO) << 16;
  d |= desc_enc(SBO) << 32;
  d |= (uint64_t)SWZ << 62;             // matrix base offset stays 0 (tiles are 1024 B aligned)
  return d;
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt){
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void fence_bar_init(){
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b){
  asm volatile("{ .reg .b64 s; mbarrier.arrive.shared::cta.b64 s, [%0]; }" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_mbar_arrive(uint64_t* b){   // cp.async completion -> mbarrier
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int phase){
  asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;"
               " @!p bra W%=; }" :: "r"(smem_u32(b)), "r"(phase) : "memory");
}
// TMA: one 2-D tile, global -> smem, tx-counted on `bar`. Coordinates are (k, row).
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int k, int row, uint32_t bar){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(map), "r"(k), "r"(row), "r"(bar) : "memory");
}
__device__ __forceinline__ void cp16(void* smem, const void* gmem, int src_bytes){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"(smem_u32(smem)), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void wg_fence(){ asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit(){ asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template<int N> __device__ __forceinline__ void wg_wait(){
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}
// 128 B swizzle store offset (cp.async control path only; TMA does this in hardware)
__device__ __forceinline__ int swz_off(int r, int c){ return r * BK + ((c ^ (r & 7)) * 16); }
__device__ __forceinline__ char* align1k(void* p){
  uint32_t o = smem_u32(p) & 1023u;
  return (char*)p + ((1024u - o) & 1023u);
}

// ---------------------------------------------------------------- fp8 wgmma (generated)
// wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 -- SS form, both operands from
// smem descriptors. fp8 shapes take NO trans immediates (K-major only): d, a, b, scale-d,
// imm-scale-a, imm-scale-b. scale_d is a runtime predicate so a promotion can reset D.
__device__ __forceinline__ void wgmma_n128(float* d, uint64_t da, uint64_t db, int sd){
  asm volatile(
    "{\n.reg .pred p;\n"
    "setp.ne.b32 p, %66, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {"
    "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,"
    "%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,"
    "%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63"
    "}, %64, %65, p, 1, 1;\n}\n"
    :
      "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),
      "+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),
      "+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),
      "+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),
      "+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),
      "+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),
      "+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),
      "+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
    : "l"(da), "l"(db), "r"(sd));
}

// wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 -- SS form, both operands from
// smem descriptors. fp8 shapes take NO trans immediates (K-major only): d, a, b, scale-d,
// imm-scale-a, imm-scale-b. scale_d is a runtime predicate so a promotion can reset D.
__device__ __forceinline__ void wgmma_n256(float* d, uint64_t da, uint64_t db, int sd){
  asm volatile(
    "{\n.reg .pred p;\n"
    "setp.ne.b32 p, %130, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 {"
    "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,"
    "%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,"
    "%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,"
    "%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,"
    "%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,"
    "%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127"
    "}, %128, %129, p, 1, 1;\n}\n"
    :
      "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),
      "+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),
      "+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),
      "+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),
      "+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),
      "+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),
      "+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),
      "+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+f"(d[64]),
      "+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),"+f"(d[72]),
      "+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),"+f"(d[80]),
      "+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),"+f"(d[88]),
      "+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),"+f"(d[96]),
      "+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),"+f"(d[104]),
      "+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),
      "+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),
      "+f"(d[119]),"+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),
      "+f"(d[126]),"+f"(d[127])
    : "l"(da), "l"(db), "r"(sd));
}

template<int BN> __device__ __forceinline__ void wgmma_bn(float* d, uint64_t da, uint64_t db, int sd);
template<> __device__ __forceinline__ void wgmma_bn<128>(float* d, uint64_t da, uint64_t db, int sd){ wgmma_n128(d,da,db,sd); }
template<> __device__ __forceinline__ void wgmma_bn<256>(float* d, uint64_t da, uint64_t db, int sd){ wgmma_n256(d,da,db,sd); }

// ---------------------------------------------------------------- config
// PRODM : 0 = cp.async producer (store-side XOR swizzle)   [A/B control]
//         1 = TMA producer      (hardware swizzle from the tensor map)
// MODE  : 0 = no register control at all (ptxas picks one count for the whole CTA)
//         1 = __maxnreg__(ENTRY) clamp only
//         2 = __maxnreg__(ENTRY) + setmaxnreg.dec(PREG) / .inc(CREG)
// PROMK : 0 = raw wgmma accumulation; else promote the wgmma accumulator into an f32 CUDA-core
//         shadow every PROMK K-elements (DeepGEMM two-level accumulation). Must divide/be
//         divided by BK=128 and be a multiple of 32.
template<int BM,int BN,int NS,int PROMK,int PRODM,int MODE,int PREG,int CREG>
struct Cfg {
  static constexpr int CWG     = BM / 64;              // consumer warpgroups (1 m64 slab each)
  static constexpr int THREADS = 128 * (CWG + 1);      // + 1 producer warpgroup
  static constexpr int NACC    = BN / 2;               // f32 accumulators per thread
  static constexpr int ATILE   = BM * BK;              // bytes
  static constexpr int BTILE   = BN * BK;              // bytes
  static constexpr int MBAR    = 2 * NS * 8;
  static constexpr int SMEM    = MBAR + 1024 + NS * (ATILE + BTILE);
  static constexpr int PS      = PROMK / 32;           // wgmma k32 substeps between promotions
};

// ---------------------------------------------------------------- kernel body
template<int BM,int BN,int NS,int PROMK,int PRODM,int MODE,int PREG,int CREG>
__device__ __forceinline__ void gemm_body(const CUtensorMap* mapA, const CUtensorMap* mapB,
                                          const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                          float* __restrict__ C, const float* __restrict__ as,
                                          const float* __restrict__ ws, int M, int N, int K){
  using G = Cfg<BM,BN,NS,PROMK,PRODM,MODE,PREG,CREG>;
  constexpr int CWG = G::CWG, NACC = G::NACC, ATILE = G::ATILE, BTILE = G::BTILE, PS = G::PS;

  extern __shared__ __align__(128) char plow_smem[];
  uint64_t* full  = (uint64_t*)plow_smem;
  uint64_t* empty = full + NS;
  char* base = align1k(plow_smem + G::MBAR);
  uint8_t* As = (uint8_t*)base;                 // [NS][BM][BK] 128B-swizzled
  uint8_t* Bs = (uint8_t*)base + NS * ATILE;    // [NS][BN][BK] 128B-swizzled

  const int tid = threadIdx.x;
  if (tid < NS) {
    // TMA: only the elected producer thread arrives on `full`. cp.async: all 128 producer threads.
    mbar_init(full + tid, PRODM ? 1 : 128);
    mbar_init(empty + tid, CWG);                // one elected thread per consumer warpgroup
  }
  __syncthreads();
  fence_bar_init();

  const int mtiles = (M + BM - 1) / BM, ntiles = (N + BN - 1) / BN;
  const int total = mtiles * ntiles, ksteps = (K + BK - 1) / BK;

  if (tid < 128) {
    // ======================= producer warpgroup =======================
    if constexpr (MODE == 2) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(PREG));
    if constexpr (PRODM) {
      // TMA: a single elected thread issues 2 bulk-tensor copies per stage. The other 127
      // producer threads have nothing to do and retire immediately, freeing their (already
      // dec'd) registers -- this is the whole point of moving the producer to TMA.
      if (tid == 0) {
        int st = 0;
        for (int t = blockIdx.x; t < total; t += gridDim.x) {
          const int tm = (t / ntiles) * BM, tn = (t % ntiles) * BN;
          for (int ks = 0; ks < ksteps; ks++, st++) {
            const int s = st % NS;
            if (st >= NS) mbar_wait(empty + s, ((st / NS) + 1) & 1);
            mbar_expect_tx(full + s, (uint32_t)(ATILE + BTILE));
            tma2d(smem_u32(As + s * ATILE), mapA, ks * BK, tm, smem_u32(full + s));
            tma2d(smem_u32(Bs + s * BTILE), mapB, ks * BK, tn, smem_u32(full + s));
          }
        }
      }
    } else {
      int st = 0;
      for (int t = blockIdx.x; t < total; t += gridDim.x) {
        const int tm = (t / ntiles) * BM, tn = (t % ntiles) * BN;
        for (int ks = 0; ks < ksteps; ks++, st++) {
          const int s = st % NS;
          if (st >= NS) mbar_wait(empty + s, ((st / NS) + 1) & 1);
          uint8_t* Ad = As + s * ATILE;
          uint8_t* Bd = Bs + s * BTILE;
          const int k0 = ks * BK;
          // NOTE: deliberately left for ptxas to unroll -- clamping this loop cripples the
          // control. This is the fair cp.async A/B: same tiles, same 128 B swizzle, same warp
          // specialisation, same ring depth; only the transport differs.
#pragma unroll
          for (int L = tid; L < BM * CHUNK; L += 128) {
            const int r = L / CHUNK, c = L % CHUNK, gr = tm + r, gk = k0 + c * 16;
            int bytes = 0; const uint8_t* g = A;
            if (gr < M && gk < K) { g = A + (size_t)gr * K + gk; int rem = K - gk; bytes = rem < 16 ? rem : 16; }
            cp16(Ad + swz_off(r, c), g, bytes);
          }
#pragma unroll
          for (int L = tid; L < BN * CHUNK; L += 128) {
            const int r = L / CHUNK, c = L % CHUNK, gr = tn + r, gk = k0 + c * 16;
            int bytes = 0; const uint8_t* g = B;
            if (gr < N && gk < K) { g = B + (size_t)gr * K + gk; int rem = K - gk; bytes = rem < 16 ? rem : 16; }
            cp16(Bd + swz_off(r, c), g, bytes);
          }
          cp_mbar_arrive(full + s);
        }
      }
    }
  } else {
    // ======================= consumer warpgroups =======================
    if constexpr (MODE == 2) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(CREG));
    const int ctid = tid - 128, cwg = ctid >> 7, lt = ctid & 127;
    const int warp = lt >> 5, lane = lt & 31;
    const int arow = cwg * 64;                   // this warpgroup's m64 slab inside the BM tile

    float acc[NACC];
    float pacc[PROMK ? NACC : 1];
    int st = 0;
    for (int t = blockIdx.x; t < total; t += gridDim.x) {
      const int tm = (t / ntiles) * BM, tn = (t % ntiles) * BN;
#pragma unroll
      for (int i = 0; i < NACC; i++) acc[i] = 0.f;
      if constexpr (PROMK) {
#pragma unroll
        for (int i = 0; i < NACC; i++) pacc[i] = 0.f;
      }
      int reset = 0, prev = -1;

      for (int ks = 0; ks < ksteps; ks++, st++) {
        const int s = st % NS;
        mbar_wait(full + s, (st / NS) & 1);
        const uint8_t* Ac = As + s * ATILE + arow * BK;
        const uint8_t* Bc = Bs + s * BTILE;
        wg_fence();
        if constexpr (PS > 0 && PS < KSUB) {
          // promotion strictly inside a stage (PROMK < BK)
#pragma unroll
          for (int sub = 0; sub < KSUB; sub++) {
            wgmma_bn<BN>(acc, make_desc(Ac + sub * 32), make_desc(Bc + sub * 32), reset ? 0 : 1);
            reset = 0;
            if ((sub + 1) % PS == 0) {
              wg_commit(); wg_wait<0>();
#pragma unroll
              for (int j = 0; j < NACC; j++) pacc[j] += acc[j];
              reset = 1;
              if (sub + 1 < KSUB) wg_fence();
            }
          }
          if (lt == 0) mbar_arrive(empty + s);        // all groups already drained
        } else {
#pragma unroll
          for (int sub = 0; sub < KSUB; sub++) {
            wgmma_bn<BN>(acc, make_desc(Ac + sub * 32), make_desc(Bc + sub * 32), reset ? 0 : 1);
            reset = 0;
          }
          wg_commit();
          const bool prom = (PS > 0) && (((ks + 1) % (PS / KSUB == 0 ? 1 : PS / KSUB)) == 0);
          if (prom) {
            wg_wait<0>();
#pragma unroll
            for (int j = 0; j < NACC; j++) pacc[j] += acc[j];
            reset = 1;
            if (lt == 0) { if (prev >= 0) mbar_arrive(empty + prev); mbar_arrive(empty + s); }
            prev = -1;
          } else {
            wg_wait<1>();                              // keep one wgmma group in flight
            if (lt == 0 && prev >= 0) mbar_arrive(empty + prev);
            prev = s;
          }
        }
      }
      wg_wait<0>();
      if (lt == 0 && prev >= 0) mbar_arrive(empty + prev);
      if constexpr (PROMK) {
        if (!reset) {
#pragma unroll
          for (int j = 0; j < NACC; j++) pacc[j] += acc[j];
        }
#pragma unroll
        for (int j = 0; j < NACC; j++) acc[j] = pacc[j];
      }

      // ---- epilogue: wgmma C-fragment -> global, per-row x per-col dequant scales ----
      const int r0 = tm + arow + warp * 16 + (lane >> 2), r1 = r0 + 8;
      const float sa0 = (r0 < M) ? as[r0] : 0.f, sa1 = (r1 < M) ? as[r1] : 0.f;
#pragma unroll
      for (int g = 0; g < BN / 8; g++) {
        const int c0 = tn + 8 * g + 2 * (lane & 3), c1 = c0 + 1;
        const float w0 = (c0 < N) ? ws[c0] : 0.f, w1 = (c1 < N) ? ws[c1] : 0.f;
        if (r0 < M) {
          if (c0 < N) C[(size_t)r0 * N + c0] = acc[4 * g + 0] * sa0 * w0;
          if (c1 < N) C[(size_t)r0 * N + c1] = acc[4 * g + 1] * sa0 * w1;
        }
        if (r1 < M) {
          if (c0 < N) C[(size_t)r1 * N + c0] = acc[4 * g + 2] * sa1 * w0;
          if (c1 < N) C[(size_t)r1 * N + c1] = acc[4 * g + 3] * sa1 * w1;
        }
      }
    }
  }
}

// ---------------------------------------------------------------- kernel entries
// The register cap MUST be __maxnreg__, not __launch_bounds__ -- ptxas silently drops the
// setmaxnreg effect with the latter (established in hopper_warpspec_prefill.cu).
template<int BM,int BN,int NS,int PROMK,int PRODM,int MODE,int ENTRY,int PREG,int CREG>
__global__ void __maxnreg__(ENTRY) k_cap(const __grid_constant__ CUtensorMap mapA,
                                         const __grid_constant__ CUtensorMap mapB,
                                         const uint8_t* A, const uint8_t* B, float* C,
                                         const float* as, const float* ws, int M, int N, int K){
  gemm_body<BM,BN,NS,PROMK,PRODM,MODE,PREG,CREG>(&mapA,&mapB,A,B,C,as,ws,M,N,K);
}
template<int BM,int BN,int NS,int PROMK,int PRODM,int ENTRY,int PREG,int CREG>
__global__ void k_free(const __grid_constant__ CUtensorMap mapA,
                       const __grid_constant__ CUtensorMap mapB,
                       const uint8_t* A, const uint8_t* B, float* C,
                       const float* as, const float* ws, int M, int N, int K){
  gemm_body<BM,BN,NS,PROMK,PRODM,0,PREG,CREG>(&mapA,&mapB,A,B,C,as,ws,M,N,K);
}

// ---------------------------------------------------------------- tensor-core ceiling probe
// The 1979 TF/s figure is a 700 W SXM datasheet number. This part is an H100 NVL capped at
// 310 W, so it is not a fair denominator on its own. This kernel measures what the fp8 tensor
// cores can actually retire on THIS GPU: 2 consumer warpgroups per CTA (same as the real
// kernel), 1 CTA per SM, wgmma m64n128k32 issued back to back out of a resident smem tile with
// ZERO global traffic and zero pipeline bookkeeping. That is the real 100%-line.
__global__ void __launch_bounds__(256) k_qgmma_rate(float* out, int iters){
  extern __shared__ __align__(128) char sm[];
  char* base = align1k(sm);
  for (int i = threadIdx.x; i < 64*BK + 128*BK; i += 256) base[i] = 0;   // e4m3 0x00 == 0.0f
  __syncthreads();
  const uint64_t da = make_desc(base), db = make_desc(base + 64*BK);
  float acc[64];
#pragma unroll
  for (int i=0;i<64;i++) acc[i]=0.f;
  wg_fence();
  for (int i=0;i<iters;i++){
#pragma unroll
    for (int j=0;j<4;j++) wgmma_n128(acc, da, db, 1);
    wg_commit();
    wg_wait<1>();
  }
  wg_wait<0>();
  float s=0;
#pragma unroll
  for (int i=0;i<64;i++) s+=acc[i];
  if (s != 0.f) out[threadIdx.x] = s;      // never taken; defeats DCE
}
static double qgmma_ceiling(int sms){
  const int SM = 64*BK + 128*BK + 1024, IT = 20000;
  CK(cudaFuncSetAttribute((const void*)k_qgmma_rate, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  float* d; CK(cudaMalloc(&d, 1024));
  k_qgmma_rate<<<sms,256,SM>>>(d, 2000); CK(cudaDeviceSynchronize());
  double best=0;
  for(int r=0;r<5;r++){
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    k_qgmma_rate<<<sms,256,SM>>>(d, IT);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    // 2 warpgroups/CTA x 4 wgmma x m64n128k32 MACs x 2 flop
    double fl = 2.0*sms*2.0*4.0*IT*64.0*128.0*32.0*2.0/2.0;
    fl = (double)sms * 2.0 /*warpgroups*/ * 4.0 /*wgmma*/ * IT * (64.0*128.0*32.0*2.0);
    double tf = fl/((double)ms*1e-3)/1e12;
    if(tf>best) best=tf;
  }
  cudaFree(d);
  return best;
}

// ---------------------------------------------------------------- host variant table
struct Var {
  const char* name;
  int BM, BN, NS, PROMK, PRODM, MODE, ENTRY, PREG, CREG, THREADS;
  size_t smem;
  void (*launch)(int, const CUtensorMap&, const CUtensorMap&, const uint8_t*, const uint8_t*,
                 float*, const float*, const float*, int, int, int);
  void (*prep)(int*, size_t*, int*);
};

template<int BM,int BN,int NS,int PROMK,int PRODM,int MODE,int ENTRY,int PREG,int CREG>
struct Runner {
  using G = Cfg<BM,BN,NS,PROMK,PRODM,MODE,PREG,CREG>;
  static auto fn(){
    if constexpr (MODE == 0) return k_free<BM,BN,NS,PROMK,PRODM,ENTRY,PREG,CREG>;
    else                     return k_cap <BM,BN,NS,PROMK,PRODM,MODE,ENTRY,PREG,CREG>;
  }
  static void launch(int grid, const CUtensorMap& ma, const CUtensorMap& mb, const uint8_t* A,
                     const uint8_t* B, float* C, const float* as, const float* ws,
                     int M, int N, int K){
    fn()<<<grid, G::THREADS, G::SMEM>>>(ma, mb, A, B, C, as, ws, M, N, K);
  }
  static void prep(int* regs, size_t* local, int* occ){
    CK(cudaFuncSetAttribute((const void*)fn(), cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, (const void*)fn()));
    *regs = a.numRegs; *local = a.localSizeBytes;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(occ, (const void*)fn(), G::THREADS, G::SMEM));
  }
  static Var var(const char* nm){
    return Var{nm, BM, BN, NS, PROMK, PRODM, MODE, ENTRY, PREG, CREG, G::THREADS,
               (size_t)G::SMEM, &launch, &prep};
  }
};

#define VAR(nm,BM,BN,NS,PK,PM,MD,EN,PR,CR) \
  Runner<BM,BN,NS,PK,PM,MD,EN,PR,CR>::var(nm)

// ---------------------------------------------------------------- host helpers
static inline uint8_t f2e4m3(float f){ __nv_fp8_e4m3 v(f); return (uint8_t)v.__x; }
static inline float   e4m32f(uint8_t b){ __nv_fp8_e4m3 v; v.__x = (__nv_fp8_storage_t)b; return (float)v; }

static uint32_t g_xs = 0x1234567u;
static float frand(){ g_xs^=g_xs<<13; g_xs^=g_xs>>17; g_xs^=g_xs<<5;
                      return ((g_xs>>8)*(1.0f/8388608.0f))-1.0f; }

// TMA tensor map over a [rows, K] e4m3 k-contiguous matrix; tile box = (BK, boxRows).
// SWIZZLE_128B is declared HERE, so the TMA engine writes the smem tile already swizzled.
static void make_map(CUtensorMap* map, const void* p, int rows, int K, int boxRows){
  memset(map, 0, sizeof(*map));
  uint64_t gdim[2]   = { (uint64_t)K, (uint64_t)rows };
  uint64_t gstride[1]= { (uint64_t)K };          // bytes; must be a multiple of 16
  uint32_t bdim[2]   = { (uint32_t)BK, (uint32_t)boxRows };
  uint32_t estr[2]   = { 1, 1 };
  CKD(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)p, gdim, gstride,
                             bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// f32 CPU oracle over the SAME e4m3 bytes, with per-row / per-col scales.
static void cpu_oracle(const uint8_t* A, const uint8_t* B, std::vector<float>& C,
                       int M, int N, int K, const std::vector<float>& as, const std::vector<float>& ws){
  std::vector<float> Af((size_t)M*K), Bf((size_t)N*K);
  for (size_t i=0;i<(size_t)M*K;i++) Af[i]=e4m32f(A[i]);
  for (size_t i=0;i<(size_t)N*K;i++) Bf[i]=e4m32f(B[i]);
  unsigned T=std::thread::hardware_concurrency(); if(!T)T=8; if((int)T>M)T=M;
  std::vector<std::thread> th;
  auto work=[&](int m0,int m1){
    for(int m=m0;m<m1;m++){ const float* a=&Af[(size_t)m*K];
      for(int n=0;n<N;n++){ const float* b=&Bf[(size_t)n*K]; float s=0.f;
        for(int k=0;k<K;k++) s+=a[k]*b[k];
        C[(size_t)m*N+n]=s*as[m]*ws[n]; } } };
  int per=(M+(int)T-1)/(int)T;
  for(unsigned t=0;t<T;t++){ int m0=(int)t*per, m1=(m0+per>M)?M:m0+per; if(m0<m1) th.emplace_back(work,m0,m1); }
  for(auto& x:th) x.join();
}
static double relL2(const std::vector<float>& d, const std::vector<float>& r){
  double n=0,e=0; for(size_t i=0;i<r.size();i++){ double x=(double)d[i]-(double)r[i]; n+=x*x; e+=(double)r[i]*(double)r[i]; }
  return e>0?std::sqrt(n/e):std::sqrt(n);
}

struct Shape { int M,N,K; };

struct Buf {
  int M,N,K; uint8_t *dA,*dB; float *dC,*das,*dws;
  std::vector<uint8_t> hA,hB; std::vector<float> has,hws,ref;
  CUtensorMap mA64,mA128,mB128,mB256;
  void init(int M_,int N_,int K_,bool oracle){
    M=M_;N=N_;K=K_;
    hA.resize((size_t)M*K); hB.resize((size_t)N*K); has.resize(M); hws.resize(N);
    for(size_t i=0;i<hA.size();i++) hA[i]=f2e4m3(frand());
    for(size_t i=0;i<hB.size();i++) hB[i]=f2e4m3(frand());
    for(int i=0;i<M;i++) has[i]=0.25f+0.5f*((i%7)/7.0f);
    for(int i=0;i<N;i++) hws[i]=0.5f+1.0f*((i%5)/5.0f);
    CK(cudaMalloc(&dA,hA.size())); CK(cudaMalloc(&dB,hB.size()));
    CK(cudaMalloc(&dC,(size_t)M*N*4)); CK(cudaMalloc(&das,M*4)); CK(cudaMalloc(&dws,N*4));
    CK(cudaMemcpy(dA,hA.data(),hA.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hB.data(),hB.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(das,has.data(),M*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dws,hws.data(),N*4,cudaMemcpyHostToDevice));
    make_map(&mA64, dA, M, K, 64);  make_map(&mA128, dA, M, K, 128);
    make_map(&mB128,dB, N, K, 128); make_map(&mB256, dB, N, K, 256);
    if(oracle){ ref.assign((size_t)M*N,0.f); cpu_oracle(hA.data(),hB.data(),ref,M,N,K,has,hws); }
  }
  const CUtensorMap& amap(int BM) const { return BM==64?mA64:mA128; }
  const CUtensorMap& bmap(int BN) const { return BN==128?mB128:mB256; }
  void free_(){ cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(das);cudaFree(dws); }
};

static int g_sm = 132;
static int grid_for(const Var& v, int M, int N){
  int t = ((M+v.BM-1)/v.BM) * ((N+v.BN-1)/v.BN);
  return t < g_sm ? t : g_sm;
}
static void run(const Var& v, Buf& b){
  v.launch(grid_for(v,b.M,b.N), b.amap(v.BM), b.bmap(v.BN), b.dA, b.dB, b.dC, b.das, b.dws,
           b.M, b.N, b.K);
}
static double check(const Var& v, Buf& b){
  CK(cudaMemset(b.dC,0,(size_t)b.M*b.N*4));
  run(v,b); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> h((size_t)b.M*b.N);
  CK(cudaMemcpy(h.data(),b.dC,h.size()*4,cudaMemcpyDeviceToHost));
  return relL2(h,b.ref);
}
// MEASUREMENT CAVEAT (read before trusting any number below): this H100 NVL is (a) power
// capped at 310 W -- SM clock was observed swinging 1530..1785 MHz under sustained fp8 load --
// and (b) SHARED with another tenant (92 GB resident, alternating idle / 100%-utilised phases
// lasting seconds). Measuring variants one after another therefore assigns each variant a
// different amount of interference: in one run a single variant caught a clean stretch and
// reported 654 TF/s across all nine of its windows while every other variant sat at ~320.
// FIX: the sweep is INTERLEAVED -- PASSES full round-robin passes over (variant x shape), one
// 100-iteration window each, keeping the per-cell MAXIMUM. Every variant then gets the same
// number of chances at an uncontended phase, and the max is the cleanest observation.
// Clock locking is not permitted for this user, so this is the best available methodology.
static constexpr int PASSES = 30;
static double one_window(const Var& v, Buf& b, int iters){
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for(int i=0;i<iters;i++) run(v,b);
  cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (2.0*b.M*b.N*b.K)/((double)ms/iters*1e-3)/1e12;
}

// H100 SXM/NVL dense fp8 tensor-core peak (no sparsity)
static constexpr double PEAK = 1979.0;

int main(int argc, char** argv){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  g_sm = p.multiProcessorCount;
  int maxsm = p.sharedMemPerBlockOptin;
  printf("== TMA + warp-spec + 128B swizzle + fp8 WGMMA + two-level accumulation (sm_90a) ==\n");
  printf("device: %s  SMs=%d  smem/block optin=%d B  BK=%d\n\n", p.name, g_sm, maxsm, BK);

  Var V[] = {
    VAR("tma_128x128_ns3",128,128,3,0,1,2,168,40,232),
    VAR("tma_128x128_ns4",128,128,4,0,1,2,168,40,232),
    VAR("tma_128x128_ns5",128,128,5,0,1,2,168,40,232),
    VAR("tma_128x128_ns6",128,128,6,0,1,2,168,40,232),
    VAR("tma_128x256_ns3",128,256,3,0,1,2,168,40,232),
    VAR("tma_128x256_ns4",128,256,4,0,1,2,168,40,232),
    VAR("tma_64x128_ns4",64,128,4,0,1,2,200,32,248),
    VAR("tma_64x128_ns6",64,128,6,0,1,2,200,32,248),
    VAR("tma_64x128_ns8",64,128,8,0,1,2,200,32,248),
    VAR("tma_64x128_ns4_occ2",64,128,4,0,1,2,128,32,224),
    VAR("tma_64x256_ns4",64,256,4,0,1,2,200,32,248),
    VAR("tma_64x256_ns5",64,256,5,0,1,2,200,32,248),
    VAR("tma_128x128_ns5_dec24",128,128,5,0,1,2,168,24,240),
    VAR("tma_128x128_ns5_dec64",128,128,5,0,1,2,168,64,216),
    VAR("tma_128x128_ns5_dec88",128,128,5,0,1,2,168,88,208),
    VAR("tma_128x128_ns5_free",128,128,5,0,1,0,168,0,0),
    VAR("tma_128x128_ns5_clamp",128,128,5,0,1,1,168,0,0),
    VAR("cpa_128x128_ns5_dec40",128,128,5,0,0,2,168,40,232),
    VAR("cpa_128x128_ns5_dec88",128,128,5,0,0,2,168,88,208),
    VAR("cpa_128x128_ns5_free",128,128,5,0,0,0,168,0,0),
    VAR("cpa_128x256_ns4_free",128,256,4,0,0,0,168,0,0),
    VAR("tma_128x128_ns5_prom64",128,128,5,64,1,2,168,40,232),
    VAR("tma_64x128_ns6_prom64",64,128,6,64,1,2,200,32,248),
    VAR("tma_128x128_ns5_prom128",128,128,5,128,1,2,168,40,232),
    VAR("tma_64x128_ns6_prom128",64,128,6,128,1,2,200,32,248),
    VAR("tma_128x128_ns5_prom256",128,128,5,256,1,2,168,40,232),
    VAR("tma_64x128_ns6_prom256",64,128,6,256,1,2,200,32,248),
    VAR("tma_128x256_ns4_prom128",128,256,4,128,1,2,168,40,232),
  };
  const int NV = (int)(sizeof(V)/sizeof(V[0]));

  // ---- prep: register / spill accounting ----
  std::vector<int> regs(NV), occ(NV); std::vector<size_t> loc(NV); std::vector<char> ok(NV,1);
  printf("--- variants (regs = ptxas allocation for the entry region = what drives occupancy) ---\n");
  printf("%-26s %-9s %-3s %-5s %-5s %-6s %8s %5s %6s %8s %4s %5s %5s\n",
         "variant","tile","NS","prod","mode","promK","smemKB","thr","regs","localB","occ","dec","inc");
  for (int i=0;i<NV;i++){
    if ((int)V[i].smem > maxsm){ ok[i]=0; printf("%-26s SKIP (smem %zu > %d)\n",V[i].name,V[i].smem,maxsm); continue; }
    V[i].prep(&regs[i],&loc[i],&occ[i]);
    char tile[16]; snprintf(tile,sizeof tile,"%dx%d",V[i].BM,V[i].BN);
    const char* md = V[i].MODE==0?"free":(V[i].MODE==1?"clamp":"smr");
    printf("%-26s %-9s %-3d %-5s %-5s %-6d %8.1f %5d %6d %8zu %4d %5d %5d\n",
           V[i].name,tile,V[i].NS,V[i].PRODM?"tma":"cp.a",md,V[i].PROMK,
           V[i].smem/1024.0,V[i].THREADS,regs[i],loc[i],occ[i],
           V[i].MODE==2?V[i].PREG:0, V[i].MODE==2?V[i].CREG:0);
  }

  // ---- hard correctness gate ----
  Shape sh[] = {{64,128,32},{128,256,64},{512,4096,3840},{512,15360,3840},{200,4096,3840}};
  const int NSH = (int)(sizeof(sh)/sizeof(sh[0]));
  printf("\n=== CORRECTNESS (f32 CPU oracle over the same e4m3 bytes, per-row x per-col scales) ===\n");
  printf("gate relL2 < 6e-3\n\n");
  printf("%-20s %-12s %-12s %-12s %-12s %-8s\n",
         "shape (M,N,K)","no-promote","promote 64","promote 128","promote 256","gate");
  bool all_pass = true;
  // canonical config for the accuracy table
  int i_np=-1,i_p64=-1,i_p128=-1,i_p256=-1;
  for(int i=0;i<NV;i++){
    if(!ok[i]) continue;
    if(!strcmp(V[i].name,"tma_128x128_ns5"))        i_np=i;
    if(!strcmp(V[i].name,"tma_128x128_ns5_prom64")) i_p64=i;
    if(!strcmp(V[i].name,"tma_128x128_ns5_prom128"))i_p128=i;
    if(!strcmp(V[i].name,"tma_128x128_ns5_prom256"))i_p256=i;
  }
  std::vector<double> acc_np(NSH,0), acc_p128(NSH,0);
  for (int s=0;s<NSH;s++){
    Buf b; b.init(sh[s].M,sh[s].N,sh[s].K,true);
    double e0 = i_np  >=0 ? check(V[i_np],b)  : -1;
    double e1 = i_p64 >=0 ? check(V[i_p64],b) : -1;
    double e2 = i_p128>=0 ? check(V[i_p128],b): -1;
    double e3 = i_p256>=0 ? check(V[i_p256],b): -1;
    acc_np[s]=e0; acc_p128[s]=e2;
    bool pass = e0<6e-3 && e2<6e-3;
    all_pass &= pass;
    char nm[32]; snprintf(nm,sizeof nm,"(%d,%d,%d)",sh[s].M,sh[s].N,sh[s].K);
    printf("%-20s %-12.4e %-12.4e %-12.4e %-12.4e %-8s\n",nm,e0,e1,e2,e3,pass?"PASS":"FAIL");
    b.free_();
  }
  printf("\nnote: promote 128 vs none = %.2fx error reduction at K=3840 (shape 3)\n",
         acc_p128[2]>0 ? acc_np[2]/acc_p128[2] : 0.0);

  // ---- every variant must also be correct ----
  printf("\n=== per-variant correctness @ (512,4096,3840) ===\n");
  {
    Buf b; b.init(512,4096,3840,true);
    for(int i=0;i<NV;i++){
      if(!ok[i]) continue;
      double e = check(V[i],b);
      bool pass = e < 6e-3;
      if(!pass){ all_pass=false; ok[i]=0; }
      printf("  %-26s relL2=%.4e  %s\n",V[i].name,e,pass?"PASS":"FAIL");
    }
    b.free_();
  }
  printf("\nRESULT: %s\n", all_pass?"PASS":"FAIL");

  // ---- benchmark ----
  Shape bs[] = {{512,4096,3840},{512,15360,3840},{200,4096,3840}};
  const double ceil_tc = qgmma_ceiling(g_sm);
  printf("\n=== TENSOR-CORE CEILING ON THIS PART ===\n");
  printf("back-to-back wgmma m64n128k32 e4m3, resident smem operands, no global traffic,\n"
         "2 warpgroups x %d SMs: %.1f TF/s = %.1f%% of the 1979 TF/s (700 W SXM) datasheet peak.\n"
         "This 310 W NVL part is the reason the datasheet number is unreachable; treat %.0f TF/s\n"
         "as the real 100%% line for the GEMM below.\n", g_sm, ceil_tc, 100.0*ceil_tc/PEAK, ceil_tc);
  printf("\n=== THROUGHPUT (TFLOP/s; %% of %.0f TF/s H100 dense fp8 peak) ===\n",PEAK);
  printf("method: %d interleaved round-robin passes over (variant x shape), 100 iterations per\n"
         "        window, per-cell MAX kept. See the MEASUREMENT CAVEAT above.\n", PASSES);
  printf("references from the validated in-tree probes (same GPU, same oracle):\n");
  printf("  wgmma  cp.async 64x128x64 no-swizzle : 172.9 (512,4096,3840) / 169.2 (512,15360,3840)\n");
  printf("  mma.sync m16n8k32 e4m3 (f16-emulated): 116.5 / 120.9\n\n");
  std::vector<Buf> bb(3);
  for(int s=0;s<3;s++) bb[s].init(bs[s].M,bs[s].N,bs[s].K,false);
  std::vector<double> hi(NV*3,0.0), lo(NV*3,1e30);
  for(int i=0;i<NV;i++){ if(!ok[i]) continue;
    for(int s=0;s<3;s++){ for(int w=0;w<5;w++) run(V[i],bb[s]); }
  }
  CK(cudaDeviceSynchronize());
  // shuffle the visit order every pass so no variant stays phase-locked to the co-tenant
  std::vector<int> ord(NV); for(int i=0;i<NV;i++) ord[i]=i;
  uint32_t rs = 12345u;
  for(int pass=0; pass<PASSES; pass++){
    for(int i=NV-1;i>0;i--){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; int j=(int)(rs%(uint32_t)(i+1));
                             int t=ord[i]; ord[i]=ord[j]; ord[j]=t; }
    for(int oi=0;oi<NV;oi++){
      const int i = ord[oi];
      if(!ok[i]) continue;
      for(int s=0;s<3;s++){
        double tf = one_window(V[i],bb[s],100);
        if(tf>hi[i*3+s]) hi[i*3+s]=tf;
        if(tf<lo[i*3+s]) lo[i*3+s]=tf;
      }
    }
  }
  printf("%-26s","variant");
  for(int s=0;s<3;s++){ char nm[32]; snprintf(nm,sizeof nm,"%d,%d,%d",bs[s].M,bs[s].N,bs[s].K); printf(" %22s",nm); }
  printf("  %s\n","worst@sh0");
  printf("  %%%% is of the measured tensor-core ceiling (%.0f TF/s); w = waves = tiles/min(tiles,%d)\n\n", ceil_tc, g_sm);
  double best=0; int ibest=-1;
  for(int i=0;i<NV;i++){
    if(!ok[i]) continue;
    printf("%-26s",V[i].name);
    for(int s=0;s<3;s++){
      int tiles = ((bs[s].M+V[i].BM-1)/V[i].BM)*((bs[s].N+V[i].BN-1)/V[i].BN);
      printf("  %7.1f(%4.1f%%,%4.2fw)", hi[i*3+s], 100.0*hi[i*3+s]/ceil_tc,
             (double)tiles/(double)(tiles<g_sm?tiles:g_sm));
    }
    printf("  %8.0f\n", lo[i*3+0]);
    if(hi[i*3+0]>best){ best=hi[i*3+0]; ibest=i; }
  }
  for(int s=0;s<3;s++) bb[s].free_();
  if(ibest>=0)
    printf("\nBEST @ (512,4096,3840): %s  %.1f TF/s (%.1f%% of the measured tensor-core ceiling)  "
           "[BM=%d BN=%d NS=%d prod=%s mode=%s dec=%d inc=%d promote=%d]\n"
           "  vs in-tree wgmma cp.async probe 172.9 TF/s -> %.2fx\n"
           "  vs mma.sync (f16-emulated e4m3)  116.5 TF/s -> %.2fx\n",
           V[ibest].name,best,100.0*best/ceil_tc,V[ibest].BM,V[ibest].BN,V[ibest].NS,
           V[ibest].PRODM?"TMA":"cp.async", V[ibest].MODE==0?"free":(V[ibest].MODE==1?"clamp":"smr"),
           V[ibest].PREG,V[ibest].CREG,V[ibest].PROMK, best/172.9, best/116.5);
  return all_pass?0:1;
}

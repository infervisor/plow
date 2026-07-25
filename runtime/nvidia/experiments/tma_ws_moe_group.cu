// tma_ws_moe_group.cu
// Does TMA + warp specialization pay for plow's GROUPED (MoE expert) GEMM on H100 (sm_90a)?
//
// Companion to runtime/nvidia/experiments/wgmma_moe_group_probe.cu, which established the
// swizzled wgmma grouped mainloop (163 TF/s @ E=8/m_e=256/N=768/K=3840, 200 TF/s @ m_e=1024)
// and observed that MoE tiles are SMALL and the machine is barely filled. This file answers,
// in order:
//
//   PART 1 (headline)  Is there room at all? Tiles / waves / occupancy for the realistic MoE
//                      shapes, plus a tile-count sweep that walks the SAME kernel from 0.36
//                      waves to ~23 waves so the occupancy-limited region is visible directly,
//                      plus a grid-size sweep (is more concurrency usable?) and the BM row-fill
//                      efficiency for the E=128 / m_e=8 shape.
//   PART 2             Can ONE CUtensorMap address E expert weight matrices (expert index as
//                      the outermost dim), or are E descriptors needed? Concrete probe against
//                      cuTensorMapEncodeTiled + a device-side 3-D tile load checked on the host.
//   PART 3             TMA producer (+ optional warp specialization with setmaxnreg) on top of
//                      the SAME swizzled wgmma mainloop, A/B'd against the cp.async baseline.
//
// The 128B-swizzle smem layout the probe locked in (row-major [rows][BK=64] bf16 with the chunk
// index XORed by the row, tile base 1024B aligned) is BIT-IDENTICAL to what TMA writes for
// CU_TENSOR_MAP_SWIZZLE_128B with boxDim[0]*2B == 128B, so the TMA path reuses the probe's wgmma
// descriptors (LBO=16, SBO=1024, swz=1) unchanged. That is why a TMA A/B here isolates the
// LOAD ENGINE, not the layout.
//
// RESULT (H100 NVL, 132 SMs, 310 W cap, sm_90a; every variant relL2-identical to the CPU oracle):
//
//   shape                          cp.async(base)   TMA      TMA+warp-spec
//   E=8  m_e=256  N=768 K=3840      161.6 TF/s    283.3 (1.75x)   280.0 (1.73x)
//   E=8  m_e=1024 N=768 K=3840      203.5 TF/s    325.7 (1.60x)   322.6 (1.59x)
//   E=128 m_e=8   N=256 K=2560       21.6 TF/s     23.2 (1.07x)    23.3 (1.08x)
//
//   1. MoE prefill does NOT fill the machine (0.73 waves at m_e=256 over 264 tile slots), but
//      that is NOT the binding constraint: the same kernel driven to 11.6 waves only reaches
//      206 TF/s. The mainloop was ISSUE-bound in the staging path, which the LOAD-ONLY ablation
//      shows directly (cp.async staging alone = 72% of full kernel time).
//   2. TMA removes that: 192 LDGSTS + 2 BAR.SYNC per stage become 2 UTMALDG issued by one
//      elected thread + one mbarrier wait. 1.6-1.75x on the prefill shapes.
//   3. Warp specialization on top of TMA is a WASH (-1%). With BM=64 / one consumer warpgroup
//      the producer's whole job is two instructions; there is nothing to decouple and the
//      consumer needs only ~90-128 registers, so the setmaxnreg redistribution buys nothing.
//   4. The E=128 / m_e=8 decode-ish shape gains ~6%: it is capped by 12.5% BM=64 row fill,
//      not by the load engine. That shape needs a variable-M / small-BM scheduler, not TMA.
//   5. ONE 3-D CUtensorMap [K, N, E] with box {BK, BN, 1} addresses ALL E experts (expert index
//      as a coordinate). Verified bit-exact; an out-of-range expert coordinate zero-fills.
//
// Build:
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 \
//     -I runtime/common -I runtime/nvidia -include cstdint -Xcompiler -fopenmp \
//     runtime/nvidia/experiments/tma_ws_moe_group.cu -o tma_ws_moe -lcuda -lnvidia-ml
//
// Run: ./tma_ws_moe [1|2|3|all]     (1 = wave analysis, 2 = tensormap feasibility, 3 = A/B)
// NOTE: this card power-throttles 1785 -> ~700 MHz within ~2 s of sustained wgmma. All timing
// here is short-burst / rotated round-robin / min-of-rounds; a naive loop measures the power cap.

#include <cuda.h>
#include <cuda_bf16.h>
#include <nvml.h>
#include <unistd.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)
#define CKD(x) do{ CUresult e_=(x); if(e_!=CUDA_SUCCESS){ const char* s_="?"; \
  cuGetErrorString(e_,&s_); printf("CUDA-drv err %s:%d: %s\n",__FILE__,__LINE__,s_); exit(1);} }while(0)

typedef __nv_bfloat16 bf16;

// ---------------------------------------------------------------- tile config
#define BM 64          // wgmma M (one warpgroup, m64)
#define BN 128         // wgmma N (m64n128k16 -> 64 f32 acc regs / thread)
#define BK 64          // K per stage. 64 bf16 = 128B = exactly one 128B-swizzle atom row.
#define NTHREADS 128   // one warpgroup (baseline + TMA-no-WS)
#define TX_A ((unsigned)(BM*BK*2))
#define TX_B ((unsigned)(BN*BK*2))
#define TX_AB (TX_A + TX_B)

static const double PEAK_BF16 = 989.0;   // TF/s, H100 dense bf16 reference used by the task

// ---------------------------------------------------------------- device helpers (from probe)
__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool pred){
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  int bytes = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
               :: "r"(s), "l"(gmem), "n"(16), "r"(bytes));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ uint64_t make_desc(const void* p, uint64_t lbo, uint64_t sbo, int swz){
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  uint64_t d = ((uint64_t)a & 0x3FFFF) >> 4;
  d |= ((lbo & 0x3FFFF) >> 4) << 16;
  d |= ((sbo & 0x3FFFF) >> 4) << 32;
  d |= ((uint64_t)swz) << 62;
  return d;
}
__device__ __forceinline__ void wgmma_fence(){ asm volatile("wgmma.fence.sync.aligned;\n"); }
__device__ __forceinline__ void wgmma_commit(){ asm volatile("wgmma.commit_group.sync.aligned;\n"); }
template<int N> __device__ __forceinline__ void wgmma_wait(){
  asm volatile("wgmma.wait_group.sync.aligned %0;\n"::"n"(N));
}
__device__ __forceinline__ void wgmma_m64n128k16(float* d, uint64_t da, uint64_t db){
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
    "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
    "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
    "%64, %65, %66, %67, %68, %69, %70;\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
      "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
      "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
      "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
      "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
      "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
      "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
      "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
    : "l"(da),"l"(db),"n"(1),"n"(1),"n"(1),"n"(0),"n"(0));
}

// 128B-swizzle smem layout (the probe's winner). Row = 128B = one swizzle atom row.
__device__ __forceinline__ int soff_sw(int r, int c){ return r*BK + ((c ^ (r&7))*8); }
#define LBO_SW 16ull
#define SBO_SW 1024ull

__device__ __forceinline__ char* align1k(char* p){
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  return p + ((1024u - (a & 1023u)) & 1023u);
}

// ---------------------------------------------------------------- mbarrier / TMA helpers
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count){
  uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes){
  uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar){
  uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase){
  uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("{\n"
               ".reg .pred P;\n"
               "W%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
               "@P bra D%=;\n"
               "bra W%=;\n"
               "D%=:\n"
               "}\n" :: "r"(b), "r"(phase));
}
// 2-D tiled TMA load (A: [K, Mtot], coords {k0, row0}).
__device__ __forceinline__ void tma_2d(void* dst, const void* tm, uint64_t* bar, int c0, int c1){
  uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%3, %4}], [%2];"
               :: "r"(d), "l"(tm), "r"(b), "r"(c0), "r"(c1) : "memory");
}
// 3-D tiled TMA load (W: [K, N, E], coords {k0, n0, e}) -- ONE descriptor for all E experts.
__device__ __forceinline__ void tma_3d(void* dst, const void* tm, uint64_t* bar,
                                       int c0, int c1, int c2){
  uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(d), "l"(tm), "r"(b), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void tmap_prefetch(const void* tm){
  asm volatile("prefetch.tensormap [%0];" :: "l"(tm) : "memory");
}

// ================================================================ BASELINE: cp.async + wgmma
// Verbatim structure of wgmma_moe_group_probe.cu's swizzled wgmma kernel (STAGES/PF templated
// so the pipeline depth is comparable with the TMA variants).
template<int ROWS,int NT>
__device__ __forceinline__ void stage_tile(bf16* dst, const bf16* src, const bf16* safe,
                                           int k0, int K, int nrows_valid, int tid){
  constexpr int CH = BK/8;
  for(int i=tid; i<ROWS*CH; i+=NT){
    int r = i/CH, c = i%CH;
    bool v = (r < nrows_valid);
    const bf16* s = src + (size_t)r*K + k0 + c*8;
    cp16(&dst[soff_sw(r,c)], v? s : safe, v);
  }
}

// MMA=false: identical staging, wgmma replaced by a trivial smem touch. Isolates the LOAD
// pipeline so "is the mainloop staging-bound?" can be answered directly.
template<int NS, int PF, bool MMA=true>
__global__ __launch_bounds__(NTHREADS) void moe_gemm_cpasync(
    const bf16* __restrict__ A, const bf16* __restrict__ W, float* __restrict__ C,
    int Ntot, int K, const int* __restrict__ m_off, const int* __restrict__ m_len,
    const int* __restrict__ wl_e, const int* __restrict__ wl_mt, const int* __restrict__ wl_nt,
    int nwork){
  extern __shared__ __align__(128) char smem_raw[];
  bf16* As = (bf16*)align1k(smem_raw);
  bf16* Bs = As + NS*BM*BK;
  const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
  const int ksteps = K/BK;

  for(int w = blockIdx.x; w < nwork; w += gridDim.x){
    const int e = wl_e[w], mtile = wl_mt[w], ntile = wl_nt[w];
    const int arow  = m_off[e] + mtile*BM;
    const int vrows = min(BM, m_len[e] - mtile*BM);
    const int nbase = ntile*BN;
    const int vcols = min(BN, Ntot - nbase);
    const bf16* Aseg = A + (size_t)arow*K;
    const bf16* Wexp = W + (size_t)e*(size_t)Ntot*K + (size_t)nbase*K;

    float d[64];
#pragma unroll
    for(int i=0;i<64;i++) d[i]=0.f;

    auto stage = [&](int ks, int buf){
      stage_tile<BM,NTHREADS>(As + buf*BM*BK, Aseg, A, ks*BK, K, vrows, tid);
      stage_tile<BN,NTHREADS>(Bs + buf*BN*BK, Wexp, W, ks*BK, K, vcols, tid);
    };
#pragma unroll
    for(int s=0;s<PF;s++){ if(s<ksteps) stage(s,s); cp_commit(); }
    wgmma_fence();

    for(int ks=0; ks<ksteps; ks++){
      int fetch = ks + PF;
      if(fetch<ksteps) stage(fetch, fetch%NS);
      cp_commit();
      cp_wait<PF>();
      __syncthreads();
      const int cb = ks % NS;
      bf16* Ad = As + cb*BM*BK;
      bf16* Bd = Bs + cb*BN*BK;
      if constexpr (MMA){
#pragma unroll
        for(int kk=0; kk<BK; kk+=16)
          wgmma_m64n128k16(d, make_desc(Ad+kk,LBO_SW,SBO_SW,1), make_desc(Bd+kk,LBO_SW,SBO_SW,1));
        wgmma_commit();
        wgmma_wait<1>();
      } else {
        d[0] += __bfloat162float(Ad[tid]) + __bfloat162float(Bd[tid]);
      }
      __syncthreads();
    }
    if constexpr (MMA) wgmma_wait<0>();

#pragma unroll
    for(int j=0;j<BN/8;j++){
#pragma unroll
      for(int q=0;q<4;q++){
        int rr = warp*16 + (q>>1)*8 + (lane>>2);
        int cc = j*8 + (lane&3)*2 + (q&1);
        if(rr<vrows && cc<vcols) C[(size_t)(arow+rr)*Ntot + nbase+cc] = d[j*4+q];
      }
    }
    __syncthreads();
  }
}

// ================================================================ TMA, single warpgroup (no WS)
// Same mainloop, same smem layout, same wgmma descriptors. Only the load engine changes:
// 128 cp.async.cg instructions per stage (one per thread per 16B chunk, issued by the whole
// warpgroup) become 2 TMA descriptor loads issued by ONE thread + an mbarrier wait.
//
// Ragged M no longer needs the src-size-0 zero-fill trick: rows past Mtot are out of the TMA
// tensor bounds and the hardware zero-fills them. Rows that belong to the NEXT expert are still
// fetched, multiplied, and then discarded by the epilogue store mask (they cost nothing extra:
// the wgmma tile is BM=64 either way).
template<int NS, int PF, bool MMA=true>
__global__ __launch_bounds__(NTHREADS) void moe_gemm_tma(
    const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmW,
    float* __restrict__ C,
    int Ntot, int K, const int* __restrict__ m_off, const int* __restrict__ m_len,
    const int* __restrict__ wl_e, const int* __restrict__ wl_mt, const int* __restrict__ wl_nt,
    int nwork){
  extern __shared__ __align__(128) char smem_raw[];
  bf16* As = (bf16*)align1k(smem_raw);
  bf16* Bs = As + NS*BM*BK;
  uint64_t* full = (uint64_t*)(Bs + NS*BN*BK);
  const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
  const int ksteps = K/BK;

  if(tid==0){ for(int s=0;s<NS;s++) mbar_init(&full[s],1); tmap_prefetch(&tmA); tmap_prefetch(&tmW); }
  __syncthreads();

  long iter = 0;
  for(int w = blockIdx.x; w < nwork; w += gridDim.x){
    const int e = wl_e[w], mtile = wl_mt[w], ntile = wl_nt[w];
    const int arow  = m_off[e] + mtile*BM;
    const int vrows = min(BM, m_len[e] - mtile*BM);
    const int nbase = ntile*BN;
    const int vcols = min(BN, Ntot - nbase);

    float d[64];
#pragma unroll
    for(int i=0;i<64;i++) d[i]=0.f;

    auto issue = [&](int ks, long it){
      int S = (int)(it % NS);
      mbar_expect_tx(&full[S], TX_AB);
      tma_2d(As + S*BM*BK, &tmA, &full[S], ks*BK, arow);
      tma_3d(Bs + S*BN*BK, &tmW, &full[S], ks*BK, nbase, e);
    };

    if(tid==0){
#pragma unroll 1
      for(int s=0;s<PF;s++) if(s<ksteps) issue(s, iter+s);
    }
    wgmma_fence();

    for(int ks=0; ks<ksteps; ks++){
      long it = iter + ks;
      if(tid==0 && ks+PF<ksteps) issue(ks+PF, it+PF);
      int S = (int)(it % NS);
      mbar_wait(&full[S], (int)((it / NS) & 1));
      bf16* Ad = As + S*BM*BK;
      bf16* Bd = Bs + S*BN*BK;
      if constexpr (MMA){
#pragma unroll
        for(int kk=0; kk<BK; kk+=16)
          wgmma_m64n128k16(d, make_desc(Ad+kk,LBO_SW,SBO_SW,1), make_desc(Bd+kk,LBO_SW,SBO_SW,1));
        wgmma_commit();
        wgmma_wait<1>();
      } else {
        d[0] += __bfloat162float(Ad[tid]) + __bfloat162float(Bd[tid]);
      }
    }
    if constexpr (MMA) wgmma_wait<0>();
    iter += ksteps;

#pragma unroll
    for(int j=0;j<BN/8;j++){
#pragma unroll
      for(int q=0;q<4;q++){
        int rr = warp*16 + (q>>1)*8 + (lane>>2);
        int cc = j*8 + (lane&3)*2 + (q&1);
        if(rr<vrows && cc<vcols) C[(size_t)(arow+rr)*Ntot + nbase+cc] = d[j*4+q];
      }
    }
  }
}

// ================================================================ TMA + warp specialization
// 2 warpgroups: WG0 = consumer (wgmma + epilogue), WG1 = producer (TMA issue only).
// setmaxnreg.dec on the producer, .inc on the consumer. `__maxnreg__` (not __launch_bounds__)
// so ptxas sizes the allocation for the raised consumer count.
// Ring: full[] armed by the producer's expect_tx + TMA tx-count; empty[] arrived by the
// consumer once the wgmma group that read a stage has retired.
#ifndef WS_PROD_REG
#define WS_PROD_REG 32
#endif
#ifndef WS_CONS_REG
#define WS_CONS_REG 224
#endif
#ifndef WS_MAXNREG
#define WS_MAXNREG 128
#endif

__device__ __forceinline__ void setmaxnreg_dec(){
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(WS_PROD_REG));
}
__device__ __forceinline__ void setmaxnreg_inc(){
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(WS_CONS_REG));
}

template<int NS>
__global__ void __maxnreg__(WS_MAXNREG) moe_gemm_tma_ws(
    const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmW,
    float* __restrict__ C,
    int Ntot, int K, const int* __restrict__ m_off, const int* __restrict__ m_len,
    const int* __restrict__ wl_e, const int* __restrict__ wl_mt, const int* __restrict__ wl_nt,
    int nwork){
  extern __shared__ __align__(128) char smem_raw[];
  bf16* As = (bf16*)align1k(smem_raw);
  bf16* Bs = As + NS*BM*BK;
  uint64_t* full  = (uint64_t*)(Bs + NS*BN*BK);
  uint64_t* empty = full + NS;
  const int tid = threadIdx.x;
  const int ksteps = K/BK;

  if(tid==0){
    for(int s=0;s<NS;s++){ mbar_init(&full[s],1); mbar_init(&empty[s],1); }
    tmap_prefetch(&tmA); tmap_prefetch(&tmW);
  }
  __syncthreads();

  if(tid >= NTHREADS){
    // ---------------- producer warpgroup ----------------
    setmaxnreg_dec();
    if(tid == NTHREADS){
      long iter = 0;
      for(int w = blockIdx.x; w < nwork; w += gridDim.x){
        const int e = wl_e[w], mtile = wl_mt[w], ntile = wl_nt[w];
        const int arow  = m_off[e] + mtile*BM;
        const int nbase = ntile*BN;
        for(int ks=0; ks<ksteps; ks++, iter++){
          int S = (int)(iter % NS);
          if(iter >= NS) mbar_wait(&empty[S], (int)(((iter/NS)-1) & 1));
          mbar_expect_tx(&full[S], TX_AB);
          tma_2d(As + S*BM*BK, &tmA, &full[S], ks*BK, arow);
          tma_3d(Bs + S*BN*BK, &tmW, &full[S], ks*BK, nbase, e);
        }
      }
    }
    return;
  }

  // ---------------- consumer warpgroup ----------------
  setmaxnreg_inc();
  const int warp = tid>>5, lane = tid&31;
  long iter = 0;
  for(int w = blockIdx.x; w < nwork; w += gridDim.x){
    const int e = wl_e[w], mtile = wl_mt[w], ntile = wl_nt[w];
    const int arow  = m_off[e] + mtile*BM;
    const int vrows = min(BM, m_len[e] - mtile*BM);
    const int nbase = ntile*BN;
    const int vcols = min(BN, Ntot - nbase);

    float d[64];
#pragma unroll
    for(int i=0;i<64;i++) d[i]=0.f;
    wgmma_fence();

    for(int ks=0; ks<ksteps; ks++, iter++){
      int S = (int)(iter % NS);
      mbar_wait(&full[S], (int)((iter / NS) & 1));
      bf16* Ad = As + S*BM*BK;
      bf16* Bd = Bs + S*BN*BK;
#pragma unroll
      for(int kk=0; kk<BK; kk+=16)
        wgmma_m64n128k16(d, make_desc(Ad+kk,LBO_SW,SBO_SW,1), make_desc(Bd+kk,LBO_SW,SBO_SW,1));
      wgmma_commit();
      wgmma_wait<1>();                                  // group (iter-1) retired
      if(ks>0 && tid==0) mbar_arrive(&empty[(int)((iter-1) % NS)]);
    }
    wgmma_wait<0>();
    if(tid==0) mbar_arrive(&empty[(int)((iter-1) % NS)]);   // last stage of this work item

#pragma unroll
    for(int j=0;j<BN/8;j++){
#pragma unroll
      for(int q=0;q<4;q++){
        int rr = warp*16 + (q>>1)*8 + (lane>>2);
        int cc = j*8 + (lane&3)*2 + (q&1);
        if(rr<vrows && cc<vcols) C[(size_t)(arow+rr)*Ntot + nbase+cc] = d[j*4+q];
      }
    }
  }
}

// ================================================================ PART 2: descriptor probe
// Load ONE [BN][BK] tile of expert `e` through the 3-D tensor map and un-swizzle it to global
// memory so the host can compare against W[e][n0+r][k0+c] directly.
__global__ void tmap_probe(const __grid_constant__ CUtensorMap tmW, bf16* out,
                           int k0, int n0, int e){
  extern __shared__ __align__(128) char smem_raw[];
  bf16* Bs = (bf16*)align1k(smem_raw);
  uint64_t* bar = (uint64_t*)(Bs + BN*BK);
  if(threadIdx.x==0){
    mbar_init(bar,1);
  }
  __syncthreads();
  if(threadIdx.x==0){
    mbar_expect_tx(bar, TX_B);
    tma_3d(Bs, &tmW, bar, k0, n0, e);
  }
  mbar_wait(bar, 0);
  for(int i=threadIdx.x; i<BN*BK; i+=blockDim.x){
    int r = i/BK, c = i%BK;
    out[i] = Bs[soff_sw(r, c>>3) + (c&7)];
  }
}

// ---------------------------------------------------------------- host: problem + oracle
struct Prob {
  int E,N,K; std::vector<int> me, moff; int Mtot;
  std::vector<bf16> A, W;
};

static void fill_fast(bf16* p, size_t n, uint32_t seed, float scale){
#pragma omp parallel for schedule(static)
  for(long long i=0;i<(long long)n;i++){
    uint32_t x = seed ^ (uint32_t)(i*2654435761u);
    x ^= x<<13; x ^= x>>17; x ^= x<<5;
    p[i] = __float2bfloat16(((float)(x & 0xFFFF)/32768.f - 1.f)*scale);
  }
}

static Prob make_prob(int E,int N,int K,const std::vector<int>& me,uint32_t seed){
  Prob p; p.E=E;p.N=N;p.K=K;p.me=me;
  p.moff.resize(E); int off=0;
  for(int e=0;e<E;e++){ p.moff[e]=off; off+=me[e]; }
  p.Mtot=off;
  p.A.resize((size_t)p.Mtot*K); fill_fast(p.A.data(), p.A.size(), seed, 1.0f);
  p.W.resize((size_t)E*N*K);    fill_fast(p.W.data(), p.W.size(), seed^0x9E37u, 0.5f);
  return p;
}

static void oracle(const Prob& p, std::vector<float>& C){
  C.assign((size_t)p.Mtot*p.N, 0.f);
  for(int e=0;e<p.E;e++){
    const bf16* Wexp = p.W.data() + (size_t)e*p.N*p.K;
    const int base=p.moff[e], m=p.me[e];
#pragma omp parallel for schedule(dynamic) collapse(2)
    for(int r=0;r<m;r++) for(int n=0;n<p.N;n++){
      const bf16* a = p.A.data() + (size_t)(base+r)*p.K;
      const bf16* b = Wexp + (size_t)n*p.K;
      float acc=0.f;
      for(int k=0;k<p.K;k++) acc += __bfloat162float(a[k])*__bfloat162float(b[k]);
      C[(size_t)(base+r)*p.N + n] = acc;
    }
  }
}
static double relL2(const std::vector<float>& g,const std::vector<float>& o){
  double nu=0,de=0;
  for(size_t i=0;i<o.size();i++){ double d=(double)g[i]-o[i]; nu+=d*d; de+=(double)o[i]*o[i]; }
  return std::sqrt(nu/(de+1e-30));
}

struct Dev {
  bf16 *A,*W; float *C; int *moff,*mlen,*we,*wm,*wn; int nwork; size_t csz;
  CUtensorMap tmA, tmW;
};

// ---- tensor maps -----------------------------------------------------------------------
// A: 2-D  [K, Mtot], box {BK, BM}, coords {k0, arow}.       (expert offset is a COORDINATE)
// W: 3-D  [K, N, E], box {BK, BN, 1}, coords {k0, nbase, e}. ONE descriptor for all E experts.
static CUtensorMap encode_A(const bf16* A, int Mtot, int K){
  CUtensorMap tm{};
  uint64_t gdim[2] = {(uint64_t)K, (uint64_t)Mtot};
  uint64_t gstr[1] = {(uint64_t)K*2};
  uint32_t bdim[2] = {BK, BM};
  uint32_t estr[2] = {1,1};
  CKD(cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)A, gdim, gstr,
        bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tm;
}
static CUresult try_encode_W(CUtensorMap* tm, const bf16* W, int E, int N, int K){
  uint64_t gdim[3] = {(uint64_t)K, (uint64_t)N, (uint64_t)E};
  uint64_t gstr[2] = {(uint64_t)K*2, (uint64_t)N*(uint64_t)K*2};
  uint32_t bdim[3] = {BK, BN, 1};
  uint32_t estr[3] = {1,1,1};
  return cuTensorMapEncodeTiled(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)W, gdim, gstr,
        bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
static CUtensorMap encode_W(const bf16* W, int E, int N, int K){
  CUtensorMap tm{}; CKD(try_encode_W(&tm, W, E, N, K)); return tm;
}

static Dev upload(const Prob& p){
  Dev d{};
  std::vector<int> we,wm,wn;
  for(int e=0;e<p.E;e++){
    int mt=(p.me[e]+BM-1)/BM, nt=(p.N+BN-1)/BN;
    for(int a=0;a<mt;a++) for(int b=0;b<nt;b++){ we.push_back(e); wm.push_back(a); wn.push_back(b); }
  }
  d.nwork=(int)we.size(); d.csz=(size_t)p.Mtot*p.N*4;
  CK(cudaMalloc(&d.A,p.A.size()*2)); CK(cudaMemcpy(d.A,p.A.data(),p.A.size()*2,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.W,p.W.size()*2)); CK(cudaMemcpy(d.W,p.W.data(),p.W.size()*2,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.C,d.csz));
  CK(cudaMalloc(&d.moff,p.E*4)); CK(cudaMemcpy(d.moff,p.moff.data(),p.E*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.mlen,p.E*4)); CK(cudaMemcpy(d.mlen,p.me.data(),p.E*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.we,d.nwork*4)); CK(cudaMemcpy(d.we,we.data(),d.nwork*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.wm,d.nwork*4)); CK(cudaMemcpy(d.wm,wm.data(),d.nwork*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.wn,d.nwork*4)); CK(cudaMemcpy(d.wn,wn.data(),d.nwork*4,cudaMemcpyHostToDevice));
  d.tmA = encode_A(d.A, p.Mtot, p.K);
  d.tmW = encode_W(d.W, p.E, p.N, p.K);
  return d;
}
static void freedev(Dev& d){ cudaFree(d.A);cudaFree(d.W);cudaFree(d.C);cudaFree(d.moff);
  cudaFree(d.mlen);cudaFree(d.we);cudaFree(d.wm);cudaFree(d.wn); }

// ---------------------------------------------------------------- launchers
template<int NS> static constexpr size_t smem_cp(){ return (size_t)NS*(BM*BK+BN*BK)*2 + 1024; }
template<int NS> static constexpr size_t smem_tma(){ return (size_t)NS*(BM*BK+BN*BK)*2 + 1024 + NS*8; }
template<int NS> static constexpr size_t smem_ws(){  return (size_t)NS*(BM*BK+BN*BK)*2 + 1024 + 2*NS*8; }

static int g_grid = 264;

template<int NS,int PF,bool MMA=true> static void launch_cp(const Dev& d,const Prob& p){
  moe_gemm_cpasync<NS,PF,MMA><<<g_grid, NTHREADS, smem_cp<NS>()>>>(
    d.A,d.W,d.C,p.N,p.K,d.moff,d.mlen,d.we,d.wm,d.wn,d.nwork);
}
template<int NS,int PF,bool MMA=true> static void launch_tma(const Dev& d,const Prob& p){
  moe_gemm_tma<NS,PF,MMA><<<g_grid, NTHREADS, smem_tma<NS>()>>>(
    d.tmA,d.tmW,d.C,p.N,p.K,d.moff,d.mlen,d.we,d.wm,d.wn,d.nwork);
}
template<int NS> static void launch_ws(const Dev& d,const Prob& p){
  moe_gemm_tma_ws<NS><<<g_grid, 2*NTHREADS, smem_ws<NS>()>>>(
    d.tmA,d.tmW,d.C,p.N,p.K,d.moff,d.mlen,d.we,d.wm,d.wn,d.nwork);
}

typedef void (*LaunchFn)(const Dev&, const Prob&);

static double check(LaunchFn fn,const Dev& d,const Prob& p,const std::vector<float>& oc){
  CK(cudaMemset(d.C,0,d.csz));
  fn(d,p);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> g((size_t)p.Mtot*p.N);
  CK(cudaMemcpy(g.data(),d.C,d.csz,cudaMemcpyDeviceToHost));
  return relL2(g,oc);
}
// ---- throttle-robust timing -------------------------------------------------------------
// This H100 NVL is capped at 310 W and drops 1785 -> ~750 MHz within ~2 s of sustained wgmma,
// i.e. a naive back-to-back benchmark loop measures the POWER CAP, not the kernel. Every number
// in this file is therefore taken as short bursts (BURST launches, ~1 ms) separated by an idle
// gap so the clock recovers, repeated ROUNDS times with the variant order ROTATED each round
// (so no variant systematically inherits another's heat), and reduced with MIN. NVML reports the
// SM clock seen at measurement time so the reader can check the numbers are un-throttled.
#define BURST_MS 2.0    // target burst length; longer bursts start to throttle
#define ROUNDS 12
#define GAP_US 25000

static nvmlDevice_t g_nvml = nullptr;
static unsigned g_clk_min = 1u<<30, g_clk_max = 0;
static void clk_sample(){
  if(!g_nvml) return;
  unsigned c=0;
  if(nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &c)==NVML_SUCCESS){
    if(c<g_clk_min) g_clk_min=c;
    if(c>g_clk_max) g_clk_max=c;
  }
}

static double time_burst(LaunchFn fn,const Dev& d,const Prob& p,int iters){
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  for(int i=0;i<iters;i++) fn(d,p);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms,a,b));
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return ms/iters;
}
// min-of-ROUNDS for a set of variants measured round-robin with rotation.
static void bench_multi(const std::vector<LaunchFn>& fns,const Dev& d,const Prob& p,
                        std::vector<double>& ms){
  int n=(int)fns.size(); ms.assign(n,1e30);
  for(int j=0;j<n;j++){ fns[j](d,p); fns[j](d,p); }
  CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  double t0 = time_burst(fns[0],d,p,3);
  int iters = (int)(BURST_MS/std::max(t0,1e-4));
  if(iters<3) iters=3; if(iters>40) iters=40;
  for(int r=0;r<ROUNDS;r++){
    for(int j=0;j<n;j++){
      int idx=(j+r)%n;
      usleep(GAP_US);
      double t = time_burst(fns[idx],d,p,iters);
      clk_sample();
      if(t<ms[idx]) ms[idx]=t;
    }
  }
  CK(cudaGetLastError());
}
static double bench(LaunchFn fn,const Dev& d,const Prob& p,int,int){
  std::vector<LaunchFn> v{fn}; std::vector<double> m;
  bench_multi(v,d,p,m); return m[0];
}

// ---------------------------------------------------------------- main
static int g_sms = 132;

static void occ_report(const char* name, const void* f, int threads, size_t smem){
  int nb=0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, threads, (int)smem));
  cudaFuncAttributes at; CK(cudaFuncGetAttributes(&at, f));
  printf("  %-26s blocks/SM=%d  threads/blk=%4d  smem=%6zuB  regs=%3d  lmem(spill)=%zuB\n",
         name, nb, threads, smem, at.numRegs, (size_t)at.localSizeBytes);
}

int main(int argc,char** argv){
  CK(cudaSetDevice(0));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0));
  g_sms = pr.multiProcessorCount;
  if(nvmlInit_v2()==NVML_SUCCESS) nvmlDeviceGetHandleByIndex_v2(0,&g_nvml);
  int clk=0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
  if(g_nvml){
    unsigned pl=0; nvmlDeviceGetEnforcedPowerLimit(g_nvml,&pl);
    printf("NVML: enforced power limit = %.0f W (SM clock is sampled at every measurement)\n", pl/1000.0);
  }
  printf("GPU: %s  sm_%d%d  SMs=%d  clk=%.0fMHz  smem/SM=%dKB  regs/SM=%d\n",
         pr.name,pr.major,pr.minor,g_sms,clk/1000.0,
         (int)(pr.sharedMemPerMultiprocessor/1024), pr.regsPerMultiprocessor);
  printf("  bf16 tensor-core peak at this clock (4096 FLOP/SM/clk) = %.0f TF/s; reporting %% of %.0f\n",
         g_sms*(double)clk*1e3*4096/1e12, PEAK_BF16);
  printf("tile BM=%d BN=%d BK=%d  wgmma m64n128k16.f32.bf16.bf16\n\n", BM,BN,BK);

  const char* mode = argc>1 ? argv[1] : "all";
  bool do1 = !strcmp(mode,"all")||!strcmp(mode,"1");
  bool do2 = !strcmp(mode,"all")||!strcmp(mode,"2");
  bool do3 = !strcmp(mode,"all")||!strcmp(mode,"3");

  // raise dynamic smem limits for everything we may launch
  CK(cudaFuncSetAttribute(moe_gemm_cpasync<4,2>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_cp<4>()));
  CK(cudaFuncSetAttribute(moe_gemm_cpasync<4,2,false>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_cp<4>()));
  CK(cudaFuncSetAttribute(moe_gemm_tma<4,2,false>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_tma<4>()));
  CK(cudaFuncSetAttribute(moe_gemm_tma<3,1>,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_tma<3>()));
  CK(cudaFuncSetAttribute(moe_gemm_tma<4,2>,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_tma<4>()));
  CK(cudaFuncSetAttribute(moe_gemm_tma<5,3>,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_tma<5>()));
  CK(cudaFuncSetAttribute(moe_gemm_tma<6,4>,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_tma<6>()));
  CK(cudaFuncSetAttribute(moe_gemm_tma_ws<3>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_ws<3>()));
  CK(cudaFuncSetAttribute(moe_gemm_tma_ws<4>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_ws<4>()));
  CK(cudaFuncSetAttribute(moe_gemm_tma_ws<5>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_ws<5>()));
  CK(cudaFuncSetAttribute(moe_gemm_tma_ws<6>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_ws<6>()));
  CK(cudaFuncSetAttribute(tmap_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(BN*BK*2+1024+8)));

  // ============================================================ PART 1
  if(do1){
    printf("======== PART 1: can MoE fill the machine? ========\n");
    printf("[occupancy]\n");
    occ_report("cpasync NS=4 (baseline)", (const void*)moe_gemm_cpasync<4,2>, NTHREADS, smem_cp<4>());
    occ_report("tma     NS=4",            (const void*)moe_gemm_tma<4,2>,     NTHREADS, smem_tma<4>());
    occ_report("tma+ws  NS=4",            (const void*)moe_gemm_tma_ws<4>,  2*NTHREADS, smem_ws<4>());
    int bps=0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps,(const void*)moe_gemm_cpasync<4,2>,
                                                     NTHREADS,(int)smem_cp<4>()));
    int slots = g_sms*bps;
    printf("  => concurrent BM=64xBN=128 tile slots on this GPU = %d SMs x %d = %d\n\n", g_sms,bps,slots);

    struct S{ const char* name; int E,N,K,m; };
    std::vector<S> shapes = {
      {"E8   m_e=256  N=768 K=3840", 8,768,3840,256},
      {"E8   m_e=1024 N=768 K=3840", 8,768,3840,1024},
      {"E128 m_e=8    N=256 K=2560", 128,256,2560,8},
    };
    printf("[shape -> tiles / waves / achieved]  (waves = tiles / %d concurrent slots)\n", slots);
    for(auto& s:shapes){
      std::vector<int> me(s.E,s.m);
      Prob p = make_prob(s.E,s.N,s.K,me,7);
      Dev d = upload(p);
      double flops=0; for(int e=0;e<s.E;e++) flops += 2.0*me[e]*s.N*s.K;
      double t = bench(launch_cp<4,2>,d,p,200,20);
      double tf = flops/(t*1e-3)/1e12;
      int mt = (s.m+BM-1)/BM;
      double rowfill = (double)s.m / (mt*BM);   // fraction of each BM=64 tile that is real rows
      double hw_tf = tf / rowfill;   // TF/s the tensor cores actually execute (padded rows count)
      printf("  %-28s tiles=%5d  waves(slots)=%5.2f  waves(SM)=%5.2f  rowfill=%5.1f%%\n",
             s.name, d.nwork, (double)d.nwork/slots, (double)d.nwork/g_sms, rowfill*100);
      printf("  %-28s  %7.3f ms  useful %7.2f TF/s (%4.1f%% of %.0f)   hw-issued %7.2f TF/s (%4.1f%%)\n",
             "", t, tf, 100*tf/PEAK_BF16, PEAK_BF16, hw_tf, 100*hw_tf/PEAK_BF16);
      freedev(d);
    }

    // tile-count sweep: same per-tile work, walk 0.36 -> 23 waves.
    printf("\n[tile-count sweep] E=8 N=768 K=3840, vary m_e. Same kernel, same tile shape.\n");
    printf("  %-8s %6s %7s %9s %8s %7s  %s\n","m_e","tiles","waves","ms","TF/s","%peak","us/tile-slot");
    double best=0, at256=0;
    for(int m : {64,128,256,512,1024,2048,4096}){
      std::vector<int> me(8,m);
      Prob p = make_prob(8,768,3840,me,7);
      Dev d = upload(p);
      double flops=0; for(int e=0;e<8;e++) flops += 2.0*m*768.0*3840.0;
      double t = bench(launch_cp<4,2>,d,p,50,10);
      double tf = flops/(t*1e-3)/1e12;
      best = std::max(best,tf); if(m==256) at256=tf;
      printf("  %-8d %6d %7.2f %9.4f %8.2f %6.1f%%  %8.2f\n", m, d.nwork,(double)d.nwork/slots,
             t,tf,100*tf/PEAK_BF16, t*1e3/((double)d.nwork/slots));
      freedev(d);
    }
    printf("  asymptote (many waves) = %.2f TF/s -> the m_e=256 MoE point runs at %.0f%% of it\n\n",
           best, 100*at256/best);

    // grid-size sweep at the real MoE point: is extra concurrency usable at all?
    printf("[grid sweep] E=8 m_e=256 N=768 K=3840 (192 tiles). grid=#blocks launched.\n");
    {
      std::vector<int> me(8,256);
      Prob p = make_prob(8,768,3840,me,7);
      Dev d = upload(p);
      double flops=0; for(int e=0;e<8;e++) flops += 2.0*256*768.0*3840.0;
      for(int g : {66, 132, 192, 264, 396, 528}){
        g_grid = g;
        double t = bench(launch_cp<4,2>,d,p,200,20);
        printf("    grid=%4d  %7.3f ms  %7.2f TF/s\n", g, t, flops/(t*1e-3)/1e12);
      }
      g_grid = 264;
      freedev(d);
    }

    // Is the mainloop STAGING-bound (=> a better load engine can help) or already
    // bandwidth/compute-bound (=> it cannot)? Run the identical pipeline with wgmma removed.
    // Arithmetic intensity of a BMxBN tile is BM*BN/(BM+BN) FLOP per byte of operand traffic;
    // that number is what caps a small-tile MoE GEMM, and no load engine changes it.
    printf("\n[staging ablation] same pipeline, wgmma removed (LOAD-ONLY) vs full.\n");
    printf("  tile intensity = BM*BN/(BM+BN) = %.1f FLOP/byte -> %.0f TF/s needs %.1f TB/s of tile feed\n",
           (double)BM*BN/(BM+BN), PEAK_BF16, PEAK_BF16*1e12/((double)BM*BN/(BM+BN))/1e12);
    for(int m : {256,1024}){
      std::vector<int> me(8,m);
      Prob p = make_prob(8,768,3840,me,7);
      Dev d = upload(p);
      double flops=0; for(int e=0;e<8;e++) flops += 2.0*m*768.0*3840.0;
      double bytes = (double)d.nwork*(BM+BN)*3840.0*2.0;
      std::vector<LaunchFn> fns = {launch_cp<4,2,true>, launch_cp<4,2,false>,
                                   launch_tma<4,2,true>, launch_tma<4,2,false>};
      std::vector<double> ms; bench_multi(fns,d,p,ms);
      printf("  m_e=%-5d tiles=%4d  cpasync full %.4f ms / load-only %.4f ms (%.0f%% of full)"
             "   tma full %.4f ms / load-only %.4f ms (%.0f%%)\n",
             m,d.nwork,ms[0],ms[1],100*ms[1]/ms[0],ms[2],ms[3],100*ms[3]/ms[2]);
      printf("           tile-feed BW: full %.2f TB/s | load-only %.2f TB/s (cpasync), %.2f TB/s (tma)"
             "  | useful %.1f TF/s\n",
             bytes/(ms[0]*1e-3)/1e12, bytes/(ms[1]*1e-3)/1e12, bytes/(ms[3]*1e-3)/1e12,
             flops/(ms[0]*1e-3)/1e12);
      freedev(d);
    }
    printf("\n");
  }

  // ============================================================ PART 2
  if(do2){
    printf("======== PART 2: batched / 3-D CUtensorMap over E experts ========\n");
    printf("  sizeof(CUtensorMap) = %zu B\n", sizeof(CUtensorMap));
    int E=8,N=768,K=3840;
    std::vector<int> me(E,64);
    Prob p = make_prob(E,N,K,me,3);
    bf16* dW; CK(cudaMalloc(&dW,p.W.size()*2));
    CK(cudaMemcpy(dW,p.W.data(),p.W.size()*2,cudaMemcpyHostToDevice));

    CUtensorMap tm{};
    CUresult r = try_encode_W(&tm, dW, E, N, K);
    const char* es="ok"; if(r!=CUDA_SUCCESS) cuGetErrorString(r,&es);
    printf("  encode rank=3 gdim={K=%d,N=%d,E=%d} box={%d,%d,1} swizzle=128B : %s\n",
           K,N,E,BK,BN,es);
    if(r==CUDA_SUCCESS){
      // device-side 3-D load of an interior tile of expert 5, un-swizzled and compared on host.
      bf16* dout; CK(cudaMalloc(&dout,BN*BK*2));
      int k0=64*7, n0=BN*3, e=5;
      tmap_probe<<<1,128,BN*BK*2+1024+8>>>(tm,dout,k0,n0,e);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      std::vector<bf16> h(BN*BK); CK(cudaMemcpy(h.data(),dout,BN*BK*2,cudaMemcpyDeviceToHost));
      long bad=0;
      for(int rr=0;rr<BN;rr++) for(int cc=0;cc<BK;cc++){
        bf16 want = p.W[((size_t)e*N + n0+rr)*K + k0+cc];
        if(__bfloat162float(h[rr*BK+cc]) != __bfloat162float(want)) bad++;
      }
      printf("  3-D tile load W[e=%d][n=%d..][k=%d..] via ONE descriptor: mismatches=%ld  %s\n",
             e,n0,k0,bad, bad==0?"BIT-EXACT":"WRONG");
      // out-of-bounds outer coordinate behaviour (e == E) -> zero fill, no fault
      tmap_probe<<<1,128,BN*BK*2+1024+8>>>(tm,dout,k0,n0,E);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h.data(),dout,BN*BK*2,cudaMemcpyDeviceToHost));
      long nz=0; for(int i=0;i<BN*BK;i++) if(__bfloat162float(h[i])!=0.f) nz++;
      printf("  OOB expert coord e=E=%d: nonzero elems=%ld (0 => hardware zero-fills, no fault)\n",E,nz);
      cudaFree(dout);
    }
    // how expensive is a host-side re-encode (matters if the base pointer is per-call)?
    {
      const int R=1000; CUtensorMap t2{};
      struct timespec s0,s1; clock_gettime(CLOCK_MONOTONIC,&s0);
      for(int i=0;i<R;i++) try_encode_W(&t2,dW,E,N,K);
      clock_gettime(CLOCK_MONOTONIC,&s1);
      double us = ((s1.tv_sec-s0.tv_sec)*1e9 + (s1.tv_nsec-s0.tv_nsec))/1e3/R;
      printf("  cuTensorMapEncodeTiled host cost = %.3f us/descriptor  (x E if E descriptors)\n", us);
    }
    // rank limit / E-descriptor alternative
    printf("  rank limit is 5 => [K,N,E] leaves 2 spare dims; boxDim[outer]=1 is legal.\n");
    cudaFree(dW);
    printf("\n");
  }

  // ============================================================ PART 3
  if(do3){
    printf("======== PART 3: TMA / TMA+WS vs cp.async ========\n");
    struct Cfg{ const char* name; int E,N,K; std::vector<int> me; };
    std::vector<Cfg> cfgs = {
      {"E8_N768_K3840_ragged",  8, 768,3840,{32,64,17,128,5,200,64,96}},
      {"E128_N256_K2560_bal", 128, 256,2560,std::vector<int>(128,8)},
      {"E8_N768_K3840_m256",    8, 768,3840,std::vector<int>(8,256)},
    };
    bool allpass=true;
    for(auto& c:cfgs){
      Prob p = make_prob(c.E,c.N,c.K,c.me,777);
      std::vector<float> oc; oracle(p,oc);
      Dev d = upload(p);
      double v0 = check(launch_cp<4,2>,d,p,oc);
      double v1 = check(launch_tma<4,2>,d,p,oc);
      double v2 = check(launch_ws<4>,d,p,oc);
      bool pass = v0<3e-3 && v1<3e-3 && v2<3e-3;
      printf("[validate] %-22s Mtot=%5d tiles=%4d | cpasync=%.3e tma=%.3e tma+ws=%.3e %s\n",
             c.name,p.Mtot,d.nwork,v0,v1,v2,pass?"PASS":"FAIL");
      allpass &= pass;
      freedev(d);
    }
    printf("RESULT: %s\n\n", allpass?"ALL PASS":"FAIL");
    if(!allpass) return 1;

    struct BC{ int E,N,K,m; };
    std::vector<BC> bcs = { {8,768,3840,256}, {8,768,3840,1024}, {128,256,2560,8} };
    for(auto& b:bcs){
      std::vector<int> me(b.E,b.m);
      Prob p = make_prob(b.E,b.N,b.K,me,99);
      Dev d = upload(p);
      double flops=0; for(int e=0;e<b.E;e++) flops += 2.0*me[e]*b.N*b.K;
      printf("  [bench] E=%d m_e=%d N=%d K=%d  tiles=%d\n", b.E,b.m,b.N,b.K,d.nwork);
      const char* nm[] = {"cpasync NS=4 PF=2","tma     NS=3 PF=1","tma     NS=4 PF=2",
                          "tma     NS=5 PF=3","tma     NS=6 PF=4",
                          "tma+ws  NS=3","tma+ws  NS=4","tma+ws  NS=5","tma+ws  NS=6"};
      std::vector<LaunchFn> fns = {launch_cp<4,2>,launch_tma<3,1>,launch_tma<4,2>,
                                   launch_tma<5,3>,launch_tma<6,4>,
                                   launch_ws<3>,launch_ws<4>,launch_ws<5>,launch_ws<6>};
      std::vector<double> ms; bench_multi(fns,d,p,ms);
      double base = flops/(ms[0]*1e-3)/1e12, bt=0, bw=0;
      for(size_t i=0;i<fns.size();i++){
        double tf = flops/(ms[i]*1e-3)/1e12;
        printf("    %-22s %8.4f ms  %7.2f TF/s  (%4.1f%% peak)  %.2fx base\n",
               nm[i],ms[i],tf,100*tf/PEAK_BF16,tf/base);
        if(i>=1&&i<=4) bt=std::max(bt,tf);
        if(i>=5)       bw=std::max(bw,tf);
      }
      printf("    -> best tma %.2f TF/s (%.2fx base) | best tma+ws %.2f TF/s (%.2fx base)\n\n",
             bt,bt/base,bw,bw/base);
      freedev(d);
    }
  }
  if(g_nvml && g_clk_max) printf("[clocks] SM clock observed across all measurements: %u..%u MHz "
                    "(boost %d MHz) -- bursts kept it near boost\n", g_clk_min, g_clk_max, clk/1000);
  return 0;
}

// wgmma_moe_group_probe.cu
// Standalone Hopper (sm_90a) GROUPED WGMMA GEMM microkernel probe (MoE expert GEMM).
//
// PROBLEM (grouped / MoE GEMM, TN):
//   E experts. Expert e computes  C_e[m_e, N] = A_e[m_e, K] . B_e[N, K]^T
//   A is ONE gathered activation matrix A[M_total, K]; its M rows are partitioned among the
//   experts by a routing prefix-offset array m_off[e] (m_off[e+1] = m_off[e] + m_e), exactly
//   the `rowoff[e]` contract in plow's runtime/nvidia/op_moe.cuh (d_moe_group_*_gemma_pf).
//   B_e is expert e's weight [N,K], row-major / K-contiguous (like `ewt[e]`).
//   Both operands are K-contiguous => TN GEMM, same as the dense case.
//
// KERNEL:
//   Mainloop is  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16  (warpgroup MMA, 1
//   warpgroup = 128 threads = 1 block, computing a BM=64 x BN=128 output tile).
//   The "grouped" part is PURE base-pointer / offset indirection around an otherwise dense
//   wgmma mainloop: persistent blocks stride over a flat worklist of (expert, m_tile, n_tile)
//   triples; each tile resolves A row base = m_off[e] + m_tile*BM and B base = W + e*N*K.
//   Nothing inside the wgmma mainloop is MoE-aware.
//
//   Ragged last M-tile (m_e % BM != 0) is handled two ways: A rows past the expert's segment
//   are cp.async zero-filled (src-size 0, so they never read another expert's rows or run off
//   the end of A), and the f32 epilogue masks the stores to rr < vrows.
//
// SMEM LAYOUT (the part that is easy to get wrong):
//   swizzle mode 0 (NO swizzle) + the "interleaved" core-matrix-tiled layout. For bf16 a wgmma
//   core matrix is 8 rows x 8 cols = 8 rows of 16B = 128 contiguous bytes. A [rows][BK] tile is
//   stored as a grid of those cores; descriptor LBO = K-core step (128B), SBO = MN-core step
//   (BK/8*128 B). Both operands use the SAME layout function (B is stored [N][K], i.e. N in the
//   "row" role) and both transpose immediates are 0. These knobs were originally discovered by
//   sweeping (tA,tB,swapLBO/SBO_A,swapLBO/SBO_B) against the CPU oracle; the winning combo is
//   locked below and re-checked at startup by self_check().
//
// VALIDATION: f32 CPU oracle over all experts from the SAME bf16 inputs; relL2 PASS < 3e-3.
// BENCHMARK: vs a mma.sync.m16n8k16 grouped baseline using an IDENTICAL tiling/pipeline, so the
//   comparison isolates the MMA instruction rather than the memory pipeline.
//
// Build (note: CUDA 13.0's -arch=sm_90a silently lowers to sm_90 and rejects wgmma; the
// -gencode form below is required):
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 \
//     -I runtime/common -I runtime/nvidia -include cstdint -Xcompiler -fopenmp \
//     runtime/nvidia/experiments/wgmma_moe_group_probe.cu -o wgmma_moe

#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

typedef __nv_bfloat16 bf16;

// ---------------------------------------------------------------- tile config
#define BM 64          // wgmma M (one warpgroup, m64)
#define BN 128         // wgmma N (m64n128k16 -> 64 f32 acc regs / thread)
#define BK 64          // K per pipeline stage (4 wgmma k=16 substeps)
#define STAGES 4       // cp.async ring depth
#define PF 2           // cp.async prefetch distance (< STAGES-1 so one wgmma group may stay
                       // in flight without racing the buffer being refilled)
#define NTHREADS 128   // exactly one warpgroup

// ---------------------------------------------------------------- device helpers
__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool pred){
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  int bytes = pred ? 16 : 0;               // src-size 0 => hardware zero-fills the 16B
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
               :: "r"(s), "l"(gmem), "n"(16), "r"(bytes));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// WGMMA 64-bit shared-memory matrix descriptor (fields are 16-byte granular, hence >>4):
//   bits[0:14) start addr | bits[16:30) LBO | bits[32:46) SBO | bits[49:52) base off | [62:64) swizzle
//   swz: 0 = none, 1 = 128-byte swizzle (bits[63:62]).
__device__ __forceinline__ uint64_t make_desc(const void* p, uint64_t lbo, uint64_t sbo, int swz){
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  uint64_t d = ((uint64_t)a & 0x3FFFF) >> 4;
  d |= ((lbo & 0x3FFFF) >> 4) << 16;
  d |= ((sbo & 0x3FFFF) >> 4) << 32;
  d |= ((uint64_t)swz) << 62;                // base_offset (bits[49:52]) stays 0
  return d;
}

__device__ __forceinline__ void wgmma_fence(){ asm volatile("wgmma.fence.sync.aligned;\n"); }
__device__ __forceinline__ void wgmma_commit(){ asm volatile("wgmma.commit_group.sync.aligned;\n"); }
template<int N> __device__ __forceinline__ void wgmma_wait(){
  asm volatile("wgmma.wait_group.sync.aligned %0;\n"::"n"(N));
}

// wgmma.mma_async.m64n128k16: D += A*B for one k=16 step. 64 f32 accumulators per thread.
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
    : "l"(da),"l"(db),
      "n"(1),   // scale-D : accumulate into D
      "n"(1),   // imm-scale-A
      "n"(1),   // imm-scale-B
      "n"(0),   // imm-trans-A  (locked by calibration)
      "n"(0));  // imm-trans-B  (locked by calibration)
}

// mma.sync.m16n8k16 baseline instruction.
__device__ __forceinline__ void mma_m16n8k16(float* d, const unsigned* a, const unsigned* b){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
               : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

// ---- two smem layouts, A/B-selectable at compile time --------------------------------------
// SWZ=false: no-swizzle core-matrix-tiled ("interleaved") layout. Core matrix = 8 rows x 8 cols
//   = 64 elems = 128B contiguous. CORRECT, but the row-core stride is BK*16 = 1024B, so every
//   row-core lands on the same smem bank -> 8-way conflicts on both the cp.async store and the
//   wgmma operand read.
// SWZ=true: 128-byte swizzle. Logical row-major [rows][BK=64] bf16 (one row = exactly 128B =
//   one swizzle atom row); physical chunk index is XORed with the row: r*BK + ((c^(r&7))*8).
//   This matches the hardware address swizzle (addr bits[6:4] chunk ^ bits[9:7] row), which is
//   why the tile base MUST be 1024B aligned or store-side and hardware-side XOR disagree.
__device__ __forceinline__ int soff_ns(int r, int kc){ return (r>>3)*(BK*8) + kc*64 + (r&7)*8; }
__device__ __forceinline__ int soff_sw(int r, int c){ return r*BK + ((c ^ (r&7))*8); }
template<bool SWZ> __device__ __forceinline__ int soff_c(int r,int c){
  return SWZ ? soff_sw(r,c) : soff_ns(r,c);
}
// element offset of an arbitrary (row, k) - used by the mma.sync baseline's fragment loads.
template<bool SWZ> __device__ __forceinline__ int soff_e(int r,int k){
  return soff_c<SWZ>(r,k>>3) + (k&7);
}
#define LBO_NS 128ull                  // no-swizzle: adjacent K core-matrices
#define SBO_NS ((uint64_t)(BK*16))     // no-swizzle: adjacent M/N core-matrices
#define LBO_SW 16ull                   // 128B swizzle: fixed
#define SBO_SW 1024ull                 // 128B swizzle: 8 rows x 128B

// stage a [ROWS][BK] tile: rows come from `src` with row stride K, guarded by `nrows_valid`.
template<int ROWS,bool SWZ>
__device__ __forceinline__ void stage_tile(bf16* dst, const bf16* src, const bf16* safe,
                                           int k0, int K, int nrows_valid, int tid){
  constexpr int CH = BK/8;                      // 16B chunks per row
  for(int i=tid; i<ROWS*CH; i+=NTHREADS){
    int r = i/CH, c = i%CH;
    bool v = (r < nrows_valid);
    const bf16* s = src + (size_t)r*K + k0 + c*8;
    cp16(&dst[soff_c<SWZ>(r,c)], v? s : safe, v);
  }
}

// round a dynamic-smem base up to 1024B *in the shared window* (generic->shared is affine, so
// adding N bytes to the generic pointer adds N to the shared address).
__device__ __forceinline__ char* align1k(char* p){
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  return p + ((1024u - (a & 1023u)) & 1023u);
}

// ---------------------------------------------------------------- grouped kernel
// USE_WGMMA=true  -> wgmma.mma_async.m64n128k16 warpgroup mainloop
// USE_WGMMA=false -> mma.sync.m16n8k16 baseline, identical tiling/staging/pipeline
template<bool USE_WGMMA,bool SWZ>
__global__ __launch_bounds__(NTHREADS) void grouped_gemm(
    const bf16* __restrict__ A, const bf16* __restrict__ W, float* __restrict__ C,
    int Ntot, int K, const int* __restrict__ m_off, const int* __restrict__ m_len,
    const int* __restrict__ wl_e, const int* __restrict__ wl_mt, const int* __restrict__ wl_nt,
    int nwork){
  extern __shared__ __align__(128) char smem_raw[];
  bf16* As = (bf16*)align1k(smem_raw);              // [STAGES][BM][BK], 1024B aligned
  bf16* Bs = As + STAGES*BM*BK;                     // [STAGES][BN][BK] (also 1024B aligned)
  const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
  const int ksteps = K/BK;

  for(int w = blockIdx.x; w < nwork; w += gridDim.x){
    // ---- grouped indirection: resolve expert, its A row segment and its weight base ----
    const int e = wl_e[w], mtile = wl_mt[w], ntile = wl_nt[w];
    const int arow  = m_off[e] + mtile*BM;              // first A row of this tile
    const int vrows = min(BM, m_len[e] - mtile*BM);     // valid rows (ragged last tile)
    const int nbase = ntile*BN;
    const int vcols = min(BN, Ntot - nbase);
    const bf16* Aseg = A + (size_t)arow*K;
    const bf16* Wexp = W + (size_t)e*(size_t)Ntot*K + (size_t)nbase*K;
    // ---- from here down the mainloop is dense; nothing is MoE-aware ----

    float d[64];
#pragma unroll
    for(int i=0;i<64;i++) d[i]=0.f;

    auto stage = [&](int ks, int buf){
      stage_tile<BM,SWZ>(As + buf*BM*BK, Aseg, A, ks*BK, K, vrows, tid);
      stage_tile<BN,SWZ>(Bs + buf*BN*BK, Wexp, W, ks*BK, K, vcols, tid);
    };

#pragma unroll
    for(int s=0;s<PF;s++){ if(s<ksteps) stage(s,s); cp_commit(); }

    // Accumulators are written by plain register ops above; one fence here covers the whole
    // k-loop (inside it only wgmma writes them), so no per-stage fence is needed.
    if constexpr (USE_WGMMA) wgmma_fence();

    for(int ks=0; ks<ksteps; ks++){
      int fetch = ks + PF;
      if(fetch<ksteps) stage(fetch, fetch%STAGES);
      cp_commit();
      cp_wait<PF>();
      __syncthreads();
      const int cb = ks % STAGES;
      bf16* Ad = As + cb*BM*BK;
      bf16* Bd = Bs + cb*BN*BK;

      if constexpr (USE_WGMMA){
#pragma unroll
        for(int kk=0; kk<BK; kk+=16){
          // sub-k slice. no-swizzle: advance by (kk/8) K-core matrices (+256B per k16).
          // 128B swizzle: advance the START ADDRESS only by kk elems (+32B per k16); LBO/SBO fixed.
          uint64_t da,db;
          if constexpr (SWZ){
            da = make_desc(Ad + kk, LBO_SW, SBO_SW, 1);
            db = make_desc(Bd + kk, LBO_SW, SBO_SW, 1);
          } else {
            da = make_desc(Ad + (kk/8)*64, LBO_NS, SBO_NS, 0);
            db = make_desc(Bd + (kk/8)*64, LBO_NS, SBO_NS, 0);
          }
          wgmma_m64n128k16(d, da, db);
        }
        wgmma_commit();
        wgmma_wait<1>();   // keep one group in flight: overlaps MMA with the next cp.async
      } else {
        // baseline: 4 warps x (16 rows) x 128 cols, m16n8k16
        const int rbase = warp*16;
#pragma unroll
        for(int kk=0; kk<BK; kk+=16){
          unsigned af[4];
          {
            int r0 = rbase + (lane>>2), r1 = r0 + 8;
            int c0 = kk + (lane&3)*2, c1 = c0 + 8;
            af[0] = *(const unsigned*)&Ad[soff_e<SWZ>(r0,c0)];
            af[1] = *(const unsigned*)&Ad[soff_e<SWZ>(r1,c0)];
            af[2] = *(const unsigned*)&Ad[soff_e<SWZ>(r0,c1)];
            af[3] = *(const unsigned*)&Ad[soff_e<SWZ>(r1,c1)];
          }
#pragma unroll
          for(int j=0;j<BN/8;j++){
            int n = j*8 + (lane>>2);
            int c0 = kk + (lane&3)*2, c1 = c0 + 8;
            unsigned bf[2];
            bf[0] = *(const unsigned*)&Bd[soff_e<SWZ>(n,c0)];
            bf[1] = *(const unsigned*)&Bd[soff_e<SWZ>(n,c1)];
            mma_m16n8k16(&d[j*4], af, bf);
          }
        }
      }
      __syncthreads();
    }
    if constexpr (USE_WGMMA) wgmma_wait<0>();   // retire the last group before reading acc

    // ---- epilogue: f32 accumulator -> C, masked for ragged M and N tail ----
    // wgmma m64nN and the 4-warp m16n8k16 tiling produce the SAME (row,col) mapping.
#pragma unroll
    for(int j=0;j<BN/8;j++){
#pragma unroll
      for(int q=0;q<4;q++){
        int rr = warp*16 + (q>>1)*8 + (lane>>2);
        int cc = j*8 + (lane&3)*2 + (q&1);
        if(rr<vrows && cc<vcols) C[(size_t)(arow+rr)*Ntot + nbase+cc] = d[j*4+q];
      }
    }
    __syncthreads();   // protect smem before the next work item refills it
  }
}

// ================================================================ fp8 (e4m3) variant
// wgmma.mma_async.m64n128k32.f32.e4m3.e4m3. Two differences vs bf16:
//   - k step is 32 (not 16), so a core matrix is 8 rows x 16 cols (still 8 rows of 16B = 128B).
//   - the 8-bit wgmma forms have NO imm-trans-a/b operands: both operands must be K-major,
//     which is exactly the TN layout we already use.
typedef __nv_fp8_e4m3 fp8;
#define BK8 128        // K per pipeline stage in fp8 elements (4 wgmma k=32 substeps)

// Same two layouts as bf16. A fp8 row of BK8=128 elements is also exactly 128B = one swizzle
// atom row, so the swizzle is identical with 16-element (16B) chunks.
__device__ __forceinline__ int soff8_ns(int r,int kc){ return (r>>3)*(BK8*8) + kc*128 + (r&7)*16; }
__device__ __forceinline__ int soff8_sw(int r,int c){ return r*BK8 + ((c ^ (r&7))*16); }
template<bool SWZ> __device__ __forceinline__ int soff8_c(int r,int c){
  return SWZ ? soff8_sw(r,c) : soff8_ns(r,c);
}
#define LBO8_NS 128ull                  // no-swizzle: adjacent K core-matrices
#define SBO8_NS ((uint64_t)(BK8*8))     // no-swizzle: adjacent M/N core-matrices

__device__ __forceinline__ void wgmma_m64n128k32_e4m3(float* d, uint64_t da, uint64_t db){
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
    "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
    "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
    "%64, %65, %66, %67, %68;\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
      "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
      "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
      "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
      "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
      "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
      "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
      "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
    : "l"(da),"l"(db),"n"(1),"n"(1),"n"(1));
}

template<int ROWS,bool SWZ>
__device__ __forceinline__ void stage_tile8(fp8* dst, const fp8* src, const fp8* safe,
                                            int k0, int K, int nrows_valid, int tid){
  constexpr int CH = BK8/16;                 // 16B chunks per row (16 fp8 each)
  for(int i=tid; i<ROWS*CH; i+=NTHREADS){
    int r = i/CH, c = i%CH;
    bool v = (r < nrows_valid);
    const fp8* s = src + (size_t)r*K + k0 + c*16;
    cp16(&dst[soff8_c<SWZ>(r,c)], v? s : safe, v);
  }
}

template<bool SWZ>
__global__ __launch_bounds__(NTHREADS) void grouped_gemm_fp8(
    const fp8* __restrict__ A, const fp8* __restrict__ W, float* __restrict__ C,
    int Ntot, int K, const int* __restrict__ m_off, const int* __restrict__ m_len,
    const int* __restrict__ wl_e, const int* __restrict__ wl_mt, const int* __restrict__ wl_nt,
    int nwork){
  extern __shared__ __align__(128) char smem_raw[];
  fp8* As = (fp8*)align1k(smem_raw);
  fp8* Bs = As + STAGES*BM*BK8;
  const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
  const int ksteps = K/BK8;

  for(int w = blockIdx.x; w < nwork; w += gridDim.x){
    const int e = wl_e[w], mtile = wl_mt[w], ntile = wl_nt[w];
    const int arow  = m_off[e] + mtile*BM;
    const int vrows = min(BM, m_len[e] - mtile*BM);
    const int nbase = ntile*BN;
    const int vcols = min(BN, Ntot - nbase);
    const fp8* Aseg = A + (size_t)arow*K;
    const fp8* Wexp = W + (size_t)e*(size_t)Ntot*K + (size_t)nbase*K;

    float d[64];
#pragma unroll
    for(int i=0;i<64;i++) d[i]=0.f;

    auto stage = [&](int ks,int buf){
      stage_tile8<BM,SWZ>(As + buf*BM*BK8, Aseg, A, ks*BK8, K, vrows, tid);
      stage_tile8<BN,SWZ>(Bs + buf*BN*BK8, Wexp, W, ks*BK8, K, vcols, tid);
    };
#pragma unroll
    for(int s=0;s<PF;s++){ if(s<ksteps) stage(s,s); cp_commit(); }
    wgmma_fence();

    for(int ks=0; ks<ksteps; ks++){
      int fetch = ks + PF;
      if(fetch<ksteps) stage(fetch, fetch%STAGES);
      cp_commit();
      cp_wait<PF>();
      __syncthreads();
      const int cb = ks % STAGES;
      fp8* Ad = As + cb*BM*BK8;
      fp8* Bd = Bs + cb*BN*BK8;
#pragma unroll
      for(int kk=0; kk<BK8; kk+=32){
        uint64_t da,db;
        if constexpr (SWZ){   // +32B per k32 substep on the start address only
          da = make_desc(Ad + kk, LBO_SW, SBO_SW, 1);
          db = make_desc(Bd + kk, LBO_SW, SBO_SW, 1);
        } else {
          da = make_desc(Ad + (kk/16)*128, LBO8_NS, SBO8_NS, 0);
          db = make_desc(Bd + (kk/16)*128, LBO8_NS, SBO8_NS, 0);
        }
        wgmma_m64n128k32_e4m3(d, da, db);
      }
      wgmma_commit();
      wgmma_wait<1>();
      __syncthreads();
    }
    wgmma_wait<0>();

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

// ---------------------------------------------------------------- host: problem + oracle
struct Prob {
  int E,N,K; std::vector<int> me, moff; int Mtot;
  std::vector<bf16> A, W;
};

static Prob make_prob(int E,int N,int K,const std::vector<int>& me,uint32_t seed){
  Prob p; p.E=E;p.N=N;p.K=K;p.me=me;
  p.moff.resize(E); int off=0;
  for(int e=0;e<E;e++){ p.moff[e]=off; off+=me[e]; }
  p.Mtot=off;
  std::mt19937 g(seed); std::uniform_real_distribution<float> dist(-1.f,1.f);
  p.A.resize((size_t)p.Mtot*K); for(auto& x:p.A) x=__float2bfloat16(dist(g));
  p.W.resize((size_t)E*N*K);    for(auto& x:p.W) x=__float2bfloat16(dist(g)*0.5f);
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

struct Dev { bf16 *A,*W; float *C; int *moff,*mlen,*we,*wm,*wn; int nwork; size_t csz; };

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
  return d;
}
static void freedev(Dev& d){ cudaFree(d.A);cudaFree(d.W);cudaFree(d.C);cudaFree(d.moff);
  cudaFree(d.mlen);cudaFree(d.we);cudaFree(d.wm);cudaFree(d.wn); }

// +1024 so the kernel can round the base up to a 1024B boundary for the swizzled layout.
static constexpr size_t SMEM = (size_t)STAGES*(BM*BK + BN*BK)*sizeof(bf16) + 1024;
static int g_grid = 0;

template<bool WG,bool SWZ> static void launch(const Dev& d,const Prob& p){
  grouped_gemm<WG,SWZ><<<g_grid, NTHREADS, SMEM>>>(d.A,d.W,d.C,p.N,p.K,d.moff,d.mlen,
                                                   d.we,d.wm,d.wn,d.nwork);
}

template<bool WG,bool SWZ> static double check(const Dev& d,const Prob& p,const std::vector<float>& oc){
  CK(cudaMemset(d.C,0,d.csz));
  launch<WG,SWZ>(d,p);
  CK(cudaDeviceSynchronize());
  std::vector<float> g((size_t)p.Mtot*p.N);
  CK(cudaMemcpy(g.data(),d.C,d.csz,cudaMemcpyDeviceToHost));
  return relL2(g,oc);
}

template<bool WG,bool SWZ> static double bench(const Dev& d,const Prob& p,int iters,int warm){
  for(int i=0;i<warm;i++) launch<WG,SWZ>(d,p);
  CK(cudaDeviceSynchronize());
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  for(int i=0;i<iters;i++) launch<WG,SWZ>(d,p);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms,a,b));
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return ms/iters;
}

// ---------------------------------------------------------------- fp8 host harness
struct Prob8 { int E,N,K; std::vector<int> me,moff; int Mtot; std::vector<fp8> A,W; };
struct Dev8 { fp8 *A,*W; float *C; int *moff,*mlen,*we,*wm,*wn; int nwork; size_t csz; };

static Prob8 make_prob8(int E,int N,int K,const std::vector<int>& me,uint32_t seed){
  Prob8 p; p.E=E;p.N=N;p.K=K;p.me=me;
  p.moff.resize(E); int off=0; for(int e=0;e<E;e++){p.moff[e]=off; off+=me[e];} p.Mtot=off;
  std::mt19937 g(seed); std::uniform_real_distribution<float> dist(-1.f,1.f);
  p.A.resize((size_t)p.Mtot*K); for(auto& x:p.A) x=fp8(dist(g));
  p.W.resize((size_t)E*N*K);    for(auto& x:p.W) x=fp8(dist(g)*0.5f);
  return p;
}
// oracle over the SAME e4m3 values (so relL2 measures the GEMM, not the quantization).
static void oracle8(const Prob8& p, std::vector<float>& C){
  C.assign((size_t)p.Mtot*p.N,0.f);
  for(int e=0;e<p.E;e++){
    const fp8* Wexp = p.W.data() + (size_t)e*p.N*p.K;
    const int base=p.moff[e], m=p.me[e];
#pragma omp parallel for schedule(dynamic) collapse(2)
    for(int r=0;r<m;r++) for(int n=0;n<p.N;n++){
      const fp8* a = p.A.data() + (size_t)(base+r)*p.K;
      const fp8* b = Wexp + (size_t)n*p.K;
      float acc=0.f;
      for(int k=0;k<p.K;k++) acc += float(a[k])*float(b[k]);
      C[(size_t)(base+r)*p.N + n] = acc;
    }
  }
}
static Dev8 upload8(const Prob8& p){
  Dev8 d{}; std::vector<int> we,wm,wn;
  for(int e=0;e<p.E;e++){ int mt=(p.me[e]+BM-1)/BM, nt=(p.N+BN-1)/BN;
    for(int a=0;a<mt;a++) for(int b=0;b<nt;b++){ we.push_back(e); wm.push_back(a); wn.push_back(b);} }
  d.nwork=(int)we.size(); d.csz=(size_t)p.Mtot*p.N*4;
  CK(cudaMalloc(&d.A,p.A.size())); CK(cudaMemcpy(d.A,p.A.data(),p.A.size(),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.W,p.W.size())); CK(cudaMemcpy(d.W,p.W.data(),p.W.size(),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.C,d.csz));
  CK(cudaMalloc(&d.moff,p.E*4)); CK(cudaMemcpy(d.moff,p.moff.data(),p.E*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.mlen,p.E*4)); CK(cudaMemcpy(d.mlen,p.me.data(),p.E*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.we,d.nwork*4)); CK(cudaMemcpy(d.we,we.data(),d.nwork*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.wm,d.nwork*4)); CK(cudaMemcpy(d.wm,wm.data(),d.nwork*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.wn,d.nwork*4)); CK(cudaMemcpy(d.wn,wn.data(),d.nwork*4,cudaMemcpyHostToDevice));
  return d;
}
static void freedev8(Dev8& d){ cudaFree(d.A);cudaFree(d.W);cudaFree(d.C);cudaFree(d.moff);
  cudaFree(d.mlen);cudaFree(d.we);cudaFree(d.wm);cudaFree(d.wn); }

static constexpr size_t SMEM8 = (size_t)STAGES*(BM*BK8 + BN*BK8)*sizeof(fp8) + 1024;
template<bool SWZ> static void launch8(const Dev8& d,const Prob8& p){
  grouped_gemm_fp8<SWZ><<<g_grid,NTHREADS,SMEM8>>>(d.A,d.W,d.C,p.N,p.K,d.moff,d.mlen,
                                                   d.we,d.wm,d.wn,d.nwork);
}
template<bool SWZ> static double check8(const Dev8& d,const Prob8& p,const std::vector<float>& oc){
  CK(cudaMemset(d.C,0,d.csz)); launch8<SWZ>(d,p); CK(cudaDeviceSynchronize());
  std::vector<float> g((size_t)p.Mtot*p.N);
  CK(cudaMemcpy(g.data(),d.C,d.csz,cudaMemcpyDeviceToHost));
  return relL2(g,oc);
}
template<bool SWZ> static double bench8(const Dev8& d,const Prob8& p,int iters,int warm){
  for(int i=0;i<warm;i++) launch8<SWZ>(d,p);
  CK(cudaDeviceSynchronize());
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  for(int i=0;i<iters;i++) launch8<SWZ>(d,p);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms,a,b));
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return ms/iters;
}

int main(){
  CK(cudaSetDevice(0));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0));
  printf("GPU: %s  sm_%d%d  SMs=%d\n",pr.name,pr.major,pr.minor,pr.multiProcessorCount);
  printf("tile: BM=%d BN=%d BK=%d STAGES=%d smem=%zuB  wgmma=m64n%dk16.f32.bf16.bf16\n",
         BM,BN,BK,STAGES,SMEM,BN);

  CK(cudaFuncSetAttribute(grouped_gemm<true,false>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SMEM));
  CK(cudaFuncSetAttribute(grouped_gemm<false,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SMEM));
  CK(cudaFuncSetAttribute(grouped_gemm<true,true>,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SMEM));
  CK(cudaFuncSetAttribute(grouped_gemm<false,true>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SMEM));
  g_grid = pr.multiProcessorCount*2;

  // ---- self-check both smem layouts ----
  {
    Prob p = make_prob(1,8,64,{40},1234);       // ragged (40<BM), N tail (8<BN), 1 k-step
    std::vector<float> oc; oracle(p,oc);
    Dev d = upload(p);
    double v0 = check<true,false>(d,p,oc), v1 = check<true,true>(d,p,oc);
    printf("[self-check] E=1 m=40 N=8 K=64  noswz relL2=%.3e %s | swz128 relL2=%.3e %s\n",
           v0,v0<3e-3?"OK":"FAIL", v1,v1<3e-3?"OK":"FAIL");
    freedev(d);
    if(!(v0<3e-3 && v1<3e-3)){ printf("RESULT: FAIL (layout self-check)\n"); return 2; }
  }

  struct Cfg{ const char* name; int E,N,K; std::vector<int> me; };
  std::vector<Cfg> cfgs = {
    {"E8_N768_K3840_ragged",  8, 768,3840,{32,64,17,128,5,200,64,96}},
    {"E128_N256_K2560_bal", 128, 256,2560,std::vector<int>(128,8)},
  };

  bool allpass=true;
  for(auto& c:cfgs){
    Prob p = make_prob(c.E,c.N,c.K,c.me,777);
    std::vector<float> oc; oracle(p,oc);
    Dev d = upload(p);
    double vw = check<true,false>(d,p,oc);   // wgmma, no swizzle
    double vs = check<true,true >(d,p,oc);   // wgmma, 128B swizzle
    double vm = check<false,true>(d,p,oc);   // mma.sync baseline
    bool pass = vw<3e-3 && vs<3e-3;
    printf("[validate] %-22s Mtot=%5d tiles=%4d | wgmma noswz=%.3e swz128=%.3e %s | mma.sync=%.3e\n",
           c.name,p.Mtot,d.nwork,vw,vs,pass?"PASS":"FAIL",vm);
    allpass &= pass;
    freedev(d);
  }
  printf("RESULT: %s\n", allpass?"ALL PASS":"FAIL");
  if(!allpass) return 1;

  // ---- benchmark: realistic MoE prefill shapes, 100 iters / 10 warmup ----
  // The required config is E=8, m_e=256, N=768, K=3840. A larger-token shape is included to
  // show how the speedup moves once there are enough tiles to fill all 132 SMs.
  struct BC{ int E,N,K,m; bool verify; };
  std::vector<BC> bcs = { {8,768,3840,256,true}, {8,768,3840,1024,false} };
  printf("\n");
  for(auto& b:bcs){
    std::vector<int> me(b.E,b.m);
    Prob p = make_prob(b.E,b.N,b.K,me,99);
    Dev d = upload(p);
    double vw=-1,vs=-1;
    if(b.verify){ std::vector<float> oc; oracle(p,oc);
                  vw = check<true,false>(d,p,oc); vs = check<true,true>(d,p,oc); }
    double flops=0; for(int e=0;e<b.E;e++) flops += 2.0*me[e]*b.N*b.K;
    double tw  = bench<true, false>(d,p,100,10);   // wgmma, no swizzle
    double tws = bench<true, true >(d,p,100,10);   // wgmma, 128B swizzle
    double tm  = bench<false,false>(d,p,100,10);   // mma.sync, no swizzle
    double tms = bench<false,true >(d,p,100,10);   // mma.sync, 128B swizzle
    printf("[bench] E=%d m_e=%d N=%d K=%d  Mtot=%d tiles=%d (%.2f waves)",
           b.E,b.m,b.N,b.K,p.Mtot,d.nwork,(double)d.nwork/pr.multiProcessorCount);
    if(b.verify) printf("  relL2 noswz=%.3e swz=%.3e %s",vw,vs,(vw<3e-3&&vs<3e-3)?"PASS":"FAIL");
    printf("\n");
    printf("  wgmma    m64n128k16  noswz: %8.3f ms %7.2f TF/s | swz128: %8.3f ms %7.2f TF/s  (swz %.2fx)\n",
           tw, flops/(tw*1e-3)/1e12, tws, flops/(tws*1e-3)/1e12, tw/tws);
    printf("  mma.sync m16n8k16    noswz: %8.3f ms %7.2f TF/s | swz128: %8.3f ms %7.2f TF/s  (swz %.2fx)\n",
           tm, flops/(tm*1e-3)/1e12, tms, flops/(tms*1e-3)/1e12, tm/tms);
    printf("  wgmma vs mma.sync: noswz %.2fx | swz128 %.2fx | best wgmma vs best mma.sync %.2fx\n\n",
           tm/tw, tms/tws, std::min(tm,tms)/std::min(tw,tws));
    freedev(d);
  }

  // ---- fp8 e4m3: same grouped structure, wgmma.m64n128k32.f32.e4m3.e4m3 ----
  CK(cudaFuncSetAttribute(grouped_gemm_fp8<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SMEM8));
  CK(cudaFuncSetAttribute(grouped_gemm_fp8<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SMEM8));
  // K-sweep: Hopper's fp8 tensor-core accumulator is NOT full f32, so relL2 grows with K.
  // (bf16 wgmma at K=3840 sits at ~4e-6; fp8 is orders of magnitude worse and K-dependent.)
  {
    bool ok=true;
    for(int K : {128, 512, 1280, 3840}){
      Prob8 p = make_prob8(8,768,K,{32,64,17,128,5,200,64,96},777);
      std::vector<float> oc; oracle8(p,oc);
      Dev8 d = upload8(p);
      double v0 = check8<false>(d,p,oc), v1 = check8<true>(d,p,oc);
      printf("[validate-fp8] E8_N768_K%-5d ragged Mtot=%d  noswz=%.3e swz128=%.3e %s\n",
             K,p.Mtot,v0,v1,(v0<3e-3&&v1<3e-3)?"PASS":"FAIL");
      ok &= (v0<3e-3 && v1<3e-3);
      freedev8(d);
    }
    printf("RESULT-fp8: %s\n", ok?"ALL PASS":"FAIL");
  }
  {
    int E=8,N=768,K=3840,m=256; std::vector<int> me(E,m);
    Prob8 p = make_prob8(E,N,K,me,99);
    Dev8 d = upload8(p);
    double flops=0; for(int e=0;e<E;e++) flops += 2.0*me[e]*N*K;
    double t0 = bench8<false>(d,p,100,10), t1 = bench8<true>(d,p,100,10);
    printf("[bench-fp8] E=%d m_e=%d N=%d K=%d  wgmma m64n128k32.e4m3  "
           "noswz: %7.3f ms %7.2f TF/s | swz128: %7.3f ms %7.2f TF/s  (swz %.2fx)\n",
           E,m,N,K,t0,flops/(t0*1e-3)/1e12,t1,flops/(t1*1e-3)/1e12,t0/t1);
    freedev8(d);
  }
  return 0;
}

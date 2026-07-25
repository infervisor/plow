// Hopper (sm_90a) thread-block CLUSTER + TMA MULTICAST probe/benchmark.
//
// Answers, on H100 NVL:
//   1. Do clusters launch at all, and what is the max cluster size?
//   2. Does distributed shared memory (DSMEM, `mapa` / cluster.map_shared_rank)
//      work between CTAs of a cluster?
//   3. Does `cp.async.bulk.tensor.2d...multicast::cluster` work, and what SASS
//      does it emit vs the non-multicast form?
//   4. Does multicasting a SHARED WEIGHT TILE to C CTAs beat each-CTA-loads-its-own
//      at plow's Gemma prefill shapes (K=3840, N=4096 / N=15360)?
//   5. Does requiring a cluster dim break plow's persistent cooperative
//      megakernel invariant (grid == occ x SM_count, all blocks co-resident)?
//
// BUILD (executables MUST use the -gencode form; -arch=sm_90a -o exe is rejected
// for sm_90a-only opcodes, and -arch=native resolves to plain sm_90):
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 \
//        -I runtime/common -I runtime/nvidia -include cstdint \
//        runtime/nvidia/experiments/hopper_cluster_multicast.cu -lcuda -o /tmp/hcm
//
// Every GPU run must be serialized:  flock /tmp/plow_gpu.lock /tmp/hcm
//
// L2 DISCIPLINE: H100 NVL L2 is 50 MB. Every weight buffer here is replicated
// past ~470 MB and cycled, or the numbers are L2, not HBM. The real streaming
// ceiling is measured first (grid-stride float4 read of 2.5 GB); anything above
// it is invalid by construction.

// ===========================================================================
// RESULTS (H100 NVL, 132 SMs, L2 = 60 MB measured, CUDA 13.0, driver 570)
// ===========================================================================
// PRIMITIVES — all three exist and are correct:
//   * max cluster size:  8 portable, 16 with cudaFuncAttributeNonPortableCluster-
//     SizeAllowed; 32 rejected (cudaErrorInvalidClusterSize). Ranks verified at
//     every size. cudaOccupancyMaxActiveClusters: 528 @C=2, 248 @C=4, 124 @C=8.
//   * DSMEM: PASS at C=2/4/8 — every CTA read every peer's smem. `mapa` folds
//     into address math on SR_CgaCtaId; peer reads issue as LD.E through the
//     distributed-shared aperture. Cluster barrier = UCGABAR_ARV / UCGABAR_WAIT
//     (+ CGAERRBAR, MEMBAR.ALL.GPU, CCTL.IVALL — an expensive sequence).
//   * multicast TMA: PASS — ONE issue by rank 0 leaves the identical tile in all
//     C CTAs. SASS: `UTMALDG.2D` (plain) vs `UTMALDG.2D.MULTICAST [UR],[UR],UR`
//     (mask in a uniform register). The two benchmark kernels are otherwise
//     instruction-identical (35 LDS.128, 5 TMA, same mbarrier ops).
//
// MEASURED HBM CEILING: 3697 GB/s (92% of the 4023 GB/s spec).
//   This kernel's transport-only ceiling (LITE, C=1) is 2880-3197 GB/s.
//
// MULTICAST vs PER-CTA (BN=128 BK=64 NSTAGE=4, 94.5 KB smem, 2 blk/SM, grid 264):
//   K=3840, N=4096 (weights replicated to 510 MB)
//     C=2   per-CTA 0.491 ms  |  multicast 0.524 ms   -> 0.94x
//     C=4   per-CTA 1.418 ms  |  multicast 1.622 ms   -> 0.87x
//     C=8   per-CTA 2.549 ms  |  multicast 3.280 ms   -> 0.78x
//   K=3840, N=15360 (562 MB)
//     C=2   per-CTA 0.514 ms  |  multicast 0.558 ms   -> 0.92x
//     C=4   per-CTA 1.584 ms  |  multicast 1.820 ms   -> 0.87x
//     C=8   per-CTA 3.158 ms  |  multicast 3.646 ms   -> 0.87x
//   Transport-only (LITE) at N=4096: per-CTA 2718 / 964 / 497 GB/s unique at
//   C=2/4/8 vs multicast 2662 / 917 / 494. A 2x-deeper pipeline (BN=64,
//   NSTAGE=8, identical smem) reproduces it exactly: 576 vs 571 and 275 vs 278
//   GB/s — so the tie is NOT a producer-concurrency artifact.
//
// MULTICAST NEVER WINS, AND HERE IS WHY:
//   The C duplicate reads in the per-CTA baseline never reach DRAM. They are
//   L2 hits. At C=2 the baseline sustains 5996 GB/s of L2->SM traffic on top of
//   a 2998 GB/s DRAM stream — L2 has ~2x the DRAM ceiling in reserve, so
//   deduplicating those reads buys nothing. Multicast also does NOT reduce the
//   shared-memory WRITE volume: C x 16 KB still lands in C CTAs' smem either
//   way. It removes the one resource that was not scarce and adds DSMEM
//   mbarrier round-trips (remote arrive + the producer waiting on C credits),
//   which is the 3-13% it loses. Multicast would only pay if C x DRAM_rate
//   exceeded L2 bandwidth; on H100 that needs C > ~2, and by C=4 the kernel is
//   already limited by unique-bytes-in-flight, which multicast does not change.
//
// PLOW INTEGRATION — two independent blockers, either one is fatal:
//   1. GRID DIVISIBILITY. plow launches the interpreter with
//      cuLaunchCooperativeKernel at grid = occ x 132 and hard-gates
//      grid == packet n_cu (exec/gpu.rs:652, :1979). A cluster dim must divide
//      the grid. 132 = 4 x 33, so C=2 and C=4 always divide; C=8 divides only
//      when occ is even (264 OK, 132/396 REJECTED with
//      cudaErrorInvalidClusterSize); C=16 never divides. Confirmed by launch.
//      Also: cuLaunchCooperativeKernel cannot carry a cluster dim at all — plow
//      would have to move to cuLaunchKernelEx with CU_LAUNCH_ATTRIBUTE_-
//      COOPERATIVE + _CLUSTER_DIMENSION (verified ACCEPTED together here), i.e.
//      a new driver symbol in device/cuda.rs.
//   2. THE INTERPRETER HAS NO LOCKSTEP TO BUILD A CLUSTER ON. Every block walks
//      its OWN instruction stream (interp_sm120.cu:1283 `prog.stream_ofs[cu]`,
//      `stream_len[cu]`) with per-entry counter gates, and the GQ path claims
//      work by atomicAdd. Blocks are heterogeneous by construction: at any
//      instant the C CTAs of a cluster may be in different ops of different
//      layers. Multicast REQUIRES all C to be inside the same tile loop (every
//      consumer must publish expect_tx before the producer issues, and the
//      producer must collect C empty-credits per stage). A cluster barrier
//      across divergent streams deadlocks. Making it safe means the packet
//      compiler must guarantee cluster-aligned identical schedules for every
//      group of C blocks — a change to the scheduling model, not the kernel.
//
// VERDICT: works, does not pay, and conflicts with the persistent-grid model.
//   DO NOT PURSUE for plow prefill GEMM/MoE. If Hopper prefill needs more
//   transport, the README's existing conclusion still holds: TMA + producer/
//   consumer warp specialization + 128x256 tiles WITHIN one CTA, no clusters.
// ===========================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

namespace cg = cooperative_groups;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CKD(x) do{ CUresult e=(x); if(e!=CUDA_SUCCESS){ const char*s; cuGetErrorString(e,&s); \
  printf("CU ERR %s @%d: %s\n",#x,__LINE__,s); exit(1);} }while(0)

// ------------------------------------------------------------------ tunables
// -DPLOW_BN / -DPLOW_NSTAGE let a second build re-run the sweep at constant smem
// but 2x the pipeline depth, to prove the result is not a producer-concurrency
// artifact (multicast has C x fewer producers than the per-CTA baseline).
#ifndef PLOW_BN
#define PLOW_BN 128
#endif
#ifndef PLOW_NSTAGE
#define PLOW_NSTAGE 4
#endif
static const int BN      = PLOW_BN;   // weight tile rows (N)
static const int BK      = 64;        // weight tile cols (K)  -> 128 B innermost
static const int NSTAGE  = PLOW_NSTAGE;  // pipeline depth
static const int THREADS = 256;
static const int BM      = 2;     // A rows per CTA; kept in smem as f32 so the
                                  // consume loop is LDS.128 + FMA only (transport-bound,
                                  // not scalar-LDS-bound -- see the header note).
static const int CMAX    = 16;    // largest cluster we ever try

static const int TILE_B  = BN * BK * 2;   // 16384 B per weight tile

// ------------------------------------------------------- mbarrier/TMA helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p){
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt){
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :: "r"(smem_u32(b)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, int bytes){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int phase){
  asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;"
               " @!p bra W%=; }" :: "r"(smem_u32(b)), "r"(phase) : "memory");
}
// DSMEM address translation: local shared addr -> same offset in CTA `rank`.
__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, int rank){
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
// Arrive on an mbarrier that may live in ANOTHER CTA of the cluster.
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t addr){
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(addr) : "memory");
}
// TMA 2-D, single destination CTA  (the current plow behaviour).
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map,
                                      int c0, int c1, uint32_t bar){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
// TMA 2-D MULTICAST: one HBM read -> every CTA whose bit is set in `mask`.
__device__ __forceinline__ void tma2d_mc(uint32_t dst, const CUtensorMap* map,
                                         int c0, int c1, uint32_t bar, uint16_t mask){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               ".multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
               :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}

// ============================================================ 0. HBM ceiling
// Grid-stride float4 read. Buffer must be >= 2 GB so L2 (50 MB) is irrelevant.
__global__ void k_stream(const float4* __restrict__ p, size_t n4, float* out){
  float4 a = make_float4(0,0,0,0);
  for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n4;
      i += (size_t)gridDim.x*blockDim.x){
    float4 v = p[i]; a.x+=v.x; a.y+=v.y; a.z+=v.z; a.w+=v.w;
  }
  if(a.x==12345.f) out[0]=a.x+a.y+a.z+a.w;   // never true; defeats DCE
}

// ==================================================== 1. cluster launch probe
__global__ void k_cluster_probe(int* out){
  cg::cluster_group cl = cg::this_cluster();
  if(threadIdx.x==0){
    out[blockIdx.x*2+0] = (int)cl.num_blocks();
    out[blockIdx.x*2+1] = (int)cl.block_rank();
  }
}

// ============================================================== 2. DSMEM probe
// Every CTA writes its rank-tagged pattern into its own smem; after a cluster
// barrier, each CTA reads EVERY peer's smem through `mapa` and checksums it.
// A correct result proves distributed shared memory is real.
__global__ void k_dsmem_probe(int* out, int C){
  __shared__ int buf[32];
  cg::cluster_group cl = cg::this_cluster();
  const int rank = (int)cl.block_rank();
  if(threadIdx.x < 32) buf[threadIdx.x] = rank*1000 + threadIdx.x;
  cl.sync();
  if(threadIdx.x == 0){
    int sum = 0, expect = 0;
    for(int r=0;r<C;r++){
      uint32_t peer = mapa_shared(smem_u32(buf), r);
      for(int i=0;i<32;i++){
        int v;
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(v) : "r"(peer + i*4));
        sum += v;
        expect += r*1000 + i;
      }
    }
    out[blockIdx.x*2+0] = sum;
    out[blockIdx.x*2+1] = expect;
  }
  cl.sync();   // no CTA may exit while peers still read its smem
}

// ============================== 3. multicast TMA correctness (small, verified)
// Rank 0 issues ONE multicast tensor load; every CTA in the cluster must end up
// with the identical tile in its own smem.
__global__ void k_mc_probe(const __grid_constant__ CUtensorMap map, int C,
                           unsigned long long* out){
  extern __shared__ char smem_raw[];
  uint32_t tb = (smem_u32(smem_raw) + 127u) & ~127u;
  __shared__ __align__(8) uint64_t bar;
  cg::cluster_group cl = cg::this_cluster();
  const int rank = (int)cl.block_rank();

  if(threadIdx.x==0) mbar_init(&bar, 1);
  __syncthreads();
  cl.sync();                              // every CTA's barrier is live

  if(threadIdx.x==0) mbar_expect(&bar, TILE_B);
  cl.sync();                              // all expect_tx precede the issue
  if(rank==0 && threadIdx.x==0)
    tma2d_mc(tb, &map, 0, 0, smem_u32(&bar), (uint16_t)((1u<<C)-1u));
  if(threadIdx.x==0) mbar_wait(&bar, 0);
  __syncthreads();

  const __nv_bfloat16* t = (const __nv_bfloat16*)__cvta_shared_to_generic(tb);
  unsigned long long s = 0;
  for(int i=threadIdx.x;i<BN*BK;i+=blockDim.x)
    s += (unsigned long long)(unsigned int)__bfloat162float(t[i]);
  atomicAdd(&out[blockIdx.x], s);
  cl.sync();
}

// ============ 4. SASS witness: two one-line kernels, nothing else in the body
__global__ void k_sass_plain(const __grid_constant__ CUtensorMap map, uint32_t* sink){
  __shared__ __align__(128) char tile[TILE_B];
  __shared__ __align__(8) uint64_t bar;
  if(threadIdx.x==0){
    mbar_init(&bar,1); mbar_expect(&bar,TILE_B);
    tma2d(smem_u32(tile), &map, 0, 0, smem_u32(&bar));
    mbar_wait(&bar,0);
  }
  __syncthreads();
  sink[0] = (uint32_t)tile[threadIdx.x];
}
__global__ void k_sass_mc(const __grid_constant__ CUtensorMap map, uint32_t* sink){
  __shared__ __align__(128) char tile[TILE_B];
  __shared__ __align__(8) uint64_t bar;
  cg::cluster_group cl = cg::this_cluster();
  if(threadIdx.x==0){ mbar_init(&bar,1); mbar_expect(&bar,TILE_B); }
  cl.sync();
  if(cl.block_rank()==0 && threadIdx.x==0)
    tma2d_mc(smem_u32(tile), &map, 0, 0, smem_u32(&bar), 0xffu);
  if(threadIdx.x==0) mbar_wait(&bar,0);
  __syncthreads();
  sink[0] = (uint32_t)tile[threadIdx.x];
  cl.sync();
}

// ===================================================== 5. the real benchmark
//
// Weight-broadcast GEMM-like loop.  A cluster of C CTAs all consume the SAME
// B tile B[n-tile][k-stage]; CTA rank r owns a different A row-block.
// Work item = (replica, n-tile); replicas exist only to blow past L2.
// The global weight view is [R*N, K] bf16, so ONE tensor map covers every
// replica: row coord = rep*N + ntile*BN.
//
// MC=true  -> rank 0 issues one .multicast::cluster load per (item,kstage);
//             the cluster's `empty` barriers live in rank 0's smem and every
//             consumer arrives on them through DSMEM.
// MC=false -> every CTA issues its own load of the same tile (plow today).

template<bool MC, bool LITE>
__global__ __launch_bounds__(THREADS)
void k_bcast(const __grid_constant__ CUtensorMap map,
             const __nv_bfloat16* __restrict__ A,
             float* __restrict__ out,
             int K, int nNTiles, int nItems, int nClusters, int C){
  extern __shared__ char smem_raw[];
  uint32_t base = (smem_u32(smem_raw) + 127u) & ~127u;
  const uint32_t tile0 = base;                       // NSTAGE x TILE_B
  float* As = (float*)__cvta_shared_to_generic(base + NSTAGE*TILE_B);  // [BM][K] f32
  __shared__ __align__(8) uint64_t full[NSTAGE];
  __shared__ __align__(8) uint64_t empty[NSTAGE];    // only rank 0's copy is used

  cg::cluster_group cl = cg::this_cluster();
  const int rank      = MC ? (int)cl.block_rank() : (int)(blockIdx.x % (unsigned)C);
  const int clusterId = MC ? (int)(blockIdx.x / (unsigned)C) : (int)(blockIdx.x / (unsigned)C);
  const int tid = threadIdx.x;

  // A row-block for this rank, resident for the whole kernel (BM x K bf16).
  for(int i=tid;i<BM*K;i+=THREADS){
    int m = i / K, k = i % K;
    As[i] = __bfloat162float(A[(size_t)(rank*BM+m)*K + k]);
  }
  if(tid==0){
    #pragma unroll
    for(int s=0;s<NSTAGE;s++){ mbar_init(&full[s],1); mbar_init(&empty[s], MC?C:1); }
  }
  __syncthreads();
  if(MC) cl.sync();

  const int nk = K / BK;
  const uint16_t mask = (uint16_t)((1u<<C)-1u);
  // this cluster's item list: clusterId, clusterId+nClusters, ...
  const int myItems = (nItems - clusterId + nClusters - 1) / nClusters;
  const int T = myItems * nk;               // total pipeline stages
  if(T <= 0) return;

  uint32_t empty_addr[NSTAGE];              // rank 0's empty[s], via DSMEM
  #pragma unroll
  for(int s=0;s<NSTAGE;s++)
    empty_addr[s] = MC ? mapa_shared(smem_u32(&empty[s]), 0) : smem_u32(&empty[s]);

  auto coords = [&](int t, int& c0, int& c1){
    int item = clusterId + (t / nk) * nClusters;
    int rep  = item / nNTiles, nt = item % nNTiles;
    c0 = (t % nk) * BK;
    c1 = rep * (nNTiles*BN) + nt * BN;
  };

  // ---- prologue: publish expect_tx + credit, then rank 0 (or self) issues
  if(tid==0){
    #pragma unroll
    for(int s=0;s<NSTAGE;s++){
      if(s < T){ mbar_expect(&full[s], TILE_B); mbar_arrive_cluster(empty_addr[s]); }
    }
  }
  if(MC) cl.sync(); else __syncthreads();
  if(tid==0 && (!MC || rank==0)){
    for(int s=0;s<NSTAGE && s<T;s++){
      int c0,c1; coords(s,c0,c1);
      if(MC) tma2d_mc(tile0 + s*TILE_B, &map, c0, c1, smem_u32(&full[s]), mask);
      else   tma2d   (tile0 + s*TILE_B, &map, c0, c1, smem_u32(&full[s]));
    }
  }

  float acc[BM];
  #pragma unroll
  for(int m=0;m<BM;m++) acc[m]=0.f;

  for(int t=0;t<T;t++){
    const int s = t % NSTAGE, j = t / NSTAGE;
    if(tid==0) mbar_wait(&full[s], j & 1);
    __syncthreads();

    // consume: acc[m] += sum_over_tile B[n][k] * A[m][kbase+k]
    // LITE = transport-only control: touch every 128 B of the tile once and do
    // no math, so the measured rate is the kernel's pure TMA delivery ceiling.
    const __nv_bfloat16* tb = (const __nv_bfloat16*)__cvta_shared_to_generic(tile0 + s*TILE_B);
    const int kbase = (t % nk) * BK;
    if(LITE){
      for(int e = tid*8; e < BN*BK; e += THREADS*8){
        int4 v = *(const int4*)(tb + e);
        acc[0] += (float)(v.x ^ v.y ^ v.z ^ v.w);
      }
    } else
    for(int e = tid*8; e < BN*BK; e += THREADS*8){
      int4 v = *(const int4*)(tb + e);          // 8 bf16 of B, one LDS.128
      const __nv_bfloat16* b8 = (const __nv_bfloat16*)&v;
      const float* ap = As + kbase + (e % BK);
      float av[BM][8];
      #pragma unroll
      for(int m=0;m<BM;m++){                    // 2 LDS.128 per A row (8 f32)
        *(float4*)&av[m][0] = *(const float4*)(ap + m*K);
        *(float4*)&av[m][4] = *(const float4*)(ap + m*K + 4);
      }
      #pragma unroll
      for(int q=0;q<8;q++){
        float bv = __bfloat162float(b8[q]);
        #pragma unroll
        for(int m=0;m<BM;m++) acc[m] = fmaf(bv, av[m][q], acc[m]);
      }
    }
    __syncthreads();

    // credit + re-arm, then run the producer NSTAGE ahead
    if(tid==0){
      if(t+NSTAGE < T) mbar_expect(&full[s], TILE_B);
      mbar_arrive_cluster(empty_addr[s]);
    }
    if(t+NSTAGE < T && tid==0 && (!MC || rank==0)){
      mbar_wait(&empty[s], (j+1) & 1);
      int c0,c1; coords(t+NSTAGE,c0,c1);
      if(MC) tma2d_mc(tile0 + s*TILE_B, &map, c0, c1, smem_u32(&full[s]), mask);
      else   tma2d   (tile0 + s*TILE_B, &map, c0, c1, smem_u32(&full[s]));
    }
  }

  float r = 0.f;
  #pragma unroll
  for(int m=0;m<BM;m++) r += acc[m];
  if(r == 1.2345e-30f) out[blockIdx.x] = r;    // DCE guard, never taken
  if(MC) cl.sync();
}

// ------------------------------------------------------------------- harness
static double ms_of(cudaEvent_t a, cudaEvent_t b){ float m; cudaEventElapsedTime(&m,a,b); return m; }

struct Row { int C; int N; bool mc; double ms; double useful; double issued; };

int main(){
  setvbuf(stdout,nullptr,_IONBF,0);
  CKD(cuInit(0));
  int dev=0; CK(cudaSetDevice(dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
  int clSup=0, dsm=0, mclk=0;
  CK(cudaDeviceGetAttribute(&clSup, cudaDevAttrClusterLaunch, dev));
  CK(cudaDeviceGetAttribute(&dsm,   cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  CK(cudaDeviceGetAttribute(&mclk,  cudaDevAttrMemoryClockRate, dev));  // kHz
  const double specBW = 2.0*(double)mclk*1e3*(p.memoryBusWidth/8)/1e9;
  printf("# %s  sm_%d%d  SMs=%d  L2=%.1f MB  smem/blk-optin=%.0f KB\n",
         p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize/1048576.0, dsm/1024.0);
  printf("# clusterLaunchSupported=%d  memBusWidth=%d bit  memClock=%.2f GHz  spec BW=%.0f GB/s\n",
         clSup, p.memoryBusWidth, mclk*1e-6, specBW);
  const int SM = p.multiProcessorCount;

  // ---------------------------------------------------------- 0. HBM ceiling
  printf("\n=== 0. streaming ceiling (grid-stride float4, 2.5 GB, L2=%.0f MB) ===\n",
         p.l2CacheSize/1048576.0);
  size_t bigBytes = 3ull<<30;                 // 3 GB arena, reused by the bench
  void* dBig=nullptr; CK(cudaMalloc(&dBig,bigBytes)); CK(cudaMemset(dBig,0x3c,bigBytes));
  float* dOut=nullptr; CK(cudaMalloc(&dOut, 1<<20));
  double ceiling = 0;
  {
    size_t sb = (size_t)2560<<20;             // 2.5 GB
    size_t n4 = sb/16;
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for(int g : {SM*4, SM*8, SM*16}){
      k_stream<<<g,256>>>((const float4*)dBig,n4,dOut); CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(e0));
      for(int i=0;i<5;i++) k_stream<<<g,256>>>((const float4*)dBig,n4,dOut);
      CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
      double ms = ms_of(e0,e1)/5.0, gbs = sb/1e9/(ms*1e-3);
      printf("  grid=%-5d  %.3f ms  %.0f GB/s\n", g, ms, gbs);
      ceiling = std::max(ceiling,gbs);
    }
    printf("  >>> measured HBM ceiling = %.0f GB/s (%.0f%% of spec)\n",
           ceiling, 100.0*ceiling/specBW);
  }

  // -------------------------------------------------- 1. max cluster size
  printf("\n=== 1. cluster launch: max supported size ===\n");
  int* dI=nullptr; CK(cudaMalloc(&dI, 4096*sizeof(int)));
  int maxPortable=0, maxNonPortable=0;
  {
    int mp=0;
    { cudaLaunchConfig_t c0{}; c0.gridDim=dim3(SM,1,1); c0.blockDim=dim3(64,1,1);
      cudaError_t ee = cudaOccupancyMaxPotentialClusterSize(&mp,(void*)k_cluster_probe,&c0);
      printf("  cudaOccupancyMaxPotentialClusterSize = %d (%s)\n", mp, cudaGetErrorString(ee));
      cudaGetLastError(); }
    for(int C : {1,2,3,4,6,8,12,16,32}){
      // sizes > 8 are "non-portable" and need an explicit opt-in
      if(C>8) cudaFuncSetAttribute((void*)k_cluster_probe,
                cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      cudaLaunchConfig_t cfg{}; cudaLaunchAttribute at[1]{};
      cfg.gridDim=dim3(C*4,1,1); cfg.blockDim=dim3(64,1,1);
      at[0].id=cudaLaunchAttributeClusterDimension; at[0].val.clusterDim={(unsigned)C,1,1};
      cfg.attrs=at; cfg.numAttrs=1;
      cudaGetLastError();
      cudaError_t e = cudaLaunchKernelEx(&cfg, k_cluster_probe, dI);
      if(e==cudaSuccess) e = cudaDeviceSynchronize();
      std::vector<int> h(C*4*2,0);
      if(e==cudaSuccess) CK(cudaMemcpy(h.data(),dI,h.size()*4,cudaMemcpyDeviceToHost));
      int nb = e==cudaSuccess ? h[0] : -1;
      // sanity: ranks 0..C-1 must all appear in the first cluster
      bool ok = (e==cudaSuccess && nb==C);
      if(ok){ int m=0; for(int b=0;b<C;b++) m |= 1<<h[b*2+1]; ok = (m == ((1<<C)-1)); }
      printf("  cluster=%-3d %-8s  num_blocks=%-3d %s\n", C,
             e==cudaSuccess?"OK":"FAIL", nb, e==cudaSuccess?(ok?"ranks verified":"RANKS BAD")
                                                          :cudaGetErrorString(e));
      int amc=0; if(e==cudaSuccess){
        cudaError_t ee = cudaOccupancyMaxActiveClusters(&amc,(void*)k_cluster_probe,&cfg);
        if(ee==cudaSuccess) printf("       cudaOccupancyMaxActiveClusters = %d\n", amc);
        cudaGetLastError();
      }
      if(ok){ if(C<=8) maxPortable=std::max(maxPortable,C); maxNonPortable=std::max(maxNonPortable,C); }
      cudaGetLastError();
    }
    printf("  >>> max portable cluster = %d ; max (non-portable opt-in) = %d\n",
           maxPortable, maxNonPortable);
  }

  // --------------------------------------------------------- 2. DSMEM probe
  printf("\n=== 2. distributed shared memory (mapa / ld.shared::cluster) ===\n");
  for(int C : {2,4,8}){
    if(C>maxNonPortable) continue;
    cudaLaunchConfig_t cfg{}; cudaLaunchAttribute at[1]{};
    cfg.gridDim=dim3(C,1,1); cfg.blockDim=dim3(64,1,1);
    at[0].id=cudaLaunchAttributeClusterDimension; at[0].val.clusterDim={(unsigned)C,1,1};
    cfg.attrs=at; cfg.numAttrs=1;
    CK(cudaMemset(dI,0,4096*4));
    CK(cudaLaunchKernelEx(&cfg, k_dsmem_probe, dI, C));
    CK(cudaDeviceSynchronize());
    std::vector<int> h(C*2); CK(cudaMemcpy(h.data(),dI,C*2*4,cudaMemcpyDeviceToHost));
    bool ok=true; for(int b=0;b<C;b++) ok &= (h[b*2]==h[b*2+1]);
    printf("  C=%-2d  every CTA read all %d peers' smem: %s (got %d, expect %d)\n",
           C, C, ok?"PASS":"FAIL", h[0], h[1]);
  }

  // ------------------------------ 3. multicast TMA correctness + 4. SASS ref
  printf("\n=== 3. cp.async.bulk.tensor.2d .multicast::cluster correctness ===\n");
  {
    // small 2-D view: [BN, BK] bf16 with a known pattern
    std::vector<__nv_bfloat16> ht(BN*BK);
    unsigned long long expect1=0;
    for(int i=0;i<BN*BK;i++){ ht[i]=__float2bfloat16((float)(i%97)); expect1 += (unsigned)(i%97); }
    CK(cudaMemcpy(dBig,ht.data(),ht.size()*2,cudaMemcpyHostToDevice));
    CUtensorMap map; memset(&map,0,sizeof(map));
    uint64_t gd[2]={(uint64_t)BK,(uint64_t)BN}; uint64_t gs[1]={(uint64_t)BK*2};
    uint32_t bd[2]={(uint32_t)BK,(uint32_t)BN}; uint32_t es[2]={1,1};
    CKD(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dBig,gd,gs,bd,es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    size_t sm = TILE_B + 256;
    CK(cudaFuncSetAttribute((void*)k_mc_probe,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm));
    unsigned long long* dU; CK(cudaMalloc(&dU, CMAX*8));
    for(int C : {2,4,8}){
      if(C>maxNonPortable) continue;
      CK(cudaMemset(dU,0,CMAX*8));
      cudaLaunchConfig_t cfg{}; cudaLaunchAttribute at[1]{};
      cfg.gridDim=dim3(C,1,1); cfg.blockDim=dim3(128,1,1); cfg.dynamicSmemBytes=sm;
      at[0].id=cudaLaunchAttributeClusterDimension; at[0].val.clusterDim={(unsigned)C,1,1};
      cfg.attrs=at; cfg.numAttrs=1;
      cudaError_t e = cudaLaunchKernelEx(&cfg, k_mc_probe, map, C, dU);
      if(e==cudaSuccess) e=cudaDeviceSynchronize();
      if(e!=cudaSuccess){ printf("  C=%-2d LAUNCH/RUN FAIL: %s\n",C,cudaGetErrorString(e));
                          cudaGetLastError(); continue; }
      std::vector<unsigned long long> hu(C); CK(cudaMemcpy(hu.data(),dU,C*8,cudaMemcpyDeviceToHost));
      bool ok=true; for(int b=0;b<C;b++) ok &= (hu[b]==expect1);
      printf("  C=%-2d  ONE multicast issued by rank0 -> all %d CTAs hold the tile: %s"
             "  (cta0 sum %llu, expect %llu)\n", C, C, ok?"PASS":"FAIL",
             (unsigned long long)hu[0], expect1);
    }
    // SASS witnesses (compiled in; dump with cuobjdump -sass)
    uint32_t* dS; CK(cudaMalloc(&dS,1024));
    k_sass_plain<<<1,128>>>(map,dS);
    { cudaLaunchConfig_t cfg{}; cudaLaunchAttribute at[1]{};
      cfg.gridDim=dim3(8,1,1); cfg.blockDim=dim3(128,1,1);
      at[0].id=cudaLaunchAttributeClusterDimension; at[0].val.clusterDim={8,1,1};
      cfg.attrs=at; cfg.numAttrs=1;
      cudaLaunchKernelEx(&cfg,k_sass_mc,map,dS); }
    CK(cudaDeviceSynchronize());
    printf("  (k_sass_plain / k_sass_mc compiled in — dump with `cuobjdump -sass`)\n");
  }

  // ------------------------------------------------------- 5. the benchmark
  printf("\n=== 5. weight-broadcast benchmark  (K=3840, BN=%d BK=%d NSTAGE=%d BM=%d) ===\n",
         BN,BK,NSTAGE,BM);
  const int K = 3840;
  size_t smB = (size_t)NSTAGE*TILE_B + (size_t)BM*K*4 + 512;   // A staged as f32
  printf("  dynamic smem = %.1f KB/CTA\n", smB/1024.0);
  CK(cudaFuncSetAttribute((void*)k_bcast<true,false>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smB));
  CK(cudaFuncSetAttribute((void*)k_bcast<true,true>,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smB));
  CK(cudaFuncSetAttribute((void*)k_bcast<false,true>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smB));
  CK(cudaFuncSetAttribute((void*)k_bcast<false,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smB));
  int occT=0,occF=0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occT,(void*)k_bcast<true,false>, THREADS,smB));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occF,(void*)k_bcast<false,false>,THREADS,smB));
  printf("  occupancy: multicast %d blk/SM, plain %d blk/SM\n", occT, occF);

  __nv_bfloat16* dA; CK(cudaMalloc(&dA,(size_t)CMAX*BM*K*2));
  {
    std::vector<__nv_bfloat16> ha((size_t)CMAX*BM*K);
    for(size_t i=0;i<ha.size();i++) ha[i]=__float2bfloat16(0.001f*(float)(i%251));
    CK(cudaMemcpy(dA,ha.data(),ha.size()*2,cudaMemcpyHostToDevice));
  }
  std::vector<Row> rows;
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  for(int N : {4096, 15360}){
    const int nNTiles = N/BN;
    const size_t perRep = (size_t)N*K*2;
    const int R = (int)std::max<size_t>(2, (480ull<<20)/perRep + 1);
    const size_t need = perRep*(size_t)R;
    if(need > bigBytes){ printf("  N=%d: needs %.2f GB > arena\n",N,need/1e9); continue; }
    const int nItems = R*nNTiles;
    const double unique = (double)need;
    printf("\n  --- N=%-6d  R=%-3d replicas  weights=%.0f MB (L2=%.0f MB)  items=%d ---\n",
           N,R,need/1048576.0,p.l2CacheSize/1048576.0,nItems);

    CUtensorMap map; memset(&map,0,sizeof(map));
    uint64_t gd[2]={(uint64_t)K,(uint64_t)N*(uint64_t)R}; uint64_t gs[1]={(uint64_t)K*2};
    uint32_t bd[2]={(uint32_t)BK,(uint32_t)BN}; uint32_t es[2]={1,1};
    CKD(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dBig,gd,gs,bd,es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    for(int C : {1,2,4,8}){
      if(C>maxNonPortable) continue;
      const int occ = std::min(occT,occF);
      int nClusters = std::max(1,(SM*occ)/C);
      int grid = nClusters*C;
      for(int v=0; v<4; v++){
        const int mc = v&1, lite = v>>1;
        if(mc==1 && C==1) continue;                 // multicast of 1 is a no-op
        cudaLaunchConfig_t cfg{}; cudaLaunchAttribute at[1]{};
        cfg.gridDim=dim3(grid,1,1); cfg.blockDim=dim3(THREADS,1,1); cfg.dynamicSmemBytes=smB;
        at[0].id=cudaLaunchAttributeClusterDimension; at[0].val.clusterDim={(unsigned)C,1,1};
        cfg.attrs=at; cfg.numAttrs=1;
        auto go = [&]()->cudaError_t{
          if(lite) return mc ? cudaLaunchKernelEx(&cfg,k_bcast<true ,true >,map,dA,dOut,K,nNTiles,nItems,nClusters,C)
                             : cudaLaunchKernelEx(&cfg,k_bcast<false,true >,map,dA,dOut,K,nNTiles,nItems,nClusters,C);
          return          mc ? cudaLaunchKernelEx(&cfg,k_bcast<true ,false>,map,dA,dOut,K,nNTiles,nItems,nClusters,C)
                             : cudaLaunchKernelEx(&cfg,k_bcast<false,false>,map,dA,dOut,K,nNTiles,nItems,nClusters,C);
        };
        cudaError_t e = go(); if(e==cudaSuccess) e=cudaDeviceSynchronize();
        if(e!=cudaSuccess){
          printf("    C=%-2d %-10s%s LAUNCH FAIL: %s\n",C,mc?"multicast":"per-CTA",
                 lite?" LITE":"", cudaGetErrorString(e));
          cudaGetLastError(); continue;
        }
        double best=1e18;
        for(int rep=0;rep<3;rep++){
          CK(cudaEventRecord(e0));
          for(int i=0;i<4;i++) CK(go());
          CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
          best = std::min(best, ms_of(e0,e1)/4.0);
        }
        double useful = unique/1e9/(best*1e-3);
        double issued = useful*(mc?1.0:(double)C);
        printf("    C=%-2d grid=%-4d %-9s%-5s %8.3f ms   unique %7.0f GB/s   issued %7.0f GB/s%s\n",
               C, grid, mc?"multicast":"per-CTA", lite?"LITE":"", best, useful, issued,
               issued>ceiling*1.02 ? "   <-- ABOVE HBM CEILING (L2 served)" : "");
        if(!lite) rows.push_back({C,N,(bool)mc,best,useful,issued});
      }
      // speedup line
      double a=0,b=0; for(auto&r:rows) if(r.C==C&&r.N==N){ (r.mc?b:a)=r.ms; }
      if(a>0&&b>0) printf("    C=%-2d              multicast speedup = %.3fx\n",C,a/b);
    }
  }

  // ------------------------------- 6. cooperative + cluster co-residency test
  printf("\n=== 6. cooperative launch + cluster dim (plow's persistent-grid invariant) ===\n");
  printf("  plow launches the interpreter with cuLaunchCooperativeKernel, grid = occ x %d SMs,\n"
         "  and hard-gates grid == packet n_cu. A cluster dim must DIVIDE that grid.\n", SM);
  for(int occ : {1,2,3}){
    int grid = occ*SM;
    printf("  grid=%-4d (occ=%d/SM):", grid, occ);
    for(int C : {2,4,8,16}) printf("  C=%d %s", C, (grid%C==0)?"divides":"NO");
    printf("\n");
  }
  {
    // Does cudaLaunchKernelEx accept COOPERATIVE + CLUSTER_DIMENSION together?
    for(int C : {2,4,8}){
      if(C>maxNonPortable) continue;
      int grid = SM; while(grid%C) grid--;          // largest multiple of C <= SM
      cudaLaunchConfig_t cfg{}; cudaLaunchAttribute at[2]{};
      cfg.gridDim=dim3(grid,1,1); cfg.blockDim=dim3(64,1,1);
      at[0].id=cudaLaunchAttributeClusterDimension; at[0].val.clusterDim={(unsigned)C,1,1};
      at[1].id=cudaLaunchAttributeCooperative;      at[1].val.cooperative=1;
      cfg.attrs=at; cfg.numAttrs=2;
      cudaGetLastError();
      cudaError_t e = cudaLaunchKernelEx(&cfg,k_cluster_probe,dI);
      if(e==cudaSuccess) e=cudaDeviceSynchronize();
      printf("  coop+cluster C=%-2d grid=%-4d : %s\n",C,grid,
             e==cudaSuccess?"ACCEPTED":cudaGetErrorString(e));
      cudaGetLastError();
    }
    // and the exact grid plow would use (SM count, not rounded)
    for(int C : {8,16}){
      if(C>maxNonPortable) continue;
      cudaLaunchConfig_t cfg{}; cudaLaunchAttribute at[2]{};
      cfg.gridDim=dim3(SM,1,1); cfg.blockDim=dim3(64,1,1);
      at[0].id=cudaLaunchAttributeClusterDimension; at[0].val.clusterDim={(unsigned)C,1,1};
      at[1].id=cudaLaunchAttributeCooperative;      at[1].val.cooperative=1;
      cfg.attrs=at; cfg.numAttrs=2;
      cudaGetLastError();
      cudaError_t e = cudaLaunchKernelEx(&cfg,k_cluster_probe,dI);
      if(e==cudaSuccess) e=cudaDeviceSynchronize();
      printf("  coop+cluster C=%-2d grid=%-4d (== SM count, NOT a multiple of C) : %s\n",
             C,SM,e==cudaSuccess?"ACCEPTED":cudaGetErrorString(e));
      cudaGetLastError();
    }
  }

  printf("\n# measured HBM ceiling was %.0f GB/s — any 'issued' figure above it was served\n"
         "# by L2/DSMEM, not DRAM.\n", ceiling);
  return 0;
}

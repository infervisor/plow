// Hopper (sm_90a) thread-block CLUSTER probe #2 — the three cases the earlier
// refutation (hopper_cluster_multicast.cu) left OPEN. That probe proved clusters/
// DSM/multicast WORK on sm_90a and that multicast does NOT pay for L2-resident
// weight broadcast (0.78-0.94x). It did NOT cover:
//
//   P1  cluster-COOPERATIVE claim from a global queue: can C blocks claim ONE
//       op-entry together (rank0 atomicAdd + DSM broadcast + cluster.sync) at
//       full grid WITHOUT the deadlock that killed clusters in per-block-stream
//       mode? What is the per-claim TAX vs plow's plain single-block atomicAdd
//       (interp_sm120.cu:1293-1340)?
//   P2  TMA multicast where the WORKING SET and the per-CTA duplicate's reuse
//       distance are swept from L2-resident to >>L2 — to locate the L2 crossover
//       (if any) at which one HBM read shared to C CTAs finally beats C L2 hits.
//   P3  DSM cross-cluster REDUCTION vs the global/L2 round-trip (genuinely new;
//       the earlier probe only checked DSM correctness, never a reduction win):
//         (a) split-K GEMM partial combine (M=64,N=256,K=16384, split 4)
//         (b) flash split-KV merge / online-softmax rescale (Tkv=131072, split 4-8)
//
// BUILD (sm_90a-only opcodes need the -gencode form; -arch=sm_90a is rejected):
//   LD_LIBRARY_PATH=/usr/local/cuda/compat \
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 \
//        runtime/nvidia/experiments/cluster_gq_probe.cu \
//        -lcuda -lnvidia-ml -o /root/.claude/jobs/c92f0b7e/tmp/build/cgq
//
// RUN (serialized + 570 compat libs):
//   gpulease h100-cluster env LD_LIBRARY_PATH=/usr/local/cuda/compat \
//        /root/.claude/jobs/c92f0b7e/tmp/build/cgq
//
// BENCH DISCIPLINE (H100 NVL is 310 W-capped, DVFS-collapses 1785->~700 MHz in
// ~2 s of sustained load): ~2 ms bursts, 25 ms idle gaps, round-robin, min-of-12,
// SM clock reported per burst via NVML. Working sets exceed L2 (60 MB) by
// replication + cycling so cold-HBM is forced. Real sustained HBM ~2.2-2.4 TB/s.
//
// SASS witnesses to cite (dump: cuobjdump -sass <cubin>):
//   cluster.sync  -> UCGABAR_ARV/UCGABAR_WAIT (+CGAERRBAR, MEMBAR.ALL.GPU)
//   mapa.shared::cluster -> MAPA / SR_CgaCtaId address fold
//   ld.shared::cluster   -> LD.E through the distributed-shared aperture
//   cp.async.bulk.tensor...multicast -> UTMALDG.2D.MULTICAST [UR],[UR],UR
//
// ===========================================================================
// RESULTS  (H100 NVL, 132 SMs, L2=60 MB, CUDA 13.0, driver 570 compat, 1785 MHz,
//           HBM ceiling 3.70 TB/s short-burst / ~2.3-2.5 TB/s sustained)
// Run in PIECES: `cgq p1` (correctness+viability), `cgq p1tax` (clean tax, own
// fresh process), `cgq p2`, `cgq p2prime` (the deciding wgmma test), `cgq p3a`,
// `cgq p3b`. P1's spin-claim FAULTS and corrupts the context, so it cannot share a
// process with the rest; the others are each robust and can run standalone.
//
// MECHANISMS ARE REAL (SASS, cuobjdump -sass):
//   cluster.sync -> UCGABAR_ARV / UCGABAR_WAIT + MEMBAR.ALL.GPU + CGAERRBAR +
//                   CCTL.IVALL  (an expensive barrier sequence)
//   claim        -> ATOMG.E.ADD.STRONG.GPU (global cursor)
//   DSM broadcast/reduce -> mapa folds into the LDS address (SR_CgaCtaId) +
//                   ld.shared::cluster reads through the distributed aperture
//   multicast    -> UTMALDG.2D.MULTICAST [UR],[UR],UR  vs plain UTMALDG.2D
//
// P1  cluster-cooperative claim from a global queue
//   CORRECTNESS (single claim, full grid 264): PASS at C=2 (132 clusters) and
//     C=4 (66 clusters) — no deadlock, all C ranks read the IDENTICAL claimed
//     index via DSM, and claims cover [0,nClusters) exactly once.
//   TAX (native, cold): single-block claim 349 ns/claim (264-way atomic
//     contention); cluster C=2 claim 1602 ns/claim -> +1254 ns, 4.60x. The
//     cluster does HALF the atomics (1 per 2 blocks) yet is 4.6x SLOWER because
//     it pays 2x cluster.sync (UCGABAR+MEMBAR.ALL.GPU). As % of an op the tax is
//     >100% of a ~1 us op and ~7% of a ~19 us op — only amortized by very coarse ops.
//   VIABILITY: *** NOT VIABLE — DEADLOCK/LIVELOCK-PRONE. *** A single isolated
//     cluster claim works, but SUSTAINED claiming faults: C=2 completes one
//     launch then "unspecified launch failure"; C=4's looping claim faults on the
//     first sustained launch (its single-claim correctness still PASSes). It only
//     runs to completion under fully-serialized compute-sanitizer (memcheck: 0
//     errors — so it is NOT a memory bug; it is a forward-progress failure of the
//     cluster.sync spin under the 310 W DVFS collapse). This re-introduces exactly
//     the deadlock class hopper_cluster_multicast.cu warned about.
//
// P2  TMA multicast, working-set sweep 30 MB (L2-resident) -> 1440 MB (24x L2)
//   multicast NEVER wins: mc/per-CTA = 0.78-0.98x across the ENTIRE sweep; NO
//   crossover. per-CTA unique GB/s does not collapse crossing the L2 boundary
//   (~1100-1300 GB/s C=2, flat), proving the C duplicate reads are ALWAYS L2 hits
//   — co-resident cluster CTAs read the shared tile simultaneously (reuse
//   distance ~0), so total working-set size is irrelevant. Extends the earlier
//   L2-resident refutation across the full off-L2 range.
//
// P3a split-K GEMM combine (M=64,N=256,K=16384, split 4)   correctness relL2=1.7e-3
//   DSM reduce LOSES. Isolated reduce (partials >L2): DSM/global = 1.15x (C=2),
//   1.33x (C=4) SLOWER. Global reduce hits 1.1-1.6 TB/s (clean coalesced HBM);
//   DSM only 0.40-0.60 TB/s (cluster.sync-bound). End-to-end GEMM compute (2.5 ms)
//   dwarfs the combine (DSM +0.34 ms vs global +0.04 ms) — path choice 2nd-order.
//
// P3b flash split-KV merge (Tkv=131072, split 4/8)   correctness relL2=1.5-1.7e-3
//   DSM merge LOSES decisively and worsens with more splits:
//     C=4  DSM 0.0925 ms (275 GB/s)  vs global 0.0335 ms (2285 GB/s) -> 2.76x
//     C=8  DSM 0.1956 ms (130 GB/s)  vs global 0.0585 ms (2476 GB/s) -> 3.35x
//   The global d_flash_merge is a coalesced HBM-bandwidth kernel at ~peak
//   sustained (2.3-2.5 TB/s); DSM is throttled to 130-275 GB/s by cluster.sync +
//   per-element ld.shared::cluster aperture loads. HBM was never the bottleneck
//   for the global path, so "saving" HBM buys nothing while the barrier costs all.
//
// P2' THE DECIDING TEST — wgmma bf16 GEMM, cluster WEIGHT multicast vs per-CTA
//     (fast.cu's regime: TMA multicast of the shared operand into a wgmma pipeline;
//      static launch-time cluster, M-split so C CTAs share the B weight panel).
//   MECHANISM: runs to completion at full grid with NO P1 fault (bounded tile loop;
//     cluster.sync only at prologue/teardown, SASS-confirmed 3x UCGABAR outside the
//     k-loop, 0 in the mainloop) — so P1's 4.6x tax / deadlock does NOT apply to a
//     launch-time cluster GEMM. Correctness relL2 ~4e-6..8e-6 both variants.
//   SASS: k_cgemm<true> = 2x UTMALDG.2D.MULTICAST (B) + 2x UTMALDG.2D (A);
//         k_cgemm<false> = 4x UTMALDG.2D, ZERO UCGABAR. Both HGMMA.64x128x16.
//   RESULT — multicast is MARGINAL and SIGN-FLIPS with arithmetic intensity:
//     BM=64  (occ 2, load-heavier): mc/perCTA = 1.01-1.04x  -> multicast WINS ~1-4%
//     BM=128 (occ 1, compute-heavier): mc/perCTA = 0.97-0.99x -> multicast LOSES 1-3%
//     (M,N,K in {4096^3, 2048/8192/8192, 8192/4096/4096}; C=2 and C=4; 1785 MHz.)
//   Both configs reach only ~30% of the wgmma ceiling (uniform single-warpgroup, not
//     warp-specialized), so this does NOT reach fast.cu's true ~70%-peak regime. What
//     it DOES establish, decisively, over the achievable range: the multicast benefit
//     SHRINKS toward zero and turns NEGATIVE as arithmetic intensity rises, because
//     the shared-operand load is not the bottleneck and multicast adds cross-CTA
//     mbarrier round-trips. Best case observed = +4%; it is break-even at best for
//     the larger tiles plow would actually use. fast.cu's multicast is thus a small
//     margin optimization inside an already-tuned kernel, not a standalone win.
//   OTHER DATATYPES (fp8 e4m3, m64n128k32): NOT measured, but the trend predicts it
//     LOSES harder — fp8 halves the shared-operand bytes (half the multicast saving)
//     AND doubles tensor throughput (~4x arithmetic intensity vs bf16), pushing
//     further into the compute-bound region where multicast already turns negative.
//   PLOW: even the +4% best case needs the separate prefill-segment object to launch
//     with a cluster dim, for a low-single-digit and intensity-fragile gain -> the
//     single-CTA-tile assumption for prefill holds. (A last-mile warp-specialized
//     >60%-peak A/B is the only thing left unmeasured; the trend argues it won't pay.)
//
// VERDICT: all three still do not pay on H100. P1 is worse than break-even — it
//   is deadlock-prone in the persistent-grid regime. DSM reductions (P3) lose
//   because cluster.sync (UCGABAR+MEMBAR.ALL.GPU) is dramatically more expensive
//   than a coalesced HBM read, and HBM is not the scarce resource for these
//   small reductions. DO NOT PURSUE clusters for plow's scheduler or reductions.
// ===========================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <nvml.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <unistd.h>

namespace cg = cooperative_groups;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CKD(x) do{ CUresult e=(x); if(e!=CUDA_SUCCESS){ const char*s; cuGetErrorString(e,&s); \
  printf("CU ERR %s @%d: %s\n",#x,__LINE__,s); exit(1);} }while(0)

// ------------------------------------------------------- device helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p){
  return (uint32_t)__cvta_generic_to_shared(p);
}
// local shared addr -> same offset inside CTA `rank` of this cluster (DSM aperture)
__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, int rank){
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ uint32_t ld_smem_cluster_u32(uint32_t addr){
  uint32_t v; asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(addr)); return v;
}
__device__ __forceinline__ float ld_smem_cluster_f32(uint32_t addr){
  float v; asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(addr)); return v;
}
// ------------------------------------------------ mbarrier / TMA (for P2)
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt){
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt):"memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, int bytes){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(smem_u32(b)), "r"(bytes):"memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int phase){
  asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;"
               " @!p bra W%=; }" :: "r"(smem_u32(b)), "r"(phase):"memory");
}
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t addr){
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(addr):"memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst,const CUtensorMap* map,int c0,int c1,uint32_t bar){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst),"l"(map),"r"(c0),"r"(c1),"r"(bar):"memory");
}
__device__ __forceinline__ void tma2d_mc(uint32_t dst,const CUtensorMap* map,int c0,int c1,
                                          uint32_t bar,uint16_t mask){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               ".multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
               :: "r"(dst),"l"(map),"r"(c0),"r"(c1),"r"(bar),"h"(mask):"memory");
}

// ======================================================= HBM ceiling probe
__global__ void k_stream(const float4* __restrict__ p, size_t n4, float* out){
  float4 a=make_float4(0,0,0,0);
  for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n4;i+=(size_t)gridDim.x*blockDim.x){
    float4 v=p[i]; a.x+=v.x; a.y+=v.y; a.z+=v.z; a.w+=v.w;
  }
  if(a.x==12345.f) out[0]=a.x+a.y+a.z+a.w;
}

// =====================================================================
// P1. cluster-cooperative CLAIM from a global queue.
// Both kernels run the IDENTICAL claim protocol shape as interp_sm120.cu, then
// a stub "op" of `work` register-FMA iterations. Only WHO claims changes.
//
// SINGLE (plow today, interp_sm120.cu:1319-1327):
//   __syncthreads; tid0: gq=atomicAdd(cursor,1); __syncthreads; ix=gq; if end break; op
// CLUSTER (proposed):
//   cluster.sync; rank0+tid0: gq=atomicAdd(cursor,1)  (into rank0's smem)
//   cluster.sync;  every block's tid0 reads rank0's gq via DSM (mapa+ld.shared::cluster)
//   __syncthreads; ix=gq; if end break; all C blocks co-run the op
// The last (over-limit) claim is seen identically by all C ranks -> they break
// together AFTER both cluster.syncs -> no barrier mismatch -> no deadlock.
// =====================================================================
__device__ __forceinline__ float stub_op(unsigned ix, int work){
  float a = 1.0f + (float)(ix & 7);
  #pragma unroll 4
  for(int i=0;i<work;i++) a = fmaf(a, 0.9999999f, 1e-7f);
  return a;
}

__global__ __launch_bounds__(256)
void k_claim_single(uint32_t* cursor, unsigned total, int work, float* sink){
  __shared__ unsigned gq;
  float acc=0.f;
  for(;;){
    __syncthreads();
    if(threadIdx.x==0) gq = atomicAdd(cursor, 1u);
    __syncthreads();
    unsigned ix = gq;
    if(ix >= total) break;
    acc += stub_op(ix, work);
  }
  if(acc==1.2345e-30f) sink[blockIdx.x]=acc;  // DCE guard
}

__global__ __launch_bounds__(256)
void k_claim_cluster(uint32_t* cursor, unsigned total, int work, float* sink){
  __shared__ unsigned gq;       // only rank 0's copy is authoritative
  __shared__ unsigned ix_bc;    // per-block broadcast of the claimed index
  cg::cluster_group cl = cg::this_cluster();
  const int rank = (int)cl.block_rank();
  const uint32_t rank0_gq = mapa_shared(smem_u32(&gq), 0);
  float acc=0.f;
  for(;;){
    cl.sync();                                   // whole cluster retires prev op
    if(rank==0 && threadIdx.x==0) gq = atomicAdd(cursor, 1u);
    cl.sync();                                   // rank0's write visible cluster-wide
    if(threadIdx.x==0) ix_bc = ld_smem_cluster_u32(rank0_gq);  // DSM broadcast
    __syncthreads();
    unsigned ix = ix_bc;
    if(ix >= total) break;
    acc += stub_op(ix, work);                    // all C blocks co-work the op
  }
  if(acc==1.2345e-30f) sink[blockIdx.x]=acc;
}

// Fixed-iteration variant (no cursor exhaustion): isolates whether the claim
// protocol (cluster.sync + rank0 atomicAdd + DSM broadcast) in a loop is stable
// at a given C, independent of the exhaustion boundary logic.
__global__ __launch_bounds__(256)
void k_cluster_spin(uint32_t* cursor, int iters, float* sink){
  __shared__ unsigned gq; __shared__ unsigned ix_bc;
  cg::cluster_group cl=cg::this_cluster();
  const int rank=(int)cl.block_rank();
  const uint32_t rank0_gq=mapa_shared(smem_u32(&gq),0);
  float acc=0.f;
  for(int it=0; it<iters; it++){
    cl.sync();
    if(rank==0 && threadIdx.x==0) gq=atomicAdd(cursor,1u);
    cl.sync();
    if(threadIdx.x==0) ix_bc=ld_smem_cluster_u32(rank0_gq);
    __syncthreads();
    acc += stub_op(ix_bc,0);
  }
  if(acc==1.2345e-30f) sink[blockIdx.x]=acc;
}

// correctness: every block records the index it claimed, ONE claim per worker.
__global__ void k_claim_single_ck(uint32_t* cursor, unsigned total, uint32_t* got){
  __shared__ unsigned gq;
  __syncthreads();
  if(threadIdx.x==0) gq = atomicAdd(cursor,1u);
  __syncthreads();
  if(threadIdx.x==0) got[blockIdx.x] = gq;
}
__global__ void k_claim_cluster_ck(uint32_t* cursor, unsigned total, uint32_t* got){
  __shared__ unsigned gq;
  cg::cluster_group cl = cg::this_cluster();
  const int rank=(int)cl.block_rank();
  const uint32_t rank0_gq = mapa_shared(smem_u32(&gq),0);
  cl.sync();
  if(rank==0 && threadIdx.x==0) gq = atomicAdd(cursor,1u);
  cl.sync();
  if(threadIdx.x==0) got[blockIdx.x] = ld_smem_cluster_u32(rank0_gq);
  cl.sync();
}

// =====================================================================
// P2. TMA weight-broadcast multicast across a WORKING-SET SWEEP.
// Lifted from hopper_cluster_multicast.cu k_bcast (proven correct). A cluster of
// C CTAs shares the SAME B weight tiles; rank r owns a different A row-block.
// Diagnostic: per-CTA UNIQUE GB/s vs the HBM ceiling.
//   unique GB/s ~ ceiling      -> the C duplicate reads are L2 hits (mc can't help)
//   unique GB/s ~ ceiling / C  -> the duplicates MISS L2 (mc SHOULD win ~Cx)
// Sweep replicas R so the working set spans L2-resident (<60 MB) to >>L2.
// =====================================================================
static const int P2_BN=128, P2_BK=64, P2_NSTAGE=4, P2_THREADS=256, P2_BM=2;
static const int P2_TILE_B = P2_BN*P2_BK*2;   // 16384 B

template<bool MC,bool LITE>
__global__ __launch_bounds__(P2_THREADS)
void k_bcast(const __grid_constant__ CUtensorMap map, const __nv_bfloat16* __restrict__ A,
             float* __restrict__ out, int K, int nNTiles, int nItems, int nClusters, int C){
  extern __shared__ char smem_raw[];
  uint32_t base=(smem_u32(smem_raw)+127u)&~127u;
  const uint32_t tile0=base;
  float* As=(float*)__cvta_shared_to_generic(base + P2_NSTAGE*P2_TILE_B);
  __shared__ __align__(8) uint64_t full[P2_NSTAGE];
  __shared__ __align__(8) uint64_t empty[P2_NSTAGE];
  cg::cluster_group cl=cg::this_cluster();
  const int rank = MC ? (int)cl.block_rank() : (int)(blockIdx.x%(unsigned)C);
  const int clusterId = (int)(blockIdx.x/(unsigned)C);
  const int tid=threadIdx.x;
  for(int i=tid;i<P2_BM*K;i+=P2_THREADS){ int m=i/K,k=i%K; As[i]=__bfloat162float(A[(size_t)(rank*P2_BM+m)*K+k]); }
  if(tid==0){
    #pragma unroll
    for(int s=0;s<P2_NSTAGE;s++){ mbar_init(&full[s],1); mbar_init(&empty[s], MC?C:1); }
  }
  __syncthreads();
  if(MC) cl.sync();
  const int nk=K/P2_BK;
  const uint16_t mask=(uint16_t)((1u<<C)-1u);
  const int myItems=(nItems-clusterId+nClusters-1)/nClusters;
  const int T=myItems*nk;
  if(T<=0) return;
  uint32_t empty_addr[P2_NSTAGE];
  #pragma unroll
  for(int s=0;s<P2_NSTAGE;s++) empty_addr[s]= MC ? mapa_shared(smem_u32(&empty[s]),0):smem_u32(&empty[s]);
  auto coords=[&](int t,int&c0,int&c1){
    int item=clusterId+(t/nk)*nClusters; int rep=item/nNTiles, nt=item%nNTiles;
    c0=(t%nk)*P2_BK; c1=rep*(nNTiles*P2_BN)+nt*P2_BN;
  };
  if(tid==0){
    #pragma unroll
    for(int s=0;s<P2_NSTAGE;s++) if(s<T){ mbar_expect(&full[s],P2_TILE_B); mbar_arrive_cluster(empty_addr[s]); }
  }
  if(MC) cl.sync(); else __syncthreads();
  if(tid==0 && (!MC||rank==0)){
    for(int s=0;s<P2_NSTAGE && s<T;s++){ int c0,c1; coords(s,c0,c1);
      if(MC) tma2d_mc(tile0+s*P2_TILE_B,&map,c0,c1,smem_u32(&full[s]),mask);
      else   tma2d   (tile0+s*P2_TILE_B,&map,c0,c1,smem_u32(&full[s])); }
  }
  float acc[P2_BM];
  #pragma unroll
  for(int m=0;m<P2_BM;m++) acc[m]=0.f;
  for(int t=0;t<T;t++){
    const int s=t%P2_NSTAGE, j=t/P2_NSTAGE;
    if(tid==0) mbar_wait(&full[s], j&1);
    __syncthreads();
    const __nv_bfloat16* tb=(const __nv_bfloat16*)__cvta_shared_to_generic(tile0+s*P2_TILE_B);
    const int kbase=(t%nk)*P2_BK;
    if(LITE){
      for(int e=tid*8;e<P2_BN*P2_BK;e+=P2_THREADS*8){ int4 v=*(const int4*)(tb+e); acc[0]+=(float)(v.x^v.y^v.z^v.w); }
    } else
    for(int e=tid*8;e<P2_BN*P2_BK;e+=P2_THREADS*8){
      int4 v=*(const int4*)(tb+e); const __nv_bfloat16* b8=(const __nv_bfloat16*)&v;
      const float* ap=As+kbase+(e%P2_BK); float av[P2_BM][8];
      #pragma unroll
      for(int m=0;m<P2_BM;m++){ *(float4*)&av[m][0]=*(const float4*)(ap+m*K); *(float4*)&av[m][4]=*(const float4*)(ap+m*K+4); }
      #pragma unroll
      for(int q=0;q<8;q++){ float bv=__bfloat162float(b8[q]);
        #pragma unroll
        for(int m=0;m<P2_BM;m++) acc[m]=fmaf(bv,av[m][q],acc[m]); }
    }
    __syncthreads();
    if(tid==0){ if(t+P2_NSTAGE<T) mbar_expect(&full[s],P2_TILE_B); mbar_arrive_cluster(empty_addr[s]); }
    if(t+P2_NSTAGE<T && tid==0 && (!MC||rank==0)){
      mbar_wait(&empty[s],(j+1)&1); int c0,c1; coords(t+P2_NSTAGE,c0,c1);
      if(MC) tma2d_mc(tile0+s*P2_TILE_B,&map,c0,c1,smem_u32(&full[s]),mask);
      else   tma2d   (tile0+s*P2_TILE_B,&map,c0,c1,smem_u32(&full[s])); }
  }
  float r=0.f;
  #pragma unroll
  for(int m=0;m<P2_BM;m++) r+=acc[m];
  if(r==1.2345e-30f) out[blockIdx.x]=r;
  if(MC) cl.sync();
}

// =====================================================================
// P2'. THE DECIDING TEST: compute-bound wgmma bf16 GEMM with a STATIC cluster
// dim, sharing the WEIGHT (B) operand across an M-split cluster (fast.cu's regime).
//   A/B: (a) each CTA loads its own B weight tiles via TMA;
//        (b) rank 0 multicasts the shared B tile to all C CTAs (UTMALDG.2D.MULTICAST).
// A (activations) is per-CTA in both. The wgmma SS-descriptor + 128B-swizzle recipe
// is the validated one from wgmma_bf16_probe.cu / tma_ws_gemm_bf16.cu. cluster.sync
// appears ONCE at prologue (make barriers live before the first multicast) — NEVER
// in the k-loop; the loop recycles buffers via mbarrier arrive (empty credits via
// DSM), exactly as a launch-time cluster should (contrast P1's per-claim spin).
// =====================================================================
static const int CG_MS=2;                    // m64 wgmma slabs per stage -> BM=128
static const int CG_BM=64*CG_MS, CG_BN=128, CG_BK=64, CG_KSUB=CG_BK/16;
static const int CG_NACC=64, CG_NS=4, CG_WG=128;
static const int CG_ATILE=CG_BM*CG_BK, CG_BTILE=CG_BN*CG_BK;         // bf16 elems/stage
static const int CG_ATB=CG_ATILE*2, CG_BTB=CG_BTILE*2;              // bytes/stage
static const uint64_t CG_LBO=16, CG_SBO=1024;

__device__ __forceinline__ uint64_t cg_desc(uint32_t a32){
  uint64_t a=(uint64_t)a32, d=0;
  d |= (a & 0x3FFFFull)>>4;
  d |= ((CG_LBO & 0x3FFFFull)>>4)<<16;
  d |= ((CG_SBO & 0x3FFFFull)>>4)<<32;
  d |= (uint64_t)1<<62;   // 128B swizzle
  return d;
}
__device__ __forceinline__ void cg_fence(){ asm volatile("wgmma.fence.sync.aligned;\n"::); }
__device__ __forceinline__ void cg_commit(){ asm volatile("wgmma.commit_group.sync.aligned;\n"::); }
template<int N> __device__ __forceinline__ void cg_wgwait(){ asm volatile("wgmma.wait_group.sync.aligned %0;\n"::"n"(N)); }
__device__ __forceinline__ void cg_wgmma(float(&d)[CG_NACC], uint64_t da, uint64_t db, int sd){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
    "%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,"
    "%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
      "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
      "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
      "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
      "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
      "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
      "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
      "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
    : "l"(da),"l"(db),"r"(sd));
}
__device__ __forceinline__ uint32_t cg_align1k(uint32_t a){ return (a+1023u)&~1023u; }

template<bool MC>
__global__ __launch_bounds__(CG_WG)
void k_cgemm(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
             float* __restrict__ Cout, int M, int N, int K, int nClusters, int C){
  extern __shared__ char cs_raw[];
  uint32_t abase=cg_align1k(smem_u32(cs_raw));
  const uint32_t As0=abase, Bs0=abase + CG_NS*CG_ATB;   // both 1024B-aligned (ATB,BTB mult of 1024)
  __shared__ __align__(8) uint64_t fA[CG_NS], fB[CG_NS], eA[CG_NS], eB[CG_NS];
  cg::cluster_group cl=cg::this_cluster();
  const int rank = MC ? (int)cl.block_rank() : (int)(blockIdx.x%(unsigned)C);
  const int clusterId=(int)(blockIdx.x/(unsigned)C);
  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;

  if(tid==0){
    #pragma unroll
    for(int s=0;s<CG_NS;s++){ mbar_init(&fA[s],1); mbar_init(&fB[s],1);
      mbar_init(&eA[s],1); mbar_init(&eB[s], MC?C:1); }
  }
  __syncthreads();
  if(MC) cl.sync();

  const int mtiles=M/CG_BM, ntiles=N/CG_BN, ksteps=K/CG_BK;
  const int superM=mtiles/C;                        // C consecutive m-tiles per cluster super-row
  const int totalSuper=superM*ntiles;
  const uint16_t mask=(uint16_t)((1u<<C)-1u);
  uint32_t eB_addr[CG_NS];
  #pragma unroll
  for(int s=0;s<CG_NS;s++) eB_addr[s]= MC? mapa_shared(smem_u32(&eB[s]),0): smem_u32(&eB[s]);

  const int myS=(totalSuper - clusterId + nClusters -1)/nClusters;
  const int T=myS*ksteps;
  if(T<=0) return;
  auto coord=[&](int t,int&tm,int&tn,int&koff){
    int ul=t/ksteps, ks=t%ksteps, u=clusterId+ul*nClusters;
    int sm=u/ntiles, tnt=u%ntiles;
    tm=(sm*C+rank)*CG_BM; tn=tnt*CG_BN; koff=ks*CG_BK;
  };

  // prologue: expects + empty credits, then issue NS stages
  if(tid==0){
    #pragma unroll
    for(int s=0;s<CG_NS;s++) if(s<T){
      mbar_expect(&fA[s],CG_ATB); mbar_expect(&fB[s],CG_BTB);
      mbar_arrive_cluster(smem_u32(&eA[s]));            // A empty (local) primed
      mbar_arrive_cluster(eB_addr[s]);                  // B empty (rank0 for MC) primed
    }
  }
  if(MC) cl.sync(); else __syncthreads();
  if(tid==0){
    for(int s=0;s<CG_NS && s<T;s++){ int tm,tn,ko; coord(s,tm,tn,ko);
      tma2d(As0+s*CG_ATB, &mapA, ko, tm, smem_u32(&fA[s]));            // A per-CTA
      if(!MC)          tma2d   (Bs0+s*CG_BTB, &mapB, ko, tn, smem_u32(&fB[s]));
      else if(rank==0) tma2d_mc(Bs0+s*CG_BTB, &mapB, ko, tn, smem_u32(&fB[s]), mask);
    }
  }

  float acc[CG_MS][CG_NACC];
  for(int t=0;t<T;t++){
    const int s=t%CG_NS, j=t/CG_NS, ks=t%ksteps;
    if(tid==0){ mbar_wait(&fA[s], j&1); mbar_wait(&fB[s], j&1); }
    __syncthreads();
    // wgmma consume: D[m,n128] += A[m,k]*B[n128,k], MS m64-slabs x KSUB k16 substeps
    cg_fence();
    #pragma unroll
    for(int sub=0;sub<CG_KSUB;sub++){
      int sd=(ks==0 && sub==0)?0:1;
      const uint64_t db=cg_desc(Bs0+s*CG_BTB+sub*32);
      #pragma unroll
      for(int m=0;m<CG_MS;m++)
        cg_wgmma(acc[m], cg_desc(As0+s*CG_ATB + m*64*CG_BK*2 + sub*32), db, sd);
    }
    cg_commit(); cg_wgwait<0>();
    __syncthreads();
    // credit + refill NS ahead
    if(tid==0){
      if(t+CG_NS<T){ mbar_expect(&fA[s],CG_ATB); mbar_expect(&fB[s],CG_BTB); }
      mbar_arrive_cluster(smem_u32(&eA[s]));
      mbar_arrive_cluster(eB_addr[s]);
    }
    if(t+CG_NS<T && tid==0){
      int tm,tn,ko; coord(t+CG_NS,tm,tn,ko);
      mbar_wait(&eA[s],(j+1)&1);
      tma2d(As0+s*CG_ATB,&mapA,ko,tm,smem_u32(&fA[s]));
      if(!MC){ mbar_wait(&eB[s],(j+1)&1); tma2d(Bs0+s*CG_BTB,&mapB,ko,tn,smem_u32(&fB[s])); }
      else if(rank==0){ mbar_wait(&eB[s],(j+1)&1); tma2d_mc(Bs0+s*CG_BTB,&mapB,ko,tn,smem_u32(&fB[s]),mask); }
    }
    // flush at tile boundary (last k-step of this tile)
    if(ks==ksteps-1){
      int tm,tn,ko; coord(t,tm,tn,ko);
      const int cbase=tn+(lane&3)*2;
      #pragma unroll
      for(int m=0;m<CG_MS;m++){
        const int r0=tm+m*64+warp*16+(lane>>2), r1=r0+8;
        #pragma unroll
        for(int q=0;q<CG_BN/8;q++){ int c=cbase+q*8;
          if(r0<M){ if(c<N)Cout[(size_t)r0*N+c]=acc[m][4*q]; if(c+1<N)Cout[(size_t)r0*N+c+1]=acc[m][4*q+1]; }
          if(r1<M){ if(c<N)Cout[(size_t)r1*N+c]=acc[m][4*q+2]; if(c+1<N)Cout[(size_t)r1*N+c+1]=acc[m][4*q+3]; }
        }
      }
    }
  }
  if(MC) cl.sync();
}

// wgmma back-to-back rate ceiling (no memory traffic): sustained TF/s ceiling.
__global__ __launch_bounds__(CG_WG) void k_wgmma_rate(float* sink,int iters){
  __shared__ __align__(1024) __nv_bfloat16 buf[CG_BM*CG_BK + CG_BN*CG_BK];
  const int tid=threadIdx.x;
  for(int i=tid;i<CG_BM*CG_BK+CG_BN*CG_BK;i+=CG_WG) buf[i]=(__nv_bfloat16)1;
  __syncthreads();
  float acc[CG_MS][CG_NACC];
  #pragma unroll
  for(int m=0;m<CG_MS;m++) for(int i=0;i<CG_NACC;i++) acc[m][i]=0.f;
  uint64_t db=cg_desc(smem_u32(buf+CG_BM*CG_BK));
  for(int it=0;it<iters;it++){
    cg_fence();
    #pragma unroll
    for(int sub=0;sub<CG_KSUB;sub++)
      #pragma unroll
      for(int m=0;m<CG_MS;m++) cg_wgmma(acc[m], cg_desc(smem_u32(buf)+(uint32_t)(m*64*CG_BK*2)), db, 1);
    cg_commit(); cg_wgwait<0>();
  }
  float r=0; for(int m=0;m<CG_MS;m++) for(int i=0;i<CG_NACC;i++) r+=acc[m][i];
  if(r==1.2345e-30f) sink[blockIdx.x]=r;
}

// =====================================================================
// P3(a). split-K GEMM partial-combine: DSM reduce vs global round-trip.
// One problem = C[M=64][N=256] = sum over C K-slices of A[64][K] * B[K][256].
// A batch of B independent problems fills the grid and blows past L2.
// =====================================================================
static const int G_M=64, G_N=256, G_BK=32;   // output tile + K-staging chunk
// compute this rank's K-slice partial into partial_smem[G_M*G_N] (col-major-in-n: [m*G_N+n])
__device__ void gemm_partial(const __nv_bfloat16* A, const __nv_bfloat16* B, int K,
                             int klo, int klen, float* partial_smem, float* As){
  const int tid=threadIdx.x;            // 256 threads, thread t owns column n=t, all 64 rows
  float acc[G_M];
  #pragma unroll
  for(int m=0;m<G_M;m++) acc[m]=0.f;
  for(int kk=0; kk<klen; kk+=G_BK){
    // stage A[0..63][klo+kk .. +G_BK] -> As[G_M*G_BK]
    for(int i=tid;i<G_M*G_BK;i+=256){ int m=i/G_BK, k=i%G_BK; As[i]=__bfloat162float(A[(size_t)m*K+(klo+kk+k)]); }
    __syncthreads();
    #pragma unroll
    for(int k=0;k<G_BK;k++){
      float b=__bfloat162float(B[(size_t)(klo+kk+k)*G_N + tid]);   // coalesced across n
      #pragma unroll
      for(int m=0;m<G_M;m++) acc[m]=fmaf(As[m*G_BK+k], b, acc[m]);
    }
    __syncthreads();
  }
  #pragma unroll
  for(int m=0;m<G_M;m++) partial_smem[m*G_N+tid]=acc[m];
}

// DSM end-to-end: cluster of C CTAs, rank r computes K-slice partial, then all
// ranks reduce via DSM (rank r sums its N-column-slice across C peers) -> O.
__global__ __launch_bounds__(256)
void k_gemm_dsm(const __nv_bfloat16* Aall, const __nv_bfloat16* Ball, __nv_bfloat16* Oall,
                int K, int C){
  extern __shared__ float sh[];
  float* partial = sh;                 // [G_M*G_N]
  float* As = sh + G_M*G_N;            // [G_M*G_BK]
  cg::cluster_group cl=cg::this_cluster();
  const int rank=(int)cl.block_rank();
  const int prob=(int)(blockIdx.x / (unsigned)C);
  const __nv_bfloat16* A=Aall + (size_t)prob*G_M*K;
  const __nv_bfloat16* B=Ball + (size_t)prob*(size_t)K*G_N;
  const int klen=K/C, klo=rank*klen;
  gemm_partial(A,B,K,klo,klen,partial,As);
  cl.sync();
  // rank r owns columns [r*G_N/C, +G_N/C); sum across all C peers via DSM
  const int cols=G_N/C, c0=rank*cols;
  const int tid=threadIdx.x;
  const uint32_t local=smem_u32(partial);
  for(int i=tid;i<G_M*cols;i+=256){
    int m=i/cols, cc=i%cols, n=c0+cc;
    float s=0.f;
    for(int p=0;p<C;p++){ uint32_t a=mapa_shared(local + (uint32_t)((m*G_N+n)*4), p); s+=ld_smem_cluster_f32(a); }
    Oall[(size_t)prob*G_M*G_N + m*G_N+n]=__float2bfloat16(s);
  }
  cl.sync();
}

// Global path kernel 1: each of C CTAs computes its partial -> global scratch[prob][rank].
__global__ __launch_bounds__(256)
void k_gemm_part_write(const __nv_bfloat16* Aall, const __nv_bfloat16* Ball, float* scratch,
                       int K, int C){
  extern __shared__ float sh[];
  float* partial=sh; float* As=sh+G_M*G_N;
  const int rank=(int)(blockIdx.x % (unsigned)C);
  const int prob=(int)(blockIdx.x / (unsigned)C);
  const __nv_bfloat16* A=Aall+(size_t)prob*G_M*K;
  const __nv_bfloat16* B=Ball+(size_t)prob*(size_t)K*G_N;
  const int klen=K/C, klo=rank*klen;
  gemm_partial(A,B,K,klo,klen,partial,As);
  __syncthreads();
  float* dst=scratch + ((size_t)prob*C+rank)*G_M*G_N;
  for(int i=threadIdx.x;i<G_M*G_N;i+=256) dst[i]=partial[i];
}
// Global path kernel 2 (also the ISOLATED reduce): one CTA reads C partials -> O.
__global__ __launch_bounds__(256)
void k_reduce_global(const float* scratch, __nv_bfloat16* Oall, int C){
  const int prob=blockIdx.x;
  const float* src=scratch + (size_t)prob*C*G_M*G_N;
  for(int i=threadIdx.x;i<G_M*G_N;i+=256){
    float s=0.f;
    for(int p=0;p<C;p++) s+=src[(size_t)p*G_M*G_N + i];
    Oall[(size_t)prob*G_M*G_N + i]=__float2bfloat16(s);
  }
}
// ISOLATED DSM reduce: cluster C, rank r loads its OWN partial from global into
// smem (1/C of the HBM the global path reads), cluster.sync, DSM-sum -> O.
__global__ __launch_bounds__(256)
void k_reduce_dsm(const float* scratch, __nv_bfloat16* Oall, int C){
  extern __shared__ float sh[];        // [G_M*G_N]
  cg::cluster_group cl=cg::this_cluster();
  const int rank=(int)cl.block_rank();
  const int prob=(int)(blockIdx.x/(unsigned)C);
  const float* mine=scratch + ((size_t)prob*C+rank)*G_M*G_N;
  for(int i=threadIdx.x;i<G_M*G_N;i+=256) sh[i]=mine[i];  // own partial from HBM
  cl.sync();
  const int cols=G_N/C, c0=rank*cols;
  const uint32_t local=smem_u32(sh);
  for(int i=threadIdx.x;i<G_M*cols;i+=256){
    int m=i/cols, n=c0+i%cols; float s=0.f;
    for(int p=0;p<C;p++){ uint32_t a=mapa_shared(local+(uint32_t)((m*G_N+n)*4),p); s+=ld_smem_cluster_f32(a); }
    Oall[(size_t)prob*G_M*G_N+m*G_N+n]=__float2bfloat16(s);
  }
  cl.sync();
}

// =====================================================================
// P3(b). flash split-KV merge: DSM vs global d_flash_merge.
// Partials (O_i[BQ][HD] unnormalised, m_i[BQ], l_i[BQ]) are pre-generated for C
// splits per problem. The MERGE (online-softmax rescale) is what is under test.
//   final = (sum_i O_i*exp(m_i-M)) / (sum_i l_i*exp(m_i-M)),  M=max_i m_i
// Layout: Opart[prob][split][BQ][HD] f32 ; ml[prob][split][BQ][2] f32 (m,l)
// =====================================================================
static const int F_BQ=16, F_HD=128;
// GLOBAL merge (current plow d_flash_merge): one CTA/problem reads all C splits.
__global__ __launch_bounds__(F_HD)
void k_flash_merge_global(const float* Opart, const float* ml, __nv_bfloat16* O, int C){
  const int prob=blockIdx.x, tid=threadIdx.x;  // tid = head-dim lane
  for(int q=0;q<F_BQ;q++){
    float M=-1e30f;
    for(int p=0;p<C;p++) M=fmaxf(M, ml[(((size_t)prob*C+p)*F_BQ+q)*2+0]);
    float l=0.f, acc=0.f;
    for(int p=0;p<C;p++){
      float mi=ml[(((size_t)prob*C+p)*F_BQ+q)*2+0], li=ml[(((size_t)prob*C+p)*F_BQ+q)*2+1];
      float w=__expf(mi-M); l+=li*w;
      acc+=w*Opart[(((size_t)prob*C+p)*F_BQ+q)*F_HD + tid];
    }
    O[((size_t)prob*F_BQ+q)*F_HD+tid]=__float2bfloat16(acc/(l>0.f?l:1.f));
  }
}
// DSM merge: cluster C, rank r loads its OWN split's partial into smem (1/C of the
// HBM the global path reads), cluster.sync, merges rows assigned to it via DSM.
__global__ __launch_bounds__(F_HD)
void k_flash_merge_dsm(const float* Opart, const float* ml, __nv_bfloat16* O, int C){
  extern __shared__ float sh[];        // O_r[F_BQ*F_HD] then ml_r[F_BQ*2]
  float* Osm=sh; float* mlsm=sh+F_BQ*F_HD;
  cg::cluster_group cl=cg::this_cluster();
  const int rank=(int)cl.block_rank();
  const int prob=(int)(blockIdx.x/(unsigned)C);
  const int tid=threadIdx.x;
  for(int i=tid;i<F_BQ*F_HD;i+=F_HD) Osm[i]=Opart[(((size_t)prob*C+rank)*F_BQ)*F_HD + i];
  if(tid<F_BQ*2) mlsm[tid]=ml[(((size_t)prob*C+rank)*F_BQ)*2 + tid];
  cl.sync();
  const uint32_t Oloc=smem_u32(Osm), mloc=smem_u32(mlsm);
  // rank r merges rows q where q%C==rank
  for(int q=rank;q<F_BQ;q+=C){
    float Mx=-1e30f;
    for(int p=0;p<C;p++){ uint32_t a=mapa_shared(mloc+(uint32_t)((q*2)*4),p); Mx=fmaxf(Mx, ld_smem_cluster_f32(a)); }
    float l=0.f, acc=0.f;
    for(int p=0;p<C;p++){
      uint32_t am=mapa_shared(mloc+(uint32_t)((q*2)*4),p);
      uint32_t al=mapa_shared(mloc+(uint32_t)((q*2+1)*4),p);
      float mi=ld_smem_cluster_f32(am), li=ld_smem_cluster_f32(al);
      float w=__expf(mi-Mx); l+=li*w;
      uint32_t ao=mapa_shared(Oloc+(uint32_t)((q*F_HD+tid)*4),p);
      acc+=w*ld_smem_cluster_f32(ao);
    }
    O[((size_t)prob*F_BQ+q)*F_HD+tid]=__float2bfloat16(acc/(l>0.f?l:1.f));
  }
  cl.sync();
}

// ======================================================= NVML clock helper
static nvmlDevice_t g_nvml=nullptr;
static bool nvml_ok=false;
static unsigned sm_clock(){ unsigned c=0; if(nvml_ok) nvmlDeviceGetClockInfo(g_nvml,NVML_CLOCK_SM,&c); return c; }

// ======================================================= bench runner
// Round-robin, ~2 ms bursts, 25 ms gaps, min-of-12, mean SM clock per case.
struct Case { std::string name; std::function<void()> once; };
struct Res  { double ms; double clk; };
// force_iters>0 pins one-launch-per-burst (required for cluster kernels: rapid
// back-to-back cluster launches fault on the 570/CUDA13 driver here).
static std::vector<Res> bench_group(std::vector<Case>& cs, int rounds=12, double burst_s=0.002,
                                    int force_iters=0){
  const int n=cs.size();
  std::vector<int> iters(n,1);
  std::vector<Res> best(n,{1e18,0});
  std::vector<double> clk_sum(n,0); std::vector<int> clk_cnt(n,0);
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  // calibrate iters/case
  for(int i=0;i<n;i++){
    cs[i].once(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0)); cs[i].once(); CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms,e0,e1);
    iters[i]= force_iters>0 ? force_iters : std::max(1,(int)(burst_s*1000.0/std::max(1e-3f,ms)));
    if(iters[i]>100000) iters[i]=100000;
    usleep(25000);
  }
  for(int r=0;r<rounds;r++){
    for(int i=0;i<n;i++){
      CK(cudaEventRecord(e0));
      for(int k=0;k<iters[i];k++) cs[i].once();
      CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
      float ms; cudaEventElapsedTime(&ms,e0,e1);
      double per=ms/iters[i];
      if(per<best[i].ms) best[i].ms=per;
      unsigned c=sm_clock(); if(c){ clk_sum[i]+=c; clk_cnt[i]++; }
      usleep(25000);
    }
  }
  for(int i=0;i<n;i++) best[i].clk = clk_cnt[i]? clk_sum[i]/clk_cnt[i] : 0;
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return best;
}

static void clusterCfg(cudaLaunchConfig_t& cfg, cudaLaunchAttribute* at, dim3 grid, dim3 blk,
                       int C, size_t smem){
  cfg=cudaLaunchConfig_t{}; cfg.gridDim=grid; cfg.blockDim=blk; cfg.dynamicSmemBytes=smem;
  at[0].id=cudaLaunchAttributeClusterDimension; at[0].val.clusterDim={(unsigned)C,1,1};
  cfg.attrs=at; cfg.numAttrs=1;
}

int main(int argc,char**argv){
  setvbuf(stdout,nullptr,_IONBF,0);
  std::string only = argc>1? argv[1] : "all";
  auto want=[&](const char* s){ return only=="all" || only==s; };
  CKD(cuInit(0));
  int dev=0; CK(cudaSetDevice(dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
  int mclk=0; CK(cudaDeviceGetAttribute(&mclk,cudaDevAttrMemoryClockRate,dev));
  const double specBW=2.0*(double)mclk*1e3*(p.memoryBusWidth/8)/1e9;
  const int SM=p.multiProcessorCount;
  if(nvmlInit()==NVML_SUCCESS && nvmlDeviceGetHandleByIndex(0,&g_nvml)==NVML_SUCCESS) nvml_ok=true;
  printf("# %s sm_%d%d SMs=%d L2=%.0fMB specBW=%.0f GB/s  NVML=%s\n",
         p.name,p.major,p.minor,SM,p.l2CacheSize/1048576.0,specBW,nvml_ok?"on":"off");

  // ---- shared big arena + HBM ceiling ----
  size_t arenaB=(size_t)3<<30; void* dBig=nullptr; CK(cudaMalloc(&dBig,arenaB)); CK(cudaMemset(dBig,0x3c,arenaB));
  float* dOut=nullptr; CK(cudaMalloc(&dOut,1<<20));
  double ceiling=0;
  if(only!="p1tax"){
    size_t sb=(size_t)2560<<20, n4=sb/16;
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for(int g:{SM*8,SM*16}){
      k_stream<<<g,256>>>((const float4*)dBig,n4,dOut); CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(e0)); for(int i=0;i<5;i++) k_stream<<<g,256>>>((const float4*)dBig,n4,dOut);
      CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
      float ms; cudaEventElapsedTime(&ms,e0,e1); double gbs=sb/1e9/((ms/5)*1e-3);
      ceiling=std::max(ceiling,gbs);
    }
    printf("# HBM ceiling (short burst) = %.0f GB/s (%.0f%% spec); SM clk now %u MHz\n",
           ceiling,100*ceiling/specBW,sm_clock());
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }

  // =================================================================== P1
  if(want("p1")){
  printf("\n========== P1: cluster-cooperative claim from a global queue ==========\n");
    uint32_t* dCur; CK(cudaMalloc(&dCur,64)); float* dSink; CK(cudaMalloc(&dSink,4096*4));
    // --- correctness: one claim per worker, full grid ---
    printf("-- correctness (one claim/worker, verify broadcast + set coverage) --\n");
    {
      int grid=264;
      uint32_t* dGot; CK(cudaMalloc(&dGot,grid*4));
      // single
      CK(cudaMemset(dCur,0,4)); CK(cudaMemset(dGot,0xff,grid*4));
      k_claim_single_ck<<<grid,256>>>(dCur,grid,dGot); CK(cudaDeviceSynchronize());
      std::vector<uint32_t> h(grid); CK(cudaMemcpy(h.data(),dGot,grid*4,cudaMemcpyDeviceToHost));
      std::vector<int> seen(grid,0); bool ok=true; for(int i=0;i<grid;i++){ if(h[i]<(uint32_t)grid) seen[h[i]]++; else ok=false;} for(int i=0;i<grid;i++) ok&=(seen[i]==1);
      printf("   single  grid=264: claims cover [0,264) exactly once: %s\n",ok?"PASS":"FAIL");
      // cluster C=2,4: all C ranks in a cluster must read the SAME claimed index
      for(int C:{2,4}){
        int g=264;
        CK(cudaMemset(dCur,0,4)); CK(cudaMemset(dGot,0xff,g*4));
        cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(g),dim3(256),C,0);
        cudaError_t e=cudaLaunchKernelEx(&cfg,k_claim_cluster_ck,dCur,(unsigned)g,dGot);
        if(e==cudaSuccess) e=cudaDeviceSynchronize();
        if(e!=cudaSuccess){ printf("   cluster C=%d grid=264: LAUNCH/RUN FAIL %s\n",C,cudaGetErrorString(e)); cudaGetLastError(); continue; }
        CK(cudaMemcpy(h.data(),dGot,g*4,cudaMemcpyDeviceToHost));
        bool bok=true; int nCl=g/C; std::vector<int> cl_seen(nCl,0);
        for(int cl=0;cl<nCl;cl++){ uint32_t v=h[cl*C]; for(int r=1;r<C;r++) bok&=(h[cl*C+r]==v); if(v<(uint32_t)nCl) cl_seen[v]++; }
        bool cov=true; for(int i=0;i<nCl;i++) cov&=(cl_seen[i]==1);
        printf("   cluster C=%d grid=264 (%d clusters): no deadlock, all C ranks agree: %s ; claims cover [0,%d): %s\n",
               C,nCl,bok?"PASS":"FAIL",nCl,cov?"PASS":"FAIL");
      }
      cudaFree(dGot);
    }
    // --- overhead: fixed claims-per-worker P; latency = burst_ms / P ---
    printf("-- per-claim latency (work=0) and tax vs a representative op --\n");
    const int P=1500;    // claims per worker (sequential depth); one launch = one ~2ms burst
    // ROBUST single-block claim path (bench_group, min-of-12)
    auto measure_single=[&](int work)->Res{
      unsigned total=(unsigned)P*264u;
      std::vector<Case> cs;
      cs.push_back({"x",[=]{ cudaMemsetAsync(dCur,0,4); k_claim_single<<<264,256>>>(dCur,total,work,dSink); }});
      auto r=bench_group(cs,12,0.002,0); return {r[0].ms,r[0].clk};
    };
    // DEFENSIVE cluster claim timing: sustained cluster launches fault after a few
    // launches on this driver, so time each launch individually, catch the fault,
    // and take min over however many succeed (nok reported). MUST run after all the
    // robust single measurements — the fault corrupts the CUDA context.
    auto time_cluster=[&](int C,int work,int attempts,int& nok,bool& faulted)->Res{
      int nWorkers=264/C; unsigned total=(unsigned)P*(unsigned)nWorkers;
      cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      double best=1e18,clk=0; nok=0; faulted=false;
      for(int a=0;a<attempts;a++){
        if(cudaMemset(dCur,0,4)!=cudaSuccess){faulted=true;break;}
        cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(264),dim3(256),C,0);
        cudaEventRecord(e0);
        cudaError_t e=cudaLaunchKernelEx(&cfg,k_claim_cluster,dCur,total,work,dSink);
        cudaEventRecord(e1);
        if(e==cudaSuccess) e=cudaEventSynchronize(e1);
        if(e!=cudaSuccess){ faulted=true; cudaGetLastError(); break; }
        float ms; cudaEventElapsedTime(&ms,e0,e1); if(ms<best)best=ms; nok++;
        unsigned c=sm_clock(); if(c)clk=c; usleep(25000);
      }
      cudaEventDestroy(e0);cudaEventDestroy(e1);
      return {best,clk};
    };
    // single latency + representative op sizes FIRST (robust)
    Res s0=measure_single(0);
    double lat_s=s0.ms*1e6/P;
    printf("   single  (atomicAdd + 2x __syncthreads)         : %.1f ns/claim  (clk %.0f MHz)\n",lat_s,s0.clk);
    double op_only[4]; const int works[4]={256,1024,4096,16384};
    for(int i=0;i<4;i++){ Res sw=measure_single(works[i]); op_only[i]=sw.ms*1e6/P-lat_s; }
    // cluster C=2 (defensive; this may fault the context, so it is the LAST timed thing)
    int nok=0; bool faulted=false;
    Res c2=time_cluster(2,0,12,nok,faulted);
    double lat_c2=c2.ms*1e6/P;
    printf("   cluster C=2 (1 atomicAdd/2blk + 2x cluster.sync + DSM): %.1f ns/claim  (clk %.0f MHz)"
           "  [min of %d successful launch(es)%s]\n",lat_c2,c2.clk,nok,faulted?", then FAULTED":"");
    if(nok>0){
      printf("   >>> per-claim TAX (C=2) = +%.1f ns  (%.2fx the single-block claim)\n",lat_c2-lat_s,lat_c2/lat_s);
      printf("-- representative op sizes -> C=2 claim-tax as %% of op --\n");
      for(int i=0;i<4;i++)
        printf("   work=%-6d op-only=%.0f ns : C=2 tax = %.1f%% of op\n",works[i],op_only[i],100*(lat_c2-lat_s)/op_only[i]);
    }
    // ---- C=4 VIABILITY at full grid (LAST: a hang/fault here corrupts the context) ----
    printf("-- C=4 viability at full grid=264 (66 clusters); native concurrent execution --\n");
    float* dSpin=dSink;
    { // (i) fixed-iteration spin: protocol in a loop, no exhaustion boundary
      CK(cudaMemset(dCur,0,4));
      cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(264),dim3(256),4,0);
      cudaError_t e=cudaLaunchKernelEx(&cfg,k_cluster_spin,dCur,500,dSpin);
      if(e==cudaSuccess) e=cudaDeviceSynchronize();
      printf("   C=4 fixed-spin (500 iters, cluster.sync+atomic+DSM in a loop): %s\n",
             e==cudaSuccess?"OK (no deadlock)":cudaGetErrorString(e));
      cudaGetLastError();
    }
    { // (ii) cursor-driven claim, full grid
      CK(cudaMemset(dCur,0,4));
      cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(264),dim3(256),4,0);
      cudaError_t e=cudaLaunchKernelEx(&cfg,k_claim_cluster,dCur,(unsigned)(1000u*66u),0,dSpin);
      if(e==cudaSuccess) e=cudaDeviceSynchronize();
      printf("   C=4 cursor-driven claim (full grid):                          %s\n",
             e==cudaSuccess?"OK (no deadlock)":cudaGetErrorString(e));
      cudaGetLastError();
    }
    cudaFree(dCur); cudaFree(dSink);
  }

  // ============================================== P1-TAX (lean, run as its own
  // fresh process to avoid the DVFS collapse that livelocks the cluster spin).
  if(only=="p1tax"){
  printf("\n========== P1-TAX: per-claim latency (minimal preamble) ==========\n");
    const int P=1500;
    uint32_t* dCur; CK(cudaMalloc(&dCur,64)); float* dSink; CK(cudaMalloc(&dSink,4096*4));
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time1=[&](bool cluster,int C,int work,cudaError_t& err)->double{
      int nWorkers = cluster? 264/C : 264; unsigned total=(unsigned)P*(unsigned)nWorkers;
      err=cudaMemset(dCur,0,4); if(err)return 0;
      cudaEventRecord(e0);
      if(cluster){ cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(264),dim3(256),C,0);
        err=cudaLaunchKernelEx(&cfg,k_claim_cluster,dCur,total,work,dSink); }
      else { k_claim_single<<<264,256>>>(dCur,total,work,dSink); err=cudaGetLastError(); }
      cudaEventRecord(e1);
      if(!err) err=cudaEventSynchronize(e1);
      if(err){ cudaGetLastError(); return 0; }
      float ms; cudaEventElapsedTime(&ms,e0,e1); return ms;
    };
    cudaError_t err;
    // single: warmup + min of 5
    time1(false,1,0,err);
    double ls=1e18; for(int i=0;i<5;i++){ double m=time1(false,1,0,err); if(!err&&m<ls)ls=m; usleep(25000);}
    double lat_s=ls*1e6/P;
    printf("   single  claim: %.1f ns/claim (clk %.0f MHz)\n",lat_s,(double)sm_clock());
    // cluster C=2: min over successful launches
    { double lc=1e18; int nok=0; bool f=false;
      for(int i=0;i<5 && !f;i++){ double m=time1(true,2,0,err); if(err){f=true;break;} if(m<lc)lc=m; nok++; usleep(30000);}
      if(nok>0){ double lat=lc*1e6/P;
        printf("   cluster C=2 claim: %.1f ns/claim (clk %.0f MHz) [min of %d ok%s]\n",lat,(double)sm_clock(),nok,f?", then faulted":"");
        printf("   >>> TAX C=2 = +%.1f ns (%.2fx single); as %% of op: 256-flop op~1.2us->%.0f%%, 4k-flop~19us->%.1f%%\n",
               lat-lat_s,lat/lat_s,100*(lat-lat_s)/1200.0,100*(lat-lat_s)/19000.0);
      } else printf("   cluster C=2 claim: FAULTED on first launch (%s)\n",cudaGetErrorString(err));
    }
    return 0;  // p1tax is standalone; context may be corrupt now
  }

  // =================================================================== P2
  if(want("p2")){
  printf("\n========== P2: TMA multicast, working-set sweep (L2 -> >>L2) ==========\n");
    const int K=3840, N=4096, BN=P2_BN, nNTiles=N/BN;
    size_t smB=(size_t)P2_NSTAGE*P2_TILE_B + (size_t)P2_BM*K*4 + 512;
    for(auto f:{(void*)k_bcast<false,false>,(void*)k_bcast<true,false>})
      CK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smB));
    int occ=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,(void*)k_bcast<false,false>,P2_THREADS,smB));
    __nv_bfloat16* dA; CK(cudaMalloc(&dA,(size_t)16*P2_BM*K*2));
    { std::vector<__nv_bfloat16> ha((size_t)16*P2_BM*K); for(size_t i=0;i<ha.size();i++) ha[i]=__float2bfloat16(0.001f*(i%251)); CK(cudaMemcpy(dA,ha.data(),ha.size()*2,cudaMemcpyHostToDevice)); }
    printf("   occ=%d blk/SM, smem=%.0f KB, N=%d, per-replica weight=%.1f MB. per-CTA unique GB/s:\n"
           "     ~ceiling => duplicate is an L2 hit (mc cannot help);  ~ceiling/C => off-L2 (mc should win)\n",
           occ,smB/1024.0,N,(double)N*K*2/1048576.0);
    printf("   %-6s %-6s | %-22s | %-22s | mc/perCTA\n","R","WS_MB","per-CTA (uniqGB/s,ms)","multicast (uniqGB/s,ms)");
    for(int R:{1,2,4,8,16,32,48}){
      size_t need=(size_t)N*K*2*R; if(need>arenaB){ printf("   R=%d skip (>arena)\n",R); continue; }
      int nItems=R*nNTiles;
      CUtensorMap map; memset(&map,0,sizeof(map));
      uint64_t gd[2]={(uint64_t)K,(uint64_t)N*(uint64_t)R}; uint64_t gs[1]={(uint64_t)K*2};
      uint32_t bd[2]={(uint32_t)P2_BK,(uint32_t)P2_BN}; uint32_t es[2]={1,1};
      CKD(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dBig,gd,gs,bd,es,
          CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
      for(int C:{2,4}){
        int nClusters=std::max(1,(SM*occ)/C), grid=nClusters*C;
        std::vector<Case> cs;
        cs.push_back({"perCTA",[=]{ cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(grid),dim3(P2_THREADS),C,smB);
          cudaLaunchKernelEx(&cfg,k_bcast<false,false>,map,dA,dOut,K,nNTiles,nItems,nClusters,C); }});
        cs.push_back({"mc",[=]{ cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(grid),dim3(P2_THREADS),C,smB);
          cudaLaunchKernelEx(&cfg,k_bcast<true,false>,map,dA,dOut,K,nNTiles,nItems,nClusters,C); }});
        auto r=bench_group(cs,12);
        double uniq=(double)need/1e9;
        double a_gb=uniq/(r[0].ms*1e-3), b_gb=uniq/(r[1].ms*1e-3);
        printf("   %-6d %-6.0f C=%d | %8.0f GB/s %7.3f ms | %8.0f GB/s %7.3f ms | %.3fx (clk %.0f)\n",
               R,need/1048576.0,C,a_gb,r[0].ms,b_gb,r[1].ms,r[0].ms/r[1].ms,r[0].clk);
      }
    }
    printf("   HBM ceiling=%.0f GB/s. per-CTA unique GB/s that stays ~ceiling across the sweep\n"
           "   proves the duplicate is ALWAYS an L2 hit (co-resident cluster reads are simultaneous).\n",ceiling);
    cudaFree(dA);
  }

  // =================================================================== P3a
  if(want("p3a")){
  printf("\n========== P3a: split-K GEMM combine — DSM vs global ==========\n");
    const int K=16384;
    // end-to-end batch (fits ~2ms): B problems, each A[64][K]bf16 + B[K][256]bf16 = 10 MB
    const int Bee=64;
    // isolated-reduce batch (partials > L2): B problems * C * 64KB
    const int Bre=264;
    size_t smB=(size_t)(G_M*G_N + G_M*G_BK)*4;
    for(auto f:{(void*)k_gemm_dsm,(void*)k_gemm_part_write})
      CK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smB));
    CK(cudaFuncSetAttribute((void*)k_reduce_dsm,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(G_M*G_N*4)));
    // inputs for end-to-end (Bee problems)
    __nv_bfloat16 *dAe,*dBe,*dOe; float* dScr;
    CK(cudaMalloc(&dAe,(size_t)Bee*G_M*K*2)); CK(cudaMalloc(&dBe,(size_t)Bee*K*G_N*2));
    CK(cudaMalloc(&dOe,(size_t)Bee*G_M*G_N*2));
    { std::vector<__nv_bfloat16> ha((size_t)Bee*G_M*K), hb((size_t)Bee*K*G_N);
      srand(1); for(auto&x:ha) x=__float2bfloat16((rand()/(float)RAND_MAX-0.5f)*0.1f);
      for(auto&x:hb) x=__float2bfloat16((rand()/(float)RAND_MAX-0.5f)*0.1f);
      CK(cudaMemcpy(dAe,ha.data(),ha.size()*2,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dBe,hb.data(),hb.size()*2,cudaMemcpyHostToDevice)); }
    for(int C:{4}){
      CK(cudaMalloc(&dScr,(size_t)Bee*C*G_M*G_N*4));
      // correctness: CPU f32 ref for problem 0
      std::vector<__nv_bfloat16> hA((size_t)G_M*K),hB((size_t)K*G_N); std::vector<float> ref(G_M*G_N,0);
      CK(cudaMemcpy(hA.data(),dAe,hA.size()*2,cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hB.data(),dBe,hB.size()*2,cudaMemcpyDeviceToHost));
      for(int m=0;m<G_M;m++) for(int n=0;n<G_N;n++){ float s=0; for(int k=0;k<K;k++) s+=__bfloat162float(hA[(size_t)m*K+k])*__bfloat162float(hB[(size_t)k*G_N+n]); ref[m*G_N+n]=s; }
      auto relL2=[&](const char* tag){ std::vector<__nv_bfloat16> ho(G_M*G_N); CK(cudaMemcpy(ho.data(),dOe,G_M*G_N*2,cudaMemcpyDeviceToHost));
        double num=0,den=0; for(int i=0;i<G_M*G_N;i++){ double d=__bfloat162float(ho[i])-ref[i]; num+=d*d; den+=(double)ref[i]*ref[i]; }
        double rl=sqrt(num/den); printf("   [ck] %-14s relL2=%.2e %s\n",tag,rl,rl<3e-3?"PASS":"FAIL"); };
      // DSM end-to-end
      { cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(Bee*C),dim3(256),C,smB);
        CK(cudaLaunchKernelEx(&cfg,k_gemm_dsm,dAe,dBe,dOe,K,C)); CK(cudaDeviceSynchronize()); relL2("DSM e2e"); }
      // global end-to-end (write partials + reduce)
      { k_gemm_part_write<<<Bee*C,256,smB>>>(dAe,dBe,dScr,K,C);
        k_reduce_global<<<Bee,256>>>(dScr,dOe,C); CK(cudaDeviceSynchronize()); relL2("global e2e"); }
      // timing end-to-end
      std::vector<Case> cs;
      cs.push_back({"dsm",[=]{ cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(Bee*C),dim3(256),C,smB);
        cudaLaunchKernelEx(&cfg,k_gemm_dsm,dAe,dBe,dOe,K,C); }});
      cs.push_back({"glob",[=]{ k_gemm_part_write<<<Bee*C,256,smB>>>(dAe,dBe,dScr,K,C); k_reduce_global<<<Bee,256>>>(dScr,dOe,C); }});
      cs.push_back({"gemm_only",[=]{ k_gemm_part_write<<<Bee*C,256,smB>>>(dAe,dBe,dScr,K,C); }});
      auto r=bench_group(cs,12);
      printf("   end-to-end split-K (B=%d,C=%d,K=%d): DSM %.3f ms | global(write+reduce) %.3f ms | gemm-only %.3f ms  (clk %.0f)\n",
             Bee,C,K,r[0].ms,r[1].ms,r[2].ms,r[0].clk);
      printf("     -> reduction cost:  DSM %.3f ms  vs  global write+reduce %.3f ms  (GEMM compute dominates)\n",
             r[0].ms-r[2].ms, r[1].ms-r[2].ms);
      cudaFree(dScr);
    }
    cudaFree(dAe); cudaFree(dBe); cudaFree(dOe);
    // ISOLATED reduce (partials > L2): shows the HBM the two paths actually move
    printf("-- isolated reduce (partials pre-filled, B=%d > L2): DSM reads own partial + DSM peers; global reads all C from HBM --\n",Bre);
    for(int C:{2,4}){
      float* dScr2; __nv_bfloat16* dO2;
      CK(cudaMalloc(&dScr2,(size_t)Bre*C*G_M*G_N*4)); CK(cudaMalloc(&dO2,(size_t)Bre*G_M*G_N*2));
      CK(cudaMemset(dScr2,0x3c,(size_t)Bre*C*G_M*G_N*4));
      std::vector<Case> cs;
      cs.push_back({"dsm",[=]{ cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(Bre*C),dim3(256),C,(size_t)G_M*G_N*4);
        cudaLaunchKernelEx(&cfg,k_reduce_dsm,dScr2,dO2,C); }});
      cs.push_back({"glob",[=]{ k_reduce_global<<<Bre,256>>>(dScr2,dO2,C); }});
      auto r=bench_group(cs,12);
      double scrMB=(double)Bre*C*G_M*G_N*4/1048576.0;
      double glob_hbm=(double)Bre*C*G_M*G_N*4 + (double)Bre*G_M*G_N*2;   // read all + write O
      double dsm_hbm =(double)Bre*G_M*G_N*4 + (double)Bre*G_M*G_N*2;     // read own + write O
      printf("   C=%d scratch=%.0f MB: DSM %.4f ms (%.0f GB/s eff) | global %.4f ms (%.0f GB/s eff) | DSM/global %.3fx (clk %.0f)\n",
             C,scrMB, r[0].ms, dsm_hbm/1e9/(r[0].ms*1e-3), r[1].ms, glob_hbm/1e9/(r[1].ms*1e-3), r[0].ms/r[1].ms, r[0].clk);
      cudaFree(dScr2); cudaFree(dO2);
    }
  }

  // =================================================================== P3b
  if(want("p3b")){
  printf("\n========== P3b: flash split-KV merge — DSM vs global d_flash_merge ==========\n");
    const int Bp=2048;    // problems; partials C*Bp*(BQ*HD*4 + BQ*2*4) must exceed L2
    CK(cudaFuncSetAttribute((void*)k_flash_merge_dsm,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)((F_BQ*F_HD+F_BQ*2)*4)));
    for(int C:{4,8}){
      float *dOp,*dMl; __nv_bfloat16* dO;
      CK(cudaMalloc(&dOp,(size_t)Bp*C*F_BQ*F_HD*4)); CK(cudaMalloc(&dMl,(size_t)Bp*C*F_BQ*2*4)); CK(cudaMalloc(&dO,(size_t)Bp*F_BQ*F_HD*2));
      // synthetic but valid partials: O_i = sum p*v (>0-ish), m_i spread so rescale is exercised
      std::vector<float> hOp((size_t)Bp*C*F_BQ*F_HD), hMl((size_t)Bp*C*F_BQ*2);
      srand(7);
      for(size_t prob=0;prob<(size_t)Bp;prob++) for(int s=0;s<C;s++) for(int q=0;q<F_BQ;q++){
        float mi=(float)(s*1.3f + (q%5)*0.7f);       // varied maxima across splits
        float li=1.0f + 0.5f*(rand()/(float)RAND_MAX);
        hMl[((prob*C+s)*F_BQ+q)*2+0]=mi; hMl[((prob*C+s)*F_BQ+q)*2+1]=li;
        for(int d=0;d<F_HD;d++) hOp[((prob*C+s)*F_BQ+q)*F_HD+d]=li*(0.5f+0.01f*d+0.1f*s);
      }
      CK(cudaMemcpy(dOp,hOp.data(),hOp.size()*4,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dMl,hMl.data(),hMl.size()*4,cudaMemcpyHostToDevice));
      // CPU ref merge for problem 0
      std::vector<float> ref(F_BQ*F_HD);
      for(int q=0;q<F_BQ;q++){ float M=-1e30f; for(int s=0;s<C;s++) M=std::max(M,hMl[((0*C+s)*F_BQ+q)*2+0]);
        float l=0; std::vector<float> acc(F_HD,0);
        for(int s=0;s<C;s++){ float mi=hMl[((0*C+s)*F_BQ+q)*2+0], li=hMl[((0*C+s)*F_BQ+q)*2+1]; float w=expf(mi-M); l+=li*w;
          for(int d=0;d<F_HD;d++) acc[d]+=w*hOp[((0*C+s)*F_BQ+q)*F_HD+d]; }
        for(int d=0;d<F_HD;d++) ref[q*F_HD+d]=acc[d]/l; }
      auto relL2=[&](const char* tag){ std::vector<__nv_bfloat16> ho(F_BQ*F_HD); CK(cudaMemcpy(ho.data(),dO,F_BQ*F_HD*2,cudaMemcpyDeviceToHost));
        double num=0,den=0; for(int i=0;i<F_BQ*F_HD;i++){ double d=__bfloat162float(ho[i])-ref[i]; num+=d*d; den+=(double)ref[i]*ref[i]; }
        double rl=sqrt(num/den); printf("   [ck] C=%d %-8s relL2=%.2e %s\n",C,tag,rl,rl<3e-3?"PASS":"FAIL"); };
      // correctness
      k_flash_merge_global<<<Bp,F_HD>>>(dOp,dMl,dO,C); CK(cudaDeviceSynchronize()); relL2("global");
      { cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(Bp*C),dim3(F_HD),C,(size_t)(F_BQ*F_HD+F_BQ*2)*4);
        CK(cudaLaunchKernelEx(&cfg,k_flash_merge_dsm,dOp,dMl,dO,C)); CK(cudaDeviceSynchronize()); relL2("DSM"); }
      // timing
      std::vector<Case> cs;
      cs.push_back({"dsm",[=]{ cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(Bp*C),dim3(F_HD),C,(size_t)(F_BQ*F_HD+F_BQ*2)*4);
        cudaLaunchKernelEx(&cfg,k_flash_merge_dsm,dOp,dMl,dO,C); }});
      cs.push_back({"glob",[=]{ k_flash_merge_global<<<Bp,F_HD>>>(dOp,dMl,dO,C); }});
      auto r=bench_group(cs,12);
      double glob_hbm=(double)Bp*C*F_BQ*F_HD*4 + (double)Bp*C*F_BQ*2*4 + (double)Bp*F_BQ*F_HD*2;
      double dsm_hbm =(double)Bp*F_BQ*F_HD*4  + (double)Bp*F_BQ*2*4  + (double)Bp*F_BQ*F_HD*2;
      double partMB=(double)Bp*C*(F_BQ*F_HD*4+F_BQ*2*4)/1048576.0;
      printf("   C=%d Tkv-split=%d partials=%.0f MB (>L2): DSM %.4f ms (%.0f GB/s) | global %.4f ms (%.0f GB/s) | DSM/global %.3fx (clk %.0f)\n",
             C,C,partMB, r[0].ms, dsm_hbm/1e9/(r[0].ms*1e-3), r[1].ms, glob_hbm/1e9/(r[1].ms*1e-3), r[0].ms/r[1].ms, r[0].clk);
      cudaFree(dOp); cudaFree(dMl); cudaFree(dO);
    }
  }

  // =================================================================== P2'
  if(want("p2prime")){
  printf("\n========== P2': compute-bound wgmma bf16 GEMM, cluster weight-multicast vs per-CTA ==========\n");
    // 128B-swizzled tensor maps: A[M,K] and B[N,K], both K-contiguous, box {BK, rows}
    auto make_swz=[&](CUtensorMap* m, void* base, int rows, int K, int boxRows){
      uint64_t gd[2]={(uint64_t)K,(uint64_t)rows}; uint64_t gs[1]={(uint64_t)K*2};
      uint32_t bd[2]={(uint32_t)CG_BK,(uint32_t)boxRows}; uint32_t es[2]={1,1};
      memset(m,0,sizeof(*m));
      CKD(cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,base,gd,gs,bd,es,
          CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    };
    const size_t smB=(size_t)CG_NS*(CG_ATB+CG_BTB)+2048;
    for(auto f:{(void*)k_cgemm<false>,(void*)k_cgemm<true>})
      CK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smB));
    int occ=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,(void*)k_cgemm<false>,CG_WG,smB));
    printf("   tile BM=%d BN=%d BK=%d NS=%d, smem=%.0f KB, occ=%d blk/SM (share the WEIGHT B along M-split)\n",
           CG_BM,CG_BN,CG_BK,CG_NS,smB/1024.0,occ);

    // wgmma rate ceiling (back-to-back, no memory)
    double tf_ceil=0;
    { int grid=SM*occ, iters=20000; float* dS; CK(cudaMalloc(&dS,grid*4));
      CK(cudaFuncSetAttribute((void*)k_wgmma_rate,cudaFuncAttributeMaxDynamicSharedMemorySize,0));
      cudaEvent_t e0,e1; CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
      k_wgmma_rate<<<grid,CG_WG>>>(dS,200); CK(cudaDeviceSynchronize());
      double best=1e18; for(int r=0;r<12;r++){ CK(cudaEventRecord(e0)); k_wgmma_rate<<<grid,CG_WG>>>(dS,iters);
        CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize()); float ms; cudaEventElapsedTime(&ms,e0,e1);
        best=std::min(best,(double)ms); usleep(25000);}
      double flop=2.0*CG_BM*CG_BN*16.0*CG_KSUB*(double)iters*(double)grid;
      tf_ceil=flop/1e12/(best*1e-3);
      printf("   wgmma back-to-back ceiling = %.0f TF/s (%.0f MHz)\n",tf_ceil,(double)sm_clock());
      cudaFree(dS); cudaEventDestroy(e0); cudaEventDestroy(e1); }

    struct Shp{int M,N,K;};
    Shp shapes[]={{4096,4096,4096},{2048,8192,8192},{8192,4096,4096}};
    printf("   %-18s %-3s | %-20s | %-20s | mc/perCTA  (correctness relL2)\n","M,N,K","C","per-CTA (TF/s,ms)","multicast (TF/s,ms)");
    for(auto sh:shapes){
      int M=sh.M,N=sh.N,K=sh.K;
      size_t As=(size_t)M*K*2, Bs=(size_t)N*K*2, Cs=(size_t)M*N*4;
      if(As+Bs+Cs > arenaB){ printf("   %d,%d,%d skip(>arena)\n",M,N,K); continue; }
      __nv_bfloat16 *dAm,*dBm; float* dCm;
      CK(cudaMalloc(&dAm,As)); CK(cudaMalloc(&dBm,Bs)); CK(cudaMalloc(&dCm,Cs));
      std::vector<float> hAf((size_t)M*K), hBf((size_t)N*K);
      { std::vector<__nv_bfloat16> hA((size_t)M*K),hB((size_t)N*K); srand(11);
        for(size_t i=0;i<hA.size();i++){ float v=(rand()/(float)RAND_MAX-0.5f)*0.2f; hA[i]=__float2bfloat16(v); hAf[i]=__bfloat162float(hA[i]); }
        for(size_t i=0;i<hB.size();i++){ float v=(rand()/(float)RAND_MAX-0.5f)*0.2f; hB[i]=__float2bfloat16(v); hBf[i]=__bfloat162float(hB[i]); }
        CK(cudaMemcpy(dAm,hA.data(),As,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dBm,hB.data(),Bs,cudaMemcpyHostToDevice)); }
      CUtensorMap mapA,mapB; make_swz(&mapA,dAm,M,K,CG_BM); make_swz(&mapB,dBm,N,K,CG_BN);
      // CPU oracle for a corner tile block (rows 0..BM-1, cols 0..BN-1) — cheap check
      auto check=[&](const char* tag)->double{
        std::vector<float> hc((size_t)CG_BM*N); // read first BM rows
        CK(cudaMemcpy(hc.data(),dCm,(size_t)CG_BM*N*4,cudaMemcpyDeviceToHost));
        double num=0,den=0;
        for(int m=0;m<CG_BM;m++) for(int n=0;n<CG_BN;n++){ // check first BN cols
          double s=0; for(int k=0;k<K;k++) s+=(double)hAf[(size_t)m*K+k]*hBf[(size_t)n*K+k];
          double d=hc[(size_t)m*N+n]-s; num+=d*d; den+=s*s; }
        return sqrt(num/den);
      };
      for(int C:{2,4}){
        int mtiles=M/CG_BM;
        if(mtiles%C){ printf("   %d,%d,%d C=%d skip(mtiles%%C)\n",M,N,K,C); continue; }
        int nClusters=std::max(1,(SM*occ)/C), grid=nClusters*C;
        double res[2]={0,0}, rl[2]={0,0}, clk=0;
        for(int v=0;v<2;v++){
          auto go=[&]{ cudaLaunchConfig_t cfg; cudaLaunchAttribute at[1]; clusterCfg(cfg,at,dim3(grid),dim3(CG_WG),C,smB);
            if(v==0) cudaLaunchKernelEx(&cfg,k_cgemm<false>,mapA,mapB,dCm,M,N,K,nClusters,C);
            else     cudaLaunchKernelEx(&cfg,k_cgemm<true >,mapA,mapB,dCm,M,N,K,nClusters,C); };
          CK(cudaMemset(dCm,0,Cs)); go(); cudaError_t e=cudaDeviceSynchronize();
          if(e!=cudaSuccess){ printf("   %d,%d,%d C=%d %s LAUNCH FAIL %s\n",M,N,K,C,v?"mc":"perCTA",cudaGetErrorString(e)); cudaGetLastError(); res[v]=-1; continue; }
          rl[v]=check(v?"mc":"perCTA");
          std::vector<Case> cs; cs.push_back({"g",go}); auto r=bench_group(cs,12); res[v]=r[0].ms; clk=r[0].clk;
        }
        if(res[0]>0&&res[1]>0){
          double flop=2.0*M*N*K;
          double tf0=flop/1e12/(res[0]*1e-3), tf1=flop/1e12/(res[1]*1e-3);
          printf("   %5d,%5d,%5d C=%d | %6.0f TF/s %6.3f ms | %6.0f TF/s %6.3f ms | %.3fx  (relL2 %.1e/%.1e, clk %.0f)\n",
                 M,N,K,C, tf0,res[0], tf1,res[1], res[0]/res[1], rl[0],rl[1], clk);
          printf("        per-CTA %.0f%% of wgmma ceiling ; multicast %.0f%% ; %s\n",
                 100*tf0/tf_ceil,100*tf1/tf_ceil, res[1]<res[0]*0.99?"MULTICAST WINS":(res[1]>res[0]*1.01?"multicast loses":"tie"));
        }
      }
      cudaFree(dAm); cudaFree(dBm); cudaFree(dCm);
    }
    printf("   verdict: multicast pays iff mc<perCTA AND per-CTA is TMA/L2-bound (far below ceiling);\n"
           "   if per-CTA already near ceiling, the operand load was not the bottleneck.\n");
  }

  printf("\n# done. HBM ceiling %.0f GB/s. See per-section verdicts above.\n",ceiling);
  if(nvml_ok) nvmlShutdown();
  return 0;
}

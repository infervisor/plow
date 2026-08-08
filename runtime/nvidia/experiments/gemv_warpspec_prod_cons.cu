// gemv_warpspec_prod_cons.cu — REVIVAL: row-double-buffered producer/consumer GEMV (M=1)
// for the plow persistent megakernel on sm_120a. Designed in the RTX campaign, NEVER
// measured until now.
//
// Hypothesis (from the design notes "Revival candidate"): a uniform one-warp-
// per-row GEMV pays a memory-controller turnaround between consecutive rows. Split the
// block into PRODUCER warps (cp.async weight rows into a depth-S smem ring) and CONSUMER
// warps (cooperatively dot each row from smem), so producers pre-stage row r+1..r+S-1
// while consumers finish row r. Double/triple-buffered across OUTPUT ROWS (not K-tiles) —
// distinct from warpspec_ab.cu (K-tile warp-spec) which was refuted.
//
// This file was shipped with three defects (never ran):
//   (1) __syncthreads() inside the consumer-only branch — warp-divergent barrier, hangs.
//       FIX: named barrier `bar.sync 1, NW_CONS*32` scoped to the consumer warps only.
//   (2) mbarrier.arrive.expect_tx (a TMA primitive) mixed with cp.async — the tx byte
//       count is never decremented by cp.async, so try_wait never flips.
//       FIX: init full[] with the producer-thread arrival count and signal completion with
//       cp.async.mbarrier.arrive.noinc (the correct cp.async<->mbarrier tie).
//   (3) harness measured L2 (23 MB weight tensors fit the ~96 MB L2), not HBM.
//       FIX: replicate weights past L2 and cycle, exactly like fp8_gemv.cu.
//
// Build: /usr/local/cuda/bin/nvcc -arch=sm_120a -O3 -o /tmp/g gemv_warpspec_prod_cons.cu
// Run:   gpulease p9-proto /tmp/g

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

#define HBM_SPEC 1790.0   // GB/s, RTX PRO 6000 Blackwell GDDR7
#define BLOCK    256
#define NW_PROD  2
#define NW_CONS  6
#define MAXST    4        // max ring depth

// ---- inline helpers ----
__device__ __forceinline__ uint32_t sptr(const void *p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(void *b, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sptr(b)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(void *b) {
  asm volatile("{ .reg .b64 s; mbarrier.arrive.shared::cta.b64 s, [%0]; }" ::"r"(sptr(b)));
}
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(void *b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(sptr(b)));
}
__device__ __forceinline__ void mbar_wait(void *b, uint32_t parity) {
  asm volatile(
    "{ .reg .pred P; WSPIN%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; "
    "@!P bra WSPIN%=; }" ::"r"(sptr(b)), "r"(parity));
}
__device__ __forceinline__ void cp_async_cg16(void *dst, const void *src) {
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(sptr(dst)), "l"(src));
}
__device__ __forceinline__ void bar_sync(int id, int cnt) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(cnt));
}
__device__ __forceinline__ float warp_reduce(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// bf16 dot-8 with 16B vector loads from smem
__device__ __forceinline__ float dot8_smem(const __nv_bfloat16 *w, const __nv_bfloat16 *x, float acc) {
  #pragma unroll
  for (int j = 0; j < 8; j++)
    acc = __fmaf_rn(__bfloat162float(w[j]), __bfloat162float(x[j]), acc);
  return acc;
}

// ============================ BASELINE: faithful d_gemv (op_gemm.cuh) ============================
// One warp owns one output row; GV_UNROLL=8 vectorized weight loads (uint4/16B) issued before
// consume; x staged in smem; warp_sum32 reduce. Byte-identical structure to gemv_rows<1>.
#define GV_UNROLL 8
__global__ __launch_bounds__(BLOCK, 1)
void gemv_uniform_bf16(const __nv_bfloat16 *__restrict__ W,
                       const __nv_bfloat16 *__restrict__ x,
                       __nv_bfloat16 *__restrict__ C,
                       int N, int K, unsigned slice, unsigned nblk) {
  extern __shared__ __align__(16) char smem[];
  __nv_bfloat16 *xs = (__nv_bfloat16 *)smem;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  slice = blockIdx.x;                    // standalone grid launch: block == slice
  for (int i = threadIdx.x; i < K; i += BLOCK) xs[i] = x[i];
  __syncthreads();

  const unsigned per = (N + nblk - 1) / nblk;
  const unsigned n0 = slice * per;
  const unsigned n1 = (n0 + per < (unsigned)N) ? (n0 + per) : (unsigned)N;
  const unsigned nchunk = (K + 255) / 256;

  for (unsigned n = n0 + warp; n < n1; n += 8) {
    const __nv_bfloat16 *wrow = W + (size_t)n * K;
    float acc = 0.f;
    for (unsigned c = 0; c < nchunk; c += GV_UNROLL) {
      __nv_bfloat16 wv[GV_UNROLL][8];
      unsigned kk[GV_UNROLL];
      #pragma unroll
      for (int u = 0; u < GV_UNROLL; u++) {
        unsigned k = (c + u) * 256 + lane * 8;
        kk[u] = k;
        if (k < (unsigned)K) *(uint4 *)wv[u] = *(const uint4 *)(wrow + k);
      }
      #pragma unroll
      for (int u = 0; u < GV_UNROLL; u++) {
        if (kk[u] < (unsigned)K) acc = dot8_smem(wv[u], xs + kk[u], acc);
      }
    }
    acc = warp_reduce(acc);
    if (lane == 0) C[n] = __float2bfloat16(acc);
  }
}

// ============================ PRODUCER/CONSUMER, depth-S ring ============================
// Warps 0..NW_PROD-1  : producers, cp.async weight rows into ring, arrive full[s]
// Warps NW_PROD..7    : consumers, dot ring[s] cooperatively, arrive empty[s]
// stages runtime (<= MAXST). smem: xs[K] | ring[stages*K] | bars[2*MAXST] | reduce[NW_CONS]
__global__ __launch_bounds__(BLOCK, 1)
void gemv_prodcons_bf16(const __nv_bfloat16 *__restrict__ W,
                        const __nv_bfloat16 *__restrict__ x,
                        __nv_bfloat16 *__restrict__ C,
                        int N, int K, unsigned slice, unsigned nblk, int stages) {
  extern __shared__ __align__(16) char smem[];
  __nv_bfloat16 *xs = (__nv_bfloat16 *)smem;
  __nv_bfloat16 *ring = (__nv_bfloat16 *)(smem + (size_t)K * 2);
  uint64_t *full = (uint64_t *)(ring + (size_t)stages * K);
  uint64_t *empty = full + MAXST;
  float *reduce_buf = (float *)(empty + MAXST);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  slice = blockIdx.x;                    // standalone grid launch: block == slice

  if (tid == 0) {
    for (int s = 0; s < stages; s++) {
      mbar_init(&full[s], NW_PROD * 32);   // 64 producer threads arrive per fill
      mbar_init(&empty[s], NW_CONS * 32);  // 192 consumer threads arrive per drain
    }
  }
  for (int i = tid; i < K; i += BLOCK) xs[i] = x[i];
  __syncthreads();

  const unsigned per = (N + nblk - 1) / nblk;
  const unsigned n0 = slice * per;
  const unsigned n1 = (n0 + per < (unsigned)N) ? (n0 + per) : (unsigned)N;
  const unsigned my_rows = (n1 > n0) ? (n1 - n0) : 0u;   // last block: slice*per may exceed N

  if (warp < NW_PROD) {
    // ---- PRODUCERS ----
    const int ptid = tid;                       // 0..63
    const int issues = (K + 64 * 8 - 1) / (64 * 8);
    for (unsigned row = 0; row < my_rows; row++) {
      const int s = row % stages;
      // wait until consumers have drained the previous occupant of this slot
      if (row >= (unsigned)stages)
        mbar_wait(&empty[s], (row / stages - 1) & 1);
      __nv_bfloat16 *dst = ring + (size_t)s * K;
      const __nv_bfloat16 *src = W + (size_t)(n0 + row) * K;
      for (int r = 0; r < issues; r++) {
        int off = (r * 64 + ptid) * 8;
        if (off < K) cp_async_cg16(dst + off, src + off);
      }
      cp_async_mbar_arrive_noinc(&full[s]);     // ties this thread's cp.async completion
    }
  } else {
    // ---- CONSUMERS ----
    const int cw = warp - NW_PROD;              // 0..NW_CONS-1
    for (unsigned row = 0; row < my_rows; row++) {
      const int s = row % stages;
      mbar_wait(&full[s], (row / stages) & 1);
      const __nv_bfloat16 *w_row = ring + (size_t)s * K;
      float acc = 0.f;
      for (int k = (cw * 32 + lane) * 8; k < K; k += NW_CONS * 32 * 8)
        acc = dot8_smem(w_row + k, xs + k, acc);
      acc = warp_reduce(acc);
      if (lane == 0) reduce_buf[cw] = acc;
      bar_sync(1, NW_CONS * 32);                // consumers-only barrier (NOT __syncthreads)
      if (cw == 0 && lane == 0) {
        float tot = 0.f;
        for (int w = 0; w < NW_CONS; w++) tot += reduce_buf[w];
        C[n0 + row] = __float2bfloat16(tot);
      }
      bar_sync(1, NW_CONS * 32);
      mbar_arrive(&empty[s]);
    }
  }
}

// ============================ HARNESS ============================
struct Shape { int K, N; const char *tag; };

int main(int argc, char **argv) {
  cudaDeviceProp prop; CHK(cudaGetDeviceProperties(&prop, 0));
  const int grid = prop.multiProcessorCount;
  printf("# %s  SMs=%d  HBM_spec=%.0f GB/s\n", prop.name, grid, HBM_SPEC);

  std::vector<Shape> shapes = {
    {2816, 4096, "K2816 N4096  (attn o/qkv-ish)"},
    {2816, 8192, "K2816 N8192  (o-proj full)"},
    {2816, 1408, "K2816 N1408"},
    {2816,  704, "K2816 N704   (moe gate/up row)"},
    {2816, 2112, "K2816 N2112"},
    { 704, 2816, "K704  N2816  (moe down)"},
    {2112, 2816, "K2112 N2816"},
    {2816, 262144, "K2816 N262144 (lm_head)"},
  };

  printf("\n%-30s %10s | %-24s | %s\n", "shape", "uniform", "prodcons (GB/s, %HBM)", "verdict");
  printf("%-30s %10s |  S=2      S=3      S=4   |\n", "", "GB/s %HBM");

  for (auto sh : shapes) {
    const int K = sh.K, N = sh.N;
    const size_t wb = (size_t)N * K * 2;
    // Replicate weights past L2 (128 MB on this card) and cycle so every rep is a
    // cold HBM read, not an L2 hit. Target ~3x L2; cap copies and total footprint.
    int COPIES = 1;
    while ((size_t)COPIES * wb < (size_t)400 * 1024 * 1024
           && (size_t)(COPIES + 1) * wb < (size_t)4 * 1024 * 1024 * 1024
           && COPIES < 256) COPIES++;

    std::vector<__nv_bfloat16> hW(wb / 2), hx(K);
    srand(42);
    for (size_t i = 0; i < wb / 2; i++) hW[i] = __float2bfloat16((float)(rand() % 2001 - 1000) / 4000.f);
    for (int i = 0; i < K; i++)          hx[i] = __float2bfloat16((float)(rand() % 2001 - 1000) / 4000.f);

    std::vector<__nv_bfloat16 *> dW(COPIES);
    CHK(cudaMalloc(&dW[0], wb));
    CHK(cudaMemcpy(dW[0], hW.data(), wb, cudaMemcpyHostToDevice));
    for (int c = 1; c < COPIES; c++) { CHK(cudaMalloc(&dW[c], wb)); CHK(cudaMemcpy(dW[c], dW[0], wb, cudaMemcpyDeviceToDevice)); }
    __nv_bfloat16 *dx, *dC, *dCref;
    CHK(cudaMalloc(&dx, K * 2)); CHK(cudaMalloc(&dC, (size_t)N * 2)); CHK(cudaMalloc(&dCref, (size_t)N * 2));
    CHK(cudaMemcpy(dx, hx.data(), K * 2, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1; CHK(cudaEventCreate(&e0)); CHK(cudaEventCreate(&e1));
    const int WARM = 5, REP = 50;
    auto bw = [&](float ms){ return (double)wb / (ms * 1e-3) / 1e9; };

    // ---- uniform baseline ----
    size_t smem_u = (size_t)K * 2;
    if (smem_u > 48 * 1024) CHK(cudaFuncSetAttribute((void *)gemv_uniform_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_u));
    gemv_uniform_bf16<<<grid, BLOCK, smem_u>>>(dW[0], dx, dCref, N, K, 0, grid);
    CHK(cudaDeviceSynchronize());
    for (int i = 0; i < WARM; i++) gemv_uniform_bf16<<<grid, BLOCK, smem_u>>>(dW[i % COPIES], dx, dCref, N, K, 0, grid);
    CHK(cudaEventRecord(e0));
    for (int i = 0; i < REP; i++) gemv_uniform_bf16<<<grid, BLOCK, smem_u>>>(dW[i % COPIES], dx, dCref, N, K, 0, grid);
    CHK(cudaEventRecord(e1)); CHK(cudaDeviceSynchronize());
    float msu; CHK(cudaEventElapsedTime(&msu, e0, e1)); msu /= REP;
    double bwu = bw(msu);

    std::vector<__nv_bfloat16> hCref(N), hC(N);
    CHK(cudaMemcpy(hCref.data(), dCref, (size_t)N * 2, cudaMemcpyDeviceToHost));

    // ---- prodcons, sweep stages ----
    double bwp[3]; int mism[3];
    for (int si = 0; si < 3; si++) {
      int stages = si + 2;
      size_t smem_p = (size_t)K * 2 + (size_t)stages * K * 2 + 2 * MAXST * 8 + NW_CONS * 4;
      if (smem_p > 48 * 1024) CHK(cudaFuncSetAttribute((void *)gemv_prodcons_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_p));
      CHK(cudaMemset(dC, 0, (size_t)N * 2));
      gemv_prodcons_bf16<<<grid, BLOCK, smem_p>>>(dW[0], dx, dC, N, K, 0, grid, stages);
      cudaError_t le = cudaDeviceSynchronize();
      if (le != cudaSuccess) { printf("  [S=%d launch err: %s]\n", stages, cudaGetErrorString(le)); bwp[si] = 0; mism[si] = -1; continue; }
      CHK(cudaMemcpy(hC.data(), dC, (size_t)N * 2, cudaMemcpyDeviceToHost));
      int m = 0;
      for (int i = 0; i < N; i++) { float a = __bfloat162float(hC[i]), b = __bfloat162float(hCref[i]); if (fabsf(a - b) > 0.02f * fmaxf(1.f, fabsf(b))) m++; }
      mism[si] = m;
      for (int i = 0; i < WARM; i++) gemv_prodcons_bf16<<<grid, BLOCK, smem_p>>>(dW[i % COPIES], dx, dC, N, K, 0, grid, stages);
      CHK(cudaEventRecord(e0));
      for (int i = 0; i < REP; i++) gemv_prodcons_bf16<<<grid, BLOCK, smem_p>>>(dW[i % COPIES], dx, dC, N, K, 0, grid, stages);
      CHK(cudaEventRecord(e1)); CHK(cudaDeviceSynchronize());
      float msp; CHK(cudaEventElapsedTime(&msp, e0, e1)); msp /= REP;
      bwp[si] = bw(msp);
    }

    double best = fmax(bwp[0], fmax(bwp[1], bwp[2]));
    const char *verd = (best > bwu * 1.02) ? "PRODCONS WIN" : (best > bwu * 0.98 ? "tie" : "uniform wins");
    printf("%-30s %6.0f %4.0f%% | %5.0f/%2.0f%% %5.0f/%2.0f%% %5.0f/%2.0f%% | %s%s\n",
           sh.tag, bwu, 100 * bwu / HBM_SPEC,
           bwp[0], 100 * bwp[0] / HBM_SPEC, bwp[1], 100 * bwp[1] / HBM_SPEC, bwp[2], 100 * bwp[2] / HBM_SPEC,
           verd, (mism[0] + mism[1] + mism[2] > 0) ? "  [MISMATCH!]" : "");

    for (int c = 0; c < COPIES; c++) cudaFree(dW[c]);
    cudaFree(dx); cudaFree(dC); cudaFree(dCref);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }
  return 0;
}

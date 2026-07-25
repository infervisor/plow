// Warp specialization on a memory-bound fp8 decode GEMV (M=1), sm_120a / RTX 5090.
//
// Question: does dedicating warps to data movement (cp.async.cg or TMA cp.async.bulk)
// with mbarrier producer/consumer coordination beat the uniform "every warp alternates
// load and compute" structure on a 1-D streaming weight read?
//
// Contract shared by ALL variants (so results are bit-identical):
//   * lane l of the warp owning output row n accumulates k = t*KTILE + q*512 + l*16 .. +15
//     in strictly increasing k, 16 FFMA per 16 bytes, fp8 e4m3 widened on load.
//   * per-output-channel f32 scale applied ONCE in the epilogue, after the shfl reduce.
// Warp specialization moves DATA, not MATH -> outputs must compare bit-identical.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <cuda_fp8.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

__device__ __forceinline__ float2 fp8x2_to_float2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
  return __half22float2(*reinterpret_cast<__half2 *>(&h));
}

// 16 fp8 weights * 16 activations, fixed FFMA order. THE numeric kernel; identical everywhere.
__device__ __forceinline__ float acc16(const uint4 &w, const float *xp, float acc) {
  const uint16_t *wp = (const uint16_t *)&w;
  float xv[16];
  *(float4 *)(xv + 0)  = *(const float4 *)(xp + 0);
  *(float4 *)(xv + 4)  = *(const float4 *)(xp + 4);
  *(float4 *)(xv + 8)  = *(const float4 *)(xp + 8);
  *(float4 *)(xv + 12) = *(const float4 *)(xp + 12);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float2 f = fp8x2_to_float2(wp[j]);
    acc = fmaf(f.x, xv[2 * j], acc);
    acc = fmaf(f.y, xv[2 * j + 1], acc);
  }
  return acc;
}
__device__ __forceinline__ float warp_reduce(float acc) {
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;
}

// ---------------- mbarrier helpers ----------------
__device__ __forceinline__ uint32_t sptr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(void *b, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sptr(b)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(void *b) {
  asm volatile("{ .reg .b64 s; mbarrier.arrive.shared::cta.b64 s, [%0]; }" ::"r"(sptr(b)));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(void *b, uint32_t bytes) {
  asm volatile("{ .reg .b64 s; mbarrier.arrive.expect_tx.shared::cta.b64 s, [%0], %1; }" ::"r"(
      sptr(b)), "r"(bytes));
}
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(void *b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(sptr(b)));
}
__device__ __forceinline__ void mbar_wait(void *b, uint32_t parity) {
  asm volatile(
      "{ .reg .pred P; WSPIN: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; "
      "@!P bra WSPIN; }" ::"r"(sptr(b)),
      "r"(parity));
}
__device__ __forceinline__ void cp_async_cg16(void *dst, const void *src) {
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(sptr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_bulk(void *dst, const void *src, uint32_t bytes, void *b) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(sptr(dst)),
      "l"(src), "r"(bytes), "r"(sptr(b)));
}

// ---------------- (a) UNIFORM: every warp loads and computes ----------------
template <int NW>
__global__ __launch_bounds__(NW * 32) void gemv_uniform(const uint8_t *__restrict__ W,
                                                        const float *__restrict__ x,
                                                        const float *__restrict__ scale,
                                                        float *__restrict__ y, int N, int K,
                                                        int ncopy) {
  extern __shared__ __align__(16) char smem[];
  float *xs = (float *)smem;
  for (int i = threadIdx.x; i < K; i += NW * 32) xs[i] = x[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int steps = K / 512;
  const size_t wb = (size_t)N * K;
  for (int cp = 0; cp < ncopy; ++cp)
  for (int n = blockIdx.x * NW + warp; n < N; n += gridDim.x * NW) {
    const uint8_t *row = W + cp * wb + (size_t)n * K;
    float acc = 0.f;
#pragma unroll 1
    for (int s = 0; s < steps; ++s) {
      const int off = s * 512 + lane * 16;
      uint4 w = *(const uint4 *)(row + off);
      acc = acc16(w, xs + off, acc);
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[n] = acc * scale[n];
  }
}


// (a') UNIFORM with 8-byte loads (the incumbent's uint2 inner loop width).
template <int NW>
__global__ __launch_bounds__(NW * 32) void gemv_uniform8(const uint8_t *__restrict__ W,
                                                         const float *__restrict__ x,
                                                         const float *__restrict__ scale,
                                                         float *__restrict__ y, int N, int K,
                                                         int ncopy) {
  extern __shared__ __align__(16) char smem[];
  float *xs = (float *)smem;
  for (int i = threadIdx.x; i < K; i += NW * 32) xs[i] = x[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int steps = K / 256;
  const size_t wb = (size_t)N * K;
  for (int cp = 0; cp < ncopy; ++cp)
  for (int n = blockIdx.x * NW + warp; n < N; n += gridDim.x * NW) {
    const uint8_t *row = W + cp * wb + (size_t)n * K;
    float acc = 0.f;
#pragma unroll 1
    for (int s = 0; s < steps; ++s) {
      const int off = s * 256 + lane * 8;
      uint2 w = *(const uint2 *)(row + off);
      const uint16_t *wp = (const uint16_t *)&w;
      float4 x0 = *(const float4 *)(xs + off), x1 = *(const float4 *)(xs + off + 4);
      const float *xv0 = (const float *)&x0, *xv1 = (const float *)&x1;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = fp8x2_to_float2(wp[j]);
        const float *xv = (j < 2) ? xv0 : xv1;
        int b = (2 * j) & 3;
        acc = fmaf(f.x, xv[b], acc);
        acc = fmaf(f.y, xv[b + 1], acc);
      }
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[n] = acc * scale[n];
  }
}

// ---------------- (b) SPECIALIZED: NLOAD loader warps + NCOMP compute warps ----------------
// MODE 0 = cp.async.cg (all loader threads issue 16B copies)
// MODE 1 = TMA cp.async.bulk (one elected thread issues NCOMP 1-D bulk copies)
// NOSYNC = negative control: barrier waits removed (must produce wrong answers)
template <int NLOAD, int NCOMP, int STAGES, int KTILE, int MODE, bool NOSYNC = false>
__global__ __launch_bounds__((NLOAD + NCOMP) * 32) void gemv_spec(const uint8_t *__restrict__ W,
                                                                  const float *__restrict__ x,
                                                                  const float *__restrict__ scale,
                                                                  float *__restrict__ y, int N,
                                                                  int K, int ncopy) {
  constexpr int NW = NLOAD + NCOMP;
  constexpr int STAGE_B = NCOMP * KTILE;
  extern __shared__ __align__(16) char smem[];
  float *xs = (float *)smem;                                   // K floats
  uint8_t *ring = (uint8_t *)(smem + ((K * 4 + 15) & ~15));    // STAGES * STAGE_B
  uint64_t *bar = (uint64_t *)(ring + STAGES * STAGE_B);       // full[STAGES], empty[STAGES]

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&bar[s], MODE == 0 ? NLOAD * 32 : 1);  // full
      mbar_init(&bar[STAGES + s], NCOMP * 32);         // empty
    }
  }
  for (int i = tid; i < K; i += NW * 32) xs[i] = x[i];
  __syncthreads();

  const int tiles = K / KTILE;
  const int ngroups = N / NCOMP;
  const size_t wb = (size_t)N * K;

  if (warp < NLOAD) {  // ---------------- loader ----------------
    const int ltid = tid;
    int i = 0;
    for (int cp = 0; cp < ncopy; ++cp)
    for (int g = blockIdx.x; g < ngroups; g += gridDim.x) {
      for (int t = 0; t < tiles; ++t, ++i) {
        const int s = i % STAGES, r = i / STAGES;
        if (!NOSYNC && r > 0) mbar_wait(&bar[STAGES + s], (uint32_t)((r - 1) & 1));
        if (MODE == 0) {
          constexpr int UNITS = NCOMP * KTILE / 16;
#pragma unroll
          for (int u = ltid; u < UNITS; u += NLOAD * 32) {
            const int c = u / (KTILE / 16), j = u % (KTILE / 16);
            cp_async_cg16(ring + s * STAGE_B + c * KTILE + j * 16,
                          W + cp * wb + (size_t)(g * NCOMP + c) * K + t * KTILE + j * 16);
          }
          cp_async_mbar_arrive_noinc(&bar[s]);
        } else {
          if (ltid == 0) {
            mbar_arrive_expect_tx(&bar[s], STAGE_B);
#pragma unroll
            for (int c = 0; c < NCOMP; ++c)
              cp_async_bulk(ring + s * STAGE_B + c * KTILE,
                            W + cp * wb + (size_t)(g * NCOMP + c) * K + t * KTILE, KTILE, &bar[s]);
          }
        }
      }
    }
  } else {  // ---------------- compute ----------------
    const int c = warp - NLOAD;
    int i = 0;
    for (int cp = 0; cp < ncopy; ++cp)
    for (int g = blockIdx.x; g < ngroups; g += gridDim.x) {
      float acc = 0.f;
      for (int t = 0; t < tiles; ++t, ++i) {
        const int s = i % STAGES, r = i / STAGES;
        if (!NOSYNC) mbar_wait(&bar[s], (uint32_t)(r & 1));
        const uint8_t *p = ring + s * STAGE_B + c * KTILE;
#pragma unroll
        for (int q = 0; q < KTILE / 512; ++q) {
          const int off = q * 512 + lane * 16;
          uint4 w = *(const uint4 *)(p + off);
          acc = acc16(w, xs + t * KTILE + off, acc);
        }
        __syncwarp();
        mbar_arrive(&bar[STAGES + s]);
      }
      acc = warp_reduce(acc);
      const int n = g * NCOMP + c;
      if (lane == 0) y[n] = acc * scale[n];
    }
  }
}

// ---------------- harness ----------------
struct Variant {
  std::string name;
  int threads;
  int smem_extra;  // bytes on top of K*4
  std::function<void(int, int, size_t, const uint8_t *, const float *, const float *, float *, int,
                     int, int)>
      launch;
  const void *fn;
  int ncomp;
};

static int g_N, g_K;

template <int NW>
void add_uniform(std::vector<Variant> &v) {
  char nm[64];
  snprintf(nm, sizeof nm, "uniform NW=%d", NW);
  v.push_back({nm, NW * 32, 0,
               [](int grid, int thr, size_t sm, const uint8_t *W, const float *x, const float *s,
                  float *y, int N, int K, int nc) { gemv_uniform<NW><<<grid, thr, sm>>>(W, x, s, y, N, K, nc); },
               (const void *)gemv_uniform<NW>, NW});
}
template <int NW>
void add_uniform8(std::vector<Variant> &v) {
  char nm[64];
  snprintf(nm, sizeof nm, "uniform8B NW=%d", NW);
  v.push_back({nm, NW * 32, 0,
               [](int grid, int thr, size_t sm, const uint8_t *W, const float *x, const float *s,
                  float *y, int N, int K, int nc) { gemv_uniform8<NW><<<grid, thr, sm>>>(W, x, s, y, N, K, nc); },
               (const void *)gemv_uniform8<NW>, NW});
}
template <int NL, int NC, int ST, int KT, int MODE, bool NS = false>
void add_spec(std::vector<Variant> &v) {
  char nm[80];
  snprintf(nm, sizeof nm, "%s L=%d C=%d st=%d kt=%d%s", MODE == 0 ? "cp.async" : "TMA-bulk", NL, NC,
           ST, KT, NS ? " NOSYNC" : "");
  v.push_back({nm, (NL + NC) * 32, ST * NC * KT + 2 * ST * 8,
               [](int grid, int thr, size_t sm, const uint8_t *W, const float *x, const float *s,
                  float *y, int N, int K, int nc) {
                 gemv_spec<NL, NC, ST, KT, MODE, NS><<<grid, thr, sm>>>(W, x, s, y, N, K, nc);
               },
               (const void *)gemv_spec<NL, NC, ST, KT, MODE, NS>, NC});
}

int main(int argc, char **argv) {
  int shape_sel = argc > 1 ? atoi(argv[1]) : -1;
  struct Shape { const char *nm; int N, K; int copies; };
  Shape shapes[] = {{"q_proj  [4096,2560]", 4096, 2560, 32},
                    {"gateup  [9728,2560]", 9728, 2560, 16},
                    {"down    [2560,9728]", 2560, 9728, 16}};

  cudaDeviceProp prop;
  CHK(cudaGetDeviceProperties(&prop, 0));
  printf("# %s  SMs=%d  L2=%.1f MB  smemPerBlockOptin=%zu\n", prop.name, prop.multiProcessorCount,
         prop.l2CacheSize / 1e6, (size_t)prop.sharedMemPerBlockOptin);

  for (int si = 0; si < 3; ++si) {
    if (shape_sel >= 0 && si != shape_sel) continue;
    const char *cm = getenv("COPY_MUL");
    const int N = shapes[si].N, K = shapes[si].K, COPIES = shapes[si].copies * (cm ? atoi(cm) : 1);
    g_N = N; g_K = K;
    const size_t wb = (size_t)N * K;
    printf("\n=== %s   weights %.2f MiB x %d copies = %.0f MiB working set ===\n", shapes[si].nm,
           wb / 1048576.0, COPIES, wb * COPIES / 1048576.0);

    // host data
    std::vector<uint8_t> hW(wb);
    std::vector<float> hx(K), hS(N);
    srand(12345);
    for (size_t i = 0; i < wb; ++i) {
      float v = (float)(rand() % 2001 - 1000) / 1000.0f;
      __nv_fp8_e4m3 q(v);
      hW[i] = *(uint8_t *)&q;
    }
    for (int i = 0; i < K; ++i) hx[i] = (float)(rand() % 2001 - 1000) / 1000.0f;
    for (int i = 0; i < N; ++i) hS[i] = 0.01f + (float)(rand() % 100) / 10000.0f;

    uint8_t *dW; float *dx, *dS, *dy, *dy_ref;
    CHK(cudaMalloc(&dW, wb));
    CHK(cudaMalloc(&dx, K * 4)); CHK(cudaMalloc(&dS, N * 4));
    CHK(cudaMalloc(&dy, N * 4)); CHK(cudaMalloc(&dy_ref, N * 4));
    CHK(cudaMemcpy(dW, hW.data(), wb, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dx, hx.data(), K * 4, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dS, hS.data(), N * 4, cudaMemcpyHostToDevice));
    uint8_t *dWbig;  // COPIES identical replicas, contiguous: one launch streams all of them
    CHK(cudaMalloc(&dWbig, wb * COPIES));
    for (int c = 0; c < COPIES; ++c)
      CHK(cudaMemcpy(dWbig + (size_t)c * wb, dW, wb, cudaMemcpyDeviceToDevice));

    std::vector<Variant> V;
    add_uniform<4>(V);
    add_uniform<8>(V);
    add_uniform<16>(V);
    add_uniform8<8>(V);
    add_uniform8<4>(V);
    add_spec<1, 7, 8, 512, 0>(V);
    add_spec<2, 6, 8, 512, 0>(V);
    add_spec<4, 4, 8, 512, 0>(V);
    add_spec<6, 2, 8, 512, 0>(V);
    add_spec<1, 3, 8, 512, 0>(V);
    add_spec<2, 2, 8, 512, 0>(V);
    add_spec<1, 1, 8, 512, 0>(V);
    add_spec<4, 4, 4, 512, 0>(V);
    add_spec<4, 4, 16, 512, 0>(V);
    add_spec<1, 7, 8, 512, 1>(V);
    add_spec<1, 4, 8, 512, 1>(V);
    add_spec<1, 3, 8, 512, 1>(V);
    add_spec<1, 8, 4, 512, 1>(V);
    if (K == 2560) {
      add_spec<4, 4, 4, 2560, 0>(V);
      add_spec<1, 4, 4, 2560, 1>(V);
    }
    add_spec<4, 4, 8, 512, 0, true>(V);  // negative control

    // reference: uniform NW=8 at a fixed grid
    std::vector<float> hy(N), href(N);
    printf("%-28s %6s %5s %8s %9s %7s %8s %s\n", "variant", "grid", "b/SM", "ms", "GB/s", "%peak",
           "ms/tok", "bitcmp");
    bool have_ref = false;
    for (auto &v : V) {
      cudaFuncAttributes fa;
      CHK(cudaFuncGetAttributes(&fa, v.fn));
      size_t sm = (size_t)K * 4 + v.smem_extra;
      sm = (sm + 15) & ~15ull;
      if (sm > 48 * 1024)
        CHK(cudaFuncSetAttribute(v.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
      int occ = 0;
      CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, v.fn, v.threads, sm));
      if (N % v.ncomp) { printf("%-28s  SKIP (N%%%d != 0)\n", v.name.c_str(), v.ncomp); continue; }
      if (occ == 0) { printf("%-28s  OCCUPANCY 0 (smem %zu regs %d)\n", v.name.c_str(), sm, fa.numRegs); continue; }
      // grid sweep around occ*SMs
      int base = occ * prop.multiProcessorCount;
      int grids[] = {base / 2, base, base * 2, base * 4};
      for (int gi = 0; gi < 4; ++gi) {
        int grid = grids[gi];
        if (grid < 1) continue;
        // warm + correctness
        CHK(cudaMemset(dy, 0, N * 4));
        v.launch(grid, v.threads, sm, dW, dx, dS, dy, N, K, 1);
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(hy.data(), dy, N * 4, cudaMemcpyDeviceToHost));
        int mism = 0;
        if (!have_ref) { href = hy; have_ref = true; }
        else for (int i = 0; i < N; ++i) if (memcmp(&hy[i], &href[i], 4)) ++mism;

        cudaEvent_t ev0, ev1; CHK(cudaEventCreate(&ev0)); CHK(cudaEventCreate(&ev1));
        const int REP = 20;
        for (int i = 0; i < 3; ++i) v.launch(grid, v.threads, sm, dWbig, dx, dS, dy, N, K, COPIES);
        CHK(cudaDeviceSynchronize());
        CHK(cudaEventRecord(ev0));
        for (int i = 0; i < REP; ++i) v.launch(grid, v.threads, sm, dWbig, dx, dS, dy, N, K, COPIES);
        CHK(cudaEventRecord(ev1));
        CHK(cudaDeviceSynchronize());
        float ms; CHK(cudaEventElapsedTime(&ms, ev0, ev1)); ms /= REP;
        double gbs = (double)wb * COPIES / (ms * 1e-3) / 1e9;
        ms /= COPIES;  // report per-shape-pass time
        char lbl[96];
        snprintf(lbl, sizeof lbl, "%s%s", v.name.c_str(), gi == 1 ? " *" : "");
        printf("%-28s %6d %5d %8.4f %9.1f %6.1f%% %8.3f %s\n", lbl, grid, occ, ms, gbs,
               100.0 * gbs / 1673.0, 4.0e3 / gbs, mism ? "MISMATCH" : "identical");
        if (mism) printf("      ^ %d/%d rows differ from reference\n", mism, N);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
      }
      printf("%-28s regs=%d spills(st/ld)=%d/%d smem=%zu\n", "   ptxas:", fa.numRegs,
             (int)fa.localSizeBytes, (int)fa.localSizeBytes, sm);
    }
    cudaFree(dWbig); cudaFree(dW);
    cudaFree(dx); cudaFree(dS); cudaFree(dy); cudaFree(dy_ref);
  }
  return 0;
}

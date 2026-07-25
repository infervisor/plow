// hbm_ceiling_h100.cu — E1 of the 26B/H100 beat-vLLM campaign.
//
// Question: plow's decode GEMV runs at 23% of HBM peak at 1 block/SM. Is that
// because a *pure streaming read* cannot go faster at that occupancy (memory-level
// parallelism starvation — occupancy is then the only lever), or because the GEMV
// arm itself wastes bandwidth (then the arm is the lever and occupancy is a
// sideshow)?
//
// Measures pure read BW of a large HBM-resident buffer as a function of
//   blocks/SM  x  outstanding 16B loads per thread (unroll depth).
// Nothing but loads + a cheap accumulate the compiler cannot remove.
//
//   nvcc -arch=sm_90a -O3 -o hbm hbm_ceiling_h100.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CHK(x)                                                                        \
    do {                                                                              \
        cudaError_t e = (x);                                                          \
        if (e != cudaSuccess) {                                                       \
            printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__);                 \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

struct v4 {
    float4 v;
};

// Grid-stride streaming read. UN independent 16B loads in flight per thread per
// iteration; the accumulate is float4-add so the loads cannot be dead-coded.
template <int UN>
__global__ __launch_bounds__(256) void readbw(const float4* __restrict__ src, size_t n4,
                                              float* __restrict__ sink) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = tid; i < n4; i += stride * UN) {
        float4 t[UN];
#pragma unroll
        for (int u = 0; u < UN; u++) {
            const size_t idx = i + (size_t)u * stride;
            t[u] = (idx < n4) ? src[idx] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int u = 0; u < UN; u++) {
            acc.x += t[u].x;
            acc.y += t[u].y;
            acc.z += t[u].z;
            acc.w += t[u].w;
        }
    }
    if (acc.x == 1234.5f) sink[0] = acc.x + acc.y + acc.z + acc.w;  // never true
}

static float* g_flush = nullptr;
static size_t g_flush_n = 0;
static void flushL2() { cudaMemsetAsync(g_flush, 0, g_flush_n, 0); }

template <int UN>
static double run(const float4* d, size_t n4, float* sink, int nblk, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) readbw<UN><<<nblk, 256>>>(d, n4, sink);
    cudaDeviceSynchronize();
    flushL2();
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) readbw<UN><<<nblk, 256>>>(d, n4, sink);
    cudaEventRecord(b);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    const double bytes = (double)n4 * 16.0 * iters;
    return bytes / (ms * 1e-3) / 1e9;  // GB/s
}

int main() {
    cudaDeviceProp p;
    CHK(cudaGetDeviceProperties(&p, 0));
    const int nSM = p.multiProcessorCount;
    int memClkKHz = 0, busBits = 0;
    CHK(cudaDeviceGetAttribute(&memClkKHz, cudaDevAttrMemoryClockRate, 0));
    CHK(cudaDeviceGetAttribute(&busBits, cudaDevAttrGlobalMemoryBusWidth, 0));
    printf("# %s  SMs=%d  L2=%.1f MB  memClk=%d MHz  bus=%d bit\n", p.name, nSM,
           p.l2CacheSize / 1048576.0, memClkKHz / 1000, busBits);
    const double spec = 2.0 * (double)memClkKHz * 1e3 * (busBits / 8) / 1e9;
    printf("# spec peak (2*memClk*bus/8) = %.0f GB/s\n", spec);

    // 1 GiB buffer — 16x L2, so every read is a genuine HBM read.
    const size_t bytes = 1ull << 30;
    const size_t n4 = bytes / 16;
    float4* d = nullptr;
    float* sink = nullptr;
    CHK(cudaMalloc(&d, bytes));
    CHK(cudaMalloc(&sink, 1024));
    CHK(cudaMemset(d, 1, bytes));
    g_flush_n = 128ull << 20;
    CHK(cudaMalloc(&g_flush, g_flush_n));

    const int occs[] = {1, 2, 3, 4, 6, 8};
    printf("\n# pure streaming read, 1 GiB, 256 thr/blk. GB/s (%% of spec peak)\n");
    printf("%-10s", "blk/SM");
    for (int un : {1, 2, 4, 8}) printf("   UN=%-14d", un);
    printf("\n");
    double best = 0;
    for (int occ : occs) {
        const int nblk = nSM * occ;
        printf("%-10d", occ);
        double r1 = run<1>(d, n4, sink, nblk, 20);
        double r2 = run<2>(d, n4, sink, nblk, 20);
        double r4 = run<4>(d, n4, sink, nblk, 20);
        double r8 = run<8>(d, n4, sink, nblk, 20);
        for (double r : {r1, r2, r4, r8}) {
            printf("   %7.0f (%4.1f%%)", r, 100.0 * r / spec);
            if (r > best) best = r;
        }
        printf("\n");
    }
    printf("\n# BEST achievable read BW = %.0f GB/s (%.1f%% of spec)\n", best, 100.0 * best / spec);
    printf("# 26B decode reads 7.63 GB/token -> floor = %.3f ms/token at this BW\n",
           7.63e9 / (best * 1e9) * 1e3);
    printf("# vLLM measured 4.833 ms/token @ctx1024 -> vLLM achieves %.0f GB/s (%.1f%% of best)\n",
           7.63e9 / 4.833e-3 / 1e9, 100.0 * (7.63e9 / 4.833e-3 / 1e9) / best);
    return 0;
}

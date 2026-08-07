// tma_uni_gemm_ab.cu — A/B the IN-TREE sm_90a prefill GEMM bodies outside the megakernel:
//   d_gemm_sm90      (shipped cp.async + wgmma, 2 consumer warpgroups)
//   d_gemm_sm90_tma  (uniform-TMA rewrite: same 2 consumer warpgroups, one elected thread
//                     issues cp.async.bulk.tensor on an mbarrier ring)
//
// Motivation: in-model (Gemma-4-12B serve, T=4096) the TMA arm measured ~1.78x SLOWER than
// the cp.async body (1332 vs 750 ms TTFT) while every probe said TMA staging should WIN
// (tma_ws_moe_group.cu: 1.6-1.75x). This harness runs the exact production bodies at the
// exact per-op shapes the 12B prefill emits, grid 132 x 256 threads, occ-1, so whichever
// way it lands it attributes the regression to the KERNEL vs the INTEGRATION.
//
// Build (executables need the -gencode form; see experiments/README.md):
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common \
//     -I runtime/nvidia -include cstdint runtime/nvidia/experiments/tma_uni_gemm_ab.cu \
//     -lcuda -o /tmp/tma_uni_ab
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define PLOW_NV_HOPPER 1
#define PLOW_NV_THREADS 256u
#define PLOW_NV_TMA_GEMM 1
#define PGM90_FORK_GLU 0
// Stand-ins for the op_gemm.cuh context the header normally rides in.
#define PGM_ARENA_BF16 (128 * 1024)
#define PLOW_ACT_SILU_ 0u
__device__ __forceinline__ float act_silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float act_gelu_tanh(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
#include "op_gemm_sm90.cuh"

using bf16 = __nv_bfloat16;

__global__ __launch_bounds__(256, 1) void k_cp(bf16* C, const bf16* A, const bf16* B, unsigned m,
                                               unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    d_gemm_sm90(C, A, B, m, n, k, 0, blockIdx.x, gridDim.x, arena);
}
__global__ __launch_bounds__(256, 1) void k_tma(bf16* C, const void* mA, const void* mB,
                                                unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    d_gemm_sm90_tma(C, mA, mB, m, n, k, 0, blockIdx.x, gridDim.x, arena);
}

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__);                             \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#define CKD(x)                                                                                     \
    do {                                                                                           \
        CUresult e_ = (x);                                                                         \
        if (e_ != CUDA_SUCCESS) {                                                                  \
            printf("CU err %d @%d\n", (int)e_, __LINE__);                                          \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static void make_map(CUtensorMap* mp, void* base, int rows, int K) {
    uint64_t gd[2] = {(uint64_t)K, (uint64_t)rows};
    uint64_t gs[1] = {(uint64_t)K * 2};
    uint32_t bd[2] = {64u, 128u};
    uint32_t es[2] = {1, 1};
    memset(mp, 0, sizeof(*mp));
    CKD(cuTensorMapEncodeTiled(mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

static uint32_t xs = 0x1234567u;
static float frand() {
    xs ^= xs << 13;
    xs ^= xs >> 17;
    xs ^= xs << 5;
    return ((xs >> 8) * (1.0f / 8388608.0f)) - 1.0f;
}

int main() {
    int grid = 132;
    const unsigned SMEM = 160 * 1024; // bytes; covers cp (97 KiB) and TMA (130 KiB) claims
    CK(cudaFuncSetAttribute(k_cp, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CK(cudaFuncSetAttribute(k_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

    struct Shape {
        unsigned m, n, k;
        const char* what;
    } shapes[] = {
        {1024, 4096, 3840, "q_proj  chunk"},
        {1024, 2048, 3840, "kv_proj chunk"},
        {1024, 3840, 4096, "o_proj  chunk"},
        {1024, 3840, 15360, "down    chunk"},
        {128, 4096, 3840, "q_proj  T=128"},
    };
    printf("grid=%d threads=256 smem=%u B  (in-tree bodies, occ-1)\n", grid, SMEM);
    printf("%-14s %10s %10s %8s\n", "shape", "cp.async", "tma-uni", "tma/cp");
    for (auto& s : shapes) {
        size_t am = (size_t)s.m * s.k, bm = (size_t)s.n * s.k, cm = (size_t)s.m * s.n;
        bf16 *A, *B, *C;
        CK(cudaMalloc(&A, am * 2));
        CK(cudaMalloc(&B, bm * 2));
        CK(cudaMalloc(&C, cm * 2));
        std::vector<bf16> h(am > bm ? am : bm);
        for (size_t i = 0; i < am; i++) h[i] = __float2bfloat16(frand());
        CK(cudaMemcpy(A, h.data(), am * 2, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < bm; i++) h[i] = __float2bfloat16(frand());
        CK(cudaMemcpy(B, h.data(), bm * 2, cudaMemcpyHostToDevice));
        CUtensorMap mA, mB;
        make_map(&mA, A, s.m, s.k);
        make_map(&mB, B, s.n, s.k);
        CUtensorMap *dA, *dB;
        CK(cudaMalloc(&dA, 256));
        dB = dA + 1;
        CK(cudaMemcpy(dA, &mA, 128, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, &mB, 128, cudaMemcpyHostToDevice));

        auto bench = [&](bool tma) -> float {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            for (int w = 0; w < 3; w++) {
                if (tma)
                    k_tma<<<grid, 256, SMEM>>>(C, dA, dB, s.m, s.n, s.k);
                else
                    k_cp<<<grid, 256, SMEM>>>(C, A, B, s.m, s.n, s.k);
            }
            CK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            const int it = 20;
            for (int i = 0; i < it; i++) {
                if (tma)
                    k_tma<<<grid, 256, SMEM>>>(C, dA, dB, s.m, s.n, s.k);
                else
                    k_cp<<<grid, 256, SMEM>>>(C, A, B, s.m, s.n, s.k);
            }
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            CK(cudaGetLastError());
            return ms / it;
        };
        // interleaved, min of 3 rounds
        float cp = 1e9f, tm = 1e9f;
        for (int r = 0; r < 3; r++) {
            float a = bench(false), b = bench(true);
            if (a < cp) cp = a;
            if (b < tm) tm = b;
        }
        double fl = 2.0 * s.m * s.n * s.k;
        printf("%-14s %7.3f ms %7.3f ms %7.2fx   (%5.1f vs %5.1f TF/s)\n", s.what, cp, tm,
               tm / cp, fl / (cp * 1e9), fl / (tm * 1e9));
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaFree(dA);
    }
    return 0;
}

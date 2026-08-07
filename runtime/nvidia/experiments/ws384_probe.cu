// ws384_probe.cu — standalone T31 384-thread producer/consumer GEMM body, fp8 + bf16, at the
// 12B shapes. Establishes the body's own ceiling so in-model deltas can be attributed to the
// interpreter context (launch floors, gates) rather than the mainloop.
//
// Build:
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common \
//     -I runtime/nvidia -include cstdint runtime/nvidia/experiments/ws384_probe.cu \
//     -lcuda -o /tmp/ws384_probe
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define PLOW_NV_HOPPER 1
#define PLOW_NV_THREADS 256u
#define PLOW_NV_TMA_GEMM 1
#define PLOW_NV_W8A8 1
#define PGM90_FP8_PROMOTE 0
#define PGM90_FORK_GLU 0
#define PGM90_UNI_BN256 1
#define PLOW_NV_SEG_GEMM 1
#define PLOW_NV_SEGMENTS 1
#define PLOW_NV_SEG_WS384 1
#define PGM90_TMA_STAGES 3
#ifdef PROBE_SMEPI
#define PGM90_WS384_SMEPI 1
#endif
#define PGM_ARENA_BF16 (128 * 1024)
#define PLOW_ACT_SILU_ 0u
__device__ __forceinline__ float act_silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float act_gelu_tanh(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
#include "op_gemm_sm90.cuh"

using bf16 = __nv_bfloat16;

template <bool E4M3>
__global__ __maxnreg__(160) void k_ws(bf16* C, const void* mA, const void* mB, const float* as,
                                      const float* ws, unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        d_gemm_sm90_tma_ws384_role<true, E4M3>(C, mA, mB, as, ws, m, n, k, 0, blockIdx.x,
                                               gridDim.x, arena);
    } else {
        sm90_reg_inc(224);
        d_gemm_sm90_tma_ws384_role<false, E4M3>(C, mA, mB, as, ws, m, n, k, 0, blockIdx.x,
                                                gridDim.x, arena);
    }
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

static void make_map(CUtensorMap* mp, void* base, int rows, int K, bool e4m3) {
    uint64_t gd[2] = {(uint64_t)K, (uint64_t)rows};
    uint64_t gs[1] = {(uint64_t)K * (e4m3 ? 1 : 2)};
    uint32_t bd[2] = {e4m3 ? 128u : 64u, 128u};
    uint32_t es[2] = {1, 1};
    memset(mp, 0, sizeof(*mp));
    CKD(cuTensorMapEncodeTiled(mp,
                               e4m3 ? CU_TENSOR_MAP_DATA_TYPE_UINT8
                                    : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               2, base, gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main() {
    const int grid = 132;
    const unsigned SMEM = 200 * 1024;
    CK(cudaFuncSetAttribute(k_ws<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CK(cudaFuncSetAttribute(k_ws<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

    struct Shape {
        unsigned m, n, k;
        const char* what;
    } shapes[] = {
        {4096, 4096, 3840, "q_proj"},
        {4096, 3840, 4096, "o_proj"},
        {4096, 15360, 3840, "gate/up"},
        {4096, 3840, 15360, "down"},
    };
    printf("ws384 standalone, grid=%d x 384thr, smem=%u\n%-10s %10s %10s\n", grid, SMEM, "shape",
           "fp8", "bf16");
    for (auto& s : shapes) {
        size_t am = (size_t)s.m * s.k, bm = (size_t)s.n * s.k, cm = (size_t)s.m * s.n;
        void *A8, *B8, *Ab, *Bb;
        bf16* C;
        float *as, *ws;
        CK(cudaMalloc(&A8, am));
        CK(cudaMalloc(&B8, bm));
        CK(cudaMalloc(&Ab, am * 2));
        CK(cudaMalloc(&Bb, bm * 2));
        CK(cudaMalloc(&C, cm * 2));
        CK(cudaMalloc(&as, s.m * 4));
        CK(cudaMalloc(&ws, s.n * 4));
        CK(cudaMemset(A8, 0x3c, am));
        CK(cudaMemset(B8, 0x3c, bm));
        CK(cudaMemset(Ab, 0x3c, am * 2));
        CK(cudaMemset(Bb, 0x3c, bm * 2));
        std::vector<float> one(s.m > s.n ? s.m : s.n, 1.0f);
        CK(cudaMemcpy(as, one.data(), s.m * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ws, one.data(), s.n * 4, cudaMemcpyHostToDevice));
        CUtensorMap m8a, m8b, mba, mbb;
        make_map(&m8a, A8, s.m, s.k, true);
        make_map(&m8b, B8, s.n, s.k, true);
        make_map(&mba, Ab, s.m, s.k, false);
        make_map(&mbb, Bb, s.n, s.k, false);
        CUtensorMap* d;
        CK(cudaMalloc(&d, 512));
        CK(cudaMemcpy(d + 0, &m8a, 128, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d + 1, &m8b, 128, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d + 2, &mba, 128, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d + 3, &mbb, 128, cudaMemcpyHostToDevice));

        auto bench = [&](bool fp8) -> float {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            for (int w = 0; w < 3; w++) {
                if (fp8)
                    k_ws<true><<<grid, 384, SMEM>>>(C, d + 0, d + 1, as, ws, s.m, s.n, s.k);
                else
                    k_ws<false><<<grid, 384, SMEM>>>(C, d + 2, d + 3, nullptr, nullptr, s.m,
                                                     s.n, s.k);
            }
            CK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            const int it = 10;
            for (int i = 0; i < it; i++) {
                if (fp8)
                    k_ws<true><<<grid, 384, SMEM>>>(C, d + 0, d + 1, as, ws, s.m, s.n, s.k);
                else
                    k_ws<false><<<grid, 384, SMEM>>>(C, d + 2, d + 3, nullptr, nullptr, s.m,
                                                     s.n, s.k);
            }
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            CK(cudaGetLastError());
            return ms / it;
        };
        float a = 1e9f, b = 1e9f;
        for (int r = 0; r < 3; r++) {
            float x = bench(true), y = bench(false);
            if (x < a) a = x;
            if (y < b) b = y;
        }
        double fl = 2.0 * s.m * s.n * s.k;
        printf("%-10s %6.1f TF/s %6.1f TF/s\n", s.what, fl / (a * 1e9), fl / (b * 1e9));
        cudaFree(A8); cudaFree(B8); cudaFree(Ab); cudaFree(Bb); cudaFree(C);
        cudaFree(as); cudaFree(ws); cudaFree(d);
    }
    return 0;
}

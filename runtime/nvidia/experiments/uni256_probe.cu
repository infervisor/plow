// uni256_probe.cu — A/B the T15 uniform m128n256 fp8 body against the n128 uniform w8a8
// body, standalone at the exact 12B chunk shapes, grid 132 x 256, occ-1. Run under ncu to
// attribute the remaining GEMM gap (in-model ~606 TF/s vs 1979 peak).
//
// Build:
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common \
//     -I runtime/nvidia -include cstdint runtime/nvidia/experiments/uni256_probe.cu \
//     -lcuda -o /tmp/uni256_probe
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
#define PGM90_TMA_STAGES 3
#define PGM_ARENA_BF16 (128 * 1024)
#define PLOW_ACT_SILU_ 0u
__device__ __forceinline__ float act_silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float act_gelu_tanh(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
#include "op_gemm_sm90.cuh"

using bf16 = __nv_bfloat16;

__global__ __launch_bounds__(256, 1) void k_n128(bf16* C, const void* mA, const void* mB,
                                                 const float* as, const float* ws, unsigned m,
                                                 unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    d_gemm_w8a8_sm90_tma(C, mA, mB, as, ws, m, n, k, 0, blockIdx.x, gridDim.x, arena);
}
__global__ __launch_bounds__(256, 1) void k_n256(bf16* C, const void* mA, const void* mB,
                                                 const float* as, const float* ws, unsigned m,
                                                 unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    d_gemm_w8a8_sm90_tma_uni256(C, mA, mB, as, ws, m, n, k, 0, blockIdx.x, gridDim.x, arena);
}

__global__ __launch_bounds__(256, 1) void k_bf128(bf16* C, const void* mA, const void* mB,
                                                  unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    d_gemm_sm90_tma(C, mA, mB, m, n, k, 0, blockIdx.x, gridDim.x, arena);
}
__global__ __launch_bounds__(256, 1) void k_bf256(bf16* C, const void* mA, const void* mB,
                                                  unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    d_gemm_sm90_tma_uni256(C, mA, mB, m, n, k, 0, blockIdx.x, gridDim.x, arena);
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

static void make_map8(CUtensorMap* mp, void* base, int rows, int K) {
    uint64_t gd[2] = {(uint64_t)K, (uint64_t)rows};
    uint64_t gs[1] = {(uint64_t)K};
    uint32_t bd[2] = {128u, 128u};
    uint32_t es[2] = {1, 1};
    memset(mp, 0, sizeof(*mp));
    CKD(cuTensorMapEncodeTiled(mp, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main(int argc, char** argv) {
    const int grid = 132;
    const unsigned SMEM = 200 * 1024;
    CK(cudaFuncSetAttribute(k_n128, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CK(cudaFuncSetAttribute(k_n256, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

    struct Shape {
        unsigned m, n, k;
        const char* what;
    } shapes[] = {
        {4096, 4096, 3840, "q_proj"},
        {4096, 512, 3840, "kv_proj"},
        {4096, 3840, 4096, "o_proj"},
        {4096, 15360, 3840, "gate/up"},
        {4096, 3840, 15360, "down"},
    };
    printf("grid=%d threads=256 smem=%u (occ-1)\n", grid, SMEM);
    printf("%-10s %10s %10s %8s\n", "shape", "n128", "n256", "n256/n128");
    for (auto& s : shapes) {
        size_t am = (size_t)s.m * s.k, bm = (size_t)s.n * s.k, cm = (size_t)s.m * s.n;
        uint8_t *A, *B;
        bf16* C;
        float *as, *ws;
        CK(cudaMalloc(&A, am));
        CK(cudaMalloc(&B, bm));
        CK(cudaMalloc(&C, cm * 2));
        CK(cudaMalloc(&as, s.m * 4));
        CK(cudaMalloc(&ws, s.n * 4));
        CK(cudaMemset(A, 0x3c, am));
        CK(cudaMemset(B, 0x3c, bm));
        std::vector<float> one(s.m > s.n ? s.m : s.n, 1.0f);
        CK(cudaMemcpy(as, one.data(), s.m * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ws, one.data(), s.n * 4, cudaMemcpyHostToDevice));
        CUtensorMap mA, mB;
        make_map8(&mA, A, s.m, s.k);
        make_map8(&mB, B, s.n, s.k);
        CUtensorMap *dA, *dB;
        CK(cudaMalloc(&dA, 256));
        dB = dA + 1;
        CK(cudaMemcpy(dA, &mA, 128, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, &mB, 128, cudaMemcpyHostToDevice));

        auto bench = [&](bool n256) -> float {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            for (int w = 0; w < 3; w++) {
                if (n256)
                    k_n256<<<grid, 256, SMEM>>>(C, dA, dB, as, ws, s.m, s.n, s.k);
                else
                    k_n128<<<grid, 256, SMEM>>>(C, dA, dB, as, ws, s.m, s.n, s.k);
            }
            CK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            const int it = 10;
            for (int i = 0; i < it; i++) {
                if (n256)
                    k_n256<<<grid, 256, SMEM>>>(C, dA, dB, as, ws, s.m, s.n, s.k);
                else
                    k_n128<<<grid, 256, SMEM>>>(C, dA, dB, as, ws, s.m, s.n, s.k);
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
            float x = bench(false), y = bench(true);
            if (x < a) a = x;
            if (y < b) b = y;
        }
        double fl = 2.0 * s.m * s.n * s.k;
        printf("%-10s %7.3f ms %7.3f ms %6.2fx   (%6.1f vs %6.1f TF/s)\n", s.what, a, b, b / a,
               fl / (a * 1e9), fl / (b * 1e9));
        cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(as); cudaFree(ws); cudaFree(dA);
    }
    // ---- bf16 pass ----
    CK(cudaFuncSetAttribute(k_bf128, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CK(cudaFuncSetAttribute(k_bf256, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    printf("\nbf16:\n%-10s %10s %10s %8s\n", "shape", "n128", "n256", "n256/n128");
    for (auto& s : shapes) {
        size_t am = (size_t)s.m * s.k, bm = (size_t)s.n * s.k, cm = (size_t)s.m * s.n;
        bf16 *A, *B, *C;
        CK(cudaMalloc(&A, am * 2));
        CK(cudaMalloc(&B, bm * 2));
        CK(cudaMalloc(&C, cm * 2));
        CK(cudaMemset(A, 0x3c, am * 2));
        CK(cudaMemset(B, 0x3c, bm * 2));
        CUtensorMap mA, mB;
        {   // bf16 maps: inner box 64 elems (=128 B)
            uint64_t gd[2] = {(uint64_t)s.k, (uint64_t)s.m};
            uint64_t gs[1] = {(uint64_t)s.k * 2};
            uint32_t bd[2] = {64u, 128u};
            uint32_t es[2] = {1, 1};
            memset(&mA, 0, sizeof(mA));
            CKD(cuTensorMapEncodeTiled(&mA, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, A, gd, gs, bd,
                                       es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                       CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                       CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
            gd[1] = (uint64_t)s.n;
            memset(&mB, 0, sizeof(mB));
            CKD(cuTensorMapEncodeTiled(&mB, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, B, gd, gs, bd,
                                       es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                       CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                       CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        }
        CUtensorMap *dA, *dB;
        CK(cudaMalloc(&dA, 256));
        dB = dA + 1;
        CK(cudaMemcpy(dA, &mA, 128, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, &mB, 128, cudaMemcpyHostToDevice));
        auto bench = [&](bool n256) -> float {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            for (int w = 0; w < 3; w++) {
                if (n256) k_bf256<<<grid, 256, SMEM>>>(C, dA, dB, s.m, s.n, s.k);
                else k_bf128<<<grid, 256, SMEM>>>(C, dA, dB, s.m, s.n, s.k);
            }
            CK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            const int it = 10;
            for (int i = 0; i < it; i++) {
                if (n256) k_bf256<<<grid, 256, SMEM>>>(C, dA, dB, s.m, s.n, s.k);
                else k_bf128<<<grid, 256, SMEM>>>(C, dA, dB, s.m, s.n, s.k);
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
            float x = bench(false), y = bench(true);
            if (x < a) a = x;
            if (y < b) b = y;
        }
        double fl = 2.0 * s.m * s.n * s.k;
        printf("%-10s %7.3f ms %7.3f ms %6.2fx   (%6.1f vs %6.1f TF/s)\n", s.what, a, b, b / a,
               fl / (a * 1e9), fl / (b * 1e9));
        cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(dA);
    }
    return 0;
}

// segws_repro.cu — standalone repro for the lean-object warp-spec deadlock.
//
// d_gemm_w8a8_sm90_tma_ws (op_gemm_sm90.cuh, PLOW_NV_SEG_GEMM=1) HANGS inside the
// interpreter (serve run pins the GPU at 100%) but its shape — entry __maxnreg__(128),
// producer setmaxnreg.dec 32 / consumer inc 224, NS=3 TMA ring — passes standalone in
// tma_ws_gemm_bf16.cu. The interpreter differs in three ways this harness reproduces
// one at a time (MODE bits), calling the REAL in-tree body:
//   bit 0: per-op RE-ENTRY   — call the body several times in one kernel launch
//   bit 1: MIXED OPS         — a uniform 256-thread dummy op between GEMM calls
//   bit 2: GQ-STYLE BARRIERS — __syncthreads() + elected atomicAdd claim between ops
//
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common \
//     -I runtime/nvidia -include cstdint runtime/nvidia/experiments/segws_repro.cu \
//     -lcuda -o /tmp/segws_repro && timeout 60 /tmp/segws_repro <mode>
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
#define PLOW_NV_SEG_GEMM 1
#define PLOW_NV_SEGMENTS 1
#define PGM90_TMA_STAGES 3
#define PGM90_FORK_GLU 0
#define PLOW_NV_W8A8 1
#define PGM90_FP8_PROMOTE 0
#define PGM_ARENA_BF16 (128 * 1024)
#define PLOW_ACT_SILU_ 0u
__device__ __forceinline__ float act_silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float act_gelu_tanh(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
#include "op_gemm_sm90.cuh"

using bf16 = __nv_bfloat16;

__device__ unsigned g_claim;
__device__ unsigned g_beat[264]; // per-block progress heartbeat

__global__ __maxnreg__(128) void k_repro(bf16* C, const void* mA, const void* mB,
                                         const float* as, const float* ws, unsigned m,
                                         unsigned n, unsigned k, int nops, int mode,
                                         float* dummy) {
    extern __shared__ bf16 arena[];
    for (int op = 0; op < nops; op++) {
        if (mode & 4) { // GQ-style claim
            __syncthreads();
            if (threadIdx.x == 0) atomicAdd(&g_claim, 1u);
            __syncthreads();
        }
        if ((mode & 2) && (op & 1)) { // mixed: uniform dummy op
            if (!(mode & 8) || blockIdx.x < 132) { // bit3: PARTIAL participation
                for (unsigned i = threadIdx.x; i < 4096; i += blockDim.x)
                    dummy[i] = dummy[i] * 1.0000001f + 1.0f;
                __syncthreads();
            }
        } else {
            d_gemm_w8a8_sm90_tma_ws(C, mA, mB, as, ws, m, n, k, 0, blockIdx.x, gridDim.x,
                                    arena);
        }
        if (threadIdx.x == 0) g_beat[blockIdx.x] = op + 1;
        if (!(mode & 1)) break; // single invocation
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
    const int mode = argc > 1 ? atoi(argv[1]) : 7;
    const unsigned m = argc > 2 ? (unsigned)atoi(argv[2]) : 1024, n = 2048, k = 3840;
    const unsigned SMEM = 100 * 1024;
    CK(cudaFuncSetAttribute(k_repro, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int grid = 264;

    uint8_t *A, *B;
    bf16* C;
    float *as, *ws, *dummy;
    CK(cudaMalloc(&A, (size_t)m * k));
    CK(cudaMalloc(&B, (size_t)n * k));
    CK(cudaMalloc(&C, (size_t)m * n * 2));
    CK(cudaMalloc(&as, m * 4));
    CK(cudaMalloc(&ws, n * 4));
    CK(cudaMalloc(&dummy, 4096 * 4));
    CK(cudaMemset(A, 0x3c, (size_t)m * k));
    CK(cudaMemset(B, 0x3c, (size_t)n * k));
    std::vector<float> ones(n, 1.0f);
    CK(cudaMemcpy(as, ones.data(), m * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ws, ones.data(), n * 4, cudaMemcpyHostToDevice));
    CUtensorMap mA, mB;
    make_map8(&mA, A, m, k);
    make_map8(&mB, B, n, k);
    CUtensorMap* dM;
    CK(cudaMalloc(&dM, 256));
    CK(cudaMemcpy(dM, &mA, 128, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dM + 1, &mB, 128, cudaMemcpyHostToDevice));

    printf("mode=%d m=%u (bit0 re-entry, bit1 mixed, bit2 gq-claim, bit3 partial, bit4 multi-launch) grid=%d\n",
           mode, m, grid);
    const int launches = (mode & 16) ? 5 : 1;
    cudaError_t e = cudaSuccess;
    for (int L = 0; L < launches && e == cudaSuccess; L++) {
        k_repro<<<grid, 256, SMEM>>>(C, dM, dM + 1, as, ws, m, n, k, 8, mode, dummy);
        e = cudaDeviceSynchronize();
    }
    printf("sync: %s\n", cudaGetErrorString(e));
    unsigned beat[264];
    CK(cudaMemcpyFromSymbol(beat, g_beat, sizeof(beat)));
    unsigned mn = ~0u, mx = 0;
    for (int i = 0; i < grid; i++) { mn = beat[i] < mn ? beat[i] : mn; mx = beat[i] > mx ? beat[i] : mx; }
    printf("heartbeat min=%u max=%u\n", mn, mx);
    return e != cudaSuccess;
}

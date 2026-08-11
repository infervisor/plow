#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_gemm.h"

typedef union __align__(16) { bf16 xstage[8192]; float part[PLOW_WAVES]; } cohort_smem;

struct Cohort {
    unsigned count;
    unsigned n[4];
    unsigned blocks[4];
};

template <int ID>
__device__ __forceinline__ Cohort cohort_desc() {
    if constexpr (ID == 0) return {3, {6144, 128, 12, 0}, {304, 128, 12, 0}};
    if constexpr (ID == 1) return {4, {1536, 512, 64, 1536}, {128, 128, 64, 128}};
    return {2, {896, 3584, 0, 0}, {128, 128, 0, 0}};
}

__device__ __forceinline__ void stage_x(cohort_smem& sm, const bf16* x, unsigned k) {
    const unsigned nv = k >> 3;
    for (unsigned i = threadIdx.x; i < nv; i += PLOW_THREADS)
        *(bf16v8*)(sm.xstage + i * 8) = ld_glob8(as_glob(x) + i * 8);
    for (unsigned i = (nv << 3) + threadIdx.x; i < k; i += PLOW_THREADS) sm.xstage[i] = x[i];
}

template <int ID>
__device__ __forceinline__ void cohort_control(bf16* C, const bf16* x, const bf16* W,
                                               unsigned nrep, unsigned k) {
    __shared__ cohort_smem sm;
    const Cohort d = cohort_desc<ID>();
    unsigned total_tasks = 0;
    unsigned total_n = 0;
#pragma unroll
    for (unsigned s = 0; s < d.count; ++s) {
        total_tasks += d.blocks[s];
        total_n += d.n[s];
    }
    const size_t slab = (size_t)total_n * k;
    for (unsigned e = 0; e < nrep; ++e) {
        for (unsigned task = blockIdx.x; task < total_tasks; task += gridDim.x) {
            unsigned task0 = 0;
            unsigned n0 = 0;
#pragma unroll
            for (unsigned s = 0; s < d.count; ++s) {
                const unsigned task1 = task0 + d.blocks[s];
                if (task >= task0 && task < task1) {
                    stage_x(sm, x, k);
                    __syncthreads();
                    gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(
                        C + n0, x, W + (size_t)e * slab + (size_t)n0 * k, nullptr, nullptr, 1u,
                        d.n[s], k, task - task0, d.blocks[s], sm.xstage);
                    __syncthreads();
                }
                task0 = task1;
                n0 += d.n[s];
            }
        }
    }
}

template <int ID>
__device__ __forceinline__ void cohort_fused(bf16* C, const bf16* x, const bf16* W,
                                             unsigned nrep, unsigned k) {
    __shared__ cohort_smem sm;
    const Cohort d = cohort_desc<ID>();
    unsigned total_n = 0;
#pragma unroll
    for (unsigned s = 0; s < d.count; ++s) total_n += d.n[s];
    const size_t slab = (size_t)total_n * k;
    for (unsigned e = 0; e < nrep; ++e) {
        stage_x(sm, x, k);
        __syncthreads();
        gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(C, x, W + (size_t)e * slab, nullptr,
                                                        nullptr, 1u, total_n, k, blockIdx.x,
                                                        gridDim.x, sm.xstage);
        __syncthreads();
    }
}

#define COHORT_KERNELS(ID)                                                                         \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_control_##ID(                    \
        bf16* C, const bf16* x, const bf16* W, unsigned nrep, unsigned k) {                         \
        cohort_control<ID>(C, x, W, nrep, k);                                                       \
    }                                                                                              \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_fused_##ID(                      \
        bf16* C, const bf16* x, const bf16* W, unsigned nrep, unsigned k) {                         \
        cohort_fused<ID>(C, x, W, nrep, k);                                                         \
    }

COHORT_KERNELS(0)
COHORT_KERNELS(1)
COHORT_KERNELS(2)

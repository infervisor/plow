// Benchmark-only geometry override: no interpreter or block-wide norm is launched.
#include <cstdint>
#include "sm120_common.cuh"
#if BENCH_CTA_THREADS != 256 && BENCH_CTA_THREADS != 512
#error "BENCH_CTA_THREADS must be 256 or 512"
#endif
#undef PLOW_NV_THREADS
#undef PLOW_NV_WARPS
#define PLOW_NV_THREADS BENCH_CTA_THREADS
#define PLOW_NV_WARPS (BENCH_CTA_THREADS / 32)
#include "op_gemm.cuh"

namespace {
using bf16 = __nv_bfloat16;
constexpr unsigned arena_bytes = 32768, guard_words = 16;

template <unsigned Op>
__global__ __launch_bounds__(BENCH_CTA_THREADS, 1)
void gemv_cta(bf16* out, const bf16* x, const bf16* w0, const bf16* w1,
              const bf16* w2, unsigned nq, unsigned nk, unsigned nv, unsigned k,
              unsigned* errors) {
    extern __shared__ unsigned shared[];
    auto* arena = reinterpret_cast<bf16*>(shared + guard_words);
    auto* tail = shared + guard_words + arena_bytes / sizeof(unsigned);
    if (threadIdx.x < guard_words) {
        shared[threadIdx.x] = 0xa5a5a5a5;
        tail[threadIdx.x] = 0xa5a5a5a5;
    }
    __syncthreads();
    if constexpr (Op == 0) {
        if (k * sizeof(bf16) <= 12352)
            d_gemv(out, x, w0, 1, nq, k, blockIdx.x, gridDim.x, arena);
        else
            d_gemv(out, x, w0, 1, nq, k, blockIdx.x, gridDim.x);
    } else if constexpr (Op == 1)
        d_gemv_qkv(out, out + nq, out + nq + nk, x, w0, w1, w2,
                   1, nq, nk, nv, k, blockIdx.x, gridDim.x, arena);
    else
        d_gemv_glu(out, x, w0, w1, 1, nq, k, 0, blockIdx.x, gridDim.x, arena);
    __syncthreads();
    if (threadIdx.x < guard_words &&
        (shared[threadIdx.x] != 0xa5a5a5a5 || tail[threadIdx.x] != 0xa5a5a5a5))
        atomicAdd(errors, 1u);
}
}

#define CTA_JOIN_(a,b) a##b
#define CTA_JOIN(a,b) CTA_JOIN_(a,b)
extern "C" const void* CTA_JOIN(gemv_cta_kernel_, BENCH_CTA_THREADS)(unsigned op) {
    if (op == 0) return reinterpret_cast<const void*>(gemv_cta<0>);
    if (op == 1) return reinterpret_cast<const void*>(gemv_cta<1>);
    if (op == 2) return reinterpret_cast<const void*>(gemv_cta<2>);
    return nullptr;
}

#include <cuda_runtime.h>
#include <cmath>
#define PLOW_NV_SPEECH 1
#define PLOW_NV_HOPPER 1
#define PLOW_NV_GEMM_F16_NODE 1
#include "op_elementwise.cuh"

extern "C" __device__ const unsigned plow_gemm_f16_node_abi = 1;

static __device__ __forceinline__ bool gn_span(uint64_t base, uint64_t rows, uint64_t cols,
                                             unsigned width, uint64_t& end) {
    if (!cols || rows > UINT64_MAX / cols || rows * cols > UINT64_MAX / width) return false;
    const uint64_t bytes = rows * cols * width;
    if (base > UINT64_MAX - bytes) return false;
    end = base + bytes;
    return true;
}

static __device__ __forceinline__ bool gn_overlap(uint64_t a, uint64_t ae, uint64_t b, uint64_t be) {
    return a < be && b < ae;
}

template <bool HALF_OUT>
static __device__ __forceinline__ void gn_run(
    const PlowDevInst* in, const uint64_t* tensors, unsigned tensor_count) {
    if (!in || !tensors || !tensor_count || tensor_count > 0xffffu || ((uint64_t)in & 7u) || ((uint64_t)tensors & 7u)) {
        asm("trap;"); return;
    }
    const unsigned m = in->i[0], n = in->i[1], k = in->i[2], flags = in->i[7];
    constexpr unsigned required = 3u | 1024u | 8192u;
    constexpr unsigned allowed = required | 4u | 8u | 16u | 512u | 16384u;
    if (in->op != PLOW_DOP_GEMV_F32_OUT || blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1
        || gridDim.y != 1 || gridDim.z != 1 || m < 16 || !n || n % 128 || !k || k % 64
        || m > UINT32_MAX-127u || ((uint64_t)m+127u)/128u*(n/128u) > UINT32_MAX
        || (flags & required) != required || (flags & ~allowed) || bool(flags & 16384u) != HALF_OUT
        || ((flags & 512u) && (!(flags & 4u) || !in->i[3]))) {
        asm("trap;"); return;
    }
    if ((flags & 512u) && (((uint64_t)m+in->i[3]-1u)/in->i[3])*n > UINT32_MAX) {
        asm("trap;"); return;
    }
    for (unsigned t = 0; t < 5; t++) {
        if (t == 3 && !(flags & 4u)) continue;
        if (t == 4 && !(flags & 8u)) continue;
        if (in->t[t] == 0xffffu || in->t[t] >= tensor_count || !tensors[in->t[t]] || (tensors[in->t[t]] & 15u)) {
            asm("trap;"); return;
        }
    }
    void* output = (void*)tensors[in->t[0]];
    const auto* x = (const unsigned short*)tensors[in->t[1]];
    const auto* w = (const unsigned short*)tensors[in->t[2]];
    const auto* bias = flags & 4u ? (const float*)tensors[in->t[3]] : nullptr;
    const auto* residual = flags & 8u ? (const float*)tensors[in->t[4]] : nullptr;
    const unsigned group = flags & 512u ? in->i[3] : 0;
    const uint64_t c = (uint64_t)output;
    uint64_t ce, xe, we, be, re, ie, te;
    if (!gn_span(c, m, n, flags & 16384u ? 2 : 4, ce)
        || !gn_span((uint64_t)x, m, k, 2, xe) || !gn_span((uint64_t)w, n, k, 2, we)
        || !gn_span((uint64_t)in, 1, sizeof(PlowDevInst), 1, ie)
        || !gn_span((uint64_t)tensors, tensor_count, 1, 8, te)
        || gn_overlap(c, ce, (uint64_t)x, xe) || gn_overlap(c, ce, (uint64_t)w, we)
        || gn_overlap(c, ce, (uint64_t)in, ie) || gn_overlap(c, ce, (uint64_t)tensors, te)
        || (bias && (!gn_span((uint64_t)bias, group ? ((uint64_t)m + group - 1) / group : 1, n, 4, be)
            || gn_overlap(c, ce, (uint64_t)bias, be)))
        || (residual && (!gn_span((uint64_t)residual, m, n, 4, re)
            || (gn_overlap(c, ce, (uint64_t)residual, re)
                && ((flags & 16384u) || output != residual))))) {
        asm("trap;"); return;
    }
    extern __shared__ float arena[];
    if constexpr (HALF_OUT)
        gemv_f16w_wide<128, true>(output, x, w, bias, group, residual, flags >> 4 & 15u,
            m, n, k, blockIdx.x, gridDim.x, arena);
    else
        gemv_f16w_wide<128, false>(output, x, w, bias, group, residual, flags >> 4 & 15u,
            m, n, k, blockIdx.x, gridDim.x, arena);
}

extern "C" __global__ void gemm_f16_sm90_f32_node(
    const PlowDevInst* in, const uint64_t* tensors, unsigned tensor_count) {
    gn_run<false>(in, tensors, tensor_count);
}

extern "C" __global__ void gemm_f16_sm90_f16_node(
    const PlowDevInst* in, const uint64_t* tensors, unsigned tensor_count) {
    gn_run<true>(in, tensors, tensor_count);
}

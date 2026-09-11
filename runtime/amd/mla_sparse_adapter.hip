#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "amd_common.h"
#endif

struct SparsePackArgs {
    uint4 *q, *kv;
    const uint4 *qa, *qr, *ck, *kr;
    unsigned *qp, *kp, *last, *splits;
    unsigned rows, ctx;
};
static_assert(sizeof(SparsePackArgs) == 88);

template <bool FP8>
__device__ void sparse_pack(SparsePackArgs a, const float* scales) {
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    for (unsigned i = first; i < a.rows * 8u * 72u; i += stride) {
        const unsigned row = i / 72u, col = i % 72u;
        a.q[i] = col < 64u ? a.qa[row * 64u + col] : a.qr[row * 8u + col - 64u];
    }
    for (unsigned i = first; i < a.ctx * 72u; i += stride) {
        const unsigned row = i / 72u, col = i % 72u;
        if (col >= 64u) {
            a.kv[i] = a.kr[row * 8u + col - 64u];
        } else if constexpr (FP8) {
#if defined(__HIP_DEVICE_COMPILE__)
            bf16v8 v = fp8v8_to_bf16v8(ld_glob_fp8v8(
                (const unsigned char*)a.ck + (size_t)row * 512u + col * 8u));
#pragma unroll
            for (unsigned j = 0; j < 8; ++j) v[j] = f2bf(bf2f(v[j]) * scales[row]);
            __builtin_memcpy(&a.kv[i], &v, 16);
#endif
        } else {
            a.kv[i] = a.ck[row * 64u + col];
        }
    }
    for (unsigned row = first; row <= a.rows; row += stride) {
        a.qp[row] = row;
        a.kp[row] = row * 2048u;
        a.splits[row] = row * 2u;
        if (row < a.rows) a.last[row] = 1;
    }
}

extern "C" __global__ void plow_mla_sparse_pack(SparsePackArgs a) {
    sparse_pack<false>(a, nullptr);
}

struct SparsePackFp8Args {
    SparsePackArgs base;
    const float* scales;
};
static_assert(sizeof(SparsePackFp8Args) == 96);

extern "C" __global__ void plow_mla_sparse_pack_fp8(SparsePackFp8Args a) {
    sparse_pack<true>(a.base, a.scales);
}

struct SparsePackSingleArgs {
    SparsePackFp8Args base;
    float2* ml;
};
static_assert(sizeof(SparsePackSingleArgs) == 104);

extern "C" __global__ void plow_mla_sparse_pack_fp8_single(SparsePackSingleArgs a) {
    sparse_pack<true>(a.base.base, a.base.scales);
    for (unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
         row < a.base.base.rows * 8u; row += blockDim.x * gridDim.x) {
        a.ml[row] = make_float2(0.0f, 1.0f);
    }
}

struct SparseReduceArgs {
    float *out, *ml;
    const float *part, *lse;
    unsigned rows, pad;
};
static_assert(sizeof(SparseReduceArgs) == 40);

extern "C" __global__ void plow_mla_sparse_reduce(SparseReduceArgs a) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < a.rows * 8u * 128u;
         i += blockDim.x * gridDim.x) {
        const unsigned hrow = i / 128u, col = i % 128u;
        const unsigned l = (hrow / 8u) * 16u + hrow % 8u;
        const float m = fmaxf(a.lse[l], a.lse[l + 8u]);
        const float wa = expf(a.lse[l] - m), wb = expf(a.lse[l + 8u] - m);
        const float4 pa = ((const float4*)a.part)[l * 128u + col];
        const float4 pb = ((const float4*)a.part)[(l + 8u) * 128u + col];
        ((float4*)a.out)[i] = make_float4(
            (pa.x * wa + pb.x * wb) / (wa + wb),
            (pa.y * wa + pb.y * wb) / (wa + wb),
            (pa.z * wa + pb.z * wb) / (wa + wb),
            (pa.w * wa + pb.w * wb) / (wa + wb));
        if (col == 0) {
            a.ml[hrow * 2u] = 0;
            a.ml[hrow * 2u + 1u] = 1;
        }
    }
}

extern "C" __device__ unsigned plow_mla_sparse_adapter_abi_1 = 1;
extern "C" __device__ unsigned plow_mla_sparse_adapter_fp8_abi_1 = 1;
#if defined(PLOW_MLA_SPARSE_SINGLE_PASS)
extern "C" __device__ unsigned plow_mla_sparse_single_abi_1 = 1;
#endif

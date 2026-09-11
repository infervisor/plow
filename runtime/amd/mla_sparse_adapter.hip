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
__device__ void sparse_pack(SparsePackArgs a, const float* __restrict__ scales) {
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    uint4* __restrict__ q = a.q;
    uint4* __restrict__ kv = a.kv;
    const uint4* __restrict__ qa = a.qa;
    const uint4* __restrict__ qr = a.qr;
    const uint4* __restrict__ kr = a.kr;
    for (unsigned i = first; i < a.rows * 8u * 64u; i += stride) {
        q[(i >> 6) * 72u + (i & 63u)] = qa[i];
    }
    for (unsigned i = first; i < a.rows * 8u * 8u; i += stride) {
        q[(i >> 3) * 72u + 64u + (i & 7u)] = qr[i];
    }
    for (unsigned i = first; i < a.ctx * 8u; i += stride) {
        kv[(i >> 3) * 72u + 64u + (i & 7u)] = kr[i];
    }
    if constexpr (FP8) {
#if defined(__HIP_DEVICE_COMPILE__)
        const unsigned char* __restrict__ ck = (const unsigned char*)a.ck;
        const unsigned n = a.ctx * 64u;
        unsigned i = first;
        // Four 8-byte FP8 loads and their scales issued before any store: the stride loop
        // with one load per store could not overlap iterations (kv may alias ck).
        for (; i + 3u * stride < n; i += 4u * stride) {
            bf16v8 v[4];
            float s[4];
#pragma unroll
            for (unsigned u = 0; u < 4; ++u) {
                const unsigned j = i + u * stride, row = j >> 6;
                v[u] = fp8v8_to_bf16v8(ld_glob_fp8v8(ck + (size_t)row * 512u + (j & 63u) * 8u));
                s[u] = scales[row];
            }
#pragma unroll
            for (unsigned u = 0; u < 4; ++u) {
                const unsigned j = i + u * stride;
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) v[u][e] = f2bf(bf2f(v[u][e]) * s[u]);
                __builtin_memcpy(&kv[(j >> 6) * 72u + (j & 63u)], &v[u], 16);
            }
        }
        for (; i < n; i += stride) {
            const unsigned row = i >> 6;
            bf16v8 v = fp8v8_to_bf16v8(ld_glob_fp8v8(ck + (size_t)row * 512u + (i & 63u) * 8u));
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) v[e] = f2bf(bf2f(v[e]) * scales[row]);
            __builtin_memcpy(&kv[row * 72u + (i & 63u)], &v, 16);
        }
#endif
    } else {
        const uint4* __restrict__ ck = a.ck;
        for (unsigned i = first; i < a.ctx * 64u; i += stride) {
            kv[(i >> 6) * 72u + (i & 63u)] = ck[i];
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

struct SparseDecodePackArgs {
    uint4 *q, *kv;
    unsigned* kvi;
    const uint4 *qa, *qr;
    const unsigned char* ck;
    const uint4* kr;
    const int* idx;
    const int* kvlen;
    const float* scales;
    unsigned *qp, *kp, *last, *splits;
    unsigned rows, ctx, nsplit, pad;
};
static_assert(sizeof(SparseDecodePackArgs) == 128);

// Decode: each row's selected FP8 latent rows (per-row scales) and BF16 rope keys are gathered
// into one packed BF16 CSR with identity kv_indices; rows are ragged at min(kv_len, 2048).
extern "C" __global__ void plow_mla_sparse_decode_pack(SparseDecodePackArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    unsigned kp[21];
    kp[0] = 0;
    for (unsigned b = 0; b < a.rows; ++b) {
        const int len = a.kvlen[b];
        const unsigned live = len < 0 ? 0u : (len > 2048 ? 2048u : (unsigned)len);
        kp[b + 1] = kp[b] + live;
    }
    for (unsigned i = first; i < a.rows * 8u * 72u; i += stride) {
        const unsigned row = i / 72u, col = i % 72u;
        a.q[i] = col < 64u ? a.qa[row * 64u + col] : a.qr[row * 8u + col - 64u];
    }
    const unsigned total = kp[a.rows];
    for (unsigned i = first; i < total * 72u; i += stride) {
        const unsigned slot = i / 72u, col = i % 72u;
        unsigned b = 0;
        while (kp[b + 1] <= slot) ++b;
        const unsigned key = (unsigned)a.idx[b * 2048u + (slot - kp[b])];
        const size_t src = (size_t)b * a.ctx + key;
        if (col >= 64u) {
            a.kv[i] = a.kr[src * 8u + col - 64u];
        } else {
            bf16v8 v = fp8v8_to_bf16v8(ld_glob_fp8v8(a.ck + src * 512u + col * 8u));
            const float s = a.scales[src];
#pragma unroll
            for (unsigned j = 0; j < 8; ++j) v[j] = f2bf(bf2f(v[j]) * s);
            __builtin_memcpy(&a.kv[i], &v, 16);
            if (col == 0) a.kvi[slot] = slot;
        }
    }
    for (unsigned b = first; b <= a.rows; b += stride) {
        a.qp[b] = b;
        a.kp[b] = kp[b];
        a.splits[b] = b * a.nsplit;
        if (b < a.rows) a.last[b] = 1;
    }
#endif
}

struct SparseDecodeRelayoutArgs {
    float *opart, *ml;
    const float *part, *lse;
    unsigned rows, nsplit;
};
static_assert(sizeof(SparseDecodeRelayoutArgs) == 40);

// AITER writes [row][split][head][512] normalized partials with one LSE each; the merge reads
// [row][head][split] with (m, l). (m, l) = (lse, 1) makes l*exp(m - M) the exact LSE weight.
extern "C" __global__ void plow_mla_sparse_decode_relayout(SparseDecodeRelayoutArgs a) {
    const unsigned ns = a.nsplit;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < a.rows * 8u * ns * 128u;
         i += blockDim.x * gridDim.x) {
        const unsigned col = i % 128u, unit = i / 128u;
        const unsigned s = unit % ns, hrow = unit / ns;
        const unsigned src = (hrow / 8u * ns + s) * 8u + hrow % 8u;
        ((float4*)a.opart)[unit * 128u + col] = ((const float4*)a.part)[src * 128u + col];
        if (col == 0) {
            a.ml[unit * 2u] = a.lse[src];
            a.ml[unit * 2u + 1u] = 1.0f;
        }
    }
}

extern "C" __device__ unsigned plow_mla_sparse_adapter_abi_1 = 1;
extern "C" __device__ unsigned plow_mla_sparse_decode_abi_1 = 1;
extern "C" __device__ unsigned plow_mla_sparse_adapter_fp8_abi_1 = 1;
#if defined(PLOW_MLA_SPARSE_SINGLE_PASS)
extern "C" __device__ unsigned plow_mla_sparse_single_abi_1 = 1;
#endif

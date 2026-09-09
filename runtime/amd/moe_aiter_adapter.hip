#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "amd_common.h"
#endif

template <unsigned N, unsigned K, bool GU>
__device__ void pack_weight(uint4* out, const unsigned long long* wt) {
    constexpr unsigned vectors = N * K / 16;
    const unsigned e = blockIdx.x / 8u;
    const auto* first = (const uint4*)wt[e * 3u + (GU ? 0u : 2u)];
    const auto* second = GU ? (const uint4*)wt[e * 3u + 1u] : first;
    for (unsigned v = blockIdx.x % 8u * blockDim.x + threadIdx.x; v < vectors;
         v += blockDim.x * 8u) {
        const unsigned row = (v / (32u * (K / 32u))) * 16u + v % 16u;
        const unsigned col = (v / 32u) % (K / 32u) * 2u + (v / 16u) % 2u;
        const unsigned source_row = GU ? row % 256u : row;
        const auto* src = GU && row >= 256u ? second : first;
        out[e * vectors + v] = src[source_row * (K / 16u) + col];
    }
}

extern "C" __global__ void pack_moe(uint4* gu, uint4* down, float* gs, float* ds,
                         const unsigned long long* wt, const unsigned long long* st) {
    // Runtime weights already scrub OCP -0. Keep their bytes and double FNUZ scales.
    pack_weight<512, 6144, true>(gu, wt);
    pack_weight<6144, 256, false>(down, wt);
    if (blockIdx.x % 8u == 0 && threadIdx.x < 96u) {
        const unsigned e = blockIdx.x / 8u, j = threadIdx.x;
        gs[e * 192u + j] = 2.0f * ((const float*)st[e * 3u])[j];
        gs[e * 192u + 96u + j] = 2.0f * ((const float*)st[e * 3u + 1u])[j];
        ds[e * 96u + j] = 2.0f * ((const float*)st[e * 3u + 2u])[j];
    }
}

struct MoePrepareArgs {
    unsigned short* q;
    float* qs;
    uint4* out;
    unsigned* ids;
    float* weights;
    unsigned* experts;
    unsigned* valid;
    const unsigned short* x;
    const unsigned* meta;
    const unsigned* row_token;
    const unsigned* row_part;
    const float* row_weight;
    unsigned rows;
    unsigned pad;
};
static_assert(sizeof(MoePrepareArgs) == 104);

extern "C" __global__ void plow_moe_aiter_prepare(MoePrepareArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    const unsigned lane = threadIdx.x % 64u;
    for (unsigned group = first / 64u; group < a.rows * 48u; group += stride / 64u) {
        const unsigned row = group / 48u, col = group % 48u;
        const unsigned pos = group * 128u + lane * 2u;
        const float x0 = bf2f(a.x[pos]), x1 = bf2f(a.x[pos + 1u]);
        const float amax = wave_max(fmaxf(1e-10f, fmaxf(fabsf(x0), fabsf(x1))));
        const float scale = amax * (1.0f / 240.0f);
        const float inverse = 1.0f / scale;
        a.q[pos / 2u] = (unsigned short)__builtin_amdgcn_cvt_pk_fp8_f32(
            x0 * inverse, x1 * inverse, 0, false);
        if (lane == 0) a.qs[col * a.rows + row] = scale;
    }
    for (unsigned i = first; i < a.rows * 768u; i += stride)
        a.out[i] = make_uint4(0, 0, 0, 0);
    const unsigned padded = a.meta[3u * 256u] * 64u;
    for (unsigned i = first; i < padded; i += stride) {
        const unsigned token = a.row_token[i], part = a.row_part[i];
        const bool live = token < a.rows && part < a.rows * 8u;
        a.ids[i] = live ? token | ((part % 8u) << 24) : a.rows;
        a.weights[i] = live ? a.row_weight[i] : 0.0f;
    }
    for (unsigned i = first; i < padded / 32u; i += stride) {
        unsigned lo = 0, hi = 256;
        while (lo < hi) {
            const unsigned mid = (lo + hi) / 2u;
            if (a.meta[512u + mid + 1u] <= i / 2u) lo = mid + 1u;
            else hi = mid;
        }
        a.experts[i] = lo;
    }
    if (first == 0) {
        a.valid[0] = padded;
        a.valid[1] = a.rows;
    }
#endif
}

extern "C" __global__ void plow_moe_aiter_store(float* out, const unsigned short* src,
                                                 unsigned rows) {
#if defined(__HIP_DEVICE_COMPILE__)
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * 768u;
         i += blockDim.x * gridDim.x) {
        const bf16v8 v = ((const bf16v8*)src)[i];
        ((float4*)out)[i * 2u] = make_float4(bf2f(v[0]), bf2f(v[1]), bf2f(v[2]), bf2f(v[3]));
        ((float4*)out)[i * 2u + 1u] = make_float4(bf2f(v[4]), bf2f(v[5]), bf2f(v[6]), bf2f(v[7]));
    }
#endif
}

extern "C" __device__ unsigned plow_moe_aiter_fp8_abi_1 = 1;

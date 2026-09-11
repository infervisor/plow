#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "amd_common.h"
#endif

template <unsigned N, unsigned K, bool GU>
__device__ void pack_weight(uint4* out, const unsigned long long* wt) {
    constexpr unsigned vectors = N * K / 16;
    /* One expert per 8 blocks; the caller sizes the grid at 8 * n_exp, which under
     * PLOW_GLM_MOE_SHARED_FOLD is 8 * (n_exp + 1) and packs the shared expert as the last one. */
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
    // Rows per sorted expert block: 32 for the 32x256 objects, 64 for the 64x256 tile.
    // Plow aligns at 64, so a 64-row block maps 1:1 onto one meta group.
    unsigned block_m;
    // Nonzero: permute the block order so that the K = 256 / block_m consecutive blocks of one
    // expert sit 8 positions apart (same XCD, concurrent workgroups) and share one L2 for the
    // expert's 4.7 MB of weights. Pure reorder: every block keeps its rows and expert.
    unsigned swizzle;
    // Routing shape. 256/8 for the ordinary GLM route; 257/9 under the shared-expert fold, where
    // the last expert IS the shared expert and every token holds a constant slot on it. Appended
    // after `swizzle`, so each adapter ABI reads a prefix of this block: 104 bytes (legacy),
    // 112 (swizzle; `n_exp` lands in what was its pad), 120 (this).
    unsigned n_exp;
    unsigned topk;
    unsigned pad;
};
static_assert(sizeof(MoePrepareArgs) == 120);

__device__ __forceinline__ unsigned swizzled_block(unsigned p, unsigned block_m, unsigned blocks) {
    const unsigned k = 256u / block_m, chunk = 8u * k;
    const unsigned c = p / chunk, r = p % chunk;
    if ((c + 1u) * chunk > blocks) return p;
    return c * chunk + (r % 8u) * k + r / 8u;
}

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
    const unsigned padded = a.meta[3u * a.n_exp] * 64u;
    const unsigned blocks = padded / a.block_m;
    for (unsigned i = first; i < padded; i += stride) {
        const unsigned li = a.swizzle
            ? swizzled_block(i / a.block_m, a.block_m, blocks) * a.block_m + i % a.block_m
            : i;
        const unsigned token = a.row_token[li], part = a.row_part[li];
        const bool live = token < a.rows && part < a.rows * a.topk;
        a.ids[i] = live ? token | ((part % a.topk) << 24) : a.rows;
        a.weights[i] = live ? a.row_weight[li] : 0.0f;
    }
    const unsigned per_group = 64u / a.block_m;
    for (unsigned i = first; i < blocks; i += stride) {
        const unsigned b = a.swizzle ? swizzled_block(i, a.block_m, blocks) : i;
        unsigned lo = 0, hi = a.n_exp;
        while (lo < hi) {
            const unsigned mid = (lo + hi) / 2u;
            if (a.meta[2u * a.n_exp + mid + 1u] <= b / per_group) lo = mid + 1u;
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
extern "C" __device__ unsigned plow_moe_aiter_tile64_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_aiter_swizzle_abi_1 = 1;
/* `MoePrepareArgs` grew n_exp/topk (104 -> 112 bytes) and `plow_moe_aiter_flat_pack` grew n_exp.
 * An adapter without this marker reads `block_m` where the new one writes `n_exp` and would walk
 * `meta` at the wrong stride — refuse it in the loader rather than sort a garbage routing table. */
extern "C" __device__ unsigned plow_moe_aiter_nexp_abi_1 = 1;

struct MoeFlatPackArgs {
    uint4* gu;
    uint4* down;
    float* gs;
    float* ds;
    const unsigned long long* wt;
    const unsigned long long* st;
    const uint2* table;
    unsigned* ids;
    float* weights;
    unsigned rows;
    // Slots per row in `table` — the decode router's top-k. Unchanged by the shared-expert fold,
    // which is a prefill fold; only the expert TABLE grows, and this kernel's grid with it.
    unsigned topk;
};
static_assert(sizeof(MoeFlatPackArgs) == 80);

extern "C" __global__ void plow_moe_aiter_flat_pack(MoeFlatPackArgs a) {
    __shared__ unsigned active;
    if (threadIdx.x == 0) active = 0;
    __syncthreads();
    const unsigned e = blockIdx.x / 8u;
    if (threadIdx.x < a.rows * a.topk) {
        const uint2 route = a.table[threadIdx.x];
        if (blockIdx.x == 0) {
            a.ids[threadIdx.x] = route.x;
            a.weights[threadIdx.x] = __uint_as_float(route.y);
        }
        if (route.x == e) atomicExch(&active, 1u);
    }
    __syncthreads();
    if (!active) return;
    pack_weight<512, 6144, true>(a.gu, a.wt);
    pack_weight<6144, 256, false>(a.down, a.wt);
    if (blockIdx.x % 8u == 0 && threadIdx.x < 96u) {
        const unsigned j = threadIdx.x;
        a.gs[e * 192u + j] = 2.0f * ((const float*)a.st[e * 3u])[j];
        a.gs[e * 192u + 96u + j] = 2.0f * ((const float*)a.st[e * 3u + 1u])[j];
        a.ds[e * 96u + j] = 2.0f * ((const float*)a.st[e * 3u + 2u])[j];
    }
}

extern "C" __device__ unsigned plow_moe_flat_fp8_abi_1 = 1;

struct MoeFlatRouteArgs {
    unsigned* ids;
    float* weights;
    const uint2* table;
    unsigned rows;
    unsigned topk;
};
static_assert(sizeof(MoeFlatRouteArgs) == 32);

extern "C" __global__ void plow_moe_aiter_flat_route(MoeFlatRouteArgs a) {
    if (threadIdx.x < a.rows * a.topk) {
        const uint2 route = a.table[threadIdx.x];
        a.ids[threadIdx.x] = route.x;
        a.weights[threadIdx.x] = __uint_as_float(route.y);
    }
}

extern "C" __device__ unsigned plow_moe_resident_fp8_abi_1 = 1;

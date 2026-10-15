#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "amd_common.h"
#endif

/* Model width the adapter is built for (scripts/build_moe_aiter.sh): GLM-5.2's 6144 on gfx942,
 * GLM-5.3-Flash's 4096 on gfx950. gfx942 AITER objects read FNUZ e4m3 (max 240, half the OCP
 * value of the same bits, so the OCP-scaled weight scales double); gfx950 objects read OCP. */
#ifndef MOE_AITER_H
#define MOE_AITER_H 6144
#endif
#define MOE_AITER_KB (MOE_AITER_H / 128)
#if defined(__gfx950__)
#define MOE_AITER_FP8_MAX 448.0f
#define MOE_AITER_W_SCALE 1.0f
#else
#define MOE_AITER_FP8_MAX 240.0f
#define MOE_AITER_W_SCALE 2.0f
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
    pack_weight<512, MOE_AITER_H, true>(gu, wt);
    pack_weight<MOE_AITER_H, 256, false>(down, wt);
    if (blockIdx.x % 8u == 0 && threadIdx.x < 2u * MOE_AITER_KB) {
        const unsigned e = blockIdx.x / 8u, j = threadIdx.x;
        gs[e * 4u * MOE_AITER_KB + j] = MOE_AITER_W_SCALE * ((const float*)st[e * 3u])[j];
        gs[e * 4u * MOE_AITER_KB + 2u * MOE_AITER_KB + j] =
            MOE_AITER_W_SCALE * ((const float*)st[e * 3u + 1u])[j];
        ds[e * 2u * MOE_AITER_KB + j] = MOE_AITER_W_SCALE * ((const float*)st[e * 3u + 2u])[j];
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
    // Nonzero: `out` already holds the shared expert's partial and the fused call accumulates the
    // routed partials onto it, so it is not zeroed. Was `pad`; ABI plow_moe_aiter_keep_out_abi_1.
    unsigned keep_out;
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
    for (unsigned group = first / 64u; group < a.rows * MOE_AITER_KB; group += stride / 64u) {
        const unsigned row = group / MOE_AITER_KB, col = group % MOE_AITER_KB;
        const unsigned pos = group * 128u + lane * 2u;
        const float x0 = bf2f(a.x[pos]), x1 = bf2f(a.x[pos + 1u]);
        const float amax = wave_max(fmaxf(1e-10f, fmaxf(fabsf(x0), fabsf(x1))));
        const float scale = amax * (1.0f / MOE_AITER_FP8_MAX);
        const float inverse = 1.0f / scale;
        a.q[pos / 2u] = (unsigned short)__builtin_amdgcn_cvt_pk_fp8_f32(
            x0 * inverse, x1 * inverse, 0, false);
        if (lane == 0) a.qs[col * a.rows + row] = scale;
    }
    if (!a.keep_out)
        for (unsigned i = first; i < a.rows * (MOE_AITER_H / 8u); i += stride)
            a.out[i] = make_uint4(0, 0, 0, 0);
#if defined(__gfx950__)
    /* gfx950 decode tiles are 16/32 rows and the AITER kernel pays full weight traffic for an
     * all-pad block, so the 64-row align groups are compacted: expert e keeps
     * ceil(count/block_m) blocks, back to back. Every workgroup rebuilds the block prefix. */
    __shared__ unsigned start[513];
    for (unsigned e = threadIdx.x; e < a.n_exp; e += blockDim.x)
        start[e + 1u] = ((unsigned)a.meta[a.n_exp + e] + a.block_m - 1u) / a.block_m;
    __syncthreads();
    if (threadIdx.x == 0) {
        start[0] = 0;
        for (unsigned e = 0; e < a.n_exp; e++) start[e + 1u] += start[e];
    }
    __syncthreads();
    const unsigned blocks = start[a.n_exp], padded = blocks * a.block_m;
    for (unsigned i = first; i < padded; i += stride) {
        const unsigned b = i / a.block_m;
        unsigned lo = 0, hi = a.n_exp;
        while (lo < hi) {
            const unsigned mid = (lo + hi) / 2u;
            if (start[mid + 1u] <= b) lo = mid + 1u;
            else hi = mid;
        }
        const unsigned r = (b - start[lo]) * a.block_m + i % a.block_m;
        const bool live = r < (unsigned)a.meta[a.n_exp + lo];
        const unsigned li = (unsigned)a.meta[lo] + r;
        const unsigned token = live ? a.row_token[li] : a.rows, part = live ? a.row_part[li] : 0u;
        a.ids[i] = live ? token | ((part % a.topk) << 24) : a.rows;
        a.weights[i] = live ? a.row_weight[li] : 0.0f;
        if (i % a.block_m == 0u) a.experts[b] = lo;
    }
#else
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
#endif
    if (first == 0) {
        a.valid[0] = padded;
        a.valid[1] = a.rows;
    }
#endif
}

extern "C" __global__ void plow_moe_aiter_store(float* out, const unsigned short* src,
                                                 unsigned rows) {
#if defined(__HIP_DEVICE_COMPILE__)
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * (MOE_AITER_H / 8u);
         i += blockDim.x * gridDim.x) {
        const bf16v8 v = ((const bf16v8*)src)[i];
        ((float4*)out)[i * 2u] = make_float4(bf2f(v[0]), bf2f(v[1]), bf2f(v[2]), bf2f(v[3]));
        ((float4*)out)[i * 2u + 1u] = make_float4(bf2f(v[4]), bf2f(v[5]), bf2f(v[6]), bf2f(v[7]));
    }
#endif
}

/* `plow_moe_aiter_prepare` straight from the router's [rows][topk] (expert u32, weight f32)
 * table: workgroup 0 sorts it into block_m-compacted AITER blocks, the others quantize x and
 * zero `out`. Replaces the interpreter align + its segment boundary (decode rungs). */
struct MoeTableArgs {
    unsigned short* q;
    float* qs;
    uint4* out;
    unsigned* ids;
    float* weights;
    unsigned* experts;
    unsigned* valid;
    const unsigned short* x;
    const uint2* table;
    unsigned rows;
    unsigned block_m;
    unsigned n_exp;
    unsigned topk;
    unsigned keep_out;
    unsigned pad;
};
static_assert(sizeof(MoeTableArgs) == 96);

#ifndef PLOW_MOE_TABLE_SCAN
#define PLOW_MOE_TABLE_SCAN 0
#endif

extern "C" __global__ void plow_moe_aiter_prepare_table(MoeTableArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    if (blockIdx.x == 0) {
        __shared__ unsigned cnt[512], start[513], cur[512];
        const unsigned slots = a.rows * a.topk, tid = threadIdx.x;
        for (unsigned e = tid; e < a.n_exp; e += blockDim.x) cnt[e] = cur[e] = 0;
        __syncthreads();
        for (unsigned s = tid; s < slots; s += blockDim.x) {
            const unsigned e = a.table[s].x;
            if (e < a.n_exp) atomicAdd(&cnt[e], 1u);
        }
        __syncthreads();
#if PLOW_MOE_TABLE_SCAN
        /* Wave 0 scans: lane l owns experts [l*per, l*per+per), an exclusive prefix over the
         * lane sums, then each lane writes its run. Same starts as the serial loop below, whose
         * 288 dependent LDS round trips dominated this kernel at decode. */
        if (tid < 64u) {
            const unsigned per = (a.n_exp + 63u) / 64u, e0 = tid * per;
            unsigned own = 0;
            for (unsigned j = 0; j < per; j++)
                if (e0 + j < a.n_exp) own += (cnt[e0 + j] + a.block_m - 1u) / a.block_m;
            unsigned inc = own;
            for (unsigned d = 1; d < 64u; d <<= 1) {
                const unsigned v = __shfl_up(inc, d, 64);
                if (tid >= d) inc += v;
            }
            unsigned b = inc - own;
            for (unsigned j = 0; j < per; j++)
                if (e0 + j < a.n_exp) {
                    start[e0 + j] = b;
                    b += (cnt[e0 + j] + a.block_m - 1u) / a.block_m;
                }
            if (tid == 63u) start[a.n_exp] = inc;
        }
#else
        if (tid == 0) {
            unsigned b = 0;
            for (unsigned e = 0; e < a.n_exp; e++) {
                start[e] = b;
                b += (cnt[e] + a.block_m - 1u) / a.block_m;
            }
            start[a.n_exp] = b;
        }
#endif
        __syncthreads();
        const unsigned blocks = start[a.n_exp], padded = blocks * a.block_m;
#if PLOW_MOE_TABLE_SCAN
        /* Per expert instead of a binary search per padded entry: same experts[] and the same
         * padding entries (positions >= cnt[e] of e's blocks). */
        for (unsigned e = tid; e < a.n_exp; e += blockDim.x) {
            const unsigned b0 = start[e], b1 = start[e + 1u];
            for (unsigned b = b0; b < b1; b++) a.experts[b] = e;
            for (unsigned i = b0 * a.block_m + cnt[e]; i < b1 * a.block_m; i++) {
                a.ids[i] = a.rows;
                a.weights[i] = 0.0f;
            }
        }
#else
        for (unsigned i = tid; i < padded; i += blockDim.x) {
            const unsigned b = i / a.block_m;
            unsigned lo = 0, hi = a.n_exp;
            while (lo < hi) {
                const unsigned mid = (lo + hi) / 2u;
                if (start[mid + 1u] <= b) lo = mid + 1u;
                else hi = mid;
            }
            if (i % a.block_m == 0u) a.experts[b] = lo;
            if ((b - start[lo]) * a.block_m + i % a.block_m >= cnt[lo]) {
                a.ids[i] = a.rows;
                a.weights[i] = 0.0f;
            }
        }
#endif
        for (unsigned s = tid; s < slots; s += blockDim.x) {
            const uint2 r = a.table[s];
            if (r.x >= a.n_exp) continue;
            const unsigned at = start[r.x] * a.block_m + atomicAdd(&cur[r.x], 1u);
            a.ids[at] = (s / a.topk) | ((s % a.topk) << 24);
            a.weights[at] = __uint_as_float(r.y);
        }
        if (tid == 0) {
            a.valid[0] = padded;
            a.valid[1] = a.rows;
        }
        if (gridDim.x > 1) return;
    }
    const unsigned workers = gridDim.x > 1 ? gridDim.x - 1u : 1u;
    const unsigned wg = gridDim.x > 1 ? blockIdx.x - 1u : 0u;
    const unsigned first = wg * blockDim.x + threadIdx.x;
    const unsigned stride = workers * blockDim.x;
    const unsigned lane = threadIdx.x % 64u;
    for (unsigned group = first / 64u; group < a.rows * MOE_AITER_KB; group += stride / 64u) {
        const unsigned row = group / MOE_AITER_KB, col = group % MOE_AITER_KB;
        const unsigned pos = group * 128u + lane * 2u;
        const float x0 = bf2f(a.x[pos]), x1 = bf2f(a.x[pos + 1u]);
        const float amax = wave_max(fmaxf(1e-10f, fmaxf(fabsf(x0), fabsf(x1))));
        const float scale = amax * (1.0f / MOE_AITER_FP8_MAX);
        const float inverse = 1.0f / scale;
        a.q[pos / 2u] = (unsigned short)__builtin_amdgcn_cvt_pk_fp8_f32(
            x0 * inverse, x1 * inverse, 0, false);
        if (lane == 0) a.qs[col * a.rows + row] = scale;
    }
    if (!a.keep_out)
        for (unsigned i = first; i < a.rows * (MOE_AITER_H / 8u); i += stride)
            a.out[i] = make_uint4(0, 0, 0, 0);
#endif
}

extern "C" __device__ unsigned plow_moe_aiter_table_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_aiter_fp8_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_aiter_tile64_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_aiter_swizzle_abi_1 = 1;
/* `MoePrepareArgs` grew n_exp/topk (104 -> 112 bytes) and `plow_moe_aiter_flat_pack` grew n_exp.
 * An adapter without this marker reads `block_m` where the new one writes `n_exp` and would walk
 * `meta` at the wrong stride — refuse it in the loader rather than sort a garbage routing table. */
extern "C" __device__ unsigned plow_moe_aiter_nexp_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_aiter_keep_out_abi_1 = 1;

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
    pack_weight<512, MOE_AITER_H, true>(a.gu, a.wt);
    pack_weight<MOE_AITER_H, 256, false>(a.down, a.wt);
    if (blockIdx.x % 8u == 0 && threadIdx.x < 2u * MOE_AITER_KB) {
        const unsigned j = threadIdx.x;
        a.gs[e * 4u * MOE_AITER_KB + j] = MOE_AITER_W_SCALE * ((const float*)a.st[e * 3u])[j];
        a.gs[e * 4u * MOE_AITER_KB + 2u * MOE_AITER_KB + j] =
            MOE_AITER_W_SCALE * ((const float*)a.st[e * 3u + 1u])[j];
        a.ds[e * 2u * MOE_AITER_KB + j] = MOE_AITER_W_SCALE * ((const float*)a.st[e * 3u + 2u])[j];
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

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

extern "C" __global__ void plow_moe_aiter_prepare_table(MoeTableArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    if (blockIdx.x == 0) {
        /* Parallel over experts: per-expert block counts, a workgroup scan, then each expert's
         * own block ids and pad rows. A thread-0 serial prefix over 289 experts plus a binary
         * search per padded row cost ~25 us at 128 rows; this is the decode critical path. */
        __shared__ unsigned cnt[512], start[513], cur[512], wsum[8];
        const unsigned slots = a.rows * a.topk, tid = threadIdx.x, nt = blockDim.x;
        for (unsigned e = tid; e < 512u; e += nt) cnt[e] = cur[e] = 0;
        __syncthreads();
        for (unsigned s = tid; s < slots; s += nt) {
            const unsigned e = a.table[s].x;
            if (e < a.n_exp) atomicAdd(&cnt[e], 1u);
        }
        __syncthreads();
        /* Two experts per thread (n_exp <= 512, blockDim 256): exclusive scan of block counts. */
        const unsigned e0 = 2u * tid, e1 = e0 + 1u;
        const unsigned b0 = e0 < a.n_exp ? (cnt[e0] + a.block_m - 1u) / a.block_m : 0u;
        const unsigned b1 = e1 < a.n_exp ? (cnt[e1] + a.block_m - 1u) / a.block_m : 0u;
        const unsigned lane = tid % 64u, wave = tid / 64u;
        unsigned incl = b0 + b1;
        for (unsigned o = 1; o < 64u; o <<= 1) {
            const unsigned v = __shfl_up(incl, o);
            if (lane >= o) incl += v;
        }
        if (lane == 63u) wsum[wave] = incl;
        __syncthreads();
        unsigned base = 0;
        for (unsigned w = 0; w < wave; w++) base += wsum[w];
        const unsigned excl = base + incl - b0 - b1;
        if (e0 <= a.n_exp) start[e0] = excl;
        if (e1 <= a.n_exp) start[e1] = excl + b0;
        __syncthreads();
        const unsigned blocks = start[a.n_exp], padded = blocks * a.block_m;
        for (unsigned e = tid; e < a.n_exp; e += nt) {
            const unsigned lo = start[e], hi = start[e + 1u];
            for (unsigned b = lo; b < hi; b++) a.experts[b] = e;
            for (unsigned i = lo * a.block_m + cnt[e]; i < hi * a.block_m; i++) {
                a.ids[i] = a.rows;
                a.weights[i] = 0.0f;
            }
        }
        for (unsigned s = tid; s < slots; s += nt) {
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

/* PLOW_AMD_ROUTER_NATIVE: the decode router segment (GemmF32 score + MoeRouterTopkPf) as two
 * kernels. `plow_moe_router_gemm`: f32 logits[rows][n_exp] = x[rows][K] . w[n_exp][K]^T, one
 * 16x16 output tile per workgroup, the four waves splitting K and summing through LDS.
 * `plow_moe_router_topk`: wave per token, the interpreter's PLOW_MOE_ROUTER_PF_WAVE selection
 * and gate tail (sigmoid, bias, top-k, normalize, scale, shared-expert tail slot). */
struct MoeRouterGemmArgs {
    float* logit;
    const unsigned short* x;
    const unsigned short* w;
    unsigned rows;
    unsigned n_exp;
    unsigned k;
    unsigned pad;
};
static_assert(sizeof(MoeRouterGemmArgs) == 40);

/* Eight waves split K (512 each, 16 fully unrolled 16-byte loads per operand in flight per lane):
 * the score GEMM is load-latency bound at these shapes, not MFMA bound. */
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
__device__ __forceinline__ void router_gemm_tile(const MoeRouterGemmArgs& a) {
    __shared__ f32x4 part[8][64];
    const unsigned tiles_n = (a.n_exp + 15u) / 16u;
    const unsigned t0 = blockIdx.x / tiles_n * 16u, e0 = blockIdx.x % tiles_n * 16u;
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned r = lane & 15u, kg = lane >> 4;
    const unsigned tok = t0 + r, exp = e0 + r;
    const bool tok_ok = tok < a.rows, exp_ok = exp < a.n_exp;
    const unsigned kw = a.k / 8u;
    const bf16x8* xp = (const bf16x8*)(a.x + (size_t)(tok_ok ? tok : 0u) * a.k + wave * kw + kg * 8u);
    const bf16x8* wp = (const bf16x8*)(a.w + (size_t)(exp_ok ? exp : 0u) * a.k + wave * kw + kg * 8u);
    f32x4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    const bf16x8 zero = {};
    constexpr unsigned U = 16u;
    for (unsigned s0 = 0; s0 < kw / 32u; s0 += U) {
        bf16x8 av[U], bv[U];
#pragma unroll
        for (unsigned u = 0; u < U; u++) {
            av[u] = tok_ok ? xp[(s0 + u) * 4u] : zero;
            bv[u] = exp_ok ? wp[(s0 + u) * 4u] : zero;
        }
#pragma unroll
        for (unsigned u = 0; u < U; u++)
            acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(av[u], bv[u], acc, 0, 0, 0);
    }
    part[wave][lane] = acc;
    __syncthreads();
    if (wave == 0) {
        f32x4 s = part[0][lane];
#pragma unroll
        for (unsigned w = 1; w < 8u; w++) s += part[w][lane];
        const unsigned e = e0 + (lane & 15u);
#pragma unroll
        for (unsigned i = 0; i < 4u; i++) {
            const unsigned t = t0 + (lane >> 4) * 4u + i;
            if (t < a.rows && e < a.n_exp) a.logit[(size_t)t * a.n_exp + e] = s[i];
        }
    }
}
#endif

extern "C" __global__ __launch_bounds__(512) void plow_moe_router_gemm(MoeRouterGemmArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    router_gemm_tile(a);
#endif
}

struct MoeRouterTopkArgs {
    uint2* table;
    const float* logit;
    const float* bias;
    unsigned rows;
    unsigned n_exp;
    unsigned k;
    unsigned norm;
    unsigned shared_tail;
    float route_scale;
};
static_assert(sizeof(MoeRouterTopkArgs) == 48);

#if defined(__HIP_DEVICE_COMPILE__)
/* One wave selects one token's top-k; `wl`/`ws` are that wave's 8-slot LDS scratch. */
__device__ __forceinline__ void router_topk_row(const MoeRouterTopkArgs& a, unsigned tok,
                                                unsigned* wl, float* ws) {
    const unsigned lane = threadIdx.x & 63u;
    const unsigned per = (a.n_exp + 63u) / 64u;
    float s[5];
    unsigned long long key[5];
#pragma unroll
    for (unsigned q = 0; q < 5; q++) {
        s[q] = 0.0f;
        key[q] = 0ull;
        if (q >= per) continue;
        const unsigned e = lane + q * 64u;
        if (e >= a.n_exp) continue;
        const float l = a.logit[(size_t)tok * a.n_exp + e];
        s[q] = 1.0f / (1.0f + expf(-l));
        float sc = s[q] + (a.bias ? a.bias[e] : 0.0f);
        unsigned sb;
        __builtin_memcpy(&sb, &sc, 4);
        sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
        key[q] = ((unsigned long long)sb << 32) | (unsigned long long)(a.n_exp - 1u - e);
    }
    /* k rounds of a wave-wide max over the packed keys (unique per expert, so the winner is
     * one lane and one slot): the same order the rank-by-comparison form produces. The round
     * reduces the 32 high key bits (the ordered biased score) through DPP and settles equal
     * scores, which are rare, with a 64-bit shuffle reduction. */
    for (unsigned j = 0; j < a.k; j++) {
        unsigned long long best = 0ull;
        unsigned bq = 0u;
        float bs = 0.0f;
#pragma unroll
        for (unsigned q = 0; q < 5; q++)
            if (q < per && key[q] > best) {
                best = key[q];
                bq = q;
                bs = s[q];
            }
        const unsigned hi = (unsigned)(best >> 32);
        unsigned m = hi;
        m = max(m, (unsigned)__builtin_amdgcn_update_dpp(0, (int)m, 0xb1, 0xf, 0xf, true));
        m = max(m, (unsigned)__builtin_amdgcn_update_dpp(0, (int)m, 0x4e, 0xf, 0xf, true));
        m = max(m, (unsigned)__builtin_amdgcn_update_dpp(0, (int)m, 0x141, 0xf, 0xf, true));
        m = max(m, (unsigned)__builtin_amdgcn_update_dpp(0, (int)m, 0x140, 0xf, 0xf, true));
        m = max(m, (unsigned)__builtin_amdgcn_ds_swizzle((int)m, (16 << 10) | 31));
        m = max(m, (unsigned)__shfl_xor((int)m, 32, 64));
        const unsigned long long cand = __ballot(hi == m && best != 0ull);
        bool take;
        if (__popcll(cand) == 1) {
            take = lane == (unsigned)(__ffsll((long long)cand) - 1);
        } else {
            unsigned long long win = hi == m ? best : 0ull;
#pragma unroll
            for (unsigned o = 32; o >= 1; o >>= 1) {
                const unsigned lo = __shfl_xor((unsigned)win, o), h2 = __shfl_xor((unsigned)(win >> 32), o);
                const unsigned long long v = ((unsigned long long)h2 << 32) | lo;
                win = v > win ? v : win;
            }
            take = win != 0ull && best == win;
        }
        if (take) {
            wl[j] = lane + bq * 64u;
            ws[j] = bs;
#pragma unroll
            for (unsigned q = 0; q < 5; q++)
                if (q == bq) key[q] = 0ull;
        }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    if (lane == 0) {
        uint2* trow = a.table + (size_t)tok * (a.k + a.shared_tail);
        float gate[8];
        for (unsigned j = 0; j < a.k; j++) gate[j] = ws[j];
        float sum = 0.0f;
        for (unsigned j = 0; j < a.k; j++) sum += gate[j];
        for (unsigned j = 0; j < a.k; j++) {
            if (a.norm && sum != 0.0f) gate[j] /= sum;
            gate[j] *= a.route_scale;
            trow[j] = make_uint2(wl[j], __float_as_uint(gate[j]));
        }
        if (a.shared_tail) trow[a.k] = make_uint2(a.n_exp, __float_as_uint(1.0f));
    }
}
#endif

extern "C" __global__ __launch_bounds__(256) void plow_moe_router_topk(MoeRouterTopkArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned wl_all[4][8];
    __shared__ float ws_all[4][8];
    const unsigned wave = threadIdx.x >> 6;
    const unsigned tok = blockIdx.x * 4u + wave;
    if (tok >= a.rows) return;
    router_topk_row(a, tok, wl_all[wave], ws_all[wave]);
#endif
}

/* PLOW_AMD_ROUTER_FUSED: the score GEMM and the top-k in one launch. Each 16-token row of tiles
 * counts its finished expert tiles; the last tile's workgroup selects those 16 tokens' top-k
 * from the f32 logits the others published, then re-arms the counter. */
struct MoeRouterFusedArgs {
    MoeRouterGemmArgs g;
    MoeRouterTopkArgs t;
    unsigned* ctr;
};
static_assert(sizeof(MoeRouterFusedArgs) == 96);

extern "C" __global__ __launch_bounds__(512) void plow_moe_router_fused(MoeRouterFusedArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    router_gemm_tile(a.g);
    __shared__ unsigned last;
    __shared__ unsigned wl_all[8][8];
    __shared__ float ws_all[8][8];
    const unsigned tiles_n = (a.g.n_exp + 15u) / 16u, rt = blockIdx.x / tiles_n;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&a.ctr[rt], 1u) == tiles_n - 1u;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const unsigned wave = threadIdx.x >> 6;
    for (unsigned tok = rt * 16u + wave; tok < min(rt * 16u + 16u, a.t.rows); tok += 8u)
        router_topk_row(a.t, tok, wl_all[wave], ws_all[wave]);
    if (threadIdx.x == 0) a.ctr[rt] = 0u;
#endif
}

extern "C" __device__ unsigned plow_moe_router_abi_2 = 1;
extern "C" __device__ unsigned plow_moe_router_fused_abi_1 = 1;

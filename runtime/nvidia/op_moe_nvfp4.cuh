/* op_moe_nvfp4.cuh — NVFP4 (modelopt, e2m1 + per-16 e4m3 block scale + per-tensor f32 scale)
 * latent MoE family for sm_120/sm_121 (Nemotron-3.5 Super): f32 router, sigmoid+bias top-k, token
 * align, decode grouped GEMV, prefill grouped GEMM, shared expert (E=1 dense mode).
 * ABI: dev_isa.h ops 240-246; spec in the packet contract notes for the nemotron emitter.
 *
 * Precision: W4A16. w = e2m1 * e4m3_scale is formed EXACTLY in bf16 (<= 5 significant bits), the
 * activation stays bf16, mma.sync m16n8k16 bf16 accumulates in f32, and the per-expert
 * weight_scale_2 multiplies the f32 accumulator. No activation quantization (input_scale unused).
 * Router: bf16 x bf16 products are exact in the f32 accumulator, so the tensor-core logits equal an
 * f32 GEMM up to summation order.
 *
 * Weight layout (checkpoint bytes concatenated along the expert axis, no repack):
 *   W  u8   [E][N][K/2]   byte j of a row = k 2j (low nibble) | k 2j+1 (high nibble)
 *   S  e4m3 [E][N][K/16]
 *   gs f32  [E]
 * Requires 256 threads, K % 128 == 0 and N % 16 == 0 (decode), K % 32 == 0 and N % 128 == 0
 * (prefill tiles). Every op traps on a violated shape constraint. */
#ifndef PLOW_NV_OP_MOE_NVFP4_CUH
#define PLOW_NV_OP_MOE_NVFP4_CUH

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define NF4_UNUSED 0xFFFFFFFFu
/* Decode ops (router GEMV, grouped GEMV) run any T <= MAXROWS: rows go in m16 passes. */
#define NF4_DEC_MAXROWS 64u
#define NF4_DEC_MAXSLOTS (NF4_DEC_MAXROWS * 32u)
/* Dense (shared expert) decode items above this many rows take the two-tile entry. */
#define NF4_DEC_WIDE_ROWS 16u
#define NF4_BM 64
#define NF4_BN 128
#define NF4_BK 64
#define NF4_ST 72 /* bf16 smem row stride: 144 B, an odd multiple of 16 B (ldmatrix conflict-free) */
#define NF4_TILE_ARENA_BYTES (2 * (NF4_BM + NF4_BN) * NF4_ST * 2)
/* Decode grouping scratch: cnt[E] + off[E] (u32) + experts[slots] + slot list[slots] (u16). */
#define NF4_DEC_ARENA_BYTES(E) (8u * (E) + 4u * NF4_DEC_MAXSLOTS)
#define NF4_ALIGN_ARENA_BYTES(E) (8u * (E))
static_assert(NF4_DEC_ARENA_BYTES(1024u) <= NF4_TILE_ARENA_BYTES, "decode grouping scratch exceeds the MoE arena");

__device__ __forceinline__ void nf4_mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                        unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned nf4_prmt(unsigned a, unsigned b, unsigned s) {
    unsigned d;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}

__device__ __forceinline__ unsigned nf4_bmul2(unsigned a, unsigned s) {
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a),
                               *reinterpret_cast<__nv_bfloat162*>(&s));
    return *reinterpret_cast<unsigned*>(&r);
}

/* Four e2m1 nibbles (h bits 0-15, element 0 lowest) -> bf16x2 {e0,e1} and {e2,e3}, each times the
 * bf16x2 block scale sc2. Magnitude through two 8-entry byte tables (prmt index 0-7), sign from
 * bit 3 of each nibble. Exact: e2m1 * e4m3 needs at most 5 significant bits. */
__device__ __forceinline__ void nf4_deq4(unsigned h, unsigned sc2, unsigned& b0, unsigned& b1) {
    const unsigned sel = h & 0x7777u;
    const unsigned hi = nf4_prmt(0x3F3F3F00u, 0x40404040u, sel);
    const unsigned lo = nf4_prmt(0xC0800000u, 0xC0804000u, sel);
    unsigned x0 = nf4_prmt(lo, hi, 0x5140u) | ((h << 12) & 0x8000u) | ((h << 24) & 0x80000000u);
    unsigned x1 = nf4_prmt(lo, hi, 0x7362u) | ((h << 4) & 0x8000u) | ((h << 16) & 0x80000000u);
    b0 = nf4_bmul2(x0, sc2);
    b1 = nf4_bmul2(x1, sc2);
}

/* e4m3 byte -> bf16x2 (value replicated). e4m3 is a subset of bf16. */
__device__ __forceinline__ unsigned nf4_sc2(unsigned v) {
    __nv_fp8_e4m3 f;
    f.__x = (__nv_fp8_storage_t)(v & 0xFFu);
    const __nv_bfloat16 b = __float2bfloat16((float)f);
    const unsigned short u = *reinterpret_cast<const unsigned short*>(&b);
    return (unsigned)u | ((unsigned)u << 16);
}

__device__ __forceinline__ unsigned nf4_f2bf2(float a, float b) {
    __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned*>(&r);
}

__device__ __forceinline__ void nf4_cp16(void* smem, const void* g, unsigned bytes) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void nf4_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void nf4_cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void nf4_ldsm_x4(unsigned (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

/* Bulk L2 prefetch (UBLKPF.L2): one instruction stages a whole weight block, which is what
 * keeps enough bytes in flight at 8 warps/SM. p 16 B aligned, bytes % 16 == 0. */
__device__ __forceinline__ void nf4_pf_l2(const void* p, unsigned bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
#endif
}

#define NF4_PF_WIN 6u
/* Line prefetch to L2 of the 64 B a row contributes to one 128-K window. */
__device__ __forceinline__ void nf4_pf_line(const void* p) {
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
}

__device__ __forceinline__ float nf4_relu2(float v) {
    v = fmaxf(v, 0.0f);
    return v * v;
}

/* Rows mb..mb+16*MT-1 of one router item (8 experts n0.., K chunk ks) as MT m16 tiles. */
template <int MT>
static __device__ __forceinline__ void nf4_router_pass(float* __restrict__ logits,
    const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ wp, unsigned T, unsigned E,
    unsigned H, unsigned kc, unsigned kb, unsigned ks, unsigned n0, unsigned mb) {
    const unsigned lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    constexpr int R = 2 * MT;
    bool rv[R];
    const uint4* xp[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const unsigned gi = mb + g + 8u * r;
        rv[r] = gi < T;
        xp[r] = reinterpret_cast<const uint4*>(x + (size_t)gi * H + kb + 8u * t);
    }
    float acc[MT][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[m][q] = 0.f;
    const uint4 z = make_uint4(0, 0, 0, 0);
    /* Issue a whole 256-K batch of loads before the first mma consumes one. */
    for (unsigned k0 = 0; k0 < kc; k0 += 256u) {
        uint4 w[8], a[R][8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const unsigned k = k0 + 32u * j;
            const bool in = k < kc;
            w[j] = in ? __ldg(wp + k / 8u) : z;
#pragma unroll
            for (int r = 0; r < R; r++) a[r][j] = (in && rv[r]) ? __ldg(xp[r] + k / 8u) : z;
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const uint4 &a0 = a[2 * m][j], &b0 = a[2 * m + 1][j];
                nf4_mma(acc[m], a0.x, b0.x, a0.y, b0.y, w[j].x, w[j].y);
                nf4_mma(acc[m], a0.z, b0.z, a0.w, b0.w, w[j].z, w[j].w);
            }
    }
    float* o = logits + (size_t)ks * T * E;
    const unsigned c = n0 + 2u * t;
#pragma unroll
    for (int m = 0; m < MT; m++) {
        const unsigned g0 = mb + g + 16u * m, g1 = g0 + 8u;
        if (rv[2 * m]) *reinterpret_cast<float2*>(o + (size_t)g0 * E + c) = make_float2(acc[m][0], acc[m][1]);
        if (rv[2 * m + 1]) *reinterpret_cast<float2*>(o + (size_t)g1 * E + c) = make_float2(acc[m][2], acc[m][3]);
    }
}

/* ---------------------------------------------------------------------------------------------
 * ROUTER LOGITS (op 240): logits[ks][t][e] = sum_{k in chunk ks} x[t][k] * W[e][k], f32.
 * T <= NF4_DEC_MAXROWS: item = (8 experts, K chunk), weights streamed with 16 B lane loads, the
 * rows in passes of one or two m16 tiles (a later pass re-reads the item's weights from L1/L2);
 * each row's mma sequence is the same in either (bitwise equal); the K permutation
 * (lane t holds k 8t..8t+7 of each 32-group) is applied to A and B alike.
 * Wider T: the tiled GEMM below (KS must be 1). */
static __device__ void nf4_router_logits_dec(float* __restrict__ logits,
                                             const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ W, unsigned T,
                                             unsigned E, unsigned H, unsigned KS, unsigned slice,
                                             unsigned nblk) {
    const unsigned lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    const unsigned gw = slice * 8u + (threadIdx.x >> 5), nw = nblk * 8u;
    const unsigned kc = H / KS, ntile = E / 8u, items = ntile * KS;
    auto prefetch = [&](unsigned it) {
        if (it >= items || lane >= 8u) return;
        const unsigned ks = it / ntile, n0 = (it - ks * ntile) * 8u;
        nf4_pf_l2(W + (size_t)(n0 + lane) * H + ks * kc, kc * 2u);
    };
    prefetch(gw);
    prefetch(gw + nw);
    for (unsigned it = gw; it < items; it += nw) {
        prefetch(it + 2u * nw);
        const unsigned ks = it / ntile, n0 = (it - ks * ntile) * 8u;
        const unsigned kb = ks * kc;
        const uint4* wp = reinterpret_cast<const uint4*>(W + (size_t)(n0 + g) * H + kb + 8u * t);
        /* Above 16 rows one walk feeds two m16 tiles from each weight load. */
        unsigned mb = 0;
        for (; mb + 16u < T; mb += 32u) nf4_router_pass<2>(logits, x, wp, T, E, H, kc, kb, ks, n0, mb);
        for (; mb < T; mb += 16u) nf4_router_pass<1>(logits, x, wp, T, E, H, kc, kb, ks, n0, mb);
    }
}

/* ---------------------------------------------------------------------------------------------
 * ROUTER TOP-K (op 241), one warp per token:
 *   l = sum_ks logits[ks][t][:]; s = sigmoid(l); pick the k largest of s + bias (ties -> lower id);
 *   w_j = s_j / (sum_j s_j + 1e-20) (flags bit0, else raw s_j); w_j *= scale.
 * table[t*k + j] = {u32 expert, f32 w_j}, j in descending biased-score order. */
/* Warp selection over lane-resident scores s/b (expert j*32 + lane), writes token tok's slots. */
__device__ __forceinline__ void nf4_topk_select(float (&s)[32], float (&b)[32],
                                                unsigned char* __restrict__ table, unsigned tok,
                                                unsigned k, unsigned flags, float scale) {
    const unsigned lane = threadIdx.x & 31u;
    {
        unsigned my_e = NF4_UNUSED;
        float my_w = 0.f;
        for (unsigned r = 0; r < k; r++) {
            float bv = -INFINITY, sv = 0.f;
            unsigned bj = 0;
#pragma unroll
            for (unsigned j = 0; j < 32u; j++)
                if (b[j] > bv) { bv = b[j]; sv = s[j]; bj = j; }
            unsigned be = bj * 32u + lane;
            if (bv == -INFINITY) be = 0xFFFFFFFEu;
#pragma unroll
            for (unsigned off = 16; off; off >>= 1) {
                const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, off);
                const unsigned oe = __shfl_xor_sync(0xFFFFFFFFu, be, off);
                const float os = __shfl_xor_sync(0xFFFFFFFFu, sv, off);
                if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; sv = os; }
            }
            if ((be & 31u) == lane) {
#pragma unroll
                for (unsigned j = 0; j < 32u; j++)
                    if (j == (be >> 5)) b[j] = -INFINITY;
            }
            if (lane == r) { my_e = be; my_w = sv; }
        }
        float sum = lane < k ? my_w : 0.f;
#pragma unroll
        for (unsigned off = 16; off; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
        if (lane < k) {
            float w = (flags & 1u) ? my_w / (sum + 1e-20f) : my_w;
            unsigned char* p = table + ((size_t)tok * k + lane) * 8u;
            *reinterpret_cast<unsigned*>(p) = my_e;
            *reinterpret_cast<float*>(p + 4) = w * scale;
        }
    }
}

/* Prefill top-k (KS == 1), NJ = E / 32 biased scores per lane, Q tokens per warp interleaved (a
 * round is a chain of dependent shuffles; independent chains hide each other's latency). Each
 * lane caches its best (strict >, so the lowest expert id wins a lane-local tie); a round is one
 * warp argmax (ties -> lower id, as nf4_topk_select) and only the winning lane rescans. The
 * unbiased sigmoid of a selected expert is recomputed from its logit at the end (same expression),
 * so no score array is carried. Same selection, order and weights as nf4_topk_select. */
/* `taken` masks the lane's already selected slots: the array itself is never written after the
 * load, so it stays in registers (a store at the winner's slot index would put it in local memory). */
template <unsigned NJ>
__device__ __forceinline__ void nf4_lane_best(const float (&b)[NJ], unsigned taken, float& bv, unsigned& bj) {
    bv = -INFINITY; bj = 0u;
#pragma unroll
    for (unsigned j = 0; j < NJ; j++)
        if (!((taken >> j) & 1u) && b[j] > bv) { bv = b[j]; bj = j; }
}

template <unsigned NJ>
static __device__ void nf4_router_topk_pf(unsigned char* __restrict__ table, const float* __restrict__ logits,
                                          const float* __restrict__ bias, unsigned T, unsigned k,
                                          unsigned flags, float scale, unsigned slice, unsigned nblk) {
    constexpr unsigned Q = 2u;
    constexpr unsigned E = NJ * 32u;
    const unsigned lane = threadIdx.x & 31u;
    for (unsigned t0 = (slice * 8u + (threadIdx.x >> 5)) * Q; t0 < T; t0 += nblk * 8u * Q) {
        float b[Q][NJ], lb[Q];
        unsigned lj[Q], my_e[Q], taken[Q];
#pragma unroll
        for (unsigned q = 0; q < Q; q++) {
            const unsigned tok = min(t0 + q, T - 1u);
#pragma unroll
            for (unsigned j = 0; j < NJ; j++) {
                const unsigned e = j * 32u + lane;
                /* __frcp_rn: the correctly rounded 1/x, bit-equal to the division, no slow path. */
                const float sv = __frcp_rn(1.0f + expf(-logits[(size_t)tok * E + e]));
                b[q][j] = sv + (bias ? bias[e] : 0.f);
            }
            taken[q] = 0u;
            nf4_lane_best<NJ>(b[q], 0u, lb[q], lj[q]);
            my_e[q] = NF4_UNUSED;
        }
        for (unsigned r = 0; r < k; r++) {
            float bv[Q];
            unsigned be[Q];
#pragma unroll
            for (unsigned q = 0; q < Q; q++) {
                bv[q] = lb[q];
                be[q] = lb[q] == -INFINITY ? 0xFFFFFFFEu : lj[q] * 32u + lane;
            }
#pragma unroll
            for (unsigned off = 16; off; off >>= 1) {
#pragma unroll
                for (unsigned q = 0; q < Q; q++) {
                    const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv[q], off);
                    const unsigned oe = __shfl_xor_sync(0xFFFFFFFFu, be[q], off);
                    if (ov > bv[q] || (ov == bv[q] && oe < be[q])) { bv[q] = ov; be[q] = oe; }
                }
            }
#pragma unroll
            for (unsigned q = 0; q < Q; q++) {
                if ((be[q] & 31u) == lane) {
                    taken[q] |= 1u << (be[q] >> 5);
                    nf4_lane_best<NJ>(b[q], taken[q], lb[q], lj[q]);
                }
                if (lane == r) my_e[q] = be[q];
            }
        }
#pragma unroll
        for (unsigned q = 0; q < Q; q++) {
            const unsigned tok = t0 + q;
            float my_w = 0.f;
            if (tok < T && lane < k && my_e[q] < E)
                my_w = __frcp_rn(1.0f + expf(-logits[(size_t)tok * E + my_e[q]]));
            float sum = my_w;
#pragma unroll
            for (unsigned off = 16; off; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
            if (tok < T && lane < k) {
                const float w = (flags & 1u) ? my_w / (sum + 1e-20f) : my_w;
                unsigned char* p = table + ((size_t)tok * k + lane) * 8u;
                *reinterpret_cast<uint2*>(p) = make_uint2(my_e[q], __float_as_uint(w * scale));
            }
        }
    }
}

/* KS == 1: one warp per token. KS > 1 (decode): one block per token, so the KS partial loads of
 * all experts are in flight at once (a warp would serialise 16 x KS dependent L2 round trips),
 * staged through the arena (8E bytes), then warp 0 selects. */
static __device__ void nf4_router_topk(unsigned char* __restrict__ table,
                                       const float* __restrict__ logits,
                                       const float* __restrict__ bias,
                                       const int* __restrict__ active, unsigned T, unsigned E,
                                       unsigned k, unsigned KS, unsigned flags, float scale,
                                       unsigned slice, unsigned nblk, float* arena) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned nj = E / 32u;
    float s[32], b[32];
    if (KS == 1u && E == 512u) {
        nf4_router_topk_pf<16>(table, logits, bias, T, k, flags, scale, slice, nblk);
        return;
    }
    if (KS == 1u) {
        for (unsigned tok = slice * 8u + (threadIdx.x >> 5); tok < T; tok += nblk * 8u) {
#pragma unroll
            for (unsigned j = 0; j < 32u; j++) {
                s[j] = 0.f;
                b[j] = -INFINITY;
                if (j < nj) {
                    const unsigned e = j * 32u + lane;
                    s[j] = 1.0f / (1.0f + expf(-logits[(size_t)tok * E + e]));
                    b[j] = s[j] + (bias ? bias[e] : 0.f);
                }
            }
            nf4_topk_select(s, b, table, tok, k, flags, scale);
        }
        return;
    }
    float* ss = arena;
    float* sb = arena + E;
    for (unsigned tok = slice; tok < T; tok += nblk) {
        /* An inactive decode row routes nowhere (block-uniform: one block per token). */
        if (active && active[tok] <= 0) {
            if (threadIdx.x < k) {
                unsigned char* p = table + ((size_t)tok * k + threadIdx.x) * 8u;
                *reinterpret_cast<unsigned*>(p) = NF4_UNUSED;
                *reinterpret_cast<float*>(p + 4) = 0.f;
            }
            continue;
        }
        for (unsigned e = threadIdx.x; e < E; e += 256u) {
            float p[4] = {0.f, 0.f, 0.f, 0.f};
            unsigned q = 0;
            for (; q + 4u <= KS; q += 4u)
#pragma unroll
                for (unsigned u = 0; u < 4u; u++) p[u] += logits[((size_t)(q + u) * T + tok) * E + e];
            for (; q < KS; q++) p[0] += logits[((size_t)q * T + tok) * E + e];
            const float l = (p[0] + p[1]) + (p[2] + p[3]);
            const float sv = 1.0f / (1.0f + expf(-l));
            ss[e] = sv;
            sb[e] = sv + (bias ? bias[e] : 0.f);
        }
        __syncthreads();
        if (threadIdx.x < 32u) {
#pragma unroll
            for (unsigned j = 0; j < 32u; j++) {
                s[j] = j < nj ? ss[j * 32u + lane] : 0.f;
                b[j] = j < nj ? sb[j * 32u + lane] : -INFINITY;
            }
            nf4_topk_select(s, b, table, tok, k, flags, scale);
        }
        __syncthreads();
    }
}

/* ---------------------------------------------------------------------------------------------
 * Decode grouping: every block rebuilds the same expert -> slot lists from the table (T*k <= 512
 * slots). Experts are enumerated in ascending id; the slot order inside an expert is atomic-order,
 * which no output depends on (every mma row is computed independently). */
struct Nf4Groups {
    unsigned U;
    const unsigned short* ex;   /* [U] expert id */
    const unsigned* off;        /* [E] slot-list offset, indexed by expert */
    const unsigned* cnt;        /* [E] */
    const unsigned short* slot; /* [S] */
};

static __device__ Nf4Groups nf4_group(const unsigned char* __restrict__ table, unsigned S,
                                      unsigned E, unsigned* arena) {
    unsigned* cnt = arena;
    unsigned* off = arena + E;
    unsigned short* ex = reinterpret_cast<unsigned short*>(arena + 2u * E);
    unsigned short* sl = ex + NF4_DEC_MAXSLOTS;
    __shared__ unsigned s_wsum[8];
    __shared__ unsigned s_wuse[8];
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    for (unsigned e = tid; e < E; e += 256u) cnt[e] = 0u;
    __syncthreads();
    for (unsigned s = tid; s < S; s += 256u) {
        const unsigned e = *reinterpret_cast<const unsigned*>(table + (size_t)s * 8u);
        if (e < E) atomicAdd(&cnt[e], 1u);
    }
    __syncthreads();
    /* Block scan over E in contiguous per-thread runs. */
    const unsigned per = (E + 255u) / 256u, e0 = tid * per, e1 = min(e0 + per, E);
    unsigned rs = 0, ru = 0;
    for (unsigned e = e0; e < e1; e++) { rs += cnt[e]; ru += cnt[e] ? 1u : 0u; }
    unsigned is = rs, iu = ru;
#pragma unroll
    for (unsigned d = 1; d < 32u; d <<= 1) {
        const unsigned a = __shfl_up_sync(0xFFFFFFFFu, is, d), b = __shfl_up_sync(0xFFFFFFFFu, iu, d);
        if (lane >= d) { is += a; iu += b; }
    }
    if (lane == 31u) { s_wsum[warp] = is; s_wuse[warp] = iu; }
    __syncthreads();
    unsigned bs = 0, bu = 0, U = 0;
    for (unsigned w = 0; w < 8u; w++) {
        if (w < warp) { bs += s_wsum[w]; bu += s_wuse[w]; }
        U += s_wuse[w];
    }
    unsigned ps = bs + is - rs, pu = bu + iu - ru;
    for (unsigned e = e0; e < e1; e++) {
        off[e] = ps;
        if (cnt[e]) ex[pu++] = (unsigned short)e;
        ps += cnt[e];
    }
    __syncthreads();
    for (unsigned e = tid; e < E; e += 256u) cnt[e] = 0u;
    __syncthreads();
    for (unsigned s = tid; s < S; s += 256u) {
        const unsigned e = *reinterpret_cast<const unsigned*>(table + (size_t)s * 8u);
        if (e < E) sl[off[e] + atomicAdd(&cnt[e], 1u)] = (unsigned short)s;
    }
    __syncthreads();
    Nf4Groups gr;
    gr.U = U; gr.ex = ex; gr.off = off; gr.cnt = cnt; gr.slot = sl;
    return gr;
}

/* One pass of the decode grouped GEMV item (expert e, weight rows n0..n0+15): rows mb..mb+16*MT-1
 * of the item's cnt as MT m16 tiles, one walk over K. */
template <int PHASE, int MT>
static __device__ __forceinline__ void nf4_dec_pass(void* __restrict__ out,
    const __nv_bfloat16* __restrict__ act, const unsigned char* __restrict__ table,
    const uint8_t* __restrict__ We, const uint8_t* __restrict__ Se, float sc, const Nf4Groups& gr,
    bool dense, unsigned so, unsigned cnt, unsigned mb, unsigned n0, unsigned N, unsigned K,
    unsigned ksel, unsigned mode, unsigned nwin, size_t ldw, size_t lds, unsigned lane) {
    const unsigned g = lane >> 2, t = lane & 3u;
    constexpr int R = 2 * MT; /* row groups: g, g + 8, g + 16, g + 24 */
    bool v[R];
    unsigned sl[R];
    const uint4* xp[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const unsigned gi = mb + g + 8u * r;
        v[r] = gi < cnt;
        sl[r] = v[r] ? (dense ? gi : gr.slot[so + gi]) : 0u;
        /* Activation row: up reads x[token], down reads fu[slot] (dense: token == slot). */
        const unsigned ar = (PHASE == 0 && !dense) ? sl[r] / ksel : sl[r];
        xp[r] = reinterpret_cast<const uint4*>(act + (size_t)ar * K + 32u * t);
    }
    const uint4* w0p = reinterpret_cast<const uint4*>(We + (size_t)(n0 + g) * ldw + 16u * t);
    const uint4* w1p = reinterpret_cast<const uint4*>(We + (size_t)(n0 + 8u + g) * ldw + 16u * t);
    const unsigned short* c0p = reinterpret_cast<const unsigned short*>(Se + (size_t)(n0 + g) * lds + 2u * t);
    const unsigned short* c1p = reinterpret_cast<const unsigned short*>(Se + (size_t)(n0 + 8u + g) * lds + 2u * t);
    float acc[MT][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[m][j][q] = 0.f;
    /* Everything for window w+1 (weights, scales, activations) is in flight while w computes:
     * an activation round trip per window, even an L2 hit, otherwise serialises the walk. */
    /* Weights run two windows ahead (and PF_WIN ahead in L2 for large items). */
    if (t == 0u)
        for (unsigned p = 2u; p < NF4_PF_WIN && p < nwin; p++) {
            nf4_pf_line(w0p + p * 4u);
            nf4_pf_line(w1p + p * 4u);
        }
    uint4 w0 = __ldg(w0p), w1 = __ldg(w1p);
    unsigned c0 = __ldg(c0p), c1 = __ldg(c1p);
    uint4 m0 = w0, m1 = w1;
    unsigned mc0 = c0, mc1 = c1;
    if (nwin > 1u) {
        m0 = __ldg(w0p + 4u); m1 = __ldg(w1p + 4u);
        mc0 = __ldg(c0p + 4u); mc1 = __ldg(c1p + 4u);
    }
    uint4 xr[R][4];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int q = 0; q < 4; q++) xr[r][q] = v[r] ? __ldg(xp[r] + q) : make_uint4(0, 0, 0, 0);
    for (unsigned win = 0; win < nwin; win++) {
        if (t == 0u && win + NF4_PF_WIN < nwin) {
            nf4_pf_line(w0p + (win + NF4_PF_WIN) * 4u);
            nf4_pf_line(w1p + (win + NF4_PF_WIN) * 4u);
        }
        const uint4 nw0 = m0, nw1 = m1;
        const unsigned nc0 = mc0, nc1 = mc1;
        if (win + 2u < nwin) {
            m0 = __ldg(w0p + (win + 2u) * 4u);
            m1 = __ldg(w1p + (win + 2u) * 4u);
            mc0 = __ldg(c0p + (win + 2u) * 4u);
            mc1 = __ldg(c1p + (win + 2u) * 4u);
        }
        uint4 nx[R][4];
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int q = 0; q < 4; q++) nx[r][q] = xr[r][q];
        if (win + 1u < nwin) {
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int q = 0; q < 4; q++)
                    if (v[r]) nx[r][q] = __ldg(xp[r] + (win + 1u) * 16u + q);
        }
        const unsigned sa0 = nf4_sc2(c0), sb0 = nf4_sc2(c0 >> 8);
        const unsigned sa1 = nf4_sc2(c1), sb1 = nf4_sc2(c1 >> 8);
        const unsigned* q0 = reinterpret_cast<const unsigned*>(&w0);
        const unsigned* q1 = reinterpret_cast<const unsigned*>(&w1);
#pragma unroll
        for (int s = 0; s < 8; s++) {
            unsigned b0, b1, b2, b3;
            nf4_deq4((q0[s >> 1] >> (16 * (s & 1))) & 0xFFFFu, s < 4 ? sa0 : sb0, b0, b1);
            nf4_deq4((q1[s >> 1] >> (16 * (s & 1))) & 0xFFFFu, s < 4 ? sa1 : sb1, b2, b3);
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const unsigned* p0 = reinterpret_cast<const unsigned*>(xr[2 * m]);
                const unsigned* p1 = reinterpret_cast<const unsigned*>(xr[2 * m + 1]);
                const unsigned a0 = p0[2 * s], a2 = p0[2 * s + 1];
                const unsigned a1 = p1[2 * s], a3 = p1[2 * s + 1];
                nf4_mma(acc[m][0], a0, a1, a2, a3, b0, b1);
                nf4_mma(acc[m][1], a0, a1, a2, a3, b2, b3);
            }
        }
        w0 = nw0; w1 = nw1; c0 = nc0; c1 = nc1;
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int q = 0; q < 4; q++) xr[r][q] = nx[r][q];
    }
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const float* a = acc[m][j];
        const unsigned col = n0 + 8u * j + 2u * t;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            if (!v[2 * m + h]) continue;
            const unsigned slot = sl[2 * m + h];
            const float y0 = a[2 * h] * sc, y1 = a[2 * h + 1] * sc;
            if (PHASE == 0) {
                *reinterpret_cast<unsigned*>(reinterpret_cast<__nv_bfloat16*>(out) + (size_t)slot * N + col) =
                    nf4_f2bf2(nf4_relu2(y0), nf4_relu2(y1));
            } else if (mode == 0u) {
                const float gate = *reinterpret_cast<const float*>(table + (size_t)slot * 8u + 4u);
                *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + (size_t)slot * N + col) =
                    make_float2(y0 * gate, y1 * gate);
            } else {
                *reinterpret_cast<unsigned*>(reinterpret_cast<__nv_bfloat16*>(out) + (size_t)slot * N + col) =
                    nf4_f2bf2(y0, y1);
            }
        }
    }
}

/* ---------------------------------------------------------------------------------------------
 * Decode grouped GEMV (ops 242/243). One warp item = (expert, 16 weight rows), all K, every slot
 * routed to that expert as one m16 tile. Lane (g,t) streams 16 B of weight row n0+8j+g per
 * 128-K window (k 32t..32t+31) plus its 2 scale bytes; the same K permutation feeds A from the
 * slot's activation row. PHASE 0 = up (+relu^2 -> bf16 fu), 1 = down (gate-scaled f32 part, or
 * bf16 dense out). */
/* The two-tile pass in its own register allocation: inlined, its pressure spills the one-tile
 * pass that every routed item runs. */
template <int PHASE>
static __device__ __noinline__ void nf4_dec_pass2(void* __restrict__ out,
    const __nv_bfloat16* __restrict__ act, const unsigned char* __restrict__ table,
    const uint8_t* __restrict__ We, const uint8_t* __restrict__ Se, float sc, Nf4Groups gr,
    unsigned cnt, unsigned mb, unsigned n0, unsigned N, unsigned K, unsigned ksel, unsigned mode,
    unsigned nwin, size_t ldw, size_t lds, unsigned lane) {
    nf4_dec_pass<PHASE, 2>(out, act, table, We, Se, sc, gr, true, 0u, cnt, mb, n0, N, K, ksel, mode,
                           nwin, ldw, lds, lane);
}

template <int PHASE, bool WIDE>
static __device__ void nf4_moe_dec(void* __restrict__ out, const __nv_bfloat16* __restrict__ act,
                                   const unsigned char* __restrict__ table,
                                   const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                   const float* __restrict__ gs, unsigned T, unsigned ksel,
                                   unsigned N, unsigned K, unsigned E, unsigned mode,
                                   unsigned slice, unsigned nblk, unsigned* arena) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned gw = slice * 8u + (threadIdx.x >> 5), nw = nblk * 8u;
    const bool dense = table == nullptr;
    const unsigned nslots = dense ? T : T * ksel;
    Nf4Groups gr;
    if (dense) { gr.U = 1u; gr.ex = nullptr; gr.off = nullptr; gr.cnt = nullptr; gr.slot = nullptr; }
    else gr = nf4_group(table, nslots, E, arena);

    /* Down/part: a slot routed to no expert still owns a part row; zero it. */
    if (PHASE == 1 && !dense && mode == 0u) {
        float* part = reinterpret_cast<float*>(out);
        for (unsigned s = slice; s < nslots; s += nblk) {
            const unsigned e = *reinterpret_cast<const unsigned*>(table + (size_t)s * 8u);
            if (e < E) continue;
            for (unsigned h = threadIdx.x; h < N; h += 256u) part[(size_t)s * N + h] = 0.f;
        }
    }

    const unsigned nchunk = N / 16u, items = gr.U * nchunk;
    const size_t ldw = K / 2u, lds = K / 16u;
    const unsigned nwin = K / 128u;
    /* Weights are prefetched per row PF_WIN windows ahead inside the walk (staging whole up items
     * two items ahead measured 4-12% slower at T = 1..32); scales are staged whole. */
    auto prefetch = [&](unsigned it) {
        if (it >= items || lane != 0u) return;
        const unsigned u = it / nchunk, n0 = (it - u * nchunk) * 16u;
        const size_t row = (size_t)(dense ? 0u : gr.ex[u]) * N + n0;
        nf4_pf_l2(S + row * lds, (unsigned)(16u * lds));
    };
    prefetch(gw);
    prefetch(gw + nw);
    for (unsigned it = gw; it < items; it += nw) {
        prefetch(it + 2u * nw);
        const unsigned u = it / nchunk, n0 = (it - u * nchunk) * 16u;
        const unsigned e = dense ? 0u : gr.ex[u];
        const unsigned cnt = dense ? T : gr.cnt[e];
        const unsigned so = dense ? 0u : gr.off[e];
        if (cnt > T) __trap(); /* a token routed twice to one expert */
        const uint8_t* We = W + (size_t)e * N * ldw;
        const uint8_t* Se = S + (size_t)e * N * lds;
        /* Rows in passes of MT m16 tiles; a later pass re-reads this item's weights from L1/L2. A
         * dense (shared expert) item above 16 rows walks K once for two tiles: the second tile's
         * mma reuses each window's dequantized weights. Each row's mma sequence is the same in
         * either arm (bitwise equal). WIDE (dense only) is its own entry: with the two-tile call
         * in this loop the routed one-tile walk measured +0.4 ms per B=1 step. */
        unsigned mb = 0;
        if (WIDE)
            for (; mb + 16u < cnt; mb += 32u)
                nf4_dec_pass2<PHASE>(out, act, table, We, Se, gs[e], gr, cnt, mb, n0, N, K, ksel,
                                     mode, nwin, ldw, lds, lane);
        for (; mb < cnt; mb += 16u)
            nf4_dec_pass<PHASE, 1>(out, act, table, We, Se, gs[e], gr, dense, so, cnt, mb, n0, N,
                                   K, ksel, mode, nwin, ldw, lds, lane);
    }
}

/* ---------------------------------------------------------------------------------------------
 * ALIGN (op 244): CSR of the T*k slots by expert.
 * meta (i32): [0,E) rowoff, [E,2E) cnt, [2E,3E+1) prefix of ceil(cnt/64) m-tiles.
 * Rows are unpadded; total rows = slots routed to an expert < E.
 * Every block counts all slots (and those before its own slice) in shared memory, so the
 * scattered row writes spread over the grid without a grid barrier: block b writes the slots of
 * its slice at rowoff[e] + (count of e before the slice) + a block-local atomic rank. Block 0
 * writes meta. Row order inside an expert is arbitrary; no output depends on it. */
static __device__ void nf4_moe_align(int* __restrict__ meta, const unsigned char* __restrict__ table,
                                     unsigned* __restrict__ row_token, unsigned* __restrict__ row_partidx,
                                     float* __restrict__ row_gate, unsigned T, unsigned E,
                                     unsigned ksel, unsigned slice, unsigned nblk, unsigned* arena) {
    const unsigned S = T * ksel, chunk = (S + nblk - 1u) / nblk;
    const unsigned sb = min(slice * chunk, S), se = min(sb + chunk, S);
    if (sb >= se && slice != 0u) return;
    unsigned* cnt = arena;
    unsigned* pre = arena + E;
    unsigned* fill = arena + 2u * E;
    __shared__ unsigned s_ws[8], s_wt[8];
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    for (unsigned e = tid; e < E; e += 256u) cnt[e] = pre[e] = 0u;
    __syncthreads();
    constexpr unsigned U = 8u; /* table loads in flight per thread */
    for (unsigned s0 = 0; s0 < S; s0 += 256u * U) {
        unsigned e[U];
#pragma unroll
        for (unsigned u = 0; u < U; u++) {
            const unsigned s = s0 + u * 256u + tid;
            e[u] = s < S ? *reinterpret_cast<const unsigned*>(table + (size_t)s * 8u) : ~0u;
        }
#pragma unroll
        for (unsigned u = 0; u < U; u++) {
            if (e[u] >= E) continue;
            atomicAdd(&cnt[e[u]], 1u);
            if (s0 + u * 256u + tid < sb) atomicAdd(&pre[e[u]], 1u);
        }
    }
    __syncthreads();
    const unsigned per = (E + 255u) / 256u, e0 = tid * per, e1 = min(e0 + per, E);
    unsigned rs = 0, rt = 0;
    for (unsigned e = e0; e < e1; e++) { rs += cnt[e]; rt += (cnt[e] + 63u) / 64u; }
    unsigned is = rs, it = rt;
#pragma unroll
    for (unsigned d = 1; d < 32u; d <<= 1) {
        const unsigned a = __shfl_up_sync(0xFFFFFFFFu, is, d), b = __shfl_up_sync(0xFFFFFFFFu, it, d);
        if (lane >= d) { is += a; it += b; }
    }
    if (lane == 31u) { s_ws[warp] = is; s_wt[warp] = it; }
    __syncthreads();
    unsigned bs = 0, bt = 0, TT = 0;
    for (unsigned w = 0; w < 8u; w++) {
        if (w < warp) { bs += s_ws[w]; bt += s_wt[w]; }
        TT += s_wt[w];
    }
    unsigned ps = bs + is - rs, pt = bt + it - rt;
    for (unsigned e = e0; e < e1; e++) {
        if (slice == 0u) {
            meta[e] = (int)ps;
            meta[E + e] = (int)cnt[e];
            meta[2u * E + e] = (int)pt;
        }
        fill[e] = ps + pre[e];
        ps += cnt[e];
        pt += (cnt[e] + 63u) / 64u;
    }
    if (slice == 0u && tid == 0) meta[3u * E] = (int)TT;
    __syncthreads();
    for (unsigned s = sb + tid; s < se; s += 256u) {
        const uint2 v = *reinterpret_cast<const uint2*>(table + (size_t)s * 8u);
        if (v.x >= E) continue;
        const unsigned r = atomicAdd(&fill[v.x], 1u);
        row_token[r] = s / ksel;
        row_partidx[r] = s;
        row_gate[r] = __uint_as_float(v.y);
    }
}

/* ---------------------------------------------------------------------------------------------
 * Tiled grouped GEMM (ops 240 at T > 16, 245, 246): C[64 rows x 128 cols] per item, BK=32 double
 * buffer. A (bf16 activation rows, gathered) by cp.async; B by cp.async (MODE 0, bf16 rows) or
 * nvfp4 global -> registers -> exact bf16 dequant -> smem (MODE 1/2). 8 warps = 2 (m) x 4 (n),
 * warp tile 32 x 32, ldmatrix A/B, mma m16n8k16.
 * MODE 0 router: f32 logits[r][n]. MODE 1 up: fu_g[r][n] = bf16(relu(gs*acc)^2).
 * MODE 2 down: part[row_partidx[r]][n] = gs*row_gate[r]*acc (dmode 0) or bf16 out[r][n] (dmode 1). */
static __device__ __forceinline__ unsigned nf4_expert_of_tile(const int* tilep, unsigned mt, unsigned E) {
    unsigned lo = 0, hi = E; /* largest e with tilep[e] <= mt */
    while (hi - lo > 1u) {
        const unsigned mid = (lo + hi) >> 1;
        if ((unsigned)tilep[mid] <= mt) lo = mid; else hi = mid;
    }
    return lo;
}

template <int MODE>
static __device__ void nf4_tiled(void* __restrict__ out, const __nv_bfloat16* __restrict__ act,
                                 const int* __restrict__ meta, const void* __restrict__ Wv,
                                 const uint8_t* __restrict__ S, const float* __restrict__ gs,
                                 const unsigned* __restrict__ row_token,
                                 const unsigned* __restrict__ row_partidx,
                                 const float* __restrict__ row_gate, unsigned T, unsigned N,
                                 unsigned K, unsigned E, unsigned dmode, unsigned slice,
                                 unsigned nblk, __nv_bfloat16* arena) {
    __nv_bfloat16* As = arena;
    __nv_bfloat16* Bs = arena + 2 * NF4_BM * NF4_ST;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned wm = warp >> 2, wn = warp & 3u, g = lane >> 2, t = lane & 3u;
    const unsigned ntn = N / NF4_BN;
    const unsigned mtiles = meta ? (unsigned)meta[3u * E] : (T + NF4_BM - 1u) / NF4_BM;
    const unsigned items = mtiles * ntn, nk = K / NF4_BK;
    const size_t ldw = MODE == 0 ? (size_t)K * 2u : K / 2u, lds = K / 16u;
    /* A staging role: row tid/4, 16 B chunk tid%4. B: row tid/2, half tid%2 (16 k). */
    const unsigned ar = tid >> 2, ac = tid & 3u, br = tid >> 1, bh = tid & 1u;
    /* Stage the next item's weight block (128 contiguous rows + scales) into L2: the block holds
     * only one B stage (8 B / thread) in flight on its own. */
    auto prefetch = [&](unsigned item) {
        if (MODE == 0 || item >= items || tid != 0u) return;
        const unsigned mt = item / ntn, n0 = (item - mt * ntn) * NF4_BN;
        const size_t row = (size_t)(meta ? nf4_expert_of_tile(meta + 2u * E, mt, E) : 0u) * N + n0;
        if (NF4_BN * ldw <= 65536u)
            nf4_pf_l2(reinterpret_cast<const uint8_t*>(Wv) + row * ldw, (unsigned)(NF4_BN * ldw));
        nf4_pf_l2(S + row * lds, (unsigned)(NF4_BN * lds));
    };
    /* Larger blocks (down: 168 KiB) are prefetched per row 8 k-steps ahead instead. */
    const bool line_pf = MODE != 0 && NF4_BN * ldw > 65536u;
    prefetch(slice);
    for (unsigned item = slice; item < items; item += nblk) {
        prefetch(item + nblk);
        const unsigned mt = item / ntn, n0 = (item - mt * ntn) * NF4_BN;
        unsigned e = 0, r0, rc;
        if (meta) {
            e = nf4_expert_of_tile(meta + 2u * E, mt, E);
            const unsigned lt = mt - (unsigned)meta[2u * E + e];
            r0 = (unsigned)meta[e] + lt * NF4_BM;
            rc = min((unsigned)NF4_BM, (unsigned)meta[E + e] - lt * NF4_BM);
        } else {
            r0 = mt * NF4_BM;
            rc = min((unsigned)NF4_BM, T - r0);
        }
        const bool av = ar < rc;
        unsigned arow = r0 + (av ? ar : 0u);
        if (MODE == 1 && meta) arow = row_token[arow];
        const __nv_bfloat16* asrc = act + (size_t)arow * K + ac * 8u;
        const uint8_t* wrow = reinterpret_cast<const uint8_t*>(Wv) + ((size_t)e * N + n0 + br) * ldw;
        const uint8_t* srow = MODE == 0 ? nullptr : S + ((size_t)e * N + n0 + br) * lds + 2u * bh;

        float acc[2][4][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;

        /* BK = 64: A row ar gets chunks ac and ac+4 (16 B each); B row br, half bh = 32 k
         * (16 B fp4 + 2 scale bytes, or 64 B bf16). */
        uint4 bq = make_uint4(0, 0, 0, 0);
        unsigned bsc = 0;
        auto stage_a = [&](unsigned ks, unsigned buf) {
            __nv_bfloat16* ad = As + buf * NF4_BM * NF4_ST + ar * NF4_ST + ac * 8u;
            nf4_cp16(ad, asrc + ks * NF4_BK, av ? 16u : 0u);
            nf4_cp16(ad + 32, asrc + ks * NF4_BK + 32, av ? 16u : 0u);
            if (MODE == 0) {
                const uint8_t* src = wrow + (size_t)ks * NF4_BK * 2u + bh * 64u;
                __nv_bfloat16* dst = Bs + buf * NF4_BN * NF4_ST + br * NF4_ST + bh * 32u;
#pragma unroll
                for (int q = 0; q < 4; q++) nf4_cp16(dst + 8 * q, src + 16 * q, 16u);
            }
        };
        auto load_b = [&](unsigned ks) {
            if (MODE != 0) {
                bq = __ldg(reinterpret_cast<const uint4*>(wrow + (size_t)ks * (NF4_BK / 2) + bh * 16u));
                bsc = __ldg(reinterpret_cast<const unsigned short*>(srow + ks * (NF4_BK / 16)));
            }
        };
        auto store_b = [&](unsigned buf) {
            if (MODE != 0) {
                const unsigned s0 = nf4_sc2(bsc), s1 = nf4_sc2(bsc >> 8);
                uint4 o[4];
                nf4_deq4(bq.x & 0xFFFFu, s0, o[0].x, o[0].y);
                nf4_deq4(bq.x >> 16, s0, o[0].z, o[0].w);
                nf4_deq4(bq.y & 0xFFFFu, s0, o[1].x, o[1].y);
                nf4_deq4(bq.y >> 16, s0, o[1].z, o[1].w);
                nf4_deq4(bq.z & 0xFFFFu, s1, o[2].x, o[2].y);
                nf4_deq4(bq.z >> 16, s1, o[2].z, o[2].w);
                nf4_deq4(bq.w & 0xFFFFu, s1, o[3].x, o[3].y);
                nf4_deq4(bq.w >> 16, s1, o[3].z, o[3].w);
                uint4* dst = reinterpret_cast<uint4*>(Bs + buf * NF4_BN * NF4_ST + br * NF4_ST + bh * 32u);
#pragma unroll
                for (int q = 0; q < 4; q++) dst[q] = o[q];
            }
        };

        stage_a(0, 0);
        nf4_cp_commit();
        load_b(0);
        store_b(0);
        for (unsigned ks = 0; ks < nk; ks++) {
            const unsigned cb = ks & 1u, nb = cb ^ 1u;
            nf4_cp_wait0();
            __syncthreads();
            if (ks + 1u < nk) {
                stage_a(ks + 1u, nb);
                nf4_cp_commit();
                load_b(ks + 1u);
            }
            if (line_pf && bh == 0u && (ks & 3u) == 0u && ks + 8u < nk)
                nf4_pf_line(wrow + (size_t)(ks + 8u) * (NF4_BK / 2));
            const __nv_bfloat16* Ab = As + cb * NF4_BM * NF4_ST;
            const __nv_bfloat16* Bb = Bs + cb * NF4_BN * NF4_ST;
#pragma unroll
            for (unsigned kk = 0; kk < NF4_BK; kk += 16u) {
                unsigned af[2][4], bf[2][4];
#pragma unroll
                for (int mi = 0; mi < 2; mi++)
                    nf4_ldsm_x4(af[mi], Ab + (wm * 32u + mi * 16u + (lane & 15u)) * NF4_ST + kk + (lane >> 4) * 8u);
#pragma unroll
                for (int p = 0; p < 2; p++)
                    nf4_ldsm_x4(bf[p], Bb + (wn * 32u + p * 16u + (lane & 7u) + ((lane >> 4) << 3)) * NF4_ST +
                                           kk + ((lane >> 3) & 1u) * 8u);
#pragma unroll
                for (int mi = 0; mi < 2; mi++)
#pragma unroll
                    for (int nj = 0; nj < 4; nj++)
                        nf4_mma(acc[mi][nj], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                bf[nj >> 1][(nj & 1) * 2], bf[nj >> 1][(nj & 1) * 2 + 1]);
            }
            if (ks + 1u < nk) store_b(nb);
        }
        const float sc = MODE == 0 ? 1.0f : gs[e];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned lr = wm * 32u + mi * 16u + g + 8u * h;
                if (lr >= rc) continue;
                const unsigned r = r0 + lr;
                size_t orow = r;
                float rg = 1.0f;
                if (MODE == 2 && dmode == 0u) { orow = row_partidx[r]; rg = row_gate[r]; }
#pragma unroll
                for (int nj = 0; nj < 4; nj++) {
                    const unsigned col = n0 + wn * 32u + nj * 8u + 2u * t;
                    const float y0 = acc[mi][nj][2 * h] * sc, y1 = acc[mi][nj][2 * h + 1] * sc;
                    if (MODE == 0) {
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + orow * N + col) = make_float2(y0, y1);
                    } else if (MODE == 1) {
                        *reinterpret_cast<unsigned*>(reinterpret_cast<__nv_bfloat16*>(out) + orow * N + col) =
                            nf4_f2bf2(nf4_relu2(y0), nf4_relu2(y1));
                    } else if (dmode == 0u) {
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + orow * N + col) =
                            make_float2(y0 * rg, y1 * rg);
                    } else {
                        *reinterpret_cast<unsigned*>(reinterpret_cast<__nv_bfloat16*>(out) + orow * N + col) =
                            nf4_f2bf2(y0, y1);
                    }
                }
            }
        __syncthreads();
    }
}

/* ---------------------------------------------------------------------------------------------
 * Op entry points (argument order = dev_isa.h operand order). */
static __device__ __noinline__ void d_moe_router_logits_f32(float* logits, const __nv_bfloat16* x,
                                               const __nv_bfloat16* W, unsigned T, unsigned E,
                                               unsigned H, unsigned KS, unsigned slice,
                                               unsigned nblk, float* arena) {
    KS = KS ? KS : 1u;
    if (E % NF4_BN || H % (32u * KS) || (T > NF4_DEC_MAXROWS && KS != 1u) || blockDim.x != 256u) {
        __trap();
        return;
    }
    if (T <= NF4_DEC_MAXROWS)
        nf4_router_logits_dec(logits, x, W, T, E, H, KS, slice, nblk);
    else
        nf4_tiled<0>(logits, x, nullptr, W, nullptr, nullptr, nullptr, nullptr, nullptr, T, E, H,
                     1u, 0u, slice, nblk, reinterpret_cast<__nv_bfloat16*>(arena));
}

static __device__ __noinline__ void d_moe_router_sigmoid_topk(unsigned char* table, const float* logits,
                                                 const float* bias, const int* active,
                                                 unsigned T, unsigned E,
                                                 unsigned k, unsigned KS, unsigned flags,
                                                 float scale, unsigned slice, unsigned nblk,
                                                 float* arena) {
    if (E % 32u || E > 1024u || k == 0u || k > 32u || k > E || blockDim.x != 256u) {
        __trap();
        return;
    }
    nf4_router_topk(table, logits, bias, KS > 1u ? active : nullptr, T, E, k, KS ? KS : 1u, flags,
                    scale == 0.f ? 1.f : scale, slice, nblk, arena);
}

template <bool WIDE>
static __device__ __noinline__ void d_moe_up_nvfp4_t(__nv_bfloat16* fu, const __nv_bfloat16* x,
                                      const unsigned char* table, const uint8_t* W,
                                      const uint8_t* S, const float* gs, unsigned T, unsigned ksel,
                                      unsigned I, unsigned K, unsigned E, unsigned slice,
                                      unsigned nblk, float* arena) {
    if (T > NF4_DEC_MAXROWS || (table && T * ksel > NF4_DEC_MAXSLOTS) || (!table && E != 1u) ||
        (WIDE && table) || K % 128u || I % 16u || blockDim.x != 256u) {
        __trap();
        return;
    }
    nf4_moe_dec<0, WIDE>(fu, x, table, W, S, gs, T, ksel, I, K, E, 0u, slice, nblk,
                         reinterpret_cast<unsigned*>(arena));
}

template <bool WIDE>
static __device__ __noinline__ void d_moe_down_nvfp4_t(void* out, const __nv_bfloat16* fu,
                                        const unsigned char* table, const uint8_t* W,
                                        const uint8_t* S, const float* gs, unsigned T,
                                        unsigned ksel, unsigned H, unsigned I, unsigned E,
                                        unsigned mode, unsigned slice, unsigned nblk, float* arena) {
    if (T > NF4_DEC_MAXROWS || (table && T * ksel > NF4_DEC_MAXSLOTS) || (!table && E != 1u) ||
        (WIDE && table) || (mode == 0u) != (table != nullptr) || I % 128u || H % 16u ||
        blockDim.x != 256u) {
        __trap();
        return;
    }
    nf4_moe_dec<1, WIDE>(out, fu, table, W, S, gs, T, ksel, H, I, E, mode, slice, nblk,
                         reinterpret_cast<unsigned*>(arena));
}

static __device__ __forceinline__ void d_moe_up_nvfp4(__nv_bfloat16* fu, const __nv_bfloat16* x,
                                      const unsigned char* table, const uint8_t* W,
                                      const uint8_t* S, const float* gs, unsigned T, unsigned ksel,
                                      unsigned I, unsigned K, unsigned E, unsigned slice,
                                      unsigned nblk, float* arena) {
    if (!table && T > NF4_DEC_WIDE_ROWS)
        d_moe_up_nvfp4_t<true>(fu, x, table, W, S, gs, T, ksel, I, K, E, slice, nblk, arena);
    else
        d_moe_up_nvfp4_t<false>(fu, x, table, W, S, gs, T, ksel, I, K, E, slice, nblk, arena);
}

static __device__ __forceinline__ void d_moe_down_nvfp4(void* out, const __nv_bfloat16* fu,
                                        const unsigned char* table, const uint8_t* W,
                                        const uint8_t* S, const float* gs, unsigned T,
                                        unsigned ksel, unsigned H, unsigned I, unsigned E,
                                        unsigned mode, unsigned slice, unsigned nblk, float* arena) {
    if (!table && T > NF4_DEC_WIDE_ROWS)
        d_moe_down_nvfp4_t<true>(out, fu, table, W, S, gs, T, ksel, H, I, E, mode, slice, nblk, arena);
    else
        d_moe_down_nvfp4_t<false>(out, fu, table, W, S, gs, T, ksel, H, I, E, mode, slice, nblk, arena);
}

static __device__ __noinline__ void d_moe_align_nvfp4(int* meta, const unsigned char* table, unsigned* row_token,
                                         unsigned* row_partidx, float* row_gate, unsigned T,
                                         unsigned E, unsigned ksel, unsigned slice, unsigned nblk,
                                         float* arena) {
    if (blockDim.x != 256u || E > 65535u) { __trap(); return; }
    nf4_moe_align(meta, table, row_token, row_partidx, row_gate, T, E, ksel, slice, nblk,
                  reinterpret_cast<unsigned*>(arena));
}

static __device__ __noinline__ void d_moe_up_nvfp4_pf(__nv_bfloat16* fug, const __nv_bfloat16* x,
                                         const int* meta, const uint8_t* W, const uint8_t* S,
                                         const float* gs, const unsigned* row_token, unsigned T,
                                         unsigned I, unsigned K, unsigned E, unsigned slice,
                                         unsigned nblk, float* arena) {
    if (K % NF4_BK || I % NF4_BN || (!meta && E != 1u) || (meta && !row_token) ||
        blockDim.x != 256u) {
        __trap();
        return;
    }
    nf4_tiled<1>(fug, x, meta, W, S, gs, row_token, nullptr, nullptr, T, I, K, E, 0u, slice, nblk,
                 reinterpret_cast<__nv_bfloat16*>(arena));
}

static __device__ __noinline__ void d_moe_down_nvfp4_pf(void* out, const __nv_bfloat16* fug, const int* meta,
                                           const uint8_t* W, const uint8_t* S, const float* gs,
                                           const unsigned* row_partidx, const float* row_gate,
                                           unsigned T, unsigned H, unsigned I, unsigned E,
                                           unsigned mode, unsigned slice, unsigned nblk,
                                           float* arena) {
    if (I % NF4_BK || H % NF4_BN || (!meta && E != 1u) || (mode == 0u) != (meta != nullptr) ||
        (meta && (!row_partidx || !row_gate)) || blockDim.x != 256u) {
        __trap();
        return;
    }
    nf4_tiled<2>(out, fug, meta, W, S, gs, nullptr, row_partidx, row_gate, T, H, I, E, mode,
                 slice, nblk, reinterpret_cast<__nv_bfloat16*>(arena));
}

#endif /* PLOW_NV_OP_MOE_NVFP4_CUH */

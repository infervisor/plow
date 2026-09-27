/* op_gemv_wide_sm90.cuh — wide-batch decode GEMV (PLOW_NV_GEMV_WIDE, sm_90a): C[m][n] = x[m][:] . W[n][:]
 * for up to 128 activation rows in ONE weight pass.
 *
 * WHY. The mma.sync walk (op_gemv_mma.cuh) re-reads the weights once per GV_MM_MAX rows, and each
 * of its warps loads every activation row it multiplies straight from L1/L2: at 128 rows the
 * activation traffic is ~16x the weight traffic and the walk runs at 12-20% of the HBM roofline
 * (Veena B=128 ctx 1024: 13 ms of GEMV against a 2.2 ms weight stream).
 *
 * SHAPE. Swap-AB warpgroup MMA m64n64k16: a 64-row weight slab is wgmma's A, 64 activation rows
 * its B, both staged K-major into 128B-swizzled smem by a cp.async ring. Up to 64 rows (SB=false)
 * the two warpgroups take different slabs of a 128-row weight tile against the same activations;
 * up to 128 (SB=true) they share one 64-row slab and split the activations. Either way a thread
 * holds 32 accumulators: an m64n128 accumulator inlined into the decode entry cost its other arms
 * more (flash-decode +40% at B=64) than it saved. GLU slabs interleave gate and up rows in 8-row
 * groups, so each thread holds both halves of its outputs (no exchange). Outputs are stored
 * straight from the accumulators: the ring is the whole smem claim, which keeps the launch under
 * the carve-out step where L1 shrinks to 28 KiB and flash-decode slows ~1.7x.
 *
 * STREAM-K. The (tile, k-block) space is cut into equal contiguous ranges, one per block, so every
 * block streams the same weight bytes whatever N is (N=1024..8192 is 8..64 tiles over 132 SMs).
 * A tile split across blocks is reduced through a global fp32 workspace: every contributor parks
 * its fragment, then each reduces an equal share of the tile, summing in block order (bit-stable).
 * Output rows are therefore written by other blocks than their owner — consumers must wait on the
 * whole op (coarse gates), which is what dense decoders compile to.
 *
 * CONTRACT. K % 64 == 0; QKV additionally Nq % 64 == Nk % 64 == 0 (a slab never straddles two
 * matrices); nblk <= GW_MAXB; tiles <= GW_MAXT; arena >= gw_arena_bytes(MP). Callers check. */
#pragma once
#include "sm90_wgmma.cuh"

#ifndef PLOW_NV_GW_STAGES
#define PLOW_NV_GW_STAGES 4
#endif
#define GW_BK 64
#define GW_MAXB 160
#define GW_MAXT 4096

/* A stage is 192 staged rows of 64 k either way: 128 weight x 64 activation rows, or 64 x 128. */
__host__ __device__ constexpr unsigned gw_arena_bytes() {
    return PLOW_NV_GW_STAGES * 192u * GW_BK * 2u + 1024u;
}
#define GW_ARENA_FLOATS_MAX ((gw_arena_bytes() + 3u) / 4u)

/* Every block's two parked fragments (256 threads x 32 f32). */
__device__ float plow_gw_ws[GW_MAXB * 2 * 256 * 32];
/* Per tile: [0] fragments parked, [1] reducers done. Zero between ops (the last reducer of a
 * tile resets its pair). */
__device__ unsigned plow_gw_ctr[2 * GW_MAXT];

struct gw_mma {
    static __device__ __forceinline__ void run(float* d, uint64_t da, uint64_t db, int s) {
        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
            "%32, %33, p, 1, 1, 0, 0; }\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
              "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
              "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
              "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
              "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
              "+f"(d[30]), "+f"(d[31])
            : "l"(da), "l"(db), "r"(s)
            : "memory");
    }
};

enum { GW_PLAIN = 0, GW_GLU = 1, GW_QKV = 2 };

struct GwArgs {
    const __nv_bfloat16* x;
    const __nv_bfloat16* W[3]; /* plain: W; GLU: gate, up; QKV: q, k, v */
    __nv_bfloat16* C[3];
    unsigned N[3];
    unsigned M, K, act;
};

static __device__ __forceinline__ __nv_bfloat16 gemma_glu_epilogue(float gate, float up,
                                                                  unsigned act);

/* SB = 0: 128 weight rows (a slab per warpgroup) x 64 activation rows; SB = 1 (split batch): one
 * 64-row slab shared, 128 activation rows split between the warpgroups. (A 128 x 128 tile with an
 * m64n128 accumulator reads the activations half as often and streams the 156951-row head 1.3x
 * faster standalone, but in the decode entry it is 5-10% slower per step at B=64/128.) */
template <int SB> __host__ __device__ constexpr unsigned gw_tn() { return SB ? 64u : 128u; }
template <int SB> __host__ __device__ constexpr unsigned gw_xr() { return SB ? 128u : 64u; }

/* Weight row `r` of tile `t`, or nullptr past the end. A GLU slab row r is output
 * TN/2 t + 32 (r/64) + 8 (r/16 % 4) + r % 8 of gate (r % 16 < 8) or up. */
template <int MODE, int SB>
__device__ __forceinline__ const __nv_bfloat16* gw_row(const GwArgs& a, unsigned t, unsigned r) {
    constexpr unsigned TN = gw_tn<SB>();
    if constexpr (MODE == GW_GLU) {
        const unsigned n = TN / 2u * t + 32u * (r >> 6) + 8u * ((r >> 4) & 3u) + (r & 7u);
        return n < a.N[0] ? ((r & 8u) ? a.W[1] : a.W[0]) + (size_t)n * a.K : nullptr;
    } else if constexpr (MODE == GW_QKV) {
        unsigned g = TN * t + r;
        if (g < a.N[0]) return a.W[0] + (size_t)g * a.K;
        g -= a.N[0];
        if (g < a.N[1]) return a.W[1] + (size_t)g * a.K;
        g -= a.N[1];
        return g < a.N[2] ? a.W[2] + (size_t)g * a.K : nullptr;
    } else {
        const unsigned n = TN * t + r;
        return n < a.N[0] ? a.W[0] + (size_t)n * a.K : nullptr;
    }
}

template <int MODE, int SB>
__device__ __forceinline__ unsigned gw_tiles(const GwArgs& a) {
    constexpr unsigned TN = gw_tn<SB>();
    if constexpr (MODE == GW_GLU) return (a.N[0] + TN / 2u - 1u) / (TN / 2u);
    else if constexpr (MODE == GW_QKV) return (a.N[0] + a.N[1] + a.N[2] + TN - 1u) / TN;
    else return (a.N[0] + TN - 1u) / TN;
}

/* First unit of block c's stream-K range: floor(c L / B). */
__device__ __forceinline__ unsigned gw_first_unit(unsigned c, unsigned L, unsigned B) {
    return (unsigned)(((unsigned long long)c * L) / B);
}
/* Block whose range holds unit u. */
__device__ __forceinline__ unsigned gw_owner(unsigned u, unsigned L, unsigned B) {
    unsigned c = (unsigned)(((unsigned long long)u * B) / L);
    while (c + 1u < B && gw_first_unit(c + 1u, L, B) <= u) c++;
    while (c > 0u && gw_first_unit(c, L, B) > u) c--;
    return c;
}

/* Store accumulator element (thread th, register j) of tile t. GLU: j is a gate register
 * (bit 1 clear) and u the value of its up twin j | 2 in the same thread. */
template <int MODE, int SB>
__device__ __forceinline__ void gw_put(const GwArgs& a, unsigned t, unsigned th, unsigned j,
                                       float v, float u) {
    constexpr unsigned TN = gw_tn<SB>();
    const unsigned wg = th >> 7, wiw = (th >> 5) & 3u, lane = th & 31u;
    const unsigned m = (SB ? 64u * wg : 0u) + 8u * (j >> 2) + 2u * (lane & 3u) + (j & 1u);
    if (m >= a.M) return;
    if constexpr (MODE == GW_GLU) {
        const unsigned n = TN / 2u * t + (SB ? 0u : 32u * wg) + 8u * wiw + (lane >> 2);
        if (n < a.N[0]) a.C[0][(size_t)m * a.N[0] + n] = gemma_glu_epilogue(v, u, a.act);
    } else {
        unsigned n = TN * t + (SB ? 0u : 64u * wg) + 16u * wiw + (lane >> 2) + 8u * ((j >> 1) & 1u);
        unsigned s = 0;
        if constexpr (MODE == GW_QKV) {
            if (n >= a.N[0]) { n -= a.N[0]; s = 1; if (n >= a.N[1]) { n -= a.N[1]; s = 2; } }
        }
        const unsigned N = s == 0 ? a.N[0] : (s == 1 ? a.N[1] : a.N[2]);
        __nv_bfloat16* C = s == 0 ? a.C[0] : (s == 1 ? a.C[1] : a.C[2]);
        if (n < N) C[(size_t)m * N + n] = __float2bfloat16(v);
    }
}

template <int MODE, int SB>
__device__ __forceinline__ void gw_store(const GwArgs& a, const float (&acc)[32], unsigned t) {
    const unsigned tid = threadIdx.x;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        if constexpr (MODE == GW_GLU) {
            if (!(j & 2)) gw_put<MODE, SB>(a, t, tid, j, acc[j], acc[j | 2]);
        } else {
            gw_put<MODE, SB>(a, t, tid, j, acc[j], 0.0f);
        }
    }
}

__device__ __forceinline__ unsigned gw_ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

/* Contributors [first, last] of tile t, and the parked slot of `first` (the only one that can
 * have entered the tile mid-range: every later contributor's range starts inside it). */
struct GwTile { unsigned first, last, first_slot; };
__device__ __forceinline__ GwTile gw_tile(unsigned t, unsigned KB, unsigned L, unsigned B) {
    const unsigned first = gw_owner(t * KB, L, B);
    return {first, gw_owner(t * KB + KB - 1u, L, B), gw_first_unit(first, L, B) / KB == t ? 0u : 1u};
}

template <int MODE, int SB>
__device__ void gemv_wide_sm90(const GwArgs& a, unsigned slice, unsigned nblk, void* arena) {
    constexpr int S = PLOW_NV_GW_STAGES;
    constexpr unsigned TN = gw_tn<SB>(), XR = gw_xr<SB>();     /* staged weight / activation rows */
    constexpr unsigned NACC = 32;
    constexpr unsigned WBUF = TN * GW_BK, XBUF = XR * GW_BK;    /* bf16 per stage */
    __nv_bfloat16* base = (__nv_bfloat16*)sm90_align1024(arena);
    __nv_bfloat16* Ws = base;
    __nv_bfloat16* Xs = base + S * WBUF;
    const unsigned tid = threadIdx.x, wg = tid >> 7;
    float* const ws = plow_gw_ws;
    unsigned* const ctr = plow_gw_ctr;
    const unsigned KB = a.K / GW_BK, T = gw_tiles<MODE, SB>(a);
    const unsigned L = T * KB;
    const unsigned u0 = gw_first_unit(slice, L, nblk), u1 = gw_first_unit(slice + 1u, L, nblk);
    const unsigned nu = u1 - u0;

    const unsigned crow = tid >> 3, cc = tid & 7u;
    const __nv_bfloat16* xr[XR / 32];
#pragma unroll
    for (int j = 0; j < (int)(XR / 32); j++) {
        const unsigned m = crow + 32u * j;
        xr[j] = m < a.M ? a.x + (size_t)m * a.K + cc * 8u : nullptr;
    }
    /* One continuous ring over this block's units: the next tile's loads are in flight while a
     * finished tile drains. The staging cursor runs S-1 units ahead of the compute cursor. */
    unsigned st_t = u0 / KB, st_kb = u0 % KB;
    const __nv_bfloat16* wr[TN / 32];
    auto rows = [&]() {
#pragma unroll
        for (int j = 0; j < (int)(TN / 32); j++) {
            const __nv_bfloat16* p = gw_row<MODE, SB>(a, st_t, crow + 32u * j);
            wr[j] = p ? p + cc * 8u : nullptr;
        }
    };
    rows();
    auto stage = [&](unsigned buf) {
        const unsigned k = st_kb * GW_BK;
        __nv_bfloat16* w = Ws + buf * WBUF;
        __nv_bfloat16* xs = Xs + buf * XBUF;
#pragma unroll
        for (int j = 0; j < (int)(TN / 32); j++)
            sm90_cp16(&w[sm90_swz_off<GW_BK, 8>(crow + 32 * j, cc)], wr[j] ? wr[j] + k : a.x,
                      wr[j] ? 16 : 0);
#pragma unroll
        for (int j = 0; j < (int)(XR / 32); j++)
            sm90_cp16(&xs[sm90_swz_off<GW_BK, 8>(crow + 32 * j, cc)], xr[j] ? xr[j] + k : a.x,
                      xr[j] ? 16 : 0);
        if (++st_kb == KB) { st_kb = 0; st_t++; rows(); }
    };
    /* Split tiles this block parked a fragment for: its first and last segment at most. */
    unsigned split0 = 0xffffffffu, split1 = 0xffffffffu;

    float acc[NACC];
#pragma unroll
    for (int s = 0; s < S - 1; s++) {
        if ((unsigned)s < nu) stage(s);
        sm90_cp_commit();
    }
    unsigned t = u0 / KB, kb = u0 % KB;
    for (unsigned i = 0; i < nu; i++) {
        const bool seg_first = (i == 0u) || kb == 0u;
        const bool seg_last = (i + 1u == nu) || kb + 1u == KB;
        const unsigned cur = i % S;
        sm90_cp_wait<S - 2>();
        __syncthreads();
        const __nv_bfloat16* Wc = Ws + cur * WBUF + (SB ? 0u : wg * 64u * GW_BK);
        const __nv_bfloat16* Xc = Xs + cur * XBUF + (SB ? wg * 64u * GW_BK : 0u);
        sm90_wg_fence();
#pragma unroll
        for (int sub = 0; sub < GW_BK / 16; sub++)
            gw_mma::run(acc, sm90_desc(Wc + sub * 16), sm90_desc(Xc + sub * 16),
                        (seg_first && sub == 0) ? 0 : 1);
        sm90_wg_commit();
        sm90_wg_wait<1>();
        __syncthreads();
        if (i + S - 1 < nu) stage((i + S - 1) % S);
        sm90_cp_commit();
        if (seg_last) {
            sm90_wg_wait<0>();
            if (kb + 1u == KB && i + 1u >= KB) {
                gw_store<MODE, SB>(a, acc, t);
            } else {
                /* Park the fragment; slot 0 = this block's first segment, 1 = its last. */
                const unsigned slot = (t == u0 / KB) ? 0u : 1u;
                float* mine = ws + (size_t)(slice * 2u + slot) * NACC * 256u;
#pragma unroll
                for (int j = 0; j < (int)NACC; j++) __stcg(mine + (size_t)j * 256u + tid, acc[j]);
                __threadfence();
                __syncthreads();
                if (tid == 0) atomicAdd(&ctr[2u * t], 1u);
                if (slot == 0u) split0 = t; else split1 = t;
            }
        }
        if (++kb == KB) { kb = 0; t++; }
    }
    sm90_cp_wait<0>();

    /* Every contributor of a split tile reduces an equal share of it, summing the parked
     * fragments in block order (bit-stable), once all of them are parked. All blocks of the op
     * are co-resident and none waits on anything later in the program, so the wait terminates.
     * Both tiles' waits come first: the second is usually complete by then. */
    if (tid == 0) {
        for (unsigned q = 0; q < 2u; q++) {
            const unsigned tq = q == 0u ? split0 : split1;
            if (tq == 0xffffffffu) continue;
            const GwTile g = gw_tile(tq, KB, L, nblk);
            while (gw_ld_acquire(&ctr[2u * tq]) < g.last - g.first + 1u) {
            }
        }
    }
    __syncthreads();
    __threadfence();
#pragma unroll 1
    for (unsigned q = 0; q < 2u; q++) {
        const unsigned tq = q == 0u ? split0 : split1;
        if (tq == 0xffffffffu) continue;
        const GwTile g = gw_tile(tq, KB, L, nblk);
        const unsigned nc = g.last - g.first + 1u, me = slice - g.first;
        /* Units of 4 consecutive fragment threads (one float4) of one register; GLU takes the
         * gate registers only and loads each one's up twin (register + 2) beside it. */
        constexpr unsigned NJ = MODE == GW_GLU ? NACC / 2u : NACC;
        constexpr unsigned E4 = 64u * NJ;
        const unsigned p0 = (me * E4) / nc, p1 = ((me + 1u) * E4) / nc;
        /* Loads are issued CU contributors at a time before any is summed, so a share costs a
         * few L2 round trips rather than one per contributor; the sum stays in block order. */
        constexpr int CU = MODE == GW_GLU ? 4 : 8;
        for (unsigned pb = p0 + tid; pb < p1; pb += 2u * blockDim.x) {
            float4 v[2], w[2];
            unsigned jr[2], th4[2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                v[h] = w[h] = make_float4(0.f, 0.f, 0.f, 0.f);
                const unsigned p = pb + (unsigned)h * blockDim.x, jj = p / 64u;
                jr[h] = MODE == GW_GLU ? 4u * (jj >> 1) + (jj & 1u) : jj;
                th4[h] = (p % 64u) * 4u;
            }
            for (unsigned c0 = g.first; c0 <= g.last; c0 += CU) {
                float4 lv[CU][2], lw[CU][2];
#pragma unroll
                for (int q2 = 0; q2 < CU; q2++) {
                    const unsigned c = c0 + q2;
                    const unsigned cs = c == g.first ? g.first_slot : 0u;
                    const float* cb = ws + (size_t)(c * 2u + cs) * NACC * 256u;
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const bool live = c <= g.last && pb + (unsigned)h * blockDim.x < p1;
                        const float4* src = (const float4*)(cb + (size_t)jr[h] * 256u + th4[h]);
                        lv[q2][h] = live ? __ldcg(src) : make_float4(0.f, 0.f, 0.f, 0.f);
                        if constexpr (MODE == GW_GLU)
                            lw[q2][h] = live ? __ldcg(src + 2 * 64) : make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                }
#pragma unroll
                for (int q2 = 0; q2 < CU; q2++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        v[h].x += lv[q2][h].x; v[h].y += lv[q2][h].y; v[h].z += lv[q2][h].z; v[h].w += lv[q2][h].w;
                        if constexpr (MODE == GW_GLU) {
                            w[h].x += lw[q2][h].x; w[h].y += lw[q2][h].y; w[h].z += lw[q2][h].z; w[h].w += lw[q2][h].w;
                        }
                    }
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                if (pb + (unsigned)h * blockDim.x >= p1) continue;
                gw_put<MODE, SB>(a, tq, th4[h] + 0u, jr[h], v[h].x, w[h].x);
                gw_put<MODE, SB>(a, tq, th4[h] + 1u, jr[h], v[h].y, w[h].y);
                gw_put<MODE, SB>(a, tq, th4[h] + 2u, jr[h], v[h].z, w[h].z);
                gw_put<MODE, SB>(a, tq, th4[h] + 3u, jr[h], v[h].w, w[h].w);
            }
        }
        __syncthreads();
        if (tid == 0 && atomicAdd(&ctr[2u * tq + 1u], 1u) == nc - 1u) {
            ctr[2u * tq] = 0u;
            ctr[2u * tq + 1u] = 0u;
        }
    }
}

__device__ __forceinline__ GwArgs gw_args(const __nv_bfloat16* x, const __nv_bfloat16* w0,
                                          const __nv_bfloat16* w1, const __nv_bfloat16* w2,
                                          __nv_bfloat16* c0, __nv_bfloat16* c1, __nv_bfloat16* c2,
                                          unsigned n0, unsigned n1, unsigned n2, unsigned M,
                                          unsigned K, unsigned act) {
    GwArgs a;
    a.x = x;
    a.W[0] = w0; a.W[1] = w1; a.W[2] = w2;
    a.C[0] = c0; a.C[1] = c1; a.C[2] = c2;
    a.N[0] = n0; a.N[1] = n1; a.N[2] = n2;
    a.M = M; a.K = K; a.act = act;
    return a;
}

/* Below these many weight bytes the split-K fixed cost (park, wait, reduce: ~10 us) loses to the
 * mma.sync walk's extra passes. Standalone H100 A/B (experiments/gemv_wide_h100.cu): at 64 rows
 * the walk wins on Chatterbox's 2-17 MB and Qwen3's 8 MB projections; at 128 rows the wide arm
 * wins from ~4 MB (Chatterbox o_proj, 2 MB, is the one loss). */
#ifndef PLOW_NV_GW_MIN_BYTES
#define PLOW_NV_GW_MIN_BYTES (24u << 20)
#endif
#ifndef PLOW_NV_GW_MIN_BYTES_SB
#define PLOW_NV_GW_MIN_BYTES_SB (4u << 20)
#endif
/* false: shape outside the contract, the caller falls back (uniform over the block). */
template <int MODE>
static __device__ __forceinline__ bool d_gemv_wide(const GwArgs& a, unsigned slice, unsigned nblk,
                                                   float* arena) {
    const unsigned long long rows = MODE == GW_GLU ? 2ull * a.N[0] : (unsigned long long)a.N[0] + a.N[1] + a.N[2];
    if ((a.K & 63u) || a.K == 0u || a.M > 128u || nblk > GW_MAXB ||
        gw_tiles<MODE, 0>(a) > GW_MAXT ||
        rows * a.K * 2ull < (a.M > 64u ? PLOW_NV_GW_MIN_BYTES_SB : PLOW_NV_GW_MIN_BYTES))
        return false;
    if (MODE == GW_QKV && ((a.N[0] & 63u) || (a.N[1] & 63u))) return false;
    if (a.M > 64u) {
        if (gw_tiles<MODE, 1>(a) > GW_MAXT) return false;
        gemv_wide_sm90<MODE, 1>(a, slice, nblk, arena);
    } else {
        gemv_wide_sm90<MODE, 0>(a, slice, nblk, arena);
    }
    return true;
}

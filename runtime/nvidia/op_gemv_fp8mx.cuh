/* op_gemv_fp8mx.cuh -- PLOW_DOP_GEMM_FP8_MX (op 198) at decode rows (T <= 64) on the interpreter.
 *
 * DeepSeek-V4.1's block-fp8 projections at a [32, 32] ue8m0 grid, weight-streaming. The weight is
 * the MMA's A operand (16 output rows per warp tile), the <= 64 tokens are B (NT n8 tiles), and
 * every MMA covers exactly one 32-wide K block, so the per-block scales promote the f32 partial:
 *   fp8 = 1: x is e4m3 with ue8m0 scales xs [T][K/32] (ActQuantMx t2) -- m16n8k32 e4m3,
 *            acc += d * sw * sx (kernel.py fp8_gemm);
 *   fp8 = 0: x is bf16 (V4.1's wo_a einsum) -- the weight decodes exactly to bf16, two m16n8k16,
 *            acc += d * sw.
 * Weights stream through a warp-private ring of D chunks (16 rows x 4 blocks = 16 x 128 B) filled
 * by coalesced 16 B cp.async, D - 1 chunks in flight. A thread's fragment is 8 weight bytes of rows
 * g and g + 8 at k = 8 * t4 and the same 8 of each token, so the MMA's logical K is a permutation of
 * the block's 32 (order-free up to f32 rounding). Each lane's cp.async sources are precomputed (a
 * chunk is one add), and a chunk's 4 blocks are unrolled so the next block's token fragments and
 * scales load under this block's MMAs. Split-K over the S warps of one CTA, summed in the arena in
 * split order.
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace plow_gv8 {
constexpr unsigned MAX_T = 64, CB = 4, PITCH = CB * 32u + 16u, STAGE = 16u * PITCH, MAX_D = 4;
/* arena floats a warp ring of `d` stages needs, for all 8 warps */
__host__ __device__ constexpr unsigned arena_floats(unsigned d) { return 8u * d * STAGE / 4u; }
__device__ __forceinline__ float e8m0(uint8_t b) { return __uint_as_float((uint32_t)b << 23); }
__device__ __forceinline__ void mma_e4m3(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f));
}
__device__ __forceinline__ void mma_bf16_z(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f));
}
/* two e4m3 bytes -> bf16x2 (exact: e4m3 fits bf16) */
__device__ __forceinline__ uint32_t e4m3x2_bf16x2(uint16_t v) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(v));
    float lo, hi;
    asm("cvt.f32.f16 %0, %1;" : "=f"(lo) : "h"((uint16_t)(h2 & 0xffffu)));
    asm("cvt.f32.f16 %0, %1;" : "=f"(hi) : "h"((uint16_t)(h2 >> 16)));
    return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ uint2 lds8(uint32_t a) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a));
    return v;
}
/* split-K ways: the tile count times S should fill the machine's warps, S | 8, K/32 % S == 0 */
__device__ __forceinline__ unsigned splits(unsigned tiles, unsigned kb, unsigned nblk) {
    unsigned s = 1;
    while (s < 8u && tiles * s * 2u <= nblk * 8u && kb % (s * 2u) == 0) s *= 2u;
    return s;
}
/* Cross-CTA split-K (a scratch is bound): one (tile, split) item per warp of the whole grid, any S <= K/32
 * with uneven K ranges. A warp's K walk is a serial chain (~0.25 us per 32-wide block at T <= 8), so the
 * shortest walk is the lowest latency. */
__host__ __device__ constexpr unsigned sk_splits(unsigned tiles, unsigned kb, unsigned nblk) {
    return tiles >= nblk * 8u ? 1u : (nblk * 8u / tiles < kb ? nblk * 8u / tiles : kb);
}
/* scratch layout: SK_TILES per-tile arrival counters (zero at rest), then one NT x 128-float partial per item. The
 * counter block is fixed-size so ops of different shapes can share one scratch without partials landing on counters. */
constexpr unsigned SK_TILES = 4096u;
__host__ __device__ constexpr size_t sk_scratch_bytes(unsigned T, unsigned nblk) {
    return SK_TILES * 4u + (size_t)nblk * 8u * ((T + 7u) / 8u) * 128u * 4u;
}

struct Gv8Args {
    const uint8_t *x, *xs, *W, *ws;
    unsigned T, N, K, groups, kb, ld_x, depth;
};

/* One block's token operand: NT fragments (+ their two scale bytes each for fp8). */
template <unsigned NT, bool FP8>
struct Tok {
    uint32_t v[NT][FP8 ? 2 : 4];
    float s[NT][2];
};
/* A thread's token rows: tile j reads token j*8+g (fragment) and tokens j*8+2*t4, +1 (scales), each
 * clamped to a real row. A padded token's fragment and scale only reach output columns >= T, which
 * are never stored, so the loads need no masks. */
template <unsigned NT, bool FP8>
struct TokRows {
    const uint8_t* x;  /* group + t4 base; + frag[j] + block * 32 * esz */
    const uint8_t* xs; /* group base; + s0/s1[j] + block */
    uint32_t frag[NT], s0[NT], s1[NT];
    __device__ __forceinline__ TokRows(const Gv8Args& a, unsigned grp) {
        const unsigned lane = threadIdx.x & 31u, g = lane >> 2, t4 = lane & 3u, esz = FP8 ? 1u : 2u, xsb = a.groups * a.kb;
        x = a.x + ((size_t)grp * a.K + t4 * 8u) * esz;
        xs = a.xs + (size_t)grp * a.kb;
#pragma unroll
        for (unsigned j = 0; j < NT; j++) {
            frag[j] = min(j * 8u + g, a.T - 1u) * a.ld_x * esz;
            s0[j] = min(j * 8u + t4 * 2u, a.T - 1u) * xsb;
            s1[j] = min(j * 8u + t4 * 2u + 1u, a.T - 1u) * xsb;
        }
    }
    __device__ __forceinline__ void load(Tok<NT, FP8>& o, unsigned b) const {
#pragma unroll
        for (unsigned j = 0; j < NT; j++) {
            if (FP8) {
                const uint2 v = __ldg(reinterpret_cast<const uint2*>(x + frag[j] + b * 32u));
                o.v[j][0] = v.x;
                o.v[j][1] = v.y;
                o.s[j][0] = e8m0(__ldg(xs + s0[j] + b));
                o.s[j][1] = e8m0(__ldg(xs + s1[j] + b));
            } else {
                const uint4 v = __ldg(reinterpret_cast<const uint4*>(x + frag[j] + b * 64u));
                o.v[j][0] = v.x;
                o.v[j][1] = v.y;
                o.v[j][2] = v.z;
                o.v[j][3] = v.w;
            }
        }
    }
};
/* cp.async.wait_group D - 1 (the immediate must be constant) */
__device__ __forceinline__ void wait_ring(unsigned d) {
    if (d == 4u) asm volatile("cp.async.wait_group 3;\n" ::);
    else if (d == 3u) asm volatile("cp.async.wait_group 2;\n" ::);
    else asm volatile("cp.async.wait_group 1;\n" ::);
}
template <unsigned NT, bool FP8>
__device__ __forceinline__ void tile(const Gv8Args& a, unsigned grp, unsigned n0, unsigned b_lo, unsigned b_hi, uint32_t ring,
                                     float (&acc)[NT][4]) {
    const unsigned lane = threadIdx.x & 31u, g = lane >> 2, t4 = lane & 3u, D = a.depth, kbw = b_hi - b_lo, chunks = (kbw + CB - 1u) / CB;
    /* a lane's pieces of a chunk: rows (lane >> 3) + 4 i, 16 B at column (lane & 7) * 16; a chunk is + CB * 32 bytes, a
     * piece past b_hi reads the chunk-0 bytes with src-size 0 */
    const unsigned wcol = (lane & 7u) * 16u, wblk = wcol / 32u;
    const uint8_t* wsrc = a.W + (size_t)(grp * a.N + n0 + (lane >> 3)) * a.K + b_lo * 32u + wcol;
    const size_t wstep = (size_t)4u * a.K;
    const uint32_t wdst = ring + (lane >> 3) * PITCH + wcol;
    auto issue = [&](unsigned c, unsigned stage) {
        if (c < chunks) {
            const bool wv = c * CB + wblk < kbw;
            const size_t wo = wv ? c * CB * 32u : 0u;
#pragma unroll
            for (unsigned i = 0; i < 4u; i++) cp16(wdst + stage * STAGE + i * 4u * PITCH, wsrc + i * wstep + wo, wv);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    for (unsigned c = 0; c + 1u < D; c++) issue(c, c);
    /* the next block's token fragments and weight scale load under this block's MMAs (the unrolled chunk renames them) */
    const uint8_t* wsr = a.ws + (size_t)((grp * a.N + n0) / 32u) * a.kb;
    const TokRows<NT, FP8> rows(a, grp);
    Tok<NT, FP8> nx;
    rows.load(nx, b_lo);
    float swn = e8m0(__ldg(wsr + b_lo));
    const uint32_t wfr = ring + g * PITCH + t4 * 8u;
    unsigned sc = 0, si = D - 1u;
    for (unsigned c = 0; c < chunks; c++) {
        __syncwarp(); /* every lane is done reading stage si (chunk c - 1) before it is refilled */
        issue(c + D - 1u, si);
        wait_ring(D);
        __syncwarp();
        si = si + 1u == D ? 0u : si + 1u;
        const uint32_t so = sc * STAGE;
        sc = sc + 1u == D ? 0u : sc + 1u;
#pragma unroll
        for (unsigned bl = 0; bl < CB; bl++) {
            const unsigned b = b_lo + c * CB + bl;
            if (b < b_hi) {
                const Tok<NT, FP8> cur = nx;
                const float sw = swn;
                const unsigned bn = min(b + 1u, b_hi - 1u);
                rows.load(nx, bn);
                swn = e8m0(__ldg(wsr + bn));
                const uint2 wa = lds8(wfr + so + bl * 32u), wb = lds8(wfr + so + 8u * PITCH + bl * 32u);
                if (FP8) {
                    const uint32_t am[4] = {wa.x, wb.x, wa.y, wb.y};
#pragma unroll
                    for (unsigned j = 0; j < NT; j++) {
                        float d[4];
                        mma_e4m3(d, am, cur.v[j]);
                        const float s0 = sw * cur.s[j][0], s1 = sw * cur.s[j][1];
                        acc[j][0] = fmaf(d[0], s0, acc[j][0]);
                        acc[j][1] = fmaf(d[1], s1, acc[j][1]);
                        acc[j][2] = fmaf(d[2], s0, acc[j][2]);
                        acc[j][3] = fmaf(d[3], s1, acc[j][3]);
                    }
                } else {
                    /* weights to bf16: mma 0 takes bytes 0..3 of each row, mma 1 bytes 4..7 */
                    const uint32_t a0[4] = {e4m3x2_bf16x2(wa.x & 0xffffu), e4m3x2_bf16x2(wb.x & 0xffffu), e4m3x2_bf16x2(wa.x >> 16),
                                            e4m3x2_bf16x2(wb.x >> 16)};
                    const uint32_t a1[4] = {e4m3x2_bf16x2(wa.y & 0xffffu), e4m3x2_bf16x2(wb.y & 0xffffu), e4m3x2_bf16x2(wa.y >> 16),
                                            e4m3x2_bf16x2(wb.y >> 16)};
#pragma unroll
                    for (unsigned j = 0; j < NT; j++) {
                        float d0[4], d1[4];
                        mma_bf16_z(d0, a0, cur.v[j]);
                        mma_bf16_z(d1, a1, cur.v[j] + 2);
#pragma unroll
                        for (unsigned i = 0; i < 4u; i++) acc[j][i] = fmaf(d0[i] + d1[i], sw, acc[j][i]);
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncwarp();
}

template <unsigned NT, bool FP8>
__device__ __noinline__ void run(__nv_bfloat16* __restrict__ C, const Gv8Args& a_, unsigned slice, unsigned nblk, float* arena,
                                 uint8_t* scratch) {
    const Gv8Args a = a_;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3u;
    const unsigned ld_c = a.groups * a.N, tiles = a.groups * (a.N / 16u);
    /* items are (tile, split) pairs, split-minor; item base + warp keeps a tile's in-CTA splits on consecutive warps */
    const bool gsk = scratch != nullptr;
    if (gsk && tiles > SK_TILES) __trap();
    /* Cross-CTA splits (scratch only) when they shorten a warp's K walk by >= 8 blocks; below that the published-partial
     * round trip costs more. In-CTA splits divide 8, so their segments always hold the whole tile and never publish. */
    const unsigned s_in = splits(tiles, a.kb, nblk), s_x = gsk ? sk_splits(tiles, a.kb, nblk) : 1u;
    const unsigned S = a.kb / s_in >= a.kb / s_x + 8u ? s_x : s_in, items = tiles * S;
    unsigned* const ctr = reinterpret_cast<unsigned*>(scratch);
    float* const part = reinterpret_cast<float*>(scratch + SK_TILES * 4u);
    const uint32_t ring = (uint32_t)__cvta_generic_to_shared(arena) + warp * a.depth * STAGE;
    for (unsigned base = slice * 8u; base < items; base += nblk * 8u) {
        const unsigned it = base + warp;
        const bool live = it < items;
        const unsigned t = it / S, sk = it - t * S;
        const unsigned grp = live ? t / (a.N / 16u) : 0u, n0 = live ? (t % (a.N / 16u)) * 16u : 0u;
        float acc[NT][4] = {};
        if (live) tile<NT, FP8>(a, grp, n0, a.kb * sk / S, a.kb * (sk + 1u) / S, ring, acc);
        bool store = live && S == 1u;
        if (S > 1u) {
            __syncthreads(); /* the rings are drained; the arena becomes the split-K exchange */
            float* red = arena + (size_t)warp * (NT * 128u);
            if (live)
#pragma unroll
                for (unsigned j = 0; j < NT; j++)
#pragma unroll
                    for (unsigned i = 0; i < 4u; i++) red[j * 128u + i * 32u + lane] = acc[j][i];
            __syncthreads();
            /* A segment is this CTA's run of consecutive warps on one tile; its first warp sums them in K order. A
             * segment holding the whole tile stores; otherwise (cross-CTA split) it publishes its partial and the
             * tile's last segment to arrive sums all of them in K order (deterministic) and re-arms the counter. */
            const unsigned seg0 = max(t * S, base);
            if (live && it == seg0) {
                const unsigned seg1 = min(min(t * S + S, base + 8u), items);
                for (unsigned w2 = warp + 1u; w2 < warp + (seg1 - seg0); w2++) {
                    const float* o = arena + (size_t)w2 * (NT * 128u);
#pragma unroll
                    for (unsigned j = 0; j < NT; j++)
#pragma unroll
                        for (unsigned i = 0; i < 4u; i++) acc[j][i] += o[j * 128u + i * 32u + lane];
                }
                if (seg0 == t * S && seg1 == t * S + S) {
                    store = true;
                } else {
                    float* const my = part + (size_t)seg0 * (NT * 128u);
#pragma unroll
                    for (unsigned j = 0; j < NT; j++)
#pragma unroll
                        for (unsigned i = 0; i < 4u; i++) __stcg(my + j * 128u + i * 32u + lane, acc[j][i]);
                    __threadfence();
                    __syncwarp();
                    const unsigned nseg = (t * S + S - 1u) / 8u - (t * S) / 8u + 1u;
                    unsigned last = 0;
                    if (lane == 0) last = atomicAdd(ctr + t, 1u) == nseg - 1u;
                    store = __shfl_sync(~0u, last, 0) != 0u;
                    if (store) {
                        __threadfence();
                        for (unsigned q = 0, i0 = t * S; i0 < t * S + S; q++, i0 = ((t * S) / 8u + q) * 8u) {
                            const float* const o = part + (size_t)i0 * (NT * 128u);
#pragma unroll
                            for (unsigned j = 0; j < NT; j++)
#pragma unroll
                                for (unsigned i = 0; i < 4u; i++) {
                                    const float v = __ldcg(o + j * 128u + i * 32u + lane);
                                    acc[j][i] = q ? acc[j][i] + v : v;
                                }
                        }
                        if (lane == 0) ctr[t] = 0u;
                    }
                }
            }
        }
        if (store) {
#pragma unroll
            for (unsigned j = 0; j < NT; j++) {
                const unsigned c0 = j * 8u + t4 * 2u;
                const size_t col = (size_t)grp * a.N + n0 + g;
                if (c0 < a.T) {
                    C[(size_t)c0 * ld_c + col] = __float2bfloat16_rn(acc[j][0]);
                    C[(size_t)c0 * ld_c + col + 8u] = __float2bfloat16_rn(acc[j][2]);
                }
                if (c0 + 1u < a.T) {
                    C[(size_t)(c0 + 1u) * ld_c + col] = __float2bfloat16_rn(acc[j][1]);
                    C[(size_t)(c0 + 1u) * ld_c + col + 8u] = __float2bfloat16_rn(acc[j][3]);
                }
            }
        }
        if (S > 1u) __syncthreads();
    }
}
}  // namespace plow_gv8

/* C[t][g*N + n] = sum_k x[t][g*K + k] * W[g*N + n][k], t < T <= 64, over `groups` diagonal blocks.
 * arena: >= plow_gv8::arena_floats(2) floats (and >= 8 * 8 * 128 when split); deeper rings use up
 * to arena_floats(MAX_D). scratch: nullptr (splits stay inside a CTA) or plow_gv8::sk_scratch_bytes(T, nblk) bytes
 * whose counters are zero, left zero. */
__device__ __noinline__ void d_gemv_fp8mx(__nv_bfloat16* __restrict__ C, const uint8_t* __restrict__ x, const uint8_t* __restrict__ xs,
                                          const uint8_t* __restrict__ W, const uint8_t* __restrict__ ws, unsigned T, unsigned N,
                                          unsigned K, unsigned groups, bool fp8, unsigned slice, unsigned nblk, float* arena,
                                          unsigned arena_floats_, uint8_t* scratch = nullptr) {
    using namespace plow_gv8;
    unsigned depth = MAX_D;
    while (depth > 2u && arena_floats(depth) > arena_floats_) depth--;
    const Gv8Args a{x, xs, W, ws, T, N, K, groups, K / 32u, groups * K, depth};
    const unsigned nt = (T + 7u) / 8u;
    if (fp8) {
        if (nt <= 1u) run<1, true>(C, a, slice, nblk, arena, scratch);
        else if (nt <= 2u) run<2, true>(C, a, slice, nblk, arena, scratch);
        else if (nt <= 4u) run<4, true>(C, a, slice, nblk, arena, scratch);
        else run<8, true>(C, a, slice, nblk, arena, scratch);
    } else {
        if (nt <= 1u) run<1, false>(C, a, slice, nblk, arena, scratch);
        else if (nt <= 2u) run<2, false>(C, a, slice, nblk, arena, scratch);
        else if (nt <= 4u) run<4, false>(C, a, slice, nblk, arena, scratch);
        else run<8, false>(C, a, slice, nblk, arena, scratch);
    }
}

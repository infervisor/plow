/* op_gemv_fp8mx.cuh -- PLOW_DOP_GEMM_FP8_MX (op 210) at decode rows (T <= 64) on the interpreter.
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
 * Tokens and scales come from shared memory: the CTA stages x (and xs) once, and each warp its tile's
 * weight-scale row, so the per-block operand loads are shared-memory hits instead of L2 round trips
 * (the interpreter's arena leaves ~30 KB of L1). mode 2 quantizes a bf16 x into that stage with
 * ActQuantMx's exact numerics (d_act_quant_mx), so decode needs no separate ActQuantMx op.
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>
#include "../common/op_act_quant_mx.h"

namespace plow_gv8 {
constexpr unsigned MAX_T = 64, CB = 4, PITCH = CB * 32u + 16u, STAGE = 16u * PITCH, MAX_D = 4, WSB = 256;
/* arena floats a warp ring of `d` stages needs, for all 8 warps */
__host__ __device__ constexpr unsigned arena_floats(unsigned d) { return 8u * d * STAGE / 4u; }
__device__ __forceinline__ float e8m0(uint8_t b) { return __uint_as_float((uint32_t)b << 23); }
__device__ __forceinline__ void mma_e4m3(float* d, const uint32_t* a, const uint32_t* b) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f));
}
__device__ __forceinline__ void mma_bf16_z(float* d, const uint32_t* a, const uint32_t* b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
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
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ uint2 lds8(uint32_t a) {
    return *reinterpret_cast<const uint2*>(__cvta_shared_to_generic(a));
}
/* split-K ways: the tile count times S should fill the machine's warps, S | 8, K/32 % S == 0 */
__device__ __forceinline__ unsigned splits(unsigned tiles, unsigned kb, unsigned nblk) {
    unsigned s = 1;
    while (s < 8u && tiles * s * 2u <= nblk * 8u && kb % (s * 2u) == 0) s *= 2u;
    return s;
}

struct Gv8Args {
    const uint8_t *x, *xs, *W, *ws;
    unsigned T, N, K, groups, kb, ld_x, depth;
    uint8_t* wsb; /* per-warp WSB-byte weight-scale rows (shared), or null */
    unsigned probe;
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
                const uint2 v = *reinterpret_cast<const uint2*>(x + frag[j] + b * 32u);
                o.v[j][0] = v.x;
                o.v[j][1] = v.y;
                o.s[j][0] = e8m0(xs[s0[j] + b]);
                o.s[j][1] = e8m0(xs[s1[j] + b]);
            } else {
                const uint4 v = *reinterpret_cast<const uint4*>(x + frag[j] + b * 64u);
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
    if (d == 4u) asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    else if (d == 3u) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    else asm volatile("cp.async.wait_group 1;\n" ::: "memory");
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
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
#if PLOW_NV_TRACE
    const bool pr = a.probe && threadIdx.x == 0;
    if (pr) plow_probe(a.probe + 2u);
#endif
    for (unsigned c = 0; c + 1u < D; c++) issue(c, c);
    const uint8_t* wsr = a.ws + (size_t)((grp * a.N + n0) / 32u) * a.kb;
    if (a.wsb && kbw <= WSB) {
        uint8_t* buf = a.wsb + (threadIdx.x >> 5) * WSB;
        __syncwarp(); /* the previous tile's readers are done */
#pragma unroll 8
        for (unsigned i = lane; i < kbw; i += 32u) buf[i] = __ldg(wsr + b_lo + i);
        __syncwarp();
        wsr = buf - b_lo;
    }
    const TokRows<NT, FP8> rows(a, grp);
#if PLOW_NV_TRACE
    if (pr) plow_probe(a.probe + 3u);
#endif
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
                Tok<NT, FP8> cur;
                rows.load(cur, b);
                const float sw = e8m0(wsr[b]);
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
#if PLOW_NV_TRACE
        if (pr && c < 24u) plow_probe(a.probe + 4u + c);
#endif
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncwarp();
}

template <unsigned NT, bool FP8>
__device__ __noinline__ void run(__nv_bfloat16* __restrict__ C, const Gv8Args& a_, unsigned slice, unsigned nblk, float* arena) {
    const Gv8Args a = a_;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3u;
    const unsigned ld_c = a.groups * a.N, tiles = a.groups * (a.N / 16u), S = splits(tiles, a.kb, nblk), per_cta = 8u / S;
    const unsigned sk = warp % S, kb_per = a.kb / S;
    const uint32_t ring = (uint32_t)__cvta_generic_to_shared(arena) + warp * a.depth * STAGE;
    for (unsigned tile0 = slice * per_cta; tile0 < tiles; tile0 += nblk * per_cta) {
        const unsigned t = tile0 + warp / S;
        const bool live = t < tiles;
        const unsigned grp = live ? t / (a.N / 16u) : 0u, n0 = live ? (t % (a.N / 16u)) * 16u : 0u;
        float acc[NT][4] = {};
        if (live) tile<NT, FP8>(a, grp, n0, sk * kb_per, (sk + 1u) * kb_per, ring, acc);
        if (S > 1u) {
            __syncthreads(); /* the rings are drained; the arena becomes the split-K exchange */
            float* red = arena + (size_t)warp * (NT * 128u);
            if (live)
#pragma unroll
                for (unsigned j = 0; j < NT; j++)
#pragma unroll
                    for (unsigned i = 0; i < 4u; i++) red[j * 128u + i * 32u + lane] = acc[j][i];
            __syncthreads();
            if (live && sk == 0u)
                for (unsigned s2 = 1; s2 < S; s2++) {
                    const float* o = arena + (size_t)(warp + s2) * (NT * 128u);
#pragma unroll
                    for (unsigned j = 0; j < NT; j++)
#pragma unroll
                        for (unsigned i = 0; i < 4u; i++) acc[j][i] += o[j * 128u + i * 32u + lane];
                }
        }
        if (live && sk == 0u) {
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
 * to arena_floats(MAX_D). */
__device__ __noinline__ void d_gemv_fp8mx(__nv_bfloat16* __restrict__ C, const uint8_t* __restrict__ x, const uint8_t* __restrict__ xs,
                                          const uint8_t* __restrict__ W, const uint8_t* __restrict__ ws, unsigned T, unsigned N,
                                          unsigned K, unsigned groups, unsigned mode, unsigned slice, unsigned nblk, float* arena,
                                          unsigned arena_floats_, const __nv_bfloat16* __restrict__ up = nullptr, unsigned act = 0,
                                          float limit = 0.0f) {
    using namespace plow_gv8;
    /* mode: 0 = bf16 x, 1 = e4m3 x + ue8m0 xs, 2 = bf16 x quantized here (e4m3 + ue8m0, as ActQuantMx),
     * 3 = as 2 on the Glu op's rows bf16(glu_pair(x, up, act, limit)): x is the gate */
    const bool fp8 = mode != 0u;
    const unsigned ldx = groups * K, xn = T * ldx * (fp8 ? 1u : 2u), xsn = fp8 ? T * (ldx / 32u) : 0u;
    const unsigned xb = (xn + 15u) & ~15u, xsb = (xsn + 15u) & ~15u, stage = (xb + xsb + 8u * WSB) / 4u;
    const bool st = arena_floats(2) + stage <= arena_floats_;
    if (mode >= 2u && !st) __trap();
    const unsigned used = st ? stage : 0u;
    unsigned depth = MAX_D;
    while (depth > 2u && arena_floats(depth) + used > arena_floats_) depth--;
    uint8_t* const sx = reinterpret_cast<uint8_t*>(arena);
#if PLOW_NV_TRACE
    __shared__ unsigned probe_base;
    if (threadIdx.x == 0) {
        probe_base = slice == 0 ? plow_probe_begin() : 0u;
        if (probe_base) {
            plow_probe(probe_base);
            g_probe[probe_base + 30u] = blockIdx.x;
        }
    }
    __syncthreads();
    const unsigned probe = probe_base;
#else
    const unsigned probe = 0;
#endif
    if (st) {
        if (mode >= 2u) {
            const uint16_t* xq = reinterpret_cast<const uint16_t*>(x);
            if (mode == 3u) {
                /* staged past the quantized copy, where the ring (not live yet) goes */
                __nv_bfloat16* gx = reinterpret_cast<__nv_bfloat16*>(arena + stage);
                const unsigned n = T * ldx;
                if ((n * 2u + 3u) / 4u + stage > arena_floats_) __trap();
                /* 16 B loads, all issued before the first store (gx may alias for the compiler) */
                constexpr unsigned R = 4u;
                const unsigned step = blockDim.x * 8u;
                for (unsigned o0 = threadIdx.x * 8u; o0 < n; o0 += R * step) {
                    uint4 vg[R], vu[R];
#pragma unroll
                    for (unsigned r = 0; r < R; r++)
                        if (o0 + r * step < n) {
                            vg[r] = *reinterpret_cast<const uint4*>(x + (size_t)(o0 + r * step) * 2u);
                            vu[r] = *reinterpret_cast<const uint4*>(up + o0 + r * step);
                        }
#pragma unroll
                    for (unsigned r = 0; r < R; r++)
                        if (o0 + r * step < n) {
                            const __nv_bfloat16* g8 = reinterpret_cast<const __nv_bfloat16*>(&vg[r]);
                            const __nv_bfloat16* u8 = reinterpret_cast<const __nv_bfloat16*>(&vu[r]);
                            uint4 vo;
                            __nv_bfloat16* o8 = reinterpret_cast<__nv_bfloat16*>(&vo);
#pragma unroll
                            for (unsigned j = 0; j < 8u; j++)
                                o8[j] = __float2bfloat16(glu_pair(__bfloat162float(g8[j]), __bfloat162float(u8[j]), act, limit));
                            *reinterpret_cast<uint4*>(gx + o0 + r * step) = vo;
                        }
                }
                __syncthreads();
                xq = reinterpret_cast<const uint16_t*>(gx);
            }
            d_act_quant_mx(reinterpret_cast<uint16_t*>(sx), xq, T, ldx, 0u, 1u, sx + xb);
        } else {
            /* 4 loads in flight per thread per round */
            const unsigned step = blockDim.x * 16u;
            for (unsigned o = threadIdx.x * 16u; o < xn; o += 4u * step) {
                uint4 v[4];
#pragma unroll
                for (unsigned u = 0; u < 4u; u++)
                    if (o + u * step < xn) v[u] = *reinterpret_cast<const uint4*>(x + o + u * step);
#pragma unroll
                for (unsigned u = 0; u < 4u; u++)
                    if (o + u * step < xn) *reinterpret_cast<uint4*>(sx + o + u * step) = v[u];
            }
            for (unsigned o = threadIdx.x; o < xsn; o += blockDim.x) sx[xb + o] = xs[o];
        }
        __syncthreads();
    }
#if PLOW_NV_TRACE
    if (threadIdx.x == 0 && probe) plow_probe(probe + 1u);
#endif
    const Gv8Args a{st ? sx : x, st ? sx + xb : xs, W, ws, T, N, K, groups, K / 32u, ldx, depth, st ? sx + xb + xsb : nullptr, probe};
    arena += used;
    const unsigned nt = (T + 7u) / 8u;
    if (fp8) {
        if (nt <= 1u) run<1, true>(C, a, slice, nblk, arena);
        else if (nt <= 2u) run<2, true>(C, a, slice, nblk, arena);
        else if (nt <= 4u) run<4, true>(C, a, slice, nblk, arena);
        else run<8, true>(C, a, slice, nblk, arena);
    } else {
        if (nt <= 1u) run<1, false>(C, a, slice, nblk, arena);
        else if (nt <= 2u) run<2, false>(C, a, slice, nblk, arena);
        else if (nt <= 4u) run<4, false>(C, a, slice, nblk, arena);
        else run<8, false>(C, a, slice, nblk, arena);
    }
#if PLOW_NV_TRACE
    if (threadIdx.x == 0 && probe) plow_probe(probe + 28u);
#endif
}

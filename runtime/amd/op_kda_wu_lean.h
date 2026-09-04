#ifndef PLOW_OP_KDA_WU_LEAN_H
#define PLOW_OP_KDA_WU_LEAN_H

#include "op_kda.h"

/* Keep every product separately rounded, as the shipping Wu body's backend does. */
#pragma clang fp contract(off)

/* Lean BT64 W/U factor build, gfx950, dense chunks, D = V = 128, q pre-scaled in place.
 *
 * One four-wave workgroup owns one (chunk, head). It stages bf16(Ainv) as [m][s] and the
 * transposed MFMA operands X^T[d][s] = bf16(beta k exp2(g)), Y^T[v][s] = bf16(beta v) in LDS
 * once from coalesced loads, then wave w forms the 16 output tiles of token rows
 * [16w, 16w+16) with the shipping products in swapped A/B roles, so a lane holds four
 * consecutive output channels of one row (8-byte stores). The shipping body re-derives every
 * operand element for each of its 64 (row block, column tile) items from exec-masked scalar
 * loads. Products, K order, and rounding points are those of
 * d_kda_chunk_wu_bt64(q_precompute = true). KEYS also emits the carry's scaled-key hi/lo pair
 * k exp2(g_last - g) (the kda_chunk_key_factor formula) from the same k/g loads. */
namespace plow_kda_wu_lean {
constexpr unsigned D = 128u, V = 128u, BT = 64u;
constexpr unsigned ST = BT + 8u; /* [row][s] bf16 tile row stride: 16-byte rows, conflict-free */
constexpr unsigned ATILE = BT * ST;
constexpr unsigned XTILE = D * ST;
constexpr unsigned LDS_BYTES = (ATILE + 2u * XTILE) * 2u;
typedef unsigned short u16x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ bf16 lo16(unsigned w) { return (bf16)(w & 0xffffu); }
__device__ __forceinline__ bf16 hi16(unsigned w) { return (bf16)(w >> 16); }
__device__ __forceinline__ unsigned pack2(bf16 lo, bf16 hi) {
    return (unsigned)lo | ((unsigned)hi << 16);
}
} // namespace plow_kda_wu_lean

/* `n_chunks` dense BT64 chunks; `lds` is LDS_BYTES of 16-byte aligned scratch.
 * Must be launched with exactly four waves per workgroup. */
template <bool KEYS>
__device__ void d_kda_chunk_wu_bt64_lean(
    bf16* __restrict__ W, bf16* __restrict__ U, bf16* __restrict__ q,
    bf16* __restrict__ key_hi, bf16* __restrict__ key_lo, const float* __restrict__ Ainv,
    const bf16* __restrict__ k, const bf16* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, unsigned n_chunks, unsigned T, unsigned H, float scale,
    unsigned slice, unsigned nblk, bf16* __restrict__ lds) {
    using namespace plow_kda_wu_lean;
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned m = lane & 15u, kg = lane >> 4;
    bf16* const asm_ = lds;
    bf16* const xsm = asm_ + ATILE;
    bf16* const ysm = xsm + XTILE;
    const size_t n_items = (size_t)n_chunks * H;
    for (size_t item = slice; item < n_items; item += nblk) {
        const unsigned chunk = (unsigned)(item / H), h = (unsigned)(item % H);
        const unsigned row0 = chunk * BT;
        const unsigned valid = min(BT, T - row0);

        /* bf16(Ainv) rows: 16 per pass, one f32x4 per thread, rows/columns >= valid are zero. */
        {
            const unsigned am = tid >> 4, as = (tid & 15u) * 4u;
#pragma unroll
            for (unsigned p = 0; p < 4u; ++p) {
                const unsigned mm = 16u * p + am;
                const size_t rowm = min(row0 + mm, T - 1u);
                const f32x4 a =
                    *reinterpret_cast<const f32x4*>(Ainv + (rowm * H + h) * 64u + as);
                u16x4 ab;
#pragma unroll
                for (unsigned j = 0; j < 4u; ++j)
                    ab[j] = (mm < valid && as + j < valid) ? f2bf(a[j]) : (bf16)0;
                *reinterpret_cast<u16x4*>(asm_ + mm * ST + as) = ab;
            }
        }
        /* Operand rows s in [16 wave, 16 wave + 16), channels d = 2 lane, 2 lane + 1. */
        {
            const unsigned s0 = 16u * wave, d0 = 2u * lane;
            unsigned kw[16], vw[16], qw[16];
            f32x2 gw[16];
#pragma unroll
            for (unsigned i = 0; i < 16u; ++i) {
                const size_t rows = min(row0 + s0 + i, T - 1u);
                const size_t base = (rows * H + h) * D + d0;
                kw[i] = *reinterpret_cast<const unsigned*>(k + base);
                vw[i] = *reinterpret_cast<const unsigned*>(v + base);
                qw[i] = *reinterpret_cast<const unsigned*>(q + base);
                gw[i] = *reinterpret_cast<const f32x2*>(g + base);
            }
            f32x2 gl = {0.0f, 0.0f};
            if constexpr (KEYS)
                gl = *reinterpret_cast<const f32x2*>(g + ((size_t)(row0 + valid - 1u) * H + h) * D +
                                                     d0);
            u32x4 xr[2][2], yr[2][2];
#pragma unroll
            for (unsigned i = 0; i < 16u; ++i) {
                const unsigned s = s0 + i;
                const bool on = s < valid;
                const size_t rows = min(row0 + s, T - 1u);
                const float b = beta[rows * H + h];
                const size_t base = (rows * H + h) * D + d0;
                bf16 x[2], y[2], qn[2], kh[2], kl[2];
#pragma unroll
                for (unsigned e = 0; e < 2u; ++e) {
                    const bf16 kb = e ? hi16(kw[i]) : lo16(kw[i]);
                    const bf16 vb = e ? hi16(vw[i]) : lo16(vw[i]);
                    const bf16 qb = e ? hi16(qw[i]) : lo16(qw[i]);
                    const float gv = gw[i][e];
                    const float eg = exp2f(gv);
                    x[e] = on ? f2bf(bf2f(kb) * b * eg) : (bf16)0;
                    y[e] = on ? f2bf(bf2f(vb) * b) : (bf16)0;
                    const bf16 qs = f2bf(bf2f(qb) * scale);
                    qn[e] = f2bf(bf2f(qs) * eg);
                    if constexpr (KEYS) {
                        const float scaled = bf2f(kb) * exp2f(gl[e] - gv);
                        kh[e] = f2bf(scaled);
                        kl[e] = f2bf(scaled - bf2f(kh[e]));
                    }
                }
                if (on) {
                    *reinterpret_cast<unsigned*>(q + base) = pack2(qn[0], qn[1]);
                    if constexpr (KEYS) {
                        *reinterpret_cast<unsigned*>(key_hi + base) = pack2(kh[0], kh[1]);
                        *reinterpret_cast<unsigned*>(key_lo + base) = pack2(kl[0], kl[1]);
                    }
                }
#pragma unroll
                for (unsigned e = 0; e < 2u; ++e) {
                    if (i & 1u) {
                        xr[e][i >> 3][(i & 7u) >> 1] |= (unsigned)x[e] << 16;
                        yr[e][i >> 3][(i & 7u) >> 1] |= (unsigned)y[e] << 16;
                    } else {
                        xr[e][i >> 3][(i & 7u) >> 1] = x[e];
                        yr[e][i >> 3][(i & 7u) >> 1] = y[e];
                    }
                }
            }
#pragma unroll
            for (unsigned e = 0; e < 2u; ++e) {
#pragma unroll
                for (unsigned hf = 0; hf < 2u; ++hf) {
                    *reinterpret_cast<u32x4*>(xsm + (d0 + e) * ST + s0 + 8u * hf) = xr[e][hf];
                    *reinterpret_cast<u32x4*>(ysm + (d0 + e) * ST + s0 + 8u * hf) = yr[e][hf];
                }
            }
        }
        __syncthreads();

        /* out^T[n][m] = sum_s X^T[n][s] Ainv[m][s]: A = operand rows, B = Ainv rows, so lane
         * (m, kg) holds out[m0 + m][n0 + 4 kg .. +4]. */
        const unsigned m0 = 16u * wave;
        const unsigned mv = valid > m0 ? min(16u, valid - m0) : 0u;
        if (mv != 0u) {
            bf16x8 aB[2];
#pragma unroll
            for (unsigned s0i = 0; s0i < 2u; ++s0i)
                aB[s0i] = *reinterpret_cast<const bf16x8*>(asm_ + (m0 + m) * ST + 32u * s0i +
                                                           8u * kg);
            const size_t row = row0 + m0 + m;
#pragma unroll
            for (unsigned ot = 0; ot < 16u; ++ot) {
                const bool make_w = ot < 8u;
                const bf16* const src = make_w ? xsm : ysm;
                const unsigned n0 = (ot & 7u) * 16u;
                f32x4 acc = (f32x4)(0.0f);
#pragma unroll
                for (unsigned s0i = 0; s0i < 2u; ++s0i) {
                    const bf16x8 xa = *reinterpret_cast<const bf16x8*>(src + (n0 + m) * ST +
                                                                       32u * s0i + 8u * kg);
                    acc = plow_mfma_bf16_16x16(xa, aB[s0i], acc);
                }
                if (m < mv) {
                    u16x4 o;
#pragma unroll
                    for (unsigned e = 0; e < 4u; ++e) o[e] = f2bf(acc[e]);
                    bf16* const dst = (make_w ? W : U) + (row * H + h) * D + n0 + 4u * kg;
                    *reinterpret_cast<u16x4*>(dst) = o;
                }
            }
        }
        __syncthreads();
    }
}

#endif

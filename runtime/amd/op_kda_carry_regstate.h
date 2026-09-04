#ifndef PLOW_OP_KDA_CARRY_REGSTATE_H
#define PLOW_OP_KDA_CARRY_REGSTATE_H

#include "op_kda.h"

/* Register-resident BT64 chunk carry, gfx950, dense chunks, D = V = 128, q pre-scaled.
 *
 * Four waves own one (head, V16) state tile. Wave w owns token rows [16w, 16w+16) of every chunk
 * for the V' and output products and state columns [32w, 32w+32) for the update, which stay in
 * f32 MFMA accumulators across the chunk loop. The update MFMA runs with the shipping operands in
 * swapped A/B roles: the same products, K order, and hi/lo split, transposed so a lane holds four
 * consecutive d for one v. That is the layout the [v][d] bf16 snapshot in LDS wants, so the
 * cross-wave state exchange is one 8-byte write and four 16-byte reads per lane per chunk.
 *
 * Every chunk factor for chunk c+1 is loaded during chunk c from clamped addresses and masked
 * after the load, so the chunk loop carries no bounds branch around a global load. Recurrence,
 * MFMA order, and rounding points are those of d_kda_chunk_carry_bt64<true>. */
namespace plow_kda_regstate {
constexpr unsigned D = 128u, V = 128u, BT = 64u;
constexpr unsigned SST = D + 8u;   /* [v][d] bf16 state snapshot row stride */
constexpr unsigned VST = BT + 8u;  /* [v][s] bf16 V' row stride */
constexpr unsigned LDS_BYTES = (16u * SST + 16u * VST) * 2u;
typedef unsigned short u16x4 __attribute__((ext_vector_type(4)));

struct Factors {
    bf16x8 w[4], q[4];
    bf16 u[4];
    f32x4 aqk[2][2];
    bf16 k[2][2][8];
    float g[2][2][8];
    float gl[2];
    f32x4 gl4[2];
};

__device__ __forceinline__ void load_rows(
    Factors& f, const bf16* __restrict__ q, const bf16* __restrict__ W,
    const bf16* __restrict__ U, unsigned T, unsigned H, unsigned h, unsigned v0,
    unsigned row0, unsigned wave, unsigned token, unsigned kg) {
    const unsigned m0 = 16u * wave;
    const size_t rowm = min(row0 + m0 + token, T - 1u);
    const bf16* wp = W + (rowm * H + h) * D + 8u * kg;
    const bf16* qp = q + (rowm * H + h) * D + 8u * kg;
#pragma unroll
    for (unsigned t = 0; t < 4u; ++t) {
        f.w[t] = *reinterpret_cast<const bf16x8*>(wp + 32u * t);
        f.q[t] = *reinterpret_cast<const bf16x8*>(qp + 32u * t);
    }
#pragma unroll
    for (unsigned e = 0; e < 4u; ++e) {
        const size_t rowe = min(row0 + m0 + 4u * kg + e, T - 1u);
        f.u[e] = U[(rowe * H + h) * V + v0 + token];
    }
}

__device__ __forceinline__ void load_aqk(
    Factors& f, const float* __restrict__ Aqk, unsigned T, unsigned H, unsigned h,
    unsigned row0, unsigned wave, unsigned token, unsigned kg) {
    const size_t rowm = min(row0 + 16u * wave + token, T - 1u);
    const float* ap = Aqk + (rowm * H + h) * BT + 8u * kg;
#pragma unroll
    for (unsigned s0 = 0; s0 < 2u; ++s0) {
        f.aqk[s0][0] = *reinterpret_cast<const f32x4*>(ap + 32u * s0);
        f.aqk[s0][1] = *reinterpret_cast<const f32x4*>(ap + 32u * s0 + 4u);
    }
}

__device__ __forceinline__ void load_keys(
    Factors& f, const bf16* __restrict__ k, const float* __restrict__ g, unsigned T,
    unsigned H, unsigned h, unsigned row0, unsigned valid, unsigned wave, unsigned token,
    unsigned kg) {
    const size_t last = row0 + valid - 1u;
#pragma unroll
    for (unsigned b = 0; b < 2u; ++b) {
        const unsigned d0 = 32u * wave + 16u * b;
#pragma unroll
        for (unsigned s0 = 0; s0 < 2u; ++s0) {
#pragma unroll
            for (unsigned jj = 0; jj < 8u; ++jj) {
                const size_t row = min(row0 + 32u * s0 + 8u * kg + jj, T - 1u);
                const size_t i = (row * H + h) * D + d0 + token;
                f.k[b][s0][jj] = k[i];
                f.g[b][s0][jj] = g[i];
            }
        }
        const float* gl = g + (last * H + h) * D + d0;
        f.gl[b] = gl[token];
        f.gl4[b] = *reinterpret_cast<const f32x4*>(gl + 4u * kg);
    }
}

__device__ __forceinline__ u16x4 pack_bf16x4(f32x4 v) {
    u16x4 r;
#pragma unroll
    for (unsigned e = 0; e < 4u; ++e) r[e] = f2bf(v[e]);
    return r;
}

} // namespace plow_kda_regstate

/* `n_chunks` dense BT64 chunks; `ssm`/`vsm` are the two LDS snapshot tiles (LDS_BYTES total).
 * Must be launched with exactly four waves per workgroup. */
__device__ void d_kda_chunk_carry_bt64_regstate(
    bf16* __restrict__ out, float* __restrict__ state, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ W, const bf16* __restrict__ U,
    const float* __restrict__ Aqk, const float* __restrict__ g, unsigned n_chunks,
    unsigned T, unsigned H, unsigned slice, unsigned nblk, bf16* __restrict__ ssm,
    bf16* __restrict__ vsm) {
    using namespace plow_kda_regstate;
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned token = lane & 15u, kg = lane >> 4;
    const unsigned m0 = 16u * wave;
    const size_t n_items = (size_t)H * (V / 16u);
    for (size_t item = slice; item < n_items; item += nblk) {
        const unsigned h = (unsigned)(item / (V / 16u));
        const unsigned v0 = (unsigned)(item % (V / 16u)) * 16u;
        float* srow = state + ((size_t)h * V + v0 + token) * D + 32u * wave + 4u * kg;
        f32x4 S[2];
#pragma unroll
        for (unsigned b = 0; b < 2u; ++b) {
            S[b] = *reinterpret_cast<const f32x4*>(srow + 16u * b);
            *reinterpret_cast<u16x4*>(ssm + token * SST + 32u * wave + 16u * b + 4u * kg) =
                pack_bf16x4(S[b]);
        }
        Factors F;
        load_rows(F, q, W, U, T, H, h, v0, 0u, wave, token, kg);
        load_aqk(F, Aqk, T, H, h, 0u, wave, token, kg);
        load_keys(F, k, g, T, H, h, 0u, min(BT, T), wave, token, kg);
        __syncthreads();

        for (unsigned c = 0; c < n_chunks; ++c) {
            const unsigned row0 = c * BT;
            const unsigned valid = min(BT, T - row0);
            const unsigned mv = valid > m0 ? min(16u, valid - m0) : 0u;
            const unsigned cn = min(c + 1u, n_chunks - 1u);
            const unsigned row0n = cn * BT;

            f32x4 pred = (f32x4)(0.0f), fs = (f32x4)(0.0f);
#pragma unroll
            for (unsigned t = 0; t < 4u; ++t) {
                const bf16x8 sB =
                    *reinterpret_cast<const bf16x8*>(ssm + token * SST + 32u * t + 8u * kg);
                pred = plow_mfma_bf16_16x16(F.w[t], sB, pred);
                fs = plow_mfma_bf16_16x16(F.q[t], sB, fs);
            }
            u16x4 vp;
#pragma unroll
            for (unsigned e = 0; e < 4u; ++e)
                vp[e] = 4u * kg + e < mv ? f2bf(bf2f(F.u[e]) - pred[e]) : (bf16)0;
            *reinterpret_cast<u16x4*>(vsm + token * VST + m0 + 4u * kg) = vp;
            load_rows(F, q, W, U, T, H, h, v0, row0n, wave, token, kg);
            __syncthreads();

            bf16x8 vB[2], af[2];
#pragma unroll
            for (unsigned s0 = 0; s0 < 2u; ++s0) {
                vB[s0] = *reinterpret_cast<const bf16x8*>(vsm + token * VST + 32u * s0 + 8u * kg);
#pragma unroll
                for (unsigned jj = 0; jj < 8u; ++jj) {
                    const unsigned s = 32u * s0 + 8u * kg + jj;
                    const bf16 a = f2bf(F.aqk[s0][jj >> 2][jj & 3u]);
                    af[s0][jj] = __builtin_bit_cast(bf16_t, token < mv && s < valid ? a : (bf16)0);
                }
            }
            f32x4 local = (f32x4)(0.0f);
#pragma unroll
            for (unsigned s0 = 0; s0 < 2u; ++s0) local = plow_mfma_bf16_16x16(af[s0], vB[s0], local);
#pragma unroll
            for (unsigned e = 0; e < 4u; ++e) {
                const unsigned ml = 4u * kg + e;
                if (ml < mv)
                    out[((size_t)(row0 + m0 + ml) * H + h) * V + v0 + token] =
                        f2bf(fs[e] + local[e]);
            }
            load_aqk(F, Aqk, T, H, h, row0n, wave, token, kg);

#pragma unroll
            for (unsigned b = 0; b < 2u; ++b) {
                f32x4 upd = (f32x4)(0.0f);
#pragma unroll
                for (unsigned s0 = 0; s0 < 2u; ++s0) {
                    bf16x8 kh, kl;
#pragma unroll
                    for (unsigned jj = 0; jj < 8u; ++jj) {
                        const unsigned s = 32u * s0 + 8u * kg + jj;
                        const float scaled =
                            bf2f(F.k[b][s0][jj]) * exp2f(F.gl[b] - F.g[b][s0][jj]);
                        const bf16 high = f2bf(scaled);
                        const bf16 low = f2bf(scaled - bf2f(high));
                        kh[jj] = __builtin_bit_cast(bf16_t, s < valid ? high : (bf16)0);
                        kl[jj] = __builtin_bit_cast(bf16_t, s < valid ? low : (bf16)0);
                    }
                    upd = plow_mfma_bf16_16x16(kh, vB[s0], upd);
                    upd = plow_mfma_bf16_16x16(kl, vB[s0], upd);
                }
#pragma unroll
                for (unsigned e = 0; e < 4u; ++e)
                    S[b][e] = S[b][e] * exp2f(F.gl4[b][e]) + upd[e];
                *reinterpret_cast<u16x4*>(ssm + token * SST + 32u * wave + 16u * b + 4u * kg) =
                    pack_bf16x4(S[b]);
            }
            load_keys(F, k, g, T, H, h, row0n, min(BT, T - row0n), wave, token, kg);
            __syncthreads();
        }
#pragma unroll
        for (unsigned b = 0; b < 2u; ++b)
            *reinterpret_cast<f32x4*>(srow + 16u * b) = S[b];
    }
}

#endif

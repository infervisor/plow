#ifndef PLOW_OP_KDA_CARRY_REGSTATE_H
#define PLOW_OP_KDA_CARRY_REGSTATE_H

#include "op_kda.h"

/* The shipping carry's backend fuses only its state update (v_pk_fma_f32); pin that with an explicit
 * fma and keep every other product/sum separately rounded. */
#pragma clang fp contract(off)

/* Register-resident BT64 chunk carry, gfx950, dense chunks, D = V = 128, q pre-scaled.
 *
 * Four waves own one (head, V16) state tile. Wave w owns token rows [16w, 16w+16) of every chunk
 * for the V' and output products and state columns [32w, 32w+32) for the update, which stay in
 * f32 MFMA accumulators across the chunk loop. The update MFMA runs with the shipping operands in
 * swapped A/B roles: the same products, K order, and hi/lo split, transposed so a lane holds four
 * consecutive d for one v. That is the layout the [v][d] bf16 snapshot in LDS wants, so the
 * cross-wave state exchange is one 8-byte write and four 16-byte reads per lane per chunk.
 *
 * The scaled-key factors K exp2(g_last - g) are built one chunk ahead in row layout (one lane per
 * token row, coalesced loads) and staged as bf16 hi/lo tiles in a per-wave LDS region; the update
 * MFMA reads them transposed with immediate-offset 16-bit LDS loads. Every other chunk factor for
 * chunk c+1 is loaded during chunk c from clamped addresses and masked after the load, so the chunk
 * loop carries no bounds branch around a global load. Recurrence, MFMA order, and rounding points
 * are those of d_kda_chunk_carry_bt64<true>. */
namespace plow_kda_regstate {
constexpr unsigned D = 128u, V = 128u, BT = 64u;
constexpr unsigned SST = D + 8u;   /* [v][d] bf16 state snapshot row stride */
constexpr unsigned VST = BT + 8u;  /* [v][s] bf16 V' row stride */
constexpr unsigned KST = 36u;      /* [s][d] bf16 key-factor tile row stride (32 columns) */
constexpr unsigned KTILE = BT * KST;
constexpr unsigned LDS_BYTES = (16u * SST + 16u * VST + 4u * 2u * KTILE) * 2u;
typedef unsigned short u16x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));

/* Branch-free RNE with the same result as f2bf for every input (qNaN for NaN). */
__device__ __forceinline__ bf16 f2bf_sel(float f) {
    unsigned u;
    __builtin_memcpy(&u, &f, 4);
    const unsigned rne = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    const unsigned qnan = (u >> 16) | 0x0040u;
    return (bf16)(((u & 0x7fffffffu) > 0x7f800000u) ? qnan : rne);
}

/* HW = gfx950 v_cvt_pk_bf16_f32 (RNE; NaN quieting may differ from f2bf only in NaN payload). */
template <bool HW>
__device__ __forceinline__ bf16 f2bf_rs(float f) {
    if constexpr (HW) return __builtin_bit_cast(bf16, (__bf16)f);
    else return f2bf_sel(f);
}

struct Factors {
    bf16x8 w[4], q[4];
    bf16 u[4];
    f32x4 aqk[2][2];
    u32x4 kn[4];       /* k[row0 + lane][32w .. 32w+32), bf16 pairs */
    f32x4 gn[8];       /* g at the same positions */
    const float* glp;  /* g_last[32w ..], same chunk as kn/gn */
    f32x4 gl4[2];      /* g_last[32w + 16b + 4kg .. +4], one chunk ahead */
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

/* State decay gates g_last[d] for this lane's four d per block. */
__device__ __forceinline__ void load_decay(
    Factors& f, const float* __restrict__ g, unsigned H, unsigned h, unsigned row0,
    unsigned valid, unsigned wave, unsigned kg) {
    const size_t last = row0 + valid - 1u;
    const float* glp = g + (last * H + h) * D + 32u * wave;
#pragma unroll
    for (unsigned b = 0; b < 2u; ++b)
        f.gl4[b] = *reinterpret_cast<const f32x4*>(glp + 16u * b + 4u * kg);
}

__device__ __forceinline__ void load_keys(
    Factors& f, const bf16* __restrict__ k, const float* __restrict__ g, unsigned T,
    unsigned H, unsigned h, unsigned row0, unsigned valid, unsigned wave, unsigned lane) {
    const size_t row = min(row0 + lane, T - 1u);
    const size_t base = (row * H + h) * D + 32u * wave;
#pragma unroll
    for (unsigned t = 0; t < 4u; ++t)
        f.kn[t] = *reinterpret_cast<const u32x4*>(k + base + 8u * t);
#pragma unroll
    for (unsigned t = 0; t < 8u; ++t)
        f.gn[t] = *reinterpret_cast<const f32x4*>(g + base + 4u * t);
    const size_t last = row0 + valid - 1u;
    f.glp = g + (last * H + h) * D + 32u * wave;
}

/* Row `lane` of the chunk's key factors into this wave's hi/lo tiles; rows >= valid are zero. */
template <bool HW>
__device__ __forceinline__ void make_key_tiles(
    const Factors& f, bf16* __restrict__ khi, bf16* __restrict__ klo, unsigned valid,
    unsigned lane) {
    bf16* hp = khi + lane * KST;
    bf16* lp = klo + lane * KST;
#pragma unroll
    for (unsigned t = 0; t < 4u; ++t) {
        u32x4 hv, lv;
#pragma unroll
        for (unsigned j = 0; j < 8u; ++j) {
            const unsigned d = 8u * t + j;
            const unsigned kw = f.kn[t][j >> 1];
            const bf16 kb = (bf16)((j & 1u) ? kw >> 16 : kw & 0xffffu);
            float scaled = bf2f(kb) * exp2f(f.glp[d] - f.gn[d >> 2][d & 3u]);
            scaled = lane < valid ? scaled : 0.0f;
            const bf16 high = f2bf_rs<HW>(scaled);
            const bf16 low = f2bf_rs<HW>(scaled - bf2f(high));
            if (j & 1u) {
                hv[j >> 1] |= (unsigned)high << 16;
                lv[j >> 1] |= (unsigned)low << 16;
            } else {
                hv[j >> 1] = high;
                lv[j >> 1] = low;
            }
        }
        *reinterpret_cast<u32x2*>(hp + 8u * t) = u32x2{hv[0], hv[1]};
        *reinterpret_cast<u32x2*>(hp + 8u * t + 4u) = u32x2{hv[2], hv[3]};
        *reinterpret_cast<u32x2*>(lp + 8u * t) = u32x2{lv[0], lv[1]};
        *reinterpret_cast<u32x2*>(lp + 8u * t + 4u) = u32x2{lv[2], lv[3]};
    }
}

template <bool HW>
__device__ __forceinline__ u16x4 pack_bf16x4(f32x4 v) {
    u16x4 r;
#pragma unroll
    for (unsigned e = 0; e < 4u; ++e) r[e] = f2bf_rs<HW>(v[e]);
    return r;
}

} // namespace plow_kda_regstate

/* `n_chunks` dense BT64 chunks; `lds` is LDS_BYTES of 16-byte aligned scratch.
 * Must be launched with exactly four waves per workgroup. TIMED accumulates wave-0 s_memtime
 * phase cycles into timers[slice * 8 + 0..6] (bench instrumentation only). */
template <bool TIMED = false, bool HW_CVT = false>
__device__ void d_kda_chunk_carry_bt64_regstate(
    bf16* __restrict__ out, float* __restrict__ state, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ W, const bf16* __restrict__ U,
    const float* __restrict__ Aqk, const float* __restrict__ g, unsigned n_chunks,
    unsigned T, unsigned H, unsigned slice, unsigned nblk, bf16* __restrict__ lds,
    unsigned long long* __restrict__ timers = nullptr) {
    using namespace plow_kda_regstate;
    unsigned long long acc[7] = {0, 0, 0, 0, 0, 0, 0};
    const unsigned long long t_begin = TIMED ? __builtin_amdgcn_s_memtime() : 0ull;
#define PLOW_RS_STAMP(v) const unsigned long long v = TIMED ? __builtin_amdgcn_s_memtime() : 0ull
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned token = lane & 15u, kg = lane >> 4;
    const unsigned m0 = 16u * wave;
    bf16* const ssm = lds;
    bf16* const vsm = ssm + 16u * SST;
    bf16* const khi = vsm + 16u * VST + wave * 2u * KTILE;
    bf16* const klo = khi + KTILE;
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
                pack_bf16x4<HW_CVT>(S[b]);
        }
        Factors F;
        load_rows(F, q, W, U, T, H, h, v0, 0u, wave, token, kg);
        load_aqk(F, Aqk, T, H, h, 0u, wave, token, kg);
        load_decay(F, g, H, h, 0u, min(BT, T), wave, kg);
        load_keys(F, k, g, T, H, h, 0u, min(BT, T), wave, lane);
        make_key_tiles<HW_CVT>(F, khi, klo, min(BT, T), lane);
        {
            const unsigned c1 = min(1u, n_chunks - 1u);
            load_keys(F, k, g, T, H, h, c1 * BT, min(BT, T - c1 * BT), wave, lane);
        }
        __syncthreads();

        for (unsigned c = 0; c < n_chunks; ++c) {
            const unsigned row0 = c * BT;
            const unsigned valid = min(BT, T - row0);
            const unsigned mv = valid > m0 ? min(16u, valid - m0) : 0u;
            const unsigned cn = min(c + 1u, n_chunks - 1u);
            const unsigned row0n = cn * BT;
            const unsigned validn = min(BT, T - row0n);
            const unsigned cnn = min(c + 2u, n_chunks - 1u);

            PLOW_RS_STAMP(t0);
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
                vp[e] = 4u * kg + e < mv ? f2bf_rs<HW_CVT>(bf2f(F.u[e]) - pred[e]) : (bf16)0;
            *reinterpret_cast<u16x4*>(vsm + token * VST + m0 + 4u * kg) = vp;
            PLOW_RS_STAMP(t1);
            load_rows(F, q, W, U, T, H, h, v0, row0n, wave, token, kg);
            __syncthreads();
            PLOW_RS_STAMP(t2);

            bf16x8 vB[2], af[2];
#pragma unroll
            for (unsigned s0 = 0; s0 < 2u; ++s0) {
                vB[s0] = *reinterpret_cast<const bf16x8*>(vsm + token * VST + 32u * s0 + 8u * kg);
#pragma unroll
                for (unsigned jj = 0; jj < 8u; ++jj) {
                    const unsigned s = 32u * s0 + 8u * kg + jj;
                    const bf16 a = f2bf_rs<HW_CVT>(F.aqk[s0][jj >> 2][jj & 3u]);
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
                        f2bf_rs<HW_CVT>(fs[e] + local[e]);
            }
            PLOW_RS_STAMP(t3);
            load_aqk(F, Aqk, T, H, h, row0n, wave, token, kg);

#pragma unroll
            for (unsigned b = 0; b < 2u; ++b) {
                f32x4 upd = (f32x4)(0.0f);
#pragma unroll
                for (unsigned s0 = 0; s0 < 2u; ++s0) {
                    const bf16* hp = khi + (32u * s0 + 8u * kg) * KST + 16u * b + token;
                    const bf16* lp = klo + (32u * s0 + 8u * kg) * KST + 16u * b + token;
                    bf16x8 kh, kl;
#pragma unroll
                    for (unsigned jj = 0; jj < 8u; ++jj) {
                        kh[jj] = __builtin_bit_cast(bf16_t, hp[jj * KST]);
                        kl[jj] = __builtin_bit_cast(bf16_t, lp[jj * KST]);
                    }
                    upd = plow_mfma_bf16_16x16(kh, vB[s0], upd);
                    upd = plow_mfma_bf16_16x16(kl, vB[s0], upd);
                }
#pragma unroll
                for (unsigned e = 0; e < 4u; ++e)
                    S[b][e] = __builtin_fmaf(S[b][e], exp2f(F.gl4[b][e]), upd[e]);
                *reinterpret_cast<u16x4*>(ssm + token * SST + 32u * wave + 16u * b + 4u * kg) =
                    pack_bf16x4<HW_CVT>(S[b]);
            }
            PLOW_RS_STAMP(t4);
            load_decay(F, g, H, h, row0n, validn, wave, kg);
            make_key_tiles<HW_CVT>(F, khi, klo, validn, lane);
            load_keys(F, k, g, T, H, h, cnn * BT, min(BT, T - cnn * BT), wave, lane);
            PLOW_RS_STAMP(t5);
            __syncthreads();
            if constexpr (TIMED) {
                const unsigned long long t6 = __builtin_amdgcn_s_memtime();
                acc[0] += t1 - t0; acc[1] += t2 - t1; acc[2] += t3 - t2;
                acc[3] += t4 - t3; acc[4] += t5 - t4; acc[5] += t6 - t5;
            }
        }
#pragma unroll
        for (unsigned b = 0; b < 2u; ++b)
            *reinterpret_cast<f32x4*>(srow + 16u * b) = S[b];
    }
    if constexpr (TIMED) {
        acc[6] = __builtin_amdgcn_s_memtime() - t_begin;
        if (tid == 0)
            for (unsigned i = 0; i < 7u; ++i) timers[slice * 8u + i] = acc[i];
    }
#undef PLOW_RS_STAMP
}

#endif

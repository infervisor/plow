#define PLOW_WG_WAVES 8
#include "op_elementwise.h"
#include "op_kda.h"

namespace {

constexpr unsigned BT = 64u;
constexpr unsigned D = 128u;
constexpr unsigned V = 128u;

template <unsigned BV, bool KEY_FACTORS, bool CACHED_DECAY>
__device__ __forceinline__ void carry_update_d16(
    float* __restrict__ st, const bf16* __restrict__ k,
    const bf16* __restrict__ vsm, const float* __restrict__ g,
    const bf16* __restrict__ kh, const bf16* __restrict__ kr,
    const float* __restrict__ decay,
    unsigned T, unsigned H, unsigned h, unsigned row0, unsigned valid,
    unsigned last, unsigned vv, unsigned v_stride, unsigned st_stride,
    unsigned d0, unsigned token, unsigned kgroup) {
    f32x4 upd = (f32x4)(0.0f);
    for (unsigned s0 = 0; s0 < BT; s0 += 32u) {
        bf16x8 vf = (bf16x8)((bf16_t)0), kf = (bf16x8)((bf16_t)0);
        bf16x8 kres = (bf16x8)((bf16_t)0);
        const unsigned s = s0 + 8u * kgroup;
#pragma unroll
        for (unsigned j = 0; j < 8u; ++j) {
            if (token < vv && s + j < valid)
                vf[j] = __builtin_bit_cast(
                    bf16_t, vsm[(s + j) * v_stride + token]);
            if (s + j < valid) {
                if constexpr (KEY_FACTORS) {
                    kf[j] = __builtin_bit_cast(
                        bf16_t, kh[(s + j) * st_stride + d0 + token]);
                    kres[j] = __builtin_bit_cast(
                        bf16_t, kr[(s + j) * st_stride + d0 + token]);
                } else {
                    const size_t i = ((size_t)(row0 + s + j) * H + h) * D +
                                     d0 + token;
                    const size_t il = (last * H + h) * D + d0 + token;
                    const float scaled = bf2f(k[i]) * exp2f(g[il] - g[i]);
                    const bf16 high = f2bf(scaled);
                    kf[j] = __builtin_bit_cast(bf16_t, high);
                    kres[j] = __builtin_bit_cast(
                        bf16_t, f2bf(scaled - bf2f(high)));
                }
            }
        }
        upd = plow_mfma_bf16_16x16(vf, kf, upd);
        upd = plow_mfma_bf16_16x16(vf, kres, upd);
    }
#pragma unroll
    for (unsigned e = 0; e < 4u; ++e) {
        const unsigned vl = 4u * kgroup + e;
        if (vl < vv) {
            const size_t il = (last * H + h) * D + d0 + token;
            const float factor = CACHED_DECAY ? decay[d0 + token] : exp2f(g[il]);
            st[vl * st_stride + d0 + token] =
                st[vl * st_stride + d0 + token] * factor + upd[e];
        }
    }
}

template <unsigned BV, unsigned WAVES, bool STAGED, bool OVERLAP_TAIL = false,
          bool STAGE_KEYS = false>
__device__ void carry_vtile(
    bf16* __restrict__ out, float* __restrict__ state,
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ W, const bf16* __restrict__ U,
    const float* __restrict__ Aqk, const float* __restrict__ g,
    unsigned T, unsigned H, unsigned slice, unsigned nblk,
    unsigned char* __restrict__ shared) {
    constexpr unsigned ST_STRIDE = STAGED ? D + 4u : D;
    constexpr unsigned V_STRIDE = STAGED ? BV + 4u : BV;
    constexpr unsigned A_STRIDE = STAGED ? BT + 4u : BT;

    float* st = reinterpret_cast<float*>(shared);
    bf16* vsm = reinterpret_cast<bf16*>(st + BV * ST_STRIDE);
    float* osm = reinterpret_cast<float*>(vsm + BT * V_STRIDE);
    bf16* wp = nullptr;
    bf16* qp = nullptr;
    bf16* up = nullptr;
    float* ap = nullptr;
    bf16* kh = nullptr;
    bf16* kr = nullptr;
    float* decay = nullptr;
    if constexpr (STAGED) {
        wp = reinterpret_cast<bf16*>(osm + BT * V_STRIDE);
        qp = wp + BT * ST_STRIDE;
        up = qp + BT * ST_STRIDE;
        ap = reinterpret_cast<float*>(up + BT * V_STRIDE);
        kh = reinterpret_cast<bf16*>(ap + BT * A_STRIDE);
        kr = kh + BT * ST_STRIDE;
    } else if constexpr (STAGE_KEYS) {
        kh = reinterpret_cast<bf16*>(osm + BT * V_STRIDE);
        kr = kh + BT * D;
        decay = reinterpret_cast<float*>(kr + BT * D);
    }

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned wave = tid >> 6;
    const unsigned token = lane & 15u;
    const unsigned kgroup = lane >> 4;
    const unsigned vtiles = (V + BV - 1u) / BV;
    const size_t n_items = (size_t)H * vtiles;

    for (size_t item = slice; item < n_items; item += nblk) {
        const unsigned h = (unsigned)(item / vtiles);
        const unsigned v0 = (unsigned)(item % vtiles) * BV;
        const unsigned vv = min(BV, V - v0);
        for (unsigned x = tid; x < vv * D; x += WAVES * 64u)
            st[(x / D) * ST_STRIDE + x % D] =
                state[((size_t)h * V + v0 + x / D) * D + x % D];
        __syncthreads();

        const unsigned n_chunks = (T + BT - 1u) / BT;
        for (unsigned c = 0; c < n_chunks; ++c) {
            const unsigned row0 = c * BT;
            const unsigned valid = min(BT, T - row0);
            const size_t last = row0 + valid - 1u;

            if constexpr (STAGE_KEYS) {
                for (unsigned x = tid; x < valid * D; x += WAVES * 64u) {
                    const unsigned m = x / D, d = x % D;
                    const size_t i = ((size_t)(row0 + m) * H + h) * D + d;
                    const size_t il = (last * H + h) * D + d;
                    const float scaled = bf2f(k[i]) * exp2f(g[il] - g[i]);
                    const bf16 high = f2bf(scaled);
                    kh[m * D + d] = high;
                    kr[m * D + d] = f2bf(scaled - bf2f(high));
                }
                if (tid < D) decay[tid] = exp2f(g[(last * H + h) * D + tid]);
                __syncthreads();
            }

            if constexpr (STAGED) {
                for (unsigned x = tid; x < BT * D; x += WAVES * 64u) {
                    const unsigned m = x / D, d = x % D;
                    const size_t i = ((size_t)(row0 + m) * H + h) * D + d;
                    wp[m * ST_STRIDE + d] = m < valid ? W[i] : (bf16)0;
                    qp[m * ST_STRIDE + d] = m < valid ? q[i] : (bf16)0;
                    if (m < valid) {
                        const size_t il = (last * H + h) * D + d;
                        const float scaled = bf2f(k[i]) * exp2f(g[il] - g[i]);
                        const bf16 high = f2bf(scaled);
                        kh[m * ST_STRIDE + d] = high;
                        kr[m * ST_STRIDE + d] = f2bf(scaled - bf2f(high));
                    } else {
                        kh[m * ST_STRIDE + d] = (bf16)0;
                        kr[m * ST_STRIDE + d] = (bf16)0;
                    }
                }
                for (unsigned x = tid; x < BT * BV; x += WAVES * 64u) {
                    const unsigned m = x / BV, vlocal = x % BV;
                    const size_t i = ((size_t)(row0 + m) * H + h) * V + v0 + vlocal;
                    up[m * V_STRIDE + vlocal] =
                        m < valid && vlocal < vv ? U[i] : (bf16)0;
                }
                for (unsigned x = tid; x < BT * BT; x += WAVES * 64u) {
                    const unsigned m = x / BT, s = x % BT;
                    const size_t i = ((size_t)(row0 + m) * H + h) * BT + s;
                    ap[m * A_STRIDE + s] = m < valid && s < valid ? Aqk[i] : 0.0f;
                }
                __syncthreads();
            }

            constexpr unsigned V_MFMA = (BV + 15u) / 16u;
            for (unsigned task = wave; task < 4u * V_MFMA; task += WAVES) {
                const unsigned mb = task & 3u;
                const unsigned vh = task >> 2;
                const unsigned m0 = mb * 16u;
                const unsigned mv = valid > m0 ? min(16u, valid - m0) : 0u;
                if (mv == 0u) continue;
                f32x4 pred = (f32x4)(0.0f), from_state = (f32x4)(0.0f);
                for (unsigned d0 = 0; d0 < D; d0 += 32u) {
                    bf16x8 wf = (bf16x8)((bf16_t)0), qf = (bf16x8)((bf16_t)0);
                    bf16x8 sf = (bf16x8)((bf16_t)0);
                    const unsigned d = d0 + 8u * kgroup;
                    if (token < mv) {
                        const unsigned m = m0 + token;
#pragma unroll
                        for (unsigned j = 0; j < 8u; ++j) {
                            const size_t i = ((size_t)(row0 + m) * H + h) * D + d + j;
                            wf[j] = __builtin_bit_cast(
                                bf16_t, STAGED ? wp[m * ST_STRIDE + d + j] : W[i]);
                            qf[j] = __builtin_bit_cast(
                                bf16_t, STAGED ? qp[m * ST_STRIDE + d + j] : q[i]);
                        }
                    }
                    const unsigned vl = vh * 16u + token;
                    if (vl < vv) {
#pragma unroll
                        for (unsigned j = 0; j < 8u; ++j)
                            sf[j] = __builtin_bit_cast(
                                bf16_t, f2bf(st[vl * ST_STRIDE + d + j]));
                    }
                    pred = plow_mfma_bf16_16x16(wf, sf, pred);
                    from_state = plow_mfma_bf16_16x16(qf, sf, from_state);
                }
#pragma unroll
                for (unsigned e = 0; e < 4u; ++e) {
                    const unsigned ml = 4u * kgroup + e;
                    const unsigned vl = vh * 16u + token;
                    if (ml < mv && vl < vv) {
                        const unsigned m = m0 + ml;
                        const size_t ui = ((size_t)(row0 + m) * H + h) * V + v0 + vl;
                        const bf16 uv = STAGED ? up[m * V_STRIDE + vl] : U[ui];
                        vsm[m * V_STRIDE + vl] = f2bf(bf2f(uv) - pred[e]);
                        osm[m * V_STRIDE + vl] = from_state[e];
                    }
                }
            }
            __syncthreads();

            for (unsigned task = wave; task < 4u * V_MFMA; task += WAVES) {
                const unsigned mb = task & 3u;
                const unsigned vh = task >> 2;
                const unsigned m0 = mb * 16u;
                const unsigned mv = valid > m0 ? min(16u, valid - m0) : 0u;
                if (mv == 0u) continue;
                f32x4 local = (f32x4)(0.0f);
                for (unsigned s0 = 0; s0 < BT; s0 += 32u) {
                    bf16x8 af = (bf16x8)((bf16_t)0), vf = (bf16x8)((bf16_t)0);
                    const unsigned s = s0 + 8u * kgroup;
#pragma unroll
                    for (unsigned j = 0; j < 8u; ++j) {
                        if (token < mv && s + j < valid) {
                            const unsigned m = m0 + token;
                            const size_t ai = ((size_t)(row0 + m) * H + h) * BT + s + j;
                            af[j] = __builtin_bit_cast(
                                bf16_t, f2bf(STAGED ? ap[m * A_STRIDE + s + j] : Aqk[ai]));
                        }
                        const unsigned vl = vh * 16u + token;
                        if (vl < vv && s + j < valid)
                            vf[j] = __builtin_bit_cast(
                                bf16_t, vsm[(s + j) * V_STRIDE + vl]);
                    }
                    local = plow_mfma_bf16_16x16(af, vf, local);
                }
#pragma unroll
                for (unsigned e = 0; e < 4u; ++e) {
                    const unsigned ml = 4u * kgroup + e;
                    const unsigned vl = vh * 16u + token;
                    if (ml < mv && vl < vv) {
                        const unsigned m = m0 + ml;
                        out[((size_t)(row0 + m) * H + h) * V + v0 + vl] =
                            f2bf(osm[m * V_STRIDE + vl] + local[e]);
                    }
                }
            }
            if constexpr (OVERLAP_TAIL) {
                const unsigned d0 = ((wave + 4u) & 7u) * 16u;
                carry_update_d16<BV, STAGED || STAGE_KEYS, STAGE_KEYS>(
                    st, k, vsm, g, kh, kr, decay, T, H, h, row0, valid, (unsigned)last,
                    vv, V_STRIDE, ST_STRIDE, d0, token, kgroup);
            } else {
                __syncthreads();
                for (unsigned task = wave; task < 8u * V_MFMA; task += WAVES) {
                    const unsigned vh = task >> 3;
                    const unsigned d0 = (task & 7u) * 16u;
                    carry_update_d16<BV, STAGED || STAGE_KEYS, STAGE_KEYS>(
                        st + vh * 16u * ST_STRIDE, k, vsm + vh * 16u, g, kh, kr, decay,
                        T, H, h, row0, valid, (unsigned)last, min(16u, vv - vh * 16u),
                        V_STRIDE, ST_STRIDE, d0, token, kgroup);
                }
            }
            __syncthreads();
        }

        for (unsigned x = tid; x < vv * D; x += WAVES * 64u)
            state[((size_t)h * V + v0 + x / D) * D + x % D] =
                st[(x / D) * ST_STRIDE + x % D];
        __syncthreads();
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(512, 1)
void k_carry_control(bf16* out, float* state, const bf16* q, const bf16* k,
                     const bf16* W, const bf16* U, const float* Aqk, const float* g,
                     unsigned T, unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    float* st = reinterpret_cast<float*>(shared);
    bf16* vsm = reinterpret_cast<bf16*>(st + 16u * D);
    float* osm = reinterpret_cast<float*>(vsm + BT * 16u);
    d_kda_chunk_carry_bt64<true>(out, state, q, k, W, U, Aqk, g, nullptr,
        (T + BT - 1u) / BT, T, H, D, V, 1.0f, blockIdx.x, gridDim.x, st, vsm, osm);
}

extern "C" __global__ __launch_bounds__(512, 1)
void k_carry_v32_wg512(bf16* out, float* state, const bf16* q, const bf16* k,
                       const bf16* W, const bf16* U, const float* Aqk, const float* g,
                       unsigned T, unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    carry_vtile<32, 8, false>(out, state, q, k, W, U, Aqk, g, T, H,
                        blockIdx.x, gridDim.x, shared);
}

extern "C" __global__ __launch_bounds__(256, 1)
void k_carry_v8_wg256(bf16* out, float* state, const bf16* q, const bf16* k,
                      const bf16* W, const bf16* U, const float* Aqk, const float* g,
                      unsigned T, unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    carry_vtile<8, 4, false>(out, state, q, k, W, U, Aqk, g, T, H,
                           blockIdx.x, gridDim.x, shared);
}

extern "C" __global__ __launch_bounds__(256, 1)
void k_carry_v16_wg256(bf16* out, float* state, const bf16* q, const bf16* k,
                       const bf16* W, const bf16* U, const float* Aqk, const float* g,
                       unsigned T, unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    carry_vtile<16, 4, false>(out, state, q, k, W, U, Aqk, g, T, H,
                            blockIdx.x, gridDim.x, shared);
}

extern "C" __global__ __launch_bounds__(256, 1)
void k_carry_v32_wg256(bf16* out, float* state, const bf16* q, const bf16* k,
                       const bf16* W, const bf16* U, const float* Aqk, const float* g,
                       unsigned T, unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    carry_vtile<32, 4, false>(out, state, q, k, W, U, Aqk, g, T, H,
                        blockIdx.x, gridDim.x, shared);
}

extern "C" __global__ __launch_bounds__(512, 1)
void k_carry_v32_staged(bf16* out, float* state, const bf16* q, const bf16* k,
                        const bf16* W, const bf16* U, const float* Aqk, const float* g,
                        unsigned T, unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    carry_vtile<32, 8, true>(out, state, q, k, W, U, Aqk, g, T, H,
                       blockIdx.x, gridDim.x, shared);
}

extern "C" __global__ __launch_bounds__(512, 1)
void k_carry_tail_overlap(bf16* out, float* state, const bf16* q, const bf16* k,
                          const bf16* W, const bf16* U, const float* Aqk,
                          const float* g, unsigned T, unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    carry_vtile<16, 8, false, true>(out, state, q, k, W, U, Aqk, g, T, H,
                                   blockIdx.x, gridDim.x, shared);
}

extern "C" __global__ __launch_bounds__(512, 1)
void k_carry_key_stage(bf16* out, float* state, const bf16* q, const bf16* k,
                         const bf16* W, const bf16* U, const float* Aqk,
                         const float* g, unsigned T, unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    carry_vtile<16, 8, false, false, true>(out, state, q, k, W, U, Aqk, g, T, H,
                                          blockIdx.x, gridDim.x, shared);
}

extern "C" __global__ __launch_bounds__(512, 1)
void k_carry_key_stage_tail_overlap(bf16* out, float* state, const bf16* q,
                              const bf16* k, const bf16* W, const bf16* U,
                              const float* Aqk, const float* g, unsigned T,
                              unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    carry_vtile<16, 8, false, true, true>(out, state, q, k, W, U, Aqk, g, T, H,
                                         blockIdx.x, gridDim.x, shared);
}

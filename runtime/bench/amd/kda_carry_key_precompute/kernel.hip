#define PLOW_WG_WAVES 4
#include "op_elementwise.h"
#include "op_kda.h"

extern "C" __device__ unsigned plow_kda_carry_key_precompute_abi_1 = 1;

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_key_factors(bf16* __restrict__ key_hi, bf16* __restrict__ key_lo,
                   const bf16* __restrict__ k, const float* __restrict__ g,
                   unsigned T, unsigned H, unsigned D) {
    const size_t count = (size_t)T * H * D;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        const unsigned row = (unsigned)(i / ((size_t)H * D));
        const unsigned hd = (unsigned)(i - (size_t)row * H * D);
        const unsigned last = min((row | 63u), T - 1u);
        const size_t il = (size_t)last * H * D + hd;
        const float scaled = bf2f(k[i]) * exp2f(g[il] - g[i]);
        const bf16 high = f2bf(scaled);
        key_hi[i] = high;
        key_lo[i] = f2bf(scaled - bf2f(high));
    }
}

template <bool PRECOMPUTED_KEY>
__device__ void wu(bf16* __restrict__ W, bf16* __restrict__ U,
                   const float* __restrict__ Ainv, const bf16* __restrict__ k,
                   const bf16* __restrict__ write_key, const bf16* __restrict__ v,
                   const float* __restrict__ g, const float* __restrict__ beta,
                   unsigned T, unsigned H) {
    constexpr unsigned D = 128, V = 128, dtiles = 8, vtiles = 8, otiles = 16;
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned col = lane & 15u, kgroup = lane >> 4;
    const unsigned n_chunks = (T + 63u) / 64u;
    const size_t n_items = (size_t)n_chunks * H * 4u * otiles;
    for (size_t item = (size_t)blockIdx.x * PLOW_WAVES + wave; item < n_items;
         item += (size_t)gridDim.x * PLOW_WAVES) {
        size_t z = item;
        const unsigned ot = z % otiles; z /= otiles;
        const unsigned mb = z & 3u; z >>= 2;
        const unsigned h = z % H, chunk = z / H;
        const uint2 desc = kda_chunk_desc(nullptr, chunk, T);
        const unsigned m0 = mb * 16u;
        const unsigned mv = desc.y > m0 ? min(16u, desc.y - m0) : 0u;
        if (mv == 0) continue;
        const bool make_w = ot < dtiles;
        const unsigned out0 = (make_w ? ot : ot - dtiles) * 16u;
        const size_t row0 = desc.x;
        f32x4 acc = (f32x4)(0.0f);
        for (unsigned s0 = 0; s0 < 64u; s0 += 32u) {
            bf16x8 af = (bf16x8)((bf16_t)0), bf = (bf16x8)((bf16_t)0);
            const unsigned s = s0 + 8u * kgroup;
#pragma unroll
            for (unsigned j = 0; j < 8; j++) {
                const unsigned sl = s + j;
                if (col < mv && sl < desc.y) {
                    const size_t ai = ((row0 + m0 + col) * H + h) * 64u + sl;
                    af[j] = __builtin_bit_cast(bf16_t, f2bf(Ainv[ai]));
                }
                if (out0 + col < 128u && sl < desc.y) {
                    const size_t row = row0 + sl;
                    if (make_w) {
                        const size_t i = (row * H + h) * D + out0 + col;
                        if constexpr (PRECOMPUTED_KEY)
                            bf[j] = __builtin_bit_cast(bf16_t, write_key[i]);
                        else {
                            const float x = bf2f(k[i]) * beta[row * H + h] * exp2f(g[i]);
                            bf[j] = __builtin_bit_cast(bf16_t, f2bf(x));
                        }
                    } else {
                        const size_t i = (row * H + h) * V + out0 + col;
                        bf[j] = __builtin_bit_cast(
                            bf16_t, f2bf(bf2f(v[i]) * beta[row * H + h]));
                    }
                }
            }
            acc = plow_mfma_bf16_16x16(af, bf, acc);
        }
#pragma unroll
        for (unsigned e = 0; e < 4; e++) {
            const unsigned ml = 4u * kgroup + e;
            if (ml < mv && out0 + col < 128u) {
                const size_t row = row0 + m0 + ml;
                const size_t oi = (row * H + h) * 128u + out0 + col;
                (make_w ? W : U)[oi] = f2bf(acc[e]);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_wu_control(bf16* W, bf16* U, const float* Ainv, const bf16* k,
                  const bf16* v, const float* g, const float* beta,
                  unsigned T, unsigned H) {
    wu<false>(W, U, Ainv, k, nullptr, v, g, beta, T, H);
}

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_wu_key_factors(bf16* W, bf16* U, bf16* key_hi, bf16* key_lo,
                      const float* Ainv, const bf16* k, const bf16* v,
                      const float* g, const float* beta, unsigned T, unsigned H) {
    constexpr unsigned D = 128;
    const size_t count = (size_t)T * H * D;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        const unsigned row = (unsigned)(i / ((size_t)H * D));
        const unsigned last = min((row | 63u), T - 1u);
        const size_t il = (size_t)last * H * D + i % ((size_t)H * D);
        const float scaled = bf2f(k[i]) * exp2f(g[il] - g[i]);
        const bf16 high = f2bf(scaled);
        key_hi[i] = high;
        key_lo[i] = f2bf(scaled - bf2f(high));
    }
    wu<false>(W, U, Ainv, k, nullptr, v, g, beta, T, H);
}

template <bool PRECOMPUTED_KEY>
__device__ void carry(
    bf16* __restrict__ out, float* __restrict__ state, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ key_hi,
    const bf16* __restrict__ key_lo, const bf16* __restrict__ W,
    const bf16* __restrict__ U, const float* __restrict__ Aqk,
    const float* __restrict__ g, unsigned T, unsigned H,
    float* __restrict__ st, bf16* __restrict__ vsm, float* __restrict__ osm) {
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned token = lane & 15u, kgroup = lane >> 4;
    constexpr unsigned D = 128, V = 128, vtiles = 8;
    const size_t n_items = (size_t)H * vtiles;
    for (size_t item = blockIdx.x; item < n_items; item += gridDim.x) {
        const unsigned h = (unsigned)(item / vtiles), vt = (unsigned)(item % vtiles);
        const unsigned v0 = vt * 16u, vv = 16u;
        for (unsigned x = tid; x < vv * D; x += blockDim.x)
            st[x] = state[((size_t)h * V + v0 + x / D) * D + x % D];
        __syncthreads();
        const unsigned n_chunks = (T + 63u) / 64u;
        for (unsigned c = 0; c < n_chunks; c++) {
            const uint2 desc = kda_chunk_desc(nullptr, c, T);
            const size_t row0 = desc.x;
            for (unsigned mb = wave; mb < 4u; mb += PLOW_WAVES) {
                const unsigned m0 = mb * 16u;
                const unsigned mv = desc.y > m0 ? min(16u, desc.y - m0) : 0u;
                if (mv == 0) continue;
                f32x4 pred = (f32x4)(0.0f), from_state = (f32x4)(0.0f);
                for (unsigned d0 = 0; d0 < D; d0 += 32u) {
                    bf16x8 wf = (bf16x8)((bf16_t)0), qf = (bf16x8)((bf16_t)0);
                    bf16x8 sf = (bf16x8)((bf16_t)0);
                    const unsigned d = d0 + 8u * kgroup;
                    if (token < mv) {
                        const size_t row = row0 + m0 + token;
#pragma unroll
                        for (unsigned j = 0; j < 8; j++) {
                            const size_t i = (row * H + h) * D + d + j;
                            wf[j] = __builtin_bit_cast(bf16_t, W[i]);
                            qf[j] = __builtin_bit_cast(bf16_t, q[i]);
                        }
                    }
                    if (token < vv) {
#pragma unroll
                        for (unsigned j = 0; j < 8; j++)
                            sf[j] = __builtin_bit_cast(bf16_t, f2bf(st[token * D + d + j]));
                    }
                    pred = plow_mfma_bf16_16x16(wf, sf, pred);
                    from_state = plow_mfma_bf16_16x16(qf, sf, from_state);
                }
#pragma unroll
                for (unsigned e = 0; e < 4; e++) {
                    const unsigned ml = 4u * kgroup + e;
                    if (ml < mv && token < vv) {
                        const size_t row = row0 + m0 + ml;
                        vsm[(m0 + ml) * 16u + token] =
                            f2bf(bf2f(U[(row * H + h) * V + v0 + token]) - pred[e]);
                        osm[(m0 + ml) * 16u + token] = from_state[e];
                    }
                }
            }
            __syncthreads();
            for (unsigned mb = wave; mb < 4u; mb += PLOW_WAVES) {
                const unsigned m0 = mb * 16u;
                const unsigned mv = desc.y > m0 ? min(16u, desc.y - m0) : 0u;
                if (mv == 0) continue;
                f32x4 local = (f32x4)(0.0f);
                for (unsigned s0 = 0; s0 < 64u; s0 += 32u) {
                    bf16x8 af = (bf16x8)((bf16_t)0), vf = (bf16x8)((bf16_t)0);
                    const unsigned s = s0 + 8u * kgroup;
#pragma unroll
                    for (unsigned j = 0; j < 8; j++) {
                        if (token < mv && s + j < desc.y) {
                            const size_t ai = ((row0 + m0 + token) * H + h) * 64u + s + j;
                            af[j] = __builtin_bit_cast(bf16_t, f2bf(Aqk[ai]));
                        }
                        if (token < vv && s + j < desc.y)
                            vf[j] = __builtin_bit_cast(bf16_t, vsm[(s + j) * 16u + token]);
                    }
                    local = plow_mfma_bf16_16x16(af, vf, local);
                }
#pragma unroll
                for (unsigned e = 0; e < 4; e++) {
                    const unsigned ml = 4u * kgroup + e;
                    if (ml < mv && token < vv) {
                        const size_t row = row0 + m0 + ml;
                        out[(row * H + h) * V + v0 + token] =
                            f2bf(osm[(m0 + ml) * 16u + token] + local[e]);
                    }
                }
            }
            __syncthreads();

            const size_t last = row0 + desc.y - 1u;
            for (unsigned d0 = wave * 16u; d0 < D; d0 += PLOW_WAVES * 16u) {
                f32x4 upd = (f32x4)(0.0f);
                for (unsigned s0 = 0; s0 < 64u; s0 += 32u) {
                    bf16x8 vf = (bf16x8)((bf16_t)0), kf = (bf16x8)((bf16_t)0);
                    bf16x8 kr = (bf16x8)((bf16_t)0);
                    const unsigned s = s0 + 8u * kgroup;
#pragma unroll
                    for (unsigned j = 0; j < 8; j++) {
                        if (token < vv && s + j < desc.y)
                            vf[j] = __builtin_bit_cast(bf16_t, vsm[(s + j) * 16u + token]);
                        if (d0 + token < D && s + j < desc.y) {
                            const size_t row = row0 + s + j;
                            const size_t i = (row * H + h) * D + d0 + token;
                            if constexpr (PRECOMPUTED_KEY) {
                                kf[j] = __builtin_bit_cast(bf16_t, key_hi[i]);
                                kr[j] = __builtin_bit_cast(bf16_t, key_lo[i]);
                            } else {
                                const size_t il = (last * H + h) * D + d0 + token;
                                const float scaled = bf2f(k[i]) * exp2f(g[il] - g[i]);
                                const bf16 high = f2bf(scaled);
                                kf[j] = __builtin_bit_cast(bf16_t, high);
                                kr[j] = __builtin_bit_cast(bf16_t, f2bf(scaled - bf2f(high)));
                            }
                        }
                    }
                    upd = plow_mfma_bf16_16x16(vf, kf, upd);
                    upd = plow_mfma_bf16_16x16(vf, kr, upd);
                }
#pragma unroll
                for (unsigned e = 0; e < 4; e++) {
                    const unsigned vl = 4u * kgroup + e;
                    if (vl < vv && d0 + token < D) {
                        const size_t il = (last * H + h) * D + d0 + token;
                        st[vl * D + d0 + token] =
                            st[vl * D + d0 + token] * exp2f(g[il]) + upd[e];
                    }
                }
            }
            __syncthreads();
        }
        for (unsigned x = tid; x < vv * D; x += blockDim.x)
            state[((size_t)h * V + v0 + x / D) * D + x % D] = st[x];
        __syncthreads();
    }
}

template <bool PRECOMPUTED_KEY>
__device__ void carry_entry(
    bf16* out, float* state, const bf16* q, const bf16* k, const bf16* key_hi,
    const bf16* key_lo, const bf16* W, const bf16* U, const float* Aqk,
    const float* g, unsigned T, unsigned H) {
    constexpr unsigned D = 128;
    extern __shared__ __align__(16) unsigned char shared[];
    float* st = reinterpret_cast<float*>(shared);
    bf16* vsm = reinterpret_cast<bf16*>(st + 16u * D);
    float* osm = reinterpret_cast<float*>(vsm + 64u * 16u);
    carry<PRECOMPUTED_KEY>(out, state, q, k, key_hi, key_lo, W, U, Aqk, g,
                           T, H, st, vsm, osm);
}

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_carry_control(bf16* out, float* state, const bf16* q, const bf16* k,
                     const bf16* W, const bf16* U, const float* Aqk, const float* g,
                     unsigned T, unsigned H) {
    carry_entry<false>(out, state, q, k, nullptr, nullptr, W, U, Aqk, g, T, H);
}

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_carry_precomputed(bf16* out, float* state, const bf16* q, const bf16* k,
                         const bf16* key_hi, const bf16* key_lo, const bf16* W,
                         const bf16* U, const float* Aqk, const float* g,
                         unsigned T, unsigned H) {
    carry_entry<true>(out, state, q, k, key_hi, key_lo, W, U, Aqk, g, T, H);
}

#define PLOW_WG_WAVES 8
#include "op_elementwise.h"
#include "op_kda.h"
#include "op_kda_carry_regstate.h"

namespace {

constexpr unsigned BT = 64u;
constexpr unsigned D = 128u;
constexpr unsigned V = 128u;

/* The shipping V16/WG512 body with s_memtime stamps around its three phases and barriers.
 * Wave 0 lane 0 accumulates per-workgroup cycle totals: [p1, b1, p2, b2, p3, b3, total]. */
__device__ void carry_control_timed(
    bf16* __restrict__ out, float* __restrict__ state, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ W, const bf16* __restrict__ U,
    const float* __restrict__ Aqk, const float* __restrict__ g_cumsum_log2,
    unsigned n_chunks, unsigned T, unsigned H, unsigned slice, unsigned nblk,
    float* __restrict__ st, bf16* __restrict__ vsm, float* __restrict__ osm,
    unsigned long long* __restrict__ timers) {
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned token = lane & 15u, kgroup = lane >> 4;
    const unsigned vtiles = (V + 15u) / 16u;
    const size_t n_items = (size_t)H * vtiles;
    unsigned long long acc[7] = {0, 0, 0, 0, 0, 0, 0};
    const unsigned long long t_begin = __builtin_amdgcn_s_memtime();
    for (size_t item = slice; item < n_items; item += nblk) {
        const unsigned h = (unsigned)(item / vtiles), vt = (unsigned)(item % vtiles);
        const unsigned v0 = vt * 16u, vv = min(16u, V - v0);
        for (unsigned x = tid; x < vv * D; x += PLOW_THREADS)
            st[x] = state[((size_t)h * V + v0 + x / D) * D + x % D];
        __syncthreads();
        for (unsigned c = 0; c < n_chunks; c++) {
            const uint2 desc = kda_chunk_desc(nullptr, c, T);
            const size_t row0 = desc.x;
            const unsigned long long t0 = __builtin_amdgcn_s_memtime();
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
            const unsigned long long t1 = __builtin_amdgcn_s_memtime();
            __syncthreads();
            const unsigned long long t2 = __builtin_amdgcn_s_memtime();
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
            const unsigned long long t3 = __builtin_amdgcn_s_memtime();
            __syncthreads();
            const unsigned long long t4 = __builtin_amdgcn_s_memtime();
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
                            const size_t il = (last * H + h) * D + d0 + token;
                            const float scaled =
                                bf2f(k[i]) * exp2f(g_cumsum_log2[il] - g_cumsum_log2[i]);
                            const bf16 high = f2bf(scaled);
                            kf[j] = __builtin_bit_cast(bf16_t, high);
                            kr[j] = __builtin_bit_cast(bf16_t, f2bf(scaled - bf2f(high)));
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
                            st[vl * D + d0 + token] * exp2f(g_cumsum_log2[il]) + upd[e];
                    }
                }
            }
            const unsigned long long t5 = __builtin_amdgcn_s_memtime();
            __syncthreads();
            const unsigned long long t6 = __builtin_amdgcn_s_memtime();
            acc[0] += t1 - t0; acc[1] += t2 - t1; acc[2] += t3 - t2;
            acc[3] += t4 - t3; acc[4] += t5 - t4; acc[5] += t6 - t5;
        }
        for (unsigned x = tid; x < vv * D; x += PLOW_THREADS)
            state[((size_t)h * V + v0 + x / D) * D + x % D] = st[x];
        __syncthreads();
    }
    acc[6] = __builtin_amdgcn_s_memtime() - t_begin;
    if (tid == 0)
        for (unsigned i = 0; i < 7u; ++i) timers[slice * 8u + i] = acc[i];
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
void k_carry_control_timed(bf16* out, float* state, const bf16* q, const bf16* k,
                           const bf16* W, const bf16* U, const float* Aqk, const float* g,
                           unsigned T, unsigned H, unsigned long long* timers) {
    extern __shared__ __align__(16) unsigned char shared[];
    float* st = reinterpret_cast<float*>(shared);
    bf16* vsm = reinterpret_cast<bf16*>(st + 16u * D);
    float* osm = reinterpret_cast<float*>(vsm + BT * 16u);
    carry_control_timed(out, state, q, k, W, U, Aqk, g, (T + BT - 1u) / BT, T, H,
                        blockIdx.x, gridDim.x, st, vsm, osm, timers);
}

extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void k_carry_regstate(bf16* out, float* state, const bf16* q, const bf16* k,
                      const bf16* W, const bf16* U, const float* Aqk, const float* g,
                      unsigned T, unsigned H) {
    __shared__ __align__(16) bf16 lds[plow_kda_regstate::LDS_BYTES / 2u];
    d_kda_chunk_carry_bt64_regstate(out, state, q, k, W, U, Aqk, g, (T + BT - 1u) / BT,
                                    T, H, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void k_carry_regstate_hwcvt(bf16* out, float* state, const bf16* q, const bf16* k,
                            const bf16* W, const bf16* U, const float* Aqk, const float* g,
                            unsigned T, unsigned H) {
    __shared__ __align__(16) bf16 lds[plow_kda_regstate::LDS_BYTES / 2u];
    d_kda_chunk_carry_bt64_regstate<false, true>(out, state, q, k, W, U, Aqk, g,
                                                 (T + BT - 1u) / BT, T, H, blockIdx.x,
                                                 gridDim.x, lds);
}

/* The kda_chunk_key_factor pair: k exp2(g_last - g) split into bf16 hi/lo, [T][H][D]. */
extern "C" __global__ __launch_bounds__(256)
void k_key_precompute(bf16* key_hi, bf16* key_lo, const bf16* k, const float* g,
                      unsigned T, unsigned H) {
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
}

extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void k_carry_regstate_keyfeed(bf16* out, float* state, const bf16* q, const bf16* k,
                              const bf16* W, const bf16* U, const float* Aqk, const float* g,
                              unsigned T, unsigned H, const bf16* key_hi, const bf16* key_lo) {
    __shared__ __align__(16) bf16 lds[plow_kda_regstate::LDS_BYTES / 2u];
    d_kda_chunk_carry_bt64_regstate<false, false, true>(
        out, state, q, k, W, U, Aqk, g, (T + BT - 1u) / BT, T, H, blockIdx.x, gridDim.x, lds,
        nullptr, key_hi, key_lo);
}

extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void k_carry_regstate_keyfeed_timed(bf16* out, float* state, const bf16* q, const bf16* k,
                                    const bf16* W, const bf16* U, const float* Aqk,
                                    const float* g, unsigned T, unsigned H,
                                    unsigned long long* timers, const bf16* key_hi,
                                    const bf16* key_lo) {
    __shared__ __align__(16) bf16 lds[plow_kda_regstate::LDS_BYTES / 2u];
    d_kda_chunk_carry_bt64_regstate<true, false, true>(
        out, state, q, k, W, U, Aqk, g, (T + BT - 1u) / BT, T, H, blockIdx.x, gridDim.x, lds,
        timers, key_hi, key_lo);
}

extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void k_carry_regstate_timed(bf16* out, float* state, const bf16* q, const bf16* k,
                            const bf16* W, const bf16* U, const float* Aqk, const float* g,
                            unsigned T, unsigned H, unsigned long long* timers) {
    __shared__ __align__(16) bf16 lds[plow_kda_regstate::LDS_BYTES / 2u];
    d_kda_chunk_carry_bt64_regstate<true>(out, state, q, k, W, U, Aqk, g, (T + BT - 1u) / BT,
                                          T, H, blockIdx.x, gridDim.x, lds, timers);
}

/* Standalone batched-decode KDA chain (op 125, descriptor version 3, PLOW_KDA_DECODE_WIDE).
 *
 * One workgroup per (row, head) runs the interpreter's KdaConv3 -> KdaStateStepG -> KdaGatedNorm
 * chain for that pair through the SAME device bodies (kda_conv_range's independent-row arm,
 * d_kda_state_step_t's wide arm, d_kda_gated_norm's per-row body), so every element sees the
 * identical arithmetic and the outputs are bit-identical to the three interpreter packets. The
 * intermediates (mix q/k/v, o) still round-trip through their bf16 tensors; only the workgroup
 * that wrote them reads them. Unlike the interpreter (one 256-VGPR workgroup per CU, B*H/256
 * items serial per workgroup) every item is resident at once. The intermediates never cross a
 * workgroup, so __syncthreads' workgroup-scope fences order them (no agent-scope L2 writeback). */
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda.h"

extern "C" __device__ unsigned plow_kda_decode_wide_3 = 3;

/* PLOW_KDA_DECODE_LANES (object define, default 0): the state step with one value row per lane
 * quad. Lane (w, l) owns row v = 16w + l/4 and the 32 key positions d = 16i + 4(l%4) + e, so the
 * S^T k and S^T q reductions are two quad DPP steps for 16 rows at once instead of a 64-lane
 * reduction per column, and every state load issues up front. Same recurrence as
 * d_kda_state_step_t's wide arm; only the f32 summation order of pk / pq differs. */
#ifndef PLOW_KDA_DECODE_LANES
#define PLOW_KDA_DECODE_LANES 0
#endif
#if PLOW_KDA_DECODE_LANES
__device__ __forceinline__ float kda_quad_sum(float x) {
    x += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(
                                       0, __builtin_bit_cast(int, x), 0xB1, 0xF, 0xF, false));
    x += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(
                                       0, __builtin_bit_cast(int, x), 0x4E, 0xF, 0xF, false));
    return x;
}

__device__ void kda_state_step_lanes(
    bf16* __restrict__ o, const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ v, const bf16* __restrict__ g_raw,
    const bf16* __restrict__ beta_raw, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, unsigned gate_mode, float lb, float* __restrict__ state,
    unsigned row, unsigned h, unsigned H, unsigned flags, float scale, float* __restrict__ lds,
    const unsigned* __restrict__ parked) {
    constexpr unsigned D = 128u;
    if (parked && parked[row]) return;
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned vr = 16u * wave + (lane >> 2), part = lane & 3u;
    const size_t hd = (size_t)row * H * D + (size_t)h * D;
    float* srow = state + ((size_t)row * H + h) * D * D + (size_t)vr * D + 4u * part;
    f32x4 s[8];
#pragma unroll
    for (unsigned i = 0; i < 8u; ++i) s[i] = *reinterpret_cast<const f32x4*>(srow + 16u * i);
    float* l_q = lds;
    float* l_k = lds + D;
    float* l_g = lds + 2 * D;
    const float a_h = __expf(a_log[h]);
    float qs = 0.0f, ks = 0.0f;
    for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
        const float qv = bf2f(q[hd + d]), kv = bf2f(k[hd + d]);
        l_q[d] = qv;
        l_k[d] = kv;
        const float sgm = bf2f(g_raw[hd + d]) + dt_bias[(size_t)h * D + d];
        const float gv = (gate_mode == PLOW_KDA_GATE_LOWER_BOUND) ? lb * kda_sigmoid(a_h * sgm)
                                                                  : -a_h * kda_softplus(sgm);
        l_g[d] = __expf(gv);
        qs += qv * qv;
        ks += kv * kv;
    }
    if (flags & PLOW_KDA_F_QK_L2NORM) {
        qs = block_sum(qs, lds + 3 * D);
        ks = block_sum(ks, lds + 3 * D + PLOW_WAVES);
        const float rq = scale * rsqrtf(qs + 1e-6f), rk = rsqrtf(ks + 1e-6f);
        for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
            l_q[d] *= rq;
            l_k[d] *= rk;
        }
    } else {
        for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) l_q[d] *= scale;
    }
    __syncthreads();
    const float b = kda_sigmoid(bf2f(beta_raw[(size_t)row * H + h]));
    float pk = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < 8u; ++i) {
        const f32x4 gg = *reinterpret_cast<const f32x4*>(l_g + 16u * i + 4u * part);
        const f32x4 kk = *reinterpret_cast<const f32x4*>(l_k + 16u * i + 4u * part);
#pragma unroll
        for (unsigned e = 0; e < 4u; ++e) {
            s[i][e] *= gg[e];
            pk += s[i][e] * kk[e];
        }
    }
    pk = kda_quad_sum(pk);
    const float bu = b * (bf2f(v[hd + vr]) - pk);
    float pq = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < 8u; ++i) {
        const f32x4 kk = *reinterpret_cast<const f32x4*>(l_k + 16u * i + 4u * part);
        const f32x4 qq = *reinterpret_cast<const f32x4*>(l_q + 16u * i + 4u * part);
#pragma unroll
        for (unsigned e = 0; e < 4u; ++e) {
            s[i][e] += bu * kk[e];
            pq += s[i][e] * qq[e];
        }
        *reinterpret_cast<f32x4*>(srow + 16u * i) = s[i];
    }
    pq = kda_quad_sum(pq);
    if (part == 0u) o[hd + vr] = f2bf(pq);
}
#endif

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void plow_kda_decode_wide_v3(
    bf16* y, const bf16* q_raw, const bf16* k_raw, const bf16* v_raw,
    const float* wq, const float* wk, const float* wv, float* csq, float* csk,
    float* csv, const bf16* forget_raw, const bf16* beta_raw,
    const bf16* output_gate_raw, const float* a_log, const float* dt_bias, float* state,
    const float* norm_w, const unsigned* parked, unsigned rows, unsigned heads, unsigned dim,
    unsigned bv, unsigned conv_w, unsigned flags, unsigned gate_mode, float lower_bound,
    float scale, float norm_eps, bf16* mq, bf16* mk, bf16* mv, bf16* o) {
    if (blockDim.x != PLOW_THREADS || rows < 2u || heads == 0u || dim != 128u || bv == 0u ||
        conv_w == 0u || conv_w > 8u || !(flags & PLOW_KDA_F_SEQ_ROWS) ||
        (flags & ~(PLOW_KDA_F_QK_L2NORM | PLOW_KDA_F_SEQ_ROWS)) || gridDim.x != rows * heads)
        __builtin_trap();
    __shared__ float lds[5u * 128u + 2u * PLOW_WAVES];
    const unsigned item = blockIdx.x;
    const unsigned row = item / heads;
    const unsigned h = item - row * heads;
    const unsigned proj = heads * dim;
    const size_t rp = (size_t)row * proj;
    const size_t rcs = rp * conv_w;
    const unsigned* pk = parked ? parked + row : nullptr;
    const unsigned c0 = h * dim, c1 = c0 + dim;
    /* op 109, independent-sequence arm: this row's window, this head's channels. */
    kda_conv_range(mq + rp, q_raw + rp, wq, csq + rcs, 1u, proj, conv_w, 1u, c0, c1,
                   (size_t)proj * conv_w, pk);
    kda_conv_range(mk + rp, k_raw + rp, wk, csk + rcs, 1u, proj, conv_w, 1u, c0, c1,
                   (size_t)proj * conv_w, pk);
    kda_conv_range(mv + rp, v_raw + rp, wv, csv + rcs, 1u, proj, conv_w, 1u, c0, c1,
                   (size_t)proj * conv_w, pk);
    __syncthreads();
    /* op 110, wide arm: grid == T*H items, so this workgroup's only item is `item`. */
#if PLOW_KDA_DECODE_LANES
    kda_state_step_lanes(o, mq, mk, mv, forget_raw, beta_raw, a_log, dt_bias, gate_mode,
                         lower_bound, state, row, h, heads, flags, scale, lds, parked);
#else
    d_kda_state_step_g(o, mq, mk, mv, forget_raw, beta_raw, a_log, dt_bias, gate_mode,
                       lower_bound, state, rows, heads, dim, bv, flags, scale, item,
                       rows * heads, lds, (size_t)heads * dim * dim, parked);
#endif
    __syncthreads();
    /* op 103 for row `item` (= row * H + h) on wave 0: d_kda_gated_norm's body. */
    if ((threadIdx.x >> 6) == 0u) {
        const unsigned lane = threadIdx.x & (PLOW_WAVE - 1);
        const size_t base = (size_t)item * dim;
        float ss = 0.0f;
        for (unsigned d = lane; d < dim; d += PLOW_WAVE) {
            const float x = bf2f(o[base + d]);
            ss += x * x;
        }
        const float inv = rsqrtf(wave_sum(ss) / (float)dim + norm_eps);
        for (unsigned d = lane; d < dim; d += PLOW_WAVE)
            y[base + d] = f2bf(bf2f(o[base + d]) * inv * norm_w[d] *
                               kda_sigmoid(bf2f(output_gate_raw[base + d])));
    }
}

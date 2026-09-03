#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda.h"

extern "C" __device__ unsigned plow_kda_decode_fused_256x16_2 = 2;

namespace {
constexpr unsigned KdaD = 128;
constexpr unsigned KdaBv = 8;
constexpr unsigned KdaW = 4;
constexpr unsigned KdaGroups = 16;
constexpr unsigned KdaIters = KdaD / KdaGroups;

template <int Ctrl>
__device__ __forceinline__ float kda_fused_dpp_add(float x) {
    const int shuffled = __builtin_amdgcn_update_dpp(
        0, __builtin_bit_cast(int, x), Ctrl, 0xf, 0xf, true);
    return x + __builtin_bit_cast(float, shuffled);
}

__device__ __forceinline__ float kda_fused_row_sum(float x) {
    x = kda_fused_dpp_add<0xb1>(x);
    x = kda_fused_dpp_add<0x4e>(x);
    x = kda_fused_dpp_add<0x141>(x);
    return kda_fused_dpp_add<0x140>(x);
}

__device__ __forceinline__ f32x4 kda_fused_load_state(const float* p) {
    return __builtin_nontemporal_load(reinterpret_cast<const f32x4*>(p));
}

__device__ __forceinline__ void kda_fused_store_state(float* p, f32x4 x) {
    __builtin_nontemporal_store(x, reinterpret_cast<f32x4*>(p));
}

__device__ __forceinline__ bf16 kda_fused_conv4(
    const bf16* raw, const float* weight, float* conv_state, unsigned channel) {
    const float x1 = conv_state[(size_t)channel * KdaW + 1];
    const float x2 = conv_state[(size_t)channel * KdaW + 2];
    const float x3 = conv_state[(size_t)channel * KdaW + 3];
    const float x4 = bf2f(raw[channel]);
    const float value = x1 * weight[(size_t)channel * KdaW] +
                        x2 * weight[(size_t)channel * KdaW + 1] +
                        x3 * weight[(size_t)channel * KdaW + 2] +
                        x4 * weight[(size_t)channel * KdaW + 3];
    conv_state[(size_t)channel * KdaW] = x1;
    conv_state[(size_t)channel * KdaW + 1] = x2;
    conv_state[(size_t)channel * KdaW + 2] = x3;
    conv_state[(size_t)channel * KdaW + 3] = x4;
    return f2bf(act_silu(value));
}
}  // namespace

extern "C" __global__ __launch_bounds__(256)
void plow_kda_decode_fused_256x16_v2(
    bf16* y, const bf16* q_raw, const bf16* k_raw, const bf16* v_raw,
    const float* wq, const float* wk, const float* wv, float* csq, float* csk,
    float* csv, const bf16* forget_raw, const bf16* beta_raw,
    const bf16* output_gate_raw, const float* a_log, const float* dt_bias, float* state,
    const float* norm_w, const unsigned* parked, unsigned rows, unsigned heads,
    unsigned dim, unsigned bv, unsigned conv_w, unsigned flags,
    unsigned gate_mode, float lower_bound, float scale, float norm_eps) {
    if (blockDim.x != 256 || (rows != 1 && rows != 8) || heads == 0 ||
        dim != KdaD || bv != KdaBv || conv_w != KdaW ||
        !(flags & PLOW_KDA_F_QK_L2NORM) ||
        (rows > 1 && !(flags & PLOW_KDA_F_SEQ_ROWS)) ||
        gate_mode != PLOW_KDA_GATE_LOWER_BOUND || gridDim.x != rows * heads)
        __builtin_trap();

    const unsigned row = blockIdx.x / heads;
    const unsigned h = blockIdx.x - row * heads;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned rlane = tid & 15u;
    const unsigned group = tid >> 4;
    const size_t proj = (size_t)heads * KdaD;
    const size_t row_p = (size_t)row * proj;
    const size_t row_cs = row_p * KdaW;
    const size_t hd0 = (size_t)h * KdaD;
    if (parked && parked[row]) {
        if (tid < KdaD) y[row_p + hd0 + tid] = f2bf(0.0f);
        return;
    }
    q_raw += row_p;
    k_raw += row_p;
    v_raw += row_p;
    csq += row_cs;
    csk += row_cs;
    csv += row_cs;
    forget_raw += row_p;
    output_gate_raw += row_p;
    beta_raw += (size_t)row * heads;
    state += (size_t)row * heads * KdaD * KdaD;
    y += row_p;

    f32x4 state_raw[KdaIters][2];
    const unsigned d0 = 4u * rlane;
#pragma unroll
    for (unsigned t = 0; t < KdaIters; ++t) {
        const unsigned j = t * KdaGroups + group;
        const float* column = state + (hd0 + j) * KdaD;
        state_raw[t][0] = kda_fused_load_state(column + d0);
        state_raw[t][1] = kda_fused_load_state(column + 64u + d0);
    }

    __shared__ float q[KdaD], k[KdaD], decay[KdaD], v[KdaD], sums[8];
    __shared__ bf16 out[KdaD];
    float qss = 0.0f, kss = 0.0f;
    if (tid < KdaD) {
        const unsigned channel = (unsigned)hd0 + tid;
        const bf16 qbf = kda_fused_conv4(q_raw, wq, csq, channel);
        const bf16 kbf = kda_fused_conv4(k_raw, wk, csk, channel);
        const bf16 vbf = kda_fused_conv4(v_raw, wv, csv, channel);
        const float qv = bf2f(qbf), kv = bf2f(kbf);
        q[tid] = qv;
        k[tid] = kv;
        v[tid] = bf2f(vbf);
        decay[tid] = __expf(lower_bound * kda_sigmoid(
            __expf(a_log[h]) * (bf2f(forget_raw[hd0 + tid]) + dt_bias[hd0 + tid])));
        qss = qv * qv;
        kss = kv * kv;
    }
    qss = wave_sum(qss);
    kss = wave_sum(kss);
    if (lane == 0) {
        sums[tid >> 6] = qss;
        sums[4 + (tid >> 6)] = kss;
    }
    __syncthreads();
    const float rq = scale * rsqrtf(sums[0] + sums[1] + sums[2] + sums[3] + 1.0e-6f);
    const float rk = rsqrtf(sums[4] + sums[5] + sums[6] + sums[7] + 1.0e-6f);
    if (tid < KdaD) {
        q[tid] *= rq;
        k[tid] *= rk;
    }
    __syncthreads();

#pragma unroll
    for (unsigned t = 0; t < KdaIters; ++t) {
        const unsigned j = t * KdaGroups + group;
        f32x4 s0 = state_raw[t][0], s1 = state_raw[t][1];
        float hk = 0.0f, hq = 0.0f, kq = 0.0f;
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            const unsigned d = d0 + e;
            s0[e] *= decay[d];
            hk += s0[e] * k[d];
            hq += s0[e] * q[d];
            kq += k[d] * q[d];
        }
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            const unsigned d = 64u + d0 + e;
            s1[e] *= decay[d];
            hk += s1[e] * k[d];
            hq += s1[e] * q[d];
            kq += k[d] * q[d];
        }
        hk = kda_fused_row_sum(hk);
        hq = kda_fused_row_sum(hq);
        kq = kda_fused_row_sum(kq);
        const float update = kda_sigmoid(bf2f(beta_raw[h])) * (v[j] - hk);
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            s0[e] += update * k[d0 + e];
            s1[e] += update * k[64u + d0 + e];
        }
        float* column = state + (hd0 + j) * KdaD;
        kda_fused_store_state(column + d0, s0);
        kda_fused_store_state(column + 64u + d0, s1);
        if (rlane == 0) out[j] = f2bf(hq + update * kq);
    }
    __syncthreads();

    float oss = 0.0f;
    if (tid < KdaD) {
        const float ov = bf2f(out[tid]);
        oss = ov * ov;
    }
    oss = wave_sum(oss);
    if (lane == 0) sums[tid >> 6] = oss;
    __syncthreads();
    const float inv = rsqrtf((sums[0] + sums[1] + sums[2] + sums[3]) /
                                 (float)KdaD + norm_eps);
    if (tid < KdaD) {
        const float ov = bf2f(out[tid]);
        y[hd0 + tid] = f2bf(ov * inv * norm_w[tid] *
                             kda_sigmoid(bf2f(output_gate_raw[hd0 + tid])));
    }
}

#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda.h"

namespace {
constexpr unsigned H = 12;
constexpr unsigned D = 128;
constexpr unsigned BV = 8;
constexpr unsigned W = 4;
constexpr unsigned P = H * D;

__device__ __forceinline__ bf16 conv_one(const bf16* raw, const float* weight,
                                         const float* source, float* target,
                                         unsigned channel) {
    float win[W], tap[W];
#pragma unroll
    for (unsigned j = 0; j < W; ++j) {
        win[j] = source[(size_t)channel * W + j];
        tap[j] = weight[(size_t)channel * W + j];
    }
#pragma unroll
    for (unsigned j = 0; j + 1 < W; ++j) win[j] = win[j + 1];
    win[W - 1] = bf2f(raw[channel]);
    float value = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < W; ++j) value += win[j] * tap[j];
#pragma unroll
    for (unsigned j = 0; j < W; ++j) target[(size_t)channel * W + j] = win[j];
    return f2bf(act_silu(value));
}
}  // namespace

extern "C" __global__ __launch_bounds__(512, 1)
void kda_decode_fused(bf16* y, const bf16* q_raw, const bf16* k_raw,
                      const bf16* v_raw, const float* weight,
                      const float* conv_source, float* conv_target,
                      const bf16* gate_raw, const bf16* beta_raw,
                      const float* a_log, const float* dt_bias, float* state,
                      const float* norm_w, unsigned rows) {
    const unsigned row = blockIdx.x / H;
    const unsigned h = blockIdx.x - row * H;
    if (row >= rows) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned wave = tid >> 6;
    const size_t row_p = (size_t)row * P;
    const size_t row_cs = row_p * W;
    const size_t hd0 = (size_t)h * D;
    q_raw += row_p;
    k_raw += row_p;
    v_raw += row_p;
    conv_source += row_cs;
    conv_target += row_cs;
    gate_raw += row_p;
    beta_raw += (size_t)row * H;
    state += (size_t)row * H * D * D;
    y += row_p;

    __shared__ float q[D], k[D], decay[D], v[BV], red[16];
    __shared__ bf16 out[D];
    float qsum = 0.0f, ksum = 0.0f;
    if (tid < D) {
        const unsigned c = (unsigned)hd0 + tid;
        const bf16 qv = conv_one(q_raw, weight, conv_source, conv_target, c);
        const bf16 kv = conv_one(k_raw, weight + (size_t)P * W,
                                 conv_source + (size_t)rows * P * W,
                                 conv_target + (size_t)rows * P * W, c);
        q[tid] = bf2f(qv);
        k[tid] = bf2f(kv);
        decay[tid] = __expf(-5.0f * kda_sigmoid(
            __expf(a_log[h]) * (bf2f(gate_raw[hd0 + tid]) + dt_bias[hd0 + tid])));
        qsum = q[tid] * q[tid];
        ksum = k[tid] * k[tid];
    }
    qsum = block_sum(qsum, red);
    ksum = block_sum(ksum, red + 8);
    const float rq = rsqrtf((float)D) / sqrtf(qsum + 1.0e-6f);
    const float rk = 1.0f / sqrtf(ksum + 1.0e-6f);
    if (tid < D) {
        q[tid] *= rq;
        k[tid] *= rk;
    }
    __syncthreads();

    for (unsigned tile = 0; tile < D / BV; ++tile) {
        if (tid < BV) {
            const unsigned c = (unsigned)hd0 + tile * BV + tid;
            v[tid] = bf2f(conv_one(v_raw, weight + (size_t)2u * P * W,
                                   conv_source + (size_t)rows * 2u * P * W,
                                   conv_target + (size_t)rows * 2u * P * W, c));
        }
        __syncthreads();
        const unsigned j = tile * BV + wave;
        float* column = state + hd0 * D + (size_t)j * D;
        float sc0 = column[lane] * decay[lane];
        float sc1 = column[64u + lane] * decay[64u + lane];
        float pk = wave_sum(sc0 * k[lane] + sc1 * k[64u + lane]);
        const float update = kda_sigmoid(bf2f(beta_raw[h])) * (v[wave] - pk);
        sc0 += update * k[lane];
        sc1 += update * k[64u + lane];
        column[lane] = sc0;
        column[64u + lane] = sc1;
        const float value = wave_sum(sc0 * q[lane] + sc1 * q[64u + lane]);
        if (lane == 0) out[j] = f2bf(value);
        __syncthreads();
    }

    if (wave == 0) {
        float ss = 0.0f;
        for (unsigned d = lane; d < D; d += 64u) {
            const float value = bf2f(out[d]);
            ss += value * value;
        }
        const float inv = rsqrtf(wave_sum(ss) / (float)D + 1.0e-5f);
        for (unsigned d = lane; d < D; d += 64u)
            y[hd0 + d] = f2bf(bf2f(out[d]) * inv * norm_w[d] *
                                  kda_sigmoid(bf2f(gate_raw[hd0 + d])));
    }
}

extern "C" __global__ __launch_bounds__(512, 1)
void kda_conv_control(bf16* mix, const bf16* q_raw, const bf16* k_raw,
                      const bf16* v_raw, const float* weight,
                      float* conv_state, unsigned rows) {
    d_kda_conv3(mix, mix + (size_t)rows * P, mix + (size_t)rows * 2u * P,
                q_raw, k_raw, v_raw,
                weight, weight + (size_t)P * W, weight + (size_t)2u * P * W,
                conv_state, conv_state + (size_t)rows * P * W,
                conv_state + (size_t)rows * 2u * P * W, rows, P, W, 1,
                blockIdx.x, gridDim.x, (size_t)P * W, nullptr);
}

extern "C" __global__ __launch_bounds__(512, 1)
void kda_step_control(bf16* output, const bf16* mix, const bf16* gate_raw,
                      const bf16* beta_raw, const float* a_log,
                      const float* dt_bias, float* state, unsigned rows) {
    extern __shared__ float lds[];
    d_kda_state_step_g(output, mix, mix + (size_t)rows * P,
                       mix + (size_t)rows * 2u * P, gate_raw, beta_raw, a_log,
                       dt_bias, PLOW_KDA_GATE_LOWER_BOUND, -5.0f, state, rows,
                       H, D, BV, PLOW_KDA_F_QK_L2NORM | PLOW_KDA_F_SEQ_ROWS,
                       rsqrtf((float)D), blockIdx.x, gridDim.x, lds,
                       (size_t)H * D * D, nullptr);
}

template <int Ctrl>
__device__ __forceinline__ float kda_bv8_dpp_add(float x) {
    const int shuffled = __builtin_amdgcn_update_dpp(
        0, __builtin_bit_cast(int, x), Ctrl, 0xf, 0xf, true);
    return x + __builtin_bit_cast(float, shuffled);
}

__device__ __forceinline__ float kda_bv8_row_sum(float x) {
    x = kda_bv8_dpp_add<0xb1>(x);
    x = kda_bv8_dpp_add<0x4e>(x);
    x = kda_bv8_dpp_add<0x141>(x);
    return kda_bv8_dpp_add<0x140>(x);
}

__device__ __forceinline__ f32x4 kda_bv8_load_state(const float* p) {
    return __builtin_nontemporal_load(reinterpret_cast<const f32x4*>(p));
}

__device__ __forceinline__ void kda_bv8_store_state(float* p, f32x4 x) {
    __builtin_nontemporal_store(x, reinterpret_cast<f32x4*>(p));
}

extern "C" __global__ __launch_bounds__(128)
void kda_step_bv8x16(bf16* output, const bf16* mix, const bf16* gate_raw,
                     const bf16* beta_raw, const float* a_log,
                     const float* dt_bias, float* state, unsigned rows) {
    constexpr unsigned Tiles = D / BV;
    const unsigned item = blockIdx.x;
    const unsigned row = item / (H * Tiles);
    if (row >= rows) return;
    const unsigned rem = item - row * H * Tiles;
    const unsigned h = rem / Tiles;
    const unsigned tile = rem - h * Tiles;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned rlane = tid & 15u;
    const unsigned group = tid >> 4;
    const size_t row_p = (size_t)row * P;
    const size_t hd0 = (size_t)h * D;

    __shared__ float q[D], k[D], decay[D], v[BV], sums[4];
    const float qv = bf2f(mix[row_p + hd0 + tid]);
    const float kv = bf2f(mix[(size_t)rows * P + row_p + hd0 + tid]);
    q[tid] = qv;
    k[tid] = kv;
    const float gate_input = bf2f(gate_raw[row_p + hd0 + tid]) + dt_bias[hd0 + tid];
    decay[tid] = __expf(-5.0f * kda_sigmoid(__expf(a_log[h]) * gate_input));
    const float qs = wave_sum(qv * qv);
    const float ks = wave_sum(kv * kv);
    if (lane == 0) {
        sums[tid >> 6] = qs;
        sums[2 + (tid >> 6)] = ks;
    }
    if (tid < BV)
        v[tid] = bf2f(mix[(size_t)rows * 2u * P + row_p + hd0 + tile * BV + tid]);
    __syncthreads();
    const float rq = rsqrtf((float)D) * rsqrtf(sums[0] + sums[1] + 1.0e-6f);
    const float rk = rsqrtf(sums[2] + sums[3] + 1.0e-6f);
    q[tid] *= rq;
    k[tid] *= rk;
    __syncthreads();

    const unsigned j = tile * BV + group;
    float* column = state + (size_t)row * H * D * D + (hd0 + j) * D;
    const unsigned d0 = 4u * rlane;
    f32x4 s0 = kda_bv8_load_state(column + d0);
    f32x4 s1 = kda_bv8_load_state(column + 64u + d0);
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
    hk = kda_bv8_row_sum(hk);
    hq = kda_bv8_row_sum(hq);
    kq = kda_bv8_row_sum(kq);
    const float update = kda_sigmoid(bf2f(beta_raw[(size_t)row * H + h])) * (v[group] - hk);
#pragma unroll
    for (unsigned e = 0; e < 4; ++e) {
        s0[e] += update * k[d0 + e];
        s1[e] += update * k[64u + d0 + e];
    }
    kda_bv8_store_state(column + d0, s0);
    kda_bv8_store_state(column + 64u + d0, s1);
    if (rlane == 0) output[row_p + hd0 + j] = f2bf(hq + update * kq);
}

extern "C" __global__ __launch_bounds__(128)
void kda_decode_fused_bv8x16(
    bf16* y, const bf16* q_raw, const bf16* k_raw, const bf16* v_raw,
    const float* weight, const float* conv_source, float* conv_target,
    const bf16* gate_raw, const bf16* beta_raw, const float* a_log,
    const float* dt_bias, float* state, const float* norm_w, unsigned rows) {
    const unsigned row = blockIdx.x / H;
    const unsigned h = blockIdx.x - row * H;
    if (row >= rows) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned rlane = tid & 15u;
    const unsigned group = tid >> 4;
    const size_t row_p = (size_t)row * P;
    const size_t row_cs = row_p * W;
    const size_t hd0 = (size_t)h * D;
    q_raw += row_p;
    k_raw += row_p;
    v_raw += row_p;
    conv_source += row_cs;
    conv_target += row_cs;
    gate_raw += row_p;
    beta_raw += (size_t)row * H;
    state += (size_t)row * H * D * D;
    y += row_p;

    __shared__ float q[D], k[D], decay[D], v[BV], sums[4];
    __shared__ bf16 out[D];
    const unsigned c = (unsigned)hd0 + tid;
    const bf16 qbf = conv_one(q_raw, weight, conv_source, conv_target, c);
    const bf16 kbf = conv_one(k_raw, weight + (size_t)P * W,
                              conv_source + (size_t)rows * P * W,
                              conv_target + (size_t)rows * P * W, c);
    const float qv = bf2f(qbf), kv = bf2f(kbf);
    q[tid] = qv;
    k[tid] = kv;
    decay[tid] = __expf(-5.0f * kda_sigmoid(
        __expf(a_log[h]) * (bf2f(gate_raw[hd0 + tid]) + dt_bias[hd0 + tid])));
    const float qs = wave_sum(qv * qv);
    const float ks = wave_sum(kv * kv);
    if (lane == 0) {
        sums[tid >> 6] = qs;
        sums[2 + (tid >> 6)] = ks;
    }
    __syncthreads();
    const float rq = rsqrtf((float)D) * rsqrtf(sums[0] + sums[1] + 1.0e-6f);
    const float rk = rsqrtf(sums[2] + sums[3] + 1.0e-6f);
    q[tid] *= rq;
    k[tid] *= rk;
    __syncthreads();

    for (unsigned tile = 0; tile < D / BV; ++tile) {
        if (rlane == 0) {
            const unsigned vc = (unsigned)hd0 + tile * BV + group;
            v[group] = bf2f(conv_one(v_raw, weight + (size_t)2u * P * W,
                                     conv_source + (size_t)rows * 2u * P * W,
                                     conv_target + (size_t)rows * 2u * P * W, vc));
        }
        __syncthreads();
        const unsigned j = tile * BV + group;
        float* column = state + (hd0 + j) * D;
        const unsigned d0 = 4u * rlane;
        f32x4 s0 = kda_bv8_load_state(column + d0);
        f32x4 s1 = kda_bv8_load_state(column + 64u + d0);
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
        hk = kda_bv8_row_sum(hk);
        hq = kda_bv8_row_sum(hq);
        kq = kda_bv8_row_sum(kq);
        const float update = kda_sigmoid(bf2f(beta_raw[h])) * (v[group] - hk);
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            s0[e] += update * k[d0 + e];
            s1[e] += update * k[64u + d0 + e];
        }
        kda_bv8_store_state(column + d0, s0);
        kda_bv8_store_state(column + 64u + d0, s1);
        if (rlane == 0) out[j] = f2bf(hq + update * kq);
        __syncthreads();
    }

    const float ov = bf2f(out[tid]);
    const float ss = wave_sum(ov * ov);
    if (lane == 0) sums[tid >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((sums[0] + sums[1]) / (float)D + 1.0e-5f);
    y[hd0 + tid] = f2bf(ov * inv * norm_w[tid] *
                         kda_sigmoid(bf2f(gate_raw[hd0 + tid])));
}

extern "C" __global__ __launch_bounds__(256)
void kda_decode_fused_256x16(
    bf16* y, const bf16* q_raw, const bf16* k_raw, const bf16* v_raw,
    const float* weight, const float* conv_source, float* conv_target,
    const bf16* gate_raw, const bf16* beta_raw, const float* a_log,
    const float* dt_bias, float* state, const float* norm_w, unsigned rows) {
    constexpr unsigned Groups = 16;
    constexpr unsigned Iters = D / Groups;
    const unsigned row = blockIdx.x / H;
    const unsigned h = blockIdx.x - row * H;
    if (row >= rows) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned rlane = tid & 15u;
    const unsigned group = tid >> 4;
    const size_t row_p = (size_t)row * P;
    const size_t row_cs = row_p * W;
    const size_t hd0 = (size_t)h * D;
    q_raw += row_p;
    k_raw += row_p;
    v_raw += row_p;
    conv_source += row_cs;
    conv_target += row_cs;
    gate_raw += row_p;
    beta_raw += (size_t)row * H;
    state += (size_t)row * H * D * D;
    y += row_p;

    f32x4 state_raw[Iters][2];
    const unsigned d0 = 4u * rlane;
#pragma unroll
    for (unsigned t = 0; t < Iters; ++t) {
        const unsigned j = t * Groups + group;
        const float* column = state + (hd0 + j) * D;
        state_raw[t][0] = kda_bv8_load_state(column + d0);
        state_raw[t][1] = kda_bv8_load_state(column + 64u + d0);
    }

    __shared__ float q[D], k[D], decay[D], v[D], sums[8];
    __shared__ bf16 out[D];
    float qss = 0.0f, kss = 0.0f;
    if (tid < D) {
        const unsigned c = (unsigned)hd0 + tid;
        const bf16 qbf = conv_one(q_raw, weight, conv_source, conv_target, c);
        const bf16 kbf = conv_one(k_raw, weight + (size_t)P * W,
                                  conv_source + (size_t)rows * P * W,
                                  conv_target + (size_t)rows * P * W, c);
        const bf16 vbf = conv_one(v_raw, weight + (size_t)2u * P * W,
                                  conv_source + (size_t)rows * 2u * P * W,
                                  conv_target + (size_t)rows * 2u * P * W, c);
        const float qv = bf2f(qbf), kv = bf2f(kbf);
        q[tid] = qv;
        k[tid] = kv;
        v[tid] = bf2f(vbf);
        decay[tid] = __expf(-5.0f * kda_sigmoid(
            __expf(a_log[h]) * (bf2f(gate_raw[hd0 + tid]) + dt_bias[hd0 + tid])));
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
    const float rq = rsqrtf((float)D) *
                     rsqrtf(sums[0] + sums[1] + sums[2] + sums[3] + 1.0e-6f);
    const float rk = rsqrtf(sums[4] + sums[5] + sums[6] + sums[7] + 1.0e-6f);
    if (tid < D) {
        q[tid] *= rq;
        k[tid] *= rk;
    }
    __syncthreads();

#pragma unroll
    for (unsigned t = 0; t < Iters; ++t) {
        const unsigned j = t * Groups + group;
        f32x4 s0 = state_raw[t][0];
        f32x4 s1 = state_raw[t][1];
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
        hk = kda_bv8_row_sum(hk);
        hq = kda_bv8_row_sum(hq);
        kq = kda_bv8_row_sum(kq);
        const float update = kda_sigmoid(bf2f(beta_raw[h])) * (v[j] - hk);
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            s0[e] += update * k[d0 + e];
            s1[e] += update * k[64u + d0 + e];
        }
        float* column = state + (hd0 + j) * D;
        kda_bv8_store_state(column + d0, s0);
        kda_bv8_store_state(column + 64u + d0, s1);
        if (rlane == 0) out[j] = f2bf(hq + update * kq);
    }
    __syncthreads();

    float oss = 0.0f;
    if (tid < D) {
        const float ov = bf2f(out[tid]);
        oss = ov * ov;
    }
    oss = wave_sum(oss);
    if (lane == 0) sums[tid >> 6] = oss;
    __syncthreads();
    const float inv = rsqrtf((sums[0] + sums[1] + sums[2] + sums[3]) /
                                 (float)D + 1.0e-5f);
    if (tid < D) {
        const float ov = bf2f(out[tid]);
        y[hd0 + tid] = f2bf(ov * inv * norm_w[tid] *
                             kda_sigmoid(bf2f(gate_raw[hd0 + tid])));
    }
}

extern "C" __global__ __launch_bounds__(512, 1)
void kda_norm_control(bf16* y, const bf16* output, const float* norm_w,
                      const bf16* gate_raw, unsigned rows) {
    d_kda_gated_norm(y, output, norm_w, gate_raw, rows, H, D, 1.0e-5f,
                     blockIdx.x, gridDim.x);
}

#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda.h"

namespace {
constexpr unsigned H = 12;
constexpr unsigned D = 128;
constexpr unsigned BV = 8;
constexpr unsigned W = 4;
constexpr unsigned P = H * D;
constexpr unsigned ITEMS = H * D / BV;

__device__ __forceinline__ bf16 conv_one(const bf16* raw, const float* weight,
                                         const float* source, float* target, unsigned channel,
                                         bool write_state) {
    float win[8], tap[8];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        win[j] = j < W ? source[(size_t)channel * W + j] : 0.0f;
        tap[j] = j < W ? weight[(size_t)channel * W + j] : 0.0f;
    }
#pragma unroll
    for (unsigned j = 0; j + 1 < 8; ++j) win[j] = win[j + 1];
    win[W - 1] = bf2f(raw[channel]);
    float value = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) value += win[j] * tap[j];
    if (write_state) {
#pragma unroll
        for (unsigned j = 0; j < W; ++j) target[(size_t)channel * W + j] = win[j];
    }
    return f2bf(act_silu(value));
}
}  // namespace

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void kda_conv3_control(bf16* mix, const bf16* raw, const float* weight, float* conv_state,
                       unsigned layer) {
    const size_t p3 = 3u * P;
    mix += (size_t)layer * p3;
    raw += (size_t)layer * p3;
    weight += (size_t)layer * p3 * W;
    conv_state += (size_t)layer * p3 * W;
    d_kda_conv3(mix, mix + P, mix + 2u * P, raw, raw + P, raw + 2u * P, weight,
                weight + (size_t)P * W, weight + (size_t)2u * P * W, conv_state,
                conv_state + (size_t)P * W, conv_state + (size_t)2u * P * W, 1, P, W, 1,
                blockIdx.x, gridDim.x, 0, nullptr);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void kda_step_control(bf16* output, const bf16* mix, const bf16* gate_raw,
                      const bf16* beta_raw, const float* a_log, const float* dt_bias,
                      float* state, unsigned layer) {
    extern __shared__ float lds[];
    output += (size_t)layer * P;
    mix += (size_t)layer * 3u * P;
    gate_raw += (size_t)layer * P;
    beta_raw += (size_t)layer * H;
    a_log += (size_t)layer * H;
    dt_bias += (size_t)layer * P;
    state += (size_t)layer * H * D * D;
    d_kda_state_step_g(output, mix, mix + P, mix + 2u * P, gate_raw, beta_raw, a_log,
                       dt_bias, PLOW_KDA_GATE_LOWER_BOUND, -5.0f, state, 1, H, D, BV,
                       PLOW_KDA_F_QK_L2NORM, rsqrtf((float)D), blockIdx.x, gridDim.x, lds, 0,
                       nullptr);
}

__device__ __forceinline__ void kda_grid_sync(unsigned* control, unsigned blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned generation = atomicAdd(&control[1], 0u);
        const unsigned arrivals = atomicAdd(&control[0], 1u) + 1u;
        if (arrivals == blocks) {
            atomicExch(&control[0], 0u);
            atomicExch(&control[1], generation + 1u);
        } else {
            while (atomicAdd(&control[1], 0u) == generation) {
            }
        }
        __threadfence();
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void kda_conv_step_barrier(bf16* output, bf16* mix, const bf16* raw, const float* weight,
                           float* conv_state, const bf16* gate_raw, const bf16* beta_raw,
                           const float* a_log, const float* dt_bias, float* state,
                           unsigned* control, unsigned layer) {
    extern __shared__ float lds[];
    const size_t p3 = 3u * P;
    const size_t cs_stride = p3 * W;
    raw += (size_t)layer * p3;
    weight += (size_t)layer * cs_stride;
    conv_state += (size_t)layer * cs_stride;
    mix += (size_t)layer * p3;
    gate_raw += (size_t)layer * P;
    beta_raw += (size_t)layer * H;
    a_log += (size_t)layer * H;
    dt_bias += (size_t)layer * P;
    state += (size_t)layer * H * D * D;
    output += (size_t)layer * P;

    d_kda_conv3(mix, mix + P, mix + 2u * P, raw, raw + P, raw + 2u * P, weight,
                weight + (size_t)P * W, weight + (size_t)2u * P * W, conv_state,
                conv_state + (size_t)P * W, conv_state + (size_t)2u * P * W, 1, P, W, 1,
                blockIdx.x, gridDim.x, 0, nullptr);
    kda_grid_sync(control, gridDim.x);
    d_kda_state_step_g(output, mix, mix + P, mix + 2u * P, gate_raw, beta_raw, a_log, dt_bias,
                       PLOW_KDA_GATE_LOWER_BOUND, -5.0f, state, 1, H, D, BV,
                       PLOW_KDA_F_QK_L2NORM, rsqrtf((float)D), blockIdx.x, gridDim.x, lds, 0,
                       nullptr);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void kda_conv_step_db(bf16* output, bf16* mix_out, const bf16* raw, const float* weight,
                      const float* conv_source, float* conv_target, const bf16* gate_raw,
                      const bf16* beta_raw, const float* a_log, const float* dt_bias,
                      float* state, unsigned layer) {
    extern __shared__ float lds[];
    const unsigned item = blockIdx.x;
    if (item >= ITEMS) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned wave = tid >> 6;
    const unsigned h = item / (D / BV);
    const unsigned tile = item % (D / BV);
    const size_t p3 = 3u * P;
    const size_t cs_stride = p3 * W;
    const size_t state_stride = H * D * D;
    raw += (size_t)layer * p3;
    weight += (size_t)layer * cs_stride;
    conv_source += (size_t)layer * cs_stride;
    conv_target += (size_t)layer * cs_stride;
    mix_out += (size_t)layer * p3;
    gate_raw += (size_t)layer * P;
    beta_raw += (size_t)layer * H;
    a_log += (size_t)layer * H;
    dt_bias += (size_t)layer * P;
    state += (size_t)layer * state_stride;
    output += (size_t)layer * P;

    float* l_q = lds;
    float* l_k = lds + D;
    float* l_g = lds + 2u * D;
    float* l_v = lds + 3u * D + 2u * PLOW_WAVES;
    const unsigned hd0 = h * D;
    float qsum = 0.0f, ksum = 0.0f;
    if (tid < D) {
        const unsigned channel = hd0 + tid;
        const bf16 q = conv_one(raw, weight, conv_source, conv_target, channel, tile == 0);
        const bf16 k = conv_one(raw + P, weight + (size_t)P * W,
                                conv_source + (size_t)P * W,
                                conv_target + (size_t)P * W, channel, tile == 0);
        if (tile == 0) {
            mix_out[channel] = q;
            mix_out[P + channel] = k;
        }
        l_q[tid] = bf2f(q);
        l_k[tid] = bf2f(k);
        const float gate = -5.0f * kda_sigmoid(__expf(a_log[h]) *
                                               (bf2f(gate_raw[channel]) + dt_bias[channel]));
        l_g[tid] = __expf(gate);
        qsum = l_q[tid] * l_q[tid];
        ksum = l_k[tid] * l_k[tid];
    }
    if (tid < BV) {
        const unsigned channel = hd0 + tile * BV + tid;
        const bf16 v = conv_one(raw + 2u * P, weight + (size_t)2u * P * W,
                                conv_source + (size_t)2u * P * W,
                                conv_target + (size_t)2u * P * W, channel, true);
        mix_out[2u * P + channel] = v;
        l_v[tid] = bf2f(v);
    }
    qsum = block_sum(qsum, lds + 3u * D);
    ksum = block_sum(ksum, lds + 3u * D + PLOW_WAVES);
    const float rq = rsqrtf((float)D) / sqrtf(qsum + 1e-6f);
    const float rk = 1.0f / sqrtf(ksum + 1e-6f);
    if (tid < D) {
        l_q[tid] *= rq;
        l_k[tid] *= rk;
    }
    __syncthreads();

    const unsigned j = tile * BV + wave;
    float* column = state + (size_t)h * D * D + (size_t)j * D;
    float sc[2];
    float pk = 0.0f;
#pragma unroll
    for (unsigned r = 0; r < 2; ++r) {
        const unsigned d = r * PLOW_WAVE + lane;
        sc[r] = column[d] * l_g[d];
        pk += sc[r] * l_k[d];
    }
    pk = wave_sum(pk);
    const float update = kda_sigmoid(bf2f(beta_raw[h])) * (l_v[wave] - pk);
    float pq = 0.0f;
#pragma unroll
    for (unsigned r = 0; r < 2; ++r) {
        const unsigned d = r * PLOW_WAVE + lane;
        sc[r] += update * l_k[d];
        column[d] = sc[r];
        pq += sc[r] * l_q[d];
    }
    pq = wave_sum(pq);
    if (lane == 0) output[hd0 + j] = f2bf(pq);
}

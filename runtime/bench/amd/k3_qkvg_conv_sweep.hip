#include "amd_common.h"
#include "op_elementwise.h"
#include "op_gemm.h"
#include "op_kda.h"

struct QkvgConvArgs {
    unsigned long long x;
    unsigned long long wq;
    unsigned long long wk;
    unsigned long long wv;
    unsigned long long wg;
    unsigned long long raw_q;
    unsigned long long raw_k;
    unsigned long long raw_v;
    unsigned long long raw_g;
    unsigned long long mix_q;
    unsigned long long mix_k;
    unsigned long long mix_v;
    unsigned long long conv_wq;
    unsigned long long conv_wk;
    unsigned long long conv_wv;
    unsigned long long conv_sq;
    unsigned long long conv_sk;
    unsigned long long conv_sv;
};

namespace {
constexpr unsigned N = 1536;
constexpr unsigned K = 7168;
constexpr unsigned W = 4;

template <int UN>
__device__ __forceinline__ float qkvg_dot(const bf16* weight, unsigned row,
                                          const bf16* lds) {
    const unsigned lane = threadIdx.x & 63u;
    const unsigned step = PLOW_WAVE * 8u;
    const unsigned nchunk = (K + step - 1u) / step;
    const auto* const wg = as_glob(weight);
    const __amdgpu_buffer_rsrc_t resource = PLOW_GV_RSRC(wg + (size_t)row * K, K);
    float acc = 0.0f;
    for (unsigned c = 0; c < nchunk; c += UN) {
        bf16v8 values[UN];
#pragma unroll
        for (int u = 0; u < UN; ++u)
            values[u] = buf_ld8(resource, ((c + (unsigned)u) * step + lane * 8u) * 2u);
#pragma unroll
        for (int u = 0; u < UN; ++u) {
            const unsigned k = (c + (unsigned)u) * step + lane * 8u;
            const unsigned kx = k < K ? k : 0u;
            acc += dot8(values[u], ld_lds8(lds + kx), 0.0f);
        }
    }
    return wave_sum(acc);
}
}  // namespace

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void k3_qkvg_control(const QkvgConvArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_qkvg((bf16*)a->raw_q, (bf16*)a->raw_k, (bf16*)a->raw_v, (bf16*)a->raw_g,
                 (const bf16*)a->x, (const bf16*)a->wq, (const bf16*)a->wk,
                 (const bf16*)a->wv, (const bf16*)a->wg, 1, N, N, N, N, K, blockIdx.x,
                 gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void k3_conv3_control(const QkvgConvArgs* a) {
    d_kda_conv3((bf16*)a->mix_q, (bf16*)a->mix_k, (bf16*)a->mix_v,
                (const bf16*)a->raw_q, (const bf16*)a->raw_k, (const bf16*)a->raw_v,
                (const float*)a->conv_wq, (const float*)a->conv_wk,
                (const float*)a->conv_wv, (float*)a->conv_sq, (float*)a->conv_sk,
                (float*)a->conv_sv, 1, N, W, 1, blockIdx.x, gridDim.x, 0, nullptr);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void k3_qkvg_conv(const QkvgConvArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    stage_x_lds(lds, (const bf16*)a->x, K);
    __syncthreads();

    const unsigned wave = threadIdx.x >> 6;
    const unsigned qkv_total = 3u * N;
    const unsigned qkv_per = (qkv_total + gridDim.x - 1u) / gridDim.x;
    const unsigned qkv_begin = blockIdx.x * qkv_per;
    const unsigned qkv_end = min(qkv_begin + qkv_per, qkv_total);
    for (unsigned n = qkv_begin + wave; n < qkv_end; n += PLOW_WAVES) {
        float value;
        bf16* output;
        unsigned row;
        if (n < N) {
            row = n;
            value = qkvg_dot<7>((const bf16*)a->wq, row, lds);
            output = (bf16*)a->raw_q;
        } else if (n < 2u * N) {
            row = n - N;
            value = qkvg_dot<7>((const bf16*)a->wk, row, lds);
            output = (bf16*)a->raw_k;
        } else {
            row = n - 2u * N;
            value = qkvg_dot<7>((const bf16*)a->wv, row, lds);
            output = (bf16*)a->raw_v;
        }
        if ((threadIdx.x & 63u) == 0) output[row] = f2bf(value);
    }

    const unsigned g_per = (N + gridDim.x - 1u) / gridDim.x;
    const unsigned g_begin = blockIdx.x * g_per;
    const unsigned g_end = min(g_begin + g_per, N);
    for (unsigned row = g_begin + wave; row < g_end; row += PLOW_WAVES) {
        const float value = qkvg_dot<7>((const bf16*)a->wg, row, lds);
        if ((threadIdx.x & 63u) == 0) ((bf16*)a->raw_g)[row] = f2bf(value);
    }
    __syncthreads();

    d_kda_conv3((bf16*)a->mix_q, (bf16*)a->mix_k, (bf16*)a->mix_v,
                (const bf16*)a->raw_q, (const bf16*)a->raw_k, (const bf16*)a->raw_v,
                (const float*)a->conv_wq, (const float*)a->conv_wk,
                (const float*)a->conv_wv, (float*)a->conv_sq, (float*)a->conv_sk,
                (float*)a->conv_sv, 1, N, W, 1, blockIdx.x, gridDim.x, 0, nullptr);
}

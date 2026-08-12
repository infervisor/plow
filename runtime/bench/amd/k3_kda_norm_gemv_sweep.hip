#include "amd_common.h"
#include "op_gemm.h"
#include "op_kda.h"

struct KdaNormGemvArgs {
    unsigned long long o;
    unsigned long long g;
    unsigned long long gamma;
    unsigned long long y;
    unsigned long long w;
    unsigned long long out;
};

namespace {
constexpr unsigned H = 12;
constexpr unsigned D = 128;
constexpr unsigned K = H * D;
constexpr unsigned N = 7168;
constexpr float EPS = 1.0e-5f;

__device__ __forceinline__ void stage_kda_norm(bf16* lds, const bf16* o, const bf16* g,
                                               const float* gamma, bf16* check) {
    const unsigned lane = threadIdx.x & 63u;
    const unsigned wave = threadIdx.x >> 6;
    for (unsigned h0 = 0; h0 < H; h0 += PLOW_WAVES) {
        const unsigned h = h0 + wave;
        if (h >= H) continue;
        const size_t base = (size_t)h * D;
        float ss = 0.0f;
        for (unsigned d = lane; d < D; d += PLOW_WAVE) {
            const float value = bf2f(o[base + d]);
            ss += value * value;
        }
        const float inv = rsqrtf(wave_sum(ss) / (float)D + EPS);
        for (unsigned d = lane; d < D; d += PLOW_WAVE) {
            const bf16 value = f2bf(bf2f(o[base + d]) * inv * gamma[d] *
                                       kda_sigmoid(bf2f(g[base + d])));
            lds[base + d] = value;
            if (check) check[base + d] = value;
        }
    }
    __syncthreads();
}
}  // namespace

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void k3_kda_norm_control(const KdaNormGemvArgs* a) {
    d_kda_gated_norm((bf16*)a->y, (const bf16*)a->o, (const float*)a->gamma,
                     (const bf16*)a->g, 1, H, D, EPS, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void k3_kda_gemv_control(const KdaNormGemvArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv((bf16*)a->out, (const bf16*)a->y, (const bf16*)a->w, nullptr, nullptr, 1, N, K,
           0, 0.0f, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void k3_kda_norm_check(const KdaNormGemvArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    stage_kda_norm(lds, (const bf16*)a->o, (const bf16*)a->g, (const float*)a->gamma,
                   (bf16*)a->y);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void k3_kda_norm_gemv(const KdaNormGemvArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    stage_kda_norm(lds, (const bf16*)a->o, (const bf16*)a->g, (const float*)a->gamma, nullptr);
    gemv_rows<PLOW_GEMV_MM, true, false>((bf16*)a->out, nullptr, (const bf16*)a->w, nullptr,
                                         nullptr, 1, N, K, blockIdx.x, gridDim.x, lds);
}

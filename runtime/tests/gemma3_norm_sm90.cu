#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define PLOW_NV_GEMMA 1
#define PLOW_NV_GEMMA3 1
#include "../nvidia/op_norm.cuh"

#define CUDA(call) do { auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while (0)
using Bf16 = __nv_bfloat16;

__global__ void norm(Bf16* out, const Bf16* x, const Bf16* g, unsigned rows, unsigned dim) {
    __shared__ float part[32];
    d_rmsnorm(out, x, g, rows, dim, 1e-6f, 0, blockIdx.x, gridDim.x, part);
}

__global__ void head(Bf16* out, const Bf16* x, const Bf16* g, const float* c,
                     const float* s, const int* pos, unsigned rows) {
    d_headnorm_rope<256>(out, x, g, c, s, pos, rows, 1, 1e-6f, 0, 0, 0, 0,
                       blockIdx.x, gridDim.x);
}

__global__ void sandwich(Bf16* out, Bf16* residual, const Bf16* x, const Bf16* g,
                         unsigned rows, unsigned dim) {
    __shared__ float part[32];
    d_norm_residual_norm(out, residual, x, x, g, g, rows, dim, 1e-6f, 1.0f,
                         blockIdx.x, gridDim.x, part);
}

static float rounded(float x) { return __bfloat162float(__float2bfloat16(x)); }

int main() {
    for (unsigned mode : {0u, 1u, 2u}) for (unsigned rows : {1u, 4u, 129u}) {
        bool rope = mode == 1;
        unsigned dim = rope ? 256 : 3840;
        size_t n = size_t(rows) * dim;
        Bf16 *x, *g, *out, *residual;
        float *c, *s;
        int* pos;
        CUDA(cudaMallocManaged(&x, n * sizeof(Bf16)));
        CUDA(cudaMallocManaged(&out, n * sizeof(Bf16)));
        CUDA(cudaMallocManaged(&residual, n * sizeof(Bf16)));
        CUDA(cudaMallocManaged(&g, dim * sizeof(Bf16)));
        CUDA(cudaMallocManaged(&c, rows * 128 * sizeof(float)));
        CUDA(cudaMallocManaged(&s, rows * 128 * sizeof(float)));
        CUDA(cudaMallocManaged(&pos, rows * sizeof(int)));
        for (size_t i = 0; i < n; ++i) x[i] = __float2bfloat16(std::sin(float(i) * .37f));
        for (unsigned j = 0; j < dim; ++j) g[j] = __float2bfloat16(.001f * float(int(j % 71) - 35));
        for (unsigned r = 0; r < rows; ++r) {
            pos[r] = r;
            for (unsigned j = 0; j < 128; ++j) {
                double angle = (double(r) / 8) / std::pow(1e6, 2.0 * j / 256);
                c[r * 128 + j] = std::cos(angle);
                s[r * 128 + j] = std::sin(angle);
            }
        }
        if (mode == 2) sandwich<<<4, 256>>>(out, residual, x, g, rows, dim);
        else if (rope) head<<<4, 256>>>(out, x, g, c, s, pos, rows);
        else norm<<<4, 256>>>(out, x, g, rows, dim);
        CUDA(cudaGetLastError());
        CUDA(cudaDeviceSynchronize());
        double error = 0, energy = 0;
        for (unsigned r = 0; r < rows; ++r) {
            double sum = 0;
            for (unsigned j = 0; j < dim; ++j) {
                float v = __bfloat162float(x[size_t(r) * dim + j]);
                sum += double(v) * v;
            }
            float inv = 1 / std::sqrt(float(sum / dim) + 1e-6f);
            std::vector<float> normalized(dim);
            for (unsigned j = 0; j < dim; ++j)
                normalized[j] = rounded(__bfloat162float(x[size_t(r) * dim + j]) * inv *
                                        (1 + __bfloat162float(g[j])));
            if (mode == 2) {
                double sum_residual = 0;
                for (unsigned j = 0; j < dim; ++j) {
                    normalized[j] = rounded(__bfloat162float(x[size_t(r) * dim + j]) + normalized[j]);
                    sum_residual += double(normalized[j]) * normalized[j];
                }
                float inv_residual = 1 / std::sqrt(float(sum_residual / dim) + 1e-6f);
                for (unsigned j = 0; j < dim; ++j)
                    normalized[j] = rounded(normalized[j] * inv_residual * (1 + __bfloat162float(g[j])));
            }
            for (unsigned j = 0; j < dim; ++j) {
                float expected = normalized[j];
                if (rope) {
                    unsigned pair = j % 128;
                    float cc = rounded(c[r * 128 + pair]), ss = rounded(s[r * 128 + pair]);
                    expected = rounded(j < 128 ? normalized[j] * cc - normalized[j + 128] * ss
                                               : normalized[j] * cc + normalized[j - 128] * ss);
                }
                double delta = __bfloat162float(out[size_t(r) * dim + j]) - expected;
                error += delta * delta;
                energy += double(expected) * expected;
            }
        }
        double rel = std::sqrt(error / energy);
        std::printf("%s rows=%u relative_l2=%.8g\n", mode == 2 ? "sandwich" : rope ? "headnorm_rope" : "rmsnorm", rows, rel);
        if (!(rel < .001)) return 1;
        CUDA(cudaFree(x)); CUDA(cudaFree(out)); CUDA(cudaFree(g));
        CUDA(cudaFree(residual));
        CUDA(cudaFree(c)); CUDA(cudaFree(s)); CUDA(cudaFree(pos));
    }
}

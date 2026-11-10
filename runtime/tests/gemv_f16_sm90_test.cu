/* nvcc -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common -I runtime/nvidia
 *   runtime/tests/gemv_f16_sm90_test.cu -o gemv_f16_sm90_test */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#define PLOW_NV_SPEECH 1
#define PLOW_NV_HOPPER 1
#include "op_speech_f32.cuh"
#include "op_elementwise.cuh"

#define CK(call) do { auto e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void project(float* c, const float* x, const unsigned short* w, const float* bias,
                        const float* res, const float* gamma, const float* beta,
                        unsigned flags, unsigned group, unsigned m, unsigned n, unsigned k) {
    d_gemv_f32out_mode(c, x, w, bias, res, gamma, beta, 1e-5f, flags, group, m, n, k,
                      blockIdx.x, gridDim.x, sp_smem, nullptr);
}

__global__ void normalize(float* out, const float* x, const float* g, const float* b,
                          unsigned rows, unsigned width, unsigned flags) {
    d_layernorm_f32(out, x, g, b, rows, width, flags, 1e-5f, blockIdx.x, gridDim.x, sp_smem);
}

int main(int argc, char** argv) {
    const bool sanitize = argc == 2 && !strcmp(argv[1], "--sanitize");
    CK(cudaFuncSetAttribute(project, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024));
    unsigned rng = 1701;
    auto next = [&] { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                     return (float)((int)(rng % 2001) - 1000) / 997.f; };
    unsigned cases = 0, small = 0, small_differs = 0;
    double small_rel = 0;
    for (unsigned k : {192u, 256u, 448u, 1024u, 3072u, 4096u}) {
        if (sanitize && k != 192u && k != 1024u && k != 3072u) continue;
        const unsigned n = (k == 192 || k == 1024 || k == 3072) ? 3072 : k == 448 ? 192 : 128, max_m = 257;
        std::vector<float> x(max_m * k), bias(max_m * n), residual(max_m * n), gamma(k), beta(k);
        std::vector<__half> w(n * k);
        for (auto& v : x) v = next();
        for (auto& v : w) v = __float2half_rn(next() * .05f);
        for (auto& v : bias) v = next() * .1f;
        for (auto& v : residual) v = next() * .1f;
        for (auto& v : gamma) v = 1.f + next() * .1f;
        for (auto& v : beta) v = next() * .1f;
        float *dx, *db, *dr, *dg, *dt, *dc;
        unsigned short *dw, *dxh;
        CK(cudaMalloc(&dxh, max_m * k * 2));
        CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dw, w.size() * 2));
        CK(cudaMalloc(&db, bias.size() * 4)); CK(cudaMalloc(&dr, residual.size() * 4));
        CK(cudaMalloc(&dg, k * 4)); CK(cudaMalloc(&dt, k * 4)); CK(cudaMalloc(&dc, (max_m * n + 32) * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dr, residual.data(), residual.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dg, gamma.data(), k * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dt, beta.data(), k * 4, cudaMemcpyHostToDevice));
        if (k >= 1024u) {
            float* ln; unsigned short* lnh;
            CK(cudaMalloc(&ln, max_m * k * 4)); CK(cudaMalloc(&lnh, (max_m * k + 32) * 2));
            CK(cudaMemset(lnh, 0x7f, (max_m * k + 32) * 2));
            normalize<<<132, 256>>>(ln, dx, dg, dt, max_m, k, 0);
            normalize<<<132, 256>>>((float*)lnh, dx, dg, dt, max_m, k, 8);
            CK(cudaDeviceSynchronize());
            std::vector<float> lnf(max_m * k); std::vector<unsigned short> lnhost(max_m * k + 32);
            CK(cudaMemcpy(lnf.data(), ln, lnf.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(lnhost.data(), lnh, lnhost.size() * 2, cudaMemcpyDeviceToHost));
            for (unsigned j = 0; j < lnhost.size(); j++) {
                const unsigned short expected = j < lnf.size() ? __half_as_ushort(__float2half_rn(lnf[j])) : 0x7f7f;
                if (lnhost[j] != expected) { fprintf(stderr, "FAIL LayerNorm f16 K=%u i=%u\n", k, j); return 1; }
            }
            CK(cudaFree(ln)); CK(cudaFree(lnh));
        }
        for (unsigned mode = 0; mode < 12; mode++) {
            if (sanitize && mode != 4u && mode != 6u && mode != 8u && mode != 10u) continue;
            const bool ln = mode & 1, gelu = mode & 2;
            const bool half_out = mode & 8, half = mode & 12;
            const unsigned group = gelu ? 5 : 0;
            const unsigned flags = 3 | 1024 | 4 | 8 | (ln && !half ? 256 : 0) | (gelu ? 16 | 512 : 0)
                                   | (half ? 8192 : 0) | (half_out ? 16384 : 0);
            std::vector<double> ref(max_m * n);
            std::vector<__half> xhalf(max_m * k);
            for (unsigned row = 0; row < max_m; row++) {
                double mean = 0, var = 0;
                for (unsigned j = 0; j < k; j++) mean += x[row * k + j];
                mean /= k;
                for (unsigned j = 0; j < k; j++) { double d = x[row * k + j] - mean; var += d * d; }
                const double inv = 1 / sqrt(var / k + 1e-5);
                std::vector<float> input(k);
                for (unsigned j = 0; j < k; j++) {
                    float v = x[row * k + j];
                    if (ln) v = (float)((v - mean) * inv * gamma[j] + beta[j]);
                    input[j] = __half2float(__float2half_rn(v));
                    xhalf[row * k + j] = __float2half_rn(v);
                }
                for (unsigned col = 0; col < n; col++) {
                    double v = 0;
                    for (unsigned j = 0; j < k; j++) v += input[j] * (double)__half2float(w[col * k + j]);
                    v = (float)((float)v + bias[(group ? row / group * n : 0) + col]);
                    v = (float)(v + residual[row * n + col]);
                    if (gelu) v = .5 * v * (1 + erf(v * .70710678118654752));
                    if (half_out) v = __half2float(__float2half_rn((float)v));
                    ref[row * n + col] = v;
                }
            }
            CK(cudaMemcpy(dxh, xhalf.data(), xhalf.size() * 2, cudaMemcpyHostToDevice));
            std::vector<float> widest;
            for (unsigned m : {257u, 256u, 225u, 224u, 201u, 200u, 129u, 128u, 65u, 64u, 33u, 32u, 31u, 20u, 17u, 16u, 8u, 1u}) {
                if (sanitize && m != 257u && m != 224u && m != 200u && m != 33u && m != 16u) continue;
                if (half && m < 16u) continue;
                for (bool alias : {false, true}) {
                    if (half_out && alias) continue;
                    CK(cudaMemset(dc, 0x7f, (max_m * n + 32) * 4));
                    if (alias) CK(cudaMemcpy(dc, dr, m * n * 4, cudaMemcpyDeviceToDevice));
                    project<<<132, 256, 100 * 1024>>>(dc, half ? (const float*)dxh : dx, dw, db, alias ? dc : dr, dg, dt,
                                                    flags, group, m, n, k);
                    CK(cudaDeviceSynchronize());
                    std::vector<float> got(m * n + 32);
                    bool tail_bad = false;
                    if (half_out) {
                        std::vector<unsigned short> raw(got.size());
                        CK(cudaMemcpy(raw.data(), dc, raw.size() * 2, cudaMemcpyDeviceToHost));
                        for (unsigned j = 0; j < m * n; j++) got[j] = __half2float(__ushort_as_half(raw[j]));
                        for (unsigned j = m * n; j < raw.size(); j++) tail_bad |= raw[j] != 0x7f7f;
                    } else CK(cudaMemcpy(got.data(), dc, got.size() * 4, cudaMemcpyDeviceToHost));
                    double error = 0, norm = 0;
                    for (unsigned j = 0; j < m * n; j++) {
                        error += (got[j] - ref[j]) * (got[j] - ref[j]); norm += ref[j] * ref[j];
                    }
                    const double rel = sqrt(error / norm);
                    bool bad = tail_bad || !std::isfinite(rel) || rel > 2e-4;
                    for (unsigned j = m * n; !half_out && j < got.size(); j++) {
                        unsigned bits; memcpy(&bits, &got[j], 4); bad |= bits != 0x7f7f7f7fu;
                    }
                    if (m == max_m && !alias) widest = got;
                    if (m >= 16) {
                        bad |= memcmp(got.data(), widest.data(), m * n * 4) != 0;
                    } else {
                        /* Under 16 rows the split-K MMA arm runs: another accumulation order, so the
                         * same row differs from its wider-capacity bits by rounding. Bounded, not equal. */
                        double d = 0, w2 = 0;
                        for (unsigned j = 0; j < m * n; j++) {
                            d += ((double)got[j] - widest[j]) * ((double)got[j] - widest[j]);
                            w2 += (double)widest[j] * widest[j];
                        }
                        const double across = sqrt(d / w2);
                        small++;
                        small_differs += memcmp(got.data(), widest.data(), m * n * 4) != 0;
                        small_rel = fmax(small_rel, across);
                        bad |= !std::isfinite(across) || across > 4e-4;
                    }
                    if (bad) { fprintf(stderr, "FAIL K=%u M=%u mode=%u alias=%u relL2=%g\n", k, m, mode, alias, rel); return 1; }
                    cases++;
                }
            }
        }
        CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(db)); CK(cudaFree(dr));
        CK(cudaFree(dg)); CK(cudaFree(dt)); CK(cudaFree(dc)); CK(cudaFree(dxh));
    }
    printf("PASS %u projection cases: oracle, row tails, grouped bias, residual alias, GELU, capacity invariance from 16 rows\n", cases);
    printf("below 16 rows vs the same rows at 257: %u of %u cases differ in bits, max relL2 %.3g (bound 4e-4)\n",
           small_differs, small, small_rel);
}

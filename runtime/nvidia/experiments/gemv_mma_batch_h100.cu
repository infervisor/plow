/* gemv_mma_batch_h100.cu — H100 (sm_90a) A/B of the BATCH>1 decode GEMV walk: the shipped
 * dot8 row walk (gemv_rows<MM>, gemv_glu_rows<MM>, gemv_qkv_rows<MM>) against the tensor-core
 * walk in op_gemv_mma.cuh, at Gemma-4-12B decode shapes.
 *
 * Correctness: every arm is compared against an f32 CPU reference (relL2), so the MMA walk's
 * different accumulation order is measured, not assumed. Timing: cudaEvent min-of-N over the
 * interpreter's geometry (132 blocks x 256 threads, one block per SM), reported as ms and as
 * effective weight bandwidth (the number that says whether the walk is bandwidth-bound).
 *
 * Build:
 *   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -DPLOW_NV_GEMV_MMA=1 -DGV_MM_MAX=16 \
 *        -Iruntime/common -Iruntime/nvidia runtime/nvidia/experiments/gemv_mma_batch_h100.cu -o gemv_mma_ab
 * Run: ./gemv_mma_ab [M=16]
 *
 * Build the REFERENCE side with -DPLOW_NV_GEMV_MMA=0: with the define on, gemv_rows<16> itself
 * dispatches to the MMA walk and the A/B compares the MMA walk with itself. The reference kernels
 * instantiate the 16-row rung for every M; the interpreter's walk picks the smallest rung that
 * covers M (gv_mm<2/4/8/16>), so at M<16 the shipped dot8 cost is somewhat lower than reported.
 *
 * RESULTS (H100 SXM5, 132x256, min of 20, 2026-09-17):
 *   M=16: o_proj 221 -> 2156 GB/s (9.8x), down 100 -> 1424 (14.2x), qkv 282 -> 1871 (6.6x),
 *         gate|up 366 -> 2571 (7.0x); relL2 1.7e-3 both sides.
 *   M=8:  4.1-8.8x.  M=4: 3.1-5.8x.  M=2: 2.5-5.2x (16-row reference, see above).
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "sm120_common.cuh"
#include "op_gemm.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const unsigned GRID = 132, BLOCK = 256;

template <int MM>
__global__ void __launch_bounds__(256, 1) k_rows(__nv_bfloat16* C, const __nv_bfloat16* x,
                                                 const __nv_bfloat16* W, unsigned M, unsigned N,
                                                 unsigned K) {
    gemv_rows<MM>(C, x, W, M, N, K, blockIdx.x, gridDim.x);
}
__global__ void __launch_bounds__(256, 1) k_rows_mma(__nv_bfloat16* C, const __nv_bfloat16* x,
                                                     const __nv_bfloat16* W, unsigned M, unsigned N,
                                                     unsigned K) {
    gemv_rows_mma<false>(C, x, W, M, N, K, blockIdx.x, gridDim.x);
}
template <int MM>
__global__ void __launch_bounds__(256, 1) k_glu(__nv_bfloat16* C, const __nv_bfloat16* x,
                                                const __nv_bfloat16* Wg, const __nv_bfloat16* Wu,
                                                unsigned M, unsigned N, unsigned K) {
    gemv_glu_rows<MM>(C, x, Wg, Wu, M, N, K, 0u, blockIdx.x, gridDim.x);
}
__global__ void __launch_bounds__(256, 1) k_glu_mma(__nv_bfloat16* C, const __nv_bfloat16* x,
                                                    const __nv_bfloat16* Wg, const __nv_bfloat16* Wu,
                                                    unsigned M, unsigned N, unsigned K) {
    gemv_glu_rows_mma(C, x, Wg, Wu, M, N, K, 0u, blockIdx.x, gridDim.x);
}
template <int MM>
__global__ void __launch_bounds__(256, 1) k_qkv(__nv_bfloat16* Cq, __nv_bfloat16* Ck, __nv_bfloat16* Cv,
                                                const __nv_bfloat16* x, const __nv_bfloat16* Wq,
                                                const __nv_bfloat16* Wk, const __nv_bfloat16* Wv,
                                                unsigned M, unsigned Nq, unsigned Nk, unsigned Nv,
                                                unsigned K) {
    gemv_qkv_rows<MM>(Cq, Ck, Cv, x, Wq, Wk, Wv, M, Nq, Nk, Nv, K, blockIdx.x, gridDim.x);
}
__global__ void __launch_bounds__(256, 1) k_qkv_mma(__nv_bfloat16* Cq, __nv_bfloat16* Ck, __nv_bfloat16* Cv,
                                                    const __nv_bfloat16* x, const __nv_bfloat16* Wq,
                                                    const __nv_bfloat16* Wk, const __nv_bfloat16* Wv,
                                                    unsigned M, unsigned Nq, unsigned Nk, unsigned Nv,
                                                    unsigned K) {
    gemv_qkv_rows_mma<false>(Cq, Ck, Cv, x, Wq, Wk, Wv, M, Nq, Nk, Nv, K, blockIdx.x, gridDim.x);
}

static void fill(std::vector<__nv_bfloat16>& v, unsigned seed) {
    unsigned s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < v.size(); i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = __float2bfloat16(((float)((s >> 8) & 0xFFFF) / 65536.0f - 0.5f) * 0.1f);
    }
}
static float b2f(__nv_bfloat16 h) { return __bfloat162float(h); }
static float gelu_tanh(float x) {
    const float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}
static double relL2(const std::vector<float>& ref, const std::vector<__nv_bfloat16>& got) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = (double)ref[i] - (double)b2f(got[i]);
        num += d * d; den += (double)ref[i] * ref[i];
    }
    return sqrt(num / (den > 0 ? den : 1));
}
static void cpu_gemv(std::vector<float>& C, const std::vector<__nv_bfloat16>& x,
                     const std::vector<__nv_bfloat16>& W, unsigned M, unsigned N, unsigned K) {
    C.assign((size_t)M * N, 0.f);
    for (unsigned m = 0; m < M; m++)
        for (unsigned n = 0; n < N; n++) {
            double s = 0;
            for (unsigned k = 0; k < K; k++) s += (double)b2f(x[(size_t)m * K + k]) * b2f(W[(size_t)n * K + k]);
            C[(size_t)m * N + n] = (float)s;
        }
}

template <class F>
static float time_ms(F launch, int reps = 20) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    launch(); CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < reps; i++) {
        CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); best = std::min(best, ms);
    }
    return best;
}

template <class T> static T* up(const std::vector<T>& h) {
    T* d; CK(cudaMalloc(&d, h.size() * sizeof(T))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}

int main(int argc, char** argv) {
    const unsigned M = argc > 1 ? (unsigned)atoi(argv[1]) : 16u;
    const unsigned K = 3840;
    printf("M=%u  grid %u x %u\n", M, GRID, BLOCK);

    /* ---- plain: o-proj-like (N=3840,K=4096) and down (N=3840,K=15360) ---- */
    struct Shape { const char* name; unsigned N, K; } shapes[] = {{"o_proj", 3840, 4096}, {"down", 3840, 15360}, {"qkv-as-gemv", 8192, 3840}};
    for (auto& s : shapes) {
        std::vector<__nv_bfloat16> hx((size_t)M * s.K), hW((size_t)s.N * s.K), hC((size_t)M * s.N);
        fill(hx, 1); fill(hW, 2);
        std::vector<float> ref; cpu_gemv(ref, hx, hW, M, s.N, s.K);
        __nv_bfloat16 *dx = up(hx), *dW = up(hW), *dC = up(hC);
        auto run_ref = [&] { k_rows<16><<<GRID, BLOCK>>>(dC, dx, dW, M, s.N, s.K); };
        auto run_mma = [&] { k_rows_mma<<<GRID, BLOCK>>>(dC, dx, dW, M, s.N, s.K); };
        CK(cudaMemset(dC, 0, hC.size() * 2)); run_ref(); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost)); const double e_ref = relL2(ref, hC);
        CK(cudaMemset(dC, 0, hC.size() * 2)); run_mma(); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost)); const double e_mma = relL2(ref, hC);
        const float t_ref = time_ms(run_ref), t_mma = time_ms(run_mma);
        const double gb = (double)s.N * s.K * 2 / 1e9;
        printf("%-12s N=%5u K=%5u | rows<16> %7.3f ms %6.0f GB/s relL2 %.2e | mma %7.3f ms %6.0f GB/s relL2 %.2e | %.2fx\n",
               s.name, s.N, s.K, t_ref, gb / (t_ref / 1e3), e_ref, t_mma, gb / (t_mma / 1e3), e_mma, t_ref / t_mma);
        CK(cudaFree(dx)); CK(cudaFree(dW)); CK(cudaFree(dC));
    }
    /* ---- GLU gate|up (N=15360, K=3840), gelu_tanh ---- */
    {
        const unsigned N = 15360;
        std::vector<__nv_bfloat16> hx((size_t)M * K), hG((size_t)N * K), hU((size_t)N * K), hC((size_t)M * N);
        fill(hx, 3); fill(hG, 4); fill(hU, 5);
        std::vector<float> rg, ru; cpu_gemv(rg, hx, hG, M, N, K); cpu_gemv(ru, hx, hU, M, N, K);
        std::vector<float> ref(rg.size());
        for (size_t i = 0; i < ref.size(); i++) ref[i] = gelu_tanh(rg[i]) * ru[i];
        __nv_bfloat16 *dx = up(hx), *dG = up(hG), *dU = up(hU), *dC = up(hC);
        auto run_ref = [&] { k_glu<16><<<GRID, BLOCK>>>(dC, dx, dG, dU, M, N, K); };
        auto run_mma = [&] { k_glu_mma<<<GRID, BLOCK>>>(dC, dx, dG, dU, M, N, K); };
        run_ref(); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost)); const double e_ref = relL2(ref, hC);
        CK(cudaMemset(dC, 0, hC.size() * 2)); run_mma(); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost)); const double e_mma = relL2(ref, hC);
        const float t_ref = time_ms(run_ref), t_mma = time_ms(run_mma);
        const double gb = 2.0 * N * K * 2 / 1e9;
        printf("%-12s N=%5u K=%5u | glu<16>  %7.3f ms %6.0f GB/s relL2 %.2e | mma %7.3f ms %6.0f GB/s relL2 %.2e | %.2fx\n",
               "gate|up", N, K, t_ref, gb / (t_ref / 1e3), e_ref, t_mma, gb / (t_mma / 1e3), e_mma, t_ref / t_mma);
        CK(cudaFree(dx)); CK(cudaFree(dG)); CK(cudaFree(dU)); CK(cudaFree(dC));
    }
    /* ---- fused QKV (Nq=4096, Nk=Nv=2048, K=3840) ---- */
    {
        const unsigned Nq = 4096, Nk = 2048, Nv = 2048;
        std::vector<__nv_bfloat16> hx((size_t)M * K), hWq((size_t)Nq * K), hWk((size_t)Nk * K), hWv((size_t)Nv * K);
        std::vector<__nv_bfloat16> hCq((size_t)M * Nq), hCk((size_t)M * Nk), hCv((size_t)M * Nv);
        fill(hx, 6); fill(hWq, 7); fill(hWk, 8); fill(hWv, 9);
        std::vector<float> rq, rk, rv; cpu_gemv(rq, hx, hWq, M, Nq, K); cpu_gemv(rk, hx, hWk, M, Nk, K); cpu_gemv(rv, hx, hWv, M, Nv, K);
        __nv_bfloat16 *dx = up(hx), *dWq = up(hWq), *dWk = up(hWk), *dWv = up(hWv), *dCq = up(hCq), *dCk = up(hCk), *dCv = up(hCv);
        auto run_ref = [&] { k_qkv<16><<<GRID, BLOCK>>>(dCq, dCk, dCv, dx, dWq, dWk, dWv, M, Nq, Nk, Nv, K); };
        auto run_mma = [&] { k_qkv_mma<<<GRID, BLOCK>>>(dCq, dCk, dCv, dx, dWq, dWk, dWv, M, Nq, Nk, Nv, K); };
        auto check = [&](double& eq, double& ek, double& ev) {
            CK(cudaMemcpy(hCq.data(), dCq, hCq.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hCk.data(), dCk, hCk.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hCv.data(), dCv, hCv.size() * 2, cudaMemcpyDeviceToHost));
            eq = relL2(rq, hCq); ek = relL2(rk, hCk); ev = relL2(rv, hCv);
        };
        double a, b, c, d, e, f;
        run_ref(); CK(cudaDeviceSynchronize()); check(a, b, c);
        CK(cudaMemset(dCq, 0, hCq.size() * 2)); CK(cudaMemset(dCk, 0, hCk.size() * 2)); CK(cudaMemset(dCv, 0, hCv.size() * 2));
        run_mma(); CK(cudaDeviceSynchronize()); check(d, e, f);
        const float t_ref = time_ms(run_ref), t_mma = time_ms(run_mma);
        const double gb = (double)(Nq + Nk + Nv) * K * 2 / 1e9;
        printf("%-12s N=%5u K=%5u | qkv<16>  %7.3f ms %6.0f GB/s relL2 %.1e/%.1e/%.1e | mma %7.3f ms %6.0f GB/s relL2 %.1e/%.1e/%.1e | %.2fx\n",
               "qkv", Nq + Nk + Nv, K, t_ref, gb / (t_ref / 1e3), a, b, c, t_mma, gb / (t_mma / 1e3), d, e, f, t_ref / t_mma);
    }
    return 0;
}

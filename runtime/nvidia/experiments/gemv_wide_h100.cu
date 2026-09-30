/* gemv_wide_h100.cu — A/B of the wide-batch decode GEMV (op_gemv_wide_sm90.cuh) against the
 * shipped mma.sync walk (d_gemv / d_gemv_glu / d_gemv_qkv at GV_MM_MAX=32, PAIR=1) at the speech
 * decoders' shapes, on the interpreter's geometry (132 x 256, one block per SM).
 *
 * Weights rotate over enough copies to exceed L2, so every launch streams from HBM.
 * Correctness: relL2 against an fp32 GPU reference.
 *
 * Build:
 *   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -DPLOW_NV_HOPPER=1 \
 *        -DPLOW_NV_GEMV_MMA=1 -DPLOW_NV_GEMV_MMA_PAIR=1 -DGV_MM_MAX=32 \
 *        -Iruntime/common -Iruntime/nvidia runtime/nvidia/experiments/gemv_wide_h100.cu -o gemv_wide
 * Run: ./gemv_wide [M ...] */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "sm120_common.cuh"
#include "op_gemm.cuh"
#include "op_gemv_wide_sm90.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const unsigned GRID = 132, BLOCK = 256;

__global__ void __launch_bounds__(256, 1) k_base(int mode, __nv_bfloat16* C0, __nv_bfloat16* C1,
    __nv_bfloat16* C2, const __nv_bfloat16* x, const __nv_bfloat16* W0, const __nv_bfloat16* W1,
    const __nv_bfloat16* W2, unsigned M, unsigned N0, unsigned N1, unsigned N2, unsigned K) {
    if (mode == GW_PLAIN) d_gemv(C0, x, W0, M, N0, K, blockIdx.x, gridDim.x);
    else if (mode == GW_GLU) d_gemv_glu(C0, x, W0, W1, M, N0, K, 1u, blockIdx.x, gridDim.x);
    else d_gemv_qkv(C0, C1, C2, x, W0, W1, W2, M, N0, N1, N2, K, blockIdx.x, gridDim.x);
}
template <int MODE>
__global__ void __launch_bounds__(256, 1) k_wide(GwArgs a) {
    extern __shared__ float arena[];
    if (!d_gemv_wide<MODE>(a, blockIdx.x, gridDim.x, arena)) __trap();
}
__global__ void k_ref(float* C, const __nv_bfloat16* x, const __nv_bfloat16* W, unsigned M,
                      unsigned N, unsigned K) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * N) return;
    const unsigned m = i / N, n = i % N;
    float s = 0.f;
    for (unsigned k = 0; k < K; k++) s += __bfloat162float(x[(size_t)m * K + k]) * __bfloat162float(W[(size_t)n * K + k]);
    C[i] = s;
}
__global__ void k_fill(__nv_bfloat16* p, size_t n, unsigned seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned s = (unsigned)(i * 2654435761u) ^ seed;
        s ^= s >> 13; s *= 0x5bd1e995u; s ^= s >> 15;
        p[i] = __float2bfloat16(((float)(s & 0xFFFF) / 65536.0f - 0.5f) * 0.1f);
    }
}

static float gelu_silu(float g, int act) { (void)act; return g / (1.0f + expf(-g)); }

template <int MODE> static void set_smem() {
    CK(cudaFuncSetAttribute(k_wide<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gw_arena_bytes()));
}
template <int MODE> static void launch_wide(unsigned, const GwArgs& a) {
    k_wide<MODE><<<GRID, BLOCK, gw_arena_bytes()>>>(a);
}

struct Shape { const char* name; int mode; unsigned N0, N1, N2, K; };

int main(int argc, char** argv) {
    std::vector<unsigned> Ms;
    for (int i = 1; i < argc; i++) Ms.push_back(atoi(argv[i]));
    if (Ms.empty()) Ms = {64, 128};
    set_smem<GW_PLAIN>(); set_smem<GW_GLU>(); set_smem<GW_QKV>();
    const Shape shapes[] = {
        {"veena.qkv", GW_QKV, 3072, 1024, 1024, 3072}, {"veena.o", GW_PLAIN, 3072, 0, 0, 3072},
        {"veena.glu", GW_GLU, 8192, 0, 0, 3072},        {"veena.down", GW_PLAIN, 3072, 0, 0, 8192},
        {"veena.lm", GW_PLAIN, 156951, 0, 0, 3072},
        {"qwen.qkv", GW_QKV, 2048, 1024, 1024, 2048},   {"qwen.o", GW_PLAIN, 2048, 0, 0, 2048},
        {"qwen.glu", GW_GLU, 6144, 0, 0, 2048},         {"qwen.down", GW_PLAIN, 2048, 0, 0, 6144},
        {"qwen.lm", GW_PLAIN, 151936, 0, 0, 2048},
        {"cbx.qkv", GW_QKV, 1024, 1024, 1024, 1024},    {"cbx.o", GW_PLAIN, 1024, 0, 0, 1024},
        {"cbx.glu", GW_GLU, 4096, 0, 0, 1024},          {"cbx.down", GW_PLAIN, 1024, 0, 0, 4096},
        {"cbx.lm", GW_PLAIN, 8194, 0, 0, 1024},
    };
    for (const Shape& s : shapes) {
        const unsigned nw = s.mode == GW_GLU ? 2 : (s.mode == GW_QKV ? 3 : 1);
        const unsigned Ns[3] = {s.N0, s.mode == GW_GLU ? s.N0 : s.N1, s.N2};
        size_t wbytes = 0;
        for (unsigned i = 0; i < nw; i++) wbytes += (size_t)Ns[i] * s.K * 2;
        const int copies = (int)std::max<size_t>(2, (size_t)(200u << 20) / wbytes + 1);
        std::vector<__nv_bfloat16*> W(copies * 3, nullptr);
        for (int c = 0; c < copies; c++)
            for (unsigned i = 0; i < nw; i++) {
                CK(cudaMalloc(&W[c * 3 + i], (size_t)Ns[i] * s.K * 2));
                k_fill<<<1024, 256>>>(W[c * 3 + i], (size_t)Ns[i] * s.K, 17u * i + 3u);
            }
        const unsigned MMAX = 128;
        __nv_bfloat16 *x, *C[3], *Cb[3];
        CK(cudaMalloc(&x, (size_t)MMAX * s.K * 2));
        k_fill<<<256, 256>>>(x, (size_t)MMAX * s.K, 99u);
        for (unsigned i = 0; i < 3; i++) {
            const size_t n = (size_t)MMAX * (Ns[i] ? Ns[i] : 1);
            CK(cudaMalloc(&C[i], n * 2)); CK(cudaMalloc(&Cb[i], n * 2));
        }
        float* ref;
        const unsigned NR = std::max(Ns[0], std::max(Ns[1], Ns[2]));
        CK(cudaMalloc(&ref, (size_t)MMAX * NR * 4 * 2));
        for (unsigned M : Ms) {
            const unsigned mp = M <= 64 ? 64 : 128;
            auto args = [&](int c) {
                GwArgs a{};
                a.x = x; a.M = M; a.K = s.K; a.act = 1u;
                for (unsigned i = 0; i < 3; i++) { a.W[i] = W[c * 3 + i]; a.C[i] = C[i]; a.N[i] = Ns[i]; }
                if (s.mode == GW_GLU) a.N[1] = 0;
                return a;
            };
            auto run_wide = [&](int c) {
                GwArgs a = args(c);
                if (s.mode == GW_PLAIN) launch_wide<GW_PLAIN>(mp, a);
                else if (s.mode == GW_GLU) launch_wide<GW_GLU>(mp, a);
                else launch_wide<GW_QKV>(mp, a);
            };
            auto run_base = [&](int c) {
                k_base<<<GRID, BLOCK>>>(s.mode, Cb[0], Cb[1], Cb[2], x, W[c * 3], W[c * 3 + 1],
                                        W[c * 3 + 2], M, Ns[0], Ns[1], Ns[2], s.K);
            };
            /* correctness on copy 0 */
            run_wide(0); run_base(0);
            CK(cudaDeviceSynchronize());
            double worst = 0, worstb = 0;
            const unsigned nout = s.mode == GW_GLU ? 1 : nw;
            for (unsigned i = 0; i < nout; i++) {
                const unsigned N = Ns[i];
                std::vector<float> r((size_t)M * N);
                if (s.mode == GW_GLU) {
                    std::vector<float> rg((size_t)M * N), ru((size_t)M * N);
                    k_ref<<<(unsigned)(((size_t)M * N + 255) / 256), 256>>>(ref, x, W[0], M, N, s.K);
                    CK(cudaMemcpy(rg.data(), ref, rg.size() * 4, cudaMemcpyDeviceToHost));
                    k_ref<<<(unsigned)(((size_t)M * N + 255) / 256), 256>>>(ref, x, W[1], M, N, s.K);
                    CK(cudaMemcpy(ru.data(), ref, ru.size() * 4, cudaMemcpyDeviceToHost));
                    for (size_t j = 0; j < r.size(); j++) r[j] = gelu_silu(rg[j], 1) * ru[j];
                } else {
                    k_ref<<<(unsigned)(((size_t)M * N + 255) / 256), 256>>>(ref, x, W[i], M, N, s.K);
                    CK(cudaMemcpy(r.data(), ref, r.size() * 4, cudaMemcpyDeviceToHost));
                }
                std::vector<__nv_bfloat16> g((size_t)M * N), gb((size_t)M * N);
                CK(cudaMemcpy(g.data(), C[i], g.size() * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(gb.data(), Cb[i], gb.size() * 2, cudaMemcpyDeviceToHost));
                double num = 0, numb = 0, den = 0;
                for (size_t j = 0; j < r.size(); j++) {
                    const double d = r[j] - __bfloat162float(g[j]), db = r[j] - __bfloat162float(gb[j]);
                    num += d * d; numb += db * db; den += (double)r[j] * r[j];
                }
                worst = std::max(worst, sqrt(num / den)); worstb = std::max(worstb, sqrt(numb / den));
            }
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            auto time = [&](auto&& fn) {
                for (int i = 0; i < 3; i++) fn(i % copies);
                float best = 1e9f;
                for (int rep = 0; rep < 5; rep++) {
                    CK(cudaEventRecord(e0));
                    for (int i = 0; i < 20; i++) fn(i % copies);
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                    best = std::min(best, ms * 1000.f / 20.f);
                }
                return best;
            };
            const float tw = time(run_wide), tb = time(run_base);
            const float roof = std::max((double)wbytes / 3.15e12, 2.0 * M * wbytes / 2 / 803e12) * 1e6;
            printf("%-11s M=%3u  base %8.1f us (%3.0f%%)  wide %8.1f us (%3.0f%%)  x%.2f  relL2 base %.1e wide %.1e\n",
                   s.name, M, tb, 100 * roof / tb, tw, 100 * roof / tw, tb / tw, worstb, worst);
            fflush(stdout);
        }
        for (auto p : W) if (p) cudaFree(p);
        cudaFree(x); cudaFree(ref);
        for (unsigned i = 0; i < 3; i++) { cudaFree(C[i]); cudaFree(Cb[i]); }
    }
    return 0;
}

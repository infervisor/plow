/* flashpre_hd128_sm120.cu — oracle + timing for the hd128 causal flash prefill on sm_120/121:
 * d_flash_prefill<128,64,64> (the shipped PIPE body) vs d_flash_pf128 (op_flash_pf128.cuh, FA2
 * form) at the Nemotron-H attention geometry (32 q-heads / 2 kv-heads, NoPE), one chunk of T query
 * rows after a prefix of P cached rows. Checks both against an f64 host reference on sampled
 * (row, head) pairs and against each other; prints ms, TFLOP/s and % of the BF16 ceiling.
 *
 * Build (inside nix develop):
 *   nvcc -std=c++17 -arch=sm_121a -O3 -I runtime/common -I runtime/nvidia -include cstdint \
 *     runtime/tests/flashpre_hd128_sm120.cu -o flashpre_hd128
 * Run: flashpre_hd128 [T=8192] [priors=0,8192,32768,131072,253952] [iters=3] [bf16 TF=93.9]
 *        [arms=old,new] [kv_stride=0 (= P + T)]
 * Arms: old = d_flash_prefill, new = d_flash_pf128 on bf16 K/V, new8 = d_flash_pf128<0, true> on
 * the E4M3 cache (K/V are rounded through e4m3 first, so every arm sees the same values and new8
 * must equal new bitwise); v1 / v2 = the E4M3 staging variants VAR 1 / 2 (bitwise equal to new too).
 * Exit 1 when a rel-L2 exceeds 2e-2 (reference), two kernels differ by more than 1e-2, or new8
 * differs from new in any bit. */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#include "sm120_common.cuh" /* pulls op_attention.cuh (d_flash_prefill) */
#include "op_flash_pf128.cuh"

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERROR %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(2); } } while (0)

static const unsigned NH = 32, NKV = 2, HD = 128;

__global__ void __launch_bounds__(256, 1)
k_old(float* Opart, float* ml, const bf16* Q, const bf16* K, const bf16* V, bf16* O, unsigned sq,
      unsigned skv, unsigned q0, unsigned stride, float scale) {
    extern __shared__ float arena[];
    d_flash_prefill<128, 64, 64>(Opart, ml, Q, K, V, O, sq, skv, NH, NKV, q0, 0, 1, stride,
                                 0xFFFFFFFFu, scale, blockIdx.x, gridDim.x, arena, nullptr);
}
__global__ void __launch_bounds__(256, 1)
k_new(const bf16* Q, const bf16* K, const bf16* V, bf16* O, unsigned sq, unsigned skv, unsigned q0,
      unsigned stride, float scale) {
    extern __shared__ float arena[];
    d_flash_pf128<>(nullptr, Q, K, V, O, sq, skv, NH, NKV, q0, stride, 0xFFFFFFFFu, scale,
                    blockIdx.x, gridDim.x, arena);
}
template <int VAR>
__global__ void __launch_bounds__(256, 1)
k_new8(const bf16* Q, const bf16* K, const bf16* V, bf16* O, unsigned sq, unsigned skv, unsigned q0,
       unsigned stride, float scale) {
    extern __shared__ float arena[];
    d_flash_pf128<VAR, true>(nullptr, Q, K, V, O, sq, skv, NH, NKV, q0, stride, 0xFFFFFFFFu, scale,
                             blockIdx.x, gridDim.x, arena);
}
/* x -> sat_e4m3(x) (the cache writer's saturating RNE store): bytes to q8, the exact bf16 back to x. */
__global__ void k_e4m3(bf16* x, unsigned char* q8, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const __nv_fp8_storage_t b = __nv_cvt_float_to_fp8(__bfloat162float(x[i]), __NV_SATFINITE, __NV_E4M3);
        q8[i] = b;
        const __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
        x[i] = __float2bfloat16(__half2float(*reinterpret_cast<const __half*>(&h)));
    }
}
__global__ void k_fill(bf16* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        p[i] = __float2bfloat16(amp * ((float)(int32_t)x / 2147483648.0f));
    }
}

static float b2f(bf16 v) { uint16_t u; memcpy(&u, &v, 2); uint32_t w = (uint32_t)u << 16; float f; memcpy(&f, &w, 4); return f; }

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? (unsigned)atoi(argv[1]) : 8192u;
    std::vector<unsigned> priors;
    {
        std::string s = argc > 2 ? argv[2] : "0,8192,32768,131072,253952";
        size_t a = 0;
        while (a < s.size()) { size_t b = s.find(',', a); if (b == std::string::npos) b = s.size();
            priors.push_back((unsigned)atoi(s.substr(a, b - a).c_str())); a = b + 1; }
    }
    const int iters = argc > 3 ? atoi(argv[3]) : 3;
    const double tf_ceil = argc > 4 ? atof(argv[4]) : 93.9;
    const std::string arms = std::string(",") + (argc > 5 ? argv[5] : "old,new") + ",";
    const unsigned stride_arg = argc > 6 ? (unsigned)atoi(argv[6]) : 0u;
    constexpr int NA = 5;
    const char* arm_name[NA] = {"old", "new", "new8", "v1", "v2"};
    bool run[NA];
    for (int a = 0; a < NA; a++) run[a] = arms.find(std::string(",") + arm_name[a] + ",") != std::string::npos;
    const bool kv8 = run[2] || run[3] || run[4];
    int dev = 0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    const int grid = p.multiProcessorCount;
    printf("# %s, %d SMs; T=%u NH=%u NKV=%u HD=%u\n", p.name, grid, T, NH, NKV, HD);
    const size_t smem_old = (size_t)FA_PRE_SMEM_FLOATS(128, 64, 64) * 4, smem_new = FP2_SMEM_BYTES;
    CK(cudaFuncSetAttribute(k_old, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_old));
    CK(cudaFuncSetAttribute(k_new, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_new));
    const size_t smem8 = FP2_SMEM8_BYTES;
    CK(cudaFuncSetAttribute(k_new8<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8));
    CK(cudaFuncSetAttribute(k_new8<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8));
    CK(cudaFuncSetAttribute(k_new8<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8));
    const float scale = 1.f / sqrtf((float)HD);
    int fail = 0;
    for (unsigned P : priors) {
        const unsigned skv = P + T, stride = stride_arg > skv ? stride_arg : skv;
        bf16 *Q, *K, *V, *O[NA]; float *Op, *ml;
        unsigned char *K8 = nullptr, *V8 = nullptr;
        const size_t nq = (size_t)T * NH * HD, nk = (size_t)NKV * stride * HD;
        CK(cudaMalloc(&Q, nq * 2)); CK(cudaMalloc(&K, nk * 2)); CK(cudaMalloc(&V, nk * 2));
        for (int a = 0; a < NA; a++) { CK(cudaMalloc(&O[a], nq * 2)); CK(cudaMemset(O[a], 0, nq * 2)); }
        CK(cudaMalloc(&Op, nq * 4)); CK(cudaMalloc(&ml, (size_t)T * NH * 8));
        k_fill<<<1024, 256>>>(Q, nq, 11u + P, 1.0f);
        k_fill<<<1024, 256>>>(K, nk, 23u + P, 1.0f);
        k_fill<<<1024, 256>>>(V, nk, 37u + P, 1.0f);
        if (kv8) {
            CK(cudaMalloc(&K8, nk)); CK(cudaMalloc(&V8, nk));
            k_e4m3<<<1024, 256>>>(K, K8, nk);
            k_e4m3<<<1024, 256>>>(V, V8, nk);
        }
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        float ms[NA] = {};
        for (int which = 0; which < NA; which++) {
            if (!run[which]) continue;
            std::vector<float> t;
            for (int it = 0; it < iters + 1; it++) {
                CK(cudaEventRecord(e0));
                if (which == 0) k_old<<<grid, 256, smem_old>>>(Op, ml, Q, K, V, O[0], T, skv, P, stride, scale);
                else if (which == 1) k_new<<<grid, 256, smem_new>>>(Q, K, V, O[1], T, skv, P, stride, scale);
                else if (which == 2) k_new8<0><<<grid, 256, smem8>>>(Q, (const bf16*)K8, (const bf16*)V8, O[2], T, skv, P, stride, scale);
                else if (which == 3) k_new8<1><<<grid, 256, smem8>>>(Q, (const bf16*)K8, (const bf16*)V8, O[3], T, skv, P, stride, scale);
                else k_new8<2><<<grid, 256, smem8>>>(Q, (const bf16*)K8, (const bf16*)V8, O[4], T, skv, P, stride, scale);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
                float x; CK(cudaEventElapsedTime(&x, e0, e1)); if (it) t.push_back(x);
            }
            std::sort(t.begin(), t.end());
            ms[which] = t[t.size() / 2];
        }
        std::vector<bf16> hq(nq), h[NA];
        CK(cudaMemcpy(hq.data(), Q, nq * 2, cudaMemcpyDeviceToHost));
        for (int a = 0; a < NA; a++)
            if (run[a]) { h[a].resize(nq); CK(cudaMemcpy(h[a].data(), O[a], nq * 2, cudaMemcpyDeviceToHost)); }
        /* f64 reference on sampled rows (first, middle, last) x two heads per KV group */
        const unsigned rows[] = {0, 1, 63, 64, 127, T / 2, T - 65, T - 1};
        const unsigned heads[] = {0, 15, 16, 31};
        std::vector<bf16> hk((size_t)skv * HD), hv((size_t)skv * HD);
        double rd[NA] = {}, rn = 0;
        for (unsigned hh : heads) {
            const unsigned kh = hh / (NH / NKV);
            CK(cudaMemcpy(hk.data(), K + (size_t)kh * stride * HD, (size_t)skv * HD * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hv.data(), V + (size_t)kh * stride * HD, (size_t)skv * HD * 2, cudaMemcpyDeviceToHost));
            for (unsigned r : rows) {
                if (r >= T) continue;
                const unsigned n = P + r + 1;
                std::vector<double> s(n), acc(HD, 0.0);
                double mx = -1e300;
                for (unsigned k = 0; k < n; k++) {
                    double d = 0;
                    for (unsigned c = 0; c < HD; c++) d += (double)b2f(hq[((size_t)r * NH + hh) * HD + c]) * b2f(hk[(size_t)k * HD + c]);
                    s[k] = d * scale; mx = s[k] > mx ? s[k] : mx;
                }
                double l = 0;
                for (unsigned k = 0; k < n; k++) { s[k] = exp(s[k] - mx); l += s[k];
                    for (unsigned c = 0; c < HD; c++) acc[c] += s[k] * b2f(hv[(size_t)k * HD + c]); }
                for (unsigned c = 0; c < HD; c++) {
                    const double ref = acc[c] / l;
                    rn += ref * ref;
                    for (int a = 0; a < NA; a++)
                        if (run[a]) { const double x = b2f(h[a][((size_t)r * NH + hh) * HD + c]); rd[a] += (x - ref) * (x - ref); }
                }
            }
        }
        const double flop = 4.0 * NH * HD * ((double)T * P + (double)T * (T + 1) / 2);
        printf("P=%-7u", P);
        int base = 0; /* every other arm is compared with the first one run, `new` when present */
        while (!run[base]) base++;
        if (run[1]) base = 1;
        for (int a = 0; a < NA; a++) {
            if (!run[a]) continue;
            const double rel = sqrt(rd[a] / rn);
            printf(" | %s %9.3f ms %5.1f TF/s %3.0f%% rel %.2e", arm_name[a], ms[a], flop / ms[a] / 1e9,
                   100 * flop / ms[a] / 1e9 / tf_ceil, rel);
            if (rel > 2e-2) fail = 1;
            if (a == base) continue;
            double dd = 0, dn = 0;
            size_t diff = 0;
            for (size_t i = 0; i < nq; i++) {
                const double x = b2f(h[base][i]), y = b2f(h[a][i]);
                dd += (x - y) * (x - y); dn += x * x;
                diff += memcmp(&h[base][i], &h[a][i], 2) != 0;
            }
            printf(" vs-%s %.2e (%zu bf16 differ)", arm_name[base], sqrt(dd / (dn + 1e-30)), diff);
            if (sqrt(dd / (dn + 1e-30)) > 1e-2 || (a >= 2 && base == 1 && diff)) fail = 1;
        }
        printf("\n");
        fflush(stdout);
        cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(Op); cudaFree(ml);
        for (int a = 0; a < NA; a++) cudaFree(O[a]);
        if (K8) { cudaFree(K8); cudaFree(V8); }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    printf(fail ? "FAIL\n" : "PASS\n");
    return fail;
}

/* flashpre_hd256_sm120.cu — oracle + timing for the hd256 flash prefill on sm_120/121:
 * d_flash_prefill<256,64,32> (the shipped PIPE body) vs d_flash_pf256 (op_flash_pf256.cuh, FA2
 * split-D form), one chunk of T query rows after a prefix of P cached rows. Default geometry is
 * Qwen3.8-27B full attention (24 q-heads / 4 kv-heads, causal); `nh`, `nkv` and `window` cover the
 * Gemma-4 sliding layers. nsplit > 1 runs FlashMerge (d_flash_merge<256>) after the prefill.
 * Checks every arm against an f64 host reference on sampled (row, head) pairs and against `old`;
 * prints ms, TFLOP/s and % of the BF16 ceiling. `sm1=s` instead times slice s of the 48-block grid
 * alone on one SM (the ubench slice) for every arm and skips the checks.
 *
 * Build (inside nix develop; L = a dir holding a libcuda.so.1 symlink to the driver's own, as
 * runtime/ubench/run_ubench_cu.sh links it: nix glibc does not find the host driver otherwise):
 *   nvcc -std=c++17 -arch=sm_121a -O3 -I runtime/common -I runtime/nvidia -include cstdint \
 *     -L $L -lcuda -Xlinker -rpath=$L runtime/tests/flashpre_hd256_sm120.cu -o flashpre_hd256
 * Run: flashpre_hd256 [T=8192] [priors=0,8192,24576] [iters=3] [bf16 TF=126.7] [arms=old,new]
 *        [nsplit=1] [nh=24] [nkv=4] [window=0] [sm1=-1]
 * Exit 1 when a rel-L2 exceeds 2e-2 (reference) or two kernels differ by more than 1e-2. */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#include "sm120_common.cuh" /* pulls op_attention.cuh (d_flash_prefill, d_flash_merge) */
#include "op_flash_pf256.cuh"

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERROR %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(2); } } while (0)

static const unsigned HD = 256;

struct Geo { unsigned sq, skv, nh, nkv, q0, window, ns, stride, slice, nblk; float scale; };

__global__ void __launch_bounds__(256, 1)
k_old(float* Opart, float* ml, const bf16* Q, const bf16* K, const bf16* V, bf16* O, Geo g) {
    extern __shared__ float arena[];
    const unsigned s = g.nblk ? g.slice : blockIdx.x, n = g.nblk ? g.nblk : gridDim.x;
    d_flash_prefill<256, 64, 32>(Opart, ml, Q, K, V, O, g.sq, g.skv, g.nh, g.nkv, g.q0, g.window, g.ns,
                                 g.stride, 0xFFFFFFFFu, g.scale, s, n, arena, nullptr);
}
__global__ void __launch_bounds__(256, 1)
k_new(float* Opart, float* ml, const bf16* Q, const bf16* K, const bf16* V, bf16* O, Geo g) {
    extern __shared__ float arena[];
    const unsigned s = g.nblk ? g.slice : blockIdx.x, n = g.nblk ? g.nblk : gridDim.x;
    d_flash_pf256(Opart, ml, Q, K, V, O, g.sq, g.skv, g.nh, g.nkv, g.q0, g.window, g.ns, g.stride,
                  0xFFFFFFFFu, g.scale, s, n, arena);
}
__global__ void __launch_bounds__(256, 1)
k_merge(bf16* O, const float* Opart, const float* ml, unsigned rows, unsigned nh, unsigned ns) {
    d_flash_merge<256>(O, Opart, ml, rows, nh, ns, blockIdx.x, gridDim.x);
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
        std::string s = argc > 2 ? argv[2] : "0,8192,24576";
        size_t a = 0;
        while (a < s.size()) { size_t b = s.find(',', a); if (b == std::string::npos) b = s.size();
            priors.push_back((unsigned)atoi(s.substr(a, b - a).c_str())); a = b + 1; }
    }
    const int iters = argc > 3 ? atoi(argv[3]) : 3;
    const double tf_ceil = argc > 4 ? atof(argv[4]) : 126.7;
    const std::string arms = std::string(",") + (argc > 5 ? argv[5] : "old,new") + ",";
    const unsigned NS = argc > 6 ? (unsigned)atoi(argv[6]) : 1u;
    const unsigned NH = argc > 7 ? (unsigned)atoi(argv[7]) : 24u;
    const unsigned NKV = argc > 8 ? (unsigned)atoi(argv[8]) : 4u;
    const unsigned WIN = argc > 9 ? (unsigned)atoi(argv[9]) : 0u;
    const int sm1 = argc > 10 ? atoi(argv[10]) : -1;
    constexpr int NA = 2;
    const char* arm_name[NA] = {"old", "new"};
    bool run[NA];
    for (int a = 0; a < NA; a++) run[a] = arms.find(std::string(",") + arm_name[a] + ",") != std::string::npos;
    int dev = 0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    const int grid = p.multiProcessorCount;
    printf("# %s, %d SMs; T=%u NH=%u NKV=%u HD=%u nsplit=%u window=%u%s\n", p.name, grid, T, NH, NKV, HD,
           NS, WIN, sm1 >= 0 ? " (one slice on one SM)" : "");
    const size_t smem_old = (size_t)FA_PRE_SMEM_FLOATS(256, 64, 32) * 4, smem_new = F256_SMEM_BYTES;
    CK(cudaFuncSetAttribute(k_old, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_old));
    CK(cudaFuncSetAttribute(k_new, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_new));
    const float scale = 1.f / sqrtf((float)HD);
    int fail = 0;
    for (unsigned P : priors) {
        const unsigned skv = P + T, stride = skv;
        bf16 *Q, *K, *V, *O[NA]; float *Op, *ml;
        const size_t nq = (size_t)T * NH * HD, nk = (size_t)NKV * stride * HD;
        CK(cudaMalloc(&Q, nq * 2)); CK(cudaMalloc(&K, nk * 2)); CK(cudaMalloc(&V, nk * 2));
        for (int a = 0; a < NA; a++) { CK(cudaMalloc(&O[a], nq * 2)); CK(cudaMemset(O[a], 0, nq * 2)); }
        CK(cudaMalloc(&Op, nq * 4 * NS)); CK(cudaMalloc(&ml, (size_t)T * NH * 8 * NS));
        k_fill<<<1024, 256>>>(Q, nq, 11u + P, 1.0f);
        k_fill<<<1024, 256>>>(K, nk, 23u + P, 1.0f);
        k_fill<<<1024, 256>>>(V, nk, 37u + P, 1.0f);
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        float ms[NA] = {};
        for (int which = 0; which < NA; which++) {
            if (!run[which]) continue;
            Geo g{T, skv, NH, NKV, P, WIN, NS, stride, sm1 >= 0 ? (unsigned)sm1 : 0u, sm1 >= 0 ? (unsigned)grid : 0u, scale};
            const int launch = sm1 >= 0 ? 1 : grid;
            std::vector<float> t;
            for (int it = 0; it < iters + 1; it++) {
                CK(cudaEventRecord(e0));
                if (which == 0) k_old<<<launch, 256, smem_old>>>(Op, ml, Q, K, V, O[0], g);
                else k_new<<<launch, 256, smem_new>>>(Op, ml, Q, K, V, O[1], g);
                if (NS > 1 && sm1 < 0) k_merge<<<grid, 256>>>(O[which], Op, ml, T, NH, NS);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
                float x; CK(cudaEventElapsedTime(&x, e0, e1)); if (it) t.push_back(x);
            }
            std::sort(t.begin(), t.end());
            ms[which] = t[t.size() / 2];
        }
        double flop = 0; /* exact causal / window key count */
        for (unsigned r = 0; r < T; r++) {
            const unsigned a = P + r, lo = WIN && a + 1 > WIN ? a + 1 - WIN : 0;
            flop += 4.0 * NH * HD * (a + 1 - lo);
        }
        if (sm1 >= 0) {
            printf("P=%-7u slice %d", P, sm1);
            for (int a = 0; a < NA; a++)
                if (run[a]) printf(" | %s %9.3f ms", arm_name[a], ms[a]);
            printf("\n");
        } else {
            std::vector<bf16> hq(nq), h[NA];
            CK(cudaMemcpy(hq.data(), Q, nq * 2, cudaMemcpyDeviceToHost));
            for (int a = 0; a < NA; a++)
                if (run[a]) { h[a].resize(nq); CK(cudaMemcpy(h[a].data(), O[a], nq * 2, cudaMemcpyDeviceToHost)); }
            const unsigned rows[] = {0, 1, 15, 16, 63, 64, 127, T / 2, T - 65, T - 1};
            std::vector<unsigned> heads = {0, NH / NKV - 1, NH / NKV, NH - 1};
            std::vector<bf16> hk((size_t)skv * HD), hv((size_t)skv * HD);
            double rd[NA] = {}, rn = 0;
            for (unsigned hh : heads) {
                const unsigned kh = hh / (NH / NKV);
                CK(cudaMemcpy(hk.data(), K + (size_t)kh * stride * HD, (size_t)skv * HD * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hv.data(), V + (size_t)kh * stride * HD, (size_t)skv * HD * 2, cudaMemcpyDeviceToHost));
                for (unsigned r : rows) {
                    if (r >= T) continue;
                    const unsigned a = P + r, lo = WIN && a + 1 > WIN ? a + 1 - WIN : 0;
                    std::vector<double> s(a + 1), acc(HD, 0.0);
                    double mx = -1e300;
                    for (unsigned k = lo; k <= a; k++) {
                        double d = 0;
                        for (unsigned c = 0; c < HD; c++) d += (double)b2f(hq[((size_t)r * NH + hh) * HD + c]) * b2f(hk[(size_t)k * HD + c]);
                        s[k] = d * scale; mx = s[k] > mx ? s[k] : mx;
                    }
                    double l = 0;
                    for (unsigned k = lo; k <= a; k++) { s[k] = exp(s[k] - mx); l += s[k];
                        for (unsigned c = 0; c < HD; c++) acc[c] += s[k] * b2f(hv[(size_t)k * HD + c]); }
                    for (unsigned c = 0; c < HD; c++) {
                        const double ref = acc[c] / l;
                        rn += ref * ref;
                        for (int x = 0; x < NA; x++)
                            if (run[x]) { const double y = b2f(h[x][((size_t)r * NH + hh) * HD + c]); rd[x] += (y - ref) * (y - ref); }
                    }
                }
            }
            printf("P=%-7u", P);
            int base = 0;
            while (base < NA && !run[base]) base++;
            for (int a = 0; a < NA; a++) {
                if (!run[a]) continue;
                const double rel = sqrt(rd[a] / rn);
                printf(" | %s %9.3f ms %5.1f TF/s %3.0f%% rel %.2e", arm_name[a], ms[a], flop / ms[a] / 1e9,
                       100 * flop / ms[a] / 1e9 / tf_ceil, rel);
                if (!(rel <= 2e-2)) fail = 1;
                if (a == base) continue;
                double dd = 0, dn = 0;
                size_t diff = 0;
                for (size_t i = 0; i < nq; i++) {
                    const double x = b2f(h[base][i]), y = b2f(h[a][i]);
                    dd += (x - y) * (x - y); dn += x * x;
                    diff += memcmp(&h[base][i], &h[a][i], 2) != 0;
                }
                printf(" vs-%s %.2e (%zu bf16 differ)", arm_name[base], sqrt(dd / (dn + 1e-30)), diff);
                if (!(sqrt(dd / (dn + 1e-30)) <= 1e-2)) fail = 1;
            }
            printf("\n");
        }
        fflush(stdout);
        cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(Op); cudaFree(ml);
        for (int a = 0; a < NA; a++) cudaFree(O[a]);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    printf(fail ? "FAIL\n" : "PASS\n");
    return fail;
}

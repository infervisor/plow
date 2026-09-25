/* index_select_pf_v2_gfx950.hip — DSA prefill top-k select (op 118): d_index_select_pf_v2
 * (PLOW_DSA_SELECT_V2) vs the shipped d_index_select_pf<true>, same interpreter geometry (256
 * workgroups grid-strided over rows). Sets must be identical per row, both must equal a CPU
 * exact top-k on sampled rows, and rows >= n_tok must stay untouched (the packet's scratch
 * has T rows; a ragged chunk writes only its live rows). Prints per-call timing.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/index_select_pf_v2_gfx950.hip -o isel_v2
 */
#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr unsigned TOPK = 2048, STRIDE = 16384, NWG = 256;

extern "C" __global__ __launch_bounds__(512, 1) void k_old(int* idx, const float* sc, const int* len,
                                                          unsigned n_tok) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned hist[SEL_NB + 4];
    d_index_select_pf<true>(idx, sc, len, n_tok, TOPK, STRIDE, blockIdx.x, gridDim.x, hist, hist + SEL_NB);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_new(int* idx, const float* sc, const int* len,
                                                          unsigned n_tok) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned hist[4096 + 16];
    d_index_select_pf_v2(idx, sc, len, n_tok, TOPK, STRIDE, blockIdx.x, gridDim.x, hist, hist + 4096);
#endif
}

static unsigned long long rng = 0x9E3779B97F4A7C15ull;
static unsigned nxt() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)(rng >> 11); }
static float gauss() { float a = 0; for (int i = 0; i < 4; i++) a += (nxt() & 0xFFFFF) / 1048576.0f; return a - 2.0f; }

int main() {
    const size_t n = (size_t)STRIDE * STRIDE;
    std::vector<float> h(n);
    /* Rows cycle through three regimes: continuous, 40% exact-zero plateau (the relu floor), and
     * coarse 1/8 quantization (huge exact-score ties at the boundary). */
    for (size_t r = 0; r < STRIDE; r++) {
        const unsigned mode = r % 3;
        for (size_t c = 0; c < STRIDE; c++) {
            float v = gauss();
            if (mode == 1 && (nxt() % 10) < 4) v = 0.0f;
            if (mode == 2) v = (float)(int)(v * 8.0f) / 8.0f;
            h[r * STRIDE + c] = v;
        }
    }
    float* dsc; int *dlen, *d_old, *d_new;
    CK(hipMalloc(&dsc, n * 4));
    CK(hipMalloc(&dlen, 4));
    CK(hipMalloc(&d_old, (size_t)STRIDE * TOPK * 4));
    CK(hipMalloc(&d_new, (size_t)STRIDE * TOPK * 4));
    CK(hipMemcpy(dsc, h.data(), n * 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    int fails = 0;
    for (unsigned T : {4096u, 8192u, 16384u, 16376u}) {
        int len = (int)T;
        CK(hipMemcpy(dlen, &len, 4, hipMemcpyHostToDevice));
        CK(hipMemset(d_old, 0xF9, (size_t)STRIDE * TOPK * 4));
        CK(hipMemset(d_new, 0xF9, (size_t)STRIDE * TOPK * 4));
        float ms[2] = {0, 0};
        for (int k = 0; k < 2; k++) {
            auto fn = k ? k_new : k_old;
            int* out = k ? d_new : d_old;
            hipLaunchKernelGGL(fn, dim3(NWG), dim3(512), 0, 0, out, dsc, dlen, T);
            CK(hipDeviceSynchronize());
            const int reps = 5;
            CK(hipEventRecord(e0));
            for (int i = 0; i < reps; i++) hipLaunchKernelGGL(fn, dim3(NWG), dim3(512), 0, 0, out, dsc, dlen, T);
            CK(hipEventRecord(e1));
            CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms[k], e0, e1));
            ms[k] /= reps;
        }
        std::vector<int> a((size_t)STRIDE * TOPK), b((size_t)STRIDE * TOPK);
        CK(hipMemcpy(a.data(), d_old, a.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), d_new, b.size() * 4, hipMemcpyDeviceToHost));
        unsigned bad = 0, badref = 0, untouched_bad = 0;
        for (unsigned r = 0; r < STRIDE; r++) {
            int* ra = &a[(size_t)r * TOPK];
            int* rb = &b[(size_t)r * TOPK];
            if (r >= T) {
                for (unsigned j = 0; j < TOPK; j++) untouched_bad += (ra[j] != (int)0xF9F9F9F9) + (rb[j] != (int)0xF9F9F9F9);
                continue;
            }
            std::sort(ra, ra + TOPK);
            std::sort(rb, rb + TOPK);
            if (memcmp(ra, rb, TOPK * 4)) bad++;
            if (r % 61 == 0 || r + 3 >= T) {
                const unsigned rl = r + 1;
                std::vector<int> ref;
                if (rl <= TOPK) {
                    for (unsigned j = 0; j < TOPK; j++) ref.push_back(j < rl ? (int)j : -1);
                } else {
                    std::vector<std::pair<unsigned long long, int>> keys(rl);
                    for (unsigned s = 0; s < rl; s++) {
                        unsigned sb;
                        memcpy(&sb, &h[(size_t)r * STRIDE + s], 4);
                        sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
                        keys[s] = {((unsigned long long)sb << 24) | (rl - 1 - s), (int)s};
                    }
                    std::nth_element(keys.begin(), keys.begin() + TOPK, keys.end(),
                                     [](auto& x, auto& y) { return x.first > y.first; });
                    for (unsigned j = 0; j < TOPK; j++) ref.push_back(keys[j].second);
                }
                std::sort(ref.begin(), ref.end());
                if (memcmp(ref.data(), rb, TOPK * 4)) badref++;
            }
        }
        printf("T=%5u old %8.1f us  new %8.1f us  (%.2fx)  rows-differ %u  new-vs-cpu-ref-bad %u  rows>=T-touched %u\n",
               T, ms[0] * 1e3, ms[1] * 1e3, ms[0] / ms[1], bad, badref, untouched_bad);
        fails += bad + badref + untouched_bad;
    }
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
}

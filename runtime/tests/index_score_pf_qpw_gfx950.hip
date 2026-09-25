/* index_score_pf_qpw_gfx950.hip — DSA prefill indexer score (op 117): d_index_score_pf_rowq
 * (QPW rows per wave, PLOW_DSA_IDX_QPW) vs the shipped row-resident d_index_score_pf_row, same
 * 256-workgroup grid. Output must be byte-identical over the causal region. Timing per call.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/index_score_pf_qpw_gfx950.hip -o iscore_qpw
 */
#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr unsigned STRIDE = 16384, NWG = 256, TN = 64;
constexpr float SCALE = 0.015624999f;

#define KT_HALVES (TN * (128 + 8))
extern "C" __global__ __launch_bounds__(512, 1) void k_row(float* s, const bf16* q, const bf16* k, const bf16* w,
                                                          const int* len, unsigned n) {
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(FA_PAD <= 8, "tile");
    __shared__ __align__(16) bf16 kt[KT_HALVES];
    d_index_score_pf_row<128, 32, TN>(s, q, k, w, len, n, STRIDE, SCALE, blockIdx.x, gridDim.x, kt);
#endif
}
template <int Q, unsigned TNQ = TN, unsigned SP = 1024>
__device__ void run_q(float* s, const bf16* q, const bf16* k, const bf16* w, const int* len, unsigned n) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ __align__(16) bf16 kt[TNQ * (128 + 8)];
    d_index_score_pf_rowq<128, TNQ, Q, SP>(s, q, k, w, len, n, STRIDE, SCALE, blockIdx.x, gridDim.x, kt);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_q2(float* s, const bf16* q, const bf16* k, const bf16* w,
                                                         const int* len, unsigned n) { run_q<2, 128, 16384>(s, q, k, w, len, n); }
extern "C" __global__ __launch_bounds__(512, 1) void k_q3(float* s, const bf16* q, const bf16* k, const bf16* w,
                                                         const int* len, unsigned n) { run_q<1, 128, 16384>(s, q, k, w, len, n); }

static unsigned long long rng = 0x243F6A8885A308D3ull;
static unsigned nxt() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)(rng >> 11); }
static unsigned short rb(float lo, float hi) {
    float v = lo + (hi - lo) * ((nxt() & 0xFFFFFF) / 16777216.0f);
    unsigned u; memcpy(&u, &v, 4); return (unsigned short)(u >> 16);
}

int main() {
    const unsigned T = STRIDE;
    std::vector<unsigned short> hq((size_t)T * 32 * 128), hk((size_t)T * 128), hw((size_t)T * 32);
    for (auto& x : hq) x = rb(-2, 2);
    for (auto& x : hk) x = rb(-2, 2);
    for (auto& x : hw) x = rb(-1, 1);
    bf16 *dq, *dk, *dw; float *s0, *s1; int* dlen;
    CK(hipMalloc(&dq, hq.size() * 2)); CK(hipMalloc(&dk, hk.size() * 2)); CK(hipMalloc(&dw, hw.size() * 2));
    CK(hipMalloc(&s0, (size_t)T * STRIDE * 4)); CK(hipMalloc(&s1, (size_t)T * STRIDE * 4)); CK(hipMalloc(&dlen, 4));
    CK(hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int fails = 0;
    std::vector<float> a((size_t)T * STRIDE), b((size_t)T * STRIDE);
    for (unsigned n : {4096u, 8192u, 16384u}) {
        int len = (int)n;
        CK(hipMemcpy(dlen, &len, 4, hipMemcpyHostToDevice));
        decltype(&k_row) ks[3] = {k_row, k_q2, k_q3};
        float ms[3];
        for (int v = 0; v < 3; v++) {
            float* out = v ? s1 : s0;
            CK(hipMemset(out, 0, (size_t)T * STRIDE * 4));
            hipLaunchKernelGGL(ks[v], dim3(NWG), dim3(512), 0, 0, out, dq, dk, dw, dlen, n);
            CK(hipDeviceSynchronize());
            const int reps = 5;
            CK(hipEventRecord(e0));
            for (int i = 0; i < reps; i++) hipLaunchKernelGGL(ks[v], dim3(NWG), dim3(512), 0, 0, out, dq, dk, dw, dlen, n);
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms[v], e0, e1)); ms[v] /= reps;
            if (v == 0) { CK(hipMemcpy(a.data(), s0, a.size() * 4, hipMemcpyDeviceToHost)); continue; }
            CK(hipMemcpy(b.data(), s1, b.size() * 4, hipMemcpyDeviceToHost));
            size_t bad = 0;
            for (unsigned r = 0; r < n; r++)
                bad += memcmp(&a[(size_t)r * STRIDE], &b[(size_t)r * STRIDE], (size_t)(r + 1) * 4) != 0;
            printf("T=%5u QPW=%d rows-differ %zu\n", n, v == 1 ? 2 : 1, bad);
            fails += bad != 0;
        }
        const double fl = (double)n * (n + 1) / 2 * 32 * 128 * 2;
        printf("T=%5u row %8.1f us (%.0f TF/s)  rowq<2,128,16384> %8.1f us (%.0f)  rowq<1,128,16384> %8.1f us (%.0f)\n", n, ms[0] * 1e3,
               fl / ms[0] / 1e9, ms[1] * 1e3, fl / ms[1] / 1e9, ms[2] * 1e3, fl / ms[2] / 1e9);
    }
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
}

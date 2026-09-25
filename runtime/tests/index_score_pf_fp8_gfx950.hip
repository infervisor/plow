/* index_score_pf_fp8_gfx950.hip — DSA prefill indexer score, FP8 arm (PLOW_DSA_IDX_FP8):
 * d_index_score_pf_fp8 vs a CPU model of vLLM 0.29's recipe (e4m3 RNE q per (token, head) with
 * ue8m0(max(amax/448, 1e-10)), k per position with ue8m0(max(amax, 1e-4)/448),
 * ws = ((w*qs)*128^-0.5)*32^-0.5, score = sum_h ws*relu(q8.k8) * ks). The quantized operands must
 * match exactly; the f32 accumulation order differs, so scores are checked to 1e-5 relative to the
 * row's |ws|.|dot| mass. Timing vs the bf16 row arm at T=4096/8192/16384.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/index_score_pf_fp8_gfx950.hip -o iscore_fp8
 */
#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr unsigned STRIDE = 16384, NWG = 256;

extern "C" __global__ __launch_bounds__(512, 1) void k_row(float* s, const bf16* q, const bf16* k, const bf16* w,
                                                          const int* len, unsigned n) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ __align__(16) bf16 kt[64 * (128 + 8)];
    d_index_score_pf_row<128, 32, 64>(s, q, k, w, len, n, STRIDE, 0.015624999f, blockIdx.x, gridDim.x, kt);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_fp8(float* s, const bf16* q, const bf16* k, const bf16* w,
                                                          const int* len, unsigned n) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ __align__(16) unsigned char lds[128 * 144 + 128 * 4];
    d_index_score_pf_fp8<128u, 2, 16384u>(s, q, k, w, len, n, STRIDE, blockIdx.x, gridDim.x, lds);
#endif
}

static float b2f(unsigned short b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static float e4m3_val(unsigned c) {
    const unsigned e = (c >> 3) & 15u, m = c & 7u;
    const float v = e ? ldexpf(1.0f + m / 8.0f, (int)e - 7) : ldexpf(m / 8.0f, -6);
    return (c & 0x80u) ? -v : v;
}
static std::vector<float> POS; /* positive codes 0..126 in increasing value */
static float q8(float x) {     /* RNE to e4m3fn, |x| <= 448 */
    const float a = fabsf(x);
    auto it = std::lower_bound(POS.begin(), POS.end(), a);
    unsigned hi = (unsigned)(it - POS.begin());
    if (hi >= POS.size()) hi = (unsigned)POS.size() - 1;
    unsigned best = hi;
    if (hi > 0) {
        const float dl = a - POS[hi - 1], dh = POS[hi] - a;
        if (dl < dh || (dl == dh && ((hi - 1) & 1u) == 0u)) best = hi - 1;
    }
    return x < 0 ? -POS[best] : POS[best];
}
static float ue8m0(float raw) { unsigned b; memcpy(&b, &raw, 4); b = (b + 0x7fffffu) & 0xff800000u; float f; memcpy(&f, &b, 4); return f; }

static unsigned long long rng = 0x243F6A8885A308D3ull;
static unsigned nxt() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)(rng >> 11); }
static unsigned short rb(float lo, float hi) {
    float v = lo + (hi - lo) * ((nxt() & 0xFFFFFF) / 16777216.0f);
    unsigned u; memcpy(&u, &v, 4); return (unsigned short)(u >> 16);
}

int main() {
    for (unsigned c = 0; c < 127; c++) POS.push_back(e4m3_val(c));
    const unsigned T = STRIDE;
    std::vector<unsigned short> hq((size_t)T * 32 * 128), hk((size_t)T * 128), hw((size_t)T * 32);
    for (size_t i = 0; i < hq.size(); i++) hq[i] = rb(-2, 2);
    for (size_t i = 0; i < hk.size(); i++) hk[i] = rb(-2, 2) ;
    for (unsigned r = 0; r < T; r += 7) for (unsigned d = 0; d < 128; d++) hk[(size_t)r * 128 + d] = rb(-0.01f, 0.01f);
    for (auto& x : hw) x = rb(-1, 1);
    bf16 *dq, *dk, *dw; float *s0, *s1; int* dlen;
    CK(hipMalloc(&dq, hq.size() * 2)); CK(hipMalloc(&dk, hk.size() * 2)); CK(hipMalloc(&dw, hw.size() * 2));
    CK(hipMalloc(&s0, (size_t)T * STRIDE * 4)); CK(hipMalloc(&s1, (size_t)T * STRIDE * 4)); CK(hipMalloc(&dlen, 4));
    CK(hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
    /* CPU quantized keys for the whole cache */
    std::vector<float> k8((size_t)T * 128), ks(T);
    for (unsigned p = 0; p < T; p++) {
        float am = 0;
        for (unsigned d = 0; d < 128; d++) am = fmaxf(am, fabsf(b2f(hk[(size_t)p * 128 + d])));
        ks[p] = ue8m0(fmaxf(am, 1e-4f) * (1.0f / 448.0f));
        for (unsigned d = 0; d < 128; d++) k8[(size_t)p * 128 + d] = q8(b2f(hk[(size_t)p * 128 + d]) / ks[p]);
    }
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int fails = 0;
    std::vector<float> out((size_t)T * STRIDE);
    for (unsigned n : {4096u, 8192u, 16384u}) {
        int len = (int)n;
        CK(hipMemcpy(dlen, &len, 4, hipMemcpyHostToDevice));
        float ms[2];
        for (int v = 0; v < 2; v++) {
            auto fn = v ? k_fp8 : k_row;
            float* o = v ? s1 : s0;
            CK(hipMemset(o, 0, (size_t)T * STRIDE * 4));
            hipLaunchKernelGGL(fn, dim3(NWG), dim3(512), 0, 0, o, dq, dk, dw, dlen, n);
            CK(hipDeviceSynchronize());
            CK(hipEventRecord(e0));
            for (int i = 0; i < 5; i++) hipLaunchKernelGGL(fn, dim3(NWG), dim3(512), 0, 0, o, dq, dk, dw, dlen, n);
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms[v], e0, e1)); ms[v] /= 5;
        }
        CK(hipMemcpy(out.data(), s1, out.size() * 4, hipMemcpyDeviceToHost));
        double worst = 0; size_t checked = 0, bad = 0;
        for (unsigned t = 0; t < n; t += (n / 48) + 1) {
            float ws[32], qv[32][128];
            for (unsigned h = 0; h < 32; h++) {
                float am = 0;
                for (unsigned d = 0; d < 128; d++) am = fmaxf(am, fabsf(b2f(hq[((size_t)t * 32 + h) * 128 + d])));
                const float qs = ue8m0(fmaxf(am * (1.0f / 448.0f), 1e-10f));
                for (unsigned d = 0; d < 128; d++) qv[h][d] = q8(b2f(hq[((size_t)t * 32 + h) * 128 + d]) / qs);
                float x = b2f(hw[(size_t)t * 32 + h]) * qs;
                x *= 0.08838834764831845f;
                ws[h] = x * 0.1767766952966369f;
            }
            for (unsigned p = 0; p <= t; p++) {
                double acc = 0, mass = 0;
                for (unsigned h = 0; h < 32; h++) {
                    double dot = 0;
                    for (unsigned d = 0; d < 128; d++) dot += (double)qv[h][d] * k8[(size_t)p * 128 + d];
                    if (dot > 0) acc += ws[h] * dot;
                    mass += fabs(ws[h] * dot);
                }
                const double ref = acc * ks[p], got = out[(size_t)t * STRIDE + p];
                const double err = fabs(got - ref) / (mass * ks[p] + 1e-30);
                worst = std::max(worst, err);
                bad += err > 1e-5;
                checked++;
            }
        }
        const double fl = (double)n * (n + 1) / 2 * 32 * 128 * 2;
        printf("T=%5u bf16-row %8.1f us (%.0f TF/s)  fp8 %8.1f us (%.0f TF/s)  checked %zu worst-rel %.2e bad %zu\n",
               n, ms[0] * 1e3, fl / ms[0] / 1e9, ms[1] * 1e3, fl / ms[1] / 1e9, checked, worst, bad);
        fails += bad != 0;
    }
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
}

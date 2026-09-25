/* index_score_pf_fp8_gfx950.hip — DSA prefill indexer score, FP8 arm (PLOW_DSA_IDX_FP8,
 * runtime/amd/dsa_score_fp8.h) vs a CPU model of vLLM 0.29 ROCm's recipe, and timing vs the
 * shipped bf16 QPW arm (d_index_score_pf_rowq<128,128,2,16384>), same 256 x 512-thread grid.
 * CPU model: exact quotients, ue8m0 = 2^ceil(log2), e4m3fn RNE, exact dot rounded to f32, then the
 * pinned aiter gluon kernel's head-sum order (see head_sum2), + the other lane half, x kscale. The
 * MFMA does not round its 64-term sums exactly, so the check is relative to sum |w * q8 * k8| * ks.
 * `dump DIR [T]` writes the operands and scores for the in-image vLLM comparison.
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
#include <string>
#include <vector>
#include "amd_common.h"
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#include "dsa_score_fp8.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr unsigned STRIDE = 16384, NWG = 256;
constexpr float SCALE = 0.015624999f;

extern "C" __global__ __launch_bounds__(512, 1) void k_qpw(float* s, const bf16* q, const bf16* k, const bf16* w,
                                                          const int* len, unsigned n) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ __align__(16) bf16 kt[128 * (128 + 8)];
    d_index_score_pf_rowq<128, 128u, 2, 16384u>(s, q, k, w, len, n, STRIDE, SCALE, blockIdx.x, gridDim.x, kt);
#endif
}
#define FP8_ARM(name, TN, QPW, ABL)                                                                     \
    extern "C" __global__ __launch_bounds__(512, 1) void name(float* s, const bf16* q, const bf16* k,   \
                                                              const bf16* w, const int* len, unsigned n) { \
        FP8_BODY(TN, QPW, ABL)                                                                          \
    }
#if defined(__HIP_DEVICE_COMPILE__)
#define FP8_BODY(TN, QPW, ABL)                                                              \
    __shared__ __align__(16) unsigned char lds[dsa_fp8::lds_bytes<TN>()];                   \
    d_index_score_pf_fp8<TN, QPW, ABL>(s, q, k, w, len, n, STRIDE, blockIdx.x, gridDim.x, lds);
#else
#define FP8_BODY(TN, QPW, ABL)
#endif
FP8_ARM(k_fp8_128_2, 128u, 2, 0)
FP8_ARM(k_fp8_256_2, 256u, 2, 0)
FP8_ARM(k_fp8_64_2, 64u, 2, 0)
FP8_ARM(k_fp8_256_sc, 256u, 2, 1024)

static float b2f(unsigned short b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static float e4m3_val(unsigned c) {
    const unsigned e = (c >> 3) & 15u, m = c & 7u;
    return e ? ldexpf(1.0f + m / 8.0f, (int)e - 7) : ldexpf(m / 8.0f, -6);
}
static std::vector<float> POS; /* codes 0..126 (0x7f is NaN), increasing */
static float q8(float x) {     /* RNE to e4m3fn, saturating */
    const float a = fabsf(x);
    auto it = std::lower_bound(POS.begin(), POS.end(), a);
    unsigned hi = (unsigned)(it - POS.begin());
    if (hi >= POS.size()) hi = (unsigned)POS.size() - 1;
    unsigned best = hi;
    if (hi > 0 && POS[hi] != a) {
        const float dl = a - POS[hi - 1], dh = POS[hi] - a;
        if (dl < dh || (dl == dh && ((hi - 1) & 1u) == 0u)) best = hi - 1;
    }
    return x < 0 ? -POS[best] : POS[best];
}
/* vLLM: s = x / 448 (f32 divide), 2^ceil(log2(s)) */
static float ue8m0(float s) { return (float)exp2(ceil(log2((double)s))); }
static float kscale_of(float amax) { return ue8m0((float)((double)std::max(amax, 1e-4f) / 448.0)); }
static float qscale_of(float amax) { return ue8m0(std::max((float)((double)amax / 448.0), 1e-10f)); }

static unsigned long long rng = 0x243F6A8885A308D3ull;
static unsigned nxt() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)(rng >> 11); }
static unsigned short fb(float v) { unsigned u; memcpy(&u, &v, 4); return (unsigned short)(u >> 16); }
static unsigned short rb(float lo, float hi) { return fb(lo + (hi - lo) * ((nxt() & 0xFFFFFF) / 16777216.0f)); }

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !strcmp(argv[1], "quick");
    for (unsigned c = 0; c < 127; c++) POS.push_back(e4m3_val(c));
    const unsigned T = STRIDE;
    std::vector<unsigned short> hq((size_t)T * 32 * 128), hk((size_t)T * 128), hw((size_t)T * 32);
    for (auto& x : hq) x = rb(-2, 2);
    for (auto& x : hk) x = rb(-2, 2);
    for (auto& x : hw) x = rb(-1, 1);
    /* scale edge cases: amax = 1.75 * 2^m makes amax/448 an exact power of two */
    for (unsigned r = 0; r < T; r += 5) hk[(size_t)r * 128 + (r % 128)] = fb(r % 10 ? 1.75f : -3.5f);
    for (unsigned r = 3; r < T; r += 7)
        for (unsigned d = 0; d < 128; d++) hk[(size_t)r * 128 + d] = rb(-1e-5f, 1e-5f); /* 1e-4 floor */
    for (size_t g = 0; g < (size_t)T * 32; g += 3) hq[g * 128 + (g % 128)] = fb(g % 2 ? 1.75f : -0.875f);
    for (size_t g = 1; g < (size_t)T * 32; g += 11)
        for (unsigned d = 0; d < 128; d++) hq[g * 128 + d] = rb(-0.01f, 0.01f);

    bf16 *dq, *dk, *dw; float *s0, *s1; int* dlen;
    CK(hipMalloc(&dq, hq.size() * 2)); CK(hipMalloc(&dk, hk.size() * 2)); CK(hipMalloc(&dw, hw.size() * 2));
    CK(hipMalloc(&s0, (size_t)T * STRIDE * 4)); CK(hipMalloc(&s1, (size_t)T * STRIDE * 4)); CK(hipMalloc(&dlen, 4));
    CK(hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
    std::vector<float> k8((size_t)T * 128), ksc(T);
    for (unsigned p = 0; p < T; p++) {
        float am = 0;
        for (unsigned d = 0; d < 128; d++) am = fmaxf(am, fabsf(b2f(hk[(size_t)p * 128 + d])));
        ksc[p] = kscale_of(am);
        for (unsigned d = 0; d < 128; d++) k8[(size_t)p * 128 + d] = q8(b2f(hk[(size_t)p * 128 + d]) / ksc[p]);
    }
    struct Arm { const char* name; decltype(&k_qpw) fn; };
    if (argc > 2 && !strcmp(argv[1], "dump")) { /* inputs + this arm's scores for the vLLM kernel check */
        const unsigned n = argc > 3 ? (unsigned)atoi(argv[3]) : 4096u;
        int len = (int)n;
        CK(hipMemcpy(dlen, &len, 4, hipMemcpyHostToDevice));
        CK(hipMemset(s1, 0, (size_t)T * STRIDE * 4));
        std::vector<float> o((size_t)n * STRIDE);
        auto put = [&](const char* name, const void* p, size_t bytes) {
            std::string f = std::string(argv[2]) + "/" + name;
            FILE* fp = fopen(f.c_str(), "wb");
            if (!fp || fwrite(p, 1, bytes, fp) != bytes) { printf("write %s failed\n", f.c_str()); exit(1); }
            fclose(fp);
        };
        put("q.bin", hq.data(), (size_t)n * 32 * 128 * 2);
        put("k.bin", hk.data(), (size_t)n * 128 * 2);
        put("w.bin", hw.data(), (size_t)n * 32 * 2);
        std::vector<float> c((size_t)n * n);
        const Arm vars[] = {{"plow.bin", k_fp8_256_2}, {"plow_scaled.bin", k_fp8_256_sc}};
        for (const Arm& v : vars) {
            CK(hipMemset(s1, 0, (size_t)T * STRIDE * 4));
            hipLaunchKernelGGL(v.fn, dim3(NWG), dim3(512), 0, 0, s1, dq, dk, dw, dlen, n);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(o.data(), s1, o.size() * 4, hipMemcpyDeviceToHost));
            for (unsigned r = 0; r < n; r++) memcpy(&c[(size_t)r * n], &o[(size_t)r * STRIDE], (size_t)n * 4);
            put(v.name, c.data(), c.size() * 4);
        }
        printf("dumped T=%u to %s\n", n, argv[2]);
        return 0;
    }
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int fails = 0;
    std::vector<float> out((size_t)T * STRIDE);
    const Arm arms[] = {{"fp8<128,2>", k_fp8_128_2}, {"fp8<256,2>", k_fp8_256_2}, {"fp8<64,2>", k_fp8_64_2}};
    struct Case { unsigned len, n; };
    const Case cases[] = {{4096, 4096}, {8192, 8192}, {16384, 16384}, {8192, 4096}};
    for (const Case cs : cases) {
        if (quick && cs.len > 8192) continue;
        const unsigned n = cs.n, q0 = cs.len - cs.n;
        int len = (int)cs.len;
        CK(hipMemcpy(dlen, &len, 4, hipMemcpyHostToDevice));
        auto timeit = [&](decltype(&k_qpw) fn, float* o) {
            CK(hipMemset(o, 0, (size_t)T * STRIDE * 4));
            hipLaunchKernelGGL(fn, dim3(NWG), dim3(512), 0, 0, o, dq, dk, dw, dlen, n);
            CK(hipDeviceSynchronize());
            const int reps = 10;
            CK(hipEventRecord(e0));
            for (int i = 0; i < reps; i++) hipLaunchKernelGGL(fn, dim3(NWG), dim3(512), 0, 0, o, dq, dk, dw, dlen, n);
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            return ms / reps;
        };
        const double fl = ((double)n * (n + 1) / 2 + (double)n * q0) * 32 * 128 * 2;
        const float mq = timeit(k_qpw, s0);
        printf("len=%5u n=%5u bf16 qpw %8.1f us (%4.0f TF/s)\n", cs.len, n, mq * 1e3, fl / mq / 1e9);
        std::vector<unsigned> rows;
        for (unsigned t = 0; t < n; t += n / 24 + 1) rows.push_back(t);
        for (unsigned t = n - 3; t < n; t++) rows.push_back(t);
        std::vector<std::vector<float>> refs(rows.size()), mass(rows.size());
        for (size_t ri = 0; ri < rows.size(); ri++) {
            const unsigned t = rows[ri];
            float ws[32];
            std::vector<float> qv(32 * 128);
            for (unsigned h = 0; h < 32; h++) {
                const unsigned short* qr = &hq[((size_t)t * 32 + h) * 128];
                float am = 0;
                for (unsigned d = 0; d < 128; d++) am = fmaxf(am, fabsf(b2f(qr[d])));
                const float qs = qscale_of(am);
                for (unsigned d = 0; d < 128; d++) qv[h * 128 + d] = q8(b2f(qr[d]) / qs);
                float x = b2f(hw[(size_t)t * 32 + h]) * qs;
                x *= 0.08838834764831845f;
                ws[h] = x * 0.1767766952966369f;
            }
            for (unsigned p = 0; p <= q0 + t; p++) {
                float dot[32];
                double m = 0;
                for (unsigned h = 0; h < 32; h++) {
                    double s = 0, sa = 0;
                    for (unsigned d = 0; d < 128; d++) {
                        const double pr = (double)qv[h * 128 + d] * k8[(size_t)p * 128 + d];
                        s += pr;
                        sa += fabs(pr);
                    }
                    dot[h] = (float)s;
                    m += fabs((double)ws[h]) * sa;
                }
                float part[2];
                for (unsigned kh = 0; kh < 2; kh++) {
                    float wr[16], rr[16];
                    for (unsigned i = 0; i < 16; i++) {
                        const unsigned h = 4 * kh + (i % 4) + 8 * (i / 4);
                        wr[i] = ws[h];
                        rr[i] = dot[h] > 0.0f ? dot[h] : 0.0f;
                    }
                    float acc = wr[1] * rr[1];
                    acc = fmaf(wr[0], rr[0], acc);
                    for (unsigned i = 2; i < 10; i++) acc = fmaf(wr[i], rr[i], acc);
                    for (unsigned i = 10; i < 16; i++) {
                        const volatile float pr = wr[i] * rr[i];
                        acc = acc + pr;
                    }
                    part[kh] = acc;
                }
                refs[ri].push_back((part[0] + part[1]) * ksc[p]);
                mass[ri].push_back((float)(m * ksc[p]));
            }
        }
        for (const Arm& a : arms) {
            const float ms = timeit(a.fn, s1);
            CK(hipMemcpy(out.data(), s1, out.size() * 4, hipMemcpyDeviceToHost));
            double worst = 0; size_t checked = 0, exact = 0, bad = 0;
            for (size_t ri = 0; ri < rows.size(); ri++)
                for (size_t p = 0; p < refs[ri].size(); p++) {
                    const float ref = refs[ri][p], got = out[(size_t)rows[ri] * STRIDE + p];
                    const double err = fabs((double)got - ref) / ((double)mass[ri][p] + 1e-30);
                    worst = std::max(worst, err);
                    exact += got == ref;
                    if (!(err <= 1e-4) && bad < 4) printf("    bad row %u pos %zu got %.9g ref %.9g mass %.3g\n", rows[ri], p, got, ref, mass[ri][p]);
                    bad += !(err <= 1e-4);
                    checked++;
                }
            printf("len=%5u n=%5u %-11s %8.1f us (%4.0f TF/s)  checked %zu exact %.4f worst-rel %.2e bad %zu\n",
                   cs.len, n, a.name, ms * 1e3, fl / ms / 1e9, checked, (double)exact / checked, worst, bad);
            fails += bad != 0;
        }
    }
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
}

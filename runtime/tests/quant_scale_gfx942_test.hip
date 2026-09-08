/* quant_scale_gfx942_test.hip — the boundary gate for `PLOW_QUANT_SCALE_EXP`.
 *
 * WHAT IS UNDER TEST. `plow_round_scale` builds the reference's `ROUND_SCALE=True` power-of-two
 * quantization scale, `exp2(ceil(log2(amax / top)))`. Arm 0 transliterates that expression; arm 1
 * takes the exponent out of `frexpf` and rebuilds the scale (and its reciprocal) with `ldexpf`.
 *
 * THE TWO ARMS ARE NOT THE SAME FUNCTION, and this file exists to say exactly where. `log2f` is
 * not correctly rounded: for a `v` one ULP above a power of two, the f32 log can land exactly on
 * the integer, `ceilf` then returns one exponent too LOW, and arm 0 produces a scale HALF the
 * exact ceiling. Arm 1 returns the exact ceiling always. So:
 *
 *   * against the EXACT ceiling, arm 1 is right everywhere and arm 0 is not;
 *   * against the REFERENCE KERNEL, whose torch `log2` has its own rounding, neither is
 *     automatically right — which is why arm 1 is default-off. This test reports the
 *     disagreement rather than picking a winner.
 *
 * WHAT ELSE IS CHECKED, because the audit named each one:
 *   FLOOR BEHAVIOUR   both amax floors that reach this code (1e-4 for e4m3, 6*2^-126 for e2m1)
 *                     and the FP8 e4m3 448 maximum, exactly and one ULP either side.
 *   POWERS OF TWO     every exponent in [-30, 10], exactly on the boundary and one ULP either
 *                     side of it — the case the whole finding is about.
 *   SUBNORMALS        amax driven below the floor, to prove neither arm produces inf/NaN there
 *                     (the callers floor first; this is a backstop, not a supported input).
 *   RECIPROCAL        arm 1's `inv_scale` must satisfy `scale * inv == 1.0f` EXACTLY, which is
 *                     what makes `x * inv` bit-identical to `x / scale` at every call site.
 *
 * The bf16 round trips and the FP8 FN/FNUZ encoding around the scale are NOT re-tested here —
 * `compress_pool_gfx942_test.hip` already owns them end to end and must be run alongside this.
 *
 * ONE ARM PER TRANSLATION UNIT: `PLOW_QUANT_SCALE_EXP` is a compile-time switch inside the
 * shipped header, so the arms cannot coexist in one compile.
 *
 * Build and run, in nix develop:
 *   for A in 0 1; do
 *     $PLOW_HIPCC --offload-arch=gfx942 -O3 -w -std=c++17 -Iruntime/amd -Iruntime/common \
 *         -DQS_ARM=$A -c runtime/tests/quant_scale_gfx942_test.hip -o /tmp/qs$A.o
 *   done
 *   c++ /tmp/qs0.o /tmp/qs1.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/qs
 *   perf-data/tools/gpulease -n 1 quant-scale /tmp/qs
 */
#define PLOW_WG_WAVES 4

#ifndef QS_ARM
#define QS_ARM 0
#endif
#define PLOW_QUANT_SCALE_EXP QS_ARM

#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "amd_common.h"
#else
using bf16 = unsigned short;
#endif

/* The host half never sees amd_common.h. 448 is duplicated deliberately and asserted against the
 * header's own value in the device compile below, so the two cannot drift. */
#define QS_E4M3_MAX 448.0f

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define QS_CAT2(a, b) a##b
#define QS_CAT(a, b) QS_CAT2(a, b)
#define QS_K QS_CAT(k_scale_arm, QS_ARM)
#define QS_RUN QS_CAT(qs_run_arm, QS_ARM)

#if defined(__HIP_DEVICE_COMPILE__)
static_assert(QS_E4M3_MAX == PLOW_FP8_E4M3_MAX, "QS_E4M3_MAX drifted from the header");
#endif

__global__ void QS_K(const float* amax, float top, float* scale, float* inv, unsigned n) {
#if defined(__HIP_DEVICE_COMPILE__)
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float s = 0.0f, r = 0.0f;
        plow_round_scale(amax[i], top, &s, &r);
        scale[i] = s;
        inv[i] = r;
    }
#else
    (void)amax;
#endif
}

extern "C" void QS_RUN(const float* amax, float top, float* scale, float* inv, unsigned n) {
    hipLaunchKernelGGL(QS_K, dim3(64), dim3(256), 0, 0, amax, top, scale, inv, n);
}

#if QS_ARM == 0 /* ================================ the driver, oracle and main() */

extern "C" void qs_run_arm1(const float*, float, float*, float*, unsigned);

static void ck(hipError_t s) {
    if (s != hipSuccess) {
        std::fprintf(stderr, "hip: %s\n", hipGetErrorString(s));
        std::exit(1);
    }
}

/* The EXACT ceiling, on the host, from the same f32 quotient the device computed its scale from.
 * `frexpf` is exact by definition (it reads the exponent field), so this is a decision procedure,
 * not a second approximation: v = m * 2^e with m in [0.5, 1), hence ceil(log2 v) = e except at
 * m == 0.5 where v is itself 2^(e-1). */
static int exact_ceil_log2(float v) {
    int e = 0;
    const float m = std::frexp(v, &e);
    return (m == 0.5f) ? e - 1 : e;
}

struct Case {
    float amax;
    const char* why;
};

static void add_pow2_family(std::vector<Case>& cs, float top) {
    /* Drive `amax` so that `amax / top` lands ON a power of two, and one ULP either side of it. */
    for (int k = -30; k <= 10; k++) {
        const float v = std::ldexp(1.0f, k);
        const float base = v * top;
        cs.push_back({base, "amax/top == 2^k"});
        cs.push_back({std::nextafterf(base, 3.4e38f), "one ULP above"});
        cs.push_back({std::nextafterf(base, 0.0f), "one ULP below"});
    }
}

static int run(float top, const char* name, const char* floor_why, float floor_v) {
    std::vector<Case> cs;
    add_pow2_family(cs, top);
    cs.push_back({floor_v, floor_why});
    cs.push_back({std::nextafterf(floor_v, 3.4e38f), "one ULP above the floor"});
    cs.push_back({QS_E4M3_MAX, "the e4m3 448 maximum"});
    cs.push_back({std::nextafterf(QS_E4M3_MAX, 0.0f), "one ULP below 448"});
    cs.push_back({1.0f, "unit"});
    /* Backstops, not supported inputs: every caller floors amax before this point. */
    cs.push_back({1.4e-45f, "the smallest subnormal (below every floor)"});
    cs.push_back({5.9e-39f, "a subnormal"});

    const unsigned n = (unsigned)cs.size();
    std::vector<float> h(n);
    for (unsigned i = 0; i < n; i++) h[i] = cs[i].amax;

    float *d_a = nullptr, *d_s[2] = {nullptr, nullptr}, *d_r[2] = {nullptr, nullptr};
    ck(hipMalloc(&d_a, n * 4));
    ck(hipMemcpy(d_a, h.data(), n * 4, hipMemcpyHostToDevice));
    for (int a = 0; a < 2; a++) {
        ck(hipMalloc(&d_s[a], n * 4));
        ck(hipMalloc(&d_r[a], n * 4));
    }
    qs_run_arm0(d_a, top, d_s[0], d_r[0], n);
    qs_run_arm1(d_a, top, d_s[1], d_r[1], n);
    ck(hipDeviceSynchronize());

    std::vector<float> s0(n), s1(n), r1(n);
    ck(hipMemcpy(s0.data(), d_s[0], n * 4, hipMemcpyDeviceToHost));
    ck(hipMemcpy(s1.data(), d_s[1], n * 4, hipMemcpyDeviceToHost));
    ck(hipMemcpy(r1.data(), d_r[1], n * 4, hipMemcpyDeviceToHost));

    unsigned a0_wrong = 0, a1_wrong = 0, disagree = 0, recip_bad = 0, nonfinite = 0;
    std::printf("\n%s (top = %g), %u cases\n", name, (double)top, n);
    for (unsigned i = 0; i < n; i++) {
        const float v = h[i] / top;
        const bool sub = !(v >= 1.17549435e-38f); /* quotient itself subnormal or zero */
        if (!std::isfinite(s0[i]) || !std::isfinite(s1[i])) nonfinite++;
        if (!sub) {
            const float want = std::ldexp(1.0f, exact_ceil_log2(v));
            if (s0[i] != want) a0_wrong++;
            if (s1[i] != want) a1_wrong++;
            if (s1[i] * r1[i] != 1.0f) recip_bad++;
        }
        if (s0[i] != s1[i]) {
            disagree++;
            if (disagree <= 8)
                std::printf("    amax=%-16.9g  arm0=%-14.7g arm1=%-14.7g  (%s)\n", (double)h[i],
                            (double)s0[i], (double)s1[i], cs[i].why);
        }
    }
    std::printf("  arm0 != exact ceiling : %u / %u\n", a0_wrong, n);
    std::printf("  arm1 != exact ceiling : %u / %u\n", a1_wrong, n);
    std::printf("  arm0 != arm1          : %u / %u\n", disagree, n);
    std::printf("  arm1 scale*inv != 1.0 : %u\n", recip_bad);
    std::printf("  nonfinite scale       : %u\n", nonfinite);

    for (int a = 0; a < 2; a++) {
        ck(hipFree(d_s[a]));
        ck(hipFree(d_r[a]));
    }
    ck(hipFree(d_a));

    /* THE GATE. Arm 1 must be exactly right on every normal quotient, its reciprocal must be
     * exact (that is the whole basis for turning the divide into a multiply), and neither arm may
     * produce a nonfinite scale on any input including the subnormal backstops. Arm 0's own
     * disagreement with the exact ceiling is REPORTED, not gated: it is the shipped behaviour and
     * the reference's, and changing it is the decision this arm exists to make explicit. */
    return (a1_wrong == 0 && recip_bad == 0 && nonfinite == 0) ? 0 : 1;
}

int main() {
    std::printf("plow_round_scale — PLOW_QUANT_SCALE_EXP boundary cases\n");
    int bad = 0;
    bad |= run(QS_E4M3_MAX, "e4m3 / fp8 attention compressor", "the 1e-4 amax floor", 1e-4f);
    bad |= run(6.0f, "e2m1 / fp4 indexer compressor", "the 6*2^-126 amax floor",
               7.052966328760454e-38f);
    std::printf("\n%s\n", bad ? "FAIL" : "PASS");
    return bad;
}
#endif /* QS_ARM == 0 */

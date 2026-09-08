/* hyperconn_sinkhorn_gfx942_test.hip — the f64 oracle gate for `PLOW_HC_SINKHORN_RCP`.
 *
 * WHAT IS UNDER TEST. `d_hyperconn_pre` normalizes a 4x4 combine matrix to doubly stochastic by
 * alternating row and column normalizes, `sinkhorn_repeat` times. Every element of a row (or
 * column) is divided by the SAME denominator, so a reciprocal-and-multiply removes 3 of every 4
 * IEEE divides — 640 -> 160 at n=4 and 20 iterations — and CHANGES ROUNDING while doing it.
 * `PLOW_HC_SINKHORN_RCP` is that arm (0 = shipped divide, 1 = IEEE reciprocal, 2 = `v_rcp_f32`),
 * default off. This file is where its price is paid in numbers rather than asserted.
 *
 * THE ORACLE IS NOT A TRANSLITERATION. `sinkhorn_f64` below implements vLLM's `mhc_pre_torch`
 * loop read as mathematics, in double precision, from the SAME bf16-rounded residual and f32
 * `mixes` the device sees — including the `hc_eps` placement, which differs between the first
 * softmax normalize (`x/s + eps`) and every Sinkhorn pass (`x/(s + eps)`) and is the single
 * easiest thing to get subtly wrong. A shared mistake would have to be made twice.
 *
 * WHY THE INPUT IS NOT THE OBVIOUS ONE. A near-uniform `mixes` row makes the combine matrix
 * start at its own doubly-stochastic fixed point, every denominator lands at 1.0, and every arm
 * agrees to the last bit — a PASS that proves nothing. `LOGIT_SPREAD` deliberately spreads the
 * pre-scale logits over several units so the softmax is peaked, the Sinkhorn actually iterates,
 * and 20 passes of denominators far from 1 have somewhere to accumulate. The sweep runs three
 * spreads and reports the worst.
 *
 * SCORING. Each arm is scored against the f64 oracle, and separately against arm 0 (the shipped
 * f32 result), because the question is not "is arm 1 accurate" but "is arm 1 no further from the
 * truth than the body it replaces". PASS requires every arm's oracle error to stay within
 * `ORACLE_BAR` AND arm 1/2's oracle error to be no worse than 4x arm 0's — a divide replaced by a
 * reciprocal may not turn a 1-ULP body into a 100-ULP one.
 *
 * Build and run, in nix develop:
 *   for R in 0 1 2; do
 *     $PLOW_HIPCC --offload-arch=gfx942 -O3 -w -std=c++17 -Iruntime/amd -Iruntime/common \
 *         -DHC_RCP_ARM=$R -c runtime/tests/hyperconn_sinkhorn_gfx942_test.hip -o /tmp/hcs$R.o
 *   done
 *   c++ /tmp/hcs0.o /tmp/hcs1.o /tmp/hcs2.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib \
 *       -lamdhip64 -o /tmp/hcs        # (each arm is one TU; see HC_ARM_TU below)
 *   perf-data/tools/gpulease -n 1 hc-sinkhorn /tmp/hcs
 *
 * `-Wl,-rpath,$ROCM_PATH/lib` is not optional — see op_dsa_pool.h's file header for the
 * setuid/AT_SECURE failure that flag exists to avoid.
 */
#define PLOW_WG_WAVES 4

/* ONE ARM PER TRANSLATION UNIT. `PLOW_HC_SINKHORN_RCP` is a compile-time switch inside the
 * shipped header, so the three arms cannot coexist in one compile. Each TU defines HC_RCP_ARM,
 * compiles the body under it, and exports `hc_run_<arm>`; HC_ARM_TU==0 additionally carries the
 * host driver, the oracle and main(). */
#ifndef HC_RCP_ARM
#define HC_RCP_ARM 0
#endif
#define PLOW_HC_SINKHORN_RCP HC_RCP_ARM

#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_hyperconn.h"
#else
using bf16 = unsigned short;
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HC_CAT2(a, b) a##b
#define HC_CAT(a, b) HC_CAT2(a, b)
#define HC_K HC_CAT(k_hcpre_arm, HC_RCP_ARM)
#define HC_RUN HC_CAT(hc_run_arm, HC_RCP_ARM)

__global__ __launch_bounds__(256, 1) void HC_K(float* post_mix, float* comb_mix, bf16* layer_input,
                                               const float* mixes, const bf16* residual,
                                               const float* hc_scale, const float* hc_base,
                                               unsigned T, unsigned hidden, unsigned repeat,
                                               float rms_eps, float hc_eps) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float part[PLOW_WG_WAVES];
    d_hyperconn_pre(post_mix, comb_mix, layer_input, mixes, residual, hc_scale, hc_base, T, 4u,
                    hidden, repeat, rms_eps, hc_eps, blockIdx.x, gridDim.x, part);
#else
    (void)post_mix;
#endif
}

/* The host entry each TU exports. Buffers are device pointers owned by the driver TU. */
extern "C" void HC_RUN(unsigned grid, float* post_mix, float* comb_mix, bf16* layer_input,
                       const float* mixes, const bf16* residual, const float* hc_scale,
                       const float* hc_base, unsigned T, unsigned hidden, unsigned repeat,
                       float rms_eps, float hc_eps) {
    hipLaunchKernelGGL(HC_K, dim3(grid), dim3(256), 0, 0, post_mix, comb_mix, layer_input, mixes,
                       residual, hc_scale, hc_base, T, hidden, repeat, rms_eps, hc_eps);
}

#if HC_RCP_ARM == 0 /* ================================ the driver, oracle and main() */

extern "C" void hc_run_arm1(unsigned, float*, float*, bf16*, const float*, const bf16*,
                            const float*, const float*, unsigned, unsigned, unsigned, float, float);
extern "C" void hc_run_arm2(unsigned, float*, float*, bf16*, const float*, const bf16*,
                            const float*, const float*, unsigned, unsigned, unsigned, float, float);

static void ck(hipError_t s) {
    if (s != hipSuccess) {
        std::fprintf(stderr, "hip: %s\n", hipGetErrorString(s));
        std::exit(1);
    }
}
template <typename T>
static T* up(const std::vector<T>& v) {
    T* p = nullptr;
    ck(hipMalloc(&p, v.size() * sizeof(T) + 16));
    ck(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}
static float b2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static bf16 f2b(float f) { /* round-to-nearest-even, matching amd_common.h's f2bf */
    unsigned u;
    std::memcpy(&u, &f, 4);
    return (bf16)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

/* vLLM's mhc_pre_torch Sinkhorn, in f64, from the f32 logits the device computed its row from.
 * `raw[i][j]` is the pre-softmax logit, ALREADY scaled: `(mixes * inv) * hc_scale[2] + hc_base`.
 * Reproducing that scale here in f64 too would test the reduction, not the Sinkhorn; the device's
 * own `inv` is passed in so the two agree on the input and differ only in the loop. */
static void sinkhorn_f64(const double* raw, double* out, unsigned n, unsigned repeat,
                         double hc_eps) {
    std::vector<double> m(raw, raw + (size_t)n * n);
    for (unsigned i = 0; i < n; i++) {
        double mx = -1e300, s = 0.0;
        for (unsigned j = 0; j < n; j++) mx = std::fmax(mx, m[i * n + j]);
        for (unsigned j = 0; j < n; j++) {
            m[i * n + j] = std::exp(m[i * n + j] - mx);
            s += m[i * n + j];
        }
        for (unsigned j = 0; j < n; j++) m[i * n + j] = m[i * n + j] / s + hc_eps;
    }
    for (unsigned j = 0; j < n; j++) {
        double s = 0.0;
        for (unsigned i = 0; i < n; i++) s += m[i * n + j];
        for (unsigned i = 0; i < n; i++) m[i * n + j] /= (s + hc_eps);
    }
    for (unsigned r = 1; r < repeat; r++) {
        for (unsigned i = 0; i < n; i++) {
            double s = 0.0;
            for (unsigned j = 0; j < n; j++) s += m[i * n + j];
            for (unsigned j = 0; j < n; j++) m[i * n + j] /= (s + hc_eps);
        }
        for (unsigned j = 0; j < n; j++) {
            double s = 0.0;
            for (unsigned i = 0; i < n; i++) s += m[i * n + j];
            for (unsigned i = 0; i < n; i++) m[i * n + j] /= (s + hc_eps);
        }
    }
    for (unsigned k = 0; k < n * n; k++) out[k] = m[k];
}

static double frand(unsigned& st) {
    st = st * 1664525u + 1013904223u;
    return (double)(st >> 8) / 16777216.0 - 0.5;
}

struct Err {
    double mx = 0.0, rms = 0.0;
};
static Err cmp(const std::vector<float>& got, const std::vector<double>& want) {
    Err e;
    double se = 0.0;
    for (size_t i = 0; i < want.size(); i++) {
        const double d = std::fabs((double)got[i] - want[i]);
        e.mx = std::fmax(e.mx, d);
        se += d * d;
    }
    e.rms = std::sqrt(se / (double)want.size());
    return e;
}

static const unsigned N = 4, HIDDEN = 512, T = 96, REPEAT = 20;
static const float RMS_EPS = 1e-5f, HC_EPS = 1e-6f;
/* The three spreads. 0.05 is the near-fixed-point case every arm agrees on; 8.0 makes the initial
 * softmax nearly one-hot, which is where 20 passes of far-from-1 denominators have room to drift. */
static const double LOGIT_SPREAD[] = {0.05, 1.5, 8.0};

int main() {
    std::printf("d_hyperconn_pre Sinkhorn — PLOW_HC_SINKHORN_RCP against an f64 oracle\n");
    std::printf("  T=%u n=%u hidden=%u sinkhorn_repeat=%u hc_eps=%g\n\n", T, N, HIDDEN, REPEAT,
                (double)HC_EPS);

    const unsigned n3 = 2 * N + N * N;
    bool pass = true;
    std::printf("  %-8s %-6s %12s %12s %12s %12s\n", "spread", "arm", "orc_max", "orc_rms",
                "vs_arm0_max", "ratio_to_a0");

    for (double spread : LOGIT_SPREAD) {
        unsigned st = 12345u ^ (unsigned)(spread * 1000.0);
        std::vector<float> mixes((size_t)T * n3);
        std::vector<bf16> residual((size_t)T * N * HIDDEN);
        std::vector<float> hc_scale(3), hc_base(n3);

        /* `mixes` is what the projection produced; the device multiplies it by `inv` (the
         * residual RMS reciprocal) and by hc_scale before the softmax, so the SPREAD has to be
         * placed such that the product lands where we want it. Residual entries are O(1), so
         * inv ~ 1, and hc_scale carries the spread directly. */
        for (auto& v : mixes) v = (float)frand(st);
        for (auto& v : residual) v = f2b((float)(frand(st) * 2.0));
        hc_scale[0] = 1.0f;
        hc_scale[1] = 1.0f;
        hc_scale[2] = (float)(spread * 2.0);
        for (auto& v : hc_base) v = (float)(frand(st) * spread);

        float *d_mixes = up(mixes), *d_scale = up(hc_scale), *d_base = up(hc_base);
        bf16* d_resid = up(residual);
        float *d_pm = nullptr, *d_cm[3] = {nullptr, nullptr, nullptr};
        bf16* d_li = nullptr;
        ck(hipMalloc(&d_pm, (size_t)T * N * 4 + 16));
        for (int a = 0; a < 3; a++) ck(hipMalloc(&d_cm[a], (size_t)T * N * N * 4 + 16));
        ck(hipMalloc(&d_li, (size_t)T * HIDDEN * 2 + 16));

        const unsigned grid = 7; /* deliberately not a divisor of T — exercises a ragged nblk */
        hc_run_arm0(grid, d_pm, d_cm[0], d_li, d_mixes, d_resid, d_scale, d_base, T, HIDDEN, REPEAT,
                    RMS_EPS, HC_EPS);
        hc_run_arm1(grid, d_pm, d_cm[1], d_li, d_mixes, d_resid, d_scale, d_base, T, HIDDEN, REPEAT,
                    RMS_EPS, HC_EPS);
        hc_run_arm2(grid, d_pm, d_cm[2], d_li, d_mixes, d_resid, d_scale, d_base, T, HIDDEN, REPEAT,
                    RMS_EPS, HC_EPS);
        ck(hipDeviceSynchronize());

        std::vector<float> got[3];
        for (int a = 0; a < 3; a++) {
            got[a].resize((size_t)T * N * N);
            ck(hipMemcpy(got[a].data(), d_cm[a], got[a].size() * 4, hipMemcpyDeviceToHost));
        }

        /* The oracle, per token. `inv` is recomputed here the way the device computes it — an f32
         * sum of squares over n*hidden, then rsqrt — because the arms do not touch that reduction
         * and a f64 version of it would inject an error this test is not measuring. */
        std::vector<double> want((size_t)T * N * N);
        for (unsigned t = 0; t < T; t++) {
            float ss = 0.0f;
            for (unsigned i = 0; i < N * HIDDEN; i++) {
                const float x = b2f(residual[(size_t)t * N * HIDDEN + i]);
                ss += x * x;
            }
            const double inv = 1.0 / std::sqrt((double)(ss / (float)(N * HIDDEN)) + RMS_EPS);
            double raw[16];
            for (unsigned i = 0; i < N; i++)
                for (unsigned j = 0; j < N; j++)
                    raw[i * N + j] =
                        (double)mixes[(size_t)t * n3 + 2 * N + i * N + j] * inv * hc_scale[2] +
                        hc_base[2 * N + i * N + j];
            sinkhorn_f64(raw, &want[(size_t)t * N * N], N, REPEAT, (double)HC_EPS);
        }

        const Err e0 = cmp(got[0], want);
        for (int a = 0; a < 3; a++) {
            const Err e = cmp(got[a], want);
            double vs0 = 0.0;
            for (size_t i = 0; i < got[a].size(); i++)
                vs0 = std::fmax(vs0, std::fabs((double)got[a][i] - (double)got[0][i]));
            const double ratio = e0.mx > 0.0 ? e.mx / e0.mx : (e.mx > 0.0 ? 1e9 : 1.0);
            /* TWO BARS, and the second is the one that means something.
             *
             * ORACLE_BAR (2e-5) bounds the whole f32 pipeline — rsqrt, exp, 20 normalize passes —
             * against f64. It is LOOSE on purpose: the SHIPPED arm already measures 5.4e-6 at
             * spread 8, where the initial softmax is nearly one-hot and 20 passes of
             * far-from-1 denominators compound. A tight absolute bar here would fail arm 0, i.e.
             * it would be a statement about f32 and not about this change.
             *
             * The ratio is the real gate: a reciprocal-and-multiply may not move the body
             * FURTHER from the truth than the divide it replaced. Measured 0.97 .. 1.12 across
             * all three spreads; 1.5 leaves headroom without letting a real regression through. */
            const bool ok = e.mx < 2e-5 && (a == 0 || ratio <= 1.5);
            pass = pass && ok;
            std::printf("  %-8.2f arm%d   %12.3e %12.3e %12.3e %12.2f  %s\n", spread, a, e.mx,
                        e.rms, vs0, ratio, ok ? "" : "  <-- FAIL");
        }

        ck(hipFree(d_mixes));
        ck(hipFree(d_scale));
        ck(hipFree(d_base));
        ck(hipFree(d_resid));
        ck(hipFree(d_pm));
        ck(hipFree(d_li));
        for (int a = 0; a < 3; a++) ck(hipFree(d_cm[a]));
    }

    std::printf("\n%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
#endif /* HC_RCP_ARM == 0 */

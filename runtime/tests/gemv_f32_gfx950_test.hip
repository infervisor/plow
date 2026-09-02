/* gemv_f32_gfx950_test.hip — fp32-output GEMV (op 128, DevOp::GemvF32) hardware test.
 *
 * GLM-5.3-Flash's DSA indexer needs its `index_kpool_compress_gate` projection at fp32
 * precision (IndexScoreKpool's `W` operand — see op_gemm.h's `d_gemv_f32` header and
 * crates/packet/src/dev.rs's `DevOp::GemvF32` doc comment for why: the reference computes
 * this exact weight in fp32 throughout, a documented accuracy requirement, not laziness).
 * This op is pure arithmetic (a dot product, no quantization/rounding tricks) so unlike the
 * DSA-pool ops it needs no vLLM-generated oracle — the correctness question is "does the
 * kernel compute a correct fp32 dot product," checked here against an independently
 * written f64 CPU reference computed in this same test file.
 *
 * Two passes, the op body is device-only by include chain:
 *   hipcc --offload-arch=gfx950 -O3 -w -DGF32_DEVICE --genco \
 *       runtime/tests/gemv_f32_gfx950_test.hip -o /tmp/gf32.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/gemv_f32_gfx950_test.hip -o /tmp/gf32 -L/opt/rocm/lib -lamdhip64
 *   perf-data/tools/gpulease -n 1 gf32 'sg render -c "/tmp/gf32 /tmp/gf32.co"'
 */

#ifdef GF32_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_gemm.h"

struct GArgs {
    unsigned long long c, x, w;
    unsigned m, n, k;
};
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_gemv_f32(GArgs* a) {
    d_gemv_f32((float*)a->c, (const bf16*)a->x, (const float*)a->w, a->m, a->n, a->k,
              blockIdx.x, gridDim.x);
}

#else
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

typedef unsigned short bf16;
static bf16 f2b(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static float b2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
#define HC(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); return 1; }        \
    } while (0)

struct GArgs {
    unsigned long long c, x, w;
    unsigned m, n, k;
};

/* Small deterministic PRNG (xorshift32) -- no external dependency, reproducible. */
static unsigned rng_state = 0x9e3779b9u;
static float rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & 0xFFFFu) / 65535.0f - 0.5f) * 4.0f; /* [-2, 2) */
}

int main(int argc, char** argv) {
    if (argc != 2) { printf("usage: %s <gf32.co>\n", argv[0]); return 1; }
    HC(hipSetDevice(0));
    hipModule_t mod;
    HC(hipModuleLoad(&mod, argv[1]));
    hipFunction_t f_gemv;
    HC(hipModuleGetFunction(&f_gemv, mod, "k_gemv_f32"));

    const unsigned M = 3, N = 32, K = 300; /* non-round K/N on purpose: no special-case masking */

    std::vector<float> x_f(M * K), w_f(N * K);
    std::vector<bf16> x_h(M * K);
    for (unsigned i = 0; i < M * K; i++) {
        x_f[i] = rnd();
        x_h[i] = f2b(x_f[i]);
    }
    for (unsigned i = 0; i < N * K; i++) w_f[i] = rnd();

    /* Independently-written f64 CPU reference. Compares against the bf16-ROUND-TRIPPED x
     * (matching what the kernel actually reads), so the tolerance below is measuring the
     * kernel's own reduction accuracy, not bf16 quantization noise from x. */
    std::vector<double> want(M * N);
    for (unsigned m = 0; m < M; m++)
        for (unsigned n = 0; n < N; n++) {
            double acc = 0.0;
            for (unsigned k = 0; k < K; k++)
                acc += (double)b2f(x_h[m * K + k]) * (double)w_f[n * K + k];
            want[m * N + n] = acc;
        }

    void *d_x, *d_w, *d_c;
    HC(hipMalloc(&d_x, M * K * sizeof(bf16)));
    HC(hipMalloc(&d_w, N * K * sizeof(float)));
    HC(hipMalloc(&d_c, M * N * sizeof(float)));
    HC(hipMemcpy(d_x, x_h.data(), M * K * sizeof(bf16), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_w, w_f.data(), N * K * sizeof(float), hipMemcpyHostToDevice));

    GArgs ga = {};
    ga.c = (unsigned long long)d_c;
    ga.x = (unsigned long long)d_x;
    ga.w = (unsigned long long)d_w;
    ga.m = M;
    ga.n = N;
    ga.k = K;
    void* d_ga;
    HC(hipMalloc(&d_ga, sizeof(ga)));
    HC(hipMemcpy(d_ga, &ga, sizeof(ga), hipMemcpyHostToDevice));
    void* params[] = {&d_ga};
    HC(hipModuleLaunchKernel(f_gemv, 4, 1, 1, 512, 1, 1, 0, 0, params, nullptr));
    HC(hipDeviceSynchronize());

    std::vector<float> got(M * N);
    HC(hipMemcpy(got.data(), d_c, M * N * sizeof(float), hipMemcpyDeviceToHost));

    /* Combined absolute+relative tolerance (the standard `allclose` shape), not pure
     * relative error: a K=300-term fp32 tree reduction (GPU, wave_sum) compared against an
     * f64 linear-order CPU reference legitimately disagrees by a few ULPs of the
     * ACCUMULATION, and when the true sum is near zero (heavy cancellation among ~300
     * terms in [-2,2)) that tiny absolute noise is a huge relative error despite the
     * kernel being correct -- confirmed here (max |got-want| ~= 3e-6 across the whole
     * output; the flagged case at pure-relative 1e-5 has |want|~=3e-4). atol is sized to
     * ~K*eps_fp32*max_term_magnitude (300*1.2e-7*4 ~= 1.4e-4); rtol covers real
     * reduction-order fp32 error for the large-magnitude outputs. */
    int bad = 0;
    double max_rel = 0.0;
    const double atol = 2e-4, rtol = 1e-4;
    for (unsigned i = 0; i < M * N; i++) {
        double g = got[i], w = want[i];
        double rel = fabs(g - w) / (fabs(w) > 1e-6 ? fabs(w) : 1.0);
        if (rel > max_rel) max_rel = rel;
        if (fabs(g - w) > atol + rtol * fabs(w)) {
            if (bad < 5) printf("MISMATCH C[%u]: got %.9g want %.9g (rel %.6g)\n", i, g, w, rel);
            bad++;
        }
    }
    printf("GemvF32: %d mismatch(es) over %u outputs, max relerr %.6g\n", bad, M * N, max_rel);
    return bad ? 1 : 0;
}
#endif

/* hyperconn_gfx950_test.hip — GLM-5.3-Flash's hyper-connections (ops 121/122) against the
 * numerical oracle at /home/shaswot/plow-work/mhc_oracle.npz, built by running vLLM's real
 * reference math (mhc_pre_torch/mhc_post_torch) directly — not derived from prose. See
 * op_hyperconn.h's header and DevOp::HyperConnPre/Post's doc comments (crates/packet/src/
 * dev.rs) for the full contract this kernel implements.
 *
 * This op's INPUT is `mixes`, the output of an ordinary Gemv/Gemm projection
 * (`residual_flat @ fn.T`) — that projection is deliberately NOT this op's job (see the
 * header). The oracle only records `residual`/`fn` (mhc_pre_torch's own inputs), so this
 * test computes `mixes` itself on the host with a plain f64 matmul before launching the
 * device kernel, exactly mirroring what the real Gemv packet ahead of this one would produce.
 *
 * Two passes, op_hyperconn.h is device-only by include chain:
 *   hipcc --offload-arch=gfx950 -O3 -w -DHYPERCONN_DEVICE --genco \
 *       runtime/tests/hyperconn_gfx950_test.hip -o /tmp/hc.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/hyperconn_gfx950_test.hip -o /tmp/hc -L/opt/rocm/lib -lamdhip64
 *   perf-data/tools/gpulease -n 1 hc 'sg render -c "/tmp/hc /tmp/hc.co"'
 */

#ifdef HYPERCONN_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_hyperconn.h"

struct HArgs {
    unsigned long long post_mix, comb_mix, layer_input, mixes, residual, hc_scale, hc_base;
    unsigned long long new_residual, x_out;
    unsigned T, N, HIDDEN, SINKHORN;
    float rms_eps, hc_eps;
};

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_pre(HArgs* a) {
    __shared__ float part[PLOW_WAVES];
    d_hyperconn_pre((float*)a->post_mix, (float*)a->comb_mix, (bf16*)a->layer_input,
                    (const float*)a->mixes, (const bf16*)a->residual, (const float*)a->hc_scale,
                    (const float*)a->hc_base, a->T, a->N, a->HIDDEN, a->SINKHORN, a->rms_eps,
                    a->hc_eps, blockIdx.x, gridDim.x, part);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_post(HArgs* a) {
    d_hyperconn_post((bf16*)a->new_residual, (const bf16*)a->x_out, (const bf16*)a->residual,
                     (const float*)a->post_mix, (const float*)a->comb_mix, a->T, a->N, a->HIDDEN,
                     blockIdx.x, gridDim.x);
}

#else
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "hyperconn_oracle_fixture.h"

typedef unsigned short bf16;
static float b2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static bf16 f2b(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
#define HC(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); return 1; }        \
    } while (0)

struct HArgs {
    unsigned long long post_mix, comb_mix, layer_input, mixes, residual, hc_scale, hc_base;
    unsigned long long new_residual, x_out;
    unsigned T, N, HIDDEN, SINKHORN;
    float rms_eps, hc_eps;
};

int main(int argc, char** argv) {
    if (argc != 2) { printf("usage: %s <hyperconn.co>\n", argv[0]); return 1; }

    /* Round the oracle's f32 residual to bf16, exactly as the device sees it (the oracle
     * itself started from a bf16 tensor, so this should be a no-op — asserted below). */
    std::vector<bf16> residual_bf(MHC_T * MHC_N * MHC_HIDDEN);
    for (int i = 0; i < MHC_T * MHC_N * MHC_HIDDEN; i++) {
        residual_bf[i] = f2b(mhc_residual[i]);
        if (fabsf(b2f(residual_bf[i]) - mhc_residual[i]) > 1e-6f)
            printf("WARN: residual[%d] not already bf16-exact (%.9g vs %.9g) — oracle/device "
                   "rounding may not match exactly\n",
                   i, b2f(residual_bf[i]), mhc_residual[i]);
    }

    /* mixes = residual_flat[T, N*HIDDEN] @ fn.T[N*HIDDEN, N3] — the projection this op does
     * NOT do; computed here in f64 exactly as the real Gemv packet ahead of it would produce
     * (modulo the Gemv's own bf16xbf16->f32 accumulation, which residual's bf16-exactness
     * above makes moot for this input). */
    std::vector<float> mixes(MHC_T * MHC_N3);
    for (int t = 0; t < MHC_T; t++)
        for (int j = 0; j < MHC_N3; j++) {
            double acc = 0.0;
            for (int k = 0; k < MHC_N * MHC_HIDDEN; k++)
                acc += (double)mhc_residual[t * MHC_N * MHC_HIDDEN + k] * (double)mhc_fn[j * MHC_N * MHC_HIDDEN + k];
            mixes[t * MHC_N3 + j] = (float)acc;
        }

    HC(hipSetDevice(0));
    hipModule_t mod;
    HC(hipModuleLoad(&mod, argv[1]));
    hipFunction_t f_pre, f_post;
    HC(hipModuleGetFunction(&f_pre, mod, "k_pre"));
    HC(hipModuleGetFunction(&f_post, mod, "k_post"));

    void *d_post_mix, *d_comb_mix, *d_layer_input, *d_mixes, *d_residual, *d_hc_scale, *d_hc_base;
    void *d_new_residual, *d_x_out;
    HC(hipMalloc(&d_post_mix, MHC_T * MHC_N * sizeof(float)));
    HC(hipMalloc(&d_comb_mix, MHC_T * MHC_N * MHC_N * sizeof(float)));
    HC(hipMalloc(&d_layer_input, MHC_T * MHC_HIDDEN * sizeof(bf16)));
    HC(hipMalloc(&d_mixes, MHC_T * MHC_N3 * sizeof(float)));
    HC(hipMalloc(&d_residual, MHC_T * MHC_N * MHC_HIDDEN * sizeof(bf16)));
    HC(hipMalloc(&d_hc_scale, 3 * sizeof(float)));
    HC(hipMalloc(&d_hc_base, MHC_N3 * sizeof(float)));
    HC(hipMalloc(&d_new_residual, MHC_T * MHC_N * MHC_HIDDEN * sizeof(bf16)));
    HC(hipMalloc(&d_x_out, MHC_T * MHC_HIDDEN * sizeof(bf16)));

    HC(hipMemcpy(d_mixes, mixes.data(), mixes.size() * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_residual, residual_bf.data(), residual_bf.size() * sizeof(bf16), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_hc_scale, mhc_hc_scale, 3 * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_hc_base, mhc_hc_base, MHC_N3 * sizeof(float), hipMemcpyHostToDevice));
    std::vector<bf16> x_out_bf(MHC_T * MHC_HIDDEN);
    for (int i = 0; i < MHC_T * MHC_HIDDEN; i++) x_out_bf[i] = f2b(mhc_x_out[i]);
    HC(hipMemcpy(d_x_out, x_out_bf.data(), x_out_bf.size() * sizeof(bf16), hipMemcpyHostToDevice));

    HArgs args = {};
    args.post_mix = (unsigned long long)d_post_mix;
    args.comb_mix = (unsigned long long)d_comb_mix;
    args.layer_input = (unsigned long long)d_layer_input;
    args.mixes = (unsigned long long)d_mixes;
    args.residual = (unsigned long long)d_residual;
    args.hc_scale = (unsigned long long)d_hc_scale;
    args.hc_base = (unsigned long long)d_hc_base;
    args.new_residual = (unsigned long long)d_new_residual;
    args.x_out = (unsigned long long)d_x_out;
    args.T = MHC_T;
    args.N = MHC_N;
    args.HIDDEN = MHC_HIDDEN;
    args.SINKHORN = MHC_SINKHORN_REPEAT;
    args.rms_eps = MHC_RMS_EPS;
    args.hc_eps = MHC_HC_EPS;

    void* d_args;
    HC(hipMalloc(&d_args, sizeof(HArgs)));
    HC(hipMemcpy(d_args, &args, sizeof(HArgs), hipMemcpyHostToDevice));
    void* kparams[] = {&d_args};

    HC(hipModuleLaunchKernel(f_pre, 1, 1, 1, 512, 1, 1, 0, 0, kparams, nullptr));
    HC(hipDeviceSynchronize());
    HC(hipModuleLaunchKernel(f_post, 1, 1, 1, 512, 1, 1, 0, 0, kparams, nullptr));
    HC(hipDeviceSynchronize());

    std::vector<float> post_mix(MHC_T * MHC_N), comb_mix(MHC_T * MHC_N * MHC_N);
    std::vector<bf16> layer_input_bf(MHC_T * MHC_HIDDEN), new_residual_bf(MHC_T * MHC_N * MHC_HIDDEN);
    HC(hipMemcpy(post_mix.data(), d_post_mix, post_mix.size() * sizeof(float), hipMemcpyDeviceToHost));
    HC(hipMemcpy(comb_mix.data(), d_comb_mix, comb_mix.size() * sizeof(float), hipMemcpyDeviceToHost));
    HC(hipMemcpy(layer_input_bf.data(), d_layer_input, layer_input_bf.size() * sizeof(bf16), hipMemcpyDeviceToHost));
    HC(hipMemcpy(new_residual_bf.data(), d_new_residual, new_residual_bf.size() * sizeof(bf16), hipMemcpyDeviceToHost));

    int bad = 0;
    double max_err = 0.0;
#define CHECK(name, got, want, tol)                                                              \
    do {                                                                                         \
        double d_ = fabs((double)(got) - (double)(want));                                        \
        if (d_ > max_err) max_err = d_;                                                          \
        if (d_ > (tol)) { printf("MISMATCH %s: got %.9g want %.9g (|d|=%.3g)\n", name, (double)(got), (double)(want), d_); bad++; } \
    } while (0)

    for (int i = 0; i < MHC_T * MHC_N; i++)
        CHECK("post_mix", post_mix[i], mhc_expect_post_mix[i], 1e-3);
    for (int i = 0; i < MHC_T * MHC_N * MHC_N; i++)
        CHECK("comb_mix", comb_mix[i], mhc_expect_comb_mix[i], 1e-3);
    for (int i = 0; i < MHC_T * MHC_HIDDEN; i++)
        CHECK("layer_input", b2f(layer_input_bf[i]), mhc_expect_layer_input[i], 3e-2 /* bf16 quant */);
    for (int i = 0; i < MHC_T * MHC_N * MHC_HIDDEN; i++)
        CHECK("new_residual", b2f(new_residual_bf[i]), mhc_expect_new_residual[i], 3e-2 /* bf16 quant */);

    printf("hyperconn oracle check: %d mismatch(es), max |err| = %.6g\n", bad, max_err);
    return bad ? 1 : 0;
}
#endif

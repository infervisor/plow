/* Exact two-output GemvQkv(Nv=0) oracle for the KDA decode projection shape.
 *
 * Device:
 *   ROCM_ROOT=$(dirname $(dirname $(command -v hipcc)))
 *   hipcc --offload-arch=gfx950 -O3 -w -DPLOW_GEMV_MM=1 -DQKV2_DEVICE --genco \
 *     -Rpass-analysis=kernel-resource-usage runtime/tests/gemv_qkv_nv0_gfx950_test.hip \
 *     -o /tmp/qkv2.bundle -Iruntime/amd -Iruntime/common 2>/tmp/qkv2.resources
 *   $ROCM_ROOT/lib/llvm/bin/clang-offload-bundler --unbundle -type=o -input=/tmp/qkv2.bundle \
 *     -targets=hipv4-amdgcn-amd-amdhsa--gfx950 -output=/tmp/qkv2.co
 * Host:
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I$ROCM_ROOT/include \
 *     runtime/tests/gemv_qkv_nv0_gfx950_test.hip -o /tmp/qkv2_test \
 *     -L$ROCM_ROOT/lib -lamdhip64
 *
 * The grids match the emitted blocked-column fixed points: 128 and 12 workgroups for the two
 * control packets, 140 for their fused output-axis concatenation. The timing compares the whole
 * two-launch control sequence against the one-launch candidate.
 *
 * Measured on MI355X/gfx950, 8 alternating folds x 5000 iterations after 100 warmups:
 * control 7.8879 us, fused 3.5032 us, delta -4.3847 us (-55.59%), 0 differing bf16 outputs.
 */

struct KArgs {
    unsigned long long c0, c1, c2, x, w0, w1, w2;
    unsigned m, n0, n1, n2, k;
};

#ifdef QKV2_DEVICE
#include <hip/hip_runtime.h>
#include "op_gemm.h"

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_gemv(const KArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv((bf16*)a->c0, (const bf16*)a->x, (const bf16*)a->w0, nullptr, nullptr, a->m, a->n0,
           a->k, 0, 0.0f, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_qkv2(const KArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_qkv((bf16*)a->c0, (bf16*)a->c1, (bf16*)a->c2, (const bf16*)a->x,
               (const bf16*)a->w0, (const bf16*)a->w1, (const bf16*)a->w2, a->m, a->n0, a->n1,
               a->n2, a->k, blockIdx.x, gridDim.x, lds);
}

#else
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(e)                                                                                  \
    do {                                                                                           \
        hipError_t _e = (e);                                                                       \
        if (_e != hipSuccess) {                                                                    \
            printf("FAIL %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(_e));                 \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static constexpr unsigned M = 1, K = 7168, N0 = 128, N1 = 12;
static constexpr unsigned B0 = 128, B1 = 12, BF = 140, THREADS = 512;
static unsigned rng_state = 0x9e3779b9u;

static unsigned short f2bf(float f) {
    unsigned u;
    memcpy(&u, &f, sizeof(u));
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf2f(unsigned short h) {
    unsigned u = (unsigned)h << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}
static float random_float() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 8) & 0xffffu) / 32768.0f - 1.0f;
}
static void* alloc(size_t bytes) {
    void* p = nullptr;
    HIP_OK(hipMalloc(&p, bytes));
    HIP_OK(hipMemset(p, 0, bytes));
    return p;
}
static std::vector<unsigned short> fill(void* dst, size_t n) {
    std::vector<unsigned short> h(n);
    for (auto& v : h) v = f2bf(random_float());
    HIP_OK(hipMemcpy(dst, h.data(), n * sizeof(h[0]), hipMemcpyHostToDevice));
    return h;
}
static std::vector<unsigned short> read(void* src, size_t n) {
    std::vector<unsigned short> h(n);
    HIP_OK(hipMemcpy(h.data(), src, n * sizeof(h[0]), hipMemcpyDeviceToHost));
    return h;
}
static size_t compare(const char* name, const std::vector<unsigned short>& got,
                      const std::vector<unsigned short>& want) {
    size_t different = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == want[i]) continue;
        ++different;
        max_abs = std::max(max_abs, fabs((double)bf2f(got[i]) - bf2f(want[i])));
    }
    printf("  %-18s differing halves %zu / %zu, max|d| %.6g\n", name, different, got.size(),
           max_abs);
    return different;
}

static hipFunction_t gemv, qkv2;
static void *a0, *a1, *af;
static void launch(hipFunction_t f, void* args, unsigned blocks) {
    size_t size = sizeof(void*);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
    HIP_OK(hipModuleLaunchKernel(f, blocks, 1, 1, THREADS, 1, 1, 0, nullptr, nullptr, config));
}
static void control() {
    launch(gemv, a0, B0);
    launch(gemv, a1, B1);
}
static void fused() { launch(qkv2, af, BF); }
static void* put(const KArgs& a) {
    void* p = alloc(sizeof(a));
    HIP_OK(hipMemcpy(p, &a, sizeof(a), hipMemcpyHostToDevice));
    return p;
}
static double time_us(void (*body)(), unsigned iters) {
    hipEvent_t begin, end;
    HIP_OK(hipEventCreate(&begin));
    HIP_OK(hipEventCreate(&end));
    for (unsigned i = 0; i < 100; ++i) body();
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipEventRecord(begin));
    for (unsigned i = 0; i < iters; ++i) body();
    HIP_OK(hipEventRecord(end));
    HIP_OK(hipEventSynchronize(end));
    float ms = 0.0f;
    HIP_OK(hipEventElapsedTime(&ms, begin, end));
    HIP_OK(hipEventDestroy(begin));
    HIP_OK(hipEventDestroy(end));
    return 1000.0 * ms / iters;
}

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/qkv2.co";
    const unsigned iters = argc > 2 ? (unsigned)atoi(argv[2]) : 3000;
    hipDeviceProp_t prop;
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    hipModule_t module;
    HIP_OK(hipModuleLoad(&module, object));
    HIP_OK(hipModuleGetFunction(&gemv, module, "k_gemv"));
    HIP_OK(hipModuleGetFunction(&qkv2, module, "k_qkv2"));

    void* x = alloc(K * 2);
    void* w0 = alloc((size_t)N0 * K * 2);
    void* w1 = alloc((size_t)N1 * K * 2);
    void* c0 = alloc(N0 * 2);
    void* c1 = alloc(N1 * 2);
    void* c2 = alloc(N1 * 2);
    fill(x, K);
    fill(w0, (size_t)N0 * K);
    fill(w1, (size_t)N1 * K);

    KArgs x0{(unsigned long long)c0, 0, 0, (unsigned long long)x, (unsigned long long)w0, 0, 0,
             M, N0, 0, 0, K};
    KArgs x1{(unsigned long long)c1, 0, 0, (unsigned long long)x, (unsigned long long)w1, 0, 0,
             M, N1, 0, 0, K};
    KArgs xf{(unsigned long long)c0, (unsigned long long)c1, (unsigned long long)c2,
             (unsigned long long)x, (unsigned long long)w0, (unsigned long long)w1, 0,
             M, N0, N1, 0, K};
    a0 = put(x0);
    a1 = put(x1);
    af = put(xf);

    control();
    HIP_OK(hipDeviceSynchronize());
    const auto ref0 = read(c0, N0), ref1 = read(c1, N1);
    HIP_OK(hipMemset(c0, 0, N0 * 2));
    HIP_OK(hipMemset(c1, 0, N1 * 2));
    HIP_OK(hipMemset(c2, 0xee, N1 * 2));
    fused();
    HIP_OK(hipDeviceSynchronize());

    printf("gemv_qkv_nv0_gfx950_test: %s, M=%u K=%u N=%u+%u, WG=%u, blocks=%u+%u vs %u\n",
           prop.gcnArchName, M, K, N0, N1, THREADS, B0, B1, BF);
    size_t bad = compare("output 0", read(c0, N0), ref0);
    bad += compare("output 1", read(c1, N1), ref1);
    const auto poison = read(c2, N1);
    size_t touched = 0;
    for (auto v : poison) touched += v != 0xeeeeu;
    printf("  %-18s %zu halves written (must be 0)\n", "Nv=0 output", touched);
    bad += touched;

    printf("timing: 100 warmups, %u iterations per cell, alternating sequence/fused order\n", iters);
    std::vector<double> tc, tf;
    for (unsigned fold = 0; fold < 8; ++fold) {
        const bool candidate_first = fold & 1;
        double c, f;
        if (candidate_first) {
            f = time_us(fused, iters);
            c = time_us(control, iters);
        } else {
            c = time_us(control, iters);
            f = time_us(fused, iters);
        }
        tc.push_back(c);
        tf.push_back(f);
        printf("  fold %u %s: control(two) %.4f us, fused %.4f us, delta %+.4f us (%+.2f%%)\n",
               fold + 1, candidate_first ? "F->C" : "C->F", c, f, f - c, 100.0 * (f - c) / c);
    }
    std::sort(tc.begin(), tc.end());
    std::sort(tf.begin(), tf.end());
    const double cm = (tc[3] + tc[4]) / 2.0, fm = (tf[3] + tf[4]) / 2.0;
    printf("median: control(two) %.4f us, fused %.4f us, delta %+.4f us (%+.2f%%)\n", cm, fm,
           fm - cm, 100.0 * (fm - cm) / cm);
    printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
#endif

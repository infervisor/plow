/* gemv_qkvg_gfx950_test.hip — the FUSED q|k|v|g decode GEMV, checked and timed on hardware.
 *
 * `DevOp::GemvQkvg` merges Kimi-K3's four KDA projections (q_proj, k_proj, v_proj and the
 * full-rank output gate g_proj — all [12288, 7168] read against the same pre-normed x[7168])
 * into ONE packet along the OUTPUT dimension. This file exists to hold that claim to three
 * numbers, because the module doc that proposed the fusion could assert none of them:
 *
 *   1. CORRECTNESS. The fused sweep must agree with four independent `d_gemv` calls
 *      ELEMENTWISE, not statistically. Both accumulate the same K chunks in the same order
 *      (the unroll factor batches the LOADS, it does not reorder the adds), so the bar here is
 *      BIT-IDENTICAL and anything else is a bug rather than rounding.
 *   2. THE 3-STREAM PATH IS UNCHANGED. `d_gemv_qkv` (op 22, shipping) is now a forwarder into
 *      the 4-stream body with Ng = 0. Sharing one body is what keeps the interpreter from
 *      inlining a second copy and paying for it in registers — and it is also exactly the kind
 *      of change that silently perturbs an op nobody re-tested. So op 22 is re-checked against
 *      the same reference, and its g_out is POISONED first to prove the dead arm stays dead.
 *   3. IS IT ACTUALLY FASTER. The four projections were ALREADY 256-CU packets, so each CU ran
 *      4 x 48 columns back-to-back down its own stream either way. The fusion cannot move fewer
 *      bytes; what it removes is three counter gates and three redundant LDS stagings of x.
 *      That is a small number and it deserves to be printed rather than assumed.
 *
 * A caveat this file cannot escape and does not pretend to: it measures BARE KERNELS, so the
 * three counter gates the fusion deletes are in NEITHER column. The gap reported is the launch
 * plus staging saving only, and it is a LOWER BOUND on the in-interpreter win.
 *
 * TWO PASSES, because `op_gemm.h` is DEVICE-ONLY. Its bodies hand `address_space(1)` pointers
 * to helpers declared generic, which the device compiler accepts and the host pass rejects —
 * `interp.hip` never notices because it is only ever built with `--genco`. So the device half
 * and the host half of this test are compiled separately and joined by the HIP module API,
 * which is also how plowrt loads the real interpreter.
 *
 *   hipcc --offload-arch=gfx950 -O3 -w -DPLOW_GEMV_MM=4 -DQKVG_DEVICE --genco \
 *       runtime/tests/gemv_qkvg_gfx950_test.hip -o /tmp/qkvg.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/gemv_qkvg_gfx950_test.hip -o /tmp/qkvg_test -L/opt/rocm/lib -lamdhip64
 *   sg render -c '/tmp/qkvg_test /tmp/qkvg.co'
 *
 * MEASURED, gfx950 / 256 CU, PLOW_GEMV_MM=4:
 *   fused vs 4 separate  0 differing halves of 4 x 12288 (bit-identical), and op 22 likewise
 *   4 separate GEMVs     0.1511 ms   4663 GB/s
 *   1 fused GemvQkvg     0.1108 ms   6362 GB/s     -26.7% (three runs: -22.7 / -25.6 / -26.7)
 */

/* ONE argument per kernel, a pointer to this struct in device memory. The alternative is
 * packing 15 scalars into a HIP_LAUNCH_PARAM buffer whose layout must be kept in step with the
 * kernel signature by hand — a silent-corruption shape this tree has enough of already. */
struct KArgs {
    unsigned long long Cq, Ck, Cv, Cg, x, Wq, Wk, Wv, Wg;
    unsigned M, Nq, Nk, Nv, Ng, K;
};

#ifdef QKVG_DEVICE
/* ======================= DEVICE HALF (--genco) ======================= */
#include <hip/hip_runtime.h>
#include "op_gemm.h"

/* The interpreter's LDS arena, verbatim — these kernels stand in for its dispatch arms. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_gemv(const KArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv((bf16*)a->Cq, (const bf16*)a->x, (const bf16*)a->Wq, nullptr, nullptr, a->M, a->Nq,
           a->K, 0, 0.0f, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_qkvg(const KArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_qkvg((bf16*)a->Cq, (bf16*)a->Ck, (bf16*)a->Cv, (bf16*)a->Cg, (const bf16*)a->x,
                (const bf16*)a->Wq, (const bf16*)a->Wk, (const bf16*)a->Wv, (const bf16*)a->Wg,
                a->M, a->Nq, a->Nk, a->Nv, a->Ng, a->K, blockIdx.x, gridDim.x, lds);
}

/* op 22, through the forwarder — the regression guard on the shared body. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_qkv(const KArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_qkv((bf16*)a->Cq, (bf16*)a->Ck, (bf16*)a->Cv, (const bf16*)a->x, (const bf16*)a->Wq,
               (const bf16*)a->Wk, (const bf16*)a->Wv, a->M, a->Nq, a->Nk, a->Nv, a->K,
               blockIdx.x, gridDim.x, lds);
}

#else
/* ========================= HOST HALF (-x c++) ========================= */
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(e)                                                                                  \
    do {                                                                                           \
        hipError_t _e = (e);                                                                       \
        if (_e != hipSuccess) {                                                                    \
            printf("FAIL %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(_e));                  \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

/* K3's KDA geometry: hidden 7168, H*D = 96*128 = 12288, decode M = 1, 256 CUs. */
static const unsigned K3_K = 7168, K3_N = 12288, K3_M = 1, NBLK = 256, NTHREADS = 512;

static unsigned rng_state = 0x2545F491u;
static float frand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}
/* bf16 round-to-nearest-even, so the host reference sees EXACTLY the device's operand bits. */
static unsigned short f2bf_h(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static float bf2f_h(unsigned short h) {
    unsigned u = (unsigned)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void* dev_alloc(size_t bytes) {
    void* p = nullptr;
    HIP_OK(hipMalloc(&p, bytes));
    HIP_OK(hipMemset(p, 0, bytes));
    return p;
}
static std::vector<unsigned short> fill_bf16(void* d, size_t n) {
    std::vector<unsigned short> h(n);
    for (size_t i = 0; i < n; i++) h[i] = f2bf_h(frand());
    HIP_OK(hipMemcpy(d, h.data(), n * 2, hipMemcpyHostToDevice));
    return h;
}
static std::vector<unsigned short> read_bf16(void* d, size_t n) {
    std::vector<unsigned short> h(n);
    HIP_OK(hipMemcpy(h.data(), d, n * 2, hipMemcpyDeviceToHost));
    return h;
}

/* Elementwise comparison. Reports the worst absolute and relative gap alongside the count, so a
 * NON-zero result can be judged rather than only counted. */
static size_t cmp(const char* what, const std::vector<unsigned short>& a,
                  const std::vector<unsigned short>& b) {
    size_t ndiff = 0;
    double worst_abs = 0, worst_rel = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == b[i]) continue;
        ndiff++;
        const double x = bf2f_h(a[i]), y = bf2f_h(b[i]), d = fabs(x - y);
        if (d > worst_abs) worst_abs = d;
        const double den = fabs(y) > 1e-30 ? fabs(y) : 1e-30;
        if (d / den > worst_rel) worst_rel = d / den;
    }
    printf("  %-26s differing halves %8zu / %zu   max|d| %.6g   max rel %.6g\n", what, ndiff,
           a.size(), worst_abs, worst_rel);
    return ndiff;
}

/* Absolute accuracy against an f64 host dot product for ONE output column. Fused and unfused
 * share this, so it prices the bf16 GEMV itself rather than the fusion. */
static double col_err(const std::vector<unsigned short>& C, const std::vector<unsigned short>& x,
                      const std::vector<unsigned short>& W, unsigned n, unsigned K) {
    double acc = 0;
    for (unsigned k = 0; k < K; k++)
        acc += (double)bf2f_h(x[k]) * (double)bf2f_h(W[(size_t)n * K + k]);
    return fabs((double)bf2f_h(C[n]) - acc) / (fabs(acc) > 1e-30 ? fabs(acc) : 1e-30);
}

static hipFunction_t f_gemv, f_qkv, f_qkvg;
static void* a_q;   /* per-projection KArgs for the four separate GEMVs */
static void* a_k;
static void* a_v;
static void* a_g;
static void* a_fused;
static void* a_qkv3;

static void launch(hipFunction_t f, void* args_dev) {
    size_t sz = sizeof(void*);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_dev, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                   HIP_LAUNCH_PARAM_END};
    HIP_OK(hipModuleLaunchKernel(f, NBLK, 1, 1, NTHREADS, 1, 1, 0, nullptr, nullptr, cfg));
}

static void run_four_separate() {
    launch(f_gemv, a_q);
    launch(f_gemv, a_k);
    launch(f_gemv, a_v);
    launch(f_gemv, a_g);
}
static void run_fused() { launch(f_qkvg, a_fused); }

static float time_ms(void (*body)(), int iters) {
    hipEvent_t s, e;
    HIP_OK(hipEventCreate(&s));
    HIP_OK(hipEventCreate(&e));
    for (int i = 0; i < 10; i++) body();
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipEventRecord(s));
    for (int i = 0; i < iters; i++) body();
    HIP_OK(hipEventRecord(e));
    HIP_OK(hipEventSynchronize(e));
    float ms = 0;
    HIP_OK(hipEventElapsedTime(&ms, s, e));
    HIP_OK(hipEventDestroy(s));
    HIP_OK(hipEventDestroy(e));
    return ms / (float)iters;
}

static void* put_args(const KArgs& a) {
    void* d = dev_alloc(sizeof(KArgs));
    HIP_OK(hipMemcpy(d, &a, sizeof(KArgs), hipMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "/tmp/qkvg.co";
    hipDeviceProp_t prop;
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    hipModule_t mod;
    HIP_OK(hipModuleLoad(&mod, co));
    HIP_OK(hipModuleGetFunction(&f_gemv, mod, "k_gemv"));
    HIP_OK(hipModuleGetFunction(&f_qkv, mod, "k_qkv"));
    HIP_OK(hipModuleGetFunction(&f_qkvg, mod, "k_qkvg"));

    printf("gemv_qkvg_gfx950_test on %s (%d CUs), object %s\n", prop.gcnArchName,
           prop.multiProcessorCount, co);
    printf("shape: M=%u  Nq=Nk=Nv=Ng=%u  K=%u  blocks=%u  (Kimi-K3 KDA decode)\n\n", K3_M, K3_N,
           K3_K, NBLK);

    const size_t wn = (size_t)K3_N * K3_K, on = (size_t)K3_M * K3_N;
    void* x = dev_alloc((size_t)K3_M * K3_K * 2);
    void* Wq = dev_alloc(wn * 2);
    void* Wk = dev_alloc(wn * 2);
    void* Wv = dev_alloc(wn * 2);
    void* Wg = dev_alloc(wn * 2);
    void* Cq = dev_alloc(on * 2);
    void* Ck = dev_alloc(on * 2);
    void* Cv = dev_alloc(on * 2);
    void* Cg = dev_alloc(on * 2);

    const auto hx = fill_bf16(x, (size_t)K3_M * K3_K);
    const auto hWq = fill_bf16(Wq, wn);
    const auto hWk = fill_bf16(Wk, wn);
    const auto hWv = fill_bf16(Wv, wn);
    const auto hWg = fill_bf16(Wg, wn);

    const unsigned long long X = (unsigned long long)x;
    /* k_gemv reads Cq/Wq/Nq only, so each projection gets its own arg block. */
    KArgs t{};
    t.x = X;
    t.M = K3_M;
    t.Nq = t.Nk = t.Nv = K3_N;
    t.K = K3_K;
    t.Cq = (unsigned long long)Cq;
    t.Wq = (unsigned long long)Wq;
    a_q = put_args(t);
    t.Cq = (unsigned long long)Ck;
    t.Wq = (unsigned long long)Wk;
    a_k = put_args(t);
    t.Cq = (unsigned long long)Cv;
    t.Wq = (unsigned long long)Wv;
    a_v = put_args(t);
    t.Cq = (unsigned long long)Cg;
    t.Wq = (unsigned long long)Wg;
    a_g = put_args(t);

    KArgs f{};
    f.Cq = (unsigned long long)Cq;
    f.Ck = (unsigned long long)Ck;
    f.Cv = (unsigned long long)Cv;
    f.Cg = (unsigned long long)Cg;
    f.x = X;
    f.Wq = (unsigned long long)Wq;
    f.Wk = (unsigned long long)Wk;
    f.Wv = (unsigned long long)Wv;
    f.Wg = (unsigned long long)Wg;
    f.M = K3_M;
    f.Nq = f.Nk = f.Nv = f.Ng = K3_N;
    f.K = K3_K;
    a_fused = put_args(f);
    KArgs f3 = f;
    f3.Cg = 0;
    f3.Wg = 0;
    f3.Ng = 0;
    a_qkv3 = put_args(f3);

    /* ---- the reference: four independent GEMVs, the packets this fusion replaces ---- */
    run_four_separate();
    HIP_OK(hipDeviceSynchronize());
    const auto rq = read_bf16(Cq, on), rk = read_bf16(Ck, on), rv = read_bf16(Cv, on),
               rg = read_bf16(Cg, on);
    printf("absolute accuracy vs an f64 host dot product (prices the bf16 GEMV, not the fusion):\n");
    printf("  q col 0     rel err %.3e     q col %-5u rel err %.3e\n", col_err(rq, hx, hWq, 0, K3_K),
           K3_N - 1, col_err(rq, hx, hWq, K3_N - 1, K3_K));
    printf("  g col 0     rel err %.3e     g col %-5u rel err %.3e\n\n", col_err(rg, hx, hWg, 0, K3_K),
           K3_N - 1, col_err(rg, hx, hWg, K3_N - 1, K3_K));

    size_t bad = 0;

    /* ---- 1. the fused 4-stream sweep ---- */
    HIP_OK(hipMemset(Cq, 0, on * 2));
    HIP_OK(hipMemset(Ck, 0, on * 2));
    HIP_OK(hipMemset(Cv, 0, on * 2));
    HIP_OK(hipMemset(Cg, 0, on * 2));
    run_fused();
    HIP_OK(hipDeviceSynchronize());
    printf("GemvQkvg (fused, Ng=%u) vs four separate GEMVs, elementwise:\n", K3_N);
    bad += cmp("q_proj", read_bf16(Cq, on), rq);
    bad += cmp("k_proj", read_bf16(Ck, on), rk);
    bad += cmp("v_proj", read_bf16(Cv, on), rv);
    bad += cmp("g_proj", read_bf16(Cg, on), rg);

    /* ---- 2. op 22 through the shared body; g_out is poisoned and must stay poisoned ---- */
    HIP_OK(hipMemset(Cq, 0, on * 2));
    HIP_OK(hipMemset(Ck, 0, on * 2));
    HIP_OK(hipMemset(Cv, 0, on * 2));
    HIP_OK(hipMemset(Cg, 0xEE, on * 2));
    launch(f_qkv, a_qkv3);
    HIP_OK(hipDeviceSynchronize());
    printf("\nGemvQkv (op 22, now a forwarder with Ng=0) vs the same reference:\n");
    bad += cmp("q_proj", read_bf16(Cq, on), rq);
    bad += cmp("k_proj", read_bf16(Ck, on), rk);
    bad += cmp("v_proj", read_bf16(Cv, on), rv);
    {
        const auto poisoned = read_bf16(Cg, on);
        size_t touched = 0;
        for (size_t i = 0; i < poisoned.size(); i++) touched += (poisoned[i] != 0xEEEEu);
        printf("  %-26s %zu halves written (must be 0)\n", "g_out left alone", touched);
        bad += touched;
    }

    /* ---- 3. timing ---- */
    const int iters = 300;
    const float t4 = time_ms(run_four_separate, iters);
    const float t1 = time_ms(run_fused, iters);
    const double bytes = 4.0 * (double)wn * 2.0; /* the four weight matrices, streamed once */
    printf("\ntiming (%d iters, BARE KERNELS — the three counter gates the fusion deletes are in\n"
           "neither column, so this gap is a LOWER BOUND on the packet-level win):\n",
           iters);
    printf("  4 separate GEMVs : %8.4f ms   %7.1f GB/s\n", t4, bytes / (t4 * 1e-3) / 1e9);
    printf("  1 fused GemvQkvg : %8.4f ms   %7.1f GB/s\n", t1, bytes / (t1 * 1e-3) / 1e9);
    printf("  delta            : %+8.4f ms  (%+.2f%%)\n", t1 - t4, 100.0 * (t1 - t4) / t4);

    printf("\n%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
#endif

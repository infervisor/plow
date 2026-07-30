/* gemv_qkv_mxfp4_gfx950_test.hip — the FUSED mxfp4 q|k|v decode GEMV, checked and timed on hardware.
 *
 * `DevOp::GemvQkvMxfp4` (op 110) merges Kimi-K3's three MLA down-projections — `q_a` (1536),
 * `kv_a` (512) and `k_rope` (64), all read against the same pre-normed `x[7168]` — into ONE packet
 * along the OUTPUT dimension. This file holds that claim to four numbers, because the doc comment
 * that proposed the fusion can assert none of them:
 *
 *   1. BYTE-EXACTNESS, and it is the bar rather than a nicety. The fused sweep must agree with
 *      three independent `d_gemv_mxfp4` calls ELEMENTWISE. Both accumulate the same K chunks in
 *      the same order at the same unroll (the UN table is keyed on K alone), so anything other
 *      than zero differing halves is a bug and not rounding.
 *   2. THE 1-STREAM PATH IS UNCHANGED. `d_gemv_mxfp4` (op 91, shipping) is now a forwarder into
 *      the 3-stream body with Nk = Nv = 0. Sharing one body is what keeps the interpreter from
 *      inlining a second copy and paying for it in registers — and it is exactly the kind of
 *      change that silently perturbs an op nobody re-tested. `gemv_mxfp4_forwarder_ab.sh` diffs
 *      op 91's output against an object built from the PRE-CHANGE header; here the guard is that
 *      the 2-stream form (Nv = 0, the `q_nope|q_rope` shape) leaves a POISONED `v_out` alone.
 *   3. IS IT ACTUALLY FASTER. The three projections were already 256-CU packets. The fusion
 *      cannot move fewer bytes; what it removes is two counter gates and two redundant LDS
 *      stagings of `x` — and, at these widths, the LAUNCH FLOOR of two whole kernels.
 *   4. WHAT EACH CU ACTUALLY OWNS. `k_rope` is 64 columns over 256 workgroups: `per = 1`, so 192
 *      CUs own NOTHING and still stage x. The merge is along the OUTPUT axis precisely so this
 *      number RISES rather than falls — printed below, not asserted from the design.
 *
 * A caveat this file cannot escape and does not pretend to: it measures BARE KERNELS, so the two
 * counter gates the fusion deletes are in NEITHER column. The gap reported is the launch plus
 * staging saving only, and it is a LOWER BOUND on the in-interpreter win.
 *
 * TWO PASSES, because `op_gemm.h` is DEVICE-ONLY (see gemv_qkvg_gfx950_test.hip's note).
 *
 *   hipcc --offload-arch=gfx950 -O3 -w -DPLOW_GEMV_MM=4 -DPLOW_K3=1 -DQKVMX_DEVICE --genco \
 *       runtime/tests/gemv_qkv_mxfp4_gfx950_test.hip -o /tmp/qkvmx.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/gemv_qkv_mxfp4_gfx950_test.hip -o /tmp/qkvmx_test -L/opt/rocm/lib -lamdhip64
 *   sg render -c 'ROCR_VISIBLE_DEVICES=7 /tmp/qkvmx_test /tmp/qkvmx.co'
 */

/* ONE argument per kernel, a pointer to this struct in device memory — the same reason the bf16
 * twin gives: a hand-packed HIP_LAUNCH_PARAM buffer is a silent-corruption shape. */
struct KArgs {
    unsigned long long Cq, Ck, Cv, x, Wq, Wk, Wv, Sq, Sk, Sv;
    unsigned M, Nq, Nk, Nv, K;
};

#ifdef QKVMX_DEVICE
/* ======================= DEVICE HALF (--genco) ======================= */
#include <hip/hip_runtime.h>
#include "op_gemm.h"

/* The interpreter's LDS arena, verbatim — these kernels stand in for its dispatch arms. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_gemv_mxfp4(const KArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_mxfp4((bf16*)a->Cq, (const bf16*)a->x, (const unsigned char*)a->Wq,
                 (const unsigned char*)a->Sq, a->M, a->Nq, a->K, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_qkv_mxfp4(const KArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_qkv_mxfp4((bf16*)a->Cq, (bf16*)a->Ck, (bf16*)a->Cv, (const bf16*)a->x,
                     (const unsigned char*)a->Wq, (const unsigned char*)a->Wk,
                     (const unsigned char*)a->Wv, (const unsigned char*)a->Sq,
                     (const unsigned char*)a->Sk, (const unsigned char*)a->Sv, a->M, a->Nq, a->Nk,
                     a->Nv, a->K, blockIdx.x, gridDim.x, lds);
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

/* Kimi-K3 MLA decode: hidden 7168; q_a -> q_lora 1536, kv_a -> kv_lora 512, k_rope -> 64. */
static const unsigned K3_K = 7168, K3_NQ = 1536, K3_NK = 512, K3_NV = 64, K3_M = 1;
static const unsigned NBLK = 256, NTHREADS = 512;

static unsigned rng_state = 0x2545F491u;
static unsigned urand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static float frand() { return (float)((urand() >> 8) & 0xFFFFu) / 32768.0f - 1.0f; }

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
/* OCP e2m1: sign + 3-bit magnitude code. The host decode exists only for the f64 accuracy
 * reference; correctness of the FUSION is decided by the elementwise device-vs-device compare. */
static const float FP4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
static float fp4_h(unsigned nib) { return (nib & 8u) ? -FP4[nib & 7u] : FP4[nib & 7u]; }
static float e8m0_h(unsigned char b) {
    unsigned u = (unsigned)b << 23;
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
static std::vector<unsigned char> fill_u8(void* d, size_t n, int lo, int hi) {
    std::vector<unsigned char> h(n);
    for (size_t i = 0; i < n; i++) h[i] = (unsigned char)(lo + (int)(urand() % (unsigned)(hi - lo)));
    HIP_OK(hipMemcpy(d, h.data(), n, hipMemcpyHostToDevice));
    return h;
}
static std::vector<unsigned short> read_bf16(void* d, size_t n) {
    std::vector<unsigned short> h(n);
    HIP_OK(hipMemcpy(h.data(), d, n * 2, hipMemcpyDeviceToHost));
    return h;
}

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

/* Absolute accuracy against an f64 host dot over the DECODED fp4 weights — prices the mxfp4 GEMV
 * itself, not the fusion (fused and unfused share it exactly). */
static double col_err(const std::vector<unsigned short>& C, const std::vector<unsigned short>& x,
                      const std::vector<unsigned char>& W, const std::vector<unsigned char>& S,
                      unsigned n, unsigned K) {
    double acc = 0;
    for (unsigned k = 0; k < K; k++) {
        const unsigned char byte = W[(size_t)n * (K / 2) + k / 2];
        const unsigned nib = (k & 1u) ? (byte >> 4) : (byte & 0xFu); /* low nibble = even k */
        acc += (double)bf2f_h(x[k]) * (double)fp4_h(nib) * (double)e8m0_h(S[(size_t)n * (K / 32) + k / 32]);
    }
    return fabs((double)bf2f_h(C[n]) - acc) / (fabs(acc) > 1e-30 ? fabs(acc) : 1e-30);
}

static hipFunction_t f_gemv, f_qkv;
static void *a_q, *a_k, *a_v, *a_fused, *a_two;

static void launch(hipFunction_t f, void* args_dev) {
    size_t sz = sizeof(void*);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_dev, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                   HIP_LAUNCH_PARAM_END};
    HIP_OK(hipModuleLaunchKernel(f, NBLK, 1, 1, NTHREADS, 1, 1, 0, nullptr, nullptr, cfg));
}
static void run_three_separate() {
    launch(f_gemv, a_q);
    launch(f_gemv, a_k);
    launch(f_gemv, a_v);
}
static void run_fused() { launch(f_qkv, a_fused); }

static float time_ms(void (*body)(), int iters) {
    hipEvent_t s, e;
    HIP_OK(hipEventCreate(&s));
    HIP_OK(hipEventCreate(&e));
    for (int i = 0; i < 20; i++) body();
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

/* The kernel's own ownership arithmetic (GV_BLOCKED, gemv_rows_mxfp4): workgroup `slice` owns
 * [slice*per, slice*per+per) with per = ceil(N/nblk). Printed for the split and fused forms so the
 * "merge along the OUTPUT dimension" claim is a measurement rather than an assertion. */
static void report_per_cu(const char* what, unsigned n, unsigned nblk) {
    const unsigned per = (n + nblk - 1) / nblk;
    const unsigned live = (n + per - 1) / per;
    printf("  %-22s N=%-6u per-CU columns %-3u   CUs with work %3u / %u   idle %3u\n", what, n, per,
           live, nblk, nblk - live);
}

int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "/tmp/qkvmx.co";
    hipDeviceProp_t prop;
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    hipModule_t mod;
    HIP_OK(hipModuleLoad(&mod, co));
    HIP_OK(hipModuleGetFunction(&f_gemv, mod, "k_gemv_mxfp4"));
    HIP_OK(hipModuleGetFunction(&f_qkv, mod, "k_qkv_mxfp4"));

    printf("gemv_qkv_mxfp4_gfx950_test on %s (%d CUs), pci %02x:%02x.0, object %s\n",
           prop.gcnArchName, prop.multiProcessorCount, prop.pciBusID, prop.pciDeviceID, co);
    printf("shape: M=%u  Nq=%u (q_a)  Nk=%u (kv_a)  Nv=%u (k_rope)  K=%u  blocks=%u\n\n", K3_M,
           K3_NQ, K3_NK, K3_NV, K3_K, NBLK);

    printf("per-CU work, from the kernel's own GV_BLOCKED ownership arithmetic:\n");
    report_per_cu("q_a   (split)", K3_NQ, NBLK);
    report_per_cu("kv_a  (split)", K3_NK, NBLK);
    report_per_cu("k_rope(split)", K3_NV, NBLK);
    report_per_cu("q|kv|k_rope fused", K3_NQ + K3_NK + K3_NV, NBLK);
    printf("\n");

    const size_t wq = (size_t)K3_NQ * K3_K / 2, wk = (size_t)K3_NK * K3_K / 2,
                 wv = (size_t)K3_NV * K3_K / 2;
    const size_t sq = (size_t)K3_NQ * K3_K / 32, sk = (size_t)K3_NK * K3_K / 32,
                 sv = (size_t)K3_NV * K3_K / 32;

    void* x = dev_alloc((size_t)K3_M * K3_K * 2);
    void* Wq = dev_alloc(wq);
    void* Wk = dev_alloc(wk);
    void* Wv = dev_alloc(wv);
    void* Sq = dev_alloc(sq);
    void* Sk = dev_alloc(sk);
    void* Sv = dev_alloc(sv);
    void* Cq = dev_alloc((size_t)K3_M * K3_NQ * 2);
    void* Ck = dev_alloc((size_t)K3_M * K3_NK * 2);
    void* Cv = dev_alloc((size_t)K3_M * K3_NV * 2);

    const auto hx = fill_bf16(x, (size_t)K3_M * K3_K);
    const auto hWq = fill_u8(Wq, wq, 0, 256);
    const auto hWk = fill_u8(Wk, wk, 0, 256);
    const auto hWv = fill_u8(Wv, wv, 0, 256);
    /* E8M0 around 2^-1..2^1 so a column's dot stays in bf16 range at K=7168. */
    const auto hSq = fill_u8(Sq, sq, 126, 129);
    const auto hSk = fill_u8(Sk, sk, 126, 129);
    const auto hSv = fill_u8(Sv, sv, 126, 129);

    KArgs t{};
    t.x = (unsigned long long)x;
    t.M = K3_M;
    t.K = K3_K;
    /* k_gemv_mxfp4 reads Cq/Wq/Sq/Nq only, so each projection gets its own arg block. */
    t.Cq = (unsigned long long)Cq;
    t.Wq = (unsigned long long)Wq;
    t.Sq = (unsigned long long)Sq;
    t.Nq = K3_NQ;
    a_q = put_args(t);
    t.Cq = (unsigned long long)Ck;
    t.Wq = (unsigned long long)Wk;
    t.Sq = (unsigned long long)Sk;
    t.Nq = K3_NK;
    a_k = put_args(t);
    t.Cq = (unsigned long long)Cv;
    t.Wq = (unsigned long long)Wv;
    t.Sq = (unsigned long long)Sv;
    t.Nq = K3_NV;
    a_v = put_args(t);

    KArgs f{};
    f.Cq = (unsigned long long)Cq;
    f.Ck = (unsigned long long)Ck;
    f.Cv = (unsigned long long)Cv;
    f.x = (unsigned long long)x;
    f.Wq = (unsigned long long)Wq;
    f.Wk = (unsigned long long)Wk;
    f.Wv = (unsigned long long)Wv;
    f.Sq = (unsigned long long)Sq;
    f.Sk = (unsigned long long)Sk;
    f.Sv = (unsigned long long)Sv;
    f.M = K3_M;
    f.Nq = K3_NQ;
    f.Nk = K3_NK;
    f.Nv = K3_NV;
    f.K = K3_K;
    a_fused = put_args(f);
    /* The 2-stream form (Nv = 0) the q_nope|q_rope pair uses: v operands absent. */
    KArgs f2 = f;
    f2.Cv = 0;
    f2.Wv = 0;
    f2.Sv = 0;
    f2.Nv = 0;
    a_two = put_args(f2);

    /* ---- the reference: three independent mxfp4 GEMVs, the packets this fusion replaces ---- */
    run_three_separate();
    HIP_OK(hipDeviceSynchronize());
    const auto rq = read_bf16(Cq, K3_NQ), rk = read_bf16(Ck, K3_NK), rv = read_bf16(Cv, K3_NV);
    printf("absolute accuracy vs an f64 host dot over the decoded fp4 (prices the GEMV, not the "
           "fusion):\n");
    printf("  q_a  col 0    rel err %.3e     col %-5u rel err %.3e\n",
           col_err(rq, hx, hWq, hSq, 0, K3_K), K3_NQ - 1,
           col_err(rq, hx, hWq, hSq, K3_NQ - 1, K3_K));
    printf("  krope col 0   rel err %.3e     col %-5u rel err %.3e\n\n",
           col_err(rv, hx, hWv, hSv, 0, K3_K), K3_NV - 1,
           col_err(rv, hx, hWv, hSv, K3_NV - 1, K3_K));

    size_t bad = 0;

    /* ---- 1. the fused 3-stream sweep, ELEMENTWISE ---- */
    HIP_OK(hipMemset(Cq, 0, (size_t)K3_NQ * 2));
    HIP_OK(hipMemset(Ck, 0, (size_t)K3_NK * 2));
    HIP_OK(hipMemset(Cv, 0, (size_t)K3_NV * 2));
    run_fused();
    HIP_OK(hipDeviceSynchronize());
    printf("GemvQkvMxfp4 (fused) vs three separate GemvMxfp4, elementwise:\n");
    bad += cmp("q_a", read_bf16(Cq, K3_NQ), rq);
    bad += cmp("kv_a", read_bf16(Ck, K3_NK), rk);
    bad += cmp("k_rope", read_bf16(Cv, K3_NV), rv);

    /* ---- 2. the 2-stream form; v_out is poisoned and must stay poisoned ---- */
    HIP_OK(hipMemset(Cq, 0, (size_t)K3_NQ * 2));
    HIP_OK(hipMemset(Ck, 0, (size_t)K3_NK * 2));
    HIP_OK(hipMemset(Cv, 0xEE, (size_t)K3_NV * 2));
    launch(f_qkv, a_two);
    HIP_OK(hipDeviceSynchronize());
    printf("\ntwo-stream form (Nv=0, the q_nope|q_rope shape) vs the same reference:\n");
    bad += cmp("q_a", read_bf16(Cq, K3_NQ), rq);
    bad += cmp("kv_a", read_bf16(Ck, K3_NK), rk);
    {
        const auto poisoned = read_bf16(Cv, K3_NV);
        size_t touched = 0;
        for (size_t i = 0; i < poisoned.size(); i++) touched += (poisoned[i] != 0xEEEEu);
        printf("  %-26s %zu halves written (must be 0)\n", "v_out left alone", touched);
        bad += touched;
    }

    /* ---- 3. timing ---- */
    const int iters = 500;
    const float t3 = time_ms(run_three_separate, iters);
    const float t1 = time_ms(run_fused, iters);
    const double bytes = (double)(wq + wk + wv + sq + sk + sv);
    printf("\ntiming (%d iters, BARE KERNELS — the two counter gates the fusion deletes are in\n"
           "neither column, so this gap is a LOWER BOUND on the packet-level win):\n",
           iters);
    printf("  3 separate GemvMxfp4 : %8.4f ms   %7.1f GB/s of weight+scale stream\n", t3,
           bytes / (t3 * 1e-3) / 1e9);
    printf("  1 fused GemvQkvMxfp4 : %8.4f ms   %7.1f GB/s\n", t1, bytes / (t1 * 1e-3) / 1e9);
    printf("  delta                : %+8.4f ms  (%+.2f%%)\n", t1 - t3, 100.0 * (t1 - t3) / t3);

    printf("\n%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
#endif

/* gemm_glu_mxfp4_gfx950_test.hip — the FUSED mxfp4 gate|up prefill GEMM, checked and timed.
 *
 * `DevOp::GemmGluMxfp4` (op 109) is the T-row twin of decode's `GemvGluMxfp4` and the fp4 twin of
 * `GemmGlu`. Its absence is what made the SHARED-EXPERT PREFILL the one place an mxfp4 packet paid
 * for its encoding in HBM TRAFFIC rather than only in a different weight fetch: the pair unfused
 * into two `GemmMxfp4` plus a `Glu`, which materialises gate and up ([T, N] bf16 each) to HBM and
 * reads them both back.
 *
 * THE TRAFFIC, counted rather than asserted (K3 geometry, K = hidden = 7168):
 *   weights, both matrices   2 * N * K / 2 bytes fp4  +  2 * N * K / 32 bytes E8M0
 *   activation A             T * K * 2 bytes, read once per tile column either way
 *   UNFUSED ONLY            gate + up written   2 * T * N * 2 B
 *                           gate + up read back 2 * T * N * 2 B
 *                           fu written          T * N * 2 B   (the fused arm writes only this)
 *   => 8 bytes per output element of avoidable traffic, plus two extra packets and their gates.
 *
 * WHAT IS AND IS NOT EXPECTED TO MATCH. The fused arm keeps gate and up in the f32 ACCUMULATOR
 * across the SwiGLU; the unfused triple rounds both to bf16 before it. So they are NOT bit-equal
 * and must not be: the fused one is strictly more accurate, exactly as the bf16 `GemmGlu` is
 * against its own triple. Both are therefore compared to an f64 host reference, and the fused arm
 * must not be WORSE. (Byte-exactness is the bar for the decode QKV fusion, where both sides
 * compute the identical per-column dot; it is the wrong bar here and claiming it would be a lie.)
 *
 * TILE. `d_gemm_glu_mxfp4` exists at 256x256 only — the epilogue's wave->column remap needs
 * SN == 2 — so this file also times the unfused pair at EVERY fp4 rung, which is what makes the
 * emitter's `glu_fusion_wins_mxfp4` restriction checkable rather than asserted: if a narrower rung
 * beats 256x256 by more than the round trip costs, the pair should stay unfused at that shape.
 *
 *   hipcc --offload-arch=gfx950 -O3 -w -DGGMX_DEVICE --genco \
 *       runtime/tests/gemm_glu_mxfp4_gfx950_test.hip -o /tmp/ggmx.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/gemm_glu_mxfp4_gfx950_test.hip -o /tmp/ggmx_test -L/opt/rocm/lib -lamdhip64
 *   sg render -c 'ROCR_VISIBLE_DEVICES=7 /tmp/ggmx_test /tmp/ggmx.co'
 */

struct GArgs {
    unsigned long long C, A, Wg, Wu, Sg, Su, Cu; /* Cu = the `up` staging the unfused pair needs */
    unsigned M, N, K, act;
};

#ifdef GGMX_DEVICE
/* ======================= DEVICE HALF (--genco) ======================= */
#include <hip/hip_runtime.h>
#include "op_gemm.h"
#include "op_elementwise.h"

#define GGMX_GEMM(NAME, BODY)                                                                      \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(const GArgs* a) {               \
        __shared__ bf16 lds[GM_LDS_HALVES];                                                         \
        BODY((bf16*)a->C, (const bf16*)a->A, (const unsigned char*)a->Wg,                            \
             (const unsigned char*)a->Sg, a->M, a->N, a->K, blockIdx.x, gridDim.x, lds);            \
    }
GGMX_GEMM(k_gemm_c0, d_gemm_mxfp4)
GGMX_GEMM(k_gemm_med, d_gemm_med_mxfp4)
GGMX_GEMM(k_gemm_small, d_gemm_small_mxfp4)
GGMX_GEMM(k_gemm_wide, d_gemm_wide_mxfp4)
GGMX_GEMM(k_gemm_c5, d_gemm_c5_mxfp4)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_glu(const GArgs* a) {
    d_glu((bf16*)a->C, (const bf16*)a->C, (const bf16*)a->Cu, a->M * a->N, a->act, blockIdx.x,
          gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_gemm_glu(const GArgs* a) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemm_glu_mxfp4((bf16*)a->C, (const bf16*)a->A, (const unsigned char*)a->Wg,
                     (const unsigned char*)a->Wu, (const unsigned char*)a->Sg,
                     (const unsigned char*)a->Su, a->M, a->N, a->K, a->act, blockIdx.x, gridDim.x,
                     lds);
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

/* PLOW_ACT_SILU_ = 1 (op_elementwise.h); 0 is GELU-tanh, which is what an `act` typo silently
 * selects — and the resulting output is finite, plausible and wrong by ~100% on negative gates. */
static const unsigned NBLK = 256, NTHREADS = 512, ACT_SILU = 1;

static unsigned rng_state = 0x13579BDFu;
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
static const float FP4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
static float fp4_h(unsigned nib) { return (nib & 8u) ? -FP4[nib & 7u] : FP4[nib & 7u]; }
static float e8m0_h(unsigned char b) {
    unsigned u = (unsigned)b << 23;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static double silu_h(double g) { return g / (1.0 + exp(-g)); }

static void* dev_alloc(size_t bytes) {
    void* p = nullptr;
    HIP_OK(hipMalloc(&p, bytes));
    HIP_OK(hipMemset(p, 0, bytes));
    return p;
}
static std::vector<unsigned short> fill_bf16(void* d, size_t n) {
    std::vector<unsigned short> h(n);
    for (size_t i = 0; i < n; i++) h[i] = f2bf_h(frand() * 0.25f);
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

/* One output element, in f64, straight off the decoded fp4. */
static double ref_elem(const std::vector<unsigned short>& A, const std::vector<unsigned char>& Wg,
                       const std::vector<unsigned char>& Sg, const std::vector<unsigned char>& Wu,
                       const std::vector<unsigned char>& Su, unsigned m, unsigned n, unsigned K) {
    double g = 0, u = 0;
    for (unsigned k = 0; k < K; k++) {
        const double a = bf2f_h(A[(size_t)m * K + k]);
        const unsigned char bg = Wg[(size_t)n * (K / 2) + k / 2], bu = Wu[(size_t)n * (K / 2) + k / 2];
        const unsigned ng = (k & 1u) ? (bg >> 4) : (bg & 0xFu);
        const unsigned nu = (k & 1u) ? (bu >> 4) : (bu & 0xFu);
        g += a * fp4_h(ng) * e8m0_h(Sg[(size_t)n * (K / 32) + k / 32]);
        u += a * fp4_h(nu) * e8m0_h(Su[(size_t)n * (K / 32) + k / 32]);
    }
    return silu_h(g) * u;
}

static hipFunction_t f_c0, f_med, f_small, f_wide, f_c5, f_glu, f_fused;
static void launch(hipFunction_t f, void* args_dev) {
    size_t sz = sizeof(void*);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_dev, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                   HIP_LAUNCH_PARAM_END};
    HIP_OK(hipModuleLaunchKernel(f, NBLK, 1, 1, NTHREADS, 1, 1, 0, nullptr, nullptr, cfg));
}

/* Timing needs a nullary callable, and the shape loop needs the operands to vary — hence file
 * statics rather than lambdas with captures. */
static hipFunction_t g_split;
static void *g_gate_args, *g_up_args, *g_glu_args, *g_fused_args;
static void run_split() {
    launch(g_split, g_gate_args);
    launch(g_split, g_up_args);
    launch(f_glu, g_glu_args);
}
static void run_fused() { launch(f_fused, g_fused_args); }

static float time_ms(void (*body)(), int iters) {
    hipEvent_t s, e;
    HIP_OK(hipEventCreate(&s));
    HIP_OK(hipEventCreate(&e));
    for (int i = 0; i < 5; i++) body();
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

static void* put_args(const GArgs& a) {
    void* d = dev_alloc(sizeof(GArgs));
    HIP_OK(hipMemcpy(d, &a, sizeof(GArgs), hipMemcpyHostToDevice));
    return d;
}

static int bad = 0;

static void run_shape(unsigned T, unsigned N, unsigned K) {
    const size_t wn = (size_t)N * K / 2, sn = (size_t)N * K / 32, on = (size_t)T * N;
    void* A = dev_alloc((size_t)T * K * 2);
    void* Wg = dev_alloc(wn);
    void* Wu = dev_alloc(wn);
    void* Sg = dev_alloc(sn);
    void* Su = dev_alloc(sn);
    void* C = dev_alloc(on * 2);
    void* Cu = dev_alloc(on * 2);

    const auto hA = fill_bf16(A, (size_t)T * K);
    const auto hWg = fill_u8(Wg, wn, 0, 256);
    const auto hWu = fill_u8(Wu, wn, 0, 256);
    const auto hSg = fill_u8(Sg, sn, 126, 129);
    const auto hSu = fill_u8(Su, sn, 126, 129);

    GArgs base{};
    base.A = (unsigned long long)A;
    base.M = T;
    base.N = N;
    base.K = K;
    base.act = ACT_SILU;
    GArgs gate = base;
    gate.C = (unsigned long long)C;
    gate.Wg = (unsigned long long)Wg;
    gate.Sg = (unsigned long long)Sg;
    GArgs up = base;
    up.C = (unsigned long long)Cu;
    up.Wg = (unsigned long long)Wu;
    up.Sg = (unsigned long long)Su;
    GArgs glu = base;
    glu.C = (unsigned long long)C;
    glu.Cu = (unsigned long long)Cu;
    GArgs fus = base;
    fus.C = (unsigned long long)C;
    fus.Wg = (unsigned long long)Wg;
    fus.Wu = (unsigned long long)Wu;
    fus.Sg = (unsigned long long)Sg;
    fus.Su = (unsigned long long)Su;

    g_gate_args = put_args(gate);
    g_up_args = put_args(up);
    g_glu_args = put_args(glu);
    g_fused_args = put_args(fus);

    printf("\n--- T=%-5u N=%-5u K=%-5u  (%zu output elements)\n", T, N, K, on);

    /* ---- accuracy: both arms against an f64 host reference over a sample of elements ---- */
    g_split = f_c0;
    run_split();
    HIP_OK(hipDeviceSynchronize());
    const auto un = read_bf16(C, on);
    HIP_OK(hipMemset(C, 0, on * 2));
    run_fused();
    HIP_OK(hipDeviceSynchronize());
    const auto fu = read_bf16(C, on);

    double wu_un = 0, wu_fu = 0, wdiff = 0;
    const unsigned NS = 64;
    for (unsigned s = 0; s < NS; s++) {
        const unsigned m = (unsigned)(urand() % T), n = (unsigned)(urand() % N);
        const double r = ref_elem(hA, hWg, hSg, hWu, hSu, m, n, K);
        const double den = fabs(r) > 1e-6 ? fabs(r) : 1e-6;
        const double eu = fabs(bf2f_h(un[(size_t)m * N + n]) - r) / den;
        const double ef = fabs(bf2f_h(fu[(size_t)m * N + n]) - r) / den;
        if (eu > wu_un) wu_un = eu;
        if (ef > wu_fu) wu_fu = ef;
        const double d = fabs((double)bf2f_h(un[(size_t)m * N + n]) - bf2f_h(fu[(size_t)m * N + n]));
        if (d / den > wdiff) wdiff = d / den;
    }
    printf("  accuracy over %u sampled elements vs an f64 host reference:\n", NS);
    printf("    unfused (2 GemmMxfp4 + Glu) worst rel err %.4e\n", wu_un);
    printf("    fused   (GemmGluMxfp4)      worst rel err %.4e   %s\n", wu_fu,
           wu_fu <= wu_un * 1.05 ? "<= unfused (the f32 accumulator is kept across the GLU)"
                                 : "WORSE THAN UNFUSED");
    printf("    fused vs unfused, worst rel gap %.4e (they are NOT bit-equal by construction)\n",
           wdiff);
    if (wu_fu > wu_un * 1.05) bad++;
    if (wu_fu > 0.05) {
        printf("    FAIL: fused arm is not tracking the reference\n");
        bad++;
    }

    /* ---- timing: the fused arm, and the unfused pair at every fp4 rung ---- */
    const int iters = T >= 2048 ? 20 : 60;
    const float tf = time_ms(run_fused, iters);
    struct {
        const char* name;
        hipFunction_t f;
    } rungs[] = {{"256x256", f_c0}, {"128x128", f_med}, {"64x128", f_small},
                 {"128x256", f_wide}, {"192x256", f_c5}};
    float best = 1e30f, t256 = 0;
    const char* bestn = "";
    printf("  timing (%d iters):\n", iters);
    for (auto& r : rungs) {
        g_split = r.f;
        const float t = time_ms(run_split, iters);
        printf("    unfused 2x%-8s + Glu : %8.4f ms\n", r.name, t);
        if (strcmp(r.name, "256x256") == 0) t256 = t;
        if (t < best) {
            best = t;
            bestn = r.name;
        }
    }
    printf("    FUSED GemmGluMxfp4      : %8.4f ms   %+.2f%% vs the SAME tile unfused\n", tf,
           100.0 * (tf - t256) / t256);

    /* HBM traffic the fusion removes: gate+up written and read back, minus the one fu write the
     * fused arm still does. Weights and A are identical in both arms. */
    const double saved = 4.0 * (double)on * 2.0;
    printf("  HBM traffic the fusion removes: %.2f MB (gate+up written and read back, %zu elements "
           "x 8 B); best unfused rung was %s at %.4f ms -> fused is %+.2f%% against it\n",
           saved / 1e6, on, bestn, best, 100.0 * (tf - best) / best);

    HIP_OK(hipFree(A));
    HIP_OK(hipFree(Wg));
    HIP_OK(hipFree(Wu));
    HIP_OK(hipFree(Sg));
    HIP_OK(hipFree(Su));
    HIP_OK(hipFree(C));
    HIP_OK(hipFree(Cu));
}

int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "/tmp/ggmx.co";
    hipDeviceProp_t prop;
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    hipModule_t mod;
    HIP_OK(hipModuleLoad(&mod, co));
    HIP_OK(hipModuleGetFunction(&f_c0, mod, "k_gemm_c0"));
    HIP_OK(hipModuleGetFunction(&f_med, mod, "k_gemm_med"));
    HIP_OK(hipModuleGetFunction(&f_small, mod, "k_gemm_small"));
    HIP_OK(hipModuleGetFunction(&f_wide, mod, "k_gemm_wide"));
    HIP_OK(hipModuleGetFunction(&f_c5, mod, "k_gemm_c5"));
    HIP_OK(hipModuleGetFunction(&f_glu, mod, "k_glu"));
    HIP_OK(hipModuleGetFunction(&f_fused, mod, "k_gemm_glu"));

    printf("gemm_glu_mxfp4_gfx950_test on %s (%d CUs), pci %02x:%02x.0, object %s\n",
           prop.gcnArchName, prop.multiProcessorCount, prop.pciBusID, prop.pciDeviceID, co);
    /* N is the shared expert's per-rank gate|up width. K3's is
     * `num_shared_experts * moe_intermediate_size / TP` = 6144 / TP — the same number
     * `scripts/k3_shapes.py` files as `k3-moe-shared-gateup`, which is 768 at the reference TP8. */
    printf("Kimi-K3 shared-expert prefill: K = hidden = 7168, N = imoe_l = 6144/TP "
           "(TP1 6144 / TP4 1536 / TP8 768)\n");

    const unsigned K = 7168;
    const unsigned Ts[] = {128, 512, 2048, 4096, 8192};
    const unsigned Ns[] = {6144, 1536, 768};
    for (unsigned n : Ns)
        for (unsigned t : Ts) run_shape(t, n, K);

    printf("\n%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
#endif

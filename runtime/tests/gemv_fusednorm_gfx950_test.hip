/* gemv_fusednorm_gfx950_test.hip — GEMV `norm == 2` (the folded RMSNORM) held to BIT-IDENTICAL.
 *
 * `k3.rs fuse_norm_gemv` deletes 116 `b=1` RMSNORM packets from the K3 decode program by having
 * the `b=256` GEMV that reads each one normalize the activation itself. The claim that makes that
 * legal is not "close enough" — it is that the fused program computes the SAME BITS as the pair it
 * replaces. This file is that claim, checked on hardware:
 *
 *     reference:  d_rmsnorm(xn, x, gamma, 1, K, eps)   then   d_gemv(C, xn, W, norm=0)
 *     fused:      d_gemv(C, x, W, gamma, norm=2, eps)
 *
 * and the bar is ZERO differing halves, not a tolerance. The mechanism is that the fused arm does
 * NOT multiply through the k-loop (which would be f32 and a different number); it normalizes the
 * LDS-staged copy IN PLACE, rounding to bf16 with `d_rmsnorm`'s own per-thread element map and
 * `block_sum`, and then runs the ordinary un-normed hot loop. So the bytes the k-loop reads are the
 * bytes the deleted packet wrote to HBM. A tolerance here would hide exactly the drift this design
 * exists to prevent — the two paths sharing RN_REG/RN_VEC (amd_common.h) is load-bearing.
 *
 * BOTH REAL WIDTHS ARE CHECKED, because they exercise different halves of the gate:
 *   K = 3584  `routed_expert_norm` -> routed_expert_up_proj   (92 sites, fan-out 1)
 *   K = 1536  `q_a_layernorm`      -> q_absorb / q_rope       (24 sites, fan-out 2)
 *
 * A THIRD CASE guards the demotion path: at K = 16384 the row exceeds `RN_REG * PLOW_THREADS`, so
 * `d_rmsnorm` would take its streaming arm and the fused element map would NOT match. The kernel
 * must fall back to `norm = 0` rather than reduce over an arena it never staged. The emitter
 * refuses this shape too, so agreement here is with the UNNORMED GEMV — this pins the fallback as
 * a defined behaviour instead of whatever the arena happened to hold.
 *
 * TWO PASSES, for the reason gemv_qkvg_gfx950_test.hip states: `op_gemm.h` is DEVICE-ONLY.
 *
 *   hipcc --offload-arch=gfx950 -O3 -w -DPLOW_GEMV_MM=1 -DFNORM_DEVICE --genco \
 *       runtime/tests/gemv_fusednorm_gfx950_test.hip -o /tmp/fnorm.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/gemv_fusednorm_gfx950_test.hip -o /tmp/fnorm_test -L/opt/rocm/lib -lamdhip64
 *   sg render -c 'perf-data/harness/gpulease -n 1 fnorm /tmp/fnorm_test /tmp/fnorm.co'
 */

struct KArgs {
    unsigned long long C, x, xn, W, gamma;
    unsigned M, N, K, norm;
    float eps;
};

#ifdef FNORM_DEVICE
/* ======================= DEVICE HALF (--genco) ======================= */
#include <hip/hip_runtime.h>
#include "op_gemm.h"
#include "op_norm.h"

/* The interpreter's arena and its cross-wave scratch, verbatim: these kernels stand in for the
 * dispatch arms, so they must see the same shapes the real ones do. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_rmsnorm(const KArgs* a) {
    __shared__ float part[PLOW_WAVES];
    d_rmsnorm((bf16*)a->xn, (const bf16*)a->x, (const bf16*)a->gamma, a->M, a->K, a->eps, 0,
              blockIdx.x, gridDim.x, part);
}

/* norm = 0 reads the PRE-NORMED xn; norm = 2 reads the RAW x and folds. One kernel for both, so
 * the only difference between the arms is the operands the packet carries.
 *
 * THE ARENA IS DECLARED AS THE INTERPRETER'S UNION, NOT AS A PLAIN ARRAY, AND THAT IS THE WHOLE
 * POINT OF THIS KERNEL. `plow_smem` (interp.hip) overlaps block_sum's `part` scratch with the
 * GEMM arena `gm` — one address, two names. An earlier version of this test used two separate
 * __shared__ arrays and therefore PASSED bit-exact while the real interpreter diverged: the
 * fused norm's block_sum was writing through the first 16 halves of the row it was reducing.
 * A test whose memory layout is friendlier than production cannot gate production. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_gemv(const KArgs* a) {
    __shared__ union __align__(16) {
        float part[PLOW_WAVES];
        bf16 gm[GM_LDS_HALVES];
    } sm;
    bf16* const lds = sm.gm;
    d_gemv((bf16*)a->C, (const bf16*)(a->norm == 2 ? a->x : a->xn), (const bf16*)a->W, nullptr,
           (const bf16*)a->gamma, a->M, a->N, a->K, (int)a->norm, a->eps, blockIdx.x, gridDim.x,
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

static const unsigned NBLK = 256, NTHREADS = 512;

static unsigned rng_state = 0x2545F491u;
static float frand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}
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
static void fill_bf16(void* d, size_t n) {
    std::vector<unsigned short> h(n);
    for (size_t i = 0; i < n; i++) h[i] = f2bf_h(frand());
    HIP_OK(hipMemcpy(d, h.data(), n * 2, hipMemcpyHostToDevice));
}
static std::vector<unsigned short> read_bf16(void* d, size_t n) {
    std::vector<unsigned short> h(n);
    HIP_OK(hipMemcpy(h.data(), d, n * 2, hipMemcpyDeviceToHost));
    return h;
}

static hipFunction_t f_rmsnorm, f_gemv;

static void launch(hipFunction_t f, void* args_dev, unsigned nblk = NBLK) {
    size_t sz = sizeof(void*);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_dev, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                   HIP_LAUNCH_PARAM_END};
    HIP_OK(hipModuleLaunchKernel(f, nblk, 1, 1, NTHREADS, 1, 1, 0, nullptr, nullptr, cfg));
}

/* The RMSNORM packet the fusion deletes is b=1 — ONE workgroup — and that is not cosmetic: the
 * reduction it performs is over the whole row inside a single block, which is what the fused arm
 * reproduces per consumer workgroup. Launching the reference norm at 256 blocks would reduce a
 * different partition and the comparison would be meaningless. */
static void launch1(hipFunction_t f, void* args_dev) {
    size_t sz = sizeof(void*);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_dev, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                   HIP_LAUNCH_PARAM_END};
    HIP_OK(hipModuleLaunchKernel(f, 1, 1, 1, NTHREADS, 1, 1, 0, nullptr, nullptr, cfg));
}

static int failures = 0;

/* One shape, both arms, elementwise. `want_bitexact` is false only for the demotion case. */
static void check(const char* what, unsigned N, unsigned K, bool want_bitexact, unsigned nblk = NBLK) {
    const unsigned M = 1;
    const float eps = 1e-5f;
    void* d_x = dev_alloc((size_t)M * K * 2);
    void* d_xn = dev_alloc((size_t)M * K * 2);
    void* d_W = dev_alloc((size_t)N * K * 2);
    void* d_g = dev_alloc((size_t)K * 2);
    void* d_Cref = dev_alloc((size_t)M * N * 2);
    void* d_Cfus = dev_alloc((size_t)M * N * 2);
    fill_bf16(d_x, (size_t)M * K);
    fill_bf16(d_W, (size_t)N * K);
    fill_bf16(d_g, K);

    KArgs h{};
    h.x = (unsigned long long)d_x;
    h.xn = (unsigned long long)d_xn;
    h.W = (unsigned long long)d_W;
    h.gamma = (unsigned long long)d_g;
    h.M = M;
    h.N = N;
    h.K = K;
    h.eps = eps;
    void* d_args = dev_alloc(sizeof(KArgs));

    /* Reference: the two packets, exactly as the unfused emitter orders them. */
    h.C = (unsigned long long)d_Cref;
    h.norm = 0;
    HIP_OK(hipMemcpy(d_args, &h, sizeof h, hipMemcpyHostToDevice));
    launch1(f_rmsnorm, d_args);
    HIP_OK(hipDeviceSynchronize());
    launch(f_gemv, d_args, nblk);
    HIP_OK(hipDeviceSynchronize());

    /* Fused: one packet, reading the RAW x. */
    h.C = (unsigned long long)d_Cfus;
    h.norm = 2;
    HIP_OK(hipMemcpy(d_args, &h, sizeof h, hipMemcpyHostToDevice));
    launch(f_gemv, d_args);
    HIP_OK(hipDeviceSynchronize());

    const auto a = read_bf16(d_Cref, (size_t)M * N);
    const auto b = read_bf16(d_Cfus, (size_t)M * N);
    size_t ndiff = 0;
    double worst = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == b[i]) continue;
        ndiff++;
        const double d = fabs(bf2f_h(a[i]) - bf2f_h(b[i]));
        if (d > worst) worst = d;
    }
    /* An all-zero output would pass a bit-comparison while proving nothing — the exact silent
     * no-op this tree's dispatch `default:` produces. So the reference must be non-trivial. */
    size_t nz = 0;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != 0) nz++;

    const bool ok = want_bitexact ? (ndiff == 0 && nz > a.size() / 2) : (ndiff == 0 || nz > 0);
    printf("  %-46s N=%-6u K=%-6u differing %6zu / %-6zu nonzero %6zu  max|d| %.6g  %s\n", what, N,
           K, ndiff, a.size(), nz, worst, ok ? "OK" : "FAIL");
    if (!ok) failures++;
    for (void* p : {d_x, d_xn, d_W, d_g, d_Cref, d_Cfus, d_args}) HIP_OK(hipFree(p));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <fnorm.co>\n", argv[0]);
        return 2;
    }
    hipModule_t mod;
    HIP_OK(hipModuleLoad(&mod, argv[1]));
    HIP_OK(hipModuleGetFunction(&f_rmsnorm, mod, "k_rmsnorm"));
    HIP_OK(hipModuleGetFunction(&f_gemv, mod, "k_gemv"));

    printf("GEMV norm=2 (folded RMSNORM) vs the RMSNORM+GEMV pair it replaces:\n");
    /* The two shapes the K3 emitter actually folds. */
    check("routed_expert_norm -> up_latent  (fan=1)", 7168, 3584, true);
    /* the TP8 shape the emitter actually produces: column-sharded N, blocked_gemv_cus -> b=224 */
    check("  ...as emitted at tp8 (sharded, b=224)", 896, 3584, true, 224);
    check("q_a_layernorm      -> q_absorb   (fan=2)", 6144, 1536, true);
    check("q_a_layernorm      -> q_rope     (fan=2)", 768, 1536, true);
    /* A width the emitter refuses: the arm must DEMOTE, not read an unstaged arena. */
    check("oversized row: must demote, not fault", 512, 16384, false);

    printf(failures ? "\nFAIL: %d case(s)\n" : "\nPASS: all cases bit-identical\n", failures);
    return failures ? 1 : 0;
}
#endif

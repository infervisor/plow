/* kda_step_cdna3_test.hip — the KDA recurrence (ops 102/112) and the gated output norm (op 103)
 * against an f64 host reference, at Kimi-K3's exact geometry, on whatever part runs it.
 *                                                                              [K3-KDA-CDNA3]
 * WHY THIS EXISTS when kda_block_gfx950_test already gates the layer: that gate needs the REAL
 * checkpoint (its fixture is written by kda_real_oracle.py from safetensors), and the K3 snapshot
 * is 1.4 TiB that most boxes — including the gfx942 box this was written on — do not have. The
 * KDA arms are pure VALU and compile everywhere, but "compiles on gfx942" and "computes §2.2 of
 * docs/kimi-k3-kda.md on gfx942" are different claims. This closes the gap with synthetic
 * weights: the recurrence has no learned structure to miss — any full-rank input distinguishes a
 * correct state walk from a transposed, un-decayed, or mis-gated one.
 *
 * WHAT IS PINNED, all at H=96 D=128 (the only KDA shape that exists):
 *   - op 112 (`d_kda_state_step_g`), the PRODUCTION form: bounded gate lb*sigmoid(exp(A_log)*
 *     (g+dt_bias)) inlined, beta = sigmoid(raw), qk L2 norm with eps INSIDE the sqrt, q scaled
 *     by D^-1/2, decay -> delta -> rank-1 write -> read of the UPDATED state, V-FIRST [h][v][k]
 *     state layout. The reference implements the spec, not the kernel, so a shared mistake would
 *     have to be made twice independently.
 *   - PREFILL mode (bstride=0): T=5 tokens threading serially through ONE state.
 *   - BATCHED-DECODE mode (bstride=H*D*D): 3 rows, each its OWN state — the axis fold whose
 *     absence ran batch-16 decode serially (see the kernel's header).
 *   - op 103 (`d_kda_gated_norm`): RMSNorm over D inside a head, one shared [D] weight, sigmoid
 *     on the RAW gate, gate applied AFTER the norm.
 *
 * Two passes, op_kda.h is device-only by include chain:
 *   hipcc --offload-arch=gfx942 -O3 -w -DKDA3_DEVICE --genco \
 *       runtime/tests/kda_step_cdna3_test.hip -o /tmp/kda3.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/kda_step_cdna3_test.hip -o /tmp/kda3 -L/opt/rocm/lib -lamdhip64
 *   perf-data/harness/gpulease -n 1 kda3 '/tmp/kda3 /tmp/kda3.co'
 */

#ifdef KDA3_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_elementwise.h" /* act_silu, used by op_kda's conv */
#include "op_kda.h"

struct KArgs {
    unsigned long long o, q, k, v, g_raw, beta_raw, a_log, dt_bias, state, y, norm_w, gate2;
    unsigned T, H, D, BV, flags, gate_mode;
    float lb, scale, eps;
    unsigned long long bstride;
};

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_step(KArgs* a) {
    __shared__ float lds[4 * 128 + 2 * PLOW_WAVES];
    d_kda_state_step_g((bf16*)a->o, (const bf16*)a->q, (const bf16*)a->k, (const bf16*)a->v,
                       (const bf16*)a->g_raw, (const bf16*)a->beta_raw, (const float*)a->a_log,
                       (const float*)a->dt_bias, a->gate_mode, a->lb, (float*)a->state, a->T, a->H,
                       a->D, a->BV, a->flags, a->scale, blockIdx.x, gridDim.x, lds,
                       (size_t)a->bstride, nullptr);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_norm(KArgs* a) {
    d_kda_gated_norm((bf16*)a->y, (const bf16*)a->o, (const float*)a->norm_w,
                     (const bf16*)a->gate2, a->T, a->H, a->D, a->eps, blockIdx.x, gridDim.x);
}

#else
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

typedef unsigned short bf16;
static float b2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2b(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }
#define HC(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); return 1; }        \
    } while (0)

struct KArgs {
    unsigned long long o, q, k, v, g_raw, beta_raw, a_log, dt_bias, state, y, norm_w, gate2;
    unsigned T, H, D, BV, flags, gate_mode;
    float lb, scale, eps;
    unsigned long long bstride;
};

/* §2.2 of docs/kimi-k3-kda.md, in f64, from the bf16-rounded inputs the device sees. */
static void ref_step(unsigned T, unsigned H, unsigned D, const std::vector<bf16>& q,
                     const std::vector<bf16>& k, const std::vector<bf16>& v,
                     const std::vector<bf16>& graw, const std::vector<bf16>& braw,
                     const std::vector<float>& alog, const std::vector<float>& dtb, double lb,
                     double scale, std::vector<double>& S /* [rows?][H][V][K] */,
                     std::vector<double>& o, bool rows_independent) {
    std::vector<double> qn(D), kn(D), a(D);
    for (unsigned t = 0; t < T; t++) {
        const unsigned srow = rows_independent ? t : 0;
        for (unsigned h = 0; h < H; h++) {
            const size_t hd = (size_t)t * H * D + (size_t)h * D;
            double qs = 0, ks = 0;
            for (unsigned d = 0; d < D; d++) {
                qn[d] = b2f(q[hd + d]); ks += (double)b2f(k[hd + d]) * b2f(k[hd + d]);
                qs += qn[d] * qn[d]; kn[d] = b2f(k[hd + d]);
            }
            const double rq = scale / sqrt(qs + 1e-6), rk = 1.0 / sqrt(ks + 1e-6);
            for (unsigned d = 0; d < D; d++) { qn[d] *= rq; kn[d] *= rk; }
            const double ah = exp((double)alog[h]);
            for (unsigned d = 0; d < D; d++)
                a[d] = exp(lb * sigm(ah * ((double)b2f(graw[hd + d]) + (double)dtb[h * D + d])));
            const double beta = sigm((double)b2f(braw[(size_t)t * H + h]));
            double* Sh = S.data() + ((size_t)srow * H + h) * D * D; /* V-FIRST [v][k] */
            for (unsigned j = 0; j < D; j++) {
                double* col = Sh + (size_t)j * D;
                double pk = 0;
                for (unsigned d = 0; d < D; d++) { col[d] *= a[d]; pk += col[d] * kn[d]; }
                const double u = (double)b2f(v[hd + j]) - pk;
                double pq = 0;
                for (unsigned d = 0; d < D; d++) {
                    col[d] += beta * u * kn[d];
                    pq += col[d] * qn[d];
                }
                o[hd + j] = pq;
            }
        }
    }
}

int main(int argc, char** argv) {
    hipDeviceProp_t p; HC(hipGetDeviceProperties(&p, 0));
    printf("dev: %s\n", p.gcnArchName);
    hipModule_t M; HC(hipModuleLoad(&M, argc > 1 ? argv[1] : "/tmp/kda3.co"));
    hipFunction_t fstep, fnorm;
    HC(hipModuleGetFunction(&fstep, M, "k_step"));
    HC(hipModuleGetFunction(&fnorm, M, "k_norm"));

    const unsigned H = 96, D = 128, BV = 16, GRID = 37;
    srand(1337);

    auto run_mode = [&](const char* nm, unsigned T, bool batched) -> double {
        const unsigned nstate = batched ? T : 1;
        std::vector<bf16> q(T * H * D), k(q.size()), v(q.size()), graw(q.size()), braw(T * H);
        std::vector<float> alog(H), dtb(H * D);
        for (auto& x : q) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : k) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : v) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : graw) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        for (auto& x : braw) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        /* checkpoint-shaped: exp(A_log) in [0.5, 12], dt_bias mean ~ -4.6 */
        for (auto& x : alog) x = logf(0.5f + (float)rand() / RAND_MAX * 11.5f);
        for (auto& x : dtb) x = -7.f + (float)rand() / RAND_MAX * 6.f;
        std::vector<float> S0((size_t)nstate * H * D * D);
        for (auto& x : S0) x = ((float)rand() / RAND_MAX - .5f) * .5f;

        std::vector<double> Sref(S0.begin(), S0.end()), oref((size_t)T * H * D, 0.0);
        ref_step(T, H, D, q, k, v, graw, braw, alog, dtb, -5.0, 1.0 / sqrt(128.0), Sref, oref,
                 batched);

#define UP(vec, elt) ({ void* d_; HC2(hipMalloc(&d_, (vec).size() * sizeof(elt)));                \
        HC2(hipMemcpy(d_, (vec).data(), (vec).size() * sizeof(elt), hipMemcpyHostToDevice)); d_; })
#define HC2(x) if (hipError_t e_ = (x)) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); }
        KArgs a{};
        a.q = (unsigned long long)(size_t)UP(q, bf16); a.k = (unsigned long long)(size_t)UP(k, bf16);
        a.v = (unsigned long long)(size_t)UP(v, bf16);
        a.g_raw = (unsigned long long)(size_t)UP(graw, bf16);
        a.beta_raw = (unsigned long long)(size_t)UP(braw, bf16);
        a.a_log = (unsigned long long)(size_t)UP(alog, float);
        a.dt_bias = (unsigned long long)(size_t)UP(dtb, float);
        a.state = (unsigned long long)(size_t)UP(S0, float);
        void* d_o; HC(hipMalloc(&d_o, (size_t)T * H * D * 2)); HC(hipMemset(d_o, 0, (size_t)T * H * D * 2));
        a.o = (unsigned long long)(size_t)d_o;
        a.T = T; a.H = H; a.D = D; a.BV = BV;
        a.flags = 1u /* PLOW_KDA_F_QK_L2NORM */; a.gate_mode = 1u /* LOWER_BOUND */;
        a.lb = -5.0f; a.scale = (float)(1.0 / sqrt(128.0)); a.eps = 1e-5f;
        a.bstride = batched ? (unsigned long long)H * D * D : 0ull;
        void* d_a; HC(hipMalloc(&d_a, sizeof a)); HC(hipMemcpy(d_a, &a, sizeof a, hipMemcpyHostToDevice));
        void* args[] = {&d_a};
        HC(hipModuleLaunchKernel(fstep, GRID, 1, 1, 512, 1, 1, 0, 0, args, nullptr));
        HC(hipDeviceSynchronize());

        std::vector<float> Sg(S0.size());
        std::vector<bf16> og((size_t)T * H * D);
        HC(hipMemcpy(Sg.data(), (void*)(size_t)a.state, Sg.size() * 4, hipMemcpyDeviceToHost));
        HC(hipMemcpy(og.data(), d_o, og.size() * 2, hipMemcpyDeviceToHost));
        double se = 0, sw = 0, se2 = 0, sw2 = 0;
        for (size_t i = 0; i < Sg.size(); i++) {
            const double d_ = Sg[i] - Sref[i]; se += d_ * d_; sw += Sref[i] * Sref[i];
        }
        for (size_t i = 0; i < og.size(); i++) {
            const double d_ = b2f(og[i]) - oref[i]; se2 += d_ * d_; sw2 += oref[i] * oref[i];
        }
        const double srms = sqrt(se / (sw + 1e-30)), orms = sqrt(se2 / (sw2 + 1e-30));
        printf("  %-18s T=%u  state rms rel %.3e (bar 1e-4)   o rms rel %.3e (bar 1e-2)\n", nm, T,
               srms, orms);
        /* op 103 on this mode's output */
        std::vector<float> nw(D);
        std::vector<bf16> g2((size_t)T * H * D);
        for (auto& x : nw) x = .5f + (float)rand() / RAND_MAX;
        for (auto& x : g2) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        a.y = (unsigned long long)(size_t)UP(g2, bf16); /* reuse as y buffer, overwritten */
        a.norm_w = (unsigned long long)(size_t)UP(nw, float);
        a.gate2 = (unsigned long long)(size_t)UP(g2, bf16);
        HC(hipMemcpy(d_a, &a, sizeof a, hipMemcpyHostToDevice));
        HC(hipModuleLaunchKernel(fnorm, GRID, 1, 1, 512, 1, 1, 0, 0, args, nullptr));
        HC(hipDeviceSynchronize());
        std::vector<bf16> yg((size_t)T * H * D);
        HC(hipMemcpy(yg.data(), (void*)(size_t)a.y, yg.size() * 2, hipMemcpyDeviceToHost));
        double se3 = 0, sw3 = 0;
        for (unsigned t = 0; t < T; t++)
            for (unsigned h = 0; h < H; h++) {
                const size_t base = ((size_t)t * H + h) * D;
                double ss = 0;
                for (unsigned d = 0; d < D; d++) { const double x = b2f(og[base + d]); ss += x * x; }
                const double inv = 1.0 / sqrt(ss / D + 1e-5);
                for (unsigned d = 0; d < D; d++) {
                    const double want =
                        (double)b2f(og[base + d]) * inv * nw[d] * sigm((double)b2f(g2[base + d]));
                    const double d_ = b2f(yg[base + d]) - want;
                    se3 += d_ * d_; sw3 += want * want;
                }
            }
        const double yrms = sqrt(se3 / (sw3 + 1e-30));
        printf("  %-18s gated-norm rms rel %.3e (bar 1e-2)\n", nm, yrms);
        return fmax(fmax(srms / 1e-4, orms / 1e-2), yrms / 1e-2);
    };

    double worst = 0;
    worst = fmax(worst, run_mode("prefill (serial)", 5, false));
    worst = fmax(worst, run_mode("batched decode", 3, true));
    printf("%s\n", worst < 1.0 ? "PASS" : "FAIL");
    return worst >= 1.0;
}
#endif

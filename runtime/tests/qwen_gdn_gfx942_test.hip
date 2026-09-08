/* qwen_gdn_gfx942_test.hip — the CPU/f64 oracle gate for runtime/amd/op_qwen_gdn.h.
 *
 * Ten bodies, ten oracles. Each oracle implements the SPEC (runtime/nvidia/op_qwen_gdn.cuh read as
 * maths, and vLLM's Qwen3-Next reference behind it) in double precision from the bf16-rounded
 * inputs the device sees — deliberately NOT a transliteration of the kernel, so a shared mistake
 * would have to be made twice, independently, in two different index orders. The state layout in
 * particular is spelled out here as [slot][head][v][k] rather than mirrored from the kernel: with
 * vdim == kdim a transposed state has exactly the right norm, so no magnitude check finds it and
 * the independent indexing IS the assertion.
 *
 * WHY THIS EXISTS AT ALL. AMD's interpreter dispatch `default:` writes NOTHING rather than
 * trapping, so a Gated DeltaNet arm that is subtly wrong — or absent — produces fluent output, not
 * a fault. A linear-attention recurrence compounds a per-step error across the whole sequence and
 * is invisible in serving text. Nothing here may be taken on faith from a serving run.
 *
 * Scoring: every check reports `measured / bar`; `worst = fmax(...)`; PASS is `worst < 1.0`.
 * Bars are printed next to the measurement so a regression is legible without reading this file.
 *
 * Build and run (GRID is deliberately not a divisor of any extent, to exercise a ragged nblk):
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -DQGDN_DEVICE --genco \
 *       runtime/tests/qwen_gdn_gfx942_test.hip -o /tmp/qgdn.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/qwen_gdn_gfx942_test.hip -o /tmp/qgdn -L/opt/rocm/lib -lamdhip64
 *   perf-data/tools/gpulease -n 1 qgdn '/tmp/qgdn /tmp/qgdn.co'
 */

#ifdef QGDN_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_qwen_gdn.h"

struct QArgs {
    unsigned long long out, in, weight, history, active, qkv, a, b, a_log, dt_bias, state;
    unsigned long long core, z, gamma, q, k, v, packed, gate, cos, sin, positions, alpha, beta;
    unsigned channels, width, batch, hk, hv, kdim, vdim, heads, dim, cols, tokens;
    unsigned rotary, context, normalize, alog_f32, prefill;
    float scale, eps, offset;
};

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_conv(QArgs* a) {
    d_qwen_gdn_conv((bf16*)a->out, (const bf16*)a->in, (const bf16*)a->weight, (bf16*)a->history,
                    (const int*)a->active, a->channels, a->width, a->batch, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_step(QArgs* a) {
    d_qwen_gdn_step((bf16*)a->out, (const bf16*)a->qkv, (const bf16*)a->a, (const bf16*)a->b,
                    (const void*)a->a_log, (const bf16*)a->dt_bias, (float*)a->state,
                    (const int*)a->active, a->hk, a->hv, a->kdim, a->vdim, a->batch, a->scale,
                    a->eps, a->alog_f32, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_gated_norm(QArgs* a) {
    d_qwen_gated_norm((bf16*)a->out, (const bf16*)a->core, (const bf16*)a->z,
                      (const bf16*)a->gamma, (const int*)a->active, a->heads, a->dim, a->batch,
                      a->eps, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_split(QArgs* a) {
    d_qwen_q_gate_split((bf16*)a->q, (bf16*)a->gate, (const bf16*)a->packed, (const int*)a->active,
                        a->heads, a->dim, a->batch, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_sigmoid_gate(QArgs* a) {
    d_qwen_sigmoid_gate((bf16*)a->out, (const bf16*)a->in, (const bf16*)a->gate,
                        (const int*)a->active, a->cols, a->batch, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_rmsnorm(QArgs* a) {
    d_qwen_rmsnorm((bf16*)a->out, (const bf16*)a->in, (const bf16*)a->gamma, (const int*)a->active,
                   a->dim, a->batch, a->eps, a->offset, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_headnorm_rope(QArgs* a) {
    d_qwen_headnorm_rope((bf16*)a->out, (const bf16*)a->in, (const bf16*)a->gamma,
                         (const float*)a->cos, (const float*)a->sin, (const int*)a->positions,
                         (const int*)a->active, a->heads, a->dim, a->rotary, a->batch, a->context,
                         a->normalize, a->eps, a->offset, blockIdx.x, gridDim.x, a->prefill);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_conv_prefill(QArgs* a) {
    d_qwen_gdn_conv_prefill((bf16*)a->out, (const bf16*)a->in, (const bf16*)a->weight,
                            (bf16*)a->history, a->channels, a->width, a->tokens, blockIdx.x,
                            gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_qkv_prep(QArgs* a) {
    d_qwen_gdn_qkv_prep((bf16*)a->q, (bf16*)a->k, (bf16*)a->v, (const bf16*)a->packed, a->hk,
                        a->hv, a->kdim, a->vdim, a->tokens, a->eps, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_gate_prep(QArgs* a) {
    d_qwen_gdn_gate_prep((float*)a->alpha, (float*)a->beta, (const bf16*)a->a, (const bf16*)a->b,
                         (const bf16*)a->a_log, (const bf16*)a->dt_bias, a->hv, a->tokens,
                         blockIdx.x, gridDim.x);
}

#else
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifndef QGDN_THREADS
#define QGDN_THREADS 512
#endif

typedef unsigned short bf16;
static float b2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2b(float f) {
    unsigned u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }
/* The reference rounds beta and the sigmoid gate to bf16 before use; the oracle must too. */
static double rbeta(bf16 raw) { return (double)b2f(f2b((float)sigm((double)b2f(raw)))); }
static double softplus(double x) { return x > 20.0 ? x : log1p(exp(x)); }

#define HC(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); return 1; }          \
    } while (0)

struct QArgs {
    unsigned long long out, in, weight, history, active, qkv, a, b, a_log, dt_bias, state;
    unsigned long long core, z, gamma, q, k, v, packed, gate, cos, sin, positions, alpha, beta;
    unsigned channels, width, batch, hk, hv, kdim, vdim, heads, dim, cols, tokens;
    unsigned rotary, context, normalize, alog_f32, prefill;
    float scale, eps, offset;
};

static double rnd() { return (double)rand() / RAND_MAX - 0.5; }

/* rms-relative and max-relative of a bf16 result against an f64 oracle. */
struct Err { double rms, mx; };
static Err cmp_bf16(const std::vector<bf16>& got, const std::vector<double>& want) {
    double se = 0, sw = 0, mx = 0, scale = 0;
    for (size_t i = 0; i < want.size(); i++) scale = fmax(scale, fabs(want[i]));
    for (size_t i = 0; i < want.size(); i++) {
        const double d = (double)b2f(got[i]) - want[i];
        se += d * d; sw += want[i] * want[i];
        mx = fmax(mx, fabs(d));
    }
    return {sqrt(se / (sw + 1e-30)), mx / (scale + 1e-30)};
}
static Err cmp_f32(const std::vector<float>& got, const std::vector<double>& want) {
    double se = 0, sw = 0, mx = 0, scale = 0;
    for (size_t i = 0; i < want.size(); i++) scale = fmax(scale, fabs(want[i]));
    for (size_t i = 0; i < want.size(); i++) {
        const double d = (double)got[i] - want[i];
        se += d * d; sw += want[i] * want[i];
        mx = fmax(mx, fabs(d));
    }
    return {sqrt(se / (sw + 1e-30)), mx / (scale + 1e-30)};
}

static double worst = 0.0;
static void score(const char* nm, Err e, double rms_bar, double max_bar) {
    printf("  %-22s rms rel %.3e (bar %.0e)   max rel %.3e (bar %.0e)%s\n", nm, e.rms, rms_bar,
           e.mx, max_bar, (e.rms < rms_bar && e.mx < max_bar) ? "" : "   <-- FAIL");
    worst = fmax(worst, fmax(e.rms / rms_bar, e.mx / max_bar));
}

int main(int argc, char** argv) {
    hipDeviceProp_t p;
    HC(hipGetDeviceProperties(&p, 0));
    printf("dev: %s   threads %u\n", p.gcnArchName, (unsigned)QGDN_THREADS);
    hipModule_t M;
    HC(hipModuleLoad(&M, argc > 1 ? argv[1] : "/tmp/qgdn.co"));
    hipFunction_t f_conv, f_step, f_gnorm, f_split, f_sgate, f_rms, f_hnr, f_convpf, f_prep, f_gp;
    HC(hipModuleGetFunction(&f_conv, M, "k_conv"));
    HC(hipModuleGetFunction(&f_step, M, "k_step"));
    HC(hipModuleGetFunction(&f_gnorm, M, "k_gated_norm"));
    HC(hipModuleGetFunction(&f_split, M, "k_split"));
    HC(hipModuleGetFunction(&f_sgate, M, "k_sigmoid_gate"));
    HC(hipModuleGetFunction(&f_rms, M, "k_rmsnorm"));
    HC(hipModuleGetFunction(&f_hnr, M, "k_headnorm_rope"));
    HC(hipModuleGetFunction(&f_convpf, M, "k_conv_prefill"));
    HC(hipModuleGetFunction(&f_prep, M, "k_qkv_prep"));
    HC(hipModuleGetFunction(&f_gp, M, "k_gate_prep"));

    /* Qwen3.5 geometry: 16 key heads / 48 value heads, 128-wide k and v, conv width 4, a 256-wide
     * full-attention head with a 64-wide partial rotary. GRID divides no extent below. */
    const unsigned HK = 16, HV = 48, DK = 128, DV = 128, WIDTH = 4, HD = 256, ROT = 64;
    const unsigned GRID = 37, BATCH = 3, TOKENS = 11;
    const unsigned CHANNELS = 2 * HK * DK + HV * DV;
    srand(1337);

    QArgs a{};
    a.scale = (float)(1.0 / sqrt((double)DK));
    a.eps = 1e-6f;
    void* d_a;
    HC(hipMalloc(&d_a, sizeof a));
    void* args[] = {&d_a};
    auto sync = [&](hipFunction_t f) -> int {
        HC(hipMemcpy(d_a, &a, sizeof a, hipMemcpyHostToDevice));
        HC(hipModuleLaunchKernel(f, GRID, 1, 1, QGDN_THREADS, 1, 1, 0, 0, args, nullptr));
        HC(hipDeviceSynchronize());
        return 0;
    };
#define UP(vec, elt)                                                                               \
    ({                                                                                             \
        void* d_;                                                                                  \
        HC(hipMalloc(&d_, (vec).size() * sizeof(elt)));                                            \
        HC(hipMemcpy(d_, (vec).data(), (vec).size() * sizeof(elt), hipMemcpyHostToDevice));         \
        (unsigned long long)(size_t)d_;                                                            \
    })
#define OUT(n, elt)                                                                                \
    ({                                                                                             \
        void* d_;                                                                                  \
        HC(hipMalloc(&d_, (n) * sizeof(elt)));                                                      \
        HC(hipMemset(d_, 0, (n) * sizeof(elt)));                                                    \
        (unsigned long long)(size_t)d_;                                                            \
    })

    /* Every decode-shaped op takes an `active` mask; a slot that is off must be left untouched.
     * Slot 1 is off in every such case below and its oracle rows stay at the buffer's initial
     * zero, so an arm that ignores the mask writes a value there and fails. */
    std::vector<int> active{1, 0, 1};
    const unsigned long long d_active = UP(active, int);

    /* ---- op 136: decode conv + SiLU, with the history shift ---- */
    {
        std::vector<bf16> in((size_t)BATCH * CHANNELS), w((size_t)CHANNELS * WIDTH);
        std::vector<bf16> hist((size_t)BATCH * CHANNELS * (WIDTH - 1));
        for (auto& x : in) x = f2b((float)(rnd() * 2.0));
        for (auto& x : w) x = f2b((float)(rnd() * 1.0));
        for (auto& x : hist) x = f2b((float)(rnd() * 2.0));
        std::vector<bf16> h0 = hist;
        std::vector<double> oref((size_t)BATCH * CHANNELS, 0.0);
        std::vector<double> href(hist.size(), 0.0);
        for (size_t i = 0; i < hist.size(); i++) href[i] = (double)b2f(hist[i]);
        for (unsigned s = 0; s < BATCH; s++)
            for (unsigned c = 0; c < CHANNELS; c++) {
                const size_t row = (size_t)s * CHANNELS + c;
                if (!active[s]) { oref[row] = 0.0; continue; }
                double sum = 0.0;
                for (unsigned j = 0; j + 1 < WIDTH; j++)
                    sum += (double)b2f(hist[row * (WIDTH - 1) + j]) * (double)b2f(w[(size_t)c * WIDTH + j]);
                sum += (double)b2f(in[row]) * (double)b2f(w[(size_t)c * WIDTH + WIDTH - 1]);
                oref[row] = sum * sigm(sum);
                for (unsigned j = 0; j + 2 < WIDTH; j++)
                    href[row * (WIDTH - 1) + j] = (double)b2f(hist[row * (WIDTH - 1) + j + 1]);
                href[row * (WIDTH - 1) + WIDTH - 2] = (double)b2f(in[row]);
            }
        a.in = UP(in, bf16); a.weight = UP(w, bf16); a.history = UP(hist, bf16);
        a.out = OUT((size_t)BATCH * CHANNELS, bf16);
        a.active = d_active; a.channels = CHANNELS; a.width = WIDTH; a.batch = BATCH;
        sync(f_conv);
        std::vector<bf16> got((size_t)BATCH * CHANNELS), hgot(hist.size());
        HC(hipMemcpy(got.data(), (void*)(size_t)a.out, got.size() * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(hgot.data(), (void*)(size_t)a.history, hgot.size() * 2, hipMemcpyDeviceToHost));
        score("gdn_conv out", cmp_bf16(got, oref), 4e-3, 4e-3);
        score("gdn_conv history", cmp_bf16(hgot, href), 1e-12, 1e-12); /* pure copies: exact */
    }

    /* ---- op 137: the decode recurrence ---- */
    {
        std::vector<bf16> qkv((size_t)BATCH * CHANNELS), av((size_t)BATCH * HV);
        std::vector<bf16> bv((size_t)BATCH * HV), alog(HV), dtb(HV);
        for (auto& x : qkv) x = f2b((float)(rnd() * 2.0));
        for (auto& x : av) x = f2b((float)(rnd() * 4.0));
        for (auto& x : bv) x = f2b((float)(rnd() * 4.0));
        /* checkpoint-shaped: exp(A_log) in [0.5, 12], dt_bias mean ~ -4.6 */
        for (auto& x : alog) x = f2b(logf(0.5f + (float)(rnd() + 0.5) * 11.5f));
        for (auto& x : dtb) x = f2b((float)(-7.0 + (rnd() + 0.5) * 6.0));
        std::vector<float> S0((size_t)BATCH * HV * DV * DK);
        for (auto& x : S0) x = (float)(rnd() * 0.5);

        std::vector<double> Sref(S0.begin(), S0.end());
        std::vector<double> oref((size_t)BATCH * HV * DV, 0.0);
        for (unsigned s = 0; s < BATCH; s++) {
            if (!active[s]) continue;
            const bf16* x = qkv.data() + (size_t)s * CHANNELS;
            for (unsigned hd_ = 0; hd_ < HV; hd_++) {
                const unsigned qh = hd_ / (HV / HK);
                double qq = 0, kk = 0;
                std::vector<double> qn(DK), kn(DK);
                for (unsigned d = 0; d < DK; d++) {
                    qn[d] = (double)b2f(x[qh * DK + d]);
                    kn[d] = (double)b2f(x[HK * DK + qh * DK + d]);
                    qq += qn[d] * qn[d]; kk += kn[d] * kn[d];
                }
                const double qs = (double)a.scale / sqrt(qq + (double)a.eps);
                const double ks = 1.0 / sqrt(kk + (double)a.eps);
                for (unsigned d = 0; d < DK; d++) { qn[d] *= qs; kn[d] *= ks; }
                const double dt = (double)b2f(av[(size_t)s * HV + hd_]) + (double)b2f(dtb[hd_]);
                const double decay = exp(-exp((double)b2f(alog[hd_])) * softplus(dt));
                const double beta = rbeta(bv[(size_t)s * HV + hd_]);
                /* [slot][head][v][k], V-FIRST — spelled out, not mirrored from the kernel. */
                double* Sh = Sref.data() + ((size_t)s * HV + hd_) * DV * DK;
                for (unsigned vc = 0; vc < DV; vc++) {
                    double* col = Sh + (size_t)vc * DK;
                    double proj = 0;
                    for (unsigned d = 0; d < DK; d++) { col[d] *= decay; proj += col[d] * kn[d]; }
                    const double delta =
                        ((double)b2f(x[2 * HK * DK + hd_ * DV + vc]) - proj) * beta;
                    double res = 0;
                    for (unsigned d = 0; d < DK; d++) {
                        col[d] += delta * kn[d];
                        res += col[d] * qn[d];
                    }
                    oref[((size_t)s * HV + hd_) * DV + vc] = res;
                }
            }
        }
        a.qkv = UP(qkv, bf16); a.a = UP(av, bf16); a.b = UP(bv, bf16);
        a.a_log = UP(alog, bf16); a.dt_bias = UP(dtb, bf16); a.state = UP(S0, float);
        a.out = OUT((size_t)BATCH * HV * DV, bf16);
        a.hk = HK; a.hv = HV; a.kdim = DK; a.vdim = DV; a.batch = BATCH; a.alog_f32 = 0;
        sync(f_step);
        std::vector<bf16> got((size_t)BATCH * HV * DV);
        std::vector<float> Sg(S0.size());
        HC(hipMemcpy(got.data(), (void*)(size_t)a.out, got.size() * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(Sg.data(), (void*)(size_t)a.state, Sg.size() * 4, hipMemcpyDeviceToHost));
        score("gdn_step out", cmp_bf16(got, oref), 6e-3, 2e-2);
        score("gdn_step state", cmp_f32(Sg, Sref), 1e-5, 1e-4);
    }

    /* ---- op 138: gated RMSNorm ---- */
    {
        const unsigned N = (size_t)BATCH * HV * DV;
        std::vector<bf16> core(N), z(N), g(DV);
        for (auto& x : core) x = f2b((float)(rnd() * 2.0));
        for (auto& x : z) x = f2b((float)(rnd() * 4.0));
        for (auto& x : g) x = f2b((float)(0.5 + rnd() + 0.5));
        std::vector<double> ref(N, 0.0);
        for (unsigned row = 0; row < BATCH * HV; row++) {
            if (!active[row / HV]) continue;
            const size_t off = (size_t)row * DV;
            double ss = 0;
            for (unsigned d = 0; d < DV; d++) ss += (double)b2f(core[off + d]) * (double)b2f(core[off + d]);
            const double inv = 1.0 / sqrt(ss / DV + (double)a.eps);
            for (unsigned d = 0; d < DV; d++) {
                const double gate = (double)b2f(z[off + d]);
                ref[off + d] = (double)b2f(core[off + d]) * inv * (double)b2f(g[d]) * gate * sigm(gate);
            }
        }
        a.core = UP(core, bf16); a.z = UP(z, bf16); a.gamma = UP(g, bf16);
        a.out = OUT(N, bf16); a.heads = HV; a.dim = DV; a.batch = BATCH; a.eps = 1e-6f;
        sync(f_gnorm);
        std::vector<bf16> got(N);
        HC(hipMemcpy(got.data(), (void*)(size_t)a.out, got.size() * 2, hipMemcpyDeviceToHost));
        score("gated_norm", cmp_bf16(got, ref), 4e-3, 8e-3);
    }

    /* ---- op 139: q / gate split. Pure gather: must be EXACT. ---- */
    {
        const unsigned cols = HV * DV;
        std::vector<bf16> packed((size_t)BATCH * HV * 2 * DV);
        for (auto& x : packed) x = f2b((float)(rnd() * 2.0));
        std::vector<double> qref((size_t)BATCH * cols, 0.0), gref((size_t)BATCH * cols, 0.0);
        for (unsigned i = 0; i < BATCH * cols; i++) {
            if (!active[i / cols]) continue;
            const size_t src = (size_t)(i / DV) * 2 * DV + i % DV;
            qref[i] = (double)b2f(packed[src]);
            gref[i] = (double)b2f(packed[src + DV]);
        }
        a.packed = UP(packed, bf16);
        a.q = OUT((size_t)BATCH * cols, bf16);
        a.gate = OUT((size_t)BATCH * cols, bf16);
        a.heads = HV; a.dim = DV; a.batch = BATCH;
        sync(f_split);
        std::vector<bf16> gq((size_t)BATCH * cols), gg((size_t)BATCH * cols);
        HC(hipMemcpy(gq.data(), (void*)(size_t)a.q, gq.size() * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(gg.data(), (void*)(size_t)a.gate, gg.size() * 2, hipMemcpyDeviceToHost));
        score("q_gate_split q", cmp_bf16(gq, qref), 1e-12, 1e-12);
        score("q_gate_split gate", cmp_bf16(gg, gref), 1e-12, 1e-12);
    }

    /* ---- op 140: sigmoid gate ---- */
    {
        const unsigned cols = HV * DV;
        std::vector<bf16> in((size_t)BATCH * cols), gate((size_t)BATCH * cols);
        for (auto& x : in) x = f2b((float)(rnd() * 2.0));
        for (auto& x : gate) x = f2b((float)(rnd() * 4.0));
        std::vector<double> ref((size_t)BATCH * cols, 0.0);
        for (unsigned i = 0; i < BATCH * cols; i++)
            if (active[i / cols]) ref[i] = (double)b2f(in[i]) * rbeta(gate[i]);
        a.in = UP(in, bf16); a.gate = UP(gate, bf16);
        a.out = OUT((size_t)BATCH * cols, bf16); a.cols = cols; a.batch = BATCH;
        sync(f_sgate);
        std::vector<bf16> got((size_t)BATCH * cols);
        HC(hipMemcpy(got.data(), (void*)(size_t)a.out, got.size() * 2, hipMemcpyDeviceToHost));
        score("sigmoid_gate", cmp_bf16(got, ref), 4e-3, 8e-3);
    }

    /* ---- op 141: RMSNorm with a (gamma + offset) weight ---- */
    {
        const unsigned DIM = 2048;
        std::vector<bf16> in((size_t)BATCH * DIM), g(DIM);
        for (auto& x : in) x = f2b((float)(rnd() * 2.0));
        for (auto& x : g) x = f2b((float)(rnd() * 0.5));
        const double off = 1.0;
        std::vector<double> ref((size_t)BATCH * DIM, 0.0);
        for (unsigned r = 0; r < BATCH; r++) {
            if (!active[r]) continue;
            const size_t base = (size_t)r * DIM;
            double ss = 0;
            for (unsigned d = 0; d < DIM; d++) ss += (double)b2f(in[base + d]) * (double)b2f(in[base + d]);
            const double inv = 1.0 / sqrt(ss / DIM + (double)a.eps);
            for (unsigned d = 0; d < DIM; d++)
                ref[base + d] = (double)b2f(in[base + d]) * inv * ((double)b2f(g[d]) + off);
        }
        a.in = UP(in, bf16); a.gamma = UP(g, bf16);
        a.out = OUT((size_t)BATCH * DIM, bf16);
        a.dim = DIM; a.batch = BATCH; a.offset = (float)off;
        sync(f_rms);
        std::vector<bf16> got((size_t)BATCH * DIM);
        HC(hipMemcpy(got.data(), (void*)(size_t)a.out, got.size() * 2, hipMemcpyDeviceToHost));
        score("qwen_rmsnorm", cmp_bf16(got, ref), 4e-3, 8e-3);
    }

    /* ---- op 142: per-head RMSNorm + partial RoPE, scattered into a KV ring ----
     * The cross-lane pair exchange is the one place the 64-lane wave changed the algorithm, so
     * this case runs BOTH the ring-scatter (context != 0) and the flat (context == 0) form. */
    for (unsigned ring = 0; ring < 2; ring++) {
        const unsigned HEADS = 8, CTX = ring ? 64u : 0u;
        const unsigned N = ring ? (size_t)BATCH * HEADS * CTX * HD : (size_t)BATCH * HEADS * HD;
        std::vector<bf16> in((size_t)BATCH * HEADS * HD), g(HD);
        /* The table is indexed by POSITION, not by ring depth, so it must cover max(pos) even when
         * the flat form writes no ring. */
        std::vector<float> cs((size_t)64 * (ROT / 2)), sn(cs.size());
        std::vector<int> pos{5, 3, 17};
        for (auto& x : in) x = f2b((float)(rnd() * 2.0));
        for (auto& x : g) x = f2b((float)(rnd() * 0.5));
        for (size_t i = 0; i < cs.size(); i++) {
            const double th = (double)i * 0.017 + 0.3;
            cs[i] = (float)cos(th); sn[i] = (float)sin(th);
        }
        const double off = 1.0;
        std::vector<double> ref(N, 0.0);
        for (unsigned row = 0; row < BATCH * HEADS; row++) {
            const unsigned slot = row / HEADS, head = row % HEADS;
            if (!active[slot]) continue;
            const int pp = pos[slot];
            std::vector<double> x(HD);
            double ss = 0;
            for (unsigned d = 0; d < HD; d++) { x[d] = (double)b2f(in[(size_t)row * HD + d]); ss += x[d] * x[d]; }
            const double inv = 1.0 / sqrt(ss / HD + (double)a.eps);
            for (unsigned d = 0; d < HD; d++)
                x[d] = (double)b2f(f2b((float)(x[d] * inv * ((double)b2f(g[d]) + off))));
            const unsigned half = ROT / 2;
            for (unsigned d = 0; d < half; d++) {
                const double c = (double)b2f(f2b(cs[(size_t)pp * half + d]));
                const double s = (double)b2f(f2b(sn[(size_t)pp * half + d]));
                const double lo = x[d], hi = x[d + half];
                x[d] = lo * c - hi * s;
                x[d + half] = hi * c + lo * s;
            }
            const size_t dest = ring ? ((size_t)slot * HEADS + head) * CTX * HD + (size_t)pp * HD
                                     : (size_t)row * HD;
            for (unsigned d = 0; d < HD; d++) ref[dest + d] = x[d];
        }
        a.in = UP(in, bf16); a.gamma = UP(g, bf16);
        a.cos = UP(cs, float); a.sin = UP(sn, float); a.positions = UP(pos, int);
        a.out = OUT(N, bf16);
        a.heads = HEADS; a.dim = HD; a.rotary = ROT; a.batch = BATCH; a.context = CTX;
        a.normalize = 1; a.offset = (float)off; a.prefill = 0;
        sync(f_hnr);
        std::vector<bf16> got(N);
        HC(hipMemcpy(got.data(), (void*)(size_t)a.out, got.size() * 2, hipMemcpyDeviceToHost));
        score(ring ? "headnorm_rope ring" : "headnorm_rope flat", cmp_bf16(got, ref), 4e-3, 8e-3);
    }
    a.rotary = 0; a.context = 0;

    /* ---- op 143: chunked conv over a prefill window ---- */
    {
        std::vector<bf16> in((size_t)TOKENS * CHANNELS), w((size_t)CHANNELS * WIDTH);
        std::vector<bf16> hist((size_t)CHANNELS * (WIDTH - 1));
        for (auto& x : in) x = f2b((float)(rnd() * 2.0));
        for (auto& x : w) x = f2b((float)(rnd() * 1.0));
        for (auto& x : hist) x = f2b((float)(rnd() * 2.0));
        std::vector<double> oref((size_t)TOKENS * CHANNELS, 0.0), href(hist.size(), 0.0);
        for (unsigned c = 0; c < CHANNELS; c++) {
            for (unsigned t = 0; t < TOKENS; t++) {
                double sum = 0;
                for (unsigned j = 0; j < WIDTH; j++) {
                    const int src = (int)t + (int)j - (int)(WIDTH - 1);
                    const double x = src < 0 ? (double)b2f(hist[(size_t)c * (WIDTH - 1) + t + j])
                                             : (double)b2f(in[(size_t)src * CHANNELS + c]);
                    sum += x * (double)b2f(w[(size_t)c * WIDTH + j]);
                }
                oref[(size_t)t * CHANNELS + c] = sum * sigm(sum);
            }
            for (unsigned j = 0; j + 1 < WIDTH; j++) {
                const int src = (int)TOKENS + (int)j - (int)(WIDTH - 1);
                href[(size_t)c * (WIDTH - 1) + j] =
                    src < 0 ? (double)b2f(hist[(size_t)c * (WIDTH - 1) + TOKENS + j])
                            : (double)b2f(in[(size_t)src * CHANNELS + c]);
            }
        }
        a.in = UP(in, bf16); a.weight = UP(w, bf16); a.history = UP(hist, bf16);
        a.out = OUT((size_t)TOKENS * CHANNELS, bf16);
        a.channels = CHANNELS; a.width = WIDTH; a.tokens = TOKENS;
        sync(f_convpf);
        std::vector<bf16> got((size_t)TOKENS * CHANNELS), hgot(hist.size());
        HC(hipMemcpy(got.data(), (void*)(size_t)a.out, got.size() * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(hgot.data(), (void*)(size_t)a.history, hgot.size() * 2, hipMemcpyDeviceToHost));
        score("gdn_conv_prefill out", cmp_bf16(got, oref), 4e-3, 4e-3);
        score("gdn_conv_prefill hist", cmp_bf16(hgot, href), 1e-12, 1e-12);
    }

    /* ---- op 144: prefill qkv prep ---- */
    {
        std::vector<bf16> packed((size_t)TOKENS * CHANNELS);
        for (auto& x : packed) x = f2b((float)(rnd() * 2.0));
        std::vector<double> qref((size_t)TOKENS * HK * DK), kref(qref.size());
        std::vector<double> vref((size_t)TOKENS * HV * DV);
        for (unsigned t = 0; t < TOKENS; t++)
            for (unsigned h = 0; h < HK; h++) {
                double qq = 0, kk = 0;
                std::vector<double> qv(DK), kv(DK);
                for (unsigned d = 0; d < DK; d++) {
                    qv[d] = (double)b2f(packed[(size_t)t * CHANNELS + h * DK + d]);
                    kv[d] = (double)b2f(packed[(size_t)t * CHANNELS + HK * DK + h * DK + d]);
                    qq += qv[d] * qv[d]; kk += kv[d] * kv[d];
                }
                const double qi = 1.0 / sqrt(qq + (double)a.eps), ki = 1.0 / sqrt(kk + (double)a.eps);
                for (unsigned d = 0; d < DK; d++) {
                    qref[((size_t)t * HK + h) * DK + d] = qv[d] * qi;
                    kref[((size_t)t * HK + h) * DK + d] = kv[d] * ki;
                }
            }
        for (size_t i = 0; i < vref.size(); i++)
            vref[i] = (double)b2f(packed[(size_t)(i / (HV * DV)) * CHANNELS + 2 * HK * DK + i % (HV * DV)]);
        a.packed = UP(packed, bf16);
        a.q = OUT(qref.size(), bf16); a.k = OUT(kref.size(), bf16); a.v = OUT(vref.size(), bf16);
        a.hk = HK; a.hv = HV; a.kdim = DK; a.vdim = DV; a.tokens = TOKENS;
        sync(f_prep);
        std::vector<bf16> gq(qref.size()), gk(kref.size()), gv(vref.size());
        HC(hipMemcpy(gq.data(), (void*)(size_t)a.q, gq.size() * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(gk.data(), (void*)(size_t)a.k, gk.size() * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(gv.data(), (void*)(size_t)a.v, gv.size() * 2, hipMemcpyDeviceToHost));
        score("qkv_prep q", cmp_bf16(gq, qref), 4e-3, 8e-3);
        score("qkv_prep k", cmp_bf16(gk, kref), 4e-3, 8e-3);
        score("qkv_prep v", cmp_bf16(gv, vref), 1e-12, 1e-12); /* pure gather */
    }

    /* ---- op 145: the prefill gate pair. f32 out, so the bar is the transcendental's, not bf16's.
     * `alpha` here must equal the `decay` op 137 computes inline for the same inputs. ---- */
    {
        std::vector<bf16> av((size_t)TOKENS * HV), bv((size_t)TOKENS * HV), alog(HV), dtb(HV);
        for (auto& x : av) x = f2b((float)(rnd() * 4.0));
        for (auto& x : bv) x = f2b((float)(rnd() * 4.0));
        for (auto& x : alog) x = f2b(logf(0.5f + (float)(rnd() + 0.5) * 11.5f));
        for (auto& x : dtb) x = f2b((float)(-7.0 + (rnd() + 0.5) * 6.0));
        std::vector<double> aref((size_t)TOKENS * HV), bref(aref.size());
        for (unsigned i = 0; i < TOKENS * HV; i++) {
            const unsigned h = i % HV;
            const double dt = (double)b2f(av[i]) + (double)b2f(dtb[h]);
            aref[i] = exp(-exp((double)b2f(alog[h])) * softplus(dt));
            bref[i] = rbeta(bv[i]);
        }
        a.a = UP(av, bf16); a.b = UP(bv, bf16); a.a_log = UP(alog, bf16); a.dt_bias = UP(dtb, bf16);
        a.alpha = OUT(aref.size(), float); a.beta = OUT(bref.size(), float);
        a.hv = HV; a.tokens = TOKENS;
        sync(f_gp);
        std::vector<float> ga(aref.size()), gb(bref.size());
        HC(hipMemcpy(ga.data(), (void*)(size_t)a.alpha, ga.size() * 4, hipMemcpyDeviceToHost));
        HC(hipMemcpy(gb.data(), (void*)(size_t)a.beta, gb.size() * 4, hipMemcpyDeviceToHost));
        score("gate_prep alpha", cmp_f32(ga, aref), 1e-6, 1e-5);
        score("gate_prep beta", cmp_f32(gb, bref), 1e-12, 1e-12); /* bf16-rounded: exact */
    }

    printf("%s (worst %.3f)\n", worst < 1.0 ? "PASS" : "FAIL", worst);
    return worst >= 1.0;
}
#endif

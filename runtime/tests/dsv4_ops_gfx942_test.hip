/* The three small DeepSeek-V4 arms, against host references.  [DSV4-ROUTE] [DSV4-MHC] [DSV4-IROPE]
 *
 *  1. d_moe_router_topk's `sqrtsoftplus` / f32-logit / hash arms (flags bits 2, 3, 4) --
 *     model.py:569-589. Checked against a host router that reimplements the whole selection
 *     (transform, bias-on-selection-only, lowest-id tie-break, norm_topk, route_scale)
 *     independently, plus a REGRESSION case asserting the shipped sigmoid arm is untouched.
 *  2. d_hyperconn_pre's `head_only` arm -- the learned gated 4 -> 1 tower exit, model.py:709-717.
 *  3. d_rope_inverse_o -- the conjugate rotation on the attention output's last 64 dims,
 *     model.py:539.
 *
 * In nix develop:
 *   $PLOW_HIPCC --offload-arch=gfx942 -O3 -w -std=c++17 -Iruntime/amd -Iruntime/common \
 *       -c runtime/tests/dsv4_ops_gfx942_test.hip -o /tmp/dsv4.o
 *   c++ /tmp/dsv4.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/dsv4
 *   perf-data/tools/gpulease -n 1 v4-ops /tmp/dsv4
 */
#define PLOW_WG_WAVES 4
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_compress.h"
#include "op_elementwise.h"
#include "op_hyperconn.h"
#include "op_moe.h"
#else
using bf16 = unsigned short;
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

__global__ __launch_bounds__(256, 1) void k_router(unsigned char* table, const bf16* logit,
                                                   const float* bias, unsigned n_exp, unsigned k,
                                                   unsigned flags, float rs,
                                                   const int32_t* tid2eid, int token_id) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float lds[4096];
    d_moe_router_topk(table, logit, bias, n_exp, k, flags, rs, 0, 1, lds, 0, 0, tid2eid,
                      token_id);
#endif
}

__global__ __launch_bounds__(256, 1) void k_hcpre(float* post_mix, float* comb_mix,
                                                  bf16* layer_input, const float* mixes,
                                                  const bf16* residual, const float* hc_scale,
                                                  const float* hc_base, unsigned T, unsigned n,
                                                  unsigned hidden, float rms_eps, float hc_eps,
                                                  unsigned head_only) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float part[4];
    d_hyperconn_pre(post_mix, comb_mix, layer_input, mixes, residual, hc_scale, hc_base, T, n,
                    hidden, 20, rms_eps, hc_eps, blockIdx.x, gridDim.x, part, head_only != 0);
#endif
}

__global__ __launch_bounds__(256, 1) void k_irope(bf16* o, const float* cosb, const float* sinb,
                                                  unsigned n_tok, unsigned n_head, unsigned D,
                                                  unsigned rd, unsigned pos0) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_rope_inverse_o(o, cosb, sinb, n_tok, n_head, D, rd, pos0, blockIdx.x, gridDim.x);
#endif
}

/* act code 4: the clamped SwiGLU, on BOTH the dense (op_elementwise) and routed
 * (op_moe) entry points. Poisoned codes are checked too: a gate-only caller handed 4 must
 * produce NaN, not a plausible unclamped answer. */
__global__ __launch_bounds__(256, 1) void k_glu(bf16* out, const bf16* g, const bf16* u,
                                                unsigned n, unsigned act, float limit) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_glu(out, g, u, n, act, blockIdx.x, gridDim.x, limit);
#endif
}

__global__ __launch_bounds__(256, 1) void k_moeglu(float* out, const bf16* g, const bf16* u,
                                                   unsigned n, unsigned act, float limit) {
#if defined(__HIP_DEVICE_COMPILE__)
    for (unsigned i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
        out[i] = moe_glu(bf2f(g[i]), bf2f(u[i]), act, 0.0f, limit);
    /* The gate-only entry point must POISON act 4 rather than silently leave `up` unclamped. */
    if (blockIdx.x == 0 && threadIdx.x == 0) out[n] = moe_act(1.0f, act);
#endif
}

/* ------------------------------------------------------------------ host side */
static int fails = 0;
static unsigned rs_ = 0x2468ACEu;
static float rnd() {
    rs_ ^= rs_ << 13;
    rs_ ^= rs_ >> 17;
    rs_ ^= rs_ << 5;
    return (float)(int)(rs_ >> 8) / 8388608.0f - 1.0f;
}
static bf16 f2b(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static float b2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static void say(const char* what, bool ok, const char* detail) {
    std::printf("  %-50s %s  %s\n", what, ok ? "ok " : "FAIL", detail);
    if (!ok) fails++;
}

/* ---- 1. router ---- */
enum { SIG = 1, NORM = 2, SQSP = 4, F32L = 8, HASH = 16 };

static void router_case(const char* name, unsigned n_exp, unsigned k, unsigned flags, float rscale,
                        bool with_bias) {
    std::vector<float> lf(n_exp);
    std::vector<bf16> lb(n_exp);
    std::vector<float> bias(n_exp);
    for (unsigned e = 0; e < n_exp; e++) {
        lf[e] = rnd() * 4.0f;
        lb[e] = f2b(lf[e]);
        if (!(flags & F32L)) lf[e] = b2f(lb[e]); /* the bf16 arm's logit IS the rounded value */
        bias[e] = with_bias ? rnd() * 0.5f : 0.0f;
    }
    const int tok = 7;
    std::vector<int32_t> lut(64 * k);
    for (auto& v : lut) v = (int32_t)((unsigned)(rnd() * 1e6f) % n_exp);
    /* Distinct expert ids for this token's row: a repeated id is not a thing the checkpoint's
     * tid2eid can produce for a top-k route, and it would make the norm ambiguous. */
    for (unsigned j = 0; j < k; j++)
        for (unsigned i = 0; i < j; i++)
            if (lut[(size_t)tok * k + i] == lut[(size_t)tok * k + j])
                lut[(size_t)tok * k + j] = (lut[(size_t)tok * k + j] + 1) % (int)n_exp;

    std::vector<unsigned char> table((size_t)k * 8, 0);
    auto* dt = up(table);
    auto* dbias = with_bias ? up(bias) : (float*)nullptr;
    auto* dlut = up(lut);
    const bf16* dlog =
        (flags & F32L) ? (const bf16*)up(lf) : (const bf16*)up(lb);
    hipLaunchKernelGGL(k_router, dim3(1), dim3(256), 0, 0, dt, dlog, dbias, n_exp, k, flags,
                       rscale, dlut, tok);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    ck(hipMemcpy(table.data(), dt, table.size(), hipMemcpyDeviceToHost));

    /* ---- host reference, written from model.py:569-589 ---- */
    std::vector<double> sc(n_exp);
    for (unsigned e = 0; e < n_exp; e++) {
        const double l = lf[e];
        if (flags & SQSP)
            sc[e] = std::sqrt(std::log1p(std::exp(-std::fabs(l))) + std::fmax(l, 0.0));
        else if (flags & SIG)
            sc[e] = 1.0 / (1.0 + std::exp(-l));
        else
            sc[e] = l; /* softmax filled below */
    }
    if (!(flags & (SQSP | SIG))) {
        double m = -1e300, s = 0;
        for (auto v : sc) m = std::fmax(m, v);
        for (auto& v : sc) { v = std::exp(v - m); s += v; }
        for (auto& v : sc) v /= s;
    }
    std::vector<unsigned> want(k);
    if (flags & HASH) {
        for (unsigned j = 0; j < k; j++) want[j] = (unsigned)lut[(size_t)tok * k + j];
    } else {
        std::vector<double> key(n_exp);
        std::vector<char> used(n_exp, 0);
        for (unsigned e = 0; e < n_exp; e++) key[e] = sc[e] + bias[e];
        for (unsigned j = 0; j < k; j++) {
            int best = -1;
            for (unsigned e = 0; e < n_exp; e++)
                if (!used[e] && (best < 0 || key[e] > key[best])) best = (int)e; /* lowest id ties */
            used[best] = 1;
            want[j] = (unsigned)best;
        }
    }
    std::vector<double> g(k);
    double sum = 0;
    for (unsigned j = 0; j < k; j++) { g[j] = sc[want[j]]; sum += g[j]; }
    for (unsigned j = 0; j < k; j++) {
        if ((flags & NORM) && sum != 0) g[j] /= sum;
        g[j] *= rscale;
    }

    bool ok = true;
    char det[192];
    double worst = 0;
    for (unsigned j = 0; j < k; j++) {
        unsigned gid;
        float gg;
        std::memcpy(&gid, table.data() + (size_t)j * 8, 4);
        std::memcpy(&gg, table.data() + (size_t)j * 8 + 4, 4);
        if (gid != want[j]) ok = false;
        const double rel = std::fabs((double)gg - g[j]) / std::fmax(std::fabs(g[j]), 1e-9);
        worst = std::fmax(worst, rel);
        if (rel > 3e-6) ok = false;
    }
    std::snprintf(det, sizeof det, "(k=%u ids match=%d, worst gate rel %.2e)", k, (int)ok, worst);
    say(name, ok, det);
    hipFree(dt);
    hipFree(dlut);
    if (dbias) hipFree(dbias);
    hipFree((void*)dlog);
}

/* ---- 2. mHC head contraction ---- */
static void hc_head_case(unsigned T, unsigned n, unsigned hidden) {
    const float rms_eps = 1e-6f, hc_eps = 1e-6f;
    std::vector<float> mixes((size_t)T * n), scale(3), base(2 * n + n * n);
    std::vector<bf16> res((size_t)T * n * hidden), out((size_t)T * hidden, 0);
    for (auto& v : mixes) v = rnd() * 20.0f;
    for (auto& v : res) v = f2b(rnd());
    for (auto& v : base) v = rnd();
    scale[0] = 0.7f;
    scale[1] = 0.3f;
    scale[2] = 0.5f;
    std::vector<float> pm((size_t)T * n, -1.0f), cm((size_t)T * n * n, -1.0f);
    auto *dm = up(mixes), *dsc = up(scale), *dbs = up(base), *dpm = up(pm), *dcm = up(cm);
    auto* dr = up(res);
    auto* dout = up(out);
    hipLaunchKernelGGL(k_hcpre, dim3(8), dim3(256), 0, 0, dpm, dcm, dout, dm, dr, dsc, dbs, T, n,
                       hidden, rms_eps, hc_eps, 1u);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    ck(hipMemcpy(out.data(), dout, out.size() * sizeof(bf16), hipMemcpyDeviceToHost));
    ck(hipMemcpy(pm.data(), dpm, pm.size() * sizeof(float), hipMemcpyDeviceToHost));

    double worst = 0;
    for (unsigned t = 0; t < T; t++) {
        double ss = 0;
        for (unsigned i = 0; i < n * hidden; i++) {
            const double x = b2f(res[(size_t)t * n * hidden + i]);
            ss += x * x;
        }
        const double inv = 1.0 / std::sqrt(ss / (double)(n * hidden) + rms_eps);
        double pre[8];
        for (unsigned j = 0; j < n; j++)
            pre[j] = 1.0 / (1.0 + std::exp(-(mixes[(size_t)t * n + j] * inv * scale[0] + base[j]))) +
                     hc_eps;
        for (unsigned d = 0; d < hidden; d++) {
            double acc = 0;
            for (unsigned i = 0; i < n; i++)
                acc += pre[i] * b2f(res[((size_t)t * n + i) * hidden + d]);
            const double got = b2f(out[(size_t)t * hidden + d]);
            worst = std::fmax(worst, std::fabs(got - acc) / std::fmax(std::fabs(acc), 1e-4));
        }
    }
    /* head_only must leave post_mix ALONE: writing it would read mixes[n..3n+n^2), off the end
     * of a [n, n*hidden] head projection's row. */
    unsigned touched = 0;
    for (auto v : pm)
        if (v != -1.0f) touched++;
    char det[160];
    std::snprintf(det, sizeof det, "(T=%u worst rel %.2e, post_mix untouched=%d)", T, worst,
                  (int)(touched == 0));
    say("hc_head learned gated 4->1 (head_only)", worst < 8e-3 && touched == 0, det);
    hipFree(dm); hipFree(dsc); hipFree(dbs); hipFree(dpm); hipFree(dcm);
    hipFree(dr); hipFree(dout);
}

/* ---- 3. inverse RoPE ---- */
static void irope_case(unsigned n_tok, unsigned n_head, unsigned D, unsigned rd, unsigned pos0) {
    const unsigned h2 = rd / 2, npos = 2048;
    std::vector<bf16> o((size_t)n_tok * n_head * D);
    for (auto& v : o) v = f2b(rnd() * 2.0f);
    const std::vector<bf16> o0 = o;
    std::vector<float> cosb((size_t)npos * h2), sinb((size_t)npos * h2);
    for (unsigned p = 0; p < npos; p++)
        for (unsigned i = 0; i < h2; i++) {
            const double th = (double)p / std::pow(160000.0, 2.0 * i / (double)rd);
            cosb[(size_t)p * h2 + i] = (float)std::cos(th);
            sinb[(size_t)p * h2 + i] = (float)std::sin(th);
        }
    auto* dO = up(o);
    auto *dc = up(cosb), *ds = up(sinb);
    hipLaunchKernelGGL(k_irope, dim3(16), dim3(256), 0, 0, dO, dc, ds, n_tok, n_head, D, rd,
                       pos0);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    ck(hipMemcpy(o.data(), dO, o.size() * sizeof(bf16), hipMemcpyDeviceToHost));

    double worst = 0;
    unsigned moved = 0;
    for (unsigned t = 0; t < n_tok; t++)
        for (unsigned h = 0; h < n_head; h++)
            for (unsigned d = 0; d < D; d++) {
                const size_t idx = (((size_t)t * n_head + h) * D) + d;
                double want;
                if (d < D - rd) {
                    want = b2f(o0[idx]); /* the nope half must be untouched */
                } else {
                    const unsigned m = (d - (D - rd)) / 2;
                    const size_t p = (size_t)(pos0 + t) * h2 + m;
                    const double c = cosb[p], sn = -sinb[p]; /* conjugate */
                    const double x0 = b2f(o0[(((size_t)t * n_head + h) * D) + D - rd + 2 * m]);
                    const double x1 = b2f(o0[(((size_t)t * n_head + h) * D) + D - rd + 2 * m + 1]);
                    want = ((d - (D - rd)) & 1u) ? (x0 * sn + x1 * c) : (x0 * c - x1 * sn);
                    if (o[idx] != o0[idx]) moved++;
                }
                worst = std::fmax(worst, std::fabs((double)b2f(o[idx]) - want) /
                                             std::fmax(std::fabs(want), 1e-3));
            }
    char det[160];
    std::snprintf(det, sizeof det, "(t%u h%u D%u rd%u pos0=%u worst rel %.2e, %u roped moved)",
                  n_tok, n_head, D, rd, pos0, worst, moved);
    say("inverse RoPE on O's last rd dims", worst < 8e-3 && moved > 0, det);
    hipFree(dO); hipFree(dc); hipFree(ds);
}

static void act_case() {
    const unsigned n = 4096;
    const float L = 10.0f;
    std::vector<bf16> g(n), u(n), o(n, 0);
    /* Straddle the clamp on BOTH sides and on both branches: without values past +-L the case
     * cannot tell act 4 from a plain SwiGLU. */
    for (unsigned i = 0; i < n; i++) {
        g[i] = f2b(rnd() * 30.0f);
        u[i] = f2b(rnd() * 30.0f);
    }
    std::vector<float> mo(n + 1, 0);
    auto *dg = up(g);
    auto* du = up(u);
    auto* dout = up(o);
    auto* dmo = up(mo);
    hipLaunchKernelGGL(k_glu, dim3(8), dim3(256), 0, 0, dout, dg, du, n, 4u, L);
    ck(hipGetLastError());
    hipLaunchKernelGGL(k_moeglu, dim3(8), dim3(256), 0, 0, dmo, dg, du, n, 4u, L);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    ck(hipMemcpy(o.data(), dout, o.size() * sizeof(bf16), hipMemcpyDeviceToHost));
    ck(hipMemcpy(mo.data(), dmo, mo.size() * sizeof(float), hipMemcpyDeviceToHost));

    double wd = 0, wm = 0;
    unsigned clamped = 0;
    for (unsigned i = 0; i < n; i++) {
        const double gv = b2f(g[i]), uv = b2f(u[i]);
        if (gv > L || uv > L || uv < -L) clamped++;
        const double gc = std::fmin(gv, (double)L);
        const double want = (gc / (1.0 + std::exp(-gc))) * std::fmin(std::fmax(uv, -(double)L), (double)L);
        const double sc = std::fmax(std::fabs(want), 1e-3);
        wd = std::fmax(wd, std::fabs((double)b2f(o[i]) - want) / sc);
        wm = std::fmax(wm, std::fabs((double)mo[i] - want) / sc);
    }
    const bool poisoned = std::isnan(mo[n]);
    char det[192];
    std::snprintf(det, sizeof det,
                  "(d_glu rel %.2e, moe_glu rel %.2e, %u/%u clamped, gate-only poisoned=%d)", wd,
                  wm, clamped, n, (int)poisoned);
    say("clamped SwiGLU, act code 4", wd < 5e-3 && wm < 1e-6 && clamped > n / 4 && poisoned, det);
    hipFree(dg); hipFree(du); hipFree(dout); hipFree(dmo);
}

int main() {
    std::printf("DeepSeek-V4 router / mHC-head / inverse-RoPE / clamped-SwiGLU arms\n");
    router_case("router SIGMOID+norm (SHIPPED, must not move)", 256, 8, SIG | NORM, 1.0f, true);
    router_case("router SOFTMAX (SHIPPED, must not move)", 128, 4, NORM, 1.0f, false);
    router_case("router sqrtsoftplus + bias + norm, scale 1.5", 256, 6, SQSP | NORM, 1.5f, true);
    router_case("router sqrtsoftplus, f32 logit", 256, 6, SQSP | F32L | NORM, 1.5f, true);
    router_case("router sqrtsoftplus, f32, no bias", 256, 6, SQSP | F32L | NORM, 1.5f, false);
    router_case("router HASH (tid2eid) + sqrtsoftplus f32", 256, 6,
                SQSP | F32L | NORM | HASH, 1.5f, true);
    router_case("router HASH, bias present (must be ignored)", 256, 6,
                SQSP | F32L | NORM | HASH, 1.5f, true);
    hc_head_case(5, 4, 4096);
    irope_case(3, 8, 512, 64, 0);
    irope_case(17, 8, 512, 64, 1000);
    act_case();
    std::printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

/* v4_moe_oracle_gfx942.hip — `d_v4_grouped_linear`, `d_v4_moe_route` and
 * `d_v4_clamped_swiglu` against HOST TRANSCRIPTIONS of `Attention.forward`'s
 * `wo_a` einsum, `Gate.forward` and `Expert.forward`.
 *
 * THREE OPS, THREE DISTINCT WAYS TO BE PLAUSIBLY WRONG, all controlled here:
 *
 *   wo_a    a dense [G*R, H*D] linear of the same element count mixes groups
 *           the reference keeps apart. NEGATIVE-CONTROLLED.
 *   router  the selection bias shifts WHICH experts run and must NOT reach the
 *           combine weights; scoring is sqrt(softplus(.)), not softmax or
 *           sigmoid, both of which already exist in this tree. BOTH
 *           NEGATIVE-CONTROLLED.
 *   swiglu  the clamp is ONE-SIDED on the gate and TWO-SIDED on the up branch,
 *           and binds only in the tail. NEGATIVE-CONTROLLED, with inputs that
 *           actually reach the limit — with |x| < limit the control passes and
 *           the gate would be blind.
 *
 * A hash layer is swept too: its expert set comes from tid2eid[token_id] and
 * its scores supply only the weights.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/v4_moe_oracle_gfx942.hip -o /tmp/v4moe && /tmp/v4moe
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amd_common.h"
#include "op_deepseek_v4.h"

#define HC_CHECK(x)                                                                               \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_) {                                                                                 \
            printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__);                              \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

enum { LDS_BYTES = 8192 };

__global__ __launch_bounds__(PLOW_THREADS) void k_wo(bf16* out, const bf16* o, const bf16* w,
                                                     unsigned T, unsigned G, unsigned R,
                                                     unsigned W) {
    d_v4_grouped_linear(out, o, w, T, G, R, W, blockIdx.x, gridDim.x);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_route(int* sel, float* wts, const float* logits,
                                                        const float* bias,
                                                        const long long* tid2eid, const int* ids,
                                                        unsigned T, unsigned E, unsigned K,
                                                        float rs) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_v4_moe_route(sel, wts, nullptr, logits, bias, tid2eid, ids, T, E, K, rs, blockIdx.x, gridDim.x,
                   (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_glu(bf16* out, const bf16* g, const bf16* u,
                                                      unsigned n, float lim) {
    d_v4_clamped_swiglu(out, g, u, n, lim);
}

static bf16 f2bf_host(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    const unsigned r = (u >> 16) & 1u;
    return (bf16)((u + 0x7fffu + r) >> 16);
}
static float bf2f_host(bf16 b) {
    const unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float frand(void) { return (float)drand48() * 2.0f - 1.0f; }

static int fail(const char* what, double worst, double tol) {
    printf("  %-28s max rel %.3e  %s\n", what, worst, worst <= tol ? "ok" : "FAIL");
    return worst <= tol ? 0 : 1;
}

/* ---- wo_a ---------------------------------------------------------------- */
static int test_wo(unsigned T, unsigned G, unsigned R, unsigned W) {
    std::vector<bf16> o((size_t)T * G * W), w((size_t)G * R * W);
    for (auto& v : o) v = f2bf_host(frand());
    const float s = 1.0f / sqrtf((float)W);
    for (auto& v : w) v = f2bf_host(frand() * s);
    bf16 *d_o, *d_w, *d_out;
    HC_CHECK(hipMalloc(&d_o, o.size() * 2));
    HC_CHECK(hipMalloc(&d_w, w.size() * 2));
    HC_CHECK(hipMalloc(&d_out, (size_t)T * G * R * 2));
    HC_CHECK(hipMemcpy(d_o, o.data(), o.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_w, w.data(), w.size() * 2, hipMemcpyHostToDevice));
    k_wo<<<T < 4 ? T : 4, PLOW_THREADS>>>(d_out, d_o, d_w, T, G, R, W);
    HC_CHECK(hipDeviceSynchronize());
    std::vector<bf16> got((size_t)T * G * R);
    HC_CHECK(hipMemcpy(got.data(), d_out, got.size() * 2, hipMemcpyDeviceToHost));

    std::vector<double> ref(got.size());
    double rms = 0.0;
    for (unsigned t = 0; t < T; t++)
        for (unsigned g = 0; g < G; g++)
            for (unsigned r = 0; r < R; r++) {
                double a = 0.0;
                for (unsigned k = 0; k < W; k++) /* group g reads ONLY its own slice */
                    a += bf2f_host(o[(size_t)t * G * W + (size_t)g * W + k]) *
                         bf2f_host(w[((size_t)g * R + r) * W + k]);
                ref[(size_t)t * G * R + g * R + r] = a;
                rms += a * a;
            }
    rms = sqrt(rms / (double)ref.size()) + 1e-9;
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); i++)
        worst = std::max(worst, fabs(ref[i] - bf2f_host(got[i])) / rms);
    hipFree(d_o);
    hipFree(d_w);
    hipFree(d_out);
    return fail("grouped_linear(wo_a)", worst, 2e-2);
}

/* ---- router -------------------------------------------------------------- */
static int test_route(unsigned T, unsigned E, unsigned K, bool hash) {
    const float route_scale = 1.5f, V = 64.0f;
    std::vector<float> logits((size_t)T * E), bias(E);
    std::vector<long long> tab((size_t)(int)V * K);
    std::vector<int> ids(T);
    for (auto& v : logits) v = frand() * 2.0f;
    for (auto& v : bias) v = frand();
    for (size_t i = 0; i < tab.size(); i++) tab[i] = (long long)(lrand48() % E);
    for (auto& v : ids) v = (int)(lrand48() % (long)V);

    float *d_lg, *d_b, *d_w;
    long long* d_tab;
    int *d_ids, *d_sel;
    HC_CHECK(hipMalloc(&d_lg, logits.size() * 4));
    HC_CHECK(hipMalloc(&d_b, bias.size() * 4));
    HC_CHECK(hipMalloc(&d_tab, tab.size() * 8));
    HC_CHECK(hipMalloc(&d_ids, ids.size() * 4));
    HC_CHECK(hipMalloc(&d_sel, (size_t)T * K * 4));
    HC_CHECK(hipMalloc(&d_w, (size_t)T * K * 4));
    HC_CHECK(hipMemcpy(d_lg, logits.data(), logits.size() * 4, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_b, bias.data(), bias.size() * 4, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_tab, tab.data(), tab.size() * 8, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_ids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
    k_route<<<T < 4 ? T : 4, PLOW_THREADS>>>(d_sel, d_w, d_lg, hash ? nullptr : d_b,
                                             hash ? d_tab : nullptr, d_ids, T, E, K, route_scale);
    HC_CHECK(hipDeviceSynchronize());
    std::vector<int> sel((size_t)T * K);
    std::vector<float> wts((size_t)T * K);
    HC_CHECK(hipMemcpy(sel.data(), d_sel, sel.size() * 4, hipMemcpyDeviceToHost));
    HC_CHECK(hipMemcpy(wts.data(), d_w, wts.size() * 4, hipMemcpyDeviceToHost));

    int bad = 0;
    double worst_w = 0.0;
    for (unsigned t = 0; t < T; t++) {
        std::vector<double> sc(E);
        for (unsigned e = 0; e < E; e++) sc[e] = sqrt(log1p(exp((double)logits[(size_t)t * E + e])));
        std::vector<int> want(K, -1);
        if (hash) {
            for (unsigned k = 0; k < K; k++) want[k] = (int)tab[(size_t)ids[t] * K + k];
        } else {
            std::vector<char> used(E, 0);
            for (unsigned k = 0; k < K; k++) {
                double best = -INFINITY;
                int bi = -1;
                for (unsigned e = 0; e < E; e++) {
                    if (used[e]) continue;
                    const double v = sc[e] + bias[e]; /* biased for SELECTION */
                    if (v > best) {
                        best = v;
                        bi = (int)e;
                    }
                }
                want[k] = bi;
                used[bi] = 1;
            }
        }
        double sum = 0.0;
        for (unsigned k = 0; k < K; k++) sum += sc[want[k]]; /* UNBIASED for the weight */
        for (unsigned k = 0; k < K; k++) {
            if (sel[(size_t)t * K + k] != want[k]) bad = 1;
            const double ref = sc[want[k]] / sum * (double)route_scale;
            worst_w = std::max(worst_w, fabs(ref - (double)wts[(size_t)t * K + k]) / (fabs(ref) + 1e-9));
        }
    }
    printf("  %-28s experts %s, weights max rel %.3e  %s\n", hash ? "moe_route(hash)" : "moe_route",
           bad ? "MISMATCH" : "exact", worst_w, (!bad && worst_w <= 1e-5) ? "ok" : "FAIL");
    hipFree(d_lg);
    hipFree(d_b);
    hipFree(d_tab);
    hipFree(d_ids);
    hipFree(d_sel);
    hipFree(d_w);
    return (!bad && worst_w <= 1e-5) ? 0 : 1;
}

/* ---- clamped SwiGLU ------------------------------------------------------ */
static int test_glu(unsigned n, float lim) {
    std::vector<bf16> g(n), u(n);
    /* Inputs MUST reach the limit or the clamp is untested and its control
     * would pass. +-3*limit puts most of the mass outside it. */
    for (unsigned i = 0; i < n; i++) {
        g[i] = f2bf_host(frand() * 3.0f * lim);
        u[i] = f2bf_host(frand() * 3.0f * lim);
    }
    bf16 *d_g, *d_u, *d_o;
    HC_CHECK(hipMalloc(&d_g, n * 2));
    HC_CHECK(hipMalloc(&d_u, n * 2));
    HC_CHECK(hipMalloc(&d_o, n * 2));
    HC_CHECK(hipMemcpy(d_g, g.data(), n * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_u, u.data(), n * 2, hipMemcpyHostToDevice));
    k_glu<<<8, PLOW_THREADS>>>(d_o, d_g, d_u, n, lim);
    HC_CHECK(hipDeviceSynchronize());
    std::vector<bf16> got(n);
    HC_CHECK(hipMemcpy(got.data(), d_o, n * 2, hipMemcpyDeviceToHost));

    std::vector<double> ref(n);
    double rms = 0.0;
    for (unsigned i = 0; i < n; i++) {
        double gg = bf2f_host(g[i]), uu = bf2f_host(u[i]);
        gg = std::min(gg, (double)lim);                            /* ONE-sided */
        uu = std::min(std::max(uu, -(double)lim), (double)lim);    /* TWO-sided */
        ref[i] = gg / (1.0 + exp(-gg)) * uu;
        rms += ref[i] * ref[i];
    }
    rms = sqrt(rms / (double)n) + 1e-9;
    double worst = 0.0;
    for (unsigned i = 0; i < n; i++)
        worst = std::max(worst, fabs(ref[i] - bf2f_host(got[i])) / rms);
    hipFree(d_g);
    hipFree(d_u);
    hipFree(d_o);
    return fail("clamped_swiglu", worst, 2e-2);
}

int main(void) {
    srand48(13);
    int bad = 0;
    /* V4: o_groups 8, o_lora_rank 1024, 64 heads x 512 / 8 = 4096 wide. */
    bad |= test_wo(4, 8, 64, 4096);
    /* 256 routed experts, top-6, route scale 1.5. */
    bad |= test_route(8, 256, 6, false);
    bad |= test_route(8, 256, 6, true);
    bad |= test_glu(1 << 16, 10.0f);
    printf("%s\n", bad ? "FAIL" : "PASS — wo_a / router / clamped SwiGLU match the reference");
    return bad;
}

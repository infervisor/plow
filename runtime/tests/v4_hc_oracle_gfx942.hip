/* v4_hc_oracle_gfx942.hip — V4 hyper-connections against a HOST TRANSCRIPTION
 * of the reference.
 *
 * WHY THIS EXISTS. `d_hc_reduce` / `d_hc_expand` replace the residual add on
 * EVERY sub-layer of a 43-layer model (86 sites), and a plain residual add is
 * the `HC == 1, pre = post = comb = 1` special case of them — same shapes, no
 * fault, fluent wrong output. There is no existing gate this could ride on,
 * because there is no other op in the tree with parallel residual streams.
 *
 * THE REFERENCE, transcribed from `inference/model.py` (`Block.hc_pre`,
 * `hc_post`, `hc_head`) and `inference/kernel.py` (`hc_split_sinkhorn_kernel`,
 * which is the authority for the normalization — the model file does not
 * contain it):
 *
 *     xf     = x.flatten(2).float()                 # [HC*D]
 *     rsqrt  = rsqrt(xf.square().mean(-1) + norm_eps)
 *     mixes  = F.linear(xf, hc_fn) * rsqrt          # [(2+HC)*HC]
 *     pre[j]  = sigmoid(mixes[j]    * scale[0] + base[j])    + hc_eps
 *     post[j] = sigmoid(mixes[j+HC] * scale[1] + base[j+HC]) * 2
 *     comb    = mixes[2HC:] * scale[2] + base[2HC:]          # [HC,HC]
 *     comb    = comb.softmax(-1) + hc_eps
 *     comb    = comb / (comb.sum(-2) + hc_eps)
 *     (iters-1)x:  comb = comb / (comb.sum(-1) + hc_eps)
 *                  comb = comb / (comb.sum(-2) + hc_eps)
 *     reduce  = sum_j pre[j] * x[j]                          # RAW x, not xf*rsqrt
 *     expand  = post[k] * branch + sum_j comb[j][k] * x[j]
 *
 * WHAT THIS PINS that a code read does not:
 *   * `post` carries a factor of 2 and `pre` does not. It is in the tilelang
 *     kernel only.
 *   * pass 0 of the Sinkhorn is a row SOFTMAX with `+eps` after it; every later
 *     pass is a plain sum-divide. `iters` symmetric passes would be close and
 *     wrong.
 *   * the reduce mixes the RAW streams — `rsqrt` scales the MIXES only.
 *   * `hc_head` is a DIFFERENT formula, not a special case: `[HC, HC*D]`
 *     weights, a scalar scale, no Sinkhorn, no `post`/`comb`.
 *
 * ALSO SWEPT, because each is its own axis:
 *   * T > 1 with several workgroups, which partition the tokens.
 *   * `out` ALIASING `residual` in the expand, which is how a layer uses it.
 *   * HC = 4 (every shipped checkpoint) and HC = 2 (that the fixed-size
 *     register vectors are not hard-coded to 4).
 *
 * TOLERANCE. Output is bf16 (~4e-3 relative floor); the host reduces in double
 * and the device in f32 across 8 waves, so the orders differ. The gate is max
 * elementwise difference relative to the row RMS, 2e-2 — the same TOL the K3
 * rung gates and `k3_attn_res_oracle_gfx950.hip` use.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/v4_hc_oracle_gfx942.hip -o /tmp/v4hc && /tmp/v4hc
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

enum { LDS_BYTES = 65536 };

__global__ __launch_bounds__(PLOW_THREADS) void k_reduce(bf16* out, float* mix_out, const bf16* x,
                                                         const float* fn, const float* scale,
                                                         const float* base, unsigned T, unsigned D,
                                                         unsigned HC, unsigned iters,
                                                         float norm_eps, float hc_eps) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_hc_reduce(out, mix_out, x, fn, scale, base, T, D, HC, iters, norm_eps, hc_eps, blockIdx.x,
                gridDim.x, (float*)raw);
}

__global__ __launch_bounds__(PLOW_THREADS) void k_expand(bf16* out, const bf16* branch,
                                                         const bf16* residual, const float* mix_in,
                                                         unsigned T, unsigned D, unsigned HC) {
    d_hc_expand(out, branch, residual, mix_in, T, D, HC, blockIdx.x, gridDim.x);
}

__global__ __launch_bounds__(PLOW_THREADS) void k_head(bf16* out, const bf16* x, const float* fn,
                                                       const float* scale, const float* base,
                                                       unsigned T, unsigned D, unsigned HC,
                                                       float norm_eps, float hc_eps) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_hc_reduce_head(out, x, fn, scale, base, T, D, HC, norm_eps, hc_eps, blockIdx.x, gridDim.x,
                     (float*)raw);
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

/* The reference split, in double. */
static void ref_split(const std::vector<double>& mixes, const float* scale, const float* base,
                      unsigned HC, unsigned iters, double eps, std::vector<double>& pre,
                      std::vector<double>& post, std::vector<double>& comb) {
    pre.assign(HC, 0.0);
    post.assign(HC, 0.0);
    comb.assign(HC * HC, 0.0);
    for (unsigned j = 0; j < HC; j++) {
        pre[j] = 1.0 / (1.0 + exp(-(mixes[j] * scale[0] + base[j]))) + eps;
        post[j] = 2.0 / (1.0 + exp(-(mixes[j + HC] * scale[1] + base[j + HC])));
    }
    for (unsigned j = 0; j < HC; j++)
        for (unsigned k = 0; k < HC; k++) {
            const unsigned m = 2u * HC + j * HC + k;
            comb[j * HC + k] = mixes[m] * scale[2] + base[m];
        }
    for (unsigned j = 0; j < HC; j++) { /* row softmax, then +eps */
        double mx = comb[j * HC];
        for (unsigned k = 1; k < HC; k++) mx = std::max(mx, comb[j * HC + k]);
        double s = 0.0;
        for (unsigned k = 0; k < HC; k++) {
            comb[j * HC + k] = exp(comb[j * HC + k] - mx);
            s += comb[j * HC + k];
        }
        for (unsigned k = 0; k < HC; k++) comb[j * HC + k] = comb[j * HC + k] / s + eps;
    }
    for (unsigned k = 0; k < HC; k++) { /* column pass */
        double c = eps;
        for (unsigned j = 0; j < HC; j++) c += comb[j * HC + k];
        for (unsigned j = 0; j < HC; j++) comb[j * HC + k] /= c;
    }
    for (unsigned it = 1; it < iters; it++) {
        for (unsigned j = 0; j < HC; j++) {
            double r = eps;
            for (unsigned k = 0; k < HC; k++) r += comb[j * HC + k];
            for (unsigned k = 0; k < HC; k++) comb[j * HC + k] /= r;
        }
        for (unsigned k = 0; k < HC; k++) {
            double c = eps;
            for (unsigned j = 0; j < HC; j++) c += comb[j * HC + k];
            for (unsigned j = 0; j < HC; j++) comb[j * HC + k] /= c;
        }
    }
}

/* `mixes` for one token: (hc_fn @ xf) * rsqrt(mean(xf^2) + norm_eps). */
static void ref_mixes(const bf16* xt, const float* fn, unsigned N, unsigned MIX, double norm_eps,
                      std::vector<double>& mixes) {
    double ss = 0.0;
    for (unsigned i = 0; i < N; i++) {
        const double v = bf2f_host(xt[i]);
        ss += v * v;
    }
    const double rs = 1.0 / sqrt(ss / (double)N + norm_eps);
    mixes.assign(MIX, 0.0);
    for (unsigned m = 0; m < MIX; m++) {
        double d = 0.0;
        for (unsigned i = 0; i < N; i++) d += bf2f_host(xt[i]) * (double)fn[(size_t)m * N + i];
        mixes[m] = d * rs;
    }
}

static double rms_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return sqrt(s / (double)v.size()) + 1e-9;
}

static int check(const char* what, const std::vector<double>& ref, const bf16* got, double tol) {
    double worst = 0.0;
    const double r = rms_of(ref);
    for (size_t i = 0; i < ref.size(); i++)
        worst = std::max(worst, fabs(ref[i] - bf2f_host(got[i])) / r);
    printf("  %-28s max rel %.3e  %s\n", what, worst, worst <= tol ? "ok" : "FAIL");
    return worst <= tol ? 0 : 1;
}

static int run_case(unsigned T, unsigned D, unsigned HC, unsigned iters, bool alias) {
    const unsigned MIX = (2u + HC) * HC, N = HC * D;
    const float norm_eps = 1e-6f, hc_eps = 1e-6f;
    const double TOL = 2e-2;
    printf("T=%u D=%u HC=%u iters=%u alias=%d\n", T, D, HC, iters, (int)alias);

    std::vector<bf16> x((size_t)T * N), branch((size_t)T * D);
    std::vector<float> fn((size_t)MIX * N), base(MIX), scale(3), hfn((size_t)HC * N), hbase(HC);
    for (auto& v : x) v = f2bf_host(frand());
    for (auto& v : branch) v = f2bf_host(frand());
    /* 1/sqrt(N) keeps the mixes O(1), which is where sigmoid and softmax are
     * actually sensitive — a saturated coefficient would hide a wrong formula. */
    const float s = 1.0f / sqrtf((float)N);
    for (auto& v : fn) v = frand() * s;
    for (auto& v : hfn) v = frand() * s;
    for (auto& v : base) v = frand();
    for (auto& v : hbase) v = frand();
    for (auto& v : scale) v = frand();

    bf16 *d_x, *d_out, *d_branch, *d_exp;
    float *d_fn, *d_scale, *d_base, *d_mix, *d_hfn, *d_hbase;
    HC_CHECK(hipMalloc(&d_x, x.size() * 2));
    HC_CHECK(hipMalloc(&d_out, (size_t)T * D * 2));
    HC_CHECK(hipMalloc(&d_branch, branch.size() * 2));
    HC_CHECK(hipMalloc(&d_exp, (size_t)T * N * 2));
    HC_CHECK(hipMalloc(&d_fn, fn.size() * 4));
    HC_CHECK(hipMalloc(&d_hfn, hfn.size() * 4));
    HC_CHECK(hipMalloc(&d_scale, 12));
    HC_CHECK(hipMalloc(&d_base, base.size() * 4));
    HC_CHECK(hipMalloc(&d_hbase, hbase.size() * 4));
    HC_CHECK(hipMalloc(&d_mix, (size_t)T * (HC + HC * HC) * 4));
    HC_CHECK(hipMemcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_branch, branch.data(), branch.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_fn, fn.data(), fn.size() * 4, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_hfn, hfn.data(), hfn.size() * 4, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_scale, scale.data(), 12, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_base, base.data(), base.size() * 4, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_hbase, hbase.data(), hbase.size() * 4, hipMemcpyHostToDevice));

    const unsigned blocks = T < 4u ? T : 4u;
    k_reduce<<<blocks, PLOW_THREADS>>>(d_out, d_mix, d_x, d_fn, d_scale, d_base, T, D, HC, iters,
                                       norm_eps, hc_eps);
    HC_CHECK(hipDeviceSynchronize());

    /* The expand writes back over the streams in a real layer, so exercise that. */
    if (alias) HC_CHECK(hipMemcpy(d_exp, d_x, x.size() * 2, hipMemcpyDeviceToDevice));
    k_expand<<<blocks, PLOW_THREADS>>>(alias ? d_exp : d_exp, d_branch, alias ? d_exp : d_x, d_mix,
                                       T, D, HC);
    HC_CHECK(hipDeviceSynchronize());

    k_head<<<blocks, PLOW_THREADS>>>(d_out + 0, d_x, d_hfn, d_scale, d_hbase, T, D, HC, norm_eps,
                                     hc_eps);
    std::vector<bf16> got_head((size_t)T * D);
    HC_CHECK(hipMemcpy(got_head.data(), d_out, got_head.size() * 2, hipMemcpyDeviceToHost));
    /* re-run reduce, since the head just overwrote `d_out` */
    k_reduce<<<blocks, PLOW_THREADS>>>(d_out, d_mix, d_x, d_fn, d_scale, d_base, T, D, HC, iters,
                                       norm_eps, hc_eps);
    HC_CHECK(hipDeviceSynchronize());

    std::vector<bf16> got_red((size_t)T * D), got_exp((size_t)T * N);
    HC_CHECK(hipMemcpy(got_red.data(), d_out, got_red.size() * 2, hipMemcpyDeviceToHost));
    HC_CHECK(hipMemcpy(got_exp.data(), d_exp, got_exp.size() * 2, hipMemcpyDeviceToHost));

    std::vector<double> ref_red((size_t)T * D), ref_exp((size_t)T * N), ref_hd((size_t)T * D);
    for (unsigned t = 0; t < T; t++) {
        const bf16* xt = &x[(size_t)t * N];
        std::vector<double> mixes, pre, post, comb;
        ref_mixes(xt, fn.data(), N, MIX, norm_eps, mixes);
        ref_split(mixes, scale.data(), base.data(), HC, iters, hc_eps, pre, post, comb);
        for (unsigned d = 0; d < D; d++) {
            double acc = 0.0;
            for (unsigned j = 0; j < HC; j++) acc += pre[j] * bf2f_host(xt[(size_t)j * D + d]);
            ref_red[(size_t)t * D + d] = acc;
        }
        for (unsigned k = 0; k < HC; k++)
            for (unsigned d = 0; d < D; d++) {
                double acc = post[k] * bf2f_host(branch[(size_t)t * D + d]);
                for (unsigned j = 0; j < HC; j++)
                    acc += comb[j * HC + k] * bf2f_host(xt[(size_t)j * D + d]);
                ref_exp[(size_t)t * N + (size_t)k * D + d] = acc;
            }
        /* hc_head: its own weights, a scalar scale, no Sinkhorn. */
        std::vector<double> hmix;
        ref_mixes(xt, hfn.data(), N, HC, norm_eps, hmix);
        for (unsigned d = 0; d < D; d++) {
            double acc = 0.0;
            for (unsigned j = 0; j < HC; j++) {
                const double p =
                    1.0 / (1.0 + exp(-(hmix[j] * scale[0] + hbase[j]))) + (double)hc_eps;
                acc += p * bf2f_host(xt[(size_t)j * D + d]);
            }
            ref_hd[(size_t)t * D + d] = acc;
        }
    }

    int bad = 0;
    bad |= check("hc_reduce", ref_red, got_red.data(), TOL);
    bad |= check("hc_expand", ref_exp, got_exp.data(), TOL);
    bad |= check("hc_reduce_head", ref_hd, got_head.data(), TOL);

    hipFree(d_x);
    hipFree(d_out);
    hipFree(d_branch);
    hipFree(d_exp);
    hipFree(d_fn);
    hipFree(d_hfn);
    hipFree(d_scale);
    hipFree(d_base);
    hipFree(d_hbase);
    hipFree(d_mix);
    return bad;
}

int main(void) {
    srand48(1);
    int bad = 0;
    /* HC=4 iters=20 is the shipped configuration; D=4096 is V4's hidden size. */
    bad |= run_case(1, 4096, 4, 20, false);
    bad |= run_case(5, 4096, 4, 20, true);
    /* HC=2 proves the fixed-size register vectors are not hard-coded to 4, and
     * iters=1 exercises the softmax-then-column path with no symmetric rounds. */
    bad |= run_case(3, 512, 2, 1, true);
    printf("%s\n", bad ? "FAIL"
                       : "PASS — d_hc_reduce/expand/head match the model.py + kernel.py reference");
    return bad;
}

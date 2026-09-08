/* d_flash_mla_prefill_v2<512, 0, ...> — the DR=0 (NoPE) MLA prefill arm.  [DSV4-MLA-PF]
 *
 * DeepSeek-V4-Flash caches ONE 512-wide latent row per token that is simultaneously K and V,
 * with the 64 rope dims living INSIDE that row (model.py:511 ropes kv[..., -64:] before the
 * cache write, model.py:506 ropes q). So QK and PV both contract over the full 512 and there
 * is no separate Krope cache: the geometry is DK=512, DR=0, not GLM/V3's DK=512, DR=64.
 *
 * TWO ORACLES, because one of them is exact and the other one is independent:
 *
 *  (1) BIT-IDENTITY against the SHIPPED <512,64> arm fed a zero rope strip. At DR=64 the QK
 *      MFMA contracts 576 = the same 512 followed by 64 products of zeros, accumulated into
 *      the same f32 register in the same kt order; x + 0.0f is exact, so the two arms must
 *      agree BIT FOR BIT on Opart and on (m,l). This is the strongest possible statement that
 *      the DR=0 specialization changed nothing but the work it skips, and it is the same
 *      device-as-oracle discipline mla_gfx950_test.c uses for the dense prefill.
 *
 *  (2) An f64 CPU reference of the same causal attention, compared on the NORMALIZED output
 *      Opart/l. The frame/max cancels in that ratio, so the reference needs no knowledge of
 *      FA_MLA_PF2_DEFER's exponent origin. Tolerance is the harness band mla_gfx950_test.c
 *      already justifies for this class (2e-2 max, 5e-3 rms, both scaled by |O|max) — the
 *      accumulation is bf16 P against a bf16 latent, so a per-element relative bound is the
 *      wrong shape.
 *
 * The GATHER arm gets both as well, over a randomly selected per-query row set packed into
 * the op-119 union layout (header of u32 counts, then per q-tile: int upos[cap], u32 mlo[cap],
 * u32 mhi[cap]). n_head is 8 because the head-batched pack is 8 queries x 8 heads = 64 rows.
 *
 * In nix develop:
 *   $PLOW_HIPCC --offload-arch=gfx942 -O3 -w -Iruntime/amd -Iruntime/common \
 *       -c runtime/tests/mla_prefill_nope_gfx942_test.hip -o /tmp/mlanope.o
 *   c++ /tmp/mlanope.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/mlanope
 *   perf-data/tools/gpulease -n 1 mla-nope-pf /tmp/mlanope
 */
#define PLOW_WG_WAVES 4
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#else
using bf16 = unsigned short;
#endif
#include <cmath>
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
    ck(hipMalloc(&p, v.size() * sizeof(T)));
    ck(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

template <typename T>
static void down(std::vector<T>& v, const T* p) {
    ck(hipMemcpy(v.data(), p, v.size() * sizeof(T), hipMemcpyDeviceToHost));
}

/* ---------------------------------------------------------------- device wrappers
 * ONE definition each, body guarded: the host pass must still see the __global__ so HIP
 * emits its launch stub (mixed_flash_prefill_gfx942_test.hip does the same). */
__global__ __launch_bounds__(256, 1) void k_nope(float* Op, float* ml, const bf16* Qabs,
                                                 const bf16* Ckv, const int* kv_len,
                                                 unsigned nb, unsigned nt, unsigned nh,
                                                 unsigned kvs, unsigned win, float sc) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[FA_MLA_PF2_LDS_BYTES(512, 0) / 2];
    d_flash_mla_prefill_v2<512, 0, false, false>(Op, ml, Qabs, nullptr, Ckv, nullptr, kv_len,
                                                 nb, nt, nh, kvs, win, sc, PLOW_KV_MASK_NONE,
                                                 blockIdx.x, gridDim.x, lds);
#endif
}

__global__ __launch_bounds__(256, 1) void k_rope0(float* Op, float* ml, const bf16* Qabs,
                                                  const bf16* Qrope, const bf16* Ckv,
                                                  const bf16* Krope, const int* kv_len,
                                                  unsigned nb, unsigned nt, unsigned nh,
                                                  unsigned kvs, unsigned win, float sc) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[FA_MLA_PF2_LDS_BYTES(512, 64) / 2];
    d_flash_mla_prefill_v2<512, 64, false, false>(Op, ml, Qabs, Qrope, Ckv, Krope, kv_len, nb,
                                                  nt, nh, kvs, win, sc, PLOW_KV_MASK_NONE,
                                                  blockIdx.x, gridDim.x, lds);
#endif
}

__global__ __launch_bounds__(256, 1) void k_nope_g(float* Op, float* ml, const bf16* Qabs,
                                                   const bf16* Ckv, const int* kv_len,
                                                   const unsigned char* uni, unsigned cap,
                                                   unsigned nb, unsigned nt, unsigned nh,
                                                   unsigned kvs, float sc) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[(FA_MLA_PF2_LDS_BYTES(512, 0) + 256) / 2];
    d_flash_mla_prefill_v2<512, 0, true, false>(Op, ml, Qabs, nullptr, Ckv, nullptr, kv_len,
                                                nb, nt, nh, kvs, 0u, sc, PLOW_KV_MASK_NONE,
                                                blockIdx.x, gridDim.x, lds, uni, cap);
#endif
}

__global__ __launch_bounds__(256, 1) void k_rope0_g(float* Op, float* ml, const bf16* Qabs,
                                                    const bf16* Qrope, const bf16* Ckv,
                                                    const bf16* Krope, const int* kv_len,
                                                    const unsigned char* uni, unsigned cap,
                                                    unsigned nb, unsigned nt, unsigned nh,
                                                    unsigned kvs, float sc) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[(FA_MLA_PF2_LDS_BYTES(512, 64) + 256) / 2];
    d_flash_mla_prefill_v2<512, 64, true, false>(Op, ml, Qabs, Qrope, Ckv, Krope, kv_len, nb,
                                                 nt, nh, kvs, 0u, sc, PLOW_KV_MASK_NONE,
                                                 blockIdx.x, gridDim.x, lds, uni, cap);
#endif
}

/* ---------------------------------------------------------------- host helpers */
static const int DK = 512, DR = 64;
static int fails = 0;

static unsigned rng_s = 0x13572468u;
static float rnd() { /* xorshift in [-1,1) */
    rng_s ^= rng_s << 13;
    rng_s ^= rng_s >> 17;
    rng_s ^= rng_s << 5;
    return (float)(int)(rng_s >> 8) / 8388608.0f - 1.0f;
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

/* Normalized-output comparison against the f64 reference, in mla_gfx950_test.c's shape:
 * error scaled by the magnitude of the whole output, not per element (a softmax-weighted
 * mixture crosses zero and a per-element relative bound is meaningless there). */
static void cmp_ref(const char* what, const std::vector<float>& got,
                    const std::vector<double>& want) {
    double amax = 0, emax = 0, se = 0, sw = 0;
    for (size_t i = 0; i < want.size(); i++) {
        const double d = std::fabs((double)got[i] - want[i]);
        amax = std::fmax(amax, std::fabs(want[i]));
        emax = std::fmax(emax, d);
        se += d * d;
        sw += want[i] * want[i];
    }
    const double rmax = amax > 0 ? emax / amax : emax;
    const double rrms = sw > 0 ? std::sqrt(se / sw) : std::sqrt(se);
    const bool ok = rmax < 2e-2 && rrms < 5e-3;
    std::printf("  %-46s %s  (max %.5f  rms %.6f  of |O|max=%.3f)\n", what,
                ok ? "ok " : "FAIL", rmax, rrms, amax);
    if (!ok) fails++;
}

static void cmp_exact(const char* what, const std::vector<float>& a,
                      const std::vector<float>& b) {
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); i++)
        if (std::memcmp(&a[i], &b[i], 4) != 0) diff++;
    const bool ok = diff == 0;
    std::printf("  %-46s %s  (%zu of %zu words differ)\n", what, ok ? "ok " : "FAIL", diff,
                a.size());
    if (!ok) fails++;
}

/* ---------------------------------------------------------------- dense case */
static void dense_case(unsigned nb, unsigned nt, unsigned nh, unsigned ctx, unsigned win) {
    const float sc = 1.0f / std::sqrt((float)DK);
    std::vector<bf16> Q((size_t)nb * nt * nh * DK), C((size_t)nb * ctx * DK);
    std::vector<bf16> Qr((size_t)nb * nt * nh * DR, 0), Kr((size_t)nb * ctx * DR, 0);
    /* Q AMPLITUDE IS PART OF THE TEST, not cosmetic. At |q| ~ |kv| ~ 0.5 the 512-wide dot
     * lands a logit std of ~0.15 and the softmax is nearly uniform, so the output barely
     * depends on the scores and a 5% scale error in the reference does not cross the
     * tolerance -- i.e. the harness stops testing the softmax. 16 puts the logit std near
     * 2.7 and a genuinely peaked mixture on the other side of it. mla_ref.rs records the
     * same trap for the decode fixture. */
    for (auto& v : Q) v = f2b(rnd() * 16.0f);
    for (auto& v : C) v = f2b(rnd() * 0.5f);
    std::vector<int> kl(nb, (int)ctx);

    const size_t on = (size_t)nb * nt * nh * DK, mn = (size_t)nb * nt * nh * 2;
    std::vector<float> O0(on, 0), M0(mn, 0), O1(on, 0), M1(mn, 0);
    auto *dQ = up(Q), *dC = up(C), *dQr = up(Qr), *dKr = up(Kr);
    auto* dkl = up(kl);
    auto *dO0 = up(O0), *dM0 = up(M0), *dO1 = up(O1), *dM1 = up(M1);

    hipLaunchKernelGGL(k_nope, dim3(64), dim3(256), 0, 0, dO0, dM0, dQ, dC, dkl, nb, nt, nh,
                       ctx, win, sc);
    ck(hipGetLastError());
    hipLaunchKernelGGL(k_rope0, dim3(64), dim3(256), 0, 0, dO1, dM1, dQ, dQr, dC, dKr, dkl, nb,
                       nt, nh, ctx, win, sc);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    down(O0, dO0);
    down(M0, dM0);
    down(O1, dO1);
    down(M1, dM1);

    char t[128];
    std::snprintf(t, sizeof t, "dense b%u t%u h%u ctx%u win%u : Opart == <512,64>@rope0", nb,
                  nt, nh, ctx, win);
    cmp_exact(t, O0, O1);
    std::snprintf(t, sizeof t, "dense b%u t%u h%u ctx%u win%u : (m,l) == <512,64>@rope0", nb,
                  nt, nh, ctx, win);
    cmp_exact(t, M0, M1);

    /* f64 reference on the normalized output. */
    std::vector<double> R(on, 0.0);
    std::vector<float> N(on, 0.0f);
    const unsigned q_pos0 = ctx - nt;
    for (unsigned b = 0; b < nb; b++)
        for (unsigned q = 0; q < nt; q++) {
            const unsigned qg = q_pos0 + q;
            for (unsigned h = 0; h < nh; h++) {
                const bf16* qp = &Q[(((size_t)b * nt + q) * nh + h) * DK];
                double mx = -1e300;
                std::vector<double> s(ctx, -1e300);
                for (unsigned j = 0; j <= qg; j++) {
                    if (win && (qg - j) >= win) continue;
                    const bf16* kp = &C[((size_t)b * ctx + j) * DK];
                    double acc = 0;
                    for (int d = 0; d < DK; d++) acc += (double)b2f(qp[d]) * (double)b2f(kp[d]);
                    s[j] = acc * (double)sc;
                    mx = std::fmax(mx, s[j]);
                }
                double den = 0;
                std::vector<double> acc(DK, 0.0);
                for (unsigned j = 0; j <= qg; j++) {
                    if (s[j] == -1e300) continue;
                    const double p = std::exp(s[j] - mx);
                    den += p;
                    const bf16* kp = &C[((size_t)b * ctx + j) * DK];
                    for (int d = 0; d < DK; d++) acc[d] += p * (double)b2f(kp[d]);
                }
                const size_t o = (((size_t)b * nt + q) * nh + h) * DK;
                const float l = M0[(((size_t)b * nt + q) * nh + h) * 2 + 1];
                for (int d = 0; d < DK; d++) {
                    R[o + d] = den > 0 ? acc[d] / den : 0.0;
                    N[o + d] = l > 0 ? O0[o + d] / l : 0.0f;
                }
            }
        }
    std::snprintf(t, sizeof t, "dense b%u t%u h%u ctx%u win%u : O/l vs f64 reference", nb, nt,
                  nh, ctx, win);
    cmp_ref(t, N, R);

    hipFree(dQ);
    hipFree(dC);
    hipFree(dQr);
    hipFree(dKr);
    hipFree(dkl);
    hipFree(dO0);
    hipFree(dM0);
    hipFree(dO1);
    hipFree(dM1);
}

/* ---------------------------------------------------------------- gather case */
static void gather_case(unsigned nb, unsigned nt, unsigned ctx, unsigned topk) {
    const unsigned nh = 8, QP = 8; /* the head-batched pack is 8 queries x 8 heads */
    const float sc = 1.0f / std::sqrt((float)DK);
    const unsigned n_qt = (nt + QP - 1) / QP;
    std::vector<bf16> Q((size_t)nb * nt * nh * DK), C((size_t)nb * ctx * DK);
    std::vector<bf16> Qr((size_t)nb * nt * nh * DR, 0), Kr((size_t)nb * ctx * DR, 0);
    for (auto& v : Q) v = f2b(rnd() * 16.0f); /* see dense_case: a flat softmax tests nothing */
    for (auto& v : C) v = f2b(rnd() * 0.5f);
    std::vector<int> kl(nb, (int)ctx);
    const unsigned q_pos0 = ctx - nt;

    /* Per-query causal selection, unioned per pack; then the op-119 byte layout.
     * NOTE the GATHER arm reads ONE union table for the whole launch (no batch axis), so
     * this case is nb == 1 by construction — the same shape the DSA prefill emits. */
    std::vector<std::vector<unsigned>> sel(nt);
    for (unsigned q = 0; q < nt; q++) {
        const unsigned qg = q_pos0 + q;
        for (unsigned j = 0; j <= qg; j++)
            if ((rng_s = rng_s * 1664525u + 1013904223u) % 100u < topk) sel[q].push_back(j);
        if (sel[q].empty()) sel[q].push_back(qg); /* never leave a row with no evidence */
    }
    unsigned cap = 0;
    std::vector<std::vector<unsigned>> uni(n_qt);
    for (unsigned t = 0; t < n_qt; t++) {
        std::vector<char> seen(ctx, 0);
        for (unsigned k = 0; k < QP; k++) {
            const unsigned q = t * QP + k;
            if (q >= nt) break;
            for (unsigned j : sel[q])
                if (!seen[j]) {
                    seen[j] = 1;
                    uni[t].push_back(j);
                }
        }
        cap = std::max<unsigned>(cap, (unsigned)uni[t].size());
    }
    const size_t hdr = ((size_t)n_qt * 4 + 255) / 256 * 256;
    std::vector<unsigned char> U(hdr + (size_t)n_qt * cap * 12, 0);
    for (unsigned t = 0; t < n_qt; t++) {
        ((unsigned*)U.data())[t] = (unsigned)uni[t].size();
        int* pos = (int*)(U.data() + hdr + (size_t)t * cap * 12);
        unsigned* lo = (unsigned*)(U.data() + hdr + (size_t)t * cap * 12 + (size_t)cap * 4);
        unsigned* hi = (unsigned*)(U.data() + hdr + (size_t)t * cap * 12 + (size_t)cap * 8);
        for (unsigned u = 0; u < uni[t].size(); u++) {
            pos[u] = (int)uni[t][u];
            unsigned long long m = 0;
            for (unsigned k = 0; k < QP; k++) {
                const unsigned q = t * QP + k;
                if (q >= nt) break;
                for (unsigned j : sel[q])
                    if (j == uni[t][u]) m |= 1ull << k;
            }
            lo[u] = (unsigned)(m & 0xffffffffu);
            hi[u] = (unsigned)(m >> 32);
        }
    }

    const size_t on = (size_t)nb * nt * nh * DK, mn = (size_t)nb * nt * nh * 2;
    std::vector<float> O0(on, 0), M0(mn, 0), O1(on, 0), M1(mn, 0);
    auto *dQ = up(Q), *dC = up(C), *dQr = up(Qr), *dKr = up(Kr);
    auto* dkl = up(kl);
    auto* dU = up(U);
    auto *dO0 = up(O0), *dM0 = up(M0), *dO1 = up(O1), *dM1 = up(M1);

    hipLaunchKernelGGL(k_nope_g, dim3(64), dim3(256), 0, 0, dO0, dM0, dQ, dC, dkl, dU, cap, nb,
                       nt, nh, ctx, sc);
    ck(hipGetLastError());
    hipLaunchKernelGGL(k_rope0_g, dim3(64), dim3(256), 0, 0, dO1, dM1, dQ, dQr, dC, dKr, dkl,
                       dU, cap, nb, nt, nh, ctx, sc);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    down(O0, dO0);
    down(M0, dM0);
    down(O1, dO1);
    down(M1, dM1);

    char t[128];
    std::snprintf(t, sizeof t, "gather t%u ctx%u sel~%u%% : Opart == <512,64>@rope0", nt, ctx,
                  topk);
    cmp_exact(t, O0, O1);
    std::snprintf(t, sizeof t, "gather t%u ctx%u sel~%u%% : (m,l) == <512,64>@rope0", nt, ctx,
                  topk);
    cmp_exact(t, M0, M1);

    std::vector<double> R(on, 0.0);
    std::vector<float> N(on, 0.0f);
    for (unsigned q = 0; q < nt; q++)
        for (unsigned h = 0; h < nh; h++) {
            const bf16* qp = &Q[((size_t)q * nh + h) * DK];
            double mx = -1e300;
            std::vector<double> s(sel[q].size());
            for (size_t i = 0; i < sel[q].size(); i++) {
                const bf16* kp = &C[(size_t)sel[q][i] * DK];
                double acc = 0;
                for (int d = 0; d < DK; d++) acc += (double)b2f(qp[d]) * (double)b2f(kp[d]);
                s[i] = acc * (double)sc;
                mx = std::fmax(mx, s[i]);
            }
            double den = 0;
            std::vector<double> acc(DK, 0.0);
            for (size_t i = 0; i < sel[q].size(); i++) {
                const double p = std::exp(s[i] - mx);
                den += p;
                const bf16* kp = &C[(size_t)sel[q][i] * DK];
                for (int d = 0; d < DK; d++) acc[d] += p * (double)b2f(kp[d]);
            }
            const size_t o = ((size_t)q * nh + h) * DK;
            const float l = M0[((size_t)q * nh + h) * 2 + 1];
            for (int d = 0; d < DK; d++) {
                R[o + d] = den > 0 ? acc[d] / den : 0.0;
                N[o + d] = l > 0 ? O0[o + d] / l : 0.0f;
            }
        }
    std::snprintf(t, sizeof t, "gather t%u ctx%u sel~%u%% : O/l vs f64 reference", nt, ctx,
                  topk);
    cmp_ref(t, N, R);

    hipFree(dQ);
    hipFree(dC);
    hipFree(dQr);
    hipFree(dKr);
    hipFree(dkl);
    hipFree(dU);
    hipFree(dO0);
    hipFree(dM0);
    hipFree(dO1);
    hipFree(dM1);
}

int main() {
    std::printf("d_flash_mla_prefill_v2<512,0> — DeepSeek-V4 NoPE MLA prefill\n");
    /* Token counts straddle the BQ=64 tile edge, as mla_gfx950_test.c's TOKS[] does. */
    dense_case(1, 1, 8, 128, 0);
    dense_case(1, 64, 8, 128, 0);
    dense_case(1, 65, 8, 192, 0);
    dense_case(1, 130, 8, 256, 0);
    dense_case(2, 96, 4, 256, 0);
    /* DeepSeek-V4's kind-A/window layers: a 128-token sliding window. */
    dense_case(1, 130, 8, 256, 128);
    /* GATHER: the CSA layers' selected union. */
    gather_case(1, 8, 64, 40);
    gather_case(1, 64, 128, 25);
    gather_case(1, 130, 256, 10);
    std::printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

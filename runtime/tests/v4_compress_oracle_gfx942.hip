/* v4_compress_oracle_gfx942.hip — `d_v4_kv_compress` against a HOST
 * TRANSCRIPTION of `inference/model.py::Compressor.forward` at `start_pos == 0`.
 *
 * WHY THIS EXISTS. 41 of 43 layers compress their KV history, and the pooling
 * is learned and per-dim — there is nothing else in the tree shaped like it, so
 * no existing gate covers any part of it.
 *
 * THE REFERENCE, for output group `g`:
 *
 *     kv    = wkv  @ x[tok]                        # [COFF*D]
 *     score = wgate@ x[tok] + ape[pos_in_window]   # [COFF*D]
 *     rows  = ratio           (plain)  : tok = g*ratio + i,     cols [0:D]
 *           = 2*ratio         (overlap): rows [0,ratio) -> tok = (g-1)*ratio+i, cols [0:D]
 *                                        rows [ratio,2r) -> tok = g*ratio+i,    cols [D:2D]
 *     out[d] = sum_rows softmax_over_rows(score[row][d]) * kv[row][d]
 *     out    = rmsnorm(out, norm_w)
 *
 * WHAT THIS PINS:
 *   * THE SOFTMAX IS PER OUTPUT DIM, not one weight per pooled row.
 *     NEGATIVE-CONTROLLED.
 *   * the overlapped form reads the PREVIOUS window through the FIRST half of
 *     the projection and the current through the SECOND. Swapping the halves is
 *     NEGATIVE-CONTROLLED.
 *   * group 0 of an overlapped layer has no previous window: those rows are
 *     inert (-inf score), not zeros averaged in.
 *   * `ape` is indexed by position within the token's OWN window, so an
 *     overlapped group draws ape rows from two different windows.
 *
 * SWEPT: ratio 4 with overlap (21 layers) and ratio 128 without (20 layers),
 * both at D=512; several groups, so group 0's missing window is exercised
 * alongside groups that have one.
 *
 * TOLERANCE 2e-2 relative to the row RMS, as in the other V4 gates. Host in
 * double.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/v4_compress_oracle_gfx942.hip -o /tmp/v4c && /tmp/v4c
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

enum { LDS_BYTES = 16384 };

__global__ __launch_bounds__(PLOW_THREADS) void k_comp(bf16* out, const bf16* x, const bf16* wkv,
                                                       const bf16* wgate, const float* ape,
                                                       const bf16* norm_w, unsigned T,
                                                       unsigned HID, unsigned D, unsigned ratio,
                                                       unsigned overlap, float eps) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_v4_kv_compress(out, x, wkv, wgate, ape, norm_w, T, HID, D, ratio, overlap, eps, blockIdx.x,
                     gridDim.x, (float*)raw);
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

static int run_case(unsigned T, unsigned HID, unsigned D, unsigned ratio, unsigned overlap) {
    const unsigned COFF = overlap ? 2u : 1u, G = T / ratio, NROW = overlap ? 2u * ratio : ratio;
    const float eps = 1e-6f;
    const double TOL = 2e-2;
    printf("T=%u HID=%u D=%u ratio=%u overlap=%u groups=%u\n", T, HID, D, ratio, overlap, G);

    std::vector<bf16> x((size_t)T * HID), wkv((size_t)COFF * D * HID), wgate((size_t)COFF * D * HID),
        nw(D);
    std::vector<float> ape((size_t)ratio * COFF * D);
    for (auto& v : x) v = f2bf_host(frand());
    /* 1/sqrt(HID) keeps the projections O(1) so the softmax is in its sensitive
     * range — saturated scores would hide a wrong pooling. */
    const float s = 1.0f / sqrtf((float)HID);
    for (auto& v : wkv) v = f2bf_host(frand() * s);
    for (auto& v : wgate) v = f2bf_host(frand() * s);
    for (auto& v : ape) v = frand();
    for (auto& v : nw) v = f2bf_host(1.0f + 0.1f * frand());

    bf16 *d_x, *d_wkv, *d_wg, *d_nw, *d_out;
    float* d_ape;
    HC_CHECK(hipMalloc(&d_x, x.size() * 2));
    HC_CHECK(hipMalloc(&d_wkv, wkv.size() * 2));
    HC_CHECK(hipMalloc(&d_wg, wgate.size() * 2));
    HC_CHECK(hipMalloc(&d_nw, nw.size() * 2));
    HC_CHECK(hipMalloc(&d_ape, ape.size() * 4));
    HC_CHECK(hipMalloc(&d_out, (size_t)G * D * 2));
    HC_CHECK(hipMemcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_wkv, wkv.data(), wkv.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_wg, wgate.data(), wgate.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_nw, nw.data(), nw.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_ape, ape.data(), ape.size() * 4, hipMemcpyHostToDevice));

    const unsigned blocks = G < 4u ? G : 4u;
    k_comp<<<blocks, PLOW_THREADS>>>(d_out, d_x, d_wkv, d_wg, d_ape, d_nw, T, HID, D, ratio,
                                     overlap, eps);
    HC_CHECK(hipDeviceSynchronize());
    std::vector<bf16> got((size_t)G * D);
    HC_CHECK(hipMemcpy(got.data(), d_out, got.size() * 2, hipMemcpyDeviceToHost));

    std::vector<double> ref((size_t)G * D);
    for (unsigned g = 0; g < G; g++) {
        std::vector<double> pooled(D, 0.0);
        for (unsigned d = 0; d < D; d++) {
            std::vector<double> sc, kvv;
            for (unsigned r = 0; r < NROW; r++) {
                unsigned tok, wrow, arow;
                if (!overlap) {
                    tok = g * ratio + r;
                    wrow = d;
                    arow = r;
                } else if (r < ratio) {
                    if (g == 0) continue;
                    tok = (g - 1) * ratio + r;
                    wrow = d;
                    arow = r;
                } else {
                    const unsigned i = r - ratio;
                    tok = g * ratio + i;
                    wrow = D + d;
                    arow = i;
                }
                double kd = 0.0, sd = 0.0;
                for (unsigned i = 0; i < HID; i++) {
                    const double xv = bf2f_host(x[(size_t)tok * HID + i]);
                    kd += xv * bf2f_host(wkv[(size_t)wrow * HID + i]);
                    sd += xv * bf2f_host(wgate[(size_t)wrow * HID + i]);
                }
                sd += ape[(size_t)arow * COFF * D + wrow];
                kvv.push_back(kd);
                sc.push_back(sd);
            }
            double mx = -INFINITY;
            for (double v : sc) mx = std::max(mx, v);
            double den = 0.0, num = 0.0;
            for (size_t i = 0; i < sc.size(); i++) {
                const double e = exp(sc[i] - mx);
                den += e;
                num += e * kvv[i];
            }
            pooled[d] = den > 0.0 ? num / den : 0.0;
        }
        double ss = 0.0;
        for (unsigned d = 0; d < D; d++) ss += pooled[d] * pooled[d];
        const double rs = 1.0 / sqrt(ss / (double)D + eps);
        for (unsigned d = 0; d < D; d++)
            ref[(size_t)g * D + d] = pooled[d] * rs * bf2f_host(nw[d]);
    }

    double worst = 0.0, rms = 0.0;
    for (double v : ref) rms += v * v;
    rms = sqrt(rms / (double)ref.size()) + 1e-9;
    for (size_t i = 0; i < ref.size(); i++)
        worst = std::max(worst, fabs(ref[i] - bf2f_host(got[i])) / rms);
    printf("  %-28s max rel %.3e  %s\n", "kv_compress", worst, worst <= TOL ? "ok" : "FAIL");

    hipFree(d_x);
    hipFree(d_wkv);
    hipFree(d_wg);
    hipFree(d_nw);
    hipFree(d_ape);
    hipFree(d_out);
    return worst <= TOL ? 0 : 1;
}

int main(void) {
    srand48(11);
    int bad = 0;
    /* The two shipped shapes at V4's head_dim. HID is scaled down from 4096 to
     * keep the host oracle's O(G*D*NROW*HID) double loop tractable; the pooling
     * structure this gate is about does not depend on it. */
    bad |= run_case(32, 256, 512, 4, 1);   /* ratio 4, overlapping — 21 layers */
    bad |= run_case(512, 128, 512, 128, 0); /* ratio 128, plain — 20 layers */
    /* The indexer's compressor: same op, index width. */
    bad |= run_case(16, 256, 128, 4, 1);
    printf("%s\n", bad ? "FAIL" : "PASS — d_v4_kv_compress matches Compressor.forward");
    return bad;
}

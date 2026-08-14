/* v4_compress_step_gfx942.hip — the decode-incremental KV compressor against
 * the PREFILL form of the same op.
 *
 * WHY THIS GATE AND NOT ANOTHER HOST TRANSCRIPTION. `d_v4_kv_compress` is
 * already gated against `Compressor.forward` at `start_pos == 0`. What the
 * decode path has to be is THE SAME FUNCTION, arrived at one token at a time
 * through a carried window state — so the strongest statement available is that
 * feeding a sequence through T decode steps produces, entry for entry, what
 * pooling that sequence in one shot produces. A second host transcription would
 * only re-check my reading of the reference; this checks the thing that can
 * actually be wrong, which is the state machine.
 *
 * WHAT IT PINS, none of which prefill can exercise:
 *   * the ring slot is `start_pos % ratio` (plain) or `ratio + start_pos % ratio`
 *     (overlap), and an emit happens when `(start_pos+1) % ratio == 0`.
 *   * the overlap SHIFT — `state[:ratio] = state[ratio:]` after each emit — is
 *     what makes a window see its predecessor. Prefill has both windows in hand
 *     and never needs it. NEGATIVE-CONTROLLED by dropping it.
 *   * group 0 of an overlapped layer skips the absent predecessor even when
 *     runtime state is zero-initialized.
 *
 * The projections are computed on the host and handed in, exactly as the
 * emitter will hand in two `Gemv` results, so this isolates the state machine.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/v4_compress_step_gfx942.hip -o /tmp/v4cs && /tmp/v4cs
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

__global__ __launch_bounds__(PLOW_THREADS) void k_pf(bf16* out, const bf16* x, const bf16* wkv,
                                                     const bf16* wg, const float* ape,
                                                     const bf16* nw, unsigned T, unsigned HID,
                                                     unsigned D, unsigned ratio, unsigned ov,
                                                     float eps) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_v4_kv_compress(out, x, wkv, wg, ape, nw, T, HID, D, ratio, ov, eps, blockIdx.x, gridDim.x,
                     (float*)raw);
}

__global__ __launch_bounds__(PLOW_THREADS) void k_step(bf16* out, float* kvs, float* scs,
                                                       const float* kvp, const float* scp,
                                                       const float* ape, const bf16* nw,
                                                       unsigned D, unsigned ratio, unsigned ov,
                                                       unsigned start_pos, float eps) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_v4_kv_compress_step(out, kvs, scs, kvp, scp, ape, nw, D, ratio, ov, start_pos, eps,
                          blockIdx.x, gridDim.x, (float*)raw);
}

static bf16 f2bf_host(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf2f_host(bf16 b) {
    const unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float frand(void) { return (float)drand48() * 2.0f - 1.0f; }

static int run_case(unsigned T, unsigned HID, unsigned D, unsigned ratio, unsigned ov,
                    bool break_shift) {
    const unsigned COFF = ov ? 2u : 1u, W = COFF * D, G = T / ratio;
    const float eps = 1e-6f;
    printf("T=%u HID=%u D=%u ratio=%u overlap=%u groups=%u%s\n", T, HID, D, ratio, ov, G,
           break_shift ? "   [control: shift disabled]" : "");

    std::vector<bf16> x((size_t)T * HID), wkv((size_t)W * HID), wg((size_t)W * HID), nw(D);
    std::vector<float> ape((size_t)ratio * W);
    const float s = 1.0f / sqrtf((float)HID);
    for (auto& v : x) v = f2bf_host(frand());
    for (auto& v : wkv) v = f2bf_host(frand() * s);
    for (auto& v : wg) v = f2bf_host(frand() * s);
    for (auto& v : ape) v = frand();
    for (auto& v : nw) v = f2bf_host(1.0f + 0.1f * frand());

    /* ---- prefill reference, on device -------------------------------- */
    bf16 *d_x, *d_wkv, *d_wg, *d_nw, *d_pf;
    float* d_ape;
    HC_CHECK(hipMalloc(&d_x, x.size() * 2));
    HC_CHECK(hipMalloc(&d_wkv, wkv.size() * 2));
    HC_CHECK(hipMalloc(&d_wg, wg.size() * 2));
    HC_CHECK(hipMalloc(&d_nw, nw.size() * 2));
    HC_CHECK(hipMalloc(&d_ape, ape.size() * 4));
    HC_CHECK(hipMalloc(&d_pf, (size_t)G * D * 2));
    HC_CHECK(hipMemcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_wkv, wkv.data(), wkv.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_wg, wg.data(), wg.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_nw, nw.data(), nw.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_ape, ape.data(), ape.size() * 4, hipMemcpyHostToDevice));
    k_pf<<<G < 4 ? G : 4, PLOW_THREADS>>>(d_pf, d_x, d_wkv, d_wg, d_ape, d_nw, T, HID, D, ratio,
                                          ov, eps);
    HC_CHECK(hipDeviceSynchronize());
    std::vector<bf16> pf((size_t)G * D);
    HC_CHECK(hipMemcpy(pf.data(), d_pf, pf.size() * 2, hipMemcpyDeviceToHost));

    /* ---- decode: one token at a time --------------------------------- */
    std::vector<float> kvs((size_t)COFF * ratio * W, 0.0f);
    std::vector<float> scs((size_t)COFF * ratio * W, 0.0f);
    float *d_kvs, *d_scs, *d_kvp, *d_scp;
    bf16* d_slot;
    HC_CHECK(hipMalloc(&d_kvs, kvs.size() * 4));
    HC_CHECK(hipMalloc(&d_scs, scs.size() * 4));
    HC_CHECK(hipMalloc(&d_kvp, W * 4));
    HC_CHECK(hipMalloc(&d_scp, W * 4));
    HC_CHECK(hipMalloc(&d_slot, (size_t)G * D * 2));
    HC_CHECK(hipMemcpy(d_kvs, kvs.data(), kvs.size() * 4, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_scs, scs.data(), scs.size() * 4, hipMemcpyHostToDevice));
    HC_CHECK(hipMemset(d_slot, 0, (size_t)G * D * 2));

    std::vector<float> kvp(W), scp(W);
    for (unsigned t = 0; t < T; t++) {
        for (unsigned o = 0; o < W; o++) {
            double a = 0.0, b = 0.0;
            for (unsigned i = 0; i < HID; i++) {
                const double xv = bf2f_host(x[(size_t)t * HID + i]);
                a += xv * bf2f_host(wkv[(size_t)o * HID + i]);
                b += xv * bf2f_host(wg[(size_t)o * HID + i]);
            }
            kvp[o] = (float)a;
            scp[o] = (float)b;
        }
        HC_CHECK(hipMemcpy(d_kvp, kvp.data(), W * 4, hipMemcpyHostToDevice));
        HC_CHECK(hipMemcpy(d_scp, scp.data(), W * 4, hipMemcpyHostToDevice));
        const unsigned g = t / ratio;
        k_step<<<1, PLOW_THREADS>>>(d_slot + (size_t)g * D, d_kvs, d_scs, d_kvp, d_scp, d_ape,
                                    d_nw, D, ratio, break_shift ? 0u : ov, t, eps);
        HC_CHECK(hipDeviceSynchronize());
    }
    std::vector<bf16> dec((size_t)G * D);
    HC_CHECK(hipMemcpy(dec.data(), d_slot, dec.size() * 2, hipMemcpyDeviceToHost));

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < pf.size(); i++) {
        const double a = bf2f_host(pf[i]), b = bf2f_host(dec[i]);
        num = std::max(num, fabs(a - b));
        den += a * a;
    }
    const double rel = num / (sqrt(den / (double)pf.size()) + 1e-9);
    if (rel > 3e-2) {
        const double rms = sqrt(den / (double)pf.size()) + 1e-9;
        for (unsigned g = 0; g < G; g++) {
            double gm = 0.0;
            for (unsigned d = 0; d < D; d++)
                gm = std::max(gm, fabs(bf2f_host(pf[(size_t)g * D + d]) -
                                       bf2f_host(dec[(size_t)g * D + d])));
            printf("      group %2u  max rel %.3e\n", g, gm / rms);
        }
    }
    const bool ok = rel <= 3e-2;
    printf("  decode vs prefill        max rel %.3e  %s\n", rel, ok ? "ok" : "FAIL");

    hipFree(d_x); hipFree(d_wkv); hipFree(d_wg); hipFree(d_nw); hipFree(d_ape); hipFree(d_pf);
    hipFree(d_kvs); hipFree(d_scs); hipFree(d_kvp); hipFree(d_scp); hipFree(d_slot);
    return ok ? 0 : 1;
}

int main(void) {
    srand48(29);
    int bad = 0;
    /* ratio 4 overlapping — the 21 indexed layers. */
    bad |= run_case(32, 256, 512, 4, 1, false);
    /* ratio 128 plain — the other 20 compressed layers. */
    bad |= run_case(512, 128, 512, 128, 0, false);
    /* The indexer's own compressor width. */
    bad |= run_case(16, 256, 128, 4, 1, false);
    /* CONTROL: disabling the overlap (and so the shift) must break the match —
     * otherwise this gate never actually tested the carried window. */
    const int ctl = run_case(32, 256, 512, 4, 1, true);
    printf("  control (no overlap/shift): %s\n", ctl ? "ok (caught)" : "FAIL (gate is blind)");
    bad |= (ctl == 0);
    printf("%s\n", bad ? "FAIL" : "PASS — decode steps reproduce the prefill compressor");
    return bad;
}

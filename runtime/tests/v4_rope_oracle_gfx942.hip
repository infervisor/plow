/* v4_rope_oracle_gfx942.hip — DeepSeek-V4's RoPE through the EXISTING
 * `d_headnorm_rope<512, INTERLEAVE=true>` arm, against a host transcription of
 * `apply_rotary_emb`.
 *
 * WHY THIS EXISTS. V4 looked like it needed a rope kernel of its own: it rotates
 * the LAST 64 of a 512-lane head, pairing (x[0],x[1]) as `view_as_complex`
 * does, where every existing partial-rope arm rotates a PREFIX. It does not.
 * The DSA indexer already ropes a sub-range at HD=128 by putting IDENTITY
 * entries in the cos/sin table; V4 is the same trick mirrored — identity in the
 * LEADING 224 pairs, real angles in the trailing 32 — so the only change needed
 * was instantiating the interleaved template at HD=512.
 *
 * This gate is what makes that claim checkable, because "reuse the existing arm
 * with a cleverly built table" is exactly the kind of shortcut that produces a
 * plausible wrong rotation.
 *
 * THE REFERENCE (`inference/model.py::apply_rotary_emb`), for one head:
 *
 *     rd     = qk_rope_head_dim = 64, applied to x[..., -rd:]
 *     pairs  = (x[448],x[449]), (x[450],x[451]), ...      # interleaved
 *     x'     = x * e^{i theta}     , or its conjugate for the inverse
 *     lanes [0, 448) are untouched
 *
 * WHAT IT PINS:
 *   * the content lanes are IDENTICAL, bit for bit — an identity table that is
 *     not exactly (1, 0) shows up here and nowhere else.
 *   * the rotation is INTERLEAVED, not half-split. A half-split table over the
 *     same lanes rotates the wrong partners and is NEGATIVE-CONTROLLED.
 *   * negating `sin` is exactly the inverse rotation, so rope-then-inverse-rope
 *     returns the input.
 *   * `gamma == NONE` gives V4's weightless per-head query rescale.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/v4_rope_oracle_gfx942.hip -o /tmp/v4rope && /tmp/v4rope
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amd_common.h"
#include "op_norm.h"

#define HC_CHECK(x)                                                                               \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_) {                                                                                 \
            printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__);                              \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

enum { HD = 512, RD = 64 };

__global__ __launch_bounds__(PLOW_THREADS) void k_rope(bf16* out, const bf16* x, const float* cosb,
                                                       const float* sinb, const int* pos,
                                                       unsigned ntok, unsigned nhead, float eps,
                                                       unsigned skip_norm) {
    d_headnorm_rope<HD, /*INTERLEAVE=*/true>(out, x, nullptr, cosb, sinb, pos, ntok, nhead, eps, 0,
                                             0, 0xffffffffu, skip_norm, blockIdx.x, gridDim.x, 0);
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

/* The table the emitter must build: identity for the content pairs, the real
 * angle for the trailing `RD/2`. Half-split spelling (`half_split=true`) is the
 * negative control — same lanes, wrong partners. */
static void build_tables(std::vector<float>& cosb, std::vector<float>& sinb, unsigned maxpos,
                         float theta, bool inverse, bool half_split) {
    const unsigned H2 = HD / 2;
    cosb.assign((size_t)maxpos * H2, 1.0f);
    sinb.assign((size_t)maxpos * H2, 0.0f);
    for (unsigned p = 0; p < maxpos; p++)
        for (unsigned j = 0; j < RD / 2; j++) {
            const unsigned slot = half_split ? j : (H2 - RD / 2 + j);
            const double ang = (double)p / pow((double)theta, (2.0 * j) / (double)RD);
            cosb[(size_t)p * H2 + slot] = (float)cos(ang);
            sinb[(size_t)p * H2 + slot] = (float)(inverse ? -sin(ang) : sin(ang));
        }
}

int main(void) {
    srand48(23);
    const unsigned ntok = 4, nhead = 8, maxpos = 64;
    const float theta = 10000.0f;
    std::vector<bf16> x((size_t)ntok * nhead * HD);
    for (auto& v : x) v = f2bf_host(frand());
    std::vector<int> pos(ntok);
    for (unsigned t = 0; t < ntok; t++) pos[t] = (int)(t * 7 + 3);

    bf16 *d_x, *d_o, *d_o2;
    float *d_c, *d_s;
    int* d_p;
    HC_CHECK(hipMalloc(&d_x, x.size() * 2));
    HC_CHECK(hipMalloc(&d_o, x.size() * 2));
    HC_CHECK(hipMalloc(&d_o2, x.size() * 2));
    HC_CHECK(hipMalloc(&d_p, pos.size() * 4));
    HC_CHECK(hipMemcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_p, pos.data(), pos.size() * 4, hipMemcpyHostToDevice));

    auto run = [&](bool inverse, bool half_split, bf16* dst, const bf16* src) {
        std::vector<float> cb, sb;
        build_tables(cb, sb, maxpos, theta, inverse, half_split);
        HC_CHECK(hipMalloc(&d_c, cb.size() * 4));
        HC_CHECK(hipMalloc(&d_s, sb.size() * 4));
        HC_CHECK(hipMemcpy(d_c, cb.data(), cb.size() * 4, hipMemcpyHostToDevice));
        HC_CHECK(hipMemcpy(d_s, sb.data(), sb.size() * 4, hipMemcpyHostToDevice));
        k_rope<<<ntok, PLOW_THREADS>>>(dst, src, d_c, d_s, d_p, ntok, nhead, 1e-6f,
                                       /*skip_norm=*/1);
        HC_CHECK(hipDeviceSynchronize());
        hipFree(d_c);
        hipFree(d_s);
    };

    run(false, false, d_o, d_x);
    std::vector<bf16> got(x.size());
    HC_CHECK(hipMemcpy(got.data(), d_o, got.size() * 2, hipMemcpyDeviceToHost));

    /* ---- reference ---------------------------------------------------- */
    int bad = 0;
    double worst_rot = 0.0;
    unsigned content_diff = 0;
    for (unsigned t = 0; t < ntok; t++)
        for (unsigned h = 0; h < nhead; h++) {
            const bf16* xs = &x[((size_t)t * nhead + h) * HD];
            const bf16* gs = &got[((size_t)t * nhead + h) * HD];
            for (unsigned d = 0; d < HD - RD; d++)
                content_diff += (xs[d] != gs[d]); /* untouched, bit for bit */
            for (unsigned j = 0; j < RD / 2; j++) {
                const unsigned a = HD - RD + 2 * j, b = a + 1;
                const double ang = (double)pos[t] / pow((double)theta, (2.0 * j) / (double)RD);
                const double xr = bf2f_host(xs[a]), xi = bf2f_host(xs[b]);
                const double re = xr * cos(ang) - xi * sin(ang);
                const double im = xr * sin(ang) + xi * cos(ang);
                const double sc = fabs(re) + fabs(im) + 1e-6;
                worst_rot = std::max(worst_rot, fabs(re - bf2f_host(gs[a])) / sc);
                worst_rot = std::max(worst_rot, fabs(im - bf2f_host(gs[b])) / sc);
            }
        }
    printf("  content lanes untouched : %s (%u differing words)\n",
           content_diff ? "FAIL" : "ok", content_diff);
    printf("  rotated lanes           : max rel %.3e  %s\n", worst_rot,
           worst_rot <= 2e-2 ? "ok" : "FAIL");
    bad |= (content_diff != 0) || (worst_rot > 2e-2);

    /* ---- inverse undoes forward --------------------------------------- */
    run(true, false, d_o2, d_o);
    std::vector<bf16> back(x.size());
    HC_CHECK(hipMemcpy(back.data(), d_o2, back.size() * 2, hipMemcpyDeviceToHost));
    /* Relative to the RMS of the tensor, as every other V4 gate measures. A
     * per-element denominator is wrong here: `x` is uniform on [-1,1], so the
     * elements nearest zero make `|a-b|/|a|` explode on two bf16 round trips
     * and report 2.4e-1 for a rotation that is correct. */
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        const double a = bf2f_host(x[i]), b = bf2f_host(back[i]);
        num = std::max(num, fabs(a - b));
        den += a * a;
    }
    const double worst_rt = num / (sqrt(den / (double)x.size()) + 1e-9);
    printf("  inverse(forward(x)) == x: max rel %.3e  %s\n", worst_rt,
           worst_rt <= 3e-2 ? "ok" : "FAIL");
    bad |= worst_rt > 3e-2;

    /* ---- negative control: half-split table over the same lanes -------- */
    run(false, true, d_o2, d_x);
    HC_CHECK(hipMemcpy(back.data(), d_o2, back.size() * 2, hipMemcpyDeviceToHost));
    double worst_ns = 0.0;
    for (size_t i = 0; i < x.size(); i++)
        worst_ns = std::max(worst_ns, fabs(bf2f_host(got[i]) - bf2f_host(back[i])));
    printf("  control (wrong pairing) : differs by %.3f  %s\n", worst_ns,
           worst_ns > 1e-2 ? "ok (caught)" : "FAIL (gate is blind)");
    bad |= worst_ns <= 1e-2;

    printf("%s\n", bad ? "FAIL"
                       : "PASS — V4 RoPE rides the existing HD=512 interleaved arm");
    hipFree(d_x);
    hipFree(d_o);
    hipFree(d_o2);
    hipFree(d_p);
    return bad;
}

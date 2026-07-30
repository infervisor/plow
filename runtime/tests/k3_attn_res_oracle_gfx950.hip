/* k3_attn_res_oracle_gfx950.hip — `d_attn_res` against a HOST TRANSCRIPTION of `_apply_attn_res`.
 *
 * WHY THIS EXISTS. Every AttnRes gate in the tree so far compares the op to ITSELF:
 * `k3_fuse_exact_gfx950.hip` sweeps nb 0..8 but only asserts fused == unfused, and the three real
 * -weight rung gates (`k3_block`, `k3_moe_block`, `k3_mla_block`) pin layers 0, 1 and 3 — where
 * `nb` is 0 or 1 and NEVER more. A 93-layer K3 emits `nb` up to 8:
 *
 *     nb=0 x1   nb=1..7 x24 each   nb=8 x18       (187 sites, `plowrt disasm --program 1`)
 *
 * so **170 of the 187 AttnRes invocations in the shipped model run at an `nb` no gate has ever
 * checked against a reference.** That is the surface this file closes, and it is depth-only: a
 * `K3_NLAYERS<=12` asset cannot reach it, which is exactly why the bring-up looked healthy at 5
 * layers and collapsed at 93.
 *
 * THE REFERENCE, transcribed from `modeling_kimi_linear.py:1075` (`_apply_attn_res`):
 *
 *     v        = cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)   # [T, nb+1, H]
 *     variance = v.float().pow(2).mean(-1, keepdim=True)
 *     k        = v.float() * rsqrt(variance + eps)
 *     scores   = (k * score_weight).sum(-1)                              # score_weight already folded
 *     probs    = scores.softmax(-1)
 *     out      = probs @ v.float()                                       # the RAW rows, not k
 *
 * Two things this pins that a code read does not: the mix is over the RAW `v`, and `variance` is
 * RMSNorm's `mean(x^2)`, not a mean-centred one.
 *
 * WHAT ELSE IS SWEPT, because each is its own untested axis:
 *   * `NBCAP != NB` — the ring strides by the CAPACITY. At T=1 the two are indistinguishable
 *     (the token index is 0 so the stride never multiplies), so only a T>1 case can see it.
 *   * `T > 1` with several workgroups, which partition the TOKENS.
 *   * the SNAPSHOT PUSH, whose row must land one past the live count and must not be mixed in.
 *   * the fused post-norm, against `rmsnorm(mix)` computed on the host from the bf16-rounded mix.
 *
 * TOLERANCE. The output is bf16, so the floor is ~4e-3 relative; the host reduces in double and
 * the device in f32 across 8 waves, so the orders differ. The gate is on the max elementwise
 * difference relative to the row's RMS — 2e-2, the same `TOL` the three K3 rung gates use.
 *
 *   /opt/rocm/bin/hipcc --offload-arch=gfx950 -O3 -w -DPLOW_K3=1 \
 *       -I runtime/amd -I runtime/common runtime/tests/k3_attn_res_oracle_gfx950.hip -o /tmp/k3aro
 *   /tmp/k3aro
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amd_common.h"
#include "op_k3.h"

#define HC(x)                                                                                     \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_) {                                                                                 \
            printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__);                              \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

enum { LDS_BYTES = 147464 };

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void k_ar(
    bf16* dst, const bf16* prefix, bf16* blkres, const float* score_w, const bf16* gamma,
    unsigned T, unsigned HID, unsigned NB, unsigned NBCAP, float eps, const bf16* push_src,
    unsigned push_row) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_attn_res(dst, prefix, blkres, score_w, T, HID, NB, NBCAP, eps, blockIdx.x, gridDim.x,
               (float*)raw, push_src, push_row, gamma);
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

/* `_apply_attn_res` for one token, in double. `blk` is that token's slice of the ring. */
static void ref_token(std::vector<float>& out, const bf16* blk, const bf16* prefix,
                      const float* sw, const bf16* gamma, unsigned HID, unsigned NB, float eps) {
    std::vector<double> s(NB + 1);
    for (unsigned r = 0; r <= NB; r++) {
        const bf16* v = (r < NB) ? blk + (size_t)r * HID : prefix;
        double ss = 0.0, dot = 0.0;
        for (unsigned d = 0; d < HID; d++) {
            const double x = bf2f_host(v[d]);
            ss += x * x;
            dot += x * (double)sw[d];
        }
        s[r] = dot / std::sqrt(ss / (double)HID + (double)eps);
    }
    double m = s[0];
    for (unsigned r = 1; r <= NB; r++) m = s[r] > m ? s[r] : m;
    double z = 0.0;
    for (unsigned r = 0; r <= NB; r++) {
        s[r] = std::exp(s[r] - m);
        z += s[r];
    }
    for (unsigned r = 0; r <= NB; r++) s[r] /= z;

    out.assign(HID, 0.0f);
    for (unsigned d = 0; d < HID; d++) {
        double a = 0.0;
        for (unsigned r = 0; r <= NB; r++) {
            const bf16* v = (r < NB) ? blk + (size_t)r * HID : prefix;
            a += s[r] * (double)bf2f_host(v[d]);
        }
        /* The device stores the mix as bf16 BEFORE the fused norm reduces over it, so the host
         * must round here too or the two norms see different inputs. */
        out[d] = bf2f_host(f2bf_host((float)a));
    }
    if (gamma) {
        double ss = 0.0;
        for (unsigned d = 0; d < HID; d++) ss += (double)out[d] * (double)out[d];
        const double inv = 1.0 / std::sqrt(ss / (double)HID + (double)eps);
        for (unsigned d = 0; d < HID; d++)
            out[d] = (float)((double)out[d] * inv * (double)bf2f_host(gamma[d]));
    }
}

int main(void) {
    srand48(0xA77E5);
    const unsigned HID = 7168, NBCAP = 8;
    const float eps = 1e-5f;
    const float TOL = 2e-2f;
    int bad = 0;

    for (unsigned T : {1u, 3u}) {
        for (unsigned NB = 0; NB <= 8; NB++) {
            for (int push : {0, 1}) {
                for (int fuse : {0, 1}) {
                    /* A push needs a row past the live count, so `nb == NBCAP` cannot also push —
                     * the same refusal `emit_attn_res` asserts. */
                    if (push && NB >= NBCAP) continue;

                    std::vector<bf16> h_prefix((size_t)T * HID), h_gamma(HID),
                        h_blk((size_t)T * NBCAP * HID), h_push((size_t)T * HID);
                    std::vector<float> h_sw(HID);
                    for (unsigned i = 0; i < HID; i++) {
                        h_gamma[i] = f2bf_host(0.5f + 0.5f * frand());
                        h_sw[i] = frand();
                    }
                    /* Rows at deliberately DIFFERENT magnitudes: the softmax is over RMS-normalised
                     * scores but mixes the raw rows, so a scale bug hides when every row matches. */
                    for (size_t i = 0; i < h_prefix.size(); i++) h_prefix[i] = f2bf_host(frand());
                    for (size_t i = 0; i < h_push.size(); i++) h_push[i] = f2bf_host(frand());
                    for (unsigned t = 0; t < T; t++)
                        for (unsigned r = 0; r < NBCAP; r++)
                            for (unsigned d = 0; d < HID; d++)
                                h_blk[((size_t)t * NBCAP + r) * HID + d] =
                                    f2bf_host(frand() * (0.25f * (float)(r + 1)));

                    bf16 *d_prefix, *d_blk, *d_gamma, *d_out, *d_push;
                    float* d_sw;
                    HC(hipMalloc(&d_prefix, h_prefix.size() * 2));
                    HC(hipMalloc(&d_push, h_push.size() * 2));
                    HC(hipMalloc(&d_gamma, HID * 2));
                    HC(hipMalloc(&d_blk, h_blk.size() * 2));
                    HC(hipMalloc(&d_sw, HID * 4));
                    HC(hipMalloc(&d_out, h_prefix.size() * 2));
                    HC(hipMemcpy(d_prefix, h_prefix.data(), h_prefix.size() * 2,
                                 hipMemcpyHostToDevice));
                    HC(hipMemcpy(d_push, h_push.data(), h_push.size() * 2, hipMemcpyHostToDevice));
                    HC(hipMemcpy(d_gamma, h_gamma.data(), HID * 2, hipMemcpyHostToDevice));
                    HC(hipMemcpy(d_blk, h_blk.data(), h_blk.size() * 2, hipMemcpyHostToDevice));
                    HC(hipMemcpy(d_sw, h_sw.data(), HID * 4, hipMemcpyHostToDevice));
                    HC(hipMemset(d_out, 0, h_prefix.size() * 2));

                    const unsigned nblk = T > 1 ? 2u : 1u;
                    hipLaunchKernelGGL(k_ar, dim3(nblk), dim3(PLOW_THREADS), 0, 0, d_out, d_prefix,
                                       d_blk, d_sw, fuse ? d_gamma : nullptr, T, HID, NB, NBCAP,
                                       eps, push ? d_push : nullptr, NB);
                    HC(hipDeviceSynchronize());

                    std::vector<bf16> got(h_prefix.size()), ring(h_blk.size());
                    HC(hipMemcpy(got.data(), d_out, got.size() * 2, hipMemcpyDeviceToHost));
                    HC(hipMemcpy(ring.data(), d_blk, ring.size() * 2, hipMemcpyDeviceToHost));

                    float worst = 0.0f;
                    for (unsigned t = 0; t < T; t++) {
                        std::vector<float> want;
                        ref_token(want, h_blk.data() + (size_t)t * NBCAP * HID,
                                  h_prefix.data() + (size_t)t * HID, h_sw.data(),
                                  fuse ? h_gamma.data() : nullptr, HID, NB, eps);
                        double rms = 0.0;
                        for (unsigned d = 0; d < HID; d++) rms += (double)want[d] * want[d];
                        rms = std::sqrt(rms / (double)HID) + 1e-6;
                        for (unsigned d = 0; d < HID; d++) {
                            const float g = bf2f_host(got[(size_t)t * HID + d]);
                            const float e = (float)(std::fabs(g - want[d]) / rms);
                            worst = e > worst ? e : worst;
                        }
                    }
                    /* The push must land on row NB of EVERY token's own slice, and the mix above
                     * must not have seen it (rows [0, NB) are what `ref_token` read). */
                    int push_ok = 1;
                    if (push)
                        for (unsigned t = 0; t < T && push_ok; t++)
                            for (unsigned d = 0; d < HID; d++)
                                if (ring[((size_t)t * NBCAP + NB) * HID + d] !=
                                    h_push[(size_t)t * HID + d]) {
                                    push_ok = 0;
                                    break;
                                }

                    const int ok = worst <= TOL && push_ok;
                    printf("T=%u nb=%u push=%d fuse=%d  worst %.3e  push %s  %s\n", T, NB, push,
                           fuse, worst, push ? (push_ok ? "ok" : "WRONG ROW") : "-",
                           ok ? "PASS" : "FAIL");
                    if (!ok) bad++;

                    HC(hipFree(d_prefix));
                    HC(hipFree(d_push));
                    HC(hipFree(d_gamma));
                    HC(hipFree(d_blk));
                    HC(hipFree(d_sw));
                    HC(hipFree(d_out));
                }
            }
        }
    }
    printf("%s\n", bad ? "FAIL" : "PASS — d_attn_res matches _apply_attn_res at every emitted nb");
    return bad ? 1 : 0;
}

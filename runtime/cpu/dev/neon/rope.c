/* rope.c — HEADNORM_ROPE (NEON): per-(token, head) RMSNorm then RoPE, golden semantics
 * (norm.c g_headnorm_rope), vectorised over hd: two f32 passes, RoPE on f32 chunks, one bf16
 * round on store. */
#include "neon.h"

N_K(n_headnorm_rope) {
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 2);
    const float* cosb = PLOW_CPU_TEN(in, T, 3);
    const float* sinb = PLOW_CPU_TEN(in, T, 4);
    const int32_t* pos = PLOW_CPU_TEN(in, T, 5);
    const uint32_t ntok = in->i[0], nhead = in->i[1], hd = in->i[2], out_row0 = in->i[3];
    const uint32_t skip_norm = in->i[4], n_batch_kv = in->i[6];
    const uint32_t out_stride = in->fj[1].u, kv_mask = in->fj[2].u;
    const float eps = in->fj[0].f;
    const int interleave = in->i[5] == 2u ? 0 : (hd == 64u) || (hd == 128u && in->i[5] == 1u);
    const uint32_t H2 = hd >> 1, total = ntok * nhead;
    /* H2 must be a whole 4-lane chunk. */
    if (hd > 512u || (hd & 7u)) {
        g_headnorm_rope(in, slice, nblk, T, ctx);
        return;
    }
    float v[512] __attribute__((aligned(16)));
    const float32x4_t sgn = {-1.0f, 1.0f, -1.0f, 1.0f};

    for (uint32_t w0 = slice * G_WAVES; w0 < total; w0 += nblk * G_WAVES) {
        for (uint32_t wi = 0; wi < G_WAVES && w0 + wi < total; wi++) {
            const uint32_t w = w0 + wi, t = w / nhead, hh = w % nhead;
            const uint32_t position = pos ? (uint32_t)pos[t] : out_row0 + t;
            const plow_bf16* xr = x + ((size_t)t * nhead + hh) * hd;
            const size_t obase =
                out_stride
                    ? (n_batch_kv != 0
                           ? ((size_t)(t * nhead + hh) * out_stride + (position & kv_mask)) * hd
                           : ((size_t)hh * out_stride + ((out_row0 + t) & kv_mask)) * hd)
                    : ((size_t)(out_row0 + t) * nhead + hh) * hd;

            float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
            for (uint32_t i = 0; i < hd; i += 8) {
                const uint16x8_t u = vld1q_u16(xr + i);
                const float32x4_t lo = n_lo_f32(u), hi = n_hi_f32(u);
                vst1q_f32(v + i, lo);
                vst1q_f32(v + i + 4, hi);
                a0 = vfmaq_f32(a0, lo, lo);
                a1 = vfmaq_f32(a1, hi, hi);
            }
            const float ss = n_hsum(vaddq_f32(a0, a1));
            const float inv = skip_norm ? 1.0f : g_rsqrt(ss / (float)hd + eps);
            const float32x4_t vinv = vdupq_n_f32(inv);
            for (uint32_t i = 0; i < hd; i += 4) {
                float32x4_t q = vmulq_f32(vld1q_f32(v + i), vinv);
                if (gamma) q = vmulq_f32(q, n_load4(gamma + i));
                vst1q_f32(v + i, q);
            }

            plow_bf16* o = out + obase;
            if (!cosb) {
                for (uint32_t i = 0; i < hd; i += 4) n_store4(o + i, vld1q_f32(v + i));
                continue;
            }
            const size_t p = (size_t)position * H2;
            if (!interleave) {
                for (uint32_t i = 0; i < H2; i += 4) {
                    const float32x4_t c = vld1q_f32(cosb + p + i), s = vld1q_f32(sinb + p + i);
                    const float32x4_t lo = vld1q_f32(v + i), hi = vld1q_f32(v + i + H2);
                    n_store4(o + i, vfmsq_f32(vmulq_f32(lo, c), hi, s));
                    n_store4(o + i + H2, vfmaq_f32(vmulq_f32(hi, c), lo, s));
                }
            } else {
                for (uint32_t i = 0; i < hd; i += 4) {
                    const float32x2_t c2 = vld1_f32(cosb + p + (i >> 1)), s2 = vld1_f32(sinb + p + (i >> 1));
                    const float32x2x2_t cz = vzip_f32(c2, c2), sz = vzip_f32(s2, s2);
                    const float32x4_t c = vcombine_f32(cz.val[0], cz.val[1]);
                    const float32x4_t s = vcombine_f32(sz.val[0], sz.val[1]);
                    const float32x4_t q = vld1q_f32(v + i);
                    const float32x4_t partner = vrev64q_f32(q);
                    n_store4(o + i, vfmaq_f32(vmulq_f32(q, c), vmulq_f32(partner, s), sgn));
                }
            }
        }
    }
}

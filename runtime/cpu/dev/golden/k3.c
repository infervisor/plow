/* Kimi-K3 attention-residual mix (op_k3.h d_attn_res, single-workgroup arm).
 * t0=out t1=prefix t2=blkres(ring [T][NBCAP][HID]) t3=score_w(f32[HID]) t4=push_src? t5=gamma?
 * i0=T i1=HID i2=NB i3=push_row i4=NBCAP  f0=eps.  Rows r<NB are ring rows, r==NB is prefix:
 *   score_r = sum(x_r * w) * rsqrt(mean(x_r^2) + eps) ; p = softmax(score) ; out = bf16(sum p_r x_r)
 * then, with gamma, out = RMSNorm(out, gamma) over the ROUNDED mix. Malformed geometry poisons. */
#include "golden.h"

#define ATTNRES_MAXB 16u

G_K(g_attn_res) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* prefix = PLOW_CPU_TEN(in, T, 1);
    plow_bf16* blkres = PLOW_CPU_TEN(in, T, 2);
    const float* score_w = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* push_src = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 5);
    const uint32_t Tn = in->i[0], HID = in->i[1], NB = in->i[2], push_row = in->i[3];
    const uint32_t NBCAP = in->i[4];
    const float eps = in->fj[0].f;
    if (NB > ATTNRES_MAXB || NBCAP < NB || (push_src && push_row >= NBCAP)) {
        for (uint32_t t = slice; t < Tn; t += nblk) g_poison_row(out + (size_t)t * HID, HID);
        return;
    }
    float sco[ATTNRES_MAXB + 1];
    for (uint32_t t = slice; t < Tn; t += nblk) {
        const size_t pofs = (size_t)t * HID, bofs = (size_t)t * NBCAP * HID;
        if (push_src) memcpy(blkres + bofs + (size_t)push_row * HID, push_src + pofs, (size_t)HID * 2);
        float m = -INFINITY;
        for (uint32_t r = 0; r <= NB; r++) {
            const plow_bf16* vr = r < NB ? blkres + bofs + (size_t)r * HID : prefix + pofs;
            float ss = 0.0f, sw = 0.0f;
            for (uint32_t d = 0; d < HID; d++) {
                const float x = plow_bf2f(vr[d]);
                ss += x * x;
                sw += x * score_w[d];
            }
            sco[r] = sw * g_rsqrt(ss / (float)HID + eps);
            m = m > sco[r] ? m : sco[r];
        }
        float z = 0.0f;
        for (uint32_t r = 0; r <= NB; r++) {
            sco[r] = expf(sco[r] - m);
            z += sco[r];
        }
        for (uint32_t r = 0; r <= NB; r++) sco[r] /= z;

        float ss = 0.0f;
        for (uint32_t d = 0; d < HID; d++) {
            float a = 0.0f;
            for (uint32_t r = 0; r < NB; r++) a += sco[r] * plow_bf2f(blkres[bofs + (size_t)r * HID + d]);
            a += sco[NB] * plow_bf2f(prefix[pofs + d]);
            const plow_bf16 ob = plow_f2bf(a);
            out[pofs + d] = ob;
            const float f = plow_bf2f(ob);
            ss += f * f;
        }
        if (gamma) {
            const float inv = g_rsqrt(ss / (float)HID + eps);
            for (uint32_t d = 0; d < HID; d++)
                out[pofs + d] = plow_f2bf(plow_bf2f(out[pofs + d]) * inv * plow_bf2f(gamma[d]));
        }
    }
}

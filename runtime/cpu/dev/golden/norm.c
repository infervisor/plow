/* Norm family: op_norm.h ported 1:1. Rows are the slice axis (row = slice; row += nblk),
 * except HEADNORM_ROPE whose item is a (token, head) packed G_WAVES per workgroup. */
#include "golden.h"

static float row_ss(const plow_bf16* x, uint32_t n) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const float v = plow_bf2f(x[i]);
        ss += v * v;
    }
    return ss;
}

/* t0=out t1=x t2=gamma?  i0=rows i1=feat i2=out_row0  f0=eps.
 * t3/t4 (fused w8a8 activation quant) are not ported: a packet carrying them gets a qNaN row. */
G_K(g_rmsnorm) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 2);
    const int quant = PLOW_CPU_TEN(in, T, 3) != NULL;
    const uint32_t rows = in->i[0], feat = in->i[1], out_row0 = in->i[2];
    const float eps = in->fj[0].f;
    for (uint32_t row = slice; row < rows; row += nblk) {
        const plow_bf16* xr = x + (size_t)row * feat;
        plow_bf16* o = out + (size_t)(out_row0 + row) * feat;
        if (quant) { g_poison_row(o, feat); continue; }
        const float inv = g_rsqrt(row_ss(xr, feat) / (float)feat + eps);
        for (uint32_t i = 0; i < feat; i++) {
            const float g = gamma ? plow_bf2f(gamma[i]) : 1.0f;
            o[i] = plow_f2bf(plow_bf2f(xr[i]) * inv * g);
        }
    }
}

/* t0=rms(f32) t1=x  i0=rows i1=feat  f0=eps */
G_K(g_rowrms) {
    (void)ctx;
    float* rms = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint32_t rows = in->i[0], feat = in->i[1];
    const float eps = in->fj[0].f;
    for (uint32_t row = slice; row < rows; row += nblk)
        rms[row] = g_rsqrt(row_ss(x + (size_t)row * feat, feat) / (float)feat + eps);
}

/* t0=out t1=x t2=gamma t3=beta  i0=rows i1=feat i3=out_row0  f0=eps
 * y = (x - mean) * rsqrt(E[x^2] - mean^2 + eps) * gamma + beta */
G_K(g_layernorm) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* beta = PLOW_CPU_TEN(in, T, 3);
    const uint32_t rows = in->i[0], feat = in->i[1], out_row0 = in->i[3];
    const float eps = in->fj[0].f;
    for (uint32_t row = slice; row < rows; row += nblk) {
        const plow_bf16* xr = x + (size_t)row * feat;
        plow_bf16* o = out + (size_t)(out_row0 + row) * feat;
        float s = 0.0f, ss = 0.0f;
        for (uint32_t i = 0; i < feat; i++) {
            const float v = plow_bf2f(xr[i]);
            s += v;
            ss += v * v;
        }
        const float mean = s / (float)feat, msq = ss / (float)feat;
        const float inv = g_rsqrt(msq - mean * mean + eps);
        for (uint32_t i = 0; i < feat; i++) {
            const float g = gamma ? plow_bf2f(gamma[i]) : 1.0f;
            const float b = beta ? plow_bf2f(beta[i]) : 0.0f;
            o[i] = plow_f2bf((plow_bf2f(xr[i]) - mean) * inv * g + b);
        }
    }
}

/* t0=out t1=x t2=gamma? t3=cos? t4=sin? t5=pos(i32)?
 * i0=ntok i1=nhead i2=hd i3=out_row0 i4=skip_norm i5=interleave(hd==128) i6=n_batch_kv
 * f0=eps  fj1.u=out_stride (0 = plain [ntok][nhead][hd])  fj2.u=kv_mask (sliding ring).
 * Half-split RoPE (i, i+hd/2); interleaved GPT-J pairs when hd==64 or (hd==128 && i5==1). */
G_K(g_headnorm_rope) {
    (void)ctx;
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
    const int interleave = (hd == 64u) || (hd == 128u && in->i[5] == 1u);
    const uint32_t H2 = hd >> 1, total = ntok * nhead;
    if (hd > 512u) return;
    float v[512], r[512];

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

            float ss = 0.0f;
            for (uint32_t i = 0; i < hd; i++) {
                v[i] = plow_bf2f(xr[i]);
                ss += v[i] * v[i];
            }
            const float inv = skip_norm ? 1.0f : g_rsqrt(ss / (float)hd + eps);
            for (uint32_t i = 0; i < hd; i++) v[i] = v[i] * inv * (gamma ? plow_bf2f(gamma[i]) : 1.0f);

            if (cosb) {
                const size_t p = (size_t)position * H2;
                for (uint32_t i = 0; i < hd; i++) {
                    if (!interleave) {
                        const uint32_t j = i < H2 ? i : i - H2;
                        const float c = cosb[p + j], s = sinb[p + j];
                        r[i] = i < H2 ? v[i] * c - v[i + H2] * s : v[i] * c + v[i - H2] * s;
                    } else {
                        const float c = cosb[p + (i >> 1)], s = sinb[p + (i >> 1)];
                        const float partner = v[i ^ 1u];
                        r[i] = (i & 1u) == 0u ? v[i] * c - partner * s : v[i] * c + partner * s;
                    }
                }
                for (uint32_t i = 0; i < hd; i++) v[i] = r[i];
            }
            for (uint32_t i = 0; i < hd; i++) out[obase + i] = plow_f2bf(v[i]);
        }
    }
}

/* t0=out t1=a t2=b t3=gamma?  i0=rows i1=feat  f0=eps f1=scale
 * out = (a + RMSNorm(b, gamma)) * scale */
G_K(g_norm_residual) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* a = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* b = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 3);
    const uint32_t rows = in->i[0], feat = in->i[1];
    const float eps = in->fj[0].f, scale = in->fj[1].f;
    for (uint32_t row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        const float inv = g_rsqrt(row_ss(b + base, feat) / (float)feat + eps);
        for (uint32_t i = 0; i < feat; i++) {
            const float g = gamma ? plow_bf2f(gamma[i]) : 1.0f;
            out[base + i] =
                plow_f2bf((plow_bf2f(a[base + i]) + plow_bf2f(b[base + i]) * inv * g) * scale);
        }
    }
}

/* t0=out t1=resid t2=a t3=b t4=gamma?  i0=rows i1=feat  f0=eps
 * resid = a + b ; out = RMSNorm(resid, gamma).  resid may alias a. */
G_K(g_add_norm) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    plow_bf16* resid = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* a = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* b = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 4);
    const uint32_t rows = in->i[0], feat = in->i[1];
    const float eps = in->fj[0].f;
    for (uint32_t row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        float ss = 0.0f;
        for (uint32_t i = 0; i < feat; i++) {
            const float f = plow_bf2f(a[base + i]) + plow_bf2f(b[base + i]);
            ss += f * f;
        }
        const float inv = g_rsqrt(ss / (float)feat + eps);
        for (uint32_t i = 0; i < feat; i++) {
            const float g = gamma ? plow_bf2f(gamma[i]) : 1.0f;
            const float f = plow_bf2f(a[base + i]) + plow_bf2f(b[base + i]);
            resid[base + i] = plow_f2bf(f);
            out[base + i] = plow_f2bf(f * inv * g);
        }
    }
}

/* t0=out t1=resid t2=a t3=b t4=gamma_b? t5=gamma_n?  i0=rows i1=feat  f0=eps f1=scale
 * resid = bf16((a + RMSNorm(b, gb)) * scale) ; out = RMSNorm(resid, gn).  resid may alias a. */
G_K(g_norm_residual_norm) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    plow_bf16* resid = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* a = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* b = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* gb = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* gn = PLOW_CPU_TEN(in, T, 5);
    const uint32_t rows = in->i[0], feat = in->i[1];
    const float eps = in->fj[0].f, scale = in->fj[1].f;
    for (uint32_t row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        const float invb = g_rsqrt(row_ss(b + base, feat) / (float)feat + eps);
        float ssr = 0.0f;
        for (uint32_t i = 0; i < feat; i++) {
            const float g = gb ? plow_bf2f(gb[i]) : 1.0f;
            const plow_bf16 rb =
                plow_f2bf((plow_bf2f(a[base + i]) + plow_bf2f(b[base + i]) * invb * g) * scale);
            resid[base + i] = rb;
            const float rf = plow_bf2f(rb);
            ssr += rf * rf;
        }
        const float invr = g_rsqrt(ssr / (float)feat + eps);
        for (uint32_t i = 0; i < feat; i++) {
            const float g = gn ? plow_bf2f(gn[i]) : 1.0f;
            out[base + i] = plow_f2bf(plow_bf2f(resid[base + i]) * invr * g);
        }
    }
}

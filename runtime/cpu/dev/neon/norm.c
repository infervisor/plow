/* Norm family, NEON: golden/norm.c semantics. Two L1-resident passes per row, f32
 * accumulation, one bf16 round per store. */
#include "neon.h"

/* Sum of squares of (a + b) over n. */
static float row_ss_sum(const plow_bf16* a, const plow_bf16* b, uint32_t n) {
    float32x4_t acc = vdupq_n_f32(0);
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t f = vaddq_f32(n_load4(a + i), n_load4(b + i));
        acc = vfmaq_f32(acc, f, f);
    }
    float s = n_hsum(acc);
    for (; i < n; i++) {
        const float f = plow_bf2f(a[i]) + plow_bf2f(b[i]);
        s += f * f;
    }
    return s;
}

/* t0=out t1=x t2=gamma?  i0=rows i1=feat i2=out_row0  f0=eps */
N_K(n_rmsnorm) {
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
        n_scale_row(o, xr, gamma, g_rsqrt(n_row_ss(xr, feat) / (float)feat + eps), feat);
    }
}

/* t0=rms(f32) t1=x  i0=rows i1=feat  f0=eps */
N_K(n_rowrms) {
    (void)ctx;
    float* rms = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint32_t rows = in->i[0], feat = in->i[1];
    const float eps = in->fj[0].f;
    for (uint32_t row = slice; row < rows; row += nblk)
        rms[row] = g_rsqrt(n_row_ss(x + (size_t)row * feat, feat) / (float)feat + eps);
}

/* t0=out t1=a t2=b t3=gamma?  i0=rows i1=feat  f0=eps f1=scale
 * out = (a + RMSNorm(b, gamma)) * scale */
N_K(n_norm_residual) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* a = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* b = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 3);
    const uint32_t rows = in->i[0], feat = in->i[1];
    const float eps = in->fj[0].f, scale = in->fj[1].f;
    const float32x4_t vs = vdupq_n_f32(scale);
    for (uint32_t row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        const float inv = g_rsqrt(n_row_ss(b + base, feat) / (float)feat + eps);
        const float32x4_t vinv = vdupq_n_f32(inv);
        uint32_t i = 0;
        for (; i + 4 <= feat; i += 4) {
            float32x4_t nb = vmulq_f32(n_load4(b + base + i), vinv);
            if (gamma) nb = vmulq_f32(nb, n_load4(gamma + i));
            n_store4(out + base + i, vmulq_f32(vaddq_f32(n_load4(a + base + i), nb), vs));
        }
        for (; i < feat; i++) {
            const float g = gamma ? plow_bf2f(gamma[i]) : 1.0f;
            out[base + i] =
                plow_f2bf((plow_bf2f(a[base + i]) + plow_bf2f(b[base + i]) * inv * g) * scale);
        }
    }
}

/* t0=out t1=resid t2=a t3=b t4=gamma?  i0=rows i1=feat  f0=eps
 * resid = a + b ; out = RMSNorm(resid, gamma).  resid may alias a. */
N_K(n_add_norm) {
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
        const float inv = g_rsqrt(row_ss_sum(a + base, b + base, feat) / (float)feat + eps);
        const float32x4_t vinv = vdupq_n_f32(inv);
        uint32_t i = 0;
        for (; i + 4 <= feat; i += 4) {
            const float32x4_t f = vaddq_f32(n_load4(a + base + i), n_load4(b + base + i));
            float32x4_t v = vmulq_f32(f, vinv);
            if (gamma) v = vmulq_f32(v, n_load4(gamma + i));
            n_store4(resid + base + i, f);
            n_store4(out + base + i, v);
        }
        for (; i < feat; i++) {
            const float g = gamma ? plow_bf2f(gamma[i]) : 1.0f;
            const float f = plow_bf2f(a[base + i]) + plow_bf2f(b[base + i]);
            resid[base + i] = plow_f2bf(f);
            out[base + i] = plow_f2bf(f * inv * g);
        }
    }
}

/* t0=out t1=resid t2=a t3=b t4=gamma_b? t5=gamma_n?  i0=rows i1=feat  f0=eps f1=scale
 * resid = bf16((a + RMSNorm(b, gb)) * scale) ; out = RMSNorm(resid, gn).  resid may alias a. */
N_K(n_norm_residual_norm) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    plow_bf16* resid = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* a = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* b = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* gb = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* gn = PLOW_CPU_TEN(in, T, 5);
    const uint32_t rows = in->i[0], feat = in->i[1];
    const float eps = in->fj[0].f, scale = in->fj[1].f;
    const float32x4_t vs = vdupq_n_f32(scale);
    for (uint32_t row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        const float invb = g_rsqrt(n_row_ss(b + base, feat) / (float)feat + eps);
        const float32x4_t vinvb = vdupq_n_f32(invb);
        float32x4_t ssr = vdupq_n_f32(0);
        uint32_t i = 0;
        for (; i + 4 <= feat; i += 4) {
            float32x4_t nb = vmulq_f32(n_load4(b + base + i), vinvb);
            if (gb) nb = vmulq_f32(nb, n_load4(gb + i));
            const float32x4_t v = n_round4(vmulq_f32(vaddq_f32(n_load4(a + base + i), nb), vs));
            ssr = vfmaq_f32(ssr, v, v);
            n_store4(resid + base + i, v);
        }
        float ss = n_hsum(ssr);
        for (; i < feat; i++) {
            const float g = gb ? plow_bf2f(gb[i]) : 1.0f;
            const plow_bf16 rb =
                plow_f2bf((plow_bf2f(a[base + i]) + plow_bf2f(b[base + i]) * invb * g) * scale);
            resid[base + i] = rb;
            const float rf = plow_bf2f(rb);
            ss += rf * rf;
        }
        n_scale_row(out + base, resid + base, gn, g_rsqrt(ss / (float)feat + eps), feat);
    }
}

/* t0=out t1=x t2=gamma t3=beta  i0=rows i1=feat i3=out_row0  f0=eps
 * y = (x - mean) * rsqrt(E[x^2] - mean^2 + eps) * gamma + beta */
N_K(n_layernorm) {
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
        float32x4_t vs = vdupq_n_f32(0), vss = vdupq_n_f32(0);
        uint32_t i = 0;
        for (; i + 4 <= feat; i += 4) {
            const float32x4_t v = n_load4(xr + i);
            vs = vaddq_f32(vs, v);
            vss = vfmaq_f32(vss, v, v);
        }
        float s = n_hsum(vs), ss = n_hsum(vss);
        for (; i < feat; i++) {
            const float v = plow_bf2f(xr[i]);
            s += v;
            ss += v * v;
        }
        const float mean = s / (float)feat, msq = ss / (float)feat;
        const float inv = g_rsqrt(msq - mean * mean + eps);
        const float32x4_t vm = vdupq_n_f32(mean), vi = vdupq_n_f32(inv);
        i = 0;
        for (; i + 4 <= feat; i += 4) {
            float32x4_t v = vmulq_f32(vsubq_f32(n_load4(xr + i), vm), vi);
            if (gamma) v = vmulq_f32(v, n_load4(gamma + i));
            if (beta) v = vaddq_f32(v, n_load4(beta + i));
            n_store4(o + i, v);
        }
        for (; i < feat; i++) {
            const float g = gamma ? plow_bf2f(gamma[i]) : 1.0f;
            const float b = beta ? plow_bf2f(beta[i]) : 0.0f;
            o[i] = plow_f2bf((plow_bf2f(xr[i]) - mean) * inv * g + b);
        }
    }
}

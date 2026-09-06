/* Norm family, AVX-512: golden/norm.c semantics. Two L1-resident passes per row, f32
 * accumulation, one bf16 round per store (spec §5 RMSNorm pattern). */
#include "avx512.h"

static inline float hsum(__m512 v) { return _mm512_reduce_add_ps(v); }
#define row_ss v_row_ss
#define scale_row v_scale_row

/* Sum of squares of (a + b) over n. */
static float row_ss_sum(const plow_bf16* a, const plow_bf16* b, uint32_t n) {
    __m512 acc = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 f = _mm512_add_ps(v_load_bf16(a + i), v_load_bf16(b + i));
        acc = _mm512_fmadd_ps(f, f, acc);
    }
    if (i < n) {
        const __mmask16 m = v_tail16(n - i);
        const __m512 f = _mm512_add_ps(v_load_bf16_mask(a + i, m), v_load_bf16_mask(b + i, m));
        acc = _mm512_fmadd_ps(f, f, acc);
    }
    return hsum(acc);
}

/* resid = a + b (bf16) and out = bf16(resid * inv * gamma?), one pass. See v_scale_row_g on the
 * unmasked-bulk / template-gamma shape. */
static inline __attribute__((always_inline)) void add_norm_row(plow_bf16* out, plow_bf16* resid,
                                                               const plow_bf16* a, const plow_bf16* b,
                                                               const plow_bf16* gamma, __m512 vinv,
                                                               uint32_t n, const int HAS_G) {
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 f = _mm512_add_ps(v_load_bf16(a + i), v_load_bf16(b + i));
        v_store_bf16(resid + i, f);
        __m512 v = _mm512_mul_ps(f, vinv);
        if (HAS_G) v = _mm512_mul_ps(v, v_load_bf16(gamma + i));
        v_store_bf16(out + i, v);
    }
    if (i < n) {
        const __mmask16 m = v_tail16(n - i);
        const __m512 f = _mm512_add_ps(v_load_bf16_mask(a + i, m), v_load_bf16_mask(b + i, m));
        v_store_bf16_mask(resid + i, m, f);
        __m512 v = _mm512_mul_ps(f, vinv);
        if (HAS_G) v = _mm512_mul_ps(v, v_load_bf16_mask(gamma + i, m));
        v_store_bf16_mask(out + i, m, v);
    }
}

/* out = bf16((a + RMSNorm(b, gamma)) * scale); rb? also takes the bf16-rounded value and its
 * sum of squares (the NORM_RESIDUAL_NORM first pass). */
static inline __attribute__((always_inline)) __m512 norm_resid_row(plow_bf16* out, const plow_bf16* a,
                                                                   const plow_bf16* b, const plow_bf16* gamma,
                                                                   __m512 vinv, __m512 scale, uint32_t n,
                                                                   const int HAS_G, const int ROUND) {
    __m512 ssr = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 nb = _mm512_mul_ps(v_load_bf16(b + i), vinv);
        if (HAS_G) nb = _mm512_mul_ps(nb, v_load_bf16(gamma + i));
        __m512 v = _mm512_mul_ps(_mm512_add_ps(v_load_bf16(a + i), nb), scale);
        if (ROUND) {
            v = v_round_bf16(v);
            ssr = _mm512_fmadd_ps(v, v, ssr);
        }
        v_store_bf16(out + i, v);
    }
    if (i < n) {
        const __mmask16 m = v_tail16(n - i);
        __m512 nb = _mm512_mul_ps(v_load_bf16_mask(b + i, m), vinv);
        if (HAS_G) nb = _mm512_mul_ps(nb, v_load_bf16_mask(gamma + i, m));
        __m512 v = _mm512_mul_ps(_mm512_add_ps(v_load_bf16_mask(a + i, m), nb), scale);
        if (ROUND) {
            v = _mm512_maskz_mov_ps(m, v_round_bf16(v));
            ssr = _mm512_fmadd_ps(v, v, ssr);
        }
        v_store_bf16_mask(out + i, m, v);
    }
    return ssr;
}

/* t0=out t1=x t2=gamma?  i0=rows i1=feat i2=out_row0  f0=eps (t3 quant fold -> qNaN row). */
V_K(v_rmsnorm) {
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
        scale_row(o, xr, gamma, g_rsqrt(row_ss(xr, feat) / (float)feat + eps), feat);
    }
}

V_K(v_rowrms) {
    (void)ctx;
    float* rms = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint32_t rows = in->i[0], feat = in->i[1];
    const float eps = in->fj[0].f;
    for (uint32_t row = slice; row < rows; row += nblk)
        rms[row] = g_rsqrt(row_ss(x + (size_t)row * feat, feat) / (float)feat + eps);
}

/* y = (x - mean) * rsqrt(E[x^2] - mean^2 + eps) * gamma + beta */
V_K(v_layernorm) {
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
        __m512 s = _mm512_setzero_ps(), ss = _mm512_setzero_ps();
        for (uint32_t i = 0; i < feat; i += 16) {
            const __mmask16 m = i + 16 <= feat ? 0xFFFF : v_tail16(feat - i);
            const __m512 v = v_load_bf16_mask(xr + i, m);
            s = _mm512_add_ps(s, v);
            ss = _mm512_fmadd_ps(v, v, ss);
        }
        const float mean = hsum(s) / (float)feat, msq = hsum(ss) / (float)feat;
        const __m512 vmean = _mm512_set1_ps(mean);
        const __m512 vinv = _mm512_set1_ps(g_rsqrt(msq - mean * mean + eps));
        for (uint32_t i = 0; i < feat; i += 16) {
            const __mmask16 m = i + 16 <= feat ? 0xFFFF : v_tail16(feat - i);
            __m512 v = _mm512_mul_ps(_mm512_sub_ps(v_load_bf16_mask(xr + i, m), vmean), vinv);
            if (gamma) v = _mm512_mul_ps(v, v_load_bf16_mask(gamma + i, m));
            if (beta) v = _mm512_add_ps(v, v_load_bf16_mask(beta + i, m));
            v_store_bf16_mask(o + i, m, v);
        }
    }
}

/* out = (a + RMSNorm(b, gamma)) * scale */
V_K(v_norm_residual) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* a = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* b = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 3);
    const uint32_t rows = in->i[0], feat = in->i[1];
    const float eps = in->fj[0].f;
    const __m512 scale = _mm512_set1_ps(in->fj[1].f);
    for (uint32_t row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        const __m512 vinv = _mm512_set1_ps(g_rsqrt(row_ss(b + base, feat) / (float)feat + eps));
        if (gamma)
            norm_resid_row(out + base, a + base, b + base, gamma, vinv, scale, feat, 1, 0);
        else
            norm_resid_row(out + base, a + base, b + base, NULL, vinv, scale, feat, 0, 0);
    }
}

/* resid = a + b ; out = RMSNorm(resid, gamma). resid may alias a: read a before writing. */
V_K(v_add_norm) {
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
        const __m512 vinv =
            _mm512_set1_ps(g_rsqrt(row_ss_sum(a + base, b + base, feat) / (float)feat + eps));
        if (gamma)
            add_norm_row(out + base, resid + base, a + base, b + base, gamma, vinv, feat, 1);
        else
            add_norm_row(out + base, resid + base, a + base, b + base, NULL, vinv, feat, 0);
    }
}

/* resid = bf16((a + RMSNorm(b, gb)) * scale) ; out = RMSNorm(resid, gn). resid may alias a. */
V_K(v_norm_residual_norm) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    plow_bf16* resid = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* a = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* b = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* gb = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* gn = PLOW_CPU_TEN(in, T, 5);
    const uint32_t rows = in->i[0], feat = in->i[1];
    const float eps = in->fj[0].f;
    const __m512 scale = _mm512_set1_ps(in->fj[1].f);
    for (uint32_t row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        const __m512 vinvb = _mm512_set1_ps(g_rsqrt(row_ss(b + base, feat) / (float)feat + eps));
        const __m512 ssr = gb ? norm_resid_row(resid + base, a + base, b + base, gb, vinvb, scale, feat, 1, 1)
                              : norm_resid_row(resid + base, a + base, b + base, NULL, vinvb, scale, feat, 0, 1);
        scale_row(out + base, resid + base, gn, g_rsqrt(hsum(ssr) / (float)feat + eps), feat);
    }
}

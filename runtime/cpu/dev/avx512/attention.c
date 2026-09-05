/* attention.c — FLASH_PREFILL / FLASH_DECODE / FLASH_MERGE (AVX-512).
 *
 * Same work decomposition, KV layout (head-major, ring mask) and partial format as the golden
 * tier (golden/attention.c); the softmax runs blockwise (32 keys prefill, 4 keys decode) in f32
 * instead of per key, which differs from golden in f32 rounding only. QK^T uses vdpbf16ps over
 * 32-element chunks (spec §4.2), PV widens V to f32 and FMAs against broadcast P (spec §4.1/4.2).
 * Decode folds the FA_GF query heads of one work item onto each K/V row load. */
#include "avx512.h"

#define FA_BQ_TILE 128u
#define FA_BKV 32u
#define FA_GF 2u
#define FA_KB 4u /* decode keys per softmax block: FA_KB x FA_GF = 8 dpbf16 chains */

static inline __m512 v_dp(__m512 acc, const plow_bf16* a, const plow_bf16* b) {
    return _mm512_dpbf16_ps(acc, (__m512bh)_mm512_loadu_si512((const void*)a),
                            (__m512bh)_mm512_loadu_si512((const void*)b));
}

/* One query row against keys [j0, j1) of a 32-key tile: S[j] = scale * q . k_j. */
static inline void v_scores(const plow_bf16* q, const plow_bf16* kbase, const uint32_t* krow,
                            uint32_t j0, uint32_t j1, uint32_t D, float scale, float* S) {
    uint32_t j = j0;
    for (; j + 4 <= j1; j += 4) {
        const plow_bf16* k0 = kbase + (size_t)krow[j] * D;
        const plow_bf16* k1 = kbase + (size_t)krow[j + 1] * D;
        const plow_bf16* k2 = kbase + (size_t)krow[j + 2] * D;
        const plow_bf16* k3 = kbase + (size_t)krow[j + 3] * D;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (uint32_t d = 0; d < D; d += 32) {
            a0 = v_dp(a0, k0 + d, q + d);
            a1 = v_dp(a1, k1 + d, q + d);
            a2 = v_dp(a2, k2 + d, q + d);
            a3 = v_dp(a3, k3 + d, q + d);
        }
        S[j] = _mm512_reduce_add_ps(a0) * scale;
        S[j + 1] = _mm512_reduce_add_ps(a1) * scale;
        S[j + 2] = _mm512_reduce_add_ps(a2) * scale;
        S[j + 3] = _mm512_reduce_add_ps(a3) * scale;
    }
    for (; j < j1; j++) {
        const plow_bf16* k0 = kbase + (size_t)krow[j] * D;
        __m512 a0 = _mm512_setzero_ps();
        for (uint32_t d = 0; d < D; d += 32) a0 = v_dp(a0, k0 + d, q + d);
        S[j] = _mm512_reduce_add_ps(a0) * scale;
    }
}

/* acc[d] = acc[d]*corr + sum_j P[j] * V_j[d] over keys [j0, j1). */
static inline void v_pv(float* acc, float corr, const float* P, const plow_bf16* vbase,
                        const uint32_t* vrow, uint32_t j0, uint32_t j1, uint32_t D) {
    const __m512 vc = _mm512_set1_ps(corr);
    for (uint32_t d = 0; d < D; d += 16) {
        __m512 a = _mm512_mul_ps(_mm512_load_ps(acc + d), vc);
        for (uint32_t j = j0; j < j1; j++)
            a = _mm512_fmadd_ps(_mm512_set1_ps(P[j]), v_load_bf16(vbase + (size_t)vrow[j] * D + d), a);
        _mm512_store_ps(acc + d, a);
    }
}

/* t0=Opart t1=mlpart t2=Q t3=K t4=V t5=O_final?
 * i0=n_q i1=n_kv i2=n_head i3=n_kv_head i4=q_pos0 i5=window i6=hd i7=nsplit
 * f0=scale fj1.u=kv_stride fj2.u=kv_mask. */
V_K(v_flash_prefill) {
    float* Opart = PLOW_CPU_TEN(in, T, 0);
    float* mlpart = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* Q = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* K = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* V = PLOW_CPU_TEN(in, T, 4);
    plow_bf16* O_final = PLOW_CPU_TEN(in, T, 5);
    const uint32_t n_q = in->i[0], n_kv = in->i[1], n_head = in->i[2], n_kv_head = in->i[3];
    const uint32_t q_pos0 = in->i[4], window = in->i[5], D = in->i[6];
    const uint32_t nsplit = in->i[7] ? in->i[7] : 1u;
    const float scale = in->fj[0].f;
    const uint32_t kv_stride = in->fj[1].u, kv_mask = in->fj[2].u;
    if (D > 512u || (D & 31u)) {
        g_flash_prefill(in, slice, nblk, T, ctx);
        return;
    }
    const uint32_t gqa = n_head / n_kv_head;
    const uint32_t q_tiles = (n_q + FA_BQ_TILE - 1) / FA_BQ_TILE;
    const uint32_t n_work = q_tiles * n_head * nsplit;
    float acc[512] __attribute__((aligned(64)));
    float S[FA_BKV], P[FA_BKV];
    uint32_t row[FA_BKV];

    for (uint32_t w = slice; w < n_work; w += nblk) {
        const uint32_t sp = w % nsplit, h = (w / nsplit) % n_head, qt = w / (nsplit * n_head);
        const uint32_t hkv = h / gqa;
        const uint32_t q_base = qt * FA_BQ_TILE;
        const uint32_t q_tile_last = q_pos0 + q_base + FA_BQ_TILE - 1;
        const uint32_t kv_end = q_tile_last + 1 < n_kv ? q_tile_last + 1 : n_kv;
        const uint32_t q_tile_first = q_pos0 + q_base;
        const uint32_t win_lo = (window && q_tile_first >= window) ? q_tile_first - window + 1 : 0;
        const uint32_t kv_lo = (win_lo / FA_BKV) * FA_BKV;
        const uint32_t tiles_kv = kv_end > kv_lo ? (kv_end - kv_lo + FA_BKV - 1) / FA_BKV : 0u;
        const uint32_t per = (tiles_kv + nsplit - 1) / nsplit;
        const uint32_t my_lo = kv_lo + sp * per * FA_BKV;
        uint32_t my_hi = kv_lo + (sp + 1) * per * FA_BKV;
        if (my_hi > kv_end) my_hi = kv_end;
        const plow_bf16* kbase = K + (size_t)hkv * kv_stride * D;
        const plow_bf16* vbase = V + (size_t)hkv * kv_stride * D;

        for (uint32_t qi = q_base; qi < q_base + FA_BQ_TILE && qi < n_q; qi++) {
            const plow_bf16* q = Q + ((size_t)qi * n_head + h) * D;
            const uint32_t qg = q_pos0 + qi;
            float m = G_NEG_INF, l = 0.0f;
            memset(acc, 0, sizeof(float) * D);
            for (uint32_t kt = my_lo; kt < my_hi; kt += FA_BKV) {
                const uint32_t nk = my_hi - kt < FA_BKV ? my_hi - kt : FA_BKV;
                /* Valid keys of this row inside the tile: causal end and window start. */
                uint32_t jhi = kt <= qg ? qg - kt + 1 : 0;
                if (jhi > nk) jhi = nk;
                if (kt + jhi > n_kv) jhi = n_kv > kt ? n_kv - kt : 0;
                uint32_t jlo = 0;
                if (window && qg + 1 > window) {
                    const uint32_t w0 = qg + 1 - window;
                    jlo = w0 > kt ? w0 - kt : 0;
                }
                if (jlo >= jhi) continue;
                for (uint32_t j = jlo; j < jhi; j++) row[j] = (kt + j) & kv_mask;
                v_scores(q, kbase, row, jlo, jhi, D, scale, S);
                float bm = S[jlo];
                for (uint32_t j = jlo + 1; j < jhi; j++) bm = bm > S[j] ? bm : S[j];
                const float mnew = m > bm ? m : bm;
                const float corr = m == G_NEG_INF ? 0.0f : expf(m - mnew);
                /* P = bf16(exp(s - m)), the MFMA operand round; l sums the rounded values. */
                float psum = 0.0f;
                for (uint32_t j = jlo; j < jhi; j += 16) {
                    const uint32_t n = jhi - j < 16 ? jhi - j : 16;
                    const __mmask16 msk = v_tail16(n) ? v_tail16(n) : 0xFFFF;
                    const __m512 s = _mm512_maskz_loadu_ps(msk, S + j);
                    __m512 e = v_expf(_mm512_sub_ps(s, _mm512_set1_ps(mnew)));
                    e = _mm512_maskz_mov_ps(msk, v_round_bf16(e));
                    _mm512_mask_storeu_ps(P + j, msk, e);
                    psum += _mm512_reduce_add_ps(e);
                }
                l = l * corr + psum;
                m = mnew;
                v_pv(acc, corr, P, vbase, row, jlo, jhi, D);
            }
            if (nsplit == 1u && O_final) {
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                const __m512 vinv = _mm512_set1_ps(inv);
                plow_bf16* orow = O_final + ((size_t)qi * n_head + h) * D;
                for (uint32_t d = 0; d < D; d += 16)
                    v_store_bf16(orow + d, _mm512_mul_ps(_mm512_load_ps(acc + d), vinv));
                continue;
            }
            float* op = Opart + ((size_t)(qi * n_head + h) * nsplit + sp) * D;
            memcpy(op, acc, sizeof(float) * D);
            float* ml = mlpart + ((size_t)(qi * n_head + h) * nsplit + sp) * 2;
            ml[0] = m;
            ml[1] = l;
        }
    }
}

/* t0=Opart t1=mlpart t2=Q t3=K t4=V t5=kv_len(i32)
 * i0=n_batch i1=n_head i2=n_kv_head i3=kv_stride i4=window i5=nsplit i6=hd i7=kv_mask f0=scale.
 * Keys in [lo, hi) are all causal/window-valid by construction (golden's per-key test is
 * implied by first = len - window and hi <= len), so no mask inside the loop. */
V_K(v_flash_decode) {
    float* Opart = PLOW_CPU_TEN(in, T, 0);
    float* mlpart = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* Q = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* K = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* V = PLOW_CPU_TEN(in, T, 4);
    const int32_t* kv_len = PLOW_CPU_TEN(in, T, 5);
    const int nrf = (in->i[1] & 0x10000u) != 0;
    const uint32_t n_batch = in->i[0];
    const uint32_t n_head = in->i[1] & 0xFFFFu, n_kv_head = in->i[2];
    const uint32_t kv_stride = in->i[3], window = in->i[4], nsplit = in->i[5];
    const uint32_t D = in->i[6] & 0xFFFFu, kv_mask = in->i[7];
    const float scale = in->fj[0].f;
    if (nrf || D > 512u || (D & 31u) || nsplit == 0u) {
        g_flash_decode(in, slice, nblk, T, ctx); /* NRF fold is not ported: golden poisons */
        return;
    }
    const uint32_t gqa = n_head / n_kv_head;
    const uint32_t n_grp = (n_head + FA_GF - 1) / FA_GF;
    const uint32_t n_work = n_batch * n_grp * nsplit;
    float acc[FA_GF][512] __attribute__((aligned(64)));
    float m[FA_GF], l[FA_GF], corr[FA_GF];
    float S[FA_KB][FA_GF], P[FA_KB][FA_GF];

    for (uint32_t w = slice; w < n_work; w += nblk) {
        const uint32_t sp = w % nsplit, hg = (w / nsplit) % n_grp, b = w / (nsplit * n_grp);
        const uint32_t h0 = hg * FA_GF, hkv = h0 / gqa;
        const uint32_t nh = n_head - h0 < FA_GF ? n_head - h0 : FA_GF;
        const uint32_t len = (uint32_t)kv_len[b];
        const uint32_t first = (window && len > window) ? len - window : 0u;
        const uint32_t span = len - first, per = (span + nsplit - 1) / nsplit;
        const uint32_t lo = first + sp * per, hi = lo + per < len ? lo + per : len;
        const plow_bf16* kbase = K + ((size_t)b * n_kv_head + hkv) * kv_stride * D;
        const plow_bf16* vbase = V + ((size_t)b * n_kv_head + hkv) * kv_stride * D;
        const plow_bf16* q[FA_GF];
        for (uint32_t h = 0; h < nh; h++) {
            q[h] = Q + ((size_t)b * n_head + h0 + h) * D;
            m[h] = G_NEG_INF;
            l[h] = 0.0f;
            memset(acc[h], 0, sizeof(float) * D);
        }

        for (uint32_t kv = lo; kv < hi; kv += FA_KB) {
            const uint32_t nk = hi - kv < FA_KB ? hi - kv : FA_KB;
            const plow_bf16* kr[FA_KB];
            const plow_bf16* vr[FA_KB];
            for (uint32_t j = 0; j < nk; j++) {
                kr[j] = kbase + (size_t)((kv + j) & kv_mask) * D;
                vr[j] = vbase + (size_t)((kv + j) & kv_mask) * D;
            }
            if (kv + FA_KB < hi) {
                for (uint32_t j = 0; j < FA_KB; j++) {
                    const size_t r = (size_t)((kv + FA_KB + j) & kv_mask) * D;
                    _mm_prefetch((const char*)(kbase + r), _MM_HINT_T0);
                    _mm_prefetch((const char*)(vbase + r), _MM_HINT_T0);
                }
            }
            /* QK^T: nk x nh independent dpbf16 chains, one K row load per chunk. */
            __m512 a[FA_KB][FA_GF];
            for (uint32_t j = 0; j < nk; j++)
                for (uint32_t h = 0; h < nh; h++) a[j][h] = _mm512_setzero_ps();
            for (uint32_t d = 0; d < D; d += 32) {
                for (uint32_t j = 0; j < nk; j++) {
                    const __m512bh kc = (__m512bh)_mm512_loadu_si512((const void*)(kr[j] + d));
                    for (uint32_t h = 0; h < nh; h++)
                        a[j][h] = _mm512_dpbf16_ps(
                            a[j][h], kc, (__m512bh)_mm512_loadu_si512((const void*)(q[h] + d)));
                }
            }
            for (uint32_t h = 0; h < nh; h++) {
                float bm = G_NEG_INF;
                for (uint32_t j = 0; j < nk; j++) {
                    S[j][h] = _mm512_reduce_add_ps(a[j][h]) * scale;
                    bm = bm > S[j][h] ? bm : S[j][h];
                }
                const float mnew = m[h] > bm ? m[h] : bm;
                corr[h] = m[h] == G_NEG_INF ? 0.0f : expf(m[h] - mnew);
                float ps = 0.0f;
                for (uint32_t j = 0; j < nk; j++) {
                    P[j][h] = expf(S[j][h] - mnew);
                    ps += P[j][h];
                }
                l[h] = l[h] * corr[h] + ps;
                m[h] = mnew;
            }
            /* PV: each V chunk widened once, applied to every head of the group. */
            for (uint32_t d = 0; d < D; d += 16) {
                __m512 vc[FA_KB];
                for (uint32_t j = 0; j < nk; j++) vc[j] = v_load_bf16(vr[j] + d);
                for (uint32_t h = 0; h < nh; h++) {
                    __m512 x = _mm512_mul_ps(_mm512_load_ps(acc[h] + d), _mm512_set1_ps(corr[h]));
                    for (uint32_t j = 0; j < nk; j++) x = _mm512_fmadd_ps(_mm512_set1_ps(P[j][h]), vc[j], x);
                    _mm512_store_ps(acc[h] + d, x);
                }
            }
        }
        for (uint32_t h = 0; h < nh; h++) {
            float* op = Opart + ((size_t)(b * n_head + h0 + h) * nsplit + sp) * D;
            float* ml = mlpart + ((size_t)(b * n_head + h0 + h) * nsplit + sp) * 2;
            memcpy(op, acc[h], sizeof(float) * D);
            ml[0] = m[h];
            ml[1] = l[h];
        }
    }
}

/* t0=O t1=Opart t2=mlpart  i0=n_batch i1=n_head i2=nsplit i3=hd. Work = (row, head, d-chunk),
 * dsplit = ceil(nblk / (n_batch*n_head)) exactly as golden / devgen flash_merge_map. */
V_K(v_flash_merge) {
    (void)ctx;
    plow_bf16* O = PLOW_CPU_TEN(in, T, 0);
    const float* Opart = PLOW_CPU_TEN(in, T, 1);
    const float* mlpart = PLOW_CPU_TEN(in, T, 2);
    const uint32_t n_batch = in->i[0], n_head = in->i[1], nsplit = in->i[2], D = in->i[3];
    const uint32_t n_bh = n_batch * n_head;
    if (n_bh == 0u || nsplit == 0u) return;
    if (nsplit > 64u) {
        g_flash_merge(in, slice, nblk, T, ctx);
        return;
    }
    const uint32_t dsplit = (nblk + n_bh - 1) / n_bh;
    const uint32_t dchunk = (D + dsplit - 1) / dsplit;
    const uint32_t n_work = n_bh * dsplit;
    float wgt[64];
    for (uint32_t w = slice; w < n_work; w += nblk) {
        const uint32_t dp = w % dsplit, hb = w / dsplit;
        const uint32_t d0 = dp * dchunk, d1 = d0 + dchunk < D ? d0 + dchunk : D;
        const float* ml = mlpart + (size_t)hb * nsplit * 2;
        float gm = G_NEG_INF;
        for (uint32_t s = 0; s < nsplit; s++) gm = gm > ml[s * 2] ? gm : ml[s * 2];
        float gl = 0.0f;
        for (uint32_t s = 0; s < nsplit; s++) {
            wgt[s] = ml[s * 2] != G_NEG_INF ? expf(ml[s * 2] - gm) : 0.0f;
            gl += ml[s * 2 + 1] * wgt[s];
        }
        const float inv = gl > 0.0f ? 1.0f / gl : 0.0f;
        const __m512 vinv = _mm512_set1_ps(inv);
        const float* obase = Opart + (size_t)hb * nsplit * D;
        plow_bf16* orow = O + (size_t)hb * D;
        for (uint32_t d = d0; d < d1; d += 16) {
            const uint32_t n = d1 - d < 16 ? d1 - d : 16;
            const __mmask16 msk = n == 16 ? 0xFFFF : v_tail16(n);
            __m512 acc = _mm512_setzero_ps();
            for (uint32_t s = 0; s < nsplit; s++)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(wgt[s]),
                                      _mm512_maskz_loadu_ps(msk, obase + (size_t)s * D + d), acc);
            v_store_bf16_mask(orow + d, msk, _mm512_mul_ps(acc, vinv));
        }
    }
}

void v_register_attention(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_HEADNORM_ROPE] = v_headnorm_rope;
    tab[PLOW_DOP_FLASH_PREFILL] = v_flash_prefill;
    tab[PLOW_DOP_FLASH_DECODE] = v_flash_decode;
    tab[PLOW_DOP_FLASH_MERGE] = v_flash_merge;
}

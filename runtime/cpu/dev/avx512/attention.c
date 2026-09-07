/* attention.c — FLASH_PREFILL / FLASH_DECODE / FLASH_MERGE (AVX-512).
 *
 * Same work decomposition, KV layout (head-major, ring mask) and partial format as the golden
 * tier (golden/attention.c); the softmax runs blockwise (32 keys prefill, 8 keys decode) in f32
 * instead of per key, which differs from golden in f32 rounding only. QK^T and PV both run on
 * vdpbf16ps (spec §4.2): prefill against a transposed K tile with the 32 keys in lanes, decode on
 * register-resident chains; PV interleaves key pairs so P rides in the bf16 pair operand. */
#include "avx512.h"

#define FA_BQ_TILE 128u
#define FA_BKV 32u
#define FA_GF 2u
#define FA_KB 8u /* decode keys per softmax block: FA_KB x FA_GF = 16 dpbf16 chains */
#define FA_NG 4u /* decode head groups folded onto one K/V pass; FA_NG * FA_GF * 512 f32 of acc */

typedef uint32_t __attribute__((may_alias)) v_u32a;

/* PV runs on vdpbf16ps over key-pair-interleaved V rows (vpunpck{l,h}wd of rows j, j+1), so a
 * 32-wide d chunk accumulates as (lo, hi) vectors whose lane 4L+i holds d = 8L+i / 8L+4+i.
 * v_pv_unperm restores natural d order once per output row. */
static inline void v_pv_unperm(float* out, const float* acc) {
    const __m512 lo = _mm512_load_ps(acc), hi = _mm512_load_ps(acc + 16);
    _mm512_storeu_ps(out, _mm512_permutex2var_ps(
                              lo, _mm512_set_epi32(23, 22, 21, 20, 7, 6, 5, 4, 19, 18, 17, 16, 3, 2, 1, 0), hi));
    _mm512_storeu_ps(out + 16,
                     _mm512_permutex2var_ps(
                         lo, _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 27, 26, 25, 24, 11, 10, 9, 8), hi));
}

/* --- prefill --------------------------------------------------------------------------- */

#define FA_RB 4u /* query rows per pass: 16 QK chains, 16 PV chains */
#define FA_KT_U32 (16u * 512u) /* K^T tile: [hd/2][32 keys] u32 (bf16 pairs), 32 KiB at hd 512 */
#define FA_PF_SCRATCH ((size_t)FA_BQ_TILE * 512u * 4u + (size_t)FA_KT_U32 * 4u)

/* 16x16 transpose of 32-bit lanes: r[i] lane j -> r[j] lane i. */
static inline void v_tr16(__m512i* r) {
    __m512i t[16], u[16];
#pragma GCC unroll 8
    for (int i = 0; i < 8; i++) {
        t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
    }
#pragma GCC unroll 4
    for (int g = 0; g < 4; g++) {
        u[4 * g] = _mm512_unpacklo_epi64(t[4 * g], t[4 * g + 2]);
        u[4 * g + 1] = _mm512_unpackhi_epi64(t[4 * g], t[4 * g + 2]);
        u[4 * g + 2] = _mm512_unpacklo_epi64(t[4 * g + 1], t[4 * g + 3]);
        u[4 * g + 3] = _mm512_unpackhi_epi64(t[4 * g + 1], t[4 * g + 3]);
    }
    /* u[4g+k]: rows 4g..4g+3, 128-bit lane L = column 4L+k. */
#pragma GCC unroll 4
    for (int k = 0; k < 4; k++) {
        t[k] = _mm512_shuffle_i32x4(u[k], u[4 + k], 0x88);
        t[4 + k] = _mm512_shuffle_i32x4(u[k], u[4 + k], 0xDD);
        t[8 + k] = _mm512_shuffle_i32x4(u[8 + k], u[12 + k], 0x88);
        t[12 + k] = _mm512_shuffle_i32x4(u[8 + k], u[12 + k], 0xDD);
    }
#pragma GCC unroll 4
    for (int k = 0; k < 4; k++) {
        r[k] = _mm512_shuffle_i32x4(t[k], t[8 + k], 0x88);
        r[8 + k] = _mm512_shuffle_i32x4(t[k], t[8 + k], 0xDD);
        r[4 + k] = _mm512_shuffle_i32x4(t[4 + k], t[12 + k], 0x88);
        r[12 + k] = _mm512_shuffle_i32x4(t[4 + k], t[12 + k], 0xDD);
    }
}

/* Kt[dp][j] = (K_j[2dp], K_j[2dp+1]) for the 32 ring rows of a tile: QK^T then runs as
 * dpbf16(Kt[dp], bcast(Q[2dp..2dp+1])) with the 32 keys in lanes, no horizontal reductions. */
static inline void v_build_kt(uint32_t* Kt, const plow_bf16* kbase, const uint32_t* row, uint32_t D) {
    __m512i r[16];
    for (uint32_t kh = 0; kh < 2; kh++)
        for (uint32_t g = 0; g < D; g += 32) {
#pragma GCC unroll 16
            for (uint32_t i = 0; i < 16; i++)
                r[i] = _mm512_loadu_si512((const void*)(kbase + (size_t)row[kh * 16 + i] * D + g));
            v_tr16(r);
#pragma GCC unroll 16
            for (uint32_t p = 0; p < 16; p++)
                _mm512_store_si512((void*)(Kt + ((size_t)(g / 2 + p)) * 32 + kh * 16), r[p]);
        }
}

/* One V key pair (2jp, 2jp+1) into RB rows' (lo, hi) accumulators. */
static inline __attribute__((always_inline)) void v_pf_pair(__m512 (*X)[2], __m512i lo, __m512i hi,
                                                            const uint32_t (*pp)[16], uint32_t jp,
                                                            const uint32_t RB) {
#pragma GCC unroll 4
    for (uint32_t r = 0; r < RB; r++) {
        const __m512bh pb = (__m512bh)_mm512_set1_epi32((int)pp[r][jp]);
        X[r][0] = _mm512_dpbf16_ps(X[r][0], (__m512bh)lo, pb);
        X[r][1] = _mm512_dpbf16_ps(X[r][1], (__m512bh)hi, pb);
    }
}

/* RB query rows against one 32-key tile (Kt built, ring rows in row[], per-row valid key range
 * [jlo, jhi), possibly empty). Blockwise online softmax; P = bf16(exp(s - m)) (the MFMA operand
 * round), l sums the rounded values. acc is [RB][512] in the pair layout. */
static inline __attribute__((always_inline)) void v_pf_block(
    const plow_bf16* const* q, const uint32_t* Kt, const plow_bf16* vbase, const uint32_t* row,
    const uint32_t* jlo, const uint32_t* jhi, float* acc, float* m, float* l, uint32_t D, float scale,
    const uint32_t RB) {
    __m512 s[FA_RB][4]; /* [row][key half + 2 * d parity] */
#pragma GCC unroll 4
    for (uint32_t r = 0; r < FA_RB; r++)
#pragma GCC unroll 4
        for (uint32_t k = 0; k < 4; k++) s[r][k] = _mm512_setzero_ps();
    for (uint32_t dp = 0; dp < D / 2; dp += 2) {
        const uint32_t* kt = Kt + (size_t)dp * 32;
        const __m512bh k0 = (__m512bh)_mm512_load_si512((const void*)kt);
        const __m512bh k1 = (__m512bh)_mm512_load_si512((const void*)(kt + 16));
        const __m512bh k2 = (__m512bh)_mm512_load_si512((const void*)(kt + 32));
        const __m512bh k3 = (__m512bh)_mm512_load_si512((const void*)(kt + 48));
#pragma GCC unroll 4
        for (uint32_t r = 0; r < RB; r++) {
            const v_u32a* qp = (const v_u32a*)q[r];
            const __m512bh qa = (__m512bh)_mm512_set1_epi32((int)qp[dp]);
            const __m512bh qb = (__m512bh)_mm512_set1_epi32((int)qp[dp + 1]);
            s[r][0] = _mm512_dpbf16_ps(s[r][0], k0, qa);
            s[r][1] = _mm512_dpbf16_ps(s[r][1], k1, qa);
            s[r][2] = _mm512_dpbf16_ps(s[r][2], k2, qb);
            s[r][3] = _mm512_dpbf16_ps(s[r][3], k3, qb);
        }
    }
    const __m512 vscale = _mm512_set1_ps(scale);
    float mnew[FA_RB] = {0}, corr[FA_RB] = {0};
    float cv[16] __attribute__((aligned(64))) = {0};
    __mmask16 m0[FA_RB] = {0}, m1[FA_RB] = {0};
    uint32_t ju_lo = FA_BKV, ju_hi = 0;
#pragma GCC unroll 4
    for (uint32_t r = 0; r < RB; r++) {
        s[r][0] = _mm512_mul_ps(_mm512_add_ps(s[r][0], s[r][2]), vscale);
        s[r][1] = _mm512_mul_ps(_mm512_add_ps(s[r][1], s[r][3]), vscale);
        const uint32_t vb = jlo[r] < jhi[r]
                                ? (jhi[r] >= 32u ? 0xFFFFFFFFu : (1u << jhi[r]) - 1u) & ~((1u << jlo[r]) - 1u)
                                : 0u;
        m0[r] = (__mmask16)vb;
        m1[r] = (__mmask16)(vb >> 16);
        mnew[r] = m[r];
        if (!vb) continue; /* corr = exp(0) = 1, P = 0: the row is untouched by this tile */
        if (jlo[r] < ju_lo) ju_lo = jlo[r];
        if (jhi[r] > ju_hi) ju_hi = jhi[r];
        float bm = _mm512_mask_reduce_max_ps(m0[r], s[r][0]);
        const float bm1 = _mm512_mask_reduce_max_ps(m1[r], s[r][1]);
        bm = bm > bm1 ? bm : bm1;
        mnew[r] = m[r] > bm ? m[r] : bm;
        cv[r] = m[r] - mnew[r];
    }
    _mm512_store_ps(cv, v_expf(_mm512_load_ps(cv)));
    uint32_t pp[FA_RB][16] __attribute__((aligned(64)));
#pragma GCC unroll 4
    for (uint32_t r = 0; r < RB; r++) {
        corr[r] = m[r] == G_NEG_INF ? 0.0f : cv[r];
        m[r] = mnew[r];
        const __m512 mv = _mm512_set1_ps(mnew[r]);
        const __m512 p0 = v_round_bf16(_mm512_maskz_mov_ps(m0[r], v_expf(_mm512_sub_ps(s[r][0], mv))));
        const __m512 p1 = v_round_bf16(_mm512_maskz_mov_ps(m1[r], v_expf(_mm512_sub_ps(s[r][1], mv))));
        l[r] = l[r] * corr[r] + _mm512_reduce_add_ps(_mm512_add_ps(p0, p1));
        _mm512_store_si512((void*)pp[r], (__m512i)_mm512_cvtne2ps_pbh(p1, p0));
    }
    if (ju_lo >= ju_hi) return;
    /* PV over the union of the rows' key ranges; a key outside a row's range has P = 0 there. */
    const uint32_t jp_lo = ju_lo / 2, jp_hi = (ju_hi + 1) / 2;
    for (uint32_t c = 0; c < D; c += 32) {
        __m512 x[FA_RB][2], y[FA_RB][2];
#pragma GCC unroll 4
        for (uint32_t r = 0; r < FA_RB; r++) {
            const __m512 cr = _mm512_set1_ps(corr[r]);
            x[r][0] = _mm512_mul_ps(_mm512_load_ps(acc + r * 512 + c), cr);
            x[r][1] = _mm512_mul_ps(_mm512_load_ps(acc + r * 512 + c + 16), cr);
            y[r][0] = _mm512_setzero_ps();
            y[r][1] = _mm512_setzero_ps();
        }
        uint32_t jp = jp_lo;
        for (; jp < jp_hi; jp++) {
            const uint32_t j1 = 2 * jp + 1 < ju_hi ? 2 * jp + 1 : 2 * jp; /* odd tail: its P is 0 */
            const __m512i va = _mm512_loadu_si512((const void*)(vbase + (size_t)row[2 * jp] * D + c));
            const __m512i vb = _mm512_loadu_si512((const void*)(vbase + (size_t)row[j1] * D + c));
            const __m512i lo = _mm512_unpacklo_epi16(va, vb), hi = _mm512_unpackhi_epi16(va, vb);
            if (jp & 1u) v_pf_pair(y, lo, hi, pp, jp, RB);
            else v_pf_pair(x, lo, hi, pp, jp, RB);
        }
#pragma GCC unroll 4
        for (uint32_t r = 0; r < RB; r++) {
            _mm512_store_ps(acc + r * 512 + c, _mm512_add_ps(x[r][0], y[r][0]));
            _mm512_store_ps(acc + r * 512 + c + 16, _mm512_add_ps(x[r][1], y[r][1]));
        }
    }
}

/* t0=Opart t1=mlpart t2=Q t3=K t4=V t5=O_final?
 * i0=n_q i1=n_kv i2=n_head i3=n_kv_head i4=q_pos0 i5=window i6=hd i7=nsplit
 * f0=scale fj1.u=kv_stride fj2.u=kv_mask.
 * KV tiles outer, query rows inner, so K^T is built once per tile and every V row is loaded
 * once per FA_RB rows. Row state (acc, m, l) for the whole 128-row q tile lives in ctx->scratch. */
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
    if (D > 512u || (D & 31u) || !ctx || !ctx->scratch || ctx->scratch_bytes < FA_PF_SCRATCH) {
        g_flash_prefill(in, slice, nblk, T, ctx);
        return;
    }
    const uint32_t gqa = n_head / n_kv_head;
    const uint32_t q_tiles = (n_q + FA_BQ_TILE - 1) / FA_BQ_TILE;
    const uint32_t n_work = q_tiles * n_head * nsplit;
    float* acc = ctx->scratch; /* [FA_BQ_TILE][512], pair layout */
    uint32_t* Kt = (uint32_t*)(acc + (size_t)FA_BQ_TILE * 512u);
    float m[FA_BQ_TILE], l[FA_BQ_TILE];
    float tmp[32] __attribute__((aligned(64)));
    uint32_t row[FA_BKV], jlo[FA_RB], jhi[FA_RB];
    const plow_bf16* q[FA_RB];

    for (uint32_t w = slice; w < n_work; w += nblk) {
        const uint32_t sp = w % nsplit, h = (w / nsplit) % n_head, qt = w / (nsplit * n_head);
        const uint32_t hkv = h / gqa;
        const uint32_t q_base = qt * FA_BQ_TILE;
        const uint32_t q_end = q_base + FA_BQ_TILE < n_q ? q_base + FA_BQ_TILE : n_q;
        const uint32_t n_rows = q_end - q_base;
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

        for (uint32_t r = 0; r < n_rows; r++) {
            m[r] = G_NEG_INF;
            l[r] = 0.0f;
            memset(acc + (size_t)r * 512, 0, sizeof(float) * D);
        }
        for (uint32_t kt = my_lo; kt < my_hi; kt += FA_BKV) {
            const uint32_t nk = my_hi - kt < FA_BKV ? my_hi - kt : FA_BKV;
            for (uint32_t j = 0; j < FA_BKV; j++) row[j] = (kt + j) & kv_mask;
            v_build_kt(Kt, kbase, row, D);
            for (uint32_t r0 = 0; r0 < n_rows; r0 += FA_RB) {
                const uint32_t nr = n_rows - r0 < FA_RB ? n_rows - r0 : FA_RB;
                for (uint32_t r = 0; r < FA_RB; r++) {
                    const uint32_t qi = q_base + r0 + (r < nr ? r : 0);
                    const uint32_t qg = q_pos0 + qi;
                    q[r] = Q + ((size_t)qi * n_head + h) * D;
                    /* Valid keys of this row inside the tile: causal end and window start. */
                    uint32_t hi_ = kt <= qg ? qg - kt + 1 : 0;
                    if (hi_ > nk) hi_ = nk;
                    if (kt + hi_ > n_kv) hi_ = n_kv > kt ? n_kv - kt : 0;
                    uint32_t lo_ = 0;
                    if (window && qg + 1 > window) {
                        const uint32_t w0 = qg + 1 - window;
                        lo_ = w0 > kt ? w0 - kt : 0;
                    }
                    jlo[r] = lo_;
                    jhi[r] = r < nr ? hi_ : 0u;
                }
                uint32_t any = 0;
                for (uint32_t r = 0; r < nr; r++) any |= jlo[r] < jhi[r];
                if (!any) continue;
                if (nr == FA_RB) v_pf_block(q, Kt, vbase, row, jlo, jhi, acc + (size_t)r0 * 512, m + r0, l + r0, D, scale, FA_RB);
                else v_pf_block(q, Kt, vbase, row, jlo, jhi, acc + (size_t)r0 * 512, m + r0, l + r0, D, scale, nr);
            }
        }
        for (uint32_t r = 0; r < n_rows; r++) {
            const uint32_t qi = q_base + r;
            const float* ar = acc + (size_t)r * 512;
            if (nsplit == 1u && O_final) {
                const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
                const __m512 vinv = _mm512_set1_ps(inv);
                plow_bf16* orow = O_final + ((size_t)qi * n_head + h) * D;
                for (uint32_t d = 0; d < D; d += 32) {
                    v_pv_unperm(tmp, ar + d);
                    v_store_bf16(orow + d, _mm512_mul_ps(_mm512_load_ps(tmp), vinv));
                    v_store_bf16(orow + d + 16, _mm512_mul_ps(_mm512_load_ps(tmp + 16), vinv));
                }
                continue;
            }
            float* op = Opart + ((size_t)(qi * n_head + h) * nsplit + sp) * D;
            for (uint32_t d = 0; d < D; d += 32) v_pv_unperm(op + d, ar + d);
            float* ml = mlpart + ((size_t)(qi * n_head + h) * nsplit + sp) * 2;
            ml[0] = m[r];
            ml[1] = l[r];
        }
    }
}

/* --- decode ---------------------------------------------------------------------------- */

/* Sum each of 16 vectors into lane i of the result (transposing reduction: 31 shuffles + 15 adds
 * vs 16 horizontal sums of ~8 uops each). */
static inline __m512 v_hsum16(const __m512* v) {
    __m512 t[8], u[4], w[2];
    #pragma GCC unroll 8
    for (int k = 0; k < 8; k++)
        t[k] = _mm512_add_ps(_mm512_shuffle_f32x4(v[2 * k], v[2 * k + 1], 0x44),
                             _mm512_shuffle_f32x4(v[2 * k], v[2 * k + 1], 0xEE));
    #pragma GCC unroll 8
    for (int k = 0; k < 4; k++)
        u[k] = _mm512_add_ps(_mm512_shuffle_f32x4(t[2 * k], t[2 * k + 1], 0x88),
                             _mm512_shuffle_f32x4(t[2 * k], t[2 * k + 1], 0xDD));
    #pragma GCC unroll 8
    for (int k = 0; k < 2; k++)
        w[k] = _mm512_add_ps(_mm512_shuffle_ps(u[2 * k], u[2 * k + 1], 0x44),
                             _mm512_shuffle_ps(u[2 * k], u[2 * k + 1], 0xEE));
    const __m512 z = _mm512_add_ps(_mm512_shuffle_ps(w[0], w[1], 0x88), _mm512_shuffle_ps(w[0], w[1], 0xDD));
    return _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), z);
}

/* One decode block: NK keys x NH heads, NK/NH compile-time in the hot instantiation so the 16
 * dpbf16 chains stay in registers. S lanes are j*FA_GF+h. P stays f32 (hi + lo bf16 parts for the
 * dpbf16 PV): one v_expf for the block, one for the corr terms. */
static inline __attribute__((always_inline)) void v_dec_block(
    const plow_bf16* const* q, const plow_bf16* const* kr, const plow_bf16* const* vr, float* acc,
    float* m, float* l, uint32_t D, float scale, const uint32_t NK, const uint32_t NH) {
    __m512 a[FA_KB * FA_GF];
    #pragma GCC unroll 16
    for (uint32_t i = 0; i < FA_KB * FA_GF; i++) a[i] = _mm512_setzero_ps();
    for (uint32_t d = 0; d < D; d += 32) {
        #pragma GCC unroll 8
        for (uint32_t j = 0; j < NK; j++) {
            const __m512bh kc = (__m512bh)_mm512_loadu_si512((const void*)(kr[j] + d));
            #pragma GCC unroll 8
            for (uint32_t h = 0; h < NH; h++)
                a[j * FA_GF + h] = _mm512_dpbf16_ps(
                    a[j * FA_GF + h], kc, (__m512bh)_mm512_loadu_si512((const void*)(q[h] + d)));
        }
    }
    const __mmask16 valid = (__mmask16)(((1u << (2 * NK)) - 1u) & (NH == 2 ? 0xFFFFu : 0x5555u));
    __m512 s;
    if (NK == FA_KB && NH == FA_GF) {
        s = v_hsum16(a);
    } else {
        float sv[16] __attribute__((aligned(64))) = {0};
        #pragma GCC unroll 8
        for (uint32_t j = 0; j < NK; j++)
            #pragma GCC unroll 8
            for (uint32_t h = 0; h < NH; h++) sv[j * FA_GF + h] = _mm512_reduce_add_ps(a[j * FA_GF + h]);
        s = _mm512_load_ps(sv);
    }
    s = _mm512_mul_ps(s, _mm512_set1_ps(scale));
    float mnew[FA_GF] = {0}, corr[FA_GF] = {0};
    float cv[16] __attribute__((aligned(64))) = {0};
    __m512 msub = _mm512_setzero_ps();
    #pragma GCC unroll 8
    for (uint32_t h = 0; h < NH; h++) {
        const __mmask16 hm = valid & (h ? 0xAAAA : 0x5555);
        const float bm = _mm512_mask_reduce_max_ps(hm, s);
        mnew[h] = m[h] > bm ? m[h] : bm;
        msub = _mm512_mask_mov_ps(msub, hm, _mm512_set1_ps(mnew[h]));
        cv[h] = m[h] - mnew[h];
    }
    const __m512 p = _mm512_maskz_mov_ps(valid, v_expf(_mm512_sub_ps(s, msub)));
    _mm512_store_ps(cv, v_expf(_mm512_load_ps(cv)));
    #pragma GCC unroll 8
    for (uint32_t h = 0; h < NH; h++) {
        const __mmask16 hm = valid & (h ? 0xAAAA : 0x5555);
        corr[h] = m[h] == G_NEG_INF ? 0.0f : cv[h];
        l[h] = l[h] * corr[h] + _mm512_mask_reduce_add_ps(hm, p);
        m[h] = mnew[h];
    }
    /* P stays f32-accurate through the bf16 PV: P = hi + lo, both bf16, two dpbf16 per pair.
     * Lanes (j, j+1) of one head pair into a u32: pp[4*jp + h] = P_2jp | P_2jp+1 << 16. */
    const __m512 phi = v_round_bf16(p);
    const __m512i pbh = _mm512_srli_epi32(_mm512_castps_si512(phi), 16);
    const __m512i pbl = _mm512_srli_epi32(_mm512_castps_si512(v_round_bf16(_mm512_sub_ps(p, phi))), 16);
    uint32_t pp[2][16] __attribute__((aligned(64)));
    _mm512_store_si512(pp[0], _mm512_or_si512(pbh, _mm512_slli_epi32(_mm512_alignr_epi32(_mm512_setzero_si512(), pbh, 2), 16)));
    _mm512_store_si512(pp[1], _mm512_or_si512(pbl, _mm512_slli_epi32(_mm512_alignr_epi32(_mm512_setzero_si512(), pbl, 2), 16)));
    const uint32_t NP = (NK + 1) / 2;
    for (uint32_t d = 0; d < D; d += 32) {
        __m512i lo[FA_KB / 2] = {0}, hi[FA_KB / 2] = {0};
        #pragma GCC unroll 8
        for (uint32_t jp = 0; jp < NP; jp++) {
            const uint32_t j1 = 2 * jp + 1 < NK ? 2 * jp + 1 : 2 * jp; /* odd tail: its P lane is 0 */
            const __m512i va = _mm512_loadu_si512((const void*)(vr[2 * jp] + d));
            const __m512i vb = _mm512_loadu_si512((const void*)(vr[j1] + d));
            lo[jp] = _mm512_unpacklo_epi16(va, vb);
            hi[jp] = _mm512_unpackhi_epi16(va, vb);
        }
        #pragma GCC unroll 8
        for (uint32_t h = 0; h < NH; h++) {
            float* ah = acc + h * 512 + d;
            const __m512 c = _mm512_set1_ps(corr[h]);
            __m512 x0 = _mm512_mul_ps(_mm512_load_ps(ah), c), x1 = _mm512_mul_ps(_mm512_load_ps(ah + 16), c);
            __m512 y0 = _mm512_setzero_ps(), y1 = y0;
            #pragma GCC unroll 8
            for (uint32_t jp = 0; jp < NP; jp++) {
                const __m512bh ph = (__m512bh)_mm512_set1_epi32((int)pp[0][4 * jp + h]);
                const __m512bh pl = (__m512bh)_mm512_set1_epi32((int)pp[1][4 * jp + h]);
                x0 = _mm512_dpbf16_ps(x0, (__m512bh)lo[jp], ph);
                x1 = _mm512_dpbf16_ps(x1, (__m512bh)hi[jp], ph);
                y0 = _mm512_dpbf16_ps(y0, (__m512bh)lo[jp], pl);
                y1 = _mm512_dpbf16_ps(y1, (__m512bh)hi[jp], pl);
            }
            _mm512_store_ps(ah, _mm512_add_ps(x0, y0));
            _mm512_store_ps(ah + 16, _mm512_add_ps(x1, y1));
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
    /* The gqa q heads behind one kv head read the same K/V rows, so a work item covers ng head
     * groups at once and each row is fetched once instead of ng times (measured: 4.0x the needed
     * load traffic at gqa 8, which put the kernel at the streaming roofline). Per-head arithmetic
     * and its block order are untouched, so every ng gives bit-identical output. Folding is also
     * what shrinks the work list, so ng is capped by the acc footprint (FA_NG) and by leaving at
     * least two evenly divided work items per slice — at batch 1 that pins ng to 1. */
    uint32_t ng = 1u;
    if (nblk && gqa >= FA_GF && (gqa % FA_GF) == 0u && (n_head % FA_GF) == 0u) {
        for (uint32_t c = FA_NG; c > 1u; c >>= 1) {
            if (((gqa / FA_GF) % c) != 0u) continue;
            const uint32_t nw = n_batch * (n_head / (FA_GF * c)) * nsplit;
            if (nw >= 2u * nblk && (nw % nblk) == 0u) { ng = c; break; }
        }
    }
    const uint32_t gs = FA_GF * ng;
    const uint32_t n_grp = (n_head + gs - 1u) / gs;
    const uint32_t n_work = n_batch * n_grp * nsplit;
    const size_t row_bytes = (size_t)D * 2u;
    float acc[FA_NG][FA_GF * 512] __attribute__((aligned(64)));
    float m[FA_NG][FA_GF], l[FA_NG][FA_GF];
    const plow_bf16* q[FA_NG][FA_GF];
    uint32_t nh[FA_NG];

    for (uint32_t w = slice; w < n_work; w += nblk) {
        const uint32_t sp = w % nsplit, hg = (w / nsplit) % n_grp, b = w / (nsplit * n_grp);
        const uint32_t h0 = hg * gs, hkv = h0 / gqa;
        const uint32_t len = (uint32_t)kv_len[b];
        const uint32_t first = (window && len > window) ? len - window : 0u;
        const uint32_t span = len - first, per = (span + nsplit - 1) / nsplit;
        const uint32_t lo = first + sp * per, hi = lo + per < len ? lo + per : len;
        const plow_bf16* kbase = K + ((size_t)b * n_kv_head + hkv) * kv_stride * D;
        const plow_bf16* vbase = V + ((size_t)b * n_kv_head + hkv) * kv_stride * D;
        for (uint32_t g = 0; g < ng; g++) {
            const uint32_t gh0 = h0 + g * FA_GF;
            nh[g] = n_head - gh0 < FA_GF ? n_head - gh0 : FA_GF;
            for (uint32_t h = 0; h < FA_GF; h++) {
                q[g][h] = Q + ((size_t)b * n_head + gh0 + (h < nh[g] ? h : 0)) * D;
                m[g][h] = G_NEG_INF;
                l[g][h] = 0.0f;
            }
            memset(acc[g], 0, sizeof(float) * 512 * nh[g]);
        }
        if (lo < hi) {
            const size_t n0 = row_bytes * (hi - lo < FA_KB ? hi - lo : FA_KB);
            for (size_t off = 0; off < n0; off += 64) {
                _mm_prefetch((const char*)(kbase + (size_t)(lo & kv_mask) * D) + off, _MM_HINT_T0);
                _mm_prefetch((const char*)(vbase + (size_t)(lo & kv_mask) * D) + off, _MM_HINT_T0);
            }
        }

        for (uint32_t kv = lo; kv < hi; kv += FA_KB) {
            const uint32_t nk = hi - kv < FA_KB ? hi - kv : FA_KB;
            const plow_bf16* kr[FA_KB];
            const plow_bf16* vr[FA_KB];
            for (uint32_t j = 0; j < nk; j++) {
                kr[j] = kbase + (size_t)((kv + j) & kv_mask) * D;
                vr[j] = vbase + (size_t)((kv + j) & kv_mask) * D;
            }
            /* Whole next block: rows are contiguous in the ring but a block spans a page at
             * hd >= 256, where the L2 streamer restarts. */
            if (kv + FA_KB < hi) {
                const uint32_t nn = hi - kv - FA_KB < FA_KB ? hi - kv - FA_KB : FA_KB;
                for (uint32_t j = 0; j < nn; j++) {
                    const size_t r = (size_t)((kv + FA_KB + j) & kv_mask) * D;
                    for (size_t off = 0; off < row_bytes; off += 64) {
                        _mm_prefetch((const char*)(kbase + r) + off, _MM_HINT_T0);
                        _mm_prefetch((const char*)(vbase + r) + off, _MM_HINT_T0);
                    }
                }
            }
            for (uint32_t g = 0; g < ng; g++) {
                if (nk == FA_KB && nh[g] == FA_GF)
                    v_dec_block(q[g], kr, vr, acc[g], m[g], l[g], D, scale, FA_KB, FA_GF);
                else if (nk == FA_KB) v_dec_block(q[g], kr, vr, acc[g], m[g], l[g], D, scale, FA_KB, 1u);
                else v_dec_block(q[g], kr, vr, acc[g], m[g], l[g], D, scale, nk, nh[g]);
            }
        }
        for (uint32_t g = 0; g < ng; g++)
            for (uint32_t h = 0; h < nh[g]; h++) {
                const uint32_t hh = h0 + g * FA_GF + h;
                float* op = Opart + ((size_t)(b * n_head + hh) * nsplit + sp) * D;
                float* ml = mlpart + ((size_t)(b * n_head + hh) * nsplit + sp) * 2;
                for (uint32_t d = 0; d < D; d += 32) v_pv_unperm(op + d, acc[g] + h * 512 + d);
                ml[0] = m[g][h];
                ml[1] = l[g][h];
            }
    }
}

/* t0=O t1=Opart t2=mlpart t3=sinks?  i0=n_batch i1=n_head i2=nsplit i3=hd. Work = (row, head,
 * d-chunk), dsplit = ceil(nblk / (n_batch*n_head)) exactly as golden / devgen flash_merge_map.
 * The sink (golden g_flash_merge) is folded into (gm, gl) once per (row, head). */
V_K(v_flash_merge) {
    (void)ctx;
    plow_bf16* O = PLOW_CPU_TEN(in, T, 0);
    const float* Opart = PLOW_CPU_TEN(in, T, 1);
    const float* mlpart = PLOW_CPU_TEN(in, T, 2);
    const PLOW_SINK_T* sinks = PLOW_CPU_TEN(in, T, 3);
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
        const float sink = sinks ? PLOW_SINK_LOAD(sinks[hb % n_head]) : G_NEG_INF;
        if (sink > gm) gm = sink;
        float gl = sinks ? expf(sink - gm) : 0.0f;
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

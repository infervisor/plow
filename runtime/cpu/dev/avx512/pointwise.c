/* Pointwise ops, AVX-512: golden/elementwise.c semantics, 16 lanes, one bf16 round on store. */
#include "avx512.h"

/* out = (a + b) * scale, or (pre + bf16(a + b)) * scale. */
V_K(v_residual) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* a = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* b = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* pre = PLOW_CPU_TEN(in, T, 3);
    const __m512 scale = _mm512_set1_ps(in->fj[0].f);
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i += 16) {
        const __mmask16 m = i + 16 <= hi ? 0xFFFF : v_tail16(hi - i);
        __m512 s = _mm512_add_ps(v_load_bf16_mask(a + i, m), v_load_bf16_mask(b + i, m));
        if (pre) s = _mm512_add_ps(v_load_bf16_mask(pre + i, m), v_round_bf16(s));
        v_store_bf16_mask(out + i, m, _mm512_mul_ps(s, scale));
    }
}

/* out = pair(gate, up): act(gate) * up, or the act-3 swiglu_oai pair form (f0 alpha, f1 limit). */
V_K(v_glu) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* gate = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* up = PLOW_CPU_TEN(in, T, 2);
    const uint32_t act = in->i[1];
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i += 16) {
        const __mmask16 m = i + 16 <= hi ? 0xFFFF : v_tail16(hi - i);
        const __m512 o = v_glu_pair(v_load_bf16_mask(gate + i, m), v_load_bf16_mask(up + i, m), act, f0, f1);
        v_store_bf16_mask(out + i, m, o);
    }
}

/* out = cap * tanh(x / cap) */
V_K(v_softcap) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const float cap = in->fj[0].f;
    const __m512 vcap = _mm512_set1_ps(cap), vinv = _mm512_set1_ps(1.0f / cap);
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i += 16) {
        const __mmask16 m = i + 16 <= hi ? 0xFFFF : v_tail16(hi - i);
        const __m512 t = v_tanh(_mm512_mul_ps(v_load_bf16_mask(x + i, m), vinv));
        v_store_bf16_mask(out + i, m, _mm512_mul_ps(vcap, t));
    }
}

/* out[t] = bf16(table[ids[t]] * scale) */
V_K(v_embed) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* table = PLOW_CPU_TEN(in, T, 1);
    const int32_t* ids = PLOW_CPU_TEN(in, T, 2);
    const uint32_t ntok = in->i[0], hidden = in->i[1];
    const __m512 scale = _mm512_set1_ps(in->fj[0].f);
    for (uint32_t t = slice; t < ntok; t += nblk) {
        const plow_bf16* src = table + (size_t)ids[t] * hidden;
        plow_bf16* dst = out + (size_t)t * hidden;
        uint32_t i = 0;
        for (; i + 16 <= hidden; i += 16)
            v_store_bf16(dst + i, _mm512_mul_ps(v_load_bf16(src + i), scale));
        if (i < hidden) {
            const __mmask16 m = v_tail16(hidden - i);
            v_store_bf16_mask(dst + i, m, _mm512_mul_ps(v_load_bf16_mask(src + i, m), scale));
        }
    }
}

/* Packed-key max of x[b][lo..hi) into part[b][slice]. Lane-parallel (key, index) then fold;
 * a later equal key never replaces (its index is larger), matching the scalar `p > best`. */
static uint64_t amax_range(const plow_bf16* x, uint32_t lo, uint32_t hi) {
    __m512i bk = _mm512_setzero_si512(), bi = _mm512_setzero_si512();
    const __m512i lane = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    uint32_t i = lo;
    for (; i + 16 <= hi; i += 16) {
        const __m512i b = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(x + i)));
        const __m512i k = v_amax_key(b);
        const __mmask16 gt = _mm512_cmpgt_epu32_mask(k, bk);
        bk = _mm512_mask_mov_epi32(bk, gt, k);
        bi = _mm512_mask_mov_epi32(bi, gt, _mm512_add_epi32(_mm512_set1_epi32((int)i), lane));
    }
    uint64_t best = v_amax_fold(bk, bi, 0);
    for (; i < hi; i++) {
        const uint64_t p = g_amax_pack(x[i], i);
        best = p > best ? p : best;
    }
    return best;
}

V_K(v_argmax) {
    (void)ctx;
    uint64_t* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint32_t n = in->i[0], B = in->i[1] ? in->i[1] : 1u;
    uint32_t lo, hi;
    g_range(n, slice, nblk, &lo, &hi);
    for (uint32_t b = 0; b < B; b++)
        part[(size_t)b * nblk + slice] = amax_range(x + (size_t)b * n, lo, hi);
}

V_K(v_argmax_fin) {
    (void)ctx; (void)nblk;
    if (slice != 0) return;
    int32_t* ids = PLOW_CPU_TEN(in, T, 0);
    const uint64_t* part = PLOW_CPU_TEN(in, T, 1);
    const uint32_t nparts = in->i[0], B = in->i[1] ? in->i[1] : 1u;
    for (uint32_t b = 0; b < B; b++) {
        const uint64_t* pb = part + (size_t)b * nparts;
        uint64_t best = 0;
        for (uint32_t i = 0; i < nparts; i++) best = pb[i] > best ? pb[i] : best;
        ids[b] = (int32_t) ~(uint32_t)(best & 0xFFFFFFFFull);
    }
}

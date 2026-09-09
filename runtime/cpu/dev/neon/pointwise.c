/* Pointwise ops, NEON: golden/elementwise.c semantics, 4 lanes, one bf16 round on store. */
#include "neon.h"

/* out = (a + b) * scale, or (pre + bf16(a + b)) * scale. */
N_K(n_residual) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* a = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* b = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* pre = PLOW_CPU_TEN(in, T, 3);
    const float scale = in->fj[0].f;
    const float32x4_t vs = vdupq_n_f32(scale);
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    uint32_t i = lo;
    for (; i + 4 <= hi; i += 4) {
        float32x4_t s = vaddq_f32(n_load4(a + i), n_load4(b + i));
        if (pre) s = vaddq_f32(n_load4(pre + i), n_round4(s));
        n_store4(out + i, vmulq_f32(s, vs));
    }
    for (; i < hi; i++) {
        const float s = plow_bf2f(a[i]) + plow_bf2f(b[i]);
        out[i] = pre ? plow_f2bf((plow_bf2f(pre[i]) + plow_bf2f(plow_f2bf(s))) * scale)
                     : plow_f2bf(s * scale);
    }
}

/* out = pair(gate, up). */
N_K(n_glu) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* gate = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* up = PLOW_CPU_TEN(in, T, 2);
    const uint32_t act = in->i[1];
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    uint32_t i = lo;
    for (; i + 4 <= hi; i += 4)
        n_store4(out + i, n_glu_pair(n_load4(gate + i), n_load4(up + i), act, f0, f1));
    if (i < hi) {
        const uint32_t c = hi - i;
        n_store4_n(out + i, n_glu_pair(n_load4_n(gate + i, c), n_load4_n(up + i, c), act, f0, f1), c);
    }
}

/* out = cap * tanh(x / cap) */
N_K(n_softcap) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const float cap = in->fj[0].f;
    const float32x4_t vcap = vdupq_n_f32(cap), vinv = vdupq_n_f32(1.0f / cap);
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    uint32_t i = lo;
    for (; i + 4 <= hi; i += 4)
        n_store4(out + i, vmulq_f32(vcap, n_tanh(vmulq_f32(n_load4(x + i), vinv))));
    if (i < hi) {
        const uint32_t c = hi - i;
        n_store4_n(out + i, vmulq_f32(vcap, n_tanh(vmulq_f32(n_load4_n(x + i, c), vinv))), c);
    }
}

/* out[t] = bf16(table[ids[t]] * scale) */
N_K(n_embed) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* table = PLOW_CPU_TEN(in, T, 1);
    const int32_t* ids = PLOW_CPU_TEN(in, T, 2);
    const uint32_t ntok = in->i[0], hidden = in->i[1];
    const float scale = in->fj[0].f;
    const float32x4_t vs = vdupq_n_f32(scale);
    for (uint32_t t = slice; t < ntok; t += nblk) {
        const plow_bf16* src = table + (size_t)ids[t] * hidden;
        plow_bf16* dst = out + (size_t)t * hidden;
        uint32_t i = 0;
        for (; i + 4 <= hidden; i += 4) n_store4(dst + i, vmulq_f32(n_load4(src + i), vs));
        for (; i < hidden; i++) dst[i] = plow_f2bf(plow_bf2f(src[i]) * scale);
    }
}

/* Packed-key max of x[b][lo..hi) into part[b][slice]. The key is 16 bits of order-preserving
 * bf16 pattern over the inverted index; the scalar pack is ~4 ops per element, so a lane-wise
 * pass over the bf16 pattern first finds the max KEY16, then the lowest index holding it. */
static uint64_t amax_range(const plow_bf16* x, uint32_t lo, uint32_t hi) {
    uint16x8_t bk = vdupq_n_u16(0);
    const uint16x8_t sign = vdupq_n_u16(0x8000);
    uint32_t i = lo;
    for (; i + 8 <= hi; i += 8) {
        const uint16x8_t b = vld1q_u16(x + i);
        const uint16x8_t neg = vtstq_u16(b, sign);
        const uint16x8_t k = vbslq_u16(neg, vmvnq_u16(b), vorrq_u16(b, sign));
        bk = vmaxq_u16(bk, k);
    }
    uint32_t best16 = vmaxvq_u16(bk);
    for (uint32_t j = i; j < hi; j++) {
        const plow_bf16 b = x[j];
        const uint32_t k = (b & 0x8000u) ? (uint32_t)(uint16_t)~b : (uint32_t)(b | 0x8000u);
        best16 = k > best16 ? k : best16;
    }
    if (hi <= lo) return 0;
    /* Lowest index with that key (golden's `p > best` keeps the earliest index). */
    for (uint32_t j = lo; j < hi; j++) {
        const plow_bf16 b = x[j];
        const uint32_t k = (b & 0x8000u) ? (uint32_t)(uint16_t)~b : (uint32_t)(b | 0x8000u);
        if (k == best16) return g_amax_pack(b, j);
    }
    return 0;
}

N_K(n_argmax) {
    (void)ctx;
    uint64_t* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint32_t n = in->i[0], B = in->i[1] ? in->i[1] : 1u;
    uint32_t lo, hi;
    g_range(n, slice, nblk, &lo, &hi);
    for (uint32_t b = 0; b < B; b++)
        part[(size_t)b * nblk + slice] = amax_range(x + (size_t)b * n, lo, hi);
}

/* Slice 0 folds the partials; identical to golden (kept as a tier entry so the tier table is
 * complete for the argmax family). */
N_K(n_argmax_fin) { g_argmax_fin(in, slice, nblk, T, ctx); }

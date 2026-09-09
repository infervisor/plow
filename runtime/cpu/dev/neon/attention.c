/* Flash family, NEON: golden/attention.c semantics (head-major KV, sliding ring, unnormalized
 * split partials). Per KV row: one bf16 dot for the score, then a 4-lane f32 rescale-and-
 * accumulate over D. The online softmax stays per element as in golden, so the two tiers agree
 * to f32 rounding. */
#include "neon.h"

#define FA_BQ_TILE 128u
#define FA_BKV 32u
#define FA_GF 2u

static inline float dot_bf16_n(const plow_bf16* a, const plow_bf16* b, uint32_t D) {
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    uint32_t d = 0;
    for (; d + 16 <= D; d += 16) {
        acc0 = vbfdotq_f32(acc0, n_ldbh(a + d), n_ldbh(b + d));
        acc1 = vbfdotq_f32(acc1, n_ldbh(a + d + 8), n_ldbh(b + d + 8));
    }
    for (; d + 8 <= D; d += 8) acc0 = vbfdotq_f32(acc0, n_ldbh(a + d), n_ldbh(b + d));
    float s = n_hsum(vaddq_f32(acc0, acc1));
    for (; d < D; d++) s += plow_bf2f(a[d]) * plow_bf2f(b[d]);
    return s;
}

/* acc[d] = acc[d] * corr + pe * v[d] */
static inline void axpy_row(float* acc, const plow_bf16* vr, float corr, float pe, uint32_t D) {
    const float32x4_t vc = vdupq_n_f32(corr), vp = vdupq_n_f32(pe);
    uint32_t d = 0;
    for (; d + 8 <= D; d += 8) {
        const uint16x8_t u = vld1q_u16(vr + d);
        vst1q_f32(acc + d, vfmaq_f32(vmulq_f32(vld1q_f32(acc + d), vc), vp, n_lo_f32(u)));
        vst1q_f32(acc + d + 4, vfmaq_f32(vmulq_f32(vld1q_f32(acc + d + 4), vc), vp, n_hi_f32(u)));
    }
    for (; d < D; d++) acc[d] = acc[d] * corr + pe * plow_bf2f(vr[d]);
}

N_K(n_flash_prefill) {
    (void)ctx;
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
    if (D > 512u) return;
    const uint32_t gqa = n_head / n_kv_head;
    const uint32_t q_tiles = (n_q + FA_BQ_TILE - 1) / FA_BQ_TILE;
    const uint32_t n_work = q_tiles * n_head * nsplit;
    float acc[512] __attribute__((aligned(16)));

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
            for (uint32_t kg = my_lo; kg < my_hi; kg++) {
                if (!(kg < n_kv && kg <= qg && (!window || qg - kg < window))) continue;
                const plow_bf16* kr = kbase + (size_t)(kg & kv_mask) * D;
                const plow_bf16* vr = vbase + (size_t)(kg & kv_mask) * D;
                const float s = dot_bf16_n(q, kr, D) * scale;
                const float mnew = m > s ? m : s;
                const float corr = m == G_NEG_INF ? 0.0f : expf(m - mnew);
                const float pe = plow_bf2f(plow_f2bf(expf(s - mnew)));
                l = l * corr + pe;
                m = mnew;
                axpy_row(acc, vr, corr, pe, D);
            }
            if (nsplit == 1u && O_final) {
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                const float32x4_t vi = vdupq_n_f32(inv);
                plow_bf16* orow = O_final + ((size_t)qi * n_head + h) * D;
                uint32_t d = 0;
                for (; d + 4 <= D; d += 4) n_store4(orow + d, vmulq_f32(vld1q_f32(acc + d), vi));
                for (; d < D; d++) orow[d] = plow_f2bf(acc[d] * inv);
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

N_K(n_flash_decode) {
    (void)ctx;
    float* Opart = PLOW_CPU_TEN(in, T, 0);
    float* mlpart = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* Q = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* K = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* V = PLOW_CPU_TEN(in, T, 4);
    const int32_t* kv_len = PLOW_CPU_TEN(in, T, 5);
    const int nrf = (in->i[1] & 0x10000u) != 0;
    if (nrf) { g_flash_decode(in, slice, nblk, T, ctx); return; }
    const uint32_t n_batch = in->i[0];
    const uint32_t n_head = in->i[1] & 0xFFFFu, n_kv_head = in->i[2];
    const uint32_t kv_stride = in->i[3];
    const uint32_t window = in->i[4];
    const uint32_t nsplit = in->i[5];
    const uint32_t D = in->i[6] & 0xFFFFu, kv_mask = in->i[7];
    const float scale = in->fj[0].f;
    if (D > 512u || nsplit == 0u) return;
    const uint32_t gqa = n_head / n_kv_head;
    /* A work item carries FA_GF query heads of ONE kv head; when GQA is not a multiple of FA_GF
     * (Llama-3.2-3B: 24/8 = 3) it carries one, and the emitter must have sized the work the same
     * way (PLOW_FA_GF_FULL=1). */
    const uint32_t gf = gqa % FA_GF == 0u ? FA_GF : 1u;
    const uint32_t n_grp = (n_head + gf - 1) / gf;
    const uint32_t n_work = n_batch * n_grp * nsplit;
    float acc[512] __attribute__((aligned(16)));

    for (uint32_t w = slice; w < n_work; w += nblk) {
        const uint32_t sp = w % nsplit, hg = (w / nsplit) % n_grp, b = w / (nsplit * n_grp);
        const uint32_t h0 = hg * gf, hkv = h0 / gqa;
        const uint32_t len = (uint32_t)kv_len[b], qpos = len - 1;
        const uint32_t first = (window && len > window) ? len - window : 0u;
        const uint32_t span = len - first, per = (span + nsplit - 1) / nsplit;
        const uint32_t lo = first + sp * per, hi = lo + per < len ? lo + per : len;
        const plow_bf16* kbase = K + ((size_t)b * n_kv_head + hkv) * kv_stride * D;
        const plow_bf16* vbase = V + ((size_t)b * n_kv_head + hkv) * kv_stride * D;

        for (uint32_t h = h0; h < h0 + gf && h < n_head; h++) {
            float* op = Opart + ((size_t)(b * n_head + h) * nsplit + sp) * D;
            float* ml = mlpart + ((size_t)(b * n_head + h) * nsplit + sp) * 2;
            const plow_bf16* q = Q + ((size_t)b * n_head + h) * D;
            float m = G_NEG_INF, l = 0.0f;
            memset(acc, 0, sizeof(float) * D);
            for (uint32_t kv = lo; kv < hi; kv++) {
                if (!(kv <= qpos && (!window || qpos - kv < window))) continue;
                const plow_bf16* kr = kbase + (size_t)(kv & kv_mask) * D;
                const plow_bf16* vr = vbase + (size_t)(kv & kv_mask) * D;
                const float s = dot_bf16_n(kr, q, D) * scale;
                const float mnew = m > s ? m : s;
                const float corr = m == G_NEG_INF ? 0.0f : expf(m - mnew);
                const float pe = expf(s - mnew);
                l = l * corr + pe;
                m = mnew;
                axpy_row(acc, vr, corr, pe, D);
            }
            memcpy(op, acc, sizeof(float) * D);
            ml[0] = m;
            ml[1] = l;
        }
    }
}

N_K(n_flash_merge) {
    (void)ctx;
    plow_bf16* O = PLOW_CPU_TEN(in, T, 0);
    const float* Opart = PLOW_CPU_TEN(in, T, 1);
    const float* mlpart = PLOW_CPU_TEN(in, T, 2);
    const PLOW_SINK_T* sinks = PLOW_CPU_TEN(in, T, 3);
    const uint32_t n_batch = in->i[0], n_head = in->i[1], nsplit = in->i[2], D = in->i[3];
    const uint32_t n_bh = n_batch * n_head;
    if (n_bh == 0u || nsplit > 64u) { g_flash_merge(in, slice, nblk, T, ctx); return; }
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
            if (ml[s * 2] != G_NEG_INF) gl += ml[s * 2 + 1] * wgt[s];
        }
        const float inv = gl > 0.0f ? 1.0f / gl : 0.0f;
        const float32x4_t vi = vdupq_n_f32(inv);
        const float* obase = Opart + (size_t)hb * nsplit * D;
        uint32_t d = d0;
        for (; d + 4 <= d1; d += 4) {
            float32x4_t acc = vdupq_n_f32(0);
            for (uint32_t s = 0; s < nsplit; s++)
                acc = vfmaq_f32(acc, vld1q_f32(obase + (size_t)s * D + d), vdupq_n_f32(wgt[s]));
            n_store4(O + (size_t)hb * D + d, vmulq_f32(acc, vi));
        }
        for (; d < d1; d++) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < nsplit; s++) acc += obase[(size_t)s * D + d] * wgt[s];
            O[(size_t)hb * D + d] = plow_f2bf(acc * inv);
        }
    }
}

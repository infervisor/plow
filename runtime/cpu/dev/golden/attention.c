/* Flash family: op_attention.h semantics. KV caches are HEAD-MAJOR [kv_head][ctx][hd] with a
 * sliding-window ring (`row & kv_mask`). Split-KV partials are UNNORMALIZED (o, m, l), f32:
 *   Opart [row][head][split][D]   mlpart [row][head][split][2]
 * Online softmax runs per element here (the GPU runs it per 32-row tile); the rounding differs
 * in f32 only. P is rounded to bf16 before PV in prefill (MFMA operand), kept f32 in decode. */
#include "golden.h"

#define FA_BQ_TILE 128u /* query rows per work item: 4 waves x FA_BQ (the Gemma flash object) */
#define FA_BKV 32u
#define FA_GF 2u        /* PLOW_FA_GF: query heads fused per decode item */

/* t0=Opart t1=mlpart t2=Q t3=K t4=V t5=O_final?
 * i0=n_q i1=n_kv i2=n_head i3=n_kv_head i4=q_pos0 i5=window i6=hd i7=nsplit
 * f0=scale  fj1.u=kv_stride  fj2.u=kv_mask.  Q is [n_q][n_head][hd]; K/V head-major.
 * nsplit==1 with t5 present writes the normalized bf16 output straight to t5 and no partial. */
G_K(g_flash_prefill) {
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
    float acc[512];

    for (uint32_t w = slice; w < n_work; w += nblk) {
        const uint32_t sp = w % nsplit, h = (w / nsplit) % n_head, qt = w / (nsplit * n_head);
        const uint32_t hkv = h / gqa;
        const uint32_t q_base = qt * FA_BQ_TILE;
        /* The split carves the TILE's causal/window-valid KV range, in whole FA_BKV tiles. */
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
                float s = 0.0f;
                for (uint32_t d = 0; d < D; d++) s += plow_bf2f(q[d]) * plow_bf2f(kr[d]);
                s *= scale;
                const float mnew = m > s ? m : s;
                const float corr = m == G_NEG_INF ? 0.0f : expf(m - mnew);
                const float pe = plow_bf2f(plow_f2bf(expf(s - mnew)));
                l = l * corr + pe;
                m = mnew;
                for (uint32_t d = 0; d < D; d++) acc[d] = acc[d] * corr + pe * plow_bf2f(vr[d]);
            }
            if (nsplit == 1u && O_final) {
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                plow_bf16* orow = O_final + ((size_t)qi * n_head + h) * D;
                for (uint32_t d = 0; d < D; d++) orow[d] = plow_f2bf(acc[d] * inv);
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
 * i0=n_batch i1=n_head i2=n_kv_head i3=kv_stride i4=window i5=nsplit i6=hd i7=kv_mask  f0=scale
 * Q is [n_batch][n_head][hd]; K/V are [n_batch][kv_head][kv_stride][hd].
 * i1 bit 16 (the NRF HeadNormRope fold) is not ported: its partials are poisoned with NaN. */
G_K(g_flash_decode) {
    (void)ctx;
    float* Opart = PLOW_CPU_TEN(in, T, 0);
    float* mlpart = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* Q = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* K = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* V = PLOW_CPU_TEN(in, T, 4);
    const int32_t* kv_len = PLOW_CPU_TEN(in, T, 5);
    const int nrf = (in->i[1] & 0x10000u) != 0;
    const uint32_t n_batch = nrf ? (in->i[0] & 0xFFu) : in->i[0];
    const uint32_t n_head = in->i[1] & 0xFFFFu, n_kv_head = in->i[2];
    const uint32_t kv_stride = nrf ? (in->i[3] & 0xFFFFFu) : in->i[3];
    const uint32_t window = nrf ? (in->i[0] >> 8) : in->i[4];
    const uint32_t nsplit = nrf ? (in->i[3] >> 20) : in->i[5];
    const uint32_t D = in->i[6] & 0xFFFFu, kv_mask = in->i[7];
    const float scale = in->fj[0].f;
    if (D > 512u || nsplit == 0u) return;
    const uint32_t gqa = n_head / n_kv_head;
    const uint32_t n_grp = (n_head + FA_GF - 1) / FA_GF;
    const uint32_t n_work = n_batch * n_grp * nsplit;
    float acc[512];

    for (uint32_t w = slice; w < n_work; w += nblk) {
        const uint32_t sp = w % nsplit, hg = (w / nsplit) % n_grp, b = w / (nsplit * n_grp);
        const uint32_t h0 = hg * FA_GF, hkv = h0 / gqa;
        const uint32_t len = (uint32_t)kv_len[b], qpos = len - 1;
        const uint32_t first = (window && len > window) ? len - window : 0u;
        const uint32_t span = len - first, per = (span + nsplit - 1) / nsplit;
        const uint32_t lo = first + sp * per, hi = lo + per < len ? lo + per : len;
        const plow_bf16* kbase = K + ((size_t)b * n_kv_head + hkv) * kv_stride * D;
        const plow_bf16* vbase = V + ((size_t)b * n_kv_head + hkv) * kv_stride * D;

        for (uint32_t h = h0; h < h0 + FA_GF && h < n_head; h++) {
            float* op = Opart + ((size_t)(b * n_head + h) * nsplit + sp) * D;
            float* ml = mlpart + ((size_t)(b * n_head + h) * nsplit + sp) * 2;
            if (nrf) {
                for (uint32_t d = 0; d < D; d++) op[d] = NAN;
                ml[0] = NAN;
                ml[1] = NAN;
                continue;
            }
            const plow_bf16* q = Q + ((size_t)b * n_head + h) * D;
            float m = G_NEG_INF, l = 0.0f;
            memset(acc, 0, sizeof(float) * D);
            for (uint32_t kv = lo; kv < hi; kv++) {
                if (!(kv <= qpos && (!window || qpos - kv < window))) continue;
                const plow_bf16* kr = kbase + (size_t)(kv & kv_mask) * D;
                const plow_bf16* vr = vbase + (size_t)(kv & kv_mask) * D;
                float s = 0.0f;
                for (uint32_t d = 0; d < D; d++) s += plow_bf2f(kr[d]) * plow_bf2f(q[d]);
                s *= scale;
                const float mnew = m > s ? m : s;
                const float corr = m == G_NEG_INF ? 0.0f : expf(m - mnew);
                const float pe = expf(s - mnew);
                l = l * corr + pe;
                m = mnew;
                for (uint32_t d = 0; d < D; d++) acc[d] = acc[d] * corr + pe * plow_bf2f(vr[d]);
            }
            memcpy(op, acc, sizeof(float) * D);
            ml[0] = m;
            ml[1] = l;
        }
    }
}

/* t0=O t1=Opart t2=mlpart t3=sinks?(PLOW_SINK_T[n_head])  i0=n_batch i1=n_head i2=nsplit i3=hd
 * Work is (row, head, d-chunk) with dsplit = ceil(nblk / (n_batch*n_head)) — must match
 * flash_merge_map() in crates/devgen/src/lib.rs.
 * Sinks (GPT-OSS): one extra UNSCALED logit per head that has no value row. It enters the
 * softmax denominator exactly once, here: gm' = max(gm, sink_h), gl' = gl*e^(gm-gm') + e^(sink_h-gm'). */
G_K(g_flash_merge) {
    (void)ctx;
    plow_bf16* O = PLOW_CPU_TEN(in, T, 0);
    const float* Opart = PLOW_CPU_TEN(in, T, 1);
    const float* mlpart = PLOW_CPU_TEN(in, T, 2);
    const PLOW_SINK_T* sinks = PLOW_CPU_TEN(in, T, 3);
    const uint32_t n_batch = in->i[0], n_head = in->i[1], nsplit = in->i[2], D = in->i[3];
    const uint32_t n_bh = n_batch * n_head;
    if (n_bh == 0u) return;
    const uint32_t dsplit = (nblk + n_bh - 1) / n_bh;
    const uint32_t dchunk = (D + dsplit - 1) / dsplit;
    const uint32_t n_work = n_bh * dsplit;
    for (uint32_t w = slice; w < n_work; w += nblk) {
        const uint32_t dp = w % dsplit, hb = w / dsplit;
        const uint32_t d0 = dp * dchunk, d1 = d0 + dchunk < D ? d0 + dchunk : D;
        const float* ml = mlpart + (size_t)hb * nsplit * 2;
        float gm = G_NEG_INF;
        for (uint32_t s = 0; s < nsplit; s++) gm = gm > ml[s * 2] ? gm : ml[s * 2];
        const float sink = sinks ? PLOW_SINK_LOAD(sinks[hb % n_head]) : G_NEG_INF;
        if (sink > gm) gm = sink;
        float gl = sinks ? expf(sink - gm) : 0.0f;
        for (uint32_t s = 0; s < nsplit; s++)
            if (ml[s * 2] != G_NEG_INF) gl += ml[s * 2 + 1] * expf(ml[s * 2] - gm);
        const float inv = gl > 0.0f ? 1.0f / gl : 0.0f;
        const float* obase = Opart + (size_t)hb * nsplit * D;
        for (uint32_t d = d0; d < d1; d++) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < nsplit; s++)
                if (ml[s * 2] != G_NEG_INF) acc += obase[(size_t)s * D + d] * expf(ml[s * 2] - gm);
            O[(size_t)hb * D + d] = plow_f2bf(acc * inv);
        }
    }
}

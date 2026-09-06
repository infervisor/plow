/* avx512/route.c — MoE router top-k (83) and combine (87); align stays golden (a single-slice
 * scatter). Same slice partitions as golden/moe_route.c. */
#include "avx512.h"
#include "../golden/gptoss.h"

/* t0=out([T,H] bf16) t1=residual? t2=shared? t3=part([T*k,H] f32)  i0=H i1=k i2=T.
 * A slice owns a flat [lo, hi) run of (token, h); the k partials of one h are summed in the
 * golden order, so the result is bit-identical. */
V_K(v_moe_combine_pf) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* residual = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* shared = PLOW_CPU_TEN(in, T, 2);
    const float* part = PLOW_CPU_TEN(in, T, 3);
    const uint32_t H = in->i[0], k = in->i[1], nt = in->i[2] ? in->i[2] : 1u;
    uint32_t lo, hi;
    g_range(nt * H, slice, nblk, &lo, &hi);
    uint32_t i = lo;
    while (i < hi) {
        const uint32_t tok = i / H, h0 = i - tok * H;
        const uint32_t n = h0 + (hi - i) < H ? (uint32_t)(hi - i) : H - h0;
        const float* pt = part + (size_t)tok * k * H;
        uint32_t h = h0;
        for (; h < h0 + n; h += 16u) {
            const __mmask16 m = h0 + n - h >= 16u ? (__mmask16)0xFFFF : v_tail16(h0 + n - h);
            __m512 acc = residual ? v_load_bf16_mask(residual + (size_t)tok * H + h, m) : _mm512_setzero_ps();
            if (shared) acc = _mm512_add_ps(acc, v_load_bf16_mask(shared + (size_t)tok * H + h, m));
            for (uint32_t j = 0; j < k; j++)
                acc = _mm512_add_ps(acc, _mm512_maskz_loadu_ps(m, pt + (size_t)j * H + h));
            v_store_bf16_mask(out + (size_t)tok * H + h, m, acc);
        }
        i += n;
    }
}

#define ROUTE_MAX_TOPK 16u

/* golden route_token with the O(n_exp^2) rank-by-counting replaced by k vector selection passes.
 * Golden ranks e by #{f : key_f > key_e} and puts e at wl[rank]; the keys are a total order, so
 * wl[j] is exactly the j-th largest key — what v_topk_u32 returns, with the same lowest-id tie
 * break. Scores, the bias add and the gate tail are golden's scalar math, unchanged. */
static void v_route_token(plow_moe_route* tab, const plow_bf16* logit, const float* bias, uint32_t n_exp,
                          uint32_t k, uint32_t flags, float route_scale, float* score, uint32_t* key) {
    const int sigmoid = (flags & 1u) != 0, norm_topk = (flags & 2u) != 0;
    if (sigmoid) {
        for (uint32_t e = 0; e < n_exp; e++) score[e] = 1.0f / (1.0f + expf(-plow_bf2f(logit[e])));
    } else {
        float m = -1e30f, s = 0.0f;
        for (uint32_t e = 0; e < n_exp; e++) { score[e] = plow_bf2f(logit[e]); if (score[e] > m) m = score[e]; }
        for (uint32_t e = 0; e < n_exp; e++) { score[e] = expf(score[e] - m); s += score[e]; }
        for (uint32_t e = 0; e < n_exp; e++) score[e] /= s;
    }
    if (bias) {
        for (uint32_t e = 0; e < n_exp; e += 16u) {
            const __mmask16 m = n_exp - e >= 16u ? (__mmask16)0xFFFFu : v_tail16(n_exp - e);
            _mm512_mask_storeu_epi32(key + e, m,
                                     v_key32(_mm512_add_ps(_mm512_maskz_loadu_ps(m, score + e),
                                                           _mm512_maskz_loadu_ps(m, bias + e))));
        }
    } else {
        v_key_row(key, score, n_exp);
    }
    uint32_t wl[ROUTE_MAX_TOPK];
    v_topk_u32(key, n_exp, k, wl);
    float gate[ROUTE_MAX_TOPK], sum = 0.0f;
    for (uint32_t j = 0; j < k; j++) { gate[j] = score[wl[j]]; sum += gate[j]; }
    for (uint32_t j = 0; j < k; j++) {
        if (norm_topk && sum != 0.0f) gate[j] /= sum;
        gate[j] *= route_scale;
        tab[j].eid = wl[j];
        tab[j].gate = gate[j];
    }
}

/* t0=table([T*k]) t1=logit([T,n_exp] bf16) t3=bias?  i1=n_exp i2=k i3=flags i4=T i6=n_group
 * i7=topk_group  f0=route_scale. Token t is owned by slice t % nblk, as golden. */
V_K(v_moe_router_topk_pf) {
    const uint32_t n_exp = in->i[1], k = in->i[2], flags = in->i[3], nt = in->i[4] ? in->i[4] : 1u;
    /* k > ROUTE_MAX_TOPK (UNUSED tail), k > n_exp (golden's wl default) and i6 > 1 (poison) are
     * golden's edges; scratch holds score[n_exp] + key[n_exp]. */
    if (k > ROUTE_MAX_TOPK || k > n_exp || n_exp > V_TOPK_MAX_E || in->i[6] > 1u || !ctx || !ctx->scratch ||
        ctx->scratch_bytes < (size_t)n_exp * 8u) {
        g_moe_router_topk_pf(in, slice, nblk, T, ctx);
        return;
    }
    plow_moe_route* table = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* logit = PLOW_CPU_TEN(in, T, 1);
    const float* bias = PLOW_CPU_TEN(in, T, 3);
    float* score = (float*)ctx->scratch;
    uint32_t* key = (uint32_t*)(score + n_exp);
    for (uint32_t tok = slice; tok < nt; tok += nblk)
        v_route_token(table + (size_t)tok * k, logit + (size_t)tok * n_exp, bias, n_exp, k, flags,
                      in->fj[0].f, score, key);
}

void v_register_route(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_MOE_COMBINE_PF] = v_moe_combine_pf;
    tab[PLOW_DOP_MOE_ROUTER_TOPK_PF] = v_moe_router_topk_pf;
}

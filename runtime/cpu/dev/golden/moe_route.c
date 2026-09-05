/* golden/moe_route.c — MoE routing ops shared by every MoE family (dev_isa.h, op_moe.h):
 *   83 MOE_ROUTER_TOPK_PF  per-token softmax/sigmoid + top-k -> table[{u32 eid, f32 gate}]
 *   84 MOE_ALIGN_PF        ONE slice: histogram slots by expert, MPF_BM-padded prefix, scatter
 *   87 MOE_COMBINE_PF      out = residual + shared? + sum_j part[tok*k+j]
 * Ported 1:1 from runtime/amd/op_moe.h (d_moe_router_topk{,_pf}, d_moe_align_pf phase 0/3,
 * d_moe_combine_pf). Selection is by the packed key ordered(score+bias) with lower expert id
 * winning ties; gates are the UNBIASED scores, optionally normalised over the winners, then
 * scaled by f0 — exactly the GPU's byte order. */
#include "gptoss.h"
#include <math.h>
#include <string.h>

#define MPF_BM 64u
#define ROUTE_MAX_TOPK 16u

static inline uint32_t f32_key(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u); /* monotone f32 -> u32 */
}

/* One token: logit[n_exp] -> table[k]. */
static void route_token(plow_moe_route* tab, const plow_bf16* logit, const float* bias,
                        uint32_t n_exp, uint32_t k, uint32_t flags, float route_scale,
                        float* score /* scratch [n_exp] */) {
    const int sigmoid = (flags & 1u) != 0, norm_topk = (flags & 2u) != 0;
    for (uint32_t j = ROUTE_MAX_TOPK; j < k; j++) { tab[j].eid = PLOW_EXPERT_UNUSED; tab[j].gate = 0.0f; }
    if (k > ROUTE_MAX_TOPK) k = ROUTE_MAX_TOPK;
    if (sigmoid) {
        for (uint32_t e = 0; e < n_exp; e++) score[e] = 1.0f / (1.0f + expf(-plow_bf2f(logit[e])));
    } else {
        float m = -1e30f, s = 0.0f;
        for (uint32_t e = 0; e < n_exp; e++) { score[e] = plow_bf2f(logit[e]); if (score[e] > m) m = score[e]; }
        for (uint32_t e = 0; e < n_exp; e++) { score[e] = expf(score[e] - m); s += score[e]; }
        for (uint32_t e = 0; e < n_exp; e++) score[e] /= s;
    }
    /* rank(e) = #{f : key_f > key_e}; keys are unique (id in the low bits), winners = rank < k. */
    uint32_t wl[ROUTE_MAX_TOPK];
    for (uint32_t j = 0; j < k; j++) wl[j] = n_exp - 1u;
    for (uint32_t e = 0; e < n_exp; e++) {
        const uint64_t ke = ((uint64_t)f32_key(score[e] + (bias ? bias[e] : 0.0f)) << 20) |
                            (uint64_t)((n_exp - 1u - e) & 0xFFFFFu);
        uint32_t rank = 0;
        for (uint32_t f = 0; f < n_exp; f++) {
            const uint64_t kf = ((uint64_t)f32_key(score[f] + (bias ? bias[f] : 0.0f)) << 20) |
                                (uint64_t)((n_exp - 1u - f) & 0xFFFFFu);
            rank += kf > ke;
        }
        if (rank < k) wl[rank] = e;
    }
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
 * i7=topk_group  f0=route_scale. Token t is owned by slice t % nblk (op_moe.h token loop). */
G_K(g_moe_router_topk_pf) {
    plow_moe_route* table = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* logit = PLOW_CPU_TEN(in, T, 1);
    const float* bias = PLOW_CPU_TEN(in, T, 3);
    const uint32_t n_exp = in->i[1], k = in->i[2], flags = in->i[3], nt = in->i[4] ? in->i[4] : 1u;
    const float route_scale = in->fj[0].f;
    if (!ctx || !ctx->scratch || ctx->scratch_bytes < n_exp * 4u || in->i[6] > 1u) {
        /* group-limited routing (i6 > 1) is not ported: poison so it cannot pass silently. */
        for (uint32_t s = slice; s < nt * k; s += nblk) { table[s].eid = PLOW_EXPERT_UNUSED; table[s].gate = NAN; }
        return;
    }
    float* score = (float*)ctx->scratch; /* n_exp f32: <= 1 KiB for every shipping model */
    for (uint32_t tok = slice; tok < nt; tok += nblk)
        route_token(table + (size_t)tok * k, logit + (size_t)tok * n_exp, bias, n_exp, k, flags,
                    route_scale, score);
}

/* t0=meta(i32 [3*n_exp+1]) t1=table t2=row_token(u32) t3=row_partidx(u32) t4=row_gate(f32)
 * i0=T i1=n_exp i2=k. Single slice. Live rows of expert e occupy [rowoff[e], rowoff[e]+cnt[e]) in
 * ascending slot order; the padded tail up to the next MPF_BM multiple carries UNUSED markers. */
G_K(g_moe_align_pf) {
    (void)ctx;
    if (slice != 0 || nblk == 0) return;
    int32_t* meta = PLOW_CPU_TEN(in, T, 0);
    const plow_moe_route* table = PLOW_CPU_TEN(in, T, 1);
    uint32_t* row_token = PLOW_CPU_TEN(in, T, 2);
    uint32_t* row_partidx = PLOW_CPU_TEN(in, T, 3);
    float* row_gate = PLOW_CPU_TEN(in, T, 4);
    const uint32_t nt = in->i[0], n_exp = in->i[1], k = in->i[2], nslot = nt * k;
    int32_t* rowoff = meta;
    int32_t* cnt = meta + n_exp;
    int32_t* tilep = meta + 2u * n_exp;
    for (uint32_t e = 0; e < n_exp; e++) cnt[e] = 0;
    for (uint32_t s = 0; s < nslot; s++) {
        const uint32_t eid = table ? table[s].eid : 0u;
        if (eid < n_exp) cnt[eid]++;
    }
    uint32_t off = 0, tiles = 0;
    for (uint32_t e = 0; e < n_exp; e++) {
        rowoff[e] = (int32_t)off;
        tilep[e] = (int32_t)tiles;
        const uint32_t t = ((uint32_t)cnt[e] + MPF_BM - 1u) / MPF_BM;
        tiles += t;
        off += t * MPF_BM;
    }
    tilep[n_exp] = (int32_t)tiles;
    for (uint32_t r = 0; r < off; r++) { row_token[r] = PLOW_EXPERT_UNUSED; row_partidx[r] = PLOW_EXPERT_UNUSED; row_gate[r] = 0.0f; }
    for (uint32_t s = 0; s < nslot; s++) {
        const uint32_t eid = table ? table[s].eid : 0u;
        if (eid >= n_exp) continue;
        const uint32_t pos = (uint32_t)rowoff[eid]++;
        row_token[pos] = s / k;
        row_partidx[pos] = s;
        row_gate[pos] = table ? table[s].gate : 1.0f;
    }
    for (uint32_t e = 0; e < n_exp; e++) rowoff[e] -= cnt[e]; /* restore segment starts */
}

/* t0=out([T,H] bf16) t1=residual? t2=shared? t3=part([T*k,H] f32)  i0=H i1=k i2=T. */
G_K(g_moe_combine_pf) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* residual = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* shared = PLOW_CPU_TEN(in, T, 2);
    const float* part = PLOW_CPU_TEN(in, T, 3);
    const uint32_t H = in->i[0], k = in->i[1], nt = in->i[2] ? in->i[2] : 1u;
    uint32_t lo, hi;
    g_range(nt * H, slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) {
        const uint32_t tok = i / H, h = i - tok * H;
        float acc = residual ? plow_bf2f(residual[i]) : 0.0f;
        if (shared) acc += plow_bf2f(shared[i]);
        const float* pt = part + (size_t)tok * k * H;
        for (uint32_t j = 0; j < k; j++) acc += pt[(size_t)j * H + h];
        out[i] = plow_f2bf(acc);
    }
}

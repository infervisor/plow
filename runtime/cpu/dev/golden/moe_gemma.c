/* golden/moe_gemma.c — Gemma-4 26B-A4B hybrid MoE ops (dev_isa.h 63/68/69/70/71 decode, 73/75/76/77
 * prefill; 74 ALIGN is g_moe_align_pf), ported 1:1 from runtime/amd/op_moe.h's `_gemma` family.
 *
 * Expert weights are reached through `ewt`, a host-filled u64[n_exp*2] table: ewt[e*2+0] = the
 * FUSED gate_up tensor of expert e ([2*I][H] bf16, gate rows [0,I), up rows [I,2I)), ewt[e*2+1] =
 * its down tensor ([H][I]). A zero base (expert not resident) or eid >= n_exp skips the slot: GLU
 * writes nothing, DOWN writes an explicit 0 partial. Routing table entries are {u32 eid, f32 gate};
 * the gate already carries norm_topk and the per-expert scale. Gemma GLU is gelu_tanh (act 0). */
#include "gptoss.h"
#include <math.h>
#include <string.h>

#define GM_NB(v) ((v) ? (v) : 1u)

#include <stdlib.h>
/* PLOW_MOE_XN_BF16=1: round the fused-norm expert input xn through bf16 (HF's numerics: the
 * pre-FFN norm output is a bf16 tensor). Default keeps xn in f32 (the AMD reference). */
static int gm_xn_bf16(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PLOW_MOE_XN_BF16"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v;
}
static inline float gm_round_xn(float x) { return gm_xn_bf16() ? plow_bf2f(plow_f2bf(x)) : x; }

static inline const plow_bf16* ewt_base(const uint64_t* ewt, uint32_t eid, uint32_t which) {
    return (const plow_bf16*)(uintptr_t)ewt[(size_t)eid * 2u + which];
}

static inline float gm_dot_bf16_f32(const plow_bf16* w, const float* x, uint32_t n) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) acc = fmaf(plow_bf2f(w[i]), x[i], acc);
    return acc;
}

static inline float gm_dot_bf16(const plow_bf16* w, const plow_bf16* x, uint32_t n) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) acc = fmaf(plow_bf2f(w[i]), plow_bf2f(x[i]), acc);
    return acc;
}

static inline float gm_invrms(const plow_bf16* r, uint32_t H, float eps) {
    float ss = 0.0f;
    for (uint32_t h = 0; h < H; h++) { const float v = plow_bf2f(r[h]); ss += v * v; }
    return 1.0f / sqrtf(ss / (float)H + eps);
}

/* softmax -> top-k (lower id wins ties) -> norm_topk -> per-expert scale, on one row's f32 scores
 * (consumed). Identical selection key to op_moe.h gmoe_softmax_topk_tail. */
static void gm_topk_tail(plow_moe_route* tab, float* sc, const plow_bf16* pes, uint32_t n_exp, uint32_t k) {
    float m = -1e30f;
    for (uint32_t e = 0; e < n_exp; e++) m = fmaxf(m, sc[e]);
    float s = 0.0f;
    for (uint32_t e = 0; e < n_exp; e++) { sc[e] = expf(sc[e] - m); s += sc[e]; }
    for (uint32_t e = 0; e < n_exp; e++) sc[e] /= s;
    for (uint32_t j = 0; j < k; j++) {
        uint64_t best = 0; uint32_t bid = 0;
        for (uint32_t e = 0; e < n_exp; e++) {
            uint32_t sb; memcpy(&sb, &sc[e], 4);
            sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
            const uint64_t key = ((uint64_t)sb << 20) | (uint64_t)((n_exp - 1u - e) & 0xFFFFFu);
            if (key > best) { best = key; bid = e; }
        }
        tab[j].eid = bid; tab[j].gate = sc[bid];
        sc[bid] = -1e30f;
    }
    float gs = 0.0f;
    for (uint32_t j = 0; j < k; j++) gs += tab[j].gate;
    for (uint32_t j = 0; j < k; j++) {
        float g = tab[j].gate;
        if (gs != 0.0f) g /= gs;
        tab[j].gate = g * plow_bf2f(pes[tab[j].eid]);
    }
}

/* One router row: weightless RMS, h2 = r*invrms*scale*root, sc[e] = h2 . proj[e], tail. */
static void gm_router_row(plow_moe_route* tab, const plow_bf16* resid, const plow_bf16* proj,
                          const plow_bf16* scale, const plow_bf16* pes, uint32_t H, uint32_t n_exp,
                          uint32_t k, float root, float eps, float* h2, float* sc) {
    const float inv = gm_invrms(resid, H, eps);
    for (uint32_t h = 0; h < H; h++) h2[h] = plow_bf2f(resid[h]) * inv * plow_bf2f(scale[h]) * root;
    for (uint32_t e = 0; e < n_exp; e++) sc[e] = gm_dot_bf16_f32(proj + (size_t)e * H, h2, H);
    gm_topk_tail(tab, sc, pes, n_exp, k);
}

/* 69: t0=score(f32 [B][E]) t1=resid([B][H]) t2=proj([E][H]) t3=scale[H]  i0=H i1=E i2=B  f0=root f1=eps.
 * Slices own (row, expert) pairs; the RMS scalar is recomputed per pair (tiny). */
G_K(g_moe_router_gemma_score_fast) {
    (void)ctx;
    float* score = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* resid = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* proj = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* scale = PLOW_CPU_TEN(in, T, 3);
    const uint32_t H = in->i[0], E = in->i[1], nrow = GM_NB(in->i[2]);
    const float root = in->fj[0].f, eps = in->fj[1].f;
    uint32_t lo, hi;
    g_range(nrow * E, slice, nblk, &lo, &hi);
    uint32_t cur_row = 0xFFFFFFFFu; float inv = 0.0f;
    for (uint32_t idx = lo; idx < hi; idx++) {
        const uint32_t row = idx / E, e = idx - row * E;
        const plow_bf16* rr = resid + (size_t)row * H;
        if (row != cur_row) { inv = gm_invrms(rr, H, eps); cur_row = row; }
        const plow_bf16* pr = proj + (size_t)e * H;
        float acc = 0.0f;
        for (uint32_t h = 0; h < H; h++)
            acc = fmaf(plow_bf2f(rr[h]) * inv * plow_bf2f(scale[h]) * root, plow_bf2f(pr[h]), acc);
        score[(size_t)row * E + e] = acc;
    }
}

/* 68: t0=table([B][k]) t1=score(f32 [B][E]) t2=pes(bf16 [E])  i1=E i2=k i3=B. Row per slice. */
G_K(g_moe_router_gemma_topk) {
    plow_moe_route* table = PLOW_CPU_TEN(in, T, 0);
    const float* score = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* pes = PLOW_CPU_TEN(in, T, 2);
    const uint32_t E = in->i[1], k = in->i[2], nrow = GM_NB(in->i[3]);
    if (!ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)E * 4u) return;
    float* sc = ctx->scratch;
    for (uint32_t row = slice; row < nrow; row += GM_NB(nblk)) {
        memcpy(sc, score + (size_t)row * E, (size_t)E * 4u);
        gm_topk_tail(table + (size_t)row * k, sc, pes, E, k);
    }
}

/* 71: t0=fu([B*k][I]) t1=resid([B][H]) t2=table t3=ewt t4=gamma[H]  i0=k i1=I i2=H i3=E i5=B  f0=eps.
 * fu[slot][n] = gelu_tanh(gate_e[n] . xn) * (up_e[n] . xn), xn = resid * invrms * gamma (f32). */
G_K(g_moe_expert_glu_norm_gemma) {
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* resid = PLOW_CPU_TEN(in, T, 1);
    const plow_moe_route* table = PLOW_CPU_TEN(in, T, 2);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 4);
    const uint32_t k = in->i[0], I = in->i[1], H = in->i[2], E = in->i[3], nrow = GM_NB(in->i[5]);
    const float eps = in->fj[0].f;
    if (!ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)H * 4u) return;
    float* xn = ctx->scratch;
    const uint32_t nslot = nrow * k;
    uint32_t lo, hi;
    g_range(nslot * I, slice, nblk, &lo, &hi);
    uint32_t cur_row = 0xFFFFFFFFu;
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t slot = idx / I, n0 = idx - slot * I;
        const uint32_t n1 = n0 + (hi - idx) < I ? n0 + (hi - idx) : I;
        idx += n1 - n0;
        const uint32_t eid = table[slot].eid;
        if (eid >= E) continue;
        const plow_bf16* gu = ewt_base(ewt, eid, 0);
        if (!gu) continue;
        const uint32_t row = slot / k;
        if (row != cur_row) {
            const plow_bf16* rr = resid + (size_t)row * H;
            const float inv = gm_invrms(rr, H, eps);
            for (uint32_t h = 0; h < H; h++) xn[h] = gm_round_xn(plow_bf2f(rr[h]) * inv * plow_bf2f(gamma[h]));
            cur_row = row;
        }
        for (uint32_t n = n0; n < n1; n++) {
            const float g = gm_dot_bf16_f32(gu + (size_t)n * H, xn, H);
            const float u = gm_dot_bf16_f32(gu + (size_t)(I + n) * H, xn, H);
            fu[(size_t)slot * I + n] = plow_f2bf(g_gelu_tanh(g) * u);
        }
    }
}

/* 63: t0=part(f32 [B*k][H]) t1=fu([B*k][I]) t2=table t3=ewt  i0=k i1=H i2=I i3=E i5=B.
 * part[slot][h] = gate * (down_e[h] . fu[slot]); a skipped slot writes 0. */
G_K(g_moe_expert_down_gemma) {
    (void)ctx;
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const plow_moe_route* table = PLOW_CPU_TEN(in, T, 2);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 3);
    const uint32_t k = in->i[0], H = in->i[1], I = in->i[2], E = in->i[3], nrow = GM_NB(in->i[5]);
    const uint32_t nslot = nrow * k;
    uint32_t lo, hi;
    g_range(nslot * H, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t slot = idx / H, h0 = idx - slot * H;
        const uint32_t h1 = h0 + (hi - idx) < H ? h0 + (hi - idx) : H;
        idx += h1 - h0;
        const uint32_t eid = table[slot].eid;
        const plow_bf16* dn = eid < E ? ewt_base(ewt, eid, 1) : NULL;
        float* ps = part + (size_t)slot * H;
        if (!dn) { for (uint32_t h = h0; h < h1; h++) ps[h] = 0.0f; continue; }
        const plow_bf16* fr = fu + (size_t)slot * I;
        const float gate = table[slot].gate;
        for (uint32_t h = h0; h < h1; h++) ps[h] = gate * gm_dot_bf16(dn + (size_t)h * I, fr, I);
    }
}

/* Combine + RMSNorm + residual for one row: out = sum*rsqrt(mean(sum^2)+eps)*gamma + res. */
static void gm_combine_norm_row(plow_bf16* out, const float* part, const plow_bf16* res,
                                const plow_bf16* gamma, uint32_t H, uint32_t k, float eps, float* acc) {
    float ss = 0.0f;
    for (uint32_t h = 0; h < H; h++) {
        float a = 0.0f;
        for (uint32_t s = 0; s < k; s++) a += part[(size_t)s * H + h];
        acc[h] = a; ss += a * a;
    }
    const float inv = 1.0f / sqrtf(ss / (float)H + eps);
    for (uint32_t h = 0; h < H; h++)
        out[h] = plow_f2bf(acc[h] * inv * plow_bf2f(gamma[h]) + plow_bf2f(res[h]));
}

/* 70: t0=out([B][H]) t1=part(f32 [B][k][H]) t2=resid([B][H]) t3=gamma  i0=H i1=k i2=B  f0=eps. */
G_K(g_moe_combine_norm_gemma) {
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const float* part = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* resid = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 3);
    const uint32_t H = in->i[0], k = in->i[1], nrow = GM_NB(in->i[2]);
    const float eps = in->fj[0].f;
    if (!ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)H * 4u) return;
    for (uint32_t row = slice; row < nrow; row += GM_NB(nblk))
        gm_combine_norm_row(out + (size_t)row * H, part + (size_t)row * k * H, resid + (size_t)row * H,
                            gamma, H, k, eps, ctx->scratch);
}

/* 73: t0=table([T][k]) t1=resid([T][H]) t2=proj t3=scale t4=pes  i0=H i1=E i2=k i3=T  f0=root f1=eps. */
G_K(g_moe_router_gemma_pf) {
    plow_moe_route* table = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* resid = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* proj = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* scale = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* pes = PLOW_CPU_TEN(in, T, 4);
    const uint32_t H = in->i[0], E = in->i[1], k = in->i[2], nt = GM_NB(in->i[3]);
    const float root = in->fj[0].f, eps = in->fj[1].f;
    if (!ctx || !ctx->scratch || ctx->scratch_bytes < ((size_t)H + E) * 4u) return;
    float* h2 = ctx->scratch;
    float* sc = h2 + H;
    for (uint32_t tok = slice; tok < nt; tok += GM_NB(nblk))
        gm_router_row(table + (size_t)tok * k, resid + (size_t)tok * H, proj, scale, pes, H, E, k, root, eps, h2, sc);
}

/* 75: t0=fu_g([rows][I]) t1=xn2([T][H]) t2=ewt t3=meta(i32) t4=row_token  i0=I i1=H i2=E i5=act.
 * Gathered rows of expert e: [rowoff[e], rowoff[e]+cnt[e]); pad rows (row_token UNUSED) skipped. */
G_K(g_moe_group_glu_gemma_pf) {
    (void)ctx;
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 2);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 3);
    const uint32_t* row_token = PLOW_CPU_TEN(in, T, 4);
    const uint32_t I = in->i[0], H = in->i[1], E = in->i[2], act = in->i[5];
    uint32_t lo, hi;
    g_range(E * I, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t e = idx / I, n0 = idx - e * I;
        const uint32_t n1 = n0 + (hi - idx) < I ? n0 + (hi - idx) : I;
        idx += n1 - n0;
        const uint32_t r0 = (uint32_t)meta[e], cnt = (uint32_t)meta[E + e];
        const plow_bf16* gu = ewt_base(ewt, e, 0);
        if (cnt == 0u || !gu) continue;
        for (uint32_t r = r0; r < r0 + cnt; r++) {
            const uint32_t tok = row_token[r];
            if (tok == PLOW_EXPERT_UNUSED) continue;
            const plow_bf16* xr = x + (size_t)tok * H;
            for (uint32_t n = n0; n < n1; n++) {
                const float g = gm_dot_bf16(gu + (size_t)n * H, xr, H);
                const float u = gm_dot_bf16(gu + (size_t)(I + n) * H, xr, H);
                fu[(size_t)r * I + n] = plow_f2bf(g_act_gate_only(g, act) * u);
            }
        }
    }
}

/* 76: t0=part(f32 [T*k][H]) t1=fu_g t2=ewt t3=meta t4=row_partidx t5=row_gate  i0=H i1=I i2=E.
 * part[row_partidx[r]][h] = row_gate[r] * (down_e[h] . fu_g[r]); pad rows dropped. */
G_K(g_moe_group_down_gemma_pf) {
    (void)ctx;
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 2);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 3);
    const uint32_t* row_partidx = PLOW_CPU_TEN(in, T, 4);
    const float* row_gate = PLOW_CPU_TEN(in, T, 5);
    const uint32_t H = in->i[0], I = in->i[1], E = in->i[2];
    uint32_t lo, hi;
    g_range(E * H, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t e = idx / H, h0 = idx - e * H;
        const uint32_t h1 = h0 + (hi - idx) < H ? h0 + (hi - idx) : H;
        idx += h1 - h0;
        const uint32_t r0 = (uint32_t)meta[e], cnt = (uint32_t)meta[E + e];
        const plow_bf16* dn = ewt_base(ewt, e, 1);
        if (cnt == 0u || !dn) continue;
        for (uint32_t r = r0; r < r0 + cnt; r++) {
            const uint32_t pidx = row_partidx[r];
            if (pidx == PLOW_EXPERT_UNUSED) continue;
            const plow_bf16* fr = fu + (size_t)r * I;
            float* pr = part + (size_t)pidx * H;
            for (uint32_t h = h0; h < h1; h++) pr[h] = row_gate[r] * gm_dot_bf16(dn + (size_t)h * I, fr, I);
        }
    }
}

/* 77: t0=out([T][H]) t1=part(f32 [T][k][H]) t2=h1([T][H]) t3=gamma  i0=H i1=k i2=T  f0=eps. */
G_K(g_moe_combine_norm_gemma_pf) {
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const float* part = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* h1 = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 3);
    const uint32_t H = in->i[0], k = in->i[1], nt = GM_NB(in->i[2]);
    const float eps = in->fj[0].f;
    if (!ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)H * 4u) return;
    for (uint32_t tok = slice; tok < nt; tok += GM_NB(nblk))
        gm_combine_norm_row(out + (size_t)tok * H, part + (size_t)tok * k * H, h1 + (size_t)tok * H, gamma, H, k, eps,
                            ctx->scratch);
}

/* avx512/moe.c — GPT-OSS flat-tensor MXFP4 MoE, AVX-512 BF16 (dev_isa.h ops 147-150).
 *
 * Same slice partition (golden/gptoss.h) and numerics as golden/moe.c. Inner loop is
 * mxfp4_common.h plow_mx_dot_rm: RB packed weight rows x M staged activation rows, 64 weights per
 * step (two vpermw LUT lookups -> two vdpbf16ps -> two broadcast FMAs with the block scales).
 * Decode stages one x row per slot (k slots share a row); prefill stages up to 8 gathered token
 * rows per pass over the expert's weight range, so the dequant is amortised over M rows.
 * Epilogues run 16 outputs at a time: bias in f32, v_glu_pair (act 3 = swiglu_oai), one bf16 round
 * (GLU) / gate * (dot + bias) in f32 (DOWN). Scratch: M * staged_len(K) bf16. */
#include "avx512.h"
#include "../mxfp4_common.h"
#include "../golden/gptoss.h"

extern plow_mx_vlut plow_v_mx_lut; /* avx512/gptoss.c, initialised by v_register_gptoss */

static inline uint32_t row_g(uint32_t n, uint32_t layout) { return layout ? n : 2u * n; }
static inline uint32_t row_u(uint32_t n, uint32_t I, uint32_t layout) { return layout ? I + n : 2u * n + 1u; }

static inline int mx_usable(const PlowCpuCtx* ctx, uint32_t K, uint32_t M) {
    return (K & 31u) == 0u && K / PLOW_MX_BLK <= PLOW_MX_MAX_BLOCKS && ctx && ctx->scratch &&
           ctx->scratch_bytes >= (size_t)M * plow_mx_staged_len(K) * sizeof(plow_bf16);
}

/* g[m][j], u[m][j] for outputs n+j (j < cnt <= 16) of expert rows We/Se against M staged rows. */
static void glu_dots(const uint8_t* We, const uint8_t* Se, size_t ldw, size_t lds, uint32_t I,
                     uint32_t K, uint32_t layout, const plow_bf16* XP, size_t ldx, uint32_t M,
                     uint32_t n, uint32_t cnt, float g[8][16], float u[8][16]) {
    const plow_mx_vlut* v = &plow_v_mx_lut;
    const uint32_t RB = plow_mx_rb_for(M);
    float o[4 * 8];
    if (layout == 0u && RB >= 2u) {
        /* Interleaved rows: RB consecutive rows = RB/2 (gate, up) pairs in one dot call. */
        const uint32_t per = RB / 2u;
        for (uint32_t j = 0; j < cnt; j += per) {
            const uint32_t np = cnt - j < per ? cnt - j : per;
            plow_mx_gemv_rows(v, We + (size_t)(2u * (n + j)) * ldw, ldw, Se + (size_t)(2u * (n + j)) * lds,
                              lds, XP, ldx, K, 2u * np, M, o);
            for (uint32_t p = 0; p < np; p++)
                for (uint32_t m = 0; m < M; m++) {
                    g[m][j + p] = o[(2u * p) * M + m];
                    u[m][j + p] = o[(2u * p + 1u) * M + m];
                }
        }
        return;
    }
    for (uint32_t j = 0; j < cnt; j += RB) {
        const uint32_t rb = cnt - j < RB ? cnt - j : RB;
        /* layout 1: gate rows n+j.. and up rows I+n+j.. are each contiguous; layout 0 with RB==1
         * takes rows 2n and 2n+1 one at a time. */
        const uint32_t rg = row_g(n + j, layout), ru = row_u(n + j, I, layout);
        plow_mx_gemv_rows(v, We + (size_t)rg * ldw, ldw, Se + (size_t)rg * lds, lds, XP, ldx, K, rb, M, o);
        for (uint32_t r = 0; r < rb; r++)
            for (uint32_t m = 0; m < M; m++) g[m][j + r] = o[r * M + m];
        plow_mx_gemv_rows(v, We + (size_t)ru * ldw, ldw, Se + (size_t)ru * lds, lds, XP, ldx, K, rb, M, o);
        for (uint32_t r = 0; r < rb; r++)
            for (uint32_t m = 0; m < M; m++) u[m][j + r] = o[r * M + m];
    }
}

/* fu[n .. n+cnt) = pair(g + bias_g, u + bias_u) for one output row. */
static inline void glu_epilogue(plow_bf16* out, const float* g, const float* u, const plow_bf16* be,
                                uint32_t I, uint32_t layout, uint32_t n, uint32_t cnt, uint32_t act,
                                float f0, float f1) {
    const __mmask16 mk = cnt == 16u ? 0xFFFF : v_tail16(cnt);
    __m512 gv = _mm512_maskz_loadu_ps(mk, g), uv = _mm512_maskz_loadu_ps(mk, u);
    if (be) {
        float bg[16], bu[16];
        for (uint32_t j = 0; j < cnt; j++) {
            bg[j] = plow_bf2f(be[row_g(n + j, layout)]);
            bu[j] = plow_bf2f(be[row_u(n + j, I, layout)]);
        }
        gv = _mm512_add_ps(gv, _mm512_maskz_loadu_ps(mk, bg));
        uv = _mm512_add_ps(uv, _mm512_maskz_loadu_ps(mk, bu));
    }
    v_store_bf16_mask(out + n, mk, v_glu_pair(gv, uv, act, f0, f1));
}

/* t0=fu t1=x t2=table t3=W_gu t4=S_gu t5=bias_gu?  i0=k i1=I i2=K i3=n_exp i4=layout i5=act
 * i6=n_batch(0 = 1)  f0/f1 (golden g_moe_glu_mx). */
V_K(v_moe_glu_mx) {
    const uint32_t k = in->i[0], I = in->i[1], K = in->i[2], E = in->i[3], layout = in->i[4];
    const uint32_t act = in->i[5], B = in->i[6] ? in->i[6] : 1u;
    if (!mx_usable(ctx, K, 1u) || k == 0u) { g_moe_glu_mx(in, slice, nblk, T, ctx); return; }
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_moe_route* tab = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    const size_t N2 = 2u * I, ldw = K / 2u, lds = K / PLOW_MX_BLK, ldx = plow_mx_staged_len(K);
    plow_bf16* XP = ctx->scratch;
    uint32_t staged = UINT32_MAX;
    float g[8][16], u[8][16];
    uint32_t lo, hi;
    g_range(B * k * I, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t s = idx / I, n0 = idx % I;
        const uint32_t n1 = n0 + (hi - idx) < I ? n0 + (hi - idx) : I;
        idx += n1 - n0;
        const uint32_t eid = tab[s].eid;
        if (eid >= E) continue;
        if (staged != s / k) {
            staged = s / k;
            plow_mx_stage_x(&plow_v_mx_lut, XP, x + (size_t)staged * K, K);
        }
        const uint8_t* We = W + (size_t)eid * N2 * ldw;
        const uint8_t* Se = S + (size_t)eid * N2 * lds;
        const plow_bf16* be = bias ? bias + (size_t)eid * N2 : NULL;
        plow_bf16* out = fu + (size_t)s * I;
        for (uint32_t n = n0; n < n1;) {
            const uint32_t cnt = n1 - n < 16u ? n1 - n : 16u;
            glu_dots(We, Se, ldw, lds, I, K, layout, XP, ldx, 1u, n, cnt, g, u);
            glu_epilogue(out, g[0], u[0], be, I, layout, n, cnt, act, f0, f1);
            n += cnt;
        }
    }
}

/* part[h .. h+cnt) = gate * (W rows h.. . XP + bias) for one slot / gathered row. */
static void down_rows(const uint8_t* We, const uint8_t* Se, size_t ldw, size_t lds, uint32_t I,
                      const plow_bf16* XP, size_t ldx, uint32_t h0, uint32_t h1, const plow_bf16* be,
                      float gate, float* pr) {
    float o[4];
    for (uint32_t h = h0; h < h1;) {
        const uint32_t rb = h1 - h < 4u ? h1 - h : 4u;
        plow_mx_gemv_rows(&plow_v_mx_lut, We + (size_t)h * ldw, ldw, Se + (size_t)h * lds, lds, XP, ldx, I,
                          rb, 1u, o);
        for (uint32_t r = 0; r < rb; r++) pr[h + r] = gate * (o[r] + (be ? plow_bf2f(be[h + r]) : 0.0f));
        h += rb;
    }
}

/* t0=part(f32) t1=fu t2=table t3=W_d t4=S_d t5=bias_d?  i0=k i1=H i2=I i3=n_exp i6=n_batch. */
V_K(v_moe_down_mx) {
    const uint32_t k = in->i[0], H = in->i[1], I = in->i[2], E = in->i[3];
    const uint32_t B = in->i[6] ? in->i[6] : 1u;
    if (!mx_usable(ctx, I, 1u) || k == 0u) { g_moe_down_mx(in, slice, nblk, T, ctx); return; }
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const plow_moe_route* tab = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const size_t ldw = I / 2u, lds = I / PLOW_MX_BLK, ldx = plow_mx_staged_len(I);
    plow_bf16* XP = ctx->scratch;
    uint32_t staged = UINT32_MAX;
    uint32_t lo, hi;
    g_range(B * k * H, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t s = idx / H, h0 = idx % H;
        const uint32_t h1 = h0 + (hi - idx) < H ? h0 + (hi - idx) : H;
        idx += h1 - h0;
        float* pr = part + (size_t)s * H;
        const uint32_t eid = tab[s].eid;
        if (eid >= E) {
            memset(pr + h0, 0, (size_t)(h1 - h0) * sizeof(float));
            continue;
        }
        if (staged != s) {
            staged = s;
            plow_mx_stage_x(&plow_v_mx_lut, XP, fu + (size_t)s * I, I);
        }
        down_rows(W + (size_t)eid * H * ldw, S + (size_t)eid * H * lds, ldw, lds, I, XP, ldx, h0, h1,
                  bias ? bias + (size_t)eid * H : NULL, tab[s].gate, pr);
    }
}

/* Stage up to 8 real rows of expert segment [r, rend) whose `key[r]` != PLOW_EXPERT_UNUSED:
 * XP row m <- src[src_row(r)]; rows[m] = r. Returns M and advances *r. */
static uint32_t stage_segment(plow_bf16* XP, size_t ldx, uint32_t K, const plow_bf16* src,
                              const uint32_t* key, int key_is_src_row, uint32_t* r, uint32_t rend,
                              uint32_t rows[8]) {
    uint32_t M = 0;
    while (*r < rend && M < 8u) {
        const uint32_t rr = (*r)++;
        if (key[rr] == PLOW_EXPERT_UNUSED) continue;
        const uint32_t srow = key_is_src_row ? key[rr] : rr;
        plow_mx_stage_x(&plow_v_mx_lut, XP + (size_t)M * ldx, src + (size_t)srow * K, K);
        rows[M++] = rr;
    }
    return M;
}

/* PREFILL GLU: t0=fu_g t1=xn2 t2=W_gu t3=S_gu t4=meta t5=row_token t6=bias_gu?
 * i0=I i1=K i2=n_exp i3=layout i5=act  f0/f1 (golden g_moe_glu_mx_pf). */
V_K(v_moe_glu_mx_pf) {
    const uint32_t I = in->i[0], K = in->i[1], E = in->i[2], layout = in->i[3], act = in->i[5];
    if (!mx_usable(ctx, K, 8u)) { g_moe_glu_mx_pf(in, slice, nblk, T, ctx); return; }
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 3);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 4);
    const uint32_t* row_token = PLOW_CPU_TEN(in, T, 5);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 6);
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    const size_t N2 = 2u * I, ldw = K / 2u, lds = K / PLOW_MX_BLK, ldx = plow_mx_staged_len(K);
    plow_bf16* XP = ctx->scratch;
    float g[8][16], u[8][16];
    uint32_t rows[8];
    uint32_t lo, hi;
    g_range(E * I, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t e = idx / I, n0 = idx % I;
        const uint32_t n1 = n0 + (hi - idx) < I ? n0 + (hi - idx) : I;
        idx += n1 - n0;
        const uint32_t r0 = (uint32_t)meta[e], rend = r0 + (uint32_t)meta[E + e];
        const uint8_t* We = W + (size_t)e * N2 * ldw;
        const uint8_t* Se = S + (size_t)e * N2 * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * N2 : NULL;
        for (uint32_t r = r0; r < rend;) {
            const uint32_t M = stage_segment(XP, ldx, K, x, row_token, 1, &r, rend, rows);
            if (!M) continue;
            for (uint32_t n = n0; n < n1;) {
                const uint32_t cnt = n1 - n < 16u ? n1 - n : 16u;
                glu_dots(We, Se, ldw, lds, I, K, layout, XP, ldx, M, n, cnt, g, u);
                for (uint32_t m = 0; m < M; m++)
                    glu_epilogue(fu + (size_t)rows[m] * I, g[m], u[m], be, I, layout, n, cnt, act, f0, f1);
                n += cnt;
            }
        }
    }
}

/* PREFILL DOWN: t0=part t1=fu_g t2=W_d t3=S_d t4=meta t5=bias_d? t6=row_partidx t7=row_gate
 * i0=H i1=I i2=n_exp (golden g_moe_down_mx_pf). */
V_K(v_moe_down_mx_pf) {
    const uint32_t H = in->i[0], I = in->i[1], E = in->i[2];
    if (!mx_usable(ctx, I, 8u)) { g_moe_down_mx_pf(in, slice, nblk, T, ctx); return; }
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 3);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const uint32_t* row_partidx = PLOW_CPU_TEN(in, T, 6);
    const float* row_gate = PLOW_CPU_TEN(in, T, 7);
    const size_t ldw = I / 2u, lds = I / PLOW_MX_BLK, ldx = plow_mx_staged_len(I);
    plow_bf16* XP = ctx->scratch;
    uint32_t rows[8];
    float o[4 * 8];
    uint32_t lo, hi;
    g_range(E * H, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t e = idx / H, h0 = idx % H;
        const uint32_t h1 = h0 + (hi - idx) < H ? h0 + (hi - idx) : H;
        idx += h1 - h0;
        const uint32_t r0 = (uint32_t)meta[e], rend = r0 + (uint32_t)meta[E + e];
        const uint8_t* We = W + (size_t)e * H * ldw;
        const uint8_t* Se = S + (size_t)e * H * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * H : NULL;
        for (uint32_t r = r0; r < rend;) {
            const uint32_t M = stage_segment(XP, ldx, I, fu, row_partidx, 0, &r, rend, rows);
            if (!M) continue;
            const uint32_t RB = plow_mx_rb_for(M);
            for (uint32_t h = h0; h < h1;) {
                const uint32_t rb = h1 - h < RB ? h1 - h : RB;
                plow_mx_gemv_rows(&plow_v_mx_lut, We + (size_t)h * ldw, ldw, Se + (size_t)h * lds, lds, XP,
                                  ldx, I, rb, M, o);
                for (uint32_t m = 0; m < M; m++) {
                    float* pr = part + (size_t)row_partidx[rows[m]] * H;
                    const float gate = row_gate[rows[m]];
                    for (uint32_t rr = 0; rr < rb; rr++)
                        pr[h + rr] = gate * (o[rr * M + m] + (be ? plow_bf2f(be[h + rr]) : 0.0f));
                }
                h += rb;
            }
        }
    }
}

/* ---- Batched decode grouped by expert (147/148 at B >= 2) --------------------------------------
 * The per-slot decode kernels stream + dequantize an expert once per SLOT, so a rung-8 step costs
 * ~8x rung 1. Here the B*k slots are sorted by expert and each selected expert's rows are
 * dequantized once for all its M <= 8 slots (a row never selects an expert twice, so M <= B).
 * Per-slot outputs are unchanged (fu[slot], part[slot]); every slice builds the same grouping, so
 * slice ownership (g_range over distinct experts x columns) stays deterministic. Sentinel slots
 * (eid >= E) are left out: GLU writes nothing, DOWN zeroes them (slice 0). An AMX-tile variant of
 * this (weights staged into A tiles) measured 40 % SLOWER in serving: most experts carry one row,
 * where the tile staging costs ~2x this fused even/odd-nibble dequant-dot. */
#define MXB_MAX_SLOTS 256u
typedef struct {
    uint32_t nd;
    uint32_t eid[MXB_MAX_SLOTS];
    uint32_t off[MXB_MAX_SLOTS + 1];
    uint32_t slot[MXB_MAX_SLOTS];
} mxb_groups;

static int mxb_group(const plow_moe_route* tab, uint32_t nslot, uint32_t E, mxb_groups* g) {
    if (nslot > MXB_MAX_SLOTS) return 0;
    uint32_t n = 0;
    for (uint32_t s = 0; s < nslot; s++) {
        if (tab[s].eid >= E) continue;
        uint32_t i = n++;
        while (i && tab[g->slot[i - 1]].eid > tab[s].eid) {
            g->slot[i] = g->slot[i - 1];
            i--;
        }
        g->slot[i] = s;
    }
    g->nd = 0;
    for (uint32_t i = 0; i < n;) {
        const uint32_t e = tab[g->slot[i]].eid;
        g->eid[g->nd] = e;
        g->off[g->nd] = i;
        while (i < n && tab[g->slot[i]].eid == e) i++;
        g->nd++;
    }
    g->off[g->nd] = n;
    return 1;
}

/* Stage up to 8 slots of one expert group: XP row m <- src row (slot / src_div). */
static uint32_t mxb_stage(plow_bf16* XP, size_t ldx, uint32_t K, const plow_bf16* src, uint32_t src_div,
                          const mxb_groups* g, uint32_t* i, uint32_t iend, uint32_t slots[8]) {
    uint32_t M = 0;
    while (*i < iend && M < 8u) {
        const uint32_t s = g->slot[(*i)++];
        plow_mx_stage_x(&plow_v_mx_lut, XP + (size_t)M * ldx, src + (size_t)(s / src_div) * K, K);
        slots[M++] = s;
    }
    return M;
}

/* 147 at B >= 2 (slot map as v_moe_glu_mx). */
V_K(v_moe_glu_mx_b) {
    const uint32_t k = in->i[0], I = in->i[1], K = in->i[2], E = in->i[3], layout = in->i[4];
    const uint32_t act = in->i[5], B = in->i[6] ? in->i[6] : 1u;
    const plow_moe_route* tab = PLOW_CPU_TEN(in, T, 2);
    mxb_groups g;
    if (B < 2u || k == 0u || !mx_usable(ctx, K, 8u) || !mxb_group(tab, B * k, E, &g)) {
        v_moe_glu_mx(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    const size_t N2 = 2u * I, ldw = K / 2u, lds = K / PLOW_MX_BLK, ldx = plow_mx_staged_len(K);
    plow_bf16* XP = ctx->scratch;
    float gg[8][16], uu[8][16];
    uint32_t slots[8];
    uint32_t lo, hi;
    g_range(g.nd * I, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t d = idx / I, n0 = idx - d * I;
        const uint32_t n1 = n0 + (hi - idx) < I ? n0 + (hi - idx) : I;
        idx += n1 - n0;
        const uint32_t e = g.eid[d];
        const uint8_t* We = W + (size_t)e * N2 * ldw;
        const uint8_t* Se = S + (size_t)e * N2 * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * N2 : NULL;
        for (uint32_t i = g.off[d]; i < g.off[d + 1];) {
            const uint32_t M = mxb_stage(XP, ldx, K, x, k, &g, &i, g.off[d + 1], slots);
            for (uint32_t n = n0; n < n1;) {
                const uint32_t cnt = n1 - n < 16u ? n1 - n : 16u;
                glu_dots(We, Se, ldw, lds, I, K, layout, XP, ldx, M, n, cnt, gg, uu);
                for (uint32_t m = 0; m < M; m++)
                    glu_epilogue(fu + (size_t)slots[m] * I, gg[m], uu[m], be, I, layout, n, cnt, act, f0, f1);
                n += cnt;
            }
        }
    }
}

/* 148 at B >= 2 (slot map as v_moe_down_mx). */
V_K(v_moe_down_mx_b) {
    const uint32_t k = in->i[0], H = in->i[1], I = in->i[2], E = in->i[3];
    const uint32_t B = in->i[6] ? in->i[6] : 1u;
    const plow_moe_route* tab = PLOW_CPU_TEN(in, T, 2);
    mxb_groups g;
    if (B < 2u || k == 0u || !mx_usable(ctx, I, 8u) || !mxb_group(tab, B * k, E, &g)) {
        v_moe_down_mx(in, slice, nblk, T, ctx);
        return;
    }
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const size_t ldw = I / 2u, lds = I / PLOW_MX_BLK, ldx = plow_mx_staged_len(I);
    plow_bf16* XP = ctx->scratch;
    if (slice == 0)
        for (uint32_t s = 0; s < B * k; s++)
            if (tab[s].eid >= E) memset(part + (size_t)s * H, 0, (size_t)H * sizeof(float));
    uint32_t slots[8];
    float o[4 * 8];
    uint32_t lo, hi;
    g_range(g.nd * H, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t d = idx / H, h0 = idx - d * H;
        const uint32_t h1 = h0 + (hi - idx) < H ? h0 + (hi - idx) : H;
        idx += h1 - h0;
        const uint32_t e = g.eid[d];
        const uint8_t* We = W + (size_t)e * H * ldw;
        const uint8_t* Se = S + (size_t)e * H * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * H : NULL;
        for (uint32_t i = g.off[d]; i < g.off[d + 1];) {
            const uint32_t M = mxb_stage(XP, ldx, I, fu, 1u, &g, &i, g.off[d + 1], slots);
            const uint32_t RB = plow_mx_rb_for(M);
            for (uint32_t h = h0; h < h1;) {
                const uint32_t rb = h1 - h < RB ? h1 - h : RB;
                plow_mx_gemv_rows(&plow_v_mx_lut, We + (size_t)h * ldw, ldw, Se + (size_t)h * lds, lds, XP, ldx, I,
                                  rb, M, o);
                for (uint32_t m = 0; m < M; m++) {
                    float* pr = part + (size_t)slots[m] * H;
                    const float gate = tab[slots[m]].gate;
                    for (uint32_t rr = 0; rr < rb; rr++)
                        pr[h + rr] = gate * (o[rr * M + m] + (be ? plow_bf2f(be[h + rr]) : 0.0f));
                }
                h += rb;
            }
        }
    }
}

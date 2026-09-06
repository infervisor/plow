/* avx512/moe_gemma.c — Gemma-4 26B-A4B hybrid MoE, AVX-512 BF16 tier: decode 69 (router score),
 * 71 (fused-norm expert GLU), 63 (expert down) — per slot at B = 1, grouped by expert at B >= 2 —
 * and prefill 75/76 (grouped expert GLU / down).
 * Same slice partitions as golden/moe_gemma.c (contiguous g_range over the flat item span); the
 * expert weights are reached through the host-filled `ewt` table. Router top-k, align and the
 * combine-norms stay golden (per-row scalar work). */
#include "avx512.h"
#include "../golden/gptoss.h"

#define PF_W 512u /* bf16 elements = 1 KiB ahead of every weight row */
#define UNROLL _Pragma("GCC unroll 8")

static inline const plow_bf16* ewt_base(const uint64_t* ewt, uint32_t eid, uint32_t which) {
    return (const plow_bf16*)(uintptr_t)ewt[(size_t)eid * 2u + which];
}

/* out[r*M + m] = Wr[r] . X[m] (bf16 x bf16, vdpbf16ps), RB*M <= 16, K % 32 == 0. RB/M constant per
 * instantiation and every loop force-unrolled: without UNROLL gcc -O2 keeps acc[][] on the stack
 * (load/dpbf16/store per accumulator per step). Weight rows are arbitrary pointers so the GLU can
 * pair gate n, gate n+1, up n, up n+1 in one pass. */
static inline __attribute__((always_inline)) void dot_rm(const plow_bf16* const* Wr, const plow_bf16* X,
                                                         size_t ldx, uint32_t K, const uint32_t RB,
                                                         const uint32_t M, float* out) {
    __m512 acc[8][8];
    UNROLL
    for (uint32_t r = 0; r < RB; r++)
        UNROLL
        for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_setzero_ps();
    for (uint32_t k = 0; k < K; k += 32u) {
        __m512bh xv[8];
        UNROLL
        for (uint32_t m = 0; m < M; m++) xv[m] = (__m512bh)_mm512_loadu_si512((const void*)(X + m * ldx + k));
        UNROLL
        for (uint32_t r = 0; r < RB; r++) {
            const plow_bf16* w = Wr[r] + k;
            _mm_prefetch((const char*)(w + PF_W), _MM_HINT_T0);
            const __m512bh wv = (__m512bh)_mm512_loadu_si512((const void*)w);
            UNROLL
            for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_dpbf16_ps(acc[r][m], wv, xv[m]);
        }
    }
    UNROLL
    for (uint32_t r = 0; r < RB; r++)
        UNROLL
        for (uint32_t m = 0; m < M; m++) out[r * M + m] = _mm512_reduce_add_ps(acc[r][m]);
}
#define DOT_CASE(RB_, M_) case (RB_) * 16 + (M_): dot_rm(Wr, X, ldx, K, RB_, M_, out); break;
static void dots_p(const plow_bf16* const* Wr, const plow_bf16* X, size_t ldx, uint32_t K, uint32_t RB,
                   uint32_t M, float* out) {
    switch (RB * 16 + M) {
        DOT_CASE(8, 1) DOT_CASE(8, 2)
        DOT_CASE(4, 1) DOT_CASE(4, 2) DOT_CASE(4, 3) DOT_CASE(4, 4)
        DOT_CASE(2, 1) DOT_CASE(2, 2) DOT_CASE(2, 3) DOT_CASE(2, 4)
        DOT_CASE(2, 5) DOT_CASE(2, 6) DOT_CASE(2, 7) DOT_CASE(2, 8)
        DOT_CASE(1, 1) DOT_CASE(1, 2) DOT_CASE(1, 3) DOT_CASE(1, 4)
        DOT_CASE(1, 5) DOT_CASE(1, 6) DOT_CASE(1, 7) DOT_CASE(1, 8)
        default:
            for (uint32_t r = 0; r < RB; r++)
                for (uint32_t m = 0; m < M; m++) dot_rm(Wr + r, X + m * ldx, ldx, K, 1, 1, out + r * M + m);
    }
}
/* Rows W, W+ldw, ... (RB <= 8). */
static inline void dots(const plow_bf16* W, size_t ldw, const plow_bf16* X, size_t ldx, uint32_t K, uint32_t RB,
                        uint32_t M, float* out) {
    const plow_bf16* Wr[8];
    for (uint32_t r = 0; r < RB; r++) Wr[r] = W + r * ldw;
    dots_p(Wr, X, ldx, K, RB, M, out);
}
static inline uint32_t rb_for(uint32_t M) { return M <= 4u ? 4u : 2u; }

/* 4 bf16 weight rows against ONE f32 activation row (the fused-norm xn is kept in f32, as the AMD
 * body does: a bf16 xn measured 8x relative error on small act(g)*u outputs). K % 16 == 0. */
static inline void dotf_r4(const plow_bf16* W0, const plow_bf16* W1, const plow_bf16* W2, const plow_bf16* W3,
                           const float* x, uint32_t K, float* out) {
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t k = 0; k < K; k += 16u) {
        const __m512 xv = _mm512_loadu_ps(x + k);
        _mm_prefetch((const char*)(W0 + k + PF_W), _MM_HINT_T0);
        _mm_prefetch((const char*)(W1 + k + PF_W), _MM_HINT_T0);
        _mm_prefetch((const char*)(W2 + k + PF_W), _MM_HINT_T0);
        _mm_prefetch((const char*)(W3 + k + PF_W), _MM_HINT_T0);
        a0 = _mm512_fmadd_ps(v_load_bf16(W0 + k), xv, a0);
        a1 = _mm512_fmadd_ps(v_load_bf16(W1 + k), xv, a1);
        a2 = _mm512_fmadd_ps(v_load_bf16(W2 + k), xv, a2);
        a3 = _mm512_fmadd_ps(v_load_bf16(W3 + k), xv, a3);
    }
    out[0] = _mm512_reduce_add_ps(a0); out[1] = _mm512_reduce_add_ps(a1);
    out[2] = _mm512_reduce_add_ps(a2); out[3] = _mm512_reduce_add_ps(a3);
}

#include <stdlib.h>
static int v_xn_bf16(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PLOW_MOE_XN_BF16"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v;
}
static inline float row_invrms(const plow_bf16* r, uint32_t H, float eps) {
    return g_rsqrt(v_row_ss(r, H) / (float)H + eps);
}

/* 69: t0=score(f32 [B][E]) t1=resid t2=proj([E][H]) t3=scale  i0=H i1=E i2=B  f0=root f1=eps.
 * h2 = resid * invrms * scale * root staged once per row in scratch (f32), 4 experts per pass. */
V_K(v_moe_router_gemma_score_fast) {
    const uint32_t H = in->i[0], E = in->i[1], nrow = in->i[2] ? in->i[2] : 1u;
    if ((H & 15u) || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)H * 4u) {
        g_moe_router_gemma_score_fast(in, slice, nblk, T, ctx);
        return;
    }
    float* score = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* resid = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* proj = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* scale = PLOW_CPU_TEN(in, T, 3);
    const float root = in->fj[0].f, eps = in->fj[1].f;
    float* h2 = ctx->scratch;
    uint32_t lo, hi;
    g_range(nrow * E, slice, nblk, &lo, &hi);
    uint32_t cur = 0xFFFFFFFFu;
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t row = idx / E, e0 = idx - row * E;
        const uint32_t e1 = e0 + (hi - idx) < E ? e0 + (hi - idx) : E;
        idx += e1 - e0;
        if (row != cur) {
            const plow_bf16* rr = resid + (size_t)row * H;
            const __m512 sc = _mm512_set1_ps(row_invrms(rr, H, eps) * root);
            for (uint32_t h = 0; h < H; h += 16u)
                _mm512_storeu_ps(h2 + h, _mm512_mul_ps(_mm512_mul_ps(v_load_bf16(rr + h), sc), v_load_bf16(scale + h)));
            cur = row;
        }
        float* srow = score + (size_t)row * E;
        uint32_t e = e0;
        for (; e + 4u <= e1; e += 4u)
            dotf_r4(proj + (size_t)e * H, proj + (size_t)(e + 1) * H, proj + (size_t)(e + 2) * H, proj + (size_t)(e + 3) * H, h2, H, srow + e);
        for (; e < e1; e++) {
            const plow_bf16* pr = proj + (size_t)e * H;
            float o[4];
            dotf_r4(pr, pr, pr, pr, h2, H, o);
            srow[e] = o[0];
        }
    }
}

/* 71: t0=fu([B*k][I]) t1=resid([B][H]) t2=table t3=ewt t4=gamma  i0=k i1=I i2=H i3=E i5=B  f0=eps.
 * xn (f32) staged per row; per output channel pair: gate n, gate n+1, up n, up n+1 in one pass. */
V_K(v_moe_expert_glu_norm_gemma) {
    const uint32_t k = in->i[0], I = in->i[1], H = in->i[2], E = in->i[3], nrow = in->i[5] ? in->i[5] : 1u;
    if ((H & 15u) || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)H * 4u) {
        g_moe_expert_glu_norm_gemma(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* resid = PLOW_CPU_TEN(in, T, 1);
    const plow_moe_route* table = PLOW_CPU_TEN(in, T, 2);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 4);
    const float eps = in->fj[0].f;
    float* xn = ctx->scratch;
    const uint32_t nslot = nrow * k;
    uint32_t lo, hi;
    g_range(nslot * I, slice, nblk, &lo, &hi);
    uint32_t cur = 0xFFFFFFFFu;
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t slot = idx / I, n0 = idx - slot * I;
        const uint32_t n1 = n0 + (hi - idx) < I ? n0 + (hi - idx) : I;
        idx += n1 - n0;
        const uint32_t eid = table[slot].eid;
        if (eid >= E) continue;
        const plow_bf16* gu = ewt_base(ewt, eid, 0);
        if (!gu) continue;
        const uint32_t row = slot / k;
        if (row != cur) {
            const plow_bf16* rr = resid + (size_t)row * H;
            const __m512 inv = _mm512_set1_ps(row_invrms(rr, H, eps));
            const int rnd = v_xn_bf16();
            for (uint32_t h = 0; h < H; h += 16u) {
                __m512 v = _mm512_mul_ps(_mm512_mul_ps(v_load_bf16(rr + h), inv), v_load_bf16(gamma + h));
                if (rnd) v = v_round_bf16(v);
                _mm512_storeu_ps(xn + h, v);
            }
            cur = row;
        }
        plow_bf16* frow = fu + (size_t)slot * I;
        uint32_t n = n0;
        for (; n + 2u <= n1; n += 2u) {
            float o[4];
            dotf_r4(gu + (size_t)n * H, gu + (size_t)(n + 1) * H, gu + (size_t)(I + n) * H, gu + (size_t)(I + n + 1) * H, xn, H, o);
            frow[n] = plow_f2bf(g_gelu_tanh(o[0]) * o[2]);
            frow[n + 1] = plow_f2bf(g_gelu_tanh(o[1]) * o[3]);
        }
        if (n < n1) {
            float o[4];
            dotf_r4(gu + (size_t)n * H, gu + (size_t)n * H, gu + (size_t)(I + n) * H, gu + (size_t)(I + n) * H, xn, H, o);
            frow[n] = plow_f2bf(g_gelu_tanh(o[0]) * o[2]);
        }
    }
}

/* pr[m][h] = gate[m] * (dn[h] . XP[m]) for h in [h0, h1), M <= 8 staged rows. While M <= 2 the rows
 * are streamed as two interleaved runs (h.. and h+half.., RB = 8): a slice's down rows are one
 * contiguous 1.4 KB-row run, and one sequential stream per thread leaves DRAM parallelism unused. */
static void down_cols(const plow_bf16* dn, uint32_t I, const plow_bf16* XP, uint32_t M, uint32_t h0, uint32_t h1,
                      float* const* pr, const float* gate) {
    float o[4 * 8];
    uint32_t h = h0;
    if (M <= 2u) {
        const uint32_t half = ((h1 - h0) / 2u) & ~3u;
        for (; h + 4u <= h0 + half; h += 4u) {
            const plow_bf16* Wr[8];
            for (uint32_t r = 0; r < 4u; r++) { Wr[r] = dn + (size_t)(h + r) * I; Wr[4 + r] = dn + (size_t)(h + half + r) * I; }
            dots_p(Wr, XP, I, I, 8u, M, o);
            for (uint32_t m = 0; m < M; m++)
                for (uint32_t r = 0; r < 4u; r++) { pr[m][h + r] = gate[m] * o[r * M + m]; pr[m][h + half + r] = gate[m] * o[(4 + r) * M + m]; }
        }
        if (half) h = h0 + 2u * half;
    }
    const uint32_t RB = rb_for(M);
    while (h < h1) {
        const uint32_t rb = h1 - h < RB ? h1 - h : RB;
        dots(dn + (size_t)h * I, I, XP, I, I, rb, M, o);
        for (uint32_t m = 0; m < M; m++)
            for (uint32_t rr = 0; rr < rb; rr++) pr[m][h + rr] = gate[m] * o[rr * M + m];
        h += rb;
    }
}

/* 63: t0=part(f32 [B*k][H]) t1=fu([B*k][I]) t2=table t3=ewt  i0=k i1=H i2=I i3=E i5=B. */
V_K(v_moe_expert_down_gemma) {
    const uint32_t k = in->i[0], H = in->i[1], I = in->i[2], E = in->i[3], nrow = in->i[5] ? in->i[5] : 1u;
    if (I & 31u) { g_moe_expert_down_gemma(in, slice, nblk, T, ctx); return; }
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const plow_moe_route* table = PLOW_CPU_TEN(in, T, 2);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 3);
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
        down_cols(dn, I, fu + (size_t)slot * I, 1u, h0, h1, &ps, &table[slot].gate);
    }
}

/* Gather up to 8 live rows of an expert segment into XP ([M][K] bf16, contiguous). */
static uint32_t gather_rows(plow_bf16* XP, uint32_t K, const plow_bf16* src, const uint32_t* key, int key_is_src,
                            uint32_t* r, uint32_t rend, uint32_t rows[8]) {
    uint32_t M = 0;
    while (*r < rend && M < 8u) {
        const uint32_t rr = (*r)++;
        if (key[rr] == PLOW_EXPERT_UNUSED) continue;
        const uint32_t s = key_is_src ? key[rr] : rr;
        memcpy(XP + (size_t)M * K, src + (size_t)s * K, (size_t)K * 2u);
        rows[M++] = rr;
    }
    return M;
}

/* 75: t0=fu_g([rows][I]) t1=xn2([T][H]) t2=ewt t3=meta t4=row_token  i0=I i1=H i2=E i5=act. */
V_K(v_moe_group_glu_gemma_pf) {
    const uint32_t I = in->i[0], H = in->i[1], E = in->i[2], act = in->i[5];
    if ((H & 31u) || act > 1u || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)8 * H * 2u) {
        g_moe_group_glu_gemma_pf(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 2);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 3);
    const uint32_t* row_token = PLOW_CPU_TEN(in, T, 4);
    plow_bf16* XP = ctx->scratch;
    uint32_t rows[8];
    float o[2 * 8];
    uint32_t lo, hi;
    g_range(E * I, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t e = idx / I, n0 = idx - e * I;
        const uint32_t n1 = n0 + (hi - idx) < I ? n0 + (hi - idx) : I;
        idx += n1 - n0;
        const uint32_t r0 = (uint32_t)meta[e], rend = r0 + (uint32_t)meta[E + e];
        const plow_bf16* gu = ewt_base(ewt, e, 0);
        if (rend == r0 || !gu) continue;
        for (uint32_t r = r0; r < rend;) {
            const uint32_t M = gather_rows(XP, H, x, row_token, 1, &r, rend, rows);
            if (!M) continue;
            for (uint32_t n = n0; n < n1; n++) {
                /* RB = 2 with row stride I*H: row 0 = gate n, row 1 = up n, against the M rows. */
                dots(gu + (size_t)n * H, (size_t)I * H, XP, H, H, 2u, M, o);
                for (uint32_t m = 0; m < M; m++)
                    fu[(size_t)rows[m] * I + n] = plow_f2bf(g_act_gate_only(o[m], act) * o[M + m]);
            }
        }
    }
}

/* 76: t0=part(f32 [T*k][H]) t1=fu_g t2=ewt t3=meta t4=row_partidx t5=row_gate  i0=H i1=I i2=E. */
V_K(v_moe_group_down_gemma_pf) {
    const uint32_t H = in->i[0], I = in->i[1], E = in->i[2];
    if ((I & 31u) || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)8 * I * 2u) {
        g_moe_group_down_gemma_pf(in, slice, nblk, T, ctx);
        return;
    }
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 2);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 3);
    const uint32_t* row_partidx = PLOW_CPU_TEN(in, T, 4);
    const float* row_gate = PLOW_CPU_TEN(in, T, 5);
    plow_bf16* XP = ctx->scratch;
    uint32_t rows[8];
    float o[4 * 8];
    uint32_t lo, hi;
    g_range(E * H, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t e = idx / H, h0 = idx - e * H;
        const uint32_t h1 = h0 + (hi - idx) < H ? h0 + (hi - idx) : H;
        idx += h1 - h0;
        const uint32_t r0 = (uint32_t)meta[e], rend = r0 + (uint32_t)meta[E + e];
        const plow_bf16* dn = ewt_base(ewt, e, 1);
        if (rend == r0 || !dn) continue;
        for (uint32_t r = r0; r < rend;) {
            const uint32_t M = gather_rows(XP, I, fu, row_partidx, 0, &r, rend, rows);
            if (!M) continue;
            const uint32_t RB = rb_for(M);
            for (uint32_t h = h0; h < h1;) {
                const uint32_t rb = h1 - h < RB ? h1 - h : RB;
                dots(dn + (size_t)h * I, I, XP, I, I, rb, M, o);
                for (uint32_t m = 0; m < M; m++) {
                    float* pr = part + (size_t)row_partidx[rows[m]] * H;
                    const float gate = row_gate[rows[m]];
                    for (uint32_t rr = 0; rr < rb; rr++) pr[h + rr] = gate * o[rr * M + m];
                }
                h += rb;
            }
        }
    }
}

/* ---- Batched decode grouped by expert (71/63 at B >= 2; design as avx512/moe.c mxb_*) ----------
 * The per-slot kernels stream an expert once per SLOT (up to B*k passes). Here the B*k slots are
 * sorted by expert, each expert's live slots (M <= 8 per pass; a row never selects an expert twice,
 * so M <= B) are staged as bf16 rows and the expert's weights are streamed once with RB x M register
 * dots. The GLU input xn is rounded to bf16 for vdpbf16ps (the prefill 75 path's xn2 numerics);
 * the B = 1 path keeps f32 xn. Outputs are per slot as before; every slice builds the same
 * grouping, so slice ownership (g_range over the weighted expert x column span) is deterministic.
 * Sentinel slots (eid >= E, or expert not resident): GLU writes nothing, DOWN zeroes (slice 0).
 * PLOW_MOE_GEMMA_NOGROUP=1 forces the per-slot path (A/B). */
#define GMB_MAX_SLOTS 256u
typedef struct {
    uint32_t nd;
    uint32_t eid[GMB_MAX_SLOTS];
    uint32_t off[GMB_MAX_SLOTS + 1];
    uint32_t slot[GMB_MAX_SLOTS];
} gmb_groups;

static int gmb_nogroup(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PLOW_MOE_GEMMA_NOGROUP"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v;
}

/* Insertion sort of the live slots by eid (stable: slot order within an expert), then run bounds. */
static int gmb_group(const plow_moe_route* tab, uint32_t nslot, uint32_t E, const uint64_t* ewt, uint32_t which,
                     gmb_groups* g) {
    if (nslot > GMB_MAX_SLOTS) return 0;
    uint32_t n = 0;
    for (uint32_t s = 0; s < nslot; s++) {
        if (tab[s].eid >= E || !ewt_base(ewt, tab[s].eid, which)) continue;
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

/* Slice ownership over WEIGHTED columns: expert d costs GMB_W_MEM + GMB_W_DOT * M_d units per column.
 * The pass is bandwidth-bound (one 64 B line per 32 weights vs M dpbf16), so bytes dominate and the
 * dot term only tilts the split toward slices holding single-row experts. Column boundaries are
 * floor((unit - P[d]) / w_d), identical for both neighbours of a cut. */
#define GMB_W_MEM 8u
#define GMB_W_DOT 1u
static void gmb_prefix(const gmb_groups* g, uint32_t N, uint32_t* P) {
    P[0] = 0;
    for (uint32_t d = 0; d < g->nd; d++)
        P[d + 1] = P[d] + (GMB_W_MEM + GMB_W_DOT * (g->off[d + 1] - g->off[d])) * N;
}
static int gmb_cols(const uint32_t* P, uint32_t d, uint32_t N, uint32_t lo, uint32_t hi, uint32_t* n0,
                    uint32_t* n1) {
    if (P[d + 1] <= lo || P[d] >= hi) return 0;
    const uint32_t w = (P[d + 1] - P[d]) / N;
    *n0 = ((lo > P[d] ? lo : P[d]) - P[d]) / w;
    *n1 = ((hi < P[d + 1] ? hi : P[d + 1]) - P[d]) / w;
    return *n0 < *n1;
}

/* xn row (bf16) = resid * invrms * gamma. */
static void gmb_stage_xn(plow_bf16* XP, const plow_bf16* rr, const plow_bf16* gamma, uint32_t H, float eps) {
    const __m512 inv = _mm512_set1_ps(row_invrms(rr, H, eps));
    for (uint32_t h = 0; h < H; h += 16u)
        v_store_bf16(XP + h, _mm512_mul_ps(_mm512_mul_ps(v_load_bf16(rr + h), inv), v_load_bf16(gamma + h)));
}

/* 71 at B >= 2. Per pass over an expert's staged rows: 16 output channels at a time, RB = 4 (gate
 * n, gate n+1, up n, up n+1) for M <= 4 else RB = 2 (gate n, up n); the epilogue is the 16-lane
 * v_gelu_tanh(g) * u rounded once to bf16. */
V_K(v_moe_expert_glu_norm_gemma_b) {
    const uint32_t k = in->i[0], I = in->i[1], H = in->i[2], E = in->i[3], nrow = in->i[5] ? in->i[5] : 1u;
    const plow_moe_route* table = PLOW_CPU_TEN(in, T, 2);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 3);
    gmb_groups g;
    if (nrow < 2u || k == 0u || (H & 31u) || gmb_nogroup() || !ctx || !ctx->scratch ||
        ctx->scratch_bytes < (size_t)8 * H * 2u || !gmb_group(table, nrow * k, E, ewt, 0, &g)) {
        v_moe_expert_glu_norm_gemma(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* resid = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 4);
    const float eps = in->fj[0].f;
    plow_bf16* XP = ctx->scratch;
    uint32_t slots[8];
    float o[4 * 8], gg[8][16] = {{0}}, uu[8][16] = {{0}};
    const plow_bf16* Wr[4];
    uint32_t P[GMB_MAX_SLOTS + 1], lo, hi;
    gmb_prefix(&g, I, P);
    g_range(P[g.nd], slice, nblk, &lo, &hi);
    for (uint32_t d = 0; d < g.nd; d++) {
        uint32_t n0, n1;
        if (!gmb_cols(P, d, I, lo, hi, &n0, &n1)) continue;
        const plow_bf16* gu = ewt_base(ewt, g.eid[d], 0);
        for (uint32_t i = g.off[d]; i < g.off[d + 1];) {
            uint32_t M = 0;
            while (i < g.off[d + 1] && M < 8u) {
                const uint32_t s = g.slot[i++];
                gmb_stage_xn(XP + (size_t)M * H, resid + (size_t)(s / k) * H, gamma, H, eps);
                slots[M++] = s;
            }
            for (uint32_t n = n0; n < n1;) {
                const uint32_t cnt = n1 - n < 16u ? n1 - n : 16u;
                uint32_t j = 0;
                if (M <= 4u) {
                    for (; j + 2u <= cnt; j += 2u) {
                        Wr[0] = gu + (size_t)(n + j) * H; Wr[1] = Wr[0] + H;
                        Wr[2] = gu + (size_t)(I + n + j) * H; Wr[3] = Wr[2] + H;
                        dots_p(Wr, XP, H, H, 4u, M, o);
                        for (uint32_t m = 0; m < M; m++) {
                            gg[m][j] = o[m]; gg[m][j + 1] = o[M + m];
                            uu[m][j] = o[2 * M + m]; uu[m][j + 1] = o[3 * M + m];
                        }
                    }
                }
                for (; j < cnt; j++) {
                    Wr[0] = gu + (size_t)(n + j) * H; Wr[1] = gu + (size_t)(I + n + j) * H;
                    dots_p(Wr, XP, H, H, 2u, M, o);
                    for (uint32_t m = 0; m < M; m++) { gg[m][j] = o[m]; uu[m][j] = o[M + m]; }
                }
                const __mmask16 mk = cnt < 16u ? v_tail16(cnt) : (__mmask16)0xFFFFu;
                for (uint32_t m = 0; m < M; m++) {
                    const __m512 v = _mm512_mul_ps(v_gelu_tanh(_mm512_loadu_ps(gg[m])), _mm512_loadu_ps(uu[m]));
                    v_store_bf16_mask(fu + (size_t)slots[m] * I + n, mk, v);
                }
                n += cnt;
            }
        }
    }
}

/* 63 at B >= 2 (staged fu rows, RB x M dots as 76). */
V_K(v_moe_expert_down_gemma_b) {
    const uint32_t k = in->i[0], H = in->i[1], I = in->i[2], E = in->i[3], nrow = in->i[5] ? in->i[5] : 1u;
    const plow_moe_route* table = PLOW_CPU_TEN(in, T, 2);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 3);
    gmb_groups g;
    if (nrow < 2u || k == 0u || (I & 31u) || gmb_nogroup() || !ctx || !ctx->scratch ||
        ctx->scratch_bytes < (size_t)8 * I * 2u || !gmb_group(table, nrow * k, E, ewt, 1, &g)) {
        v_moe_expert_down_gemma(in, slice, nblk, T, ctx);
        return;
    }
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    plow_bf16* XP = ctx->scratch;
    if (slice == 0)
        for (uint32_t s = 0; s < nrow * k; s++)
            if (table[s].eid >= E || !ewt_base(ewt, table[s].eid, 1)) memset(part + (size_t)s * H, 0, (size_t)H * 4u);
    float* pr[8];
    float gate[8];
    uint32_t P[GMB_MAX_SLOTS + 1], lo, hi;
    gmb_prefix(&g, H, P);
    g_range(P[g.nd], slice, nblk, &lo, &hi);
    for (uint32_t d = 0; d < g.nd; d++) {
        uint32_t h0, h1;
        if (!gmb_cols(P, d, H, lo, hi, &h0, &h1)) continue;
        const plow_bf16* dn = ewt_base(ewt, g.eid[d], 1);
        for (uint32_t i = g.off[d]; i < g.off[d + 1];) {
            uint32_t M = 0;
            while (i < g.off[d + 1] && M < 8u) {
                const uint32_t s = g.slot[i++];
                memcpy(XP + (size_t)M * I, fu + (size_t)s * I, (size_t)I * 2u);
                pr[M] = part + (size_t)s * H;
                gate[M++] = table[s].gate;
            }
            down_cols(dn, I, XP, M, h0, h1, pr, gate);
        }
    }
}

/* 70/77: out = bf16(sum_k part * rsqrt(mean(sum^2) + eps) * gamma + res). The k partials of one
 * h are summed in the golden order, so only the sum of squares (a horizontal f32 reduction here,
 * sequential in golden) reassociates. */
static void combine_norm_row(plow_bf16* out, const float* part, const plow_bf16* res,
                             const plow_bf16* gamma, uint32_t H, uint32_t k, float eps, float* acc) {
    __m512 ssv = _mm512_setzero_ps();
    for (uint32_t h = 0; h < H; h += 16u) {
        __m512 a = _mm512_loadu_ps(part + h);
        for (uint32_t s = 1; s < k; s++) a = _mm512_add_ps(a, _mm512_loadu_ps(part + (size_t)s * H + h));
        _mm512_store_ps(acc + h, a);
        ssv = _mm512_fmadd_ps(a, a, ssv);
    }
    const __m512 inv = _mm512_set1_ps(g_rsqrt(_mm512_reduce_add_ps(ssv) / (float)H + eps));
    for (uint32_t h = 0; h < H; h += 16u) {
        const __m512 a = _mm512_mul_ps(_mm512_mul_ps(_mm512_load_ps(acc + h), inv), v_load_bf16(gamma + h));
        v_store_bf16(out + h, _mm512_add_ps(a, v_load_bf16(res + h)));
    }
}

/* 70: t0=out([B][H]) t1=part(f32 [B][k][H]) t2=resid t3=gamma  i0=H i1=k i2=B  f0=eps. */
V_K(v_moe_combine_norm_gemma) {
    const uint32_t H = in->i[0], k = in->i[1], nrow = in->i[2] ? in->i[2] : 1u;
    if ((H & 15u) || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)H * 4u) {
        g_moe_combine_norm_gemma(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const float* part = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* resid = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 3);
    const float eps = in->fj[0].f;
    const uint32_t nb = nblk ? nblk : 1u;
    for (uint32_t row = slice; row < nrow; row += nb)
        combine_norm_row(out + (size_t)row * H, part + (size_t)row * k * H, resid + (size_t)row * H, gamma,
                         H, k, eps, ctx->scratch);
}

/* 77: t0=out([T][H]) t1=part(f32 [T][k][H]) t2=h1 t3=gamma  i0=H i1=k i2=T  f0=eps. */
V_K(v_moe_combine_norm_gemma_pf) {
    const uint32_t H = in->i[0], k = in->i[1], nt = in->i[2] ? in->i[2] : 1u;
    if ((H & 15u) || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)H * 4u) {
        g_moe_combine_norm_gemma_pf(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const float* part = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* h1 = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 3);
    const float eps = in->fj[0].f;
    const uint32_t nb = nblk ? nblk : 1u;
    for (uint32_t tok = slice; tok < nt; tok += nb)
        combine_norm_row(out + (size_t)tok * H, part + (size_t)tok * k * H, h1 + (size_t)tok * H, gamma, H,
                         k, eps, ctx->scratch);
}

void v_register_moe_gemma(plow_cpu_kernel_fn* tab) {
    v_register_route(tab); /* route.c: family-independent MoE combine; no registrar of its own */
    tab[PLOW_DOP_MOE_COMBINE_NORM_GEMMA] = v_moe_combine_norm_gemma;
    tab[PLOW_DOP_MOE_COMBINE_NORM_GEMMA_PF] = v_moe_combine_norm_gemma_pf;
    tab[PLOW_DOP_MOE_ROUTER_GEMMA_SCORE_FAST] = v_moe_router_gemma_score_fast;
    tab[PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA] = v_moe_expert_glu_norm_gemma_b;
    tab[PLOW_DOP_MOE_EXPERT_DOWN_GEMMA] = v_moe_expert_down_gemma_b;
    tab[PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF] = v_moe_group_glu_gemma_pf;
    tab[PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF] = v_moe_group_down_gemma_pf;
}

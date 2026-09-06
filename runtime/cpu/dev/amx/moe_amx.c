/* moe_amx.c — Gemma-4 MoE grouped prefill (75 GLU, 76 down) on AMX-BF16, tier X.
 *
 * Expert weight rows are the A operand (TILELOADD straight from the [rows][K] expert matrix, no
 * repack); the live token rows of one expert segment are the B operand, packed once per 32 rows as
 * VNNI tiles (two 16-column tiles). Per K step 2 A + 2 B loads feed 4 TDPBF16PS = 32 weight rows x
 * 32 tokens, so an expert matrix streams once per 32 tokens instead of once per 8 (AVX-512 dots).
 * Slice ownership = golden (g_range over E*N columns); K % 32 != 0 falls back to golden. */
#include <immintrin.h>
#include <string.h>
#include "cpu_dev_internal.h"
#include "golden/golden.h"
#include "golden/gptoss.h"
#include "../avx512/avx512.h"
#include "../mxfp4_common.h"
#include "amx_common.h"

#define XM_MAX 32u
#define XM_PF 1024u /* bytes ahead of the current weight byte per streamed row */

static inline const plow_bf16* ewt_base(const uint64_t* ewt, uint32_t eid, uint32_t which) {
    return (const plow_bf16*)(uintptr_t)ewt[(size_t)eid * 2u + which];
}

/* xp: per K block two B tiles ([16 k-pairs][16 columns] u32); column m of tile m/16 = row m. */
static void pack_x2(uint8_t* xp, const plow_bf16* const* rowp, uint32_t M, uint32_t K) {
    const uint32_t nkb = K / 32u;
    /* One in-register transpose per 16 rows; the second tile is skipped (never loaded) at M <= 16. */
    for (uint32_t t = 0; t < 2u && t * 16u < M; t++) {
        const plow_bf16* const* rp = rowp + t * 16u;
        const uint32_t mv = M - t * 16u;
        for (uint32_t kb = 0; kb < nkb; kb++) {
            __m512i r[16];
#pragma GCC unroll 16
            for (uint32_t m = 0; m < 16; m++)
                r[m] = m < mv ? _mm512_loadu_si512((const void*)(rp[m] + (size_t)kb * 32u))
                              : _mm512_setzero_si512();
            plow_amx_tr16x16(r);
            uint8_t* tile = xp + (size_t)kb * 2048u + t * 1024u;
#pragma GCC unroll 16
            for (uint32_t p = 0; p < 16; p++) _mm512_storeu_si512((void*)(tile + p * 64u), r[p]);
        }
    }
}

/* A tile for a partial 16-row group: only `rows` rows are copied; the rest are never read back. */
static __thread uint8_t g_wtile[2][1024] __attribute__((aligned(64)));
static inline void stage_rows(uint8_t* t, const char* w, size_t ldw_b, uint32_t rows) {
    for (uint32_t r = 0; r < rows; r++)
        _mm512_store_si512((void*)(t + r * 64u), _mm512_loadu_si512((const void*)(w + r * ldw_b)));
}

/* out[r][c] (f32, row stride 32) = W[r] . x[c] for r < rows (<= 32), c < 16*nxt. */
static void dot_block(const plow_bf16* W, size_t ldw, uint32_t rows, const uint8_t* xp, uint32_t nkb,
                      uint32_t nxt, float* out) {
    const size_t ldw_b = ldw * 2u;
    const uint32_t r0 = rows < 16u ? rows : 16u, r1 = rows - r0;
    const char* w0 = (const char*)W;
    const char* w1 = w0 + 16u * ldw_b;
    _tile_zero(0);
    if (nxt > 1u) _tile_zero(1);
    if (r1) {
        _tile_zero(2);
        if (nxt > 1u) _tile_zero(3);
    }
    for (uint32_t kb = 0; kb < nkb; kb++) {
        const int odd = (kb & 1u) && r1;
        const char* pf = (odd ? w1 : w0) + (size_t)kb * 64u + XM_PF;
        const uint32_t pr = odd ? r1 : r0;
        for (uint32_t r = 0; r < pr; r++) _mm_prefetch(pf + r * ldw_b, _MM_HINT_T0);
        _tile_loadd(6, xp + (size_t)kb * 2048u, 64);
        if (nxt > 1u) _tile_loadd(7, xp + (size_t)kb * 2048u + 1024u, 64);
        if (r0 == 16u) {
            _tile_loadd(4, w0 + (size_t)kb * 64u, ldw_b);
        } else {
            stage_rows(g_wtile[0], w0 + (size_t)kb * 64u, ldw_b, r0);
            _tile_loadd(4, g_wtile[0], 64);
        }
        _tile_dpbf16ps(0, 4, 6);
        if (nxt > 1u) _tile_dpbf16ps(1, 4, 7);
        if (r1) {
            if (r1 == 16u) {
                _tile_loadd(5, w1 + (size_t)kb * 64u, ldw_b);
            } else {
                stage_rows(g_wtile[1], w1 + (size_t)kb * 64u, ldw_b, r1);
                _tile_loadd(5, g_wtile[1], 64);
            }
            _tile_dpbf16ps(2, 5, 6);
            if (nxt > 1u) _tile_dpbf16ps(3, 5, 7);
        }
    }
    _tile_stored(0, out, 128);
    if (nxt > 1u) _tile_stored(1, out + 16, 128);
    if (r1) {
        _tile_stored(2, out + 16 * 32, 128);
        if (nxt > 1u) _tile_stored(3, out + 16 * 32 + 16, 128);
    }
}

/* Next <= 32 live rows of segment [*r, rend): key[row] == UNUSED rows are padding. */
static uint32_t take_rows(const uint32_t* key, uint32_t* r, uint32_t rend, uint32_t rows[XM_MAX]) {
    uint32_t M = 0;
    while (*r < rend && M < XM_MAX) {
        const uint32_t rr = (*r)++;
        if (key[rr] != PLOW_EXPERT_UNUSED) rows[M++] = rr;
    }
    return M;
}

#define X_K(name) \
    static void name(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T, PlowCpuCtx* ctx)

/* 75: t0=fu_g([rows][I]) t1=xn2([T][H]) t2=ewt t3=meta t4=row_token  i0=I i1=H i2=E i5=act. */
X_K(x_moe_group_glu_gemma_pf) {
    const uint32_t I = in->i[0], H = in->i[1], E = in->i[2], act = in->i[5];
    if ((H & 31u) || act > 1u || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)(H / 32u) * 2048u) {
        g_moe_group_glu_gemma_pf(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 2);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 3);
    const uint32_t* row_token = PLOW_CPU_TEN(in, T, 4);
    uint8_t* xp = ctx->scratch;
    const uint32_t nkb = H / 32u;
    const plow_bf16* rowp[XM_MAX];
    uint32_t rows[XM_MAX];
    float g[32 * 32] __attribute__((aligned(64)));
    float u[32 * 32] __attribute__((aligned(64)));
    float of[32] __attribute__((aligned(64)));
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
            const uint32_t M = take_rows(row_token, &r, rend, rows);
            if (!M) continue;
            for (uint32_t m = 0; m < M; m++) rowp[m] = x + (size_t)row_token[rows[m]] * H;
            pack_x2(xp, rowp, M, H);
            const uint32_t nxt = M > 16u ? 2u : 1u;
            for (uint32_t n = n0; n < n1;) {
                const uint32_t rw = n1 - n < 32u ? n1 - n : 32u;
                dot_block(gu + (size_t)n * H, H, rw, xp, nkb, nxt, g);
                dot_block(gu + (size_t)(I + n) * H, H, rw, xp, nkb, nxt, u);
                for (uint32_t rr = 0; rr < rw; rr++) {
                    for (uint32_t c = 0; c < nxt; c++)
                        _mm512_store_ps(of + c * 16u, v_glu_pair(_mm512_load_ps(g + rr * 32u + c * 16u),
                                                                 _mm512_load_ps(u + rr * 32u + c * 16u), act, 0.0f, 0.0f));
                    for (uint32_t m = 0; m < M; m++) fu[(size_t)rows[m] * I + n + rr] = plow_f2bf(of[m]);
                }
                n += rw;
            }
        }
    }
}

/* 76: t0=part(f32 [T*k][H]) t1=fu_g t2=ewt t3=meta t4=row_partidx t5=row_gate  i0=H i1=I i2=E. */
X_K(x_moe_group_down_gemma_pf) {
    const uint32_t H = in->i[0], I = in->i[1], E = in->i[2];
    if ((I & 31u) || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)(I / 32u) * 2048u) {
        g_moe_group_down_gemma_pf(in, slice, nblk, T, ctx);
        return;
    }
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const uint64_t* ewt = PLOW_CPU_TEN(in, T, 2);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 3);
    const uint32_t* row_partidx = PLOW_CPU_TEN(in, T, 4);
    const float* row_gate = PLOW_CPU_TEN(in, T, 5);
    uint8_t* xp = ctx->scratch;
    const uint32_t nkb = I / 32u;
    const plow_bf16* rowp[XM_MAX];
    uint32_t rows[XM_MAX];
    float o[32 * 32] __attribute__((aligned(64)));
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
            const uint32_t M = take_rows(row_partidx, &r, rend, rows);
            if (!M) continue;
            for (uint32_t m = 0; m < M; m++) rowp[m] = fu + (size_t)rows[m] * I;
            pack_x2(xp, rowp, M, I);
            const uint32_t nxt = M > 16u ? 2u : 1u;
            for (uint32_t h = h0; h < h1;) {
                const uint32_t rw = h1 - h < 32u ? h1 - h : 32u;
                dot_block(dn + (size_t)h * I, I, rw, xp, nkb, nxt, o);
                for (uint32_t m = 0; m < M; m++) {
                    float* pr = part + (size_t)row_partidx[rows[m]] * H + h;
                    const float gate = row_gate[rows[m]];
                    for (uint32_t rr = 0; rr < rw; rr++) pr[rr] = gate * o[rr * 32u + m];
                }
                h += rw;
            }
        }
    }
}

/* ---- GPT-OSS flat MXFP4 experts (149 GLU, 150 down): A tiles dequantized per K step ---------- */

extern plow_mx_vlut plow_v_mx_lut; /* avx512/gptoss.c */

/* Dequantize `rows` MXFP4 weight rows (K wide) into a bf16 buffer with row stride `ldo`, so one
 * unpack serves every token block of that expert. The per-tile-load variant below (dot_block_mx)
 * re-unpacks the whole expert matrix for each 32-token block, and the unpack is ~3x the tile work,
 * so at prefill widths (64+ rows per expert) hoisting it here is the difference between MoE prefill
 * being unpack-bound and tile-bound. Same math as stage_mx4_rows, strided output. */
static void dequant_strip(plow_bf16* out, size_t ldo, const uint8_t* W, size_t rs, const uint8_t* S,
                          size_t ss, uint32_t rows, uint32_t nkb) {
    const __m512i il = _mm512_set_epi16(47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41, 9, 40, 8,
                                        39, 7, 38, 6, 37, 5, 36, 4, 35, 3, 34, 2, 33, 1, 32, 0);
    const __m512i mag = _mm512_set1_epi16(0x7FFF);
    for (uint32_t r = 0; r < rows; r++) {
        const uint8_t* w = W + (size_t)r * rs;
        const uint8_t* sc = S + (size_t)r * ss;
        plow_bf16* o = out + (size_t)r * ldo;
        for (uint32_t kb = 0; kb < nkb; kb++) {
            const __m512i b = _mm512_cvtepu8_epi16(
                _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i*)(w + (size_t)kb * 16u))));
            const __m512i ev = _mm512_permutexvar_epi16(b, plow_v_mx_lut.lut);
            const __m512i od = _mm512_permutexvar_epi16(_mm512_srli_epi16(b, 4), plow_v_mx_lut.lut);
            __m512i v = _mm512_permutex2var_epi16(ev, il, od);
            const int e = (int)sc[kb] - 127;
            const __mmask32 nz = _mm512_test_epi16_mask(v, mag);
            v = _mm512_mask_add_epi16(v, nz, v, _mm512_set1_epi16((short)(e << 7)));
            _mm512_storeu_si512((void*)(o + (size_t)kb * 32u), v);
        }
    }
}

/* Take up to `cap` live rows of segment [*r, rend) (key == UNUSED is padding). */
static uint32_t take_rows_cap(const uint32_t* key, uint32_t* r, uint32_t rend, uint32_t cap,
                              uint32_t* rows) {
    uint32_t M = 0;
    while (*r < rend && M < cap) {
        const uint32_t rr = (*r)++;
        if (key[rr] != PLOW_EXPERT_UNUSED) rows[M++] = rr;
    }
    return M;
}

/* Rows per pass: as many 32-token blocks of packed x as the scratch holds after the weight
 * buffers, capped at MXPF_BLK_MAX. */
#define MXPF_BLK_MAX 8u
#define MXPF_ROW_MAX (MXPF_BLK_MAX * 32u)
static inline uint32_t mxpf_blocks(size_t scratch, size_t wbuf_bytes, uint32_t nkb) {
    const size_t xblk = (size_t)nkb * 2048u;
    if (scratch <= wbuf_bytes + xblk) return 0;
    size_t nb = (scratch - wbuf_bytes) / xblk;
    if (nb > MXPF_BLK_MAX) nb = MXPF_BLK_MAX;
    return (uint32_t)nb;
}

/* 149: t0=fu_g t1=xn2 t2=W_gu t3=S_gu t4=meta t5=row_token t6=bias_gu?  i0=I i1=K i2=E i3=layout
 * i5=act f0/f1. layout 0: gate row 2n / up row 2n+1 (stride 2 rows); 1: gate n / up I+n. */
X_K(x_moe_glu_mx_pf) {
    const uint32_t I = in->i[0], K = in->i[1], E = in->i[2], layout = in->i[3], act = in->i[5];
    const size_t wbuf_glu = 2u * 32u * (size_t)K * 2u;
    const uint32_t nxb = (K & 31u) || !ctx || !ctx->scratch
                             ? 0u
                             : mxpf_blocks(ctx->scratch_bytes, wbuf_glu, K / 32u);
    if (!nxb) {
        g_moe_glu_mx_pf(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 3);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 4);
    const uint32_t* row_token = PLOW_CPU_TEN(in, T, 5);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 6);
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    const size_t N2 = 2u * I, ldw = K / 2u, lds = K / 32u;
    const size_t rs = layout ? ldw : 2u * ldw, ss = layout ? lds : 2u * lds;
    const uint32_t nkb = K / 32u, cap = nxb * 32u;
    const size_t xblk = (size_t)nkb * 2048u;
    uint8_t* xp = ctx->scratch;
    plow_bf16* wg = (plow_bf16*)(ctx->scratch + (size_t)nxb * xblk);
    plow_bf16* wu = wg + 32u * (size_t)K;
    const plow_bf16* rowp[MXPF_ROW_MAX];
    uint32_t rows[MXPF_ROW_MAX];
    float g[32 * 32] __attribute__((aligned(64)));
    float u[32 * 32] __attribute__((aligned(64)));
    float of[32] __attribute__((aligned(64)));
    uint32_t lo, hi;
    g_range(E * I, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t e = idx / I, n0 = idx - e * I;
        const uint32_t n1 = n0 + (hi - idx) < I ? n0 + (hi - idx) : I;
        idx += n1 - n0;
        const uint32_t r0 = (uint32_t)meta[e], rend = r0 + (uint32_t)meta[E + e];
        if (rend == r0) continue;
        const uint8_t* We = W + (size_t)e * N2 * ldw;
        const uint8_t* Se = S + (size_t)e * N2 * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * N2 : NULL;
        for (uint32_t r = r0; r < rend;) {
            const uint32_t M = take_rows_cap(row_token, &r, rend, cap, rows);
            if (!M) continue;
            const uint32_t nb = (M + 31u) / 32u;
            for (uint32_t m = 0; m < M; m++) rowp[m] = x + (size_t)row_token[rows[m]] * K;
            for (uint32_t b = 0; b < nb; b++) {
                const uint32_t mb = M - b * 32u < 32u ? M - b * 32u : 32u;
                pack_x2(xp + (size_t)b * xblk, rowp + b * 32u, mb, K);
            }
            for (uint32_t n = n0; n < n1;) {
                const uint32_t rw = n1 - n < 32u ? n1 - n : 32u;
                const uint32_t rg = layout ? n : 2u * n, ru = layout ? I + n : 2u * n + 1u;
                dequant_strip(wg, K, We + (size_t)rg * ldw, rs, Se + (size_t)rg * lds, ss, rw, nkb);
                dequant_strip(wu, K, We + (size_t)ru * ldw, rs, Se + (size_t)ru * lds, ss, rw, nkb);
                for (uint32_t b = 0; b < nb; b++) {
                    const uint32_t mb = M - b * 32u < 32u ? M - b * 32u : 32u;
                    const uint32_t nxt = mb > 16u ? 2u : 1u;
                    const uint32_t* rw_ = rows + b * 32u;
                    dot_block(wg, K, rw, xp + (size_t)b * xblk, nkb, nxt, g);
                    dot_block(wu, K, rw, xp + (size_t)b * xblk, nkb, nxt, u);
                    for (uint32_t rr = 0; rr < rw; rr++) {
                        const __m512 bg = _mm512_set1_ps(be ? plow_bf2f(be[rg + rr * (layout ? 1u : 2u)]) : 0.0f);
                        const __m512 bu = _mm512_set1_ps(be ? plow_bf2f(be[ru + rr * (layout ? 1u : 2u)]) : 0.0f);
                        for (uint32_t c = 0; c < nxt; c++)
                            _mm512_store_ps(of + c * 16u,
                                            v_glu_pair(_mm512_add_ps(_mm512_load_ps(g + rr * 32u + c * 16u), bg),
                                                       _mm512_add_ps(_mm512_load_ps(u + rr * 32u + c * 16u), bu), act, f0, f1));
                        for (uint32_t m = 0; m < mb; m++) fu[(size_t)rw_[m] * I + n + rr] = plow_f2bf(of[m]);
                    }
                }
                n += rw;
            }
        }
    }
}

/* 150: t0=part t1=fu_g t2=W_d t3=S_d t4=meta t5=bias_d? t6=row_partidx t7=row_gate  i0=H i1=I i2=E. */
X_K(x_moe_down_mx_pf) {
    const uint32_t H = in->i[0], I = in->i[1], E = in->i[2];
    const size_t wbuf_dn = 32u * (size_t)I * 2u;
    const uint32_t nxb = (I & 31u) || !ctx || !ctx->scratch
                             ? 0u
                             : mxpf_blocks(ctx->scratch_bytes, wbuf_dn, I / 32u);
    if (!nxb) {
        g_moe_down_mx_pf(in, slice, nblk, T, ctx);
        return;
    }
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 3);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const uint32_t* row_partidx = PLOW_CPU_TEN(in, T, 6);
    const float* row_gate = PLOW_CPU_TEN(in, T, 7);
    const size_t ldw = I / 2u, lds = I / 32u;
    const uint32_t nkb = I / 32u, cap = nxb * 32u;
    const size_t xblk = (size_t)nkb * 2048u;
    uint8_t* xp = ctx->scratch;
    plow_bf16* wd = (plow_bf16*)(ctx->scratch + (size_t)nxb * xblk);
    const plow_bf16* rowp[MXPF_ROW_MAX];
    uint32_t rows[MXPF_ROW_MAX];
    float o[32 * 32] __attribute__((aligned(64)));
    uint32_t lo, hi;
    g_range(E * H, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t e = idx / H, h0 = idx - e * H;
        const uint32_t h1 = h0 + (hi - idx) < H ? h0 + (hi - idx) : H;
        idx += h1 - h0;
        const uint32_t r0 = (uint32_t)meta[e], rend = r0 + (uint32_t)meta[E + e];
        if (rend == r0) continue;
        const uint8_t* We = W + (size_t)e * H * ldw;
        const uint8_t* Se = S + (size_t)e * H * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * H : NULL;
        for (uint32_t r = r0; r < rend;) {
            const uint32_t M = take_rows_cap(row_partidx, &r, rend, cap, rows);
            if (!M) continue;
            const uint32_t nb = (M + 31u) / 32u;
            for (uint32_t m = 0; m < M; m++) rowp[m] = fu + (size_t)rows[m] * I;
            for (uint32_t b = 0; b < nb; b++) {
                const uint32_t mb = M - b * 32u < 32u ? M - b * 32u : 32u;
                pack_x2(xp + (size_t)b * xblk, rowp + b * 32u, mb, I);
            }
            for (uint32_t h = h0; h < h1;) {
                const uint32_t rw = h1 - h < 32u ? h1 - h : 32u;
                dequant_strip(wd, I, We + (size_t)h * ldw, ldw, Se + (size_t)h * lds, lds, rw, nkb);
                for (uint32_t b = 0; b < nb; b++) {
                    const uint32_t mb = M - b * 32u < 32u ? M - b * 32u : 32u;
                    const uint32_t nxt = mb > 16u ? 2u : 1u;
                    const uint32_t* rw_ = rows + b * 32u;
                    dot_block(wd, I, rw, xp + (size_t)b * xblk, nkb, nxt, o);
                    for (uint32_t m = 0; m < mb; m++) {
                        float* pr = part + (size_t)row_partidx[rw_[m]] * H + h;
                        const float gate = row_gate[rw_[m]];
                        for (uint32_t rr = 0; rr < rw; rr++)
                            pr[rr] = gate * (o[rr * 32u + m] + (be ? plow_bf2f(be[h + rr]) : 0.0f));
                    }
                }
                h += rw;
            }
        }
    }
}

void plow_cpu_register_amx_moe(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF] = x_moe_group_glu_gemma_pf;
    tab[PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF] = x_moe_group_down_gemma_pf;
    tab[PLOW_DOP_MOE_GLU_MX_PF] = x_moe_glu_mx_pf;
    tab[PLOW_DOP_MOE_DOWN_MX_PF] = x_moe_down_mx_pf;
}

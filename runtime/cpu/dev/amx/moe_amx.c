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

#define XM_MAX 32u
#define XM_PF 1024u /* bytes ahead of the current weight byte per streamed row */

static inline const plow_bf16* ewt_base(const uint64_t* ewt, uint32_t eid, uint32_t which) {
    return (const plow_bf16*)(uintptr_t)ewt[(size_t)eid * 2u + which];
}

/* xp: per K block two B tiles ([16 k-pairs][16 columns] u32); column m of tile m/16 = row m. */
static void pack_x2(uint8_t* xp, const plow_bf16* const* rowp, uint32_t M, uint32_t K) {
    const uint32_t nkb = K / 32u;
    const __m512i idx = _mm512_mullo_epi32(
        _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), _mm512_set1_epi32(16));
    for (uint32_t kb = 0; kb < nkb; kb++) {
        uint32_t* tile = (uint32_t*)(xp + (size_t)kb * 2048u);
        memset(tile, 0, 2048u);
        for (uint32_t m = 0; m < M; m++) {
            const __m512i v = _mm512_loadu_si512((const void*)(rowp[m] + (size_t)kb * 32u));
            _mm512_i32scatter_epi32((void*)(tile + (m >> 4) * 256u + (m & 15u)), idx, v, 4);
        }
    }
}

/* A tile for a partial 16-row group: only `rows` rows are copied; the rest are never read back. */
static __thread uint8_t g_wtile[2][1024] __attribute__((aligned(64)));
static inline void stage_rows(uint8_t* t, const char* w, size_t ldw_b, uint32_t rows) {
    for (uint32_t r = 0; r < rows; r++) memcpy(t + r * 64u, w + r * ldw_b, 64u);
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

/* Row r of the tile <- 32 e2m1 values of packed row (W + r*rs, scale S[r*ss + kb]) at block kb,
 * scaled by 2^(s-127) as an exponent add on nonzero lanes (as gemv_amx.c stage_mx4). */
static inline void stage_mx4_rows(uint8_t* t, const uint8_t* W, size_t rs, const uint8_t* S, size_t ss,
                                  uint32_t kb, uint32_t rows) {
    const __m512i il = _mm512_set_epi16(47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41, 9, 40, 8,
                                        39, 7, 38, 6, 37, 5, 36, 4, 35, 3, 34, 2, 33, 1, 32, 0);
    for (uint32_t r = 0; r < rows; r++) {
        const __m512i b = _mm512_cvtepu8_epi16(
            _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i*)(W + r * rs + (size_t)kb * 16u))));
        const __m512i ev = _mm512_permutexvar_epi16(b, plow_v_mx_lut.lut);
        const __m512i od = _mm512_permutexvar_epi16(_mm512_srli_epi16(b, 4), plow_v_mx_lut.lut);
        __m512i v = _mm512_permutex2var_epi16(ev, il, od);
        const int e = (int)S[r * ss + kb] - 127;
        const __mmask32 nz = _mm512_test_epi16_mask(v, _mm512_set1_epi16(0x7FFF));
        v = _mm512_mask_add_epi16(v, nz, v, _mm512_set1_epi16((short)(e << 7)));
        _mm512_store_si512((void*)(t + r * 64u), v);
    }
}

/* out[r][c] (row stride 32) = dequant(W row r) . x[c], rows <= 32 at row strides rs/ss (bytes). */
static void dot_block_mx(const uint8_t* W, size_t rs, const uint8_t* S, size_t ss, uint32_t rows,
                         const uint8_t* xp, uint32_t nkb, uint32_t nxt, float* out) {
    const uint32_t r0 = rows < 16u ? rows : 16u, r1 = rows - r0;
    _tile_zero(0);
    if (nxt > 1u) _tile_zero(1);
    if (r1) {
        _tile_zero(2);
        if (nxt > 1u) _tile_zero(3);
    }
    for (uint32_t kb = 0; kb < nkb; kb++) {
        _tile_loadd(6, xp + (size_t)kb * 2048u, 64);
        if (nxt > 1u) _tile_loadd(7, xp + (size_t)kb * 2048u + 1024u, 64);
        stage_mx4_rows(g_wtile[0], W, rs, S, ss, kb, r0);
        _tile_loadd(4, g_wtile[0], 64);
        _tile_dpbf16ps(0, 4, 6);
        if (nxt > 1u) _tile_dpbf16ps(1, 4, 7);
        if (r1) {
            stage_mx4_rows(g_wtile[1], W + 16u * rs, rs, S + 16u * ss, ss, kb, r1);
            _tile_loadd(5, g_wtile[1], 64);
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

/* 149: t0=fu_g t1=xn2 t2=W_gu t3=S_gu t4=meta t5=row_token t6=bias_gu?  i0=I i1=K i2=E i3=layout
 * i5=act f0/f1. layout 0: gate row 2n / up row 2n+1 (stride 2 rows); 1: gate n / up I+n. */
X_K(x_moe_glu_mx_pf) {
    const uint32_t I = in->i[0], K = in->i[1], E = in->i[2], layout = in->i[3], act = in->i[5];
    if ((K & 31u) || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)(K / 32u) * 2048u) {
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
    const uint32_t nkb = K / 32u;
    uint8_t* xp = ctx->scratch;
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
        if (rend == r0) continue;
        const uint8_t* We = W + (size_t)e * N2 * ldw;
        const uint8_t* Se = S + (size_t)e * N2 * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * N2 : NULL;
        for (uint32_t r = r0; r < rend;) {
            const uint32_t M = take_rows(row_token, &r, rend, rows);
            if (!M) continue;
            for (uint32_t m = 0; m < M; m++) rowp[m] = x + (size_t)row_token[rows[m]] * K;
            pack_x2(xp, rowp, M, K);
            const uint32_t nxt = M > 16u ? 2u : 1u;
            for (uint32_t n = n0; n < n1;) {
                const uint32_t rw = n1 - n < 32u ? n1 - n : 32u;
                const uint32_t rg = layout ? n : 2u * n, ru = layout ? I + n : 2u * n + 1u;
                dot_block_mx(We + (size_t)rg * ldw, rs, Se + (size_t)rg * lds, ss, rw, xp, nkb, nxt, g);
                dot_block_mx(We + (size_t)ru * ldw, rs, Se + (size_t)ru * lds, ss, rw, xp, nkb, nxt, u);
                for (uint32_t rr = 0; rr < rw; rr++) {
                    const __m512 bg = _mm512_set1_ps(be ? plow_bf2f(be[rg + rr * (layout ? 1u : 2u)]) : 0.0f);
                    const __m512 bu = _mm512_set1_ps(be ? plow_bf2f(be[ru + rr * (layout ? 1u : 2u)]) : 0.0f);
                    for (uint32_t c = 0; c < nxt; c++)
                        _mm512_store_ps(of + c * 16u,
                                        v_glu_pair(_mm512_add_ps(_mm512_load_ps(g + rr * 32u + c * 16u), bg),
                                                   _mm512_add_ps(_mm512_load_ps(u + rr * 32u + c * 16u), bu), act, f0, f1));
                    for (uint32_t m = 0; m < M; m++) fu[(size_t)rows[m] * I + n + rr] = plow_f2bf(of[m]);
                }
                n += rw;
            }
        }
    }
}

/* 150: t0=part t1=fu_g t2=W_d t3=S_d t4=meta t5=bias_d? t6=row_partidx t7=row_gate  i0=H i1=I i2=E. */
X_K(x_moe_down_mx_pf) {
    const uint32_t H = in->i[0], I = in->i[1], E = in->i[2];
    if ((I & 31u) || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)(I / 32u) * 2048u) {
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
    const uint32_t nkb = I / 32u;
    uint8_t* xp = ctx->scratch;
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
        if (rend == r0) continue;
        const uint8_t* We = W + (size_t)e * H * ldw;
        const uint8_t* Se = S + (size_t)e * H * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * H : NULL;
        for (uint32_t r = r0; r < rend;) {
            const uint32_t M = take_rows(row_partidx, &r, rend, rows);
            if (!M) continue;
            for (uint32_t m = 0; m < M; m++) rowp[m] = fu + (size_t)rows[m] * I;
            pack_x2(xp, rowp, M, I);
            const uint32_t nxt = M > 16u ? 2u : 1u;
            for (uint32_t h = h0; h < h1;) {
                const uint32_t rw = h1 - h < 32u ? h1 - h : 32u;
                dot_block_mx(We + (size_t)h * ldw, ldw, Se + (size_t)h * lds, lds, rw, xp, nkb, nxt, o);
                for (uint32_t m = 0; m < M; m++) {
                    float* pr = part + (size_t)row_partidx[rows[m]] * H + h;
                    const float gate = row_gate[rows[m]];
                    for (uint32_t rr = 0; rr < rw; rr++)
                        pr[rr] = gate * (o[rr * 32u + m] + (be ? plow_bf2f(be[h + rr]) : 0.0f));
                }
                h += rw;
            }
        }
    }
}

/* ---- Batched decode (147/148) grouped by expert ------------------------------------------------
 * The per-slot kernels stream (and dequantize) an expert once per SLOT, so a rung-8 step costs 8x a
 * rung-1 step. Here the B*k slots are sorted by expert and each selected expert's rows are staged once
 * for all its M <= B slots (a row never selects an expert twice). Per-slot results are unchanged;
 * every slice builds the same grouping, so slice ownership (g_range over distinct experts x columns)
 * is deterministic. Sentinel slots (eid >= E): GLU writes nothing, DOWN zeroes them (slice 0). */
#define XD_MAX_SLOTS 256u
typedef struct {
    uint32_t nd;
    uint32_t eid[XD_MAX_SLOTS];
    uint32_t off[XD_MAX_SLOTS + 1];
    uint32_t slot[XD_MAX_SLOTS];
} xd_groups;

static int xd_group(const plow_moe_route* tab, uint32_t nslot, uint32_t E, xd_groups* g) {
    if (nslot > XD_MAX_SLOTS) return 0;
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

/* 147: t0=fu([B*k][I]) t1=x([B][K]) t2=table t3=W_gu t4=S_gu t5=bias?  i0=k i1=I i2=K i3=E i4=layout
 * i5=act i6=B  f0/f1. Slot s reads x row s/k. */
X_K(x_moe_glu_mx) {
    const uint32_t k = in->i[0], I = in->i[1], K = in->i[2], E = in->i[3], layout = in->i[4];
    const uint32_t act = in->i[5], B = in->i[6] ? in->i[6] : 1u;
    xd_groups g;
    if (B < 2u || k == 0u || (K & 31u) || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)(K / 32u) * 2048u) {
        v_moe_glu_mx(in, slice, nblk, T, ctx);
        return;
    }
    const plow_moe_route* tab = PLOW_CPU_TEN(in, T, 2);
    if (!xd_group(tab, B * k, E, &g)) {
        v_moe_glu_mx(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    const size_t N2 = 2u * I, ldw = K / 2u, lds = K / 32u;
    const size_t rs = layout ? ldw : 2u * ldw, ss = layout ? lds : 2u * lds;
    const uint32_t nkb = K / 32u;
    uint8_t* xp = ctx->scratch;
    const plow_bf16* rowp[XM_MAX];
    uint32_t slots[XM_MAX];
    float gg[32 * 32] __attribute__((aligned(64)));
    float uu[32 * 32] __attribute__((aligned(64)));
    float of[32] __attribute__((aligned(64)));
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
            uint32_t M = 0;
            while (i < g.off[d + 1] && M < XM_MAX) {
                slots[M] = g.slot[i++];
                rowp[M] = x + (size_t)(slots[M] / k) * K;
                M++;
            }
            pack_x2(xp, rowp, M, K);
            const uint32_t nxt = M > 16u ? 2u : 1u;
            for (uint32_t n = n0; n < n1;) {
                const uint32_t rw = n1 - n < 32u ? n1 - n : 32u;
                const uint32_t rg = layout ? n : 2u * n, ru = layout ? I + n : 2u * n + 1u;
                dot_block_mx(We + (size_t)rg * ldw, rs, Se + (size_t)rg * lds, ss, rw, xp, nkb, nxt, gg);
                dot_block_mx(We + (size_t)ru * ldw, rs, Se + (size_t)ru * lds, ss, rw, xp, nkb, nxt, uu);
                for (uint32_t rr = 0; rr < rw; rr++) {
                    const __m512 bg = _mm512_set1_ps(be ? plow_bf2f(be[rg + rr * (layout ? 1u : 2u)]) : 0.0f);
                    const __m512 bu = _mm512_set1_ps(be ? plow_bf2f(be[ru + rr * (layout ? 1u : 2u)]) : 0.0f);
                    for (uint32_t c = 0; c < nxt; c++)
                        _mm512_store_ps(of + c * 16u,
                                        v_glu_pair(_mm512_add_ps(_mm512_load_ps(gg + rr * 32u + c * 16u), bg),
                                                   _mm512_add_ps(_mm512_load_ps(uu + rr * 32u + c * 16u), bu), act, f0, f1));
                    for (uint32_t m = 0; m < M; m++) fu[(size_t)slots[m] * I + n + rr] = plow_f2bf(of[m]);
                }
                n += rw;
            }
        }
    }
}

/* 148: t0=part(f32 [B*k][H]) t1=fu t2=table t3=W_d t4=S_d t5=bias?  i0=k i1=H i2=I i3=E i6=B. */
X_K(x_moe_down_mx) {
    const uint32_t k = in->i[0], H = in->i[1], I = in->i[2], E = in->i[3], B = in->i[6] ? in->i[6] : 1u;
    xd_groups g;
    if (B < 2u || k == 0u || (I & 31u) || !ctx || !ctx->scratch || ctx->scratch_bytes < (size_t)(I / 32u) * 2048u) {
        v_moe_down_mx(in, slice, nblk, T, ctx);
        return;
    }
    const plow_moe_route* tab = PLOW_CPU_TEN(in, T, 2);
    if (!xd_group(tab, B * k, E, &g)) {
        v_moe_down_mx(in, slice, nblk, T, ctx);
        return;
    }
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const size_t ldw = I / 2u, lds = I / 32u;
    const uint32_t nkb = I / 32u;
    uint8_t* xp = ctx->scratch;
    if (slice == 0)
        for (uint32_t s = 0; s < B * k; s++)
            if (tab[s].eid >= E) memset(part + (size_t)s * H, 0, (size_t)H * sizeof(float));
    const plow_bf16* rowp[XM_MAX];
    uint32_t slots[XM_MAX];
    float o[32 * 32] __attribute__((aligned(64)));
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
            uint32_t M = 0;
            while (i < g.off[d + 1] && M < XM_MAX) {
                slots[M] = g.slot[i++];
                rowp[M] = fu + (size_t)slots[M] * I;
                M++;
            }
            pack_x2(xp, rowp, M, I);
            const uint32_t nxt = M > 16u ? 2u : 1u;
            for (uint32_t h = h0; h < h1;) {
                const uint32_t rw = h1 - h < 32u ? h1 - h : 32u;
                dot_block_mx(We + (size_t)h * ldw, ldw, Se + (size_t)h * lds, lds, rw, xp, nkb, nxt, o);
                for (uint32_t m = 0; m < M; m++) {
                    float* pr = part + (size_t)slots[m] * H + h;
                    const float gate = tab[slots[m]].gate;
                    for (uint32_t rr = 0; rr < rw; rr++)
                        pr[rr] = gate * (o[rr * 32u + m] + (be ? plow_bf2f(be[h + rr]) : 0.0f));
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
    tab[PLOW_DOP_MOE_GLU_MX] = x_moe_glu_mx;
    tab[PLOW_DOP_MOE_DOWN_MX] = x_moe_down_mx;
}

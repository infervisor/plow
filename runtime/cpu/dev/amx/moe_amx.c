/* moe_amx.c — Gemma-4 MoE grouped prefill (75 GLU, 76 down) on AMX-BF16, tier X.
 *
 * Expert weight rows are the A operand (TILELOADD straight from the [rows][K] expert matrix, no
 * repack); the live token rows of one expert segment are the B operand, packed once per 32 rows as
 * VNNI tiles (two 16-column tiles). Per K step 2 A + 2 B loads feed 4 TDPBF16PS = 32 weight rows x
 * 32 tokens, so an expert matrix streams once per 32 tokens instead of once per 8 (AVX-512 dots).
 * Slice ownership = golden (g_range over E*N columns); K % 32 != 0 falls back to golden. */
#include <immintrin.h>
#include <stdlib.h>
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

/* Work-weighted column ownership for the grouped prefill ops. Cost per column of expert e is the
 * strip dequant (paid once per column now that it is hoisted) plus one tile pass per 32 rows, and
 * an expert with no rows costs nothing at all. Splitting E*N columns evenly instead handed a slice
 * that owns busy experts ~1.5x its neighbours' work (measured busy min 660 / max 990 ms over a
 * 1652 ms prefill, 50% idle), and every dependent op boundary waits for the slowest slice. */
#define MXPF_W_DEQ 7u
#define MXPF_W_DOT 2u
#define MXPF_MAX_E 1024u
static void mxpf_prefix(const int32_t* meta, uint32_t E, uint32_t N, uint32_t* P) {
    P[0] = 0;
    for (uint32_t e = 0; e < E; e++) {
        const uint32_t cnt = (uint32_t)meta[E + e];
        const uint32_t w = cnt ? MXPF_W_DEQ + MXPF_W_DOT * ((cnt + 31u) / 32u) : 0u;
        P[e + 1] = P[e] + w * N;
    }
}
/* Columns [n0, n1) of expert e owned by the unit range [lo, hi). Boundaries are computed the same
 * way from both sides of a cut, so ownership is a partition. */
static int mxpf_cols(const uint32_t* P, uint32_t e, uint32_t N, uint32_t lo, uint32_t hi,
                     uint32_t* n0, uint32_t* n1) {
    if (P[e + 1] == P[e] || P[e + 1] <= lo || P[e] >= hi) return 0;
    const uint32_t w = (P[e + 1] - P[e]) / N;
    *n0 = ((lo > P[e] ? lo : P[e]) - P[e]) / w;
    *n1 = ((hi < P[e + 1] ? hi : P[e + 1]) - P[e]) / w;
    return *n0 < *n1;
}

/* ---- AMX-INT8 variant of the two grouped MXFP4 prefill kernels (PLOW_MOE_INT8=1) ------------
 *
 * MXFP4 magnitudes are exact int8: 2 * |e2m1| is {0,1,2,3,4,6,8,12} and the doubling folds into
 * the (power-of-two) E8M0 block scale, so the WEIGHT side is lossless. Only the activations are
 * quantized, per token row at amax/127.
 *
 * The block scale varies along K, so one int32 accumulation over the whole K needs a single scale
 * per weight row: emax = the row's largest E8M0 byte, each block pre-shifted left by
 * 3 - (emax - e_b). Blocks within 8x of the row max stay exact; deeper ones round onto the row's
 * 1/8-of-a-code grid (they contribute proportionally less). The exact alternative -- drain the C
 * tiles into f32 once per 32-K block -- costs ~1024 cvt+fma per 16 cycles of TMUL and is several
 * times SLOWER than the bf16 path it replaces, so it is not implemented. */

static int moe_i8(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PLOW_MOE_INT8"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v;
}

/* Lane i: the int8 code for nibble i & 15 of a block that sits 2^-d below its row's max scale. */
static __attribute__((aligned(64))) int16_t g_i8lut[9][32];
static int g_i8lut_ok;
static void i8lut_init(void) {
    static const int m2[8] = {0, 1, 2, 3, 4, 6, 8, 12}; /* 2 * e2m1 magnitude */
    for (int d = 0; d < 9; d++)
        for (int i = 0; i < 32; i++) {
            const int c = i & 15, v = m2[c & 7];
            const int q = d <= 3 ? (v << (3 - d)) : (v + (1 << (d - 4))) >> (d - 3);
            g_i8lut[d][i] = (int16_t)((c & 8) ? -q : q);
        }
    g_i8lut_ok = 1;
}

/* Dequantize `rows` MXFP4 rows to int8 (stride ldo, tail past K zeroed) and hand back the one f32
 * scale per row that the epilogue applies: 2^(emax-127) / 2 (magnitude doubling) / 8 (the shift). */
static void dequant_strip_i8(int8_t* out, size_t ldo, const uint8_t* W, size_t rs, const uint8_t* S,
                             size_t ss, uint32_t rows, uint32_t nkb, float* ws) {
    const __m512i il = _mm512_set_epi16(47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41, 9, 40, 8,
                                        39, 7, 38, 6, 37, 5, 36, 4, 35, 3, 34, 2, 33, 1, 32, 0);
    for (uint32_t r = 0; r < rows; r++) {
        const uint8_t* w = W + (size_t)r * rs;
        const uint8_t* sc = S + (size_t)r * ss;
        int8_t* o = out + (size_t)r * ldo;
        uint32_t emax = 0;
        for (uint32_t kb = 0; kb < nkb; kb++) if (sc[kb] > emax) emax = sc[kb];
        ws[r] = plow_e8m0_to_f32((uint8_t)emax) * (1.0f / 16.0f);
        for (uint32_t kb = 0; kb < nkb; kb++) {
            const __m512i b = _mm512_cvtepu8_epi16(
                _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i*)(w + (size_t)kb * 16u))));
            uint32_t d = emax - sc[kb];
            if (d > 8u) d = 8u;
            const __m512i lut = _mm512_load_si512((const void*)g_i8lut[d]);
            const __m512i ev = _mm512_permutexvar_epi16(b, lut);
            const __m512i od = _mm512_permutexvar_epi16(_mm512_srli_epi16(b, 4), lut);
            _mm256_storeu_si256((__m256i*)(o + (size_t)kb * 32u),
                                _mm512_cvtepi16_epi8(_mm512_permutex2var_epi16(ev, il, od)));
        }
        for (size_t j = (size_t)nkb * 32u; j < ldo; j++) o[j] = 0;
    }
}

/* One activation row (K bf16, K % 32 == 0) -> int8 at amax/127; returns that scale. bf16 magnitudes
 * order the same as u16, so the amax reduction is integer. */
static float quant_row_i8(int8_t* q, const plow_bf16* x, uint32_t K, uint32_t Kp) {
    __m512i a = _mm512_setzero_si512();
    for (uint32_t k = 0; k < K; k += 32u)
        a = _mm512_max_epu16(a, _mm512_and_si512(_mm512_loadu_si512((const void*)(x + k)),
                                                 _mm512_set1_epi16(0x7FFF)));
    const __m512i w32 = _mm512_set1_epi32(0xFFFF);
    const uint32_t mb = (uint32_t)_mm512_reduce_max_epu32(
        _mm512_max_epu32(_mm512_and_si512(a, w32), _mm512_srli_epi32(a, 16)));
    const float am = plow_bf2f((plow_bf16)mb);
    if (!(am > 0.0f)) { memset(q, 0, Kp); return 0.0f; }
    const __m512 vq = _mm512_set1_ps(127.0f / am);
    for (uint32_t k = 0; k < K; k += 16u) {
        const __m512 f = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(x + k))), 16));
        _mm_storeu_si128((__m128i*)(q + k),
                         _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(f, vq))));
    }
    memset(q + K, 0, Kp - K);
    return am * (1.0f / 127.0f);
}

/* Same 16x16 u32 transpose as pack_x2: a 64-byte int8 row is 16 lanes of 4 k-values, which is
 * exactly the VNNI4 B tile TDPBSSD wants, so one tile step covers 64 k instead of 32. */
static void pack_x2_i8(uint8_t* xp, const int8_t* q, size_t ldq, uint32_t M, uint32_t nkt) {
    for (uint32_t t = 0; t < 2u && t * 16u < M; t++) {
        const uint32_t mv = M - t * 16u;
        for (uint32_t kt = 0; kt < nkt; kt++) {
            __m512i r[16];
#pragma GCC unroll 16
            for (uint32_t m = 0; m < 16; m++)
                r[m] = m < mv ? _mm512_loadu_si512((const void*)(q + (size_t)(t * 16u + m) * ldq +
                                                                 (size_t)kt * 64u))
                              : _mm512_setzero_si512();
            plow_amx_tr16x16(r);
            uint8_t* tile = xp + (size_t)kt * 2048u + t * 1024u;
#pragma GCC unroll 16
            for (uint32_t p = 0; p < 16; p++) _mm512_storeu_si512((void*)(tile + p * 64u), r[p]);
        }
    }
}

/* out[r][c] (int32, row stride 32) = W[r] . x[c]; W int8 with row stride ldw bytes. */
static void dot_block_i8(const int8_t* W, size_t ldw, uint32_t rows, const uint8_t* xp, uint32_t nkt,
                         uint32_t nxt, int32_t* out) {
    const uint32_t r0 = rows < 16u ? rows : 16u, r1 = rows - r0;
    const char* w0 = (const char*)W;
    const char* w1 = w0 + 16u * ldw;
    _tile_zero(0);
    if (nxt > 1u) _tile_zero(1);
    if (r1) {
        _tile_zero(2);
        if (nxt > 1u) _tile_zero(3);
    }
    for (uint32_t kt = 0; kt < nkt; kt++) {
        const int odd = (kt & 1u) && r1;
        const char* pf = (odd ? w1 : w0) + (size_t)kt * 64u + XM_PF;
        const uint32_t pr = odd ? r1 : r0;
        for (uint32_t r = 0; r < pr; r++) _mm_prefetch(pf + r * ldw, _MM_HINT_T0);
        _tile_loadd(6, xp + (size_t)kt * 2048u, 64);
        if (nxt > 1u) _tile_loadd(7, xp + (size_t)kt * 2048u + 1024u, 64);
        if (r0 == 16u) {
            _tile_loadd(4, w0 + (size_t)kt * 64u, ldw);
        } else {
            stage_rows(g_wtile[0], w0 + (size_t)kt * 64u, ldw, r0);
            _tile_loadd(4, g_wtile[0], 64);
        }
        _tile_dpbssd(0, 4, 6);
        if (nxt > 1u) _tile_dpbssd(1, 4, 7);
        if (r1) {
            if (r1 == 16u) {
                _tile_loadd(5, w1 + (size_t)kt * 64u, ldw);
            } else {
                stage_rows(g_wtile[1], w1 + (size_t)kt * 64u, ldw, r1);
                _tile_loadd(5, g_wtile[1], 64);
            }
            _tile_dpbssd(2, 5, 6);
            if (nxt > 1u) _tile_dpbssd(3, 5, 7);
        }
    }
    _tile_stored(0, out, 128);
    if (nxt > 1u) _tile_stored(1, out + 16, 128);
    if (r1) {
        _tile_stored(2, out + 16 * 32, 128);
        if (nxt > 1u) _tile_stored(3, out + 16 * 32 + 16, 128);
    }
}

X_K(x_moe_glu_mx_pf_i8) {
    const uint32_t I = in->i[0], K = in->i[1], E = in->i[2], layout = in->i[3], act = in->i[5];
    const uint32_t Kp = (K + 63u) & ~63u, nkt = Kp / 64u;
    const size_t wbuf = 3u * 32u * (size_t)Kp; /* gate strip + up strip + the quantized token rows */
    const uint32_t nxb = (K & 31u) || !ctx || !ctx->scratch || E > MXPF_MAX_E
                             ? 0u
                             : mxpf_blocks(ctx->scratch_bytes, wbuf, nkt);
    if (!nxb) {
        g_moe_glu_mx_pf(in, slice, nblk, T, ctx);
        return;
    }
    if (!g_i8lut_ok) i8lut_init();
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
    const size_t xblk = (size_t)nkt * 2048u;
    uint8_t* xp = ctx->scratch;
    int8_t* wg = (int8_t*)(ctx->scratch + (size_t)nxb * xblk);
    int8_t* wu = wg + 32u * (size_t)Kp;
    int8_t* qb = wu + 32u * (size_t)Kp;
    uint32_t rows[MXPF_ROW_MAX];
    float as[MXPF_ROW_MAX];
    float wsg[32], wsu[32];
    int32_t gi[32 * 32] __attribute__((aligned(64)));
    int32_t ui[32 * 32] __attribute__((aligned(64)));
    float of[32] __attribute__((aligned(64)));
    uint32_t lo, hi;
    uint32_t P[MXPF_MAX_E + 1];
    mxpf_prefix(meta, E, I, P);
    g_range(P[E], slice, nblk, &lo, &hi);
    for (uint32_t e = 0; e < E; e++) {
        uint32_t n0, n1;
        if (!mxpf_cols(P, e, I, lo, hi, &n0, &n1)) continue;
        const uint32_t r0 = (uint32_t)meta[e], rend = r0 + (uint32_t)meta[E + e];
        if (rend == r0) continue;
        const uint8_t* We = W + (size_t)e * N2 * ldw;
        const uint8_t* Se = S + (size_t)e * N2 * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * N2 : NULL;
        for (uint32_t r = r0; r < rend;) {
            const uint32_t M = take_rows_cap(row_token, &r, rend, cap, rows);
            if (!M) continue;
            const uint32_t nb = (M + 31u) / 32u;
            for (uint32_t b = 0; b < nb; b++) {
                const uint32_t mb = M - b * 32u < 32u ? M - b * 32u : 32u;
                for (uint32_t m = 0; m < mb; m++)
                    as[b * 32u + m] = quant_row_i8(qb + (size_t)m * Kp,
                                                   x + (size_t)row_token[rows[b * 32u + m]] * K, K, Kp);
                pack_x2_i8(xp + (size_t)b * xblk, qb, Kp, mb, nkt);
            }
            for (uint32_t m = M; m < nb * 32u; m++) as[m] = 0.0f;
            for (uint32_t n = n0; n < n1;) {
                const uint32_t rw = n1 - n < 32u ? n1 - n : 32u;
                const uint32_t rg = layout ? n : 2u * n, ru = layout ? I + n : 2u * n + 1u;
                dequant_strip_i8(wg, Kp, We + (size_t)rg * ldw, rs, Se + (size_t)rg * lds, ss, rw, nkb, wsg);
                dequant_strip_i8(wu, Kp, We + (size_t)ru * ldw, rs, Se + (size_t)ru * lds, ss, rw, nkb, wsu);
                for (uint32_t b = 0; b < nb; b++) {
                    const uint32_t mb = M - b * 32u < 32u ? M - b * 32u : 32u;
                    const uint32_t nxt = mb > 16u ? 2u : 1u;
                    const uint32_t* rw_ = rows + b * 32u;
                    __m512 asv[2];
                    asv[0] = _mm512_loadu_ps(as + b * 32u);
                    asv[1] = nxt > 1u ? _mm512_loadu_ps(as + b * 32u + 16u) : asv[0];
                    dot_block_i8(wg, Kp, rw, xp + (size_t)b * xblk, nkt, nxt, gi);
                    dot_block_i8(wu, Kp, rw, xp + (size_t)b * xblk, nkt, nxt, ui);
                    for (uint32_t rr = 0; rr < rw; rr++) {
                        const __m512 sg = _mm512_set1_ps(wsg[rr]), su = _mm512_set1_ps(wsu[rr]);
                        const __m512 bg = _mm512_set1_ps(be ? plow_bf2f(be[rg + rr * (layout ? 1u : 2u)]) : 0.0f);
                        const __m512 bu = _mm512_set1_ps(be ? plow_bf2f(be[ru + rr * (layout ? 1u : 2u)]) : 0.0f);
                        for (uint32_t c = 0; c < nxt; c++)
                            _mm512_store_ps(
                                of + c * 16u,
                                v_glu_pair(_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(
                                                               (const void*)(gi + rr * 32u + c * 16u))),
                                                           _mm512_mul_ps(sg, asv[c]), bg),
                                           _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(
                                                               (const void*)(ui + rr * 32u + c * 16u))),
                                                           _mm512_mul_ps(su, asv[c]), bu),
                                           act, f0, f1));
                        for (uint32_t m = 0; m < mb; m++) fu[(size_t)rw_[m] * I + n + rr] = plow_f2bf(of[m]);
                    }
                }
                n += rw;
            }
        }
    }
}

X_K(x_moe_down_mx_pf_i8) {
    const uint32_t H = in->i[0], I = in->i[1], E = in->i[2];
    const uint32_t Ip = (I + 63u) & ~63u, nkt = Ip / 64u;
    const size_t wbuf = 2u * 32u * (size_t)Ip; /* down strip + the quantized fu rows */
    const uint32_t nxb = (I & 31u) || !ctx || !ctx->scratch || E > MXPF_MAX_E
                             ? 0u
                             : mxpf_blocks(ctx->scratch_bytes, wbuf, nkt);
    if (!nxb) {
        g_moe_down_mx_pf(in, slice, nblk, T, ctx);
        return;
    }
    if (!g_i8lut_ok) i8lut_init();
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
    const size_t xblk = (size_t)nkt * 2048u;
    uint8_t* xp = ctx->scratch;
    int8_t* wd = (int8_t*)(ctx->scratch + (size_t)nxb * xblk);
    int8_t* qb = wd + 32u * (size_t)Ip;
    uint32_t rows[MXPF_ROW_MAX];
    float as[MXPF_ROW_MAX];
    float wsd[32];
    int32_t oi[32 * 32] __attribute__((aligned(64)));
    uint32_t lo, hi;
    uint32_t P[MXPF_MAX_E + 1];
    mxpf_prefix(meta, E, H, P);
    g_range(P[E], slice, nblk, &lo, &hi);
    for (uint32_t e = 0; e < E; e++) {
        uint32_t h0, h1;
        if (!mxpf_cols(P, e, H, lo, hi, &h0, &h1)) continue;
        const uint32_t r0 = (uint32_t)meta[e], rend = r0 + (uint32_t)meta[E + e];
        if (rend == r0) continue;
        const uint8_t* We = W + (size_t)e * H * ldw;
        const uint8_t* Se = S + (size_t)e * H * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * H : NULL;
        for (uint32_t r = r0; r < rend;) {
            const uint32_t M = take_rows_cap(row_partidx, &r, rend, cap, rows);
            if (!M) continue;
            const uint32_t nb = (M + 31u) / 32u;
            for (uint32_t b = 0; b < nb; b++) {
                const uint32_t mb = M - b * 32u < 32u ? M - b * 32u : 32u;
                for (uint32_t m = 0; m < mb; m++)
                    as[b * 32u + m] =
                        quant_row_i8(qb + (size_t)m * Ip, fu + (size_t)rows[b * 32u + m] * I, I, Ip);
                pack_x2_i8(xp + (size_t)b * xblk, qb, Ip, mb, nkt);
            }
            for (uint32_t m = M; m < nb * 32u; m++) as[m] = 0.0f;
            for (uint32_t h = h0; h < h1;) {
                const uint32_t rw = h1 - h < 32u ? h1 - h : 32u;
                dequant_strip_i8(wd, Ip, We + (size_t)h * ldw, ldw, Se + (size_t)h * lds, lds, rw, nkb, wsd);
                for (uint32_t b = 0; b < nb; b++) {
                    const uint32_t mb = M - b * 32u < 32u ? M - b * 32u : 32u;
                    const uint32_t nxt = mb > 16u ? 2u : 1u;
                    const uint32_t* rw_ = rows + b * 32u;
                    dot_block_i8(wd, Ip, rw, xp + (size_t)b * xblk, nkt, nxt, oi);
                    for (uint32_t m = 0; m < mb; m++) {
                        float* pr = part + (size_t)row_partidx[rw_[m]] * H + h;
                        const float gate = row_gate[rw_[m]], am = as[b * 32u + m];
                        for (uint32_t rr = 0; rr < rw; rr++)
                            pr[rr] = gate * ((float)oi[rr * 32u + m] * wsd[rr] * am +
                                             (be ? plow_bf2f(be[h + rr]) : 0.0f));
                    }
                }
                h += rw;
            }
        }
    }
}

/* 149: t0=fu_g t1=xn2 t2=W_gu t3=S_gu t4=meta t5=row_token t6=bias_gu?  i0=I i1=K i2=E i3=layout
 * i5=act f0/f1. layout 0: gate row 2n / up row 2n+1 (stride 2 rows); 1: gate n / up I+n. */
X_K(x_moe_glu_mx_pf) {
    if (moe_i8()) { x_moe_glu_mx_pf_i8(in, slice, nblk, T, ctx); return; }
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
    if (E > MXPF_MAX_E) {
        g_moe_glu_mx_pf(in, slice, nblk, T, ctx);
        return;
    }
    uint32_t P[MXPF_MAX_E + 1];
    mxpf_prefix(meta, E, I, P);
    g_range(P[E], slice, nblk, &lo, &hi);
    for (uint32_t e = 0; e < E; e++) {
        uint32_t n0, n1;
        if (!mxpf_cols(P, e, I, lo, hi, &n0, &n1)) continue;
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
    if (moe_i8()) { x_moe_down_mx_pf_i8(in, slice, nblk, T, ctx); return; }
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
    if (E > MXPF_MAX_E) {
        g_moe_down_mx_pf(in, slice, nblk, T, ctx);
        return;
    }
    uint32_t P[MXPF_MAX_E + 1];
    mxpf_prefix(meta, E, H, P);
    g_range(P[E], slice, nblk, &lo, &hi);
    for (uint32_t e = 0; e < E; e++) {
        uint32_t h0, h1;
        if (!mxpf_cols(P, e, H, lo, hi, &h0, &h1)) continue;
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

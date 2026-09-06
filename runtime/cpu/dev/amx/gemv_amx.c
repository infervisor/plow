/* gemv_amx.c — batched decode GEMV family (4 <= M <= 16) on AMX-BF16, tier X.
 *
 * The WEIGHTS are the A operand: 16 row-major rows x 32 K per TILELOADD straight from memory,
 * no repack. The M activation rows are the B operand, packed ONCE per call into VNNI tiles
 * ([K/32 blocks][16 k-pairs][16 columns = sequences][2]). One tile load + one TDPBF16PS per
 * 16 weight rows x 32 K = 64 B of weights per cycle per core, so a step stays DRAM-bound at any
 * M <= 16 and rung 8 costs what rung 1 does. The AVX-512 GEMV (RB x M register dots) re-reads
 * the M x-rows from L2 per weight-row pair: 457 ms at rung 8 vs 250 at rung 4 (bf16, Gemma-4-12B).
 * Same slice ownership as golden (g_range over columns); fused norms / M < 4 / K % 32 fall back to
 * the AVX-512 kernels. */
#include <immintrin.h>
#include <string.h>
#include "cpu_dev_internal.h"
#include "golden/golden.h"
#include "../avx512/avx512.h"
#include "../fp8_common.h"
#include "../mxfp4_common.h"

/* M = 4 measured 276 ms on this path vs 250 on the AVX-512 RB=4 dots (Gemma-4-12B bf16); the
 * crossover is between 4 and 5. */
#define XGV_MIN_M 5u
#define XGV_MAX_M 16u
#define XGV_PF 1024u /* bytes ahead of the current weight byte per streamed row */

/* B tiles for x[M][K] (row stride ldx elements): tile kb at xp + kb*1024, row p (k pair), column m,
 * u32 = (x[m][kb*32+2p], x[m][kb*32+2p+1]). Columns m >= M are zero. */
static void pack_x_tiles(uint8_t* xp, const plow_bf16* X, size_t ldx, uint32_t M, uint32_t K) {
    const uint32_t nkb = K / 32u;
    const __m512i idx = _mm512_mullo_epi32(
        _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), _mm512_set1_epi32(16));
    for (uint32_t kb = 0; kb < nkb; kb++) {
        uint32_t* tile = (uint32_t*)(xp + (size_t)kb * 1024u);
        memset(tile, 0, 1024u);
        for (uint32_t m = 0; m < M; m++) {
            const __m512i v = _mm512_loadu_si512((const void*)(X + m * ldx + (size_t)kb * 32u));
            _mm512_i32scatter_epi32((void*)(tile + m), idx, v, 4);
        }
    }
}

/* out[r][c] (f32, 16 columns) = W[n0+r] . x[c] for r < 16*ntile (ntile in {1, 2}). */
static inline void dot_tiles(const plow_bf16* W, size_t ldw_b, const uint8_t* xp, uint32_t nkb,
                             uint32_t ntile, float* out) {
    _tile_zero(0);
    if (ntile > 1u) _tile_zero(1);
    const char* w0 = (const char*)W;
    const char* w1 = w0 + 16u * ldw_b;
    for (uint32_t kb = 0; kb < nkb; kb++) {
        /* Sixteen streams per tile at stride ldw: keep each row XGV_PF ahead (half the rows per
         * step so the prefetch traffic stays at 16 per kb). */
        const char* pf = ((kb & 1u) && ntile > 1u ? w1 : w0) + (size_t)kb * 64u + XGV_PF;
        for (uint32_t r = 0; r < 16u; r++) _mm_prefetch(pf + r * ldw_b, _MM_HINT_T0);
        _tile_loadd(6, xp + (size_t)kb * 1024u, 64);
        _tile_loadd(4, w0 + (size_t)kb * 64u, ldw_b);
        _tile_dpbf16ps(0, 4, 6);
        if (ntile > 1u) {
            _tile_loadd(5, w1 + (size_t)kb * 64u, ldw_b);
            _tile_dpbf16ps(1, 5, 6);
        }
    }
    _tile_stored(0, out, 64);
    if (ntile > 1u) _tile_stored(1, out + 16 * 16, 64);
}

/* ---- Quantized weights: dequantize a 16-row x 32-K strip into an A tile per K step ---------- */

extern plow_fp8_vlut plow_amx_fp8_lut; /* gemm_amx.c */
extern plow_mx_vlut plow_v_mx_lut;     /* avx512/gptoss.c */
static __thread uint8_t g_atile[2][1024] __attribute__((aligned(64)));

/* e4m3 rows (stride ldw bytes): row r of the tile <- bf16(W[r][kb*32 .. +32)). */
static inline void stage_fp8(uint8_t* tile, const uint8_t* W, size_t ldw, uint32_t kb) {
    for (uint32_t r = 0; r < 16u; r++)
        _mm512_store_si512((void*)(tile + r * 64u),
                           plow_fp8x32_to_bf16(&plow_amx_fp8_lut,
                                               _mm256_loadu_si256((const __m256i*)(W + r * ldw + (size_t)kb * 32u))));
}

/* MXFP4 rows (16 packed bytes per 32-block, one E8M0 byte per block): row r of the tile <-
 * e2m1 LUT values in natural order times 2^(s-127), applied as an exponent add on nonzero lanes. */
static inline void stage_mx4(uint8_t* tile, const uint8_t* W, size_t ldw, const uint8_t* S, size_t lds,
                             uint32_t kb) {
    const __m512i il = _mm512_set_epi16(47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41, 9, 40, 8,
                                        39, 7, 38, 6, 37, 5, 36, 4, 35, 3, 34, 2, 33, 1, 32, 0);
    for (uint32_t r = 0; r < 16u; r++) {
        const __m512i b = _mm512_cvtepu8_epi16(
            _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i*)(W + r * ldw + (size_t)kb * 16u))));
        const __m512i ev = _mm512_permutexvar_epi16(b, plow_v_mx_lut.lut);
        const __m512i od = _mm512_permutexvar_epi16(_mm512_srli_epi16(b, 4), plow_v_mx_lut.lut);
        __m512i v = _mm512_permutex2var_epi16(ev, il, od);
        const int e = (int)S[r * lds + kb] - 127;
        const __mmask32 nz = _mm512_test_epi16_mask(v, _mm512_set1_epi16(0x7FFF));
        v = _mm512_mask_add_epi16(v, nz, v, _mm512_set1_epi16((short)(e << 7)));
        _mm512_store_si512((void*)(tile + r * 64u), v);
    }
}

/* out[r][c] for 16*ntile weight rows starting at W (quantized), like dot_tiles. kind 0 fp8, 1 mxfp4. */
static inline void dot_tiles_q(int kind, const uint8_t* W, size_t ldw, const uint8_t* S, size_t lds,
                               const uint8_t* xp, uint32_t nkb, uint32_t ntile, float* out) {
    _tile_zero(0);
    if (ntile > 1u) _tile_zero(1);
    for (uint32_t kb = 0; kb < nkb; kb++) {
        _tile_loadd(6, xp + (size_t)kb * 1024u, 64);
        if (kind == 0) stage_fp8(g_atile[0], W, ldw, kb);
        else stage_mx4(g_atile[0], W, ldw, S, lds, kb);
        _tile_loadd(4, g_atile[0], 64);
        _tile_dpbf16ps(0, 4, 6);
        if (ntile > 1u) {
            if (kind == 0) stage_fp8(g_atile[1], W + 16u * ldw, ldw, kb);
            else stage_mx4(g_atile[1], W + 16u * ldw, ldw, S + 16u * lds, lds, kb);
            _tile_loadd(5, g_atile[1], 64);
            _tile_dpbf16ps(1, 5, 6);
        }
    }
    _tile_stored(0, out, 64);
    if (ntile > 1u) _tile_stored(1, out + 16 * 16, 64);
}

/* Tail rows (< 16) of a quantized span: dequantize a row to bf16 in g_atile and dot with x. */
static void dot_tail_q(int kind, const uint8_t* W, size_t ldw, const uint8_t* S, size_t lds,
                       const plow_bf16* X, size_t ldx, uint32_t M, uint32_t K, uint32_t rows, float* out) {
    for (uint32_t r = 0; r < rows; r++) {
        __m512 acc[XGV_MAX_M];
        for (uint32_t m = 0; m < M; m++) acc[m] = _mm512_setzero_ps();
        for (uint32_t kb = 0; kb < K / 32u; kb++) {
            if (kind == 0) stage_fp8(g_atile[0], W + r * ldw, 0, kb); /* one row into tile row 0..15 (all same) */
            else stage_mx4(g_atile[0], W + r * ldw, 0, S + r * lds, 0, kb);
            const __m512bh wv = (__m512bh)_mm512_load_si512((const void*)g_atile[0]);
            for (uint32_t m = 0; m < M; m++)
                acc[m] = _mm512_dpbf16_ps(acc[m], wv,
                                          (__m512bh)_mm512_loadu_si512((const void*)(X + m * ldx + (size_t)kb * 32u)));
        }
        for (uint32_t m = 0; m < M; m++) out[r * 16u + m] = _mm512_reduce_add_ps(acc[m]);
        for (uint32_t m = M; m < 16u; m++) out[r * 16u + m] = 0.0f;
    }
}

static void dot_rows_q(int kind, const uint8_t* W, size_t ldw, const uint8_t* S, size_t lds,
                       const plow_bf16* X, const uint8_t* xp, uint32_t M, uint32_t K, uint32_t n,
                       uint32_t rows, float* out) {
    const uint32_t full = rows / 16u;
    if (full) dot_tiles_q(kind, W + (size_t)n * ldw, ldw, S ? S + (size_t)n * lds : NULL, lds, xp, K / 32u, full, out);
    if (rows > full * 16u)
        dot_tail_q(kind, W + (size_t)(n + full * 16u) * ldw, ldw, S ? S + (size_t)(n + full * 16u) * lds : NULL, lds,
                   X, K, M, K, rows - full * 16u, out + full * 16u * 16u);
}

/* Rows that do not fill a 16-row tile: AVX-512 dots with one accumulator per activation row. */
static void dot_tail(const plow_bf16* W, size_t ldw, const plow_bf16* X, size_t ldx, uint32_t M,
                     uint32_t K, uint32_t rows, float* out) {
    for (uint32_t r = 0; r < rows; r++) {
        __m512 acc[XGV_MAX_M];
        for (uint32_t m = 0; m < M; m++) acc[m] = _mm512_setzero_ps();
        const plow_bf16* w = W + r * ldw;
        for (uint32_t k = 0; k < K; k += 32u) {
            const __m512bh wv = (__m512bh)_mm512_loadu_si512((const void*)(w + k));
            for (uint32_t m = 0; m < M; m++)
                acc[m] = _mm512_dpbf16_ps(acc[m], wv, (__m512bh)_mm512_loadu_si512((const void*)(X + m * ldx + k)));
        }
        for (uint32_t m = 0; m < M; m++) out[r * 16u + m] = _mm512_reduce_add_ps(acc[m]);
        for (uint32_t m = M; m < 16u; m++) out[r * 16u + m] = 0.0f;
    }
}

/* out[r][c] for weight rows [n, n+rows), rows <= 32: tiles for the 16-row groups, dots for the rest. */
static void dot_rows(const plow_bf16* W, size_t ldw, const plow_bf16* X, const uint8_t* xp, uint32_t M,
                     uint32_t K, uint32_t n, uint32_t rows, float* out) {
    const uint32_t full = rows / 16u;
    if (full) dot_tiles(W + (size_t)n * ldw, ldw * 2u, xp, K / 32u, full, out);
    if (rows > full * 16u)
        dot_tail(W + (size_t)(n + full * 16u) * ldw, ldw, X, K, M, K, rows - full * 16u, out + full * 16u * 16u);
}

static void store_span(plow_bf16* C, size_t ldc, uint32_t n, uint32_t rows, uint32_t M, const float* out,
                       const plow_bf16* bias) {
    for (uint32_t r = 0; r < rows; r++) {
        const float b = bias ? plow_bf2f(bias[n + r]) : 0.0f;
        for (uint32_t m = 0; m < M; m++) C[m * ldc + n + r] = plow_f2bf(out[r * 16u + m] + b);
    }
}

/* C[m][n0..n1) = x[m] . W[n]^T (+ bias[n]). */
static void gemv_span_amx(plow_bf16* C, size_t ldc, const plow_bf16* W, const plow_bf16* X,
                          const uint8_t* xp, uint32_t M, uint32_t K, uint32_t n0, uint32_t n1,
                          const plow_bf16* bias) {
    float out[32 * 16] __attribute__((aligned(64)));
    for (uint32_t n = n0; n < n1;) {
        const uint32_t rows = n1 - n < 32u ? n1 - n : 32u;
        dot_rows(W, K, X, xp, M, K, n, rows, out);
        store_span(C, ldc, n, rows, M, out, bias);
        n += rows;
    }
}

static int xgv_usable(const PlowCpuCtx* ctx, uint32_t M, uint32_t K) {
    return M >= XGV_MIN_M && M <= XGV_MAX_M && (K & 31u) == 0u && K > 0 && ctx && ctx->scratch &&
           ctx->scratch_bytes >= (size_t)(K / 32u) * 1024u;
}

#define X_K(name) \
    static void name(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T, PlowCpuCtx* ctx)

/* t0=C t1=x t2=W t3=rms? t4=gamma? t7=bias?  i0=M i1=N i2=K i3=norm i4=a_row0 */
X_K(x_gemv) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], norm = in->i[3];
    if (norm != 0u || !xgv_usable(ctx, M, K)) {
        v_gemv(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const plow_bf16* W = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 7);
    uint8_t* xp = ctx->scratch;
    pack_x_tiles(xp, x, K, M, K);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    gemv_span_amx(C, N, W, x, xp, M, K, n0, n1, bias);
}

/* t0=fu t1=x t2=W_gate t5=W_up t6=bias_gate? t7=bias_up?  i0=M i1=N i2=K i5=act f0/f1 */
X_K(x_gemv_glu) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    if (!xgv_usable(ctx, M, K)) {
        v_gemv_glu(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* Wg = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* Wu = PLOW_CPU_TEN(in, T, 5);
    const plow_bf16* bg = PLOW_CPU_TEN(in, T, 6);
    const plow_bf16* bu = PLOW_CPU_TEN(in, T, 7);
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    uint8_t* xp = ctx->scratch;
    pack_x_tiles(xp, x, K, M, K);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    float g[32 * 16] __attribute__((aligned(64)));
    float u[32 * 16] __attribute__((aligned(64)));
    const __mmask16 mk = v_tail16(M);
    for (uint32_t n = n0; n < n1;) {
        const uint32_t rows = n1 - n < 32u ? n1 - n : 32u;
        dot_rows(Wg, K, x, xp, M, K, n, rows, g);
        dot_rows(Wu, K, x, xp, M, K, n, rows, u);
        for (uint32_t r = 0; r < rows; r++) {
            __m512 gv = _mm512_maskz_loadu_ps(mk, g + r * 16u);
            __m512 uv = _mm512_maskz_loadu_ps(mk, u + r * 16u);
            if (bg) gv = _mm512_add_ps(gv, _mm512_set1_ps(plow_bf2f(bg[n + r])));
            if (bu) uv = _mm512_add_ps(uv, _mm512_set1_ps(plow_bf2f(bu[n + r])));
            float of[16] __attribute__((aligned(64)));
            _mm512_store_ps(of, v_glu_pair(gv, uv, act, f0, f1));
            for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n + r] = plow_f2bf(of[m]);
        }
        n += rows;
    }
}

/* t0=q t1=x t2=W_q t3=k t4=W_k t5=v t6=W_v t7=q-norm gamma?  i0=M i1=Nq i2=K i3=Nk i4=Nv
 * i5/i6/i7 = bias handles. The q-norm fold stays on the AVX-512 kernel. */
X_K(x_gemv_qkv) {
    const uint32_t M = in->i[0], Nq = in->i[1], K = in->i[2], Nk = in->i[3], Nv = in->i[4];
    if (PLOW_CPU_TEN(in, T, 7) || !xgv_usable(ctx, M, K)) {
        v_gemv_qkv(in, slice, nblk, T, ctx);
        return;
    }
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    plow_bf16* Cs[3] = {PLOW_CPU_TEN(in, T, 0), PLOW_CPU_TEN(in, T, 3), PLOW_CPU_TEN(in, T, 5)};
    const plow_bf16* Ws[3] = {PLOW_CPU_TEN(in, T, 2), PLOW_CPU_TEN(in, T, 4), PLOW_CPU_TEN(in, T, 6)};
    const plow_bf16* Bs[3] = {G_QKV_BIAS(in, T, 5), G_QKV_BIAS(in, T, 6), G_QKV_BIAS(in, T, 7)};
    const uint32_t Ns[3] = {Nq, Nk, Nv};
    uint8_t* xp = ctx->scratch;
    pack_x_tiles(xp, x, K, M, K);
    uint32_t n0, n1;
    g_range(Nq + Nk + Nv, slice, nblk, &n0, &n1);
    uint32_t S0 = 0;
    for (uint32_t s = 0; s < 3; s++) {
        const uint32_t S1 = S0 + Ns[s];
        const uint32_t a = n0 > S0 ? n0 : S0, b = n1 < S1 ? n1 : S1;
        if (a < b) gemv_span_amx(Cs[s], Ns[s], Ws[s], x, xp, M, K, a - S0, b - S0, Bs[s]);
        S0 = S1;
    }
}

void v_gemv_fp8(const PlowDevInst*, uint32_t, uint32_t, void* const*, PlowCpuCtx*);
void v_gemv_glu_fp8(const PlowDevInst*, uint32_t, uint32_t, void* const*, PlowCpuCtx*);
void v_gemv_glu_mxfp4(const PlowDevInst*, uint32_t, uint32_t, void* const*, PlowCpuCtx*);

/* t0=C t1=x t2=W(e4m3) t5=w_scale i0=M i1=N i2=K i4=a_row0 (i3 NRN fold -> AVX-512/golden). */
X_K(x_gemv_fp8) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    const float* ws = PLOW_CPU_TEN(in, T, 5);
    if (in->i[3] != 0u || !ws || !xgv_usable(ctx, M, K)) {
        v_gemv_fp8(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    uint8_t* xp = ctx->scratch;
    pack_x_tiles(xp, x, K, M, K);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    float out[32 * 16] __attribute__((aligned(64)));
    for (uint32_t n = n0; n < n1;) {
        const uint32_t rows = n1 - n < 32u ? n1 - n : 32u;
        dot_rows_q(0, W, K, NULL, 0, x, xp, M, K, n, rows, out);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n + r] = plow_f2bf(out[r * 16u + m] * ws[n + r]);
        n += rows;
    }
}

/* t0=fu t1=x t2=Wg t3=g_scale t4=u_scale t5=Wu i0=M i1=N i2=K i5=act */
X_K(x_gemv_glu_fp8) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    const float* gs = PLOW_CPU_TEN(in, T, 3);
    const float* us = PLOW_CPU_TEN(in, T, 4);
    if (act > 1u || !gs || !us || !xgv_usable(ctx, M, K)) {
        v_gemv_glu_fp8(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* Wg = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* Wu = PLOW_CPU_TEN(in, T, 5);
    uint8_t* xp = ctx->scratch;
    pack_x_tiles(xp, x, K, M, K);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    float g[32 * 16] __attribute__((aligned(64)));
    float u[32 * 16] __attribute__((aligned(64)));
    const __mmask16 mk = v_tail16(M);
    for (uint32_t n = n0; n < n1;) {
        const uint32_t rows = n1 - n < 32u ? n1 - n : 32u;
        dot_rows_q(0, Wg, K, NULL, 0, x, xp, M, K, n, rows, g);
        dot_rows_q(0, Wu, K, NULL, 0, x, xp, M, K, n, rows, u);
        for (uint32_t r = 0; r < rows; r++) {
            const __m512 gv = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, g + r * 16u), _mm512_set1_ps(gs[n + r]));
            const __m512 uv = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, u + r * 16u), _mm512_set1_ps(us[n + r]));
            float of[16] __attribute__((aligned(64)));
            _mm512_store_ps(of, v_glu_pair(gv, uv, act, 0.0f, 0.0f));
            for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n + r] = plow_f2bf(of[m]);
        }
        n += rows;
    }
}

/* t0=C t1=x t2=W(fp4) t3=S(e8m0)  i0=M i1=N i2=K */
X_K(x_gemv_mxfp4) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    if (!xgv_usable(ctx, M, K)) {
        v_gemv_mxfp4(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 3);
    uint8_t* xp = ctx->scratch;
    pack_x_tiles(xp, x, K, M, K);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    float out[32 * 16] __attribute__((aligned(64)));
    for (uint32_t n = n0; n < n1;) {
        const uint32_t rows = n1 - n < 32u ? n1 - n : 32u;
        dot_rows_q(1, W, K / 2u, S, K / 32u, x, xp, M, K, n, rows, out);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n + r] = plow_f2bf(out[r * 16u + m]);
        n += rows;
    }
}

/* t0=C t1=x t2=Wg t5=Wu t3=Sg t4=Su  i0=M i1=N i2=K i5=act */
X_K(x_gemv_glu_mxfp4) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    if (act > 1u || !xgv_usable(ctx, M, K)) {
        v_gemv_glu_mxfp4(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* Wg = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* Wu = PLOW_CPU_TEN(in, T, 5);
    const uint8_t* Sg = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* Su = PLOW_CPU_TEN(in, T, 4);
    uint8_t* xp = ctx->scratch;
    pack_x_tiles(xp, x, K, M, K);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    float g[32 * 16] __attribute__((aligned(64)));
    float u[32 * 16] __attribute__((aligned(64)));
    const __mmask16 mk = v_tail16(M);
    for (uint32_t n = n0; n < n1;) {
        const uint32_t rows = n1 - n < 32u ? n1 - n : 32u;
        dot_rows_q(1, Wg, K / 2u, Sg, K / 32u, x, xp, M, K, n, rows, g);
        dot_rows_q(1, Wu, K / 2u, Su, K / 32u, x, xp, M, K, n, rows, u);
        for (uint32_t r = 0; r < rows; r++) {
            float of[16] __attribute__((aligned(64)));
            _mm512_store_ps(of, v_glu_pair(_mm512_maskz_loadu_ps(mk, g + r * 16u),
                                           _mm512_maskz_loadu_ps(mk, u + r * 16u), act, 0.0f, 0.0f));
            for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n + r] = plow_f2bf(of[m]);
        }
        n += rows;
    }
}

void plow_cpu_register_amx_gemv(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_GEMV] = x_gemv;
    tab[PLOW_DOP_GEMV_GLU] = x_gemv_glu;
    tab[PLOW_DOP_GEMV_QKV] = x_gemv_qkv;
    tab[PLOW_DOP_GEMV_FP8] = x_gemv_fp8;
    tab[PLOW_DOP_GEMV_GLU_FP8] = x_gemv_glu_fp8;
    tab[PLOW_DOP_GEMV_MXFP4] = x_gemv_mxfp4;
    tab[PLOW_DOP_GEMV_GLU_MXFP4] = x_gemv_glu_mxfp4;
}

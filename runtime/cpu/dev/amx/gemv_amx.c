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

#define XGV_MIN_M 4u
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

void plow_cpu_register_amx_gemv(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_GEMV] = x_gemv;
    tab[PLOW_DOP_GEMV_GLU] = x_gemv_glu;
    tab[PLOW_DOP_GEMV_QKV] = x_gemv_qkv;
}

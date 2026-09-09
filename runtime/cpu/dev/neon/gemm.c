/* GEMM family, NEON bf16 — prefill, compute-bound. Same tile ownership as golden/gemm.c
 * (linear tile id = slice + k*nblk over (BM, BN) output tiles). Micro-kernel: a 4x4 block of
 * (activation row, weight row) dots, 16 f32x4 accumulators fed by `vbfdotq_f32` over 8 bf16 of
 * K per step, horizontal reduce at the end. The A rows of a 4-row block stay in L1 across the
 * tile's N sweep. */
#include "neon.h"

#define UNROLL _Pragma("clang loop unroll(full)")

/* out[r][c] = A[r] . B[c] for r < RM, c < RN (compile-time at every call site). */
static inline __attribute__((always_inline)) void mk(const plow_bf16* A, size_t lda,
                                                     const plow_bf16* B, size_t ldb, uint32_t K,
                                                     const uint32_t RM, const uint32_t RN,
                                                     float out[4][4]) {
    float32x4_t acc[4][4];
    UNROLL
    for (uint32_t r = 0; r < RM; r++)
        UNROLL
        for (uint32_t c = 0; c < RN; c++) acc[r][c] = vdupq_n_f32(0);
    uint32_t k = 0;
    for (; k + 8 <= K; k += 8) {
        bfloat16x8_t av[4], bv[4];
        UNROLL
        for (uint32_t r = 0; r < RM; r++) av[r] = n_ldbh(A + r * lda + k);
        UNROLL
        for (uint32_t c = 0; c < RN; c++) bv[c] = n_ldbh(B + c * ldb + k);
        UNROLL
        for (uint32_t r = 0; r < RM; r++)
            UNROLL
            for (uint32_t c = 0; c < RN; c++) acc[r][c] = vbfdotq_f32(acc[r][c], av[r], bv[c]);
    }
    UNROLL
    for (uint32_t r = 0; r < RM; r++)
        UNROLL
        for (uint32_t c = 0; c < RN; c++) {
            float s = n_hsum(acc[r][c]);
            for (uint32_t kk = k; kk < K; kk++) s += plow_bf2f(A[r * lda + kk]) * plow_bf2f(B[c * ldb + kk]);
            out[r][c] = s;
        }
}

/* C[m0..m1)[n0..n1) = bf16((A . B^T) * rs[m]? * cs[n]? + bias[n]?) for one output tile, 4x4
 * blocks with 1-wide tails. `B` is indexed from n0 (a dequantized tile may start at row 0). */
void n_gemm_tile(plow_bf16* C, size_t ldc, const plow_bf16* A, const plow_bf16* B, size_t b_row0,
                 const plow_bf16* bias, const float* rs, const float* cs, uint32_t K, uint32_t m0,
                 uint32_t m1, uint32_t n0, uint32_t n1) {
    float o[4][4];
    for (uint32_t m = m0; m < m1; m += 4) {
        const uint32_t rm = m1 - m < 4u ? m1 - m : 4u;
        const plow_bf16* a = A + (size_t)m * K;
        for (uint32_t n = n0; n < n1; n += 4) {
            const uint32_t rn = n1 - n < 4u ? n1 - n : 4u;
            const plow_bf16* b = B + (size_t)(n - b_row0) * K;
            if (rm == 4u && rn == 4u) mk(a, K, b, K, K, 4, 4, o);
            else if (rn == 4u) {
                for (uint32_t r = 0; r < rm; r++) {
                    float t[4][4];
                    mk(a + (size_t)r * K, K, b, K, K, 1, 4, t);
                    for (uint32_t c = 0; c < 4; c++) o[r][c] = t[0][c];
                }
            } else if (rm == 4u) {
                for (uint32_t c = 0; c < rn; c++) {
                    float t[4][4];
                    mk(a, K, b + (size_t)c * K, K, K, 4, 1, t);
                    for (uint32_t r = 0; r < 4; r++) o[r][c] = t[r][0];
                }
            } else {
                for (uint32_t r = 0; r < rm; r++)
                    for (uint32_t c = 0; c < rn; c++) {
                        float t[4][4];
                        mk(a + (size_t)r * K, K, b + (size_t)c * K, K, K, 1, 1, t);
                        o[r][c] = t[0][0];
                    }
            }
            for (uint32_t r = 0; r < rm; r++)
                for (uint32_t c = 0; c < rn; c++) {
                    const float bb = bias ? plow_bf2f(bias[n + c]) : 0.0f;
                    const float sc = (rs ? rs[m + r] : 1.0f) * (cs ? cs[n + c] : 1.0f);
                    C[(size_t)(m + r) * ldc + n + c] = plow_f2bf(o[r][c] * sc + bb);
                }
        }
    }
}

/* t0=C t1=A t2=B t7=bias?  i0=M i1=N i2=K i4=a_row0 i5=c_row0 */
static void gemm_op(const PlowDevInst* in, void* const* T, uint32_t BM, uint32_t BN,
                    uint32_t slice, uint32_t nblk) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    plow_bf16* C = (plow_bf16*)PLOW_CPU_TEN(in, T, 0) + (size_t)in->i[5] * N;
    const plow_bf16* A = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const plow_bf16* B = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 7);
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        n_gemm_tile(C, N, A, B, 0, bias, NULL, NULL, K, m0, m1, n0, n1);
    }
}

N_K(n_gemm)       { (void)ctx; gemm_op(in, T, 256, 256, slice, nblk); }
N_K(n_gemm_small) { (void)ctx; gemm_op(in, T, 64, 128, slice, nblk); }
N_K(n_gemm_med)   { (void)ctx; gemm_op(in, T, 128, 128, slice, nblk); }
N_K(n_gemm_wide)  { (void)ctx; gemm_op(in, T, 128, 256, slice, nblk); }
N_K(n_gemm_c5)    { (void)ctx; gemm_op(in, T, 192, 256, slice, nblk); }

/* t0=fu t1=x t2=W_gate t5=W_up t6=bias_gate? t7=bias_up?  i0=M i1=N i2=K i5=act  f0/f1 = act
 * immediates. The fused 256x256 tile emits BN/2 = 128 output columns. */
/* One GLU tile: C[m][n] = pair(A[m].Wg[n] * gs[n]? + bg[n]?, A[m].Wu[n] * us[n]? + bu[n]?).
 * Wg/Wu are indexed from `w_row0` (a dequantized tile starts at row 0). */
void n_gemm_glu_tile(plow_bf16* C, size_t ldc, const plow_bf16* A, const plow_bf16* Wg,
                     const plow_bf16* Wu, size_t w_row0, const plow_bf16* bg, const plow_bf16* bu,
                     const float* gs, const float* us, uint32_t K, uint32_t m0, uint32_t m1,
                     uint32_t n0, uint32_t n1, uint32_t act, float f0, float f1) {
    float g[4][4], u[4][4];
    for (uint32_t m = m0; m < m1; m++) {
        const plow_bf16* a = A + (size_t)m * K;
        for (uint32_t n = n0; n < n1; n += 4) {
            const uint32_t rn = n1 - n < 4u ? n1 - n : 4u;
            const plow_bf16* wg = Wg + (size_t)(n - w_row0) * K;
            const plow_bf16* wu = Wu + (size_t)(n - w_row0) * K;
            if (rn == 4u) {
                mk(a, K, wg, K, K, 1, 4, g);
                mk(a, K, wu, K, K, 1, 4, u);
            } else {
                for (uint32_t c = 0; c < rn; c++) {
                    float t[4][4];
                    mk(a, K, wg + (size_t)c * K, K, K, 1, 1, t);
                    g[0][c] = t[0][0];
                    mk(a, K, wu + (size_t)c * K, K, K, 1, 1, t);
                    u[0][c] = t[0][0];
                }
                for (uint32_t c = rn; c < 4; c++) g[0][c] = u[0][c] = 0.0f;
            }
            for (uint32_t c = 0; c < rn; c++) {
                if (gs) g[0][c] *= gs[n + c];
                if (us) u[0][c] *= us[n + c];
                if (bg) g[0][c] += plow_bf2f(bg[n + c]);
                if (bu) u[0][c] += plow_bf2f(bu[n + c]);
            }
            const float32x4_t o = n_glu_pair(vld1q_f32(g[0]), vld1q_f32(u[0]), act, f0, f1);
            n_store4_n(C + (size_t)m * ldc + n, o, rn);
        }
    }
}

N_K(n_gemm_glu) {
    (void)ctx;
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* A = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* Wg = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* Wu = PLOW_CPU_TEN(in, T, 5);
    const plow_bf16* bg = PLOW_CPU_TEN(in, T, 6);
    const plow_bf16* bu = PLOW_CPU_TEN(in, T, 7);
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    const uint32_t BM = 256, BN = 128;
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        n_gemm_glu_tile(C, N, A, Wg, Wu, 0, bg, bu, NULL, NULL, K, m0, m1, n0, n1, act, f0, f1);
    }
}

/* GEMV family, NEON bf16 — the decode hot path, bandwidth-bound.
 *
 * Weights are row-major bf16 W[N][K] as loaded. Inner loop: `vbfdotq_f32` over 8 bf16 of K per
 * step, RB weight rows x M activation rows of independent accumulators (RB*M <= 16 of the 32
 * NEON registers), M=1 additionally splits K in two for 8 accumulators, f32 accumulate, one
 * horizontal reduce per output. Slicing is golden's GV_BLOCKED contiguous column ownership
 * (g_range), so the fine gemv->headnorm dependency map holds. Mirrors avx512/gemv.c. */
#include "neon.h"

#define UNROLL _Pragma("clang loop unroll(full)")

/* out[r*M + m] = W[r] . X[m] for r < RB, m < M; RB/M are compile-time at every call site. */
static inline __attribute__((always_inline)) void dot_rm(const plow_bf16* W, size_t ldw,
                                                         const plow_bf16* X, size_t ldx,
                                                         uint32_t K, const uint32_t RB,
                                                         const uint32_t M, float* out) {
    float32x4_t acc[4][8];
    UNROLL
    for (uint32_t r = 0; r < RB; r++)
        UNROLL
        for (uint32_t m = 0; m < M; m++) acc[r][m] = vdupq_n_f32(0);
    uint32_t k = 0;
    for (; k + 8 <= K; k += 8) {
        bfloat16x8_t xv[8];
        UNROLL
        for (uint32_t m = 0; m < M; m++) xv[m] = n_ldbh(X + m * ldx + k);
        UNROLL
        for (uint32_t r = 0; r < RB; r++) {
            const bfloat16x8_t wv = n_ldbh(W + r * ldw + k);
            UNROLL
            for (uint32_t m = 0; m < M; m++) acc[r][m] = vbfdotq_f32(acc[r][m], wv, xv[m]);
        }
    }
    UNROLL
    for (uint32_t r = 0; r < RB; r++)
        UNROLL
        for (uint32_t m = 0; m < M; m++) {
            float s = n_hsum(acc[r][m]);
            for (uint32_t kk = k; kk < K; kk++)
                s += plow_bf2f(W[r * ldw + kk]) * plow_bf2f(X[m * ldx + kk]);
            out[r * M + m] = s;
        }
}

/* M = 1, 4 rows, K split in two interleaved partial sums: 8 accumulators. */
static inline __attribute__((always_inline)) void dot_m1_r4(const plow_bf16* W, size_t ldw,
                                                            const plow_bf16* x, uint32_t K,
                                                            float* out) {
    float32x4_t acc[4][2];
    UNROLL
    for (uint32_t r = 0; r < 4; r++) acc[r][0] = acc[r][1] = vdupq_n_f32(0);
    uint32_t k = 0;
    for (; k + 16 <= K; k += 16) {
        const bfloat16x8_t x0 = n_ldbh(x + k), x1 = n_ldbh(x + k + 8);
        UNROLL
        for (uint32_t r = 0; r < 4; r++) {
            const plow_bf16* w = W + r * ldw + k;
            acc[r][0] = vbfdotq_f32(acc[r][0], n_ldbh(w), x0);
            acc[r][1] = vbfdotq_f32(acc[r][1], n_ldbh(w + 8), x1);
        }
    }
    for (; k + 8 <= K; k += 8) {
        const bfloat16x8_t x0 = n_ldbh(x + k);
        UNROLL
        for (uint32_t r = 0; r < 4; r++) acc[r][0] = vbfdotq_f32(acc[r][0], n_ldbh(W + r * ldw + k), x0);
    }
    UNROLL
    for (uint32_t r = 0; r < 4; r++) {
        float s = n_hsum(vaddq_f32(acc[r][0], acc[r][1]));
        for (uint32_t kk = k; kk < K; kk++) s += plow_bf2f(W[r * ldw + kk]) * plow_bf2f(x[kk]);
        out[r] = s;
    }
}

#define DOT_CASE(RB_, M_) \
    case (RB_) * 16 + (M_): dot_rm(W, ldw, X, ldx, K, RB_, M_, out); break;

/* Row block width for M: 4 rows while RB*M <= 16, else 2. */
static inline uint32_t rb_for(uint32_t M) { return M <= 4u ? 4u : 2u; }

/* out[r*M + m] = W[r] . X[m], RB in {1,2,4}, M in 1..8. */
static void gemv_rows(const plow_bf16* W, size_t ldw, const plow_bf16* X, size_t ldx, uint32_t K,
                      uint32_t RB, uint32_t M, float* out) {
    if (RB == 4u && M == 1u) { dot_m1_r4(W, ldw, X, K, out); return; }
    switch (RB * 16 + M) {
        DOT_CASE(4, 2) DOT_CASE(4, 3) DOT_CASE(4, 4)
        DOT_CASE(2, 5) DOT_CASE(2, 6) DOT_CASE(2, 7) DOT_CASE(2, 8)
        DOT_CASE(1, 1) DOT_CASE(1, 2) DOT_CASE(1, 3) DOT_CASE(1, 4)
        DOT_CASE(1, 5) DOT_CASE(1, 6) DOT_CASE(1, 7) DOT_CASE(1, 8)
        default:
            for (uint32_t r = 0; r < RB; r++)
                for (uint32_t m = 0; m < M; m++) dot_rm(W + r * ldw, ldw, X + m * ldx, ldx, K, 1, 1, out + r * M + m);
    }
}

/* norm==1: acc = sum_k w*x*gamma in f32 (no intermediate bf16), then * rms[m]. xg[m][k] f32 is
 * staged in scratch; two weight rows per step. */
static inline __attribute__((always_inline)) void dotf_rm(const plow_bf16* W, size_t ldw,
                                                          const float* XG, size_t ldx, uint32_t K,
                                                          const uint32_t RB, const uint32_t M,
                                                          float* out) {
    float32x4_t acc[2][8];
    UNROLL
    for (uint32_t r = 0; r < RB; r++)
        UNROLL
        for (uint32_t m = 0; m < M; m++) acc[r][m] = vdupq_n_f32(0);
    uint32_t k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t xv[8];
        UNROLL
        for (uint32_t m = 0; m < M; m++) xv[m] = vld1q_f32(XG + m * ldx + k);
        UNROLL
        for (uint32_t r = 0; r < RB; r++) {
            const float32x4_t wv = n_load4(W + r * ldw + k);
            UNROLL
            for (uint32_t m = 0; m < M; m++) acc[r][m] = vfmaq_f32(acc[r][m], wv, xv[m]);
        }
    }
    UNROLL
    for (uint32_t r = 0; r < RB; r++)
        UNROLL
        for (uint32_t m = 0; m < M; m++) {
            float s = n_hsum(acc[r][m]);
            for (uint32_t kk = k; kk < K; kk++) s += plow_bf2f(W[r * ldw + kk]) * XG[m * ldx + kk];
            out[r * M + m] = s;
        }
}

#define DOTF_CASE(RB_, M_) \
    case (RB_) * 16 + (M_): dotf_rm(W, ldw, XG, ldx, K, RB_, M_, out); break;

static void gemvf_rows(const plow_bf16* W, size_t ldw, const float* XG, size_t ldx, uint32_t K,
                       uint32_t RB, uint32_t M, float* out) {
    switch (RB * 16 + M) {
        DOTF_CASE(2, 1) DOTF_CASE(2, 2) DOTF_CASE(2, 3) DOTF_CASE(2, 4)
        DOTF_CASE(2, 5) DOTF_CASE(2, 6) DOTF_CASE(2, 7) DOTF_CASE(2, 8)
        DOTF_CASE(1, 1) DOTF_CASE(1, 2) DOTF_CASE(1, 3) DOTF_CASE(1, 4)
        DOTF_CASE(1, 5) DOTF_CASE(1, 6) DOTF_CASE(1, 7) DOTF_CASE(1, 8)
        default: break;
    }
}

/* Stage the norm==2 / q-norm operand: xn[m] = bf16(x[m] * rsqrt(mean(x^2) + eps) * gamma). */
static void prenorm_rows(plow_bf16* xn, const plow_bf16* x, const plow_bf16* gamma, uint32_t M,
                         uint32_t K, float eps) {
    for (uint32_t m = 0; m < M; m++) {
        const plow_bf16* xm = x + (size_t)m * K;
        n_scale_row(xn + (size_t)m * K, xm, gamma, g_rsqrt(n_row_ss(xm, K) / (float)K + eps), K);
    }
}

/* C[m][n0..n1) = X[m] . W[n]^T (+ bias[n]), plain bf16 store. */
static void gemv_span(plow_bf16* C, size_t ldc, const plow_bf16* W, const plow_bf16* X,
                      size_t ldx, uint32_t M, uint32_t K, uint32_t n0, uint32_t n1,
                      const plow_bf16* bias) {
    float out[32];
    const uint32_t RB = rb_for(M);
    uint32_t n = n0;
    for (; n + RB <= n1; n += RB) {
        gemv_rows(W + (size_t)n * K, K, X, ldx, K, RB, M, out);
        for (uint32_t r = 0; r < RB; r++) {
            const float b = bias ? plow_bf2f(bias[n + r]) : 0.0f;
            for (uint32_t m = 0; m < M; m++) C[m * ldc + n + r] = plow_f2bf(out[r * M + m] + b);
        }
    }
    for (; n < n1; n++) {
        gemv_rows(W + (size_t)n * K, K, X, ldx, K, 1, M, out);
        const float b = bias ? plow_bf2f(bias[n]) : 0.0f;
        for (uint32_t m = 0; m < M; m++) C[m * ldc + n] = plow_f2bf(out[m] + b);
    }
}

/* t0=C t1=x t2=W t3=rms? t4=gamma? t7=bias?  i0=M i1=N i2=K i3=norm i4=a_row0  f0=eps */
N_K(n_gemv) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], norm = in->i[3];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const plow_bf16* W = PLOW_CPU_TEN(in, T, 2);
    const float* rms = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 7);
    const float eps = in->fj[0].f;
    if (M == 0u || M > 8u) { g_gemv(in, slice, nblk, T, ctx); return; }
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    if (norm == 1u) {
        const size_t need = (size_t)M * K * sizeof(float);
        if (!ctx || !ctx->scratch || ctx->scratch_bytes < need) { g_gemv(in, slice, nblk, T, ctx); return; }
        float* XG = ctx->scratch;
        for (uint32_t m = 0; m < M; m++) {
            uint32_t k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t v = n_load4(x + (size_t)m * K + k);
                if (gamma) v = vmulq_f32(v, n_load4(gamma + k));
                vst1q_f32(XG + (size_t)m * K + k, v);
            }
            for (; k < K; k++)
                XG[(size_t)m * K + k] = plow_bf2f(x[(size_t)m * K + k]) * (gamma ? plow_bf2f(gamma[k]) : 1.0f);
        }
        float out[16];
        uint32_t n = n0;
        for (; n + 2 <= n1; n += 2) {
            gemvf_rows(W + (size_t)n * K, K, XG, K, K, 2, M, out);
            for (uint32_t r = 0; r < 2; r++) {
                const float b = bias ? plow_bf2f(bias[n + r]) : 0.0f;
                for (uint32_t m = 0; m < M; m++) C[m * N + n + r] = plow_f2bf(out[r * M + m] * rms[m] + b);
            }
        }
        for (; n < n1; n++) {
            gemvf_rows(W + (size_t)n * K, K, XG, K, K, 1, M, out);
            const float b = bias ? plow_bf2f(bias[n]) : 0.0f;
            for (uint32_t m = 0; m < M; m++) C[m * N + n] = plow_f2bf(out[m] * rms[m] + b);
        }
        return;
    }
    const plow_bf16* X = x;
    if (norm == 2u) {
        const size_t need = (size_t)M * K * sizeof(plow_bf16);
        if (!ctx || !ctx->scratch || ctx->scratch_bytes < need) { g_gemv(in, slice, nblk, T, ctx); return; }
        prenorm_rows(ctx->scratch, x, gamma, M, K, eps);
        X = ctx->scratch;
    }
    gemv_span(C, N, W, X, K, M, K, n0, n1, bias);
}

/* t0=fu t1=x t2=W_gate t5=W_up t6=bias_gate? t7=bias_up?  i0=M i1=N i2=K i5=act  f0/f1 = act
 * immediates. */
N_K(n_gemv_glu) {
    (void)ctx;
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* Wg = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* Wu = PLOW_CPU_TEN(in, T, 5);
    const plow_bf16* bg = PLOW_CPU_TEN(in, T, 6);
    const plow_bf16* bu = PLOW_CPU_TEN(in, T, 7);
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    if (M == 0u || M > 8u) { g_gemv_glu(in, slice, nblk, T, ctx); return; }
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    const uint32_t RB = rb_for(M);
    float g[32], u[32], of[32];
    for (uint32_t n = n0; n < n1;) {
        const uint32_t rb = n + RB <= n1 ? RB : 1u;
        gemv_rows(Wg + (size_t)n * K, K, x, K, K, rb, M, g);
        gemv_rows(Wu + (size_t)n * K, K, x, K, K, rb, M, u);
        if (bg || bu)
            for (uint32_t r = 0; r < rb; r++)
                for (uint32_t m = 0; m < M; m++) {
                    if (bg) g[r * M + m] += plow_bf2f(bg[n + r]);
                    if (bu) u[r * M + m] += plow_bf2f(bu[n + r]);
                }
        const uint32_t cnt = rb * M; /* <= 16 */
        for (uint32_t i = 0; i < cnt; i += 4) {
            float gt[4] = {0, 0, 0, 0}, ut[4] = {0, 0, 0, 0};
            const uint32_t c = cnt - i < 4u ? cnt - i : 4u;
            for (uint32_t j = 0; j < c; j++) { gt[j] = g[i + j]; ut[j] = u[i + j]; }
            vst1q_f32(of + i, n_glu_pair(vld1q_f32(gt), vld1q_f32(ut), act, f0, f1));
        }
        for (uint32_t r = 0; r < rb; r++)
            for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n + r] = plow_f2bf(of[r * M + m]);
        n += rb;
    }
}

/* t0=q t1=x t2=W_q t3=k t4=W_k t5=v t6=W_v t7=q-norm gamma?  i0=M i1=Nq i2=K i3=Nk i4=Nv
 * i5/i6/i7 = bias_q/k/v tensor handles (0 = absent)  f0=eps */
N_K(n_gemv_qkv) {
    const uint32_t M = in->i[0], Nq = in->i[1], K = in->i[2], Nk = in->i[3], Nv = in->i[4];
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* gnorm = PLOW_CPU_TEN(in, T, 7);
    const float eps = in->fj[0].f;
    plow_bf16* Cs[3] = {PLOW_CPU_TEN(in, T, 0), PLOW_CPU_TEN(in, T, 3), PLOW_CPU_TEN(in, T, 5)};
    const plow_bf16* Ws[3] = {PLOW_CPU_TEN(in, T, 2), PLOW_CPU_TEN(in, T, 4), PLOW_CPU_TEN(in, T, 6)};
    const plow_bf16* Bs[3] = {G_QKV_BIAS(in, T, 5), G_QKV_BIAS(in, T, 6), G_QKV_BIAS(in, T, 7)};
    const uint32_t Ns[3] = {Nq, Nk, Nv};
    if (M == 0u || M > 8u) { g_gemv_qkv(in, slice, nblk, T, ctx); return; }
    const plow_bf16* X = x;
    if (gnorm) {
        const size_t need = (size_t)M * K * sizeof(plow_bf16);
        if (!ctx || !ctx->scratch || ctx->scratch_bytes < need) { g_gemv_qkv(in, slice, nblk, T, ctx); return; }
        prenorm_rows(ctx->scratch, x, gnorm, M, K, eps);
        X = ctx->scratch;
    }
    uint32_t n0, n1;
    g_range(Nq + Nk + Nv, slice, nblk, &n0, &n1);
    uint32_t S0 = 0;
    for (uint32_t s = 0; s < 3; s++) {
        const uint32_t S1 = S0 + Ns[s];
        const uint32_t a = n0 > S0 ? n0 : S0, b = n1 < S1 ? n1 : S1;
        if (a < b) gemv_span(Cs[s], Ns[s], Ws[s], X, K, M, K, a - S0, b - S0, Bs[s]);
        S0 = S1;
    }
}

/* t0=C(logits) t1=x t2=W t3=part(u64[nblk])  i0=1 i1=N i2=K i4=a_row0  f0=cap (0 = none)
 * GEMV -> bf16 -> SOFTCAP -> bf16 -> ARGMAX partial; 4 columns per epilogue. */
N_K(n_gemv_argmax) {
    (void)ctx;
    const uint32_t N = in->i[1], K = in->i[2];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const plow_bf16* W = PLOW_CPU_TEN(in, T, 2);
    uint64_t* part = PLOW_CPU_TEN(in, T, 3);
    const float cap = in->fj[0].f;
    const float32x4_t vcap = vdupq_n_f32(cap), vinv = vdupq_n_f32(cap > 0.0f ? 1.0f / cap : 0.0f);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    uint64_t best = 0;
    float acc[4];
    uint32_t n = n0;
    for (; n + 4 <= n1; n += 4) {
        dot_m1_r4(W + (size_t)n * K, K, x, K, acc);
        float32x4_t lg = n_round4(vld1q_f32(acc));
        if (cap > 0.0f) lg = vmulq_f32(vcap, n_tanh(vmulq_f32(lg, vinv)));
        n_store4(C + n, lg);
        for (uint32_t j = 0; j < 4; j++) {
            const uint64_t key = g_amax_pack(C[n + j], n + j);
            best = key > best ? key : best;
        }
    }
    for (; n < n1; n++) {
        float o;
        gemv_rows(W + (size_t)n * K, K, x, K, K, 1, 1, &o);
        const plow_bf16 lg = plow_f2bf(o);
        const plow_bf16 sc = cap > 0.0f ? plow_f2bf(cap * tanhf(plow_bf2f(lg) / cap)) : lg;
        C[n] = sc;
        const uint64_t key = g_amax_pack(sc, n);
        best = key > best ? key : best;
    }
    part[slice] = best;
}

/* Registrar lives in this translation unit (see avx512/gemv.c for why: a standalone
 * register.o is never extracted from the static archive). Called by plow_cpu_init after the
 * golden registrar when the host reports FEAT_BF16 + FEAT_I8MM. */
void plow_cpu_register_neon(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_GEMV] = n_gemv;
    tab[PLOW_DOP_GEMV_GLU] = n_gemv_glu;
    tab[PLOW_DOP_GEMV_QKV] = n_gemv_qkv;
    tab[PLOW_DOP_GEMV_ARGMAX] = n_gemv_argmax;
    tab[PLOW_DOP_GEMM] = n_gemm;
    tab[PLOW_DOP_GEMM_SMALL] = n_gemm_small;
    tab[PLOW_DOP_GEMM_MED] = n_gemm_med;
    tab[PLOW_DOP_GEMM_WIDE] = n_gemm_wide;
    tab[PLOW_DOP_GEMM_C5] = n_gemm_c5;
    tab[PLOW_DOP_GEMM_GLU] = n_gemm_glu;
    tab[PLOW_DOP_RESIDUAL] = n_residual;
    tab[PLOW_DOP_GLU] = n_glu;
    tab[PLOW_DOP_SOFTCAP] = n_softcap;
    tab[PLOW_DOP_EMBED] = n_embed;
    tab[PLOW_DOP_ARGMAX] = n_argmax;
    tab[PLOW_DOP_ARGMAX_FIN] = n_argmax_fin;
    tab[PLOW_DOP_RMSNORM] = n_rmsnorm;
    tab[PLOW_DOP_ROWRMS] = n_rowrms;
    tab[PLOW_DOP_LAYERNORM] = n_layernorm;
    tab[PLOW_DOP_NORM_RESIDUAL] = n_norm_residual;
    tab[PLOW_DOP_ADD_NORM] = n_add_norm;
    tab[PLOW_DOP_NORM_RESIDUAL_NORM] = n_norm_residual_norm;
    tab[PLOW_DOP_HEADNORM_ROPE] = n_headnorm_rope;
    tab[PLOW_DOP_FLASH_PREFILL] = n_flash_prefill;
    tab[PLOW_DOP_FLASH_DECODE] = n_flash_decode;
    tab[PLOW_DOP_FLASH_MERGE] = n_flash_merge;
}


/* GEMV family, AVX-512 BF16 — the decode hot path, bandwidth-bound.
 *
 * Weights are row-major bf16 W[N][K] as loaded (the VNNI prepack lands with the AMX tier; the
 * block shape below is the same 32-column unit, so a prepacked path slots in beside `dot_rm`).
 * Inner loop (spec §2.2 / §6.6): `vdpbf16ps` over 32 bf16 of K per step, RB weight rows x M
 * activation rows of independent accumulators (RB*M <= 16), M=1 additionally splits K in two
 * for 8 accumulators (llama.cpp), `prefetcht0` 512 B ahead of every weight row (oneDNN's tuned
 * distance), f32 accumulate, one horizontal reduce per output. Slicing is golden's GV_BLOCKED
 * contiguous column ownership (g_range), so the fine gemv->headnorm dependency map holds. */
#include "avx512.h"

#define PF_DIST 256u /* bf16 elements = 512 B */

static inline __m512bh ldbh(const plow_bf16* p) { return (__m512bh)_mm512_loadu_si512(p); }
static inline __m512bh ldbh_mask(const plow_bf16* p, __mmask32 m) {
    return (__m512bh)_mm512_maskz_loadu_epi16(m, p);
}

/* out[r*M + m] = W[r] . X[m] for r < RB, m < M. RB/M are constants at every call site
 * (see gemv_rows), so the acc array becomes registers after unrolling. */
static inline __attribute__((always_inline)) void dot_rm(const plow_bf16* W, size_t ldw,
                                                         const plow_bf16* X, size_t ldx,
                                                         uint32_t K, const uint32_t RB,
                                                         const uint32_t M, float* out) {
    __m512 acc[4][8];
    for (uint32_t r = 0; r < RB; r++)
        for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_setzero_ps();
    uint32_t k = 0;
    for (; k + 32 <= K; k += 32) {
        __m512bh xv[8];
        for (uint32_t m = 0; m < M; m++) xv[m] = ldbh(X + m * ldx + k);
        for (uint32_t r = 0; r < RB; r++) {
            const plow_bf16* w = W + r * ldw + k;
            _mm_prefetch((const char*)(w + PF_DIST), _MM_HINT_T0);
            const __m512bh wv = ldbh(w);
            for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_dpbf16_ps(acc[r][m], wv, xv[m]);
        }
    }
    if (k < K) {
        const __mmask32 mk = (__mmask32)((1u << (K - k)) - 1u);
        __m512bh xv[8];
        for (uint32_t m = 0; m < M; m++) xv[m] = ldbh_mask(X + m * ldx + k, mk);
        for (uint32_t r = 0; r < RB; r++) {
            const __m512bh wv = ldbh_mask(W + r * ldw + k, mk);
            for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_dpbf16_ps(acc[r][m], wv, xv[m]);
        }
    }
    for (uint32_t r = 0; r < RB; r++)
        for (uint32_t m = 0; m < M; m++) out[r * M + m] = _mm512_reduce_add_ps(acc[r][m]);
}

/* M = 1, 4 rows, K split in two interleaved partial sums: 8 accumulators (spec §2.2). */
static inline __attribute__((always_inline)) void dot_m1_r4(const plow_bf16* W, size_t ldw,
                                                            const plow_bf16* x, uint32_t K,
                                                            float* out) {
    __m512 acc[4][2];
    for (uint32_t r = 0; r < 4; r++) acc[r][0] = acc[r][1] = _mm512_setzero_ps();
    uint32_t k = 0;
    for (; k + 64 <= K; k += 64) {
        const __m512bh x0 = ldbh(x + k), x1 = ldbh(x + k + 32);
        for (uint32_t r = 0; r < 4; r++) {
            const plow_bf16* w = W + r * ldw + k;
            _mm_prefetch((const char*)(w + PF_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(w + PF_DIST + 32), _MM_HINT_T0);
            acc[r][0] = _mm512_dpbf16_ps(acc[r][0], ldbh(w), x0);
            acc[r][1] = _mm512_dpbf16_ps(acc[r][1], ldbh(w + 32), x1);
        }
    }
    for (; k + 32 <= K; k += 32) {
        const __m512bh x0 = ldbh(x + k);
        for (uint32_t r = 0; r < 4; r++) acc[r][0] = _mm512_dpbf16_ps(acc[r][0], ldbh(W + r * ldw + k), x0);
    }
    if (k < K) {
        const __mmask32 mk = (__mmask32)((1u << (K - k)) - 1u);
        const __m512bh x0 = ldbh_mask(x + k, mk);
        for (uint32_t r = 0; r < 4; r++)
            acc[r][1] = _mm512_dpbf16_ps(acc[r][1], ldbh_mask(W + r * ldw + k, mk), x0);
    }
    for (uint32_t r = 0; r < 4; r++) out[r] = _mm512_reduce_add_ps(_mm512_add_ps(acc[r][0], acc[r][1]));
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
        default: /* unreachable for M <= 8 */
            for (uint32_t r = 0; r < RB; r++)
                for (uint32_t m = 0; m < M; m++) dot_rm(W + r * ldw, ldw, X + m * ldx, ldx, K, 1, 1, out + r * M + m);
    }
}

/* norm==1: acc = sum_k w*x*gamma in f32 (no intermediate bf16), then * rms[m]. xg[m][k] f32
 * is staged in scratch; two weight rows per step, RB*M <= 16 accumulators. */
static inline __attribute__((always_inline)) void dotf_rm(const plow_bf16* W, size_t ldw,
                                                          const float* XG, size_t ldx, uint32_t K,
                                                          const uint32_t RB, const uint32_t M,
                                                          float* out) {
    __m512 acc[2][8];
    for (uint32_t r = 0; r < RB; r++)
        for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_setzero_ps();
    for (uint32_t k = 0; k < K; k += 16) {
        const __mmask16 mk = k + 16 <= K ? 0xFFFF : v_tail16(K - k);
        __m512 xv[8];
        for (uint32_t m = 0; m < M; m++) xv[m] = _mm512_maskz_loadu_ps(mk, XG + m * ldx + k);
        for (uint32_t r = 0; r < RB; r++) {
            const plow_bf16* w = W + r * ldw + k;
            _mm_prefetch((const char*)(w + PF_DIST), _MM_HINT_T0);
            const __m512 wv = v_load_bf16_mask(w, mk);
            for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_fmadd_ps(wv, xv[m], acc[r][m]);
        }
    }
    for (uint32_t r = 0; r < RB; r++)
        for (uint32_t m = 0; m < M; m++) out[r * M + m] = _mm512_reduce_add_ps(acc[r][m]);
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
        v_scale_row(xn + (size_t)m * K, xm, gamma, g_rsqrt(v_row_ss(xm, K) / (float)K + eps), K);
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
V_K(v_gemv) {
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
        for (uint32_t m = 0; m < M; m++)
            for (uint32_t k = 0; k < K; k += 16) {
                const __mmask16 mk = k + 16 <= K ? 0xFFFF : v_tail16(K - k);
                __m512 v = v_load_bf16_mask(x + (size_t)m * K + k, mk);
                if (gamma) v = _mm512_mul_ps(v, v_load_bf16_mask(gamma + k, mk));
                _mm512_mask_storeu_ps(XG + (size_t)m * K + k, mk, v);
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
 * immediates (situ beta/lbeta, swiglu_oai alpha/limit). */
V_K(v_gemv_glu) {
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
    float g[32], u[32];
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
        const __mmask16 mk = v_tail16(cnt);
        const __m512 gv = _mm512_maskz_loadu_ps(mk, g), uv = _mm512_maskz_loadu_ps(mk, u);
        const __m512 o = v_glu_pair(gv, uv, act, f0, f1);
        float of[16];
        _mm512_mask_storeu_ps(of, mk, o);
        for (uint32_t r = 0; r < rb; r++)
            for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n + r] = plow_f2bf(of[r * M + m]);
        n += rb;
    }
}

/* t0=q t1=x t2=W_q t3=k t4=W_k t5=v t6=W_v t7=q-norm gamma?  i0=M i1=Nq i2=K i3=Nk i4=Nv
 * i5/i6/i7 = bias_q/k/v tensor handles (0 = absent)  f0=eps
 * Blocked ownership over the concatenated q|k|v span; t7 = the q_a_layernorm fold. */
V_K(v_gemv_qkv) {
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
 * GEMV -> bf16 -> SOFTCAP -> bf16 -> ARGMAX partial; 16 columns per epilogue. */
V_K(v_gemv_argmax) {
    (void)ctx;
    const uint32_t N = in->i[1], K = in->i[2];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const plow_bf16* W = PLOW_CPU_TEN(in, T, 2);
    uint64_t* part = PLOW_CPU_TEN(in, T, 3);
    const float cap = in->fj[0].f;
    const __m512 vcap = _mm512_set1_ps(cap), vinv = _mm512_set1_ps(cap > 0.0f ? 1.0f / cap : 0.0f);
    const __m512i lane = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    __m512i bk = _mm512_setzero_si512(), bi = _mm512_setzero_si512();
    float acc[16];
    uint32_t n = n0;
    for (; n + 16 <= n1; n += 16) {
        for (uint32_t r = 0; r < 16; r += 4) dot_m1_r4(W + (size_t)(n + r) * K, K, x, K, acc + r);
        __m512 lg = v_round_bf16(_mm512_loadu_ps(acc));
        if (cap > 0.0f) lg = _mm512_mul_ps(vcap, v_tanh(_mm512_mul_ps(lg, vinv)));
        const __m256bh sc = _mm512_cvtneps_pbh(lg);
        _mm256_storeu_si256((__m256i*)(C + n), (__m256i)sc);
        const __m512i k = v_amax_key(_mm512_cvtepu16_epi32((__m256i)sc));
        const __mmask16 gt = _mm512_cmpgt_epu32_mask(k, bk);
        bk = _mm512_mask_mov_epi32(bk, gt, k);
        bi = _mm512_mask_mov_epi32(bi, gt, _mm512_add_epi32(_mm512_set1_epi32((int)n), lane));
    }
    uint64_t best = v_amax_fold(bk, bi, 0);
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

/* Registrar lives in this translation unit (not its own file) so that any reference to a
 * kernel symbol pulls the strong definition out of the static archive along with it — a
 * standalone register.o is never extracted when dispatch.o's weak default already
 * satisfies the symbol. Called by plow_cpu_init after the golden registrar when cpuid
 * reports AVX-512 F/BW/VL/BF16. */
void plow_cpu_register_avx512(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_RESIDUAL] = v_residual;
    tab[PLOW_DOP_GLU] = v_glu;
    tab[PLOW_DOP_SOFTCAP] = v_softcap;
    tab[PLOW_DOP_EMBED] = v_embed;
    tab[PLOW_DOP_ARGMAX] = v_argmax;
    tab[PLOW_DOP_ARGMAX_FIN] = v_argmax_fin;
    tab[PLOW_DOP_RMSNORM] = v_rmsnorm;
    tab[PLOW_DOP_ROWRMS] = v_rowrms;
    tab[PLOW_DOP_LAYERNORM] = v_layernorm;
    tab[PLOW_DOP_NORM_RESIDUAL] = v_norm_residual;
    tab[PLOW_DOP_ADD_NORM] = v_add_norm;
    tab[PLOW_DOP_NORM_RESIDUAL_NORM] = v_norm_residual_norm;
    tab[PLOW_DOP_GEMV] = v_gemv;
    tab[PLOW_DOP_GEMV_GLU] = v_gemv_glu;
    tab[PLOW_DOP_GEMV_QKV] = v_gemv_qkv;
    tab[PLOW_DOP_GEMV_ARGMAX] = v_gemv_argmax;
    v_register_attention(tab);
    v_register_gptoss(tab);
    v_register_moe_gemma(tab);
}

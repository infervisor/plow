/* GEMM / GEMV family: op_gemm.h semantics, f32 accumulate over K in order, bf16 store.
 *
 * GEMM slicing: output tiles of the op's nominal (BM, BN), linear tile id = slice + k*nblk,
 * row-major over (tm, tn) — the GPU's SWZ=0 order (its XCD/grouped-M swizzles are bijections
 * on the same tile set). GEMV slicing: GV_BLOCKED contiguous column runs, the ownership
 * PLOW_FINE's gemv->headnorm dependency map assumes. */
#include "golden.h"

static float dot_bf16(const plow_bf16* a, const plow_bf16* b, uint32_t k) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < k; i++) acc += plow_bf2f(a[i]) * plow_bf2f(b[i]);
    return acc;
}

/* C[M,N] = A[M,K] . B[N,K]^T over this slice's tiles. NORM: a' = bf16(a * rms[m] * gamma[k]). */
static void gemm_tiles(plow_bf16* C, const plow_bf16* A, const plow_bf16* B, const float* rms,
                       const plow_bf16* gamma, uint32_t M, uint32_t N, uint32_t K, uint32_t BM,
                       uint32_t BN, uint32_t slice, uint32_t nblk) {
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        for (uint32_t m = m0; m < m1; m++) {
            const plow_bf16* a = A + (size_t)m * K;
            for (uint32_t n = n0; n < n1; n++) {
                const plow_bf16* b = B + (size_t)n * K;
                float acc = 0.0f;
                if (rms) {
                    for (uint32_t k = 0; k < K; k++) {
                        const float av =
                            plow_bf2f(plow_f2bf(plow_bf2f(a[k]) * rms[m] * plow_bf2f(gamma[k])));
                        acc += av * plow_bf2f(b[k]);
                    }
                } else {
                    acc = dot_bf16(a, b, K);
                }
                C[(size_t)m * N + n] = plow_f2bf(acc);
            }
        }
    }
}

/* t0=C t1=A t2=B  i0=M i1=N i2=K i4=a_row0 i5=c_row0 */
static void gemm_op(const PlowDevInst* in, void* const* T, uint32_t BM, uint32_t BN,
                    uint32_t slice, uint32_t nblk) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    plow_bf16* C = (plow_bf16*)PLOW_CPU_TEN(in, T, 0) + (size_t)in->i[5] * N;
    const plow_bf16* A = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const plow_bf16* B = PLOW_CPU_TEN(in, T, 2);
    gemm_tiles(C, A, B, NULL, NULL, M, N, K, BM, BN, slice, nblk);
}

G_K(g_gemm)       { (void)ctx; gemm_op(in, T, 256, 256, slice, nblk); }
G_K(g_gemm_small) { (void)ctx; gemm_op(in, T, 64, 128, slice, nblk); }
G_K(g_gemm_med)   { (void)ctx; gemm_op(in, T, 128, 128, slice, nblk); }
G_K(g_gemm_wide)  { (void)ctx; gemm_op(in, T, 128, 256, slice, nblk); }
G_K(g_gemm_c5)    { (void)ctx; gemm_op(in, T, 192, 256, slice, nblk); }

/* t0=C t1=A t2=B t3=rms(f32) t4=gamma  i0=M i1=N i2=K */
G_K(g_gemm_norm) {
    (void)ctx;
    gemm_tiles(PLOW_CPU_TEN(in, T, 0), PLOW_CPU_TEN(in, T, 1), PLOW_CPU_TEN(in, T, 2),
               PLOW_CPU_TEN(in, T, 3), PLOW_CPU_TEN(in, T, 4), in->i[0], in->i[1], in->i[2], 256,
               256, slice, nblk);
}

/* t0=fu t1=x t2=W_gate t5=W_up  i0=M i1=N i2=K i5=act.  fu = act(x.Wg^T) * (x.Wu^T).
 * The fused 256x256 tile emits BN/2 = 128 output columns. */
G_K(g_gemm_glu) {
    (void)ctx;
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* A = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* Wg = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* Wu = PLOW_CPU_TEN(in, T, 5);
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    const uint32_t BM = 256, BN = 128;
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        for (uint32_t m = m0; m < m1; m++)
            for (uint32_t n = n0; n < n1; n++) {
                const float g = dot_bf16(A + (size_t)m * K, Wg + (size_t)n * K, K);
                const float u = dot_bf16(A + (size_t)m * K, Wu + (size_t)n * K, K);
                C[(size_t)m * N + n] = plow_f2bf(g_act_gate_only(g, act) * u);
            }
    }
}

/* Row RMS scalar of x[m] (the norm==2 / q-norm fold: the staged row is renormalized in place
 * as bf16(x * inv * gamma) before the plain dot). */
static float row_inv(const plow_bf16* x, uint32_t K, float eps) {
    float ss = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
        const float v = plow_bf2f(x[k]);
        ss += v * v;
    }
    return g_rsqrt(ss / (float)K + eps);
}

static float dot_normed(const plow_bf16* w, const plow_bf16* x, const plow_bf16* gamma, float inv,
                        uint32_t K) {
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
        const float g = gamma ? plow_bf2f(gamma[k]) : 1.0f;
        acc += plow_bf2f(w[k]) * plow_bf2f(plow_f2bf(plow_bf2f(x[k]) * inv * g));
    }
    return acc;
}

/* t0=C t1=x t2=W t3=rms? t4=gamma?  i0=M i1=N i2=K i3=norm i4=a_row0  f0=eps
 * norm 0: plain; 1: acc = (sum w*x*gamma) * rms[m]; 2: row RMS computed here, applied to x. */
G_K(g_gemv) {
    (void)ctx;
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], norm = in->i[3];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const plow_bf16* W = PLOW_CPU_TEN(in, T, 2);
    const float* rms = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 4);
    const float eps = in->fj[0].f;
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    for (uint32_t m = 0; m < M; m++) {
        const plow_bf16* xm = x + (size_t)m * K;
        const float inv = norm == 2u ? row_inv(xm, K, eps) : 1.0f;
        for (uint32_t n = n0; n < n1; n++) {
            const plow_bf16* w = W + (size_t)n * K;
            float acc;
            if (norm == 1u) {
                acc = 0.0f;
                for (uint32_t k = 0; k < K; k++) {
                    const float g = gamma ? plow_bf2f(gamma[k]) : 1.0f;
                    acc += plow_bf2f(w[k]) * plow_bf2f(xm[k]) * g;
                }
                acc *= rms[m];
            } else if (norm == 2u) {
                acc = dot_normed(w, xm, gamma, inv, K);
            } else {
                acc = dot_bf16(w, xm, K);
            }
            C[(size_t)m * N + n] = plow_f2bf(acc);
        }
    }
}

/* t0=fu t1=x t2=W_gate t5=W_up  i0=M i1=N i2=K i5=act  f0=beta f1=lbeta (situ, act==2) */
G_K(g_gemv_glu) {
    (void)ctx;
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* Wg = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* Wu = PLOW_CPU_TEN(in, T, 5);
    const float beta = in->fj[0].f, lbeta = in->fj[1].f;
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    for (uint32_t n = n0; n < n1; n++)
        for (uint32_t m = 0; m < M; m++) {
            const float g = dot_bf16(Wg + (size_t)n * K, x + (size_t)m * K, K);
            const float u = dot_bf16(Wu + (size_t)n * K, x + (size_t)m * K, K);
            const float o = act == 2u ? g_situ_gate(g, beta) * g_situ_up(u, lbeta)
                                      : g_act_gate_only(g, act) * u;
            C[(size_t)m * N + n] = plow_f2bf(o);
        }
}

/* t0=q t1=x t2=W_q t3=k t4=W_k t5=v t6=W_v t7=q-norm gamma?  i0=M i1=Nq i2=K i3=Nk i4=Nv  f0=eps
 * Blocked ownership over the concatenated q|k|v span. t7 present = the q_a_layernorm fold. */
G_K(g_gemv_qkv) {
    (void)ctx;
    const uint32_t M = in->i[0], Nq = in->i[1], K = in->i[2], Nk = in->i[3], Nv = in->i[4];
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* gnorm = PLOW_CPU_TEN(in, T, 7);
    const float eps = in->fj[0].f;
    plow_bf16* Cs[3] = {PLOW_CPU_TEN(in, T, 0), PLOW_CPU_TEN(in, T, 3), PLOW_CPU_TEN(in, T, 5)};
    const plow_bf16* Ws[3] = {PLOW_CPU_TEN(in, T, 2), PLOW_CPU_TEN(in, T, 4),
                              PLOW_CPU_TEN(in, T, 6)};
    const uint32_t Ns[3] = {Nq, Nk, Nv};
    uint32_t n0, n1;
    g_range(Nq + Nk + Nv, slice, nblk, &n0, &n1);
    for (uint32_t m = 0; m < M; m++) {
        const plow_bf16* xm = x + (size_t)m * K;
        const float inv = gnorm ? row_inv(xm, K, eps) : 1.0f;
        for (uint32_t n = n0; n < n1; n++) {
            uint32_t s = 0, col = n;
            while (col >= Ns[s]) { col -= Ns[s]; s++; }
            const plow_bf16* w = Ws[s] + (size_t)col * K;
            const float acc = gnorm ? dot_normed(w, xm, gnorm, inv, K) : dot_bf16(w, xm, K);
            Cs[s][(size_t)m * Ns[s] + col] = plow_f2bf(acc);
        }
    }
}

/* t0=C(logits) t1=x t2=W t3=part(u64[nblk])  i0=1 i1=N i2=K i4=a_row0  f0=cap (0 = none)
 * GEMV -> SOFTCAP -> ARGMAX partial, bit for bit (runtime/nvidia/op_gemm.cuh d_gemv_argmax). */
G_K(g_gemv_argmax) {
    (void)ctx;
    const uint32_t N = in->i[1], K = in->i[2];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const plow_bf16* W = PLOW_CPU_TEN(in, T, 2);
    uint64_t* part = PLOW_CPU_TEN(in, T, 3);
    const float cap = in->fj[0].f, inv = cap > 0.0f ? 1.0f / cap : 0.0f;
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    uint64_t best = 0;
    for (uint32_t n = n0; n < n1; n++) {
        const plow_bf16 lg = plow_f2bf(dot_bf16(W + (size_t)n * K, x, K));
        const plow_bf16 sc = cap > 0.0f ? plow_f2bf(cap * tanhf(plow_bf2f(lg) * inv)) : lg;
        C[n] = sc;
        const uint64_t key = g_amax_pack(sc, n);
        best = key > best ? key : best;
    }
    part[slice] = best;
}

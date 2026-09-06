/* cpu_dev_golden_test.c — the golden CPU kernels against inline naive f32 references.
 *
 * Every op runs over ALL slices for nblk in {1, 3}, so the slicing contract (disjoint,
 * complete) is what is tested, not just the math. bf16 outputs: 1e-2 relative (+1e-2 abs). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cpu_dev.h"

static int fails = 0;
#define CHECK(c, ...)                                                    \
    do {                                                                 \
        if (!(c)) {                                                      \
            fails++;                                                     \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                  \
            printf(__VA_ARGS__);                                         \
            printf("\n");                                                \
        }                                                                \
    } while (0)

static uint32_t rng = 0x9E3779B9u;
static float frand(void) { /* [-1, 1) */
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) & 0xFFFFFF) / 8388608.0f - 1.0f;
}
static void fill_bf16(plow_bf16* p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) p[i] = plow_f2bf(frand() * scale);
}
static int close_f(float got, float want) {
    const float tol = 1e-2f * fabsf(want) + 1e-2f;
    return fabsf(got - want) <= tol;
}
static int close_bf(plow_bf16 got, float want) { return close_f(plow_bf2f(got), want); }

static PlowDevInst inst(uint16_t op) {
    PlowDevInst in;
    memset(&in, 0, sizeof(in));
    in.op = op;
    for (int k = 0; k < 8; k++) in.t[k] = PLOW_TENSOR_NONE;
    return in;
}

/* Run `in` over all slices, for one nblk. */
static void run_all(const PlowDevInst* in, uint32_t nblk, void* const* T, PlowCpuCtx* ctx) {
    PlowDevInst i2 = *in;
    i2.blocks = (uint16_t)nblk;
    for (uint32_t s = 0; s < nblk; s++)
        CHECK(plow_cpu_exec(&i2, s, nblk, T, ctx) == 0, "op %u has no kernel", in->op);
}

static const uint32_t NBLKS[2] = {1, 3};

static void test_residual(PlowCpuCtx* ctx) {
    const uint32_t n = 1000;
    plow_bf16 *a = malloc(n * 2), *b = malloc(n * 2), *out = malloc(n * 2);
    fill_bf16(a, n, 2.0f);
    fill_bf16(b, n, 2.0f);
    void* T[3] = {out, a, b};
    PlowDevInst in = inst(PLOW_DOP_RESIDUAL);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    in.i[0] = n;
    in.fj[0].f = 0.75f;
    for (int k = 0; k < 2; k++) {
        memset(out, 0, n * 2);
        run_all(&in, NBLKS[k], T, ctx);
        for (uint32_t i = 0; i < n; i++) {
            const float want = (plow_bf2f(a[i]) + plow_bf2f(b[i])) * 0.75f;
            CHECK(close_bf(out[i], want), "residual nblk=%u i=%u got %f want %f", NBLKS[k], i,
                  plow_bf2f(out[i]), want);
        }
    }
    free(a); free(b); free(out);
}

static void test_rmsnorm(PlowCpuCtx* ctx) {
    const uint32_t rows = 5, feat = 256;
    const float eps = 1e-6f;
    plow_bf16 *x = malloc(rows * feat * 2), *g = malloc(feat * 2), *out = malloc(rows * feat * 2);
    fill_bf16(x, rows * feat, 3.0f);
    fill_bf16(g, feat, 1.0f);
    void* T[3] = {out, x, g};
    PlowDevInst in = inst(PLOW_DOP_RMSNORM);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    in.i[0] = rows; in.i[1] = feat;
    in.fj[0].f = eps;
    for (int k = 0; k < 2; k++) {
        memset(out, 0, rows * feat * 2);
        run_all(&in, NBLKS[k], T, ctx);
        for (uint32_t r = 0; r < rows; r++) {
            float ss = 0.0f;
            for (uint32_t i = 0; i < feat; i++) {
                const float v = plow_bf2f(x[r * feat + i]);
                ss += v * v;
            }
            const float inv = 1.0f / sqrtf(ss / feat + eps);
            for (uint32_t i = 0; i < feat; i++) {
                const float want = plow_bf2f(x[r * feat + i]) * inv * plow_bf2f(g[i]);
                CHECK(close_bf(out[r * feat + i], want), "rmsnorm nblk=%u r=%u i=%u got %f want %f",
                      NBLKS[k], r, i, plow_bf2f(out[r * feat + i]), want);
            }
        }
    }
    free(x); free(g); free(out);
}

static void test_gemv(PlowCpuCtx* ctx) {
    const uint32_t N = 70, K = 96;
    for (uint32_t M = 1; M <= 4; M++) {
        plow_bf16 *x = malloc(M * K * 2), *W = malloc(N * K * 2), *C = malloc(M * N * 2);
        fill_bf16(x, M * K, 1.0f);
        fill_bf16(W, N * K, 0.5f);
        void* T[3] = {C, x, W};
        PlowDevInst in = inst(PLOW_DOP_GEMV);
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
        in.i[0] = M; in.i[1] = N; in.i[2] = K;
        for (int k = 0; k < 2; k++) {
            memset(C, 0, M * N * 2);
            run_all(&in, NBLKS[k], T, ctx);
            for (uint32_t m = 0; m < M; m++)
                for (uint32_t n = 0; n < N; n++) {
                    float want = 0.0f;
                    for (uint32_t kk = 0; kk < K; kk++)
                        want += plow_bf2f(x[m * K + kk]) * plow_bf2f(W[n * K + kk]);
                    CHECK(close_bf(C[m * N + n], want), "gemv M=%u nblk=%u m=%u n=%u got %f want %f",
                          M, NBLKS[k], m, n, plow_bf2f(C[m * N + n]), want);
                }
        }
        free(x); free(W); free(C);
    }
}

static void test_gemm(PlowCpuCtx* ctx) {
    /* GEMM_SMALL's 64x128 tile over 96x256 -> 2x2 = 4 tiles, so nblk=3 exercises striding. */
    const uint32_t M = 96, N = 256, K = 64;
    plow_bf16 *A = malloc(M * K * 2), *B = malloc(N * K * 2), *C = malloc(M * N * 2);
    fill_bf16(A, M * K, 1.0f);
    fill_bf16(B, N * K, 0.5f);
    void* T[3] = {C, A, B};
    const uint16_t ops[2] = {PLOW_DOP_GEMM_SMALL, PLOW_DOP_GEMM};
    for (int o = 0; o < 2; o++) {
        PlowDevInst in = inst(ops[o]);
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
        in.i[0] = M; in.i[1] = N; in.i[2] = K;
        for (int k = 0; k < 2; k++) {
            memset(C, 0, M * N * 2);
            run_all(&in, NBLKS[k], T, ctx);
            for (uint32_t m = 0; m < M; m++)
                for (uint32_t n = 0; n < N; n++) {
                    float want = 0.0f;
                    for (uint32_t kk = 0; kk < K; kk++)
                        want += plow_bf2f(A[m * K + kk]) * plow_bf2f(B[n * K + kk]);
                    CHECK(close_bf(C[m * N + n], want), "gemm op=%u nblk=%u m=%u n=%u got %f want %f",
                          ops[o], NBLKS[k], m, n, plow_bf2f(C[m * N + n]), want);
                }
        }
    }
    free(A); free(B); free(C);
}

static float gelu_tanh(float x) {
    const float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

static void test_glu(PlowCpuCtx* ctx) {
    const uint32_t n = 777;
    plow_bf16 *g = malloc(n * 2), *u = malloc(n * 2), *out = malloc(n * 2);
    fill_bf16(g, n, 3.0f);
    fill_bf16(u, n, 2.0f);
    void* T[3] = {out, g, u};
    for (uint32_t act = 0; act < 2; act++) {
        PlowDevInst in = inst(PLOW_DOP_GLU);
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
        in.i[0] = n; in.i[1] = act;
        for (int k = 0; k < 2; k++) {
            memset(out, 0, n * 2);
            run_all(&in, NBLKS[k], T, ctx);
            for (uint32_t i = 0; i < n; i++) {
                const float gv = plow_bf2f(g[i]);
                const float a = act ? gv / (1.0f + expf(-gv)) : gelu_tanh(gv);
                const float want = a * plow_bf2f(u[i]);
                CHECK(close_bf(out[i], want), "glu act=%u nblk=%u i=%u got %f want %f", act,
                      NBLKS[k], i, plow_bf2f(out[i]), want);
            }
        }
    }
    free(g); free(u); free(out);
}

/* FLASH_DECODE (split-KV partials) then FLASH_MERGE, against naive softmax attention over the
 * head-major KV cache. Two batches, GQA 2, a sliding window on the second pass. */
static void test_flash_decode(PlowCpuCtx* ctx) {
    const uint32_t B = 2, H = 4, HKV = 2, D = 128, stride = 64, nsplit = 3;
    const uint32_t kv_mask = 0xFFFFFFFFu;
    const float scale = 1.0f / sqrtf((float)D);
    int32_t kv_len[2] = {40, 57};
    plow_bf16* Q = malloc(B * H * D * 2);
    plow_bf16* K = malloc(B * HKV * stride * D * 2);
    plow_bf16* V = malloc(B * HKV * stride * D * 2);
    float* Opart = malloc(B * H * nsplit * D * 4);
    float* ml = malloc(B * H * nsplit * 2 * 4);
    plow_bf16* O = malloc(B * H * D * 2);
    fill_bf16(Q, B * H * D, 1.0f);
    fill_bf16(K, B * HKV * stride * D, 1.0f);
    fill_bf16(V, B * HKV * stride * D, 1.0f);
    void* T[6] = {Opart, ml, Q, K, V, kv_len};
    void* Tm[3] = {O, Opart, ml};
    for (uint32_t window = 0; window <= 16; window += 16) {
        PlowDevInst fd = inst(PLOW_DOP_FLASH_DECODE);
        fd.t[0] = 0; fd.t[1] = 1; fd.t[2] = 2; fd.t[3] = 3; fd.t[4] = 4; fd.t[5] = 5;
        fd.i[0] = B; fd.i[1] = H; fd.i[2] = HKV; fd.i[3] = stride; fd.i[4] = window;
        fd.i[5] = nsplit; fd.i[6] = D; fd.i[7] = kv_mask;
        fd.fj[0].f = scale;
        PlowDevInst fm = inst(PLOW_DOP_FLASH_MERGE);
        fm.t[0] = 0; fm.t[1] = 1; fm.t[2] = 2;
        fm.i[0] = B; fm.i[1] = H; fm.i[2] = nsplit; fm.i[3] = D;
        for (int k = 0; k < 2; k++) {
            memset(Opart, 0, B * H * nsplit * D * 4);
            memset(ml, 0, B * H * nsplit * 8);
            memset(O, 0, B * H * D * 2);
            run_all(&fd, NBLKS[k], T, ctx);
            run_all(&fm, NBLKS[k], Tm, ctx);
            for (uint32_t b = 0; b < B; b++)
                for (uint32_t h = 0; h < H; h++) {
                    const uint32_t hkv = h / (H / HKV);
                    const uint32_t len = kv_len[b], qpos = len - 1;
                    const uint32_t first = (window && len > window) ? len - window : 0;
                    const plow_bf16* q = Q + (b * H + h) * D;
                    const plow_bf16* kb = K + ((b * HKV + hkv) * stride) * D;
                    const plow_bf16* vb = V + ((b * HKV + hkv) * stride) * D;
                    float s[64], mx = -1e30f, z = 0.0f;
                    for (uint32_t kv = first; kv <= qpos; kv++) {
                        float d = 0.0f;
                        for (uint32_t e = 0; e < D; e++) d += plow_bf2f(q[e]) * plow_bf2f(kb[kv * D + e]);
                        s[kv] = d * scale;
                        mx = mx > s[kv] ? mx : s[kv];
                    }
                    for (uint32_t kv = first; kv <= qpos; kv++) { s[kv] = expf(s[kv] - mx); z += s[kv]; }
                    for (uint32_t e = 0; e < D; e++) {
                        float want = 0.0f;
                        for (uint32_t kv = first; kv <= qpos; kv++) want += s[kv] / z * plow_bf2f(vb[kv * D + e]);
                        CHECK(close_bf(O[(b * H + h) * D + e], want),
                              "flash win=%u nblk=%u b=%u h=%u d=%u got %f want %f", window, NBLKS[k],
                              b, h, e, plow_bf2f(O[(b * H + h) * D + e]), want);
                    }
                }
        }
    }
    free(Q); free(K); free(V); free(Opart); free(ml); free(O);
}

int main(void) {
    const int isa = plow_cpu_init(PLOW_CPU_ISA_AMX);
    CHECK(isa >= 0, "plow_cpu_init failed: %d", isa);
    printf("isa tier: %d\n", isa);
    PlowCpuCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    CHECK(plow_cpu_thread_init(&ctx) == 0, "thread init");
    CHECK(ctx.isa == (uint32_t)isa, "ctx isa");

    const uint16_t p0[] = {
        PLOW_DOP_NOP, PLOW_DOP_RESIDUAL, PLOW_DOP_RMSNORM, PLOW_DOP_ROWRMS, PLOW_DOP_LAYERNORM,
        PLOW_DOP_HEADNORM_ROPE, PLOW_DOP_NORM_RESIDUAL, PLOW_DOP_ADD_NORM,
        PLOW_DOP_NORM_RESIDUAL_NORM, PLOW_DOP_GLU, PLOW_DOP_EMBED, PLOW_DOP_SOFTCAP,
        PLOW_DOP_GEMM, PLOW_DOP_GEMM_SMALL, PLOW_DOP_GEMM_MED, PLOW_DOP_GEMM_WIDE,
        PLOW_DOP_GEMM_C5, PLOW_DOP_GEMM_NORM, PLOW_DOP_GEMM_GLU, PLOW_DOP_GEMV,
        PLOW_DOP_GEMV_GLU, PLOW_DOP_GEMV_QKV, PLOW_DOP_GEMV_ARGMAX, PLOW_DOP_ARGMAX,
        PLOW_DOP_ARGMAX_FIN, PLOW_DOP_FLASH_PREFILL, PLOW_DOP_FLASH_DECODE,
        PLOW_DOP_FLASH_MERGE, PLOW_DOP_ATTN_RES};
    for (size_t i = 0; i < sizeof(p0) / sizeof(p0[0]); i++)
        CHECK(plow_cpu_has(p0[i]), "missing kernel for op %u", p0[i]);
    CHECK(!plow_cpu_has(PLOW_DOP_XREDUCE), "XREDUCE must not be registered");

    /* NOP dispatches and touches nothing. */
    PlowDevInst nop = inst(PLOW_DOP_NOP);
    CHECK(plow_cpu_exec(&nop, 0, 1, NULL, &ctx) == 0, "nop exec");

    test_residual(&ctx);
    test_rmsnorm(&ctx);
    test_gemv(&ctx);
    test_gemm(&ctx);
    test_glu(&ctx);
    test_flash_decode(&ctx);

    if (fails) {
        printf("%d failure(s)\n", fails);
        return 1;
    }
    printf("cpu_dev_golden: all passed\n");
    return 0;
}

/* CPU golden GEMM — correctness-first, no tiling/vectorization. */
#include "cpu_kernels.h"
#include <math.h>

static float act_apply(int act, float x) {
    switch (act) {
        case PLOW_ACT_SILU:       return x / (1.0f + expf(-x));
        case PLOW_ACT_GELU:       return 0.5f * x * (1.0f + erff(x * 0.70710678f));
        case PLOW_ACT_GELU_TANH: {
            float c = 0.79788456f * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + tanhf(c));
        }
        case PLOW_ACT_RELU:       return x > 0.0f ? x : 0.0f;
        case PLOW_ACT_SIGMOID:    return 1.0f / (1.0f + expf(-x));
        case PLOW_ACT_QUICK_GELU: return x / (1.0f + expf(-1.702f * x));
        default:                  return x; /* PLOW_ACT_NONE */
    }
}

/* c[m,n] = a[m,k] · b[n,k]^T (+ bias[n]), then act. b is row-major [n,k]. */
void plow_gemm_ref(float* c, const float* a, const float* b, const float* bias,
                   uint32_t m, uint32_t n, uint32_t k, int act) {
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            float acc = bias ? bias[j] : 0.0f;
            for (uint32_t p = 0; p < k; p++)
                acc += a[i * k + p] * b[j * k + p];
            c[i * n + j] = act_apply(act, acc);
        }
    }
}

void cpu_gemm(const void* body, kctx* ctx) {
    const PlowGemmBody* g = (const PlowGemmBody*)body;
    const PlowBinding* b = ctx->bind;
    if (!b) return;
    const float* A = (const float*)ctx->slots[b->in0];
    const float* B = (const float*)ctx->slots[b->in1];
    const float* bias = (b->in2 != PLOW_SLOT_NONE) ? (const float*)ctx->slots[b->in2] : NULL;
    float* C = (float*)ctx->slots[g->out];
    plow_gemm_ref(C, A, B, bias, g->m, g->n, g->k, b->detail);
}

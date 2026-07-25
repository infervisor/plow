/* CPU golden row ops — reduce (rmsnorm/layernorm/softmax) and pointwise. */
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
        default:                  return x;
    }
}

/* Per row over `feat`: rmsnorm / layernorm / softmax. gamma (scale) may be NULL. */
void plow_row_reduce_ref(float* out, const float* x, const float* gamma,
                         uint32_t rows, uint32_t feat, int norm, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * feat;
        float* or_ = out + (size_t)r * feat;
        if (norm == PLOW_NORM_RMS) {
            float ss = 0.0f;
            for (uint32_t i = 0; i < feat; i++) ss += xr[i] * xr[i];
            float inv = 1.0f / sqrtf(ss / (float)feat + eps);
            for (uint32_t i = 0; i < feat; i++)
                or_[i] = xr[i] * inv * (gamma ? gamma[i] : 1.0f);
        } else if (norm == PLOW_NORM_LAYER) {
            float mean = 0.0f;
            for (uint32_t i = 0; i < feat; i++) mean += xr[i];
            mean /= (float)feat;
            float var = 0.0f;
            for (uint32_t i = 0; i < feat; i++) { float d = xr[i] - mean; var += d * d; }
            var /= (float)feat;
            float inv = 1.0f / sqrtf(var + eps);
            for (uint32_t i = 0; i < feat; i++)
                or_[i] = (xr[i] - mean) * inv * (gamma ? gamma[i] : 1.0f);
        } else { /* PLOW_NORM_SOFTMAX */
            float mx = -INFINITY;
            for (uint32_t i = 0; i < feat; i++) if (xr[i] > mx) mx = xr[i];
            float sum = 0.0f;
            for (uint32_t i = 0; i < feat; i++) { or_[i] = expf(xr[i] - mx); sum += or_[i]; }
            for (uint32_t i = 0; i < feat; i++) or_[i] /= sum;
        }
    }
}

/* act(a) when b==NULL, else elementwise a (ew) b. */
void plow_row_pointwise_ref(float* out, const float* a, const float* b,
                            uint32_t n, int act, int ew) {
    for (uint32_t i = 0; i < n; i++) {
        float v;
        if (b) {
            switch (ew) {
                case PLOW_EW_ADD: v = a[i] + b[i]; break;
                case PLOW_EW_SUB: v = a[i] - b[i]; break;
                case PLOW_EW_MUL: v = a[i] * b[i]; break;
                case PLOW_EW_DIV: v = a[i] / b[i]; break;
                default:          v = a[i]; break;
            }
        } else {
            v = act_apply(act, a[i]);
        }
        out[i] = v;
    }
}

void cpu_row_reduce(const void* body, kctx* ctx) {
    const PlowRowBody* r = (const PlowRowBody*)body;
    const PlowBinding* b = ctx->bind;
    if (!b) return;
    const float* X = (const float*)ctx->slots[b->in0];
    const float* gamma = (b->in1 != PLOW_SLOT_NONE) ? (const float*)ctx->slots[b->in1] : NULL;
    float* O = (float*)ctx->slots[r->out];
    plow_row_reduce_ref(O, X, gamma, r->rows, r->feat, b->detail, b->scale);
}

void cpu_row_pointwise(const void* body, kctx* ctx) {
    const PlowRowBody* r = (const PlowRowBody*)body;
    const PlowBinding* b = ctx->bind;
    if (!b) return;
    const float* A = (const float*)ctx->slots[b->in0];
    const float* B = (r->operands > 1 && b->in1 != PLOW_SLOT_NONE)
                         ? (const float*)ctx->slots[b->in1] : NULL;
    float* O = (float*)ctx->slots[r->out];
    /* detail high nibble = ew kind, low nibble = act kind (host convention). */
    int act = b->detail & 0x0F;
    int ew = (b->detail >> 4) & 0x0F;
    plow_row_pointwise_ref(O, A, B, r->rows * r->feat, act, ew);
}

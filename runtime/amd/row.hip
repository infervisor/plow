/* ROCm/HIP golden row ops (BACKEND_ROCM, variant 0x00) — naive single-thread
 * reduce (rmsnorm/layernorm/softmax) + pointwise (act/elementwise) references
 * for MI300, must match the CPU oracle. Mirrors runtime/nvidia/row.cu. */
#include <hip/hip_runtime.h>
#include "kernel.h"
#include "packet.h"

__global__ void hip_row_reduce_naive_f32(float* out, const float* x, const float* gamma,
                                         unsigned rows, unsigned feat, int norm, float eps) {
    for (unsigned r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * feat;
        float* orow = out + (size_t)r * feat;
        if (norm == PLOW_NORM_RMS) {
            float ss = 0.0f;
            for (unsigned i = 0; i < feat; ++i) ss += xr[i] * xr[i];
            float inv = rsqrtf(ss / feat + eps);
            for (unsigned i = 0; i < feat; ++i) orow[i] = xr[i] * inv * (gamma ? gamma[i] : 1.0f);
        } else if (norm == PLOW_NORM_LAYER) {
            float mean = 0.0f; for (unsigned i = 0; i < feat; ++i) mean += xr[i]; mean /= feat;
            float var = 0.0f; for (unsigned i = 0; i < feat; ++i) { float d = xr[i] - mean; var += d * d; } var /= feat;
            float inv = rsqrtf(var + eps);
            for (unsigned i = 0; i < feat; ++i) orow[i] = (xr[i] - mean) * inv * (gamma ? gamma[i] : 1.0f);
        } else {
            float mx = -1e30f; for (unsigned i = 0; i < feat; ++i) mx = fmaxf(mx, xr[i]);
            float sum = 0.0f; for (unsigned i = 0; i < feat; ++i) { orow[i] = expf(xr[i] - mx); sum += orow[i]; }
            for (unsigned i = 0; i < feat; ++i) orow[i] /= sum;
        }
    }
}

extern "C" void hip_row_reduce_golden(const void* body, kctx* ctx) {
    const PlowRowBody* r = (const PlowRowBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const float* X = (const float*)ctx->slots[bd->in0];
    const float* gamma = (bd->in1 != PLOW_SLOT_NONE) ? (const float*)ctx->slots[bd->in1] : nullptr;
    float* O = (float*)ctx->slots[r->out];
    hipLaunchKernelGGL(hip_row_reduce_naive_f32, dim3(1), dim3(1), 0, (hipStream_t)ctx->stream,
                       O, X, gamma, r->rows, r->feat, bd->detail, bd->scale);
}

__device__ static float hip_act_apply(int act, float x) {
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

__global__ void hip_row_pointwise_naive_f32(float* out, const float* a, const float* b,
                                            unsigned n, int act, int ew) {
    for (unsigned i = 0; i < n; ++i) {
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
            v = hip_act_apply(act, a[i]);
        }
        out[i] = v;
    }
}

extern "C" void hip_row_pointwise_golden(const void* body, kctx* ctx) {
    const PlowRowBody* r = (const PlowRowBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const float* A = (const float*)ctx->slots[bd->in0];
    const float* B = (r->operands > 1 && bd->in1 != PLOW_SLOT_NONE)
                         ? (const float*)ctx->slots[bd->in1] : nullptr;
    float* O = (float*)ctx->slots[r->out];
    int act = bd->detail & 0x0F;
    int ew = (bd->detail >> 4) & 0x0F;
    hipLaunchKernelGGL(hip_row_pointwise_naive_f32, dim3(1), dim3(1), 0, (hipStream_t)ctx->stream,
                       O, A, B, r->rows * r->feat, act, ew);
}

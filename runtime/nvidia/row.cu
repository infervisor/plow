/* CUDA row ops. variant 0x00 naive golden reduce + pointwise. */
#include "gpu_common.h"

__global__ void plow_row_reduce_naive_f32(float* out, const float* x, const float* gamma,
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

extern "C" void cuda_row_reduce_golden(const void* body, kctx* ctx) {
    const PlowRowBody* r = (const PlowRowBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const float* X = (const float*)ctx->slots[bd->in0];
    const float* gamma = (bd->in1 != PLOW_SLOT_NONE) ? (const float*)ctx->slots[bd->in1] : nullptr;
    float* O = (float*)ctx->slots[r->out];
    plow_row_reduce_naive_f32<<<1, 1, 0, (GPU_STREAM)ctx->stream>>>(
        O, X, gamma, r->rows, r->feat, bd->detail, bd->scale);
}

__device__ static float plow_act_apply(int act, float x) {
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

/* act(a) when b==NULL, else elementwise a (ew) b. Symmetric to cpu_row_pointwise. */
__global__ void plow_row_pointwise_naive_f32(float* out, const float* a, const float* b,
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
            v = plow_act_apply(act, a[i]);
        }
        out[i] = v;
    }
}

extern "C" void cuda_row_pointwise_golden(const void* body, kctx* ctx) {
    const PlowRowBody* r = (const PlowRowBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const float* A = (const float*)ctx->slots[bd->in0];
    const float* B = (r->operands > 1 && bd->in1 != PLOW_SLOT_NONE)
                         ? (const float*)ctx->slots[bd->in1] : nullptr;
    float* O = (float*)ctx->slots[r->out];
    /* detail high nibble = ew kind, low nibble = act kind (host convention). */
    int act = bd->detail & 0x0F;
    int ew = (bd->detail >> 4) & 0x0F;
    plow_row_pointwise_naive_f32<<<1, 1, 0, (GPU_STREAM)ctx->stream>>>(
        O, A, B, r->rows * r->feat, act, ew);
}

// ==========================================================================
// Gemma 4 fused Row variants — single-SM, self-distributed, memory-bound.
//
// Contract: ONE packet = whole op (PlowRowBody{rows,feat,br,coord,out}). The
// kernel launches P=#SMs blocks; block s owns row-groups s, s+P, ... (grid-
// stride, group size = br). One warp handles one (row, head); no MMA. gamma /
// rope-freqs loaded once, resident across the block's owned rows. All are
// bandwidth-limited — the math is negligible next to the HBM traffic.
// ==========================================================================

// F3 FusedNormRope (K path): RMSNorm over head_dim then RoPE on the first
// feat/2 dims. F2 adds a final *1/sqrt(head_dim) scale (Q path). Bindings:
// in0=X, in1=gamma(qk_norm), in2=rope-freq table or NONE(compute on the fly),
// scale=eps (or rope theta when in2==NONE), coord=position base. rotary_dims =
// feat/2 (Gemma partial_rotary_factor=0.5) — derived, not stored.
extern "C" void cuda_row_normrope_bf16(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // per (row m, head h): ss=Sum(x^2) over feat (warp shuffle) -> rms;
    //   norm=x*rms*gamma; rotate dims [0,feat/2) with cos/sin(theta,pos);
    //   write out. No scale (K path).
    // TODO: bf16 vectorized load/store; on-the-fly RoPE for decode, table (in2)
    //       for prefill. Inner loop inspired by ThunderKittens row primitives.
}
extern "C" void cuda_row_normropescale_bf16(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // Same as normrope, then multiply all dims by 1/sqrt(feat) (Q path scale).
    // TODO.
}

// F5 SwiGLU: out = silu(gate) * up. Bindings: in0=gate, in1=up, operands=2.
extern "C" void cuda_row_swiglu_bf16(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // vectorized float4: g=gate[i]; out[i]=(g/(1+exp(-g)))*up[i]. Pure bandwidth.
    // TODO.
}

// S3 Residual add: out = x + residual. Bindings: in0=x, in1=residual, operands=2.
extern "C" void cuda_row_residual_bf16(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // vectorized float4 add. TODO.
}

// S4 Standalone RMSNorm (final norm, K14): reduce-shaped. Bindings: in0=X,
// in1=gamma, scale=eps.
extern "C" void cuda_row_rmsnorm_bf16(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // per row: rms over feat -> x*rms*gamma. Same reduction as normrope Phase 0,
    // no rope. TODO.
}

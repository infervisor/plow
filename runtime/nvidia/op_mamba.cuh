/* op_mamba.cuh — Nemotron-3 Mamba-2 SSD mixer core for sm_120 (DevOp::Mamba2Scan / PLOW_DOP_MAMBA2_SCAN).
 *
 * ============================== UNVERIFIED ON GPU ==============================
 * This is a CORRECTNESS-FIRST, NOT perf-tuned, device implementation. There is no Nemotron
 * checkpoint and no GPU parity harness on this box, so it has ONLY been verified to nvcc-COMPILE
 * (scripts/build_sm120_cubin.sh). Its MATH mirrors the f32 golden `mamba_ref` in
 * crates/plowc/src/bin/gemma4.rs (module nemotron_tests), which IS numerically checked against an
 * independent closed-form recurrence (max-abs ~1e-8). But this kernel has never executed on a GPU.
 * DO NOT treat it as validated. Perf work needed (see PERF note at the bottom) before it is useful
 * beyond exercising the interpreter dispatch.
 * ===============================================================================
 *
 * Single-CU (the emit puts this op on ONE workgroup: blocks==1 => slice 0). The block's threads
 * cooperate; the scan itself is embarrassingly parallel across (head, channel) pairs and sequential
 * over T within each pair, so NO cross-thread reduction is needed for the scan. conv1d is computed
 * ON DEMAND (no materialized conv_out buffer), so this needs no scratch arena.
 *
 * Math (mirror of the Rust golden):
 *   conv_dim = d_inner + 2*n_groups*d_state; xBC splits into x[d_inner] | B[n_groups*d_state] | C[..]
 *   conv1d: causal depthwise, kernel d_conv, + conv_b, then SiLU. Past inputs come from conv_state.
 *   per head h (group g = h/(n_head/n_groups)):  A[h] = -exp(A_log[h])
 *     dt[t,h] = softplus(dt_raw[t,h] + dt_bias[h])   (NO time_step_limit clamp — assumption)
 *     dA      = exp(dt[t,h] * A[h])
 *     h_state[p,n] = dA*h_state[p,n] + dt*x[t,h,p]*B[t,g,n];  y[t,h,p] = Σ_n C[t,g,n]*h_state[p,n] + D[h]*x
 *   gated RMSNorm over d_inner per token:  yg = y*silu(z); out = yg * rsqrt(mean(yg^2)+eps) * norm_w
 */
#pragma once
#include "sm120_common.cuh"

/* Per-thread SSM state cap. Real Nemotron d_state = 128; 256 covers headroom. A larger d_state would
 * need a different (global-state) layout — asserted-away, not silently wrong. */
#ifndef MAMBA_MAX_DSTATE
#define MAMBA_MAX_DSTATE 256
#endif

static __device__ __forceinline__ float mamba_silu(float x) {
    return x / (1.0f + __expf(-x));
}
static __device__ __forceinline__ float mamba_softplus(float x) {
    return x > 20.0f ? x : __logf(1.0f + __expf(x));
}

/* Causal depthwise conv1d + SiLU output for (token t, channel c), computed on demand. Reads the
 * ORIGINAL conv_state for pos<0 (early tokens); conv_state is only updated AFTER all scan reads, so
 * this is race-free within the single block. */
static __device__ __forceinline__ float
mamba_conv_at(unsigned t, unsigned c, const __nv_bfloat16* __restrict__ xbc,
              const __nv_bfloat16* __restrict__ conv_w, const float* __restrict__ conv_b,
              const float* __restrict__ conv_state, unsigned d_conv, unsigned conv_dim) {
    float acc = conv_b ? conv_b[c] : 0.0f;
    const int pad = (int)d_conv - 1;
    for (unsigned k = 0; k < d_conv; k++) {
        int pos = (int)t - pad + (int)k;
        float inp;
        if (pos < 0) {
            inp = conv_state[(size_t)(pos + pad) * conv_dim + c];
        } else {
            inp = __bfloat162float(xbc[(size_t)pos * conv_dim + c]);
        }
        acc += __bfloat162float(conv_w[(size_t)c * d_conv + k]) * inp;
    }
    return mamba_silu(acc);
}

static __device__ void
d_mamba2_scan(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ xbc,
              const __nv_bfloat16* __restrict__ dt_raw, const __nv_bfloat16* __restrict__ z,
              const __nv_bfloat16* __restrict__ conv_w, const float* __restrict__ params,
              float* __restrict__ conv_state, float* __restrict__ ssm_state, unsigned T,
              unsigned d_inner, unsigned n_head, unsigned head_dim, unsigned d_state,
              unsigned n_groups, unsigned d_conv, unsigned conv_dim, float eps, unsigned slice) {
    /* Single-CU correctness-first: only slice 0 does the work. */
    if (slice != 0) return;
    if (d_state > MAMBA_MAX_DSTATE) return; /* compile-time cap; larger needs a global-state layout */

    const float* A_log = params;
    const float* Dp = params + n_head;
    const float* dt_bias = params + 2u * n_head;
    const float* conv_b = params + 3u * n_head;
    const float* norm_w = params + 3u * n_head + conv_dim;

    const unsigned tid = threadIdx.x;
    const unsigned nthr = blockDim.x;
    const unsigned hpg = n_head / n_groups;
    const unsigned gsz = n_groups * d_state;

    /* ---- scan: thread owns (head,p) = channel `idx` in [0,d_inner), grid-strided ---- */
    for (unsigned idx = tid; idx < d_inner; idx += nthr) {
        const unsigned h = idx / head_dim;
        const unsigned p = idx % head_dim;
        const unsigned g = h / hpg;
        const float A = -__expf(A_log[h]);
        const float Dh = Dp[h];
        const float dtb = dt_bias[h];

        float hloc[MAMBA_MAX_DSTATE];
        for (unsigned n = 0; n < d_state; n++)
            hloc[n] = ssm_state[((size_t)h * head_dim + p) * d_state + n];

        for (unsigned t = 0; t < T; t++) {
            const float dtv = mamba_softplus(__bfloat162float(dt_raw[(size_t)t * n_head + h]) + dtb);
            const float dA = __expf(dtv * A);
            const float xv = mamba_conv_at(t, idx, xbc, conv_w, conv_b, conv_state, d_conv, conv_dim);
            float acc = 0.0f;
            for (unsigned n = 0; n < d_state; n++) {
                const float bn =
                    mamba_conv_at(t, d_inner + g * d_state + n, xbc, conv_w, conv_b, conv_state,
                                  d_conv, conv_dim);
                const float cn =
                    mamba_conv_at(t, d_inner + gsz + g * d_state + n, xbc, conv_w, conv_b,
                                  conv_state, d_conv, conv_dim);
                hloc[n] = dA * hloc[n] + dtv * xv * bn;
                acc += cn * hloc[n];
            }
            /* store PRE-norm y into out; the norm phase below reads it back per token. */
            out[(size_t)t * d_inner + idx] = __float2bfloat16(acc + Dh * xv);
        }
        for (unsigned n = 0; n < d_state; n++)
            ssm_state[((size_t)h * head_dim + p) * d_state + n] = hloc[n];
    }

    __syncthreads(); /* all conv_state reads (via mamba_conv_at) complete before the update below */

    /* ---- gated RMSNorm: thread owns a token t, grid-strided ---- */
    for (unsigned t = tid; t < T; t += nthr) {
        double ss = 0.0;
        for (unsigned c = 0; c < d_inner; c++) {
            const float yv = __bfloat162float(out[(size_t)t * d_inner + c]);
            const float zv = __bfloat162float(z[(size_t)t * d_inner + c]);
            const float g = yv * mamba_silu(zv);
            ss += (double)g * (double)g;
        }
        const float rms = rsqrtf((float)(ss / (double)d_inner) + eps);
        for (unsigned c = 0; c < d_inner; c++) {
            const float yv = __bfloat162float(out[(size_t)t * d_inner + c]);
            const float zv = __bfloat162float(z[(size_t)t * d_inner + c]);
            const float g = yv * mamba_silu(zv);
            out[(size_t)t * d_inner + c] = __float2bfloat16(g * rms * norm_w[c]);
        }
    }

    /* ---- conv_state update: thread owns a channel c, ascending j (race-free shift) ----
     * New conv_state = last (d_conv-1) inputs of [old_conv_state ; xbc]. Ascending j reads the source
     * (index T+j > j when T>=1) before it is overwritten, so the in-place shift is safe per thread. */
    const int pad = (int)d_conv - 1;
    for (unsigned c = tid; c < conv_dim && pad > 0; c += nthr) {
        for (int j = 0; j < pad; j++) {
            int src = (int)T - pad + j; /* global input position feeding new row j */
            float val;
            if (src >= 0) {
                val = __bfloat162float(xbc[(size_t)src * conv_dim + c]);
            } else {
                val = conv_state[(size_t)(src + pad) * conv_dim + c];
            }
            conv_state[(size_t)j * conv_dim + c] = val;
        }
    }
}

/* PERF (deferred — this op is correctness-first & unverified):
 *  - conv1d for B/C channels is recomputed by every head_dim thread of a head (redundant); stage
 *    conv_out into smem/global once per (t,channel).
 *  - single-CU: the scan should be chunked (SSD block-decomposition) across CUs for prefill, and the
 *    per-head work spread over the grid for decode. All of that is future work.
 *  - `hloc` is per-thread local memory (d_state floats) => spills; a register/smem tiling is needed.
 */

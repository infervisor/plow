/* Muse Spark block ops 218-220 (runtime/common/dev_isa.h). Compiled with the PLOW_QWEN_GDN arms,
 * whose QwenHeadNormRope / QwenSigmoidGate the Muse attention also uses, so the existing
 * `plow_qwen_gdn_arms_1` load-time check covers them.
 *
 * Every bf16 round below is one the reference (`modeling_muse_spark.py`) makes; the transcendentals
 * are the precise ones for the same reason op_qwen_gdn.h gives. */
#ifndef PLOW_OP_MUSE_H
#define PLOW_OP_MUSE_H

#include "amd_common.h"

#define PLOW_MUSE_NORM_PLAIN 0u
#define PLOW_MUSE_NORM_CENTERED 1u
#define PLOW_MUSE_NORM_INPUT_SCALED 2u

__device__ __forceinline__ float muse_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* INPUT_SCALED rounds `x*(1+w)` to bf16 before the norm (`output.type_as(x)` in the reference). */
__device__ __forceinline__ float muse_norm_in(float x, float w, unsigned mode) {
    return mode == PLOW_MUSE_NORM_INPUT_SCALED ? bf2f(f2bf(x * (1.0f + w))) : x;
}

__device__ __forceinline__ float muse_norm_gain(float w, bool has_w, unsigned mode) {
    if (!has_w || mode == PLOW_MUSE_NORM_INPUT_SCALED) return 1.0f;
    return mode == PLOW_MUSE_NORM_CENTERED ? 1.0f + w : w;
}

/* ---- op 218: row RMSNorm, one workgroup per row ----
 * 16-byte loads: a decode row is one workgroup, so the scalar walk was dim/PLOW_THREADS dependent
 * HBM round trips per pass (19 us at dim 8192; this form is two loads per thread per pass). */
__device__ void d_muse_rmsnorm(bf16* out, const bf16* x, const bf16* w, unsigned rows,
                               unsigned dim, unsigned mode, float eps, float oscale,
                               unsigned slice, unsigned nblk, float* part) {
    /* `oscale` (0 = 1) folds the final norm's downstream logit multiplier into its output. */
    if (oscale == 0.0f) oscale = 1.0f;
    if (mode > PLOW_MUSE_NORM_INPUT_SCALED || (mode != PLOW_MUSE_NORM_PLAIN && !w) ||
        (dim & 7u) != 0u) {
        if (threadIdx.x == 0) __builtin_trap();
        return;
    }
    const bool has_w = w != nullptr;
    for (unsigned row = slice; row < rows; row += nblk) {
        const bf16* xr = x + (size_t)row * dim;
        float ss = 0.0f;
#pragma unroll 4
        for (unsigned i = threadIdx.x * 8u; i < dim; i += PLOW_THREADS * 8u) {
            const bf16v8 v = ld_glob8(xr + i);
            const bf16v8 g = has_w ? ld_glob8(w + i) : bf16v8_zero();
#pragma unroll
            for (unsigned j = 0; j < 8; j++) {
                const float f = muse_norm_in(bf2f(v[j]), bf2f(g[j]), mode);
                ss += f * f;
            }
        }
        const float r = rsqrtf(block_sum(ss, part) / (float)dim + eps);
        bf16* o = out + (size_t)row * dim;
#pragma unroll 4
        for (unsigned i = threadIdx.x * 8u; i < dim; i += PLOW_THREADS * 8u) {
            const bf16v8 v = ld_glob8(xr + i);
            const bf16v8 g = has_w ? ld_glob8(w + i) : bf16v8_zero();
            bf16v8 y;
#pragma unroll
            for (unsigned j = 0; j < 8; j++) {
                const float gw = bf2f(g[j]);
                y[j] = f2bf(muse_norm_in(bf2f(v[j]), gw, mode) * r * muse_norm_gain(gw, has_w, mode) *
                             oscale);
            }
            st_glob8(o + i, y);
        }
    }
}

/* ---- op 219: gated residual ---- */
__device__ void d_muse_residual_gate(bf16* out, const bf16* res, const bf16* h, const bf16* g,
                                     unsigned n, unsigned rows, float temperature, float eps,
                                     unsigned slice, unsigned nblk) {
    for (unsigned i = slice * PLOW_THREADS + threadIdx.x; i < rows * n; i += nblk * PLOW_THREADS) {
        const float wg = bf2f(g[i % n]) / temperature;
        const float beta = muse_sigmoid(wg);
        const float alpha = sqrtf(fmaxf(muse_sigmoid(-wg) * (1.0f + beta), eps));
        out[i] = f2bf(alpha * bf2f(res[i]) + beta * bf2f(h[i]));
    }
}

/* ---- op 220: post-expert input-scaled norm, then the weighted combine (one token) ----
 * `part[j]` is `gate_j * y_j` (f32) from the grouped down projection; the reference norms `y_j`
 * (its bf16 expert output) before weighting. `ybf` (i2 = 1, TP) reads `part` as the bf16
 * unweighted `y[k][H]` the cross-rank reduce of the down partials produced. Every workgroup reduces all k rows (k*H f32 from L2) so
 * no cross-workgroup handoff is needed, then owns its slice of H. The k sums share one pass and
 * one barrier pair (k serial block reductions were 45 us at k=8, H=4096). */
#define PLOW_MUSE_MAX_TOPK 16u
__device__ __forceinline__ float muse_post_expert_z(float p, float gate, float w) {
    return bf2f(f2bf(bf2f(f2bf(p / gate)) * (1.0f + w)));
}

__device__ __forceinline__ float muse_expert_y(const float* part, size_t i, unsigned ybf) {
    return ybf ? bf2f(((const bf16*)part)[i]) : part[i];
}

__device__ void d_muse_moe_combine_norm(bf16* out, const float* part, const unsigned char* table,
                                        const bf16* w, unsigned H, unsigned k, float eps,
                                        unsigned slice, unsigned nblk, float* lds,
                                        unsigned ybf = 0) {
    if (k > PLOW_MUSE_MAX_TOPK || (H & 7u) != 0u) {
        if (threadIdx.x == 0) __builtin_trap();
        return;
    }
    float* wpart = lds;                                 /* [PLOW_WAVES][k] */
    float* rinv = lds + PLOW_WAVES * PLOW_MUSE_MAX_TOPK; /* [k] */
    float ss[PLOW_MUSE_MAX_TOPK];
#pragma unroll
    for (unsigned j = 0; j < PLOW_MUSE_MAX_TOPK; j++) ss[j] = 0.0f;
    for (unsigned i = threadIdx.x * 8u; i < H; i += PLOW_THREADS * 8u) {
        const bf16v8 wv = ld_glob8(w + i);
#pragma unroll
        for (unsigned j = 0; j < PLOW_MUSE_MAX_TOPK; j++) {
            if (j >= k) break;
            const float gate = moe_slot_gate(table, j);
            if (moe_slot_expert(table, j) == PLOW_EXPERT_UNUSED || !(gate > 0.0f)) continue;
            float pv[8];
            if (ybf) {
                const bf16v8 yv = ld_glob8((const bf16*)part + (size_t)j * H + i);
#pragma unroll
                for (unsigned e = 0; e < 8; e++) pv[e] = bf2f(yv[e]);
            } else {
                const float4* pj = (const float4*)(part + (size_t)j * H + i);
                const float4 a = pj[0], b = pj[1];
                pv[0] = a.x, pv[1] = a.y, pv[2] = a.z, pv[3] = a.w;
                pv[4] = b.x, pv[5] = b.y, pv[6] = b.z, pv[7] = b.w;
            }
#pragma unroll
            for (unsigned e = 0; e < 8; e++) {
                const float z = ybf ? bf2f(f2bf(pv[e] * (1.0f + bf2f(wv[e]))))
                                    : muse_post_expert_z(pv[e], gate, bf2f(wv[e]));
                ss[j] += z * z;
            }
        }
    }
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
#pragma unroll
    for (unsigned j = 0; j < PLOW_MUSE_MAX_TOPK; j++) {
        if (j >= k) break;
        const float v = wave_sum(ss[j]);
        if (lane == 0) wpart[wave * PLOW_MUSE_MAX_TOPK + j] = v;
    }
    __syncthreads();
    if (threadIdx.x < k) {
        const unsigned j = threadIdx.x;
        float t = 0.0f;
#pragma unroll
        for (unsigned v = 0; v < PLOW_WAVES; v++) t += wpart[v * PLOW_MUSE_MAX_TOPK + j];
        const bool live =
            moe_slot_expert(table, j) != PLOW_EXPERT_UNUSED && moe_slot_gate(table, j) > 0.0f;
        rinv[j] = live ? rsqrtf(t / (float)H + eps) : 0.0f;
    }
    __syncthreads();
    for (unsigned d = slice * PLOW_THREADS + threadIdx.x; d < H; d += nblk * PLOW_THREADS) {
        float acc = 0.0f;
        for (unsigned j = 0; j < k; j++) {
            if (rinv[j] == 0.0f) continue;
            const float gate = moe_slot_gate(table, j);
            const float y = muse_expert_y(part, (size_t)j * H + d, ybf);
            const float z = ybf ? bf2f(f2bf(y * (1.0f + bf2f(w[d]))))
                                : muse_post_expert_z(y, gate, bf2f(w[d]));
            acc += gate * bf2f(f2bf(z * rinv[j]));
        }
        out[d] = f2bf(acc);
    }
    __syncthreads();
}

#endif

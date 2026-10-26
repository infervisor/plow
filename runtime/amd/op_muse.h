/* Muse Spark block ops 226-228 (runtime/common/dev_isa.h). Compiled with the PLOW_QWEN_GDN arms,
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
/* Opt-in (bit-identical): the one-row TP combine-norm reads its routing table once and issues
 * its k loads together. */
#ifndef PLOW_MUSE_CMB_ILP
#define PLOW_MUSE_CMB_ILP 0
#endif

__device__ __forceinline__ float muse_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* INPUT_SCALED rounds `x*(1+w)` to bf16 before the norm (`output.type_as(x)` in the reference). */
__device__ __forceinline__ float muse_norm_in(float x, float w, unsigned mode) {
    return mode == PLOW_MUSE_NORM_INPUT_SCALED ? bf2f(f2bf(x * (1.0f + w))) : x;
}

__device__ __forceinline__ float muse_norm_gain(float w, bool has_w, unsigned mode) {
    if (!has_w || mode == PLOW_MUSE_NORM_INPUT_SCALED) return 1.0f;
    return mode == PLOW_MUSE_NORM_CENTERED ? 1.0f + w : w;
}

/* Band-major copy `bm[c / bw][row][c % bw]` of a `[rows][dim]` row (`bw` = 0: none): the TP
 * row-parallel latent projections read this rank's column band as a dense `[rows][bw]` matrix. */
__device__ __forceinline__ void muse_st_band(bf16* bm, unsigned bw, unsigned rows, unsigned row,
                                             unsigned c, bf16v8 v) {
    if (bw) st_glob8(bm + (size_t)(c / bw) * rows * bw + (size_t)row * bw + c % bw, v);
}

/* ---- op 226: row RMSNorm, one workgroup per row ----
 * 16-byte loads: a decode row is one workgroup, so the scalar walk was dim/PLOW_THREADS dependent
 * HBM round trips per pass (19 us at dim 8192; this form is two loads per thread per pass). */
__device__ void d_muse_rmsnorm(bf16* out, const bf16* x, const bf16* w, unsigned rows,
                               unsigned dim, unsigned mode, float eps, float oscale,
                               unsigned slice, unsigned nblk, float* part, bf16* bm = nullptr,
                               unsigned bw = 0) {
    /* `oscale` (0 = 1) folds the final norm's downstream logit multiplier into its output. */
    if (oscale == 0.0f) oscale = 1.0f;
    if (mode > PLOW_MUSE_NORM_INPUT_SCALED || (mode != PLOW_MUSE_NORM_PLAIN && !w) ||
        (dim & 7u) != 0u || (bw & 7u) != 0u || (bw && dim % bw != 0u)) {
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
            muse_st_band(bm, bw, rows, row, i, y);
        }
    }
}

/* ---- op 227: gated residual ---- */
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

/* Row chunks op 229 keeps in registers between its two norms (dim <= chunks * 8 * threads). */
#define PLOW_MUSE_NGN_CHUNKS 2u

/* ---- op 229 (decode objects): row norm -> gated residual -> row norm, one workgroup per row ----
 * Ops 226 -> 227 -> 226 in one packet: every intermediate is rounded to bf16 where those ops store
 * it and every sum keeps their per-thread order, so it is bit-identical to the three-op chain.
 * `out_res` = the gated residual (may alias `res`); `out_n` = its norm (+ band copy). */
__device__ void d_muse_norm_gate_norm(bf16* out_res, bf16* out_n, const bf16* h, const bf16* w1,
                                      const bf16* res, const bf16* g, const bf16* w2,
                                      unsigned rows, unsigned dim, unsigned mode1, unsigned mode2,
                                      float eps1, float eps2, float temperature, float geps,
                                      float oscale, unsigned slice, unsigned nblk, float* part,
                                      bf16* bm = nullptr, unsigned bw = 0) {
    if (oscale == 0.0f) oscale = 1.0f;
    if (mode1 > PLOW_MUSE_NORM_INPUT_SCALED || mode2 > PLOW_MUSE_NORM_INPUT_SCALED ||
        (mode1 != PLOW_MUSE_NORM_PLAIN && !w1) || (mode2 != PLOW_MUSE_NORM_PLAIN && !w2) ||
        (dim & 7u) != 0u || dim > PLOW_MUSE_NGN_CHUNKS * PLOW_THREADS * 8u || (bw & 7u) != 0u ||
        (bw && dim % bw != 0u)) {
        if (threadIdx.x == 0) __builtin_trap();
        return;
    }
    const bool hw1 = w1 != nullptr, hw2 = w2 != nullptr;
    for (unsigned row = slice; row < rows; row += nblk) {
        const bf16* hr = h + (size_t)row * dim;
        const bf16* rr = res + (size_t)row * dim;
        bf16* orr = out_res + (size_t)row * dim;
        bf16* on = out_n + (size_t)row * dim;
        float ss = 0.0f;
#pragma unroll 4
        for (unsigned i = threadIdx.x * 8u; i < dim; i += PLOW_THREADS * 8u) {
            const bf16v8 v = ld_glob8(hr + i);
            const bf16v8 a = hw1 ? ld_glob8(w1 + i) : bf16v8_zero();
#pragma unroll
            for (unsigned j = 0; j < 8; j++) {
                const float f = muse_norm_in(bf2f(v[j]), bf2f(a[j]), mode1);
                ss += f * f;
            }
        }
        const float r1 = rsqrtf(block_sum(ss, part) / (float)dim + eps1);
        float ss2 = 0.0f;
        bf16v8 keep[PLOW_MUSE_NGN_CHUNKS];
#pragma unroll
        for (unsigned c = 0; c < PLOW_MUSE_NGN_CHUNKS; c++) {
            const unsigned i = (c * PLOW_THREADS + threadIdx.x) * 8u;
            if (i >= dim) break;
            const bf16v8 v = ld_glob8(hr + i);
            const bf16v8 a = hw1 ? ld_glob8(w1 + i) : bf16v8_zero();
            const bf16v8 rv = ld_glob8(rr + i);
            const bf16v8 gv = ld_glob8(g + i);
            const bf16v8 b = hw2 ? ld_glob8(w2 + i) : bf16v8_zero();
            bf16v8 o;
#pragma unroll
            for (unsigned j = 0; j < 8; j++) {
                const float aw = bf2f(a[j]);
                const bf16 y1 =
                    f2bf(muse_norm_in(bf2f(v[j]), aw, mode1) * r1 * muse_norm_gain(aw, hw1, mode1));
                const float wg = bf2f(gv[j]) / temperature;
                const float beta = muse_sigmoid(wg);
                const float alpha = sqrtf(fmaxf(muse_sigmoid(-wg) * (1.0f + beta), geps));
                o[j] = f2bf(alpha * bf2f(rv[j]) + beta * bf2f(y1));
                const float f = muse_norm_in(bf2f(o[j]), bf2f(b[j]), mode2);
                ss2 += f * f;
            }
            st_glob8(orr + i, o);
            keep[c] = o;
        }
        const float r2 = rsqrtf(block_sum(ss2, part) / (float)dim + eps2);
#pragma unroll
        for (unsigned c = 0; c < PLOW_MUSE_NGN_CHUNKS; c++) {
            const unsigned i = (c * PLOW_THREADS + threadIdx.x) * 8u;
            if (i >= dim) break;
            const bf16v8 o = keep[c];
            const bf16v8 b = hw2 ? ld_glob8(w2 + i) : bf16v8_zero();
            bf16v8 y;
#pragma unroll
            for (unsigned j = 0; j < 8; j++) {
                const float bw2 = bf2f(b[j]);
                y[j] = f2bf(muse_norm_in(bf2f(o[j]), bw2, mode2) * r2 *
                            muse_norm_gain(bw2, hw2, mode2) * oscale);
            }
            st_glob8(on + i, y);
            muse_st_band(bm, bw, rows, row, i, y);
        }
    }
}

/* ---- op 228: post-expert input-scaled norm, then the weighted combine ----
 * `part[j]` is `gate_j * y_j` (f32) from the grouped down projection; the reference norms `y_j`
 * (its bf16 expert output) before weighting. `ybf` (i2, TP) reads `part` as bf16 `[k][H]` the
 * cross-rank reduce of the down partials produced: 1 = unweighted `y` (decode), 2 = `gate * y`
 * (the grouped prefill down's bf16 scatter). One row: every workgroup reduces all k rows (k*H
 * from L2) so no cross-workgroup handoff is needed, then owns its slice of H. The k sums share
 * one pass and one barrier pair (k serial block reductions were 45 us at k=8, H=4096). Several
 * rows (i3): one workgroup per row, band-major copy of the output in `bm` (i4 = band width). */
#define PLOW_MUSE_MAX_TOPK 16u
__device__ __forceinline__ float muse_post_expert_z(float p, float gate, float w) {
    return bf2f(f2bf(bf2f(f2bf(p / gate)) * (1.0f + w)));
}

__device__ __forceinline__ float muse_expert_y(const float* part, size_t i, unsigned ybf) {
    return ybf ? bf2f(((const bf16*)part)[i]) : part[i];
}

/* The bf16-rounded, gain-scaled post-expert value whose RMS the norm takes. */
__device__ __forceinline__ float muse_expert_z(float p, float gate, float w, unsigned ybf) {
    return ybf == 1u ? bf2f(f2bf(p * (1.0f + w))) : muse_post_expert_z(p, gate, w);
}

#if PLOW_MUSE_CMB_ILP
/* muse_moe_combine_norm_row for bf16 `part` (TP) and k <= 8, with the same arithmetic in the same
 * order: the routing table is read once up front and every pass issues its k loads before the
 * first use. The generic row re-reads the table per slot and gates each load on it, a chain of
 * dependent round trips (~21 us at k = 8, H = 4096). */
#define PLOW_MUSE_CMB_K 8u
__device__ void muse_moe_combine_norm_row_k8(bf16* out, const bf16* part,
                                             const unsigned char* table, const bf16* w, unsigned H,
                                             unsigned k, float eps, unsigned slice, unsigned nblk,
                                             float* lds, unsigned ybf, bf16* bm, unsigned bw,
                                             unsigned rows, unsigned row) {
    float* wpart = lds;                                 /* [PLOW_WAVES][k] */
    float* rinv = lds + PLOW_WAVES * PLOW_MUSE_MAX_TOPK; /* [k] */
    float gate[PLOW_MUSE_CMB_K], ss[PLOW_MUSE_CMB_K];
    bool live[PLOW_MUSE_CMB_K];
#pragma unroll
    for (unsigned j = 0; j < PLOW_MUSE_CMB_K; j++) {
        gate[j] = j < k ? moe_slot_gate(table, j) : 0.0f;
        live[j] = j < k && moe_slot_expert(table, j) != PLOW_EXPERT_UNUSED && gate[j] > 0.0f;
        ss[j] = 0.0f;
    }
    for (unsigned i = threadIdx.x * 8u; i < H; i += PLOW_THREADS * 8u) {
        const bf16v8 wv = ld_glob8(w + i);
        bf16v8 yv[PLOW_MUSE_CMB_K];
#pragma unroll
        for (unsigned j = 0; j < PLOW_MUSE_CMB_K; j++)
            if (j < k) yv[j] = ld_glob8(part + (size_t)j * H + i);
#pragma unroll
        for (unsigned j = 0; j < PLOW_MUSE_CMB_K; j++) {
            if (!live[j]) continue;
#pragma unroll
            for (unsigned e = 0; e < 8; e++) {
                const float z = muse_expert_z(bf2f(yv[j][e]), gate[j], bf2f(wv[e]), ybf);
                ss[j] += z * z;
            }
        }
    }
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
#pragma unroll
    for (unsigned j = 0; j < PLOW_MUSE_CMB_K; j++) {
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
        const bool lj =
            moe_slot_expert(table, j) != PLOW_EXPERT_UNUSED && moe_slot_gate(table, j) > 0.0f;
        rinv[j] = lj ? rsqrtf(t / (float)H + eps) : 0.0f;
    }
    __syncthreads();
    for (unsigned d = slice * PLOW_THREADS + threadIdx.x; d < H; d += nblk * PLOW_THREADS) {
        float y[PLOW_MUSE_CMB_K];
#pragma unroll
        for (unsigned j = 0; j < PLOW_MUSE_CMB_K; j++)
            if (j < k) y[j] = bf2f(part[(size_t)j * H + d]);
        const float wd = bf2f(w[d]);
        float acc = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < PLOW_MUSE_CMB_K; j++) {
            if (j >= k) break;
            if (rinv[j] == 0.0f) continue;
            const float z = muse_expert_z(y[j], gate[j], wd, ybf);
            acc += gate[j] * bf2f(f2bf(z * rinv[j]));
        }
        out[d] = f2bf(acc);
        if (bw) bm[(size_t)(d / bw) * rows * bw + (size_t)row * bw + d % bw] = f2bf(acc);
    }
    __syncthreads();
}
#endif

__device__ void muse_moe_combine_norm_row(bf16* out, const float* part,
                                          const unsigned char* table, const bf16* w, unsigned H,
                                          unsigned k, float eps, unsigned slice, unsigned nblk,
                                          float* lds, unsigned ybf, bf16* bm, unsigned bw,
                                          unsigned rows, unsigned row) {
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
                const float z = muse_expert_z(pv[e], gate, bf2f(wv[e]), ybf);
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
            const float z = muse_expert_z(y, gate, bf2f(w[d]), ybf);
            acc += gate * bf2f(f2bf(z * rinv[j]));
        }
        out[d] = f2bf(acc);
        if (bw) bm[(size_t)(d / bw) * rows * bw + (size_t)row * bw + d % bw] = f2bf(acc);
    }
    __syncthreads();
}

__device__ void d_muse_moe_combine_norm(bf16* out, const float* part, const unsigned char* table,
                                        const bf16* w, unsigned H, unsigned k, float eps,
                                        unsigned slice, unsigned nblk, float* lds,
                                        unsigned ybf = 0, unsigned rows = 0, bf16* bm = nullptr,
                                        unsigned bw = 0) {
    if (k > PLOW_MUSE_MAX_TOPK || (H & 7u) != 0u || ybf > 2u || (bw && H % bw != 0u)) {
        if (threadIdx.x == 0) __builtin_trap();
        return;
    }
    if (rows <= 1u) {
#if PLOW_MUSE_CMB_ILP
        if (ybf && k <= PLOW_MUSE_CMB_K) {
            muse_moe_combine_norm_row_k8(out, (const bf16*)part, table, w, H, k, eps, slice, nblk,
                                         lds, ybf, bm, bw, 1u, 0u);
            return;
        }
#endif
        muse_moe_combine_norm_row(out, part, table, w, H, k, eps, slice, nblk, lds, ybf, bm, bw,
                                  1u, 0u);
        return;
    }
    const size_t prow = (size_t)k * H;
    for (unsigned r = slice; r < rows; r += nblk) {
        const float* pr = ybf ? (const float*)((const bf16*)part + r * prow) : part + r * prow;
        muse_moe_combine_norm_row(out + (size_t)r * H, pr, table + (size_t)r * k * 8, w, H, k, eps,
                                  0u, 1u, lds, ybf, bm, bw, rows, r);
    }
}

#endif

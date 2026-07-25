/* MEASURED ON THE PART (RTX 5090, 170 SM, 96 MB L2, under the GPU lease; n_head=64, DK=512,
 * DR=64, bench = runtime/tests/mla_sm120_bench.cu). Best decode latency per layer:
 *     ctx    GF nsplit    us    GB/s (latent stream)
 *     4096    8    16    37.0    1019      <- L2-RESIDENT (latent 4.7 MB)
 *    16384    8    16   135.3    1116      <- L2-RESIDENT (18.9 MB)
 *    32768    8    16   266.5    1133      <- L2-RESIDENT (37.7 MB)
 *   131072    8    16  1208.1    1000      <- HBM-BOUND   (151 MB > 96 MB L2)
 * GF=8 is the best ABSOLUTE latency at every ctx (1.21x over GF=4, 2.24x over GF=2 at 128k),
 * confirming that the latent re-read count n_head/GF is the dominant term. ptxas: GF=2 62 reg,
 * GF=4 84 reg (no spill), GF=8 128 reg (8 B spill).
 * CAVEAT / OPEN: GF=8 plateaus at ~1000 GB/s = 56% of the 1.79 TB/s HBM roofline and does not
 * improve past nsplit=16. Cause NOT identified (packed bf16 conversion was tried and refuted,
 * see mla_dot8). Do not present this kernel as roofline-saturating.
 *
 * op_mla.cuh — MLA (Multi-head Latent Attention) DECODE for sm_120a (RTX 5090 / GB202).
 *                                                                     [DEEPSEEK-MLA][SM120]
 * Port of the AMD reference bodies in runtime/amd/op_attention.h:
 *   d_flash_mla_decode<DK,DR,GF,GATHER>   (op_attention.h:1200)  -> PLOW_DOP_FLASH_MLA_DECODE (50)
 *                                                                   / PLOW_DOP_FLASH_GATHER_DECODE (54)
 *   d_mla_merge_fold<DK,VT>               (op_attention.h:2114)  -> PLOW_DOP_MLA_MERGE_FOLD (57)
 *
 * MATH (absorbed formulation, identical to the AMD kernel and runtime/tests/mla_ref.rs):
 *   KV cache is a single HEAD-SHARED low-rank latent per token,
 *       C_kv  [b][ctx][DK]   kv_lora_rank      (GLM-5.2 / DeepSeek: 512)
 *       K_rope[b][ctx][DR]   decoupled RoPE key (64), shared across all heads
 *   W_uk is absorbed into the query (q_abs = W_uk^T q_nope) and W_uv into the output, so the
 *   O(ctx) loop NEVER reconstructs per-head K/V:
 *       score[h][t] = scale * ( q_abs[h] . C_kv[t]  +  q_rope[h] . K_rope[t] )
 *       p           = softmax_t(score)
 *       oacc[h]     = sum_t p[t] * C_kv[t]                    (latent-wide, DK)
 *       o[h]        = W_uv[h]^T . oacc[h]                     (in R^V, ONCE per query)
 *
 * WHY ABSORBED (and not explicit) ON THIS PART — the arithmetic.
 *   Explicit form would reconstruct, per KV row, K[h] = W_uk[h]^T C_kv (QN=192) and
 *   V[h] = W_uv[h]^T C_kv (V=128) for all NH=64 heads:
 *       64 * 512 * (192+128) = 10.5 M MAC / kv-row,
 *   versus absorbed:
 *       64 * (512 + 64 + 512) = 69.6 K MAC / kv-row      (~150x less)
 *   for the SAME 1152 B/row of HBM traffic. At M=1 there are no query rows to amortize the
 *   reconstruction over, so absorbed wins by ~150x. (Explicit only becomes interesting at
 *   large prefill M, which is not this file.)
 *
 * WHY SCALAR FP32 FMA AND NOT mma.sync ON sm_120.
 *   sm_120a has NO wgmma and NO tcgen05: tensor math is warp-scoped mma.sync m16n8k16 only,
 *   which has a fixed M=16. Decode is M=1 per head, so filling M requires packing 16 query
 *   HEADS into the M dimension — which forces n_grp=1 (all heads in one workgroup), the exact
 *   configuration the AMD side measured as SLOWER (op_attention.h:1409-1417). More decisive,
 *   the roofline says the tensor core is not needed at all:
 *     - per head-group of GF heads, one KV row costs 1152 B of HBM and GF*1088*2 flop
 *       => intensity = 2.13*GF flop/B   (GF=4 -> 8.5, GF=8 -> 17.0)
 *     - RTX 5090 fp32 CUDA-core peak ~ 170 SM * 128 lane * 2 flop * ~2.4 GHz = 104 TFLOP/s
 *       over ~1.79 TB/s HBM => machine balance ~ 58 flop/B.
 *   17 << 58, so even at GF=8 the loop is HBM-bound by >3x on the PLAIN FP32 pipe. Tensor
 *   cores would buy nothing and cost the head-packing occupancy. The lever that matters is
 *   GF (latent re-read count = n_head/GF), not FLOPs.
 *
 * WARP32. Every reduction here is re-derived for a 32-lane warp: the AMD source reduces with
 * 6 __shfl_xor steps over wave64 and slices Ssm with `lane += 64`. Here it is 5 shfl steps
 * (offset 16..1) with `lane += 32`. This is NOT a transliteration.
 *
 * SMEM BUDGET (PLOW_THREADS=256, TILE=256, DK=512, DR=64, NG=4):
 *   Ssm GF*256 f32 | hmax/hsum 2*8 f32 | qsm GF*512 bf16 | qrsm GF*64 bf16 | osm 4*512 f32
 *   GF=2: 12.6 KiB   GF=4: 17.0 KiB   GF=8: 25.7 KiB      (limit ~100 KiB opt-in)
 *   => GF=4 leaves room for >=4 concurrent blocks/SM on smem alone.
 */
#ifndef PLOW_OP_MLA_CUH
#define PLOW_OP_MLA_CUH

#include <cuda_bf16.h>

#ifndef PLOW_THREADS
#define PLOW_THREADS 256
#endif
#define MLA_WARPS (PLOW_THREADS / 32)
#define MLA_TILE PLOW_THREADS /* KV rows scored per iteration: one row per thread */

typedef __nv_bfloat16 mla_bf16;

#define MLA_NEG_INF (-3.0e38f)
#define MLA_EXP(x) __expf(x)

/* 8 packed bf16 = one 16 B vector load. */
struct __align__(16) mla_bf16v8 {
    mla_bf16 v[8];
};
__device__ __forceinline__ mla_bf16v8 mla_ld8(const mla_bf16* p) {
    return *(const mla_bf16v8*)p;
}
/* dot8: 8-wide bf16 x bf16 -> f32 accumulate (matches the AMD dot8 reduction order).
 * Uses __bfloat1622float2 (2 bf16 -> 2 f32 per instruction) rather than scalar __bfloat162float.
 * HONEST NOTE: this was tried as a fix for the GF=8 @ctx=128k plateau at ~55% of the HBM
 * roofline, on the hypothesis that the inner dot is convert-issue bound. MEASURED: 1218 us ->
 * 1208 us, i.e. ~1% — the hypothesis is REFUTED and the GF=8 plateau is NOT the bf16 convert.
 * Kept because it is no worse and slightly tidier; the real cause is still open (leading
 * suspects: LDS issue/bank behaviour on the GF-wide qsm re-read, or memory-level parallelism
 * limited by 128 registers -> 2 blocks/SM). Do not cite this as an optimization. */
__device__ __forceinline__ float mla_dot8(const mla_bf16v8& a, const mla_bf16v8& b, float acc) {
    const __nv_bfloat162* a2 = (const __nv_bfloat162*)a.v;
    const __nv_bfloat162* b2 = (const __nv_bfloat162*)b.v;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float2 x = __bfloat1622float2(a2[i]), y = __bfloat1622float2(b2[i]);
        acc = fmaf(x.x, y.x, acc);
        acc = fmaf(x.y, y.y, acc);
    }
    return acc;
}

/* --- WARP32 reductions (re-derived; the AMD reference is wave64). --- */
__device__ __forceinline__ float mla_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float mla_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* Row-group geometry for the PV accumulate: each thread owns 8 consecutive latent columns,
 * NDT threads cover one DK-wide row, NG row-groups run concurrently. */
#define MLA_NDT(DK) ((DK) / 8)
#define MLA_NG(DK) (PLOW_THREADS / MLA_NDT(DK))

/* smem floats required by d_flash_mla_decode_sm120. */
#define MLA_DEC_SMEM_FLOATS(DK, DR, GF)                                                            \
    ((GF) * MLA_TILE + 2 * MLA_WARPS + (GF) * ((DK) / 2) + (GF) * ((DR) / 2) + MLA_NG(DK) * (DK))

/* ============================================================================================
 * MLA latent flash DECODE (M=1).                        PLOW_DOP_FLASH_MLA_DECODE / _GATHER
 *
 * Operand contract (matches the emitter, crates/plowc/src/bin/gemma4.rs:1975-1990):
 *   t0=Opart(f32 [b][h][nsplit][DK])  t1=mlpart(f32 [b][h][nsplit][2])
 *   t2=Qabs(bf16 [b][h][DK])          t3=Qrope(bf16 [b][h][DR])
 *   t4=Ckv(bf16 [b][kv_stride][DK])   t5=Krope(bf16 [b][kv_stride][DR])
 *   t6=kv_len(i32[b])                 t7=idx(i32 [b][top_k], GATHER only)
 *   i0=n_batch i1=n_head i2=kv_stride i3=window i4=nsplit i5=kv_mask i6=top_k i7=GF ; f0=scale
 * Emits latent-wide (O_partial, m, l); d_mla_merge_fold_sm120 consumes them.
 * ==========================================================================================*/
template <int DK, int DR, int GF, bool GATHER = false>
__device__ void d_flash_mla_decode_sm120(float* __restrict__ Opart, float* __restrict__ mlpart,
                                         const mla_bf16* __restrict__ Qabs,
                                         const mla_bf16* __restrict__ Qrope,
                                         const mla_bf16* __restrict__ Ckv,
                                         const mla_bf16* __restrict__ Krope,
                                         const int* __restrict__ kv_len, unsigned n_batch,
                                         unsigned n_head, unsigned kv_stride, unsigned window,
                                         float scale, unsigned nsplit, unsigned kv_mask,
                                         unsigned slice, unsigned nblk, float* smem,
                                         const int* __restrict__ idx = nullptr,
                                         unsigned top_k = 0) {
    static_assert(GF <= MLA_WARPS, "one warp per fused head in the softmax reduction");
    const unsigned n_grp = n_head / GF;
    const unsigned n_work = n_batch * n_grp * nsplit;
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5, lane = tid & 31;

    float* Ssm = smem;                          /* [GF][MLA_TILE] scores, then probabilities */
    float* hmax = Ssm + GF * MLA_TILE;          /* [MLA_WARPS] */
    float* hsum = hmax + MLA_WARPS;             /* [MLA_WARPS] */
    mla_bf16* qsm = (mla_bf16*)(hsum + MLA_WARPS);  /* [GF][DK] absorbed query rows */
    mla_bf16* qrsm = qsm + GF * DK;                 /* [GF][DR] rope query rows */
    float* osm = (float*)(qrsm + GF * DR);          /* [NG][DK] PV row-group fold buffer */

    constexpr unsigned NDT = MLA_NDT(DK);
    constexpr unsigned NG = MLA_NG(DK);
    constexpr int VU = (GF <= 4) ? 4 : 2; /* PV column-read unroll (latency hiding) */
    const unsigned dbase = (tid % NDT) * 8;
    const unsigned grp = tid / NDT;

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned hg = (w / nsplit) % n_grp;
        const unsigned b = w / (nsplit * n_grp);
        const unsigned h0 = hg * GF;

        const unsigned len = (unsigned)kv_len[b];
        const unsigned qpos = len - 1;
        const unsigned first = GATHER ? 0u : ((window && len > window) ? (len - window) : 0u);
        const unsigned span = GATHER ? top_k : (len - first);
        const unsigned per = (span + nsplit - 1) / nsplit;
        const unsigned lo = first + sp * per;
        const unsigned hi =
            GATHER ? (lo + per < top_k ? lo + per : top_k) : (lo + per < len ? lo + per : len);

        const mla_bf16* cbase = Ckv + (size_t)b * kv_stride * DK;
        const mla_bf16* rbase = Krope + (size_t)b * kv_stride * DR;
        const int* ibase = GATHER ? idx + (size_t)b * top_k : nullptr;

        __syncthreads();
        for (unsigned i = tid; i < GF * DK; i += PLOW_THREADS)
            qsm[i] = Qabs[((size_t)b * n_head + h0 + i / DK) * DK + i % DK];
        for (unsigned i = tid; i < GF * DR; i += PLOW_THREADS)
            qrsm[i] = Qrope[((size_t)b * n_head + h0 + i / DR) * DR + i % DR];
        __syncthreads();

        float m_st[GF], l_st[GF], oacc[GF][8];
#pragma unroll
        for (int g = 0; g < GF; g++) {
            m_st[g] = MLA_NEG_INF;
            l_st[g] = 0.0f;
#pragma unroll
            for (int u = 0; u < 8; u++) oacc[g][u] = 0.0f;
        }

        for (unsigned kv0 = lo; kv0 < hi; kv0 += MLA_TILE) {
            /* ---- SCORES: thread `tid` owns latent row kv0+tid, dots it against all GF queries.
             * The latent row crosses HBM ONCE per head-group -> traffic ~ n_head/GF. ---- */
            const unsigned kv = kv0 + tid;
            const unsigned row = GATHER ? (kv < hi ? (unsigned)ibase[kv] : 0u) : (kv & kv_mask);
            const bool keep = GATHER ? (kv < hi)
                                     : (kv < hi && kv <= qpos && (!window || (qpos - kv) < window));
            float s[GF];
#pragma unroll
            for (int g = 0; g < GF; g++) s[g] = MLA_NEG_INF;
            if (keep) {
                float dot[GF];
#pragma unroll
                for (int g = 0; g < GF; g++) dot[g] = 0.0f;
                const mla_bf16* crow = cbase + (size_t)row * DK;
#pragma unroll
                for (int d = 0; d < DK; d += 8) {
                    const mla_bf16v8 c8 = mla_ld8(crow + d);
#pragma unroll
                    for (int g = 0; g < GF; g++)
                        dot[g] = mla_dot8(c8, mla_ld8(qsm + g * DK + d), dot[g]);
                }
                const mla_bf16* rrow = rbase + (size_t)row * DR;
#pragma unroll
                for (int d = 0; d < DR; d += 8) {
                    const mla_bf16v8 r8 = mla_ld8(rrow + d);
#pragma unroll
                    for (int g = 0; g < GF; g++)
                        dot[g] = mla_dot8(r8, mla_ld8(qrsm + g * DR + d), dot[g]);
                }
#pragma unroll
                for (int g = 0; g < GF; g++) s[g] = dot[g] * scale;
            }
#pragma unroll
            for (int g = 0; g < GF; g++) Ssm[g * MLA_TILE + tid] = s[g];
            __syncthreads();

            /* ---- online softmax: warp g reduces head g's MLA_TILE scores (WARP32: lane += 32,
             * then 5 shfl_xor steps). ---- */
            if (warp < GF) {
                float mx = MLA_NEG_INF;
                for (unsigned i = lane; i < MLA_TILE; i += 32)
                    mx = fmaxf(mx, Ssm[warp * MLA_TILE + i]);
                mx = mla_warp_max(mx);
                if (lane == 0) hmax[warp] = mx;
            }
            __syncthreads();

            float mnew[GF], corr[GF], pe[GF];
#pragma unroll
            for (int g = 0; g < GF; g++) {
                mnew[g] = fmaxf(m_st[g], hmax[g]);
                corr[g] = (m_st[g] == MLA_NEG_INF) ? 0.0f : MLA_EXP(m_st[g] - mnew[g]);
                pe[g] = (mnew[g] == MLA_NEG_INF || s[g] == MLA_NEG_INF)
                            ? 0.0f
                            : MLA_EXP(s[g] - mnew[g]);
            }
            __syncthreads();
#pragma unroll
            for (int g = 0; g < GF; g++) Ssm[g * MLA_TILE + tid] = pe[g];
            __syncthreads();

            if (warp < GF) {
                float sm = 0.0f;
                for (unsigned i = lane; i < MLA_TILE; i += 32) sm += Ssm[warp * MLA_TILE + i];
                sm = mla_warp_sum(sm);
                if (lane == 0) hsum[warp] = sm;
            }
            __syncthreads();

#pragma unroll
            for (int g = 0; g < GF; g++) {
                l_st[g] = l_st[g] * corr[g] + hsum[g];
                m_st[g] = mnew[g];
#pragma unroll
                for (int u = 0; u < 8; u++) oacc[g][u] *= corr[g];
            }

            /* ---- PV: V IS the latent C_kv (k_eq_v), DK wide. Cooperative column read. ---- */
            const unsigned rmax = (hi - kv0 < MLA_TILE) ? (hi - kv0) : MLA_TILE;
            unsigned r = grp;
            for (; r + (VU - 1) * NG < rmax; r += VU * NG) {
                mla_bf16v8 vv[VU];
#pragma unroll
                for (int cc = 0; cc < VU; cc++) {
                    const unsigned t = kv0 + r + (unsigned)cc * NG;
                    const size_t vrow = GATHER ? (size_t)(unsigned)ibase[t] : (size_t)(t & kv_mask);
                    vv[cc] = mla_ld8(cbase + vrow * DK + dbase);
                }
#pragma unroll
                for (int cc = 0; cc < VU; cc++) {
                    float vf[8];
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const float2 t2 = __bfloat1622float2(((const __nv_bfloat162*)vv[cc].v)[u]);
                        vf[2 * u] = t2.x;
                        vf[2 * u + 1] = t2.y;
                    }
#pragma unroll
                    for (int g = 0; g < GF; g++) {
                        const float pw = Ssm[g * MLA_TILE + r + (unsigned)cc * NG];
#pragma unroll
                        for (int u = 0; u < 8; u++) oacc[g][u] = fmaf(pw, vf[u], oacc[g][u]);
                    }
                }
            }
            for (; r < rmax; r += NG) {
                const size_t vrow =
                    GATHER ? (size_t)(unsigned)ibase[kv0 + r] : (size_t)((kv0 + r) & kv_mask);
                const mla_bf16v8 v = mla_ld8(cbase + vrow * DK + dbase);
                float vf[8];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const float2 t2 = __bfloat1622float2(((const __nv_bfloat162*)v.v)[u]);
                    vf[2 * u] = t2.x;
                    vf[2 * u + 1] = t2.y;
                }
#pragma unroll
                for (int g = 0; g < GF; g++) {
                    const float pw = Ssm[g * MLA_TILE + r];
#pragma unroll
                    for (int u = 0; u < 8; u++) oacc[g][u] = fmaf(pw, vf[u], oacc[g][u]);
                }
            }
            __syncthreads();
        }

        /* ---- fold the NG row-groups, emit latent-wide partials, one head at a time. ---- */
#pragma unroll
        for (int g = 0; g < GF; g++) {
            __syncthreads();
#pragma unroll
            for (int u = 0; u < 8; u++) osm[grp * DK + dbase + u] = oacc[g][u];
            __syncthreads();

            const unsigned h = h0 + (unsigned)g;
            float* op = Opart + ((size_t)(b * n_head + h) * nsplit + sp) * DK;
            for (unsigned d = tid; d < DK; d += PLOW_THREADS) {
                float acc = 0.0f;
#pragma unroll
                for (unsigned gg = 0; gg < NG; gg++) acc += osm[gg * DK + d];
                op[d] = acc;
            }
            if (tid == 0) {
                float* ml = mlpart + ((size_t)(b * n_head + h) * nsplit + sp) * 2;
                ml[0] = m_st[g];
                ml[1] = l_st[g];
            }
        }
    }
}

/* ============================================================================================
 * FUSED MLA merge + W_uv fold.                                        PLOW_DOP_MLA_MERGE_FOLD
 *   t0=O(bf16 [b][h][V]) t1=Opart t2=mlpart t3=Wuv(bf16 [h][DK][V], l-major)
 *   i0=n_batch i1=n_head i2=V i4=nsplit
 * Online-softmax-merges the nsplit latent partials into olat[DK] in smem (f32), then folds
 * olat @ W_uv[h] straight to o[h][V]. Work is split over (b, head, V-tile of VT columns) so
 * the 170 SMs stay fed even at n_batch*n_head = 64.
 * ==========================================================================================*/
template <int DK, int VT>
__device__ void d_mla_merge_fold_sm120(mla_bf16* __restrict__ O, const float* __restrict__ Opart,
                                       const float* __restrict__ mlpart,
                                       const mla_bf16* __restrict__ Wuv, unsigned n_batch,
                                       unsigned n_head, unsigned V, unsigned nsplit,
                                       unsigned slice, unsigned nblk, float* olds /* DK floats */) {
    const unsigned tid = threadIdx.x;
    const unsigned vtiles = (V + VT - 1) / VT;
    const unsigned n_work = n_batch * n_head * vtiles;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned vt = w % vtiles;
        const unsigned bh = w / vtiles;
        const unsigned h = bh % n_head, b = bh / n_head;
        const float* ml = mlpart + (size_t)(b * n_head + h) * nsplit * 2;

        float gm = MLA_NEG_INF;
        for (unsigned s = 0; s < nsplit; s++) gm = fmaxf(gm, ml[s * 2]);
        float gl = 0.0f;
        for (unsigned s = 0; s < nsplit; s++) {
            if (ml[s * 2] == MLA_NEG_INF) continue;
            gl += ml[s * 2 + 1] * MLA_EXP(ml[s * 2] - gm);
        }
        const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

        __syncthreads();
        const float* opb = Opart + (size_t)(b * n_head + h) * nsplit * DK;
        for (unsigned d = tid; d < DK; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned s = 0; s < nsplit; s++) {
                if (ml[s * 2] == MLA_NEG_INF) continue;
                acc += opb[(size_t)s * DK + d] * MLA_EXP(ml[s * 2] - gm);
            }
            olds[d] = acc * inv;
        }
        __syncthreads();

        const mla_bf16* wv = Wuv + (size_t)h * DK * V;
        const unsigned v0 = vt * VT, v1 = (v0 + VT < V) ? (v0 + VT) : V;
        for (unsigned v = v0 + tid; v < v1; v += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned l = 0; l < DK; l++)
                acc = fmaf(olds[l], __bfloat162float(wv[(size_t)l * V + v]), acc);
            O[(size_t)(b * n_head + h) * V + v] = __float2bfloat16(acc);
        }
    }
}

#endif /* PLOW_OP_MLA_CUH */

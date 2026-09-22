/* op_gemv_mma.cuh — tensor-core row-block walk for BATCH>=8 decode GEMV (PLOW_NV_GEMV_MMA=1).
 *
 * WHY. gemv_rows<MM> streams each weight vector once, then dots it against MM activation rows
 * with dot8 on the CUDA cores: at MM=16 that is 16 activation loads and 128 FMAs per 16-byte
 * weight chunk, ~400 GFLOP of f32 FMA and ~25 G L1 loads per Gemma-4-12B step. Measured on
 * H100 (cell campaign-bf16-ladder, GV_MM_MAX=16, in128 so prefill is negligible): TPOT 12.7 ms
 * at B=1, 18.1 at B=4, 44.6 at B=16 — decode scales ~linearly with the batch instead of staying
 * at the weight-streaming floor (~7.4 ms for 24.8 GB at 3.35 TB/s). mma.sync m16n8k16 does the
 * same 16-row dot in one instruction per k16, so the walk goes back to being bandwidth-bound.
 *
 * SHAPE. One warp owns 8 consecutive output rows n (the mma N) and all 16 activation rows (the
 * mma M) for the whole K walk; K advances 32 per step (two k16 mmas). Fragment loads come
 * straight from global memory with a VIRTUAL K ORDER: lane (g = lane>>2, t = lane&3) loads the
 * 8 contiguous bf16 W[nb+g][kb+8t .. +8) as one 16-byte vector and treats them as the mma's
 * k positions {2t,2t+1,2t+8,2t+9} of step 0 and step 1. The A fragments (x rows g and g+8) use
 * the same mapping, and a dot product is invariant to a K permutation applied to both operands,
 * so the result is exact w.r.t. the true k order (f32 accumulation order differs from dot8's,
 * so outputs are NOT bit-identical to gemv_rows — same class of difference as any tile change).
 *
 * MASKING. Activation rows >= `rows` and weight rows >= N read from a clamped valid row and are
 * simply not stored (a garbage A row only pollutes its own C row, a garbage B column only its
 * own C column). No zeroing, no per-lane branch in the K loop.
 *
 * CONTRACT. K % 32 == 0 (every Gemma-4 K: 3840/4096/15360). Callers fall back to the dot8 walk
 * otherwise, and the QKV variant additionally needs Nq % 8 == Nk % 8 == 0 so an 8-row block never
 * straddles two weight matrices. */
#pragma once

#ifndef PLOW_NV_GEMV_MMA
#define PLOW_NV_GEMV_MMA 0
#endif
/* A single-stream GEMV walks two row blocks per k-step (manifest-set, dense packets). h100-sxm5
 * step_bench ms at B=1/4/16, Gemma-4-12B packet object: 10.99/11.70/14.25 -> 10.92/11.60/13.94 at
 * ctx 1024, 11.10/12.22/16.32 -> 11.03/12.10/15.96 at ctx 8192. That entry sits at the
 * 255-register cap (12 B of spills with the pair in); 18 registers under it the same change is
 * worth 0.27 ms at every rung (10.91/11.61/13.97 -> 10.64/11.33/13.76). Off for the MoE 26B:
 * 5.80/9.94/15.61 -> 5.83/10.15/15.69, its dense GEMVs are short walks. +14 KiB static smem. */
#ifndef PLOW_NV_GEMV_MMA_PAIR
#define PLOW_NV_GEMV_MMA_PAIR 0
#endif
/* k32 steps in flight per lane: UNB x (NW x 16 B) of weight loads before the first mma.
 * 12, measured on h100-sxm5 (step_bench ms at B=1/4/16): Gemma-4-12B 11.93/13.16/16.03 at 8,
 * 11.18/12.58/15.25 at 12, 11.52/12.93/15.77 at 16; 26B 5.79/10.06/15.89, 5.80/9.93/15.62,
 * 5.82/10.00/15.70. A 4-wide tail group for the leftover steps did not rescue 16. */
#ifndef PLOW_NV_GEMV_MMA_UNB
#define PLOW_NV_GEMV_MMA_UNB 12
#endif

static __device__ __forceinline__ __nv_bfloat16 gemma_glu_epilogue(float gate, float up,
                                                                 unsigned act);

__device__ __forceinline__ void gvmma_mma16816(float (&d)[4], unsigned a0, unsigned a1,
                                               unsigned a2, unsigned a3, unsigned b0,
                                               unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* acc[i][mt] is the mma C fragment of x[16mt..16mt+16) x W[i][nb..nb+8)^T: lane (g,t) holds
 * C[g][2t], C[g][2t+1], C[g+8][2t], C[g+8][2t+1]. MT m-tiles share one weight pass, so a 32- or
 * 64-row rung still streams the weights once. */
/* ONE: the caller has at most 8 activation rows (rungs 1-8). Rows 8-15 of the mma are then never
 * stored, so the second activation load of every k-step can alias the first: a third of the
 * walk's loads. */
template <int NW, int MT, bool ONE = false>
__device__ __forceinline__ void gvmma_tile(float (&acc)[NW][MT][4], const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* const (&W)[NW], unsigned nb,
                                           unsigned rows, unsigned N, unsigned K,
                                           unsigned kb_begin = 0u, unsigned kb_end = 0xffffffffu) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned g = lane >> 2, t = lane & 3;
    const __nv_bfloat16* xr[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const unsigned r0 = 16u * mt + g, r1 = r0 + 8u;
        xr[mt][0] = x + (size_t)(r0 < rows ? r0 : 0u) * K + t * 8u;
        xr[mt][1] = x + (size_t)(r1 < rows ? r1 : 0u) * K + t * 8u;
    }
    const unsigned n = nb + g;
    const __nv_bfloat16* w[NW];
#pragma unroll
    for (int i = 0; i < NW; i++) w[i] = W[i] + (size_t)(n < N ? n : 0u) * K + t * 8u;
#pragma unroll
    for (int i = 0; i < NW; i++)
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][mt][j] = 0.0f;

    /* Loads in flight scale down with the tile count: MT=4 (the 64-row rung) at the full depth
     * pushed the sm_90a decode object to 255 regs + 3.7 KB spills; depth 8/MT keeps it clean. */
    constexpr unsigned UNB = (PLOW_NV_GEMV_MMA_UNB / MT) < 2u ? 2u : (PLOW_NV_GEMV_MMA_UNB / MT);
    const unsigned nkb = (kb_end < (K >> 5)) ? kb_end : (K >> 5);
    unsigned kb = kb_begin;
    for (; kb + UNB <= nkb; kb += UNB) {
        uint4 wv[NW][UNB];
#pragma unroll
        for (unsigned u = 0; u < UNB; u++)
#pragma unroll
            for (int i = 0; i < NW; i++) wv[i][u] = *(const uint4*)(w[i] + (kb + u) * 32u);
#pragma unroll
        for (unsigned u = 0; u < UNB; u++) {
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                const uint4 a0 = *(const uint4*)(xr[mt][0] + (kb + u) * 32u);
                const uint4 a1 = ONE ? a0 : *(const uint4*)(xr[mt][1] + (kb + u) * 32u);
#pragma unroll
                for (int i = 0; i < NW; i++) {
                    gvmma_mma16816(acc[i][mt], a0.x, a1.x, a0.y, a1.y, wv[i][u].x, wv[i][u].y);
                    gvmma_mma16816(acc[i][mt], a0.z, a1.z, a0.w, a1.w, wv[i][u].z, wv[i][u].w);
                }
            }
        }
    }
    for (; kb < nkb; kb++) {
        uint4 wv[NW];
#pragma unroll
        for (int i = 0; i < NW; i++) wv[i] = *(const uint4*)(w[i] + kb * 32u);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const uint4 a0 = *(const uint4*)(xr[mt][0] + kb * 32u);
            const uint4 a1 = ONE ? a0 : *(const uint4*)(xr[mt][1] + kb * 32u);
#pragma unroll
            for (int i = 0; i < NW; i++) {
                gvmma_mma16816(acc[i][mt], a0.x, a1.x, a0.y, a1.y, wv[i].x, wv[i].y);
                gvmma_mma16816(acc[i][mt], a0.z, a1.z, a0.w, a1.w, wv[i].z, wv[i].w);
            }
        }
    }
}

/* Row-block partition shared by the three walks: block `slice` owns 8-row blocks
 * [rb0, rb1) of the N (or concatenated N) range; warps stride through them. */
struct gvmma_range { unsigned rb0, rb1; };
__device__ __forceinline__ gvmma_range gvmma_partition(unsigned N, unsigned slice, unsigned nblk) {
    const unsigned nrb = (N + 7u) >> 3;
    const unsigned per = (nrb + nblk - 1u) / nblk;
    /* Trailing blocks can start past nrb (480 row blocks over 132 blocks: block 131 starts at
     * 524); clamp so rb1 - rb0 is 0 there, never an unsigned underflow. */
    const unsigned rb0 = (slice * per < nrb) ? slice * per : nrb;
    const unsigned rb1 = (rb0 + per < nrb) ? (rb0 + per) : nrb;
    return {rb0, rb1};
}

/* Two adjacent bf16 of one C row as a 4-byte store: n is even and N is even (N % 8 == 0). */
__device__ __forceinline__ void gvmma_store2(__nv_bfloat16* C, unsigned N, unsigned m, unsigned n,
                                             float c0, float c1) {
    if (n + 1u < N) {
        *(__nv_bfloat162*)(C + (size_t)m * N + n) = __floats2bfloat162_rn(c0, c1);
    } else if (n < N) {
        C[(size_t)m * N + n] = __float2bfloat16(c0);
    }
}

/* Split-K reduction slots: partial warps (part > 0) park their C fragments here, part 0 sums.
 * Sized for the widest tile; static so the arms need no arena hand-off. */
template <int MT>
struct gvmma_red_t { float v[PLOW_NV_WARPS - 1][32][MT * 4]; };
/* One slot set per MT, shared by the GEMV and GLU split-K walks. */
template <int MT>
__device__ __forceinline__ gvmma_red_t<MT>& gvmma_red() {
    __shared__ gvmma_red_t<MT> red;
    return red;
}

template <bool BIAS, int MT>
__device__ __forceinline__ void gvmma_store_tile(__nv_bfloat16* __restrict__ C, const float (&acc)[MT][4],
                                                 unsigned nb, unsigned rows, unsigned N,
                                                 const __nv_bfloat16* __restrict__ bias) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned n = nb + 2u * t;
    float b0 = 0.0f, b1 = 0.0f;
    if constexpr (BIAS) {
        b0 = (n < N) ? __bfloat162float(bias[n]) : 0.0f;
        b1 = (n + 1u < N) ? __bfloat162float(bias[n + 1u]) : 0.0f;
    }
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const unsigned m0 = 16u * mt + g;
        if (m0 < rows) gvmma_store2(C, N, m0, n, acc[mt][0] + b0, acc[mt][1] + b1);
        if (m0 + 8u < rows) gvmma_store2(C, N, m0 + 8u, n, acc[mt][2] + b0, acc[mt][3] + b1);
    }
}

template <bool BIAS, int MT = 1, bool ONE = false>
__device__ __forceinline__ void gemv_rows_mma(__nv_bfloat16* __restrict__ C,
                                              const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ W, unsigned rows,
                                              unsigned N, unsigned K, unsigned slice,
                                              unsigned nblk,
                                              const __nv_bfloat16* __restrict__ bias = nullptr) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const gvmma_range r = gvmma_partition(N, slice, nblk);
    const __nv_bfloat16* const W1[1] = {W};
    const unsigned per = r.rb1 - r.rb0;
    const unsigned nkb = K >> 5;
#if PLOW_NV_GEMV_MMA_PAIR
    /* TWO ROW BLOCKS PER K-STEP. A k-step's cost is mostly its round trip, not its bytes: the
     * two-stream GLU step measured 1.56x a one-stream step while moving 2x the weights (GLU walks
     * at ~3.2 TB/s, down / o_proj / lm_head at ~2.4). So a single-stream GEMV walks row blocks rb
     * and rb+1 as two streams of one tile. The second stream of an odd tail re-reads the first's
     * rows and is not stored. */
    {
        const unsigned ngrp = (per + 1u) >> 1;
        unsigned gpow = 1u;
        while (gpow < ngrp) gpow <<= 1;
        /* Narrow N: the pairs of this block split K over S warps each, as the unpaired split-K
         * below does per row block — taken only when S divides the k-steps, else that one runs. */
        if (per != 0u && gpow <= PLOW_NV_WARPS / 2u && (nkb % (PLOW_NV_WARPS / gpow)) == 0u) {
            __shared__ gvmma_red_t<MT> red2[2];
            const unsigned S = PLOW_NV_WARPS / gpow;
            const unsigned grp = warp / S, part = warp % S;
            const unsigned rb = r.rb0 + 2u * grp;
            const bool live = grp < ngrp;
            const bool two = live && rb + 1u < r.rb1 && ((rb + 2u) << 3) <= N;
            const __nv_bfloat16* const W2[2] = {W, two ? W + (size_t)8u * K : W};
            float acc[2][MT][4];
            const unsigned span = nkb / S;
            gvmma_tile<2, MT, ONE>(acc, x, W2, live ? (rb << 3) : (r.rb0 << 3), rows, N, K,
                                   part * span, (part + 1u) * span);
            if (part != 0u) {
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int mt = 0; mt < MT; mt++)
#pragma unroll
                        for (int j = 0; j < 4; j++)
                            red2[i].v[warp - 1u - grp][lane][mt * 4 + j] = acc[i][mt][j];
            }
            __syncthreads();
            if (part == 0u && live) {
                for (unsigned q = 1u; q < S; q++) {
                    const unsigned slot = (grp * S + q) - 1u - grp;
#pragma unroll
                    for (int i = 0; i < 2; i++)
#pragma unroll
                        for (int mt = 0; mt < MT; mt++)
#pragma unroll
                            for (int j = 0; j < 4; j++)
                                acc[i][mt][j] += red2[i].v[slot][lane][mt * 4 + j];
                }
                gvmma_store_tile<BIAS, MT>(C, acc[0], rb << 3, rows, N, bias);
                if (two) gvmma_store_tile<BIAS, MT>(C, acc[1], (rb + 1u) << 3, rows, N, bias);
            }
            __syncthreads(); /* red2 is reused by the next call on this block */
            return;
        }
        /* Wide N: every warp has at least two row blocks of its own. */
        if (per >= 2u * PLOW_NV_WARPS) {
            for (unsigned rb = r.rb0 + 2u * warp; rb < r.rb1; rb += 2u * PLOW_NV_WARPS) {
                const bool two = rb + 1u < r.rb1 && ((rb + 2u) << 3) <= N;
                const __nv_bfloat16* const W2[2] = {W, two ? W + (size_t)8u * K : W};
                float acc[2][MT][4];
                gvmma_tile<2, MT, ONE>(acc, x, W2, rb << 3, rows, N, K);
                gvmma_store_tile<BIAS, MT>(C, acc[0], rb << 3, rows, N, bias);
                if (two) gvmma_store_tile<BIAS, MT>(C, acc[1], (rb + 1u) << 3, rows, N, bias);
            }
            return;
        }
    }
#endif
    /* SPLIT-K. A narrow N (down: 3840 -> 480 row blocks over 132 blocks = 4 per block) leaves
     * half the warps idle and the walk at 1.4 TB/s vs 2.2-2.6 for wide shapes. When the block
     * has at most WARPS/2 row blocks, S = WARPS/next_pow2(per) warps share one row block and
     * each walks 1/S of K; parts > 0 park their fragments in smem and part 0 sums them in a
     * fixed order (bit-stable across runs). One tile per warp, so the block barrier is uniform. */
    unsigned per_pow = 1u;
    while (per_pow < per) per_pow <<= 1;
    if (per != 0u && per_pow <= PLOW_NV_WARPS / 2u && (nkb % (PLOW_NV_WARPS / per_pow)) == 0u) {
        gvmma_red_t<MT>& red = gvmma_red<MT>();
        const unsigned S = PLOW_NV_WARPS / per_pow;
        const unsigned grp = warp / S, part = warp % S;
        const unsigned rb = r.rb0 + grp;
        const bool live = grp < per;
        float acc[1][MT][4];
        const unsigned span = nkb / S;
        gvmma_tile<1, MT, ONE>(acc, x, W1, live ? (rb << 3) : (r.rb0 << 3), rows, N, K, part * span,
                          (part + 1u) * span);
        if (part != 0u) {
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int j = 0; j < 4; j++) red.v[warp - 1u - grp][lane][mt * 4 + j] = acc[0][mt][j];
        }
        __syncthreads();
        if (part == 0u && live) {
            for (unsigned p = 1u; p < S; p++) {
                const unsigned slot = (grp * S + p) - 1u - grp;
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[0][mt][j] += red.v[slot][lane][mt * 4 + j];
            }
            gvmma_store_tile<BIAS, MT>(C, acc[0], rb << 3, rows, N, bias);
        }
        __syncthreads(); /* red is reused by the next call on this block */
        return;
    }
    for (unsigned rb = r.rb0 + warp; rb < r.rb1; rb += PLOW_NV_WARPS) {
        float acc[1][MT][4];
        gvmma_tile<1, MT, ONE>(acc, x, W1, rb << 3, rows, N, K);
        gvmma_store_tile<BIAS, MT>(C, acc[0], rb << 3, rows, N, bias);
    }
}

template <int MT = 1, bool ONE = false>
__device__ __forceinline__ void gemv_glu_rows_mma(__nv_bfloat16* __restrict__ C,
                                                  const __nv_bfloat16* __restrict__ x,
                                                  const __nv_bfloat16* __restrict__ Wg,
                                                  const __nv_bfloat16* __restrict__ Wu,
                                                  unsigned rows, unsigned N, unsigned K,
                                                  unsigned act, unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned g = lane >> 2, t = lane & 3;
    const gvmma_range r = gvmma_partition(N, slice, nblk);
    const __nv_bfloat16* const W2[2] = {Wg, Wu};
    auto store = [&](const float (&acc)[2][MT][4], unsigned rb) {
        const unsigned n = (rb << 3) + 2u * t;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const unsigned m0 = 16u * mt + g, m1 = m0 + 8u;
            if (m0 < rows) {
                if (n < N) C[(size_t)m0 * N + n] = gemma_glu_epilogue(acc[0][mt][0], acc[1][mt][0], act);
                if (n + 1u < N) C[(size_t)m0 * N + n + 1u] = gemma_glu_epilogue(acc[0][mt][1], acc[1][mt][1], act);
            }
            if (m1 < rows) {
                if (n < N) C[(size_t)m1 * N + n] = gemma_glu_epilogue(acc[0][mt][2], acc[1][mt][2], act);
                if (n + 1u < N) C[(size_t)m1 * N + n + 1u] = gemma_glu_epilogue(acc[0][mt][3], acc[1][mt][3], act);
            }
        }
    };
#if !PLOW_NV_GEMV_MMA_PAIR
    /* SPLIT-K, as gemv_rows_mma's: the 26B's N=2112 is 2 row blocks per block, so 6 of 8 warps sat
     * idle. gate and up reduce one after the other through the one-stream slots. 26B step_bench
     * ms at B=16/4/1, ctx 1024: 11.64/8.37/5.55 -> 11.51/8.17/5.50. Packets that stamp the pair
     * walk have wide N and never split; the arm's code alone cost the 12B 0.09-0.12 ms. */
    const unsigned per = r.rb1 - r.rb0, nkb = K >> 5;
    unsigned per_pow = 1u;
    while (per_pow < per) per_pow <<= 1;
    if (per != 0u && per_pow <= PLOW_NV_WARPS / 2u && (nkb % (PLOW_NV_WARPS / per_pow)) == 0u) {
        gvmma_red_t<MT>& red = gvmma_red<MT>();
        const unsigned S = PLOW_NV_WARPS / per_pow;
        const unsigned grp = warp / S, part = warp % S;
        const unsigned rb = r.rb0 + grp;
        const bool live = grp < per;
        float acc[2][MT][4];
        const unsigned span = nkb / S;
        gvmma_tile<2, MT, ONE>(acc, x, W2, live ? (rb << 3) : (r.rb0 << 3), rows, N, K, part * span,
                               (part + 1u) * span);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (part != 0u) {
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
#pragma unroll
                    for (int j = 0; j < 4; j++) red.v[warp - 1u - grp][lane][mt * 4 + j] = acc[i][mt][j];
            }
            __syncthreads();
            if (part == 0u && live) {
                for (unsigned p = 1u; p < S; p++) {
                    const unsigned slot = (grp * S + p) - 1u - grp;
#pragma unroll
                    for (int mt = 0; mt < MT; mt++)
#pragma unroll
                        for (int j = 0; j < 4; j++) acc[i][mt][j] += red.v[slot][lane][mt * 4 + j];
                }
            }
            __syncthreads(); /* red is reused by the up stream / the next call on this block */
        }
        if (part == 0u && live) store(acc, rb);
        return;
    }
#endif
    for (unsigned rb = r.rb0 + warp; rb < r.rb1; rb += PLOW_NV_WARPS) {
        float acc[2][MT][4];
        gvmma_tile<2, MT, ONE>(acc, x, W2, rb << 3, rows, N, K);
        store(acc, rb);
    }
}

/* Fused q|k|v: row blocks over the concatenated [0, Nq+Nk+Nv); Nq % 8 == Nk % 8 == 0 (caller
 * checks) so a block lies inside one matrix. */
template <bool BIAS, int MT = 1, bool ONE = false>
__device__ __forceinline__ void gemv_qkv_rows_mma(
    __nv_bfloat16* Cq, __nv_bfloat16* Ck, __nv_bfloat16* Cv, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ Wq, const __nv_bfloat16* __restrict__ Wk,
    const __nv_bfloat16* __restrict__ Wv, unsigned rows, unsigned Nq, unsigned Nk, unsigned Nv,
    unsigned K, unsigned slice, unsigned nblk, const __nv_bfloat16* __restrict__ bq = nullptr,
    const __nv_bfloat16* __restrict__ bk = nullptr, const __nv_bfloat16* __restrict__ bv = nullptr) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned g = lane >> 2, t = lane & 3;
    const gvmma_range r = gvmma_partition(Nq + Nk + Nv, slice, nblk);
    for (unsigned rb = r.rb0 + warp; rb < r.rb1; rb += PLOW_NV_WARPS) {
        const unsigned gn = rb << 3;
        const __nv_bfloat16* W;
        __nv_bfloat16* C;
        const __nv_bfloat16* bias;
        unsigned Nx, n0;
        if (gn < Nq) { W = Wq; C = Cq; Nx = Nq; n0 = 0u; bias = bq; }
        else if (gn < Nq + Nk) { W = Wk; C = Ck; Nx = Nk; n0 = Nq; bias = bk; }
        else { W = Wv; C = Cv; Nx = Nv; n0 = Nq + Nk; bias = bv; }
        const __nv_bfloat16* const W1[1] = {W};
        float acc[1][MT][4];
        gvmma_tile<1, MT, ONE>(acc, x, W1, gn - n0, rows, Nx, K);
        const unsigned n = (gn - n0) + 2u * t;
        float b0 = 0.0f, b1 = 0.0f;
        if constexpr (BIAS) {
            b0 = (n < Nx) ? __bfloat162float(bias[n]) : 0.0f;
            b1 = (n + 1u < Nx) ? __bfloat162float(bias[n + 1u]) : 0.0f;
        }
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const unsigned m0 = 16u * mt + g;
            if (m0 < rows) gvmma_store2(C, Nx, m0, n, acc[0][mt][0] + b0, acc[0][mt][1] + b1);
            if (m0 + 8u < rows) gvmma_store2(C, Nx, m0 + 8u, n, acc[0][mt][2] + b0, acc[0][mt][3] + b1);
        }
    }
}

/* CLAIM-AHEAD L2 PREFETCH (PLOW_GEMV_PREFETCH, the AMD L8 knob). The interpreter calls these
 * between a packet's claim and its gate: the block asks L2 for the head of every stream its walk
 * starts with, so the slice's first k-steps come from L2 once the gate opens and HBM works through
 * the narrow producers (NRN, attention, the previous walk's tail) instead of idling. The budget is
 * split evenly over the heads because the block finishes with its slowest warp. The shapes mirror
 * the walks above; a mismatch costs bandwidth, never correctness (hints only). Gemma-4-12B
 * step_bench ms at B=1/4/16, ctx 192 (control 10.55/10.75/11.44): 16 KiB 10.48/10.65/11.47,
 * 32 KiB 10.44/10.60/11.43, 48 KiB 10.43/10.58/11.44, 64 KiB 10.44/10.57/11.41, 128 KiB
 * 10.48/10.66/11.48, 256 KiB 10.49/10.76/11.64 — past ~64 KiB the prefetch competes with the
 * walks still in flight. */
#if PLOW_GEMV_PREFETCH
#ifndef PLOW_NV_GEMV_PF_BYTES
#define PLOW_NV_GEMV_PF_BYTES 65536u
#endif
__device__ __forceinline__ void gvmma_pf_l2(const __nv_bfloat16* p, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
/* Bytes per head: the budget over `heads`, at most one span, whole 128 B lines. */
__device__ __forceinline__ unsigned gvmma_pf_bytes(unsigned heads, unsigned span_cols) {
    unsigned b = PLOW_NV_GEMV_PF_BYTES / heads;
    if (b > span_cols * 2u) b = span_cols * 2u;
    return b & ~127u;
}
/* Rows [row0, row0 + nrow) of W, each as S heads at columns 0, K/S, 2K/S, ... */
__device__ __forceinline__ void gvmma_pf_heads(const __nv_bfloat16* W, unsigned row0, unsigned nrow,
                                               unsigned K, unsigned S, unsigned bytes) {
    const unsigned span = K / S;
    for (unsigned p = threadIdx.x; p < nrow * S; p += blockDim.x)
        gvmma_pf_l2(W + (size_t)(row0 + p / S) * K + (p % S) * span, bytes);
}
/* gemv_rows_mma: R row blocks in the first wave, S K-spans each. */
__device__ __forceinline__ void gvmma_pf_rows(const __nv_bfloat16* W, unsigned N, unsigned K,
                                              unsigned slice, unsigned nblk) {
    const gvmma_range r = gvmma_partition(N, slice, nblk);
    const unsigned per = r.rb1 - r.rb0, nkb = K >> 5;
    if (per == 0u || (K & 31u)) return;
    unsigned R = per < PLOW_NV_WARPS ? per : PLOW_NV_WARPS, S = 1u;
    bool paired = false;
#if PLOW_NV_GEMV_MMA_PAIR
    const unsigned ngrp = (per + 1u) >> 1;
    unsigned gpow = 1u;
    while (gpow < ngrp) gpow <<= 1;
    if (gpow <= PLOW_NV_WARPS / 2u && (nkb % (PLOW_NV_WARPS / gpow)) == 0u) {
        R = per; S = PLOW_NV_WARPS / gpow; paired = true;
    } else if (per >= 2u * PLOW_NV_WARPS) {
        R = 2u * PLOW_NV_WARPS; paired = true;
    }
#endif
    if (!paired) {
        unsigned per_pow = 1u;
        while (per_pow < per) per_pow <<= 1;
        if (per_pow <= PLOW_NV_WARPS / 2u && (nkb % (PLOW_NV_WARPS / per_pow)) == 0u) {
            R = per; S = PLOW_NV_WARPS / per_pow;
        }
    }
    const unsigned row0 = r.rb0 << 3;
    const unsigned nrow = (R << 3) < N - row0 ? (R << 3) : N - row0;
    const unsigned bytes = gvmma_pf_bytes(nrow * S, K / S);
    if (bytes) gvmma_pf_heads(W, row0, nrow, K, S, bytes);
}
/* gemv_glu_rows_mma: one row block per warp, gate and up. */
__device__ __forceinline__ void gvmma_pf_glu(const __nv_bfloat16* Wg, const __nv_bfloat16* Wu,
                                             unsigned N, unsigned K, unsigned slice, unsigned nblk) {
    const gvmma_range r = gvmma_partition(N, slice, nblk);
    const unsigned per = r.rb1 - r.rb0;
    if (per == 0u || (K & 31u)) return;
    const unsigned row0 = r.rb0 << 3;
    const unsigned R = per < PLOW_NV_WARPS ? per : PLOW_NV_WARPS;
    const unsigned nrow = (R << 3) < N - row0 ? (R << 3) : N - row0;
    const unsigned bytes = gvmma_pf_bytes(2u * nrow, K);
    if (!bytes) return;
    gvmma_pf_heads(Wg, row0, nrow, K, 1u, bytes);
    gvmma_pf_heads(Wu, row0, nrow, K, 1u, bytes);
}
/* gemv_qkv_rows_mma: one row block per warp over the concatenated [q; k; v] rows. */
__device__ __forceinline__ void gvmma_pf_qkv(const __nv_bfloat16* Wq, const __nv_bfloat16* Wk,
                                             const __nv_bfloat16* Wv, unsigned Nq, unsigned Nk,
                                             unsigned Nv, unsigned K, unsigned slice, unsigned nblk) {
    const unsigned Nx = Nq + Nk + Nv;
    const gvmma_range r = gvmma_partition(Nx, slice, nblk);
    const unsigned per = r.rb1 - r.rb0;
    if (per == 0u || (K & 31u)) return;
    const unsigned row0 = r.rb0 << 3;
    const unsigned R = per < PLOW_NV_WARPS ? per : PLOW_NV_WARPS;
    const unsigned nrow = (R << 3) < Nx - row0 ? (R << 3) : Nx - row0;
    const unsigned bytes = gvmma_pf_bytes(nrow, K);
    if (!bytes) return;
    for (unsigned p = threadIdx.x; p < nrow; p += blockDim.x) {
        const unsigned g = row0 + p;
        const __nv_bfloat16* w = g < Nq ? Wq + (size_t)g * K
                               : g < Nq + Nk ? Wk + (size_t)(g - Nq) * K
                                             : Wv + (size_t)(g - Nq - Nk) * K;
        gvmma_pf_l2(w, bytes);
    }
}
#endif

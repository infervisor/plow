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
/* The last wave of each walk splits its leftover row blocks over K (manifest `gemv_rem_split`,
 * recipe opt-in: it reorders those sums). */
#ifndef PLOW_NV_GEMV_REM_SPLIT
#define PLOW_NV_GEMV_REM_SPLIT 0
#endif
/* k32 steps in flight per lane: UNB x (NW x 16 B) of weight loads before the first mma.
 * 12, measured on h100-sxm5 (step_bench ms at B=1/4/16): Gemma-4-12B 11.93/13.16/16.03 at 8,
 * 11.18/12.58/15.25 at 12, 11.52/12.93/15.77 at 16; 26B 5.79/10.06/15.89, 5.80/9.93/15.62,
 * 5.82/10.00/15.70. A 4-wide tail group for the leftover steps did not rescue 16. */
#ifndef PLOW_NV_GEMV_MMA_UNB
#define PLOW_NV_GEMV_MMA_UNB 12
#endif
/* ADDNORM FOLD (manifest `norm_fold`, emit knob PLOW_DECODE_NORM_FOLD): the pre-norm decode packet
 * `x += o; h = RMSNorm(x) * gamma` leaves the chain with its gate. The producer (the o_proj / down
 * GEMV) adds bf16(acc) into the residual stream in place, writes h' = x * gamma unscaled and, per
 * 8-column row block, each row's sum of squares ([N / 8][32] f32). The consumer (GemvQkv / GemvGlu)
 * walks h' and scales its accumulators by inv[m] = rsqrt(sum / K + eps): h's bf16 rounding moves
 * ahead of the 1/rms scale, everything else is the AddNorm's arithmetic. */
#ifndef PLOW_NV_NORM_FOLD
#define PLOW_NV_NORM_FOLD 0
#endif
/* CROSS-BLOCK SPLIT-K (manifest `gemv_xk`, emit knob PLOW_DECODE_GEMV_XK = the first rung, sm_120a
 * decode): from that many rows the Gemv walk (o_proj / down, plain and fold producer) splits K over
 * blocks instead of only over warps. The row walk's every block reads all rows x K of the
 * activations for its 8-row blocks (0.6B down at B=32: 196 KB of them per 48 KB of weights). A
 * unit is 64 output rows (one row block per warp) x K / sb, sb the largest of 8 / 4 / 2 with
 * (N / 64) * sb <= nblk units; the group's last block sums the sb partials in order (bit-stable
 * across runs, not bit-identical to the row walk) and runs the epilogue. `gemv_xk_qkv` (knob
 * PLOW_DECODE_GEMV_XK_QKV) splits the GemvQkv walk (plain and fold consumer) the same way; its own
 * define because the 1.7B object loses ~1% on every rung to the arm's presence alone. */
#ifndef PLOW_NV_GEMV_XK
#define PLOW_NV_GEMV_XK 0
#endif
#ifndef PLOW_NV_GEMV_XK_QKV
#define PLOW_NV_GEMV_XK_QKV 0
#endif
struct GvFoldOut {
    __nv_bfloat16* resid;
    const __nv_bfloat16* gamma;
    float* part;
};
struct GvFoldIn {
    const float* part; /* the producer's N = the consumer's K: K / 8 row blocks of 32 rows */
    float eps;
};

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
     * pushed the sm_90a decode object to 255 regs + 3.7 KB spills; depth 8/MT keeps it clean.
     *
     * But `UNB/MT` overshoots at MT=2. Loads in flight are what hides HBM latency, and halving
     * them at the 32-row rung is where the batched-decode knee lives: marginal ms per added
     * stream 0.037 (B4->8), 0.069 (B8->16), 0.163 (B16->32), and 2.20 TB/s at B=1 against 1.78 at
     * B=32 over the same 23.8 GB weight stream. So MT=2 takes 2/3 of the depth rather than half,
     * MT=1 keeps the full 12, and MT>=4 keeps `UNB/MT` exactly — no shipped decode ladder
     * instantiates the 64-row rung, so it is unmeasured. A global knob cannot express that.
     *
     * MEASURED IN SITU: step_bench at ctx 128 against two Gemma-4-12B packets built from the same
     * recipe and the same emit env, differing ONLY in this formula (both REG:255 STACK:192
     * SHARED:40464; 3 reps, spread <= 0.011 ms):
     *
     *     UNB@MT1 / UNB@MT2      B=1      B=16      B=32
     *          12 / 6         10.724    11.582    14.239     <- UNB/MT
     *          12 / 8         10.766    11.586    13.720     <- this formula: B=32 -3.64%
     *
     * B=1 pays +0.39% although its MT=1 code is unchanged: the decode object is one kernel and
     * the deeper MT=2 unroll adds 41.6 KB to it. Served C32 tracks the kernel once diluted by the
     * rest of TPOT (3 ladders per arm) — at 1024 in the step is 14.2 of 24.2 ms, predicting
     * -2.1% against -2.19% measured; at 8192 in, 14.2 of 82.1 ms, predicting and measuring
     * -0.63%. Below 8192 the served ladder's own run-to-run band (TTFT spread on identical
     * prefill objects: 6.63% at 1024, 1.37% at 4096, 0.21% at 8192, 3.32% at 15000) is wider
     * than the effect, so the ladder cannot adjudicate this change — the packet-level step can.
     *
     * RETRACTED: "NOT ACTIVATION-BOUND" AND "NOT TENSOR-CORE ISSUE". Both claims stood here on
     * probes that measured nothing. Each injected an `#if`-guarded edit and enabled it with
     * -DGVMMA_PROBE_* through PLOW_EXTRA_DEFINES. That variable is NOT plumbed into a
     * `campaign.py build`: its knob is registered as
     *     KnobSpec::new("def.PLOW_EXTRA_DEFINES", None, Layer::ObjectDefine, ...)
     * whose second field -- the environment binding -- is None. Only build_sm90a_cubin.sh and the
     * tune sweeps read it, and those build the GENERIC object, not the packet. The macro was
     * never defined, the guard compiled to the original source, and each probe compared the
     * shipped kernel against ITSELF. The deltas (activations +0.01%/+0.03%, mma +0.09%/0.00%)
     * are run-to-run noise, which is why they looked so clean -- as did the identical
     * REG:255 STACK:192 SHARED:40464 gate, which matched because it was the same object twice.
     * A macro-gated probe cannot be trusted here: edit the source UNCONDITIONALLY and gate on
     * the probe cubin's md5 DIFFERING from the control's before believing a number.
     *
     * WHAT THE WALK ACTUALLY COSTS. Re-measured with an unconditional half-K edit (gvmma_tile
     * walks half its k-block range, so every walk streams half its weights), two real packets,
     * md5-gated. With step = F + W: W = 2*(step-half), F = 2*half - step.
     *
     *     B    ctx     step     half       W        F
     *     1    128    10.765   6.683   8.164   2.601
     *     32   128    13.690   8.932   9.516   4.174
     *     32   8192   19.036  14.250   9.572   9.464
     *
     * Against ~22.3 GB of layer weights and the H100's 3.35 TB/s:
     *   B=1  : 22.3/8.164 = 2.73 TB/s = 81.5% of roofline -- the walk is NOT the 67% case. That
     *          earlier figure divided total weight bytes by the WHOLE step and was an artifact.
     *   B=32 : 22.3/9.516 = 2.34 TB/s = 70%.
     *
     * So batching costs this walk +1.35 ms on a BYTE-IDENTICAL weight stream (gvmma_partition
     * depends on N and nblk, not on rows). The retracted claim that the batching cost is not in
     * this walk is therefore wrong in its own terms: about 1.35 ms of it is right here, and
     * reaching 90% of roofline at B=32 would return ~2.1 ms of the 19.036 ms step.
     *
     * At B=32 ctx 8192 the step is almost exactly half walk (9.572) and half everything-else
     * (9.464 = attention/KV traversal, norms, sampling, entry). F grows +5.29 ms from ctx 128 to
     * 8192, which is the KV traversal, and carries 2.601 ms of fixed cost already present at
     * B=1 ctx 128.
     *
     * AND IT IS ACTIVATION-BOUND AFTER ALL. Both withdrawn probes were re-run with
     * UNCONDITIONAL edits, each gated on its cubin md5 differing from the control:
     *
     *     probe                 B=32 ctx128        B=32 ctx8192       B=1 ctx128
     *     a1 = a0 (activations) 13.689 -> 12.548   19.032 -> 17.803   10.762 -> 10.489
     *                           -1.141 (-8.34%)    -1.229 (-6.46%)    -0.273 (-2.54%)
     *     half mma issue        13.689 -> 12.676   19.032 -> 17.966   10.762 -> 10.022
     *                           -1.013 (-7.40%)    -1.066 (-5.60%)    -0.740 (-6.88%)
     *
     * The withdrawn versions of these same probes reported +0.01% and +0.09%. Subtracting the
     * B=1 effect isolates each term's share of the +1.35 ms that batching adds to this walk:
     * activations 1.141 - 0.273 = 0.868 ms, mma issue 1.013 - 0.740 = 0.273 ms. So the walk IS
     * activation-bound at B=32, and PAIR=4 -- which the invalid probe said not to build -- is the
     * indicated change: at MT=2 each activation pair currently feeds NW=2 weight row blocks, and
     * NW=4 would feed four, halving activation traffic per weight byte exactly as a1=a0 does.
     *
     * TWO CAVEATS before spending a build on it. (1) Both probe objects came out REG:247
     * STACK:160 against the control's REG:255 STACK:192, so part of the gain may be lower
     * register/stack pressure rather than the removed work; occupancy is unchanged (both fit one
     * 256-thread block per SM) and LOCAL:0 in all three, so the confound should be small, but it
     * is not zero. (2) A probe that DELETES work is an upper bound on what a legitimate
     * rearrangement can recover. PAIR=4 is not free: wv[NW][UNB] at NW=4 needs UNB halved to 4 to
     * hold the same prefetch register budget, which trades activation traffic against prefetch
     * depth in k -- the very depth the per-MT fix above was tuned for. Measure it; do not assume
     * the 0.868 ms.
     *
     * What stands independently, being real source changes on genuinely different packets: the
     * per-MT prefetch depth above (-3.64% at B=32); and from PACKLOG, the served step matches the
     * isolated one (19.100 vs 19.036 ms) at 0.268-0.269 ms/row over a 10.78 ms fixed pass.
     *
     * DO NOT re-tune this with scripts/build_sm90a_cubin.sh. That path never defines
     * PLOW_NV_GEMV_MMA_PAIR or _B1 (decode object SHARED:14480), while every packet sets both to
     * 1 from manifest.rs (SHARED:40464). With PAIR=1 the plain GEMV walks two ROW BLOCKS as NW=2,
     * and the prefetch buffer is wv[NW][UNB] — so loads in flight are NW*UNB and the generic
     * object runs half the depth of the shipped one. Its ladder (12/6 16.883, 16/8 16.423,
     * 24/12 16.648 at B=32) happens to rank these two arms the same way, but it overstated the
     * gap and is not the same kernel. Build the A/B from the packet, or with
     * PLOW_CUBIN_CONFIG=<packet>/assets/plow_config.h, and gate on REG/STACK/SHARED matching the
     * shipped object before believing a number.
     */
    constexpr unsigned UNB_MT = MT == 1   ? PLOW_NV_GEMV_MMA_UNB
                                : MT == 2 ? (PLOW_NV_GEMV_MMA_UNB * 2u) / 3u
                                          : PLOW_NV_GEMV_MMA_UNB / MT;
    constexpr unsigned UNB = UNB_MT < 2u ? 2u : UNB_MT;
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

/* REMAINDER SPLIT-K, shared by the three walks' tails. After the whole waves of one row block per
 * warp, rem (1..7) row blocks remain; up to 4 of them are cut into S K-spans with rem * S <= 8
 * units, one per warp, so the last wave keeps every warp streaming instead of 1..4 of 8. Parts > 0
 * park their fragments in the 7 reduction slots and part 0 sums them in order (bit-stable).
 * 27B B1 on 132 blocks: GemvGlu's 17 row blocks per block ran a third wave on one warp (72% of
 * HBM), qkv's 10 a second on two (53%), the fused k/v's 2 one wave on two. step_bench ctx 1K
 * B1/8/16 23.75/27.47/31.50 -> 22.94/26.96/31.20 ms (GEMV and GLU tails 0.47 ms each at B1),
 * decode object stack 240 -> 256. A tile under `if (live)` spilled it to 480..1152 instead. */
__device__ __forceinline__ unsigned gvmma_rem_split(unsigned rem, unsigned nkb) {
    unsigned S = rem == 1u ? 8u : rem == 2u ? 4u : rem <= 4u ? 2u : 1u;
    while (S > 1u && (nkb % S) != 0u) S >>= 1;
    return S;
}

/* Two adjacent bf16 of one C row as a 4-byte store: n is even, so the pair is aligned whenever
 * the row start m*N is. An odd N (a 156951-row vocab head) leaves odd rows at odd elements. */
__device__ __forceinline__ void gvmma_store2(__nv_bfloat16* C, unsigned N, unsigned m, unsigned n,
                                             float c0, float c1) {
    if (n + 1u < N) {
        const size_t o = (size_t)m * N + n;
        if ((o & 1u) == 0u) {
            *(__nv_bfloat162*)(C + o) = __floats2bfloat162_rn(c0, c1);
        } else {
            C[o] = __float2bfloat16(c0);
            C[o + 1u] = __float2bfloat16(c1);
        }
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

#if PLOW_NV_NORM_FOLD
/* Fold producer store: N % 8 == 0, so the lane's column pair is always in range. */
template <int MT>
__device__ __forceinline__ void gvmma_fold_store_tile(__nv_bfloat16* __restrict__ C, const float (&acc)[MT][4],
                                                      unsigned nb, unsigned rows, unsigned N,
                                                      const GvFoldOut& fo) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned n = nb + 2u * t;
    const float2 gm = __bfloat1622float2(*(const __nv_bfloat162*)(fo.gamma + n));
    const float g0 = norm_weight(gm.x), g1 = norm_weight(gm.y);
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned m = 16u * mt + 8u * h + g;
            float ss = 0.0f;
            if (m < rows) {
                const size_t o = (size_t)m * N + n;
                const float2 r = __bfloat1622float2(*(const __nv_bfloat162*)(fo.resid + o));
                const float f0 = r.x + __bfloat162float(__float2bfloat16(acc[mt][2 * h]));
                const float f1 = r.y + __bfloat162float(__float2bfloat16(acc[mt][2 * h + 1]));
                *(__nv_bfloat162*)(fo.resid + o) = __floats2bfloat162_rn(f0, f1);
                *(__nv_bfloat162*)(C + o) = __floats2bfloat162_rn(f0 * g0, f1 * g1);
                ss = fmaf(f0, f0, f1 * f1);
            }
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            if (t == 0u && m < rows) fo.part[(nb >> 3) * 32u + m] = ss;
        }
}

__device__ __forceinline__ float* gvmma_fold_inv_smem() {
    __shared__ float inv[32];
    return inv;
}
/* Fold consumer: inv[m] for m < rows from the producer's K / 8 partials (K float4s, K <= 2048),
 * summed in a fixed order (lane l's float4 covers rows 4(l & 7) .. +3 of row blocks
 * l / 8 + 4w + 32i). Block-uniform. */
__device__ __forceinline__ void gvmma_fold_inv(float* inv, const GvFoldIn& fi, unsigned rows, unsigned K) {
    __shared__ float4 s[PLOW_NV_WARPS][8];
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK, warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const float4* p = (const float4*)fi.part;
    float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (unsigned i = 0; i < 8u; i++) {
        const unsigned k = threadIdx.x + i * PLOW_NV_THREADS;
        if (k < K) {
            const float4 v = __ldcg(p + k);
            a.x += v.x;
            a.y += v.y;
            a.z += v.z;
            a.w += v.w;
        }
    }
#pragma unroll
    for (unsigned o = 8u; o < 32u; o <<= 1) {
        a.x += __shfl_xor_sync(0xffffffffu, a.x, o);
        a.y += __shfl_xor_sync(0xffffffffu, a.y, o);
        a.z += __shfl_xor_sync(0xffffffffu, a.z, o);
        a.w += __shfl_xor_sync(0xffffffffu, a.w, o);
    }
    if (lane < 8u) s[warp][lane] = a;
    __syncthreads();
    if (threadIdx.x < rows) {
        float r = 0.0f;
#pragma unroll
        for (unsigned w = 0; w < PLOW_NV_WARPS; w++) r += ((const float*)&s[w][threadIdx.x >> 2])[threadIdx.x & 3u];
        inv[threadIdx.x] = rsqrtf(r * __fdividef(1.0f, (float)K) + fi.eps);
    }
    __syncthreads();
}
/* acc *= inv of each fragment row (rows past `rows` are never stored). */
template <int NW, int MT>
__device__ __forceinline__ void gvmma_fold_scale(float (&acc)[NW][MT][4]) {
    const float* inv = gvmma_fold_inv_smem();
    const unsigned g = (threadIdx.x & PLOW_NV_LANE_MASK) >> 2;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const float s0 = inv[(16u * mt + g) & 31u], s1 = inv[(16u * mt + g + 8u) & 31u];
#pragma unroll
        for (int i = 0; i < NW; i++) {
            acc[i][mt][0] *= s0;
            acc[i][mt][1] *= s0;
            acc[i][mt][2] *= s1;
            acc[i][mt][3] *= s1;
        }
    }
}
#endif

#if PLOW_NV_GEMV_XK
/* Fragments of every unit's warps; a group's counter is reset by its last block. */
__device__ float gvmma_xk_part[256u * PLOW_NV_WARPS * 32u * 8u];
__device__ unsigned gvmma_xk_cnt[256];
__device__ __forceinline__ unsigned gvmma_xk_sb(unsigned rows, unsigned N, unsigned K, unsigned nblk) {
    if (rows < PLOW_NV_GEMV_XK || (N & 63u) || (K & 31u)) return 0u;
    const unsigned ng = N >> 6, nkb = K >> 5;
    unsigned sb = 8u;
    while (sb > 1u && (ng * sb > nblk || (nkb % sb) != 0u)) sb >>= 1;
    return sb > 1u && ng * sb <= 256u ? sb : 0u;
}
/* FOLD: the fold producer's epilogue; FOLD_IN: the fold consumer's inv scale. W / C / N are the
 * matrix holding the unit's rows, n0 its first row in the concatenated range the units walk. */
template <bool BIAS, int MT, bool ONE, bool FOLD, bool FOLD_IN = false>
__device__ __forceinline__ void gvmma_xk_unit(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ W, unsigned rows, unsigned N,
                                              unsigned K, unsigned slice, unsigned sb,
                                              const __nv_bfloat16* __restrict__ bias, const GvFoldOut* fo,
                                              const GvFoldIn* fi = nullptr, unsigned n0 = 0u) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK, warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned g = slice / sb, s = slice % sb, rb = g * PLOW_NV_WARPS + warp, span = (K >> 5) / sb;
    const __nv_bfloat16* const W1[1] = {W};
    float acc[1][MT][4];
    gvmma_tile<1, MT, ONE>(acc, x, W1, (rb << 3) - n0, rows, N, K, s * span, (s + 1u) * span);
    constexpr unsigned F = 32u * MT * 4u;
    float* P = gvmma_xk_part + ((size_t)(g * sb + s) * PLOW_NV_WARPS + warp) * F;
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int j = 0; j < 4; j++) P[(mt * 4 + j) * 32u + lane] = acc[0][mt][j];
    /* The interpreter's signal pattern (fa_decode_rg_item's merge fold). */
    __shared__ unsigned last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        last = atomicAdd(gvmma_xk_cnt + g, 1u) == sb - 1u;
        __threadfence();
    }
    __syncthreads();
    if (!last) return;
    float sum[1][MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int j = 0; j < 4; j++) sum[0][mt][j] = 0.0f;
#pragma unroll
    for (unsigned q = 0; q < 8u; q++) {
        if (q < sb) {
            const float* Q = gvmma_xk_part + ((size_t)(g * sb + q) * PLOW_NV_WARPS + warp) * F;
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int j = 0; j < 4; j++) sum[0][mt][j] += __ldcg(Q + (mt * 4 + j) * 32u + lane);
        }
    }
#if PLOW_NV_NORM_FOLD
    if constexpr (FOLD_IN) {
        gvmma_fold_inv(gvmma_fold_inv_smem(), *fi, rows, K);
        gvmma_fold_scale(sum);
    }
    if constexpr (FOLD) gvmma_fold_store_tile<MT>(C, sum[0], (rb << 3) - n0, rows, N, *fo);
    else
#endif
    gvmma_store_tile<BIAS, MT>(C, sum[0], (rb << 3) - n0, rows, N, bias);
    if (threadIdx.x == 0) atomicExch(gvmma_xk_cnt + g, 0u);
}
#endif
template <bool BIAS, int MT = 1, bool ONE = false, bool FOLD = false>
__device__ __forceinline__ void gemv_rows_mma(__nv_bfloat16* __restrict__ C,
                                              const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ W, unsigned rows,
                                              unsigned N, unsigned K, unsigned slice,
                                              unsigned nblk,
                                              const __nv_bfloat16* __restrict__ bias = nullptr,
                                              const GvFoldOut* fo = nullptr) {
    static_assert(!FOLD || (!PLOW_NV_GEMV_MMA_PAIR && !PLOW_NV_GEMV_REM_SPLIT),
                  "the norm fold takes the split-K and plain walks only");
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const gvmma_range r = gvmma_partition(N, slice, nblk);
    const __nv_bfloat16* const W1[1] = {W};
    const unsigned per = r.rb1 - r.rb0;
    const unsigned nkb = K >> 5;
#if PLOW_NV_GEMV_XK
    if constexpr (MT <= 2) {
        if (const unsigned sb = gvmma_xk_sb(rows, N, K, nblk)) {
            if (slice < (N >> 6) * sb) gvmma_xk_unit<BIAS, MT, ONE, FOLD>(C, x, W, rows, N, K, slice, sb, bias, fo);
            return;
        }
    }
#endif
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
#if PLOW_NV_NORM_FOLD
            if constexpr (FOLD) gvmma_fold_store_tile<MT>(C, acc[0], rb << 3, rows, N, *fo);
            else
#endif
            gvmma_store_tile<BIAS, MT>(C, acc[0], rb << 3, rows, N, bias);
        }
        __syncthreads(); /* red is reused by the next call on this block */
        return;
    }
    /* Waves of one row block per warp; a short last wave splits its row blocks over K. MT 1 only:
     * the two-tile instantiation held across the barriers spilled the decode object. */
#if PLOW_NV_GEMV_REM_SPLIT
    if constexpr (MT == 1) {
    for (unsigned base = 0u; base < per; base += PLOW_NV_WARPS) {
        const unsigned left = per - base;
        const unsigned S = left >= PLOW_NV_WARPS ? 1u : gvmma_rem_split(left, nkb);
        const unsigned grp = warp / S, part = warp % S, span = nkb / S;
        const bool live = grp < left;
        /* Dead warps walk the wave's first row block and store nothing: an unconditional
         * tile keeps the arm's registers where the split-K arms' are. */
        const unsigned rb = r.rb0 + base + (live ? grp : 0u);
        float acc[1][MT][4];
        gvmma_tile<1, MT, ONE>(acc, x, W1, rb << 3, rows, N, K, part * span, (part + 1u) * span);
        if (S > 1u) {
            gvmma_red_t<MT>& red = gvmma_red<MT>();
            if (part != 0u) {
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
#pragma unroll
                    for (int j = 0; j < 4; j++) red.v[warp - 1u - grp][lane][mt * 4 + j] = acc[0][mt][j];
            }
            __syncthreads();
            if (live && part == 0u) {
                for (unsigned p = 1u; p < S; p++) {
                    const unsigned slot = (grp * S + p) - 1u - grp;
#pragma unroll
                    for (int mt = 0; mt < MT; mt++)
#pragma unroll
                        for (int j = 0; j < 4; j++) acc[0][mt][j] += red.v[slot][lane][mt * 4 + j];
                }
            }
            __syncthreads(); /* red is reused by the next wave / call on this block */
        }
        if (live && part == 0u) gvmma_store_tile<BIAS, MT>(C, acc[0], rb << 3, rows, N, bias);
    }
    return;
    }
#endif
    for (unsigned rb = r.rb0 + warp; rb < r.rb1; rb += PLOW_NV_WARPS) {
        float acc[1][MT][4];
        gvmma_tile<1, MT, ONE>(acc, x, W1, rb << 3, rows, N, K);
#if PLOW_NV_NORM_FOLD
        if constexpr (FOLD) gvmma_fold_store_tile<MT>(C, acc[0], rb << 3, rows, N, *fo);
        else
#endif
        gvmma_store_tile<BIAS, MT>(C, acc[0], rb << 3, rows, N, bias);
    }
}

template <int MT = 1, bool ONE = false, bool FOLD = false>
__device__ __forceinline__ void gemv_glu_rows_mma(__nv_bfloat16* __restrict__ C,
                                                  const __nv_bfloat16* __restrict__ x,
                                                  const __nv_bfloat16* __restrict__ Wg,
                                                  const __nv_bfloat16* __restrict__ Wu,
                                                  unsigned rows, unsigned N, unsigned K,
                                                  unsigned act, unsigned slice, unsigned nblk,
                                                  const GvFoldIn* fi = nullptr) {
    static_assert(!FOLD || !PLOW_NV_GEMV_REM_SPLIT, "the norm fold takes the split-K and plain walks only");
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
#if PLOW_NV_NORM_FOLD
        if constexpr (FOLD) {
            gvmma_fold_inv(gvmma_fold_inv_smem(), *fi, rows, K);
            if (part == 0u && live) gvmma_fold_scale(acc);
        }
#endif
        if (part == 0u && live) store(acc, rb);
        return;
    }
#endif
    /* Waves of one row block per warp; a short last wave splits its row blocks over K, gate and
     * up reducing one after the other through the one-stream slots. MT 1 only (see gemv_rows_mma). */
#if PLOW_NV_GEMV_REM_SPLIT
    if constexpr (MT == 1) {
    const unsigned per_g = r.rb1 - r.rb0, nkb_g = K >> 5;
    for (unsigned base = 0u; base < per_g; base += PLOW_NV_WARPS) {
        const unsigned left = per_g - base;
        const unsigned S = left >= PLOW_NV_WARPS ? 1u : gvmma_rem_split(left, nkb_g);
        const unsigned grp = warp / S, part = warp % S, span = nkb_g / S;
        const bool live = grp < left;
        /* Dead warps walk the wave's first row block and store nothing: an unconditional
         * tile keeps the arm's registers where the split-K arms' are. */
        const unsigned rb = r.rb0 + base + (live ? grp : 0u);
        float acc[2][MT][4];
        gvmma_tile<2, MT, ONE>(acc, x, W2, rb << 3, rows, N, K, part * span, (part + 1u) * span);
        if (S > 1u) {
            gvmma_red_t<MT>& red = gvmma_red<MT>();
#pragma unroll
            for (int i = 0; i < 2; i++) {
                if (part != 0u) {
#pragma unroll
                    for (int mt = 0; mt < MT; mt++)
#pragma unroll
                        for (int j = 0; j < 4; j++) red.v[warp - 1u - grp][lane][mt * 4 + j] = acc[i][mt][j];
                }
                __syncthreads();
                if (live && part == 0u) {
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
        }
        if (live && part == 0u) store(acc, rb);
    }
    return;
    }
#endif
#if PLOW_NV_NORM_FOLD
    if constexpr (FOLD) {
        /* One wave: inv lands between the walk and the stores. A wider N computes it first. */
        if (r.rb1 - r.rb0 <= PLOW_NV_WARPS) {
            const unsigned rb = r.rb0 + warp;
            float acc[2][MT][4];
            if (rb < r.rb1) gvmma_tile<2, MT, ONE>(acc, x, W2, rb << 3, rows, N, K);
            gvmma_fold_inv(gvmma_fold_inv_smem(), *fi, rows, K);
            if (rb < r.rb1) {
                gvmma_fold_scale(acc);
                store(acc, rb);
            }
            return;
        }
        gvmma_fold_inv(gvmma_fold_inv_smem(), *fi, rows, K);
    }
#endif
    for (unsigned rb = r.rb0 + warp; rb < r.rb1; rb += PLOW_NV_WARPS) {
        float acc[2][MT][4];
        gvmma_tile<2, MT, ONE>(acc, x, W2, rb << 3, rows, N, K);
#if PLOW_NV_NORM_FOLD
        if constexpr (FOLD) gvmma_fold_scale(acc);
#endif
        store(acc, rb);
    }
}

/* Fused q|k|v: row blocks over the concatenated [0, Nq+Nk+Nv); Nq % 8 == Nk % 8 == 0 (caller
 * checks) so a block lies inside one matrix. */
template <bool BIAS, int MT = 1, bool ONE = false, bool FOLD = false>
__device__ __forceinline__ void gemv_qkv_rows_mma(
    __nv_bfloat16* Cq, __nv_bfloat16* Ck, __nv_bfloat16* Cv, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ Wq, const __nv_bfloat16* __restrict__ Wk,
    const __nv_bfloat16* __restrict__ Wv, unsigned rows, unsigned Nq, unsigned Nk, unsigned Nv,
    unsigned K, unsigned slice, unsigned nblk, const __nv_bfloat16* __restrict__ bq = nullptr,
    const __nv_bfloat16* __restrict__ bk = nullptr, const __nv_bfloat16* __restrict__ bv = nullptr,
    const GvFoldIn* fi = nullptr) {
    static_assert(!FOLD || !PLOW_NV_GEMV_REM_SPLIT, "the norm fold takes the plain walk only");
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned g = lane >> 2, t = lane & 3;
#if PLOW_NV_GEMV_XK && PLOW_NV_GEMV_XK_QKV
    /* Cross-block split-K over the concatenated rows; a 64-row unit lies inside one matrix. */
    if constexpr (MT <= 2) {
        if (const unsigned sb = ((Nq | Nk) & 63u) ? 0u : gvmma_xk_sb(rows, Nq + Nk + Nv, K, nblk)) {
            if (slice < ((Nq + Nk + Nv) >> 6) * sb) {
                const unsigned gn = (slice / sb) << 6;
                const bool iq = gn < Nq, ik = !iq && gn < Nq + Nk;
                gvmma_xk_unit<BIAS, MT, ONE, false, FOLD>(iq ? Cq : ik ? Ck : Cv, x, iq ? Wq : ik ? Wk : Wv, rows,
                                                          iq ? Nq : ik ? Nk : Nv, K, slice, sb,
                                                          iq ? bq : ik ? bk : bv, nullptr, fi,
                                                          iq ? 0u : ik ? Nq : Nq + Nk);
            }
            return;
        }
    }
#endif
    const gvmma_range r = gvmma_partition(Nq + Nk + Nv, slice, nblk);
    /* Waves of one row block per warp; a short last wave splits its row blocks over K. MT 1 only
     * (see gemv_rows_mma). */
#if PLOW_NV_GEMV_REM_SPLIT
    if constexpr (MT == 1) {
    const unsigned per = r.rb1 - r.rb0, nkb = K >> 5;
    for (unsigned base = 0u; base < per; base += PLOW_NV_WARPS) {
        const unsigned left = per - base;
        const unsigned S = left >= PLOW_NV_WARPS ? 1u : gvmma_rem_split(left, nkb);
        const unsigned grp = warp / S, part = warp % S, span = nkb / S;
        const bool live = grp < left;
        /* Dead warps walk the wave's first row block and store nothing: an unconditional
         * tile keeps the arm's registers where the split-K arms' are. */
        const unsigned rb = r.rb0 + base + (live ? grp : 0u);
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
        gvmma_tile<1, MT, ONE>(acc, x, W1, gn - n0, rows, Nx, K, part * span, (part + 1u) * span);
        if (S > 1u) {
            gvmma_red_t<MT>& red = gvmma_red<MT>();
            if (part != 0u) {
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
#pragma unroll
                    for (int j = 0; j < 4; j++) red.v[warp - 1u - grp][lane][mt * 4 + j] = acc[0][mt][j];
            }
            __syncthreads();
            if (live && part == 0u) {
                for (unsigned p = 1u; p < S; p++) {
                    const unsigned slot = (grp * S + p) - 1u - grp;
#pragma unroll
                    for (int mt = 0; mt < MT; mt++)
#pragma unroll
                        for (int j = 0; j < 4; j++) acc[0][mt][j] += red.v[slot][lane][mt * 4 + j];
                }
            }
            __syncthreads(); /* red is reused by the next wave / call on this block */
        }
        if (!(live && part == 0u)) continue;
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
    return;
    }
#endif
#if PLOW_NV_NORM_FOLD
    if constexpr (FOLD) {
        /* One wave: inv lands between the walk and the stores. A wider N computes it first. */
        if (r.rb1 - r.rb0 <= PLOW_NV_WARPS) {
            const unsigned rb = r.rb0 + warp;
            const bool live = rb < r.rb1;
            const unsigned gn = (live ? rb : r.rb0) << 3;
            const __nv_bfloat16* W;
            __nv_bfloat16* C;
            const __nv_bfloat16* bias;
            unsigned Nx, n0;
            if (gn < Nq) { W = Wq; C = Cq; Nx = Nq; n0 = 0u; bias = bq; }
            else if (gn < Nq + Nk) { W = Wk; C = Ck; Nx = Nk; n0 = Nq; bias = bk; }
            else { W = Wv; C = Cv; Nx = Nv; n0 = Nq + Nk; bias = bv; }
            const __nv_bfloat16* const W1[1] = {W};
            float acc[1][MT][4];
            if (live) gvmma_tile<1, MT, ONE>(acc, x, W1, gn - n0, rows, Nx, K);
            gvmma_fold_inv(gvmma_fold_inv_smem(), *fi, rows, K);
            if (!live) return;
            gvmma_fold_scale(acc);
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
            return;
        }
        gvmma_fold_inv(gvmma_fold_inv_smem(), *fi, rows, K);
    }
#endif
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
#if PLOW_NV_NORM_FOLD
        if constexpr (FOLD) gvmma_fold_scale(acc);
#endif
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
#if PLOW_NV_GEMV_XK
/* The cross-block unit's 64 weight rows at its K span; a fold producer's also the group's residual
 * columns and gamma (any of the group's blocks may run the epilogue). */
__device__ __forceinline__ void gvmma_pf_xk(const __nv_bfloat16* W, const __nv_bfloat16* resid,
                                            const __nv_bfloat16* gamma, unsigned rows, unsigned N,
                                            unsigned K, unsigned slice, unsigned sb, unsigned n0 = 0u) {
    const unsigned g = slice / sb, s = slice % sb, kspan = K / sb;
    unsigned bytes = PLOW_NV_GEMV_PF_BYTES / 64u;
    if (bytes > kspan * 2u) bytes = kspan * 2u;
    bytes &= ~127u;
    if (W && bytes && threadIdx.x < 64u) gvmma_pf_l2(W + (size_t)(g * 64u + threadIdx.x - n0) * K + s * kspan, bytes);
    if (resid) {
        const unsigned m = blockDim.x - 1u - threadIdx.x;
        if (m < rows) gvmma_pf_l2(resid + (size_t)m * N + g * 64u, 128u);
        else if (m == rows) gvmma_pf_l2(gamma + g * 64u, 128u);
    }
}
#endif
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
#if PLOW_NV_NORM_FOLD
/* Fold producer: the residual row segments and gamma its epilogue reads (upper threads, clear of
 * the weight heads). */
__device__ __forceinline__ void gvmma_pf_fold(const __nv_bfloat16* resid, const __nv_bfloat16* gamma,
                                              unsigned rows, unsigned N, unsigned slice, unsigned nblk) {
    const gvmma_range r = gvmma_partition(N, slice, nblk);
    if (r.rb1 <= r.rb0) return;
    const unsigned bytes = (r.rb1 - r.rb0) * 16u, m = blockDim.x - 1u - threadIdx.x;
    if (m < rows) gvmma_pf_l2(resid + (size_t)m * N + (r.rb0 << 3), bytes);
    else if (m == rows) gvmma_pf_l2(gamma + (r.rb0 << 3), bytes);
}
#endif
#endif

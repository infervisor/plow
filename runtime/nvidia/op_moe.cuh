/* op_moe.cuh — block-fp8 MoE expert GEMV core for sm_120a (RTX 5090 / GB202).
 *
 * Port of the AMD reference `runtime/amd/op_moe.h` (wave_dot_fp8_blk + the grouped
 * MOE_GROUP_{GLU,DOWN}_FP8_BLK bodies) to NVIDIA warp32.
 *
 * ============================ SCALE LAYOUT (read this) ============================
 * Two DIFFERENT fp8 schemes live in this repo; this file implements the BLOCK one.
 *
 *   per-output-channel (GemvFp8/GemvGluFp8, dense Gemma path): wscale is f32[N], ONE
 *     multiplier per output row, applied ONCE in the epilogue AFTER the cross-lane
 *     reduce (`runtime/amd/op_gemm.h:1439`: wave_sum(acc[m]) * wscale[n]).
 *
 *   block-fp8 (this file; DeepSeek/GLM `weight_block_size=[128,128]`, the tensor is
 *     HF `*_weight_scale_inv`): wscale is a 2-D f32 grid [ceil(N/128)][ceil(K/128)],
 *     ROW-MAJOR, with KB = ceil(K/128) the row pitch. The dequantized weight is
 *
 *         W_f32[n][k] = e4m3_to_f32(Wq[n][k]) * S[(n>>7)*KB + (k>>7)]
 *
 *     The scale therefore VARIES ALONG K and does NOT factor out of the K-reduction.
 *     It must be applied per 128-K block, INSIDE the accumulation:
 *
 *         acc = sum_over_kblocks( ( sum_{k in block} w_q[k]*x[k] ) * S[nrow + kb] )
 *
 *     Hoisting it to the epilogue (the per-channel idiom) is the #1 silent-wrong-answer
 *     trap here — it silently produces a plausible-looking but wrong vector.
 *
 * The lane tiling is chosen so a lane's 16 contiguous fp8 always lie inside ONE 128-K
 * block (16 | 128), hence exactly one scale load per lane per chunk. This invariant is
 * what makes the "multiply the chunk partial by bs" form legal, and it holds for warp32
 * exactly as it did for wave64 — only the chunk STRIDE changes (32*16=512 vs 64*16=1024).
 *
 * ============================== WARP32 RE-DERIVATION ==============================
 * The AMD reduce (`amd_common.h` wave_sum) is a 6-step DPP/permlane butterfly over 64
 * lanes. Transliterating it is wrong here. warp_sum below is a 5-step __shfl_xor_sync
 * butterfly over 32 lanes with a full 0xffffffff mask -> all-reduce (every lane holds
 * the total), matching wave_sum's all-reduce semantics.
 */
#ifndef PLOW_NV_OP_MOE_CUH
#define PLOW_NV_OP_MOE_CUH

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef PLOW_NV_BF16_T
#define PLOW_NV_BF16_T
typedef __nv_bfloat16 bf16;
#endif

#define PLOW_NV_WARP 32u
/* K elements consumed by one warp pass: 32 lanes x 16 fp8 = 512 = 4 blocks of 128. */
#define PLOW_NV_FP8_STEP (PLOW_NV_WARP * 16u)

/* --- warp32 all-reduce sum (re-derivation of amd_common.h wave_sum for 32 lanes) --- */
__device__ __forceinline__ float plow_warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off, 32);
    return v;
}

/* e4m3 x2 -> half2 (hardware cvt, sm_89+). Exact: every e4m3 value is representable in fp16. */
__device__ __forceinline__ float2 plow_e4m3x2_to_f32x2(unsigned short packed) {
    __half2 h;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(*(unsigned*)&h) : "h"(packed));
    return __half22float2(h);
}

/* --- the core: block-fp8 dot of ONE output channel, reduced across a 32-lane warp -----
 * x    : bf16[K] activation (row already offset by the caller)
 * Wrow : e4m3[K] one output channel's quantized weights (contiguous, K-major)
 * srow  : this channel's block-scale ROW = S + (n>>7)*KB, i.e. KB f32 entries
 * Returns the f32 dot on EVERY lane (all-reduce).
 * Requires K % 16 == 0 for the vector loads; the kb clamp keeps overshoot lanes in
 * bounds when K is not a multiple of PLOW_NV_FP8_STEP (their partial is zeroed).      */
__device__ __forceinline__ float plow_warp_dot_fp8_blk(const bf16* __restrict__ x,
                                                       const unsigned char* __restrict__ Wrow,
                                                       const float* __restrict__ srow, unsigned K,
                                                       unsigned lane) {
    const unsigned nchunk = (K + PLOW_NV_FP8_STEP - 1u) / PLOW_NV_FP8_STEP;
    const unsigned KB = (K + 127u) >> 7;
    float acc = 0.0f;
    for (unsigned c = 0; c < nchunk; c++) {
        const unsigned k = c * PLOW_NV_FP8_STEP + lane * 16u;
        const bool live = (k < K);
        unsigned kb = k >> 7;
        if (kb >= KB) kb = KB - 1u;      /* overshoot lane: keep the scale read in bounds */
        const unsigned kx = live ? k : 0u;

        /* 16 e4m3 = one 16B vector load; 16 bf16 = two 16B vector loads. The weight row is
         * read exactly once per step -> stream it evict-first so it cannot displace the x row
         * (re-read by every warp of the slot) from L1/L2. */
        const uint4 wv = __ldcs((const uint4*)(Wrow + kx));
        const uint4 xa = *(const uint4*)(x + kx);
        const uint4 xb = *(const uint4*)(x + kx + 8);

        const unsigned wq[4] = {wv.x, wv.y, wv.z, wv.w};
        const unsigned xr[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};

        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) { /* 4 words x 4 fp8 = 16 */
            const float2 w0 = plow_e4m3x2_to_f32x2((unsigned short)(wq[i] & 0xffffu));
            const float2 w1 = plow_e4m3x2_to_f32x2((unsigned short)(wq[i] >> 16));
            const float2 x0 = __bfloat1622float2(*(const __nv_bfloat162*)&xr[2 * i]);
            const float2 x1 = __bfloat1622float2(*(const __nv_bfloat162*)&xr[2 * i + 1]);
            p = fmaf(w0.x, x0.x, p);
            p = fmaf(w0.y, x0.y, p);
            p = fmaf(w1.x, x1.x, p);
            p = fmaf(w1.y, x1.y, p);
        }
#ifdef PLOW_MOE_BAD_SCALE_EPILOGUE
        /* NEGATIVE CONTROL ONLY (never define this in a shipping build): the per-channel
         * idiom wrongly transplanted to block scales — accumulate raw, scale once at the
         * end. This is the exact silent-wrong-answer trap the header warns about; the
         * validation harness must FAIL when it is enabled, or the harness proves nothing. */
        acc += live ? p : 0.0f;
    }
    return plow_warp_sum(acc) * srow[0];
#else
        /* per-128-block dequant applied to THIS chunk's partial — see header note. */
        acc += live ? p * srow[kb] : 0.0f;
    }
    return plow_warp_sum(acc);
#endif
}

/* --- FUSED gate+up twin of plow_warp_dot_fp8_blk ---------------------------------------
 * ONE pass over x with TWO independent weight streams in flight. The unfused form ran the
 * whole gate dot (including its 5-step shuffle reduce) to completion before the up dot's
 * first load could issue, and loaded + converted the x row twice. Here the two 16B weight
 * loads issue back to back every chunk (2x memory-level parallelism on this latency-bound
 * decode path) and x is converted once. Each accumulator keeps EXACTLY the unfused chunk
 * and fma order, so both dots are bit-identical to two plow_warp_dot_fp8_blk calls. */
__device__ __forceinline__ float2 plow_warp_dot2_fp8_blk(
    const bf16* __restrict__ x, const unsigned char* __restrict__ Wgrow,
    const unsigned char* __restrict__ Wurow, const float* __restrict__ sgrow,
    const float* __restrict__ surow, unsigned K, unsigned lane) {
    const unsigned nchunk = (K + PLOW_NV_FP8_STEP - 1u) / PLOW_NV_FP8_STEP;
    const unsigned KB = (K + 127u) >> 7;
    float accg = 0.0f, accu = 0.0f;
    for (unsigned c = 0; c < nchunk; c++) {
        const unsigned k = c * PLOW_NV_FP8_STEP + lane * 16u;
        const bool live = (k < K);
        unsigned kb = k >> 7;
        if (kb >= KB) kb = KB - 1u;
        const unsigned kx = live ? k : 0u;

        const uint4 wgv = __ldcs((const uint4*)(Wgrow + kx));
        const uint4 wuv = __ldcs((const uint4*)(Wurow + kx));
        const uint4 xa = *(const uint4*)(x + kx);
        const uint4 xb = *(const uint4*)(x + kx + 8);

        const unsigned wg[4] = {wgv.x, wgv.y, wgv.z, wgv.w};
        const unsigned wu[4] = {wuv.x, wuv.y, wuv.z, wuv.w};
        const unsigned xr[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};

        float pg = 0.0f, pu = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 x0 = __bfloat1622float2(*(const __nv_bfloat162*)&xr[2 * i]);
            const float2 x1 = __bfloat1622float2(*(const __nv_bfloat162*)&xr[2 * i + 1]);
            const float2 g0 = plow_e4m3x2_to_f32x2((unsigned short)(wg[i] & 0xffffu));
            const float2 g1 = plow_e4m3x2_to_f32x2((unsigned short)(wg[i] >> 16));
            const float2 u0 = plow_e4m3x2_to_f32x2((unsigned short)(wu[i] & 0xffffu));
            const float2 u1 = plow_e4m3x2_to_f32x2((unsigned short)(wu[i] >> 16));
            pg = fmaf(g0.x, x0.x, pg);
            pg = fmaf(g0.y, x0.y, pg);
            pg = fmaf(g1.x, x1.x, pg);
            pg = fmaf(g1.y, x1.y, pg);
            pu = fmaf(u0.x, x0.x, pu);
            pu = fmaf(u0.y, x0.y, pu);
            pu = fmaf(u1.x, x1.x, pu);
            pu = fmaf(u1.y, x1.y, pu);
        }
#ifdef PLOW_MOE_BAD_SCALE_EPILOGUE
        /* NEGATIVE CONTROL ONLY — same wrong per-tensor epilogue as the unfused helper, so
         * the harness still fails when the control build is enabled. */
        accg += live ? pg : 0.0f;
        accu += live ? pu : 0.0f;
    }
    return make_float2(plow_warp_sum(accg) * sgrow[0], plow_warp_sum(accu) * surow[0]);
#else
        accg += live ? pg * sgrow[kb] : 0.0f;
        accu += live ? pu * surow[kb] : 0.0f;
    }
    return make_float2(plow_warp_sum(accg), plow_warp_sum(accu));
#endif
}

/* --- routing table accessors (identical byte layout to the AMD path: 8B per slot) --- */
__device__ __forceinline__ unsigned plow_moe_slot_expert(const unsigned char* table, unsigned slot) {
    return *(const unsigned*)(table + (size_t)slot * 8);
}
__device__ __forceinline__ float plow_moe_slot_gate(const unsigned char* table, unsigned slot) {
    return *(const float*)(table + (size_t)slot * 8 + 4);
}

__device__ __forceinline__ float plow_moe_act(float g, unsigned act) {
    /* act 0 = SiLU (GLM/DeepSeek swiglu), 1 = GELU-tanh. Fast reciprocal: the IEEE `/`
     * inlined a ~90-instr FCHK+slow-path division (cold for Gemma, pure code bloat). */
    if (act == 0u) return g * __fdividef(1.0f, 1.0f + __expf(-g));
    const float c = 0.7978845608028654f;
    return 0.5f * g * (1.0f + tanhf(c * (g + 0.044715f * g * g * g)));
}

/* ================= DENSE (non-routed) block-fp8 FFN, decode megakernel ops ============
 * P1.5 bring-up: GLM/Kimi/DeepSeek dense (first_k_dense_replace) FFN. Ports the AMD
 * bodies runtime/amd/op_moe.h d_dense_glu_fp8_blk + op_gemm.h gemv_rows_fp8_blk to the
 * sm_120 in-megakernel warp-slice schedule (one WARP per output channel, exactly like
 * d_headnorm_rope / the arena d_gemv_fp8). Reuses the existing block-fp8 dot primitives
 * (plow_warp_dot2_fp8_blk / plow_warp_dot_fp8_blk), so the numerics are the validated
 * MoE ones; only the output map differs (direct weight+scale pointers, no routing table).
 * x is read straight from global (re-read per warp, resident in L1/L2) — matches AMD. */

/* DENSE_GLU_FP8_BLK (op 47): fu[n] = act(gate_n·x) * (up_n·x), n in [0,N). N=intermediate,
 * K=hidden. Wg/Wu = e4m3[N][K]; Sg/Su = block-scale grids [ceil(N/128)][ceil(K/128)] f32. */
static __device__ void d_dense_glu_fp8_blk(bf16* __restrict__ fu, const bf16* __restrict__ x,
                                           const unsigned char* __restrict__ Wg,
                                           const unsigned char* __restrict__ Wu,
                                           const float* __restrict__ Sg, const float* __restrict__ Su,
                                           unsigned N, unsigned K, unsigned act, unsigned slice,
                                           unsigned nblk) {
    const unsigned KB = (K + 127u) >> 7;
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned wstride = nblk * PLOW_NV_WARPS;
    for (unsigned n = slice * PLOW_NV_WARPS + warp; n < N; n += wstride) {
        const unsigned nrow = (n >> 7) * KB;
        const float2 gu = plow_warp_dot2_fp8_blk(x, Wg + (size_t)n * K, Wu + (size_t)n * K,
                                                 Sg + nrow, Su + nrow, K, lane);
        if (lane == 0) fu[n] = __float2bfloat16(plow_moe_act(gu.x, act) * gu.y);
    }
}

/* GEMV_FP8_BLK (op 44), dense DOWN projection: C[n] = W_n · x, n in [0,N). W = e4m3[N][K];
 * wscale = block-scale grid [ceil(N/128)][ceil(K/128)] f32. M=1 on the decode path. */
static __device__ void d_gemv_fp8_blk(bf16* __restrict__ C, const bf16* __restrict__ x,
                                      const unsigned char* __restrict__ W,
                                      const float* __restrict__ wscale, unsigned N, unsigned K,
                                      unsigned slice, unsigned nblk) {
    const unsigned KB = (K + 127u) >> 7;
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned wstride = nblk * PLOW_NV_WARPS;
    for (unsigned n = slice * PLOW_NV_WARPS + warp; n < N; n += wstride) {
        const unsigned nrow = (n >> 7) * KB;
        const float y = plow_warp_dot_fp8_blk(x, W + (size_t)n * K, wscale + nrow, K, lane);
        if (lane == 0) C[n] = __float2bfloat16(y);
    }
}

/* ================= GROUPED expert GLU (twin of MOE_GROUP_GLU_FP8_BLK, op 48) =========
 * FLAT (slot, out-channel) sweep: the grid walks a single flat index space of
 * nslot*I_moe outputs, ONE WARP PER OUTPUT. nslot = ntok*k for a multi-token dispatch.
 *
 * Why flat and not expert-parallel: with one warp per (slot,channel), every unit of work
 * is exactly one K=H dot, so the schedule is load-balanced BY CONSTRUCTION regardless of
 * how skewed the routing is. An expert-parallel assignment (block per expert) would
 * stall on the hottest expert. See the report for the measured imbalance numbers.
 *
 * wtab[eid*3 + {0,1,2}] = {Wgate, Wup, Wdown} device pointers (0 = expert not resident).
 * stab[eid*3 + {0,1,2}] = matching block-scale grid pointers.
 * `table` is nslot 8-byte {u32 expert_id, f32 gate} records; eid >= n_exp is the
 * sentinel for an unused slot (skipped, exactly as the AMD path).                      */
static __global__ void plow_moe_group_glu_fp8_blk(bf16* __restrict__ fu, const bf16* __restrict__ x,
                                           const unsigned char* __restrict__ table,
                                           const unsigned long long* __restrict__ wtab,
                                           const unsigned long long* __restrict__ stab,
                                           unsigned nslot, unsigned k, unsigned I_moe, unsigned H,
                                           unsigned n_exp, unsigned act) {
    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned nwarp = (gridDim.x * blockDim.x) >> 5;
    const unsigned KB = (H + 127u) >> 7;
    const unsigned total = nslot * I_moe;

    for (unsigned f = gwarp; f < total; f += nwarp) {
        const unsigned slot = f / I_moe;
        const unsigned n = f - slot * I_moe;
        const unsigned eid = plow_moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;                       /* sentinel slot */
        const unsigned long long wg = wtab[(size_t)eid * 3 + 0];
        if (wg == 0ull) continue;                         /* expert not resident on this rank */
        const unsigned char* Wg = (const unsigned char*)(size_t)wg;
        const unsigned char* Wu = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 1];
        const float* Sg = (const float*)(size_t)stab[(size_t)eid * 3 + 0];
        const float* Su = (const float*)(size_t)stab[(size_t)eid * 3 + 1];
        const unsigned nrow = (n >> 7) * KB;              /* this channel's block-scale row */
        const bf16* xt = x + (size_t)(slot / k) * H;      /* slot -> token */
        const float2 gu = plow_warp_dot2_fp8_blk(xt, Wg + (size_t)n * H, Wu + (size_t)n * H,
                                                 Sg + nrow, Su + nrow, H, lane);
        if (lane == 0)
            fu[(size_t)slot * I_moe + n] = __float2bfloat16(plow_moe_act(gu.x, act) * gu.y);
    }
}

/* ================ GROUPED expert DOWN (twin of MOE_GROUP_DOWN_FP8_BLK, op 49) ========
 * One warp per (slot, hidden row h). Writes the GATE-SCALED f32 partial for a fixed slot,
 * so the downstream combine is a deterministic fixed-order sum (skipped slots write 0). */
static __global__ void plow_moe_group_down_fp8_blk(float* __restrict__ part, const bf16* __restrict__ fu,
                                            const unsigned char* __restrict__ table,
                                            const unsigned long long* __restrict__ wtab,
                                            const unsigned long long* __restrict__ stab,
                                            unsigned nslot, unsigned H, unsigned I_moe,
                                            unsigned n_exp) {
    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned nwarp = (gridDim.x * blockDim.x) >> 5;
    const unsigned KB = (I_moe + 127u) >> 7;
    const unsigned total = nslot * H;

    for (unsigned f = gwarp; f < total; f += nwarp) {
        const unsigned slot = f / H;
        const unsigned h = f - slot * H;
        const unsigned eid = plow_moe_slot_expert(table, slot);
        float* part_slot = part + (size_t)slot * H;
        if (eid >= n_exp || wtab[(size_t)eid * 3 + 2] == 0ull) {
            if (lane == 0) part_slot[h] = 0.0f;           /* deterministic zero partial */
            continue;
        }
        const float gate = plow_moe_slot_gate(table, slot);
        const unsigned char* Wd = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 2];
        const float* Sd = (const float*)(size_t)stab[(size_t)eid * 3 + 2];
        const bf16* fu_slot = fu + (size_t)slot * I_moe;
        const float y = plow_warp_dot_fp8_blk(fu_slot, Wd + (size_t)h * I_moe,
                                              Sd + (size_t)(h >> 7) * KB, I_moe, lane);
        if (lane == 0) part_slot[h] = gate * y;
    }
}

/* ================== EXPERT-PARALLEL variant (for the imbalance measurement) ==========
 * One THREAD BLOCK per slot; the block's warps split that slot's I_moe channels. This is
 * the "natural" assignment and the one that degrades under skewed routing once slots
 * carry unequal work (multi-token, expert-grouped dispatch). Kept only as the A/B arm of
 * the load-imbalance experiment — the flat kernel above is the shipping form.           */
static __global__ void plow_moe_slot_glu_fp8_blk(bf16* __restrict__ fu, const bf16* __restrict__ x,
                                          const unsigned char* __restrict__ table,
                                          const unsigned long long* __restrict__ wtab,
                                          const unsigned long long* __restrict__ stab, unsigned k,
                                          unsigned I_moe, unsigned H, unsigned n_exp,
                                          unsigned act) {
    const unsigned slot = blockIdx.x;
    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned wave = threadIdx.x >> 5;
    const unsigned nw = blockDim.x >> 5;
    const unsigned eid = plow_moe_slot_expert(table, slot);
    if (eid >= n_exp) return;
    const unsigned long long wg = wtab[(size_t)eid * 3 + 0];
    if (wg == 0ull) return;
    const unsigned char* Wg = (const unsigned char*)(size_t)wg;
    const unsigned char* Wu = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 1];
    const float* Sg = (const float*)(size_t)stab[(size_t)eid * 3 + 0];
    const float* Su = (const float*)(size_t)stab[(size_t)eid * 3 + 1];
    const unsigned KB = (H + 127u) >> 7;
    const bf16* xt = x + (size_t)(slot / k) * H;
    for (unsigned n = wave; n < I_moe; n += nw) {
        const unsigned nrow = (n >> 7) * KB;
        const float2 gu = plow_warp_dot2_fp8_blk(xt, Wg + (size_t)n * H, Wu + (size_t)n * H,
                                                 Sg + nrow, Su + nrow, H, lane);
        if (lane == 0)
            fu[(size_t)slot * I_moe + n] = __float2bfloat16(plow_moe_act(gu.x, act) * gu.y);
    }
}

/* ================================================================================
 * bf16 Gemma-4 26B-A4B (sparse-MoE) DECODE bodies (plans/rtx-08-gemma4-moe-26b.md).
 *
 * These are the bf16 twins of the block-fp8 grouped kernels above, specialised to the
 * Gemma-4 MoE block: SOFTMAX router with a weightless-RMS + per-channel scale + H^-0.5
 * pre-transform and a PER-EXPERT gate scale; FUSED gate_up expert weights ([E,2*I,H]);
 * gelu_pytorch_tanh activation. Every one of these differences is a silent-wrong-answer
 * trap if dropped — see the recon note in the plan. B=1 decode: x is one [H] row.
 *
 * expert_weight_table (`ewt`) layout for Gemma: 2 u64 per expert —
 *   ewt[eid*2+0] = base of gate_up_proj[eid]  ([2*I_moe, H] row-major, gate rows [0,I),
 *                                               up rows [I,2I))
 *   ewt[eid*2+1] = base of down_proj[eid]      ([H, I_moe] row-major)
 * ================================================================================ */

/* ================= BATCHED (B>1) DECODE — work-item ordering ==============================
 * For B sequence slots the routed work is B*k slots instead of k. Two orderings are possible
 * for the flat (slot, channel) sweep, and the choice IS the batching thesis:
 *
 *   slot-major  f = s*I + n   — every warp of a slot runs together, but the B*k slots of one
 *                               channel are spread a whole sweep apart. Two slots that routed
 *                               to the SAME expert then re-fetch that expert's rows from HBM.
 *   channel-major f = n*S + s — (S = B*k) all S slots of channel n are ADJACENT in the flat
 *                               index, so they are co-resident in the same grid wave. Every
 *                               slot that routed to expert e reads W_e[n] within a few hundred
 *                               warps of the first reader: the row comes from HBM once and from
 *                               L2 thereafter (reuse distance ~S*2 rows = 2.8 MB at B=32 vs a
 *                               128 MB L2). Weight HBM bytes/step collapse from
 *                               (B*k) x 11.9 MB to (#distinct experts) x 11.9 MB.
 *
 * B>1 therefore uses CHANNEL-MAJOR. This is a pure permutation of independent one-warp outputs,
 * so it changes no arithmetic — see plans/batch-decode.md. B==1 keeps the legacy slot-major
 * order (S == k makes the two equivalent for locality, and it protects the B=1 TPOT gate).
 *
 * Chosen over an explicit expert SORT + register-level multi-row reuse (the gemv_rows<MM> form,
 * i.e. one warp loads W_e[n] once and dots it against every row routed to e): that needs an
 * extra align/scatter packet at decode scale AND MM accumulators per warp, and the megakernel is
 * register-bound at 174 regs / 1 block per SM — spilling there would cost every other op. The
 * channel-major order buys the same HBM reduction through L2 at zero register cost. Explicit
 * register reuse is the next lever if L2 miss counters ever say otherwise.                    */
#define PLOW_MOE_MAXB 32u

/* ---- ROW-BLOCKED DECODE GEMV (H100 campaign E2) -----------------------------------------
 * At the megakernel's 1 block/SM a warp that owns ONE output row keeps only UN weight loads
 * in flight, and the H100 needs ~16 to reach its bandwidth (measured: a pure read at 1 blk/SM
 * goes 1222 -> 2490 GB/s as loads-in-flight go 1 -> 8; runtime/nvidia/experiments/
 * hbm_ceiling_h100.cu). Giving a warp GV_MOE_RB rows multiplies loads-in-flight by RB at
 * constant occupancy. Default OFF so every sm_120 object stays byte-identical; the sm_90a
 * build sets -DPLOW_NV_GEMV_RB=1. See perf-data/gemma26b-h100-gemv-mlp.md. */
#ifndef PLOW_NV_GEMV_RB
#define PLOW_NV_GEMV_RB 0
#endif
#ifndef GV_MOE_RB
#define GV_MOE_RB 2 /* output channels per warp; 2*RB weight streams in the GLU arm */
#endif
#ifndef GV_MOE_RB_DN
/* DOWN arm streams. 4 measured 18 us faster on bf16 but 16 us slower on fp8 and pushes the
 * megakernel REG 177 -> 229; 2 is the wash that keeps the register headroom. */
#define GV_MOE_RB_DN 2
#endif
/* Staging fu in the arena MEASURED SLOWER (7.183 vs 7.060 ms): the extra __syncthreads on
 * every MoE-down op (30 per token) costs more than the redundant fu reads it removes, which
 * were L1 hits anyway (fu is 11 KiB). Kept behind the flag as a recorded negative. */
/* Lane-split DOWN. Default OFF so sm_120 objects stay byte-identical; the sm_90a build turns
 * it on. MEASURED at 1 block/SM: bf16 7.060 -> 6.766 ms (fp8 neutral -- it runs the fp8 down
 * arm). At 2 blocks/SM: 6.387 -> 6.106. */
#ifndef PLOW_MOE_XN_BF16
#define PLOW_MOE_XN_BF16 0
#endif
#ifndef PLOW_MOE_ROUTER_WIDE
#define PLOW_MOE_ROUTER_WIDE 0
#endif
#ifndef PLOW_MOE_COMBINE_ALLBLK
#define PLOW_MOE_COMBINE_ALLBLK 0
#endif
#ifndef PLOW_MOE_DOWN_LANESPLIT
#define PLOW_MOE_DOWN_LANESPLIT 0
#endif
#ifndef PLOW_MOE_DOWN_SG
#define PLOW_MOE_DOWN_SG 4u
#endif
#ifndef PLOW_MOE_DOWN_STAGE_FU
#define PLOW_MOE_DOWN_STAGE_FU 0
#endif
/* Chunks pre-issued per stream in the RB-gated MoE arms. MEASURED at 1 block/SM with the
 * lane-split down arm and warp-per-row flash in place: UN=2 gives bf16 6.194 / fp8 5.709 vs
 * UN=4's 6.288 / 5.709 -- the shipped default had been carried over from an earlier round and
 * was no longer optimal once the other arms moved. Exactly the shape-dependence that argues
 * for putting these knobs under the tuner (plans/tuner-decode-sweep.md). */
#ifndef GV_MOE_UN
#define GV_MOE_UN 2
#endif
#ifndef PLOW_MOE_XN_MAX
#define PLOW_MOE_XN_MAX 2816u /* f32 staging capacity for the normalized x (11 KiB). Two arms
                               * stage (expert GLU + router score); their STATIC smem adds to the
                               * kernel total, and the runtime only raises the DYNAMIC limit, so
                               * oversizing this drives cuOccupancyMaxActiveBlocks to 0. */
#endif

#if PLOW_NV_GEMV_RB
/* dot of 8 bf16 weights against 8 f32 staged activations, same FMA order as dot8. */
__device__ __forceinline__ float dot8_fx(const bf16v8& w, const float* __restrict__ xs, float acc) {
#pragma unroll
    for (int j = 0; j < 8; j++) acc = fmaf(xs[j], __bfloat162float(w.x[j]), acc);
    return acc;
}
#endif

/* Flat work index -> (slot s in [0,S), channel n in [0,W)).
 * PLOW_MOE_SLOT_MAJOR=1 is the A/B NEGATIVE CONTROL: it forces the legacy slot-major order at
 * B>1 too. Same outputs (a permutation of independent warps), only the weight locality changes,
 * so the TPOT delta between the two builds IS the measured value of the reuse. */
__device__ __forceinline__ void plow_moe_unflat(unsigned f, unsigned S, unsigned W, unsigned nrow,
                                                unsigned* s, unsigned* n) {
#if defined(PLOW_MOE_SLOT_MAJOR) && PLOW_MOE_SLOT_MAJOR
    (void)nrow;
    *s = f / W; *n = f - *s * W;
#else
    if (nrow == 1u) { *s = f / W; *n = f - *s * W; }   /* legacy slot-major */
    else            { *n = f / S; *s = f - *n * S; }   /* channel-major: L2 weight reuse */
#endif
}

/* Per-row weightless/plain RMS scalars for a batch of rows, computed once per CTA into smem.
 * inv[r] = rsqrt(mean(resid[r]^2) + eps). Identical reduction shape (and thus identical result)
 * to the single-row bodies it replaces. `red` is the caller's 32-float warp-partial scratch. */
__device__ __forceinline__ void plow_moe_row_rms(float* __restrict__ inv, float* __restrict__ red,
                                                 const bf16* __restrict__ resid, unsigned H,
                                                 unsigned nrow, float eps) {
    const unsigned tid = threadIdx.x, nth = blockDim.x;
    const unsigned lane = tid & 31u, warp = tid >> 5;
    const unsigned nw = (nth + 31u) >> 5;
    for (unsigned r = 0; r < nrow; r++) {
        const bf16* rr = resid + (size_t)r * H;
        float part = 0.0f;
        for (unsigned h = tid; h < H; h += nth) {
            const float v = __bfloat162float(rr[h]);
            part += v * v;
        }
        part = plow_warp_sum(part);
        if (lane == 0) red[warp] = part;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            for (unsigned i = 0; i < nw; i++) s += red[i];
            inv[r] = rsqrtf(s / (float)H + eps);
        }
        __syncthreads();
    }
}

/* gelu_pytorch_tanh, bit-for-bit with sm120_common.cuh act_gelu_tanh (same fma form). */
__device__ __forceinline__ float plow_moe_gelu_tanh(float x) {
    const float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

/* bf16 dot of one output row across a 32-lane warp (all-reduce), x/w both bf16[K].
 *
 * VECTORIZED to the exact inner loop the dense-MLP decode GEMV uses (gemv_rows, op_gemm.cuh):
 * 128-bit loads (ld_glob8, 8 bf16/lane), GV_UNROLL=8 loads in flight before any is consumed,
 * dot8 FMA, warp_sum32 reduce. This is the #1 MoE decode lever (plans/rtx-08 §perf): the earlier
 * scalar stride-32 single-accumulator dot was latency-bound at ~30% HBM BW (the megakernel caps
 * at 8 warps/SM, so ILP inside the warp — not occupancy — hides load latency); the vectorized
 * path runs the expert GEMVs at the dense path's ~55% BW. K need not be a GV_STEP multiple: the
 * `(k<K)?…:bf16v8_zero()` load + `kk[u]>=K` skip guard zero the overshoot lanes (K=704 down:
 * 704=2·256+192, last chunk partial; K=2816 glu: exact). Depends on op_attention.cuh (ld_glob8/
 * dot8/bf16v8/warp_sum32) + op_gemm.cuh (GV_STEP/GV_UNROLL), both included before op_moe.cuh in
 * the interp and the oracle TUs. NOTE warp_sum32 is FA_RED_OFF0-gated, so the MoE dot is now
 * covered by the sm120_interp_op_test_w64 negative control. */
__device__ __forceinline__ float plow_warp_dot_bf16(const bf16* __restrict__ x,
                                                    const bf16* __restrict__ w, unsigned K,
                                                    unsigned lane) {
    const unsigned nchunk = (K + GV_STEP - 1u) / GV_STEP;
    float acc = 0.0f;
    for (unsigned c = 0; c < nchunk; c += GV_UNROLL) {
        bf16v8 wv[GV_UNROLL];
        unsigned kk[GV_UNROLL];
#pragma unroll
        for (int u = 0; u < GV_UNROLL; u++) {
            const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8u;
            kk[u] = k;
            wv[u] = (k < K) ? ld_glob8(w + k) : bf16v8_zero();
        }
#pragma unroll
        for (int u = 0; u < GV_UNROLL; u++) {
            if (kk[u] >= K) continue;
            const bf16v8 xv = ld_glob8(x + kk[u]);
            acc = dot8(wv[u], xv, acc);
        }
    }
    return warp_sum32(acc);
}

/* ---- ROUTER (PLOW_DOP_MOE_ROUTER_GEMMA) — ONE block ------------------------------------
 * table(out) = [k] of {u32 expert_id, f32 gate}.
 *   r    = resid * rsqrt(mean(resid^2)+eps)          (weightless RMSNorm)
 *   h2   = r * scale[h] * root                        (root = H^-0.5)
 *   logit[e] = sum_h h2[h] * proj[e][h]               (proj [n_exp, H], no bias)
 *   prob = softmax(logit)                             (fp32, over all n_exp)
 *   top-k via k-pass masked argmax, LOWEST-ID tie-break (packed key, mirrors op_moe.h)
 *   gate = prob(winner) / sum(prob(winners)) * per_expert_scale[winner]   (norm_topk always)
 * arena: float[H + n_exp] scratch (h2 then scores). Single block: slice!=0 returns.
 * BATCH B>1: the block loops the B rows (table/resid rows are [B][k]/[B][H]) — the exact
 * per-row body, so every row is bit-identical to its own B=1 result. */
/* One row of the router — the historical single-token body, verbatim. */
static __device__ void d_moe_router_gemma_row(unsigned char* __restrict__ table,
                                   const bf16* __restrict__ resid,
                                   const bf16* __restrict__ proj, const bf16* __restrict__ scale,
                                   const bf16* __restrict__ pes, unsigned H, unsigned n_exp,
                                   unsigned k, float root, float eps,
                                   float* __restrict__ arena) {
    float* h2 = arena;           /* [H]     */
    float* sc = arena + H;       /* [n_exp] */
    const unsigned tid = threadIdx.x, nth = blockDim.x;
    const unsigned lane = tid & 31u, warp = tid >> 5;
    __shared__ float red[32];    /* up to 32 warps of partial sumsq */

    /* 1. weightless RMS: block reduction of sum(resid^2). */
    float part = 0.0f;
    for (unsigned h = tid; h < H; h += nth) {
        const float v = __bfloat162float(resid[h]);
        part += v * v;
    }
    part = plow_warp_sum(part);
    if (lane == 0) red[warp] = part;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        const unsigned nw = (nth + 31u) >> 5;
        for (unsigned i = 0; i < nw; i++) s += red[i];
        red[0] = s;
    }
    __syncthreads();
    const float invrms = rsqrtf(red[0] / (float)H + eps);
    __syncthreads(); /* red[0] is reused as a plain scalar below; no writer races the readers */

    /* 2. h2 = resid * invrms * scale * root. */
    for (unsigned h = tid; h < H; h += nth)
        h2[h] = __bfloat162float(resid[h]) * invrms * __bfloat162float(scale[h]) * root;
    __syncthreads();

    /* 3. logits: one thread per expert (n_exp <= 256 tiny). */
    for (unsigned e = tid; e < n_exp; e += nth) {
        const bf16* pr = proj + (size_t)e * H;
        float acc = 0.0f;
        for (unsigned h = 0; h < H; h++) acc = fmaf(h2[h], __bfloat162float(pr[h]), acc);
        sc[e] = acc;
    }
    __syncthreads();

    /* 4. softmax + top-k (lowest-id tie) + norm_topk + per-expert scale, serial on thread 0. */
    if (tid == 0) {
        float m = -1e30f;
        for (unsigned e = 0; e < n_exp; e++) m = fmaxf(m, sc[e]);
        float s = 0.0f;
        for (unsigned e = 0; e < n_exp; e++) { sc[e] = __expf(sc[e] - m); s += sc[e]; }
        for (unsigned e = 0; e < n_exp; e++) sc[e] /= s; /* prob */

        for (unsigned j = 0; j < k; j++) {
            unsigned long long best = 0ull;
            unsigned bid = 0;
            for (unsigned e = 0; e < n_exp; e++) {
                unsigned sb;
                float scv = sc[e];
                __builtin_memcpy(&sb, &scv, 4);
                sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u); /* monotone f32->u32 */
                const unsigned long long key =
                    ((unsigned long long)sb << 20) | (unsigned long long)((n_exp - 1u - e) & 0xFFFFFu);
                if (key > best) { best = key; bid = e; }
            }
            *(unsigned*)(table + (size_t)j * 8) = bid;
            *(float*)(table + (size_t)j * 8 + 4) = sc[bid];
            sc[bid] = -1e30f; /* kill so the next pass cannot re-pick it */
        }
        float gs = 0.0f;
        for (unsigned j = 0; j < k; j++) gs += *(float*)(table + (size_t)j * 8 + 4);
        for (unsigned j = 0; j < k; j++) {
            const unsigned win = *(unsigned*)(table + (size_t)j * 8);
            float gate = *(float*)(table + (size_t)j * 8 + 4);
            if (gs != 0.0f) gate /= gs; /* norm_topk (always on for Gemma) */
            gate *= __bfloat162float(pes[win]);
            *(float*)(table + (size_t)j * 8 + 4) = gate;
        }
    }
    __syncthreads(); /* arena is reused by the next row */
}

/* Batch wrapper: blocks stride the B rows; table/resid are [B][k]/[B][H]. */
static __device__ void d_moe_router_gemma(unsigned char* __restrict__ table, const bf16* __restrict__ resid,
                                   const bf16* __restrict__ proj, const bf16* __restrict__ scale,
                                   const bf16* __restrict__ pes, unsigned H, unsigned n_exp,
                                   unsigned k, float root, float eps, unsigned slice, unsigned nblk,
                                   unsigned nrow, float* __restrict__ arena) {
    const unsigned stride = nblk ? nblk : 1u;
    for (unsigned row = slice; row < nrow; row += stride)
        d_moe_router_gemma_row(table + (size_t)row * k * 8, resid + (size_t)row * H, proj, scale,
                               pes, H, n_exp, k, root, eps, arena);
}

/* ---- ROUTER SCORE SPLIT (PLOW_DOP_MOE_ROUTER_GEMMA_SCORE) -----------------------------
 * The old router assigns one scalar thread to each expert dot, so its 128 independent H-wide
 * reductions occupy only one CTA. Here every warp owns one expert: lanes read adjacent residual,
 * scale and projection elements, then lane 0 consumes those values through shuffles in increasing
 * hidden-index order. Eight experts/CTA means Gemma's E=128 fills sixteen SMs while retaining the
 * legacy scalar dot's exact fmaf association. The weightless-RMS scalar is intentionally recomputed
 * by each CTA; this is only 2*H bf16 bytes per CTA and avoids a third packet/global-scratch gate.
 * BATCH B>1: the work space is the (row, expert) PAIR space [B*n_exp], still warp-per-pair, so
 * the score GEMV fills the machine at B=32 exactly as it does at B=1 (16 -> 188 CTAs). Each CTA
 * computes all B weightless-RMS scalars up front (B block reductions, 2*H bf16 each, L2-hot) —
 * the same redundancy the single-row body already accepted, and 2.6% of the step's MoE traffic
 * at B=32. score/resid are [B][n_exp]/[B][H]. Per-row arithmetic is untouched. */
static __device__ void d_moe_router_gemma_score(float* __restrict__ score,
                                          const bf16* __restrict__ resid,
                                          const bf16* __restrict__ proj,
                                          const bf16* __restrict__ scale,
                                          unsigned H, unsigned n_exp, float root, float eps,
                                          unsigned slice, unsigned nblk, unsigned nrow) {
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;
    __shared__ float red[32];
    __shared__ float invs[PLOW_MOE_MAXB];

    /* Same 256-thread f32 RMS reduction shape as the legacy router, once per row. */
    plow_moe_row_rms(invs, red, resid, H, nrow, eps);

    /* Grid stride in blocks, eight consecutive (row,expert) pairs per block. */
    const unsigned npair = nrow * n_exp;
    for (unsigned idx = slice * 8u + warp; idx < npair; idx += nblk * 8u) {
        const unsigned row = idx / n_exp;
        const unsigned e = idx - row * n_exp;
        const bf16* rr = resid + (size_t)row * H;
        const float invrms = invs[row];
        const bf16* pr = proj + (size_t)e * H;
        float acc = 0.0f;
        for (unsigned h0 = 0; h0 < H; h0 += 32u) {
            const unsigned h = h0 + lane;
            float term_h2 = 0.0f, term_w = 0.0f;
            if (h < H) {
                term_h2 = __bfloat162float(rr[h]) * invrms *
                          __bfloat162float(scale[h]) * root;
                term_w = __bfloat162float(pr[h]);
            }
#pragma unroll
            for (unsigned src = 0; src < 32u; src++) {
                const float h2 = __shfl_sync(0xffffffffu, term_h2, src);
                const float w = __shfl_sync(0xffffffffu, term_w, src);
                if (lane == 0 && h0 + src < H) acc = fmaf(h2, w, acc);
            }
        }
        if (lane == 0) score[(size_t)row * n_exp + e] = acc;
    }
}

/* Association-changing fast experiment: identical block/expert mapping and RMS transform as the
 * exact scorer, but a conventional strided lane dot + warp reduction replaces the ordered shuffle
 * replay. Kept as a distinct opcode so a fast experiment can never silently change exact packets. */
static __device__ void d_moe_router_gemma_score_fast(float* __restrict__ score,
                                               const bf16* __restrict__ resid,
                                               const bf16* __restrict__ proj,
                                               const bf16* __restrict__ scale,
                                               unsigned H, unsigned n_exp, float root, float eps,
                                               unsigned slice, unsigned nblk, unsigned nrow,
                                               float* __restrict__ arena) {
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;
    __shared__ float red[32];
    __shared__ float invs[PLOW_MOE_MAXB];
    plow_moe_row_rms(invs, red, resid, H, nrow, eps);

#if PLOW_NV_GEMV_RB
    /* Vectorized twin (H100 campaign round 3). The body below reads resid/scale/proj with
     * SCALAR 2-byte loads and recomputes h2 = resid*invrms*scale*root from global for EVERY
     * expert row (128 x H per layer of redundant reads). Measured 0.560 ms of a 7.9 ms decode
     * step at ~38 GB/s to produce 721 KB of logits.
     * Stage h2 once per CTA as f32 (so each FMA sees the same value the scalar body computes)
     * and stream proj with the 16 B ld_glob8 every other GEMV arm uses. The per-lane K
     * partition changes (lane-strided -> 8-wide chunks), so the warp-sum rounding of a logit
     * differs; selection is unchanged in practice and gated by gpu_lifecycle. */
    if (nrow == 1u && H <= PLOW_MOE_XN_MAX) {
#if PLOW_MOE_XN_BF16
        bf16* h2s = (bf16*)arena;
#else
        float* h2s = arena; /* dynamic arena, see the note in the expert-GLU arm */
#endif
        const float invrms = invs[0];
        for (unsigned h = tid; h < H; h += blockDim.x) {
            const float v = __bfloat162float(resid[h]) * invrms * __bfloat162float(scale[h]) * root;
#if PLOW_MOE_XN_BF16
            h2s[h] = __float2bfloat16(v);
#else
            h2s[h] = v;
#endif
        }
        __syncthreads();
        const unsigned nchunk = (H + GV_STEP - 1u) / GV_STEP;
#if PLOW_MOE_ROUTER_WIDE
        /* ONE EXPERT PER BLOCK, all 8 warps splitting H. The default map below is
         * `slice*8 + warp`, i.e. 8 experts per block, so with n_exp=128 only 16 blocks get any
         * work -- 16 of 132 at occ-1 and 16 of 264 at occ-2. The router then sits on the
         * critical path of every other block: measured 0.424 ms of a 4.61 ms fp8 step at occ-2,
         * where at occ-1 it had been hidden behind slower arms.
         * Here block `slice` owns expert `slice`, its warps split the K sweep, and one smem
         * reduction folds them. That uses 128 blocks instead of 16 and makes each expert's dot
         * 8x more parallel. Blocks past n_exp skip the loop entirely -- `slice` is block-uniform,
         * so the __syncthreads below is never divergently reached. */
        __shared__ float rw_red[PLOW_NV_WARPS];
        for (unsigned e = slice; e < n_exp; e += nblk) {
            const bf16* pr = proj + (size_t)e * H;
            float acc = 0.0f;
            for (unsigned c = warp * GV_MOE_UN; c < nchunk; c += PLOW_NV_WARPS * GV_MOE_UN) {
                bf16v8 wv[GV_MOE_UN];
                unsigned kk[GV_MOE_UN];
#pragma unroll
                for (int u = 0; u < GV_MOE_UN; u++) {
                    kk[u] = (c + (unsigned)u) * GV_STEP + lane * 8u;
                    wv[u] = (kk[u] < H) ? ld_glob8(pr + kk[u]) : bf16v8_zero();
                }
#pragma unroll
                for (int u = 0; u < GV_MOE_UN; u++) {
                    if (kk[u] >= H) continue;
                    acc = dot8_fx(wv[u], h2s + kk[u], acc);
                }
            }
            acc = plow_warp_sum(acc);
            if (lane == 0) rw_red[warp] = acc;
            __syncthreads();
            if (tid == 0) {
                float t = 0.0f;
                for (unsigned i = 0; i < PLOW_NV_WARPS; i++) t += rw_red[i];
                score[e] = t;
            }
            __syncthreads();
        }
        return;
#endif
        for (unsigned e = slice * 8u + warp; e < n_exp; e += nblk * 8u) {
            const bf16* pr = proj + (size_t)e * H;
            float acc = 0.0f;
            for (unsigned c = 0; c < nchunk; c += GV_MOE_UN) {
                bf16v8 wv[GV_MOE_UN];
                unsigned kk[GV_MOE_UN];
#pragma unroll
                for (int u = 0; u < GV_MOE_UN; u++) {
                    kk[u] = (c + (unsigned)u) * GV_STEP + lane * 8u;
                    wv[u] = (kk[u] < H) ? ld_glob8(pr + kk[u]) : bf16v8_zero();
                }
#pragma unroll
                for (int u = 0; u < GV_MOE_UN; u++) {
                    if (kk[u] >= H) continue;
#if PLOW_MOE_XN_BF16
                    acc = dot8(wv[u], ld_smem8(h2s + kk[u]), acc);
#else
                    acc = dot8_fx(wv[u], h2s + kk[u], acc);
#endif
                }
            }
            acc = plow_warp_sum(acc);
            if (lane == 0) score[e] = acc;
        }
        return;
    }
#endif
    const unsigned npair = nrow * n_exp;
    for (unsigned idx = slice * 8u + warp; idx < npair; idx += nblk * 8u) {
        const unsigned row = idx / n_exp;
        const unsigned e = idx - row * n_exp;
        const bf16* rr = resid + (size_t)row * H;
        const float invrms = invs[row];
        const bf16* pr = proj + (size_t)e * H;
        float acc = 0.0f;
        for (unsigned h = lane; h < H; h += 32u) {
            const float h2 = __bfloat162float(rr[h]) * invrms *
                             __bfloat162float(scale[h]) * root;
            acc = fmaf(h2, __bfloat162float(pr[h]), acc);
        }
        acc = plow_warp_sum(acc);
        if (lane == 0) score[(size_t)row * n_exp + e] = acc;
    }
}

/* ---- ROUTER SOFTMAX/TOP-K TAIL (PLOW_DOP_MOE_ROUTER_GEMMA_TOPK) -----------------------
 * ONE CTA. Dynamic arena holds the mutable probabilities; routing-table slots themselves are
 * the win/gate scratch, avoiding the legacy thread-local win[8]/gate[8] arrays and their local
 * stack traffic. */
/* One row: the historical serial softmax/top-k/norm_topk/per-expert-scale ordering, verbatim. */
static __device__ void d_moe_router_gemma_topk_row(unsigned char* __restrict__ table,
                                         const float* __restrict__ score,
                                         const bf16* __restrict__ pes,
                                         unsigned n_exp, unsigned k,
                                         float* __restrict__ arena) {
    float* sc = arena;
    for (unsigned e = threadIdx.x; e < n_exp; e += blockDim.x) sc[e] = score[e];
    __syncthreads();
#if PLOW_NV_GEMV_RB
    /* WARP-PARALLEL TOP-K (H100 campaign round 3). The default body below runs the entire
     * 128-expert softmax AND the k x n_exp masked-argmax scan on THREAD 0 alone: ~1400 serial
     * dependent iterations per layer, 30x per token, while 132 blocks x 256 threads idle.
     * Measured 0.760 ms of a 7.92 ms decode step (9.6%) to pick 8 experts out of 128.
     *
     * BIT-EXACT: max and argmax are order-independent (the packed key encodes the lowest-id
     * tie-break, so the winner is unique regardless of reduction order), and the exp/divide are
     * per-element. The softmax DENOMINATOR is still summed sequentially by lane 0 in the
     * original e=0..n_exp order, because a tree sum would round differently and could in
     * principle move a gate. Same writes, same values. */
    {
        const unsigned lane = threadIdx.x & 31u;
        if (threadIdx.x < 32u) {
            float m = -1e30f;
            for (unsigned e = lane; e < n_exp; e += 32u) m = fmaxf(m, sc[e]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
            for (unsigned e = lane; e < n_exp; e += 32u) sc[e] = __expf(sc[e] - m);
            __syncwarp();
            float s = 0.0f;
            if (lane == 0)
                for (unsigned e = 0; e < n_exp; e++) s += sc[e]; /* original order, exact */
            s = __shfl_sync(~0u, s, 0);
            for (unsigned e = lane; e < n_exp; e += 32u) sc[e] /= s;
            __syncwarp();

            for (unsigned j = 0; j < k; j++) {
                unsigned long long best = 0ull;
                for (unsigned e = lane; e < n_exp; e += 32u) {
                    unsigned sb;
                    const float scv = sc[e];
                    __builtin_memcpy(&sb, &scv, 4);
                    sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
                    const unsigned long long key =
                        ((unsigned long long)sb << 20) |
                        (unsigned long long)((n_exp - 1u - e) & 0xFFFFFu);
                    if (key > best) best = key;
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    const unsigned long long t = __shfl_xor_sync(~0u, best, o);
                    if (t > best) best = t;
                }
                const unsigned bid = n_exp - 1u - (unsigned)(best & 0xFFFFFull);
                if (lane == 0) {
                    *(unsigned*)(table + (size_t)j * 8) = bid;
                    *(float*)(table + (size_t)j * 8 + 4) = sc[bid];
                    sc[bid] = -1e30f;
                }
                __syncwarp();
            }
            if (lane == 0) {
                float gs = 0.0f;
                for (unsigned j = 0; j < k; j++) gs += *(float*)(table + (size_t)j * 8 + 4);
                for (unsigned j = 0; j < k; j++) {
                    const unsigned win = *(unsigned*)(table + (size_t)j * 8);
                    float gate = *(float*)(table + (size_t)j * 8 + 4);
                    if (gs != 0.0f) gate /= gs;
                    gate *= __bfloat162float(pes[win]);
                    *(float*)(table + (size_t)j * 8 + 4) = gate;
                }
            }
        }
        __syncthreads(); /* arena is reused by the next row */
        return;
    }
#endif
    /* The serial tail is thread 0's, but NO thread may return early: when one block owns more
     * than one batched row the loop below runs again and every thread must reach that barrier. */
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (unsigned e = 0; e < n_exp; e++) m = fmaxf(m, sc[e]);
        float s = 0.0f;
        for (unsigned e = 0; e < n_exp; e++) { sc[e] = __expf(sc[e] - m); s += sc[e]; }
        for (unsigned e = 0; e < n_exp; e++) sc[e] /= s;

        for (unsigned j = 0; j < k; j++) {
            unsigned long long best = 0ull;
            unsigned bid = 0;
            for (unsigned e = 0; e < n_exp; e++) {
                unsigned sb;
                const float scv = sc[e];
                __builtin_memcpy(&sb, &scv, 4);
                sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
                const unsigned long long key =
                    ((unsigned long long)sb << 20) |
                    (unsigned long long)((n_exp - 1u - e) & 0xFFFFFu);
                if (key > best) { best = key; bid = e; }
            }
            *(unsigned*)(table + (size_t)j * 8) = bid;
            *(float*)(table + (size_t)j * 8 + 4) = sc[bid];
            sc[bid] = -1e30f;
        }
        float gs = 0.0f;
        for (unsigned j = 0; j < k; j++) gs += *(float*)(table + (size_t)j * 8 + 4);
        for (unsigned j = 0; j < k; j++) {
            const unsigned win = *(unsigned*)(table + (size_t)j * 8);
            float gate = *(float*)(table + (size_t)j * 8 + 4);
            if (gs != 0.0f) gate /= gs;
            gate *= __bfloat162float(pes[win]);
            *(float*)(table + (size_t)j * 8 + 4) = gate;
        }
    }
    __syncthreads(); /* arena is reused by the next row */
}

/* Batch wrapper: one block per row (blocks stride B); table/score are [B][k]/[B][n_exp]. */
static __device__ void d_moe_router_gemma_topk(unsigned char* __restrict__ table,
                                         const float* __restrict__ score,
                                         const bf16* __restrict__ pes,
                                         unsigned n_exp, unsigned k, unsigned slice,
                                         unsigned nblk, unsigned nrow,
                                         float* __restrict__ arena) {
    const unsigned stride = nblk ? nblk : 1u;
    for (unsigned row = slice; row < nrow; row += stride)
        d_moe_router_gemma_topk_row(table + (size_t)row * k * 8, score + (size_t)row * n_exp, pes,
                                    n_exp, k, arena);
}

/* ---- EXPERT GATE/UP (PLOW_DOP_MOE_EXPERT_GLU_GEMMA) — flat, one warp per output ----------
 * fu[slot][n] = gelu_tanh(gate_e . x) * (up_e . x), fused gate_up weight. One warp per
 * (slot, channel n in [0,I_moe)); slots are the k routed experts, x is the single [H] row.
 * The arena variant stages x into smem once, then all warps read from smem — eliminates L1
 * contention across the 8 warps sharing the same activation vector.
 * BATCH B>1: x is [B][H], table is [B][k] and fu is [B][k][I_moe]; the flat sweep runs over the
 * B*k slots in CHANNEL-MAJOR order (see the ordering note above) so slots sharing an expert hit
 * that expert's weight rows in L2 instead of re-reading HBM. */
static __device__ void d_moe_expert_glu_gemma(bf16*, const bf16*, const unsigned char*,
                                       const unsigned long long*, unsigned, unsigned, unsigned,
                                       unsigned, unsigned, unsigned, unsigned);
static __device__ void d_moe_expert_glu_gemma(bf16* __restrict__ fu, const bf16* __restrict__ x,
                                       const unsigned char* __restrict__ table,
                                       const unsigned long long* __restrict__ ewt, unsigned k,
                                       unsigned I_moe, unsigned H, unsigned n_exp, unsigned slice,
                                       unsigned nblk, unsigned nrow, bf16* __restrict__ arena) {
    /* B>1: staging every row would need nrow*2*H smem (180 KB at B=32, H=2816) — far past the
     * arena. The batched rows are read straight from global instead; they are B*5.6 KB total and
     * sit in L2 for the whole packet, while the weight stream (the actual cost) is unaffected. */
    if (nrow != 1u) {
        d_moe_expert_glu_gemma(fu, x, table, ewt, k, I_moe, H, n_exp, slice, nblk, nrow);
        return;
    }
    bf16* xs = arena;
    for (unsigned i = threadIdx.x; i < H; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned nw = blockDim.x >> 5;
    const unsigned total = k * I_moe;
    const unsigned per = (total + nblk - 1u) / nblk;
    const unsigned f0 = slice * per;
    const unsigned f1 = (f0 + per < total) ? (f0 + per) : total;
    const unsigned nchunk = (H + GV_STEP - 1u) / GV_STEP;

    for (unsigned f = f0 + (threadIdx.x >> 5); f < f1; f += nw) {
        const unsigned slot = f / I_moe;
        const unsigned n = f - slot * I_moe;
        const unsigned eid = plow_moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;
        const unsigned long long gub = ewt[(size_t)eid * 2 + 0];
        if (gub == 0ull) continue;
        const bf16* grow = (const bf16*)(size_t)gub + (size_t)n * H;
        const bf16* urow = (const bf16*)(size_t)gub + (size_t)(I_moe + n) * H;
        float ag = 0.0f, au = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL_GLU) {
            bf16v8 gv[GV_UNROLL_GLU], uv[GV_UNROLL_GLU];
            unsigned kk[GV_UNROLL_GLU];
#pragma unroll
            for (int i = 0; i < GV_UNROLL_GLU; i++) {
                const unsigned k_ = (c + (unsigned)i) * GV_STEP + lane * 8u;
                kk[i] = k_;
                gv[i] = (k_ < H) ? ld_glob8(grow + k_) : bf16v8_zero();
            }
#pragma unroll
            for (int i = 0; i < GV_UNROLL_GLU; i++) {
                uv[i] = (kk[i] < H) ? ld_glob8(urow + kk[i]) : bf16v8_zero();
            }
#pragma unroll
            for (int i = 0; i < GV_UNROLL_GLU; i++) {
                if (kk[i] >= H) continue;
                const bf16v8 xv = ld_smem8(xs + kk[i]);
                ag = dot8(gv[i], xv, ag);
                au = dot8(uv[i], xv, au);
            }
        }
        const float g = warp_sum32(ag);
        const float u = warp_sum32(au);
        if (lane == 0) fu[(size_t)slot * I_moe + n] = __float2bfloat16(plow_moe_gelu_tanh(g) * u);
    }
}

/* Non-arena overload for standalone test kernels (reads x from L1/global) and the B>1 path. */
static __device__ void d_moe_expert_glu_gemma(bf16* __restrict__ fu, const bf16* __restrict__ x,
                                       const unsigned char* __restrict__ table,
                                       const unsigned long long* __restrict__ ewt, unsigned k,
                                       unsigned I_moe, unsigned H, unsigned n_exp, unsigned slice,
                                       unsigned nblk, unsigned nrow) {
    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned nw = blockDim.x >> 5;
    const unsigned nslot = nrow * k;
    const unsigned total = nslot * I_moe;
    const unsigned per = (total + nblk - 1u) / nblk;
    const unsigned f0 = slice * per;
    const unsigned f1 = (f0 + per < total) ? (f0 + per) : total;
    const unsigned nchunk = (H + GV_STEP - 1u) / GV_STEP;

    for (unsigned f = f0 + (threadIdx.x >> 5); f < f1; f += nw) {
        unsigned slot, n;
        plow_moe_unflat(f, nslot, I_moe, nrow, &slot, &n);
        const bf16* xr = x + (size_t)(slot / k) * H;
        const unsigned eid = plow_moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;
        const unsigned long long gub = ewt[(size_t)eid * 2 + 0];
        if (gub == 0ull) continue;
        const bf16* grow = (const bf16*)(size_t)gub + (size_t)n * H;
        const bf16* urow = (const bf16*)(size_t)gub + (size_t)(I_moe + n) * H;
        float ag = 0.0f, au = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL_GLU) {
            bf16v8 gv[GV_UNROLL_GLU], uv[GV_UNROLL_GLU];
            unsigned kk[GV_UNROLL_GLU];
#pragma unroll
            for (int i = 0; i < GV_UNROLL_GLU; i++) {
                const unsigned k_ = (c + (unsigned)i) * GV_STEP + lane * 8u;
                kk[i] = k_;
                gv[i] = (k_ < H) ? ld_glob8(grow + k_) : bf16v8_zero();
            }
#pragma unroll
            for (int i = 0; i < GV_UNROLL_GLU; i++) {
                uv[i] = (kk[i] < H) ? ld_glob8(urow + kk[i]) : bf16v8_zero();
            }
#pragma unroll
            for (int i = 0; i < GV_UNROLL_GLU; i++) {
                if (kk[i] >= H) continue;
                const bf16v8 xv = ld_glob8(xr + kk[i]);
                ag = dot8(gv[i], xv, ag);
                au = dot8(uv[i], xv, au);
            }
        }
        const float g = warp_sum32(ag);
        const float u = warp_sum32(au);
        if (lane == 0) fu[(size_t)slot * I_moe + n] = __float2bfloat16(plow_moe_gelu_tanh(g) * u);
    }
}

/* ---- EXPERT DOWN (PLOW_DOP_MOE_EXPERT_DOWN_GEMMA) — flat, one warp per output ------------
 * part[slot][h] = gate_slot * (down_e[h] . fu[slot]).  Gate-scaled f32 partial, fixed slot
 * order, so the combine is a deterministic sum. Skipped slots write 0. One warp per (slot,h). */
static __device__ void d_moe_expert_down_gemma(float* __restrict__ part, const bf16* __restrict__ fu,
                                        const unsigned char* __restrict__ table,
                                        const unsigned long long* __restrict__ ewt, unsigned k,
                                        unsigned H, unsigned I_moe, unsigned n_exp, unsigned slice,
                                        unsigned nblk, unsigned nrow, float* __restrict__ arena) {
#if PLOW_MOE_DOWN_LANESPLIT
    /* LANE-SPLIT DOWN (H100 round 6). The DOWN arm has a SHORT K (I_moe=704 => only 3 chunks of
     * 256 per lane), so per-row cost -- expert lookup, the 32-lane warp reduction -- is amortised
     * over almost nothing, and it measured the worst GEMV in the step (1163 GB/s at occ-2 vs
     * 3269 achievable). Row-blocking cannot fix it: RB needs wv[RB][UN] registers and RB>2 blows
     * the budget.
     * Instead split the warp into 4 sub-groups of 8 lanes, one output channel each. That puts 4
     * rows in flight per warp with ONE accumulator per lane (no wv[][] array at all), gives each
     * lane a longer contiguous run (I_moe/8 = 88 elems), and shrinks the reduction from 5 shuffle
     * steps to 3. A sub-group's 8 lanes read 128 contiguous bytes, so each warp instruction is 4
     * x 128 B rather than one 512 B burst.
     * Reduction order changes (8-lane tree, not 32), so outputs are numerically equivalent, not
     * bit-identical. Needs I_moe % 64 == 0 (704 = 11*64). */
    constexpr unsigned LSG = PLOW_MOE_DOWN_SG;       /* sub-groups (channels) per warp */
    constexpr unsigned LSL = 32u / LSG;               /* lanes per sub-group */
    constexpr unsigned LCH = LSL * 8u;                /* elems a sub-group covers per chunk */
    if (nrow == 1u && (I_moe % LCH) == 0u) {
        const unsigned lane = threadIdx.x & 31u;
        const unsigned sg = lane / LSL;
        const unsigned sl = lane % LSL;
        const unsigned nw = blockDim.x >> 5;
        const unsigned total = k * H;
        const unsigned per = (total + nblk - 1u) / nblk;
        const unsigned f0 = slice * per;
        const unsigned f1 = (f0 + per < total) ? (f0 + per) : total;
        const unsigned nch = I_moe / LCH;
        for (unsigned fb = f0 + (threadIdx.x >> 5) * LSG; fb < f1; fb += nw * LSG) {
            const unsigned f = fb + sg;
            bool valid = false, live = false;
            float gate = 0.0f;
            float* dst = nullptr;
            const bf16* wr = nullptr;
            const bf16* xr = nullptr;
            if (f < f1) {
                valid = true;
                unsigned slot, h;
                plow_moe_unflat(f, k, H, 1u, &slot, &h);
                dst = part + (size_t)slot * H + h;
                const unsigned eid = plow_moe_slot_expert(table, slot);
                const unsigned long long db = (eid < n_exp) ? ewt[(size_t)eid * 2 + 1] : 0ull;
                if (db != 0ull) {
                    live = true;
                    gate = plow_moe_slot_gate(table, slot);
                    wr = (const bf16*)(size_t)db + (size_t)h * I_moe;
                    xr = fu + (size_t)slot * I_moe;
                }
            }
            float acc = 0.0f;
            if (live) {
                unsigned c = 0;
                for (; c + 2u <= nch; c += 2u) { /* 2 chunks pre-issued */
                    const unsigned k0 = c * LCH + sl * 8u, k1 = (c + 1u) * LCH + sl * 8u;
                    const bf16v8 w0 = ld_glob8(wr + k0), w1 = ld_glob8(wr + k1);
                    const bf16v8 x0 = ld_glob8(xr + k0), x1 = ld_glob8(xr + k1);
                    acc = dot8(w0, x0, acc);
                    acc = dot8(w1, x1, acc);
                }
                for (; c < nch; c++) {
                    const unsigned k0 = c * LCH + sl * 8u;
                    acc = dot8(ld_glob8(wr + k0), ld_glob8(xr + k0), acc);
                }
            }
#pragma unroll
            for (int o = 1; o < (int)LSL; o <<= 1) acc += __shfl_xor_sync(~0u, acc, o);
            if (sl == 0u && valid) *dst = live ? gate * acc : 0.0f;
        }
        return;
    }
#endif
#if PLOW_NV_GEMV_RB
    /* B=1 decode: give each warp GV_MOE_RB output channels so it keeps RB*UN weight loads in
     * flight. plow_warp_dot_bf16 is already 16 B-vectorized, so this adds only the missing
     * memory-level parallelism; the per-row FMA order is unchanged (bit-identical outputs).
     * f is slot-major for nrow==1, so RB consecutive f share the expert and step h by 1. */
    if (nrow == 1u) {
        /* Stage the whole fu row-set (k*I_moe bf16 = 11 KiB at k=8, I_moe=704) in the arena.
         * Every warp otherwise re-reads its slot's 1.4 KiB of fu from global for EVERY one of
         * its output channels -- ~240 KiB of redundant traffic per block per layer. */
#if PLOW_MOE_DOWN_STAGE_FU
        bf16* fus = (bf16*)arena;
        const unsigned fu_n = k * I_moe;
        if (fu_n > PLOW_MOE_XN_MAX * 2u) goto down_scalar; /* arena too small: unblocked body */
        for (unsigned i = threadIdx.x; i < fu_n; i += blockDim.x) fus[i] = fu[i];
        __syncthreads();
#endif
        const unsigned lane_r = threadIdx.x & (PLOW_NV_WARP - 1u);
        const unsigned nw_r = blockDim.x >> 5;
        const unsigned total_r = k * H;
        const unsigned per_r = (total_r + nblk - 1u) / nblk;
        const unsigned f0_r = slice * per_r;
        const unsigned f1_r = (f0_r + per_r < total_r) ? (f0_r + per_r) : total_r;
        const unsigned nchunk = (I_moe + GV_STEP - 1u) / GV_STEP;
        for (unsigned fb = f0_r + (threadIdx.x >> 5) * GV_MOE_RB_DN; fb < f1_r;
             fb += nw_r * GV_MOE_RB_DN) {
            const bf16* wr[GV_MOE_RB_DN];
            const bf16* xr[GV_MOE_RB_DN];
            float* dst[GV_MOE_RB_DN];
            float gate[GV_MOE_RB_DN];
            bool valid[GV_MOE_RB_DN], live[GV_MOE_RB_DN];
#pragma unroll
            for (int r = 0; r < GV_MOE_RB_DN; r++) {
                valid[r] = false;
                live[r] = false;
                wr[r] = nullptr;
                xr[r] = nullptr;
                dst[r] = nullptr;
                gate[r] = 0.0f;
                const unsigned f = fb + (unsigned)r;
                if (f >= f1_r) continue;
                unsigned slot, h;
                plow_moe_unflat(f, k, H, 1u, &slot, &h);
                valid[r] = true;
                dst[r] = part + (size_t)slot * H + h;
                const unsigned eid = plow_moe_slot_expert(table, slot);
                const unsigned long long db = (eid < n_exp) ? ewt[(size_t)eid * 2 + 1] : 0ull;
                if (db == 0ull) continue; /* deterministic zero partial below */
                live[r] = true;
                gate[r] = plow_moe_slot_gate(table, slot);
                wr[r] = (const bf16*)(size_t)db + (size_t)h * I_moe;
#if PLOW_MOE_DOWN_STAGE_FU
                xr[r] = fus + (size_t)slot * I_moe;
#else
                xr[r] = fu + (size_t)slot * I_moe;
#endif
            }
            float acc[GV_MOE_RB_DN];
#pragma unroll
            for (int r = 0; r < GV_MOE_RB_DN; r++) acc[r] = 0.0f;
            for (unsigned c = 0; c < nchunk; c += GV_MOE_UN) {
                bf16v8 wv[GV_MOE_RB_DN][GV_MOE_UN];
                unsigned kk[GV_MOE_UN];
#pragma unroll
                for (int u = 0; u < GV_MOE_UN; u++)
                    kk[u] = (c + (unsigned)u) * GV_STEP + lane_r * 8u;
#pragma unroll
                for (int r = 0; r < GV_MOE_RB_DN; r++) {
#pragma unroll
                    for (int u = 0; u < GV_MOE_UN; u++)
                        wv[r][u] = (live[r] && kk[u] < I_moe) ? ld_glob8(wr[r] + kk[u])
                                                             : bf16v8_zero();
                }
#pragma unroll
                for (int u = 0; u < GV_MOE_UN; u++) {
                    if (kk[u] >= I_moe) continue;
#pragma unroll
                    for (int r = 0; r < GV_MOE_RB_DN; r++) {
                        if (!live[r]) continue;
#if PLOW_MOE_DOWN_STAGE_FU
                        acc[r] = dot8(wv[r][u], ld_smem8(xr[r] + kk[u]), acc[r]);
#else
                        acc[r] = dot8(wv[r][u], ld_glob8(xr[r] + kk[u]), acc[r]);
#endif
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < GV_MOE_RB_DN; r++) {
                const float y = warp_sum32(acc[r]);
                if (lane_r == 0 && valid[r]) *dst[r] = live[r] ? gate[r] * y : 0.0f;
            }
        }
        return;
    }
down_scalar:
#endif
    {
    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned nw = blockDim.x >> 5;
    const unsigned nslot = nrow * k;      /* B>1: part is [B][k][H], fu is [B][k][I_moe] */
    const unsigned total = nslot * H;
    const unsigned per = (total + nblk - 1u) / nblk;
    const unsigned f0 = slice * per;
    const unsigned f1 = (f0 + per < total) ? (f0 + per) : total;

    for (unsigned f = f0 + (threadIdx.x >> 5); f < f1; f += nw) {
        unsigned slot, h;
        plow_moe_unflat(f, nslot, H, nrow, &slot, &h);
        const unsigned eid = plow_moe_slot_expert(table, slot);
        float* pslot = part + (size_t)slot * H;
        const unsigned long long db = (eid < n_exp) ? ewt[(size_t)eid * 2 + 1] : 0ull;
        if (db == 0ull) {
            if (lane == 0) pslot[h] = 0.0f;                 /* deterministic zero partial */
            continue;
        }
        const float gate = plow_moe_slot_gate(table, slot);
        const bf16* Wd = (const bf16*)(size_t)db;
        const float y = plow_warp_dot_bf16(fu + (size_t)slot * I_moe, Wd + (size_t)h * I_moe, I_moe, lane);
        if (lane == 0) pslot[h] = gate * y;
    }
    }
}

/* ---- Per-output-channel FP8 Gemma experts -----------------------------------------------
 * The offline quantizer stores one e4m3 row scale per output channel, so the scale factors out
 * of the K reduction. Keep the fused gate_up layout and Gemma GELU/top-k semantics identical to
 * the bf16 bodies above; only the selected expert's weight stream and epilogue scale change. */
#ifndef GV_UNROLL_FP8
#define GV_UNROLL_FP8 4
#endif
__device__ __forceinline__ float plow_warp_dot_fp8_row(
        const bf16* __restrict__ x, const uint8_t* __restrict__ w, unsigned K, unsigned lane) {
    const unsigned nchunk = (K + GV_STEP - 1u) / GV_STEP;
    float acc = 0.0f;
    for (unsigned c = 0; c < nchunk; c += GV_UNROLL_FP8) {
        uint2 wq[GV_UNROLL_FP8];
        bf16v8 xv[GV_UNROLL_FP8];
        unsigned kk[GV_UNROLL_FP8];
#pragma unroll
        for (int u = 0; u < GV_UNROLL_FP8; u++) {
            kk[u] = (c + (unsigned)u) * GV_STEP + lane * 8u;
            wq[u] = (kk[u] < K) ? *(const uint2*)(w + kk[u]) : uint2{0, 0};
        }
#pragma unroll
        for (int u = 0; u < GV_UNROLL_FP8; u++) {
            xv[u] = (kk[u] < K) ? ld_glob8(x + kk[u]) : bf16v8_zero();
        }
#pragma unroll
        for (int u = 0; u < GV_UNROLL_FP8; u++) {
            if (kk[u] < K) acc = dot8_fp8(wq[u], xv[u], acc);
        }
    }
    return warp_sum32(acc);
}

/* Non-arena B>1 overload (fwd decl): reads x rows from global, channel-major sweep. */
static __device__ void d_moe_expert_glu_gemma_fp8(
        bf16* __restrict__ fu, const bf16* __restrict__ x,
        const unsigned char* __restrict__ table,
        const unsigned long long* __restrict__ ewt,
        const unsigned long long* __restrict__ est, unsigned k, unsigned I_moe, unsigned H,
        unsigned n_exp, unsigned slice, unsigned nblk, unsigned nrow);
static __device__ void d_moe_expert_glu_gemma_fp8(
        bf16* __restrict__ fu, const bf16* __restrict__ x,
        const unsigned char* __restrict__ table,
        const unsigned long long* __restrict__ ewt,
        const unsigned long long* __restrict__ est, unsigned k, unsigned I_moe, unsigned H,
        unsigned n_exp, unsigned slice, unsigned nblk, unsigned nrow,
        bf16* __restrict__ arena) {
    /* B>1: staging every row would overflow the arena (bf16 twin of the bf16 body). The batched
     * rows are read straight from global (B*5.6 KB, L2-resident); the weight stream is unaffected. */
    if (nrow != 1u) {
        d_moe_expert_glu_gemma_fp8(fu, x, table, ewt, est, k, I_moe, H, n_exp, slice, nblk, nrow);
        return;
    }
    bf16* xs = arena;
    for (unsigned i = threadIdx.x; i < H; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned nw = blockDim.x >> 5;
    const unsigned total = k * I_moe;
    const unsigned per = (total + nblk - 1u) / nblk;
    const unsigned f0 = slice * per;
    const unsigned f1 = (f0 + per < total) ? (f0 + per) : total;
    const unsigned nchunk = (H + GV_STEP - 1u) / GV_STEP;
    for (unsigned f = f0 + (threadIdx.x >> 5); f < f1; f += nw) {
        const unsigned slot = f / I_moe;
        const unsigned n = f - slot * I_moe;
        const unsigned eid = plow_moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;
        const unsigned long long wb = ewt[(size_t)eid * 2 + 0];
        const unsigned long long sb = est[(size_t)eid * 2 + 0];
        if (wb == 0ull || sb == 0ull) continue;
        const uint8_t* grow = (const uint8_t*)(size_t)wb + (size_t)n * H;
        const uint8_t* urow = (const uint8_t*)(size_t)wb + (size_t)(I_moe + n) * H;
        const float* scale = (const float*)(size_t)sb;
        float ag = 0.0f, au = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL_FP8) {
            uint2 gv[GV_UNROLL_FP8], uv[GV_UNROLL_FP8];
            unsigned kk[GV_UNROLL_FP8];
#pragma unroll
            for (int i = 0; i < GV_UNROLL_FP8; i++) {
                kk[i] = (c + (unsigned)i) * GV_STEP + lane * 8u;
                gv[i] = (kk[i] < H) ? *(const uint2*)(grow + kk[i]) : uint2{0, 0};
                uv[i] = (kk[i] < H) ? *(const uint2*)(urow + kk[i]) : uint2{0, 0};
            }
#pragma unroll
            for (int i = 0; i < GV_UNROLL_FP8; i++) {
                if (kk[i] >= H) continue;
                const bf16v8 xv = ld_smem8(xs + kk[i]);
                ag = dot8_fp8(gv[i], xv, ag);
                au = dot8_fp8(uv[i], xv, au);
            }
        }
        const float g = warp_sum32(ag) * scale[n];
        const float u = warp_sum32(au) * scale[I_moe + n];
        if (lane == 0) fu[(size_t)slot * I_moe + n] = __float2bfloat16(plow_moe_gelu_tanh(g) * u);
    }
}

/* Non-arena overload for standalone test kernels (reads x from L1/global) and the B>1 path.
 * B>1: x is [nrow][H], table is [nrow][k], fu is [nrow][k][I_moe]; the flat sweep runs over the
 * nrow*k slots in CHANNEL-MAJOR order (matches the bf16 body) so slots sharing an expert reuse
 * that expert's weight rows in L2. nrow==1 keeps the legacy slot-major order (byte-identical). */
static __device__ void d_moe_expert_glu_gemma_fp8(
        bf16* __restrict__ fu, const bf16* __restrict__ x,
        const unsigned char* __restrict__ table,
        const unsigned long long* __restrict__ ewt,
        const unsigned long long* __restrict__ est, unsigned k, unsigned I_moe, unsigned H,
        unsigned n_exp, unsigned slice, unsigned nblk, unsigned nrow) {
    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned nw = blockDim.x >> 5;
    const unsigned nslot = nrow * k;
    const unsigned total = nslot * I_moe;
    const unsigned per = (total + nblk - 1u) / nblk;
    const unsigned f0 = slice * per;
    const unsigned f1 = (f0 + per < total) ? (f0 + per) : total;
    const unsigned nchunk = (H + GV_STEP - 1u) / GV_STEP;
    for (unsigned f = f0 + (threadIdx.x >> 5); f < f1; f += nw) {
        unsigned slot, n;
        plow_moe_unflat(f, nslot, I_moe, nrow, &slot, &n);
        const bf16* xr = x + (size_t)(slot / k) * H;
        const unsigned eid = plow_moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;
        const unsigned long long wb = ewt[(size_t)eid * 2 + 0];
        const unsigned long long sb = est[(size_t)eid * 2 + 0];
        if (wb == 0ull || sb == 0ull) continue;
        const uint8_t* grow = (const uint8_t*)(size_t)wb + (size_t)n * H;
        const uint8_t* urow = (const uint8_t*)(size_t)wb + (size_t)(I_moe + n) * H;
        const float* scale = (const float*)(size_t)sb;
        float ag = 0.0f, au = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL_FP8) {
            uint2 gv[GV_UNROLL_FP8], uv[GV_UNROLL_FP8];
            unsigned kk[GV_UNROLL_FP8];
#pragma unroll
            for (int i = 0; i < GV_UNROLL_FP8; i++) {
                kk[i] = (c + (unsigned)i) * GV_STEP + lane * 8u;
                gv[i] = (kk[i] < H) ? *(const uint2*)(grow + kk[i]) : uint2{0, 0};
                uv[i] = (kk[i] < H) ? *(const uint2*)(urow + kk[i]) : uint2{0, 0};
            }
#pragma unroll
            for (int i = 0; i < GV_UNROLL_FP8; i++) {
                if (kk[i] >= H) continue;
                const bf16v8 xv = ld_glob8(xr + kk[i]);
                ag = dot8_fp8(gv[i], xv, ag);
                au = dot8_fp8(uv[i], xv, au);
            }
        }
        const float g = warp_sum32(ag) * scale[n];
        const float u = warp_sum32(au) * scale[I_moe + n];
        if (lane == 0) fu[(size_t)slot * I_moe + n] = __float2bfloat16(plow_moe_gelu_tanh(g) * u);
    }
}

static __device__ void d_moe_expert_down_gemma_fp8(
        float* __restrict__ part, const bf16* __restrict__ fu,
        const unsigned char* __restrict__ table,
        const unsigned long long* __restrict__ ewt,
        const unsigned long long* __restrict__ est, unsigned k, unsigned H, unsigned I_moe,
        unsigned n_exp, unsigned slice, unsigned nblk, unsigned nrow) {
#if PLOW_MOE_DOWN_LANESPLIT
    /* fp8 twin of the lane-split bf16 DOWN arm (round 7): SG rows in flight per warp with one
     * accumulator per lane. Same short-K motivation -- I_moe=704 amortises a 32-lane reduction
     * over almost nothing. Reduction width changes => numerically equivalent, not bit-equal. */
    {
        constexpr unsigned LSG = PLOW_MOE_DOWN_SG, LSL = 32u / PLOW_MOE_DOWN_SG, LCH = LSL * 8u;
        if (nrow == 1u && (I_moe % LCH) == 0u) {
            const unsigned lane_l = threadIdx.x & 31u;
            const unsigned sg = lane_l / LSL, sl = lane_l % LSL;
            const unsigned nw_l = blockDim.x >> 5;
            const unsigned total_l = k * H;
            const unsigned per_l = (total_l + nblk - 1u) / nblk;
            const unsigned f0_l = slice * per_l;
            const unsigned f1_l = (f0_l + per_l < total_l) ? (f0_l + per_l) : total_l;
            const unsigned nch = I_moe / LCH;
            for (unsigned fb = f0_l + (threadIdx.x >> 5) * LSG; fb < f1_l; fb += nw_l * LSG) {
                const unsigned f = fb + sg;
                bool valid = false, live = false;
                float gate = 0.0f, sc = 0.0f;
                float* dst = nullptr;
                const uint8_t* wr = nullptr;
                const bf16* xr = nullptr;
                if (f < f1_l) {
                    valid = true;
                    unsigned slot, h;
                    plow_moe_unflat(f, k, H, 1u, &slot, &h);
                    dst = part + (size_t)slot * H + h;
                    const unsigned eid = plow_moe_slot_expert(table, slot);
                    const unsigned long long wb = (eid < n_exp) ? ewt[(size_t)eid * 2 + 1] : 0ull;
                    const unsigned long long sb = (eid < n_exp) ? est[(size_t)eid * 2 + 1] : 0ull;
                    if (wb != 0ull && sb != 0ull) {
                        live = true;
                        gate = plow_moe_slot_gate(table, slot);
                        sc = ((const float*)(size_t)sb)[h];
                        wr = (const uint8_t*)(size_t)wb + (size_t)h * I_moe;
                        xr = fu + (size_t)slot * I_moe;
                    }
                }
                float acc = 0.0f;
                if (live) {
                    for (unsigned c = 0; c < nch; c++) {
                        const unsigned k0 = c * LCH + sl * 8u;
                        acc = dot8_fp8(*(const uint2*)(wr + k0), ld_glob8(xr + k0), acc);
                    }
                }
#pragma unroll
                for (int o = 1; o < (int)LSL; o <<= 1) acc += __shfl_xor_sync(~0u, acc, o);
                if (sl == 0u && valid) *dst = live ? gate * (acc * sc) : 0.0f;
            }
            return;
        }
    }
#endif
    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned nw = blockDim.x >> 5;
    const unsigned nslot = nrow * k;      /* B>1: part is [B][k][H], fu is [B][k][I_moe] */
    const unsigned total = nslot * H;
    const unsigned per = (total + nblk - 1u) / nblk;
    const unsigned f0 = slice * per;
    const unsigned f1 = (f0 + per < total) ? (f0 + per) : total;
    for (unsigned f = f0 + (threadIdx.x >> 5); f < f1; f += nw) {
        unsigned slot, h;
        plow_moe_unflat(f, nslot, H, nrow, &slot, &h);
        const unsigned eid = plow_moe_slot_expert(table, slot);
        float* pslot = part + (size_t)slot * H;
        const unsigned long long wb = (eid < n_exp) ? ewt[(size_t)eid * 2 + 1] : 0ull;
        const unsigned long long sb = (eid < n_exp) ? est[(size_t)eid * 2 + 1] : 0ull;
        if (wb == 0ull || sb == 0ull) {
            if (lane == 0) pslot[h] = 0.0f;
            continue;
        }
        const uint8_t* Wd = (const uint8_t*)(size_t)wb;
        const float* scale = (const float*)(size_t)sb;
        const float y = plow_warp_dot_fp8_row(fu + (size_t)slot * I_moe,
                                              Wd + (size_t)h * I_moe, I_moe, lane) * scale[h];
        if (lane == 0) pslot[h] = plow_moe_slot_gate(table, slot) * y;
    }
}

/* ---- COMBINE (PLOW_DOP_MOE_COMBINE_GEMMA) — sum the k gate-scaled partials ----------------
 * moe[h] = sum_{slot} part[slot][h], f32-accumulated in fixed slot order, rounded to bf16.
 * The Gemma post-norms / h1+h2 add / residual / layer_scalar are ordinary norm+residual ops. */
static __device__ void d_moe_combine_gemma(bf16* __restrict__ moe, const float* __restrict__ part,
                                    unsigned H, unsigned k, unsigned slice, unsigned nblk) {
    const unsigned tid = slice * blockDim.x + threadIdx.x;
    const unsigned nthr = nblk * blockDim.x;
    for (unsigned h = tid; h < H; h += nthr) {
        float acc = 0.0f;
        for (unsigned slot = 0; slot < k; slot++) acc += part[(size_t)slot * H + h];
        moe[h] = __float2bfloat16(acc);
    }
}

/* ---- FUSED COMBINE + RMSNORM + RESIDUAL (PLOW_DOP_MOE_COMBINE_NORM_GEMMA) ---------------
 * Replaces the 3-op tail chain (MoeCombineGemma → RmsNorm → Residual) with one op, saving
 * 2 counter gates per layer × 30 layers = 60 gates/token. Single-row (decode M=1):
 *   sum[h] = Σ_slot part[slot*H + h]             (f32 combine)
 *   out[h] = sum[h] * rsqrt(mean(sum²)+eps) * gamma[h] + resid[h]    (norm + residual)
 * ONE block (slice!=0 returns); arena holds block_sum scratch.
 * BATCH B>1: ONE BLOCK PER ROW (blocks stride B) — out/resid are [B][H], part is [B][k][H].
 * Each block runs the identical per-row body, so every row is bit-identical to its B=1 result. */
static __device__ void d_moe_combine_norm_gemma(bf16* __restrict__ out,
                                         const float* __restrict__ part,
                                         const bf16* __restrict__ resid,
                                         const bf16* __restrict__ gamma,
                                         unsigned H, unsigned k, float eps,
                                         unsigned slice, unsigned nblk, unsigned nrow,
                                         float* __restrict__ arena) {
    const unsigned tid = threadIdx.x;
    const unsigned nth = blockDim.x;
    const unsigned lane = tid & 31u, warp = tid >> 5;
    __shared__ float red[32];
    const unsigned stride = nblk ? nblk : 1u;

#if PLOW_MOE_COMBINE_ALLBLK
    /* ALL-BLOCK COMBINE (H100 round 8). The body below runs on ONE block -- the row loop is
     * strided by `slice` and decode has nrow==1 -- so every other block gates on a single CTA
     * moving 90 KiB of f32 partials. Measured 0.317 ms of a 6.12 ms step at 2 blocks/SM, and it
     * gets WORSE with more blocks because they all wait.
     * The RMS denominator needs all H, which is why it was never split. So instead every block
     * recomputes the (L2-resident) reduction redundantly -- identical thread mapping, so every
     * block derives a bit-identical `inv` with no cross-block gate -- and then each block writes
     * only ITS OWN slice of the output. No race: the slices are disjoint. Redundant reads cost
     * L2 bandwidth, which is ~100x cheaper here than serialising 263 blocks behind one. */
    if (nrow == 1u) {
        const float* pt = part;
        const bf16* res = resid;
        bf16* o = out;
        float ss = 0.0f;
        if ((H & 3u) == 0u) {
            for (unsigned h = tid * 4u; h < H; h += nth * 4u) {
                float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
                for (unsigned slot = 0; slot < k; slot++) {
                    const float4 v = *(const float4*)(pt + (size_t)slot * H + h);
                    acc4.x += v.x; acc4.y += v.y; acc4.z += v.z; acc4.w += v.w;
                }
                *(float4*)(arena + h) = acc4;
                ss += acc4.x * acc4.x + acc4.y * acc4.y + acc4.z * acc4.z + acc4.w * acc4.w;
            }
        } else {
            for (unsigned h = tid; h < H; h += nth) {
                float acc = 0.0f;
                for (unsigned slot = 0; slot < k; slot++) acc += pt[(size_t)slot * H + h];
                arena[h] = acc;
                ss += acc * acc;
            }
        }
        ss = plow_warp_sum(ss);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float t = 0.0f;
            const unsigned nw = (nth + 31u) >> 5;
            for (unsigned i = 0; i < nw; i++) t += red[i];
            red[0] = rsqrtf(t / (float)H + eps);
        }
        __syncthreads();
        const float inv = red[0];
        /* each block writes only its own disjoint slice */
        const unsigned per_h = (H + stride - 1u) / stride;
        const unsigned h0 = slice * per_h;
        const unsigned h1 = (h0 + per_h < H) ? (h0 + per_h) : H;
        for (unsigned h = h0 + tid; h < h1; h += nth) {
            const float v = arena[h] * inv * __bfloat162float(gamma[h]);
            o[h] = __float2bfloat16(v + __bfloat162float(res[h]));
        }
        __syncthreads();
        return;
    }
#endif
    for (unsigned row = slice; row < nrow; row += stride) {
        const float* pt = part + (size_t)row * k * H;
        const bf16* res = resid + (size_t)row * H;
        bf16* o = out + (size_t)row * H;

        /* Pass 1: combine (Σ slots) and accumulate sum-of-squares for RMS. */
        float ss = 0.0f;
#if PLOW_NV_GEMV_RB
        /* This op runs on ONE block (the row loop is strided by `slice`, and decode has
         * nrow==1), so its 90 KB of f32 partials are moved by 256 threads alone -- measured
         * 0.332 ms of a 7.9 ms step at ~8 GB/s. Reading the partials as float4 cuts the
         * transactions 4x and gives each thread 4 independent slot streams. H % 4 == 0 is
         * guaranteed by the emitter (hidden % 8 == 0) and `part` rows are 16 B aligned.
         * The ss grouping changes, so its rounding does (it was already a parallel reduction). */
        if ((H & 3u) == 0u) {
            for (unsigned h = tid * 4u; h < H; h += nth * 4u) {
                float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
                for (unsigned slot = 0; slot < k; slot++) {
                    const float4 v = *(const float4*)(pt + (size_t)slot * H + h);
                    acc4.x += v.x; acc4.y += v.y; acc4.z += v.z; acc4.w += v.w;
                }
                *(float4*)(arena + h) = acc4;
                ss += acc4.x * acc4.x + acc4.y * acc4.y + acc4.z * acc4.z + acc4.w * acc4.w;
            }
        } else
#endif
        for (unsigned h = tid; h < H; h += nth) {
            float acc = 0.0f;
            for (unsigned slot = 0; slot < k; slot++) acc += pt[(size_t)slot * H + h];
            arena[h] = acc;
            ss += acc * acc;
        }
        /* Block reduction of sum-of-squares. */
        ss = plow_warp_sum(ss);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            const unsigned nw = (nth + 31u) >> 5;
            for (unsigned i = 0; i < nw; i++) s += red[i];
            red[0] = rsqrtf(s / (float)H + eps);
        }
        __syncthreads();
        const float inv = red[0];

        /* Pass 2: norm * gamma + residual, write bf16 output. */
        for (unsigned h = tid; h < H; h += nth) {
            const float v = arena[h] * inv * __bfloat162float(gamma[h]);
            o[h] = __float2bfloat16(v + __bfloat162float(res[h]));
        }
        __syncthreads(); /* arena/red reused by the next row */
    }
}

/* ---- FUSED MoE LAYER TAIL (PLOW_DOP_MOE_COMBINE_RESID_NORM_GEMMA) -----
 * (MoeCombineNormGemma -> NormResidualNorm) in ONE packet: removes a 1-block packet
 * boundary from every layer's critical tail (comb -> NRN -> next-layer QKV).
 * Bit-exact to the pair: b and the new residual are ROUNDED to bf16 before the next
 * reduction reads them, reproducing the two ops' HBM round trips without the traffic.
 *   b  = h1 + RMSNorm(sum_k part, g_pf2)          (op70's output)
 *   x  = (x + RMSNorm(b, g_po)) * ls  (rounded)   (the running residual)
 *   hn = RMSNorm(x, gn)                           (next sublayer's input)
 * arena holds one f32[H] staging row, overwritten pass to pass. */
static __device__ void d_moe_combine_resid_norm_gemma(
        bf16* __restrict__ hn, bf16* __restrict__ x, const float* __restrict__ part,
        const bf16* __restrict__ h1, const bf16* __restrict__ g_pf2,
        const bf16* __restrict__ g_po, const bf16* __restrict__ gn, unsigned H, unsigned k,
        float eps, float ls, unsigned slice, float* __restrict__ arena) {
    if (slice != 0) return;
    const unsigned tid = threadIdx.x;
    const unsigned nth = blockDim.x;
    const unsigned lane = tid & 31u, warp = tid >> 5;
    __shared__ float red[32];
    const unsigned nw = (nth + 31u) >> 5;

#define P9_BLOCK_RED(ssv, out)                                                                     \
    do {                                                                                           \
        float s_ = plow_warp_sum(ssv);                                                             \
        if (lane == 0) red[warp] = s_;                                                             \
        __syncthreads();                                                                           \
        if (tid == 0) {                                                                            \
            float t_ = 0.0f;                                                                       \
            for (unsigned i_ = 0; i_ < nw; i_++) t_ += red[i_];                                    \
            red[0] = rsqrtf(t_ / (float)H + eps);                                                  \
        }                                                                                          \
        __syncthreads();                                                                           \
        out = red[0];                                                                              \
    } while (0)

    /* Pass 1: combine (sum over slots), ss of the f32 combine. */
    float ss = 0.0f;
    for (unsigned h = tid; h < H; h += nth) {
        float acc = 0.0f;
        for (unsigned slot = 0; slot < k; slot++) acc += part[(size_t)slot * H + h];
        arena[h] = acc;
        ss += acc * acc;
    }
    float inv1;
    P9_BLOCK_RED(ss, inv1);

    /* Pass 2: b = bf16(comb*inv1*g_pf2 + h1); ss over the ROUNDED b. */
    ss = 0.0f;
    for (unsigned h = tid; h < H; h += nth) {
        const float v = arena[h] * inv1 * __bfloat162float(g_pf2[h]);
        const bf16 bh = __float2bfloat16(v + __bfloat162float(h1[h]));
        const float bf = __bfloat162float(bh);
        arena[h] = bf;
        ss += bf * bf;
    }
    __syncthreads(); /* red[] reused */
    float inv2;
    P9_BLOCK_RED(ss, inv2);

    /* Pass 3: r = bf16((x + b*inv2*g_po) * ls); ss over the ROUNDED r. */
    ss = 0.0f;
    for (unsigned h = tid; h < H; h += nth) {
        const float v =
            (__bfloat162float(x[h]) + arena[h] * inv2 * __bfloat162float(g_po[h])) * ls;
        const bf16 rh = __float2bfloat16(v);
        const float rf = __bfloat162float(rh);
        arena[h] = rf;
        ss += rf * rf;
    }
    __syncthreads();
    float inv3;
    P9_BLOCK_RED(ss, inv3);

    /* Pass 4: store the new residual and the next-sublayer normed input. */
    for (unsigned h = tid; h < H; h += nth) {
        const float rf = arena[h];
        x[h] = __float2bfloat16(rf);
        hn[h] = __float2bfloat16(rf * inv3 * __bfloat162float(gn[h]));
    }
#undef P9_BLOCK_RED
}

/* ---- ROW-BLOCKED FUSED NORM + EXPERT GLU (PLOW_NV_GEMV_RB, H100 campaign E2) -------------
 * The default body below is the single largest decode op (33 % of the step on 26B/H100) and
 * it leaves ~2.4x on the table for two reasons, both measured
 * (runtime/nvidia/experiments/gemv_lab_h100.cu, perf-data/gemma26b-h100-gemv-mlp.md):
 *   1. it loads wg/wu with SCALAR bf16 (2 B/lane) instead of the 16 B ld_glob8 every other
 *      GEMV arm uses, and with no unroll — so a warp keeps ~1 load in flight where the H100
 *      needs ~16 at 1 block/SM;
 *   2. it recomputes xn = resid*inv*gamma from global for EVERY output channel (5632 channels
 *      x H=2816 of redundant reads per layer).
 * This twin stages xn once per CTA (as f32, so the value fed to each FMA is bit-identical to
 * the default body's) and gives each warp GV_MOE_RB channels = 2*GV_MOE_RB independent weight
 * streams. Measured on the real shape: 837 -> ~2000 GB/s.
 * The per-lane K partition changes (lane-strided -> 8-wide vector chunks), so the warp-sum
 * rounding differs from the default body; outputs are numerically equivalent, not bit-equal.
 * Gated OFF by default => the sm_120 objects are byte-identical; the sm_90a build enables it. */
#if PLOW_NV_GEMV_RB

static __device__ void d_moe_expert_glu_norm_gemma_rb(
        bf16* __restrict__ fu, const bf16* __restrict__ resid, const bf16* __restrict__ gamma,
        const unsigned char* __restrict__ table, const unsigned long long* __restrict__ ewt,
        unsigned k, unsigned I_moe, unsigned H, unsigned n_exp, float eps, unsigned slice,
        unsigned nblk, float* __restrict__ arena) {
    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned nw = blockDim.x >> 5;

    __shared__ float rms_red[32];
    __shared__ float invs[PLOW_MOE_MAXB];
    /* Stage into the DYNAMIC arena, not static smem: the runtime only raises the dynamic
     * limit, so static staging buffers push the per-block total past the carveout and drive
     * cuOccupancyMaxActiveBlocksPerMultiprocessor to 0. The arena is sized by the largest op
     * claim (34432 B on this packet) and each instruction fully consumes it before the next
     * gate, so H*4 <= PLOW_MOE_XN_MAX*4 = 11264 B of it is free scratch here. */
#if PLOW_MOE_XN_BF16
    /* Stage xn as bf16, not f32. dot8_fx reads 32 B of smem per lane per chunk where every
     * other GEMV arm reads 16 B, and the extra smem traffic showed up as MoE-GLU sitting at
     * 2243 GB/s while QKV -- same unroll depth, bf16 x -- reaches 2968. Rounding xn to bf16 is
     * what gemv_rows/gemv_qkv already do with their x, and resid/gamma are bf16 to begin with. */
    bf16* xn_s = (bf16*)arena;
#else
    float* xn_s = arena;
#endif
    plow_moe_row_rms(invs, rms_red, resid, H, 1u, eps);
    {
        const float inv = invs[0];
        for (unsigned h = threadIdx.x; h < H; h += blockDim.x) {
            const float v = __bfloat162float(resid[h]) * inv * __bfloat162float(gamma[h]);
#if PLOW_MOE_XN_BF16
            xn_s[h] = __float2bfloat16(v);
#else
            xn_s[h] = v;
#endif
        }
    }
    __syncthreads();

    const unsigned total = k * I_moe; /* nrow == 1 => nslot == k */
    const unsigned per = (total + nblk - 1u) / nblk;
    const unsigned f0 = slice * per;
    const unsigned f1 = (f0 + per < total) ? (f0 + per) : total;
    const unsigned nchunk = (H + GV_STEP - 1u) / GV_STEP;

    for (unsigned fb = f0 + (threadIdx.x >> 5) * GV_MOE_RB; fb < f1; fb += nw * GV_MOE_RB) {
        /* Per-warp-uniform row descriptors: fb..fb+RB may straddle slots and skipped experts. */
        const bf16* wg[GV_MOE_RB];
        const bf16* wu[GV_MOE_RB];
        unsigned dst[GV_MOE_RB];
        bool ok[GV_MOE_RB];
#pragma unroll
        for (int r = 0; r < GV_MOE_RB; r++) {
            ok[r] = false;
            wg[r] = nullptr;
            wu[r] = nullptr;
            dst[r] = 0;
            const unsigned f = fb + (unsigned)r;
            if (f >= f1) continue;
            unsigned slot, n;
            plow_moe_unflat(f, k, I_moe, 1u, &slot, &n);
            const unsigned eid = plow_moe_slot_expert(table, slot);
            if (eid >= n_exp) continue;
            const unsigned long long gub = ewt[(size_t)eid * 2 + 0];
            if (gub == 0ull) continue;
            const bf16* gu = (const bf16*)(size_t)gub;
            wg[r] = gu + (size_t)n * H;
            wu[r] = gu + (size_t)(I_moe + n) * H;
            dst[r] = slot * I_moe + n;
            ok[r] = true;
        }
        float accg[GV_MOE_RB], accu[GV_MOE_RB];
#pragma unroll
        for (int r = 0; r < GV_MOE_RB; r++) {
            accg[r] = 0.0f;
            accu[r] = 0.0f;
        }
        for (unsigned c = 0; c < nchunk; c += GV_MOE_UN) {
            bf16v8 gv[GV_MOE_RB][GV_MOE_UN], uv[GV_MOE_RB][GV_MOE_UN];
            unsigned kk[GV_MOE_UN];
#pragma unroll
            for (int u = 0; u < GV_MOE_UN; u++) kk[u] = (c + (unsigned)u) * GV_STEP + lane * 8u;
#pragma unroll
            for (int r = 0; r < GV_MOE_RB; r++) {
#pragma unroll
                for (int u = 0; u < GV_MOE_UN; u++) {
                    const bool v = ok[r] && kk[u] < H;
                    gv[r][u] = v ? ld_glob8(wg[r] + kk[u]) : bf16v8_zero();
                    uv[r][u] = v ? ld_glob8(wu[r] + kk[u]) : bf16v8_zero();
                }
            }
#pragma unroll
            for (int u = 0; u < GV_MOE_UN; u++) {
                if (kk[u] >= H) continue;
#if PLOW_MOE_XN_BF16
                const bf16v8 xv = ld_smem8(xn_s + kk[u]);
#pragma unroll
                for (int r = 0; r < GV_MOE_RB; r++) {
                    accg[r] = dot8(gv[r][u], xv, accg[r]);
                    accu[r] = dot8(uv[r][u], xv, accu[r]);
                }
#else
                const float* xp = xn_s + kk[u];
#pragma unroll
                for (int r = 0; r < GV_MOE_RB; r++) {
                    accg[r] = dot8_fx(gv[r][u], xp, accg[r]);
                    accu[r] = dot8_fx(uv[r][u], xp, accu[r]);
                }
#endif
            }
        }
#pragma unroll
        for (int r = 0; r < GV_MOE_RB; r++) {
            const float g = plow_warp_sum(accg[r]);
            const float u = plow_warp_sum(accu[r]);
            if (lane == 0 && ok[r]) fu[dst[r]] = __float2bfloat16(plow_moe_gelu_tanh(g) * u);
        }
    }
}
#endif /* PLOW_NV_GEMV_RB */

/* ---- FUSED NORM + EXPERT GLU (PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA) -----
 * Same as d_moe_expert_glu_gemma but takes the RAW residual + gamma and computes
 * RMS normalization inline, eliminating a separate RmsNorm packet + counter gate.
 * Each CTA redundantly computes inv = rsqrt(mean(resid^2)+eps) — the residual is
 * 2*H bytes (5.6 KB @ H=2816), trivially in L2 after the router read it. */
static __device__ void d_moe_expert_glu_norm_gemma(bf16* __restrict__ fu,
                                            const bf16* __restrict__ resid,
                                            const bf16* __restrict__ gamma,
                                            const unsigned char* __restrict__ table,
                                            const unsigned long long* __restrict__ ewt, unsigned k,
                                            unsigned I_moe, unsigned H, unsigned n_exp, float eps,
                                            unsigned slice, unsigned nblk, unsigned nrow,
                                            float* __restrict__ arena) {
#if PLOW_NV_GEMV_RB
    /* B=1 decode on a stageable H takes the vectorized row-blocked twin. */
    if (nrow == 1u && H <= PLOW_MOE_XN_MAX) {
        d_moe_expert_glu_norm_gemma_rb(fu, resid, gamma, table, ewt, k, I_moe, H, n_exp, eps,
                                       slice, nblk, arena);
        return;
    }
#endif
    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned nw = blockDim.x >> 5;

    /* --- inline RMS norm: inv[r] = rsqrt(mean(resid[r]²) + eps), one per batched row --- */
    __shared__ float rms_red[32];
    __shared__ float invs[PLOW_MOE_MAXB];
    plow_moe_row_rms(invs, rms_red, resid, H, nrow, eps);

    /* --- expert gate/up dots with inline x = resid * inv * gamma --- */
    const unsigned nslot = nrow * k;
    const unsigned total = nslot * I_moe;
    const unsigned per = (total + nblk - 1u) / nblk;
    const unsigned f0 = slice * per;
    const unsigned f1 = (f0 + per < total) ? (f0 + per) : total;

    for (unsigned f = f0 + (threadIdx.x >> 5); f < f1; f += nw) {
        unsigned slot, n;
        plow_moe_unflat(f, nslot, I_moe, nrow, &slot, &n);
        const unsigned row = slot / k;
        const bf16* rr = resid + (size_t)row * H;
        const float inv = invs[row];
        const unsigned eid = plow_moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;
        const unsigned long long gub = ewt[(size_t)eid * 2 + 0];
        if (gub == 0ull) continue;
        const bf16* gu = (const bf16*)(size_t)gub;
        /* dot with inline norm: acc += (resid[h]*inv*gamma[h]) * W[n*H+h] */
        const bf16* wg = gu + (size_t)n * H;
        const bf16* wu = gu + (size_t)(I_moe + n) * H;
        float accg = 0.0f, accu = 0.0f;
        for (unsigned h = lane; h < H; h += 32u) {
            float xn = __bfloat162float(rr[h]) * inv * __bfloat162float(gamma[h]);
            accg = fmaf(xn, __bfloat162float(wg[h]), accg);
            accu = fmaf(xn, __bfloat162float(wu[h]), accu);
        }
        accg = plow_warp_sum(accg);
        accu = plow_warp_sum(accu);
        if (lane == 0) fu[(size_t)slot * I_moe + n] = __float2bfloat16(plow_moe_gelu_tanh(accg) * accu);
    }
}

/* ================================================================================
 * Gemma-4 26B-A4B bf16 grouped-MoE PREFILL bodies (plans/p9-26b-prefill-moe.md).
 *
 * Token-sorted grouped expert GEMM for T>1. Router + align/sort are plain bodies; the two
 * grouped GEMMs reuse op_gemm.cuh's tiled-GEMM helpers (the PGM_ and pgm_ symbols), so they
 * compile only where op_gemm.cuh was included first (interp_sm120.cu and the oracle TU both do).
 * ================================================================================ */

/* ---- T-TOKEN ROUTER (PLOW_DOP_MOE_ROUTER_GEMMA_PF) --------------------------------------
 * Block-per-token loop of the exact decode router. For token t, the block runs the identical
 * weightless-RMS -> h2 -> logits -> softmax -> top-k (lowest-id tie) -> norm_topk -> per-expert
 * scale on resid[t] and writes routing_table[t*k .. t*k+k]. Bit-identical to decode per token. */
static __device__ void d_moe_router_gemma_pf(unsigned char* __restrict__ table,
                                      const bf16* __restrict__ resid, const bf16* __restrict__ proj,
                                      const bf16* __restrict__ scale, const bf16* __restrict__ pes,
                                      unsigned H, unsigned n_exp, unsigned k, unsigned T, float root,
                                      float eps, unsigned slice, unsigned nblk,
                                      float* __restrict__ arena) {
    float* h2 = arena;      /* [H]     */
    float* sc = arena + H;  /* [n_exp] */
    const unsigned tid = threadIdx.x, nth = blockDim.x;
    const unsigned lane = tid & 31u, warp = tid >> 5;
    __shared__ float red[32];

    for (unsigned tok = slice; tok < T; tok += nblk) {
        const bf16* r = resid + (size_t)tok * H;
        unsigned char* tab = table + (size_t)tok * k * 8;

        /* 1. weightless RMS of resid[tok]. */
        float part = 0.0f;
        for (unsigned h = tid; h < H; h += nth) {
            const float v = __bfloat162float(r[h]);
            part += v * v;
        }
        part = plow_warp_sum(part);
        if (lane == 0) red[warp] = part;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            const unsigned nw = (nth + 31u) >> 5;
            for (unsigned i = 0; i < nw; i++) s += red[i];
            red[0] = s;
        }
        __syncthreads();
        const float invrms = rsqrtf(red[0] / (float)H + eps);
        __syncthreads();

        /* 2. h2 = resid * invrms * scale * root. */
        for (unsigned h = tid; h < H; h += nth)
            h2[h] = __bfloat162float(r[h]) * invrms * __bfloat162float(scale[h]) * root;
        __syncthreads();

        /* 3. logits: WARP per expert. The prior thread-per-expert loop read proj[e][h] with an
         * inter-thread stride of H (each thread streamed a whole expert row), so the global loads
         * were fully UNCOALESCED and only n_exp<=128 of 256 threads did any work. Warp-per-expert
         * gives consecutive lanes consecutive h (coalesced), keeps all 8 warps busy, and reduces in
         * f32 (a warp tree instead of a serial chain — a reassociation, still f32 precision). */
        for (unsigned e = warp; e < n_exp; e += (nth >> 5)) {
            const bf16* pr = proj + (size_t)e * H;
            float acc = 0.0f;
            for (unsigned h = lane; h < H; h += 32u)
                acc = fmaf(h2[h], __bfloat162float(pr[h]), acc);
            acc = plow_warp_sum(acc);
            if (lane == 0) sc[e] = acc;
        }
        __syncthreads();

        /* 4. softmax + top-k (lowest-id tie) + norm_topk + per-expert scale, serial on thread 0. */
        if (tid == 0) {
            float m = -1e30f;
            for (unsigned e = 0; e < n_exp; e++) m = fmaxf(m, sc[e]);
            float s = 0.0f;
            for (unsigned e = 0; e < n_exp; e++) { sc[e] = __expf(sc[e] - m); s += sc[e]; }
            for (unsigned e = 0; e < n_exp; e++) sc[e] /= s;
            for (unsigned j = 0; j < k; j++) {
                unsigned long long best = 0ull;
                unsigned bid = 0;
                for (unsigned e = 0; e < n_exp; e++) {
                    unsigned sb;
                    float scv = sc[e];
                    __builtin_memcpy(&sb, &scv, 4);
                    sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
                    const unsigned long long key =
                        ((unsigned long long)sb << 20) |
                        (unsigned long long)((n_exp - 1u - e) & 0xFFFFFu);
                    if (key > best) { best = key; bid = e; }
                }
                *(unsigned*)(tab + (size_t)j * 8) = bid;
                *(float*)(tab + (size_t)j * 8 + 4) = sc[bid];
                sc[bid] = -1e30f;
            }
            float gs = 0.0f;
            for (unsigned j = 0; j < k; j++) gs += *(float*)(tab + (size_t)j * 8 + 4);
            for (unsigned j = 0; j < k; j++) {
                const unsigned win = *(unsigned*)(tab + (size_t)j * 8);
                float gate = *(float*)(tab + (size_t)j * 8 + 4);
                if (gs != 0.0f) gate /= gs;
                gate *= __bfloat162float(pes[win]);
                *(float*)(tab + (size_t)j * 8 + 4) = gate;
            }
        }
        __syncthreads(); /* arena reused next token */
    }
}

/* ---- ALIGN/SORT (PLOW_DOP_MOE_ALIGN_GEMMA_PF) — ONE block ------------------------------
 * Histogram T*k routing slots by expert; padded prefix to PGM_BM tile boundaries; scatter
 * (token, part-row, gate) into expert-contiguous gathered rows. Atomic scatter is order-safe:
 * per-row GEMM math is independent and the combine order is fixed by (token,slot). */
#define PLOW_MOE_MAXE 256u
static __device__ void d_moe_align_gemma_pf(int* __restrict__ meta, const unsigned char* __restrict__ table,
                                     unsigned* __restrict__ row_token, unsigned* __restrict__ row_partidx,
                                     float* __restrict__ row_gate, unsigned T, unsigned n_exp,
                                     unsigned k, unsigned slice) {
    if (slice != 0) return; /* single block */
    const unsigned tid = threadIdx.x, nth = blockDim.x;
    const unsigned BM = (unsigned)PGM_BM;
    __shared__ unsigned cnt[PLOW_MOE_MAXE];
    __shared__ unsigned cur[PLOW_MOE_MAXE];
    __shared__ unsigned s_total_pad;

    for (unsigned e = tid; e < n_exp; e += nth) cnt[e] = 0u;
    __syncthreads();

    const unsigned nslot = T * k;
    for (unsigned idx = tid; idx < nslot; idx += nth) {
        const unsigned eid = plow_moe_slot_expert(table, idx);
        if (eid < n_exp) atomicAdd(&cnt[eid], 1u);
    }
    __syncthreads();

    /* thread 0: padded prefix (tile boundaries), meta, cur init. */
    if (tid == 0) {
        int* rowoff = meta;              /* [n_exp]   padded segment start row */
        int* mcnt = meta + n_exp;        /* [n_exp]   count */
        int* tilep = meta + 2u * n_exp;  /* [n_exp+1] tile prefix */
        unsigned tp = 0u;                /* running tile count */
        for (unsigned e = 0; e < n_exp; e++) {
            tilep[e] = (int)tp;
            rowoff[e] = (int)(tp * BM);
            mcnt[e] = (int)cnt[e];
            cur[e] = tp * BM;            /* scatter cursor = padded start */
            const unsigned tiles = (cnt[e] + BM - 1u) / BM;
            tp += tiles;
        }
        tilep[n_exp] = (int)tp;          /* total_tiles */
        s_total_pad = tp * BM;
    }
    __syncthreads();

    /* pad-init: every gathered row starts UNUSED; scatter overwrites the live ones. */
    const unsigned total_pad = s_total_pad;
    for (unsigned r = tid; r < total_pad; r += nth) {
        row_token[r] = PLOW_EXPERT_UNUSED;
        row_partidx[r] = PLOW_EXPERT_UNUSED;
        row_gate[r] = 0.0f;
    }
    __syncthreads();

    /* scatter. */
    for (unsigned idx = tid; idx < nslot; idx += nth) {
        const unsigned eid = plow_moe_slot_expert(table, idx);
        if (eid >= n_exp) continue;
        const unsigned pos = atomicAdd(&cur[eid], 1u);
        row_token[pos] = idx / k;   /* source token */
        row_partidx[pos] = idx;     /* destination row of part[T*k, H] = token*k + slot */
        row_gate[pos] = plow_moe_slot_gate(table, idx);
    }
}

/* ---- T-ROW COMBINE + SANDWICH (PLOW_DOP_MOE_COMBINE_NORM_GEMMA_PF) ----------------------
 * Block-per-token loop of d_moe_combine_norm_gemma. */
static __device__ void d_moe_combine_norm_gemma_pf(bf16* __restrict__ out, const float* __restrict__ part,
                                            const bf16* __restrict__ h1, const bf16* __restrict__ gamma,
                                            unsigned H, unsigned k, unsigned T, float eps,
                                            unsigned slice, unsigned nblk, float* __restrict__ arena) {
    const unsigned tid = threadIdx.x, nth = blockDim.x;
    const unsigned lane = tid & 31u, warp = tid >> 5;
    __shared__ float red[32];

    for (unsigned tok = slice; tok < T; tok += nblk) {
        const float* pt = part + (size_t)tok * k * H;
        const bf16* res = h1 + (size_t)tok * H;
        bf16* o = out + (size_t)tok * H;

        float ss = 0.0f;
        for (unsigned h = tid; h < H; h += nth) {
            float acc = 0.0f;
            for (unsigned slot = 0; slot < k; slot++) acc += pt[(size_t)slot * H + h];
            arena[h] = acc;
            ss += acc * acc;
        }
        ss = plow_warp_sum(ss);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            const unsigned nw = (nth + 31u) >> 5;
            for (unsigned i = 0; i < nw; i++) s += red[i];
            red[0] = rsqrtf(s / (float)H + eps);
        }
        __syncthreads();
        const float inv = red[0];
        for (unsigned h = tid; h < H; h += nth) {
            const float v = arena[h] * inv * __bfloat162float(gamma[h]);
            o[h] = __float2bfloat16(v + __bfloat162float(res[h]));
        }
        __syncthreads(); /* arena reused next token */
    }
}

#ifdef PGM_BM
/* cp.async-stage a [BM][BK] A tile with a per-row GATHER: gathered output row `rowbase+row`
 * reads source token rowsrc[rowbase+row] (UNUSED -> zero-filled). K-contiguous, K width = k. */
__device__ __forceinline__ void pgm_stage_a_gather(__nv_bfloat16* Ad,
                                                   const __nv_bfloat16* __restrict__ A,
                                                   const unsigned* __restrict__ rowsrc, int tid,
                                                   int rowbase, int kbase, unsigned k) {
    const int KCH = PGM_BK / 8;
    for (int L = tid; L < PGM_BM * KCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / KCH, kk8 = (L % KCH) * 8;
        const unsigned src = rowsrc[rowbase + row];
        const int kk = kbase + kk8;
        const bool in = (src != PLOW_EXPERT_UNUSED) && (kk + 8 <= (int)k);
        const __nv_bfloat16* g = in ? A + (size_t)src * k + kk : A;
        pgm_cp_async_cg16(&Ad[row * PGM_ASTRIDE + kk8], g, in ? 16 : 0);
    }
}

/* find expert e s.t. tilep[e] <= mtile < tilep[e+1] (n_exp<=128, linear scan is cheap). */
__device__ __forceinline__ int pgm_moe_expert_of_mtile(const int* __restrict__ tilep, int mtile,
                                                       int n_exp) {
    int e = 0;
    while (e + 1 < n_exp && tilep[e + 1] <= mtile) e++;
    return e;
}

#if defined(PLOW_NV_HOPPER)
/* ---- HOPPER FORK (sm_90a) ----------------------------------------------------------------
 * The four grouped prefill GEMM bodies below are replaced by warpgroup-MMA (wgmma) twins with
 * BYTE-IDENTICAL signatures; dispatch in interp_sm120.cu is untouched. sm_120a has no wgmma and
 * keeps the mma.sync bodies verbatim in the #else arm. NOTE: op_moe_sm90.cuh RAISES the
 * PGM_ARENA_BF16 claim (its 128B-swizzled BK=64 tiles need 96 KiB + 1 KiB alignment). */
#include "op_moe_sm90.cuh"
#else

/* ---- GROUPED GATE/UP GEMM + GeGLU (PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF) ----------------------
 * Twin of d_gemm_glu: flat tile list over (workitem->(expert,m_tile)) x n_tiles, A gathered
 * from xn2 via row_token, B = fused gate_up ewt[e*2+0] ([2I,H]). Output fu_g[row*I_moe + n]. */
static __device__ void d_moe_group_glu_gemma_pf(__nv_bfloat16* __restrict__ fu,
                                         const __nv_bfloat16* __restrict__ xn2,
                                         const unsigned long long* __restrict__ ewt,
                                         const int* __restrict__ meta,
                                         const unsigned* __restrict__ row_token, unsigned I_moe,
                                         unsigned H, unsigned n_exp, unsigned act, unsigned slice,
                                         unsigned nblk, __nv_bfloat16* arena) {
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const int total_tiles = tilep[n_exp];
    const int tiles_n = ((int)I_moe + PGM_BN - 1) / PGM_BN;
    const int ntiles = total_tiles * tiles_n;
    const unsigned K = H;
    const int ksteps = ((int)K + PGM_BK - 1) / PGM_BK;

    __nv_bfloat16* As = arena;
    __nv_bfloat16* Bgs0 = arena + PGM_GLU_STAGES * PGM_ABUF;
    __nv_bfloat16* Bus0 = Bgs0 + PGM_GLU_STAGES * PGM_BBUF;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int mtile = tile / tiles_n;
        const int ntile = tile % tiles_n;
        const int e = pgm_moe_expert_of_mtile(tilep, mtile, (int)n_exp);
        const int rowbase = rowoff[e] + (mtile - tilep[e]) * PGM_BM;
        const int tn = ntile * PGM_BN;
        const __nv_bfloat16* Wg = (const __nv_bfloat16*)(size_t)ewt[(size_t)e * 2 + 0];
        const __nv_bfloat16* Wu = Wg + (size_t)I_moe * H;

        float accg[PGM_MFRAG][PGM_NFRAG][4], accu[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++)
            for (int j = 0; j < PGM_NFRAG; j++)
                for (int ee = 0; ee < 4; ee++) { accg[i][j][ee] = 0.f; accu[i][j][ee] = 0.f; }

        auto stage = [&](int ks, int buf) {
            pgm_stage_a_gather(As + buf * PGM_ABUF, xn2, row_token, tid, rowbase, ks * PGM_BK, K);
            pgm_stage_b(Bgs0 + buf * PGM_BBUF, Wg, tid, tn, ks * PGM_BK, I_moe, K, (int)K);
            pgm_stage_b(Bus0 + buf * PGM_BBUF, Wu, tid, tn, ks * PGM_BK, I_moe, K, (int)K);
        };
#pragma unroll
        for (int s = 0; s < PGM_GLU_STAGES - 1; s++) {
            if (s < ksteps) { stage(s, s); }
            pgm_cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_GLU_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_GLU_STAGES);
            pgm_cp_commit();
            pgm_cp_wait<PGM_GLU_STAGES - 1>();
            __syncthreads();
            const int cb = ks % PGM_GLU_STAGES;
            __nv_bfloat16* Ad = As + cb * PGM_ABUF;
            __nv_bfloat16* Bgd = Bgs0 + cb * PGM_BBUF;
            __nv_bfloat16* Bud = Bus0 + cb * PGM_BBUF;
#pragma unroll
            for (int kf = 0; kf < PGM_BK; kf += 16) {
                unsigned af[PGM_MFRAG][4];
                pgm_load_afrags(af, Ad, wm, kf, lane);
                unsigned bg[PGM_NFRAG][2], bu[PGM_NFRAG][2];
                pgm_load_bfrags(bg, Bgd, wn, kf, lane);
                pgm_load_bfrags(bu, Bud, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++) {
                        pgm_mma(accg[mi][nj], af[mi], bg[nj], accg[mi][nj]);
                        pgm_mma(accu[mi][nj], af[mi], bu[nj], accu[mi][nj]);
                    }
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int ee = 0; ee < 4; ee++) {
                    int rr = gr + (ee / 2) * 8;
                    int cc = tn + gc + (ee % 2);
                    if (rr < PGM_BM && cc < (int)I_moe) {
                        float g = accg[mi][nj][ee];
                        float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
                        fu[(size_t)(rowbase + rr) * I_moe + cc] =
                            __float2bfloat16(a * accu[mi][nj][ee]);
                    }
                }
            }
        __syncthreads();
    }
}

/* ---- GROUPED DOWN GEMM + gate-scale + scatter (PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF) ---------
 * A = fu_gathered (contiguous per segment), B = down ewt[e*2+1] ([H,I_moe]), N=H K=I_moe.
 * Epilogue multiplies row_gate and SCATTERS to part[row_partidx*H + h]; pad rows skipped. */
static __device__ void d_moe_group_down_gemma_pf(float* __restrict__ part,
                                          const __nv_bfloat16* __restrict__ fu,
                                          const unsigned long long* __restrict__ ewt,
                                          const int* __restrict__ meta,
                                          const unsigned* __restrict__ row_partidx,
                                          const float* __restrict__ row_gate, unsigned H,
                                          unsigned I_moe, unsigned n_exp, unsigned slice,
                                          unsigned nblk, __nv_bfloat16* arena) {
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const int total_tiles = tilep[n_exp];
    const int tiles_n = ((int)H + PGM_BN - 1) / PGM_BN;
    const int ntiles = total_tiles * tiles_n;
    const unsigned K = I_moe;
    const int ksteps = ((int)K + PGM_BK - 1) / PGM_BK;

    __nv_bfloat16* As = arena;
    __nv_bfloat16* Bs = arena + PGM_STAGES * PGM_ABUF;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int mtile = tile / tiles_n;
        const int ntile = tile % tiles_n;
        const int e = pgm_moe_expert_of_mtile(tilep, mtile, (int)n_exp);
        const int rowbase = rowoff[e] + (mtile - tilep[e]) * PGM_BM;
        const int tn = ntile * PGM_BN;
        const __nv_bfloat16* Wd = (const __nv_bfloat16*)(size_t)ewt[(size_t)e * 2 + 1];

        float acc[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++)
            for (int j = 0; j < PGM_NFRAG; j++)
                for (int ee = 0; ee < 4; ee++) acc[i][j][ee] = 0.f;

        auto stage = [&](int ks, int buf) {
            /* A rows contiguous: a_row0 = rowbase, tm = 0, m = PGM_BM (all rows valid). */
            pgm_stage_a(As + buf * PGM_ABUF, fu, tid, 0, ks * PGM_BK, PGM_BM, K, rowbase);
            pgm_stage_b(Bs + buf * PGM_BBUF, Wd, tid, tn, ks * PGM_BK, H, K, (int)K);
        };
#pragma unroll
        for (int s = 0; s < PGM_STAGES - 1; s++) {
            if (s < ksteps) { stage(s, s); }
            pgm_cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_STAGES);
            pgm_cp_commit();
            pgm_cp_wait<PGM_STAGES - 1>();
            __syncthreads();
            const int cb = ks % PGM_STAGES;
            __nv_bfloat16* Ad = As + cb * PGM_ABUF;
            __nv_bfloat16* Bd = Bs + cb * PGM_BBUF;
#pragma unroll
            for (int kf = 0; kf < PGM_BK; kf += 16) {
                unsigned af[PGM_MFRAG][4];
                pgm_load_afrags(af, Ad, wm, kf, lane);
                unsigned bf[PGM_NFRAG][2];
                pgm_load_bfrags(bf, Bd, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++)
                        pgm_mma(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int ee = 0; ee < 4; ee++) {
                    int rr = gr + (ee / 2) * 8;      /* row within tile */
                    int cc = tn + gc + (ee % 2);     /* h in [0,H) */
                    if (rr < PGM_BM && cc < (int)H) {
                        const unsigned pidx = row_partidx[rowbase + rr];
                        if (pidx != PLOW_EXPERT_UNUSED)
                            part[(size_t)pidx * H + cc] = row_gate[rowbase + rr] * acc[mi][nj][ee];
                    }
                }
            }
        __syncthreads();
    }
}

#if PLOW_NV_W8A8
/* ---- fp8 (w8a8) GROUPED prefill GEMMs (beat26b) -----------------------------------------
 * Native w8a8 twins of d_moe_group_{glu,down}_gemma_pf: BOTH operands e4m3, mma.sync.m16n8k32,
 * BK8=64, pgm_sw8 swizzle — reuse the 08a2bdd dense w8a8 mainloop helpers (pgm_stage_a8/b8,
 * pgm_load_*frags_w8a8, pgm_mma_fp8_k32) from op_gemm.cuh. Activation is per-token/-row e4m3 with
 * an f32 scale from QuantFp8; the epilogue dequants acc * a_scale[row] * w_scale[channel]. The
 * expert weight bases come from ewt (e4m3) and their per-output-channel scales from est. */

/* gather e4m3 A rows by row_token into a swizzled [BM][BK8] fp8 tile (K contiguous). Pad rows
 * (row_token==UNUSED) zero-fill (cp.async src_bytes=0), exactly like pgm_stage_a_gather. */
__device__ __forceinline__ void pgm_stage_a8_gather(uint8_t* Ad8, const uint8_t* __restrict__ A,
        const unsigned* __restrict__ rowsrc, int tid, int rowbase, int kbase, unsigned k) {
    const int LCH = PGM_BK8 / 16;
    for (int L = tid; L < PGM_BM * LCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / LCH, kk16 = (L % LCH) * 16;
        const unsigned src = rowsrc[rowbase + row];
        const int kk = kbase + kk16;
        const bool in = (src != PLOW_EXPERT_UNUSED) && (kk + 16 <= (int)k);
        const uint8_t* g = in ? A + (size_t)src * k + kk : A;
        pgm_cp_async_cg16(&Ad8[pgm_sw8(row * PGM_BK8 + kk16)], g, in ? 16 : 0);
    }
}

/* GROUPED gate/up GEMM + GeGLU, w8a8. A gathered e4m3 (xq8 by row_token) + per-token ascale;
 * Wg/Wu e4m3 + per-channel sg/su (sc[cc] gate, sc[I_moe+cc] up). Pad rows write fu=0 (bf16 parity). */
static __device__ void d_moe_group_glu_gemma_pf_w8a8(
        __nv_bfloat16* __restrict__ fu, const uint8_t* __restrict__ xq8,
        const float* __restrict__ ascale, const unsigned long long* __restrict__ ewt,
        const unsigned long long* __restrict__ est, const int* __restrict__ meta,
        const unsigned* __restrict__ row_token, unsigned I_moe, unsigned H, unsigned n_exp,
        unsigned act, unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const int total_tiles = tilep[n_exp];
    const int tiles_n = ((int)I_moe + PGM_BN - 1) / PGM_BN;
    const int ntiles = total_tiles * tiles_n;
    const unsigned K = H;
    const int ksteps = ((int)K + PGM_BK8 - 1) / PGM_BK8;
    uint8_t* As = (uint8_t*)arena;
    uint8_t* Bg = As + PGM_GLU_STAGES * PGM_A8BUF8;
    uint8_t* Bu = Bg + PGM_GLU_STAGES * PGM_B8BUF8;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int mtile = tile / tiles_n, ntile = tile % tiles_n;
        const int e = pgm_moe_expert_of_mtile(tilep, mtile, (int)n_exp);
        const int rowbase = rowoff[e] + (mtile - tilep[e]) * PGM_BM;
        const int tn = ntile * PGM_BN;
        const uint8_t* Wg = (const uint8_t*)(size_t)ewt[(size_t)e * 2 + 0];
        const uint8_t* Wu = Wg + (size_t)I_moe * H;
        const float* sc = (const float*)(size_t)est[(size_t)e * 2 + 0]; /* [2*I_moe] */

        float accg[PGM_MFRAG][PGM_NFRAG][4], accu[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++) for (int j = 0; j < PGM_NFRAG; j++)
            for (int ee = 0; ee < 4; ee++) { accg[i][j][ee]=0.f; accu[i][j][ee]=0.f; }

        auto stage = [&](int ks, int buf) {
            pgm_stage_a8_gather(As + buf * PGM_A8BUF8, xq8, row_token, tid, rowbase, ks * PGM_BK8, K);
            pgm_stage_b8(Bg + buf * PGM_B8BUF8, Wg, tid, tn, ks * PGM_BK8, I_moe, K, (int)K);
            pgm_stage_b8(Bu + buf * PGM_B8BUF8, Wu, tid, tn, ks * PGM_BK8, I_moe, K, (int)K);
        };
#pragma unroll
        for (int s = 0; s < PGM_GLU_STAGES - 1; s++) { if (s < ksteps) stage(s, s); pgm_cp_commit(); }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_GLU_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_GLU_STAGES);
            pgm_cp_commit(); pgm_cp_wait<PGM_GLU_STAGES - 1>(); __syncthreads();
            const int cb = ks % PGM_GLU_STAGES;
#pragma unroll
            for (int kf = 0; kf < PGM_BK8; kf += 32) {
                unsigned af[PGM_MFRAG][4]; pgm_load_afrags_w8a8(af, As + cb * PGM_A8BUF8, wm, kf, lane);
                unsigned bg[PGM_NFRAG][2], bu[PGM_NFRAG][2];
                pgm_load_bfrags_w8a8(bg, Bg + cb * PGM_B8BUF8, wn, kf, lane);
                pgm_load_bfrags_w8a8(bu, Bu + cb * PGM_B8BUF8, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++) {
                        pgm_mma_fp8_k32(accg[mi][nj], af[mi], bg[nj], accg[mi][nj]);
                        pgm_mma_fp8_k32(accu[mi][nj], af[mi], bu[nj], accu[mi][nj]);
                    }
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int ee = 0; ee < 4; ee++) {
                    int rr = gr + (ee / 2) * 8, cc = tn + gc + (ee % 2);
                    if (rr < PGM_BM && cc < (int)I_moe) {
                        const unsigned tok = row_token[rowbase + rr];
                        if (tok == PLOW_EXPERT_UNUSED) {           /* pad row: fu=0 (bf16 parity) */
                            fu[(size_t)(rowbase + rr) * I_moe + cc] = __float2bfloat16(0.f);
                            continue;
                        }
                        const float as = ascale[tok];
                        float g = accg[mi][nj][ee] * as * sc[cc];
                        float u = accu[mi][nj][ee] * as * sc[I_moe + cc];
                        float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
                        fu[(size_t)(rowbase + rr) * I_moe + cc] = __float2bfloat16(a * u);
                    }
                }
            }
        __syncthreads();
    }
}

/* GROUPED down GEMM + gate-scale + scatter, w8a8. A = fu8 (contiguous per segment) + per-row
 * fscale; Wd e4m3 + per-channel dsc[cc]. Epilogue acc * gate * fscale[row] * dsc[h]; pad skipped. */
static __device__ void d_moe_group_down_gemma_pf_w8a8(
        float* __restrict__ part, const uint8_t* __restrict__ fu8, const float* __restrict__ fscale,
        const unsigned long long* __restrict__ ewt, const unsigned long long* __restrict__ est,
        const int* __restrict__ meta, const unsigned* __restrict__ row_partidx,
        const float* __restrict__ row_gate, unsigned H, unsigned I_moe, unsigned n_exp,
        unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const int total_tiles = tilep[n_exp];
    const int tiles_n = ((int)H + PGM_BN - 1) / PGM_BN;
    const int ntiles = total_tiles * tiles_n;
    const unsigned K = I_moe;
    const int ksteps = ((int)K + PGM_BK8 - 1) / PGM_BK8;
    uint8_t* As = (uint8_t*)arena;
    uint8_t* Bs = As + PGM_STAGES * PGM_A8BUF8;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int mtile = tile / tiles_n, ntile = tile % tiles_n;
        const int e = pgm_moe_expert_of_mtile(tilep, mtile, (int)n_exp);
        const int rowbase = rowoff[e] + (mtile - tilep[e]) * PGM_BM;
        const int tn = ntile * PGM_BN;
        const uint8_t* Wd = (const uint8_t*)(size_t)ewt[(size_t)e * 2 + 1];
        const float* dsc = (const float*)(size_t)est[(size_t)e * 2 + 1]; /* [H] */

        float acc[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++) for (int j = 0; j < PGM_NFRAG; j++)
            for (int ee = 0; ee < 4; ee++) acc[i][j][ee]=0.f;

        auto stage = [&](int ks, int buf) {
            pgm_stage_a8(As + buf * PGM_A8BUF8, fu8, tid, 0, ks * PGM_BK8, PGM_BM, K, rowbase);
            pgm_stage_b8(Bs + buf * PGM_B8BUF8, Wd, tid, tn, ks * PGM_BK8, H, K, (int)K);
        };
#pragma unroll
        for (int s = 0; s < PGM_STAGES - 1; s++) { if (s < ksteps) stage(s, s); pgm_cp_commit(); }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_STAGES);
            pgm_cp_commit(); pgm_cp_wait<PGM_STAGES - 1>(); __syncthreads();
            const int cb = ks % PGM_STAGES;
#pragma unroll
            for (int kf = 0; kf < PGM_BK8; kf += 32) {
                unsigned af[PGM_MFRAG][4]; pgm_load_afrags_w8a8(af, As + cb * PGM_A8BUF8, wm, kf, lane);
                unsigned bf[PGM_NFRAG][2]; pgm_load_bfrags_w8a8(bf, Bs + cb * PGM_B8BUF8, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++)
                        pgm_mma_fp8_k32(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int ee = 0; ee < 4; ee++) {
                    int rr = gr + (ee / 2) * 8, cc = tn + gc + (ee % 2);
                    if (rr < PGM_BM && cc < (int)H) {
                        const unsigned pidx = row_partidx[rowbase + rr];
                        if (pidx != PLOW_EXPERT_UNUSED)
                            part[(size_t)pidx * H + cc] =
                                row_gate[rowbase + rr] * fscale[rowbase + rr] * dsc[cc] * acc[mi][nj][ee];
                    }
                }
            }
        __syncthreads();
    }
}
#endif /* PLOW_NV_W8A8 */
#endif /* PLOW_NV_HOPPER fork */
#endif /* PGM_BM */

#endif /* PLOW_NV_OP_MOE_CUH */

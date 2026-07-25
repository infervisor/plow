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
    for (int off = 32; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off, 32);
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

        /* 16 e4m3 = one 16B vector load; 16 bf16 = two 16B vector loads. */
        const uint4 wv = *(const uint4*)(Wrow + kx);
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

/* --- routing table accessors (identical byte layout to the AMD path: 8B per slot) --- */
__device__ __forceinline__ unsigned plow_moe_slot_expert(const unsigned char* table, unsigned slot) {
    return *(const unsigned*)(table + (size_t)slot * 8);
}
__device__ __forceinline__ float plow_moe_slot_gate(const unsigned char* table, unsigned slot) {
    return *(const float*)(table + (size_t)slot * 8 + 4);
}

__device__ __forceinline__ float plow_moe_act(float g, unsigned act) {
    /* act 0 = SiLU (GLM/DeepSeek swiglu), 1 = GELU-tanh. */
    if (act == 0u) return g / (1.0f + __expf(-g));
    const float c = 0.7978845608028654f;
    return 0.5f * g * (1.0f + tanhf(c * (g + 0.044715f * g * g * g)));
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
__global__ void plow_moe_group_glu_fp8_blk(bf16* __restrict__ fu, const bf16* __restrict__ x,
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
        const float g = plow_warp_dot_fp8_blk(xt, Wg + (size_t)n * H, Sg + nrow, H, lane);
        const float u = plow_warp_dot_fp8_blk(xt, Wu + (size_t)n * H, Su + nrow, H, lane);
        if (lane == 0) fu[(size_t)slot * I_moe + n] = __float2bfloat16(plow_moe_act(g, act) * u);
    }
}

/* ================ GROUPED expert DOWN (twin of MOE_GROUP_DOWN_FP8_BLK, op 49) ========
 * One warp per (slot, hidden row h). Writes the GATE-SCALED f32 partial for a fixed slot,
 * so the downstream combine is a deterministic fixed-order sum (skipped slots write 0). */
__global__ void plow_moe_group_down_fp8_blk(float* __restrict__ part, const bf16* __restrict__ fu,
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
__global__ void plow_moe_slot_glu_fp8_blk(bf16* __restrict__ fu, const bf16* __restrict__ x,
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
        const float g = plow_warp_dot_fp8_blk(xt, Wg + (size_t)n * H, Sg + nrow, H, lane);
        const float u = plow_warp_dot_fp8_blk(xt, Wu + (size_t)n * H, Su + nrow, H, lane);
        if (lane == 0) fu[(size_t)slot * I_moe + n] = __float2bfloat16(plow_moe_act(g, act) * u);
    }
}

#endif /* PLOW_NV_OP_MOE_CUH */

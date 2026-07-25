/* sm120_common.cuh — shared device primitives for the sm_120 packet interpreter.
 *
 * This header deliberately INCLUDES the validated op_attention.cuh rather than restating
 * its helpers. op_attention.cuh (flash-decode + merge, relL2 1.7e-03, 1474 GB/s) is
 * trust-high and harvested byte-identical; it already defines bf16v8 / ld_glob8 / dot8 /
 * warp_max32 / warp_sum32 / PLOW_NV_THREADS / PLOW_NV_WARPS. Redefining any of those here
 * would either be an ODR clash or, worse, a SECOND slightly-different reduction. Everything
 * below is strictly ADDITIVE.
 *
 * WAVE64 -> WARP32. Every reduction in runtime/amd/*.h is wave64: `__shfl_xor(v, off, 64)`
 * with off starting at 32, `lane = tid & 63`, `wave = tid >> 6`, and PLOW_WAVES = 8 waves of
 * 64 = 512 threads. Here a warp is 32 lanes and the block is 256 threads = 8 warps of 32.
 * Transliterating the AMD offsets would leave half of every reduction unfolded and produce a
 * silently too-small sum — not a crash. Every reduction below was re-derived, and the test
 * suite carries a negative control that builds the wrong (wave64) offsets on purpose to prove
 * the tests can catch exactly this.
 */
#pragma once
#include "op_attention.cuh"

/* Block geometry. The interpreter launches __launch_bounds__(256,1), so these are fixed. */
#define PLOW_NV_LANE 32u
#define PLOW_NV_LANE_MASK 31u
#define PLOW_NV_WARP_SHIFT 5u

/* 16-byte store, the partner of op_attention.cuh's ld_glob8. */
__device__ __forceinline__ void st_glob8(__nv_bfloat16* p, const bf16v8& v) {
    *(uint4*)p = *(const uint4*)&v;
}
__device__ __forceinline__ bf16v8 bf16v8_zero() {
    bf16v8 r;
    *(uint4*)&r = make_uint4(0u, 0u, 0u, 0u);
    return r;
}

/* Block-wide f32 sum over 8 warps of 32. `part` is >= PLOW_NV_WARPS floats of smem.
 * AMD twin: block_sum() in amd_common.h (4 waves of 64 at 512 threads). */
__device__ __forceinline__ float block_sum(float v, float* part) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    v = warp_sum32(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    /* EVERY thread folds the 8 partials (same ascending order as the old thread-0 fold, so the
     * value is bit-identical) instead of thread 0 folding while 255 threads idle across a
     * second barrier + smem broadcast round trip. 3 barriers -> 2; the trailing barrier stays:
     * part[] is a reused arena slot, no thread may race ahead and clobber it. */
    float r = part[0];
#pragma unroll
    for (unsigned w = 1; w < PLOW_NV_WARPS; w++) r += part[w];
    __syncthreads();
    return r;
}

/* Activations. Must match runtime/amd/op_elementwise.h bit-for-bit in FORM (same fma order),
 * since the golden oracle is f32 and any reassociation shows up in the bf16 comparison. */
__device__ __forceinline__ float act_gelu_tanh(float x) {
    const float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}
/* silu: the IEEE `/` here inlined a ~90-instr FCHK+slow-path-CALL division per element in
 * d_glu/d_gemv_glu (19 -> 9 issue slots/element with the fast reciprocal). <=1ulp f32 shift vs
 * correctly-rounded division, invisible after the bf16 store round on the relL2-gated GLU ops
 * (ARGMAX, the bit-gated op, never sees a silu). */
__device__ __forceinline__ float act_silu(float x) {
    return x * __fdividef(1.0f, 1.0f + __expf(-x));
}
enum { PLOW_ACT_GELU_TANH_ = 0, PLOW_ACT_SILU_ = 1 };

/* ---- argmax packed key --------------------------------------------------------------
 * REPRODUCED BIT-EXACTLY from runtime/amd/op_elementwise.h amax_pack(). The u64 is
 *   [63:32] order-preserving u32 image of the bf16 value
 *   [31:0]  ~index   (complement => a plain unsigned max breaks ties toward the LOWEST index)
 * Any deviation — carrying the index uncomplemented, or widening the key — changes which
 * index wins a tie, and ties are common in a 151936-wide logit row. Hence bit-exactness,
 * not tolerance, is the gate for ARGMAX/ARGMAX_FIN.
 *
 * `b` is the RAW bf16 bit pattern (unsigned short), not a float: __nv_bfloat16 must be
 * bit-cast, never converted. */
__device__ __forceinline__ unsigned long long amax_pack(__nv_bfloat16 bv, unsigned i) {
    const unsigned short b = *(const unsigned short*)&bv;
    const unsigned key = (b & 0x8000u) ? (unsigned)(unsigned short)~b : (unsigned)(b | 0x8000u);
    return ((unsigned long long)key << 32) | (unsigned long long)(~i);
}

__device__ __forceinline__ unsigned long long warp_max_u64(unsigned long long v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { /* 16, NOT the AMD 32: 32 lanes */
        const unsigned long long o = __shfl_xor_sync(0xffffffffu, v, off, 32);
        v = o > v ? o : v;
    }
    return v;
}

__device__ __forceinline__ unsigned long long block_max_u64(unsigned long long v,
                                                            unsigned long long* part) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    v = warp_max_u64(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    /* All-threads fold, 3 barriers -> 2 — see block_sum. max is order-independent anyway. */
    unsigned long long r = part[0];
#pragma unroll
    for (unsigned w = 1; w < PLOW_NV_WARPS; w++) r = part[w] > r ? part[w] : r;
    __syncthreads();
    return r;
}

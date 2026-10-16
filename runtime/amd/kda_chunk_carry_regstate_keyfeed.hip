/* PLOW_KDA_SPANS=1 builds only the all-spans entry point (its own object). */
#ifndef PLOW_KDA_SPANS
#define PLOW_KDA_SPANS 0
#endif
#ifdef PLOW_CONFIG
#include PLOW_CONFIG
#endif
#ifndef PLOW_PACKET_HASH
#define PLOW_PACKET_HASH 0ull
#endif

/* PLOW_KDA_CARRY_XCD (default 1 here): a head's eight V16 tiles on one XCD (op_kda_carry_regstate.h;
 * item -> workgroup map only, bit-identical). =0 is the round-robin rollback. */
#ifndef PLOW_KDA_CARRY_XCD
#define PLOW_KDA_CARRY_XCD 1
#endif

#define PLOW_WG_WAVES 4
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda_carry_regstate.h"

extern "C" __device__ unsigned plow_kda_carry_keyfeed_abi_1 = 1;
extern "C" __device__ unsigned plow_kda_carry_keyfeed_bt64_d128_v128_qpre_1 = 1;
extern "C" __device__ unsigned plow_kda_carry_keyfeed_wave64_1 = 1;
extern "C" __device__ unsigned plow_kda_carry_keyfeed_no_spill_1 = 1;
extern "C" __device__ unsigned plow_kda_carry_keyfeed_static_lds_43520 = 1;
extern "C" __device__ unsigned plow_kda_carry_keyfeed_vgpr_le_256 = 1;
extern "C" __device__ unsigned plow_kda_carry_keyfeed_scratch_pair_bf16_1 = 1;

static_assert(plow_kda_regstate::LDS_BYTES == 43520u, "marker names the LDS footprint");

#if PLOW_PACKET_HASH != 0ull
extern "C" __device__ unsigned plow_packet_hash_lo =
    (unsigned)(PLOW_PACKET_HASH & 0xffffffffull);
extern "C" __device__ unsigned plow_packet_hash_hi =
    (unsigned)(PLOW_PACKET_HASH >> 32);
#endif

/* The register-state carry fed with the lean Wu's scaled-key hi/lo pair ([T][H][D] bf16 each);
 * `scale` is already folded into q. */
#if !PLOW_KDA_SPANS
extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void plow_kda_chunk_carry_regstate_keyfeed_gfx950(
    bf16* __restrict__ out, float* __restrict__ state,
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ W, const bf16* __restrict__ U,
    const float* __restrict__ Aqk, const float* __restrict__ g,
    const bf16* __restrict__ key_hi, const bf16* __restrict__ key_lo,
    unsigned t, unsigned heads, unsigned dim, unsigned value_dim, float scale) {
    __shared__ __align__(16) bf16 lds[plow_kda_regstate::LDS_BYTES / 2u];
    if (t < 512u || heads == 0u || dim != 128u || value_dim != 128u || !(scale > 0.0f)) return;
    d_kda_chunk_carry_bt64_regstate<false, false, true>(
        out, state, q, k, W, U, Aqk, g, (t + 63u) / 64u, t, heads, blockIdx.x, gridDim.x, lds,
        nullptr, key_hi, key_lo);
}
#endif

#if PLOW_KDA_SPANS
/* PLOW_AMD_PACKED_LEAN_KDA_BATCH: every span of a packed launch in one dispatch; workgroup `b`
 * carries span `b / per_span` (item slice `b % per_span`, state at its `state_slot`). */
extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void plow_kda_chunk_carry_regstate_keyfeed_spans_gfx950(
    bf16* __restrict__ out, float* __restrict__ state,
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ W, const bf16* __restrict__ U,
    const float* __restrict__ Aqk, const float* __restrict__ g,
    const bf16* __restrict__ key_hi, const bf16* __restrict__ key_lo,
    unsigned t_unused, unsigned heads, unsigned dim, unsigned value_dim, float scale,
    const PlowPrefillSpan* __restrict__ spans, unsigned n_spans, unsigned per_span,
    unsigned min_rows) {
    __shared__ __align__(16) bf16 lds[plow_kda_regstate::LDS_BYTES / 2u];
    (void)t_unused;
    const unsigned si = blockIdx.x / per_span;
    if (si >= n_spans || heads == 0u || dim != 128u || value_dim != 128u || !(scale > 0.0f))
        return;
    const unsigned t = spans[si].n_rows;
    if (t == 0u || t < min_rows) return;
    const size_t r = spans[si].row0;
    const size_t rd = r * heads * dim, rv = r * heads * value_dim, ra = r * heads * 64u;
    float* ss = state + (size_t)spans[si].state_slot * heads * value_dim * dim;
    d_kda_chunk_carry_bt64_regstate<false, false, true>(
        out + rv, ss, q + rd, k + rd, W + rd, U + rv, Aqk + ra, g + rd, (t + 63u) / 64u, t,
        heads, blockIdx.x - si * per_span, per_span, lds, nullptr, key_hi + rd, key_lo + rd);
}
#endif

#ifndef PLOW_KDA_CHUNK_WU_LEAN_COMMON_H
#define PLOW_KDA_CHUNK_WU_LEAN_COMMON_H

#ifdef PLOW_CONFIG
#include PLOW_CONFIG
#endif
#ifndef PLOW_PACKET_HASH
#define PLOW_PACKET_HASH 0ull
#endif

#define PLOW_WG_WAVES 4
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda_wu_lean.h"

extern "C" __device__ unsigned plow_kda_wu_lean_abi_1 = 1;
extern "C" __device__ unsigned plow_kda_wu_lean_bt64_d128_v128_qpre_1 = 1;
extern "C" __device__ unsigned plow_kda_wu_lean_wave64_1 = 1;
extern "C" __device__ unsigned plow_kda_wu_lean_no_spill_1 = 1;
extern "C" __device__ unsigned plow_kda_wu_lean_static_lds_46080 = 1;
extern "C" __device__ unsigned plow_kda_wu_lean_vgpr_le_256 = 1;

static_assert(plow_kda_wu_lean::LDS_BYTES == 46080u, "marker names the LDS footprint");

#if PLOW_PACKET_HASH != 0ull
extern "C" __device__ unsigned plow_packet_hash_lo =
    (unsigned)(PLOW_PACKET_HASH & 0xffffffffull);
extern "C" __device__ unsigned plow_packet_hash_hi =
    (unsigned)(PLOW_PACKET_HASH >> 32);
#endif

/* One workgroup of four waves per (chunk, head) item; `q` is pre-scaled in place. KEYS also
 * writes the scaled-key hi/lo pair ([T][H][D] bf16 each) the keyfeed carry consumes. */
template <bool KEYS>
__device__ __forceinline__ void plow_kda_chunk_wu_lean_body(
    bf16* __restrict__ W, bf16* __restrict__ U, bf16* __restrict__ q,
    bf16* __restrict__ key_hi, bf16* __restrict__ key_lo, const float* __restrict__ Ainv,
    const bf16* __restrict__ k, const bf16* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, unsigned t, unsigned heads, unsigned dim,
    unsigned value_dim, float scale) {
    __shared__ __align__(16) bf16 lds[plow_kda_wu_lean::LDS_BYTES / 2u];
    if (t == 0u || heads == 0u || dim != 128u || value_dim != 128u || !(scale > 0.0f)) return;
    d_kda_chunk_wu_bt64_lean<KEYS>(W, U, q, key_hi, key_lo, Ainv, k, v, g, beta,
                                   (t + 63u) / 64u, t, heads, scale, blockIdx.x, gridDim.x,
                                   lds);
}

/* PLOW_AMD_PACKED_LEAN_KDA_BATCH: every span of a packed launch in one dispatch; workgroup `b`
 * serves span `b / per_span` as item slice `b % per_span`, operands rebased to the span's rows
 * as the per-span launches rebase them. Spans under `min_rows` are skipped. */
template <bool KEYS>
__device__ __forceinline__ void plow_kda_chunk_wu_lean_spans_body(
    bf16* __restrict__ W, bf16* __restrict__ U, bf16* __restrict__ q,
    bf16* __restrict__ key_hi, bf16* __restrict__ key_lo, const float* __restrict__ Ainv,
    const bf16* __restrict__ k, const bf16* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, unsigned heads, unsigned dim, unsigned value_dim,
    float scale, const PlowPrefillSpan* __restrict__ spans, unsigned n_spans,
    unsigned per_span, unsigned min_rows) {
    __shared__ __align__(16) bf16 lds[plow_kda_wu_lean::LDS_BYTES / 2u];
    const unsigned si = blockIdx.x / per_span;
    if (si >= n_spans || heads == 0u || dim != 128u || value_dim != 128u || !(scale > 0.0f))
        return;
    const unsigned t = spans[si].n_rows;
    if (t == 0u || t < min_rows) return;
    const size_t r = spans[si].row0;
    const size_t rd = r * heads * dim, rv = r * heads * value_dim, rh = r * heads;
    const size_t ra = r * heads * 64u;
    d_kda_chunk_wu_bt64_lean<KEYS>(W + rd, U + rv, q + rd, key_hi ? key_hi + rd : nullptr,
                                   key_lo ? key_lo + rd : nullptr, Ainv + ra, k + rd, v + rv,
                                   g + rd, beta + rh, (t + 63u) / 64u, t, heads, scale,
                                   blockIdx.x - si * per_span, per_span, lds);
}

#endif

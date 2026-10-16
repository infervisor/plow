#ifdef PLOW_CONFIG
#include PLOW_CONFIG
#endif
#ifndef PLOW_PACKET_HASH
#define PLOW_PACKET_HASH 0ull
#endif
#ifndef PLOW_KDA_CARRY_XCD
#define PLOW_KDA_CARRY_XCD 1
#endif

#define PLOW_WG_WAVES 4
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda_carry_regstate.h"

extern "C" __device__ unsigned plow_kda_carry_multi_abi_1 = 1;
extern "C" __device__ unsigned plow_kda_carry_multi_bt64_d128_v128_qpre_1 = 1;

#if PLOW_PACKET_HASH != 0ull
extern "C" __device__ unsigned plow_packet_hash_lo =
    (unsigned)(PLOW_PACKET_HASH & 0xffffffffull);
extern "C" __device__ unsigned plow_packet_hash_hi =
    (unsigned)(PLOW_PACKET_HASH >> 32);
#endif

/* PLOW_KDA_CARRY_MULTI: up to 16 packed spans' serial key-fed carries in one launch (grid
 * H * 8 * n, workgroup block j = span j), instead of one 64-workgroup launch per span. */
struct KdaCarryMultiArgs {
    bf16* out;
    float* state;
    const bf16* q;
    const bf16* k;
    const bf16* W;
    const bf16* U;
    const float* Aqk;
    const float* g;
    const bf16* key_hi;
    const bf16* key_lo;
    unsigned heads, n;
    unsigned span[16][3]; /* row0, rows, state slot */
};

extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void plow_kda_chunk_carry_multi_gfx950(KdaCarryMultiArgs a) {
    __shared__ __align__(16) bf16 lds[plow_kda_regstate::LDS_BYTES / 2u];
    const unsigned items = a.heads * 8u;
    const unsigned j = blockIdx.x / items, slice = blockIdx.x % items;
    if (j >= a.n || j >= 16u) return;
    const unsigned row0 = a.span[j][0], rows = a.span[j][1];
    if (rows == 0u) return;
    const size_t rk = (size_t)row0 * a.heads * 128u, rv = (size_t)row0 * a.heads * 128u;
    d_kda_chunk_carry_bt64_regstate<false, false, true, 0>(
        a.out + rv, a.state + (size_t)a.span[j][2] * (size_t)a.heads * 128u * 128u, a.q + rk,
        a.k + rk, a.W + rk, a.U + rv, a.Aqk + (size_t)row0 * a.heads * 64u, a.g + rk,
        (rows + 63u) / 64u, rows, a.heads, slice, items, lds, nullptr, a.key_hi + rk,
        a.key_lo + rk);
}


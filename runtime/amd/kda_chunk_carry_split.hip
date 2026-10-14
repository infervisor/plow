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

extern "C" __device__ unsigned plow_kda_carry_split_abi_1 = 1;
extern "C" __device__ unsigned plow_kda_carry_split_bt64_d128_v128_qpre_1 = 1;
extern "C" __device__ unsigned plow_kda_carry_split_wave64_1 = 1;
extern "C" __device__ unsigned plow_kda_carry_split_static_lds_43520 = 1;

#if PLOW_PACKET_HASH != 0ull
extern "C" __device__ unsigned plow_packet_hash_lo =
    (unsigned)(PLOW_PACKET_HASH & 0xffffffffull);
extern "C" __device__ unsigned plow_packet_hash_hi =
    (unsigned)(PLOW_PACKET_HASH >> 32);
#endif

/* Sequence-split key-fed carry. The recurrence is affine in the state, so segment p >= 1 runs
 * twice in parallel with segment 0: from a zero state (partial output, partial final state Sloc)
 * and from the identity with U = 0 (Qhat: the output's coefficients on the segment's initial
 * state; M: the segment's state transition). `combine` chains the true segment-start states
 * (Sst) and the final state; `fix` adds Qhat . Sst to the outputs of segments >= 1.
 * scratch: Sloc, M, Sst ([nseg-1][H][V][D] f32 each) then Qhat ([T][H][D] bf16, absolute rows). */
struct KdaCarrySplitArgs {
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
    float* scratch;
    unsigned t, heads, nseg, seg_len;
};

namespace {
constexpr unsigned SD = 128u, SV = 128u, SBT = 64u;

__device__ __forceinline__ size_t split_state_elems(unsigned heads) {
    return (size_t)heads * SV * SD;
}
__device__ __forceinline__ float* split_sloc(const KdaCarrySplitArgs& a, unsigned p) {
    return a.scratch + (size_t)(p - 1u) * split_state_elems(a.heads);
}
__device__ __forceinline__ float* split_m(const KdaCarrySplitArgs& a, unsigned p) {
    return a.scratch + (size_t)(a.nseg - 1u + p - 1u) * split_state_elems(a.heads);
}
__device__ __forceinline__ float* split_sst(const KdaCarrySplitArgs& a, unsigned p) {
    return a.scratch + (size_t)(2u * (a.nseg - 1u) + p - 1u) * split_state_elems(a.heads);
}
__device__ __forceinline__ bf16* split_qhat(const KdaCarrySplitArgs& a) {
    return reinterpret_cast<bf16*>(a.scratch +
                                   (size_t)3u * (a.nseg - 1u) * split_state_elems(a.heads));
}
} // namespace

/* grid = H * 8 * (2 * nseg - 1): job 0 = segment 0 on the real state, job 2p-1 / 2p = segment p
 * from zero / from the identity. */
extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void plow_kda_chunk_carry_split_seg_gfx950(KdaCarrySplitArgs a) {
    __shared__ __align__(16) bf16 lds[plow_kda_regstate::LDS_BYTES / 2u];
    const unsigned items = a.heads * (SV / 16u);
    const unsigned job = blockIdx.x / items, slice = blockIdx.x % items;
    const unsigned p = (job + 1u) / 2u;
    const unsigned row0 = p * a.seg_len;
    if (p >= a.nseg || row0 >= a.t) return;
    const unsigned tp = min(a.seg_len, a.t - row0);
    const unsigned nch = (tp + SBT - 1u) / SBT;
    const size_t rk = (size_t)row0 * a.heads * SD, rv = (size_t)row0 * a.heads * SV;
    const size_t ra = (size_t)row0 * a.heads * SBT;
    const bf16* q = a.q + rk;
    const bf16* k = a.k + rk;
    const bf16* W = a.W + rk;
    const bf16* U = a.U + rv;
    const float* Aqk = a.Aqk + ra;
    const float* g = a.g + rk;
    const bf16* kh = a.key_hi + rk;
    const bf16* kl = a.key_lo + rk;
    if (job == 0u) {
        d_kda_chunk_carry_bt64_regstate<false, false, true, 0>(
            a.out + rv, a.state, q, k, W, U, Aqk, g, nch, tp, a.heads, slice, items, lds,
            nullptr, kh, kl);
    } else if (job & 1u) {
        d_kda_chunk_carry_bt64_regstate<false, false, true, 1>(
            a.out + rv, split_sloc(a, p), q, k, W, U, Aqk, g, nch, tp, a.heads, slice, items,
            lds, nullptr, kh, kl);
    } else {
        d_kda_chunk_carry_bt64_regstate<false, false, true, 2>(
            split_qhat(a) + rk, split_m(a, p), q, k, W, U, Aqk, g, nch, tp, a.heads, slice,
            items, lds, nullptr, kh, kl);
    }
}

/* grid = H * V / 2: two state rows per workgroup, one thread per column. Sst is stored as a
 * bf16 hi/lo pair (hi [H][V][D] then lo) in the segment's f32-sized slot, ready for the fix MFMA. */
extern "C" __global__ __launch_bounds__(256)
void plow_kda_chunk_carry_split_combine_gfx950(KdaCarrySplitArgs a) {
    __shared__ float srow[2][SD];
    const unsigned r = threadIdx.x >> 7, c = threadIdx.x & 127u;
    const unsigned hv = blockIdx.x * 2u + r;
    const unsigned h = hv / SV;
    const size_t off = (size_t)hv * SD;
    const size_t half = split_state_elems(a.heads);
    float s = a.state[off + c];
    for (unsigned p = 1; p < a.nseg; ++p) {
        bf16* sst = reinterpret_cast<bf16*>(split_sst(a, p));
        const bf16 hi = f2bf(s);
        sst[off + c] = hi;
        sst[half + off + c] = f2bf(s - bf2f(hi));
        const float* m = split_m(a, p) + (size_t)h * SV * SD + c;
        float mv[SD];
#pragma unroll
        for (unsigned d = 0; d < SD; ++d) mv[d] = m[d * SD];
        srow[r][c] = s;
        __syncthreads();
        float acc = split_sloc(a, p)[off + c];
#pragma unroll
        for (unsigned d = 0; d < SD; ++d) acc = __builtin_fmaf(srow[r][d], mv[d], acc);
        s = acc;
        __syncthreads();
    }
    a.state[off + c] = s;
}

/* grid = H * ceil((t - seg_len) / 256): o[t][h][v] += sum_d Qhat[t][h][d] Sst[p][h][v][d], with
 * Sst as its bf16 hi/lo pair (two MFMAs). Wave w owns rows [64w, 64w + 64) of the 256-row tile. */
extern "C" __global__ __launch_bounds__(256)
void plow_kda_chunk_carry_split_fix_gfx950(KdaCarrySplitArgs a) {
    const unsigned h = blockIdx.x % a.heads, tile = blockIdx.x / a.heads;
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned token = lane & 15u, kg = lane >> 4;
    const unsigned rw = a.seg_len + tile * 256u + wave * 64u;
    if (rw >= a.t) return;
    const bf16* qh = split_qhat(a);
    const size_t half = split_state_elems(a.heads);
    bf16x8 af[4][4];
    unsigned p[4];
#pragma unroll
    for (unsigned m = 0; m < 4u; ++m) {
        const unsigned row = rw + 16u * m + token;
        const unsigned rc = min(row, a.t - 1u);
        p[m] = min(rw + 16u * m, a.t - 1u) / a.seg_len;
#pragma unroll
        for (unsigned kb = 0; kb < 4u; ++kb) {
            const bf16x8 v = *reinterpret_cast<const bf16x8*>(
                qh + ((size_t)rc * a.heads + h) * SD + 32u * kb + 8u * kg);
            af[m][kb] = row < a.t ? v : (bf16x8)((bf16_t)0);
        }
    }
    for (unsigned n = 0; n < SV / 16u; ++n) {
        f32x4 acc[4];
#pragma unroll
        for (unsigned m = 0; m < 4u; ++m) acc[m] = (f32x4)(0.0f);
        /* Segment boundaries are multiples of 64 rows, so each 16-row m-tile has one segment. */
#pragma unroll
        for (unsigned m = 0; m < 4u; ++m) {
            const bf16* sh = reinterpret_cast<const bf16*>(split_sst(a, p[m])) +
                             ((size_t)h * SV + 16u * n + token) * SD + 8u * kg;
#pragma unroll
            for (unsigned kb = 0; kb < 4u; ++kb) {
                const bf16x8 bh = *reinterpret_cast<const bf16x8*>(sh + 32u * kb);
                const bf16x8 bl = *reinterpret_cast<const bf16x8*>(sh + half + 32u * kb);
                acc[m] = plow_mfma_bf16_16x16(af[m][kb], bh, acc[m]);
                acc[m] = plow_mfma_bf16_16x16(af[m][kb], bl, acc[m]);
            }
        }
#pragma unroll
        for (unsigned m = 0; m < 4u; ++m)
#pragma unroll
            for (unsigned e = 0; e < 4u; ++e) {
                const unsigned row = rw + 16u * m + 4u * kg + e;
                if (row < a.t) {
                    bf16* o = a.out + ((size_t)row * a.heads + h) * SV + 16u * n + token;
                    *o = f2bf(bf2f(*o) + acc[m][e]);
                }
            }
    }
}

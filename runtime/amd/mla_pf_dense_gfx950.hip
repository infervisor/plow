// Dedicated NoPE absorbed-MLA flash prefill (GLM-5.3-Flash / DeepSeek-V4 geometry), gfx950.
// The dense causal twin of d_mla_sparse_pf (mla_sparse_pf.h): same 64-row work item (8 queries x
// 8 heads share one key walk), same wave map and MFMA fragment maps, but the keys are the
// contiguous causal prefix, so there is no union table, no mask words and no gather.
//
// Wave w = (row tile rt = w & 1: queries 4rt..4rt+3, key half kh = w >> 1) over 64-key chunks:
//   S^T = K Q^T for its 32 keys x 32 rows (32 k-steps of v_mfma_f32_32x32x16_bf16), row maxima
//   exchanged with the partner wave, P halves swapped through LDS, then O^T[dv half kh] += V^T P^T
//   over all 64 keys (V = the latent rows themselves).
// Output contract = d_flash_mla_prefill_v2<512, 0> (ns causal KV splits): UNNORMALIZED f32
// partials Opart[row][head][ns][512] and base-2 (m, l) in mlpart[row][head][ns][2], m in
// FA_SCALE'd units. flags bit 0 = the W_ofold epilogue instead (ns == 1): normalized bf16 rows
// [row][head][512] in Opart, mlpart unwritten. m is a lagging frame (true max <= m + 8), which the merge accepts like any
// per-row exponent origin. Not bit-identical to the interpreter body (different reduction order).
//
// LDS: two 64-key stages of two 256-wide latent sub-tiles in the row-pair layout, a P exchange and
// the row statistics: 152,064 B, one workgroup (one wave per SIMD) per CU.
#ifdef PLOW_CONFIG
#include PLOW_CONFIG
#endif
#ifndef PLOW_PACKET_HASH
#define PLOW_PACKET_HASH 0ull
#endif
#define PLOW_WG_WAVES 4
#include "amd_common.h"

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" __device__ unsigned plow_mla_pf_dense_abi_1 = 1;
extern "C" __device__ unsigned plow_mla_pf_dense_nope_h8_d512_1 = 1;
#endif

namespace mla_pfd {
constexpr unsigned QP = 8, NH = 8, DK = 512, BKV = 64;
constexpr unsigned PS = 560;                 // latent row-pair stride (halves)
constexpr unsigned SUB = BKV / 2 * PS;       // one 256-wide latent sub-tile (halves)
constexpr unsigned STAGE = 2 * SUB;          // halves
constexpr unsigned PEX_B = 2 * STAGE * 2;    // byte offsets
constexpr unsigned STATS_B = PEX_B + 4 * 2 * 64 * 16;
constexpr unsigned LDS_BYTES = STATS_B + 4 * 32 * 4;  // 152,064
typedef bf16_t bx4 __attribute__((ext_vector_type(4)));
typedef float fx4 __attribute__((ext_vector_type(4)));
}  // namespace mla_pfd
static_assert(mla_pfd::LDS_BYTES == 152064u, "LDS footprint");
/* LDS fragment loads kept in flight ahead of the consuming MFMA (QK and PV loops). */
#ifndef PLOW_MLA_PFD_PF
#define PLOW_MLA_PFD_PF 4
#endif

__device__ __forceinline__ unsigned mla_pfd_off(unsigned row, unsigned col) {
    return (row >> 1) * mla_pfd::PS + ((col >> 3) * 2u + (row & 1u)) * 8u + (col & 7u);
}

__device__ __forceinline__ mla_pfd::bx4 mla_pfd_tr16(const bf16* p) {
    auto* lp = (mla_pfd::bx4 __attribute__((address_space(3)))*)(void*)p;
    return __builtin_amdgcn_ds_read_tr16_b64_v4bf16(lp);
}

extern "C" __global__ __launch_bounds__(256, 1) void plow_mla_pf_dense_gfx950(
    float* __restrict__ Opart, float* __restrict__ mlpart, const bf16* __restrict__ Q,
    const bf16* __restrict__ Ckv, const int* __restrict__ kv_len,
    const PlowKvSpan* __restrict__ spans, unsigned n_spans, unsigned n_tok, unsigned n_head,
    unsigned kv_stride, unsigned kv_mask, unsigned ns, float scale, unsigned flags) {
#if defined(__HIP_DEVICE_COMPILE__)
    using namespace mla_pfd;
    (void)n_head;
    (void)kv_stride;
    __shared__ __attribute__((aligned(16))) unsigned char lds[LDS_BYTES];
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned g = lane >> 5, r32 = lane & 31u;
    const unsigned g16 = (lane >> 4) & 1u, i16 = lane & 15u;
    const unsigned rt = wave & 1u, kh = wave >> 1, pw = wave ^ 2u;
    const float sl2 = scale * 1.4426950408889634f;
    const unsigned nsp = ns ? ns : 1u;
    bf16* const lh = (bf16*)lds;
    float* const stats = (float*)(lds + STATS_B);
    bf16x8* const pex = (bf16x8*)(lds + PEX_B);
    const unsigned ql = 4u * rt + (r32 >> 3), hh = r32 & 7u;

    // Work item -> (span, query pack, split). Heaviest packs first within a span.
    unsigned w = blockIdx.x;
    unsigned row0 = 0, rows = n_tok, qpos0 = 0, len = 0;
    const bf16* ck = Ckv;
    if (n_spans == 0u) {
        len = (unsigned)kv_len[0];
        qpos0 = len - n_tok;
    } else {
        unsigned s = 0;
        for (; s < n_spans; s++) {
            const unsigned items = (spans[s].n_rows + QP - 1u) / QP * nsp;
            if (w < items) break;
            w -= items;
        }
        if (s == n_spans) return;
        row0 = spans[s].row0;
        rows = spans[s].n_rows;
        qpos0 = spans[s].kv_row0;
        len = spans[s].kv_len;
        ck = Ckv + (size_t)spans[s].kv_base * DK;
    }
    const unsigned n_qt = (rows + QP - 1u) / QP;
    if (w >= n_qt * nsp) return;
    const unsigned qt = n_qt - 1u - w / nsp;
    const unsigned sp = w % nsp;

    // Causal walk of this pack: keys [0, kv_end); split sp takes a BKV-aligned share.
    const unsigned qlast = qt * QP + QP - 1u;
    unsigned kv_end = qpos0 + (qlast < rows ? qlast : rows - 1u) + 1u;
    if (kv_end > len) kv_end = len;
    const unsigned ntl = (kv_end + BKV - 1u) / BKV;
    const unsigned cpt = (ntl + nsp - 1u) / nsp;
    const unsigned walk_lo = sp * cpt * BKV;
    unsigned walk_end = (sp + 1u) * cpt * BKV;
    if (walk_end > kv_end) walk_end = kv_end;
    const unsigned nch = walk_lo < walk_end ? (walk_end - walk_lo + BKV - 1u) / BKV : 0u;
    const unsigned qfirst = qpos0 + qt * QP;  // smallest query position in the pack

    const auto* cg = as_glob(ck);
    auto stage = [&](unsigned c) {
        bf16* st = lh + (c & 1u) * STAGE;
        const unsigned c0 = walk_lo + c * BKV;
#pragma unroll
        for (unsigned k = 0; k < 8; k++) {
            const unsigned p = 8u * wave + k, row = 2u * p + (lane & 1u);
            const unsigned kv = c0 + row < walk_end ? c0 + row : walk_lo;  // finite, masked below
            const size_t kr = (size_t)(kv & kv_mask) * DK;
#pragma unroll
            for (unsigned u = 0; u < 2; u++)
                cp_async16(cg + kr + u * 256u + 8u * (lane >> 1), st + u * SUB + p * PS);
        }
    };

    if (nch) stage(0);

    const unsigned qi = qt * QP + ql;
    const bool qv = qi < rows;
    const size_t qrow = (size_t)(row0 + (qv ? qi : rows - 1u)) * NH + hh;
    const unsigned qpos = qpos0 + (qv ? qi : rows - 1u);
    bf16x8 qf[32];
#pragma unroll
    for (unsigned s = 0; s < 32; s++)
        qf[s] = *(const PLOW_GLOB bf16x8*)(as_glob(Q) + qrow * DK + 16u * s + 8u * g);

    f32x16 o[8];
#pragma unroll
    for (unsigned n = 0; n < 8; n++) o[n] = f32x16{};
    float mrow = -INFINITY, lrow = 0.0f;
    const unsigned R = 32u * kh + r32;  // this lane's K row in the chunk (QK A operand)

    for (unsigned c = 0; c < nch; c++) {
        cp_async_wait();
        __syncthreads();  // B1: chunk c landed; chunk c-1 fully consumed
        if (c + 1 < nch) stage(c + 1);
        const bf16* st = lh + (c & 1u) * STAGE;
        const unsigned c0 = walk_lo + c * BKV;

        f32x16 s = f32x16{};
        {
            const bf16* kp = st + mla_pfd_off(R, 8u * g);
            constexpr unsigned PF = PLOW_MLA_PFD_PF;
            bf16x8 ka[PF];
#pragma unroll
            for (unsigned y = 0; y + 1 < PF; y++)
                ka[y] = *(const bf16x8*)(kp + (y >> 4) * SUB + 32u * (y & 15u));
#pragma unroll
            for (unsigned x = 0; x < 32; x++) {
                if (x + PF - 1 < 32) {
                    const unsigned y = x + PF - 1;
                    ka[y % PF] = *(const bf16x8*)(kp + (y >> 4) * SUB + 32u * (y & 15u));
                }
                s = plow_mfma_bf16_32x32(ka[x % PF], qf[x], s);
            }
        }
        // key k = c0 + 32kh + 4g + (i&3) + 8(i>>2); only the diagonal / ragged chunks mask.
        float mx = -INFINITY;
        if (c0 + BKV - 1u > qfirst || c0 + BKV > walk_end) {
#pragma unroll
            for (unsigned i = 0; i < 16; i++) {
                const unsigned key = c0 + 32u * kh + 4u * g + (i & 3u) + 8u * (i >> 2);
                const float x = (key <= qpos && key < walk_end) ? s[i] * sl2 : -INFINITY;
                s[i] = x;
                mx = fmaxf(mx, x);
            }
        } else {
#pragma unroll
            for (unsigned i = 0; i < 16; i++) {
                s[i] *= sl2;
                mx = fmaxf(mx, s[i]);
            }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 32));
        if (g == 0u) stats[wave * 32u + r32] = mx;
        __syncthreads();  // B2: row maxima visible
        mx = fmaxf(mx, stats[pw * 32u + r32]);
        if (__any(mx > mrow + 8.0f)) {
            const float mnew = fmaxf(mrow, mx);
            const float corr = mnew == -INFINITY ? 1.0f : __builtin_amdgcn_exp2f(mrow - mnew);
            lrow *= corr;
#pragma unroll
            for (unsigned n = 0; n < 8; n++) o[n] *= corr;
            mrow = mnew;
        }
        const float msafe = mrow == -INFINITY ? 0.0f : mrow;
        float ls = 0.0f;
        bf16x8 pf[2];
#pragma unroll
        for (unsigned i = 0; i < 16; i++) {
            const float e = __builtin_amdgcn_exp2f(s[i] - msafe);
            ls += e;
            pf[i >> 3][i & 7u] = (bf16_t)e;
        }
        lrow += ls + __shfl_xor(ls, 32);
        pex[(wave * 2u + 0u) * 64u + lane] = pf[0];
        pex[(wave * 2u + 1u) * 64u + lane] = pf[1];
        __syncthreads();  // B3: both P halves visible
        const bf16x8 pp0 = pex[(pw * 2u + 0u) * 64u + lane];
        const bf16x8 pp1 = pex[(pw * 2u + 1u) * 64u + lane];

        // O^T[dv][row] += V^T P^T; V rows = chunk keys of latent sub-tile kh (dv kh*256..+255).
        const bf16* vp =
            st + kh * SUB + mla_pfd_off(4u * g + (i16 >> 2), g16 * 16u + (i16 & 3u) * 4u);
        auto vfrag = [&](unsigned ks, unsigned n) {
            const bx4 lo = mla_pfd_tr16(vp + ks * 8u * PS + n * 64u);
            const bx4 hi = mla_pfd_tr16(vp + (ks * 8u + 4u) * PS + n * 64u);
            return bf16x8{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
        };
        constexpr unsigned PF = PLOW_MLA_PFD_PF;
        bf16x8 va[PF];
#pragma unroll
        for (unsigned y = 0; y + 1 < PF; y++) va[y] = vfrag(y >> 3, y & 7u);
#pragma unroll
        for (unsigned x = 0; x < 32; x++) {
            const unsigned ks = x >> 3, n = x & 7u;
            if (x + PF - 1 < 32) va[(x + PF - 1) % PF] = vfrag((x + PF - 1) >> 3, (x + PF - 1) & 7u);
            const bool own = (ks >> 1) == kh;
            const bf16x8 pb = (ks & 1u) ? (own ? pf[1] : pp1) : (own ? pf[0] : pp0);
            o[n] = plow_mfma_bf16_32x32(va[x % PF], pb, o[n]);
        }
    }

    __syncthreads();
    if (g == 0u) stats[wave * 32u + r32] = lrow;
    __syncthreads();
    const float lt = lrow + stats[pw * 32u + r32];
    if (qv && (flags & 1u)) {
        // W_ofold epilogue (ns == 1): normalized bf16 latent rows for the fused o-GEMM.
        const float inv = lt > 0.0f ? 1.0f / lt : 0.0f;
        bf16* orow = (bf16*)Opart + (((size_t)row0 + qi) * NH + hh) * DK + kh * 256u;
#pragma unroll
        for (unsigned n = 0; n < 8; n++)
#pragma unroll
            for (unsigned i4 = 0; i4 < 4; i4++) {
                bx4 v;
#pragma unroll
                for (unsigned e = 0; e < 4; e++) v[e] = (bf16_t)(o[n][i4 * 4 + e] * inv);
                *(PLOW_GLOB bx4*)as_glob(orow + n * 32u + 4u * g + 8u * i4) = v;
            }
    } else if (qv) {
        const size_t oh = (((size_t)row0 + qi) * NH + hh) * nsp + sp;
        float* orow = Opart + oh * DK + kh * 256u;
#pragma unroll
        for (unsigned n = 0; n < 8; n++)
#pragma unroll
            for (unsigned i4 = 0; i4 < 4; i4++) {
                fx4 v;
#pragma unroll
                for (unsigned e = 0; e < 4; e++) v[e] = o[n][i4 * 4 + e];
                *(PLOW_GLOB fx4*)as_glob(orow + n * 32u + 4u * g + 8u * i4) = v;
            }
        if (kh == 0u && g == 0u) {
            float* ml = mlpart + oh * 2u;
            ml[0] = mrow == -INFINITY ? -3.0e38f : mrow;
            ml[1] = lt;
        }
    }
#endif
}

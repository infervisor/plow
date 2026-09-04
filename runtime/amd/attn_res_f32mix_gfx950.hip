/* attn_res_f32mix_gfx950.hip — standalone AttnRes phase object with pinned-vLLM 0.28 semantics.
 *
 * One token per workgroup (AR_THREADS lanes, AR_NV vectors of AR_VEC columns each, HID = 7168),
 * persistent over `t += gridDim.x`. Contract (vllm/models/kimi_k3/amd/ops/attn_res.py, BLOCK_L = 1):
 *
 *   prefix  = bf16(res_a + res_b)              (res_pre: bf16(res_pre + bf16(res_a + res_b)))
 *   ring[push_row] = push_src                  (snapshot layers; the mix reads rows [0, nb))
 *   for r in 0..nb, then prefix:               online softmax over BF16 sources
 *       s_r   = sum(x * score_w) * rsqrt(sum(x * x) / HID + eps)
 *       m'    = max(m, s_r);  mixed = mixed * exp(m - m') + exp(s_r - m') * x;  denom likewise
 *   mixed *= 1 / denom                          f32, never rounded
 *   out    = bf16(mixed * rsqrt(sum(mixed^2) / HID + out_eps) * gamma)
 *
 * Reduction order is fixed: AR_NV*AR_VEC sequential FMAs per lane, a DPP row-shift tree per
 * wave, then the wave partials through the same tree. No LDS row staging, no BF16 round trip
 * between the mix and the norm, one row live per lane plus AR_DEPTH in flight.
 *
 * Geometry: 4 waves x 4 columns x 7 vectors, depth 1, 133 VGPRs / occupancy 3 (default). Each
 * token walks nb+2 barriers, so the per-row chain (loads -> DPP sums -> LDS -> barrier -> DPP
 * fold -> exp) is the latency to hide, and hiding it means tokens per CU: three 4-wave
 * workgroups per CU measured 0.260 ms at T8192/nb8 against a 0.221 ms streaming floor, where
 * the 7-wave x 8-column shape (-DAR_THREADS=448u -DAR_VEC=8u -DAR_DEPTH=2 -DAR_MIN_WAVES=4:
 * 98 VGPRs, occupancy 4, two tokens per CU) measured 0.282 ms. The interpreter body is 0.59 ms.
 * Geometry outside the contract poisons `out` with qNaN rather than NOP.
 *
 * `ar_body<F32OUT>` is shared with runtime/bench/amd/attnres_f32mix/oracle.hip, which compiles
 * the same body with an f32 side output to prove the mix against the reference before rounding.
 * Build with -fno-slp-vectorize: packed-f32 FMA formation costs ~30 VGPRs of operand shuffling. */
#ifdef PLOW_CONFIG
#include PLOW_CONFIG
#endif
#ifndef PLOW_PACKET_HASH
#define PLOW_PACKET_HASH 0ull
#endif

#ifndef AR_THREADS
#define AR_THREADS 256u
#endif
#ifndef AR_VEC
#define AR_VEC 4u
#endif
#ifndef AR_DEPTH
#define AR_DEPTH 1 /* rows in flight ahead of the one being reduced: 1 or 2 */
#endif
#ifndef AR_MIN_WAVES
#define AR_MIN_WAVES 3 /* waves per SIMD the register budget is sized for */
#endif
#ifndef PLOW_WG_WAVES
#define PLOW_WG_WAVES (AR_THREADS / 64u)
#endif
#include "amd_common.h"

#define AR_HID 7168u
#define AR_WAVES (AR_THREADS / 64u)
#define AR_NV (AR_HID / (AR_THREADS * AR_VEC)) /* vectors per lane per row */
#define AR_SWEEP (AR_THREADS * AR_VEC)         /* columns per vector sweep */
#define AR_MAXB 8u
static_assert(AR_NV * AR_THREADS * AR_VEC == AR_HID, "lane map must tile the row exactly");
static_assert(AR_VEC == 4u || AR_VEC == 8u, "8- or 16-byte vectors");

typedef bf16 ar_vec __attribute__((ext_vector_type(AR_VEC)));

/* Every global access is `uniform base + 32-bit lane byte offset`. */
__device__ __forceinline__ ar_vec ar_ld(const PLOW_GLOB bf16* base, unsigned byte_off) {
    return *(const PLOW_GLOB ar_vec*)((const PLOW_GLOB char*)base + byte_off);
}
__device__ __forceinline__ ar_vec ar_ld_nt(const PLOW_GLOB bf16* base, unsigned byte_off) {
    return __builtin_nontemporal_load(
        (const PLOW_GLOB ar_vec*)((const PLOW_GLOB char*)base + byte_off));
}
__device__ __forceinline__ void ar_st(PLOW_GLOB bf16* base, unsigned byte_off, ar_vec v) {
    *(PLOW_GLOB ar_vec*)((PLOW_GLOB char*)base + byte_off) = v;
}
__device__ __forceinline__ void ar_load_row(ar_vec (&v)[AR_NV], const PLOW_GLOB bf16* row,
                                            unsigned off0) {
#pragma unroll
    for (unsigned q = 0; q < AR_NV; ++q) v[q] = ar_ld(row, off0 + q * (AR_SWEEP * 2u));
}
/* Ring rows are read exactly once per site; keep them out of L2 so the prefix re-read and the
 * shared score/gamma vectors stay resident. */
__device__ __forceinline__ void ar_load_row_nt(ar_vec (&v)[AR_NV], const PLOW_GLOB bf16* row,
                                               unsigned off0) {
#pragma unroll
    for (unsigned q = 0; q < AR_NV; ++q) v[q] = ar_ld_nt(row, off0 + q * (AR_SWEEP * 2u));
}
__device__ __forceinline__ void ar_store_row(PLOW_GLOB bf16* row, unsigned off0,
                                             const ar_vec (&v)[AR_NV]) {
#pragma unroll
    for (unsigned q = 0; q < AR_NV; ++q) ar_st(row, off0 + q * (AR_SWEEP * 2u), v[q]);
}
__device__ __forceinline__ void ar_copy_row(ar_vec (&d)[AR_NV], const ar_vec (&s)[AR_NV]) {
#pragma unroll
    for (unsigned q = 0; q < AR_NV; ++q) d[q] = s[q];
}

/* Wave64 sum with DPP row shifts + row broadcasts (gfx9 encodings), total read back from lane
 * 63 as a wave-uniform SGPR. Fixed order. Replaces the ds_bpermute tree, whose six LDS round
 * trips per reduction were the per-row latency chain that kept this object off the streaming
 * floor. */
template <int CTRL, int ROW_MASK>
__device__ __forceinline__ float ar_dpp_add(float v) {
    const int x = __builtin_bit_cast(int, v);
    const int y = __builtin_amdgcn_update_dpp(0, x, CTRL, ROW_MASK, 0xf, true);
    return v + __builtin_bit_cast(float, y);
}
__device__ __forceinline__ float ar_wave_sum(float v) {
    v = ar_dpp_add<0x111, 0xf>(v); /* row_shr:1 */
    v = ar_dpp_add<0x112, 0xf>(v); /* row_shr:2 */
    v = ar_dpp_add<0x114, 0xf>(v); /* row_shr:4 */
    v = ar_dpp_add<0x118, 0xf>(v); /* row_shr:8 */
    v = ar_dpp_add<0x142, 0xa>(v); /* row_bcast:15 */
    v = ar_dpp_add<0x143, 0xc>(v); /* row_bcast:31 */
    return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 63));
}

/* Branchless RNE (value-identical to amd_common.h's f2bf over every bit pattern). The branchy
 * form's exec-mask save/restore around the per-token conversions inflated this object's register
 * peak; the megakernel's measurement against the select form does not transfer here. */
__device__ __forceinline__ bf16 ar_f2bf(float f) {
    unsigned u;
    __builtin_memcpy(&u, &f, 4);
    const unsigned rne = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    const unsigned qnan = (u >> 16) | 0x0040u;
    return (bf16)(((u & 0x7fffffffu) > 0x7f800000u) ? qnan : rne);
}

template <bool F32OUT>
__device__ __forceinline__ void ar_body(
    bf16* __restrict__ out, bf16* __restrict__ prefix, bf16* __restrict__ ring,
    const float* __restrict__ score_w, const bf16* __restrict__ push_src,
    const bf16* __restrict__ gamma, const bf16* __restrict__ res_a,
    const bf16* __restrict__ res_b, const bf16* __restrict__ res_pre, unsigned T, unsigned HID,
    unsigned NB, unsigned push_row, unsigned NBCAP, float eps, float out_eps, unsigned reserved,
    float* __restrict__ f32out) {
    __shared__ float part[2][AR_WAVES][2];
    /* score_w is staged in LDS once per workgroup and re-read per row rather than held in
     * registers: those VGPRs buy the rows in flight. */
    __shared__ __align__(16) float wl[AR_HID];
    const unsigned tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    const unsigned c0 = tid * AR_VEC, off0 = c0 * 2u; /* column, byte offset of lane vector 0 */
    auto* og = as_glob(out);

    if (HID != AR_HID || NB > AR_MAXB || NBCAP < NB || (push_src != nullptr && push_row >= NBCAP) ||
        (res_a != nullptr) != (res_b != nullptr) || (res_pre != nullptr && res_a == nullptr) ||
        reserved != 0u) {
        for (unsigned t = blockIdx.x; t < T; t += gridDim.x)
            for (unsigned d = tid; d < HID; d += AR_THREADS)
                st_act1(og + (size_t)t * HID + d, (bf16)0x7fc1u);
        return;
    }

    {
        const auto* sg = as_glob(score_w);
        f32x4* wv = (f32x4*)wl;
        for (unsigned i = tid; i < AR_HID / 4u; i += AR_THREADS)
            wv[i] = *(const PLOW_GLOB f32x4*)(sg + i * 4u);
    }
    __syncthreads();

    const auto* pg = as_glob((const bf16*)prefix);
    const auto* rg = as_glob((const bf16*)ring);
    unsigned k = 0; /* barrier counter: LDS partials alternate buffers across ALL barriers */

    for (unsigned t = blockIdx.x; t < T; t += gridDim.x) {
        const size_t pofs = (size_t)t * AR_HID;
        const size_t bofs = (size_t)t * NBCAP * AR_HID;

        ar_vec pv[AR_NV];
        if (res_a != nullptr) {
            ar_vec a[AR_NV], b[AR_NV], pre[AR_NV];
            ar_load_row(a, as_glob(res_a) + pofs, off0);
            ar_load_row(b, as_glob(res_b) + pofs, off0);
            if (res_pre != nullptr) ar_load_row(pre, as_glob(res_pre) + pofs, off0);
#pragma unroll
            for (unsigned q = 0; q < AR_NV; ++q) {
#pragma unroll
                for (unsigned j = 0; j < AR_VEC; ++j) {
                    const bf16 inner = ar_f2bf(bf2f(a[q][j]) + bf2f(b[q][j]));
                    pv[q][j] = res_pre != nullptr ? ar_f2bf(bf2f(pre[q][j]) + bf2f(inner)) : inner;
                }
            }
            ar_store_row(as_glob(prefix) + pofs, off0, pv);
        } else {
            ar_load_row(pv, pg + pofs, off0);
        }
        if (push_src != nullptr) {
            ar_vec s[AR_NV];
            ar_load_row(s, as_glob(push_src) + pofs, off0);
            ar_store_row(as_glob(ring) + bofs + (size_t)push_row * AR_HID, off0, s);
        }

        /* Compiler fence: the first ring rows are issued only after the materialization has
         * retired its operands, so the a/b/pre vectors and the rows in flight never coexist in
         * registers. Latency is covered by the other tokens resident on the CU. */
        asm volatile("" ::: "memory");

        float m = -INFINITY, denom = 0.0f;
        float mixed[AR_NV][AR_VEC];
#pragma unroll
        for (unsigned q = 0; q < AR_NV; ++q)
#pragma unroll
            for (unsigned j = 0; j < AR_VEC; ++j) mixed[q][j] = 0.0f;

        /* Source r of nb+1: ring rows first, the prefix last. AR_DEPTH rows are kept in flight
         * ahead of the one being reduced; the prefix is re-read at its turn (an L2 hit: this
         * workgroup wrote or read it microseconds ago) rather than held live across the loop. */
        auto src_row = [&](unsigned r, ar_vec (&v)[AR_NV]) {
            if (r < NB) ar_load_row_nt(v, rg + bofs + (size_t)r * AR_HID, off0);
            else ar_load_row(v, pg + pofs, off0);
        };
        ar_vec cur[AR_NV], n1[AR_NV];
        if (NB > 0u) ar_load_row_nt(cur, rg + bofs, off0);
        else ar_copy_row(cur, pv);
#if AR_DEPTH == 2
        if (NB > 0u) src_row(1u, n1);
        else ar_copy_row(n1, cur);
#endif

        for (unsigned r = 0; r <= NB; ++r) {
#if AR_DEPTH == 2
            ar_vec n2[AR_NV];
            if (r + 2u <= NB) src_row(r + 2u, n2);
            else ar_copy_row(n2, n1);
#else
            if (r + 1u <= NB) src_row(r + 1u, n1);
            else ar_copy_row(n1, cur);
#endif

            float ss = 0.0f, sw = 0.0f;
#pragma unroll
            for (unsigned q = 0; q < AR_NV; ++q) {
                const float* wq = wl + c0 + q * AR_SWEEP;
#pragma unroll
                for (unsigned j = 0; j < AR_VEC; ++j) {
                    const float x = bf2f(cur[q][j]);
                    ss = fmaf(x, x, ss);
                    sw = fmaf(x, wq[j], sw);
                }
            }
            ss = ar_wave_sum(ss);
            sw = ar_wave_sum(sw);
            /* Keep the row PACKED across the barrier; the unpacked f32 copies would otherwise
             * stay live for the mix and double the row's register cost. */
#pragma unroll
            for (unsigned q = 0; q < AR_NV; ++q) asm volatile("" : "+v"(cur[q]));
            const unsigned buf = k & 1u;
            if (lane == 0) {
                part[buf][wave][0] = ss;
                part[buf][wave][1] = sw;
            }
            __syncthreads();
            ++k;
            /* Cross-wave fold: lane i < AR_WAVES takes wave i's partial through the same tree. */
            const float tss = ar_wave_sum(lane < AR_WAVES ? part[buf][lane][0] : 0.0f);
            const float tsw = ar_wave_sum(lane < AR_WAVES ? part[buf][lane][1] : 0.0f);
            const float s = tsw * rsqrtf(tss * (1.0f / (float)AR_HID) + eps);
            const float nm = fmaxf(m, s);
            const float os = __expf(m - nm), bs = __expf(s - nm);
            denom = denom * os + bs;
#pragma unroll
            for (unsigned q = 0; q < AR_NV; ++q)
#pragma unroll
                for (unsigned j = 0; j < AR_VEC; ++j)
                    mixed[q][j] = fmaf(bs, bf2f(cur[q][j]), mixed[q][j] * os);
            m = nm;
            ar_copy_row(cur, n1);
#if AR_DEPTH == 2
            ar_copy_row(n1, n2);
#endif
        }
        const float rden = 1.0f / denom;
#pragma unroll
        for (unsigned q = 0; q < AR_NV; ++q)
#pragma unroll
            for (unsigned j = 0; j < AR_VEC; ++j) mixed[q][j] *= rden;

        float inv = 1.0f;
        ar_vec g[AR_NV];
        if (gamma != nullptr) {
            ar_load_row(g, as_glob(gamma), off0);
            float ss = 0.0f;
#pragma unroll
            for (unsigned q = 0; q < AR_NV; ++q)
#pragma unroll
                for (unsigned j = 0; j < AR_VEC; ++j) ss = fmaf(mixed[q][j], mixed[q][j], ss);
            ss = ar_wave_sum(ss);
            const unsigned buf = k & 1u;
            if (lane == 0) part[buf][wave][0] = ss;
            __syncthreads();
            ++k;
            const float tss = ar_wave_sum(lane < AR_WAVES ? part[buf][lane][0] : 0.0f);
            inv = rsqrtf(tss * (1.0f / (float)AR_HID) + out_eps);
        }
        ar_vec o[AR_NV];
#pragma unroll
        for (unsigned q = 0; q < AR_NV; ++q) {
#pragma unroll
            for (unsigned j = 0; j < AR_VEC; ++j) {
                const float y = gamma != nullptr ? mixed[q][j] * inv * bf2f(g[q][j]) : mixed[q][j];
                o[q][j] = ar_f2bf(y);
                if constexpr (F32OUT) f32out[pofs + c0 + q * AR_SWEEP + j] = y;
            }
        }
        ar_store_row(og + pofs, off0, o);
    }
}

#ifndef PLOW_ATTNRES_F32MIX_BODY_ONLY
extern "C" __device__ unsigned plow_attn_res_f32mix_abi_1 = 1;
extern "C" __device__ unsigned plow_attn_res_f32mix_hid7168_1 = 1;
extern "C" __device__ unsigned plow_attn_res_f32mix_online_softmax_1 = 1;
extern "C" __device__ unsigned plow_attn_res_f32mix_wave64_1 = 1;
extern "C" __device__ unsigned plow_attn_res_f32mix_no_spill_1 = 1;
extern "C" __device__ unsigned plow_attn_res_f32mix_vgpr_le_168 = 1;
/* Workgroup size the loader must launch with; read from the object, never assumed. */
extern "C" __device__ unsigned plow_attn_res_f32mix_threads = AR_THREADS;

#if PLOW_PACKET_HASH != 0ull
extern "C" __device__ unsigned plow_packet_hash_lo =
    (unsigned)(PLOW_PACKET_HASH & 0xffffffffull);
extern "C" __device__ unsigned plow_packet_hash_hi =
    (unsigned)(PLOW_PACKET_HASH >> 32);
#endif

/* Kernarg layout == plowrt `AttnResF32MixArgs` (104 B): nine pointers, then
 * t, hid, nb, push_row, nbcap, eps, out_eps, reserved. */
extern "C" __global__ __launch_bounds__(AR_THREADS, AR_MIN_WAVES)
__attribute__((amdgpu_waves_per_eu(AR_MIN_WAVES, 8)))
void plow_attn_res_f32mix_gfx950(
    bf16* __restrict__ out, bf16* __restrict__ prefix, bf16* __restrict__ ring,
    const float* __restrict__ score_w, const bf16* __restrict__ push_src,
    const bf16* __restrict__ gamma, const bf16* __restrict__ res_a,
    const bf16* __restrict__ res_b, const bf16* __restrict__ res_pre, unsigned T, unsigned HID,
    unsigned NB, unsigned push_row, unsigned NBCAP, float eps, float out_eps, unsigned reserved) {
    ar_body<false>(out, prefix, ring, score_w, push_src, gamma, res_a, res_b, res_pre, T, HID, NB,
                   push_row, NBCAP, eps, out_eps, reserved, nullptr);
}
#endif

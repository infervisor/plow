/* op_gemm_fp8_block_sm90.cuh — block-scaled FP8 GEMM for decode rows (sm_90a).
 *
 * out[m][n] = bf16( sum_g  P_g[m][n] * a_scale[g][m] * w_scale[n/128][g] )
 * with x FP8 [M4][K] (1x128 groups, scales a_scale[K/128][M4]), W FP8 [N][K] (128x128 blocks,
 * scales w_scale[N/128][K/128]). P_g is a fresh FP32 WGMMA accumulator over exactly one 128-K
 * group (four k32 MMAs); groups are promoted in a fixed order (ascending, or rotated per tile
 * with ROT) with one fma each (the
 * DeepGEMM two-level accumulation). Split-K cuts at group boundaries (floor(s*G/S)); each split
 * writes FP32 partials [split][M][N] and fp8_block_reduce sums splits in ascending order.
 *
 * Swap-AB: W is the WGMMA A operand (64 N rows per consumer warpgroup), x is B (MT columns), so
 * the decode rows ride the narrow n8/n16/n32 dimension. One producer warp issues 2-D TMA for
 * the W tile (128 N x 128 K, 128-byte swizzle) and the x tile (MT x 128); two consumer
 * warpgroups each take 64 of the tile's 128 N rows. Work items (n tile, split) are assigned by
 * a static grid stride, so a CTA's producer runs ahead into its next item.
 */
#pragma once

#include "sm90_wgmma.cuh"

#define PLOW_FP8B_BN 128
#define PLOW_FP8B_BK 128
#define PLOW_FP8B_THREADS 384

template <unsigned MT>
__device__ __forceinline__ void fp8b_mma(float (&acc)[MT / 2], uint64_t a, uint64_t b, int accumulate) {
    if constexpr (MT == 8) {
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3}, %4, %5, p, 1, 1; }"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "l"(a), "l"(b), "r"(accumulate));
    } else if constexpr (MT == 16) {
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %10, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1; }"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
              "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
            : "l"(a), "l"(b), "r"(accumulate));
    } else if constexpr (MT == 32) {
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %18, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1; }"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
              "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]),
              "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]),
              "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15])
            : "l"(a), "l"(b), "r"(accumulate));
    } else {
        static_assert(MT == 8 || MT == 16 || MT == 32);
    }
}

/* Shared-memory bytes of one launch with `nc` consumer warpgroups (tile = 64*nc N rows):
 * stages x (W tile + x tile, each 1024-aligned) + stages x MT a-scales + full/empty barriers +
 * the fused-reduce flag + the 1024-byte alignment pad. */
__host__ __device__ constexpr unsigned fp8b_smem_bytes(unsigned mt, unsigned stages, unsigned nc = 2) {
    return stages * (64 * nc * PLOW_FP8B_BK + (mt * PLOW_FP8B_BK + 1023) / 1024 * 1024) +
           stages * mt * 4 + stages * 2 * 8 + 16 + 1024;
}

/* Group range of split `s` of `S` over `G` groups. */
__device__ __forceinline__ unsigned fp8b_split_begin(unsigned s, unsigned S, unsigned G) {
    return (unsigned)(((unsigned long long)s * G) / S);
}

/* NC consumer warpgroups each own 64 N rows of a 64*NC-row tile; PIPE overlaps a group's WGMMA
 * with the previous group's promotion (two accumulator sets); FUSED makes the last split CTA of
 * a tile sum the tile's partials in split order (self-resetting per-tile counters in `count`,
 * zero before the first launch) instead of a second reduce launch. ROT starts each tile's split at a
 * different group (rotation 7*tile mod its group count) and promotes in that rotated order: CTAs
 * then stream different x groups instead of all hitting the same L2 lines at once. */
template <unsigned MT, unsigned STAGES, unsigned NC = 2, bool PIPE = false, bool FUSED = false,
          bool ROT = false>
__device__ __forceinline__ void d_gemm_fp8_block(
    __nv_bfloat16* __restrict__ out, float* __restrict__ partial, unsigned* __restrict__ count,
    const void* w_map, const void* x_map,
    const float* __restrict__ a_scale, const float* __restrict__ w_scale,
    unsigned M, unsigned M4, unsigned N, unsigned K, unsigned splits, uint8_t* arena) {
    constexpr unsigned BN = 64 * NC, WB = BN * PLOW_FP8B_BK, XB = MT * PLOW_FP8B_BK;
    /* Every stage and both tiles start 1024-byte aligned (the 128-byte swizzle atom). */
    constexpr unsigned STAGE = WB + (XB + 1023) / 1024 * 1024, ACC = MT / 2;
    uint8_t* base = reinterpret_cast<uint8_t*>(sm90_align1024(arena));
    float* sscale = reinterpret_cast<float*>(base + STAGES * STAGE);
    uint64_t* full = reinterpret_cast<uint64_t*>(sscale + STAGES * MT);
    uint64_t* empty = full + STAGES;
    unsigned* last = reinterpret_cast<unsigned*>(empty + STAGES);
    const unsigned tid = threadIdx.x, wg = tid / 128;
    const unsigned G = K / PLOW_FP8B_BK, ntiles = N / BN, items = ntiles * splits;
    if (tid == 0) {
        for (unsigned s = 0; s < STAGES; ++s) {
            sm90_mbar_init(&full[s], 1);
            sm90_mbar_init(&empty[s], 4 * NC); /* every consumer warp */
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (wg == 0) {
        /* Producer: one thread issues TMA; warp 0 lanes stage the a-scales. */
        if (tid >= 32) return;
        if (tid == 0) { sm90_tmap_prefetch(w_map); sm90_tmap_prefetch(x_map); }
        unsigned iter = 0;
        for (unsigned item = blockIdx.x; item < items; item += gridDim.x) {
            const unsigned tile = item / splits, split = item % splits;
            const unsigned g0 = fp8b_split_begin(split, splits, G), g1 = fp8b_split_begin(split + 1, splits, G);
            const unsigned rot = ROT ? (tile * 7u) % (g1 - g0) : 0u;
            for (unsigned gi = g0; gi < g1; ++gi, ++iter) {
                const unsigned g = ROT ? g0 + (gi - g0 + rot) % (g1 - g0) : gi;
                const unsigned s = iter % STAGES, phase = (iter / STAGES) & 1;
                if (iter >= STAGES) sm90_mbar_wait(&empty[s], phase ^ 1);
                if (tid < MT) sscale[s * MT + tid] = tid < M ? __ldg(a_scale + (size_t)g * M4 + tid) : 0.f;
                __syncwarp();
                if (tid == 0) {
                    uint8_t* stage = base + s * STAGE;
                    sm90_mbar_expect(&full[s], (int)(WB + XB));
                    sm90_tma2d(sm90_su32(stage), w_map, (int)(g * PLOW_FP8B_BK), (int)(tile * BN), sm90_su32(&full[s]));
                    sm90_tma2d(sm90_su32(stage + WB), x_map, (int)(g * PLOW_FP8B_BK), 0, sm90_su32(&full[s]));
                }
                __syncwarp();
            }
        }
        return;
    }

    /* Consumers: warpgroup c = wg - 1 owns rows [64c, 64c + 64) of each N tile. */
    const unsigned c = wg - 1, warp = (tid / 32) & 3, lane = tid & 31;
    unsigned iter = 0;
    for (unsigned item = blockIdx.x; item < items; item += gridDim.x) {
        const unsigned tile = item / splits, split = item % splits;
        const unsigned g0 = fp8b_split_begin(split, splits, G), g1 = fp8b_split_begin(split + 1, splits, G);
        const unsigned rot = ROT ? (tile * 7u) % (g1 - g0) : 0u;
        auto gmap = [&](unsigned gi) { return ROT ? g0 + (gi - g0 + rot) % (g1 - g0) : gi; };
        float sum[ACC], pa[ACC], pb[ACC];
#pragma unroll
        for (unsigned i = 0; i < ACC; ++i) sum[i] = 0.f;
        auto issue = [&](float (&acc)[ACC], unsigned s) {
            uint8_t* stage = base + s * STAGE;
            sm90_wg_fence();
#pragma unroll
            for (unsigned sub = 0; sub < 4; ++sub)
                fp8b_mma<MT>(acc, sm90_desc(stage + c * 64 * PLOW_FP8B_BK + sub * 32),
                             sm90_desc(stage + WB + sub * 32), sub != 0);
            sm90_wg_commit();
        };
        /* Promote a finished group: its scales leave the stage before the stage is released. */
        auto promote = [&](float (&acc)[ACC], unsigned s, float ws) {
            float as[ACC];
#pragma unroll
            for (unsigned i = 0; i < ACC; ++i) as[i] = sscale[s * MT + 8 * (i / 4) + 2 * (lane & 3) + (i & 1)];
            __syncwarp();
            if (lane == 0) sm90_mbar_arrive(&empty[s]);
#pragma unroll
            for (unsigned i = 0; i < ACC; ++i) sum[i] = __fmaf_rn(acc[i], __fmul_rn(as[i], ws), sum[i]);
        };
        if constexpr (!PIPE) {
            for (unsigned g = g0; g < g1; ++g, ++iter) {
                const unsigned s = iter % STAGES, phase = (iter / STAGES) & 1;
                const float ws = __ldg(w_scale + (size_t)(tile * BN / 128) * G + gmap(g));
                sm90_mbar_wait(&full[s], phase);
                issue(pa, s);
                sm90_wg_wait<0>();
                promote(pa, s, ws);
            }
        } else {
            /* Two groups per trip so pa / pb are fixed per call site: ptxas then sees the
             * promoted set disjoint from the one in flight and keeps the wgmma pipeline. */
            auto load = [&](unsigned g, unsigned& s) {
                s = iter % STAGES;
                const unsigned phase = (iter / STAGES) & 1;
                ++iter;
                const float ws = __ldg(w_scale + (size_t)(tile * BN / 128) * G + gmap(g));
                sm90_mbar_wait(&full[s], phase);
                return ws;
            };
            unsigned s, ps;
            float ws, pws = load(g0, ps);
            issue(pa, ps);
            unsigned g = g0 + 1;
            for (; g + 1 < g1; g += 2) {
                ws = load(g, s);
                issue(pb, s);
                sm90_wg_wait<1>();
                promote(pa, ps, pws);
                pws = load(g + 1, ps);
                issue(pa, ps);
                sm90_wg_wait<1>();
                promote(pb, s, ws);
            }
            if (g < g1) {
                ws = load(g, s);
                issue(pb, s);
                sm90_wg_wait<1>();
                promote(pa, ps, pws);
                sm90_wg_wait<0>();
                promote(pb, s, ws);
            } else {
                sm90_wg_wait<0>();
                promote(pa, ps, pws);
            }
        }
        /* reg i = 4*panel + 2*hi + lo: n = 16*warp + lane/4 + 8*hi, m = 8*panel + 2*(lane%4) + lo. */
        const unsigned n0 = tile * BN + c * 64 + warp * 16 + (lane >> 2);
        if (splits == 1) {
#pragma unroll
            for (unsigned i = 0; i < ACC; ++i) {
                const unsigned m = 8 * (i / 4) + 2 * (lane & 3) + (i & 1), n = n0 + 8 * ((i >> 1) & 1);
                if (m < M) out[(size_t)m * N + n] = __float2bfloat16_rn(sum[i]);
            }
            continue;
        }
#pragma unroll
        for (unsigned i = 0; i < ACC; ++i) {
            const unsigned m = 8 * (i / 4) + 2 * (lane & 3) + (i & 1), n = n0 + 8 * ((i >> 1) & 1);
            if (m < M) partial[((size_t)split * M + m) * N + n] = sum[i];
        }
        if constexpr (FUSED) {
            /* Publish this split, count it; the tile's last split sums every split in order. */
            __threadfence();
            asm volatile("bar.sync 1, %0;" :: "n"(128 * NC) : "memory");
            if (tid == 128) {
                const unsigned done = atomicAdd(&count[tile], 1u) + 1;
                *last = done == splits;
                if (done == splits) count[tile] = 0;
            }
            asm volatile("bar.sync 1, %0;" :: "n"(128 * NC) : "memory");
            if (*last) {
                __threadfence();
                const size_t mn = (size_t)M * N;
                for (unsigned e = tid - 128; e < M * BN; e += 128 * NC) {
                    const unsigned m = e / BN, n = tile * BN + e % BN;
                    const size_t at = (size_t)m * N + n;
                    float acc = __ldcg(partial + at);
                    for (unsigned sp = 1; sp < splits; ++sp) acc += __ldcg(partial + sp * mn + at);
                    out[at] = __float2bfloat16_rn(acc);
                }
            }
            asm volatile("bar.sync 1, %0;" :: "n"(128 * NC) : "memory");
        }
    }
}

/* Sum the splits of `partial` [splits][M][N] in ascending order, one BF16 rounding. */
__device__ __forceinline__ void d_fp8_block_reduce(__nv_bfloat16* __restrict__ out,
    const float* __restrict__ partial, unsigned M, unsigned N, unsigned splits) {
    const size_t total = (size_t)M * N;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        float acc = partial[i];
        for (unsigned s = 1; s < splits; ++s) acc += partial[(size_t)s * total + i];
        out[i] = __float2bfloat16_rn(acc);
    }
}

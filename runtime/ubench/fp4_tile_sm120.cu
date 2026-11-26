/* fp4_tile_sm120.cu — single-SM NVFP4 tile loop study (GB10 = sm_121a, any sm_12xa).
 *
 * The W4A4 prefill MoE walk (runtime/nvidia/op_moe_nvfp4_pf.cuh nf4v_tiled) reduced to its k-loop:
 * mma.sync m16n8k64 kind::mxf4nvf4 block_scale scale_vec::4X ue4m3 (SASS OMMA.SF.16864), fragments
 * by ldmatrix.x4 from the same smem stage layout (rows of 64 B, 16 B chunk ^ (row >> 1 & 3), then
 * 8 B of e4m3 scales per row, A rows then B rows), warps 2 (m) x WN (n) of MI m16 x NJ n8.
 *
 * Modes (one kernel per shape x mode, launched at gridDim = 1, cycles from clock64):
 *   0 registers only (fragments loaded once): the tensor pipe with production operand variety;
 *   1 one smem stage resident, ldmatrix + scale LDS per k64, no barrier;
 *   2 as 1 plus a block barrier per k128 step;
 *   3 cp.async ring (NST stages of k128) from global, all threads stage, barrier per step: the
 *     production k-loop. Operand rows are tiles of nkt k128 steps: step ks reads tile ks / nkt at
 *     p + tile * ts + row * kb + (ks % nkt) * 64 (scales at s, strides / 8), so one kernel replays
 *     the production layouts ([E * N][K / 2] rows, tile after tile) or a pre-tiled one (kb = 64:
 *     a stage is one contiguous block), with ts = 0 for an L2-resident operand. pfd > 0: the B rows
 *     of step ks + pfd are prefetched into L2 (cp.async.bulk.prefetch.L2) as step ks is staged;
 *   4 as 3, software-pipelined: half 1's fragments are read before half 0's MMAs and the next
 *     step's half 0 before half 1's; the ring wait and barrier sit between the halves, and the
 *     prefetch goes into the slot that barrier frees (same 2-step prefetch distance as mode 3).
 *
 *   5 the mode 3 ring alone (wait, barrier, stage): the staging rate without fragment reads or MMAs;
 *   6 as 5 without the 8 B scale copies;  7 as 3 without the scale copies (scales stay in smem);
 *   8 as 3 fed by bulk copies (cp.async.bulk + mbarrier, the TMA engine's 1-D form): one thread
 *     copies a pre-tiled stage (A rows, B rows, A scales, B scales: 4 contiguous blocks; the global
 *     blocks carry the smem swizzle) per k128 step;
 *   11 as 3 with the A and B tiles of a stage landed by two TMA 2-D boxes (cp.async.bulk.tensor, u8
 *     [rows][kb] tensor maps, 64 B swizzle = the ring's swizzle) on the slot's mbarrier, one thread
 *     issuing; the 8 B scales by cp.async.
 *
 *   nvcc -arch=sm_121a -O3 -cubin fp4_tile_sm120.cu -o fp4_tile_sm121a.cubin
 */
#include <stdint.h>

struct FtStamp {
    unsigned long long c0, c1, t0, t1;
};

static __device__ __forceinline__ unsigned long long ft_gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __device__ __forceinline__ void ft_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                              unsigned sfa, unsigned sfb) {
    asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

static __device__ __forceinline__ void ft_ldsm(unsigned (&r)[4], unsigned s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
static __device__ __forceinline__ uint2 ft_lds64(unsigned s) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(s));
    return v;
}
static __device__ __forceinline__ unsigned ft_lds32(unsigned s) {
    unsigned v;
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(s));
    return v;
}
static __device__ __forceinline__ void ft_cp16(unsigned s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g));
}
static __device__ __forceinline__ void ft_cp8(unsigned s, const void* g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(g));
}
static __device__ __forceinline__ void ft_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
static __device__ __forceinline__ void ft_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

static __device__ __forceinline__ unsigned ft_swz(unsigned row, unsigned chunk) {
    return row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4);
}

struct FtOperand {
    const uint8_t* p; /* e2m1 bytes */
    const uint8_t* s; /* e4m3 scales */
    unsigned kb, nkt, ts;
};

static __device__ __forceinline__ void ft_pf_l2(const void* p, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void ft_mbar_init(unsigned a, unsigned n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(n));
}
static __device__ __forceinline__ void ft_mbar_expect(unsigned a, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes) : "memory");
}
static __device__ __forceinline__ void ft_mbar_wait(unsigned a, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 " @!p bra W;\n}\n" ::"r"(a), "r"(parity) : "memory");
}
/* TMA 2-D tile load: box at (byte column c0, row c1) of the host-encoded tensor map in global memory */
static __device__ __forceinline__ void ft_tma2d(unsigned dst, const void* map, unsigned c0, unsigned c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
static __device__ __forceinline__ void ft_bulk(unsigned dst, const void* src, unsigned bytes, unsigned bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

template <unsigned MI, unsigned NJ, unsigned WN, unsigned NW, unsigned MODE, unsigned NSTG>
static __device__ __forceinline__ void ft_tile(FtStamp* st, float* sink, const FtOperand A, const FtOperand B,
                                               unsigned nk, unsigned pfd, const void* tmA, const void* tmB) {
    constexpr unsigned NT = NW * 32u, WT = NJ * 8u, BM = (NW / WN) * MI * 16u, BN = WN * WT;
    constexpr unsigned STAGE = (BM + BN) * 72u, NST = MODE >= 3u ? NSTG : 1u;
    extern __shared__ __align__(16) uint8_t ft_smem[];
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned wm = warp / WN, wn = warp % WN, g = lane >> 2;
    /* 1 KiB-aligned ring (the TMA 64 B swizzle is on absolute smem address bits; the launch adds 1 KiB) */
    const unsigned ring = ((unsigned)__cvta_generic_to_shared(ft_smem) + 1023u) & ~1023u;
    const unsigned wr = wm * MI * 16u;
    /* per-lane smem byte offsets inside a stage (swizzle chunk for k64 half 0; half 1 = chunk ^ 2) */
    unsigned offa[MI], offb[NJ / 2u];
#pragma unroll
    for (unsigned mi = 0; mi < MI; mi++) offa[mi] = ft_swz(wr + mi * 16u + (lane & 15u), lane >> 4);
#pragma unroll
    for (unsigned p = 0; p < NJ / 2u; p++)
        offb[p] = BM * 64u + ft_swz(wn * WT + p * 16u + (lane & 7u) + ((lane >> 4) << 3), (lane >> 3) & 1u);
    const unsigned osa = (BM + BN) * 64u + (wr + g + 8u * (lane & 1u)) * 8u;
    const unsigned osb = (BM + BN) * 64u + (BM + wn * WT + g) * 8u;
    constexpr unsigned SAS = 128u, SBS = 64u; /* scale step per m16 / n8 */

    /* staging: 16 B chunks of A rows then B rows, then the 8 B scale rows */
    constexpr unsigned NCH = (BM + BN) * 4u, CPT = (NCH + NT - 1u) / NT, SPT = (BM + BN + NT - 1u) / NT;
    auto at = [](const FtOperand& o, unsigned ks, unsigned row, size_t& d, size_t& sd) {
        const unsigned t = ks / o.nkt, kk = ks - t * o.nkt;
        d = (size_t)t * o.ts + (size_t)row * o.kb + kk * 64u;
        sd = (size_t)t * (o.ts / 8u) + (size_t)row * (o.kb / 8u) + kk * 8u;
    };
    auto stage = [&](unsigned ks, unsigned slot) {
        const unsigned base = ring + slot * STAGE;
#pragma unroll
        for (unsigned j = 0; j < CPT; j++) {
            const unsigned c = tid + j * NT;
            if (NCH % NT == 0u || c < NCH) {
                const unsigned row = c >> 2, ch = c & 3u;
                size_t d, sd;
                at(row < BM ? A : B, ks, row < BM ? row : row - BM, d, sd);
                ft_cp16(base + ft_swz(row, ch), (row < BM ? A.p : B.p) + d + ch * 16u);
            }
        }
#pragma unroll
        for (unsigned j = 0; j < (MODE == 6u || MODE == 7u ? 0u : SPT); j++) {
            const unsigned row = tid + j * NT;
            if ((BM + BN) % NT == 0u || row < BM + BN) {
                size_t d, sd;
                at(row < BM ? A : B, ks, row < BM ? row : row - BM, d, sd);
                ft_cp8(base + (BM + BN) * 64u + row * 8u, (row < BM ? A.s : B.s) + sd);
            }
        }
        if (pfd && ks + pfd < nk) {
#pragma unroll
            for (unsigned j = 0; j < (BN + NT - 1u) / NT; j++) {
                const unsigned row = tid + j * NT;
                if (row < BN) {
                    size_t d, sd;
                    at(B, ks + pfd, row, d, sd);
                    ft_pf_l2(B.p + d, 64u);
                }
            }
        }
    };

    float acc[MI][NJ][4];
#pragma unroll
    for (unsigned i = 0; i < MI; i++)
#pragma unroll
        for (unsigned j = 0; j < NJ; j++)
#pragma unroll
            for (unsigned q = 0; q < 4u; q++) acc[i][j][q] = 0.f;

    /* mode 8: one thread fills a stage with four bulk copies (A, B bytes; A, B scales) completing on
     * the slot's mbarrier; operands pre-tiled (kb = 64, nkt = 1: stage ks at p + ks * ts) */
    __shared__ __align__(8) unsigned long long ft_bar[4];
    const unsigned mb = (unsigned)__cvta_generic_to_shared(ft_bar);
    auto bulk = [&](unsigned ks, unsigned slot) {
        if (tid == 0u) {
            const unsigned base = ring + slot * STAGE, bar = mb + slot * 8u;
            ft_mbar_expect(bar, (BM + BN) * 72u);
            ft_bulk(base, A.p + (size_t)ks * A.ts, BM * 64u, bar);
            ft_bulk(base + BM * 64u, B.p + (size_t)ks * B.ts, BN * 64u, bar);
            ft_bulk(base + (BM + BN) * 64u, A.s + (size_t)ks * (A.ts / 8u), BM * 8u, bar);
            ft_bulk(base + (BM + BN) * 64u + BM * 8u, B.s + (size_t)ks * (B.ts / 8u), BN * 8u, bar);
        }
    };
    /* mode 11: A and B tiles by one TMA 2-D box each (u8 [rows][kb] maps, box 64 B x BM / BN rows, 64 B
     * swizzle = the smem chunk ^ (row >> 1 & 3) layout), scales by cp.async as mode 3 */
    auto tstage = [&](unsigned ks, unsigned slot) {
        const unsigned base = ring + slot * STAGE, bar = mb + slot * 8u;
        if (tid == 0u) {
            ft_mbar_expect(bar, (BM + BN) * 64u);
            const unsigned ta = ks / A.nkt, tb = ks / B.nkt;
            ft_tma2d(base, tmA, (ks - ta * A.nkt) * 64u, ta * (A.ts / A.kb), bar);
            ft_tma2d(base + BM * 64u, tmB, (ks - tb * B.nkt) * 64u, tb * (B.ts / B.kb), bar);
        }
#pragma unroll
        for (unsigned j = 0; j < SPT; j++) {
            const unsigned row = tid + j * NT;
            if ((BM + BN) % NT == 0u || row < BM + BN) {
                size_t d, sd;
                at(row < BM ? A : B, ks, row < BM ? row : row - BM, d, sd);
                ft_cp8(base + (BM + BN) * 64u + row * 8u, (row < BM ? A.s : B.s) + sd);
            }
        }
    };
    if constexpr (MODE < 3u) {
        stage(0u, 0u);
        ft_commit();
        ft_wait<0>();
    } else if constexpr (MODE == 8u || MODE == 11u) {
        if (tid == 0u) {
#pragma unroll
            for (unsigned s = 0; s < NST; s++) ft_mbar_init(mb + s * 8u, 1u);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();
        if constexpr (MODE == 11u) {
#pragma unroll
            for (unsigned s = 0; s < NST - 1u; s++) {
                tstage(s, s);
                ft_commit();
            }
        } else {
#pragma unroll
            for (unsigned s = 0; s < NST - 1u; s++) bulk(s, s);
        }
    } else {
#pragma unroll
        for (unsigned s = 0; s < NST - 1u; s++) {
            stage(s, s);
            ft_commit();
        }
    }
    __syncthreads();
    unsigned long long c0 = 0, t0 = 0;
    if (tid == 0u) {
        t0 = ft_gtime();
        c0 = clock64();
    }
    __syncthreads();

    unsigned af[2][MI][4], bf[2][NJ / 2u][4];
    uint2 sa[MI], sb[NJ];
    auto frags = [&](unsigned base, unsigned half, unsigned buf) {
#pragma unroll
        for (unsigned mi = 0; mi < MI; mi++) ft_ldsm(af[buf][mi], base + (offa[mi] ^ (half << 5)));
#pragma unroll
        for (unsigned p = 0; p < NJ / 2u; p++) ft_ldsm(bf[buf][p], base + (offb[p] ^ (half << 5)));
    };
    auto scales = [&](unsigned base) {
#pragma unroll
        for (unsigned mi = 0; mi < MI; mi++) sa[mi] = ft_lds64(base + osa + mi * SAS);
#pragma unroll
        for (unsigned nj = 0; nj < NJ; nj++) sb[nj] = ft_lds64(base + osb + nj * SBS);
    };
    auto mmas = [&](unsigned half, unsigned buf) {
#pragma unroll
        for (unsigned mi = 0; mi < MI; mi++)
#pragma unroll
            for (unsigned nj = 0; nj < NJ; nj++)
                ft_mma(acc[mi][nj], af[buf][mi], bf[buf][nj >> 1][(nj & 1u) * 2u], bf[buf][nj >> 1][(nj & 1u) * 2u + 1u],
                       half ? sa[mi].y : sa[mi].x, half ? sb[nj].y : sb[nj].x);
    };

    if constexpr (MODE == 0u) {
        scales(ring);
        frags(ring, 0u, 0u);
        frags(ring, 1u, 1u);
        for (unsigned ks = 0; ks < nk; ks++) {
            mmas(0u, 0u);
            mmas(1u, 1u);
        }
    } else if constexpr (MODE <= 2u) {
        for (unsigned ks = 0; ks < nk; ks++) {
            if (MODE == 2u) __syncthreads();
            scales(ring);
#pragma unroll
            for (unsigned h = 0; h < 2u; h++) {
                frags(ring, h, 0u);
                mmas(h, 0u);
            }
        }
    } else if constexpr (MODE == 8u) {
        for (unsigned ks = 0; ks < nk; ks++) {
            ft_mbar_wait(mb + (ks % NST) * 8u, (ks / NST) & 1u);
            __syncthreads();
            if (ks + NST - 1u < nk) bulk(ks + NST - 1u, (ks + NST - 1u) % NST);
            const unsigned base = ring + (ks % NST) * STAGE;
            scales(base);
#pragma unroll
            for (unsigned h = 0; h < 2u; h++) {
                frags(base, h, 0u);
                mmas(h, 0u);
            }
        }
    } else if constexpr (MODE == 11u) {
        for (unsigned ks = 0; ks < nk; ks++) {
            ft_wait<NST - 2u>();
            ft_mbar_wait(mb + (ks % NST) * 8u, (ks / NST) & 1u);
            __syncthreads();
            if (ks + NST - 1u < nk) tstage(ks + NST - 1u, (ks + NST - 1u) % NST);
            ft_commit();
            const unsigned base = ring + (ks % NST) * STAGE;
            scales(base);
#pragma unroll
            for (unsigned h = 0; h < 2u; h++) {
                frags(base, h, 0u);
                mmas(h, 0u);
            }
        }
    } else if constexpr (MODE == 5u || MODE == 6u) {
        for (unsigned ks = 0; ks < nk; ks++) {
            ft_wait<NST - 2u>();
            __syncthreads();
            if (ks + NST - 1u < nk) stage(ks + NST - 1u, (ks + NST - 1u) % NST);
            ft_commit();
        }
    } else if constexpr (MODE == 3u || MODE == 7u) {
        for (unsigned ks = 0; ks < nk; ks++) {
            ft_wait<NST - 2u>();
            __syncthreads();
            if (ks + NST - 1u < nk) stage(ks + NST - 1u, (ks + NST - 1u) % NST);
            ft_commit();
            const unsigned base = ring + (ks % NST) * STAGE;
            scales(base);
#pragma unroll
            for (unsigned h = 0; h < 2u; h++) {
                frags(base, h, 0u);
                mmas(h, 0u);
            }
        }
    } else {
        /* Half 1's scales and fragments are read at the top of step ks, before half 0's MMAs; between
         * the halves the ring wait (stage ks + 1) and the barrier, after which no warp reads slot
         * ks % NST again: stage ks + NST goes there (step 0 also issues stage NST - 1 into the slot
         * the prologue left free). Same prefetch distance as mode 3 (2 steps), no load latency at
         * the top of a step. */
        unsigned sx[2][MI], sy[2][NJ];
        auto sc = [&](unsigned base, unsigned half) {
#pragma unroll
            for (unsigned mi = 0; mi < MI; mi++) sx[half][mi] = ft_lds32(base + osa + mi * SAS + 4u * half);
#pragma unroll
            for (unsigned nj = 0; nj < NJ; nj++) sy[half][nj] = ft_lds32(base + osb + nj * SBS + 4u * half);
        };
        auto mm = [&](unsigned h) {
#pragma unroll
            for (unsigned mi = 0; mi < MI; mi++)
#pragma unroll
                for (unsigned nj = 0; nj < NJ; nj++)
                    ft_mma(acc[mi][nj], af[h][mi], bf[h][nj >> 1][(nj & 1u) * 2u], bf[h][nj >> 1][(nj & 1u) * 2u + 1u],
                           sx[h][mi], sy[h][nj]);
        };
        ft_wait<NST - 2u>();
        __syncthreads();
        sc(ring, 0u);
        frags(ring, 0u, 0u);
        for (unsigned ks = 0; ks < nk; ks++) {
            const unsigned base = ring + (ks % NST) * STAGE;
            sc(base, 1u);
            frags(base, 1u, 1u);
            mm(0u);
            if (ks + 1u < nk) {
                if (ks == 0u) ft_wait<NST - 3u>();
                else ft_wait<NST - 2u>();
                __syncthreads();
                if (ks == 0u) {
                    if (NST - 1u < nk) stage(NST - 1u, NST - 1u);
                    ft_commit();
                }
                if (ks + NST < nk) stage(ks + NST, ks % NST);
                ft_commit();
                const unsigned nb = ring + ((ks + 1u) % NST) * STAGE;
                sc(nb, 0u);
                frags(nb, 0u, 0u);
            }
            mm(1u);
        }
    }

    __syncthreads();
    if (tid == 0u) {
        st[blockIdx.x].c0 = c0;
        st[blockIdx.x].c1 = clock64();
        st[blockIdx.x].t0 = t0;
        st[blockIdx.x].t1 = ft_gtime();
    }
    float s = 0.f;
#pragma unroll
    for (unsigned i = 0; i < MI; i++)
#pragma unroll
        for (unsigned j = 0; j < NJ; j++) s += acc[i][j][0] + acc[i][j][1] + acc[i][j][2] + acc[i][j][3];
    if (s == 1.2345f) sink[tid] = s;
}

/* ft_<BM>x<BN>_w<NW>_m<MODE>: block tile BM x BN, NW warps (2 x NW/2 when NW > 1). */
#define FT_K(MI, NJ, WN, NW, MODE, NST, NAME)                                                              \
    extern "C" __global__ void __launch_bounds__(NW * 32, 1)                                              \
        NAME(FtStamp* st, float* sink, const FtOperand A, const FtOperand B, unsigned nk, unsigned pfd,    \
             const void* tmA, const void* tmB) {                                                          \
        ft_tile<MI, NJ, WN, NW, MODE, NST>(st, sink, A, B, nk, pfd, tmA, tmB);                            \
    }
#define FT_SHAPE(MI, NJ, WN, NW, TAG)                       \
    FT_K(MI, NJ, WN, NW, 0, 1, ft_##TAG##_m0)               \
    FT_K(MI, NJ, WN, NW, 1, 1, ft_##TAG##_m1)               \
    FT_K(MI, NJ, WN, NW, 2, 1, ft_##TAG##_m2)               \
    FT_K(MI, NJ, WN, NW, 3, 3, ft_##TAG##_m3)               \
    FT_K(MI, NJ, WN, NW, 4, 3, ft_##TAG##_m4)

FT_SHAPE(4, 8, 4, 8, 128x256_w8)  /* production routed / shared down (BN 256) */
FT_SHAPE(4, 6, 4, 8, 128x192_w8)  /* production routed up (BN 192) */
FT_SHAPE(4, 4, 4, 8, 128x128_w8)  /* 64 x 32 warp tiles */
FT_SHAPE(4, 8, 2, 4, 128x128_w4)  /* one warp per SMSP, 64 x 64 */
FT_SHAPE(2, 8, 4, 8, 64x256_w8)   /* 32 x 64 warp tiles */
/* tensor-pipe latency / chains: one warp, MI x NJ independent accumulators, registers only */
FT_K(1, 2, 1, 1, 0, 1, ft_chain2)
FT_K(1, 4, 1, 1, 0, 1, ft_chain4)
FT_K(2, 4, 1, 1, 0, 1, ft_chain8)
FT_K(2, 8, 1, 1, 0, 1, ft_chain16)
FT_K(4, 8, 1, 1, 0, 1, ft_chain32)
FT_K(4, 8, 1, 2, 0, 1, ft_chain32_w2)
/* 4-stage rings where the stage fits (128 x 192: 92 KB; 128 x 128: 74 KB) */
FT_K(4, 6, 4, 8, 3, 4, ft_128x192_w8_m3s4)
FT_K(4, 6, 4, 8, 4, 4, ft_128x192_w8_m4s4)
FT_K(4, 4, 4, 8, 3, 4, ft_128x128_w8_m3s4)
FT_K(4, 4, 4, 8, 4, 4, ft_128x128_w8_m4s4)
/* staging alone / without scale copies */
FT_K(4, 8, 4, 8, 5, 3, ft_128x256_w8_m5)
FT_K(4, 8, 4, 8, 6, 3, ft_128x256_w8_m6)
FT_K(4, 8, 4, 8, 7, 3, ft_128x256_w8_m7)
FT_K(4, 6, 4, 8, 5, 3, ft_128x192_w8_m5)
FT_K(4, 6, 4, 8, 6, 3, ft_128x192_w8_m6)
FT_K(4, 6, 4, 8, 7, 3, ft_128x192_w8_m7)
/* bulk-copy (TMA 1-D) feed */
FT_K(4, 8, 4, 8, 8, 3, ft_128x256_w8_m8)
FT_K(4, 6, 4, 8, 8, 3, ft_128x192_w8_m8)
FT_K(4, 6, 4, 8, 8, 4, ft_128x192_w8_m8s4)
FT_K(4, 4, 4, 8, 8, 3, ft_128x128_w8_m8)
FT_K(4, 8, 2, 4, 8, 3, ft_128x128_w4_m8)
/* TMA 2-D feed */
FT_K(4, 8, 4, 8, 11, 3, ft_128x256_w8_m11)
FT_K(4, 6, 4, 8, 11, 3, ft_128x192_w8_m11)
FT_K(4, 6, 4, 8, 11, 4, ft_128x192_w8_m11s4)
FT_K(4, 4, 4, 8, 11, 3, ft_128x128_w8_m11)

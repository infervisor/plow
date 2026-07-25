/* gemm_occ1_poc.hip — STANDALONE occ-1 (4-wave, 512-reg) register-deep-prefetch GEMM PoC.
 *
 *   C[M,N] = A[M,K] . B[N,K]^T   (bf16 operands, f32 accumulate, bf16 store)
 *
 * THE HYPOTHESIS UNDER TEST (see plans/gemm-occ1-poc.md):
 *   plow's production prefill GEMM sits at MfmaUtil=30% at occ-2 (8 waves, 256-reg). The
 *   diagnosed wall is the LDS-read -> MFMA DEPENDENCY CHAIN: each ds_read feeds the very next
 *   MFMA with an s_barrier between and no independent work to hide the ds_read latency, and at
 *   occ-2 there are not enough registers to prefetch deep. hipBLASLt/CK reach 50-90% of peak by
 *   running occ-1 (1 wave/SIMD, 512 reg) with a DEEP SOFTWARE PIPELINE that prefetches K-slices
 *   into REGISTERS 2-3 iterations ahead of the MFMA. This kernel builds exactly that and lets
 *   rocprof measure whether MfmaUtil rises above 30%.
 *
 * THE MECHANISM (differs from op_gemm.h's d_gemm_t):
 *   - 4 waves / WG (2x2 grid), __launch_bounds__(256,1) => 1 wave/SIMD => 512-VGPR budget.
 *   - LDS K-tile (BK) double-buffered across the K loop, register FETCH+COMMIT staging.
 *   - Inside a K-tile the BK/16 MFMA-K slices are contracted through a REGISTER RING of depth
 *     DEPTH: all DEPTH future slices' A/B fragments are read from LDS into distinct registers
 *     BEFORE the MFMA that consumes slice s fires, so each MFMA reads operands already resident
 *     in registers while the next slices' ds_reads are in flight. There is NO s_barrier inside
 *     the slice loop (the whole tile is LDS-resident) — this is what breaks read->MFMA.
 *   - DEPTH is swept (2, 3, 4=full-tile) as separate compiled kernels.
 *
 * Reuses the PROVEN MFMA register layout + helpers from amd_common.h (confirmed on hardware).
 * NOT wired into the interpreter — this is a measurement harness only.
 */
#include "amd_common.h"

#ifndef OCC1_LB
#define OCC1_LB 1 /* min waves/EU: 1 => occ-1, the whole point (512-reg budget) */
#endif

/* One occ-1 register-deep GEMM. BM x BN output tile, 4 waves in a 2x2 grid, BK contracted per
 * LDS tile, DEPTH-deep register prefetch over the BK/16 MFMA-K slices. */
template <int BM, int BN, int BK, int DEPTH, int WM, int WN>
__device__ __forceinline__ void occ1_gemm(bf16* __restrict__ C, const bf16* __restrict__ A,
                                          const bf16* __restrict__ B, unsigned M, unsigned N,
                                          unsigned K, bf16* lds) {
    constexpr int THREADS = WM * WN * PLOW_WAVE; /* 256 */
    constexpr int SM = BM / WM / MFMA_M;         /* A-accumulator tiles per wave */
    constexpr int SN = BN / WN / MFMA_N;         /* B-accumulator tiles per wave */
    constexpr int NSTEP = BK / MFMA_K;           /* MFMA-K slices per LDS tile   */
    constexpr int ST = BK + 8;                   /* padded LDS row stride (halves): breaks the
                                                  * 32-bank conflict, cheap at occ-1's full LDS */
    constexpr int TILE = (BM + BN) * ST;         /* halves per LDS buffer        */
    constexpr int APT = BM * BK / THREADS;        /* A halves staged per thread   */
    constexpr int BPT = BN * BK / THREADS;
    constexpr int APASS = APT / 8;                /* 16-byte staging passes       */
    constexpr int BPASS = BPT / 8;
    /* DEPTH+1 ring slots: consuming slot s while prefetching slot s+DEPTH must not alias, i.e.
     * (s+DEPTH) % RING != s % RING — which requires RING > DEPTH. DEPTH+1 is the minimum. */
    constexpr int RING = DEPTH + 1;

    static_assert(APT % 8 == 0 && BPT % 8 == 0, "tile must stage in 16-byte units");
    static_assert(BM % (WM * MFMA_M) == 0 && BN % (WN * MFMA_N) == 0, "tile must fit wave grid");
    static_assert(DEPTH >= 1 && DEPTH <= NSTEP, "prefetch depth within one LDS tile");
    static_assert(THREADS == 256, "occ-1 PoC is a 4-wave, 256-thread kernel");

    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wm = wave / WN, wn = wave % WN;
    const unsigned frow = mfma_frag_row(lane);

    const unsigned tn = (N + BN - 1) / BN, tm = (M + BM - 1) / BM;
    const unsigned n_tiles = tm * tn;

#define A_SM(b) (lds + (b) * TILE)
#define B_SM(b) (lds + (b) * TILE + BM * ST)

/* global -> registers (ra/rb). Rows past M/N are zero-filled so the MFMA sees clean zeros. */
#define FETCH(ra, rb, m0, n0, k0)                                                             \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                     \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        const unsigned r = (m0) + e / BK, kk = (k0) + e % BK;                                 \
        if (r < M) *(bf16v8*)&(ra)[it * 8] = ld_glob8(as_glob(A) + (size_t)r * K + kk);       \
        else _Pragma("unroll") for (int j = 0; j < 8; j++)(ra)[it * 8 + j] = 0;               \
    }                                                                                         \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                     \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        const unsigned r = (n0) + e / BK, kk = (k0) + e % BK;                                 \
        if (r < N) *(bf16v8*)&(rb)[it * 8] = ld_glob8(as_glob(B) + (size_t)r * K + kk);       \
        else _Pragma("unroll") for (int j = 0; j < 8; j++)(rb)[it * 8 + j] = 0;               \
    }

/* registers -> LDS (padded row stride). */
#define COMMIT(ra, rb, buf)                                                                   \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                     \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        __builtin_memcpy(&A_SM(buf)[(e / BK) * ST + e % BK], &(ra)[it * 8], 16);              \
    }                                                                                         \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                     \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        __builtin_memcpy(&B_SM(buf)[(e / BK) * ST + e % BK], &(rb)[it * 8], 16);              \
    }

/* LDS -> one slice's A/B fragments (ds_read_b128). `af`,`bf_` are the ring slot. */
#define READ_FRAG(af, bf_, buf, step)                                                         \
    _Pragma("unroll") for (int i = 0; i < SM; i++) {                                           \
        const unsigned arow = wm * (BM / WM) + i * MFMA_M + frow;                             \
        __builtin_memcpy(&(af)[i], &A_SM(buf)[arow * ST + mfma_frag_k(lane, (step) * MFMA_K)], 16); \
    }                                                                                         \
    _Pragma("unroll") for (int j = 0; j < SN; j++) {                                           \
        const unsigned brow = wn * (BN / WN) + j * MFMA_N + frow;                             \
        __builtin_memcpy(&(bf_)[j], &B_SM(buf)[brow * ST + mfma_frag_k(lane, (step) * MFMA_K)], 16); \
    }

#define MFMA_SLICE(af, bf_)                                                                   \
    _Pragma("unroll") for (int i = 0; i < SM; i++)                                             \
        _Pragma("unroll") for (int j = 0; j < SN; j++)                                         \
            acc[i][j] = __builtin_amdgcn_mfma_f32_32x32x16_bf16((af)[i], (bf_)[j], acc[i][j], 0, 0, 0);

    for (unsigned tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        const unsigned m0 = (tile / tn) * BM, n0 = (tile % tn) * BN;

        f32x16 acc[SM][SN];
#pragma unroll
        for (int i = 0; i < SM; i++)
#pragma unroll
            for (int j = 0; j < SN; j++) acc[i][j] = (f32x16)(0.0f);

        __align__(16) bf16 ra[APT], rb[BPT];         /* global-stage buffer */
        bf16x8 raf[RING][SM], rbf[RING][SN];          /* the REGISTER RING   */

        const unsigned NT = (K + BK - 1) / BK;

        /* stage tile 0 into LDS buf 0 */
        __syncthreads();
        FETCH(ra, rb, m0, n0, 0);
        COMMIT(ra, rb, 0);
        __syncthreads();

#pragma unroll 1
        for (unsigned kt = 0; kt < NT; kt++) {
            const unsigned buf = kt & 1;
            const unsigned kn = (kt + 1) * BK;

            /* Issue THIS tile's successor global load up front so it lands behind the whole
             * slice-contraction below (register staging, committed after the MFMAs). */
            if (kn < K) { FETCH(ra, rb, m0, n0, kn); }

            /* --- register-ring pipelined contraction over the NSTEP MFMA-K slices ---
             * Prologue: fill the ring with the first DEPTH slices. The compiler is free to keep
             * these ds_reads in flight across the MFMAs below because each ring slot is a
             * DISTINCT register set — this is the decoupling that breaks read->MFMA. */
#pragma unroll
            for (int d = 0; d < DEPTH; d++) { READ_FRAG(raf[d], rbf[d], buf, d); }

#pragma unroll
            for (int s = 0; s < NSTEP; s++) {
                const int nx = s + DEPTH;
                if (nx < NSTEP) { READ_FRAG(raf[nx % RING], rbf[nx % RING], buf, nx); }
                MFMA_SLICE(raf[s % RING], rbf[s % RING]);
            }

            /* commit the successor tile into the other LDS buffer */
            if (kn < K) {
                __syncthreads();
                COMMIT(ra, rb, buf ^ 1);
                __syncthreads();
            }
        }

        /* coalesced store */
        auto* const Cg = as_glob(C);
#pragma unroll
        for (int i = 0; i < SM; i++)
#pragma unroll
            for (int j = 0; j < SN; j++) {
                const unsigned nn = n0 + wn * (BN / WN) + j * MFMA_N + mfma_acc_n(lane);
                if (nn >= N) continue;
#pragma unroll
                for (int e = 0; e < 16; e++) {
                    const unsigned mm = m0 + wm * (BM / WM) + i * MFMA_M + mfma_acc_m(lane, e);
                    if (mm < M) Cg[(size_t)mm * N + nn] = f2bf(acc[i][j][e]);
                }
            }
    }
#undef A_SM
#undef B_SM
#undef FETCH
#undef COMMIT
#undef READ_FRAG
#undef MFMA_SLICE
}

/* One __global__ per (tile, depth). __launch_bounds__(256, OCC1_LB): the 2nd arg is MIN waves/EU
 * on AMD, so 1 lets the allocator take all 512 registers (occ-1). LDS is sized to the config. */
#define OCC1_KERNEL(NAME, BM, BN, BK, DEPTH)                                                   \
    extern "C" __global__ __launch_bounds__(256, OCC1_LB) void NAME(                           \
        bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N, unsigned K) {           \
        __shared__ bf16 lds[2 * (BM + BN) * (BK + 8)];                                         \
        occ1_gemm<BM, BN, BK, DEPTH, 2, 2>(C, A, B, M, N, K, lds);                             \
    }

/* Primary tile: 128x256, BK=64 (NSTEP=4). Sweep prefetch depth 2/3/4. */
OCC1_KERNEL(occ1_128x256_d2, 128, 256, 64, 2)
OCC1_KERNEL(occ1_128x256_d3, 128, 256, 64, 3)
OCC1_KERNEL(occ1_128x256_d4, 128, 256, 64, 4)

/* Alternate tile: 256x128 (same accumulator count, taller M). */
OCC1_KERNEL(occ1_256x128_d2, 256, 128, 64, 2)
OCC1_KERNEL(occ1_256x128_d3, 256, 128, 64, 3)
OCC1_KERNEL(occ1_256x128_d4, 256, 128, 64, 4)

/* WIDER ACCUMULATOR at occ-1: more independent MFMA streams per wave to hide the MFMA
 * accumulator-latency RAW chain (the suspected true cap). 256x256 => SM*SN=16 acc chains;
 * 192x256 => 12; vs 128x256's 8. If MfmaUtil rises with accumulator count, the cap is the
 * accumulator chain, not the operand read. */
OCC1_KERNEL(occ1_192x256_d3, 192, 256, 64, 3)
OCC1_KERNEL(occ1_256x256_d2, 256, 256, 64, 2)
OCC1_KERNEL(occ1_256x256_d3, 256, 256, 64, 3)

/* Deeper pipeline over a longer LDS tile: 128x128, BK=128 (NSTEP=8), depth up to 6. Smaller tile
 * (worse arithmetic intensity) but the ring can go deeper — isolates depth from tile. */
OCC1_KERNEL(occ1_128x128_bk128_d3, 128, 128, 128, 3)
OCC1_KERNEL(occ1_128x128_bk128_d4, 128, 128, 128, 4)
OCC1_KERNEL(occ1_128x128_bk128_d6, 128, 128, 128, 6)

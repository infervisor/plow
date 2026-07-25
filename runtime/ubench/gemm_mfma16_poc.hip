/* gemm_mfma16_poc.hip — STANDALONE bf16 GEMM using the 16x16x32 MFMA (the shape hipBLASLt/CK
 * select for the Qwen prefill shapes), to test the accumulator-chain-count hypothesis.
 *
 *   C[M,N] = A[M,K] . B[N,K]^T   (bf16 operands, f32 accumulate, bf16 store)
 *
 * BACKGROUND (branch gemm-mfma-shape, see verdict): plow's production prefill GEMM uses
 * v_mfma_f32_32x32x16_bf16 (32x32 output, f32x16 = 16-VGPR accumulator) and fits ~12 independent
 * accumulator tiles/wave -> MfmaUtil ~26-40%. Disassembly of libhipblaslt's bf16 TN kernels shows
 * they use v_mfma_f32_16x16x32_bf16 (16x16 output, f32x4 = 4-VGPR accumulator) with MIWT up to
 * 8x5 = 40 independent accumulator tiles/wave issued back-to-back -> MfmaUtil 52-63%. Same ~160-VGPR
 * accumulator budget, 3.3x more independent MFMA->MFMA RAW chains in flight.
 *
 * THIS KERNEL matches the library's MFMA shape and sweeps the independent-chain count (SM*SN) at
 * occ-2 (8-wave, __launch_bounds__(512,2), 256-VGPR budget — plow's PRODUCTION budget) to measure
 * whether MfmaUtil rises from ~30% toward the library's ~60%.
 *
 * 16x16x32 bf16 MFMA register layout (64 lanes, symmetric to amd_common.h's 32x32x16):
 *   A[16][32] : lane l holds  m = l%16,  k = 8*(l/16) + j   (j = 0..7, l/16 = 0..3 -> k = 0..31)
 *   B[32][16] : lane l holds  n = l%16,  k = 8*(l/16) + j
 *   D[16][16] : lane l holds  n = l%16,  m = 4*(l/16) + e   (e = 0..3)
 */
#include "amd_common.h"

#define M16 16
#define N16 16
#define K16 32

__device__ __forceinline__ unsigned f16_frag_k(unsigned lane, unsigned koff) {
    return koff + 8 * (lane / 16);
}
__device__ __forceinline__ unsigned f16_frag_row(unsigned lane) { return lane % 16; }
__device__ __forceinline__ unsigned f16_acc_m(unsigned lane, unsigned e) {
    return 4 * (lane / 16) + e;
}
__device__ __forceinline__ unsigned f16_acc_n(unsigned lane) { return lane % 16; }

template <int BM, int BN, int BK, int WM, int WN>
__device__ __forceinline__ void mfma16_gemm(bf16* __restrict__ C, const bf16* __restrict__ A,
                                            const bf16* __restrict__ B, unsigned M, unsigned N,
                                            unsigned K, bf16* lds) {
    constexpr int THREADS = WM * WN * PLOW_WAVE;
    constexpr int SM = BM / WM / M16; /* independent 16x16 A-accumulator tiles per wave */
    constexpr int SN = BN / WN / N16; /* ... and B; SM*SN = independent MFMA chains/wave */
    constexpr int NSTEP = BK / K16;   /* MFMA-K slices per LDS tile */
    constexpr int ST = BK + 8;        /* padded LDS row stride (halves) */
    constexpr int TILE = (BM + BN) * ST;
    constexpr int APT = BM * BK / THREADS;
    constexpr int BPT = BN * BK / THREADS;
    constexpr int APASS = APT / 8;
    constexpr int BPASS = BPT / 8;

    static_assert(APT % 8 == 0 && BPT % 8 == 0, "tile must stage in 16-byte units");
    static_assert(BM % (WM * M16) == 0 && BN % (WN * N16) == 0, "tile must fit wave grid");

    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wm = wave / WN, wn = wave % WN;
    const unsigned frow = f16_frag_row(lane);

    const unsigned tn = (N + BN - 1) / BN, tm = (M + BM - 1) / BM;
    const unsigned n_tiles = tm * tn;

#define A_SM(b) (lds + (b) * TILE)
#define B_SM(b) (lds + (b) * TILE + BM * ST)

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

#define COMMIT(ra, rb, buf)                                                                   \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                     \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        __builtin_memcpy(&A_SM(buf)[(e / BK) * ST + e % BK], &(ra)[it * 8], 16);              \
    }                                                                                         \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                     \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        __builtin_memcpy(&B_SM(buf)[(e / BK) * ST + e % BK], &(rb)[it * 8], 16);              \
    }

    for (unsigned tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        const unsigned m0 = (tile / tn) * BM, n0 = (tile % tn) * BN;

        f32x4 acc[SM][SN];
#pragma unroll
        for (int i = 0; i < SM; i++)
#pragma unroll
            for (int j = 0; j < SN; j++) acc[i][j] = (f32x4)(0.0f);

        __align__(16) bf16 ra[APT], rb[BPT];

        const unsigned NT = (K + BK - 1) / BK;

        __syncthreads();
        FETCH(ra, rb, m0, n0, 0);
        COMMIT(ra, rb, 0);
        __syncthreads();

#pragma unroll 1
        for (unsigned kt = 0; kt < NT; kt++) {
            const unsigned buf = kt & 1;
            const unsigned kn = (kt + 1) * BK;

            if (kn < K) { FETCH(ra, rb, m0, n0, kn); }

#pragma unroll
            for (int s = 0; s < NSTEP; s++) {
                bf16x8 af[SM], bfr[SN];
#pragma unroll
                for (int i = 0; i < SM; i++) {
                    const unsigned arow = wm * (BM / WM) + i * M16 + frow;
                    __builtin_memcpy(&af[i], &A_SM(buf)[arow * ST + f16_frag_k(lane, s * K16)], 16);
                }
#pragma unroll
                for (int j = 0; j < SN; j++) {
                    const unsigned brow = wn * (BN / WN) + j * N16 + frow;
                    __builtin_memcpy(&bfr[j], &B_SM(buf)[brow * ST + f16_frag_k(lane, s * K16)], 16);
                }
                /* SM*SN independent MFMAs issued back-to-back (distinct acc[i][j]) — the chains. */
#pragma unroll
                for (int i = 0; i < SM; i++)
#pragma unroll
                    for (int j = 0; j < SN; j++)
                        acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(af[i], bfr[j], acc[i][j],
                                                                           0, 0, 0);
            }

            if (kn < K) {
                __syncthreads();
                COMMIT(ra, rb, buf ^ 1);
                __syncthreads();
            }
        }

        auto* const Cg = as_glob(C);
#pragma unroll
        for (int i = 0; i < SM; i++)
#pragma unroll
            for (int j = 0; j < SN; j++) {
                const unsigned nn = n0 + wn * (BN / WN) + j * N16 + f16_acc_n(lane);
                if (nn >= N) continue;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const unsigned mm = m0 + wm * (BM / WM) + i * M16 + f16_acc_m(lane, e);
                    if (mm < M) Cg[(size_t)mm * N + nn] = f2bf(acc[i][j][e]);
                }
            }
    }
#undef A_SM
#undef B_SM
#undef FETCH
#undef COMMIT
}

/* __launch_bounds__(THREADS, WPEU): WPEU=2 => occ-2 (256-reg budget, plow's production budget). */
#define MFMA16_KERNEL(NAME, BM, BN, BK, WM, WN, WPEU)                                          \
    extern "C" __global__ __launch_bounds__(WM* WN* PLOW_WAVE, WPEU) void NAME(                \
        bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N, unsigned K) {           \
        __shared__ bf16 lds[2 * (BM + BN) * (BK + 8)];                                         \
        mfma16_gemm<BM, BN, BK, WM, WN>(C, A, B, M, N, K, lds);                                \
    }

/* ---- occ-2, 8-wave (512-thread) sweep of independent-chain count (SM*SN) ---- */
/* WM=2,WN=4 wave grid. chains = (BM/2/16)*(BN/4/16). */
MFMA16_KERNEL(m16_128x256_o2, 128, 256, 64, 2, 4, 2) /* chains = 4*4  = 16 */
MFMA16_KERNEL(m16_192x256_o2, 192, 256, 64, 2, 4, 2) /* chains = 6*4  = 24 */
MFMA16_KERNEL(m16_256x256_o2, 256, 256, 64, 2, 4, 2) /* chains = 8*4  = 32 */
MFMA16_KERNEL(m16_256x128_o2, 256, 128, 64, 2, 4, 2) /* chains = 8*2  = 16 */
MFMA16_KERNEL(m16_128x128_o2, 128, 128, 64, 2, 4, 2) /* chains = 4*2  =  8 */

/* ---- occ-2, 4-wave (256-thread) workgroups — the library's WG=256 layout ---- */
/* WM=2,WN=2 wave grid. chains = (BM/2/16)*(BN/2/16). 256x160 MIWT8_5 = the down/o_proj winner. */
MFMA16_KERNEL(m16_256x160_w4_o2, 256, 160, 64, 2, 2, 2) /* chains = 8*5 = 40 (library match) */
MFMA16_KERNEL(m16_256x256_w4_o2, 256, 256, 64, 2, 2, 2) /* chains = 8*8 = 64 */
MFMA16_KERNEL(m16_192x256_w4_o2, 192, 256, 64, 2, 2, 2) /* chains = 6*8 = 48 */

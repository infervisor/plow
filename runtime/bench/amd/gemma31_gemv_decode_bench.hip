/* bf16 decode GEMV (`gemv_rows`, DevOp::Gemv) at the REAL Gemma-4 31B BF16 TP1 decode shapes.
 *
 * WHY THIS EXISTS. `Projection + activation` is 22.66 ms of plow's 27.47 ms short decode step and
 * about 28.90 of 39.25 ms at batch 4, against 14.62 / 16.05 ms for vLLM 0.28 on the same box
 * (docs/amd/gemma4-31b-mi300x-trace-20260907.md). The whole projection family is one body,
 * `gemv_rows`, so the question "where does the batched projection lose 2x" is a question about
 * ONE loop and can be asked without a server.
 *
 * The shapes and grids come from `plowrt disasm build-gemma31/assets-glm53/model.pkt --program 4`
 * (and `--program 1`), so `nblk` is the emitter's own dispatch width, not a CU count: gate, up
 * and q run 152 workgroups, k/v run 76, o_proj / down / lm_head run 304.
 *
 * WHAT EVERY ARM HOLDS FIXED. Production body included VERBATIM from op_gemm.h, launch geometry
 * mirroring the persistent interpreter (blockDim = PLOW_THREADS = 512, grid = the packet's `b`),
 * d_gemv_t's own staged/unstaged decision, packet framing (`stage_x_lds` + `__syncthreads()` on
 * both sides of the body), an in-kernel rep loop walking a fresh weight slab out of a 3 GiB arena
 * so the number is a STREAM and not an L2 hit, hipEvent median-of-IT, palindromic interleave, and
 * an A/A control arm that is byte-identical device code under a second name.
 *
 * EVERY ARM HERE IS ARITHMETIC-PRESERVING BY CONSTRUCTION, and the host checks that elementwise
 * on device against the base arm. Each output column keeps its own chunk order (increasing k),
 * its own lane->k map (lane L holds k = chunk*512 + 8L), its own `dot8(w, x, 0.0f)` pair nesting,
 * its own `acc += ` chunk accumulation and its own `wave_sum`. What moves is only:
 *
 *   UN     how many of a column's 16-byte weight loads are issued before any is consumed.
 *   R      how many output COLUMNS a wave carries at once (`gemv_rows_rs`, already in the tree).
 *          The workgroup's column interval [gv_n0, gv_n1) is unchanged, so PLOW_FINE's
 *          gemv->headnorm dependency map is unaffected; only which wave takes which column moves.
 *   KTILE  where `x` is read from when M*K does not fit LDS. Production falls back to reading x
 *          from GLOBAL for every output column; this arm instead walks K in LDS-sized tiles and
 *          keeps one accumulator per (column, row). Same chunks, same order, one barrier per tile.
 *   MPRED  a wave-uniform ladder on the runtime M so an MM=4 object running a batch-1 packet
 *          stops computing three dot products it is about to throw away. The shipping decode
 *          object IS PLOW_GEMV_MM=4 (`plow_gemv_mm_cap_4` in interp_decode_gq.elf) and serves
 *          programs T=1, T=2 and T=4, so that waste is on the batch-1 path today.
 *
 * Build (inside `nix develop /app/plow`):
 *   "$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 --genco \
 *     -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=4 -DPLOW_WG_WAVES=8 \
 *     -DGM_BM=192 -DGM_BN=256 -DGM_BK=64 -DGM_DBUF=1 \
 *     -Iruntime/amd -Iruntime/common runtime/bench/amd/gemma31_gemv_decode_bench.hip -o gv31.co
 */
#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_gemm.h"

/* The interpreter's LDS arena, verbatim: one union of GM_LDS_HALVES halves (64,512 B on the
 * CDNA3 192x256x64 tile), which is what d_gemv_t's `M*K <= GM_LDS_HALVES` staging test measures
 * itself against. Sizing this differently would change which shapes stage and invalidate the
 * whole comparison. */
typedef union __align__(16) {
    bf16 x[GM_LDS_HALVES];
} gv31_smem;

/* ---------------------------------------------------------------------------------------------
 * ARM: K-TILED ACTIVATION STAGING for the shapes where M*K does not fit LDS.
 *
 * THE DEFECT THIS FIXES. `d_gemv_t` stages x into LDS only when `M*K <= GM_LDS_HALVES` (32,256
 * halves). At the Gemma-4 31B decode shapes that test FAILS for exactly the shapes that carry a
 * third of the weight bytes:
 *
 *     down     M=2  K=21504 -> 43,008 halves     M=4 -> 86,016      60 packets/token
 *     o_proj   M=4  K= 8192 -> 32,768 halves     (32,256 is the budget: it misses by 512)  50
 *     o_full   M=2  K=16384 -> 32,768            M=4 -> 65,536                             10
 *
 * The fallback reads x from GLOBAL, once per (output column, k-chunk, row). At M=4 that is FOUR
 * extra 16-byte vector loads for every ONE weight load: the activation is small and L2-resident,
 * but it is spending the same vector-memory issue slots and the same L1 return bandwidth that
 * the weight stream is supposed to own.
 *
 * K-tiling keeps the LDS staging and pays for it with an outer loop: walk K in tiles of KT
 * halves (the largest multiple of one 512-half chunk with MM*KT <= GM_LDS_HALVES), stage
 * min(M,MM) rows of that tile, and accumulate every one of this wave's columns through it before
 * moving on. The accumulators are the price -- RC columns x MM rows of f32 held across the whole
 * K walk -- and RC is bounded because these are exactly the WIDE-K, NARROW-N shapes:
 * ceil(gv_per / PLOW_WAVES) is 3 at N=5376 over 304 workgroups.
 *
 * BIT-EXACT: column n still accumulates chunks 0, 1, 2, ... in increasing k order with the same
 * lane->k map and the same per-chunk `dot8(w, x, 0.0f)`; the tile boundary falls on a chunk
 * boundary by construction (KT is a multiple of 512) and adds no reassociation. `wave_sum` and
 * the store are unchanged. The caller must guarantee ceil(gv_per/PLOW_WAVES) <= RC.
 * ------------------------------------------------------------------------------------------- */
template <int MM, int UN, int RC>
__device__ __forceinline__ void gemv_rows_ktile(bf16* __restrict__ C_, const bf16* __restrict__ x_,
                                                const bf16* __restrict__ W_, unsigned M,
                                                unsigned N, unsigned K, unsigned slice,
                                                unsigned nblk, bf16* __restrict__ lds) {
    constexpr unsigned STEP = PLOW_WAVE * 8;
    /* One tile of x for every one of the MM compiled rows, rounded DOWN to a whole chunk. */
    constexpr unsigned KT = ((unsigned)GM_LDS_HALVES / (unsigned)MM) / STEP * STEP;
    static_assert(KT >= STEP, "K tile must hold at least one 512-half chunk");
    static_assert((unsigned)MM * KT <= (unsigned)GM_LDS_HALVES, "K tile overruns the LDS arena");
    constexpr unsigned CT = KT / STEP; /* chunks per tile */

    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;

    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);

    /* Descriptors and accumulators live across the whole K walk: RC columns x MM rows. */
    __amdgpu_buffer_rsrc_t wr[RC];
    float acc[RC][MM];
#pragma unroll
    for (int r = 0; r < RC; r++) {
        const unsigned n = gv_n0 + wave + (unsigned)r * PLOW_WAVES;
        wr[r] = PLOW_GV_RSRC(W + (size_t)(n < gv_n1 ? n : gv_n0) * K, K);
#pragma unroll
        for (int m = 0; m < MM; m++) acc[r][m] = 0.0f;
    }

    for (unsigned kb = 0; kb < K; kb += KT) {
        const unsigned kt = (K - kb < KT) ? (K - kb) : KT;
        /* Stage rows 0..min(M,MM)-1 of this tile, contiguous per row at stride KT. Rows past M
         * are never read (the `live` predicate below reads row 0 for them, exactly as
         * gemv_rows does), so they are not staged either. */
        __syncthreads();
        for (unsigned m = 0; m < (unsigned)MM && m < M; m++)
            stage_x_lds(lds + m * KT, x_ + (size_t)m * K + kb, kt);
        __syncthreads();

        const unsigned nch = (kt + STEP - 1) / STEP;
        for (unsigned c = 0; c < nch; c += UN) {
            bf16v8 wv[RC][UN];
#pragma unroll
            for (int r = 0; r < RC; r++)
#pragma unroll
                for (int u = 0; u < UN; u++) {
                    /* Past THIS TILE the row still has real weights, so the overshoot cannot be
                     * left to the row-length bounds check the single-tile body relies on: steer
                     * the offset past `num_records` and let the hardware return zero. Without
                     * this the tail of one tile adds chunks belonging to the next. */
                    const unsigned koff = (c + (unsigned)u) * STEP + lane * 8;
                    wv[r][u] = buf_ld8(wr[r], (koff < kt) ? ((kb + koff) * 2u) : (K * 2u));
                }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned koff = (c + (unsigned)u) * STEP + lane * 8;
                const unsigned kx = (koff < kt) ? koff : 0u;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    const bool live = ((unsigned)m < M);
                    const bf16v8 xv = ld_lds8(lds + (live ? (unsigned)m : 0u) * KT + kx);
#pragma unroll
                    for (int r = 0; r < RC; r++)
                        acc[r][m] += live ? dot8(wv[r][u], xv, 0.0f) : 0.0f;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MM; m++)
#pragma unroll
        for (int r = 0; r < RC; r++) {
            const float t = wave_sum(acc[r][m]);
            const unsigned n = gv_n0 + wave + (unsigned)r * PLOW_WAVES;
            if (lane == 0 && (unsigned)m < M && n < gv_n1) st_act1(&C[(size_t)m * N + n], f2bf(t));
        }
    /* The next packet stages over the low end of this arena; leave it quiet. */
    __syncthreads();
}

/* ---------------------------------------------------------------------------------------------
 * ARM: WAVE-UNIFORM M LADDER.
 *
 * `gemv_rows` predicates every activation row with `live = (m < M)` and then computes the dot
 * product ANYWAY, discarding it (`acc[m] += live ? a : 0.0f`). That is free when MM == M. It is
 * not free on the shipping decode object, which is compiled PLOW_GEMV_MM=4 and serves the T=1 and
 * T=2 programs as well: a batch-1 decode step runs four dot8 chains per chunk and throws three
 * away -- 24 VALU operations per 16 bytes of weight on gfx942, which has no v_dot2c_f32_bf16.
 *
 * M is a packet immediate: workgroup-uniform, so the ladder below is a SCALAR branch taken once
 * per k-chunk group, not a per-lane mask. Bit-exact: rows below M take the identical path, rows
 * at or above M were discarded before and are left at zero and never stored.
 * ------------------------------------------------------------------------------------------- */
template <int MM, bool XLDS, int UN, int MA>
__device__ __forceinline__ void gemv_rows_mp_k(float* __restrict__ acc,
                                               __amdgpu_buffer_rsrc_t wr, const bf16* __restrict__ x,
                                               const bf16* lds, unsigned K, unsigned nchunk,
                                               unsigned lane) {
    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };
    for (unsigned c = 0; c < nchunk; c += UN) {
        bf16v8 wv[UN];
#pragma unroll
        for (int u = 0; u < UN; u++)
            wv[u] = buf_ld8(wr, ((c + (unsigned)u) * PLOW_WAVE * 8 + lane * 8) * 2u);
#pragma unroll
        for (int u = 0; u < UN; u++) {
            const unsigned k = (c + (unsigned)u) * PLOW_WAVE * 8 + lane * 8;
            const unsigned kx = (k < K) ? k : 0u;
#pragma unroll
            for (int m = 0; m < MA; m++) acc[m] += dot8(wv[u], xv8(m, kx), 0.0f);
        }
    }
}

template <int MM, bool XLDS, int UN>
__device__ __forceinline__ void gemv_rows_mp(bf16* __restrict__ C_, const bf16* __restrict__ x_,
                                             const bf16* __restrict__ W_, unsigned M, unsigned N,
                                             unsigned K, unsigned slice, unsigned nblk,
                                             const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned nchunk = (K + PLOW_WAVE * 8 - 1) / (PLOW_WAVE * 8);
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);

    for (unsigned n = gv_n0 + wave; n < gv_n1; n += PLOW_WAVES) {
        const __amdgpu_buffer_rsrc_t wr = PLOW_GV_RSRC(W + (size_t)n * K, K);
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;
        if (MM > 1 && M == 1u)
            gemv_rows_mp_k<MM, XLDS, UN, 1>(acc, wr, x, lds, K, nchunk, lane);
        else if (MM > 2 && M == 2u)
            gemv_rows_mp_k<MM, XLDS, UN, (MM > 2 ? 2 : MM)>(acc, wr, x, lds, K, nchunk, lane);
        else
            gemv_rows_mp_k<MM, XLDS, UN, MM>(acc, wr, x, lds, K, nchunk, lane);
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = wave_sum(acc[m]);
            if (lane == 0 && (unsigned)m < M) st_act1(&C[(size_t)m * N + n], f2bf(t));
        }
    }
}

/* ---------------------------------------------------------------------------------------------
 * R-SPLIT WITH A MATCHED TAIL. `gemv_rows_rs` in the tree runs its leftover columns at
 * GV_UNROLL (11), which at MM=4 is a second, much heavier register footprint. Here the tail
 * takes the SAME unroll as the arm under test, so the measurement prices one body shape.
 * ------------------------------------------------------------------------------------------- */
template <int MM, bool XLDS, int UN, int R>
__device__ __forceinline__ void gemv_rows_rsx(bf16* __restrict__ C, const bf16* __restrict__ x,
                                              const bf16* __restrict__ W, unsigned M, unsigned N,
                                              unsigned K, unsigned slice, unsigned nblk,
                                              const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    unsigned n = gv_n0 + wave;
    for (; n + (unsigned)(R - 1) * PLOW_WAVES < gv_n1; n += PLOW_WAVES * (unsigned)R)
        gemv_rows_r<MM, XLDS, UN, R>(C, x, W, M, N, K, n, lane, lds);
    for (; n < gv_n1; n += PLOW_WAVES)
        gemv_rows_r<MM, XLDS, UN * R, 1>(C, x, W, M, N, K, n, lane, lds);
}

/* ---------------------------------------------------------------------------------------------
 * PACKET FRAMING. d_gemv_t's own staged/unstaged decision, its stage_x_lds, its barriers. Both
 * XLDS instantiations are compiled and the runtime `if` picks, exactly as production does, so the
 * register allocation the measurement sees is the one the interpreter would get.
 * ------------------------------------------------------------------------------------------- */
#define GV31_ARM(NAME, MMV, EXPR)                                                                  \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(                               \
        bf16* C, const bf16* x, const bf16* W, unsigned nrep, unsigned N, unsigned K,              \
        unsigned M) {                                                                              \
        constexpr int MM = MMV;                                                                    \
        __shared__ gv31_smem sm;                                                                   \
        for (unsigned e = 0; e < nrep; e++) {                                                      \
            const bf16* const Wp = W + (size_t)e * N * K;                                          \
            if ((size_t)M * K <= (size_t)GM_LDS_HALVES) {                                          \
                stage_x_lds(sm.x, x, (size_t)M * K);                                               \
                __syncthreads();                                                                   \
                { EXPR(true, sm.x); }                                                              \
            } else {                                                                               \
                { EXPR(false, nullptr); }                                                          \
            }                                                                                      \
            __syncthreads();                                                                       \
        }                                                                                          \
    }

/* --- the three expression macros the arms are built from --- */
#define X_ROWS(XL, L, UN)                                                                          \
    gemv_rows<MM, XL, false, UN>(C, x, Wp, nullptr, nullptr, M, N, K, blockIdx.x, gridDim.x, L)
#define X_RS(XL, L, UN, R)                                                                         \
    gemv_rows_rsx<MM, XL, UN, R>(C, x, Wp, M, N, K, blockIdx.x, gridDim.x, L)
#define X_MP(XL, L, UN)                                                                            \
    gemv_rows_mp<MM, XL, UN>(C, x, Wp, M, N, K, blockIdx.x, gridDim.x, L)

#define ARM_ROWS(NAME, MMV, UN) GV31_ARM(NAME, MMV, EROWS_##UN)
#define EROWS_2(XL, L) X_ROWS(XL, L, 2)
#define EROWS_3(XL, L) X_ROWS(XL, L, 3)
#define EROWS_4(XL, L) X_ROWS(XL, L, 4)
#define EROWS_6(XL, L) X_ROWS(XL, L, 6)
#define EROWS_8(XL, L) X_ROWS(XL, L, 8)
#define EROWS_11(XL, L) X_ROWS(XL, L, 11)

#define ARM_RS(NAME, MMV, UN, R) GV31_ARM(NAME, MMV, ERS_##UN##_##R)
#define ERS_2_2(XL, L) X_RS(XL, L, 2, 2)
#define ERS_3_2(XL, L) X_RS(XL, L, 3, 2)
#define ERS_4_2(XL, L) X_RS(XL, L, 4, 2)
#define ERS_6_2(XL, L) X_RS(XL, L, 6, 2)
#define ERS_2_4(XL, L) X_RS(XL, L, 2, 4)
#define ERS_3_4(XL, L) X_RS(XL, L, 3, 4)
#define ERS_4_3(XL, L) X_RS(XL, L, 4, 3)

#define ARM_MP(NAME, MMV, UN) GV31_ARM(NAME, MMV, EMP_##UN)
#define EMP_3(XL, L) X_MP(XL, L, 3)
#define EMP_6(XL, L) X_MP(XL, L, 6)
#define EMP_11(XL, L) X_MP(XL, L, 11)

/* The K-tile arm never stages through the frame: it does its own staging per tile and only ever
 * runs on shapes the frame would have left unstaged. It is written as a whole kernel. */
#define ARM_KT(NAME, MMV, UN, RC)                                                                  \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(                               \
        bf16* C, const bf16* x, const bf16* W, unsigned nrep, unsigned N, unsigned K,              \
        unsigned M) {                                                                              \
        __shared__ gv31_smem sm;                                                                   \
        for (unsigned e = 0; e < nrep; e++)                                                        \
            gemv_rows_ktile<MMV, UN, RC>(C, x, W + (size_t)e * N * K, M, N, K, blockIdx.x,          \
                                         gridDim.x, sm.x);                                         \
    }

/* =========================== THE ARM MATRIX ================================================= */
/* MM=1 */
ARM_ROWS(k_m1_base, 1, 11)
ARM_ROWS(k_m1_aa, 1, 11)
ARM_ROWS(k_m1_un6, 1, 6)
ARM_ROWS(k_m1_un4, 1, 4)
ARM_RS(k_m1_r2u6, 1, 6, 2)
ARM_RS(k_m1_r2u4, 1, 4, 2)
ARM_RS(k_m1_r4u3, 1, 3, 4)
ARM_MP(k_m1_mp, 1, 11)
ARM_KT(k_m1_kt, 1, 6, 4)

/* MM=2 */
ARM_ROWS(k_m2_base, 2, 11)
ARM_ROWS(k_m2_aa, 2, 11)
ARM_ROWS(k_m2_un6, 2, 6)
ARM_ROWS(k_m2_un4, 2, 4)
ARM_RS(k_m2_r2u6, 2, 6, 2)
ARM_RS(k_m2_r2u4, 2, 4, 2)
ARM_RS(k_m2_r4u3, 2, 3, 4)
ARM_MP(k_m2_mp, 2, 11)
ARM_KT(k_m2_kt, 2, 6, 4)

/* MM=4 -- the shipping decode object's bucket. base is GV_UNROLL_M4 = 6. */
ARM_ROWS(k_m4_base, 4, 6)
ARM_ROWS(k_m4_aa, 4, 6)
ARM_ROWS(k_m4_un3, 4, 3)
ARM_ROWS(k_m4_un4, 4, 4)
ARM_ROWS(k_m4_un8, 4, 8)
ARM_ROWS(k_m4_un11, 4, 11)
ARM_RS(k_m4_r2u2, 4, 2, 2)
ARM_RS(k_m4_r2u3, 4, 3, 2)
ARM_RS(k_m4_r2u4, 4, 4, 2)
ARM_RS(k_m4_r2u6, 4, 6, 2)
ARM_RS(k_m4_r4u2, 4, 2, 4)
ARM_MP(k_m4_mp, 4, 6)
ARM_KT(k_m4_kt, 4, 4, 4)

/* MM=8 -- not a shipping bucket for this asset; kept so the M sweep the report asks for exists. */
ARM_ROWS(k_m8_base, 8, 3)
ARM_ROWS(k_m8_aa, 8, 3)
ARM_ROWS(k_m8_un2, 8, 2)
ARM_ROWS(k_m8_un4, 8, 4)
ARM_ROWS(k_m8_un6, 8, 6)
ARM_RS(k_m8_r2u2, 8, 2, 2)
ARM_RS(k_m8_r2u3, 8, 3, 2)
ARM_RS(k_m8_r4u2, 8, 2, 4)
ARM_MP(k_m8_mp, 8, 3)
ARM_KT(k_m8_kt, 8, 3, 4)

/* MFMA batched decode GEMV vs the shipped VALU `gemv_rows`, at Gemma-4 31B BF16 TP1's own
 * decode shapes.
 *
 * WHY THIS EXISTS. At concurrency 4 plow streams decode weights at 839 GB/s where vLLM 0.28
 * streams them at 2111 GB/s on the same card, against a measured 4112 GB/s HBM read ceiling.
 * The diagnosis is arithmetic, not memory: `gemv_rows` spends ~24 VALU ops per 16 B of weight
 * PER ACTIVATION ROW (gfx942 has no `v_dot2c_f32_bf16`), so at batch 4 the VALU work is 4x
 * while the byte stream is 1x. `gemv_rows_mfma4` (op_gemm.h, GV_MFMA4) replaces those 24 ops
 * with two `v_mfma_f32_4x4x4bf16_1k`, keeping the shipped memory pattern byte for byte. This
 * bench asks how much of the 2.5x that buys, and it is the ONLY place the two can be compared
 * without a server in the way.
 *
 * METHODOLOGY, identical to gemma31_gemv_decode_bench.hip. Production bodies included VERBATIM
 * from op_gemm.h; launch geometry mirrors the persistent interpreter (blockDim = PLOW_THREADS,
 * grid = the packet's own `b` from `plowrt disasm .../model.pkt --program 4`); d_gemv_t's own
 * staged/unstaged decision and packet framing (`stage_x_lds` + `__syncthreads()` on both sides);
 * an in-kernel rep loop walking a FRESH weight slab out of a 3 GiB arena so the number is a
 * stream and not an L2 hit; hipEvent median-of-IT; palindromic interleave; and an A/A control
 * arm that is byte-identical device code under a second name.
 *
 * THE MFMA ARMS ARE NOT ARITHMETIC-PRESERVING and the host does not pretend otherwise: it
 * reports relative L2 against the VALU base and against a double-precision CPU reference, and
 * it separately checks that each MFMA arm's row 0 is BIT-IDENTICAL at M=1 and at M=4 (a
 * prompt's output must not depend on how many other sequences share the step).
 *
 * Build (inside `nix develop /app/plow`):
 *   "$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 --genco \
 *     -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=4 -DPLOW_WG_WAVES=8 -DGV_MFMA4=1 \
 *     -DGM_BM=192 -DGM_BN=256 -DGM_BK=64 -DGM_DBUF=1 \
 *     -Iruntime/amd -Iruntime/common runtime/bench/amd/gemma31_mfma_decode_bench.hip -o mf31.co
 */
#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_gemm.h"

typedef union __align__(16) {
    bf16 x[GM_LDS_HALVES];
} mf31_smem;

/* Packet framing, verbatim from gemma31_gemv_decode_bench.hip. */
#define MF31_ARM(NAME, MMV, EXPR)                                                                  \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(                               \
        bf16* C, const bf16* x, const bf16* W, unsigned nrep, unsigned N, unsigned K,              \
        unsigned M) {                                                                              \
        constexpr int MM = MMV;                                                                    \
        __shared__ mf31_smem sm;                                                                   \
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

#define X_ROWS(XL, L, UN)                                                                          \
    gemv_rows<MM, XL, false, UN>(C, x, Wp, nullptr, nullptr, M, N, K, blockIdx.x, gridDim.x, L)
#define X_MF(XL, L, UN, YT)                                                                        \
    gemv_rows_mfma4<MM, XL, UN, YT>(C, x, Wp, M, N, K, blockIdx.x, gridDim.x, L)

#define ARM_ROWS(NAME, MMV, UN) MF31_ARM(NAME, MMV, EROWS_##UN)
#define EROWS_2(XL, L) X_ROWS(XL, L, 2)
#define EROWS_3(XL, L) X_ROWS(XL, L, 3)
#define EROWS_6(XL, L) X_ROWS(XL, L, 6)
#define EROWS_11(XL, L) X_ROWS(XL, L, 11)

#define ARM_MF(NAME, MMV, UN, YT) MF31_ARM(NAME, MMV, EMF_##UN##_##YT)
#define EMF_2_1(XL, L) X_MF(XL, L, 2, 1)
#define EMF_3_1(XL, L) X_MF(XL, L, 3, 1)
#define EMF_4_1(XL, L) X_MF(XL, L, 4, 1)
#define EMF_6_1(XL, L) X_MF(XL, L, 6, 1)
#define EMF_8_1(XL, L) X_MF(XL, L, 8, 1)
#define EMF_11_1(XL, L) X_MF(XL, L, 11, 1)
#define EMF_2_2(XL, L) X_MF(XL, L, 2, 2)
#define EMF_3_2(XL, L) X_MF(XL, L, 3, 2)
#define EMF_4_2(XL, L) X_MF(XL, L, 4, 2)
#define EMF_6_2(XL, L) X_MF(XL, L, 6, 2)
#define EMF_8_2(XL, L) X_MF(XL, L, 8, 2)
#define EMF_2_3(XL, L) X_MF(XL, L, 2, 3)
#define EMF_3_3(XL, L) X_MF(XL, L, 3, 3)
#define EMF_4_3(XL, L) X_MF(XL, L, 4, 3)
#define EMF_6_3(XL, L) X_MF(XL, L, 6, 3)
#define EMF_2_4(XL, L) X_MF(XL, L, 2, 4)
#define EMF_3_4(XL, L) X_MF(XL, L, 3, 4)
#define EMF_4_4(XL, L) X_MF(XL, L, 4, 4)
#define EMF_6_4(XL, L) X_MF(XL, L, 6, 4)

/* =========================== THE ARM MATRIX ================================================= */
/* MM=1: not the target, kept so the sweep has the concurrency-1 rung the tiers already serve. */
ARM_ROWS(k_m1_base, 1, 11)
ARM_ROWS(k_m1_aa, 1, 11)
ARM_MF(k_m1_mf_u6y1, 1, 6, 1)
ARM_MF(k_m1_mf_u11y1, 1, 11, 1)
ARM_MF(k_m1_mf_u4y2, 1, 4, 2)
ARM_MF(k_m1_mf_u6y2, 1, 6, 2)
ARM_MF(k_m1_mf_u3y4, 1, 3, 4)

/* MM=2 */
ARM_ROWS(k_m2_base, 2, 11)
ARM_ROWS(k_m2_aa, 2, 11)
ARM_MF(k_m2_mf_u6y1, 2, 6, 1)
ARM_MF(k_m2_mf_u11y1, 2, 11, 1)
ARM_MF(k_m2_mf_u4y2, 2, 4, 2)
ARM_MF(k_m2_mf_u6y2, 2, 6, 2)
ARM_MF(k_m2_mf_u3y4, 2, 3, 4)
ARM_MF(k_m2_mf_u6y4, 2, 6, 4)
ARM_MF(k_m2_mf_u4y3, 2, 4, 3)

/* MM=4 -- the shipping decode object's bucket. base is GV_UNROLL_M4 = 6. */
ARM_ROWS(k_m4_base, 4, 6)
ARM_ROWS(k_m4_aa, 4, 6)
ARM_MF(k_m4_mf_u3y1, 4, 3, 1)
ARM_MF(k_m4_mf_u6y1, 4, 6, 1)
ARM_MF(k_m4_mf_u8y1, 4, 8, 1)
ARM_MF(k_m4_mf_u11y1, 4, 11, 1)
ARM_MF(k_m4_mf_u2y2, 4, 2, 2)
ARM_MF(k_m4_mf_u4y2, 4, 4, 2)
ARM_MF(k_m4_mf_u6y2, 4, 6, 2)
ARM_MF(k_m4_mf_u8y2, 4, 8, 2)
ARM_MF(k_m4_mf_u3y3, 4, 3, 3)
ARM_MF(k_m4_mf_u4y3, 4, 4, 3)
ARM_MF(k_m4_mf_u2y4, 4, 2, 4)
ARM_MF(k_m4_mf_u3y4, 4, 3, 4)

/* MM=8 */
ARM_ROWS(k_m8_base, 8, 3)
ARM_ROWS(k_m8_aa, 8, 3)
ARM_MF(k_m8_mf_u2y1, 8, 2, 1)
ARM_MF(k_m8_mf_u3y1, 8, 3, 1)
ARM_MF(k_m8_mf_u6y1, 8, 6, 1)
ARM_MF(k_m8_mf_u2y2, 8, 2, 2)
ARM_MF(k_m8_mf_u3y2, 8, 3, 2)
ARM_MF(k_m8_mf_u4y1, 8, 4, 1)

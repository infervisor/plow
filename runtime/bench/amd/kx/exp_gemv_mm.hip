/* kx experiment: gemv_mm — what does a batch-1 packet pay on an MM=4 decode object?
 *
 * THE QUESTION. `PLOW_GEMV_MM` is a COMPILED CEILING, not the batch: one instantiation serves
 * every M <= MM by predicating each activation row with `live = (m < M)` -- and then computing
 * its dot product anyway (op_gemm.h:2251-2270, `a = dot8(wv[u], xv, a)` runs for all MM rows and
 * only the accumulate is predicated). On gfx942, which has no v_dot2c_f32_bf16, each discarded
 * row is 24 VALU ops per 16 bytes of weight -- and the unroll is cut with it, `gv_unroll_for(4)`
 * being GV_UNROLL_M4 = 6 against GV_UNROLL = 11.
 *
 * So: same shape, same M=1, same useful work, two objects. The only thing that moves is a -D.
 *
 * NOTHING IS EDITED. `gemv_rows` is the shipped body, included from op_gemm.h and called exactly
 * as `d_gemv_t` calls it -- packet framing (stage_x_lds + __syncthreads() on both sides), the
 * interpreter's own LDS arena, the emitter's own workgroup count.
 */
#include "kx.h"

#ifndef KX_HOST
#include "op_gemm.h"

/* The interpreter's LDS arena, verbatim: one union of GM_LDS_HALVES halves, which is what
 * d_gemv_t's `M*K <= GM_LDS_HALVES` staging test measures itself against. Sizing it differently
 * would change which shapes stage and invalidate the comparison. */
typedef union __align__(16) { bf16 x[GM_LDS_HALVES]; } kx_gv_smem;

/* One packet of `gemv_rows` at the compiled bucket, exactly as d_gemv_t frames it. M is a
 * RUNTIME value from the shape (kx_d2), which is the whole point: the bucket is compiled, the
 * rows are not. `nrep` walks a fresh [N,K] slab out of the arena so the number is a STREAM and
 * not an L2 hit. */
template <int MM, unsigned MFORCE>
__device__ __forceinline__ void kx_gemv_packet(KX_SIG) {
    __shared__ kx_gv_smem sm;
    const unsigned N = kx_d0, K = kx_d1;
    const unsigned M = MFORCE ? MFORCE : kx_d2;
    auto* const C = (bf16*)kx_out;
    const auto* const x = (const bf16*)kx_x;
    const auto* const W = (const bf16*)kx_w;
    (void)kx_aux;
    for (unsigned e = 0; e < kx_nrep; e++) {
        stage_x_lds(sm.x, x, (size_t)M * K);
        __syncthreads();
        gemv_rows<MM, true, false, gv_unroll_for(MM)>(C, x, W + (size_t)e * N * K, nullptr, nullptr,
                                                      M, N, K, blockIdx.x, kx_nblk, sm.x);
        __syncthreads();
    }
}

KX_BODY(b_gemv) { kx_gemv_packet<PLOW_GEMV_MM, 0>(KX_FWD); }
KX_ARM_OF(k_gemv, b_gemv)
/* A/A: byte-identical device code to k_gemv under a second name. KX_BODY is __forceinline__, so
 * "byte-identical" is a property of the compiler, not of my care in copy-pasting. */
KX_ARM_OF(k_gemv_aa, b_gemv)

/* The wide bucket actually USED wide, for scale: this is the only column where MM=4 is doing the
 * work it was compiled for. Not a candidate — context. */
KX_BODY(b_gemv_m4) { kx_gemv_packet<PLOW_GEMV_MM, 4>(KX_FWD); }
KX_ARM_OF(k_gemv_m4, b_gemv_m4)

#else /* ------------------------------------------------------- host descriptor */

static const kx_variant VAR[] = {
    {"mm1", "-DPLOW_GEMV_MM=1"},
    {"mm4", "-DPLOW_GEMV_MM=4"},
};

static const kx_arm ARM[] = {
    {"k_gemv", 0, KX_BASE | KX_REF, "MM=1 object, M=1 packet — the tier-matched object"},
    {"k_gemv_aa", 0, KX_AA, "byte-identical device code to k_gemv"},
    {"k_gemv", 1, KX_EXACT, "MM=4 object, M=1 packet — SAME useful work, three dot chains discarded"},
    {"k_gemv_m4", 1, KX_EXACT | KX_SLOWOK, "MM=4 object, M=4 packet — 4x the useful work, for scale"},
};

/* Gemma-4 31B BF16 TP1, the T=4 decode program: 60 layers, 50 sliding (hd 256, o K=8192) + 10
 * full (hd 512, o K=16384). `grid` is the packet's own `b` from
 * `plowrt disasm build-gemma31/assets-glm53/model.pkt --program 4`, not a CU count. */
#define SLAB(N, K) ((double)(N) * (double)(K) * 2.0)
static const kx_shape SHP[] = {
    {"q_proj", 8192, 5376, 1, 152, 60, SLAB(8192, 5376), 0, 8192},
    {"kv_proj", 4096, 5376, 1, 76, 120, SLAB(4096, 5376), 0, 4096},
    {"o_slide", 5376, 8192, 1, 304, 50, SLAB(5376, 8192), 0, 5376},
    {"o_full", 5376, 16384, 1, 304, 10, SLAB(5376, 16384), 0, 5376},
    {"gate_up", 21504, 5376, 1, 152, 120, SLAB(21504, 5376), 0, 21504},
    {"down", 5376, 21504, 1, 304, 60, SLAB(5376, 21504), 0, 5376},
    {"lm_head", 262144, 5376, 1, 304, 1, SLAB(262144, 5376), 0, 262144},
};

static const kx_exp EXP = {
    "gemv_mm",
    "what does a batch-1 packet pay for running on an MM=4 decode object?",
    "runtime/bench/amd/kx/exp_gemv_mm.hip",
    /* scripts/build_gfx942.sh's AX_DECODE, minus PLOW_GEMV_MM which is the variant axis. */
    "-DPLOW_BUCKET_DECODE=1 -DPLOW_WG_WAVES=8 -DPLOW_MOE_GEMMA=1 -DPLOW_MOE_GEMMA_PF=1",
    "gfx942",
    512,
    KX_DT_BF16,
    VAR,
    2,
    ARM,
    4,
    SHP,
    7,
    0.98,
    1.02,
    0.0,
    "MM=4 at M=1 costs ~1.67x of MM=1 at M=1 for identical output "
    "(docs/amd/gemma4-31b-mi300x.md: 25.86 vs 15.52 ms/token)",
};

extern "C" const kx_exp* kx_experiment(void) { return &EXP; }

#endif

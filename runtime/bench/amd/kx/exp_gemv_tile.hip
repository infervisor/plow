/* kx experiment: gemv_tile — at the short-K decode shapes, does R=2 beat R=4?
 *
 * THE QUESTION. A wave keeps min(UN, nchunk) 16-byte loads outstanding per output column, and
 * nchunk = ceil(K/512) is 2 at K=768 and 3 at K=1536 -- so at these shapes the unroll is capped by
 * the ROW, not by UN, and the wave runs `nchunk` SERIAL latency rounds. R output columns per
 * wave-step multiply the outstanding loads by R and divide the round count by R. The obvious move
 * is to make R as large as the register budget allows. It is the wrong move.
 *
 * THE RAGGED TAIL is why. `gemv_rows_rs` runs full R-groups and then drops the leftover columns
 * into an R=1 tail. At o_proj, gv_per = 28 deals waves 0-3 four columns and waves 4-7 three, so
 * with R=4 HALF THE WAVES complete zero full groups and run their entire column count through the
 * tail. R=2 divides every K3 gv_per far more often.
 *
 * NOTHING IS EDITED. Both arms are the shipped `gemv_rows_rs` (op_gemm.h:2445) at different
 * template arguments, and the baseline is the shipped single-column `gemv_rows`. R*UN is held at
 * 14 vectors = 56 VGPRs in the candidate so the comparison is register-neutral, which is the same
 * constraint the shipped GV_RS_UN=7 is chosen under.
 *
 * BIT-EXACT BY CONSTRUCTION, which is why both candidates are declared KX_EXACT: each column
 * keeps its own chunk order, lane->k map, dot8 nesting, accumulation order and wave_sum. Only
 * WHICH WAVE takes which column moves, and the workgroup's column ownership [gv_n0, gv_n1) is
 * unchanged. Anything but bit-identical output is a bug, not a rounding difference.
 */
#include "kx.h"

#ifndef KX_HOST
#include "op_gemm.h"

typedef union __align__(16) { bf16 x[GM_LDS_HALVES]; } kx_gv_smem;

/* d_gemv_t's packet framing, with the column body left as a template parameter. */
template <int R, int UN>
__device__ __forceinline__ void kx_split_packet(KX_SIG) {
    __shared__ kx_gv_smem sm;
    const unsigned N = kx_d0, K = kx_d1;
    auto* const C = (bf16*)kx_out;
    const auto* const x = (const bf16*)kx_x;
    const auto* const W = (const bf16*)kx_w;
    (void)kx_aux;
    (void)kx_d2;
    for (unsigned e = 0; e < kx_nrep; e++) {
        stage_x_lds(sm.x, x, K);
        __syncthreads();
        const bf16* Wp = W + (size_t)e * N * K;
        if constexpr (R == 1)
            gemv_rows<1, true, false, GV_UNROLL>(C, x, Wp, nullptr, nullptr, 1u, N, K, blockIdx.x, kx_nblk,
                                                 sm.x);
        else
            gemv_rows_rs<1, true, UN, R, GV_UNROLL>(C, x, Wp, 1u, N, K, blockIdx.x, kx_nblk,
                                                    sm.x);
        __syncthreads();
    }
}

KX_BODY(b_r1) { kx_split_packet<1, GV_UNROLL>(KX_FWD); }
KX_ARM_OF(k_r1, b_r1)
KX_ARM_OF(k_r1_aa, b_r1)

/* THE SHIPPED CHOICE: GV_RS_R=2, GV_RS_UN=7. R*UN = 14 vectors. */
KX_BODY(b_r2un7) { kx_split_packet<2, 7>(KX_FWD); }
KX_ARM_OF(k_r2un7, b_r2un7)

/* THE REJECTED ALTERNATIVE, kept so the reason stays measurable: same 12-14 vectors of register
 * pressure, twice the columns per group, and the tail eats it. */
KX_BODY(b_r4un3) { kx_split_packet<4, 3>(KX_FWD); }
KX_ARM_OF(k_r4un3, b_r4un3)

#else /* ------------------------------------------------------- host descriptor */

static const kx_variant VAR[] = {{"dec", ""}};

static const kx_arm ARM[] = {
    {"k_r1", 0, KX_BASE | KX_REF, "shipped single-column gemv_rows, UN=GV_UNROLL"},
    {"k_r1_aa", 0, KX_AA, "byte-identical device code to k_r1"},
    {"k_r2un7", 0, KX_EXACT, "gemv_rows_rs R=2 UN=7 — the shipped R-split (GV_RS_R/GV_RS_UN)"},
    {"k_r4un3", 0, KX_EXACT, "gemv_rows_rs R=4 UN=3 — the rejected alternative"},
};

/* Kimi-K3 TP8 fp8-KV decode, from `plowrt disasm <blob>/model.pkt --program 1`: (N, K, blk,
 * instances-per-token). nchunk = ceil(K/512) is in the shape name's neighbourhood: 1536 -> 3,
 * 768 -> 2, 3584 -> 7, 7168 -> 14. The last two rows are the CONTROL: at nchunk >= 8 the
 * single-column body already holds 8-14 loads per wave and the split should read ~1.00x. */
#define SLAB(N, K) ((double)(N) * (double)(K) * 2.0)
static const kx_shape SHP[] = {
    {"o_proj", 7168, 1536, 0, 256, 93, SLAB(7168, 1536), 0, 7168},
    {"shared_down", 7168, 768, 0, 256, 92, SLAB(7168, 768), 0, 7168},
    {"q_absorb", 6144, 1536, 0, 256, 24, SLAB(6144, 1536), 0, 6144},
    {"q_rope", 768, 1536, 0, 256, 24, SLAB(768, 1536), 0, 768},
    {"moe_up_latent", 7168, 3584, 0, 256, 92, SLAB(7168, 3584), 0, 7168},
    {"moe_down_lat", 3584, 7168, 0, 256, 92, SLAB(3584, 7168), 0, 3584},
    {"q_a", 1536, 7168, 0, 256, 48, SLAB(1536, 7168), 0, 1536},
};

static const kx_exp EXP = {
    "gemv_tile",
    "at the short-K decode shapes, does R=2 beat R=4 — and is either bit-exact?",
    "runtime/bench/amd/kx/exp_gemv_tile.hip",
    "-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_WG_WAVES=8 -DPLOW_MOE_GEMMA=1 "
    "-DPLOW_MOE_GEMMA_PF=1",
    "gfx942",
    512,
    KX_DT_BF16,
    VAR,
    1,
    ARM,
    4,
    SHP,
    7,
    0.98,
    1.02,
    0.0,
    "R=2/UN=7 wins the short-K shapes, R=4/UN=3 wins less (gfx950 record in op_gemm.h:2305: "
    "o_proj 1.42x vs 1.21x); nchunk>=8 shapes read 0.99-1.00x for both",
};

extern "C" const kx_exp* kx_experiment(void) { return &EXP; }

#endif

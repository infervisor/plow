/* test_kernels.hip — GOLDEN-DATA TEST WRAPPERS ONLY. Not on the serving path.
 *
 * The serving path is the persistent interpreter (interp.hip): one launch, no
 * per-op dispatch. These __global__ wrappers exist so the CPU-reference tests
 * can drive each op in isolation — and they call the SAME __device__ functions
 * the interpreter calls, so a passing test validates the megakernel's actual
 * code rather than a parallel implementation of it.
 *
 * They map (blockIdx, gridDim) onto the interpreter's (slice, nblk), which is
 * the only difference between a test launch and a packet.
 */
#include "amd_common.h"
#include "op_attention.h"
#include "op_elementwise.h"
#include "op_gemm.h"
#include "op_moe.h"
#include "op_norm.h"

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_rmsnorm_bf16(
    bf16* out, const bf16* x, const bf16* gamma, unsigned rows, unsigned feat, float eps) {
    __shared__ float part[PLOW_WAVES];
    d_rmsnorm(out, x, gamma, rows, feat, eps, 0, blockIdx.x, gridDim.x, part);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_rowrms_bf16(
    float* rms, const bf16* x, unsigned rows, unsigned feat, float eps) {
    __shared__ float part[PLOW_WAVES];
    d_rowrms(rms, x, rows, feat, eps, blockIdx.x, gridDim.x, part);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_headnorm_rope_bf16(
    bf16* out, const bf16* x, const bf16* gamma, const float* cosb, const float* sinb,
    const int* pos, unsigned ntok, unsigned nhead, unsigned hd, float eps) {
    if (hd == 256)
        d_headnorm_rope<256>(out, x, gamma, cosb, sinb, pos, ntok, nhead, eps, 0, 0, PLOW_KV_MASK_NONE,
                             /*skip_norm=*/0, blockIdx.x, gridDim.x);
    else
        d_headnorm_rope<512>(out, x, gamma, cosb, sinb, pos, ntok, nhead, eps, 0, 0, PLOW_KV_MASK_NONE,
                    /*skip_norm=*/0, blockIdx.x, gridDim.x);
}

/* GLM-5.2 INTERLEAVED partial RoPE on the 64-dim rope slice (HD=64, no norm). Even/odd pairing. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void glm_rope_interleave_bf16(
    bf16* out, const bf16* x, const float* cosb, const float* sinb, const int* pos, unsigned ntok,
    unsigned nhead) {
    d_headnorm_rope<64, /*INTERLEAVE=*/true>(out, x, /*gamma=*/nullptr, cosb, sinb, pos, ntok, nhead,
                                             /*eps=*/0.0f, 0, 0, PLOW_KV_MASK_NONE,
                                             /*skip_norm=*/1, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_residual_add_bf16(
    bf16* out, const bf16* a, const bf16* b, unsigned n, float scale) {
    d_residual(out, a, b, n, scale, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_geglu_bf16(
    bf16* out, const bf16* gate, const bf16* up, unsigned n) {
    d_glu(out, gate, up, n, PLOW_ACT_GELU_TANH_, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_embed_scale_bf16(
    bf16* out, const bf16* table, const int* ids, unsigned ntok, unsigned hidden, float scale) {
    d_embed(out, table, ids, ntok, hidden, scale, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_logit_softcap_bf16(
    bf16* out, const bf16* x, unsigned n, float cap) {
    d_softcap(out, x, n, cap, blockIdx.x, gridDim.x);
}

/* One SEPARATE kernel per tile config.
 *
 * This is not cosmetic. A runtime switch over the template instantiations makes
 * the compiler allocate registers for the WORST config across all arms, so every
 * config spills (measured: 3588 B/thread). Register allocation is per-kernel, so
 * bucket specialisation has to be a compile-time split — which is exactly what a
 * compiler-driven bucket system is for. */
#define GEMM_VARIANT(NAME, BM, BN, BK)                                             \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(                     \
        bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N, unsigned K) { \
        __shared__ bf16 lds[GM_LDS_HALVES_T(BM, BN, BK)];                          \
        d_gemm_t<BM, BN, BK, GM_WM, GM_WN, false>(C, A, B, nullptr, nullptr, M, N, K, blockIdx.x, \
                                    gridDim.x, lds);                               \
    }


GEMM_VARIANT(gemm_c0, 256, 256, 64)
GEMM_VARIANT(gemm_c1, 256, 128, 64)
GEMM_VARIANT(gemm_c2, 128, 256, 64)
GEMM_VARIANT(gemm_c3, 128, 128, 64)
GEMM_VARIANT(gemm_c4, 64, 128, 64)  /* the small tile: for shapes that cannot fill 256 CUs */

/* FP8 (w8a8) GEMM variants — same tile family, fp8 operands, dequant epilogue. LDS is bytes, so
 * the tile fits in HALF the halves the bf16 arena reserves (rounded up to a whole bf16 count). */
#define GEMM_FP8_VARIANT(NAME, BM, BN, BK)                                          \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(  \
        bf16* C, const unsigned char* A, const unsigned char* B, const float* asc,   \
        const float* wsc, unsigned M, unsigned N, unsigned K) {                      \
        __shared__ bf16 lds[GM_LDS_HALVES_T(BM, BN, BK)];                            \
        d_gemm_fp8_t<BM, BN, BK, GM_WM, GM_WN>(C, A, B, asc, wsc, M, N, K, blockIdx.x, \
                                    gridDim.x, lds);                                 \
    }
GEMM_FP8_VARIANT(gemm_fp8_c0, 256, 256, 64)
GEMM_FP8_VARIANT(gemm_fp8_c3, 128, 128, 64)
GEMM_FP8_VARIANT(gemm_fp8_c4, 64, 128, 64)
GEMM_FP8_VARIANT(gemm_fp8_c5, 192, 256, 64) /* the Qwen prefill tile, vs bf16 c5 */

/* MXFP4 (w4a16) GEMM variants — the five SELECTABLE rungs, one kernel each.
 *
 * These exist so `runtime/ubench/gemm_tile_sweep.c` can RANK the mxfp4 ladder. Until they did,
 * `tunedb`'s `RUNGS` table mapped every rung to a `Gemm*Mxfp4` opcode and `plowc tune best
 * --quant Mxfp4` had a column with nothing in it: the store held 3410 records and every single
 * one was `quant = None`, because the only mxfp4 GEMM this file exported was `d_gemm_mxfp4_k`,
 * the 256x256 GOLDEN, which measures one tile and cannot compare five.
 *
 * Textually the bf16 `GEMM_VARIANT` with `WFP4=true` and the fp4 operand pair — exactly what
 * `op_gemm.h`'s `PLOW_GM_MXFP4_TILE` instantiates for the interpreter, so what is measured here
 * is the body the interpreter runs, at the tile the sweep names. The template arguments are
 * spelled out to their defaults up to `WFP4` because `WFP4` is the 11th and C++ has no
 * designated template arguments.
 *
 * LDS is the bf16 arena, NOT the fp8 one: w4a16 dequants fp4->bf16 in the B-FETCH and stages
 * bf16, so a tile occupies the same halves the bf16 rung does. */
#define GEMM_MXFP4_VARIANT(NAME, BM, BN, BK)                                                 \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(         \
        bf16* C, const bf16* A, const unsigned char* W, const unsigned char* wscale,         \
        unsigned M, unsigned N, unsigned K) {                                                \
        __shared__ bf16 lds[GM_LDS_HALVES_T(BM, BN, BK)];                                    \
        d_gemm_t<BM, BN, BK, GM_WM, GM_WN, false, GM_SWZ, GM_WGM, (GM_PP != 0), true, false, \
                 true>(C, A, (const bf16*)W, nullptr, nullptr, M, N, K, blockIdx.x,          \
                       gridDim.x, lds, nullptr, 0, wscale);                                  \
    }
GEMM_MXFP4_VARIANT(gemm_mxfp4_c0, 256, 256, 64)
GEMM_MXFP4_VARIANT(gemm_mxfp4_c2, 128, 256, 64)
GEMM_MXFP4_VARIANT(gemm_mxfp4_c3, 128, 128, 64)
GEMM_MXFP4_VARIANT(gemm_mxfp4_c4, 64, 128, 64)
GEMM_MXFP4_VARIANT(gemm_mxfp4_c5, 192, 256, 64)

/* NON-POWER-OF-2 macro tiles — the qwen-prefill-gemm sweep. The hypothesis was that a
 * smaller, ARCH-only accumulator (avoiding 256x256's 128 AGPRs and their v_accvgpr
 * bracketing) would raise efficiency. acc regs = (BM/64)*(BN/128)*16 : 192x256->96
 * 320x128->80 384x128->96 128x384->96 192x128->48; all measured 0 AGPR / occ 2 standalone. */
GEMM_VARIANT(gemm_c5, 192, 256, 64)
GEMM_VARIANT(gemm_c6, 320, 128, 64)
GEMM_VARIANT(gemm_c7, 384, 128, 64)
GEMM_VARIANT(gemm_c8, 128, 384, 64)
GEMM_VARIANT(gemm_c9, 192, 128, 64)
/* 192x384 / 256x384 were tried: both exceed the 160 KiB LDS limit AND fall to AGPR
 * (192x384: 220V+144A) — the double register+LDS cliff. Not viable.
 *
 * SWEEP RESULT — the ranking is POWER-REGIME dependent, so it splits two ways (all numbers
 * % of 1660 TF/s, M=8192, INTERLEAVED to defeat the clock-ramp artifact in gemm_bench_8k.c):
 *
 *  STANDALONE, idle GPU (abundant power):        THROUGH THE INTERPRETER, Qwen (power-limited,
 *    tile      q    kv   o    gate down            the sustained serving regime):
 *    256x256   50   44   40   56   46                tile      q    o    gate down   kv
 *    128x256   41   37   40   45   44                256x256   32   27   33   29     28
 *    192x256   36   26   37   40   40                192x256   34   36   36   35     26
 *    128x384   42   32   43   48   51              => 192x256 WINS the serving regime for Qwen
 *  => 256x256 wins when power is abundant.            (o +32%, down +21%, gate +11%, q +6%).
 *
 * And it is K-dependent: through the interpreter, GEMMA (K=5376) prefers 256x256 (192 loses
 * 4-8%), the opposite of Qwen (K=2560). No single global winner. 256x256 stays the DEFAULT;
 * a Qwen prefill object should build with -DGM_BM=192. o_proj/down (N=2560) are additionally
 * FILL-limited (62.5% at BN=256) where 128x384 helps — a per-shape tile-slot job
 * (plans/tile-specific-gemm.md). Compile-time-K specialisation was a WASH (+0.4%). These
 * variants are kept as the calibration set the picker's per-shape sweep draws from. */

/* The default the interpreter carries. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_gemm_bf16(
    bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N, unsigned K, unsigned cfg) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    (void)cfg; /* the tile is a compile-time bucket; see the GEMM_VARIANT kernels */
    d_gemm(C, A, B, M, N, K, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_gemm_norm_bf16(
    bf16* C, const bf16* A, const bf16* B, const float* rms, const bf16* gamma, unsigned M,
    unsigned N, unsigned K) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemm_norm(C, A, B, rms, gamma, M, N, K, blockIdx.x, gridDim.x, lds);
}

/* One GEMV kernel per decode batch bucket. The bucket is a COMPILE-TIME constant
 * (see PLOW_GEMV_MM in op_gemm.h): a runtime chain over the buckets inlines all five
 * instantiations and the register allocator then budgets for the worst arm, which is
 * what used to cost the whole interpreter its 8-wave dispatch. d_gemv_t<MM> takes a
 * runtime M and predicates on `m < M`, so one instantiation serves every M <= MM. */
#define GEMV_VARIANT(NAME, MM)                                                              \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(        \
        bf16* C, const bf16* x, const bf16* W, const float* rms, const bf16* gamma,         \
        unsigned M, unsigned N, unsigned K, unsigned norm) {                                \
        __shared__ bf16 lds[GM_LDS_HALVES];                                                 \
        d_gemv_t<MM>(C, x, W, rms, gamma, M, N, K, (int)norm, 1e-6f, blockIdx.x, gridDim.x,  \
                     lds);                                                                  \
    }

/* ACTIVATION QUANT — the w8a8 prefill's activation half, and the piece the GEMM_FP8_VARIANT
 * kernels above cannot test: they take a ready-made ascale, so nothing exercised the per-row
 * absmax/round path that produces it. (The GEMM itself is already covered by gemm_fp8_c0, whose
 * 256/256/64 tile is the one the interpreter carries.) */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void d_quant_fp8_k(
    unsigned char* xq, const bf16* x, float* ascale, unsigned M, unsigned K) {
    d_quant_fp8(xq, x, ascale, M, K, blockIdx.x, gridDim.x);
}

/* MXFP4 (OCP microscaling e2m1 + one E8M0 scale per 32 K) decode GEMV, w4a16.
 * W is packed 2 fp4/byte (row stride K/2 bytes, low nibble = even k); S is one E8M0 byte per
 * 32-K block (row stride K/32). The scale is folded into the fp4->bf16 convert and that fold is
 * EXACT because an MX scale is a power of two — so unlike gemv_fp8_blk there is no dequant
 * argument and no epilogue multiply. The golden test's job is to confirm that on real silicon. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void d_gemv_mxfp4_k(
    bf16* C, const bf16* x, const unsigned char* W, const unsigned char* S, unsigned M,
    unsigned N, unsigned K) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_mxfp4(C, x, W, S, M, N, K, blockIdx.x, gridDim.x, lds);
}

/* MXFP4 decode fused gate|up GEMV+GLU (w4a16). Gate/up are two fp4 matrices Wg/Wu each with its own
 * E8M0 scale row Sg/Su; the SwiGLU epilogue fuses act(g)*u (act: 0=silu). Same packed layout and
 * scale semantics as d_gemv_mxfp4_k (no epilogue dequant — the scale folds into the convert). The
 * golden test confirms the gate/up fusion + SwiGLU epilogue on real silicon. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void d_gemv_glu_mxfp4_k(
    bf16* C, const bf16* x, const unsigned char* Wg, const unsigned char* Wu,
    const unsigned char* Sg, const unsigned char* Su, unsigned M, unsigned N, unsigned K,
    unsigned act) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_glu_mxfp4(C, x, Wg, Wu, Sg, Su, M, N, K, act, blockIdx.x, gridDim.x, lds);
}

/* MXFP4 (w4a16) prefill GEMM: bf16 activations A[M,K] x packed fp4 weights W[N,K/2] with E8M0 scale
 * rows wscale[N,K/32]. Reuses the bf16 wide-K MFMA (only the weight fetch dequants). The golden
 * confirms the dequant-on-load path and the folded-scale numerics on real silicon. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void d_gemm_mxfp4_k(
    bf16* C, const bf16* A, const unsigned char* W, const unsigned char* wscale, unsigned M,
    unsigned N, unsigned K) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemm_mxfp4(C, A, W, wscale, M, N, K, blockIdx.x, gridDim.x, lds);
}

/* Block-fp8 (DeepSeek/GLM weight_block_size [128,128]) decode GEMV. wscale is the
 * ceil(N/128) x ceil(K/128) f32 dequant grid. One workgroup (nblk=1, slice=0). */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemv_fp8_blk(
    bf16* C, const bf16* x, const unsigned char* W, const float* wscale, unsigned M, unsigned N,
    unsigned K) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_fp8_blk(C, x, W, wscale, M, N, K, blockIdx.x, gridDim.x, lds);
}

/* Block-fp8 DENSE PREFILL GEMM (w8a16) — op 107, the T-row twin of `gemv_fp8_blk` above.
 * A[M,K] bf16 x W[N,K] e4m3 with the SAME ceil(N/128) x ceil(K/128) f32 `weight_scale_inv` grid,
 * so a divergence between this golden and `gemv_fp8_blk` on the same weights IS a scale-convention
 * bug — which is the one failure mode this kernel has that does not crash. LDS is the 128x128x64
 * tile's own footprint, not the object-wide arena: the interpreter passes `sm->gm`, which is sized
 * by the largest tile any packet may select, and this wrapper is the only place the real number is
 * visible. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void d_gemm_fp8_blk_k(
    bf16* C, const bf16* A, const unsigned char* W, const float* wscale, unsigned M, unsigned N,
    unsigned K) {
    __shared__ bf16 lds[GM_LDS_HALVES_T(GM_BLK_BM, GM_BLK_BN, GM_BLK_BK)];
    d_gemm_fp8_blk(C, A, W, wscale, M, N, K, blockIdx.x, gridDim.x, lds);
}

/* Block-fp8 MoE expert gate/up + down, one slot, driven through the real table indirection.
 *
 * The trailing `enc` is the RUNTIME weight-encoding field the interpreter reads from the
 * instruction (i[6] on the decode ops 45/46, i[3] on the prefill ops 85/86 — the slots differ
 * because i[3] is already n_exp on decode). These wrappers did not pass it, so this whole file
 * stopped compiling the moment the field landed and `test_kernels.elf` went STALE on disk —
 * exactly the failure its own build rule warns about ("building the interpreter and not these
 * is how t_attn once passed three times against a stale kernel"). Pinned to FP8BLK rather than
 * defaulted in op_moe.h: a default would let the next caller silently omit it again, and these
 * two entry points are the block-fp8 goldens by name. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void moe_expert_glu_fp8_blk_k(
    bf16* fu, const bf16* x, const unsigned char* table, const unsigned long long* wtab,
    const unsigned long long* stab, unsigned I_moe, unsigned H, unsigned n_exp, unsigned act) {
    // K3's `situ` added `beta`/`lbeta` to this signature (the activation transforms the UP branch
    // too, so it needs both). They are read ONLY when `act == PLOW_MOE_ACT_SITU`; this golden
    // exercises the block-fp8 arm at the classic activations, so 0/0 is the correct pair and keeps
    // the kernel bit-identical to what it tested before that parameter existed.
    d_moe_expert_glu_fp8_blk(fu, x, table, wtab, stab, 0, I_moe, H, n_exp, act, blockIdx.x,
                             gridDim.x, PLOW_MOE_ENC_FP8BLK, 0.0f, 0.0f);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void moe_expert_down_fp8_blk_k(
    float* part, const bf16* fu, const unsigned char* table, const unsigned long long* wtab,
    const unsigned long long* stab, unsigned H, unsigned I_moe, unsigned n_exp) {
    d_moe_expert_down_fp8_blk(part, fu, table, wtab, stab, 0, H, I_moe, n_exp, blockIdx.x,
                              gridDim.x, PLOW_MOE_ENC_FP8BLK);
}

/* Block-fp8 DENSE MLP gate/up (fused SwiGLU) on named weights — GLM dense layers 0-2. Op 47.
 *
 * The two trailing zeros are the K3 situ betas, which `d_dense_glu_fp8_blk` grew when the
 * epilogue became `moe_glu(g, u, act, beta, lbeta)`. This golden did not grow with it and the
 * file stopped compiling; the values are inert for every act but `PLOW_MOE_ACT_SITU`, where
 * `moe_glu` returns `k3_situ_gate(g, beta) * k3_situ_up(u, lbeta)`. So this wrapper is a golden
 * for the NON-situ epilogues only, and a situ arm here would need real betas threaded in as
 * kernel arguments rather than these placeholders. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void dense_glu_fp8_blk_k(
    bf16* fu, const bf16* x, const unsigned char* Wg, const unsigned char* Wu, const float* Sg,
    const float* Su, unsigned N, unsigned K, unsigned act) {
    /* beta/lbeta are the GLU's clamp knobs; `0,0` is "unclamped", matching the golden
     * `moe_expert_glu_fp8_blk_k` wrapper above. They arrived on `d_dense_glu_fp8_blk` with the
     * MXFP4 MoE work and this wrapper was not moved with them, which left test_kernels.hip
     * uncompilable — and with it the whole tile-sweep harness, which loads test_kernels.elf. */
    d_dense_glu_fp8_blk(fu, x, Wg, Wu, Sg, Su, N, K, act, blockIdx.x, gridDim.x, 0.0f, 0.0f);
}

GEMV_VARIANT(gemma_gemv_bf16, 1) /* the default decode bucket */
GEMV_VARIANT(gemma_gemv_m4, 4)
GEMV_VARIANT(gemma_gemv_m8, 8)
GEMV_VARIANT(gemma_gemv_m16, 16)

/* ---------------------------------------------------------------------------------------
 * THE DECODE-GEMV ROW CAMPAIGN (runtime/ubench/gemv_row_sweep.c, scripts/rebench_tune_gemv.sh).
 *
 * The GEMM tile campaign has measured every prefill shape since the tunedb existed. The GEMV
 * path — which is ALL of decode; the decode program contains zero `Gemm` ops — has never had
 * a shape campaign at all. Its row bucket is selected by CAPACITY (does the object's compiled
 * MM cover the packet's M) and never by measurement, and `runtime/ubench/gemv_bw_gfx950.hip`
 * is a bandwidth microbench, not a sweep over real shapes.
 *
 * WHAT THESE ARE FOR, precisely. Not per-op tile selection — there is none to make; MM is a
 * macro of the object and the K-unroll is a runtime branch inside the kernel. They exist to
 * produce the M CURVE at the shapes the compiler actually asks for, which is the only thing
 * that can price the two OBJECT-level decisions: which `PLOW_GEMV_MM`, and whether the walk
 * is on. `plans/knob-contract.md` §6g-BATCH has the endpoint of that curve as aggregate tok/s
 * (57.9 / 106.5 / 141.7 / 202.3 / 142.4 at B=1/2/4/8/16, so B=8 is the peak and B=16 a 30%
 * regression); this is the per-shape decomposition underneath it.
 *
 * EVERY VARIANT WALKS, and that is deliberate. The loop below IS `gemv_walk` written out at an
 * explicit MM instead of at the `PLOW_GEMV_MM` macro, so one object can carry all five rungs.
 * At M == MM it makes exactly one trip and is the non-walking form; at M > MM it is the arm
 * §6g-WALK proposes. Both are therefore measurable from one build, which is the only way to
 * put them in the same table — a separate object per rung would put them under separate build
 * digests and they could never be compared as the same measurement.
 *
 * The staging is INSIDE the loop, so the LDS bound is `min(MM, M) * K` and not `M * K`. That
 * is the whole correctness claim of the walk, and it is what lets MM=8 serve M=16 at
 * K=5376 where `16 * 5376 = 86016 > GM_LDS_HALVES = 73728` would have overrun.
 */
#define GEMV_WALK_VARIANT(NAME, MM)                                                          \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(         \
        bf16* C, const bf16* x, const bf16* W, const float* rms, const bf16* gamma,          \
        unsigned M, unsigned N, unsigned K, unsigned norm) {                                 \
        __shared__ bf16 lds[GM_LDS_HALVES];                                                  \
        for (unsigned m0 = 0; m0 < M; m0 += (MM)) {                                          \
            const unsigned rem = M - m0, M_ = rem < (MM) ? rem : (MM);                       \
            d_gemv_t<MM>(C + (size_t)m0 * N, x + (size_t)m0 * K, W, rms, gamma, M_, N, K,    \
                         (int)norm, 1e-6f, blockIdx.x, gridDim.x, lds);                      \
            __syncthreads();                                                                 \
        }                                                                                    \
    }

/* Fused gate|up + SwiGLU epilogue — the `glu_fused` arm the LDS gate turns off at t=16 today.
 * Body mirrors `d_gemv_glu` exactly, with MM explicit; the K==4096 unroll switch is kept
 * because it is a real shape specialisation (Llama/Qwen `nchunk` is exactly 8 there) and
 * dropping it would measure a kernel the interpreter does not run. */
#define GEMV_GLU_WALK_VARIANT(NAME, MM)                                                      \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(         \
        bf16* C, const bf16* x, const bf16* Wg, const bf16* Wu, unsigned M, unsigned N,      \
        unsigned K, unsigned act) {                                                          \
        __shared__ bf16 lds[GM_LDS_HALVES];                                                  \
        for (unsigned m0 = 0; m0 < M; m0 += (MM)) {                                          \
            const unsigned rem = M - m0, M_ = rem < (MM) ? rem : (MM);                       \
            const bf16* x_ = x + (size_t)m0 * K;                                             \
            bf16* C_ = C + (size_t)m0 * N;                                                   \
            for (unsigned i = threadIdx.x; i < M_ * K; i += PLOW_THREADS) lds[i] = x_[i];    \
            __syncthreads();                                                                 \
            if (K == 4096)                                                                   \
                gemv_glu_rows<MM, PLOW_GLU_K4096_UN>(C_, Wg, Wu, M_, N, K, act, blockIdx.x,  \
                                                     gridDim.x, lds);                        \
            else                                                                             \
                gemv_glu_rows<MM>(C_, Wg, Wu, M_, N, K, act, blockIdx.x, gridDim.x, lds);    \
            __syncthreads();                                                                 \
        }                                                                                    \
    }

/* Fused Q|K|V — the `fuse_qkv` arm, and the one whose UNENFORCED LDS precondition produced
 * §6g-BATCH's third silent-corruption bug (slots 13/14/15 fluent-but-wrong at B=16, because
 * `gemv_qkvg_rows` has no global-read arm and exactly 13 of 16 rows fit the arena).
 *
 * `Ng = 0` with Cg/Wg null is the 3-stream form of the 4-stream body, exactly as `d_gemv_qkv`
 * forwards into it. Before this, these five variants still called `gemv_qkv_rows` — the name
 * the body carried BEFORE `DevOp::GemvQkvg` merged it into `gemv_qkvg_rows` — so
 * `test_kernels.hip` did not compile at all on that commit, and with it went the whole golden
 * object: every `runtime/tests` golden and both ubench sweeps load `test_kernels.elf`.
 *
 * NOT MIRRORED HERE, deliberately: `d_gemv_qkv` now also takes a `K == 7168 -> UN=7` rung (the
 * commit prices it at 13% of the fusion's win), which these branches do not have. Adding it
 * would change what the bf16 q|k|v arm measures, so it is left as a stated divergence for the
 * owner of that body rather than changed underneath a benchmark. */
#define GEMV_QKV_WALK_VARIANT(NAME, MM)                                                      \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(         \
        bf16* Cq, bf16* Ck, bf16* Cv, const bf16* x, const bf16* Wq, const bf16* Wk,         \
        const bf16* Wv, unsigned M, unsigned Nq, unsigned Nk, unsigned Nv, unsigned K) {     \
        __shared__ bf16 lds[GM_LDS_HALVES];                                                  \
        for (unsigned m0 = 0; m0 < M; m0 += (MM)) {                                          \
            const unsigned rem = M - m0, M_ = rem < (MM) ? rem : (MM);                       \
            const bf16* x_ = x + (size_t)m0 * K;                                             \
            for (unsigned i = threadIdx.x; i < M_ * K; i += PLOW_THREADS) lds[i] = x_[i];    \
            __syncthreads();                                                                 \
            /* `gemv_qkv_rows` became `gemv_qkvg_rows` when the fourth (gate) stream landed;   \
             * op_gemm.h's own `d_gemv_qkv` is now a forwarder that passes `Cg/Wg = nullptr`   \
             * and `Ng = 0`, and this golden wrapper takes the same route. It calls the        \
             * `_rows` template rather than `d_gemv_qkv` because the walk fixes the unroll per  \
             * K, which the entry point chooses for itself. */                                  \
            if (K == 2560)                                                                   \
                gemv_qkvg_rows<MM, 5>(Cq + (size_t)m0 * Nq, Ck + (size_t)m0 * Nk,            \
                                      Cv + (size_t)m0 * Nv, nullptr, Wq, Wk, Wv, nullptr,    \
                                      M_, Nq, Nk, Nv, 0u, K, blockIdx.x, gridDim.x, lds);    \
            else if (K == 4096)                                                              \
                gemv_qkvg_rows<MM, 8>(Cq + (size_t)m0 * Nq, Ck + (size_t)m0 * Nk,            \
                                      Cv + (size_t)m0 * Nv, nullptr, Wq, Wk, Wv, nullptr,    \
                                      M_, Nq, Nk, Nv, 0u, K, blockIdx.x, gridDim.x, lds);    \
            else                                                                             \
                gemv_qkvg_rows<MM, 6>(Cq + (size_t)m0 * Nq, Ck + (size_t)m0 * Nk,            \
                                      Cv + (size_t)m0 * Nv, nullptr, Wq, Wk, Wv, nullptr,    \
                                      M_, Nq, Nk, Nv, 0u, K, blockIdx.x, gridDim.x, lds);    \
            __syncthreads();                                                                 \
        }                                                                                    \
    }

/* The five rungs `PLOW_GEMV_MM` can take (`PLOW_GEMV_MAXM` caps at 16). `tunedb::gemv`'s
 * `gemv_sample_bucket` parses the bucket back out of these names, so the `_m<N>` suffix is
 * load-bearing, not decoration. */
GEMV_WALK_VARIANT(gemv_m1, 1)
GEMV_WALK_VARIANT(gemv_m2, 2)
GEMV_WALK_VARIANT(gemv_m4, 4)
GEMV_WALK_VARIANT(gemv_m8, 8)
GEMV_WALK_VARIANT(gemv_m16, 16)
GEMV_GLU_WALK_VARIANT(gemv_glu_m1, 1)
GEMV_GLU_WALK_VARIANT(gemv_glu_m2, 2)
GEMV_GLU_WALK_VARIANT(gemv_glu_m4, 4)
GEMV_GLU_WALK_VARIANT(gemv_glu_m8, 8)
GEMV_GLU_WALK_VARIANT(gemv_glu_m16, 16)
GEMV_QKV_WALK_VARIANT(gemv_qkv_m1, 1)
GEMV_QKV_WALK_VARIANT(gemv_qkv_m2, 2)
GEMV_QKV_WALK_VARIANT(gemv_qkv_m4, 4)
GEMV_QKV_WALK_VARIANT(gemv_qkv_m8, 8)
GEMV_QKV_WALK_VARIANT(gemv_qkv_m16, 16)

/* D=128 (Llama/Qwen) prefill. Exercises the BKV=64 two-subtile online-softmax path;
 * O_final=nullptr so the CPU-reference test drives the partial+merge path like the
 * 256/512 cases. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_flash_prefill_128(
    float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, unsigned n_q,
    unsigned n_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0, unsigned window,
    float scale, unsigned nsplit) {
    __shared__ bf16 lds[FA_LDS_HALVES(128)];
    d_flash_prefill<128>(Opart, mlpart, /*O_final=*/nullptr, Q, K, V, n_q, n_kv, n_head, n_kv_head,
                         q_pos0, window, scale, n_kv, PLOW_KV_MASK_NONE, nsplit, blockIdx.x,
                         gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_flash_merge_128(
    bf16* O, const float* Opart, const float* mlpart, unsigned n_batch, unsigned n_head,
    unsigned nsplit) {
    d_flash_merge<128>(O, Opart, mlpart, n_batch, n_head, nsplit, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_flash_prefill_256(
    float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, unsigned n_q,
    unsigned n_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0, unsigned window,
    float scale, unsigned nsplit) {
    __shared__ bf16 lds[FA_LDS_HALVES(256)];
    /* kv_stride == n_kv here: the golden cache is exactly n_kv rows. In the model it is the
     * ALLOCATED context, which is why it has to be carried rather than inferred. */
    d_flash_prefill<256>(Opart, mlpart, /*O_final=*/nullptr, Q, K, V, n_q, n_kv, n_head, n_kv_head,
                         q_pos0, window, scale, n_kv, PLOW_KV_MASK_NONE, nsplit, blockIdx.x,
                         gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_flash_prefill_512(
    float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, unsigned n_q,
    unsigned n_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0, unsigned window,
    float scale, unsigned nsplit) {
    __shared__ bf16 lds[FA_LDS_HALVES(512)];
    d_flash_prefill<512>(Opart, mlpart, /*O_final=*/nullptr, Q, K, V, n_q, n_kv, n_head, n_kv_head,
                         q_pos0, window, scale, n_kv, PLOW_KV_MASK_NONE, nsplit, blockIdx.x,
                         gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_flash_decode_256(
    float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, const int* kv_len,
    unsigned n_batch, unsigned n_head, unsigned n_kv_head, unsigned kv_stride, unsigned window,
    float scale, unsigned nsplit) {
    __shared__ float lds[FA_DEC_LDS_FLOATS(256, PLOW_FA_GF(256))];
    d_flash_decode<256, PLOW_FA_GF(256)>(Opart, mlpart, Q, K, V, kv_len, n_batch, n_head, n_kv_head, kv_stride,
                        window, scale, nsplit, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_flash_decode_512(
    float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, const int* kv_len,
    unsigned n_batch, unsigned n_head, unsigned n_kv_head, unsigned kv_stride, unsigned window,
    float scale, unsigned nsplit) {
    __shared__ float lds[FA_DEC_LDS_FLOATS(512, PLOW_FA_GF(512))];
    d_flash_decode<512, PLOW_FA_GF(512)>(Opart, mlpart, Q, K, V, kv_len, n_batch, n_head, n_kv_head, kv_stride,
                        window, scale, nsplit, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_flash_merge_256(
    bf16* O, const float* Opart, const float* mlpart, unsigned n_batch, unsigned n_head,
    unsigned nsplit) {
    d_flash_merge<256>(O, Opart, mlpart, n_batch, n_head, nsplit, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_flash_merge_512(
    bf16* O, const float* Opart, const float* mlpart, unsigned n_batch, unsigned n_head,
    unsigned nsplit) {
    d_flash_merge<512>(O, Opart, mlpart, n_batch, n_head, nsplit, blockIdx.x, gridDim.x);
}

/* DeepSeek MLA decode: latent KV (DK=512) + shared rope key (DR=64), GF=8 head fusion.
 * Reuses gemma_flash_merge_512 for the latent-wide LSE merge; mla_o_uv_fold_512 folds
 * the merged latent down to v_head_dim.  See op_attention.h d_flash_mla_decode. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_flash_decode_512(
    float* Opart, float* mlpart, const bf16* Qabs, const bf16* Qrope, const bf16* Ckv,
    const bf16* Krope, const int* kv_len, unsigned n_batch, unsigned n_head, unsigned kv_stride,
    unsigned window, float scale, unsigned nsplit) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    d_flash_mla_decode<512, 64, GLM_MLA_GF>(Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len, n_batch, n_head,
                                   kv_stride, window, scale, nsplit, PLOW_KV_MASK_NONE, blockIdx.x,
                                   gridDim.x, lds);
}

/* MLA PREFILL — n_tok queries, per-token causal bound. [DEEPSEEK-MLA]
 * Emits the SAME (Opart,m,l) partials, now [b][t][head][1][DK] -> gemma_flash_merge_512 +
 * mla_o_uv_fold_512 per (b,t,head). nsplit is fixed at 1 (see op_attention.h).
 *
 * ORACLE NOTE for hardware day: at n_tok=1 this must be BIT-IDENTICAL to
 * mla_flash_decode_512 with nsplit=1 — same kernel, T=1. Run that equality first; it
 * validates the whole generalisation against an already-validated kernel before any new
 * reference is needed. Then check t<n_tok-1 rows against runtime/tests/mla_ref.rs with the
 * context truncated to len-n_tok+t+1, which is what the causal bound makes them equal to. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_flash_prefill_512(
    float* Opart, float* mlpart, const bf16* Qabs, const bf16* Qrope, const bf16* Ckv,
    const bf16* Krope, const int* kv_len, unsigned n_batch, unsigned n_tok, unsigned n_head,
    unsigned kv_stride, unsigned window, float scale) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    d_flash_mla_prefill<512, 64, GLM_MLA_GF>(Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len,
                                             n_batch, n_tok, n_head, kv_stride, window, scale,
                                             PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds);
}

/* MLA PREFILL, TILED MFMA — the shipped path (PLOW_MLA_PF_MFMA). Same operands, same
 * (Opart,m,l) partials, so the harness can run it against `mla_flash_prefill_512` on one
 * fixture and diff. It is NOT bit-exact against that kernel and must not be checked as if
 * it were: MFMA sums the DK contraction in a different order, so the gate is a relative
 * tolerance. What IS exact is the fp8 association — see the kernel's header. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void mla_flash_prefill_mfma_512(float* Opart, float* mlpart, const bf16* Qabs,
                                const bf16* Qrope, const bf16* Ckv, const bf16* Krope,
                                const int* kv_len, unsigned n_batch, unsigned n_tok,
                                unsigned n_head, unsigned kv_stride, unsigned window,
                                float scale) {
    __shared__ float lds[FA_MLA_PF_LDS_FLOATS(512, 64)];
    d_flash_mla_prefill_mfma<512, 64, /*FP8=*/false>(
        Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len, n_batch, n_tok, n_head, kv_stride,
        window, scale, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds, nullptr, 0);
}

/* FP8-LATENT MLA prefill, both bodies. This is op 110 — the arm Kimi-K3 actually dispatches
 * — so it gets its own pair of entry points rather than riding on the bf16 check. `kv_scale`
 * is the two-strip array d_flash_mla_decode documents: ckv row scales, then krot row scales.
 * The tiled kernel does NOT dequantize into LDS, so the two bodies associate the scales
 * differently by design and the harness compares them with a tolerance. [MLA-FP8-KV] */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void mla_flash_prefill_fp8_512(float* Opart, float* mlpart, const bf16* Qabs,
                               const bf16* Qrope, const bf16* Ckv, const bf16* Krope,
                               const int* kv_len, unsigned n_batch, unsigned n_tok,
                               unsigned n_head, unsigned kv_stride, unsigned window,
                               float scale, const float* kv_scale, unsigned krot_fp8) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    d_flash_mla_prefill_fp8<512, 64, GLM_MLA_GF>(Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len,
                                                 n_batch, n_tok, n_head, kv_stride, window,
                                                 scale, PLOW_KV_MASK_NONE, blockIdx.x,
                                                 gridDim.x, lds, kv_scale, krot_fp8);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void mla_flash_prefill_mfma_fp8_512(float* Opart, float* mlpart, const bf16* Qabs,
                                    const bf16* Qrope, const bf16* Ckv, const bf16* Krope,
                                    const int* kv_len, unsigned n_batch, unsigned n_tok,
                                    unsigned n_head, unsigned kv_stride, unsigned window,
                                    float scale, const float* kv_scale, unsigned krot_fp8) {
    __shared__ float lds[FA_MLA_PF_LDS_FLOATS(512, 64)];
    d_flash_mla_prefill_mfma<512, 64, /*FP8=*/true>(
        Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len, n_batch, n_tok, n_head, kv_stride,
        window, scale, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds, kv_scale, krot_fp8);
}

/* GATHERED MLA prefill (GLM DSA / DeepSeek sparse). idx is [b][t][top_k] — one selected set
 * PER QUERY, the axis the decode gather does not have. No mask: the selected set is causal by
 * construction, matching FLASH_GATHER_DECODE. [DEEPSEEK-MLA] */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_gather_prefill_512(
    float* Opart, float* mlpart, const bf16* Qabs, const bf16* Qrope, const bf16* Ckv,
    const bf16* Krope, const int* kv_len, const int* idx, unsigned top_k, unsigned n_batch,
    unsigned n_tok, unsigned n_head, unsigned kv_stride, float scale) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    d_flash_gather_prefill<512, 64, GLM_MLA_GF>(Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len,
                                                idx, top_k, n_batch, n_tok, n_head, kv_stride,
                                                scale, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x,
                                                lds);
}

/* Head-packed MFMA MLA decode (all heads on the MFMA M-dim; latent read once). Emits the
 * SAME (Opart,m,l) partials -> reuse gemma_flash_merge_512 + mla_o_uv_fold_512. [DEEPSEEK-MLA] */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_flash_decode_mfma_512(
    float* Opart, float* mlpart, const bf16* Qabs, const bf16* Qrope, const bf16* Ckv,
    const bf16* Krope, const int* kv_len, unsigned n_batch, unsigned n_head, unsigned kv_stride,
    unsigned window, float scale, unsigned nsplit) {
    __shared__ float lds[FA_MLA_MFMA_LDS_FLOATS(512, 64)];
    d_flash_mla_decode_mfma<512, 64>(Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len, n_batch,
                                     n_head, kv_stride, window, scale, nsplit, PLOW_KV_MASK_NONE,
                                     blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_gather_decode_mfma_512(
    float* Opart, float* mlpart, const bf16* Qabs, const bf16* Qrope, const bf16* Ckv,
    const bf16* Krope, const int* kv_len, const int* idx, unsigned top_k, unsigned n_batch,
    unsigned n_head, unsigned kv_stride, float scale, unsigned nsplit) {
    __shared__ float lds[FA_MLA_MFMA_LDS_FLOATS(512, 64)];
    d_flash_mla_decode_mfma<512, 64, /*GATHER=*/true>(Opart, mlpart, Qabs, Qrope, Ckv, Krope,
                                                      kv_len, n_batch, n_head, kv_stride,
                                                      /*window=*/0, scale, nsplit, PLOW_KV_MASK_NONE,
                                                      blockIdx.x, gridDim.x, lds, idx, top_k);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_o_uv_fold_512(
    bf16* O, const bf16* Olat, const bf16* Wuv, unsigned n_batch, unsigned n_head, unsigned V) {
    __shared__ bf16 lds[512];
    d_o_uv_fold<512>(O, Olat, Wuv, n_batch, n_head, V, blockIdx.x, gridDim.x, lds);
}

/* Fused MLA merge+fold (kills the external d_flash_merge pass + Olat round-trip). Two VT points
 * to A/B the merge-redundancy vs fold-occupancy tradeoff: VT=256 = whole-V per WG (no redundancy,
 * fusion-only, nh_l WGs); VT=32 = 8 V-tiles/head (fills 8x more CUs). See d_mla_merge_fold.
 * The arena is DK + PLOW_WAVES*VT floats: olds, then the fold's cross-wave l-slice scratch. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_merge_fold_512_v256(
    bf16* O, const float* Opart, const float* mlpart, const bf16* Wuv, unsigned n_batch,
    unsigned n_head, unsigned V, unsigned nsplit) {
    __shared__ float olds[512 + PLOW_WAVES * 256];
    d_mla_merge_fold<512, 256>(O, Opart, mlpart, Wuv, n_batch, n_head, V, nsplit, blockIdx.x,
                               gridDim.x, olds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_merge_fold_512_v32(
    bf16* O, const float* Opart, const float* mlpart, const bf16* Wuv, unsigned n_batch,
    unsigned n_head, unsigned V, unsigned nsplit) {
    __shared__ float olds[512 + PLOW_WAVES * 32];
    d_mla_merge_fold<512, 32>(O, Opart, mlpart, Wuv, n_batch, n_head, V, nsplit, blockIdx.x,
                              gridDim.x, olds);
}

/* DSA lightning-indexer score (DI=index_head_dim=128). score[t] = sum_h w[h]*ReLU(q[h].k[t]). */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void index_score_128(
    float* Score, const bf16* Qidx, const bf16* Kidx, const bf16* W, const int* kv_len,
    unsigned n_batch, unsigned index_heads, unsigned kv_stride, float scale) {
    __shared__ bf16 qlds[32 * 128]; /* index_n_heads(32) * DI(128) */
    d_index_score<128>(Score, Qidx, Kidx, W, kv_len, n_batch, index_heads, kv_stride, scale,
                       blockIdx.x, gridDim.x, qlds);
}

/* DSA lightning-indexer score — coalesced/register-cached (HI=32, DI=128); see d_index_score_fast. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void index_score_fast_128(
    float* Score, const bf16* Qidx, const bf16* Kidx, const bf16* W, const int* kv_len,
    unsigned n_batch, unsigned index_heads /*must be 32*/, unsigned kv_stride, float scale) {
    __shared__ bf16 qlds[32 * 128];
    (void)index_heads;
    d_index_score_fast<128, 32>(Score, Qidx, Kidx, W, kv_len, n_batch, kv_stride, scale,
                                blockIdx.x, gridDim.x, qlds);
}

/* DSA lightning-indexer score — WIDE-K MFMA (perf floor 1). Same signature as index_score_fast_128
 * (drop-in for the sibling's emit wiring); streams the key matrix contiguously through LDS and runs
 * the [HI x DI].[DI x ctx] score-dot as 32x32x16 bf16 MFMA subtiles. See d_index_score_mfma. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void index_score_mfma_128(
    float* Score, const bf16* Qidx, const bf16* Kidx, const bf16* W, const int* kv_len,
    unsigned n_batch, unsigned index_heads /*must be 32*/, unsigned kv_stride, float scale) {
    __shared__ bf16 qlds[32 * (128 + FA_PAD)];        /* HI(32) x padded DI            */
    __shared__ bf16 ktile[PLOW_WAVES * 32 * (128 + FA_PAD)]; /* TILE_N x padded DI      */
    __shared__ float wlds[32];                        /* per-head lightning weights    */
    (void)index_heads;
    /* Standalone launch: the workgroup id IS the logical slice. The persistent interpreter passes
     * `e.slice` instead — see the note at the `st = slice` loop in d_index_score_mfma. */
    d_index_score_mfma<128, 32>(Score, Qidx, Kidx, W, kv_len, n_batch, kv_stride, scale,
                                blockIdx.x, gridDim.x, qlds, ktile, wlds);
}

/* DSA top-k SELECT via radix threshold (G4) — ONE cooperative launch (gridDim==nCU, all WGs
 * co-resident) that grid-synchronises between radix passes. Reads f32 scores from HBM, writes top_k
 * idx. gHist + gCtl[3] must be zero on entry; the kernel leaves them clean. This BASELINE wrapper
 * pins the flat 7-pass / serial-scan / plain-barrier configuration (bench "sel" column). */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void index_select_coop(
    int* idx, const float* Score, unsigned len, unsigned top_k, unsigned* gHist, unsigned* gCtl,
    const int* kv_len) {
    __shared__ unsigned lh[SEL_NB]; /* 8-bit radix histogram (256 bins) */
    __shared__ unsigned red[2];
    d_index_select_coop<false, false, false>(idx, Score, len, top_k, gHist, gCtl, blockIdx.x,
                                             gridDim.x, lh, red, kv_len);
}

/* SHIPPED SELECT (bench "sel-a" column) — the configuration d_index_select_coop defaults to, which
 * the persistent interp inherits with no ABI change: coherent-atomic grid barrier + PARALLEL LDS
 * histogram read-back + the byte-aligned-key fewer-passes fast path (4 score passes; the 3 index
 * passes run only to split a genuine score tie). Selection is EXACT (== CPU radix top-k, lowest-index
 * tie-break) — measured ~2.1-3.4x faster than the baseline above at 8k/32k/128k/256k. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void index_select_coop_a(
    int* idx, const float* Score, unsigned len, unsigned top_k, unsigned* gHist, unsigned* gCtl,
    const int* kv_len) {
    __shared__ unsigned lh[SEL_NB];
    __shared__ unsigned red[4];
    /* Standalone launch: workgroup id == logical slice. The interpreter passes `e.slice`. */
    d_index_select_coop<true, true, true>(idx, Score, len, top_k, gHist, gCtl, blockIdx.x,
                                          gridDim.x, lh, red, kv_len);
}

/* DSA selector (§3.2): on-device top-k KV selection -> idx table for the gather. Scores the
 * indexer query against every KV index key, packed-key rank, lowest-index tie-break. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void attn_select(
    int* idx, const bf16* Qidx, const bf16* Kidx, const int* kv_len, unsigned n_batch,
    unsigned index_dim, unsigned kv_stride, unsigned top_k, float scale) {
    __shared__ unsigned long long keys[8192];
    d_attn_select(idx, Qidx, Kidx, kv_len, n_batch, index_dim, kv_stride, top_k, scale, blockIdx.x,
                  gridDim.x, keys);
}

/* Sparse DSA compose (§3.5): gathered MLA decode — the SAME latent flash reading only the
 * top_k SELECTED latent rows via the idx table.  Composes over MLA; emits latent partials for
 * the same gemma_flash_merge_512 + mla_o_uv_fold_512 tail. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_gather_decode_512(
    float* Opart, float* mlpart, const bf16* Qabs, const bf16* Qrope, const bf16* Ckv,
    const bf16* Krope, const int* kv_len, const int* idx, unsigned top_k, unsigned n_batch,
    unsigned n_head, unsigned kv_stride, float scale, unsigned nsplit) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    d_flash_mla_decode<512, 64, GLM_MLA_GF, /*GATHER=*/true>(Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len,
                                                    n_batch, n_head, kv_stride, /*window=*/0, scale,
                                                    nsplit, PLOW_KV_MASK_NONE, blockIdx.x,
                                                    gridDim.x, lds, idx, top_k);
}

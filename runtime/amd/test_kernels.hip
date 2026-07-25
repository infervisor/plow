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

/* Block-fp8 (DeepSeek/GLM weight_block_size [128,128]) decode GEMV. wscale is the
 * ceil(N/128) x ceil(K/128) f32 dequant grid. One workgroup (nblk=1, slice=0). */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemv_fp8_blk(
    bf16* C, const bf16* x, const unsigned char* W, const float* wscale, unsigned M, unsigned N,
    unsigned K) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_fp8_blk(C, x, W, wscale, M, N, K, blockIdx.x, gridDim.x, lds);
}

/* Block-fp8 MoE expert gate/up + down, one slot, driven through the real table indirection. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void moe_expert_glu_fp8_blk_k(
    bf16* fu, const bf16* x, const unsigned char* table, const unsigned long long* wtab,
    const unsigned long long* stab, unsigned I_moe, unsigned H, unsigned n_exp, unsigned act) {
    d_moe_expert_glu_fp8_blk(fu, x, table, wtab, stab, 0, I_moe, H, n_exp, act, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void moe_expert_down_fp8_blk_k(
    float* part, const bf16* fu, const unsigned char* table, const unsigned long long* wtab,
    const unsigned long long* stab, unsigned H, unsigned I_moe, unsigned n_exp) {
    d_moe_expert_down_fp8_blk(part, fu, table, wtab, stab, 0, H, I_moe, n_exp, blockIdx.x, gridDim.x);
}

/* Block-fp8 DENSE MLP gate/up (fused SwiGLU) on named weights — GLM dense layers 0-2. Op 47. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void dense_glu_fp8_blk_k(
    bf16* fu, const bf16* x, const unsigned char* Wg, const unsigned char* Wu, const float* Sg,
    const float* Su, unsigned N, unsigned K, unsigned act) {
    d_dense_glu_fp8_blk(fu, x, Wg, Wu, Sg, Su, N, K, act, blockIdx.x, gridDim.x);
}

GEMV_VARIANT(gemma_gemv_bf16, 1) /* the default decode bucket */
GEMV_VARIANT(gemma_gemv_m4, 4)
GEMV_VARIANT(gemma_gemv_m8, 8)
GEMV_VARIANT(gemma_gemv_m16, 16)

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
 * fusion-only, nh_l WGs); VT=32 = 8 V-tiles/head (fills 8x more CUs). See d_mla_merge_fold. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_merge_fold_512_v256(
    bf16* O, const float* Opart, const float* mlpart, const bf16* Wuv, unsigned n_batch,
    unsigned n_head, unsigned V, unsigned nsplit) {
    __shared__ float olds[512];
    d_mla_merge_fold<512, 256>(O, Opart, mlpart, Wuv, n_batch, n_head, V, nsplit, blockIdx.x,
                               gridDim.x, olds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_merge_fold_512_v32(
    bf16* O, const float* Opart, const float* mlpart, const bf16* Wuv, unsigned n_batch,
    unsigned n_head, unsigned V, unsigned nsplit) {
    __shared__ float olds[512];
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
    d_index_score_mfma<128, 32>(Score, Qidx, Kidx, W, kv_len, n_batch, kv_stride, scale,
                                gridDim.x, qlds, ktile, wlds);
}

/* DSA top-k SELECT via radix threshold (G4) — ONE cooperative launch (gridDim==nCU, all WGs
 * co-resident) that grid-synchronises between radix passes. Reads f32 scores from HBM, writes top_k
 * idx. gHist + gCtl[3] must be zero on entry; the kernel leaves them clean. This BASELINE wrapper
 * pins the flat 7-pass / serial-scan / plain-barrier configuration (bench "sel" column). */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void index_select_coop(
    int* idx, const float* Score, unsigned len, unsigned top_k, unsigned* gHist, unsigned* gCtl) {
    __shared__ unsigned lh[SEL_NB]; /* 8-bit radix histogram (256 bins) */
    __shared__ unsigned red[2];
    d_index_select_coop<false, false, false>(idx, Score, len, top_k, gHist, gCtl, gridDim.x, lh, red);
}

/* SHIPPED SELECT (bench "sel-a" column) — the configuration d_index_select_coop defaults to, which
 * the persistent interp inherits with no ABI change: coherent-atomic grid barrier + PARALLEL LDS
 * histogram read-back + the byte-aligned-key fewer-passes fast path (4 score passes; the 3 index
 * passes run only to split a genuine score tie). Selection is EXACT (== CPU radix top-k, lowest-index
 * tie-break) — measured ~2.1-3.4x faster than the baseline above at 8k/32k/128k/256k. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void index_select_coop_a(
    int* idx, const float* Score, unsigned len, unsigned top_k, unsigned* gHist, unsigned* gCtl) {
    __shared__ unsigned lh[SEL_NB];
    __shared__ unsigned red[4];
    d_index_select_coop<true, true, true>(idx, Score, len, top_k, gHist, gCtl, gridDim.x, lh, red);
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

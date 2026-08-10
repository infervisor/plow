/* CUDA GEMM kernels.
 *   variant 0x00 (GOLDEN): single-thread naive reference — must match the CPU
 *                          oracle bit-for-bit-ish (fp tolerance).
 *   variant 0x01 (BF16):   Hopper wgmma / Blackwell tcgen05 — extracted from
 *                          fast.cu / ThunderKittens (see runtime/extern).
 *   variant 0x02 (FP8):    DeepGEMM (SM90/SM100, fine-grained scaling).
 *   variant 0x03 (W4A8):   LiquidGEMM.
 *
 * Build-gated: compiled only with -DPLOW_CUDA=ON on a CUDA toolkit. The
 * performant launches are stubbed (TODO) until the reference repos are vendored.
 */
#include "gpu_common.h"

// ---- naive golden: one thread does the whole tile (correctness only) -------
__global__ void plow_gemm_naive_f32(float* c, const float* a, const float* b,
                                    const float* bias, unsigned m, unsigned n, unsigned k) {
    for (unsigned i = 0; i < m; ++i)
        for (unsigned j = 0; j < n; ++j) {
            float acc = bias ? bias[j] : 0.0f;
            for (unsigned p = 0; p < k; ++p) acc += a[i * k + p] * b[j * k + p];
            c[i * n + j] = acc;
        }
}

extern "C" void cuda_gemm_golden(const void* body, kctx* ctx) {
    const PlowGemmBody* g = (const PlowGemmBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const float* A = (const float*)ctx->slots[bd->in0];
    const float* B = (const float*)ctx->slots[bd->in1];
    const float* bias = (bd->in2 != PLOW_SLOT_NONE) ? (const float*)ctx->slots[bd->in2] : nullptr;
    float* C = (float*)ctx->slots[g->out];
    plow_gemm_naive_f32<<<1, 1, 0, (GPU_STREAM)ctx->stream>>>(C, A, B, bias, g->m, g->n, g->k);
}

// ---- performant variants (extracted from reference repos) ------------------
extern "C" void cuda_gemm_bf16_hopper(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: adapt fast.cu wgmma.m64.nN.k16 + TMA launcher; map GemmBody{m,n,k,
    //       bm,bn,bk,coord,out} to its grid/cluster params. See extern/fast.cu.
}
extern "C" void cuda_gemm_bf16_blackwell(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: ThunderKittens tcgen05 path (uses GemmBody.tmem). See extern/ThunderKittens.
}
extern "C" void cuda_gemm_fp8(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: DeepGEMM SM90/SM100 fp8 with fine-grained scaling. See extern/DeepGEMM.
}
extern "C" void cuda_gemm_w4a8(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: LiquidGEMM W4A8. See extern/LiquidGEMM.
}

// ==========================================================================
// Gemma 4 fused GEMM variants — single-SM, self-distributed multi-tile.
//
// Contract shared by every variant below:
//  * ONE packet = the whole op (full m,n,k in PlowGemmBody). The kernel is
//    launched with P = #SMs persistent blocks; block s owns output tiles
//    s, s+P, s+2P, ... (grid-stride) and DMAs each tile's operands itself.
//    Tile count T = ceil(m/bm)*ceil(n/bn); coord0/coord1 = base tile origin
//    (coord1 high bits = split-index for the split-K variants).
//  * Worker layout: 1 warp issues TMA (double-buffered B), one warpgroup
//    (4 warps) runs wgmma.m64.nN.k16. gamma resident across all owned tiles.
//  * SRAM budget: prefill BM=128,BN=256,BK=64 -> ~48 KiB tile + 16 KiB gamma;
//    decode BM=64,BN=256,BK=64 + split-K. Both fit 222 KiB usable on H100.
// ==========================================================================

// F1 FusedNormLinear (q/kv/gate/up proj): RMSNorm(A, gamma, eps)-prologue then
// bf16 wgmma. Bindings: in0=A(pre-norm), in1=B(weight), in2=gamma, scale=eps,
// detail=norm-kind(low)|epilogue-act(high). out=result slot.
extern "C" void cuda_gemm_norm_bf16_hopper(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // Phase 0 (in-kernel RMS pre-pass): one warpgroup streams the tile's A
    //   row-block, accumulates per-row Sum(x^2) -> rms[m]=rsqrt(ss/k+eps) into
    //   a shared BM-float vector. ~0.1us vs ms-scale weight DMA.
    // Phase 1 (mainloop): per BK chunk, TMA-load B[bk,BN] (double-buffered);
    //   apply A_norm[m,p]=A[m,p]*rms[m]*gamma[p] while staging A into the wgmma
    //   operand SRAM, then wgmma.m64.nBN.k16 accumulate.
    // Phase 2 (epilogue): optional act (detail>>4), store C[BM,BN] to slots[out].
    // TODO: adapt fast.cu wgmma launcher; fold the norm into the A-stage step.
}
extern "C" void cuda_gemm_norm_bf16_blackwell(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: ThunderKittens tcgen05 path (GemmBody.tmem) + norm prologue as above.
}

// Split-K decode forms: each block's grid-stride index space is T*split_k; it
// derives (tile, split-index) from the flat id, writes a PARTIAL to a scratch
// slot + bumps a counter. A follow-on Row reduce-add packet sums the partials.
// The norm is NOT split — for decode's tiny A each block recomputes rms locally.
extern "C" void cuda_gemm_bf16_splitk(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: plain bf16 wgmma over a K-slice -> partial accumulate. (o/down/lm_head)
}
extern "C" void cuda_gemm_norm_splitk_bf16(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: norm-prologue bf16 wgmma over a K-slice -> partial. (q/kv/gate/up decode)
}

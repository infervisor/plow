/* ROCm/HIP performant kernels for MI300 (CDNA3) — extracted from HipKittens
 * (MFMA 16x16x16 / 32x32x8, LDS staging, 64-wide wavefronts). Build-gated;
 * launches stubbed until extern/ThunderKittens (HipKittens) is vendored. */
#include <hip/hip_runtime.h>
#include "kernel.h"
#include "packet.h"

extern "C" void hip_gemm_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: HipKittens MFMA GEMM. Map GemmBody{m,n,k,bm,bn,bk} to its tile config.
}

extern "C" void hip_flash_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: HipKittens attention (MI300). See extern/ThunderKittens (HipKittens).
}

// ==========================================================================
// Gemma 4 fused MI300 (CDNA3) variants — single-CU, self-distributed. Mirror of
// runtime/nvidia/{gemm,row,flash,layout}.cu but with MFMA (m16n16k16 / m32n32k8),
// 64-wide wavefronts, and the 64 KiB LDS budget. Reductions use ds_swizzle /
// ds_bpermute, NOT __shfl_sync. Same PlowBinding data path as the CUDA side.
//
// LDS pressure (60 KiB usable): norm-prologue GEMM caps BN=128 and keeps gamma
// in bf16 (8 KiB) + single-buffers A (weight B stays double-buffered as the
// DMA-bound operand). Fallback: BK=32. Tile params in plans/gemma-4-31b-*.md.
// ==========================================================================

// F1 FusedNormLinear (q/kv/gate/up): RMSNorm-prologue -> MFMA. in0=A, in1=B,
// in2=gamma(bf16), scale=eps. Prefill BM=128,BN=128,BK=64; decode BM=64 split-K.
extern "C" void hip_gemm_norm_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // Phase 0: per-row Sum(x^2) (64-wide wave reduction) -> rms; Phase 1: apply
    // A*rms*gamma on the A-stage into LDS, MFMA accumulate; Phase 2: store tile.
    // TODO: HipKittens MFMA mainloop + fused norm prologue.
}
// Split-K decode forms (partial accumulate + counter; Row reduce-add combines).
extern "C" void hip_gemm_mfma_splitk(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: plain bf16 MFMA over a K-slice -> partial. (o/down/lm_head decode)
}
extern "C" void hip_gemm_norm_mfma_splitk(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: norm-prologue MFMA over a K-slice -> partial. (q/kv/gate/up decode)
}

// F2/F3 FusedNormRope(Scale) — Row family, per (row, head), 64-wide reduction.
extern "C" void hip_row_normrope_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // rms over head_dim (ds_swizzle) -> *gamma -> RoPE first feat/2 dims. K path.
    // TODO.
}
extern "C" void hip_row_normropescale_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // As normrope, then *1/sqrt(head_dim). Q path. TODO.
}
// F5 SwiGLU / S3 residual / S4 RMSNorm — bandwidth-bound, dwordx4 vector ops.
extern "C" void hip_row_swiglu_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // out = silu(gate)*up (in0=gate,in1=up). TODO.
}
extern "C" void hip_row_residual_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // out = x + residual (in0=x,in1=residual). TODO.
}
extern "C" void hip_row_rmsnorm_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // standalone final RMSNorm (in0=X,in1=gamma,scale=eps). TODO.
}

// S2 attention: qk_norm+RoPE done upstream (Row), so flash is standard. GQA 32:8.
extern "C" void hip_flash_causal_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // Prefill full-causal FA-2 with MFMA + online softmax. BQ=64,BKV=128. TODO.
}
extern "C" void hip_flash_sliding_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // As causal + sliding-window masking. TODO.
}
extern "C" void hip_flash_decode_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // Decode BQ=1: split-KV partials across waves + online-softmax merge. TODO.
}

// F4 FusedEmbeddingScale — strided gather + scale (Layout family).
extern "C" void hip_layout_gather_scale_mfma(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // per token t: out[t,:] = embed[ids[t],:] * scale(sqrt(D)). Memory-bound. TODO.
}

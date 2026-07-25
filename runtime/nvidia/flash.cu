/* CUDA attention. variant 0x00 naive golden; 0x01 ThunderKittens flash. */
#include "gpu_common.h"

__global__ void plow_flash_naive_f32(float* o, const float* q, const float* k,
                                     const float* v, unsigned sq, unsigned skv,
                                     unsigned hd, unsigned heads, int causal) {
    float scale = rsqrtf((float)hd);
    for (unsigned h = 0; h < heads; ++h) {
        const float* qh = q + (size_t)h * sq * hd;
        const float* kh = k + (size_t)h * skv * hd;
        const float* vh = v + (size_t)h * skv * hd;
        float* oh = o + (size_t)h * sq * hd;
        for (unsigned i = 0; i < sq; ++i) {
            unsigned last = causal ? (i < skv ? i : skv - 1) : skv - 1;
            float mx = -1e30f, sum = 0.0f;
            for (unsigned j = 0; j <= last; ++j) {
                float s = 0.0f;
                for (unsigned d = 0; d < hd; ++d) s += qh[i * hd + d] * kh[j * hd + d];
                s *= scale; if (s > mx) mx = s;
            }
            for (unsigned d = 0; d < hd; ++d) oh[i * hd + d] = 0.0f;
            for (unsigned j = 0; j <= last; ++j) {
                float s = 0.0f;
                for (unsigned d = 0; d < hd; ++d) s += qh[i * hd + d] * kh[j * hd + d];
                float e = expf(s * scale - mx); sum += e;
                for (unsigned d = 0; d < hd; ++d) oh[i * hd + d] += e * vh[j * hd + d];
            }
            for (unsigned d = 0; d < hd; ++d) oh[i * hd + d] /= sum;
        }
    }
}

extern "C" void cuda_flash_golden(const void* body, kctx* ctx) {
    const PlowFlashBody* f = (const PlowFlashBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const float* Q = (const float*)ctx->slots[bd->in0];
    const float* K = (const float*)ctx->slots[bd->in1];
    const float* V = (const float*)ctx->slots[bd->in2];
    float* O = (float*)ctx->slots[f->out];
    plow_flash_naive_f32<<<1, 1, 0, (GPU_STREAM)ctx->stream>>>(
        O, Q, K, V, f->seq_q, f->seq_kv, f->head_dim, f->heads, bd->detail != 0);
}

extern "C" void cuda_flash_tk(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: ThunderKittens attention fwd (Hopper/Blackwell). See extern/ThunderKittens.
}

// ==========================================================================
// Gemma 4 attention variants — single-SM, self-distributed. GQA 32:8, hd=128.
// qk_norm + RoPE happen BEFORE flash (Row NormRope kernels), so flash consumes
// already-normed/roped Q,K and stays standard. Bindings: in0=Q,in1=K,in2=V,
// scale=1/sqrt(hd) (+ sliding window), detail=mask flags, out=result.
//
// One packet = whole op. Blocks self-distribute over (head-group, Q-tile).
// ==========================================================================

// Prefill full-causal (FA-2 tiling): BQ=128, BKV=128/256. Load Q tile resident;
// loop KV tiles (TMA double-buffered): S=Q.K^T (wgmma), online-softmax in regs,
// O+=P.V (wgmma); causal-mask the diagonal KV tile.
extern "C" void cuda_flash_causal_bf16(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: FA-2 wgmma mainloop + online softmax. Inspired by ThunderKittens.
}
// Prefill sliding-window: as causal but also skip/mask KV tiles outside the
// window (window size from PlowBinding.scale / FlashBody.coord q-position base).
extern "C" void cuda_flash_sliding_bf16(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: causal path + window masking of far KV tiles.
}
// Decode (BQ=1): no Q-loop. Split the long KV across the block's worker warps
// (split-KV); each warp computes a partial (m_i, l_i, O_i); a final warp merges
// via online-softmax. Distinct kernel because the reduction structure differs.
extern "C" void cuda_flash_decode_bf16(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: split-KV partials + online-softmax merge (both mask kinds via detail).
}

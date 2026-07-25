/* CUDA LAYOUT (v4 descriptor). kind 0: contiguous cudaMemcpyAsync. kind 1: a
 * naive single-thread strided gather/scatter (golden; mirrors cpu_layout). */
#include "gpu_common.h"

/* Scatter `src` (contiguous, shape `pshape`) into `dst` at `out_base`, walking
 * the piece's row-major multi-index with the output's `out_stride`. */
__device__ void plow_scatter_piece(uint8_t* dst, const uint8_t* src, uint32_t r, uint32_t es,
                                   const uint32_t* pshape, const uint32_t* out_stride,
                                   size_t out_base) {
    size_t in_str[PLOW_LAYOUT_MAX_RANK];
    size_t acc = 1;
    for (int d = (int)r - 1; d >= 0; --d) { in_str[d] = acc; acc *= pshape[d]; }
    size_t total = (r > 0) ? 1 : 0;
    for (uint32_t d = 0; d < r; ++d) total *= pshape[d];
    uint32_t idx[PLOW_LAYOUT_MAX_RANK];
    for (uint32_t d = 0; d < r; ++d) idx[d] = 0;
    for (size_t n = 0; n < total; ++n) {
        size_t in_off = 0, out_off = out_base;
        for (uint32_t d = 0; d < r; ++d) {
            in_off += (size_t)idx[d] * in_str[d];
            out_off += (size_t)idx[d] * out_stride[d];
        }
        for (uint32_t k = 0; k < es; ++k) dst[out_off * es + k] = src[in_off * es + k];
        for (int d = (int)r - 1; d >= 0; --d) { if (++idx[d] < pshape[d]) break; idx[d] = 0; }
    }
}

__global__ void plow_layout_strided(uint8_t* dst, const uint8_t* src, const uint8_t* src1,
                                    PlowLayoutBody L) {
    const uint32_t r = L.rank, es = L.elem_size;
    if (L.kind == 2) { /* binary concat */
        const uint32_t axis = L.in_base, split = L.out_base;
        uint32_t p0[PLOW_LAYOUT_MAX_RANK], p1[PLOW_LAYOUT_MAX_RANK];
        for (uint32_t d = 0; d < r; ++d) { p0[d] = L.shape[d]; p1[d] = L.shape[d]; }
        p0[axis] = split;
        p1[axis] = L.shape[axis] - split;
        plow_scatter_piece(dst, src, r, es, p0, L.out_stride, 0);
        plow_scatter_piece(dst, src1, r, es, p1, L.out_stride, (size_t)split * L.out_stride[axis]);
        return;
    }
    size_t total = (r > 0) ? 1 : 0;
    for (uint32_t d = 0; d < r; ++d) total *= L.shape[d];
    uint32_t idx[PLOW_LAYOUT_MAX_RANK];
    for (uint32_t d = 0; d < r; ++d) idx[d] = 0;
    for (size_t n = 0; n < total; ++n) {
        size_t in_off = L.in_base, out_off = L.out_base;
        for (uint32_t d = 0; d < r; ++d) {
            in_off += (size_t)idx[d] * L.in_stride[d];
            out_off += (size_t)idx[d] * L.out_stride[d];
        }
        for (uint32_t k = 0; k < es; ++k) dst[out_off * es + k] = src[in_off * es + k];
        for (int d = (int)r - 1; d >= 0; --d) {
            if (++idx[d] < L.shape[d]) break;
            idx[d] = 0;
        }
    }
}

extern "C" void cuda_layout_golden(const void* body, kctx* ctx) {
    const PlowLayoutBody* L = (const PlowLayoutBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const uint8_t* src = (const uint8_t*)ctx->slots[bd->in0];
    const uint8_t* src1 = (bd->in1 != PLOW_SLOT_NONE) ? (const uint8_t*)ctx->slots[bd->in1] : nullptr;
    uint8_t* dst = (uint8_t*)ctx->slots[L->out];
    if (L->kind == 0) {
        size_t total = (L->rank > 0) ? 1 : 0;
        for (uint32_t d = 0; d < L->rank; ++d) total *= L->shape[d];
        cudaMemcpyAsync(dst, src, total * L->elem_size, cudaMemcpyDeviceToDevice,
                        (cudaStream_t)ctx->stream);
        return;
    }
    plow_layout_strided<<<1, 1, 0, (GPU_STREAM)ctx->stream>>>(dst, src, src1, *L);
}

/* CONTROL family (nop / host-coord): no data effect, counters advance in the
 * interpreter. Mirrors cpu_host. */
extern "C" void cuda_host(const void* body, kctx* ctx) {
    (void)body;
    (void)ctx;
}

// F4 FusedEmbeddingScale (K13, once/forward): gather rows from the embedding
// table by token id, scale by sqrt(D)=64. A strided gather -> lives in Layout.
// Bindings: in0=token-ids, in1=embed table, scale=sqrt(D), out=result.
// Self-distributed: P blocks stride over the T output tokens; each gathers its
// row (random HBM access into the 2 GB table) and scales. Memory-bound.
extern "C" void cuda_layout_gather_scale_bf16(const void* body, kctx* ctx) {
    (void)body; (void)ctx;
    // TODO: per token t: row = embed[ ids[t] , : ]; out[t,:] = row * scale.
    //       Vectorized bf16 copy; coalesce within each gathered row.
}

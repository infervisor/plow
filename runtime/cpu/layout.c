/* CPU golden LAYOUT — strided block copy driven by the v4 descriptor.
 *   kind 0: contiguous copy (fast path) of product(shape)*elem_size bytes.
 *   kind 1: out[out_base+Σidx·out_stride] = in[in_base+Σidx·in_stride] over shape
 *           (transpose / broadcast / slice).
 *   kind 2: binary concat — scatter in0 into [0,split) and in1 into [split,end)
 *           along axis=in_base, split=out_base. Both sources from the binding.
 * Source(s) are the binding's in0(/in1) slots; destination is the body's `out`. */
#include "cpu_kernels.h"
#include <string.h>

/* Row-major element strides of `shape` (rank r) into `stride`. */
static void row_strides(const uint32_t* shape, uint32_t r, size_t* stride) {
    size_t acc = 1;
    for (int d = (int)r - 1; d >= 0; d--) { stride[d] = acc; acc *= shape[d]; }
}

/* Scatter `src` (contiguous, shape `pshape`) into `dst` at `out_base` with the
 * output's `out_stride`, walking the piece's row-major multi-index. */
static void scatter_piece(uint8_t* dst, const uint8_t* src, uint32_t r, uint32_t es,
                          const uint32_t* pshape, const uint32_t* out_stride, size_t out_base) {
    size_t in_str[PLOW_LAYOUT_MAX_RANK];
    row_strides(pshape, r, in_str);
    size_t total = (r > 0) ? 1 : 0;
    for (uint32_t d = 0; d < r; d++) total *= pshape[d];
    uint32_t idx[PLOW_LAYOUT_MAX_RANK] = {0};
    for (size_t n = 0; n < total; n++) {
        size_t in_off = 0, out_off = out_base;
        for (uint32_t d = 0; d < r; d++) {
            in_off += (size_t)idx[d] * in_str[d];
            out_off += (size_t)idx[d] * out_stride[d];
        }
        memcpy(dst + out_off * es, src + in_off * es, es);
        for (int d = (int)r - 1; d >= 0; d--) { if (++idx[d] < pshape[d]) break; idx[d] = 0; }
    }
}

void plow_copy_ref(void* dst, const void* src, uint32_t bytes) {
    memcpy(dst, src, bytes);
}

void cpu_layout(const void* body, kctx* ctx) {
    const PlowLayoutBody* L = (const PlowLayoutBody*)body;
    const PlowBinding* b = ctx->bind;
    if (!b) return;
    const uint8_t* src = (const uint8_t*)ctx->slots[b->in0];
    uint8_t* dst = (uint8_t*)ctx->slots[L->out];
    if (dst == src) return; /* aliased view (reshape): bytes already in place */
    const uint32_t es = L->elem_size;
    const uint32_t r = L->rank;

    size_t total = (r > 0) ? 1 : 0;
    for (uint32_t d = 0; d < r; d++) total *= L->shape[d];

    if (L->kind == 0) {
        plow_copy_ref(dst, src, (uint32_t)(total * es)); /* contiguous */
        return;
    }

    if (L->kind == 2) {
        /* Binary concat: in0 -> [0,split), in1 -> [split,end) along `axis`. */
        const uint8_t* src1 = (const uint8_t*)ctx->slots[b->in1];
        const uint32_t axis = L->in_base, split = L->out_base;
        uint32_t p0[PLOW_LAYOUT_MAX_RANK], p1[PLOW_LAYOUT_MAX_RANK];
        for (uint32_t d = 0; d < r; d++) { p0[d] = L->shape[d]; p1[d] = L->shape[d]; }
        p0[axis] = split;
        p1[axis] = L->shape[axis] - split;
        scatter_piece(dst, src, r, es, p0, L->out_stride, 0);
        scatter_piece(dst, src1, r, es, p1, L->out_stride, (size_t)split * L->out_stride[axis]);
        return;
    }

    /* Strided gather/scatter: walk the row-major output multi-index. */
    uint32_t idx[PLOW_LAYOUT_MAX_RANK] = {0};
    for (size_t n = 0; n < total; n++) {
        size_t in_off = L->in_base, out_off = L->out_base;
        for (uint32_t d = 0; d < r; d++) {
            in_off += (size_t)idx[d] * L->in_stride[d];
            out_off += (size_t)idx[d] * L->out_stride[d];
        }
        memcpy(dst + out_off * es, src + in_off * es, es);
        for (int d = (int)r - 1; d >= 0; d--) { /* increment, last axis fastest */
            if (++idx[d] < L->shape[d]) break;
            idx[d] = 0;
        }
    }
}

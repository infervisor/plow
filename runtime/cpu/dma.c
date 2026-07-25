/* CPU golden DMA — models tma_load/tma_store as a copy between an HBM tensor
 * and an SRAM slot. load: tensor[in0]→slot[out]; store: slot[in0]→tensor[out]. */
#include "cpu_kernels.h"

void cpu_dma(const void* body, kctx* ctx) {
    const PlowDmaBody* d = (const PlowDmaBody*)body;
    const PlowBinding* b = ctx->bind;
    if (!b) return;
    /* in0 is the source handle; slot `d->slot` is the SRAM slot. The load/store
     * direction is encoded by the opcode variant, surfaced via bind->detail
     * (0 = load tensor→slot, 1 = store slot→tensor). */
    if (b->detail == 0) {
        const void* src = ctx->tensors[b->in0];
        plow_copy_ref(ctx->slots[d->slot], src, d->bytes);
    } else {
        void* dst = ctx->tensors[b->in0];
        plow_copy_ref(dst, ctx->slots[d->slot], d->bytes);
    }
}

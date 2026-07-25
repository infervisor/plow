/* CPU golden RDMA — models a cross-unit P2P transfer as a slot→slot copy
 * (single-process host harness: src and dst slots live in the same address
 * space). in0 = source slot, in2 = destination slot. */
#include "cpu_kernels.h"

void cpu_rdma(const void* body, kctx* ctx) {
    const PlowRdmaBody* r = (const PlowRdmaBody*)body;
    const PlowBinding* b = ctx->bind;
    if (!b) return;
    plow_copy_ref(ctx->slots[b->in2], ctx->slots[b->in0], r->bytes);
}

/* CUDA DMA — golden models tma_load/store as cudaMemcpyAsync; the performant
 * path issues real TMA (cp.async.bulk) inside the compute kernels. */
#include "gpu_common.h"

extern "C" void cuda_dma_golden(const void* body, kctx* ctx) {
    const PlowDmaBody* d = (const PlowDmaBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    cudaStream_t s = (cudaStream_t)ctx->stream;
    if (bd->detail == 0) /* load: tensor -> slot */
        cudaMemcpyAsync(ctx->slots[d->slot], ctx->tensors[bd->in0], d->bytes, cudaMemcpyDefault, s);
    else                 /* store: slot -> tensor */
        cudaMemcpyAsync(ctx->tensors[bd->in0], ctx->slots[d->slot], d->bytes, cudaMemcpyDefault, s);
}

extern "C" void cuda_rdma_golden(const void* body, kctx* ctx) {
    const PlowRdmaBody* r = (const PlowRdmaBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    cudaMemcpyAsync(ctx->slots[bd->in2], ctx->slots[bd->in0], r->bytes,
                    cudaMemcpyDeviceToDevice, (cudaStream_t)ctx->stream);
}

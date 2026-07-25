/* cpu_kernels.h — CPU golden kernel_fn adapters (internal to runtime/cpu). */
#ifndef PLOW_CPU_KERNELS_H
#define PLOW_CPU_KERNELS_H

#include "kernel.h"

void cpu_gemm(const void* body, kctx* ctx);
void cpu_flash(const void* body, kctx* ctx);
void cpu_row_reduce(const void* body, kctx* ctx);
void cpu_row_pointwise(const void* body, kctx* ctx);
void cpu_layout(const void* body, kctx* ctx);
void cpu_dma(const void* body, kctx* ctx);
void cpu_rdma(const void* body, kctx* ctx);
void cpu_host(const void* body, kctx* ctx);

#endif /* PLOW_CPU_KERNELS_H */

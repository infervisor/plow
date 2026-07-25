/* Blackwell (sm_100a) dispatch table: golden kernels.
 *
 * Performant variants (tcgen05 bf16, fp8 DeepGEMM, TK flash) are NOT registered
 * until their reference sources are vendored (see extern/README.md): an
 * unimplemented stub would otherwise be dispatched, compute nothing, and report
 * success. Leaving the slot empty makes dt_dispatch return -1 instead. */
#include "dispatch.h"
#include "packet.h"

extern "C" {
void cuda_gemm_golden(const void*, kctx*);
void cuda_flash_golden(const void*, kctx*);
void cuda_row_reduce_golden(const void*, kctx*);
void cuda_row_pointwise_golden(const void*, kctx*);
void cuda_dma_golden(const void*, kctx*);
void cuda_rdma_golden(const void*, kctx*);
void cuda_layout_golden(const void*, kctx*);
void cuda_host(const void*, kctx*);

void plow_register_cuda_blackwell(dispatch_table* dt) {
    dt_init(dt);
    dt_register(dt, PLOW_NOP,            cuda_host);
    dt_register(dt, PLOW_HOST_COORD,     cuda_host);
    dt_register(dt, PLOW_TMA_LOAD,       cuda_dma_golden);
    dt_register(dt, PLOW_TMA_STORE,      cuda_dma_golden);
    dt_register(dt, PLOW_RDMA,           cuda_rdma_golden);
    dt_register(dt, PLOW_LAYOUT,         cuda_layout_golden);
    dt_register(dt, PLOW_ROW_REDUCE,     cuda_row_reduce_golden);
    dt_register(dt, PLOW_ROW_POINTWISE,  cuda_row_pointwise_golden);
    dt_register(dt, PLOW_GEMM,           cuda_gemm_golden);
    dt_register(dt, PLOW_FLASH,          cuda_flash_golden);

    /* Gemma 4 fused variants — same slot map as register_hopper.cu, except the
     * norm-prologue GEMM uses the tcgen05 body:
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_GEMM, PLOW_VARIANT_NORM_BF16),
     *                    cuda_gemm_norm_bf16_blackwell);
     * (the Row/Flash/Layout fused kernels are arch-agnostic and shared with
     * Hopper). NOT registered until their bodies are real. */
}
}

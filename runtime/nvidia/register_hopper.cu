/* Hopper (sm_90a) dispatch table: golden kernels.
 *
 * Performant variants (bf16 wgmma, fp8 DeepGEMM, w4a8, TK flash) are NOT
 * registered until their reference sources are vendored (see extern/README.md):
 * an unimplemented stub would otherwise be dispatched, compute nothing, and
 * report success. Leaving the slot empty makes dt_dispatch return -1 instead. */
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

void plow_register_cuda_hopper(dispatch_table* dt) {
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

    /* Gemma 4 fused variants — skeletons in gemm.cu/row.cu/flash.cu/layout.cu.
     * NOT registered until their bodies are real (unregistered => dt_dispatch
     * returns -1, which is safer than a stub that reports success). Intended
     * mapping (uncomment per variant as it lands; index = family<<8|variant):
     *
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_GEMM,  PLOW_VARIANT_NORM_BF16),
     *                    cuda_gemm_norm_bf16_hopper);           // q/kv/gate/up
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_GEMM,  PLOW_VARIANT_NORM_SPLITK_BF16),
     *                    cuda_gemm_norm_splitk_bf16);           // decode
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_GEMM,  PLOW_VARIANT_BF16_SPLITK),
     *                    cuda_gemm_bf16_splitk);                // o/down/lm_head decode
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_FLASH, PLOW_VARIANT_FLASH_CAUSAL_BF16),
     *                    cuda_flash_causal_bf16);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_FLASH, PLOW_VARIANT_FLASH_SLIDING_BF16),
     *                    cuda_flash_sliding_bf16);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_FLASH, PLOW_VARIANT_FLASH_DECODE_BF16),
     *                    cuda_flash_decode_bf16);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_ROW,   PLOW_VARIANT_ROW_NORMROPE_BF16),
     *                    cuda_row_normrope_bf16);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_ROW,   PLOW_VARIANT_ROW_NORMROPESCALE_BF16),
     *                    cuda_row_normropescale_bf16);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_ROW,   PLOW_VARIANT_ROW_SWIGLU_BF16),
     *                    cuda_row_swiglu_bf16);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_ROW,   PLOW_VARIANT_ROW_RESIDUAL_ADD_BF16),
     *                    cuda_row_residual_bf16);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_ROW,   PLOW_VARIANT_ROW_RMS_BF16),
     *                    cuda_row_rmsnorm_bf16);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_LAYOUT,PLOW_VARIANT_LAYOUT_GATHER_SCALE_BF16),
     *                    cuda_layout_gather_scale_bf16);
     */
}
}

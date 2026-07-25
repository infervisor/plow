/* MI300 (gfx942 / CDNA3) dispatch table: golden kernels.
 *
 * Performant MFMA variants (gemm/flash) are NOT registered until HipKittens is
 * vendored (see extern/README.md): an unimplemented stub would otherwise be
 * dispatched, compute nothing, and report success. Leaving the slot empty makes
 * dt_dispatch return -1 instead. */
#include "dispatch.h"
#include "packet.h"

extern "C" {
void hip_gemm_golden(const void*, kctx*);
void hip_flash_golden(const void*, kctx*);
void hip_row_reduce_golden(const void*, kctx*);
void hip_row_pointwise_golden(const void*, kctx*);
void hip_dma_golden(const void*, kctx*);
void hip_rdma_golden(const void*, kctx*);
void hip_layout_golden(const void*, kctx*);
void hip_host(const void*, kctx*);

void plow_register_rocm_mi300(dispatch_table* dt) {
    dt_init(dt);
    dt_register(dt, PLOW_NOP,            hip_host);
    dt_register(dt, PLOW_HOST_COORD,     hip_host);
    dt_register(dt, PLOW_TMA_LOAD,       hip_dma_golden);
    dt_register(dt, PLOW_TMA_STORE,      hip_dma_golden);
    dt_register(dt, PLOW_RDMA,           hip_rdma_golden);
    dt_register(dt, PLOW_LAYOUT,         hip_layout_golden);
    dt_register(dt, PLOW_ROW_REDUCE,     hip_row_reduce_golden);
    dt_register(dt, PLOW_ROW_POINTWISE,  hip_row_pointwise_golden);
    dt_register(dt, PLOW_GEMM,           hip_gemm_golden);
    dt_register(dt, PLOW_FLASH,          hip_flash_golden);

    /* Gemma 4 fused MFMA variants — skeletons in mfma.hip. NOT registered until
     * their bodies are real (unregistered => dt_dispatch returns -1). Intended
     * mapping (uncomment per variant as it lands):
     *
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_GEMM,  PLOW_VARIANT_NORM_BF16),
     *                    hip_gemm_norm_mfma);                   // q/kv/gate/up
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_GEMM,  PLOW_VARIANT_NORM_SPLITK_BF16),
     *                    hip_gemm_norm_mfma_splitk);            // decode
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_GEMM,  PLOW_VARIANT_BF16_SPLITK),
     *                    hip_gemm_mfma_splitk);                 // o/down/lm_head decode
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_FLASH, PLOW_VARIANT_FLASH_CAUSAL_BF16),
     *                    hip_flash_causal_mfma);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_FLASH, PLOW_VARIANT_FLASH_SLIDING_BF16),
     *                    hip_flash_sliding_mfma);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_FLASH, PLOW_VARIANT_FLASH_DECODE_BF16),
     *                    hip_flash_decode_mfma);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_ROW,   PLOW_VARIANT_ROW_NORMROPE_BF16),
     *                    hip_row_normrope_mfma);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_ROW,   PLOW_VARIANT_ROW_NORMROPESCALE_BF16),
     *                    hip_row_normropescale_mfma);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_ROW,   PLOW_VARIANT_ROW_SWIGLU_BF16),
     *                    hip_row_swiglu_mfma);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_ROW,   PLOW_VARIANT_ROW_RESIDUAL_ADD_BF16),
     *                    hip_row_residual_mfma);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_ROW,   PLOW_VARIANT_ROW_RMS_BF16),
     *                    hip_row_rmsnorm_mfma);
     *   dt_register(dt, PLOW_OP(0,PLOW_FAMILY_LAYOUT,PLOW_VARIANT_LAYOUT_GATHER_SCALE_BF16),
     *                    hip_layout_gather_scale_mfma);
     */
}
}

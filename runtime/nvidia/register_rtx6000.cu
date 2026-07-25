/* register_rtx6000.cu — dispatch table for the RTX PRO 6000 Blackwell
 * (GB202, SM 12.0 / sm_120a, consumer Blackwell).
 *
 * Unlike register_blackwell.cu (datacenter B200, tcgen05) and register_hopper.cu
 * (wgmma), this table wires the sm_120 warp-`mma.sync` Gemma kernels in
 * gemma_sm120.cu. Those bodies are REAL (not stubs), so the fused variants are
 * registered directly. Golden f32 kernels remain the fallback for ops with no
 * bf16 performant form (correctness oracle).
 */
#include "dispatch.h"
#include "packet.h"

extern "C" {
/* golden (f32) fallbacks — shared with the other NVIDIA tables */
void cuda_gemm_golden(const void*, kctx*);
void cuda_flash_golden(const void*, kctx*);
void cuda_row_reduce_golden(const void*, kctx*);
void cuda_row_pointwise_golden(const void*, kctx*);
void cuda_dma_golden(const void*, kctx*);
void cuda_rdma_golden(const void*, kctx*);
void cuda_layout_golden(const void*, kctx*);
void cuda_host(const void*, kctx*);

/* sm_120 performant Gemma-family kernels (gemma_sm120.cu) */
void cuda_gemm_norm_bf16_sm120(const void*, kctx*);
void cuda_gemm_bf16_sm120(const void*, kctx*);
void cuda_gemm_norm_splitk_bf16_sm120(const void*, kctx*);
void cuda_gemm_bf16_splitk_sm120(const void*, kctx*);
void cuda_flash_causal_bf16_sm120(const void*, kctx*);
void cuda_flash_sliding_bf16_sm120(const void*, kctx*);
void cuda_flash_decode_bf16_sm120(const void*, kctx*);
void cuda_row_normrope_bf16_sm120(const void*, kctx*);
void cuda_row_normropescale_bf16_sm120(const void*, kctx*);
void cuda_row_swiglu_bf16_sm120(const void*, kctx*);
void cuda_row_residual_bf16_sm120(const void*, kctx*);
void cuda_row_rmsnorm_bf16_sm120(const void*, kctx*);
void cuda_layout_gather_scale_bf16_sm120(const void*, kctx*);

void plow_register_cuda_rtx6000(dispatch_table* dt) {
    dt_init(dt);
    /* base opcodes: DMA/RDMA/LAYOUT/CONTROL golden, GEMM/FLASH/ROW golden. */
    dt_register(dt, PLOW_NOP,           cuda_host);
    dt_register(dt, PLOW_HOST_COORD,    cuda_host);
    dt_register(dt, PLOW_TMA_LOAD,      cuda_dma_golden);
    dt_register(dt, PLOW_TMA_STORE,     cuda_dma_golden);
    dt_register(dt, PLOW_RDMA,          cuda_rdma_golden);
    dt_register(dt, PLOW_LAYOUT,        cuda_layout_golden);
    dt_register(dt, PLOW_ROW_REDUCE,    cuda_row_reduce_golden);
    dt_register(dt, PLOW_ROW_POINTWISE, cuda_row_pointwise_golden);
    dt_register(dt, PLOW_GEMM,          cuda_gemm_golden);
    dt_register(dt, PLOW_FLASH,         cuda_flash_golden);

    /* Gemma 4 fused / performant variants (index = family<<8 | variant). */
    /* F1 FusedNormLinear (q/kv/gate/up prefill) + plain GEMM (o/down/lm_head). */
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_GEMM, PLOW_VARIANT_NORM_BF16),
                    cuda_gemm_norm_bf16_sm120);
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_GEMM, PLOW_VARIANT_BF16),
                    cuda_gemm_bf16_sm120);
    /* decode split-K partial forms (reduced by a follow-on Row add). */
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_GEMM, PLOW_VARIANT_NORM_SPLITK_BF16),
                    cuda_gemm_norm_splitk_bf16_sm120);
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_GEMM, PLOW_VARIANT_BF16_SPLITK),
                    cuda_gemm_bf16_splitk_sm120);
    /* S2 FlashAttention: prefill causal / sliding, decode split-KV. */
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_FLASH, PLOW_VARIANT_FLASH_CAUSAL_BF16),
                    cuda_flash_causal_bf16_sm120);
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_FLASH, PLOW_VARIANT_FLASH_SLIDING_BF16),
                    cuda_flash_sliding_bf16_sm120);
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_FLASH, PLOW_VARIANT_FLASH_DECODE_BF16),
                    cuda_flash_decode_bf16_sm120);
    /* F2/F3/F5/S3/S4 Row kernels. */
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_ROW, PLOW_VARIANT_ROW_NORMROPE_BF16),
                    cuda_row_normrope_bf16_sm120);
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_ROW, PLOW_VARIANT_ROW_NORMROPESCALE_BF16),
                    cuda_row_normropescale_bf16_sm120);
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_ROW, PLOW_VARIANT_ROW_SWIGLU_BF16),
                    cuda_row_swiglu_bf16_sm120);
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_ROW, PLOW_VARIANT_ROW_RESIDUAL_ADD_BF16),
                    cuda_row_residual_bf16_sm120);
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_ROW, PLOW_VARIANT_ROW_RMS_BF16),
                    cuda_row_rmsnorm_bf16_sm120);
    /* F4 FusedEmbeddingScale (gather + scale). */
    dt_register(dt, PLOW_OP(0, PLOW_FAMILY_LAYOUT, PLOW_VARIANT_LAYOUT_GATHER_SCALE_BF16),
                    cuda_layout_gather_scale_bf16_sm120);
}
}

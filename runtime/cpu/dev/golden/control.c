#include "golden.h"
#include "gptoss.h"

G_K(g_nop) {
    (void)in; (void)slice; (void)nblk; (void)T; (void)ctx;
}

void plow_cpu_register_golden(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_NOP] = g_nop;
    tab[PLOW_DOP_RESIDUAL] = g_residual;
    tab[PLOW_DOP_GLU] = g_glu;
    tab[PLOW_DOP_SOFTCAP] = g_softcap;
    tab[PLOW_DOP_CAST_F32_BF16] = g_cast_f32_bf16;
    tab[PLOW_DOP_ROW_GATHER] = g_row_gather;
    tab[PLOW_DOP_ZERO_F32] = g_zero_f32;
    tab[PLOW_DOP_GEMM_SPLITK] = g_gemm_splitk;
    tab[PLOW_DOP_EMBED] = g_embed;
    tab[PLOW_DOP_ARGMAX] = g_argmax;
    tab[PLOW_DOP_ARGMAX_FIN] = g_argmax_fin;
    tab[PLOW_DOP_RMSNORM] = g_rmsnorm;
    tab[PLOW_DOP_ROWRMS] = g_rowrms;
    tab[PLOW_DOP_LAYERNORM] = g_layernorm;
    tab[PLOW_DOP_HEADNORM_ROPE] = g_headnorm_rope;
    tab[PLOW_DOP_HEADNORM_ROPE_FP8] = g_headnorm_rope;
    tab[PLOW_DOP_NORM_RESIDUAL] = g_norm_residual;
    tab[PLOW_DOP_ADD_NORM] = g_add_norm;
    tab[PLOW_DOP_NORM_RESIDUAL_NORM] = g_norm_residual_norm;
    tab[PLOW_DOP_PER_LAYER_INPUT] = g_per_layer_input;
    tab[PLOW_DOP_Q8_GEMM_F32] = g_q8_gemm_f32;
    tab[PLOW_DOP_LAYERNORM_F32] = g_layernorm_f32;
    tab[PLOW_DOP_SCALED_ADD_F32] = g_scaled_add_f32;
    tab[PLOW_DOP_GLU_F32] = g_glu_f32;
    tab[PLOW_DOP_CAUSAL_DEPTHWISE_CONV1D_F32] = g_causal_depthwise_conv1d_f32;
    tab[PLOW_DOP_RELATIVE_ATTENTION_F32] = g_relative_attention_f32;
    tab[PLOW_DOP_SILU_F32] = g_silu_f32;
    tab[PLOW_DOP_DENSE_GEMM_F32] = g_dense_gemm_f32;
    tab[PLOW_DOP_EMBED_F16_F32] = g_embed_f16_f32;
    tab[PLOW_DOP_EMBED_OVERLAY_BF16] = g_embed_overlay_bf16;
    tab[PLOW_DOP_LSTM_CELL_F32] = g_lstm_cell_f32;
    tab[PLOW_DOP_ARGMAX_F32] = g_argmax_f32;
    tab[PLOW_DOP_RELU_F32] = g_relu_f32;
    tab[PLOW_DOP_BROADCAST_ADD_F32] = g_broadcast_add_f32;
    tab[PLOW_DOP_CONV2D_F32] = g_conv2d_f32;
    tab[PLOW_DOP_PACK_NCFW_ROWS_F32] = g_pack_ncfw_rows_f32;
    tab[PLOW_DOP_GROUPED_ATTENTION_F32] = g_grouped_attention_f32;
    tab[PLOW_DOP_GEMM] = g_gemm;
    tab[PLOW_DOP_GEMM_SMALL] = g_gemm_small;
    tab[PLOW_DOP_GEMM_MED] = g_gemm_med;
    tab[PLOW_DOP_GEMM_WIDE] = g_gemm_wide;
    tab[PLOW_DOP_GEMM_C5] = g_gemm_c5;
    tab[PLOW_DOP_GEMM_NORM] = g_gemm_norm;
    tab[PLOW_DOP_GEMM_GLU] = g_gemm_glu;
    tab[PLOW_DOP_GEMV] = g_gemv;
    tab[PLOW_DOP_GEMV_GLU] = g_gemv_glu;
    tab[PLOW_DOP_GEMV_QKV] = g_gemv_qkv;
    tab[PLOW_DOP_GEMV_ARGMAX] = g_gemv_argmax;
    tab[PLOW_DOP_FLASH_PREFILL] = g_flash_prefill;
    tab[PLOW_DOP_FLASH_DECODE] = g_flash_decode;
    tab[PLOW_DOP_FLASH_PREFILL_FP8] = g_flash_prefill;
    tab[PLOW_DOP_FLASH_DECODE_FP8] = g_flash_decode;
    tab[PLOW_DOP_FLASH_MERGE] = g_flash_merge;
    tab[PLOW_DOP_ATTN_RES] = g_attn_res;
    tab[PLOW_DOP_GEMV_AFFINE_Q4] = g_gemv_affine_q4;
    tab[PLOW_DOP_GEMM_AFFINE_Q4] = g_gemm_affine_q4;
    tab[PLOW_DOP_GEMV_MXFP4] = g_gemv_mxfp4;
    tab[PLOW_DOP_GEMV_GLU_MXFP4] = g_gemv_glu_mxfp4;
    tab[PLOW_DOP_GEMM_MXFP4] = g_gemm_mxfp4;
    tab[PLOW_DOP_GEMM_MED_MXFP4] = g_gemm_med_mxfp4;
    tab[PLOW_DOP_GEMM_SMALL_MXFP4] = g_gemm_small_mxfp4;
    tab[PLOW_DOP_GEMM_WIDE_MXFP4] = g_gemm_wide_mxfp4;
    tab[PLOW_DOP_GEMM_C5_MXFP4] = g_gemm_c5_mxfp4;
    tab[PLOW_DOP_GEMM_GLU_MXFP4] = g_gemm_glu_mxfp4;
    tab[PLOW_DOP_MOE_GLU_MX] = g_moe_glu_mx;
    tab[PLOW_DOP_MOE_DOWN_MX] = g_moe_down_mx;
    tab[PLOW_DOP_MOE_GLU_MX_PF] = g_moe_glu_mx_pf;
    tab[PLOW_DOP_MOE_DOWN_MX_PF] = g_moe_down_mx_pf;
    tab[PLOW_DOP_MOE_ROUTER_TOPK_PF] = g_moe_router_topk_pf;
    tab[PLOW_DOP_MOE_ALIGN_PF] = g_moe_align_pf;
    tab[PLOW_DOP_MOE_COMBINE_PF] = g_moe_combine_pf;
    tab[PLOW_DOP_MOE_ROUTER_GEMMA_SCORE_FAST] = g_moe_router_gemma_score_fast;
    tab[PLOW_DOP_MOE_ROUTER_GEMMA_SCORE] = g_moe_router_gemma_score_fast; /* exact = same serial f32 math here */
    tab[PLOW_DOP_MOE_ROUTER_GEMMA_TOPK] = g_moe_router_gemma_topk;
    tab[PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA] = g_moe_expert_glu_norm_gemma;
    tab[PLOW_DOP_MOE_EXPERT_DOWN_GEMMA] = g_moe_expert_down_gemma;
    tab[PLOW_DOP_MOE_COMBINE_NORM_GEMMA] = g_moe_combine_norm_gemma;
    tab[PLOW_DOP_MOE_ROUTER_GEMMA_PF] = g_moe_router_gemma_pf;
    tab[PLOW_DOP_MOE_ALIGN_GEMMA_PF] = g_moe_align_pf;
    tab[PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF] = g_moe_group_glu_gemma_pf;
    tab[PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF] = g_moe_group_down_gemma_pf;
    tab[PLOW_DOP_MOE_COMBINE_NORM_GEMMA_PF] = g_moe_combine_norm_gemma_pf;
}

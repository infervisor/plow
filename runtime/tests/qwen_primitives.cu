#include <cuda_runtime.h>
#include "../common/dev_isa.h"
#include "../nvidia/op_qwen_gdn.cuh"
#include "../nvidia/op_gemm.cuh"
#ifndef PLOW_NV_FP8_M1_BK256
#define PLOW_NV_FP8_M1_BK256 0
#endif
#ifndef PLOW_NV_FP8_M1_BK512
#define PLOW_NV_FP8_M1_BK512 0
#endif
#ifndef PLOW_NV_FP8_M1_BK1024
#define PLOW_NV_FP8_M1_BK1024 0
#endif
struct Args { void* t[8]; int i[8]; union {float f; int j;} fj[2]; };
#define TEN(n) in->t[n]
__global__ void run(unsigned op, Args args) {
extern __shared__ __nv_bfloat16 arena[];
const Args* in=&args; unsigned slice=blockIdx.x,nblk=gridDim.x;
switch(op) {
#if defined(PLOW_NV_FP8_M1) && PLOW_NV_FP8_M1
    case PLOW_DOP_GEMM_FP8:
        d_gemm_w8a8((__nv_bfloat16*)TEN(0), (const uint8_t*)TEN(1), (const uint8_t*)TEN(2),
            (const float*)TEN(3), (const float*)TEN(4), in->i[0], in->i[1], in->i[2],
            0, slice, nblk, arena);
        break;
#endif
    case PLOW_DOP_QUANT_FP8:
        d_quant_fp8((uint8_t*)TEN(0), (__nv_bfloat16*)TEN(1), (float*)TEN(2),
            in->i[0], in->i[1], slice, nblk);
        break;
    case PLOW_DOP_QWEN_GDN_CONV:
        d_qwen_gdn_conv((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
            (const __nv_bfloat16*)TEN(2), (__nv_bfloat16*)TEN(3), (const int*)TEN(4),
            in->i[0], in->i[1], in->i[2], slice, nblk);
        break;
    case PLOW_DOP_QWEN_GDN_STEP:
        d_qwen_gdn_step((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
            (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3), TEN(4),
            (const __nv_bfloat16*)TEN(5), (float*)TEN(6), (const int*)TEN(7),
            in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f,
            in->fj[1].f, in->i[5], slice, nblk);
        break;
    case PLOW_DOP_QWEN_GATED_NORM:
        d_qwen_gated_norm((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
            (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3), (const int*)TEN(4),
            in->i[0], in->i[1], in->i[2], in->fj[0].f, slice, nblk);
        break;
    case PLOW_DOP_QWEN_Q_GATE_SPLIT:
        d_qwen_q_gate_split((__nv_bfloat16*)TEN(0), (__nv_bfloat16*)TEN(1),
            (const __nv_bfloat16*)TEN(2), (const int*)TEN(3),
            in->i[0], in->i[1], in->i[2], slice, nblk);
        break;
    case PLOW_DOP_QWEN_SIGMOID_GATE:
        d_qwen_sigmoid_gate((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
            (const __nv_bfloat16*)TEN(2), (const int*)TEN(3), in->i[0], in->i[1], slice, nblk);
        break;
    case PLOW_DOP_QWEN_RMSNORM:
        d_qwen_rmsnorm((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
            (const __nv_bfloat16*)TEN(2), (const int*)TEN(3), in->i[0], in->i[1],
            in->fj[0].f, in->fj[1].f, slice, nblk);
        break;
    case PLOW_DOP_QWEN_HEADNORM_ROPE:
        d_qwen_headnorm_rope((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
            (const __nv_bfloat16*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
            (const int*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1], in->i[2],
            in->i[3], in->i[4], in->i[5], in->fj[0].f, in->fj[1].f, slice, nblk, in->i[6]);
        break;
    case PLOW_DOP_QWEN_GDN_CONV_PREFILL:
        d_qwen_gdn_conv_prefill((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
            (const __nv_bfloat16*)TEN(2), (__nv_bfloat16*)TEN(3), in->i[0], in->i[1],
            in->i[2], slice, nblk);
        break;
    case PLOW_DOP_QWEN_GDN_QKV_PREP:
        d_qwen_gdn_qkv_prep((__nv_bfloat16*)TEN(0), (__nv_bfloat16*)TEN(1),
            (__nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3), in->i[0], in->i[1],
            in->i[2], in->i[3], in->i[4], in->fj[0].f, slice, nblk);
        break;
    case PLOW_DOP_QWEN_GDN_GATE_PREP:
        d_qwen_gdn_gate_prep((float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
            (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4),
            (const __nv_bfloat16*)TEN(5), in->i[0], in->i[1], slice, nblk);
        break;

}}
extern "C" int qwen_test(unsigned op, void** tensors, const int* integers, const float* floats, void* stream) {
#if PLOW_NV_FP8_M1_BK1024
static const cudaError_t configured = cudaFuncSetAttribute(run, cudaFuncAttributeMaxDynamicSharedMemorySize, 74752);
if (configured != cudaSuccess) return (int)configured;
#endif
Args a={}; for(int i=0;i<8;i++)a.t[i]=tensors[i]; for(int i=0;i<8;i++)a.i[i]=integers[i]; for(int i=0;i<2;i++)a.fj[i].f=floats[i];
run<<<132,256,op == PLOW_DOP_GEMM_FP8 ? (PLOW_NV_FP8_M1_BK1024 ? 74752 : PLOW_NV_FP8_M1_BK512 ? 37888 : PLOW_NV_FP8_M1_BK256 ? 19456 : 12352) : 0,(cudaStream_t)stream>>>(op,a);return (int)cudaGetLastError();
}

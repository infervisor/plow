#define PLOW_WG_WAVES 8
#define PLOW_MOE_PF_EPI 1
#define PLOW_MOE_PF_DET 1
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_norm.h"
#include "op_elementwise.h"
#include "op_gemm.h"
#include "op_moe.h"
#endif
__global__ __launch_bounds__(512, 1) void align_moe(void *meta, void *table,
                                                    void *rt, void *rp,
                                                    void *rg, unsigned M,
                                                    unsigned phase) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned lds[2048];
    d_moe_align_pf((int *)meta, (unsigned char *)table, (unsigned *)rt,
                   (unsigned *)rp, (float *)rg, M, 256, 8, blockIdx.x, lds,
                   phase, gridDim.x, 64);
#endif
}
__global__ __launch_bounds__(512, 1) void glu_moe(void *fu, void *x, void *wt,
                                                  void *st, void *meta,
                                                  void *rt) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ __align__(16) bf16 lds[32768];
    d_moe_group_glu_pf((bf16 *)fu, (bf16 *)x, (unsigned long long *)wt,
                       (unsigned long long *)st, (int *)meta, (unsigned *)rt,
                       256, 6144, 256, 1, 1, blockIdx.x, gridDim.x, 0, 0, lds);
#endif
}
__global__ __launch_bounds__(512, 1) void down_moe(void *part, void *fu,
                                                   void *wt, void *st,
                                                   void *meta, void *rp,
                                                   void *rg) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ __align__(16) bf16 lds[32768];
    d_moe_group_down_pf((float *)part, (bf16 *)fu, (unsigned long long *)wt,
                        (unsigned long long *)st, (int *)meta, (unsigned *)rp,
                        (float *)rg, 6144, 256, 256, 1, blockIdx.x, gridDim.x,
                        lds, nullptr, 0, 0, 4);
#endif
}
__global__ void combine_moe(void *out, void *part, unsigned M) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_moe_combine_pf((bf16 *)out, nullptr, nullptr, (float *)part, 6144, 8, M,
                     blockIdx.x, gridDim.x, 0, 1);
#endif
}
extern "C" int plow_moe(void *out, void *x, void *wt, void *st, void *table,
                        void *meta, void *rt, void *rp, void *rg, void *fu,
                        void *part, unsigned M, void *s) {
    hipStream_t stream = (hipStream_t)s;
    hipError_t rc = hipMemsetAsync(part, 0, (size_t)M * 6144 * 8, stream);
    if (rc != hipSuccess)
        return rc;
    align_moe<<<64, 512, 0, stream>>>(meta, table, rt, rp, rg, M, 1);
    align_moe<<<1, 512, 0, stream>>>(meta, table, rt, rp, rg, M, 2);
    align_moe<<<64, 512, 0, stream>>>(meta, table, rt, rp, rg, M, 3);
    align_moe<<<64, 512, 0, stream>>>(meta, table, rt, rp, rg, M, 4);
    glu_moe<<<304, 512, 0, stream>>>(fu, x, wt, st, meta, rt);
    down_moe<<<304, 512, 0, stream>>>(part, fu, wt, st, meta, rp, rg);
    combine_moe<<<304, 512, 0, stream>>>(out, part, M);
    return hipGetLastError();
}

#include "moe_aiter_adapter.hip"

extern "C" int plow_moe_pack(void* gu, void* down, void* gs, void* ds,
                             void* wt, void* st, void* stream) {
    pack_moe<<<256 * 8, 256, 0, (hipStream_t)stream>>>((uint4*)gu, (uint4*)down,
        (float*)gs, (float*)ds, (const unsigned long long*)wt,
        (const unsigned long long*)st);
    return hipGetLastError();
}

__global__ void reduce_moe_slots(double* acc, unsigned short* out,
                                 const unsigned short* parts, unsigned rows) {
#if defined(__HIP_DEVICE_COMPILE__)
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * 6144u;
         i += blockDim.x * gridDim.x) {
        const unsigned row = i / 6144u, col = i % 6144u;
        double sum = 0;
#pragma unroll
        for (unsigned slot = 0; slot < 8; ++slot)
            sum += (double)bf2f(parts[(row * 8u + slot) * 6144u + col]);
        acc[i] = sum;
        out[i] = f2bf((float)sum);
    }
#endif
}

extern "C" int plow_moe_reduce_slots(void* acc, void* out, void* parts,
                                     unsigned rows, void* stream) {
    reduce_moe_slots<<<304, 256, 0, (hipStream_t)stream>>>((double*)acc,
        (unsigned short*)out, (const unsigned short*)parts, rows);
    return hipGetLastError();
}

static hipModule_t moe_module;
static hipFunction_t moe_function;

extern "C" int plow_moe_native_load(const char* path) {
    hipError_t status = hipModuleLoad(&moe_module, path);
    if (status != hipSuccess) return status;
    return hipModuleGetFunction(&moe_function, moe_module,
        "_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256E");
}

extern "C" int plow_moe_native_prepare(MoePrepareArgs* args, void* stream) {
    plow_moe_aiter_prepare<<<304 * 4, 256, 0, (hipStream_t)stream>>>(*args);
    return hipGetLastError();
}

extern "C" int plow_moe_native_align(void* meta, void* table, void* rt, void* rp,
                                      void* rg, unsigned rows, void* stream) {
    align_moe<<<64, 512, 0, (hipStream_t)stream>>>(meta, table, rt, rp, rg, rows, 1);
    align_moe<<<1, 512, 0, (hipStream_t)stream>>>(meta, table, rt, rp, rg, rows, 2);
    align_moe<<<64, 512, 0, (hipStream_t)stream>>>(meta, table, rt, rp, rg, rows, 3);
    align_moe<<<64, 512, 0, (hipStream_t)stream>>>(meta, table, rt, rp, rg, rows, 4);
    return hipGetLastError();
}

extern "C" int plow_moe_native_execute(void* out, void* q, void* gu, void* down,
                                         void* qs, void* gs, void* ds, void* ids,
                                         void* weights, void* experts, void* valid,
                                         unsigned rows, void* stream) {
    uint64_t args[56] = {};
    args[0] = (uint64_t)out;
    args[2] = (uint64_t)q;
    args[4] = (uint64_t)gu;
    args[6] = (uint64_t)valid;
    args[8] = (uint64_t)down;
    args[10] = (uint64_t)qs;
    args[12] = (uint64_t)gs;
    args[14] = (uint64_t)ds;
    args[18] = (uint64_t)ids;
    args[20] = (uint64_t)weights;
    args[22] = (uint64_t)experts;
    args[24] = 6144;
    args[26] = 256;
    args[28] = rows;
    args[30] = 256;
    args[32] = 6144;
    args[34] = 6144;
    args[36] = 256;
    args[38] = 6144 * 2;
    args[40] = 512 * 6144;
    args[42] = 256 * 6144;
    args[44] = 192 * 4;
    args[46] = 96 * 4;
    args[48] = 256 * 4;
    args[50] = 8;
    args[52] = 304;
    args[54] = 1;
    size_t size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
    return hipModuleLaunchKernel(moe_function, 304, 1, 1, 256, 1, 1, 0,
                                  (hipStream_t)stream, nullptr, config);
}

extern "C" int plow_moe_native_store(void* out, void* src, unsigned rows, void* stream) {
    plow_moe_aiter_store<<<304, 256, 0, (hipStream_t)stream>>>((float*)out,
        (const unsigned short*)src, rows);
    return hipGetLastError();
}

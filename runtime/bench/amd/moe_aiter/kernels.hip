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

template <unsigned N, unsigned K, bool GU>
__device__ void pack_weight(uint4* out, const unsigned long long* wt) {
    constexpr unsigned vectors = N * K / 16;
    const unsigned e = blockIdx.x / 8u;
    const auto* first = (const uint4*)wt[e * 3u + (GU ? 0u : 2u)];
    const auto* second = GU ? (const uint4*)wt[e * 3u + 1u] : first;
    for (unsigned v = blockIdx.x % 8u * blockDim.x + threadIdx.x; v < vectors;
         v += blockDim.x * 8u) {
        const unsigned row = (v / (32u * (K / 32u))) * 16u + v % 16u;
        const unsigned col = (v / 32u) % (K / 32u) * 2u + (v / 16u) % 2u;
        const unsigned source_row = GU ? row % 256u : row;
        const auto* src = GU && row >= 256u ? second : first;
        out[e * vectors + v] = src[source_row * (K / 16u) + col];
    }
}

__global__ void pack_moe(uint4* gu, uint4* down, float* gs, float* ds,
                         const unsigned long long* wt, const unsigned long long* st) {
    // Runtime weights already scrub OCP -0. Keep their bytes and double FNUZ scales.
    pack_weight<512, 6144, true>(gu, wt);
    pack_weight<6144, 256, false>(down, wt);
    if (blockIdx.x % 8u == 0 && threadIdx.x < 96u) {
        const unsigned e = blockIdx.x / 8u, j = threadIdx.x;
        gs[e * 192u + j] = 2.0f * ((const float*)st[e * 3u])[j];
        gs[e * 192u + 96u + j] = 2.0f * ((const float*)st[e * 3u + 1u])[j];
        ds[e * 96u + j] = 2.0f * ((const float*)st[e * 3u + 2u])[j];
    }
}

extern "C" int plow_moe_pack(void* gu, void* down, void* gs, void* ds,
                             void* wt, void* st, void* stream) {
    pack_moe<<<256 * 8, 256, 0, (hipStream_t)stream>>>((uint4*)gu, (uint4*)down,
        (float*)gs, (float*)ds, (const unsigned long long*)wt,
        (const unsigned long long*)st);
    return hipGetLastError();
}

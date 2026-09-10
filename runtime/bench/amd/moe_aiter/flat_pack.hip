#define PLOW_MOE_BENCH_DECODE 1
#include "kernels.hip"

__global__ void pack_flat_active(uint4* gu, uint4* down, float* gs, float* ds,
    const unsigned long long* wt, const unsigned long long* st,
    const uint2* table, unsigned* ids, float* weights, unsigned rows) {
    __shared__ unsigned active;
    if (threadIdx.x == 0) active = 0;
    __syncthreads();
    const unsigned e = blockIdx.x / 8u;
    if (threadIdx.x < rows * 8u) {
        const uint2 route = table[threadIdx.x];
        if (blockIdx.x == 0) {
            ids[threadIdx.x] = route.x;
            weights[threadIdx.x] = __uint_as_float(route.y);
        }
        if (route.x == e) atomicExch(&active, 1u);
    }
    __syncthreads();
    if (!active) return;
    pack_weight<512, 6144, true>(gu, wt);
    pack_weight<6144, 256, false>(down, wt);
    if (blockIdx.x % 8u == 0 && threadIdx.x < 96u) {
        const unsigned j = threadIdx.x;
        gs[e * 192u + j] = 2.0f * ((const float*)st[e * 3u])[j];
        gs[e * 192u + 96u + j] = 2.0f * ((const float*)st[e * 3u + 1u])[j];
        ds[e * 96u + j] = 2.0f * ((const float*)st[e * 3u + 2u])[j];
    }
}

extern "C" int plow_moe_flat_pack(void* gu, void* down, void* gs, void* ds,
    void* wt, void* st, void* table, void* ids, void* weights,
    unsigned rows, void* stream) {
    if (!rows || rows > 32) return hipErrorInvalidValue;
    pack_flat_active<<<256 * 8, 256, 0, (hipStream_t)stream>>>(
        (uint4*)gu, (uint4*)down, (float*)gs, (float*)ds,
        (const unsigned long long*)wt, (const unsigned long long*)st,
        (const uint2*)table, (unsigned*)ids, (float*)weights, rows);
    return hipGetLastError();
}

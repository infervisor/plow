#define PLOW_MOE_BENCH_DECODE 1
#include "kernels.hip"

extern "C" int plow_moe_flat_pack(void* gu, void* down, void* gs, void* ds,
    void* wt, void* st, void* table, void* ids, void* weights,
    unsigned rows, void* stream) {
    if (!rows || rows > 32) return hipErrorInvalidValue;
    MoeFlatPackArgs args{(uint4*)gu, (uint4*)down, (float*)gs, (float*)ds,
        (const unsigned long long*)wt, (const unsigned long long*)st,
        (const uint2*)table, (unsigned*)ids, (float*)weights, rows, 0};
    plow_moe_aiter_flat_pack<<<256 * 8, 256, 0, (hipStream_t)stream>>>(args);
    return hipGetLastError();
}

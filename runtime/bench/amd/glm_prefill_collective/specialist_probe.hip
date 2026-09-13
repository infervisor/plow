#include "op_collective.h"

struct Args {
    bf16* out;
    const void* const* peers;
    uint32_t* status;
    uint64_t xoff;
    uint32_t n, rank;
};
static_assert(sizeof(Args) == 40);

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES_PER_EU)
void collective(Args a) {
    d_xreduce_twoshot_mega(a.out, a.peers, 8, a.rank, a.n, 256, a.xoff,
                          0, 1, 1000000000ull, a.status, blockIdx.x, gridDim.x);
}

extern "C" __global__ void initialize(bf16* data, bf16* out, uint32_t n,
                                      uint32_t rank, uint32_t reuse) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        data[i] = f2bf(float(rank + 1) + float(int(i % 127) - 63) / 128.0f + float(reuse % 3) / 8.0f);
        out[i] = 0x7fa1;
    }
}

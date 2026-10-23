// KV block store gather/scatter: one launch copies every segment of a table (`plow_kv_seg`),
// one thread block per segment, 16-byte vectors. Segments are 16-byte aligned and sized (the
// store's block layout guarantees it). Shipped as PTX (`kv_segcopy.ptx`, embedded in plowrt and
// JIT-compiled by the driver at load); regenerate with
//   nvcc -arch=sm_90 -O3 -ptx -o runtime/nvidia/kv_segcopy.ptx runtime/nvidia/kv_segcopy.cu
#include <cstdint>
struct plow_kv_seg {
    uint64_t src;
    uint64_t dst;
    uint32_t bytes;
    uint32_t pad;
};
extern "C" __global__ void plow_kv_segcopy(const plow_kv_seg* __restrict__ table, unsigned n) {
    unsigned s = blockIdx.x;
    if (s >= n) return;
    const uint4* __restrict__ src = reinterpret_cast<const uint4*>(table[s].src);
    uint4* __restrict__ dst = reinterpret_cast<uint4*>(table[s].dst);
    unsigned v = table[s].bytes >> 4;
    for (unsigned i = threadIdx.x; i < v; i += blockDim.x) dst[i] = src[i];
}

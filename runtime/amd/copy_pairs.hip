#define PLOW_WG_WAVES 4
#include "amd_common.h"

extern "C" __device__ unsigned plow_copy_pairs_abi_1 = 1;

/* PLOW_AMD_SNAP_KERNEL: a batch of device-to-device copies as ONE queue-ordered dispatch (prefix
 * snapshots: ~300 tensors per slot). Workgroup `b` copies piece `b`; the host splits copies into
 * pieces of at most 1 MiB. 16-byte aligned pieces move as uint4, the rest bytewise. */
struct PlowCopyPiece {
    uint64_t dst;
    uint64_t src;
    uint64_t bytes;
};

extern "C" __global__ __launch_bounds__(256, 4)
void plow_copy_pairs_gfx950(const PlowCopyPiece* __restrict__ pieces, unsigned n) {
    if (blockIdx.x >= n) return;
    const PlowCopyPiece p = pieces[blockIdx.x];
    if (((p.dst | p.src | p.bytes) & 15ull) == 0ull) {
        uint4* d = reinterpret_cast<uint4*>(p.dst);
        const uint4* s = reinterpret_cast<const uint4*>(p.src);
        const uint64_t n16 = p.bytes >> 4;
        for (uint64_t i = threadIdx.x; i < n16; i += blockDim.x) d[i] = s[i];
    } else {
        unsigned char* d = reinterpret_cast<unsigned char*>(p.dst);
        const unsigned char* s = reinterpret_cast<const unsigned char*>(p.src);
        for (uint64_t i = threadIdx.x; i < p.bytes; i += blockDim.x) d[i] = s[i];
    }
}

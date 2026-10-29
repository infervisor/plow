// KV block store gather/scatter by block descriptor (plus `plow_small_copy`, below). The
// descriptors ride in the launch parameters (no table upload); each CTA copies one (block, layout
// entry) run in 16-byte vectors. A layout entry is one (tensor, head) run of a store class: slot
// `s`, slot row `r` at `dev + s * slot_stride + r * row_bytes`, its copy in the block at
// `block + block_off`, `bytes` long (16-byte aligned and sized). Shipped as PTX (`kv_segcopy.ptx`,
// embedded in plowrt and JIT-compiled by the driver at load); regenerate with
//   nvcc -arch=sm_90 -O3 -ptx -o runtime/nvidia/kv_segcopy.ptx runtime/nvidia/kv_segcopy.cu
#include <cstdint>
struct plow_kv_layout {
    uint64_t dev;
    uint64_t slot_stride;
    uint32_t row_bytes;
    uint32_t block_off;
    uint32_t bytes;
    uint32_t pad;
};
struct plow_kv_desc {
    uint64_t block;
    uint32_t row;
    uint32_t slot;
};
// 32,528 parameter bytes: under the 32,764-byte kernel parameter limit (CUDA 12.1+, sm_70+).
#define PLOW_KV_DESCS 2032
struct plow_kv_blockcopy_args {
    const plow_kv_layout* layout;
    uint32_t to_block;
    uint32_t n;
    plow_kv_desc d[PLOW_KV_DESCS];
};
extern "C" __global__ void plow_kv_blockcopy(const __grid_constant__ plow_kv_blockcopy_args a) {
    const unsigned i = blockIdx.y;
    if (i >= a.n) return;
    const plow_kv_layout l = a.layout[blockIdx.x];
    const plow_kv_desc d = a.d[i];
    uint4* blk = reinterpret_cast<uint4*>(d.block + l.block_off);
    uint4* dev = reinterpret_cast<uint4*>(l.dev + (uint64_t)d.slot * l.slot_stride +
                                          (uint64_t)d.row * l.row_bytes);
    const uint4* __restrict__ src = a.to_block ? dev : blk;
    uint4* __restrict__ dst = a.to_block ? blk : dev;
    const unsigned v = l.bytes >> 4;
    for (unsigned j = threadIdx.x; j < v; j += blockDim.x) dst[j] = src[j];
}
// Small engine copies between device and pinned host memory (UVA-mapped) on the SMs: a cuMemcpy
// waits behind every bulk copy queued on the same copy engine (the host tier's spills).
extern "C" __global__ void plow_small_copy(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
                                           unsigned words) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}

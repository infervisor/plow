#include <hip/hip_runtime.h>
#include <stdint.h>

#define GETREG_IMM(id, ofs, size) ((id) | ((ofs) << 6) | (((size) - 1) << 11))

extern "C" __global__ __launch_bounds__(512, 1) void xcd_hsa_probe(uint32_t* out) {
    if (threadIdx.x == 0)
        out[blockIdx.x] =
            __builtin_amdgcn_s_getreg(GETREG_IMM(20 /* HW_REG_XCC_ID */, 0, 32)) & 0xf;
}

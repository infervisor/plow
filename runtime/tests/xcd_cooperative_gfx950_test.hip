#include <hip/hip_runtime.h>
#include <stdio.h>

#define GETREG_IMM(id, ofs, size) ((id) | ((ofs) << 6) | (((size) - 1) << 11))

__global__ __launch_bounds__(512, 1) void xcd_cooperative_probe(unsigned* out) {
    if (threadIdx.x == 0)
        out[blockIdx.x] =
            __builtin_amdgcn_s_getreg(GETREG_IMM(20 /* HW_REG_XCC_ID */, 0, 32));
}

int main() {
    unsigned *device = nullptr, *host = nullptr;
    if (hipMalloc(&device, 256 * sizeof(unsigned)) != hipSuccess ||
        hipHostMalloc(&host, 256 * sizeof(unsigned)) != hipSuccess)
        return 1;
    void* args[] = {&device};
    if (hipLaunchCooperativeKernel((const void*)xcd_cooperative_probe, dim3(256), dim3(512),
                                   args, 0, nullptr) != hipSuccess ||
        hipMemcpy(host, device, 256 * sizeof(unsigned), hipMemcpyDeviceToHost) != hipSuccess)
        return 1;
    unsigned counts[32] = {};
    unsigned high = 0;
    for (unsigned i = 0; i < 256; ++i) {
        high |= host[i] & ~0xfu;
        counts[host[i] & 0x1fu]++;
    }
    printf("high=%#x", high);
    for (unsigned i = 0; i < 16; ++i)
        if (counts[i]) printf(" %u=%u", i, counts[i]);
    printf("\n");
    hipHostFree(host);
    hipFree(device);
    return high || counts[0] != 32 || counts[1] != 32 || counts[2] != 32 || counts[3] != 32 ||
           counts[4] != 32 || counts[5] != 32 || counts[6] != 32 || counts[7] != 32;
}

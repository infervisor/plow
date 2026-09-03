/* Verify that public HIP/HSA CU masks cannot isolate one gfx950 XCD.
 *
 * Build outside nix develop: the system ROCm compiler requires the system glibc.
 *   /opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 -w \
 *     runtime/tests/xcd_cu_mask_gfx950_test.hip -o /tmp/xcd_cu_mask
 * Run under gpulease and the render group.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

#define GETREG_IMM(id, ofs, size) ((id) | ((ofs) << 6) | (((size) - 1) << 11))
#define HWREG_XCC_ID 20

#define CK(x)                                                                                \
    do {                                                                                     \
        hipError_t e_ = (x);                                                                 \
        if (e_ != hipSuccess) {                                                              \
            fprintf(stderr, "FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x,                 \
                    hipGetErrorString(e_));                                                   \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

__global__ __launch_bounds__(512, 1) void probe(unsigned* xcc) {
    if (threadIdx.x == 0)
        xcc[blockIdx.x] =
            __builtin_amdgcn_s_getreg(GETREG_IMM(HWREG_XCC_ID, 0, 32)) & 0xf;
}

static int run_mask(const char* name, const unsigned mask[8], int expected_xcd,
                    unsigned* device, unsigned* host) {
    hipStream_t stream;
    CK(hipExtStreamCreateWithCUMask(&stream, 8, mask));
    unsigned readback[8] = {};
    CK(hipExtStreamGetCUMask(stream, 8, readback));
    const bool mask_matches = memcmp(mask, readback, sizeof(readback)) == 0;
    printf("%s requested:", name);
    for (unsigned i = 0; i < 8; ++i) printf(" %08x", mask[i]);
    printf("\n%s readback :", name);
    for (unsigned i = 0; i < 8; ++i) printf(" %08x", readback[i]);
    printf(" (%s)\n", mask_matches ? "exact" : "DIFFERS");
    CK(hipMemsetAsync(device, 0xff, 32 * sizeof(unsigned), stream));
    hipLaunchKernelGGL(probe, dim3(32), dim3(512), 0, stream, device);
    CK(hipMemcpyAsync(host, device, 32 * sizeof(unsigned), hipMemcpyDeviceToHost, stream));
    CK(hipStreamSynchronize(stream));
    CK(hipStreamDestroy(stream));

    unsigned counts[16] = {};
    for (unsigned i = 0; i < 32; ++i)
        if (host[i] < 16) counts[host[i]]++;
    printf("%s:", name);
    for (unsigned i = 0; i < 8; ++i) printf(" %u=%u", i, counts[i]);
    printf("\n");
    if (!mask_matches) return 1;
    if (expected_xcd >= 0) return counts[expected_xcd] == 32 ? 0 : 1;
    for (unsigned i = 0; i < 8; ++i) if (counts[i] != 4) return 1;
    return 0;
}

int main() {
    hipDeviceProp_t props;
    CK(hipGetDeviceProperties(&props, 0));
    if (props.multiProcessorCount != 256 || strncmp(props.gcnArchName, "gfx950", 6) != 0) {
        fprintf(stderr, "SKIP: requires gfx950 with 256 CUs, found %s with %d\n",
                props.gcnArchName, props.multiProcessorCount);
        return 77;
    }

    unsigned *device, *host;
    CK(hipMalloc(&device, 32 * sizeof(unsigned)));
    CK(hipHostMalloc(&host, 32 * sizeof(unsigned)));
    int failed = 0;
    for (unsigned word = 0; word < 8; ++word) {
        unsigned contiguous[8] = {};
        contiguous[word] = 0xffffffffu;
        char name[32];
        snprintf(name, sizeof name, "mask-word[%u]", word);
        failed |= run_mask(name, contiguous, -1, device, host);
    }
    for (unsigned xcd = 0; xcd < 8; ++xcd) {
        unsigned one_cu[8] = {};
        one_cu[xcd / 32] = 1u << (xcd % 32);
        char name[32];
        snprintf(name, sizeof name, "one-cu[%u]", xcd);
        failed |= run_mask(name, one_cu, -1, device, host);

        unsigned cu_pair[8] = {};
        for (unsigned local_cu = 0; local_cu < 2; ++local_cu) {
            const unsigned bit = xcd + 8 * local_cu;
            cu_pair[bit / 32] |= 1u << (bit % 32);
        }
        snprintf(name, sizeof name, "cu-pair[%u]", xcd);
        failed |= run_mask(name, cu_pair, -1, device, host);

        unsigned strided[8] = {};
        for (unsigned local_cu = 0; local_cu < 32; ++local_cu) {
            const unsigned bit = xcd + 8 * local_cu;
            strided[bit / 32] |= 1u << (bit % 32);
        }
        snprintf(name, sizeof name, "strided[%u]", xcd);
        failed |= run_mask(name, strided, -1, device, host);
    }
    CK(hipHostFree(host));
    CK(hipFree(device));
    if (failed) {
        fprintf(stderr, "FAIL: one or more CU masks changed the expected symmetric XCD map\n");
        return 1;
    }
    printf("PASS: CU masks remain symmetric across gfx950 XCDs; none provides XCD affinity\n");
    return 0;
}

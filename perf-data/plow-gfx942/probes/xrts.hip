/* Full SHIPPED two-shot at varying packet WIDTH (nblk) -- 8x MI300X, gfx942.
 *
 * Calls d_xreduce_twoshot_mega itself, so this is the packet and not a model of it:
 * gate_rs + RS reduce + gate_ag + staggered AG, 512 threads/WG (PLOW_THREADS), at the @8k
 * GLM-5.2 shape (n = 8192*6144 bf16 = 100.7 MB partial per rank). All 8 devices run
 * concurrently, so the fabric contention is the real thing.
 *
 * WHY: a packet's workgroup count IS its emitted `cus` list length, and the GLM PREFILL
 * emitter pinned every collective to all 304. This probe is what showed 304 is ~48% past
 * the fabric's concurrency peak. Build:
 *   hipcc --offload-arch=gfx942 -O3 -I<repo>/runtime -o xrts xrts.hip
 * (the device-only include guard is because the header's address_space(1) helpers do not
 * compile in the host pass).
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
#define PLOW_TH 512
#ifdef __HIP_DEVICE_COMPILE__
#include "../../../runtime/common/dev_isa.h"
#include "../../../runtime/amd/op_collective.h"
#endif

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)

static const int NR = 8;
static const size_t N = (size_t)8192 * 6144;   /* @8k: T*hidden elements */
static const size_t SLOT = 0;                  /* partial at region base */
static const size_t XCTR_OFF = 256u * 1024u * 1024u; /* counters well past the partial */

__global__ __launch_bounds__(PLOW_TH) void k_ts(void* out, const void* const* peers,
                                                uint32_t rank, uint32_t n, uint32_t g0, int reps) {
#ifdef __HIP_DEVICE_COMPILE__
    for (int r = 0; r < reps; r++)
        d_xreduce_twoshot_mega((bf16*)out, peers, NR, rank, n, (uint32_t)SLOT, XCTR_OFF,
                               g0 + 2u * (uint32_t)r, g0 + 2u * (uint32_t)r + 1u,
                               (uint64_t)1 << 40, nullptr, blockIdx.x, gridDim.x);
#else
    (void)out; (void)peers; (void)rank; (void)n; (void)g0; (void)reps;
#endif
}

int main() {
    void* regions[NR];
    void* outs[NR];
    void** dpeers[NR];
    const size_t REGION = (size_t)512 * 1024 * 1024;
    const int REPS = 20;

    for (int d = 0; d < NR; d++) {
        CHECK(hipSetDevice(d));
        for (int p = 0; p < NR; p++) if (p != d) hipDeviceEnablePeerAccess(p, 0);
        CHECK(hipMalloc(&regions[d], REGION));
        CHECK(hipMemset(regions[d], 0x3c, N * 2));
        CHECK(hipMalloc(&outs[d], N * 2));
    }
    for (int d = 0; d < NR; d++) {
        CHECK(hipSetDevice(d));
        CHECK(hipMalloc(&dpeers[d], NR * sizeof(void*)));
        CHECK(hipMemcpy(dpeers[d], regions, NR * sizeof(void*), hipMemcpyHostToDevice));
    }
    hipStream_t st[NR];
    for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamCreate(&st[d])); }

    printf("=== full two-shot packet (RS + 2 rendezvous + AG), n=%zu (%.1f MB partial)\n",
           N, (double)N * 2 / 1e6);
    const int wgs[] = {304, 256, 224, 192, 176, 152, 128, 112, 96, 76, 64, 38};
    for (unsigned w = 0; w < sizeof(wgs) / sizeof(wgs[0]); w++) {
        const int nwg = wgs[w];
        for (int d = 0; d < NR; d++) {
            CHECK(hipSetDevice(d));
            CHECK(hipMemset((char*)regions[d] + XCTR_OFF, 0,
                            (size_t)(2 * (REPS + 2) + 4) * 32u * 4));
            CHECK(hipMemset(regions[d], 0x3c, N * 2));
        }
        for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipDeviceSynchronize()); }
        auto t0 = std::chrono::steady_clock::now();
        for (int d = 0; d < NR; d++) {
            CHECK(hipSetDevice(d));
            hipLaunchKernelGGL(k_ts, dim3(nwg), dim3(PLOW_TH), 0, st[d], outs[d],
                               (const void* const*)dpeers[d], (uint32_t)d, (uint32_t)N, 0u, REPS);
        }
        for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(st[d])); }
        double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count() / REPS;
        /* fabric bytes/rank: RS reads (N-1)/N of each of 8 partials over the owned slice
         * (7/8 remote of n/8*8 = n) + AG reads 7/8 of n. */
        double gbs = (double)N * 2 * 2 / us / 1e3;
        printf("  nblk=%3d  %8.1f us/collective   %6.1f GB/s aggregate (RS+AG reads)\n",
               nwg, us, gbs);
    }
    return 0;
}

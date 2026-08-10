/* Peer-read bandwidth ceiling probe for the two-shot all-reduce on 8x MI300X.
 *
 * Emulates the AG phase's traffic: each of the 8 devices reads the 7 peers'
 * 12.6 MB slices (and its own local one) into a local buffer. Variants:
 *   width:  scalar ushort (2 B/lane, what the kernel ships) vs uint4 (16 B/lane)
 *   order:  seq  — all workgroups walk slices s=0..7 together (ships today)
 *           stag — workgroup w starts at slice (w + rank) % 8, so all links stream
 * All 8 devices run concurrently (real fabric contention), 304 WGs x 256 thr each.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)

static const int NR = 8;
static const size_t SLICE = 6291456; /* elements (bf16-sized ushort): 12.6 MB */

__global__ void k_copy(ushort* dst, ushort* const* srcs, int rank, int width16, int stag,
                       size_t slice_elems) {
    const unsigned wg = blockIdx.x, nwg = gridDim.x;
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    for (int i = 0; i < NR; i++) {
        const int s = stag ? (int)((wg + rank + i) % NR) : i;
        const ushort* src = srcs[s];
        ushort* d = dst + (size_t)s * slice_elems;
        if (width16) {
            const uint4* s4 = (const uint4*)src;
            uint4* d4 = (uint4*)d;
            const size_t n4 = slice_elems / 8;
            for (size_t e = (size_t)wg * nt + tid; e < n4; e += (size_t)nwg * nt)
                d4[e] = s4[e];
        } else {
            for (size_t e = (size_t)wg * nt + tid; e < slice_elems; e += (size_t)nwg * nt)
                d[e] = src[e];
        }
    }
}


__global__ void k_copy_w(ushort* dst, ushort* const* srcs, int rank, int wsel, size_t slice_elems) {
    const unsigned wg = blockIdx.x, nwg = gridDim.x;
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    for (int i = 0; i < NR; i++) {
        const int s = (int)((wg + rank + i) % NR);
        const ushort* src = srcs[s];
        ushort* d = dst + (size_t)s * slice_elems;
        if (wsel == 0) { /* 4B */
            const uint* s2 = (const uint*)src; uint* d2 = (uint*)d;
            const size_t n2 = slice_elems / 2;
            for (size_t e = (size_t)wg * nt + tid; e < n2; e += (size_t)nwg * nt) d2[e] = s2[e];
        } else if (wsel == 1) { /* 8B */
            const uint2* s2 = (const uint2*)src; uint2* d2 = (uint2*)d;
            const size_t n2 = slice_elems / 4;
            for (size_t e = (size_t)wg * nt + tid; e < n2; e += (size_t)nwg * nt) d2[e] = s2[e];
        } else { /* 2B x4 unroll */
            const size_t n = slice_elems;
            const size_t step = (size_t)nwg * nt;
            for (size_t e = (size_t)wg * nt + tid; e + 3 * step < n; e += 4 * step) {
                ushort a = src[e], b = src[e + step], c = src[e + 2 * step], f = src[e + 3 * step];
                d[e] = a; d[e + step] = b; d[e + 2 * step] = c; d[e + 3 * step] = f;
            }
        }
    }
}

int main() {
    ushort* bufs[NR];   /* peer-visible source slice per device */
    ushort* dsts[NR];   /* local destination per device */
    ushort** dsrcs[NR]; /* device-side srcs table per device */
    for (int d = 0; d < NR; d++) {
        CHECK(hipSetDevice(d));
        for (int p = 0; p < NR; p++)
            if (p != d) hipDeviceEnablePeerAccess(p, 0); /* ignore already-enabled */
        CHECK(hipMalloc(&bufs[d], SLICE * 2));
        CHECK(hipMalloc(&dsts[d], (size_t)NR * SLICE * 2));
        CHECK(hipMemset(bufs[d], 0x3c, SLICE * 2));
    }
    for (int d = 0; d < NR; d++) {
        CHECK(hipSetDevice(d));
        CHECK(hipMalloc(&dsrcs[d], NR * sizeof(void*)));
        CHECK(hipMemcpy(dsrcs[d], bufs, NR * sizeof(void*), hipMemcpyHostToDevice));
    }
    hipStream_t st[NR];
    for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamCreate(&st[d])); }

    const double bytes_per_rank = (double)NR * SLICE * 2; /* read volume (1/8 local) */
    for (int width16 = 0; width16 <= 1; width16++)
        for (int stag = 0; stag <= 1; stag++) {
            /* warmup */
            for (int d = 0; d < NR; d++) {
                CHECK(hipSetDevice(d));
                hipLaunchKernelGGL(k_copy, dim3(304), dim3(256), 0, st[d], dsts[d], dsrcs[d], d,
                                   width16, stag, SLICE);
            }
            for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(st[d])); }
            auto t0 = std::chrono::steady_clock::now();
            const int REP = 5;
            for (int rep = 0; rep < REP; rep++)
                for (int d = 0; d < NR; d++) {
                    CHECK(hipSetDevice(d));
                    hipLaunchKernelGGL(k_copy, dim3(304), dim3(256), 0, st[d], dsts[d], dsrcs[d], d,
                                       width16, stag, SLICE);
                }
            for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(st[d])); }
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count() / REP;
            printf("width=%s order=%s : %7.2f ms  %7.1f GB/s per rank (7/8 remote)\n",
                   width16 ? "16B" : " 2B", stag ? "stag" : " seq", ms,
                   bytes_per_rank / ms / 1e6);
        }

    for (int wsel = 0; wsel <= 2; wsel++) {
        for (int d = 0; d < NR; d++) {
            hipSetDevice(d);
            hipLaunchKernelGGL(k_copy_w, dim3(304), dim3(256), 0, st[d], dsts[d], dsrcs[d], d, wsel, SLICE);
        }
        for (int d = 0; d < NR; d++) { hipSetDevice(d); hipStreamSynchronize(st[d]); }
        auto t0 = std::chrono::steady_clock::now();
        const int REP = 5;
        for (int rep = 0; rep < REP; rep++)
            for (int d = 0; d < NR; d++) {
                hipSetDevice(d);
                hipLaunchKernelGGL(k_copy_w, dim3(304), dim3(256), 0, st[d], dsts[d], dsrcs[d], d, wsel, SLICE);
            }
        for (int d = 0; d < NR; d++) { hipSetDevice(d); hipStreamSynchronize(st[d]); }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / REP;
        const char* names[] = {"4B stag", "8B stag", "2Bx4 stag"};
        printf("width=%-9s : %7.2f ms  %7.1f GB/s per rank\n", names[wsel], ms, bytes_per_rank / ms / 1e6);
    }
    return 0;
}

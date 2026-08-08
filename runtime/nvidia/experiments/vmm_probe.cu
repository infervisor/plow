// CUDA VMM feasibility probe for the head-major prefix-cache design
//: can a "virtually-contiguous KV address
// view" be built from 2 MiB physical granules with cuMemAddressReserve /
// cuMemCreate / cuMemMap / cuMemSetAccess, and what does each step cost?
//
// All core VMM calls go through the SAME dlopen pattern plowrt uses
// (crates/plowrt/src/device/cuda.rs: SONAME first, then dev symlink, then
// absolute homes; symbols via dlsym — NO -lcuda link). cudart is used only
// for convenience (context init, kernels, memcpy), as the production runtime
// would already have a live context.
//
// Sections:
//   [1] granularity      cuMemGetAllocationGranularity (min + recommended)
//   [2] multi-map        SAME physical handle mapped into TWO VA ranges at
//                        DIFFERENT offsets (the prefix-sharing primitive);
//                        write via view A, read via view B, byte-verify
//   [3] latency          per-granule create/map/setaccess/unmap at 2 MiB and
//                        16 MiB granule sizes, batched map + one-shot
//                        setaccess, D2D copy bandwidth reference, and the full
//                        realistic 31B@128k prefix attach from the design doc:
//                        10 full layers x 4 kv heads x {K,V} x 131072 rows x
//                        1 KiB = 10 GiB = 5120 x 2 MiB granules (or 640 x 16 MiB)
//   [5] growth-under-use cuMemMap+cuMemSetAccess of NEW tail granules while a
//                        kernel actively reads the already-mapped prefix of
//                        the SAME VA range — legality + stall cost
//   [6] VA budget        reserve 40 GiB (8 slots x 5 GiB) and larger
//
// Build (off-GPU):  nvcc -arch=sm_120a -O2 -o vmm_probe vmm_probe.cu -ldl
// Run:              gpulease <tag> ./vmm_probe
//
// ============================ RESULTS =====================================
// NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 580.82.07, CUDA 13.0
// (driver-api 13000), 95.0 GiB VRAM, 188 SMs. Measured 2026-07-19, two
// independent leases, numbers reproduced within ~5%.
//
// [1] granularity: minimum = 2 MiB, recommended = 2 MiB. The design's 2 MiB
//     granule IS the hardware granule; no slack.
// [2] multi-map: LEGAL and byte-exact. Same handle mapped into two VA ranges
//     simultaneously at different in-range offsets (owner+0 / borrower+2G):
//     host write via A reads back via B, kernel write via A verifies via B —
//     0 mismatches in 4 MiB. After the OWNER unmaps and cuMemRelease's the
//     handle, the borrower's mapping still reads intact: the driver refcounts
//     mappings, host-side refcount is policy only.
// [3] latency, per 2 MiB granule (N=64):
//       cuMemCreate     13.8 us avg     cuMemMap   1.0 us avg (0.32 us/g when
//       issued back-to-back; 16 MiB granules map at 0.3 us)
//       cuMemSetAccess  ~69 us/granule — and it is PER UNDERLYING GRANULE
//       MAPPING, not per byte and NOT amortized by batching: one call over
//       64 x 2 MiB (128 MiB) = 4.2 ms, one call over 64 x 16 MiB (1 GiB) =
//       4.3 ms. Equal granule count => equal cost at 8x the bytes.
//       cuMemUnmap      ~39 us/granule (full-range unmap of 64 = 2.5 ms).
//     THE decisive number — 31B@128k full-layer prefix attach (10 GiB,
//     borrower maps EXISTING handles + one setaccess), by granule size:
//       2 MiB  x 5120: map 5.8  ms + setaccess 358 ms = 364  ms; detach 193 ms
//       16 MiB x  640: map 0.20 ms + setaccess  44 ms = 44.3 ms; detach  25 ms
//       64 MiB x  160: map 0.05 ms + setaccess  11 ms = 11.4 ms; detach 5.9 ms
//     Owner-side cuMemCreate of the 10 GiB: 76 ms at 2 MiB, 9 ms at 16 MiB,
//     2.3 ms at 64 MiB (paid at prefill, amortized). D2D reference on this box:
//     731 GB/s => the status-quo 10 GiB prefix blit costs 14.7 ms (not the
//     6.7 ms the design assumed from 1.5 TB/s — D2D traffic is read+write).
//     So: cuMemMap meets the design's <=1.3 us/map bar, but cuMemSetAccess
//     is the real cost and was not in the design's model. 2 MiB granules
//     LOSE to the copy 25x; 16 MiB lose 3x; 64 MiB wins (11.4 vs 14.7 ms).
//     Per-byte: setaccess ~34.5 us/MiB-of-2MiB-granules vs copy 1.4 us/MiB
//     => break-even granule size is ~50 MiB, independent of prefix length.
// [5] growth-under-use: SAFE. Mapping + setaccess of new tail granules while
//     a kernel spins reading the mapped prefix of the SAME VA range: no
//     error, no corruption of prefix or new granules, and NO implicit sync —
//     per-granule map+setaccess 66 us during the kernel vs 67 us idle; all
//     16 maps landed 1.1 ms into a 401 ms kernel which ran to schedule.
// [6] VA budget: 10 GiB (one 31B/128k seq), 80 GiB (8 slots), 256 GiB and
//     1 TiB reservations ALL succeed in 3-19 us each. VA is free; reserve
//     the worst case up front.
// ==========================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

// ---- dlopen shim: mirrors crates/plowrt/src/device/cuda.rs ---------------
static void* drv = nullptr;
#define DECL(fn) static decltype(&fn) p_##fn = nullptr
DECL(cuGetErrorName);
DECL(cuMemGetAllocationGranularity);
DECL(cuMemAddressReserve);
DECL(cuMemAddressFree);
DECL(cuMemCreate);
DECL(cuMemRelease);
DECL(cuMemMap);
DECL(cuMemUnmap);
DECL(cuMemSetAccess);
#undef DECL

static void load_driver() {
    const char* cands[] = {
        "libcuda.so.1", "libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
    };
    for (const char* c : cands) { drv = dlopen(c, RTLD_NOW); if (drv) break; }
    if (!drv) { printf("FATAL dlopen libcuda: %s\n", dlerror()); exit(1); }
#define LOAD(fn) do { p_##fn = (decltype(&fn))dlsym(drv, #fn); \
    if (!p_##fn) { printf("FATAL dlsym " #fn "\n"); exit(1); } } while (0)
    LOAD(cuGetErrorName);
    LOAD(cuMemGetAllocationGranularity);
    LOAD(cuMemAddressReserve);
    LOAD(cuMemAddressFree);
    LOAD(cuMemCreate);
    LOAD(cuMemRelease);
    LOAD(cuMemMap);
    LOAD(cuMemUnmap);
    LOAD(cuMemSetAccess);
#undef LOAD
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define DK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char* s = "?"; p_cuGetErrorName(r, &s); \
    printf("DRV ERR %s @%d: %s\n", #x, __LINE__, s); exit(1); } } while (0)

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static CUmemGenericAllocationHandle mk(size_t sz) {
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = 0;
    CUmemGenericAllocationHandle h;
    DK(p_cuMemCreate(&h, sz, &p, 0));
    return h;
}
static void acc(CUdeviceptr ptr, size_t sz) {
    CUmemAccessDesc d = {};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = 0;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DK(p_cuMemSetAccess(ptr, sz, &d, 1));
}

__global__ void fill_pattern(unsigned* p, size_t n, unsigned seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = seed ^ (unsigned)i;
}
__global__ void check_pattern(const unsigned* p, size_t n, unsigned seed, unsigned* bad) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        if (p[i] != (seed ^ (unsigned)i)) atomicAdd(bad, 1u);
}
// Spins reading buf[0..nwords) repeatedly for ~`cycles` GPU cycles.
__global__ void spin_read(const unsigned* buf, size_t nwords, long long cycles,
                          unsigned* sink) {
    long long t0 = clock64();
    unsigned a = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    do {
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nwords; i += stride)
            a ^= buf[i];
    } while (clock64() - t0 < cycles);
    if (a == 0xdeadbeefu) *sink = a; // keep the loop alive
}

int main() {
    load_driver();
    CK(cudaSetDevice(0));
    CK(cudaFree(0)); // primary context up; driver calls below share it
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    size_t free_b, tot_b;
    CK(cudaMemGetInfo(&free_b, &tot_b));
    int drv_ver = 0;
    CK(cudaDriverGetVersion(&drv_ver));
    printf("# %s  SMs=%d  vram total=%.1f GiB free=%.1f GiB  driver-api=%d\n",
           prop.name, prop.multiProcessorCount, tot_b / 1073741824.0,
           free_b / 1073741824.0, drv_ver);

    // ---- [1] granularity -------------------------------------------------
    CUmemAllocationProp ap = {};
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id = 0;
    size_t gmin = 0, grec = 0;
    DK(p_cuMemGetAllocationGranularity(&gmin, &ap, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    DK(p_cuMemGetAllocationGranularity(&grec, &ap, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    printf("\n[1] granularity: minimum=%zu (%.1f MiB)  recommended=%zu (%.1f MiB)\n",
           gmin, gmin / 1048576.0, grec, grec / 1048576.0);
    const size_t G = gmin; // granule size used everywhere below

    // ---- [2] multi-map: same handle in two VA ranges ---------------------
    {
        CUdeviceptr va, vb;
        DK(p_cuMemAddressReserve(&va, 4 * G, 0, 0, 0)); // owner view
        DK(p_cuMemAddressReserve(&vb, 8 * G, 0, 0, 0)); // borrower view
        CUmemGenericAllocationHandle h0 = mk(G), h1 = mk(G);
        // owner: h0 at +0, h1 at +G
        DK(p_cuMemMap(va, G, 0, h0, 0));
        DK(p_cuMemMap(va + G, G, 0, h1, 0));
        acc(va, 2 * G);
        // borrower: SAME handles at a DIFFERENT offset (+2G, +3G) — the
        // shape a shared prefix has inside a borrower's larger view.
        DK(p_cuMemMap(vb + 2 * G, G, 0, h0, 0));
        DK(p_cuMemMap(vb + 3 * G, G, 0, h1, 0));
        acc(vb + 2 * G, 2 * G);

        size_t nw = 2 * G / 4;
        // host write via A, host read via B
        std::vector<unsigned> hbuf(nw), rbuf(nw);
        for (size_t i = 0; i < nw; i++) hbuf[i] = 0xa5000000u ^ (unsigned)i;
        CK(cudaMemcpy((void*)va, hbuf.data(), 2 * G, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(rbuf.data(), (void*)(vb + 2 * G), 2 * G, cudaMemcpyDeviceToHost));
        size_t mm = 0;
        for (size_t i = 0; i < nw; i++) mm += (rbuf[i] != hbuf[i]);
        // kernel write via A, kernel read (verify) via B
        unsigned *dbad, hbad = 0;
        CK(cudaMalloc(&dbad, 4));
        CK(cudaMemset(dbad, 0, 4));
        fill_pattern<<<256, 256>>>((unsigned*)va, nw, 0x51u);
        CK(cudaDeviceSynchronize());
        check_pattern<<<256, 256>>>((const unsigned*)(vb + 2 * G), nw, 0x51u, dbad);
        CK(cudaMemcpy(&hbad, dbad, 4, cudaMemcpyDeviceToHost));
        printf("[2] multi-map: host A->B mismatches=%zu  kernel A->B mismatches=%u  -> %s\n",
               mm, hbad, (mm == 0 && hbad == 0) ? "LEGAL, byte-exact" : "BROKEN");
        // release the handle while the second mapping persists (refcount check)
        DK(p_cuMemUnmap(va, 2 * G));
        DK(p_cuMemRelease(h0));
        DK(p_cuMemRelease(h1)); // borrower mapping must keep memory alive
        CK(cudaMemcpy(rbuf.data(), (void*)(vb + 2 * G), 2 * G, cudaMemcpyDeviceToHost));
        size_t mm2 = 0;
        for (size_t i = 0; i < nw; i++) mm2 += (rbuf[i] != (0x51u ^ (unsigned)i));
        printf("    after owner unmap+release, borrower still reads: mismatches=%zu (%s)\n",
               mm2, mm2 == 0 ? "refcounted, OK" : "BROKEN");
        DK(p_cuMemUnmap(vb + 2 * G, 2 * G));
        DK(p_cuMemAddressFree(va, 4 * G));
        DK(p_cuMemAddressFree(vb, 8 * G));
        CK(cudaFree(dbad));
    }

    // ---- [3] latency -----------------------------------------------------
    {
        const int N = 64; // 128 MiB
        CUdeviceptr va;
        DK(p_cuMemAddressReserve(&va, N * G, 0, 0, 0));
        std::vector<CUmemGenericAllocationHandle> hs(N);
        std::vector<double> tcreate(N), tmap(N), tacc(N);
        for (int i = 0; i < N; i++) {
            double t0 = now_ms();
            hs[i] = mk(G);
            tcreate[i] = now_ms() - t0;
        }
        for (int i = 0; i < N; i++) {
            double t0 = now_ms();
            DK(p_cuMemMap(va + (size_t)i * G, G, 0, hs[i], 0));
            tmap[i] = now_ms() - t0;
        }
        for (int i = 0; i < N; i++) {
            double t0 = now_ms();
            acc(va + (size_t)i * G, G);
            tacc[i] = now_ms() - t0;
        }
        auto stat = [](std::vector<double>& v, const char* name) {
            double s = 0, mx = 0;
            for (double x : v) { s += x; mx = std::max(mx, x); }
            printf("    %-14s avg %7.1f us  max %7.1f us\n", name, 1e3 * s / v.size(), 1e3 * mx);
        };
        printf("[3] per-2MiB-granule latency (N=%d):\n", N);
        stat(tcreate, "cuMemCreate");
        stat(tmap, "cuMemMap");
        stat(tacc, "cuMemSetAccess");
        double t0 = now_ms();
        DK(p_cuMemUnmap(va, (size_t)N * G));
        double tun = now_ms() - t0;
        printf("    full-range cuMemUnmap of %d granules: %.1f us total\n", N, 1e3 * tun);
        // batched: re-map all, ONE setaccess over the whole range
        t0 = now_ms();
        for (int i = 0; i < N; i++) DK(p_cuMemMap(va + (size_t)i * G, G, 0, hs[i], 0));
        double tmapb = now_ms() - t0;
        t0 = now_ms();
        acc(va, (size_t)N * G);
        double taccb = now_ms() - t0;
        printf("    batched 64-granule (128 MiB): map %.1f us total (%.2f us/g), "
               "ONE setaccess %.1f us (%.2f us/g)\n",
               1e3 * tmapb, 1e3 * tmapb / N, 1e3 * taccb, 1e3 * taccb / N);
        DK(p_cuMemUnmap(va, (size_t)N * G));
        for (int i = 0; i < N; i++) DK(p_cuMemRelease(hs[i]));
        DK(p_cuMemAddressFree(va, (size_t)N * G));

        // 16 MiB granule sweep (the design's map-count mitigation, doc section 3)
        {
            const size_t G16 = 8 * G;
            const int N16 = 64; // 1 GiB
            CUdeviceptr v16;
            DK(p_cuMemAddressReserve(&v16, N16 * G16, 0, 0, 0));
            std::vector<CUmemGenericAllocationHandle> h16(N16);
            std::vector<double> tc(N16), tm(N16);
            for (int i = 0; i < N16; i++) {
                double t0 = now_ms();
                h16[i] = mk(G16);
                tc[i] = now_ms() - t0;
            }
            for (int i = 0; i < N16; i++) {
                double t0 = now_ms();
                DK(p_cuMemMap(v16 + (size_t)i * G16, G16, 0, h16[i], 0));
                tm[i] = now_ms() - t0;
            }
            double t0 = now_ms();
            acc(v16, (size_t)N16 * G16);
            double ta = now_ms() - t0;
            printf("    16MiB granules (N=%d):\n", N16);
            stat(tc, "cuMemCreate");
            stat(tm, "cuMemMap");
            printf("    one setaccess over 1 GiB: %.1f us\n", 1e3 * ta);
            DK(p_cuMemUnmap(v16, (size_t)N16 * G16));
            for (int i = 0; i < N16; i++) DK(p_cuMemRelease(h16[i]));
            DK(p_cuMemAddressFree(v16, (size_t)N16 * G16));
        }

        // D2D reference: what the status-quo prefix blit actually runs at
        {
            size_t db = 1ull << 30;
            void *s, *d;
            CK(cudaMalloc(&s, db));
            CK(cudaMalloc(&d, db));
            CK(cudaMemset(s, 1, db));
            CK(cudaMemcpy(d, s, db, cudaMemcpyDeviceToDevice)); // warm
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 4; i++) CK(cudaMemcpy(d, s, db, cudaMemcpyDeviceToDevice));
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            printf("    D2D copy reference: %.0f GB/s -> a 10 GiB 31B@128k prefix blit = %.1f ms\n",
                   4.0 * db / (ms / 1e3) / 1e9, 10.737418240 / (4.0 * db / (ms / 1e3) / 1e9) * 1e3);
            CK(cudaFree(s));
            CK(cudaFree(d));
            CK(cudaEventDestroy(e0));
            CK(cudaEventDestroy(e1));
        }

        // realistic full attach, the design-doc number (rtx-09 section 3):
        // 31B@128k full-layer prefix = 10 layers x 4 kv heads x {K,V} x
        // 131072 rows x 1 KiB = 10 GiB = 5120 x 2MiB granules (640 x 16MiB).
        const size_t gmul[3] = {1, 8, 32}; // 2, 16, 64 MiB granules
        for (int gi = 0; gi < 3; gi++) {
            const size_t GA = gmul[gi] * G;
            const size_t NA = (10ull << 30) / GA;
            printf("    31B@128k attach as %zu x %zu MiB granules (%.1f GiB):\n",
                   NA, GA >> 20, NA * GA / 1073741824.0);
            std::vector<CUmemGenericAllocationHandle> ah(NA);
            t0 = now_ms();
            for (size_t i = 0; i < NA; i++) ah[i] = mk(GA);
            double tcre_all = now_ms() - t0;
            CUdeviceptr av;
            DK(p_cuMemAddressReserve(&av, NA * GA, 0, 0, 0));
            t0 = now_ms();
            for (size_t i = 0; i < NA; i++) DK(p_cuMemMap(av + i * GA, GA, 0, ah[i], 0));
            double tmap_all = now_ms() - t0;
            t0 = now_ms();
            acc(av, NA * GA);
            double tacc_all = now_ms() - t0;
            t0 = now_ms();
            DK(p_cuMemUnmap(av, NA * GA));
            double tun_all = now_ms() - t0;
            printf("      owner-side cuMemCreate x%zu: %.1f ms (paid at prefill, not attach)\n",
                   NA, tcre_all);
            printf("      ATTACH (map x%zu + 1 setaccess): %.2f + %.2f = %.2f ms\n",
                   NA, tmap_all, tacc_all, tmap_all + tacc_all);
            printf("      DETACH (full-range unmap):       %.2f ms\n", tun_all);
            for (size_t i = 0; i < NA; i++) DK(p_cuMemRelease(ah[i]));
            DK(p_cuMemAddressFree(av, NA * GA));
        }
    }

    // ---- [5] growth-under-use --------------------------------------------
    {
        const int NPRE = 8, NGROW = 16, NTOT = NPRE + NGROW;
        CUdeviceptr va;
        DK(p_cuMemAddressReserve(&va, (size_t)NTOT * G, 0, 0, 0));
        std::vector<CUmemGenericAllocationHandle> hs(NTOT);
        for (int i = 0; i < NPRE; i++) {
            hs[i] = mk(G);
            DK(p_cuMemMap(va + (size_t)i * G, G, 0, hs[i], 0));
        }
        acc(va, (size_t)NPRE * G);
        size_t nw = (size_t)NPRE * G / 4;
        fill_pattern<<<256, 256>>>((unsigned*)va, nw, 0x77u);
        CK(cudaDeviceSynchronize());
        for (int i = NPRE; i < NTOT; i++) hs[i] = mk(G); // handles pre-created

        // idle-GPU baseline: map+setaccess per granule with nothing running
        std::vector<double> tidle(NGROW), tbusy(NGROW);
        for (int i = NPRE; i < NTOT; i++) {
            double t0 = now_ms();
            DK(p_cuMemMap(va + (size_t)i * G, G, 0, hs[i], 0));
            acc(va + (size_t)i * G, G);
            tidle[i - NPRE] = now_ms() - t0;
        }
        DK(p_cuMemUnmap(va + (size_t)NPRE * G, (size_t)NGROW * G));

        // now with a kernel actively reading the prefix of the SAME range
        int khz = 0;
        CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
        long long cyc = (long long)khz * 400; // ~400 ms spin
        unsigned* dsink;
        CK(cudaMalloc(&dsink, 4));
        cudaStream_t s;
        CK(cudaStreamCreate(&s));
        double tk0 = now_ms();
        spin_read<<<64, 256, 0, s>>>((const unsigned*)va, nw, cyc, dsink);
        CK(cudaGetLastError());
        for (int i = NPRE; i < NTOT; i++) {
            double t0 = now_ms();
            DK(p_cuMemMap(va + (size_t)i * G, G, 0, hs[i], 0));
            acc(va + (size_t)i * G, G);
            tbusy[i - NPRE] = now_ms() - t0;
        }
        double tmaps_done = now_ms() - tk0;
        CK(cudaStreamSynchronize(s));
        double tkern = now_ms() - tk0;
        // prefix integrity after growth-under-use
        unsigned *dbad, hbad = 0;
        CK(cudaMalloc(&dbad, 4));
        CK(cudaMemset(dbad, 0, 4));
        check_pattern<<<256, 256>>>((const unsigned*)va, nw, 0x77u, dbad);
        // and the new granules are usable
        fill_pattern<<<256, 256>>>((unsigned*)(va + (size_t)NPRE * G),
                                   (size_t)NGROW * G / 4, 0x99u);
        check_pattern<<<256, 256>>>((const unsigned*)(va + (size_t)NPRE * G),
                                    (size_t)NGROW * G / 4, 0x99u, dbad);
        CK(cudaMemcpy(&hbad, dbad, 4, cudaMemcpyDeviceToHost));
        double si = 0, sb = 0;
        for (int i = 0; i < NGROW; i++) { si += tidle[i]; sb += tbusy[i]; }
        printf("[5] growth-under-use: map+setaccess per granule  idle %.1f us  "
               "during-kernel %.1f us\n", 1e3 * si / NGROW, 1e3 * sb / NGROW);
        printf("    all %d maps done %.1f ms into a %.1f ms kernel (%s); "
               "corrupted words=%u (%s)\n",
               NGROW, tmaps_done, tkern,
               tmaps_done < tkern * 0.8 ? "NO implicit sync" : "SYNCED — maps waited",
               hbad, hbad == 0 ? "prefix+growth intact" : "BROKEN");
        DK(p_cuMemUnmap(va, (size_t)NTOT * G));
        for (int i = 0; i < NTOT; i++) DK(p_cuMemRelease(hs[i]));
        DK(p_cuMemAddressFree(va, (size_t)NTOT * G));
        CK(cudaFree(dsink));
        CK(cudaFree(dbad));
        CK(cudaStreamDestroy(s));
    }

    // ---- [6] VA budget ---------------------------------------------------
    {
        struct { const char* what; size_t sz; } cases[] = {
            {"per-seq 31B/128k view (10 GiB)", 10ull << 30},
            {"8 slots (80 GiB)", 80ull << 30},
            {"256 GiB", 256ull << 30},
            {"1 TiB", 1ull << 40},
        };
        printf("[6] VA reservations:\n");
        for (auto& c : cases) {
            CUdeviceptr p = 0;
            double t0 = now_ms();
            CUresult r = p_cuMemAddressReserve(&p, c.sz, 0, 0, 0);
            double dt = now_ms() - t0;
            if (r == CUDA_SUCCESS) {
                printf("    %-32s OK  (%.1f us)  base=0x%llx\n", c.what, 1e3 * dt,
                       (unsigned long long)p);
                DK(p_cuMemAddressFree(p, c.sz));
            } else {
                const char* s = "?";
                p_cuGetErrorName(r, &s);
                printf("    %-32s FAILED: %s\n", c.what, s);
            }
        }
    }

    printf("\nDONE\n");
    return 0;
}

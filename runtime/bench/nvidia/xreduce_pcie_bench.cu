/* xreduce_pcie_bench.cu — two-GPU all-reduce bodies (op_collective.cuh) against PCIe copy
 * ceilings. One process drives both devices; regions are either page-locked host memory mapped
 * into both (the process rank-group transport) or each rank's VRAM reached by peer access.
 *
 * Build (nix shell, clean compiler env as runtime/cmake/nvcc_cubin.sh):
 *   nix develop --command bash -c 'env -i PATH="$PLOW_NVCC_PATH" ROCM_PATH="$ROCM_PATH" \
 *     NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS" "$PLOW_NVCC" -std=c++17 -O3 -arch=sm_120a \
 *     -I runtime/common -I runtime/nvidia runtime/bench/nvidia/xreduce_pcie_bench.cu \
 *     -o /tmp/xr_bench'
 * Run: xr_bench [nblk ...] (both devices visible). */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dev_isa.h"
#include "op_collective.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)

constexpr int kIters = 40;

/* `iters` back-to-back collectives in one cooperative launch, each on fresh gate lines (2 per
 * iteration + the status line 0). The grid sync stands in for the interpreter's local gate
 * between packets: a rank arrives at the next collective only after all its blocks finished
 * reading the previous one, so the in-place two-shot slot is never overwritten under a reader. */
__global__ void __launch_bounds__(256, 1)
k_loop(PlowProgram prog, PlowDevInst in, void* out, void* resid, void* out2, bool twoshot,
       int iters) {
    void* tensors[] = {out, resid, out2};
    for (int it = 0; it < iters; it++) {
        in.i[3] = 1u + 2u * it;
        in.i[4] = 2u + 2u * it;
        if (twoshot) d_xreduce_twoshot_nv(prog, &in, tensors, blockIdx.x, gridDim.x);
        else d_xreduce_nv(prog, &in, tensors, blockIdx.x, gridDim.x);
        cooperative_groups::this_grid().sync();
    }
}

__global__ void k_stream_ld(const uint4* p, size_t n, uint4* sink) {
    uint4 a{};
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        uint4 v;
        asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i) : "memory");
        a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
    }
    if (a.x == 0x12345678u) sink[0] = a;
}

__global__ void k_stream_st(uint4* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%1,%1,%1};" :: "l"(p + i), "r"((uint32_t)i)
                     : "memory");
}

struct Dev {
    cudaStream_t s;
    cudaEvent_t a, b;
};
static Dev dev[2];

template <class F>
static double time_both(F launch, bool both = true) {
    for (int d = 0; d < (both ? 2 : 1); d++) {
        CK(cudaSetDevice(d));
        CK(cudaEventRecord(dev[d].a, dev[d].s));
        launch(d);
        CK(cudaEventRecord(dev[d].b, dev[d].s));
    }
    double ms = 0;
    for (int d = 0; d < (both ? 2 : 1); d++) {
        CK(cudaSetDevice(d));
        CK(cudaEventSynchronize(dev[d].b));
        float t;
        CK(cudaEventElapsedTime(&t, dev[d].a, dev[d].b));
        ms = t > ms ? t : ms;
    }
    return ms;
}

static void ceilings(int sms, bool peer) {
    const size_t bytes = 256u << 20;
    void *d0, *d1, *h0, *h1;
    CK(cudaSetDevice(0)); CK(cudaMalloc(&d0, bytes)); CK(cudaHostAlloc(&h0, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    CK(cudaSetDevice(1)); CK(cudaMalloc(&d1, bytes)); CK(cudaHostAlloc(&h1, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    uint4* sink[2];
    for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaMalloc(&sink[d], 16)); }
    auto gbs = [&](double ms, double b) { return b / (ms * 1e6); };
    for (int rep = 0; rep < 2; rep++) {
        const bool last = rep == 1;
        double t;
        const size_t nv = bytes / 16;
        t = time_both([&](int) { k_stream_ld<<<sms * 4, 256, 0, dev[0].s>>>((const uint4*)h0, nv, sink[0]); }, false);
        if (last) std::printf("CEIL zero-copy read host v4    %7.1f GB/s\n", gbs(t, bytes));
        t = time_both([&](int) { k_stream_st<<<sms * 4, 256, 0, dev[0].s>>>((uint4*)h0, nv); }, false);
        if (last) std::printf("CEIL zero-copy write host v4   %7.1f GB/s\n", gbs(t, bytes));
        t = time_both([&](int d) {
            CK(cudaMemcpyAsync(d ? h1 : h0, d ? d1 : d0, bytes, cudaMemcpyDeviceToHost, dev[d].s));
        });
        if (last) std::printf("CEIL D2H pinned (both GPUs)    %7.1f GB/s (sum)\n", gbs(t, 2.0 * bytes));
        t = time_both([&](int d) {
            CK(cudaMemcpyAsync(d ? d1 : d0, d ? h0 : h1, bytes, cudaMemcpyHostToDevice, dev[d].s));
        });
        if (last) std::printf("CEIL H2D pinned (both GPUs)    %7.1f GB/s (sum)\n", gbs(t, 2.0 * bytes));
        t = time_both([&](int) { CK(cudaMemcpyAsync(h0, d0, bytes, cudaMemcpyDeviceToHost, dev[0].s)); }, false);
        if (last) std::printf("CEIL D2H pinned (1 GPU)        %7.1f GB/s\n", gbs(t, bytes));
        t = time_both([&](int) { CK(cudaMemcpyAsync(d0, h0, bytes, cudaMemcpyHostToDevice, dev[0].s)); }, false);
        if (last) std::printf("CEIL H2D pinned (1 GPU)        %7.1f GB/s\n", gbs(t, bytes));
        if (!peer) continue;
        t = time_both([&](int) { CK(cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes, dev[0].s)); }, false);
        if (last) std::printf("CEIL memcpyPeer 0->1           %7.1f GB/s\n", gbs(t, bytes));
        t = time_both([&](int d) {
            if (d == 0) CK(cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes / 2, dev[0].s));
            else CK(cudaMemcpyPeerAsync((char*)d0 + bytes / 2, 0, (char*)d1 + bytes / 2, 1, bytes / 2, dev[1].s));
        });
        if (last) std::printf("CEIL memcpyPeer bidirectional  %7.1f GB/s (sum)\n", gbs(t, bytes));
        t = time_both([&](int) { k_stream_ld<<<sms * 4, 256, 0, dev[0].s>>>((const uint4*)d1, nv, sink[0]); }, false);
        if (last) std::printf("CEIL peer read v4 (0 reads 1)  %7.1f GB/s\n", gbs(t, bytes));
        t = time_both([&](int) { k_stream_st<<<sms * 4, 256, 0, dev[0].s>>>((uint4*)d1, nv); }, false);
        if (last) std::printf("CEIL peer write v4 (0 -> 1)    %7.1f GB/s\n", gbs(t, bytes));
    }
    CK(cudaSetDevice(0)); CK(cudaFree(d0)); CK(cudaFreeHost(h0)); CK(cudaFree(sink[0]));
    CK(cudaSetDevice(1)); CK(cudaFree(d1)); CK(cudaFreeHost(h1)); CK(cudaFree(sink[1]));
}

static uint16_t to_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
static float from_bf16(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

static bool g_fail = false;

/* Flag round trip with the collective's own primitives: rank `me` waits for `k` in its local
 * word, then stores `k` into the other side's word; rank 0 starts. One thread per device. */
__global__ void k_ping(uint32_t* mine, uint32_t* theirs, int me, int rounds, uint32_t* fail) {
    for (int k = 1; k <= rounds; k++) {
        if (me == 0) xr_store(theirs, (uint32_t)k);
        const unsigned long long t0 = xr_gtime();
        while (xr_ld32(mine) != (uint32_t)k)
            if (xr_gtime() - t0 > 1000000000ull) { *fail = k; return; }
        if (me == 1) xr_store(theirs, (uint32_t)k);
    }
}

static bool ping(bool p2p) {
    const int rounds = 2000;
    uint32_t* w[2];
    uint32_t* fail[2];
    char* host = nullptr;
    if (p2p) {
        for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaMalloc(&w[d], 4096)); CK(cudaMemset(w[d], 0, 4096)); }
    } else {
        CK(cudaHostAlloc(&host, 8192, cudaHostAllocPortable | cudaHostAllocMapped));
        std::memset(host, 0, 8192);
        w[0] = (uint32_t*)host;
        w[1] = (uint32_t*)(host + 4096);
    }
    for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaMallocManaged(&fail[d], 4)); *fail[d] = 0; CK(cudaDeviceSynchronize()); }
    const double ms = time_both([&](int d) { k_ping<<<1, 1, 0, dev[d].s>>>(w[d], w[1 - d], d, rounds, fail[d]); });
    std::printf("PING %-5s round trip %7.2f us  fail=%u/%u\n", p2p ? "p2p" : "host", ms * 1e3 / rounds,
                *fail[0], *fail[1]);
    bool ok = !*fail[0] && !*fail[1];
    if (!ok) g_fail = true;
    /* Peer store visibility both ways, checked from the host. */
    if (p2p) {
        for (int d = 0; d < 2; d++) {
            CK(cudaSetDevice(d));
            k_stream_st<<<4, 256, 0, dev[d].s>>>((uint4*)w[1 - d], 256);
            CK(cudaStreamSynchronize(dev[d].s));
            std::vector<uint32_t> got(1024);
            CK(cudaMemcpy(got.data(), w[1 - d], 4096, cudaMemcpyDefault));
            bool vis = true;
            for (int i = 0; i < 1024; i++) vis &= got[i] == (uint32_t)(i / 4);
            std::printf("P2P store %d -> %d visible: %s\n", d, 1 - d, vis ? "yes" : "NO");
            if (!vis) { g_fail = true; ok = false; }
        }
    }
    for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaFree(fail[d])); if (p2p) CK(cudaFree(w[d])); }
    if (host) CK(cudaFreeHost(host));
    return ok;
}

/* One shape: a single concurrent collective on both ranks checked byte for byte against the
 * host f32-in-rank-order reference (with status), then kIters timed back to back from fresh
 * inputs and gates. Reports per-collective time and algorithm bandwidth (n*2 bytes / time). */
static void collective(bool p2p, bool twoshot, uint32_t n, unsigned nblk) {
    const uint32_t slot = 0;
    const size_t bytes = (size_t)n * 2, xoff = (bytes + 4095) & ~size_t(4095);
    const size_t region = xoff + (size_t)(2 * kIters + 2) * PLOW_CTR_STRIDE * 4;
    char* host = nullptr;
    char* base[2];
    if (p2p) {
        for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaMalloc(&base[d], region)); }
    } else {
        CK(cudaHostAlloc(&host, 2 * region, cudaHostAllocPortable | cudaHostAllocMapped));
        base[0] = host;
        base[1] = host + region;
    }
    std::vector<uint16_t> part[2], ref(n), got(n);
    for (int r = 0; r < 2; r++) {
        part[r].resize(n);
        for (uint32_t e = 0; e < n; e++) part[r][e] = to_bf16(((e * 37u + r * 101u) % 997u) / 499.0f - 1.0f);
    }
    for (uint32_t e = 0; e < n; e++) ref[e] = to_bf16(0.0f + from_bf16(part[0][e]) + from_bf16(part[1][e]));
    void* outp[2];
    void** peers[2];
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        CK(cudaMalloc(&outp[d], bytes));
        CK(cudaMalloc(&peers[d], 2 * sizeof(void*)));
        CK(cudaMemcpy(peers[d], base, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    }
    auto reset = [&] {
        for (int d = 0; d < 2; d++) {
            CK(cudaSetDevice(d));
            CK(cudaMemcpy(base[d] + slot, part[d].data(), bytes, cudaMemcpyDefault));
            CK(cudaMemset(base[d] + xoff, 0, region - xoff));
            CK(cudaMemset(outp[d], 0xff, bytes));
            CK(cudaDeviceSynchronize());
        }
    };
    PlowDevInst in{};
    for (auto& t : in.t) t = PLOW_TENSOR_NONE;
    in.t[0] = 0;
    in.i[0] = n;
    in.i[1] = 2;
    in.i[2] = slot;
    in.fj[2].u = 1;
    auto launch = [&](int d, int iters) {
        PlowProgram prog{};
        prog.peer_scratch = peers[d];
        prog.n_gpu = 2;
        prog.rank = d;
        prog.xctr = (uint32_t*)(base[d] + xoff);
        void* o = outp[d];
        void* none = nullptr;
        void* args[] = {&prog, &in, &o, &none, &none, &twoshot, &iters};
        CK(cudaLaunchCooperativeKernel((void*)k_loop, nblk, 256, args, 0, dev[d].s));
    };
    auto status_ok = [&](const char* what) {
        for (int d = 0; d < 2; d++) {
            uint32_t st;
            CK(cudaSetDevice(d));
            CK(cudaMemcpy(&st, base[d] + xoff, 4, cudaMemcpyDefault));
            if (st) { std::printf("FAIL %s rank %d status %x\n", what, d, st); g_fail = true; return false; }
        }
        return true;
    };
    reset();
    time_both([&](int d) { launch(d, 1); });
    bool ok = status_ok("check");
    for (int d = 0; d < 2 && ok; d++) {
        CK(cudaSetDevice(d));
        CK(cudaMemcpy(got.data(), outp[d], bytes, cudaMemcpyDeviceToHost));
        if (std::memcmp(got.data(), ref.data(), bytes) != 0) {
            std::printf("FAIL bytes rank %d n=%u %s\n", d, n, twoshot ? "twoshot" : "oneshot");
            ok = false;
            g_fail = true;
        }
    }
    reset();
    const double ms = time_both([&](int d) { launch(d, kIters); });
    ok &= status_ok("timed");
    const double per = ms / kIters;
    std::printf("XR %-5s %-7s n=%-9u nblk=%-3u %9.2f us/coll  %7.2f GB/s alg  %s\n",
                p2p ? "p2p" : "host", twoshot ? "twoshot" : "oneshot", n, nblk, per * 1e3,
                bytes / (per * 1e6), ok ? "bytes=ok" : "INVALID");
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        CK(cudaFree(outp[d]));
        CK(cudaFree(peers[d]));
        if (p2p) CK(cudaFree(base[d]));
    }
    if (host) CK(cudaFreeHost(host));
}

int main(int argc, char** argv) {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { std::fprintf(stderr, "need 2 devices\n"); return 2; }
    int can01 = 0, can10 = 0, sms = 0;
    CK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        CK(cudaSetDeviceFlags(cudaDeviceMapHost));
        if (can01 && can10) CK(cudaDeviceEnablePeerAccess(1 - d, 0));
        CK(cudaStreamCreateWithFlags(&dev[d].s, cudaStreamNonBlocking));
        CK(cudaEventCreate(&dev[d].a));
        CK(cudaEventCreate(&dev[d].b));
    }
    std::printf("devices=2 sms=%d peer=%d/%d\n", sms, can01, can10);
    ceilings(sms, can01 && can10);
    ping(false);
    /* Peer access the driver reports but the platform drops (IOMMU translation) shows up
     * here; the collectives then only run on the host transport. */
    const bool p2p_ok = can01 && can10 && ping(true);
    std::vector<unsigned> nblks;
    for (int a = 1; a < argc; a++) nblks.push_back((unsigned)atoi(argv[a]));
    if (nblks.empty()) nblks = {(unsigned)sms, 32u, 16u, 8u, 4u, 2u, 1u};
    for (bool p2p : {false, true}) {
        if (p2p && !p2p_ok) continue;
        for (unsigned nblk : nblks) {
            collective(p2p, false, 2560, nblk);
            collective(p2p, false, 3840, nblk);
            collective(p2p, true, 4096u * 2560u, nblk);
            collective(p2p, true, 4096u * 3840u, nblk);
        }
    }
    return g_fail ? 1 : 0;
}

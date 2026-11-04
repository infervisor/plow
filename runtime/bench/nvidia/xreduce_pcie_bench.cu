/* xreduce_pcie_bench.cu — two-GPU all-reduce bodies (op_collective.cuh) against PCIe copy
 * ceilings. One process drives both devices; regions are either page-locked host memory mapped
 * into both (host), host plus a rank-local VRAM mirror (staged), or peer-accessible VRAM.
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
#include <algorithm>
#include <cmath>
#include <type_traits>
#include "sm120_common.cuh"
#include "op_norm.cuh"
#include "op_elementwise.cuh"
#include "op_gemm.cuh"

#include "dev_isa.h"
#include "op_collective.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)

constexpr int kIters = 40;

__host__ __device__ static float partial_value(unsigned rank, unsigned iteration, uint32_t e) {
    const uint32_t h = e * 37u + rank * 101u + iteration * 179u;
    return (int(h % 1021u) - 510) * (1.0f / 512.0f);
}

/* Fresh inputs expose publish/arrival races. A zero-length dependent collective after
 * each result retires all remote readers before the next publication reuses its host slot.
 * The grid barriers also retire local readers before the producer overwrites the mirror. */
template <bool F32>
__global__ void __launch_bounds__(256, 1)
k_loop(PlowProgram prog, PlowDevInst in, void* out, void* resid, void* out2, bool twoshot,
       int iters, uint16_t* history, unsigned long long* times) {
    void* tensors[] = {out, resid, out2};
    const auto grid = cooperative_groups::this_grid();
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    const uint32_t n = in.i[0], e0 = in.i[5];
    void* local = prog.peer_scratch[prog.n_gpu];
    using Partial = typename std::conditional<F32, float, __nv_bfloat16>::type;
    auto* partial = (Partial*)((char*)(local ? local : prog.peer_scratch[prog.rank]) + in.i[2]);
    for (int it = 0; it < iters; it++) {
        unsigned long long start = 0, collective_start = 0;
        if (tid == 0) start = xr_gtime();
        if (prog.rank == 1 && (it & 1) && tid == 0) {
            const auto delay = xr_gtime();
            while (xr_gtime() - delay < 5000ull) {}
        }
        for (uint32_t e = tid; e < n; e += stride) {
            const float v = partial_value(prog.rank, it, e0 + e);
            if constexpr (F32) partial[e0 + e] = v;
            else {
                if (local) partial[e0 + e] = __float2bfloat16_rn(v);
                else xr_st(partial + e0 + e, v);
            }
        }
        grid.sync();
        if (tid == 0) collective_start = xr_gtime();
        in.i[3] = 1u + 3u * it;
        in.i[4] = F32 ? 0u : 2u + 3u * it;
        if (twoshot) d_xreduce_twoshot_nv(prog, &in, tensors, blockIdx.x, gridDim.x);
        else d_xreduce_nv(prog, &in, tensors, blockIdx.x, gridDim.x);
        grid.sync();
        if (tid == 0) {
            const auto end = xr_gtime();
            times[2 * it] = end - start;
            times[2 * it + 1] = end - collective_start;
        }
        for (uint32_t e = tid; e < n; e += stride)
            history[(size_t)it * n + e] = ((const uint16_t*)out)[e0 + e];
        grid.sync();
        PlowDevInst retire = in;
        retire.op = PLOW_DOP_XREDUCE;
        retire.i[0] = 0;
        retire.i[3] = 3u + 3u * it;
        retire.i[5] = 0;
        if constexpr (F32) retire.i[4] = 0;
        d_xreduce_nv(prog, &retire, tensors, blockIdx.x, gridDim.x);
        grid.sync();
    }
}

struct Projection {
    unsigned m, n, k, copies;
    bool prefill;
    __nv_bfloat16* input[2];
    __nv_bfloat16* weight[2];
    uint32_t* evict[2];
    size_t evict_elements;
    std::vector<double> reference[2];
    bool fixed_input = false;
};

__host__ __device__ static float projection_input(unsigned row, unsigned k, unsigned iteration) {
    return (int((row * 3u + (k % 8u) * 5u + iteration * 7u) % 8u) - 4) / 32.0f;
}
__host__ __device__ static float projection_weight(unsigned row, unsigned k, unsigned rank) {
    const float base = (int((row * 5u + (k % 8u) * 3u) % 8u) - 4) / 16.0f;
    return rank ? -base + 1.0f / 1024.0f : base;
}
__global__ void k_projection_weight(__nv_bfloat16* out, size_t count, unsigned n,
                                     unsigned k, unsigned rank) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = __float2bfloat16_rn(projection_weight((i / k) % n, i % k, rank));
}

template <bool F32, bool Prefill>
__global__ void __launch_bounds__(256, 1)
k_projection_loop(PlowProgram prog, PlowDevInst in, void* out, int iters,
                  uint16_t* history, unsigned long long* times, unsigned m,
                  unsigned n, unsigned k, unsigned copies, __nv_bfloat16* input,
                  const __nv_bfloat16* weights, uint32_t* partial_snapshot,
                  uint32_t* evict, size_t evict_elements, bool fixed_input) {
    extern __shared__ __nv_bfloat16 arena[];
    using Partial = typename std::conditional<F32, float, __nv_bfloat16>::type;
    const auto grid = cooperative_groups::this_grid();
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    void* local = prog.peer_scratch[prog.n_gpu];
    auto* partial = (Partial*)((char*)(local ? local : prog.peer_scratch[prog.rank]) + in.i[2]);
    void* tensors[] = {out, nullptr, nullptr};
    const uint32_t elements = m * n;
    for (int it = 0; it < iters; it++) {
        unsigned long long start = 0, block_start = 0, collective_start = 0;
        if (tid == 0) start = xr_gtime();
        for (size_t e = tid; e < evict_elements; e += stride) evict[e] += 1u;
        if (!fixed_input)
            for (size_t e = tid; e < (size_t)m * k; e += stride)
                input[e] = __float2bfloat16_rn(projection_input(e / k, e % k, it));
        grid.sync();
        if (tid == 0) block_start = xr_gtime();
        const auto* weight = weights + (size_t)(it % copies) * n * k;
        if constexpr (Prefill) {
            if constexpr (F32) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
                d_gemm(partial, input, weight, m, n, k, 0, blockIdx.x, gridDim.x, arena);
#else
                __trap();
#endif
            } else d_gemm(partial, input, weight, m, n, k, 0, blockIdx.x, gridDim.x, arena);
        } else if constexpr (F32) {
#if PLOW_PACKET_REQUIRES_XR_PARTIAL_F32 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
            d_gemv_tp_f32(partial, input, weight, m, n, k, blockIdx.x, gridDim.x);
#else
            __trap();
#endif
        }
        else
            d_gemv(partial, input, weight, m, n, k, blockIdx.x, gridDim.x);
        grid.sync();
        if (partial_snapshot) {
            for (uint32_t e = tid; e < elements; e += stride) {
                if constexpr (F32) partial_snapshot[(size_t)it * elements + e] = __float_as_uint(partial[e]);
                else partial_snapshot[(size_t)it * elements + e] = __bfloat16_as_ushort(partial[e]);
            }
            grid.sync();
        }
        if (tid == 0) collective_start = xr_gtime();
        in.i[3] = 1u + 3u * it;
        in.i[4] = Prefill && !F32 ? 2u + 3u * it : 0u;
        if constexpr (Prefill && !F32)
            d_xreduce_twoshot_nv(prog, &in, tensors, blockIdx.x, gridDim.x);
        else
            d_xreduce_nv(prog, &in, tensors, blockIdx.x, gridDim.x);
        grid.sync();
        if (tid == 0) {
            const auto end = xr_gtime();
            times[3 * it] = end - start;
            times[3 * it + 1] = end - block_start;
            times[3 * it + 2] = end - collective_start;
        }
        for (uint32_t e = tid; e < elements; e += stride)
            history[(size_t)it * elements + e] = ((const uint16_t*)out)[e];
        grid.sync();
        PlowDevInst retire = in;
        retire.op = PLOW_DOP_XREDUCE;
        retire.i[0] = 0;
        retire.i[3] = 3u + 3u * it;
        retire.i[4] = 0;
        d_xreduce_nv(prog, &retire, tensors, blockIdx.x, gridDim.x);
        grid.sync();
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
        t = time_both([&](int d) { k_stream_ld<<<sms * 4, 256, 0, dev[d].s>>>(
            (const uint4*)(d ? d1 : d0), nv, sink[d]); });
        if (last) std::printf("CEIL local VRAM read v4       %7.1f GB/s per GPU\n", gbs(t, bytes));
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
static double g_relative_l2 = 0;

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

/* Verify every element of every iteration on both ranks.
 * Device timers exclude snapshot copies and the slot-retirement collective. */
static void collective(const char* transport, bool twoshot, uint32_t n, unsigned nblk,
                       uint32_t e0 = 0, bool f32 = false, const Projection* projection = nullptr) {
    const bool p2p = std::strcmp(transport, "p2p") == 0;
    const bool staged = std::strcmp(transport, "staged") == 0;
    const uint32_t slot = projection || f32 ? 128u : 0u;
    const size_t bytes = (size_t)n * 2, span_bytes = ((size_t)e0 + n) * 2;
    const size_t partial_bytes = ((size_t)e0 + n) * (f32 ? 4u : 2u);
    const size_t xoff = (slot + partial_bytes + 128u + 4095u) & ~size_t(4095);
    const int iterations = projection ? 8 : kIters;
    const unsigned time_fields = projection ? 3u : 2u;
    const size_t region = xoff + (size_t)(3 * kIters + 2) * PLOW_CTR_STRIDE * 4;
    char* host = nullptr;
    char* base[2];
    if (p2p) {
        for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaMalloc(&base[d], region)); }
    } else {
        CK(cudaHostAlloc(&host, 2 * region, cudaHostAllocPortable | cudaHostAllocMapped));
        base[0] = host;
        base[1] = host + region;
    }
    const size_t snapshots = iterations;
    std::vector<uint16_t> ref(n), got(n);
    void* outp[2];
    void* output_alloc[2];
    uint32_t* partial_snapshot[2] = {};
    void** peers[2];
    char* local[2] = {};
    uint16_t* history[2];
    unsigned long long* times[2];
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        CK(cudaMalloc(&output_alloc[d], span_bytes + (projection ? 256 : 0)));
        outp[d] = (char*)output_alloc[d] + (projection ? 128 : 0);
        if (projection) CK(cudaMalloc(&partial_snapshot[d], (size_t)n * 4 * (projection->fixed_input ? iterations : 1)));
        CK(cudaMalloc(&history[d], snapshots * bytes));
        CK(cudaMalloc(&times[d], time_fields * kIters * sizeof(unsigned long long)));
        if (staged) CK(cudaMalloc(&local[d], region));
        void* table[] = {base[0], base[1], local[d]};
        CK(cudaMalloc(&peers[d], sizeof(table)));
        CK(cudaMemcpy(peers[d], table, sizeof(table), cudaMemcpyHostToDevice));
    }
    auto reset = [&] {
        for (int d = 0; d < 2; d++) {
            CK(cudaSetDevice(d));
            CK(cudaMemsetAsync(base[d], 0xa5, region, dev[d].s));
            CK(cudaMemsetAsync(base[d] + xoff, 0, region - xoff, dev[d].s));
            if (staged) {
                CK(cudaMemsetAsync(local[d], 0xa5, region, dev[d].s));
                CK(cudaMemsetAsync(local[d] + xoff, 0, region - xoff, dev[d].s));
            }
            CK(cudaMemsetAsync(output_alloc[d], 0xff, span_bytes + (projection ? 256 : 0), dev[d].s));
            CK(cudaMemsetAsync(history[d], 0xff, snapshots * bytes, dev[d].s));
            CK(cudaStreamSynchronize(dev[d].s));
        }
    };
    PlowDevInst in{};
    for (auto& t : in.t) t = PLOW_TENSOR_NONE;
    in.t[0] = 0;
    in.op = twoshot ? PLOW_DOP_XREDUCE2 : PLOW_DOP_XREDUCE;
    in.i[0] = n;
    in.i[1] = 2;
    in.i[2] = slot;
    in.i[5] = e0;
    in.fj[2].u = 1;
    if (f32) in.fj[0].u = PLOW_XREDUCE_PARTIAL_F32_MARK;
    auto launch = [&](int d, int iters, bool audit = false) {
        PlowProgram prog{};
        prog.peer_scratch = peers[d];
        prog.n_gpu = 2;
        prog.rank = d;
        prog.xctr = (uint32_t*)(base[d] + xoff);
        void* o = outp[d];
        void* none = nullptr;
        if (!projection) {
            void* args[] = {&prog, &in, &o, &none, &none, &twoshot, &iters, &history[d], &times[d]};
            void* kernel = f32 ? (void*)k_loop<true> : (void*)k_loop<false>;
            CK(cudaLaunchCooperativeKernel(kernel, nblk, 256, args, 0, dev[d].s));
        } else {
            unsigned m = projection->m, columns = projection->n, k = projection->k;
            unsigned copies = projection->copies;
            auto* input = projection->input[d];
            auto* weights = projection->weight[d];
            uint32_t* snapshot = iters == 1 || audit ? partial_snapshot[d] : nullptr;
            uint32_t* evict = projection->evict[d];
            size_t evict_elements = projection->evict_elements;
            bool fixed_input = projection->fixed_input;
            void* args[] = {&prog, &in, &o, &iters, &history[d], &times[d],
                           &m, &columns, &k, &copies, &input, &weights, &snapshot, &evict, &evict_elements, &fixed_input};
            void* kernel = projection->prefill
                ? (f32 ? (void*)k_projection_loop<true, true> : (void*)k_projection_loop<false, true>)
                : (f32 ? (void*)k_projection_loop<true, false> : (void*)k_projection_loop<false, false>);
            const size_t smem = projection->prefill ? PGM_ARENA_BF16 * sizeof(__nv_bfloat16) : 0;
            CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
            int active = 0, sms = 0;
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 256, smem));
            CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d));
            if ((unsigned)(active * sms) < nblk) {
                std::fprintf(stderr, "projection grid is not cooperatively resident\n");
                std::exit(2);
            }
            if (iters == 1) {
                cudaFuncAttributes attr;
                CK(cudaFuncGetAttributes(&attr, kernel));
                std::printf("RESOURCE rank=%d f32=%d phase=%s regs=%d local=%zu shared=%zu active=%d\n",
                    d, f32, projection->prefill ? "prefill" : "decode", attr.numRegs,
                    attr.localSizeBytes, smem + attr.sharedSizeBytes, active);
            }
            CK(cudaLaunchCooperativeKernel(kernel, nblk, 256, args, smem, dev[d].s));
        }
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
    double max_relative_l2 = 0;
    auto check = [&](int iters, bool audit = false) {
        bool ok = status_ok("collective");
        for (int it = 0; it < iters; it++) {
            double oracle[2][8][8] = {};
            if (projection && !projection->fixed_input) {
                for (unsigned r = 0; r < 2; r++)
                    for (unsigned row = 0; row < 8; row++)
                        for (unsigned col = 0; col < 8; col++) {
                            double sum = 0;
                            for (unsigned kk = 0; kk < 8; kk++)
                                sum += (double)from_bf16(to_bf16(projection_input(row, kk, it)))
                                     * from_bf16(to_bf16(projection_weight(col, kk, r)));
                            oracle[r][row][col] = sum * (projection->k / 8u);
                        }
            }
            auto expected_partial = [&](unsigned r, uint32_t e) {
                return projection->fixed_input ? projection->reference[r][e]
                    : oracle[r][(e / projection->n) % 8u][(e % projection->n) % 8u];
            };
            for (uint32_t e = 0; e < n; e++) {
                float sum = 0.0f;
                for (unsigned r = 0; r < 2; r++) {
                    const float value = projection
                        ? (float)expected_partial(r, e)
                        : partial_value(r, it, e0 + e);
                    sum += f32 ? value : from_bf16(to_bf16(value));
                }
                ref[e] = to_bf16(sum);
            }
            if (projection && (iters == 1 || audit)) {
                for (int d = 0; d < 2; d++) {
                    CK(cudaSetDevice(d));
                    std::vector<uint32_t> partial(n);
                    CK(cudaMemcpy(partial.data(), partial_snapshot[d] + (size_t)it * n, (size_t)n * 4, cudaMemcpyDeviceToHost));
                    double error2 = 0, norm2 = 0;
                    for (uint32_t e = 0; e < n; e++) {
                        const double expected = expected_partial(d, e);
                        const float value = (float)expected;
                        if (projection->fixed_input) {
                            float actual;
                            if (f32) std::memcpy(&actual, &partial[e], 4);
                            else actual = from_bf16((uint16_t)partial[e]);
                            const double error = actual - expected;
                            error2 += error * error;
                            norm2 += expected * expected;
                            continue;
                        }
                        uint32_t want;
                        if (f32) std::memcpy(&want, &value, 4);
                        else want = to_bf16(value);
                        if (partial[e] != want) {
                            std::printf("FAIL partial rank=%d e=%u got=%08x want=%08x\n", d, e, partial[e], want);
                            ok = false; g_fail = true; break;
                        }
                    }
                    if (projection->fixed_input) {
                        const double rel = norm2 > 0 ? std::sqrt(error2 / norm2)
                            : (error2 == 0 ? 0 : INFINITY);
                        const double limit = f32 ? 1e-4 : 1.0 / 256.0;
                        const bool partial_ok = std::isfinite(rel) && rel <= limit;
                        std::printf("REAL_PARTIAL rank=%d f32=%d rel_l2=%.9g pass=%d\n", d, f32, rel, partial_ok);
                        if (!partial_ok) { ok = false; g_fail = true; }
                    }
                }
            }
            const unsigned snapshot = it;
            for (int d = 0; d < 2; d++) {
                CK(cudaSetDevice(d));
                CK(cudaMemcpy(got.data(), history[d] + (size_t)snapshot * n, bytes, cudaMemcpyDeviceToHost));
                if (projection) {
                    double squared_error = 0, reference_norm = 0;
                    for (uint32_t e = 0; e < n; e++) {
                        const double expected = expected_partial(0, e) + expected_partial(1, e);
                        const double error = from_bf16(got[e]) - expected;
                        squared_error += error * error;
                        reference_norm += expected * expected;
                    }
                    const double rel = reference_norm > 0 ? std::sqrt(squared_error / reference_norm)
                        : (squared_error == 0 ? 0 : INFINITY);
                    if (!std::isfinite(rel)) { ok = false; g_fail = true; }
                    max_relative_l2 = std::max(max_relative_l2, rel);
                    if (projection->fixed_input && d == 0) ref = got;
                }
                if (std::memcmp(got.data(), ref.data(), bytes) != 0) {
                    std::printf("FAIL bytes %s rank=%d it=%d n=%u e0=%u %s\n",
                                transport, d, it, n, e0, twoshot ? "twoshot" : "oneshot");
                    ok = false;
                    g_fail = true;
                }
            }
        }
        for (int d = 0; d < 2; d++) {
            CK(cudaSetDevice(d));
            auto guards = [&](const char* pointer) {
                std::vector<unsigned char> before(slot), after(xoff - slot - partial_bytes);
                if (slot) CK(cudaMemcpy(before.data(), pointer, slot, cudaMemcpyDefault));
                CK(cudaMemcpy(after.data(), pointer + slot + partial_bytes, after.size(), cudaMemcpyDefault));
                return std::all_of(before.begin(), before.end(), [](unsigned char x) { return x == 0xa5; })
                    && std::all_of(after.begin(), after.end(), [](unsigned char x) { return x == 0xa5; });
            };
            if (!guards(base[d]) || (staged && !guards(local[d]))) {
                std::printf("FAIL partial guards rank=%d\n", d); ok = false; g_fail = true;
            }
            if (projection) {
                unsigned char guard[256];
                CK(cudaMemcpy(guard, output_alloc[d], 128, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(guard + 128, (char*)outp[d] + span_bytes, 128, cudaMemcpyDeviceToHost));
                if (!std::all_of(guard, guard + 256, [](unsigned char x) { return x == 0xff; })) {
                    std::printf("FAIL output guards rank=%d\n", d); ok = false; g_fail = true;
                }
            }
            for (int it = 0; it < iters; it++) {
                for (unsigned phase = 1; phase <= 3; phase++) {
                    if (phase == 2 && !twoshot) continue;
                    const unsigned gate = phase + 3u * it;
                    uint32_t line[32];
                    CK(cudaMemcpy(line, base[d] + xoff + gate * PLOW_CTR_STRIDE * 4,
                                  sizeof(line), cudaMemcpyDefault));
                    if (line[0] != 1 || line[1] != 1
                        || line[kXrAggWord] != (!staged && phase == 2 ? nblk : 0u)
                        || line[kXrRelayWord] != 0u) {
                        std::printf("FAIL host gate rank=%d gate=%u arrivals=%u/%u agg=%u relay=%u\n",
                                    d, gate, line[0], line[1], line[kXrAggWord], line[kXrRelayWord]);
                        ok = false; g_fail = true;
                    }
                    if (staged) {
                        CK(cudaMemcpy(line, local[d] + xoff + gate * PLOW_CTR_STRIDE * 4,
                                      sizeof(line), cudaMemcpyDeviceToHost));
                        if (line[kXrAggWord] != nblk || line[kXrRelayWord] != 1u) {
                            std::printf("FAIL staged gate rank=%d gate=%u agg=%u relay=%u\n",
                                        d, gate, line[kXrAggWord], line[kXrRelayWord]);
                            ok = false; g_fail = true;
                        }
                    }
                }
            }
        }
        return ok;
    };
    auto cleanup = [&] {
        for (int d = 0; d < 2; d++) {
            CK(cudaSetDevice(d));
            CK(cudaFree(output_alloc[d]));
            if (projection) CK(cudaFree(partial_snapshot[d]));
            CK(cudaFree(peers[d]));
            CK(cudaFree(history[d]));
            CK(cudaFree(times[d]));
            if (staged) CK(cudaFree(local[d]));
            if (p2p) CK(cudaFree(base[d]));
        }
        if (host) CK(cudaFreeHost(host));
    };
    reset();
    time_both([&](int d) { launch(d, 1); });
    bool ok = check(1);
    if (!ok) { cleanup(); return; }
    if (projection && projection->fixed_input) {
        reset();
        time_both([&](int d) { launch(d, iterations, true); });
        if (!check(iterations, true)) { cleanup(); return; }
    }
    reset();
    const double ms = time_both([&](int d) { launch(d, iterations); });
    ok &= check(iterations);
    double inclusive = 0, collective_us = 0, block_us = 0, producer_us = 0;
    for (int d = 0; d < 2; d++) {
        unsigned long long elapsed[3 * kIters];
        CK(cudaSetDevice(d));
        CK(cudaMemcpy(elapsed, times[d], time_fields * kIters * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        unsigned long long with_producer = 0, without_producer = 0, body = 0;
        for (int it = 0; it < iterations; it++) {
            with_producer += elapsed[time_fields * it];
            without_producer += elapsed[time_fields * it + time_fields - 1];
            body += elapsed[time_fields * it + (projection ? 1 : 0)];
        }
        const double inc = with_producer / (1000.0 * iterations);
        const double coll = without_producer / (1000.0 * iterations);
        block_us = std::max(block_us, body / (1000.0 * iterations));
        producer_us = std::max(producer_us, (body - without_producer) / (1000.0 * iterations));
        inclusive = inc > inclusive ? inc : inclusive;
        collective_us = coll > collective_us ? coll : collective_us;
    }
    std::printf("XR %-6s %-7s n=%-9u e0=%u nblk=%-3u %9.2f us/coll %9.2f us/produce+coll "
                "%9.2f us/loop %7.2f GB/s alg %s\n",
                transport, twoshot ? "twoshot" : "oneshot", n, e0, nblk, collective_us,
                inclusive, ms * 1e3 / iterations, bytes / (collective_us * 1e3),
                ok ? "bytes=ok" : "INVALID");
    const double relative_l2 = max_relative_l2;
    g_relative_l2 = relative_l2;
    if (projection && f32 && !(relative_l2 <= 1.0 / 256.0)) { ok = false; g_fail = true; }
    if (projection)
        std::printf("PROJECTION_BLOCK transport=%s f32=%d phase=%s M=%u N=%u K=%u blocks=%u "
                    "block_us=%.6f input_and_block_us=%.6f collective_us=%.6f loop_us=%.6f producer_us=%.6f rel_l2=%.9g exact=%d guards=%d\n",
                    transport, f32, projection->prefill ? "prefill" : "decode", projection->m,
                    projection->n, projection->k, nblk, block_us, inclusive, collective_us,
                    ms * 1e3 / iterations, producer_us, relative_l2, ok, ok);
    else if (f32)
        std::printf("FP32_COLLECTIVE transport=%s n=%u blocks=%u exact=%d guards=%d\n",
                    transport, n, nblk, ok, ok);
    cleanup();
}

static void projection_cases(unsigned blocks, bool peer) {
    int l2 = 0;
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    if (l2 <= 0) { std::fprintf(stderr, "unknown L2 capacity\n"); std::exit(2); }
    auto run = [&](unsigned n, unsigned k, bool prefill, std::vector<unsigned> rows) {
        if (g_fail) return;
        Projection p{};
        p.n = n; p.k = k; p.prefill = prefill;
        p.evict_elements = (4ull * l2 + 3u) / 4u;
        const size_t weight_elements = (size_t)n * k;
        p.copies = (unsigned)std::max<size_t>(2, (4ull * l2 + weight_elements * 2 - 1) / (weight_elements * 2));
        const unsigned largest = *std::max_element(rows.begin(), rows.end());
        std::printf("PROJECTION_INPUT N=%u K=%u copies=%u weight_pool_bytes=%zu L2=%d\n",
                    n, k, p.copies, weight_elements * p.copies * 2, l2);
        for (int d = 0; d < 2; d++) {
            CK(cudaSetDevice(d));
            CK(cudaMalloc(&p.input[d], (size_t)largest * k * 2));
            CK(cudaMalloc(&p.weight[d], weight_elements * p.copies * 2));
            CK(cudaMalloc(&p.evict[d], p.evict_elements * 4));
            CK(cudaMemset(p.evict[d], 0, p.evict_elements * 4));
            k_projection_weight<<<blocks * 4, 256, 0, dev[d].s>>>(
                p.weight[d], weight_elements * p.copies, n, k, d);
            CK(cudaStreamSynchronize(dev[d].s));
        }
        for (unsigned m : rows) {
            if (g_fail) break;
            p.m = m;
            for (const char* transport : {"host", "staged", "p2p"}) {
                if (g_fail) break;
                if (std::strcmp(transport, "p2p") == 0 && !peer) continue;
                for (unsigned repeat = 1; repeat <= 2; repeat++) {
                    if (g_fail) break;
                    std::printf("PROJECTION_ABAB repeat=%u\n", repeat);
                    collective(transport, prefill, m * n, blocks, 0, false, &p);
                    if (g_fail) break;
                    collective(transport, false, m * n, blocks, 0, true, &p);
                }
            }
        }
        for (int d = 0; d < 2; d++) {
            CK(cudaSetDevice(d)); CK(cudaFree(p.input[d])); CK(cudaFree(p.weight[d])); CK(cudaFree(p.evict[d]));
        }
    };
    run(3840, 2048, true, {128, 512, 1024, 2048, 4096, 8192});
    run(3840, 4096, true, {128, 512, 1024, 2048, 4096, 8192});
    run(3840, 7680, true, {128, 512, 1024, 2048, 4096, 8192});
    run(3840, 7680, false, {1, 2, 4, 8, 16, 32});
    run(70, 40, true, {33});
    run(70, 40, false, {5});
}

static void real_projection(const char* path, unsigned blocks, bool peer) {
    FILE* file = std::fopen(path, "rb");
    if (!file) { std::perror(path); std::exit(2); }
    auto read = [&](void* dst, size_t bytes) {
        if (std::fread(dst, 1, bytes, file) != bytes) {
            std::fprintf(stderr, "short real projection fixture\n"); std::exit(2);
        }
    };
    uint32_t h[8];
    read(h, sizeof(h));
    if (h[0] != 0x58524631u || h[1] != 1 || !h[2] || h[2] > 128
        || h[3] != 3840 || h[5] > 1 || h[6] || h[7]
        || (h[5] ? (h[4] != 4096 && h[4] != 7680) : (h[4] != 7680 || h[2] > 32))) {
        std::fprintf(stderr, "unsupported real projection fixture\n"); std::exit(2);
    }
    Projection p{};
    p.m = h[2]; p.n = h[3]; p.k = h[4]; p.prefill = h[5]; p.fixed_input = true;
    int l2 = 0;
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    if (l2 <= 0) { std::fprintf(stderr, "unknown L2 capacity\n"); std::exit(2); }
    p.evict_elements = (4ull * l2 + 3u) / 4u;
    const size_t weight_elements = (size_t)p.n * p.k;
    p.copies = (unsigned)std::max<size_t>(2, (4ull * l2 + weight_elements * 2 - 1) / (weight_elements * 2));
    for (int d = 0; d < 2; d++) {
        std::vector<uint16_t> input((size_t)p.m * p.k), weight(weight_elements);
        read(input.data(), input.size() * 2);
        read(weight.data(), weight.size() * 2);
        p.reference[d].resize((size_t)p.m * p.n);
        read(p.reference[d].data(), p.reference[d].size() * sizeof(double));
        if (!std::all_of(p.reference[d].begin(), p.reference[d].end(),
                         [](double x) { return std::isfinite(x); })) {
            std::fprintf(stderr, "nonfinite real reference\n"); std::exit(2);
        }
        CK(cudaSetDevice(d));
        CK(cudaMalloc(&p.input[d], input.size() * 2));
        CK(cudaMalloc(&p.weight[d], weight_elements * p.copies * 2));
        CK(cudaMalloc(&p.evict[d], p.evict_elements * 4));
        CK(cudaMemset(p.evict[d], 0, p.evict_elements * 4));
        CK(cudaMemcpy(p.input[d], input.data(), input.size() * 2, cudaMemcpyHostToDevice));
        for (unsigned copy = 0; copy < p.copies; copy++)
            CK(cudaMemcpy(p.weight[d] + copy * weight_elements, weight.data(), weight.size() * 2, cudaMemcpyHostToDevice));
    }
    if (std::fgetc(file) != EOF) { std::fprintf(stderr, "oversized real fixture\n"); std::exit(2); }
    std::fclose(file);
    for (const char* transport : {"host", "staged", "p2p"}) {
        if (g_fail) break;
        if (std::strcmp(transport, "p2p") == 0 && !peer) continue;
        for (unsigned repeat = 1; repeat <= 2; repeat++) {
            std::printf("REAL_ABAB fixture=%s transport=%s repeat=%u\n", path, transport, repeat);
            collective(transport, p.prefill, p.m * p.n, blocks, 0, false, &p);
            const double control = g_relative_l2;
            if (g_fail) break;
            collective(transport, false, p.m * p.n, blocks, 0, true, &p);
            const bool ok = !g_fail && g_relative_l2 <= 1.0 / 256.0 && g_relative_l2 <= control;
            std::printf("REAL_PAIR control_l2=%.9g candidate_l2=%.9g pass=%d\n", control, g_relative_l2, ok);
            if (!ok) { g_fail = true; break; }
        }
    }
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d)); CK(cudaFree(p.input[d])); CK(cudaFree(p.weight[d])); CK(cudaFree(p.evict[d]));
    }
}

int main(int argc, char** argv) {
    bool fp32_partials = false, projection = false;
    const char* real = nullptr;
    std::vector<unsigned> nblks;
    for (int a = 1; a < argc; a++) {
        if (std::strcmp(argv[a], "--fp32-partials") == 0) fp32_partials = true;
        else if (std::strcmp(argv[a], "--projection-block") == 0) projection = true;
        else if (std::strcmp(argv[a], "--projection-real") == 0) {
            if (real || a + 1 >= argc) { std::fprintf(stderr, "one fixture path required\n"); return 2; }
            real = argv[++a];
        }
        else {
            char* end = nullptr;
            const unsigned long value = std::strtoul(argv[a], &end, 10);
            if (!value || !end || *end || value > 4096) {
                std::fprintf(stderr, "invalid block count: %s\n", argv[a]); return 2;
            }
            nblks.push_back((unsigned)value);
        }
    }
    if ((unsigned)fp32_partials + (unsigned)projection + (unsigned)(real != nullptr) > 1) {
        std::fprintf(stderr, "run synthetic and projection suites separately\n"); return 2;
    }
#if !PLOW_PACKET_REQUIRES_XR_PARTIAL_F32
    if (fp32_partials || projection || real) {
        std::fprintf(stderr, "FP32 tests require PLOW_PACKET_REQUIRES_XR_PARTIAL_F32=1\n");
        return 2;
    }
#endif
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { std::fprintf(stderr, "need 2 devices\n"); return 2; }
    if (fp32_partials || projection || real) {
        for (int d = 0; d < 2; d++) {
            cudaDeviceProp prop;
            CK(cudaGetDeviceProperties(&prop, d));
            if (prop.major != 12 || prop.minor != 0) {
                std::fprintf(stderr, "FP32 partial block requires sm120 on both devices\n"); return 2;
            }
        }
    }
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
    if (projection || real) {
        if (nblks.size() > 1) { std::fprintf(stderr, "projection block takes one grid\n"); return 2; }
        if (real) real_projection(real, nblks.empty() ? (unsigned)sms : nblks[0], p2p_ok);
        else projection_cases(nblks.empty() ? (unsigned)sms : nblks[0], p2p_ok);
        return g_fail ? 1 : 0;
    }
    if (nblks.empty()) nblks = {(unsigned)sms, 32u, 16u, 8u, 4u, 2u, 1u};
    for (const char* transport : {"host", "staged", "p2p"}) {
        if (std::strcmp(transport, "p2p") == 0 && !p2p_ok) continue;
        for (unsigned nblk : nblks) {
            if (fp32_partials) {
                for (unsigned rows : {1u, 2u, 4u, 8u, 16u, 32u, 128u, 512u, 1024u, 2048u, 4096u, 8192u}) {
                    collective(transport, false, rows * 3840u, nblk, 0, true);
                    if (g_fail) return 1;
                }
                collective(transport, false, 2561, nblk, 0, true);
                if (g_fail) return 1;
                continue;
            }
            collective(transport, false, 2560, nblk);
            collective(transport, false, 3840, nblk);
            collective(transport, true, 4096u * 2560u, nblk);
            collective(transport, true, 4096u * 3840u, nblk);
            collective(transport, false, 2561, nblk);
            collective(transport, true, 2561, nblk, 5);
            collective(transport, true, 4096u * 2560u + 3u, nblk, 5);
        }
    }
    return g_fail ? 1 : 0;
}

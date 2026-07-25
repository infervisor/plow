// Perf A/B for the head-major prefix-cache design (plans/rtx-09-prefix-headmajor.md):
// does reading KV through a VMM-mapped virtually-contiguous view cost anything
// vs plain cudaMalloc — in particular when the 2 MiB physical granules are
// FRAGMENTED (created amid allocator churn, mapped in shuffled order), which
// is what a long-lived prefix cache looks like after eviction/re-admission?
//
// KV-streaming decode microbench with the same access shape as
// op_attention.cuh d_flash_decode (kv_l2.cu lineage): head-major KV
// [KVH][L][HD] bf16, each K/V row is 512 B (HD=256), each row crosses HBM
// once and feeds GF=4 query heads, warp-strided timesteps, split-K grid.
// L2 is evicted between steps by a 1 GiB weight stream (decode reality).
//
// Configs, identical kernel + identical layout, only the backing differs:
//   a) cudaMalloc                     (baseline)
//   b) VMM contiguous: granules created in order, mapped in order
//   c) VMM fragmented: granules created interleaved with decoy churn
//      (2x decoys, odd ones freed mid-create), then mapped in shuffled
//      order — physically scattered, virtually contiguous
//
// Build (off-GPU):  nvcc -arch=sm_120a -O2 -o vmm_kv_ab vmm_kv_ab.cu -ldl
// Run:              gpulease <tag> ./vmm_kv_ab
//
// ============================ RESULTS =====================================
// NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 580.82.07, CUDA 13.0.
// Three runs across two leases (45 timed iters each), deltas vs cudaMalloc:
//
//   ctx 16k  (64 MiB KV):  a) 0.128-0.129 ms ~522 GB/s
//                          b) VMM contiguous +1.1..+1.4%
//                          c) VMM fragmented +5.8..+6.7%
//   ctx 128k (512 MiB KV): a) 0.437 ms 1228 GB/s
//                          b) VMM contiguous -0.8..-4.9% (FASTER)
//                          c) VMM fragmented -8.6..-8.8% (FASTER, all runs)
//
// VERDICT: no TLB penalty where it matters. At 128k — the regime prefix
// sharing exists for — the VMM view is never slower, and the physically
// scattered mapping is reproducibly ~8.7% FASTER than cudaMalloc (physical
// interleave apparently spreads HBM channel load). At 16k the fragmented
// view costs up to +6.7% (~8 us/step) — bounded, and vanishes at long ctx.
// Outputs match baseline to relL2 ~2e-7 (atomicAdd ordering only).
//
// fault-safety (design 12-E): with stride=131072 and only 16384 rows/head
// mapped, the kernel ran clean and matched a dense same-stride reference —
// the unmapped VA tail of a max_ctx-strided window is never touched when
// the loop bound is respected. Negative control (L = mapped+1) faulted with
// cudaErrorIllegalAddress: an addressing bug is loud, not silent. This also
// covers the driver half of 12-D: per-seq windows separated by unmapped VA
// turn cross-slot overruns into faults by construction.
// ==========================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

// ---- dlopen shim (same pattern as crates/plowrt/src/device/cuda.rs) ------
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
    LOAD(cuGetErrorName); LOAD(cuMemGetAllocationGranularity);
    LOAD(cuMemAddressReserve); LOAD(cuMemAddressFree);
    LOAD(cuMemCreate); LOAD(cuMemRelease);
    LOAD(cuMemMap); LOAD(cuMemUnmap); LOAD(cuMemSetAccess);
#undef LOAD
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define DK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char* s = "?"; p_cuGetErrorName(r, &s); \
    printf("DRV ERR %s @%d: %s\n", #x, __LINE__, s); exit(1); } } while (0)

static CUmemGenericAllocationHandle mk(size_t sz) {
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = 0;
    CUmemGenericAllocationHandle h;
    DK(p_cuMemCreate(&h, sz, &p, 0));
    return h;
}

// ---- microbench geometry (31B head-major shape: 512 B rows) ---------------
static const int HD = 256; // head dim (bf16) -> 512 B per row per head
static const int KVH = 4;  // kv heads
static const int GF = 4;   // query heads sharing one KV row
static const int WARP = 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// d_flash_decode-shaped KV read: K,V laid out [KVH][L][HD] bf16; each lane
// owns 8 of the 256 dims (one uint4 = 16 B per row), each K/V row crosses
// HBM once and feeds all GF query heads; warp-strided timesteps, split-K.
// `stride` = rows per head slot (kv_stride). The design runs kv_stride =
// max_ctx with only `L` rows mapped — the loop bound must keep every access
// inside [0, L) so the unmapped tail VA is never touched.
__global__ void flash_decode(const __nv_bfloat16* __restrict__ Kc,
                             const __nv_bfloat16* __restrict__ Vc,
                             const float* __restrict__ Q,   // [KVH*GF, HD]
                             float* __restrict__ out,       // [KVH*GF, HD]
                             int L, int stride, int nsplit) {
    const int h = blockIdx.x / nsplit;
    const int sp = blockIdx.x % nsplit;
    const int lane = threadIdx.x % WARP;
    const int wid = threadIdx.x / WARP;
    const int nw = blockDim.x / WARP;

    float q[GF][8], accd[GF][8];
#pragma unroll
    for (int g = 0; g < GF; ++g)
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            q[g][d] = Q[(size_t)(h * GF + g) * HD + lane * 8 + d];
            accd[g][d] = 0.f;
        }

    const int per = (L + nsplit - 1) / nsplit;
    const int t0 = sp * per, t1 = min(L, t0 + per);
    for (int t = t0 + wid; t < t1; t += nw) {
        const uint4 kv = *(const uint4*)(Kc + ((size_t)h * stride + t) * HD + lane * 8);
        const uint4 vv = *(const uint4*)(Vc + ((size_t)h * stride + t) * HD + lane * 8);
        const __nv_bfloat16* kh = (const __nv_bfloat16*)&kv;
        const __nv_bfloat16* vh = (const __nv_bfloat16*)&vv;
        float kf[8], vf[8];
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            kf[d] = __bfloat162float(kh[d]);
            vf[d] = __bfloat162float(vh[d]);
        }
#pragma unroll
        for (int g = 0; g < GF; ++g) {
            float s = 0.f;
#pragma unroll
            for (int d = 0; d < 8; ++d) s += q[g][d] * kf[d];
            s = warp_sum(s);
            float w = __expf(s * 0.0625f - 8.f);
#pragma unroll
            for (int d = 0; d < 8; ++d) accd[g][d] += w * vf[d];
        }
    }
#pragma unroll
    for (int g = 0; g < GF; ++g)
#pragma unroll
        for (int d = 0; d < 8; ++d)
            atomicAdd(&out[(size_t)(h * GF + g) * HD + lane * 8 + d], accd[g][d]);
}

// L2 evictor: streams `nelem` bf16 of "weights" (what a decode step does
// between attention ops), so each timed decode reads KV from HBM cold.
__global__ void weight_stream(const __nv_bfloat16* W, float* sink, size_t nelem) {
    float a = 0.f;
    size_t stride = (size_t)gridDim.x * blockDim.x * 8;
    for (size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8; i < nelem; i += stride) {
        uint4 v = *(const uint4*)(W + i);
        const __nv_bfloat16* h = (const __nv_bfloat16*)&v;
#pragma unroll
        for (int j = 0; j < 8; j++) a += __bfloat162float(h[j]);
    }
    if (a == 1234.5678f) sink[0] = a;
}

struct Backing {
    __nv_bfloat16* base = nullptr;
    // teardown state
    bool vmm = false;
    CUdeviceptr va = 0;
    size_t vasz = 0;
    std::vector<CUmemGenericAllocationHandle> hs;
    void destroy() {
        if (vmm) {
            DK(p_cuMemUnmap(va, vasz));
            for (auto h : hs) DK(p_cuMemRelease(h));
            DK(p_cuMemAddressFree(va, vasz));
        } else {
            CK(cudaFree(base));
        }
    }
};

static Backing make_backing(int mode, size_t bytes, size_t G) {
    Backing b;
    if (mode == 0) { // cudaMalloc
        CK(cudaMalloc(&b.base, bytes));
        return b;
    }
    size_t n = (bytes + G - 1) / G;
    b.vmm = true;
    b.vasz = n * G;
    DK(p_cuMemAddressReserve(&b.va, b.vasz, 0, 0, 0));
    b.hs.resize(n);
    if (mode == 1) { // contiguous: create in order, map in order
        for (size_t i = 0; i < n; i++) b.hs[i] = mk(G);
        for (size_t i = 0; i < n; i++) DK(p_cuMemMap(b.va + i * G, G, 0, b.hs[i], 0));
    } else { // fragmented: decoy churn during creation, shuffled map order
        std::vector<CUmemGenericAllocationHandle> decoys;
        for (size_t i = 0; i < n; i++) {
            CUmemGenericAllocationHandle d0 = mk(G), d1 = mk(G);
            b.hs[i] = mk(G);
            DK(p_cuMemRelease(d0)); // free one decoy now -> holes
            decoys.push_back(d1);   // keep one -> interleaved occupancy
        }
        for (auto d : decoys) DK(p_cuMemRelease(d));
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(0xC0FFEE);
        std::shuffle(order.begin(), order.end(), rng);
        for (size_t i = 0; i < n; i++)
            DK(p_cuMemMap(b.va + i * G, G, 0, b.hs[order[i]], 0));
    }
    CUmemAccessDesc d = {};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = 0;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DK(p_cuMemSetAccess(b.va, b.vasz, &d, 1));
    b.base = (__nv_bfloat16*)b.va;
    return b;
}

int main() {
    load_driver();
    CK(cudaSetDevice(0));
    CK(cudaFree(0));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("# %s SMs=%d\n", prop.name, prop.multiProcessorCount);
    const int SMs = prop.multiProcessorCount;

    CUmemAllocationProp ap = {};
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id = 0;
    size_t G = 0;
    DK(p_cuMemGetAllocationGranularity(&G, &ap, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    // 1 GiB evictor
    size_t wbytes = 1ull << 30;
    __nv_bfloat16* dW;
    CK(cudaMalloc(&dW, wbytes));
    CK(cudaMemset(dW, 0x3c, wbytes));
    float* dsink;
    CK(cudaMalloc(&dsink, 4));

    float* dQ;
    CK(cudaMalloc(&dQ, (size_t)KVH * GF * HD * 4));
    {
        std::vector<float> hq((size_t)KVH * GF * HD);
        for (size_t i = 0; i < hq.size(); i++) hq[i] = (float)((int)(i * 29 % 97) - 48) / 48.f;
        CK(cudaMemcpy(dQ, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
    }
    float* dOut;
    CK(cudaMalloc(&dOut, (size_t)KVH * GF * HD * 4));

    const char* names[3] = {"a) cudaMalloc", "b) VMM contiguous", "c) VMM fragmented"};
    const int WARM = 5, ITERS = 45;
    printf("\n%-8s %-9s %-20s %10s %9s %8s\n", "ctxL", "KV MiB", "backing", "ms/step",
           "KV GB/s", "vs a");
    printf("--------------------------------------------------------------------\n");

    for (int L : {16384, 131072}) {
        size_t half = (size_t)KVH * L * HD * 2; // K bytes ([KVH][L][HD] bf16)
        size_t kvb = 2 * half;
        int nsplit = std::max(1, (SMs * 2) / KVH);
        int grid = KVH * nsplit, blk = 256;
        int gridW = SMs * 4;

        double base_ms = 0;
        std::vector<float> ref;
        for (int mode = 0; mode < 3; mode++) {
            Backing bk = make_backing(mode, kvb, G);
            CK(cudaMemset(bk.base, 0x3c, kvb));
            __nv_bfloat16* Kc = bk.base;
            __nv_bfloat16* Vc = bk.base + half / 2; // half bytes / 2B per bf16

            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            double sum = 0;
            for (int it = 0; it < WARM + ITERS; it++) {
                weight_stream<<<gridW, 256>>>(dW, dsink, wbytes / 2); // evict L2
                CK(cudaMemset(dOut, 0, (size_t)KVH * GF * HD * 4));
                CK(cudaEventRecord(e0));
                flash_decode<<<grid, blk>>>(Kc, Vc, dQ, dOut, L, L, nsplit);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                if (it >= WARM) sum += ms;
            }
            CK(cudaGetLastError());
            double ms = sum / ITERS;
            // output correctness vs baseline (same memset pattern -> identical math)
            std::vector<float> out((size_t)KVH * GF * HD);
            CK(cudaMemcpy(out.data(), dOut, out.size() * 4, cudaMemcpyDeviceToHost));
            double rd = 0;
            if (mode == 0) { ref = out; base_ms = ms; }
            else {
                double num = 0, den = 1e-30;
                for (size_t i = 0; i < out.size(); i++) {
                    num += (out[i] - ref[i]) * (out[i] - ref[i]);
                    den += ref[i] * ref[i];
                }
                rd = sqrt(num / den);
            }
            printf("%-8d %-9.0f %-20s %10.4f %9.0f %+7.1f%%  relL2 %.1e\n", L,
                   kvb / 1048576.0, names[mode], ms, kvb / ms / 1e6,
                   100.0 * (ms - base_ms) / base_ms, rd);
            CK(cudaEventDestroy(e0));
            CK(cudaEventDestroy(e1));
            bk.destroy();
        }
    }
    // ---- fault safety (design doc section 12-E): kv_stride = max_ctx with
    // only `len` rows mapped. The kernel loop bound is [0, L); the tail of
    // each head's max_ctx VA window stays UNMAPPED. If the kernel ever
    // touched kv >= len this faults (cudaErrorIllegalAddress) instead of
    // silently reading garbage — exactly the property the design wants.
    {
        const int MAXL = 131072, L = 16384;
        const size_t headb = (size_t)MAXL * HD * 2;      // 64 MiB head slot
        const size_t half = (size_t)KVH * headb;          // K half, 256 MiB VA
        const size_t mappedb = (size_t)L * HD * 2;        // 8 MiB mapped/head

        // dense cudaMalloc reference at the same stride
        __nv_bfloat16* ref;
        CK(cudaMalloc(&ref, 2 * half));
        CK(cudaMemset(ref, 0x3c, 2 * half));
        int nsplit = std::max(1, (SMs * 2) / KVH);
        CK(cudaMemset(dOut, 0, (size_t)KVH * GF * HD * 4));
        flash_decode<<<KVH * nsplit, 256>>>(ref, ref + half / 2, dQ, dOut, L, MAXL, nsplit);
        CK(cudaDeviceSynchronize());
        std::vector<float> rout((size_t)KVH * GF * HD);
        CK(cudaMemcpy(rout.data(), dOut, rout.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaFree(ref));

        // sparse VMM view: map ONLY the first L rows of each head's window
        CUdeviceptr va;
        DK(p_cuMemAddressReserve(&va, 2 * half, 0, 0, 0));
        std::vector<CUmemGenericAllocationHandle> hs;
        CUmemAccessDesc ad = {};
        ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ad.location.id = 0;
        ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        for (int t = 0; t < 2; t++)
            for (int h = 0; h < KVH; h++) {
                size_t off = (size_t)t * half + (size_t)h * headb;
                for (size_t o = 0; o < mappedb; o += G) {
                    CUmemGenericAllocationHandle hh = mk(G);
                    hs.push_back(hh);
                    DK(p_cuMemMap(va + off + o, G, 0, hh, 0));
                }
                DK(p_cuMemSetAccess(va + off, mappedb, &ad, 1));
                CK(cudaMemset((void*)(va + off), 0x3c, mappedb));
            }
        CK(cudaMemset(dOut, 0, (size_t)KVH * GF * HD * 4));
        flash_decode<<<KVH * nsplit, 256>>>((__nv_bfloat16*)va,
                                            (__nv_bfloat16*)(va + half), dQ, dOut,
                                            L, MAXL, nsplit);
        cudaError_t e = cudaDeviceSynchronize();
        std::vector<float> sout((size_t)KVH * GF * HD);
        if (e == cudaSuccess)
            CK(cudaMemcpy(sout.data(), dOut, sout.size() * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 1e-30;
        for (size_t i = 0; i < sout.size(); i++) {
            num += (sout[i] - rout[i]) * (sout[i] - rout[i]);
            den += rout[i] * rout[i];
        }
        printf("\nfault-safety: stride=%d, mapped rows=%d/head: %s, relL2 vs dense %.1e -> %s\n",
               MAXL, L, cudaGetErrorString(e), sqrt(num / den),
               (e == cudaSuccess && sqrt(num / den) < 1e-6) ? "SAFE" : "UNSAFE");

        // negative control LAST (corrupts the context if it faults, as it must):
        // read one row past the mapped extent — this SHOULD fault.
        flash_decode<<<KVH * nsplit, 256>>>((__nv_bfloat16*)va,
                                            (__nv_bfloat16*)(va + half), dQ, dOut,
                                            L + 1, MAXL, nsplit);
        e = cudaDeviceSynchronize();
        printf("negative control (read 1 row past mapped): %s (%s)\n",
               cudaGetErrorString(e),
               e != cudaSuccess ? "faults as expected — violations are detectable"
                                : "DID NOT FAULT — unmapped reads are silent?!");
    }

    printf("\nDONE\n");
    return 0;
}

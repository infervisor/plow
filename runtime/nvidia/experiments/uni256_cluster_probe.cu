// uni256_cluster_probe.cu — T28: cluster-pair TMA MULTICAST variant of the uniform m128n256
// fp8 body. Hypothesis: the uni256 body sits at the L2 SERVICE wall (48 KB/stage/SM ~ 11 TB/s
// aggregate); multicasting the 32 KB B tile to both CTAs of a 2-CTA cluster cuts L2 reads to
// (16+16+32)/2CTAs = 32 KB/stage/SM -> ~1.5x headroom.
//
// Pairing: cluster c owns m-tile pair (2q, 2q+1) x n-tile — rank r computes m-tile 2q+r, both
// share the SAME B (n-window), fetched ONCE by rank 0 with .multicast::cluster mask 0b11.
// bempty on rank 0 counts 4 arrivals (2 local warpgroup reps + 2 remote from rank 1 via
// mapa + mbarrier.arrive.cluster), so B is never overwritten while rank 1 still reads it.
//
// Correctness: C compared byte-for-byte against the non-cluster n256 body.
//
// Build:
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common \
//     -I runtime/nvidia -include cstdint runtime/nvidia/experiments/uni256_cluster_probe.cu \
//     -lcuda -o /tmp/uni256_cluster_probe
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define PLOW_NV_HOPPER 1
#define PLOW_NV_THREADS 256u
#define PLOW_NV_TMA_GEMM 1
#define PLOW_NV_W8A8 1
#define PGM90_FP8_PROMOTE 0
#define PGM90_FORK_GLU 0
#define PGM90_UNI_BN256 1
#define PGM90_TMA_STAGES 3
#define PGM_ARENA_BF16 (128 * 1024)
#define PLOW_ACT_SILU_ 0u
__device__ __forceinline__ float act_silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float act_gelu_tanh(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
#include "op_gemm_sm90.cuh"

using bf16 = __nv_bfloat16;

/* ---- cluster helpers -------------------------------------------------------------------- */
__device__ __forceinline__ unsigned clu_rank() {
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
/* Remote shared address of `local` in cluster CTA `rank`. */
__device__ __forceinline__ uint32_t clu_mapa(uint32_t local, unsigned rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local), "r"(rank));
    return r;
}
__device__ __forceinline__ void clu_arrive_remote(uint32_t remote_bar) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(remote_bar)
                 : "memory");
}
__device__ __forceinline__ void clu_sync() {
    asm volatile("barrier.cluster.arrive;\n\tbarrier.cluster.wait;\n" ::: "memory");
}
/* Multicast 2-D TMA: lands at the same smem offset in every CTA of `mask`; tx bytes count at
 * each receiving CTA's OWN mbarrier (same offset as `bar`). */
__device__ __forceinline__ void clu_tma2d_mc(uint32_t dst, const void* map, int c0, int c1,
                                             uint32_t bar, unsigned short mask) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(dst),
        "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask)
        : "memory");
}

/* ---- the cluster-pair kernel ------------------------------------------------------------- */
__global__ void __cluster_dims__(2, 1, 1)
    k_clu(bf16* __restrict__ C, const void* mapA, const void* mapB, const float* __restrict__ as,
          const float* __restrict__ ws, unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 arena_bf[];
    float* arena = (float*)arena_bf;
    constexpr int NS = PGM90_UNI256_NS;
    uint64_t* bfull = (uint64_t*)arena;
    uint64_t* bempty = bfull + NS;
    uint8_t* base = (uint8_t*)sm90_align1024(arena + PGM90_U256_MBAR_BF16);
    uint8_t* As = base;
    uint8_t* Bs = base + NS * PGM90_A8BUF;
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;
    const unsigned rank = clu_rank();

    if (tid < NS) {
        sm90_mbar_init(bfull + tid, 1);
        /* rank 0's bempty gates the shared-B reissue: 2 local + 2 remote arrivals. */
        sm90_mbar_init(bempty + tid, rank == 0 ? 4 : 2);
    }
    clu_sync(); /* both CTAs' mbars live before any cross-CTA traffic */

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_U256_BN - 1) / PGM90_U256_BN;
    const int tiles_m2 = (tiles_m + 1) / 2; /* m-tile PAIRS */
    const int npairs = tiles_m2 * tiles_n;
    const int ksteps = ((int)k + PGM90_BK8 - 1) / PGM90_BK8;
    const unsigned nclu = gridDim.x / 2;
    const unsigned cslice = blockIdx.x / 2;

    /* Remote handle on rank 0's bempty ring (rank 1 only uses it). */
    uint32_t bempty0_lo = clu_mapa(sm90_su32(bempty), 0);

    int ist = 0;
    auto issue = [&](int pt, int ks) {
        const int s = ist % NS;
        if (ist >= NS) sm90_mbar_wait(bempty + s, ((ist / NS) + 1) & 1);
        int tmi2, tni;
        sm90_tile_remap(pt, tiles_m2, tiles_n, &tmi2, &tni);
        const int tm = (tmi2 * 2 + (int)rank) * PGM90_BM;
        const int tn = tni * PGM90_U256_BN;
        sm90_mbar_expect(bfull + s, PGM90_U256_TXB);
        const uint32_t bar = sm90_su32(bfull + s);
        /* A: per-rank fetch (distinct m rows). */
        sm90_tma2d(sm90_su32(As + s * PGM90_A8BUF), mapA, ks * PGM90_BK8, tm, bar);
        /* B: rank 0 multicasts the shared n-window to BOTH CTAs. */
        if (rank == 0) {
            uint8_t* bs = Bs + s * PGM90_U256_BBUF;
            clu_tma2d_mc(sm90_su32(bs), mapB, ks * PGM90_BK8, tn, bar, 0x3);
            clu_tma2d_mc(sm90_su32(bs + 128 * PGM90_BK8), mapB, ks * PGM90_BK8, tn + 128, bar,
                         0x3);
        }
        ist++;
    };
    auto stage_at = [&](int i, int& pt, int& ks) {
        pt = (int)cslice + (i / ksteps) * (int)nclu;
        ks = i % ksteps;
    };
    const int total = ((npairs - (int)cslice) + (int)nclu - 1) / (int)nclu * ksteps;
    if (tid == 0) {
        for (int i = 0; i < NS - 1 && i < total; i++) {
            int tl, ks;
            stage_at(i, tl, ks);
            issue(tl, ks);
        }
    }

    int st = 0;
    for (int pt = (int)cslice; pt < npairs; pt += (int)nclu) {
        int tmi2, tni;
        sm90_tile_remap(pt, tiles_m2, tiles_n, &tmi2, &tni);
        const int tm = (tmi2 * 2 + (int)rank) * PGM90_BM;
        const int tn = tni * PGM90_U256_BN;
        float acc[2 * PGM90_NACC];
        int prev = -1;
        for (int ks = 0; ks < ksteps; ks++, st++) {
            const int s = st % NS;
            sm90_mbar_wait(bfull + s, (st / NS) & 1);
            const uint8_t* Ac = As + s * PGM90_A8BUF + wg * PGM90_MSLAB * PGM90_BK8;
            const uint8_t* Bc = Bs + s * PGM90_U256_BBUF;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < PGM90_KSUB8; sub++) {
                const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                wgmma_m64n256k32(acc, sm90_desc(Ac + sub * 32), sm90_desc(Bc + sub * 32), sd);
            }
            sm90_wg_commit();
            sm90_wg_wait<1>();
            if (prev >= 0 && (tid & 127) == 0) {
                sm90_mbar_arrive(bempty + prev); /* local */
                if (rank == 1)
                    clu_arrive_remote(bempty0_lo + (uint32_t)(prev * 8)); /* free shared B */
            }
            prev = s;
            if (tid == 0 && st + NS - 1 < total) {
                int tl, kk;
                stage_at(st + NS - 1, tl, kk);
                issue(tl, kk);
            }
        }
        sm90_wg_wait<0>();
        if (prev >= 0 && (tid & 127) == 0) {
            sm90_mbar_arrive(bempty + prev);
            if (rank == 1) clu_arrive_remote(bempty0_lo + (uint32_t)(prev * 8));
        }

        const int r0 = tm + wg * PGM90_MSLAB + wiw * 16 + (lane >> 2);
        const int c0 = tn + 2 * (lane & 3);
#pragma unroll
        for (int g = 0; g < PGM90_U256_BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = r0 + 8 * hi;
                if (rr >= (int)m) continue;
                const float a = as[rr];
                const int cc = c0 + 8 * g;
                if (cc + 1 < (int)n) {
                    *(__nv_bfloat162*)(C + (size_t)rr * n + cc) = __floats2bfloat162_rn(
                        acc[4 * g + 2 * hi + 0] * a * ws[cc],
                        acc[4 * g + 2 * hi + 1] * a * ws[cc + 1]);
                }
            }
    }
    /* Cluster-wide quiesce before mbar invalidation: a peer may still be arriving remotely. */
    clu_sync();
    if (tid < NS) {
        sm90_mbar_inval(bfull + tid);
        sm90_mbar_inval(bempty + tid);
    }
    __syncthreads();
}

__global__ __launch_bounds__(256, 1) void k_ref(bf16* C, const void* mA, const void* mB,
                                                const float* as, const float* ws, unsigned m,
                                                unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    d_gemm_w8a8_sm90_tma_uni256(C, mA, mB, as, ws, m, n, k, 0, blockIdx.x, gridDim.x, arena);
}

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__);                             \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#define CKD(x)                                                                                     \
    do {                                                                                           \
        CUresult e_ = (x);                                                                         \
        if (e_ != CUDA_SUCCESS) {                                                                  \
            printf("CU err %d @%d\n", (int)e_, __LINE__);                                          \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static void make_map8(CUtensorMap* mp, void* base, int rows, int K) {
    uint64_t gd[2] = {(uint64_t)K, (uint64_t)rows};
    uint64_t gs[1] = {(uint64_t)K};
    uint32_t bd[2] = {128u, 128u};
    uint32_t es[2] = {1, 1};
    memset(mp, 0, sizeof(*mp));
    CKD(cuTensorMapEncodeTiled(mp, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

static uint32_t xs = 0x1234567u;
static uint32_t xr() {
    xs ^= xs << 13;
    xs ^= xs >> 17;
    xs ^= xs << 5;
    return xs;
}

int main() {
    const int grid = 132;
    const unsigned SMEM = 200 * 1024;
    CK(cudaFuncSetAttribute(k_ref, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CK(cudaFuncSetAttribute(k_clu, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

    struct Shape {
        unsigned m, n, k;
        const char* what;
    } shapes[] = {
        {4096, 4096, 3840, "q_proj"},
        {4096, 3840, 4096, "o_proj"},
        {4096, 15360, 3840, "gate/up"},
        {4096, 3840, 15360, "down"},
    };
    printf("grid=%d cluster=2 smem=%u\n%-10s %10s %10s %8s\n", grid, SMEM, "shape", "n256",
           "clu-mc", "clu/n256");
    for (auto& s : shapes) {
        size_t am = (size_t)s.m * s.k, bm = (size_t)s.n * s.k, cm = (size_t)s.m * s.n;
        uint8_t *A, *B;
        bf16 *C0, *C1;
        float *as, *ws;
        CK(cudaMalloc(&A, am));
        CK(cudaMalloc(&B, bm));
        CK(cudaMalloc(&C0, cm * 2));
        CK(cudaMalloc(&C1, cm * 2));
        CK(cudaMalloc(&as, s.m * 4));
        CK(cudaMalloc(&ws, s.n * 4));
        std::vector<uint8_t> h(am > bm ? am : bm);
        for (auto& v : h) v = (uint8_t)(xr() & 0x3f); /* positive e4m3, small */
        CK(cudaMemcpy(A, h.data(), am, cudaMemcpyHostToDevice));
        for (auto& v : h) v = (uint8_t)(xr() & 0x3f);
        CK(cudaMemcpy(B, h.data(), bm, cudaMemcpyHostToDevice));
        std::vector<float> one(s.m > s.n ? s.m : s.n, 1.0f);
        CK(cudaMemcpy(as, one.data(), s.m * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ws, one.data(), s.n * 4, cudaMemcpyHostToDevice));
        CUtensorMap mA, mB;
        make_map8(&mA, A, s.m, s.k);
        make_map8(&mB, B, s.n, s.k);
        CUtensorMap *dA, *dB;
        CK(cudaMalloc(&dA, 256));
        dB = dA + 1;
        CK(cudaMemcpy(dA, &mA, 128, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, &mB, 128, cudaMemcpyHostToDevice));

        /* correctness first */
        CK(cudaMemset(C0, 0, cm * 2));
        CK(cudaMemset(C1, 0, cm * 2));
        k_ref<<<grid, 256, SMEM>>>(C0, dA, dB, as, ws, s.m, s.n, s.k);
        k_clu<<<grid, 256, SMEM>>>(C1, dA, dB, as, ws, s.m, s.n, s.k);
        CK(cudaDeviceSynchronize());
        std::vector<bf16> h0(cm), h1(cm);
        CK(cudaMemcpy(h0.data(), C0, cm * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h1.data(), C1, cm * 2, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < cm; i++)
            if (memcmp(&h0[i], &h1[i], 2)) bad++;
        auto bench = [&](bool clu) -> float {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            for (int w = 0; w < 3; w++) {
                if (clu)
                    k_clu<<<grid, 256, SMEM>>>(C1, dA, dB, as, ws, s.m, s.n, s.k);
                else
                    k_ref<<<grid, 256, SMEM>>>(C0, dA, dB, as, ws, s.m, s.n, s.k);
            }
            CK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            const int it = 10;
            for (int i = 0; i < it; i++) {
                if (clu)
                    k_clu<<<grid, 256, SMEM>>>(C1, dA, dB, as, ws, s.m, s.n, s.k);
                else
                    k_ref<<<grid, 256, SMEM>>>(C0, dA, dB, as, ws, s.m, s.n, s.k);
            }
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            CK(cudaGetLastError());
            return ms / it;
        };
        float a = 1e9f, b = 1e9f;
        for (int r = 0; r < 3; r++) {
            float x = bench(false), y = bench(true);
            if (x < a) a = x;
            if (y < b) b = y;
        }
        double fl = 2.0 * s.m * s.n * s.k;
        printf("%-10s %7.3f ms %7.3f ms %6.2fx  (%6.1f vs %6.1f TF/s)  mismatches=%zu\n", s.what,
               a, b, b / a, fl / (a * 1e9), fl / (b * 1e9), bad);
        cudaFree(A); cudaFree(B); cudaFree(C0); cudaFree(C1); cudaFree(as); cudaFree(ws);
        cudaFree(dA);
    }
    return 0;
}

/* px13_tma_stage_bench.cu — is TMA actually a cheaper OPERAND-STAGING path than `cp.async.cg`
 * on sm_120a, for the EXACT tile the w8a8 prefill GEMM stages?
 *
 * PX-9 attributed plow's 61-66%-of-peak w8a8 GEMM to the cp.async staging path by ELIMINATION
 * (mainloop 94.3% of ceiling, cuBLASLt 95-99% on the same shapes, stage depth worth 0%), and
 * named TMA as the fix. Before porting a TMA GEMM this file tests the premise directly, with
 * no mma in the way: stage the same [128][64] e4m3 A tile and [128][64] e4m3 B tile, into the
 * same 3-deep 49152 B ring, at the same 256 threads, and time it three ways.
 *
 * WHY THIS AND NOT THE FA RESULT. `PLOW_NV_FA_TMA` measured ~2x SLOWER on attention
 * (perf-data/px4-flash-streaming.md), but it is NOT the same instruction: it issues one 1-D
 * `cp.async.bulk` PER ROW from a SINGLE thread (32 serial copies per tile). Arm `bulk1d` below
 * reproduces that shape so the two results are on one axis; arm `tma2d` is the 2-D
 * `cp.async.bulk.tensor` a GEMM would actually use — one instruction stages the whole tile.
 *
 * ARMS (all stage the identical bytes into the identical smem ring):
 *   ldgsts  256 threads x 2 lines each, `cp.async.cg` 16 B  — AS SHIPPED (pgm_stage_a8/b8)
 *   bulk1d  tid 0 issues 128 x 64 B `cp.async.bulk` + mbarrier — the FA arm's shape
 *   tma2d   tid 0 issues ONE `cp.async.bulk.tensor.2d` per tile + mbarrier — the GEMM shape
 *
 * The instrument is SM cycles per staged K-tile (both operands), as PX-9: a clock that moves
 * with power draw cannot corrupt a ratio taken in cycles.
 *
 * BUILD (plain env — nix CPATH collides with the CUDA headers):
 *   perf-data/px13_build_tma.sh
 * RUN:
 *   perf-data/tools/gpulease px13tma /tmp/px13tma
 *
 * NOT a correctness test on its own: `verify` mode checks that all three arms land the SAME
 * bytes in the SAME smem slots, which is the property a staging swap has to have.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)
#define DK(x) do { CUresult r_=(x); if(r_!=CUDA_SUCCESS){ const char* s_=nullptr; \
    cuGetErrorString(r_,&s_); printf("CU %s @%d: %s\n",#x,__LINE__,s_?s_:"?"); exit(1);} } while(0)

/* The production tile, verbatim from op_gemm.cuh. */
static const int BM = 128, BN = 128, BK8 = 64, STAGES = 3;
static const int ABUF = BM * BK8;            /* 8192 B */
static const int BBUF = BN * BK8;            /* 8192 B */
static const int RING = STAGES * (ABUF + BBUF); /* 49152 B — the shipped w8a8 arena */
static const int THREADS = 256;

/* ---- staging primitives ---- */
__device__ __forceinline__ void cp_cg16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void mbar_init(void* bar, unsigned count) {
    unsigned s = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(s), "r"(count));
}
__device__ __forceinline__ void mbar_expect(void* bar, unsigned bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(s), "r"(bytes));
}
/* NOTE: the barrier is initialised with arrive-count 1 and `mbar_expect_tx` IS that arrival.
 * An earlier version also had the other 255 threads call `mbarrier.arrive` — the count then
 * over-completes, the phase flips while the bulk copies are still in flight, and the launch
 * faults with `unspecified launch failure`. Only the issuing thread arrives; everyone waits. */
__device__ __forceinline__ void mbar_wait(void* bar, unsigned parity) {
    unsigned s = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("{\n\t.reg .pred P%=;\n"
                 "LAB%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 P%=, [%0], %1;\n"
                 "\t@!P%= bra LAB%=;\n\t}\n" ::"r"(s), "r"(parity));
}
__device__ __forceinline__ void cp_bulk_1d(void* dst, const void* src, unsigned bytes, void* bar) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    unsigned b = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];\n" ::"r"(d), "l"(src), "r"(bytes), "r"(b) : "memory");
}
__device__ __forceinline__ void cp_bulk_tensor_2d(void* dst, const void* tmap, int x, int y,
                                                  void* bar) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    unsigned b = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];\n" ::"r"(d), "l"(tmap), "r"(x), "r"(y), "r"(b)
                 : "memory");
}

/* ---- the three staging arms, one K-tile of A and one of B ---- */
enum { MODE_LDGSTS = 0, MODE_BULK1D = 1, MODE_TMA2D = 2 };

template <int MODE>
__global__ void __launch_bounds__(THREADS, 1)
k_stage(unsigned long long* cyc, unsigned* sink, const uint8_t* __restrict__ A,
        const uint8_t* __restrict__ B, int K, int ksteps, int tm, int iters,
        const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB) {
    extern __shared__ __align__(128) uint8_t sm[];
    uint8_t* ring = sm;
    /* mbarriers live past the ring, 8-byte aligned; 2 per stage (A, B). */
    unsigned long long* mbar = (unsigned long long*)(sm + RING);
    const int tid = threadIdx.x;
    const int tn = (int)blockIdx.x * BN;

    if (MODE != MODE_LDGSTS && tid == 0)
        for (int s = 0; s < 2 * STAGES; s++) mbar_init(&mbar[s], 1);
    __syncthreads();
    unsigned phase[2 * STAGES] = {0, 0, 0, 0, 0, 0};

    auto stage = [&](int ks, int buf) {
        uint8_t* Ad = ring + buf * (ABUF + BBUF);
        uint8_t* Bd = Ad + ABUF;
        const int kb = ks * BK8;
        if (MODE == MODE_LDGSTS) {
            /* 512 lines per operand over 256 threads = 2 each — pgm_stage_a8/b8's shape. */
            for (int L = tid; L < BM * (BK8 / 16); L += THREADS) {
                const int row = L / (BK8 / 16), k16 = (L % (BK8 / 16)) * 16;
                cp_cg16(&Ad[row * BK8 + k16], A + (size_t)(tm + row) * K + kb + k16);
            }
            for (int L = tid; L < BN * (BK8 / 16); L += THREADS) {
                const int row = L / (BK8 / 16), k16 = (L % (BK8 / 16)) * 16;
                cp_cg16(&Bd[row * BK8 + k16], B + (size_t)(tn + row) * K + kb + k16);
            }
            cp_commit();
        } else if (MODE == MODE_BULK1D) {
            if (tid == 0) {
                mbar_expect(&mbar[2 * buf], ABUF);
                for (int r = 0; r < BM; r++)
                    cp_bulk_1d(&Ad[r * BK8], A + (size_t)(tm + r) * K + kb, BK8, &mbar[2 * buf]);
                mbar_expect(&mbar[2 * buf + 1], BBUF);
                for (int r = 0; r < BN; r++)
                    cp_bulk_1d(&Bd[r * BK8], B + (size_t)(tn + r) * K + kb, BK8, &mbar[2 * buf + 1]);
            }
        } else {
            if (tid == 0) {
                mbar_expect(&mbar[2 * buf], ABUF);
                cp_bulk_tensor_2d(Ad, &tmA, kb, tm, &mbar[2 * buf]);
                mbar_expect(&mbar[2 * buf + 1], BBUF);
                cp_bulk_tensor_2d(Bd, &tmB, kb, tn, &mbar[2 * buf + 1]);
            }
        }
    };
    auto wait = [&](int buf) {
        if (MODE == MODE_LDGSTS) { cp_wait<STAGES - 1>(); }
        else {
            mbar_wait(&mbar[2 * buf], phase[2 * buf] & 1u); phase[2 * buf]++;
            mbar_wait(&mbar[2 * buf + 1], phase[2 * buf + 1] & 1u); phase[2 * buf + 1]++;
        }
        __syncthreads();
    };

    __syncthreads();
    long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
#pragma unroll 1
        for (int s = 0; s < STAGES - 1; s++) stage(s, s);
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % STAGES);
            else if (MODE == MODE_LDGSTS) cp_commit();
            wait(ks % STAGES);
            /* Touch the landed tile so the staging cannot be dead-code eliminated. */
            if (ring[(ks % STAGES) * (ABUF + BBUF) + tid] == 0xffu) sink[0]++;
            __syncthreads();
        }
    }
    long long t1 = clock64();
    if (tid == 0) cyc[blockIdx.x] = (unsigned long long)(t1 - t0);
}

/* Dump the first `n` bytes of each staged tile so the three arms can be diffed. */
template <int MODE>
__global__ void __launch_bounds__(THREADS, 1)
k_verify(uint8_t* out, const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, int K,
         int tm, const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB) {
    extern __shared__ __align__(128) uint8_t sm[];
    unsigned long long* mbar = (unsigned long long*)(sm + RING);
    const int tid = threadIdx.x, tn = (int)blockIdx.x * BN;
    if (MODE != MODE_LDGSTS && tid == 0) { mbar_init(&mbar[0], 1); mbar_init(&mbar[1], 1); }
    __syncthreads();
    uint8_t* Ad = sm; uint8_t* Bd = sm + ABUF;
    if (MODE == MODE_LDGSTS) {
        for (int L = tid; L < BM * 4; L += THREADS)
            cp_cg16(&Ad[(L / 4) * BK8 + (L % 4) * 16], A + (size_t)(tm + L / 4) * K + (L % 4) * 16);
        for (int L = tid; L < BN * 4; L += THREADS)
            cp_cg16(&Bd[(L / 4) * BK8 + (L % 4) * 16], B + (size_t)(tn + L / 4) * K + (L % 4) * 16);
        cp_commit(); cp_wait<0>();
    } else if (MODE == MODE_BULK1D) {
        if (tid == 0) {
            mbar_expect(&mbar[0], ABUF + BBUF);
            for (int r = 0; r < BM; r++) cp_bulk_1d(&Ad[r * BK8], A + (size_t)(tm + r) * K, BK8, &mbar[0]);
            for (int r = 0; r < BN; r++) cp_bulk_1d(&Bd[r * BK8], B + (size_t)(tn + r) * K, BK8, &mbar[0]);
        }
        mbar_wait(&mbar[0], 0);
    } else {
        if (tid == 0) {
            mbar_expect(&mbar[0], ABUF + BBUF);
            cp_bulk_tensor_2d(Ad, &tmA, 0, tm, &mbar[0]);
            cp_bulk_tensor_2d(Bd, &tmB, 0, tn, &mbar[0]);
        }
        mbar_wait(&mbar[0], 0);
    }
    __syncthreads();
    for (int i = tid; i < ABUF + BBUF; i += THREADS) out[i] = sm[i];
}

static uint32_t rng = 12345u;
static uint32_t xr() { rng = rng * 1664525u + 1013904223u; return rng; }
static uint8_t* dev_bytes(size_t n) {
    std::vector<uint8_t> h(n);
    for (size_t i = 0; i < n; i++) h[i] = (uint8_t)(xr() & 0xffu);
    uint8_t* d; CK(cudaMalloc(&d, n)); CK(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice));
    return d;
}

static CUtensorMap make_map(const void* g, int rows, int K) {
    CUtensorMap tm; memset(&tm, 0, sizeof(tm));
    uint64_t gdim[2] = {(uint64_t)K, (uint64_t)rows};   /* dim0 = K, contiguous */
    uint64_t gstr[1] = {(uint64_t)K};                   /* bytes; must be a multiple of 16 */
    uint32_t bdim[2] = {(uint32_t)BK8, (uint32_t)BM};   /* the [128][64] tile */
    uint32_t estr[2] = {1, 1};
    DK(cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)g, gdim, gstr, bdim,
                              estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tm;
}

template <int MODE>
static double run(const char* label, int P, int G, const uint8_t* A, const uint8_t* B, int K,
                  int ksteps, const CUtensorMap& tmA, const CUtensorMap& tmB, double* out_cpt) {
    const size_t smem = RING + 2 * STAGES * sizeof(unsigned long long);
    CK(cudaFuncSetAttribute(k_stage<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    unsigned long long* cyc; CK(cudaMalloc(&cyc, sizeof(unsigned long long) * G));
    unsigned* sink; CK(cudaMalloc(&sink, 4)); CK(cudaMemset(sink, 0, 4));
    const int iters = 200;
    k_stage<MODE><<<G, THREADS, smem>>>(cyc, sink, A, B, K, ksteps, 0, 4, tmA, tmB);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    k_stage<MODE><<<G, THREADS, smem>>>(cyc, sink, A, B, K, ksteps, 0, iters, tmA, tmB);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); CK(cudaGetLastError());
    std::vector<unsigned long long> h(G);
    CK(cudaMemcpy(h.data(), cyc, sizeof(unsigned long long) * G, cudaMemcpyDeviceToHost));
    double mean = 0; for (int i = 0; i < G; i++) mean += (double)h[i];
    mean /= G;
    const double cpt = mean / ((double)iters * ksteps);        /* cycles per staged K-tile */
    const double ghz = mean / (ms * 1e-3) / 1e9;
    /* Both operands per K-tile = 16384 B; G blocks in flight. */
    const double gbs = 16384.0 * (double)iters * ksteps * G / (ms * 1e-3) / 1e9;
    printf("  %-10s %8.3f ms  %9.1f cyc/K-tile  %6.3f GHz  %8.1f GB/s\n", label, ms, cpt, ghz, gbs);
    if (out_cpt) *out_cpt = cpt;
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(cyc); cudaFree(sink);
    return cpt;
}

template <int MODE>
static void verify(std::vector<uint8_t>& out, const uint8_t* A, const uint8_t* B, int K,
                   const CUtensorMap& tmA, const CUtensorMap& tmB) {
    const size_t smem = RING + 2 * STAGES * sizeof(unsigned long long);
    CK(cudaFuncSetAttribute(k_verify<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    uint8_t* d; CK(cudaMalloc(&d, ABUF + BBUF));
    CK(cudaMemset(d, 0, ABUF + BBUF));
    k_verify<MODE><<<1, THREADS, smem>>>(d, A, B, K, 0, tmA, tmB);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("  MODE %d verify FAILED: %s\n", MODE, cudaGetErrorString(e)); exit(1); }
    CK(cudaGetLastError());
    out.resize(ABUF + BBUF);
    CK(cudaMemcpy(out.data(), d, ABUF + BBUF, cudaMemcpyDeviceToHost));
    cudaFree(d);
}

int main(int argc, char** argv) {
    CK(cudaFree(nullptr));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    const int P = pr.multiProcessorCount;
    /* gate|up's weight: N=15360, K=3840. A is the M=1024 activation plane (the real bucket). */
    const int K = 3840, N = 15360, M = 1024;
    const int ksteps = K / BK8;
    const int G = N / BN;                    /* one N-tile per block; 120 <= 170 SMs, 1 blk/SM */
    printf("# %s SMs=%d  tile [%d][%d] e4m3, ring %d B, %d thr, %d blocks, K=%d (%d K-tiles)\n",
           pr.name, P, BM, BK8, RING, THREADS, G, K, ksteps);

    uint8_t* A = dev_bytes((size_t)M * K);
    uint8_t* B = dev_bytes((size_t)N * K);
    CUtensorMap tmA = make_map(A, M, K), tmB = make_map(B, N, K);

    printf("[verify] all three arms must land identical bytes\n");
    std::vector<uint8_t> v0, v1, v2;
    verify<MODE_LDGSTS>(v0, A, B, K, tmA, tmB);
    verify<MODE_BULK1D>(v1, A, B, K, tmA, tmB);
    verify<MODE_TMA2D>(v2, A, B, K, tmA, tmB);
    size_t d1 = 0, d2 = 0;
    for (size_t i = 0; i < v0.size(); i++) { d1 += (v0[i] != v1[i]); d2 += (v0[i] != v2[i]); }
    printf("  bulk1d vs ldgsts: %zu/%zu differing bytes   %s\n", d1, v0.size(), d1 ? "FAIL" : "PASS");
    printf("  tma2d  vs ldgsts: %zu/%zu differing bytes   %s\n", d2, v0.size(), d2 ? "FAIL" : "PASS");

    printf("[stage] cycles per staged K-tile (A+B = 16384 B), no mma\n");
    double c0 = 0, c1 = 0, c2 = 0;
    run<MODE_LDGSTS>("ldgsts", P, G, A, B, K, ksteps, tmA, tmB, &c0);
    run<MODE_BULK1D>("bulk1d", P, G, A, B, K, ksteps, tmA, tmB, &c1);
    run<MODE_TMA2D> ("tma2d ", P, G, A, B, K, ksteps, tmA, tmB, &c2);
    printf("  -> bulk1d %.3fx ldgsts, tma2d %.3fx ldgsts (>1 = SLOWER)\n", c1 / c0, c2 / c0);
    (void)argc; (void)argv;
    return 0;
}

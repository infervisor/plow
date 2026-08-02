/* px9_gemm_body_bench.cu — WHY does plow's w8a8 prefill GEMM body stop at ~60% of the in-tree
 * fp8 peak?  PX-7 measured the ceiling (305/321/243/312 TFLOP/s at an oracle grid, u = 1.000,
 * L2-cold) and killed the occupancy branch.  This file ATTRIBUTES the remaining gap.
 *
 * px7_w8a8_ceiling_bench.cu is a campaign record and is left untouched (the convention PX-6 and
 * PX-7 both used).  Its oracle-grid + L2-cold protocol is reused VERBATIM in [gemm] below, so
 * the numbers are directly comparable.
 *
 * THE INSTRUMENT: everything in the ladder is reported in SM CLOCK CYCLES PER QMMA, not
 * TFLOP/s.  TFLOP/s hides a clock that moves with the power draw; cycles/QMMA does not.  A
 * register-only QMMA loop gives the hardware issue ceiling in the same unit, so every rung of
 * the ladder is a pure ratio against it and no "peak" number has to be trusted.
 *
 * [ladder]  1 block/SM, 256 thr, the REAL acc[2][8][4] register footprint, the REAL 2x8 mma
 *           block, the REAL swizzled fp8 smem tiles.  Rungs:
 *             0 mma only, operands resident in registers          -> hardware issue ceiling
 *             1 + frag loads from smem, AS SHIPPED (scalar u32)   -> LDS + address-math cost
 *             2 + frag loads, VECTORIZED uint2 (proposed)         -> the A/B for the fix
 *             3 rung 1 + __syncthreads per K-tile                 -> barrier cost
 *             4 rung 2 + __syncthreads per K-tile                 -> barrier cost, vectorized
 *             5 rung 2 with the swizzle REMOVED                   -> what the swizzle buys/costs
 *           Rung 0 also reports the achieved SM clock (clock64 vs cudaEvent), which is the only
 *           honest way to turn cycles/QMMA into TFLOP/s.
 *
 * [gemm]    the real d_gemm_w8a8 / d_gemm_glu_w8a8 at PX-7's oracle grid + L2-cold protocol.
 * [lt]      cuBLASLt fp8 (e4m3 TN, per-tensor scale) at the same shapes — the upper-bound
 *           reference.  If cuBLASLt also lands at ~60% the "peak" itself is the thing to doubt.
 *
 * BUILD (plain env — nix CPATH collides with the CUDA math headers):
 *   export PATH=/usr/local/cuda/bin:/usr/bin:/bin; unset CPATH LIBRARY_PATH LD_LIBRARY_PATH
 *   nvcc -arch=sm_120a -O3 -I runtime/common -I runtime/nvidia \
 *        perf-data/px9_gemm_body_bench.cu -o /tmp/px9 -lcublasLt
 * RUN (always under the lease):
 *   perf-data/harness/gpulease px9 /tmp/px9 [ladder|gemm|lt|all]
 *
 * NOT a correctness test — random operands, no reference.  The vectorized frag read proposed in
 * rung 2 is a pure address bijection over the SAME bytes (see px9_load_* below) and is gated by
 * experiments/fp8_verify.cu before it can ship.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "sm120_common.cuh"
#include "op_gemm.cuh"

#ifndef PX9_NO_LT
#include <cublasLt.h>
#endif

typedef __nv_bfloat16 bf16;

#ifndef PX9_COLD_MB
#define PX9_COLD_MB 700
#endif
static const int ITERS = 30, WARM = 8;

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

static uint32_t rng = 12345u;
static uint32_t xr() { rng = rng*1664525u + 1013904223u; return rng; }
static void* dev_bytes(size_t n) {
    std::vector<uint8_t> h(n);
    for (size_t i = 0; i < n; i++) h[i] = (uint8_t)(xr() & 0x6fu);
    void* d; CK(cudaMalloc(&d, n)); CK(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice));
    return d;
}
static float* dev_scales(size_t n) {
    std::vector<float> h(n, 1.0f/448.0f);
    float* d; CK(cudaMalloc(&d, n*sizeof(float)));
    CK(cudaMemcpy(d, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    return d;
}
static bf16* dev_bf16(size_t n) {
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; i++) h[i] = (uint16_t)(0x3c00u | (xr() & 0x03ffu));
    bf16* d; CK(cudaMalloc(&d, n*sizeof(bf16)));
    CK(cudaMemcpy(d, h.data(), n*sizeof(bf16), cudaMemcpyHostToDevice));
    return d;
}

/* ===================================================================================
 * PROPOSED BODY FIX — vectorize the fp8 fragment read.
 *
 * pgm_load_afrags_w8a8 reads af[mi][0] at byte kb and af[mi][2] at kb+4 (and the same pair on
 * the +8 row); pgm_load_bfrags_w8a8 reads bf[nj][0] at kb and bf[nj][1] at kb+4.  kb is always
 * a multiple of 8 and [kb, kb+8) never crosses a 16-byte line, so pgm_sw8 (which permutes whole
 * 16-byte lines) satisfies sw8(off+4) == sw8(off)+4 for every kb the readers use.  The compiler
 * cannot prove that through the XOR, so it emits TWO LDS.32 plus a second full swizzle address
 * chain where ONE LDS.64 would do.  Reading a uint2 collapses both.
 * Same bytes, same lanes, same mma -> bit-identical accumulation.
 * =================================================================================== */
__device__ __forceinline__ void px9_load_afrags_v2(unsigned (&af)[PGM_MFRAG][4],
                                                   const uint8_t* Ad8, int wm, int kf, int lane) {
    const int kb = kf + 8 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < PGM_MFRAG; mi++) {
        const int rlo = wm * PGM_WM + mi * 16 + (lane >> 2);
        const int rhi = rlo + 8;
        uint2 lo = *(const uint2*)(Ad8 + pgm_sw8(rlo * PGM_BK8 + kb));
        uint2 hi = *(const uint2*)(Ad8 + pgm_sw8(rhi * PGM_BK8 + kb));
        af[mi][0] = lo.x; af[mi][2] = lo.y;
        af[mi][1] = hi.x; af[mi][3] = hi.y;
    }
}
__device__ __forceinline__ void px9_load_bfrags_v2(unsigned (&bf)[PGM_NFRAG][2],
                                                   const uint8_t* Bd8, int wn, int kf, int lane) {
    const int kb = kf + 8 * (lane & 3);
#pragma unroll
    for (int nj = 0; nj < PGM_NFRAG; nj++) {
        const int col = wn * PGM_WN + nj * 8 + (lane >> 2);
        uint2 v = *(const uint2*)(Bd8 + pgm_sw8(col * PGM_BK8 + kb));
        bf[nj][0] = v.x; bf[nj][1] = v.y;
    }
}
/* PX-9 DEAD END, recorded because it produced an illegal-memory-access mid-run: hoisting the
 * swizzled fragment offsets out of the K loop and adding kf as an immediate is WRONG.
 * pgm_sw8's XOR mask sits on bits 4..6 and kf = 32 flips bit 5, so sw8(off + 32) == sw8(off) ^ 32,
 * NOT sw8(off) + 32 (verified: 2048 of 4096 (row, lane) cases mismatch). The +4 inside a 16-byte
 * line does commute, which is what the uint2 read above relies on; the +32 across k-subgroups does
 * not. A correct hoist needs TWO base sets, one per k-subgroup — and rung 1 vs rung 0 below shows
 * the whole address-math bill is 0.6%, so it would not be worth the registers anyway.
 */

/* Unswizzled twin of the vectorized reader — rung 5's control. */
__device__ __forceinline__ void px9_load_afrags_v2ns(unsigned (&af)[PGM_MFRAG][4],
                                                     const uint8_t* Ad8, int wm, int kf, int lane) {
    const int kb = kf + 8 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < PGM_MFRAG; mi++) {
        const int rlo = wm * PGM_WM + mi * 16 + (lane >> 2);
        uint2 lo = *(const uint2*)(Ad8 + rlo * PGM_BK8 + kb);
        uint2 hi = *(const uint2*)(Ad8 + (rlo + 8) * PGM_BK8 + kb);
        af[mi][0] = lo.x; af[mi][2] = lo.y;
        af[mi][1] = hi.x; af[mi][3] = hi.y;
    }
}
__device__ __forceinline__ void px9_load_bfrags_v2ns(unsigned (&bf)[PGM_NFRAG][2],
                                                     const uint8_t* Bd8, int wn, int kf, int lane) {
    const int kb = kf + 8 * (lane & 3);
#pragma unroll
    for (int nj = 0; nj < PGM_NFRAG; nj++) {
        const int col = wn * PGM_WN + nj * 8 + (lane >> 2);
        uint2 v = *(const uint2*)(Bd8 + col * PGM_BK8 + kb);
        bf[nj][0] = v.x; bf[nj][1] = v.y;
    }
}

/* ============================== [ladder] ============================== */
/* Two resident stage buffers per operand; `sel` (a kernel arg the compiler cannot fold) picks
 * one per iteration so the frag loads CANNOT be hoisted out of the timing loop. */
#define PX9_LA (PGM_BM * PGM_BK8)   /* 8192 B */
#define PX9_LB (PGM_BN * PGM_BK8)   /* 8192 B */
#define PX9_LSMEM (2 * PX9_LA + 2 * PX9_LB)

template <int RUNG>
__global__ void __launch_bounds__(256, 1)
k_ladder(float* sink, unsigned long long* cyc, int iters, unsigned sel) {
    extern __shared__ uint8_t lsm[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;
    for (int i = tid; i < PX9_LSMEM / 4; i += 256)
        ((unsigned*)lsm)[i] = 0x3a3a3a3au ^ (unsigned)(i * 2654435761u);
    __syncthreads();
    uint8_t* As = lsm;
    uint8_t* Bs = lsm + 2 * PX9_LA;

    float acc[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
    for (int i = 0; i < PGM_MFRAG; i++)
        for (int j = 0; j < PGM_NFRAG; j++)
            for (int e = 0; e < 4; e++) acc[i][j][e] = (float)(tid & 1) * 1e-30f;
    /* RUNG 0's operands live in registers for the whole loop (the issue ceiling). */
    unsigned raf[PGM_MFRAG][4], rbf[PGM_NFRAG][2];
#pragma unroll
    for (int i = 0; i < PGM_MFRAG; i++)
        for (int e = 0; e < 4; e++) raf[i][e] = 0x3a3a3a3au ^ (unsigned)(tid + i * 7 + e);
#pragma unroll
    for (int j = 0; j < PGM_NFRAG; j++)
        for (int e = 0; e < 2; e++) rbf[j][e] = 0x38383838u ^ (unsigned)(tid + j * 13 + e);

    __syncthreads();
    long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
        const int cb = (int)((unsigned)it & sel);   /* 0 or 1, opaque to the compiler */
#pragma unroll
        for (int kf = 0; kf < PGM_BK8; kf += 32) {
            unsigned af[PGM_MFRAG][4], bf[PGM_NFRAG][2];
            if constexpr (RUNG == 0) {
#pragma unroll
                for (int i = 0; i < PGM_MFRAG; i++)
                    for (int e = 0; e < 4; e++) af[i][e] = raf[i][e];
#pragma unroll
                for (int j = 0; j < PGM_NFRAG; j++)
                    for (int e = 0; e < 2; e++) bf[j][e] = rbf[j][e];
            } else if constexpr (RUNG == 1 || RUNG == 3) {
                pgm_load_afrags_w8a8(af, As + cb * PX9_LA, wm, kf, lane);
                pgm_load_bfrags_w8a8<PGM_WN, PGM_NFRAG>(bf, Bs + cb * PX9_LB, wn, kf, lane);
            } else if constexpr (RUNG == 2 || RUNG == 4) {
                px9_load_afrags_v2(af, As + cb * PX9_LA, wm, kf, lane);
                px9_load_bfrags_v2(bf, Bs + cb * PX9_LB, wn, kf, lane);
            } else {
                px9_load_afrags_v2ns(af, As + cb * PX9_LA, wm, kf, lane);
                px9_load_bfrags_v2ns(bf, Bs + cb * PX9_LB, wn, kf, lane);
            }
#pragma unroll
            for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                for (int nj = 0; nj < PGM_NFRAG; nj++)
                    pgm_mma_fp8_k32(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
        }
        if constexpr (RUNG == 3 || RUNG == 4) __syncthreads();
    }
    long long t1 = clock64();
    if (tid == 0) cyc[blockIdx.x] = (unsigned long long)(t1 - t0);
    float s = 0;
#pragma unroll
    for (int i = 0; i < PGM_MFRAG; i++)
        for (int j = 0; j < PGM_NFRAG; j++)
            for (int e = 0; e < 4; e++) s += acc[i][j][e];
    if (s == 12345.678f) *sink = s;
}

/* mma-per-K-tile of the ladder body: 2 k-subgroups x MFRAG x NFRAG. */
static const int PX9_MMA_PER_IT = 2 * PGM_MFRAG * PGM_NFRAG;   /* 32 */

template <int RUNG>
static void ladder(const char* label, int P, double* out_cpm, double* out_ghz) {
    float* sink; CK(cudaMalloc(&sink, 4));
    unsigned long long* cyc; CK(cudaMalloc(&cyc, sizeof(unsigned long long) * P));
    CK(cudaFuncSetAttribute(k_ladder<RUNG>, cudaFuncAttributeMaxDynamicSharedMemorySize, PX9_LSMEM));
    const int iters = 40000;
    k_ladder<RUNG><<<P, 256, PX9_LSMEM>>>(sink, cyc, 200, 1u);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    k_ladder<RUNG><<<P, 256, PX9_LSMEM>>>(sink, cyc, iters, 1u);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaGetLastError());
    std::vector<unsigned long long> h(P);
    CK(cudaMemcpy(h.data(), cyc, sizeof(unsigned long long) * P, cudaMemcpyDeviceToHost));
    double mean = 0; for (int i = 0; i < P; i++) mean += (double)h[i];
    mean /= P;
    /* cycles per QMMA per WARP (the loop body issues PX9_MMA_PER_IT mma per iteration). */
    const double cpm = mean / ((double)iters * PX9_MMA_PER_IT);
    const double ghz = mean / (ms * 1e-3) / 1e9;
    /* 8 warps issue concurrently on one SM -> SM-level FLOP/clk, then TFLOP/s at that clock. */
    const double flop_clk_sm = 8192.0 * 8.0 / cpm;
    const double tf = flop_clk_sm * P * ghz * 1e9 / 1e12;
    printf("  %-46s %8.3f ms  %8.2f cyc/QMMA  %7.3f GHz  %7.0f FLOP/clk/SM  %8.1f TFLOP/s\n",
           label, ms, cpm, ghz, flop_clk_sm, tf);
    if (out_cpm) *out_cpm = cpm;
    if (out_ghz) *out_ghz = ghz;
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(sink); cudaFree(cyc);
}

/* ============================== [gemm] ============================== */
__global__ void k_w8a8(bf16* C, const uint8_t* A, const uint8_t* B,
                       const float* as, const float* ws,
                       unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 sm[];
    d_gemm_w8a8(C, A, B, as, ws, m, n, k, 0, blockIdx.x, gridDim.x, sm);
}
__global__ void k_w8a8_glu(bf16* C, const uint8_t* A, const uint8_t* Wg, const uint8_t* Wu,
                           const float* as, const float* sg, const float* su,
                           unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 sm[];
    d_gemm_glu_w8a8(C, A, Wg, Wu, as, sg, su, m, n, k, 0, blockIdx.x, gridDim.x, sm);
}
__global__ void k_bf16(bf16* C, const bf16* A, const bf16* B,
                       unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 sm[];
    d_gemm(C, A, B, m, n, k, 0, blockIdx.x, gridDim.x, sm);
}

/* largest divisor of T that is <= P — zero wave quantization by construction (PX-6/PX-7). */
static int oracle_grid(unsigned T, int P) {
    for (int g = std::min<int>(P, (int)T); g >= 1; g--) if (T % (unsigned)g == 0) return g;
    return 1;
}

/* PX-13: the N-tile is per-opcode, so the oracle grid must be computed with the tile the body
 * ACTUALLY runs — PGM_GLU_BN for the GLU arm, PGM_BN for the plain projections. Using the plain
 * BN for a GLU shape would assert u=1.000 on a tile count the kernel never produces. */
static unsigned bn_of(int glu) { return glu ? (unsigned)PGM_GLU_BN : (unsigned)PGM_BN; }

struct Shape { const char* name; unsigned N, K; int glu; };
static const Shape SHAPES[] = {
    {"gate|up", 15360, 3840, 1},
    {"down",     3840, 15360, 0},
    {"q_full",   8192, 3840, 0},
    {"o_full",   3840, 8192, 0},
};
static const int NSHAPE = 4;

/* PX-13 OCCUPANCY CONTROL. In the megakernel the dynamic-smem arena is a UNION over every op
 * body and flash-prefill's 79360 B claim dominates the GEMM's, so the prefill object runs 1
 * block/SM no matter what the GEMM tile asks for. An isolated bench that requests only the
 * GEMM's own arena can therefore report an occupancy win the megakernel will never see.
 * PX13_PAD_SMEM=<bytes> pads BOTH w8a8 launches up to that claim so the arm is measured at the
 * occupancy it will actually run at. 0 = the isolated (unpadded) arena. */
static size_t pad_smem(size_t s) {
    static size_t pad = (size_t)-1;
    if (pad == (size_t)-1) { const char* e = getenv("PX13_PAD_SMEM"); pad = e ? (size_t)atoll(e) : 0; }
    return s > pad ? s : pad;
}

static void section_gemm(int P, int with_bf16) {
    const size_t smem   = (size_t)PGM_ARENA_BF16      * sizeof(bf16);
    const size_t smem8  = pad_smem((size_t)PGM_ARENA_W8A8      * sizeof(bf16));
    const size_t smem8g = pad_smem((size_t)PGM_ARENA_GLU_W8A8  * sizeof(bf16));
    CK(cudaFuncSetAttribute(k_w8a8,     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8));
    CK(cudaFuncSetAttribute(k_w8a8_glu, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8g));
    /* the bf16 twin's arena can exceed the 99 KiB opt-in cap at BN=256; it is only a control. */
    if (with_bf16) CK(cudaFuncSetAttribute(k_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           (int)smem));
    printf("%-9s %6s %6s %6s %5s %6s %8s %9s %9s %10s\n",
           "shape","M","T","G*","u","arm","ms","TFLOP/s","%503.8","cyc/QMMA");
    /* PX-13: M is the PREFILL CHUNK, and the deployed packet's largest prefill bucket is 1024,
     * not 8192. A tile choice tuned at M=8192 is tuned at a tile count 8x larger than the one
     * the runtime ever launches, so PX13_MS overrides the M list. */
    unsigned Ms[8] = {8192, 2048, 0, 0, 0, 0, 0, 0}; size_t nms = 2;
    if (const char* e = getenv("PX13_MS")) {
        nms = 0; const char* p = e;
        while (*p && nms < 8) { Ms[nms++] = (unsigned)strtoul(p, (char**)&p, 10); while (*p==','||*p==' ') p++; }
    }
    for (size_t mi = 0; mi < nms; mi++) {
        unsigned M = Ms[mi];
        for (int si = 0; si < NSHAPE; si++) {
            const Shape& s = SHAPES[si];
            unsigned bn = bn_of(s.glu);
            unsigned tm = (M + PGM_BM - 1)/PGM_BM, tn = (s.N + bn - 1)/bn;
            unsigned T = tm*tn;
            int G = oracle_grid(T, P);
            double u = (double)T / ((double)((T + G - 1)/G) * G);
            if (u < 0.9999) { printf("!! oracle grid failed for %s M=%u\n", s.name, M); exit(1); }

            size_t wn = (size_t)s.N * s.K;
            int nrep = (int)std::max<size_t>(2, ((size_t)PX9_COLD_MB<<20) /
                                                 std::max<size_t>(wn*(s.glu?2:1), 1));
            nrep = std::min(nrep, 16);
            if (const char* e = getenv("PX9_NREP")) nrep = std::max(1, atoi(e));
            std::vector<uint8_t*> Bg(nrep), Bu(s.glu?nrep:0);
            for (int r = 0; r < nrep; r++) {
                Bg[r] = (uint8_t*)dev_bytes(wn);
                if (s.glu) Bu[r] = (uint8_t*)dev_bytes(wn);
            }
            uint8_t* A8 = (uint8_t*)dev_bytes((size_t)M*s.K);
            float* as = dev_scales(M); float* ws = dev_scales(s.N); float* ws2 = dev_scales(s.N);
            bf16* C = nullptr; CK(cudaMalloc(&C, (size_t)M*s.N*sizeof(bf16)));

            auto run8 = [&](int it) {
                int r = it % nrep;
                if (s.glu) k_w8a8_glu<<<G,256,smem8g>>>(C,A8,Bg[r],Bu[r],as,ws,ws2,M,s.N,s.K);
                else       k_w8a8    <<<G,256,smem8 >>>(C,A8,Bg[r],as,ws,M,s.N,s.K);
            };
            for (int i = 0; i < WARM; i++) run8(i);
            CK(cudaDeviceSynchronize());
            cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0));
            for (int i = 0; i < ITERS; i++) run8(i);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms8 = 0; CK(cudaEventElapsedTime(&ms8,e0,e1)); ms8 /= ITERS;
            CK(cudaGetLastError());

            double msb = 0;
            if (with_bf16) {
                bf16* Bb = dev_bf16(wn); bf16* Ab = dev_bf16((size_t)M*s.K);
                for (int i = 0; i < WARM; i++) k_bf16<<<G,256,smem>>>(C,Ab,Bb,M,s.N,s.K);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0));
                for (int i = 0; i < ITERS; i++) k_bf16<<<G,256,smem>>>(C,Ab,Bb,M,s.N,s.K);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float t=0; CK(cudaEventElapsedTime(&t,e0,e1)); msb = t/ITERS;
                CK(cudaGetLastError());
                cudaFree(Bb); cudaFree(Ab);
            }
            /* FULL-GRID arm.  The oracle grid removes wave quantization but LEAVES SMs IDLE
             * whenever the largest divisor of T below P is well under P (q_full: G*=128 of 170
             * = 75% of the machine).  PX-7 quoted only the oracle-grid number, so its "% of
             * peak" is understated by exactly that idle fraction.  Timing the same shape at
             * grid = P measures the trade instead of assuming it. */
            double msP = 0;
            {
                cudaEvent_t f0,f1; CK(cudaEventCreate(&f0)); CK(cudaEventCreate(&f1));
                auto runP = [&](int it){ int r = it % nrep;
                    if (s.glu) k_w8a8_glu<<<P,256,smem8g>>>(C,A8,Bg[r],Bu[r],as,ws,ws2,M,s.N,s.K);
                    else       k_w8a8    <<<P,256,smem8 >>>(C,A8,Bg[r],as,ws,M,s.N,s.K); };
                for (int i = 0; i < WARM; i++) runP(i);
                CK(cudaDeviceSynchronize()); CK(cudaEventRecord(f0));
                for (int i = 0; i < ITERS; i++) runP(i);
                CK(cudaEventRecord(f1)); CK(cudaEventSynchronize(f1));
                float t=0; CK(cudaEventElapsedTime(&t,f0,f1)); msP = t/ITERS;
                CK(cudaGetLastError());
                cudaEventDestroy(f0); cudaEventDestroy(f1);
            }
            cudaEventDestroy(e0); cudaEventDestroy(e1);

            double fl  = 2.0*M*s.N*s.K*(s.glu?2.0:1.0);
            double tf8 = fl/(ms8*1e-3)/1e12;
            double tfP = fl/(msP*1e-3)/1e12;
            printf("%-9s %6u %6u %6d %5.3f %6s %8.4f %9.1f %8.1f%% %9.1f%%SM\n",
                   s.name, M, T, G, u, s.glu?"glu8":"w8a8", ms8, tf8, 100.0*tf8/503.8,
                   100.0*(double)G/P);
            printf("%-9s %6u %6u %6d %5s %6s %8.4f %9.1f %8.1f%% %9.3fx vs G*\n",
                   "", M, T, P, "-", "fullG", msP, tfP, 100.0*tfP/503.8, tfP/tf8);
            if (with_bf16) {
                double tfb = 2.0*M*s.N*s.K/(msb*1e-3)/1e12;
                printf("%-9s %6u %6u %6d %5.3f %6s %8.4f %9.1f %8.1f%% %10s\n",
                       "", M, T, G, u, "bf16", msb, tfb, 100.0*tfb/209.5, "-");
            }
            for (int r = 0; r < nrep; r++) { cudaFree(Bg[r]); if (s.glu) cudaFree(Bu[r]); }
            cudaFree(A8); cudaFree(as); cudaFree(ws); cudaFree(ws2); cudaFree(C);
        }
    }
}

/* [one] — a SINGLE launch of one shape, for `ncu`.  Same operands and same grid as [gemm];
 * L2 is not cycled (one launch has nothing to cycle) so DRAM counters here are the compulsory
 * + capacity traffic of one cold-ish pass, which is exactly what the L2 question needs. */
static void section_one(int P, int si, unsigned M, int full_grid) {
    const Shape& s = SHAPES[si];
    const size_t smem8  = (size_t)PGM_ARENA_W8A8      * sizeof(bf16);
    const size_t smem8g = (size_t)PGM_ARENA_GLU_W8A8  * sizeof(bf16);
    CK(cudaFuncSetAttribute(k_w8a8,     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8));
    CK(cudaFuncSetAttribute(k_w8a8_glu, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8g));
    unsigned bn = bn_of(s.glu);
    unsigned tm = (M + PGM_BM - 1)/PGM_BM, tn = (s.N + bn - 1)/bn, T = tm*tn;
    int G = full_grid ? P : oracle_grid(T, P);
    size_t wn = (size_t)s.N * s.K;
    uint8_t* Bg = (uint8_t*)dev_bytes(wn);
    uint8_t* Bu = s.glu ? (uint8_t*)dev_bytes(wn) : nullptr;
    uint8_t* A8 = (uint8_t*)dev_bytes((size_t)M*s.K);
    float* as = dev_scales(M); float* ws = dev_scales(s.N); float* ws2 = dev_scales(s.N);
    bf16* C = nullptr; CK(cudaMalloc(&C, (size_t)M*s.N*sizeof(bf16)));
    printf("# [one] %s M=%u N=%u K=%u T=%u G=%d (%s) SMs=%d\n",
           s.name, M, s.N, s.K, T, G, full_grid ? "full grid" : "oracle grid", P);
    if (s.glu) k_w8a8_glu<<<G,256,smem8g>>>(C,A8,Bg,Bu,as,ws,ws2,M,s.N,s.K);
    else       k_w8a8    <<<G,256,smem8 >>>(C,A8,Bg,as,ws,M,s.N,s.K);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    cudaFree(Bg); if (Bu) cudaFree(Bu);
    cudaFree(A8); cudaFree(as); cudaFree(ws); cudaFree(ws2); cudaFree(C);
}

/* ============================== [hash] ==============================
 * PX-13 NUMERICS GATE. The per-opcode N-tile changes only how the output plane is CUT into
 * tiles; every output element still accumulates over the whole of K in the same k32 order, so
 * the result must be BIT-IDENTICAL, not merely close. This section runs each shape once on
 * deterministic operands (fixed seed, fixed allocation order) and prints an FNV-1a hash of the
 * whole C plane. Two binaries agree bit-for-bit iff their hashes agree. Shapes are cut down in
 * M so the host-side hash is cheap, but K is the real K (60 K-tiles) so the whole mainloop and
 * the whole epilogue are covered. */
static unsigned long long fnv(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static void section_hash(int P) {
    const size_t smem8  = (size_t)PGM_ARENA_W8A8      * sizeof(bf16);
    const size_t smem8g = (size_t)PGM_ARENA_GLU_W8A8  * sizeof(bf16);
    CK(cudaFuncSetAttribute(k_w8a8,     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8));
    CK(cudaFuncSetAttribute(k_w8a8_glu, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8g));
    const unsigned M = 512;
    for (int si = 0; si < NSHAPE; si++) {
        const Shape& s = SHAPES[si];
        rng = 987654321u;                       /* same operand bytes in every binary */
        uint8_t* A8 = (uint8_t*)dev_bytes((size_t)M * s.K);
        uint8_t* Bg = (uint8_t*)dev_bytes((size_t)s.N * s.K);
        uint8_t* Bu = s.glu ? (uint8_t*)dev_bytes((size_t)s.N * s.K) : nullptr;
        float* as = dev_scales(M); float* ws = dev_scales(s.N); float* ws2 = dev_scales(s.N);
        bf16* C = nullptr; CK(cudaMalloc(&C, (size_t)M * s.N * sizeof(bf16)));
        CK(cudaMemset(C, 0, (size_t)M * s.N * sizeof(bf16)));
        if (s.glu) k_w8a8_glu<<<P,256,smem8g>>>(C,A8,Bg,Bu,as,ws,ws2,M,s.N,s.K);
        else       k_w8a8    <<<P,256,smem8 >>>(C,A8,Bg,as,ws,M,s.N,s.K);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        std::vector<bf16> h((size_t)M * s.N);
        CK(cudaMemcpy(h.data(), C, h.size()*sizeof(bf16), cudaMemcpyDeviceToHost));
        printf("%-9s M=%u N=%u K=%u  %s  FNV=%016llx\n", s.name, M, s.N, s.K,
               s.glu ? "glu8" : "w8a8", fnv(h.data(), h.size()*sizeof(bf16)));
        cudaFree(A8); cudaFree(Bg); if (Bu) cudaFree(Bu);
        cudaFree(as); cudaFree(ws); cudaFree(ws2); cudaFree(C);
    }
}

/* ============================== [lt] ============================== */
#ifndef PX9_NO_LT
#define LTK(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
    printf("cublasLt %s @%d: %d\n",#x,__LINE__,(int)s_); return; } } while(0)

/* cuBLASLt fp8: D[m,n] = alpha * A[m,k] . B[n,k]^T.  cuBLASLt is COLUMN-major, so we compute
 * D^T = B . A^T with B as "A" (op=T) and A as "B" (op=N) — the standard TN fp8 recipe: the fp8
 * matmul supports opA=T, opB=N only.  Both operands e4m3, f32 compute, bf16 out.  Per-tensor
 * scales (cuBLASLt has no per-row scale) — this is a SPEED reference only, not a plow twin. */
static void lt_shape(cublasLtHandle_t lt, void* ws, size_t wsz, const char* name,
                     int M, int N, int K, int glu) {
    int nrep = (int)std::max<size_t>(2, ((size_t)PX9_COLD_MB<<20) / std::max<size_t>((size_t)N*K, 1));
    nrep = std::min(nrep, 16);
    if (const char* e = getenv("PX9_NREP")) nrep = std::max(1, atoi(e));
    const uint8_t *A8 = (const uint8_t*)dev_bytes((size_t)M*K);
    std::vector<const uint8_t*> B8v(nrep);
    for (int r = 0; r < nrep; r++) B8v[r] = (const uint8_t*)dev_bytes((size_t)N*K);
    bf16* D = nullptr; CK(cudaMalloc(&D, (size_t)M*N*sizeof(bf16)));
    float *sa, *sb, *sd;
    CK(cudaMalloc(&sa, 4)); CK(cudaMalloc(&sb, 4)); CK(cudaMalloc(&sd, 4));
    float one = 1.f/448.f; CK(cudaMemcpy(sa,&one,4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sb,&one,4,cudaMemcpyHostToDevice));
    float dsc = 1.f; CK(cudaMemcpy(sd,&dsc,4,cudaMemcpyHostToDevice));

    cublasLtMatmulDesc_t op = nullptr;
    LTK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
    LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
    LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sb, sizeof(sb)));
    LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sa, sizeof(sa)));

    /* column-major: Alt = B (K x N, ld=K, op=T) ; Blt = A (K x M, ld=K, op=N) ; D (N x M, ld=N) */
    cublasLtMatrixLayout_t la=nullptr, lb=nullptr, ld=nullptr;
    LTK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, N, K));
    LTK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, M, K));
    LTK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF,    N, M, N));

    cublasLtMatmulPreference_t pref = nullptr;
    LTK(cublasLtMatmulPreferenceCreate(&pref));
    LTK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &wsz, sizeof(wsz)));
    cublasLtMatmulHeuristicResult_t heur[1]; int nheur = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 1, heur, &nheur);
    if (hs != CUBLAS_STATUS_SUCCESS || nheur == 0) {
        printf("%-9s %6d  cuBLASLt: NO ALGO (status %d, n=%d)\n", name, M, (int)hs, nheur);
        return;
    }
    float alpha = 1.f, beta = 0.f;
    auto run = [&](int it) {
        return cublasLtMatmul(lt, op, &alpha, B8v[it % nrep], la, A8, lb, &beta, D, ld, D, ld,
                              &heur[0].algo, ws, wsz, 0);
    };
    for (int i = 0; i < WARM; i++) if (run(i) != CUBLAS_STATUS_SUCCESS) {
        printf("%-9s %6d  cuBLASLt: MATMUL FAILED\n", name, M); return; }
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < ITERS; i++) run(i);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms /= ITERS;
    CK(cudaGetLastError());
    double fl = 2.0*(double)M*N*K;
    double tf = fl/(ms*1e-3)/1e12;
    printf("%-9s %6d %6d %6d %8.4f ms %9.1f TFLOP/s %8.1f%% of 503.8  nrep=%d%s\n",
           name, M, N, K, ms, tf, 100.0*tf/503.8, nrep, glu ? "  (x2 for the gate|up pair)" : "");
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(op);
    cudaFree((void*)A8); for (int r = 0; r < nrep; r++) cudaFree((void*)B8v[r]); cudaFree(D); cudaFree(sa); cudaFree(sb); cudaFree(sd);
}

static void section_lt() {
    cublasLtHandle_t lt = nullptr;
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) { printf("cublasLtCreate FAILED\n"); return; }
    const size_t wsz = 64u<<20; void* ws = nullptr; CK(cudaMalloc(&ws, wsz));
    printf("# cuBLASLt fp8 e4m3 TN, per-tensor scale, bf16 out. Weight replication + cycling is\n"
           "# the SAME L2-cold protocol plow's arm uses (PX9_NREP=1 forces the warm variant).\n"
           "# cuBLASLt still picks its own grid, so this is an upper bound, not a twin.\n");
    const unsigned Ms[] = {8192, 2048};
    for (size_t mi = 0; mi < 2; mi++)
        for (int si = 0; si < NSHAPE; si++)
            lt_shape(lt, ws, wsz, SHAPES[si].name, (int)Ms[mi],
                     (int)SHAPES[si].N, (int)SHAPES[si].K, SHAPES[si].glu);
    cudaFree(ws); cublasLtDestroy(lt);
}
#else
static void section_lt() { printf("# cuBLASLt section compiled out (PX9_NO_LT)\n"); }
#endif

int main(int argc, char** argv) {
    const char* sec = (argc > 1) ? argv[1] : "all";
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    const int P = pr.multiProcessorCount;
    int o8 = 0, og = 0;
    CK(cudaFuncSetAttribute(k_w8a8, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)((size_t)PGM_ARENA_W8A8*sizeof(bf16))));
    CK(cudaFuncSetAttribute(k_w8a8_glu, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)((size_t)PGM_ARENA_GLU_W8A8*sizeof(bf16))));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o8, k_w8a8, 256,
                                                     (size_t)PGM_ARENA_W8A8*sizeof(bf16)));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&og, k_w8a8_glu, 256,
                                                     (size_t)PGM_ARENA_GLU_W8A8*sizeof(bf16)));
    printf("# %s  SMs=%d  L2=%.0f MiB  occ w8a8=%d glu=%d\n",
           pr.name, P, pr.l2CacheSize/1048576.0, o8, og);
    printf("# BM=%d BN=%d GLU_BN=%d BK8=%d STAGES=%d GLU_STAGES=%d  MFRAG=%d NFRAG=%d GLU_NFRAG=%d  sw8=%s\n",
           PGM_BM, PGM_BN, PGM_GLU_BN, PGM_BK8, PGM_STAGES, PGM_GLU_STAGES, PGM_MFRAG, PGM_NFRAG,
           PGM_GLU_NFRAG,
#ifdef PGM_SW8_OFF
           "OFF"
#else
           "on"
#endif
           );
    printf("# arena: w8a8 %zu B  glu-w8a8 %zu B  ladder %d B   PX13_PAD_SMEM=%zu B\n\n",
           (size_t)PGM_ARENA_W8A8*sizeof(bf16), (size_t)PGM_ARENA_GLU_W8A8*sizeof(bf16), PX9_LSMEM,
           pad_smem(0));

    if (!strcmp(sec,"ladder") || !strcmp(sec,"all")) {
        printf("[ladder] 1 blk/SM, 8 warps, acc[%d][%d][4] = %d f32/thread, %d QMMA per iteration\n",
               PGM_MFRAG, PGM_NFRAG, PGM_MFRAG*PGM_NFRAG*4, PX9_MMA_PER_IT);
        double c0 = 0, ghz = 0;
        ladder<0>("0 mma only (operands in registers)      CEILING", P, &c0, &ghz);
        double c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0;
        ladder<1>("1 + smem frags, AS SHIPPED (scalar u32)", P, &c1, nullptr);
        ladder<2>("2 + smem frags, VECTORIZED (uint2)     ", P, &c2, nullptr);
        ladder<3>("3 = rung 1 + __syncthreads per K-tile  ", P, &c3, nullptr);
        ladder<4>("4 = rung 2 + __syncthreads per K-tile  ", P, &c4, nullptr);
        ladder<5>("5 = rung 2, swizzle REMOVED (control)  ", P, &c5, nullptr);
        printf("  -> ceiling %.2f cyc/QMMA; shipped frag read costs %+.1f%%, vectorized %+.1f%%,\n"
               "     barrier adds %+.1f%% (shipped) / %+.1f%% (vectorized), unswizzled %+.1f%%\n\n",
               c0, 100.0*(c1/c0-1), 100.0*(c2/c0-1), 100.0*(c3/c1-1), 100.0*(c4/c2-1),
               100.0*(c5/c2-1));
    }
    if (!strcmp(sec,"gemm") || !strcmp(sec,"all")) {
        printf("[gemm] PX-7 protocol: oracle grid G*|T (u=1.000 asserted), L2-cold %d MB, "
               "%d warm + %d timed\n", PX9_COLD_MB, WARM, ITERS);
        section_gemm(P, !strcmp(sec,"all") ? 0 : 0);
        printf("\n");
    }
    if (!strcmp(sec,"hash")) { printf("[hash] bit-exactness gate\n"); section_hash(P); return 0; }
    if (!strcmp(sec,"one")) {
        int si = (argc > 2) ? atoi(argv[2]) : 0;
        unsigned M = (argc > 3) ? (unsigned)atoi(argv[3]) : 8192u;
        int fg = (argc > 4) ? atoi(argv[4]) : 0;
        section_one(P, si, M, fg);
        return 0;
    }
    if (!strcmp(sec,"lt") || !strcmp(sec,"all")) { printf("[lt]\n"); section_lt(); }
    return 0;
}

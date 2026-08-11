/* px7_w8a8_ceiling_bench.cu — what is plow's fp8 (w8a8) prefill GEMM worth with wave
 * quantization REMOVED?  Companion to px6_wavequant_bench.cu; that file is a campaign record
 * and is left untouched (same convention PX-6 used for gemm_occ_bench.cu).
 *
 * WHY.  End-to-end, plow's 127k prefill runs its GEMM at ~190 TFLOP/s against this card's
 * in-tree measured fp8 peak of 503.8 TFLOP/s (op_gemm.cuh:1282, rtx-05) — 38%, for the SAME
 * mma.m16n8k32.f32.e4m3 instruction class vLLM uses at ~peak.  Two candidate explanations, and
 * they lead to completely different fixes:
 *
 *   (a) SCHEDULING — the tiles are fine but the megakernel starves them (occ_per_sm = 1 because
 *       flash-prefill's 85,248 B smem sets the whole object's occupancy).  Fix = make the lean
 *       occ-2 GEMM segment object reachable from serve.  Runtime work.
 *   (b) KERNEL BODY — a tile simply costs this much.  Fix = d_gemm_w8a8 itself.  Kernel work.
 *
 * PX-6 already answered the analogous question for bf16: 116.5 TFLOPS at u = 1.000, i.e. 56% of
 * bf16 peak with ZERO quantization, so bf16 is body-bound.  Nobody has measured the fp8 arm that
 * way, and at the chunk sizes long context uses (tm = 16 at chunk 2048) wave efficiency is
 * already ~94%, so quantization CANNOT be the explanation there.  This bench nails it down.
 *
 * METHOD (PX-6's, reused so the numbers are comparable):
 *   - oracle grid G* = largest divisor of T that is <= P, so every block gets exactly T/G* tiles
 *     and quantization is zero BY CONSTRUCTION (u = 1.000 exactly, asserted below);
 *   - L2-cold: weights replicated to >= PX7_COLD_MB and cycled per iteration, because a 96 MiB
 *     L2 on this card will otherwise serve a 31 MB weight entirely from cache (PX-6 E1a:
 *     4090 GB/s warm at 32 MB vs 1695.6 GB/s at 2 GB);
 *   - 8 warm + 30 timed, cudaEvent around the whole batch;
 *   - run it under perf-data/tools/gpulease.
 *
 * NULL CONTROL: the bf16 arm at the same shape/grid must reproduce PX-6's ~116 TFLOPS.  If it
 * does not, the harness is wrong and the fp8 number means nothing.
 *
 * NOT a correctness test — operands are random bytes and no reference is computed.  Any
 * production change this motivates needs its own greedy-token parity check.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "sm120_common.cuh"
#include "op_gemm.cuh"

typedef __nv_bfloat16 bf16;

#ifndef PX7_COLD_MB
#define PX7_COLD_MB 700
#endif
static const int ITERS = 30, WARM = 8;

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

static uint32_t rng = 12345u;
static uint32_t xr() { rng = rng*1664525u + 1013904223u; return rng; }

static void* dev_bytes(size_t n) {
    std::vector<uint8_t> h(n);
    /* e4m3 bytes: avoid 0x7f/0xff (NaN) so the mma never sees a NaN operand. */
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

/* largest divisor of T that is <= P — zero quantization by construction (PX-6). */
static int oracle_grid(unsigned T, int P) {
    for (int g = std::min<int>(P, (int)T); g >= 1; g--) if (T % (unsigned)g == 0) return g;
    return 1;
}
/* largest divisor of T in (P, 2P] — the 2-blocks/SM oracle. 0 if none exists. */
static int oracle_grid2(unsigned T, int P) {
    for (int g = std::min<int>(2*P, (int)T); g > P; g--) if (T % (unsigned)g == 0) return g;
    return 0;
}

struct Shape { const char* name; unsigned N, K; int glu; };

int main(int argc, char** argv) {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    const int P = pr.multiProcessorCount;
    const size_t smem   = (size_t)PGM_ARENA_BF16      * sizeof(bf16);
    const size_t smem8  = (size_t)PGM_ARENA_W8A8     * sizeof(bf16);
    const size_t smem8g = (size_t)PGM_ARENA_GLU_W8A8 * sizeof(bf16);
    CK(cudaFuncSetAttribute(k_w8a8,     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8));
    CK(cudaFuncSetAttribute(k_w8a8_glu, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8g));
    CK(cudaFuncSetAttribute(k_bf16,     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    int o8 = 0, og = 0, ob = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o8, k_w8a8,     256, smem8));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&og, k_w8a8_glu, 256, smem8g));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ob, k_bf16,     256, smem));
    printf("# %s  SMs=%d  L2=%.0f MiB  occ w8a8=%d glu=%d bf16=%d\n",
           pr.name, P, pr.l2CacheSize/1048576.0, o8, og, ob);
    printf("# smem/blk: bf16 %zu B (occ-1 claim)  w8a8 %zu B  glu-w8a8 %zu B  "
           "(2 blocks/SM needs <= 50176)\n", smem, smem8, smem8g);
    printf("# BM=%d BN=%d BK8=%d STAGES=%d GLU_STAGES=%d  cold=%d MB  iters=%d\n",
           PGM_BM, PGM_BN, PGM_BK8, PGM_STAGES, PGM_GLU_STAGES, PX7_COLD_MB, ITERS);
    printf("# fp8 peak (in-tree rtx-05) = 503.8 TFLOP/s ; bf16 peak = 209.5\n\n");

    /* Real Gemma-4-12B prefill GEMMs. gate|up is the big one (2/3 of the FLOPs). */
    const Shape shapes[] = {
        {"gate|up", 15360, 3840, 1},   /* x48 */
        {"down",     3840, 15360, 0},  /* x48 */
        {"q_slide",  4096, 3840, 0},   /* x40 */
        {"k_slide",  2048, 3840, 0},   /* x40 */
        {"v_slide",  2048, 3840, 0},   /* x40 */
        {"o_slide",  3840, 4096, 0},   /* x40 */
        {"q_full",   8192, 3840, 0},   /* x8  */
        {"k_full",    512, 3840, 0},   /* x8  (k_eq_v: no v_proj) */
        {"o_full",   3840, 8192, 0},   /* x8  */
    };
    /* layer multiplicity, index-parallel to shapes[] */
    const int mult[] = { 48, 48, 40, 40, 40, 40, 8, 8, 8 };
    double gemm_ms_total[2] = {0.0, 0.0};
    /* M = prefill chunk. 2048 and 8192 are the rungs long context actually runs. */
    const unsigned Ms[] = {2048, 8192};

    printf("%-9s %6s %6s %6s %5s %6s %8s %9s %9s %7s\n",
           "shape","M","T","G*","u","arm","ms","TFLOP/s","%peak","vs bf16");
    for (size_t mi = 0; mi < sizeof(Ms)/sizeof(Ms[0]); mi++) {
        unsigned M = Ms[mi];
        for (size_t si = 0; si < sizeof(shapes)/sizeof(shapes[0]); si++) {
            const Shape& s = shapes[si];
            unsigned tm = (M + PGM_BM - 1)/PGM_BM, tn = (s.N + PGM_BN - 1)/PGM_BN;
            unsigned T = tm*tn;
            int G = oracle_grid(T, P);
            double u = (double)T / ((double)((T + G - 1)/G) * G);
            if (u < 0.9999) { printf("!! oracle grid failed for %s M=%u\n", s.name, M); return 1; }

            size_t wn = (size_t)s.N * s.K;
            int nrep = (int)std::max<size_t>(2, ((size_t)PX7_COLD_MB<<20) /
                                                 std::max<size_t>(wn*(s.glu?2:1), 1));
            nrep = std::min(nrep, 16);
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

            /* bf16 twin at the SAME shape/grid — the null control against PX-6's 116.5. */
            double msb = 0;
            {
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
            cudaEventDestroy(e0); cudaEventDestroy(e1);

            int G2 = oracle_grid2(T, P);
            double ms2 = 0;
            if (G2) {
                cudaEvent_t f0,f1; CK(cudaEventCreate(&f0)); CK(cudaEventCreate(&f1));
                auto run2 = [&](int it){ int r = it % nrep;
                    if (s.glu) k_w8a8_glu<<<G2,256,smem8g>>>(C,A8,Bg[r],Bu[r],as,ws,ws2,M,s.N,s.K);
                    else       k_w8a8    <<<G2,256,smem8 >>>(C,A8,Bg[r],as,ws,M,s.N,s.K); };
                for (int i = 0; i < WARM; i++) run2(i);
                CK(cudaDeviceSynchronize()); CK(cudaEventRecord(f0));
                for (int i = 0; i < ITERS; i++) run2(i);
                CK(cudaEventRecord(f1)); CK(cudaEventSynchronize(f1));
                float t=0; CK(cudaEventElapsedTime(&t,f0,f1)); ms2 = t/ITERS;
                CK(cudaGetLastError());
                cudaEventDestroy(f0); cudaEventDestroy(f1);
            }
            double fl  = 2.0*M*s.N*s.K*(s.glu?2.0:1.0);   /* w8a8 arm: glu does 2 matmuls */
            double flb = 2.0*M*s.N*s.K;                   /* bf16 control is PLAIN d_gemm */
            double tf8 = fl/(ms8*1e-3)/1e12, tfb = flb/(msb*1e-3)/1e12;
            printf("%-9s %6u %6u %6d %5.3f %6s %8.4f %9.1f %8.1f%% %7.2fx\n",
                   s.name, M, T, G, u, s.glu?"glu8":"w8a8", ms8, tf8, 100.0*tf8/503.8, tf8/tfb);
            printf("%-9s %6u %6u %6d %5.3f %6s %8.4f %9.1f %8.1f%% %7s\n",
                   "", M, T, G, u, "bf16", msb, tfb, 100.0*tfb/209.5, "-");
            if (G2) {
                double tf2 = fl/(ms2*1e-3)/1e12;
                printf("%-9s %6u %6u %6d %5.3f %6s %8.4f %9.1f %8.1f%% %6.3fx  <- 2 blk/SM\n",
                       "", M, T, G2, 1.0, s.glu?"glu8":"w8a8", ms2, tf2,
                       100.0*tf2/503.8, tf2/(fl/(ms8*1e-3)/1e12));
            }

            gemm_ms_total[mi] += ms8 * mult[si];
            for (int r = 0; r < nrep; r++) { cudaFree(Bg[r]); if (s.glu) cudaFree(Bu[r]); }
            cudaFree(A8); cudaFree(as); cudaFree(ws); cudaFree(ws2); cudaFree(C);
        }
        printf("  -> GEMM-ONLY model for ONE prefill chunk of M=%u: %.2f ms "
               "(sum over 48 layers, isolated, u=1.000, 1 blk/SM)\n\n",
               M, gemm_ms_total[mi]);
    }
    printf("# Subtract these from the measured per-chunk prefill time to size the non-GEMM\n"
           "# remainder (counter gates, QUANT_FP8, norms/RoPE/residual, real-grid quantization,\n"
           "# and the flash prefill, which grows with context and is NOT in this model).\n");
    return 0;
}

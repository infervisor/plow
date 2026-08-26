/* Isolated bf16 GEMM/GEMM_GLU vs cuBLASLt, for Gemma-4-12B's real prefill shapes, on the
 * ACTUAL deployed tile config (PGM_BN=192, PGM_BN_GLU=128 — the winning bf16 config from this
 * session). Mirrors perf-data/px9-gemm-body.md's method (oracle grid, L2-cold, full-grid arm)
 * but for bf16xbf16 instead of w8a8, since no such comparison exists yet for this GPU/dtype.
 *
 * Build: nvcc -O3 -arch=sm_120a -DPGM_BN=192 -DPGM_BN_GLU=128 -I <repo>/runtime/nvidia \
 *   bf16_gemm_vs_cublas.cu -o bf16bench -lcublasLt
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

typedef __nv_bfloat16 bf16;
#include "op_gemm.cuh"

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); exit(2);} } while(0)
#define LTK(x) do { cublasStatus_t s_=(x); if (s_!=CUBLAS_STATUS_SUCCESS){ \
    printf("cublasLt %s @%d: %d\n",#x,__LINE__,(int)s_); exit(2);} } while(0)

static const int WARM = 5, ITERS = 30;
static const size_t COLD_MB = 700; /* PX-9's own L2-cold budget */

static int oracle_grid(unsigned T, int P) {
    for (int g = std::min<int>(P, (int)T); g >= 1; g--) if (T % (unsigned)g == 0) return g;
    return 1;
}

__global__ void k_gemm(bf16* C, const bf16* A, const bf16* B, unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 sm[];
    d_gemm(C, A, B, m, n, k, 0, blockIdx.x, gridDim.x, sm);
}
__global__ void k_gemm_glu(bf16* C, const bf16* A, const bf16* Wg, const bf16* Wu,
                           unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 sm[];
    d_gemm_glu(C, A, Wg, Wu, m, n, k, 0, blockIdx.x, gridDim.x, sm);
}

struct Shape { const char* name; unsigned N, K; int glu; };
static const Shape SHAPES[] = {
    {"gate|up", 15360, 3840, 1},
    {"down",     3840, 15360, 0},
    {"q_full",   8192, 3840, 0},
    {"o_full",   3840, 8192, 0},
};
static const int NSHAPE = 4;

static bf16* dev_bf16(size_t n) {
    bf16* d; CK(cudaMalloc(&d, n*sizeof(bf16)));
    CK(cudaMemset(d, 0x3c, n*sizeof(bf16))); /* ~1.0 in bf16, nonzero, avoids denormal weirdness */
    return d;
}

static void bench_plow(unsigned M, unsigned P) {
    printf("%-9s %6s %6s %6s %5s %6s %8s %9s\n","shape","M","T","G*","u","arm","ms","TFLOP/s");
    for (int si = 0; si < NSHAPE; si++) {
        const Shape& s = SHAPES[si];
        unsigned bn = s.glu ? (unsigned)PGM_BN_GLU : (unsigned)PGM_BN;
        unsigned tm = (M + PGM_BM - 1)/PGM_BM, tn = (s.N + bn - 1)/bn;
        unsigned T = tm*tn;
        int G = oracle_grid(T, P);

        size_t wn = (size_t)s.N * s.K;
        int nrep = (int)std::max<size_t>(2, ((size_t)COLD_MB<<20) / std::max<size_t>(wn*(s.glu?2:1),1));
        nrep = std::min(nrep, 16);
        std::vector<bf16*> Bg(nrep), Bu(s.glu?nrep:0);
        for (int r = 0; r < nrep; r++) { Bg[r] = dev_bf16(wn); if (s.glu) Bu[r] = dev_bf16(wn); }
        bf16* A = dev_bf16((size_t)M*s.K);
        bf16* C; CK(cudaMalloc(&C, (size_t)M*s.N*sizeof(bf16)));

        size_t smem = s.glu ? (size_t)PGM_ARENA_GLU*sizeof(bf16) : (size_t)PGM_ARENA_PLAIN*sizeof(bf16);
        if (s.glu) CK(cudaFuncSetAttribute(k_gemm_glu, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
        else       CK(cudaFuncSetAttribute(k_gemm,     cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

        auto run = [&](int it, int grid){
            int r = it % nrep;
            if (s.glu) k_gemm_glu<<<grid,256,smem>>>(C,A,Bg[r],Bu[r],M,s.N,s.K);
            else       k_gemm    <<<grid,256,smem>>>(C,A,Bg[r],M,s.N,s.K);
        };
        for (int i=0;i<WARM;i++) run(i,G);
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0));
        for (int i=0;i<ITERS;i++) run(i,G);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float msG=0; CK(cudaEventElapsedTime(&msG,e0,e1)); msG/=ITERS;
        CK(cudaGetLastError());

        for (int i=0;i<WARM;i++) run(i,P);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i=0;i<ITERS;i++) run(i,P);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float msP=0; CK(cudaEventElapsedTime(&msP,e0,e1)); msP/=ITERS;
        CK(cudaGetLastError());

        double fl = 2.0*M*s.N*s.K*(s.glu?2.0:1.0);
        double tfG = fl/(msG*1e-3)/1e12, tfP = fl/(msP*1e-3)/1e12;
        printf("%-9s %6u %6u %6d %5.3f %6s %8.4f %9.1f\n", s.name,M,T,G,(double)T/((T+G-1)/G*G),"oracle",msG,tfG);
        printf("%-9s %6u %6u %6d %5s %6s %8.4f %9.1f\n", "",M,T,P,"-","fullG",msP,tfP);

        for (int r=0;r<nrep;r++){ cudaFree(Bg[r]); if (s.glu) cudaFree(Bu[r]); }
        cudaFree(A); cudaFree(C);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
}

static void bench_cublas(unsigned M) {
    cublasLtHandle_t lt; LTK(cublasLtCreate(&lt));
    void* ws; size_t wsz = 256*1024*1024; CK(cudaMalloc(&ws, wsz));
    printf("\n%-9s %6s %8s %9s\n","shape(cuBLASLt)","M","ms","TFLOP/s");
    for (int si = 0; si < NSHAPE; si++) {
        const Shape& s = SHAPES[si];
        /* GLU shape measured as ONE bf16xbf16 matmul at its (N,K) — same simplification px9's
         * own k_bf16 control used; the two-B-stream GLU cost is ~2x this by construction (same
         * A ring, same mma throughput, just two independent accumulations) so this is still a
         * fair per-FLOP reference. */
        size_t wn = (size_t)s.N*s.K;
        int nrep = (int)std::max<size_t>(2, ((size_t)COLD_MB<<20)/std::max<size_t>(wn,1));
        nrep = std::min(nrep, 16);
        std::vector<bf16*> Bv(nrep);
        for (int r=0;r<nrep;r++) Bv[r] = dev_bf16(wn);
        bf16* A = dev_bf16((size_t)M*s.K);
        bf16* C; CK(cudaMalloc(&C,(size_t)M*s.N*sizeof(bf16)));

        cublasLtMatmulDesc_t op=nullptr;
        LTK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,&tA,sizeof(tA)));
        LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,&tB,sizeof(tB)));
        cublasLtMatrixLayout_t la=nullptr, lb=nullptr, ld=nullptr;
        LTK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, s.K, s.N, s.K)); /* B: [N,K] row-major = [K,N] col-major, TN */
        LTK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, s.K, M,   s.K)); /* A: [M,K] row-major = [K,M] col-major */
        LTK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, s.N, M,   s.N)); /* C: [M,N] row-major = [N,M] col-major */
        cublasLtMatmulPreference_t pref=nullptr;
        LTK(cublasLtMatmulPreferenceCreate(&pref));
        LTK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz)));
        cublasLtMatmulHeuristicResult_t heur; int nres=0;
        LTK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 1, &heur, &nres));
        if (nres==0) { printf("!! no cuBLASLt algo for %s\n", s.name); continue; }
        float alpha=1.f, beta=0.f;
        auto run=[&](int it,cudaStream_t st){ int r=it%nrep;
            LTK(cublasLtMatmul(lt, op, &alpha, Bv[r], la, A, lb, &beta, C, ld, C, ld,
                               &heur.algo, ws, wsz, st)); };
        for (int i=0;i<WARM;i++) run(i,0);
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0));
        for (int i=0;i<ITERS;i++) run(i,0);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=ITERS;
        double fl = 2.0*M*s.N*s.K;
        printf("%-9s %6u %8.4f %9.1f\n", s.name, M, ms, fl/(ms*1e-3)/1e12);

        cublasLtMatmulDescDestroy(op); cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
        cublasLtMatmulPreferenceDestroy(pref);
        for (int r=0;r<nrep;r++) cudaFree(Bv[r]);
        cudaFree(A); cudaFree(C);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    cudaFree(ws);
    cublasLtDestroy(lt);
}

int main(int argc, char** argv) {
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    int P = prop.multiProcessorCount;
    unsigned M = argc > 1 ? (unsigned)atoi(argv[1]) : 8192;
    printf("PGM_BN=%d PGM_BN_GLU=%d PGM_BM=%d SMs=%d M=%u\n", PGM_BN, PGM_BN_GLU, PGM_BM, P, M);
    bench_plow(M, (unsigned)P);
    bench_cublas(M);
    return 0;
}

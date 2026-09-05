/* Isolated bf16 GEMM/GEMM_GLU vs cuBLASLt, for Gemma-4-12B's real prefill shapes, on the
 * ACTUAL deployed tile config (PGM_BN=192, PGM_BN_GLU=128 — the winning bf16 config from this
 * session). Mirrors perf-data/px9-gemm-body.md's method (oracle grid, L2-cold, full-grid arm)
 * but for bf16xbf16 instead of w8a8, since no such comparison exists yet for this GPU/dtype.
 *
 * Build: nvcc -O3 -arch=sm_120a -DPGM_BN=192 -DPGM_BN_GLU=128 -I <repo>/runtime/nvidia \
 *   bf16_gemm_vs_cublas.cu -o bf16bench -lcublasLt
 */
#include <cuda.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#ifdef PLOW_BENCH_QWEN_GEMV
#include <cublas_v2.h>
#endif
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

/* Opt-in Hopper: -DPLOW_BENCH_WS384=1 -gencode arch=compute_90a,code=sm_90a -lcuda. */
#ifdef PLOW_BENCH_WS384
#define PLOW_NV_HOPPER 1
#define PLOW_NV_TMA_GEMM 1
#define PLOW_NV_SEG_GEMM 1
#define PLOW_NV_SEGMENTS 1
#define PLOW_NV_SEG_WS384 1
#define PGM90_UNI_BN256 1
#define PGM90_TMA_STAGES 3
#endif
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
#ifdef PLOW_BENCH_QWEN_GEMV
#ifdef PLOW_BENCH_GEMV_M16
    {"gemma_down",3840,15360,0}, {"gemma_q",8192,3840,0}, {"gemma_o",3840,8192,0},
#endif
    {"a_or_b", 48, 5120, 0}, {"qkv", 10240, 5120, 0},
    {"z", 6144, 5120, 0}, {"gdn_out", 5120, 6144, 0},
    {"q_full", 12288, 5120, 0}, {"k_or_v", 1024, 5120, 0},
    {"gate_or_up", 17408, 5120, 0}, {"down", 5120, 17408, 0},
    {"lm_head", 248320, 5120, 0},
    {"fused_ba", 96, 5120, 0}, {"fused_qkvz", 16384, 5120, 0},
    {"fused_qkv", 14336, 5120, 0}, {"fused_gtup", 34816, 5120, 0},
#else
#ifndef PLOW_BENCH_WS384
    {"gate|up", 15360, 3840, 1},
#endif
    {"down",     3840, 15360, 0},
    {"q_full",   8192, 3840, 0},
    {"o_full",   3840, 8192, 0},
#endif
};
static const int NSHAPE = sizeof(SHAPES) / sizeof(SHAPES[0]);

#if defined(PLOW_BENCH_WS384) || defined(PLOW_BENCH_QWEN_GEMV)
__global__ void init_nonconstant(bf16* d, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned v = (unsigned)i ^ seed;
        v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
        d[i] = __float2bfloat16(((int)(v & 1023u) - 512) / 1024.f);
    }
}
#ifdef PLOW_BENCH_WS384
__global__ __maxnreg__(160) void k_ws384(bf16* C, const void* ma, const void* mb,
                                       unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        d_gemm_sm90_tma_ws384_role<true, false>(C, ma, mb, nullptr, nullptr,
            m, n, k, 0, blockIdx.x, gridDim.x, arena);
    } else {
        sm90_reg_inc(224);
        d_gemm_sm90_tma_ws384_role<false, false>(C, ma, mb, nullptr, nullptr,
            m, n, k, 0, blockIdx.x, gridDim.x, arena);
    }
}
static CUtensorMap bf16_map(void* base, unsigned rows, unsigned k) {
    CUtensorMap map{};
    uint64_t dims[] = {k, rows}, strides[] = {2ull * k};
    uint32_t box[] = {64, 128}, elements[] = {1, 1};
    CUresult rc = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        base, dims, strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (rc != CUDA_SUCCESS) { printf("tensor map failed: %d\n", (int)rc); exit(2); }
    return map;
}
#endif
#endif

#ifdef PLOW_BENCH_QWEN_GEMV
__global__ void k_decode_gemv(bf16* c, const bf16* a, const bf16* w, unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
#if defined(PLOW_NV_HOPPER) && PLOW_NV_GEMV_M16_MMA
    if (m == 16 && n >= 1024 && k && !(k % 64)) {
        d_gemv_sm90_m16(c, a, w, n, k, blockIdx.x, gridDim.x, arena);
        return;
    }
#endif
#if defined(PLOW_NV_HOPPER) && PLOW_NV_GEMV_KPANEL
    if (m == 1 && n == 5120 && k == 17408 && blockDim.x == 256 &&
        (5120u + gridDim.x - 1u) / gridDim.x <= 40u) {
        d_gemv_sm90_kpanel(c, a, w, blockIdx.x, gridDim.x);
        return;
    }
#endif
#if defined(PLOW_NV_HOPPER) && PLOW_NV_GEMV_XREG
    if (m == 1 && n >= 1024 && (k == 5120 || k == 6144)) {
        if (k == 5120) d_gemv_sm90_xreg<5120>(c, a, w, n, blockIdx.x, gridDim.x);
        else d_gemv_sm90_xreg<6144>(c, a, w, n, blockIdx.x, gridDim.x);
        return;
    }
#endif
    if (k * sizeof(bf16) <= 12352)
        d_gemv(c, a, w, m, n, k, blockIdx.x, gridDim.x, arena);
    else d_gemv(c, a, w, m, n, k, blockIdx.x, gridDim.x);
}
__global__ void evict_l2(unsigned* p, size_t n) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)
        p[i] += 1;
}
static void cold_flush() {
    static unsigned* buffer = nullptr;
    if (!buffer) { CK(cudaMalloc(&buffer, COLD_MB<<20)); CK(cudaMemset(buffer,0,COLD_MB<<20)); }
    evict_l2<<<256,256>>>(buffer,(COLD_MB<<20)/sizeof(unsigned));
    CK(cudaGetLastError());
}
#else
static void cold_flush() {}
#endif

static bf16* dev_bf16(size_t n) {
    bf16* d; CK(cudaMalloc(&d, n*sizeof(bf16)));
#if defined(PLOW_BENCH_WS384) || defined(PLOW_BENCH_QWEN_GEMV)
    static unsigned seed = 1;
    init_nonconstant<<<256, 256>>>(d, n, seed++);
    CK(cudaGetLastError());
#else
    CK(cudaMemset(d, 0x3c, n*sizeof(bf16)));
#endif
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
#ifdef PLOW_BENCH_QWEN_GEMV
        if (wn * sizeof(bf16) < (4u << 20)) nrep = ITERS;
#endif
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
        if (nres==0) {
            printf("!! no cuBLASLt algo for %s\n", s.name);
#if defined(PLOW_BENCH_WS384) || defined(PLOW_BENCH_QWEN_GEMV)
            exit(2);
#else
            continue;
#endif
        }
        float alpha=1.f, beta=0.f;
        auto run=[&](int it,cudaStream_t st){ int r=it%nrep;
            LTK(cublasLtMatmul(lt, op, &alpha, Bv[r], la, A, lb, &beta, C, ld, C, ld,
                               &heur.algo, ws, wsz, st)); };
#if defined(PLOW_BENCH_WS384) || defined(PLOW_BENCH_QWEN_GEMV)
        cublasLtMatmulHeuristicResult_t candidates[32];
        LTK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 32, candidates, &nres));
        cudaEvent_t tune0, tune1; CK(cudaEventCreate(&tune0)); CK(cudaEventCreate(&tune1));
        float best = INFINITY; int selected = -1;
        for (int a = 0; a < nres; a++) {
            if (candidates[a].state != CUBLAS_STATUS_SUCCESS) continue;
            heur = candidates[a];
            for (int i = 0; i < WARM; i++) run(i, 0);
            cold_flush();
            CK(cudaEventRecord(tune0));
            for (int i = 0; i < ITERS; i++) run(i, 0);
            CK(cudaEventRecord(tune1)); CK(cudaEventSynchronize(tune1));
            float elapsed; CK(cudaEventElapsedTime(&elapsed, tune0, tune1));
            if (elapsed < best) { best = elapsed; selected = a; }
        }
        if (selected < 0) { printf("no usable cuBLASLt candidate\n"); exit(2); }
        heur = candidates[selected];
        printf("cuBLASLt selected=%d candidates=%d workspace=%zu\n", selected, nres, heur.workspaceSize);
        CK(cudaEventDestroy(tune0)); CK(cudaEventDestroy(tune1));
#endif
        for (int i=0;i<WARM;i++) run(i,0);
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        cold_flush();
        CK(cudaEventRecord(e0));
        for (int i=0;i<ITERS;i++) run(i,0);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=ITERS;
        double fl = 2.0*M*s.N*s.K;
        printf("%-9s %6u %8.4f %9.1f\n", s.name, M, ms, fl/(ms*1e-3)/1e12);

#if defined(PLOW_BENCH_WS384) || defined(PLOW_BENCH_QWEN_GEMV)
        int dev; CK(cudaGetDevice(&dev));
        cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
#ifdef PLOW_BENCH_QWEN_GEMV
        bf16* cp; CK(cudaMalloc(&cp, (size_t)M*s.N*sizeof(bf16)));
        auto run_plow = [&](int it) {
            k_decode_gemv<<<prop.multiProcessorCount,256,12352>>>(cp,A,Bv[it%nrep],M,s.N,s.K);
        };
#else
        const unsigned smem = PGM90_U256_ARENA * sizeof(bf16);
        CK(cudaFuncSetAttribute(k_ws384, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        std::vector<CUtensorMap> maps{bf16_map(A, M, s.K)};
        for (auto b : Bv) maps.push_back(bf16_map(b, s.N, s.K));
        CUtensorMap* dm; CK(cudaMalloc(&dm, maps.size() * sizeof(CUtensorMap)));
        CK(cudaMemcpy(dm, maps.data(), maps.size() * sizeof(CUtensorMap), cudaMemcpyHostToDevice));
        bf16* cp; CK(cudaMalloc(&cp, (size_t)M * s.N * sizeof(bf16)));
        auto run_plow = [&](int it) {
            k_ws384<<<prop.multiProcessorCount, 384, smem>>>(cp, dm, dm + 1 + it % nrep, M, s.N, s.K);
        };
#endif
        run(0, 0); run_plow(0);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<bf16> ref((size_t)M * s.N), got(ref.size());
        CK(cudaMemcpy(ref.data(), C, ref.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got.data(), cp, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
        double err2 = 0, ref2 = 0, maxerr = 0, maxref = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            double r = __bfloat162float(ref[i]), v = __bfloat162float(got[i]);
            if (!std::isfinite(r) || !std::isfinite(v)) { printf("nonfinite output\n"); exit(3); }
            double e = v - r;
            err2 += e * e; ref2 += r * r;
            maxerr = std::max(maxerr, std::abs(e)); maxref = std::max(maxref, std::abs(r));
        }
        double rel = std::sqrt(err2 / std::max(ref2, 1e-30));
        printf("correctness %s M=%u relL2=%.6g max_abs=%.6g max_ref=%.6g\n", s.name, M, rel, maxerr, maxref);
        if (rel > 0.006 || maxerr > 0.05 + 0.02 * maxref) exit(3);
        for (int i = 0; i < WARM; i++) run_plow(i);
        CK(cudaDeviceSynchronize()); cold_flush(); CK(cudaEventRecord(e0));
        for (int i = 0; i < ITERS; i++) run_plow(i);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float pms; CK(cudaEventElapsedTime(&pms, e0, e1)); pms /= ITERS;
        printf("%-9s %6u %8.4f %9.1f Plow speedup=%.4f cold_MiB=%.1f\n", s.name, M, pms,
               fl / (pms * 1e-3) / 1e12, ms / pms, (double)nrep * wn * sizeof(bf16) / (1024 * 1024));
#ifdef PLOW_BENCH_QWEN_GEMV
        printf("bandwidth %s Lt_GBs=%.1f Plow_GBs=%.1f arena=12352 M=%u\n",
            s.name, 2.0*wn/(ms*1e6), 2.0*wn/(pms*1e6), M);
#else
        CK(cudaFree(dm));
#endif
        CK(cudaFree(cp));
#endif


#ifdef PLOW_BENCH_QWEN_GEMV
        cublasHandle_t blas; LTK(cublasCreate(&blas));
        bf16* cb; CK(cudaMalloc(&cb,(size_t)M*s.N*sizeof(bf16)));
        auto run_blas = [&](int it) {
            LTK(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,s.N,M,s.K,
                &alpha,Bv[it%nrep],CUDA_R_16BF,s.K,A,CUDA_R_16BF,s.K,
                &beta,cb,CUDA_R_16BF,s.N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        };
        run(0,0); run_blas(0); CK(cudaDeviceSynchronize());
        std::vector<bf16> br((size_t)M*s.N), bg(br.size());
        CK(cudaMemcpy(br.data(),C,br.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bg.data(),cb,bg.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
        double be=0, bn=0;
        for (size_t i=0;i<br.size();i++) {
            double x=__bfloat162float(br[i]), y=__bfloat162float(bg[i]);
            if (!std::isfinite(x)||!std::isfinite(y)) exit(3);
            be+=(x-y)*(x-y); bn+=x*x;
        }
        double brel=std::sqrt(be/std::max(bn,1e-30));
        printf("cuBLAS correctness %s relL2=%.6g\n",s.name,brel);
        if (brel>0.006) exit(3);
        for (int i=0;i<WARM;i++) run_blas(i);
        cold_flush(); CK(cudaEventRecord(e0));
        for (int i=0;i<ITERS;i++) run_blas(i);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float bms; CK(cudaEventElapsedTime(&bms,e0,e1)); bms/=ITERS;
        printf("cuBLAS %s M=%u ms=%.6f GBs=%.1f\n",s.name,M,bms,2.0*wn/(bms*1e6));
        CK(cudaFree(cb)); LTK(cublasDestroy(blas));
#endif

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
#ifdef PLOW_BENCH_QWEN_GEMV
#ifdef PLOW_BENCH_GEMV_M16
    const unsigned rows = 16;
#else
    const unsigned rows = 1;
#endif
    if (argc > 1 && M != rows) { printf("GEMV mode requires M=%u\n", rows); return 2; }
    if (prop.major != 9) { printf("GEMV comparison requires Hopper\n"); return 2; }
    printf("BF16 GEMV vs cuBLASLt M=%u SMs=%d GV_UNROLL=%d\n", rows, P, GV_UNROLL);
    bench_cublas(rows);
    return 0;
#endif
#if defined(PLOW_BENCH_WS384) || defined(PLOW_BENCH_QWEN_GEMV)
    printf("Hopper ws384 BF16 vs cuBLASLt SMs=%d\n", P);
    if (prop.major != 9) { printf("ws384 comparison requires Hopper\n"); return 2; }
    if (argc > 1) bench_cublas(M);
    else for (unsigned rows : {128u, 1024u, 4096u}) bench_cublas(rows);
#else
    printf("PGM_BN=%d PGM_BN_GLU=%d PGM_BM=%d SMs=%d M=%u\n", PGM_BN, PGM_BN_GLU, PGM_BM, P, M);
    bench_plow(M, (unsigned)P);
    bench_cublas(M);
#endif
    return 0;
}

#ifdef PLOW_BENCH_FP8_ABI
/* Compile as a shared library with -DPLOW_BENCH_FP8_ABI=1 -lcublasLt.
 * Device scale pointers and all input/output buffers must outlive queued work. */
struct Fp8M1 {
    cublasLtHandle_t lt{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t w{}, a{}, out{};
    cublasLtMatmulPreference_t pref{};
    cublasLtMatmulHeuristicResult_t algo{};
    void* workspace{};
    size_t workspace_bytes = 64u << 20;
};
extern "C" void plow_fp8_m1_destroy(void* opaque) {
    auto* h = static_cast<Fp8M1*>(opaque);
    if (!h) return;
    if (h->workspace) cudaFree(h->workspace);
    if (h->pref) cublasLtMatmulPreferenceDestroy(h->pref);
    if (h->out) cublasLtMatrixLayoutDestroy(h->out);
    if (h->a) cublasLtMatrixLayoutDestroy(h->a);
    if (h->w) cublasLtMatrixLayoutDestroy(h->w);
    if (h->op) cublasLtMatmulDescDestroy(h->op);
    if (h->lt) cublasLtDestroy(h->lt);
    delete h;
}
extern "C" int plow_fp8_m1_create(int n, int k, int physical_m,
    const float* weight_scale, const float* activation_scale, void** result) {
    if (!result) return -1;
    *result = nullptr;
    if (n <= 0 || k <= 0 || (physical_m != 1 && physical_m != 16) ||
        !weight_scale || !activation_scale) return -1;
    auto* h = new Fp8M1;
#define FP8_TRY(expr) do { auto rc = (expr); if (rc != CUBLAS_STATUS_SUCCESS) { \
    plow_fp8_m1_destroy(h); return (int)rc; } } while (0)
    FP8_TRY(cublasLtCreate(&h->lt));
    FP8_TRY(cublasLtMatmulDescCreate(&h->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
    FP8_TRY(cublasLtMatmulDescSetAttribute(h->op, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
    FP8_TRY(cublasLtMatmulDescSetAttribute(h->op, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal)));
    FP8_TRY(cublasLtMatmulDescSetAttribute(h->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scale, sizeof(weight_scale)));
    FP8_TRY(cublasLtMatmulDescSetAttribute(h->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &activation_scale, sizeof(activation_scale)));
    FP8_TRY(cublasLtMatrixLayoutCreate(&h->w, CUDA_R_8F_E4M3, k, n, k));
    FP8_TRY(cublasLtMatrixLayoutCreate(&h->a, CUDA_R_8F_E4M3, k, physical_m, k));
    FP8_TRY(cublasLtMatrixLayoutCreate(&h->out, CUDA_R_16BF, n, physical_m, n));
    FP8_TRY(cublasLtMatmulPreferenceCreate(&h->pref));
    FP8_TRY(cublasLtMatmulPreferenceSetAttribute(h->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &h->workspace_bytes, sizeof(h->workspace_bytes)));
    int found = 0;
    FP8_TRY(cublasLtMatmulAlgoGetHeuristic(h->lt, h->op, h->w, h->a, h->out, h->out,
                                        h->pref, 1, &h->algo, &found));
    if (!found || h->algo.state != CUBLAS_STATUS_SUCCESS) { plow_fp8_m1_destroy(h); return -2; }
    if (cudaMalloc(&h->workspace, h->workspace_bytes) != cudaSuccess) { plow_fp8_m1_destroy(h); return -3; }
    *result = h;
    return 0;
#undef FP8_TRY
}
extern "C" int plow_fp8_m1_run(void* opaque, const void* weight, const void* activation,
                              void* output, void* stream) {
    auto* h = static_cast<Fp8M1*>(opaque);
    if (!h || !weight || !activation || !output) return -1;
    const float alpha = 1.0f, beta = 0.0f;
    return (int)cublasLtMatmul(h->lt, h->op, &alpha, weight, h->w, activation, h->a,
        &beta, output, h->out, output, h->out, &h->algo.algo,
        h->workspace, h->workspace_bytes, (cudaStream_t)stream);
}
#endif

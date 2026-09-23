/* Isolated bf16 GEMM/GEMM_GLU vs cuBLASLt on real prefill projection shapes. The default
 * sm_120 path uses the deployed PGM_BN/PGM_BN_GLU tile; PLOW_BENCH_WS384 exercises Hopper's
 * production TMA/wgmma body. Hopper comparisons use bounded-error output gates, rotated cold
 * weights, symmetric L2 eviction, alternating order, and median timing across short bursts.
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
#include <chrono>
#include <thread>

/* Opt-in Hopper: -DPLOW_BENCH_WS384=1 -gencode arch=compute_90a,code=sm_90a -lcuda. */
#ifdef PLOW_BENCH_M64N64
#define PLOW_NV_SEG_M64N64 1
#endif
#ifdef PLOW_BENCH_M64N128
#define PLOW_NV_SEG_M64N128 1
#endif
#ifdef PLOW_BENCH_WS384
#define PLOW_NV_HOPPER 1
#define PLOW_NV_TMA_GEMM 1
#define PLOW_NV_SEG_GEMM 1
#define PLOW_NV_SEGMENTS 1
#define PLOW_NV_SEG_WS384 1
#define PGM90_UNI_BN256 1
#ifndef PGM90_TMA_STAGES
#define PGM90_TMA_STAGES 3
#endif
/* -DPLOW_BENCH_W8A8 swaps the SAME ws384 body to its E4M3 instantiation — the shipped
 * interp_sm90a_pfgemm.cubin's w8a8 arm (build_sm90a_gemma4_segments.sh adds exactly
 * PLOW_NV_W8A8=1 and PGM90_FP8_PROMOTE on top of the bf16 flags) — and points the cuBLASLt
 * reference at e4m3 with OUTER_VEC per-channel/per-token scales. */
#ifdef PLOW_BENCH_W8A8
#define PLOW_NV_W8A8 1
#ifndef PGM90_FP8_PROMOTE
#define PGM90_FP8_PROMOTE 1
#endif
#endif
#endif
typedef __nv_bfloat16 bf16;
#include "op_gemm.cuh"

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); exit(2);} } while(0)
#define LTK(x) do { cublasStatus_t s_=(x); if (s_!=CUBLAS_STATUS_SUCCESS){ \
    printf("cublasLt %s @%d: %d\n",#x,__LINE__,(int)s_); exit(2);} } while(0)

static const int WARM = 5, ITERS = 30;
#ifdef PLOW_BENCH_GEMM_ODOWN
static const size_t COLD_MB = 2048;
#else
static const size_t COLD_MB = 700; /* PX-9's own L2-cold budget */
#endif

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
#ifdef PLOW_BENCH_GEMMA4_ALL
    {"gate_or_up",15360,3840,0}, {"local_k_or_v",2048,3840,0},
    {"down",3840,15360,0}, {"local_q",4096,3840,0},
    {"local_o",3840,4096,0}, {"global_q",8192,3840,0},
    {"global_k_or_v",512,3840,0}, {"global_o",3840,8192,0},
#ifdef PLOW_BENCH_GEMMA4_26B
    /* segment_roles::CUBLASLT_PREFILL_GEMMA4_26B_SHAPES — Gemma-4-26B-A4B dense projections
     * (hidden 2816, dense inter 2112). The routed-expert GEMMs are MoE ops, not here. */
    {"m26_gate_up",2112,2816,0}, {"m26_down",2816,2112,0},
    {"m26_local_q",4096,2816,0}, {"m26_local_kv",2048,2816,0},
    {"m26_local_o",2816,4096,0}, {"m26_global_q",8192,2816,0},
    {"m26_global_k",1024,2816,0}, {"m26_global_o",2816,8192,0},
#endif
#elif defined(PLOW_BENCH_GEMM_ODOWN)
    {"g12_o_local",3840,4096,0}, {"g12_o_full",3840,8192,0}, {"g12_down",3840,15360,0},
    {"g31_o_local",5376,8192,0}, {"g31_o_full",5376,16384,0}, {"g31_down",5376,21504,0},
    {"qwen_o",5120,6144,0}, {"qwen_down",5120,17408,0}, {"tail",3906,648,0},
#ifdef PLOW_BENCH_GEMMA_QGATE
    {"g12_q_local",4096,3840,0}, {"g12_q_full",8192,3840,0}, {"g12_gate",15360,3840,0},
    {"g31_q_local",8192,5376,0}, {"g31_q_full",16384,5376,0}, {"g31_gate",21504,5376,0},
    {"g31_k_local",4096,5376,0}, {"g31_k_full",2048,5376,0},
#endif
#elif defined(PLOW_BENCH_QWEN_GEMV)
#ifdef PLOW_BENCH_GEMV_M16
    {"gemma_down",3840,15360,0}, {"gemma_q",8192,3840,0}, {"gemma_o",3840,8192,0},
#endif
#ifdef PLOW_BENCH_GEMV_M16_BK128
    {"g31_down",5376,21504,0}, {"g31_q",16384,5376,0}, {"g31_o",5376,16384,0},
    {"m16_k64",1030,64,0}, {"m16_k192",1030,192,0}, {"m16_k128",1030,128,0},
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
#if defined(PLOW_BENCH_M64N64) || defined(PLOW_BENCH_M64N128)
__global__ __maxnreg__(128) void k_m64(bf16* C, const void* ma, const void* mb,
                                       unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        d_gemm_sm90_tma_m64_role<true, PLOW_NV_SEG_M64N128 ? 128 : 64>(C, ma, mb, m, n, k, 0, blockIdx.x, gridDim.x, arena);
    } else {
        sm90_reg_inc(224);
        d_gemm_sm90_tma_m64_role<false, PLOW_NV_SEG_M64N128 ? 128 : 64>(C, ma, mb, m, n, k, 0, blockIdx.x, gridDim.x, arena);
    }
}
#endif
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
#ifdef PLOW_BENCH_W8A8
/* The SAME body, E4M3=true: what interp_sm90a_pfgemm.cubin runs for DevOp::GemmFp8
 * (interp_sm120.cu PLOW_DOP_GEMM_FP8 -> d_gemm_sm90_tma_ws384_role<PROD,true>). */
__global__ __maxnreg__(160) void k_ws384_fp8(bf16* C, const void* ma, const void* mb,
                                             const float* ascale, const float* wscale,
                                             unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        d_gemm_sm90_tma_ws384_role<true, true>(C, ma, mb, ascale, wscale,
            m, n, k, 0, blockIdx.x, gridDim.x, arena);
    } else {
        sm90_reg_inc(224);
        d_gemm_sm90_tma_ws384_role<false, true>(C, ma, mb, ascale, wscale,
            m, n, k, 0, blockIdx.x, gridDim.x, arena);
    }
}
/* CudaBackend::encode_tmap_e4m3's recipe, byte for byte: UINT8, rank 2, inner box 128. */
static CUtensorMap e4m3_map(void* base, unsigned rows, unsigned k) {
    CUtensorMap map{};
    uint64_t dims[] = {k, rows}, strides[] = {(uint64_t)k};
    uint32_t box[] = {128, 128}, elements[] = {1, 1};
    CUresult rc = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        base, dims, strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (rc != CUDA_SUCCESS) { printf("e4m3 tensor map failed: %d\n", (int)rc); exit(2); }
    return map;
}
__global__ void init_e4m3(uint8_t* d, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned v = (unsigned)i ^ seed;
        v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
        /* sign | exp 0b0110/0b0111 | mantissa: +-[0.5, 1.9375], never 0/inf/nan. */
        d[i] = (uint8_t)(((v & 1u) << 7) | (((v >> 1) & 1u) ? 0x38u : 0x30u) | ((v >> 4) & 7u));
    }
}
#endif
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
#if defined(PLOW_NV_HOPPER) && PLOW_NV_GEMV_M16_MMA
static constexpr unsigned decode_arena_bytes =
    PLOW_NV_GEMV_M16_ARENA_BYTES > 12352u ? PLOW_NV_GEMV_M16_ARENA_BYTES : 12352u;
#else
static constexpr unsigned decode_arena_bytes = 12352u;
#endif
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
#if defined(PLOW_BENCH_GEMV_M16_BK128) && PLOW_NV_GEMV_M16_BK128
__global__ void k_m16_bk64_reference(bf16* c, const bf16* a, const bf16* w,
                                    unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    d_gemv_sm90_m16_stage<64>(c, a, w, n, k, blockIdx.x, gridDim.x, arena);
}
#endif
#endif

#if defined(PLOW_BENCH_QWEN_GEMV) || defined(PLOW_BENCH_GEMM_ODOWN)
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
    static unsigned seed = [] {
        const char* value = std::getenv("PLOW_BENCH_SEED");
        return value ? (unsigned)std::strtoul(value, nullptr, 10) : 1u;
    }();
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
        const char* filter = getenv("PLOW_BENCH_SHAPE");
        if (filter && strcmp(filter, s.name)) continue;
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
        const char* filter = getenv("PLOW_BENCH_SHAPE");
        if (filter && strcmp(filter, s.name)) continue;
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
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
#if !defined(PLOW_BENCH_WS384) && !defined(PLOW_BENCH_QWEN_GEMV)
        for (int i=0;i<WARM;i++) run(i,0);
        CK(cudaDeviceSynchronize());
        cold_flush();
        CK(cudaEventRecord(e0));
        for (int i=0;i<ITERS;i++) run(i,0);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=ITERS;
        double fl = 2.0*M*s.N*s.K;
        printf("%-9s %6u %8.4f %9.1f\n", s.name, M, ms, fl/(ms*1e-3)/1e12);
#else
        float ms = 0;
        const double fl = 2.0*M*s.N*s.K;
#endif

#if defined(PLOW_BENCH_WS384) || defined(PLOW_BENCH_QWEN_GEMV)
        int dev; CK(cudaGetDevice(&dev));
        cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
#ifdef PLOW_BENCH_QWEN_GEMV
#ifdef PLOW_BENCH_GEMV_M16_BK128
        bf16* cp_alloc;
        CK(cudaMalloc(&cp_alloc, ((size_t)M * s.N + 16) * sizeof(bf16)));
        CK(cudaMemset(cp_alloc, 0xa5, ((size_t)M * s.N + 16) * sizeof(bf16)));
        bf16* cp = cp_alloc + 8;
#else
        bf16* cp; CK(cudaMalloc(&cp, (size_t)M*s.N*sizeof(bf16)));
#endif
        auto run_plow = [&](int it) {
            k_decode_gemv<<<prop.multiProcessorCount,256,decode_arena_bytes>>>(cp,A,Bv[it%nrep],M,s.N,s.K);
        };
#else
#if defined(PLOW_BENCH_M64N64) || defined(PLOW_BENCH_M64N128)
        const unsigned smem = PGM90_TMA_ARENA * sizeof(bf16);
        CK(cudaFuncSetAttribute(k_m64, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
#else
        const unsigned smem = PGM90_WS384_ARENA * sizeof(bf16);
        cudaFuncAttributes ws384_attr{};
        CK(cudaFuncGetAttributes(&ws384_attr, k_ws384));
        CK(cudaFuncSetAttribute(k_ws384, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
#endif
        std::vector<CUtensorMap> maps{bf16_map(A, M, s.K)};
        for (auto b : Bv) maps.push_back(bf16_map(b, s.N, s.K));
        CUtensorMap* dm; CK(cudaMalloc(&dm, maps.size() * sizeof(CUtensorMap)));
        CK(cudaMemcpy(dm, maps.data(), maps.size() * sizeof(CUtensorMap), cudaMemcpyHostToDevice));
        bf16* cp_alloc;
        CK(cudaMalloc(&cp_alloc, ((size_t)M * s.N + 16) * sizeof(bf16)));
        CK(cudaMemset(cp_alloc, 0xa5, ((size_t)M * s.N + 16) * sizeof(bf16)));
        bf16* cp = cp_alloc + 8;
        auto run_plow = [&](int it) {
#if defined(PLOW_BENCH_M64N64) || defined(PLOW_BENCH_M64N128)
            k_m64<<<prop.multiProcessorCount, 256, smem>>>(cp, dm, dm + 1 + it % nrep, M, s.N, s.K);
#else
            k_ws384<<<prop.multiProcessorCount, 384, smem>>>(cp, dm, dm + 1 + it % nrep, M, s.N, s.K);
#endif
        };
#endif
        run(0, 0); run_plow(0);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<bf16> ref((size_t)M * s.N), got(ref.size());
        CK(cudaMemcpy(ref.data(), C, ref.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got.data(), cp, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
#if defined(PLOW_BENCH_WS384) || defined(PLOW_BENCH_GEMV_M16_BK128)
        uint16_t guards[16];
        CK(cudaMemcpy(guards, cp_alloc, 8 * sizeof(bf16), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(guards + 8, cp + (size_t)M * s.N, 8 * sizeof(bf16), cudaMemcpyDeviceToHost));
        for (auto guard : guards) if (guard != 0xa5a5) { printf("output guard overwritten\n"); exit(3); }
#endif
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
#if defined(PLOW_BENCH_GEMV_M16_BK128) && PLOW_NV_GEMV_M16_BK128
        if (M == 16 && s.N >= 1024 && s.K && !(s.K % 64)) {
            bf16* cr; CK(cudaMalloc(&cr, got.size() * sizeof(bf16)));
            k_m16_bk64_reference<<<prop.multiProcessorCount,256,decode_arena_bytes>>>(cr,A,Bv[0],s.N,s.K);
            CK(cudaGetLastError());
            std::vector<bf16> base(got.size());
            CK(cudaMemcpy(base.data(),cr,base.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
            size_t mismatches = 0;
            for (size_t i = 0; i < base.size(); i++)
                mismatches += reinterpret_cast<const uint16_t*>(base.data())[i] !=
                              reinterpret_cast<const uint16_t*>(got.data())[i];
            CK(cudaFree(cr));
            printf("BK64 exact %s M=%u mismatches=%zu\n",s.name,M,mismatches);
            if (mismatches) exit(3);
        }
#endif
        constexpr int compare_rounds = 6;
        constexpr int compare_iters = 3;
        float lt_ms[compare_rounds], plow_ms[compare_rounds];
        auto time_body = [&](auto&& body) {
            cold_flush();
            CK(cudaEventRecord(e0));
            for (int i = 0; i < compare_iters; i++) body(i);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float elapsed = 0;
            CK(cudaEventElapsedTime(&elapsed, e0, e1));
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            return elapsed / compare_iters;
        };
        for (int i = 0; i < WARM; i++) { run(i, 0); run_plow(i); }
        CK(cudaDeviceSynchronize());
        for (int round = 0; round < compare_rounds; round++) {
            if ((round & 1) == 0) {
                lt_ms[round] = time_body([&](int i) { run(i, 0); });
                plow_ms[round] = time_body(run_plow);
            } else {
                plow_ms[round] = time_body(run_plow);
                lt_ms[round] = time_body([&](int i) { run(i, 0); });
            }
        }
        std::sort(lt_ms, lt_ms + compare_rounds);
        std::sort(plow_ms, plow_ms + compare_rounds);
        ms = 0.5f * (lt_ms[compare_rounds / 2 - 1] + lt_ms[compare_rounds / 2]);
        const float pms = 0.5f * (plow_ms[compare_rounds / 2 - 1] + plow_ms[compare_rounds / 2]);
        printf("%-9s %6u %8.4f %9.1f cuBLASLt rounds=%d\n", s.name, M, ms,
               fl / (ms * 1e-3) / 1e12, compare_rounds);
        printf("%-9s %6u %8.4f %9.1f Plow speedup=%.4f cold_MiB=%.1f\n", s.name, M, pms,
               fl / (pms * 1e-3) / 1e12, ms / pms, (double)nrep * wn * sizeof(bf16) / (1024 * 1024));
#ifdef PLOW_BENCH_QWEN_GEMV
        printf("bandwidth %s Lt_GBs=%.1f Plow_GBs=%.1f arena=%u M=%u\n",
            s.name, 2.0*wn/(ms*1e6), 2.0*wn/(pms*1e6), decode_arena_bytes, M);
#else
        CK(cudaFree(dm));
#endif
#if defined(PLOW_BENCH_WS384) || defined(PLOW_BENCH_GEMV_M16_BK128)
        CK(cudaFree(cp_alloc));
#else
        CK(cudaFree(cp));
#endif
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

#ifdef PLOW_BENCH_W8A8
/* ---- W8A8 route comparison ------------------------------------------------------------
 * Same shapes, same rotation/alternation/median protocol as bench_cublas, but both arms are
 * plow's W8A8 contract (DevOp::GemmFp8): C[M,N] bf16 = (A[M,K] e4m3 . B[N,K] e4m3^T) scaled
 * by a_scale[m] (per token) and w_scale[n] (per output channel).
 *   [plow]      the shipped ws384 body, E4M3 instantiation
 *   [lt_vec]    cuBLASLt e4m3 with CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F on A and B —
 *               the ONLY mode that expresses plow's scales without re-quantizing
 *   [lt_scalar] cuBLASLt e4m3 with per-tensor scales — NOT a plow twin, timed only to show
 *               what the vector scale mode costs
 * Kept as its own function rather than threaded through bench_cublas with #ifdefs, so the
 * bf16 arm stays byte-identical to the one earlier campaigns published from. */
static uint8_t* dev_e4m3(size_t n) {
    static unsigned seed = 7u;
    uint8_t* d; CK(cudaMalloc(&d, n));
    init_e4m3<<<256, 256>>>(d, n, seed++);
    CK(cudaGetLastError());
    return d;
}
/* Distinct, non-constant patterns on purpose: with a constant scale vector an A/B scale SWAP
 * between plow's (a_scale[m], w_scale[n]) and cuBLASLt's (A=weight, B=activation) is invisible
 * to the correctness gate. These two ramps are different functions of the index, so a swap
 * fails the gate instead of passing it. */
static float* dev_f32(size_t n, float base, float step) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; i++) h[i] = base * (1.f + step * (float)(i % 13));
    float* d; CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

static void bench_w8a8(unsigned M) {
    cublasLtHandle_t lt; LTK(cublasLtCreate(&lt));
    size_t wsz = 256 * 1024 * 1024; void* ws; CK(cudaMalloc(&ws, wsz));
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    const unsigned smem = PGM90_WS384_ARENA * sizeof(bf16);
    CK(cudaFuncSetAttribute(k_ws384_fp8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    static int announced = 0;
    if (!announced++)
        printf("# cuBLASLt runtime version %zu (plowrt resolves libcublasLt.so.13 first)\n",
               cublasLtGetVersion());
    for (int si = 0; si < NSHAPE; si++) {
        const Shape& s = SHAPES[si];
        const char* filter = getenv("PLOW_BENCH_SHAPE");
        if (filter && strcmp(filter, s.name)) continue;
        const size_t wn = (size_t)s.N * s.K;
        int nrep = (int)std::max<size_t>(2, ((size_t)COLD_MB << 20) / std::max<size_t>(wn, 1));
        nrep = std::min(nrep, 16);
        std::vector<uint8_t*> Bv(nrep);
        for (int r = 0; r < nrep; r++) Bv[r] = dev_e4m3(wn);
        uint8_t* A = dev_e4m3((size_t)M * s.K);
        /* plow's QuantFp8 writes a_scale[m] = rowmax|x[m,:]|/448; 1/448 is that scale for a
         * unit-magnitude row, and w_scale is the same order. Values do not affect timing. */
        float* asc = dev_f32(M, 1.f / 448.f, 0.07f);
        float* wsc = dev_f32(s.N, 1.f / 448.f, 0.23f);
        float* sca1 = dev_f32(1, 1.f / 448.f, 0.f);
        bf16 *D, *cp_alloc;
        CK(cudaMalloc(&D, (size_t)M * s.N * sizeof(bf16)));
        CK(cudaMalloc(&cp_alloc, ((size_t)M * s.N + 16) * sizeof(bf16)));
        CK(cudaMemset(cp_alloc, 0xa5, ((size_t)M * s.N + 16) * sizeof(bf16)));
        bf16* cp = cp_alloc + 8;

        cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
        LTK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, s.K, s.N, s.K)); /* Lt A = weight, M_Lt = N */
        LTK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, s.K, M,   s.K)); /* Lt B = acts,   N_Lt = M */
        LTK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF,    s.N, M,   s.N));
        cublasLtMatmulPreference_t pref = nullptr;
        LTK(cublasLtMatmulPreferenceCreate(&pref));
        LTK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;

        /* Build one descriptor per scale mode; keep both so the cost of OUTER_VEC is visible. */
        struct Arm { const char* tag; cublasLtMatmulDesc_t op; cublasLtMatmulHeuristicResult_t heur;
                     int have; int cand; };
        Arm arms[2] = {{"lt_vec", nullptr, {}, 0, 0}, {"lt_scalar", nullptr, {}, 0, 0}};
        for (int a = 0; a < 2; a++) {
            cublasLtMatmulDesc_t op = nullptr;
            LTK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
            LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
            const void* ap = (a == 0) ? (const void*)wsc : (const void*)sca1;
            const void* bp = (a == 0) ? (const void*)asc : (const void*)sca1;
            LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ap, sizeof(ap)));
            LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bp, sizeof(bp)));
            if (a == 0) {
                int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
                cublasStatus_t ra = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
                cublasStatus_t rb = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
                if (ra != CUBLAS_STATUS_SUCCESS || rb != CUBLAS_STATUS_SUCCESS) {
                    printf("%-13s %6u N=%-6u K=%-6u OUTER_VEC set-attr REFUSED a=%d b=%d\n",
                           s.name, M, s.N, s.K, (int)ra, (int)rb);
                    cublasLtMatmulDescDestroy(op);
                    continue;
                }
            }
            arms[a].op = op;
            int nres = 0;
            cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 1,
                                                               &arms[a].heur, &nres);
            arms[a].have = (hs == CUBLAS_STATUS_SUCCESS && nres > 0 &&
                            arms[a].heur.state == CUBLAS_STATUS_SUCCESS);
            arms[a].cand = nres;
            if (!arms[a].have)
                printf("%-13s %6u N=%-6u K=%-6u %-9s NOALGO status=%d n=%d\n",
                       s.name, M, s.N, s.K, arms[a].tag, (int)hs, nres);
        }

        const float alpha = 1.f, beta = 0.f;
        auto run_lt = [&](int a, int it) {
            return cublasLtMatmul(lt, arms[a].op, &alpha, Bv[it % nrep], la, A, lb,
                                  &beta, D, ld, D, ld, &arms[a].heur.algo, ws, wsz, 0);
        };
        std::vector<CUtensorMap> maps{e4m3_map(A, M, s.K)};
        for (auto b : Bv) maps.push_back(e4m3_map(b, s.N, s.K));
        CUtensorMap* dm; CK(cudaMalloc(&dm, maps.size() * sizeof(CUtensorMap)));
        CK(cudaMemcpy(dm, maps.data(), maps.size() * sizeof(CUtensorMap), cudaMemcpyHostToDevice));
        auto run_plow = [&](int it) {
            k_ws384_fp8<<<prop.multiProcessorCount, 384, smem>>>(cp, dm, dm + 1 + it % nrep,
                                                                 asc, wsc, M, s.N, s.K);
        };
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

        /* Pick each Lt arm's algorithm by TIMING up to 32 heuristic candidates, exactly as
         * bench_cublas does for bf16 — otherwise the bf16 Lt column is autotuned and the fp8 one
         * is not, and the four-way comparison silently favours bf16. */
        for (int a = 0; a < 2; a++) {
            if (!arms[a].have) continue;
            cublasLtMatmulHeuristicResult_t cands[32]; int nres = 0;
            if (cublasLtMatmulAlgoGetHeuristic(lt, arms[a].op, la, lb, ld, ld, pref, 32,
                                               cands, &nres) != CUBLAS_STATUS_SUCCESS || nres == 0)
                continue;
            float best = INFINITY; int sel = -1;
            for (int c = 0; c < nres; c++) {
                if (cands[c].state != CUBLAS_STATUS_SUCCESS) continue;
                arms[a].heur = cands[c];
                bool ok = true;
                for (int i = 0; i < WARM && ok; i++) ok = run_lt(a, i) == CUBLAS_STATUS_SUCCESS;
                if (!ok) continue;
                CK(cudaEventRecord(e0));
                for (int i = 0; i < ITERS; i++) run_lt(a, i);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
                float el; CK(cudaEventElapsedTime(&el, e0, e1));
                if (el < best) { best = el; sel = c; }
            }
            if (sel < 0) { arms[a].have = 0; continue; }
            arms[a].heur = cands[sel];
            arms[a].cand = nres;
            printf("%-13s %6u %-9s selected=%d candidates=%d workspace=%zu\n",
                   s.name, M, arms[a].tag, sel, nres, arms[a].heur.workspaceSize);
        }

        /* Correctness gate: identical e4m3 bytes and identical scale vectors through both
         * arms, so only the accumulation order differs. */
        if (arms[0].have) {
            if (run_lt(0, 0) != CUBLAS_STATUS_SUCCESS) { printf("%s lt_vec MATMUL FAILED\n", s.name); exit(3); }
            run_plow(0);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            std::vector<bf16> ref((size_t)M * s.N), got(ref.size());
            CK(cudaMemcpy(ref.data(), D, ref.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(got.data(), cp, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
            uint16_t guards[16];
            CK(cudaMemcpy(guards, cp_alloc, 8 * sizeof(bf16), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(guards + 8, cp + (size_t)M * s.N, 8 * sizeof(bf16), cudaMemcpyDeviceToHost));
            for (auto g : guards) if (g != 0xa5a5) { printf("output guard overwritten\n"); exit(3); }
            double err2 = 0, ref2 = 0, maxerr = 0, maxref = 0;
            for (size_t i = 0; i < ref.size(); i++) {
                double r = __bfloat162float(ref[i]), v = __bfloat162float(got[i]);
                if (!std::isfinite(r) || !std::isfinite(v)) { printf("nonfinite output\n"); exit(3); }
                double e = v - r;
                err2 += e * e; ref2 += r * r;
                maxerr = std::max(maxerr, std::abs(e)); maxref = std::max(maxref, std::abs(r));
            }
            const double rel = std::sqrt(err2 / std::max(ref2, 1e-30));
            printf("correctness %s M=%u relL2=%.6g max_abs=%.6g max_ref=%.6g\n",
                   s.name, M, rel, maxerr, maxref);
            /* Both arms consume identical e4m3 bytes and identical scale vectors, so only the
             * f32 accumulation order differs. Anything larger means the OUTER_VEC scale mapping
             * is wrong (an A/B swap, or a vector of the wrong length), not a rounding story. */
            if (rel > 0.01) { printf("W8A8 route mismatch: scale mapping is wrong\n"); exit(3); }
        }

        constexpr int rounds = 6, iters = 3;
        auto time_body = [&](auto&& body) {
            cold_flush();
            CK(cudaEventRecord(e0));
            for (int i = 0; i < iters; i++) body(i);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float elapsed = 0; CK(cudaEventElapsedTime(&elapsed, e0, e1));
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            return elapsed / iters;
        };
        float plow_r[rounds], vec_r[rounds], sca_r[rounds];
        for (int i = 0; i < WARM; i++) {
            run_plow(i);
            if (arms[0].have) run_lt(0, i);
            if (arms[1].have) run_lt(1, i);
        }
        CK(cudaDeviceSynchronize());
        for (int r = 0; r < rounds; r++) {
            if ((r & 1) == 0) {
                vec_r[r] = arms[0].have ? time_body([&](int i) { run_lt(0, i); }) : 0.f;
                plow_r[r] = time_body(run_plow);
                sca_r[r] = arms[1].have ? time_body([&](int i) { run_lt(1, i); }) : 0.f;
            } else {
                sca_r[r] = arms[1].have ? time_body([&](int i) { run_lt(1, i); }) : 0.f;
                plow_r[r] = time_body(run_plow);
                vec_r[r] = arms[0].have ? time_body([&](int i) { run_lt(0, i); }) : 0.f;
            }
        }
        auto med = [&](float* v) {
            std::sort(v, v + rounds);
            return 0.5f * (v[rounds / 2 - 1] + v[rounds / 2]);
        };
        const double fl = 2.0 * M * s.N * s.K;
        const float pms = med(plow_r), vms = med(vec_r), sms = med(sca_r);
        printf("%-13s %6u %6u %6u %9.5f %8.1f w8a8_plow  nrep=%d cold_MiB=%.1f\n",
               s.name, M, s.N, s.K, pms, fl / (pms * 1e-3) / 1e12, nrep,
               (double)nrep * wn / (1024 * 1024));
        if (arms[0].have)
            printf("%-13s %6u %6u %6u %9.5f %8.1f w8a8_lt_vec  plow/lt=%.4f\n",
                   s.name, M, s.N, s.K, vms, fl / (vms * 1e-3) / 1e12, pms / vms);
        if (arms[1].have)
            printf("%-13s %6u %6u %6u %9.5f %8.1f w8a8_lt_scalar  vec/scalar=%.4f\n",
                   s.name, M, s.N, s.K, sms, fl / (sms * 1e-3) / 1e12,
                   arms[0].have ? vms / sms : 0.f);

        cudaEventDestroy(e0); cudaEventDestroy(e1);
        for (int a = 0; a < 2; a++) if (arms[a].op) cublasLtMatmulDescDestroy(arms[a].op);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
        CK(cudaFree(dm)); CK(cudaFree(cp_alloc)); CK(cudaFree(D));
        CK(cudaFree(asc)); CK(cudaFree(wsc)); CK(cudaFree(sca1)); CK(cudaFree(A));
        for (int r = 0; r < nrep; r++) CK(cudaFree(Bv[r]));
    }
    cudaFree(ws); cublasLtDestroy(lt);
}
#endif

#ifdef PLOW_BENCH_GEMV_CTA
#include "bf16_gemv_cta_probe.cuh"
#endif

int main(int argc, char** argv) {
#ifdef PLOW_BENCH_GEMV_CTA
    return bench_gemv_cta(argc, argv);
#endif
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    int P = prop.multiProcessorCount;
    unsigned M = argc > 1 ? (unsigned)atoi(argv[1]) : 8192;
#ifdef PLOW_BENCH_QWEN_GEMV
#ifdef PLOW_BENCH_GEMV_M16_BK128
    const unsigned rows = argc > 1 ? M : 16u;
    if (rows != 1 && rows != 4 && rows != 16) { printf("M16 BK128 probe requires M=1/4/16\n"); return 2; }
#else
#ifdef PLOW_BENCH_GEMV_M16
    const unsigned rows = 16;
#else
    const unsigned rows = 1;
#endif
    if (argc > 1 && M != rows) { printf("GEMV mode requires M=%u\n", rows); return 2; }
#endif
    if (prop.major != 9) { printf("GEMV comparison requires Hopper\n"); return 2; }
    printf("BF16 GEMV vs cuBLASLt M=%u SMs=%d GV_UNROLL=%d\n", rows, P, GV_UNROLL);
    bench_cublas(rows);
    return 0;
#endif
#if defined(PLOW_BENCH_WS384) || defined(PLOW_BENCH_QWEN_GEMV)
#ifdef PLOW_BENCH_W8A8
#define bench_route bench_w8a8
    printf("Hopper ws384 W8A8 (e4m3, per-token x per-channel) vs cuBLASLt SMs=%d\n", P);
#else
#define bench_route bench_cublas
    printf("Hopper ws384 BF16 vs cuBLASLt SMs=%d\n", P);
#endif
    if (prop.major != 9) { printf("ws384 comparison requires Hopper\n"); return 2; }
    if (argc > 1) bench_route(M);
    else {
#ifdef PLOW_BENCH_GEMMA4_ALL
        for (unsigned rows : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u,
                              512u, 1024u, 2048u, 4096u, 8192u})
            bench_route(rows);
#else
        for (unsigned rows : {128u, 1024u, 4096u}) bench_route(rows);
#endif
    }
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

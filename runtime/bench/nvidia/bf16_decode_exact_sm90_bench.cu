// Isolated Gemma-4 H100 decode projection probe. The experimental path maps weight rows
// to WGMMA m64 and request rows to WGMMA n8/n16, with a TMA producer warpgroup and two
// consumer warpgroups. No production dispatch or defaults are changed.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#define CU(x) do { auto e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define LT(x) do { auto e = (x); if (e != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "%s:%d: cublas status %d\n", #x, __LINE__, int(e)); exit(1); } } while (0)
#define DR(x) do { auto e = (x); if (e != CUDA_SUCCESS) { \
    const char* text; cuGetErrorString(e, &text); \
    fprintf(stderr, "%s:%d: %s\n", #x, __LINE__, text); exit(1); } } while (0)
using bf16 = __nv_bfloat16;

#define PLOW_NV_HOPPER 1
#define PLOW_NV_GEMMA 1
#define GV_MM_MAX 8
#include "../../nvidia/sm90_wgmma.cuh"
#include "../../nvidia/op_gemm.cuh"
#include "../../nvidia/op_gemm_splitk.cuh"
#include "../../nvidia/op_gemv_transposed.cuh"

constexpr int WG_BN = 128, WG_BK = 64, WG_STAGES = 5;

__device__ __forceinline__ void decode_m64n8(float* d, uint64_t a, uint64_t b, int scale) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0; }\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a), "l"(b), "r"(scale) : "memory");
}

__device__ __forceinline__ void decode_m64n16(float* d, uint64_t a, uint64_t b, int scale) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0; }\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
          "+f"(d[6]), "+f"(d[7])
        : "l"(a), "l"(b), "r"(scale) : "memory");
}

template<int CAP, bool SPLIT, bool SMR>
__global__ void __maxnreg__(128) transposed_wgmma(
        bf16* out, float* partial, const CUtensorMap* xmap, const CUtensorMap* wmap,
        int M, int N, int K, int splits) {
    static_assert(CAP == 8 || CAP == 16);
    constexpr int WE = WG_BN * WG_BK, XE = CAP * WG_BK;
    constexpr int TX = (WE + XE) * sizeof(bf16);
    extern __shared__ char raw[];
    uint64_t* full = reinterpret_cast<uint64_t*>(raw);
    uint64_t* empty = full + WG_STAGES;
    bf16* ws = reinterpret_cast<bf16*>(sm90_align1024(empty + WG_STAGES));
    bf16* xs = ws + WG_STAGES * WE;
    int tid = threadIdx.x, wg = tid >> 7;
    if (tid < WG_STAGES) {
        sm90_mbar_init(full + tid, 1);
        sm90_mbar_init(empty + tid, 2);
    }
    __syncthreads();
    int all = K / WG_BK, per = (all + splits - 1) / splits;
    int first = blockIdx.y * per, last = min(first + per, all);
    if (wg == 0) {
        if constexpr (SMR) asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
        if (tid == 0) for (int step = first, st = 0; step < last; ++step, ++st) {
            int s = st % WG_STAGES;
            if (st >= WG_STAGES) sm90_mbar_wait(empty + s, ((st / WG_STAGES) + 1) & 1);
            sm90_mbar_expect(full + s, TX);
            uint32_t bar = sm90_su32(full + s);
            sm90_tma2d(sm90_su32(ws + s * WE), wmap, step * WG_BK,
                       blockIdx.x * WG_BN, bar);
            sm90_tma2d(sm90_su32(xs + s * XE), xmap, step * WG_BK, 0, bar);
        }
    } else {
        if constexpr (SMR) asm volatile("setmaxnreg.inc.sync.aligned.u32 176;" ::: "memory");
        int cwg = wg - 1, lt = tid & 127, warp = lt >> 5, lane = lt & 31;
        float acc[CAP / 2] = {};
        int prev = -1;
        for (int step = first, st = 0; step < last; ++step, ++st) {
            int s = st % WG_STAGES;
            sm90_mbar_wait(full + s, (st / WG_STAGES) & 1);
            const bf16* sw = ws + s * WE + cwg * 64 * WG_BK;
            const bf16* sx = xs + s * XE;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int scale = (st == 0 && sub == 0) ? 0 : 1;
                if constexpr (CAP == 8) decode_m64n8(acc, sm90_desc(sw + sub * 16),
                                                     sm90_desc(sx + sub * 16), scale);
                else decode_m64n16(acc, sm90_desc(sw + sub * 16),
                                   sm90_desc(sx + sub * 16), scale);
            }
            sm90_wg_commit(); sm90_wg_wait<1>();
            if (prev >= 0 && lt == 0) sm90_mbar_arrive(empty + prev);
            prev = s;
        }
        sm90_wg_wait<0>();
        if (prev >= 0 && lt == 0) sm90_mbar_arrive(empty + prev);
#pragma unroll
        for (int g = 0; g < CAP / 8; ++g) for (int hi = 0; hi < 2; ++hi)
            for (int lo = 0; lo < 2; ++lo) {
                int n = blockIdx.x * WG_BN + cwg * 64 + sm90_acc_row(warp, lane, hi);
                int m = sm90_acc_col(g, lane, lo);
                if (m < M && n < N) {
                    float v = acc[sm90_acc_reg(g, hi, lo)];
                    if constexpr (SPLIT) partial[(size_t(blockIdx.y) * M + m) * N + n] = v;
                    else out[size_t(m) * N + n] = __float2bfloat16(v);
                }
            }
    }
}

template<int RM, int BK, int STAGES, bool SPLIT>
__global__ __launch_bounds__(128) void transposed_tc(
        bf16* out, float* partial, const bf16* x, const bf16* w,
        int M, int N, int K, int splits) {
    extern __shared__ bf16 sm[];
    d_gemv_transposed_tc<RM, BK, STAGES, SPLIT>(out, partial, x, w, M, N, K, splits, sm);
}
__global__ void reduce_parts(bf16* out, const float* partial, int count, int splits) {
    d_gemv_reduce_parts(out, partial, count, splits);
}
__global__ void control_tc(float* p, const bf16* x, const bf16* w, int M, int N, int K, int splits) {
    extern __shared__ char control_sm[];
    d_gemm_splitk<16,64,128,4,3>(p, x, w, M, N, K, splits, blockIdx.x, gridDim.x, control_sm);
}

__global__ __launch_bounds__(256) void native_gemv(bf16* y, const bf16* x, const bf16* w, int M, int N, int K) {
    d_gemv(y, x, w, M, N, K, blockIdx.x, gridDim.x);
}

struct Variant { std::string name; std::function<void()> run; };
template<int RM, int BK, int STAGES>
void add_variant(std::vector<Variant>& variants, bf16* y, float* part,
                 const bf16* x, const bf16* w, int M, int N, int K, int splits) {
    size_t smem = STAGES * (64 + RM) * (BK + 8) * sizeof(bf16);
    auto kernel = splits == 1 ? transposed_tc<RM,BK,STAGES,false> : transposed_tc<RM,BK,STAGES,true>;
    CU(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    auto name = "transpose_bk" + std::to_string(BK) + "_d" + std::to_string(STAGES) + "_s" + std::to_string(splits);
    variants.push_back({name, [=] {
        kernel<<<dim3((N + 63) / 64, splits),128,smem>>>(y, part, x, w, M, N, K, splits);
        if (splits > 1) reduce_parts<<<(M*N + 255)/256,256>>>(y, part, M*N, splits);
    }});
}

static CUtensorMap make_map(void* base, int rows, int K, int box_rows) {
    CUtensorMap map{};
    uint64_t dims[2] = {uint64_t(K), uint64_t(rows)};
    uint64_t strides[1] = {uint64_t(K) * sizeof(bf16)};
    uint32_t box[2] = {WG_BK, uint32_t(box_rows)}, estride[2] = {1, 1};
    DR(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, dims, strides,
                              box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                              CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

template<int CAP, bool SMR>
void add_wgmma(std::vector<Variant>& variants, bf16* y, float* part,
               const CUtensorMap* maps, int M, int N, int K, int splits) {
    constexpr size_t smem = 2 * WG_STAGES * sizeof(uint64_t) + 1023 +
                            WG_STAGES * (WG_BN + CAP) * WG_BK * sizeof(bf16);
    auto kernel = splits == 1 ? transposed_wgmma<CAP,false,SMR>
                              : transposed_wgmma<CAP,true,SMR>;
    CU(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    std::string name = std::string(SMR ? "wgmma_smr_s" : "wgmma_free_s") +
                       std::to_string(splits);
    variants.push_back({name, [=] {
        kernel<<<dim3((N + WG_BN - 1) / WG_BN, splits),384,smem>>>(
            y, part, maps, maps + 1, M, N, K, splits);
        if (splits > 1) reduce_parts<<<(M*N + 255)/256,256>>>(y, part, M*N, splits);
    }});
}

constexpr int GUARD = 64;
static void reset_guarded(bf16* allocation, size_t count) {
    CU(cudaMemset(allocation, 0xa5, (count + 2 * GUARD) * sizeof(bf16)));
}
static void check_guarded(const char* name, bf16* allocation, size_t count) {
    std::array<uint16_t, 2 * GUARD> h{};
    CU(cudaMemcpy(h.data(), allocation, GUARD * sizeof(bf16), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(h.data() + GUARD, allocation + GUARD + count, GUARD * sizeof(bf16),
                  cudaMemcpyDeviceToHost));
    for (uint16_t v : h) if (v != 0xa5a5u) {
        fprintf(stderr, "%s overwrote output canary\n", name); exit(3);
    }
}
__global__ void random_values(bf16* p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        unsigned z = unsigned(i) + seed;
        z = (z ^ (z >> 16)) * 0x7feb352dU; z = (z ^ (z >> 15)) * 0x846ca68bU; z ^= z >> 16;
        p[i] = __float2bfloat16((int(z & 65535) - 32768) / 327680.f);
    }
}
// Reading an unrelated 192 MiB buffer displaces weights without counting the
// eviction kernel in the projection timing. CUDA events include split reduction.
__global__ void evict(const unsigned* p, unsigned* sink, size_t n) {
    unsigned v = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) v ^= p[i];
    atomicXor(sink, v);
}
double measure(const std::function<void()>& run, const unsigned* flush, unsigned* sink, int reps) {
    cudaEvent_t a, b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
    std::vector<float> samples;
    for (int i = -3; i < reps; ++i) {
        evict<<<528,256>>>(flush, sink, (192ull << 20) / 4);
        CU(cudaEventRecord(a)); run(); CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
        float ms; CU(cudaEventElapsedTime(&ms, a, b)); if (i >= 0) samples.push_back(ms);
    }
    CU(cudaEventDestroy(a)); CU(cudaEventDestroy(b));
    std::sort(samples.begin(), samples.end()); return samples[samples.size()/2] * 1000;
}
int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 31;
    int only_m = argc > 2 ? atoi(argv[2]) : 0;
    int only_shape = argc > 3 ? atoi(argv[3]) : -1;
    const bool gemma4 = true;
    if (reps < 3 || (only_m && only_m != 1 && only_m != 2 && only_m != 4 && only_m != 8 && only_m != 16 && only_m != 32)) return 2;
    DR(cuInit(0));
    cudaDeviceProp prop; CU(cudaGetDeviceProperties(&prop, 0));
    printf("# %s, %d SMs, cold weights, %d repetitions; synthetic BF16 inputs\n", prop.name, prop.multiProcessorCount, reps);
    unsigned *flush, *sink; CU(cudaMalloc(&flush,192ull<<20)); CU(cudaMemset(flush,0x5a,192ull<<20));
    CU(cudaMalloc(&sink,4)); CU(cudaMemset(sink,0,4));
    CUmodule module = nullptr;
    if (argc > 5) {
        DR(cuModuleLoad(&module, argv[5]));
        CUdeviceptr marker; size_t bytes; unsigned abi;
        DR(cuModuleGetGlobal(&marker, &bytes, module, "plow_gemv_transposed_abi"));
        if (bytes != sizeof(abi)) return 1;
        DR(cuMemcpyDtoH(&abi, marker, sizeof(abi)));
        if (abi != 1) return 1;
    }
    cublasLtHandle_t lt; LT(cublasLtCreate(&lt));
    void* workspace; size_t workspace_bytes = 256ull << 20; CU(cudaMalloc(&workspace, workspace_bytes));
    printf("M,N,K,variant,us,weight_GBs,lt_mismatches,max_abs_lt,cpu_max_abs,cpu_failures,repeat_mismatches\n");
    const std::vector<std::pair<int,int>> shapes = gemma4
        ? std::vector<std::pair<int,int>>{{512,3840},{2048,3840},{3840,4096},{3840,8192},
            {3840,15360},{4096,3840},{8192,3840},{15360,3840},{262144,3840}}
        : std::vector<std::pair<int,int>>{{8192,5376},{4096,5376},{16384,5376},{2048,5376},
            {21504,5376},{5376,8192},{5376,16384},{5376,21504},{83,136}};
    const std::vector<int> batches = only_m == 32 ? std::vector<int>{32} :
        (gemma4 ? std::vector<int>{1,2,4,8,16} : std::vector<int>{4,8,16});
    if (only_shape < -1 || only_shape >= int(shapes.size())) return 2;
    for (int shape = 0; shape < int(shapes.size()); ++shape) for (int M : batches) {
        if (only_shape >= 0 && shape != only_shape) continue;
        if (only_m && M != only_m) continue;
        auto& sh = shapes[shape];
        int N = sh.first, K = sh.second, count = M*N;
        bf16 *x, *w, *yalloc, *refalloc; float* partial;
        int cap = M <= 8 ? 8 : 16;
        CU(cudaMalloc(&x,size_t(cap)*K*2)); CU(cudaMemset(x,0,size_t(cap)*K*2));
        CU(cudaMalloc(&w,size_t(N)*K*2));
        CU(cudaMalloc(&yalloc,(size_t(count)+2*GUARD)*2));
        CU(cudaMalloc(&refalloc,(size_t(count)+2*GUARD)*2));
        bf16* y = yalloc + GUARD; bf16* ref = refalloc + GUARD;
        reset_guarded(yalloc,count); reset_guarded(refalloc,count);
        CU(cudaMalloc(&partial,size_t(count)*8*4));
        unsigned seed = std::getenv("PLOW_SEED") ? unsigned(std::strtoul(std::getenv("PLOW_SEED"),nullptr,10)) : 0u;
        random_values<<<528,256>>>(x,size_t(M)*K,13+seed); random_values<<<528,256>>>(w,size_t(N)*K,91+seed);
        CUtensorMap hmaps[2] = {make_map(x,cap,K,cap), make_map(w,N,K,WG_BN)};
        CUtensorMap* maps; CU(cudaMalloc(&maps,sizeof(hmaps)));
        CU(cudaMemcpy(maps,hmaps,sizeof(hmaps),cudaMemcpyHostToDevice));
        cublasLtMatmulDesc_t op; LT(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        cublasOperation_t tr = CUBLAS_OP_T; LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&tr,sizeof(tr)));
        cublasLtMatrixLayout_t wl,xl,yl;
        LT(cublasLtMatrixLayoutCreate(&wl,CUDA_R_16BF,K,N,K));
        LT(cublasLtMatrixLayoutCreate(&xl,CUDA_R_16BF,K,M,K));
        LT(cublasLtMatrixLayoutCreate(&yl,CUDA_R_16BF,N,M,N));
        cublasLtMatmulPreference_t pref; LT(cublasLtMatmulPreferenceCreate(&pref));
        LT(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)));
        cublasLtMatmulHeuristicResult_t algs[16]; int nalg;
        LT(cublasLtMatmulAlgoGetHeuristic(lt,op,wl,xl,yl,yl,pref,16,algs,&nalg));
        if (!nalg) { fprintf(stderr,"no cuBLASLt algorithm\n"); return 1; }
        float alpha=1,beta=0; int best=-1; double best_us=1e30;
        for (int a=0;a<nalg;++a) {
            if (algs[a].state != CUBLAS_STATUS_SUCCESS) continue;
            auto run=[&] { LT(cublasLtMatmul(lt,op,&alpha,w,wl,x,xl,&beta,ref,yl,ref,yl,&algs[a].algo,workspace,workspace_bytes,0)); };
            double us=measure(run,flush,sink,7); if(us<best_us){best_us=us;best=a;}
        }
        if(best<0) return 1;
        std::vector<Variant> variants;
        variants.push_back({"cublaslt",[&] { LT(cublasLtMatmul(lt,op,&alpha,w,wl,x,xl,&beta,y,yl,y,yl,&algs[best].algo,workspace,workspace_bytes,0)); }});
        for(int grid : {132,528}) variants.push_back({"native_gemv_grid"+std::to_string(grid),[=] {
            native_gemv<<<grid,256>>>(y,x,w,M,N,K);
        }});
        for(int s : {1,4,8}) {
            if (M <= 16) variants.push_back({"existing_s"+std::to_string(s),[=] {
                CU(cudaMemset(partial,0,size_t(count)*4));
                control_tc<<<528,128,3*(16+64)*(128+8)*2>>>(partial,x,w,M,N,K,s);
                reduce_parts<<<(count+255)/256,256>>>(y,partial,count,1);
            }});
            if(M<=8) { add_variant<8,128,3>(variants,y,partial,x,w,M,N,K,s); add_variant<8,256,2>(variants,y,partial,x,w,M,N,K,s); }
            else if (M<=16) { add_variant<16,128,3>(variants,y,partial,x,w,M,N,K,s); add_variant<16,256,2>(variants,y,partial,x,w,M,N,K,s); }
            else { add_variant<32,128,3>(variants,y,partial,x,w,M,N,K,s); add_variant<32,256,2>(variants,y,partial,x,w,M,N,K,s); }
            if (module) for (int bk : {128,256}) {
                int rm = M <= 8 ? 8 : M <= 16 ? 16 : 32, stages = bk == 128 ? 3 : 2;
                std::string symbol = "plow_gemv_bf16_m" + std::to_string(rm) +
                    "_bk" + std::to_string(bk) + "_s" + std::to_string(stages);
                CUfunction kernel, reduce;
                DR(cuModuleGetFunction(&kernel, module, symbol.c_str()));
                DR(cuModuleGetFunction(&reduce, module, "plow_gemv_bf16_reduce"));
                unsigned smem = stages * (64 + rm) * (bk + 8) * sizeof(bf16);
                DR(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
                auto name = "driver_transpose_bk" + std::to_string(bk) + "_d" +
                    std::to_string(stages) + "_s" + std::to_string(s);
                variants.push_back({name,[=]() mutable {
                    void* args[] = {&y,&partial,&x,&w,&M,&N,&K,&s};
                    DR(cuLaunchKernel(kernel,(N+63)/64,s,1,128,1,1,smem,nullptr,args,nullptr));
                    if (s > 1) {
                        void* reduce_args[] = {&y,&partial,&count,&s};
                        DR(cuLaunchKernel(reduce,(count+255)/256,1,1,256,1,1,0,nullptr,reduce_args,nullptr));
                    }
                }});
            }
        }
        for(int s : {1,2,4,8}) {
            if(M<=8) { add_wgmma<8,false>(variants,y,partial,maps,M,N,K,s);
                       add_wgmma<8,true>(variants,y,partial,maps,M,N,K,s); }
            else { add_wgmma<16,false>(variants,y,partial,maps,M,N,K,s);
                   add_wgmma<16,true>(variants,y,partial,maps,M,N,K,s); }
        }
        CU(cudaFuncSetAttribute(control_tc,cudaFuncAttributeMaxDynamicSharedMemorySize,3*(16+64)*(128+8)*2));
        reset_guarded(yalloc,count); variants[0].run(); CU(cudaDeviceSynchronize());
        check_guarded("cublaslt",yalloc,count);
        std::vector<bf16> hr(count),hy(count),hagain(count),hx(size_t(M)*K),hw(size_t(N)*K);
        CU(cudaMemcpy(hr.data(),y,count*2,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(hx.data(),x,hx.size()*2,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(hw.data(),w,hw.size()*2,cudaMemcpyDeviceToHost));
        std::vector<std::pair<int,double>> oracle;
        for(int j=0;j<128;++j){int i=int((size_t(j)*104729+17)%count); double v=0;
            for(int k=0;k<K;++k) v+=double(__bfloat162float(hx[size_t(i/N)*K+k]))*__bfloat162float(hw[size_t(i%N)*K+k]);
            oracle.push_back({i,v});}
        const char* only_variant = std::getenv("PLOW_ONLY_VARIANT");
        for(auto& v:variants) {
            if (only_variant && v.name != only_variant && v.name != "cublaslt") continue;
            reset_guarded(yalloc,count); v.run(); CU(cudaDeviceSynchronize());
            check_guarded(v.name.c_str(),yalloc,count);
            CU(cudaMemcpy(hy.data(),y,count*2,cudaMemcpyDeviceToHost));
            reset_guarded(yalloc,count); v.run(); CU(cudaDeviceSynchronize());
            check_guarded(v.name.c_str(),yalloc,count);
            CU(cudaMemcpy(hagain.data(),y,count*2,cudaMemcpyDeviceToHost));
            int mismatches=0,repeat=0,fail=0; double maxabs=0,cpumax=0;
            for(int i=0;i<count;++i){float got=__bfloat162float(hy[i]), refv=__bfloat162float(hr[i]);
                if(!std::isfinite(got)) {fprintf(stderr,"nonfinite %s\n",v.name.c_str());return 1;}
                mismatches+=got!=refv;repeat+=got!=__bfloat162float(hagain[i]);maxabs=std::max(maxabs,std::abs(double(got)-refv));}
            for(auto q:oracle){double err=std::abs(double(__bfloat162float(hy[q.first]))-q.second);cpumax=std::max(cpumax,err);
                fail+=err>0.0001+std::abs(q.second)/128;}
            double us=measure(v.run,flush,sink,reps);
            printf("%d,%d,%d,%s,%.3f,%.2f,%d,%.7g,%.7g,%d,%d\n",M,N,K,v.name.c_str(),us,double(N)*K*2/us/1000,mismatches,maxabs,cpumax,fail,repeat);fflush(stdout);
            if(fail || ((v.name.find("transpose")!=std::string::npos ||
                         v.name.find("wgmma")!=std::string::npos) && repeat)) return 1;
        }
        LT(cublasLtMatmulPreferenceDestroy(pref));LT(cublasLtMatrixLayoutDestroy(wl));LT(cublasLtMatrixLayoutDestroy(xl));
        LT(cublasLtMatrixLayoutDestroy(yl));LT(cublasLtMatmulDescDestroy(op));
        CU(cudaFree(maps));CU(cudaFree(x));CU(cudaFree(w));CU(cudaFree(yalloc));
        CU(cudaFree(refalloc));CU(cudaFree(partial));
    }
    LT(cublasLtDestroy(lt));CU(cudaFree(workspace));CU(cudaFree(flush));CU(cudaFree(sink));
    if (module) DR(cuModuleUnload(module));
}

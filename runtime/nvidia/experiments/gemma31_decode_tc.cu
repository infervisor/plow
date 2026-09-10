// Standalone H100 low-batch projection probe. Build with nvcc -O3 -arch=sm_90a
// -lcublasLt. No production dispatch or defaults are changed by this experiment.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <algorithm>
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

#define PLOW_NV_GEMMA 1
#define GV_MM_MAX 8
#include "../op_gemm.cuh"
#include "../op_gemm_splitk.cuh"

#include "../op_gemv_transposed.cuh"

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
    const bool gemma4 = argc > 4 && std::string(argv[4]) == "gemma4";
    if (argc > 4 && !gemma4) return 2;
    if (reps < 3 || (only_m && only_m != 1 && only_m != 2 && only_m != 4 && only_m != 8 && only_m != 16 && only_m != 32)) return 2;
    cudaDeviceProp prop; CU(cudaGetDeviceProperties(&prop, 0));
    printf("# %s, %d SMs, cold weights, %d repetitions; synthetic BF16 inputs\n", prop.name, prop.multiProcessorCount, reps);
    unsigned *flush, *sink; CU(cudaMalloc(&flush,192ull<<20)); CU(cudaMemset(flush,0x5a,192ull<<20));
    CU(cudaMalloc(&sink,4)); CU(cudaMemset(sink,0,4));
    CUmodule module = nullptr;
    if (argc > 5) {
        DR(cuInit(0)); DR(cuModuleLoad(&module, argv[5]));
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
            {3840,15360},{4096,3840},{8192,3840},{15360,3840},{262144,3840},{83,136}}
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
        bf16 *x, *w, *y, *ref; float* partial;
        CU(cudaMalloc(&x,size_t(M)*K*2)); CU(cudaMalloc(&w,size_t(N)*K*2));
        CU(cudaMalloc(&y,count*2)); CU(cudaMalloc(&ref,count*2)); CU(cudaMalloc(&partial,size_t(count)*8*4));
        random_values<<<528,256>>>(x,size_t(M)*K,13); random_values<<<528,256>>>(w,size_t(N)*K,91);
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
        CU(cudaFuncSetAttribute(control_tc,cudaFuncAttributeMaxDynamicSharedMemorySize,3*(16+64)*(128+8)*2));
        variants[0].run(); CU(cudaDeviceSynchronize());
        std::vector<bf16> hr(count),hy(count),hagain(count),hx(size_t(M)*K),hw(size_t(N)*K);
        CU(cudaMemcpy(hr.data(),y,count*2,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(hx.data(),x,hx.size()*2,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(hw.data(),w,hw.size()*2,cudaMemcpyDeviceToHost));
        std::vector<std::pair<int,double>> oracle;
        for(int j=0;j<128;++j){int i=int((size_t(j)*104729+17)%count); double v=0;
            for(int k=0;k<K;++k) v+=double(__bfloat162float(hx[size_t(i/N)*K+k]))*__bfloat162float(hw[size_t(i%N)*K+k]);
            oracle.push_back({i,v});}
        for(auto& v:variants) {
            v.run(); CU(cudaDeviceSynchronize()); CU(cudaMemcpy(hy.data(),y,count*2,cudaMemcpyDeviceToHost));
            v.run(); CU(cudaDeviceSynchronize()); CU(cudaMemcpy(hagain.data(),y,count*2,cudaMemcpyDeviceToHost));
            int mismatches=0,repeat=0,fail=0; double maxabs=0,cpumax=0;
            for(int i=0;i<count;++i){float got=__bfloat162float(hy[i]), refv=__bfloat162float(hr[i]);
                if(!std::isfinite(got)) {fprintf(stderr,"nonfinite %s\n",v.name.c_str());return 1;}
                mismatches+=got!=refv;repeat+=got!=__bfloat162float(hagain[i]);maxabs=std::max(maxabs,std::abs(double(got)-refv));}
            for(auto q:oracle){double err=std::abs(double(__bfloat162float(hy[q.first]))-q.second);cpumax=std::max(cpumax,err);
                fail+=err>0.0001+std::abs(q.second)/128;}
            double us=measure(v.run,flush,sink,reps);
            printf("%d,%d,%d,%s,%.3f,%.2f,%d,%.7g,%.7g,%d,%d\n",M,N,K,v.name.c_str(),us,double(N)*K*2/us/1000,mismatches,maxabs,cpumax,fail,repeat);fflush(stdout);
            if(fail || (v.name.find("transpose")!=std::string::npos && repeat)) return 1;
        }
        LT(cublasLtMatmulPreferenceDestroy(pref));LT(cublasLtMatrixLayoutDestroy(wl));LT(cublasLtMatrixLayoutDestroy(xl));
        LT(cublasLtMatrixLayoutDestroy(yl));LT(cublasLtMatmulDescDestroy(op));
        CU(cudaFree(x));CU(cudaFree(w));CU(cudaFree(y));CU(cudaFree(ref));CU(cudaFree(partial));
    }
    LT(cublasLtDestroy(lt));CU(cudaFree(workspace));CU(cudaFree(flush));CU(cudaFree(sink));
    if (module) DR(cuModuleUnload(module));
}

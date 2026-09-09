// Standalone H100 low-batch projection probe. Build with nvcc -O3 -arch=sm_90a
// -lcublasLt. No production dispatch or defaults are changed by this experiment.
#include <cuda_runtime.h>
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
using bf16 = __nv_bfloat16;

#define PLOW_NV_GEMMA 1
#define GV_MM_MAX 8
#include "../op_gemm.cuh"
#include "../op_gemm_splitk.cuh"

// Weight rows fill MMA's m16 dimension; requests occupy n8. Thus M<=8 uses
// one MMA per weight tile without padding the request dimension to 16.
template<int RM, int BK, int STAGES, bool SPLIT>
__global__ __launch_bounds__(128) void transposed_tc(
        bf16* out, float* partial, const bf16* x, const bf16* w,
        int M, int N, int K, int splits) {
    constexpr int BN = 64, XS = BK + 8, WS = BK + 8;
    extern __shared__ bf16 sm[];
    bf16* wx = sm;
    bf16* xx = sm + STAGES * BN * WS;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int nt = blockIdx.x, sp = blockIdx.y;
    int steps = (K + BK - 1) / BK, per = (steps + splits - 1) / splits;
    int first = sp * per, last = min(first + per, steps);
    float acc[RM / 8][4] = {};
    auto stage = [&](int step, int buf) {
        for (int i = tid; i < BN * (BK / 8); i += 128) {
            int row = i / (BK / 8), k = (i % (BK / 8)) * 8;
            bool valid = nt * BN + row < N && step * BK + k + 8 <= K;
            const bf16* p = valid ? w + size_t(nt * BN + row) * K + step * BK + k : w;
            pgm_cp_async_cg16(wx + (buf * BN + row) * WS + k, p, valid ? 16 : 0);
        }
        for (int i = tid; i < RM * (BK / 8); i += 128) {
            int row = i / (BK / 8), k = (i % (BK / 8)) * 8;
            bool valid = row < M && step * BK + k + 8 <= K;
            const bf16* p = valid ? x + size_t(row) * K + step * BK + k : x;
            pgm_cp_async_cg16(xx + (buf * RM + row) * XS + k, p, valid ? 16 : 0);
        }
    };
    for (int s = 0; s < STAGES - 1; ++s) {
        if (first + s < last) stage(first + s, s);
        pgm_cp_commit();
    }
    for (int i = 0; i < last - first; ++i) {
        if (first + i + STAGES - 1 < last)
            stage(first + i + STAGES - 1, (i + STAGES - 1) % STAGES);
        pgm_cp_commit();
        pgm_cp_wait<STAGES - 1>();
        __syncthreads();
        int buf = i % STAGES;
        #pragma unroll
        for (int k = 0; k < BK; k += 16) {
            unsigned a[4];
            pgm_ldmatrix_x4(a, wx + (buf * BN + warp * 16 + lane % 16) * WS + k + (lane / 16) * 8);
            #pragma unroll
            for (int r = 0; r < RM / 8; ++r) {
                unsigned b[2];
                pgm_ldmatrix_x2(b, xx + (buf * RM + r * 8 + (lane & 7)) * XS + k + ((lane >> 3) & 1) * 8);
                pgm_mma(acc[r], a, b, acc[r]);
            }
        }
        __syncthreads();
    }
    pgm_cp_wait<0>();
    #pragma unroll
    for (int r = 0; r < RM / 8; ++r) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            int n = nt * BN + warp * 16 + lane / 4 + (e / 2) * 8;
            int m = r * 8 + (lane % 4) * 2 + e % 2;
            if (m < M && n < N) {
                if constexpr (SPLIT) partial[(size_t(sp) * M + m) * N + n] = acc[r][e];
                else out[size_t(m) * N + n] = __float2bfloat16(acc[r][e]);
            }
        }
    }
}
__global__ void reduce_parts(bf16* out, const float* partial, int count, int splits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float sum = 0;
        for (int s = 0; s < splits; ++s) sum += partial[size_t(s) * count + i];
        out[i] = __float2bfloat16(sum);
    }
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
    if (reps < 3 || (only_m && only_m != 4 && only_m != 8 && only_m != 16)) return 2;
    cudaDeviceProp prop; CU(cudaGetDeviceProperties(&prop, 0));
    printf("# %s, %d SMs, cold weights, %d repetitions; synthetic BF16 inputs\n", prop.name, prop.multiProcessorCount, reps);
    unsigned *flush, *sink; CU(cudaMalloc(&flush,192ull<<20)); CU(cudaMemset(flush,0x5a,192ull<<20));
    CU(cudaMalloc(&sink,4)); CU(cudaMemset(sink,0,4));
    cublasLtHandle_t lt; LT(cublasLtCreate(&lt));
    void* workspace; size_t workspace_bytes = 256ull << 20; CU(cudaMalloc(&workspace, workspace_bytes));
    printf("M,N,K,variant,us,weight_GBs,lt_mismatches,max_abs_lt,cpu_max_abs,cpu_failures,repeat_mismatches\n");
    const int shapes[][2] = {{8192,5376},{4096,5376},{16384,5376},{2048,5376},
                            {21504,5376},{5376,8192},{5376,16384},{5376,21504},{83,136}};
    if (only_shape < -1 || only_shape >= int(sizeof(shapes)/sizeof(shapes[0]))) return 2;
    for (int shape = 0; shape < int(sizeof(shapes)/sizeof(shapes[0])); ++shape) for (int M : {4,8,16}) {
        if (only_shape >= 0 && shape != only_shape) continue;
        if (only_m && M != only_m) continue;
        auto& sh = shapes[shape];
        int N = sh[0], K = sh[1], count = M*N;
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
            variants.push_back({"existing_s"+std::to_string(s),[=] {
                CU(cudaMemset(partial,0,size_t(count)*4));
                control_tc<<<528,128,3*(16+64)*(128+8)*2>>>(partial,x,w,M,N,K,s);
                reduce_parts<<<(count+255)/256,256>>>(y,partial,count,1);
            }});
            if(M<=8) { add_variant<8,128,3>(variants,y,partial,x,w,M,N,K,s); add_variant<8,256,2>(variants,y,partial,x,w,M,N,K,s); }
            else { add_variant<16,128,3>(variants,y,partial,x,w,M,N,K,s); add_variant<16,256,2>(variants,y,partial,x,w,M,N,K,s); }
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
            if(fail || (v.name.find("transpose")==0 && repeat)) return 1;
        }
        LT(cublasLtMatmulPreferenceDestroy(pref));LT(cublasLtMatrixLayoutDestroy(wl));LT(cublasLtMatrixLayoutDestroy(xl));
        LT(cublasLtMatrixLayoutDestroy(yl));LT(cublasLtMatmulDescDestroy(op));
        CU(cudaFree(x));CU(cudaFree(w));CU(cudaFree(y));CU(cudaFree(ref));CU(cudaFree(partial));
    }
    LT(cublasLtDestroy(lt));CU(cudaFree(workspace));CU(cudaFree(flush));CU(cudaFree(sink));
}

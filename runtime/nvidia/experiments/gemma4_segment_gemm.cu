#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define PLOW_NV_PREFILL 1
#define PLOW_NV_SEGMENTS 1
#define PLOW_NV_SEG_GEMM 1
#define PLOW_NV_GEMM_ONLY 1
#define PLOW_NV_SEG_WS384 1
#define PGM90_UNI_BN256 1
#define PLOW_NV_TMA_GEMM 1
#define PLOW_NV_GEMMA 1
#define PLOW_NV_MLA 0
#define PLOW_NV_MAMBA 0
#define PLOW_NV_DSA 0
#include "dev_isa.h"
#define PLOW_NV_HOPPER 1
#define PLOW_NV_THREADS 256u
#define PGM_ARENA_BF16 (128 * 1024)
#define PLOW_ACT_SILU_ 0u
__device__ __forceinline__ float act_silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float act_gelu_tanh(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
#include "op_gemm_sm90.cuh"

using bf16 = __nv_bfloat16;
#define CK(x) do { auto e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); std::exit(2); } } while (0)
#define CD(x) do { auto e = (x); if (e != CUDA_SUCCESS) { \
    const char* text; cuGetErrorString(e,&text); std::fprintf(stderr,"%s: %s\n",#x,text); std::exit(2); } } while (0)

template<class T> static T* upload(const std::vector<T>& h) {
    T* d;
    CK(cudaMalloc(&d,h.size()*sizeof(T)));
    CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));
    return d;
}

template<bool FP8>
__global__ __maxnreg__(160) void body(bf16* out, const void* ma, const void* mb,
                                     const float* sa, const float* sb,
                                     unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 body_arena[];
    if constexpr (!FP8) { sa = nullptr; sb = nullptr; }
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        d_gemm_sm90_tma_ws384_role<true,FP8>(out,ma,mb,sa,sb,m,n,k,0,blockIdx.x,gridDim.x,body_arena);
    } else {
        sm90_reg_inc(224);
        d_gemm_sm90_tma_ws384_role<false,FP8>(out,ma,mb,sa,sb,m,n,k,0,blockIdx.x,gridDim.x,body_arena);
    }
}

static CUtensorMap make_map(void* base, unsigned rows, unsigned k, bool fp8) {
    CUtensorMap map{};
    uint64_t dims[]{k,rows}, strides[]{uint64_t(k)*(fp8?1:2)};
    uint32_t box[]{fp8?128u:64u,128}, steps[]{1,1};
    auto e = cuTensorMapEncodeTiled(&map,fp8?CU_TENSOR_MAP_DATA_TYPE_UINT8:CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,base,dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (e != CUDA_SUCCESS) { std::fprintf(stderr,"tensor map: %d\n",int(e)); std::exit(2); }
    return map;
}

int main(int argc, char** argv) {
    if (argc != 6 && (argc != 7 || std::strcmp(argv[6],"--exact-body"))) {
        std::fprintf(stderr,"usage: probe M N K bf16|fp8 interpreter.cubin [--exact-body]\n"); return 2;
    }
    const bool exact_body=argc==7;
    const unsigned m=std::strtoul(argv[1],nullptr,10), n=std::strtoul(argv[2],nullptr,10), k=std::strtoul(argv[3],nullptr,10);
    const bool fp8=std::strcmp(argv[4],"fp8")==0;
    if (!m || !n || !k || n%2 || k%128 || (!fp8 && std::strcmp(argv[4],"bf16"))) return 2;
    constexpr unsigned grid=132, threads=384;
    const unsigned smem=2*PGM90_WS384_ARENA;
    std::vector<float> ah(size_t(m)*k), bh(size_t(n)*k);
    uint32_t seed=123;
    auto fill=[&](std::vector<float>& h) -> void* {
        std::vector<bf16> b16(h.size());
        std::vector<__nv_fp8_e4m3> b8(h.size());
        for (size_t i=0;i<h.size();++i) {
            seed^=seed<<13; seed^=seed>>17; seed^=seed<<5;
            float value=float(int32_t(seed))/2147483648.0f;
            if (fp8) { b8[i]=__nv_fp8_e4m3(value); h[i]=float(b8[i]); }
            else { b16[i]=__float2bfloat16(value); h[i]=__bfloat162float(b16[i]); }
        }
        return fp8?static_cast<void*>(upload(b8)):static_cast<void*>(upload(b16));
    };
    void* a=fill(ah), *b=fill(bh);
    bf16* out;
    CK(cudaMalloc(&out,size_t(m)*n*2));
    auto maps=upload(std::vector<CUtensorMap>{make_map(a,m,k,fp8),make_map(b,n,k,fp8)});
    std::vector<float> sah(m), sbh(n);
    for (unsigned i=0;i<m;++i) sah[i]=0.5f+float((i*17)%97)/128;
    for (unsigned i=0;i<n;++i) sbh[i]=0.25f+float((i*7)%53)/64;
    float* sa=upload(sah);
    float* sb=upload(sbh);
    PlowDevInst inst{};
    inst.op=fp8?PLOW_DOP_GEMM_FP8:PLOW_DOP_GEMM;
    inst.blocks=grid;
    for (auto& t:inst.t) t=PLOW_TENSOR_NONE;
    inst.t[0]=0; inst.t[1]=1; inst.t[2]=2; inst.t[3]=3; inst.t[4]=4;
    inst.i[0]=m; inst.i[1]=n; inst.i[2]=k; inst.i[6]=5; inst.i[7]=6;
    std::vector<PlowStreamEnt> entries(grid);
    for (unsigned i=0;i<grid;++i) { entries[i].slice=i; entries[i].wait_len=1; entries[i].succ_len=1; }
    PlowProgram p{};
    p.insts=upload(std::vector<PlowDevInst>{inst});
    p.gq_stream=upload(entries);
    p.gq_seg_ofs=upload(std::vector<uint32_t>{0,grid});
    p.gq_cursor=upload(std::vector<uint32_t>(PLOW_CTR_STRIDE));
    p.waits=upload(std::vector<PlowWait>{{0,grid}});
    p.succs=upload(std::vector<uint32_t>{1});
    std::vector<uint32_t> initial(2*PLOW_CTR_STRIDE);
    initial[0]=grid;
    p.counters=upload(initial);
    p.tensors=upload(std::vector<void*>{out,a,b,sa,sb,maps,maps+1});
    const void* plain=fp8?(const void*)body<true>:(const void*)body<false>;
    CUmodule module; CUfunction interp;
    CD(cuModuleLoad(&module,argv[5]));
    CD(cuModuleGetFunction(&interp,module,"_Z19interp_sm90a_pfgemm11PlowProgram"));
    // Materialize the runtime function handle before requesting the Hopper opt-in arena.
    cudaFuncAttributes plain_attr{};
    CK(cudaFuncGetAttributes(&plain_attr, plain));
    CK(cudaFuncSetAttribute(plain,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    CD(cuFuncSetAttribute(interp,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,smem));
    void* ma=maps; void* mb=maps+1;
    unsigned mm=m,nn=n,kk=k;
    void* args[]{&out,&ma,&mb,&sa,&sb,&mm,&nn,&kk};
    void* pargs[]{&p};
    auto launch=[&](bool packet) {
        if (packet) {
            CK(cudaMemsetAsync(p.gq_cursor,0,PLOW_CTR_STRIDE*sizeof(uint32_t)));
            CK(cudaMemsetAsync(PLOW_CTR(p.counters,1),0,PLOW_CTR_STRIDE*sizeof(uint32_t)));
        }
        if(packet) CD(cuLaunchKernel(interp,grid,1,1,threads,1,1,smem,nullptr,pargs,nullptr));
        else CK(cudaLaunchKernel(plain,dim3(grid),dim3(threads),args,smem,nullptr));
    };
    bool ok=true;
    std::vector<bf16> reference;
    for (unsigned mode=0;mode<3;++mode) {
        const bool packet=mode!=0;
        if (mode==2) {
            // Independent output tiles need no dependency edges in this diagnostic program.
            for (auto& entry:entries) { entry.wait_len=0; entry.succ_len=0; }
            CK(cudaMemcpy(const_cast<PlowStreamEnt*>(p.gq_stream),entries.data(),
                          entries.size()*sizeof(PlowStreamEnt),cudaMemcpyHostToDevice));
        }
        launch(packet); CK(cudaDeviceSynchronize());
        std::vector<bf16> got(size_t(m)*n);
        CK(cudaMemcpy(got.data(),out,got.size()*2,cudaMemcpyDeviceToHost));
        if (exact_body) {
            if (!packet) reference=got;
            else if (std::memcmp(reference.data(),got.data(),got.size()*sizeof(bf16))) {
                std::fprintf(stderr,"mode %u: full output differs from standalone body\n",mode);
                ok=false;
            }
        }
        double err2=0,ref2=0,maxerr=0;
        for (unsigned i=0;i<257;++i) {
            unsigned r=(uint64_t(i)*7919)%m, c=(uint64_t(i)*104729)%n;
            double expected=0;
            for (unsigned j=0;j<k;++j) expected+=double(ah[size_t(r)*k+j])*bh[size_t(c)*k+j];
            if(fp8) expected*=double(sah[r])*sbh[c];
            double err=double(__bfloat162float(got[size_t(r)*n+c]))-expected;
            err2+=err*err; ref2+=expected*expected; maxerr=std::max(maxerr,std::abs(err));
        }
        double rel=std::sqrt(err2/std::max(ref2,1e-30));
        ok &= rel<0.004;
        for (bf16 v:got) ok &= std::isfinite(__bfloat162float(v));
        if(packet) {
            uint32_t completed;
            CK(cudaMemcpy(&completed,PLOW_CTR(p.counters,1),4,cudaMemcpyDeviceToHost));
            ok &= completed==(mode==1?grid:0);
        }
        cudaEvent_t start,stop; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
        std::vector<float> times;
        for(unsigned rep=0;rep<15;++rep) {
            // Counter reset is ordered before the event; measured duration includes only the kernel.
            if(packet) {
                CK(cudaMemsetAsync(p.gq_cursor,0,PLOW_CTR_STRIDE*4));
                CK(cudaMemsetAsync(PLOW_CTR(p.counters,1),0,PLOW_CTR_STRIDE*4));
            }
            CK(cudaEventRecord(start));
            if(packet) CD(cuLaunchKernel(interp,grid,1,1,threads,1,1,smem,nullptr,pargs,nullptr));
            else CK(cudaLaunchKernel(plain,dim3(grid),dim3(threads),args,smem,nullptr));
            CK(cudaEventRecord(stop)); CK(cudaEventSynchronize(stop));
            float ms; CK(cudaEventElapsedTime(&ms,start,stop)); times.push_back(ms*1000);
        }
        std::sort(times.begin(),times.end());
        std::printf("%u,%u,%u,%s,%s,%.3f,%.8g,%.8g,%s\n",m,n,k,argv[4],
                    mode==0?"body":mode==1?"segment":"queue",times[7],rel,maxerr,ok?"PASS":"FAIL");
        CK(cudaEventDestroy(start)); CK(cudaEventDestroy(stop));
    }
    CK(cudaDeviceReset());
    return ok?0:1;
}

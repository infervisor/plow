#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#define PLOW_NV_HOPPER 1
#define PLOW_NV_GEMMA 1
#define GV_MM_MAX 16
#define PLOW_NV_FP8_DECODE_MMA 1
#define PLOW_NV_QUANT_FP8_VLLM 1
#ifdef PLOW_FP8_W8A8_PERSISTENT_PROBE
#define PLOW_NV_FP8_DECODE_WGMMA 1
#endif
#include "op_gemm.cuh"
#include "gemma_fp8_w8a8_decode.cuh"
#include "gemma_fp8_w8a8_local.cuh"

#define CHECK(call) do { auto error = (call); if (error != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); exit(1); } } while (0)
using bf16 = __nv_bfloat16;
template <class T> T* alloc(size_t count) { T* p; CHECK(cudaMalloc(&p, count * sizeof(T))); return p; }
static uint8_t encode(float value) { __nv_fp8_e4m3 q(value); return *reinterpret_cast<uint8_t*>(&q); }
static float decode(uint8_t value) { __nv_fp8_e4m3 q; *reinterpret_cast<uint8_t*>(&q) = value; return (float)q; }

__global__ void quantize(uint8_t* q, bf16* x, float* scales, unsigned M, unsigned K) {
    d_quant_fp8(q, x, scales, M, K, 0, 1);
}
template <bool Glu, bool CombinedQkv = false>
__global__ void w8a16_control(bf16* y, const bf16* x, const uint8_t* w, const uint8_t* u,
    const float* ws, const float* us, unsigned M, unsigned N, unsigned K, unsigned first, unsigned blocks, unsigned act = PLOW_ACT_GELU_TANH_) {
    unsigned slice=first+blockIdx.x;
    if constexpr(CombinedQkv){const unsigned preceding=slice<66?0:slice<99?8192:12288;y+=(size_t)M*preceding;w+=(size_t)preceding*K;ws+=preceding;N=slice<66?8192:4096;blocks=slice<66?66:33;slice-=slice<66?0:slice<99?66:99;}
    if constexpr (Glu) d_gemv_glu_fp8(y,x,w,u,ws,us,M,N,K,act,slice,blocks);
    else d_gemv_fp8(y,x,w,ws,M,N,K,slice,blocks);
}
__global__ void flush_w8a8(unsigned* p) {
    for (unsigned i=blockIdx.x*256+threadIdx.x;i<64*1024*1024;i+=gridDim.x*256) p[i]+=i;
}

template <unsigned Rows, bool Glu, bool Promote, bool Qkv = false>
static void launch_w8a8(bf16* y, const uint8_t* x, const uint8_t* w, const uint8_t* u,
    const float* xs, const float* ws, const float* us, unsigned M, unsigned N, unsigned K,
    unsigned first, unsigned count, unsigned blocks) {
    constexpr unsigned bytes=decode_wgmma_bytes<Rows,Glu>;
    decode_w8a8<Rows,Glu,Promote,Qkv><<<count,256,bytes>>>(y,x,w,u,xs,ws,us,M,N,K,first,blocks);
}

#ifdef PLOW_FP8_W8A8_PERSISTENT_PROBE
template <bool Glu, bool Qkv = false>
__global__ void decode_w8a8_persistent(bf16* y, const bf16* x, const uint8_t* w, const uint8_t* u,
    const float* ws, const float* us, unsigned M, unsigned N, unsigned K,
    unsigned first, unsigned blocks, unsigned act = PLOW_ACT_GELU_TANH_) {
    extern __shared__ __align__(16) bf16 persistent_arena[];
    unsigned slice = first + blockIdx.x;
    if constexpr (Qkv) {
        const unsigned preceding = slice < 66 ? 0 : slice < 99 ? 8192 : 12288;
        y += (size_t)M * preceding; w += (size_t)preceding * K; ws += preceding;
        N = slice < 66 ? 8192 : 4096; blocks = slice < 66 ? 66 : 33;
        slice -= slice < 66 ? 0 : slice < 99 ? 66 : 99;
    }
    if constexpr (Glu) d_gemv_glu_fp8(y,x,w,u,ws,us,M,N,K,act,slice,blocks,persistent_arena);
    else d_gemv_fp8(y,x,w,ws,M,N,K,slice,blocks,persistent_arena);
}
#endif

template <unsigned Rows, bool Glu, bool Qkv = false>
static void launch_local(bf16* y, const bf16* x, const uint8_t* w, const uint8_t* u,
    const float* ws, const float* us, unsigned M, unsigned N, unsigned K,
    unsigned first, unsigned count, unsigned blocks) {
#ifdef PLOW_FP8_W8A8_PERSISTENT_PROBE
    decode_w8a8_persistent<Glu,Qkv><<<count,256,PLOW_NV_FP8_DECODE_WGMMA_ARENA_BYTES>>>(
        y,x,w,u,ws,us,M,N,K,first,blocks);
#else
    constexpr unsigned bytes=decode_wgmma_local_bytes<Rows,Glu>;
    decode_w8a8_local<Rows,Glu,Qkv><<<count,256,bytes>>>(
        y,x,w,u,ws,us,M,N,K,first,blocks);
#endif
}

static void run(unsigned M,unsigned N,unsigned K,unsigned blocks,bool glu,bool exact,unsigned* flush,bool qkv=false) {
    std::mt19937 rng(131+M+N+K); std::uniform_real_distribution<float> random(-1.f,1.f);
    std::vector<bf16> x((size_t)M*K),control((size_t)M*N),out(control.size()),promoted(control.size());
    std::vector<uint8_t> w((size_t)N*K),u(w.size()),q(x.size());
    std::vector<float> ws(N),us(N),xs(M);
    for(auto& v:x) v=__float2bfloat16(exact ? (int)(rng()%7)-3.f : random(rng));
    if(exact) for(unsigned m=0;m<M;++m) x[(size_t)m*K]=__float2bfloat16(448.f);
    for(auto* values:{&w,&u}) for(auto& v:*values) v=encode(exact ? (int)(rng()%7)-3.f : random(rng));
    for(unsigned n=0;n<N;++n) { ws[n]=exact?.03125f:.01f+.03f*std::abs(random(rng)); us[n]=exact?.0625f:.01f+.03f*std::abs(random(rng)); }
    auto* dx=alloc<bf16>(x.size()); auto* dq=alloc<uint8_t>(q.size()); auto* dw=alloc<uint8_t>(w.size()); auto* du=alloc<uint8_t>(u.size());
    auto* dxs=alloc<float>(M);auto* dws=alloc<float>(N);auto* dus=alloc<float>(N);auto* storage=alloc<bf16>(out.size()+128);auto* dy=storage+64;
    CHECK(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(du,u.data(),u.size(),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dws,ws.data(),N*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dus,us.data(),N*4,cudaMemcpyHostToDevice));
    auto quant=[&]{quantize<<<1,256>>>(dq,dx,dxs,M,K);CHECK(cudaGetLastError());};
    auto launch=[&](unsigned mode,unsigned first,unsigned count){
        if(!mode) { if(qkv) w8a16_control<false,true><<<count,256>>>(dy,dx,dw,du,dws,dus,M,N,K,first,blocks);else if(glu) w8a16_control<true><<<count,256>>>(dy,dx,dw,du,dws,dus,M,N,K,first,blocks);else w8a16_control<false><<<count,256>>>(dy,dx,dw,du,dws,dus,M,N,K,first,blocks); }
#define LOCAL(R,G,Q) launch_local<R,G,Q>(dy,dx,dw,du,dws,dus,M,N,K,first,count,blocks)
        else if(mode==3){if(qkv){if(M<=8)LOCAL(8,false,true);else LOCAL(16,false,true);}else if(glu){if(M<=8)LOCAL(8,true,false);else LOCAL(16,true,false);}else{if(M<=8)LOCAL(8,false,false);else LOCAL(16,false,false);}}
#undef LOCAL
#define QKV(R,P) launch_w8a8<R,false,P,true>(dy,dq,dw,du,dxs,dws,dus,M,N,K,first,count,blocks)
        else if(qkv){if(M<=8){if(mode==1)QKV(8,false);else QKV(8,true);}else{if(mode==1)QKV(16,false);else QKV(16,true);}}
#undef QKV
#define LAUNCH(R,G,P) launch_w8a8<R,G,P>(dy,dq,dw,du,dxs,dws,dus,M,N,K,first,count,blocks)
        else if(M<=8) {if(glu) {if(mode==1) LAUNCH(8,true,false);else LAUNCH(8,true,true);}else {if(mode==1) LAUNCH(8,false,false);else LAUNCH(8,false,true);}}
        else {if(glu) {if(mode==1) LAUNCH(16,true,false);else LAUNCH(16,true,true);}else {if(mode==1) LAUNCH(16,false,false);else LAUNCH(16,false,true);}}
#undef LAUNCH
        CHECK(cudaGetLastError());
    };
    quant();CHECK(cudaMemcpy(q.data(),dq,q.size(),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(xs.data(),dxs,M*4,cudaMemcpyDeviceToHost));
    unsigned qbad=0,sbad=0;
    for(unsigned m=0;m<M;++m){ float amax=0;for(unsigned k=0;k<K;++k) amax=std::max(amax,std::abs(__bfloat162float(x[(size_t)m*K+k])));
        const float scale=std::max(amax/448.f,1.f/(448.f*512.f));sbad+=scale!=xs[m];
        for(unsigned k=0;k<K;++k){float v=__bfloat162float(x[(size_t)m*K+k])/scale;qbad+=q[(size_t)m*K+k]!=encode(std::max(-448.f,std::min(448.f,v)));}}
    if(qbad||sbad){fprintf(stderr,"quantization mismatch bytes%u scales%u\n",qbad,sbad);exit(2);}
    launch(0,0,blocks);CHECK(cudaMemcpy(control.data(),dy,out.size()*2,cudaMemcpyDeviceToHost));
    launch(2,0,blocks);CHECK(cudaMemcpy(promoted.data(),dy,out.size()*2,cudaMemcpyDeviceToHost));
    cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
    for(unsigned mode=1;mode<4;++mode){
        const std::vector<unsigned> slices=exact?std::vector<unsigned>{0,blocks/2,blocks-1}:std::vector<unsigned>{0};
        for(unsigned slice:slices){CHECK(cudaMemset(storage,0x5a,(out.size()+128)*2));launch(mode,slice,exact?1:blocks);CHECK(cudaMemcpy(out.data(),dy,out.size()*2,cudaMemcpyDeviceToHost));
            std::vector<bf16> guards(128);CHECK(cudaMemcpy(guards.data(),storage,128,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(guards.data()+64,dy+out.size(),128,cudaMemcpyDeviceToHost));
            for(auto v:guards)if(*reinterpret_cast<uint16_t*>(&v)!=0x5a5a){fprintf(stderr,"guard overwrite\n");exit(2);}
            const unsigned per=(N+blocks-1)/blocks;double diff2=0,norm2=0,maxabs=0,oracle2=0,oracle_norm2=0;unsigned changed=0,samples=0,promoted_different=0;
            for(size_t i=0;i<out.size();++i){unsigned m=i/N,n=i%N;
                if(qkv){const unsigned preceding=i<(size_t)M*8192?0:i<(size_t)M*12288?8192:12288;const unsigned columns=preceding?4096:8192;const size_t local=i-(size_t)M*preceding;m=local/columns;n=preceding+local%columns;}
                const bool owned=!exact||(n>=slice*per&&n<std::min((slice+1)*per,N));
                if(!owned){if(*reinterpret_cast<uint16_t*>(&out[i])!=0x5a5a){fprintf(stderr,"slice ownership violation\n");exit(2);}continue;}
                float got=__bfloat162float(out[i]),ref=__bfloat162float(control[i]);if(!std::isfinite(got)){fprintf(stderr,"nonfinite\n");exit(2);}
                promoted_different+=*reinterpret_cast<uint16_t*>(&out[i])!=*reinterpret_cast<uint16_t*>(&promoted[i]);
                const double delta=got-ref;changed+=delta!=0;diff2+=delta*delta;norm2+=(double)ref*ref;maxabs=std::max(maxabs,std::abs(delta));
                if(exact||(i*104729%out.size())<257){double gate=0,up=0;for(unsigned k=0;k<K;++k){double a=decode(q[(size_t)m*K+k]);gate+=a*decode(w[(size_t)n*K+k]);if(glu)up+=a*decode(u[(size_t)n*K+k]);}
                    float g=ws[n]*(xs[m]*(float)gate),r=g;if(glu){float a=.5f*g*(1.f+tanhf(.7978845608028654f*(g+.044715f*g*g*g)));r=a*(us[n]*(xs[m]*(float)up));}
                    r=__bfloat162float(__float2bfloat16(r));oracle2+=(double)(got-r)*(got-r);oracle_norm2+=(double)r*r;++samples;}}
            const double relative=std::sqrt(diff2/(norm2+1e-30)),oracle=std::sqrt(oracle2/(oracle_norm2+1e-30));
            if((exact&&changed)||oracle>(mode==1?.02:.004)||(mode==3&&promoted_different)){fprintf(stderr,"oracle failure mode%u exact%u changed%u promoted_different%u rel%g\n",mode,exact,changed,promoted_different,oracle);exit(2);}
            printf("check M=%u N=%u K=%u blocks=%u glu=%u qkv=%u mode=%u slice=%u exact=%u quant_bytes_exact=1 quant_scales_exact=1 changed=%u promoted_different=%u bf16_relL2=%.9g max_abs=%.9g quantized_oracle_relL2=%.9g samples=%u PASS\n",M,N,K,blocks,glu,qkv,mode,slice,exact,changed,promoted_different,relative,maxabs,oracle,samples);
        }
    }
    if(!exact)for(unsigned mode=0;mode<4;++mode)for(unsigned include_quant=0;include_quant<2;++include_quant){if((!mode&&include_quant)||(mode==3&&!include_quant))continue;
        for(unsigned rep=0;rep<4;++rep){if(include_quant&&mode!=3)quant();launch(mode,0,blocks);}CHECK(cudaDeviceSynchronize());std::vector<float> times;
        for(unsigned rep=0;rep<15;++rep){flush_w8a8<<<528,256>>>(flush);CHECK(cudaEventRecord(begin));if(include_quant&&mode!=3)quant();launch(mode,0,blocks);CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));times.push_back(ms*1000);}
        std::sort(times.begin(),times.end());printf("time M=%u N=%u K=%u blocks=%u glu=%u qkv=%u mode=%u quant_included=%u median_us=%.3f min_us=%.3f max_us=%.3f\n",M,N,K,blocks,glu,qkv,mode,include_quant,times[7],times.front(),times.back());fflush(stdout);}
    CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaFree(dx));CHECK(cudaFree(dq));CHECK(cudaFree(dw));CHECK(cudaFree(du));CHECK(cudaFree(dxs));CHECK(cudaFree(dws));CHECK(cudaFree(dus));CHECK(cudaFree(storage));
}

#ifdef PLOW_FP8_W8A8_PERSISTENT_PROBE
static void check_fallback(unsigned M, unsigned K, bool glu, unsigned act) {
    constexpr unsigned N=79, blocks=13;
    std::mt19937 rng(M+K+act); std::uniform_real_distribution<float> random(-1.f,1.f);
    std::vector<bf16> x((size_t)M*K), ref((size_t)M*N), out(ref.size());
    std::vector<uint8_t> w((size_t)N*K), u(w.size()); std::vector<float> scale(N,.03125f);
    for(auto& v:x)v=__float2bfloat16(random(rng));
    for(auto* values:{&w,&u})for(auto& v:*values)v=encode(random(rng));
    auto* dx=alloc<bf16>(x.size());auto* dw=alloc<uint8_t>(w.size());auto* du=alloc<uint8_t>(u.size());
    auto* ds=alloc<float>(N);auto* dy=alloc<bf16>(out.size());
    CHECK(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(du,u.data(),u.size(),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ds,scale.data(),N*4,cudaMemcpyHostToDevice));
    if(glu)w8a16_control<true><<<blocks,256>>>(dy,dx,dw,du,ds,ds,M,N,K,0,blocks,act);
    else w8a16_control<false><<<blocks,256>>>(dy,dx,dw,du,ds,ds,M,N,K,0,blocks,act);
    CHECK(cudaGetLastError());CHECK(cudaMemcpy(ref.data(),dy,ref.size()*2,cudaMemcpyDeviceToHost));
    CHECK(cudaMemset(dy,0x5a,out.size()*2));
    if(glu)decode_w8a8_persistent<true><<<blocks,256,PLOW_NV_FP8_DECODE_WGMMA_ARENA_BYTES>>>(dy,dx,dw,du,ds,ds,M,N,K,0,blocks,act);
    else decode_w8a8_persistent<false><<<blocks,256,PLOW_NV_FP8_DECODE_WGMMA_ARENA_BYTES>>>(dy,dx,dw,du,ds,ds,M,N,K,0,blocks,act);
    CHECK(cudaGetLastError());CHECK(cudaMemcpy(out.data(),dy,out.size()*2,cudaMemcpyDeviceToHost));
    unsigned different=0;
    for(size_t i=0;i<out.size();++i)different+=*reinterpret_cast<uint16_t*>(&out[i])!=*reinterpret_cast<uint16_t*>(&ref[i]);
    if(different){fprintf(stderr,"fallback mismatch M%u K%u glu%u act%u count%u\n",M,K,glu,act,different);exit(2);}
    printf("fallback M=%u N=%u K=%u glu=%u act=%u different=0 PASS\n",M,N,K,glu,act);
    CHECK(cudaFree(dx));CHECK(cudaFree(dw));CHECK(cudaFree(du));CHECK(cudaFree(ds));CHECK(cudaFree(dy));
}
#endif

int main(int argc,char** argv){
    if(argc!=1&&(argc!=6||atoi(argv[1])<1||atoi(argv[1])>16||atoi(argv[2])<1||atoi(argv[3])<16||atoi(argv[3])%16||atoi(argv[4])<1||(atoi(argv[5])!=0&&atoi(argv[5])!=1))){fprintf(stderr,"usage: %s [M=1..16 N>0 K=multiple-of-16 blocks>0 glu=0|1]\n",argv[0]);return 1;}
#define ATTR(R,G,P) CHECK(cudaFuncSetAttribute(decode_w8a8<R,G,P>,cudaFuncAttributeMaxDynamicSharedMemorySize,decode_wgmma_bytes<R,G>))
    ATTR(8,false,false);ATTR(8,false,true);ATTR(8,true,false);ATTR(8,true,true);ATTR(16,false,false);ATTR(16,false,true);ATTR(16,true,false);ATTR(16,true,true);
#undef ATTR
#define QATTR(R,P) CHECK(cudaFuncSetAttribute(decode_w8a8<R,false,P,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,decode_wgmma_bytes<R,false>))
    QATTR(8,false);QATTR(8,true);QATTR(16,false);QATTR(16,true);
#undef QATTR
#define LATTR(R,G,Q) CHECK(cudaFuncSetAttribute(decode_w8a8_local<R,G,Q>,cudaFuncAttributeMaxDynamicSharedMemorySize,decode_wgmma_local_bytes<R,G>))
    LATTR(8,false,false);LATTR(8,true,false);LATTR(16,false,false);LATTR(16,true,false);LATTR(8,false,true);LATTR(16,false,true);
#undef LATTR
#ifdef PLOW_FP8_W8A8_PERSISTENT_PROBE
#define PATTR(G,Q) CHECK(cudaFuncSetAttribute(decode_w8a8_persistent<G,Q>,cudaFuncAttributeMaxDynamicSharedMemorySize,PLOW_NV_FP8_DECODE_WGMMA_ARENA_BYTES))
    PATTR(false,false);PATTR(true,false);PATTR(false,true);
#undef PATTR
#endif
    auto* flush=alloc<unsigned>(64*1024*1024);CHECK(cudaMemset(flush,0,256*1024*1024));
#ifdef PLOW_FP8_W8A8_PERSISTENT_PROBE
    for(unsigned M:{1u,2u,4u})for(bool glu:{false,true})run(M,71,256,13,glu,true,flush);
    for(unsigned M:{1u,2u,4u,8u,16u})for(bool glu:{false,true})check_fallback(M,136,glu,PLOW_ACT_GELU_TANH_);
    for(unsigned M:{8u,16u})check_fallback(M,256,true,PLOW_ACT_SILU_);
#endif
    for(unsigned M:{7u,8u,15u,16u})for(bool glu:{false,true})run(M,71,256,13,glu,true,flush);
    run(8,71,272,132,false,true,flush);
    if(argc==6)run(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),false,flush);
    else if(argc==1)for(unsigned M:{8u,16u}){run(M,8192,5376,66,false,false,flush);run(M,4096,5376,33,false,false,flush);run(M,16384,5376,132,false,false,flush,true);run(M,5376,8192,132,false,false,flush);run(M,21504,5376,132,true,false,flush);run(M,5376,21504,132,false,false,flush);}
    CHECK(cudaFree(flush));
}

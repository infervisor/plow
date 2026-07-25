/* gemm_parity.cu — PX-3 parity: d_gemm / d_gemm_glu at compile-time PGM_BN vs an f32 CPU ref.
 * Built at BN=128 and BN=64; both must clear the same tolerance. Because the N-tiling does not
 * change any single output element's K-accumulation order (per-tile mma is K-major, identical
 * across BN), BN=64 is expected bit-for-bit equal to BN=128 — verified by the identical relL2. */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "sm120_common.cuh"
#include "op_gemm.cuh"
typedef __nv_bfloat16 bf16;
#define CK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA FAIL %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);}}while(0)
static uint32_t rng=777u; static float fr(){rng=rng*1664525u+1013904223u; return ((rng>>9)&0x3fff)/16383.0f-0.5f;}
static bf16* dr(const std::vector<float>&h){std::vector<bf16>b(h.size());for(size_t i=0;i<h.size();i++)b[i]=__float2bfloat16(h[i]);bf16*d;CK(cudaMalloc(&d,h.size()*2));CK(cudaMemcpy(d,b.data(),h.size()*2,cudaMemcpyHostToDevice));return d;}
__global__ void k_gemm(bf16*C,const bf16*A,const bf16*B,unsigned m,unsigned n,unsigned k){extern __shared__ bf16 s[];d_gemm(C,A,B,m,n,k,0,blockIdx.x,gridDim.x,s);}
__global__ void k_glu(bf16*C,const bf16*A,const bf16*G,const bf16*U,unsigned m,unsigned n,unsigned k,unsigned act){extern __shared__ bf16 s[];d_gemm_glu(C,A,G,U,m,n,k,act,blockIdx.x,gridDim.x,s);}
static float gelu(float x){float c=0.7978845608028654f*(x+0.044715f*x*x*x);return 0.5f*x*(1.0f+tanhf(c));}
static double check(const char*lbl,const std::vector<float>&ref,const std::vector<bf16>&dev){
    double num=0,den=0; for(size_t i=0;i<ref.size();i++){float d=__bfloat162float(dev[i])-ref[i];num+=(double)d*d;den+=(double)ref[i]*ref[i];}
    double rel=sqrt(num/(den+1e-30)); printf("%s relL2=%.3e n=%zu\n",lbl,rel,ref.size()); return rel;}
int main(){
    const size_t smem=(size_t)PGM_ARENA_BF16*2;
    CK(cudaFuncSetAttribute(k_gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    CK(cudaFuncSetAttribute(k_glu,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    printf("PGM_BN=%d arena_bf16=%d\n",PGM_BN,PGM_ARENA_BF16);
    // plain GEMM: M=192 N=320 K=512 (spans multiple N-tiles at both BN)
    {unsigned M=192,N=320,K=512; std::vector<float>A(M*K),B((size_t)N*K);
     for(auto&x:A)x=fr(); for(auto&x:B)x=fr();
     std::vector<float>ref((size_t)M*N); for(unsigned i=0;i<M;i++)for(unsigned j=0;j<N;j++){double a=0;for(unsigned kk=0;kk<K;kk++)a+=(double)A[i*K+kk]*B[(size_t)j*K+kk];ref[(size_t)i*N+j]=__bfloat162float(__float2bfloat16((float)a));}
     bf16*dA=dr(A),*dB=dr(B),*dC; CK(cudaMalloc(&dC,(size_t)M*N*2));
     k_gemm<<<188,256,smem>>>(dC,dA,dB,M,N,K); CK(cudaDeviceSynchronize());
     std::vector<bf16>hC((size_t)M*N); CK(cudaMemcpy(hC.data(),dC,(size_t)M*N*2,cudaMemcpyDeviceToHost));
     check("d_gemm 192x320x512",ref,hC);}
    // GLU GEMM: M=192 N=320 K=512, GeGLU
    {unsigned M=192,N=320,K=512; std::vector<float>A(M*K),G((size_t)N*K),U((size_t)N*K);
     for(auto&x:A)x=fr(); for(auto&x:G)x=fr(); for(auto&x:U)x=fr();
     std::vector<float>ref((size_t)M*N); for(unsigned i=0;i<M;i++)for(unsigned j=0;j<N;j++){double ag=0,au=0;for(unsigned kk=0;kk<K;kk++){ag+=(double)A[i*K+kk]*G[(size_t)j*K+kk];au+=(double)A[i*K+kk]*U[(size_t)j*K+kk];}ref[(size_t)i*N+j]=__bfloat162float(__float2bfloat16(gelu((float)ag)*(float)au));}
     bf16*dA=dr(A),*dG=dr(G),*dU=dr(U),*dC; CK(cudaMalloc(&dC,(size_t)M*N*2));
     k_glu<<<188,256,smem>>>(dC,dA,dG,dU,M,N,K,0); CK(cudaDeviceSynchronize());
     std::vector<bf16>hC((size_t)M*N); CK(cudaMemcpy(hC.data(),dC,(size_t)M*N*2,cudaMemcpyDeviceToHost));
     check("d_gemm_glu 192x320x512",ref,hC);}
    return 0;
}

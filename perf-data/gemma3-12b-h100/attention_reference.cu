#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "sm120_common.cuh"
using bf16 = __nv_bfloat16;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); exit(2); } } while(0)
__global__ void run(bf16* o,const bf16* q,const bf16* k,const bf16* v,unsigned rows,unsigned ctx,unsigned win) {
  extern __shared__ float sm[];
  d_flash_prefill<256,64,32>(nullptr,nullptr,q,k,v,o,rows,ctx,16,8,ctx-rows,win,1,ctx,0xFFFFFFFFu,0.0625f,blockIdx.x,gridDim.x,sm);
}
static void fill(std::vector<bf16>& x,unsigned seed) {
  for(size_t i=0;i<x.size();++i) {
    unsigned a=(unsigned)i+seed; a=(a^(a>>16))*0x7feb352du; a=(a^(a>>15))*0x846ca68bu;
    x[i]=__float2bfloat16(((int)(a&2047)-1024)/1024.f);
  }
}
int main() {
  const size_t smem=FA_PRE_SMEM_FLOATS(256,64,32)*sizeof(float);
  CK(cudaFuncSetAttribute(run,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  for(unsigned rows: {128u,1024u}) for(unsigned ctx: {1024u,16384u}) for(unsigned win: {0u,1024u}) {
    std::vector<bf16> q((size_t)rows*16*256),k((size_t)ctx*8*256),v(k.size()),o(q.size());
    fill(q,17);fill(k,911);fill(v,1234);
    bf16 *dq,*dk,*dv,*doo;
    CK(cudaMalloc(&dq,q.size()*2));CK(cudaMalloc(&dk,k.size()*2));CK(cudaMalloc(&dv,v.size()*2));CK(cudaMalloc(&doo,o.size()*2));
    CK(cudaMemcpy(dq,q.data(),q.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk,k.data(),k.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv,v.data(),v.size()*2,cudaMemcpyHostToDevice));
    run<<<132,256,smem>>>(doo,dq,dk,dv,rows,ctx,win);CK(cudaGetLastError());
    CK(cudaMemcpy(o.data(),doo,o.size()*2,cudaMemcpyDeviceToHost));
    for(auto x:o) if(!std::isfinite(__bfloat162float(x))) return 3;
    double err=0,refnorm=0,maxerr=0;
    for(unsigned row: {0u,31u,rows-1}) for(unsigned head: {0u,1u,15u}) {
      unsigned end=ctx-rows+row+1,start=win&&end>win?end-win:0;
      std::vector<double> scores(end-start);double peak=-1e30,sum=0;
      for(unsigned pos=start;pos<end;++pos) {
        double s=0;
        for(unsigned d=0;d<256;++d) s+=(double)__bfloat162float(q[((size_t)row*16+head)*256+d])*__bfloat162float(k[((size_t)(head/2)*ctx+pos)*256+d]);
        scores[pos-start]=s/16.;peak=std::max(peak,s/16.);
      }
      for(double& s:scores) {s=std::exp(s-peak);sum+=s;}
      for(unsigned d=0;d<256;++d) {
        double ref=0;
        for(unsigned pos=start;pos<end;++pos) ref+=scores[pos-start]*__bfloat162float(v[((size_t)(head/2)*ctx+pos)*256+d]);
        ref/=sum;double diff=__bfloat162float(o[((size_t)row*16+head)*256+d])-ref;
        err+=diff*diff;refnorm+=ref*ref;maxerr=std::max(maxerr,std::abs(diff));
      }
    }
    double rel=std::sqrt(err/refnorm);
    printf("rows=%u ctx=%u window=%u sampled_vectors=9 rel_l2=%.8g max_abs=%.8g\n",rows,ctx,win,rel,maxerr);fflush(stdout);
    if(rel>0.01 || maxerr>0.01) return 4;
    CK(cudaFree(dq));CK(cudaFree(dk));CK(cudaFree(dv));CK(cudaFree(doo));
  }
}

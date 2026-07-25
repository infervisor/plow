// Does setmaxnreg.dec raise achieved blocks/SM?  A/B differs ONLY in the operand,
// so both kernels get identical ptxas register allocation (128).
#include <cstdio>
#include <cstdlib>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);} }while(0)

template<int N>
__global__ void __maxnreg__(128) occ(int* live,int* maxlive){
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;"::"n"(N));
  if (threadIdx.x==0){ int n=atomicAdd(live,1)+1; atomicMax(maxlive,n); }
  __syncthreads();
  long long t0=clock64(); while(clock64()-t0 < 60000000LL);
  __syncthreads();
  if (threadIdx.x==0) atomicSub(live,1);
}
template<int N> void run(cudaDeviceProp&p,int*lv,int*mx){
  int occpred; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occpred,(void*)occ<N>,256,0));
  CK(cudaMemset(lv,0,4)); CK(cudaMemset(mx,0,4));
  occ<N><<<p.multiProcessorCount*16,256>>>(lv,mx);
  CK(cudaDeviceSynchronize());
  int h; CK(cudaMemcpy(&h,mx,4,cudaMemcpyDeviceToHost));
  printf("dec->%3d regs : cudaOccupancyMaxActiveBlocksPerSM=%d | measured max concurrent=%d (%.2f blk/SM)\n",
         N,occpred,h,(double)h/p.multiProcessorCount);
}
int main(){ cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("%s SMs=%d regsPerSM=%d  block=256 thr, __maxnreg__(128)\n",p.name,p.multiProcessorCount,p.regsPerMultiprocessor);
  int *lv,*mx; CK(cudaMalloc(&lv,4)); CK(cudaMalloc(&mx,4));
  run<128>(p,lv,mx); run<64>(p,lv,mx); run<24>(p,lv,mx);
  return 0;}

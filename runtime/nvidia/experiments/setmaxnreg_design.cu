// The actionable question: can a LOW entry __maxnreg__ (high occupancy) still give
// compute warps a HIGH register count, by having loader warps donate?
// 256 thr, __maxnreg__(64) -> CTA pool = 64*256 = 16384 regs -> 4 blocks/SM (vs 2 at 128).
// WG0 (loader) dec 64->24  : uses 24*128  =  3072
// WG1 (compute) inc 64->104: uses 104*128 = 13312   ; total = 16384 exact fit.
#include <cstdio>
#include <cstdlib>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);} }while(0)

__global__ void __maxnreg__(64) split(int* live,int* maxlive,float* out,const float* a){
  if (threadIdx.x < 128) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;"::"n"(24));
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"::"n"(104));
    float acc[80];
    #pragma unroll
    for(int i=0;i<80;i++) acc[i]=a[i];
    for(int k=0;k<64;k++){
      #pragma unroll
      for(int i=0;i<80;i++) acc[i]=fmaf(acc[i],1.0000001f,a[k&31]);
    }
    float s=0;
    #pragma unroll
    for(int i=0;i<80;i++) s+=acc[i];
    if(threadIdx.x==128) out[blockIdx.x]=s;
  }
  if (threadIdx.x==0){ int n=atomicAdd(live,1)+1; atomicMax(maxlive,n); }
  __syncthreads();
  long long t0=clock64(); while(clock64()-t0 < 60000000LL);
  __syncthreads();
  if (threadIdx.x==0) atomicSub(live,1);
}
int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  int occpred; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occpred,(void*)split,256,0));
  int *lv,*mx; float *o,*a; int G=p.multiProcessorCount*16;
  CK(cudaMalloc(&lv,4)); CK(cudaMalloc(&mx,4)); CK(cudaMalloc(&o,4*G)); CK(cudaMalloc(&a,4*128));
  CK(cudaMemset(lv,0,4)); CK(cudaMemset(mx,0,4));
  float ha[128]; for(int i=0;i<128;i++) ha[i]=1.0f+i*0.01f;
  CK(cudaMemcpy(a,ha,sizeof(ha),cudaMemcpyHostToDevice));
  split<<<G,256>>>(lv,mx,o,a);
  cudaError_t e=cudaDeviceSynchronize();
  int h; CK(cudaMemcpy(&h,mx,4,cudaMemcpyDeviceToHost));
  float ho; CK(cudaMemcpy(&ho,o,4,cudaMemcpyDeviceToHost));
  printf("split loader/compute: sync=%s\n",cudaGetErrorString(e));
  printf("  cudaOccupancyMaxActiveBlocksPerSM=%d   measured max concurrent=%d (%.2f blk/SM)\n",
         occpred,h,(double)h/p.multiProcessorCount);
  printf("  compute-warp result out[0]=%.6f (expect finite, non-zero)\n",ho);
  return 0;
}

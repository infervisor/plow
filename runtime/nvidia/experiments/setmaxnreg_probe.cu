// setmaxnreg on sm_120a: does it execute, and is the effect observable?
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(2);} }while(0)

// ---- Test 1: pool-exhaustion. 256 thr = 2 warpgroups, entry maxnreg 128.
// CTA pool = 128*256 = 32768 regs. WG1 inc 128->232 needs (232-128)*128 = 13312.
// WG0 dec 128->24 releases (128-24)*128 = 13312.  Exact fit.
// If setmaxnreg is REAL: DO_DEC=1 completes, DO_DEC=0 blocks forever.
// If setmaxnreg is a NO-OP: both complete.
template<int DO_DEC>
__global__ void __maxnreg__(128) pool_test(int* out){
  if (threadIdx.x < 128) {
    if (DO_DEC) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;"::"n"(24));
    if (threadIdx.x==0) atomicAdd(out+0,1);   // WG0 arrived
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"::"n"(232));
    if (threadIdx.x==128) atomicAdd(out+1,1); // WG1 got its registers
  }
}

// ---- Test 2: does dec raise achieved blocks/SM? (CTAPOOL => predict NO)
template<int DO_DEC>
__global__ void __maxnreg__(128) occ_test(int* live,int* maxlive){
  if (DO_DEC) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;"::"n"(24));
  if (threadIdx.x==0){ int n=atomicAdd(live,1)+1; atomicMax(maxlive,n); }
  __syncthreads();
  long long t0=clock64(); while(clock64()-t0 < 40000000LL);   // hold residency
  __syncthreads();
  if (threadIdx.x==0) atomicSub(live,1);
}

int main(int argc,char**argv){
  const char* mode = argc>1?argv[1]:"all";
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  if(!strcmp(mode,"all")) printf("GPU %s  SMs=%d  cc %d.%d  regsPerSM=%d\n",p.name,p.multiProcessorCount,p.major,p.minor,p.regsPerMultiprocessor);

  if(!strcmp(mode,"pool_dec")||!strcmp(mode,"pool_nodec")){
    int dec = !strcmp(mode,"pool_dec");
    int *d; CK(cudaMalloc(&d,8)); CK(cudaMemset(d,0,8));
    if(dec) pool_test<1><<<1,256>>>(d); else pool_test<0><<<1,256>>>(d);
    cudaError_t e=cudaDeviceSynchronize();
    int h[2]={-1,-1}; cudaMemcpy(h,d,8,cudaMemcpyDeviceToHost);
    printf("[%s] sync=%s  WG0_arrived=%d WG1_got_regs=%d\n",mode,cudaGetErrorString(e),h[0],h[1]);
    fflush(stdout); return e==cudaSuccess?0:1;
  }

  if(!strcmp(mode,"all")){
    // --- occupancy: static prediction vs measured, dec vs no-dec
    int occ_nd,occ_d;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_nd,(void*)occ_test<0>,256,0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_d ,(void*)occ_test<1>,256,0));
    printf("cudaOccupancyMaxActiveBlocksPerSM: no-dec=%d  dec=%d  (256 thr, __maxnreg__(128))\n",occ_nd,occ_d);
    int *lv,*mx; CK(cudaMalloc(&lv,4)); CK(cudaMalloc(&mx,4));
    int grid = p.multiProcessorCount*16;
    for(int d=0;d<2;d++){
      CK(cudaMemset(lv,0,4)); CK(cudaMemset(mx,0,4));
      if(d) occ_test<1><<<grid,256>>>(lv,mx); else occ_test<0><<<grid,256>>>(lv,mx);
      CK(cudaDeviceSynchronize());
      int h; CK(cudaMemcpy(&h,mx,4,cudaMemcpyDeviceToHost));
      printf("measured achieved: dec=%d  max concurrent blocks=%d  => %.2f blocks/SM\n",
             d,h,(double)h/p.multiProcessorCount);
    }
  }
  return 0;
}

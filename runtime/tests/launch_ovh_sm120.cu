/* launch_ovh_sm120.cu — C-1R H1(b): per-op kernel-launch overhead the fused decode
 * megakernel exists to avoid. A hybrid that pulls the weight-GEMVs out as separate
 * oversubscribed launches pays this per GEMV op per token per layer. Dependent launches on
 * one stream serialize, so the added wall time per token ~= (#GEMV launches) x (per-launch
 * serialized gap). We measure that gap. Build:
 *   nvcc -std=c++17 -O3 -arch=sm_120a runtime/tests/launch_ovh_sm120.cu -o /tmp/lovh
 */
#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)
__global__ void noop(){}
__global__ void tiny(float* p){ if(threadIdx.x==0&&blockIdx.x==0) p[0]+=1.0f; }
int main(){
  float* d; cudaMalloc(&d,4);
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int N=20000; float ms;
  for(int i=0;i<1000;i++) noop<<<188,256>>>();
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(e0));
  for(int i=0;i<N;i++) noop<<<188,256>>>();
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  cudaEventElapsedTime(&ms,e0,e1);
  printf("noop   GRID=188  : %.3f us/launch (device-side back-to-back)\n", ms*1000.0/N);
  CK(cudaEventRecord(e0));
  for(int i=0;i<N;i++) noop<<<940,256>>>();
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  cudaEventElapsedTime(&ms,e0,e1);
  printf("noop   GRID=940  : %.3f us/launch\n", ms*1000.0/N);
  CK(cudaEventRecord(e0));
  for(int i=0;i<N;i++) tiny<<<940,256>>>(d);
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  cudaEventElapsedTime(&ms,e0,e1);
  printf("tiny   GRID=940  : %.3f us/launch (dependent, 1 elem RAW)\n", ms*1000.0/N);
  cudaDeviceSynchronize();
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  for(int i=0;i<N;i++) noop<<<940,256>>>();
  clock_gettime(CLOCK_MONOTONIC,&t1);
  double hus=((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/1e3/N;
  CK(cudaDeviceSynchronize());
  printf("host-enqueue only: %.3f us/launch (CPU cudaLaunchKernel cost)\n", hus);

  /* (5) CUDA graph: capture a chain of 200 dependent tiny kernels, replay, measure per-op.
   * This is the relevant floor for a graph-captured de-fused decode (no per-op host dispatch). */
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaGraph_t graph; cudaGraphExec_t ge;
  int CHAIN=200;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for(int i=0;i<CHAIN;i++) tiny<<<940,256,0,s>>>(d);
  CK(cudaStreamEndCapture(s,&graph));
  CK(cudaGraphInstantiate(&ge,graph,0));
  for(int i=0;i<50;i++) CK(cudaGraphLaunch(ge,s));
  CK(cudaStreamSynchronize(s));
  int R=2000;
  CK(cudaEventRecord(e0,s));
  for(int i=0;i<R;i++) CK(cudaGraphLaunch(ge,s));
  CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
  cudaEventElapsedTime(&ms,e0,e1);
  printf("graph-replay 200-chain GRID=940 : %.3f us/op (dependent, graph-captured)\n",
         ms*1000.0/R/CHAIN);
  return 0;
}

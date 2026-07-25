/* zg0_bwcal_sm120.cu — calibrate the TRUE achievable HBM read bandwidth on this card, to fix
 * the %wall denominator for ZG-0. Measures a pure streaming read of a buffer at several sizes,
 * both cold-L2 (flush before each timed launch) and warm/sustained (N launches back-to-back,
 * pipeline never drains — the regime the fp8-decode 1522 GB/s number lives in). */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void kread(const uint4* __restrict__ p, size_t n8, float* sink) {
    float a = 0.f;
    for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < n8; i += (size_t)gridDim.x*blockDim.x) {
        uint4 v = p[i]; a += (float)(v.x ^ v.y ^ v.z ^ v.w);
    }
    __shared__ float r[8];
    for (int o=16;o>0;o>>=1) a += __shfl_down_sync(~0u,a,o);
    if((threadIdx.x&31)==0) r[threadIdx.x>>5]=a; __syncthreads();
    if(threadIdx.x==0){ float s=0; for(int i=0;i<(int)(blockDim.x>>5);i++) s+=r[i]; atomicAdd(sink,s); }
}

int main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0));
    int sm=pr.multiProcessorCount;
    printf("# %s SMs=%d\n", pr.name, sm);
    const double WALL=1535.0;
    float* sink; CK(cudaMalloc(&sink,4));
    void* flush; size_t fb=256ull<<20; CK(cudaMalloc(&flush,fb));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    size_t sizes[] = {32ull<<20, 64ull<<20, 112ull<<20, 256ull<<20, 512ull<<20, 1024ull<<20, 2048ull<<20};
    for (size_t bytes : sizes) {
        uint4* buf; CK(cudaMalloc(&buf, bytes)); CK(cudaMemset(buf, 1, bytes));
        size_t n8 = bytes/16; int grid=sm*8;
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for(int i=0;i<3;i++) kread<<<grid,256,0,st>>>(buf,n8,sink); CK(cudaDeviceSynchronize());
        /* cold: flush L2 before each */
        double coldbest=1e30;
        for(int it=0;it<20;it++){ CK(cudaMemsetAsync(flush,it,fb,st));
            CK(cudaEventRecord(e0,st)); kread<<<grid,256,0,st>>>(buf,n8,sink); CK(cudaEventRecord(e1,st));
            CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); if(ms<coldbest)coldbest=ms; }
        /* warm sustained: R launches back-to-back, no flush, no drain */
        int R=50; CK(cudaEventRecord(e0,st));
        for(int i=0;i<R;i++) kread<<<grid,256,0,st>>>(buf,n8,sink);
        CK(cudaEventRecord(e1,st)); CK(cudaEventSynchronize(e1));
        float wms; CK(cudaEventElapsedTime(&wms,e0,e1)); double warm=wms/R;
        double cgb=(double)bytes/(coldbest*1e-3)/1e9, wgb=(double)bytes/(warm*1e-3)/1e9;
        printf("size=%5zuMB  cold=%7.1f GB/s (%4.1f%%)   warm=%7.1f GB/s (%4.1f%%)\n",
            bytes>>20, cgb, 100*cgb/WALL, wgb, 100*wgb/WALL);
        CK(cudaFree(buf));
    }
    return 0;
}

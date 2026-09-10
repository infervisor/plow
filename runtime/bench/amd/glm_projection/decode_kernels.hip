#define PLOW_WG_WAVES 8
#define PLOW_GEMV_MM 16
#define PLOW_GEMV_WALK 1
#define PLOW_GEMV_LG 1
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_gemm.h"
#endif
__global__ __launch_bounds__(512) void projection(unsigned short* c, const unsigned short* x,
 const unsigned short* w, unsigned m, unsigned n, unsigned k) {
#if defined(__HIP_DEVICE_COMPILE__)
 __shared__ bf16 lds[GM_LDS_HALVES];
 d_gemv(c,x,w,nullptr,nullptr,m,n,k,0,0.0f,blockIdx.x,gridDim.x,lds);
#endif
}
extern "C" int plow_projection(void* c, void* x, void* w, unsigned m,unsigned n,unsigned k,void* stream) {
 projection<<<304,512,0,(hipStream_t)stream>>>((unsigned short*)c,(unsigned short*)x,(unsigned short*)w,m,n,k);
 return hipGetLastError();
}

/* occ_probe.cu — PX-3 gate: load the real _pfgemm cubin and query occupancy via the driver API.
 * Reports registers + max active blocks/SM at the object's own dynamic-smem arena. */
#include <cstdio>
#include <cuda.h>
#define D(x) do{CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%d %s -> %s\n",__FILE__,__LINE__,#x,s);return 1;}}while(0)
static int probe(const char* cubin, unsigned arena_bytes){
    CUmodule m; CUfunction f;
    D(cuModuleLoad(&m,cubin));
    D(cuModuleGetFunction(&f,m,"_Z19interp_sm120_pfgemm11PlowProgram"));
    int regs=0,smem_static=0;
    D(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,f));
    D(cuFuncGetAttribute(&smem_static,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f));
    D(cuFuncSetAttribute(f,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,(int)arena_bytes));
    int blocks=0;
    D(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,f,256,arena_bytes));
    printf("%s: regs=%d static_smem=%d dyn_arena=%u (%.1f KiB) -> blocks/SM=%d\n",
           cubin,regs,smem_static,arena_bytes,arena_bytes/1024.0,blocks);
    D(cuModuleUnload(m));
    return 0;
}
int main(int argc,char**argv){
    D(cuInit(0)); CUdevice d; D(cuDeviceGet(&d,0)); CUcontext c; D(cuDevicePrimaryCtxRetain(&c,d)); D(cuCtxSetCurrent(c));
    // BN=128 halved: arena 20480 bf16 = 40960 B; BN=64 full: 23040 bf16 = 46080 B
    if(probe("/tmp/pfgemm_bn128.cubin",40960)) return 1;
    if(probe("/tmp/pfgemm_bn64.cubin",46080)) return 1;
    return 0;
}

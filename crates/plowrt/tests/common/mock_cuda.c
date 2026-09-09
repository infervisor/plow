// Minimal CUDA driver for context-ownership tests; unused entry points are stubs.
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static _Thread_local uintptr_t ctx;
static _Thread_local unsigned bound_generation;
static atomic_uint refs[2], generation[2];
static int valid_context(void) { return ctx && bound_generation == atomic_load(&generation[ctx-1]); }
int cuInit(unsigned f) { return 0; }
int cuDriverGetVersion(int *v) { *v=13000; return 0; }
int cuGetErrorName(int e, const char **v) { *v="MOCK_ERROR"; return 0; }
int cuDeviceGet(int *d, int n) { *d=n; return n<2 ? 0 : 101; }
int cuDeviceGetCount(int *n) { *n=2; return 0; }
int cuDeviceGetName(char *p, int n, int d) { snprintf(p,n,"Mock GPU %d",d); return 0; }
int cuDeviceGetAttribute(int *p, int a, int d) { *p=1; return 0; }
int cuDevicePrimaryCtxRetain(void **p, int d) { atomic_fetch_add(&refs[d],1); *p=(void *)(uintptr_t)(d+1); return 0; }
int cuDevicePrimaryCtxRelease_v2(int d) { if (atomic_fetch_sub(&refs[d],1)==1) atomic_fetch_add(&generation[d],1); return 0; }
int cuCtxSetCurrent(void *p) { ctx=(uintptr_t)p; bound_generation=ctx ? atomic_load(&generation[ctx-1]) : 0; return 0; }
int cuCtxSynchronize(void) { return 0; }
int cuModuleLoadData(void) { return 0; }
int cuModuleUnload(void) { return 0; }
int cuModuleGetFunction(void) { return 0; }
int cuModuleGetGlobal_v2(void) { return 0; }
int cuFuncSetAttribute(void) { return 0; }
int cuOccupancyMaxActiveBlocksPerMultiprocessor(void) { return 0; }
int cuMemAlloc_v2(uint64_t *p, size_t n) { if (!valid_context()) return 201; uint64_t *mem=calloc(1,n+8); if (!mem) return 2; *mem=ctx; *p=(uintptr_t)(mem+1); return 0; }
int cuMemFree_v2(uint64_t p) { free((uint64_t *)(uintptr_t)p-1); return 0; }
int cuMemGetInfo_v2(size_t *f, size_t *t) { if (!valid_context()) return 201; *f=*t=ctx*(80ULL<<30); return 0; }
int cuMemcpyHtoD_v2(uint64_t p, const void *s, size_t n) { memcpy((void *)(uintptr_t)p,s,n); return 0; }
int cuMemcpyDtoH_v2(void *d, uint64_t p, size_t n) { memcpy(d,(void *)(uintptr_t)p,n); return 0; }
int cuMemsetD8_v2(void) { return 0; }
int cuMemHostAlloc(void) { return 0; }
int cuMemFreeHost(void) { return 0; }
int cuLaunchCooperativeKernel(void) { return 0; }
int cuLaunchKernel(void) { return 0; }
int cuMemGetAllocationGranularity(void) { return 0; }
int cuMemAddressReserve(void) { return 0; }
int cuMemAddressFree(void) { return 0; }
int cuMemCreate(void) { return 0; }
int cuMemRelease(void) { return 0; }
int cuMemMap(void) { return 0; }
int cuMemUnmap(void) { return 0; }
int cuMemSetAccess(void) { return 0; }
int cuMemcpyDtoD_v2(void) { return 0; }
int cuStreamCreate(void) { return 0; }
int cuStreamDestroy_v2(void) { return 0; }
int cuStreamSynchronize(void) { return 0; }
int cuMemcpyHtoDAsync_v2(void) { return 0; }
int cuMemcpyDtoHAsync_v2(void) { return 0; }
int cuMemsetD8Async(void) { return 0; }
int cuEventCreate(void) { return 0; }
int cuEventDestroy_v2(void) { return 0; }
int cuEventRecord(void) { return 0; }
int cuEventQuery(void) { return 0; }
int cuEventSynchronize(void) { return 0; }
int cuEventElapsedTime(void) { return 0; }
int cuStreamBeginCapture(void) { return 0; }
int cuStreamEndCapture(void) { return 0; }
int cuGraphCreate(void) { return 0; }
int cuGraphAddKernelNode_v2(void) { return 0; }
int cuGraphInstantiateWithFlags(void) { return 0; }
int cuGraphLaunch(void) { return 0; }
int cuGraphDestroy(void) { return 0; }
int cuGraphExecDestroy(void) { return 0; }
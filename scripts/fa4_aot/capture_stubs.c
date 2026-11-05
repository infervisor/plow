// CPU-only capture of the CuTe DSL host launch: stands in for the CUDA runtime entry points the
// exported host code calls, and prints the launch it would make (config, attributes, kernel
// parameter bytes). Param sizes come from PLOW_CAP_SIZES ("s0,s1,..." from the cubin's KPARAM_INFO).
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static FILE* out() { static FILE* f; if (!f) f = stdout; return f; }

cudaError_t _cudaLibraryLoadData(cudaLibrary_t* lib, const void* code, void* a, void* b, unsigned c,
                                 void* d, void* e, unsigned g) {
    const unsigned char* p = (const unsigned char*)code;
    while (!(p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F')) p++;
    uint64_t shoff, phoff; uint16_t shentsize, shnum, phentsize, phnum;
    memcpy(&phoff, p + 0x20, 8); memcpy(&shoff, p + 0x28, 8); memcpy(&phentsize, p + 0x36, 2);
    memcpy(&phnum, p + 0x38, 2); memcpy(&shentsize, p + 0x3a, 2); memcpy(&shnum, p + 0x3c, 2);
    uint64_t size = shoff + (uint64_t)shentsize * shnum;
    if (phoff + (uint64_t)phentsize * phnum > size) size = phoff + (uint64_t)phentsize * phnum;
    for (int i = 0; i < shnum; i++) {
        const unsigned char* sh = p + shoff + (uint64_t)i * shentsize;
        uint32_t type; uint64_t off, sz;
        memcpy(&type, sh + 4, 4); memcpy(&off, sh + 0x18, 8); memcpy(&sz, sh + 0x20, 8);
        if (type != 8 && off + sz > size) size = off + sz;
    }
    for (int i = 0; i < phnum; i++) {
        const unsigned char* ph = p + phoff + (uint64_t)i * phentsize;
        uint64_t off, sz;
        memcpy(&off, ph + 8, 8); memcpy(&sz, ph + 0x20, 8);
        if (off + sz > size) size = off + sz;
    }
    const char* path = getenv("PLOW_CAP_CUBIN");
    if (path) { FILE* f = fopen(path, "wb"); fwrite(p, 1, size, f); fclose(f); }
    fprintf(out(), "library_load elf_offset=%ld bytes=%llu\n", (long)(p - (const unsigned char*)code), (unsigned long long)size);
    *lib = (cudaLibrary_t)0x1000;
    return cudaSuccess;
}
cudaError_t _cudaLibraryUnload(cudaLibrary_t lib) { return cudaSuccess; }
cudaError_t _cudaLibraryGetKernel(cudaKernel_t* k, cudaLibrary_t lib, const char* name) {
    fprintf(out(), "kernel %s\n", name);
    *k = (cudaKernel_t)0x2000;
    return cudaSuccess;
}
cudaError_t _cudaFuncSetAttribute(const void* f, int attr, int value) {
    fprintf(out(), "func_attr %d %d\n", attr, value);
    return cudaSuccess;
}
cudaError_t _cudaKernelSetAttributeForDevice(cudaKernel_t k, int attr, int value, int dev) {
    fprintf(out(), "kernel_attr %d %d\n", attr, value);
    return cudaSuccess;
}
cudaError_t _cudaDeviceGetAttribute(int* v, int attr, int dev) {
    // H100 SXM5: multiProcessorCount 132, maxSharedMemoryPerBlockOptin 232448.
    *v = attr == cudaDevAttrMultiProcessorCount ? 132
       : attr == cudaDevAttrMaxSharedMemoryPerBlockOptin ? 232448 : 0;
    fprintf(out(), "device_attr %d -> %d\n", attr, *v);
    return cudaSuccess;
}
CUresult _cuKernelGetAttribute(int* v, int attr, CUkernel k, CUdevice dev) {
    *v = 0;
    fprintf(out(), "cu_kernel_attr %d -> 0\n", attr);
    return CUDA_SUCCESS;
}
cudaError_t _cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
cudaError_t _cudaLaunchKernelEx(const cudaLaunchConfig_t* c, const void* func, void** args) {
    fprintf(out(), "launch grid=%u,%u,%u block=%u,%u,%u smem=%zu nattrs=%u\n", c->gridDim.x, c->gridDim.y,
            c->gridDim.z, c->blockDim.x, c->blockDim.y, c->blockDim.z, c->dynamicSmemBytes, c->numAttrs);
    for (unsigned i = 0; i < c->numAttrs; i++) {
        const unsigned char* v = (const unsigned char*)&c->attrs[i].val;
        fprintf(out(), "attr %d ", (int)c->attrs[i].id);
        for (int j = 0; j < 16; j++) fprintf(out(), "%02x", v[j]);
        fprintf(out(), "\n");
    }
    const char* sizes = getenv("PLOW_CAP_SIZES");
    int i = 0;
    for (const char* s = sizes; s && *s; i++) {
        int n = atoi(s);
        const unsigned char* p = (const unsigned char*)args[i];
        fprintf(out(), "param %d %d ", i, n);
        for (int j = 0; j < n; j++) fprintf(out(), "%02x", p[j]);
        fprintf(out(), "\n");
        s = strchr(s, ','); if (s) s++;
    }
    return cudaSuccess;
}
cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t e) { return "stub"; }
const char* cudaGetErrorName(cudaError_t e) { return "stub"; }

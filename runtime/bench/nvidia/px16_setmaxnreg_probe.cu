/* px16_setmaxnreg_probe.cu — does sm_120a support NVIDIA's DYNAMIC register allocation
 * (`setmaxnreg`, PTX ISA 8.0), and if so does it change the occupancy arithmetic?
 *
 * Two separate questions, answered separately:
 *   k_smn   — does the instruction assemble for sm_120a at all? (sm_90a is the known-good control)
 *   k_pool  — semantics: setmaxnreg.dec releases registers to the CTA's OWN pool, and
 *             setmaxnreg.inc takes them back from it. Occupancy is decided at LAUNCH from the
 *             kernel's declared register count, so releasing cannot make room for a third CTA.
 *             `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is the arbiter; the host side of
 *             this probe prints it for a kernel that decs to 24 and one that does not.
 */
#include <cstdio>
#include <cuda_runtime.h>

__global__ void __launch_bounds__(256, 2) k_smn(float* o) {
    /* 2 warpgroups of 128. Producer warpgroup releases, consumer acquires — the CUTLASS shape. */
    if (threadIdx.x < 128) asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
    else                   asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    o[threadIdx.x] = (float)threadIdx.x;
}

__global__ void __launch_bounds__(256, 1) k_plain(float* o) {
    float a[64];
#pragma unroll
    for (int i = 0; i < 64; i++) a[i] = (float)(threadIdx.x + i);
#pragma unroll
    for (int i = 1; i < 64; i++) a[0] += a[i] * a[i - 1];
    o[threadIdx.x] = a[0];
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# %s cc %d.%d, %d SMs, %d regs/SM\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.regsPerBlock);
    cudaFuncAttributes a{};
    int blk = 0;
    if (cudaFuncGetAttributes(&a, (const void*)k_smn) == cudaSuccess) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk, (const void*)k_smn, 256, 0);
        printf("k_smn  (setmaxnreg dec24/inc232): regs=%d spill=%d blocks/SM=%d\n",
               a.numRegs, (int)a.localSizeBytes, blk);
    }
    if (cudaFuncGetAttributes(&a, (const void*)k_plain) == cudaSuccess) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk, (const void*)k_plain, 256, 0);
        printf("k_plain(no setmaxnreg)          : regs=%d spill=%d blocks/SM=%d\n",
               a.numRegs, (int)a.localSizeBytes, blk);
    }
    float* d = nullptr;
    cudaMalloc(&d, 256 * 4);
    k_smn<<<1, 256>>>(d);
    printf("launch k_smn: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    cudaFree(d);
    return 0;
}

// fp8_mma_rate_probe.cu — settle the sm_120 mma.sync rate question for the flash QK lever:
// is m16n8k32.e4m3 with F32 accumulate 2x the bf16 m16n8k16 rate, or is the 2x f16-acc-only
// (consumer-segmentation, as on Ada GeForce)? Register-only operands, 8 warps/block, both a
// THROUGHPUT variant (4 independent chains/warp) and a LATENCY variant (1 dependent chain).
// Build: nvcc -std=c++17 -arch=sm_120a -O3 fp8_mma_rate_probe.cu -o probe
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("ERR %s: %s\n",#x,cudaGetErrorString(e)); return 1;} }while(0)

template <int MODE, int CHAINS>
__global__ void __launch_bounds__(256, 1) k_mma(float* sink, int iters) {
    float d[CHAINS][4];
    unsigned a[4], b[2];
    __half2 h[CHAINS][2];
#pragma unroll
    for (int c = 0; c < CHAINS; c++) {
        d[c][0] = threadIdx.x * 1e-30f; d[c][1] = d[c][2] = d[c][3] = 0.f;
        h[c][0] = __floats2half2_rn(d[c][0], 0.f); h[c][1] = __floats2half2_rn(0.f, 0.f);
    }
    a[0] = threadIdx.x; a[1] = 0x3c003c00u; a[2] = 0x38003800u; a[3] = 1u;
    b[0] = 0x3c003c00u; b[1] = threadIdx.x ^ 5u;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int c = 0; c < CHAINS; c++) {
            if constexpr (MODE == 0) { /* bf16 m16n8k16, f32 acc (the shipped QK) */
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            } else if constexpr (MODE == 1) { /* e4m3 m16n8k32, f32 acc (Lever A as written) */
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            } else { /* e4m3 m16n8k32, f16 acc (the consumer 2x path?) */
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                             "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(*(unsigned*)&h[c][0]), "+r"(*(unsigned*)&h[c][1])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
        }
    }
    float s = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; c++)
        s += (MODE == 2) ? (__low2float(h[c][0]) + __low2float(h[c][1])) : (d[c][0] + d[c][3]);
    if (s == 12345.678f) *sink = s;
}

template <int MODE, int CHAINS>
static double bench(const char* label, int blocks) {
    float* sink; cudaMalloc(&sink, 4);
    const int iters = 200000;
    k_mma<MODE, CHAINS><<<blocks, 256>>>(sink, 100); cudaDeviceSynchronize();
    cudaEvent_t x, y; cudaEventCreate(&x); cudaEventCreate(&y);
    cudaEventRecord(x);
    k_mma<MODE, CHAINS><<<blocks, 256>>>(sink, iters);
    cudaEventRecord(y); cudaEventSynchronize(y);
    float ms = 0; cudaEventElapsedTime(&ms, x, y);
    /* MACs per mma: k16 = 16*8*16 = 2048, k32 = 4096 */
    const double macs = (double)blocks * 8 /*warps*/ * (double)iters * CHAINS *
                        ((MODE == 0) ? 2048.0 : 4096.0);
    const double tflops = 2.0 * macs / (ms * 1e-3) / 1e12;
    printf("  %-44s %8.3f ms  %8.1f TFLOP/s\n", label, ms, tflops);
    cudaFree(sink);
    return tflops;
}

int main() {
    cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s SMs=%d\n", p.name, p.multiProcessorCount);
    const int B = p.multiProcessorCount;
    printf("THROUGHPUT (4 independent chains/warp):\n");
    bench<0, 4>("bf16 m16n8k16 f32acc", B);
    bench<1, 4>("e4m3 m16n8k32 f32acc", B);
    bench<2, 4>("e4m3 m16n8k32 f16acc", B);
    printf("LATENCY-BOUND (1 dependent chain/warp):\n");
    bench<0, 1>("bf16 m16n8k16 f32acc", B);
    bench<1, 1>("e4m3 m16n8k32 f32acc", B);
    bench<2, 1>("e4m3 m16n8k32 f16acc", B);
    return 0;
}

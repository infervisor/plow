// icache_footprint.hip — does INSTRUCTION-cache footprint cost anything on gfx950?
//
// THE QUESTION. plow's AMD interpreter is a persistent megakernel: every op body is compiled into
// one object (interp_decode_fp8kv_k3.elf .text = 346,240 B). Consecutive packets in a K3 decode
// step run DIFFERENT op bodies, so the instruction stream jumps across a large binary. If the L1
// instruction cache cannot hold the set of bodies one layer touches, every packet re-fetches its
// body from L2.
//
// THE INSTRUMENT. A switch-dispatch loop over NBODY distinct `noinline` bodies, round-robin, for a
// FIXED NSTEP. Every body has the SAME instruction count and the SAME data traffic (none) and
// differs only in its immediate constants, so the compiler cannot merge them (identical-code
// folding) and cannot shorten them (strict FP: (x*a+b)*c+d != x*(ac)+(bc+d)).
//
//   work(NBODY=1) == work(NBODY=32) exactly: NSTEP bodies executed either way.
//   footprint(NBODY=n) = n * sizeof(one body).
//
// So any time difference between the arms is CODE FOOTPRINT and nothing else. NBODY=1 is also the
// A/A control against itself.
//
// This is deliberately the WORST CASE for I-cache. Each body is STRAIGHT-LINE: every instruction
// is fetched exactly once per invocation, nothing is amortised over loop trips. Real op bodies are
// loops (fetch the inner loop once, run it hundreds of times), so whatever this measures is an
// UPPER BOUND on the effect in the real interpreter.
//
// BUILD/RUN:
//   hipcc --offload-arch=gfx950 -O3 -std=c++17 -w -DNBODY=16 icache_footprint.hip -o icf16
//   flock /tmp/plow_gpu.lock sg render -c 'LD_LIBRARY_PATH=/opt/rocm/lib ./icf16'
//
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

#ifndef NBODY
#define NBODY 1              // distinct bodies in the round-robin (power of 2, <= 32)
#endif
#ifndef BLEN
#define BLEN 512             // affine steps per body -> sets the per-body code size
#endif
#ifndef NSTEP
#define NSTEP 4096           // body invocations per thread; IDENTICAL across all NBODY arms
#endif
#define THREADS 512          // 8 waves, PLOW_THREADS

// One affine step. `I` (body id) and `k` (step index) pick the constants, so body I's instruction
// bytes differ from body J's while the instruction COUNT is identical.
#define AFFINE(I, k)                                                                   \
    x = x * (1.0f + (float)(((I) * 37 + (k) * 11) & 255) * (1.0f / 4096.0f)) +          \
        (float)(((I) * 53 + (k) * 7) & 511) * (1.0f / 8192.0f);

#define A8(I, b)  AFFINE(I,(b)+0) AFFINE(I,(b)+1) AFFINE(I,(b)+2) AFFINE(I,(b)+3)      \
                  AFFINE(I,(b)+4) AFFINE(I,(b)+5) AFFINE(I,(b)+6) AFFINE(I,(b)+7)
#define A64(I, b) A8(I,(b)+0) A8(I,(b)+8) A8(I,(b)+16) A8(I,(b)+24)                     \
                  A8(I,(b)+32) A8(I,(b)+40) A8(I,(b)+48) A8(I,(b)+56)

// BLEN affine steps of straight-line code. Measure the resulting byte size from the ELF symbol
// table (llvm-readelf --symbols) rather than trusting an estimate.
template <int I>
__device__ __attribute__((noinline)) float body(float x) {
    A64(I, 0)
#if BLEN >= 128
    A64(I, 64)
#endif
#if BLEN >= 192
    A64(I, 128)
#endif
#if BLEN >= 256
    A64(I, 192)
#endif
#if BLEN >= 384
    A64(I, 256) A64(I, 320)
#endif
#if BLEN >= 512
    A64(I, 384) A64(I, 448)
#endif
    return x;
}

struct Args { float* seed; float* sink; int nstep; };

#define CASE(I) case I: x = body<I>(x); break;

__global__ void __launch_bounds__(THREADS) chain(Args a) {
    // 90 KB of LDS forces 1 block/CU, matching the interpreter's occupancy-2 / 8-waves-per-WG shape
    // so each CU pair sees the same L1I pressure the real megakernel puts on it.
    __shared__ char pad[90 * 1024];
    ((volatile char*)pad)[threadIdx.x & 1023] = 0;

    float x = a.seed[threadIdx.x];
    for (int s = 0; s < a.nstep; s++) {
        switch (s & (NBODY - 1)) {
            CASE(0)
#if NBODY > 1
            CASE(1)
#endif
#if NBODY > 2
            CASE(2) CASE(3)
#endif
#if NBODY > 4
            CASE(4) CASE(5) CASE(6) CASE(7)
#endif
#if NBODY > 8
            CASE(8)  CASE(9)  CASE(10) CASE(11)
            CASE(12) CASE(13) CASE(14) CASE(15)
#endif
#if NBODY > 16
            CASE(16) CASE(17) CASE(18) CASE(19)
            CASE(20) CASE(21) CASE(22) CASE(23)
            CASE(24) CASE(25) CASE(26) CASE(27)
            CASE(28) CASE(29) CASE(30) CASE(31)
#endif
            default: break;
        }
    }
    a.sink[blockIdx.x * THREADS + threadIdx.x] = x;
}

#define CK(e) do { hipError_t _e = (e); if (_e) { printf("HIP ERR %s:%d %s\n", __FILE__, __LINE__, \
    hipGetErrorString(_e)); exit(2); } } while (0)

static int cmpf(const void* a, const void* b) {
    float x = *(const float*)a, y = *(const float*)b; return x < y ? -1 : x > y ? 1 : 0;
}

int main(int argc, char** argv) {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    const int GD = p.multiProcessorCount;
    const int reps = (argc > 1) ? atoi(argv[1]) : 21;
    const int nstep = (argc > 2) ? atoi(argv[2]) : NSTEP;

    Args a{};
    CK(hipMalloc(&a.seed, THREADS * sizeof(float)));
    CK(hipMalloc(&a.sink, (size_t)GD * THREADS * sizeof(float)));
    float h[THREADS]; for (int i = 0; i < THREADS; i++) h[i] = 1.0f + i * 1e-4f;
    CK(hipMemcpy(a.seed, h, sizeof(h), hipMemcpyHostToDevice));
    a.nstep = nstep;

    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (int w = 0; w < 3; w++) { hipLaunchKernelGGL(chain, dim3(GD), dim3(THREADS), 0, 0, a); }
    CK(hipDeviceSynchronize());

    float* ms = (float*)malloc(reps * sizeof(float));
    for (int r = 0; r < reps; r++) {
        CK(hipEventRecord(e0));
        hipLaunchKernelGGL(chain, dim3(GD), dim3(THREADS), 0, 0, a);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        CK(hipEventElapsedTime(&ms[r], e0, e1));
    }
    qsort(ms, reps, sizeof(float), cmpf);
    const float med = ms[reps / 2];
    // per body invocation, in ns: NSTEP invocations, serial per wave
    printf("NBODY=%-3d BLEN=%-4d NSTEP=%-6d blocks=%d  median %.4f ms  min %.4f  max %.4f  "
           "| %.2f ns/body-invocation\n",
           NBODY, BLEN, nstep, GD, med, ms[0], ms[reps - 1], med * 1e6f / (float)nstep);
    return 0;
}

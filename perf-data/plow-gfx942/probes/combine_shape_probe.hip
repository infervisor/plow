/* op-87 (MoeCombinePf) shape probe, gfx942.
 *
 * The fusion's saving is NOT the combine's BYTES -- PLOW_MOE_PF_PART16 already halved them for
 * 0% (glm52-moepf-activation-arms.md). It is the combine's SHAPE: today it reads k=8 streams at
 * H*4 = 24 KB stride per output element; fused, it reads ONE contiguous accumulator. This probe
 * prices exactly that difference at the shipped T=8192 TP8 shape.
 *
 *   0  current    out[i] = bf16(shared[i] + SUM_j part[tok*k*H + j*H + h])   1.611 GB strided
 *   1  part16     same with bf16 part                                       0.806 GB strided
 *   2  fused      out[i] = bf16(shared[i] + acc[i])                         0.201 GB contiguous
 *   3  fused+zero same, and zero acc for the next layer                     + 0.201 GB store
 *   4  zero-only  acc[i] = 0                                (the GLU-prologue seed cost)
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef __bf16 bf16;
#define WGS 512
#define GRID 304

__global__ __launch_bounds__(WGS) void c_cur(bf16* out, const bf16* sh, const float* part,
                                             unsigned H, unsigned k, unsigned T) {
    const size_t total = (size_t)T * H;
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < total; i += (size_t)GRID * WGS) {
        const unsigned tok = (unsigned)(i / H), h = (unsigned)(i - (size_t)tok * H);
        float acc = (float)sh[i];
        const float* pt = part + (size_t)tok * k * H;
        for (unsigned j = 0; j < k; j++) acc += pt[(size_t)j * H + h];
        out[i] = (bf16)acc;
    }
}
__global__ __launch_bounds__(WGS) void c_p16(bf16* out, const bf16* sh, const bf16* part,
                                             unsigned H, unsigned k, unsigned T) {
    const size_t total = (size_t)T * H;
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < total; i += (size_t)GRID * WGS) {
        const unsigned tok = (unsigned)(i / H), h = (unsigned)(i - (size_t)tok * H);
        float acc = (float)sh[i];
        const bf16* pt = part + (size_t)tok * k * H;
        for (unsigned j = 0; j < k; j++) acc += (float)pt[(size_t)j * H + h];
        out[i] = (bf16)acc;
    }
}
__global__ __launch_bounds__(WGS) void c_fused(bf16* out, const bf16* sh, const float* acc,
                                               unsigned H, unsigned k, unsigned T) {
    const size_t total = (size_t)T * H;
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < total; i += (size_t)GRID * WGS)
        out[i] = (bf16)((float)sh[i] + acc[i]);
}
__global__ __launch_bounds__(WGS) void c_fusedz(bf16* out, const bf16* sh, float* acc, unsigned H,
                                                unsigned k, unsigned T) {
    const size_t total = (size_t)T * H;
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < total; i += (size_t)GRID * WGS) {
        out[i] = (bf16)((float)sh[i] + acc[i]);
        __builtin_nontemporal_store(0.0f, acc + i);
    }
}
__global__ __launch_bounds__(WGS) void c_zero(float* acc, unsigned H, unsigned T) {
    const size_t total = (size_t)T * H;
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < total; i += (size_t)GRID * WGS)
        __builtin_nontemporal_store(0.0f, acc + i);
}

#define CK(x) do { hipError_t e = (x); if (e) { printf("ERR %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)
int main(void) {
    const unsigned T = 8192, H = 6144, K = 8;
    const size_t nout = (size_t)T * H;
    bf16 *d_out, *d_sh, *d_p16; float *d_part, *d_acc;
    CK(hipMalloc(&d_out, nout * 2)); CK(hipMalloc(&d_sh, nout * 2));
    CK(hipMalloc(&d_part, nout * K * 4)); CK(hipMalloc(&d_p16, nout * K * 2));
    CK(hipMalloc(&d_acc, nout * 4));
    CK(hipMemset(d_sh, 0, nout * 2)); CK(hipMemset(d_part, 0, nout * K * 4));
    CK(hipMemset(d_p16, 0, nout * K * 2)); CK(hipMemset(d_acc, 0, nout * 4));
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const char* nm[5] = {"current   k=8 f32 strided  (1.61 GB read)",
                         "part16    k=8 bf16 strided (0.81 GB read)",
                         "FUSED     1 f32 contiguous (0.20 GB read)",
                         "FUSED+zero  + 0.20 GB nt store",
                         "zero-only   0.20 GB nt store (GLU prologue)"};
    const double gb[5] = {1.611 + .1 + .1, .806 + .1 + .1, .201 + .1 + .1, .201 + .1 + .1 + .201, .201};
    for (int arm = 0; arm < 5; arm++) {
        float best = 1e30f;
        for (int rep = 0; rep < 4; rep++) {
            CK(hipEventRecord(a));
            switch (arm) {
            case 0: c_cur<<<GRID, WGS>>>(d_out, d_sh, d_part, H, K, T); break;
            case 1: c_p16<<<GRID, WGS>>>(d_out, d_sh, d_p16, H, K, T); break;
            case 2: c_fused<<<GRID, WGS>>>(d_out, d_sh, d_acc, H, K, T); break;
            case 3: c_fusedz<<<GRID, WGS>>>(d_out, d_sh, d_acc, H, K, T); break;
            case 4: c_zero<<<GRID, WGS>>>(d_acc, H, T); break;
            }
            CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
            float ms; CK(hipEventElapsedTime(&ms, a, b)); if (rep) best = ms < best ? ms : best;
        }
        printf("%-44s  %8.3f ms   %6.0f GB/s\n", nm[arm], best, gb[arm] * 1000.0 / best);
    }
    return 0;
}

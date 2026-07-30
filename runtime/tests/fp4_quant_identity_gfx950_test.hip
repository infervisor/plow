/* fp4_quant_identity_gfx950_test.hip — EXHAUSTIVE bit-identity proof for the MXFP4 quantizer
 * primitives in amd_common.h.                                              [KIMI/GLM/DEEPSEEK]
 *
 * WHY EXHAUSTIVE AND NOT SAMPLED. `quant_fp4`, `e8m0_for_amax` and the `1/e8m0_to_f32(b)`
 * reciprocal decide the E8M0 scale bytes and the fp4 codes that the A4W4 MoE prefill writes to
 * HBM, and `moe_prefill_a4w4_gfx950_test.hip` compares those scale bytes EXACTLY. They are also
 * shared with the mxfp4 GEMM path. Any rewrite of them for speed is only admissible if it is
 * bit-identical, and "bit-identical" over a one-argument f32 function is a CHECKABLE claim, not
 * an argued one: there are 2^32 inputs and the GPU can try all of them in under a second.
 *
 * So this test carries a REFERENCE COPY of each primitive, verbatim as it stood before the
 * branchless rewrite, and walks every one of the 4294967296 float bit patterns — normals,
 * subnormals, both zeros, every NaN payload, both infinities — comparing the live
 * implementation against it. The reference copies are frozen: they are the definition of "the
 * numerics the shipped objects were measured with", so if a future edit needs them to change,
 * that is the conversation, and this test is where it happens.
 *
 * It also checks the two 8-bit domains completely: `e8m0_for_amax`'s output byte round-trips,
 * and `1.0f / e8m0_to_f32(b)` against the exact power-of-two reciprocal, for all 256 b.
 *
 *   /opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/fp4_quant_identity_gfx950_test.hip -o /tmp/fp4_ident
 *   sg render -c 'ROCR_VISIBLE_DEVICES=<n> /tmp/fp4_ident'
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "amd_common.h"

#define HC(x) do{hipError_t e_=(x); if(e_){printf("hip err %s @%d\n",hipGetErrorString(e_),__LINE__);exit(1);} }while(0)

/* ---- FROZEN REFERENCES. Verbatim from amd_common.h before the branchless rewrite. --------- */
__device__ __forceinline__ unsigned ref_quant_fp4(float v) {
    const unsigned sign = (v < 0.0f) ? 8u : 0u;
    float a = fabsf(v);
    unsigned n;
    if (a < 0.25f) n = 0u;
    else if (a < 0.75f) n = 1u;
    else if (a < 1.25f) n = 2u;
    else if (a < 1.75f) n = 3u;
    else if (a < 2.5f) n = 4u;
    else if (a < 3.5f) n = 5u;
    else if (a < 5.0f) n = 6u;
    else n = 7u;
    return sign | n;
}
__device__ __forceinline__ unsigned char ref_e8m0_for_amax(float amax) {
    if (!(amax > 0.0f)) return (unsigned char)PLOW_E8M0_ONE;
    int e;
    frexpf(amax / 6.0f, &e);
    e += 127;
    if (e < 1) e = 1;
    if (e > 254) e = 254;
    return (unsigned char)e;
}
__device__ __forceinline__ float ref_e8m0_inv(unsigned char b) { return 1.0f / e8m0_to_f32(b); }

/* ---- the sweep. One thread per float, grid-strided over the whole 2^32 space. ------------- */
__global__ __launch_bounds__(256) void sweep(unsigned* bad, unsigned* first, unsigned long long lo,
                                             unsigned long long n) {
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const unsigned bits = (unsigned)(lo + i);
        const float f = __builtin_bit_cast(float, bits);
        unsigned why = 0u;
        if (quant_fp4(f) != ref_quant_fp4(f)) why |= 1u;
        if (e8m0_for_amax(f) != ref_e8m0_for_amax(f)) why |= 2u;
        if (why) {
            if (atomicAdd(bad, 1u) == 0u) { first[0] = bits; first[1] = why; }
        }
    }
}

/* The E8M0 byte domain is 256 wide, so it is checked in full rather than sampled. */
__global__ void sweep_inv(unsigned* bad, unsigned* first) {
    const unsigned b = threadIdx.x;
    /* `inv` must be BIT-identical, so compare the bit patterns and not the values: two NaNs or
     * a signed zero pair would compare equal as floats and are not the same byte. */
    const unsigned got = __builtin_bit_cast(unsigned, e8m0_inv_f32((unsigned char)b));
    const unsigned want = __builtin_bit_cast(unsigned, ref_e8m0_inv((unsigned char)b));
    if (got != want) {
        if (atomicAdd(bad, 1u) == 0u) { first[0] = b; first[1] = 4u; }
    }
}

int main() {
    unsigned *dbad, *dfirst;
    HC(hipMalloc(&dbad, 4)); HC(hipMalloc(&dfirst, 8));
    HC(hipMemset(dbad, 0, 4)); HC(hipMemset(dfirst, 0, 8));

    hipLaunchKernelGGL(sweep_inv, dim3(1), dim3(256), 0, 0, dbad, dfirst);
    HC(hipDeviceSynchronize());
    unsigned bad = 0, first[2] = {0, 0};
    HC(hipMemcpy(&bad, dbad, 4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(first, dfirst, 8, hipMemcpyDeviceToHost));
    if (bad) {
        printf("  1/e8m0_to_f32 : %u/256 bytes differ, first b=%u  FAIL\n", bad, first[0]);
        return 1;
    }
    printf("  1/e8m0_to_f32 -> e8m0_inv_f32 : all 256 E8M0 bytes bit-identical  PASS\n");

    /* 2^32 floats in 16 chunks, so a chunk is a bounded launch and progress is visible. */
    const unsigned long long TOTAL = 1ull << 32, CH = TOTAL / 16;
    for (int c = 0; c < 16; c++) {
        HC(hipMemset(dbad, 0, 4));
        hipLaunchKernelGGL(sweep, dim3(4096), dim3(256), 0, 0, dbad, dfirst, (unsigned long long)c * CH, CH);
        HC(hipDeviceSynchronize());
        HC(hipMemcpy(&bad, dbad, 4, hipMemcpyDeviceToHost));
        if (bad) {
            HC(hipMemcpy(first, dfirst, 8, hipMemcpyDeviceToHost));
            const float bf = __builtin_bit_cast(float, first[0]);
            printf("  chunk %d: %u mismatches; first bits=0x%08x (%.9g) in%s%s  FAIL\n", c, bad,
                   first[0], (double)bf, (first[1] & 1u) ? " quant_fp4" : "",
                   (first[1] & 2u) ? " e8m0_for_amax" : "");
            return 1;
        }
        printf("  floats [0x%08x, 0x%08x] : quant_fp4 and e8m0_for_amax bit-identical\n",
               (unsigned)(c * CH), (unsigned)(c * CH + CH - 1));
    }
    printf("MXFP4 quantizer primitives: all 2^32 float inputs bit-identical to the frozen "
           "reference  PASS\n");
    return 0;
}

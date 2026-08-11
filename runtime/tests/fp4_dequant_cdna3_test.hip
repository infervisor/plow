/* fp4_dequant_cdna3_test.hip — the CDNA3 mxfp4 decode, held to the e2m1 ladder on real silicon.
 *                                                                              [MXFP4-CDNA3]
 * WHAT THIS GUARDS, and why a host sweep cannot.
 *
 * gfx942 has no fp4 datatype, no `cvt_scalef32` and no scaled matrix core, so `fp4_to_bf16v8x4`
 * takes a software arm (amd_arch.h `plow_fp4x8_to_f16x4`). That arm does not select between the
 * eight e2m1 magnitudes; it places the code's 2 exponent bits into an fp16 EXPONENT FIELD and its
 * mantissa bit into the fp16 mantissa, and lets IEEE decide which side of the normal/subnormal
 * boundary the value falls on — codes 0 and 1 come out as fp16 SUBNORMALS and codes 2..7 as
 * normals, both exactly the true value times 2^-14, which the block scale absorbs.
 *
 * That identity is arithmetic and can be checked anywhere. What CANNOT be checked anywhere is
 * whether THIS PART agrees: if the hardware or the compiler's denormal mode flushes fp16
 * subnormals to zero, then code 1 — the value 0.5, one magnitude in eight — silently decodes to
 * 0.0. Nothing faults. The GEMV returns finite, plausible, wrong numbers, which is precisely the
 * failure mode `perf-data/kimi-k3-kernel-gap.md` §4c warns reads as "the model is just bad".
 *
 * So this runs the REAL `fp4_to_bf16v8x4` on device over every nibble value in every one of the 8
 * positions within a word, at E8M0 scales spanning the exponent range a quantizer emits, and
 * compares BIT-FOR-BIT against the ladder built on the host. Bit-for-bit and not a tolerance:
 * e2m1 carries <= 3 significant bits and the scale is a power of two, so every value is exact in
 * bf16 and there is no rounding argument to make. A tolerance here would hide exactly the bug the
 * file exists to catch.
 *
 * On gfx950 this compiles to the native `cvt_scalef32_pk_bf16_fp4` path and must pass unchanged —
 * that is the cross-arch claim (the two arches agree VALUE for VALUE) made checkable.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w runtime/tests/fp4_dequant_cdna3_test.hip \
 *         -o /tmp/fp4dq -Iruntime/amd -Iruntime/common
 *   perf-data/tools/gpulease -n 1 fp4dq /tmp/fp4dq
 */
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "amd_common.h"

__global__ void k_dequant(const unsigned* win, const unsigned char* sb, unsigned short* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    fp4v32 w;
#pragma unroll
    for (int j = 0; j < 4; j++) w[j] = win[i * 4 + j];
    bf16v8 a, b, c, d;
    fp4_to_bf16v8x4(w, e8m0_to_f32(sb[i]), a, b, c, d);
    bf16v8* o = (bf16v8*)(out + (size_t)i * 32);
    o[0] = a;
    o[1] = b;
    o[2] = c;
    o[3] = d;
}

static float ref_fp4(unsigned n) {
    static const float m[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    const float v = m[n & 7];
    return (n & 8) ? -v : v;
}
static float b2f(unsigned short b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
#define HC(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_) {                                                                                  \
            printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__);                            \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int main(void) {
    hipDeviceProp_t p;
    HC(hipGetDeviceProperties(&p, 0));
    printf("dev: %s\n", p.gcnArchName);

    /* Case i lays nibble value (i + j*8 + k) & 15 at position k of word j, so sweeping i over 256
     * cases puts EVERY one of the 16 codes at EVERY one of the 8 positions in a word. */
    const int NC = 256;
    unsigned hw[NC * 4];
    unsigned char hs[NC];
    for (int i = 0; i < NC; i++) {
        for (int j = 0; j < 4; j++) {
            unsigned v = 0;
            for (int k = 0; k < 8; k++) v |= ((unsigned)((i + j * 8 + k) & 15)) << (4 * k);
            hw[i * 4 + j] = v;
        }
        /* 127 is the neutral E8M0 byte; the rest span what a real weight block carries. */
        static const unsigned char sc[8] = {127, 126, 128, 120, 134, 100, 150, 1};
        hs[i] = sc[i & 7];
    }

    unsigned* dw;
    unsigned char* ds;
    unsigned short* dout;
    HC(hipMalloc(&dw, sizeof hw));
    HC(hipMalloc(&ds, sizeof hs));
    HC(hipMalloc(&dout, (size_t)NC * 32 * 2));
    HC(hipMemcpy(dw, hw, sizeof hw, hipMemcpyHostToDevice));
    HC(hipMemcpy(ds, hs, sizeof hs, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_dequant, dim3(NC / 64), dim3(64), 0, 0, dw, ds, dout);
    HC(hipDeviceSynchronize());
    unsigned short* ho = (unsigned short*)malloc((size_t)NC * 32 * 2);
    HC(hipMemcpy(ho, dout, (size_t)NC * 32 * 2, hipMemcpyDeviceToHost));

    long bad = 0, n = 0;
    for (int i = 0; i < NC; i++) {
        const float s = ldexpf(1.0f, (int)hs[i] - 127);
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 8; k++) {
                const unsigned nib = (hw[i * 4 + j] >> (4 * k)) & 0xF;
                const float want = ref_fp4(nib) * s;
                const float got = b2f(ho[(size_t)i * 32 + j * 8 + k]);
                n++;
                if (memcmp(&got, &want, 4) != 0) { /* bitwise: -0 must stay -0 */
                    if (bad < 12)
                        printf("MISMATCH case%d word%d elem%d nib=%u scale=2^%d got=%g want=%g\n", i,
                               j, k, nib, (int)hs[i] - 127, got, want);
                    bad++;
                }
            }
    }
    printf("%s: %ld halves checked over 16 codes x 8 positions x 8 scales, %ld mismatched\n",
           bad ? "FAIL" : "PASS", n, bad);
    return bad != 0;
}

/* a4w4_gfx950_test.hip — the A4W4 (MXFP4 on BOTH operands) ISA contract, asserted on hardware.
 *
 * plow's mxfp4 path is w4a16 everywhere: fp4 weights are dequantized to bf16 and fed to a bf16
 * MFMA (op_gemm.h "WFP4 (w4a16 mxfp4 weights)"). That wins weight BANDWIDTH only — the matrix
 * core still runs at the bf16 rate and activations still cross at 16 bits. CDNA4 can take fp4
 * on both operands with hardware-applied per-32-element E8M0 scales. This test pins down what
 * that instruction actually does, because every fact below was wrong or missing somewhere in
 * the tree before it was measured:
 *
 *   1. fp4 operand layout (lane -> row, k) and that only the LOW 4 dwords of the v8i32 are read.
 *   2. THE E8M0 SCALE BYTE IS BIASED BY 127. Byte 0 is 2^-127 and flushes the product to ZERO.
 *      op_gemm.h calls byte 0 "the NEUTRAL (2^0=1) MX scale, verified on device" — that comment
 *      is wrong about the ISA. It is harmless where it sits ONLY because constant scale operands
 *      make the backend select the UNSCALED v_mfma_f32_32x32x64_f8f6f4 and drop them entirely
 *      (check its disassembly: 34x unscaled, zero v_mfma_scale_*). A RUNTIME scale operand
 *      selects the scaled form and byte 0 then silently zeroes the output. That combination —
 *      a comment that is wrong about the hardware but right about its own kernel — is exactly
 *      what this file exists to stop the next A4W4 kernel from inheriting.
 *   3. per-block scales read from memory are applied EXACTLY.
 *   4. the f32 accumulator layout is the SAME as the bf16 32x32 MFMA, so every existing
 *      epilogue (mfma_acc_m / mfma_acc_n) carries over unchanged.
 *
 * Self-contained on purpose: it needs no packet, no interpreter and no fixture, so it can be
 * the FIRST thing run on a new toolchain.
 *
 *   nix develop --command bash -c '/opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 -w \
 *       runtime/tests/a4w4_gfx950_test.hip -o /tmp/a4w4_test'
 *   sg render -c 'HIP_VISIBLE_DEVICES=1 /tmp/a4w4_test'
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

typedef int   mfma_op __attribute__((ext_vector_type(8)));
typedef float f32x16  __attribute__((ext_vector_type(16)));

/* OCP e2m1: sign + {0, .5, 1, 1.5, 2, 3, 4, 6} */
__device__ __host__ static float fp4_dec(unsigned n) {
    const float T[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = T[n & 7u];
    return (n & 8u) ? -v : v;
}
__device__ __host__ static float e8m0(unsigned char b) { return ldexpf(1.0f, (int)b - 127); }

/* A[32][64] and B[32][64] fp4 (2/byte), E8M0 scale per (row, 32-k block). sa/sb < 0 means
 * "read the real per-block scale from memory"; otherwise the literal byte is forced. */
__global__ void a4w4_probe(float* out, const unsigned char* A, const unsigned char* B,
                           const unsigned char* SA, const unsigned char* SB, int sa_b, int sb_b) {
    const unsigned lane = threadIdx.x & 63u;
    const unsigned row = lane % 32u;   /* operand row  = lane % 32                  */
    const unsigned kh = lane / 32u;    /* k half       = 32*(lane/32) + [0..31]     */
    mfma_op a{}, b{};
    const unsigned char* ap = A + row * 32u + kh * 16u; /* row stride 64 fp4 = 32 B */
    const unsigned char* bp = B + row * 32u + kh * 16u;
    /* 32 fp4 = 16 B = 4 dwords, in the LOW half of the v8i32; the upper 4 are not read. */
    for (int d = 0; d < 4; ++d) {
        unsigned wa = 0, wb = 0;
        for (int j = 0; j < 4; ++j) {
            wa |= (unsigned)ap[d * 4 + j] << (8 * j);
            wb |= (unsigned)bp[d * 4 + j] << (8 * j);
        }
        a[d] = (int)wa;
        b[d] = (int)wb;
    }
    const int sa = (sa_b < 0) ? (int)SA[row * 2u + kh] : sa_b;
    const int sb = (sb_b < 0) ? (int)SB[row * 2u + kh] : sb_b;
    f32x16 acc = (f32x16)(0.0f);
    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, acc, /*cbsz*/ 4, /*blgp*/ 4, 0, sa,
                                                          0, sb);
    /* D[32][32]: lane l holds n = l%32, m = 4*(l/32) + (i%4) + 8*(i/4) — the bf16 MFMA layout. */
    const unsigned n = lane % 32u;
    for (int i = 0; i < 16; ++i) out[(4u * (lane / 32u) + (i % 4) + 8u * (i / 4)) * 32u + n] = acc[i];
}

static int fails = 0;
static void expect(const char* what, double err, bool want_match) {
    const bool ok = (err < 1e-5) == want_match;
    printf("  %-58s %s  (max rel err %.3e)\n", what, ok ? "PASS" : "FAIL", err);
    if (!ok) fails++;
}

int main() {
    const int M = 32, N = 32, K = 64;
    unsigned char A[32 * 32], B[32 * 32], SA[64], SB[64];
    for (int i = 0; i < 32 * 32; i++) {
        A[i] = (unsigned char)((i * 7 + 1) & 0x77);
        B[i] = (unsigned char)((i * 5 + 3) & 0x77);
    }
    for (int i = 0; i < 64; i++) {
        SA[i] = (unsigned char)(127 + ((i % 3) - 1));
        SB[i] = (unsigned char)(127 + (i % 2));
    }
    unsigned char *dA, *dB, *dSA, *dSB;
    float* dO;
    (void)hipMalloc(&dA, sizeof A); (void)hipMalloc(&dB, sizeof B);
    (void)hipMalloc(&dSA, sizeof SA); (void)hipMalloc(&dSB, sizeof SB);
    (void)hipMalloc(&dO, M * N * 4);
    (void)hipMemcpy(dA, A, sizeof A, hipMemcpyHostToDevice);
    (void)hipMemcpy(dB, B, sizeof B, hipMemcpyHostToDevice);
    (void)hipMemcpy(dSA, SA, sizeof SA, hipMemcpyHostToDevice);
    (void)hipMemcpy(dSB, SB, sizeof SB, hipMemcpyHostToDevice);

    float R[1024], G[1024];
    auto ref = [&](bool scaled) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                double s = 0;
                for (int k = 0; k < K; k++) {
                    const unsigned an = (A[m * 32 + k / 2] >> ((k & 1) * 4)) & 0xF;
                    const unsigned bn = (B[n * 32 + k / 2] >> ((k & 1) * 4)) & 0xF;
                    double v = (double)fp4_dec(an) * (double)fp4_dec(bn);
                    if (scaled) v *= (double)e8m0(SA[m * 2 + k / 32]) * (double)e8m0(SB[n * 2 + k / 32]);
                    s += v;
                }
                R[m * N + n] = (float)s;
            }
    };
    auto run = [&](int sa, int sb) {
        hipLaunchKernelGGL(a4w4_probe, dim3(1), dim3(64), 0, 0, dO, dA, dB, dSA, dSB, sa, sb);
        (void)hipMemcpy(G, dO, sizeof G, hipMemcpyDeviceToHost);
        double e = 0;
        for (int i = 0; i < M * N; i++) e = fmax(e, fabs(G[i] - R[i]) / (fabs(R[i]) + 1e-6));
        return e;
    };

    printf("A4W4 (MXFP4 both operands) ISA contract on gfx950:\n");
    ref(false);
    expect("E8M0 byte 127 is NEUTRAL (2^0), and the layout is as documented", run(127, 127), true);
    expect("E8M0 byte 0 is NOT neutral (it is 2^-127, i.e. output zeroed)", run(0, 0), false);
    ref(true);
    expect("per-block E8M0 scales from memory are applied EXACTLY", run(-1, -1), true);

    printf("\n%s (%d failure%s)\n", fails ? "A4W4 CONTRACT VIOLATED" : "A4W4 CONTRACT HOLDS", fails,
           fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}

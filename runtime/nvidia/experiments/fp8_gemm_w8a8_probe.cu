// rtx-07 T7 L2 probe: prove the PRODUCTION w8a8 tiled GEMM (op_gemm.cuh d_gemm_w8a8 /
// d_gemm_glu_w8a8) is bit-exact vs a dequantized-f32 reference on one full 128x128 tile —
// the cp.async staging + m16n8k32 e4m3 fragment reads + TWO-SCALE (a_scale[m]*w_scale[n])
// dequant epilogue, end to end, before porting is trusted.
//
// Method (exactness by construction, the fp8_verify trick extended to a whole tile with scales):
//   * operands are SMALL INTEGERS in [-6,6] -> exactly e4m3-representable, and every product
//     A_int*B_int (<=36) and the K-sum (<= 36*K) is exact in f32;
//   * the per-row / per-channel scales are POWERS OF TWO (0.5/1/2) -> fp8*scale is exact and the
//     epilogue multiply introduces no rounding.
//   So the kernel's (sum A_int*B_int)*a_scale[m]*w_scale[n] must BIT-MATCH the f64 reference cast
//   to f32. Any fragment-map or epilogue error shows up as a nonzero mismatch immediately.
//
//   nvcc -arch=sm_120a -O2 -I ../common -I .. -o /tmp/w8a8probe fp8_gemm_w8a8_probe.cu && /tmp/w8a8probe

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#define PLOW_NV_GEMMA 1
#define PLOW_NV_PREFILL 1
#define PLOW_NV_FA_GF 2
#include "sm120_common.cuh"
#include "op_gemm.cuh"

using bf16 = __nv_bfloat16;
static uint8_t e4m3_enc(float v) { __nv_fp8_e4m3 q(v); return *(const uint8_t*)&q; }
static float e4m3_dec(uint8_t b) { __nv_fp8_e4m3 q; *(uint8_t*)&q = b; return (float)q; }

__global__ void k_w8a8(bf16* C, const uint8_t* A, const uint8_t* B, const float* as,
                       const float* ws, unsigned M, unsigned N, unsigned K, unsigned a_row0) {
    extern __shared__ bf16 sm[];
    d_gemm_w8a8(C, A, B, as, ws, M, N, K, a_row0, blockIdx.x, gridDim.x, sm);
}
__global__ void k_w8a8_glu(bf16* C, const uint8_t* A, const uint8_t* Wg, const uint8_t* Wu,
                           const float* as, const float* sg, const float* su, unsigned M,
                           unsigned N, unsigned K, unsigned act) {
    extern __shared__ bf16 sm[];
    d_gemm_glu_w8a8(C, A, Wg, Wu, as, sg, su, M, N, K, act, blockIdx.x, gridDim.x, sm);
}

static float pow2(int p) { return ldexpf(1.0f, p); }

int main() {
    const unsigned M = 128, N = 128, K = 64; /* 2 k-tiles -> exercises the cp.async ring */
    srand(20260719);
    std::vector<uint8_t> A(M * K), B(N * K), Wg(N * K), Wu(N * K);
    std::vector<float> as(M), ws(N), sg(N), su(N);
    for (unsigned m = 0; m < M; m++) as[m] = pow2((int)(m % 3) - 1);          /* 0.5 / 1 / 2 */
    for (unsigned n = 0; n < N; n++) { ws[n] = pow2((int)(n % 3) - 1); sg[n] = pow2((int)((n + 1) % 3) - 1); su[n] = pow2((int)((n + 2) % 3) - 1); }
    auto rint = []() { return (float)(rand() % 13 - 6); };                    /* -6..6 */
    std::vector<float> Ai(M * K), Bi(N * K), Wgi(N * K), Wui(N * K);
    for (unsigned i = 0; i < M * K; i++) { Ai[i] = rint(); A[i] = e4m3_enc(Ai[i]); Ai[i] = e4m3_dec(A[i]); }
    for (unsigned i = 0; i < N * K; i++) { Bi[i] = rint(); B[i] = e4m3_enc(Bi[i]); Bi[i] = e4m3_dec(B[i]);
                                           Wgi[i] = rint(); Wg[i] = e4m3_enc(Wgi[i]); Wgi[i] = e4m3_dec(Wg[i]);
                                           Wui[i] = rint(); Wu[i] = e4m3_enc(Wui[i]); Wui[i] = e4m3_dec(Wu[i]); }

    uint8_t *dA, *dB, *dWg, *dWu; float *das, *dws, *dsg, *dsu; bf16 *dC;
    cudaMalloc(&dA, M * K); cudaMalloc(&dB, N * K); cudaMalloc(&dWg, N * K); cudaMalloc(&dWu, N * K);
    cudaMalloc(&das, M * 4); cudaMalloc(&dws, N * 4); cudaMalloc(&dsg, N * 4); cudaMalloc(&dsu, N * 4);
    cudaMalloc(&dC, M * N * sizeof(bf16));
    cudaMemcpy(dA, A.data(), M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dWg, Wg.data(), N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dWu, Wu.data(), N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(das, as.data(), M * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dws, ws.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsg, sg.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsu, su.data(), N * 4, cudaMemcpyHostToDevice);

    const size_t smem = (size_t)PGM_ARENA_BF16 * sizeof(bf16);
    cudaFuncSetAttribute(k_w8a8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(k_w8a8_glu, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    int fail = 0;

    // ---- plain GEMM: C[m][n] = (sum_k A_int*B_int) * a_scale[m] * w_scale[n] ----
    k_w8a8<<<188, 256, smem>>>(dC, dA, dB, das, dws, M, N, K, 0);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("GEMM CUDA ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    std::vector<bf16> hC(M * N); cudaMemcpy(hC.data(), dC, M * N * sizeof(bf16), cudaMemcpyDeviceToHost);
    int bad = 0; float worst = 0;
    for (unsigned m = 0; m < M; m++) for (unsigned n = 0; n < N; n++) {
        double s = 0; for (unsigned k = 0; k < K; k++) s += (double)Ai[m * K + k] * Bi[n * K + k];
        float ref = (float)(s * as[m] * ws[n]);
        float got = __bfloat162float(hC[m * N + n]);
        float rb = __bfloat162float(__float2bfloat16(ref)); /* the kernel rounds to bf16 at the store */
        float d = fabsf(got - rb); if (d > worst) worst = d; if (d != 0.f) bad++;
    }
    printf("w8a8 GEMM  128x128x64  exact-mismatches=%4d / %u   max|err|=%g   %s\n",
           bad, M * N, worst, bad ? "FAIL" : "PASS");
    fail |= (bad != 0);

    // ---- GLU: gelu-tanh(a_scale*sg*gate) * (a_scale*su*up) ----
    k_w8a8_glu<<<188, 256, smem>>>(dC, dA, dWg, dWu, das, dsg, dsu, M, N, K, PLOW_ACT_GELU_TANH_);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("GLU CUDA ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(hC.data(), dC, M * N * sizeof(bf16), cudaMemcpyDeviceToHost);
    bad = 0; worst = 0; double reld = 0, refe = 0;
    for (unsigned m = 0; m < M; m++) for (unsigned n = 0; n < N; n++) {
        double g = 0, u = 0;
        for (unsigned k = 0; k < K; k++) { g += (double)Ai[m * K + k] * Wgi[n * K + k]; u += (double)Ai[m * K + k] * Wui[n * K + k]; }
        float gg = (float)(g * as[m] * sg[n]);
        float gv = 0.5f * gg * (1.0f + tanhf(0.7978845608f * (gg + 0.044715f * gg * gg * gg)));
        float ref = gv * (float)(u * as[m] * su[n]);
        float got = __bfloat162float(hC[m * N + n]);
        float d = fabsf(got - ref); if (d > worst) worst = d; reld += (double)d * d; refe += (double)ref * ref;
        if (fabsf(d) > 0.06f * (fabsf(ref) + 1e-3f)) bad++;
    }
    printf("w8a8 GLU   128x128x64  relL2=%.3e  max|err|=%g  band-fails=%d   %s\n",
           sqrt(reld / (refe + 1e-30)), worst, bad, bad ? "FAIL" : "PASS(gelu-tanh tol)");
    fail |= (bad != 0);

    printf("%s\n", fail ? "w8a8 probe: FAIL" : "w8a8 probe: ok");
    return fail;
}

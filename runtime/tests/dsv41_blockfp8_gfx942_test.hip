/* DeepSeek-V4.1's [32,32] ue8m0 block-fp8 GEMM against a host reference.  [DSV41-BLKFP8]
 *
 * V4.1 quantizes block-fp8 at `weight_block_size: [32, 32]` with UE8M0 scales, where GLM-5.2 and
 * DeepSeek-V3 use [128, 128] with arbitrary f32 `weight_scale_inv`. BOTH differences are real and
 * EITHER ALONE silently rescales every output -- the shapes stay self-consistent, nothing crashes,
 * and the model merely gets worse. From the released shards: `attn.wq_a.weight` is [1280, 5120]
 * and its `.scale` is `F8_E8M0 [40, 160]`, i.e. 1280/32 x 5120/32.
 *
 * `d_gemm_fp8_mx` (-> `d_gemm_t<WFP8MX>`) is the arm that reads that convention. It gates 39.8% of
 * a V4.1 8k prefill -- every projection in the model (wq_a/wq_b/wkv/wo_a/wo_b, the shared expert,
 * engram.wkv, indexer.wq_b). The routed experts do NOT need it; they are MXFP4.
 *
 * The reference decodes e4m3 and ue8m0 FROM FIRST PRINCIPLES rather than calling the runtime's
 * helpers: an oracle that calls the code under test proves nothing about the convention, which is
 * precisely what is being checked here.
 *
 * In nix develop:
 *   $PLOW_HIPCC --offload-arch=gfx942 -O3 -w -std=c++17 -Iruntime/amd -Iruntime/common \
 *       -DPLOW_WG_WAVES=8 -c runtime/tests/dsv41_blockfp8_gfx942_test.hip -o /tmp/mx.o
 *   c++ /tmp/mx.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/mx
 *   perf-data/tools/gpulease -n 1 dsv41-blkfp8 /tmp/mx
 */
#ifndef PLOW_WG_WAVES
#define PLOW_WG_WAVES 8
#endif
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
/* `op_gemm.h`, NOT `op_gemm_common.h`: the common body is shared and the arch header is what sets
 * the tile defaults it needs (it `#error`s if included directly). */
#include "amd_common.h"
#include "op_gemm.h"
#else
using bf16 = unsigned short;
#endif
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void ck(hipError_t s) {
    if (s != hipSuccess) {
        std::fprintf(stderr, "hip: %s\n", hipGetErrorString(s));
        std::exit(1);
    }
}

static int fails = 0;

#define V41_BLK 32

/* The wrapper maps (blockIdx, gridDim) onto the interpreter's (slice, nblk) -- the only difference
 * between a test launch and a packet, exactly as `test_kernels.hip` does. */
__global__ __launch_bounds__(PLOW_WG_WAVES * 64) void k_gemm_fp8_mx(bf16* C, const bf16* A,
                                                                    const unsigned char* W,
                                                                    const unsigned char* S,
                                                                    unsigned M, unsigned N,
                                                                    unsigned K) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[GM_LDS_HALVES_T(GM_MX_BM, GM_MX_BN, 32)];
    d_gemm_fp8_mx(C, A, W, S, M, N, K, blockIdx.x, gridDim.x, lds);
#endif
}

static float bf2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static bf16 f2bf(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (bf16)(u >> 16);
}

/* OCP e4m3 (torch.float8_e4m3fn), written out rather than shared with the kernel. */
static double e4m3_decode(unsigned char b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    double v;
    if (e == 0) v = (m / 8.0) * 0.015625;
    else v = (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return s ? -v : v;
}

/* UE8M0: a BARE 8-bit exponent, value 2^(b-127). The device spells this as a bit-cast of
 * `b << 23`; spelled as an ldexp here so the two are independent derivations of one number. */
static double e8m0_decode(unsigned char b) { return std::ldexp(1.0, (int)b - 127); }

/* C[m][n] = sum_k A[m][k] * e4m3(W[n][k]) * e8m0(S[(n/32)*KB + (k/32)]).
 *
 * The reference SAMPLES (m, n) pairs: at the real projection shapes the full product is tens of
 * G f64 MACs on one core, and a tiling, swizzle or scale-index bug is SYSTEMATIC across the output
 * rather than concentrated in a few elements. */
static void mx_case(const char* label, unsigned M, unsigned N, unsigned K) {
    const unsigned NB = (N + V41_BLK - 1u) / V41_BLK, KB = (K + V41_BLK - 1u) / V41_BLK;
    const size_t nW = (size_t)N * K, nS = (size_t)NB * KB, nA = (size_t)M * K, nC = (size_t)M * N;

    std::vector<unsigned char> hW(nW), hS(nS);
    std::vector<bf16> hA(nA), hC(nC);

    /* Small e4m3 magnitudes (exp field <= 7, |v| < 2) so a K-long dot stays well-conditioned and a
     * real layout bug shows as ~100% error rather than a few percent of rounding noise. */
    for (size_t i = 0; i < nW; i++) {
        const unsigned e = rand() % 8, m = rand() % 8, s = rand() % 2;
        hW[i] = (unsigned char)((s << 7) | (e << 3) | m);
    }
    for (size_t i = 0; i < nA; i++) hA[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    /* Exponents spread around 2^0, WIDE enough that a misread block picks up a visibly different
     * scale. A narrow spread would let a wrong (n>>7 instead of n>>5) index pass by luck. */
    for (size_t i = 0; i < nS; i++) hS[i] = (unsigned char)(127 - 4 + (rand() % 9));

    unsigned char *dW, *dS;
    bf16 *dA, *dC;
    ck(hipMalloc(&dW, nW));
    ck(hipMalloc(&dS, nS));
    ck(hipMalloc(&dA, nA * 2));
    ck(hipMalloc(&dC, nC * 2));
    ck(hipMemcpy(dW, hW.data(), nW, hipMemcpyHostToDevice));
    ck(hipMemcpy(dS, hS.data(), nS, hipMemcpyHostToDevice));
    ck(hipMemcpy(dA, hA.data(), nA * 2, hipMemcpyHostToDevice));
    ck(hipMemset(dC, 0, nC * 2));

    /* 304 CUs, one workgroup each -- the interpreter's (slice, nblk) mapping. The thread count
     * is spelled from PLOW_WG_WAVES here rather than PLOW_THREADS, which only exists on the
     * device side of this translation unit. */
    const unsigned GRID = 304, THREADS = PLOW_WG_WAVES * 64;
    hipLaunchKernelGGL(k_gemm_fp8_mx, dim3(GRID), dim3(THREADS), 0, 0, dC, dA, dW, dS, M, N, K);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    ck(hipMemcpy(hC.data(), dC, nC * 2, hipMemcpyDeviceToHost));

    double worst = 0.0;
    const unsigned SM = M < 16 ? M : 16, SN = N < 16 ? N : 16;
    for (unsigned mi = 0; mi < SM; mi++) {
        const unsigned m = (unsigned)((uint64_t)mi * M / SM);
        for (unsigned ni = 0; ni < SN; ni++) {
            const unsigned n = (unsigned)((uint64_t)ni * N / SN);
            double want = 0.0;
            for (unsigned kk = 0; kk < K; kk++)
                want += (double)bf2f(hA[(size_t)m * K + kk]) * e4m3_decode(hW[(size_t)n * K + kk]) *
                        e8m0_decode(hS[(size_t)(n / V41_BLK) * KB + (kk / V41_BLK)]);
            const double got = bf2f(hC[(size_t)m * N + n]);
            const double rel = std::fabs(got - want) / (std::fabs(want) + 1e-2);
            if (rel > worst) worst = rel;
        }
    }
    const bool ok = worst < 3e-2;
    std::printf("  %-26s M=%5u N=%5u K=%5u  %s  worst rel %.4e\n", label, M, N, K,
                ok ? "PASS" : "FAIL", worst);
    if (!ok) fails++;

    ck(hipFree(dW));
    ck(hipFree(dS));
    ck(hipFree(dA));
    ck(hipFree(dC));
}

int main() {
    std::srand(1234);
    std::printf("DeepSeek-V4.1 block-fp8 [32,32] ue8m0 GEMM vs a host reference\n");
    /* V4.1's REAL projection shapes at tp=8, with M a prefill bucket rather than 8192 (the
     * reference is f64 on one core). K is always a 32-multiple, which every V4.1 projection is by
     * construction -- the scale grid could not exist otherwise. */
    mx_case("wq_a  5120->1280", 512, 1280, 5120);
    mx_case("wq_b  1280->4096 tp8", 512, 4096, 1280);
    mx_case("wkv   5120->512 thin", 512, 512, 5120);
    mx_case("wo_a  4096->1024 grp", 512, 1024, 4096);
    mx_case("wo_b  1024->5120 tp8", 512, 5120, 1024);
    mx_case("shared g/u 5120->576", 512, 576, 5120);
    mx_case("shared down 288->5120", 512, 5120, 288);
    mx_case("indexer wq_b 1280->4096", 512, 4096, 1280);
    /* engram.wkv: 24 gathered rows x 256 = the widest K among the projections. */
    mx_case("engram wkv 6144->25600", 64, 25600, 6144);
    /* A larger M, where the tile is wave-bound rather than tail-bound. */
    mx_case("wo_b M=2048", 2048, 5120, 1024);
    /* Ragged M and N (K stays a 32-multiple): tile tails on both output axes at once... */
    mx_case("ragged M,N tails", 100, 130, 256);
    /* ...and an N that is NOT a multiple of the 32-wide scale block, which is where a per-lane
     * N-block index that assumed alignment would break. */
    mx_case("ragged N=33 (blk+1)", 64, 33, 128);

    std::printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

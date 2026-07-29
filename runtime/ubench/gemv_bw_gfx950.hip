/* gemv_bw_gfx950.hip — what fraction of HBM the DECODE GEMV family actually reaches.
 *
 * WHY THIS EXISTS. Gemma-4 31B decode is 92% GEMV-family and streams every weight byte per
 * token, so token latency IS weights / achieved-bandwidth. End to end plow reaches 3490 GB/s
 * against the 6200 GB/s this machine was MEASURED to give (op_gemm.h's ceiling note — not the
 * datasheet 8 TB/s, which overstates the headroom). vLLM implies ~4546 GB/s on the same model
 * and box. That is a 56%-vs-73% gap, and every cheap explanation is already spent: loads are
 * b128 (`buf_ld8`), GV_UNROLL=11 is tuned to divide K=5376 exactly, and 16 waves is arithmetically
 * out of reach (4 waves/SIMD caps the kernel at 128 registers; the decode object is at 248).
 *
 * So the question is no longer "which knob" but "is the loop already near what this ACCESS
 * PATTERN will give". This measures that directly:
 *
 *   stream    a pure grid-strided read of the same bytes with the same 16-B-per-lane loads and
 *             the same unroll. This is the CEILING FOR THIS ACCESS PATTERN, which is the honest
 *             denominator -- not a vendor number, and not a different kernel's number.
 *   gemv      the real d_gemv body, one op, whole GPU, at the real decode shapes.
 *   gemv_glu  the fused gate|up body (two weight streams, one activation).
 *   gemv_qkv  the fused q|k|v body (three weight streams, one activation).
 *
 * The gap between `stream` and `gemv` is what the GEMV loop loses to everything that is not
 * the load: issue slots, the wave reduction, the epilogue, LDS staging of x. The gap between
 * `stream` and 6200 is what the access pattern itself loses. Reporting both separates
 * "the kernel is leaving bandwidth on the table" from "this pattern does not reach peak",
 * which is exactly the distinction that decides whether single-stream decode is worth chasing.
 *
 * Self-contained: hipcc gemv_bw_gfx950.hip -o gemv_bw --offload-arch=gfx950 -I../amd -I../common
 * Run under ROCR_VISIBLE_DEVICES alone; do NOT also set HIP_VISIBLE_DEVICES (it breaks ROCr
 * device enumeration for these binaries).
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

/* op_gemm.h is DEVICE-ONLY here. Its buffer-resource helpers cast through
 * `__attribute__((address_space(1)))`, which is meaningless in the host pass — hipcc parses the
 * same TU twice and the host pass rejects 20 of those casts. test_kernels.hip never hit this
 * because it is built `--genco` (device only) and loaded through ROCr; this benchmark is a plain
 * host+device program on purpose, so it needs the split. amd_common.h is safe in both passes:
 * `as_glob` is a template and is only instantiated where it is used. */
#include "amd_common.h"
#ifdef __HIP_DEVICE_COMPILE__
#include "op_gemm.h"
#define BODY(x) x
#else
#define BODY(x)
#endif

#ifndef __HIP_DEVICE_COMPILE__
/* Host-visible mirrors of the launch geometry the device pass gets from op_gemm.h. */
#ifndef GV_UNROLL
#define GV_UNROLL 11
#endif
#endif

#define HIPOK(e)                                                                                  \
    do {                                                                                          \
        hipError_t e_ = (e);                                                                      \
        if (e_ != hipSuccess) {                                                                   \
            printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__);          \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

/* The same LDS arena the interpreter gives an op, so staging behaviour matches. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void k_gemv(
    bf16* C, const bf16* x, const bf16* W, unsigned M, unsigned N, unsigned K) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv(C, x, W, nullptr, nullptr, M, N, K, 0, 0.0f, blockIdx.x, gridDim.x, lds);
#else
    (void)C; (void)x; (void)W; (void)M; (void)N; (void)K;
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void k_gemv_glu(
    bf16* C, const bf16* x, const bf16* Wg, const bf16* Wu, unsigned M, unsigned N, unsigned K) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_glu(C, x, Wg, Wu, M, N, K, PLOW_ACT_SILU_, blockIdx.x, gridDim.x, lds);
#else
    (void)C; (void)x; (void)Wg; (void)Wu; (void)M; (void)N; (void)K;
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void k_gemv_qkv(
    bf16* Cq, bf16* Ck, bf16* Cv, const bf16* x, const bf16* Wq, const bf16* Wk, const bf16* Wv,
    unsigned M, unsigned Nq, unsigned Nk, unsigned Nv, unsigned K) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemv_qkv(Cq, Ck, Cv, x, Wq, Wk, Wv, M, Nq, Nk, Nv, K, blockIdx.x, gridDim.x, lds);
#else
    (void)Cq;(void)Ck;(void)Cv;(void)x;(void)Wq;(void)Wk;(void)Wv;(void)M;(void)Nq;(void)Nk;(void)Nv;(void)K;
#endif
}

/* THE CEILING FOR THIS ACCESS PATTERN. Reads exactly the bytes the GEMV reads, in the same
 * 16-B-per-lane buffer loads, with the same number in flight (UN), and does the minimum with
 * them that the compiler cannot delete. No reduction, no epilogue, no LDS. */
template <int UN>
__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void k_stream(float* out,
                                                                        const bf16* __restrict__ W,
                                                                        unsigned N, unsigned K) {
#ifdef __HIP_DEVICE_COMPILE__
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const size_t rows = (size_t)N;
    float acc = 0.0f;
    for (size_t n = (size_t)blockIdx.x * PLOW_WAVES + wave; n < rows; n += (size_t)gridDim.x * PLOW_WAVES) {
        const bf16* row = W + n * K;
        const unsigned step = PLOW_WAVE * 8; /* 64 lanes x 8 halves = one 16-B load each */
        unsigned k = lane * 8;
        for (; k + (UN - 1) * step < K; k += UN * step) {
            bf16v8 v[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) __builtin_memcpy(&v[u], row + k + u * step, 16);
#pragma unroll
            for (int u = 0; u < UN; u++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc += bf2f(v[u][j]);
        }
        for (; k < K; k += step) {
            bf16v8 v;
            __builtin_memcpy(&v, row + k, 16);
#pragma unroll
            for (int j = 0; j < 8; j++) acc += bf2f(v[j]);
        }
    }
    /* Never true; keeps `acc` live so nothing above is dead-code-eliminated. */
    if (acc == 1.2345e-30f) out[0] = acc;
#else
    (void)out; (void)W; (void)N; (void)K;
#endif
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 9;
    int dev = 0;
    HIPOK(hipGetDevice(&dev));
    hipDeviceProp_t p;
    HIPOK(hipGetDeviceProperties(&p, dev));
    const unsigned nblk = p.multiProcessorCount;
    printf("gemv_bw: %s  CUs=%u  waves/WG=%d  GV_UNROLL=%d\n", p.gcnArchName, nblk, PLOW_WAVES,
           GV_UNROLL);
    printf("  ceiling for reference: 6200 GB/s (op_gemm.h, MEASURED on this machine)\n\n");

    /* Gemma-4 31B decode shapes. `bytes` is the WEIGHT stream, which is what decode is bound by. */
    struct Shape { const char* name; unsigned N, K; } shapes[] = {
        {"q_proj      (N=16384 K=5376)", 16384, 5376},
        {"kv_proj     (N= 2048 K=5376)", 2048, 5376},
        {"o_proj      (N= 5376 K=8192)", 5376, 8192},
        {"gate|up     (N=21504 K=5376)", 21504, 5376},
        {"down_proj   (N= 5376 K=21504)", 5376, 21504},
        {"lm_head     (N=262144 K=5376)", 262144, 5376},
    };

    size_t maxw = 0;
    for (auto& s : shapes) maxw = std::max(maxw, (size_t)s.N * s.K);
    bf16 *dW = nullptr, *dW2 = nullptr, *dx = nullptr, *dC = nullptr, *dC2 = nullptr, *dC3 = nullptr;
    float* dout = nullptr;
    HIPOK(hipMalloc(&dW, maxw * 2));
    HIPOK(hipMalloc(&dW2, maxw * 2));
    HIPOK(hipMalloc(&dx, (size_t)32768 * 2));
    HIPOK(hipMalloc(&dC, (size_t)262144 * 2));
    HIPOK(hipMalloc(&dC2, (size_t)262144 * 2));
    HIPOK(hipMalloc(&dC3, (size_t)262144 * 2));
    HIPOK(hipMalloc(&dout, 4));
    HIPOK(hipMemset(dW, 0x3c, maxw * 2));
    HIPOK(hipMemset(dW2, 0x3c, maxw * 2));
    HIPOK(hipMemset(dx, 0x3c, 32768 * 2));

    hipEvent_t a, b;
    HIPOK(hipEventCreate(&a));
    HIPOK(hipEventCreate(&b));

    auto run = [&](const char* tag, size_t bytes, auto&& fn) {
        std::vector<double> ms;
        for (int r = 0; r < reps; r++) {
            fn(); /* warm */
            HIPOK(hipDeviceSynchronize());
            HIPOK(hipEventRecord(a));
            fn();
            HIPOK(hipEventRecord(b));
            HIPOK(hipEventSynchronize(b));
            float t = 0;
            HIPOK(hipEventElapsedTime(&t, a, b));
            ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        const double gbs = bytes / (med / 1e3) / 1e9;
        printf("    %-14s %8.3f ms  %7.0f GB/s  %5.1f%% of 6200\n", tag, med, gbs, gbs / 6200 * 100);
        return gbs;
    };

    for (auto& s : shapes) {
        const size_t wb = (size_t)s.N * s.K * 2;
        printf("  %s   weights %.2f GiB\n", s.name, wb / 1073741824.0);
        const double sref =
            run("stream", wb, [&] { hipLaunchKernelGGL((k_stream<GV_UNROLL>), dim3(nblk), dim3(PLOW_THREADS), 0, 0, dout, dW, s.N, s.K); });
        const double g =
            run("gemv", wb, [&] { hipLaunchKernelGGL(k_gemv, dim3(nblk), dim3(PLOW_THREADS), 0, 0, dC, dx, dW, 1u, s.N, s.K); });
        printf("      -> gemv reaches %.1f%% of the same-pattern stream ceiling\n\n", g / sref * 100);
    }

    /* Fused arms: the weight stream is 2x / 3x, the activation is read once. */
    {
        const unsigned N = 21504, K = 5376;
        const size_t wb = (size_t)N * K * 2 * 2;
        printf("  gate|up FUSED (N=%u K=%u, two weight streams)   weights %.2f GiB\n", N, K,
               wb / 1073741824.0);
        run("gemv_glu", wb, [&] {
            hipLaunchKernelGGL(k_gemv_glu, dim3(nblk), dim3(PLOW_THREADS), 0, 0, dC, dx, dW, dW2, 1u, N, K);
        });
        printf("\n");
    }
    {
        const unsigned Nq = 16384, Nk = 2048, Nv = 2048, K = 5376;
        const size_t wb = ((size_t)Nq + Nk + Nv) * K * 2;
        printf("  q|k|v FUSED (Nq=%u Nk=Nv=%u K=%u)   weights %.2f GiB\n", Nq, Nk, K,
               wb / 1073741824.0);
        run("gemv_qkv", wb, [&] {
            hipLaunchKernelGGL(k_gemv_qkv, dim3(nblk), dim3(PLOW_THREADS), 0, 0, dC, dC2, dC3, dx,
                               dW, dW2, dW2, 1u, Nq, Nk, Nv, K);
        });
        printf("\n");
    }
    printf("READ: 'stream' is the ceiling FOR THIS ACCESS PATTERN. gemv/stream isolates what the\n"
           "GEMV loop loses to issue, reduction and epilogue; stream/6200 is what the pattern\n"
           "itself loses. They are different problems and only the first is a kernel bug.\n");
    return 0;
}

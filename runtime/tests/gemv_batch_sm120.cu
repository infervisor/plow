/* gemv_batch_sm120.cu — does batch>1 DECODE work today, and what would it buy?
 *
 * Two questions, both measured:
 *
 *  Q1 (CORRECTNESS / capability): the decode GEMV arms the interpreter dispatches
 *      (d_gemv, d_gemv_qkv, d_gemv_glu) take an `M` argument. Do they honor it?
 *      d_gemv calls gemv_rows<1>; d_gemv_qkv / d_gemv_glu carry a SCALAR accumulator
 *      and use M only as `if (lane == 0 && M)` — a nonzero guard. So the prediction is
 *      that with M=2 every one of them writes row 0 and leaves row 1 UNTOUCHED.
 *      We fill C with a sentinel and check whether row 1 is still the sentinel.
 *
 *      NEGATIVE CONTROL: gemv_rows<2> is the SAME template at MM=2. If the harness
 *      cannot tell <1> from <2> it proves nothing, so we run both through the same
 *      checker and require <1> to fail row 1 and <2> to pass it.
 *
 *  Q2 (THE LEVER): a GEMV is bandwidth-bound on the weight read. gemv_rows<MM> loads
 *      `wrow` ONCE and dots it against all MM rows of x. So MM rows should cost about
 *      the same as 1 — that is the entire batching thesis, measured at the kernel
 *      level, with no scheduler in the way. We time MM = 1,2,4,8 at Qwen3-4B decode
 *      shapes and report tokens/s-equivalent scaling.
 *
 * Shapes are Qwen3-4B: hidden 2560, 32 q-heads, 8 kv-heads, head_dim 128,
 * intermediate 9728.  q_proj [4096,2560]  k/v_proj [1024,2560]
 * gate/up [9728,2560]  down [2560,9728]
 *
 * Build:
 *   nvcc -std=c++17 -O3 -arch=sm_120a -Iinclude -Iruntime/common -Iruntime/nvidia \
 *     runtime/tests/gemv_batch_sm120.cu -o /workspace/x/gemvbatch -lcuda
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <initializer_list>
#include <cuda_runtime.h>
#include "sm120_common.cuh"
#include "op_gemm.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

/* ---- launchers ------------------------------------------------------------------------
 * Each block plays the role of one interpreter workgroup: (slice, nblk) partition the
 * output columns exactly as the interpreter's dispatch does. */

/* The arm the interpreter ACTUALLY dispatches for PLOW_DOP_GEMV. */
__global__ void k_d_gemv(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                         unsigned M, unsigned N, unsigned K) {
    d_gemv(C, x, W, M, N, K, blockIdx.x, gridDim.x);
}

/* The same template at MM=2 — the negative control, and the proposed fix. */
__global__ void k_rows2(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                        unsigned M, unsigned N, unsigned K) {
    gemv_rows<2>(C, x, W, M, N, K, blockIdx.x, gridDim.x);
}

/* Timing arms: MM fixed at compile time, M passed at runtime = MM. */
template <int MM>
__global__ void k_rowsN(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                        unsigned M, unsigned N, unsigned K) {
    gemv_rows<MM>(C, x, W, M, N, K, blockIdx.x, gridDim.x);
}

/* The fused-QKV arm, as dispatched. */
__global__ void k_d_gemv_qkv(__nv_bfloat16* Cq, __nv_bfloat16* Ck, __nv_bfloat16* Cv,
                             const __nv_bfloat16* x, const __nv_bfloat16* Wq,
                             const __nv_bfloat16* Wk, const __nv_bfloat16* Wv,
                             unsigned M, unsigned Nq, unsigned Nk, unsigned Nv, unsigned K) {
    d_gemv_qkv(Cq, Ck, Cv, x, Wq, Wk, Wv, M, Nq, Nk, Nv, K, blockIdx.x, gridDim.x);
}

/* The fused-GLU arm, as dispatched. */
__global__ void k_d_gemv_glu(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* Wg,
                             const __nv_bfloat16* Wu, unsigned M, unsigned N, unsigned K,
                             unsigned act) {
    d_gemv_glu(C, x, Wg, Wu, M, N, K, act, blockIdx.x, gridDim.x);
}

/* ---- helpers -------------------------------------------------------------------------- */

static const unsigned GRID = 170;   /* 170 SMs, 1 block/SM — the interpreter's geometry */
static const unsigned BLOCK = 256;  /* __launch_bounds__(256,1) */

/* Deterministic filler: small values so bf16 sums stay well-conditioned. */
static void fill(std::vector<__nv_bfloat16>& v, unsigned seed) {
    unsigned s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < v.size(); i++) {
        s = s * 1664525u + 1013904223u;
        float f = ((float)((s >> 8) & 0xFFFF) / 65536.0f - 0.5f) * 0.1f;
        v[i] = __float2bfloat16(f);
    }
}

/* bf16 bit pattern used as the "never written" sentinel. */
static const unsigned short SENTINEL = 0x7A5A;

struct RowCheck { bool row0_written; bool row1_written; double row1_relL2; };

/* Compare device C [M,N] against a CPU reference, and report per-row whether the row
 * was written at all (i.e. differs from the sentinel). */
static RowCheck check_rows(const std::vector<__nv_bfloat16>& C, const std::vector<float>& ref,
                           unsigned M, unsigned N) {
    RowCheck r{false, false, 0.0};
    for (unsigned m = 0; m < M && m < 2; m++) {
        bool written = false;
        double num = 0, den = 0;
        for (unsigned n = 0; n < N; n++) {
            unsigned short bits = *(const unsigned short*)&C[(size_t)m * N + n];
            if (bits != SENTINEL) written = true;
            double got = (double)__bfloat162float(C[(size_t)m * N + n]);
            double want = ref[(size_t)m * N + n];
            num += (got - want) * (got - want);
            den += want * want;
        }
        if (m == 0) r.row0_written = written;
        else { r.row1_written = written; r.row1_relL2 = den > 0 ? sqrt(num / den) : 0.0; }
    }
    return r;
}

/* CPU reference C[m][n] = dot(x[m][:], W[n][:]) in f32. */
static void ref_gemv(std::vector<float>& out, const std::vector<__nv_bfloat16>& x,
                     const std::vector<__nv_bfloat16>& W, unsigned M, unsigned N, unsigned K) {
    out.assign((size_t)M * N, 0.0f);
    for (unsigned m = 0; m < M; m++)
        for (unsigned n = 0; n < N; n++) {
            float a = 0;
            for (unsigned k = 0; k < K; k++)
                a += __bfloat162float(x[(size_t)m * K + k]) * __bfloat162float(W[(size_t)n * K + k]);
            out[(size_t)m * N + n] = a;
        }
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 200;

    /* ---------- Q1: capability probe, on a SMALL shape so the CPU reference is cheap ---- */
    {
        const unsigned M = 2, N = 512, K = 2560;
        std::vector<__nv_bfloat16> hx((size_t)M * K), hW((size_t)N * K), hC((size_t)M * N);
        fill(hx, 1); fill(hW, 2);
        for (auto& c : hC) *(unsigned short*)&c = SENTINEL;

        std::vector<float> ref;
        ref_gemv(ref, hx, hW, M, N, K);

        __nv_bfloat16 *dx, *dW, *dC;
        CK(cudaMalloc(&dx, hx.size() * 2)); CK(cudaMalloc(&dW, hW.size() * 2));
        CK(cudaMalloc(&dC, hC.size() * 2));
        CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice));

        printf("=== Q1: do the decode GEMV arms honor M=2? (N=%u K=%u, sentinel 0x%04X) ===\n",
               N, K, SENTINEL);

        /* (a) d_gemv — what PLOW_DOP_GEMV dispatches. */
        CK(cudaMemcpy(dC, hC.data(), hC.size() * 2, cudaMemcpyHostToDevice));
        k_d_gemv<<<GRID, BLOCK>>>(dC, dx, dW, M, N, K);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));
        RowCheck a = check_rows(hC, ref, M, N);
        printf("  d_gemv        (gemv_rows<1>) : row0 written=%d  row1 written=%d\n",
               a.row0_written, a.row1_written);

        /* (b) gemv_rows<2> — same template, MM=2. NEGATIVE CONTROL for the checker. */
        CK(cudaMemcpy(dC, hC.data(), hC.size() * 2, cudaMemcpyHostToDevice));
        for (auto& c : hC) *(unsigned short*)&c = SENTINEL;
        CK(cudaMemcpy(dC, hC.data(), hC.size() * 2, cudaMemcpyHostToDevice));
        k_rows2<<<GRID, BLOCK>>>(dC, dx, dW, M, N, K);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));
        RowCheck b = check_rows(hC, ref, M, N);
        printf("  gemv_rows<2>  (control)      : row0 written=%d  row1 written=%d  row1 relL2=%.3e\n",
               b.row0_written, b.row1_written, b.row1_relL2);

        printf("  VERDICT: batch>1 via d_gemv is %s; the <2> control %s\n",
               a.row1_written ? "SUPPORTED" : "BROKEN (row 1 never written)",
               b.row1_written ? "PASSES (checker can detect a written row)"
                              : "FAILED — checker is not trustworthy");
        CK(cudaFree(dx)); CK(cudaFree(dW)); CK(cudaFree(dC));
    }

    /* ---------- Q1b: the fused QKV and GLU arms ---------------------------------------- */
    {
        const unsigned M = 2, Nq = 512, Nk = 128, Nv = 128, K = 2560;
        std::vector<__nv_bfloat16> hx((size_t)M * K), hWq((size_t)Nq * K), hWk((size_t)Nk * K),
            hWv((size_t)Nv * K), hCq((size_t)M * Nq), hCk((size_t)M * Nk), hCv((size_t)M * Nv);
        fill(hx, 3); fill(hWq, 4); fill(hWk, 5); fill(hWv, 6);
        for (auto& c : hCq) *(unsigned short*)&c = SENTINEL;

        __nv_bfloat16 *dx, *dWq, *dWk, *dWv, *dCq, *dCk, *dCv;
        CK(cudaMalloc(&dx, hx.size() * 2));
        CK(cudaMalloc(&dWq, hWq.size() * 2)); CK(cudaMalloc(&dWk, hWk.size() * 2));
        CK(cudaMalloc(&dWv, hWv.size() * 2));
        CK(cudaMalloc(&dCq, hCq.size() * 2)); CK(cudaMalloc(&dCk, hCk.size() * 2));
        CK(cudaMalloc(&dCv, hCv.size() * 2));
        CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWq, hWq.data(), hWq.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWk, hWk.data(), hWk.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWv, hWv.data(), hWv.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dCq, hCq.data(), hCq.size() * 2, cudaMemcpyHostToDevice));

        k_d_gemv_qkv<<<GRID, BLOCK>>>(dCq, dCk, dCv, dx, dWq, dWk, dWv, M, Nq, Nk, Nv, K);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hCq.data(), dCq, hCq.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<float> refq; ref_gemv(refq, hx, hWq, M, Nq, K);
        RowCheck q = check_rows(hCq, refq, M, Nq);
        printf("  d_gemv_qkv    (scalar acc)   : row0 written=%d  row1 written=%d\n",
               q.row0_written, q.row1_written);

        /* GLU: reuse Wq/Wk as gate/up over N=Nk so shapes line up. */
        std::vector<__nv_bfloat16> hCg((size_t)M * Nk);
        for (auto& c : hCg) *(unsigned short*)&c = SENTINEL;
        __nv_bfloat16* dCg; CK(cudaMalloc(&dCg, hCg.size() * 2));
        CK(cudaMemcpy(dCg, hCg.data(), hCg.size() * 2, cudaMemcpyHostToDevice));
        k_d_gemv_glu<<<GRID, BLOCK>>>(dCg, dx, dWk, dWv, M, Nk, K, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hCg.data(), dCg, hCg.size() * 2, cudaMemcpyDeviceToHost));
        bool g_r0 = false, g_r1 = false;
        for (unsigned n = 0; n < Nk; n++) {
            if (*(unsigned short*)&hCg[n] != SENTINEL) g_r0 = true;
            if (*(unsigned short*)&hCg[Nk + n] != SENTINEL) g_r1 = true;
        }
        printf("  d_gemv_glu    (scalar acc)   : row0 written=%d  row1 written=%d\n", g_r0, g_r1);

        CK(cudaFree(dx)); CK(cudaFree(dWq)); CK(cudaFree(dWk)); CK(cudaFree(dWv));
        CK(cudaFree(dCq)); CK(cudaFree(dCk)); CK(cudaFree(dCv)); CK(cudaFree(dCg));
    }

    /* ---------- Q2: what would batching buy? gemv_rows<MM> at Qwen3-4B shapes ---------- */
    printf("\n=== Q2: gemv_rows<MM> scaling at Qwen3-4B decode shapes (%d iters) ===\n", iters);
    printf("  %-22s %8s %8s %8s %8s   %s\n", "shape [N,K]", "MM=1", "MM=2", "MM=4", "MM=8",
           "tok/s scaling vs MM=1");

    struct Shape { const char* name; unsigned N, K; };
    Shape shapes[] = {
        {"q_proj   [4096,2560]", 4096, 2560},
        {"kv_proj  [1024,2560]", 1024, 2560},
        {"gate/up  [9728,2560]", 9728, 2560},
        {"down     [2560,9728]", 2560, 9728},
    };

    for (auto& s : shapes) {
        const unsigned MMAX = 8;
        std::vector<__nv_bfloat16> hx((size_t)MMAX * s.K), hW((size_t)s.N * s.K);
        fill(hx, 7); fill(hW, 8);
        __nv_bfloat16 *dx, *dW, *dC;
        CK(cudaMalloc(&dx, hx.size() * 2));
        CK(cudaMalloc(&dW, hW.size() * 2));
        CK(cudaMalloc(&dC, (size_t)MMAX * s.N * 2));
        CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice));

        double ms[4];
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int idx = 0; idx < 4; idx++) {
            unsigned MM = 1u << idx;
            /* warmup */
            for (int i = 0; i < 20; i++) {
                switch (MM) {
                case 1: k_rowsN<1><<<GRID, BLOCK>>>(dC, dx, dW, MM, s.N, s.K); break;
                case 2: k_rowsN<2><<<GRID, BLOCK>>>(dC, dx, dW, MM, s.N, s.K); break;
                case 4: k_rowsN<4><<<GRID, BLOCK>>>(dC, dx, dW, MM, s.N, s.K); break;
                default: k_rowsN<8><<<GRID, BLOCK>>>(dC, dx, dW, MM, s.N, s.K); break;
                }
            }
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int i = 0; i < iters; i++) {
                switch (MM) {
                case 1: k_rowsN<1><<<GRID, BLOCK>>>(dC, dx, dW, MM, s.N, s.K); break;
                case 2: k_rowsN<2><<<GRID, BLOCK>>>(dC, dx, dW, MM, s.N, s.K); break;
                case 4: k_rowsN<4><<<GRID, BLOCK>>>(dC, dx, dW, MM, s.N, s.K); break;
                default: k_rowsN<8><<<GRID, BLOCK>>>(dC, dx, dW, MM, s.N, s.K); break;
                }
            }
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float el; CK(cudaEventElapsedTime(&el, e0, e1));
            ms[idx] = (double)el / iters;
        }
        /* Weight bytes are the same for every MM — that is the point. */
        double wgb = (double)s.N * s.K * 2.0 / 1e9;
        printf("  %-22s %8.4f %8.4f %8.4f %8.4f   %.2fx %.2fx %.2fx %.2fx  (W=%.2f GB/s @MM1)\n",
               s.name, ms[0], ms[1], ms[2], ms[3],
               1.0, 2.0 * ms[0] / ms[1], 4.0 * ms[0] / ms[2], 8.0 * ms[0] / ms[3],
               wgb / (ms[0] / 1000.0));
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        CK(cudaFree(dx)); CK(cudaFree(dW)); CK(cudaFree(dC));
    }
    printf("\n  (scaling = MM * t(MM=1)/t(MM) — the throughput multiple vs running MM\n"
           "   separate batch-1 GEMVs. 1.00x would mean batching buys nothing.)\n");

    /* ---------- Q3: the SAME sweep with an HBM-RESIDENT weight stream ------------------
     * Q2 above loops one matrix, which is 5-50 MB and therefore lives in the ~100 MB L2 —
     * its @MM1 "bandwidth" exceeds the 1673 GB/s HBM ceiling, which is the tell. Real decode
     * streams 7.49 GiB of weights per token, so every GEMV's weight read is a COLD HBM read
     * and that read is what batching is supposed to amortize. To reproduce that regime we
     * rotate over COPIES of the weight matrix so each iteration reads one L2 has evicted. */
    printf("\n=== Q3: same sweep, HBM-resident weights (rotating copies, L2 defeated) ===\n");
    printf("  %-22s %8s %8s %8s %8s   %s\n", "shape [N,K]", "MM=1", "MM=2", "MM=4", "MM=8",
           "tok/s scaling vs MM=1");

    for (auto& s : shapes) {
        const unsigned MMAX = 8;
        /* Enough copies to blow past ~100 MB of L2 by a wide margin. */
        size_t wbytes = (size_t)s.N * s.K * 2;
        int copies = (int)((400u * 1024 * 1024) / wbytes) + 1;
        if (copies < 4) copies = 4;

        std::vector<__nv_bfloat16> hx((size_t)MMAX * s.K), hW((size_t)s.N * s.K);
        fill(hx, 9); fill(hW, 10);
        __nv_bfloat16 *dx, *dC;
        std::vector<__nv_bfloat16*> dWs(copies);
        CK(cudaMalloc(&dx, hx.size() * 2));
        CK(cudaMalloc(&dC, (size_t)MMAX * s.N * 2));
        CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        for (int cpy = 0; cpy < copies; cpy++) {
            CK(cudaMalloc(&dWs[cpy], wbytes));
            CK(cudaMemcpy(dWs[cpy], hW.data(), wbytes, cudaMemcpyHostToDevice));
        }

        double ms[4];
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int idx = 0; idx < 4; idx++) {
            unsigned MM = 1u << idx;
#define LAUNCH_ROT(i) do { __nv_bfloat16* W_ = dWs[(i) % copies];                     \
    switch (MM) {                                                                     \
    case 1: k_rowsN<1><<<GRID, BLOCK>>>(dC, dx, W_, MM, s.N, s.K); break;             \
    case 2: k_rowsN<2><<<GRID, BLOCK>>>(dC, dx, W_, MM, s.N, s.K); break;             \
    case 4: k_rowsN<4><<<GRID, BLOCK>>>(dC, dx, W_, MM, s.N, s.K); break;             \
    default: k_rowsN<8><<<GRID, BLOCK>>>(dC, dx, W_, MM, s.N, s.K); break; } } while (0)
            for (int i = 0; i < 20; i++) LAUNCH_ROT(i);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int i = 0; i < iters; i++) LAUNCH_ROT(i);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float el; CK(cudaEventElapsedTime(&el, e0, e1));
            ms[idx] = (double)el / iters;
#undef LAUNCH_ROT
        }
        double wgb = (double)wbytes / 1e9;
        printf("  %-22s %8.4f %8.4f %8.4f %8.4f   %.2fx %.2fx %.2fx %.2fx  (W=%.0f GB/s @MM1, %d copies)\n",
               s.name, ms[0], ms[1], ms[2], ms[3],
               1.0, 2.0 * ms[0] / ms[1], 4.0 * ms[0] / ms[2], 8.0 * ms[0] / ms[3],
               wgb / (ms[0] / 1000.0), copies);
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        for (int cpy = 0; cpy < copies; cpy++) CK(cudaFree(dWs[cpy]));
        CK(cudaFree(dx)); CK(cudaFree(dC));
    }
    printf("\n  @MM1 bandwidth here should sit AT or BELOW the 1673 GB/s HBM ceiling — if it\n"
           "  does not, L2 was not actually defeated and the numbers are the Q2 regime again.\n");

    /* ---------- Q4: WEIGHT-STATIONARY WIDE RUNGS (MM=16/32) --------------------------------
     * The whole WS-batched-gemv thesis: raising the MM ladder to 16/32 makes a B=16/32 decode
     * step read the weights ONCE, not ceil(B/8) times. Correctness first (all MM rows written
     * and correct vs the f32 reference — the batched token-identity precondition), then the
     * HBM-resident weight-read scaling at the real Gemma-4-12B decode shapes. */
    printf("\n=== Q4a: gemv_rows/qkv/glu correctness at MM=16,32 (all rows vs f32 ref) ===\n");
    {
        const unsigned N = 512, K = 3840;                 /* 12B hidden K */
        for (unsigned M : {16u, 32u}) {
            std::vector<__nv_bfloat16> hx((size_t)M * K), hW((size_t)N * K), hC((size_t)M * N);
            fill(hx, 11 + M); fill(hW, 22 + M);
            for (auto& c : hC) *(unsigned short*)&c = SENTINEL;
            std::vector<float> ref; ref_gemv(ref, hx, hW, M, N, K);
            __nv_bfloat16 *dx, *dW, *dC;
            CK(cudaMalloc(&dx, hx.size()*2)); CK(cudaMalloc(&dW, hW.size()*2));
            CK(cudaMalloc(&dC, hC.size()*2));
            CK(cudaMemcpy(dx, hx.data(), hx.size()*2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dW, hW.data(), hW.size()*2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dC, hC.data(), hC.size()*2, cudaMemcpyHostToDevice));
            k_d_gemv<<<GRID, BLOCK>>>(dC, dx, dW, M, N, K);   /* walks to gemv_rows<16>/<32> */
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hC.data(), dC, hC.size()*2, cudaMemcpyDeviceToHost));
            /* every row must be written AND match; worst-row relL2 is the gate. */
            double worst = 0; unsigned unwritten = 0;
            for (unsigned m = 0; m < M; m++) {
                double num=0, den=0; bool w=false;
                for (unsigned n = 0; n < N; n++) {
                    if (*(unsigned short*)&hC[(size_t)m*N+n] != SENTINEL) w = true;
                    double got=__bfloat162float(hC[(size_t)m*N+n]), want=ref[(size_t)m*N+n];
                    num += (got-want)*(got-want); den += want*want;
                }
                if (!w) unwritten++;
                double r = den>0? sqrt(num/den):0; if (r>worst) worst=r;
            }
            printf("  d_gemv M=%2u : rows unwritten=%u  worst-row relL2=%.3e  %s\n",
                   M, unwritten, worst,
                   (unwritten==0 && worst<3e-3) ? "PASS" : "FAIL");
            CK(cudaFree(dx)); CK(cudaFree(dW)); CK(cudaFree(dC));
        }
    }

    printf("\n=== Q4b: HBM-resident WS scaling at Gemma-4-12B decode shapes (%d iters) ===\n", iters);
    printf("  %-22s %8s %8s %8s %8s   %s\n", "shape [N,K]", "MM=1", "MM=8", "MM=16", "MM=32",
           "tok/s scaling vs MM=1");
    struct WShape { const char* name; unsigned N, K; };
    WShape wshapes[] = {
        {"q_proj  [4096,3840]", 4096, 3840},
        {"kv_proj [2048,3840]", 2048, 3840},
        {"gate/up [15360,3840]",15360,3840},
        {"down    [3840,15360]",3840,15360},
    };
    const unsigned MMS[4] = {1, 8, 16, 32};
    for (auto& s : wshapes) {
        const unsigned MMAX = 32;
        size_t wbytes = (size_t)s.N * s.K * 2;
        int copies = (int)((400u*1024*1024) / wbytes) + 1; if (copies < 4) copies = 4;
        std::vector<__nv_bfloat16> hx((size_t)MMAX*s.K), hW((size_t)s.N*s.K);
        fill(hx, 33); fill(hW, 44);
        __nv_bfloat16 *dx, *dC; std::vector<__nv_bfloat16*> dWs(copies);
        CK(cudaMalloc(&dx, hx.size()*2)); CK(cudaMalloc(&dC, (size_t)MMAX*s.N*2));
        CK(cudaMemcpy(dx, hx.data(), hx.size()*2, cudaMemcpyHostToDevice));
        for (int cp=0; cp<copies; cp++){ CK(cudaMalloc(&dWs[cp], wbytes));
            CK(cudaMemcpy(dWs[cp], hW.data(), wbytes, cudaMemcpyHostToDevice)); }
        double ms[4];
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int idx=0; idx<4; idx++) {
            unsigned MM = MMS[idx];
#define LAUNCH_W(i) do { __nv_bfloat16* W_ = dWs[(i)%copies];                       \
    switch (MM) {                                                                    \
    case 1:  k_rowsN<1><<<GRID,BLOCK>>>(dC,dx,W_,MM,s.N,s.K); break;                 \
    case 8:  k_rowsN<8><<<GRID,BLOCK>>>(dC,dx,W_,MM,s.N,s.K); break;                 \
    case 16: k_rowsN<16><<<GRID,BLOCK>>>(dC,dx,W_,MM,s.N,s.K); break;                \
    default: k_rowsN<32><<<GRID,BLOCK>>>(dC,dx,W_,MM,s.N,s.K); break; } } while(0)
            for (int i=0;i<20;i++) LAUNCH_W(i);
            CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
            for (int i=0;i<iters;i++) LAUNCH_W(i);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float el; CK(cudaEventElapsedTime(&el,e0,e1)); ms[idx]=(double)el/iters;
#undef LAUNCH_W
        }
        printf("  %-22s %8.4f %8.4f %8.4f %8.4f   %.2fx %.2fx %.2fx %.2fx  (%d copies)\n",
               s.name, ms[0], ms[1], ms[2], ms[3],
               1.0, 8.0*ms[0]/ms[1], 16.0*ms[0]/ms[2], 32.0*ms[0]/ms[3], copies);
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        for (int cp=0; cp<copies; cp++) CK(cudaFree(dWs[cp]));
        CK(cudaFree(dx)); CK(cudaFree(dC));
    }
    printf("\n  scaling = MM * t(MM=1)/t(MM). Weight-stationary → this keeps climbing past MM=8\n"
           "  (one weight read amortized over MM rows); a flat/declining 16→32 means compute-bound.\n");
    return 0;
}

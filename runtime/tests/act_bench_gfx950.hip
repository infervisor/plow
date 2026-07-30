/* act_bench_gfx950.hip — wall-clock instrument for the ELEMENTWISE activation ops, at the shapes
 * Kimi-K3 and GLM-5.2 actually run them.                                    [KIMI-K3] [GLM-5.2]
 *
 * WHY A SEPARATE INSTRUMENT. `moe_prefill_a4w4_bench_gfx950.hip` next door measures the ROUTED
 * expert GEMM, where the GLU epilogue is a few dozen VALU at the tail of a K=3584 MFMA main loop
 * and is comprehensively hidden by it — that bench cannot tell you what an activation costs,
 * only that op85 does not care. The ops measured here are the opposite case: `d_situ_glu`
 * (PLOW_DOP_SITU_GLU, op 105) and `d_glu` are pure streaming passes with no MFMA to hide behind,
 * so their VALU is on the critical path and a transcendental substitution shows up directly.
 *
 * The header on `d_situ_glu` in op_k3.h claimed it "is memory-bound by an enormous margin". At
 * the arithmetic it had, that was wrong by ~4x and this bench is how that was established: three
 * bf16 accesses per element is 6 B, so at 8 TB/s the roofline is 1.3e12 elem/s, while ~109 VALU
 * per element at this part's f32 issue rate is ~0.36e12 elem/s. It was VALU-bound, not
 * memory-bound, and the measured GB/s below is the arbiter.
 *
 * SHAPES. K3 runs `situ` on every GLU of 92 of its 93 layers: the two shared experts
 * (2 * moe_intermediate = 6144 wide) on every MoE layer, and the dense FFN (18432) on the first.
 * GLM-5.2 runs `d_glu` with silu at its own dense width.
 *
 *   /opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/act_bench_gfx950.hip -o /tmp/act_bench
 *   sg render -c 'ROCR_VISIBLE_DEVICES=<n> /tmp/act_bench'
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#define PLOW_K3 1
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_k3.h"

#define HC(x) do{hipError_t e_=(x); if(e_){printf("hip err %s @%d\n",hipGetErrorString(e_),__LINE__);exit(1);} }while(0)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void b_situ(bf16* o, const bf16* g,
                                                                  const bf16* u, unsigned n,
                                                                  float beta, float lb) {
    d_situ_glu(o, g, u, n, beta, lb, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void b_glu(bf16* o, const bf16* g,
                                                                 const bf16* u, unsigned n,
                                                                 unsigned act) {
    d_glu(o, g, u, n, act, blockIdx.x, gridDim.x);
}

static double med(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const unsigned T = (argc > 1) ? (unsigned)atoi(argv[1]) : 1024u;
    const unsigned grid = (argc > 2) ? (unsigned)atoi(argv[2]) : 512u;
    const int iters = (argc > 3) ? atoi(argv[3]) : 51;

    struct Shape { unsigned w; const char* what; } sh[] = {
        {6144, "K3 shared experts  2*3072"},
        {18432, "K3 dense FFN         18432"},
        {8192, "GLM-5.2 dense         8192"},
    };

    bf16 *dg, *du, *dout;
    const size_t maxn = (size_t)T * 18432;
    HC(hipMalloc(&dg, maxn * 2));
    HC(hipMalloc(&du, maxn * 2));
    HC(hipMalloc(&dout, maxn * 2));
    /* 0x3E80 is bf16 0.25 — a mid-range activation, so no arm of any activation is skipped and
     * the exponential is exercised rather than saturated. */
    {
        std::vector<bf16> h(maxn, (bf16)0x3e80u);
        HC(hipMemcpy(dg, h.data(), maxn * 2, hipMemcpyHostToDevice));
        for (size_t i = 0; i < maxn; i++) h[i] = (bf16)(0x3e80u + (unsigned)(i & 0x7fu));
        HC(hipMemcpy(du, h.data(), maxn * 2, hipMemcpyHostToDevice));
    }

    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    printf("elementwise activations  T=%u grid=%u  (median of %d)\n", T, grid, iters);
    printf("%-28s %10s %12s %10s %12s %10s\n", "shape", "situ ms", "situ GB/s", "silu ms",
           "gelu ms", "elems");
    for (unsigned s = 0; s < sizeof(sh) / sizeof(sh[0]); s++) {
        const size_t n = (size_t)T * sh[s].w;
        const double bytes = (double)n * 6.0; /* 2 bf16 in + 1 bf16 out */
        std::vector<double> ts, tl, tg;
        for (int it = 0; it < iters; it++) {
            float ms;
            HC(hipEventRecord(e0));
            hipLaunchKernelGGL(b_situ, dim3(grid), dim3(PLOW_THREADS), 0, 0, dout, dg, du,
                               (unsigned)n, 4.0f, 25.0f);
            HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
            HC(hipEventElapsedTime(&ms, e0, e1)); if (it) ts.push_back(ms);
            HC(hipEventRecord(e0));
            hipLaunchKernelGGL(b_glu, dim3(grid), dim3(PLOW_THREADS), 0, 0, dout, dg, du,
                               (unsigned)n, 1u);
            HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
            HC(hipEventElapsedTime(&ms, e0, e1)); if (it) tl.push_back(ms);
            HC(hipEventRecord(e0));
            hipLaunchKernelGGL(b_glu, dim3(grid), dim3(PLOW_THREADS), 0, 0, dout, dg, du,
                               (unsigned)n, 0u);
            HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
            HC(hipEventElapsedTime(&ms, e0, e1)); if (it) tg.push_back(ms);
        }
        HC(hipGetLastError());
        const double a = med(ts), b = med(tl), c = med(tg);
        printf("%-28s %10.4f %9.1f    %10.4f %12.4f %10zu\n", sh[s].what, a,
               bytes / (a * 1e-3) / 1e9, b, c, n);
    }
    printf("  roofline at 8 TB/s: a 6 B/element streaming pass cannot beat %.4f ms per 1e9 elems\n",
           6.0e9 / 8.0e12 * 1e3);
    return 0;
}

/* moe_prefill_a4w4_bench_gfx950.hip — wall-clock + achieved-bandwidth instrument for the A4W4
 * grouped MoE prefill GEMM (`d_moe_group_pf_a4w4`) at KIMI-K3's MoE prefill geometry.
 *
 * WHY THIS EXISTS. The correctness test next door runs H=I=256 over 3 experts: two K-tiles per
 * GEMM, so the software pipeline it exercises is the PROLOGUE, not the steady state. Any claim
 * about the main loop's staging/MFMA overlap has to be made at a shape where the steady state
 * dominates, which for K3 is K=3584 (28 K-tiles) over 896 experts.
 *
 * SHAPE. The defaults are the emitted TP8 shard: latent/hidden H=3584, local
 * I_moe=384, 896 routed experts, top-16, T tokens.
 * Expert row counts are RAGGED (a skewed synthetic router histogram, not T*k/E flat) because a
 * flat histogram hides both the per-expert BM padding and the tile->expert search.
 *
 * WHAT IT MEASURES. Median-of-N wall time for op 85 (gate/up + fused bridge) and op 86 (down +
 * scatter) separately, plus the exact byte traffic each one is obliged to move, so the number
 * that comes out is a fraction of the HBM roofline rather than an unanchored millisecond count.
 * Weight bytes are counted PER TILE-VISIT, not per expert, because a 2-tile expert really does
 * re-read its weights; A bytes likewise carry the n-tile re-read amplification.
 *
 * Contents are memset patterns. The production-shape stage-2 check compares scatter+fixed-order
 * reduction with direct atomic accumulation before timing; the focused synthetic test next door
 * remains the stage-1 numerical oracle.
 *
 * WHAT IT SAID. MI355X (gfx950, SPX/NPS1), T=1024, grid 512, median of 20, in two steps:
 *
 *                       base      pipelined     +quantizer      VGPR      waves/SIMD
 *   op85 GLU+bridge   15.968 ms   13.903 ms      7.622 ms    150 -> 165    3 -> 3
 *   op86 DOWN+scatter  2.446 ms    2.285 ms      2.272 ms    116 -> 121    4 -> 4
 *   total             18.414 ms   16.188 ms      9.893 ms    -46.3%
 *
 * "pipelined" is the ISSUE/MFMA/COMMIT split in op_moe.h; "+quantizer" adds the branchless
 * `quant_fp4`, the integer `e8m0_for_amax`, `e8m0_inv_f32` and the packed 16-byte fp4 stores.
 * Scratch stayed 0 and MPF4_LDS_BYTES stayed 76288 throughout, and every step is bit-identical
 * (fp4_quant_identity_gfx950_test.hip proves the primitives over all 2^32 inputs).
 *
 * WHERE THE REST OF IT IS. A diagnostic build with the fp4 quantizer stubbed out to one
 * instruction per element runs op85 in 5.505 ms (46.6% of roofline), so the quantizer that
 * remains is ~2.1 ms and everything else is ~5.5 ms. op85 at 33.6% and op86 at 37.6% of an
 * 8 TB/s roofline are now close enough that the two are limited by the same thing, and it is
 * not the quantizer: an 8-wave workgroup at 165 VGPR gets ONE workgroup per CU (2 waves/SIMD)
 * even though the 76 KiB of LDS would allow two. Under 128 VGPR is the next cliff.
 *
 * WHAT THE ACTIVATION SWEEP ADDED (and why the `act` argv exists). The table above was taken at
 * `act = 1` (silu). K3 does not run silu — every K3 GLU is `situ`, which transforms BOTH branches
 * and cost 109 VALU per element. Re-measured at T=1024, grid 512, median of 31:
 *
 *                                  act=1 silu    act=2 situ (K3)
 *   op85, before the sweep           7.600 ms       7.836 ms
 *   op85, after                      7.600 ms       7.730 ms      -1.35%
 *
 * VGPR stayed 165/121, scratch stayed 0, and the silu column is UNCHANGED TO THE MILLISECOND
 * because nothing on that path was touched — which is the check that GLM-5.2's arm did not move.
 *
 * THE LESSON THIS BENCH LEARNED THE HARD WAY, recorded here because it is not obvious and it
 * cost four builds: op85's GLU epilogue is NOT VALU-bound, and edits that make it cheaper per
 * element can make the KERNEL slower. Hoisting situ's uniform `2/beta` reciprocal out of the
 * element loop — textbook, and it does speed up the standalone activation — cost 0.24 ms,
 * because the hoisted value is a VGPR live across all 28 K-tiles of a main loop that has no
 * register slack. Applying the same fast spelling to `moe_act`, an arm act=2 never executes,
 * cost 0.18 ms of pure re-scheduling. Anything touching this epilogue must be measured HERE, at
 * both `act` values, and not argued from instruction counts. See `k3_situ_gate` in amd_common.h
 * and `moe_act` in op_moe.h for the full ladders.
 *
 *   nix develop --command hipcc --offload-arch=gfx950 -O2 -Werror \
 *       -I runtime/amd -I runtime/common \
 *       runtime/tests/moe_prefill_a4w4_bench_gfx950.hip -o /tmp/moe_pf_a4w4_bench
 *   # Isolate the production-thread-count BM32/WNc8 stage-2 screen:
 *   nix develop --command hipcc --offload-arch=gfx950 -O2 -Werror \
 *       -DMPF4_BM=32 -DMPF4_WNc=8 -DPLOW_WG_WAVES=8 \
 *       -DPLOW_MOE_A4W4_STAGE2_BENCH=1 -DPLOW_MOE_PF_EPI_SIB=0 \
 *       -I runtime/amd -I runtime/common \
 *       runtime/tests/moe_prefill_a4w4_bench_gfx950.hip -o /tmp/moe_pf_a4w4_stage2
 *   argv[7] optionally overrides the TP-local expert width I.
 *   sg render -c 'ROCR_VISIBLE_DEVICES=<n> /tmp/moe_pf_a4w4_bench [T] [grid] [iters] [act] [beta] [lbeta]'
 */
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#ifndef MPF4_BM
#define MPF4_BM 64
#endif
#define PLOW_MOE_PF_ATOMIC 1
#define PLOW_MOE_PREFILL 1
#define PLOW_MOE_PF_A4W4 1
#include "amd_common.h"
#define st_act1(p, v) st_act<bf16>((PLOW_GLOB bf16*)(p), (v))
#include "op_moe.h"
#undef st_act1

#ifndef PLOW_MOE_A4W4_STAGE2_BENCH
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void b_glu(unsigned char* fu, const bf16* xn2, const unsigned long long* wtab,
           const unsigned long long* stab, const int* meta, const unsigned* row_token,
           const unsigned* row_partidx, unsigned char* fu_scale, unsigned I, unsigned H,
           unsigned n_exp, unsigned act, float beta, float lbeta) {
    __shared__ unsigned char lds[MPF4_LDS_BYTES];
    d_moe_group_pf_a4w4<true>((void*)fu, (const void*)xn2, nullptr, wtab, stab, meta, row_token,
                              row_partidx, nullptr, fu_scale, I, H, n_exp, act, blockIdx.x,
                              gridDim.x, beta, lbeta, (void*)lds);
}
#endif
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void b_down(float* part, const unsigned char* fu, const unsigned char* fu_scale,
            const unsigned long long* wtab, const unsigned long long* stab, const int* meta,
            const unsigned* row_partidx, const float* row_gate, unsigned H, unsigned I,
            unsigned n_exp, unsigned atom_ksh) {
    __shared__ unsigned char lds[MPF4_LDS_BYTES];
    d_moe_group_pf_a4w4<false>((void*)part, (const void*)fu, fu_scale, wtab, stab, meta, nullptr,
                               row_partidx, row_gate, nullptr, H, I, n_exp, 0, blockIdx.x,
                               gridDim.x, 0.0f, 0.0f, (void*)lds, atom_ksh);
}

extern "C" __global__ void b_reduce(float* out, const float* part, unsigned T,
                                    unsigned H, unsigned topk) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * H) return;
    const unsigned tok = (unsigned)(i / H), h = (unsigned)(i % H);
    float v = 0.0f;
    for (unsigned j = 0; j < topk; ++j) v += part[((size_t)tok * topk + j) * H + h];
    out[i] = v;
}

#define HC(x) do{hipError_t e_=(x); if(e_){printf("hip err %s @%d\n",hipGetErrorString(e_),__LINE__);exit(1);} }while(0)

static unsigned rngs = 12345u;
static unsigned rnd() { rngs ^= rngs << 13; rngs ^= rngs >> 17; rngs ^= rngs << 5; return rngs; }

int main(int argc, char** argv) {
    /* --- K3 MoE prefill geometry ------------------------------------------------------------ */
    const unsigned H = 3584;                 /* latent / model width, = GEMM1's K, GEMM2's N   */
    const unsigned I = (argc > 7) ? (unsigned)atoi(argv[7]) : 384u; /* TP-local expert width */
    const unsigned n_exp = 896;              /* routed experts                                 */
    const unsigned topk = 16;
    const unsigned T = (argc > 1) ? (unsigned)atoi(argv[1]) : 1024u;
    const unsigned grid = (argc > 2) ? (unsigned)atoi(argv[2]) : 512u;
    const int iters = (argc > 3) ? atoi(argv[3]) : 11;
    /* ACTIVATION, from argv, because the GLU epilogue is not free and the three activations do
     * not cost the same. `act=1` (silu) is the historical default this file's table was measured
     * with; K3 actually routes through `situ` (act=2, beta=4, linear_beta=25), which is a strictly
     * more expensive transform on both GLU branches. Benchmarking K3's MoE at silu understates
     * the epilogue, so the situ column is the one that speaks for the shipping model. */
    const unsigned act = (argc > 4) ? (unsigned)atoi(argv[4]) : 1u;
    const float beta = (argc > 5) ? (float)atof(argv[5]) : 4.0f;
    const float lbeta = (argc > 6) ? (float)atof(argv[6]) : 25.0f;
    const unsigned BM = MPF4_BM;

    /* Ragged router histogram: assign each of T*topk slots to an expert with a skew, so counts
     * land anywhere from 0 to several tiles rather than all on the same side of a BM boundary. */
    const unsigned nassign = T * topk;
    std::vector<unsigned> cnt(n_exp, 0);
    std::vector<unsigned> owner(nassign);
    for (unsigned i = 0; i < nassign; i++) {
        unsigned e = rnd() % n_exp;
        if ((rnd() & 3u) == 0u) e = (rnd() % (n_exp / 8u));   /* hot head: a skewed tail      */
        owner[i] = e; cnt[e]++;
    }
    std::vector<unsigned> tiles(n_exp + 1), rowoff(n_exp);
    unsigned tp = 0;
    for (unsigned e = 0; e < n_exp; e++) { tiles[e] = tp; rowoff[e] = tp * BM; tp += (cnt[e] + BM - 1) / BM; }
    tiles[n_exp] = tp;
    const unsigned pad = tp * BM;
    unsigned live = 0, maxc = 0; for (unsigned e = 0; e < n_exp; e++) { live += cnt[e]; maxc = std::max(maxc, cnt[e]); }

    std::vector<int> meta(3 * n_exp + 1);
    for (unsigned e = 0; e < n_exp; e++) {
        meta[e] = (int)rowoff[e]; meta[n_exp + e] = (int)cnt[e]; meta[2 * n_exp + e] = (int)tiles[e];
    }
    meta[3 * n_exp] = (int)tiles[n_exp];

    std::vector<unsigned> rtok(pad, PLOW_EXPERT_UNUSED), rpid(pad, PLOW_EXPERT_UNUSED);
    {
        std::vector<unsigned> fill(n_exp, 0);
        for (unsigned i = 0; i < nassign; i++) {
            const unsigned e = owner[i], r = fill[e]++;
            rtok[rowoff[e] + r] = i / topk;               /* token this row gathers            */
            rpid[rowoff[e] + r] = i;                      /* its slot in part[T*topk][H]       */
        }
    }

    /* --- device state ----------------------------------------------------------------------- */
    unsigned char *dfu, *dfs; bf16* dx; int* dmeta; unsigned *drt, *drp;
    float *drg, *dpart, *dacc_ref, *dacc_atom;
    unsigned long long *dwt, *dst;
    HC(hipMalloc(&dx, (size_t)T * H * 2));      HC(hipMemset(dx, 0x3c, (size_t)T * H * 2));
    HC(hipMalloc(&dfu, (size_t)pad * I / 2));   HC(hipMemset(dfu, 0x53, (size_t)pad * I / 2));
    HC(hipMalloc(&dfs, (size_t)pad * I / 32));  HC(hipMemset(dfs, 127, (size_t)pad * I / 32));
    HC(hipMalloc(&dmeta, meta.size() * 4));
    HC(hipMemcpy(dmeta, meta.data(), meta.size() * 4, hipMemcpyHostToDevice));
    HC(hipMalloc(&drt, (size_t)pad * 4)); HC(hipMemcpy(drt, rtok.data(), (size_t)pad * 4, hipMemcpyHostToDevice));
    HC(hipMalloc(&drp, (size_t)pad * 4)); HC(hipMemcpy(drp, rpid.data(), (size_t)pad * 4, hipMemcpyHostToDevice));
    std::vector<float> rgate(pad, 0.25f);
    HC(hipMalloc(&drg, (size_t)pad * 4)); HC(hipMemcpy(drg, rgate.data(), (size_t)pad * 4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dpart, (size_t)nassign * H * 4)); HC(hipMemset(dpart, 0, (size_t)nassign * H * 4));
    const size_t out_n = (size_t)T * H, out_bytes = out_n * sizeof(float);
    HC(hipMalloc(&dacc_ref, out_bytes));
    HC(hipMalloc(&dacc_atom, out_bytes));

    const size_t wgu = (size_t)I * H / 2, sgu = (size_t)I * H / 32;   /* gate, up  [I][H]      */
    const size_t wdn = (size_t)H * I / 2, sdn = (size_t)H * I / 32;   /* down      [H][I]      */
    const double wbytes = (double)(n_exp) * (double)(2 * (wgu + sgu) + wdn + sdn);
    printf("allocating %.2f GiB of expert weights (%u experts)\n", wbytes / (1024.0 * 1024 * 1024), n_exp);
    std::vector<unsigned long long> wt(n_exp * 3, 0), st(n_exp * 3, 0);
    for (unsigned e = 0; e < n_exp; e++) {
        void *a, *b, *c, *d, *f, *g;
        HC(hipMalloc(&a, wgu)); HC(hipMalloc(&b, wgu)); HC(hipMalloc(&f, wdn));
        HC(hipMalloc(&c, sgu)); HC(hipMalloc(&d, sgu)); HC(hipMalloc(&g, sdn));
        HC(hipMemset(a, 0x52, wgu)); HC(hipMemset(b, 0x25, wgu)); HC(hipMemset(f, 0x43, wdn));
        HC(hipMemset(c, 127, sgu)); HC(hipMemset(d, 127, sgu)); HC(hipMemset(g, 127, sdn));
        wt[e * 3] = (unsigned long long)a; wt[e * 3 + 1] = (unsigned long long)b; wt[e * 3 + 2] = (unsigned long long)f;
        st[e * 3] = (unsigned long long)c; st[e * 3 + 1] = (unsigned long long)d; st[e * 3 + 2] = (unsigned long long)g;
    }
    HC(hipMalloc(&dwt, wt.size() * 8)); HC(hipMemcpy(dwt, wt.data(), wt.size() * 8, hipMemcpyHostToDevice));
    HC(hipMalloc(&dst, st.size() * 8)); HC(hipMemcpy(dst, st.data(), st.size() * 8, hipMemcpyHostToDevice));

    /* --- obligatory traffic ------------------------------------------------------------------
     * GLU visits total_tiles * ceil(I/(BN/2)) tiles; DOWN visits total_tiles * ceil(H/BN). Each
     * visit stages the whole K extent of its BN weight rows and its BM A rows. */
    const unsigned tnc1 = (I + (MPF4_BN / 2) - 1) / (MPF4_BN / 2);
    const unsigned tnc2 = (H + MPF4_BN - 1) / MPF4_BN;
    const double nt1 = (double)tp * tnc1, nt2 = (double)tp * tnc2;
    const double b1 = nt1 * ((double)MPF4_BN * (H / 2 + H / 32)      /* B: gate|up fp4 + E8M0  */
                             + (double)MPF4_BM * H * 2.0)           /* A: gathered bf16        */
                     + (double)live * (I / 2 + I / 32);             /* fused-bridge output     */
    const double b2 = nt2 * ((double)MPF4_BN * (I / 2 + I / 32)      /* B: down fp4 + E8M0     */
                             + (double)MPF4_BM * (I / 2 + I / 32))  /* A: bridge fp4 + E8M0    */
                     + (double)live * H * 4.0;                      /* scattered f32 partials  */
    /* MACs: every visited tile does BM*BN*K of them, pad rows included. */
    const double f1 = nt1 * (double)MPF4_BM * MPF4_BN * H * 2.0;
    const double f2 = nt2 * (double)MPF4_BM * MPF4_BN * I * 2.0;

    const dim3 rblock(256), rgrid((unsigned)((out_n + 255) / 256));
    hipLaunchKernelGGL(b_down, dim3(grid), dim3(PLOW_THREADS), 0, 0, dpart, dfu, dfs, dwt, dst,
                       dmeta, drp, drg, H, I, n_exp, 0u);
    hipLaunchKernelGGL(b_reduce, rgrid, rblock, 0, 0, dacc_ref, dpart, T, H, topk);
    HC(hipMemset(dacc_atom, 0, out_bytes));
    hipLaunchKernelGGL(b_down, dim3(grid), dim3(PLOW_THREADS), 0, 0, dacc_atom, dfu, dfs, dwt,
                       dst, dmeta, drp, drg, H, I, n_exp, 5u);
    HC(hipDeviceSynchronize());
    std::vector<float> href(out_n), hatom(out_n);
    HC(hipMemcpy(href.data(), dacc_ref, out_bytes, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hatom.data(), dacc_atom, out_bytes, hipMemcpyDeviceToHost));
    size_t bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < out_n; ++i) {
        const double d = fabs((double)href[i] - hatom[i]);
        const double rel = d / fmax(1.0, fabs((double)href[i]));
        worst = fmax(worst, rel);
        if (!std::isfinite(hatom[i]) || rel > 2e-5) ++bad;
    }
    printf("  exact stage2 oracle: %s (%zu/%zu bad, worst rel %.3g)\n",
           bad ? "FAIL" : "PASS", bad, out_n, worst);
    if (bad) return 2;

    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    std::vector<double> t1v, t2v, tbasev, tatomv;
    for (int it = 0; it < iters; it++) {
#ifndef PLOW_MOE_A4W4_STAGE2_BENCH
        HC(hipEventRecord(e0));
        hipLaunchKernelGGL(b_glu, dim3(grid), dim3(PLOW_THREADS), 0, 0, dfu, dx, dwt, dst, dmeta,
                           drt, drp, dfs, I, H, n_exp, act, beta, lbeta);
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms = 0; HC(hipEventElapsedTime(&ms, e0, e1)); if (it) t1v.push_back(ms);
#else
        float ms = 0;
#endif
        HC(hipEventRecord(e0));
        hipLaunchKernelGGL(b_down, dim3(grid), dim3(PLOW_THREADS), 0, 0, dpart, dfu, dfs, dwt, dst,
                           dmeta, drp, drg, H, I, n_exp, 0u);
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        HC(hipEventElapsedTime(&ms, e0, e1)); if (it) t2v.push_back(ms);
        auto baseline = [&] {
            HC(hipEventRecord(e0));
            hipLaunchKernelGGL(b_down, dim3(grid), dim3(PLOW_THREADS), 0, 0, dpart, dfu, dfs,
                               dwt, dst, dmeta, drp, drg, H, I, n_exp, 0u);
            hipLaunchKernelGGL(b_reduce, rgrid, rblock, 0, 0, dacc_ref, dpart, T, H, topk);
            HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
            HC(hipEventElapsedTime(&ms, e0, e1)); if (it) tbasev.push_back(ms);
        };
        auto direct = [&] {
            HC(hipEventRecord(e0));
            HC(hipMemsetAsync(dacc_atom, 0, out_bytes, 0));
            hipLaunchKernelGGL(b_down, dim3(grid), dim3(PLOW_THREADS), 0, 0, dacc_atom, dfu, dfs,
                               dwt, dst, dmeta, drp, drg, H, I, n_exp, 5u);
            HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
            HC(hipEventElapsedTime(&ms, e0, e1)); if (it) tatomv.push_back(ms);
        };
        if (it & 1) { direct(); baseline(); } else { baseline(); direct(); }
    }
    HC(hipGetLastError());
    std::sort(t1v.begin(), t1v.end()); std::sort(t2v.begin(), t2v.end());
    std::sort(tbasev.begin(), tbasev.end()); std::sort(tatomv.begin(), tatomv.end());
#ifndef PLOW_MOE_A4W4_STAGE2_BENCH
    const double m1 = t1v[t1v.size() / 2];
#else
    const double m1 = 0.0;
#endif
    const double m2 = t2v[t2v.size() / 2];
    const double mb = tbasev[tbasev.size() / 2], ma = tatomv[tatomv.size() / 2];
    const double PEAK = 8.0e12; /* MI355X HBM3E, 8 TB/s */

    printf("K3 MoE prefill A4W4  T=%u topk=%u H=%u I=%u experts=%u grid=%u act=%u beta=%g lbeta=%g\n",
           T, topk, H, I, n_exp, grid, act, beta, lbeta);
    printf("  rows live=%u padded=%u  tiles=%u  max expert rows=%u\n", live, pad, tp, maxc);
#ifndef PLOW_MOE_A4W4_STAGE2_BENCH
    printf("  op85 GLU +bridge : %8.3f ms   %7.1f GB/s (%4.1f%% of 8TB/s)  %6.1f TFLOP/s\n",
           m1, b1 / (m1 * 1e-3) / 1e9, 100.0 * (b1 / (m1 * 1e-3)) / PEAK, f1 / (m1 * 1e-3) / 1e12);
#endif
    printf("  op86 DOWN+scatter: %8.3f ms   %7.1f GB/s (%4.1f%% of 8TB/s)  %6.1f TFLOP/s\n",
           m2, b2 / (m2 * 1e-3) / 1e9, 100.0 * (b2 / (m2 * 1e-3)) / PEAK, f2 / (m2 * 1e-3) / 1e12);
#ifndef PLOW_MOE_A4W4_STAGE2_BENCH
    printf("  total            : %8.3f ms   %7.1f GB/s (%4.1f%% of 8TB/s)\n",
           m1 + m2, (b1 + b2) / ((m1 + m2) * 1e-3) / 1e9, 100.0 * ((b1 + b2) / ((m1 + m2) * 1e-3)) / PEAK);
#endif
    printf("  stage2 scatter+reduce: %8.3f ms\n", mb);
    printf("  stage2 direct atomic : %8.3f ms (includes %.1f MiB zero) %+.1f%%\n",
           ma, out_bytes / 1048576.0, 100.0 * (ma / mb - 1.0));
    printf("  bytes: op85 %.3f GB  op86 %.3f GB\n", b1 / 1e9, b2 / 1e9);
    return 0;
}

// fa_gf_full_h100_ab.cu — GQA-fusion factor (GF_FULL) x nsplit re-sweep for the FULL-attention
// decode layer on H100 NVL (sm_90a).
//
// WHY A RE-SWEEP: runtime/nvidia/experiments/fa_gf_full_ab.cu swept GF_FULL on RTX PRO 6000
// (sm_120a, 188 SM, 128 MB L2, ~1535 GB/s) and concluded GF=4 + refilled nsplit (ns=48) wins.
// scripts/build_sm90a_cubin.sh now ships -DPLOW_NV_FA_GF_FULL=4 for H100 on the strength of that
// RTX data. H100 NVL is a different machine: 132 SM (not 188), 60 MB L2 (not 128 MB), HBM3.
//   * fewer SMs   -> the n_grp*nsplit fill point moves (the grid-aligned nsplit differs)
//   * smaller L2  -> less of the GF re-read is absorbed by cache
//   * higher BW   -> a re-read costs less wall time
// So the (GF_FULL, nsplit) optimum may move. This harness re-measures it on the real kernel.
//
// SHAPE: Gemma-4-26B FULL attention layer — n_head=16, n_kv_head=2 (GQA=8), head_dim=512,
// bf16 KV, K and V in distinct buffers, full window (window=0), linear cache (kv_stride=ctx).
//
// WHAT IS DRIVEN: the REAL production kernel bodies from runtime/nvidia/op_attention.cuh
// (d_flash_decode<512,GF> + d_flash_merge<512>), included verbatim. No mock. The launch mirrors
// production exactly: a PERSISTENT grid of n_cu blocks, slice=blockIdx.x, nblk=gridDim.x, so the
// grid-stride loop distributes the n_grp*nsplit work items over the SMs the megakernel actually
// has, and a ragged items-per-SM count costs what it costs in production. Both the decode and the
// merge are timed (merge cost scales with nsplit and is part of the nsplit tradeoff).
//
// L2 METHODOLOGY (H100 NVL L2 = 60 MB as reported by the driver): the full-layer KV working set is
//   KVH*ctx*D*2 B  x2 (K and V)  = 4096 * ctx bytes
//   8k -> 32 MB (FITS L2)   16k -> 64 MB   32k -> 128 MB   64k -> 256 MB   128k -> 512 MB
// A microbench that touches only KV would leave the 8k (and much of the 16k) working set resident
// in L2 across iterations, which does NOT reflect production: between two attention ops the
// megakernel streams GBs of weights through the same L2 and evicts the KV. So every cell is timed
// TWICE:
//   COLD  — N replicas of the KV buffer (~2 GiB cycled footprint, >30x L2) rotated per iteration,
//           so each timed launch reads addresses evicted long ago. This is the
//           production-representative number and the one the recommendation is based on.
//   HOT   — a single KV buffer, re-read every iteration, L2 keeps whatever it can. Reported to
//           expose how much L2 is flattering the short contexts.
// The machine's real streaming ceiling is measured first, for reference.
//
// CORRECTNESS: every (ctx, GF, nsplit) cell is validated against an f32 CPU oracle (full softmax
// over the whole context for the newest query token) before it is timed. Gate: relative L2 error
// <= 3e-3. A cell that fails is printed FAIL and cannot be read as a winner.
//
// PHASE 2 refines nsplit around the phase-1 winner with a fine scan plus repeat timings, to show
// that the optimum is the GRID-ALIGNED point (n_grp*nsplit == k*n_cu) and not a fluke.
//
// build (EXECUTABLE — plain -arch=sm_90a is REJECTED for sm_90a-only opcodes, and -arch=native
// resolves to sm_90; the -gencode form is required):
//   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 \
//       -gencode arch=compute_90a,code=sm_90a -O3 \
//       -I runtime/common -I runtime/nvidia -include cstdint \
//       -o <build>/fa_gf_full_h100_ab runtime/nvidia/experiments/fa_gf_full_h100_ab.cu
// run (GPU is shared — serialize):
//   flock /tmp/plow_gpu.lock <build>/fa_gf_full_h100_ab

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#define PLOW_NV_KVBOUNDS 0
#include "../op_attention.cuh" /* d_flash_decode<D,GF>, d_flash_merge<D>, FA_DEC_SMEM_FLOATS */

#define CK(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e = (x);                                                                   \
        if (e != cudaSuccess) {                                                                \
            printf("CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e));              \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

#ifndef PLOW_FA_BENCH_HD
#define PLOW_FA_BENCH_HD 512
#endif
#ifndef PLOW_FA_BENCH_NH
#define PLOW_FA_BENCH_NH 16
#endif
#ifndef PLOW_FA_BENCH_KVH
#define PLOW_FA_BENCH_KVH 2
#endif
static const int D = PLOW_FA_BENCH_HD;
static const int NH = PLOW_FA_BENCH_NH;
static const int KVH = PLOW_FA_BENCH_KVH;
#ifndef PLOW_FA_BENCH_SCALE
#define PLOW_FA_BENCH_SCALE (1.0f / sqrtf((float)D))
#endif
static const float SCALE = PLOW_FA_BENCH_SCALE;
static const double REL_GATE = 3e-3;

/* ---- launch wrappers: production-faithful persistent grid ------------------------------------
 * The megakernel runs a cooperative grid of n_cu blocks; every op body walks its work items with
 * slice=blockIdx.x, nblk=gridDim.x. Reproduced exactly so nsplit changes the ITEMS PER BLOCK the
 * same way it does in production (which is the whole point of the nsplit axis). */
template <int GF>
__global__ void decode_launch(float* Opart, float* mlpart, const __nv_bfloat16* Q,
                              const __nv_bfloat16* K, const __nv_bfloat16* V, const int* kv_len,
                              unsigned n_head, unsigned n_kv_head, unsigned kv_stride,
                              unsigned window, float scale, unsigned nsplit, unsigned kv_mask) {
    extern __shared__ float lds[];
    d_flash_decode<D, GF>(Opart, mlpart, Q, K, V, kv_len, /*n_batch*/ 1, n_head, n_kv_head,
                          kv_stride, window, scale, nsplit, kv_mask, /*slice*/ blockIdx.x,
                          /*nblk*/ gridDim.x, lds, 0);
}

__global__ void merge_launch(__nv_bfloat16* O, const float* Opart, const float* mlpart,
                             unsigned n_head, unsigned nsplit) {
    d_flash_merge<D>(O, Opart, mlpart, 1, n_head, nsplit, blockIdx.x, gridDim.x);
}

/* ---- streaming ceiling: pure grid-stride 128-bit read over a buffer far larger than L2 ------- */
__global__ void stream_read(const float4* __restrict__ p, size_t n, float* sink) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 v = p[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1e30f) *sink = acc.x + acc.y + acc.z + acc.w; /* never true; keeps loads live */
}

typedef void (*DecKern)(float*, float*, const __nv_bfloat16*, const __nv_bfloat16*,
                        const __nv_bfloat16*, const int*, unsigned, unsigned, unsigned, unsigned,
                        float, unsigned, unsigned);

static DecKern dec_kern(int gf) {
    return gf == 2 ? (DecKern)decode_launch<2>
                   : (gf == 4 ? (DecKern)decode_launch<4> : (DecKern)decode_launch<8>);
}
static size_t dec_smem(int gf) {
    return (size_t)(gf == 2   ? FA_DEC_SMEM_FLOATS(D, 2)
                    : gf == 4 ? FA_DEC_SMEM_FLOATS(D, 4)
                              : FA_DEC_SMEM_FLOATS(D, 8)) *
           4;
}

static __nv_bfloat16 rbf() { return __float2bfloat16((float)rand() / RAND_MAX * 2.f - 1.f); }

/* ---- per-context device state + f32 CPU oracle ----------------------------------------------- */
struct CtxBuf {
    int ctx = 0, nrep = 1;
    size_t kvelem = 0, kvbytes = 0, wset = 0;
    __nv_bfloat16 *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    int* dLen = nullptr;
    std::vector<float> ref;

    void init(int c) {
        ctx = c;
        kvelem = (size_t)KVH * ctx * D;
        kvbytes = kvelem * 2;
        wset = 2 * kvbytes;
        nrep = (int)((2ull << 30) / kvbytes);
        nrep = std::max(1, std::min(16, nrep));

        std::vector<__nv_bfloat16> hQ((size_t)NH * D), hK(kvelem), hV(kvelem);
        for (auto& x : hQ) x = rbf();
        for (auto& x : hK) x = rbf();
        for (auto& x : hV) x = rbf();
        int hlen = ctx;

        CK(cudaMalloc(&dQ, hQ.size() * 2));
        CK(cudaMalloc(&dK, kvbytes * nrep));
        CK(cudaMalloc(&dV, kvbytes * nrep));
        CK(cudaMemcpy(dQ, hQ.data(), hQ.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dK, hK.data(), kvbytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV, hV.data(), kvbytes, cudaMemcpyHostToDevice));
        for (int r = 1; r < nrep; r++) { /* identical content: caches are address-indexed */
            CK(cudaMemcpy(dK + r * kvelem, dK, kvbytes, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(dV + r * kvelem, dV, kvbytes, cudaMemcpyDeviceToDevice));
        }
        CK(cudaMalloc(&dLen, 4));
        CK(cudaMemcpy(dLen, &hlen, 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dO, (size_t)NH * D * 2));

        ref.assign((size_t)NH * D, 0.f);
        const int qpos = ctx - 1;
        std::vector<float> sc(ctx);
        for (int h = 0; h < NH; h++) {
            const int hkv = h / (NH / KVH);
            float m = -1e30f;
            for (int r = 0; r <= qpos; r++) {
                float d = 0;
                const __nv_bfloat16* q = &hQ[(size_t)h * D];
                const __nv_bfloat16* k = &hK[((size_t)hkv * ctx + r) * D];
                for (int i = 0; i < D; i++) d += __bfloat162float(q[i]) * __bfloat162float(k[i]);
                sc[r] = d * SCALE;
                if (sc[r] > m) m = sc[r];
            }
            float l = 0;
            for (int r = 0; r <= qpos; r++) { sc[r] = expf(sc[r] - m); l += sc[r]; }
            float* out = &ref[(size_t)h * D];
            for (int r = 0; r <= qpos; r++) {
                const __nv_bfloat16* v = &hV[((size_t)hkv * ctx + r) * D];
                const float w = sc[r];
                for (int i = 0; i < D; i++) out[i] += w * __bfloat162float(v[i]);
            }
            const float inv = 1.f / l;
            for (int i = 0; i < D; i++) out[i] *= inv;
        }
    }
    void free_all() {
        CK(cudaFree(dQ)); CK(cudaFree(dK)); CK(cudaFree(dV));
        CK(cudaFree(dO)); CK(cudaFree(dLen));
    }
};

struct Cell { double cold, hot, rel; bool ok; };

/* Validate + time one (GF, nsplit) cell. `trial` re-seeds nothing; call repeatedly for spread. */
static Cell bench(CtxBuf& c, int GF, int nsplit, int NCU, const char* dump = nullptr) {
    const size_t smem = dec_smem(GF);
    DecKern kern = dec_kern(GF);
    if (smem > 48 * 1024)
        CK(cudaFuncSetAttribute((void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    float *dOp, *dMl;
    CK(cudaMalloc(&dOp, (size_t)NH * nsplit * D * 4));
    CK(cudaMalloc(&dMl, (size_t)NH * nsplit * 2 * 4));

    auto run = [&](int rep) {
        kern<<<NCU, 256, smem>>>(dOp, dMl, c.dQ, c.dK + (size_t)rep * c.kvelem,
                                 c.dV + (size_t)rep * c.kvelem, c.dLen, NH, KVH,
                                 (unsigned)c.ctx, /*window*/ 0u, SCALE, (unsigned)nsplit,
                                 0xFFFFFFFFu);
        merge_launch<<<NCU, 256>>>(c.dO, dOp, dMl, NH, (unsigned)nsplit);
    };

    run(0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> hO((size_t)NH * D);
    CK(cudaMemcpy(hO.data(), c.dO, hO.size() * 2, cudaMemcpyDeviceToHost));
    if (dump) {
        std::vector<float> op((size_t)NH * nsplit * D), ml((size_t)NH * nsplit * 2);
        CK(cudaMemcpy(op.data(), dOp, op.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ml.data(), dMl, ml.size() * sizeof(float), cudaMemcpyDeviceToHost));
        FILE* f = fopen(dump, "wb");
        if (!f) { perror(dump); exit(1); }
        bool ok = fwrite(hO.data(), sizeof(__nv_bfloat16), hO.size(), f) == hO.size();
        ok &= fwrite(op.data(), sizeof(float), op.size(), f) == op.size();
        ok &= fwrite(ml.data(), sizeof(float), ml.size(), f) == ml.size();
        ok &= fclose(f) == 0;
        if (!ok) { fprintf(stderr, "attention dump failed: %s\n", dump); exit(1); }
    }
    double num = 0, den = 0;
    for (size_t i = 0; i < hO.size(); i++) {
        const double o = __bfloat162float(hO[i]), r = c.ref[i];
        num += (o - r) * (o - r);
        den += r * r;
    }
    Cell out;
    out.rel = sqrt(num / (den + 1e-30));
    out.ok = out.rel <= REL_GATE;

    const int iters = std::max(30, 3 * c.nrep);
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));

    for (int w = 0; w < 5; w++) run(w % c.nrep);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int it = 0; it < iters; it++) run(it % c.nrep);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    out.cold = ms / iters;

    for (int w = 0; w < 5; w++) run(0);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int it = 0; it < iters; it++) run(0);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    out.hot = ms / iters;

    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    CK(cudaFree(dOp));
    CK(cudaFree(dMl));
    return out;
}

int main(int argc, char** argv) {
    srand(1234);

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const int NCU = prop.multiProcessorCount;
    const size_t L2 = (size_t)prop.l2CacheSize;
    printf("device: %s  SM=%d  L2=%.1f MB  smem/SM=%zu KiB  cc=%d.%d\n", prop.name, NCU,
           L2 / 1048576.0, (size_t)prop.sharedMemPerMultiprocessor / 1024, prop.major, prop.minor);

    printf("attention: D=%d NH=%d KVH=%d scale=%.9g QREG=%d seed=1234\n",
           D, NH, KVH, SCALE, PLOW_NV_FA_QREG);

    if (argc != 1) {
        if (argc != 6) {
            fprintf(stderr, "usage: %s [context gf nsplit trials dump]\n", argv[0]);
            return 2;
        }
        const int ctx = atoi(argv[1]), gf = atoi(argv[2]);
        const int ns = atoi(argv[3]), trials = atoi(argv[4]);
        if (ctx <= 0 || ns <= 0 || trials <= 0 || (gf != 2 && gf != 4 && gf != 8) ||
            NH % KVH || (NH / KVH) % gf) {
            fprintf(stderr, "invalid attention shape or sweep arguments\n");
            return 2;
        }
        CtxBuf c;
        c.init(ctx);
        int fails = 0;
        for (int trial = 0; trial < trials; trial++) {
            Cell r = bench(c, gf, ns, NCU, trial == 0 ? argv[5] : nullptr);
            printf("D=%d NH=%d KVH=%d ctx=%d gf=%d ns=%d trial=%d cold_ms=%.6f "
                   "hot_ms=%.6f relL2=%.9g %s\n", D, NH, KVH, ctx, gf, ns, trial,
                   r.cold, r.hot, r.rel, r.ok ? "PASS" : "FAIL");
            fails += !r.ok;
        }
        c.free_all();
        return fails ? 1 : 0;
    }

    /* ---- register / spill report per GF instantiation (sm_90a) ---- */
    printf("\n=== registers / local (spill) bytes per GF instantiation, sm_90a ===\n");
    for (int gf : {2, 4, 8}) {
        cudaFuncAttributes a;
        CK(cudaFuncGetAttributes(&a, (const void*)dec_kern(gf)));
        printf("  GF=%d  regs=%3d  local(spill)=%zu B  dyn-smem=%.1f KiB  "
               "reg-limited blocks/SM @256thr=%d\n",
               gf, a.numRegs, (size_t)a.localSizeBytes, dec_smem(gf) / 1024.0,
               65536 / (a.numRegs * 256));
    }
    printf("  (the megakernel is __launch_bounds__(256,1): any regs<=255 is occupancy-neutral)\n");

    /* ---- streaming ceiling ---- */
    printf("\n=== streaming ceiling (grid-stride 128-bit reads, 2 GiB buffer >> L2) ===\n");
    {
        const size_t bytes = 2ull << 30;
        const size_t n4 = bytes / sizeof(float4);
        float4* dBuf;
        float* dSink;
        CK(cudaMalloc(&dBuf, bytes));
        CK(cudaMalloc(&dSink, 4));
        CK(cudaMemset(dBuf, 0, bytes));
        const int grid = NCU * 8;
        stream_read<<<grid, 256>>>(dBuf, n4, dSink);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        const int it = 20;
        for (int i = 0; i < it; i++) stream_read<<<grid, 256>>>(dBuf, n4, dSink);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        ms /= it;
        printf("  read ceiling: %.0f GB/s (%.3f ms per 2 GiB pass)\n", bytes / 1e9 / (ms / 1e3), ms);
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
        CK(cudaFree(dBuf));
        CK(cudaFree(dSink));
    }

    const int ctxs[] = {8192, 16384, 32768, 65536, 131072};
    const int NCTX = (int)(sizeof(ctxs) / sizeof(int));
    const int GFS[] = {2, 4, 8};
    /* nsplit candidates: the RTX ship values 24 and 48, the H100 grid-aligned points
     * (n_grp*ns == k*132 -> GF2 33/66, GF4 33/66, GF8 66/132) and their under/over-fill
     * neighbours, so the whole fill curve is visible rather than just its endpoints. */
    const int NSS[] = {16, 24, 33, 48, 66, 96, 132};
    const int NNS = (int)(sizeof(NSS) / sizeof(int));

    double worst_rel[3] = {0, 0, 0};
    int fails = 0;
    int best_gf[5], best_ns[5];
    double best_ms[5];

    printf("\n########## PHASE 1: full GF x nsplit x ctx sweep ##########\n");
    for (int ci = 0; ci < NCTX; ci++) {
        CtxBuf c;
        c.init(ctxs[ci]);
        printf("\n===== ctx %6d | KV working set %6.1f MB (%s L2 %.0f MB) | replicas=%d, "
               "cycled footprint %.0f MB =====\n",
               c.ctx, c.wset / 1048576.0, c.wset <= L2 ? "FITS" : "exceeds", L2 / 1048576.0,
               c.nrep, 2.0 * c.nrep * c.kvbytes / 1048576.0);
        printf("  %-3s %-4s %-7s %-9s | %9s %10s | %9s | %8s %7s | %s\n", "GF", "ns", "n_work",
               "items/SM", "COLD ms", "COLD GB/s", "HOT ms", "rows/spl", "reread", "relL2");

        best_ms[ci] = 1e30;
        best_gf[ci] = best_ns[ci] = 0;
        for (int gi = 0; gi < 3; gi++) {
            const int GF = GFS[gi], n_grp = NH / GF;
            for (int si = 0; si < NNS; si++) {
                const int ns = NSS[si];
                const int n_work = n_grp * ns;
                if (n_work > 8 * NCU) continue; /* absurd over-split, not a candidate */
                Cell r = bench(c, GF, ns, NCU);
                if (r.rel > worst_rel[gi]) worst_rel[gi] = r.rel;
                if (!r.ok) fails++;
                const double logical = (double)n_grp * c.ctx * D * 2.0 * 2.0; /* K+V per group */
                printf("  %-3d %-4d %-7d %-9.2f | %9.4f %10.0f | %9.4f | %8d %6.0fx | %.2e%s\n",
                       GF, ns, n_work, (double)n_work / NCU, r.cold,
                       logical / 1e9 / (r.cold / 1e3), r.hot, (c.ctx + ns - 1) / ns,
                       logical / (double)c.wset, r.rel, r.ok ? "" : "  <<< FAIL");
                if (r.ok && r.cold < best_ms[ci]) {
                    best_ms[ci] = r.cold; best_gf[ci] = GF; best_ns[ci] = ns;
                }
            }
        }
        printf("  -> ctx %6d BEST (cold): GF=%d ns=%d  %.4f ms\n", c.ctx, best_gf[ci], best_ns[ci],
               best_ms[ci]);
        c.free_all();
    }

    /* ---- PHASE 2: fine nsplit scan around the winner, 3 repeat timings for spread ------------
     * Tests the grid-alignment hypothesis directly: with n_grp=4 (GF=4) on 132 SMs the aligned
     * nsplit step is n_cu/gcd(n_grp,n_cu) = 33, so 33 and 66 give exactly 1 and 2 items per SM
     * while everything between them leaves a 2x straggler tail that FLASH_MERGE waits on. */
    printf("\n########## PHASE 2: fine nsplit scan, GF=4 (+GF2/GF8 anchors), 3 trials ##########\n");
    const int FINE[] = {20, 24, 28, 30, 32, 33, 34, 36, 40, 44, 48, 60, 66, 72};
    const int NFINE = (int)(sizeof(FINE) / sizeof(int));
    const int fctx[] = {32768, 131072};
    for (int ci = 0; ci < 2; ci++) {
        CtxBuf c;
        c.init(fctx[ci]);
        printf("\n===== ctx %6d fine scan =====\n", c.ctx);
        printf("  %-3s %-4s %-7s %-9s | %9s %9s %9s | %9s | %s\n", "GF", "ns", "n_work",
               "items/SM", "t1 ms", "t2 ms", "t3 ms", "min ms", "relL2");
        for (int gf : {4, 2, 8}) {
            const int n_grp = NH / gf;
            for (int si = 0; si < NFINE; si++) {
                const int ns = FINE[si];
                if (gf != 4 && !(n_grp * ns == NCU || n_grp * ns == 2 * NCU)) continue;
                double t[3];
                Cell r{};
                for (int k = 0; k < 3; k++) { r = bench(c, gf, ns, NCU); t[k] = r.cold; }
                if (!r.ok) fails++;
                printf("  %-3d %-4d %-7d %-9.2f | %9.4f %9.4f %9.4f | %9.4f | %.2e%s\n", gf, ns,
                       n_grp * ns, (double)(n_grp * ns) / NCU, t[0], t[1], t[2],
                       std::min(t[0], std::min(t[1], t[2])), r.rel, r.ok ? "" : "  <<< FAIL");
            }
        }
        c.free_all();
    }

    /* ---- PHASE 3: INTERLEAVED tie-break of the top candidates ---------------------------------
     * Phases 1 and 2 measure a candidate as a contiguous block, so a slow clock/thermal drift over
     * the run biases whichever candidate ran later. Here the candidates are round-robined within
     * one loop, so any drift hits all of them equally; `min over rounds` is then a fair ranking. */
    printf("\n########## PHASE 3: interleaved tie-break (round-robin, min of 6 rounds) ##########\n");
    {
        struct Cand { int gf, ns; };
        const Cand cands[] = {{4, 32}, {4, 33}, {4, 66}, {4, 24}, {4, 48}, {2, 16}, {8, 66}};
        const int NC = (int)(sizeof(cands) / sizeof(Cand));
        const int rounds = 6;
        for (int ci = 0; ci < 2; ci++) {
            CtxBuf c;
            c.init(fctx[ci]);
            std::vector<double> best(NC, 1e30), sum(NC, 0.0);
            for (int r = 0; r < rounds; r++)
                for (int k = 0; k < NC; k++) {
                    Cell x = bench(c, cands[k].gf, cands[k].ns, NCU);
                    if (!x.ok) fails++;
                    best[k] = std::min(best[k], x.cold);
                    sum[k] += x.cold;
                }
            printf("\n===== ctx %6d interleaved =====\n", c.ctx);
            printf("  %-3s %-4s %-7s %-9s | %9s %9s | %s\n", "GF", "ns", "n_work", "items/SM",
                   "min ms", "mean ms", "vs best");
            double gmin = 1e30;
            for (int k = 0; k < NC; k++) gmin = std::min(gmin, best[k]);
            for (int k = 0; k < NC; k++)
                printf("  %-3d %-4d %-7d %-9.2f | %9.4f %9.4f | %+6.1f%%\n", cands[k].gf,
                       cands[k].ns, (NH / cands[k].gf) * cands[k].ns,
                       (double)((NH / cands[k].gf) * cands[k].ns) / NCU, best[k],
                       sum[k] / rounds, 100.0 * (best[k] / gmin - 1.0));
            c.free_all();
        }
    }

    printf("\n=== correctness summary (relative L2 vs f32 CPU oracle, gate %.0e) ===\n", REL_GATE);
    for (int gi = 0; gi < 3; gi++)
        printf("  GF=%d worst relL2 over all (ctx,nsplit) = %.2e  %s\n", GFS[gi], worst_rel[gi],
               worst_rel[gi] <= REL_GATE ? "PASS" : "FAIL");
    printf("  cells failing the gate: %d\n", fails);

    printf("\n=== phase-1 optimum per context (cold) ===\n");
    for (int ci = 0; ci < NCTX; ci++)
        printf("  ctx %6d : GF=%d  nsplit=%d  %.4f ms\n", ctxs[ci], best_gf[ci], best_ns[ci],
               best_ms[ci]);
    return fails ? 1 : 0;
}


#ifdef PLOW_FA_LIBRARY
#include <cuda.h>
struct AttentionArgs { void* t[8]; int i[12]; float scale; };
// i: mode, HD, batch, query rows, context, Q heads, KV heads, window, splits,
//    query position, KV mask, KV stride. t: Q,K,V,O,partials,ml,lengths,TMA maps.
template<int HD, int GF>
__global__ void attention_decode(AttentionArgs a) {
    extern __shared__ float sm[];
    d_flash_decode<HD, GF>((float*)a.t[4], (float*)a.t[5],
        (const __nv_bfloat16*)a.t[0], (const __nv_bfloat16*)a.t[1],
        (const __nv_bfloat16*)a.t[2], (const int*)a.t[6], a.i[2], a.i[5], a.i[6],
        a.i[11], a.i[7], a.scale, a.i[8], (unsigned)a.i[10], blockIdx.x, gridDim.x, sm);
}
template<int HD>
__global__ void attention_merge(AttentionArgs a) {
    d_flash_merge<HD>((__nv_bfloat16*)a.t[3], (const float*)a.t[4],
        (const float*)a.t[5], a.i[2], a.i[5], a.i[8], blockIdx.x, gridDim.x);
}
template<int HD>
__global__ void attention_prefill(AttentionArgs a) {
    extern __shared__ float sm[];
    d_flash_prefill<HD,64,32>((float*)a.t[4], (float*)a.t[5],
        (const __nv_bfloat16*)a.t[0], (const __nv_bfloat16*)a.t[1],
        (const __nv_bfloat16*)a.t[2], (__nv_bfloat16*)a.t[3], a.i[3], a.i[4],
        a.i[5], a.i[6], a.i[9], a.i[7], 1, a.i[11], (unsigned)a.i[10], a.scale,
        blockIdx.x, gridDim.x, sm, nullptr, a.t[7]);
}
template<int HD, int GF>
static int attention_run(AttentionArgs a, cudaStream_t stream) {
    if (a.i[0] == 0) {
        attention_decode<HD,GF><<<132,256,FA_DEC_SMEM_FLOATS(HD,GF)*4,stream>>>(a);
        cudaError_t rc = cudaGetLastError();
        if (rc != cudaSuccess) return (int)rc;
        attention_merge<HD><<<132,256,0,stream>>>(a);
    } else {
        constexpr unsigned bytes = FA_PRE_SMEM_FLOATS(HD,64,32)*4;
        static const cudaError_t configured = cudaFuncSetAttribute(attention_prefill<HD>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (configured != cudaSuccess) return (int)configured;
        attention_prefill<HD><<<132,256,bytes,stream>>>(a);
    }
    return (int)cudaGetLastError();
}
extern "C" int plow_attention(void** tensors, const int* integers, float scale, void* stream) {
    AttentionArgs a = {};
    for (int i=0; i<8; ++i) a.t[i]=tensors[i];
    for (int i=0; i<12; ++i) a.i[i]=integers[i];
    a.scale=scale;
    if ((a.i[0]!=0 && a.i[0]!=1) || (a.i[1]!=256 && a.i[1]!=512) ||
        a.i[2]<1 || a.i[3]<1 || a.i[4]<1 || a.i[5]<1 || a.i[6]<1 ||
        a.i[5]%a.i[6] || (a.i[5]/a.i[6])%(a.i[1]==512 ? 4:2) || a.i[8]<1 ||
        a.i[11]<a.i[4] || (a.i[0]==1 && (a.i[2]!=1 || a.i[8]!=1 ||
        a.i[9]<0 || a.i[9]+a.i[3]!=a.i[4]))) return (int)cudaErrorInvalidValue;
    return a.i[1]==512 ? attention_run<512,4>(a,(cudaStream_t)stream)
                       : attention_run<256,2>(a,(cudaStream_t)stream);
}
extern "C" int plow_attention_maps(void* output, void* k, void* v, int hd, int rows, int heads) {
    if ((hd!=256 && hd!=512) || rows<1 || heads<1) return -1;
    alignas(64) CUtensorMap maps[2];
    const cuuint64_t dims[3]={(cuuint64_t)hd,(cuuint64_t)rows,(cuuint64_t)heads};
    const cuuint64_t strides[2]={(cuuint64_t)hd*2,(cuuint64_t)hd*rows*2};
    const cuuint32_t box[3]={64,32,1}, elem[3]={1,1,1};
    for (int j=0;j<2;++j) {
        CUresult rc=cuTensorMapEncodeTiled(&maps[j],CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,
            j ? v:k,dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (rc!=CUDA_SUCCESS) return -(int)rc;
    }
    return (int)cudaMemcpy(output,maps,sizeof(maps),cudaMemcpyHostToDevice);
}
#endif

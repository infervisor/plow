/* mla_bench_sm120.cu — ISOLATED PERF HARNESS for the sm_120 MLA decode + merge-fold arms (P2).
 *
 * Extends the P1 direct-launch pattern (mla_correct_sm120.cu) into a wall-clock benchmark:
 * allocates a realistic KV-latent cache in HBM (Ckv[B][ctx][512] bf16 + Krope[B][ctx][64] bf16 =
 * 1152 B/pos) plus Qabs/Qrope for NH heads, times d_flash_mla_decode_sm120<512,64,GF,false>
 * (+ d_mla_merge_fold_sm120<512,256>) with CUDA events over N iters (warmup + median/p95), and
 * emits a table with derived effective/issued GB/s vs the §3 roofline targets.
 *
 * The grid MATCHES the interp: 188 resident blocks (n_cu), 256 threads, arena = the op's smem.
 * The decode body is a grid-stride loop over n_work = B*(NH/GF)*nsplit items, so 188 blocks
 * reproduce the megakernel's 1-block/SM schedule regardless of the standalone kernel's occupancy.
 *
 * nsplit mirrors gemma4.rs::glm_nsplit(ctx, nh_l) (GLM_MLA_GF=4 internal, hardcoded 256 chip-fill).
 * Values are timing-irrelevant so buffers are device-memset (no huge H2D).
 *
 * Build (no GPU needed to build):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
 *     -I runtime/common -I runtime/nvidia runtime/tests/mla_bench_sm120.cu -o mla_bench
 * Run (needs GPU): mla_bench
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "op_mla.cuh"

typedef __nv_bfloat16 bf16;
#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA ERR %s: %s\n",#x,cudaGetErrorString(e_));exit(2);} } while(0)

#define N_CU 188  /* RTX PRO 6000 Blackwell SM count; the interp launches bps(=1)*SMs blocks. */

/* ---- launch wrappers: identical operand order to the interp FLASH_MLA_DECODE / MERGE_FOLD arms ---- */
template <int GF>
__global__ void k_mla_decode(float* Op, float* Ml, const bf16* Qa, const bf16* Qr, const bf16* Ckv,
                             const bf16* Kr, const int* kvlen, unsigned nb, unsigned nh,
                             unsigned kvs, float scale, unsigned nsplit) {
    extern __shared__ float arena[];
    d_flash_mla_decode_sm120<512, 64, GF, false>(Op, Ml, Qa, Qr, Ckv, Kr, kvlen, nb, nh, kvs,
                                                 /*window*/ 0u, scale, nsplit,
                                                 /*kv_mask*/ 0xFFFFFFFFu, blockIdx.x, gridDim.x,
                                                 arena, nullptr, 0u);
}
__global__ void k_merge_fold(bf16* O, const float* Op, const float* Ml, const bf16* Wuv,
                             unsigned nb, unsigned nh, unsigned V, unsigned nsplit) {
    extern __shared__ float smem[];
    d_mla_merge_fold_sm120<512, 256>(O, Op, Ml, Wuv, nb, nh, V, nsplit, blockIdx.x, gridDim.x, smem);
}

/* gemma4.rs::glm_nsplit (heads = per-rank nh_l; GLM_MLA_GF=4 internal; chip-fill hardcoded 256). */
static unsigned glm_nsplit(unsigned ctx, unsigned heads) {
    const unsigned FA_BKV = 32, NS_PER = 512, NS_FLOOR = 16, GF = 4;
    unsigned n_grp = heads / GF; if (n_grp < 1) n_grp = 1;
    unsigned fill = (256 + n_grp - 1) / n_grp; if (fill < 1) fill = 1;
    unsigned kv_tiles = (ctx + FA_BKV - 1) / FA_BKV; if (kv_tiles < 1) kv_tiles = 1;
    unsigned ns = ctx / NS_PER; if (ns < NS_FLOOR) ns = NS_FLOOR;
    if (ns > fill) ns = fill; if (ns > kv_tiles) ns = kv_tiles; if (ns < 1) ns = 1;
    return ns;
}

static void setsmem(const void* fn, size_t smem) {
    CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
}

struct Stat { double med, p95, mn; };
static Stat stats(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    Stat s;
    s.med = v[v.size() / 2];
    s.p95 = v[(size_t)(v.size() * 0.95)];
    s.mn = v.front();
    return s;
}

/* Run one (ctx,GF,NH,batch) point. Times decode-only and merge-only separately over `iters`.
 * grid=N_CU (188) mirrors the interp's register-bound 1-block/SM megakernel schedule; passing
 * full_occ launches at the STANDALONE kernel's max resident grid (bps*188) to measure the
 * occupancy ceiling the 209-reg megakernel forfeits. */
template <int GF>
static void bench_point(unsigned ctx, unsigned NH, unsigned V, unsigned B, int warmup, int iters,
                        bool full_occ) {
    if (NH % GF != 0) return;
    const unsigned nsplit = glm_nsplit(ctx, NH);
    const float scale = 0.08838835f;

    const size_t nckv = (size_t)B * ctx * 512, nkr = (size_t)B * ctx * 64;
    const size_t nqa = (size_t)B * NH * 512, nqr = (size_t)B * NH * 64;
    const size_t nop = (size_t)B * NH * nsplit * 512, nml = (size_t)B * NH * nsplit * 2;
    const size_t nwuv = (size_t)NH * 512 * V, no = (size_t)B * NH * V;

    bf16 *dCkv, *dKr, *dQa, *dQr, *dWuv, *dO;
    float *dOp, *dMl;
    int* dLen;
    CK(cudaMalloc(&dCkv, nckv * 2)); CK(cudaMemset(dCkv, 0x3c, nckv * 2)); /* bf16 ~0.0115 */
    CK(cudaMalloc(&dKr, nkr * 2)); CK(cudaMemset(dKr, 0x3c, nkr * 2));
    CK(cudaMalloc(&dQa, nqa * 2)); CK(cudaMemset(dQa, 0x3c, nqa * 2));
    CK(cudaMalloc(&dQr, nqr * 2)); CK(cudaMemset(dQr, 0x3c, nqr * 2));
    CK(cudaMalloc(&dWuv, nwuv * 2)); CK(cudaMemset(dWuv, 0x3c, nwuv * 2));
    CK(cudaMalloc(&dO, no * 2));
    CK(cudaMalloc(&dOp, nop * 4)); CK(cudaMalloc(&dMl, nml * 4));
    std::vector<int> hlen(B, (int)ctx);
    CK(cudaMalloc(&dLen, (size_t)B * 4)); CK(cudaMemcpy(dLen, hlen.data(), (size_t)B * 4, cudaMemcpyHostToDevice));

    const size_t dsmem = (size_t)MLA_DEC_SMEM_FLOATS(512, 64, GF) * sizeof(float);
    const size_t msmem = (size_t)512 * sizeof(float);
    setsmem((const void*)k_mla_decode<GF>, dsmem);
    setsmem((const void*)k_merge_fold, msmem);

    unsigned grid = N_CU;
    if (full_occ) {
        int bps = 1;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, (const void*)k_mla_decode<GF>, 256, dsmem));
        grid = (unsigned)bps * N_CU;
    }

    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> td, tm;

    for (int it = -warmup; it < iters; it++) {
        CK(cudaEventRecord(a));
        k_mla_decode<GF><<<grid, 256, dsmem>>>(dOp, dMl, dQa, dQr, dCkv, dKr, dLen, B, NH, ctx, scale, nsplit);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        if (it >= 0) td.push_back(ms);

        CK(cudaEventRecord(a));
        k_merge_fold<<<N_CU, 256, msmem>>>(dO, dOp, dMl, dWuv, B, NH, V, nsplit);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        if (it >= 0) tm.push_back(ms);
    }
    CK(cudaGetLastError());

    Stat sd = stats(td), sm = stats(tm);
    const double chain = sd.med + sm.med;
    const double distinct = 1152.0 * ctx * B;                 /* bytes streamed once (design §3) */
    const double issued = (double)(NH / GF) * 1152.0 * ctx * B; /* GF re-reads */
    const double eff = distinct / (sd.med * 1e-3) / 1e9;        /* decode-only GB/s */
    const double iss = issued / (sd.med * 1e-3) / 1e9;
    const double mstok = chain / B;                            /* per-token, single layer */
    const double est61 = mstok * 61.0;                         /* Kimi L=61 flash portion */
    const double tgt = (ctx == 8192) ? 0.48 : (ctx == 32768) ? 2.0 : 8.0;
    const bool meets_bw = eff >= 1150.0;
    const bool meets_ms = (NH == 128) && (est61 <= tgt * 1.05);

    printf("%7u  %2d  %3u  %3u  %5u  g%4u | dec %8.3f/%8.3f  mrg %7.3f  chain %8.3f | "
           "%7.1f (%4.1f%%) %7.1f (%4.1f%%) | ms/tok %7.4f  61L %7.3f  %s%s\n",
           ctx, GF, NH, B, nsplit, grid, sd.med, sd.p95, sm.med, chain, eff, eff / 1535.0 * 100.0, iss,
           iss / 1535.0 * 100.0, mstok, est61, meets_bw ? "BW-OK" : "bw-lo",
           (NH == 128) ? (meets_ms ? " ms-OK" : " ms-lo") : "");

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(dCkv); cudaFree(dKr); cudaFree(dQa); cudaFree(dQr); cudaFree(dWuv);
    cudaFree(dO); cudaFree(dOp); cudaFree(dMl); cudaFree(dLen);
}

int main(int argc, char** argv) {
    int warmup = 5, iters = 20;
    if (argc > 1) iters = atoi(argv[1]);
    if (argc > 2) warmup = atoi(argv[2]);

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("dev0: %s  SMs=%d  (launching N_CU=%d blocks x 256 thr)\n", prop.name,
           prop.multiProcessorCount, N_CU);
    printf("targets §3: BW eff >=1150 GB/s (75%% of 1535); Kimi 61L ms/tok 0.48/2.0/8.0 @8k/32k/128k\n\n");
    printf("    ctx  GF   NH    B nsplit  grid | dec med/p95 (ms)          mrg     chain    | "
           "eff GB/s          issued GB/s      | per-tok       61L(ms) verdict\n");

    const unsigned ctxs[] = {8192, 32768, 131072};
    const unsigned batches[] = {1, 8, 32};
    /* NH=64 -> GLM (V=256); NH=128 -> Kimi (V=128). */
    printf("==== PASS 1: grid=188 (interp's register-bound 1-block/SM schedule) ====\n");
    for (unsigned ci = 0; ci < 3; ci++) {
        for (unsigned bi = 0; bi < 3; bi++) {
            unsigned ctx = ctxs[ci], B = batches[bi];
            bench_point<2>(ctx, 64, 256, B, warmup, iters, false);
            bench_point<4>(ctx, 64, 256, B, warmup, iters, false);
            bench_point<8>(ctx, 64, 256, B, warmup, iters, false);
            bench_point<2>(ctx, 128, 128, B, warmup, iters, false);
            bench_point<4>(ctx, 128, 128, B, warmup, iters, false);
            bench_point<8>(ctx, 128, 128, B, warmup, iters, false);
        }
        printf("\n");
    }
    printf("==== PASS 2: grid=bps*188 (standalone occupancy ceiling; batch 1 & 8) ====\n");
    for (unsigned ci = 0; ci < 3; ci++) {
        for (unsigned bi = 0; bi < 2; bi++) {
            unsigned ctx = ctxs[ci], B = batches[bi];
            bench_point<2>(ctx, 64, 256, B, warmup, iters, true);
            bench_point<4>(ctx, 64, 256, B, warmup, iters, true);
            bench_point<2>(ctx, 128, 128, B, warmup, iters, true);
            bench_point<4>(ctx, 128, 128, B, warmup, iters, true);
        }
        printf("\n");
    }
    return 0;
}

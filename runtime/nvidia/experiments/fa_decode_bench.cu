/* Flash DECODE standalone bench at E4B / speech geometries: the shipped d_flash_decode (+ merge,
 * or the merge fold) against an FP64 oracle, timed like consecutive layers (rotating distinct KV
 * buffers, total > L2, back-to-back launches, persistent 132 x 256 grid like the interpreter).
 *
 *   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -I runtime/common -I runtime/nvidia \
 *        -DFDB_HD=256 -DFDB_GF=4 runtime/nvidia/experiments/fa_decode_bench.cu -o fdb256
 *   ./fdb256 B CTX WINDOW H KVH NSPLIT MODE     MODE 0 = flash + FLASH_MERGE, 1 = merge fold
 *
 * Prints us/layer, GB/s over the distinct KV bytes, and % of the 3.35 TB/s roofline.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define PLOW_NV_HOPPER 1
#ifndef PLOW_NV_FA_RG_WIDE
#define PLOW_NV_FA_RG_WIDE 1
#endif
#include "op_attention.cuh"

#ifndef FDB_HD
#define FDB_HD 256
#endif
#ifndef FDB_GF
#define FDB_GF 4
#endif
using bf16 = __nv_bfloat16;
constexpr int D = FDB_HD, GF = FDB_GF;

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); std::exit(2); } } while (0)

struct Args {
    float *opart, *ml;
    const bf16 *q, *k, *v;
    const int* kvlen;
    bf16* out;
    unsigned* ctr;
    unsigned B, H, KVH, stride, window, nsplit, mask;
    float scale;
    int fold;
};

__global__ void __launch_bounds__(256, 1) k_flash(Args a) {
    extern __shared__ float arena[];
    d_flash_decode<D, GF>(a.opart, a.ml, a.q, a.k, a.v, a.kvlen, a.B, a.H, a.KVH, a.stride, a.window,
                          a.scale, a.nsplit, a.mask, blockIdx.x, gridDim.x, arena, 0, nullptr,
                          nullptr, nullptr, a.fold ? a.out : nullptr, a.fold ? a.ctr : nullptr);
}
__global__ void __launch_bounds__(256, 1) k_merge(Args a) {
    d_flash_merge<D>(a.out, a.opart, a.ml, a.B, a.H, a.nsplit, blockIdx.x, gridDim.x);
}

static std::vector<bf16> rnd(size_t n, uint32_t seed, float amp) {
    std::vector<bf16> r(n);
    for (auto& x : r) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        x = __float2bfloat16(amp * float(int32_t(seed)) / 2147483648.0f);
    }
    return r;
}

int main(int argc, char** argv) {
    if (argc < 8) { std::fprintf(stderr, "usage: B CTX WINDOW H KVH NSPLIT MODE\n"); return 2; }
    const unsigned B = atoi(argv[1]), ctx = atoi(argv[2]), window = atoi(argv[3]);
    const unsigned H = atoi(argv[4]), KVH = atoi(argv[5]), ns = atoi(argv[6]);
    const int mode = atoi(argv[7]);
    const unsigned stride = window ? 4096u : ctx, mask = window ? 4095u : 0xffffffffu;
    const unsigned span = window ? std::min(ctx, window) : ctx;
    const size_t kv_el = (size_t)B * KVH * stride * D;
    const double kv_bytes = (double)B * KVH * span * D * 2 * 2;
    const int nrot = (int)std::max(2.0, std::min(64.0, std::ceil(160e6 / kv_bytes)));
    const float scale = 1.0f / std::sqrt((float)D);

    /* buffer 0 holds random data (checked); the rest are only timed */
    std::vector<bf16*> dk(nrot), dv(nrot);
    auto hk = rnd(kv_el, 22, 2.0f), hv = rnd(kv_el, 33, 1.0f), hq = rnd((size_t)B * H * D, 11, 2.0f);
    for (int i = 0; i < nrot; i++) {
        CK(cudaMalloc(&dk[i], kv_el * 2)); CK(cudaMalloc(&dv[i], kv_el * 2));
        if (i == 0) {
            CK(cudaMemcpy(dk[i], hk.data(), kv_el * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dv[i], hv.data(), kv_el * 2, cudaMemcpyHostToDevice));
        } else {
            CK(cudaMemset(dk[i], 0x3c, kv_el * 2)); CK(cudaMemset(dv[i], 0x3c, kv_el * 2));
        }
    }
    bf16 *dq, *dout; float *dop, *dml; int* dlen; unsigned* dctr;
    CK(cudaMalloc(&dq, hq.size() * 2)); CK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dout, (size_t)B * H * D * 2));
    CK(cudaMalloc(&dop, (size_t)B * H * ns * D * 4)); CK(cudaMalloc(&dml, (size_t)B * H * ns * 8));
    CK(cudaMalloc(&dctr, (size_t)B * H * 4)); CK(cudaMemset(dctr, 0, (size_t)B * H * 4));
    std::vector<int> hlen(B);
    for (unsigned b = 0; b < B; b++) hlen[b] = (int)ctx - (int)(b % 7); /* ragged a little */
    CK(cudaMalloc(&dlen, B * 4)); CK(cudaMemcpy(dlen, hlen.data(), B * 4, cudaMemcpyHostToDevice));

    const size_t smem = std::max<size_t>(FA_DEC_SMEM_FLOATS(D, GF) * 4, 48 * 1024);
    CK(cudaFuncSetAttribute(k_flash, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    Args a{dop, dml, dq, dk[0], dv[0], dlen, dout, dctr, B, H, KVH, stride, window, ns, mask, scale, mode};
    auto run = [&](int i) {
        a.k = dk[i]; a.v = dv[i];
        k_flash<<<132, 256, smem>>>(a);
        if (!mode) k_merge<<<132, 256>>>(a);
    };
    run(0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<bf16> ho((size_t)B * H * D);
    CK(cudaMemcpy(ho.data(), dout, ho.size() * 2, cudaMemcpyDeviceToHost));
    double worst = 0;
    for (unsigned b = 0; b < B; b += std::max(1u, B / 5))
        for (unsigned h = 0; h < H; h += 3) {
            const unsigned len = hlen[b], first = window && len > window ? len - window : 0;
            const unsigned hkv = h / (H / KVH);
            const size_t base = ((size_t)b * KVH + hkv) * stride;
            std::vector<double> sc(len - first); double mx = -1e300;
            for (unsigned p = first; p < len; p++) {
                double s = 0; const size_t kr = (base + (p & mask)) * D;
                for (int d = 0; d < D; d++) s += (double)__bfloat162float(hq[((size_t)b * H + h) * D + d]) * __bfloat162float(hk[kr + d]);
                sc[p - first] = s * scale; mx = std::max(mx, sc[p - first]);
            }
            double sum = 0; for (auto& x : sc) { x = std::exp(x - mx); sum += x; }
            double e2 = 0, r2 = 0;
            for (int d = 0; d < D; d++) {
                double o = 0;
                for (unsigned p = first; p < len; p++) o += sc[p - first] * __bfloat162float(hv[(base + (p & mask)) * D + d]);
                o /= sum;
                const double g = __bfloat162float(ho[((size_t)b * H + h) * D + d]);
                e2 += (g - o) * (g - o); r2 += o * o;
            }
            worst = std::max(worst, std::sqrt(e2 / std::max(r2, 1e-30)));
        }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    /* One CUDA graph of 4 x nrot layers: no host launch cost, like the reference timings. */
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaGraph_t gr; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int it = 0; it < 4; it++)
        for (int i = 0; i < nrot; i++) {
            a.k = dk[i]; a.v = dv[i];
            k_flash<<<132, 256, smem, st>>>(a);
            if (!mode) k_merge<<<132, 256, 0, st>>>(a);
        }
    CK(cudaStreamEndCapture(st, &gr));
    CK(cudaGraphInstantiate(&ge, gr, 0));
    CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
    std::vector<float> ts;
    for (int rep = 0; rep < 7; rep++) {
        CK(cudaEventRecord(e0, st));
        CK(cudaGraphLaunch(ge, st));
        CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ts.push_back(ms * 1000 / (4 * nrot));
    }
    std::sort(ts.begin(), ts.end());
    const float t = ts[ts.size() / 2];
    const double floor_us = kv_bytes / 3.35e12 * 1e6;
    cudaFuncAttributes fa{}; CK(cudaFuncGetAttributes(&fa, k_flash));
    std::printf("hd%d GF%d B%u ctx%u win%u H%u/%u ns%u %s: %.2f us  %.0f GB/s  floor %.2f us  %.0f%%  relL2 %.1e  regs %d lmem %zu %s\n",
                D, GF, B, ctx, window, H, KVH, ns, mode ? "fold" : "merge", t, kv_bytes / t * 1e-3,
                floor_us, 100 * floor_us / t, worst, fa.numRegs, (size_t)fa.localSizeBytes,
                worst < 1e-2 ? "OK" : "FAIL");
    return worst < 1e-2 ? 0 : 1;
}

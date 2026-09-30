/* fa_stream_hnr_bench: the streamed hd128 decode with the layer's HeadNormRope folded in
 * (d_flash_decode_stream + FaStHnr) against d_headnorm_rope x3 + the shipped row-group fold:
 * the attention output and the new K/V cache rows must match bit for bit. Then timing of
 * (HNR launch + flash) vs the fused launch, graph-timed over rotating KV like fa_decode_bench.
 *   ./fshb B CTX [STRIDE]
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define PLOW_NV_HOPPER 1
#include "op_attention.cuh"
#include "op_norm.cuh"

using bf16 = __nv_bfloat16;
constexpr int D = 128, GF = 3, H = 24, KVH = 8, NQKV = (H + 2 * KVH) * D;

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); std::exit(2); } } while (0)

struct Args {
    bf16 *qkv, *q, *k, *v, *out;
    float *cosb, *sinb, *opart, *ml;
    int *pos, *kvlen;
    unsigned* ctr;
    unsigned B, stride;
    float scale;
};

__global__ void __launch_bounds__(256, 1) k_hnr(Args a) {
    /* the light_attn level: q, k, v HeadNormRope from the fused q|k|v rows */
    for (int j = 0; j < 3; j++) {
        if (j) __syncthreads();
        const unsigned nh = j == 0 ? H : KVH, col = j == 0 ? 0 : (j == 1 ? H * D : (H + KVH) * D);
        d_headnorm_rope<D>(j == 0 ? a.q : (j == 1 ? a.k : a.v), a.qkv + col, nullptr,
                           j == 2 ? nullptr : a.cosb, j == 2 ? nullptr : a.sinb, a.pos, a.B, nh, 1e-5f, 0,
                           j == 0 ? 0 : a.stride, 0xFFFFFFFFu, 1, blockIdx.x, gridDim.x,
                           j == 0 ? 0 : a.B, nullptr, nullptr, nullptr, NQKV);
    }
}
__global__ void __launch_bounds__(256, 1) k_flash(Args a) {
    extern __shared__ float arena[];
    d_flash_decode<D, GF>(a.opart, a.ml, a.q, a.k, a.v, a.kvlen, a.B, H, KVH, a.stride, 0, a.scale, 1,
                          0xFFFFFFFFu, blockIdx.x, gridDim.x, arena, 0, nullptr, nullptr, nullptr, a.out,
                          a.ctr);
}
__global__ void __launch_bounds__(FA_ST_THREADS, 1) k_fused(Args a) {
    extern __shared__ unsigned char st_arena[];
    FaStHnr h{a.qkv, a.qkv + H * D, a.qkv + (H + KVH) * D, NQKV, NQKV, NQKV, a.cosb, a.sinb, a.pos};
    d_flash_decode_stream<D, GF>(a.out, a.q, (bf16*)a.k, (bf16*)a.v, a.kvlen, nullptr, a.B, H, KVH, a.stride, a.scale,
                                 blockIdx.x, gridDim.x, st_arena, h);
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
    if (argc < 3) { std::fprintf(stderr, "usage: B CTX [STRIDE]\n"); return 2; }
    const unsigned B = atoi(argv[1]), ctx = atoi(argv[2]);
    const unsigned stride = argc > 3 ? atoi(argv[3]) : 2048;
    const size_t kv_el = (size_t)B * KVH * stride * D;
    const double kv_bytes = (double)B * KVH * ctx * D * 2 * 2;
    const int nrot = (int)std::max(2.0, std::min(48.0, std::ceil(160e6 / kv_bytes)));
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<int> hpos(B), hlen(B);
    for (unsigned b = 0; b < B; b++) { hlen[b] = (int)ctx - (int)(b % 7); hpos[b] = hlen[b] - 1; }
    const unsigned maxpos = ctx + 8;
    std::vector<float> hc((size_t)maxpos * 64), hs((size_t)maxpos * 64);
    for (unsigned p = 0; p < maxpos; p++)
        for (unsigned i = 0; i < 64; i++) {
            const double f = std::pow(500000.0, -2.0 * i / 128.0) * p;
            hc[p * 64 + i] = (float)std::cos(f); hs[p * 64 + i] = (float)std::sin(f);
        }
    auto hk = rnd(kv_el, 22, 2.0f), hv = rnd(kv_el, 33, 1.0f), hqkv = rnd((size_t)B * NQKV, 11, 2.0f);
    std::vector<bf16*> dk(nrot), dv(nrot);
    for (int i = 0; i < nrot; i++) {
        CK(cudaMalloc(&dk[i], kv_el * 2)); CK(cudaMalloc(&dv[i], kv_el * 2));
        CK(cudaMemcpy(dk[i], hk.data(), kv_el * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv[i], hv.data(), kv_el * 2, cudaMemcpyHostToDevice));
    }
    Args a{};
    a.B = B; a.stride = stride; a.scale = scale;
    CK(cudaMalloc(&a.qkv, hqkv.size() * 2)); CK(cudaMemcpy(a.qkv, hqkv.data(), hqkv.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&a.q, (size_t)B * H * D * 2)); CK(cudaMalloc(&a.out, (size_t)B * H * D * 2));
    CK(cudaMalloc(&a.cosb, hc.size() * 4)); CK(cudaMemcpy(a.cosb, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&a.sinb, hs.size() * 4)); CK(cudaMemcpy(a.sinb, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&a.opart, (size_t)B * H * D * 4)); CK(cudaMalloc(&a.ml, (size_t)B * H * 8));
    CK(cudaMalloc(&a.ctr, (size_t)B * H * 4)); CK(cudaMemset(a.ctr, 0, (size_t)B * H * 4));
    CK(cudaMalloc(&a.pos, B * 4)); CK(cudaMemcpy(a.pos, hpos.data(), B * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&a.kvlen, B * 4)); CK(cudaMemcpy(a.kvlen, hlen.data(), B * 4, cudaMemcpyHostToDevice));
    const size_t smem = std::max<size_t>(FA_DEC_SMEM_FLOATS(D, GF) * 4, 48 * 1024);
    CK(cudaFuncSetAttribute(k_flash, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CK(cudaFuncSetAttribute(k_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FA_ST_SMEM_BYTES(D)));

    /* reference on buffer 0, fused on buffer 1 (same history, stale new rows) */
    a.k = dk[0]; a.v = dv[0];
    k_hnr<<<132, 256>>>(a); k_flash<<<132, 256, smem>>>(a); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<bf16> o1((size_t)B * H * D), o2(o1.size()), k1(kv_el), k2(kv_el), v1(kv_el), v2(kv_el);
    CK(cudaMemcpy(o1.data(), a.out, o1.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(k1.data(), dk[0], kv_el * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(v1.data(), dv[0], kv_el * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemset(a.out, 0, o1.size() * 2));
    a.k = dk[1]; a.v = dv[1];
    k_fused<<<132, FA_ST_THREADS, FA_ST_SMEM_BYTES(D)>>>(a); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(o2.data(), a.out, o2.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(k2.data(), dk[1], kv_el * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(v2.data(), dv[1], kv_el * 2, cudaMemcpyDeviceToHost));
    size_t od = 0, kd = 0, vd = 0;
    for (size_t i = 0; i < o1.size(); i++) od += memcmp(&o1[i], &o2[i], 2) != 0;
    for (size_t i = 0; i < kv_el; i++) { kd += memcmp(&k1[i], &k2[i], 2) != 0; vd += memcmp(&v1[i], &v2[i], 2) != 0; }
    std::printf("fused vs hnr+rg: out %zu/%zu  k %zu  v %zu differ\n", od, o1.size(), kd, vd);

    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int fused = 0; fused < 2; fused++) {
        cudaGraph_t gr; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        for (int it = 0; it < 4; it++)
            for (int i = 0; i < nrot; i++) {
                a.k = dk[i]; a.v = dv[i];
                if (fused) k_fused<<<132, FA_ST_THREADS, FA_ST_SMEM_BYTES(D), st>>>(a);
                else { k_hnr<<<132, 256, 0, st>>>(a); k_flash<<<132, 256, smem, st>>>(a); }
            }
        CK(cudaStreamEndCapture(st, &gr));
        CK(cudaGraphInstantiate(&ge, gr, 0));
        CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
        std::vector<float> ts;
        for (int rep = 0; rep < 7; rep++) {
            CK(cudaEventRecord(e0, st)); CK(cudaGraphLaunch(ge, st));
            CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ts.push_back(ms * 1000 / (4 * nrot));
        }
        std::sort(ts.begin(), ts.end());
        std::printf("%s B%u ctx%u stride%u: %.2f us/layer (floor %.2f)\n", fused ? "fused   " : "hnr+rg  ", B, ctx,
                    stride, ts[3], kv_bytes / 3.35e12 * 1e6);
    }
    return od || kd || vd;
}

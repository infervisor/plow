/* Gemma-4 E4B-geometry flash-prefill A/B: the shipped sm90 body vs PLOW_NV_FA_V3.
 *
 *   nvcc -arch=sm_90a -O3 -std=c++17 -I runtime/common -I runtime/nvidia -DFAB_HD=256 \
 *        runtime/nvidia/experiments/fa_v3_bench.cu -lcuda -o fab256
 *   ./fab256 ROWS KV_LEN [NREQ] [HEADS KV_HEADS]
 *
 * NREQ > 1 packs NREQ requests of ROWS/NREQ query rows (kv KV_LEN each, own slot) through the
 * varlen request table + per-slot TMA map table, as the packed prefill object does. Sampled rows
 * are checked against an FP64 oracle for both bodies; both are timed (median of 15, L2 evicted).
 */
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define PLOW_NV_HOPPER 1
#define PLOW_NV_FA_PIPE 1
#define PLOW_NV_FA_TMA 1
#define PLOW_NV_FA512_WG 1
#define PLOW_NV_FA512_BKV 32
#define PLOW_NV_PACKED_FA_TMA 1
#define PLOW_NV_FA_V3 0 /* k_old = the shipped body; k_v3 calls v3 directly */
#ifndef FA3_BOX
#define FA3_BOX 32
#endif
#include "op_attention.cuh"
#include "op_attention_sm90_v3.cuh"

#ifndef FAB_HD
#define FAB_HD 256
#endif
using bf16 = __nv_bfloat16;
constexpr int HD = FAB_HD;
constexpr int OLD_BKV = 32;

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); std::exit(2); } } while (0)
#define CD(call) do { CUresult e_ = (call); if (e_ != CUDA_SUCCESS) { const char* t_ = nullptr; \
    cuGetErrorString(e_, &t_); std::fprintf(stderr, "%s: %s\n", #call, t_ ? t_ : "?"); std::exit(2); } } while (0)

struct Geo {
    unsigned rows, n_head, n_kv, window, stride, mask;
    float scale;
};

__global__ void __launch_bounds__(256, 1) k_old(Geo g, const int* req, const bf16* q, const bf16* k,
                                                const bf16* v, bf16* out, unsigned kvlen,
                                                const void* maps) {
    extern __shared__ float arena[];
    d_flash_prefill<HD, 64, OLD_BKV>(nullptr, nullptr, q, k, v, out, g.rows, kvlen, g.n_head,
                                     g.n_kv, kvlen - g.rows, g.window, 1, g.stride, g.mask,
                                     g.scale, blockIdx.x, gridDim.x, arena, req, maps);
}

__global__ void __launch_bounds__(256, 1) k_v3(Geo g, const int* req, const bf16* q, bf16* out,
                                               unsigned kvlen, const void* maps) {
    extern __shared__ float arena[];
    d_flash_prefill_sm90_v3<HD>(q, out, g.rows, kvlen, g.n_head, g.n_kv, kvlen - g.rows,
                                g.window, g.stride, g.mask, g.scale, blockIdx.x, gridDim.x, arena,
                                req, maps);
}

__global__ void evict(unsigned* d, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        d[i] += 1;
}

static std::vector<bf16> rnd(size_t n, uint32_t seed, float amp) {
    std::vector<bf16> r(n);
    for (auto& x : r) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        x = __float2bfloat16(amp * float(int32_t(seed)) / 2147483648.0f);
    }
    return r;
}
template <class T> static T* up(const std::vector<T>& h) {
    T* d; CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: ROWS KV_LEN [NREQ] [HEADS KVHEADS]\n"); return 2; }
    const unsigned rows = atoi(argv[1]), kvlen = atoi(argv[2]);
    const unsigned nreq = argc > 3 ? atoi(argv[3]) : 1;
    const unsigned H = argc > 5 ? atoi(argv[4]) : 8, KVH = argc > 5 ? atoi(argv[5]) : 2;
    const unsigned qlen = rows / nreq;
    if (qlen * nreq != rows || kvlen < qlen) return 2;
    Geo g{};
    g.rows = rows; g.n_head = H; g.n_kv = KVH;
    g.window = HD == 256 ? 512 : 0;
    g.stride = HD == 256 ? 4096 : 8192;
    g.mask = HD == 256 ? 4095u : 0xffffffffu;
    g.scale = 1.0f / std::sqrt(float(HD));
    if (kvlen > g.stride && HD == 512) return 2;

    const size_t qn = (size_t)rows * H * HD, kvn = (size_t)nreq * KVH * g.stride * HD;
    auto hq = rnd(qn, 11, 2.0f), hk = rnd(kvn, 22, 2.0f), hv = rnd(kvn, 33, 1.0f);
    /* KV rows past each request's live length hold NaN: the kernels must never let them in. */
    for (unsigned s = 0; s < nreq; s++)
        for (unsigned hh = 0; hh < KVH; hh++)
            for (unsigned pos = kvlen; pos < std::min(g.stride, kvlen + 256); pos++) {
                const size_t row = (size_t)(s * KVH + hh) * g.stride + (pos & g.mask);
                if (HD == 256 && pos >= g.stride) continue;
                for (int d = 0; d < HD; d++) {
                    hk[row * HD + d] = __float2bfloat16(NAN);
                    hv[row * HD + d] = __float2bfloat16(NAN);
                }
            }
    bf16 *dq = up(hq), *dk = up(hk), *dv = up(hv), *o_old, *o_v3;
    CK(cudaMalloc(&o_old, qn * 2)); CK(cudaMalloc(&o_v3, qn * 2));

    /* per-slot map blobs [K map | V map] + the uint64 table the packed path indexes by slot */
    std::vector<CUtensorMap> blobs(2 * nreq);
    for (unsigned s = 0; s < nreq; s++) {
        const uint64_t dims[]{HD, g.stride, KVH};
        const uint64_t strides[]{HD * 2ull, (uint64_t)HD * g.stride * 2ull};
        const uint32_t box[]{64, FA3_BOX, 1}, steps[]{1, 1, 1};
        for (int op = 0; op < 2; op++) {
            void* base = (op ? (void*)dv : (void*)dk);
            base = (char*)base + (size_t)s * KVH * g.stride * HD * 2;
            CD(cuTensorMapEncodeTiled(&blobs[2 * s + op], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, base,
                                      dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        }
    }
    CUtensorMap* dblobs = up(blobs);
    std::vector<uint64_t> table(nreq);
    for (unsigned s = 0; s < nreq; s++) table[s] = (uint64_t)(dblobs + 2 * s);
    uint64_t* dtable = up(table);
    int* dreq = nullptr;
    if (nreq > 1) {
        std::vector<int> r{(int)nreq};
        for (unsigned s = 0; s < nreq; s++) {
            r.push_back(s * qlen); r.push_back(qlen); r.push_back(s); r.push_back(kvlen);
        }
        dreq = up(r);
    }
    const void* maps = nreq > 1 ? (const void*)dtable : (const void*)dblobs;
    const unsigned kv_arg = nreq > 1 ? kvlen : kvlen;

    const size_t sm_old = FA_PRE_SMEM_FLOATS(HD, 64, OLD_BKV) * 4;
    const size_t sm_v3 = FA3_SMEM_FLOATS(HD) * 4;
    CK(cudaFuncSetAttribute(k_old, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_old));
    CK(cudaFuncSetAttribute(k_v3, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_v3));
    cudaFuncAttributes a_old{}, a_v3{};
    CK(cudaFuncGetAttributes(&a_old, k_old));
    CK(cudaFuncGetAttributes(&a_v3, k_v3));
    const unsigned grid = 132;
    Geo gk = g;
    if (nreq > 1) gk.rows = rows;
    auto run_old = [&] { k_old<<<grid, 256, sm_old>>>(gk, dreq, dq, dk, dv, o_old, kv_arg, maps); };
    auto run_v3 = [&] { k_v3<<<grid, 256, sm_v3>>>(gk, dreq, dq, o_v3, kv_arg, maps); };
    CK(cudaMemset(o_old, 0xff, qn * 2)); CK(cudaMemset(o_v3, 0xff, qn * 2));
#ifndef FAB_NO_OLD
    run_old(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
#endif
    run_v3(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<bf16> ho(qn), h3(qn);
    CK(cudaMemcpy(ho.data(), o_old, qn * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h3.data(), o_v3, qn * 2, cudaMemcpyDeviceToHost));

    double worst[2] = {0, 0};
    bool finite = true;
    std::vector<unsigned> samp;
    for (unsigned r : {0u, 1u, 63u, 64u, 127u, 128u, qlen / 2, qlen - 2, qlen - 1}) if (r < qlen) samp.push_back(r);
    for (unsigned s = 0; s < nreq; s++)
        for (unsigned r : samp)
            for (unsigned hh = 0; hh < H; hh += (H > 4 ? 3 : 1)) {
                const unsigned qabs = kvlen - qlen + r;
                const unsigned end = qabs + 1, begin = g.window && end > g.window ? end - g.window : 0;
                const unsigned kvh = hh / (H / KVH);
                const size_t qi = ((size_t)(s * qlen + r) * H + hh) * HD;
                const size_t kb = (size_t)(s * KVH + kvh) * g.stride * HD;
                std::vector<double> sc(end - begin);
                double mx = -1e300;
                for (unsigned p = begin; p < end; p++) {
                    double a = 0;
                    const size_t ki = kb + (size_t)(p & g.mask) * HD;
                    for (int d = 0; d < HD; d++) a += double(__bfloat162float(hq[qi + d])) * __bfloat162float(hk[ki + d]);
                    sc[p - begin] = a * g.scale; mx = std::max(mx, sc[p - begin]);
                }
                double sum = 0; for (double& x : sc) { x = std::exp(x - mx); sum += x; }
                double e2[2] = {0, 0}, r2 = 0;
                for (int d = 0; d < HD; d++) {
                    double ex = 0;
                    for (unsigned p = begin; p < end; p++) ex += sc[p - begin] * __bfloat162float(hv[kb + (size_t)(p & g.mask) * HD + d]);
                    ex /= sum;
                    const double a0 = __bfloat162float(ho[qi + d]), a1 = __bfloat162float(h3[qi + d]);
                    finite &= std::isfinite(a1);
                    e2[0] += (a0 - ex) * (a0 - ex); e2[1] += (a1 - ex) * (a1 - ex); r2 += ex * ex;
                }
                for (int i = 0; i < 2; i++) worst[i] = std::max(worst[i], std::sqrt(e2[i] / std::max(r2, 1e-30)));
            }
    /* padded/tail rows must not have been touched beyond the real ones: check full finite */
    size_t nonfinite = 0;
    for (size_t i = 0; i < qn; i++) nonfinite += !std::isfinite(__bfloat162float(h3[i]));

    unsigned* trash; const size_t tb = 256ull << 20;
    CK(cudaMalloc(&trash, tb)); CK(cudaMemset(trash, 0, tb));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto timeit = [&](auto&& f) {
        for (int i = 0; i < 3; i++) f();
        std::vector<float> v;
        for (int s = 0; s < 15; s++) {
            evict<<<132, 256>>>(trash, tb / 4);
            CK(cudaEventRecord(e0)); f(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms * 1000);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    #ifdef FAB_NO_OLD
    const float t_old = 1.0f, t_v3 = timeit(run_v3);
#else
    const float t_old = timeit(run_old), t_v3 = timeit(run_v3);
#endif
    double flops = 0;
    for (unsigned r = 0; r < qlen; r++) {
        const unsigned qabs = kvlen - qlen + r;
        const unsigned n = g.window ? std::min(qabs + 1, g.window) : qabs + 1;
        flops += 4.0 * n * HD * H;
    }
    flops *= nreq;
    std::printf("hd%d rows %u kv %u nreq %u H %u/%u | old %.1f us (%.0f TF/s) relL2 %.2e regs %d lmem %zu | "
                "v3 %.1f us (%.0f TF/s) relL2 %.2e regs %d lmem %zu nonfinite %zu | x%.2f %s\n",
                HD, rows, kvlen, nreq, H, KVH, t_old, flops / t_old * 1e-6, worst[0], a_old.numRegs,
                (size_t)a_old.localSizeBytes, t_v3, flops / t_v3 * 1e-6, worst[1], a_v3.numRegs,
                (size_t)a_v3.localSizeBytes, nonfinite, t_old / t_v3,
                (finite && nonfinite == 0 && worst[1] < 5e-3) ? "OK" : "FAIL");
    return (finite && nonfinite == 0 && worst[1] < 5e-3) ? 0 : 1;
}

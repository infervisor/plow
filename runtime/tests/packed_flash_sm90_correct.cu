#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dev_isa.h"
#define PLOW_NV_HOPPER 1
#define PLOW_NV_FA_PIPE 1
#define PLOW_NV_FA_TMA 1
#define PLOW_NV_FA512_WG 1
#define PLOW_NV_PACKED_REQUEST 1
#define PLOW_NV_PACKED_FA_WGMMA 1
#define PLOW_NV_PACKED_FA_TMA 1
#include "op_attention.cuh"

using bf16 = __nv_bfloat16;
#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(2); } } while (0)

constexpr unsigned heads = 16, capacity = 128, real_rows = 98, blocks = 132;

template<int HD, int BKV>
__global__ void run_attention(const bf16* q, const bf16* k, const bf16* v,
                             bf16* out, float* partial, float* stats, const int* req,
                             unsigned kv_heads, unsigned stride, unsigned mask,
                             unsigned window, const void* maps) {
    extern __shared__ float arena[];
    d_flash_prefill_mux<HD,64,BKV>(req, partial, stats, q, k, v, out,
        capacity, 16384, heads, kv_heads, 0, window, 1, stride, mask,
        1.0f / sqrtf(float(HD)), blockIdx.x, blocks, arena, maps);
}

static std::vector<bf16> values(size_t n, uint32_t seed) {
    std::vector<bf16> result(n);
    for (auto& value : result) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        value = __float2bfloat16(float(int32_t(seed)) / 2147483648.0f);
    }
    return result;
}

template<class T> static T* upload(const std::vector<T>& host) {
    T* device;
    CK(cudaMalloc(&device, host.size() * sizeof(T)));
    CK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

template<int HD, int BKV> static bool check(unsigned kv_heads, unsigned stride,
                                          unsigned mask, unsigned window, bool tma) {
    const auto q = values(size_t(capacity) * heads * HD, 123);
    const auto k = values(size_t(3) * kv_heads * stride * HD, 456);
    const auto v = values(k.size(), 789);
    // Two ragged requests use reversed, noncontiguous slots; the last 30 rows are padding.
    const std::vector<int> req{2, 0, 65, 2, 97, 65, 33, 0, 16384};
    bf16* dq = upload(q), *dk = upload(k), *dv = upload(v), *out;
    int* dr = upload(req);
    std::vector<CUtensorMap> maps(6);
    CUtensorMap* dm = nullptr;
    uint64_t* table = nullptr;
    if (tma) {
        for (unsigned slot = 0; slot < 3; ++slot) for (unsigned operand = 0; operand < 2; ++operand) {
            uint64_t dims[]{HD, stride, kv_heads};
            uint64_t strides[]{HD * 2, uint64_t(HD) * stride * 2};
            uint32_t box[]{64, 32, 1}, steps[]{1, 1, 1};
            bf16* base = (operand ? dv : dk) + size_t(slot) * kv_heads * stride * HD;
            const CUresult result = cuTensorMapEncodeTiled(&maps[slot * 2 + operand],
                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, base, dims, strides, box, steps,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (result != CUDA_SUCCESS) { std::fprintf(stderr, "tensor map: %d\n", int(result)); std::exit(2); }
        }
        dm = upload(maps);
        table = upload(std::vector<uint64_t>{uint64_t(dm), 0, uint64_t(dm + 4)});
    }
    float *partial, *stats;
    CK(cudaMalloc(&out, q.size() * sizeof(bf16)));
    CK(cudaMalloc(&partial, q.size() * sizeof(float)));
    CK(cudaMalloc(&stats, size_t(capacity) * heads * 2 * sizeof(float)));
    CK(cudaMemset(out, 0xff, q.size() * sizeof(bf16)));
    const unsigned smem = FA_PRE_SMEM_FLOATS(HD,64,BKV) * sizeof(float);
    CK(cudaFuncSetAttribute(run_attention<HD,BKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    run_attention<HD,BKV><<<blocks,256,smem>>>(dq,dk,dv,out,partial,stats,dr,kv_heads,stride,mask,window,table);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<bf16> got(q.size());
    CK(cudaMemcpy(got.data(), out, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
    std::vector<float> samples;
    for (unsigned sample = 0; sample < 7; ++sample) {
        CK(cudaEventRecord(start));
        for (unsigned repeat = 0; repeat < 10; ++repeat)
            run_attention<HD,BKV><<<blocks,256,smem>>>(dq,dk,dv,out,partial,stats,dr,kv_heads,stride,mask,window,table);
        CK(cudaEventRecord(stop)); CK(cudaEventSynchronize(stop));
        float ms;
        CK(cudaEventElapsedTime(&ms,start,stop));
        samples.push_back(ms * 100.0f);
    }
    std::sort(samples.begin(),samples.end());
    CK(cudaEventDestroy(start)); CK(cudaEventDestroy(stop));
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        const float value = __bfloat162float(got[i]);
        ok &= std::isfinite(value);
        if (i >= size_t(real_rows) * heads * HD) ok &= value == 0.0f;
    }
    double worst = 0, max_error = 0;
    unsigned checked = 0;
    for (unsigned r = 0; r < 2; ++r) {
        const unsigned row0=req[1+4*r], len=req[2+4*r], slot=req[3+4*r], kvlen=req[4+4*r];
        for (unsigned row : {0u, len/2, len-1}) for (unsigned h : {0u, 7u, 15u}) {
            const unsigned end = kvlen-len+row+1;
            const unsigned begin = window && end > window ? end-window : 0;
            const size_t qi=(size_t(row0+row)*heads+h)*HD;
            const size_t base=(size_t(slot)*kv_heads+h/(heads/kv_heads))*stride*HD;
            std::vector<double> scores(end-begin);
            double maximum = -INFINITY;
            for (unsigned pos=begin; pos<end; ++pos) {
                const size_t ki=base+size_t(pos&mask)*HD;
                double score=0;
                for (unsigned d=0; d<HD; ++d)
                    score += double(__bfloat162float(q[qi+d])) * __bfloat162float(k[ki+d]);
                score /= std::sqrt(double(HD));
                scores[pos-begin]=score;
                maximum=std::max(maximum,score);
            }
            double sum=0;
            for (auto& score : scores) { score=std::exp(score-maximum); sum+=score; }
            double error2=0, reference2=0;
            for (unsigned d=0; d<HD; ++d) {
                double expected=0;
                for (unsigned pos=begin; pos<end; ++pos)
                    expected += scores[pos-begin] * __bfloat162float(v[base+size_t(pos&mask)*HD+d]);
                expected/=sum;
                const double error=__bfloat162float(got[qi+d])-expected;
                error2+=error*error; reference2+=expected*expected;
                max_error=std::max(max_error,std::abs(error));
                ++checked;
            }
            worst=std::max(worst,std::sqrt(error2/std::max(reference2,1e-30)));
        }
    }
    ok &= worst < 0.004 && max_error < 0.01;
    std::printf("HD=%d KV=%u window=%u maps=%d checked=%u worst_relL2=%.6g max_abs=%.6g warm_us=%.3f %s\n",
                HD,kv_heads,window,int(tma),checked,worst,max_error,samples[3],ok?"PASS":"FAIL");
    CK(cudaFree(dq)); CK(cudaFree(dk)); CK(cudaFree(dv)); CK(cudaFree(out));
    CK(cudaFree(partial)); CK(cudaFree(stats)); CK(cudaFree(dr));
    if (tma) { CK(cudaFree(table)); CK(cudaFree(dm)); }
    return ok;
}

int main() {
    bool ok = true;
    for (bool tma : {false, true}) {
        ok &= check<256,32>(8,2048,2047,1024,tma);
        ok &= check<512,16>(1,16384,0xffffffffu,0,tma);
    }
    return ok ? 0 : 1;
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define PLOW_NV_HOPPER 1
#define PLOW_FP8_KV 1
#define PLOW_NV_PREFILL 1
#define PLOW_NV_FA_PIPE 1
#define PLOW_NV_FA_GF 2
#define PLOW_NV_FA_WPR 1
#define PLOW_NV_PACKED_REQUEST 1
#include "op_attention.cuh"

using bf16 = __nv_bfloat16;
#define CK(call) do { auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(2); } } while (0)
template<class T> static T* upload(const std::vector<T>& v) {
    T* p; CK(cudaMalloc(&p, v.size() * sizeof(T)));
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
static float decode(uint8_t v) { __nv_fp8_e4m3 f; f.__x = v; return float(f); }

template<int HD>
__global__ void attention(const int* req, float* partial, float* stats, const bf16* q,
                          const uint8_t* k, const uint8_t* v, bf16* out,
                          const float* ks, const float* vs, unsigned rows,
                          unsigned kvheads, unsigned stride, unsigned mask, unsigned window) {
    extern __shared__ float arena[];
    d_flash_prefill_fp8_mux<HD>(req, partial, stats, q, k, v, out, ks, vs,
        rows, 16384, 16, kvheads, 0, window, 1, stride, mask,
        1.0f / sqrtf(float(HD)), blockIdx.x, gridDim.x, arena);
}

template<int HD> static void check(unsigned rows, unsigned kvheads, unsigned window) {
    const unsigned stride = window ? 2048 : 16384, mask = window ? 2047 : ~0u;
    const unsigned len = std::min(rows / 2 - 3, 1024u);
    const unsigned real = len + len - 5;
    std::vector<int> req = {2, 0, int(len), 2, 16384, int(len), int(len-5), 0, 8193};
    std::vector<bf16> q(size_t(rows) * 16 * HD), got(q.size());
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = __float2bfloat16(float(int((i * 17) % 127) - 63) / 73.0f);
    std::vector<uint8_t> k(size_t(3) * kvheads * stride * HD), v(k.size());
    std::vector<float> ks(size_t(3) * kvheads * stride), vs(ks.size());
    for (size_t i = 0; i < ks.size(); ++i) {
        ks[i] = 0.3f + float(i % 7) * 0.07f;
        vs[i] = 0.2f + float(i % 11) * 0.03f;
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = __nv_fp8_e4m3(float(int((i * 13) % 97) - 48) / 32.0f).__x;
        v[i] = __nv_fp8_e4m3(float(int((i * 19) % 101) - 50) / 32.0f).__x;
    }
    auto* dq = upload(q); auto* dk = upload(k); auto* dv = upload(v);
    auto* dks = upload(ks); auto* dvs = upload(vs); auto* dr = upload(req);
    bf16* out; float* partial; float* stats;
    CK(cudaMalloc(&out, got.size() * sizeof(bf16)));
    CK(cudaMalloc(&partial, got.size() * sizeof(float)));
    CK(cudaMalloc(&stats, size_t(rows) * 16 * 2 * sizeof(float)));
    CK(cudaMemset(out, 0xff, got.size() * sizeof(bf16)));
    constexpr unsigned smem = FA_PRE_SMEM_FLOATS(HD, HD == 256 ? 64 : 32, HD == 256 ? 32 : 16) * sizeof(float);
    CK(cudaFuncSetAttribute(attention<HD>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attention<HD><<<132,256,smem>>>(dr, partial, stats, dq, dk, dv, out, dks, dvs,
                                  rows, kvheads, stride, mask, window);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), out, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i)
        ok &= std::isfinite(__bfloat162float(got[i])) &&
              (i < size_t(real) * 16 * HD || __bfloat162float(got[i]) == 0.0f);
    double worst = 0, max_abs = 0;
    unsigned checked = 0;
    for (unsigned r = 0; r < 2; ++r) {
        const unsigned q0 = req[1+4*r], n = req[2+4*r], slot = req[3+4*r], kvlen = req[4+4*r];
        for (unsigned row : {0u,n/2,n-1}) for (unsigned h : {0u,7u,15u}) {
            const unsigned end = kvlen - n + row + 1;
            const unsigned begin = window && end > window ? end - window : 0;
            const size_t qi = size_t((q0 + row) * 16 + h) * HD;
            const size_t base = size_t(slot * kvheads + h / (16 / kvheads)) * stride;
            std::vector<double> scores(end - begin);
            double mx = -INFINITY;
            for (unsigned pos = begin; pos < end; ++pos) {
                const size_t ki = base + (pos & mask);
                double dot = 0;
                for (unsigned d = 0; d < HD; ++d)
                    dot += double(__bfloat162float(q[qi+d])) * decode(k[ki*HD+d]) * ks[ki];
                scores[pos-begin] = dot / std::sqrt(double(HD)); mx = std::max(mx,scores[pos-begin]);
            }
            double sum = 0;
            for (auto& s : scores) { s = std::exp(s - mx); sum += s; }
            double error2 = 0, ref2 = 0;
            for (unsigned d = 0; d < HD; ++d) {
                double expected = 0;
                for (unsigned pos = begin; pos < end; ++pos) {
                    const size_t vi = base + (pos & mask);
                    expected += scores[pos-begin] * decode(v[vi*HD+d]) * vs[vi];
                }
                expected /= sum;
                const double e = __bfloat162float(got[qi+d]) - expected;
                error2 += e*e; ref2 += expected*expected; max_abs = std::max(max_abs,std::abs(e)); ++checked;
            }
            worst = std::max(worst,std::sqrt(error2/std::max(ref2,1e-30)));
        }
    }
    ok &= worst < 0.015 && max_abs < 0.002;
    std::printf("%s FP8 rows=%u HD=%d KV=%u real=%u checked=%u relL2=%.6g max_abs=%.6g\n",
                ok ? "PASS" : "FAIL",rows,HD,kvheads,real,checked,worst,max_abs);
    for (void* p : {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dv),
                   static_cast<void*>(dks), static_cast<void*>(dvs), static_cast<void*>(dr),
                   static_cast<void*>(out), static_cast<void*>(partial), static_cast<void*>(stats)}) CK(cudaFree(p));
    if (!ok) std::exit(1);
}
int main() {
    for (unsigned rows : {128,512,1024,2048,4096,8192}) {
        check<256>(rows,8,1024);
        check<512>(rows,1,0);
    }
}

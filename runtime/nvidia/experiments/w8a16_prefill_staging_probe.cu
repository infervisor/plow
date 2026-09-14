// Build with -gencode=arch=compute_90a,code=sm_90a and PLOW_NV_W8A16_PREFETCH=0/1.
// The argument is an existing directory for full-output comparisons. Integer inputs
// and power-of-two scales make the sampled FP64 oracle exact through FP32 accumulation.
// PLOW_STAGING_MMA_SPLIT=1/2/4/8 selects the native MMA comparison;
// PLOW_STAGING_MMA_SLICED=1 uses packet ownership and OPT_BLOCKS=1 uses natural tiles.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <utility>
#define PLOW_NV_HOPPER 1
#define PLOW_NV_GEMMA 1
#define PLOW_NV_PREFILL 1
#define PLOW_NV_W8A16_WGMMA 1
#include "op_gemm.cuh"
#include "gemma_fp8_w8a16_mma.cuh"

#define CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)
using bf16 = __nv_bfloat16;

__global__ void gemm(bf16* c, const bf16* a, const uint8_t* b,
                     const float* scale, unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 arena[];
    d_gemm_fp8(c, a, b, scale, m, n, k, 0, blockIdx.x, gridDim.x, arena);
}

template<class T> T* device(const std::vector<T>& host) {
    T* out;
    CHECK(cudaMalloc(&out, host.size() * sizeof(T)));
    CHECK(cudaMemcpy(out, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return out;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const int repeats = std::getenv("PLOW_STAGING_CHECK_ONLY") ? 0 : 7;
    const bool race_case = std::getenv("PLOW_STAGING_RACE_CASE");
    const unsigned mma_split = std::getenv("PLOW_STAGING_MMA_SPLIT")
        ? std::strtoul(std::getenv("PLOW_STAGING_MMA_SPLIT"), nullptr, 10) : 0;
    const bool mma_sliced = std::getenv("PLOW_STAGING_MMA_SLICED");
    const bool mma_opt_blocks = std::getenv("PLOW_STAGING_MMA_OPT_BLOCKS");
    if (mma_split > 8 || (mma_split && (mma_split & (mma_split - 1)))) return 2;
    const unsigned smem = PGM_ARENA_BF16 * sizeof(bf16);
    CHECK(cudaFuncSetAttribute(gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    const std::vector<unsigned> rungs = std::getenv("PLOW_STAGING_ALL_RUNGS")
        ? std::vector<unsigned>{1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192}
        : std::vector<unsigned>{128,512,1024,2048};
    for (unsigned m : rungs) {
        if (mma_split && m > 64) continue;
        for (auto shape : {std::pair{15360u,3840u}, std::pair{3840u,15360u},
                           std::pair{4096u,3840u}, std::pair{512u,3840u},
                           std::pair{2048u,3840u}, std::pair{8192u,3840u},
                           std::pair{3840u,4096u}, std::pair{3840u,8192u},
                           std::pair{513u,3856u}, std::pair{513u,3848u}}) {
            if (race_case && (m != 128 || shape != std::pair{513u,3856u})) continue;
            const auto selected = race_case ? std::pair{129u,208u} : shape;
            const unsigned n = selected.first, k = selected.second;
            std::mt19937 rng(42);
            std::vector<bf16> a(size_t(m)*k), c(size_t(m)*n);
            std::vector<uint8_t> b(size_t(n)*k);
            std::vector<float> scale(n);
            for (auto& x : a) x = __float2bfloat16(int(rng()%7)-3);
            for (auto& x : b) { __nv_fp8_e4m3 q(float(int(rng()%13)-6)); x = *(uint8_t*)&q; }
            for (unsigned j=0; j<n; ++j) scale[j] = std::ldexp(1.f, -5 + int(j%3));
            auto* da=device(a); auto* db=device(b); auto* ds=device(scale); auto* dc=device(c);
            float* partial = nullptr;
            const unsigned active_mma_split = (k & 15) ? 0 : mma_split;
            unsigned packet_blocks = n == 512 && k == 3840 ? 8
                : n == 2048 && k == 3840 ? 33
                : (n == 4096 || n == 15360) && k == 3840 ? 66
                : n == 8192 && k == 3840 ? 124
                : n == 3840 && (k == 4096 || k == 8192 || k == 15360) ? 264 : 132;
            if (active_mma_split && mma_opt_blocks)
                packet_blocks = min(264u, ((m + 15) / 16) * ((n + 63) / 64));
            if (active_mma_split > 1)
                CHECK(cudaMalloc(&partial, size_t(m)*n*active_mma_split*sizeof(float)));
            auto launch = [&] {
                if (!active_mma_split) gemm<<<132,256,smem>>>(dc,da,db,ds,m,n,k);
                else if (active_mma_split == 1) {
                    if (mma_sliced)
                        gemma_fp8_probe::w8a16_mma_sliced<<<min(packet_blocks,mma_opt_blocks?264u:132u),256>>>
                            (dc,da,db,ds,m,n,k,packet_blocks);
                    else gemma_fp8_probe::w8a16_mma<false><<<dim3((n+63)/64,(m+15)/16),256>>>
                            (dc,partial,da,db,ds,m,n,k,1);
                }
                else {
                    gemma_fp8_probe::w8a16_mma<true><<<dim3((n+63)/64,(m+15)/16,active_mma_split),256>>>
                        (dc,partial,da,db,ds,m,n,k,active_mma_split);
                    gemma_fp8_probe::reduce<<<(size_t(m)*n+255)/256,256>>>
                        (dc,partial,ds,m,n,active_mma_split);
                }
                CHECK(cudaGetLastError());
            };
            launch();
            CHECK(cudaMemcpy(c.data(),dc,c.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
            for (unsigned sample=0; sample<96; ++sample) {
                unsigned row=rng()%m, col=rng()%n;
                double sum=0;
                for (unsigned j=0;j<k;++j) {
                    __nv_fp8_e4m3 q; *(uint8_t*)&q=b[size_t(col)*k+j];
                    sum+=double(__bfloat162float(a[size_t(row)*k+j]))*float(q);
                }
                bf16 expected=__float2bfloat16(float(sum*scale[col]));
                if (__bfloat162float(expected)!=__bfloat162float(c[size_t(row)*n+col])) {
                    fprintf(stderr,"oracle failure M%u N%u K%u row%u col%u\n",m,n,k,row,col); return 1;
                }
            }
            char path[4096]; snprintf(path,sizeof(path),"%s/%u-%u-%u.bin",argv[1],m,n,k);
            FILE* file=fopen(path,"wb"); if (!file) return 2;
            if (fwrite(c.data(),sizeof(bf16),c.size(),file)!=c.size()) return 2;
            fclose(file);
            cudaEvent_t start,end; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
            if (repeats) { launch(); CHECK(cudaDeviceSynchronize()); }
            CHECK(cudaEventRecord(start));
            for (int repeat=0;repeat<repeats;++repeat) launch();
            CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
            float ms; CHECK(cudaEventElapsedTime(&ms,start,end));
            printf("M=%u N=%u K=%u us=%.3f oracle=PASS smem=%u mma_split=%u sliced=%u blocks=%u\n",
                   m,n,k,repeats ? ms*1000/repeats : 0,smem,active_mma_split,
                   unsigned(mma_sliced),packet_blocks); fflush(stdout);
            CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
            CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(ds)); CHECK(cudaFree(dc));
            if (partial) CHECK(cudaFree(partial));
        }
    }
}

/* Packed attention A/B: stock request loop vs a rotated CTA tile walk.
 * Both arms use the same native attention body, linear BF16 KV and NS1, with
 * no per-slot tensor maps. The candidate
 * stays here until device correctness and complete-packet gates pass.
 *
 * Build with the clean nvcc environment used in runtime/cmake/nvcc_cubin.sh:
 *   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 -DPLOW_NV_HOPPER=1 \
 *     -DPLOW_NV_PACKED_REQUEST=1 -DPLOW_NV_FA_TMA=1 \
 *     -I runtime/common -I runtime/nvidia \
 *     runtime/nvidia/experiments/fa_varlen_bench.cu -o /tmp/fa_varlen_bench
 * --self-test and --plan do not initialize CUDA. Capture clocks beside timing
 * stdout with the campaign's nvidia-smi dmon -s pc command.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "sm120_common.cuh"

#if !PLOW_NV_PACKED_REQUEST || PLOW_NV_FA_WGITEM || defined(PLOW_FP8_KV)
#error "This experiment requires packed BF16 attention with one work item per CTA"
#endif

using bf16 = __nv_bfloat16;
constexpr int MAX_REQUESTS = 16;
#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess) { \
    fprintf(stderr,"CUDA ERROR %s: %s\n",#x,cudaGetErrorString(e_)); exit(2); } } while(0)

__host__ __device__ unsigned next_slice(unsigned first, unsigned work, unsigned grid) {
    return (first + grid - work % grid) % grid;
}

__global__ void initialize(bf16* p, size_t n, unsigned seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        unsigned x = (unsigned)i + seed;
        x = (x ^ (x >> 16)) * 0x7feb352du;
        x = (x ^ (x >> 15)) * 0x846ca68bu;
        p[i] = __float2bfloat16(((int)(x & 2047) - 1024) / 1024.0f);
    }
}

__global__ void evict(unsigned* p, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) p[i] += 1;
}

template <int HD, int BQ, int BKV, bool ROTATE>
__global__ void k_attention(const int* req, bf16* O, const bf16* Q, const bf16* K,
                           const bf16* V, float* partials, float* stats,
                           unsigned rows, unsigned nh, unsigned nkv,
                           unsigned win, unsigned kv_stride, float scale) {
    extern __shared__ float sm[];
    if constexpr (!ROTATE) {
        d_flash_prefill_mux<HD,BQ,BKV>(req, partials, stats, Q, K, V, O,
            rows, 0, nh, nkv, 0, win, 1, kv_stride, 0xFFFFFFFFu,
            scale, blockIdx.x, gridDim.x, sm);
    } else {
        unsigned first = blockIdx.x;
        for (int r = 0; r < req[0]; ++r) {
            const unsigned q0=req[1+4*r], qlen=req[2+4*r];
            const unsigned slot=req[3+4*r], kvlen=req[4+4*r];
            const unsigned work = ((qlen + BQ - 1) / BQ) * nh;
            const size_t qoff = (size_t)q0 * nh * HD;
            const size_t kvoff = (size_t)slot * nkv * kv_stride * HD;
            if (first < work) {
                d_flash_prefill<HD,BQ,BKV>(partials+qoff, stats+(size_t)q0*nh*2,
                    Q+qoff, K+kvoff, V+kvoff,
                    O+qoff, qlen, kvlen, nh, nkv, kvlen-qlen, win, 1,
                    kv_stride, 0xFFFFFFFFu, scale, first, gridDim.x, sm);
            }
            first = next_slice(first, work, gridDim.x);
            __syncthreads();
        }
        const unsigned real = req[1+4*(req[0]-1)] + req[2+4*(req[0]-1)];
        for (size_t i = (size_t)real*nh*HD + (size_t)blockIdx.x*blockDim.x + threadIdx.x;
             i < (size_t)rows*nh*HD; i += (size_t)gridDim.x*blockDim.x)
            O[i] = __float2bfloat16(0.0f);
    }
}

struct Shape { int hd, nh, nkv, window, rows, kvlen; };

static std::vector<int> requests(const Shape& s, int count, bool ragged) {
    const int real = s.rows - (ragged && s.rows > count ? std::min(7, s.rows-count) : 0);
    int left = real, row = 0;
    std::vector<int> req{count};
    for (int r = 0; r < count; ++r) {
        const int remaining = count-r;
        int take = left/remaining;
        if (ragged && remaining > 1) take = r%2 ? std::min(left-remaining+1, take*2) : 1;
        const int kvlen = ragged ? std::max(take, s.kvlen - 17*r) : s.kvlen;
        req.insert(req.end(), {row, take, (r*5+3)%MAX_REQUESTS, kvlen});
        left -= take;
        row += take;
    }
    return req;
}

static void self_test() {
    unsigned cases = 0;
    for (unsigned grid : {1u, 2u, 7u, 132u, 188u})
    for (unsigned bq : {32u, 64u})
    for (unsigned heads : {16u, 32u, 64u})
    for (int count : {1, 2, 4, 8, 16})
    for (bool ragged : {false, true}) {
        const Shape s{256, (int)heads, 1, 0, 1025, 32768};
        const auto req = requests(s, count, ragged);
        std::vector<unsigned> work;
        unsigned total = 0;
        for (int r = 0; r < count; ++r) {
            const unsigned n = (req[2+4*r]+bq-1)/bq*heads;
            work.push_back(n);
            total += n;
        }
        // Empty spans test the mapping even though admission omits them.
        work.insert(work.begin()+work.size()/2, 0);
        std::vector<unsigned> visits(total);
        for (unsigned cta = 0; cta < grid; ++cta) {
            unsigned first = cta, base = 0;
            for (unsigned n : work) {
                for (unsigned w = first; w < n; w += grid) {
                    if ((base+w)%grid != cta || ++visits[base+w] != 1) std::abort();
                }
                first = next_slice(first, n, grid);
                base += n;
            }
        }
        if (std::any_of(visits.begin(), visits.end(), [](unsigned n) { return n != 1; }))
            std::abort();
        ++cases;
    }
    printf("self-test PASS: %u tile-coverage cases; no CUDA calls\n", cases);
}

template <int HD, int BQ, int BKV>
static void bench(const Shape& s) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    const int grid = prop.multiProcessorCount;
    const unsigned stride = (s.kvlen+255)/256*256;
    const size_t smem = (size_t)FA_PRE_SMEM_FLOATS(HD,BQ,BKV)*sizeof(float);
    const size_t elements = (size_t)s.rows*s.nh*HD;
    const size_t kv_elements = (size_t)MAX_REQUESTS*s.nkv*stride*HD;
    const size_t eviction_bytes = 256ull << 20;
    const float scale = 1.0f/std::sqrt((float)HD);
    CK(cudaFuncSetAttribute(k_attention<HD,BQ,BKV,false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CK(cudaFuncSetAttribute(k_attention<HD,BQ,BKV,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    bf16 *q, *k, *v, *out;
    float *partials, *stats;
    unsigned* trash;
    int* dreq;
    CK(cudaMalloc(&q, elements*sizeof(bf16))); CK(cudaMalloc(&out, elements*sizeof(bf16)));
    CK(cudaMalloc(&k, kv_elements*sizeof(bf16))); CK(cudaMalloc(&v, kv_elements*sizeof(bf16)));
    // Stock mux offsets these bases even with NS1's fused output.
    CK(cudaMalloc(&partials, elements*sizeof(float)));
    CK(cudaMalloc(&stats, (size_t)s.rows*s.nh*2*sizeof(float)));
    CK(cudaMalloc(&trash, eviction_bytes)); CK(cudaMemset(trash, 0, eviction_bytes));
    CK(cudaMalloc(&dreq, (1+4*MAX_REQUESTS)*sizeof(int)));
    cudaEvent_t begin, end; CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
    std::vector<bf16> reference(elements), result(elements);
    printf("device=%s sm=%d.%d grid=%d threads=256 smem=%zu hd=%d heads=%d kv_heads=%d window=%d rows=%d kvlen=%d evict=%zu ns=1 kv_maps=none pipe=%d px4=%d tma=%d\n",
        prop.name, prop.major, prop.minor, grid, smem, HD, s.nh, s.nkv, s.window,
        s.rows, s.kvlen, eviction_bytes, PLOW_NV_FA_PIPE, PLOW_NV_FA_PX4, PLOW_NV_FA_TMA);
    for (int seed : {17, 911}) {
        initialize<<<grid,256>>>(q, elements, seed);
        initialize<<<grid,256>>>(k, kv_elements, seed+123);
        initialize<<<grid,256>>>(v, kv_elements, seed+987);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        for (int count : {1, 2, 4, 8, 16}) for (bool ragged : {false, true}) {
            const auto req = requests(s, count, ragged);
            CK(cudaMemcpy(dreq, req.data(), req.size()*sizeof(int), cudaMemcpyHostToDevice));
            auto launch = [&](int arm) {
                if (arm) k_attention<HD,BQ,BKV,true><<<grid,256,smem>>>(dreq,out,q,k,v,partials,stats,s.rows,s.nh,s.nkv,s.window,stride,scale);
                else k_attention<HD,BQ,BKV,false><<<grid,256,smem>>>(dreq,out,q,k,v,partials,stats,s.rows,s.nh,s.nkv,s.window,stride,scale);
                CK(cudaGetLastError());
            };
            CK(cudaMemset(out, 0xff, elements*sizeof(bf16)));
            launch(0);
            CK(cudaMemcpy(reference.data(),out,elements*sizeof(bf16),cudaMemcpyDeviceToHost));
            CK(cudaMemset(out, 0xff, elements*sizeof(bf16)));
            launch(1);
            CK(cudaMemcpy(result.data(),out,elements*sizeof(bf16),cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < elements; ++i) {
                if (!std::isfinite(__bfloat162float(result[i])) ||
                    std::memcmp(&reference[i], &result[i], sizeof(bf16))) {
                    fprintf(stderr,"FAIL seed=%d R=%d ragged=%d element=%zu stock=%g rotated=%g\n",
                        seed,count,ragged,i,__bfloat162float(reference[i]),__bfloat162float(result[i]));
                    std::exit(3);
                }
            }
            for (int repeat = 0; repeat < 2; ++repeat) {
                std::vector<float> samples[2];
                for (int iteration = -3; iteration < 20; ++iteration) {
                    for (int j = 0; j < 2; ++j) {
                        const int arm = ((iteration+3+repeat)&1)^j;
                        evict<<<grid,256>>>(trash,eviction_bytes/sizeof(unsigned));
                        CK(cudaGetLastError());
                        CK(cudaEventRecord(begin)); launch(arm); CK(cudaEventRecord(end));
                        CK(cudaEventSynchronize(end));
                        float ms; CK(cudaEventElapsedTime(&ms,begin,end));
                        if (iteration >= 0) {
                            samples[arm].push_back(ms*1000.0f);
                            printf("sample seed=%d R=%d ragged=%d repeat=%d iteration=%d arm=%d us=%.3f\n",
                                seed,count,ragged,repeat,iteration,arm,ms*1000.0f);
                        }
                    }
                }
                for (auto& values : samples) std::sort(values.begin(), values.end());
                const float stock = (samples[0][9]+samples[0][10])*0.5f;
                const float rotated = (samples[1][9]+samples[1][10])*0.5f;
                printf("summary seed=%d R=%d ragged=%d repeat=%d exact=PASS stock_us=%.3f rotated_us=%.3f speedup=%.4f\n",
                    seed,count,ragged,repeat,stock,rotated,stock/rotated);
            }
        }
    }
    CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end)); CK(cudaFree(dreq));
    CK(cudaFree(q)); CK(cudaFree(k)); CK(cudaFree(v)); CK(cudaFree(out)); CK(cudaFree(trash));
    CK(cudaFree(partials)); CK(cudaFree(stats));
}

int main(int argc, char** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "--self-test")) { self_test(); return 0; }
    const bool plan = argc > 1 && !std::strcmp(argv[1], "--plan");
    if (argc != 7+(int)plan) {
        fprintf(stderr,"usage: %s --self-test | [--plan] hd heads kv_heads window rows kvlen\n",argv[0]);
        return 2;
    }
    int values[6];
    for (int i = 0; i < 6; ++i) {
        char* end; errno = 0;
        const char* arg = argv[i+1+(int)plan];
        const long n = std::strtol(arg,&end,10);
        if (errno || end == arg || *end || n < 0 || n > 1048576) return 2;
        values[i] = (int)n;
    }
    const Shape s{values[0],values[1],values[2],values[3],values[4],values[5]};
    if ((s.hd != 256 && s.hd != 512) || s.nh == 0 || s.nh > 256 || s.nkv == 0 ||
        s.nh%s.nkv || s.rows < MAX_REQUESTS || s.rows > s.kvlen) return 2;
    if (plan) {
        for (int count : {1,2,4,8,16}) for (bool ragged : {false,true}) {
            const auto req = requests(s,count,ragged);
            for (int r = 0; r < count; ++r)
                printf("R=%d ragged=%d request=%d q0=%d qlen=%d slot=%d kvlen=%d\n",
                    count,ragged,r,req[1+4*r],req[2+4*r],req[3+4*r],req[4+4*r]);
        }
        return 0;
    }
    if (s.hd == 256) bench<256,64,32>(s);
    else bench<512,32,16>(s);
    return 0;
}

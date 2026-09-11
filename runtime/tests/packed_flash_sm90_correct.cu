#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
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
#define CD(call) do { CUresult e = (call); if (e != CUDA_SUCCESS) { \
    const char* text; cuGetErrorString(e, &text); \
    std::fprintf(stderr, "%s: %s\n", #call, text); std::exit(2); } } while (0)

static const char* interpreter_path = nullptr;
static const char* snapshot_path = nullptr;
static bool lean_hd512 = false;
static bool lean_hd256_bkv64 = false;
static float attention_scale = 0.0f;
static unsigned test_kv_length = 16384;
static unsigned test_kv_length_min = 0;
static bool campaign_timing = false;
static bool json_output = false;
static bool mapped_only = false;
static unsigned test_seed = 0;
static unsigned test_requests = 1;

#ifndef PLOW_TEST_FA_ROWS
#define PLOW_TEST_FA_ROWS 0
#endif
#ifndef PLOW_TEST_FA_HD256_ONLY
#define PLOW_TEST_FA_HD256_ONLY 0
#endif
static_assert(PLOW_TEST_FA_ROWS >= 0 && PLOW_TEST_FA_ROWS <= 16384);
constexpr unsigned heads = 16, capacity = PLOW_TEST_FA_ROWS ? PLOW_TEST_FA_ROWS : 128;
static unsigned test_rows = PLOW_TEST_FA_ROWS ? PLOW_TEST_FA_ROWS : 98;
static unsigned blocks = 132;

template<int HD, int BKV>
__global__ void run_attention(const bf16* q, const bf16* k, const bf16* v,
                             bf16* out, float* partial, float* stats, const int* req,
                             unsigned rows, unsigned kv_heads, unsigned stride, unsigned mask,
                             unsigned window, const void* maps, unsigned nblk, float scale) {
    extern __shared__ float arena[];
    d_flash_prefill_mux<HD,64,BKV>(req, partial, stats, q, k, v, out,
        rows, 16384, heads, kv_heads, 0, window, 1, stride, mask,
        scale, blockIdx.x, nblk, arena, maps);
}

__global__ void evict_attention_cache(unsigned* data, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        data[i] += 1;
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
                                          unsigned mask, unsigned window, bool tma, bool profile) {
    const float scale = attention_scale > 0.0f ? attention_scale : 1.0f / std::sqrt(float(HD));
    const auto q = values(size_t(capacity) * heads * HD, 123u ^ (test_seed * 0x9e3779b9u));
    const unsigned slots = PLOW_TEST_FA_ROWS ? test_requests : 3;
    const auto k = values(size_t(slots) * kv_heads * stride * HD,
                          456u ^ (test_seed * 0x85ebca6bu));
    const auto v = values(k.size(), 789u ^ (test_seed * 0xc2b2ae35u));
    // Two ragged requests use reversed, noncontiguous slots; the last 30 rows are padding.
    std::vector<int> req = PLOW_TEST_FA_ROWS
        ? std::vector<int>(1 + 4 * test_requests)
        : std::vector<int>{2, 0, 65, 2, 97, 65, 33, 0, 16384};
    if (PLOW_TEST_FA_ROWS) {
        req[0] = int(test_requests);
        unsigned row0 = 0;
        for (unsigned r = 0; r < test_requests; ++r) {
            const unsigned qlen = test_rows / test_requests + (r < test_rows % test_requests);
            const unsigned kvlen = test_requests == 1 ? test_kv_length
                : test_kv_length_min + uint64_t(test_kv_length - test_kv_length_min) * r /
                    (test_requests - 1);
            req[1 + 4 * r] = int(row0);
            req[2 + 4 * r] = int(qlen);
            req[3 + 4 * r] = int(r);
            req[4 + 4 * r] = int(kvlen);
            row0 += qlen;
        }
    }
    bf16* dq = upload(q), *dk = upload(k), *dv = upload(v), *out;
    unsigned* trash = nullptr;
    constexpr size_t eviction_bytes = 256ull << 20;
    if (campaign_timing) {
        CK(cudaMalloc(&trash, eviction_bytes));
        CK(cudaMemset(trash, 0, eviction_bytes));
    }
    int* dr = upload(req);
    std::vector<CUtensorMap> maps(size_t(slots) * 2);
    CUtensorMap* dm = nullptr;
    uint64_t* table = nullptr;
    if (tma) {
        for (unsigned slot = 0; slot < slots; ++slot) for (unsigned operand = 0; operand < 2; ++operand) {
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
        std::vector<uint64_t> map_table(slots);
        for (unsigned slot = 0; slot < slots; ++slot)
            map_table[slot] = uint64_t(dm + 2 * slot);
        table = upload(map_table);
    }
    float *partial, *stats;
    CK(cudaMalloc(&out, q.size() * sizeof(bf16)));
    CK(cudaMalloc(&partial, q.size() * sizeof(float)));
    CK(cudaMalloc(&stats, size_t(capacity) * heads * 2 * sizeof(float)));
    CK(cudaMemset(out, 0xff, q.size() * sizeof(bf16)));
    unsigned smem = FA_PRE_SMEM_FLOATS(HD,64,BKV) * sizeof(float);
#if PLOW_NV_FA_GQA2_PAIR
    if constexpr (HD == 256 && BKV == 32) {
        constexpr unsigned pair_smem = FA_SM90_GQA2_PAIR_FLOATS(HD,64,BKV) * sizeof(float);
        static_assert(pair_smem == 141312);
        smem = pair_smem;
    }
#elif PLOW_NV_FA_WGITEM
    if constexpr (HD == 256 && BKV == 32)
        smem = FA_SM90_WGI_FLOATS(HD,64,BKV) * sizeof(float);
#endif
    CUmodule module = nullptr;
    CUfunction interpreter = nullptr;
    PlowProgram packet{};
    std::vector<void*> packet_allocations;
    if (interpreter_path) {
        CD(cuModuleLoad(&module, interpreter_path));
        const char* entry = lean_hd256_bkv64 ? "plow_sm90a_pfattn_hd256_bkv64" :
            lean_hd512 ? "plow_sm90a_pfattn_hd512" : "_Z23interp_sm90a_pfpackedfa11PlowProgram";
        CD(cuModuleGetFunction(&interpreter, module, entry));
        CUdeviceptr arena_symbol;
        size_t arena_size;
        const char* arena_name = lean_hd256_bkv64 ? "plow_arena_bytes_pfattn_hd256_bkv64" :
            lean_hd512 ? "plow_arena_bytes_pfattn_hd512" : "plow_arena_bytes_pfpackedfa";
        CD(cuModuleGetGlobal(&arena_symbol, &arena_size, module, arena_name));
        if (arena_size != sizeof(smem)) {
            std::fprintf(stderr, "invalid interpreter arena metadata\n");
            std::exit(2);
        }
        CD(cuMemcpyDtoH(&smem, arena_symbol, sizeof(smem)));
        CD(cuFuncSetAttribute(interpreter, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
        auto put = [&](const auto& host) {
            auto* device = upload(host);
            packet_allocations.push_back(device);
            return device;
        };
        PlowDevInst inst{};
        inst.op = PLOW_DOP_FLASH_PREFILL;
        inst.blocks = blocks;
        for (unsigned i = 0; i < 8; ++i) inst.t[i] = i;
        if (!tma) inst.t[7] = PLOW_TENSOR_NONE;
        inst.i[0] = test_rows; inst.i[1] = 16384;
        inst.i[2] = heads; inst.i[3] = kv_heads;
        inst.i[5] = window; inst.i[6] = HD; inst.i[7] = 1;
        inst.fj[0].f = scale;
        inst.fj[1].u = stride; inst.fj[2].u = mask;
        std::vector<PlowStreamEnt> entries(blocks);
        for (unsigned i = 0; i < blocks; ++i) {
            entries[i].slice = i;
            entries[i].wait_len = 1;
            entries[i].succ_len = 1;
        }
        packet.insts = put(std::vector<PlowDevInst>{inst});
        packet.gq_stream = put(entries);
        packet.gq_seg_ofs = put(std::vector<uint32_t>{0, blocks});
        packet.gq_cursor = put(std::vector<uint32_t>(PLOW_CTR_STRIDE));
        packet.waits = put(std::vector<PlowWait>{{0, blocks}});
        packet.succs = put(std::vector<uint32_t>{1});
        std::vector<uint32_t> counters(2 * PLOW_CTR_STRIDE);
        counters[0] = blocks;
        packet.counters = put(counters);
        packet.tensors = put(std::vector<void*>{partial, stats, dq, dk, dv, out, dr, table});
    }
    CK(cudaFuncSetAttribute(run_attention<HD,BKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    auto reset_packet = [&] {
        if (interpreter) {
            CK(cudaMemsetAsync(packet.gq_cursor, 0, PLOW_CTR_STRIDE * sizeof(uint32_t)));
            CK(cudaMemsetAsync(PLOW_CTR(packet.counters, 1), 0, sizeof(uint32_t)));
        }
    };
    auto launch = [&] {
        if (interpreter) {
            void* args[]{&packet};
            CD(cuLaunchKernel(interpreter, blocks, 1, 1, 256, 1, 1, smem, nullptr, args, nullptr));
        } else {
            run_attention<HD,BKV><<<blocks,256,smem>>>(dq,dk,dv,out,partial,stats,dr,test_rows,
                kv_heads,stride,mask,window,table,blocks,scale);
        }
    };
    reset_packet();
    launch();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<bf16> got(q.size());
    CK(cudaMemcpy(got.data(), out, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    if (snapshot_path) {
        const std::string path = std::string(snapshot_path) + "." + std::to_string(HD) +
                                 "." + std::to_string(tma) + ".bf16";
        FILE* file = std::fopen(path.c_str(), "wb");
        if (!file || std::fwrite(got.data(), sizeof(bf16), got.size(), file) != got.size())
            std::exit(2);
        if (std::fclose(file)) std::exit(2);
    }
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
    std::vector<float> samples;
    const unsigned repeats = interpreter ? 1 : 10;
    const unsigned sample_count = campaign_timing ? 15u : interpreter ? 21u : 7u;
    for (unsigned sample = 0; sample < (profile ? 0u : sample_count); ++sample) {
        if (campaign_timing) std::this_thread::sleep_for(std::chrono::milliseconds(25));
        if (campaign_timing)
            evict_attention_cache<<<blocks,256>>>(trash, eviction_bytes / sizeof(unsigned));
        reset_packet();
        CK(cudaEventRecord(start));
        for (unsigned repeat = 0; repeat < repeats; ++repeat) launch();
        CK(cudaEventRecord(stop)); CK(cudaEventSynchronize(stop));
        float ms;
        CK(cudaEventElapsedTime(&ms,start,stop));
        samples.push_back(ms * 1000.0f / repeats);
    }
    std::sort(samples.begin(),samples.end());
    CK(cudaEventDestroy(start)); CK(cudaEventDestroy(stop));
    bool ok = true;
    if (interpreter) {
        uint32_t completed;
        CK(cudaMemcpy(&completed, PLOW_CTR(packet.counters, 1), sizeof(completed), cudaMemcpyDeviceToHost));
        ok &= completed == blocks;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        const float value = __bfloat162float(got[i]);
        ok &= std::isfinite(value);
        if (i >= size_t(test_rows) * heads * HD) ok &= value == 0.0f;
    }
    double worst = 0, max_error = 0;
    unsigned checked = 0;
    for (unsigned r = 0; r < unsigned(req[0]); ++r) {
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
                score *= scale;
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
    unsigned kv_length_min = UINT32_MAX, kv_length_max = 0;
    for (unsigned r = 0; r < unsigned(req[0]); ++r) {
        kv_length_min = std::min(kv_length_min, unsigned(req[4 + 4 * r]));
        kv_length_max = std::max(kv_length_max, unsigned(req[4 + 4 * r]));
    }
    const char* topology = req[0] == 1 ? "single"
        : kv_length_min == kv_length_max ? "packed_homogeneous" : "packed_ragged";
    if (json_output) {
        std::printf("{\"mode\":\"%s\",\"seed\":%u,\"head_dim\":%d,\"q_heads\":%u,"
                    "\"kv_heads\":%u,\"gqa\":%u,"
                    "\"topology\":\"%s\",\"requests\":%u,\"rows\":%u,\"kv_length_min\":%u,"
                    "\"kv_length_max\":%u,\"window\":%u,\"mapped\":%s,\"checked\":%u,"
                    "\"worst_rel_l2\":%.9g,\"max_abs\":%.9g,\"correct\":%s,"
                    "\"samples_us\":[",
                    interpreter ? "interpreter" : "body", test_seed, HD, heads, kv_heads,
                    heads / kv_heads,
                    topology, unsigned(req[0]), test_rows, kv_length_min, kv_length_max, window,
                    tma ? "true" : "false", checked, worst, max_error, ok ? "true" : "false");
        for (size_t i = 0; i < samples.size(); ++i)
            std::printf("%s%.9g", i ? "," : "", samples[i]);
        std::printf("]}\n");
    } else {
        std::printf("mode=%s seed=%u HD=%d BKV=%d KV=%u requests=%u rows=%u kv_length=%u window=%u maps=%d checked=%u "
                    "worst_relL2=%.6g max_abs=%.6g best_us=%.3f median_us=%.3f %s\n",
                    interpreter ? "interpreter" : "body",test_seed,HD,interpreter ? 0 : BKV,kv_heads,
                    unsigned(req[0]),test_rows,test_kv_length,window,int(tma),checked,worst,max_error,
                    samples.empty() ? NAN : samples.front(),
                    samples.empty() ? NAN : samples[samples.size()/2],ok?"PASS":"FAIL");
    }
    CK(cudaFree(dq)); CK(cudaFree(dk)); CK(cudaFree(dv)); CK(cudaFree(out));
    CK(cudaFree(partial)); CK(cudaFree(stats)); CK(cudaFree(dr));
    if (trash) CK(cudaFree(trash));
    if (tma) { CK(cudaFree(table)); CK(cudaFree(dm)); }
    for (void* allocation : packet_allocations) CK(cudaFree(allocation));
    if (module) CD(cuModuleUnload(module));
    return ok;
}

int main(int argc, char** argv) {
    bool profile = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profile") == 0) profile = true;
        else if (std::strcmp(argv[i], "--campaign-timing") == 0) campaign_timing = true;
        else if (std::strcmp(argv[i], "--json") == 0) json_output = true;
        else if (std::strcmp(argv[i], "--mapped-only") == 0) mapped_only = true;
        else if (std::strcmp(argv[i], "--lean-hd512") == 0) lean_hd512 = true;
        else if (std::strcmp(argv[i], "--lean-hd256-bkv64") == 0) lean_hd256_bkv64 = true;
        else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            char* end;
            const auto value = std::strtoul(argv[++i], &end, 10);
            if (*end || value == 0 || value > 132) return 2;
            blocks = unsigned(value);
        }
        else if (std::strcmp(argv[i], "--interpreter") == 0 && i + 1 < argc) interpreter_path = argv[++i];
        else if (std::strcmp(argv[i], "--snapshot") == 0 && i + 1 < argc) snapshot_path = argv[++i];
        else if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            char* end;
            attention_scale = std::strtof(argv[++i], &end);
            if (*end || !std::isfinite(attention_scale) || attention_scale <= 0.0f) return 2;
        }
        else if (std::strcmp(argv[i], "--kv-length") == 0 && i + 1 < argc) {
            char* end;
            const auto value = std::strtoul(argv[++i], &end, 10);
            if (*end || value == 0 || value > 16384) return 2;
            test_kv_length = unsigned(value);
        }
        else if (std::strcmp(argv[i], "--kv-length-min") == 0 && i + 1 < argc) {
            char* end;
            const auto value = std::strtoul(argv[++i], &end, 10);
            if (*end || value == 0 || value > 16384) return 2;
            test_kv_length_min = unsigned(value);
        }
        else if (std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            char* end;
            const auto value = std::strtoul(argv[++i], &end, 10);
            if (*end || value == 0 || value > capacity) return 2;
            test_rows = unsigned(value);
        }
        else if (std::strcmp(argv[i], "--requests") == 0 && i + 1 < argc) {
            char* end;
            const auto value = std::strtoul(argv[++i], &end, 10);
            if (*end || value == 0 || value > 16) return 2;
            test_requests = unsigned(value);
        }
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            char* end;
            const auto value = std::strtoul(argv[++i], &end, 10);
            if (*end || value > 1000000) return 2;
            test_seed = unsigned(value);
        }
        else return 2;
    }
    if ((lean_hd512 || lean_hd256_bkv64) && !interpreter_path) return 2;
    if (lean_hd512 && lean_hd256_bkv64) return 2;
    if (PLOW_TEST_FA_ROWS && !lean_hd512 && !PLOW_NV_FA512_KV64 &&
        !PLOW_TEST_FA_HD256_ONLY) return 2;
    if (!PLOW_TEST_FA_ROWS && test_kv_length != 16384) return 2;
    if (!PLOW_TEST_FA_ROWS && test_rows != 98) return 2;
    if (!PLOW_TEST_FA_ROWS && test_requests != 1) return 2;
    if (!test_kv_length_min) test_kv_length_min = test_kv_length;
    if ((!PLOW_TEST_FA_ROWS && test_kv_length_min != test_kv_length)
        || test_kv_length_min > test_kv_length
        || (test_requests == 1 && test_kv_length_min != test_kv_length))
        return 2;
    if (test_requests > test_rows || test_kv_length_min < (test_rows + test_requests - 1) / test_requests)
        return 2;
    if (test_requests == 1 && test_kv_length < test_rows) return 2;
    bool ok = true;
    for (bool tma : {false, true}) {
        if (mapped_only && !tma) continue;
        if (lean_hd256_bkv64) {
            if (tma) ok &= check<256,64>(8,2048,2047,1024,true,profile);
            continue;
        }
#if PLOW_TEST_FA_HD256_ONLY
        ok &= check<256,32>(8,2048,2047,1024,tma,profile);
        continue;
#endif
#if PLOW_NV_FA512_KV64 && PLOW_TEST_FA_ROWS
        if (!interpreter_path) {
            ok &= check<512,32>(1,16384,0xffffffffu,0,tma,profile);
            ok &= check<512,64>(1,16384,0xffffffffu,0,tma,profile);
            continue;
        }
#endif
        if (!lean_hd512) ok &= check<256,32>(8,2048,2047,1024,tma,profile);
        if (!interpreter_path) {
            ok &= check<256,64>(8,2048,2047,1024,tma,profile);
            ok &= check<512,16>(1,16384,0xffffffffu,0,tma,profile);
        }
        ok &= check<512,32>(1,16384,0xffffffffu,0,tma,profile);
#if PLOW_NV_FA512_KV64
        if (!interpreter_path) ok &= check<512,64>(1,16384,0xffffffffu,0,tma,profile);
#endif
    }
    return ok ? 0 : 1;
}

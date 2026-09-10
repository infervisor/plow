#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
static bool lean_hd512 = false;

#ifndef PLOW_TEST_FA_ROWS
#define PLOW_TEST_FA_ROWS 0
#endif
static_assert(PLOW_TEST_FA_ROWS == 0 ||
              (PLOW_TEST_FA_ROWS >= 64 && PLOW_TEST_FA_ROWS <= 16384 && PLOW_TEST_FA_ROWS % 64 == 0));
constexpr unsigned heads = 16, capacity = PLOW_TEST_FA_ROWS ? PLOW_TEST_FA_ROWS : 128;
constexpr unsigned real_rows = PLOW_TEST_FA_ROWS ? PLOW_TEST_FA_ROWS : 98;
static unsigned blocks = 132;

template<int HD, int BKV>
__global__ void run_attention(const bf16* q, const bf16* k, const bf16* v,
                             bf16* out, float* partial, float* stats, const int* req,
                             unsigned kv_heads, unsigned stride, unsigned mask,
                             unsigned window, const void* maps, unsigned nblk) {
    extern __shared__ float arena[];
    d_flash_prefill_mux<HD,64,BKV>(req, partial, stats, q, k, v, out,
        capacity, 16384, heads, kv_heads, 0, window, 1, stride, mask,
        1.0f / sqrtf(float(HD)), blockIdx.x, nblk, arena, maps);
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
    const auto q = values(size_t(capacity) * heads * HD, 123);
    const auto k = values(size_t(3) * kv_heads * stride * HD, 456);
    const auto v = values(k.size(), 789);
    // Two ragged requests use reversed, noncontiguous slots; the last 30 rows are padding.
    const std::vector<int> req = PLOW_TEST_FA_ROWS
        ? std::vector<int>{1, 0, int(real_rows), 0, 16384}
        : std::vector<int>{2, 0, 65, 2, 97, 65, 33, 0, 16384};
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
    unsigned smem = FA_PRE_SMEM_FLOATS(HD,64,BKV) * sizeof(float);
    CUmodule module = nullptr;
    CUfunction interpreter = nullptr;
    PlowProgram packet{};
    std::vector<void*> packet_allocations;
    if (interpreter_path) {
        CD(cuModuleLoad(&module, interpreter_path));
        CD(cuModuleGetFunction(&interpreter, module, lean_hd512 ? "plow_sm90a_pfattn_hd512" : "_Z23interp_sm90a_pfpackedfa11PlowProgram"));
        CUdeviceptr arena_symbol;
        size_t arena_size;
        CD(cuModuleGetGlobal(&arena_symbol, &arena_size, module, lean_hd512 ? "plow_arena_bytes_pfattn_hd512" : "plow_arena_bytes_pfpackedfa"));
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
        inst.i[0] = capacity; inst.i[1] = 16384;
        inst.i[2] = heads; inst.i[3] = kv_heads;
        inst.i[5] = window; inst.i[6] = HD; inst.i[7] = 1;
        inst.fj[0].f = 1.0f / sqrtf(float(HD));
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
            run_attention<HD,BKV><<<blocks,256,smem>>>(dq,dk,dv,out,partial,stats,dr,kv_heads,stride,mask,window,table,blocks);
        }
    };
    reset_packet();
    launch();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<bf16> got(q.size());
    CK(cudaMemcpy(got.data(), out, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
    std::vector<float> samples;
    const unsigned repeats = interpreter ? 1 : 10;
    for (unsigned sample = 0; sample < (profile ? 0u : interpreter ? 21u : 7u); ++sample) {
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
        if (i >= size_t(real_rows) * heads * HD) ok &= value == 0.0f;
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
    std::printf("mode=%s HD=%d BKV=%d KV=%u window=%u maps=%d checked=%u worst_relL2=%.6g max_abs=%.6g warm_us=%.3f %s\n",
                interpreter ? "interpreter" : "body",HD,interpreter ? 0 : BKV,kv_heads,window,int(tma),checked,worst,max_error,
                samples.empty() ? NAN : samples[samples.size()/2],ok?"PASS":"FAIL");
    CK(cudaFree(dq)); CK(cudaFree(dk)); CK(cudaFree(dv)); CK(cudaFree(out));
    CK(cudaFree(partial)); CK(cudaFree(stats)); CK(cudaFree(dr));
    if (tma) { CK(cudaFree(table)); CK(cudaFree(dm)); }
    for (void* allocation : packet_allocations) CK(cudaFree(allocation));
    if (module) CD(cuModuleUnload(module));
    return ok;
}

int main(int argc, char** argv) {
    bool profile = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profile") == 0) profile = true;
        else if (std::strcmp(argv[i], "--lean-hd512") == 0) lean_hd512 = true;
        else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            char* end;
            const auto value = std::strtoul(argv[++i], &end, 10);
            if (*end || value == 0 || value > 132) return 2;
            blocks = unsigned(value);
        }
        else if (std::strcmp(argv[i], "--interpreter") == 0 && i + 1 < argc) interpreter_path = argv[++i];
        else return 2;
    }
    if (lean_hd512 && !interpreter_path) return 2;
    if (PLOW_TEST_FA_ROWS && !lean_hd512) return 2;
    bool ok = true;
    for (bool tma : {false, true}) {
        if (!lean_hd512) ok &= check<256,32>(8,2048,2047,1024,tma,profile);
        if (!interpreter_path) {
            ok &= check<256,64>(8,2048,2047,1024,tma,profile);
            ok &= check<512,16>(1,16384,0xffffffffu,0,tma,profile);
        }
        ok &= check<512,32>(1,16384,0xffffffffu,0,tma,profile);
    }
    return ok ? 0 : 1;
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#define PLOW_NV_GEMMA 1
#define PLOW_NV_HOPPER 1
#define GV_MM_MAX 16
#define PLOW_NV_FP8_DECODE_MMA 1
#include "op_gemm.cuh"

#define CHECK(call) do { auto error = (call); if (error != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); exit(1); } } while (0)
using bf16 = __nv_bfloat16;

template <bool Candidate, bool Glu>
__global__ void projection(bf16* C, const bf16* x, const uint8_t* W, const uint8_t* Wu,
    const float* scale, const float* su, unsigned M, unsigned N, unsigned K,
    unsigned act, unsigned first_slice, unsigned nblk) {
    const unsigned slice = first_slice + blockIdx.x;
    if constexpr (Candidate) {
        if constexpr (Glu) d_gemv_glu_fp8(C, x, W, Wu, scale, su, M, N, K, act, slice, nblk);
        else d_gemv_fp8(C, x, W, scale, M, N, K, slice, nblk);
    } else {
        gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
            if constexpr (Glu) gemv_glu_rows_fp8<decltype(mm)::v>(C + (size_t)m0 * N,
                x + (size_t)m0 * K, W, Wu, scale, su, rows, N, K, act, slice, nblk);
            else gemv_rows_fp8<decltype(mm)::v>(C + (size_t)m0 * N,
                x + (size_t)m0 * K, W, scale, rows, N, K, slice, nblk);
        });
    }
}

__global__ void flush_fp8_persistent(unsigned* memory, unsigned count) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) memory[i] += i;
}

template <class T> T* allocate(size_t count) {
    T* result; CHECK(cudaMalloc(&result, count * sizeof(T))); return result;
}

static void run(unsigned M, unsigned N, unsigned K, unsigned blocks,
    bool exact, bool ownership, unsigned* flush) {
    std::mt19937 random(123);
    std::uniform_real_distribution<float> uniform(-1.f, 1.f);
    std::vector<bf16> x((size_t)M * K), reference((size_t)M * N), result(reference.size());
    std::vector<uint8_t> weights((size_t)N * K), up(weights.size());
    std::vector<float> scales(N), su(N);
    for (auto& value : x) value = __float2bfloat16(exact ? (int)(random() % 7) - 3 : uniform(random));
    for (auto* values : {&weights, &up}) for (auto& value : *values) {
        __nv_fp8_e4m3 encoded(exact ? (int)(random() % 7) - 3.f : uniform(random));
        value = *reinterpret_cast<uint8_t*>(&encoded);
    }
    for (unsigned n = 0; n < N; ++n) {
        scales[n] = exact ? .125f : .01f + .03f * std::abs(uniform(random));
        su[n] = exact ? .25f : .01f + .03f * std::abs(uniform(random));
    }
    auto* dx = allocate<bf16>(x.size()); auto* dw = allocate<uint8_t>(weights.size());
    auto* du = allocate<uint8_t>(up.size()); auto* ds = allocate<float>(N); auto* dsu = allocate<float>(N);
    auto* storage = allocate<bf16>(result.size() + 128); auto* dy = storage + 64;
    CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(du, up.data(), up.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ds, scales.data(), N * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dsu, su.data(), N * 4, cudaMemcpyHostToDevice));
    auto launch = [&](bool candidate, unsigned mode, unsigned first, unsigned count) {
        const unsigned act = mode == 1 ? PLOW_ACT_SILU_ : PLOW_ACT_GELU_TANH_;
        if (candidate && mode) projection<true, true><<<count, 256>>>(dy, dx, dw, du, ds, dsu, M, N, K, act, first, blocks);
        else if (candidate) projection<true, false><<<count, 256>>>(dy, dx, dw, du, ds, dsu, M, N, K, act, first, blocks);
        else if (mode) projection<false, true><<<count, 256>>>(dy, dx, dw, du, ds, dsu, M, N, K, act, first, blocks);
        else projection<false, false><<<count, 256>>>(dy, dx, dw, du, ds, dsu, M, N, K, act, first, blocks);
        CHECK(cudaGetLastError());
    };
    const unsigned per = (N + blocks - 1) / blocks;
    cudaEvent_t begin, end; CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
    for (unsigned mode = 0; mode < 3; ++mode) {
        launch(false, mode, 0, blocks);
        CHECK(cudaMemcpy(reference.data(), dy, reference.size() * 2, cudaMemcpyDeviceToHost));
        const std::vector<unsigned> slices = ownership ? std::vector<unsigned>{0, blocks / 2, blocks - 1} : std::vector<unsigned>{0};
        for (unsigned slice : slices) {
            CHECK(cudaMemset(storage, 0x5a, (result.size() + 128) * 2));
            launch(true, mode, slice, ownership ? 1 : blocks);
            CHECK(cudaMemcpy(result.data(), dy, result.size() * 2, cudaMemcpyDeviceToHost));
            std::vector<bf16> guards(128);
            CHECK(cudaMemcpy(guards.data(), storage, 128, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(guards.data() + 64, dy + result.size(), 128, cudaMemcpyDeviceToHost));
            for (auto guard : guards) if (*reinterpret_cast<uint16_t*>(&guard) != 0x5a5a) { fprintf(stderr, "guard overwritten\n"); exit(2); }
            unsigned changed = 0; double error2 = 0, norm2 = 0, max_abs = 0;
            for (size_t i = 0; i < result.size(); ++i) {
                const unsigned col = i % N;
                const bool owned = !ownership || (col >= slice * per && col < std::min((slice + 1) * per, N));
                if (!owned) {
                    if (*reinterpret_cast<uint16_t*>(&result[i]) != 0x5a5a) { fprintf(stderr, "slice ownership violated\n"); exit(2); }
                    continue;
                }
                const float got = __bfloat162float(result[i]), ref = __bfloat162float(reference[i]);
                if (!std::isfinite(got)) { fprintf(stderr, "nonfinite output\n"); exit(2); }
                const double diff = got - ref;
                changed += diff != 0; error2 += diff * diff; norm2 += (double)ref * ref;
                max_abs = std::max(max_abs, std::abs(diff));
            }
            const double relative = std::sqrt(error2 / (norm2 + 1e-30));
            if ((exact && changed) || relative > .001) { fprintf(stderr, "numeric failure M%u N%u K%u mode%u changed%u rel%g\n", M,N,K,mode,changed,relative); exit(2); }
            printf("check M=%u N=%u K=%u blocks=%u mode=%u slice=%u ownership=%u changed=%u relL2=%.9g max_abs=%.9g PASS\n",M,N,K,blocks,mode,slice,ownership,changed,relative,max_abs);
        }
        if (ownership) continue;
        for (bool candidate : {false, true}) {
            for (unsigned warm = 0; warm < 4; ++warm) launch(candidate, mode, 0, blocks);
            CHECK(cudaDeviceSynchronize()); std::vector<float> times;
            for (unsigned rep = 0; rep < (exact ? 1u : 15u); ++rep) {
                flush_fp8_persistent<<<528, 256>>>(flush, 64 * 1024 * 1024);
                CHECK(cudaEventRecord(begin)); launch(candidate, mode, 0, blocks); CHECK(cudaEventRecord(end));
                CHECK(cudaEventSynchronize(end)); float ms; CHECK(cudaEventElapsedTime(&ms, begin, end)); times.push_back(ms * 1000);
            }
            std::sort(times.begin(), times.end());
            printf("time M=%u N=%u K=%u blocks=%u mode=%u candidate=%u median_us=%.3f min_us=%.3f max_us=%.3f\n",M,N,K,blocks,mode,candidate,times[times.size()/2],times.front(),times.back());
            fflush(stdout);
        }
    }
    CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end));
    CHECK(cudaFree(dx)); CHECK(cudaFree(dw)); CHECK(cudaFree(du)); CHECK(cudaFree(ds)); CHECK(cudaFree(dsu)); CHECK(cudaFree(storage));
}

int main(int argc, char** argv) {
    if (argc != 1 && (argc != 5 || std::atoi(argv[1]) < 1 || std::atoi(argv[2]) < 1 ||
        std::atoi(argv[3]) < 8 || std::atoi(argv[3]) % 8 || std::atoi(argv[4]) < 1)) {
        fprintf(stderr, "usage: %s [M>0 N>0 K=multiple-of-8 blocks>0]\n", argv[0]); return 1;
    }
    auto* flush = allocate<unsigned>(64 * 1024 * 1024); CHECK(cudaMemset(flush, 0, 256 * 1024 * 1024));
    for (unsigned batch : {1u, 4u, 8u, 17u, 32u}) run(batch, 71, 256, 13, true, true, flush);
    run(17, 71, 256, 132, true, true, flush);
    run(8, 71, 264, 13, true, true, flush);
    if (argc == 5) run(std::atoi(argv[1]),std::atoi(argv[2]),std::atoi(argv[3]),std::atoi(argv[4]),false,false,flush);
    else if (argc == 1) for (unsigned batch : {8u,16u,32u}) {
        run(batch,8192,5376,132,false,false,flush);
        run(batch,21504,5376,132,false,false,flush);
        run(batch,5376,21504,132,false,false,flush);
    }
    CHECK(cudaFree(flush));
}
